#!/usr/bin/env python3

"""Validate and materialize Mandrel's resolved Vortex configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from typing import Any, Mapping


MANIFEST_SCHEMA = "mandrel.hardware.vortex-config-manifest.v2"
DEFAULT_REALIZATION_PROFILE = "verilator_rtlsim"
REQUIRED_XLEN = 64
DEFAULT_GENERATOR_CFLAGS = f"-DSIMULATION -DSV_DPI -DVX_CFG_XLEN={REQUIRED_XLEN}"
DEFINE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DECIMAL_RE = re.compile(r"^[+-]?[0-9]+$")


class ConfigError(RuntimeError):
    """Raised when an input or generated configuration is invalid."""


@dataclass(frozen=True)
class DesignConfig:
    schema: str
    design_id: str
    requested: dict[str, int | bool | str]
    upstream_keys: dict[str, str]


@dataclass(frozen=True)
class ResolvedConfig:
    defines: dict[str, int | bool | str]
    selected: dict[str, int | bool | str]
    sha256: str
    tag: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_design_config(path: Path) -> DesignConfig:
    try:
        with path.open("rb") as source:
            document = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read design config {path}: {exc}") from exc

    schema = document.get("schema")
    design_id = document.get("id")
    requested_table = document.get("vortex")
    upstream_table = document.get("upstream_keys")
    if not isinstance(schema, str) or not schema:
        raise ConfigError("design config requires a non-empty string 'schema'")
    if not isinstance(design_id, str) or not design_id:
        raise ConfigError("design config requires a non-empty string 'id'")
    if not isinstance(requested_table, dict):
        raise ConfigError("design config requires a [vortex] table")
    if not isinstance(upstream_table, dict):
        raise ConfigError("design config requires an [upstream_keys] table")

    requested: dict[str, int | bool | str] = {}
    for field, value in requested_table.items():
        if not isinstance(field, str) or not isinstance(value, (bool, int, str)):
            raise ConfigError(f"unsupported requested value vortex.{field}: {value!r}")
        requested[field] = value

    xlen = requested.get("xlen")
    if isinstance(xlen, bool) or not isinstance(xlen, int):
        raise ConfigError("vortex.xlen must be an integer")
    if xlen != REQUIRED_XLEN:
        raise ConfigError(
            f"vortex.xlen must be {REQUIRED_XLEN}; the pinned Vortex invocation resolves XLEN={REQUIRED_XLEN}"
        )

    tracked_fields = set(requested) - {"xlen"}
    upstream_fields = set(upstream_table)
    missing = sorted(tracked_fields - upstream_fields)
    extra = sorted(upstream_fields - tracked_fields)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("untracked requested fields: " + ", ".join(missing))
        if extra:
            details.append("upstream keys without requested fields: " + ", ".join(extra))
        raise ConfigError("[upstream_keys] must exactly track [vortex] except xlen (" + "; ".join(details) + ")")

    upstream_keys: dict[str, str] = {}
    seen_defines: set[str] = set()
    for field, define_name in upstream_table.items():
        if not isinstance(define_name, str) or DEFINE_NAME_RE.fullmatch(define_name) is None:
            raise ConfigError(f"invalid upstream define for {field}: {define_name!r}")
        if define_name in seen_defines:
            raise ConfigError(f"upstream define is tracked more than once: {define_name}")
        seen_defines.add(define_name)
        upstream_keys[field] = define_name

    return DesignConfig(
        schema=schema,
        design_id=design_id,
        requested=requested,
        upstream_keys=upstream_keys,
    )


def parse_define_value(raw_value: str | None) -> int | bool | str:
    if raw_value is None:
        return True
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(raw_value, 0)
    except ValueError:
        if DECIMAL_RE.fullmatch(raw_value):
            return int(raw_value, 10)
        return raw_value


def parse_resolved_defines(cflags: str) -> dict[str, int | bool | str]:
    """Parse gen_config.py's resolved cflags with shell-compatible tokenization."""
    try:
        tokens = shlex.split(cflags, posix=True)
    except ValueError as exc:
        raise ConfigError(f"cannot parse resolved cflags: {exc}") from exc

    defines: dict[str, int | bool | str] = {}
    for token in tokens:
        if not token.startswith("-D") or token == "-D":
            raise ConfigError(f"unexpected token in resolved cflags: {token!r}")
        assignment = token[2:]
        if "=" in assignment:
            name, raw_value = assignment.split("=", 1)
        else:
            name, raw_value = assignment, None
        if DEFINE_NAME_RE.fullmatch(name) is None:
            raise ConfigError(f"invalid resolved define name: {name!r}")
        value = parse_define_value(raw_value)
        previous = defines.get(name)
        if name in defines and previous != value:
            raise ConfigError(f"conflicting resolved values for {name}: {previous!r} and {value!r}")
        defines[name] = value

    if not defines:
        raise ConfigError("Vortex config generator returned no -D defines")
    return dict(sorted(defines.items()))


def _boolean_define_value(name: str, value: int | bool | str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ConfigError(f"boolean define {name} has non-boolean value {value!r}")


def select_resolved_values(
    requested: Mapping[str, int | bool | str],
    upstream_keys: Mapping[str, str],
    defines: Mapping[str, int | bool | str],
) -> dict[str, int | bool | str]:
    xlen = defines.get("VX_CFG_XLEN")
    if xlen is None:
        raise ConfigError("resolved defines are missing VX_CFG_XLEN")

    selected: dict[str, int | bool | str] = {"xlen": xlen}
    for field, define_name in upstream_keys.items():
        requested_value = requested[field]
        if isinstance(requested_value, bool):
            enabled_name = define_name + "D" if define_name.endswith("_ENABLE") else ""
            if enabled_name and enabled_name in defines:
                selected[field] = _boolean_define_value(enabled_name, defines[enabled_name])
            elif define_name in defines:
                selected[field] = _boolean_define_value(define_name, defines[define_name])
            else:
                selected[field] = False
        else:
            if define_name not in defines:
                raise ConfigError(f"resolved defines are missing {define_name} for vortex.{field}")
            selected[field] = defines[define_name]
    return dict(sorted(selected.items()))


def validate_requested_values(
    requested: Mapping[str, int | bool | str],
    selected: Mapping[str, int | bool | str],
) -> None:
    mismatches = [
        f"vortex.{field}: requested {requested[field]!r}, resolved {selected.get(field)!r}"
        for field in sorted(requested)
        if field not in selected
        or type(requested[field]) is not type(selected[field])
        or requested[field] != selected[field]
    ]
    if mismatches:
        raise ConfigError("requested Vortex config does not match resolved defines:\n  " + "\n  ".join(mismatches))


def canonical_resolved_json(
    defines: Mapping[str, int | bool | str], xlen: int
) -> bytes:
    payload = {"defines": dict(defines), "xlen": xlen}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def resolve_config(design: DesignConfig, cflags: str) -> ResolvedConfig:
    defines = parse_resolved_defines(cflags)
    selected = select_resolved_values(design.requested, design.upstream_keys, defines)
    validate_requested_values(design.requested, selected)
    canonical = canonical_resolved_json(defines, REQUIRED_XLEN)
    resolved_sha256 = hashlib.sha256(canonical).hexdigest()
    return ResolvedConfig(
        defines=defines,
        selected=selected,
        sha256=resolved_sha256,
        tag=resolved_sha256[:16],
    )


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_manifest(
    root: Path,
    design_path: Path,
    source_path: Path,
    design: DesignConfig,
    resolved: ResolvedConfig,
    realization_profile: str,
    generator_cflags: str,
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "design_schema": design.schema,
        "design_id": design.design_id,
        "design_config": {
            "path": _display_path(design_path, root),
            "sha256": sha256_file(design_path),
        },
        "source_config": {
            "path": _display_path(source_path, root),
            "sha256": sha256_file(source_path),
        },
        "xlen": REQUIRED_XLEN,
        "resolution": {
            "profile": realization_profile,
            "generator_cflags": shlex.split(generator_cflags, posix=True),
        },
        "resolved_sha256": resolved.sha256,
        "config_tag": resolved.tag,
        "selected": {
            "upstream_keys": dict(sorted(design.upstream_keys.items())),
            "requested": dict(sorted(design.requested.items())),
            "resolved": resolved.selected,
        },
        "defines": resolved.defines,
    }


def render_outputs(build_dir: Path, manifest: Mapping[str, Any], resolved: ResolvedConfig) -> dict[Path, bytes]:
    tag = resolved.tag
    verilog_header = (
        "`ifndef VX_MANDREL_VH\n"
        "`define VX_MANDREL_VH\n\n"
        f"// Resolved configuration SHA-256: {resolved.sha256}\n"
        f"`define VX_CFG_MANDREL_CONFIG_ID 64'h{tag}\n\n"
        "`endif\n"
    )
    cpp_header = (
        "#ifndef VX_MANDREL_H\n"
        "#define VX_MANDREL_H\n\n"
        "#include <stdint.h>\n\n"
        f"/* Resolved configuration SHA-256: {resolved.sha256} */\n"
        f"#define VX_CFG_MANDREL_CONFIG_ID UINT64_C(0x{tag})\n\n"
        "#endif\n"
    )
    return {
        build_dir / "mandrel" / "vortex-config.json": (
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        ).encode("utf-8"),
        build_dir / "mandrel" / "vortex-config.sha256": f"{resolved.sha256}\n".encode(),
        build_dir / "mandrel" / "vortex-config.tag": f"{tag}\n".encode(),
        build_dir / "hw" / "VX_mandrel.vh": verilog_header.encode(),
        build_dir / "sw" / "VX_mandrel.h": cpp_header.encode(),
    }


def atomic_write_outputs(outputs: Mapping[Path, bytes]) -> None:
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, content in outputs.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{destination.name}.", dir=destination.parent, delete=False
            ) as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            temporary_path.chmod(0o644)
            staged.append((temporary_path, destination))

        for temporary_path, destination in staged:
            if destination.is_file() and destination.read_bytes() == temporary_path.read_bytes():
                temporary_path.unlink()
            else:
                os.replace(temporary_path, destination)
    finally:
        for temporary_path, _ in staged:
            temporary_path.unlink(missing_ok=True)


def check_outputs(outputs: Mapping[Path, bytes]) -> None:
    problems: list[str] = []
    for path, expected in outputs.items():
        try:
            observed = path.read_bytes()
        except OSError as exc:
            problems.append(f"{path}: {exc}")
            continue
        if observed != expected:
            problems.append(f"{path}: stale or inconsistent")
    if problems:
        raise ConfigError("materialized Vortex config check failed:\n  " + "\n  ".join(problems))


def run_generator(
    root: Path,
    generator_path: Path,
    source_path: Path,
    generator_cflags: str,
) -> str:
    command = [
        sys.executable,
        _display_path(generator_path, root),
        f"--config={_display_path(source_path, root)}",
        f"--cflags={generator_cflags}",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ConfigError(f"cannot execute Vortex config generator: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
        raise ConfigError(f"Vortex config generator failed ({completed.returncode}): {detail}")
    return completed.stdout


def materialize(
    root: Path,
    design_path: Path,
    source_path: Path,
    generator_path: Path,
    build_dir: Path,
    realization_profile: str,
    generator_cflags: str,
    check_only: bool,
) -> ResolvedConfig:
    if not realization_profile:
        raise ConfigError("realization profile must not be empty")
    try:
        generator_tokens = shlex.split(generator_cflags, posix=True)
    except ValueError as exc:
        raise ConfigError(f"cannot parse generator cflags: {exc}") from exc
    required_xlen_flag = f"-DVX_CFG_XLEN={REQUIRED_XLEN}"
    if required_xlen_flag not in generator_tokens:
        raise ConfigError(f"generator cflags must contain {required_xlen_flag}")

    design = load_design_config(design_path)
    cflags = run_generator(root, generator_path, source_path, generator_cflags)
    resolved = resolve_config(design, cflags)
    manifest = build_manifest(
        root,
        design_path,
        source_path,
        design,
        resolved,
        realization_profile,
        generator_cflags,
    )
    outputs = render_outputs(build_dir, manifest, resolved)
    if check_only:
        check_outputs(outputs)
    else:
        atomic_write_outputs(outputs)
    return resolved


def run_self_tests() -> int:
    import unittest

    class MaterializeTests(unittest.TestCase):
        def test_parse_defines_uses_shell_tokens_and_normalizes_values(self) -> None:
            self.assertEqual(
                parse_resolved_defines("-DFLAG -DDEC=08 -DHEX=0x10 '-DNAME=STD'"),
                {"DEC": 8, "FLAG": True, "HEX": 16, "NAME": "STD"},
            )

        def test_conflicting_duplicate_define_is_rejected(self) -> None:
            with self.assertRaises(ConfigError):
                parse_resolved_defines("-DVALUE=1 -DVALUE=2")

        def test_enabled_value_wins_and_missing_boolean_is_false(self) -> None:
            requested = {"xlen": 64, "tensor": False, "int8": True, "async_copy": False}
            keys = {
                "tensor": "VX_CFG_EXT_TCU_ENABLE",
                "int8": "VX_CFG_TCU_INT8_ENABLE",
                "async_copy": "VX_CFG_EXT_DXA_ENABLE",
            }
            defines = {
                "VX_CFG_XLEN": 64,
                "VX_CFG_EXT_TCU_ENABLE": True,
                "VX_CFG_EXT_TCU_ENABLED": 0,
                "VX_CFG_TCU_INT8_ENABLE": True,
            }
            selected = select_resolved_values(requested, keys, defines)
            self.assertEqual(selected, requested)
            validate_requested_values(requested, selected)

        def test_exact_type_and_value_match_is_required(self) -> None:
            with self.assertRaises(ConfigError):
                validate_requested_values({"xlen": 64}, {"xlen": "64"})

        def test_canonical_hash_is_independent_of_define_order(self) -> None:
            first = canonical_resolved_json({"B": 2, "A": True}, 64)
            second = canonical_resolved_json({"A": True, "B": 2}, 64)
            self.assertEqual(first, b'{"defines":{"A":true,"B":2},"xlen":64}')
            self.assertEqual(hashlib.sha256(first).digest(), hashlib.sha256(second).digest())

        def test_canonical_json_uses_utf8_for_cross_language_hashing(self) -> None:
            self.assertEqual(
                canonical_resolved_json({"NAME": "µ"}, 64),
                '{"defines":{"NAME":"µ"},"xlen":64}'.encode(),
            )

        def test_atomic_write_and_check(self) -> None:
            with tempfile.TemporaryDirectory() as temporary_dir:
                destination = Path(temporary_dir) / "nested" / "value"
                outputs = {destination: b"content\n"}
                atomic_write_outputs(outputs)
                check_outputs(outputs)
                self.assertEqual(destination.read_bytes(), b"content\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MaterializeTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def _path_from_root(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def parse_args(argv: list[str]) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(default_root), help="Mandrel repository root")
    parser.add_argument(
        "--design-config",
        default="hardware/vortex/configs/current-default.toml",
        help="design config path, relative to --root by default",
    )
    parser.add_argument(
        "--source-config",
        default="external/vortex/VX_config.toml",
        help="upstream Vortex config path, relative to --root by default",
    )
    parser.add_argument(
        "--generator",
        default="external/vortex/ci/gen_config.py",
        help="upstream config generator path, relative to --root by default",
    )
    parser.add_argument(
        "--build-dir",
        default="external/vortex-build",
        help="Vortex build directory, relative to --root by default",
    )
    parser.add_argument(
        "--realization-profile",
        default=DEFAULT_REALIZATION_PROFILE,
        help="name of the Vortex realization profile being resolved",
    )
    parser.add_argument(
        "--generator-cflags",
        default=DEFAULT_GENERATOR_CFLAGS,
        help="exact cflags passed to upstream gen_config.py for this realization",
    )
    parser.add_argument("--check", action="store_true", help="verify outputs without writing them")
    parser.add_argument("--self-test", action="store_true", help="run embedded unit tests")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_tests()

    root = Path(args.root).resolve()
    try:
        resolved = materialize(
            root=root,
            design_path=_path_from_root(root, args.design_config),
            source_path=_path_from_root(root, args.source_config),
            generator_path=_path_from_root(root, args.generator),
            build_dir=_path_from_root(root, args.build_dir),
            realization_profile=args.realization_profile,
            generator_cflags=args.generator_cflags,
            check_only=args.check,
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    action = "verified" if args.check else "materialized"
    print(f"{action} Vortex config {resolved.tag} ({resolved.sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
