# Vortex hardware inputs

This directory is the tracked input surface for Mandrel's hardware branch.

```text
source.lock.toml
  + configs/*.toml
  + patches/*
    -> resolved Vortex configuration
    -> canonical manifest + SHA-256 + RTL association tag
    -> generated Vortex/Mandrel configuration headers
    -> pinned project-local Verilator RTLSim, FPGA, or synthesis realization
    -> experiment identity and evidence
```

`external/vortex` and `external/llvm-vortex` are materialized checkouts. They are not the source of experiment identity by themselves. A complete report must bind source revisions, patch identity, resolved configuration, build identity, compiler target, executable artifact, and evidence class.

## Current configuration

`configs/current-default.toml` describes the current executable Vortex SystemVerilog baseline through pinned project-local Verilator RTLSim. It does not enable TCU or DXA. Its `[upstream_keys]` table maps every tracked design field except XLEN to the corresponding resolved upstream define.

The current materializer is intentionally strict: it invokes pinned `external/vortex/ci/gen_config.py` with the same fixed `-DSIMULATION -DSV_DPI` profile passed to the Verilator RTLSim build, parses all resolved `-D` values, and rejects any difference between the tracked requested subset and the upstream resolution. It does **not** currently rewrite `external/vortex/VX_config.toml`; deliberate config override generation is the next hardware-materialization step.

Run materialization as part of normal setup:

```sh
make setup-vortex
```

Read-only integrity checking is part of:

```sh
make env-check
```

## Materialized identity

`scripts/env/materialize-vortex-config.py` canonicalizes the complete sorted resolved define set plus XLEN, computes SHA-256, and atomically writes under the configured Vortex build directory:

```text
mandrel/vortex-config.json
mandrel/vortex-config.sha256
mandrel/vortex-config.tag
hw/VX_mandrel.vh
sw/VX_mandrel.h
```

For the default build directory these paths live under `external/vortex-build/`. The JSON manifest records the tracked design input identity, upstream source-config identity, realization profile and exact generator cflags, requested/resolved subset, all resolved defines, full SHA-256, and tag. Before compiling a kernel or accepting evidence, Rust recomputes the canonical digest, validates the expected RTLSim profile and both sidecars, and derives compiler/ISA-check defines from the validated manifest.

The tag is the first 64 bits of the full digest. `VX_mandrel.vh` compiles it into Vortex RTL, where patch `0002-rtl-expose-mandrel-config-identity-csr.patch` exposes it through read-only custom CSR `0xFC5`. A generated MLIR/LLVM infrastructure probe reads the CSR before attention runs. The host accepts performance evidence only when the realized and observed tags match exactly.

The full SHA-256 is authoritative host-side configuration identity. The 64-bit CSR value is a runtime association tag, not a complete cryptographic proof. The probe runs on a separate backend/device instance so its cycles and instructions do not contaminate attention profiling.

Future configurations should vary one reviewed hardware factor at a time unless an experiment explicitly defines a factorial design.
