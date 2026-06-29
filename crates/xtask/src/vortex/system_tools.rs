use super::*;

pub(crate) fn prepare_and_print_vortex_system_tools(workspace_root: &Path) -> Result<()> {
    let mut config = VortexConfig::from_env(workspace_root)?;
    config.toolchain_mode = VortexToolchainMode::System;
    if env::var_os("MANDREL_VORTEX_TOOLDIR").is_none() {
        config.tool_dir = workspace_root.join(DEFAULT_VORTEX_SYSTEM_TOOLDIR);
    }
    log_vortex_config(&config, "preparing Vortex system-package tool layout");
    prepare_vortex_system_tools(&config)?;
    println!(
        "Vortex system tool wrappers ready at: {}",
        config.tool_dir.display()
    );
    println!("Use them with: MANDREL_VORTEX_TOOLCHAIN_MODE=system cargo vortex-install");
    Ok(())
}

pub(super) fn prepare_vortex_system_tools(config: &VortexConfig) -> Result<()> {
    if config.xlen != 64 {
        return Err(format!(
            "MANDREL_VORTEX_TOOLCHAIN_MODE=system currently supports MANDREL_VORTEX_XLEN=64 only; got {}. Use a source-built Vortex toolchain with MANDREL_VORTEX_TOOLCHAIN_MODE=skip for XLEN=32.",
            config.xlen
        ).into());
    }

    require_system_programs(["g++", "make", "python3"])?;
    let riscv_c_include_dir = require_riscv_c_library_include_dir()?;
    let riscv_c_lib_dir = require_riscv_c_library_lib_dir(&riscv_c_include_dir)?;
    let riscv_runtime = require_riscv_builtins_runtime(config)?;

    let riscv_bin = config.tool_dir.join("riscv64-gnu-toolchain/bin");
    fs::create_dir_all(&riscv_bin).map_err(|error| {
        format!(
            "failed to create system RISC-V wrapper directory '{}': {error}",
            riscv_bin.display()
        )
    })?;
    let riscv_sysroot = config
        .tool_dir
        .join("riscv64-gnu-toolchain/riscv64-unknown-elf");
    replace_symlink_or_empty_dir(&riscv_sysroot.join("include"), &riscv_c_include_dir)?;
    fs::create_dir_all(
        config
            .tool_dir
            .join("riscv64-gnu-toolchain/riscv64-unknown-elf/lib"),
    )
    .map_err(|error| format!("failed to create Vortex-compatible RISC-V lib directory: {error}"))?;

    replace_symlink_or_empty_dir(&config.tool_dir.join("libc64/lib"), &riscv_c_lib_dir)?;
    prepare_vortex_system_runtime_overrides(config, &riscv_runtime)?;

    let gcc = require_program("riscv64-unknown-elf-gcc")?;
    write_riscv_gcc_wrapper(
        &riscv_bin.join("riscv64-unknown-elf-gcc"),
        &gcc,
        &riscv_c_include_dir,
    )?;
    if let Some(gxx) = find_program_on_path("riscv64-unknown-elf-g++") {
        write_riscv_gcc_wrapper(
            &riscv_bin.join("riscv64-unknown-elf-g++"),
            &gxx,
            &riscv_c_include_dir,
        )?;
        write_riscv_gcc_wrapper(
            &riscv_bin.join("riscv64-unknown-elf-c++"),
            &gxx,
            &riscv_c_include_dir,
        )?;
    }

    for suffix in ["gcc-ar", "objdump", "objcopy"] {
        let name = format!("riscv64-unknown-elf-{suffix}");
        let source = require_program(&name)?;
        replace_symlink(&riscv_bin.join(&name), &source)?;
    }

    for suffix in [
        "ar",
        "as",
        "cpp",
        "gcc-nm",
        "gcc-ranlib",
        "ld",
        "ld.bfd",
        "nm",
        "ranlib",
        "readelf",
        "size",
        "strings",
        "strip",
    ] {
        let name = format!("riscv64-unknown-elf-{suffix}");
        if let Some(source) = find_program_on_path(&name) {
            replace_symlink(&riscv_bin.join(&name), &source)?;
        }
    }

    let llvm_bin = config.tool_dir.join("llvm-vortex/bin");
    fs::create_dir_all(&llvm_bin).map_err(|error| {
        format!(
            "failed to create system LLVM wrapper directory '{}': {error}",
            llvm_bin.display()
        )
    })?;
    for program in [
        "clang",
        "clang++",
        "ld.lld",
        "lld",
        "llvm-ar",
        "llvm-config",
        "llvm-objcopy",
        "llvm-objdump",
        "llvm-readelf",
        "llvm-size",
        "llvm-spirv",
    ] {
        if let Some(source) = find_llvm_program(program) {
            replace_symlink(&llvm_bin.join(program), &source)?;
        }
    }

    if let Some(source) = find_program_on_path("verilator") {
        let verilator_bin = config.tool_dir.join("verilator/bin");
        fs::create_dir_all(&verilator_bin).map_err(|error| {
            format!(
                "failed to create system Verilator wrapper directory '{}': {error}",
                verilator_bin.display()
            )
        })?;
        replace_symlink(&verilator_bin.join("verilator"), &source)?;
    } else {
        println!(
            "Optional system Verilator not found. Install `verilator` before using MANDREL_VORTEX_BUILD_PROFILE=full or RTL simulation."
        );
    }

    println!(
        "Prepared system-package Vortex tool layout at: {}",
        config.tool_dir.display()
    );
    match &riscv_runtime {
        RiscvBuiltinsRuntime::CompilerRt { path } => println!(
            "Using RISC-V compiler-rt builtins for Vortex device links: {}",
            path.display()
        ),
        RiscvBuiltinsRuntime::Libgcc { archive } => println!(
            "Using explicit experimental RISC-V libgcc builtins compatibility link for Vortex device links: {}",
            archive.display()
        ),
    }
    println!(
        "Note: Ubuntu LLVM is enough for the minimal software SDK path, but official Vortex kernels using `+xvortex` still need Vortex-patched LLVM."
    );
    Ok(())
}

pub(crate) fn reject_obvious_incompatible_prebuilt_tools(config: &VortexConfig) -> Result<()> {
    if env::consts::ARCH == "x86_64" {
        return Ok(());
    }

    for relative in [
        "verilator/bin/verilator_bin",
        "llvm-vortex/bin/llvm-config",
        "riscv64-gnu-toolchain/bin/riscv64-unknown-elf-gcc",
    ] {
        let path = config.tool_dir.join(relative);
        if !path.is_file() {
            continue;
        }
        let Some(file_output) = inspect_file_type(&path)? else {
            continue;
        };
        if file_output.contains("x86-64") || file_output.contains("x86_64") {
            return Err(format!(
                "MANDREL_VORTEX_TOOLCHAIN_MODE=skip is pointing at an x86_64 prebuilt tool on this {} host: {}\n{}\nUse MANDREL_VORTEX_TOOLCHAIN_MODE=system for Ubuntu/system packages, set MANDREL_VORTEX_TOOLDIR to a source-built ARM tool layout, or remove stale prebuilt artifacts under '{}'.",
                env::consts::ARCH,
                path.display(),
                file_output.trim(),
                config.tool_dir.display()
            ).into());
        }
    }

    Ok(())
}

pub(super) fn inspect_file_type(path: &Path) -> Result<Option<String>> {
    let Some(file_program) = find_program_on_path("file") else {
        warn!(path = %path.display(), "`file` command not found; cannot inspect Vortex tool binary architecture");
        return Ok(None);
    };
    let output = Command::new(file_program)
        .arg(path)
        .output()
        .map_err(|error| format!("failed to inspect '{}': {error}", path.display()))?;
    if !output.status.success() {
        warn!(path = %path.display(), status = %output.status, "`file` command failed while inspecting Vortex tool");
        return Ok(None);
    }
    Ok(Some(String::from_utf8_lossy(&output.stdout).into_owned()))
}

pub(super) fn write_riscv_gcc_wrapper(
    wrapper_path: &Path,
    real_program: &Path,
    include_dir: &Path,
) -> Result<()> {
    let content = format!(
        "#!/usr/bin/env bash\n\
         set -euo pipefail\n\
         exec {} -isystem {} \"$@\"\n",
        shell_quote_lossy(real_program.as_os_str()),
        shell_quote_lossy(include_dir.as_os_str())
    );
    replace_file_content(wrapper_path, content.as_bytes(), "RISC-V GCC wrapper")?;
    make_executable(wrapper_path)?;
    Ok(())
}

pub(super) fn replace_file_content(path: &Path, content: &[u8], description: &str) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create {description} parent '{}': {error}",
                parent.display()
            )
        })?;
    }

    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_dir() && !metadata.file_type().is_symlink() {
                return Err(format!(
                    "cannot replace directory '{}' with {description}",
                    path.display()
                )
                .into());
            }
            fs::remove_file(path).map_err(|error| {
                format!(
                    "failed to remove existing {description} '{}': {error}",
                    path.display()
                )
            })?;
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!(
                "failed to inspect existing {description} '{}': {error}",
                path.display()
            )
            .into());
        }
    }

    fs::write(path, content).map_err(|error| {
        XtaskError::message(format!(
            "failed to write {description} '{}': {error}",
            path.display()
        ))
    })
}

pub(super) fn require_riscv_c_library_include_dir() -> Result<PathBuf> {
    if let Some(path) = env::var_os("MANDREL_RISCV_C_INCLUDE_DIR").map(PathBuf::from) {
        if path.join("newlib.h").is_file() {
            return Ok(path);
        }
        return Err(format!(
            "MANDREL_RISCV_C_INCLUDE_DIR points to '{}' but newlib.h was not found there",
            path.display()
        )
        .into());
    }

    for candidate in [
        "/usr/lib/picolibc/riscv64-unknown-elf/include",
        "/usr/riscv64-unknown-elf/include",
        "/usr/lib/riscv64-unknown-elf/include",
    ] {
        let path = PathBuf::from(candidate);
        if path.join("newlib.h").is_file() {
            return Ok(path);
        }
    }

    Err(XtaskError::message(
        "missing RISC-V bare-metal C library headers: Vortex device runtime includes <newlib.h>. Install Ubuntu package `picolibc-riscv64-unknown-elf`, or set MANDREL_RISCV_C_INCLUDE_DIR to a directory containing newlib.h.",
    ))
}

pub(super) fn require_riscv_c_library_lib_dir(include_dir: &Path) -> Result<PathBuf> {
    if let Some(path) = env::var_os("MANDREL_RISCV_C_LIB_DIR").map(PathBuf::from) {
        if path.join("libc.a").is_file() && path.join("libm.a").is_file() {
            return Ok(path);
        }
        return Err(format!(
            "MANDREL_RISCV_C_LIB_DIR points to '{}' but libc.a and libm.a were not found there",
            path.display()
        )
        .into());
    }

    let mut candidates = Vec::new();
    if let Some(root) = include_dir.parent() {
        candidates.push(root.join("lib"));
        candidates.push(root.join("lib/release"));
    }
    candidates.extend([
        PathBuf::from("/usr/lib/picolibc/riscv64-unknown-elf/lib"),
        PathBuf::from("/usr/lib/picolibc/riscv64-unknown-elf/lib/release"),
        PathBuf::from("/usr/riscv64-unknown-elf/lib"),
        PathBuf::from("/usr/lib/riscv64-unknown-elf/lib"),
    ]);

    for path in candidates {
        if path.join("libc.a").is_file() && path.join("libm.a").is_file() {
            return Ok(path);
        }
    }

    Err(XtaskError::message(
        "missing RISC-V bare-metal C libraries: Vortex kernels link with -lc and -lm. Install Ubuntu package `picolibc-riscv64-unknown-elf`, or set MANDREL_RISCV_C_LIB_DIR to a directory containing libc.a and libm.a.",
    ))
}

pub(super) enum RiscvBuiltinsRuntime {
    CompilerRt { path: PathBuf },
    Libgcc { archive: PathBuf },
}

pub(super) fn require_riscv_builtins_runtime(
    config: &VortexConfig,
) -> Result<RiscvBuiltinsRuntime> {
    let allow_libgcc = allow_libgcc_builtins()?;

    if let Some(path) = env::var_os("MANDREL_RISCV_BUILTINS_LIB").map(PathBuf::from) {
        return classify_explicit_riscv_builtins_library(path, allow_libgcc);
    }

    if let Some(path) = find_compiler_rt_builtins_library(config) {
        return Ok(RiscvBuiltinsRuntime::CompilerRt { path });
    }

    if allow_libgcc {
        let archive = require_riscv_libgcc_library()?;
        return Ok(RiscvBuiltinsRuntime::Libgcc { archive });
    }

    let expected = config
        .tool_dir
        .join("libcrt64/lib/baremetal/libclang_rt.builtins-riscv64.a");
    Err(format!(
        "missing RISC-V compiler-rt builtins for Vortex system mode. Expected '{}', a system LLVM libclang_rt.builtins-riscv64.a, or MANDREL_RISCV_BUILTINS_LIB pointing to a real compiler-rt archive. Vortex upstream documents this archive as part of llvm-vortex/compiler-rt; build/install that first for the supported path. Experimental ABI-aligned libgcc compatibility is disabled by default; set MANDREL_ALLOW_LIBGCC_BUILTINS=1 only if you intentionally want to use riscv64-unknown-elf-gcc's -march=rv64imafd -mabi=lp64d libgcc.a through the Vortex libcrt compatibility path.",
        expected.display()
    ).into())
}

pub(super) fn allow_libgcc_builtins() -> Result<bool> {
    let Some(value) = non_empty_env("MANDREL_ALLOW_LIBGCC_BUILTINS") else {
        return Ok(false);
    };

    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        other => Err(format!(
            "unsupported MANDREL_ALLOW_LIBGCC_BUILTINS value '{other}'; use 1/0, true/false, yes/no, or on/off"
        ).into()),
    }
}

pub(super) fn classify_explicit_riscv_builtins_library(
    path: PathBuf,
    allow_libgcc: bool,
) -> Result<RiscvBuiltinsRuntime> {
    if !path.is_file() {
        return Err(format!(
            "MANDREL_RISCV_BUILTINS_LIB points to '{}' but no file was found there",
            path.display()
        )
        .into());
    }

    if is_libgcc_or_link_to_libgcc_archive(&path) {
        if !allow_libgcc {
            return Err(format!(
                "MANDREL_RISCV_BUILTINS_LIB points to libgcc.a at '{}', but libgcc builtins are experimental and disabled by default. Use Vortex llvm-vortex/compiler-rt for the supported path, or set MANDREL_ALLOW_LIBGCC_BUILTINS=1 to opt in explicitly.",
                path.display()
            ).into());
        }
        Ok(RiscvBuiltinsRuntime::Libgcc { archive: path })
    } else {
        Ok(RiscvBuiltinsRuntime::CompilerRt { path })
    }
}

pub(super) fn find_compiler_rt_builtins_library(config: &VortexConfig) -> Option<PathBuf> {
    let local = config
        .tool_dir
        .join("libcrt64/lib/baremetal/libclang_rt.builtins-riscv64.a");
    if is_usable_compiler_rt_candidate(&local) {
        return Some(local);
    }

    for version in [20, 19, 18, 17, 16, 15, 14] {
        for clang_version in [
            version.to_string(),
            format!("{version}.0.0"),
            format!("{version}.1.0"),
        ] {
            for subdir in ["baremetal", "linux"] {
                let candidate = PathBuf::from(format!(
                    "/usr/lib/llvm-{version}/lib/clang/{clang_version}/lib/{subdir}/libclang_rt.builtins-riscv64.a"
                ));
                if is_usable_compiler_rt_candidate(&candidate) {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

pub(super) fn is_usable_compiler_rt_candidate(path: &Path) -> bool {
    path.is_file() && !is_libgcc_or_link_to_libgcc_archive(path)
}

pub(super) fn require_riscv_libgcc_library() -> Result<PathBuf> {
    let gcc = require_program("riscv64-unknown-elf-gcc")?;
    let output = Command::new(&gcc)
        .args(["-march=rv64imafd", "-mabi=lp64d", "-print-libgcc-file-name"])
        .output()
        .map_err(|error| {
            format!(
                "failed to query RISC-V libgcc path from '{}': {error}",
                gcc.display()
            )
        })?;
    if !output.status.success() {
        return Err(format!(
            "failed to query RISC-V libgcc path from '{}' with status {}",
            gcc.display(),
            output.status
        )
        .into());
    }

    let raw = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let path = PathBuf::from(&raw);
    if path.is_file() && is_libgcc_archive(&path) {
        return Ok(path);
    }

    Err(format!(
        "MANDREL_ALLOW_LIBGCC_BUILTINS=1 was set, but riscv64-unknown-elf-gcc did not report a usable ABI-specific libgcc.a for -march=rv64imafd -mabi=lp64d. riscv64-unknown-elf-gcc reported: {raw}"
    ).into())
}

pub(super) fn is_libgcc_or_link_to_libgcc_archive(path: &Path) -> bool {
    if is_libgcc_archive(path) {
        return true;
    }

    match fs::read_link(path) {
        Ok(target) => is_libgcc_archive(&target),
        Err(_) => false,
    }
}

pub(super) fn is_libgcc_archive(path: &Path) -> bool {
    path.file_name().and_then(OsStr::to_str) == Some("libgcc.a")
}

pub(super) fn prepare_vortex_system_runtime_overrides(
    config: &VortexConfig,
    runtime: &RiscvBuiltinsRuntime,
) -> Result<()> {
    let link = config
        .tool_dir
        .join("libcrt64/lib/baremetal/libclang_rt.builtins-riscv64.a");
    let target = match runtime {
        RiscvBuiltinsRuntime::CompilerRt { path } => path,
        RiscvBuiltinsRuntime::Libgcc { archive } => archive,
    };

    if &link != target {
        replace_symlink(&link, target)?;
    }
    Ok(())
}

pub(super) fn require_system_programs<const N: usize>(programs: [&str; N]) -> Result<()> {
    let missing: Vec<&str> = programs
        .into_iter()
        .filter(|program| find_program_on_path(program).is_none())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "missing required system programs for Vortex system mode: {}. Suggested Ubuntu packages: build-essential make python3 gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf clang lld llvm-18-dev; optional full simulation: verilator.",
            missing.join(", ")
        ).into())
    }
}

pub(super) fn require_program(program: &str) -> Result<PathBuf> {
    find_program_on_path(program).ok_or_else(|| {
        XtaskError::message(format!(
            "missing required program '{program}'. Suggested Ubuntu packages: gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf"
        ))
    })
}

pub(super) fn find_program_on_path(program: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|dir| dir.join(program))
        .find(|candidate| candidate.is_file())
}

pub(super) fn find_llvm_program(program: &str) -> Option<PathBuf> {
    find_program_on_path(program)
        .or_else(|| find_program_on_path(&format!("{program}-18")))
        .or_else(|| find_program_on_path(&format!("{program}-17")))
        .or_else(|| find_program_on_path(&format!("{program}-16")))
        .or_else(|| {
            [20, 19, 18, 17, 16, 15]
                .into_iter()
                .map(|version| PathBuf::from(format!("/usr/lib/llvm-{version}/bin/{program}")))
                .find(|candidate| candidate.is_file())
        })
}

pub(super) fn replace_symlink_or_empty_dir(link: &Path, target: &Path) -> Result<()> {
    if let Some(parent) = link
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create tool wrapper parent '{}': {error}",
                parent.display()
            )
        })?;
    }

    match fs::symlink_metadata(link) {
        Ok(metadata) if metadata.file_type().is_dir() && !metadata.file_type().is_symlink() => {
            fs::remove_dir(link).map_err(|error| {
                format!(
                    "failed to replace existing directory '{}' with symlink to '{}': {error}. Remove the directory if it is intentionally non-empty.",
                    link.display(),
                    target.display()
                )
            })?;
        }
        Ok(_) => fs::remove_file(link)
            .map_err(|error| format!("failed to remove existing '{}': {error}", link.display()))?,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!("failed to inspect existing '{}': {error}", link.display()).into());
        }
    }

    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link).map_err(|error| {
            XtaskError::message(format!(
                "failed to create symlink '{}' -> '{}': {error}",
                link.display(),
                target.display()
            ))
        })
    }

    #[cfg(not(unix))]
    {
        if target.is_dir() {
            fs::create_dir_all(link).map_err(|error| {
                XtaskError::message(format!(
                    "failed to create directory '{}' for '{}': {error}",
                    link.display(),
                    target.display()
                ))
            })
        } else {
            fs::copy(target, link).map(|_| ()).map_err(|error| {
                XtaskError::message(format!(
                    "failed to copy '{}' to '{}': {error}",
                    target.display(),
                    link.display()
                ))
            })
        }
    }
}

pub(super) fn replace_symlink(link: &Path, target: &Path) -> Result<()> {
    if let Some(parent) = link
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create tool wrapper parent '{}': {error}",
                parent.display()
            )
        })?;
    }

    if let Ok(metadata) = fs::symlink_metadata(link) {
        if metadata.file_type().is_dir() && !metadata.file_type().is_symlink() {
            return Err(format!(
                "cannot replace directory '{}' with symlink to '{}'",
                link.display(),
                target.display()
            )
            .into());
        }
        fs::remove_file(link)
            .map_err(|error| format!("failed to remove existing '{}': {error}", link.display()))?;
    }

    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link).map_err(|error| {
            XtaskError::message(format!(
                "failed to create symlink '{}' -> '{}': {error}",
                link.display(),
                target.display()
            ))
        })
    }

    #[cfg(not(unix))]
    {
        fs::copy(target, link).map(|_| ()).map_err(|error| {
            XtaskError::message(format!(
                "failed to copy '{}' to '{}': {error}",
                target.display(),
                link.display()
            ))
        })
    }
}
