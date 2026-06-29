use super::*;

#[derive(Debug, Clone)]
pub(super) struct VortexSourceToolchainConfig {
    tool_dir: PathBuf,
    llvm_source_dir: PathBuf,
    llvm_build_dir: PathBuf,
    compiler_rt_build_dir: PathBuf,
    llvm_url: String,
    llvm_ref: String,
    llvm_projects: String,
    llvm_targets: String,
    jobs: usize,
}

impl VortexSourceToolchainConfig {
    fn from_env(workspace_root: &Path, tool_dir: PathBuf) -> Result<Self> {
        Ok(Self {
            tool_dir,
            llvm_source_dir: project_path_from_env(
                workspace_root,
                "MANDREL_VORTEX_LLVM_DIR",
                DEFAULT_VORTEX_LLVM_DIR,
            ),
            llvm_build_dir: project_path_from_env(
                workspace_root,
                "MANDREL_VORTEX_LLVM_BUILD_DIR",
                DEFAULT_VORTEX_LLVM_BUILD_DIR,
            ),
            compiler_rt_build_dir: project_path_from_env(
                workspace_root,
                "MANDREL_VORTEX_COMPILER_RT_BUILD_DIR",
                DEFAULT_VORTEX_COMPILER_RT_BUILD_DIR,
            ),
            llvm_url: non_empty_env("MANDREL_VORTEX_LLVM_URL")
                .unwrap_or_else(|| DEFAULT_VORTEX_LLVM_URL.to_owned()),
            llvm_ref: non_empty_env("MANDREL_VORTEX_LLVM_REF")
                .unwrap_or_else(|| DEFAULT_VORTEX_LLVM_REF.to_owned()),
            llvm_projects: non_empty_env("MANDREL_VORTEX_LLVM_PROJECTS")
                .unwrap_or_else(|| DEFAULT_VORTEX_LLVM_PROJECTS.to_owned()),
            llvm_targets: non_empty_env("MANDREL_VORTEX_LLVM_TARGETS")
                .unwrap_or_else(default_vortex_llvm_targets_for_host),
            jobs: source_toolchain_jobs()?,
        })
    }

    fn llvm_prefix(&self) -> PathBuf {
        self.tool_dir.join("llvm-vortex")
    }

    fn llvm_bin_dir(&self) -> PathBuf {
        self.llvm_prefix().join("bin")
    }

    fn llvm_projects_include(&self, project: &str) -> bool {
        self.llvm_projects
            .split([';', ','])
            .any(|candidate| candidate.trim().eq_ignore_ascii_case(project))
    }

    fn llvm_lib_dir(&self) -> PathBuf {
        self.llvm_prefix().join("lib")
    }

    fn riscv_gcc_toolchain_dir(&self) -> PathBuf {
        self.tool_dir.join("riscv64-gnu-toolchain")
    }

    fn riscv_sysroot_dir(&self) -> PathBuf {
        self.riscv_gcc_toolchain_dir().join("riscv64-unknown-elf")
    }

    fn compiler_rt_install_dir(&self) -> PathBuf {
        self.tool_dir.join("libcrt64")
    }

    fn compiler_rt_builtins_archive(&self) -> PathBuf {
        self.compiler_rt_install_dir()
            .join("lib/baremetal/libclang_rt.builtins-riscv64.a")
    }
}

pub(crate) fn install_vortex_source_toolchain(workspace_root: &Path) -> Result<()> {
    let mut vortex_config = VortexConfig::from_env(workspace_root)?;
    vortex_config.toolchain_mode = VortexToolchainMode::Skip;
    if env::var_os("MANDREL_VORTEX_TOOLDIR").is_none() {
        vortex_config.tool_dir = workspace_root.join(DEFAULT_VORTEX_SOURCE_TOOLDIR);
    }

    let source_config =
        VortexSourceToolchainConfig::from_env(workspace_root, vortex_config.tool_dir.clone())?;
    log_vortex_config(
        &vortex_config,
        "source-building Vortex llvm-vortex toolchain",
    );
    log_vortex_source_toolchain_config(&source_config);

    require_source_toolchain_programs()?;
    prepare_vortex_source_toolchain_riscv_layout(&vortex_config)?;

    clone_vortex_if_needed(&vortex_config)?;
    checkout_vortex_ref_if_requested(&vortex_config)?;
    configure_vortex_for_source_toolchain(&vortex_config)?;
    build_vortex_kernel_runtime_archive(&vortex_config)?;

    clone_or_update_vortex_llvm_source(&source_config, &vortex_config)?;
    build_vortex_llvm(&source_config)?;
    verify_vortex_llvm_install(&source_config)?;
    build_vortex_compiler_rt64(&source_config, &vortex_config)?;
    verify_vortex_compiler_rt_install(&source_config)?;

    write_vortex_env_script(&vortex_config)?;
    print_source_toolchain_success(&source_config, &vortex_config);
    Ok(())
}

pub(super) fn log_vortex_source_toolchain_config(config: &VortexSourceToolchainConfig) {
    info!(
        tool_dir = %config.tool_dir.display(),
        llvm_source_dir = %config.llvm_source_dir.display(),
        llvm_build_dir = %config.llvm_build_dir.display(),
        compiler_rt_build_dir = %config.compiler_rt_build_dir.display(),
        llvm_url = %config.llvm_url,
        llvm_ref = %config.llvm_ref,
        llvm_projects = %config.llvm_projects,
        llvm_targets = %config.llvm_targets,
        jobs = config.jobs,
        "Vortex source toolchain config"
    );
    println!("Vortex source toolchain layout:");
    println!("  tools:               {}", config.tool_dir.display());
    println!(
        "  llvm source:         {}",
        config.llvm_source_dir.display()
    );
    println!("  llvm build:          {}", config.llvm_build_dir.display());
    println!(
        "  compiler-rt build:   {}",
        config.compiler_rt_build_dir.display()
    );
    println!("  llvm url:            {}", config.llvm_url);
    println!("  llvm ref:            {}", config.llvm_ref);
    println!("  llvm projects:       {}", config.llvm_projects);
    println!("  llvm targets:        {}", config.llvm_targets);
    println!("  build jobs:          {}", config.jobs);
}

pub(super) fn default_vortex_llvm_targets_for_host() -> String {
    match env::consts::ARCH {
        "x86" | "x86_64" => "RISCV;X86".to_owned(),
        "aarch64" => "RISCV;AArch64".to_owned(),
        "arm" => "RISCV;ARM".to_owned(),
        other => {
            warn!(
                host_arch = other,
                "unknown LLVM host backend for source-built Vortex LLVM; defaulting to RISCV only"
            );
            "RISCV".to_owned()
        }
    }
}

pub(super) fn source_toolchain_jobs() -> Result<usize> {
    if let Some(raw) = non_empty_env("MANDREL_VORTEX_TOOLCHAIN_JOBS") {
        let jobs = raw
            .parse::<usize>()
            .map_err(|error| format!("invalid MANDREL_VORTEX_TOOLCHAIN_JOBS '{raw}': {error}"))?;
        if jobs == 0 {
            return Err(XtaskError::message(
                "MANDREL_VORTEX_TOOLCHAIN_JOBS must be at least 1",
            ));
        }
        return Ok(jobs);
    }

    let available = match std::thread::available_parallelism() {
        Ok(value) => value.get(),
        Err(error) => {
            warn!(%error, "failed to detect parallelism; defaulting source toolchain jobs to 1");
            1
        }
    };
    Ok(available.clamp(1, 4))
}

pub(super) fn require_source_toolchain_programs() -> Result<()> {
    let missing: Vec<&str> = [
        "git",
        "cmake",
        "ninja",
        "make",
        "python3",
        "riscv64-unknown-elf-gcc",
    ]
    .into_iter()
    .filter(|program| find_program_on_path(program).is_none())
    .collect();

    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "missing required programs for Vortex source toolchain build: {}. Suggested Ubuntu packages: build-essential cmake ninja-build git make python3 gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf picolibc-riscv64-unknown-elf.",
            missing.join(", ")
        ).into())
    }
}

pub(super) fn prepare_vortex_source_toolchain_riscv_layout(config: &VortexConfig) -> Result<()> {
    if config.xlen != 64 {
        return Err(format!(
            "cargo vortex-toolchain-source currently builds the rv64 Vortex compiler-rt path only; got MANDREL_VORTEX_XLEN={}. Use MANDREL_VORTEX_XLEN=64 or extend this command for rv32.",
            config.xlen
        ).into());
    }

    let riscv_c_include_dir = require_riscv_c_library_include_dir()?;
    let riscv_c_lib_dir = require_riscv_c_library_lib_dir(&riscv_c_include_dir)?;
    let riscv_bin = config.tool_dir.join("riscv64-gnu-toolchain/bin");
    fs::create_dir_all(&riscv_bin).map_err(|error| {
        format!(
            "failed to create source-build RISC-V tool directory '{}': {error}",
            riscv_bin.display()
        )
    })?;

    let riscv_sysroot = config
        .tool_dir
        .join("riscv64-gnu-toolchain/riscv64-unknown-elf");
    replace_symlink_or_empty_dir(&riscv_sysroot.join("include"), &riscv_c_include_dir)?;
    replace_symlink_or_empty_dir(&riscv_sysroot.join("lib"), &riscv_c_lib_dir)?;
    replace_symlink_or_empty_dir(
        &config.tool_dir.join("libc64/include"),
        &riscv_c_include_dir,
    )?;
    replace_symlink_or_empty_dir(&config.tool_dir.join("libc64/lib"), &riscv_c_lib_dir)?;

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

    link_riscv_gcc_support_dir(&config.tool_dir)?;
    println!(
        "Prepared source-build RISC-V GNU/newlib layout at: {}",
        config.tool_dir.display()
    );
    Ok(())
}

pub(super) fn link_riscv_gcc_support_dir(tool_dir: &Path) -> Result<()> {
    let archive = match require_riscv_libgcc_library() {
        Ok(path) => path,
        Err(error) => {
            warn!(%error, "could not mirror RISC-V GCC support library directory into source tool layout");
            return Ok(());
        }
    };

    let Some(version_dir) = archive.parent() else {
        warn!(archive = %archive.display(), "RISC-V libgcc path has no parent directory");
        return Ok(());
    };
    let Some(version) = version_dir.file_name() else {
        warn!(archive = %archive.display(), "RISC-V libgcc parent has no version directory name");
        return Ok(());
    };

    let link = tool_dir
        .join("riscv64-gnu-toolchain/lib/gcc/riscv64-unknown-elf")
        .join(version);
    replace_symlink_or_empty_dir(&link, version_dir)
}

pub(super) fn configure_vortex_for_source_toolchain(config: &VortexConfig) -> Result<()> {
    fs::create_dir_all(&config.build_dir).map_err(|error| {
        format!(
            "failed to create Vortex build directory '{}': {error}",
            config.build_dir.display()
        )
    })?;
    fs::create_dir_all(&config.tool_dir).map_err(|error| {
        format!(
            "failed to create Vortex source tool directory '{}': {error}",
            config.tool_dir.display()
        )
    })?;

    println!(
        "Configuring Vortex for source-built toolchain: {}",
        config.build_dir.display()
    );
    run_checked(
        config.configure_command(),
        "vortex.configure-source-toolchain",
    )
}

pub(super) fn build_vortex_kernel_runtime_archive(config: &VortexConfig) -> Result<()> {
    println!("Building Vortex sw/kernel runtime archive for compiler-rt link checks.");
    let mut make = Command::new("make");
    make.current_dir(&config.build_dir)
        .arg("-C")
        .arg("sw/kernel");
    apply_vortex_env(&mut make, config)?;
    run_checked(make, "vortex.make.sw-kernel")?;

    let archive = config.build_dir.join("sw/kernel/libvortex.a");
    if archive.is_file() {
        Ok(())
    } else {
        Err(format!(
            "Vortex sw/kernel build completed but '{}' was not found",
            archive.display()
        )
        .into())
    }
}

pub(super) fn clone_or_update_vortex_llvm_source(
    source_config: &VortexSourceToolchainConfig,
    vortex_config: &VortexConfig,
) -> Result<()> {
    if source_config.llvm_source_dir.join(".git").is_dir() {
        println!(
            "Updating Vortex LLVM checkout: {}",
            source_config.llvm_source_dir.display()
        );
        let mut remote = Command::new("git");
        remote
            .arg("--no-pager")
            .arg("-C")
            .arg(&source_config.llvm_source_dir)
            .args(["remote", "set-url", "origin"])
            .arg(&source_config.llvm_url);
        apply_github_proxy_git_env(&mut remote, vortex_config);
        run_checked(remote, "llvm-vortex.set-origin")?;

        run_checked_with_retries(
            || {
                let mut fetch = Command::new("git");
                fetch
                    .arg("--no-pager")
                    .arg("-C")
                    .arg(&source_config.llvm_source_dir)
                    .args(["fetch", "--tags", "origin"]);
                apply_github_proxy_git_env(&mut fetch, vortex_config);
                Ok(fetch)
            },
            "llvm-vortex.fetch",
            vortex_config.fetch_retries,
        )?;

        let mut checkout = Command::new("git");
        checkout
            .arg("--no-pager")
            .arg("-C")
            .arg(&source_config.llvm_source_dir)
            .arg("checkout")
            .arg(&source_config.llvm_ref);
        apply_github_proxy_git_env(&mut checkout, vortex_config);
        run_checked(checkout, "llvm-vortex.checkout")?;
        return update_vortex_llvm_submodules(source_config, vortex_config);
    }

    if source_config.llvm_source_dir.exists() {
        return Err(format!(
            "LLVM source directory '{}' exists but is not a git checkout; set MANDREL_VORTEX_LLVM_DIR or remove it",
            source_config.llvm_source_dir.display()
        ).into());
    }

    if let Some(parent) = source_config
        .llvm_source_dir
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create LLVM source parent directory '{}': {error}",
                parent.display()
            )
        })?;
    }

    println!(
        "Cloning Vortex LLVM {} into {}",
        source_config.llvm_ref,
        source_config.llvm_source_dir.display()
    );
    run_checked_with_retries(
        || {
            let mut clone = Command::new("git");
            clone
                .arg("clone")
                .arg("--recursive")
                .arg("--branch")
                .arg(&source_config.llvm_ref)
                .arg(&source_config.llvm_url)
                .arg(&source_config.llvm_source_dir);
            apply_github_proxy_git_env(&mut clone, vortex_config);
            Ok(clone)
        },
        "llvm-vortex.clone",
        vortex_config.fetch_retries,
    )
}

pub(super) fn update_vortex_llvm_submodules(
    source_config: &VortexSourceToolchainConfig,
    vortex_config: &VortexConfig,
) -> Result<()> {
    run_checked_with_retries(
        || {
            let mut submodules = Command::new("git");
            submodules
                .arg("--no-pager")
                .arg("-C")
                .arg(&source_config.llvm_source_dir)
                .args(["submodule", "update", "--init", "--recursive"]);
            apply_github_proxy_git_env(&mut submodules, vortex_config);
            Ok(submodules)
        },
        "llvm-vortex.submodule-update",
        vortex_config.fetch_retries,
    )
}

pub(super) fn build_vortex_llvm(source_config: &VortexSourceToolchainConfig) -> Result<()> {
    let llvm_source = source_config.llvm_source_dir.join("llvm");
    if !llvm_source.join("CMakeLists.txt").is_file() {
        return Err(format!(
            "Vortex LLVM source directory '{}' does not contain llvm/CMakeLists.txt; clone may have failed or layout changed",
            source_config.llvm_source_dir.display()
        ).into());
    }

    fs::create_dir_all(&source_config.llvm_build_dir).map_err(|error| {
        format!(
            "failed to create LLVM build directory '{}': {error}",
            source_config.llvm_build_dir.display()
        )
    })?;
    fs::create_dir_all(&source_config.tool_dir).map_err(|error| {
        format!(
            "failed to create source tool directory '{}': {error}",
            source_config.tool_dir.display()
        )
    })?;

    println!(
        "Configuring llvm-vortex with projects {} and targets {}",
        source_config.llvm_projects, source_config.llvm_targets
    );
    let mut configure = Command::new("cmake");
    configure
        .arg("-G")
        .arg("Ninja")
        .arg("-S")
        .arg(&llvm_source)
        .arg("-B")
        .arg(&source_config.llvm_build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg(format!(
            "-DCMAKE_INSTALL_PREFIX={}",
            source_config.llvm_prefix().display()
        ))
        .arg(format!(
            "-DLLVM_ENABLE_PROJECTS={}",
            source_config.llvm_projects
        ))
        .arg(format!(
            "-DLLVM_TARGETS_TO_BUILD={}",
            source_config.llvm_targets
        ))
        .arg("-DBUILD_SHARED_LIBS=ON")
        .arg("-DLLVM_ABI_BREAKING_CHECKS=FORCE_OFF")
        .arg("-DLLVM_INCLUDE_BENCHMARKS=OFF")
        .arg("-DLLVM_INCLUDE_EXAMPLES=OFF")
        .arg("-DLLVM_INCLUDE_TESTS=OFF");
    if source_config.llvm_projects_include("mlir") {
        configure.arg("-DMLIR_INCLUDE_TESTS=OFF");
    }
    run_checked(configure, "llvm-vortex.configure")?;

    println!(
        "Building llvm-vortex with {} job(s). This can take a long time.",
        source_config.jobs
    );
    let mut build = Command::new("cmake");
    build
        .arg("--build")
        .arg(&source_config.llvm_build_dir)
        .arg("--parallel")
        .arg(source_config.jobs.to_string());
    run_checked(build, "llvm-vortex.build")?;

    let mut install = Command::new("cmake");
    install.arg("--install").arg(&source_config.llvm_build_dir);
    run_checked(install, "llvm-vortex.install")
}

pub(super) fn verify_vortex_llvm_install(
    source_config: &VortexSourceToolchainConfig,
) -> Result<()> {
    let clang = source_config.llvm_bin_dir().join("clang");
    if !clang.is_file() {
        return Err(format!(
            "llvm-vortex install did not produce clang at '{}'",
            clang.display()
        )
        .into());
    }

    let mut version = Command::new(&clang);
    version.arg("--version");
    apply_source_llvm_env(&mut version, source_config)?;
    run_checked(version, "llvm-vortex.clang-version")?;
    verify_vortex_llvm_features(source_config)?;
    verify_vortex_mlir_tools_if_requested(source_config)
}

pub(super) fn verify_vortex_mlir_tools_if_requested(
    source_config: &VortexSourceToolchainConfig,
) -> Result<()> {
    if !source_config.llvm_projects_include("mlir") {
        return Ok(());
    }

    for tool in ["mlir-opt", "mlir-translate"] {
        let path = source_config.llvm_bin_dir().join(tool);
        if !path.is_file() {
            return Err(format!(
                "llvm-vortex install enabled MLIR but did not produce {tool} at '{}'; check MANDREL_VORTEX_LLVM_PROJECTS or the LLVM build log",
                path.display()
            ).into());
        }

        let mut version = Command::new(&path);
        version.arg("--version");
        apply_source_llvm_env(&mut version, source_config)?;
        run_checked(version, &format!("llvm-vortex.{tool}-version"))?;
    }

    Ok(())
}

pub(super) fn verify_vortex_llvm_features(
    source_config: &VortexSourceToolchainConfig,
) -> Result<()> {
    fs::create_dir_all(&source_config.compiler_rt_build_dir).map_err(|error| {
        format!(
            "failed to create compiler-rt/probe build directory '{}': {error}",
            source_config.compiler_rt_build_dir.display()
        )
    })?;
    let probe_source = source_config
        .compiler_rt_build_dir
        .join("mandrel-vortex-feature-probe.c");
    let probe_object = source_config
        .compiler_rt_build_dir
        .join("mandrel-vortex-feature-probe.o");
    fs::write(
        &probe_source,
        "void mandrel_vortex_feature_probe(void) {}\n",
    )
    .map_err(|error| {
        format!(
            "failed to write Vortex LLVM feature probe '{}': {error}",
            probe_source.display()
        )
    })?;

    let mut command = Command::new(source_config.llvm_bin_dir().join("clang"));
    command
        .arg("--target=riscv64-unknown-elf")
        .arg(format!(
            "--sysroot={}",
            source_config.riscv_sysroot_dir().display()
        ))
        .arg(format!(
            "--gcc-toolchain={}",
            source_config.riscv_gcc_toolchain_dir().display()
        ))
        .args(["-march=rv64imafd", "-mabi=lp64d"])
        .args(["-Xclang", "-target-feature", "-Xclang", "+xvortex"])
        .args(["-Xclang", "-target-feature", "-Xclang", "+zicond"])
        .args(["-mcmodel=medany", "-c"])
        .arg(&probe_source)
        .arg("-o")
        .arg(&probe_object);
    apply_source_llvm_env(&mut command, source_config)?;
    let output = run_output_checked(command, "llvm-vortex.feature-probe")?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if stderr.contains("not a recognized feature") || stderr.contains("is not recognized") {
        return Err(format!(
            "llvm-vortex feature probe rejected +xvortex/+zicond; this looks like system/upstream LLVM, not Vortex-patched LLVM. stderr: {stderr}"
        ).into());
    }
    Ok(())
}

pub(super) fn build_vortex_compiler_rt64(
    source_config: &VortexSourceToolchainConfig,
    vortex_config: &VortexConfig,
) -> Result<()> {
    let compiler_rt_source = source_config.llvm_source_dir.join("compiler-rt");
    if !compiler_rt_source.join("CMakeLists.txt").is_file() {
        return Err(format!(
            "Vortex LLVM source directory '{}' does not contain compiler-rt/CMakeLists.txt",
            source_config.llvm_source_dir.display()
        )
        .into());
    }

    let vortex_kernel_archive = vortex_config.build_dir.join("sw/kernel/libvortex.a");
    if !vortex_kernel_archive.is_file() {
        return Err(format!(
            "compiler-rt requires Vortex kernel runtime archive '{}'; build sw/kernel first",
            vortex_kernel_archive.display()
        )
        .into());
    }
    let link_script = vortex_config.source_dir.join("sw/kernel/scripts/link64.ld");
    if !link_script.is_file() {
        return Err(format!(
            "compiler-rt requires Vortex link script '{}'",
            link_script.display()
        )
        .into());
    }

    reset_cmake_build_dir_if_cached(
        &source_config.compiler_rt_build_dir,
        "compiler-rt riscv64 builtins",
    )?;

    let llvm_bin = source_config.llvm_bin_dir();
    let lld = require_vortex_lld(source_config)?;
    let c_flags = format!(
        "--gcc-toolchain={} -march=rv64imafd -mabi=lp64d -Xclang -target-feature -Xclang +xvortex -Xclang -target-feature -Xclang +zicond -mcmodel=medany -fno-rtti -fno-exceptions -fdata-sections -ffunction-sections",
        source_config.riscv_gcc_toolchain_dir().display()
    );
    let asm_flags = format!("--target=riscv64-unknown-elf {c_flags}");
    let linker_flags = format!(
        "-fuse-ld=lld -nostartfiles -Wl,-Bstatic,--gc-sections,-T,{},--defsym=STARTUP_ADDR=0x80000000 {}",
        link_script.display(),
        vortex_kernel_archive.display()
    );

    println!("Configuring Vortex compiler-rt builtins for riscv64-unknown-elf.");
    let mut configure = Command::new("cmake");
    configure
        .arg("-G")
        .arg("Ninja")
        .arg("-S")
        .arg(&compiler_rt_source)
        .arg("-B")
        .arg(&source_config.compiler_rt_build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg(format!(
            "-DCMAKE_INSTALL_PREFIX={}",
            source_config.compiler_rt_install_dir().display()
        ))
        .arg(format!("-DCMAKE_AR={}", llvm_bin.join("llvm-ar").display()))
        .arg(format!("-DCMAKE_LINKER={}", lld.display()))
        .arg(format!("-DCMAKE_NM={}", llvm_bin.join("llvm-nm").display()))
        .arg(format!(
            "-DCMAKE_RANLIB={}",
            llvm_bin.join("llvm-ranlib").display()
        ))
        .arg(format!(
            "-DCMAKE_C_COMPILER={}",
            llvm_bin.join("clang").display()
        ))
        .arg("-DCMAKE_C_COMPILER_TARGET=riscv64-unknown-elf")
        .arg(format!("-DCMAKE_C_FLAGS={c_flags}"))
        .arg(format!(
            "-DCMAKE_ASM_COMPILER={}",
            llvm_bin.join("clang").display()
        ))
        .arg("-DCMAKE_ASM_COMPILER_TARGET=riscv64-unknown-elf")
        .arg(format!("-DCMAKE_ASM_FLAGS={asm_flags}"))
        .arg(format!("-DCMAKE_EXE_LINKER_FLAGS={linker_flags}"))
        .arg(format!(
            "-DCMAKE_SYSROOT={}",
            source_config.riscv_sysroot_dir().display()
        ))
        .arg("-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY")
        .arg("-DCOMPILER_RT_OS_DIR=baremetal")
        .arg("-DCOMPILER_RT_DEFAULT_TARGET_TRIPLE=riscv64-unknown-elf")
        .arg("-DCOMPILER_RT_BUILD_BUILTINS=ON")
        .arg("-DCOMPILER_RT_BUILD_CRT=OFF")
        .arg("-DCOMPILER_RT_BUILD_CTX_PROFILE=OFF")
        .arg("-DCOMPILER_RT_BUILD_GWP_ASAN=OFF")
        .arg("-DCOMPILER_RT_BUILD_LIBFUZZER=OFF")
        .arg("-DCOMPILER_RT_BUILD_MEMPROF=OFF")
        .arg("-DCOMPILER_RT_BUILD_ORC=OFF")
        .arg("-DCOMPILER_RT_BUILD_PROFILE=OFF")
        .arg("-DCOMPILER_RT_BUILD_SANITIZERS=OFF")
        .arg("-DCOMPILER_RT_BUILD_SCUDO_STANDALONE_WITH_LLVM_LIBC=OFF")
        .arg("-DCOMPILER_RT_BUILD_STANDALONE_LIBATOMIC=OFF")
        .arg("-DCOMPILER_RT_BUILD_XRAY=OFF")
        .arg("-DCOMPILER_RT_BUILD_XRAY_NO_PREINIT=OFF")
        .arg("-DCOMPILER_RT_BAREMETAL_BUILD=ON")
        .arg("-DCOMPILER_RT_INCLUDE_TESTS=OFF");
    apply_source_llvm_env(&mut configure, source_config)?;
    run_checked(configure, "compiler-rt.configure")?;

    println!(
        "Building Vortex compiler-rt with {} job(s).",
        source_config.jobs
    );
    let mut build = Command::new("cmake");
    build
        .arg("--build")
        .arg(&source_config.compiler_rt_build_dir)
        .arg("--parallel")
        .arg(source_config.jobs.to_string());
    apply_source_llvm_env(&mut build, source_config)?;
    run_checked(build, "compiler-rt.build")?;

    let mut install = Command::new("cmake");
    install
        .arg("--install")
        .arg(&source_config.compiler_rt_build_dir);
    apply_source_llvm_env(&mut install, source_config)?;
    run_checked(install, "compiler-rt.install")
}

pub(super) fn reset_cmake_build_dir_if_cached(build_dir: &Path, description: &str) -> Result<()> {
    if build_dir.join("CMakeCache.txt").is_file() {
        println!(
            "Removing stale {description} CMake build directory '{}'.",
            build_dir.display()
        );
        fs::remove_dir_all(build_dir).map_err(|error| {
            format!(
                "failed to remove stale {description} CMake build directory '{}': {error}",
                build_dir.display()
            )
        })?;
    }

    fs::create_dir_all(build_dir).map_err(|error| {
        XtaskError::message(format!(
            "failed to create {description} build directory '{}': {error}",
            build_dir.display()
        ))
    })
}

pub(super) fn require_vortex_lld(source_config: &VortexSourceToolchainConfig) -> Result<PathBuf> {
    for program in ["ld.lld", "llvm-lld", "lld"] {
        let candidate = source_config.llvm_bin_dir().join(program);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(format!(
        "llvm-vortex install under '{}' did not provide ld.lld/llvm-lld/lld",
        source_config.llvm_bin_dir().display()
    )
    .into())
}

pub(super) fn verify_vortex_compiler_rt_install(
    source_config: &VortexSourceToolchainConfig,
) -> Result<()> {
    let archive = source_config.compiler_rt_builtins_archive();
    if archive.is_file() {
        println!("Vortex compiler-rt builtins ready: {}", archive.display());
        Ok(())
    } else {
        Err(format!(
            "compiler-rt install completed but expected builtins archive was not found: {}",
            archive.display()
        )
        .into())
    }
}

pub(super) fn apply_source_llvm_env(
    command: &mut Command,
    source_config: &VortexSourceToolchainConfig,
) -> Result<()> {
    prepend_env_path(command, "PATH", source_config.llvm_bin_dir())?;
    prepend_env_path(command, "LD_LIBRARY_PATH", source_config.llvm_lib_dir())?;
    Ok(())
}

pub(super) fn print_source_toolchain_success(
    source_config: &VortexSourceToolchainConfig,
    vortex_config: &VortexConfig,
) {
    println!("Vortex source toolchain completed.");
    println!("tools:       {}", source_config.tool_dir.display());
    println!("llvm:        {}", source_config.llvm_prefix().display());
    println!(
        "compiler-rt: {}",
        source_config.compiler_rt_builtins_archive().display()
    );
    println!(
        "next:        MANDREL_VORTEX_TOOLCHAIN_MODE=skip MANDREL_VORTEX_TOOLDIR={} cargo vortex-install",
        source_config.tool_dir.display()
    );
    print_env_usage(vortex_config);
}
