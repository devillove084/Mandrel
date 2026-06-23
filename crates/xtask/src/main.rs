use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output};

use mandrel_compiler::{
    CompileError, VortexAttentionPrefillPlan, compile_vortex_attention_prefill_kernel,
};
use mandrel_model_ir::AttentionOp;
use mandrel_vortex_backend::{
    AttentionPrefillI8Run, DEFAULT_VORTEX_SYSTEM_TOOLDIR, VortexBackend, VortexBackendConfig,
    VortexBackendError, VortexCodegenError, VortexCommandRunner, VortexConfig,
    VortexMlirKernelArtifacts, VortexMlirKernelBuildRequest, VortexStatus, VortexToolchainError,
    VortexToolchainMode, VortexToolchainResult, build_vortex_mlir_kernel_artifacts,
    generate_vortex_attention_prefill_mlir, reference_attention_prefill_i8,
};
use snafu::Snafu;
use tracing::{info, warn};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::fmt::MakeWriter;

const GITHUB_REWRITE_PATTERNS: [&str; 4] = [
    "https://github.com/",
    "http://github.com/",
    "git@github.com:",
    "ssh://git@github.com/",
];

const DEFAULT_VORTEX_SOURCE_TOOLDIR: &str = "external/vortex-source-tools";
const DEFAULT_VORTEX_LLVM_DIR: &str = "external/llvm-vortex";
const DEFAULT_VORTEX_LLVM_BUILD_DIR: &str = "external/llvm-vortex-build";
const DEFAULT_VORTEX_COMPILER_RT_BUILD_DIR: &str = "external/llvm-vortex-compiler-rt-build64";
const DEFAULT_VORTEX_LLVM_URL: &str = "https://github.com/devillove084/llvm.git";
const DEFAULT_VORTEX_LLVM_REF: &str = "vortex_3.x";
const DEFAULT_VORTEX_LLVM_PROJECTS: &str = "clang;lld;mlir";
const LOG_COMMAND_MAX_CHARS: usize = 4096;

type Result<T> = std::result::Result<T, XtaskError>;

#[derive(Debug, Snafu)]
enum XtaskError {
    #[snafu(display("{message}"))]
    Message { message: String },
    #[snafu(display("failed to spawn {phase}: {source}"))]
    CommandSpawn { phase: String, source: io::Error },
    #[snafu(display("{phase} failed with status: {status}; command: {command}"))]
    CommandFailed {
        phase: String,
        status: ExitStatus,
        command: String,
    },
    #[snafu(display("{phase} failed with status: {status}; command: {command}; stderr: {stderr}"))]
    CommandFailedWithStderr {
        phase: String,
        status: ExitStatus,
        command: String,
        stderr: String,
    },
    #[snafu(display("Vortex toolchain error: {source}"))]
    VortexToolchain { source: VortexToolchainError },
    #[snafu(display("Vortex backend error: {source}"))]
    VortexBackend { source: VortexBackendError },
    #[snafu(display("Vortex codegen error: {source}"))]
    VortexCodegen { source: VortexCodegenError },
    #[snafu(display("compile error: {source}"))]
    Compile { source: CompileError },
}

impl XtaskError {
    fn message(message: impl Into<String>) -> Self {
        Self::Message {
            message: message.into(),
        }
    }
}

impl From<String> for XtaskError {
    fn from(message: String) -> Self {
        Self::message(message)
    }
}

impl From<&str> for XtaskError {
    fn from(message: &str) -> Self {
        Self::message(message)
    }
}

impl From<VortexToolchainError> for XtaskError {
    fn from(source: VortexToolchainError) -> Self {
        Self::VortexToolchain { source }
    }
}

impl From<VortexBackendError> for XtaskError {
    fn from(source: VortexBackendError) -> Self {
        Self::VortexBackend { source }
    }
}

impl From<VortexCodegenError> for XtaskError {
    fn from(source: VortexCodegenError) -> Self {
        Self::VortexCodegen { source }
    }
}

impl From<CompileError> for XtaskError {
    fn from(source: CompileError) -> Self {
        Self::Compile { source }
    }
}

fn main() {
    let _log_guard = match init_logging() {
        Ok(guard) => guard,
        Err(error) => {
            eprintln!("failed to initialize logging: {error}");
            std::process::exit(1);
        }
    };

    if let Err(error) = run() {
        tracing::error!(%error, "xtask failed");
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let command = env::args().nth(1).unwrap_or_else(|| "help".to_owned());
    let workspace_root = workspace_root()?;
    info!(command, root = %workspace_root.display(), "starting xtask command");

    match command.as_str() {
        "help" => print_help(),
        "vortex-fetch" => fetch_vortex(&workspace_root)?,
        "vortex-system-tools" => prepare_and_print_vortex_system_tools(&workspace_root)?,
        "vortex-toolchain-source" => install_vortex_source_toolchain(&workspace_root)?,
        "vortex-install" => install_vortex(&workspace_root)?,
        "vortex-env" => write_and_print_vortex_env(&workspace_root)?,
        "vortex-status" => print_vortex_status(&workspace_root)?,
        "vortex-plan-attention" => print_attention_prefill_plan()?,
        "vortex-generate-attention" => generate_vortex_attention_kernel_source(&workspace_root)?,
        "vortex-run-attention" => run_vortex_attention_correctness(&workspace_root)?,
        "__vortex-run-attention-inner" => run_vortex_attention_correctness_inner(&workspace_root)?,
        "vortex-run-vecadd" => run_vortex_vecadd(&workspace_root)?,
        other => {
            eprintln!("unknown xtask command: {other}");
            print_help();
            return Err(XtaskError::message("xtask failed"));
        }
    }

    Ok(())
}

fn workspace_root() -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let crates_dir = manifest_dir.parent().ok_or_else(|| {
        format!(
            "xtask manifest dir has no parent: {}",
            manifest_dir.display()
        )
    })?;
    crates_dir
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| format!("crates dir has no parent: {}", crates_dir.display()).into())
}

#[derive(Clone, Copy)]
enum LogFormat {
    Compact,
    Pretty,
    Json,
}

impl LogFormat {
    fn from_env() -> Result<Self> {
        let raw = env::var("MANDREL_LOG_FORMAT").unwrap_or_else(|_| "compact".to_owned());
        match raw.as_str() {
            "compact" => Ok(Self::Compact),
            "pretty" => Ok(Self::Pretty),
            "json" => Ok(Self::Json),
            other => Err(XtaskError::message(format!(
                "unsupported MANDREL_LOG_FORMAT '{other}'; use compact, pretty, or json"
            ))),
        }
    }
}

fn init_logging() -> Result<Option<WorkerGuard>> {
    let filter = env::var("MANDREL_LOG").unwrap_or_else(|_| "info".to_owned());
    let filter =
        EnvFilter::try_new(filter).map_err(|error| format!("invalid MANDREL_LOG: {error}"))?;
    let format = LogFormat::from_env()?;

    if let Some(file_path) = env::var_os("MANDREL_LOG_FILE").map(PathBuf::from) {
        let parent = non_empty_parent(&file_path);
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create log directory '{}': {error}",
                parent.display()
            )
        })?;
        let file_name = file_path.file_name().ok_or_else(|| {
            format!(
                "MANDREL_LOG_FILE must point to a file, got '{}'",
                file_path.display()
            )
        })?;
        let file_appender = tracing_appender::rolling::never(parent, Path::new(file_name));
        let (writer, guard) = tracing_appender::non_blocking(file_appender);
        install_subscriber(filter, format, writer, false)?;
        return Ok(Some(guard));
    }

    install_subscriber(filter, format, io::stdout, true)?;
    Ok(None)
}

fn non_empty_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn install_subscriber<W>(filter: EnvFilter, format: LogFormat, writer: W, ansi: bool) -> Result<()>
where
    W: for<'writer> MakeWriter<'writer> + Send + Sync + 'static,
{
    match format {
        LogFormat::Compact => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(writer)
            .with_ansi(ansi)
            .compact()
            .try_init(),
        LogFormat::Pretty => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(writer)
            .with_ansi(ansi)
            .pretty()
            .try_init(),
        LogFormat::Json => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(writer)
            .with_ansi(ansi)
            .json()
            .try_init(),
    }
    .map_err(|error| XtaskError::message(format!("failed to install tracing subscriber: {error}")))
}

fn print_help() {
    println!(
        "xtask commands:\n\
           cargo xtask vortex-fetch        # clone/update Vortex source only under external/vortex\n\
           cargo xtask vortex-system-tools # create Vortex-compatible wrappers for Ubuntu/system packages\n\
           cargo xtask vortex-toolchain-source # source-build llvm-vortex + compiler-rt for this host\n\
           cargo xtask vortex-install      # clone, configure, prepare toolchain, build, install, write env\n\
           cargo xtask vortex-env          # write/print external/vortex-env.sh for manual shells\n\
           cargo xtask vortex-status       # show checkout/build/install/env status\n\
           cargo xtask vortex-plan-attention # print dense attention-prefill online-softmax plan\n\
           cargo xtask vortex-generate-attention # generate/validate attention MLIR through Vortex LLVM\n\
           cargo xtask vortex-run-attention # run generated attention vxbin through Vortex simx and compare output\n\
           cargo xtask vortex-run-vecadd   # run Vortex official vecadd through ci/blackbox.sh when available\n\n\
         Main backend target:\n\
           Vortex RISC-V GPGPU, route B custom backend via mandrel runtime/kernel IR\n\n\
         Local install defaults:\n\
           source: external/vortex\n\
           build:  external/vortex-build\n\
           tools:  external/vortex-source-tools by default; external/vortex-system-tools for system mode\n\
           env:    external/vortex-env.sh\n\n\
         Vortex toolchain mode:\n\
           MANDREL_VORTEX_TOOLCHAIN_MODE=auto|prebuilt|system|skip\n\
           default is host-dependent: auto on x86_64, system on non-x86_64\n\
           explicit auto uses upstream prebuilt packages only on x86_64\n\
           system maps Ubuntu/system packages into Vortex's expected local layout\n\
           skip assumes MANDREL_VORTEX_TOOLDIR is already populated\n\
           MANDREL_VORTEX_BUILD_PROFILE=full|simx|software|none controls build scope\n\
           Device codegen is MLIR-only; xtask lowers kernel.mlir with mlir-translate before clang\n\
           MANDREL_ALLOW_LIBGCC_BUILTINS=1 enables explicit experimental libgcc fallback\n\
           Source LLVM build knobs: MANDREL_VORTEX_LLVM_DIR, MANDREL_VORTEX_LLVM_BUILD_DIR,\n\
            MANDREL_VORTEX_COMPILER_RT_BUILD_DIR, MANDREL_VORTEX_LLVM_URL,\n\
            MANDREL_VORTEX_LLVM_REF, MANDREL_VORTEX_LLVM_PROJECTS,\n\
            MANDREL_VORTEX_LLVM_TARGETS, MANDREL_VORTEX_TOOLCHAIN_JOBS\n\n\
         Slow GitHub downloads in Vortex toolchain scripts:\n\
           MANDREL_GITHUB_PROXY_PREFIX=https://gh-proxy.org/ cargo vortex-install\n\
           MANDREL_FETCH_RETRIES=5 cargo vortex-install\n\n\
         Logging:\n\
           MANDREL_LOG=info|debug|trace or tracing filter syntax; default: info\n\
           MANDREL_LOG_FORMAT=compact|pretty|json; default: compact\n\
           MANDREL_LOG_FILE=logs/xtask.log writes logs to a file instead of console"
    );
}

fn fetch_vortex(workspace_root: &Path) -> Result<()> {
    let config = VortexConfig::from_env(workspace_root)?;
    log_vortex_config(&config, "fetching/verifying Vortex source");
    clone_vortex_if_needed(&config)?;
    checkout_vortex_ref_if_requested(&config)?;
    println!("Vortex checkout ready at: {}", config.source_dir.display());
    Ok(())
}

fn prepare_and_print_vortex_system_tools(workspace_root: &Path) -> Result<()> {
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

#[derive(Debug, Clone)]
struct VortexSourceToolchainConfig {
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
            .split(|ch| ch == ';' || ch == ',')
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

fn install_vortex_source_toolchain(workspace_root: &Path) -> Result<()> {
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

fn log_vortex_source_toolchain_config(config: &VortexSourceToolchainConfig) {
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

fn default_vortex_llvm_targets_for_host() -> String {
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

fn source_toolchain_jobs() -> Result<usize> {
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

fn require_source_toolchain_programs() -> Result<()> {
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

fn prepare_vortex_source_toolchain_riscv_layout(config: &VortexConfig) -> Result<()> {
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

fn link_riscv_gcc_support_dir(tool_dir: &Path) -> Result<()> {
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

fn configure_vortex_for_source_toolchain(config: &VortexConfig) -> Result<()> {
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

fn build_vortex_kernel_runtime_archive(config: &VortexConfig) -> Result<()> {
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

fn clone_or_update_vortex_llvm_source(
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

fn update_vortex_llvm_submodules(
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

fn build_vortex_llvm(source_config: &VortexSourceToolchainConfig) -> Result<()> {
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

fn verify_vortex_llvm_install(source_config: &VortexSourceToolchainConfig) -> Result<()> {
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

fn verify_vortex_mlir_tools_if_requested(
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

fn verify_vortex_llvm_features(source_config: &VortexSourceToolchainConfig) -> Result<()> {
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

fn build_vortex_compiler_rt64(
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

fn reset_cmake_build_dir_if_cached(build_dir: &Path, description: &str) -> Result<()> {
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

fn require_vortex_lld(source_config: &VortexSourceToolchainConfig) -> Result<PathBuf> {
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

fn verify_vortex_compiler_rt_install(source_config: &VortexSourceToolchainConfig) -> Result<()> {
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

fn apply_source_llvm_env(
    command: &mut Command,
    source_config: &VortexSourceToolchainConfig,
) -> Result<()> {
    prepend_env_path(command, "PATH", source_config.llvm_bin_dir())?;
    prepend_env_path(command, "LD_LIBRARY_PATH", source_config.llvm_lib_dir())?;
    Ok(())
}

fn print_source_toolchain_success(
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

fn install_vortex(workspace_root: &Path) -> Result<()> {
    let config = VortexConfig::from_env(workspace_root)?;
    log_vortex_config(
        &config,
        "installing Vortex into project-local external directory",
    );

    clone_vortex_if_needed(&config)?;
    checkout_vortex_ref_if_requested(&config)?;
    let run_prebuilt_toolchain = config.should_run_prebuilt_toolchain()?;
    if config.toolchain_mode == VortexToolchainMode::System {
        prepare_vortex_system_tools(&config)?;
    } else if config.toolchain_mode == VortexToolchainMode::Skip {
        reject_obvious_incompatible_prebuilt_tools(&config)?;
    }
    configure_and_build_vortex(&config, run_prebuilt_toolchain)?;
    write_vortex_env_script(&config)?;
    print_install_success(&config);

    Ok(())
}

fn log_vortex_config(config: &VortexConfig, message: &str) {
    info!(
        source_dir = %config.source_dir.display(),
        build_dir = %config.build_dir.display(),
        install_dir = %config.install_dir().display(),
        tool_dir = %config.tool_dir.display(),
        env_file = %config.env_file.display(),
        url = %config.url,
        clone_url = %config.clone_url(),
        download_proxy_prefix = ?config.download_proxy_prefix,
        git_proxy_prefix = ?config.git_proxy_prefix,
        fetch_retries = config.fetch_retries,
        reference = ?config.reference,
        xlen = config.xlen,
        toolchain_mode = config.toolchain_mode.as_str(),
        message
    );
}

fn clone_vortex_if_needed(config: &VortexConfig) -> Result<()> {
    if config.source_dir.join(".git").is_dir() {
        info!(source_dir = %config.source_dir.display(), "Vortex checkout already exists");
        configure_git_proxy_for_checkout(config)?;
        update_submodules(config)?;
        return Ok(());
    }

    if config.source_dir.exists() {
        return Err(format!(
            "Vortex directory '{}' exists but is not a git checkout; set MANDREL_VORTEX_DIR or remove it",
            config.source_dir.display()
        ).into());
    }

    if let Some(parent) = config
        .source_dir
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create Vortex parent directory '{}': {error}",
                parent.display()
            )
        })?;
    }

    run_checked_with_retries(
        || {
            let mut command = Command::new("git");
            command
                .arg("clone")
                .arg("--depth=1")
                .arg(config.clone_url())
                .arg(&config.source_dir);
            apply_github_proxy_git_env(&mut command, config);
            Ok(command)
        },
        "vortex.clone",
        config.fetch_retries,
    )?;
    configure_git_proxy_for_checkout(config)?;
    update_submodules(config)
}

fn update_submodules(config: &VortexConfig) -> Result<()> {
    run_checked_with_retries(
        || {
            let mut command = Command::new("git");
            command
                .arg("--no-pager")
                .arg("-C")
                .arg(&config.source_dir)
                .args(["submodule", "update", "--init", "--recursive"]);
            apply_github_proxy_git_env(&mut command, config);
            Ok(command)
        },
        "vortex.submodule-update",
        config.fetch_retries,
    )
}

fn configure_git_proxy_for_checkout(config: &VortexConfig) -> Result<()> {
    let Some(base) = config.git_proxy_base() else {
        return Ok(());
    };

    let key = format!("url.{base}.insteadOf");
    for pattern in GITHUB_REWRITE_PATTERNS {
        if git_config_contains_value(&config.source_dir, &key, pattern)? {
            continue;
        }

        let mut command = Command::new("git");
        command
            .arg("--no-pager")
            .arg("-C")
            .arg(&config.source_dir)
            .args(["config", "--local", "--add"])
            .arg(&key)
            .arg(pattern);
        run_checked(command, "vortex.git-proxy-config")?;
    }

    Ok(())
}

fn git_config_contains_value(repo_dir: &Path, key: &str, value: &str) -> Result<bool> {
    let output = Command::new("git")
        .arg("--no-pager")
        .arg("-C")
        .arg(repo_dir)
        .args(["config", "--local", "--get-all"])
        .arg(key)
        .output()
        .map_err(|error| format!("failed to inspect git config '{key}': {error}"))?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Ok(stdout.lines().any(|line| line.trim() == value));
    }

    if output.status.code() == Some(1) {
        return Ok(false);
    }

    Err(format!(
        "failed to inspect git config '{key}' with status {}; stderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    )
    .into())
}

fn checkout_vortex_ref_if_requested(config: &VortexConfig) -> Result<()> {
    let Some(reference) = &config.reference else {
        return Ok(());
    };

    run_checked_with_retries(
        || {
            let mut fetch = Command::new("git");
            fetch
                .arg("--no-pager")
                .arg("-C")
                .arg(&config.source_dir)
                .args(["fetch", "--tags", "origin"]);
            apply_github_proxy_git_env(&mut fetch, config);
            Ok(fetch)
        },
        "vortex.fetch",
        config.fetch_retries,
    )?;

    let mut checkout = Command::new("git");
    checkout
        .arg("--no-pager")
        .arg("-C")
        .arg(&config.source_dir)
        .arg("checkout")
        .arg(reference);
    apply_github_proxy_git_env(&mut checkout, config);
    run_checked(checkout, "vortex.checkout")?;
    update_submodules(config)
}

fn configure_and_build_vortex(config: &VortexConfig, run_prebuilt_toolchain: bool) -> Result<()> {
    fs::create_dir_all(&config.build_dir).map_err(|error| {
        format!(
            "failed to create Vortex build directory '{}': {error}",
            config.build_dir.display()
        )
    })?;
    fs::create_dir_all(&config.tool_dir).map_err(|error| {
        format!(
            "failed to create Vortex tool directory '{}': {error}",
            config.tool_dir.display()
        )
    })?;

    println!(
        "Configuring Vortex build directory: {}",
        config.build_dir.display()
    );
    run_checked(config.configure_command(), "vortex.configure")?;
    if config.toolchain_mode == VortexToolchainMode::System {
        let riscv_runtime = require_riscv_builtins_runtime(config)?;
        prepare_vortex_system_runtime_overrides(config, &riscv_runtime)?;
    }

    let toolchain_script = config.build_dir.join("ci/toolchain_install.sh");
    if !toolchain_script.is_file() {
        return Err(format!(
            "Vortex toolchain script was not generated at '{}'. Configure may have failed or upstream layout changed.",
            toolchain_script.display()
        ).into());
    }

    if run_prebuilt_toolchain {
        println!(
            "Installing Vortex prebuilt toolchain into: {}",
            config.tool_dir.display()
        );
        run_checked_with_retries(
            || {
                let mut toolchain = Command::new(&toolchain_script);
                toolchain.current_dir(&config.build_dir);
                apply_vortex_env(&mut toolchain, config)?;
                Ok(toolchain)
            },
            "vortex.toolchain-install",
            config.fetch_retries,
        )?;
    } else {
        println!(
            "Skipping Vortex prebuilt toolchain install because MANDREL_VORTEX_TOOLCHAIN_MODE={}. Expected tools must already be available under: {}",
            config.toolchain_mode.as_str(),
            config.tool_dir.display()
        );
    }

    let build_profile = VortexBuildProfile::from_env_or_default(config)?;
    build_vortex_with_profile(config, build_profile)?;

    if build_profile.should_install() {
        println!(
            "Installing Vortex SDK sysroot into: {}",
            config.install_dir().display()
        );
        let mut make_install = Command::new("make");
        make_install.current_dir(&config.build_dir).arg("install");
        apply_vortex_env(&mut make_install, config)?;
        run_checked(make_install, "vortex.make-install")?;
    } else {
        println!(
            "Skipping Vortex SDK install because MANDREL_VORTEX_BUILD_PROFILE={}",
            build_profile.as_str()
        );
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VortexBuildProfile {
    Full,
    Simx,
    Software,
    None,
}

impl VortexBuildProfile {
    fn from_env_or_default(config: &VortexConfig) -> Result<Self> {
        if let Some(raw) = non_empty_env("MANDREL_VORTEX_BUILD_PROFILE") {
            return match raw.as_str() {
                "full" => Ok(Self::Full),
                "simx" | "software-simx" | "sdk-simx" => Ok(Self::Simx),
                "software" | "minimal" => Ok(Self::Software),
                "none" | "configure" => Ok(Self::None),
                other => Err(format!(
                    "unsupported MANDREL_VORTEX_BUILD_PROFILE '{other}'; use full, simx, software, or none"
                ).into()),
            };
        }

        match config.toolchain_mode {
            VortexToolchainMode::System => Ok(Self::Simx),
            VortexToolchainMode::Skip if env::consts::ARCH != "x86_64" => Ok(Self::Simx),
            _ => Ok(Self::Full),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Simx => "simx",
            Self::Software => "software",
            Self::None => "none",
        }
    }

    const fn should_install(self) -> bool {
        match self {
            Self::Full | Self::Simx | Self::Software => true,
            Self::None => false,
        }
    }
}

fn build_vortex_with_profile(
    config: &VortexConfig,
    build_profile: VortexBuildProfile,
) -> Result<()> {
    match build_profile {
        VortexBuildProfile::Full => {
            println!("Building full Vortex tree. This can take a while.");
            let mut make = Command::new("make");
            make.current_dir(&config.build_dir).arg("-s");
            apply_vortex_env(&mut make, config)?;
            run_checked(make, "vortex.make")
        }
        VortexBuildProfile::Simx => {
            println!(
                "Building Vortex software SDK and simx functional simulator. Set MANDREL_VORTEX_BUILD_PROFILE=full for RTL/full-tree builds."
            );
            build_vortex_software_sdk(config)?;
            build_vortex_simx(config)
        }
        VortexBuildProfile::Software => {
            println!(
                "Building Vortex software SDK only: sw/kernel and sw/runtime/stub. Set MANDREL_VORTEX_BUILD_PROFILE=simx for the functional simulator or full for RTL/full-tree builds."
            );
            build_vortex_software_sdk(config)
        }
        VortexBuildProfile::None => {
            println!("Skipping Vortex build because MANDREL_VORTEX_BUILD_PROFILE=none");
            Ok(())
        }
    }
}

fn build_vortex_software_sdk(config: &VortexConfig) -> Result<()> {
    for subdir in ["sw/kernel", "sw/runtime/stub"] {
        let mut make = Command::new("make");
        make.current_dir(&config.build_dir).arg("-C").arg(subdir);
        apply_vortex_env(&mut make, config)?;
        run_checked(make, &format!("vortex.make.{subdir}"))?;
    }
    Ok(())
}

fn build_vortex_simx(config: &VortexConfig) -> Result<()> {
    let mut third_party = Command::new("make");
    third_party
        .current_dir(config.source_dir.join("third_party"))
        .args(["softfloat", "ramulator"]);
    apply_vortex_env(&mut third_party, config)?;
    run_checked(third_party, "vortex.make.third_party.simx-deps")?;

    let mut simx = Command::new("make");
    simx.current_dir(&config.build_dir)
        .arg("-C")
        .arg("sim/simx");
    apply_vortex_env(&mut simx, config)?;
    run_checked(simx, "vortex.make.simx")
}

fn prepare_vortex_system_tools(config: &VortexConfig) -> Result<()> {
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

fn reject_obvious_incompatible_prebuilt_tools(config: &VortexConfig) -> Result<()> {
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

fn inspect_file_type(path: &Path) -> Result<Option<String>> {
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

fn write_riscv_gcc_wrapper(
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

fn replace_file_content(path: &Path, content: &[u8], description: &str) -> Result<()> {
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

fn require_riscv_c_library_include_dir() -> Result<PathBuf> {
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

fn require_riscv_c_library_lib_dir(include_dir: &Path) -> Result<PathBuf> {
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

enum RiscvBuiltinsRuntime {
    CompilerRt { path: PathBuf },
    Libgcc { archive: PathBuf },
}

fn require_riscv_builtins_runtime(config: &VortexConfig) -> Result<RiscvBuiltinsRuntime> {
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

fn allow_libgcc_builtins() -> Result<bool> {
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

fn classify_explicit_riscv_builtins_library(
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

fn find_compiler_rt_builtins_library(config: &VortexConfig) -> Option<PathBuf> {
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

fn is_usable_compiler_rt_candidate(path: &Path) -> bool {
    path.is_file() && !is_libgcc_or_link_to_libgcc_archive(path)
}

fn require_riscv_libgcc_library() -> Result<PathBuf> {
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

fn is_libgcc_or_link_to_libgcc_archive(path: &Path) -> bool {
    if is_libgcc_archive(path) {
        return true;
    }

    match fs::read_link(path) {
        Ok(target) => is_libgcc_archive(&target),
        Err(_) => false,
    }
}

fn is_libgcc_archive(path: &Path) -> bool {
    path.file_name().and_then(OsStr::to_str) == Some("libgcc.a")
}

fn prepare_vortex_system_runtime_overrides(
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

fn require_system_programs<const N: usize>(programs: [&str; N]) -> Result<()> {
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

fn require_program(program: &str) -> Result<PathBuf> {
    find_program_on_path(program).ok_or_else(|| {
        XtaskError::message(format!(
            "missing required program '{program}'. Suggested Ubuntu packages: gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf"
        ))
    })
}

fn find_program_on_path(program: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|dir| dir.join(program))
        .find(|candidate| candidate.is_file())
}

fn find_llvm_program(program: &str) -> Option<PathBuf> {
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

fn replace_symlink_or_empty_dir(link: &Path, target: &Path) -> Result<()> {
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

fn replace_symlink(link: &Path, target: &Path) -> Result<()> {
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

fn project_path_from_env(workspace_root: &Path, env_name: &str, default: &str) -> PathBuf {
    let raw = env::var_os(env_name)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default));
    if raw.is_absolute() {
        raw
    } else {
        workspace_root.join(raw)
    }
}

fn non_empty_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn write_and_print_vortex_env(workspace_root: &Path) -> Result<()> {
    let config = VortexConfig::from_env(workspace_root)?;
    write_vortex_env_script(&config)?;
    print_env_usage(&config);
    Ok(())
}

fn write_vortex_env_script(config: &VortexConfig) -> Result<()> {
    if let Some(parent) = config
        .env_file
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create Vortex env directory '{}': {error}",
                parent.display()
            )
        })?;
    }

    ensure_download_wrappers(config)?;
    let network_exports = vortex_network_env_exports(config);
    let path_export_prefix = vortex_path_export_prefix(config);
    let content = format!(
        "# Generated by `cargo vortex-env` / `cargo vortex-install`.\n\
         # Source this file when using Vortex tools manually from a shell.\n\
         export VORTEX_HOME={}\n\
         export VORTEX_BUILD_DIR={}\n\
         export VORTEX_TOOL_DIR={}\n\
         export VORTEX_PATH={}\n\
         export PKG_CONFIG_PATH={}:\"${{PKG_CONFIG_PATH:-}}\"\n\
         export LD_LIBRARY_PATH={}:\"${{LD_LIBRARY_PATH:-}}\"\n\
         export PATH={}:\"${{PATH:-}}\"\n\
         export MANDREL_FETCH_RETRIES={}\n\
         export MANDREL_VORTEX_TOOLCHAIN_MODE={}\n\
         {}",
        shell_quote_lossy(config.source_dir.as_os_str()),
        shell_quote_lossy(config.build_dir.as_os_str()),
        shell_quote_lossy(config.tool_dir.as_os_str()),
        shell_quote_lossy(config.install_dir().as_os_str()),
        shell_quote_lossy(config.pkg_config_dir().as_os_str()),
        shell_quote_lossy(config.lib_dir().as_os_str()),
        path_export_prefix,
        config.fetch_retries,
        shell_quote_lossy(OsStr::new(config.toolchain_mode.as_str())),
        network_exports
    );

    fs::write(&config.env_file, content).map_err(|error| {
        format!(
            "failed to write Vortex env script '{}': {error}",
            config.env_file.display()
        )
    })?;
    println!("Wrote Vortex env script: {}", config.env_file.display());
    Ok(())
}

fn vortex_network_env_exports(config: &VortexConfig) -> String {
    match config.normalized_download_proxy_prefix() {
        Some(prefix) => format!(
            "export MANDREL_GITHUB_PROXY_PREFIX={}\n",
            shell_quote_lossy(OsStr::new(&prefix))
        ),
        None => String::new(),
    }
}

fn vortex_path_export_prefix(config: &VortexConfig) -> String {
    let simx = shell_quote_lossy(config.simx_dir().as_os_str());
    let bin = shell_quote_lossy(config.bin_dir().as_os_str());
    let vortex_bins = format!("{simx}:{bin}");
    if config.download_proxy_prefix.is_some() {
        format!(
            "{}:{}",
            shell_quote_lossy(config.download_wrapper_dir().as_os_str()),
            vortex_bins
        )
    } else {
        vortex_bins
    }
}

fn ensure_download_wrappers(config: &VortexConfig) -> Result<()> {
    if config.download_proxy_prefix.is_none() {
        return Ok(());
    }

    let wrapper_dir = config.download_wrapper_dir();
    fs::create_dir_all(&wrapper_dir).map_err(|error| {
        format!(
            "failed to create Vortex download wrapper directory '{}': {error}",
            wrapper_dir.display()
        )
    })?;

    let mut wrote_any = false;
    for program in ["wget", "curl"] {
        match find_program_on_path_excluding(program, &wrapper_dir) {
            Some(real_program) => {
                write_download_wrapper(&wrapper_dir, program, &real_program)?;
                wrote_any = true;
            }
            None => warn!(
                program,
                "download helper command not found while creating Vortex wrappers"
            ),
        }
    }

    if !wrote_any {
        warn!(
            wrapper_dir = %wrapper_dir.display(),
            "no wget/curl wrappers were generated; Vortex downloads may still be slow or fail"
        );
    }

    Ok(())
}

fn find_program_on_path_excluding(program: &str, excluded_dir: &Path) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    env::split_paths(&path)
        .map(|dir| dir.join(program))
        .find(|candidate| candidate.is_file() && !candidate.starts_with(excluded_dir))
}

fn write_download_wrapper(wrapper_dir: &Path, program: &str, real_program: &Path) -> Result<()> {
    let wrapper_path = wrapper_dir.join(program);
    let content = download_wrapper_script(real_program);
    fs::write(&wrapper_path, content).map_err(|error| {
        format!(
            "failed to write Vortex download wrapper '{}': {error}",
            wrapper_path.display()
        )
    })?;
    make_executable(&wrapper_path)?;
    Ok(())
}

fn download_wrapper_script(real_program: &Path) -> String {
    format!(
        "#!/usr/bin/env bash\n\
         set -euo pipefail\n\
         REAL={}\n\
         PREFIX=\"${{MANDREL_GITHUB_PROXY_PREFIX:-${{PROXY_PREFIX:-}}}}\"\n\
         if [[ -n \"$PREFIX\" && \"$PREFIX\" != */ ]]; then\n\
         \tPREFIX=\"$PREFIX/\"\n\
         fi\n\
         rewrite_arg() {{\n\
         \tlocal arg=\"$1\"\n\
         \tif [[ -n \"$PREFIX\" ]]; then\n\
         \t\tcase \"$arg\" in\n\
         \t\t\thttps://github.com/*|http://github.com/*|https://raw.githubusercontent.com/*|http://raw.githubusercontent.com/*|https://objects.githubusercontent.com/*|http://objects.githubusercontent.com/*|https://github-releases.githubusercontent.com/*|http://github-releases.githubusercontent.com/*)\n\
         \t\t\t\tprintf '%s%s' \"$PREFIX\" \"$arg\"\n\
         \t\t\t\treturn\n\
         \t\t\t\t;;\n\
         \t\tesac\n\
         \tfi\n\
         \tprintf '%s' \"$arg\"\n\
         }}\n\
         args=()\n\
         for arg in \"$@\"; do\n\
         \targs+=(\"$(rewrite_arg \"$arg\")\")\n\
         done\n\
         exec \"$REAL\" \"${{args[@]}}\"\n",
        shell_quote_lossy(real_program.as_os_str())
    )
}

fn make_executable(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(path)
            .map_err(|error| format!("failed to stat '{}': {error}", path.display()))?
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)
            .map_err(|error| format!("failed to chmod '{}': {error}", path.display()))?;
    }
    Ok(())
}

fn print_install_success(config: &VortexConfig) {
    println!("Vortex install completed.");
    println!("source:  {}", config.source_dir.display());
    println!("build:   {}", config.build_dir.display());
    println!("tools:   {}", config.tool_dir.display());
    println!("install: {}", config.install_dir().display());
    print_env_usage(config);
}

fn print_env_usage(config: &VortexConfig) {
    println!("env:     {}", config.env_file.display());
    println!("manual shell usage: source {}", config.env_file.display());
    println!("cargo xtask Vortex commands inject this environment automatically.");
}

fn print_vortex_status(workspace_root: &Path) -> Result<()> {
    let config = VortexConfig::from_env(workspace_root)?;
    let status = VortexStatus::probe(&config);

    println!("Vortex source dir:       {}", config.source_dir.display());
    println!("Vortex build dir:        {}", config.build_dir.display());
    println!(
        "Vortex install dir:      {}",
        config.install_dir().display()
    );
    println!("Vortex tool dir:         {}", config.tool_dir.display());
    println!("Vortex env file:         {}", config.env_file.display());
    println!(
        "download proxy prefix:   {}",
        config.download_proxy_prefix.as_deref().unwrap_or("<unset>")
    );
    println!(
        "git proxy prefix:        {}",
        config.git_proxy_prefix.as_deref().unwrap_or("<unset>")
    );
    println!("fetch_retries:           {}", config.fetch_retries);
    println!(
        "toolchain_mode:          {}",
        config.toolchain_mode.as_str()
    );
    println!("checkout_exists:         {}", status.checkout_exists);
    println!("build_dir_exists:        {}", status.build_dir_exists);
    println!("install_dir_exists:      {}", status.install_dir_exists);
    println!("env_file_exists:         {}", status.env_file_exists);
    println!(
        "download_wrapper_dir:    {}",
        status.download_wrapper_dir_exists
    );
    println!("blackbox_script_exists:  {}", status.blackbox_script_exists);
    println!("simx_binary_exists:     {}", status.simx_binary_exists);
    println!(
        "runtime_pkg_config:      {}",
        status.runtime_pkg_config_exists
    );
    println!(
        "kernel_pkg_config:       {}",
        status.kernel_pkg_config_exists
    );
    println!("can_run_blackbox:        {}", status.can_run_blackbox());
    println!("can_run_simx:            {}", status.can_run_simx());
    println!(
        "can_use_installed_sdk:   {}",
        status.can_use_installed_runtime()
    );
    Ok(())
}

fn print_attention_prefill_plan() -> Result<()> {
    let plan = current_attention_prefill_plan()?;
    print_attention_plan("Vortex attention-prefill plan", &plan);
    Ok(())
}

fn current_attention_prefill_plan() -> Result<VortexAttentionPrefillPlan> {
    Ok(compile_vortex_attention_prefill_kernel(
        AttentionOp::prefill_i8_demo(),
    )?)
}

fn print_attention_plan(title: &str, plan: &VortexAttentionPrefillPlan) {
    println!("{title}");
    println!(
        "shape: sequence={} head_dim={}",
        plan.op.shape.sequence, plan.op.shape.head_dim
    );
    println!(
        "kernel: {} ({:?}, {:?}, {:?})",
        plan.kernel.symbol.as_str(),
        plan.kernel.domain,
        plan.kernel.implementation,
        plan.kernel.availability
    );
    println!(
        "tile: query={} key={} head_dim={}",
        plan.schedule.tile.query, plan.schedule.tile.key, plan.schedule.tile.head_dim
    );
    println!(
        "layout: {:?}, softmax: {:?}",
        plan.schedule.kv_layout, plan.schedule.softmax
    );
    println!(
        "launch: kernel={} grid=({}, {}, {}) block=({}, {}, {}) shared_memory_bytes={}",
        plan.launch.symbol.as_str(),
        plan.launch.grid.x,
        plan.launch.grid.y,
        plan.launch.grid.z,
        plan.launch.block.x,
        plan.launch.block.y,
        plan.launch.block.z,
        plan.launch.shared_memory_bytes
    );
    println!("args:");
    for arg in &plan.launch.args {
        println!("  {}: {:?}", arg.index, arg.value);
    }
    println!("metrics:");
    println!("  logical_macs: {}", plan.metrics.logical_macs);
    println!("  scheduled_macs: {}", plan.metrics.scheduled_macs);
    println!("  kernel_launches: {}", plan.metrics.kernel_launches);
    println!("  workgroup_count: {}", plan.metrics.workgroup_count);
    println!("  thread_count: {}", plan.metrics.thread_count);
    println!("  global_bytes_read: {}", plan.metrics.global_bytes_read);
    println!(
        "  global_bytes_written: {}",
        plan.metrics.global_bytes_written
    );
    println!(
        "  local_memory_bytes_per_workgroup: {}",
        plan.metrics.local_memory_bytes_per_workgroup
    );
    if let Some(intensity) = plan.metrics.operational_intensity() {
        println!(
            "  operational_intensity_macs_per_byte: {}/{}",
            intensity.numerator, intensity.denominator
        );
    }
}

fn generate_vortex_attention_kernel_source(workspace_root: &Path) -> Result<()> {
    generate_vortex_attention_artifacts(workspace_root, true).map(|_| ())
}

fn generate_vortex_attention_artifacts(
    workspace_root: &Path,
    print_source: bool,
) -> Result<VortexMlirKernelArtifacts> {
    let config = VortexConfig::from_env(workspace_root)?;
    if config.toolchain_mode == VortexToolchainMode::Skip {
        reject_obvious_incompatible_prebuilt_tools(&config)?;
    }

    let plan = current_attention_prefill_plan()?;
    print_attention_plan("Vortex attention-prefill MLIR dispatch", &plan);
    match generate_vortex_attention_prefill_mlir(&plan) {
        Ok(generated) => {
            println!(
                "generated kernel: {} ({:?}) format={} ext=.{} headers={:?}",
                generated.symbol.as_str(),
                generated.implementation,
                generated.format.as_str(),
                generated.format.extension(),
                generated.required_headers
            );
            let artifacts = validate_attention_mlir_with_vortex_llvm(
                workspace_root,
                &config,
                generated.symbol.as_str(),
                &generated.source,
            )?;
            println!(
                "generated MLIR written to: {}",
                artifacts.mlir_path.display()
            );
            println!(
                "generated LLVM IR written to: {}",
                artifacts.ll_path.display()
            );
            println!(
                "generated object written to: {}",
                artifacts.obj_path.display()
            );
            println!("generated ELF written to: {}", artifacts.elf_path.display());
            println!(
                "generated vxbin written to: {}",
                artifacts.vxbin_path.display()
            );
            if print_source {
                println!("{}", generated.source);
            }
            Ok(artifacts)
        }
        Err(error) => Err(error.into()),
    }
}

struct XtaskCommandRunner;

impl VortexCommandRunner for XtaskCommandRunner {
    fn run(&mut self, phase: &str, command: Command) -> VortexToolchainResult<()> {
        run_checked(command, phase)
            .map_err(|error| VortexToolchainError::command_runner(phase, error.to_string()))
    }

    fn output(&mut self, phase: &str, command: Command) -> VortexToolchainResult<Output> {
        run_output_checked(command, phase)
            .map_err(|error| VortexToolchainError::command_runner(phase, error.to_string()))
    }
}

fn validate_attention_mlir_with_vortex_llvm(
    workspace_root: &Path,
    config: &VortexConfig,
    symbol_name: &str,
    source: &str,
) -> Result<VortexMlirKernelArtifacts> {
    let artifacts = VortexMlirKernelArtifacts::under_output_dir(
        &workspace_root.join("target/mandrel/vortex"),
        symbol_name,
    );
    let mut runner = XtaskCommandRunner;
    build_vortex_mlir_kernel_artifacts(
        VortexMlirKernelBuildRequest {
            workspace_root,
            config,
            symbol_name,
            source,
            artifacts: &artifacts,
            phase_prefix: "attention",
        },
        &mut runner,
    )?;

    println!("validated LLVM IR: {}", artifacts.ll_path.display());
    println!("validated object: {}", artifacts.obj_path.display());
    println!("validated ELF: {}", artifacts.elf_path.display());
    println!("validated vxbin: {}", artifacts.vxbin_path.display());
    Ok(artifacts)
}

fn run_vortex_attention_correctness(workspace_root: &Path) -> Result<()> {
    let config = VortexConfig::from_env(workspace_root)?;
    let artifacts = generate_vortex_attention_artifacts(workspace_root, false)?;
    require_file(&artifacts.vxbin_path, "generated attention vxbin")?;
    ensure_vortex_runtime_libraries(&config)?;
    let runtime = preferred_vortex_runtime_library(&config)?;

    println!(
        "Launching attention runtime correctness through Vortex simx with vxbin: {}",
        artifacts.vxbin_path.display()
    );
    let exe = env::current_exe()
        .map_err(|error| format!("failed to locate current xtask executable: {error}"))?;
    let mut command = Command::new(exe);
    command
        .current_dir(workspace_root)
        .arg("__vortex-run-attention-inner")
        .arg(&artifacts.vxbin_path)
        .env("VORTEX_DRIVER", "simx")
        .env("MANDREL_VORTEX_RUNTIME_LIB", &runtime)
        .env("MANDREL_VORTEX_RUNTIME_TRACE", "1");
    apply_vortex_env(&mut command, &config)?;
    run_checked(command, "attention.runtime_correctness")
}

fn run_vortex_attention_correctness_inner(workspace_root: &Path) -> Result<()> {
    let vxbin_path = env::args_os()
        .nth(2)
        .map(PathBuf::from)
        .ok_or_else(|| "missing vxbin path for __vortex-run-attention-inner".to_owned())?;
    require_file(&vxbin_path, "generated attention vxbin")?;

    runtime_step("compiling attention launch plan")?;
    let plan = current_attention_prefill_plan()?;
    runtime_step("building deterministic attention input")?;
    let input = deterministic_attention_prefill_input(&plan)?;
    runtime_step(&format!(
        "computing host reference for sequence={} head_dim={}",
        input.sequence, input.head_dim
    ))?;
    let expected = reference_attention_prefill_i8(&input)
        .map_err(|error| format!("failed to compute host attention reference: {error}"))?;

    let mut runtime_launch = plan.launch;
    if attention_runtime_flag("MANDREL_ATTENTION_RUNTIME_SCALAR_LAUNCH")? {
        runtime_step("using scalar launch override: grid=(1,1,1) block=(1,1,1) shared=0")?;
        runtime_launch.grid.x = 1;
        runtime_launch.grid.y = 1;
        runtime_launch.grid.z = 1;
        runtime_launch.block.x = 1;
        runtime_launch.block.y = 1;
        runtime_launch.block.z = 1;
        runtime_launch.shared_memory_bytes = 0;
    }
    runtime_step(&format!(
        "runtime launch dims grid=({}, {}, {}) block=({}, {}, {}) shared={}",
        runtime_launch.grid.x,
        runtime_launch.grid.y,
        runtime_launch.grid.z,
        runtime_launch.block.x,
        runtime_launch.block.y,
        runtime_launch.block.z,
        runtime_launch.shared_memory_bytes
    ))?;

    runtime_step("initializing Vortex backend/runtime")?;
    let config =
        VortexBackendConfig::new().with_kernel_artifact(runtime_launch.symbol, &vxbin_path);
    let mut backend = VortexBackend::new(config)
        .map_err(|error| format!("failed to initialize Vortex backend runtime: {error}"))?;
    runtime_step("launching Vortex attention kernel and reading output")?;
    let actual = backend
        .run_attention_prefill_i8(&runtime_launch, &input)
        .map_err(|error| format!("failed to run attention prefill on Vortex: {error}"))?;

    runtime_step("comparing Vortex output against host reference")?;
    compare_attention_outputs(&expected, &actual.output)?;
    println!("attention runtime correctness PASSED");
    println!("  vxbin: {}", vxbin_path.display());
    println!("  runtime root: {}", workspace_root.display());
    println!("  sequence: {}", input.sequence);
    println!("  head_dim: {}", input.head_dim);
    println!("  query_tile: {}", input.query_tile);
    println!("  key_tile: {}", input.key_tile);
    println!("  trace: {:?}", actual.trace);
    Ok(())
}

fn deterministic_attention_prefill_input(
    plan: &VortexAttentionPrefillPlan,
) -> Result<AttentionPrefillI8Run> {
    let default_sequence = plan.op.shape.sequence.min(8);
    let default_head_dim = plan.op.shape.head_dim.min(16);
    let sequence_usize = attention_runtime_extent_from_env(
        "MANDREL_ATTENTION_RUNTIME_SEQUENCE",
        default_sequence,
        plan.op.shape.sequence,
    )?;
    let head_dim_usize = attention_runtime_extent_from_env(
        "MANDREL_ATTENTION_RUNTIME_HEAD_DIM",
        default_head_dim,
        plan.op.shape.head_dim,
    )?;
    let sequence = u32::try_from(sequence_usize)
        .map_err(|_| format!("attention runtime sequence does not fit u32: {sequence_usize}"))?;
    let head_dim = u32::try_from(head_dim_usize)
        .map_err(|_| format!("attention runtime head_dim does not fit u32: {head_dim_usize}"))?;
    let query_tile = u32::try_from(plan.schedule.tile.query).map_err(|_| {
        format!(
            "attention query tile does not fit u32: {}",
            plan.schedule.tile.query
        )
    })?;
    let key_tile = u32::try_from(plan.schedule.tile.key).map_err(|_| {
        format!(
            "attention key tile does not fit u32: {}",
            plan.schedule.tile.key
        )
    })?;
    let elements = sequence_usize
        .checked_mul(head_dim_usize)
        .ok_or_else(|| "attention runtime element count overflow".to_owned())?;

    let q = (0..elements).map(|index| ((index % 5) as i8) - 2).collect();
    let k = (0..elements)
        .map(|index| (((index * 3 + 1) % 5) as i8) - 2)
        .collect();
    let v = (0..elements)
        .map(|index| (((index * 7 + 3) % 17) as i8) - 8)
        .collect();

    Ok(AttentionPrefillI8Run {
        q,
        k,
        v,
        sequence,
        head_dim,
        query_tile,
        key_tile,
    })
}

fn attention_runtime_flag(key: &str) -> Result<bool> {
    let Some(raw) = non_empty_env(key) else {
        return Ok(false);
    };
    match raw.as_str() {
        "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON" => Ok(true),
        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF" => Ok(false),
        other => Err(format!(
            "invalid {key}='{other}': expected one of 1/0, true/false, yes/no, on/off"
        )
        .into()),
    }
}

fn attention_runtime_extent_from_env(
    key: &str,
    default_value: usize,
    max_value: usize,
) -> Result<usize> {
    let Some(raw) = env::var_os(key) else {
        return Ok(default_value);
    };
    let text = raw.to_string_lossy();
    let value = text
        .parse::<usize>()
        .map_err(|error| format!("invalid {key}='{text}': {error}"))?;
    if value == 0 || value > max_value {
        return Err(format!(
            "{key} must be in 1..={max_value} for the current generated launch, got {value}"
        )
        .into());
    }
    Ok(value)
}

fn runtime_step(message: &str) -> Result<()> {
    println!("attention.runtime: {message}");
    io::stdout().flush().map_err(|error| {
        XtaskError::message(format!("failed to flush runtime progress output: {error}"))
    })
}

fn compare_attention_outputs(expected: &[i8], actual: &[i8]) -> Result<()> {
    if expected.len() != actual.len() {
        return Err(format!(
            "attention output length mismatch: expected {}, got {}",
            expected.len(),
            actual.len()
        )
        .into());
    }

    let mut mismatch_count = 0usize;
    let mut first_mismatches = Vec::new();
    for (index, (&expected_value, &actual_value)) in expected.iter().zip(actual).enumerate() {
        if expected_value != actual_value {
            mismatch_count += 1;
            if first_mismatches.len() < 16 {
                first_mismatches.push(format!(
                    "  index {index}: expected {expected_value}, got {actual_value}"
                ));
            }
        }
    }

    if mismatch_count == 0 {
        return Ok(());
    }

    Err(format!(
        "attention output mismatch: {mismatch_count}/{} elements differ\n{}",
        expected.len(),
        first_mismatches.join("\n")
    )
    .into())
}

fn run_vortex_vecadd(workspace_root: &Path) -> Result<()> {
    let config = VortexConfig::from_env(workspace_root)?;
    if config.toolchain_mode == VortexToolchainMode::System {
        prepare_vortex_system_tools(&config)?;
    }
    let status = VortexStatus::probe(&config);
    if !status.can_run_blackbox() {
        return Err(format!(
            "Vortex blackbox script is not available at {}; run `cargo vortex-fetch` or `cargo vortex-install` first",
            config.blackbox_script().display()
        ).into());
    }

    let mut command = Command::new(config.blackbox_script());
    command
        .current_dir(&config.source_dir)
        .args(["--cores=2", "--app=vecadd"]);
    apply_vortex_env(&mut command, &config)?;
    run_checked(command, "vortex.blackbox.vecadd")
}

#[allow(dead_code)]
fn ensure_vortex_runtime_libraries(config: &VortexConfig) -> Result<()> {
    let stub = config.build_dir.join("sw/runtime/libvortex.so");
    if !stub.is_file() {
        let mut make = Command::new("make");
        make.current_dir(&config.build_dir)
            .arg("-C")
            .arg("sw/runtime/stub");
        apply_vortex_env(&mut make, config)?;
        run_checked(make, "vortex.make.sw-runtime-stub")?;
    }

    let simx_driver = config.build_dir.join("sw/runtime/libvortex-simx.so");
    let simx_core = config.build_dir.join("sw/runtime/libsimx.so");
    if !simx_driver.is_file() || !simx_core.is_file() {
        let mut make = Command::new("make");
        make.current_dir(&config.build_dir)
            .arg("-C")
            .arg("sw/runtime/simx")
            .arg(format!(
                "DESTDIR={}",
                config.build_dir.join("sw/runtime").display()
            ));
        apply_vortex_env(&mut make, config)?;
        run_checked(make, "vortex.make.sw-runtime-simx")?;
    }

    require_file(&stub, "Vortex runtime libvortex.so")?;
    require_file(&simx_driver, "Vortex simx runtime driver libvortex-simx.so")?;
    require_file(&simx_core, "Vortex simx runtime core libsimx.so")
}

#[allow(dead_code)]
fn preferred_vortex_runtime_library(config: &VortexConfig) -> Result<PathBuf> {
    for candidate in vortex_runtime_library_candidates(config) {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(format!(
        "no Vortex runtime library found; tried: {}",
        vortex_runtime_library_candidates(config)
            .into_iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
    .into())
}

#[allow(dead_code)]
fn vortex_runtime_library_candidates(config: &VortexConfig) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = env::var_os("MANDREL_VORTEX_RUNTIME_LIB").map(PathBuf::from) {
        candidates.push(path);
    }
    candidates.push(config.build_dir.join("sw/runtime/libvortex.so"));
    candidates.push(config.install_dir().join("runtime/lib/libvortex.so"));
    candidates
}

#[allow(dead_code)]
fn require_file(path: &Path, description: &str) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(format!("missing {description}: {}", path.display()).into())
    }
}

#[allow(dead_code)]
fn file_contains_bytes(path: &Path, needle: &[u8]) -> Result<bool> {
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read '{}': {error}", path.display()))?;
    Ok(bytes.windows(needle.len()).any(|window| window == needle))
}

fn apply_vortex_env(command: &mut Command, config: &VortexConfig) -> Result<()> {
    command.env("VORTEX_HOME", &config.source_dir);
    command.env("VORTEX_BUILD_DIR", &config.build_dir);
    command.env("VORTEX_TOOL_DIR", &config.tool_dir);
    command.env("VORTEX_PATH", config.install_dir());
    command.env("MANDREL_FETCH_RETRIES", config.fetch_retries.to_string());
    if let Some(prefix) = config.normalized_download_proxy_prefix() {
        command.env("MANDREL_GITHUB_PROXY_PREFIX", prefix);
    }
    apply_github_proxy_git_env(command, config);
    prepend_env_path(command, "PKG_CONFIG_PATH", config.pkg_config_dir())?;
    prepend_env_paths(
        command,
        "LD_LIBRARY_PATH",
        [
            config.build_dir.join("sw/runtime"),
            config.install_dir().join("runtime/lib"),
            config.lib_dir(),
            config.tool_dir.join("llvm-vortex/lib"),
        ],
    )?;

    let mut path_entries = Vec::new();
    if config.download_proxy_prefix.is_some() {
        ensure_download_wrappers(config)?;
        path_entries.push(config.download_wrapper_dir());
    }
    path_entries.push(config.tool_dir.join("llvm-vortex/bin"));
    path_entries.push(config.bin_dir());
    prepend_env_paths(command, "PATH", path_entries)?;
    Ok(())
}

fn apply_github_proxy_git_env(command: &mut Command, config: &VortexConfig) {
    let Some(base) = config.git_proxy_base() else {
        return;
    };

    command.env(
        "GIT_CONFIG_COUNT",
        GITHUB_REWRITE_PATTERNS.len().to_string(),
    );
    for (index, pattern) in GITHUB_REWRITE_PATTERNS.iter().enumerate() {
        command.env(
            format!("GIT_CONFIG_KEY_{index}"),
            format!("url.{base}.insteadOf"),
        );
        command.env(format!("GIT_CONFIG_VALUE_{index}"), *pattern);
    }
}

fn prepend_env_path(command: &mut Command, key: &str, path: PathBuf) -> Result<()> {
    prepend_env_paths(command, key, [path])
}

fn prepend_env_paths<I>(command: &mut Command, key: &str, paths: I) -> Result<()>
where
    I: IntoIterator<Item = PathBuf>,
{
    let mut env_paths: Vec<PathBuf> = paths.into_iter().collect();
    if let Some(existing) = env::var_os(key) {
        env_paths.extend(env::split_paths(&existing));
    }
    let joined = env::join_paths(env_paths).map_err(|error| {
        XtaskError::message(format!(
            "failed to build {key} for Vortex command environment: {error}"
        ))
    })?;
    command.env(key, joined);
    Ok(())
}

fn run_checked(mut command: Command, phase: &str) -> Result<()> {
    let rendered = render_command(&command);
    let logged = truncate_command_for_log(&rendered);
    info!(phase, command = %logged, "running command");
    let status = command
        .status()
        .map_err(|source| XtaskError::CommandSpawn {
            phase: phase.to_owned(),
            source,
        })?;

    if status.success() {
        info!(phase, status = %status, "command completed");
        Ok(())
    } else {
        Err(XtaskError::CommandFailed {
            phase: phase.to_owned(),
            status,
            command: rendered,
        })
    }
}

fn run_output_checked(mut command: Command, phase: &str) -> Result<Output> {
    let rendered = render_command(&command);
    let logged = truncate_command_for_log(&rendered);
    info!(phase, command = %logged, "running command with captured output");
    let output = command
        .output()
        .map_err(|source| XtaskError::CommandSpawn {
            phase: phase.to_owned(),
            source,
        })?;

    if output.status.success() {
        info!(phase, status = %output.status, "command completed");
        Ok(output)
    } else {
        Err(XtaskError::CommandFailedWithStderr {
            phase: phase.to_owned(),
            status: output.status,
            command: rendered,
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

fn run_checked_with_retries<F>(mut make_command: F, phase: &str, attempts: u32) -> Result<()>
where
    F: FnMut() -> Result<Command>,
{
    let attempts = attempts.max(1);
    for attempt in 1..=attempts {
        match run_checked(make_command()?, phase) {
            Ok(()) => return Ok(()),
            Err(error) if attempt < attempts => {
                warn!(
                    phase,
                    attempt,
                    attempts,
                    error = %error,
                    "command failed; retrying"
                );
                eprintln!("{phase} failed on attempt {attempt}/{attempts}; retrying: {error}");
            }
            Err(error) => return Err(error),
        }
    }

    Err(XtaskError::message(format!("{phase} did not run")))
}

fn render_command(command: &Command) -> String {
    let mut parts = Vec::new();
    parts.push(shell_quote_lossy(command.get_program()));
    parts.extend(command.get_args().map(shell_quote_lossy));
    parts.join(" ")
}

fn truncate_command_for_log(command: &str) -> String {
    if command.len() <= LOG_COMMAND_MAX_CHARS {
        return command.to_owned();
    }

    let end = command
        .char_indices()
        .map(|(idx, _)| idx)
        .take_while(|idx| *idx <= LOG_COMMAND_MAX_CHARS)
        .last()
        .unwrap_or(0);
    format!(
        "{} ... <truncated: {} chars total>",
        &command[..end],
        command.len()
    )
}

fn shell_quote_lossy(value: &OsStr) -> String {
    let text = value.to_string_lossy();
    let mut quoted = String::from("'");
    for character in text.chars() {
        if character == '\'' {
            quoted.push_str("'\\''");
        } else {
            quoted.push(character);
        }
    }
    quoted.push('\'');
    quoted
}

#[cfg(test)]
mod tests {
    use super::{
        LOG_COMMAND_MAX_CHARS, Result, download_wrapper_script, make_executable, shell_quote_lossy,
        truncate_command_for_log, write_riscv_gcc_wrapper,
    };
    use mandrel_vortex_backend::{VortexMlirKernelArtifacts, vortex_kernel_entry_symbol};
    use std::env;
    use std::fs;
    use std::path::Path;
    use std::process::Command as StdCommand;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn attention_artifact_paths_follow_kernel_symbol() {
        let out_dir = Path::new("target/mandrel/vortex");
        let artifacts =
            VortexMlirKernelArtifacts::under_output_dir(out_dir, "attention_prefill_i8");

        assert_eq!(
            artifacts.mlir_path,
            out_dir.join("attention_prefill_i8.mlir")
        );
        assert_eq!(artifacts.ll_path, out_dir.join("attention_prefill_i8.ll"));
        assert_eq!(artifacts.obj_path, out_dir.join("attention_prefill_i8.o"));
        assert_eq!(
            artifacts.startup_probe_elf_path,
            out_dir.join("attention_prefill_i8.startup_probe.elf")
        );
        assert_eq!(
            artifacts.startup_object_path,
            out_dir.join("attention_prefill_i8.vx_start.o")
        );
        assert_eq!(artifacts.elf_path, out_dir.join("attention_prefill_i8.elf"));
        assert_eq!(
            artifacts.vxbin_path,
            out_dir.join("attention_prefill_i8.vxbin")
        );
    }

    #[test]
    fn command_log_truncation_preserves_short_commands() {
        let command = "'clang' '-c' 'attention.ll' '-o' 'attention.o'";

        assert_eq!(truncate_command_for_log(command), command);
    }

    #[test]
    fn command_log_truncation_marks_long_commands_without_splitting_utf8() {
        let command = format!("{}🚀{}", "a".repeat(LOG_COMMAND_MAX_CHARS), "b".repeat(32));
        let truncated = truncate_command_for_log(&command);

        assert!(truncated.contains("<truncated:"));
        assert!(truncated.ends_with(&format!("{} chars total>", command.len())));
        assert!(truncated.is_char_boundary(truncated.len()));
    }

    #[test]
    fn vortex_kentry_symbol_uses_runtime_lookup_prefix() {
        assert_eq!(
            vortex_kernel_entry_symbol("attention_prefill_i8"),
            "__vx_kentry_attention_prefill_i8"
        );
    }

    #[test]
    fn download_wrapper_prefixes_vortex_github_asset_url() -> Result<()> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("system clock before UNIX_EPOCH: {error}"))?
            .as_nanos();
        let base = env::temp_dir().join(format!(
            "mandrel-xtask-download-wrapper-{}-{nanos}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).map_err(|error| {
            format!(
                "failed to create test directory '{}': {error}",
                base.display()
            )
        })?;

        let real_program = base.join("real-wget");
        let captured_args = base.join("args.txt");
        let real_script = format!(
            "#!/usr/bin/env bash\nset -euo pipefail\nprintf '%s\\n' \"$@\" > {}\n",
            shell_quote_lossy(captured_args.as_os_str())
        );
        fs::write(&real_program, real_script).map_err(|error| {
            format!(
                "failed to write fake wget '{}': {error}",
                real_program.display()
            )
        })?;
        make_executable(&real_program)?;

        let wrapper = base.join("wget");
        fs::write(&wrapper, download_wrapper_script(&real_program))
            .map_err(|error| format!("failed to write wrapper '{}': {error}", wrapper.display()))?;
        make_executable(&wrapper)?;

        let original = "https://github.com/vortexgpgpu/vortex-toolchain-prebuilt/raw/v3.0/pocl/ubuntu/focal/pocl.tar.bz2";
        let status = StdCommand::new(&wrapper)
            .env("MANDREL_GITHUB_PROXY_PREFIX", "https://gh-proxy.org")
            .arg(original)
            .status()
            .map_err(|error| format!("failed to run wrapper '{}': {error}", wrapper.display()))?;
        if !status.success() {
            return Err(format!("wrapper exited with status {status}").into());
        }

        let captured = fs::read_to_string(&captured_args).map_err(|error| {
            format!(
                "failed to read captured args '{}': {error}",
                captured_args.display()
            )
        })?;
        assert_eq!(
            captured.trim(),
            "https://gh-proxy.org/https://github.com/vortexgpgpu/vortex-toolchain-prebuilt/raw/v3.0/pocl/ubuntu/focal/pocl.tar.bz2"
        );

        let _ = fs::remove_dir_all(&base);
        Ok(())
    }

    #[test]
    fn download_wrapper_leaves_non_github_args_unchanged() {
        let script = download_wrapper_script(Path::new("/usr/bin/curl"));

        assert!(script.contains("https://github.com/*"));
        assert!(script.contains("https://raw.githubusercontent.com/*"));
        assert!(script.contains("printf '%s' \"$arg\""));
    }

    #[cfg(unix)]
    #[test]
    fn riscv_gcc_wrapper_replaces_existing_symlink_without_touching_target() -> Result<()> {
        use std::os::unix::fs::symlink;

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("system clock before UNIX_EPOCH: {error}"))?
            .as_nanos();
        let base = env::temp_dir().join(format!(
            "mandrel-xtask-riscv-wrapper-{}-{nanos}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).map_err(|error| {
            format!(
                "failed to create test directory '{}': {error}",
                base.display()
            )
        })?;

        let real_gcc = base.join("real-riscv64-unknown-elf-gcc");
        fs::write(&real_gcc, "#!/usr/bin/env bash\nexit 0\n").map_err(|error| {
            format!("failed to write fake gcc '{}': {error}", real_gcc.display())
        })?;
        make_executable(&real_gcc)?;

        let include_dir = base.join("include");
        fs::create_dir_all(&include_dir).map_err(|error| {
            format!(
                "failed to create include dir '{}': {error}",
                include_dir.display()
            )
        })?;

        let symlink_target = base.join("system-gcc-placeholder");
        fs::write(&symlink_target, "do not overwrite\n").map_err(|error| {
            format!(
                "failed to write symlink target '{}': {error}",
                symlink_target.display()
            )
        })?;

        let wrapper = base.join("bin/riscv64-unknown-elf-gcc");
        fs::create_dir_all(
            wrapper
                .parent()
                .ok_or_else(|| format!("wrapper path '{}' has no parent", wrapper.display()))?,
        )
        .map_err(|error| format!("failed to create wrapper parent: {error}"))?;
        symlink(&symlink_target, &wrapper).map_err(|error| {
            format!(
                "failed to create test symlink '{}' -> '{}': {error}",
                wrapper.display(),
                symlink_target.display()
            )
        })?;

        write_riscv_gcc_wrapper(&wrapper, &real_gcc, &include_dir)?;

        let metadata = fs::symlink_metadata(&wrapper)
            .map_err(|error| format!("failed to stat wrapper '{}': {error}", wrapper.display()))?;
        assert!(!metadata.file_type().is_symlink());
        let wrapper_content = fs::read_to_string(&wrapper)
            .map_err(|error| format!("failed to read wrapper '{}': {error}", wrapper.display()))?;
        assert!(wrapper_content.contains("-isystem"));
        assert_eq!(
            fs::read_to_string(&symlink_target).map_err(|error| {
                format!(
                    "failed to read symlink target '{}': {error}",
                    symlink_target.display()
                )
            })?,
            "do not overwrite\n"
        );

        let _ = fs::remove_dir_all(&base);
        Ok(())
    }
}
