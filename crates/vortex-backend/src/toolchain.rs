use std::env;
use std::fs;
use std::num::ParseIntError;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::string::FromUtf8Error;

use snafu::Snafu;

pub const DEFAULT_VORTEX_DIR: &str = "external/vortex";
pub const DEFAULT_VORTEX_URL: &str = "https://github.com/vortexgpgpu/vortex.git";
pub const DEFAULT_VORTEX_BUILD_DIR: &str = "external/vortex-build";
pub const DEFAULT_VORTEX_TOOLDIR: &str = "external/vortex-source-tools";
pub const DEFAULT_VORTEX_SYSTEM_TOOLDIR: &str = "external/vortex-system-tools";
pub const DEFAULT_VORTEX_ENV_FILE: &str = "external/vortex-env.sh";
pub const DEFAULT_FETCH_RETRIES: u32 = 3;
pub const VORTEX_PREBUILT_HOST_ARCH: &str = "x86_64";

pub type VortexToolchainResult<T> = Result<T, VortexToolchainError>;

#[derive(Debug, Snafu)]
pub enum VortexToolchainError {
    #[snafu(display(
        "unsupported MANDREL_VORTEX_TOOLCHAIN_MODE '{value}'; use auto, prebuilt, system, or skip"
    ))]
    UnsupportedToolchainMode { value: String },
    #[snafu(display("unsupported MANDREL_VORTEX_XLEN '{xlen}'; use 32 or 64"))]
    UnsupportedXlen { xlen: u32 },
    #[snafu(display(
        "Vortex MLIR artifact pipeline currently targets rv64; got MANDREL_VORTEX_XLEN={xlen}"
    ))]
    UnsupportedMlirXlen { xlen: u32 },
    #[snafu(display(
        "Vortex upstream toolchain_install.sh fetches host prebuilt packages and this host is '{host_arch}', not '{expected_arch}'. To avoid running incompatible prebuilt binaries, either use Ubuntu/system packages with MANDREL_VORTEX_TOOLCHAIN_MODE=system, source-build/populate '{}' and rerun with MANDREL_VORTEX_TOOLCHAIN_MODE=skip, or force the upstream prebuilt path with MANDREL_VORTEX_TOOLCHAIN_MODE=prebuilt.",
        tool_dir.display()
    ))]
    UnsupportedPrebuiltHost {
        host_arch: &'static str,
        expected_arch: &'static str,
        tool_dir: PathBuf,
    },
    #[snafu(display("invalid MANDREL_FETCH_RETRIES '{raw}': {source}"))]
    InvalidFetchRetries { raw: String, source: ParseIntError },
    #[snafu(display("MANDREL_FETCH_RETRIES must be at least 1"))]
    FetchRetriesZero,
    #[snafu(display("cannot determine parent directory for {description} '{}'", path.display()))]
    MissingParent { description: String, path: PathBuf },
    #[snafu(display("missing {description}: {}", path.display()))]
    MissingFile { description: String, path: PathBuf },
    #[snafu(display("failed to create {description} '{}': {source}", path.display()))]
    CreateDir {
        description: String,
        path: PathBuf,
        source: std::io::Error,
    },
    #[snafu(display("failed to read '{}': {source}", path.display()))]
    ReadFile {
        path: PathBuf,
        source: std::io::Error,
    },
    #[snafu(display("failed to write '{}': {source}", path.display()))]
    WriteFile {
        path: PathBuf,
        source: std::io::Error,
    },
    #[snafu(display("failed to build {key} for Vortex command environment: {source}"))]
    JoinEnvPaths {
        key: String,
        source: env::JoinPathsError,
    },
    #[snafu(display("Vortex command phase {phase} failed: {message}"))]
    CommandRunner { phase: String, message: String },
    #[snafu(display("Vortex command phase {phase} emitted non-UTF8 output: {source}"))]
    NonUtf8Output {
        phase: String,
        source: FromUtf8Error,
    },
    #[snafu(display(
        "generated ELF '{}' does not contain required Vortex kernel entry symbol '{symbol}'; llvm-nm stdout:\n{stdout}",
        elf_path.display()
    ))]
    MissingElfSymbol {
        elf_path: PathBuf,
        symbol: String,
        stdout: String,
    },
    #[snafu(display(
        "generated vxbin '{}' is missing VXSYMTAB footer; named runtime lookup for '{symbol_name}' would fail",
        vxbin_path.display()
    ))]
    MissingVxbinFooter {
        vxbin_path: PathBuf,
        symbol_name: String,
    },
    #[snafu(display(
        "generated vxbin '{}' VXSYMTAB does not contain kernel name '{symbol_name}'",
        vxbin_path.display()
    ))]
    MissingVxbinSymbol {
        vxbin_path: PathBuf,
        symbol_name: String,
    },
    #[snafu(display("{message}"))]
    Message { message: String },
}

impl VortexToolchainError {
    pub fn message(message: impl Into<String>) -> Self {
        Self::Message {
            message: message.into(),
        }
    }

    pub fn command_runner(phase: impl Into<String>, message: impl Into<String>) -> Self {
        Self::CommandRunner {
            phase: phase.into(),
            message: message.into(),
        }
    }
}

impl From<String> for VortexToolchainError {
    fn from(message: String) -> Self {
        Self::message(message)
    }
}

impl From<&str> for VortexToolchainError {
    fn from(message: &str) -> Self {
        Self::message(message)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VortexToolchainMode {
    Auto,
    Prebuilt,
    System,
    Skip,
}

impl VortexToolchainMode {
    pub fn from_env() -> VortexToolchainResult<Self> {
        let Some(raw) = non_empty_env("MANDREL_VORTEX_TOOLCHAIN_MODE") else {
            return Ok(Self::default_for_host());
        };

        match raw.as_str() {
            "auto" => Ok(Self::Auto),
            "prebuilt" => Ok(Self::Prebuilt),
            "system" | "ubuntu" => Ok(Self::System),
            "skip" | "external" => Ok(Self::Skip),
            other => Err(VortexToolchainError::UnsupportedToolchainMode {
                value: other.to_owned(),
            }),
        }
    }

    pub fn default_for_host() -> Self {
        if env::consts::ARCH == VORTEX_PREBUILT_HOST_ARCH {
            Self::Auto
        } else {
            Self::System
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Prebuilt => "prebuilt",
            Self::System => "system",
            Self::Skip => "skip",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VortexConfig {
    pub source_dir: PathBuf,
    pub build_dir: PathBuf,
    pub tool_dir: PathBuf,
    pub env_file: PathBuf,
    pub url: String,
    pub download_proxy_prefix: Option<String>,
    pub git_proxy_prefix: Option<String>,
    pub fetch_retries: u32,
    pub reference: Option<String>,
    pub xlen: u32,
    pub toolchain_mode: VortexToolchainMode,
}

impl VortexConfig {
    pub fn from_env(workspace_root: &Path) -> VortexToolchainResult<Self> {
        let toolchain_mode = VortexToolchainMode::from_env()?;
        let source_dir =
            project_path_from_env(workspace_root, "MANDREL_VORTEX_DIR", DEFAULT_VORTEX_DIR);
        let build_dir = project_path_from_env(
            workspace_root,
            "MANDREL_VORTEX_BUILD_DIR",
            DEFAULT_VORTEX_BUILD_DIR,
        );
        let tool_dir = project_path_from_env(
            workspace_root,
            "MANDREL_VORTEX_TOOLDIR",
            DEFAULT_VORTEX_TOOLDIR,
        );
        let env_file = project_path_from_env(
            workspace_root,
            "MANDREL_VORTEX_ENV_FILE",
            DEFAULT_VORTEX_ENV_FILE,
        );
        let url = env::var("MANDREL_VORTEX_URL").unwrap_or_else(|_| DEFAULT_VORTEX_URL.to_owned());
        let download_proxy_prefix =
            non_empty_env("MANDREL_GITHUB_PROXY_PREFIX").or_else(|| non_empty_env("PROXY_PREFIX"));
        let git_proxy_prefix = non_empty_env("MANDREL_GIT_PROXY_PREFIX");
        let fetch_retries = parse_fetch_retries()?;
        let reference = non_empty_env("MANDREL_VORTEX_REF");
        let xlen = env::var("MANDREL_VORTEX_XLEN")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(64);

        if xlen != 32 && xlen != 64 {
            return Err(VortexToolchainError::UnsupportedXlen { xlen });
        }

        Ok(Self {
            source_dir,
            build_dir,
            tool_dir,
            env_file,
            url,
            download_proxy_prefix,
            git_proxy_prefix,
            fetch_retries,
            reference,
            xlen,
            toolchain_mode,
        })
    }

    pub fn install_dir(&self) -> PathBuf {
        self.build_dir.join("install")
    }

    pub fn bin_dir(&self) -> PathBuf {
        self.install_dir().join("bin")
    }

    pub fn lib_dir(&self) -> PathBuf {
        self.install_dir().join("lib")
    }

    pub fn pkg_config_dir(&self) -> PathBuf {
        self.lib_dir().join("pkgconfig")
    }

    pub fn simx_dir(&self) -> PathBuf {
        self.build_dir.join("sim/simx")
    }

    pub fn simx_binary(&self) -> PathBuf {
        self.simx_dir().join("simx")
    }

    pub fn vortex_runtime_pc(&self) -> PathBuf {
        self.pkg_config_dir().join("vortex-runtime.pc")
    }

    pub fn vortex_kernel_pc(&self) -> PathBuf {
        self.pkg_config_dir().join("vortex-kernel.pc")
    }

    pub fn blackbox_script(&self) -> PathBuf {
        self.build_dir.join("ci/blackbox.sh")
    }

    pub fn download_wrapper_dir(&self) -> PathBuf {
        self.build_dir.join("mandrel-download-wrappers")
    }

    pub fn clone_url(&self) -> String {
        match &self.git_proxy_prefix {
            Some(prefix) => maybe_proxied_github_url(&self.url, prefix),
            None => self.url.clone(),
        }
    }

    pub fn git_proxy_base(&self) -> Option<String> {
        self.git_proxy_prefix.as_deref().map(github_proxy_base)
    }

    pub fn normalized_download_proxy_prefix(&self) -> Option<String> {
        self.download_proxy_prefix
            .as_deref()
            .map(normalized_proxy_prefix)
    }

    pub fn should_run_prebuilt_toolchain(&self) -> VortexToolchainResult<bool> {
        match self.toolchain_mode {
            VortexToolchainMode::Prebuilt => Ok(true),
            VortexToolchainMode::System | VortexToolchainMode::Skip => Ok(false),
            VortexToolchainMode::Auto if env::consts::ARCH == VORTEX_PREBUILT_HOST_ARCH => Ok(true),
            VortexToolchainMode::Auto => Err(VortexToolchainError::UnsupportedPrebuiltHost {
                host_arch: env::consts::ARCH,
                expected_arch: VORTEX_PREBUILT_HOST_ARCH,
                tool_dir: self.tool_dir.clone(),
            }),
        }
    }

    pub fn configure_command(&self) -> Command {
        let mut command = Command::new(self.source_dir.join("configure"));
        command
            .current_dir(&self.build_dir)
            .arg(format!("--xlen={}", self.xlen))
            .arg(format!("--tooldir={}", self.tool_dir.display()))
            .arg(format!("--prefix={}", self.install_dir().display()));
        command
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VortexStatus {
    pub checkout_exists: bool,
    pub build_dir_exists: bool,
    pub install_dir_exists: bool,
    pub env_file_exists: bool,
    pub download_wrapper_dir_exists: bool,
    pub blackbox_script_exists: bool,
    pub simx_binary_exists: bool,
    pub runtime_pkg_config_exists: bool,
    pub kernel_pkg_config_exists: bool,
}

impl VortexStatus {
    pub fn probe(config: &VortexConfig) -> Self {
        Self {
            checkout_exists: config.source_dir.join(".git").is_dir(),
            build_dir_exists: config.build_dir.is_dir(),
            install_dir_exists: config.install_dir().is_dir(),
            env_file_exists: config.env_file.is_file(),
            download_wrapper_dir_exists: config.download_wrapper_dir().is_dir(),
            blackbox_script_exists: config.blackbox_script().is_file(),
            simx_binary_exists: config.simx_binary().is_file(),
            runtime_pkg_config_exists: config.vortex_runtime_pc().is_file(),
            kernel_pkg_config_exists: config.vortex_kernel_pc().is_file(),
        }
    }

    pub const fn can_run_blackbox(self) -> bool {
        self.checkout_exists && self.blackbox_script_exists && self.simx_binary_exists
    }

    pub const fn can_run_simx(self) -> bool {
        self.simx_binary_exists
    }

    pub const fn can_use_installed_runtime(self) -> bool {
        self.install_dir_exists && self.runtime_pkg_config_exists && self.kernel_pkg_config_exists
    }
}

pub trait VortexCommandRunner {
    fn run(&mut self, phase: &str, command: Command) -> VortexToolchainResult<()>;

    fn output(&mut self, phase: &str, command: Command) -> VortexToolchainResult<Output>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VortexMlirKernelArtifacts {
    pub mlir_path: PathBuf,
    pub ll_path: PathBuf,
    pub obj_path: PathBuf,
    pub startup_probe_elf_path: PathBuf,
    pub startup_object_path: PathBuf,
    pub elf_path: PathBuf,
    pub vxbin_path: PathBuf,
}

impl VortexMlirKernelArtifacts {
    pub fn under_output_dir(out_dir: &Path, symbol_name: &str) -> Self {
        Self {
            mlir_path: out_dir.join(format!("{symbol_name}.mlir")),
            ll_path: out_dir.join(format!("{symbol_name}.ll")),
            obj_path: out_dir.join(format!("{symbol_name}.o")),
            startup_probe_elf_path: out_dir.join(format!("{symbol_name}.startup_probe.elf")),
            startup_object_path: out_dir.join(format!("{symbol_name}.vx_start.o")),
            elf_path: out_dir.join(format!("{symbol_name}.elf")),
            vxbin_path: out_dir.join(format!("{symbol_name}.vxbin")),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VortexMlirKernelBuildRequest<'a> {
    pub workspace_root: &'a Path,
    pub config: &'a VortexConfig,
    pub symbol_name: &'a str,
    pub source: &'a str,
    pub artifacts: &'a VortexMlirKernelArtifacts,
    pub phase_prefix: &'a str,
}

#[derive(Debug, Clone, Copy)]
pub struct VortexKernelLinkRequest<'a> {
    pub workspace_root: &'a Path,
    pub config: &'a VortexConfig,
    pub kernel_object_path: &'a Path,
    pub startup_probe_elf_path: &'a Path,
    pub startup_object_path: &'a Path,
    pub elf_path: &'a Path,
    pub kentry_symbol: &'a str,
    pub phase_prefix: &'a str,
}

#[derive(Debug, Clone, Copy)]
pub struct VortexVxbinPackageRequest<'a> {
    pub workspace_root: &'a Path,
    pub config: &'a VortexConfig,
    pub elf_path: &'a Path,
    pub vxbin_path: &'a Path,
    pub phase_prefix: &'a str,
}

pub fn vortex_kernel_entry_symbol(symbol_name: &str) -> String {
    format!("__vx_kentry_{symbol_name}")
}

pub fn build_vortex_mlir_kernel_artifacts<R>(
    request: VortexMlirKernelBuildRequest<'_>,
    runner: &mut R,
) -> VortexToolchainResult<()>
where
    R: VortexCommandRunner,
{
    if request.config.xlen != 64 {
        return Err(VortexToolchainError::UnsupportedMlirXlen {
            xlen: request.config.xlen,
        });
    }

    ensure_parent_dir(
        &request.artifacts.mlir_path,
        "generated Vortex output directory",
    )?;
    fs::write(&request.artifacts.mlir_path, request.source).map_err(|source| {
        VortexToolchainError::WriteFile {
            path: request.artifacts.mlir_path.clone(),
            source,
        }
    })?;

    translate_mlir_to_llvm_ir(request, runner)?;
    compile_llvm_ir_to_vortex_object(request, runner)?;

    let kentry_symbol = vortex_kernel_entry_symbol(request.symbol_name);
    link_vortex_kernel_object_to_elf(
        VortexKernelLinkRequest {
            workspace_root: request.workspace_root,
            config: request.config,
            kernel_object_path: &request.artifacts.obj_path,
            startup_probe_elf_path: &request.artifacts.startup_probe_elf_path,
            startup_object_path: &request.artifacts.startup_object_path,
            elf_path: &request.artifacts.elf_path,
            kentry_symbol: &kentry_symbol,
            phase_prefix: request.phase_prefix,
        },
        runner,
    )?;
    verify_vortex_kernel_elf_contains_symbol(
        request.workspace_root,
        request.config,
        &request.artifacts.elf_path,
        &kentry_symbol,
        request.phase_prefix,
        runner,
    )?;
    package_vortex_elf_to_vxbin(
        VortexVxbinPackageRequest {
            workspace_root: request.workspace_root,
            config: request.config,
            elf_path: &request.artifacts.elf_path,
            vxbin_path: &request.artifacts.vxbin_path,
            phase_prefix: request.phase_prefix,
        },
        runner,
    )?;
    verify_vortex_vxbin_symbols(&request.artifacts.vxbin_path, request.symbol_name)
}

pub fn link_vortex_kernel_object_to_elf<R>(
    request: VortexKernelLinkRequest<'_>,
    runner: &mut R,
) -> VortexToolchainResult<()>
where
    R: VortexCommandRunner,
{
    let tools = VortexRv64KernelLinkTools::probe(request.config)?;
    require_file(request.kernel_object_path, "generated Vortex kernel object")?;

    run_vortex_kernel_link_command(
        request.workspace_root,
        request.config,
        &tools,
        [request.kernel_object_path],
        request.kentry_symbol,
        request.startup_probe_elf_path,
        &prefixed_phase(request.phase_prefix, "clang_link_startup_probe_elf"),
        runner,
    )?;

    let startup_flags = detect_vortex_startup_flags(
        request.workspace_root,
        request.config,
        &tools.kernel_startup,
        &tools.objdump,
        request.startup_probe_elf_path,
        request.phase_prefix,
        runner,
    )?;
    compile_vortex_startup_object(
        request.workspace_root,
        request.config,
        &tools.clang,
        &tools.startup_source,
        request.startup_object_path,
        &startup_flags,
        request.phase_prefix,
        runner,
    )?;

    run_vortex_kernel_link_command(
        request.workspace_root,
        request.config,
        &tools,
        [request.startup_object_path, request.kernel_object_path],
        request.kentry_symbol,
        request.elf_path,
        &prefixed_phase(request.phase_prefix, "clang_link_elf"),
        runner,
    )
}

pub fn package_vortex_elf_to_vxbin<R>(
    request: VortexVxbinPackageRequest<'_>,
    runner: &mut R,
) -> VortexToolchainResult<()>
where
    R: VortexCommandRunner,
{
    let objcopy = request.config.tool_dir.join("llvm-vortex/bin/llvm-objcopy");
    let vxbin_py = request.config.source_dir.join("sw/kernel/scripts/vxbin.py");
    require_file(&objcopy, "Vortex llvm-objcopy")?;
    require_file(&vxbin_py, "Vortex vxbin packager")?;
    require_file(request.elf_path, "generated Vortex ELF")?;

    let mut vxbin = Command::new("python3");
    vxbin
        .current_dir(request.workspace_root)
        .env("OBJCOPY", &objcopy)
        .arg(&vxbin_py)
        .arg(request.elf_path)
        .arg(request.vxbin_path);
    apply_vortex_command_env(&mut vxbin, request.config)?;
    runner.run(
        &prefixed_phase(request.phase_prefix, "vxbin_package"),
        vxbin,
    )
}

pub fn verify_vortex_kernel_elf_contains_symbol<R>(
    workspace_root: &Path,
    config: &VortexConfig,
    elf_path: &Path,
    symbol: &str,
    phase_prefix: &str,
    runner: &mut R,
) -> VortexToolchainResult<()>
where
    R: VortexCommandRunner,
{
    let nm = config.tool_dir.join("llvm-vortex/bin/llvm-nm");
    require_file(&nm, "Vortex llvm-nm")?;
    require_file(elf_path, "generated Vortex ELF")?;

    let mut nm_cmd = Command::new(&nm);
    nm_cmd.current_dir(workspace_root).arg(elf_path);
    apply_vortex_command_env(&mut nm_cmd, config)?;
    let output = runner.output(&prefixed_phase(phase_prefix, "elf_nm"), nm_cmd)?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout
        .lines()
        .any(|line| line.split_whitespace().last() == Some(symbol))
    {
        Ok(())
    } else {
        Err(VortexToolchainError::MissingElfSymbol {
            elf_path: elf_path.to_path_buf(),
            symbol: symbol.to_owned(),
            stdout: stdout.into_owned(),
        })
    }
}

pub fn verify_vortex_vxbin_symbols(
    vxbin_path: &Path,
    symbol_name: &str,
) -> VortexToolchainResult<()> {
    require_file(vxbin_path, "generated Vortex vxbin")?;
    if !file_contains_bytes(vxbin_path, b"VXSYMTAB")? {
        return Err(VortexToolchainError::MissingVxbinFooter {
            vxbin_path: vxbin_path.to_path_buf(),
            symbol_name: symbol_name.to_owned(),
        });
    }
    if !file_contains_bytes(vxbin_path, symbol_name.as_bytes())? {
        return Err(VortexToolchainError::MissingVxbinSymbol {
            vxbin_path: vxbin_path.to_path_buf(),
            symbol_name: symbol_name.to_owned(),
        });
    }
    Ok(())
}

pub fn apply_vortex_command_env(
    command: &mut Command,
    config: &VortexConfig,
) -> VortexToolchainResult<()> {
    command.env("VORTEX_HOME", &config.source_dir);
    command.env("VORTEX_BUILD_DIR", &config.build_dir);
    command.env("VORTEX_TOOL_DIR", &config.tool_dir);
    command.env("VORTEX_PATH", config.install_dir());
    command.env("MANDREL_FETCH_RETRIES", config.fetch_retries.to_string());
    if let Some(prefix) = config.normalized_download_proxy_prefix() {
        command.env("MANDREL_GITHUB_PROXY_PREFIX", prefix);
    }
    prepend_command_env_path(command, "PKG_CONFIG_PATH", config.pkg_config_dir())?;
    prepend_command_env_paths(
        command,
        "LD_LIBRARY_PATH",
        [
            config.build_dir.join("sw/runtime"),
            config.install_dir().join("runtime/lib"),
            config.lib_dir(),
            config.tool_dir.join("llvm-vortex/lib"),
        ],
    )?;
    prepend_command_env_paths(
        command,
        "PATH",
        [config.tool_dir.join("llvm-vortex/bin"), config.bin_dir()],
    )?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VortexRv64KernelLinkTools {
    clang: PathBuf,
    objdump: PathBuf,
    link_script: PathBuf,
    startup_source: PathBuf,
    kernel_startup: PathBuf,
    kernel_runtime: PathBuf,
    libc_dir: PathBuf,
    builtins: PathBuf,
}

impl VortexRv64KernelLinkTools {
    fn probe(config: &VortexConfig) -> VortexToolchainResult<Self> {
        let llvm_bin = config.tool_dir.join("llvm-vortex/bin");
        let tools = Self {
            clang: llvm_bin.join("clang"),
            objdump: llvm_bin.join("llvm-objdump"),
            link_script: config.source_dir.join("sw/kernel/scripts/link64.ld"),
            startup_source: config.source_dir.join("sw/kernel/src/vx_start.S"),
            kernel_startup: config
                .source_dir
                .join("sw/kernel/scripts/kernel_startup.sh"),
            kernel_runtime: config.build_dir.join("sw/kernel/libvortex2.a"),
            libc_dir: config.tool_dir.join("libc64/lib"),
            builtins: config
                .tool_dir
                .join("libcrt64/lib/baremetal/libclang_rt.builtins-riscv64.a"),
        };

        require_file(&tools.clang, "Vortex clang driver")?;
        require_file(&tools.objdump, "Vortex llvm-objdump")?;
        require_file(&tools.link_script, "Vortex rv64 linker script")?;
        require_file(&tools.startup_source, "Vortex KMU startup source")?;
        require_file(&tools.kernel_startup, "Vortex kernel startup detector")?;
        require_file(&tools.kernel_runtime, "Vortex KMU kernel runtime archive")?;
        require_file(&tools.libc_dir.join("libm.a"), "Vortex rv64 libm archive")?;
        require_file(&tools.libc_dir.join("libc.a"), "Vortex rv64 libc archive")?;
        require_file(&tools.builtins, "Vortex rv64 compiler-rt builtins archive")?;
        Ok(tools)
    }
}

fn translate_mlir_to_llvm_ir<R>(
    request: VortexMlirKernelBuildRequest<'_>,
    runner: &mut R,
) -> VortexToolchainResult<()>
where
    R: VortexCommandRunner,
{
    let mlir_translate = request
        .config
        .tool_dir
        .join("llvm-vortex/bin/mlir-translate");
    require_file(&mlir_translate, "Vortex MLIR translator")?;

    let mut translate = Command::new(&mlir_translate);
    translate
        .current_dir(request.workspace_root)
        .arg("--mlir-to-llvmir")
        .arg(&request.artifacts.mlir_path)
        .arg("-o")
        .arg(&request.artifacts.ll_path);
    apply_vortex_command_env(&mut translate, request.config)?;
    runner.run(
        &prefixed_phase(request.phase_prefix, "mlir_translate"),
        translate,
    )
}

fn compile_llvm_ir_to_vortex_object<R>(
    request: VortexMlirKernelBuildRequest<'_>,
    runner: &mut R,
) -> VortexToolchainResult<()>
where
    R: VortexCommandRunner,
{
    let clang = request.config.tool_dir.join("llvm-vortex/bin/clang");
    require_file(&clang, "Vortex clang driver")?;

    let mut clang_cmd = Command::new(&clang);
    clang_cmd
        .current_dir(request.workspace_root)
        .arg("-c")
        .arg("-target")
        .arg("riscv64-unknown-unknown-elf")
        .arg("-march=rv64imafdc_xvortex")
        .arg("-O1")
        .arg(&request.artifacts.ll_path)
        .arg("-o")
        .arg(&request.artifacts.obj_path);
    apply_vortex_command_env(&mut clang_cmd, request.config)?;
    runner.run(
        &prefixed_phase(request.phase_prefix, "clang_object"),
        clang_cmd,
    )
}

fn run_vortex_kernel_link_command<'a, R, I>(
    workspace_root: &Path,
    config: &VortexConfig,
    tools: &VortexRv64KernelLinkTools,
    input_objects: I,
    kentry_symbol: &str,
    output_elf: &Path,
    phase: &str,
    runner: &mut R,
) -> VortexToolchainResult<()>
where
    R: VortexCommandRunner,
    I: IntoIterator<Item = &'a Path>,
{
    let riscv_sysroot = config
        .tool_dir
        .join("riscv64-gnu-toolchain/riscv64-unknown-elf");
    let riscv_toolchain = config.tool_dir.join("riscv64-gnu-toolchain");

    let mut clang_cmd = Command::new(&tools.clang);
    clang_cmd
        .current_dir(workspace_root)
        .arg("-target")
        .arg("riscv64-unknown-unknown-elf")
        .arg(format!("--sysroot={}", riscv_sysroot.display()))
        .arg(format!("--gcc-toolchain={}", riscv_toolchain.display()))
        .arg("-march=rv64imafdc_xvortex")
        .arg("-mabi=lp64d")
        .arg("-mcmodel=medany")
        .arg("-fuse-ld=lld")
        .arg("-nostartfiles")
        .arg("-nostdlib");
    for object in input_objects {
        clang_cmd.arg(object);
    }
    clang_cmd
        .arg(format!(
            "-Wl,-Bstatic,--gc-sections,-T,{},--defsym=STARTUP_ADDR=0x80000000,--undefined={}",
            tools.link_script.display(),
            kentry_symbol
        ))
        .arg(&tools.kernel_runtime)
        .arg(format!("-L{}", tools.libc_dir.display()))
        .arg("-lm")
        .arg("-lc")
        .arg(&tools.builtins)
        .arg("-o")
        .arg(output_elf);
    apply_vortex_command_env(&mut clang_cmd, config)?;
    runner.run(phase, clang_cmd)
}

fn detect_vortex_startup_flags<R>(
    workspace_root: &Path,
    config: &VortexConfig,
    kernel_startup: &Path,
    objdump: &Path,
    probe_elf: &Path,
    phase_prefix: &str,
    runner: &mut R,
) -> VortexToolchainResult<Vec<String>>
where
    R: VortexCommandRunner,
{
    let mut detect = Command::new(kernel_startup);
    detect
        .current_dir(workspace_root)
        .arg(objdump)
        .arg(probe_elf);
    apply_vortex_command_env(&mut detect, config)?;
    let phase = prefixed_phase(phase_prefix, "kernel_startup_flags");
    let output = runner.output(&phase, detect)?;
    let stdout = String::from_utf8(output.stdout)
        .map_err(|source| VortexToolchainError::NonUtf8Output { phase, source })?;
    Ok(stdout
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>())
}

fn vortex_config_cflags<R>(
    workspace_root: &Path,
    config: &VortexConfig,
    phase_prefix: &str,
    runner: &mut R,
) -> VortexToolchainResult<Vec<String>>
where
    R: VortexCommandRunner,
{
    let gen_config = config.source_dir.join("ci/gen_config.py");
    let vx_config = config.source_dir.join("VX_config.toml");
    require_file(&gen_config, "Vortex config flag generator")?;
    require_file(&vx_config, "Vortex config TOML")?;

    let mut command = Command::new("python3");
    command
        .current_dir(workspace_root)
        .arg(&gen_config)
        .arg(format!("--config={}", vx_config.display()))
        .arg("--cflags=-DVX_CFG_XLEN=64");
    apply_vortex_command_env(&mut command, config)?;
    let phase = prefixed_phase(phase_prefix, "vortex_config_cflags");
    let output = runner.output(&phase, command)?;
    let stdout = String::from_utf8(output.stdout)
        .map_err(|source| VortexToolchainError::NonUtf8Output { phase, source })?;
    Ok(stdout
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>())
}

fn compile_vortex_startup_object<R>(
    workspace_root: &Path,
    config: &VortexConfig,
    clang: &Path,
    startup_source: &Path,
    startup_object: &Path,
    startup_flags: &[String],
    phase_prefix: &str,
    runner: &mut R,
) -> VortexToolchainResult<()>
where
    R: VortexCommandRunner,
{
    let riscv_sysroot = config
        .tool_dir
        .join("riscv64-gnu-toolchain/riscv64-unknown-elf");
    let riscv_toolchain = config.tool_dir.join("riscv64-gnu-toolchain");
    let kernel_include = config.source_dir.join("sw/kernel/include");
    let kernel_src = config.source_dir.join("sw/kernel/src");
    let generated_sw = config.build_dir.join("sw");
    let config_flags = vortex_config_cflags(workspace_root, config, phase_prefix, runner)?;

    require_file(
        &generated_sw.join("VX_types.h"),
        "generated Vortex VX_types.h",
    )?;
    require_file(
        &generated_sw.join("VX_config.h"),
        "generated Vortex VX_config.h",
    )?;

    let mut clang_cmd = Command::new(clang);
    clang_cmd
        .current_dir(workspace_root)
        .arg("-c")
        .arg("-target")
        .arg("riscv64-unknown-unknown-elf")
        .arg(format!("--sysroot={}", riscv_sysroot.display()))
        .arg(format!("--gcc-toolchain={}", riscv_toolchain.display()))
        .arg("-march=rv64imafdc_xvortex")
        .arg("-mabi=lp64d")
        .arg("-mcmodel=medany")
        .arg("-O3")
        .arg("-Wno-unused-command-line-argument")
        .arg("-DNDEBUG")
        .arg("-D__VORTEX__")
        .arg("-DKMU_ENABLE")
        .arg(format!("-I{}", kernel_include.display()))
        .arg(format!("-I{}", kernel_src.display()))
        .arg(format!("-I{}", generated_sw.display()));
    for flag in config_flags.iter().chain(startup_flags) {
        clang_cmd.arg(flag);
    }
    clang_cmd.arg(startup_source).arg("-o").arg(startup_object);
    apply_vortex_command_env(&mut clang_cmd, config)?;
    runner.run(
        &prefixed_phase(phase_prefix, "clang_startup_object"),
        clang_cmd,
    )
}

fn ensure_parent_dir(path: &Path, description: &str) -> VortexToolchainResult<()> {
    let parent = path
        .parent()
        .ok_or_else(|| VortexToolchainError::MissingParent {
            description: description.to_owned(),
            path: path.to_path_buf(),
        })?;
    fs::create_dir_all(parent).map_err(|source| VortexToolchainError::CreateDir {
        description: description.to_owned(),
        path: parent.to_path_buf(),
        source,
    })
}

fn require_file(path: &Path, description: &str) -> VortexToolchainResult<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(VortexToolchainError::MissingFile {
            description: description.to_owned(),
            path: path.to_path_buf(),
        })
    }
}

fn file_contains_bytes(path: &Path, needle: &[u8]) -> VortexToolchainResult<bool> {
    let bytes = fs::read(path).map_err(|source| VortexToolchainError::ReadFile {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(bytes.windows(needle.len()).any(|window| window == needle))
}

fn prepend_command_env_path(
    command: &mut Command,
    key: &str,
    path: PathBuf,
) -> VortexToolchainResult<()> {
    prepend_command_env_paths(command, key, [path])
}

fn prepend_command_env_paths<I>(
    command: &mut Command,
    key: &str,
    paths: I,
) -> VortexToolchainResult<()>
where
    I: IntoIterator<Item = PathBuf>,
{
    let mut env_paths: Vec<PathBuf> = paths.into_iter().collect();
    if let Some(existing) = env::var_os(key) {
        env_paths.extend(env::split_paths(&existing));
    }
    let joined =
        env::join_paths(env_paths).map_err(|source| VortexToolchainError::JoinEnvPaths {
            key: key.to_owned(),
            source,
        })?;
    command.env(key, joined);
    Ok(())
}

fn prefixed_phase(prefix: &str, name: &str) -> String {
    format!("{prefix}.{name}")
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

fn parse_fetch_retries() -> VortexToolchainResult<u32> {
    let Some(raw) = non_empty_env("MANDREL_FETCH_RETRIES") else {
        return Ok(DEFAULT_FETCH_RETRIES);
    };

    let retries =
        raw.parse::<u32>()
            .map_err(|source| VortexToolchainError::InvalidFetchRetries {
                raw: raw.clone(),
                source,
            })?;
    if retries == 0 {
        return Err(VortexToolchainError::FetchRetriesZero);
    }
    Ok(retries)
}

fn normalized_proxy_prefix(prefix: &str) -> String {
    if prefix.ends_with('/') {
        prefix.to_owned()
    } else {
        format!("{prefix}/")
    }
}

fn github_proxy_base(prefix: &str) -> String {
    format!("{}https://github.com/", normalized_proxy_prefix(prefix))
}

fn maybe_proxied_github_url(url: &str, prefix: &str) -> String {
    let base = github_proxy_base(prefix);
    if let Some(rest) = url.strip_prefix("https://github.com/") {
        format!("{base}{rest}")
    } else if let Some(rest) = url.strip_prefix("http://github.com/") {
        format!("{base}{rest}")
    } else if let Some(rest) = url.strip_prefix("git@github.com:") {
        format!("{base}{rest}")
    } else if let Some(rest) = url.strip_prefix("ssh://git@github.com/") {
        format!("{base}{rest}")
    } else {
        url.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::{VortexConfig, VortexStatus, VortexToolchainMode};
    use std::path::Path;

    #[test]
    fn status_probe_is_false_for_missing_checkout() {
        let config = match VortexConfig::from_env(Path::new("/tmp/mandrel-missing")) {
            Ok(config) => config,
            Err(error) => panic!("unexpected config error: {error}"),
        };
        let status = VortexStatus::probe(&config);

        assert!(!status.can_run_blackbox());
        assert!(!status.can_use_installed_runtime());
    }

    #[test]
    fn download_proxy_prefix_does_not_change_clone_url() {
        let config = VortexConfig {
            source_dir: Path::new("/tmp/mandrel/source").to_path_buf(),
            build_dir: Path::new("/tmp/mandrel/build").to_path_buf(),
            tool_dir: Path::new("/tmp/mandrel/tools").to_path_buf(),
            env_file: Path::new("/tmp/mandrel/env.sh").to_path_buf(),
            url: "https://github.com/vortexgpgpu/vortex.git".to_owned(),
            download_proxy_prefix: Some("https://gh-proxy.org".to_owned()),
            git_proxy_prefix: None,
            fetch_retries: 3,
            reference: None,
            xlen: 64,
            toolchain_mode: VortexToolchainMode::Auto,
        };

        assert_eq!(
            config.clone_url(),
            "https://github.com/vortexgpgpu/vortex.git"
        );
        assert_eq!(
            config.normalized_download_proxy_prefix().as_deref(),
            Some("https://gh-proxy.org/")
        );
    }

    #[test]
    fn skip_toolchain_mode_does_not_run_prebuilt() {
        let config = VortexConfig {
            source_dir: Path::new("/tmp/mandrel/source").to_path_buf(),
            build_dir: Path::new("/tmp/mandrel/build").to_path_buf(),
            tool_dir: Path::new("/tmp/mandrel/tools").to_path_buf(),
            env_file: Path::new("/tmp/mandrel/env.sh").to_path_buf(),
            url: "https://github.com/vortexgpgpu/vortex.git".to_owned(),
            download_proxy_prefix: None,
            git_proxy_prefix: None,
            fetch_retries: 3,
            reference: None,
            xlen: 64,
            toolchain_mode: VortexToolchainMode::Skip,
        };

        assert!(matches!(config.should_run_prebuilt_toolchain(), Ok(false)));
    }
}
