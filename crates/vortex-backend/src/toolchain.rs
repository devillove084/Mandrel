use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::string::FromUtf8Error;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use snafu::Snafu;

pub const DEFAULT_VORTEX_DIR: &str = "external/vortex";
pub const DEFAULT_VORTEX_BUILD_DIR: &str = "external/vortex-build";
pub const DEFAULT_VORTEX_TOOLDIR: &str = "external/vortex-source-tools";
pub const DEFAULT_VERILATOR_DIR: &str = "external/verilator-install";
pub const DEFAULT_PYTHON_VENV_DIR: &str = ".venv";

const VORTEX_RV64_MARCH: &str = "rv64imfd_xvortex";
const VORTEX_CONFIG_MANIFEST_SCHEMA: &str = "mandrel.hardware.vortex-config-manifest.v2";
const VORTEX_RTLSIM_REALIZATION_PROFILE: &str = "verilator_rtlsim";
const VORTEX_RTLSIM_PROFILE_DEFINES: [&str; 2] = ["-DSIMULATION", "-DSV_DPI"];
// The probe is inspected but never executed. Keep address-zero weak libc symbols within medany range.
const VORTEX_RV64_STARTUP_PROBE_ADDR: &str = "0x40000000";
const VORTEX_RV64_STARTUP_ADDR: &str = "0x180000000";

pub type VortexToolchainResult<T> = Result<T, VortexToolchainError>;

#[derive(Debug, Snafu)]
pub enum VortexToolchainError {
    #[snafu(display("unsupported MANDREL_VORTEX_XLEN '{xlen}'; use 32 or 64"))]
    UnsupportedXlen { xlen: u32 },
    #[snafu(display(
        "Vortex MLIR artifact pipeline currently targets rv64; got MANDREL_VORTEX_XLEN={xlen}"
    ))]
    UnsupportedMlirXlen { xlen: u32 },

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
    #[snafu(display("invalid materialized Vortex config identity '{}': {message}", path.display()))]
    InvalidConfigIdentity { path: PathBuf, message: String },
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
    #[snafu(display("materialized Vortex configuration is missing resolved ISA flag '{name}'"))]
    MissingVortexIsaConfig { name: String },
    #[snafu(display(
        "generated ELF '{}' has no readable RISC-V arch build attribute; llvm-readelf stdout:\n{stdout}",
        elf_path.display()
    ))]
    MissingElfIsaAttribute { elf_path: PathBuf, stdout: String },
    #[snafu(display(
        "generated ELF '{}' ISA '{elf_arch}' is incompatible with materialized RTL ISA '{rtl_isa}': {mismatches}",
        elf_path.display()
    ))]
    IncompatibleElfIsa {
        elf_path: PathBuf,
        elf_arch: String,
        rtl_isa: String,
        mismatches: String,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterializedVortexConfigIdentity {
    pub sha256: String,
    pub tag: u64,
}

#[derive(Debug, Deserialize)]
struct MaterializedVortexConfigManifest {
    schema: String,
    xlen: u32,
    resolution: VortexConfigResolution,
    resolved_sha256: String,
    config_tag: String,
    defines: BTreeMap<String, VortexConfigDefineValue>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct VortexConfigResolution {
    profile: String,
    generator_cflags: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
enum VortexConfigDefineValue {
    Boolean(bool),
    Number(serde_json::Number),
    String(String),
}

#[derive(Serialize)]
struct CanonicalResolvedVortexConfig<'a> {
    defines: &'a BTreeMap<String, VortexConfigDefineValue>,
    xlen: u32,
}

#[derive(Debug)]
struct MaterializedVortexConfig {
    identity: MaterializedVortexConfigIdentity,
    cflags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VortexConfig {
    pub source_dir: PathBuf,
    pub build_dir: PathBuf,
    pub tool_dir: PathBuf,
    pub verilator_dir: PathBuf,
    pub python_venv_dir: PathBuf,
    pub xlen: u32,
}

impl VortexConfig {
    pub fn from_env(workspace_root: &Path) -> VortexToolchainResult<Self> {
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
        let verilator_dir = project_path_from_env(
            workspace_root,
            "MANDREL_VERILATOR_DIR",
            DEFAULT_VERILATOR_DIR,
        );
        let python_venv_dir = project_path_from_env(
            workspace_root,
            "MANDREL_PYTHON_VENV_DIR",
            DEFAULT_PYTHON_VENV_DIR,
        );
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
            verilator_dir,
            python_venv_dir,
            xlen,
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

    pub fn python_binary(&self) -> PathBuf {
        self.python_venv_dir.join("bin/python")
    }

    pub fn mandrel_config_manifest_path(&self) -> PathBuf {
        self.build_dir.join("mandrel/vortex-config.json")
    }

    pub fn mandrel_config_sha256_path(&self) -> PathBuf {
        self.build_dir.join("mandrel/vortex-config.sha256")
    }

    pub fn mandrel_config_tag_path(&self) -> PathBuf {
        self.build_dir.join("mandrel/vortex-config.tag")
    }

    pub fn materialized_config_identity(
        &self,
    ) -> VortexToolchainResult<MaterializedVortexConfigIdentity> {
        Ok(self.materialized_config()?.identity)
    }

    fn materialized_config_cflags(&self) -> VortexToolchainResult<Vec<String>> {
        Ok(self.materialized_config()?.cflags)
    }

    fn materialized_config(&self) -> VortexToolchainResult<MaterializedVortexConfig> {
        let manifest_path = self.mandrel_config_manifest_path();
        let sha256_path = self.mandrel_config_sha256_path();
        let tag_path = self.mandrel_config_tag_path();
        require_file(
            &manifest_path,
            "Mandrel Vortex config manifest; run scripts/env/setup.sh vortex",
        )?;
        let manifest_json = read_trimmed_file(&manifest_path)?;
        let sha256 = read_trimmed_file(&sha256_path)?;
        let tag = read_trimmed_file(&tag_path)?;
        parse_materialized_config(
            &manifest_path,
            &manifest_json,
            &sha256_path,
            &sha256,
            &tag_path,
            &tag,
            self.xlen,
        )
    }
}

fn read_trimmed_file(path: &Path) -> VortexToolchainResult<String> {
    fs::read_to_string(path)
        .map(|contents| contents.trim().to_owned())
        .map_err(|source| VortexToolchainError::ReadFile {
            path: path.to_path_buf(),
            source,
        })
}

fn parse_materialized_config(
    manifest_path: &Path,
    manifest_json: &str,
    sha256_path: &Path,
    sha256: &str,
    tag_path: &Path,
    tag: &str,
    expected_xlen: u32,
) -> VortexToolchainResult<MaterializedVortexConfig> {
    let manifest: MaterializedVortexConfigManifest =
        serde_json::from_str(manifest_json).map_err(|error| {
            VortexToolchainError::InvalidConfigIdentity {
                path: manifest_path.to_path_buf(),
                message: format!("cannot parse config manifest JSON: {error}"),
            }
        })?;
    if manifest.schema != VORTEX_CONFIG_MANIFEST_SCHEMA {
        return Err(VortexToolchainError::InvalidConfigIdentity {
            path: manifest_path.to_path_buf(),
            message: format!(
                "expected schema {VORTEX_CONFIG_MANIFEST_SCHEMA}, got {}",
                manifest.schema
            ),
        });
    }
    if manifest.xlen != expected_xlen {
        return Err(VortexToolchainError::InvalidConfigIdentity {
            path: manifest_path.to_path_buf(),
            message: format!(
                "manifest XLEN {} does not match configured XLEN {expected_xlen}",
                manifest.xlen
            ),
        });
    }
    let expected_resolution = VortexConfigResolution {
        profile: VORTEX_RTLSIM_REALIZATION_PROFILE.to_owned(),
        generator_cflags: VORTEX_RTLSIM_PROFILE_DEFINES
            .iter()
            .map(|flag| (*flag).to_owned())
            .chain([format!("-DVX_CFG_XLEN={expected_xlen}")])
            .collect(),
    };
    if manifest.resolution != expected_resolution {
        return Err(VortexToolchainError::InvalidConfigIdentity {
            path: manifest_path.to_path_buf(),
            message: format!(
                "expected RTLSim resolution {:?}, got {:?}",
                expected_resolution, manifest.resolution
            ),
        });
    }

    let canonical = serde_json::to_vec(&CanonicalResolvedVortexConfig {
        defines: &manifest.defines,
        xlen: manifest.xlen,
    })
    .map_err(|error| VortexToolchainError::InvalidConfigIdentity {
        path: manifest_path.to_path_buf(),
        message: format!("cannot canonicalize resolved config: {error}"),
    })?;
    let computed_sha256 = format!("{:x}", Sha256::digest(canonical));
    if manifest.resolved_sha256 != computed_sha256 {
        return Err(VortexToolchainError::InvalidConfigIdentity {
            path: manifest_path.to_path_buf(),
            message: format!(
                "manifest resolved SHA-256 {} does not match recomputed {computed_sha256}",
                manifest.resolved_sha256
            ),
        });
    }
    if sha256 != manifest.resolved_sha256 {
        return Err(VortexToolchainError::InvalidConfigIdentity {
            path: sha256_path.to_path_buf(),
            message: "SHA-256 sidecar does not match config manifest".to_owned(),
        });
    }
    if tag != manifest.config_tag {
        return Err(VortexToolchainError::InvalidConfigIdentity {
            path: tag_path.to_path_buf(),
            message: "tag sidecar does not match config manifest".to_owned(),
        });
    }
    let identity = parse_materialized_config_identity(
        manifest_path,
        &manifest.resolved_sha256,
        manifest_path,
        &manifest.config_tag,
    )?;
    let cflags = manifest
        .defines
        .iter()
        .map(|(name, value)| render_config_define(manifest_path, name, value))
        .collect::<VortexToolchainResult<Vec<_>>>()?;

    Ok(MaterializedVortexConfig { identity, cflags })
}

fn render_config_define(
    manifest_path: &Path,
    name: &str,
    value: &VortexConfigDefineValue,
) -> VortexToolchainResult<String> {
    let mut bytes = name.bytes();
    let valid_start = bytes
        .next()
        .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_');
    if !valid_start || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_') {
        return Err(VortexToolchainError::InvalidConfigIdentity {
            path: manifest_path.to_path_buf(),
            message: format!("invalid resolved define name {name:?}"),
        });
    }
    let rendered = match value {
        VortexConfigDefineValue::Boolean(true) => format!("-D{name}"),
        VortexConfigDefineValue::Boolean(false) => format!("-D{name}=false"),
        VortexConfigDefineValue::Number(number) if number.is_i64() || number.is_u64() => {
            format!("-D{name}={number}")
        }
        VortexConfigDefineValue::Number(number) => {
            return Err(VortexToolchainError::InvalidConfigIdentity {
                path: manifest_path.to_path_buf(),
                message: format!("resolved define {name} has non-integer number {number}"),
            });
        }
        VortexConfigDefineValue::String(value) => format!("-D{name}={value}"),
    };
    Ok(rendered)
}

fn parse_materialized_config_identity(
    sha256_path: &Path,
    sha256: &str,
    tag_path: &Path,
    tag: &str,
) -> VortexToolchainResult<MaterializedVortexConfigIdentity> {
    if sha256.len() != 64
        || !sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(VortexToolchainError::InvalidConfigIdentity {
            path: sha256_path.to_path_buf(),
            message: "expected 64 lowercase hexadecimal SHA-256 characters".to_owned(),
        });
    }
    if tag.len() != 16
        || !tag
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(VortexToolchainError::InvalidConfigIdentity {
            path: tag_path.to_path_buf(),
            message: "expected 16 lowercase hexadecimal tag characters".to_owned(),
        });
    }
    if !sha256.starts_with(tag) {
        return Err(VortexToolchainError::InvalidConfigIdentity {
            path: tag_path.to_path_buf(),
            message: "tag is not the leading 64 bits of the resolved config SHA-256".to_owned(),
        });
    }
    let tag = u64::from_str_radix(tag, 16).map_err(|error| {
        VortexToolchainError::InvalidConfigIdentity {
            path: tag_path.to_path_buf(),
            message: format!("cannot parse hexadecimal tag: {error}"),
        }
    })?;
    Ok(MaterializedVortexConfigIdentity {
        sha256: sha256.to_owned(),
        tag,
    })
}

pub trait VortexCommandRunner {
    fn run(&mut self, phase: &str, command: Command) -> VortexToolchainResult<()>;

    fn output(&mut self, phase: &str, command: Command) -> VortexToolchainResult<Output>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VortexKernelBuildOutputs {
    pub mlir_path: PathBuf,
    pub ll_path: PathBuf,
    pub obj_path: PathBuf,
    pub startup_probe_elf_path: PathBuf,
    pub startup_object_path: PathBuf,
    pub elf_path: PathBuf,
    pub vxbin_path: PathBuf,
}

impl VortexKernelBuildOutputs {
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
    pub outputs: &'a VortexKernelBuildOutputs,
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
        &request.outputs.mlir_path,
        "generated Vortex output directory",
    )?;
    fs::write(&request.outputs.mlir_path, request.source).map_err(|source| {
        VortexToolchainError::WriteFile {
            path: request.outputs.mlir_path.clone(),
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
            kernel_object_path: &request.outputs.obj_path,
            startup_probe_elf_path: &request.outputs.startup_probe_elf_path,
            startup_object_path: &request.outputs.startup_object_path,
            elf_path: &request.outputs.elf_path,
            kentry_symbol: &kentry_symbol,
            phase_prefix: request.phase_prefix,
        },
        runner,
    )?;
    verify_vortex_kernel_elf_contains_symbol(
        request.workspace_root,
        request.config,
        &request.outputs.elf_path,
        &kentry_symbol,
        request.phase_prefix,
        runner,
    )?;
    verify_vortex_kernel_elf_isa_compatibility(
        request.workspace_root,
        request.config,
        &request.outputs.elf_path,
        request.phase_prefix,
        runner,
    )?;
    package_vortex_elf_to_vxbin(
        VortexVxbinPackageRequest {
            workspace_root: request.workspace_root,
            config: request.config,
            elf_path: &request.outputs.elf_path,
            vxbin_path: &request.outputs.vxbin_path,
            phase_prefix: request.phase_prefix,
        },
        runner,
    )?;
    verify_vortex_vxbin_symbols(&request.outputs.vxbin_path, request.symbol_name)
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
        VortexKernelLinkCommandRequest {
            workspace_root: request.workspace_root,
            config: request.config,
            tools: &tools,
            input_objects: [request.kernel_object_path],
            kentry_symbol: request.kentry_symbol,
            startup_addr: VORTEX_RV64_STARTUP_PROBE_ADDR,
            output_elf: request.startup_probe_elf_path,
            phase: &prefixed_phase(request.phase_prefix, "clang_link_startup_probe_elf"),
        },
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
        VortexKernelLinkCommandRequest {
            workspace_root: request.workspace_root,
            config: request.config,
            tools: &tools,
            input_objects: [request.startup_object_path, request.kernel_object_path],
            kentry_symbol: request.kentry_symbol,
            startup_addr: VORTEX_RV64_STARTUP_ADDR,
            output_elf: request.elf_path,
            phase: &prefixed_phase(request.phase_prefix, "clang_link_elf"),
        },
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

    let mut vxbin = Command::new(request.config.python_binary());
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

pub fn verify_vortex_kernel_elf_isa_compatibility<R>(
    workspace_root: &Path,
    config: &VortexConfig,
    elf_path: &Path,
    phase_prefix: &str,
    runner: &mut R,
) -> VortexToolchainResult<()>
where
    R: VortexCommandRunner,
{
    let readelf = config.tool_dir.join("llvm-vortex/bin/llvm-readelf");
    require_file(&readelf, "Vortex llvm-readelf")?;
    require_file(elf_path, "generated Vortex ELF")?;

    let config_flags = config.materialized_config_cflags()?;
    let rtl_isa = VortexRtlIsa::from_config_flags(config.xlen, &config_flags)?;

    let mut readelf_cmd = Command::new(&readelf);
    readelf_cmd
        .current_dir(workspace_root)
        .arg("-A")
        .arg(elf_path);
    apply_vortex_command_env(&mut readelf_cmd, config)?;
    let phase = prefixed_phase(phase_prefix, "elf_readelf_attributes");
    let output = runner.output(&phase, readelf_cmd)?;
    let stdout = String::from_utf8(output.stdout)
        .map_err(|source| VortexToolchainError::NonUtf8Output { phase, source })?;
    let elf_arch = parse_riscv_elf_arch_attribute(&stdout).ok_or_else(|| {
        VortexToolchainError::MissingElfIsaAttribute {
            elf_path: elf_path.to_path_buf(),
            stdout: stdout.clone(),
        }
    })?;
    let elf_isa = RiscvElfIsa::parse(elf_arch).ok_or_else(|| {
        VortexToolchainError::MissingElfIsaAttribute {
            elf_path: elf_path.to_path_buf(),
            stdout: stdout.clone(),
        }
    })?;
    let mismatches = elf_isa.mismatches(&rtl_isa);
    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(VortexToolchainError::IncompatibleElfIsa {
            elf_path: elf_path.to_path_buf(),
            elf_arch: elf_arch.to_owned(),
            rtl_isa: rtl_isa.summary(),
            mismatches: mismatches.join(", "),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RiscvIsaExtensions {
    xlen: u32,
    m: bool,
    a: bool,
    f: bool,
    d: bool,
    c: bool,
    v: bool,
    zicond: bool,
    xvortex: bool,
}

type RiscvElfIsa = RiscvIsaExtensions;
type VortexRtlIsa = RiscvIsaExtensions;

impl RiscvIsaExtensions {
    fn parse(arch: &str) -> Option<Self> {
        let mut tokens = arch.split('_');
        let base = tokens.next()?;
        let xlen = if base.starts_with("rv64") {
            64
        } else if base.starts_with("rv32") {
            32
        } else {
            return None;
        };
        let extensions = tokens.collect::<Vec<_>>();
        Some(Self {
            xlen,
            m: has_versioned_single_letter_extension(&extensions, "m"),
            a: has_versioned_single_letter_extension(&extensions, "a"),
            f: has_versioned_single_letter_extension(&extensions, "f"),
            d: has_versioned_single_letter_extension(&extensions, "d"),
            c: has_versioned_single_letter_extension(&extensions, "c"),
            v: has_versioned_single_letter_extension(&extensions, "v"),
            zicond: extensions.iter().any(|token| token.starts_with("zicond")),
            xvortex: extensions.iter().any(|token| token.starts_with("xvortex")),
        })
    }

    fn from_config_flags(xlen: u32, flags: &[String]) -> VortexToolchainResult<Self> {
        Ok(Self {
            xlen,
            m: resolved_vortex_extension(flags, "VX_CFG_EXT_M_ENABLED")?,
            a: resolved_vortex_extension(flags, "VX_CFG_EXT_A_ENABLED")?,
            f: resolved_vortex_extension(flags, "VX_CFG_EXT_F_ENABLED")?,
            d: resolved_vortex_extension(flags, "VX_CFG_EXT_D_ENABLED")?,
            c: resolved_vortex_extension(flags, "VX_CFG_EXT_C_ENABLED")?,
            v: resolved_vortex_extension(flags, "VX_CFG_EXT_V_ENABLED")?,
            zicond: resolved_vortex_extension(flags, "VX_CFG_EXT_ZICOND_ENABLED")?,
            xvortex: true,
        })
    }

    fn mismatches(&self, rtl: &Self) -> Vec<String> {
        let mut mismatches = Vec::new();
        if self.xlen != rtl.xlen {
            mismatches.push(format!("ELF XLEN={} but RTL XLEN={}", self.xlen, rtl.xlen));
        }
        for (name, required, available) in [
            ("M", self.m, rtl.m),
            ("A", self.a, rtl.a),
            ("F", self.f, rtl.f),
            ("D", self.d, rtl.d),
            ("C", self.c, rtl.c),
            ("V", self.v, rtl.v),
            ("Zicond", self.zicond, rtl.zicond),
        ] {
            if required && !available {
                mismatches.push(format!("ELF requires {name} but RTL disables it"));
            }
        }
        if !self.xvortex {
            mismatches.push(String::from(
                "ELF does not declare required xvortex extension",
            ));
        }
        mismatches
    }

    fn summary(&self) -> String {
        let mut summary = format!("rv{}i", self.xlen);
        for (enabled, name) in [
            (self.m, "m"),
            (self.a, "a"),
            (self.f, "f"),
            (self.d, "d"),
            (self.c, "c"),
            (self.v, "v"),
        ] {
            if enabled {
                summary.push_str(name);
            }
        }
        if self.zicond {
            summary.push_str("_zicond");
        }
        if self.xvortex {
            summary.push_str("_xvortex");
        }
        summary
    }
}

fn parse_riscv_elf_arch_attribute(stdout: &str) -> Option<&str> {
    stdout.lines().find_map(|line| {
        let value = line.trim().strip_prefix("Value:")?.trim();
        value.starts_with("rv").then_some(value)
    })
}

fn has_versioned_single_letter_extension(extensions: &[&str], name: &str) -> bool {
    extensions.iter().any(|extension| {
        extension
            .strip_prefix(name)
            .is_some_and(|version| version.as_bytes().first().is_some_and(u8::is_ascii_digit))
    })
}

fn resolved_vortex_extension(flags: &[String], name: &str) -> VortexToolchainResult<bool> {
    let prefix = format!("-D{name}=");
    flags
        .iter()
        .find_map(|flag| flag.strip_prefix(&prefix))
        .and_then(|value| match value {
            "0" => Some(false),
            "1" => Some(true),
            _ => None,
        })
        .ok_or_else(|| VortexToolchainError::MissingVortexIsaConfig {
            name: name.to_owned(),
        })
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
    command.env("VERILATOR_PATH", &config.verilator_dir);
    command.env("VORTEX_DRIVER", "rtlsim");
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
        [
            config.python_venv_dir.join("bin"),
            config.verilator_dir.join("bin"),
            config.tool_dir.join("llvm-vortex/bin"),
            config.tool_dir.join("riscv64-gnu-toolchain/bin"),
            config.bin_dir(),
        ],
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
        .arg(&request.outputs.mlir_path)
        .arg("-o")
        .arg(&request.outputs.ll_path);
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
        .arg(format!("-march={VORTEX_RV64_MARCH}"))
        .arg("-O1")
        .arg(&request.outputs.ll_path)
        .arg("-o")
        .arg(&request.outputs.obj_path);
    apply_vortex_command_env(&mut clang_cmd, request.config)?;
    runner.run(
        &prefixed_phase(request.phase_prefix, "clang_object"),
        clang_cmd,
    )
}

struct VortexKernelLinkCommandRequest<'a, I> {
    workspace_root: &'a Path,
    config: &'a VortexConfig,
    tools: &'a VortexRv64KernelLinkTools,
    input_objects: I,
    kentry_symbol: &'a str,
    startup_addr: &'a str,
    output_elf: &'a Path,
    phase: &'a str,
}

fn run_vortex_kernel_link_command<'a, R, I>(
    request: VortexKernelLinkCommandRequest<'a, I>,
    runner: &mut R,
) -> VortexToolchainResult<()>
where
    R: VortexCommandRunner,
    I: IntoIterator<Item = &'a Path>,
{
    let riscv_sysroot = request
        .config
        .tool_dir
        .join("riscv64-gnu-toolchain/riscv64-unknown-elf");
    let riscv_toolchain = request.config.tool_dir.join("riscv64-gnu-toolchain");

    let mut clang_cmd = Command::new(&request.tools.clang);
    clang_cmd
        .current_dir(request.workspace_root)
        .arg("-target")
        .arg("riscv64-unknown-unknown-elf")
        .arg(format!("--sysroot={}", riscv_sysroot.display()))
        .arg(format!("--gcc-toolchain={}", riscv_toolchain.display()))
        .arg(format!("-march={VORTEX_RV64_MARCH}"))
        .arg("-mabi=lp64d")
        .arg("-mcmodel=medany")
        .arg("-fuse-ld=lld")
        .arg("-nostartfiles")
        .arg("-nostdlib");
    for object in request.input_objects {
        clang_cmd.arg(object);
    }
    clang_cmd
        .arg(format!(
            "-Wl,-Bstatic,--gc-sections,-T,{},--defsym=STARTUP_ADDR={},--undefined={}",
            request.tools.link_script.display(),
            request.startup_addr,
            request.kentry_symbol
        ))
        .arg(&request.tools.kernel_runtime)
        .arg(format!("-L{}", request.tools.libc_dir.display()))
        .arg("-lm")
        .arg("-lc")
        .arg(&request.tools.builtins)
        .arg("-o")
        .arg(request.output_elf);
    apply_vortex_command_env(&mut clang_cmd, request.config)?;
    runner.run(request.phase, clang_cmd)
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
    let config_flags = config.materialized_config_cflags()?;

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
        .arg(format!("-march={VORTEX_RV64_MARCH}"))
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

#[cfg(test)]
mod tests {
    use super::{
        MaterializedVortexConfigIdentity, RiscvIsaExtensions, VortexConfig,
        VortexKernelBuildOutputs, VortexToolchainError, parse_materialized_config,
        parse_materialized_config_identity, parse_riscv_elf_arch_attribute,
    };
    use std::path::Path;

    #[test]
    fn config_describes_project_local_materialized_tools() {
        let root = Path::new("/tmp/mandrel");
        let config = match VortexConfig::from_env(root) {
            Ok(config) => config,
            Err(error) => panic!("unexpected config error: {error}"),
        };

        assert_eq!(config.source_dir, root.join("external/vortex"));
        assert_eq!(config.build_dir, root.join("external/vortex-build"));
        assert_eq!(config.tool_dir, root.join("external/vortex-source-tools"));
        assert_eq!(
            config.verilator_dir,
            root.join("external/verilator-install")
        );
        assert_eq!(config.python_binary(), root.join(".venv/bin/python"));
    }

    #[test]
    fn parses_materialized_vortex_config_identity() {
        let sha = "1d971b9b230fa24aa514c0a6855be71a64d6f726f96532c3e26e888c019f5d9b";
        let identity = match parse_materialized_config_identity(
            Path::new("vortex-config.sha256"),
            sha,
            Path::new("vortex-config.tag"),
            "1d971b9b230fa24a",
        ) {
            Ok(identity) => identity,
            Err(error) => panic!("unexpected identity parse error: {error}"),
        };

        assert_eq!(
            identity,
            MaterializedVortexConfigIdentity {
                sha256: sha.to_owned(),
                tag: 0x1d97_1b9b_230f_a24a,
            }
        );
    }

    #[test]
    fn validates_manifest_digest_profile_sidecars_and_resolved_cflags() {
        let sha = "07f1562729fc678b42ae58e8cd4f7f493b21260e33044a838e0b73f5e3f97a9b";
        let manifest = format!(
            "{{\"schema\":\"mandrel.hardware.vortex-config-manifest.v2\",\"xlen\":64,\"resolution\":{{\"profile\":\"verilator_rtlsim\",\"generator_cflags\":[\"-DSIMULATION\",\"-DSV_DPI\",\"-DVX_CFG_XLEN=64\"]}},\"resolved_sha256\":\"{sha}\",\"config_tag\":\"07f1562729fc678b\",\"defines\":{{\"VX_CFG_XLEN\":64,\"VX_CFG_EXT_M_ENABLED\":1}}}}"
        );
        let config = match parse_materialized_config(
            Path::new("vortex-config.json"),
            &manifest,
            Path::new("vortex-config.sha256"),
            sha,
            Path::new("vortex-config.tag"),
            "07f1562729fc678b",
            64,
        ) {
            Ok(config) => config,
            Err(error) => panic!("unexpected materialized config error: {error}"),
        };

        assert_eq!(config.identity.sha256, sha);
        assert_eq!(config.identity.tag, 0x07f1_5627_29fc_678b);
        assert_eq!(
            config.cflags,
            ["-DVX_CFG_EXT_M_ENABLED=1", "-DVX_CFG_XLEN=64"]
        );
    }

    #[test]
    fn rejects_manifest_sidecar_drift() {
        let sha = "07f1562729fc678b42ae58e8cd4f7f493b21260e33044a838e0b73f5e3f97a9b";
        let manifest = format!(
            "{{\"schema\":\"mandrel.hardware.vortex-config-manifest.v2\",\"xlen\":64,\"resolution\":{{\"profile\":\"verilator_rtlsim\",\"generator_cflags\":[\"-DSIMULATION\",\"-DSV_DPI\",\"-DVX_CFG_XLEN=64\"]}},\"resolved_sha256\":\"{sha}\",\"config_tag\":\"07f1562729fc678b\",\"defines\":{{\"VX_CFG_XLEN\":64,\"VX_CFG_EXT_M_ENABLED\":1}}}}"
        );
        let error = parse_materialized_config(
            Path::new("vortex-config.json"),
            &manifest,
            Path::new("vortex-config.sha256"),
            "17f1562729fc678b42ae58e8cd4f7f493b21260e33044a838e0b73f5e3f97a9b",
            Path::new("vortex-config.tag"),
            "07f1562729fc678b",
            64,
        );

        assert!(matches!(
            error,
            Err(VortexToolchainError::InvalidConfigIdentity { .. })
        ));
    }

    #[test]
    fn rejects_materialized_config_tag_that_does_not_match_sha() {
        let error = parse_materialized_config_identity(
            Path::new("vortex-config.sha256"),
            "1d971b9b230fa24aa514c0a6855be71a64d6f726f96532c3e26e888c019f5d9b",
            Path::new("vortex-config.tag"),
            "2d971b9b230fa24a",
        );

        assert!(matches!(
            error,
            Err(VortexToolchainError::InvalidConfigIdentity { .. })
        ));
    }

    #[test]
    fn kernel_build_output_paths_follow_kernel_symbol() {
        let out_dir = Path::new("target/mandrel/vortex");
        let outputs = VortexKernelBuildOutputs::under_output_dir(out_dir, "attention_prefill_i8");

        assert_eq!(
            outputs,
            VortexKernelBuildOutputs {
                mlir_path: out_dir.join("attention_prefill_i8.mlir"),
                ll_path: out_dir.join("attention_prefill_i8.ll"),
                obj_path: out_dir.join("attention_prefill_i8.o"),
                startup_probe_elf_path: out_dir.join("attention_prefill_i8.startup_probe.elf"),
                startup_object_path: out_dir.join("attention_prefill_i8.vx_start.o"),
                elf_path: out_dir.join("attention_prefill_i8.elf"),
                vxbin_path: out_dir.join("attention_prefill_i8.vxbin"),
            }
        );
    }

    #[test]
    fn parses_llvm_readelf_riscv_arch_attribute() {
        let stdout = "BuildAttributes {\n  Value: 16\n  Value: rv64i2p1_m2p0_f2p2_d2p2_zicsr2p0_xvortex1p0\n}\n";
        assert_eq!(
            parse_riscv_elf_arch_attribute(stdout),
            Some("rv64i2p1_m2p0_f2p2_d2p2_zicsr2p0_xvortex1p0")
        );
    }

    #[test]
    fn accepts_elf_isa_subset_of_materialized_rtl() {
        let flags = resolved_rtl_flags();
        let rtl = match RiscvIsaExtensions::from_config_flags(64, &flags) {
            Ok(isa) => isa,
            Err(error) => panic!("unexpected RTL ISA parse error: {error}"),
        };
        let elf =
            match RiscvIsaExtensions::parse("rv64i2p1_m2p0_f2p2_d2p2_zicsr2p0_zmmul1p0_xvortex1p0")
            {
                Some(isa) => isa,
                None => panic!("expected ELF ISA to parse"),
            };

        assert!(elf.mismatches(&rtl).is_empty());
    }

    #[test]
    fn rejects_elf_extensions_disabled_in_materialized_rtl() {
        let flags = resolved_rtl_flags();
        let rtl = match RiscvIsaExtensions::from_config_flags(64, &flags) {
            Ok(isa) => isa,
            Err(error) => panic!("unexpected RTL ISA parse error: {error}"),
        };
        let elf = match RiscvIsaExtensions::parse(
            "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zaamo1p0_zalrsc1p0_xvortex1p0",
        ) {
            Some(isa) => isa,
            None => panic!("expected ELF ISA to parse"),
        };

        assert_eq!(
            elf.mismatches(&rtl),
            [
                String::from("ELF requires A but RTL disables it"),
                String::from("ELF requires C but RTL disables it"),
            ]
        );
    }

    #[test]
    fn requires_resolved_rtl_extension_flags() {
        let error = match RiscvIsaExtensions::from_config_flags(64, &[]) {
            Ok(_) => panic!("expected missing config flag to fail"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            VortexToolchainError::MissingVortexIsaConfig { name }
                if name == "VX_CFG_EXT_M_ENABLED"
        ));
    }

    fn resolved_rtl_flags() -> Vec<String> {
        [
            "-DVX_CFG_EXT_M_ENABLED=1",
            "-DVX_CFG_EXT_A_ENABLED=0",
            "-DVX_CFG_EXT_F_ENABLED=1",
            "-DVX_CFG_EXT_D_ENABLED=1",
            "-DVX_CFG_EXT_C_ENABLED=0",
            "-DVX_CFG_EXT_V_ENABLED=0",
            "-DVX_CFG_EXT_ZICOND_ENABLED=1",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }
}
