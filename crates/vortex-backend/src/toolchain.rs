use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const DEFAULT_VORTEX_DIR: &str = "external/vortex";
pub const DEFAULT_VORTEX_URL: &str = "https://github.com/vortexgpgpu/vortex.git";
pub const DEFAULT_VORTEX_BUILD_DIR: &str = "external/vortex-build";
pub const DEFAULT_VORTEX_TOOLDIR: &str = "external/vortex-tools";
pub const DEFAULT_VORTEX_SYSTEM_TOOLDIR: &str = "external/vortex-system-tools";
pub const DEFAULT_VORTEX_ENV_FILE: &str = "external/vortex-env.sh";
pub const DEFAULT_FETCH_RETRIES: u32 = 3;
pub const VORTEX_PREBUILT_HOST_ARCH: &str = "x86_64";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VortexToolchainMode {
    Auto,
    Prebuilt,
    System,
    Skip,
}

impl VortexToolchainMode {
    pub fn from_env() -> Result<Self, String> {
        let Some(raw) = non_empty_env("MANDREL_VORTEX_TOOLCHAIN_MODE") else {
            return Ok(Self::default_for_host());
        };

        match raw.as_str() {
            "auto" => Ok(Self::Auto),
            "prebuilt" => Ok(Self::Prebuilt),
            "system" | "ubuntu" => Ok(Self::System),
            "skip" | "external" => Ok(Self::Skip),
            other => Err(format!(
                "unsupported MANDREL_VORTEX_TOOLCHAIN_MODE '{other}'; use auto, prebuilt, system, or skip"
            )),
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
    pub fn from_env(workspace_root: &Path) -> Result<Self, String> {
        let toolchain_mode = VortexToolchainMode::from_env()?;
        let source_dir =
            project_path_from_env(workspace_root, "MANDREL_VORTEX_DIR", DEFAULT_VORTEX_DIR);
        let build_dir = project_path_from_env(
            workspace_root,
            "MANDREL_VORTEX_BUILD_DIR",
            DEFAULT_VORTEX_BUILD_DIR,
        );
        let default_tool_dir = match toolchain_mode {
            VortexToolchainMode::System if env::var_os("MANDREL_VORTEX_TOOLDIR").is_none() => {
                DEFAULT_VORTEX_SYSTEM_TOOLDIR
            }
            _ => DEFAULT_VORTEX_TOOLDIR,
        };
        let tool_dir =
            project_path_from_env(workspace_root, "MANDREL_VORTEX_TOOLDIR", default_tool_dir);
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
            return Err(format!(
                "unsupported MANDREL_VORTEX_XLEN '{xlen}'; use 32 or 64"
            ));
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

    pub fn should_run_prebuilt_toolchain(&self) -> Result<bool, String> {
        match self.toolchain_mode {
            VortexToolchainMode::Prebuilt => Ok(true),
            VortexToolchainMode::System | VortexToolchainMode::Skip => Ok(false),
            VortexToolchainMode::Auto if env::consts::ARCH == VORTEX_PREBUILT_HOST_ARCH => Ok(true),
            VortexToolchainMode::Auto => Err(format!(
                "Vortex upstream toolchain_install.sh fetches host prebuilt packages and this host is '{}', not '{}'. To avoid running incompatible prebuilt binaries, either use Ubuntu/system packages with MANDREL_VORTEX_TOOLCHAIN_MODE=system, source-build/populate '{}' and rerun with MANDREL_VORTEX_TOOLCHAIN_MODE=skip, or force the upstream prebuilt path with MANDREL_VORTEX_TOOLCHAIN_MODE=prebuilt.",
                env::consts::ARCH,
                VORTEX_PREBUILT_HOST_ARCH,
                self.tool_dir.display()
            )),
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

fn parse_fetch_retries() -> Result<u32, String> {
    let Some(raw) = non_empty_env("MANDREL_FETCH_RETRIES") else {
        return Ok(DEFAULT_FETCH_RETRIES);
    };

    let retries = raw
        .parse::<u32>()
        .map_err(|error| format!("invalid MANDREL_FETCH_RETRIES '{raw}': {error}"))?;
    if retries == 0 {
        return Err("MANDREL_FETCH_RETRIES must be at least 1".to_owned());
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

        assert_eq!(config.should_run_prebuilt_toolchain(), Ok(false));
    }
}
