use super::*;

pub(crate) fn fetch_vortex(workspace_root: &Path) -> Result<()> {
    let config = VortexConfig::from_env(workspace_root)?;
    log_vortex_config(&config, "fetching/verifying Vortex source");
    clone_vortex_if_needed(&config)?;
    checkout_vortex_ref_if_requested(&config)?;
    println!("Vortex checkout ready at: {}", config.source_dir.display());
    Ok(())
}

pub(crate) fn install_vortex(workspace_root: &Path) -> Result<()> {
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

pub(super) fn log_vortex_config(config: &VortexConfig, message: &str) {
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

pub(super) fn clone_vortex_if_needed(config: &VortexConfig) -> Result<()> {
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

pub(super) fn update_submodules(config: &VortexConfig) -> Result<()> {
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

pub(super) fn configure_git_proxy_for_checkout(config: &VortexConfig) -> Result<()> {
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

pub(super) fn git_config_contains_value(repo_dir: &Path, key: &str, value: &str) -> Result<bool> {
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

pub(super) fn checkout_vortex_ref_if_requested(config: &VortexConfig) -> Result<()> {
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

pub(super) fn configure_and_build_vortex(
    config: &VortexConfig,
    run_prebuilt_toolchain: bool,
) -> Result<()> {
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
pub(super) enum VortexBuildProfile {
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

pub(super) fn build_vortex_with_profile(
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

pub(super) fn build_vortex_software_sdk(config: &VortexConfig) -> Result<()> {
    for subdir in ["sw/kernel", "sw/runtime/stub"] {
        let mut make = Command::new("make");
        make.current_dir(&config.build_dir).arg("-C").arg(subdir);
        apply_vortex_env(&mut make, config)?;
        run_checked(make, &format!("vortex.make.{subdir}"))?;
    }
    Ok(())
}

pub(super) fn build_vortex_simx(config: &VortexConfig) -> Result<()> {
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
