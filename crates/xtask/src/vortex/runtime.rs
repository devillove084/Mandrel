use super::*;

pub(crate) fn run_vortex_vecadd(workspace_root: &Path) -> Result<()> {
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
pub(crate) fn ensure_vortex_runtime_libraries(config: &VortexConfig) -> Result<()> {
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
pub(crate) fn preferred_vortex_runtime_library(config: &VortexConfig) -> Result<PathBuf> {
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
pub(super) fn vortex_runtime_library_candidates(config: &VortexConfig) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = env::var_os("MANDREL_VORTEX_RUNTIME_LIB").map(PathBuf::from) {
        candidates.push(path);
    }
    candidates.push(config.build_dir.join("sw/runtime/libvortex.so"));
    candidates.push(config.install_dir().join("runtime/lib/libvortex.so"));
    candidates
}

#[allow(dead_code)]
pub(crate) fn require_file(path: &Path, description: &str) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(format!("missing {description}: {}", path.display()).into())
    }
}

#[allow(dead_code)]
pub(super) fn file_contains_bytes(path: &Path, needle: &[u8]) -> Result<bool> {
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read '{}': {error}", path.display()))?;
    Ok(bytes.windows(needle.len()).any(|window| window == needle))
}
