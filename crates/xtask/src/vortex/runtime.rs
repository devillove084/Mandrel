use std::env;
use std::path::{Path, PathBuf};

use mandrel_vortex_backend::VortexConfig;

use crate::Result;

const VORTEX_SETUP_HINT: &str =
    "run `scripts/env/setup.sh vortex` to materialize the pinned Verilator RTLSim runtime";

pub(crate) fn require_vortex_runtime_libraries(config: &VortexConfig) -> Result<()> {
    let required = [
        (
            config.build_dir.join("sw/runtime/libvortex.so"),
            "Vortex runtime libvortex.so",
        ),
        (
            config.build_dir.join("sw/runtime/libvortex-rtlsim.so"),
            "Vortex RTLSim runtime driver libvortex-rtlsim.so",
        ),
        (
            config.build_dir.join("sw/runtime/librtlsim.so"),
            "Vortex RTLSim runtime core librtlsim.so",
        ),
    ];
    let missing = required
        .iter()
        .filter(|(path, _)| !path.is_file())
        .map(|(path, description)| format!("{description}: {}", path.display()))
        .collect::<Vec<_>>();

    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "missing required Vortex RTLSim runtime artifacts:\n  {}\n{VORTEX_SETUP_HINT}",
            missing.join("\n  ")
        )
        .into())
    }
}

pub(crate) fn preferred_vortex_runtime_library(config: &VortexConfig) -> Result<PathBuf> {
    let candidates = vortex_runtime_library_candidates(config);
    for candidate in &candidates {
        if candidate.is_file() {
            return Ok(candidate.clone());
        }
    }
    Err(format!(
        "no Vortex runtime library found; tried: {}; {VORTEX_SETUP_HINT}",
        candidates
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
    .into())
}

fn vortex_runtime_library_candidates(config: &VortexConfig) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = env::var_os("MANDREL_VORTEX_RUNTIME_LIB").map(PathBuf::from) {
        candidates.push(path);
    }
    candidates.push(config.build_dir.join("sw/runtime/libvortex.so"));
    candidates.push(config.install_dir().join("runtime/lib/libvortex.so"));
    candidates
}

pub(crate) fn require_file(path: &Path, description: &str) -> Result<()> {
    if path.is_file() {
        Ok(())
    } else {
        Err(format!("missing {description}: {}", path.display()).into())
    }
}
