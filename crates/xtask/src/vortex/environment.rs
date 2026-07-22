use std::process::Command;

use mandrel_vortex_backend::{VortexConfig, apply_vortex_command_env};

use crate::Result;

pub(crate) fn apply_vortex_env(command: &mut Command, config: &VortexConfig) -> Result<()> {
    apply_vortex_command_env(command, config)?;
    Ok(())
}
