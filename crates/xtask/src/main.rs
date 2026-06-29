use std::path::{Path, PathBuf};

mod attention;
mod cli;
mod command;
mod error;
mod logging;
mod moe;
mod vortex;

use cli::{Cli, XtaskCommand};
pub(crate) use error::{Result, XtaskError};

fn main() {
    let _log_guard = match logging::init_logging() {
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
    let cli = Cli::parse_args();
    let Some(command) = cli.command else {
        Cli::print_help()
            .map_err(|error| XtaskError::message(format!("failed to print help: {error}")))?;
        return Ok(());
    };

    let command_name = command.name();
    let workspace_root = workspace_root()?;
    tracing::info!(command = command_name, root = %workspace_root.display(), "starting xtask command");

    match command {
        XtaskCommand::VortexFetch => vortex::fetch_vortex(&workspace_root)?,
        XtaskCommand::VortexSystemTools => {
            vortex::prepare_and_print_vortex_system_tools(&workspace_root)?
        }
        XtaskCommand::VortexToolchainSource => {
            vortex::install_vortex_source_toolchain(&workspace_root)?
        }
        XtaskCommand::VortexInstall => vortex::install_vortex(&workspace_root)?,
        XtaskCommand::VortexEnv => vortex::write_and_print_vortex_env(&workspace_root)?,
        XtaskCommand::VortexStatus => vortex::print_vortex_status(&workspace_root)?,
        XtaskCommand::VortexPlanAttention => attention::print_attention_prefill_plan()?,
        XtaskCommand::MoeFetchTinyMixtral => moe::fetch_tiny_mixtral(&workspace_root)?,
        XtaskCommand::MoeRunReference { hf_model_dir } => {
            moe::run_moe_reference_baseline(hf_model_dir)?
        }
        XtaskCommand::MoeRunLayerReference { hf_model_dir } => {
            moe::run_moe_layer_reference_baseline(hf_model_dir)?
        }
        XtaskCommand::VortexGenerateAttention => {
            attention::generate_vortex_attention_kernel_source(&workspace_root)?
        }
        XtaskCommand::VortexRunAttention => {
            attention::run_vortex_attention_correctness(&workspace_root)?
        }
        XtaskCommand::VortexTraceAttention => {
            attention::print_attention_trace_history(&workspace_root)?
        }
        XtaskCommand::VortexRunAttentionInner { vxbin } => {
            attention::run_vortex_attention_correctness_inner(&workspace_root, vxbin)?
        }
        XtaskCommand::VortexRunVecadd => vortex::run_vortex_vecadd(&workspace_root)?,
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
