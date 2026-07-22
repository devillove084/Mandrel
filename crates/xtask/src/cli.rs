use std::io;
use std::path::PathBuf;

use clap::{CommandFactory, Parser, Subcommand};

const AFTER_LONG_HELP: &str = "\
Main backend target:
  Vortex RISC-V GPGPU through Verilator RTLSim

Environment preparation:
  scripts/env/setup.sh all
  source scripts/env/vortex-env.sh

xtask scope:
  Generate Mandrel MLIR/LLVM/object/ELF/vxbin artifacts, compile typed launch plans,
  execute operators, check exact correctness, and emit profile reports.
  Checkout, patching, uv sync, and toolchain/RTLSim builds belong to scripts/env/.

Slow Hugging Face downloads:
  MANDREL_HF_ENDPOINT=https://hf-mirror.com cargo xtask moe-fetch-tiny-mixtral

Logging:
  MANDREL_LOG=info|debug|trace or tracing filter syntax; default: info
  MANDREL_LOG_FORMAT=compact|pretty|json; default: compact
  MANDREL_LOG_FILE=logs/xtask.log writes logs to a file instead of console";

#[derive(Debug, Parser)]
#[command(
    name = "xtask",
    bin_name = "cargo xtask",
    version,
    about = "Mandrel project automation",
    long_about = "Mandrel operator compilation, Vortex RTLSim execution, attention profiling, and tiny MoE reference workflows.",
    after_long_help = AFTER_LONG_HELP
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Option<XtaskCommand>,
}

impl Cli {
    pub(crate) fn parse_args() -> Self {
        Self::parse()
    }

    pub(crate) fn print_help() -> io::Result<()> {
        Self::command().print_help()?;
        println!();
        Ok(())
    }
}

#[derive(Debug, Subcommand)]
pub(crate) enum XtaskCommand {
    #[command(about = "Print dense attention-prefill online-softmax plan")]
    VortexPlanAttention,

    #[command(about = "Download tiny Mixtral fixture using MANDREL_HF_ENDPOINT")]
    MoeFetchTinyMixtral,

    #[command(about = "Run deterministic or local HF tiny MoE reference baseline")]
    MoeRunReference {
        #[arg(value_name = "HF_MODEL_DIR")]
        hf_model_dir: Option<PathBuf>,
    },

    #[command(about = "Run deterministic or local HF tiny Mixtral layer reference baseline")]
    MoeRunLayerReference {
        #[arg(value_name = "HF_MODEL_DIR")]
        hf_model_dir: Option<PathBuf>,
    },

    #[command(about = "Generate/validate attention MLIR through Vortex LLVM")]
    VortexGenerateAttention,

    #[command(about = "Run generated attention vxbin through Verilator RTLSim and compare output")]
    VortexRunAttention,

    #[command(name = "__vortex-run-attention-inner", hide = true)]
    VortexRunAttentionInner {
        #[arg(value_name = "VXBIN")]
        vxbin: PathBuf,
    },
}

impl XtaskCommand {
    pub(crate) const fn name(&self) -> &'static str {
        match self {
            Self::VortexPlanAttention => "vortex-plan-attention",
            Self::MoeFetchTinyMixtral => "moe-fetch-tiny-mixtral",
            Self::MoeRunReference { .. } => "moe-run-reference",
            Self::MoeRunLayerReference { .. } => "moe-run-layer-reference",
            Self::VortexGenerateAttention => "vortex-generate-attention",
            Self::VortexRunAttention => "vortex-run-attention",
            Self::VortexRunAttentionInner { .. } => "__vortex-run-attention-inner",
        }
    }
}
