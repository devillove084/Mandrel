use std::io;
use std::path::PathBuf;

use clap::{CommandFactory, Parser, Subcommand};

const AFTER_LONG_HELP: &str = "\
Main backend target:
  Vortex RISC-V GPGPU, route B custom backend via mandrel runtime/kernel IR

Local install defaults:
  source: external/vortex
  build:  external/vortex-build
  tools:  external/vortex-source-tools by default; external/vortex-system-tools for system mode
  env:    external/vortex-env.sh

Vortex toolchain mode:
  MANDREL_VORTEX_TOOLCHAIN_MODE=auto|prebuilt|system|skip
  default is host-dependent: auto on x86_64, system on non-x86_64
  explicit auto uses upstream prebuilt packages only on x86_64
  system maps Ubuntu/system packages into Vortex's expected local layout
  skip assumes MANDREL_VORTEX_TOOLDIR is already populated
  MANDREL_VORTEX_BUILD_PROFILE=full|simx|software|none controls build scope
  Device codegen is MLIR-only; xtask lowers kernel.mlir with mlir-translate before clang
  MANDREL_ALLOW_LIBGCC_BUILTINS=1 enables explicit experimental libgcc fallback
  Source LLVM build knobs: MANDREL_VORTEX_LLVM_DIR, MANDREL_VORTEX_LLVM_BUILD_DIR,
    MANDREL_VORTEX_COMPILER_RT_BUILD_DIR, MANDREL_VORTEX_LLVM_URL,
    MANDREL_VORTEX_LLVM_REF, MANDREL_VORTEX_LLVM_PROJECTS,
    MANDREL_VORTEX_LLVM_TARGETS, MANDREL_VORTEX_TOOLCHAIN_JOBS

Slow GitHub/HF downloads:
  MANDREL_GITHUB_PROXY_PREFIX=https://gh-proxy.org/ cargo vortex-install
  MANDREL_HF_ENDPOINT=https://hf-mirror.com cargo xtask moe-fetch-tiny-mixtral
  MANDREL_FETCH_RETRIES=5 cargo vortex-install

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
    long_about = "Mandrel project automation for Vortex toolchain setup, attention runtime traces, and tiny MoE reference flows.",
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
    #[command(about = "Clone/update Vortex source only under external/vortex")]
    VortexFetch,

    #[command(about = "Create Vortex-compatible wrappers for Ubuntu/system packages")]
    VortexSystemTools,

    #[command(about = "Source-build llvm-vortex + compiler-rt for this host")]
    VortexToolchainSource,

    #[command(about = "Clone, configure, prepare toolchain, build, install, write env")]
    VortexInstall,

    #[command(about = "Write/print external/vortex-env.sh for manual shells")]
    VortexEnv,

    #[command(about = "Show checkout/build/install/env status")]
    VortexStatus,

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

    #[command(about = "Run generated attention vxbin through Vortex simx and compare output")]
    VortexRunAttention,

    #[command(about = "Run Vortex official vecadd through ci/blackbox.sh when available")]
    VortexRunVecadd,

    #[command(name = "__vortex-run-attention-inner", hide = true)]
    VortexRunAttentionInner {
        #[arg(value_name = "VXBIN")]
        vxbin: PathBuf,
    },
}

impl XtaskCommand {
    pub(crate) const fn name(&self) -> &'static str {
        match self {
            Self::VortexFetch => "vortex-fetch",
            Self::VortexSystemTools => "vortex-system-tools",
            Self::VortexToolchainSource => "vortex-toolchain-source",
            Self::VortexInstall => "vortex-install",
            Self::VortexEnv => "vortex-env",
            Self::VortexStatus => "vortex-status",
            Self::VortexPlanAttention => "vortex-plan-attention",
            Self::MoeFetchTinyMixtral => "moe-fetch-tiny-mixtral",
            Self::MoeRunReference { .. } => "moe-run-reference",
            Self::MoeRunLayerReference { .. } => "moe-run-layer-reference",
            Self::VortexGenerateAttention => "vortex-generate-attention",
            Self::VortexRunAttention => "vortex-run-attention",
            Self::VortexRunVecadd => "vortex-run-vecadd",
            Self::VortexRunAttentionInner { .. } => "__vortex-run-attention-inner",
        }
    }
}
