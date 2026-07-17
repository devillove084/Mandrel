use mandrel_kernel_ir::{KernelImplementation, KernelSymbol};
use snafu::Snafu;

use crate::attention_plan::AttentionPlanValidationError;

pub mod attention_mlir;
pub mod device_ir;
pub mod kernel;
pub(crate) mod mlir;
pub mod vortex_ir;

pub use attention_mlir::generate_vortex_attention_prefill_mlir;
pub use kernel::{VortexKernelPlanRef, generate_vortex_kernel_mlir};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneratedVortexSourceFormat {
    Mlir,
}

impl GeneratedVortexSourceFormat {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mlir => "mlir",
        }
    }

    pub const fn extension(self) -> &'static str {
        match self {
            Self::Mlir => "mlir",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedVortexKernelSource {
    pub symbol: KernelSymbol,
    pub implementation: KernelImplementation,
    pub format: GeneratedVortexSourceFormat,
    pub source: String,
    pub required_headers: &'static [&'static str],
}

#[derive(Debug, Clone, PartialEq, Eq, Snafu)]
pub enum VortexCodegenError {
    #[snafu(display("unsupported Vortex codegen kernel: {}", symbol.as_str()))]
    UnsupportedKernel { symbol: KernelSymbol },
    #[snafu(display("unsupported Vortex codegen signature for kernel: {}", symbol.as_str()))]
    UnsupportedSignature { symbol: KernelSymbol },
    #[snafu(display(
        "unsupported Vortex codegen schedule for kernel {}: {reason}",
        symbol.as_str()
    ))]
    UnsupportedSchedule {
        symbol: KernelSymbol,
        reason: &'static str,
    },
    #[snafu(display("invalid Vortex attention codegen plan: {source}"))]
    InvalidAttentionPlan {
        source: AttentionPlanValidationError,
    },
}
