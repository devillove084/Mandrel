use std::fmt;

use mandrel_kernel_ir::{KernelImplementation, KernelSymbol};

pub mod cpp;
pub mod device_ir;
pub(crate) mod llvm_ir;
pub mod matmul;
pub(crate) mod matmul_abi;
pub mod matmul_llvm;
pub mod matmul_tiled_llvm;
pub mod vortex_ir;

pub use matmul::generate_vortex_matmul_cpp;
pub use matmul_llvm::generate_vortex_matmul_llvm_ir;
pub use matmul_tiled_llvm::generate_experimental_vortex_tiled_matmul_llvm_ir;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedVortexKernelSource {
    pub symbol: KernelSymbol,
    pub implementation: KernelImplementation,
    pub source: String,
    pub required_headers: &'static [&'static str],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VortexCodegenError {
    UnsupportedKernel(KernelSymbol),
    UnsupportedSignature(KernelSymbol),
    UnsupportedSchedule {
        symbol: KernelSymbol,
        reason: &'static str,
    },
    FormatFailed,
}

impl fmt::Display for VortexCodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedKernel(symbol) => {
                write!(f, "unsupported Vortex codegen kernel: {}", symbol.as_str())
            }
            Self::UnsupportedSignature(symbol) => write!(
                f,
                "unsupported Vortex codegen signature for kernel: {}",
                symbol.as_str()
            ),
            Self::UnsupportedSchedule { symbol, reason } => write!(
                f,
                "unsupported Vortex codegen schedule for kernel {}: {reason}",
                symbol.as_str()
            ),
            Self::FormatFailed => write!(f, "failed to format generated Vortex source"),
        }
    }
}

impl std::error::Error for VortexCodegenError {}
