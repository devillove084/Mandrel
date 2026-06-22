pub mod artifact;
pub mod backend;
pub mod codegen;
pub mod demo;
mod executor;
pub mod ffi;
pub mod matmul;
pub mod toolchain;
pub mod vortex2;

pub use artifact::{VortexArtifact, VortexArtifactRegistry};
pub use backend::{
    Result as VortexBackendResult, VortexBackend, VortexBackendConfig, VortexBackendError,
};
pub use codegen::{
    GeneratedVortexKernelSource, VortexCodegenError,
    generate_experimental_vortex_tiled_matmul_llvm_ir, generate_vortex_matmul_cpp,
    generate_vortex_matmul_llvm_ir,
};
pub use demo::{demo_matmul_plan, vortex_capabilities};
pub use executor::{VortexLaunchDims, VortexLaunchTrace};
pub use matmul::{
    MatmulI8I32Args, MatmulLaunchReport, MatmulRunError, MatmulRunOutput, MatmulRunTrace,
    MatmulShape, run_matmul_i8_i32,
};
pub use toolchain::*;
