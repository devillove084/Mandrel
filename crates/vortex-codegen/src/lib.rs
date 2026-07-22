pub mod attention_plan;
pub mod codegen;

pub use attention_plan::{
    AttentionPlanValidationError, AttentionPlanValidationResult,
    validate_attention_prefill_i8_plan_abi,
};
pub use codegen::{
    GeneratedVortexKernelSource, GeneratedVortexSourceFormat,
    VORTEX_HARDWARE_IDENTITY_PROBE_SYMBOL, VORTEX_MANDREL_CONFIG_ID_CSR, VortexCodegenError,
    VortexKernelPlanRef, generate_vortex_attention_prefill_mlir,
    generate_vortex_hardware_identity_probe_mlir, generate_vortex_kernel_mlir,
};
