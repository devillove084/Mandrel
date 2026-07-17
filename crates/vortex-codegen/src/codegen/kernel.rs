use mandrel_compiler::VortexAttentionPrefillPlan;

use crate::codegen::{
    GeneratedVortexKernelSource, VortexCodegenError, generate_vortex_attention_prefill_mlir,
};

#[derive(Debug, Clone, Copy)]
pub enum VortexKernelPlanRef<'a> {
    AttentionPrefillI8(&'a VortexAttentionPrefillPlan),
}

pub fn generate_vortex_kernel_mlir(
    plan: VortexKernelPlanRef<'_>,
) -> Result<GeneratedVortexKernelSource, VortexCodegenError> {
    match plan {
        VortexKernelPlanRef::AttentionPrefillI8(plan) => {
            generate_vortex_attention_prefill_mlir(plan)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{VortexKernelPlanRef, generate_vortex_kernel_mlir};
    use mandrel_compiler::compile_vortex_attention_prefill_kernel;
    use mandrel_kernel_ir::{KernelImplementation, KernelSymbol};
    use mandrel_model_ir::AttentionOp;

    use crate::codegen::GeneratedVortexSourceFormat;

    #[test]
    fn dispatches_attention_to_generated_mlir() {
        let plan = match compile_vortex_attention_prefill_kernel(AttentionOp::prefill_i8_demo()) {
            Ok(plan) => plan,
            Err(error) => panic!("unexpected compile error: {error:?}"),
        };
        let generated =
            match generate_vortex_kernel_mlir(VortexKernelPlanRef::AttentionPrefillI8(&plan)) {
                Ok(generated) => generated,
                Err(error) => panic!("unexpected codegen error: {error}"),
            };

        assert_eq!(generated.symbol, KernelSymbol::AttentionPrefillI8);
        assert_eq!(
            generated.implementation,
            KernelImplementation::GeneratedVortexMlir
        );
        assert_eq!(generated.format, GeneratedVortexSourceFormat::Mlir);
        assert!(generated.source.contains("@attention_prefill_i8"));
    }
}
