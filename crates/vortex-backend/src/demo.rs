use mandrel_compiler::{VortexMatmulPlan, compile_vortex_matmul_kernel};
use mandrel_device::DeviceCapabilities;
use mandrel_model_ir::MatmulOp;

pub fn vortex_capabilities() -> DeviceCapabilities {
    DeviceCapabilities::vortex_simx_default()
}

pub fn demo_matmul_plan() -> Result<VortexMatmulPlan, String> {
    compile_vortex_matmul_kernel(MatmulOp::ggml_mul_mat_i8_demo())
        .map_err(|error| format!("failed to compile Vortex matmul plan: {error:?}"))
}

#[cfg(test)]
mod tests {
    use super::{demo_matmul_plan, vortex_capabilities};

    #[test]
    fn demo_plan_targets_matmul_kernel() {
        let plan = match demo_matmul_plan() {
            Ok(plan) => plan,
            Err(error) => panic!("unexpected plan error: {error}"),
        };

        assert_eq!(plan.launch.grid.x, 8);
        assert_eq!(plan.launch.grid.y, 8);
        assert_eq!(plan.metrics.kernel_launches, 1);
    }

    #[test]
    fn default_caps_expose_int8() {
        let caps = vortex_capabilities();

        assert!(caps.supports_int8);
    }
}
