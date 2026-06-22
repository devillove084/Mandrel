use genco::prelude::*;
use mandrel_compiler::VortexMatmulPlan;
use mandrel_kernel_ir::{KernelImplementation, KernelSymbol};
use mandrel_schedule::MatmulSchedule;

use crate::codegen::cpp::{CppTokens, render_cpp};
use crate::codegen::{GeneratedVortexKernelSource, VortexCodegenError};

const MATMUL_I8_I32_HEADERS: &[&str] = &["stdint.h", "vx_spawn2.h"];

pub fn generate_vortex_matmul_cpp(
    plan: &VortexMatmulPlan,
) -> Result<GeneratedVortexKernelSource, VortexCodegenError> {
    validate_direct_matmul_plan(plan)?;
    let symbol = plan.kernel.symbol;
    let source = render_direct_matmul_i8_i32_cpp(symbol)?;

    Ok(GeneratedVortexKernelSource {
        symbol,
        implementation: KernelImplementation::GeneratedVortexCpp,
        source,
        required_headers: MATMUL_I8_I32_HEADERS,
    })
}

pub(crate) fn validate_direct_matmul_plan(
    plan: &VortexMatmulPlan,
) -> Result<(), VortexCodegenError> {
    let symbol = plan.kernel.symbol;
    if symbol != KernelSymbol::MatmulI8I32 || plan.launch.symbol != KernelSymbol::MatmulI8I32 {
        return Err(VortexCodegenError::UnsupportedKernel(symbol));
    }
    if !plan.kernel.signature.is_matmul_i8_i8_i32() {
        return Err(VortexCodegenError::UnsupportedSignature(symbol));
    }
    if plan.schedule != MatmulSchedule::direct_thread_per_output_4x4() {
        return Err(VortexCodegenError::UnsupportedSchedule {
            symbol,
            reason: "only direct thread-per-output 4x4 schedule is implemented",
        });
    }
    if plan.launch.block.x != 4 || plan.launch.block.y != 4 || plan.launch.block.z != 1 {
        return Err(VortexCodegenError::UnsupportedSchedule {
            symbol,
            reason: "launch block must match direct 4x4x1 schedule",
        });
    }
    if plan.launch.shared_memory_bytes != 0 {
        return Err(VortexCodegenError::UnsupportedSchedule {
            symbol,
            reason: "direct matmul generator does not use local/shared memory",
        });
    }
    Ok(())
}

fn render_direct_matmul_i8_i32_cpp(symbol: KernelSymbol) -> Result<String, VortexCodegenError> {
    let kernel_name = symbol.as_str();
    let int8_t = &c::include_system("stdint.h", "int8_t");
    let int32_t = &c::include_system("stdint.h", "int32_t");
    let uint32_t = &c::include_system("stdint.h", "uint32_t");
    let uint64_t = &c::include_system("stdint.h", "uint64_t");
    let kernel_attr = &c::include_system("vx_spawn2.h", "__kernel");
    let uniform_attr = &c::include_system("vx_spawn2.h", "__UNIFORM__");

    let tokens: CppTokens = quote! {
        struct matmul_i8_i32_args_t {
            $uint64_t lhs_addr;
            $uint64_t rhs_addr;
            $uint64_t out_addr;
            $uint32_t m;
            $uint32_t n;
            $uint32_t k;
            $uint32_t lhs_stride;
            $uint32_t rhs_stride;
            $uint32_t out_stride;
        };

        $kernel_attr void $kernel_name(matmul_i8_i32_args_t* $uniform_attr arg) {
            auto lhs = reinterpret_cast<const $int8_t*>(arg->lhs_addr);
            auto rhs = reinterpret_cast<const $int8_t*>(arg->rhs_addr);
            auto out = reinterpret_cast<$int32_t*>(arg->out_addr);

            const $uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
            const $uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row >= arg->m || col >= arg->n) {
                return;
            }

            $int32_t acc = 0;
            for ($uint32_t depth = 0; depth < arg->k; ++depth) {
                const $int32_t a = static_cast<$int32_t>(lhs[row * arg->lhs_stride + depth]);
                const $int32_t b = static_cast<$int32_t>(rhs[depth * arg->rhs_stride + col]);
                acc += a * b;
            }

            out[row * arg->out_stride + col] = acc;
        }
    };

    render_cpp(tokens)
}

#[cfg(test)]
mod tests {
    use super::generate_vortex_matmul_cpp;
    use mandrel_compiler::compile_vortex_matmul_kernel;
    use mandrel_kernel_ir::{KernelImplementation, KernelSymbol};
    use mandrel_model_ir::MatmulOp;

    #[test]
    fn generates_direct_matmul_cpp_from_compiler_plan() {
        let plan = match compile_vortex_matmul_kernel(MatmulOp::ggml_mul_mat_i8_demo()) {
            Ok(plan) => plan,
            Err(error) => panic!("unexpected compile error: {error:?}"),
        };
        let generated = match generate_vortex_matmul_cpp(&plan) {
            Ok(generated) => generated,
            Err(error) => panic!("unexpected codegen error: {error}"),
        };

        assert_eq!(generated.symbol, KernelSymbol::MatmulI8I32);
        assert_eq!(
            generated.implementation,
            KernelImplementation::GeneratedVortexCpp
        );
        assert!(generated.source.contains("#include <stdint.h>"));
        assert!(generated.source.contains("#include <vx_spawn2.h>"));
        assert!(generated.source.contains("struct matmul_i8_i32_args_t"));
        assert!(generated.source.contains("__kernel void matmul_i8_i32"));
        assert!(generated.source.contains("row >= arg"));
        assert!(generated.source.contains("col >= arg"));
        assert!(generated.source.contains("lhs_stride"));
        assert!(generated.source.contains("rhs_stride"));
        assert!(generated.source.contains("out_stride"));
        assert!(generated.source.contains("depth < arg"));
        assert!(!generated.source.contains("__local_mem"));
        assert!(!generated.source.contains("__syncthreads"));
    }
}
