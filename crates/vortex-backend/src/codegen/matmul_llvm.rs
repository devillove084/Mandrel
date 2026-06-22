use mandrel_compiler::VortexMatmulPlan;
use mandrel_kernel_ir::{KernelImplementation, KernelSymbol};

use crate::codegen::device_ir::{
    CastKind, IcmpPredicate, LlvmType, PhiIncoming, VortexBinaryOp, VortexBlock,
    VortexDeviceModule, VortexKernelFunction, VortexOp, VortexTerminator,
    render_vortex_device_module_llvm_ir,
};
use crate::codegen::matmul::validate_direct_matmul_plan;
use crate::codegen::matmul_abi::{
    MATMUL_I8_I32_LLVM_HEADERS, add_matmul_i8_i32_args_struct, emit_load_matmul_i8_i32_args,
};
use crate::codegen::vortex_ir::{VortexAxis, VortexCtaCsr};
use crate::codegen::{GeneratedVortexKernelSource, VortexCodegenError};

const LLVM_SOURCE_NAME: &str = "mandrel generated Vortex LLVM IR";

pub fn generate_vortex_matmul_llvm_ir(
    plan: &VortexMatmulPlan,
) -> Result<GeneratedVortexKernelSource, VortexCodegenError> {
    validate_direct_matmul_plan(plan)?;
    let symbol = plan.kernel.symbol;
    let source = render_direct_matmul_i8_i32_llvm_ir(symbol);

    Ok(GeneratedVortexKernelSource {
        symbol,
        implementation: KernelImplementation::GeneratedVortexLlvmIr,
        source,
        required_headers: MATMUL_I8_I32_LLVM_HEADERS,
    })
}

fn render_direct_matmul_i8_i32_llvm_ir(symbol: KernelSymbol) -> String {
    let module = build_direct_matmul_i8_i32_device_module(symbol);
    render_vortex_device_module_llvm_ir(&module)
}

fn build_direct_matmul_i8_i32_device_module(symbol: KernelSymbol) -> VortexDeviceModule {
    let mut module = VortexDeviceModule::new(
        "mandrel generated vortex matmul_i8_i32",
        "mandrel://generated/vortex/matmul_i8_i32.ll",
        LLVM_SOURCE_NAME,
    );
    add_matmul_i8_i32_args_struct(&mut module);

    let mut kernel = VortexKernelFunction::new(symbol.as_str(), "arg", LlvmType::Ptr);
    kernel.push_block(build_entry_block());
    kernel.push_block(build_loop_block());
    kernel.push_block(build_loop_body_block());
    kernel.push_block(build_store_block());
    kernel.push_block(VortexBlock::new("exit", VortexTerminator::RetVoid));
    module.add_kernel(kernel);

    module
}

fn build_entry_block() -> VortexBlock {
    let mut block = VortexBlock::new("entry", VortexTerminator::cond_br("%oob", "exit", "loop"));
    emit_load_matmul_i8_i32_args(&mut block);
    emit_thread_coordinates(&mut block);
    emit_boundary_guard(&mut block);
    block
}

fn build_loop_block() -> VortexBlock {
    let mut block = VortexBlock::new(
        "loop",
        VortexTerminator::cond_br("%has_depth", "loop_body", "store"),
    );
    block.push_op(VortexOp::phi(
        "%depth",
        LlvmType::I32,
        vec![
            PhiIncoming::new("0", "entry"),
            PhiIncoming::new("%depth_next", "loop_body"),
        ],
    ));
    block.push_op(VortexOp::phi(
        "%acc",
        LlvmType::I32,
        vec![
            PhiIncoming::new("0", "entry"),
            PhiIncoming::new("%acc_next", "loop_body"),
        ],
    ));
    block.push_op(VortexOp::icmp(
        "%has_depth",
        IcmpPredicate::Ult,
        LlvmType::I32,
        "%depth",
        "%k",
    ));
    block
}

fn build_loop_body_block() -> VortexBlock {
    let mut block = VortexBlock::new("loop_body", VortexTerminator::br("loop"));
    block.push_op(VortexOp::binary(
        "%lhs_row_offset",
        VortexBinaryOp::Mul,
        LlvmType::I32,
        "%row",
        "%lhs_stride",
    ));
    block.push_op(VortexOp::binary(
        "%lhs_index32",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%lhs_row_offset",
        "%depth",
    ));
    block.push_op(VortexOp::cast(
        "%lhs_index",
        CastKind::Zext,
        LlvmType::I32,
        "%lhs_index32",
        LlvmType::I64,
    ));
    block.push_op(VortexOp::gep(
        "%lhs_ptr",
        LlvmType::I8,
        "%lhs",
        "%lhs_index",
        true,
    ));
    block.push_op(VortexOp::load("%a_i8", LlvmType::I8, "%lhs_ptr", 1));
    block.push_op(VortexOp::cast(
        "%a",
        CastKind::Sext,
        LlvmType::I8,
        "%a_i8",
        LlvmType::I32,
    ));
    block.push_op(VortexOp::binary(
        "%rhs_row_offset",
        VortexBinaryOp::Mul,
        LlvmType::I32,
        "%depth",
        "%rhs_stride",
    ));
    block.push_op(VortexOp::binary(
        "%rhs_index32",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%rhs_row_offset",
        "%col",
    ));
    block.push_op(VortexOp::cast(
        "%rhs_index",
        CastKind::Zext,
        LlvmType::I32,
        "%rhs_index32",
        LlvmType::I64,
    ));
    block.push_op(VortexOp::gep(
        "%rhs_ptr",
        LlvmType::I8,
        "%rhs",
        "%rhs_index",
        true,
    ));
    block.push_op(VortexOp::load("%b_i8", LlvmType::I8, "%rhs_ptr", 1));
    block.push_op(VortexOp::cast(
        "%b",
        CastKind::Sext,
        LlvmType::I8,
        "%b_i8",
        LlvmType::I32,
    ));
    block.push_op(VortexOp::binary_nsw(
        "%prod",
        VortexBinaryOp::Mul,
        LlvmType::I32,
        "%a",
        "%b",
    ));
    block.push_op(VortexOp::binary_nsw(
        "%acc_next",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%acc",
        "%prod",
    ));
    block.push_op(VortexOp::binary(
        "%depth_next",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%depth",
        "1",
    ));
    block
}

fn build_store_block() -> VortexBlock {
    let mut block = VortexBlock::new("store", VortexTerminator::br("exit"));
    block.push_op(VortexOp::binary(
        "%out_row_offset",
        VortexBinaryOp::Mul,
        LlvmType::I32,
        "%row",
        "%out_stride",
    ));
    block.push_op(VortexOp::binary(
        "%out_index32",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%out_row_offset",
        "%col",
    ));
    block.push_op(VortexOp::cast(
        "%out_index",
        CastKind::Zext,
        LlvmType::I32,
        "%out_index32",
        LlvmType::I64,
    ));
    block.push_op(VortexOp::gep(
        "%out_ptr",
        LlvmType::I32,
        "%out",
        "%out_index",
        true,
    ));
    block.push_op(VortexOp::store(LlvmType::I32, "%acc", "%out_ptr", 4));
    block
}

fn emit_thread_coordinates(block: &mut VortexBlock) {
    block.push_op(VortexOp::uniform_cta_csr_i32(
        "%block_y",
        "%block_y64",
        "%block_y_raw",
        VortexCtaCsr::BlockId(VortexAxis::Y),
    ));
    block.push_op(VortexOp::uniform_cta_csr_i32(
        "%block_dim_y",
        "%block_dim_y64",
        "%block_dim_y_raw",
        VortexCtaCsr::BlockDim(VortexAxis::Y),
    ));
    block.push_op(VortexOp::cta_csr_i32(
        "%thread_y",
        "%thread_y64",
        VortexCtaCsr::ThreadId(VortexAxis::Y),
    ));
    block.push_op(VortexOp::binary(
        "%row_base",
        VortexBinaryOp::Mul,
        LlvmType::I32,
        "%block_y",
        "%block_dim_y",
    ));
    block.push_op(VortexOp::binary(
        "%row",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%row_base",
        "%thread_y",
    ));

    block.push_op(VortexOp::uniform_cta_csr_i32(
        "%block_x",
        "%block_x64",
        "%block_x_raw",
        VortexCtaCsr::BlockId(VortexAxis::X),
    ));
    block.push_op(VortexOp::uniform_cta_csr_i32(
        "%block_dim_x",
        "%block_dim_x64",
        "%block_dim_x_raw",
        VortexCtaCsr::BlockDim(VortexAxis::X),
    ));
    block.push_op(VortexOp::cta_csr_i32(
        "%thread_x",
        "%thread_x64",
        VortexCtaCsr::ThreadId(VortexAxis::X),
    ));
    block.push_op(VortexOp::binary(
        "%col_base",
        VortexBinaryOp::Mul,
        LlvmType::I32,
        "%block_x",
        "%block_dim_x",
    ));
    block.push_op(VortexOp::binary(
        "%col",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%col_base",
        "%thread_x",
    ));
}

fn emit_boundary_guard(block: &mut VortexBlock) {
    block.push_op(VortexOp::icmp(
        "%row_oob",
        IcmpPredicate::Uge,
        LlvmType::I32,
        "%row",
        "%m",
    ));
    block.push_op(VortexOp::icmp(
        "%col_oob",
        IcmpPredicate::Uge,
        LlvmType::I32,
        "%col",
        "%n",
    ));
    block.push_op(VortexOp::binary(
        "%oob",
        VortexBinaryOp::Or,
        LlvmType::I1,
        "%row_oob",
        "%col_oob",
    ));
}

#[cfg(test)]
mod tests {
    use super::generate_vortex_matmul_llvm_ir;
    use mandrel_compiler::compile_vortex_matmul_kernel;
    use mandrel_kernel_ir::{KernelImplementation, KernelSymbol};
    use mandrel_model_ir::MatmulOp;

    #[test]
    fn generates_direct_matmul_llvm_ir_from_compiler_plan() {
        let plan = match compile_vortex_matmul_kernel(MatmulOp::ggml_mul_mat_i8_demo()) {
            Ok(plan) => plan,
            Err(error) => panic!("unexpected compile error: {error:?}"),
        };
        let generated = match generate_vortex_matmul_llvm_ir(&plan) {
            Ok(generated) => generated,
            Err(error) => panic!("unexpected codegen error: {error}"),
        };

        assert_eq!(generated.symbol, KernelSymbol::MatmulI8I32);
        assert_eq!(
            generated.implementation,
            KernelImplementation::GeneratedVortexLlvmIr
        );
        assert!(generated.required_headers.is_empty());
        assert!(
            generated
                .source
                .contains("target triple = \"riscv64-unknown-unknown-elf\"")
        );
        assert!(generated.source.contains("@llvm.global.annotations"));
        assert!(generated.source.contains("vortex.kernel"));
        assert!(
            generated
                .source
                .contains("define dso_local void @matmul_i8_i32")
        );
        assert!(generated.source.contains("\"vortex-kernel\""));
        assert!(generated.source.contains("@llvm.riscv.vx.uniform.i32.i32"));
        assert!(generated.source.contains("csrr $0, $1"));
        assert!(generated.source.contains("i32 3283"));
        assert!(generated.source.contains("i32 3284"));
        assert!(generated.source.contains("i32 3286"));
        assert!(generated.source.contains("i32 3287"));
        assert!(generated.source.contains("%oob = or i1 %row_oob, %col_oob"));
        assert!(generated.source.contains("%a = sext i8 %a_i8 to i32"));
        assert!(
            generated
                .source
                .contains("store i32 %acc, ptr %out_ptr, align 4")
        );
    }
}
