use crate::codegen::device_ir::{
    LlvmType, VortexBlock, VortexDeviceModule, VortexOp, VortexStructType,
};

pub(crate) const MATMUL_ARGS_STRUCT: &str = "matmul_i8_i32_args_t";
pub(crate) const MATMUL_I8_I32_LLVM_HEADERS: &[&str] = &[];

pub(crate) fn add_matmul_i8_i32_args_struct(module: &mut VortexDeviceModule) {
    module.add_struct(VortexStructType::new(
        MATMUL_ARGS_STRUCT,
        vec![
            LlvmType::I64,
            LlvmType::I64,
            LlvmType::I64,
            LlvmType::I32,
            LlvmType::I32,
            LlvmType::I32,
            LlvmType::I32,
            LlvmType::I32,
            LlvmType::I32,
        ],
    ));
}

pub(crate) fn emit_load_matmul_i8_i32_args(block: &mut VortexBlock) {
    emit_struct_field_load(block, "%lhs_addr_ptr", "%lhs_addr", LlvmType::I64, 0, 8);
    block.push_op(VortexOp::inttoptr("%lhs", LlvmType::I64, "%lhs_addr"));
    emit_struct_field_load(block, "%rhs_addr_ptr", "%rhs_addr", LlvmType::I64, 1, 8);
    block.push_op(VortexOp::inttoptr("%rhs", LlvmType::I64, "%rhs_addr"));
    emit_struct_field_load(block, "%out_addr_ptr", "%out_addr", LlvmType::I64, 2, 8);
    block.push_op(VortexOp::inttoptr("%out", LlvmType::I64, "%out_addr"));
    emit_struct_field_load(block, "%m_ptr", "%m", LlvmType::I32, 3, 4);
    emit_struct_field_load(block, "%n_ptr", "%n", LlvmType::I32, 4, 4);
    emit_struct_field_load(block, "%k_ptr", "%k", LlvmType::I32, 5, 4);
    emit_struct_field_load(block, "%lhs_stride_ptr", "%lhs_stride", LlvmType::I32, 6, 4);
    emit_struct_field_load(block, "%rhs_stride_ptr", "%rhs_stride", LlvmType::I32, 7, 4);
    emit_struct_field_load(block, "%out_stride_ptr", "%out_stride", LlvmType::I32, 8, 4);
}

fn emit_struct_field_load(
    block: &mut VortexBlock,
    ptr_name: &str,
    value_name: &str,
    ty: LlvmType,
    field: u32,
    align: u32,
) {
    block.push_op(VortexOp::struct_field_ptr(
        ptr_name,
        MATMUL_ARGS_STRUCT,
        "%arg",
        field,
    ));
    block.push_op(VortexOp::load(value_name, ty, ptr_name, align));
}
