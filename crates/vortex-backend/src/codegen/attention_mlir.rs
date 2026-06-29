use mandrel_compiler::VortexAttentionPrefillPlan;
use mandrel_kernel_ir::{KernelImplementation, KernelSymbol};
use mandrel_schedule::{AttentionKvLayout, AttentionPrefillSchedule, AttentionSoftmaxStrategy};

use crate::attention_plan::validate_attention_prefill_i8_plan_abi;
use crate::codegen::device_ir::{
    CastKind, FcmpPredicate, IcmpPredicate, LlvmType, PhiIncoming, VortexBinaryOp, VortexBlock,
    VortexDeviceModule, VortexFloatBinaryOp, VortexFloatUnaryIntrinsic, VortexKernelFunction,
    VortexOp, VortexStructType, VortexTerminator, render_vortex_device_module_mlir,
};
use crate::codegen::vortex_ir::{VortexAxis, VortexCtaCsr};
use crate::codegen::{
    GeneratedVortexKernelSource, GeneratedVortexSourceFormat, VortexCodegenError,
};

const ATTENTION_PREFILL_I8_ARGS_STRUCT: &str = "attention_prefill_i8_args_t";
const MLIR_SOURCE_NAME: &str = "mandrel generated Vortex attention prefill MLIR";

pub fn generate_vortex_attention_prefill_mlir(
    plan: &VortexAttentionPrefillPlan,
) -> Result<GeneratedVortexKernelSource, VortexCodegenError> {
    validate_attention_prefill_i8_plan(plan)?;
    let symbol = plan.kernel.symbol;
    let source = render_attention_prefill_i8_mlir(symbol, plan.schedule);

    Ok(GeneratedVortexKernelSource {
        symbol,
        implementation: KernelImplementation::GeneratedVortexMlir,
        format: GeneratedVortexSourceFormat::Mlir,
        source,
        required_headers: &[],
    })
}

fn validate_attention_prefill_i8_plan(
    plan: &VortexAttentionPrefillPlan,
) -> Result<(), VortexCodegenError> {
    let symbol = plan.kernel.symbol;
    if symbol != KernelSymbol::AttentionPrefillI8 || plan.launch.symbol != symbol {
        return Err(VortexCodegenError::UnsupportedKernel { symbol });
    }
    if !plan.kernel.signature.is_attention_prefill_i8() {
        return Err(VortexCodegenError::UnsupportedSignature { symbol });
    }
    if plan.schedule != AttentionPrefillSchedule::dense_online_4x16x64() {
        return Err(VortexCodegenError::UnsupportedSchedule {
            symbol,
            reason: "only dense online attention prefill schedule 4x16x64 is lowered to MLIR",
        });
    }
    if !matches!(plan.schedule.kv_layout, AttentionKvLayout::DenseContiguous) {
        return Err(VortexCodegenError::UnsupportedSchedule {
            symbol,
            reason: "attention prefill MLIR lowering currently supports only dense contiguous Q/K/V/O layout",
        });
    }
    validate_attention_prefill_i8_plan_abi(plan)
        .map_err(|source| VortexCodegenError::InvalidAttentionPlan { source })?;
    if !matches!(
        plan.schedule.softmax,
        AttentionSoftmaxStrategy::OnlineMaxSum
    ) {
        return Err(VortexCodegenError::UnsupportedSchedule {
            symbol,
            reason: "attention prefill MLIR lowering currently supports only online max/sum softmax metadata",
        });
    }
    if plan.launch.block.x != 4 || plan.launch.block.y != 4 || plan.launch.block.z != 1 {
        return Err(VortexCodegenError::UnsupportedSchedule {
            symbol,
            reason: "attention prefill MLIR lowering expects a 4x4x1 Vortex workgroup",
        });
    }

    Ok(())
}

fn render_attention_prefill_i8_mlir(
    symbol: KernelSymbol,
    schedule: AttentionPrefillSchedule,
) -> String {
    let module = build_attention_prefill_i8_device_module(symbol, schedule);
    render_vortex_device_module_mlir(&module)
}

fn build_attention_prefill_i8_device_module(
    symbol: KernelSymbol,
    schedule: AttentionPrefillSchedule,
) -> VortexDeviceModule {
    let mut module = VortexDeviceModule::new(
        "mandrel generated vortex attention_prefill_i8",
        "mandrel://generated/vortex/attention_prefill_i8.mlir",
        MLIR_SOURCE_NAME,
    );
    add_attention_prefill_i8_args_struct(&mut module);

    let head_dim_tile = schedule.tile.head_dim;
    let mut kernel = VortexKernelFunction::new(symbol.as_str(), "arg", LlvmType::Ptr);
    kernel.push_block(build_entry_block(head_dim_tile));
    kernel.push_block(build_dim_loop_block());
    kernel.push_block(build_score0_dot_loop_block());
    kernel.push_block(build_score0_dot_body_block());
    kernel.push_block(build_max_key_loop_block());
    kernel.push_block(build_max_dot_loop_block());
    kernel.push_block(build_max_dot_body_block());
    kernel.push_block(build_max_update_block());
    kernel.push_block(build_softmax_key_loop_block());
    kernel.push_block(build_softmax_dot_loop_block());
    kernel.push_block(build_softmax_dot_body_block());
    kernel.push_block(build_softmax_update_block());
    kernel.push_block(build_store_output_block());
    kernel.push_block(VortexBlock::new("exit", VortexTerminator::RetVoid));
    module.add_kernel(kernel);

    module
}

fn add_attention_prefill_i8_args_struct(module: &mut VortexDeviceModule) {
    module.add_struct(VortexStructType::new(
        ATTENTION_PREFILL_I8_ARGS_STRUCT,
        vec![
            LlvmType::I64,
            LlvmType::I64,
            LlvmType::I64,
            LlvmType::I64,
            LlvmType::I32,
            LlvmType::I32,
            LlvmType::I32,
            LlvmType::I32,
        ],
    ));
}

fn build_entry_block(head_dim_tile: usize) -> VortexBlock {
    let mut block = VortexBlock::new(
        "entry",
        VortexTerminator::cond_br("%query_oob", "exit", "dim_loop"),
    );
    emit_load_attention_prefill_i8_args(&mut block);
    emit_thread_coordinates(&mut block, head_dim_tile);
    block.push_op(VortexOp::icmp(
        "%query_oob",
        IcmpPredicate::Uge,
        LlvmType::I32,
        "%query",
        "%sequence",
    ));
    block
}

fn build_dim_loop_block() -> VortexBlock {
    let mut block = VortexBlock::new(
        "dim_loop",
        VortexTerminator::cond_br("%has_dim", "score0_dot_loop", "exit"),
    );
    block.push_op(VortexOp::phi(
        "%dim",
        LlvmType::I32,
        vec![
            PhiIncoming::new("%dim_start", "entry"),
            PhiIncoming::new("%dim_next", "store_output"),
        ],
    ));
    block.push_op(VortexOp::icmp(
        "%dim_lt_head",
        IcmpPredicate::Ult,
        LlvmType::I32,
        "%dim",
        "%head_dim",
    ));
    block.push_op(VortexOp::icmp(
        "%dim_lt_tile_end",
        IcmpPredicate::Ult,
        LlvmType::I32,
        "%dim",
        "%dim_tile_end",
    ));
    block.push_op(VortexOp::binary(
        "%has_dim",
        VortexBinaryOp::And,
        LlvmType::I1,
        "%dim_lt_head",
        "%dim_lt_tile_end",
    ));
    block
}

fn build_score0_dot_loop_block() -> VortexBlock {
    let mut block = VortexBlock::new(
        "score0_dot_loop",
        VortexTerminator::cond_br("%score0_has_depth", "score0_dot_body", "max_key_loop"),
    );
    block.push_op(VortexOp::phi(
        "%score0_depth",
        LlvmType::I32,
        vec![
            PhiIncoming::new("0", "dim_loop"),
            PhiIncoming::new("%score0_depth_next", "score0_dot_body"),
        ],
    ));
    block.push_op(VortexOp::phi(
        "%score0_dot",
        LlvmType::I32,
        vec![
            PhiIncoming::new("0", "dim_loop"),
            PhiIncoming::new("%score0_dot_next", "score0_dot_body"),
        ],
    ));
    block.push_op(VortexOp::icmp(
        "%score0_has_depth",
        IcmpPredicate::Ult,
        LlvmType::I32,
        "%score0_depth",
        "%head_dim",
    ));
    block.push_op(VortexOp::cast(
        "%score0_f32",
        CastKind::Sitofp,
        LlvmType::I32,
        "%score0_dot",
        LlvmType::F32,
    ));
    block
}

fn build_score0_dot_body_block() -> VortexBlock {
    let mut block = VortexBlock::new("score0_dot_body", VortexTerminator::br("score0_dot_loop"));
    emit_qk_dot_body(
        &mut block,
        "score0",
        "0",
        "%score0_depth",
        "%score0_dot",
        "%score0_depth_next",
        "%score0_dot_next",
    );
    block
}

fn build_max_key_loop_block() -> VortexBlock {
    let mut block = VortexBlock::new(
        "max_key_loop",
        VortexTerminator::cond_br("%max_has_key", "max_dot_loop", "softmax_key_loop"),
    );
    block.push_op(VortexOp::phi(
        "%max_key",
        LlvmType::I32,
        vec![
            PhiIncoming::new("1", "score0_dot_loop"),
            PhiIncoming::new("%max_key_next", "max_update"),
        ],
    ));
    block.push_op(VortexOp::phi(
        "%max_score",
        LlvmType::F32,
        vec![
            PhiIncoming::new("%score0_f32", "score0_dot_loop"),
            PhiIncoming::new("%max_score_next", "max_update"),
        ],
    ));
    block.push_op(VortexOp::icmp(
        "%max_has_key",
        IcmpPredicate::Ult,
        LlvmType::I32,
        "%max_key",
        "%sequence",
    ));
    block
}

fn build_max_dot_loop_block() -> VortexBlock {
    let mut block = VortexBlock::new(
        "max_dot_loop",
        VortexTerminator::cond_br("%max_has_depth", "max_dot_body", "max_update"),
    );
    block.push_op(VortexOp::phi(
        "%max_depth",
        LlvmType::I32,
        vec![
            PhiIncoming::new("0", "max_key_loop"),
            PhiIncoming::new("%max_depth_next", "max_dot_body"),
        ],
    ));
    block.push_op(VortexOp::phi(
        "%max_dot",
        LlvmType::I32,
        vec![
            PhiIncoming::new("0", "max_key_loop"),
            PhiIncoming::new("%max_dot_next", "max_dot_body"),
        ],
    ));
    block.push_op(VortexOp::icmp(
        "%max_has_depth",
        IcmpPredicate::Ult,
        LlvmType::I32,
        "%max_depth",
        "%head_dim",
    ));
    block
}

fn build_max_dot_body_block() -> VortexBlock {
    let mut block = VortexBlock::new("max_dot_body", VortexTerminator::br("max_dot_loop"));
    emit_qk_dot_body(
        &mut block,
        "max",
        "%max_key",
        "%max_depth",
        "%max_dot",
        "%max_depth_next",
        "%max_dot_next",
    );
    block
}

fn build_max_update_block() -> VortexBlock {
    let mut block = VortexBlock::new("max_update", VortexTerminator::br("max_key_loop"));
    block.push_op(VortexOp::cast(
        "%max_candidate_score",
        CastKind::Sitofp,
        LlvmType::I32,
        "%max_dot",
        LlvmType::F32,
    ));
    block.push_op(VortexOp::fcmp(
        "%max_is_larger",
        FcmpPredicate::Ogt,
        LlvmType::F32,
        "%max_candidate_score",
        "%max_score",
    ));
    block.push_op(VortexOp::select(
        "%max_score_next",
        "%max_is_larger",
        LlvmType::F32,
        "%max_candidate_score",
        "%max_score",
    ));
    block.push_op(VortexOp::binary(
        "%max_key_next",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%max_key",
        "1",
    ));
    block
}

fn build_softmax_key_loop_block() -> VortexBlock {
    let mut block = VortexBlock::new(
        "softmax_key_loop",
        VortexTerminator::cond_br("%softmax_has_key", "softmax_dot_loop", "store_output"),
    );
    block.push_op(VortexOp::phi(
        "%softmax_key",
        LlvmType::I32,
        vec![
            PhiIncoming::new("0", "max_key_loop"),
            PhiIncoming::new("%softmax_key_next", "softmax_update"),
        ],
    ));
    block.push_op(VortexOp::phi(
        "%denom",
        LlvmType::F32,
        vec![
            PhiIncoming::new("0.0", "max_key_loop"),
            PhiIncoming::new("%denom_next", "softmax_update"),
        ],
    ));
    block.push_op(VortexOp::phi(
        "%numer",
        LlvmType::F32,
        vec![
            PhiIncoming::new("0.0", "max_key_loop"),
            PhiIncoming::new("%numer_next", "softmax_update"),
        ],
    ));
    block.push_op(VortexOp::icmp(
        "%softmax_has_key",
        IcmpPredicate::Ult,
        LlvmType::I32,
        "%softmax_key",
        "%sequence",
    ));
    block
}

fn build_softmax_dot_loop_block() -> VortexBlock {
    let mut block = VortexBlock::new(
        "softmax_dot_loop",
        VortexTerminator::cond_br("%softmax_has_depth", "softmax_dot_body", "softmax_update"),
    );
    block.push_op(VortexOp::phi(
        "%softmax_depth",
        LlvmType::I32,
        vec![
            PhiIncoming::new("0", "softmax_key_loop"),
            PhiIncoming::new("%softmax_depth_next", "softmax_dot_body"),
        ],
    ));
    block.push_op(VortexOp::phi(
        "%softmax_dot",
        LlvmType::I32,
        vec![
            PhiIncoming::new("0", "softmax_key_loop"),
            PhiIncoming::new("%softmax_dot_next", "softmax_dot_body"),
        ],
    ));
    block.push_op(VortexOp::icmp(
        "%softmax_has_depth",
        IcmpPredicate::Ult,
        LlvmType::I32,
        "%softmax_depth",
        "%head_dim",
    ));
    block
}

fn build_softmax_dot_body_block() -> VortexBlock {
    let mut block = VortexBlock::new("softmax_dot_body", VortexTerminator::br("softmax_dot_loop"));
    emit_qk_dot_body(
        &mut block,
        "softmax",
        "%softmax_key",
        "%softmax_depth",
        "%softmax_dot",
        "%softmax_depth_next",
        "%softmax_dot_next",
    );
    block
}

fn build_softmax_update_block() -> VortexBlock {
    let mut block = VortexBlock::new("softmax_update", VortexTerminator::br("softmax_key_loop"));
    block.push_op(VortexOp::cast(
        "%softmax_score",
        CastKind::Sitofp,
        LlvmType::I32,
        "%softmax_dot",
        LlvmType::F32,
    ));
    block.push_op(VortexOp::float_binary(
        "%softmax_shifted",
        VortexFloatBinaryOp::Sub,
        LlvmType::F32,
        "%softmax_score",
        "%max_score",
    ));
    block.push_op(VortexOp::float_unary_intrinsic(
        "%softmax_weight",
        VortexFloatUnaryIntrinsic::Exp,
        LlvmType::F32,
        "%softmax_shifted",
    ));
    emit_i8_matrix_load_as_i32(
        &mut block,
        "softmax_v",
        "%v",
        "%softmax_key",
        "%dim",
        "%softmax_v_i8",
        "%softmax_v_i32",
    );
    block.push_op(VortexOp::cast(
        "%softmax_v_f32",
        CastKind::Sitofp,
        LlvmType::I32,
        "%softmax_v_i32",
        LlvmType::F32,
    ));
    block.push_op(VortexOp::float_binary(
        "%softmax_weighted_v",
        VortexFloatBinaryOp::Mul,
        LlvmType::F32,
        "%softmax_weight",
        "%softmax_v_f32",
    ));
    block.push_op(VortexOp::float_binary(
        "%denom_next",
        VortexFloatBinaryOp::Add,
        LlvmType::F32,
        "%denom",
        "%softmax_weight",
    ));
    block.push_op(VortexOp::float_binary(
        "%numer_next",
        VortexFloatBinaryOp::Add,
        LlvmType::F32,
        "%numer",
        "%softmax_weighted_v",
    ));
    block.push_op(VortexOp::binary(
        "%softmax_key_next",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%softmax_key",
        "1",
    ));
    block
}

fn build_store_output_block() -> VortexBlock {
    let mut block = VortexBlock::new("store_output", VortexTerminator::br("dim_loop"));
    block.push_op(VortexOp::float_binary(
        "%out_f32_raw",
        VortexFloatBinaryOp::Div,
        LlvmType::F32,
        "%numer",
        "%denom",
    ));
    block.push_op(VortexOp::fcmp(
        "%out_gt_i8_max",
        FcmpPredicate::Ogt,
        LlvmType::F32,
        "%out_f32_raw",
        "127.0",
    ));
    block.push_op(VortexOp::select(
        "%out_f32_hi",
        "%out_gt_i8_max",
        LlvmType::F32,
        "127.0",
        "%out_f32_raw",
    ));
    block.push_op(VortexOp::fcmp(
        "%out_lt_i8_min",
        FcmpPredicate::Olt,
        LlvmType::F32,
        "%out_f32_hi",
        "-128.0",
    ));
    block.push_op(VortexOp::select(
        "%out_f32_clamped",
        "%out_lt_i8_min",
        LlvmType::F32,
        "-128.0",
        "%out_f32_hi",
    ));
    block.push_op(VortexOp::cast(
        "%out_i32",
        CastKind::Fptosi,
        LlvmType::F32,
        "%out_f32_clamped",
        LlvmType::I32,
    ));
    block.push_op(VortexOp::cast(
        "%out_i8",
        CastKind::Trunc,
        LlvmType::I32,
        "%out_i32",
        LlvmType::I8,
    ));
    emit_i8_matrix_store(&mut block, "out", "%out", "%query", "%dim", "%out_i8");
    block.push_op(VortexOp::binary(
        "%dim_next",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%dim",
        "%block_dim_x",
    ));
    block
}

fn emit_load_attention_prefill_i8_args(block: &mut VortexBlock) {
    emit_struct_field_load(block, "%q_addr_ptr", "%q_addr", LlvmType::I64, 0, 8);
    block.push_op(VortexOp::inttoptr("%q", LlvmType::I64, "%q_addr"));
    emit_struct_field_load(block, "%k_addr_ptr", "%k_addr", LlvmType::I64, 1, 8);
    block.push_op(VortexOp::inttoptr("%k", LlvmType::I64, "%k_addr"));
    emit_struct_field_load(block, "%v_addr_ptr", "%v_addr", LlvmType::I64, 2, 8);
    block.push_op(VortexOp::inttoptr("%v", LlvmType::I64, "%v_addr"));
    emit_struct_field_load(block, "%out_addr_ptr", "%out_addr", LlvmType::I64, 3, 8);
    block.push_op(VortexOp::inttoptr("%out", LlvmType::I64, "%out_addr"));
    emit_struct_field_load(block, "%sequence_ptr", "%sequence", LlvmType::I32, 4, 4);
    emit_struct_field_load(block, "%head_dim_ptr", "%head_dim", LlvmType::I32, 5, 4);
    emit_struct_field_load(block, "%query_tile_ptr", "%query_tile", LlvmType::I32, 6, 4);
    emit_struct_field_load(block, "%key_tile_ptr", "%key_tile", LlvmType::I32, 7, 4);
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
        ATTENTION_PREFILL_I8_ARGS_STRUCT,
        "%arg",
        field,
    ));
    block.push_op(VortexOp::load(value_name, ty, ptr_name, align));
}

fn emit_thread_coordinates(block: &mut VortexBlock, head_dim_tile: usize) {
    block.push_op(VortexOp::uniform_cta_csr_i32(
        "%block_x",
        "%block_x64",
        "%block_x_raw",
        VortexCtaCsr::BlockId(VortexAxis::X),
    ));
    block.push_op(VortexOp::uniform_cta_csr_i32(
        "%block_z",
        "%block_z64",
        "%block_z_raw",
        VortexCtaCsr::BlockId(VortexAxis::Z),
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
    block.push_op(VortexOp::cta_csr_i32(
        "%thread_y",
        "%thread_y64",
        VortexCtaCsr::ThreadId(VortexAxis::Y),
    ));
    block.push_op(VortexOp::binary(
        "%query_base",
        VortexBinaryOp::Mul,
        LlvmType::I32,
        "%block_x",
        "%query_tile",
    ));
    block.push_op(VortexOp::binary(
        "%query",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%query_base",
        "%thread_y",
    ));
    block.push_op(VortexOp::binary(
        "%dim_tile_base",
        VortexBinaryOp::Mul,
        LlvmType::I32,
        "%block_z",
        head_dim_tile.to_string(),
    ));
    block.push_op(VortexOp::binary(
        "%dim_start",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%dim_tile_base",
        "%thread_x",
    ));
    block.push_op(VortexOp::binary(
        "%dim_tile_end",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%dim_tile_base",
        head_dim_tile.to_string(),
    ));
}

fn emit_qk_dot_body(
    block: &mut VortexBlock,
    prefix: &str,
    key: &str,
    depth: &str,
    acc: &str,
    depth_next: &str,
    acc_next: &str,
) {
    let q_i8 = format!("%{prefix}_q_i8");
    let q_i32 = format!("%{prefix}_q_i32");
    let k_i8 = format!("%{prefix}_k_i8");
    let k_i32 = format!("%{prefix}_k_i32");
    let product = format!("%{prefix}_qk_product");

    emit_i8_matrix_load_as_i32(
        block,
        &format!("{prefix}_q"),
        "%q",
        "%query",
        depth,
        &q_i8,
        &q_i32,
    );
    emit_i8_matrix_load_as_i32(
        block,
        &format!("{prefix}_k"),
        "%k",
        key,
        depth,
        &k_i8,
        &k_i32,
    );
    block.push_op(VortexOp::binary_nsw(
        product,
        VortexBinaryOp::Mul,
        LlvmType::I32,
        q_i32,
        k_i32,
    ));
    block.push_op(VortexOp::binary_nsw(
        acc_next,
        VortexBinaryOp::Add,
        LlvmType::I32,
        acc,
        format!("%{prefix}_qk_product"),
    ));
    block.push_op(VortexOp::binary(
        depth_next,
        VortexBinaryOp::Add,
        LlvmType::I32,
        depth,
        "1",
    ));
}

fn emit_i8_matrix_load_as_i32(
    block: &mut VortexBlock,
    prefix: &str,
    base: &str,
    row: &str,
    col: &str,
    result_i8: &str,
    result_i32: &str,
) {
    let row_offset = format!("%{prefix}_row_offset");
    let index32 = format!("%{prefix}_index32");
    let index = format!("%{prefix}_index");
    let ptr = format!("%{prefix}_ptr");

    block.push_op(VortexOp::binary(
        &row_offset,
        VortexBinaryOp::Mul,
        LlvmType::I32,
        row,
        "%head_dim",
    ));
    block.push_op(VortexOp::binary(
        &index32,
        VortexBinaryOp::Add,
        LlvmType::I32,
        row_offset,
        col,
    ));
    block.push_op(VortexOp::cast(
        &index,
        CastKind::Zext,
        LlvmType::I32,
        index32,
        LlvmType::I64,
    ));
    block.push_op(VortexOp::gep(&ptr, LlvmType::I8, base, index, true));
    block.push_op(VortexOp::load(result_i8, LlvmType::I8, ptr, 1));
    block.push_op(VortexOp::cast(
        result_i32,
        CastKind::Sext,
        LlvmType::I8,
        result_i8,
        LlvmType::I32,
    ));
}

fn emit_i8_matrix_store(
    block: &mut VortexBlock,
    prefix: &str,
    base: &str,
    row: &str,
    col: &str,
    value: &str,
) {
    let row_offset = format!("%{prefix}_row_offset");
    let index32 = format!("%{prefix}_index32");
    let index = format!("%{prefix}_index");
    let ptr = format!("%{prefix}_ptr");

    block.push_op(VortexOp::binary(
        &row_offset,
        VortexBinaryOp::Mul,
        LlvmType::I32,
        row,
        "%head_dim",
    ));
    block.push_op(VortexOp::binary(
        &index32,
        VortexBinaryOp::Add,
        LlvmType::I32,
        row_offset,
        col,
    ));
    block.push_op(VortexOp::cast(
        &index,
        CastKind::Zext,
        LlvmType::I32,
        index32,
        LlvmType::I64,
    ));
    block.push_op(VortexOp::gep(&ptr, LlvmType::I8, base, index, true));
    block.push_op(VortexOp::store(LlvmType::I8, value, ptr, 1));
}

#[cfg(test)]
mod tests {
    use super::generate_vortex_attention_prefill_mlir;
    use mandrel_compiler::{AttentionKvCacheMetadata, compile_vortex_attention_prefill_kernel};
    use mandrel_kernel_ir::{KernelImplementation, KernelSymbol};
    use mandrel_model_ir::AttentionOp;
    use mandrel_schedule::Layout2D;

    use crate::codegen::{GeneratedVortexSourceFormat, VortexCodegenError};

    #[test]
    fn generates_attention_prefill_mlir_from_compiler_plan() {
        let plan = match compile_vortex_attention_prefill_kernel(AttentionOp::prefill_i8_demo()) {
            Ok(plan) => plan,
            Err(error) => panic!("unexpected compile error: {error:?}"),
        };
        let generated = match generate_vortex_attention_prefill_mlir(&plan) {
            Ok(generated) => generated,
            Err(error) => panic!("unexpected codegen error: {error}"),
        };

        assert_eq!(generated.symbol, KernelSymbol::AttentionPrefillI8);
        assert_eq!(
            generated.implementation,
            KernelImplementation::GeneratedVortexMlir
        );
        assert_eq!(generated.format, GeneratedVortexSourceFormat::Mlir);
        assert!(generated.required_headers.is_empty());
        assert!(
            generated
                .source
                .contains("llvm.target_triple = \"riscv64-unknown-unknown-elf\"")
        );
        assert!(generated.source.contains("@attention_prefill_i8"));
        assert!(generated.source.contains("vortex-kernel"));
        assert!(generated.source.contains("attention_prefill_i8_args_t"));
        assert!(
            generated
                .source
                .contains("llvm.inline_asm has_side_effects \"csrr")
        );
        assert!(
            generated
                .source
                .contains("llvm.call @\"llvm.riscv.vx.uniform.i32.i32\"")
        );
        assert!(generated.source.contains("llvm.intr.exp"));
        assert!(generated.source.contains("llvm.fptosi"));
        assert!(generated.source.contains("llvm.store %out_i8"));
    }

    #[test]
    fn rejects_paged_kv_metadata_before_mlir_generation() {
        let mut plan = match compile_vortex_attention_prefill_kernel(AttentionOp::prefill_i8_demo())
        {
            Ok(plan) => plan,
            Err(error) => panic!("unexpected compile error: {error:?}"),
        };
        plan.metadata.kv_cache = AttentionKvCacheMetadata::Paged {
            page_size: 16,
            logical_key: Layout2D::row_major(64, 64),
            logical_value: Layout2D::row_major(64, 64),
        };

        let error = match generate_vortex_attention_prefill_mlir(&plan) {
            Ok(_) => panic!("expected paged KV metadata to be rejected"),
            Err(error) => error,
        };
        match error {
            VortexCodegenError::InvalidAttentionPlan { source } => {
                assert!(source.to_string().contains("paged KV cache metadata"));
            }
            other => panic!("unexpected codegen error: {other}"),
        }
    }
}
