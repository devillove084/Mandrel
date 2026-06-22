use mandrel_compiler::VortexMatmulPlan;
use mandrel_kernel_ir::{KernelImplementation, KernelSymbol};
use mandrel_schedule::{CopyAtom2D, Layout2D, MatmulSchedule, Shape2D, Stride2D, ThreadTileMap2D};

use crate::codegen::device_ir::{
    CastKind, IcmpPredicate, LlvmType, PhiIncoming, VortexBinaryOp, VortexBlock,
    VortexDeviceModule, VortexKernelFunction, VortexOp, VortexTerminator,
    render_vortex_device_module_llvm_ir,
};
use crate::codegen::matmul_abi::{
    MATMUL_I8_I32_LLVM_HEADERS, add_matmul_i8_i32_args_struct, emit_load_matmul_i8_i32_args,
};
use crate::codegen::vortex_ir::{VortexAxis, VortexCtaCsr};
use crate::codegen::{GeneratedVortexKernelSource, VortexCodegenError};

const LLVM_SOURCE_NAME: &str = "mandrel generated experimental tiled Vortex LLVM IR";
const TILE_M: u32 = 4;
const TILE_N: u32 = 4;
const TILE_K: u32 = 32;
const THREADS_PER_WORKGROUP: u32 = TILE_M * TILE_N;
const LHS_TILE_BYTES: u32 = TILE_M * TILE_K;
const RHS_TILE_BYTES: u32 = TILE_K * TILE_N;
const SHARED_MEMORY_BYTES: u32 = LHS_TILE_BYTES + RHS_TILE_BYTES;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TiledMatmul2DSemantics {
    lhs_layout: Layout2D,
    rhs_layout: Layout2D,
    out_layout: Layout2D,
    output_thread_map: ThreadTileMap2D,
    lhs_stage: CopyAtom2D,
    rhs_stage: CopyAtom2D,
}

pub fn generate_experimental_vortex_tiled_matmul_llvm_ir(
    plan: &VortexMatmulPlan,
) -> Result<GeneratedVortexKernelSource, VortexCodegenError> {
    validate_tiled_matmul_plan(plan)?;
    let semantics = tiled_matmul_2d_semantics(plan);
    let symbol = plan.kernel.symbol;
    let source = render_tiled_matmul_i8_i32_llvm_ir(symbol, semantics);

    Ok(GeneratedVortexKernelSource {
        symbol,
        implementation: KernelImplementation::GeneratedVortexLlvmIr,
        source,
        required_headers: MATMUL_I8_I32_LLVM_HEADERS,
    })
}

fn validate_tiled_matmul_plan(plan: &VortexMatmulPlan) -> Result<(), VortexCodegenError> {
    let symbol = plan.kernel.symbol;
    if symbol != KernelSymbol::MatmulI8I32Tiled || plan.launch.symbol != symbol {
        return Err(VortexCodegenError::UnsupportedKernel(symbol));
    }
    if !plan.kernel.signature.is_matmul_i8_i8_i32() {
        return Err(VortexCodegenError::UnsupportedSignature(symbol));
    }
    if plan.schedule != MatmulSchedule::tiled_local_4x4x32() {
        return Err(VortexCodegenError::UnsupportedSchedule {
            symbol,
            reason: "only experimental tiled local-memory 4x4x32 schedule is implemented",
        });
    }
    if plan.launch.block.x != TILE_N || plan.launch.block.y != TILE_M || plan.launch.block.z != 1 {
        return Err(VortexCodegenError::UnsupportedSchedule {
            symbol,
            reason: "launch block must match tiled 4x4x1 thread map",
        });
    }
    if plan.launch.shared_memory_bytes != SHARED_MEMORY_BYTES {
        return Err(VortexCodegenError::UnsupportedSchedule {
            symbol,
            reason: "shared memory bytes must match lhs/rhs staged tiles",
        });
    }

    let semantics = tiled_matmul_2d_semantics(plan);
    if !semantics.output_thread_map.covers_tile() {
        return Err(VortexCodegenError::UnsupportedSchedule {
            symbol,
            reason: "thread map must cover the output tile",
        });
    }

    Ok(())
}

fn tiled_matmul_2d_semantics(plan: &VortexMatmulPlan) -> TiledMatmul2DSemantics {
    let shape = plan.op.shape;
    let tile = plan.schedule.tile;

    TiledMatmul2DSemantics {
        lhs_layout: Layout2D::new(Shape2D::new(shape.m, shape.k), Stride2D::new(shape.k, 1)),
        rhs_layout: Layout2D::new(Shape2D::new(shape.k, shape.n), Stride2D::new(shape.n, 1)),
        out_layout: Layout2D::new(Shape2D::new(shape.m, shape.n), Stride2D::new(shape.n, 1)),
        output_thread_map: plan.schedule.output_thread_map(),
        lhs_stage: CopyAtom2D::global_to_local(Shape2D::new(tile.m, tile.k)),
        rhs_stage: CopyAtom2D::global_to_local(Shape2D::new(tile.k, tile.n)),
    }
}

fn render_tiled_matmul_i8_i32_llvm_ir(
    symbol: KernelSymbol,
    semantics: TiledMatmul2DSemantics,
) -> String {
    let module = build_tiled_matmul_i8_i32_device_module(symbol, semantics);
    render_vortex_device_module_llvm_ir(&module)
}

fn build_tiled_matmul_i8_i32_device_module(
    symbol: KernelSymbol,
    semantics: TiledMatmul2DSemantics,
) -> VortexDeviceModule {
    let mut module = VortexDeviceModule::new(
        "mandrel generated experimental vortex matmul_i8_i32_tiled",
        "mandrel://generated/vortex/matmul_i8_i32_tiled.ll",
        LLVM_SOURCE_NAME,
    );
    add_matmul_i8_i32_args_struct(&mut module);

    let mut kernel = VortexKernelFunction::new(symbol.as_str(), "arg", LlvmType::Ptr);
    kernel.push_block(build_entry_block(semantics));
    kernel.push_block(build_k_loop_block());
    push_stage_copy_blocks(&mut kernel);
    kernel.push_block(build_stage_barrier_block());
    kernel.push_block(build_compute_loop_block());
    kernel.push_block(build_compute_body_block());
    kernel.push_block(build_after_compute_block());
    kernel.push_block(build_k_continue_block());
    kernel.push_block(build_store_check_block());
    kernel.push_block(build_store_block());
    kernel.push_block(VortexBlock::new("exit", VortexTerminator::RetVoid));
    module.add_kernel(kernel);

    module
}

fn build_entry_block(semantics: TiledMatmul2DSemantics) -> VortexBlock {
    let mut block = VortexBlock::new("entry", VortexTerminator::br("k_loop"));
    emit_load_matmul_i8_i32_args(&mut block);
    emit_tiled_thread_coordinates(&mut block, semantics);
    emit_output_boundary_flags(&mut block);
    block.push_op(VortexOp::uniform_cta_csr_i32(
        "%cta_id",
        "%cta_id64",
        "%cta_id_raw",
        VortexCtaCsr::Id,
    ));
    block.push_op(VortexOp::uniform_cta_csr_i32(
        "%cta_size",
        "%cta_size64",
        "%cta_size_raw",
        VortexCtaCsr::Size,
    ));
    block.push_op(VortexOp::local_mem_ptr("%lhs_tile", "%local64"));
    block.push_op(VortexOp::gep(
        "%rhs_tile",
        LlvmType::I8,
        "%lhs_tile",
        LHS_TILE_BYTES.to_string(),
        true,
    ));
    block
}

fn emit_tiled_thread_coordinates(block: &mut VortexBlock, semantics: TiledMatmul2DSemantics) {
    let tile_rows = semantics.output_thread_map.tile.rows.to_string();
    let tile_cols = semantics.output_thread_map.tile.cols.to_string();
    let block_cols = semantics.output_thread_map.block.x.to_string();

    block.push_op(VortexOp::uniform_cta_csr_i32(
        "%block_y",
        "%block_y64",
        "%block_y_raw",
        VortexCtaCsr::BlockId(VortexAxis::Y),
    ));
    block.push_op(VortexOp::cta_csr_i32(
        "%thread_y",
        "%thread_y64",
        VortexCtaCsr::ThreadId(VortexAxis::Y),
    ));
    block.push_op(VortexOp::binary(
        "%tile_row_base",
        VortexBinaryOp::Mul,
        LlvmType::I32,
        "%block_y",
        tile_rows,
    ));
    block.push_op(VortexOp::binary(
        "%row",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%tile_row_base",
        "%thread_y",
    ));

    block.push_op(VortexOp::uniform_cta_csr_i32(
        "%block_x",
        "%block_x64",
        "%block_x_raw",
        VortexCtaCsr::BlockId(VortexAxis::X),
    ));
    block.push_op(VortexOp::cta_csr_i32(
        "%thread_x",
        "%thread_x64",
        VortexCtaCsr::ThreadId(VortexAxis::X),
    ));
    block.push_op(VortexOp::binary(
        "%tile_col_base",
        VortexBinaryOp::Mul,
        LlvmType::I32,
        "%block_x",
        tile_cols,
    ));
    block.push_op(VortexOp::binary(
        "%col",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%tile_col_base",
        "%thread_x",
    ));
    block.push_op(VortexOp::binary(
        "%thread_linear_y",
        VortexBinaryOp::Mul,
        LlvmType::I32,
        "%thread_y",
        block_cols,
    ));
    block.push_op(VortexOp::binary(
        "%thread_linear",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%thread_linear_y",
        "%thread_x",
    ));
}

fn emit_output_boundary_flags(block: &mut VortexBlock) {
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

fn build_k_loop_block() -> VortexBlock {
    let mut block = VortexBlock::new(
        "k_loop",
        VortexTerminator::cond_br("%has_k_tile", "stage_lhs0_eval", "store_check"),
    );
    block.push_op(VortexOp::phi(
        "%k_base",
        LlvmType::I32,
        vec![
            PhiIncoming::new("0", "entry"),
            PhiIncoming::new("%k_base_next", "k_continue"),
        ],
    ));
    block.push_op(VortexOp::phi(
        "%acc",
        LlvmType::I32,
        vec![
            PhiIncoming::new("0", "entry"),
            PhiIncoming::new("%acc_after_tile", "k_continue"),
        ],
    ));
    block.push_op(VortexOp::icmp(
        "%has_k_tile",
        IcmpPredicate::Ult,
        LlvmType::I32,
        "%k_base",
        "%k",
    ));
    block
}

fn push_stage_copy_blocks(kernel: &mut VortexKernelFunction) {
    let lhs_rounds = stage_copy_rounds(LHS_TILE_BYTES);
    let rhs_rounds = stage_copy_rounds(RHS_TILE_BYTES);
    let max_rounds = lhs_rounds.max(rhs_rounds);

    for copy_index in 0..max_rounds {
        let copy_offset = copy_index * THREADS_PER_WORKGROUP;
        if copy_index < lhs_rounds {
            let next_label = if copy_index < rhs_rounds {
                format!("stage_rhs{copy_index}_eval")
            } else {
                next_stage_copy_label(copy_index + 1, lhs_rounds, rhs_rounds)
            };
            push_lhs_stage_copy_blocks(kernel, copy_index, copy_offset, next_label);
        }
        if copy_index < rhs_rounds {
            let next_label = next_stage_copy_label(copy_index + 1, lhs_rounds, rhs_rounds);
            push_rhs_stage_copy_blocks(kernel, copy_index, copy_offset, next_label);
        }
    }
}

const fn stage_copy_rounds(tile_bytes: u32) -> u32 {
    tile_bytes.div_ceil(THREADS_PER_WORKGROUP)
}

fn next_stage_copy_label(copy_index: u32, lhs_rounds: u32, rhs_rounds: u32) -> String {
    if copy_index < lhs_rounds {
        format!("stage_lhs{copy_index}_eval")
    } else if copy_index < rhs_rounds {
        format!("stage_rhs{copy_index}_eval")
    } else {
        "stage_barrier".to_owned()
    }
}

fn push_lhs_stage_copy_blocks(
    kernel: &mut VortexKernelFunction,
    copy_index: u32,
    copy_offset: u32,
    next_label: impl Into<String>,
) {
    let next_label = next_label.into();
    let prefix = format!("lhs{copy_index}");
    let eval_label = format!("stage_lhs{copy_index}_eval");
    let load_label = format!("stage_lhs{copy_index}_load");
    let zero_label = format!("stage_lhs{copy_index}_zero");
    let join_label = format!("stage_lhs{copy_index}_join");

    let mut eval = VortexBlock::new(
        eval_label,
        VortexTerminator::cond_br(
            format!("%{prefix}_valid"),
            load_label.clone(),
            zero_label.clone(),
        ),
    );
    emit_lhs_stage_copy_eval(&mut eval, &prefix, copy_offset);
    kernel.push_block(eval);

    let mut load = VortexBlock::new(load_label, VortexTerminator::br(join_label.clone()));
    emit_lhs_stage_copy_load(&mut load, &prefix);
    kernel.push_block(load);

    let mut zero = VortexBlock::new(zero_label, VortexTerminator::br(join_label.clone()));
    zero.push_op(VortexOp::store(
        LlvmType::I8,
        "0",
        format!("%{prefix}_local_ptr"),
        1,
    ));
    kernel.push_block(zero);

    kernel.push_block(VortexBlock::new(
        join_label,
        VortexTerminator::br(next_label),
    ));
}

fn emit_lhs_stage_copy_eval(block: &mut VortexBlock, prefix: &str, copy_offset: u32) {
    block.push_op(VortexOp::binary(
        format!("%{prefix}_linear"),
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%thread_linear",
        copy_offset.to_string(),
    ));
    block.push_op(VortexOp::binary(
        format!("%{prefix}_row"),
        VortexBinaryOp::Udiv,
        LlvmType::I32,
        format!("%{prefix}_linear"),
        TILE_K.to_string(),
    ));
    block.push_op(VortexOp::binary(
        format!("%{prefix}_col"),
        VortexBinaryOp::Urem,
        LlvmType::I32,
        format!("%{prefix}_linear"),
        TILE_K.to_string(),
    ));
    block.push_op(VortexOp::binary(
        format!("%{prefix}_global_row"),
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%tile_row_base",
        format!("%{prefix}_row"),
    ));
    block.push_op(VortexOp::binary(
        format!("%{prefix}_global_k"),
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%k_base",
        format!("%{prefix}_col"),
    ));
    block.push_op(VortexOp::icmp(
        format!("%{prefix}_row_ok"),
        IcmpPredicate::Ult,
        LlvmType::I32,
        format!("%{prefix}_global_row"),
        "%m",
    ));
    block.push_op(VortexOp::icmp(
        format!("%{prefix}_k_ok"),
        IcmpPredicate::Ult,
        LlvmType::I32,
        format!("%{prefix}_global_k"),
        "%k",
    ));
    block.push_op(VortexOp::binary(
        format!("%{prefix}_valid"),
        VortexBinaryOp::And,
        LlvmType::I1,
        format!("%{prefix}_row_ok"),
        format!("%{prefix}_k_ok"),
    ));
    block.push_op(VortexOp::cast(
        format!("%{prefix}_local_index"),
        CastKind::Zext,
        LlvmType::I32,
        format!("%{prefix}_linear"),
        LlvmType::I64,
    ));
    block.push_op(VortexOp::gep(
        format!("%{prefix}_local_ptr"),
        LlvmType::I8,
        "%lhs_tile",
        format!("%{prefix}_local_index"),
        true,
    ));
}

fn emit_lhs_stage_copy_load(block: &mut VortexBlock, prefix: &str) {
    block.push_op(VortexOp::binary(
        format!("%{prefix}_global_row_offset"),
        VortexBinaryOp::Mul,
        LlvmType::I32,
        format!("%{prefix}_global_row"),
        "%lhs_stride",
    ));
    block.push_op(VortexOp::binary(
        format!("%{prefix}_global_index32"),
        VortexBinaryOp::Add,
        LlvmType::I32,
        format!("%{prefix}_global_row_offset"),
        format!("%{prefix}_global_k"),
    ));
    block.push_op(VortexOp::cast(
        format!("%{prefix}_global_index"),
        CastKind::Zext,
        LlvmType::I32,
        format!("%{prefix}_global_index32"),
        LlvmType::I64,
    ));
    block.push_op(VortexOp::gep(
        format!("%{prefix}_global_ptr"),
        LlvmType::I8,
        "%lhs",
        format!("%{prefix}_global_index"),
        true,
    ));
    block.push_op(VortexOp::load(
        format!("%{prefix}_value"),
        LlvmType::I8,
        format!("%{prefix}_global_ptr"),
        1,
    ));
    block.push_op(VortexOp::store(
        LlvmType::I8,
        format!("%{prefix}_value"),
        format!("%{prefix}_local_ptr"),
        1,
    ));
}

fn push_rhs_stage_copy_blocks(
    kernel: &mut VortexKernelFunction,
    copy_index: u32,
    copy_offset: u32,
    next_label: impl Into<String>,
) {
    let next_label = next_label.into();
    let prefix = format!("rhs{copy_index}");
    let eval_label = format!("stage_rhs{copy_index}_eval");
    let load_label = format!("stage_rhs{copy_index}_load");
    let zero_label = format!("stage_rhs{copy_index}_zero");
    let join_label = format!("stage_rhs{copy_index}_join");

    let mut eval = VortexBlock::new(
        eval_label,
        VortexTerminator::cond_br(
            format!("%{prefix}_valid"),
            load_label.clone(),
            zero_label.clone(),
        ),
    );
    emit_rhs_stage_copy_eval(&mut eval, &prefix, copy_offset);
    kernel.push_block(eval);

    let mut load = VortexBlock::new(load_label, VortexTerminator::br(join_label.clone()));
    emit_rhs_stage_copy_load(&mut load, &prefix);
    kernel.push_block(load);

    let mut zero = VortexBlock::new(zero_label, VortexTerminator::br(join_label.clone()));
    zero.push_op(VortexOp::store(
        LlvmType::I8,
        "0",
        format!("%{prefix}_local_ptr"),
        1,
    ));
    kernel.push_block(zero);

    kernel.push_block(VortexBlock::new(
        join_label,
        VortexTerminator::br(next_label),
    ));
}

fn emit_rhs_stage_copy_eval(block: &mut VortexBlock, prefix: &str, copy_offset: u32) {
    block.push_op(VortexOp::binary(
        format!("%{prefix}_linear"),
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%thread_linear",
        copy_offset.to_string(),
    ));
    block.push_op(VortexOp::binary(
        format!("%{prefix}_row"),
        VortexBinaryOp::Udiv,
        LlvmType::I32,
        format!("%{prefix}_linear"),
        TILE_N.to_string(),
    ));
    block.push_op(VortexOp::binary(
        format!("%{prefix}_col"),
        VortexBinaryOp::Urem,
        LlvmType::I32,
        format!("%{prefix}_linear"),
        TILE_N.to_string(),
    ));
    block.push_op(VortexOp::binary(
        format!("%{prefix}_global_k"),
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%k_base",
        format!("%{prefix}_row"),
    ));
    block.push_op(VortexOp::binary(
        format!("%{prefix}_global_col"),
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%tile_col_base",
        format!("%{prefix}_col"),
    ));
    block.push_op(VortexOp::icmp(
        format!("%{prefix}_k_ok"),
        IcmpPredicate::Ult,
        LlvmType::I32,
        format!("%{prefix}_global_k"),
        "%k",
    ));
    block.push_op(VortexOp::icmp(
        format!("%{prefix}_col_ok"),
        IcmpPredicate::Ult,
        LlvmType::I32,
        format!("%{prefix}_global_col"),
        "%n",
    ));
    block.push_op(VortexOp::binary(
        format!("%{prefix}_valid"),
        VortexBinaryOp::And,
        LlvmType::I1,
        format!("%{prefix}_k_ok"),
        format!("%{prefix}_col_ok"),
    ));
    block.push_op(VortexOp::cast(
        format!("%{prefix}_local_index"),
        CastKind::Zext,
        LlvmType::I32,
        format!("%{prefix}_linear"),
        LlvmType::I64,
    ));
    block.push_op(VortexOp::gep(
        format!("%{prefix}_local_ptr"),
        LlvmType::I8,
        "%rhs_tile",
        format!("%{prefix}_local_index"),
        true,
    ));
}

fn emit_rhs_stage_copy_load(block: &mut VortexBlock, prefix: &str) {
    block.push_op(VortexOp::binary(
        format!("%{prefix}_global_row_offset"),
        VortexBinaryOp::Mul,
        LlvmType::I32,
        format!("%{prefix}_global_k"),
        "%rhs_stride",
    ));
    block.push_op(VortexOp::binary(
        format!("%{prefix}_global_index32"),
        VortexBinaryOp::Add,
        LlvmType::I32,
        format!("%{prefix}_global_row_offset"),
        format!("%{prefix}_global_col"),
    ));
    block.push_op(VortexOp::cast(
        format!("%{prefix}_global_index"),
        CastKind::Zext,
        LlvmType::I32,
        format!("%{prefix}_global_index32"),
        LlvmType::I64,
    ));
    block.push_op(VortexOp::gep(
        format!("%{prefix}_global_ptr"),
        LlvmType::I8,
        "%rhs",
        format!("%{prefix}_global_index"),
        true,
    ));
    block.push_op(VortexOp::load(
        format!("%{prefix}_value"),
        LlvmType::I8,
        format!("%{prefix}_global_ptr"),
        1,
    ));
    block.push_op(VortexOp::store(
        LlvmType::I8,
        format!("%{prefix}_value"),
        format!("%{prefix}_local_ptr"),
        1,
    ));
}

fn build_stage_barrier_block() -> VortexBlock {
    let mut block = VortexBlock::new(
        "stage_barrier",
        VortexTerminator::cond_br("%oob", "after_compute", "compute_loop"),
    );
    block.push_op(VortexOp::barrier("%cta_id", "%cta_size"));
    block
}

fn build_compute_loop_block() -> VortexBlock {
    let mut block = VortexBlock::new(
        "compute_loop",
        VortexTerminator::cond_br("%has_compute", "compute_body", "after_compute"),
    );
    block.push_op(VortexOp::phi(
        "%kk",
        LlvmType::I32,
        vec![
            PhiIncoming::new("0", "stage_barrier"),
            PhiIncoming::new("%kk_next", "compute_body"),
        ],
    ));
    block.push_op(VortexOp::phi(
        "%tile_acc",
        LlvmType::I32,
        vec![
            PhiIncoming::new("%acc", "stage_barrier"),
            PhiIncoming::new("%tile_acc_next", "compute_body"),
        ],
    ));
    block.push_op(VortexOp::icmp(
        "%kk_in_tile",
        IcmpPredicate::Ult,
        LlvmType::I32,
        "%kk",
        TILE_K.to_string(),
    ));
    block.push_op(VortexOp::binary(
        "%global_k_compute",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%k_base",
        "%kk",
    ));
    block.push_op(VortexOp::icmp(
        "%kk_in_bounds",
        IcmpPredicate::Ult,
        LlvmType::I32,
        "%global_k_compute",
        "%k",
    ));
    block.push_op(VortexOp::binary(
        "%has_compute",
        VortexBinaryOp::And,
        LlvmType::I1,
        "%kk_in_tile",
        "%kk_in_bounds",
    ));
    block
}

fn build_compute_body_block() -> VortexBlock {
    let mut block = VortexBlock::new("compute_body", VortexTerminator::br("compute_loop"));
    block.push_op(VortexOp::binary(
        "%lhs_compute_row_offset",
        VortexBinaryOp::Mul,
        LlvmType::I32,
        "%thread_y",
        TILE_K.to_string(),
    ));
    block.push_op(VortexOp::binary(
        "%lhs_compute_index32",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%lhs_compute_row_offset",
        "%kk",
    ));
    block.push_op(VortexOp::cast(
        "%lhs_compute_index",
        CastKind::Zext,
        LlvmType::I32,
        "%lhs_compute_index32",
        LlvmType::I64,
    ));
    block.push_op(VortexOp::gep(
        "%lhs_compute_ptr",
        LlvmType::I8,
        "%lhs_tile",
        "%lhs_compute_index",
        true,
    ));
    block.push_op(VortexOp::load("%a_i8", LlvmType::I8, "%lhs_compute_ptr", 1));
    block.push_op(VortexOp::cast(
        "%a",
        CastKind::Sext,
        LlvmType::I8,
        "%a_i8",
        LlvmType::I32,
    ));
    block.push_op(VortexOp::binary(
        "%rhs_compute_row_offset",
        VortexBinaryOp::Mul,
        LlvmType::I32,
        "%kk",
        TILE_N.to_string(),
    ));
    block.push_op(VortexOp::binary(
        "%rhs_compute_index32",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%rhs_compute_row_offset",
        "%thread_x",
    ));
    block.push_op(VortexOp::cast(
        "%rhs_compute_index",
        CastKind::Zext,
        LlvmType::I32,
        "%rhs_compute_index32",
        LlvmType::I64,
    ));
    block.push_op(VortexOp::gep(
        "%rhs_compute_ptr",
        LlvmType::I8,
        "%rhs_tile",
        "%rhs_compute_index",
        true,
    ));
    block.push_op(VortexOp::load("%b_i8", LlvmType::I8, "%rhs_compute_ptr", 1));
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
        "%tile_acc_next",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%tile_acc",
        "%prod",
    ));
    block.push_op(VortexOp::binary(
        "%kk_next",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%kk",
        "1",
    ));
    block
}

fn build_after_compute_block() -> VortexBlock {
    let mut block = VortexBlock::new("after_compute", VortexTerminator::br("k_continue"));
    block.push_op(VortexOp::phi(
        "%acc_after_tile",
        LlvmType::I32,
        vec![
            PhiIncoming::new("%acc", "stage_barrier"),
            PhiIncoming::new("%tile_acc", "compute_loop"),
        ],
    ));
    block.push_op(VortexOp::barrier("%cta_id", "%cta_size"));
    block
}

fn build_k_continue_block() -> VortexBlock {
    let mut block = VortexBlock::new("k_continue", VortexTerminator::br("k_loop"));
    block.push_op(VortexOp::binary(
        "%k_base_next",
        VortexBinaryOp::Add,
        LlvmType::I32,
        "%k_base",
        TILE_K.to_string(),
    ));
    block
}

fn build_store_check_block() -> VortexBlock {
    VortexBlock::new(
        "store_check",
        VortexTerminator::cond_br("%oob", "exit", "store"),
    )
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

#[cfg(test)]
mod tests {
    use super::{generate_experimental_vortex_tiled_matmul_llvm_ir, tiled_matmul_2d_semantics};
    use mandrel_compiler::compile_experimental_vortex_matmul_kernel;
    use mandrel_kernel_ir::{KernelImplementation, KernelSymbol};
    use mandrel_model_ir::MatmulOp;
    use mandrel_schedule::{MemoryScope, Shape2D};

    #[test]
    fn builds_tiled_schedule_semantics_from_compiler_plan() {
        let plan = match compile_experimental_vortex_matmul_kernel(MatmulOp::ggml_mul_mat_i8_demo())
        {
            Ok(plan) => plan,
            Err(error) => panic!("unexpected compile error: {error:?}"),
        };
        let semantics = tiled_matmul_2d_semantics(&plan);

        assert_eq!(semantics.lhs_layout.shape, Shape2D::new(32, 64));
        assert_eq!(semantics.rhs_layout.shape, Shape2D::new(64, 32));
        assert_eq!(semantics.out_layout.shape, Shape2D::new(32, 32));
        assert_eq!(semantics.output_thread_map.tile, Shape2D::new(4, 4));
        assert!(semantics.output_thread_map.covers_tile());
        assert_eq!(semantics.lhs_stage.src, MemoryScope::Global);
        assert_eq!(semantics.lhs_stage.dst, MemoryScope::Local);
        assert_eq!(semantics.lhs_stage.tile, Shape2D::new(4, 32));
        assert_eq!(semantics.rhs_stage.tile, Shape2D::new(32, 4));
    }

    #[test]
    fn generates_experimental_tiled_matmul_llvm_ir_from_planned_plan() {
        let plan = match compile_experimental_vortex_matmul_kernel(MatmulOp::ggml_mul_mat_i8_demo())
        {
            Ok(plan) => plan,
            Err(error) => panic!("unexpected compile error: {error:?}"),
        };
        let generated = match generate_experimental_vortex_tiled_matmul_llvm_ir(&plan) {
            Ok(generated) => generated,
            Err(error) => panic!("unexpected codegen error: {error}"),
        };

        assert_eq!(generated.symbol, KernelSymbol::MatmulI8I32Tiled);
        assert_eq!(
            generated.implementation,
            KernelImplementation::GeneratedVortexLlvmIr
        );
        assert!(generated.required_headers.is_empty());
        assert!(
            generated
                .source
                .contains("define dso_local void @matmul_i8_i32_tiled")
        );
        assert!(generated.source.contains("@llvm.global.annotations"));
        assert!(generated.source.contains("vortex.kernel"));
        assert!(generated.source.contains("i32 3295"));
        assert!(generated.source.contains(".insn r 11, 4, 0, x0, $0, $1"));
        assert!(
            generated
                .source
                .contains("%rhs_tile = getelementptr inbounds i8")
        );
        assert!(generated.source.contains("store i8 0"));
        assert!(generated.source.contains("stage_lhs7_eval"));
        assert!(generated.source.contains("stage_rhs7_eval"));
        assert!(generated.source.contains("%acc_after_tile = phi i32"));
    }
}
