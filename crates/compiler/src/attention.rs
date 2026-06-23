use mandrel_core::ElementType;
use mandrel_kernel_ir::{Dim3, KernelArg, KernelLaunch, KernelSymbol, kernel_spec};
use mandrel_model_ir::{AttentionOp, TargetConstraints};
use mandrel_profiler::{AttentionPrefillEstimateInput, estimate_gpu_attention_prefill};
use mandrel_schedule::{
    AttentionKernelKind, AttentionPrefillSchedule, AttentionPrefillScheduleSelection,
    select_vortex_attention_prefill_schedule,
};

use crate::plan::{CompileError, VortexKernelPlan, usize_to_u32};

pub const VORTEX_ATTENTION_PREFILL_ARG_COUNT: usize = 8;
pub const VORTEX_ATTENTION_Q_BUFFER: u32 = 0;
pub const VORTEX_ATTENTION_K_BUFFER: u32 = 1;
pub const VORTEX_ATTENTION_V_BUFFER: u32 = 2;
pub const VORTEX_ATTENTION_OUT_BUFFER: u32 = 3;

pub type VortexAttentionPrefillPlan =
    VortexKernelPlan<AttentionOp, AttentionPrefillSchedule, VORTEX_ATTENTION_PREFILL_ARG_COUNT>;

pub fn compile_vortex_attention_prefill_kernel(
    op: AttentionOp,
) -> Result<VortexAttentionPrefillPlan, CompileError> {
    validate_attention_prefill_op(op)?;
    let selection =
        select_vortex_attention_prefill_schedule(op, TargetConstraints::vortex_simx_default())
            .map_err(|source| CompileError::ScheduleSelectionFailed { source })?;
    compile_selected_vortex_attention_prefill_kernel(op, selection)
}

fn validate_attention_prefill_op(op: AttentionOp) -> Result<(), CompileError> {
    if op.element_type != ElementType::I8 {
        return Err(CompileError::UnsupportedElementType);
    }
    if op.shape.sequence == 0 || op.shape.head_dim == 0 {
        return Err(CompileError::UnsupportedShape);
    }

    Ok(())
}

fn compile_selected_vortex_attention_prefill_kernel(
    op: AttentionOp,
    selection: AttentionPrefillScheduleSelection,
) -> Result<VortexAttentionPrefillPlan, CompileError> {
    let schedule = selection.schedule();
    let symbol = attention_kernel_symbol(selection.kernel());
    let kernel = match kernel_spec(symbol) {
        Some(spec) => spec,
        None => return Err(CompileError::KernelNotInCatalog { symbol }),
    };
    let metrics = match estimate_gpu_attention_prefill(AttentionPrefillEstimateInput::new(
        op.shape,
        schedule,
        op.element_type,
        op.element_type,
    )) {
        Some(metrics) => metrics,
        None => return Err(CompileError::MetricsOverflow),
    };

    let grid_x = usize_to_u32(selection.tile_counts.query_blocks)?;
    let grid_z = usize_to_u32(selection.tile_counts.head_dim_blocks)?;
    let block_x = usize_to_u32(schedule.block.x)?;
    let block_y = usize_to_u32(schedule.block.y)?;
    let shared_memory_bytes = usize_to_u32(selection.local_memory_bytes_per_workgroup)?;
    let sequence = usize_to_u32(op.shape.sequence)?;
    let head_dim = usize_to_u32(op.shape.head_dim)?;
    let query_tile = usize_to_u32(schedule.tile.query)?;
    let key_tile = usize_to_u32(schedule.tile.key)?;

    Ok(VortexKernelPlan {
        op,
        schedule,
        kernel,
        launch: KernelLaunch::new(
            symbol,
            Dim3::new(grid_x, 1, grid_z),
            Dim3::new(block_x, block_y, 1),
            shared_memory_bytes,
            [
                KernelArg::buffer(0, VORTEX_ATTENTION_Q_BUFFER),
                KernelArg::buffer(1, VORTEX_ATTENTION_K_BUFFER),
                KernelArg::buffer(2, VORTEX_ATTENTION_V_BUFFER),
                KernelArg::buffer(3, VORTEX_ATTENTION_OUT_BUFFER),
                KernelArg::u32(4, sequence),
                KernelArg::u32(5, head_dim),
                KernelArg::u32(6, query_tile),
                KernelArg::u32(7, key_tile),
            ],
        ),
        metrics,
    })
}

pub const fn attention_kernel_symbol(kind: AttentionKernelKind) -> KernelSymbol {
    match kind {
        AttentionKernelKind::PrefillOnlineSoftmax => KernelSymbol::AttentionPrefillI8,
    }
}

#[cfg(test)]
mod tests {
    use super::{attention_kernel_symbol, compile_vortex_attention_prefill_kernel};
    use mandrel_kernel_ir::{KernelArgValue, KernelAvailability, KernelSymbol};
    use mandrel_model_ir::AttentionOp;
    use mandrel_schedule::AttentionKernelKind;

    #[test]
    fn compiles_demo_to_planned_attention_prefill_launch() {
        let plan = match compile_vortex_attention_prefill_kernel(AttentionOp::prefill_i8_demo()) {
            Ok(plan) => plan,
            Err(error) => panic!("unexpected compile error: {error:?}"),
        };

        assert_eq!(plan.kernel.availability, KernelAvailability::Planned);
        assert_eq!(plan.launch.symbol, KernelSymbol::AttentionPrefillI8);
        assert_eq!(plan.launch.grid.x, 16);
        assert_eq!(plan.launch.grid.z, 1);
        assert_eq!(plan.launch.block.x, 4);
        assert_eq!(plan.launch.block.y, 4);
        assert_eq!(plan.launch.shared_memory_bytes, 2336);
        assert_eq!(plan.launch.args[4].value, KernelArgValue::U32(64));
        assert_eq!(plan.launch.args[6].value, KernelArgValue::U32(4));
        assert_eq!(plan.launch.args[7].value, KernelArgValue::U32(16));
        assert_eq!(plan.metrics.workgroup_count, 16);
    }

    #[test]
    fn maps_attention_kernel_kind_to_symbol() {
        assert_eq!(
            attention_kernel_symbol(AttentionKernelKind::PrefillOnlineSoftmax),
            KernelSymbol::AttentionPrefillI8
        );
    }
}
