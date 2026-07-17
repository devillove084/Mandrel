use mandrel_core::ElementType;
use mandrel_kernel_ir::{
    ATTENTION_PREFILL_I8_ARG_COUNT, Dim3, KernelArg, KernelLaunch, KernelSymbol, kernel_spec,
};
use mandrel_model_ir::{AttentionOp, Quantization};
use mandrel_profiler::{AttentionPrefillEstimateInput, estimate_gpu_attention_prefill};
use mandrel_schedule::{
    AttentionKernelKind, AttentionKvLayout, AttentionPrefillSchedule,
    AttentionPrefillScheduleSelection, Layout2D, select_vortex_attention_prefill_schedule,
};
use mandrel_target_ir::TargetConstraints;

use crate::plan::{CompileError, VortexKernelPlan, usize_to_u32};

pub const VORTEX_ATTENTION_PREFILL_ARG_COUNT: usize = ATTENTION_PREFILL_I8_ARG_COUNT;
pub const VORTEX_ATTENTION_Q_BUFFER: u32 = 0;
pub const VORTEX_ATTENTION_K_BUFFER: u32 = 1;
pub const VORTEX_ATTENTION_V_BUFFER: u32 = 2;
pub const VORTEX_ATTENTION_OUT_BUFFER: u32 = 3;
pub const VORTEX_ATTENTION_SEQUENCE_ARG: u32 = 4;
pub const VORTEX_ATTENTION_HEAD_DIM_ARG: u32 = 5;
pub const VORTEX_ATTENTION_QUERY_TILE_ARG: u32 = 6;
pub const VORTEX_ATTENTION_KEY_TILE_ARG: u32 = 7;

pub type VortexAttentionPrefillPlan = VortexKernelPlan<
    AttentionOp,
    AttentionPrefillSchedule,
    AttentionPrefillPlanMetadata,
    VORTEX_ATTENTION_PREFILL_ARG_COUNT,
>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionPrefillBufferRole {
    Query,
    Key,
    Value,
    Output,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionPrefillBufferMetadata {
    pub role: AttentionPrefillBufferRole,
    pub arg_index: u32,
    pub buffer_slot: u32,
    pub element_type: ElementType,
    pub layout: Layout2D,
    pub quantization: Option<Quantization>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionPrefillScalarMetadata {
    pub sequence_arg_index: u32,
    pub head_dim_arg_index: u32,
    pub query_tile_arg_index: u32,
    pub key_tile_arg_index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKvCacheMetadata {
    DenseContiguous {
        key: Layout2D,
        value: Layout2D,
    },
    Paged {
        page_size: usize,
        logical_key: Layout2D,
        logical_value: Layout2D,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionRuntimeShapePolicyKind {
    PrefixWithinCompiledShape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionRuntimeShapePolicy {
    pub kind: AttentionRuntimeShapePolicyKind,
    pub compiled_sequence: usize,
    pub compiled_head_dim: usize,
    pub default_runtime_sequence: usize,
    pub default_runtime_head_dim: usize,
    pub query_tile: usize,
    pub key_tile: usize,
    pub head_dim_tile: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionPrefillPlanMetadata {
    pub arg_count: usize,
    pub buffers: [AttentionPrefillBufferMetadata; 4],
    pub scalars: AttentionPrefillScalarMetadata,
    pub kv_cache: AttentionKvCacheMetadata,
    pub runtime_shape: AttentionRuntimeShapePolicy,
}

pub fn compile_vortex_attention_prefill_kernel(
    op: AttentionOp,
) -> Result<VortexAttentionPrefillPlan, CompileError> {
    compile_vortex_attention_prefill_kernel_for_target(op, TargetConstraints::vortex_simx_default())
}

pub fn compile_vortex_attention_prefill_kernel_for_target(
    op: AttentionOp,
    constraints: TargetConstraints,
) -> Result<VortexAttentionPrefillPlan, CompileError> {
    validate_attention_prefill_op(op)?;
    let selection = select_vortex_attention_prefill_schedule(op, constraints)
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
    let metadata = attention_prefill_plan_metadata(op, schedule);

    Ok(VortexKernelPlan {
        op,
        schedule,
        metadata,
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
        AttentionKernelKind::PrefillScalarTwoPass => KernelSymbol::AttentionPrefillI8,
    }
}

fn attention_prefill_plan_metadata(
    op: AttentionOp,
    schedule: AttentionPrefillSchedule,
) -> AttentionPrefillPlanMetadata {
    let logical_layout = Layout2D::row_major(op.shape.sequence, op.shape.head_dim);
    let unit_i8_quantization = Some(Quantization::symmetric_unit());
    let buffers = [
        AttentionPrefillBufferMetadata {
            role: AttentionPrefillBufferRole::Query,
            arg_index: 0,
            buffer_slot: VORTEX_ATTENTION_Q_BUFFER,
            element_type: op.element_type,
            layout: logical_layout,
            quantization: unit_i8_quantization,
        },
        AttentionPrefillBufferMetadata {
            role: AttentionPrefillBufferRole::Key,
            arg_index: 1,
            buffer_slot: VORTEX_ATTENTION_K_BUFFER,
            element_type: op.element_type,
            layout: logical_layout,
            quantization: unit_i8_quantization,
        },
        AttentionPrefillBufferMetadata {
            role: AttentionPrefillBufferRole::Value,
            arg_index: 2,
            buffer_slot: VORTEX_ATTENTION_V_BUFFER,
            element_type: op.element_type,
            layout: logical_layout,
            quantization: unit_i8_quantization,
        },
        AttentionPrefillBufferMetadata {
            role: AttentionPrefillBufferRole::Output,
            arg_index: 3,
            buffer_slot: VORTEX_ATTENTION_OUT_BUFFER,
            element_type: op.element_type,
            layout: logical_layout,
            quantization: unit_i8_quantization,
        },
    ];
    let kv_cache = match schedule.kv_layout {
        AttentionKvLayout::DenseContiguous => AttentionKvCacheMetadata::DenseContiguous {
            key: logical_layout,
            value: logical_layout,
        },
        AttentionKvLayout::Paged { page_size } => AttentionKvCacheMetadata::Paged {
            page_size,
            logical_key: logical_layout,
            logical_value: logical_layout,
        },
    };

    AttentionPrefillPlanMetadata {
        arg_count: VORTEX_ATTENTION_PREFILL_ARG_COUNT,
        buffers,
        scalars: AttentionPrefillScalarMetadata {
            sequence_arg_index: VORTEX_ATTENTION_SEQUENCE_ARG,
            head_dim_arg_index: VORTEX_ATTENTION_HEAD_DIM_ARG,
            query_tile_arg_index: VORTEX_ATTENTION_QUERY_TILE_ARG,
            key_tile_arg_index: VORTEX_ATTENTION_KEY_TILE_ARG,
        },
        kv_cache,
        runtime_shape: AttentionRuntimeShapePolicy {
            kind: AttentionRuntimeShapePolicyKind::PrefixWithinCompiledShape,
            compiled_sequence: op.shape.sequence,
            compiled_head_dim: op.shape.head_dim,
            default_runtime_sequence: op.shape.sequence.min(8),
            default_runtime_head_dim: op.shape.head_dim.min(16),
            query_tile: schedule.tile.query,
            key_tile: schedule.tile.key,
            head_dim_tile: schedule.tile.head_dim,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AttentionKvCacheMetadata, AttentionPrefillBufferRole, AttentionRuntimeShapePolicyKind,
        VORTEX_ATTENTION_HEAD_DIM_ARG, VORTEX_ATTENTION_KEY_TILE_ARG,
        VORTEX_ATTENTION_QUERY_TILE_ARG, VORTEX_ATTENTION_SEQUENCE_ARG, attention_kernel_symbol,
        compile_vortex_attention_prefill_kernel,
        compile_vortex_attention_prefill_kernel_for_target,
    };
    use mandrel_kernel_ir::{KernelArgValue, KernelAvailability, KernelSymbol};
    use mandrel_model_ir::{AttentionOp, Quantization};
    use mandrel_schedule::{AttentionKernelKind, Layout2D};
    use mandrel_target_ir::TargetConstraints;

    #[test]
    fn compiles_demo_to_available_attention_prefill_launch() {
        let plan = match compile_vortex_attention_prefill_kernel(AttentionOp::prefill_i8_demo()) {
            Ok(plan) => plan,
            Err(error) => panic!("unexpected compile error: {error:?}"),
        };

        assert_eq!(plan.kernel.availability, KernelAvailability::Available);
        assert_eq!(plan.launch.symbol, KernelSymbol::AttentionPrefillI8);
        assert_eq!(plan.launch.grid.x, 16);
        assert_eq!(plan.launch.grid.z, 1);
        assert_eq!(plan.launch.block.x, 4);
        assert_eq!(plan.launch.block.y, 4);
        assert_eq!(plan.launch.shared_memory_bytes, 0);
        assert_eq!(plan.launch.args[4].value, KernelArgValue::U32(64));
        assert_eq!(plan.launch.args[6].value, KernelArgValue::U32(4));
        assert_eq!(plan.launch.args[7].value, KernelArgValue::U32(1));
        assert_eq!(plan.metrics.workgroup_count, 16);
        assert_eq!(plan.metrics.lowered_macs, 33_816_576);

        assert_eq!(plan.metadata.arg_count, 8);
        assert_eq!(
            plan.metadata.scalars.sequence_arg_index,
            VORTEX_ATTENTION_SEQUENCE_ARG
        );
        assert_eq!(
            plan.metadata.scalars.head_dim_arg_index,
            VORTEX_ATTENTION_HEAD_DIM_ARG
        );
        assert_eq!(
            plan.metadata.scalars.query_tile_arg_index,
            VORTEX_ATTENTION_QUERY_TILE_ARG
        );
        assert_eq!(
            plan.metadata.scalars.key_tile_arg_index,
            VORTEX_ATTENTION_KEY_TILE_ARG
        );
        assert_eq!(
            plan.metadata.buffers[0].role,
            AttentionPrefillBufferRole::Query
        );
        assert_eq!(plan.metadata.buffers[0].arg_index, 0);
        assert_eq!(plan.metadata.buffers[0].buffer_slot, 0);
        assert_eq!(plan.metadata.buffers[0].layout, Layout2D::row_major(64, 64));
        assert_eq!(
            plan.metadata.buffers[0].quantization,
            Some(Quantization::symmetric_unit())
        );
        assert_eq!(
            plan.metadata.kv_cache,
            AttentionKvCacheMetadata::DenseContiguous {
                key: Layout2D::row_major(64, 64),
                value: Layout2D::row_major(64, 64)
            }
        );
        assert_eq!(
            plan.metadata.runtime_shape.kind,
            AttentionRuntimeShapePolicyKind::PrefixWithinCompiledShape
        );
        assert_eq!(plan.metadata.runtime_shape.default_runtime_sequence, 8);
        assert_eq!(plan.metadata.runtime_shape.default_runtime_head_dim, 16);
    }

    #[test]
    fn target_constraints_participate_in_schedule_selection() {
        let mut constraints = TargetConstraints::vortex_simx_default();
        constraints.max_workgroup_threads = 1;

        let result = compile_vortex_attention_prefill_kernel_for_target(
            AttentionOp::prefill_i8_demo(),
            constraints,
        );

        assert!(result.is_err());
    }

    #[test]
    fn maps_attention_kernel_kind_to_symbol() {
        assert_eq!(
            attention_kernel_symbol(AttentionKernelKind::PrefillScalarTwoPass),
            KernelSymbol::AttentionPrefillI8
        );
    }
}
