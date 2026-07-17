use mandrel_compiler::{
    AttentionKvCacheMetadata, AttentionPrefillBufferMetadata, AttentionPrefillBufferRole,
    AttentionRuntimeShapePolicyKind, VORTEX_ATTENTION_HEAD_DIM_ARG, VORTEX_ATTENTION_K_BUFFER,
    VORTEX_ATTENTION_KEY_TILE_ARG, VORTEX_ATTENTION_OUT_BUFFER, VORTEX_ATTENTION_PREFILL_ARG_COUNT,
    VORTEX_ATTENTION_Q_BUFFER, VORTEX_ATTENTION_QUERY_TILE_ARG, VORTEX_ATTENTION_SEQUENCE_ARG,
    VORTEX_ATTENTION_V_BUFFER, VortexAttentionPrefillPlan,
};
use mandrel_kernel_ir::{KernelArg, KernelSymbol};
use mandrel_model_ir::Quantization;
use mandrel_schedule::{AttentionKvLayout, Layout2D};
use snafu::Snafu;

pub type AttentionPlanValidationResult<T> = Result<T, AttentionPlanValidationError>;

#[derive(Debug, Clone, PartialEq, Eq, Snafu)]
pub enum AttentionPlanValidationError {
    #[snafu(display("unsupported Vortex attention_prefill_i8 plan metadata: {reason}"))]
    UnsupportedMetadata { reason: String },
    #[snafu(display("invalid Vortex attention_prefill_i8 plan metadata: {reason}"))]
    InvalidMetadata { reason: String },
}

pub fn validate_attention_prefill_i8_plan_abi(
    plan: &VortexAttentionPrefillPlan,
) -> AttentionPlanValidationResult<()> {
    validate_kernel_identity(plan)?;
    validate_arg_count(plan)?;
    validate_buffers(plan)?;
    validate_scalars(plan)?;
    validate_kv_cache(plan)?;
    validate_runtime_shape(plan)?;
    validate_launch_args(plan)?;
    Ok(())
}

fn validate_kernel_identity(
    plan: &VortexAttentionPrefillPlan,
) -> AttentionPlanValidationResult<()> {
    if plan.kernel.symbol != KernelSymbol::AttentionPrefillI8 {
        return unsupported(format!(
            "expected kernel {}, got {}",
            KernelSymbol::AttentionPrefillI8.as_str(),
            plan.kernel.symbol.as_str()
        ));
    }
    if plan.launch.symbol != KernelSymbol::AttentionPrefillI8 {
        return invalid(format!(
            "launch symbol must be {}, got {}",
            KernelSymbol::AttentionPrefillI8.as_str(),
            plan.launch.symbol.as_str()
        ));
    }
    if !plan.kernel.signature.is_attention_prefill_i8() {
        return unsupported("kernel signature is not attention_prefill_i8".to_owned());
    }
    Ok(())
}

fn validate_arg_count(plan: &VortexAttentionPrefillPlan) -> AttentionPlanValidationResult<()> {
    if plan.metadata.arg_count != VORTEX_ATTENTION_PREFILL_ARG_COUNT {
        return invalid(format!(
            "metadata arg_count={} but ABI requires {}",
            plan.metadata.arg_count, VORTEX_ATTENTION_PREFILL_ARG_COUNT
        ));
    }
    if plan.launch.args.len() != VORTEX_ATTENTION_PREFILL_ARG_COUNT {
        return invalid(format!(
            "launch arg count={} but ABI requires {}",
            plan.launch.args.len(),
            VORTEX_ATTENTION_PREFILL_ARG_COUNT
        ));
    }
    Ok(())
}

fn validate_buffers(plan: &VortexAttentionPrefillPlan) -> AttentionPlanValidationResult<()> {
    let expected_layout = logical_layout(plan);
    let expected_quantization = Some(Quantization::symmetric_unit());
    let expected = [
        BufferExpectation::new(
            AttentionPrefillBufferRole::Query,
            0,
            VORTEX_ATTENTION_Q_BUFFER,
        ),
        BufferExpectation::new(
            AttentionPrefillBufferRole::Key,
            1,
            VORTEX_ATTENTION_K_BUFFER,
        ),
        BufferExpectation::new(
            AttentionPrefillBufferRole::Value,
            2,
            VORTEX_ATTENTION_V_BUFFER,
        ),
        BufferExpectation::new(
            AttentionPrefillBufferRole::Output,
            3,
            VORTEX_ATTENTION_OUT_BUFFER,
        ),
    ];

    for (actual, expected) in plan.metadata.buffers.iter().zip(expected) {
        validate_buffer(
            *actual,
            expected,
            expected_layout,
            expected_quantization,
            plan,
        )?;
    }
    Ok(())
}

fn validate_buffer(
    actual: AttentionPrefillBufferMetadata,
    expected: BufferExpectation,
    expected_layout: Layout2D,
    expected_quantization: Option<Quantization>,
    plan: &VortexAttentionPrefillPlan,
) -> AttentionPlanValidationResult<()> {
    if actual.role != expected.role {
        return invalid(format!(
            "buffer metadata order mismatch: expected {:?}, got {:?}",
            expected.role, actual.role
        ));
    }
    if actual.arg_index != expected.arg_index {
        return invalid(format!(
            "{:?} buffer arg_index={} but ABI requires {}",
            actual.role, actual.arg_index, expected.arg_index
        ));
    }
    if actual.buffer_slot != expected.buffer_slot {
        return invalid(format!(
            "{:?} buffer slot={} but ABI requires {}",
            actual.role, actual.buffer_slot, expected.buffer_slot
        ));
    }
    if actual.element_type != plan.op.element_type {
        return invalid(format!(
            "{:?} buffer element_type={:?} but op element_type={:?}",
            actual.role, actual.element_type, plan.op.element_type
        ));
    }
    if actual.layout != expected_layout {
        return invalid(format!(
            "{:?} buffer layout={:?} but dense row-major ABI requires {:?}",
            actual.role, actual.layout, expected_layout
        ));
    }
    if actual.quantization != expected_quantization {
        return invalid(format!(
            "{:?} buffer quantization={:?} but ABI requires {:?}",
            actual.role, actual.quantization, expected_quantization
        ));
    }
    Ok(())
}

fn validate_scalars(plan: &VortexAttentionPrefillPlan) -> AttentionPlanValidationResult<()> {
    let scalars = plan.metadata.scalars;
    validate_scalar_index(
        "sequence",
        scalars.sequence_arg_index,
        VORTEX_ATTENTION_SEQUENCE_ARG,
    )?;
    validate_scalar_index(
        "head_dim",
        scalars.head_dim_arg_index,
        VORTEX_ATTENTION_HEAD_DIM_ARG,
    )?;
    validate_scalar_index(
        "query_tile",
        scalars.query_tile_arg_index,
        VORTEX_ATTENTION_QUERY_TILE_ARG,
    )?;
    validate_scalar_index(
        "key_tile",
        scalars.key_tile_arg_index,
        VORTEX_ATTENTION_KEY_TILE_ARG,
    )?;
    Ok(())
}

fn validate_scalar_index(
    name: &str,
    actual: u32,
    expected: u32,
) -> AttentionPlanValidationResult<()> {
    if actual != expected {
        return invalid(format!(
            "scalar arg {name} uses index {actual}, but ABI requires {expected}"
        ));
    }
    Ok(())
}

fn validate_kv_cache(plan: &VortexAttentionPrefillPlan) -> AttentionPlanValidationResult<()> {
    if !matches!(plan.schedule.kv_layout, AttentionKvLayout::DenseContiguous) {
        return unsupported(
            "current Vortex attention_prefill_i8 lowering supports only dense contiguous schedule KV layout"
                .to_owned(),
        );
    }
    let expected_layout = logical_layout(plan);
    match plan.metadata.kv_cache {
        AttentionKvCacheMetadata::DenseContiguous { key, value } => {
            if key != expected_layout || value != expected_layout {
                return invalid(format!(
                    "dense KV metadata key={key:?} value={value:?} but ABI requires {expected_layout:?}"
                ));
            }
        }
        AttentionKvCacheMetadata::Paged { .. } => {
            return unsupported(
                "paged KV cache metadata is not supported by current Vortex attention_prefill_i8 ABI/lowering"
                    .to_owned(),
            );
        }
    }
    Ok(())
}

fn validate_runtime_shape(plan: &VortexAttentionPrefillPlan) -> AttentionPlanValidationResult<()> {
    let runtime = plan.metadata.runtime_shape;
    if runtime.kind != AttentionRuntimeShapePolicyKind::PrefixWithinCompiledShape {
        return unsupported(format!(
            "runtime shape policy {:?} is not supported by current runtime ABI",
            runtime.kind
        ));
    }
    if runtime.compiled_sequence != plan.op.shape.sequence
        || runtime.compiled_head_dim != plan.op.shape.head_dim
    {
        return invalid(format!(
            "runtime compiled shape=({}, {}) but op shape=({}, {})",
            runtime.compiled_sequence,
            runtime.compiled_head_dim,
            plan.op.shape.sequence,
            plan.op.shape.head_dim
        ));
    }
    if runtime.default_runtime_sequence == 0
        || runtime.default_runtime_head_dim == 0
        || runtime.default_runtime_sequence > runtime.compiled_sequence
        || runtime.default_runtime_head_dim > runtime.compiled_head_dim
    {
        return invalid(format!(
            "default runtime shape=({}, {}) must be nonzero and within compiled shape=({}, {})",
            runtime.default_runtime_sequence,
            runtime.default_runtime_head_dim,
            runtime.compiled_sequence,
            runtime.compiled_head_dim
        ));
    }
    if runtime.query_tile != plan.schedule.tile.query
        || runtime.key_tile != plan.schedule.tile.key
        || runtime.head_dim_tile != plan.schedule.tile.head_dim
    {
        return invalid(format!(
            "runtime tile=({}, {}, {}) but schedule tile=({}, {}, {})",
            runtime.query_tile,
            runtime.key_tile,
            runtime.head_dim_tile,
            plan.schedule.tile.query,
            plan.schedule.tile.key,
            plan.schedule.tile.head_dim
        ));
    }
    Ok(())
}

fn validate_launch_args(plan: &VortexAttentionPrefillPlan) -> AttentionPlanValidationResult<()> {
    let sequence = usize_to_u32(plan.op.shape.sequence, "sequence")?;
    let head_dim = usize_to_u32(plan.op.shape.head_dim, "head_dim")?;
    let query_tile = usize_to_u32(plan.schedule.tile.query, "query_tile")?;
    let key_tile = usize_to_u32(plan.schedule.tile.key, "key_tile")?;
    let expected = [
        KernelArg::buffer(0, VORTEX_ATTENTION_Q_BUFFER),
        KernelArg::buffer(1, VORTEX_ATTENTION_K_BUFFER),
        KernelArg::buffer(2, VORTEX_ATTENTION_V_BUFFER),
        KernelArg::buffer(3, VORTEX_ATTENTION_OUT_BUFFER),
        KernelArg::u32(4, sequence),
        KernelArg::u32(5, head_dim),
        KernelArg::u32(6, query_tile),
        KernelArg::u32(7, key_tile),
    ];

    for (actual, expected) in plan.launch.args.iter().zip(expected) {
        if *actual != expected {
            return invalid(format!(
                "launch arg {} mismatch: got {:?}, expected {:?}",
                expected.index, actual, expected
            ));
        }
    }
    Ok(())
}

fn logical_layout(plan: &VortexAttentionPrefillPlan) -> Layout2D {
    Layout2D::row_major(plan.op.shape.sequence, plan.op.shape.head_dim)
}

fn usize_to_u32(value: usize, name: &str) -> AttentionPlanValidationResult<u32> {
    match u32::try_from(value) {
        Ok(value) => Ok(value),
        Err(_) => invalid(format!("{name}={value} does not fit u32 ABI field")),
    }
}

fn unsupported<T>(reason: String) -> AttentionPlanValidationResult<T> {
    Err(AttentionPlanValidationError::UnsupportedMetadata { reason })
}

fn invalid<T>(reason: String) -> AttentionPlanValidationResult<T> {
    Err(AttentionPlanValidationError::InvalidMetadata { reason })
}

#[derive(Debug, Clone, Copy)]
struct BufferExpectation {
    role: AttentionPrefillBufferRole,
    arg_index: u32,
    buffer_slot: u32,
}

impl BufferExpectation {
    const fn new(role: AttentionPrefillBufferRole, arg_index: u32, buffer_slot: u32) -> Self {
        Self {
            role,
            arg_index,
            buffer_slot,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AttentionPlanValidationError, validate_attention_prefill_i8_plan_abi};
    use mandrel_compiler::{AttentionKvCacheMetadata, compile_vortex_attention_prefill_kernel};
    use mandrel_kernel_ir::{KernelArg, KernelArgValue};
    use mandrel_model_ir::AttentionOp;
    use mandrel_schedule::Layout2D;

    #[test]
    fn validates_compiler_attention_prefill_plan_abi() {
        let plan = demo_plan();

        assert!(validate_attention_prefill_i8_plan_abi(&plan).is_ok());
    }

    #[test]
    fn rejects_launch_arg_slot_mismatch() {
        let mut plan = demo_plan();
        plan.launch.args[0] = KernelArg {
            index: 0,
            value: KernelArgValue::Buffer(7),
        };

        let error = validate_attention_prefill_i8_plan_abi(&plan)
            .expect_err("expected launch arg mismatch");
        assert!(error.to_string().contains("launch arg 0 mismatch"));
    }

    #[test]
    fn rejects_metadata_buffer_layout_mismatch() {
        let mut plan = demo_plan();
        plan.metadata.buffers[1].layout = Layout2D::column_major(64, 64);

        let error =
            validate_attention_prefill_i8_plan_abi(&plan).expect_err("expected layout mismatch");
        assert!(error.to_string().contains("Key buffer layout"));
    }

    #[test]
    fn rejects_paged_kv_metadata_until_lowering_supports_it() {
        let mut plan = demo_plan();
        plan.metadata.kv_cache = AttentionKvCacheMetadata::Paged {
            page_size: 16,
            logical_key: Layout2D::row_major(64, 64),
            logical_value: Layout2D::row_major(64, 64),
        };

        let error = validate_attention_prefill_i8_plan_abi(&plan)
            .expect_err("expected paged KV metadata to be rejected");
        assert!(matches!(
            error,
            AttentionPlanValidationError::UnsupportedMetadata { .. }
        ));
        assert!(error.to_string().contains("paged KV cache metadata"));
    }

    #[test]
    fn rejects_runtime_shape_tile_mismatch() {
        let mut plan = demo_plan();
        plan.metadata.runtime_shape.key_tile = 8;

        let error = validate_attention_prefill_i8_plan_abi(&plan)
            .expect_err("expected runtime shape mismatch");
        assert!(error.to_string().contains("runtime tile"));
    }

    fn demo_plan() -> mandrel_compiler::VortexAttentionPrefillPlan {
        match compile_vortex_attention_prefill_kernel(AttentionOp::prefill_i8_demo()) {
            Ok(plan) => plan,
            Err(error) => panic!("unexpected compile error: {error:?}"),
        }
    }
}
