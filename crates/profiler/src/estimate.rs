use mandrel_core::ElementType;
use mandrel_model_ir::AttentionShape;
use mandrel_schedule::AttentionPrefillSchedule;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ratio {
    pub numerator: usize,
    pub denominator: usize,
}

impl Ratio {
    pub const fn new(numerator: usize, denominator: usize) -> Option<Self> {
        if denominator == 0 {
            None
        } else {
            Some(Self {
                numerator,
                denominator,
            })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelMetrics {
    pub logical_macs: usize,
    pub scheduled_macs: usize,
    pub kernel_launches: usize,
    pub workgroup_count: usize,
    pub thread_count: usize,
    pub global_bytes_read: usize,
    pub global_bytes_written: usize,
    pub local_memory_bytes_per_workgroup: usize,
}

impl KernelMetrics {
    pub fn operational_intensity(self) -> Option<Ratio> {
        Ratio::new(
            self.scheduled_macs,
            self.global_bytes_read
                .checked_add(self.global_bytes_written)?,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionPrefillEstimateInput {
    pub shape: AttentionShape,
    pub schedule: AttentionPrefillSchedule,
    pub element_type: ElementType,
    pub out_type: ElementType,
}

impl AttentionPrefillEstimateInput {
    pub const fn new(
        shape: AttentionShape,
        schedule: AttentionPrefillSchedule,
        element_type: ElementType,
        out_type: ElementType,
    ) -> Self {
        Self {
            shape,
            schedule,
            element_type,
            out_type,
        }
    }
}

pub fn estimate_gpu_attention_prefill(
    input: AttentionPrefillEstimateInput,
) -> Option<KernelMetrics> {
    let counts = input.schedule.tile_counts(input.shape)?;
    let workgroup_count = counts.workgroups()?;
    let logical_score_macs = input
        .shape
        .sequence
        .checked_mul(input.shape.sequence)?
        .checked_mul(input.shape.head_dim)?;
    let logical_value_macs = logical_score_macs;
    let logical_macs = logical_score_macs.checked_add(logical_value_macs)?;

    let scheduled_score_macs = counts
        .query_blocks
        .checked_mul(counts.key_blocks)?
        .checked_mul(input.schedule.tile.query)?
        .checked_mul(input.schedule.tile.key)?
        .checked_mul(input.schedule.tile.head_dim)?;
    let scheduled_macs = scheduled_score_macs.checked_mul(2)?;

    let q_tile_bytes = input
        .schedule
        .tile
        .query
        .checked_mul(input.schedule.tile.head_dim)?
        .checked_mul(input.element_type.byte_size())?;
    let k_tile_bytes = input
        .schedule
        .tile
        .key
        .checked_mul(input.schedule.tile.head_dim)?
        .checked_mul(input.element_type.byte_size())?;
    let v_tile_bytes = k_tile_bytes;
    let q_bytes_read = q_tile_bytes.checked_mul(counts.query_blocks)?;
    let kv_tile_bytes = k_tile_bytes.checked_add(v_tile_bytes)?;
    let kv_bytes_read = kv_tile_bytes
        .checked_mul(counts.query_blocks)?
        .checked_mul(counts.key_blocks)?;
    let global_bytes_read = q_bytes_read.checked_add(kv_bytes_read)?;
    let global_bytes_written = input
        .shape
        .sequence
        .checked_mul(input.shape.head_dim)?
        .checked_mul(input.out_type.byte_size())?;
    let local_memory_bytes_per_workgroup = input
        .schedule
        .local_memory_bytes_per_workgroup(input.element_type.byte_size())?;
    let thread_count = workgroup_count.checked_mul(input.schedule.block.thread_count())?;

    Some(KernelMetrics {
        logical_macs,
        scheduled_macs,
        kernel_launches: 1,
        workgroup_count,
        thread_count,
        global_bytes_read,
        global_bytes_written,
        local_memory_bytes_per_workgroup,
    })
}

#[cfg(test)]
mod tests {
    use super::{AttentionPrefillEstimateInput, estimate_gpu_attention_prefill};
    use mandrel_core::ElementType;
    use mandrel_model_ir::AttentionShape;
    use mandrel_schedule::AttentionPrefillSchedule;

    #[test]
    fn estimates_attention_prefill_schedule() {
        let metrics = match estimate_gpu_attention_prefill(AttentionPrefillEstimateInput::new(
            AttentionShape::new(64, 64),
            AttentionPrefillSchedule::dense_online_4x16x64(),
            ElementType::I8,
            ElementType::I8,
        )) {
            Some(metrics) => metrics,
            None => panic!("expected attention metrics"),
        };

        assert_eq!(metrics.logical_macs, 524_288);
        assert_eq!(metrics.scheduled_macs, 524_288);
        assert_eq!(metrics.workgroup_count, 16);
        assert_eq!(metrics.thread_count, 256);
        assert_eq!(metrics.global_bytes_read, 135_168);
        assert_eq!(metrics.local_memory_bytes_per_workgroup, 2336);
    }
}
