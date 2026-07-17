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
    pub lowered_macs: usize,
    pub kernel_launches: usize,
    pub workgroup_count: usize,
    pub thread_count: usize,
    pub global_bytes_read: usize,
    pub global_bytes_written: usize,
    pub local_memory_bytes_per_workgroup: usize,
}

impl KernelMetrics {
    pub fn logical_operational_intensity(self) -> Option<Ratio> {
        Ratio::new(
            self.logical_macs,
            self.global_bytes_read
                .checked_add(self.global_bytes_written)?,
        )
    }

    pub fn lowered_operational_intensity(self) -> Option<Ratio> {
        Ratio::new(
            self.lowered_macs,
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
    let sequence_squared = input.shape.sequence.checked_mul(input.shape.sequence)?;
    let output_elements = input.shape.sequence.checked_mul(input.shape.head_dim)?;
    let logical_score_macs = sequence_squared.checked_mul(input.shape.head_dim)?;
    let logical_value_macs = logical_score_macs;
    let logical_macs = logical_score_macs.checked_add(logical_value_macs)?;

    // The current scalar lowering recomputes the full QK row twice for every output dimension.
    let lowered_qk_macs = logical_score_macs
        .checked_mul(input.shape.head_dim)?
        .checked_mul(2)?;
    let lowered_macs = lowered_qk_macs.checked_add(logical_value_macs)?;

    // Every scalar QK MAC loads one Q and one K element; every value MAC loads one V element.
    let qk_bytes_read = lowered_qk_macs
        .checked_mul(2)?
        .checked_mul(input.element_type.byte_size())?;
    let value_bytes_read = logical_value_macs.checked_mul(input.element_type.byte_size())?;
    let global_bytes_read = qk_bytes_read.checked_add(value_bytes_read)?;
    let global_bytes_written = output_elements.checked_mul(input.out_type.byte_size())?;
    let local_memory_bytes_per_workgroup = input
        .schedule
        .local_memory_bytes_per_workgroup(input.element_type.byte_size())?;
    let thread_count = workgroup_count.checked_mul(input.schedule.block.thread_count())?;

    Some(KernelMetrics {
        logical_macs,
        lowered_macs,
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
    fn estimates_executed_scalar_attention_prefill_lowering() {
        let metrics = match estimate_gpu_attention_prefill(AttentionPrefillEstimateInput::new(
            AttentionShape::new(64, 64),
            AttentionPrefillSchedule::dense_scalar_two_pass_4x1x64(),
            ElementType::I8,
            ElementType::I8,
        )) {
            Some(metrics) => metrics,
            None => panic!("expected attention metrics"),
        };

        assert_eq!(metrics.logical_macs, 524_288);
        assert_eq!(metrics.lowered_macs, 33_816_576);
        assert_eq!(metrics.workgroup_count, 16);
        assert_eq!(metrics.thread_count, 256);
        assert_eq!(metrics.global_bytes_read, 67_371_008);
        assert_eq!(metrics.global_bytes_written, 4_096);
        assert_eq!(metrics.local_memory_bytes_per_workgroup, 0);
    }
}
