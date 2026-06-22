use mandrel_core::ElementType;
use mandrel_model_ir::MatmulShape;
use mandrel_schedule::MatmulSchedule;

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
pub struct MatmulEstimateInput {
    pub shape: MatmulShape,
    pub schedule: MatmulSchedule,
    pub lhs_type: ElementType,
    pub rhs_type: ElementType,
    pub out_type: ElementType,
}

impl MatmulEstimateInput {
    pub const fn new(
        shape: MatmulShape,
        schedule: MatmulSchedule,
        lhs_type: ElementType,
        rhs_type: ElementType,
        out_type: ElementType,
    ) -> Self {
        Self {
            shape,
            schedule,
            lhs_type,
            rhs_type,
            out_type,
        }
    }
}

pub fn estimate_gpu_matmul(input: MatmulEstimateInput) -> Option<KernelMetrics> {
    let tile = input.schedule.tile;
    let counts = input.schedule.tile_counts(input.shape)?;
    let compute_tiles = counts.compute_tiles()?;
    let output_tiles = counts.output_tiles()?;

    let logical_macs = input
        .shape
        .m
        .checked_mul(input.shape.n)?
        .checked_mul(input.shape.k)?;
    let scheduled_macs = tile
        .m
        .checked_mul(tile.n)?
        .checked_mul(tile.k)?
        .checked_mul(compute_tiles)?;

    let lhs_tile_bytes = tile
        .m
        .checked_mul(tile.k)?
        .checked_mul(input.lhs_type.byte_size())?;
    let rhs_tile_bytes = tile
        .k
        .checked_mul(tile.n)?
        .checked_mul(input.rhs_type.byte_size())?;
    let out_tile_bytes = tile
        .m
        .checked_mul(tile.n)?
        .checked_mul(input.out_type.byte_size())?;

    let global_bytes_read = lhs_tile_bytes
        .checked_add(rhs_tile_bytes)?
        .checked_mul(compute_tiles)?;
    let global_bytes_written = out_tile_bytes.checked_mul(output_tiles)?;
    let local_memory_bytes_per_workgroup = input
        .schedule
        .local_memory_bytes_per_k_tile(input.lhs_type.byte_size(), input.rhs_type.byte_size())?;
    let thread_count = output_tiles.checked_mul(input.schedule.block.thread_count())?;

    Some(KernelMetrics {
        logical_macs,
        scheduled_macs,
        kernel_launches: 1,
        workgroup_count: output_tiles,
        thread_count,
        global_bytes_read,
        global_bytes_written,
        local_memory_bytes_per_workgroup,
    })
}

#[cfg(test)]
mod tests {
    use super::{MatmulEstimateInput, estimate_gpu_matmul};
    use mandrel_core::ElementType;
    use mandrel_model_ir::MatmulShape;
    use mandrel_schedule::MatmulSchedule;

    #[test]
    fn estimates_vortex_matmul() {
        let metrics = match estimate_gpu_matmul(MatmulEstimateInput::new(
            MatmulShape::new(32, 32, 64),
            MatmulSchedule::vortex_threadblock_4x4x32(),
            ElementType::I8,
            ElementType::I8,
            ElementType::I32,
        )) {
            Some(metrics) => metrics,
            None => panic!("expected matmul metrics"),
        };

        assert_eq!(metrics.logical_macs, 65_536);
        assert_eq!(metrics.kernel_launches, 1);
        assert_eq!(metrics.workgroup_count, 64);
        assert_eq!(metrics.global_bytes_written, 4096);
        assert_eq!(metrics.local_memory_bytes_per_workgroup, 256);
    }

    #[test]
    fn estimates_current_direct_matmul_schedule() {
        let metrics = match estimate_gpu_matmul(MatmulEstimateInput::new(
            MatmulShape::new(32, 32, 64),
            MatmulSchedule::direct_thread_per_output_4x4(),
            ElementType::I8,
            ElementType::I8,
            ElementType::I32,
        )) {
            Some(metrics) => metrics,
            None => panic!("expected matmul metrics"),
        };

        assert_eq!(metrics.logical_macs, 65_536);
        assert_eq!(metrics.kernel_launches, 1);
        assert_eq!(metrics.workgroup_count, 64);
        assert_eq!(metrics.local_memory_bytes_per_workgroup, 0);
    }
}
