use mandrel_core::ElementType;
use mandrel_model_ir::{AttentionOp, AttentionShape, TargetConstraints};

use crate::error::ScheduleError;
use crate::layout::ThreadBlock2D;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionTileShape {
    pub query: usize,
    pub key: usize,
    pub head_dim: usize,
}

impl AttentionTileShape {
    pub const fn new(query: usize, key: usize, head_dim: usize) -> Self {
        Self {
            query,
            key,
            head_dim,
        }
    }

    pub const fn is_nonzero(self) -> bool {
        self.query != 0 && self.key != 0 && self.head_dim != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKvLayout {
    /// Q/K/V/O are dense row-major tensors for the first prefill scaffold.
    DenseContiguous,
    /// FlashInfer-style paged KV cache layout. This is schedule metadata for future decode/prefill
    /// kernels; codegen does not lower it yet.
    Paged { page_size: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionSoftmaxStrategy {
    /// Online max/sum reduction so a score row can be streamed block by block.
    OnlineMaxSum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionPrefillSchedule {
    pub tile: AttentionTileShape,
    pub block: ThreadBlock2D,
    pub kv_layout: AttentionKvLayout,
    pub softmax: AttentionSoftmaxStrategy,
}

impl AttentionPrefillSchedule {
    pub const fn new(
        tile: AttentionTileShape,
        block: ThreadBlock2D,
        kv_layout: AttentionKvLayout,
        softmax: AttentionSoftmaxStrategy,
    ) -> Self {
        Self {
            tile,
            block,
            kv_layout,
            softmax,
        }
    }

    /// First Vortex prefill candidate: one workgroup owns a small query block and streams key/value
    /// blocks with an online-softmax contract. The dense layout keeps the smoke path simple while
    /// retaining explicit tile fields needed by paged-attention lowering later.
    pub const fn dense_online_4x16x64() -> Self {
        Self::new(
            AttentionTileShape::new(4, 16, 64),
            ThreadBlock2D::new(4, 4),
            AttentionKvLayout::DenseContiguous,
            AttentionSoftmaxStrategy::OnlineMaxSum,
        )
    }

    pub fn tile_counts(self, shape: AttentionShape) -> Option<AttentionTileCounts> {
        if !self.tile.is_nonzero() {
            return None;
        }

        Some(AttentionTileCounts {
            query_blocks: shape.sequence.div_ceil(self.tile.query),
            key_blocks: shape.sequence.div_ceil(self.tile.key),
            head_dim_blocks: shape.head_dim.div_ceil(self.tile.head_dim),
        })
    }

    pub fn local_memory_bytes_per_workgroup(self, element_bytes: usize) -> Option<usize> {
        let query_tile = self
            .tile
            .query
            .checked_mul(self.tile.head_dim)?
            .checked_mul(element_bytes)?;
        let key_tile = self
            .tile
            .key
            .checked_mul(self.tile.head_dim)?
            .checked_mul(element_bytes)?;
        let value_tile = key_tile;
        let softmax_state = self.tile.query.checked_mul(2)?.checked_mul(4)?;

        query_tile
            .checked_add(key_tile)?
            .checked_add(value_tile)?
            .checked_add(softmax_state)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKernelKind {
    PrefillOnlineSoftmax,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionPrefillScheduleCandidate {
    pub kernel: AttentionKernelKind,
    pub schedule: AttentionPrefillSchedule,
}

impl AttentionPrefillScheduleCandidate {
    pub const fn dense_online_4x16x64() -> Self {
        Self {
            kernel: AttentionKernelKind::PrefillOnlineSoftmax,
            schedule: AttentionPrefillSchedule::dense_online_4x16x64(),
        }
    }
}

pub const VORTEX_ATTENTION_PREFILL_CANDIDATES: [AttentionPrefillScheduleCandidate; 1] =
    [AttentionPrefillScheduleCandidate::dense_online_4x16x64()];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionTileCounts {
    pub query_blocks: usize,
    pub key_blocks: usize,
    pub head_dim_blocks: usize,
}

impl AttentionTileCounts {
    pub fn workgroups(self) -> Option<usize> {
        self.query_blocks.checked_mul(self.head_dim_blocks)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionPrefillScheduleSelection {
    pub candidate: AttentionPrefillScheduleCandidate,
    pub tile_counts: AttentionTileCounts,
    pub local_memory_bytes_per_workgroup: usize,
}

impl AttentionPrefillScheduleSelection {
    pub const fn kernel(self) -> AttentionKernelKind {
        self.candidate.kernel
    }

    pub const fn schedule(self) -> AttentionPrefillSchedule {
        self.candidate.schedule
    }
}

pub fn select_vortex_attention_prefill_schedule(
    op: AttentionOp,
    constraints: TargetConstraints,
) -> Result<AttentionPrefillScheduleSelection, ScheduleError> {
    if op.shape.sequence == 0 || op.shape.head_dim == 0 {
        return Err(ScheduleError::EmptyShape);
    }
    if op.element_type != ElementType::I8 {
        return Err(ScheduleError::UnsupportedElementType);
    }

    for candidate in VORTEX_ATTENTION_PREFILL_CANDIDATES {
        let schedule = candidate.schedule;
        if schedule.block.thread_count() > constraints.max_workgroup_threads {
            continue;
        }
        let Some(tile_counts) = schedule.tile_counts(op.shape) else {
            continue;
        };
        let Some(local_memory_bytes_per_workgroup) =
            schedule.local_memory_bytes_per_workgroup(op.element_type.byte_size())
        else {
            return Err(ScheduleError::ShapeOverflow);
        };
        if local_memory_bytes_per_workgroup > constraints.local_memory_bytes {
            continue;
        }

        return Ok(AttentionPrefillScheduleSelection {
            candidate,
            tile_counts,
            local_memory_bytes_per_workgroup,
        });
    }

    Err(ScheduleError::NoViableCandidate)
}

#[cfg(test)]
mod tests {
    use super::{AttentionPrefillSchedule, select_vortex_attention_prefill_schedule};
    use mandrel_model_ir::{AttentionOp, TargetConstraints};

    #[test]
    fn selects_dense_online_prefill_schedule_for_demo_attention() {
        let selection = match select_vortex_attention_prefill_schedule(
            AttentionOp::prefill_i8_demo(),
            TargetConstraints::vortex_simx_default(),
        ) {
            Ok(selection) => selection,
            Err(error) => panic!("unexpected schedule error: {error:?}"),
        };

        assert_eq!(
            selection.schedule(),
            AttentionPrefillSchedule::dense_online_4x16x64()
        );
        assert_eq!(selection.tile_counts.query_blocks, 16);
        assert_eq!(selection.tile_counts.key_blocks, 4);
        assert_eq!(selection.tile_counts.workgroups(), Some(16));
        assert_eq!(selection.local_memory_bytes_per_workgroup, 2336);
    }
}
