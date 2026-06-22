use mandrel_core::ElementType;
use mandrel_model_ir::{MatmulOp, MatmulShape, TargetConstraints};

use crate::layout::{Shape2D, ThreadBlock2D, ThreadTileMap2D};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl TileShape {
    pub const fn new(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k }
    }

    pub const fn is_nonzero(self) -> bool {
        self.m != 0 && self.n != 0 && self.k != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadBlockShape {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl ThreadBlockShape {
    pub const fn new(x: usize, y: usize, z: usize) -> Self {
        Self { x, y, z }
    }

    pub const fn thread_count(self) -> usize {
        self.x * self.y * self.z
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopOrder {
    Mnk,
    Mkn,
    Nmk,
    Nkm,
    Kmn,
    Knm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReuseStrategy {
    None,
    ReuseLhsAcrossN,
    ReuseRhsAcrossM,
    AccumulateAcrossK,
    UseLocalMemoryTiles,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulSchedule {
    pub tile: TileShape,
    pub block: ThreadBlockShape,
    pub loop_order: LoopOrder,
    pub reuse: ReuseStrategy,
}

impl MatmulSchedule {
    pub const fn new(
        tile: TileShape,
        block: ThreadBlockShape,
        loop_order: LoopOrder,
        reuse: ReuseStrategy,
    ) -> Self {
        Self {
            tile,
            block,
            loop_order,
            reuse,
        }
    }

    /// Small direct-mapped schedule that matches the first hand-written Vortex kernel.
    /// Each workgroup owns one output tile and maps one thread to one output cell.
    pub const fn direct_thread_per_output_4x4() -> Self {
        Self::new(
            TileShape::new(4, 4, 1),
            ThreadBlockShape::new(4, 4, 1),
            LoopOrder::Mnk,
            ReuseStrategy::None,
        )
    }

    /// Simx-compatible local-memory tiled candidate for the current default Vortex config.
    ///
    /// The default simx build exposes 4 warps × 4 threads, so a workgroup must stay at or below
    /// 16 threads. This schedule still exercises local-memory staging and barriers while matching
    /// that hardware limit.
    pub const fn tiled_local_4x4x32() -> Self {
        Self::new(
            TileShape::new(4, 4, 32),
            ThreadBlockShape::new(4, 4, 1),
            LoopOrder::Mnk,
            ReuseStrategy::UseLocalMemoryTiles,
        )
    }

    /// Larger local-memory candidate for future Vortex configs with at least 256 hardware threads
    /// per workgroup. It is intentionally not part of the current experimental simx candidate set.
    pub const fn tiled_local_16x16x32() -> Self {
        Self::new(
            TileShape::new(16, 16, 32),
            ThreadBlockShape::new(16, 16, 1),
            LoopOrder::Mnk,
            ReuseStrategy::UseLocalMemoryTiles,
        )
    }

    /// Current Vortex simx local-memory tiled schedule.
    pub const fn vortex_threadblock_4x4x32() -> Self {
        Self::tiled_local_4x4x32()
    }

    /// Backward-compatible alias for older docs/tests. New code should choose schedules through
    /// `select_vortex_matmul_schedule` so we do not bake one matmul kernel into the compiler.
    pub const fn vortex_threadblock_16x16x32() -> Self {
        Self::tiled_local_16x16x32()
    }

    pub const fn output_thread_map(self) -> ThreadTileMap2D {
        ThreadTileMap2D::new(
            Shape2D::new(self.tile.m, self.tile.n),
            ThreadBlock2D::new(self.block.x, self.block.y),
            Shape2D::new(1, 1),
        )
    }

    pub fn tile_counts(self, shape: MatmulShape) -> Option<MatmulTileCounts> {
        if !self.tile.is_nonzero() {
            return None;
        }

        Some(MatmulTileCounts {
            m_tiles: shape.m.div_ceil(self.tile.m),
            n_tiles: shape.n.div_ceil(self.tile.n),
            k_tiles: shape.k.div_ceil(self.tile.k),
        })
    }

    pub fn local_memory_bytes_per_k_tile(
        self,
        lhs_element_bytes: usize,
        rhs_element_bytes: usize,
    ) -> Option<usize> {
        if !matches!(self.reuse, ReuseStrategy::UseLocalMemoryTiles) {
            return Some(0);
        }

        let lhs = self
            .tile
            .m
            .checked_mul(self.tile.k)?
            .checked_mul(lhs_element_bytes)?;
        let rhs = self
            .tile
            .k
            .checked_mul(self.tile.n)?
            .checked_mul(rhs_element_bytes)?;
        lhs.checked_add(rhs)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulTileCounts {
    pub m_tiles: usize,
    pub n_tiles: usize,
    pub k_tiles: usize,
}

impl MatmulTileCounts {
    pub fn compute_tiles(self) -> Option<usize> {
        self.m_tiles
            .checked_mul(self.n_tiles)?
            .checked_mul(self.k_tiles)
    }

    pub fn output_tiles(self) -> Option<usize> {
        self.m_tiles.checked_mul(self.n_tiles)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulKernelKind {
    DirectThreadPerOutput,
    TiledLocalMemory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulScheduleCandidate {
    pub kernel: MatmulKernelKind,
    pub schedule: MatmulSchedule,
}

impl MatmulScheduleCandidate {
    pub const fn direct_thread_per_output_4x4() -> Self {
        Self {
            kernel: MatmulKernelKind::DirectThreadPerOutput,
            schedule: MatmulSchedule::direct_thread_per_output_4x4(),
        }
    }

    pub const fn tiled_local_4x4x32() -> Self {
        Self {
            kernel: MatmulKernelKind::TiledLocalMemory,
            schedule: MatmulSchedule::tiled_local_4x4x32(),
        }
    }

    pub const fn tiled_local_16x16x32() -> Self {
        Self {
            kernel: MatmulKernelKind::TiledLocalMemory,
            schedule: MatmulSchedule::tiled_local_16x16x32(),
        }
    }
}

pub const VORTEX_AVAILABLE_MATMUL_CANDIDATES: [MatmulScheduleCandidate; 1] =
    [MatmulScheduleCandidate::direct_thread_per_output_4x4()];

pub const VORTEX_EXPERIMENTAL_MATMUL_CANDIDATES: [MatmulScheduleCandidate; 2] = [
    MatmulScheduleCandidate::direct_thread_per_output_4x4(),
    MatmulScheduleCandidate::tiled_local_4x4x32(),
];

/// Default Vortex matmul candidates only include kernels that are implemented in this project.
pub const VORTEX_MATMUL_CANDIDATES: [MatmulScheduleCandidate; 1] =
    VORTEX_AVAILABLE_MATMUL_CANDIDATES;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleError {
    EmptyShape,
    UnsupportedElementType,
    NoViableCandidate,
    ShapeOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulScheduleSelection {
    pub candidate: MatmulScheduleCandidate,
    pub tile_counts: MatmulTileCounts,
    pub local_memory_bytes_per_workgroup: usize,
}

impl MatmulScheduleSelection {
    pub const fn kernel(self) -> MatmulKernelKind {
        self.candidate.kernel
    }

    pub const fn schedule(self) -> MatmulSchedule {
        self.candidate.schedule
    }
}

pub fn select_vortex_matmul_schedule(
    op: MatmulOp,
    constraints: TargetConstraints,
) -> Result<MatmulScheduleSelection, ScheduleError> {
    select_vortex_matmul_schedule_from_candidates(op, constraints, &VORTEX_MATMUL_CANDIDATES)
}

pub fn select_experimental_vortex_matmul_schedule(
    op: MatmulOp,
    constraints: TargetConstraints,
) -> Result<MatmulScheduleSelection, ScheduleError> {
    select_vortex_matmul_schedule_from_candidates(
        op,
        constraints,
        &VORTEX_EXPERIMENTAL_MATMUL_CANDIDATES,
    )
}

pub fn select_vortex_matmul_schedule_from_candidates(
    op: MatmulOp,
    constraints: TargetConstraints,
    candidates: &[MatmulScheduleCandidate],
) -> Result<MatmulScheduleSelection, ScheduleError> {
    if op.shape.m == 0 || op.shape.n == 0 || op.shape.k == 0 {
        return Err(ScheduleError::EmptyShape);
    }
    if op.types.lhs != ElementType::I8
        || op.types.rhs != ElementType::I8
        || op.types.out != ElementType::I32
    {
        return Err(ScheduleError::UnsupportedElementType);
    }
    if !constraints.supports_int8 {
        return Err(ScheduleError::UnsupportedElementType);
    }

    let mut index = 0;
    let mut best: Option<MatmulScheduleSelection> = None;
    while index < candidates.len() {
        let candidate = candidates[index];
        if let Some(selection) = viable_matmul_candidate(candidate, op, constraints)? {
            best = choose_better_matmul_schedule(best, selection);
        }
        index += 1;
    }

    match best {
        Some(selection) => Ok(selection),
        None => Err(ScheduleError::NoViableCandidate),
    }
}

fn viable_matmul_candidate(
    candidate: MatmulScheduleCandidate,
    op: MatmulOp,
    constraints: TargetConstraints,
) -> Result<Option<MatmulScheduleSelection>, ScheduleError> {
    let schedule = candidate.schedule;
    if schedule.block.thread_count() > constraints.max_workgroup_threads {
        return Ok(None);
    }
    let Some(tile_counts) = schedule.tile_counts(op.shape) else {
        return Ok(None);
    };
    let Some(local_memory_bytes_per_workgroup) =
        schedule.local_memory_bytes_per_k_tile(op.types.lhs.byte_size(), op.types.rhs.byte_size())
    else {
        return Err(ScheduleError::ShapeOverflow);
    };
    if local_memory_bytes_per_workgroup > constraints.local_memory_bytes {
        return Ok(None);
    }

    Ok(Some(MatmulScheduleSelection {
        candidate,
        tile_counts,
        local_memory_bytes_per_workgroup,
    }))
}

fn choose_better_matmul_schedule(
    current: Option<MatmulScheduleSelection>,
    candidate: MatmulScheduleSelection,
) -> Option<MatmulScheduleSelection> {
    match current {
        None => Some(candidate),
        Some(current) if candidate_score(candidate) < candidate_score(current) => Some(candidate),
        Some(current) => Some(current),
    }
}

fn candidate_score(selection: MatmulScheduleSelection) -> usize {
    let schedule = selection.schedule();
    let output_tiles = selection
        .tile_counts
        .output_tiles()
        .unwrap_or(usize::MAX / 4);
    let compute_tiles = selection
        .tile_counts
        .compute_tiles()
        .unwrap_or(usize::MAX / 4);
    let reuse_penalty = match schedule.reuse {
        ReuseStrategy::UseLocalMemoryTiles => 0,
        ReuseStrategy::AccumulateAcrossK => 1,
        ReuseStrategy::ReuseLhsAcrossN | ReuseStrategy::ReuseRhsAcrossM => 2,
        ReuseStrategy::None => 8,
    };

    compute_tiles
        .saturating_mul(16)
        .saturating_add(output_tiles)
        .saturating_add(reuse_penalty)
}

#[cfg(test)]
mod tests {
    use super::{
        MatmulKernelKind, MatmulSchedule, ScheduleError,
        select_experimental_vortex_matmul_schedule, select_vortex_matmul_schedule,
        select_vortex_matmul_schedule_from_candidates,
    };
    use mandrel_core::ElementType;
    use mandrel_model_ir::{MatmulOp, MatmulShape, MatmulTensors, MatmulTypes, TargetConstraints};

    #[test]
    fn computes_tile_counts() {
        let schedule = MatmulSchedule::vortex_threadblock_4x4x32();
        let counts = match schedule.tile_counts(MatmulShape::new(32, 48, 64)) {
            Some(counts) => counts,
            None => panic!("expected nonzero tile counts"),
        };

        assert_eq!(counts.m_tiles, 8);
        assert_eq!(counts.n_tiles, 12);
        assert_eq!(counts.k_tiles, 2);
        assert_eq!(counts.compute_tiles(), Some(192));
        assert_eq!(schedule.block.thread_count(), 16);
    }

    #[test]
    fn exposes_cute_like_output_thread_map() {
        let schedule = MatmulSchedule::direct_thread_per_output_4x4();
        let mapping = schedule.output_thread_map();

        assert_eq!(mapping.tile.rows, 4);
        assert_eq!(mapping.tile.cols, 4);
        assert_eq!(mapping.block.x, 4);
        assert_eq!(mapping.block.y, 4);
        assert!(mapping.is_one_value_per_thread());
        assert_eq!(mapping.logical_coord(3, 2, 0, 0), Some([2, 3]));
    }

    #[test]
    fn selects_current_available_direct_candidate_by_default() {
        let selection = match select_vortex_matmul_schedule(
            MatmulOp::new(
                MatmulTensors::new(0, 1, 2),
                MatmulShape::new(32, 48, 64),
                MatmulTypes::i8_to_i32(),
            ),
            TargetConstraints::vortex_simx_default(),
        ) {
            Ok(selection) => selection,
            Err(error) => panic!("unexpected schedule error: {error:?}"),
        };

        assert_eq!(selection.kernel(), MatmulKernelKind::DirectThreadPerOutput);
        assert_eq!(selection.schedule().tile.m, 4);
    }

    #[test]
    fn selects_tiled_experimental_candidate_when_resources_allow_it() {
        let selection = match select_experimental_vortex_matmul_schedule(
            MatmulOp::new(
                MatmulTensors::new(0, 1, 2),
                MatmulShape::new(32, 48, 64),
                MatmulTypes::i8_to_i32(),
            ),
            TargetConstraints::vortex_simx_default(),
        ) {
            Ok(selection) => selection,
            Err(error) => panic!("unexpected schedule error: {error:?}"),
        };

        assert_eq!(selection.kernel(), MatmulKernelKind::TiledLocalMemory);
        assert_eq!(selection.schedule(), MatmulSchedule::tiled_local_4x4x32());
    }

    #[test]
    fn reports_no_viable_candidate_for_empty_candidate_set() {
        let error = select_vortex_matmul_schedule_from_candidates(
            MatmulOp::new(
                MatmulTensors::new(0, 1, 2),
                MatmulShape::new(8, 8, 8),
                MatmulTypes::i8_to_i32(),
            ),
            TargetConstraints::vortex_simx_default(),
            &[],
        );

        assert_eq!(error, Err(ScheduleError::NoViableCandidate));
    }

    #[test]
    fn falls_back_to_direct_candidate_when_local_memory_is_tight() {
        let mut constraints = TargetConstraints::vortex_simx_default();
        constraints.local_memory_bytes = 64;
        let selection = match select_vortex_matmul_schedule(
            MatmulOp::new(
                MatmulTensors::new(0, 1, 2),
                MatmulShape::new(8, 8, 8),
                MatmulTypes::i8_to_i32(),
            ),
            constraints,
        ) {
            Ok(selection) => selection,
            Err(error) => panic!("unexpected schedule error: {error:?}"),
        };

        assert_eq!(selection.kernel(), MatmulKernelKind::DirectThreadPerOutput);
    }

    #[test]
    fn rejects_unsupported_types_during_schedule_selection() {
        let error = select_vortex_matmul_schedule(
            MatmulOp::new(
                MatmulTensors::new(0, 1, 2),
                MatmulShape::new(8, 8, 8),
                MatmulTypes::f32_to_f32(),
            ),
            TargetConstraints::vortex_simx_default(),
        );

        assert_eq!(error, Err(ScheduleError::UnsupportedElementType));
        assert_eq!(ElementType::F32.byte_size(), 4);
    }
}
