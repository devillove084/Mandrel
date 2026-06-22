use mandrel_core::ElementType;
use mandrel_kernel_ir::{
    Dim3, KernelArg, KernelLaunch, KernelSpec, KernelSymbol, available_kernel, kernel_spec,
};
use mandrel_model_ir::{MatmulOp, TargetConstraints};
use mandrel_profiler::{KernelMetrics, MatmulEstimateInput, estimate_gpu_matmul};
use mandrel_schedule::{
    MatmulKernelKind, MatmulSchedule, MatmulScheduleSelection, ScheduleError,
    select_experimental_vortex_matmul_schedule, select_vortex_matmul_schedule,
};

pub const VORTEX_MATMUL_ARG_COUNT: usize = 6;
pub const VORTEX_MATMUL_LHS_BUFFER: u32 = 0;
pub const VORTEX_MATMUL_RHS_BUFFER: u32 = 1;
pub const VORTEX_MATMUL_OUT_BUFFER: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileError {
    UnsupportedElementType,
    UnsupportedShape,
    MetricsOverflow,
    LaunchShapeOverflow,
    ScheduleSelectionFailed(ScheduleError),
    KernelNotInCatalog(KernelSymbol),
    KernelUnavailable(KernelSymbol),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VortexKernelPlan<const ARGS: usize> {
    pub op: MatmulOp,
    pub schedule: MatmulSchedule,
    pub kernel: KernelSpec,
    pub launch: KernelLaunch<ARGS>,
    pub metrics: KernelMetrics,
}

pub type VortexMatmulPlan = VortexKernelPlan<VORTEX_MATMUL_ARG_COUNT>;

pub fn compile_vortex_matmul_kernel(op: MatmulOp) -> Result<VortexMatmulPlan, CompileError> {
    validate_matmul_op(op)?;
    let selection = select_vortex_matmul_schedule(op, TargetConstraints::vortex_simx_default())
        .map_err(CompileError::ScheduleSelectionFailed)?;
    compile_selected_vortex_matmul_kernel(op, selection, true)
}

pub fn compile_experimental_vortex_matmul_kernel(
    op: MatmulOp,
) -> Result<VortexMatmulPlan, CompileError> {
    validate_matmul_op(op)?;
    let selection =
        select_experimental_vortex_matmul_schedule(op, TargetConstraints::vortex_simx_default())
            .map_err(CompileError::ScheduleSelectionFailed)?;
    compile_selected_vortex_matmul_kernel(op, selection, false)
}

fn validate_matmul_op(op: MatmulOp) -> Result<(), CompileError> {
    if op.types.lhs != ElementType::I8
        || op.types.rhs != ElementType::I8
        || op.types.out != ElementType::I32
    {
        return Err(CompileError::UnsupportedElementType);
    }

    if op.shape.m == 0 || op.shape.n == 0 || op.shape.k == 0 {
        return Err(CompileError::UnsupportedShape);
    }

    Ok(())
}

fn compile_selected_vortex_matmul_kernel(
    op: MatmulOp,
    selection: MatmulScheduleSelection,
    require_available: bool,
) -> Result<VortexMatmulPlan, CompileError> {
    let schedule = selection.schedule();
    let symbol = matmul_kernel_symbol(selection.kernel());
    let kernel = match kernel_spec(symbol) {
        Some(spec) => spec,
        None => return Err(CompileError::KernelNotInCatalog(symbol)),
    };
    if require_available && available_kernel(symbol).is_none() {
        return Err(CompileError::KernelUnavailable(symbol));
    }

    let metrics = match estimate_gpu_matmul(MatmulEstimateInput::new(
        op.shape,
        schedule,
        op.types.lhs,
        op.types.rhs,
        op.types.out,
    )) {
        Some(metrics) => metrics,
        None => return Err(CompileError::MetricsOverflow),
    };

    let grid_x = usize_to_u32(op.shape.n.div_ceil(schedule.tile.n))?;
    let grid_y = usize_to_u32(op.shape.m.div_ceil(schedule.tile.m))?;
    let block_x = usize_to_u32(schedule.block.x)?;
    let block_y = usize_to_u32(schedule.block.y)?;
    let block_z = usize_to_u32(schedule.block.z)?;
    let shared_memory_bytes = usize_to_u32(metrics.local_memory_bytes_per_workgroup)?;
    let m = usize_to_u32(op.shape.m)?;
    let n = usize_to_u32(op.shape.n)?;
    let k = usize_to_u32(op.shape.k)?;

    Ok(VortexKernelPlan {
        op,
        schedule,
        kernel,
        launch: KernelLaunch::new(
            symbol,
            Dim3::new(grid_x, grid_y, 1),
            Dim3::new(block_x, block_y, block_z),
            shared_memory_bytes,
            [
                KernelArg::buffer(0, VORTEX_MATMUL_LHS_BUFFER),
                KernelArg::buffer(1, VORTEX_MATMUL_RHS_BUFFER),
                KernelArg::buffer(2, VORTEX_MATMUL_OUT_BUFFER),
                KernelArg::u32(3, m),
                KernelArg::u32(4, n),
                KernelArg::u32(5, k),
            ],
        ),
        metrics,
    })
}

pub const fn matmul_kernel_symbol(kind: MatmulKernelKind) -> KernelSymbol {
    match kind {
        MatmulKernelKind::DirectThreadPerOutput => KernelSymbol::MatmulI8I32,
        MatmulKernelKind::TiledLocalMemory => KernelSymbol::MatmulI8I32Tiled,
    }
}

fn usize_to_u32(value: usize) -> Result<u32, CompileError> {
    match u32::try_from(value) {
        Ok(value) => Ok(value),
        Err(_) => Err(CompileError::LaunchShapeOverflow),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CompileError, compile_experimental_vortex_matmul_kernel, compile_vortex_matmul_kernel,
        matmul_kernel_symbol,
    };
    use mandrel_core::ElementType;
    use mandrel_kernel_ir::{KernelArgValue, KernelAvailability, KernelSymbol};
    use mandrel_model_ir::{MatmulOp, MatmulShape, MatmulTensors, MatmulTypes};
    use mandrel_schedule::MatmulKernelKind;

    #[test]
    fn compiles_demo_to_available_vortex_kernel_launch() {
        let plan = match compile_vortex_matmul_kernel(MatmulOp::ggml_mul_mat_i8_demo()) {
            Ok(plan) => plan,
            Err(error) => panic!("unexpected compile error: {error:?}"),
        };

        assert_eq!(plan.kernel.availability, KernelAvailability::Available);
        assert_eq!(plan.launch.symbol, KernelSymbol::MatmulI8I32);
        assert_eq!(plan.launch.grid.x, 8);
        assert_eq!(plan.launch.grid.y, 8);
        assert_eq!(plan.launch.block.x, 4);
        assert_eq!(plan.launch.block.y, 4);
        assert_eq!(plan.launch.shared_memory_bytes, 0);
        assert_eq!(plan.launch.args[3].value, KernelArgValue::U32(32));
        assert_eq!(plan.metrics.workgroup_count, 64);
    }

    #[test]
    fn compiles_experimental_demo_to_planned_tiled_kernel_launch() {
        let plan = match compile_experimental_vortex_matmul_kernel(MatmulOp::ggml_mul_mat_i8_demo())
        {
            Ok(plan) => plan,
            Err(error) => panic!("unexpected compile error: {error:?}"),
        };

        assert_eq!(plan.kernel.availability, KernelAvailability::Planned);
        assert_eq!(plan.launch.symbol, KernelSymbol::MatmulI8I32Tiled);
        assert_eq!(plan.launch.grid.x, 8);
        assert_eq!(plan.launch.grid.y, 8);
        assert_eq!(plan.launch.block.x, 4);
        assert_eq!(plan.launch.block.y, 4);
        assert_eq!(plan.launch.shared_memory_bytes, 256);
    }

    #[test]
    fn maps_schedule_kind_to_kernel_symbol() {
        assert_eq!(
            matmul_kernel_symbol(MatmulKernelKind::DirectThreadPerOutput),
            KernelSymbol::MatmulI8I32
        );
        assert_eq!(
            matmul_kernel_symbol(MatmulKernelKind::TiledLocalMemory),
            KernelSymbol::MatmulI8I32Tiled
        );
    }

    #[test]
    fn rejects_non_i8_i32_matmul_for_first_kernel() {
        let op = MatmulOp::new(
            MatmulTensors::new(0, 1, 2),
            MatmulShape::new(16, 16, 16),
            MatmulTypes {
                lhs: ElementType::F32,
                rhs: ElementType::F32,
                out: ElementType::F32,
            },
        );

        assert_eq!(
            compile_vortex_matmul_kernel(op),
            Err(CompileError::UnsupportedElementType)
        );
    }
}
