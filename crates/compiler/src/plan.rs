use mandrel_kernel_ir::{KernelLaunch, KernelSpec, KernelSymbol};
use mandrel_profiler::KernelMetrics;
use mandrel_schedule::ScheduleError;
use snafu::Snafu;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Snafu)]
pub enum CompileError {
    #[snafu(display(
        "unsupported compile element type; Vortex attention prefill currently supports i8"
    ))]
    UnsupportedElementType,
    #[snafu(display("unsupported compile shape; sequence and head_dim must be nonzero"))]
    UnsupportedShape,
    #[snafu(display("kernel metric estimation overflowed"))]
    MetricsOverflow,
    #[snafu(display("kernel launch shape arithmetic overflowed"))]
    LaunchShapeOverflow,
    #[snafu(display("schedule selection failed: {source}"))]
    ScheduleSelectionFailed { source: ScheduleError },
    #[snafu(display("kernel '{}' is not in the catalog", symbol.as_str()))]
    KernelNotInCatalog { symbol: KernelSymbol },
    #[snafu(display("kernel '{}' is not available", symbol.as_str()))]
    KernelUnavailable { symbol: KernelSymbol },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VortexKernelPlan<Op, Schedule, Metadata, const ARGS: usize> {
    pub op: Op,
    pub schedule: Schedule,
    pub metadata: Metadata,
    pub kernel: KernelSpec,
    pub launch: KernelLaunch<ARGS>,
    pub metrics: KernelMetrics,
}

pub(crate) fn usize_to_u32(value: usize) -> Result<u32, CompileError> {
    match u32::try_from(value) {
        Ok(value) => Ok(value),
        Err(_) => Err(CompileError::LaunchShapeOverflow),
    }
}
