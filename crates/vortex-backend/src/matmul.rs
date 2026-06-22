use std::fmt;
use std::mem::size_of;
use std::path::Path;

use mandrel_profiler::{KernelCounterTrace, KernelLaunchTrace, RuntimeTraceSummary};

use crate::backend::{VortexBackend, VortexBackendConfig, VortexBackendError};
use crate::executor::VortexLaunchDims;
use crate::vortex2::{Runtime, VortexError};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MatmulI8I32Args {
    pub lhs_addr: u64,
    pub rhs_addr: u64,
    pub out_addr: u64,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub lhs_stride: u32,
    pub rhs_stride: u32,
    pub out_stride: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl MatmulShape {
    pub const fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }

    pub(crate) fn lhs_len(self) -> Result<usize, MatmulRunError> {
        checked_matrix_len(self.m, self.k, "lhs")
    }

    pub(crate) fn rhs_len(self) -> Result<usize, MatmulRunError> {
        checked_matrix_len(self.k, self.n, "rhs")
    }

    pub(crate) fn out_len(self) -> Result<usize, MatmulRunError> {
        checked_matrix_len(self.m, self.n, "out")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulLaunchReport {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_memory_bytes: u32,
}

impl MatmulLaunchReport {
    pub const fn new(grid: [u32; 3], block: [u32; 3], shared_memory_bytes: u32) -> Self {
        Self {
            grid,
            block,
            shared_memory_bytes,
        }
    }

    pub const fn to_kernel_launch_trace(self, kernel_symbol: &'static str) -> KernelLaunchTrace {
        KernelLaunchTrace::new(
            kernel_symbol,
            self.grid,
            self.block,
            self.shared_memory_bytes,
        )
    }
}

impl From<VortexLaunchDims> for MatmulLaunchReport {
    fn from(dims: VortexLaunchDims) -> Self {
        Self::new(dims.grid, dims.block, dims.shared_memory_bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulRunTrace {
    pub kernel_symbol: &'static str,
    pub lhs_bytes: u64,
    pub rhs_bytes: u64,
    pub out_bytes: u64,
    pub module_cache_hit: bool,
    pub kernel_cache_hit: bool,
}

impl MatmulRunTrace {
    pub fn to_runtime_trace_summary(
        self,
        launch: MatmulLaunchReport,
    ) -> Option<RuntimeTraceSummary> {
        let host_to_device_bytes =
            usize::try_from(self.lhs_bytes.checked_add(self.rhs_bytes)?).ok()?;
        let device_to_host_bytes = usize::try_from(self.out_bytes).ok()?;
        Some(RuntimeTraceSummary::new(
            launch.to_kernel_launch_trace(self.kernel_symbol),
            host_to_device_bytes,
            device_to_host_bytes,
            KernelCounterTrace::empty(),
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulRunOutput {
    pub values: Vec<i32>,
    pub launch: MatmulLaunchReport,
    pub trace: MatmulRunTrace,
}

impl MatmulRunOutput {
    pub fn runtime_trace_summary(&self) -> Option<RuntimeTraceSummary> {
        self.trace.to_runtime_trace_summary(self.launch)
    }
}

#[derive(Debug)]
pub enum MatmulRunError {
    EmptyShape,
    ShapeOverflow {
        matrix: &'static str,
    },
    LengthMismatch {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
    Vortex(VortexError),
}

impl fmt::Display for MatmulRunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyShape => write!(f, "matmul shape dimensions must be non-zero"),
            Self::ShapeOverflow { matrix } => write!(f, "{matrix} matrix element count overflow"),
            Self::LengthMismatch {
                buffer,
                expected,
                actual,
            } => write!(
                f,
                "{buffer} buffer length mismatch: expected {expected} elements, got {actual}"
            ),
            Self::Vortex(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for MatmulRunError {}

impl From<VortexError> for MatmulRunError {
    fn from(error: VortexError) -> Self {
        Self::Vortex(error)
    }
}

pub fn run_matmul_i8_i32(
    runtime: &Runtime,
    kernel_vxbin: &Path,
    shape: MatmulShape,
    lhs: &[i8],
    rhs: &[i8],
) -> Result<MatmulRunOutput, MatmulRunError> {
    validate_shape_and_inputs(shape, lhs, rhs)?;

    let config = VortexBackendConfig::new(kernel_vxbin.to_path_buf());
    let mut backend =
        VortexBackend::from_runtime(runtime.clone(), config).map_err(matmul_error_from_backend)?;
    backend
        .mul_mat_i8_i8_i32_with_report(shape, lhs, rhs)
        .map_err(matmul_error_from_backend)
}

fn matmul_error_from_backend(error: VortexBackendError) -> MatmulRunError {
    match error {
        VortexBackendError::Matmul(error) => error,
        VortexBackendError::Compile(_) => MatmulRunError::Vortex(VortexError::InvalidValue(
            "failed to compile Vortex matmul plan",
        )),
        VortexBackendError::Vortex(error) => MatmulRunError::Vortex(error),
        VortexBackendError::MissingKernelArtifact(_) => {
            MatmulRunError::Vortex(VortexError::InvalidValue("missing Vortex kernel artifact"))
        }
        VortexBackendError::InvalidLaunch(_) => {
            MatmulRunError::Vortex(VortexError::InvalidValue("invalid Vortex launch"))
        }
        VortexBackendError::Internal(message) => {
            MatmulRunError::Vortex(VortexError::InvalidValue(message))
        }
    }
}

pub(crate) fn validate_shape_and_inputs(
    shape: MatmulShape,
    lhs: &[i8],
    rhs: &[i8],
) -> Result<(), MatmulRunError> {
    if shape.m == 0 || shape.n == 0 || shape.k == 0 {
        return Err(MatmulRunError::EmptyShape);
    }
    let lhs_expected = shape.lhs_len()?;
    if lhs.len() != lhs_expected {
        return Err(MatmulRunError::LengthMismatch {
            buffer: "lhs",
            expected: lhs_expected,
            actual: lhs.len(),
        });
    }
    let rhs_expected = shape.rhs_len()?;
    if rhs.len() != rhs_expected {
        return Err(MatmulRunError::LengthMismatch {
            buffer: "rhs",
            expected: rhs_expected,
            actual: rhs.len(),
        });
    }
    Ok(())
}

fn checked_matrix_len(rows: u32, cols: u32, matrix: &'static str) -> Result<usize, MatmulRunError> {
    let len = u64::from(rows)
        .checked_mul(u64::from(cols))
        .ok_or(MatmulRunError::ShapeOverflow { matrix })?;
    usize::try_from(len).map_err(|_| MatmulRunError::ShapeOverflow { matrix })
}

pub(crate) fn checked_byte_len<T>(
    elements: usize,
    matrix: &'static str,
) -> Result<u64, MatmulRunError> {
    let bytes = elements
        .checked_mul(size_of::<T>())
        .ok_or(MatmulRunError::ShapeOverflow { matrix })?;
    u64::try_from(bytes).map_err(|_| MatmulRunError::ShapeOverflow { matrix })
}

#[cfg(test)]
mod tests {
    use super::{MatmulI8I32Args, MatmulLaunchReport, MatmulRunTrace, MatmulShape};
    use std::mem::size_of;

    #[test]
    fn kernel_args_match_c_abi_size() {
        assert_eq!(size_of::<MatmulI8I32Args>(), 48);
    }

    #[test]
    fn computes_matrix_lengths() {
        let shape = MatmulShape::new(32, 16, 64);
        assert_eq!(shape.lhs_len().expect("lhs len"), 2048);
        assert_eq!(shape.rhs_len().expect("rhs len"), 1024);
        assert_eq!(shape.out_len().expect("out len"), 512);
    }

    #[test]
    fn converts_matmul_trace_to_runtime_summary() {
        let launch = MatmulLaunchReport::new([8, 8, 1], [4, 4, 1], 0);
        let trace = MatmulRunTrace {
            kernel_symbol: "matmul_i8_i32",
            lhs_bytes: 2048,
            rhs_bytes: 2048,
            out_bytes: 4096,
            module_cache_hit: false,
            kernel_cache_hit: false,
        };
        let summary = match trace.to_runtime_trace_summary(launch) {
            Some(summary) => summary,
            None => panic!("trace bytes should fit usize"),
        };

        assert_eq!(summary.launch.kernel_symbol, "matmul_i8_i32");
        assert_eq!(summary.launch.workgroup_count(), 64);
        assert_eq!(summary.launch.threads_per_workgroup(), 16);
        assert_eq!(summary.host_to_device_bytes, 4096);
        assert_eq!(summary.device_to_host_bytes, 4096);
        assert_eq!(summary.total_transfer_bytes(), 8192);
    }
}
