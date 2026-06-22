use std::fmt;
use std::path::{Path, PathBuf};

use mandrel_compiler::{CompileError, VortexMatmulPlan, compile_vortex_matmul_kernel};
use mandrel_kernel_ir::{Dim3, KernelSymbol};
use mandrel_model_ir::{MatmulOp, MatmulShape as ModelMatmulShape, MatmulTensors, MatmulTypes};
use mandrel_profiler::RuntimeTraceSummary;

use crate::artifact::VortexArtifactRegistry;
use crate::executor::{KernelCacheKey, VortexExecutor, VortexLaunchDims, VortexLaunchTrace};
use crate::matmul::{
    MatmulI8I32Args, MatmulLaunchReport, MatmulRunError, MatmulRunOutput, MatmulRunTrace,
    MatmulShape, checked_byte_len, validate_shape_and_inputs,
};
use crate::vortex2::{
    Buffer, Device, Event, Queue, Runtime, VX_MEM_READ, VX_MEM_WRITE, VortexDeviceCaps, VortexError,
};

pub type Result<T> = std::result::Result<T, VortexBackendError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VortexBackendConfig {
    pub device_index: u32,
    /// Backward-compatible path for the first stable matmul artifact.
    ///
    /// New code should prefer `kernel_artifact` / `with_kernel_artifact` so planned kernels can be
    /// supplied independently from the default direct matmul kernel.
    pub matmul_i8_i32_vxbin: PathBuf,
    pub artifacts: VortexArtifactRegistry,
}

impl VortexBackendConfig {
    pub fn new(matmul_i8_i32_vxbin: impl Into<PathBuf>) -> Self {
        let matmul_i8_i32_vxbin = matmul_i8_i32_vxbin.into();
        Self {
            device_index: 0,
            artifacts: VortexArtifactRegistry::new()
                .with_kernel_artifact(KernelSymbol::MatmulI8I32, matmul_i8_i32_vxbin.clone()),
            matmul_i8_i32_vxbin,
        }
    }

    pub fn from_kernel_artifact(symbol: KernelSymbol, vxbin: impl Into<PathBuf>) -> Self {
        let vxbin = vxbin.into();
        let matmul_i8_i32_vxbin = if symbol == KernelSymbol::MatmulI8I32 {
            vxbin.clone()
        } else {
            PathBuf::new()
        };
        Self {
            device_index: 0,
            artifacts: VortexArtifactRegistry::new().with_kernel_artifact(symbol, vxbin),
            matmul_i8_i32_vxbin,
        }
    }

    pub const fn with_device_index(mut self, device_index: u32) -> Self {
        self.device_index = device_index;
        self
    }

    pub fn with_kernel_artifact(mut self, symbol: KernelSymbol, vxbin: impl Into<PathBuf>) -> Self {
        let vxbin = vxbin.into();
        if symbol == KernelSymbol::MatmulI8I32 {
            self.matmul_i8_i32_vxbin = vxbin.clone();
        }
        self.artifacts.insert_kernel_artifact(symbol, vxbin);
        self
    }

    pub fn kernel_artifact(&self, symbol: KernelSymbol) -> Option<&Path> {
        if symbol == KernelSymbol::MatmulI8I32 && !self.matmul_i8_i32_vxbin.as_os_str().is_empty() {
            return Some(self.matmul_i8_i32_vxbin.as_path());
        }
        self.artifacts.kernel_artifact(symbol)
    }
}

#[derive(Debug)]
pub enum VortexBackendError {
    Matmul(MatmulRunError),
    Compile(CompileError),
    Vortex(VortexError),
    MissingKernelArtifact(KernelSymbol),
    InvalidLaunch(String),
    Internal(&'static str),
}

impl fmt::Display for VortexBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matmul(error) => write!(f, "{error}"),
            Self::Compile(error) => write!(f, "failed to compile Vortex matmul plan: {error:?}"),
            Self::Vortex(error) => write!(f, "{error}"),
            Self::MissingKernelArtifact(symbol) => write!(
                f,
                "no Vortex vxbin artifact configured for kernel {}",
                symbol.as_str()
            ),
            Self::InvalidLaunch(message) => write!(f, "invalid Vortex launch: {message}"),
            Self::Internal(message) => write!(f, "internal Vortex backend error: {message}"),
        }
    }
}

impl std::error::Error for VortexBackendError {}

impl From<MatmulRunError> for VortexBackendError {
    fn from(error: MatmulRunError) -> Self {
        Self::Matmul(error)
    }
}

impl From<CompileError> for VortexBackendError {
    fn from(error: CompileError) -> Self {
        Self::Compile(error)
    }
}

impl From<VortexError> for VortexBackendError {
    fn from(error: VortexError) -> Self {
        Self::Vortex(error)
    }
}

pub struct VortexBackend {
    executor: VortexExecutor,
    queue: Queue,
    device: Device,
    runtime: Runtime,
    config: VortexBackendConfig,
    last_launch_trace: Option<VortexLaunchTrace>,
    last_matmul_launch: Option<MatmulLaunchReport>,
    last_matmul_trace: Option<MatmulRunTrace>,
    last_matmul_runtime_trace: Option<RuntimeTraceSummary>,
}

impl VortexBackend {
    pub fn new(config: VortexBackendConfig) -> Result<Self> {
        let runtime = Runtime::load()?;
        Self::from_runtime(runtime, config)
    }

    pub fn load_from_runtime_candidates<I, P>(
        config: VortexBackendConfig,
        candidates: I,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        let runtime = Runtime::load_from_candidates(candidates)?;
        Self::from_runtime(runtime, config)
    }

    pub fn from_runtime(runtime: Runtime, config: VortexBackendConfig) -> Result<Self> {
        let device = runtime.open_device(config.device_index)?;
        let queue = device.create_queue()?;
        Ok(Self {
            executor: VortexExecutor::new(),
            queue,
            device,
            runtime,
            config,
            last_launch_trace: None,
            last_matmul_launch: None,
            last_matmul_trace: None,
            last_matmul_runtime_trace: None,
        })
    }

    pub const fn config(&self) -> &VortexBackendConfig {
        &self.config
    }

    pub const fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    pub const fn last_launch_trace(&self) -> Option<VortexLaunchTrace> {
        self.last_launch_trace
    }

    pub const fn last_matmul_launch(&self) -> Option<MatmulLaunchReport> {
        self.last_matmul_launch
    }

    pub const fn last_matmul_trace(&self) -> Option<MatmulRunTrace> {
        self.last_matmul_trace
    }

    pub const fn last_matmul_runtime_trace(&self) -> Option<RuntimeTraceSummary> {
        self.last_matmul_runtime_trace
    }

    pub fn cached_module_count(&self) -> usize {
        self.executor.cached_module_count()
    }

    pub fn cached_kernel_count(&self) -> usize {
        self.executor.cached_kernel_count()
    }

    pub fn clear_kernel_cache(&mut self) {
        self.executor.clear_kernel_cache();
    }

    pub fn mul_mat_i8_i8_i32(
        &mut self,
        shape: MatmulShape,
        lhs: &[i8],
        rhs: &[i8],
    ) -> Result<Vec<i32>> {
        self.mul_mat_i8_i8_i32_with_report(shape, lhs, rhs)
            .map(|output| output.values)
    }

    pub fn mul_mat_i8_i8_i32_with_report(
        &mut self,
        shape: MatmulShape,
        lhs: &[i8],
        rhs: &[i8],
    ) -> Result<MatmulRunOutput> {
        let plan = compile_matmul_plan_for_shape(shape)?;
        self.mul_mat_i8_i8_i32_with_plan(&plan, shape, lhs, rhs)
    }

    pub fn mul_mat_i8_i8_i32_with_plan(
        &mut self,
        plan: &VortexMatmulPlan,
        shape: MatmulShape,
        lhs: &[i8],
        rhs: &[i8],
    ) -> Result<MatmulRunOutput> {
        validate_shape_and_inputs(shape, lhs, rhs)?;
        validate_matmul_plan_matches_shape(plan, shape)?;
        let launch_dims = launch_dims_from_matmul_plan(plan)?;

        let lhs_bytes = checked_byte_len::<i8>(shape.lhs_len()?, "lhs")?;
        let rhs_bytes = checked_byte_len::<i8>(shape.rhs_len()?, "rhs")?;
        let out_len = shape.out_len()?;
        let out_bytes = checked_byte_len::<i32>(out_len, "out")?;
        let buffers = self.create_matmul_buffers(lhs_bytes, rhs_bytes, out_bytes)?;
        let args = MatmulI8I32Args {
            lhs_addr: buffers.lhs.address()?,
            rhs_addr: buffers.rhs.address()?,
            out_addr: buffers.out.address()?,
            m: shape.m,
            n: shape.n,
            k: shape.k,
            lhs_stride: shape.k,
            rhs_stride: shape.n,
            out_stride: shape.n,
        };

        self.queue.enqueue_write(&buffers.lhs, 0, lhs)?;
        self.queue.enqueue_write(&buffers.rhs, 0, rhs)?;

        let kernel_symbol = plan.launch.symbol.as_str();
        let kernel_path = self
            .config
            .kernel_artifact(plan.launch.symbol)
            .ok_or(VortexBackendError::MissingKernelArtifact(
                plan.launch.symbol,
            ))?
            .to_path_buf();
        let kernel_lookup =
            self.executor
                .ensure_kernel(&self.device, &kernel_path, kernel_symbol)?;

        let launch_event = self.launch_cached_kernel(&kernel_lookup.key, &args, 2, launch_dims)?;
        let mut values = vec![0_i32; out_len];
        let read_event =
            self.queue
                .enqueue_read_after(&mut values, &buffers.out, 0, &launch_event)?;
        read_event.wait()?;

        self.device.dump_perf()?;

        let host_to_device_bytes =
            lhs_bytes
                .checked_add(rhs_bytes)
                .ok_or(VortexBackendError::Internal(
                    "matmul host-to-device byte count overflow",
                ))?;
        let launch_trace = VortexLaunchTrace::new(
            kernel_symbol,
            launch_dims,
            host_to_device_bytes,
            out_bytes,
            kernel_lookup.module_cache_hit,
            kernel_lookup.kernel_cache_hit,
        );
        let launch = MatmulLaunchReport::from(launch_dims);
        let trace = MatmulRunTrace {
            kernel_symbol,
            lhs_bytes,
            rhs_bytes,
            out_bytes,
            module_cache_hit: kernel_lookup.module_cache_hit,
            kernel_cache_hit: kernel_lookup.kernel_cache_hit,
        };
        let runtime_trace =
            trace
                .to_runtime_trace_summary(launch)
                .ok_or(VortexBackendError::Internal(
                    "matmul trace byte count does not fit usize",
                ))?;
        self.last_launch_trace = Some(launch_trace);
        self.last_matmul_launch = Some(launch);
        self.last_matmul_trace = Some(trace);
        self.last_matmul_runtime_trace = Some(runtime_trace);
        Ok(MatmulRunOutput {
            values,
            launch,
            trace,
        })
    }

    fn create_matmul_buffers(
        &self,
        lhs_bytes: u64,
        rhs_bytes: u64,
        out_bytes: u64,
    ) -> Result<MatmulBuffers> {
        Ok(MatmulBuffers {
            lhs: self.device.create_buffer(lhs_bytes, VX_MEM_READ)?,
            rhs: self.device.create_buffer(rhs_bytes, VX_MEM_READ)?,
            out: self.device.create_buffer(out_bytes, VX_MEM_WRITE)?,
        })
    }

    fn launch_cached_kernel<A>(
        &self,
        key: &KernelCacheKey,
        args: &A,
        ndim: u32,
        dims: VortexLaunchDims,
    ) -> Result<Event> {
        let kernel = self.executor.kernel(key)?;
        let caps = self.device.capabilities()?;
        validate_launch_dims_against_caps(dims, caps)?;
        self.queue
            .enqueue_launch(
                kernel,
                args,
                ndim,
                dims.grid,
                dims.block,
                dims.shared_memory_bytes,
            )
            .map_err(VortexBackendError::from)
    }
}

fn compile_matmul_plan_for_shape(shape: MatmulShape) -> Result<VortexMatmulPlan> {
    let op = MatmulOp::new(
        MatmulTensors::new(0, 1, 2),
        ModelMatmulShape::new(shape.m as usize, shape.n as usize, shape.k as usize),
        MatmulTypes::i8_to_i32(),
    );
    compile_vortex_matmul_kernel(op).map_err(VortexBackendError::from)
}

fn validate_matmul_plan_matches_shape(plan: &VortexMatmulPlan, shape: MatmulShape) -> Result<()> {
    if plan.op.shape.m != shape.m as usize
        || plan.op.shape.n != shape.n as usize
        || plan.op.shape.k != shape.k as usize
    {
        return Err(VortexBackendError::Internal(
            "matmul plan shape does not match requested input shape",
        ));
    }
    Ok(())
}

fn launch_dims_from_matmul_plan(plan: &VortexMatmulPlan) -> Result<VortexLaunchDims> {
    let grid = dim3_to_array(plan.launch.grid);
    let block = dim3_to_array(plan.launch.block);
    if grid.contains(&0) || block.contains(&0) {
        return Err(VortexError::InvalidValue("matmul plan has a zero launch dimension").into());
    }
    Ok(VortexLaunchDims::new(
        grid,
        block,
        plan.launch.shared_memory_bytes,
    ))
}

fn validate_launch_dims_against_caps(dims: VortexLaunchDims, caps: VortexDeviceCaps) -> Result<()> {
    if dims.grid.contains(&0) || dims.block.contains(&0) {
        return Err(VortexBackendError::InvalidLaunch(
            "grid and block dimensions must be nonzero".to_owned(),
        ));
    }
    if caps.threads_per_warp == 0 || caps.warps_per_core == 0 {
        return Err(VortexBackendError::InvalidLaunch(format!(
            "device reported invalid execution resources: threads_per_warp={}, warps_per_core={}",
            caps.threads_per_warp, caps.warps_per_core
        )));
    }

    let block_threads = product_u32_as_u64(dims.block).ok_or_else(|| {
        VortexBackendError::InvalidLaunch("block thread count overflowed u64".to_owned())
    })?;
    let max_threads = caps.max_workgroup_threads().ok_or_else(|| {
        VortexBackendError::InvalidLaunch(
            "device max workgroup thread count overflowed u64".to_owned(),
        )
    })?;
    if block_threads > max_threads {
        return Err(VortexBackendError::InvalidLaunch(format!(
            "block has {block_threads} threads, but this Vortex device supports at most {max_threads} threads per workgroup ({} warps × {} threads)",
            caps.warps_per_core, caps.threads_per_warp
        )));
    }

    if dims.shared_memory_bytes != 0 {
        let warps_per_block = block_threads.div_ceil(caps.threads_per_warp);
        if warps_per_block == 0 || warps_per_block > caps.warps_per_core {
            return Err(VortexBackendError::InvalidLaunch(format!(
                "block needs {warps_per_block} warps, but this Vortex device has {} warps per core",
                caps.warps_per_core
            )));
        }
        let blocks_per_core = caps.warps_per_core / warps_per_block;
        if blocks_per_core == 0 {
            return Err(VortexBackendError::InvalidLaunch(
                "local-memory budget is undefined because no workgroup fits on a core".to_owned(),
            ));
        }
        let local_budget = caps.local_memory_bytes / blocks_per_core;
        let requested = u64::from(dims.shared_memory_bytes);
        if requested > local_budget {
            return Err(VortexBackendError::InvalidLaunch(format!(
                "shared/local memory request is {requested} bytes per workgroup, but the per-workgroup budget is {local_budget} bytes ({} bytes/core across {blocks_per_core} resident workgroups)",
                caps.local_memory_bytes
            )));
        }
    }

    Ok(())
}

fn product_u32_as_u64(values: [u32; 3]) -> Option<u64> {
    u64::from(values[0])
        .checked_mul(u64::from(values[1]))?
        .checked_mul(u64::from(values[2]))
}

const fn dim3_to_array(dim: Dim3) -> [u32; 3] {
    [dim.x, dim.y, dim.z]
}

struct MatmulBuffers {
    lhs: Buffer,
    rhs: Buffer,
    out: Buffer,
}

#[cfg(test)]
mod tests {
    use super::{
        VortexBackendConfig, VortexBackendError, compile_matmul_plan_for_shape,
        launch_dims_from_matmul_plan, validate_launch_dims_against_caps,
    };
    use crate::executor::VortexLaunchDims;
    use crate::matmul::MatmulShape;
    use crate::vortex2::VortexDeviceCaps;
    use mandrel_kernel_ir::KernelSymbol;
    use std::path::{Path, PathBuf};

    #[test]
    fn backend_config_defaults_to_device_zero() {
        let config = VortexBackendConfig::new("target/vortex/matmul_i8_i32/kernel.vxbin");

        assert_eq!(config.device_index, 0);
        assert_eq!(
            config.matmul_i8_i32_vxbin,
            PathBuf::from("target/vortex/matmul_i8_i32/kernel.vxbin")
        );
        assert_eq!(
            config.kernel_artifact(KernelSymbol::MatmulI8I32),
            Some(Path::new("target/vortex/matmul_i8_i32/kernel.vxbin"))
        );
    }

    #[test]
    fn backend_config_registers_additional_kernel_artifacts() {
        let config = VortexBackendConfig::new("target/vortex/matmul_i8_i32/kernel.vxbin")
            .with_kernel_artifact(
                KernelSymbol::MatmulI8I32Tiled,
                "target/vortex/matmul_i8_i32_tiled/kernel.vxbin",
            );

        assert_eq!(
            config.kernel_artifact(KernelSymbol::MatmulI8I32),
            Some(Path::new("target/vortex/matmul_i8_i32/kernel.vxbin"))
        );
        assert_eq!(
            config.kernel_artifact(KernelSymbol::MatmulI8I32Tiled),
            Some(Path::new("target/vortex/matmul_i8_i32_tiled/kernel.vxbin"))
        );
    }

    #[test]
    fn backend_config_allows_device_override() {
        let config = VortexBackendConfig::new("kernel.vxbin").with_device_index(2);

        assert_eq!(config.device_index, 2);
    }

    #[test]
    fn matmul_launch_dims_are_derived_from_compiler_plan() {
        let plan = match compile_matmul_plan_for_shape(MatmulShape::new(33, 35, 64)) {
            Ok(plan) => plan,
            Err(error) => panic!("unexpected compile error: {error}"),
        };
        let dims = match launch_dims_from_matmul_plan(&plan) {
            Ok(dims) => dims,
            Err(error) => panic!("unexpected launch conversion error: {error}"),
        };

        assert_eq!(plan.launch.symbol, KernelSymbol::MatmulI8I32);
        assert_eq!(dims.grid, [9, 9, 1]);
        assert_eq!(dims.block, [4, 4, 1]);
        assert_eq!(dims.shared_memory_bytes, plan.launch.shared_memory_bytes);
    }

    #[test]
    fn launch_validation_accepts_current_simx_tiled_shape() {
        let caps = simx_caps();
        let dims = VortexLaunchDims::new([8, 8, 1], [4, 4, 1], 256);

        assert!(validate_launch_dims_against_caps(dims, caps).is_ok());
    }

    #[test]
    fn launch_validation_rejects_oversized_workgroup() {
        let caps = simx_caps();
        let dims = VortexLaunchDims::new([2, 2, 1], [16, 16, 1], 1024);
        let error = match validate_launch_dims_against_caps(dims, caps) {
            Ok(()) => panic!("expected oversized workgroup to be rejected"),
            Err(error) => error,
        };

        match error {
            VortexBackendError::InvalidLaunch(message) => {
                assert!(message.contains("block has 256 threads"));
                assert!(message.contains("at most 16 threads"));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    fn simx_caps() -> VortexDeviceCaps {
        VortexDeviceCaps {
            threads_per_warp: 4,
            warps_per_core: 4,
            local_memory_bytes: 32 * 1024,
        }
    }
}
