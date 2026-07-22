use std::env;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use snafu::Snafu;

use mandrel_kernel_ir::{ATTENTION_PREFILL_I8_ARG_COUNT, Dim3, KernelLaunch, KernelSymbol};

use crate::executor::{KernelCacheKey, VortexExecutor, VortexLaunchDims, VortexLaunchTrace};
use crate::vortex2::{
    Device, Event, Queue, Runtime, VX_MEM_READ, VX_MEM_WRITE, VortexDeviceCaps, VortexError,
};

pub type Result<T> = std::result::Result<T, VortexBackendError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VortexBackendConfig {
    pub device_index: u32,
    kernel_images: Vec<(KernelSymbol, PathBuf)>,
}

impl VortexBackendConfig {
    pub const fn new() -> Self {
        Self {
            device_index: 0,
            kernel_images: Vec::new(),
        }
    }

    pub fn from_kernel_image(symbol: KernelSymbol, vxbin: impl Into<PathBuf>) -> Self {
        Self::new().with_kernel_image(symbol, vxbin)
    }

    pub const fn with_device_index(mut self, device_index: u32) -> Self {
        self.device_index = device_index;
        self
    }

    pub fn with_kernel_image(mut self, symbol: KernelSymbol, vxbin: impl Into<PathBuf>) -> Self {
        let vxbin = vxbin.into();
        if let Some(index) = self
            .kernel_images
            .iter()
            .position(|(entry_symbol, _)| entry_symbol == &symbol)
        {
            self.kernel_images[index].1 = vxbin;
        } else {
            self.kernel_images.push((symbol, vxbin));
        }
        self
    }

    pub fn kernel_image(&self, symbol: KernelSymbol) -> Option<&Path> {
        self.kernel_images
            .iter()
            .find(|(entry_symbol, _)| entry_symbol == &symbol)
            .map(|(_, vxbin)| vxbin.as_path())
    }
}

impl Default for VortexBackendConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HardwareIdentityProbeArgs {
    out_addr: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionPrefillI8Args {
    pub q_addr: u64,
    pub k_addr: u64,
    pub v_addr: u64,
    pub out_addr: u64,
    pub sequence: u32,
    pub head_dim: u32,
    pub query_tile: u32,
    pub key_tile: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionPrefillI8Run {
    pub q: Vec<i8>,
    pub k: Vec<i8>,
    pub v: Vec<i8>,
    pub sequence: u32,
    pub head_dim: u32,
    pub query_tile: u32,
    pub key_tile: u32,
}

impl AttentionPrefillI8Run {
    pub fn element_count(&self) -> Result<usize> {
        attention_element_count(self.sequence, self.head_dim)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionPrefillI8RunOutput {
    pub output: Vec<i8>,
    pub trace: VortexLaunchTrace,
}

#[derive(Debug, Snafu)]
pub enum VortexBackendError {
    #[snafu(transparent)]
    Vortex { source: VortexError },
    #[snafu(display(
        "no Vortex vxbin image configured for kernel {}",
        symbol.as_str()
    ))]
    MissingKernelImage { symbol: KernelSymbol },
    #[snafu(display("invalid Vortex launch: {message}"))]
    InvalidLaunch { message: String },
    #[snafu(display("internal Vortex backend error: {message}"))]
    Internal { message: &'static str },
}

pub struct VortexBackend {
    executor: VortexExecutor,
    queue: Queue,
    device: Device,
    runtime: Runtime,
    config: VortexBackendConfig,
    last_launch_trace: Option<VortexLaunchTrace>,
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

    pub fn device_capabilities(&self) -> Result<VortexDeviceCaps> {
        self.device.capabilities().map_err(VortexBackendError::from)
    }

    pub fn run_hardware_identity_probe(
        &mut self,
        vxbin_path: &Path,
        kernel_symbol: &'static str,
    ) -> Result<u64> {
        backend_runtime_trace("hardware identity: creating output buffer");
        let output_buffer = self.device.create_buffer(8, VX_MEM_WRITE)?;
        let args = HardwareIdentityProbeArgs {
            out_addr: output_buffer.address()?,
        };
        let dims = VortexLaunchDims::new([1, 1, 1], [1, 1, 1], 0);

        backend_runtime_trace("hardware identity: loading probe kernel");
        let kernel_lookup = self
            .executor
            .ensure_kernel(&self.device, vxbin_path, kernel_symbol)?;
        backend_runtime_trace("hardware identity: launching probe kernel");
        let launch_event = self.launch_cached_kernel(&kernel_lookup.key, &args, 1, dims)?;
        launch_event.wait()?;

        let mut output = [0u64; 1];
        backend_runtime_trace("hardware identity: reading observed RTL config tag");
        let read_event = self.queue.enqueue_read(&mut output, &output_buffer, 0)?;
        read_event.wait()?;
        Ok(output[0])
    }

    pub fn run_attention_prefill_i8(
        &mut self,
        launch: &KernelLaunch<ATTENTION_PREFILL_I8_ARG_COUNT>,
        input: &AttentionPrefillI8Run,
    ) -> Result<AttentionPrefillI8RunOutput> {
        if launch.symbol != KernelSymbol::AttentionPrefillI8 {
            return Err(VortexBackendError::InvalidLaunch {
                message: format!(
                    "attention runner requires {}, got {}",
                    KernelSymbol::AttentionPrefillI8.as_str(),
                    launch.symbol.as_str()
                ),
            });
        }

        backend_runtime_trace("attention: validating input");
        let elements = validate_attention_input(input)?;
        let buffer_bytes =
            u64::try_from(elements).map_err(|_| VortexBackendError::InvalidLaunch {
                message: "attention element count does not fit the Vortex buffer byte size"
                    .to_owned(),
            })?;
        backend_runtime_trace("attention: creating q/k/v/out buffers");
        let q_buf = self.device.create_buffer(buffer_bytes, VX_MEM_READ)?;
        let k_buf = self.device.create_buffer(buffer_bytes, VX_MEM_READ)?;
        let v_buf = self.device.create_buffer(buffer_bytes, VX_MEM_READ)?;
        let out_buf = self.device.create_buffer(buffer_bytes, VX_MEM_WRITE)?;

        backend_runtime_trace("attention: enqueue q write");
        self.queue.enqueue_write(&q_buf, 0, &input.q)?;
        backend_runtime_trace("attention: enqueue k write");
        self.queue.enqueue_write(&k_buf, 0, &input.k)?;
        backend_runtime_trace("attention: enqueue v write");
        self.queue.enqueue_write(&v_buf, 0, &input.v)?;

        backend_runtime_trace("attention: resolving buffer addresses and building args");
        let args = AttentionPrefillI8Args {
            q_addr: q_buf.address()?,
            k_addr: k_buf.address()?,
            v_addr: v_buf.address()?,
            out_addr: out_buf.address()?,
            sequence: input.sequence,
            head_dim: input.head_dim,
            query_tile: input.query_tile,
            key_tile: input.key_tile,
        };
        let host_to_device_bytes =
            buffer_bytes
                .checked_mul(3)
                .ok_or_else(|| VortexBackendError::InvalidLaunch {
                    message: "attention transfer byte count overflow".to_owned(),
                })?;
        backend_runtime_trace("attention: launching cached kernel");
        let trace =
            self.launch_kernel_with_args(launch, &args, 3, host_to_device_bytes, buffer_bytes)?;
        backend_runtime_trace("attention: launch completed");

        let mut output = vec![0i8; elements];
        backend_runtime_trace("attention: enqueue output read");
        let read_event = self.queue.enqueue_read(&mut output, &out_buf, 0)?;
        backend_runtime_trace("attention: waiting output read event");
        read_event.wait()?;
        backend_runtime_trace("attention: output read completed");

        Ok(AttentionPrefillI8RunOutput { output, trace })
    }

    pub fn launch_kernel_with_args<A, const ARGS: usize>(
        &mut self,
        launch: &KernelLaunch<ARGS>,
        args: &A,
        ndim: u32,
        host_to_device_bytes: u64,
        device_to_host_bytes: u64,
    ) -> Result<VortexLaunchTrace> {
        backend_runtime_trace("launch: deriving launch dimensions");
        let dims = launch_dims_from_kernel_launch(launch)?;
        let kernel_symbol = launch.symbol.as_str();
        let kernel_path = self
            .config
            .kernel_image(launch.symbol)
            .ok_or(VortexBackendError::MissingKernelImage {
                symbol: launch.symbol,
            })?
            .to_path_buf();
        backend_runtime_trace("launch: loading module/getting kernel");
        let kernel_lookup =
            self.executor
                .ensure_kernel(&self.device, &kernel_path, kernel_symbol)?;

        backend_runtime_trace("launch: enqueue kernel launch");
        let launch_event = self.launch_cached_kernel(&kernel_lookup.key, args, ndim, dims)?;
        backend_runtime_trace("launch: waiting launch event");
        launch_event.wait()?;
        backend_runtime_trace("launch: dumping device perf");
        self.device.dump_perf()?;
        backend_runtime_trace("launch: device perf dumped");

        let trace = VortexLaunchTrace::new(
            kernel_symbol,
            dims,
            host_to_device_bytes,
            device_to_host_bytes,
            kernel_lookup.module_cache_hit,
            kernel_lookup.kernel_cache_hit,
        );
        self.last_launch_trace = Some(trace);
        Ok(trace)
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

pub fn reference_attention_prefill_i8(input: &AttentionPrefillI8Run) -> Result<Vec<i8>> {
    let elements = validate_attention_input(input)?;
    let sequence = input.sequence as usize;
    let head_dim = input.head_dim as usize;
    let mut output = vec![0i8; elements];

    for query in 0..sequence {
        for dim in 0..head_dim {
            let mut max_score = f32::NEG_INFINITY;
            for key in 0..sequence {
                let dot = qk_dot_i32(&input.q, &input.k, query, key, head_dim);
                max_score = max_score.max(dot as f32);
            }

            let mut denom = 0.0f32;
            let mut numer = 0.0f32;
            for key in 0..sequence {
                let dot = qk_dot_i32(&input.q, &input.k, query, key, head_dim);
                let weight = ((dot as f32) - max_score).exp();
                denom += weight;
                numer += weight * f32::from(input.v[attention_index(key, dim, head_dim)]);
            }

            output[attention_index(query, dim, head_dim)] = f32_to_i8_trunc_clamped(numer / denom);
        }
    }

    Ok(output)
}

fn validate_attention_input(input: &AttentionPrefillI8Run) -> Result<usize> {
    let elements = attention_element_count(input.sequence, input.head_dim)?;
    for (name, len) in [
        ("q", input.q.len()),
        ("k", input.k.len()),
        ("v", input.v.len()),
    ] {
        if len != elements {
            return Err(VortexBackendError::InvalidLaunch {
                message: format!(
                    "attention {name} buffer has {len} elements, expected {elements} for sequence={} head_dim={}",
                    input.sequence, input.head_dim
                ),
            });
        }
    }
    if input.query_tile == 0 || input.key_tile == 0 {
        return Err(VortexBackendError::InvalidLaunch {
            message: "attention query_tile and key_tile must be nonzero".to_owned(),
        });
    }
    Ok(elements)
}

fn attention_element_count(sequence: u32, head_dim: u32) -> Result<usize> {
    if sequence == 0 || head_dim == 0 {
        return Err(VortexBackendError::InvalidLaunch {
            message: "attention sequence and head_dim must be nonzero".to_owned(),
        });
    }
    (sequence as usize)
        .checked_mul(head_dim as usize)
        .ok_or_else(|| VortexBackendError::InvalidLaunch {
            message: "attention element count overflow".to_owned(),
        })
}

fn qk_dot_i32(q: &[i8], k: &[i8], query: usize, key: usize, head_dim: usize) -> i32 {
    let mut dot = 0i32;
    for depth in 0..head_dim {
        let q_value = i32::from(q[attention_index(query, depth, head_dim)]);
        let k_value = i32::from(k[attention_index(key, depth, head_dim)]);
        dot += q_value * k_value;
    }
    dot
}

const fn attention_index(row: usize, col: usize, head_dim: usize) -> usize {
    row * head_dim + col
}

fn f32_to_i8_trunc_clamped(value: f32) -> i8 {
    value.clamp(-128.0, 127.0).trunc() as i32 as i8
}

fn backend_runtime_trace(message: &str) {
    tracing::info!(target: "mandrel_vortex_backend::runtime", step = message, "Vortex runtime step");
    if env::var_os("MANDREL_VORTEX_RUNTIME_TRACE").is_none() {
        return;
    }

    println!("vortex-backend.runtime: {message}");
    if let Err(error) = io::stdout().flush() {
        tracing::warn!(%error, "failed to flush Vortex backend runtime trace output");
    }
}

fn launch_dims_from_kernel_launch<const ARGS: usize>(
    launch: &KernelLaunch<ARGS>,
) -> Result<VortexLaunchDims> {
    let grid = dim3_to_array(launch.grid);
    let block = dim3_to_array(launch.block);
    if grid.contains(&0) || block.contains(&0) {
        return Err(VortexError::InvalidValue {
            message: "kernel plan has a zero launch dimension",
        }
        .into());
    }
    Ok(VortexLaunchDims::new(
        grid,
        block,
        launch.shared_memory_bytes,
    ))
}

fn validate_launch_dims_against_caps(dims: VortexLaunchDims, caps: VortexDeviceCaps) -> Result<()> {
    if dims.grid.contains(&0) || dims.block.contains(&0) {
        return Err(VortexBackendError::InvalidLaunch {
            message: "grid and block dimensions must be nonzero".to_owned(),
        });
    }
    if caps.threads_per_warp == 0 || caps.warps_per_core == 0 {
        return Err(VortexBackendError::InvalidLaunch {
            message: format!(
                "device reported invalid execution resources: threads_per_warp={}, warps_per_core={}",
                caps.threads_per_warp, caps.warps_per_core
            ),
        });
    }

    let block_threads =
        product_u32_as_u64(dims.block).ok_or_else(|| VortexBackendError::InvalidLaunch {
            message: "block thread count overflowed u64".to_owned(),
        })?;
    let max_threads =
        caps.max_workgroup_threads()
            .ok_or_else(|| VortexBackendError::InvalidLaunch {
                message: "device max workgroup thread count overflowed u64".to_owned(),
            })?;
    if block_threads > max_threads {
        return Err(VortexBackendError::InvalidLaunch {
            message: format!(
                "block has {block_threads} threads, but this Vortex device supports at most {max_threads} threads per workgroup ({} warps × {} threads)",
                caps.warps_per_core, caps.threads_per_warp
            ),
        });
    }

    if dims.shared_memory_bytes != 0 {
        let warps_per_block = block_threads.div_ceil(caps.threads_per_warp);
        if warps_per_block == 0 || warps_per_block > caps.warps_per_core {
            return Err(VortexBackendError::InvalidLaunch {
                message: format!(
                    "block needs {warps_per_block} warps, but this Vortex device has {} warps per core",
                    caps.warps_per_core
                ),
            });
        }
        let blocks_per_core = caps.warps_per_core / warps_per_block;
        if blocks_per_core == 0 {
            return Err(VortexBackendError::InvalidLaunch {
                message: "local-memory budget is undefined because no workgroup fits on a core"
                    .to_owned(),
            });
        }
        let local_budget = caps.local_memory_bytes / blocks_per_core;
        let requested = u64::from(dims.shared_memory_bytes);
        if requested > local_budget {
            return Err(VortexBackendError::InvalidLaunch {
                message: format!(
                    "shared/local memory request is {requested} bytes per workgroup, but the per-workgroup budget is {local_budget} bytes ({} bytes/core across {blocks_per_core} resident workgroups)",
                    caps.local_memory_bytes
                ),
            });
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

#[cfg(test)]
mod tests {
    use super::{
        AttentionPrefillI8Args, AttentionPrefillI8Run, HardwareIdentityProbeArgs,
        VortexBackendConfig, VortexBackendError, launch_dims_from_kernel_launch,
        reference_attention_prefill_i8, validate_launch_dims_against_caps,
    };
    use crate::executor::VortexLaunchDims;
    use crate::vortex2::VortexDeviceCaps;
    use mandrel_compiler::compile_vortex_attention_prefill_kernel;
    use mandrel_kernel_ir::KernelSymbol;
    use mandrel_model_ir::AttentionOp;
    use std::mem::{align_of, size_of};
    use std::path::Path;

    #[test]
    fn hardware_identity_probe_args_match_single_pointer_abi() {
        assert_eq!(size_of::<HardwareIdentityProbeArgs>(), 8);
        assert_eq!(align_of::<HardwareIdentityProbeArgs>(), 8);
    }

    #[test]
    fn attention_args_match_lowered_abi_layout() {
        assert_eq!(std::mem::size_of::<AttentionPrefillI8Args>(), 48);
        assert_eq!(align_of::<AttentionPrefillI8Args>(), 8);
    }

    #[test]
    fn reference_attention_prefill_uses_uniform_softmax_for_equal_scores() {
        let input = AttentionPrefillI8Run {
            q: vec![0, 0, 0, 0],
            k: vec![0, 0, 0, 0],
            v: vec![10, 20, 30, 40],
            sequence: 2,
            head_dim: 2,
            query_tile: 1,
            key_tile: 2,
        };

        let output = match reference_attention_prefill_i8(&input) {
            Ok(output) => output,
            Err(error) => panic!("unexpected reference error: {error}"),
        };

        assert_eq!(output, vec![20, 30, 20, 30]);
    }

    #[test]
    fn backend_config_defaults_to_device_zero_without_default_images() {
        let config = VortexBackendConfig::new();

        assert_eq!(config.device_index, 0);
        assert!(
            config
                .kernel_image(KernelSymbol::AttentionPrefillI8)
                .is_none()
        );
    }

    #[test]
    fn backend_config_registers_kernel_images() {
        let config = VortexBackendConfig::from_kernel_image(
            KernelSymbol::AttentionPrefillI8,
            "target/vortex/attention_prefill_i8/kernel.vxbin",
        );

        assert_eq!(
            config.kernel_image(KernelSymbol::AttentionPrefillI8),
            Some(Path::new("target/vortex/attention_prefill_i8/kernel.vxbin"))
        );
    }

    #[test]
    fn backend_config_replaces_existing_kernel_image() {
        let config = VortexBackendConfig::new()
            .with_kernel_image(KernelSymbol::AttentionPrefillI8, "old.vxbin")
            .with_kernel_image(KernelSymbol::AttentionPrefillI8, "new.vxbin");

        assert_eq!(
            config.kernel_image(KernelSymbol::AttentionPrefillI8),
            Some(Path::new("new.vxbin"))
        );
    }

    #[test]
    fn backend_config_allows_device_override() {
        let config = VortexBackendConfig::new().with_device_index(2);

        assert_eq!(config.device_index, 2);
    }

    #[test]
    fn attention_launch_dims_are_derived_from_compiler_plan() {
        let plan = match compile_vortex_attention_prefill_kernel(AttentionOp::prefill_i8_demo()) {
            Ok(plan) => plan,
            Err(error) => panic!("unexpected compile error: {error:?}"),
        };
        let dims = match launch_dims_from_kernel_launch(&plan.launch) {
            Ok(dims) => dims,
            Err(error) => panic!("unexpected launch conversion error: {error}"),
        };

        assert_eq!(plan.launch.symbol, KernelSymbol::AttentionPrefillI8);
        assert_eq!(dims.grid, [16, 1, 1]);
        assert_eq!(dims.block, [4, 4, 1]);
        assert_eq!(dims.shared_memory_bytes, plan.launch.shared_memory_bytes);
    }

    #[test]
    fn launch_validation_accepts_current_rtl_attention_shape() {
        let caps = rtl_caps();
        let dims = VortexLaunchDims::new([16, 1, 1], [4, 4, 1], 0);

        assert!(validate_launch_dims_against_caps(dims, caps).is_ok());
    }

    #[test]
    fn launch_validation_rejects_oversized_workgroup() {
        let caps = rtl_caps();
        let dims = VortexLaunchDims::new([2, 2, 1], [16, 16, 1], 1024);
        let error = match validate_launch_dims_against_caps(dims, caps) {
            Ok(()) => panic!("expected oversized workgroup to be rejected"),
            Err(error) => error,
        };

        match error {
            VortexBackendError::InvalidLaunch { message } => {
                assert!(message.contains("block has 256 threads"));
                assert!(message.contains("at most 16 threads"));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    fn rtl_caps() -> VortexDeviceCaps {
        VortexDeviceCaps {
            threads_per_warp: 4,
            warps_per_core: 4,
            local_memory_bytes: 16 * 1024,
        }
    }
}
