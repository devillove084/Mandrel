/// Direction for host/device transfer events in measured runtime traces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferTransferTrace {
    pub direction: TransferDirection,
    pub bytes: usize,
}

impl BufferTransferTrace {
    pub const fn new(direction: TransferDirection, bytes: usize) -> Self {
        Self { direction, bytes }
    }
}

/// Backend-neutral subset of a GPU kernel launch trace.
///
/// This intentionally stores only stable scalar fields so measured traces from Vortex RTL or
/// future backends can be compared against static `KernelMetrics` without depending on backend
/// handle layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelLaunchTrace {
    pub kernel_symbol: &'static str,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_memory_bytes: u32,
}

impl KernelLaunchTrace {
    pub const fn new(
        kernel_symbol: &'static str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_memory_bytes: u32,
    ) -> Self {
        Self {
            kernel_symbol,
            grid,
            block,
            shared_memory_bytes,
        }
    }

    pub const fn workgroup_count(self) -> u32 {
        self.grid[0] * self.grid[1] * self.grid[2]
    }

    pub const fn threads_per_workgroup(self) -> u32 {
        self.block[0] * self.block[1] * self.block[2]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelCounterTrace {
    pub instructions: Option<u64>,
    pub cycles: Option<u64>,
}

impl KernelCounterTrace {
    pub const fn empty() -> Self {
        Self {
            instructions: None,
            cycles: None,
        }
    }

    pub const fn new(instructions: Option<u64>, cycles: Option<u64>) -> Self {
        Self {
            instructions,
            cycles,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VortexPerfParseError {
    MissingPrefix,
    MissingInstructions,
    MissingCycles,
    InvalidInstructions,
    InvalidCycles,
}

pub fn parse_vortex_perf_counters(line: &str) -> Result<KernelCounterTrace, VortexPerfParseError> {
    let Some(fields) = line.trim().strip_prefix("PERF:") else {
        return Err(VortexPerfParseError::MissingPrefix);
    };

    let mut instructions = None;
    let mut cycles = None;
    for field in fields.split(',') {
        let trimmed = field.trim();
        let Some((name, value)) = trimmed.split_once('=') else {
            continue;
        };
        let name = name.trim();
        let value = value.trim();
        match name {
            "instrs" => {
                instructions = Some(
                    value
                        .parse::<u64>()
                        .map_err(|_error| VortexPerfParseError::InvalidInstructions)?,
                );
            }
            "cycles" => {
                cycles = Some(
                    value
                        .parse::<u64>()
                        .map_err(|_error| VortexPerfParseError::InvalidCycles)?,
                );
            }
            _ => {}
        }
    }

    Ok(KernelCounterTrace::new(
        Some(instructions.ok_or(VortexPerfParseError::MissingInstructions)?),
        Some(cycles.ok_or(VortexPerfParseError::MissingCycles)?),
    ))
}

pub fn find_vortex_perf_counters(
    output: &str,
) -> Option<Result<KernelCounterTrace, VortexPerfParseError>> {
    output
        .lines()
        .find(|line| line.trim().starts_with("PERF:"))
        .map(parse_vortex_perf_counters)
}

pub const MANDREL_RUNTIME_TRACE_PREFIX: &str = "MANDREL_RUNTIME_TRACE:";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MandrelRuntimeTraceParseError {
    MissingPrefix,
    MissingKernelSymbol,
    UnexpectedKernelSymbol,
    MissingGrid,
    MissingBlock,
    MissingSharedMemoryBytes,
    MissingHostToDeviceBytes,
    MissingDeviceToHostBytes,
    InvalidGrid,
    InvalidBlock,
    InvalidSharedMemoryBytes,
    InvalidHostToDeviceBytes,
    InvalidDeviceToHostBytes,
}

pub fn parse_mandrel_runtime_trace_summary(
    line: &str,
    expected_kernel_symbol: &'static str,
) -> Result<RuntimeTraceSummary, MandrelRuntimeTraceParseError> {
    let Some(fields) = line.trim().strip_prefix(MANDREL_RUNTIME_TRACE_PREFIX) else {
        return Err(MandrelRuntimeTraceParseError::MissingPrefix);
    };

    let mut kernel_symbol = None;
    let mut grid = None;
    let mut block = None;
    let mut shared_memory_bytes = None;
    let mut host_to_device_bytes = None;
    let mut device_to_host_bytes = None;

    for field in fields.split_whitespace() {
        let Some((name, value)) = field.split_once('=') else {
            continue;
        };
        match name.trim() {
            "kernel" => kernel_symbol = Some(value.trim()),
            "grid" => grid = Some(parse_u32_triplet(value.trim())),
            "block" => block = Some(parse_u32_triplet(value.trim())),
            "shared_memory_bytes" => {
                shared_memory_bytes = Some(
                    value
                        .trim()
                        .parse::<u32>()
                        .map_err(|_error| MandrelRuntimeTraceParseError::InvalidSharedMemoryBytes),
                )
            }
            "host_to_device_bytes" => {
                host_to_device_bytes = Some(
                    value
                        .trim()
                        .parse::<usize>()
                        .map_err(|_error| MandrelRuntimeTraceParseError::InvalidHostToDeviceBytes),
                )
            }
            "device_to_host_bytes" => {
                device_to_host_bytes = Some(
                    value
                        .trim()
                        .parse::<usize>()
                        .map_err(|_error| MandrelRuntimeTraceParseError::InvalidDeviceToHostBytes),
                )
            }
            _ => {}
        }
    }

    let kernel_symbol = kernel_symbol.ok_or(MandrelRuntimeTraceParseError::MissingKernelSymbol)?;
    if kernel_symbol != expected_kernel_symbol {
        return Err(MandrelRuntimeTraceParseError::UnexpectedKernelSymbol);
    }
    let grid = grid
        .ok_or(MandrelRuntimeTraceParseError::MissingGrid)?
        .ok_or(MandrelRuntimeTraceParseError::InvalidGrid)?;
    let block = block
        .ok_or(MandrelRuntimeTraceParseError::MissingBlock)?
        .ok_or(MandrelRuntimeTraceParseError::InvalidBlock)?;
    let shared_memory_bytes =
        shared_memory_bytes.ok_or(MandrelRuntimeTraceParseError::MissingSharedMemoryBytes)??;
    let host_to_device_bytes =
        host_to_device_bytes.ok_or(MandrelRuntimeTraceParseError::MissingHostToDeviceBytes)??;
    let device_to_host_bytes =
        device_to_host_bytes.ok_or(MandrelRuntimeTraceParseError::MissingDeviceToHostBytes)??;

    Ok(RuntimeTraceSummary::new(
        KernelLaunchTrace::new(expected_kernel_symbol, grid, block, shared_memory_bytes),
        host_to_device_bytes,
        device_to_host_bytes,
        KernelCounterTrace::empty(),
    ))
}

pub fn find_mandrel_runtime_trace_summary(
    output: &str,
    expected_kernel_symbol: &'static str,
) -> Option<Result<RuntimeTraceSummary, MandrelRuntimeTraceParseError>> {
    output
        .lines()
        .find(|line| line.trim().starts_with(MANDREL_RUNTIME_TRACE_PREFIX))
        .map(|line| parse_mandrel_runtime_trace_summary(line, expected_kernel_symbol))
}

fn parse_u32_triplet(value: &str) -> Option<[u32; 3]> {
    let mut parts = value.split('x');
    let x = parts.next()?.parse::<u32>().ok()?;
    let y = parts.next()?.parse::<u32>().ok()?;
    let z = parts.next()?.parse::<u32>().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some([x, y, z])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeTraceSummary {
    pub launch: KernelLaunchTrace,
    pub host_to_device_bytes: usize,
    pub device_to_host_bytes: usize,
    pub counters: KernelCounterTrace,
}

impl RuntimeTraceSummary {
    pub const fn new(
        launch: KernelLaunchTrace,
        host_to_device_bytes: usize,
        device_to_host_bytes: usize,
        counters: KernelCounterTrace,
    ) -> Self {
        Self {
            launch,
            host_to_device_bytes,
            device_to_host_bytes,
            counters,
        }
    }

    pub const fn total_transfer_bytes(self) -> usize {
        self.host_to_device_bytes + self.device_to_host_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::{
        KernelCounterTrace, KernelLaunchTrace, MandrelRuntimeTraceParseError, RuntimeTraceSummary,
        VortexPerfParseError, find_mandrel_runtime_trace_summary, find_vortex_perf_counters,
        parse_mandrel_runtime_trace_summary, parse_vortex_perf_counters,
    };

    #[test]
    fn summarizes_launch_and_transfer_shape() {
        let launch = KernelLaunchTrace::new("attention_prefill_i8", [16, 1, 1], [4, 4, 1], 2336);
        let trace = RuntimeTraceSummary::new(launch, 4096, 4096, KernelCounterTrace::empty());

        assert_eq!(trace.launch.workgroup_count(), 16);
        assert_eq!(trace.launch.threads_per_workgroup(), 16);
        assert_eq!(trace.total_transfer_bytes(), 8192);
    }

    #[test]
    fn parses_vortex_perf_counters() {
        let counters =
            match parse_vortex_perf_counters("PERF: instrs=165144, cycles=414598, IPC=0.398") {
                Ok(counters) => counters,
                Err(error) => panic!("unexpected parse error: {error:?}"),
            };

        assert_eq!(counters.instructions, Some(165_144));
        assert_eq!(counters.cycles, Some(414_598));
    }

    #[test]
    fn parses_vortex_perf_counters_with_extra_spacing() {
        let counters = match parse_vortex_perf_counters(
            "  PERF:   instrs = 7 , cycles = 11 , IPC = 0.636  ",
        ) {
            Ok(counters) => counters,
            Err(error) => panic!("unexpected parse error: {error:?}"),
        };

        assert_eq!(counters, KernelCounterTrace::new(Some(7), Some(11)));
    }

    #[test]
    fn rejects_malformed_vortex_perf_counters() {
        assert_eq!(
            parse_vortex_perf_counters("instrs=1, cycles=2"),
            Err(VortexPerfParseError::MissingPrefix)
        );
        assert_eq!(
            parse_vortex_perf_counters("PERF: cycles=2"),
            Err(VortexPerfParseError::MissingInstructions)
        );
        assert_eq!(
            parse_vortex_perf_counters("PERF: instrs=1"),
            Err(VortexPerfParseError::MissingCycles)
        );
        assert_eq!(
            parse_vortex_perf_counters("PERF: instrs=x, cycles=2"),
            Err(VortexPerfParseError::InvalidInstructions)
        );
        assert_eq!(
            parse_vortex_perf_counters("PERF: instrs=1, cycles=x"),
            Err(VortexPerfParseError::InvalidCycles)
        );
    }

    #[test]
    fn parses_mandrel_runtime_trace_summary() {
        let trace = match parse_mandrel_runtime_trace_summary(
            "MANDREL_RUNTIME_TRACE: kernel=attention_prefill_i8 grid=16x1x1 block=4x4x1 shared_memory_bytes=2336 host_to_device_bytes=384 device_to_host_bytes=128",
            "attention_prefill_i8",
        ) {
            Ok(trace) => trace,
            Err(error) => panic!("unexpected parse error: {error:?}"),
        };

        assert_eq!(trace.launch.kernel_symbol, "attention_prefill_i8");
        assert_eq!(trace.launch.grid, [16, 1, 1]);
        assert_eq!(trace.launch.block, [4, 4, 1]);
        assert_eq!(trace.launch.shared_memory_bytes, 2336);
        assert_eq!(trace.host_to_device_bytes, 384);
        assert_eq!(trace.device_to_host_bytes, 128);
        assert_eq!(trace.counters, KernelCounterTrace::empty());
    }

    #[test]
    fn rejects_malformed_mandrel_runtime_trace_summary() {
        assert_eq!(
            parse_mandrel_runtime_trace_summary(
                "kernel=attention_prefill_i8 grid=16x1x1 block=4x4x1 shared_memory_bytes=2336 host_to_device_bytes=384 device_to_host_bytes=128",
                "attention_prefill_i8",
            ),
            Err(MandrelRuntimeTraceParseError::MissingPrefix)
        );
        assert_eq!(
            parse_mandrel_runtime_trace_summary(
                "MANDREL_RUNTIME_TRACE: kernel=other grid=16x1x1 block=4x4x1 shared_memory_bytes=2336 host_to_device_bytes=384 device_to_host_bytes=128",
                "attention_prefill_i8",
            ),
            Err(MandrelRuntimeTraceParseError::UnexpectedKernelSymbol)
        );
        assert_eq!(
            parse_mandrel_runtime_trace_summary(
                "MANDREL_RUNTIME_TRACE: kernel=attention_prefill_i8 grid=16,1,1 block=4x4x1 shared_memory_bytes=2336 host_to_device_bytes=384 device_to_host_bytes=128",
                "attention_prefill_i8",
            ),
            Err(MandrelRuntimeTraceParseError::InvalidGrid)
        );
        assert_eq!(
            parse_mandrel_runtime_trace_summary(
                "MANDREL_RUNTIME_TRACE: kernel=attention_prefill_i8 grid=16x1x1 block=4x4x1 shared_memory_bytes=x host_to_device_bytes=384 device_to_host_bytes=128",
                "attention_prefill_i8",
            ),
            Err(MandrelRuntimeTraceParseError::InvalidSharedMemoryBytes)
        );
    }

    #[test]
    fn finds_mandrel_runtime_trace_summary_in_log_output() {
        let output = "attention.runtime: done\nMANDREL_RUNTIME_TRACE: kernel=attention_prefill_i8 grid=16x1x1 block=4x4x1 shared_memory_bytes=2336 host_to_device_bytes=384 device_to_host_bytes=128\n";
        let result = match find_mandrel_runtime_trace_summary(output, "attention_prefill_i8") {
            Some(result) => result,
            None => panic!("expected Mandrel runtime trace line in output"),
        };
        let trace = match result {
            Ok(trace) => trace,
            Err(error) => panic!("unexpected parse error: {error:?}"),
        };

        assert_eq!(trace.launch.workgroup_count(), 16);
        assert_eq!(trace.total_transfer_bytes(), 512);
    }

    #[test]
    fn finds_vortex_perf_counters_in_log_output() {
        let output = "attention.runtime: launching\nPERF: instrs=165144, cycles=414598, IPC=0.398\nattention.runtime: done";
        let result = match find_vortex_perf_counters(output) {
            Some(result) => result,
            None => panic!("expected PERF line in output"),
        };
        let counters = match result {
            Ok(counters) => counters,
            Err(error) => panic!("unexpected parse error: {error:?}"),
        };

        assert_eq!(
            counters,
            KernelCounterTrace::new(Some(165_144), Some(414_598))
        );
    }

    #[test]
    fn returns_none_when_log_has_no_vortex_perf_line() {
        assert_eq!(find_vortex_perf_counters("attention.runtime: done"), None);
    }
}
