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
/// This intentionally stores only stable scalar fields so measured traces from Vortex `simx`, RTL,
/// or future backends can be compared against static `KernelMetrics` without depending on backend
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
    use super::{KernelCounterTrace, KernelLaunchTrace, RuntimeTraceSummary};

    #[test]
    fn summarizes_launch_and_transfer_shape() {
        let launch = KernelLaunchTrace::new("attention_prefill_i8", [16, 1, 1], [4, 4, 1], 2336);
        let trace = RuntimeTraceSummary::new(launch, 4096, 4096, KernelCounterTrace::empty());

        assert_eq!(trace.launch.workgroup_count(), 16);
        assert_eq!(trace.launch.threads_per_workgroup(), 16);
        assert_eq!(trace.total_transfer_bytes(), 8192);
    }
}
