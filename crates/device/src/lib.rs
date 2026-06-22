#![cfg_attr(not(feature = "std"), no_std)]

use mandrel_core::ElementType;
use mandrel_kernel_ir::KernelLaunch;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceBackend {
    HostReference,
    VortexSimx,
    VortexRtl,
    VortexFpga,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceCapabilities {
    pub backend: DeviceBackend,
    pub xlen: u32,
    pub max_workgroup_threads: u32,
    pub local_memory_bytes: u32,
    pub supports_int8: bool,
    pub supports_float32: bool,
    pub supports_tensor_cores: bool,
}

impl DeviceCapabilities {
    pub const fn vortex_simx_default() -> Self {
        Self {
            backend: DeviceBackend::VortexSimx,
            xlen: 64,
            max_workgroup_threads: 16,
            local_memory_bytes: 32 * 1024,
            supports_int8: true,
            supports_float32: true,
            supports_tensor_cores: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferId(pub u32);

impl BufferId {
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySpace {
    HostVisible,
    DeviceLocal,
    Unified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferDesc {
    pub id: BufferId,
    pub bytes: usize,
    pub element_type: ElementType,
    pub memory: MemorySpace,
}

impl BufferDesc {
    pub const fn new(
        id: BufferId,
        bytes: usize,
        element_type: ElementType,
        memory: MemorySpace,
    ) -> Self {
        Self {
            id,
            bytes,
            element_type,
            memory,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceCommand<const ARGS: usize> {
    Launch(KernelLaunch<ARGS>),
    Fence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandBuffer<const COMMANDS: usize, const ARGS: usize> {
    pub commands: [DeviceCommand<ARGS>; COMMANDS],
}

impl<const COMMANDS: usize, const ARGS: usize> CommandBuffer<COMMANDS, ARGS> {
    pub const fn new(commands: [DeviceCommand<ARGS>; COMMANDS]) -> Self {
        Self { commands }
    }

    pub const fn len(&self) -> usize {
        COMMANDS
    }

    pub const fn is_empty(&self) -> bool {
        COMMANDS == 0
    }
}

#[cfg(test)]
mod tests {
    use super::{BufferId, CommandBuffer, DeviceCapabilities, DeviceCommand};
    use mandrel_kernel_ir::{Dim3, KernelArg, KernelLaunch, KernelSymbol};

    #[test]
    fn default_vortex_caps_match_current_simx_config() {
        let caps = DeviceCapabilities::vortex_simx_default();

        assert_eq!(caps.xlen, 64);
        assert!(caps.supports_int8);
        assert_eq!(caps.max_workgroup_threads, 16);
    }

    #[test]
    fn command_buffer_wraps_kernel_launch() {
        let launch = KernelLaunch::new(
            KernelSymbol::VecAddF32,
            Dim3::new(1, 1, 1),
            Dim3::new(64, 1, 1),
            0,
            [KernelArg::buffer(0, BufferId(1).as_u32())],
        );
        let command_buffer =
            CommandBuffer::new([DeviceCommand::Launch(launch), DeviceCommand::Fence]);

        assert_eq!(command_buffer.len(), 2);
        assert!(!command_buffer.is_empty());
    }
}
