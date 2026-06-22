#![cfg_attr(not(feature = "std"), no_std)]

use mandrel_core::{ElementType, Layout, Shape, TensorDesc};
use mandrel_device::{CommandBuffer, DeviceBackend, DeviceCommand};
use mandrel_kernel_ir::{Dim3, KernelArg, KernelLaunch, KernelSymbol};

pub type TensorId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeTarget {
    HostReference,
    VortexSimx,
    VortexRtl,
    VortexFpga,
}

impl RuntimeTarget {
    pub const fn device_backend(self) -> DeviceBackend {
        match self {
            Self::HostReference => DeviceBackend::HostReference,
            Self::VortexSimx => DeviceBackend::VortexSimx,
            Self::VortexRtl => DeviceBackend::VortexRtl,
            Self::VortexFpga => DeviceBackend::VortexFpga,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimePlan<const COMMANDS: usize, const ARGS: usize> {
    pub target: RuntimeTarget,
    pub command_buffer: CommandBuffer<COMMANDS, ARGS>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelIo {
    pub input: TensorDesc<2>,
    pub output: TensorDesc<2>,
}

pub const fn example_io() -> ModelIo {
    ModelIo {
        input: TensorDesc {
            shape: Shape::new([1, 128]),
            element_type: ElementType::I8,
            layout: Layout::RowMajor,
        },
        output: TensorDesc {
            shape: Shape::new([1, 64]),
            element_type: ElementType::I32,
            layout: Layout::RowMajor,
        },
    }
}

pub const fn example_vortex_plan() -> RuntimePlan<2, 6> {
    let launch = KernelLaunch::new(
        KernelSymbol::MatmulI8I32,
        Dim3::new(8, 8, 1),
        Dim3::new(4, 4, 1),
        0,
        [
            KernelArg::buffer(0, 0),
            KernelArg::buffer(1, 1),
            KernelArg::buffer(2, 2),
            KernelArg::u32(3, 32),
            KernelArg::u32(4, 32),
            KernelArg::u32(5, 64),
        ],
    );

    RuntimePlan {
        target: RuntimeTarget::VortexSimx,
        command_buffer: CommandBuffer::new([DeviceCommand::Launch(launch), DeviceCommand::Fence]),
    }
}

#[cfg(test)]
mod tests {
    use super::{RuntimeTarget, example_io, example_vortex_plan};
    use mandrel_device::{DeviceBackend, DeviceCommand};

    #[test]
    fn example_io_is_int8_to_int32() {
        let io = example_io();

        assert_eq!(io.input.shape.dims(), &[1, 128]);
        assert_eq!(io.output.shape.dims(), &[1, 64]);
    }

    #[test]
    fn example_plan_targets_vortex() {
        let plan = example_vortex_plan();

        assert_eq!(plan.target.device_backend(), DeviceBackend::VortexSimx);
        assert_eq!(plan.command_buffer.len(), 2);
        assert!(matches!(
            plan.command_buffer.commands[1],
            DeviceCommand::Fence
        ));
        assert_eq!(
            RuntimeTarget::VortexSimx.device_backend(),
            DeviceBackend::VortexSimx
        );
    }
}
