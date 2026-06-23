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
            shape: Shape::new([64, 64]),
            element_type: ElementType::I8,
            layout: Layout::RowMajor,
        },
        output: TensorDesc {
            shape: Shape::new([64, 64]),
            element_type: ElementType::I8,
            layout: Layout::RowMajor,
        },
    }
}

pub const fn example_attention_vortex_plan() -> RuntimePlan<2, 8> {
    let launch = KernelLaunch::new(
        KernelSymbol::AttentionPrefillI8,
        Dim3::new(16, 1, 1),
        Dim3::new(4, 4, 1),
        2336,
        [
            KernelArg::buffer(0, 0),
            KernelArg::buffer(1, 1),
            KernelArg::buffer(2, 2),
            KernelArg::buffer(3, 3),
            KernelArg::u32(4, 64),
            KernelArg::u32(5, 64),
            KernelArg::u32(6, 4),
            KernelArg::u32(7, 16),
        ],
    );

    RuntimePlan {
        target: RuntimeTarget::VortexSimx,
        command_buffer: CommandBuffer::new([DeviceCommand::Launch(launch), DeviceCommand::Fence]),
    }
}

#[cfg(test)]
mod tests {
    use super::{RuntimeTarget, example_attention_vortex_plan, example_io};
    use mandrel_device::{DeviceBackend, DeviceCommand};

    #[test]
    fn example_io_is_attention_prefill_shaped() {
        let io = example_io();

        assert_eq!(io.input.shape.dims(), &[64, 64]);
        assert_eq!(io.output.shape.dims(), &[64, 64]);
    }

    #[test]
    fn example_plan_targets_vortex_attention_prefill() {
        let plan = example_attention_vortex_plan();

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
