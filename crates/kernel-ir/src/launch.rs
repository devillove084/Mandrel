use crate::symbol::KernelSymbol;

/// A GPU-style three-dimensional launch shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub const fn one() -> Self {
        Self::new(1, 1, 1)
    }

    pub const fn elements(self) -> u32 {
        self.x * self.y * self.z
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelArgValue {
    Buffer(u32),
    U32(u32),
    I32(i32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelArg {
    pub index: u8,
    pub value: KernelArgValue,
}

impl KernelArg {
    pub const fn buffer(index: u8, buffer_id: u32) -> Self {
        Self {
            index,
            value: KernelArgValue::Buffer(buffer_id),
        }
    }

    pub const fn u32(index: u8, value: u32) -> Self {
        Self {
            index,
            value: KernelArgValue::U32(value),
        }
    }

    pub const fn i32(index: u8, value: i32) -> Self {
        Self {
            index,
            value: KernelArgValue::I32(value),
        }
    }
}

/// A target-independent GPU kernel launch packet. The Vortex backend is
/// responsible for mapping this to its runtime/driver ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelLaunch<const ARGS: usize> {
    pub symbol: KernelSymbol,
    pub grid: Dim3,
    pub block: Dim3,
    pub shared_memory_bytes: u32,
    pub args: [KernelArg; ARGS],
}

impl<const ARGS: usize> KernelLaunch<ARGS> {
    pub const fn new(
        symbol: KernelSymbol,
        grid: Dim3,
        block: Dim3,
        shared_memory_bytes: u32,
        args: [KernelArg; ARGS],
    ) -> Self {
        Self {
            symbol,
            grid,
            block,
            shared_memory_bytes,
            args,
        }
    }

    pub const fn workgroup_count(self) -> u32 {
        self.grid.elements()
    }

    pub const fn threads_per_workgroup(self) -> u32 {
        self.block.elements()
    }
}

#[cfg(test)]
mod tests {
    use super::{Dim3, KernelArg, KernelArgValue, KernelLaunch};
    use crate::symbol::KernelSymbol;

    #[test]
    fn launch_counts_threads_and_workgroups() {
        let launch = KernelLaunch::new(
            KernelSymbol::AttentionPrefillI8,
            Dim3::new(4, 2, 1),
            Dim3::new(16, 16, 1),
            1024,
            [KernelArg::buffer(0, 7)],
        );

        assert_eq!(launch.workgroup_count(), 8);
        assert_eq!(launch.threads_per_workgroup(), 256);
        assert_eq!(launch.args[0].value, KernelArgValue::Buffer(7));
    }
}
