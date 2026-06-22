#![cfg_attr(not(feature = "std"), no_std)]

/// Element type used by the graph/runtime boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    I8,
    I16,
    I32,
    F16,
    F32,
}

impl ElementType {
    pub const fn byte_size(self) -> usize {
        match self {
            Self::I8 => 1,
            Self::I16 | Self::F16 => 2,
            Self::I32 | Self::F32 => 4,
        }
    }
}

/// Tensor memory layout. Keep this small because it is shared by host tools,
/// the runtime, and future accelerator ABI lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    RowMajor,
    Nhwc,
    Nchw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape<const RANK: usize> {
    dims: [usize; RANK],
}

impl<const RANK: usize> Shape<RANK> {
    pub const fn new(dims: [usize; RANK]) -> Self {
        Self { dims }
    }

    pub const fn dims(&self) -> &[usize; RANK] {
        &self.dims
    }

    pub const fn element_count(&self) -> usize {
        let mut index = 0;
        let mut count = 1;
        while index < RANK {
            count *= self.dims[index];
            index += 1;
        }
        count
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorDesc<const RANK: usize> {
    pub shape: Shape<RANK>,
    pub element_type: ElementType,
    pub layout: Layout,
}

impl<const RANK: usize> TensorDesc<RANK> {
    pub const fn byte_len(&self) -> usize {
        self.shape.element_count() * self.element_type.byte_size()
    }
}
