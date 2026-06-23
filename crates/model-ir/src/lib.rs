#![cfg_attr(not(feature = "std"), no_std)]

use mandrel_core::{ElementType, Layout, Shape, TensorDesc};

pub type TensorId = u32;
pub type NodeId = u32;

/// High-level target selection for analysis and lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetKind {
    Reference,
    VortexSimx,
    VortexRtl,
    VortexFpga,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Quantization {
    pub scale_numerator: i32,
    pub scale_denominator: i32,
    pub zero_point: i32,
}

impl Quantization {
    pub const fn symmetric_unit() -> Self {
        Self {
            scale_numerator: 1,
            scale_denominator: 1,
            zero_point: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tensor {
    pub id: TensorId,
    pub desc: TensorDesc<2>,
    pub quantization: Option<Quantization>,
}

impl Tensor {
    pub const fn rank2_i8_row_major(id: TensorId, rows: usize, cols: usize) -> Self {
        Self {
            id,
            desc: TensorDesc {
                shape: Shape::new([rows, cols]),
                element_type: ElementType::I8,
                layout: Layout::RowMajor,
            },
            quantization: Some(Quantization::symmetric_unit()),
        }
    }

    pub const fn rank2_f32_row_major(id: TensorId, rows: usize, cols: usize) -> Self {
        Self {
            id,
            desc: TensorDesc {
                shape: Shape::new([rows, cols]),
                element_type: ElementType::F32,
                layout: Layout::RowMajor,
            },
            quantization: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionShape {
    pub sequence: usize,
    pub head_dim: usize,
}

impl AttentionShape {
    pub const fn new(sequence: usize, head_dim: usize) -> Self {
        Self { sequence, head_dim }
    }

    pub const fn output_elements(self) -> usize {
        self.sequence * self.head_dim
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionTensors {
    pub q: TensorId,
    pub k: TensorId,
    pub v: TensorId,
    pub out: TensorId,
}

impl AttentionTensors {
    pub const fn new(q: TensorId, k: TensorId, v: TensorId, out: TensorId) -> Self {
        Self { q, k, v, out }
    }
}

/// Compact attention-prefill payload. Scheduling and KV-cache layout are selected separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionOp {
    pub tensors: AttentionTensors,
    pub shape: AttentionShape,
    pub element_type: ElementType,
}

impl AttentionOp {
    pub const fn new(
        tensors: AttentionTensors,
        shape: AttentionShape,
        element_type: ElementType,
    ) -> Self {
        Self {
            tensors,
            shape,
            element_type,
        }
    }

    pub const fn prefill_i8_demo() -> Self {
        Self::new(
            AttentionTensors::new(0, 1, 2, 3),
            AttentionShape::new(64, 64),
            ElementType::I8,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxAxis {
    Rows,
    Cols,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoftmaxShape {
    pub rows: usize,
    pub cols: usize,
}

impl SoftmaxShape {
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    pub const fn elements(self) -> usize {
        self.rows * self.cols
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoftmaxTensors {
    pub input: TensorId,
    pub out: TensorId,
}

impl SoftmaxTensors {
    pub const fn new(input: TensorId, out: TensorId) -> Self {
        Self { input, out }
    }
}

/// Row/column softmax payload used as an attention-building-block IR node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoftmaxOp {
    pub tensors: SoftmaxTensors,
    pub shape: SoftmaxShape,
    pub axis: SoftmaxAxis,
    pub element_type: ElementType,
}

impl SoftmaxOp {
    pub const fn new(
        tensors: SoftmaxTensors,
        shape: SoftmaxShape,
        axis: SoftmaxAxis,
        element_type: ElementType,
    ) -> Self {
        Self {
            tensors,
            shape,
            axis,
            element_type,
        }
    }

    pub const fn row_f32_demo() -> Self {
        Self::new(
            SoftmaxTensors::new(0, 1),
            SoftmaxShape::new(2, 4),
            SoftmaxAxis::Rows,
            ElementType::F32,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    Attention(AttentionOp),
    Softmax(SoftmaxOp),
    Transpose2d {
        input: TensorId,
        out: TensorId,
    },
    Scale {
        input: TensorId,
        out: TensorId,
        scale: Quantization,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Node {
    pub id: NodeId,
    pub kind: OpKind,
}

/// Target-side constraints that guide schedule search and lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetConstraints {
    pub target: TargetKind,
    pub max_workgroup_threads: usize,
    pub local_memory_bytes: usize,
    pub preferred_subgroup_width: usize,
    pub supports_int8: bool,
}

impl TargetConstraints {
    pub const fn vortex_simx_default() -> Self {
        Self {
            target: TargetKind::VortexSimx,
            max_workgroup_threads: 16,
            local_memory_bytes: 32 * 1024,
            preferred_subgroup_width: 4,
            supports_int8: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AttentionOp, SoftmaxAxis, SoftmaxOp, TargetConstraints, TargetKind};

    #[test]
    fn builds_attention_prefill_demo_op() {
        let op = AttentionOp::prefill_i8_demo();

        assert_eq!(op.tensors.q, 0);
        assert_eq!(op.tensors.k, 1);
        assert_eq!(op.tensors.v, 2);
        assert_eq!(op.tensors.out, 3);
        assert_eq!(op.shape.sequence, 64);
        assert_eq!(op.shape.head_dim, 64);
        assert_eq!(op.shape.output_elements(), 4096);
    }

    #[test]
    fn builds_row_softmax_demo_op() {
        let op = SoftmaxOp::row_f32_demo();

        assert_eq!(op.tensors.input, 0);
        assert_eq!(op.tensors.out, 1);
        assert_eq!(op.shape.elements(), 8);
        assert_eq!(op.axis, SoftmaxAxis::Rows);
    }

    #[test]
    fn vortex_constraints_target_simx() {
        let constraints = TargetConstraints::vortex_simx_default();

        assert_eq!(constraints.target, TargetKind::VortexSimx);
        assert_eq!(constraints.max_workgroup_threads, 16);
        assert_eq!(constraints.preferred_subgroup_width, 4);
    }
}
