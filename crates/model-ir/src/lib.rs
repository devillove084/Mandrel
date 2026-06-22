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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl MatmulShape {
    pub const fn new(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k }
    }

    pub const fn square(size: usize) -> Self {
        Self {
            m: size,
            n: size,
            k: size,
        }
    }

    pub const fn output_elements(self) -> usize {
        self.m * self.n
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulTensors {
    pub lhs: TensorId,
    pub rhs: TensorId,
    pub out: TensorId,
}

impl MatmulTensors {
    pub const fn new(lhs: TensorId, rhs: TensorId, out: TensorId) -> Self {
        Self { lhs, rhs, out }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulTypes {
    pub lhs: ElementType,
    pub rhs: ElementType,
    pub out: ElementType,
}

impl MatmulTypes {
    pub const fn i8_to_i32() -> Self {
        Self {
            lhs: ElementType::I8,
            rhs: ElementType::I8,
            out: ElementType::I32,
        }
    }

    pub const fn f32_to_f32() -> Self {
        Self {
            lhs: ElementType::F32,
            rhs: ElementType::F32,
            out: ElementType::F32,
        }
    }
}

/// HLO/ggml-like matmul node payload. Scheduling and device placement are separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulOp {
    pub tensors: MatmulTensors,
    pub shape: MatmulShape,
    pub types: MatmulTypes,
}

impl MatmulOp {
    pub const fn new(tensors: MatmulTensors, shape: MatmulShape, types: MatmulTypes) -> Self {
        Self {
            tensors,
            shape,
            types,
        }
    }

    pub const fn ggml_mul_mat_i8_demo() -> Self {
        Self::new(
            MatmulTensors::new(0, 1, 2),
            MatmulShape::new(32, 32, 64),
            MatmulTypes::i8_to_i32(),
        )
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

/// Compact classic-attention payload for future Vortex kernel fusion experiments.
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
pub enum OpKind {
    Matmul(MatmulOp),
    Attention(AttentionOp),
    Transpose2d {
        input: TensorId,
        out: TensorId,
    },
    Softmax {
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
    use super::{AttentionOp, MatmulOp, TargetConstraints, TargetKind};

    #[test]
    fn builds_ggml_mul_mat_demo_op() {
        let op = MatmulOp::ggml_mul_mat_i8_demo();

        assert_eq!(op.shape.m, 32);
        assert_eq!(op.shape.n, 32);
        assert_eq!(op.shape.k, 64);
        assert_eq!(op.shape.output_elements(), 1024);
    }

    #[test]
    fn builds_attention_prefill_demo_op() {
        let op = AttentionOp::prefill_i8_demo();

        assert_eq!(op.tensors.q, 0);
        assert_eq!(op.tensors.k, 1);
        assert_eq!(op.tensors.v, 2);
        assert_eq!(op.tensors.out, 3);
        assert_eq!(op.shape.sequence, 64);
        assert_eq!(op.shape.head_dim, 64);
    }

    #[test]
    fn vortex_constraints_target_simx() {
        let constraints = TargetConstraints::vortex_simx_default();

        assert_eq!(constraints.target, TargetKind::VortexSimx);
        assert_eq!(constraints.max_workgroup_threads, 16);
        assert_eq!(constraints.preferred_subgroup_width, 4);
    }
}
