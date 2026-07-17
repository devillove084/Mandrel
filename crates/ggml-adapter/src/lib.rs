#![cfg_attr(not(feature = "std"), no_std)]

use mandrel_model_ir::{AttentionOp, AttentionShape, AttentionTensors};
use mandrel_target_ir::{DeviceBackend, DeviceCapabilities};

pub const BACKEND_NAME: &str = "mandrel-vortex";

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlDType {
    I8 = 1,
    I32 = 2,
    F32 = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlTensor2d {
    pub rows: u32,
    pub cols: u32,
    pub dtype: GgmlDType,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlAttentionPrefillRequest {
    pub q: GgmlTensor2d,
    pub k: GgmlTensor2d,
    pub v: GgmlTensor2d,
    pub out: GgmlTensor2d,
}

impl GgmlAttentionPrefillRequest {
    pub const fn dense_i8(sequence: u32, head_dim: u32) -> Self {
        let tensor = GgmlTensor2d {
            rows: sequence,
            cols: head_dim,
            dtype: GgmlDType::I8,
        };
        Self {
            q: tensor,
            k: tensor,
            v: tensor,
            out: tensor,
        }
    }

    pub fn to_model_ir_for(self, caps: DeviceCapabilities) -> Option<AttentionOp> {
        if !can_offload_attention_prefill(self, caps) {
            return None;
        }

        Some(AttentionOp::new(
            AttentionTensors::new(0, 1, 2, 3),
            AttentionShape::new(self.q.rows as usize, self.q.cols as usize),
            mandrel_core::ElementType::I8,
        ))
    }

    pub fn to_model_ir(self) -> Option<AttentionOp> {
        self.to_model_ir_for(DeviceCapabilities::vortex_simx_default())
    }
}

pub const fn backend_name() -> &'static str {
    BACKEND_NAME
}

pub const fn can_offload_attention_prefill(
    request: GgmlAttentionPrefillRequest,
    caps: DeviceCapabilities,
) -> bool {
    if !matches!(
        caps.backend,
        DeviceBackend::VortexSimx | DeviceBackend::VortexRtl | DeviceBackend::VortexFpga
    ) {
        return false;
    }

    caps.supports_int8
        && tensor_is_dense_i8(request.q)
        && tensor_is_dense_i8(request.k)
        && tensor_is_dense_i8(request.v)
        && tensor_is_dense_i8(request.out)
        && same_shape_dims(request.q, request.k)
        && same_shape_dims(request.q, request.v)
        && same_shape_dims(request.q, request.out)
}

const fn tensor_is_dense_i8(tensor: GgmlTensor2d) -> bool {
    tensor.rows != 0 && tensor.cols != 0 && matches!(tensor.dtype, GgmlDType::I8)
}

const fn same_shape_dims(lhs: GgmlTensor2d, rhs: GgmlTensor2d) -> bool {
    lhs.rows == rhs.rows && lhs.cols == rhs.cols
}

#[cfg(test)]
mod tests {
    use super::{GgmlAttentionPrefillRequest, backend_name};
    use mandrel_target_ir::DeviceCapabilities;

    #[test]
    fn exposes_backend_name() {
        assert_eq!(backend_name(), "mandrel-vortex");
    }

    #[test]
    fn accepts_dense_i8_attention_prefill_request() {
        let request = GgmlAttentionPrefillRequest::dense_i8(64, 64);

        assert!(super::can_offload_attention_prefill(
            request,
            DeviceCapabilities::vortex_simx_default()
        ));
        let op = match request.to_model_ir_for(DeviceCapabilities::vortex_simx_default()) {
            Some(op) => op,
            None => panic!("expected model-ir attention op"),
        };
        assert_eq!(op.shape.sequence, 64);
        assert_eq!(op.shape.head_dim, 64);
    }
}
