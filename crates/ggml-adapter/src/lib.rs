#![cfg_attr(not(feature = "std"), no_std)]

use mandrel_device::{DeviceBackend, DeviceCapabilities};
use mandrel_model_ir::{MatmulOp, MatmulShape, MatmulTensors, MatmulTypes};

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
pub struct GgmlMulMatRequest {
    pub lhs: GgmlTensor2d,
    pub rhs: GgmlTensor2d,
    pub out: GgmlTensor2d,
}

impl GgmlMulMatRequest {
    pub const fn i8_i32(m: u32, n: u32, k: u32) -> Self {
        Self {
            lhs: GgmlTensor2d {
                rows: m,
                cols: k,
                dtype: GgmlDType::I8,
            },
            rhs: GgmlTensor2d {
                rows: k,
                cols: n,
                dtype: GgmlDType::I8,
            },
            out: GgmlTensor2d {
                rows: m,
                cols: n,
                dtype: GgmlDType::I32,
            },
        }
    }

    pub fn to_model_ir_for(self, caps: DeviceCapabilities) -> Option<MatmulOp> {
        if !can_offload_mul_mat(self, caps) {
            return None;
        }

        Some(MatmulOp::new(
            MatmulTensors::new(0, 1, 2),
            MatmulShape::new(
                self.lhs.rows as usize,
                self.rhs.cols as usize,
                self.lhs.cols as usize,
            ),
            MatmulTypes::i8_to_i32(),
        ))
    }

    pub fn to_model_ir(self) -> Option<MatmulOp> {
        self.to_model_ir_for(DeviceCapabilities::vortex_simx_default())
    }
}

pub const fn backend_name() -> &'static str {
    BACKEND_NAME
}

pub const fn can_offload_mul_mat(request: GgmlMulMatRequest, caps: DeviceCapabilities) -> bool {
    if !matches!(
        caps.backend,
        DeviceBackend::VortexSimx | DeviceBackend::VortexRtl | DeviceBackend::VortexFpga
    ) {
        return false;
    }

    caps.supports_int8
        && matches!(request.lhs.dtype, GgmlDType::I8)
        && matches!(request.rhs.dtype, GgmlDType::I8)
        && matches!(request.out.dtype, GgmlDType::I32)
        && request.lhs.cols == request.rhs.rows
        && request.lhs.rows == request.out.rows
        && request.rhs.cols == request.out.cols
}

#[cfg(test)]
mod tests {
    use super::{GgmlMulMatRequest, backend_name};
    use mandrel_device::DeviceCapabilities;

    #[test]
    fn exposes_backend_name() {
        assert_eq!(backend_name(), "mandrel-vortex");
    }

    #[test]
    fn accepts_i8_mul_mat_request() {
        let request = GgmlMulMatRequest::i8_i32(32, 32, 64);

        assert!(super::can_offload_mul_mat(
            request,
            DeviceCapabilities::vortex_simx_default()
        ));
        let op = match request.to_model_ir_for(DeviceCapabilities::vortex_simx_default()) {
            Some(op) => op,
            None => panic!("expected model-ir matmul op"),
        };
        assert_eq!(op.shape.k, 64);
    }
}
