/// Kernel symbols understood by the custom Vortex backend path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelSymbol {
    VecAddF32,
    MatmulI8I32,
    MatmulI8I32Tiled,
    AttentionPrefillI8,
    SoftmaxF32,
}

impl KernelSymbol {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VecAddF32 => "vecadd_f32",
            Self::MatmulI8I32 => "matmul_i8_i32",
            Self::MatmulI8I32Tiled => "matmul_i8_i32_tiled",
            Self::AttentionPrefillI8 => "attention_prefill_i8",
            Self::SoftmaxF32 => "softmax_f32",
        }
    }
}
