/// Kernel symbols understood by the custom Vortex backend path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelSymbol {
    VecAddF32,
    AttentionPrefillI8,
    SoftmaxF32,
}

impl KernelSymbol {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VecAddF32 => "vecadd_f32",
            Self::AttentionPrefillI8 => "attention_prefill_i8",
            Self::SoftmaxF32 => "softmax_f32",
        }
    }
}
