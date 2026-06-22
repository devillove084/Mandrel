#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VortexAxis {
    X,
    Y,
    Z,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VortexCtaCsr {
    Id,
    Size,
    ThreadId(VortexAxis),
    BlockId(VortexAxis),
    BlockDim(VortexAxis),
    GridDim(VortexAxis),
    LocalMemAddr,
}

impl VortexCtaCsr {
    pub const fn encoded(self) -> u32 {
        match self {
            Self::Id => 0xCD0,
            Self::Size => 0xCD2,
            Self::ThreadId(VortexAxis::X) => 0xCD3,
            Self::ThreadId(VortexAxis::Y) => 0xCD4,
            Self::ThreadId(VortexAxis::Z) => 0xCD5,
            Self::BlockId(VortexAxis::X) => 0xCD6,
            Self::BlockId(VortexAxis::Y) => 0xCD7,
            Self::BlockId(VortexAxis::Z) => 0xCD8,
            Self::BlockDim(VortexAxis::X) => 0xCD9,
            Self::BlockDim(VortexAxis::Y) => 0xCDA,
            Self::BlockDim(VortexAxis::Z) => 0xCDB,
            Self::GridDim(VortexAxis::X) => 0xCDC,
            Self::GridDim(VortexAxis::Y) => 0xCDD,
            Self::GridDim(VortexAxis::Z) => 0xCDE,
            Self::LocalMemAddr => 0xCDF,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{VortexAxis, VortexCtaCsr};

    #[test]
    fn cta_csr_numbers_match_generated_vortex_config() {
        assert_eq!(VortexCtaCsr::ThreadId(VortexAxis::X).encoded(), 0xCD3);
        assert_eq!(VortexCtaCsr::ThreadId(VortexAxis::Y).encoded(), 0xCD4);
        assert_eq!(VortexCtaCsr::BlockId(VortexAxis::X).encoded(), 0xCD6);
        assert_eq!(VortexCtaCsr::BlockId(VortexAxis::Y).encoded(), 0xCD7);
        assert_eq!(VortexCtaCsr::BlockDim(VortexAxis::X).encoded(), 0xCD9);
        assert_eq!(VortexCtaCsr::BlockDim(VortexAxis::Y).encoded(), 0xCDA);
        assert_eq!(VortexCtaCsr::LocalMemAddr.encoded(), 0xCDF);
    }
}
