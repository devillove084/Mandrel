use mandrel_core::ElementType;

use crate::symbol::KernelSymbol;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelDomain {
    Elementwise,
    Matmul,
    Attention,
    Reduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelImplementation {
    VortexReferenceExample,
    HandWrittenVortexCpp,
    GeneratedVortexCpp,
    GeneratedVortexLlvmIr,
    FutureRustDsl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelAvailability {
    Available,
    Planned,
}

impl KernelAvailability {
    pub const fn is_available(self) -> bool {
        matches!(self, Self::Available)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelSignature {
    VecAdd {
        element_type: ElementType,
    },
    Matmul {
        lhs: ElementType,
        rhs: ElementType,
        out: ElementType,
    },
    AttentionPrefill {
        element_type: ElementType,
    },
    Softmax {
        element_type: ElementType,
    },
}

impl KernelSignature {
    pub const fn is_matmul_i8_i8_i32(self) -> bool {
        matches!(
            self,
            Self::Matmul {
                lhs: ElementType::I8,
                rhs: ElementType::I8,
                out: ElementType::I32,
            }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelSpec {
    pub symbol: KernelSymbol,
    pub domain: KernelDomain,
    pub implementation: KernelImplementation,
    pub availability: KernelAvailability,
    pub signature: KernelSignature,
}

impl KernelSpec {
    pub const fn new(
        symbol: KernelSymbol,
        domain: KernelDomain,
        implementation: KernelImplementation,
        availability: KernelAvailability,
        signature: KernelSignature,
    ) -> Self {
        Self {
            symbol,
            domain,
            implementation,
            availability,
            signature,
        }
    }

    pub const fn is_available(self) -> bool {
        self.availability.is_available()
    }
}

/// Small catalog of kernels we currently expose or plan to expose through the Rust backend.
///
/// `Available` means the project has a concrete backend path for the symbol today. `Planned`
/// entries are intentionally visible so the compiler/scheduler can reason about future lowering
/// work without accidentally selecting an unavailable kernel.
pub const VORTEX_KERNEL_CATALOG: [KernelSpec; 5] = [
    KernelSpec::new(
        KernelSymbol::VecAddF32,
        KernelDomain::Elementwise,
        KernelImplementation::VortexReferenceExample,
        KernelAvailability::Available,
        KernelSignature::VecAdd {
            element_type: ElementType::F32,
        },
    ),
    KernelSpec::new(
        KernelSymbol::MatmulI8I32,
        KernelDomain::Matmul,
        KernelImplementation::GeneratedVortexCpp,
        KernelAvailability::Available,
        KernelSignature::Matmul {
            lhs: ElementType::I8,
            rhs: ElementType::I8,
            out: ElementType::I32,
        },
    ),
    KernelSpec::new(
        KernelSymbol::MatmulI8I32Tiled,
        KernelDomain::Matmul,
        KernelImplementation::GeneratedVortexLlvmIr,
        KernelAvailability::Planned,
        KernelSignature::Matmul {
            lhs: ElementType::I8,
            rhs: ElementType::I8,
            out: ElementType::I32,
        },
    ),
    KernelSpec::new(
        KernelSymbol::AttentionPrefillI8,
        KernelDomain::Attention,
        KernelImplementation::GeneratedVortexCpp,
        KernelAvailability::Planned,
        KernelSignature::AttentionPrefill {
            element_type: ElementType::I8,
        },
    ),
    KernelSpec::new(
        KernelSymbol::SoftmaxF32,
        KernelDomain::Reduction,
        KernelImplementation::GeneratedVortexCpp,
        KernelAvailability::Planned,
        KernelSignature::Softmax {
            element_type: ElementType::F32,
        },
    ),
];

pub fn kernel_spec(symbol: KernelSymbol) -> Option<KernelSpec> {
    let mut index = 0;
    while index < VORTEX_KERNEL_CATALOG.len() {
        let spec = VORTEX_KERNEL_CATALOG[index];
        if spec.symbol == symbol {
            return Some(spec);
        }
        index += 1;
    }
    None
}

pub fn available_kernel(symbol: KernelSymbol) -> Option<KernelSpec> {
    let spec = kernel_spec(symbol)?;
    if spec.is_available() {
        Some(spec)
    } else {
        None
    }
}

pub fn first_available_matmul_i8_i8_i32() -> Option<KernelSpec> {
    let mut index = 0;
    while index < VORTEX_KERNEL_CATALOG.len() {
        let spec = VORTEX_KERNEL_CATALOG[index];
        if spec.is_available() && spec.signature.is_matmul_i8_i8_i32() {
            return Some(spec);
        }
        index += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{
        KernelAvailability, KernelImplementation, KernelSignature, KernelSymbol, available_kernel,
        first_available_matmul_i8_i8_i32, kernel_spec,
    };

    #[test]
    fn catalog_marks_current_matmul_available_and_tiled_planned() {
        let direct = match available_kernel(KernelSymbol::MatmulI8I32) {
            Some(spec) => spec,
            None => panic!("expected current matmul kernel to be available"),
        };
        let tiled = match kernel_spec(KernelSymbol::MatmulI8I32Tiled) {
            Some(spec) => spec,
            None => panic!("expected planned tiled matmul spec"),
        };

        assert!(direct.signature.is_matmul_i8_i8_i32());
        assert_eq!(
            tiled.implementation,
            KernelImplementation::GeneratedVortexLlvmIr
        );
        assert_eq!(tiled.availability, KernelAvailability::Planned);
        assert!(available_kernel(KernelSymbol::MatmulI8I32Tiled).is_none());
    }

    #[test]
    fn finds_first_available_i8_matmul() {
        let spec = match first_available_matmul_i8_i8_i32() {
            Some(spec) => spec,
            None => panic!("expected available i8 matmul"),
        };

        assert_eq!(spec.symbol, KernelSymbol::MatmulI8I32);
        assert!(matches!(spec.signature, KernelSignature::Matmul { .. }));
    }
}
