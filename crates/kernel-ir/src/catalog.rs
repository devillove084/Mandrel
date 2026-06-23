use mandrel_core::ElementType;

use crate::symbol::KernelSymbol;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelDomain {
    Elementwise,
    Attention,
    Reduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelImplementation {
    VortexReferenceExample,
    GeneratedVortexMlir,
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
    VecAdd { element_type: ElementType },
    AttentionPrefill { element_type: ElementType },
    Softmax { element_type: ElementType },
}

impl KernelSignature {
    pub const fn is_attention_prefill_i8(self) -> bool {
        matches!(
            self,
            Self::AttentionPrefill {
                element_type: ElementType::I8,
            }
        )
    }

    pub const fn is_softmax_f32(self) -> bool {
        matches!(
            self,
            Self::Softmax {
                element_type: ElementType::F32,
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
/// work without accidentally claiming an executable kernel.
pub const VORTEX_KERNEL_CATALOG: [KernelSpec; 3] = [
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
        KernelSymbol::AttentionPrefillI8,
        KernelDomain::Attention,
        KernelImplementation::GeneratedVortexMlir,
        KernelAvailability::Planned,
        KernelSignature::AttentionPrefill {
            element_type: ElementType::I8,
        },
    ),
    KernelSpec::new(
        KernelSymbol::SoftmaxF32,
        KernelDomain::Reduction,
        KernelImplementation::GeneratedVortexMlir,
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

#[cfg(test)]
mod tests {
    use super::{
        KernelAvailability, KernelImplementation, KernelSymbol, available_kernel, kernel_spec,
    };

    #[test]
    fn catalog_marks_attention_and_softmax_as_planned_mlir_kernels() {
        let attention = match kernel_spec(KernelSymbol::AttentionPrefillI8) {
            Some(spec) => spec,
            None => panic!("expected planned attention prefill spec"),
        };
        let softmax = match kernel_spec(KernelSymbol::SoftmaxF32) {
            Some(spec) => spec,
            None => panic!("expected planned softmax spec"),
        };

        assert!(attention.signature.is_attention_prefill_i8());
        assert!(softmax.signature.is_softmax_f32());
        assert_eq!(
            attention.implementation,
            KernelImplementation::GeneratedVortexMlir
        );
        assert_eq!(
            softmax.implementation,
            KernelImplementation::GeneratedVortexMlir
        );
        assert_eq!(attention.availability, KernelAvailability::Planned);
        assert!(available_kernel(KernelSymbol::AttentionPrefillI8).is_none());
    }

    #[test]
    fn vecadd_remains_an_available_sdk_reference_symbol() {
        let spec = match available_kernel(KernelSymbol::VecAddF32) {
            Some(spec) => spec,
            None => panic!("expected vecadd SDK example to be available"),
        };

        assert_eq!(spec.symbol, KernelSymbol::VecAddF32);
        assert_eq!(spec.availability, KernelAvailability::Available);
    }
}
