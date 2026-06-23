use std::path::{Path, PathBuf};

use mandrel_kernel_ir::KernelSymbol;

/// Host-side mapping from a kernel symbol to the `.vxbin` artifact that contains it.
///
/// The backend keeps this separate from scheduling/compilation so experimental kernels can be
/// smoke-tested without making them the default compiler choice.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VortexArtifactRegistry {
    entries: Vec<VortexArtifact>,
}

impl VortexArtifactRegistry {
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn with_kernel_artifact(mut self, symbol: KernelSymbol, vxbin: impl Into<PathBuf>) -> Self {
        self.insert_kernel_artifact(symbol, vxbin);
        self
    }

    pub fn insert_kernel_artifact(&mut self, symbol: KernelSymbol, vxbin: impl Into<PathBuf>) {
        let vxbin = vxbin.into();
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.symbol == symbol) {
            entry.vxbin = vxbin;
            return;
        }
        self.entries.push(VortexArtifact::new(symbol, vxbin));
    }

    pub fn kernel_artifact(&self, symbol: KernelSymbol) -> Option<&Path> {
        self.entries
            .iter()
            .find(|entry| entry.symbol == symbol)
            .map(|entry| entry.vxbin.as_path())
    }

    pub fn entries(&self) -> &[VortexArtifact] {
        &self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VortexArtifact {
    pub symbol: KernelSymbol,
    pub vxbin: PathBuf,
}

impl VortexArtifact {
    pub fn new(symbol: KernelSymbol, vxbin: impl Into<PathBuf>) -> Self {
        Self {
            symbol,
            vxbin: vxbin.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::VortexArtifactRegistry;
    use mandrel_kernel_ir::KernelSymbol;
    use std::path::Path;

    #[test]
    fn registry_returns_artifact_by_symbol() {
        let registry = VortexArtifactRegistry::new().with_kernel_artifact(
            KernelSymbol::AttentionPrefillI8,
            "target/vortex/attention_prefill_i8/kernel.vxbin",
        );

        assert_eq!(
            registry.kernel_artifact(KernelSymbol::AttentionPrefillI8),
            Some(Path::new("target/vortex/attention_prefill_i8/kernel.vxbin"))
        );
        assert!(registry.kernel_artifact(KernelSymbol::SoftmaxF32).is_none());
    }

    #[test]
    fn inserting_existing_symbol_replaces_path() {
        let mut registry = VortexArtifactRegistry::new();
        registry.insert_kernel_artifact(KernelSymbol::AttentionPrefillI8, "old.vxbin");
        registry.insert_kernel_artifact(KernelSymbol::AttentionPrefillI8, "new.vxbin");

        assert_eq!(registry.entries().len(), 1);
        assert_eq!(
            registry.kernel_artifact(KernelSymbol::AttentionPrefillI8),
            Some(Path::new("new.vxbin"))
        );
    }
}
