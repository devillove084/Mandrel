#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::{string::String, vec::Vec};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactKind {
    KernelIr,
    Mlir,
    LlvmIr,
    Object,
    Elf,
    Vxbin,
    HardwareDesign,
    VortexConfig,
    RtlManifest,
    FpgaBitstream,
    Netlist,
    SynthesisReport,
}

impl ArtifactKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::KernelIr => "kernel_ir",
            Self::Mlir => "mlir",
            Self::LlvmIr => "llvm_ir",
            Self::Object => "object",
            Self::Elf => "elf",
            Self::Vxbin => "vxbin",
            Self::HardwareDesign => "hardware_design",
            Self::VortexConfig => "vortex_config",
            Self::RtlManifest => "rtl_manifest",
            Self::FpgaBitstream => "fpga_bitstream",
            Self::Netlist => "netlist",
            Self::SynthesisReport => "synthesis_report",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DigestAlgorithm {
    Sha256,
}

impl DigestAlgorithm {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sha256 => "sha256",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactDigest {
    pub algorithm: DigestAlgorithm,
    pub value: String,
}

impl ArtifactDigest {
    pub fn sha256(value: impl Into<String>) -> Self {
        Self {
            algorithm: DigestAlgorithm::Sha256,
            value: value.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactRef {
    pub kind: ArtifactKind,
    pub path: String,
    pub digest: Option<ArtifactDigest>,
}

impl ArtifactRef {
    pub fn new(kind: ArtifactKind, path: impl Into<String>) -> Self {
        Self {
            kind,
            path: path.into(),
            digest: None,
        }
    }

    pub fn with_digest(mut self, digest: ArtifactDigest) -> Self {
        self.digest = Some(digest);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ArtifactSet {
    artifacts: Vec<ArtifactRef>,
}

impl ArtifactSet {
    pub const fn empty() -> Self {
        Self {
            artifacts: Vec::new(),
        }
    }

    pub fn with_artifact(mut self, artifact: ArtifactRef) -> Self {
        self.push(artifact);
        self
    }

    pub fn push(&mut self, artifact: ArtifactRef) {
        if let Some(existing) = self
            .artifacts
            .iter_mut()
            .find(|existing| existing.kind == artifact.kind)
        {
            *existing = artifact;
        } else {
            self.artifacts.push(artifact);
        }
    }

    pub fn find(&self, kind: ArtifactKind) -> Option<&ArtifactRef> {
        self.artifacts.iter().find(|artifact| artifact.kind == kind)
    }

    pub fn iter(&self) -> impl Iterator<Item = &ArtifactRef> {
        self.artifacts.iter()
    }

    pub fn len(&self) -> usize {
        self.artifacts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.artifacts.is_empty()
    }
}

#[cfg(feature = "std")]
mod vortex {
    use std::path::{Path, PathBuf};

    use mandrel_kernel_ir::KernelSymbol;

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

        pub fn with_kernel_artifact(
            mut self,
            symbol: KernelSymbol,
            vxbin: impl Into<PathBuf>,
        ) -> Self {
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

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct VortexMlirKernelArtifacts {
        pub mlir_path: PathBuf,
        pub ll_path: PathBuf,
        pub obj_path: PathBuf,
        pub startup_probe_elf_path: PathBuf,
        pub startup_object_path: PathBuf,
        pub elf_path: PathBuf,
        pub vxbin_path: PathBuf,
    }

    impl VortexMlirKernelArtifacts {
        pub fn under_output_dir(out_dir: &Path, symbol_name: &str) -> Self {
            Self {
                mlir_path: out_dir.join(format!("{symbol_name}.mlir")),
                ll_path: out_dir.join(format!("{symbol_name}.ll")),
                obj_path: out_dir.join(format!("{symbol_name}.o")),
                startup_probe_elf_path: out_dir.join(format!("{symbol_name}.startup_probe.elf")),
                startup_object_path: out_dir.join(format!("{symbol_name}.vx_start.o")),
                elf_path: out_dir.join(format!("{symbol_name}.elf")),
                vxbin_path: out_dir.join(format!("{symbol_name}.vxbin")),
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{VortexArtifactRegistry, VortexMlirKernelArtifacts};
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

        #[test]
        fn artifact_paths_follow_kernel_symbol() {
            let out_dir = Path::new("target/mandrel/vortex");
            let artifacts =
                VortexMlirKernelArtifacts::under_output_dir(out_dir, "attention_prefill_i8");

            assert_eq!(
                artifacts.vxbin_path,
                out_dir.join("attention_prefill_i8.vxbin")
            );
        }
    }
}

#[cfg(feature = "std")]
pub use vortex::*;

#[cfg(test)]
mod tests {
    use super::{ArtifactDigest, ArtifactKind, ArtifactRef, ArtifactSet};

    #[test]
    fn artifact_set_replaces_artifacts_by_kind_and_preserves_identity() {
        let mut artifacts = ArtifactSet::empty();
        artifacts.push(ArtifactRef::new(ArtifactKind::Mlir, "old.mlir"));
        artifacts.push(
            ArtifactRef::new(ArtifactKind::Mlir, "kernel.mlir")
                .with_digest(ArtifactDigest::sha256("abc123")),
        );

        assert_eq!(artifacts.len(), 1);
        let mlir = artifacts.find(ArtifactKind::Mlir);
        assert_eq!(
            mlir.map(|artifact| artifact.path.as_str()),
            Some("kernel.mlir")
        );
        assert_eq!(
            mlir.and_then(|artifact| artifact.digest.as_ref())
                .map(|digest| digest.value.as_str()),
            Some("abc123")
        );
    }
}
