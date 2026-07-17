#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::string::String;

use mandrel_target_ir::{DeviceBackend, DeviceCapabilities, TargetSpec};

pub const VORTEX_UPSTREAM_URL: &str = "https://github.com/vortexgpgpu/vortex.git";
pub const CURRENT_VORTEX_REVISION: &str = "c992f3f35fa17b83dcc83648a5ac4014b0ea0ac6";
pub const CURRENT_LLVM_VORTEX_REVISION: &str = "b2ecfe086df2d00233a87f3d3ebe6a74aba129e2";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VortexHardwareParameters {
    pub xlen: u32,
    pub clusters: u32,
    pub cores: u32,
    pub warps_per_core: u32,
    pub threads_per_warp: u32,
    pub local_memory_log2_bytes: u32,
    pub dcache_bytes: u32,
    pub tensor_cores: bool,
    pub async_copy: bool,
    pub tensor_core_int8: bool,
}

impl VortexHardwareParameters {
    pub const fn current_default() -> Self {
        Self {
            xlen: 64,
            clusters: 1,
            cores: 1,
            warps_per_core: 4,
            threads_per_warp: 4,
            local_memory_log2_bytes: 14,
            dcache_bytes: 16 * 1024,
            tensor_cores: false,
            async_copy: false,
            tensor_core_int8: true,
        }
    }

    pub const fn local_memory_bytes(self) -> Option<u32> {
        1u32.checked_shl(self.local_memory_log2_bytes)
    }

    pub const fn max_workgroup_threads(self) -> Option<u32> {
        self.warps_per_core.checked_mul(self.threads_per_warp)
    }

    pub const fn device_capabilities(self, backend: DeviceBackend) -> Option<DeviceCapabilities> {
        let Some(max_workgroup_threads) = self.max_workgroup_threads() else {
            return None;
        };
        let Some(local_memory_bytes) = self.local_memory_bytes() else {
            return None;
        };
        Some(DeviceCapabilities {
            backend,
            xlen: self.xlen,
            max_workgroup_threads,
            preferred_subgroup_width: self.threads_per_warp,
            local_memory_bytes,
            supports_int8: true,
            supports_float32: true,
            supports_tensor_cores: self.tensor_cores,
            supports_async_copy: self.async_copy,
        })
    }

    pub const fn target_spec(
        self,
        name: &'static str,
        backend: DeviceBackend,
    ) -> Option<TargetSpec> {
        let Some(capabilities) = self.device_capabilities(backend) else {
            return None;
        };
        Some(TargetSpec::from_device_capabilities(name, capabilities))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HardwareDesignSpec {
    pub id: &'static str,
    pub vortex: VortexHardwareParameters,
}

impl HardwareDesignSpec {
    pub const fn current_vortex_default() -> Self {
        Self {
            id: "vortex_current_default",
            vortex: VortexHardwareParameters::current_default(),
        }
    }

    pub const fn compiler_target(self) -> CompilerTargetSpec {
        CompilerTargetSpec::vortex_rv64()
    }

    pub const fn current_vortex_simx_target(self) -> TargetSpec {
        match self
            .vortex
            .target_spec("vortex_simx_default", DeviceBackend::VortexSimx)
        {
            Some(target) => target,
            None => TargetSpec::vortex_simx_default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompilerTargetSpec {
    pub target_triple: &'static str,
    pub cpu: &'static str,
    pub features: &'static str,
    pub xlen: u32,
}

impl CompilerTargetSpec {
    pub const fn vortex_rv64() -> Self {
        Self {
            target_triple: "riscv64-unknown-elf",
            cpu: "generic-rv64",
            features: "+m,+a,+f,+zicsr,+zifencei,+xvortex",
            xlen: 64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareRealizationKind {
    SimxModel,
    RtlSimulation,
    Fpga,
    Synthesis,
}

impl HardwareRealizationKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SimxModel => "simx_model",
            Self::RtlSimulation => "rtl_simulation",
            Self::Fpga => "fpga",
            Self::Synthesis => "synthesis",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealizedHardwareManifest {
    pub design: HardwareDesignSpec,
    pub realization: HardwareRealizationKind,
    pub source_repository: String,
    pub source_revision: String,
    pub configuration_identity: String,
    pub build_identity: String,
    pub compiler_target: CompilerTargetSpec,
    pub realized_target: TargetSpec,
}

impl RealizedHardwareManifest {
    pub fn current_vortex_simx(
        configuration_identity: impl Into<String>,
        build_identity: impl Into<String>,
    ) -> Self {
        let design = HardwareDesignSpec::current_vortex_default();
        let realized_target = design.current_vortex_simx_target();
        Self {
            design,
            realization: HardwareRealizationKind::SimxModel,
            source_repository: String::from(VORTEX_UPSTREAM_URL),
            source_revision: String::from(CURRENT_VORTEX_REVISION),
            configuration_identity: configuration_identity.into(),
            build_identity: build_identity.into(),
            compiler_target: design.compiler_target(),
            realized_target,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CURRENT_VORTEX_REVISION, HardwareDesignSpec, HardwareRealizationKind,
        RealizedHardwareManifest, VortexHardwareParameters,
    };
    use mandrel_target_ir::{DeviceBackend, DeviceCapabilities, TargetSpec};

    #[test]
    fn tracked_default_matches_current_vortex_configuration() {
        let parameters = VortexHardwareParameters::current_default();
        let target = parameters
            .target_spec("vortex_simx_default", DeviceBackend::VortexSimx)
            .unwrap_or_else(TargetSpec::vortex_simx_default);

        assert_eq!(parameters.clusters, 1);
        assert_eq!(parameters.cores, 1);
        assert_eq!(parameters.dcache_bytes, 16 * 1024);
        assert_eq!(
            target.device_capabilities(),
            DeviceCapabilities::vortex_simx_default()
        );
    }

    #[test]
    fn manifest_binds_design_source_config_build_and_target() {
        let manifest =
            RealizedHardwareManifest::current_vortex_simx("sha256:config", "sha256:simx-build");

        assert_eq!(
            manifest.design,
            HardwareDesignSpec::current_vortex_default()
        );
        assert_eq!(manifest.realization, HardwareRealizationKind::SimxModel);
        assert_eq!(manifest.source_revision, CURRENT_VORTEX_REVISION);
        assert_eq!(manifest.configuration_identity, "sha256:config");
        assert_eq!(manifest.build_identity, "sha256:simx-build");
    }
}
