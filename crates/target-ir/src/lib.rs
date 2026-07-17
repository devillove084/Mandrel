#![cfg_attr(not(feature = "std"), no_std)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DeviceBackend {
    HostReference,
    VortexSimx,
    VortexRtl,
    VortexFpga,
}

impl DeviceBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HostReference => "host_reference",
            Self::VortexSimx => "vortex_simx",
            Self::VortexRtl => "vortex_rtl",
            Self::VortexFpga => "vortex_fpga",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceCapabilities {
    pub backend: DeviceBackend,
    pub xlen: u32,
    pub max_workgroup_threads: u32,
    pub preferred_subgroup_width: u32,
    pub local_memory_bytes: u32,
    pub supports_int8: bool,
    pub supports_float32: bool,
    pub supports_tensor_cores: bool,
    pub supports_async_copy: bool,
}

impl DeviceCapabilities {
    pub const fn vortex_simx_default() -> Self {
        Self {
            backend: DeviceBackend::VortexSimx,
            xlen: 64,
            max_workgroup_threads: 16,
            preferred_subgroup_width: 4,
            local_memory_bytes: 16 * 1024,
            supports_int8: true,
            supports_float32: true,
            supports_tensor_cores: false,
            supports_async_copy: false,
        }
    }

    pub const fn supports(self, capability: OperationCapability) -> bool {
        match capability {
            OperationCapability::Int8Arithmetic => self.supports_int8,
            OperationCapability::Float32Arithmetic => self.supports_float32,
            OperationCapability::TensorCoreMma => self.supports_tensor_cores,
            OperationCapability::AsyncCopy => self.supports_async_copy,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetSpec {
    pub name: &'static str,
    pub backend: DeviceBackend,
    pub xlen: u32,
    pub max_workgroup_threads: u32,
    pub preferred_subgroup_width: u32,
    pub local_memory_bytes: u32,
    pub supports_int8: bool,
    pub supports_float32: bool,
    pub supports_tensor_cores: bool,
    pub supports_async_copy: bool,
}

impl TargetSpec {
    pub const fn from_device_capabilities(name: &'static str, caps: DeviceCapabilities) -> Self {
        Self {
            name,
            backend: caps.backend,
            xlen: caps.xlen,
            max_workgroup_threads: caps.max_workgroup_threads,
            preferred_subgroup_width: caps.preferred_subgroup_width,
            local_memory_bytes: caps.local_memory_bytes,
            supports_int8: caps.supports_int8,
            supports_float32: caps.supports_float32,
            supports_tensor_cores: caps.supports_tensor_cores,
            supports_async_copy: caps.supports_async_copy,
        }
    }

    pub const fn backend_name(self) -> &'static str {
        self.backend.as_str()
    }

    pub const fn device_capabilities(self) -> DeviceCapabilities {
        DeviceCapabilities {
            backend: self.backend,
            xlen: self.xlen,
            max_workgroup_threads: self.max_workgroup_threads,
            preferred_subgroup_width: self.preferred_subgroup_width,
            local_memory_bytes: self.local_memory_bytes,
            supports_int8: self.supports_int8,
            supports_float32: self.supports_float32,
            supports_tensor_cores: self.supports_tensor_cores,
            supports_async_copy: self.supports_async_copy,
        }
    }

    pub const fn vortex_simx_default() -> Self {
        Self::from_device_capabilities(
            "vortex_simx_default",
            DeviceCapabilities::vortex_simx_default(),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TargetCapability {
    Backend = 0,
    Xlen = 1,
    MaxWorkgroupThreads = 2,
    PreferredSubgroupWidth = 3,
    LocalMemoryBytes = 4,
    Int8 = 5,
    Float32 = 6,
    TensorCores = 7,
    AsyncCopy = 8,
}

impl TargetCapability {
    pub const ALL: [Self; 9] = [
        Self::Backend,
        Self::Xlen,
        Self::MaxWorkgroupThreads,
        Self::PreferredSubgroupWidth,
        Self::LocalMemoryBytes,
        Self::Int8,
        Self::Float32,
        Self::TensorCores,
        Self::AsyncCopy,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Backend => "backend",
            Self::Xlen => "xlen",
            Self::MaxWorkgroupThreads => "max_workgroup_threads",
            Self::PreferredSubgroupWidth => "preferred_subgroup_width",
            Self::LocalMemoryBytes => "local_memory_bytes",
            Self::Int8 => "supports_int8",
            Self::Float32 => "supports_float32",
            Self::TensorCores => "supports_tensor_cores",
            Self::AsyncCopy => "supports_async_copy",
        }
    }

    const fn mask(self) -> u16 {
        1 << (self as u8)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetCompatibility {
    mismatch_mask: u16,
}

impl TargetCompatibility {
    pub const fn exact(requested: TargetSpec, observed: TargetSpec) -> Self {
        let mut mismatch_mask = 0;
        if requested.backend as u8 != observed.backend as u8 {
            mismatch_mask |= TargetCapability::Backend.mask();
        }
        if requested.xlen != observed.xlen {
            mismatch_mask |= TargetCapability::Xlen.mask();
        }
        if requested.max_workgroup_threads != observed.max_workgroup_threads {
            mismatch_mask |= TargetCapability::MaxWorkgroupThreads.mask();
        }
        if requested.preferred_subgroup_width != observed.preferred_subgroup_width {
            mismatch_mask |= TargetCapability::PreferredSubgroupWidth.mask();
        }
        if requested.local_memory_bytes != observed.local_memory_bytes {
            mismatch_mask |= TargetCapability::LocalMemoryBytes.mask();
        }
        if requested.supports_int8 != observed.supports_int8 {
            mismatch_mask |= TargetCapability::Int8.mask();
        }
        if requested.supports_float32 != observed.supports_float32 {
            mismatch_mask |= TargetCapability::Float32.mask();
        }
        if requested.supports_tensor_cores != observed.supports_tensor_cores {
            mismatch_mask |= TargetCapability::TensorCores.mask();
        }
        if requested.supports_async_copy != observed.supports_async_copy {
            mismatch_mask |= TargetCapability::AsyncCopy.mask();
        }
        Self { mismatch_mask }
    }

    pub const fn is_compatible(self) -> bool {
        self.mismatch_mask == 0
    }

    pub const fn mismatch_mask(self) -> u16 {
        self.mismatch_mask
    }

    pub const fn mismatch_count(self) -> u32 {
        self.mismatch_mask.count_ones()
    }

    pub const fn mismatches(self, capability: TargetCapability) -> bool {
        self.mismatch_mask & capability.mask() != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetContract {
    pub requested: TargetSpec,
    pub observed: TargetSpec,
    pub compatibility: TargetCompatibility,
}

impl TargetContract {
    pub const fn exact(requested: TargetSpec, observed: TargetSpec) -> Self {
        Self {
            requested,
            observed,
            compatibility: TargetCompatibility::exact(requested, observed),
        }
    }

    pub const fn is_compatible(self) -> bool {
        self.compatibility.is_compatible()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetConstraints {
    pub target: DeviceBackend,
    pub max_workgroup_threads: u32,
    pub local_memory_bytes: u32,
    pub preferred_subgroup_width: u32,
    pub supports_int8: bool,
}

impl TargetConstraints {
    pub const fn from_device_capabilities(caps: DeviceCapabilities) -> Self {
        Self {
            target: caps.backend,
            max_workgroup_threads: caps.max_workgroup_threads,
            local_memory_bytes: caps.local_memory_bytes,
            preferred_subgroup_width: caps.preferred_subgroup_width,
            supports_int8: caps.supports_int8,
        }
    }

    pub const fn vortex_simx_default() -> Self {
        Self::from_device_capabilities(DeviceCapabilities::vortex_simx_default())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OperationCapability {
    Int8Arithmetic = 0,
    Float32Arithmetic = 1,
    TensorCoreMma = 2,
    AsyncCopy = 3,
}

impl OperationCapability {
    pub const ALL: [Self; 4] = [
        Self::Int8Arithmetic,
        Self::Float32Arithmetic,
        Self::TensorCoreMma,
        Self::AsyncCopy,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Int8Arithmetic => "int8_arithmetic",
            Self::Float32Arithmetic => "float32_arithmetic",
            Self::TensorCoreMma => "tensor_core_mma",
            Self::AsyncCopy => "async_copy",
        }
    }

    const fn mask(self) -> u16 {
        1 << (self as u8)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OperationCapabilities {
    mask: u16,
}

impl OperationCapabilities {
    pub const fn empty() -> Self {
        Self { mask: 0 }
    }

    pub const fn of(capability: OperationCapability) -> Self {
        Self {
            mask: capability.mask(),
        }
    }

    pub const fn with(mut self, capability: OperationCapability) -> Self {
        self.mask |= capability.mask();
        self
    }

    pub const fn contains(self, capability: OperationCapability) -> bool {
        self.mask & capability.mask() != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelRequirements {
    pub minimum_xlen: u32,
    pub workgroup_threads: u32,
    pub local_memory_bytes: u32,
    pub operations: OperationCapabilities,
}

impl KernelRequirements {
    pub const fn new(
        minimum_xlen: u32,
        workgroup_threads: u32,
        local_memory_bytes: u32,
        operations: OperationCapabilities,
    ) -> Self {
        Self {
            minimum_xlen,
            workgroup_threads,
            local_memory_bytes,
            operations,
        }
    }

    pub const fn scalar_attention_prefill_i8(workgroup_threads: u32) -> Self {
        Self::new(
            32,
            workgroup_threads,
            0,
            OperationCapabilities::of(OperationCapability::Int8Arithmetic)
                .with(OperationCapability::Float32Arithmetic),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum KernelRequirement {
    Xlen = 0,
    WorkgroupThreads = 1,
    LocalMemory = 2,
    Int8Arithmetic = 3,
    Float32Arithmetic = 4,
    TensorCoreMma = 5,
    AsyncCopy = 6,
}

impl KernelRequirement {
    pub const ALL: [Self; 7] = [
        Self::Xlen,
        Self::WorkgroupThreads,
        Self::LocalMemory,
        Self::Int8Arithmetic,
        Self::Float32Arithmetic,
        Self::TensorCoreMma,
        Self::AsyncCopy,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Xlen => "minimum_xlen",
            Self::WorkgroupThreads => "workgroup_threads",
            Self::LocalMemory => "local_memory_bytes",
            Self::Int8Arithmetic => "int8_arithmetic",
            Self::Float32Arithmetic => "float32_arithmetic",
            Self::TensorCoreMma => "tensor_core_mma",
            Self::AsyncCopy => "async_copy",
        }
    }

    const fn mask(self) -> u16 {
        1 << (self as u8)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequirementCompatibility {
    unsatisfied_mask: u16,
}

impl RequirementCompatibility {
    pub const fn evaluate(requirements: KernelRequirements, target: TargetSpec) -> Self {
        let mut unsatisfied_mask = 0;
        if target.xlen < requirements.minimum_xlen {
            unsatisfied_mask |= KernelRequirement::Xlen.mask();
        }
        if target.max_workgroup_threads < requirements.workgroup_threads {
            unsatisfied_mask |= KernelRequirement::WorkgroupThreads.mask();
        }
        if target.local_memory_bytes < requirements.local_memory_bytes {
            unsatisfied_mask |= KernelRequirement::LocalMemory.mask();
        }
        if requirements
            .operations
            .contains(OperationCapability::Int8Arithmetic)
            && !target.supports_int8
        {
            unsatisfied_mask |= KernelRequirement::Int8Arithmetic.mask();
        }
        if requirements
            .operations
            .contains(OperationCapability::Float32Arithmetic)
            && !target.supports_float32
        {
            unsatisfied_mask |= KernelRequirement::Float32Arithmetic.mask();
        }
        if requirements
            .operations
            .contains(OperationCapability::TensorCoreMma)
            && !target.supports_tensor_cores
        {
            unsatisfied_mask |= KernelRequirement::TensorCoreMma.mask();
        }
        if requirements
            .operations
            .contains(OperationCapability::AsyncCopy)
            && !target.supports_async_copy
        {
            unsatisfied_mask |= KernelRequirement::AsyncCopy.mask();
        }
        Self { unsatisfied_mask }
    }

    pub const fn is_compatible(self) -> bool {
        self.unsatisfied_mask == 0
    }

    pub const fn unsatisfied_mask(self) -> u16 {
        self.unsatisfied_mask
    }

    pub const fn unsatisfied(self, requirement: KernelRequirement) -> bool {
        self.unsatisfied_mask & requirement.mask() != 0
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DeviceCapabilities, KernelRequirement, KernelRequirements, OperationCapabilities,
        OperationCapability, RequirementCompatibility, TargetCapability, TargetCompatibility,
        TargetSpec,
    };

    #[test]
    fn default_vortex_caps_match_current_simx_config() {
        let caps = DeviceCapabilities::vortex_simx_default();

        assert_eq!(caps.xlen, 64);
        assert!(caps.supports_int8);
        assert_eq!(caps.max_workgroup_threads, 16);
        assert_eq!(caps.preferred_subgroup_width, 4);
        assert_eq!(caps.local_memory_bytes, 16 * 1024);
        assert!(!caps.supports_tensor_cores);
        assert!(!caps.supports_async_copy);
    }

    #[test]
    fn exact_identity_and_kernel_requirements_are_separate_contracts() {
        let requested = TargetSpec::vortex_simx_default();
        let mut realized = requested;
        realized.name = "larger_vortex";
        realized.max_workgroup_threads = 32;

        let identity = TargetCompatibility::exact(requested, realized);
        let requirements = RequirementCompatibility::evaluate(
            KernelRequirements::scalar_attention_prefill_i8(16),
            realized,
        );

        assert!(identity.mismatches(TargetCapability::MaxWorkgroupThreads));
        assert!(requirements.is_compatible());
    }

    #[test]
    fn reports_missing_hardware_operations_without_requiring_exact_identity() {
        let requirements = KernelRequirements::new(
            64,
            16,
            1024,
            OperationCapabilities::of(OperationCapability::TensorCoreMma)
                .with(OperationCapability::AsyncCopy),
        );
        let compatibility =
            RequirementCompatibility::evaluate(requirements, TargetSpec::vortex_simx_default());

        assert!(!compatibility.is_compatible());
        assert!(compatibility.unsatisfied(KernelRequirement::TensorCoreMma));
        assert!(compatibility.unsatisfied(KernelRequirement::AsyncCopy));
    }
}
