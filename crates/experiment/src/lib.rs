#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::{string::String, vec::Vec};
use mandrel_device::MemorySpace;
use mandrel_hardware::HardwareDesignSpec;
use mandrel_profiler::{
    KernelCounterTrace, KernelLaunchTrace, RuntimeTraceSummary, TransferDirection,
};
use mandrel_target_ir::{DeviceCapabilities, TargetContract, TargetSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadPhase {
    Prefill,
    Decode,
    Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadKind {
    AttentionPrefillI8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionWorkloadShape {
    pub batch_size: usize,
    pub sequence: usize,
    pub head_dim: usize,
    pub query_heads: Option<usize>,
    pub kv_heads: Option<usize>,
}

impl AttentionWorkloadShape {
    pub const fn single_head_prefill(sequence: usize, head_dim: usize) -> Self {
        Self {
            batch_size: 1,
            sequence,
            head_dim,
            query_heads: None,
            kv_heads: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkloadSpec {
    pub name: &'static str,
    pub kind: WorkloadKind,
    pub phase: WorkloadPhase,
    pub attention: AttentionWorkloadShape,
}

impl WorkloadSpec {
    pub const fn attention_prefill_i8_smoke(sequence: usize, head_dim: usize) -> Self {
        Self {
            name: "attention_prefill_i8_smoke",
            kind: WorkloadKind::AttentionPrefillI8,
            phase: WorkloadPhase::Prefill,
            attention: AttentionWorkloadShape::single_head_prefill(sequence, head_dim),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinkSpec {
    pub bandwidth_bytes_per_second: Option<u64>,
    pub latency_nanoseconds: Option<u64>,
}

impl LinkSpec {
    pub const fn unknown() -> Self {
        Self {
            bandwidth_bytes_per_second: None,
            latency_nanoseconds: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemorySystemSpec {
    pub local_memory_bytes: u32,
    pub host_device_link: LinkSpec,
}

impl MemorySystemSpec {
    pub const fn vortex_rtl_default() -> Self {
        Self {
            local_memory_bytes: DeviceCapabilities::vortex_rtl_default().local_memory_bytes,
            host_device_link: LinkSpec::unknown(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectnessPolicy {
    Exact,
    AbsoluteTolerance { max_abs_error_milliunits: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorrectnessResult {
    pub policy: CorrectnessPolicy,
    pub passed: bool,
    pub compared_elements: usize,
    pub mismatches: usize,
}

impl CorrectnessResult {
    pub const fn exact(compared_elements: usize, mismatches: usize) -> Self {
        Self {
            policy: CorrectnessPolicy::Exact,
            passed: mismatches == 0,
            compared_elements,
            mismatches,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferRole {
    Query,
    Key,
    Value,
    Output,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncKind {
    KernelCompletion,
    ReadbackCompletion,
    QueueFence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeEvent {
    Allocate {
        role: BufferRole,
        memory: MemorySpace,
        bytes: usize,
    },
    Copy {
        direction: TransferDirection,
        bytes: usize,
    },
    KernelLaunch(KernelLaunchTrace),
    Sync {
        kind: SyncKind,
    },
    PerfCounter(KernelCounterTrace),
    CorrectnessCheck(CorrectnessResult),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CounterSet {
    pub kernel: KernelCounterTrace,
}

impl CounterSet {
    pub const fn empty() -> Self {
        Self {
            kernel: KernelCounterTrace::empty(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DerivedMetrics {
    pub total_transfer_bytes: usize,
}

impl DerivedMetrics {
    pub const fn empty() -> Self {
        Self {
            total_transfer_bytes: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExperimentSpec {
    pub id: &'static str,
    pub workload: WorkloadSpec,
    pub hardware: HardwareDesignSpec,
    pub target: TargetSpec,
    pub memory: MemorySystemSpec,
    pub correctness: CorrectnessPolicy,
}

impl ExperimentSpec {
    pub const fn attention_prefill_i8_vortex_smoke(sequence: usize, head_dim: usize) -> Self {
        let hardware = HardwareDesignSpec::current_vortex_default();
        Self {
            id: "attention_prefill_i8_vortex_smoke",
            workload: WorkloadSpec::attention_prefill_i8_smoke(sequence, head_dim),
            hardware,
            target: hardware.current_vortex_rtl_target(),
            memory: MemorySystemSpec::vortex_rtl_default(),
            correctness: CorrectnessPolicy::Exact,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftwareOutputKind {
    Mlir,
    LlvmIr,
    Object,
    Elf,
    Vxbin,
}

impl SoftwareOutputKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mlir => "mlir",
            Self::LlvmIr => "llvm_ir",
            Self::Object => "object",
            Self::Elf => "elf",
            Self::Vxbin => "vxbin",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoftwareOutputRef {
    pub kind: SoftwareOutputKind,
    pub path: String,
}

impl SoftwareOutputRef {
    pub fn new(kind: SoftwareOutputKind, path: impl Into<String>) -> Self {
        Self {
            kind,
            path: path.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentStatus {
    Planned,
    Unsupported,
    Succeeded,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExperimentResult {
    pub spec_id: &'static str,
    pub status: ExperimentStatus,
    pub target: TargetContract,
    pub software_outputs: Vec<SoftwareOutputRef>,
    pub counters: CounterSet,
    pub events: Vec<RuntimeEvent>,
    pub correctness: Option<CorrectnessResult>,
    pub derived_metrics: DerivedMetrics,
}

impl ExperimentResult {
    pub fn from_runtime_trace_summary(
        spec: ExperimentSpec,
        observed_target: TargetSpec,
        summary: RuntimeTraceSummary,
        correctness: CorrectnessResult,
    ) -> Self {
        let target = TargetContract::exact(spec.target, observed_target);
        let mut events = events_from_runtime_trace_summary(summary);
        events.push(RuntimeEvent::CorrectnessCheck(correctness));
        Self {
            spec_id: spec.id,
            status: if !target.is_compatible() {
                ExperimentStatus::Unsupported
            } else if correctness.passed {
                ExperimentStatus::Succeeded
            } else {
                ExperimentStatus::Failed
            },
            target,
            software_outputs: Vec::new(),
            counters: CounterSet {
                kernel: summary.counters,
            },
            events,
            correctness: Some(correctness),
            derived_metrics: DerivedMetrics {
                total_transfer_bytes: summary.total_transfer_bytes(),
            },
        }
    }
}

pub fn events_from_runtime_trace_summary(summary: RuntimeTraceSummary) -> Vec<RuntimeEvent> {
    let mut events = Vec::new();
    if summary.host_to_device_bytes != 0 {
        events.push(RuntimeEvent::Copy {
            direction: TransferDirection::HostToDevice,
            bytes: summary.host_to_device_bytes,
        });
    }
    events.push(RuntimeEvent::KernelLaunch(summary.launch));
    events.push(RuntimeEvent::Sync {
        kind: SyncKind::KernelCompletion,
    });
    if summary.device_to_host_bytes != 0 {
        events.push(RuntimeEvent::Copy {
            direction: TransferDirection::DeviceToHost,
            bytes: summary.device_to_host_bytes,
        });
    }
    if summary.counters != KernelCounterTrace::empty() {
        events.push(RuntimeEvent::PerfCounter(summary.counters));
    }
    events
}

#[cfg(test)]
mod tests {
    use super::{
        CorrectnessResult, ExperimentResult, ExperimentSpec, ExperimentStatus, HardwareDesignSpec,
        RuntimeEvent, SyncKind, TargetContract, TargetSpec, events_from_runtime_trace_summary,
    };
    use mandrel_profiler::{KernelCounterTrace, KernelLaunchTrace, RuntimeTraceSummary};
    use mandrel_target_ir::{TargetCapability, TargetCompatibility};

    #[test]
    fn builds_attention_prefill_vortex_smoke_spec() {
        let spec = ExperimentSpec::attention_prefill_i8_vortex_smoke(64, 64);

        assert_eq!(spec.id, "attention_prefill_i8_vortex_smoke");
        assert_eq!(spec.workload.attention.sequence, 64);
        assert_eq!(spec.workload.attention.head_dim, 64);
        assert_eq!(spec.hardware, HardwareDesignSpec::current_vortex_default());
        assert_eq!(spec.target, spec.hardware.current_vortex_rtl_target());
    }

    #[test]
    fn exact_target_contract_reports_capability_mismatches() {
        let requested = TargetSpec::vortex_rtl_default();
        let mut observed = requested;
        observed.name = "vortex_rtl_observed";
        observed.max_workgroup_threads = 8;
        observed.local_memory_bytes = 8 * 1024;
        let contract = TargetContract::exact(requested, observed);

        assert!(!contract.is_compatible());
        assert_eq!(contract.compatibility.mismatch_count(), 2);
        assert!(
            contract
                .compatibility
                .mismatches(TargetCapability::MaxWorkgroupThreads)
        );
        assert!(
            contract
                .compatibility
                .mismatches(TargetCapability::LocalMemoryBytes)
        );
        assert_eq!(
            contract.compatibility.mismatch_mask(),
            TargetCompatibility::exact(requested, observed).mismatch_mask()
        );
    }

    #[test]
    fn converts_runtime_trace_summary_to_events() {
        let summary = RuntimeTraceSummary::new(
            KernelLaunchTrace::new("attention_prefill_i8", [16, 1, 1], [4, 4, 1], 2336),
            384,
            128,
            KernelCounterTrace::new(Some(165_144), Some(414_598)),
        );
        let events = events_from_runtime_trace_summary(summary);

        assert_eq!(events.len(), 5);
        assert!(matches!(events[1], RuntimeEvent::KernelLaunch(_)));
        assert!(matches!(
            events[2],
            RuntimeEvent::Sync {
                kind: SyncKind::KernelCompletion
            }
        ));
        assert!(matches!(events[4], RuntimeEvent::PerfCounter(_)));
    }

    #[test]
    fn builds_experiment_result_from_trace_summary() {
        let spec = ExperimentSpec::attention_prefill_i8_vortex_smoke(8, 16);
        let summary = RuntimeTraceSummary::new(
            KernelLaunchTrace::new("attention_prefill_i8", [2, 1, 1], [4, 4, 1], 2336),
            384,
            128,
            KernelCounterTrace::empty(),
        );
        let correctness = CorrectnessResult::exact(128, 0);
        let result = ExperimentResult::from_runtime_trace_summary(
            spec,
            TargetSpec::vortex_rtl_default(),
            summary,
            correctness,
        );

        assert_eq!(result.spec_id, spec.id);
        assert_eq!(result.status, ExperimentStatus::Succeeded);
        assert!(result.target.is_compatible());
        assert_eq!(result.derived_metrics.total_transfer_bytes, 512);
        assert_eq!(result.correctness, Some(correctness));
        assert!(matches!(
            result.events.last(),
            Some(RuntimeEvent::CorrectnessCheck(_))
        ));
    }

    #[test]
    fn incompatible_observed_target_cannot_produce_succeeded_result() {
        let spec = ExperimentSpec::attention_prefill_i8_vortex_smoke(8, 16);
        let mut observed = TargetSpec::vortex_rtl_default();
        observed.max_workgroup_threads = 8;
        let summary = RuntimeTraceSummary::new(
            KernelLaunchTrace::new("attention_prefill_i8", [2, 1, 1], [4, 4, 1], 0),
            384,
            128,
            KernelCounterTrace::empty(),
        );

        let result = ExperimentResult::from_runtime_trace_summary(
            spec,
            observed,
            summary,
            CorrectnessResult::exact(128, 0),
        );

        assert_eq!(result.status, ExperimentStatus::Unsupported);
        assert!(!result.target.is_compatible());
    }
}
