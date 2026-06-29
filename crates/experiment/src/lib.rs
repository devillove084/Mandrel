#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::{string::String, vec::Vec};

use mandrel_device::{DeviceBackend, DeviceCapabilities, MemorySpace};
use mandrel_profiler::{
    KernelCounterTrace, KernelLaunchTrace, RuntimeTraceSummary, TransferDirection,
};

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
pub struct TargetSpec {
    pub name: &'static str,
    pub backend: DeviceBackend,
    pub xlen: u32,
    pub max_workgroup_threads: u32,
    pub local_memory_bytes: u32,
    pub supports_int8: bool,
    pub supports_float32: bool,
    pub supports_tensor_cores: bool,
}

impl TargetSpec {
    pub const fn from_device_capabilities(name: &'static str, caps: DeviceCapabilities) -> Self {
        Self {
            name,
            backend: caps.backend,
            xlen: caps.xlen,
            max_workgroup_threads: caps.max_workgroup_threads,
            local_memory_bytes: caps.local_memory_bytes,
            supports_int8: caps.supports_int8,
            supports_float32: caps.supports_float32,
            supports_tensor_cores: caps.supports_tensor_cores,
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
    pub const fn vortex_simx_default() -> Self {
        Self {
            local_memory_bytes: DeviceCapabilities::vortex_simx_default().local_memory_bytes,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactSet {
    pub mlir: Option<String>,
    pub llvm_ir: Option<String>,
    pub object: Option<String>,
    pub elf: Option<String>,
    pub vxbin: Option<String>,
}

impl ArtifactSet {
    pub fn empty() -> Self {
        Self {
            mlir: None,
            llvm_ir: None,
            object: None,
            elf: None,
            vxbin: None,
        }
    }
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
    pub target: TargetSpec,
    pub memory: MemorySystemSpec,
    pub correctness: CorrectnessPolicy,
}

impl ExperimentSpec {
    pub const fn attention_prefill_i8_vortex_smoke(sequence: usize, head_dim: usize) -> Self {
        Self {
            id: "attention_prefill_i8_vortex_smoke",
            workload: WorkloadSpec::attention_prefill_i8_smoke(sequence, head_dim),
            target: TargetSpec::vortex_simx_default(),
            memory: MemorySystemSpec::vortex_simx_default(),
            correctness: CorrectnessPolicy::Exact,
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
    pub artifacts: ArtifactSet,
    pub counters: CounterSet,
    pub events: Vec<RuntimeEvent>,
    pub correctness: Option<CorrectnessResult>,
    pub derived_metrics: DerivedMetrics,
}

impl ExperimentResult {
    pub fn from_runtime_trace_summary(
        spec: ExperimentSpec,
        summary: RuntimeTraceSummary,
        correctness: CorrectnessResult,
    ) -> Self {
        let mut events = events_from_runtime_trace_summary(summary);
        events.push(RuntimeEvent::CorrectnessCheck(correctness));
        Self {
            spec_id: spec.id,
            status: if correctness.passed {
                ExperimentStatus::Succeeded
            } else {
                ExperimentStatus::Failed
            },
            artifacts: ArtifactSet::empty(),
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
        CorrectnessResult, ExperimentResult, ExperimentSpec, ExperimentStatus, RuntimeEvent,
        SyncKind, TargetSpec, events_from_runtime_trace_summary,
    };
    use mandrel_profiler::{KernelCounterTrace, KernelLaunchTrace, RuntimeTraceSummary};

    #[test]
    fn builds_attention_prefill_vortex_smoke_spec() {
        let spec = ExperimentSpec::attention_prefill_i8_vortex_smoke(64, 64);

        assert_eq!(spec.id, "attention_prefill_i8_vortex_smoke");
        assert_eq!(spec.workload.attention.sequence, 64);
        assert_eq!(spec.workload.attention.head_dim, 64);
        assert_eq!(spec.target, TargetSpec::vortex_simx_default());
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
        let result = ExperimentResult::from_runtime_trace_summary(spec, summary, correctness);

        assert_eq!(result.spec_id, spec.id);
        assert_eq!(result.status, ExperimentStatus::Succeeded);
        assert_eq!(result.derived_metrics.total_transfer_bytes, 512);
        assert_eq!(result.correctness, Some(correctness));
        assert!(matches!(
            result.events.last(),
            Some(RuntimeEvent::CorrectnessCheck(_))
        ));
    }
}
