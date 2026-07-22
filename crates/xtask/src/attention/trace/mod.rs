use mandrel_experiment::{CorrectnessResult, ExperimentResult, ExperimentSpec};
use mandrel_profiler::{KernelCounterTrace, RuntimeTraceSummary};
use mandrel_target_ir::{DeviceBackend, TargetContract, TargetSpec};
use tracing::warn;

mod collect;
mod print;

pub(crate) use collect::collect_attention_runtime_trace_with_enrichment;
pub(crate) use print::print_attention_runtime_trace_report;

pub(crate) const ATTENTION_RUNTIME_TRACE_KERNEL: &str = "attention_prefill_i8";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct AttentionRuntimeTraceEnrichment {
    pub(crate) wall_time_ms: Option<u64>,
    pub(crate) experiment_spec: Option<ExperimentSpec>,
}

impl AttentionRuntimeTraceEnrichment {
    pub(crate) const fn with_experiment_spec(
        wall_time_ms: u64,
        experiment_spec: ExperimentSpec,
    ) -> Self {
        Self {
            wall_time_ms: Some(wall_time_ms),
            experiment_spec: Some(experiment_spec),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct AttentionRuntimeTraceMetadata {
    pub(crate) runtime_sequence: Option<u32>,
    pub(crate) runtime_head_dim: Option<u32>,
    pub(crate) query_tile: Option<u32>,
    pub(crate) key_tile: Option<u32>,
    pub(crate) compiled_sequence: Option<u32>,
    pub(crate) compiled_head_dim: Option<u32>,
    pub(crate) head_dim_tile: Option<u32>,
    pub(crate) logical_macs: Option<u64>,
    pub(crate) lowered_macs: Option<u64>,
    pub(crate) estimated_global_bytes_read: Option<u64>,
    pub(crate) estimated_global_bytes_written: Option<u64>,
    pub(crate) estimated_local_memory_bytes_per_workgroup: Option<u64>,
    pub(crate) requested_target_backend: Option<DeviceBackend>,
    pub(crate) requested_target_xlen: Option<u32>,
    pub(crate) requested_target_max_workgroup_threads: Option<u32>,
    pub(crate) requested_target_preferred_subgroup_width: Option<u32>,
    pub(crate) requested_target_local_memory_bytes: Option<u32>,
    pub(crate) requested_target_supports_int8: Option<bool>,
    pub(crate) requested_target_supports_float32: Option<bool>,
    pub(crate) requested_target_supports_tensor_cores: Option<bool>,
    pub(crate) requested_target_supports_async_copy: Option<bool>,
    pub(crate) observed_target_backend: Option<DeviceBackend>,
    pub(crate) observed_target_xlen: Option<u32>,
    pub(crate) observed_target_preferred_subgroup_width: Option<u32>,
    pub(crate) observed_target_supports_int8: Option<bool>,
    pub(crate) observed_target_supports_float32: Option<bool>,
    pub(crate) observed_target_supports_tensor_cores: Option<bool>,
    pub(crate) observed_target_supports_async_copy: Option<bool>,
    pub(crate) target_compatible: Option<bool>,
    pub(crate) target_mismatch_mask: Option<u16>,
    pub(crate) target_threads_per_warp: Option<u64>,
    pub(crate) target_warps_per_core: Option<u64>,
    pub(crate) target_max_workgroup_threads: Option<u64>,
    pub(crate) target_local_memory_bytes: Option<u64>,
    pub(crate) workgroup_count: Option<u64>,
    pub(crate) threads_per_workgroup: Option<u64>,
    pub(crate) total_threads: Option<u64>,
    pub(crate) runtime_q_elements: Option<u64>,
    pub(crate) runtime_kv_elements: Option<u64>,
    pub(crate) runtime_output_elements: Option<u64>,
    pub(crate) runtime_q_bytes: Option<u64>,
    pub(crate) runtime_kv_bytes: Option<u64>,
    pub(crate) runtime_output_bytes: Option<u64>,
    pub(crate) module_cache_hit: Option<bool>,
    pub(crate) kernel_cache_hit: Option<bool>,
    pub(crate) wall_time_ms: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttentionRuntimeTraceRecord {
    pub(crate) metadata: AttentionRuntimeTraceMetadata,
    pub(crate) summary: RuntimeTraceSummary,
    pub(crate) correctness: Option<CorrectnessResult>,
    pub(crate) experiment_spec: Option<ExperimentSpec>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttentionRuntimeTraceReport {
    pub(crate) record: Option<AttentionRuntimeTraceRecord>,
    pub(crate) counters: KernelCounterTrace,
    pub(crate) metadata: AttentionRuntimeTraceMetadata,
}

pub(crate) fn attention_experiment_result_from_record(
    record: AttentionRuntimeTraceRecord,
) -> Option<ExperimentResult> {
    let sequence = record
        .metadata
        .runtime_sequence
        .or(record.metadata.compiled_sequence)? as usize;
    let head_dim = record
        .metadata
        .runtime_head_dim
        .or(record.metadata.compiled_head_dim)? as usize;
    let correctness = record.correctness?;
    let target = attention_target_contract_from_record(record)?;
    let spec = match record.experiment_spec {
        Some(spec) => {
            if spec.workload.attention.sequence != sequence
                || spec.workload.attention.head_dim != head_dim
                || spec.target.device_capabilities() != target.requested.device_capabilities()
            {
                return None;
            }
            spec
        }
        None => {
            let mut reconstructed =
                ExperimentSpec::attention_prefill_i8_vortex_smoke(sequence, head_dim);
            reconstructed.target = target.requested;
            reconstructed
        }
    };
    Some(ExperimentResult::from_runtime_trace_summary(
        spec,
        target.observed,
        record.summary,
        correctness,
    ))
}

fn attention_target_contract_from_record(
    record: AttentionRuntimeTraceRecord,
) -> Option<TargetContract> {
    let metadata = record.metadata;
    let mut requested = TargetSpec::vortex_rtl_default();
    requested.name = "attention_trace_requested_target";
    requested.backend = metadata
        .requested_target_backend
        .unwrap_or(requested.backend);
    requested.xlen = metadata.requested_target_xlen.unwrap_or(requested.xlen);
    requested.max_workgroup_threads = metadata
        .requested_target_max_workgroup_threads
        .unwrap_or(requested.max_workgroup_threads);
    requested.preferred_subgroup_width = metadata
        .requested_target_preferred_subgroup_width
        .unwrap_or(requested.preferred_subgroup_width);
    requested.local_memory_bytes = metadata
        .requested_target_local_memory_bytes
        .unwrap_or(requested.local_memory_bytes);
    requested.supports_int8 = metadata
        .requested_target_supports_int8
        .unwrap_or(requested.supports_int8);
    requested.supports_float32 = metadata
        .requested_target_supports_float32
        .unwrap_or(requested.supports_float32);
    requested.supports_tensor_cores = metadata
        .requested_target_supports_tensor_cores
        .unwrap_or(requested.supports_tensor_cores);
    requested.supports_async_copy = metadata
        .requested_target_supports_async_copy
        .unwrap_or(requested.supports_async_copy);

    let mut observed = requested;
    observed.name = "attention_trace_observed_target";
    observed.backend = metadata.observed_target_backend.unwrap_or(observed.backend);
    observed.xlen = metadata.observed_target_xlen.unwrap_or(observed.xlen);
    observed.max_workgroup_threads = optional_u64_to_u32(
        metadata.target_max_workgroup_threads,
        observed.max_workgroup_threads,
    )?;
    observed.preferred_subgroup_width = metadata
        .observed_target_preferred_subgroup_width
        .or_else(|| optional_u64_to_u32_option(metadata.target_threads_per_warp))
        .unwrap_or(observed.preferred_subgroup_width);
    observed.local_memory_bytes = optional_u64_to_u32(
        metadata.target_local_memory_bytes,
        observed.local_memory_bytes,
    )?;
    observed.supports_int8 = metadata
        .observed_target_supports_int8
        .unwrap_or(observed.supports_int8);
    observed.supports_float32 = metadata
        .observed_target_supports_float32
        .unwrap_or(observed.supports_float32);
    observed.supports_tensor_cores = metadata
        .observed_target_supports_tensor_cores
        .unwrap_or(observed.supports_tensor_cores);
    observed.supports_async_copy = metadata
        .observed_target_supports_async_copy
        .unwrap_or(observed.supports_async_copy);

    let target = TargetContract::exact(requested, observed);
    if metadata
        .target_compatible
        .is_some_and(|compatible| compatible != target.is_compatible())
    {
        return None;
    }
    if metadata
        .target_mismatch_mask
        .is_some_and(|mask| mask != target.compatibility.mismatch_mask())
    {
        return None;
    }
    Some(target)
}

fn optional_u64_to_u32(value: Option<u64>, default: u32) -> Option<u32> {
    match value {
        Some(value) => u32::try_from(value).ok(),
        None => Some(default),
    }
}

fn optional_u64_to_u32_option(value: Option<u64>) -> Option<u32> {
    value.and_then(|value| u32::try_from(value).ok())
}

#[cfg(test)]
mod tests {
    use super::{
        ATTENTION_RUNTIME_TRACE_KERNEL, AttentionRuntimeTraceEnrichment, DeviceBackend,
        ExperimentSpec, attention_experiment_result_from_record,
        collect_attention_runtime_trace_with_enrichment,
    };

    const TRACE_STDOUT: &str = "PERF: instrs=165144, cycles=414598, IPC=0.398\nattention.runtime: compare summary elements=128 mismatches=0 status=exact\nMANDREL_RUNTIME_TRACE: kernel=attention_prefill_i8 runtime_sequence=8 runtime_head_dim=16 query_tile=4 key_tile=1 compiled_sequence=64 compiled_head_dim=64 head_dim_tile=64 logical_macs=2048 lowered_macs=33792 estimated_global_bytes_read=66560 estimated_global_bytes_written=128 estimated_local_memory_bytes_per_workgroup=0 requested_target_backend=vortex_rtl requested_target_xlen=64 requested_target_max_workgroup_threads=16 requested_target_preferred_subgroup_width=4 requested_target_local_memory_bytes=16384 requested_target_supports_int8=true requested_target_supports_float32=true requested_target_supports_tensor_cores=false requested_target_supports_async_copy=false observed_target_backend=vortex_rtl observed_target_xlen=64 observed_target_preferred_subgroup_width=4 observed_target_supports_int8=true observed_target_supports_float32=true observed_target_supports_tensor_cores=false observed_target_supports_async_copy=false target_compatible=true target_mismatch_mask=0 target_threads_per_warp=4 target_warps_per_core=4 target_max_workgroup_threads=16 target_local_memory_bytes=16384 workgroup_count=16 threads_per_workgroup=16 total_threads=256 runtime_q_elements=128 runtime_kv_elements=256 runtime_output_elements=128 runtime_q_bytes=128 runtime_kv_bytes=256 runtime_output_bytes=128 module_cache_hit=false kernel_cache_hit=false grid=16x1x1 block=4x4x1 shared_memory_bytes=0 host_to_device_bytes=384 device_to_host_bytes=128\n";

    #[test]
    fn collects_current_attention_runtime_evidence() {
        let spec = ExperimentSpec::attention_prefill_i8_vortex_smoke(8, 16);
        let report = collect_attention_runtime_trace_with_enrichment(
            TRACE_STDOUT,
            AttentionRuntimeTraceEnrichment::with_experiment_spec(123, spec),
        );
        let record = match report.record {
            Some(record) => record,
            None => panic!("expected structured runtime evidence"),
        };

        assert_eq!(
            record.summary.launch.kernel_symbol,
            ATTENTION_RUNTIME_TRACE_KERNEL
        );
        assert_eq!(record.metadata.runtime_sequence, Some(8));
        assert_eq!(record.metadata.runtime_head_dim, Some(16));
        assert_eq!(
            record.metadata.requested_target_backend,
            Some(DeviceBackend::VortexRtl)
        );
        assert_eq!(
            record.metadata.requested_target_supports_async_copy,
            Some(false)
        );
        assert_eq!(
            record.metadata.observed_target_supports_async_copy,
            Some(false)
        );
        assert_eq!(record.summary.counters.instructions, Some(165_144));
        assert_eq!(record.summary.counters.cycles, Some(414_598));
        assert_eq!(record.experiment_spec, Some(spec));

        let result = match attention_experiment_result_from_record(record) {
            Some(result) => result,
            None => panic!("expected experiment result"),
        };
        assert_eq!(result.spec_id, "attention_prefill_i8_vortex_smoke");
        assert_eq!(result.derived_metrics.total_transfer_bytes, 512);
        assert!(
            result
                .correctness
                .is_some_and(|correctness| correctness.passed)
        );
    }

    #[test]
    fn keeps_perf_counters_when_structured_evidence_is_missing() {
        let report = collect_attention_runtime_trace_with_enrichment(
            "PERF: instrs=7, cycles=11, IPC=0.636\n",
            AttentionRuntimeTraceEnrichment::default(),
        );

        assert!(report.record.is_none());
        assert_eq!(report.counters.instructions, Some(7));
        assert_eq!(report.counters.cycles, Some(11));
    }
}
