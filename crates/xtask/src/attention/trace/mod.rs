use std::path::{Path, PathBuf};

use mandrel_experiment::{CorrectnessResult, ExperimentResult, ExperimentSpec};
use mandrel_profiler::{KernelCounterTrace, RuntimeTraceSummary};
use tracing::warn;

mod collect;
mod history;
mod jsonl;
mod print;

pub(crate) use collect::collect_attention_runtime_trace_with_enrichment;
pub(crate) use history::print_attention_trace_history;
pub(crate) use jsonl::{
    append_attention_experiment_result_jsonl, append_attention_runtime_trace_jsonl,
};
pub(crate) use print::print_attention_runtime_trace_report;

use jsonl::*;
use print::*;

pub(crate) const ATTENTION_RUNTIME_TRACE_KERNEL: &str = "attention_prefill_i8";
pub(crate) const ATTENTION_TRACE_JSONL_RELATIVE_PATH: &str =
    "target/mandrel/vortex/attention_prefill_i8.trace.jsonl";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct AttentionRuntimeTraceEnrichment {
    pub(crate) wall_time_ms: Option<u64>,
}

impl AttentionRuntimeTraceEnrichment {
    pub(crate) const fn with_wall_time_ms(wall_time_ms: u64) -> Self {
        Self {
            wall_time_ms: Some(wall_time_ms),
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttentionRuntimeTraceReport {
    pub(crate) record: Option<AttentionRuntimeTraceRecord>,
    pub(crate) counters: KernelCounterTrace,
    pub(crate) metadata: AttentionRuntimeTraceMetadata,
}

pub(crate) fn default_attention_trace_jsonl_path(workspace_root: &Path) -> PathBuf {
    workspace_root.join(ATTENTION_TRACE_JSONL_RELATIVE_PATH)
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
    let spec = ExperimentSpec::attention_prefill_i8_vortex_smoke(sequence, head_dim);
    Some(ExperimentResult::from_runtime_trace_summary(
        spec,
        record.summary,
        correctness,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        ATTENTION_RUNTIME_TRACE_KERNEL, AttentionRuntimeTraceEnrichment,
        AttentionRuntimeTraceMetadata, attention_experiment_result_from_record,
        collect_attention_runtime_trace_with_enrichment, parse_attention_runtime_trace_jsonl,
        render_attention_experiment_result_jsonl, render_attention_runtime_trace_jsonl,
    };

    const TRACE_STDOUT: &str = "PERF: instrs=165144, cycles=414598, IPC=0.398\nattention.runtime: compare summary elements=128 mismatches=0 status=exact\nMANDREL_RUNTIME_TRACE: kernel=attention_prefill_i8 runtime_sequence=8 runtime_head_dim=16 query_tile=4 key_tile=16 compiled_sequence=64 compiled_head_dim=64 head_dim_tile=64 logical_macs=2048 workgroup_count=16 threads_per_workgroup=16 total_threads=256 runtime_q_elements=128 runtime_kv_elements=256 runtime_output_elements=128 runtime_q_bytes=128 runtime_kv_bytes=256 runtime_output_bytes=128 module_cache_hit=false kernel_cache_hit=false grid=16x1x1 block=4x4x1 shared_memory_bytes=2336 host_to_device_bytes=384 device_to_host_bytes=128\n";

    #[test]
    fn collects_attention_runtime_trace_from_stdout() {
        let report = collect_attention_runtime_trace_with_enrichment(
            TRACE_STDOUT,
            AttentionRuntimeTraceEnrichment::default(),
        );
        let record = match report.record {
            Some(record) => record,
            None => panic!("expected runtime trace record"),
        };

        assert_eq!(
            record.summary.launch.kernel_symbol,
            ATTENTION_RUNTIME_TRACE_KERNEL
        );
        assert_eq!(record.metadata.runtime_sequence, Some(8));
        assert_eq!(record.metadata.runtime_head_dim, Some(16));
        assert_eq!(record.metadata.query_tile, Some(4));
        assert_eq!(record.metadata.key_tile, Some(16));
        assert_eq!(record.metadata.compiled_sequence, Some(64));
        assert_eq!(record.metadata.compiled_head_dim, Some(64));
        assert_eq!(record.metadata.head_dim_tile, Some(64));
        assert_eq!(record.metadata.logical_macs, Some(2_048));
        assert_eq!(record.metadata.workgroup_count, Some(16));
        assert_eq!(record.metadata.threads_per_workgroup, Some(16));
        assert_eq!(record.metadata.total_threads, Some(256));
        assert_eq!(record.metadata.runtime_q_elements, Some(128));
        assert_eq!(record.metadata.runtime_kv_elements, Some(256));
        assert_eq!(record.metadata.runtime_output_elements, Some(128));
        assert_eq!(record.metadata.runtime_q_bytes, Some(128));
        assert_eq!(record.metadata.runtime_kv_bytes, Some(256));
        assert_eq!(record.metadata.runtime_output_bytes, Some(128));
        assert_eq!(record.metadata.module_cache_hit, Some(false));
        assert_eq!(record.metadata.kernel_cache_hit, Some(false));
        assert_eq!(record.metadata.wall_time_ms, None);
        assert_eq!(record.summary.launch.grid, [16, 1, 1]);
        assert_eq!(record.summary.launch.block, [4, 4, 1]);
        assert_eq!(record.summary.host_to_device_bytes, 384);
        assert_eq!(record.summary.device_to_host_bytes, 128);
        assert_eq!(record.summary.counters.instructions, Some(165_144));
        assert_eq!(record.summary.counters.cycles, Some(414_598));
        let correctness = record.correctness.expect("expected correctness result");
        assert!(correctness.passed);
        assert_eq!(correctness.compared_elements, 128);
        assert_eq!(correctness.mismatches, 0);
    }

    #[test]
    fn keeps_counters_when_structured_trace_is_missing() {
        let report = collect_attention_runtime_trace_with_enrichment(
            "PERF: instrs=7, cycles=11, IPC=0.636\n",
            AttentionRuntimeTraceEnrichment::default(),
        );

        assert!(report.record.is_none());
        assert_eq!(report.counters.instructions, Some(7));
        assert_eq!(report.counters.cycles, Some(11));
    }

    #[test]
    fn renders_attention_runtime_trace_jsonl() {
        let report = collect_attention_runtime_trace_with_enrichment(
            TRACE_STDOUT,
            AttentionRuntimeTraceEnrichment::with_wall_time_ms(123),
        );
        let record = match report.record {
            Some(record) => record,
            None => panic!("expected runtime trace record"),
        };

        assert_eq!(
            render_attention_runtime_trace_jsonl(record),
            "{\"schema\":\"mandrel.runtime.trace.v0\",\"kernel\":\"attention_prefill_i8\",\"runtime_sequence\":8,\"runtime_head_dim\":16,\"query_tile\":4,\"key_tile\":16,\"compiled_sequence\":64,\"compiled_head_dim\":64,\"head_dim_tile\":64,\"logical_macs\":2048,\"workgroup_count\":16,\"threads_per_workgroup\":16,\"total_threads\":256,\"runtime_q_elements\":128,\"runtime_kv_elements\":256,\"runtime_output_elements\":128,\"runtime_q_bytes\":128,\"runtime_kv_bytes\":256,\"runtime_output_bytes\":128,\"module_cache_hit\":false,\"kernel_cache_hit\":false,\"wall_time_ms\":123,\"grid\":[16,1,1],\"block\":[4,4,1],\"shared_memory_bytes\":2336,\"host_to_device_bytes\":384,\"device_to_host_bytes\":128,\"total_transfer_bytes\":512,\"instructions\":165144,\"cycles\":414598,\"correctness_passed\":true,\"correctness_compared_elements\":128,\"correctness_mismatches\":0}"
        );
    }

    #[test]
    fn parses_attention_runtime_trace_jsonl() {
        let report = collect_attention_runtime_trace_with_enrichment(
            TRACE_STDOUT,
            AttentionRuntimeTraceEnrichment::with_wall_time_ms(123),
        );
        let record = match report.record {
            Some(record) => record,
            None => panic!("expected runtime trace record"),
        };
        let json = render_attention_runtime_trace_jsonl(record);
        let parsed = match parse_attention_runtime_trace_jsonl(&json) {
            Some(parsed) => parsed,
            None => panic!("expected JSONL record to parse"),
        };

        assert_eq!(parsed, record);
    }

    #[test]
    fn parses_legacy_attention_runtime_trace_jsonl_without_metadata() {
        let parsed = match parse_attention_runtime_trace_jsonl(
            "{\"schema\":\"mandrel.runtime.trace.v0\",\"kernel\":\"attention_prefill_i8\",\"grid\":[16,1,1],\"block\":[4,4,1],\"shared_memory_bytes\":2336,\"host_to_device_bytes\":384,\"device_to_host_bytes\":128,\"total_transfer_bytes\":512,\"instructions\":165144,\"cycles\":414598}",
        ) {
            Some(parsed) => parsed,
            None => panic!("expected legacy JSONL record to parse"),
        };

        assert_eq!(parsed.metadata, AttentionRuntimeTraceMetadata::default());
        assert_eq!(parsed.summary.launch.grid, [16, 1, 1]);
        assert_eq!(parsed.summary.counters.instructions, Some(165_144));
        assert_eq!(parsed.summary.counters.cycles, Some(414_598));
        assert_eq!(parsed.correctness, None);
    }

    #[test]
    fn converts_attention_trace_record_to_experiment_result() {
        let report = collect_attention_runtime_trace_with_enrichment(
            TRACE_STDOUT,
            AttentionRuntimeTraceEnrichment::with_wall_time_ms(123),
        );
        let record = report.record.expect("expected runtime trace record");
        let result = attention_experiment_result_from_record(record)
            .expect("expected experiment result from trace record");

        assert_eq!(result.spec_id, "attention_prefill_i8_vortex_smoke");
        assert_eq!(result.events.len(), 6);
        assert_eq!(result.derived_metrics.total_transfer_bytes, 512);
        assert_eq!(
            result.correctness.expect("expected correctness").mismatches,
            0
        );
        assert_eq!(
            render_attention_experiment_result_jsonl(&result),
            "{\"schema\":\"mandrel.experiment.result.v0\",\"spec_id\":\"attention_prefill_i8_vortex_smoke\",\"status\":\"succeeded\",\"event_count\":6,\"event_kinds\":[\"copy\",\"kernel_launch\",\"sync\",\"copy\",\"perf_counter\",\"correctness_check\"],\"total_transfer_bytes\":512,\"instructions\":165144,\"cycles\":414598,\"correctness_passed\":true,\"correctness_compared_elements\":128,\"correctness_mismatches\":0}"
        );
    }
}
