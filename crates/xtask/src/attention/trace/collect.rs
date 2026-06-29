use mandrel_profiler::{find_mandrel_runtime_trace_summary, find_vortex_perf_counters};

use super::*;

pub(crate) fn collect_attention_runtime_trace_with_enrichment(
    stdout: &str,
    enrichment: AttentionRuntimeTraceEnrichment,
) -> AttentionRuntimeTraceReport {
    let counters = attention_counters_from_stdout(stdout);
    let mut metadata = attention_metadata_from_stdout(stdout);
    apply_attention_trace_enrichment(&mut metadata, enrichment);
    let Some(trace_result) =
        find_mandrel_runtime_trace_summary(stdout, ATTENTION_RUNTIME_TRACE_KERNEL)
    else {
        warn!("Vortex runtime output did not contain a Mandrel runtime trace line");
        return AttentionRuntimeTraceReport {
            record: None,
            counters,
            metadata,
        };
    };

    let summary_without_counters = match trace_result {
        Ok(summary) => summary,
        Err(error) => {
            warn!(?error, "failed to parse Mandrel runtime trace line");
            return AttentionRuntimeTraceReport {
                record: None,
                counters,
                metadata,
            };
        }
    };

    let summary = RuntimeTraceSummary::new(
        summary_without_counters.launch,
        summary_without_counters.host_to_device_bytes,
        summary_without_counters.device_to_host_bytes,
        counters,
    );
    AttentionRuntimeTraceReport {
        record: Some(AttentionRuntimeTraceRecord { metadata, summary }),
        counters,
        metadata,
    }
}

pub(super) fn attention_counters_from_stdout(stdout: &str) -> KernelCounterTrace {
    match find_vortex_perf_counters(stdout) {
        Some(Ok(counters)) => counters,
        Some(Err(error)) => {
            warn!(?error, "failed to parse Vortex PERF counters");
            KernelCounterTrace::empty()
        }
        None => {
            warn!("Vortex runtime output did not contain a PERF counter line");
            KernelCounterTrace::empty()
        }
    }
}

pub(super) fn attention_metadata_from_stdout(stdout: &str) -> AttentionRuntimeTraceMetadata {
    let Some(line) = find_mandrel_runtime_trace_line(stdout) else {
        return AttentionRuntimeTraceMetadata::default();
    };

    AttentionRuntimeTraceMetadata {
        runtime_sequence: parse_trace_u32_field(line, "runtime_sequence"),
        runtime_head_dim: parse_trace_u32_field(line, "runtime_head_dim"),
        query_tile: parse_trace_u32_field(line, "query_tile"),
        key_tile: parse_trace_u32_field(line, "key_tile"),
        compiled_sequence: parse_trace_u32_field(line, "compiled_sequence"),
        compiled_head_dim: parse_trace_u32_field(line, "compiled_head_dim"),
        head_dim_tile: parse_trace_u32_field(line, "head_dim_tile"),
        logical_macs: parse_trace_u64_field(line, "logical_macs"),
        workgroup_count: parse_trace_u64_field(line, "workgroup_count"),
        threads_per_workgroup: parse_trace_u64_field(line, "threads_per_workgroup"),
        total_threads: parse_trace_u64_field(line, "total_threads"),
        runtime_q_elements: parse_trace_u64_field(line, "runtime_q_elements"),
        runtime_kv_elements: parse_trace_u64_field(line, "runtime_kv_elements"),
        runtime_output_elements: parse_trace_u64_field(line, "runtime_output_elements"),
        runtime_q_bytes: parse_trace_u64_field(line, "runtime_q_bytes"),
        runtime_kv_bytes: parse_trace_u64_field(line, "runtime_kv_bytes"),
        runtime_output_bytes: parse_trace_u64_field(line, "runtime_output_bytes"),
        module_cache_hit: parse_trace_bool_field(line, "module_cache_hit"),
        kernel_cache_hit: parse_trace_bool_field(line, "kernel_cache_hit"),
        wall_time_ms: parse_trace_u64_field(line, "wall_time_ms"),
    }
}

pub(super) fn find_mandrel_runtime_trace_line(output: &str) -> Option<&str> {
    output
        .lines()
        .find(|line| line.trim().starts_with("MANDREL_RUNTIME_TRACE:"))
        .map(str::trim)
}

pub(super) fn parse_trace_u32_field(line: &str, field_name: &str) -> Option<u32> {
    let raw = parse_trace_field(line, field_name)?;
    match raw.parse::<u32>() {
        Ok(value) => Some(value),
        Err(error) => {
            warn!(field_name, raw, %error, "failed to parse attention runtime trace u32 field");
            None
        }
    }
}

pub(super) fn parse_trace_u64_field(line: &str, field_name: &str) -> Option<u64> {
    let raw = parse_trace_field(line, field_name)?;
    match raw.parse::<u64>() {
        Ok(value) => Some(value),
        Err(error) => {
            warn!(field_name, raw, %error, "failed to parse attention runtime trace u64 field");
            None
        }
    }
}

pub(super) fn parse_trace_bool_field(line: &str, field_name: &str) -> Option<bool> {
    let raw = parse_trace_field(line, field_name)?;
    match raw {
        "true" | "1" => Some(true),
        "false" | "0" => Some(false),
        _ => {
            warn!(
                field_name,
                raw, "failed to parse attention runtime trace bool field"
            );
            None
        }
    }
}

pub(super) fn parse_trace_field<'a>(line: &'a str, field_name: &str) -> Option<&'a str> {
    let prefix = format!("{field_name}=");
    line.split_whitespace()
        .find_map(|field| field.strip_prefix(&prefix))
}

fn apply_attention_trace_enrichment(
    metadata: &mut AttentionRuntimeTraceMetadata,
    enrichment: AttentionRuntimeTraceEnrichment,
) {
    if enrichment.wall_time_ms.is_some() {
        metadata.wall_time_ms = enrichment.wall_time_ms;
    }
}
