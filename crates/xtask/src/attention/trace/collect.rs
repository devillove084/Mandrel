use mandrel_profiler::{find_mandrel_runtime_trace_summary, find_vortex_perf_counters};

use super::*;

pub(crate) fn collect_attention_runtime_trace_with_enrichment(
    stdout: &str,
    enrichment: AttentionRuntimeTraceEnrichment,
) -> AttentionRuntimeTraceReport {
    let counters = attention_counters_from_stdout(stdout);
    let correctness = attention_correctness_from_stdout(stdout);
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
        record: Some(AttentionRuntimeTraceRecord {
            metadata,
            summary,
            correctness,
            experiment_spec: enrichment.experiment_spec,
        }),
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

pub(super) fn attention_correctness_from_stdout(stdout: &str) -> Option<CorrectnessResult> {
    let Some(line) = stdout.lines().find_map(|line| {
        let trimmed = line.trim();
        trimmed
            .starts_with("attention.runtime: compare summary ")
            .then_some(trimmed)
    }) else {
        warn!("Vortex runtime output did not contain an attention correctness summary line");
        return None;
    };
    let elements = parse_trace_usize_field(line, "elements")?;
    let mismatches = parse_trace_usize_field(line, "mismatches")?;
    let Some(status) = parse_trace_field(line, "status") else {
        warn!("attention correctness summary did not contain a status field");
        return None;
    };
    if status != "exact" {
        warn!(status, "unsupported attention correctness summary status");
        return None;
    }

    Some(CorrectnessResult::exact(elements, mismatches))
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
        lowered_macs: parse_trace_u64_field(line, "lowered_macs"),
        estimated_global_bytes_read: parse_trace_u64_field(line, "estimated_global_bytes_read"),
        estimated_global_bytes_written: parse_trace_u64_field(
            line,
            "estimated_global_bytes_written",
        ),
        estimated_local_memory_bytes_per_workgroup: parse_trace_u64_field(
            line,
            "estimated_local_memory_bytes_per_workgroup",
        ),
        requested_target_backend: parse_trace_device_backend_field(
            line,
            "requested_target_backend",
        ),
        requested_target_xlen: parse_trace_u32_field(line, "requested_target_xlen"),
        requested_target_max_workgroup_threads: parse_trace_u32_field(
            line,
            "requested_target_max_workgroup_threads",
        ),
        requested_target_preferred_subgroup_width: parse_trace_u32_field(
            line,
            "requested_target_preferred_subgroup_width",
        ),
        requested_target_local_memory_bytes: parse_trace_u32_field(
            line,
            "requested_target_local_memory_bytes",
        ),
        requested_target_supports_int8: parse_trace_bool_field(
            line,
            "requested_target_supports_int8",
        ),
        requested_target_supports_float32: parse_trace_bool_field(
            line,
            "requested_target_supports_float32",
        ),
        requested_target_supports_tensor_cores: parse_trace_bool_field(
            line,
            "requested_target_supports_tensor_cores",
        ),
        requested_target_supports_async_copy: parse_trace_bool_field(
            line,
            "requested_target_supports_async_copy",
        ),
        observed_target_backend: parse_trace_device_backend_field(line, "observed_target_backend"),
        observed_target_xlen: parse_trace_u32_field(line, "observed_target_xlen"),
        observed_target_preferred_subgroup_width: parse_trace_u32_field(
            line,
            "observed_target_preferred_subgroup_width",
        ),
        observed_target_supports_int8: parse_trace_bool_field(
            line,
            "observed_target_supports_int8",
        ),
        observed_target_supports_float32: parse_trace_bool_field(
            line,
            "observed_target_supports_float32",
        ),
        observed_target_supports_tensor_cores: parse_trace_bool_field(
            line,
            "observed_target_supports_tensor_cores",
        ),
        observed_target_supports_async_copy: parse_trace_bool_field(
            line,
            "observed_target_supports_async_copy",
        ),
        target_compatible: parse_trace_bool_field(line, "target_compatible"),
        target_mismatch_mask: parse_trace_u16_field(line, "target_mismatch_mask"),
        target_threads_per_warp: parse_trace_u64_field(line, "target_threads_per_warp"),
        target_warps_per_core: parse_trace_u64_field(line, "target_warps_per_core"),
        target_max_workgroup_threads: parse_trace_u64_field(line, "target_max_workgroup_threads"),
        target_local_memory_bytes: parse_trace_u64_field(line, "target_local_memory_bytes"),
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

pub(super) fn parse_trace_u16_field(line: &str, field_name: &str) -> Option<u16> {
    let raw = parse_trace_field(line, field_name)?;
    match raw.parse::<u16>() {
        Ok(value) => Some(value),
        Err(error) => {
            warn!(field_name, raw, %error, "failed to parse attention runtime trace u16 field");
            None
        }
    }
}

pub(super) fn parse_trace_usize_field(line: &str, field_name: &str) -> Option<usize> {
    let raw = parse_trace_field(line, field_name)?;
    match raw.parse::<usize>() {
        Ok(value) => Some(value),
        Err(error) => {
            warn!(field_name, raw, %error, "failed to parse attention runtime trace usize field");
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

pub(super) fn parse_trace_device_backend_field(
    line: &str,
    field_name: &str,
) -> Option<DeviceBackend> {
    let raw = parse_trace_field(line, field_name)?;
    match raw {
        "host_reference" => Some(DeviceBackend::HostReference),
        "vortex_simx" => Some(DeviceBackend::VortexSimx),
        "vortex_rtl" => Some(DeviceBackend::VortexRtl),
        "vortex_fpga" => Some(DeviceBackend::VortexFpga),
        _ => {
            warn!(
                field_name,
                raw, "failed to parse attention runtime trace device backend field"
            );
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
