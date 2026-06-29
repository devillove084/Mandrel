use std::fs::{self, OpenOptions};
use std::io::{self, Write};

use mandrel_experiment::{CorrectnessResult, ExperimentResult, ExperimentStatus, RuntimeEvent};

use super::*;

pub(crate) fn append_attention_runtime_trace_jsonl(
    report: AttentionRuntimeTraceReport,
    path: &Path,
) -> io::Result<bool> {
    let Some(record) = report.record else {
        return Ok(false);
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", render_attention_runtime_trace_jsonl(record))?;
    Ok(true)
}

pub(crate) fn append_attention_experiment_result_jsonl(
    report: AttentionRuntimeTraceReport,
    path: &Path,
) -> io::Result<bool> {
    let Some(record) = report.record else {
        return Ok(false);
    };
    let Some(result) = attention_experiment_result_from_record(record) else {
        return Ok(false);
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(
        file,
        "{}",
        render_attention_experiment_result_jsonl(&result)
    )?;
    Ok(true)
}

pub(super) fn render_attention_runtime_trace_jsonl(record: AttentionRuntimeTraceRecord) -> String {
    let summary = record.summary;
    let metadata = record.metadata;
    let mut json = String::new();
    json.push('{');
    push_json_string_field(&mut json, "schema", "mandrel.runtime.trace.v0");
    json.push(',');
    push_json_string_field(&mut json, "kernel", summary.launch.kernel_symbol);
    json.push_str(",\"runtime_sequence\":");
    push_json_optional_u32(&mut json, metadata.runtime_sequence);
    json.push_str(",\"runtime_head_dim\":");
    push_json_optional_u32(&mut json, metadata.runtime_head_dim);
    json.push_str(",\"query_tile\":");
    push_json_optional_u32(&mut json, metadata.query_tile);
    json.push_str(",\"key_tile\":");
    push_json_optional_u32(&mut json, metadata.key_tile);
    json.push_str(",\"compiled_sequence\":");
    push_json_optional_u32(&mut json, metadata.compiled_sequence);
    json.push_str(",\"compiled_head_dim\":");
    push_json_optional_u32(&mut json, metadata.compiled_head_dim);
    json.push_str(",\"head_dim_tile\":");
    push_json_optional_u32(&mut json, metadata.head_dim_tile);
    json.push_str(",\"logical_macs\":");
    push_json_optional_u64(&mut json, metadata.logical_macs);
    json.push_str(",\"workgroup_count\":");
    push_json_optional_u64(&mut json, metadata.workgroup_count);
    json.push_str(",\"threads_per_workgroup\":");
    push_json_optional_u64(&mut json, metadata.threads_per_workgroup);
    json.push_str(",\"total_threads\":");
    push_json_optional_u64(&mut json, metadata.total_threads);
    json.push_str(",\"runtime_q_elements\":");
    push_json_optional_u64(&mut json, metadata.runtime_q_elements);
    json.push_str(",\"runtime_kv_elements\":");
    push_json_optional_u64(&mut json, metadata.runtime_kv_elements);
    json.push_str(",\"runtime_output_elements\":");
    push_json_optional_u64(&mut json, metadata.runtime_output_elements);
    json.push_str(",\"runtime_q_bytes\":");
    push_json_optional_u64(&mut json, metadata.runtime_q_bytes);
    json.push_str(",\"runtime_kv_bytes\":");
    push_json_optional_u64(&mut json, metadata.runtime_kv_bytes);
    json.push_str(",\"runtime_output_bytes\":");
    push_json_optional_u64(&mut json, metadata.runtime_output_bytes);
    json.push_str(",\"module_cache_hit\":");
    push_json_optional_bool(&mut json, metadata.module_cache_hit);
    json.push_str(",\"kernel_cache_hit\":");
    push_json_optional_bool(&mut json, metadata.kernel_cache_hit);
    json.push_str(",\"wall_time_ms\":");
    push_json_optional_u64(&mut json, metadata.wall_time_ms);
    json.push_str(",\"grid\":");
    push_json_u32_array(&mut json, summary.launch.grid);
    json.push_str(",\"block\":");
    push_json_u32_array(&mut json, summary.launch.block);
    json.push_str(",\"shared_memory_bytes\":");
    json.push_str(&summary.launch.shared_memory_bytes.to_string());
    json.push_str(",\"host_to_device_bytes\":");
    json.push_str(&summary.host_to_device_bytes.to_string());
    json.push_str(",\"device_to_host_bytes\":");
    json.push_str(&summary.device_to_host_bytes.to_string());
    json.push_str(",\"total_transfer_bytes\":");
    json.push_str(&summary.total_transfer_bytes().to_string());
    json.push_str(",\"instructions\":");
    push_json_optional_u64(&mut json, summary.counters.instructions);
    json.push_str(",\"cycles\":");
    push_json_optional_u64(&mut json, summary.counters.cycles);
    if let Some(correctness) = record.correctness {
        json.push_str(",\"correctness_passed\":");
        push_json_optional_bool(&mut json, Some(correctness.passed));
        json.push_str(",\"correctness_compared_elements\":");
        json.push_str(&correctness.compared_elements.to_string());
        json.push_str(",\"correctness_mismatches\":");
        json.push_str(&correctness.mismatches.to_string());
    }
    json.push('}');
    json
}

pub(super) fn render_attention_experiment_result_jsonl(result: &ExperimentResult) -> String {
    let mut json = String::new();
    json.push('{');
    push_json_string_field(&mut json, "schema", "mandrel.experiment.result.v0");
    json.push(',');
    push_json_string_field(&mut json, "spec_id", result.spec_id);
    json.push(',');
    push_json_string_field(&mut json, "status", experiment_status_as_str(result.status));
    json.push_str(",\"event_count\":");
    json.push_str(&result.events.len().to_string());
    json.push_str(",\"event_kinds\":");
    push_json_event_kinds(&mut json, &result.events);
    json.push_str(",\"total_transfer_bytes\":");
    json.push_str(&result.derived_metrics.total_transfer_bytes.to_string());
    json.push_str(",\"instructions\":");
    push_json_optional_u64(&mut json, result.counters.kernel.instructions);
    json.push_str(",\"cycles\":");
    push_json_optional_u64(&mut json, result.counters.kernel.cycles);
    if let Some(correctness) = result.correctness {
        json.push_str(",\"correctness_passed\":");
        push_json_optional_bool(&mut json, Some(correctness.passed));
        json.push_str(",\"correctness_compared_elements\":");
        json.push_str(&correctness.compared_elements.to_string());
        json.push_str(",\"correctness_mismatches\":");
        json.push_str(&correctness.mismatches.to_string());
    }
    json.push('}');
    json
}

fn experiment_status_as_str(status: ExperimentStatus) -> &'static str {
    match status {
        ExperimentStatus::Planned => "planned",
        ExperimentStatus::Unsupported => "unsupported",
        ExperimentStatus::Succeeded => "succeeded",
        ExperimentStatus::Failed => "failed",
    }
}

fn push_json_event_kinds(json: &mut String, events: &[RuntimeEvent]) {
    json.push('[');
    for (index, event) in events.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        push_json_string(json, runtime_event_kind_as_str(event));
    }
    json.push(']');
}

fn runtime_event_kind_as_str(event: &RuntimeEvent) -> &'static str {
    match event {
        RuntimeEvent::Allocate { .. } => "allocate",
        RuntimeEvent::Copy { .. } => "copy",
        RuntimeEvent::KernelLaunch(_) => "kernel_launch",
        RuntimeEvent::Sync { .. } => "sync",
        RuntimeEvent::PerfCounter(_) => "perf_counter",
        RuntimeEvent::CorrectnessCheck(_) => "correctness_check",
    }
}

pub(super) fn parse_attention_runtime_trace_jsonl(
    line: &str,
) -> Option<AttentionRuntimeTraceRecord> {
    let metadata = AttentionRuntimeTraceMetadata {
        runtime_sequence: parse_json_optional_u32(line, "runtime_sequence")?,
        runtime_head_dim: parse_json_optional_u32(line, "runtime_head_dim")?,
        query_tile: parse_json_optional_u32(line, "query_tile")?,
        key_tile: parse_json_optional_u32(line, "key_tile")?,
        compiled_sequence: parse_json_optional_u32(line, "compiled_sequence")?,
        compiled_head_dim: parse_json_optional_u32(line, "compiled_head_dim")?,
        head_dim_tile: parse_json_optional_u32(line, "head_dim_tile")?,
        logical_macs: parse_json_optional_u64(line, "logical_macs")?,
        workgroup_count: parse_json_optional_u64(line, "workgroup_count")?,
        threads_per_workgroup: parse_json_optional_u64(line, "threads_per_workgroup")?,
        total_threads: parse_json_optional_u64(line, "total_threads")?,
        runtime_q_elements: parse_json_optional_u64(line, "runtime_q_elements")?,
        runtime_kv_elements: parse_json_optional_u64(line, "runtime_kv_elements")?,
        runtime_output_elements: parse_json_optional_u64(line, "runtime_output_elements")?,
        runtime_q_bytes: parse_json_optional_u64(line, "runtime_q_bytes")?,
        runtime_kv_bytes: parse_json_optional_u64(line, "runtime_kv_bytes")?,
        runtime_output_bytes: parse_json_optional_u64(line, "runtime_output_bytes")?,
        module_cache_hit: parse_json_optional_bool(line, "module_cache_hit")?,
        kernel_cache_hit: parse_json_optional_bool(line, "kernel_cache_hit")?,
        wall_time_ms: parse_json_optional_u64(line, "wall_time_ms")?,
    };
    let summary = RuntimeTraceSummary::new(
        mandrel_profiler::KernelLaunchTrace::new(
            ATTENTION_RUNTIME_TRACE_KERNEL,
            parse_json_u32_array3(line, "grid")?,
            parse_json_u32_array3(line, "block")?,
            parse_json_u32(line, "shared_memory_bytes")?,
        ),
        parse_json_usize(line, "host_to_device_bytes")?,
        parse_json_usize(line, "device_to_host_bytes")?,
        KernelCounterTrace::new(
            parse_json_optional_u64(line, "instructions")?,
            parse_json_optional_u64(line, "cycles")?,
        ),
    );
    let correctness = parse_json_correctness(line)?;
    Some(AttentionRuntimeTraceRecord {
        metadata,
        summary,
        correctness,
    })
}

pub(super) fn parse_json_u32(line: &str, field_name: &str) -> Option<u32> {
    parse_json_number_field(line, field_name)?
        .parse::<u32>()
        .ok()
}

pub(super) fn parse_json_usize(line: &str, field_name: &str) -> Option<usize> {
    parse_json_number_field(line, field_name)?
        .parse::<usize>()
        .ok()
}

pub(super) fn parse_json_optional_u32(line: &str, field_name: &str) -> Option<Option<u32>> {
    let Some(raw) = parse_json_nullable_scalar_field(line, field_name) else {
        return Some(None);
    };
    if raw == "null" {
        Some(None)
    } else {
        Some(Some(raw.parse::<u32>().ok()?))
    }
}

pub(super) fn parse_json_optional_u64(line: &str, field_name: &str) -> Option<Option<u64>> {
    let Some(raw) = parse_json_nullable_scalar_field(line, field_name) else {
        return Some(None);
    };
    if raw == "null" {
        Some(None)
    } else {
        Some(Some(raw.parse::<u64>().ok()?))
    }
}

pub(super) fn parse_json_optional_usize(line: &str, field_name: &str) -> Option<Option<usize>> {
    let Some(raw) = parse_json_nullable_scalar_field(line, field_name) else {
        return Some(None);
    };
    if raw == "null" {
        Some(None)
    } else {
        Some(Some(raw.parse::<usize>().ok()?))
    }
}

pub(super) fn parse_json_optional_bool(line: &str, field_name: &str) -> Option<Option<bool>> {
    let Some(raw) = parse_json_nullable_scalar_field(line, field_name) else {
        return Some(None);
    };
    match raw {
        "null" => Some(None),
        "true" => Some(Some(true)),
        "false" => Some(Some(false)),
        _ => None,
    }
}

pub(super) fn parse_json_correctness(line: &str) -> Option<Option<CorrectnessResult>> {
    let passed = parse_json_optional_bool(line, "correctness_passed")?;
    let compared_elements = parse_json_optional_usize(line, "correctness_compared_elements")?;
    let mismatches = parse_json_optional_usize(line, "correctness_mismatches")?;
    match (passed, compared_elements, mismatches) {
        (None, None, None) => Some(None),
        (Some(passed), Some(compared_elements), Some(mismatches)) => {
            let result = CorrectnessResult::exact(compared_elements, mismatches);
            (result.passed == passed).then_some(Some(result))
        }
        _ => None,
    }
}

pub(super) fn parse_json_u32_array3(line: &str, field_name: &str) -> Option<[u32; 3]> {
    let marker = format!("\"{field_name}\":[");
    let start = line.find(&marker)? + marker.len();
    let end = line[start..].find(']')? + start;
    let mut parts = line[start..end].split(',');
    let x = parts.next()?.parse::<u32>().ok()?;
    let y = parts.next()?.parse::<u32>().ok()?;
    let z = parts.next()?.parse::<u32>().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some([x, y, z])
}

pub(super) fn parse_json_number_field<'a>(line: &'a str, field_name: &str) -> Option<&'a str> {
    let raw = parse_json_nullable_scalar_field(line, field_name)?;
    if raw == "null" { None } else { Some(raw) }
}

pub(super) fn parse_json_nullable_scalar_field<'a>(
    line: &'a str,
    field_name: &str,
) -> Option<&'a str> {
    let marker = format!("\"{field_name}\":");
    let start = line.find(&marker)? + marker.len();
    let end = line[start..]
        .find([',', '}'])
        .map_or(line.len(), |offset| start + offset);
    Some(line[start..end].trim())
}

pub(super) fn push_json_string_field(json: &mut String, key: &str, value: &str) {
    push_json_string(json, key);
    json.push(':');
    push_json_string(json, value);
}

pub(super) fn push_json_string(json: &mut String, value: &str) {
    json.push('"');
    for character in value.chars() {
        match character {
            '"' => json.push_str("\\\""),
            '\\' => json.push_str("\\\\"),
            '\n' => json.push_str("\\n"),
            '\r' => json.push_str("\\r"),
            '\t' => json.push_str("\\t"),
            character => json.push(character),
        }
    }
    json.push('"');
}

pub(super) fn push_json_u32_array(json: &mut String, values: [u32; 3]) {
    json.push('[');
    json.push_str(&values[0].to_string());
    json.push(',');
    json.push_str(&values[1].to_string());
    json.push(',');
    json.push_str(&values[2].to_string());
    json.push(']');
}

pub(super) fn push_json_optional_u32(json: &mut String, value: Option<u32>) {
    match value {
        Some(value) => json.push_str(&value.to_string()),
        None => json.push_str("null"),
    }
}

pub(super) fn push_json_optional_u64(json: &mut String, value: Option<u64>) {
    match value {
        Some(value) => json.push_str(&value.to_string()),
        None => json.push_str("null"),
    }
}

pub(super) fn push_json_optional_bool(json: &mut String, value: Option<bool>) {
    match value {
        Some(true) => json.push_str("true"),
        Some(false) => json.push_str("false"),
        None => json.push_str("null"),
    }
}
