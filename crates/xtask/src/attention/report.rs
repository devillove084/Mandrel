use std::fs;
use std::io;
use std::path::Path;

use mandrel_artifact::{ArtifactKind, ArtifactRef, ArtifactSet, VortexMlirKernelArtifacts};
use mandrel_experiment::{ExperimentResult, ExperimentStatus, RuntimeEvent};
use mandrel_hardware::{CURRENT_LLVM_VORTEX_REVISION, CURRENT_VORTEX_REVISION};
use mandrel_target_ir::{TargetCapability, TargetContract};

use super::trace::{AttentionRuntimeTraceRecord, attention_experiment_result_from_record};

pub(crate) fn experiment_result_with_artifacts(
    record: AttentionRuntimeTraceRecord,
    paths: &VortexMlirKernelArtifacts,
    workspace_root: &Path,
) -> Option<ExperimentResult> {
    let mut result = attention_experiment_result_from_record(record)?;
    result.artifacts = ArtifactSet::empty()
        .with_artifact(path_artifact(
            ArtifactKind::Mlir,
            &paths.mlir_path,
            workspace_root,
        ))
        .with_artifact(path_artifact(
            ArtifactKind::LlvmIr,
            &paths.ll_path,
            workspace_root,
        ))
        .with_artifact(path_artifact(
            ArtifactKind::Object,
            &paths.obj_path,
            workspace_root,
        ))
        .with_artifact(path_artifact(
            ArtifactKind::Elf,
            &paths.elf_path,
            workspace_root,
        ))
        .with_artifact(path_artifact(
            ArtifactKind::Vxbin,
            &paths.vxbin_path,
            workspace_root,
        ));
    Some(result)
}

pub(crate) fn write_experiment_json(
    result: &ExperimentResult,
    record: AttentionRuntimeTraceRecord,
    path: &Path,
) -> io::Result<()> {
    write_report(path, &render_experiment_json(result, record))
}

pub(crate) fn write_experiment_csv(
    result: &ExperimentResult,
    record: AttentionRuntimeTraceRecord,
    path: &Path,
) -> io::Result<()> {
    write_report(path, &render_experiment_csv(result, record))
}

fn write_report(path: &Path, contents: &str) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, format!("{contents}\n"))
}

fn path_artifact(kind: ArtifactKind, path: &Path, workspace_root: &Path) -> ArtifactRef {
    let portable_path = path.strip_prefix(workspace_root).unwrap_or(path);
    ArtifactRef::new(kind, portable_path.to_string_lossy().into_owned())
}

fn render_experiment_json(
    result: &ExperimentResult,
    record: AttentionRuntimeTraceRecord,
) -> String {
    let mut json = String::new();
    json.push('{');
    json_string_field(&mut json, "schema", "mandrel.experiment.result.v2");
    json.push(',');
    json_string_field(&mut json, "spec_id", result.spec_id);
    json.push(',');
    json_string_field(&mut json, "status", experiment_status_as_str(result.status));
    json.push(',');
    json_string_field(&mut json, "evidence_class", "simx_observation");

    if let Some(spec) = record.experiment_spec {
        json.push_str(",\"workload\":{");
        json_string_field(&mut json, "name", spec.workload.name);
        json.push_str(",\"sequence\":");
        json.push_str(&spec.workload.attention.sequence.to_string());
        json.push_str(",\"head_dim\":");
        json.push_str(&spec.workload.attention.head_dim.to_string());
        json.push('}');

        let hardware = spec.hardware.vortex;
        json.push_str(",\"hardware_design\":{");
        json_string_field(&mut json, "id", spec.hardware.id);
        json.push_str(",\"vortex_revision\":");
        json_string(&mut json, CURRENT_VORTEX_REVISION);
        json.push_str(",\"llvm_vortex_revision\":");
        json_string(&mut json, CURRENT_LLVM_VORTEX_REVISION);
        json.push_str(",\"xlen\":");
        json.push_str(&hardware.xlen.to_string());
        json.push_str(",\"clusters\":");
        json.push_str(&hardware.clusters.to_string());
        json.push_str(",\"cores\":");
        json.push_str(&hardware.cores.to_string());
        json.push_str(",\"warps_per_core\":");
        json.push_str(&hardware.warps_per_core.to_string());
        json.push_str(",\"threads_per_warp\":");
        json.push_str(&hardware.threads_per_warp.to_string());
        json.push_str(",\"local_memory_log2_bytes\":");
        json.push_str(&hardware.local_memory_log2_bytes.to_string());
        json.push_str(",\"dcache_bytes\":");
        json.push_str(&hardware.dcache_bytes.to_string());
        json.push_str(",\"tensor_cores\":");
        json.push_str(bool_as_str(hardware.tensor_cores));
        json.push_str(",\"async_copy\":");
        json.push_str(bool_as_str(hardware.async_copy));
        json.push('}');
    }

    json.push_str(",\"software_design\":{");
    json_string_field(&mut json, "kernel", record.summary.launch.kernel_symbol);
    json.push_str(",\"lowering\":\"dense_scalar_two_pass_4x1x64\"");
    json_optional_u32_field(&mut json, "query_tile", record.metadata.query_tile);
    json_optional_u32_field(&mut json, "key_tile", record.metadata.key_tile);
    json_optional_u32_field(&mut json, "head_dim_tile", record.metadata.head_dim_tile);
    json.push('}');

    json.push_str(",\"target_contract\":");
    render_target_contract_json(&mut json, result.target);

    json.push_str(",\"correctness\":");
    match result.correctness {
        Some(correctness) => {
            json.push('{');
            json.push_str("\"passed\":");
            json.push_str(bool_as_str(correctness.passed));
            json.push_str(",\"compared_elements\":");
            json.push_str(&correctness.compared_elements.to_string());
            json.push_str(",\"mismatches\":");
            json.push_str(&correctness.mismatches.to_string());
            json.push('}');
        }
        None => json.push_str("null"),
    }

    json.push_str(",\"counters\":{");
    json_optional_u64_field(
        &mut json,
        "instructions",
        result.counters.kernel.instructions,
        false,
    );
    json_optional_u64_field(&mut json, "cycles", result.counters.kernel.cycles, true);
    json.push('}');

    json.push_str(",\"metrics\":{");
    json.push_str("\"total_transfer_bytes\":");
    json.push_str(&result.derived_metrics.total_transfer_bytes.to_string());
    json_optional_u64_field(
        &mut json,
        "logical_macs",
        record.metadata.logical_macs,
        true,
    );
    json_optional_u64_field(
        &mut json,
        "lowered_macs",
        record.metadata.lowered_macs,
        true,
    );
    json_optional_u64_field(
        &mut json,
        "wall_time_ms",
        record.metadata.wall_time_ms,
        true,
    );
    json.push('}');

    json.push_str(",\"event_kinds\":[");
    for (index, event) in result.events.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        json_string(&mut json, runtime_event_kind_as_str(event));
    }
    json.push(']');

    json.push_str(",\"artifacts\":[");
    for (index, artifact) in result.artifacts.iter().enumerate() {
        if index != 0 {
            json.push(',');
        }
        json.push('{');
        json_string_field(&mut json, "kind", artifact.kind.as_str());
        json.push_str(",\"path\":");
        json_string(&mut json, &artifact.path);
        json.push_str(",\"digest\":");
        match &artifact.digest {
            Some(digest) => {
                json.push('{');
                json_string_field(&mut json, "algorithm", digest.algorithm.as_str());
                json.push_str(",\"value\":");
                json_string(&mut json, &digest.value);
                json.push('}');
            }
            None => json.push_str("null"),
        }
        json.push('}');
    }
    json.push(']');
    json.push('}');
    json
}

fn render_target_contract_json(json: &mut String, target: TargetContract) {
    json.push('{');
    json.push_str("\"requested\":");
    render_target_json(json, target.requested);
    json.push_str(",\"observed\":");
    render_target_json(json, target.observed);
    json.push_str(",\"exact\":");
    json.push_str(bool_as_str(target.is_compatible()));
    json.push_str(",\"mismatch_mask\":");
    json.push_str(&target.compatibility.mismatch_mask().to_string());
    json.push_str(",\"mismatches\":[");
    let mut wrote = false;
    for capability in TargetCapability::ALL {
        if target.compatibility.mismatches(capability) {
            if wrote {
                json.push(',');
            }
            json_string(json, capability.as_str());
            wrote = true;
        }
    }
    json.push_str("]}");
}

fn render_target_json(json: &mut String, target: mandrel_target_ir::TargetSpec) {
    json.push('{');
    json_string_field(json, "name", target.name);
    json.push_str(",\"backend\":");
    json_string(json, target.backend_name());
    json.push_str(",\"xlen\":");
    json.push_str(&target.xlen.to_string());
    json.push_str(",\"max_workgroup_threads\":");
    json.push_str(&target.max_workgroup_threads.to_string());
    json.push_str(",\"preferred_subgroup_width\":");
    json.push_str(&target.preferred_subgroup_width.to_string());
    json.push_str(",\"local_memory_bytes\":");
    json.push_str(&target.local_memory_bytes.to_string());
    json.push_str(",\"supports_int8\":");
    json.push_str(bool_as_str(target.supports_int8));
    json.push_str(",\"supports_float32\":");
    json.push_str(bool_as_str(target.supports_float32));
    json.push_str(",\"supports_tensor_cores\":");
    json.push_str(bool_as_str(target.supports_tensor_cores));
    json.push_str(",\"supports_async_copy\":");
    json.push_str(bool_as_str(target.supports_async_copy));
    json.push('}');
}

fn render_experiment_csv(result: &ExperimentResult, record: AttentionRuntimeTraceRecord) -> String {
    let header = "schema,spec_id,status,evidence_class,workload,sequence,head_dim,hardware_design,vortex_revision,llvm_vortex_revision,kernel,lowering,requested_backend,observed_backend,target_exact,correctness_passed,correctness_mismatches,instructions,cycles,ipc,total_transfer_bytes,logical_macs,lowered_macs,wall_time_ms";
    let spec = record.experiment_spec;
    let workload = spec.map(|spec| spec.workload.name).unwrap_or("");
    let sequence = spec
        .map(|spec| spec.workload.attention.sequence.to_string())
        .unwrap_or_default();
    let head_dim = spec
        .map(|spec| spec.workload.attention.head_dim.to_string())
        .unwrap_or_default();
    let hardware_design = spec.map(|spec| spec.hardware.id).unwrap_or("");
    let correctness_passed = result
        .correctness
        .map(|correctness| correctness.passed.to_string())
        .unwrap_or_default();
    let correctness_mismatches = result
        .correctness
        .map(|correctness| correctness.mismatches.to_string())
        .unwrap_or_default();
    let instructions = optional_u64_csv(result.counters.kernel.instructions);
    let cycles = optional_u64_csv(result.counters.kernel.cycles);
    let ipc = match (
        result.counters.kernel.instructions,
        result.counters.kernel.cycles,
    ) {
        (Some(instructions), Some(cycles)) if cycles != 0 => {
            format!("{:.6}", instructions as f64 / cycles as f64)
        }
        _ => String::new(),
    };
    let fields = [
        "mandrel.experiment.result.v2".to_owned(),
        result.spec_id.to_owned(),
        experiment_status_as_str(result.status).to_owned(),
        "simx_observation".to_owned(),
        workload.to_owned(),
        sequence,
        head_dim,
        hardware_design.to_owned(),
        CURRENT_VORTEX_REVISION.to_owned(),
        CURRENT_LLVM_VORTEX_REVISION.to_owned(),
        record.summary.launch.kernel_symbol.to_owned(),
        "dense_scalar_two_pass_4x1x64".to_owned(),
        result.target.requested.backend_name().to_owned(),
        result.target.observed.backend_name().to_owned(),
        result.target.is_compatible().to_string(),
        correctness_passed,
        correctness_mismatches,
        instructions,
        cycles,
        ipc,
        result.derived_metrics.total_transfer_bytes.to_string(),
        optional_u64_csv(record.metadata.logical_macs),
        optional_u64_csv(record.metadata.lowered_macs),
        optional_u64_csv(record.metadata.wall_time_ms),
    ];
    let row = fields
        .iter()
        .map(|field| csv_field(field))
        .collect::<Vec<_>>()
        .join(",");
    format!("{header}\n{row}")
}

fn optional_u64_csv(value: Option<u64>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

fn json_string_field(json: &mut String, name: &str, value: &str) {
    json_string(json, name);
    json.push(':');
    json_string(json, value);
}

fn json_optional_u32_field(json: &mut String, name: &str, value: Option<u32>) {
    json.push(',');
    json_string(json, name);
    json.push(':');
    match value {
        Some(value) => json.push_str(&value.to_string()),
        None => json.push_str("null"),
    }
}

fn json_optional_u64_field(json: &mut String, name: &str, value: Option<u64>, leading_comma: bool) {
    if leading_comma {
        json.push(',');
    }
    json_string(json, name);
    json.push(':');
    match value {
        Some(value) => json.push_str(&value.to_string()),
        None => json.push_str("null"),
    }
}

fn json_string(json: &mut String, value: &str) {
    json.push('"');
    for character in value.chars() {
        match character {
            '"' => json.push_str("\\\""),
            '\\' => json.push_str("\\\\"),
            '\n' => json.push_str("\\n"),
            '\r' => json.push_str("\\r"),
            '\t' => json.push_str("\\t"),
            character if character.is_control() => json.push('?'),
            character => json.push(character),
        }
    }
    json.push('"');
}

const fn bool_as_str(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

const fn experiment_status_as_str(status: ExperimentStatus) -> &'static str {
    match status {
        ExperimentStatus::Planned => "planned",
        ExperimentStatus::Unsupported => "unsupported",
        ExperimentStatus::Succeeded => "succeeded",
        ExperimentStatus::Failed => "failed",
    }
}

const fn runtime_event_kind_as_str(event: &RuntimeEvent) -> &'static str {
    match event {
        RuntimeEvent::Allocate { .. } => "allocate",
        RuntimeEvent::Copy { .. } => "copy",
        RuntimeEvent::KernelLaunch(_) => "kernel_launch",
        RuntimeEvent::Sync { .. } => "sync",
        RuntimeEvent::PerfCounter(_) => "perf_counter",
        RuntimeEvent::CorrectnessCheck(_) => "correctness_check",
    }
}

#[cfg(test)]
mod tests {
    use super::{render_experiment_csv, render_experiment_json};
    use crate::attention::trace::{
        AttentionRuntimeTraceEnrichment, collect_attention_runtime_trace_with_enrichment,
    };
    use mandrel_experiment::ExperimentSpec;

    const TRACE: &str = "PERF: instrs=165144, cycles=414598, IPC=0.398\nattention.runtime: compare summary elements=128 mismatches=0 status=exact\nMANDREL_RUNTIME_TRACE: kernel=attention_prefill_i8 runtime_sequence=8 runtime_head_dim=16 query_tile=4 key_tile=1 compiled_sequence=64 compiled_head_dim=64 head_dim_tile=64 logical_macs=2048 lowered_macs=33792 estimated_global_bytes_read=66560 estimated_global_bytes_written=128 estimated_local_memory_bytes_per_workgroup=0 requested_target_backend=vortex_simx requested_target_xlen=64 requested_target_max_workgroup_threads=16 requested_target_preferred_subgroup_width=4 requested_target_local_memory_bytes=16384 requested_target_supports_int8=true requested_target_supports_float32=true requested_target_supports_tensor_cores=false requested_target_supports_async_copy=false observed_target_backend=vortex_simx observed_target_xlen=64 observed_target_preferred_subgroup_width=4 observed_target_supports_int8=true observed_target_supports_float32=true observed_target_supports_tensor_cores=false observed_target_supports_async_copy=false target_compatible=true target_mismatch_mask=0 target_threads_per_warp=4 target_warps_per_core=4 target_max_workgroup_threads=16 target_local_memory_bytes=16384 workgroup_count=16 threads_per_workgroup=16 total_threads=256 runtime_q_elements=128 runtime_kv_elements=256 runtime_output_elements=128 runtime_q_bytes=128 runtime_kv_bytes=256 runtime_output_bytes=128 module_cache_hit=false kernel_cache_hit=false grid=16x1x1 block=4x4x1 shared_memory_bytes=0 host_to_device_bytes=384 device_to_host_bytes=128\n";

    #[test]
    fn renders_json_and_csv_without_automatic_comparison() {
        let spec = ExperimentSpec::attention_prefill_i8_vortex_smoke(8, 16);
        let report = collect_attention_runtime_trace_with_enrichment(
            TRACE,
            AttentionRuntimeTraceEnrichment::with_experiment_spec(123, spec),
        );
        let record = match report.record {
            Some(record) => record,
            None => panic!("expected structured trace"),
        };
        let result = match super::attention_experiment_result_from_record(record) {
            Some(result) => result,
            None => panic!("expected experiment result"),
        };

        let json = render_experiment_json(&result, record);
        let csv = render_experiment_csv(&result, record);

        assert!(json.contains("\"schema\":\"mandrel.experiment.result.v2\""));
        assert!(json.contains("\"evidence_class\":\"simx_observation\""));
        assert!(!json.contains("baseline"));
        assert!(csv.starts_with("schema,spec_id,status,evidence_class"));
        assert!(csv.contains("simx_observation"));
        assert!(!csv.contains("baseline"));
    }
}
