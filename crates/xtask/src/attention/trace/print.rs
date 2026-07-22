use super::*;

pub(crate) fn print_attention_runtime_trace_report(report: AttentionRuntimeTraceReport) {
    let Some(record) = report.record else {
        println!(
            "attention.runtime counters: instructions={} cycles={} ipc={} wall_time_ms={}",
            format_optional_u64(report.counters.instructions),
            format_optional_u64(report.counters.cycles),
            format_ipc(report.counters),
            format_optional_u64(report.metadata.wall_time_ms)
        );
        return;
    };

    print_attention_runtime_trace_record(record);
    if let Some(result) = attention_experiment_result_from_record(record) {
        print_attention_experiment_result(&result);
    }
}

pub(super) fn print_attention_runtime_trace_record(record: AttentionRuntimeTraceRecord) {
    let summary = record.summary;
    println!("attention.runtime trace summary:");
    println!("  kernel: {}", summary.launch.kernel_symbol);
    println!(
        "  runtime: sequence={} head_dim={} query_tile={} key_tile={}",
        format_optional_u32(record.metadata.runtime_sequence),
        format_optional_u32(record.metadata.runtime_head_dim),
        format_optional_u32(record.metadata.query_tile),
        format_optional_u32(record.metadata.key_tile)
    );
    println!(
        "  compiled: sequence={} head_dim={} head_dim_tile={}",
        format_optional_u32(record.metadata.compiled_sequence),
        format_optional_u32(record.metadata.compiled_head_dim),
        format_optional_u32(record.metadata.head_dim_tile)
    );
    println!(
        "  requested_target: backend={} xlen={} max_workgroup_threads={} preferred_subgroup_width={} local_memory_bytes={}",
        format_optional_device_backend(record.metadata.requested_target_backend),
        format_optional_u32(record.metadata.requested_target_xlen),
        format_optional_u32(record.metadata.requested_target_max_workgroup_threads),
        format_optional_u32(record.metadata.requested_target_preferred_subgroup_width),
        format_optional_u32(record.metadata.requested_target_local_memory_bytes)
    );
    println!(
        "  observed_target: backend={} xlen={} threads_per_warp={} warps_per_core={} max_workgroup_threads={} preferred_subgroup_width={} local_memory_bytes={}",
        format_optional_device_backend(record.metadata.observed_target_backend),
        format_optional_u32(record.metadata.observed_target_xlen),
        format_optional_u64(record.metadata.target_threads_per_warp),
        format_optional_u64(record.metadata.target_warps_per_core),
        format_optional_u64(record.metadata.target_max_workgroup_threads),
        format_optional_u32(record.metadata.observed_target_preferred_subgroup_width),
        format_optional_u64(record.metadata.target_local_memory_bytes)
    );
    println!(
        "  target_compatibility: exact={} mismatch_mask={}",
        format_optional_bool(record.metadata.target_compatible),
        format_optional_u16(record.metadata.target_mismatch_mask)
    );
    println!(
        "  launch: grid=({}, {}, {}) block=({}, {}, {}) shared_memory_bytes={}",
        summary.launch.grid[0],
        summary.launch.grid[1],
        summary.launch.grid[2],
        summary.launch.block[0],
        summary.launch.block[1],
        summary.launch.block[2],
        summary.launch.shared_memory_bytes
    );
    println!(
        "  execution: workgroups={} threads_per_workgroup={} total_threads={} module_cache_hit={} kernel_cache_hit={}",
        format_optional_u64(record_workgroup_count(record)),
        format_optional_u64(record_threads_per_workgroup(record)),
        format_optional_u64(record_total_threads(record)),
        format_optional_bool(record.metadata.module_cache_hit),
        format_optional_bool(record.metadata.kernel_cache_hit)
    );
    println!(
        "  workload: logical_macs={} lowered_macs={} q_elements={} kv_elements={} output_elements={}",
        format_optional_u64(record_logical_macs(record)),
        format_optional_u64(record.metadata.lowered_macs),
        format_optional_u64(record.metadata.runtime_q_elements),
        format_optional_u64(record.metadata.runtime_kv_elements),
        format_optional_u64(record_output_elements(record))
    );
    println!(
        "  lowering_memory_estimate: global_read_bytes={} global_write_bytes={} local_bytes_per_workgroup={}",
        format_optional_u64(record.metadata.estimated_global_bytes_read),
        format_optional_u64(record.metadata.estimated_global_bytes_written),
        format_optional_u64(record.metadata.estimated_local_memory_bytes_per_workgroup)
    );
    println!(
        "  memory: q_bytes={} kv_bytes={} output_bytes={} host_to_device_bytes={} device_to_host_bytes={} total_transfer_bytes={}",
        format_optional_u64(record.metadata.runtime_q_bytes),
        format_optional_u64(record.metadata.runtime_kv_bytes),
        format_optional_u64(record.metadata.runtime_output_bytes),
        summary.host_to_device_bytes,
        summary.device_to_host_bytes,
        summary.total_transfer_bytes()
    );
    println!(
        "\nPerformance counter stats for '{}':",
        summary.launch.kernel_symbol
    );
    println!(
        "  {:>16}      instructions        # Verilator RTL counter",
        format_optional_u64(summary.counters.instructions)
    );
    println!(
        "  {:>16}      cycles              # Verilator RTL counter",
        format_optional_u64(summary.counters.cycles)
    );
    println!(
        "  {:>16}      instructions/cycle  # derived",
        format_ipc(summary.counters)
    );
    println!(
        "  {:>16}      transfer bytes      # runtime events",
        summary.total_transfer_bytes()
    );
    println!(
        "  derived: instrs/output={} cycles/output={} transfer_bytes/output={} cycles/logical_mac={} logical_macs/cycle={} cycles/lowered_mac={} lowered_macs/cycle={} wall_time_ms={}",
        format_optional_f64(record_instructions_per_output(record)),
        format_optional_f64(record_cycles_per_output(record)),
        format_optional_f64(record_transfer_bytes_per_output(record)),
        format_optional_f64(record_cycles_per_logical_mac(record)),
        format_optional_f64(record_logical_macs_per_cycle(record)),
        format_optional_f64(record_cycles_per_lowered_mac(record)),
        format_optional_f64(record_lowered_macs_per_cycle(record)),
        format_optional_u64(record.metadata.wall_time_ms)
    );
}

fn print_attention_experiment_result(result: &ExperimentResult) {
    println!("attention.experiment result:");
    println!("  spec_id: {}", result.spec_id);
    println!("  status: {:?}", result.status);
    println!(
        "  target: requested={} observed={} compatible={} mismatch_mask={}",
        result.target.requested.name,
        result.target.observed.name,
        result.target.is_compatible(),
        result.target.compatibility.mismatch_mask()
    );
    println!("  events: {}", result.events.len());
    println!(
        "  total_transfer_bytes: {}",
        result.derived_metrics.total_transfer_bytes
    );
    if let Some(correctness) = result.correctness {
        println!(
            "  correctness: passed={} compared_elements={} mismatches={}",
            correctness.passed, correctness.compared_elements, correctness.mismatches
        );
    }
}

pub(super) fn format_optional_device_backend(value: Option<DeviceBackend>) -> &'static str {
    match value {
        Some(DeviceBackend::HostReference) => "host_reference",
        Some(DeviceBackend::VortexRtl) => "vortex_rtl",
        Some(DeviceBackend::VortexFpga) => "vortex_fpga",
        None => "<missing>",
    }
}

pub(super) fn format_optional_u16(value: Option<u16>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => "<missing>".to_owned(),
    }
}

pub(super) fn format_optional_u64(value: Option<u64>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => "<missing>".to_owned(),
    }
}

pub(super) fn format_optional_u32(value: Option<u32>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => "<missing>".to_owned(),
    }
}

pub(super) fn format_optional_bool(value: Option<bool>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => "<missing>".to_owned(),
    }
}

pub(super) fn format_optional_f64(value: Option<f64>) -> String {
    match value {
        Some(value) if value.is_finite() => format!("{value:.3}"),
        Some(_) => "n/a".to_owned(),
        None => "<missing>".to_owned(),
    }
}

pub(super) fn format_ipc(counters: KernelCounterTrace) -> String {
    format_optional_f64(counter_ratio(counters.instructions, counters.cycles))
}

fn record_instructions_per_output(record: AttentionRuntimeTraceRecord) -> Option<f64> {
    counter_ratio(
        record.summary.counters.instructions,
        record_output_elements(record),
    )
}

fn record_cycles_per_output(record: AttentionRuntimeTraceRecord) -> Option<f64> {
    counter_ratio(
        record.summary.counters.cycles,
        record_output_elements(record),
    )
}

fn record_transfer_bytes_per_output(record: AttentionRuntimeTraceRecord) -> Option<f64> {
    counter_ratio(
        record_total_transfer_bytes(record),
        record_output_elements(record),
    )
}

fn record_cycles_per_logical_mac(record: AttentionRuntimeTraceRecord) -> Option<f64> {
    counter_ratio(record.summary.counters.cycles, record_logical_macs(record))
}

fn record_logical_macs_per_cycle(record: AttentionRuntimeTraceRecord) -> Option<f64> {
    counter_ratio(record_logical_macs(record), record.summary.counters.cycles)
}

fn record_cycles_per_lowered_mac(record: AttentionRuntimeTraceRecord) -> Option<f64> {
    counter_ratio(record.summary.counters.cycles, record.metadata.lowered_macs)
}

fn record_lowered_macs_per_cycle(record: AttentionRuntimeTraceRecord) -> Option<f64> {
    counter_ratio(record.metadata.lowered_macs, record.summary.counters.cycles)
}

fn counter_ratio(numerator: Option<u64>, denominator: Option<u64>) -> Option<f64> {
    match (numerator, denominator) {
        (Some(_), Some(0)) => None,
        (Some(numerator), Some(denominator)) => Some(numerator as f64 / denominator as f64),
        _ => None,
    }
}

fn record_output_elements(record: AttentionRuntimeTraceRecord) -> Option<u64> {
    record.metadata.runtime_output_elements.or_else(|| {
        checked_mul_option_u64(
            record.metadata.runtime_sequence.map(u64::from),
            record.metadata.runtime_head_dim.map(u64::from),
        )
    })
}

fn record_logical_macs(record: AttentionRuntimeTraceRecord) -> Option<u64> {
    record.metadata.logical_macs.or_else(|| {
        let sequence = record.metadata.runtime_sequence.map(u64::from)?;
        let head_dim = record.metadata.runtime_head_dim.map(u64::from)?;
        let score_macs = sequence.checked_mul(sequence)?.checked_mul(head_dim)?;
        score_macs.checked_mul(2)
    })
}

fn record_workgroup_count(record: AttentionRuntimeTraceRecord) -> Option<u64> {
    record
        .metadata
        .workgroup_count
        .or_else(|| product3_u32(record.summary.launch.grid))
}

fn record_threads_per_workgroup(record: AttentionRuntimeTraceRecord) -> Option<u64> {
    record
        .metadata
        .threads_per_workgroup
        .or_else(|| product3_u32(record.summary.launch.block))
}

fn record_total_threads(record: AttentionRuntimeTraceRecord) -> Option<u64> {
    record.metadata.total_threads.or_else(|| {
        record_workgroup_count(record)?.checked_mul(record_threads_per_workgroup(record)?)
    })
}

fn record_total_transfer_bytes(record: AttentionRuntimeTraceRecord) -> Option<u64> {
    u64::try_from(record.summary.total_transfer_bytes()).ok()
}

fn product3_u32(values: [u32; 3]) -> Option<u64> {
    u64::from(values[0])
        .checked_mul(u64::from(values[1]))?
        .checked_mul(u64::from(values[2]))
}

fn checked_mul_option_u64(lhs: Option<u64>, rhs: Option<u64>) -> Option<u64> {
    lhs?.checked_mul(rhs?)
}
