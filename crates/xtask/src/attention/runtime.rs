use std::env;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use mandrel_vortex_backend::{
    VortexBackend, VortexBackendConfig, VortexConfig, reference_attention_prefill_i8,
    validate_attention_prefill_i8_plan_abi,
};

use crate::command::run_checked_capturing_stdout;
use crate::vortex::{
    apply_vortex_env, ensure_vortex_runtime_libraries, preferred_vortex_runtime_library,
    require_file,
};
use crate::{Result, XtaskError};

use super::artifacts::generate_vortex_attention_artifacts;
use super::compare::compare_attention_outputs;
use super::input::{attention_runtime_flag, deterministic_attention_prefill_input};
use super::metrics::{
    AttentionRuntimeWorkload, attention_i8_checksum, attention_i8_max_abs,
    format_mandrel_runtime_trace_line,
};
use super::plan::current_attention_prefill_plan;
use super::trace;

pub(crate) fn run_vortex_attention_correctness(workspace_root: &Path) -> Result<()> {
    let config = VortexConfig::from_env(workspace_root)?;
    let artifacts = generate_vortex_attention_artifacts(workspace_root, false)?;
    require_file(&artifacts.vxbin_path, "generated attention vxbin")?;
    ensure_vortex_runtime_libraries(&config)?;
    let runtime = preferred_vortex_runtime_library(&config)?;

    println!(
        "Launching attention runtime correctness through Vortex simx with vxbin: {}",
        artifacts.vxbin_path.display()
    );
    let exe = env::current_exe()
        .map_err(|error| format!("failed to locate current xtask executable: {error}"))?;
    let mut command = Command::new(exe);
    command
        .current_dir(workspace_root)
        .arg("__vortex-run-attention-inner")
        .arg(&artifacts.vxbin_path)
        .env("VORTEX_DRIVER", "simx")
        .env("MANDREL_VORTEX_RUNTIME_LIB", &runtime)
        .env("MANDREL_VORTEX_RUNTIME_TRACE", "1");
    apply_vortex_env(&mut command, &config)?;
    let started = Instant::now();
    let stdout = run_checked_capturing_stdout(command, "attention.runtime_correctness")?;
    let wall_time_ms = elapsed_millis_saturating(started.elapsed());
    println!("attention.runtime: outer command wall_time_ms={wall_time_ms}");
    let report = trace::collect_attention_runtime_trace_with_enrichment(
        &stdout,
        trace::AttentionRuntimeTraceEnrichment::with_wall_time_ms(wall_time_ms),
    );
    trace::print_attention_runtime_trace_report(report);
    let trace_jsonl_path = artifacts.vxbin_path.with_extension("trace.jsonl");
    if trace::append_attention_runtime_trace_jsonl(report, &trace_jsonl_path).map_err(|error| {
        XtaskError::message(format!(
            "failed to write attention runtime trace '{}': {error}",
            trace_jsonl_path.display()
        ))
    })? {
        println!(
            "attention.runtime trace jsonl: {}",
            trace_jsonl_path.display()
        );
    }
    Ok(())
}

pub(crate) fn print_attention_trace_history(workspace_root: &Path) -> Result<()> {
    let path = trace::default_attention_trace_jsonl_path(workspace_root);
    trace::print_attention_trace_history(&path, 5).map_err(|error| {
        XtaskError::message(format!(
            "failed to read attention runtime trace history '{}': {error}",
            path.display()
        ))
    })
}

pub(crate) fn run_vortex_attention_correctness_inner(
    workspace_root: &Path,
    vxbin_path: PathBuf,
) -> Result<()> {
    require_file(&vxbin_path, "generated attention vxbin")?;

    runtime_step("compiling attention launch plan")?;
    let plan = current_attention_prefill_plan()?;
    runtime_step("validating attention ABI/layout metadata")?;
    validate_attention_prefill_i8_plan_abi(&plan)?;
    runtime_step("building deterministic attention input")?;
    let input = deterministic_attention_prefill_input(&plan)?;
    let runtime_shape = plan.metadata.runtime_shape;
    runtime_step(&format!(
        "runtime shape compiled={}x{} default={}x{} actual={}x{} tile(query={}, key={}, head_dim={})",
        runtime_shape.compiled_sequence,
        runtime_shape.compiled_head_dim,
        runtime_shape.default_runtime_sequence,
        runtime_shape.default_runtime_head_dim,
        input.sequence,
        input.head_dim,
        input.query_tile,
        input.key_tile,
        runtime_shape.head_dim_tile
    ))?;
    runtime_step(&format!(
        "input buffers q={}B/{}el k={}B/{}el v={}B/{}el",
        input.q.len(),
        input.q.len(),
        input.k.len(),
        input.k.len(),
        input.v.len(),
        input.v.len()
    ))?;
    runtime_step(&format!(
        "computing host reference for sequence={} head_dim={}",
        input.sequence, input.head_dim
    ))?;
    let expected = reference_attention_prefill_i8(&input)
        .map_err(|error| format!("failed to compute host attention reference: {error}"))?;
    runtime_step(&format!(
        "host reference output elements={} bytes={} checksum={} max_abs={}",
        expected.len(),
        expected.len(),
        attention_i8_checksum(&expected),
        attention_i8_max_abs(&expected)
    ))?;

    let mut runtime_launch = plan.launch;
    if attention_runtime_flag("MANDREL_ATTENTION_RUNTIME_SCALAR_LAUNCH")? {
        runtime_step("using scalar launch override: grid=(1,1,1) block=(1,1,1) shared=0")?;
        runtime_launch.grid.x = 1;
        runtime_launch.grid.y = 1;
        runtime_launch.grid.z = 1;
        runtime_launch.block.x = 1;
        runtime_launch.block.y = 1;
        runtime_launch.block.z = 1;
        runtime_launch.shared_memory_bytes = 0;
    }
    runtime_step(&format!(
        "runtime launch dims grid=({}, {}, {}) block=({}, {}, {}) shared={}",
        runtime_launch.grid.x,
        runtime_launch.grid.y,
        runtime_launch.grid.z,
        runtime_launch.block.x,
        runtime_launch.block.y,
        runtime_launch.block.z,
        runtime_launch.shared_memory_bytes
    ))?;

    runtime_step("initializing Vortex backend/runtime")?;
    let config =
        VortexBackendConfig::new().with_kernel_artifact(runtime_launch.symbol, &vxbin_path);
    let mut backend = VortexBackend::new(config)
        .map_err(|error| format!("failed to initialize Vortex backend runtime: {error}"))?;
    runtime_step("launching Vortex attention kernel and reading output")?;
    let actual = backend
        .run_attention_prefill_i8(&runtime_launch, &input)
        .map_err(|error| format!("failed to run attention prefill on Vortex: {error}"))?;
    let workload =
        AttentionRuntimeWorkload::from_runtime(&plan, &input, actual.trace, actual.output.len())?;
    runtime_step(&format!(
        "backend cache module_hit={} kernel_hit={}",
        actual.trace.module_cache_hit, actual.trace.kernel_cache_hit
    ))?;
    runtime_step(&format!(
        "backend transfers host_to_device={}B device_to_host={}B total={}B",
        actual.trace.host_to_device_bytes,
        actual.trace.device_to_host_bytes,
        format_optional_u64(actual.trace.total_transfer_bytes())
    ))?;
    runtime_step(&format!(
        "execution shape workgroups={} threads_per_workgroup={} total_threads={}",
        workload.workgroup_count, workload.threads_per_workgroup, workload.total_threads
    ))?;
    runtime_step(&format!(
        "Vortex output elements={} bytes={} checksum={} max_abs={}",
        actual.output.len(),
        actual.output.len(),
        attention_i8_checksum(&actual.output),
        attention_i8_max_abs(&actual.output)
    ))?;

    runtime_step("comparing Vortex output against host reference")?;
    let comparison = compare_attention_outputs(&expected, &actual.output)?;
    runtime_step(&format!(
        "compare summary elements={} mismatches={} status=exact",
        comparison.elements, comparison.mismatches
    ))?;
    println!("attention runtime correctness PASSED");
    println!("  vxbin: {}", vxbin_path.display());
    println!("  runtime root: {}", workspace_root.display());
    println!("  sequence: {}", input.sequence);
    println!("  head_dim: {}", input.head_dim);
    println!("  query_tile: {}", input.query_tile);
    println!("  key_tile: {}", input.key_tile);
    println!("  logical_macs: {}", workload.logical_macs);
    println!(
        "  execution: workgroups={} threads_per_workgroup={} total_threads={}",
        workload.workgroup_count, workload.threads_per_workgroup, workload.total_threads
    );
    println!(
        "  runtime bytes: q={} kv={} output={}",
        workload.runtime_q_bytes, workload.runtime_kv_bytes, workload.runtime_output_bytes
    );
    println!("  trace: {:?}", actual.trace);
    println!(
        "{}",
        format_mandrel_runtime_trace_line(actual.trace, workload)
    );
    Ok(())
}

fn runtime_step(message: &str) -> Result<()> {
    println!("attention.runtime: {message}");
    io::stdout().flush().map_err(|error| {
        XtaskError::message(format!("failed to flush runtime progress output: {error}"))
    })
}

fn elapsed_millis_saturating(duration: Duration) -> u64 {
    match u64::try_from(duration.as_millis()) {
        Ok(value) => value,
        Err(_) => u64::MAX,
    }
}

fn format_optional_u64(value: Option<u64>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => "<overflow>".to_owned(),
    }
}
