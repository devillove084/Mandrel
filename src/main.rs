use core::fmt;

use mandrel_compiler::{VortexAttentionPrefillPlan, compile_vortex_attention_prefill_kernel};
use mandrel_ggml_adapter::{GgmlAttentionPrefillRequest, backend_name};
use mandrel_kernels::reference;
use mandrel_model_ir::{AttentionOp, SoftmaxOp};
use mandrel_profiler::KernelMetrics;
use mandrel_runtime::example_attention_vortex_plan;
use mandrel_target_ir::DeviceCapabilities;

const SOFTMAX_ROWS: usize = 2;
const SOFTMAX_COLS: usize = 4;
const SOFTMAX_LEN: usize = SOFTMAX_ROWS * SOFTMAX_COLS;
const SOFTMAX_INPUT: [f32; SOFTMAX_LEN] = [1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0];

fn main() {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "overview".to_owned());

    match command.as_str() {
        "overview" | "--help" | "-h" => print_overview(),
        "plan" => print_example_runtime_plan(),
        "attention-plan" | "compile-plan" => print_compile_attention_plan(),
        "attention-probe" | "ggml-probe" => print_attention_probe(),
        "softmax" => run_softmax_command(),
        other => {
            eprintln!("unknown command: {other}");
            eprintln!(
                "try: cargo run -- overview | plan | attention-plan | attention-probe | softmax"
            );
            std::process::exit(2);
        }
    }
}

fn print_overview() {
    println!(
        "mandrel workspace\n\n\
         focused goal:\n\
           - Rust-first RISC-V GPGPU stack co-design\n\
           - primary executable backend target: Vortex simx/RTL path\n\
           - MLIR-first generated device code\n\
           - near-term operators: attention prefill, online softmax, reductions, KV layout planning\n\n\
         useful commands:\n\
           cargo run -- plan\n\
           cargo run -- attention-plan\n\
           cargo run -- attention-probe\n\
           cargo run -- softmax\n\
           cargo vortex-status\n\
           cargo vortex-plan-attention\n\
           cargo xtask help"
    );
}

fn print_example_runtime_plan() {
    let plan = example_attention_vortex_plan();
    println!("runtime target: {:?}", plan.target);
    println!("commands: {}", plan.command_buffer.len());
    println!("plan: {plan:?}");
}

fn print_compile_attention_plan() {
    let op = AttentionOp::prefill_i8_demo();
    let plan = match compile_vortex_attention_prefill_kernel(op) {
        Ok(plan) => plan,
        Err(error) => {
            eprintln!("failed to compile Vortex attention prefill plan: {error:?}");
            std::process::exit(1);
        }
    };

    print_vortex_attention_plan("Vortex attention-prefill compile plan", &plan);
}

fn print_attention_probe() {
    let request = GgmlAttentionPrefillRequest::dense_i8(64, 64);
    let caps = DeviceCapabilities::vortex_simx_default();
    let accepted = mandrel_ggml_adapter::can_offload_attention_prefill(request, caps);

    println!("adapter backend: {}", backend_name());
    println!("probe op: dense attention prefill i8, sequence=64, head_dim=64");
    println!("can offload to Vortex caps: {accepted}");
    if let Some(op) = request.to_model_ir_for(caps) {
        println!(
            "lowered model-ir attention: sequence={} head_dim={}",
            op.shape.sequence, op.shape.head_dim
        );
    }
}

fn print_vortex_attention_plan(title: &str, plan: &VortexAttentionPrefillPlan) {
    println!("{title}");
    println!(
        "shape: sequence={} head_dim={}",
        plan.op.shape.sequence, plan.op.shape.head_dim
    );
    println!(
        "kernel: {} ({:?}, {:?})",
        plan.kernel.symbol.as_str(),
        plan.kernel.domain,
        plan.kernel.availability
    );
    println!(
        "tile: query={} key={} head_dim={}",
        plan.schedule.tile.query, plan.schedule.tile.key, plan.schedule.tile.head_dim
    );
    println!(
        "layout: {:?}, softmax: {:?}",
        plan.schedule.kv_layout, plan.schedule.softmax
    );
    println!(
        "launch: kernel={} grid=({}, {}, {}) block=({}, {}, {}) shared_memory_bytes={}",
        plan.launch.symbol.as_str(),
        plan.launch.grid.x,
        plan.launch.grid.y,
        plan.launch.grid.z,
        plan.launch.block.x,
        plan.launch.block.y,
        plan.launch.block.z,
        plan.launch.shared_memory_bytes
    );
    println!("args:");
    for arg in &plan.launch.args {
        println!("  {}: {:?}", arg.index, arg.value);
    }
    print_metrics(plan.metrics);
}

fn print_metrics(metrics: KernelMetrics) {
    println!("metrics:");
    println!("  logical_macs: {}", metrics.logical_macs);
    println!("  lowered_macs: {}", metrics.lowered_macs);
    println!("  kernel_launches: {}", metrics.kernel_launches);
    println!("  workgroup_count: {}", metrics.workgroup_count);
    println!("  thread_count: {}", metrics.thread_count);
    println!("  global_bytes_read: {}", metrics.global_bytes_read);
    println!("  global_bytes_written: {}", metrics.global_bytes_written);
    println!(
        "  local_memory_bytes_per_workgroup: {}",
        metrics.local_memory_bytes_per_workgroup
    );
    if let Some(intensity) = metrics.logical_operational_intensity() {
        println!(
            "  logical_operational_intensity_macs_per_byte: {}/{}",
            intensity.numerator, intensity.denominator
        );
    }
    if let Some(intensity) = metrics.lowered_operational_intensity() {
        println!(
            "  lowered_operational_intensity_macs_per_byte: {}/{}",
            intensity.numerator, intensity.denominator
        );
    }
}

#[derive(Debug)]
struct SoftmaxSmokeReport {
    op: SoftmaxOp,
    output: [f32; SOFTMAX_LEN],
}

#[derive(Debug)]
enum SoftmaxSmokeError {
    RowSumMismatch { row: usize, sum: f32 },
}

impl fmt::Display for SoftmaxSmokeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RowSumMismatch { row, sum } => {
                write!(formatter, "softmax row {row} sums to {sum}, expected 1.0")
            }
        }
    }
}

impl std::error::Error for SoftmaxSmokeError {}

fn run_softmax_reference() -> Result<SoftmaxSmokeReport, SoftmaxSmokeError> {
    let op = SoftmaxOp::row_f32_demo();
    let mut output = [0.0_f32; SOFTMAX_LEN];
    reference::row_softmax_f32(&SOFTMAX_INPUT, &mut output, SOFTMAX_ROWS, SOFTMAX_COLS);

    for row in 0..SOFTMAX_ROWS {
        let start = row * SOFTMAX_COLS;
        let end = start + SOFTMAX_COLS;
        let sum: f32 = output[start..end].iter().sum();
        if (sum - 1.0).abs() > 1.0e-5 {
            return Err(SoftmaxSmokeError::RowSumMismatch { row, sum });
        }
    }

    Ok(SoftmaxSmokeReport { op, output })
}

fn run_softmax_command() {
    match run_softmax_reference() {
        Ok(report) => {
            println!("row softmax reference passed");
            println!(
                "op: rows={} cols={} axis={:?}",
                report.op.shape.rows, report.op.shape.cols, report.op.axis
            );
            println!("output: {:?}", report.output);
        }
        Err(error) => {
            eprintln!("row softmax reference failed: {error}");
            std::process::exit(1);
        }
    }
}
