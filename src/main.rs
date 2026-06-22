use core::fmt;

use mandrel_compiler::{VortexMatmulPlan, compile_vortex_matmul_kernel};
use mandrel_ggml_adapter::{GgmlMulMatRequest, backend_name};
use mandrel_kernels::reference;
use mandrel_model_ir::MatmulOp;
use mandrel_profiler::KernelMetrics;
use mandrel_runtime::example_vortex_plan;
use mandrel_vortex_backend::vortex_capabilities;

const M: usize = 2;
const K: usize = 4;
const N: usize = 3;
const LHS_LEN: usize = M * K;
const RHS_LEN: usize = K * N;
const OUT_LEN: usize = M * N;

const LHS: [i8; LHS_LEN] = [1, 2, 3, 4, -1, 0, 2, 1];
const RHS: [i8; RHS_LEN] = [1, 0, 2, -1, 3, 1, 2, 1, 0, 0, -2, 1];
const EXPECTED: [i32; OUT_LEN] = [5, 1, 8, 3, 0, -1];

fn main() {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "overview".to_owned());

    match command.as_str() {
        "overview" | "--help" | "-h" => print_overview(),
        "plan" => print_example_runtime_plan(),
        "compile-plan" | "compile-matmul-plan" => print_compile_matmul_plan(),
        "ggml-probe" => print_ggml_probe(),
        "matmul" => {
            let mode_name = args.next().unwrap_or_else(|| "reference".to_owned());
            run_matmul_command(&mode_name);
        }
        other => {
            eprintln!("unknown command: {other}");
            eprintln!(
                "try: cargo run -- overview | plan | compile-plan | ggml-probe | matmul reference"
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
           - product route B: custom ggml/Vortex backend, not OpenCL-first\n\
           - host scalar path is kept only as the correctness reference\n\n\
         useful commands:\n\
           cargo run -- plan\n\
           cargo run -- compile-plan\n\
           cargo run -- ggml-probe\n\
           cargo run -- matmul reference\n\
           cargo vortex-status\n\
           cargo vortex-install\n\
           cargo vortex-plan-matmul\n\
           cargo vortex-run-vecadd\n\
           cargo xtask help"
    );
}

fn print_example_runtime_plan() {
    let plan = example_vortex_plan();

    println!("example runtime target: {:?}", plan.target);
    println!("example command count: {}", plan.command_buffer.len());
    println!("active RISC-V GPGPU backend target: Vortex simx");
}

fn print_compile_matmul_plan() {
    let op = MatmulOp::ggml_mul_mat_i8_demo();
    let plan = match compile_vortex_matmul_kernel(op) {
        Ok(plan) => plan,
        Err(error) => {
            eprintln!("failed to compile Vortex matmul plan: {error:?}");
            std::process::exit(1);
        }
    };

    print_vortex_matmul_plan(
        "Vortex custom-backend compile plan for ggml-like MUL_MAT",
        &plan,
    );
}

fn print_ggml_probe() {
    let request = GgmlMulMatRequest::i8_i32(32, 32, 64);
    let caps = vortex_capabilities();
    let accepted = mandrel_ggml_adapter::can_offload_mul_mat(request, caps);

    println!("ggml adapter backend: {}", backend_name());
    println!("probe op: MUL_MAT i8*i8 -> i32, shape M=32 N=32 K=64");
    println!("can offload to Vortex caps: {accepted}");
    if let Some(op) = request.to_model_ir_for(caps) {
        println!(
            "lowered model-ir matmul: M={} N={} K={}",
            op.shape.m, op.shape.n, op.shape.k
        );
    }
}

fn print_vortex_matmul_plan(title: &str, plan: &VortexMatmulPlan) {
    println!("{title}");
    println!(
        "shape: M={} N={} K={}",
        plan.op.shape.m, plan.op.shape.n, plan.op.shape.k
    );
    println!(
        "kernel: {} ({:?}, {:?})",
        plan.kernel.symbol.as_str(),
        plan.kernel.domain,
        plan.kernel.availability
    );
    println!(
        "tile: M={} N={} K={}",
        plan.schedule.tile.m, plan.schedule.tile.n, plan.schedule.tile.k
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
    for arg in plan.launch.args {
        println!("  {}: {:?}", arg.index, arg.value);
    }
    print_kernel_metrics("metrics", &plan.metrics);
}

fn print_kernel_metrics(title: &str, metrics: &KernelMetrics) {
    println!("{title}:");
    println!("  logical_macs: {}", metrics.logical_macs);
    println!("  scheduled_macs: {}", metrics.scheduled_macs);
    println!("  kernel_launches: {}", metrics.kernel_launches);
    println!("  workgroup_count: {}", metrics.workgroup_count);
    println!("  thread_count: {}", metrics.thread_count);
    println!("  global_bytes_read: {}", metrics.global_bytes_read);
    println!("  global_bytes_written: {}", metrics.global_bytes_written);
    println!(
        "  local_memory_bytes_per_workgroup: {}",
        metrics.local_memory_bytes_per_workgroup
    );
    if let Some(intensity) = (*metrics).operational_intensity() {
        println!(
            "  operational_intensity_macs_per_byte: {}/{}",
            intensity.numerator, intensity.denominator
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatmulMode {
    Reference,
}

impl MatmulMode {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "reference" | "ref" => Some(Self::Reference),
            _ => None,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Reference => "reference",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MatmulSmokeReport {
    mode: MatmulMode,
    executed_path: &'static str,
    m: usize,
    k: usize,
    n: usize,
    output: [i32; OUT_LEN],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatmulSmokeError {
    Mismatch {
        expected: [i32; OUT_LEN],
        actual: [i32; OUT_LEN],
    },
}

impl fmt::Display for MatmulSmokeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mismatch { expected, actual } => write!(
                formatter,
                "matmul mismatch: expected {expected:?}, actual {actual:?}"
            ),
        }
    }
}

impl std::error::Error for MatmulSmokeError {}

fn run_matmul_smoke(mode: MatmulMode) -> Result<MatmulSmokeReport, MatmulSmokeError> {
    let mut output = [0_i32; OUT_LEN];
    let executed_path = match mode {
        MatmulMode::Reference => {
            reference::matmul_i8_i32(&LHS, &RHS, &mut output, M, K, N);
            "portable scalar reference kernel"
        }
    };

    if output != EXPECTED {
        return Err(MatmulSmokeError::Mismatch {
            expected: EXPECTED,
            actual: output,
        });
    }

    Ok(MatmulSmokeReport {
        mode,
        executed_path,
        m: M,
        k: K,
        n: N,
        output,
    })
}

fn run_matmul_command(mode_name: &str) {
    let Some(mode) = MatmulMode::parse(mode_name) else {
        eprintln!("unknown matmul mode: {mode_name}");
        eprintln!("valid modes: reference");
        std::process::exit(2);
    };

    match run_matmul_smoke(mode) {
        Ok(report) => {
            println!("matmul smoke passed");
            println!("mode: {}", report.mode.as_str());
            println!("executed path: {}", report.executed_path);
            println!(
                "shape: {}x{} * {}x{}",
                report.m, report.k, report.k, report.n
            );
            println!("output: {:?}", report.output);
        }
        Err(error) => {
            eprintln!("matmul smoke failed: {error}");
            std::process::exit(1);
        }
    }
}
