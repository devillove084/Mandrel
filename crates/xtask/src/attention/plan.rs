use mandrel_compiler::{
    AttentionKvCacheMetadata, VortexAttentionPrefillPlan,
    compile_vortex_attention_prefill_kernel_for_target,
};
use mandrel_experiment::ExperimentSpec;
use mandrel_model_ir::AttentionOp;
use mandrel_target_ir::{TargetConstraints, TargetSpec};

use crate::Result;

use super::input::attention_runtime_extent_from_env;

const ATTENTION_COMPILED_SEQUENCE: usize = 64;
const ATTENTION_COMPILED_HEAD_DIM: usize = 64;
const ATTENTION_DEFAULT_RUNTIME_SEQUENCE: usize = 8;
const ATTENTION_DEFAULT_RUNTIME_HEAD_DIM: usize = 16;

pub(crate) fn print_attention_prefill_plan() -> Result<()> {
    let spec = current_attention_experiment_spec()?;
    let plan = current_attention_prefill_plan(spec)?;
    print_attention_plan("Vortex attention-prefill plan", &plan);
    Ok(())
}

pub(crate) fn current_attention_experiment_spec() -> Result<ExperimentSpec> {
    let sequence = attention_runtime_extent_from_env(
        "MANDREL_ATTENTION_RUNTIME_SEQUENCE",
        ATTENTION_DEFAULT_RUNTIME_SEQUENCE,
        ATTENTION_COMPILED_SEQUENCE,
    )?;
    let head_dim = attention_runtime_extent_from_env(
        "MANDREL_ATTENTION_RUNTIME_HEAD_DIM",
        ATTENTION_DEFAULT_RUNTIME_HEAD_DIM,
        ATTENTION_COMPILED_HEAD_DIM,
    )?;
    Ok(ExperimentSpec::attention_prefill_i8_vortex_smoke(
        sequence, head_dim,
    ))
}

pub(crate) fn current_attention_prefill_plan(
    spec: ExperimentSpec,
) -> Result<VortexAttentionPrefillPlan> {
    compile_attention_prefill_plan_for_target(spec.target)
}

fn compile_attention_prefill_plan_for_target(
    target: TargetSpec,
) -> Result<VortexAttentionPrefillPlan> {
    let constraints = TargetConstraints::from_device_capabilities(target.device_capabilities());
    Ok(compile_vortex_attention_prefill_kernel_for_target(
        AttentionOp::prefill_i8_demo(),
        constraints,
    )?)
}

pub(crate) fn print_attention_plan(title: &str, plan: &VortexAttentionPrefillPlan) {
    println!("{title}");
    println!(
        "shape: sequence={} head_dim={}",
        plan.op.shape.sequence, plan.op.shape.head_dim
    );
    println!(
        "kernel: {} ({:?}, {:?}, {:?})",
        plan.kernel.symbol.as_str(),
        plan.kernel.domain,
        plan.kernel.implementation,
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
    print_attention_metadata(plan);
    println!("metrics:");
    println!("  logical_macs: {}", plan.metrics.logical_macs);
    println!("  lowered_macs: {}", plan.metrics.lowered_macs);
    println!("  kernel_launches: {}", plan.metrics.kernel_launches);
    println!("  workgroup_count: {}", plan.metrics.workgroup_count);
    println!("  thread_count: {}", plan.metrics.thread_count);
    println!("  global_bytes_read: {}", plan.metrics.global_bytes_read);
    println!(
        "  global_bytes_written: {}",
        plan.metrics.global_bytes_written
    );
    println!(
        "  local_memory_bytes_per_workgroup: {}",
        plan.metrics.local_memory_bytes_per_workgroup
    );
    if let Some(intensity) = plan.metrics.logical_operational_intensity() {
        println!(
            "  logical_operational_intensity_macs_per_byte: {}/{}",
            intensity.numerator, intensity.denominator
        );
    }
    if let Some(intensity) = plan.metrics.lowered_operational_intensity() {
        println!(
            "  lowered_operational_intensity_macs_per_byte: {}/{}",
            intensity.numerator, intensity.denominator
        );
    }
}

fn print_attention_metadata(plan: &VortexAttentionPrefillPlan) {
    let metadata = plan.metadata;
    println!("metadata:");
    println!("  abi_arg_count: {}", metadata.arg_count);
    println!(
        "  scalar_args: sequence={} head_dim={} query_tile={} key_tile={}",
        metadata.scalars.sequence_arg_index,
        metadata.scalars.head_dim_arg_index,
        metadata.scalars.query_tile_arg_index,
        metadata.scalars.key_tile_arg_index
    );
    println!(
        "  runtime_shape: {:?} compiled=({}, {}) default=({}, {}) tile=({}, {}, {})",
        metadata.runtime_shape.kind,
        metadata.runtime_shape.compiled_sequence,
        metadata.runtime_shape.compiled_head_dim,
        metadata.runtime_shape.default_runtime_sequence,
        metadata.runtime_shape.default_runtime_head_dim,
        metadata.runtime_shape.query_tile,
        metadata.runtime_shape.key_tile,
        metadata.runtime_shape.head_dim_tile
    );
    match metadata.kv_cache {
        AttentionKvCacheMetadata::DenseContiguous { key, value } => println!(
            "  kv_cache: dense key_stride=({}, {}) value_stride=({}, {})",
            key.stride.row, key.stride.col, value.stride.row, value.stride.col
        ),
        AttentionKvCacheMetadata::Paged {
            page_size,
            logical_key,
            logical_value,
        } => println!(
            "  kv_cache: paged page_size={} logical_key_stride=({}, {}) logical_value_stride=({}, {})",
            page_size,
            logical_key.stride.row,
            logical_key.stride.col,
            logical_value.stride.row,
            logical_value.stride.col
        ),
    }
    println!("  buffers:");
    for buffer in metadata.buffers {
        println!(
            "    {:?}: arg={} slot={} layout={}x{} stride=({}, {}) element={:?} quant={:?}",
            buffer.role,
            buffer.arg_index,
            buffer.buffer_slot,
            buffer.layout.shape.rows,
            buffer.layout.shape.cols,
            buffer.layout.stride.row,
            buffer.layout.stride.col,
            buffer.element_type,
            buffer.quantization
        );
    }
}
