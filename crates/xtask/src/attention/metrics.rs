use mandrel_compiler::VortexAttentionPrefillPlan;
use mandrel_model_ir::AttentionShape;
use mandrel_profiler::{AttentionPrefillEstimateInput, estimate_gpu_attention_prefill};
use mandrel_target_ir::{TargetContract, TargetSpec};
use mandrel_vortex_backend::{AttentionPrefillI8Run, VortexDeviceCaps, VortexLaunchTrace};

use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttentionRuntimeWorkload {
    pub(crate) compiled_sequence: u32,
    pub(crate) compiled_head_dim: u32,
    pub(crate) head_dim_tile: u32,
    pub(crate) runtime_sequence: u32,
    pub(crate) runtime_head_dim: u32,
    pub(crate) query_tile: u32,
    pub(crate) key_tile: u32,
    pub(crate) logical_macs: u64,
    pub(crate) lowered_macs: u64,
    pub(crate) estimated_global_bytes_read: u64,
    pub(crate) estimated_global_bytes_written: u64,
    pub(crate) estimated_local_memory_bytes_per_workgroup: u64,
    pub(crate) target_contract: TargetContract,
    pub(crate) target_threads_per_warp: u64,
    pub(crate) target_warps_per_core: u64,
    pub(crate) target_max_workgroup_threads: u64,
    pub(crate) target_local_memory_bytes: u64,
    pub(crate) workgroup_count: u64,
    pub(crate) threads_per_workgroup: u64,
    pub(crate) total_threads: u64,
    pub(crate) runtime_q_elements: u64,
    pub(crate) runtime_kv_elements: u64,
    pub(crate) runtime_output_elements: u64,
    pub(crate) runtime_q_bytes: u64,
    pub(crate) runtime_kv_bytes: u64,
    pub(crate) runtime_output_bytes: u64,
    pub(crate) module_cache_hit: bool,
    pub(crate) kernel_cache_hit: bool,
}

impl AttentionRuntimeWorkload {
    pub(crate) fn from_runtime(
        plan: &VortexAttentionPrefillPlan,
        input: &AttentionPrefillI8Run,
        trace: VortexLaunchTrace,
        target_caps: VortexDeviceCaps,
        target_contract: TargetContract,
        output_elements: usize,
    ) -> Result<Self> {
        let runtime_shape = plan.metadata.runtime_shape;
        let compiled_sequence = usize_to_u32(
            runtime_shape.compiled_sequence,
            "attention compiled sequence",
        )?;
        let compiled_head_dim = usize_to_u32(
            runtime_shape.compiled_head_dim,
            "attention compiled head_dim",
        )?;
        let head_dim_tile = usize_to_u32(runtime_shape.head_dim_tile, "attention head_dim tile")?;
        let runtime_q_elements = usize_to_u64(input.q.len(), "attention q element count")?;
        let runtime_k_elements = usize_to_u64(input.k.len(), "attention k element count")?;
        let runtime_v_elements = usize_to_u64(input.v.len(), "attention v element count")?;
        let runtime_kv_elements = checked_add_u64(
            runtime_k_elements,
            runtime_v_elements,
            "attention runtime KV element count",
        )?;
        let runtime_output_elements =
            usize_to_u64(output_elements, "attention output element count")?;
        let workgroup_count = checked_product_u32(trace.dims.grid, "attention workgroup count")?;
        let threads_per_workgroup =
            checked_product_u32(trace.dims.block, "attention threads per workgroup")?;
        let total_threads = checked_mul_u64(
            workgroup_count,
            threads_per_workgroup,
            "attention total thread count",
        )?;
        let target_max_workgroup_threads = target_caps
            .max_workgroup_threads()
            .ok_or_else(|| "Vortex target max workgroup thread count overflow".to_owned())?;
        let lowering_metrics = estimate_gpu_attention_prefill(AttentionPrefillEstimateInput::new(
            AttentionShape::new(input.sequence as usize, input.head_dim as usize),
            plan.schedule,
            plan.op.element_type,
            plan.op.element_type,
        ))
        .ok_or_else(|| "attention runtime lowering estimate overflow".to_owned())?;

        Ok(Self {
            compiled_sequence,
            compiled_head_dim,
            head_dim_tile,
            runtime_sequence: input.sequence,
            runtime_head_dim: input.head_dim,
            query_tile: input.query_tile,
            key_tile: input.key_tile,
            logical_macs: attention_prefill_logical_macs(input.sequence, input.head_dim)?,
            lowered_macs: usize_to_u64(lowering_metrics.lowered_macs, "lowered MAC count")?,
            estimated_global_bytes_read: usize_to_u64(
                lowering_metrics.global_bytes_read,
                "estimated global bytes read",
            )?,
            estimated_global_bytes_written: usize_to_u64(
                lowering_metrics.global_bytes_written,
                "estimated global bytes written",
            )?,
            estimated_local_memory_bytes_per_workgroup: usize_to_u64(
                lowering_metrics.local_memory_bytes_per_workgroup,
                "estimated local memory bytes per workgroup",
            )?,
            target_contract,
            target_threads_per_warp: target_caps.threads_per_warp,
            target_warps_per_core: target_caps.warps_per_core,
            target_max_workgroup_threads,
            target_local_memory_bytes: target_caps.local_memory_bytes,
            workgroup_count,
            threads_per_workgroup,
            total_threads,
            runtime_q_elements,
            runtime_kv_elements,
            runtime_output_elements,
            runtime_q_bytes: runtime_q_elements,
            runtime_kv_bytes: runtime_kv_elements,
            runtime_output_bytes: runtime_output_elements,
            module_cache_hit: trace.module_cache_hit,
            kernel_cache_hit: trace.kernel_cache_hit,
        })
    }
}

pub(crate) fn attention_i8_checksum(values: &[i8]) -> i64 {
    values.iter().map(|value| i64::from(*value)).sum()
}

pub(crate) fn attention_i8_max_abs(values: &[i8]) -> i16 {
    values
        .iter()
        .map(|value| i16::from(*value).abs())
        .max()
        .unwrap_or(0)
}

pub(crate) fn format_mandrel_runtime_trace_line(
    trace: VortexLaunchTrace,
    workload: AttentionRuntimeWorkload,
) -> String {
    format!(
        "MANDREL_RUNTIME_TRACE: kernel={} runtime_sequence={} runtime_head_dim={} query_tile={} key_tile={} compiled_sequence={} compiled_head_dim={} head_dim_tile={} logical_macs={} lowered_macs={} estimated_global_bytes_read={} estimated_global_bytes_written={} estimated_local_memory_bytes_per_workgroup={} requested_target_backend={} requested_target_xlen={} requested_target_max_workgroup_threads={} requested_target_preferred_subgroup_width={} requested_target_local_memory_bytes={} requested_target_supports_int8={} requested_target_supports_float32={} requested_target_supports_tensor_cores={} requested_target_supports_async_copy={} observed_target_backend={} observed_target_xlen={} observed_target_preferred_subgroup_width={} observed_target_supports_int8={} observed_target_supports_float32={} observed_target_supports_tensor_cores={} observed_target_supports_async_copy={} target_compatible={} target_mismatch_mask={} target_threads_per_warp={} target_warps_per_core={} target_max_workgroup_threads={} target_local_memory_bytes={} workgroup_count={} threads_per_workgroup={} total_threads={} runtime_q_elements={} runtime_kv_elements={} runtime_output_elements={} runtime_q_bytes={} runtime_kv_bytes={} runtime_output_bytes={} module_cache_hit={} kernel_cache_hit={} grid={}x{}x{} block={}x{}x{} shared_memory_bytes={} host_to_device_bytes={} device_to_host_bytes={}",
        trace.kernel_symbol,
        workload.runtime_sequence,
        workload.runtime_head_dim,
        workload.query_tile,
        workload.key_tile,
        workload.compiled_sequence,
        workload.compiled_head_dim,
        workload.head_dim_tile,
        workload.logical_macs,
        workload.lowered_macs,
        workload.estimated_global_bytes_read,
        workload.estimated_global_bytes_written,
        workload.estimated_local_memory_bytes_per_workgroup,
        target_backend_as_str(workload.target_contract.requested),
        workload.target_contract.requested.xlen,
        workload.target_contract.requested.max_workgroup_threads,
        workload.target_contract.requested.preferred_subgroup_width,
        workload.target_contract.requested.local_memory_bytes,
        workload.target_contract.requested.supports_int8,
        workload.target_contract.requested.supports_float32,
        workload.target_contract.requested.supports_tensor_cores,
        workload.target_contract.requested.supports_async_copy,
        target_backend_as_str(workload.target_contract.observed),
        workload.target_contract.observed.xlen,
        workload.target_contract.observed.preferred_subgroup_width,
        workload.target_contract.observed.supports_int8,
        workload.target_contract.observed.supports_float32,
        workload.target_contract.observed.supports_tensor_cores,
        workload.target_contract.observed.supports_async_copy,
        workload.target_contract.is_compatible(),
        workload.target_contract.compatibility.mismatch_mask(),
        workload.target_threads_per_warp,
        workload.target_warps_per_core,
        workload.target_max_workgroup_threads,
        workload.target_local_memory_bytes,
        workload.workgroup_count,
        workload.threads_per_workgroup,
        workload.total_threads,
        workload.runtime_q_elements,
        workload.runtime_kv_elements,
        workload.runtime_output_elements,
        workload.runtime_q_bytes,
        workload.runtime_kv_bytes,
        workload.runtime_output_bytes,
        workload.module_cache_hit,
        workload.kernel_cache_hit,
        trace.dims.grid[0],
        trace.dims.grid[1],
        trace.dims.grid[2],
        trace.dims.block[0],
        trace.dims.block[1],
        trace.dims.block[2],
        trace.dims.shared_memory_bytes,
        trace.host_to_device_bytes,
        trace.device_to_host_bytes
    )
}

fn target_backend_as_str(target: TargetSpec) -> &'static str {
    target.backend_name()
}

fn attention_prefill_logical_macs(sequence: u32, head_dim: u32) -> Result<u64> {
    let sequence = u64::from(sequence);
    let head_dim = u64::from(head_dim);
    let score_macs = checked_mul_u64(
        checked_mul_u64(sequence, sequence, "attention score matrix MAC count")?,
        head_dim,
        "attention score dot-product MAC count",
    )?;
    checked_mul_u64(score_macs, 2, "attention prefill logical MAC count")
}

fn checked_product_u32(values: [u32; 3], label: &str) -> Result<u64> {
    let xy = checked_mul_u64(u64::from(values[0]), u64::from(values[1]), label)?;
    checked_mul_u64(xy, u64::from(values[2]), label)
}

fn checked_add_u64(lhs: u64, rhs: u64, label: &str) -> Result<u64> {
    match lhs.checked_add(rhs) {
        Some(value) => Ok(value),
        None => Err(format!("{label} overflow").into()),
    }
}

fn checked_mul_u64(lhs: u64, rhs: u64, label: &str) -> Result<u64> {
    match lhs.checked_mul(rhs) {
        Some(value) => Ok(value),
        None => Err(format!("{label} overflow").into()),
    }
}

fn usize_to_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| format!("{label} does not fit u32: {value}").into())
}

fn usize_to_u64(value: usize, label: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| format!("{label} does not fit u64: {value}").into())
}
