use std::env;

use mandrel_compiler::VortexAttentionPrefillPlan;
use mandrel_vortex_backend::AttentionPrefillI8Run;

use crate::Result;

pub(crate) fn deterministic_attention_prefill_input(
    plan: &VortexAttentionPrefillPlan,
) -> Result<AttentionPrefillI8Run> {
    let runtime_shape = plan.metadata.runtime_shape;
    let sequence_usize = attention_runtime_extent_from_env(
        "MANDREL_ATTENTION_RUNTIME_SEQUENCE",
        runtime_shape.default_runtime_sequence,
        runtime_shape.compiled_sequence,
    )?;
    let head_dim_usize = attention_runtime_extent_from_env(
        "MANDREL_ATTENTION_RUNTIME_HEAD_DIM",
        runtime_shape.default_runtime_head_dim,
        runtime_shape.compiled_head_dim,
    )?;
    let sequence = u32::try_from(sequence_usize)
        .map_err(|_| format!("attention runtime sequence does not fit u32: {sequence_usize}"))?;
    let head_dim = u32::try_from(head_dim_usize)
        .map_err(|_| format!("attention runtime head_dim does not fit u32: {head_dim_usize}"))?;
    let query_tile = u32::try_from(runtime_shape.query_tile).map_err(|_| {
        format!(
            "attention query tile does not fit u32: {}",
            runtime_shape.query_tile
        )
    })?;
    let key_tile = u32::try_from(runtime_shape.key_tile).map_err(|_| {
        format!(
            "attention key tile does not fit u32: {}",
            runtime_shape.key_tile
        )
    })?;
    let elements = sequence_usize
        .checked_mul(head_dim_usize)
        .ok_or_else(|| "attention runtime element count overflow".to_owned())?;

    let q = (0..elements).map(|index| ((index % 5) as i8) - 2).collect();
    let k = (0..elements)
        .map(|index| (((index * 3 + 1) % 5) as i8) - 2)
        .collect();
    let v = (0..elements)
        .map(|index| (((index * 7 + 3) % 17) as i8) - 8)
        .collect();

    Ok(AttentionPrefillI8Run {
        q,
        k,
        v,
        sequence,
        head_dim,
        query_tile,
        key_tile,
    })
}

pub(crate) fn attention_runtime_flag(key: &str) -> Result<bool> {
    let Some(raw) = non_empty_env(key) else {
        return Ok(false);
    };
    match raw.as_str() {
        "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON" => Ok(true),
        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF" => Ok(false),
        other => Err(format!(
            "invalid {key}='{other}': expected one of 1/0, true/false, yes/no, on/off"
        )
        .into()),
    }
}

pub(crate) fn attention_runtime_extent_from_env(
    key: &str,
    default_value: usize,
    max_value: usize,
) -> Result<usize> {
    let Some(raw) = env::var_os(key) else {
        return Ok(default_value);
    };
    let text = raw.to_string_lossy();
    let value = text
        .parse::<usize>()
        .map_err(|error| format!("invalid {key}='{text}': {error}"))?;
    if value == 0 || value > max_value {
        return Err(format!(
            "{key} must be in 1..={max_value} for the current generated launch, got {value}"
        )
        .into());
    }
    Ok(value)
}

fn non_empty_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}
