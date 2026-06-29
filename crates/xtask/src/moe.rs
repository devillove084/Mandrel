use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use mandrel_runtime::{
    TinyMixtralLayerConfig, TinyMixtralLayerFixture, TinyMoeConfig, TinyMoeFixture,
    load_tiny_mixtral_layer_fixture_from_hf_dir, load_tiny_mixtral_moe_fixture_from_hf_dir,
    run_tiny_mixtral_layer_reference, run_tiny_moe_reference,
};

use crate::command::run_checked_with_retries;
use crate::{Result, XtaskError};

const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";
const TINY_MIXTRAL_MODEL_ID: &str = "yujiepan/mixtral-tiny-random";
const TINY_MIXTRAL_OUTPUT_DIR: &str = "target/mandrel/hf/yujiepan_mixtral_tiny_random";
const TINY_MIXTRAL_FILES: [&str; 2] = ["config.json", "model.safetensors"];

pub(crate) fn fetch_tiny_mixtral(workspace_root: &Path) -> Result<()> {
    let endpoint = hf_endpoint()?;
    let output_dir = project_path_from_env(
        workspace_root,
        "MANDREL_TINY_MIXTRAL_DIR",
        TINY_MIXTRAL_OUTPUT_DIR,
    );
    fs::create_dir_all(&output_dir).map_err(|error| {
        XtaskError::message(format!(
            "failed to create tiny Mixtral output directory '{}': {error}",
            output_dir.display()
        ))
    })?;

    let retries = fetch_retries_from_env()?;
    println!("Fetching tiny Mixtral fixture");
    println!("  model:    {TINY_MIXTRAL_MODEL_ID}");
    println!("  endpoint: {endpoint}");
    println!("  output:   {}", output_dir.display());
    println!("  retries:  {retries}");

    for file_name in TINY_MIXTRAL_FILES {
        let url = hf_resolve_url(&endpoint, TINY_MIXTRAL_MODEL_ID, file_name);
        let output_path = output_dir.join(file_name);
        download_file_with_curl(
            &url,
            &output_path,
            retries,
            &format!("hf.tiny-mixtral.{file_name}"),
        )?;
        println!("  downloaded: {}", output_path.display());
    }

    println!(
        "next: cargo xtask moe-run-reference {}",
        output_dir.display()
    );
    println!(
        "next layer: cargo xtask moe-run-layer-reference {}",
        output_dir.display()
    );
    Ok(())
}

fn hf_endpoint() -> Result<String> {
    let endpoint =
        non_empty_env("MANDREL_HF_ENDPOINT").unwrap_or_else(|| DEFAULT_HF_ENDPOINT.to_owned());
    let endpoint = endpoint.trim_end_matches('/').to_owned();
    if endpoint.is_empty() {
        return Err(XtaskError::message("MANDREL_HF_ENDPOINT must not be empty"));
    }
    if !(endpoint.starts_with("https://") || endpoint.starts_with("http://")) {
        return Err(XtaskError::message(format!(
            "MANDREL_HF_ENDPOINT must start with http:// or https://, got '{endpoint}'"
        )));
    }
    Ok(endpoint)
}

fn hf_resolve_url(endpoint: &str, model_id: &str, file_name: &str) -> String {
    format!("{endpoint}/{model_id}/resolve/main/{file_name}")
}

fn fetch_retries_from_env() -> Result<u32> {
    let Some(raw) = non_empty_env("MANDREL_FETCH_RETRIES") else {
        return Ok(3);
    };
    let retries = raw.parse::<u32>().map_err(|error| {
        XtaskError::message(format!("invalid MANDREL_FETCH_RETRIES '{raw}': {error}"))
    })?;
    Ok(retries.max(1))
}

fn download_file_with_curl(url: &str, output_path: &Path, retries: u32, phase: &str) -> Result<()> {
    let parent = non_empty_parent(output_path);
    fs::create_dir_all(parent).map_err(|error| {
        XtaskError::message(format!(
            "failed to create download output directory '{}': {error}",
            parent.display()
        ))
    })?;
    let temp_path = output_path.with_extension("download.tmp");
    if temp_path.exists() {
        fs::remove_file(&temp_path).map_err(|error| {
            XtaskError::message(format!(
                "failed to remove stale temporary download '{}': {error}",
                temp_path.display()
            ))
        })?;
    }

    let temp_for_command = temp_path.clone();
    run_checked_with_retries(
        || {
            let mut command = Command::new("curl");
            command
                .arg("--location")
                .arg("--fail")
                .arg("--show-error")
                .arg("--connect-timeout")
                .arg("20")
                .arg("--retry")
                .arg("2")
                .arg("--retry-delay")
                .arg("1")
                .arg("--output")
                .arg(&temp_for_command)
                .arg(url);
            Ok(command)
        },
        phase,
        retries,
    )?;
    fs::rename(&temp_path, output_path).map_err(|error| {
        XtaskError::message(format!(
            "failed to move temporary download '{}' to '{}': {error}",
            temp_path.display(),
            output_path.display()
        ))
    })
}

pub(crate) fn run_moe_reference_baseline(model_dir: Option<PathBuf>) -> Result<()> {
    let tokens = moe_usize_from_env("MANDREL_MOE_TOKENS", 4)?;
    let layer = moe_usize_from_env("MANDREL_MOE_LAYER", 0)?;
    let (source, fixture) = if let Some(model_dir) = model_dir {
        let fixture = load_tiny_mixtral_moe_fixture_from_hf_dir(&model_dir, tokens, layer)
            .map_err(|error| XtaskError::message(error.to_string()))?;
        (
            format!(
                "local HF Mixtral MoE block layer={} from {}",
                layer,
                model_dir.display()
            ),
            fixture,
        )
    } else {
        (
            "deterministic fixture shaped like yujiepan/mixtral-tiny-random MoE block".to_owned(),
            TinyMoeFixture::deterministic(TinyMoeConfig::new(tokens, 4, 8, 8, 2)),
        )
    };
    let output = run_tiny_moe_reference(&fixture);
    let config = fixture.config;

    println!("Tiny MoE reference baseline");
    println!("source: {source}");
    println!(
        "shape: tokens={} hidden={} intermediate={} experts={} top_k={}",
        config.tokens, config.hidden, config.intermediate, config.num_experts, config.top_k
    );
    println!("buffers:");
    println!("  input_elements: {}", fixture.input.len());
    println!("  router_weight_elements: {}", fixture.router_weight.len());
    println!("  gate_weight_elements: {}", fixture.gate_weights.len());
    println!("  up_weight_elements: {}", fixture.up_weights.len());
    println!("  down_weight_elements: {}", fixture.down_weights.len());
    println!("routing:");
    print_moe_routing(
        config.tokens,
        config.top_k,
        &output.selected_experts,
        &output.routing_weights,
    );
    println!("output:");
    println!("  elements: {}", output.output.len());
    println!("  checksum: {:.8}", output.output_checksum());
    println!("  max_abs: {:.8}", output.output_max_abs());
    println!("  first_row: {:?}", &output.output[..config.hidden]);
    Ok(())
}

pub(crate) fn run_moe_layer_reference_baseline(model_dir: Option<PathBuf>) -> Result<()> {
    let tokens = moe_usize_from_env("MANDREL_MOE_TOKENS", 4)?;
    let layer = moe_usize_from_env("MANDREL_MOE_LAYER", 0)?;
    let (source, fixture) = if let Some(model_dir) = model_dir {
        let fixture = load_tiny_mixtral_layer_fixture_from_hf_dir(&model_dir, tokens, layer)
            .map_err(|error| XtaskError::message(error.to_string()))?;
        (
            format!(
                "local HF Mixtral decoder layer={} from {}",
                layer,
                model_dir.display()
            ),
            fixture,
        )
    } else {
        (
            "deterministic fixture shaped like yujiepan/mixtral-tiny-random decoder layer"
                .to_owned(),
            TinyMixtralLayerFixture::deterministic(
                TinyMixtralLayerConfig::tiny_random_mixtral_layer(tokens),
            ),
        )
    };
    let output = run_tiny_mixtral_layer_reference(&fixture);
    let config = fixture.config;
    let moe = config.moe;

    println!("Tiny Mixtral layer reference baseline");
    println!("source: {source}");
    println!(
        "shape: tokens={} hidden={} intermediate={} experts={} top_k={}",
        moe.tokens, moe.hidden, moe.intermediate, moe.num_experts, moe.top_k
    );
    println!(
        "attention: query_heads={} key_value_heads={} head_dim={} rms_norm_eps={}",
        config.query_heads, config.key_value_heads, config.head_dim, config.rms_norm_eps
    );
    println!("buffers:");
    println!("  input_elements: {}", fixture.input.len());
    println!(
        "  input_layernorm_weight_elements: {}",
        fixture.input_layernorm_weight.len()
    );
    println!(
        "  post_attention_layernorm_weight_elements: {}",
        fixture.post_attention_layernorm_weight.len()
    );
    println!("  q_proj_weight_elements: {}", fixture.q_proj_weight.len());
    println!("  k_proj_weight_elements: {}", fixture.k_proj_weight.len());
    println!("  v_proj_weight_elements: {}", fixture.v_proj_weight.len());
    println!("  o_proj_weight_elements: {}", fixture.o_proj_weight.len());
    println!(
        "  router_weight_elements: {}",
        fixture.moe.router_weight.len()
    );
    println!("  gate_weight_elements: {}", fixture.moe.gate_weights.len());
    println!("  up_weight_elements: {}", fixture.moe.up_weights.len());
    println!("  down_weight_elements: {}", fixture.moe.down_weights.len());
    println!("intermediate summaries:");
    print_vector_summary("attention_output", &output.attention_output);
    print_vector_summary("hidden_after_attention", &output.hidden_after_attention);
    print_vector_summary("moe_output", &output.moe.output);
    println!("routing:");
    print_moe_routing(
        moe.tokens,
        moe.top_k,
        &output.moe.selected_experts,
        &output.moe.routing_weights,
    );
    println!("output:");
    println!("  elements: {}", output.output.len());
    println!("  checksum: {:.8}", output.output_checksum());
    println!("  max_abs: {:.8}", output.output_max_abs());
    println!("  first_row: {:?}", &output.output[..moe.hidden]);
    Ok(())
}

fn print_moe_routing(
    tokens: usize,
    top_k: usize,
    selected_experts: &[usize],
    routing_weights: &[f32],
) {
    for token in 0..tokens {
        let start = token * top_k;
        let end = start + top_k;
        println!(
            "  token {token}: experts={:?} weights={:?}",
            &selected_experts[start..end],
            &routing_weights[start..end]
        );
    }
}

fn print_vector_summary(name: &str, values: &[f32]) {
    println!(
        "  {name}: elements={} checksum={:.8} max_abs={:.8}",
        values.len(),
        vector_checksum(values),
        vector_max_abs(values)
    );
}

fn vector_checksum(values: &[f32]) -> f32 {
    values.iter().copied().sum()
}

fn vector_max_abs(values: &[f32]) -> f32 {
    values
        .iter()
        .fold(0.0_f32, |max_value, value| max_value.max(value.abs()))
}

fn moe_usize_from_env(name: &str, default: usize) -> Result<usize> {
    let Some(raw) = non_empty_env(name) else {
        return Ok(default);
    };
    raw.parse::<usize>()
        .map_err(|error| XtaskError::message(format!("invalid {name} '{raw}': {error}")))
}

fn project_path_from_env(workspace_root: &Path, key: &str, default_relative: &str) -> PathBuf {
    env::var_os(key)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root.join(default_relative))
}

fn non_empty_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn non_empty_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}
