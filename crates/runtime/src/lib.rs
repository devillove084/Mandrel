#![cfg_attr(not(feature = "std"), no_std)]

use mandrel_core::{ElementType, Layout, Shape, TensorDesc};
use mandrel_device::{CommandBuffer, DeviceCommand};
use mandrel_kernel_ir::{Dim3, KernelArg, KernelLaunch, KernelSymbol};
use mandrel_target_ir::DeviceBackend;

#[cfg(feature = "std")]
use std::{fmt, fs, path::PathBuf};

pub type TensorId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeTarget {
    HostReference,
    VortexRtl,
    VortexFpga,
}

impl RuntimeTarget {
    pub const fn device_backend(self) -> DeviceBackend {
        match self {
            Self::HostReference => DeviceBackend::HostReference,
            Self::VortexRtl => DeviceBackend::VortexRtl,
            Self::VortexFpga => DeviceBackend::VortexFpga,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimePlan<const COMMANDS: usize, const ARGS: usize> {
    pub target: RuntimeTarget,
    pub command_buffer: CommandBuffer<COMMANDS, ARGS>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelIo {
    pub input: TensorDesc<2>,
    pub output: TensorDesc<2>,
}

pub const fn example_io() -> ModelIo {
    ModelIo {
        input: TensorDesc {
            shape: Shape::new([64, 64]),
            element_type: ElementType::I8,
            layout: Layout::RowMajor,
        },
        output: TensorDesc {
            shape: Shape::new([64, 64]),
            element_type: ElementType::I8,
            layout: Layout::RowMajor,
        },
    }
}

pub const fn example_attention_vortex_plan() -> RuntimePlan<2, 8> {
    let launch = KernelLaunch::new(
        KernelSymbol::AttentionPrefillI8,
        Dim3::new(16, 1, 1),
        Dim3::new(4, 4, 1),
        0,
        [
            KernelArg::buffer(0, 0),
            KernelArg::buffer(1, 1),
            KernelArg::buffer(2, 2),
            KernelArg::buffer(3, 3),
            KernelArg::u32(4, 64),
            KernelArg::u32(5, 64),
            KernelArg::u32(6, 4),
            KernelArg::u32(7, 1),
        ],
    );

    RuntimePlan {
        target: RuntimeTarget::VortexRtl,
        command_buffer: CommandBuffer::new([DeviceCommand::Launch(launch), DeviceCommand::Fence]),
    }
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TinyMoeConfig {
    pub tokens: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub num_experts: usize,
    pub top_k: usize,
}

#[cfg(feature = "std")]
impl TinyMoeConfig {
    pub const fn new(
        tokens: usize,
        hidden: usize,
        intermediate: usize,
        num_experts: usize,
        top_k: usize,
    ) -> Self {
        Self {
            tokens,
            hidden,
            intermediate,
            num_experts,
            top_k,
        }
    }

    /// Shape-compatible with the tiny random Mixtral fixture on Hugging Face
    /// (`yujiepan/mixtral-tiny-random`) at the MoE-block level.
    pub const fn tiny_random_mixtral_block() -> Self {
        Self::new(4, 4, 8, 8, 2)
    }

    pub const fn input_len(self) -> usize {
        self.tokens * self.hidden
    }

    pub const fn router_weight_len(self) -> usize {
        self.num_experts * self.hidden
    }

    pub const fn expert_gate_or_up_weight_len(self) -> usize {
        self.num_experts * self.intermediate * self.hidden
    }

    pub const fn expert_down_weight_len(self) -> usize {
        self.num_experts * self.hidden * self.intermediate
    }

    pub const fn routing_len(self) -> usize {
        self.tokens * self.top_k
    }

    pub const fn selected_expert_output_len(self) -> usize {
        self.tokens * self.top_k * self.hidden
    }
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq)]
pub struct TinyMoeFixture {
    pub config: TinyMoeConfig,
    pub input: Vec<f32>,
    pub router_weight: Vec<f32>,
    pub gate_weights: Vec<f32>,
    pub up_weights: Vec<f32>,
    pub down_weights: Vec<f32>,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TinyMixtralLayerConfig {
    pub moe: TinyMoeConfig,
    pub query_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TinyMixtralModelConfig {
    pub layer: TinyMixtralLayerConfig,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
}

#[cfg(feature = "std")]
impl TinyMixtralLayerConfig {
    pub const fn tiny_random_mixtral_layer(tokens: usize) -> Self {
        Self {
            moe: TinyMoeConfig::new(tokens, 4, 8, 8, 2),
            query_heads: 4,
            key_value_heads: 2,
            head_dim: 1,
            rms_norm_eps: 1.0e-5,
            rope_theta: 1_000_000.0,
        }
    }

    pub const fn query_width(self) -> usize {
        self.query_heads * self.head_dim
    }

    pub const fn key_value_width(self) -> usize {
        self.key_value_heads * self.head_dim
    }
}

#[cfg(feature = "std")]
impl TinyMixtralModelConfig {
    pub const fn new(
        layer: TinyMixtralLayerConfig,
        vocab_size: usize,
        num_hidden_layers: usize,
    ) -> Self {
        Self {
            layer,
            vocab_size,
            num_hidden_layers,
        }
    }

    pub const fn tiny_random_mixtral_model(tokens: usize) -> Self {
        Self::new(
            TinyMixtralLayerConfig::tiny_random_mixtral_layer(tokens),
            32,
            2,
        )
    }

    pub const fn tokens(self) -> usize {
        self.layer.moe.tokens
    }

    pub const fn hidden(self) -> usize {
        self.layer.moe.hidden
    }

    pub const fn embedding_weight_len(self) -> usize {
        self.vocab_size * self.hidden()
    }

    pub const fn final_norm_weight_len(self) -> usize {
        self.hidden()
    }

    pub const fn logits_len(self) -> usize {
        self.tokens() * self.vocab_size
    }
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq)]
pub struct TinyMixtralLayerFixture {
    pub config: TinyMixtralLayerConfig,
    pub input: Vec<f32>,
    pub input_layernorm_weight: Vec<f32>,
    pub post_attention_layernorm_weight: Vec<f32>,
    pub q_proj_weight: Vec<f32>,
    pub k_proj_weight: Vec<f32>,
    pub v_proj_weight: Vec<f32>,
    pub o_proj_weight: Vec<f32>,
    pub moe: TinyMoeFixture,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq)]
pub struct TinyMixtralModelFixture {
    pub config: TinyMixtralModelConfig,
    pub token_ids: Vec<usize>,
    pub embed_tokens_weight: Vec<f32>,
    pub layers: Vec<TinyMixtralLayerFixture>,
    pub final_norm_weight: Vec<f32>,
    pub lm_head_weight: Vec<f32>,
}

#[cfg(feature = "std")]
impl TinyMoeFixture {
    pub fn deterministic(config: TinyMoeConfig) -> Self {
        Self {
            config,
            input: deterministic_vector(config.input_len(), 0, 0.125),
            router_weight: deterministic_vector(config.router_weight_len(), 1_000, 0.25),
            gate_weights: deterministic_vector(
                config.expert_gate_or_up_weight_len(),
                2_000,
                0.0625,
            ),
            up_weights: deterministic_vector(config.expert_gate_or_up_weight_len(), 3_000, 0.0625),
            down_weights: deterministic_vector(config.expert_down_weight_len(), 4_000, 0.0625),
        }
    }

    pub fn tiny_random_mixtral_block() -> Self {
        Self::deterministic(TinyMoeConfig::tiny_random_mixtral_block())
    }
}

#[cfg(feature = "std")]
impl TinyMixtralLayerFixture {
    pub fn deterministic(config: TinyMixtralLayerConfig) -> Self {
        Self {
            config,
            input: deterministic_vector(config.moe.input_len(), 0, 0.125),
            input_layernorm_weight: vec![1.0_f32; config.moe.hidden],
            post_attention_layernorm_weight: vec![1.0_f32; config.moe.hidden],
            q_proj_weight: deterministic_vector(
                config.query_width() * config.moe.hidden,
                5_000,
                0.0625,
            ),
            k_proj_weight: deterministic_vector(
                config.key_value_width() * config.moe.hidden,
                6_000,
                0.0625,
            ),
            v_proj_weight: deterministic_vector(
                config.key_value_width() * config.moe.hidden,
                7_000,
                0.0625,
            ),
            o_proj_weight: deterministic_vector(
                config.moe.hidden * config.query_width(),
                8_000,
                0.0625,
            ),
            moe: TinyMoeFixture::deterministic(config.moe),
        }
    }
}

#[cfg(feature = "std")]
impl TinyMixtralModelFixture {
    pub fn deterministic(config: TinyMixtralModelConfig) -> Self {
        let layers = (0..config.num_hidden_layers)
            .map(|_| TinyMixtralLayerFixture::deterministic(config.layer))
            .collect();
        Self {
            config,
            token_ids: deterministic_token_ids(config.tokens(), config.vocab_size),
            embed_tokens_weight: deterministic_vector(config.embedding_weight_len(), 9_000, 0.0625),
            layers,
            final_norm_weight: vec![1.0_f32; config.final_norm_weight_len()],
            lm_head_weight: deterministic_vector(config.embedding_weight_len(), 10_000, 0.0625),
        }
    }
}

#[cfg(feature = "std")]
#[derive(Debug)]
pub enum TinyMoeHfLoadError {
    ReadConfig {
        path: PathBuf,
        message: String,
    },
    ParseConfig {
        path: PathBuf,
        message: String,
    },
    ReadWeights {
        path: PathBuf,
        message: String,
    },
    ParseWeights {
        path: PathBuf,
        message: String,
    },
    UnsupportedModelType {
        model_type: String,
    },
    InvalidConfig {
        message: String,
    },
    MissingTensor {
        name: String,
    },
    UnexpectedTensorShape {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    UnsupportedTensorDtype {
        name: String,
        dtype: String,
    },
    InvalidTensorData {
        name: String,
        message: String,
    },
}

#[cfg(feature = "std")]
impl fmt::Display for TinyMoeHfLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadConfig { path, message } => {
                write!(
                    formatter,
                    "failed to read HF config '{}': {message}",
                    path.display()
                )
            }
            Self::ParseConfig { path, message } => {
                write!(
                    formatter,
                    "failed to parse HF config '{}': {message}",
                    path.display()
                )
            }
            Self::ReadWeights { path, message } => write!(
                formatter,
                "failed to read safetensors weights '{}': {message}",
                path.display()
            ),
            Self::ParseWeights { path, message } => write!(
                formatter,
                "failed to parse safetensors weights '{}': {message}",
                path.display()
            ),
            Self::UnsupportedModelType { model_type } => {
                write!(
                    formatter,
                    "unsupported HF model_type '{model_type}', expected 'mixtral'"
                )
            }
            Self::InvalidConfig { message } => {
                write!(formatter, "invalid tiny MoE config: {message}")
            }
            Self::MissingTensor { name } => {
                write!(formatter, "missing safetensors tensor '{name}'")
            }
            Self::UnexpectedTensorShape {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "unexpected tensor shape for '{name}': expected {expected:?}, got {actual:?}"
            ),
            Self::UnsupportedTensorDtype { name, dtype } => {
                write!(formatter, "unsupported tensor dtype for '{name}': {dtype}")
            }
            Self::InvalidTensorData { name, message } => {
                write!(formatter, "invalid tensor data for '{name}': {message}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TinyMoeHfLoadError {}

#[cfg(feature = "std")]
#[derive(Debug, serde::Deserialize)]
struct HfMixtralConfig {
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_local_experts: usize,
    num_experts_per_tok: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f32,
    #[serde(default = "default_mixtral_rope_theta")]
    rope_theta: f32,
}

#[cfg(feature = "std")]
const fn default_mixtral_rope_theta() -> f32 {
    1_000_000.0
}

#[cfg(feature = "std")]
pub fn load_tiny_mixtral_moe_fixture_from_hf_dir(
    model_dir: impl Into<PathBuf>,
    tokens: usize,
    layer: usize,
) -> Result<TinyMoeFixture, TinyMoeHfLoadError> {
    let model_dir = model_dir.into();
    let config_path = model_dir.join("config.json");
    let weights_path = model_dir.join("model.safetensors");
    let config_source =
        fs::read_to_string(&config_path).map_err(|error| TinyMoeHfLoadError::ReadConfig {
            path: config_path.clone(),
            message: error.to_string(),
        })?;
    let hf_config: HfMixtralConfig =
        serde_json::from_str(&config_source).map_err(|error| TinyMoeHfLoadError::ParseConfig {
            path: config_path.clone(),
            message: error.to_string(),
        })?;
    let config = tiny_moe_config_from_hf_mixtral(hf_config, tokens)?;
    let weight_bytes =
        fs::read(&weights_path).map_err(|error| TinyMoeHfLoadError::ReadWeights {
            path: weights_path.clone(),
            message: error.to_string(),
        })?;
    let tensors = safetensors::SafeTensors::deserialize(&weight_bytes).map_err(|error| {
        TinyMoeHfLoadError::ParseWeights {
            path: weights_path.clone(),
            message: error.to_string(),
        }
    })?;

    load_mixtral_moe_fixture_from_tensors(
        &tensors,
        config,
        deterministic_vector(config.input_len(), 0, 0.125),
        layer,
    )
}

#[cfg(feature = "std")]
pub fn load_tiny_mixtral_layer_fixture_from_hf_dir(
    model_dir: impl Into<PathBuf>,
    tokens: usize,
    layer: usize,
) -> Result<TinyMixtralLayerFixture, TinyMoeHfLoadError> {
    let model_dir = model_dir.into();
    let config_path = model_dir.join("config.json");
    let weights_path = model_dir.join("model.safetensors");
    let config_source =
        fs::read_to_string(&config_path).map_err(|error| TinyMoeHfLoadError::ReadConfig {
            path: config_path.clone(),
            message: error.to_string(),
        })?;
    let hf_config: HfMixtralConfig =
        serde_json::from_str(&config_source).map_err(|error| TinyMoeHfLoadError::ParseConfig {
            path: config_path.clone(),
            message: error.to_string(),
        })?;
    let config = tiny_mixtral_layer_config_from_hf_mixtral(hf_config, tokens)?;
    let weight_bytes =
        fs::read(&weights_path).map_err(|error| TinyMoeHfLoadError::ReadWeights {
            path: weights_path.clone(),
            message: error.to_string(),
        })?;
    let tensors = safetensors::SafeTensors::deserialize(&weight_bytes).map_err(|error| {
        TinyMoeHfLoadError::ParseWeights {
            path: weights_path.clone(),
            message: error.to_string(),
        }
    })?;
    let input = deterministic_vector(config.moe.input_len(), 0, 0.125);
    let moe = load_mixtral_moe_fixture_from_tensors(&tensors, config.moe, input.clone(), layer)?;

    Ok(TinyMixtralLayerFixture {
        config,
        input,
        input_layernorm_weight: load_required_tensor_f32(
            &tensors,
            &mixtral_input_layernorm_tensor_name(layer),
            &[config.moe.hidden],
        )?,
        post_attention_layernorm_weight: load_required_tensor_f32(
            &tensors,
            &mixtral_post_attention_layernorm_tensor_name(layer),
            &[config.moe.hidden],
        )?,
        q_proj_weight: load_required_tensor_f32(
            &tensors,
            &mixtral_attention_tensor_name(layer, "q_proj"),
            &[config.query_width(), config.moe.hidden],
        )?,
        k_proj_weight: load_required_tensor_f32(
            &tensors,
            &mixtral_attention_tensor_name(layer, "k_proj"),
            &[config.key_value_width(), config.moe.hidden],
        )?,
        v_proj_weight: load_required_tensor_f32(
            &tensors,
            &mixtral_attention_tensor_name(layer, "v_proj"),
            &[config.key_value_width(), config.moe.hidden],
        )?,
        o_proj_weight: load_required_tensor_f32(
            &tensors,
            &mixtral_attention_tensor_name(layer, "o_proj"),
            &[config.moe.hidden, config.query_width()],
        )?,
        moe,
    })
}

#[cfg(feature = "std")]
fn load_mixtral_moe_fixture_from_tensors(
    tensors: &safetensors::SafeTensors<'_>,
    config: TinyMoeConfig,
    input: Vec<f32>,
    layer: usize,
) -> Result<TinyMoeFixture, TinyMoeHfLoadError> {
    let router_name = mixtral_router_tensor_name(layer);
    let router_weight =
        load_required_tensor_f32(tensors, &router_name, &[config.num_experts, config.hidden])?;
    let mut gate_weights = Vec::with_capacity(config.expert_gate_or_up_weight_len());
    let mut up_weights = Vec::with_capacity(config.expert_gate_or_up_weight_len());
    let mut down_weights = Vec::with_capacity(config.expert_down_weight_len());

    for expert in 0..config.num_experts {
        let gate_name = mixtral_expert_tensor_name(layer, expert, "w1");
        let up_name = mixtral_expert_tensor_name(layer, expert, "w3");
        let down_name = mixtral_expert_tensor_name(layer, expert, "w2");
        gate_weights.extend(load_required_tensor_f32(
            tensors,
            &gate_name,
            &[config.intermediate, config.hidden],
        )?);
        up_weights.extend(load_required_tensor_f32(
            tensors,
            &up_name,
            &[config.intermediate, config.hidden],
        )?);
        down_weights.extend(load_required_tensor_f32(
            tensors,
            &down_name,
            &[config.hidden, config.intermediate],
        )?);
    }

    Ok(TinyMoeFixture {
        config,
        input,
        router_weight,
        gate_weights,
        up_weights,
        down_weights,
    })
}

#[cfg(feature = "std")]
fn tiny_moe_config_from_hf_mixtral(
    hf_config: HfMixtralConfig,
    tokens: usize,
) -> Result<TinyMoeConfig, TinyMoeHfLoadError> {
    validate_mixtral_model_type(&hf_config)?;
    let config = TinyMoeConfig::new(
        tokens,
        hf_config.hidden_size,
        hf_config.intermediate_size,
        hf_config.num_local_experts,
        hf_config.num_experts_per_tok,
    );
    validate_tiny_moe_config(config)?;
    Ok(config)
}

#[cfg(feature = "std")]
fn tiny_mixtral_layer_config_from_hf_mixtral(
    hf_config: HfMixtralConfig,
    tokens: usize,
) -> Result<TinyMixtralLayerConfig, TinyMoeHfLoadError> {
    validate_mixtral_model_type(&hf_config)?;
    let moe = TinyMoeConfig::new(
        tokens,
        hf_config.hidden_size,
        hf_config.intermediate_size,
        hf_config.num_local_experts,
        hf_config.num_experts_per_tok,
    );
    validate_tiny_moe_config(moe)?;
    if hf_config.num_attention_heads == 0 || hf_config.num_key_value_heads == 0 {
        return Err(TinyMoeHfLoadError::InvalidConfig {
            message: "attention head counts must be non-zero".to_owned(),
        });
    }
    if hf_config.hidden_size % hf_config.num_attention_heads != 0 {
        return Err(TinyMoeHfLoadError::InvalidConfig {
            message: format!(
                "hidden_size must be divisible by num_attention_heads, got hidden_size={} num_attention_heads={}",
                hf_config.hidden_size, hf_config.num_attention_heads
            ),
        });
    }
    if hf_config.num_attention_heads % hf_config.num_key_value_heads != 0 {
        return Err(TinyMoeHfLoadError::InvalidConfig {
            message: format!(
                "num_attention_heads must be divisible by num_key_value_heads, got {} and {}",
                hf_config.num_attention_heads, hf_config.num_key_value_heads
            ),
        });
    }

    Ok(TinyMixtralLayerConfig {
        moe,
        query_heads: hf_config.num_attention_heads,
        key_value_heads: hf_config.num_key_value_heads,
        head_dim: hf_config.hidden_size / hf_config.num_attention_heads,
        rms_norm_eps: hf_config.rms_norm_eps,
        rope_theta: hf_config.rope_theta,
    })
}

#[cfg(feature = "std")]
fn validate_mixtral_model_type(hf_config: &HfMixtralConfig) -> Result<(), TinyMoeHfLoadError> {
    if hf_config.model_type != "mixtral" {
        return Err(TinyMoeHfLoadError::UnsupportedModelType {
            model_type: hf_config.model_type.clone(),
        });
    }
    Ok(())
}

#[cfg(feature = "std")]
fn validate_tiny_moe_config(config: TinyMoeConfig) -> Result<(), TinyMoeHfLoadError> {
    if config.tokens == 0 {
        return Err(TinyMoeHfLoadError::InvalidConfig {
            message: "tokens must be non-zero".to_owned(),
        });
    }
    if config.hidden == 0 || config.intermediate == 0 || config.num_experts == 0 {
        return Err(TinyMoeHfLoadError::InvalidConfig {
            message: "hidden, intermediate, and num_experts must be non-zero".to_owned(),
        });
    }
    if config.top_k == 0 || config.top_k > config.num_experts {
        return Err(TinyMoeHfLoadError::InvalidConfig {
            message: format!(
                "top_k must be in 1..=num_experts, got top_k={} num_experts={}",
                config.top_k, config.num_experts
            ),
        });
    }
    Ok(())
}

#[cfg(feature = "std")]
fn mixtral_router_tensor_name(layer: usize) -> String {
    format!("model.layers.{layer}.block_sparse_moe.gate.weight")
}

#[cfg(feature = "std")]
fn mixtral_expert_tensor_name(layer: usize, expert: usize, weight: &str) -> String {
    format!("model.layers.{layer}.block_sparse_moe.experts.{expert}.{weight}.weight")
}

#[cfg(feature = "std")]
fn mixtral_input_layernorm_tensor_name(layer: usize) -> String {
    format!("model.layers.{layer}.input_layernorm.weight")
}

#[cfg(feature = "std")]
fn mixtral_post_attention_layernorm_tensor_name(layer: usize) -> String {
    format!("model.layers.{layer}.post_attention_layernorm.weight")
}

#[cfg(feature = "std")]
fn mixtral_attention_tensor_name(layer: usize, projection: &str) -> String {
    format!("model.layers.{layer}.self_attn.{projection}.weight")
}

#[cfg(feature = "std")]
fn load_required_tensor_f32(
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<f32>, TinyMoeHfLoadError> {
    let tensor = tensors
        .tensor(name)
        .map_err(|_| TinyMoeHfLoadError::MissingTensor {
            name: name.to_owned(),
        })?;
    tensor_to_f32_vec(name, tensor, expected_shape)
}

#[cfg(feature = "std")]
fn tensor_to_f32_vec(
    name: &str,
    tensor: safetensors::tensor::TensorView<'_>,
    expected_shape: &[usize],
) -> Result<Vec<f32>, TinyMoeHfLoadError> {
    if tensor.shape() != expected_shape {
        return Err(TinyMoeHfLoadError::UnexpectedTensorShape {
            name: name.to_owned(),
            expected: expected_shape.to_vec(),
            actual: tensor.shape().to_vec(),
        });
    }

    match tensor.dtype() {
        safetensors::tensor::Dtype::F32 => tensor_f32_data_to_vec(name, tensor.data()),
        safetensors::tensor::Dtype::F16 => tensor_f16_data_to_vec(name, tensor.data()),
        safetensors::tensor::Dtype::BF16 => tensor_bf16_data_to_vec(name, tensor.data()),
        dtype => Err(TinyMoeHfLoadError::UnsupportedTensorDtype {
            name: name.to_owned(),
            dtype: format!("{dtype:?}"),
        }),
    }
}

#[cfg(feature = "std")]
fn tensor_f32_data_to_vec(name: &str, data: &[u8]) -> Result<Vec<f32>, TinyMoeHfLoadError> {
    if data.len() % 4 != 0 {
        return Err(TinyMoeHfLoadError::InvalidTensorData {
            name: name.to_owned(),
            message: format!("F32 byte length must be divisible by 4, got {}", data.len()),
        });
    }

    let mut values = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

#[cfg(feature = "std")]
fn tensor_f16_data_to_vec(name: &str, data: &[u8]) -> Result<Vec<f32>, TinyMoeHfLoadError> {
    if data.len() % 2 != 0 {
        return Err(TinyMoeHfLoadError::InvalidTensorData {
            name: name.to_owned(),
            message: format!("F16 byte length must be divisible by 2, got {}", data.len()),
        });
    }

    let mut values = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        values.push(f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])));
    }
    Ok(values)
}

#[cfg(feature = "std")]
fn tensor_bf16_data_to_vec(name: &str, data: &[u8]) -> Result<Vec<f32>, TinyMoeHfLoadError> {
    if data.len() % 2 != 0 {
        return Err(TinyMoeHfLoadError::InvalidTensorData {
            name: name.to_owned(),
            message: format!(
                "BF16 byte length must be divisible by 2, got {}",
                data.len()
            ),
        });
    }

    let mut values = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
        values.push(f32::from_bits(bits << 16));
    }
    Ok(values)
}

#[cfg(feature = "std")]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits & 0x7c00) >> 10;
    let mantissa = (bits & 0x03ff) as u32;

    match exponent {
        0 => {
            if mantissa == 0 {
                f32::from_bits(sign)
            } else {
                let mut normalized = mantissa;
                let mut exponent_value = -14_i32;
                while normalized & 0x0400 == 0 {
                    normalized <<= 1;
                    exponent_value -= 1;
                }
                normalized &= 0x03ff;
                let exponent_bits = (exponent_value + 127) as u32;
                f32::from_bits(sign | (exponent_bits << 23) | (normalized << 13))
            }
        }
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13)),
        _ => {
            let exponent_bits = (u32::from(exponent) + 112) << 23;
            f32::from_bits(sign | exponent_bits | (mantissa << 13))
        }
    }
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq)]
pub struct TinyMoeReferenceOutput {
    pub routing_weights: Vec<f32>,
    pub selected_experts: Vec<usize>,
    pub selected_expert_outputs: Vec<f32>,
    pub output: Vec<f32>,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq)]
pub struct TinyMixtralLayerReferenceOutput {
    pub attention_output: Vec<f32>,
    pub hidden_after_attention: Vec<f32>,
    pub moe: TinyMoeReferenceOutput,
    pub output: Vec<f32>,
}

#[cfg(feature = "std")]
impl TinyMixtralLayerReferenceOutput {
    pub fn output_checksum(&self) -> f32 {
        self.output.iter().copied().sum()
    }

    pub fn output_max_abs(&self) -> f32 {
        self.output
            .iter()
            .fold(0.0_f32, |max_value, value| max_value.max(value.abs()))
    }
}

#[cfg(feature = "std")]
impl TinyMoeReferenceOutput {
    pub fn output_checksum(&self) -> f32 {
        self.output.iter().copied().sum()
    }

    pub fn output_max_abs(&self) -> f32 {
        self.output
            .iter()
            .fold(0.0_f32, |max_value, value| max_value.max(value.abs()))
    }
}

#[cfg(feature = "std")]
pub fn run_tiny_mixtral_layer_reference(
    fixture: &TinyMixtralLayerFixture,
) -> TinyMixtralLayerReferenceOutput {
    let config = fixture.config;
    let moe_config = config.moe;
    assert_eq!(fixture.moe.config, moe_config);
    assert_eq!(fixture.input.len(), moe_config.input_len());
    assert_eq!(fixture.input_layernorm_weight.len(), moe_config.hidden);
    assert_eq!(
        fixture.post_attention_layernorm_weight.len(),
        moe_config.hidden
    );
    assert_eq!(
        fixture.q_proj_weight.len(),
        config.query_width() * moe_config.hidden
    );
    assert_eq!(
        fixture.k_proj_weight.len(),
        config.key_value_width() * moe_config.hidden
    );
    assert_eq!(
        fixture.v_proj_weight.len(),
        config.key_value_width() * moe_config.hidden
    );
    assert_eq!(
        fixture.o_proj_weight.len(),
        moe_config.hidden * config.query_width()
    );

    let mut attention_input = vec![0.0_f32; moe_config.input_len()];
    mandrel_kernels::reference::rms_norm_f32(
        &fixture.input,
        &fixture.input_layernorm_weight,
        &mut attention_input,
        moe_config.tokens,
        moe_config.hidden,
        config.rms_norm_eps,
    );

    let mut query = vec![0.0_f32; moe_config.tokens * config.query_width()];
    let mut key = vec![0.0_f32; moe_config.tokens * config.key_value_width()];
    let mut value = vec![0.0_f32; moe_config.tokens * config.key_value_width()];
    mandrel_kernels::reference::linear_f32(
        &attention_input,
        &fixture.q_proj_weight,
        None,
        &mut query,
        moe_config.tokens,
        moe_config.hidden,
        config.query_width(),
    );
    mandrel_kernels::reference::linear_f32(
        &attention_input,
        &fixture.k_proj_weight,
        None,
        &mut key,
        moe_config.tokens,
        moe_config.hidden,
        config.key_value_width(),
    );
    mandrel_kernels::reference::linear_f32(
        &attention_input,
        &fixture.v_proj_weight,
        None,
        &mut value,
        moe_config.tokens,
        moe_config.hidden,
        config.key_value_width(),
    );

    let attention_shape = mandrel_kernels::reference::CausalAttentionShape::new(
        moe_config.tokens,
        config.query_heads,
        config.key_value_heads,
        config.head_dim,
    );
    let mut attention_context = vec![0.0_f32; query.len()];
    let mut score_scratch = vec![0.0_f32; moe_config.tokens];
    mandrel_kernels::reference::causal_self_attention_f32(
        &query,
        &key,
        &value,
        &mut score_scratch,
        &mut attention_context,
        attention_shape,
    );

    let mut attention_output = vec![0.0_f32; moe_config.input_len()];
    mandrel_kernels::reference::linear_f32(
        &attention_context,
        &fixture.o_proj_weight,
        None,
        &mut attention_output,
        moe_config.tokens,
        config.query_width(),
        moe_config.hidden,
    );
    let mut hidden_after_attention = vec![0.0_f32; moe_config.input_len()];
    mandrel_kernels::reference::add_f32(
        &fixture.input,
        &attention_output,
        &mut hidden_after_attention,
    );

    let mut moe_input = vec![0.0_f32; moe_config.input_len()];
    mandrel_kernels::reference::rms_norm_f32(
        &hidden_after_attention,
        &fixture.post_attention_layernorm_weight,
        &mut moe_input,
        moe_config.tokens,
        moe_config.hidden,
        config.rms_norm_eps,
    );
    let mut moe_fixture = fixture.moe.clone();
    moe_fixture.input = moe_input;
    let moe = run_tiny_moe_reference(&moe_fixture);

    let mut output = vec![0.0_f32; moe_config.input_len()];
    mandrel_kernels::reference::add_f32(&hidden_after_attention, &moe.output, &mut output);

    TinyMixtralLayerReferenceOutput {
        attention_output,
        hidden_after_attention,
        moe,
        output,
    }
}

#[cfg(feature = "std")]
pub fn run_tiny_moe_reference(fixture: &TinyMoeFixture) -> TinyMoeReferenceOutput {
    let config = fixture.config;
    assert!(config.top_k <= config.num_experts);
    assert_eq!(fixture.input.len(), config.input_len());
    assert_eq!(fixture.router_weight.len(), config.router_weight_len());
    assert_eq!(
        fixture.gate_weights.len(),
        config.expert_gate_or_up_weight_len()
    );
    assert_eq!(
        fixture.up_weights.len(),
        config.expert_gate_or_up_weight_len()
    );
    assert_eq!(fixture.down_weights.len(), config.expert_down_weight_len());

    let mut routing_weights = std::vec![0.0_f32; config.routing_len()];
    let mut selected_experts = std::vec![usize::MAX; config.routing_len()];
    mandrel_kernels::reference::moe_router_topk_f32(
        &fixture.input,
        &fixture.router_weight,
        &mut routing_weights,
        &mut selected_experts,
        config.tokens,
        config.hidden,
        config.num_experts,
        config.top_k,
    );

    let mut selected_expert_outputs = std::vec![0.0_f32; config.selected_expert_output_len()];
    let mut scratch = std::vec![0.0_f32; config.intermediate];
    let mut single_expert_output = std::vec![0.0_f32; config.hidden];
    let expert_gate_or_up_stride = config.intermediate * config.hidden;
    let expert_down_stride = config.hidden * config.intermediate;
    let expert_shape =
        mandrel_kernels::reference::SwigluMlpShape::new(1, config.hidden, config.intermediate);

    for token in 0..config.tokens {
        let input_row = &fixture.input[token * config.hidden..(token + 1) * config.hidden];
        for slot in 0..config.top_k {
            let expert = selected_experts[token * config.top_k + slot];
            let gate_start = expert * expert_gate_or_up_stride;
            let gate_end = gate_start + expert_gate_or_up_stride;
            let down_start = expert * expert_down_stride;
            let down_end = down_start + expert_down_stride;
            mandrel_kernels::reference::swiglu_mlp_f32(
                input_row,
                &fixture.gate_weights[gate_start..gate_end],
                &fixture.up_weights[gate_start..gate_end],
                &fixture.down_weights[down_start..down_end],
                &mut scratch,
                &mut single_expert_output,
                expert_shape,
            );

            let output_start = (token * config.top_k + slot) * config.hidden;
            let output_end = output_start + config.hidden;
            selected_expert_outputs[output_start..output_end]
                .copy_from_slice(&single_expert_output);
        }
    }

    let mut output = std::vec![0.0_f32; config.input_len()];
    mandrel_kernels::reference::moe_combine_topk_f32(
        &selected_expert_outputs,
        &routing_weights,
        &mut output,
        config.tokens,
        config.top_k,
        config.hidden,
    );

    TinyMoeReferenceOutput {
        routing_weights,
        selected_experts,
        selected_expert_outputs,
        output,
    }
}

#[cfg(feature = "std")]
fn deterministic_vector(len: usize, salt: usize, scale: f32) -> Vec<f32> {
    let mut values = Vec::with_capacity(len);
    for index in 0..len {
        values.push(deterministic_value(index + salt, scale));
    }
    values
}

#[cfg(feature = "std")]
fn deterministic_token_ids(tokens: usize, vocab_size: usize) -> Vec<usize> {
    assert!(vocab_size != 0);
    (0..tokens)
        .map(|token| (token * 7 + 3) % vocab_size)
        .collect()
}

#[cfg(feature = "std")]
fn deterministic_value(index: usize, scale: f32) -> f32 {
    let bucket = ((index * 37 + 17) % 23) as f32;
    (bucket - 11.0) * scale
}

#[cfg(test)]
mod tests {
    use super::{
        RuntimeTarget, TinyMixtralLayerConfig, TinyMixtralLayerFixture, TinyMoeConfig,
        TinyMoeFixture, example_attention_vortex_plan, example_io,
        load_tiny_mixtral_layer_fixture_from_hf_dir, load_tiny_mixtral_moe_fixture_from_hf_dir,
        run_tiny_mixtral_layer_reference, run_tiny_moe_reference,
    };
    use mandrel_device::DeviceCommand;
    use mandrel_target_ir::DeviceBackend;

    #[test]
    fn example_io_is_attention_prefill_shaped() {
        let io = example_io();

        assert_eq!(io.input.shape.dims(), &[64, 64]);
        assert_eq!(io.output.shape.dims(), &[64, 64]);
    }

    #[test]
    fn example_plan_targets_vortex_attention_prefill() {
        let plan = example_attention_vortex_plan();

        assert_eq!(plan.target.device_backend(), DeviceBackend::VortexRtl);
        assert_eq!(plan.command_buffer.len(), 2);
        assert!(matches!(
            plan.command_buffer.commands[1],
            DeviceCommand::Fence
        ));
        assert_eq!(
            RuntimeTarget::VortexRtl.device_backend(),
            DeviceBackend::VortexRtl
        );
    }

    #[test]
    fn tiny_moe_config_matches_tiny_mixtral_block_shape() {
        let config = TinyMoeConfig::tiny_random_mixtral_block();

        assert_eq!(config.tokens, 4);
        assert_eq!(config.hidden, 4);
        assert_eq!(config.intermediate, 8);
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.top_k, 2);
        assert_eq!(config.input_len(), 16);
        assert_eq!(config.router_weight_len(), 32);
    }

    #[test]
    fn runs_deterministic_tiny_moe_reference_block() {
        let fixture = TinyMoeFixture::tiny_random_mixtral_block();
        let output = run_tiny_moe_reference(&fixture);
        let repeated = run_tiny_moe_reference(&fixture);
        let config = fixture.config;

        assert_eq!(output.output, repeated.output);
        assert_eq!(output.routing_weights.len(), config.routing_len());
        assert_eq!(output.selected_experts.len(), config.routing_len());
        assert_eq!(
            output.selected_expert_outputs.len(),
            config.selected_expert_output_len()
        );
        assert_eq!(output.output.len(), config.input_len());
        assert!(output.output.iter().all(|value| value.is_finite()));
        assert!(
            output
                .selected_experts
                .iter()
                .all(|expert| *expert < config.num_experts)
        );

        for token in 0..config.tokens {
            let start = token * config.top_k;
            let end = start + config.top_k;
            let sum: f32 = output.routing_weights[start..end].iter().sum();
            assert!((sum - 1.0).abs() < 1.0e-6);
        }
    }

    #[test]
    fn runs_deterministic_tiny_mixtral_layer_reference() {
        let fixture = TinyMixtralLayerFixture::deterministic(
            TinyMixtralLayerConfig::tiny_random_mixtral_layer(4),
        );
        let output = run_tiny_mixtral_layer_reference(&fixture);
        let repeated = run_tiny_mixtral_layer_reference(&fixture);
        let config = fixture.config;

        assert_eq!(output.output, repeated.output);
        assert_eq!(output.attention_output.len(), config.moe.input_len());
        assert_eq!(output.hidden_after_attention.len(), config.moe.input_len());
        assert_eq!(output.moe.output.len(), config.moe.input_len());
        assert_eq!(output.output.len(), config.moe.input_len());
        assert!(output.output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn loads_tiny_mixtral_moe_fixture_from_hf_safetensors() {
        let dir = test_temp_dir("hf-mixtral-loader");
        reset_test_dir(&dir);
        write_test_hf_mixtral_config(&dir);
        write_test_hf_mixtral_safetensors(&dir);

        let fixture = match load_tiny_mixtral_moe_fixture_from_hf_dir(&dir, 2, 0) {
            Ok(fixture) => fixture,
            Err(error) => panic!("unexpected HF loader error: {error}"),
        };
        let output = run_tiny_moe_reference(&fixture);

        assert_eq!(fixture.config, TinyMoeConfig::new(2, 2, 3, 2, 1));
        assert_eq!(fixture.router_weight, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(fixture.gate_weights.len(), 12);
        assert_eq!(fixture.up_weights.len(), 12);
        assert_eq!(fixture.down_weights.len(), 12);
        assert_eq!(output.selected_experts.len(), 2);
        assert!(output.output.iter().all(|value| value.is_finite()));

        reset_test_dir(&dir);
    }

    #[test]
    fn loads_tiny_mixtral_layer_fixture_from_hf_safetensors() {
        let dir = test_temp_dir("hf-mixtral-layer-loader");
        reset_test_dir(&dir);
        write_test_hf_mixtral_config(&dir);
        write_test_hf_mixtral_safetensors(&dir);

        let fixture = match load_tiny_mixtral_layer_fixture_from_hf_dir(&dir, 2, 0) {
            Ok(fixture) => fixture,
            Err(error) => panic!("unexpected HF layer loader error: {error}"),
        };
        let output = run_tiny_mixtral_layer_reference(&fixture);

        assert_eq!(fixture.config.moe, TinyMoeConfig::new(2, 2, 3, 2, 1));
        assert_eq!(fixture.config.query_heads, 1);
        assert_eq!(fixture.config.key_value_heads, 1);
        assert_eq!(fixture.config.head_dim, 2);
        assert_eq!(fixture.input_layernorm_weight, [1.0, 1.0]);
        assert_eq!(fixture.post_attention_layernorm_weight, [1.0, 1.0]);
        assert_eq!(fixture.q_proj_weight, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(output.attention_output.len(), 4);
        assert_eq!(output.hidden_after_attention.len(), 4);
        assert_eq!(output.moe.output.len(), 4);
        assert_eq!(output.output.len(), 4);
        assert!(output.output.iter().all(|value| value.is_finite()));

        reset_test_dir(&dir);
    }

    fn test_temp_dir(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("mandrel-runtime-{name}-{}", std::process::id()))
    }

    fn reset_test_dir(dir: &std::path::Path) {
        match std::fs::remove_dir_all(dir) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => panic!("failed to remove test dir '{}': {error}", dir.display()),
        }
    }

    fn write_test_hf_mixtral_config(dir: &std::path::Path) {
        if let Err(error) = std::fs::create_dir_all(dir) {
            panic!("failed to create test dir '{}': {error}", dir.display());
        }
        let config = r#"{
            "model_type": "mixtral",
            "hidden_size": 2,
            "intermediate_size": 3,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 0.00001
        }"#;
        if let Err(error) = std::fs::write(dir.join("config.json"), config) {
            panic!("failed to write test config: {error}");
        }
    }

    fn write_test_hf_mixtral_safetensors(dir: &std::path::Path) {
        use safetensors::tensor::{Dtype, View, serialize};
        use std::borrow::Cow;

        struct TestTensor {
            dtype: Dtype,
            shape: Vec<usize>,
            data: Vec<u8>,
        }

        impl View for TestTensor {
            fn dtype(&self) -> Dtype {
                self.dtype
            }

            fn shape(&self) -> &[usize] {
                &self.shape
            }

            fn data(&self) -> Cow<'_, [u8]> {
                self.data.as_slice().into()
            }

            fn data_len(&self) -> usize {
                self.data.len()
            }
        }

        let tensors = vec![
            test_tensor("model.layers.0.input_layernorm.weight", &[2], &[1.0, 1.0]),
            test_tensor(
                "model.layers.0.post_attention_layernorm.weight",
                &[2],
                &[1.0, 1.0],
            ),
            test_tensor(
                "model.layers.0.self_attn.q_proj.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            test_tensor(
                "model.layers.0.self_attn.k_proj.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            test_tensor(
                "model.layers.0.self_attn.v_proj.weight",
                &[2, 2],
                &[0.5, 0.0, 0.0, 0.5],
            ),
            test_tensor(
                "model.layers.0.self_attn.o_proj.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            test_tensor(
                "model.layers.0.block_sparse_moe.gate.weight",
                &[2, 2],
                &[1.0, 0.0, 0.0, 1.0],
            ),
            test_tensor(
                "model.layers.0.block_sparse_moe.experts.0.w1.weight",
                &[3, 2],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ),
            test_tensor(
                "model.layers.0.block_sparse_moe.experts.0.w2.weight",
                &[2, 3],
                &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ),
            test_tensor(
                "model.layers.0.block_sparse_moe.experts.0.w3.weight",
                &[3, 2],
                &[1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            ),
            test_tensor(
                "model.layers.0.block_sparse_moe.experts.1.w1.weight",
                &[3, 2],
                &[0.5, 0.0, 0.0, 0.5, 0.5, 0.5],
            ),
            test_tensor(
                "model.layers.0.block_sparse_moe.experts.1.w2.weight",
                &[2, 3],
                &[0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
            ),
            test_tensor(
                "model.layers.0.block_sparse_moe.experts.1.w3.weight",
                &[3, 2],
                &[0.5, 0.5, 0.5, 0.0, 0.0, 0.5],
            ),
        ];
        let serialized = match serialize(tensors, None) {
            Ok(serialized) => serialized,
            Err(error) => panic!("failed to serialize test safetensors: {error}"),
        };
        if let Err(error) = std::fs::write(dir.join("model.safetensors"), serialized) {
            panic!("failed to write test safetensors: {error}");
        }

        fn test_tensor(name: &str, shape: &[usize], values: &[f32]) -> (String, TestTensor) {
            let mut data = Vec::with_capacity(values.len() * 4);
            for value in values {
                data.extend(value.to_le_bytes());
            }
            (
                name.to_owned(),
                TestTensor {
                    dtype: Dtype::F32,
                    shape: shape.to_vec(),
                    data,
                },
            )
        }
    }
}
