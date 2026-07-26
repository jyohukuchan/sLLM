// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Fail-closed, text-decoder model configuration contracts.
//!
//! This module deliberately recognizes only architectures for which uLLM has
//! inspected a local, complete `config.json`.  It is not a permissive
//! Hugging Face config decoder: accepting an unknown `architectures` value and
//! then trying the Qwen3 executor would make a wrong model appear runnable.

use crate::package::inspect_package;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

pub const MODEL_CONFIG_FILE: &str = "config.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitectureKind {
    Qwen3,
    Gemma4Text,
    Qwen35DenseText,
    Qwen35MoeText,
}

impl ModelArchitectureKind {
    pub const fn architecture_name(self) -> &'static str {
        match self {
            Self::Qwen3 => "Qwen3ForCausalLM",
            Self::Gemma4Text => "Gemma4ForConditionalGeneration",
            Self::Qwen35DenseText => "Qwen3_5ForConditionalGeneration",
            Self::Qwen35MoeText => "Qwen3_5MoeForConditionalGeneration",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelExecutionStatus {
    /// The existing Qwen3 full-attention decoder can consume this contract.
    Qwen3FullAttention,
    /// The existing text-only Qwen3.5 AQ4_0 runtime can consume this contract.
    Qwen35Aq4Text,
    /// Config was decoded, but intentionally has no executor fallback.
    Unimplemented {
        required_executor: &'static str,
        reason: &'static str,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoadedModelConfig {
    pub source_model_dir: PathBuf,
    pub config_path: PathBuf,
    pub config_sha256: String,
    pub model: ModelConfig,
}

impl LoadedModelConfig {
    pub fn architecture_kind(&self) -> ModelArchitectureKind {
        self.model.kind()
    }

    pub fn execution_status(&self) -> ModelExecutionStatus {
        self.model.execution_status()
    }

    pub fn require_qwen3_full_attention(&self) -> Result<&Qwen3ModelConfig, String> {
        self.model.require_qwen3_full_attention()
    }

    pub fn require_qwen35_aq4_text(&self) -> Result<&Qwen35DenseTextConfig, String> {
        self.model.require_qwen35_aq4_text()
    }

    /// Returns a descriptive error after config assembly for executors that
    /// have intentionally not been implemented yet.
    pub fn require_implemented_executor(&self) -> Result<(), String> {
        match self.execution_status() {
            ModelExecutionStatus::Qwen3FullAttention | ModelExecutionStatus::Qwen35Aq4Text => {
                Ok(())
            }
            ModelExecutionStatus::Unimplemented {
                required_executor,
                reason,
            } => Err(format!(
                "architecture {} was recognized from {} but {required_executor} is not implemented: {reason}",
                self.model.kind().architecture_name(),
                self.config_path.display(),
            )),
        }
    }
}

/// The complete text-decoder contract selected from one exact `architectures`
/// value.  Wrapper architectures retain their text decoder rather than being
/// silently flattened into Qwen3.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelConfig {
    Qwen3(Qwen3ModelConfig),
    Gemma4Text(Gemma4TextConfig),
    Qwen35DenseText(Qwen35DenseTextConfig),
    Qwen35MoeText(Qwen35MoeTextConfig),
}

impl ModelConfig {
    pub const fn kind(&self) -> ModelArchitectureKind {
        match self {
            Self::Qwen3(_) => ModelArchitectureKind::Qwen3,
            Self::Gemma4Text(_) => ModelArchitectureKind::Gemma4Text,
            Self::Qwen35DenseText(_) => ModelArchitectureKind::Qwen35DenseText,
            Self::Qwen35MoeText(_) => ModelArchitectureKind::Qwen35MoeText,
        }
    }

    pub const fn execution_status(&self) -> ModelExecutionStatus {
        match self {
            Self::Qwen3(_) => ModelExecutionStatus::Qwen3FullAttention,
            Self::Qwen35DenseText(_) => ModelExecutionStatus::Qwen35Aq4Text,
            Self::Gemma4Text(_) => ModelExecutionStatus::Unimplemented {
                required_executor: "Gemma4TextExecutor",
                reason: "local/full attention, mixed head widths, extra norms, PLE, tied embedding, and logit soft-cap are not implemented",
            },
            Self::Qwen35MoeText(_) => ModelExecutionStatus::Unimplemented {
                required_executor: "Qwen35MoeExecutor",
                reason: "top-k routing, gather/scatter, grouped expert GEMM, weighted reduction, and shared expert execution are not implemented",
            },
        }
    }

    pub fn require_qwen3_full_attention(&self) -> Result<&Qwen3ModelConfig, String> {
        let Self::Qwen3(config) = self else {
            return Err(format!(
                "Qwen3 full-attention executor requires {}, got {}",
                ModelArchitectureKind::Qwen3.architecture_name(),
                self.kind().architecture_name()
            ));
        };
        config.validate_existing_executor()?;
        Ok(config)
    }

    pub fn require_qwen35_aq4_text(&self) -> Result<&Qwen35DenseTextConfig, String> {
        let Self::Qwen35DenseText(config) = self else {
            return Err(format!(
                "Qwen3.5 AQ4_0 text executor requires {}, got {}",
                ModelArchitectureKind::Qwen35DenseText.architecture_name(),
                self.kind().architecture_name()
            ));
        };
        config.validate_existing_aq4_executor()?;
        Ok(config)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecoderShapeConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub tie_word_embeddings: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseMlpConfig {
    pub activation: String,
    pub intermediate_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionConfig {
    pub bias: bool,
    pub dropout: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RmsNormWeightConvention {
    DirectWeight,
    OnePlusWeight,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderLayerKind {
    FullAttention,
    SlidingAttention,
    LinearAttention,
}

impl DecoderLayerKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FullAttention => "full_attention",
            Self::SlidingAttention => "sliding_attention",
            Self::LinearAttention => "linear_attention",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3ModelConfig {
    pub decoder: DecoderShapeConfig,
    pub attention: AttentionConfig,
    pub dense_mlp: DenseMlpConfig,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub max_window_layers: usize,
    pub use_sliding_window: bool,
    pub sliding_window: Option<usize>,
    pub norm_weight_convention: RmsNormWeightConvention,
}

impl Qwen3ModelConfig {
    /// Preserves the pre-BF Qwen3 loader's rotary width calculation.  The
    /// source config does not contain a partial rotary field for this model,
    /// so changing this here would be an execution-semantic change rather
    /// than config plumbing.
    pub fn legacy_runtime_rotary_dim(&self) -> Result<usize, String> {
        let candidate = if self.decoder.head_dim >= 4 {
            self.decoder.head_dim / 4
        } else {
            self.decoder.head_dim
        };
        let rotary_dim = candidate - (candidate % 2);
        if rotary_dim == 0 {
            return Err(format!(
                "legacy Qwen3 runtime rotary_dim is zero for head_dim={}",
                self.decoder.head_dim
            ));
        }
        Ok(rotary_dim)
    }

    pub fn validate_existing_executor(&self) -> Result<(), String> {
        if self.dense_mlp.activation != "silu" {
            return Err(format!(
                "Qwen3 full-attention executor supports hidden_act=silu, got {:?}",
                self.dense_mlp.activation
            ));
        }
        if self.attention.bias {
            return Err("Qwen3 full-attention executor does not implement attention bias".into());
        }
        if self.attention.dropout != 0.0 {
            return Err(format!(
                "Qwen3 full-attention executor requires attention_dropout=0, got {}",
                self.attention.dropout
            ));
        }
        if self.decoder.tie_word_embeddings {
            return Err(
                "Qwen3 full-attention executor does not implement tied embedding/lm-head loading"
                    .into(),
            );
        }
        if self.norm_weight_convention != RmsNormWeightConvention::DirectWeight {
            return Err("Qwen3 full-attention executor requires direct RMSNorm weights".into());
        }
        if self.use_sliding_window || self.sliding_window.is_some() {
            return Err(
                "Qwen3 full-attention executor does not implement sliding-window attention".into(),
            );
        }
        if self.max_window_layers != self.decoder.num_hidden_layers {
            return Err(format!(
                "Qwen3 full-attention executor requires max_window_layers == num_hidden_layers, got {} != {}",
                self.max_window_layers, self.decoder.num_hidden_layers
            ));
        }
        let q_width = self
            .decoder
            .num_attention_heads
            .checked_mul(self.decoder.head_dim)
            .ok_or_else(|| "Qwen3 q attention width overflows".to_string())?;
        if q_width != self.decoder.hidden_size {
            return Err(format!(
                "Qwen3 full-attention executor requires q_heads * head_dim == hidden_size, got {} * {} != {}",
                self.decoder.num_attention_heads, self.decoder.head_dim, self.decoder.hidden_size
            ));
        }
        if !self
            .decoder
            .num_attention_heads
            .is_multiple_of(self.decoder.num_key_value_heads)
        {
            return Err(format!(
                "Qwen3 full-attention executor requires q heads divisible by KV heads: {} / {}",
                self.decoder.num_attention_heads, self.decoder.num_key_value_heads
            ));
        }
        self.legacy_runtime_rotary_dim()?;
        Ok(())
    }

    pub fn validate_runtime_layer_shape(
        &self,
        layer_index: usize,
        hidden: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        value_dim: usize,
        intermediate: usize,
    ) -> Result<(), String> {
        if layer_index >= self.decoder.num_hidden_layers {
            return Err(format!(
                "Qwen3 package layer {layer_index} is outside config num_hidden_layers={}",
                self.decoder.num_hidden_layers
            ));
        }
        let expected = (
            self.decoder.hidden_size,
            self.decoder.num_attention_heads,
            self.decoder.num_key_value_heads,
            self.decoder.head_dim,
            self.decoder.head_dim,
            self.dense_mlp.intermediate_size,
        );
        let actual = (hidden, q_heads, kv_heads, head_dim, value_dim, intermediate);
        if actual != expected {
            return Err(format!(
                "Qwen3 package layer {layer_index} disagrees with config: expected hidden/q_heads/kv_heads/head_dim/value_dim/intermediate={expected:?}, got {actual:?}"
            ));
        }
        Ok(())
    }

    /// Checks a static executor's model-wide constants against the source
    /// configuration before it allocates device buffers.  SQ8_0 uses a
    /// deliberately fixed Qwen3-14B kernel stack; this turns that formerly
    /// implicit assumption into an explicit config contract without making
    /// the generic Qwen3 loader depend on SQ8_0 types.
    #[allow(clippy::too_many_arguments)]
    pub fn validate_static_runtime_shape(
        &self,
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        value_dim: usize,
        vocab_size: usize,
    ) -> Result<(), String> {
        self.validate_existing_executor()?;
        let expected = (
            self.decoder.hidden_size,
            self.decoder.num_hidden_layers,
            self.decoder.num_attention_heads,
            self.decoder.num_key_value_heads,
            self.decoder.head_dim,
            self.decoder.head_dim,
            self.decoder.vocab_size,
        );
        let actual = (
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            value_dim,
            vocab_size,
        );
        if actual != expected {
            return Err(format!(
                "Qwen3 static runtime disagrees with config: expected hidden/layers/q_heads/kv_heads/head_dim/value_dim/vocab={expected:?}, got {actual:?}"
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GemmaRopeConfig {
    pub rope_type: String,
    pub rope_theta: f32,
    pub partial_rotary_factor: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4TextConfig {
    pub decoder: DecoderShapeConfig,
    pub attention: AttentionConfig,
    pub dense_mlp: DenseMlpConfig,
    pub layer_types: Vec<DecoderLayerKind>,
    pub local_head_dim: usize,
    pub global_head_dim: usize,
    pub sliding_window: usize,
    pub sliding_rope: GemmaRopeConfig,
    pub full_rope: GemmaRopeConfig,
    pub attention_k_eq_v: bool,
    pub num_kv_shared_layers: usize,
    pub use_double_wide_mlp: bool,
    pub hidden_size_per_layer_input: usize,
    pub vocab_size_per_layer_input: usize,
    pub final_logit_softcapping: f32,
    pub enable_moe_block: bool,
    pub norm_weight_convention: RmsNormWeightConvention,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35RopeConfig {
    pub rope_type: String,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub mrope_interleaved: bool,
    pub mrope_sections: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearAttentionConfig {
    pub conv_kernel_dim: usize,
    pub key_head_dim: usize,
    pub num_key_heads: usize,
    pub num_value_heads: usize,
    pub value_head_dim: usize,
    pub state_dtype: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MtpConfig {
    pub num_hidden_layers: usize,
    pub use_dedicated_embeddings: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35HybridTextConfig {
    pub decoder: DecoderShapeConfig,
    pub attention: AttentionConfig,
    pub activation: String,
    pub layer_types: Vec<DecoderLayerKind>,
    pub full_attention_interval: usize,
    pub attn_output_gate: bool,
    pub linear_attention: LinearAttentionConfig,
    pub rope: Qwen35RopeConfig,
    pub mlp_only_layers: Vec<usize>,
    pub mtp: MtpConfig,
    pub norm_weight_convention: RmsNormWeightConvention,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35DenseTextConfig {
    pub hybrid: Qwen35HybridTextConfig,
    pub dense_mlp: DenseMlpConfig,
}

impl Qwen35DenseTextConfig {
    pub fn validate_existing_aq4_executor(&self) -> Result<(), String> {
        let config = &self.hybrid;
        if config.activation != "silu" {
            return Err(format!(
                "Qwen3.5 AQ4_0 text executor supports hidden_act=silu, got {:?}",
                config.activation
            ));
        }
        if self.dense_mlp.activation != config.activation {
            return Err("Qwen3.5 dense activation contract is internally inconsistent".into());
        }
        if config.attention.bias {
            return Err("Qwen3.5 AQ4_0 text executor does not implement attention bias".into());
        }
        if config.attention.dropout != 0.0 {
            return Err(format!(
                "Qwen3.5 AQ4_0 text executor requires attention_dropout=0, got {}",
                config.attention.dropout
            ));
        }
        if !config.attn_output_gate {
            return Err("Qwen3.5 AQ4_0 text executor requires attn_output_gate=true".into());
        }
        if !config.mlp_only_layers.is_empty() {
            return Err(format!(
                "Qwen3.5 AQ4_0 text executor does not implement mlp_only_layers={:?}",
                config.mlp_only_layers
            ));
        }
        if config.norm_weight_convention != RmsNormWeightConvention::OnePlusWeight {
            return Err("Qwen3.5 AQ4_0 text executor requires 1 + weight RMSNorm".into());
        }
        if !config
            .decoder
            .num_attention_heads
            .is_multiple_of(config.decoder.num_key_value_heads)
        {
            return Err(format!(
                "Qwen3.5 AQ4_0 q heads must be divisible by KV heads: {} / {}",
                config.decoder.num_attention_heads, config.decoder.num_key_value_heads
            ));
        }
        Ok(())
    }

    pub fn validate_package_layers(
        &self,
        vocab: usize,
        hidden: usize,
        layers: &[(usize, DecoderLayerKind)],
    ) -> Result<(), String> {
        self.validate_existing_aq4_executor()?;
        if vocab != self.hybrid.decoder.vocab_size || hidden != self.hybrid.decoder.hidden_size {
            return Err(format!(
                "Qwen3.5 AQ4_0 package embedding shape disagrees with config: package vocab/hidden=({vocab},{hidden}), config=({},{})",
                self.hybrid.decoder.vocab_size, self.hybrid.decoder.hidden_size
            ));
        }
        if layers.len() != self.hybrid.decoder.num_hidden_layers {
            return Err(format!(
                "Qwen3.5 AQ4_0 package has {} decoder layers, config declares {}",
                layers.len(),
                self.hybrid.decoder.num_hidden_layers
            ));
        }
        for (position, (layer_index, kind)) in layers.iter().enumerate() {
            if *layer_index != position {
                return Err(format!(
                    "Qwen3.5 AQ4_0 package decoder layers must be contiguous from zero: position={position} layer_index={layer_index}"
                ));
            }
            let expected = self.hybrid.layer_types[position];
            if *kind != expected {
                return Err(format!(
                    "Qwen3.5 AQ4_0 package layer {layer_index} kind={} disagrees with config layer_types={}",
                    kind.as_str(),
                    expected.as_str()
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35MoeConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub expert_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub router_aux_loss_coef: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35MoeTextConfig {
    pub hybrid: Qwen35HybridTextConfig,
    pub moe: Qwen35MoeConfig,
}

/// Reads and resolves `model_dir/config.json`.
pub fn load_model_config_from_dir(
    model_dir: impl AsRef<Path>,
) -> Result<LoadedModelConfig, String> {
    let requested_dir = model_dir.as_ref();
    let source_model_dir = fs::canonicalize(requested_dir).map_err(|err| {
        format!(
            "failed to canonicalize source model directory {}: {err}",
            requested_dir.display()
        )
    })?;
    if !fs::metadata(&source_model_dir)
        .map_err(|err| {
            format!(
                "failed to inspect source model directory {}: {err}",
                source_model_dir.display()
            )
        })?
        .is_dir()
    {
        return Err(format!(
            "source model path is not a directory: {}",
            source_model_dir.display()
        ));
    }
    let config_path = source_model_dir.join(MODEL_CONFIG_FILE);
    let bytes = fs::read(&config_path).map_err(|err| {
        format!(
            "failed to read model config {}: {err}",
            config_path.display()
        )
    })?;
    if bytes.is_empty() {
        return Err(format!("model config is empty: {}", config_path.display()));
    }
    let value: Value = serde_json::from_slice(&bytes).map_err(|err| {
        format!(
            "failed to parse model config {}: {err}",
            config_path.display()
        )
    })?;
    let model = parse_model_config_value(&value)
        .map_err(|err| format!("invalid model config {}: {err}", config_path.display()))?;
    let mut digest = Sha256::new();
    digest.update(&bytes);
    Ok(LoadedModelConfig {
        source_model_dir,
        config_path,
        config_sha256: format!("{:x}", digest.finalize()),
        model,
    })
}

/// Resolves the source model directory declared by a uLLM package before
/// reading its config.  A package with no source model directory cannot be
/// architecture-dispatched safely and is rejected rather than defaulting to
/// Qwen3.
pub fn load_model_config_from_package(
    package_dir: impl AsRef<Path>,
) -> Result<LoadedModelConfig, String> {
    let package_dir = package_dir.as_ref();
    let summary = inspect_package(package_dir).map_err(|err| {
        format!(
            "failed to inspect package {} before reading model config: {err}",
            package_dir.display()
        )
    })?;
    let source_model_dir = summary.source_model_dir.ok_or_else(|| {
        format!(
            "package {} has no source_model_dir; refusing to assume Qwen3 architecture",
            package_dir.display()
        )
    })?;
    if source_model_dir.is_empty() {
        return Err(format!(
            "package {} has an empty source_model_dir; refusing to assume Qwen3 architecture",
            package_dir.display()
        ));
    }
    load_model_config_from_dir(&source_model_dir).map_err(|err| {
        format!(
            "failed to resolve architecture config for package {}: {err}",
            package_dir.display()
        )
    })
}

fn parse_model_config_value(root: &Value) -> Result<ModelConfig, String> {
    let architecture = required_single_architecture(root)?;
    let root_model_type = required_string(root, "model_type", "root")?;
    match architecture.as_str() {
        "Qwen3ForCausalLM" => {
            require_exact_string(&root_model_type, "qwen3", "root model_type")?;
            parse_qwen3(root).map(ModelConfig::Qwen3)
        }
        "Gemma4ForConditionalGeneration" => {
            require_exact_string(&root_model_type, "gemma4", "root model_type")?;
            parse_gemma4_text(root).map(ModelConfig::Gemma4Text)
        }
        "Qwen3_5ForConditionalGeneration" => {
            require_exact_string(&root_model_type, "qwen3_5", "root model_type")?;
            parse_qwen35_dense_text(root).map(ModelConfig::Qwen35DenseText)
        }
        "Qwen3_5MoeForConditionalGeneration" => {
            require_exact_string(&root_model_type, "qwen3_5_moe", "root model_type")?;
            parse_qwen35_moe_text(root).map(ModelConfig::Qwen35MoeText)
        }
        _ => Err(format!(
            "unsupported architectures value {architecture:?}; supported values are Qwen3ForCausalLM, Gemma4ForConditionalGeneration, Qwen3_5ForConditionalGeneration, and Qwen3_5MoeForConditionalGeneration"
        )),
    }
}

fn parse_qwen3(root: &Value) -> Result<Qwen3ModelConfig, String> {
    let decoder = parse_decoder_shape(
        root,
        "root",
        required_bool(root, "tie_word_embeddings", "root")?,
    )?;
    let attention = parse_attention(root, "root")?;
    let dense_mlp = DenseMlpConfig {
        activation: required_string(root, "hidden_act", "root")?,
        intermediate_size: required_usize(root, "intermediate_size", "root")?,
    };
    let rope_theta = required_f32(root, "rope_theta", "root")?;
    require_null(root, "rope_scaling", "root")?;
    Ok(Qwen3ModelConfig {
        decoder,
        attention,
        dense_mlp,
        rope_theta,
        max_position_embeddings: required_usize(root, "max_position_embeddings", "root")?,
        max_window_layers: required_usize(root, "max_window_layers", "root")?,
        use_sliding_window: required_bool(root, "use_sliding_window", "root")?,
        sliding_window: optional_usize(root, "sliding_window", "root")?,
        norm_weight_convention: RmsNormWeightConvention::DirectWeight,
    })
}

fn parse_gemma4_text(root: &Value) -> Result<Gemma4TextConfig, String> {
    let text = required_object_field(root, "text_config", "root")?;
    let tied = required_bool(text, "tie_word_embeddings", "text_config")?;
    let root_tied = required_bool(root, "tie_word_embeddings", "root")?;
    if tied != root_tied {
        return Err(format!(
            "Gemma4 root/text tie_word_embeddings disagree: root={root_tied} text_config={tied}"
        ));
    }
    let decoder = parse_decoder_shape(text, "text_config", tied)?;
    require_exact_string(
        &decoder.model_type,
        "gemma4_text",
        "Gemma4 text_config model_type",
    )?;
    let attention = parse_attention(text, "text_config")?;
    let dense_mlp = DenseMlpConfig {
        activation: required_string(text, "hidden_activation", "text_config")?,
        intermediate_size: required_usize(text, "intermediate_size", "text_config")?,
    };
    let layer_types = parse_layer_types(
        text,
        "text_config",
        &["sliding_attention", "full_attention"],
        decoder.num_hidden_layers,
    )?;
    let rope_parameters = required_object_field(text, "rope_parameters", "text_config")?;
    let sliding_rope = parse_gemma_rope(
        required_object_field(
            rope_parameters,
            "sliding_attention",
            "text_config.rope_parameters",
        )?,
        "text_config.rope_parameters.sliding_attention",
    )?;
    let full_rope = parse_gemma_rope(
        required_object_field(
            rope_parameters,
            "full_attention",
            "text_config.rope_parameters",
        )?,
        "text_config.rope_parameters.full_attention",
    )?;
    let local_head_dim = required_usize(text, "head_dim", "text_config")?;
    let global_head_dim = required_usize(text, "global_head_dim", "text_config")?;
    if local_head_dim != decoder.head_dim {
        return Err(format!(
            "Gemma4 text_config head_dim disagrees with decoder head_dim: {local_head_dim} != {}",
            decoder.head_dim
        ));
    }
    Ok(Gemma4TextConfig {
        decoder,
        attention,
        dense_mlp,
        layer_types,
        local_head_dim,
        global_head_dim,
        sliding_window: required_usize(text, "sliding_window", "text_config")?,
        sliding_rope,
        full_rope,
        attention_k_eq_v: required_bool(text, "attention_k_eq_v", "text_config")?,
        num_kv_shared_layers: required_usize(text, "num_kv_shared_layers", "text_config")?,
        use_double_wide_mlp: required_bool(text, "use_double_wide_mlp", "text_config")?,
        hidden_size_per_layer_input: required_usize(
            text,
            "hidden_size_per_layer_input",
            "text_config",
        )?,
        vocab_size_per_layer_input: required_usize(
            text,
            "vocab_size_per_layer_input",
            "text_config",
        )?,
        final_logit_softcapping: required_f32(text, "final_logit_softcapping", "text_config")?,
        enable_moe_block: required_bool(text, "enable_moe_block", "text_config")?,
        norm_weight_convention: RmsNormWeightConvention::DirectWeight,
    })
}

fn parse_qwen35_dense_text(root: &Value) -> Result<Qwen35DenseTextConfig, String> {
    let root_tied = required_bool(root, "tie_word_embeddings", "root")?;
    let text = required_object_field(root, "text_config", "root")?;
    let hybrid = parse_qwen35_hybrid_text(text, root_tied, "qwen3_5_text")?;
    let dense_mlp = DenseMlpConfig {
        activation: hybrid.activation.clone(),
        intermediate_size: required_usize(text, "intermediate_size", "text_config")?,
    };
    Ok(Qwen35DenseTextConfig { hybrid, dense_mlp })
}

fn parse_qwen35_moe_text(root: &Value) -> Result<Qwen35MoeTextConfig, String> {
    let root_tied = required_bool(root, "tie_word_embeddings", "root")?;
    let text = required_object_field(root, "text_config", "root")?;
    let hybrid = parse_qwen35_hybrid_text(text, root_tied, "qwen3_5_moe_text")?;
    let moe = Qwen35MoeConfig {
        num_experts: required_usize(text, "num_experts", "text_config")?,
        num_experts_per_tok: required_usize(text, "num_experts_per_tok", "text_config")?,
        expert_intermediate_size: required_usize(text, "moe_intermediate_size", "text_config")?,
        shared_expert_intermediate_size: required_usize(
            text,
            "shared_expert_intermediate_size",
            "text_config",
        )?,
        router_aux_loss_coef: required_f32(text, "router_aux_loss_coef", "text_config")?,
    };
    if moe.num_experts_per_tok > moe.num_experts {
        return Err(format!(
            "Qwen3.5 MoE num_experts_per_tok={} exceeds num_experts={}",
            moe.num_experts_per_tok, moe.num_experts
        ));
    }
    Ok(Qwen35MoeTextConfig { hybrid, moe })
}

fn parse_qwen35_hybrid_text(
    text: &Value,
    tie_word_embeddings: bool,
    expected_model_type: &str,
) -> Result<Qwen35HybridTextConfig, String> {
    let decoder = parse_decoder_shape(text, "text_config", tie_word_embeddings)?;
    require_exact_string(
        &decoder.model_type,
        expected_model_type,
        "Qwen3.5 text_config model_type",
    )?;
    let attention = parse_attention(text, "text_config")?;
    let activation = required_string(text, "hidden_act", "text_config")?;
    let layer_types = parse_layer_types(
        text,
        "text_config",
        &["linear_attention", "full_attention"],
        decoder.num_hidden_layers,
    )?;
    let rope_object = required_object_field(text, "rope_parameters", "text_config")?;
    let mrope_sections =
        required_usize_array(rope_object, "mrope_section", "text_config.rope_parameters")?;
    if mrope_sections.is_empty() {
        return Err("text_config.rope_parameters.mrope_section must not be empty".into());
    }
    let rope = Qwen35RopeConfig {
        rope_type: required_string(rope_object, "rope_type", "text_config.rope_parameters")?,
        rope_theta: required_f32(rope_object, "rope_theta", "text_config.rope_parameters")?,
        partial_rotary_factor: required_f32(
            rope_object,
            "partial_rotary_factor",
            "text_config.rope_parameters",
        )?,
        mrope_interleaved: required_bool(
            rope_object,
            "mrope_interleaved",
            "text_config.rope_parameters",
        )?,
        mrope_sections,
    };
    let linear_attention = LinearAttentionConfig {
        conv_kernel_dim: required_usize(text, "linear_conv_kernel_dim", "text_config")?,
        key_head_dim: required_usize(text, "linear_key_head_dim", "text_config")?,
        num_key_heads: required_usize(text, "linear_num_key_heads", "text_config")?,
        num_value_heads: required_usize(text, "linear_num_value_heads", "text_config")?,
        value_head_dim: required_usize(text, "linear_value_head_dim", "text_config")?,
        state_dtype: required_string(text, "mamba_ssm_dtype", "text_config")?,
    };
    let mtp = MtpConfig {
        num_hidden_layers: required_usize(text, "mtp_num_hidden_layers", "text_config")?,
        use_dedicated_embeddings: required_bool(
            text,
            "mtp_use_dedicated_embeddings",
            "text_config",
        )?,
    };
    Ok(Qwen35HybridTextConfig {
        decoder,
        attention,
        activation,
        layer_types,
        full_attention_interval: required_usize(text, "full_attention_interval", "text_config")?,
        attn_output_gate: required_bool(text, "attn_output_gate", "text_config")?,
        linear_attention,
        rope,
        mlp_only_layers: required_usize_array(text, "mlp_only_layers", "text_config")?,
        mtp,
        norm_weight_convention: RmsNormWeightConvention::OnePlusWeight,
    })
}

fn parse_decoder_shape(
    object: &Value,
    scope: &str,
    tie_word_embeddings: bool,
) -> Result<DecoderShapeConfig, String> {
    let decoder = DecoderShapeConfig {
        model_type: required_string(object, "model_type", scope)?,
        hidden_size: required_usize(object, "hidden_size", scope)?,
        num_hidden_layers: required_usize(object, "num_hidden_layers", scope)?,
        num_attention_heads: required_usize(object, "num_attention_heads", scope)?,
        num_key_value_heads: required_usize(object, "num_key_value_heads", scope)?,
        head_dim: required_usize(object, "head_dim", scope)?,
        rms_norm_eps: required_f32(object, "rms_norm_eps", scope)?,
        vocab_size: required_usize(object, "vocab_size", scope)?,
        tie_word_embeddings,
    };
    if !decoder
        .num_attention_heads
        .is_multiple_of(decoder.num_key_value_heads)
    {
        return Err(format!(
            "{scope} num_attention_heads={} must be divisible by num_key_value_heads={}",
            decoder.num_attention_heads, decoder.num_key_value_heads
        ));
    }
    Ok(decoder)
}

fn parse_attention(object: &Value, scope: &str) -> Result<AttentionConfig, String> {
    Ok(AttentionConfig {
        bias: required_bool(object, "attention_bias", scope)?,
        dropout: required_nonnegative_f32(object, "attention_dropout", scope)?,
    })
}

fn parse_gemma_rope(object: &Value, scope: &str) -> Result<GemmaRopeConfig, String> {
    let partial_rotary_factor = optional_f32(object, "partial_rotary_factor", scope)?;
    Ok(GemmaRopeConfig {
        rope_type: required_string(object, "rope_type", scope)?,
        rope_theta: required_f32(object, "rope_theta", scope)?,
        partial_rotary_factor,
    })
}

fn parse_layer_types(
    object: &Value,
    scope: &str,
    accepted: &[&str],
    expected_len: usize,
) -> Result<Vec<DecoderLayerKind>, String> {
    let values = required_string_array(object, "layer_types", scope)?;
    if values.len() != expected_len {
        return Err(format!(
            "{scope}.layer_types length {} does not match num_hidden_layers={expected_len}",
            values.len()
        ));
    }
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            if !accepted.contains(&value.as_str()) {
                return Err(format!(
                    "{scope}.layer_types[{index}]={value:?} is not accepted for this architecture; accepted={accepted:?}"
                ));
            }
            match value.as_str() {
                "full_attention" => Ok(DecoderLayerKind::FullAttention),
                "sliding_attention" => Ok(DecoderLayerKind::SlidingAttention),
                "linear_attention" => Ok(DecoderLayerKind::LinearAttention),
                _ => Err(format!(
                    "{scope}.layer_types[{index}]={value:?} has no internal descriptor"
                )),
            }
        })
        .collect()
}

fn required_single_architecture(root: &Value) -> Result<String, String> {
    let values = required_string_array(root, "architectures", "root")?;
    if values.len() != 1 {
        return Err(format!(
            "root.architectures must contain exactly one architecture, got {values:?}"
        ));
    }
    Ok(values
        .into_iter()
        .next()
        .expect("checked architecture length"))
}

fn required_object_field<'a>(
    object: &'a Value,
    field: &str,
    scope: &str,
) -> Result<&'a Value, String> {
    object
        .get(field)
        .and_then(Value::as_object)
        .map(|_| object.get(field).expect("object was just checked"))
        .ok_or_else(|| format!("{scope}.{field} must be an object"))
}

fn required_string(object: &Value, field: &str, scope: &str) -> Result<String, String> {
    let value = object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{scope}.{field} must be a string"))?;
    if value.is_empty() {
        return Err(format!("{scope}.{field} must not be empty"));
    }
    Ok(value.to_string())
}

fn required_bool(object: &Value, field: &str, scope: &str) -> Result<bool, String> {
    object
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{scope}.{field} must be a boolean"))
}

fn require_null(object: &Value, field: &str, scope: &str) -> Result<(), String> {
    match object.get(field) {
        Some(value) if value.is_null() => Ok(()),
        Some(_) => Err(format!(
            "{scope}.{field} must be null; Qwen3 rope scaling is not implemented"
        )),
        None => Err(format!("{scope}.{field} must be present and null")),
    }
}

fn required_usize(object: &Value, field: &str, scope: &str) -> Result<usize, String> {
    let value = object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{scope}.{field} must be a positive integer"))?;
    let value = usize::try_from(value)
        .map_err(|_| format!("{scope}.{field}={value} exceeds this host usize"))?;
    if value == 0 {
        return Err(format!("{scope}.{field} must be greater than zero"));
    }
    Ok(value)
}

fn required_f32(object: &Value, field: &str, scope: &str) -> Result<f32, String> {
    let value = object
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{scope}.{field} must be a finite positive number"))?;
    let value = value as f32;
    if !value.is_finite() || value <= 0.0 {
        return Err(format!(
            "{scope}.{field} must be a finite positive number, got {value}"
        ));
    }
    Ok(value)
}

fn required_nonnegative_f32(object: &Value, field: &str, scope: &str) -> Result<f32, String> {
    let value = object
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{scope}.{field} must be a finite non-negative number"))?;
    let value = value as f32;
    if !value.is_finite() || value < 0.0 {
        return Err(format!(
            "{scope}.{field} must be a finite non-negative number, got {value}"
        ));
    }
    Ok(value)
}

fn optional_f32(object: &Value, field: &str, scope: &str) -> Result<Option<f32>, String> {
    let Some(value) = object.get(field) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let value = value
        .as_f64()
        .ok_or_else(|| format!("{scope}.{field} must be a finite positive number or null"))?
        as f32;
    if !value.is_finite() || value <= 0.0 {
        return Err(format!(
            "{scope}.{field} must be a finite positive number, got {value}"
        ));
    }
    Ok(Some(value))
}

fn optional_usize(object: &Value, field: &str, scope: &str) -> Result<Option<usize>, String> {
    let Some(value) = object.get(field) else {
        return Err(format!(
            "{scope}.{field} must be a positive integer or null"
        ));
    };
    if value.is_null() {
        return Ok(None);
    }
    let value = value
        .as_u64()
        .ok_or_else(|| format!("{scope}.{field} must be a positive integer or null"))?;
    let value = usize::try_from(value)
        .map_err(|_| format!("{scope}.{field}={value} exceeds this host usize"))?;
    if value == 0 {
        return Err(format!(
            "{scope}.{field} must be greater than zero when present"
        ));
    }
    Ok(Some(value))
}

fn required_string_array(object: &Value, field: &str, scope: &str) -> Result<Vec<String>, String> {
    let values = object
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{scope}.{field} must be an array"))?;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_str()
                .filter(|value| !value.is_empty())
                .map(ToString::to_string)
                .ok_or_else(|| format!("{scope}.{field}[{index}] must be a non-empty string"))
        })
        .collect()
}

fn required_usize_array(object: &Value, field: &str, scope: &str) -> Result<Vec<usize>, String> {
    let values = object
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{scope}.{field} must be an array"))?;
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let value = value.as_u64().ok_or_else(|| {
                format!("{scope}.{field}[{index}] must be a non-negative integer")
            })?;
            usize::try_from(value)
                .map_err(|_| format!("{scope}.{field}[{index}]={value} exceeds this host usize"))
        })
        .collect()
}

fn require_exact_string(actual: &str, expected: &str, label: &str) -> Result<(), String> {
    if actual == expected {
        Ok(())
    } else {
        Err(format!("{label} must be {expected:?}, got {actual:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn qwen3_config() -> Value {
        json!({
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 17408,
            "hidden_act": "silu",
            "rms_norm_eps": 0.000001,
            "rope_theta": 1000000,
            "rope_scaling": null,
            "max_position_embeddings": 40960,
            "max_window_layers": 40,
            "use_sliding_window": false,
            "sliding_window": null,
            "vocab_size": 151936,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "attention_dropout": 0.0
        })
    }

    fn gemma4_config() -> Value {
        let layer_types = (0..35)
            .map(|index| {
                if index % 5 == 4 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect::<Vec<_>>();
        json!({
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 1536,
                "num_hidden_layers": 35,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "global_head_dim": 512,
                "intermediate_size": 6144,
                "hidden_activation": "gelu_pytorch_tanh",
                "rms_norm_eps": 0.000001,
                "vocab_size": 262144,
                "vocab_size_per_layer_input": 262144,
                "tie_word_embeddings": true,
                "attention_bias": false,
                "attention_dropout": 0.0,
                "attention_k_eq_v": false,
                "layer_types": layer_types,
                "sliding_window": 512,
                "rope_parameters": {
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10000},
                    "full_attention": {"rope_type": "proportional", "rope_theta": 1000000, "partial_rotary_factor": 0.25}
                },
                "num_kv_shared_layers": 20,
                "use_double_wide_mlp": true,
                "hidden_size_per_layer_input": 256,
                "final_logit_softcapping": 30.0,
                "enable_moe_block": false
            }
        })
    }

    fn qwen35_config(moe: bool) -> Value {
        let architecture = if moe {
            "Qwen3_5MoeForConditionalGeneration"
        } else {
            "Qwen3_5ForConditionalGeneration"
        };
        let model_type = if moe { "qwen3_5_moe" } else { "qwen3_5" };
        let text_model_type = if moe {
            "qwen3_5_moe_text"
        } else {
            "qwen3_5_text"
        };
        let mut text = json!({
            "model_type": text_model_type,
            "hidden_size": if moe { 2048 } else { 4096 },
            "num_hidden_layers": if moe { 40 } else { 32 },
            "num_attention_heads": 16,
            "num_key_value_heads": if moe { 2 } else { 4 },
            "head_dim": 256,
            "hidden_act": "silu",
            "rms_norm_eps": 0.000001,
            "vocab_size": 248320,
            "attention_bias": false,
            "attention_dropout": 0.0,
            "layer_types": if moe { vec!["linear_attention", "full_attention"] } else { vec!["linear_attention", "full_attention"] },
            "full_attention_interval": 4,
            "attn_output_gate": true,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "mamba_ssm_dtype": "float32",
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10]
            },
            "mlp_only_layers": [],
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false
        });
        let layers = text["num_hidden_layers"].as_u64().unwrap() as usize;
        let layer_types = (0..layers)
            .map(|index| {
                if index % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect::<Vec<_>>();
        text["layer_types"] = json!(layer_types);
        if moe {
            text["num_experts"] = json!(256);
            text["num_experts_per_tok"] = json!(8);
            text["moe_intermediate_size"] = json!(512);
            text["shared_expert_intermediate_size"] = json!(512);
            text["router_aux_loss_coef"] = json!(0.001);
        } else {
            text["intermediate_size"] = json!(12288);
        }
        json!({
            "architectures": [architecture],
            "model_type": model_type,
            "tie_word_embeddings": false,
            "text_config": text
        })
    }

    #[test]
    fn qwen3_contract_reproduces_current_geometry() {
        let config = parse_model_config_value(&qwen3_config()).unwrap();
        let ModelConfig::Qwen3(config) = config else {
            panic!("expected Qwen3 config");
        };
        assert_eq!(config.decoder.hidden_size, 5120);
        assert_eq!(config.decoder.num_attention_heads, 40);
        assert_eq!(config.decoder.num_key_value_heads, 8);
        assert_eq!(config.decoder.head_dim, 128);
        assert_eq!(config.dense_mlp.intermediate_size, 17408);
        assert_eq!(config.legacy_runtime_rotary_dim().unwrap(), 32);
        assert_eq!(config.max_position_embeddings, 40960);
        assert!(!config.use_sliding_window);
        assert_eq!(config.sliding_window, None);
        config.validate_existing_executor().unwrap();
        config
            .validate_runtime_layer_shape(39, 5120, 40, 8, 128, 128, 17408)
            .unwrap();
    }

    #[test]
    fn gemma4_config_assembles_but_reports_unimplemented_executor() {
        let config = parse_model_config_value(&gemma4_config()).unwrap();
        let ModelConfig::Gemma4Text(config) = &config else {
            panic!("expected Gemma4 text config");
        };
        assert_eq!(config.decoder.hidden_size, 1536);
        assert_eq!(config.global_head_dim, 512);
        assert_eq!(config.layer_types[0], DecoderLayerKind::SlidingAttention);
        assert_eq!(config.layer_types[4], DecoderLayerKind::FullAttention);
        assert_eq!(config.full_rope.partial_rotary_factor, Some(0.25));
        assert!(matches!(
            ModelConfig::Gemma4Text(config.clone()).execution_status(),
            ModelExecutionStatus::Unimplemented {
                required_executor: "Gemma4TextExecutor",
                ..
            }
        ));
    }

    #[test]
    fn qwen35_dense_config_assembles_existing_aq4_text_contract() {
        let config = parse_model_config_value(&qwen35_config(false)).unwrap();
        let ModelConfig::Qwen35DenseText(config) = config else {
            panic!("expected Qwen3.5 dense config");
        };
        assert_eq!(config.hybrid.decoder.hidden_size, 4096);
        assert_eq!(config.hybrid.layer_types.len(), 32);
        assert_eq!(
            config.hybrid.layer_types[3],
            DecoderLayerKind::FullAttention
        );
        assert_eq!(config.dense_mlp.intermediate_size, 12288);
        config.validate_existing_aq4_executor().unwrap();
        let package_layers = config
            .hybrid
            .layer_types
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<_>>();
        config
            .validate_package_layers(248320, 4096, &package_layers)
            .unwrap();
        let mut wrong_layers = package_layers;
        wrong_layers[3].1 = DecoderLayerKind::LinearAttention;
        let error = config
            .validate_package_layers(248320, 4096, &wrong_layers)
            .unwrap_err();
        assert!(error.contains("layer 3 kind"), "{error}");
    }

    #[test]
    fn qwen35_moe_config_assembles_but_reports_explicit_missing_executor() {
        let config = parse_model_config_value(&qwen35_config(true)).unwrap();
        let ModelConfig::Qwen35MoeText(config) = &config else {
            panic!("expected Qwen3.5 MoE config");
        };
        assert_eq!(config.moe.num_experts, 256);
        assert_eq!(config.moe.num_experts_per_tok, 8);
        assert_eq!(config.moe.expert_intermediate_size, 512);
        assert!(matches!(
            ModelConfig::Qwen35MoeText(config.clone()).execution_status(),
            ModelExecutionStatus::Unimplemented {
                required_executor: "Qwen35MoeExecutor",
                ..
            }
        ));
    }

    #[test]
    fn unknown_architecture_is_rejected_without_qwen3_fallback() {
        let mut value = qwen3_config();
        value["architectures"] = json!(["UnknownForCausalLM"]);
        let error = parse_model_config_value(&value).unwrap_err();
        assert!(error.contains("unsupported architectures value"), "{error}");
    }

    #[test]
    fn qwen3_rope_scaling_is_rejected_until_an_executor_exists() {
        let mut value = qwen3_config();
        value["rope_scaling"] = json!({"rope_type": "linear", "factor": 2.0});
        let error = parse_model_config_value(&value).unwrap_err();
        assert!(error.contains("rope_scaling must be null"), "{error}");
    }

    #[test]
    fn package_config_requires_source_model_dir() {
        let root = std::env::temp_dir().join(format!(
            "ullm-model-config-no-source-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("manifest.json"),
            r#"{"tensors":[],"passthrough_tensors":[]}"#,
        )
        .unwrap();
        let error = load_model_config_from_package(&root).unwrap_err();
        assert!(error.contains("no source_model_dir"), "{error}");
        fs::remove_dir_all(root).unwrap();
    }
}
