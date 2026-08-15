//! Reviewed, offline source contract for `google/gemma-4-12B`.
//!
//! Gemma 4 is the first model whose upstream snapshot is a single direct
//! safetensors file and whose text architecture cannot be represented by the
//! Qwen-specific `model-lock-v1` architecture fields. This additive
//! `model-lock-v2` parser does not weaken or change any v1 identity.

use crate::model::{
    LockedFile, ModelError, TensorDType, TensorDescriptor, VerifiedCache, fingerprint_for_json,
    parse_model_source_json,
};
use serde::Deserialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const GEMMA4_12B_REPO_ID: &str = "google/gemma-4-12B";
pub const GEMMA4_12B_REVISION: &str = "023679ed352de9bb66cc873c9009ce3482585c08";
pub const GEMMA4_12B_ALIAS: &str = "gemma4-12b-bf16";
pub const GEMMA4_12B_FINGERPRINT: &str =
    "sha256:086ede4017206d33533c70d5d00cb492f2f1064a21c995477207ebada668ccff";
pub const GEMMA4_12B_TENSOR_COUNT: u64 = 677;
pub const GEMMA4_12B_TEXT_TENSOR_COUNT: u64 = 666;
pub const GEMMA4_12B_HEADER_LENGTH_BYTES: u64 = 88_952;
pub const GEMMA4_12B_HEADER_SHA256: &str =
    "e432b3ee11ff7f7d179ccbf3827af9669c03a0a28e603000d89c6e1b6c9d4bb7";
pub const GEMMA4_12B_CATALOG_SHA256: &str =
    "24e705586f0bba5e1018951a9ee09aa02b1bfccd73f5c0a82e31e29fb7c2931f";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Gemma4LayerType {
    #[serde(rename = "sliding_attention")]
    SlidingAttention,
    #[serde(rename = "full_attention")]
    FullAttention,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gemma4RopeContract {
    pub rope_type: String,
    pub rope_theta: u64,
    pub partial_rotary_factor: String,
    pub head_dim: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gemma4TextConfigContract {
    pub model_type: String,
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub num_hidden_layers: u64,
    pub num_attention_heads: u64,
    pub num_key_value_heads: u64,
    pub num_global_key_value_heads: u64,
    pub head_dim: u64,
    pub global_head_dim: u64,
    pub hidden_activation: String,
    pub max_position_embeddings: u64,
    pub sliding_window: u64,
    pub rms_norm_eps: String,
    pub attention_bias: bool,
    pub attention_dropout: String,
    pub attention_k_eq_v: bool,
    pub final_logit_softcapping: String,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub vocab_size: u64,
    pub layer_types: Vec<Gemma4LayerType>,
    pub rope: BTreeMap<String, Gemma4RopeContract>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gemma4ComponentContract {
    pub present: bool,
    pub tensor_prefixes: Vec<String>,
    pub tensor_count: u64,
    pub phase_scope: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gemma4ArchitectureContract {
    pub architectures: Vec<String>,
    pub top_level_architecture: String,
    pub model_type: String,
    pub phase_scope: String,
    pub custom_code: bool,
    pub converted: bool,
    pub moe: bool,
    pub text: Gemma4TextConfigContract,
    pub vision: Gemma4ComponentContract,
    pub audio: Gemma4ComponentContract,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gemma4TensorContract {
    pub container: String,
    pub source_path: String,
    pub header_length_field_bytes: u64,
    pub header_length_bytes: u64,
    pub data_buffer_start: u64,
    pub header_sha256: String,
    pub catalog_sha256: String,
    pub tensor_count: u64,
    pub text_tensor_count: u64,
    pub dtype: TensorDType,
    pub unknown_policy: String,
    pub duplicate_policy: String,
    pub catalog_policy: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gemma4SliceContract {
    pub tensor_name: String,
    pub source_file: String,
    pub dtype: TensorDType,
    pub shape: Vec<u64>,
    pub data_offsets: [u64; 2],
    pub absolute_byte_range: [u64; 2],
    pub byte_size: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gemma4TokenizerContract {
    pub files: Vec<String>,
    pub tokenizer_class: String,
    pub vocab_size: u64,
    pub chat_template_path: Option<String>,
    pub prompt_mode: String,
    pub special_token_ids: BTreeMap<String, u64>,
    pub stop_token_ids: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gemma4LicenseContract {
    pub id: String,
    pub statement: String,
    pub evidence_paths: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gemma4ExcludedFile {
    pub path: String,
    pub git_blob: String,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gemma4LockedModel {
    pub repo_id: String,
    pub repo_type: String,
    pub requested_revision: String,
    pub resolved_revision: String,
    pub license: Gemma4LicenseContract,
    pub evidence_files: Vec<String>,
    pub files: Vec<LockedFile>,
    pub excluded_files: Vec<Gemma4ExcludedFile>,
    pub architecture: Gemma4ArchitectureContract,
    pub tensor_contract: Gemma4TensorContract,
    pub slice_contract: Gemma4SliceContract,
    pub tokenizer_contract: Gemma4TokenizerContract,
    pub generation_config_path: String,
    pub derivation: Option<()>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Gemma4ModelLock {
    pub schema_version: String,
    pub model: Gemma4LockedModel,
    pub fingerprint: String,
    pub aliases: Vec<String>,
    pub generated_at: String,
}

impl Gemma4ModelLock {
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn supports_chat_messages(&self) -> bool {
        self.model.tokenizer_contract.chat_template_path.is_some()
    }

    /// Verifies the pinned direct-safetensors cache without downloading or
    /// executing any model-provided code.
    pub fn verify_cache(&self, cache_root: impl AsRef<Path>) -> Result<VerifiedCache, ModelError> {
        crate::model::verify_gemma4_model_cache(self, cache_root)
    }
}

/// Parse the v2 lock after applying the existing duplicate-key,
/// restricted-JCS, and control-character checks.
pub fn parse_gemma4_model_lock(bytes: &[u8]) -> Result<Gemma4ModelLock, ModelError> {
    let computed = fingerprint_for_json(bytes)?;
    let lock: Gemma4ModelLock =
        serde_json::from_slice(bytes).map_err(|error| ModelError::Schema(error.to_string()))?;
    if lock.fingerprint != computed {
        return Err(ModelError::FingerprintMismatch {
            expected: lock.fingerprint,
            actual: computed,
        });
    }
    validate_reviewed_lock(&lock)?;
    Ok(lock)
}

fn validate_reviewed_lock(lock: &Gemma4ModelLock) -> Result<(), ModelError> {
    let invalid = |message: &str| ModelError::Invalid(message.to_owned());
    if lock.schema_version != "model-lock-v2"
        || lock.model.repo_id != GEMMA4_12B_REPO_ID
        || lock.model.repo_type != "model"
        || lock.model.requested_revision != "main"
        || lock.model.resolved_revision != GEMMA4_12B_REVISION
        || lock.fingerprint != GEMMA4_12B_FINGERPRINT
        || lock.aliases != [GEMMA4_12B_ALIAS.to_owned()]
    {
        return Err(invalid("Gemma 4 lock immutable identity differs"));
    }
    if lock.model.license.id != "Apache-2.0"
        || lock.model.evidence_files != ["README.md".to_owned()]
        || lock.model.derivation.is_some()
    {
        return Err(invalid("Gemma 4 source or license evidence differs"));
    }
    let architecture = &lock.model.architecture;
    let text = &architecture.text;
    if architecture.architectures != ["Gemma4UnifiedForConditionalGeneration".to_owned()]
        || architecture.top_level_architecture != "Gemma4UnifiedForConditionalGeneration"
        || architecture.model_type != "gemma4_unified"
        || architecture.phase_scope != "text-only"
        || architecture.custom_code
        || architecture.converted
        || architecture.moe
        || text.model_type != "gemma4_unified_text"
        || text.hidden_size != 3_840
        || text.intermediate_size != 15_360
        || text.num_hidden_layers != 48
        || text.num_attention_heads != 16
        || text.num_key_value_heads != 8
        || text.num_global_key_value_heads != 1
        || text.head_dim != 256
        || text.global_head_dim != 512
        || text.hidden_activation != "gelu_pytorch_tanh"
        || text.max_position_embeddings != 262_144
        || text.sliding_window != 1_024
        || text.rms_norm_eps != "1e-6"
        || text.attention_bias
        || text.attention_dropout != "0"
        || !text.attention_k_eq_v
        || text.final_logit_softcapping != "30"
        || !text.tie_word_embeddings
        || !text.use_cache
        || text.vocab_size != 262_144
        || text.layer_types != reviewed_layer_schedule()
    {
        return Err(invalid("Gemma 4 reviewed architecture differs"));
    }
    let sliding = text
        .rope
        .get("sliding_attention")
        .ok_or_else(|| invalid("Gemma 4 sliding RoPE contract is absent"))?;
    let full = text
        .rope
        .get("full_attention")
        .ok_or_else(|| invalid("Gemma 4 full RoPE contract is absent"))?;
    if text.rope.len() != 2
        || sliding.rope_type != "default"
        || sliding.rope_theta != 10_000
        || sliding.partial_rotary_factor != "1"
        || sliding.head_dim != 256
        || full.rope_type != "proportional"
        || full.rope_theta != 1_000_000
        || full.partial_rotary_factor != "0.25"
        || full.head_dim != 512
    {
        return Err(invalid("Gemma 4 reviewed dual-RoPE contract differs"));
    }
    let tensor = &lock.model.tensor_contract;
    if tensor.container != "direct-safetensors"
        || tensor.source_path != "model.safetensors"
        || tensor.header_length_field_bytes != 8
        || tensor.header_length_bytes != GEMMA4_12B_HEADER_LENGTH_BYTES
        || tensor.data_buffer_start != GEMMA4_12B_HEADER_LENGTH_BYTES + 8
        || tensor.header_sha256 != GEMMA4_12B_HEADER_SHA256
        || tensor.catalog_sha256 != GEMMA4_12B_CATALOG_SHA256
        || tensor.tensor_count != GEMMA4_12B_TENSOR_COUNT
        || tensor.text_tensor_count != GEMMA4_12B_TEXT_TENSOR_COUNT
        || tensor.dtype != TensorDType::Bf16
        || tensor.unknown_policy != "reject"
        || tensor.duplicate_policy != "reject"
        || tensor.catalog_policy != "exact-derived-name-shape-dtype-and-range"
    {
        return Err(invalid("Gemma 4 direct safetensors contract differs"));
    }
    let tokenizer = &lock.model.tokenizer_contract;
    let expected_special_token_ids = BTreeMap::from([
        ("audio".to_owned(), 258_881),
        ("audio_begin".to_owned(), 256_000),
        ("audio_end".to_owned(), 258_883),
        ("bos".to_owned(), 2),
        ("eos".to_owned(), 1),
        ("image".to_owned(), 258_880),
        ("image_begin".to_owned(), 255_999),
        ("image_end".to_owned(), 258_882),
        ("mask".to_owned(), 4),
        ("pad".to_owned(), 0),
        ("think".to_owned(), 98),
        ("unk".to_owned(), 3),
        ("video".to_owned(), 258_884),
    ]);
    if tokenizer.files != ["tokenizer.json", "tokenizer_config.json"]
        || tokenizer.tokenizer_class != "GemmaTokenizer"
        || tokenizer.vocab_size != 262_144
        || tokenizer.chat_template_path.is_some()
        || tokenizer.prompt_mode != "raw-text-only"
        || tokenizer.special_token_ids != expected_special_token_ids
        || tokenizer.stop_token_ids != [1]
    {
        return Err(invalid("Gemma 4 base tokenizer contract differs"));
    }
    let file_paths: BTreeSet<_> = lock
        .model
        .files
        .iter()
        .map(|file| file.path.as_str())
        .collect();
    let expected_paths = BTreeSet::from([
        "README.md",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]);
    if file_paths != expected_paths || lock.model.files.len() != expected_paths.len() {
        return Err(invalid("Gemma 4 locked file set differs"));
    }
    Ok(())
}

pub fn reviewed_layer_schedule() -> Vec<Gemma4LayerType> {
    (0..48)
        .map(|layer| {
            if (layer + 1) % 6 == 0 {
                Gemma4LayerType::FullAttention
            } else {
                Gemma4LayerType::SlidingAttention
            }
        })
        .collect()
}

/// Derive the complete upstream tensor catalog from the reviewed text,
/// vision, and audio shapes. The official file stores payloads in bytewise
/// tensor-name order, so ranges are derived only after names are sorted.
pub(crate) fn expected_gemma4_tensor_catalog()
-> Result<BTreeMap<String, TensorDescriptor>, ModelError> {
    let mut shapes = BTreeMap::new();
    insert_shape(
        &mut shapes,
        "model.embed_audio.embedding_projection.weight",
        &[3_840, 640],
    )?;
    insert_shape(
        &mut shapes,
        "model.embed_vision.embedding_projection.weight",
        &[3_840, 3_840],
    )?;
    insert_shape(
        &mut shapes,
        "model.language_model.embed_tokens.weight",
        &[262_144, 3_840],
    )?;

    for (layer, layer_type) in reviewed_layer_schedule().into_iter().enumerate() {
        let prefix = format!("model.language_model.layers.{layer}");
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "post_feedforward_layernorm.weight",
            "pre_feedforward_layernorm.weight",
        ] {
            insert_shape(&mut shapes, format!("{prefix}.{suffix}"), &[3_840])?;
        }
        insert_shape(&mut shapes, format!("{prefix}.layer_scalar"), &[1])?;
        insert_shape(
            &mut shapes,
            format!("{prefix}.mlp.down_proj.weight"),
            &[3_840, 15_360],
        )?;
        insert_shape(
            &mut shapes,
            format!("{prefix}.mlp.gate_proj.weight"),
            &[15_360, 3_840],
        )?;
        insert_shape(
            &mut shapes,
            format!("{prefix}.mlp.up_proj.weight"),
            &[15_360, 3_840],
        )?;
        let (head_dim, kv_heads) = match layer_type {
            Gemma4LayerType::SlidingAttention => (256, 8),
            Gemma4LayerType::FullAttention => (512, 1),
        };
        insert_shape(
            &mut shapes,
            format!("{prefix}.self_attn.k_norm.weight"),
            &[head_dim],
        )?;
        insert_shape(
            &mut shapes,
            format!("{prefix}.self_attn.k_proj.weight"),
            &[kv_heads * head_dim, 3_840],
        )?;
        insert_shape(
            &mut shapes,
            format!("{prefix}.self_attn.o_proj.weight"),
            &[3_840, 16 * head_dim],
        )?;
        insert_shape(
            &mut shapes,
            format!("{prefix}.self_attn.q_norm.weight"),
            &[head_dim],
        )?;
        insert_shape(
            &mut shapes,
            format!("{prefix}.self_attn.q_proj.weight"),
            &[16 * head_dim, 3_840],
        )?;
        if layer_type == Gemma4LayerType::SlidingAttention {
            insert_shape(
                &mut shapes,
                format!("{prefix}.self_attn.v_proj.weight"),
                &[kv_heads * head_dim, 3_840],
            )?;
        }
    }
    insert_shape(&mut shapes, "model.language_model.norm.weight", &[3_840])?;
    for (name, shape) in [
        ("model.vision_embedder.patch_dense.bias", vec![3_840]),
        (
            "model.vision_embedder.patch_dense.weight",
            vec![3_840, 6_912],
        ),
        ("model.vision_embedder.patch_ln1.bias", vec![6_912]),
        ("model.vision_embedder.patch_ln1.weight", vec![6_912]),
        ("model.vision_embedder.patch_ln2.bias", vec![3_840]),
        ("model.vision_embedder.patch_ln2.weight", vec![3_840]),
        ("model.vision_embedder.pos_embedding", vec![1_120, 2, 3_840]),
        ("model.vision_embedder.pos_norm.bias", vec![3_840]),
        ("model.vision_embedder.pos_norm.weight", vec![3_840]),
    ] {
        insert_shape(&mut shapes, name, &shape)?;
    }

    if shapes.len() as u64 != GEMMA4_12B_TENSOR_COUNT {
        return Err(invalid("derived Gemma 4 tensor count differs"));
    }
    let text_count = shapes
        .keys()
        .filter(|name| name.starts_with("model.language_model."))
        .count() as u64;
    if text_count != GEMMA4_12B_TEXT_TENSOR_COUNT {
        return Err(invalid("derived Gemma 4 text tensor count differs"));
    }

    let mut cursor = 0u64;
    let mut catalog = BTreeMap::new();
    for (name, shape) in shapes {
        let elements = shape
            .iter()
            .try_fold(1u64, |product, dimension| product.checked_mul(*dimension));
        let byte_size = elements
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| invalid(format!("derived Gemma 4 tensor size overflows: {name}")))?;
        let end = cursor
            .checked_add(byte_size)
            .ok_or_else(|| invalid("derived Gemma 4 payload size overflows"))?;
        let absolute_start = (GEMMA4_12B_HEADER_LENGTH_BYTES + 8)
            .checked_add(cursor)
            .ok_or_else(|| invalid("derived Gemma 4 absolute start overflows"))?;
        let absolute_end = (GEMMA4_12B_HEADER_LENGTH_BYTES + 8)
            .checked_add(end)
            .ok_or_else(|| invalid("derived Gemma 4 absolute end overflows"))?;
        catalog.insert(
            name.clone(),
            TensorDescriptor {
                tensor_name: name,
                source_file: "model.safetensors".to_owned(),
                dtype: TensorDType::Bf16,
                shape,
                header_length_field_bytes: 8,
                header_length_bytes: GEMMA4_12B_HEADER_LENGTH_BYTES,
                data_buffer_start: GEMMA4_12B_HEADER_LENGTH_BYTES + 8,
                data_offset_basis: "data-buffer-relative".to_owned(),
                data_offsets: [cursor, end],
                absolute_byte_range: [absolute_start, absolute_end],
                byte_size,
            },
        );
        cursor = end;
    }
    if cursor + GEMMA4_12B_HEADER_LENGTH_BYTES + 8 != 23_919_549_408 {
        return Err(invalid("derived Gemma 4 file size differs"));
    }
    Ok(catalog)
}

pub(crate) fn gemma4_catalog_sha256(catalog: &BTreeMap<String, TensorDescriptor>) -> String {
    let mut hasher = Sha256::new();
    for descriptor in catalog.values() {
        hasher.update(descriptor.tensor_name.as_bytes());
        hasher.update(b"\t");
        let dtype = match descriptor.dtype {
            TensorDType::Bf16 => "BF16",
            TensorDType::F16 => "F16",
            TensorDType::F32 => "F32",
            TensorDType::I32 => "I32",
            TensorDType::I64 => "I64",
            TensorDType::U8 => "U8",
        };
        hasher.update(dtype.as_bytes());
        hasher.update(b"\t");
        for (index, dimension) in descriptor.shape.iter().enumerate() {
            if index != 0 {
                hasher.update(b"x");
            }
            hasher.update(dimension.to_string().as_bytes());
        }
        hasher.update(b"\t");
        hasher.update(descriptor.data_offsets[0].to_string().as_bytes());
        hasher.update(b"-");
        hasher.update(descriptor.data_offsets[1].to_string().as_bytes());
        hasher.update(b"\n");
    }
    format!("{:x}", hasher.finalize())
}

fn insert_shape(
    shapes: &mut BTreeMap<String, Vec<u64>>,
    name: impl Into<String>,
    shape: &[u64],
) -> Result<(), ModelError> {
    let name = name.into();
    if shape.is_empty()
        || shape.contains(&0)
        || shapes.insert(name.clone(), shape.to_vec()).is_some()
    {
        return Err(invalid(format!(
            "invalid or duplicate derived Gemma 4 tensor: {name}"
        )));
    }
    Ok(())
}

/// Independently validate the exact upstream `config.json` architecture used
/// by the reviewed lock. Unknown fields are rejected instead of being silently
/// accepted as a compatible future Gemma variant.
pub fn validate_gemma4_config(bytes: &[u8]) -> Result<(), ModelError> {
    let value = parse_model_source_json(bytes, "Gemma 4 config")?;
    let root = object(&value, "Gemma 4 config root")?;
    require_keys(
        root,
        &[
            "architectures",
            "audio_config",
            "audio_token_id",
            "boa_token_id",
            "boi_token_id",
            "dtype",
            "eoa_token_index",
            "eoi_token_id",
            "image_token_id",
            "initializer_range",
            "model_type",
            "text_config",
            "tie_word_embeddings",
            "transformers_version",
            "video_token_id",
            "vision_config",
        ],
        "Gemma 4 config root",
    )?;
    if string(root, "model_type")? != "gemma4_unified"
        || string(root, "dtype")? != "bfloat16"
        || !bool_value(root, "tie_word_embeddings")?
        || string_array(root, "architectures")?
            != ["Gemma4UnifiedForConditionalGeneration".to_owned()]
        || u64_value(root, "audio_token_id")? != 258_881
        || u64_value(root, "boa_token_id")? != 256_000
        || u64_value(root, "boi_token_id")? != 255_999
        || u64_value(root, "eoa_token_index")? != 258_883
        || u64_value(root, "eoi_token_id")? != 258_882
        || u64_value(root, "image_token_id")? != 258_880
        || u64_value(root, "video_token_id")? != 258_884
        || f64_value(root, "initializer_range")?.to_bits() != 0.02_f64.to_bits()
    {
        return Err(invalid("Gemma 4 top-level config differs"));
    }
    validate_text_config(object_field(root, "text_config")?)?;
    validate_known_component(
        object_field(root, "audio_config")?,
        "gemma4_unified_audio",
        "hidden_size",
        640,
    )?;
    validate_known_component(
        object_field(root, "vision_config")?,
        "gemma4_unified_vision",
        "mm_embed_dim",
        3_840,
    )?;
    Ok(())
}

fn validate_text_config(text: &Map<String, Value>) -> Result<(), ModelError> {
    require_keys(
        text,
        &[
            "attention_bias",
            "attention_dropout",
            "attention_k_eq_v",
            "bos_token_id",
            "enable_moe_block",
            "eos_token_id",
            "final_logit_softcapping",
            "global_head_dim",
            "head_dim",
            "hidden_activation",
            "hidden_size",
            "hidden_size_per_layer_input",
            "initializer_range",
            "intermediate_size",
            "layer_types",
            "max_position_embeddings",
            "model_type",
            "moe_intermediate_size",
            "num_attention_heads",
            "num_experts",
            "num_global_key_value_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "num_kv_shared_layers",
            "pad_token_id",
            "rms_norm_eps",
            "rope_parameters",
            "sliding_window",
            "tie_word_embeddings",
            "top_k_experts",
            "use_bidirectional_attention",
            "use_cache",
            "use_double_wide_mlp",
            "vocab_size",
            "vocab_size_per_layer_input",
        ],
        "Gemma 4 text config",
    )?;
    let expected_schedule: Vec<String> = reviewed_layer_schedule()
        .iter()
        .map(|kind| match kind {
            Gemma4LayerType::SlidingAttention => "sliding_attention".to_owned(),
            Gemma4LayerType::FullAttention => "full_attention".to_owned(),
        })
        .collect();
    if string(text, "model_type")? != "gemma4_unified_text"
        || u64_value(text, "hidden_size")? != 3_840
        || u64_value(text, "intermediate_size")? != 15_360
        || u64_value(text, "num_hidden_layers")? != 48
        || u64_value(text, "num_attention_heads")? != 16
        || u64_value(text, "num_key_value_heads")? != 8
        || u64_value(text, "num_global_key_value_heads")? != 1
        || u64_value(text, "head_dim")? != 256
        || u64_value(text, "global_head_dim")? != 512
        || string(text, "hidden_activation")? != "gelu_pytorch_tanh"
        || u64_value(text, "max_position_embeddings")? != 262_144
        || u64_value(text, "sliding_window")? != 1_024
        || f64_value(text, "rms_norm_eps")?.to_bits() != 1.0e-6_f64.to_bits()
        || bool_value(text, "attention_bias")?
        || f64_value(text, "attention_dropout")?.to_bits() != 0.0_f64.to_bits()
        || !bool_value(text, "attention_k_eq_v")?
        || f64_value(text, "final_logit_softcapping")?.to_bits() != 30.0_f64.to_bits()
        || !bool_value(text, "tie_word_embeddings")?
        || !bool_value(text, "use_cache")?
        || bool_value(text, "enable_moe_block")?
        || u64_value(text, "vocab_size")? != 262_144
        || string_array(text, "layer_types")? != expected_schedule
    {
        return Err(invalid("Gemma 4 text config differs"));
    }
    let rope = object_field(text, "rope_parameters")?;
    require_keys(
        rope,
        &["full_attention", "sliding_attention"],
        "Gemma 4 rope_parameters",
    )?;
    let sliding = object_field(rope, "sliding_attention")?;
    require_keys(
        sliding,
        &["rope_theta", "rope_type"],
        "Gemma 4 sliding RoPE",
    )?;
    let full = object_field(rope, "full_attention")?;
    require_keys(
        full,
        &["partial_rotary_factor", "rope_theta", "rope_type"],
        "Gemma 4 full RoPE",
    )?;
    if string(sliding, "rope_type")? != "default"
        || f64_value(sliding, "rope_theta")?.to_bits() != 10_000.0_f64.to_bits()
        || string(full, "rope_type")? != "proportional"
        || f64_value(full, "rope_theta")?.to_bits() != 1_000_000.0_f64.to_bits()
        || f64_value(full, "partial_rotary_factor")?.to_bits() != 0.25_f64.to_bits()
    {
        return Err(invalid("Gemma 4 dual-RoPE config differs"));
    }
    Ok(())
}

fn validate_known_component(
    component: &Map<String, Value>,
    expected_model_type: &str,
    hidden_field: &str,
    expected_hidden_size: u64,
) -> Result<(), ModelError> {
    if string(component, "model_type")? != expected_model_type
        || u64_value(component, hidden_field)? != expected_hidden_size
        || f64_value(component, "rms_norm_eps")?.to_bits() != 1.0e-6_f64.to_bits()
    {
        return Err(invalid("Gemma 4 known-unconsumed component differs"));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::Invalid(message.into())
}

fn object<'a>(value: &'a Value, scope: &str) -> Result<&'a Map<String, Value>, ModelError> {
    value
        .as_object()
        .ok_or_else(|| invalid(format!("{scope} must be an object")))
}

fn object_field<'a>(
    object: &'a Map<String, Value>,
    field: &str,
) -> Result<&'a Map<String, Value>, ModelError> {
    object
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| invalid(format!("{field} must be an object")))
}

fn require_keys(
    object: &Map<String, Value>,
    expected: &[&str],
    scope: &str,
) -> Result<(), ModelError> {
    let actual: BTreeSet<_> = object.keys().map(String::as_str).collect();
    let expected: BTreeSet<_> = expected.iter().copied().collect();
    if actual != expected {
        return Err(invalid(format!("{scope} field set differs")));
    }
    Ok(())
}

fn string<'a>(object: &'a Map<String, Value>, field: &str) -> Result<&'a str, ModelError> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("{field} must be a string")))
}

fn string_array(object: &Map<String, Value>, field: &str) -> Result<Vec<String>, ModelError> {
    object
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(format!("{field} must be an array")))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid(format!("{field} entries must be strings")))
        })
        .collect()
}

fn u64_value(object: &Map<String, Value>, field: &str) -> Result<u64, ModelError> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid(format!("{field} must be a non-negative integer")))
}

fn f64_value(object: &Map<String, Value>, field: &str) -> Result<f64, ModelError> {
    object
        .get(field)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| invalid(format!("{field} must be finite numeric data")))
}

fn bool_value(object: &Map<String, Value>, field: &str) -> Result<bool, ModelError> {
    object
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| invalid(format!("{field} must be boolean")))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCK_BYTES: &[u8] = include_bytes!("../../../docs/models/locks/gemma4-12b-bf16.json");
    const CONFIG_BYTES: &[u8] =
        include_bytes!("../../../ci/fixtures/model-lock-v2/gemma4-config.json");

    fn mutated_lock(mut mutate: impl FnMut(&mut serde_json::Value)) -> Vec<u8> {
        let mut value: serde_json::Value = serde_json::from_slice(LOCK_BYTES).unwrap();
        mutate(&mut value);
        let provisional = serde_json::to_vec(&value).unwrap();
        let fingerprint = fingerprint_for_json(&provisional).unwrap();
        value["fingerprint"] = serde_json::Value::String(fingerprint);
        serde_json::to_vec(&value).unwrap()
    }

    #[test]
    fn tracked_lock_has_exact_reviewed_identity_and_no_chat_template() {
        let lock = parse_gemma4_model_lock(LOCK_BYTES).expect("tracked Gemma lock is valid");
        assert_eq!(lock.fingerprint(), GEMMA4_12B_FINGERPRINT);
        assert!(!lock.supports_chat_messages());
        assert_eq!(lock.model.architecture.text.layer_types.len(), 48);
        assert_eq!(lock.model.tensor_contract.tensor_count, 677);
    }

    #[test]
    fn reviewed_lock_rejects_identity_architecture_and_template_substitution() {
        let mutations = [
            mutated_lock(|value| value["model"]["resolved_revision"] = "1".repeat(40).into()),
            mutated_lock(|value| value["model"]["architecture"]["text"]["head_dim"] = 257.into()),
            mutated_lock(|value| {
                value["model"]["tokenizer_contract"]["chat_template_path"] =
                    "chat_template.jinja".into();
                value["model"]["tokenizer_contract"]["prompt_mode"] = "chat".into();
            }),
        ];
        for bytes in mutations {
            assert!(parse_gemma4_model_lock(&bytes).is_err());
        }
    }

    #[test]
    fn duplicate_lock_key_is_rejected_before_typed_parse() {
        let duplicate = String::from_utf8(LOCK_BYTES.to_vec()).unwrap().replacen(
            "\"schema_version\": \"model-lock-v2\"",
            "\"schema_version\": \"model-lock-v2\", \"schema_version\": \"model-lock-v2\"",
            1,
        );
        assert!(parse_gemma4_model_lock(duplicate.as_bytes()).is_err());
    }

    #[test]
    fn reviewed_config_is_closed_and_rejects_shape_boundaries() {
        validate_gemma4_config(CONFIG_BYTES).expect("reviewed config is valid");
        for head_dim in [0_u64, 1, 255, 257] {
            let mut value: Value = serde_json::from_slice(CONFIG_BYTES).unwrap();
            value["text_config"]["head_dim"] = head_dim.into();
            assert!(validate_gemma4_config(&serde_json::to_vec(&value).unwrap()).is_err());
        }
        let mut unknown: Value = serde_json::from_slice(CONFIG_BYTES).unwrap();
        unknown["text_config"]["future_attention"] = true.into();
        assert!(validate_gemma4_config(&serde_json::to_vec(&unknown).unwrap()).is_err());
        let duplicate = String::from_utf8(CONFIG_BYTES.to_vec()).unwrap().replacen(
            "\"hidden_size\": 3840",
            "\"hidden_size\": 3840, \"hidden_size\": 3840",
            1,
        );
        assert!(validate_gemma4_config(duplicate.as_bytes()).is_err());
    }

    #[test]
    fn reviewed_schedule_covers_boundaries() {
        let schedule = reviewed_layer_schedule();
        assert_eq!(schedule.len(), 48);
        assert_eq!(schedule[0], Gemma4LayerType::SlidingAttention);
        assert_eq!(schedule[4], Gemma4LayerType::SlidingAttention);
        assert_eq!(schedule[5], Gemma4LayerType::FullAttention);
        assert_eq!(schedule[46], Gemma4LayerType::SlidingAttention);
        assert_eq!(schedule[47], Gemma4LayerType::FullAttention);
        assert_eq!(
            schedule
                .iter()
                .filter(|kind| **kind == Gemma4LayerType::FullAttention)
                .count(),
            8
        );
    }

    #[test]
    fn derived_catalog_matches_reviewed_counts_ranges_and_digest() {
        let catalog = expected_gemma4_tensor_catalog().expect("catalog derives exactly");
        assert_eq!(catalog.len(), 677);
        assert_eq!(gemma4_catalog_sha256(&catalog), GEMMA4_12B_CATALOG_SHA256);
        assert_eq!(
            catalog["model.embed_audio.embedding_projection.weight"].data_offsets,
            [0, 4_915_200]
        );
        assert_eq!(
            catalog["model.language_model.layers.5.self_attn.k_proj.weight"].shape,
            [512, 3_840]
        );
        assert!(!catalog.contains_key("model.language_model.layers.5.self_attn.v_proj.weight"));
        assert_eq!(
            catalog["model.language_model.layers.6.self_attn.v_proj.weight"].shape,
            [2_048, 3_840]
        );
        let slice = &catalog["model.language_model.norm.weight"];
        assert_eq!(slice.data_offsets, [23_849_099_360, 23_849_107_040]);
        assert_eq!(slice.absolute_byte_range, [23_849_188_320, 23_849_196_000]);
        let last = &catalog["model.vision_embedder.pos_norm.weight"];
        assert_eq!(last.absolute_end(), 23_919_549_408);
    }

    #[test]
    fn reviewed_registry_selects_qwen_and_gemma_by_alias_and_fingerprint() {
        let qwen = crate::model::parse_reviewed_model_lock(include_bytes!(
            "../../../docs/models/locks/qwen3.5-4b-bf16.json"
        ))
        .expect("reviewed Qwen lock parses");
        let gemma = crate::model::parse_reviewed_model_lock(LOCK_BYTES)
            .expect("reviewed Gemma lock parses");
        let registry = crate::model::ReviewedModelRegistry::new(vec![qwen, gemma])
            .expect("distinct reviewed aliases register");
        let selected = registry
            .resolve(GEMMA4_12B_ALIAS, GEMMA4_12B_FINGERPRINT)
            .expect("Gemma alias and fingerprint select");
        assert_eq!(
            selected.kind(),
            crate::model::ReviewedModelKind::Gemma4Dense
        );
        assert!(!selected.supports_chat_messages());
        assert!(
            registry
                .resolve(GEMMA4_12B_ALIAS, GEMMA4_12B_HEADER_SHA256)
                .is_err()
        );

        let mut duplicate = selected.clone();
        if let crate::model::ReviewedModelLock::Gemma4(lock) = &mut duplicate {
            lock.aliases = vec!["qwen3.5-4b-bf16".to_owned()];
        }
        assert!(
            crate::model::ReviewedModelRegistry::new(vec![registry.locks()[0].clone(), duplicate])
                .is_err()
        );
    }
}
