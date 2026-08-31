//! Strict GGUF source adapter for the reviewed Gemma 4 MTP assistant.
//!
//! The assistant GGUF is an output artifact, not a standalone target model.
//! Its tensors retain the logical ranges of the reviewed assistant
//! safetensors source so the existing MTP weight-plan and resident loader can
//! consume one source contract.  Reads translate those source-relative tensor
//! ranges to the corresponding GGUF tensor; no source path is reopened and no
//! tensor bytes are interpreted on the host.

use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fmt;

use crate::gemma4::Gemma4ModelLock;
use crate::gemma4_mtp::{
    GEMMA4_MTP_CATALOG_SHA256, GEMMA4_MTP_HEADER_SHA256, GEMMA4_MTP_MODEL_SHA256,
    GEMMA4_MTP_REPO_ID, GEMMA4_MTP_REVISION, Gemma4MtpConfig, Gemma4MtpModelLock,
    Gemma4MtpWeightSource, expected_gemma4_mtp_tensor_catalog, validate_gemma4_mtp_target,
};
use crate::gguf::{
    GEMMA4_MTP_ASSISTANT_FINGERPRINT_KEY, GEMMA4_MTP_KV_MAPPING_KEY, GEMMA4_MTP_LAYER_MAPPING_KEY,
    GEMMA4_MTP_ROLE_KEY, GEMMA4_MTP_SEMANTIC_PAIR_KEY, GEMMA4_MTP_SOURCE_RANGES_KEY,
    GEMMA4_MTP_TARGET_FINGERPRINT_KEY, GEMMA4_MTP_TOKENIZER_IDENTITY_KEY, GgufArray, GgufValue,
    VerifiedGguf,
};
use crate::model::{ModelError, TensorDescriptor};
use crate::weights::build_gemma4_mtp_weight_load_plan;

const ARCHITECTURE: &str = "gemma4mtp";
const RECIPE_SCHEMA: &str = "sllm-gguf-tensor-recipe-v1";
const SOURCE_MODEL_KEY: &str = "gemma4mtp.source_model_sha256";
const SOURCE_HEADER_KEY: &str = "gemma4mtp.source_header_sha256";
const CATALOG_KEY: &str = "gemma4mtp.tensor_catalog_sha256";
const CONTEXT_KEY: &str = "gemma4mtp.context_length";
const HIDDEN_KEY: &str = "gemma4mtp.embedding_length";
const BACKBONE_HIDDEN_KEY: &str = "gemma4mtp.backbone_embedding_length";
const BLOCK_COUNT_KEY: &str = "gemma4mtp.block_count";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct SourceRangeV1 {
    name: String,
    source_file: String,
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
    absolute_byte_range: [u64; 2],
}

/// A verified Gemma 4 MTP assistant backed by an assistant-only GGUF.
///
/// `tensors` deliberately contains the canonical safetensors descriptors,
/// rather than descriptors fabricated from GGUF offsets.  The MTP plan is
/// defined in that source coordinate system; [`read_tensor_range`] performs
/// the checked coordinate translation into the same-named GGUF tensor.
pub struct VerifiedGgufGemma4Mtp {
    lock_fingerprint: String,
    target_fingerprint: String,
    config: Gemma4MtpConfig,
    tensors: BTreeMap<String, TensorDescriptor>,
    gguf: VerifiedGguf,
}

impl fmt::Debug for VerifiedGgufGemma4Mtp {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedGgufGemma4Mtp")
            .field("lock_fingerprint", &self.lock_fingerprint)
            .field("target_fingerprint", &self.target_fingerprint)
            .field("tensor_count", &self.tensors.len())
            .field("gguf", &self.gguf.path())
            .finish_non_exhaustive()
    }
}

impl Clone for VerifiedGgufGemma4Mtp {
    fn clone(&self) -> Self {
        Self {
            lock_fingerprint: self.lock_fingerprint.clone(),
            target_fingerprint: self.target_fingerprint.clone(),
            config: self.config.clone(),
            tensors: self.tensors.clone(),
            gguf: self.gguf.clone(),
        }
    }
}

impl VerifiedGgufGemma4Mtp {
    /// Verify an assistant GGUF against the immutable assistant lock and the
    /// exact reviewed target lock before exposing it as an MTP weight source.
    pub fn verify(
        gguf: VerifiedGguf,
        lock: &Gemma4MtpModelLock,
        target: &Gemma4ModelLock,
    ) -> Result<Self, ModelError> {
        validate_gemma4_mtp_target(lock, target)?;
        verify_pair_metadata(&gguf, lock, target)?;
        let catalog = expected_gemma4_mtp_tensor_catalog()?;
        verify_gguf_tensors(&gguf, &catalog)?;
        verify_source_ranges(&gguf, &catalog)?;
        // Reuse the canonical source-plan validator.  This checks the
        // assistant consumer set, BF16 dtype, lock file geometry, and exact
        // 48-tensor/payload accounting before a resident can be provisioned.
        build_gemma4_mtp_weight_load_plan(lock, catalog.values())
            .map_err(|error| ModelError::Invalid(error.to_string()))?;
        let config = config_from_lock(lock)?;
        Ok(Self {
            lock_fingerprint: lock.fingerprint().to_owned(),
            target_fingerprint: target.fingerprint().to_owned(),
            config,
            tensors: catalog,
            gguf,
        })
    }

    /// Alias retained for callers that use the source-verifier naming style.
    pub fn from_verified_gguf(
        gguf: VerifiedGguf,
        lock: &Gemma4MtpModelLock,
        target: &Gemma4ModelLock,
    ) -> Result<Self, ModelError> {
        Self::verify(gguf, lock, target)
    }

    pub fn lock_fingerprint(&self) -> &str {
        &self.lock_fingerprint
    }

    pub fn target_fingerprint(&self) -> &str {
        &self.target_fingerprint
    }

    pub fn config(&self) -> &Gemma4MtpConfig {
        &self.config
    }

    pub fn tensors(&self) -> &BTreeMap<String, TensorDescriptor> {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.get(name)
    }

    pub fn gguf(&self) -> &VerifiedGguf {
        &self.gguf
    }

    /// Read a source-relative assistant tensor range from the GGUF payload.
    /// The descriptor remains in safetensors coordinates, while the GGUF
    /// tensor with the same canonical name supplies the bytes.
    pub fn read_tensor_range(
        &self,
        name: &str,
        tensor_offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, ModelError> {
        let descriptor = self
            .tensors
            .get(name)
            .ok_or_else(|| invalid(format!("unknown Gemma 4 MTP tensor: {name}")))?;
        let length_u64 =
            u64::try_from(length).map_err(|_| invalid("MTP GGUF read length does not fit u64"))?;
        let end = tensor_offset
            .checked_add(length_u64)
            .ok_or_else(|| invalid("MTP GGUF read range overflowed"))?;
        if end > descriptor.byte_size {
            return Err(invalid("MTP GGUF read exceeds the verified tensor range"));
        }
        let tensor = self
            .gguf
            .tensor(name)
            .ok_or_else(|| invalid(format!("verified MTP GGUF tensor is absent: {name}")))?;
        if tensor.byte_length() != descriptor.byte_size {
            return Err(invalid(format!(
                "MTP GGUF tensor byte size changed after verification: {name}"
            )));
        }
        self.gguf
            .read_tensor_range(name, tensor_offset, length)
            .map_err(|error| invalid(error.to_string()))
    }
}

impl Gemma4MtpWeightSource for VerifiedGgufGemma4Mtp {
    fn lock_fingerprint(&self) -> &str {
        self.lock_fingerprint()
    }

    fn target_fingerprint(&self) -> &str {
        self.target_fingerprint()
    }

    fn config(&self) -> &Gemma4MtpConfig {
        self.config()
    }

    fn tensors(&self) -> &BTreeMap<String, TensorDescriptor> {
        self.tensors()
    }

    fn read_tensor_range(
        &self,
        name: &str,
        tensor_offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, ModelError> {
        self.read_tensor_range(name, tensor_offset, length)
    }
}

/// Verify a parsed assistant GGUF with the reviewed target/assistant pair.
pub fn verify_gguf_gemma4_mtp(
    gguf: VerifiedGguf,
    lock: &Gemma4MtpModelLock,
    target: &Gemma4ModelLock,
) -> Result<VerifiedGgufGemma4Mtp, ModelError> {
    VerifiedGgufGemma4Mtp::verify(gguf, lock, target)
}

fn verify_pair_metadata(
    gguf: &VerifiedGguf,
    lock: &Gemma4MtpModelLock,
    target: &Gemma4ModelLock,
) -> Result<(), ModelError> {
    if gguf.architecture() != ARCHITECTURE || !gguf.is_assistant_only() {
        return Err(invalid(
            "Gemma 4 MTP GGUF is not an assistant-only artifact",
        ));
    }
    let extension = gguf
        .extension()
        .ok_or_else(|| invalid("Gemma 4 MTP GGUF extension is absent"))?;
    let expected_pair = format!(
        "gemma4mtp-pair:{}:{}",
        target.fingerprint(),
        lock.fingerprint()
    );
    if metadata_string(gguf, "general.type")? != "model"
        || metadata_string(gguf, "general.name")? != GEMMA4_MTP_REPO_ID
        || metadata_string(gguf, "general.source.url")? != GEMMA4_MTP_REVISION
        || metadata_string(gguf, "general.license")? != "Apache-2.0"
    {
        return Err(invalid("Gemma 4 MTP GGUF general metadata differs"));
    }
    if extension.recipe.schema_version != RECIPE_SCHEMA
        || extension.recipe.semantic_model_id != expected_pair
        || extension.recipe.source_lock_fingerprints
            != [
                target.fingerprint().to_owned(),
                lock.fingerprint().to_owned(),
            ]
        || !extension.recipe.bindings.is_empty()
        || !extension.recipe.logical_shapes.is_empty()
        || !extension.recipe.static_fp8_kv.is_empty()
        || !extension.recipe.known_unconsumed_tensors.is_empty()
    {
        return Err(invalid("Gemma 4 MTP GGUF recipe identity differs"));
    }
    if metadata_string(gguf, GEMMA4_MTP_ROLE_KEY)? != "assistant"
        || metadata_string(gguf, GEMMA4_MTP_SEMANTIC_PAIR_KEY)? != expected_pair
        || metadata_string(gguf, GEMMA4_MTP_TARGET_FINGERPRINT_KEY)? != target.fingerprint()
        || metadata_string(gguf, GEMMA4_MTP_ASSISTANT_FINGERPRINT_KEY)? != lock.fingerprint()
    {
        return Err(invalid("Gemma 4 MTP GGUF pair metadata differs"));
    }
    if metadata_string(gguf, SOURCE_MODEL_KEY)? != format!("sha256:{GEMMA4_MTP_MODEL_SHA256}")
        || metadata_string(gguf, SOURCE_HEADER_KEY)? != format!("sha256:{GEMMA4_MTP_HEADER_SHA256}")
        || metadata_string(gguf, CATALOG_KEY)? != format!("sha256:{GEMMA4_MTP_CATALOG_SHA256}")
    {
        return Err(invalid("Gemma 4 MTP GGUF source digest metadata differs"));
    }
    if metadata_u32_array(gguf, GEMMA4_MTP_LAYER_MAPPING_KEY)? != [0, 1, 2, 3]
        || metadata_u32_array(gguf, GEMMA4_MTP_KV_MAPPING_KEY)? != [46, 46, 46, 47]
    {
        return Err(invalid("Gemma 4 MTP GGUF layer/KV mapping differs"));
    }
    let architecture = &lock.model.architecture;
    if metadata_u32(gguf, CONTEXT_KEY)?
        != u32_value(architecture.max_position_embeddings, CONTEXT_KEY)?
        || metadata_u32(gguf, HIDDEN_KEY)? != u32_value(architecture.hidden_size, HIDDEN_KEY)?
        || metadata_u32(gguf, BACKBONE_HIDDEN_KEY)?
            != u32_value(architecture.backbone_hidden_size, BACKBONE_HIDDEN_KEY)?
        || metadata_u32(gguf, BLOCK_COUNT_KEY)?
            != u32_value(architecture.num_hidden_layers, BLOCK_COUNT_KEY)?
    {
        return Err(invalid("Gemma 4 MTP GGUF architecture metadata differs"));
    }
    verify_layer_types(gguf, architecture.layer_types.as_slice())?;
    verify_tokenizer_identity(gguf, lock, target)?;
    Ok(())
}

fn verify_layer_types(
    gguf: &VerifiedGguf,
    expected: &[crate::Gemma4LayerType],
) -> Result<(), ModelError> {
    let observed = match gguf.metadata_value("gemma4mtp.layer_types") {
        Some(GgufValue::Array(GgufArray::String(values))) => values,
        _ => return Err(invalid("Gemma 4 MTP GGUF layer types metadata is absent")),
    };
    let expected = expected
        .iter()
        .map(|layer| match layer {
            crate::Gemma4LayerType::SlidingAttention => "sliding_attention",
            crate::Gemma4LayerType::FullAttention => "full_attention",
        })
        .collect::<Vec<_>>();
    if observed.len() != expected.len()
        || observed
            .iter()
            .zip(expected)
            .any(|(observed, expected)| observed != expected)
    {
        return Err(invalid("Gemma 4 MTP GGUF layer types differ"));
    }
    Ok(())
}

fn verify_tokenizer_identity(
    gguf: &VerifiedGguf,
    lock: &Gemma4MtpModelLock,
    target: &Gemma4ModelLock,
) -> Result<(), ModelError> {
    let value = metadata_string(gguf, GEMMA4_MTP_TOKENIZER_IDENTITY_KEY)?;
    let parsed: Value = serde_json::from_str(value)
        .map_err(|error| invalid(format!("MTP tokenizer identity JSON is invalid: {error}")))?;
    if serde_json::to_string(&parsed)
        .map_err(|error| invalid(format!("serialize MTP tokenizer identity: {error}")))?
        != value
    {
        return Err(invalid("MTP tokenizer identity JSON is not canonical"));
    }
    let assistant = &lock.model.tokenizer_contract;
    let target_tokenizer = &target.model.tokenizer_contract;
    let expected = json!({
        "wire_source": assistant.wire_source,
        "assistant_tokenizer_sha256": locked_file_sha256(lock, "tokenizer.json")?,
        "assistant_tokenizer_config_sha256": locked_file_sha256(lock, "tokenizer_config.json")?,
        "target_tokenizer_sha256": locked_target_file_sha256(target, "tokenizer.json")?,
        "assistant_vocab_semantic_sha256": assistant.vocab_semantic_sha256,
        "assistant_merges_semantic_sha256": assistant.merges_semantic_sha256,
        "assistant_vocab_size": assistant.vocab_size,
        "target_vocab_size": target.model.architecture.text.vocab_size,
        "target_tokenizer_class": target_tokenizer.tokenizer_class,
        "common_generation_token_ids": assistant.common_generation_token_ids,
        "assistant_named_video_token_present": assistant.assistant_named_video_token_present,
        "target_named_video_token_id": assistant.target_named_video_token_id,
    });
    let expected = serde_json::to_string(&expected).map_err(|error| {
        invalid(format!(
            "serialize expected MTP tokenizer identity: {error}"
        ))
    })?;
    if value != expected {
        return Err(invalid("Gemma 4 MTP GGUF tokenizer identity differs"));
    }
    Ok(())
}

fn verify_gguf_tensors(
    gguf: &VerifiedGguf,
    expected: &BTreeMap<String, TensorDescriptor>,
) -> Result<(), ModelError> {
    if gguf.tensors().len() != expected.len()
        || gguf
            .tensors()
            .iter()
            .any(|tensor| !expected.contains_key(&tensor.name))
    {
        return Err(invalid("Gemma 4 MTP GGUF tensor name set differs"));
    }
    for tensor in gguf.tensors() {
        let descriptor = expected
            .get(&tensor.name)
            .ok_or_else(|| invalid("Gemma 4 MTP GGUF tensor is not in the catalog"))?;
        let mut shape = tensor.dimensions.clone();
        shape.reverse();
        if tensor.tensor_type != crate::GgufTensorType::Bf16
            || shape != descriptor.shape
            || tensor.byte_length() != descriptor.byte_size
        {
            return Err(invalid(format!(
                "Gemma 4 MTP GGUF tensor shape/dtype/size differs: {}",
                tensor.name
            )));
        }
    }
    Ok(())
}

fn verify_source_ranges(
    gguf: &VerifiedGguf,
    expected: &BTreeMap<String, TensorDescriptor>,
) -> Result<(), ModelError> {
    let value = metadata_string(gguf, GEMMA4_MTP_SOURCE_RANGES_KEY)?;
    let ranges: Vec<SourceRangeV1> = serde_json::from_str(value)
        .map_err(|error| invalid(format!("MTP source ranges JSON is invalid: {error}")))?;
    if ranges.len() != expected.len() {
        return Err(invalid("Gemma 4 MTP GGUF source range count differs"));
    }
    let mut observed = BTreeMap::new();
    for range in ranges {
        if observed.insert(range.name.clone(), range).is_some() {
            return Err(invalid("Gemma 4 MTP GGUF source range names duplicate"));
        }
    }
    for (name, descriptor) in expected {
        let range = observed
            .get(name)
            .ok_or_else(|| invalid(format!("Gemma 4 MTP source range is absent: {name}")))?;
        if range.source_file != descriptor.source_file
            || range.dtype != "BF16"
            || range.shape != descriptor.shape
            || range.data_offsets != descriptor.data_offsets
            || range.absolute_byte_range != descriptor.absolute_byte_range
        {
            return Err(invalid(format!("Gemma 4 MTP source range differs: {name}")));
        }
    }
    Ok(())
}

fn config_from_lock(lock: &Gemma4MtpModelLock) -> Result<Gemma4MtpConfig, ModelError> {
    let architecture = &lock.model.architecture;
    Ok(Gemma4MtpConfig {
        hidden_size: u32_value(architecture.hidden_size, "hidden_size")?,
        backbone_hidden_size: u32_value(architecture.backbone_hidden_size, "backbone_hidden_size")?,
        intermediate_size: u32_value(architecture.intermediate_size, "intermediate_size")?,
        layer_count: u32_value(architecture.num_hidden_layers, "layer_count")?,
        attention_heads: u32_value(architecture.num_attention_heads, "attention_heads")?,
        kv_heads: u32_value(architecture.num_key_value_heads, "kv_heads")?,
        global_kv_heads: u32_value(architecture.num_global_key_value_heads, "global_kv_heads")?,
        head_dim: u32_value(architecture.head_dim, "head_dim")?,
        global_head_dim: u32_value(architecture.global_head_dim, "global_head_dim")?,
        sliding_window: u32_value(architecture.sliding_window, "sliding_window")?,
        max_position_embeddings: u32_value(
            architecture.max_position_embeddings,
            "max_position_embeddings",
        )?,
        vocab_size: u32_value(architecture.vocab_size, "vocab_size")?,
        layer_types: architecture.layer_types.clone(),
        draft_to_target_kv_layers: lock.model.target_compatibility.draft_to_target_kv_layers,
    })
}

fn metadata_string<'a>(gguf: &'a VerifiedGguf, key: &str) -> Result<&'a str, ModelError> {
    match gguf.metadata_value(key) {
        Some(GgufValue::String(value)) if !value.is_empty() => Ok(value),
        _ => Err(invalid(format!(
            "Gemma 4 MTP GGUF metadata is missing: {key}"
        ))),
    }
}

fn metadata_u32(gguf: &VerifiedGguf, key: &str) -> Result<u32, ModelError> {
    match gguf.metadata_value(key) {
        Some(GgufValue::U32(value)) => Ok(*value),
        _ => Err(invalid(format!(
            "Gemma 4 MTP GGUF metadata is not U32: {key}"
        ))),
    }
}

fn metadata_u32_array(gguf: &VerifiedGguf, key: &str) -> Result<Vec<u32>, ModelError> {
    match gguf.metadata_value(key) {
        Some(GgufValue::Array(GgufArray::U32(values))) => Ok(values.clone()),
        _ => Err(invalid(format!(
            "Gemma 4 MTP GGUF metadata is not U32 array: {key}"
        ))),
    }
}

fn u32_value(value: u64, label: &str) -> Result<u32, ModelError> {
    u32::try_from(value).map_err(|_| invalid(format!("Gemma 4 MTP {label} exceeds U32")))
}

fn locked_file_sha256(lock: &Gemma4MtpModelLock, path: &str) -> Result<String, ModelError> {
    lock.model
        .files
        .iter()
        .find(|file| file.path == path)
        .map(|file| file.sha256.clone())
        .ok_or_else(|| invalid(format!("Gemma 4 MTP lock file is absent: {path}")))
}

fn locked_target_file_sha256(target: &Gemma4ModelLock, path: &str) -> Result<String, ModelError> {
    target
        .model
        .files
        .iter()
        .find(|file| file.path == path)
        .map(|file| file.sha256.clone())
        .ok_or_else(|| invalid(format!("Gemma 4 target lock file is absent: {path}")))
}

fn invalid(message: impl Into<String>) -> ModelError {
    ModelError::Invalid(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemma4::parse_gemma4_model_lock;
    use crate::gemma4_mtp::parse_gemma4_mtp_model_lock;
    use std::collections::BTreeSet;

    const LOCK_BYTES: &[u8] =
        include_bytes!("../../../docs/models/locks/gemma4-12b-it-assistant-bf16.json");
    const TARGET_BYTES: &[u8] =
        include_bytes!("../../../docs/models/locks/gemma4-12b-it-bf16.json");

    #[test]
    fn canonical_source_catalog_is_the_48_tensor_bf16_contract() {
        let catalog = expected_gemma4_mtp_tensor_catalog().expect("catalog derives");
        assert_eq!(catalog.len(), 48);
        assert_eq!(
            crate::gemma4_mtp_catalog_sha256(&catalog),
            GEMMA4_MTP_CATALOG_SHA256
        );
        assert!(
            catalog
                .keys()
                .all(|name| { !name.contains("k_proj") && !name.contains("v_proj") })
        );
    }

    #[test]
    fn reviewed_pair_identity_and_config_are_closed() {
        let lock = parse_gemma4_mtp_model_lock(LOCK_BYTES).expect("assistant lock");
        let target = parse_gemma4_model_lock(TARGET_BYTES).expect("target lock");
        validate_gemma4_mtp_target(&lock, &target).expect("reviewed pair");
        let config = config_from_lock(&lock).expect("config");
        assert_eq!(config.hidden_size, 1_024);
        assert_eq!(config.backbone_hidden_size, 3_840);
        assert_eq!(config.draft_to_target_kv_layers, [46, 46, 46, 47]);
        assert_eq!(
            format!(
                "gemma4mtp-pair:{}:{}",
                target.fingerprint(),
                lock.fingerprint()
            ),
            format!(
                "gemma4mtp-pair:{}:{}",
                target.fingerprint(),
                crate::GEMMA4_MTP_FINGERPRINT
            )
        );
    }

    #[test]
    fn source_range_mapping_rejects_unknown_or_missing_names() {
        let catalog = expected_gemma4_mtp_tensor_catalog().expect("catalog derives");
        let mut names = catalog.keys().cloned().collect::<BTreeSet<_>>();
        assert!(names.remove("model.norm.weight"));
        assert!(!names.contains("model.layers.0.self_attn.k_proj.weight"));
        assert_eq!(names.len(), 47);
    }

    #[test]
    #[ignore = "requires the reviewed external source cache and SLLM_GEMMA4_MTP_GGUF"]
    fn external_lossless_gguf_matches_every_source_tensor_byte() {
        let gguf_path = std::env::var("SLLM_GEMMA4_MTP_GGUF")
            .expect("SLLM_GEMMA4_MTP_GGUF must name the canonical assistant GGUF");
        let lock = parse_gemma4_mtp_model_lock(LOCK_BYTES).expect("assistant lock");
        let target = parse_gemma4_model_lock(TARGET_BYTES).expect("target lock");
        let source = lock
            .verify_cache(
                "/home/homelab1/.cache/sllm/models/google--gemma-4-12B-it-assistant",
                &target,
            )
            .expect("source assistant verifies");
        let gguf = VerifiedGguf::open(gguf_path).expect("assistant GGUF parses");
        let derived =
            verify_gguf_gemma4_mtp(gguf, &lock, &target).expect("assistant GGUF pair verifies");
        assert_eq!(derived.tensors(), source.tensors());

        const CHUNK: usize = 4 * 1024 * 1024;
        for descriptor in source.tensors().values() {
            let mut offset = 0_u64;
            while offset < descriptor.byte_size {
                let length = usize::try_from((descriptor.byte_size - offset).min(CHUNK as u64))
                    .expect("bounded chunk length");
                assert_eq!(
                    source
                        .read_tensor_range(&descriptor.tensor_name, offset, length)
                        .expect("source tensor chunk"),
                    derived
                        .read_tensor_range(&descriptor.tensor_name, offset, length)
                        .expect("GGUF tensor chunk"),
                    "tensor bytes differ at {}+{}",
                    descriptor.tensor_name,
                    offset
                );
                offset += length as u64;
            }
        }
    }
}
