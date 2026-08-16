//! Exact Qwen3.5-35B-A3B MXFP4 source and tensor-inventory contract.

use crate::{
    BudgetBoundary, DType, GenerationStopPolicyV1, LayerType, MaxNewTokensZero, PromptEvaluation,
    QuantizedTensorEncoding, SemanticOpDescriptor, SparseMoeContract, StopEvaluation,
    StopTokenHandling, TensorDType, TensorView, WEIGHT_LOAD_CHUNK_BYTES, WeightClassification,
    WeightConsumer, WeightConsumerKey, WeightLoadChunk, WeightLoadEntry, WeightLoadPlan,
};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const QWEN35_MOE_REPOSITORY: &str = "amd/Qwen3.5-35B-A3B-MXFP4";
pub const QWEN35_MOE_REVISION: &str = "2e19c6576db91e5d5a93455415619262218bf8a1";
pub const QWEN35_MOE_SEMANTIC_REPOSITORY: &str = "Qwen/Qwen3.5-35B-A3B-FP8";
pub const QWEN35_MOE_SEMANTIC_REVISION: &str = "9d1823d2dee688a6b25e77009dc727688c44936e";
pub const QWEN35_MOE_TENSOR_COUNT: usize = 63_171;
pub const QWEN35_MOE_TEXT_TENSOR_COUNT: usize = 62_053;
pub const QWEN35_MOE_TEXT_RESIDENT_BYTES: u64 = 22_009_481_856;
pub const QWEN35_MOE_VISION_TENSOR_COUNT: usize = 333;
pub const QWEN35_MOE_MTP_TENSOR_COUNT: usize = 785;
pub const QWEN35_MOE_EXPERT_PROJECTION_COUNT: usize = 40 * 256 * 3;
pub const QWEN35_MOE_LAYER_BLOB_BYTES: u64 = 434_114_560;
pub const QWEN35_MOE_LAYER_BLOB_PREFIX: &str = "__sllm_qwen35_moe_layer_blob.";
pub const QWEN35_MOE_MODEL_FINGERPRINT: &str =
    "sha256:5bca203f6ec8ab9cab4e340a6c337fff7387f9ca2fa12526c48ce999748e83b0";
pub const QWEN35_MOE_LICENSE: &str = "Apache-2.0";

const CONFIG_SHA256: &str = "9c5002446e05374776c3059711156075bdb42e1a634f91252f747d19e08515ec";
const INDEX_SHA256: &str = "b4288428d322fa713ce530496b4112409f93afcb42d456e21ac512a4338dc3c3";
const CATALOG_SHA256: &str = "31b6b186dca978fbb6b6165dc59d1a35dbd355477ab8f51e6fea79bec7a2c9e4";
const TEXT_CATALOG_SHA256: &str =
    "5bca203f6ec8ab9cab4e340a6c337fff7387f9ca2fa12526c48ce999748e83b0";
const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
const MAX_INDEX_BYTES: u64 = 16 * 1024 * 1024;
const MAX_HEADER_BYTES: u64 = 4 * 1024 * 1024;

const SUPPORT_FILES: [(&str, u64, &str); 9] = [
    (
        "LICENSE",
        11_343,
        "50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32",
    ),
    (
        "README.md",
        2_918,
        "c2a69910366e1bf368dc5e33ffe19b1a16990d790aec92d0a149330ba8c0c542",
    ),
    (
        "chat_template.jinja",
        7_756,
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    ),
    (
        "configuration.json",
        51,
        "2d4464e2ead06bc9bc718c781309ad1e7baded626d66e8dcdc8b469ba185faf0",
    ),
    (
        "generation_config.json",
        244,
        "4f25002776b741773666203dcea8f54619f177ace3ae483d311102092a4658e0",
    ),
    (
        "merges.txt",
        3_353_259,
        "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    ),
    (
        "tokenizer.json",
        12_807_982,
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    ),
    (
        "tokenizer_config.json",
        16_710,
        "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
    ),
    (
        "vocab.json",
        6_722_759,
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    ),
];

const SHARDS: [(&str, u64, &str); 14] = [
    (
        "model.safetensors-00001-of-00014.safetensors",
        1_426_769_992,
        "1460d43a2cc934c53a0fcc8d69809e1f46d9dcabc32eee53308f5f33f7ff9bc6",
    ),
    (
        "model.safetensors-00002-of-00014.safetensors",
        1_426_767_944,
        "9dc94405c9cb77bab3e2da5b33cae73eb99e0b278d6f05046f66358165e912ea",
    ),
    (
        "model.safetensors-00003-of-00014.safetensors",
        1_426_767_944,
        "916d384b99d59e0e6f3bbf3ffe2f0764637c356a3a7549c88b5fd52e654ad887",
    ),
    (
        "model.safetensors-00004-of-00014.safetensors",
        1_426_768_968,
        "9bafcfafd6db09dd1621e7c8a1a4d46d68f42bfd4a32ad9b7342a472c4724344",
    ),
    (
        "model.safetensors-00005-of-00014.safetensors",
        1_426_769_992,
        "05c8d0182f2aa7934f6c0e35f90dfe0304eb10faaf619eda98aa4b2313d034f1",
    ),
    (
        "model.safetensors-00006-of-00014.safetensors",
        1_426_768_968,
        "0b5d8b05aa63015c4d3af3070274d91771fbe3189304f00b59f172d48bbfc57b",
    ),
    (
        "model.safetensors-00007-of-00014.safetensors",
        1_426_767_944,
        "71c1874152c89b966ca62af849ce83a0a065f2e363fee80df53b20f25d7de913",
    ),
    (
        "model.safetensors-00008-of-00014.safetensors",
        1_426_767_944,
        "47d110f99018c2574a8238b6988094b13460b6533f74ad3670dfb9c959b2fde3",
    ),
    (
        "model.safetensors-00009-of-00014.safetensors",
        2_890_308_528,
        "858ba45dbc09418ac15dda4d9317c6c7a8a6511e07422584d9f6771b0759d394",
    ),
    (
        "model.safetensors-00010-of-00014.safetensors",
        1_426_775_624,
        "42fc04935bdce821f8a82ed313cc8cae05017297f10494319098837dbad06cc8",
    ),
    (
        "model.safetensors-00011-of-00014.safetensors",
        1_426_777_672,
        "d3eb7e622d61e65069c08537b9139566ace88f1cdedaeca8dc7969aa8356dd01",
    ),
    (
        "model.safetensors-00012-of-00014.safetensors",
        1_426_776_136,
        "3d6ccfdce1456d435a15c189b4d339e3c1319b72eaef363fce6ecebd6ca57c76",
    ),
    (
        "model.safetensors-00013-of-00014.safetensors",
        3_791_068_528,
        "2b61fcad30ecf23ed9cdd4521f5fdbbac529dbbca19da3e187245d79ce780a71",
    ),
    (
        "model.safetensors-00014-of-00014.safetensors",
        2_224_764_664,
        "943cfeb9f925fe9dac48508dbec4518637b90c243f914277a86db982de9de3ad",
    ),
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35MoeConfig {
    pub hidden_size: u32,
    pub layer_count: u32,
    pub attention_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub expert_count: u32,
    pub selected_expert_count: u32,
    pub expert_intermediate_size: u32,
    pub shared_expert_intermediate_size: u32,
    pub layer_types: Vec<LayerType>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35MoeRecipe {
    pub encoding: QuantizedTensorEncoding,
    pub group_size: u32,
    pub weight_scale_format: &'static str,
    pub activation_dynamic: bool,
    pub activation_scale_format: &'static str,
    pub quark_version: String,
    pub exclusion_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Qwen35MoeExpertProjection {
    Gate,
    Up,
    Down,
}

impl Qwen35MoeExpertProjection {
    const fn source_stem(self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }

    const fn logical_shape(self) -> [u64; 2] {
        match self {
            Self::Gate | Self::Up => [512, 2_048],
            Self::Down => [2_048, 512],
        }
    }

    const fn value_shape(self) -> [u64; 2] {
        match self {
            Self::Gate | Self::Up => [512, 1_024],
            Self::Down => [2_048, 256],
        }
    }

    const fn scale_shape(self) -> [u64; 2] {
        match self {
            Self::Gate | Self::Up => [512, 64],
            Self::Down => [2_048, 16],
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35MoeTensorPlane {
    pub source_file: String,
    pub source_name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub absolute_byte_range: [u64; 2],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35MoeExpertTensor {
    pub layer: u16,
    pub expert: u16,
    pub projection: Qwen35MoeExpertProjection,
    pub logical_shape: [u64; 2],
    pub encoding: QuantizedTensorEncoding,
    pub value: Qwen35MoeTensorPlane,
    pub scale: Qwen35MoeTensorPlane,
}

#[derive(Debug)]
pub struct VerifiedQwen35Moe {
    root: PathBuf,
    config: Qwen35MoeConfig,
    recipe: Qwen35MoeRecipe,
    text_planes: Vec<Qwen35MoeTensorPlane>,
    experts: BTreeMap<(u16, u16, Qwen35MoeExpertProjection), Qwen35MoeExpertTensor>,
    support_files: BTreeMap<String, Arc<[u8]>>,
    shards: BTreeMap<String, BoundShard>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ShardIdentity {
    device: u64,
    inode: u64,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

impl ShardIdentity {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            size: metadata.len(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }
}

#[derive(Debug)]
struct BoundShard {
    file: Arc<File>,
    identity: ShardIdentity,
}

impl BoundShard {
    fn read_range(&self, offset: u64, length: usize) -> Result<Vec<u8>, Qwen35MoeModelError> {
        let before = self
            .file
            .metadata()
            .map_err(|error| invalid(error.to_string()))?;
        if ShardIdentity::from_metadata(&before) != self.identity {
            return Err(invalid("verified MoE shard identity changed before read"));
        }
        let length_u64 = u64::try_from(length).map_err(|_| invalid("shard range is too large"))?;
        if offset
            .checked_add(length_u64)
            .is_none_or(|end| end > self.identity.size)
        {
            return Err(invalid("MoE shard range exceeds the verified file"));
        }
        let mut bytes = vec![0_u8; length];
        let mut read = 0_usize;
        while read < bytes.len() {
            let position = offset
                .checked_add(u64::try_from(read).map_err(|_| invalid("shard offset is too large"))?)
                .ok_or_else(|| invalid("shard read offset overflow"))?;
            let count = self
                .file
                .read_at(&mut bytes[read..], position)
                .map_err(|error| invalid(error.to_string()))?;
            if count == 0 {
                return Err(invalid("verified MoE shard returned a short read"));
            }
            read += count;
        }
        let after = self
            .file
            .metadata()
            .map_err(|error| invalid(error.to_string()))?;
        if ShardIdentity::from_metadata(&after) != self.identity {
            return Err(invalid("verified MoE shard identity changed during read"));
        }
        Ok(bytes)
    }
}

impl VerifiedQwen35Moe {
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn config(&self) -> &Qwen35MoeConfig {
        &self.config
    }
    pub fn recipe(&self) -> &Qwen35MoeRecipe {
        &self.recipe
    }
    /// Every text/output source tensor in deterministic lexical-name order.
    /// Vision and MTP planes are intentionally absent.
    pub fn text_planes(&self) -> &[Qwen35MoeTensorPlane] {
        &self.text_planes
    }
    pub fn expert(
        &self,
        layer: u16,
        expert: u16,
        projection: Qwen35MoeExpertProjection,
    ) -> Option<&Qwen35MoeExpertTensor> {
        self.experts.get(&(layer, expert, projection))
    }
    pub fn experts(&self) -> impl ExactSizeIterator<Item = &Qwen35MoeExpertTensor> {
        self.experts.values()
    }

    pub fn plane(&self, name: &str) -> Option<&Qwen35MoeTensorPlane> {
        self.text_planes
            .binary_search_by(|plane| plane.source_name.as_str().cmp(name))
            .ok()
            .and_then(|index| self.text_planes.get(index))
    }

    pub fn locked_shard(&self, name: &str) -> Option<(u64, &'static str)> {
        SHARDS
            .iter()
            .find(|(file, _, _)| *file == name)
            .map(|(_, size, digest)| (*size, *digest))
    }

    pub fn read_support_file(&self, name: &str) -> Result<Vec<u8>, Qwen35MoeModelError> {
        self.support_files
            .get(name)
            .map(|bytes| bytes.to_vec())
            .ok_or_else(|| invalid(format!("unsupported MoE support file: {name}")))
    }

    pub(crate) fn read_plane_range(
        &self,
        plane: &Qwen35MoeTensorPlane,
        relative_offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, Qwen35MoeModelError> {
        let plane_length = plane.absolute_byte_range[1]
            .checked_sub(plane.absolute_byte_range[0])
            .ok_or_else(|| invalid("MoE tensor plane range underflow"))?;
        let length_u64 =
            u64::try_from(length).map_err(|_| invalid("MoE tensor range is too large"))?;
        if relative_offset
            .checked_add(length_u64)
            .is_none_or(|end| end > plane_length)
        {
            return Err(invalid("MoE tensor range exceeds its verified plane"));
        }
        let offset = plane.absolute_byte_range[0]
            .checked_add(relative_offset)
            .ok_or_else(|| invalid("MoE tensor absolute offset overflow"))?;
        self.shards
            .get(&plane.source_file)
            .ok_or_else(|| invalid("MoE tensor plane shard is not bound"))?
            .read_range(offset, length)
    }
}

pub fn qwen35_moe_layer_blob_name(layer: u32) -> String {
    format!("{QWEN35_MOE_LAYER_BLOB_PREFIX}{layer}")
}

pub fn qwen35_moe_generation_stop_policy() -> GenerationStopPolicyV1 {
    GenerationStopPolicyV1 {
        version: 1,
        stop_token_ids: vec![248_046, 248_044],
        evaluation: StopEvaluation::NewlyGeneratedAfterArgmax,
        prompt_evaluation: PromptEvaluation::NeverStop,
        stop_token: StopTokenHandling {
            visible_output: false,
            subsequent_decode_input: false,
        },
        budget_boundary: BudgetBoundary::StopTokenWins,
        max_new_tokens_zero: MaxNewTokensZero::MaxNewTokensBeforeDecode,
        reason_version: 1,
    }
}

pub fn build_qwen35_moe_weight_load_plan(
    artifact: &VerifiedQwen35Moe,
) -> Result<WeightLoadPlan, Qwen35MoeModelError> {
    let mut entries = Vec::new();
    let mut destination = 0_u64;
    for plane in artifact.text_planes() {
        let Some(consumer) = classify_execution_plane(plane, artifact.config())? else {
            continue;
        };
        let dtype = parse_tensor_dtype(&plane.dtype)?;
        let byte_length = plane.absolute_byte_range[1]
            .checked_sub(plane.absolute_byte_range[0])
            .ok_or_else(|| invalid("execution plane byte range underflow"))?;
        let (locked_file_size, locked_file_sha256) = artifact
            .locked_shard(&plane.source_file)
            .ok_or_else(|| invalid("execution plane shard is not locked"))?;
        entries.push(WeightLoadEntry {
            tensor_name: plane.source_name.clone(),
            classification: WeightClassification::Required,
            consumer: Some(consumer),
            dtype,
            shape: plane.shape.clone(),
            source_file: plane.source_file.clone(),
            locked_file_size,
            locked_file_sha256: locked_file_sha256.to_owned(),
            source_range: plane.absolute_byte_range,
            destination_start: Some(destination),
            chunks: load_chunks(plane.absolute_byte_range[0], destination, byte_length)?,
        });
        destination = destination
            .checked_add(byte_length)
            .ok_or_else(|| invalid("execution plan destination overflow"))?;
    }
    for layer in 0..artifact.config().layer_count {
        let name = qwen35_moe_layer_blob_name(layer);
        entries.push(WeightLoadEntry {
            tensor_name: name.clone(),
            classification: WeightClassification::Required,
            consumer: Some(WeightConsumerKey {
                layer: Some(u64::from(layer)),
                role: WeightConsumer::MoeLayerBlob,
            }),
            dtype: TensorDType::U8,
            shape: vec![QWEN35_MOE_LAYER_BLOB_BYTES],
            source_file: format!("sllm://qwen35-moe/layer/{layer}"),
            locked_file_size: QWEN35_MOE_TEXT_RESIDENT_BYTES,
            locked_file_sha256: TEXT_CATALOG_SHA256.to_owned(),
            source_range: [0, QWEN35_MOE_LAYER_BLOB_BYTES],
            destination_start: Some(destination),
            chunks: load_chunks(0, destination, QWEN35_MOE_LAYER_BLOB_BYTES)?,
        });
        destination = destination
            .checked_add(QWEN35_MOE_LAYER_BLOB_BYTES)
            .ok_or_else(|| invalid("MoE blob plan destination overflow"))?;
    }
    entries.sort_by(|left, right| left.tensor_name.cmp(&right.tensor_name));
    crate::weights::WeightLoadPlan::from_verified_entries(
        crate::weights::VerifiedWeightPlanMetadata {
            schema_version: "qwen35-moe-mxfp4-load-plan-v1".to_owned(),
            repo_id: QWEN35_MOE_REPOSITORY.to_owned(),
            resolved_revision: QWEN35_MOE_REVISION.to_owned(),
            lock_fingerprint: QWEN35_MOE_MODEL_FINGERPRINT.to_owned(),
            tied_embeddings: false,
            chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
            total_destination_bytes: destination,
        },
        entries,
    )
    .map_err(|error| invalid(error.to_string()))
}

fn parse_tensor_dtype(dtype: &str) -> Result<TensorDType, Qwen35MoeModelError> {
    match dtype {
        "BF16" => Ok(TensorDType::Bf16),
        "F16" => Ok(TensorDType::F16),
        "F32" => Ok(TensorDType::F32),
        "I32" => Ok(TensorDType::I32),
        "I64" => Ok(TensorDType::I64),
        "U8" => Ok(TensorDType::U8),
        _ => Err(invalid(format!(
            "unsupported execution tensor dtype: {dtype}"
        ))),
    }
}

fn load_chunks(
    source_start: u64,
    destination_start: u64,
    byte_length: u64,
) -> Result<Vec<WeightLoadChunk>, Qwen35MoeModelError> {
    let mut chunks = Vec::new();
    let mut relative = 0_u64;
    while relative < byte_length {
        let length = (byte_length - relative).min(WEIGHT_LOAD_CHUNK_BYTES);
        chunks.push(WeightLoadChunk {
            source_offset: source_start
                .checked_add(relative)
                .ok_or_else(|| invalid("MoE plan source offset overflow"))?,
            destination_offset: destination_start
                .checked_add(relative)
                .ok_or_else(|| invalid("MoE plan destination offset overflow"))?,
            byte_length: length,
        });
        relative += length;
    }
    Ok(chunks)
}

fn classify_execution_plane(
    plane: &Qwen35MoeTensorPlane,
    config: &Qwen35MoeConfig,
) -> Result<Option<WeightConsumerKey>, Qwen35MoeModelError> {
    let top = match plane.source_name.as_str() {
        "model.language_model.embed_tokens.weight" => Some(WeightConsumer::Embedding),
        "model.language_model.norm.weight" => Some(WeightConsumer::FinalNorm),
        "lm_head.weight" => Some(WeightConsumer::OutputProjection),
        _ => None,
    };
    if let Some(role) = top {
        return Ok(Some(WeightConsumerKey { layer: None, role }));
    }
    const PREFIX: &str = "model.language_model.layers.";
    let Some(remainder) = plane.source_name.strip_prefix(PREFIX) else {
        return Err(invalid(format!(
            "unknown text execution tensor: {}",
            plane.source_name
        )));
    };
    let (layer_text, suffix) = remainder
        .split_once('.')
        .ok_or_else(|| invalid("malformed MoE layer tensor name"))?;
    let layer = layer_text
        .parse::<u32>()
        .map_err(|_| invalid("malformed MoE layer index"))?;
    let layer_type = config
        .layer_types
        .get(layer as usize)
        .ok_or_else(|| invalid("MoE layer tensor is out of range"))?;
    if suffix.starts_with("mlp.experts.") || suffix.starts_with("mlp.shared_expert") {
        return Ok(None);
    }
    let role = match suffix {
        "input_layernorm.weight" => WeightConsumer::InputNorm,
        "post_attention_layernorm.weight" => WeightConsumer::PostAttentionNorm,
        "mlp.gate.weight" => WeightConsumer::MoeRouter,
        "linear_attn.in_proj_qkv.weight" if *layer_type == LayerType::LinearAttention => {
            WeightConsumer::GdnInProjQkv
        }
        "linear_attn.in_proj_z.weight" if *layer_type == LayerType::LinearAttention => {
            WeightConsumer::GdnInProjZ
        }
        "linear_attn.in_proj_b.weight" if *layer_type == LayerType::LinearAttention => {
            WeightConsumer::GdnInProjB
        }
        "linear_attn.in_proj_a.weight" if *layer_type == LayerType::LinearAttention => {
            WeightConsumer::GdnInProjA
        }
        "linear_attn.conv1d.weight" if *layer_type == LayerType::LinearAttention => {
            WeightConsumer::GdnConv1d
        }
        "linear_attn.A_log" if *layer_type == LayerType::LinearAttention => WeightConsumer::GdnALog,
        "linear_attn.dt_bias" if *layer_type == LayerType::LinearAttention => {
            WeightConsumer::GdnDtBias
        }
        "linear_attn.norm.weight" if *layer_type == LayerType::LinearAttention => {
            WeightConsumer::GdnNorm
        }
        "linear_attn.out_proj.weight" if *layer_type == LayerType::LinearAttention => {
            WeightConsumer::GdnOutProj
        }
        "self_attn.q_proj.weight" if *layer_type == LayerType::FullAttention => {
            WeightConsumer::AttentionQ
        }
        "self_attn.k_proj.weight" if *layer_type == LayerType::FullAttention => {
            WeightConsumer::AttentionK
        }
        "self_attn.v_proj.weight" if *layer_type == LayerType::FullAttention => {
            WeightConsumer::AttentionV
        }
        "self_attn.o_proj.weight" if *layer_type == LayerType::FullAttention => {
            WeightConsumer::AttentionO
        }
        "self_attn.q_norm.weight" if *layer_type == LayerType::FullAttention => {
            WeightConsumer::AttentionQNorm
        }
        "self_attn.k_norm.weight" if *layer_type == LayerType::FullAttention => {
            WeightConsumer::AttentionKNorm
        }
        _ => {
            return Err(invalid(format!(
                "unknown MoE execution tensor: {}",
                plane.source_name
            )));
        }
    };
    Ok(Some(WeightConsumerKey {
        layer: Some(u64::from(layer)),
        role,
    }))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35MoeLayerGraph {
    pub layer: u32,
    pub layer_type: LayerType,
    pub sparse_moe: SemanticOpDescriptor,
}

/// Container-neutral, host-only graph adapter for the reviewed 40-layer MoE
/// language component. Attention/GDN execution remains in the common Qwen
/// executor; this graph owns the replacement MLP semantic boundary and exact
/// text-only resident inventory identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35MoeGraph {
    token_count: u64,
    state_capacity: u64,
    text_catalog_sha256: &'static str,
    text_tensor_count: usize,
    layers: Vec<Qwen35MoeLayerGraph>,
}

impl Qwen35MoeGraph {
    pub const fn token_count(&self) -> u64 {
        self.token_count
    }
    pub const fn state_capacity(&self) -> u64 {
        self.state_capacity
    }
    pub const fn text_catalog_sha256(&self) -> &'static str {
        self.text_catalog_sha256
    }
    pub const fn text_tensor_count(&self) -> usize {
        self.text_tensor_count
    }
    pub fn layers(&self) -> &[Qwen35MoeLayerGraph] {
        &self.layers
    }
}

pub fn build_qwen35_moe_graph(
    artifact: &VerifiedQwen35Moe,
    token_count: u64,
    state_capacity: u64,
) -> Result<Qwen35MoeGraph, Qwen35MoeModelError> {
    if token_count == 0 || state_capacity == 0 || token_count > state_capacity {
        return Err(invalid("MoE graph token/state capacity contract differs"));
    }
    let config = artifact.config();
    if config.layer_count != 40
        || config.layer_types.len() != 40
        || artifact.text_planes().len() != QWEN35_MOE_TEXT_TENSOR_COUNT
        || artifact.experts().len() != QWEN35_MOE_EXPERT_PROJECTION_COUNT
    {
        return Err(invalid("verified MoE graph inventory differs"));
    }
    let rows = usize::try_from(token_count).map_err(|_| invalid("token count overflows usize"))?;
    let hidden = usize::try_from(config.hidden_size).unwrap();
    let view = || {
        TensorView::contiguous(DType::Bf16, &[rows, hidden])
            .map_err(|error| invalid(format!("MoE graph tensor: {error}")))
    };
    let contract = SparseMoeContract::new(
        config.hidden_size,
        config.expert_count,
        config.selected_expert_count,
        config.expert_intermediate_size,
        config.shared_expert_intermediate_size,
        true,
    )
    .map_err(|error| invalid(format!("MoE graph contract: {error}")))?;
    let mut layers = Vec::with_capacity(40);
    let router = TensorView::contiguous(DType::Bf16, &[256, 2048])
        .map_err(|error| invalid(format!("MoE router tensor: {error}")))?;
    let layer_blob = TensorView::contiguous(DType::U8, &[434_114_560])
        .map_err(|error| invalid(format!("MoE layer blob tensor: {error}")))?;
    for (layer, layer_type) in config.layer_types.iter().copied().enumerate() {
        let sparse_moe = SemanticOpDescriptor::new_sparse_moe(
            vec![view()?, router.clone(), layer_blob.clone()],
            vec![view()?],
            contract,
        )
        .map_err(|error| invalid(format!("MoE layer {layer} semantic: {error}")))?;
        layers.push(Qwen35MoeLayerGraph {
            layer: layer as u32,
            layer_type,
            sparse_moe,
        });
    }
    Ok(Qwen35MoeGraph {
        token_count,
        state_capacity,
        text_catalog_sha256: TEXT_CATALOG_SHA256,
        text_tensor_count: artifact.text_planes().len(),
        layers,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Qwen35MoeModelError {
    Io { path: PathBuf, message: String },
    Invalid(String),
}

impl fmt::Display for Qwen35MoeModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => write!(
                formatter,
                "Qwen3.5 MoE I/O error at {}: {message}",
                path.display()
            ),
            Self::Invalid(message) => write!(formatter, "invalid Qwen3.5 MoE artifact: {message}"),
        }
    }
}

impl std::error::Error for Qwen35MoeModelError {}

#[derive(Clone, Debug, Deserialize)]
struct SafeTensorEntry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Clone, Debug)]
struct LocatedEntry {
    file: String,
    data_start: u64,
    entry: SafeTensorEntry,
}

fn invalid(message: impl Into<String>) -> Qwen35MoeModelError {
    Qwen35MoeModelError::Invalid(message.into())
}

fn read_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>, Qwen35MoeModelError> {
    let mut file = File::open(path).map_err(|error| Qwen35MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let opened = file.metadata().map_err(|error| Qwen35MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let path_before = fs::metadata(path).map_err(|error| Qwen35MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let identity = ShardIdentity::from_metadata(&opened);
    if !opened.is_file()
        || opened.len() > maximum
        || ShardIdentity::from_metadata(&path_before) != identity
    {
        return Err(invalid(format!(
            "bounded regular file contract differs: {}",
            path.display()
        )));
    }
    let mut bytes = Vec::new();
    (&mut file)
        .take(maximum.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| Qwen35MoeModelError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    let after = file.metadata().map_err(|error| Qwen35MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let path_after = fs::metadata(path).map_err(|error| Qwen35MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    if u64::try_from(bytes.len()).ok() != Some(identity.size)
        || ShardIdentity::from_metadata(&after) != identity
        || ShardIdentity::from_metadata(&path_after) != identity
    {
        return Err(invalid(format!(
            "bounded regular file identity changed during read: {}",
            path.display()
        )));
    }
    Ok(bytes)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn exact_u64(value: &Value, key: &str, expected: u64) -> Result<(), Qwen35MoeModelError> {
    if value.get(key).and_then(Value::as_u64) != Some(expected) {
        return Err(invalid(format!("config field differs: {key}")));
    }
    Ok(())
}

pub fn validate_qwen35_moe_config(
    bytes: &[u8],
) -> Result<(Qwen35MoeConfig, Qwen35MoeRecipe), Qwen35MoeModelError> {
    if sha256(bytes) != CONFIG_SHA256 {
        return Err(invalid("config SHA-256 differs"));
    }
    let root: Value =
        serde_json::from_slice(bytes).map_err(|error| invalid(format!("config JSON: {error}")))?;
    if root
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|v| v.first())
        .and_then(Value::as_str)
        != Some("Qwen3_5MoeForConditionalGeneration")
        || root.get("model_type").and_then(Value::as_str) != Some("qwen3_5_moe")
    {
        return Err(invalid("architecture identity differs"));
    }
    let text = root
        .get("text_config")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("text_config is absent"))?;
    let text_value = Value::Object(text.clone());
    for (key, expected) in [
        ("hidden_size", 2_048),
        ("num_hidden_layers", 40),
        ("num_attention_heads", 16),
        ("num_key_value_heads", 2),
        ("head_dim", 256),
        ("num_experts", 256),
        ("num_experts_per_tok", 8),
        ("moe_intermediate_size", 512),
        ("shared_expert_intermediate_size", 512),
        ("vocab_size", 248_320),
        ("full_attention_interval", 4),
        ("mtp_num_hidden_layers", 1),
    ] {
        exact_u64(&text_value, key, expected)?;
    }
    if text.get("model_type").and_then(Value::as_str) != Some("qwen3_5_moe_text")
        || text.get("hidden_act").and_then(Value::as_str) != Some("silu")
        || text.get("dtype").and_then(Value::as_str) != Some("bfloat16")
        || text.get("use_cache").and_then(Value::as_bool) != Some(true)
        || text
            .get("mtp_use_dedicated_embeddings")
            .and_then(Value::as_bool)
            != Some(false)
        || text.get("rms_norm_eps").and_then(Value::as_f64) != Some(1.0e-6)
    {
        return Err(invalid("text semantic field differs"));
    }
    let layer_types = text
        .get("layer_types")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("layer_types is absent"))?;
    if layer_types.len() != 40 {
        return Err(invalid("layer schedule length differs"));
    }
    let mut reviewed_layers = Vec::with_capacity(40);
    for (layer, value) in layer_types.iter().enumerate() {
        let expected = if (layer + 1) % 4 == 0 {
            "full_attention"
        } else {
            "linear_attention"
        };
        if value.as_str() != Some(expected) {
            return Err(invalid(format!("layer schedule differs at {layer}")));
        }
        reviewed_layers.push(if expected == "full_attention" {
            LayerType::FullAttention
        } else {
            LayerType::LinearAttention
        });
    }
    let quant = root
        .get("quantization_config")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("quantization_config is absent"))?;
    if quant.get("quant_method").and_then(Value::as_str) != Some("quark")
        || quant.get("quant_mode").and_then(Value::as_str) != Some("eager_mode")
        || quant.get("version").and_then(Value::as_str) != Some("0.12+87177acebb6")
    {
        return Err(invalid("Quark recipe identity differs"));
    }
    let global = quant
        .get("global_quant_config")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("global quantization config is absent"))?;
    for (name, dynamic) in [("weight", false), ("input_tensors", true)] {
        let spec = global
            .get(name)
            .and_then(Value::as_object)
            .ok_or_else(|| invalid(format!("{name} quantization spec is absent")))?;
        if spec.get("dtype").and_then(Value::as_str) != Some("fp4")
            || spec.get("qscheme").and_then(Value::as_str) != Some("per_group")
            || spec.get("group_size").and_then(Value::as_u64) != Some(32)
            || spec.get("scale_format").and_then(Value::as_str) != Some("e8m0")
            || spec.get("is_dynamic").and_then(Value::as_bool) != Some(dynamic)
        {
            return Err(invalid(format!("{name} quantization spec differs")));
        }
    }
    let exclude = quant
        .get("exclude")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("quantization exclusions are absent"))?;
    if exclude.len() != 1_364 {
        return Err(invalid("quantization exclusion count differs"));
    }
    let exclusions: BTreeSet<&str> = exclude
        .iter()
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| invalid("non-string quantization exclusion"))
        })
        .collect::<Result<_, _>>()?;
    if exclusions.len() != exclude.len() || !exclusions.contains("lm_head") {
        return Err(invalid("quantization exclusions differ"));
    }
    for layer in 0..40 {
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        for suffix in [
            "gate",
            "shared_expert.down_proj",
            "shared_expert.gate_proj",
            "shared_expert.up_proj",
            "shared_expert_gate",
        ] {
            if !exclusions.contains(format!("{prefix}.{suffix}").as_str()) {
                return Err(invalid(format!(
                    "required exclusion is absent at layer {layer}"
                )));
            }
        }
        if exclusions
            .iter()
            .any(|name| name.starts_with(&format!("{prefix}.experts.")))
        {
            return Err(invalid("routed expert is unexpectedly excluded from MXFP4"));
        }
    }
    Ok((
        Qwen35MoeConfig {
            hidden_size: 2_048,
            layer_count: 40,
            attention_heads: 16,
            kv_heads: 2,
            head_dim: 256,
            expert_count: 256,
            selected_expert_count: 8,
            expert_intermediate_size: 512,
            shared_expert_intermediate_size: 512,
            layer_types: reviewed_layers,
        },
        Qwen35MoeRecipe {
            encoding: QuantizedTensorEncoding::Mxfp4E2M1Block32E8M0,
            group_size: 32,
            weight_scale_format: "E8M0",
            activation_dynamic: true,
            activation_scale_format: "E8M0",
            quark_version: "0.12+87177acebb6".to_owned(),
            exclusion_count: 1_364,
        },
    ))
}

fn require<'a>(
    entries: &'a BTreeMap<String, LocatedEntry>,
    name: &str,
    dtype: &str,
    shape: &[u64],
) -> Result<&'a LocatedEntry, Qwen35MoeModelError> {
    let value = entries
        .get(name)
        .ok_or_else(|| invalid(format!("required tensor is absent: {name}")))?;
    if value.entry.dtype != dtype || value.entry.shape != shape {
        return Err(invalid(format!("tensor metadata differs: {name}")));
    }
    Ok(value)
}

fn plane(name: String, value: &LocatedEntry) -> Qwen35MoeTensorPlane {
    Qwen35MoeTensorPlane {
        source_file: value.file.clone(),
        source_name: name,
        dtype: value.entry.dtype.clone(),
        shape: value.entry.shape.clone(),
        absolute_byte_range: [
            value.data_start + value.entry.data_offsets[0],
            value.data_start + value.entry.data_offsets[1],
        ],
    }
}

fn validate_shard(
    path: &Path,
    expected_size: u64,
    expected_sha: &str,
) -> Result<(BoundShard, u64, BTreeMap<String, SafeTensorEntry>), Qwen35MoeModelError> {
    let mut file = File::open(path).map_err(|error| Qwen35MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let path_metadata = fs::metadata(path).map_err(|error| Qwen35MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let metadata = file.metadata().map_err(|error| Qwen35MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let identity = ShardIdentity::from_metadata(&metadata);
    if !metadata.is_file()
        || metadata.len() != expected_size
        || ShardIdentity::from_metadata(&path_metadata) != identity
    {
        return Err(invalid(format!("shard size differs: {}", path.display())));
    }
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 4 * 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| Qwen35MoeModelError::Io {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    if format!("{:x}", hasher.finalize()) != expected_sha {
        return Err(invalid(format!(
            "shard SHA-256 differs: {}",
            path.display()
        )));
    }
    let after_hash = file.metadata().map_err(|error| Qwen35MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let path_after_hash = fs::metadata(path).map_err(|error| Qwen35MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    if ShardIdentity::from_metadata(&after_hash) != identity
        || ShardIdentity::from_metadata(&path_after_hash) != identity
    {
        return Err(invalid(format!(
            "shard identity changed during verification: {}",
            path.display()
        )));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| Qwen35MoeModelError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    let mut length_bytes = [0_u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|error| Qwen35MoeModelError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    let length = u64::from_le_bytes(length_bytes);
    if length == 0 || length > MAX_HEADER_BYTES || 8 + length > expected_size {
        return Err(invalid("safetensors header length differs"));
    }
    let mut header = vec![0_u8; length as usize];
    file.read_exact(&mut header)
        .map_err(|error| Qwen35MoeModelError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    let mut raw: BTreeMap<String, Value> = serde_json::from_slice(&header)
        .map_err(|error| invalid(format!("safetensors header JSON: {error}")))?;
    raw.remove("__metadata__");
    let mut result = BTreeMap::new();
    let mut ranges = Vec::new();
    for (name, value) in raw {
        let entry: SafeTensorEntry = serde_json::from_value(value)
            .map_err(|error| invalid(format!("tensor metadata {name}: {error}")))?;
        if entry.data_offsets[1] < entry.data_offsets[0] {
            return Err(invalid(format!("tensor range underflow: {name}")));
        }
        ranges.push((entry.data_offsets, name.clone()));
        result.insert(name, entry);
    }
    ranges.sort_by_key(|item| item.0[0]);
    let mut cursor = 0_u64;
    for (range, name) in ranges {
        if range[0] != cursor {
            return Err(invalid(format!("non-contiguous tensor range: {name}")));
        }
        cursor = range[1];
    }
    if 8 + length + cursor != expected_size {
        return Err(invalid("safetensors payload extent differs"));
    }
    Ok((
        BoundShard {
            file: Arc::new(file),
            identity,
        },
        8 + length,
        result,
    ))
}

pub fn verify_qwen35_moe_artifact(
    root: impl AsRef<Path>,
) -> Result<VerifiedQwen35Moe, Qwen35MoeModelError> {
    let root = root.as_ref();
    let mut support_files = BTreeMap::new();
    for (name, size, digest) in SUPPORT_FILES {
        let bytes = read_bounded(&root.join(name), size)?;
        if bytes.len() as u64 != size || sha256(&bytes) != digest {
            return Err(invalid(format!("support file identity differs: {name}")));
        }
        support_files.insert(name.to_owned(), Arc::<[u8]>::from(bytes));
    }
    let config_bytes = read_bounded(&root.join("config.json"), MAX_CONFIG_BYTES)?;
    let (config, recipe) = validate_qwen35_moe_config(&config_bytes)?;
    let index_bytes = read_bounded(&root.join("model.safetensors.index.json"), MAX_INDEX_BYTES)?;
    if sha256(&index_bytes) != INDEX_SHA256 {
        return Err(invalid("safetensors index SHA-256 differs"));
    }
    let index: Value = serde_json::from_slice(&index_bytes)
        .map_err(|error| invalid(format!("index JSON: {error}")))?;
    if index
        .pointer("/metadata/total_size")
        .and_then(Value::as_u64)
        != Some(24_600_620_848)
    {
        return Err(invalid("index total_size differs"));
    }
    let weight_map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("index weight_map is absent"))?;
    if weight_map.len() != QWEN35_MOE_TENSOR_COUNT {
        return Err(invalid("index tensor count differs"));
    }
    let mut entries = BTreeMap::new();
    let mut shards = BTreeMap::new();
    for (file_name, size, digest) in SHARDS {
        let (shard, data_start, header) = validate_shard(&root.join(file_name), size, digest)?;
        for (name, entry) in header {
            if weight_map.get(&name).and_then(Value::as_str) != Some(file_name) {
                return Err(invalid(format!(
                    "index/header shard mapping differs: {name}"
                )));
            }
            if entries
                .insert(
                    name.clone(),
                    LocatedEntry {
                        file: file_name.to_owned(),
                        data_start,
                        entry,
                    },
                )
                .is_some()
            {
                return Err(invalid(format!("duplicate tensor across shards: {name}")));
            }
        }
        shards.insert(file_name.to_owned(), shard);
    }
    if entries.len() != QWEN35_MOE_TENSOR_COUNT
        || entries.keys().any(|name| !weight_map.contains_key(name))
    {
        return Err(invalid("header/index tensor set differs"));
    }
    let mut catalog = Sha256::new();
    let mut text_catalog = Sha256::new();
    let mut text_planes = Vec::with_capacity(QWEN35_MOE_TEXT_TENSOR_COUNT);
    let mut text_count = 0_usize;
    let mut vision_count = 0_usize;
    let mut mtp_count = 0_usize;
    let mut text_bytes = 0_u64;
    for (name, located) in &entries {
        let row = serde_json::to_string(&(
            name,
            &located.file,
            &located.entry.dtype,
            &located.entry.shape,
            &located.entry.data_offsets,
        ))
        .unwrap();
        catalog.update(row.as_bytes());
        catalog.update(b"\n");
        let bytes = located.entry.data_offsets[1] - located.entry.data_offsets[0];
        if name.starts_with("model.language_model.") || name.starts_with("lm_head.") {
            text_count += 1;
            text_bytes += bytes;
            text_catalog.update(row.as_bytes());
            text_catalog.update(b"\n");
            text_planes.push(plane(name.clone(), located));
        } else if name.starts_with("model.visual.") {
            vision_count += 1;
        } else if name.starts_with("mtp.") {
            mtp_count += 1;
        } else {
            return Err(invalid(format!("unknown tensor component: {name}")));
        }
    }
    if format!("{:x}", catalog.finalize()) != CATALOG_SHA256
        || format!("{:x}", text_catalog.finalize()) != TEXT_CATALOG_SHA256
        || text_count != QWEN35_MOE_TEXT_TENSOR_COUNT
        || text_bytes != QWEN35_MOE_TEXT_RESIDENT_BYTES
        || vision_count != QWEN35_MOE_VISION_TENSOR_COUNT
        || mtp_count != QWEN35_MOE_MTP_TENSOR_COUNT
    {
        return Err(invalid("catalog component contract differs"));
    }
    let mut experts = BTreeMap::new();
    for layer in 0..40_u16 {
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        require(
            &entries,
            &format!("{prefix}.gate.weight"),
            "BF16",
            &[256, 2_048],
        )?;
        require(
            &entries,
            &format!("{prefix}.shared_expert_gate.weight"),
            "BF16",
            &[1, 2_048],
        )?;
        for (stem, shape) in [
            ("gate_proj", [512, 2_048]),
            ("up_proj", [512, 2_048]),
            ("down_proj", [2_048, 512]),
        ] {
            require(
                &entries,
                &format!("{prefix}.shared_expert.{stem}.weight"),
                "BF16",
                &shape,
            )?;
        }
        for expert in 0..256_u16 {
            for projection in [
                Qwen35MoeExpertProjection::Gate,
                Qwen35MoeExpertProjection::Up,
                Qwen35MoeExpertProjection::Down,
            ] {
                let base = format!("{prefix}.experts.{expert}.{}", projection.source_stem());
                let value_name = format!("{base}.weight");
                let scale_name = format!("{base}.weight_scale");
                let value = require(&entries, &value_name, "U8", &projection.value_shape())?;
                let scale = require(&entries, &scale_name, "U8", &projection.scale_shape())?;
                experts.insert(
                    (layer, expert, projection),
                    Qwen35MoeExpertTensor {
                        layer,
                        expert,
                        projection,
                        logical_shape: projection.logical_shape(),
                        encoding: QuantizedTensorEncoding::Mxfp4E2M1Block32E8M0,
                        value: plane(value_name, value),
                        scale: plane(scale_name, scale),
                    },
                );
            }
        }
    }
    if experts.len() != QWEN35_MOE_EXPERT_PROJECTION_COUNT {
        return Err(invalid("expert projection count differs"));
    }
    Ok(VerifiedQwen35Moe {
        root: root.to_path_buf(),
        config,
        recipe,
        text_planes,
        experts,
        support_files,
        shards,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_shard_path() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("test clock is after the Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "sllm-qwen35-moe-shard-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn bound_shard_reads_positionally_and_rejects_same_inode_mutation() {
        let path = temporary_shard_path();
        fs::write(&path, b"0123456789abcdef").unwrap();
        let file = Arc::new(File::open(&path).unwrap());
        let identity = ShardIdentity::from_metadata(&file.metadata().unwrap());
        let shard = BoundShard { file, identity };
        assert_eq!(shard.read_range(3, 5).unwrap(), b"34567");

        let mut writer = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        writer.write_all(b"changed").unwrap();
        writer.sync_all().unwrap();
        assert!(shard.read_range(0, 1).is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn reviewed_size_budget_leaves_room_on_a_32_gib_device() {
        assert_eq!(QWEN35_MOE_TEXT_RESIDENT_BYTES, 22_009_481_856);
        let remaining_bytes = 32_u64 * 1024 * 1024 * 1024 - QWEN35_MOE_TEXT_RESIDENT_BYTES;
        assert!(remaining_bytes > 11_u64 * 1024 * 1024 * 1024);
    }

    #[test]
    fn expert_projection_shapes_preserve_ocp_block_32_packing() {
        assert_eq!(Qwen35MoeExpertProjection::Gate.value_shape(), [512, 1_024]);
        assert_eq!(Qwen35MoeExpertProjection::Gate.scale_shape(), [512, 64]);
        assert_eq!(Qwen35MoeExpertProjection::Down.value_shape(), [2_048, 256]);
        assert_eq!(Qwen35MoeExpertProjection::Down.scale_shape(), [2_048, 16]);
    }

    #[cfg(feature = "reviewed-qwen35-external-cache")]
    #[test]
    fn reviewed_external_artifact_passes_full_identity_and_inventory() {
        let Some(root) = std::env::var_os("SLLM_QWEN35_MOE_CACHE") else {
            return;
        };
        let model = verify_qwen35_moe_artifact(root).unwrap();
        assert_eq!(model.experts().len(), QWEN35_MOE_EXPERT_PROJECTION_COUNT);
        assert_eq!(model.config().layer_types.len(), 40);
        assert_eq!(model.text_planes().len(), QWEN35_MOE_TEXT_TENSOR_COUNT);
        assert_eq!(
            model.recipe().encoding,
            QuantizedTensorEncoding::Mxfp4E2M1Block32E8M0
        );
        assert!(
            model
                .expert(0, 0, Qwen35MoeExpertProjection::Gate)
                .is_some()
        );
        assert!(
            model
                .expert(39, 255, Qwen35MoeExpertProjection::Down)
                .is_some()
        );
        let plan = build_qwen35_moe_weight_load_plan(&model).unwrap();
        let graph = crate::build_qwen35_moe_execution_graph(&model, &plan, 3, 257).unwrap();
        assert_eq!(plan.entries.len(), 493);
        assert_eq!(plan.total_destination_bytes, QWEN35_MOE_TEXT_RESIDENT_BYTES);
        assert_eq!(graph.layer_types().len(), 40);
        assert_eq!(
            graph
                .nodes()
                .iter()
                .filter(|node| {
                    matches!(
                        node.kind(),
                        crate::QwenGraphNodeKind::Semantic(crate::SemanticOpKind::SparseMoe)
                    )
                })
                .count(),
            40
        );
    }
}
