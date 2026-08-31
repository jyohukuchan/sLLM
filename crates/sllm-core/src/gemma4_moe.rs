//! Exact source identity for the reviewed Gemma 4 26B-A4B NVFP4 artifact.
//!
//! This module is intentionally container-facing only. It freezes the upstream
//! config, support files, shard identities, and safetensors catalog before a
//! later load-plan or execution layer is allowed to allocate resident weights.

use crate::{
    BudgetBoundary, Gemma4LayerType, GenerationStopPolicyV1, GgufRecipeEncoding, GgufScaleRole,
    GgufTensorScope, GgufTensorType, GgufValue, MaxNewTokensZero, PromptEvaluation,
    QuantizedTensorEncoding, StopEvaluation, StopTokenHandling, TensorDType, VerifiedDerivedGguf,
    VerifiedGguf, WEIGHT_LOAD_CHUNK_BYTES, WeightClassification, WeightConsumer, WeightConsumerKey,
    WeightLoadChunk, WeightLoadEntry, WeightLoadPlan,
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

pub const GEMMA4_MOE_REPOSITORY: &str = "nvidia/Gemma-4-26B-A4B-NVFP4";
pub const GEMMA4_MOE_REVISION: &str = "a19cfe00be84568a6867111c9a68c9c44fdcffe6";
pub const GEMMA4_MOE_SEMANTIC_REPOSITORY: &str = "google/gemma-4-26B-A4B-it";
pub const GEMMA4_MOE_SEMANTIC_REVISION: &str = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52";
pub const GEMMA4_MOE_LICENSE: &str = "gemma";
pub const GEMMA4_MOE_TENSOR_COUNT: usize = 47_033;
pub const GEMMA4_MOE_TEXT_TENSOR_COUNT: usize = 46_677;
pub const GEMMA4_MOE_VISION_TENSOR_COUNT: usize = 356;
pub const GEMMA4_MOE_EXPERT_PROJECTION_COUNT: usize = 30 * 128 * 3;
pub const GEMMA4_MOE_TEXT_RESIDENT_BYTES: u64 = 17_636_771_900;
pub const GEMMA4_MOE_ADVERTISED_PAYLOAD_BYTES: u64 = 18_782_360_732;
pub const GEMMA4_MOE_MODEL_FINGERPRINT: &str =
    "sha256:69ed6c3b18fcc944d62a4ac8d6357bd760ef0181263f83f1a7f43d0415cb846f";
pub const GEMMA4_MOE_LAYER_BLOB_BYTES: u64 = 428_215_552;
pub const GEMMA4_MOE_LAYER_BLOB_PREFIX: &str = "__sllm_gemma4_moe_layer_blob.";
pub const GEMMA4_MOE_EXPERT_VALUE_BYTES: u64 = 991_232;
pub const GEMMA4_MOE_EXPERT_BLOCK_SCALE_BYTES: u64 = 123_904;
pub const GEMMA4_MOE_GATE_VALUES_OFFSET: u64 = 0;
pub const GEMMA4_MOE_GATE_SCALES_OFFSET: u64 = 126_877_696;
pub const GEMMA4_MOE_GATE_OUTER_SCALES_OFFSET: u64 = 142_737_408;
pub const GEMMA4_MOE_GATE_INPUT_SCALES_OFFSET: u64 = 142_737_920;
pub const GEMMA4_MOE_UP_VALUES_OFFSET: u64 = 142_738_432;
pub const GEMMA4_MOE_UP_SCALES_OFFSET: u64 = 269_616_128;
pub const GEMMA4_MOE_UP_OUTER_SCALES_OFFSET: u64 = 285_475_840;
pub const GEMMA4_MOE_UP_INPUT_SCALES_OFFSET: u64 = 285_476_352;
pub const GEMMA4_MOE_DOWN_VALUES_OFFSET: u64 = 285_476_864;
pub const GEMMA4_MOE_DOWN_SCALES_OFFSET: u64 = 412_354_560;
pub const GEMMA4_MOE_DOWN_OUTER_SCALES_OFFSET: u64 = 428_214_272;
pub const GEMMA4_MOE_DOWN_INPUT_SCALES_OFFSET: u64 = 428_214_784;
pub const GEMMA4_MOE_PER_EXPERT_SCALES_OFFSET: u64 = 428_215_296;

const CONFIG_SHA256: &str = "4e379cc809c617a49179a49140f553a2d6a5ec538ed480832b0c54f6ace43d98";
const INDEX_SHA256: &str = "ac5e677ed9f8d9b689170bbae0c88ce163e284d02b09100e14afddaa9ec4a15c";
const CATALOG_SHA256: &str = "39a71cfea1a8080996ccae7ec89299c6d24c399834e482389f9fb0a947100068";
const TEXT_CATALOG_SHA256: &str =
    "69ed6c3b18fcc944d62a4ac8d6357bd760ef0181263f83f1a7f43d0415cb846f";
const MAX_HEADER_BYTES: u64 = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gemma4MoeFileIdentity {
    pub file_name: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

pub const GEMMA4_MOE_SUPPORT_FILES: [Gemma4MoeFileIdentity; 7] = [
    Gemma4MoeFileIdentity {
        file_name: "README.md",
        size: 9_527,
        sha256: "5d20eecdd5d4a08acc541e9884bf967055e706baa4d100d4623d7575d65080b0",
    },
    Gemma4MoeFileIdentity {
        file_name: "chat_template.jinja",
        size: 16_934,
        sha256: "94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413",
    },
    Gemma4MoeFileIdentity {
        file_name: "generation_config.json",
        size: 208,
        sha256: "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
    },
    Gemma4MoeFileIdentity {
        file_name: "hf_quant_config.json",
        size: 5_188,
        sha256: "fca2ea21cded31e6cff2c56ee83a162dcd5d3ff292cbf2f6083702e9fc324454",
    },
    Gemma4MoeFileIdentity {
        file_name: "processor_config.json",
        size: 1_689,
        sha256: "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
    },
    Gemma4MoeFileIdentity {
        file_name: "tokenizer.json",
        size: 32_169_626,
        sha256: "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    },
    Gemma4MoeFileIdentity {
        file_name: "tokenizer_config.json",
        size: 2_095,
        sha256: "90c3a3ba5bf53818383a58e1a776cbcacd2a038d4812eaa373e1522f2d06f3df",
    },
];

pub const GEMMA4_MOE_SHARDS: [Gemma4MoeFileIdentity; 2] = [
    Gemma4MoeFileIdentity {
        file_name: "model-00001-of-00002.safetensors",
        size: 10_001_865_236,
        sha256: "b5df31122600666617b05f9be2015552cd2edff401e86b1d99b9127efdc6d819",
    },
    Gemma4MoeFileIdentity {
        file_name: "model-00002-of-00002.safetensors",
        size: 8_786_620_352,
        sha256: "ff11061ebf57327af4f1993ff758b0859d7746b0c03ca1b17ded7dec30410962",
    },
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeConfig {
    pub hidden_size: u32,
    pub layer_count: u32,
    pub attention_heads: u32,
    pub sliding_kv_heads: u32,
    pub full_kv_heads: u32,
    pub sliding_head_dim: u32,
    pub full_head_dim: u32,
    pub sliding_window: u32,
    pub max_position_embeddings: u32,
    pub vocab_size: u32,
    pub dense_intermediate_size: u32,
    pub expert_count: u32,
    pub selected_expert_count: u32,
    pub expert_intermediate_size: u32,
    pub layer_types: Vec<Gemma4LayerType>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeRecipe {
    pub encoding: QuantizedTensorEncoding,
    pub block_size: u32,
    pub value_format: &'static str,
    pub block_scale_format: &'static str,
    pub outer_scale_format: &'static str,
    pub input_scale_format: &'static str,
    pub activation_dynamic: bool,
    pub kv_cache_format: &'static str,
    /// ModelOpt `fp8_cast` uses constant amax 448, hence an implicit unit
    /// dequantization scale and no serialized per-layer scale tensors.
    pub kv_cache_scale_source: &'static str,
    pub kv_cache_dequant_scale_f32_bits: u32,
    pub kv_cache_scale_tensor_count: u32,
    pub producer: String,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Gemma4MoeExpertProjection {
    Gate,
    Up,
    Down,
}

impl Gemma4MoeExpertProjection {
    pub const fn source_stem(self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }

    pub const fn logical_shape(self) -> [u64; 2] {
        match self {
            Self::Gate | Self::Up => [704, 2_816],
            Self::Down => [2_816, 704],
        }
    }

    pub const fn value_shape(self) -> [u64; 2] {
        match self {
            Self::Gate | Self::Up => [704, 1_408],
            Self::Down => [2_816, 352],
        }
    }

    pub const fn block_scale_shape(self) -> [u64; 2] {
        match self {
            Self::Gate | Self::Up => [704, 176],
            Self::Down => [2_816, 44],
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeTensorPlane {
    pub source_file: String,
    pub source_name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub absolute_byte_range: [u64; 2],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeExpertTensor {
    pub layer: u16,
    pub expert: u16,
    pub projection: Gemma4MoeExpertProjection,
    pub logical_shape: [u64; 2],
    pub encoding: QuantizedTensorEncoding,
    pub value: Gemma4MoeTensorPlane,
    pub block_scale: Gemma4MoeTensorPlane,
    pub outer_scale: Gemma4MoeTensorPlane,
    pub input_scale: Gemma4MoeTensorPlane,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeExpertPlanes {
    pub values: Vec<u8>,
    pub block_scales: Vec<u8>,
    pub outer_scale: [u8; 4],
    pub input_scale: [u8; 4],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeLayerBlobPackInput {
    pub expert: u16,
    pub projection: Gemma4MoeExpertProjection,
    pub value_destination: [u64; 2],
    pub block_scale_destination: [u64; 2],
    pub outer_scale_destination: [u64; 2],
    pub input_scale_destination: [u64; 2],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4MoeIndex {
    pub total_parameters: u64,
    pub total_size: u64,
    weight_map: BTreeMap<String, String>,
}

impl Gemma4MoeIndex {
    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    pub fn source_file(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(String::as_str)
    }
}

#[derive(Debug)]
pub struct VerifiedGemma4Moe {
    root: PathBuf,
    config: Gemma4MoeConfig,
    recipe: Gemma4MoeRecipe,
    all_planes: Vec<Gemma4MoeTensorPlane>,
    text_planes: Vec<Gemma4MoeTensorPlane>,
    experts: BTreeMap<(u16, u16, Gemma4MoeExpertProjection), Gemma4MoeExpertTensor>,
    support_files: BTreeMap<String, Arc<[u8]>>,
    shards: BTreeMap<String, VerifiedShard>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GgufGemma4MoeExpertBinding {
    value_tensor: String,
    outer_scale_tensor: String,
    input_scale_tensor: String,
}

type GgufGemma4MoeExpertMap =
    BTreeMap<(u16, u16, Gemma4MoeExpertProjection), GgufGemma4MoeExpertBinding>;
type GgufGemma4MoeCatalog = (Vec<Gemma4MoeTensorPlane>, GgufGemma4MoeExpertMap);

#[derive(Debug)]
pub struct VerifiedGgufGemma4Moe {
    gguf: VerifiedGguf,
    file_sha256: String,
    config: Gemma4MoeConfig,
    direct_planes: Vec<Gemma4MoeTensorPlane>,
    experts: GgufGemma4MoeExpertMap,
}

impl VerifiedGemma4Moe {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn config(&self) -> &Gemma4MoeConfig {
        &self.config
    }

    pub fn recipe(&self) -> &Gemma4MoeRecipe {
        &self.recipe
    }

    pub fn all_planes(&self) -> &[Gemma4MoeTensorPlane] {
        &self.all_planes
    }

    pub fn text_planes(&self) -> &[Gemma4MoeTensorPlane] {
        &self.text_planes
    }

    pub fn expert(
        &self,
        layer: u16,
        expert: u16,
        projection: Gemma4MoeExpertProjection,
    ) -> Option<&Gemma4MoeExpertTensor> {
        self.experts.get(&(layer, expert, projection))
    }

    pub fn experts(&self) -> impl ExactSizeIterator<Item = &Gemma4MoeExpertTensor> {
        self.experts.values()
    }

    pub fn plane(&self, name: &str) -> Option<&Gemma4MoeTensorPlane> {
        self.text_planes
            .binary_search_by(|plane| plane.source_name.as_str().cmp(name))
            .ok()
            .and_then(|index| self.text_planes.get(index))
    }

    pub fn locked_shard(&self, name: &str) -> Option<Gemma4MoeFileIdentity> {
        GEMMA4_MOE_SHARDS
            .iter()
            .copied()
            .find(|identity| identity.file_name == name)
    }

    pub fn read_support_file(&self, name: &str) -> Result<Vec<u8>, Gemma4MoeModelError> {
        self.support_files
            .get(name)
            .map(|bytes| bytes.to_vec())
            .ok_or_else(|| invalid(format!("unsupported Gemma 4 MoE support file: {name}")))
    }

    pub fn read_plane_range(
        &self,
        plane: &Gemma4MoeTensorPlane,
        relative_offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, Gemma4MoeModelError> {
        let plane_length = plane.absolute_byte_range[1]
            .checked_sub(plane.absolute_byte_range[0])
            .ok_or_else(|| invalid("tensor plane range underflow"))?;
        let length_u64 = u64::try_from(length).map_err(|_| invalid("tensor range is too large"))?;
        if relative_offset
            .checked_add(length_u64)
            .is_none_or(|end| end > plane_length)
        {
            return Err(invalid("tensor read exceeds the verified plane"));
        }
        let absolute_offset = plane.absolute_byte_range[0]
            .checked_add(relative_offset)
            .ok_or_else(|| invalid("tensor read offset overflow"))?;
        self.shards
            .get(&plane.source_file)
            .ok_or_else(|| invalid("tensor shard is not bound"))?
            .read_exact_at(absolute_offset, length)
    }

    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>, Gemma4MoeModelError> {
        let plane = self
            .plane(name)
            .ok_or_else(|| invalid(format!("Gemma 4 MoE tensor is absent: {name}")))?;
        let length = plane_byte_length(plane)?;
        self.read_plane_range(plane, 0, length)
    }

    pub fn read_expert_planes(
        &self,
        layer: u16,
        expert: u16,
        projection: Gemma4MoeExpertProjection,
    ) -> Result<Gemma4MoeExpertPlanes, Gemma4MoeModelError> {
        let tensor = self
            .expert(layer, expert, projection)
            .ok_or_else(|| invalid("Gemma 4 MoE expert tensor is absent"))?;
        let values = self.read_plane_range(
            &tensor.value,
            0,
            usize::try_from(GEMMA4_MOE_EXPERT_VALUE_BYTES)
                .map_err(|_| invalid("expert value length exceeds usize"))?,
        )?;
        let block_scales = self.read_plane_range(
            &tensor.block_scale,
            0,
            usize::try_from(GEMMA4_MOE_EXPERT_BLOCK_SCALE_BYTES)
                .map_err(|_| invalid("expert scale length exceeds usize"))?,
        )?;
        Ok(Gemma4MoeExpertPlanes {
            values,
            block_scales,
            outer_scale: read_source_f32_scale(self, &tensor.outer_scale)?,
            input_scale: read_source_f32_scale(self, &tensor.input_scale)?,
        })
    }

    pub fn layer_blob_pack_inputs(
        &self,
        layer: u32,
    ) -> Result<Vec<Gemma4MoeLayerBlobPackInput>, Gemma4MoeModelError> {
        gemma4_moe_layer_blob_pack_inputs(layer)
    }
}

impl VerifiedGgufGemma4Moe {
    pub fn gguf(&self) -> &VerifiedGguf {
        &self.gguf
    }

    pub fn config(&self) -> &Gemma4MoeConfig {
        &self.config
    }

    pub fn file_sha256(&self) -> &str {
        &self.file_sha256
    }

    pub fn direct_planes(&self) -> &[Gemma4MoeTensorPlane] {
        &self.direct_planes
    }

    pub fn expert_tensor_name(
        &self,
        layer: u16,
        expert: u16,
        projection: Gemma4MoeExpertProjection,
    ) -> Option<&str> {
        self.experts
            .get(&(layer, expert, projection))
            .map(|binding| binding.value_tensor.as_str())
    }

    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>, Gemma4MoeModelError> {
        let tensor = self
            .gguf
            .tensor(name)
            .ok_or_else(|| invalid(format!("GGUF Gemma 4 MoE tensor is absent: {name}")))?;
        let length = usize::try_from(tensor.byte_length())
            .map_err(|_| invalid("GGUF Gemma 4 MoE tensor is too large"))?;
        self.gguf
            .read_tensor_range(name, 0, length)
            .map_err(|error| invalid(error.to_string()))
    }

    pub fn read_expert_planes(
        &self,
        layer: u16,
        expert: u16,
        projection: Gemma4MoeExpertProjection,
    ) -> Result<Gemma4MoeExpertPlanes, Gemma4MoeModelError> {
        let binding = self
            .experts
            .get(&(layer, expert, projection))
            .ok_or_else(|| invalid("GGUF Gemma 4 MoE expert tensor is absent"))?;
        let standard = self.read_tensor(&binding.value_tensor)?;
        let expected_blocks = projection
            .logical_shape()
            .into_iter()
            .try_fold(1_u64, |product, dimension| product.checked_mul(dimension))
            .and_then(|elements| elements.checked_div(64))
            .ok_or_else(|| invalid("GGUF expert block count overflows"))?;
        let expected_standard = expected_blocks
            .checked_mul(36)
            .ok_or_else(|| invalid("GGUF expert byte count overflows"))?;
        if u64::try_from(standard.len()).ok() != Some(expected_standard) {
            return Err(invalid("GGUF NVFP4 standard expert byte count differs"));
        }
        let mut values = Vec::with_capacity(
            usize::try_from(GEMMA4_MOE_EXPERT_VALUE_BYTES)
                .map_err(|_| invalid("expert value length exceeds usize"))?,
        );
        let mut block_scales = Vec::with_capacity(
            usize::try_from(GEMMA4_MOE_EXPERT_BLOCK_SCALE_BYTES)
                .map_err(|_| invalid("expert scale length exceeds usize"))?,
        );
        for block in standard.chunks_exact(36) {
            block_scales.extend_from_slice(&block[..4]);
            for standard_subblock in block[4..].chunks_exact(8) {
                append_adjacent_nvfp4(standard_subblock, &mut values);
            }
        }
        if values.len() as u64 != GEMMA4_MOE_EXPERT_VALUE_BYTES
            || block_scales.len() as u64 != GEMMA4_MOE_EXPERT_BLOCK_SCALE_BYTES
        {
            return Err(invalid("GGUF expert plane split length differs"));
        }
        Ok(Gemma4MoeExpertPlanes {
            values,
            block_scales,
            outer_scale: read_gguf_f32_scale(self, &binding.outer_scale_tensor)?,
            input_scale: read_gguf_f32_scale(self, &binding.input_scale_tensor)?,
        })
    }

    pub fn layer_blob_pack_inputs(
        &self,
        layer: u32,
    ) -> Result<Vec<Gemma4MoeLayerBlobPackInput>, Gemma4MoeModelError> {
        gemma4_moe_layer_blob_pack_inputs(layer)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum ExpertPlaneRole {
    Value,
    BlockScale,
    OuterScale,
    InputScale,
}

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

impl FileIdentity {
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
struct VerifiedShard {
    file: Arc<File>,
    identity: FileIdentity,
}

impl VerifiedShard {
    fn read_exact_at(&self, offset: u64, length: usize) -> Result<Vec<u8>, Gemma4MoeModelError> {
        let before = self
            .file
            .metadata()
            .map_err(|error| invalid(error.to_string()))?;
        if FileIdentity::from_metadata(&before) != self.identity {
            return Err(invalid("verified shard identity changed before read"));
        }
        let length_u64 = u64::try_from(length).map_err(|_| invalid("shard read is too large"))?;
        if offset
            .checked_add(length_u64)
            .is_none_or(|end| end > self.identity.size)
        {
            return Err(invalid("shard read exceeds the verified file"));
        }
        let mut output = vec![0_u8; length];
        let mut read = 0_usize;
        while read < output.len() {
            let position = offset
                .checked_add(u64::try_from(read).map_err(|_| invalid("shard offset overflow"))?)
                .ok_or_else(|| invalid("shard offset overflow"))?;
            let count = self
                .file
                .read_at(&mut output[read..], position)
                .map_err(|error| invalid(error.to_string()))?;
            if count == 0 {
                return Err(invalid("verified shard returned a short read"));
            }
            read += count;
        }
        let after = self
            .file
            .metadata()
            .map_err(|error| invalid(error.to_string()))?;
        if FileIdentity::from_metadata(&after) != self.identity {
            return Err(invalid("verified shard identity changed during read"));
        }
        Ok(output)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Gemma4MoeModelError {
    Io { path: PathBuf, message: String },
    Invalid(String),
}

impl fmt::Display for Gemma4MoeModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => write!(
                formatter,
                "Gemma 4 MoE I/O error at {}: {message}",
                path.display()
            ),
            Self::Invalid(message) => write!(formatter, "invalid Gemma 4 MoE artifact: {message}"),
        }
    }
}

impl std::error::Error for Gemma4MoeModelError {}

fn invalid(message: impl Into<String>) -> Gemma4MoeModelError {
    Gemma4MoeModelError::Invalid(message.into())
}

fn plane_byte_length(plane: &Gemma4MoeTensorPlane) -> Result<usize, Gemma4MoeModelError> {
    usize::try_from(
        plane.absolute_byte_range[1]
            .checked_sub(plane.absolute_byte_range[0])
            .ok_or_else(|| invalid("tensor plane byte range underflows"))?,
    )
    .map_err(|_| invalid("tensor plane length exceeds usize"))
}

fn checked_f32_scale(bytes: Vec<u8>) -> Result<[u8; 4], Gemma4MoeModelError> {
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| invalid("NVFP4 F32 scale is not four bytes"))?;
    let value = f32::from_le_bytes(bytes);
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid("NVFP4 F32 scale is non-positive or non-finite"));
    }
    Ok(bytes)
}

fn read_source_f32_scale(
    source: &VerifiedGemma4Moe,
    plane: &Gemma4MoeTensorPlane,
) -> Result<[u8; 4], Gemma4MoeModelError> {
    if plane.dtype != "F32" || !plane.shape.is_empty() || plane_byte_length(plane)? != 4 {
        return Err(invalid("source NVFP4 F32 scale metadata differs"));
    }
    checked_f32_scale(source.read_plane_range(plane, 0, 4)?)
}

fn read_gguf_f32_scale(
    source: &VerifiedGgufGemma4Moe,
    name: &str,
) -> Result<[u8; 4], Gemma4MoeModelError> {
    let tensor = source
        .gguf
        .tensor(name)
        .ok_or_else(|| invalid("GGUF NVFP4 F32 scale is absent"))?;
    if tensor.tensor_type != GgufTensorType::F32
        || tensor.dimensions.as_slice() != [1]
        || tensor.byte_length() != 4
    {
        return Err(invalid("GGUF NVFP4 F32 scale metadata differs"));
    }
    checked_f32_scale(
        source
            .gguf
            .read_tensor_range(name, 0, 4)
            .map_err(|error| invalid(error.to_string()))?,
    )
}

fn append_adjacent_nvfp4(standard: &[u8], output: &mut Vec<u8>) {
    debug_assert_eq!(standard.len(), 8);
    for adjacent in (0..16).step_by(2) {
        let code = |index: usize| {
            if index < 8 {
                standard[index] & 0x0f
            } else {
                standard[index - 8] >> 4
            }
        };
        output.push(code(adjacent) | code(adjacent + 1) << 4);
    }
}

pub fn gemma4_moe_layer_blob_name(layer: u32) -> String {
    format!("{GEMMA4_MOE_LAYER_BLOB_PREFIX}{layer}")
}

pub const fn gemma4_moe_per_expert_scale_destination() -> [u64; 2] {
    [
        GEMMA4_MOE_PER_EXPERT_SCALES_OFFSET,
        GEMMA4_MOE_LAYER_BLOB_BYTES,
    ]
}

pub fn gemma4_moe_layer_blob_pack_inputs(
    layer: u32,
) -> Result<Vec<Gemma4MoeLayerBlobPackInput>, Gemma4MoeModelError> {
    if layer >= 30 {
        return Err(invalid("Gemma 4 MoE layer blob index is out of range"));
    }
    let mut inputs = Vec::with_capacity(128 * 3);
    for projection in [
        Gemma4MoeExpertProjection::Gate,
        Gemma4MoeExpertProjection::Up,
        Gemma4MoeExpertProjection::Down,
    ] {
        let (values_base, scales_base, outer_base, input_base) = match projection {
            Gemma4MoeExpertProjection::Gate => (
                GEMMA4_MOE_GATE_VALUES_OFFSET,
                GEMMA4_MOE_GATE_SCALES_OFFSET,
                GEMMA4_MOE_GATE_OUTER_SCALES_OFFSET,
                GEMMA4_MOE_GATE_INPUT_SCALES_OFFSET,
            ),
            Gemma4MoeExpertProjection::Up => (
                GEMMA4_MOE_UP_VALUES_OFFSET,
                GEMMA4_MOE_UP_SCALES_OFFSET,
                GEMMA4_MOE_UP_OUTER_SCALES_OFFSET,
                GEMMA4_MOE_UP_INPUT_SCALES_OFFSET,
            ),
            Gemma4MoeExpertProjection::Down => (
                GEMMA4_MOE_DOWN_VALUES_OFFSET,
                GEMMA4_MOE_DOWN_SCALES_OFFSET,
                GEMMA4_MOE_DOWN_OUTER_SCALES_OFFSET,
                GEMMA4_MOE_DOWN_INPUT_SCALES_OFFSET,
            ),
        };
        for expert in 0..128_u16 {
            let expert_u64 = u64::from(expert);
            let value_start = values_base
                .checked_add(expert_u64 * GEMMA4_MOE_EXPERT_VALUE_BYTES)
                .ok_or_else(|| invalid("layer blob value offset overflows"))?;
            let scale_start = scales_base
                .checked_add(expert_u64 * GEMMA4_MOE_EXPERT_BLOCK_SCALE_BYTES)
                .ok_or_else(|| invalid("layer blob scale offset overflows"))?;
            let outer_start = outer_base + expert_u64 * 4;
            let input_start = input_base + expert_u64 * 4;
            inputs.push(Gemma4MoeLayerBlobPackInput {
                expert,
                projection,
                value_destination: [value_start, value_start + GEMMA4_MOE_EXPERT_VALUE_BYTES],
                block_scale_destination: [
                    scale_start,
                    scale_start + GEMMA4_MOE_EXPERT_BLOCK_SCALE_BYTES,
                ],
                outer_scale_destination: [outer_start, outer_start + 4],
                input_scale_destination: [input_start, input_start + 4],
            });
        }
    }
    if inputs
        .last()
        .is_none_or(|input| input.input_scale_destination[1] != GEMMA4_MOE_PER_EXPERT_SCALES_OFFSET)
    {
        return Err(invalid("Gemma 4 MoE layer blob layout differs"));
    }
    Ok(inputs)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn exact_u64(value: &Value, key: &str, expected: u64) -> Result<(), Gemma4MoeModelError> {
    if value.get(key).and_then(Value::as_u64) != Some(expected) {
        return Err(invalid(format!("config field differs: {key}")));
    }
    Ok(())
}

fn validate_config_document(
    root: &Value,
) -> Result<(Gemma4MoeConfig, Gemma4MoeRecipe), Gemma4MoeModelError> {
    if root
        .get("architectures")
        .and_then(Value::as_array)
        .is_none_or(|values| {
            values.as_slice() != [Value::String("Gemma4ForConditionalGeneration".to_owned())]
        })
        || root.get("model_type").and_then(Value::as_str) != Some("gemma4")
    {
        return Err(invalid("architecture identity differs"));
    }
    let text = root
        .get("text_config")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("text_config is absent"))?;
    let text_value = Value::Object(text.clone());
    for (key, expected) in [
        ("hidden_size", 2_816),
        ("num_hidden_layers", 30),
        ("num_attention_heads", 16),
        ("num_key_value_heads", 8),
        ("num_global_key_value_heads", 2),
        ("head_dim", 256),
        ("global_head_dim", 512),
        ("sliding_window", 1_024),
        ("max_position_embeddings", 262_144),
        ("vocab_size", 262_144),
        ("intermediate_size", 2_112),
        ("num_experts", 128),
        ("top_k_experts", 8),
        ("moe_intermediate_size", 704),
    ] {
        exact_u64(&text_value, key, expected)?;
    }
    if text.get("model_type").and_then(Value::as_str) != Some("gemma4_text")
        || text.get("dtype").and_then(Value::as_str) != Some("bfloat16")
        || text.get("hidden_activation").and_then(Value::as_str) != Some("gelu_pytorch_tanh")
        || text.get("enable_moe_block").and_then(Value::as_bool) != Some(true)
        || text.get("tie_word_embeddings").and_then(Value::as_bool) != Some(true)
        || text.get("use_cache").and_then(Value::as_bool) != Some(true)
        || text.get("attention_k_eq_v").and_then(Value::as_bool) != Some(true)
        || text.get("rms_norm_eps").and_then(Value::as_f64) != Some(1.0e-6)
    {
        return Err(invalid("text semantic field differs"));
    }
    let layer_types = text
        .get("layer_types")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("layer_types is absent"))?;
    if layer_types.len() != 30 {
        return Err(invalid("layer schedule length differs"));
    }
    let mut reviewed_layers = Vec::with_capacity(30);
    for (layer, value) in layer_types.iter().enumerate() {
        let (expected, layer_type) = if (layer + 1) % 6 == 0 {
            ("full_attention", Gemma4LayerType::FullAttention)
        } else {
            ("sliding_attention", Gemma4LayerType::SlidingAttention)
        };
        if value.as_str() != Some(expected) {
            return Err(invalid(format!("layer schedule differs at {layer}")));
        }
        reviewed_layers.push(layer_type);
    }
    let rope = text
        .get("rope_parameters")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("rope_parameters is absent"))?;
    let sliding = rope
        .get("sliding_attention")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("sliding RoPE contract is absent"))?;
    let full = rope
        .get("full_attention")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("full RoPE contract is absent"))?;
    if sliding.get("rope_type").and_then(Value::as_str) != Some("default")
        || sliding.get("rope_theta").and_then(Value::as_f64) != Some(10_000.0)
        || full.get("rope_type").and_then(Value::as_str) != Some("proportional")
        || full.get("rope_theta").and_then(Value::as_f64) != Some(1_000_000.0)
        || full.get("partial_rotary_factor").and_then(Value::as_f64) != Some(0.25)
    {
        return Err(invalid("RoPE contract differs"));
    }

    let quant = root
        .get("quantization_config")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("quantization_config is absent"))?;
    if quant.get("quant_method").and_then(Value::as_str) != Some("modelopt")
        || quant.get("quant_algo").and_then(Value::as_str) != Some("NVFP4")
    {
        return Err(invalid("ModelOpt NVFP4 identity differs"));
    }
    let producer = quant
        .get("producer")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("quantization producer is absent"))?;
    if producer.get("name").and_then(Value::as_str) != Some("modelopt")
        || producer.get("version").and_then(Value::as_str) != Some("0.43.0rc2.dev91+gc79ebc014")
    {
        return Err(invalid("quantization producer differs"));
    }
    let group = quant
        .get("config_groups")
        .and_then(Value::as_object)
        .and_then(|groups| groups.get("group_0"))
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("NVFP4 config group is absent"))?;
    for name in ["weights", "input_activations"] {
        let plane = group
            .get(name)
            .and_then(Value::as_object)
            .ok_or_else(|| invalid(format!("NVFP4 {name} contract is absent")))?;
        if plane.get("dynamic").and_then(Value::as_bool) != Some(false)
            || plane.get("num_bits").and_then(Value::as_u64) != Some(4)
            || plane.get("type").and_then(Value::as_str) != Some("float")
            || plane.get("group_size").and_then(Value::as_u64) != Some(16)
        {
            return Err(invalid(format!("NVFP4 {name} contract differs")));
        }
    }
    if group
        .get("targets")
        .and_then(Value::as_array)
        .is_none_or(|targets| targets.as_slice() != [Value::String("Linear".to_owned())])
    {
        return Err(invalid("NVFP4 target contract differs"));
    }
    let kv = quant
        .get("kv_cache_scheme")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("FP8 KV cache contract is absent"))?;
    if kv.get("dynamic").and_then(Value::as_bool) != Some(false)
        || kv.get("num_bits").and_then(Value::as_u64) != Some(8)
        || kv.get("type").and_then(Value::as_str) != Some("float")
    {
        return Err(invalid("FP8 KV cache contract differs"));
    }
    let ignore = quant
        .get("ignore")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("quantization exclusions are absent"))?;
    let exclusions: BTreeSet<&str> = ignore
        .iter()
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| invalid("non-string quantization exclusion"))
        })
        .collect::<Result<_, _>>()?;
    let mut expected_exclusions = BTreeSet::from([
        "lm_head".to_owned(),
        "model.embed_vision*".to_owned(),
        "model.vision_tower*".to_owned(),
    ]);
    for layer in 0..30 {
        for component in ["mlp", "router", "self_attn"] {
            expected_exclusions.insert(format!("model.language_model.layers.{layer}.{component}*"));
        }
    }
    if exclusions.len() != ignore.len()
        || exclusions.len() != expected_exclusions.len()
        || exclusions
            .iter()
            .copied()
            .ne(expected_exclusions.iter().map(String::as_str))
    {
        return Err(invalid("quantization exclusions differ"));
    }

    Ok((
        Gemma4MoeConfig {
            hidden_size: 2_816,
            layer_count: 30,
            attention_heads: 16,
            sliding_kv_heads: 8,
            full_kv_heads: 2,
            sliding_head_dim: 256,
            full_head_dim: 512,
            sliding_window: 1_024,
            max_position_embeddings: 262_144,
            vocab_size: 262_144,
            dense_intermediate_size: 2_112,
            expert_count: 128,
            selected_expert_count: 8,
            expert_intermediate_size: 704,
            layer_types: reviewed_layers,
        },
        Gemma4MoeRecipe {
            encoding: QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer,
            block_size: 16,
            value_format: "E2M1",
            block_scale_format: "E4M3FN",
            outer_scale_format: "F32",
            input_scale_format: "F32",
            activation_dynamic: false,
            kv_cache_format: "FP8",
            kv_cache_scale_source: "modelopt-fp8-cast-constant-amax-448",
            kv_cache_dequant_scale_f32_bits: 1.0_f32.to_bits(),
            kv_cache_scale_tensor_count: 0,
            producer: "modelopt@0.43.0rc2.dev91+gc79ebc014".to_owned(),
        },
    ))
}

pub fn validate_gemma4_moe_config(
    bytes: &[u8],
) -> Result<(Gemma4MoeConfig, Gemma4MoeRecipe), Gemma4MoeModelError> {
    if sha256(bytes) != CONFIG_SHA256 {
        return Err(invalid("config SHA-256 differs"));
    }
    let root: Value =
        serde_json::from_slice(bytes).map_err(|error| invalid(format!("config JSON: {error}")))?;
    validate_config_document(&root)
}

fn validate_index_document(
    root: &Value,
    expected_tensor_count: usize,
    expected_shard_counts: &[(&str, usize)],
) -> Result<Gemma4MoeIndex, Gemma4MoeModelError> {
    let total_parameters = root
        .pointer("/metadata/total_parameters")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("index total_parameters is absent"))?;
    let total_size = root
        .pointer("/metadata/total_size")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("index total_size is absent"))?;
    let raw_map = root
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("index weight_map is absent"))?;
    if raw_map.len() != expected_tensor_count {
        return Err(invalid("index tensor count differs"));
    }
    let mut weight_map = BTreeMap::new();
    let mut shard_counts = BTreeMap::<&str, usize>::new();
    for (name, value) in raw_map {
        if name.is_empty() {
            return Err(invalid("index contains an empty tensor name"));
        }
        let file = value
            .as_str()
            .ok_or_else(|| invalid(format!("index shard name is not a string: {name}")))?;
        if !expected_shard_counts
            .iter()
            .any(|(expected, _)| *expected == file)
        {
            return Err(invalid(format!(
                "index references an unknown shard: {file}"
            )));
        }
        *shard_counts.entry(file).or_default() += 1;
        weight_map.insert(name.clone(), file.to_owned());
    }
    if expected_shard_counts
        .iter()
        .any(|(file, count)| shard_counts.get(file).copied() != Some(*count))
    {
        return Err(invalid("index shard distribution differs"));
    }
    Ok(Gemma4MoeIndex {
        total_parameters,
        total_size,
        weight_map,
    })
}

pub fn validate_gemma4_moe_index(bytes: &[u8]) -> Result<Gemma4MoeIndex, Gemma4MoeModelError> {
    if sha256(bytes) != INDEX_SHA256 {
        return Err(invalid("safetensors index SHA-256 differs"));
    }
    let root: Value =
        serde_json::from_slice(bytes).map_err(|error| invalid(format!("index JSON: {error}")))?;
    let index = validate_index_document(
        &root,
        GEMMA4_MOE_TENSOR_COUNT,
        &[
            ("model-00001-of-00002.safetensors", 21_603),
            ("model-00002-of-00002.safetensors", 25_430),
        ],
    )?;
    if index.total_parameters != 14_386_941_232
        || index.total_size != GEMMA4_MOE_ADVERTISED_PAYLOAD_BYTES
    {
        return Err(invalid("index metadata differs"));
    }
    Ok(index)
}

pub fn gemma4_moe_generation_stop_policy() -> GenerationStopPolicyV1 {
    GenerationStopPolicyV1 {
        version: 1,
        stop_token_ids: vec![1, 106, 50],
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

fn read_locked_file(
    path: &Path,
    expected_size: u64,
    expected_digest: &str,
) -> Result<Vec<u8>, Gemma4MoeModelError> {
    let mut file = File::open(path).map_err(|error| Gemma4MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let opened = file.metadata().map_err(|error| Gemma4MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let path_metadata = fs::metadata(path).map_err(|error| Gemma4MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let identity = FileIdentity::from_metadata(&opened);
    if !opened.is_file()
        || identity.size != expected_size
        || FileIdentity::from_metadata(&path_metadata) != identity
    {
        return Err(invalid(format!(
            "locked file size/type differs: {}",
            path.display()
        )));
    }
    let capacity =
        usize::try_from(expected_size).map_err(|_| invalid("support file is too large"))?;
    let mut bytes = Vec::with_capacity(capacity);
    file.read_to_end(&mut bytes)
        .map_err(|error| Gemma4MoeModelError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    let after = file.metadata().map_err(|error| Gemma4MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let path_after = fs::metadata(path).map_err(|error| Gemma4MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    if bytes.len() != capacity
        || sha256(&bytes) != expected_digest
        || FileIdentity::from_metadata(&after) != identity
        || FileIdentity::from_metadata(&path_after) != identity
    {
        return Err(invalid(format!(
            "locked file identity differs: {}",
            path.display()
        )));
    }
    Ok(bytes)
}

fn dtype_width(dtype: &str) -> Option<u64> {
    match dtype {
        "U8" | "F8_E4M3" => Some(1),
        "BF16" => Some(2),
        "F32" => Some(4),
        _ => None,
    }
}

fn validate_tensor_byte_length(
    name: &str,
    entry: &SafeTensorEntry,
) -> Result<(), Gemma4MoeModelError> {
    let width = dtype_width(&entry.dtype)
        .ok_or_else(|| invalid(format!("unsupported tensor dtype: {name}")))?;
    let elements = entry
        .shape
        .iter()
        .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension));
    let expected = elements
        .and_then(|elements| elements.checked_mul(width))
        .ok_or_else(|| invalid(format!("tensor byte length overflow: {name}")))?;
    let actual = entry.data_offsets[1]
        .checked_sub(entry.data_offsets[0])
        .ok_or_else(|| invalid(format!("tensor range underflow: {name}")))?;
    if actual != expected {
        return Err(invalid(format!("tensor byte length differs: {name}")));
    }
    Ok(())
}

fn verify_shard(
    path: &Path,
    identity: Gemma4MoeFileIdentity,
) -> Result<(VerifiedShard, u64, BTreeMap<String, SafeTensorEntry>), Gemma4MoeModelError> {
    let mut file = File::open(path).map_err(|error| Gemma4MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let opened = file.metadata().map_err(|error| Gemma4MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let path_metadata = fs::metadata(path).map_err(|error| Gemma4MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let bound_identity = FileIdentity::from_metadata(&opened);
    if !opened.is_file()
        || opened.len() != identity.size
        || FileIdentity::from_metadata(&path_metadata) != bound_identity
    {
        return Err(invalid(format!(
            "shard size/type differs: {}",
            path.display()
        )));
    }
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 4 * 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| Gemma4MoeModelError::Io {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    if format!("{:x}", hasher.finalize()) != identity.sha256 {
        return Err(invalid(format!(
            "shard SHA-256 differs: {}",
            path.display()
        )));
    }
    let after_hash = file.metadata().map_err(|error| Gemma4MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let path_after_hash = fs::metadata(path).map_err(|error| Gemma4MoeModelError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    if FileIdentity::from_metadata(&after_hash) != bound_identity
        || FileIdentity::from_metadata(&path_after_hash) != bound_identity
    {
        return Err(invalid("shard identity changed during hashing"));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| Gemma4MoeModelError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    let mut length_bytes = [0_u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|error| Gemma4MoeModelError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    let header_length = u64::from_le_bytes(length_bytes);
    if header_length == 0
        || header_length > MAX_HEADER_BYTES
        || 8_u64
            .checked_add(header_length)
            .is_none_or(|end| end > identity.size)
    {
        return Err(invalid("safetensors header length differs"));
    }
    let header_capacity =
        usize::try_from(header_length).map_err(|_| invalid("safetensors header is too large"))?;
    let mut header_bytes = vec![0_u8; header_capacity];
    file.read_exact(&mut header_bytes)
        .map_err(|error| Gemma4MoeModelError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
    let mut raw: BTreeMap<String, Value> = serde_json::from_slice(&header_bytes)
        .map_err(|error| invalid(format!("safetensors header JSON: {error}")))?;
    raw.remove("__metadata__");
    let mut entries = BTreeMap::new();
    let mut ranges = Vec::with_capacity(raw.len());
    for (name, value) in raw {
        let entry: SafeTensorEntry = serde_json::from_value(value)
            .map_err(|error| invalid(format!("tensor metadata {name}: {error}")))?;
        validate_tensor_byte_length(&name, &entry)?;
        ranges.push((entry.data_offsets, name.clone()));
        entries.insert(name, entry);
    }
    ranges.sort_by_key(|(range, _)| range[0]);
    let mut cursor = 0_u64;
    for (range, name) in ranges {
        if range[0] != cursor {
            return Err(invalid(format!("non-contiguous tensor range: {name}")));
        }
        cursor = range[1];
    }
    let data_start = 8 + header_length;
    if data_start.checked_add(cursor) != Some(identity.size) {
        return Err(invalid("safetensors payload extent differs"));
    }
    Ok((
        VerifiedShard {
            file: Arc::new(file),
            identity: bound_identity,
        },
        data_start,
        entries,
    ))
}

fn tensor_plane(
    name: String,
    located: &LocatedEntry,
) -> Result<Gemma4MoeTensorPlane, Gemma4MoeModelError> {
    let start = located
        .data_start
        .checked_add(located.entry.data_offsets[0])
        .ok_or_else(|| invalid("tensor absolute range overflow"))?;
    let end = located
        .data_start
        .checked_add(located.entry.data_offsets[1])
        .ok_or_else(|| invalid("tensor absolute range overflow"))?;
    Ok(Gemma4MoeTensorPlane {
        source_file: located.file.clone(),
        source_name: name,
        dtype: located.entry.dtype.clone(),
        shape: located.entry.shape.clone(),
        absolute_byte_range: [start, end],
    })
}

fn parse_expert_plane_name(
    name: &str,
) -> Result<(u16, u16, Gemma4MoeExpertProjection, ExpertPlaneRole), Gemma4MoeModelError> {
    const PREFIX: &str = "model.language_model.layers.";
    let parts: Vec<_> = name
        .strip_prefix(PREFIX)
        .ok_or_else(|| invalid("expert tensor prefix differs"))?
        .split('.')
        .collect();
    if parts.len() != 5 || parts[1] != "experts" {
        return Err(invalid(format!("malformed expert tensor name: {name}")));
    }
    let layer = parts[0]
        .parse::<u16>()
        .map_err(|_| invalid(format!("invalid expert layer: {name}")))?;
    let expert = parts[2]
        .parse::<u16>()
        .map_err(|_| invalid(format!("invalid expert index: {name}")))?;
    if layer >= 30 || expert >= 128 {
        return Err(invalid(format!(
            "expert coordinate is out of range: {name}"
        )));
    }
    let projection = match parts[3] {
        "gate_proj" => Gemma4MoeExpertProjection::Gate,
        "up_proj" => Gemma4MoeExpertProjection::Up,
        "down_proj" => Gemma4MoeExpertProjection::Down,
        _ => return Err(invalid(format!("invalid expert projection: {name}"))),
    };
    let role = match parts[4] {
        "weight" => ExpertPlaneRole::Value,
        "weight_scale" => ExpertPlaneRole::BlockScale,
        "weight_scale_2" => ExpertPlaneRole::OuterScale,
        "input_scale" => ExpertPlaneRole::InputScale,
        _ => return Err(invalid(format!("invalid expert plane role: {name}"))),
    };
    Ok((layer, expert, projection, role))
}

fn validate_expert_plane_metadata(
    name: &str,
    entry: &SafeTensorEntry,
    projection: Gemma4MoeExpertProjection,
    role: ExpertPlaneRole,
) -> Result<(), Gemma4MoeModelError> {
    let (dtype, shape): (&str, &[u64]) = match role {
        ExpertPlaneRole::Value => ("U8", &projection.value_shape()),
        ExpertPlaneRole::BlockScale => ("F8_E4M3", &projection.block_scale_shape()),
        ExpertPlaneRole::OuterScale | ExpertPlaneRole::InputScale => ("F32", &[]),
    };
    if entry.dtype != dtype || entry.shape != shape {
        return Err(invalid(format!("expert tensor metadata differs: {name}")));
    }
    Ok(())
}

fn take_expert_plane(
    planes: &BTreeMap<(u16, u16, Gemma4MoeExpertProjection, ExpertPlaneRole), Gemma4MoeTensorPlane>,
    layer: u16,
    expert: u16,
    projection: Gemma4MoeExpertProjection,
    role: ExpertPlaneRole,
) -> Result<Gemma4MoeTensorPlane, Gemma4MoeModelError> {
    planes
        .get(&(layer, expert, projection, role))
        .cloned()
        .ok_or_else(|| invalid("required expert tensor plane is absent"))
}

pub fn verify_gemma4_moe_artifact(
    root: impl AsRef<Path>,
) -> Result<VerifiedGemma4Moe, Gemma4MoeModelError> {
    let root = root.as_ref();
    let mut support_files = BTreeMap::new();
    for identity in GEMMA4_MOE_SUPPORT_FILES {
        let bytes = read_locked_file(
            &root.join(identity.file_name),
            identity.size,
            identity.sha256,
        )?;
        support_files.insert(identity.file_name.to_owned(), Arc::<[u8]>::from(bytes));
    }
    let config_bytes = read_locked_file(&root.join("config.json"), 10_289, CONFIG_SHA256)?;
    let (config, recipe) = validate_gemma4_moe_config(&config_bytes)?;
    support_files.insert("config.json".to_owned(), Arc::<[u8]>::from(config_bytes));
    let index_bytes = read_locked_file(
        &root.join("model.safetensors.index.json"),
        4_977_046,
        INDEX_SHA256,
    )?;
    let index = validate_gemma4_moe_index(&index_bytes)?;
    support_files.insert(
        "model.safetensors.index.json".to_owned(),
        Arc::<[u8]>::from(index_bytes),
    );

    let mut located_entries = BTreeMap::new();
    let mut shards = BTreeMap::new();
    for identity in GEMMA4_MOE_SHARDS {
        let (shard, data_start, entries) = verify_shard(&root.join(identity.file_name), identity)?;
        for (name, entry) in entries {
            if index.source_file(&name) != Some(identity.file_name) {
                return Err(invalid(format!(
                    "index/header shard mapping differs: {name}"
                )));
            }
            let located = LocatedEntry {
                file: identity.file_name.to_owned(),
                data_start,
                entry,
            };
            if located_entries.insert(name.clone(), located).is_some() {
                return Err(invalid(format!("duplicate tensor across shards: {name}")));
            }
        }
        shards.insert(identity.file_name.to_owned(), shard);
    }
    if located_entries.len() != GEMMA4_MOE_TENSOR_COUNT
        || located_entries
            .keys()
            .any(|name| index.source_file(name).is_none())
    {
        return Err(invalid("header/index tensor set differs"));
    }

    let mut catalog = Sha256::new();
    let mut text_catalog = Sha256::new();
    let mut all_planes = Vec::with_capacity(GEMMA4_MOE_TENSOR_COUNT);
    let mut text_planes = Vec::with_capacity(GEMMA4_MOE_TEXT_TENSOR_COUNT);
    let mut expert_planes = BTreeMap::new();
    let mut text_count = 0_usize;
    let mut vision_count = 0_usize;
    let mut text_bytes = 0_u64;
    let mut total_bytes = 0_u64;
    for (name, located) in &located_entries {
        if name.ends_with(".k_scale") || name.ends_with(".v_scale") {
            return Err(invalid(format!(
                "unexpected serialized FP8 KV scale tensor: {name}"
            )));
        }
        let row = serde_json::to_string(&(
            name,
            &located.file,
            &located.entry.dtype,
            &located.entry.shape,
            &located.entry.data_offsets,
        ))
        .map_err(|error| invalid(format!("catalog serialization: {error}")))?;
        catalog.update(row.as_bytes());
        catalog.update(b"\n");
        let bytes = located.entry.data_offsets[1] - located.entry.data_offsets[0];
        total_bytes = total_bytes
            .checked_add(bytes)
            .ok_or_else(|| invalid("catalog byte count overflow"))?;
        let plane = tensor_plane(name.clone(), located)?;
        all_planes.push(plane.clone());
        if name.starts_with("model.language_model.") {
            text_count += 1;
            text_bytes = text_bytes
                .checked_add(bytes)
                .ok_or_else(|| invalid("text byte count overflow"))?;
            text_catalog.update(row.as_bytes());
            text_catalog.update(b"\n");
            text_planes.push(plane.clone());
            if name.contains(".experts.") {
                let (layer, expert, projection, role) = parse_expert_plane_name(name)?;
                validate_expert_plane_metadata(name, &located.entry, projection, role)?;
                if expert_planes
                    .insert((layer, expert, projection, role), plane)
                    .is_some()
                {
                    return Err(invalid(format!("duplicate expert plane: {name}")));
                }
            }
        } else if name.starts_with("model.embed_vision.") || name.starts_with("model.vision_tower.")
        {
            vision_count += 1;
        } else {
            return Err(invalid(format!("unknown tensor component: {name}")));
        }
    }
    if format!("{:x}", catalog.finalize()) != CATALOG_SHA256
        || format!("{:x}", text_catalog.finalize()) != TEXT_CATALOG_SHA256
        || text_count != GEMMA4_MOE_TEXT_TENSOR_COUNT
        || vision_count != GEMMA4_MOE_VISION_TENSOR_COUNT
        || text_bytes != GEMMA4_MOE_TEXT_RESIDENT_BYTES
        || total_bytes != GEMMA4_MOE_ADVERTISED_PAYLOAD_BYTES
        || expert_planes.len() != GEMMA4_MOE_EXPERT_PROJECTION_COUNT * 4
    {
        return Err(invalid("safetensors catalog contract differs"));
    }

    let mut experts = BTreeMap::new();
    for layer in 0..30_u16 {
        for expert in 0..128_u16 {
            for projection in [
                Gemma4MoeExpertProjection::Gate,
                Gemma4MoeExpertProjection::Up,
                Gemma4MoeExpertProjection::Down,
            ] {
                experts.insert(
                    (layer, expert, projection),
                    Gemma4MoeExpertTensor {
                        layer,
                        expert,
                        projection,
                        logical_shape: projection.logical_shape(),
                        encoding: QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer,
                        value: take_expert_plane(
                            &expert_planes,
                            layer,
                            expert,
                            projection,
                            ExpertPlaneRole::Value,
                        )?,
                        block_scale: take_expert_plane(
                            &expert_planes,
                            layer,
                            expert,
                            projection,
                            ExpertPlaneRole::BlockScale,
                        )?,
                        outer_scale: take_expert_plane(
                            &expert_planes,
                            layer,
                            expert,
                            projection,
                            ExpertPlaneRole::OuterScale,
                        )?,
                        input_scale: take_expert_plane(
                            &expert_planes,
                            layer,
                            expert,
                            projection,
                            ExpertPlaneRole::InputScale,
                        )?,
                    },
                );
            }
        }
    }
    if experts.len() != GEMMA4_MOE_EXPERT_PROJECTION_COUNT {
        return Err(invalid("expert projection coverage differs"));
    }

    Ok(VerifiedGemma4Moe {
        root: root.to_path_buf(),
        config,
        recipe,
        all_planes,
        text_planes,
        experts,
        support_files,
        shards,
    })
}

pub fn verify_gguf_gemma4_moe(
    verified: VerifiedDerivedGguf,
) -> Result<VerifiedGgufGemma4Moe, Gemma4MoeModelError> {
    if verified.gguf.architecture() != "gemma4moe"
        || verified.lock.semantic_model_id != format!("gemma4moe:{GEMMA4_MOE_MODEL_FINGERPRINT}")
        || verified.lock.source_lock_fingerprints.as_slice() != [GEMMA4_MOE_MODEL_FINGERPRINT]
    {
        return Err(invalid("GGUF is not the reviewed Gemma 4 MoE identity"));
    }
    validate_gemma4_moe_metadata(&verified.gguf)?;
    let (config, recipe) = validate_gemma4_moe_frontend_assets(&verified.gguf)?;
    validate_gguf_identity_recipe(&verified.gguf, &recipe)?;
    let (direct_planes, experts) = validate_gemma4_moe_gguf_catalog(&verified.gguf, &config)?;
    Ok(VerifiedGgufGemma4Moe {
        gguf: verified.gguf,
        file_sha256: verified.lock.output.sha256,
        config,
        direct_planes,
        experts,
    })
}

fn validate_gemma4_moe_metadata(gguf: &VerifiedGguf) -> Result<(), Gemma4MoeModelError> {
    let expected_keys: BTreeSet<&str> = [
        "general.architecture",
        "general.alignment",
        "general.name",
        "general.source.url",
        "general.license",
        "sllm.source.artifact.fingerprint",
        "sllm.source.semantic.repository",
        "sllm.source.semantic.revision",
        "sllm.source.recipe.producer",
        "sllm.kv.fp8.scheme",
        "sllm.kv.fp8.implicit_decode_scale_bf16",
        "gemma4moe.block_count",
        "gemma4moe.context_length",
        "gemma4moe.embedding_length",
        "gemma4moe.expert_count",
        "gemma4moe.expert_used_count",
        "gemma4moe.expert_feed_forward_length",
        "sllm.extension.version",
        "sllm.tensor_recipe",
        "sllm.tensor_recipe.sha256",
        "sllm.frontend.config_json",
        "sllm.frontend.config_json.sha256",
        "sllm.frontend.tokenizer_json",
        "sllm.frontend.tokenizer_json.sha256",
        "sllm.frontend.tokenizer_config_json",
        "sllm.frontend.tokenizer_config_json.sha256",
        "sllm.frontend.preprocessor_config_json",
        "sllm.frontend.preprocessor_config_json.sha256",
        "sllm.frontend.generation_config_json",
        "sllm.frontend.generation_config_json.sha256",
        "sllm.source.hf_quant_config_json",
        "sllm.source.hf_quant_config_json.sha256",
        "tokenizer.chat_template",
        "tokenizer.chat_template.sha256",
    ]
    .into_iter()
    .collect();
    let observed_keys: BTreeSet<&str> = gguf.metadata().keys().map(String::as_str).collect();
    if observed_keys != expected_keys {
        return Err(invalid("GGUF Gemma 4 MoE metadata key set differs"));
    }
    for (key, expected) in [
        ("general.architecture", "gemma4moe"),
        ("general.name", GEMMA4_MOE_REPOSITORY),
        ("general.source.url", GEMMA4_MOE_REVISION),
        ("general.license", GEMMA4_MOE_LICENSE),
        (
            "sllm.source.artifact.fingerprint",
            GEMMA4_MOE_MODEL_FINGERPRINT,
        ),
        (
            "sllm.source.semantic.repository",
            GEMMA4_MOE_SEMANTIC_REPOSITORY,
        ),
        (
            "sllm.source.semantic.revision",
            GEMMA4_MOE_SEMANTIC_REVISION,
        ),
        (
            "sllm.source.recipe.producer",
            "modelopt@0.43.0rc2.dev91+gc79ebc014",
        ),
        ("sllm.kv.fp8.scheme", "modelopt-fp8-cast-constant-amax-448"),
    ] {
        if gguf.metadata_value(key) != Some(&GgufValue::String(expected.to_owned())) {
            return Err(invalid(format!("GGUF Gemma 4 MoE metadata differs: {key}")));
        }
    }
    for (key, expected) in [
        ("gemma4moe.block_count", 30),
        ("gemma4moe.context_length", 262_144),
        ("gemma4moe.embedding_length", 2_816),
        ("gemma4moe.expert_count", 128),
        ("gemma4moe.expert_used_count", 8),
        ("gemma4moe.expert_feed_forward_length", 704),
    ] {
        if gguf.metadata_value(key) != Some(&GgufValue::U32(expected)) {
            return Err(invalid(format!("GGUF Gemma 4 MoE metadata differs: {key}")));
        }
    }
    if gguf.metadata_value("general.alignment") != Some(&GgufValue::U32(32))
        || gguf.metadata_value("sllm.kv.fp8.implicit_decode_scale_bf16")
            != Some(&GgufValue::U16(0x3f80))
    {
        return Err(invalid("GGUF Gemma 4 MoE scalar metadata differs"));
    }
    Ok(())
}

fn validate_gemma4_moe_frontend_assets(
    gguf: &VerifiedGguf,
) -> Result<(Gemma4MoeConfig, Gemma4MoeRecipe), Gemma4MoeModelError> {
    let config_bytes = exact_gguf_asset(gguf, "config.json", 10_289, CONFIG_SHA256)?;
    let (config, recipe) = validate_gemma4_moe_config(&config_bytes)?;
    for (asset_name, source_name) in [
        ("tokenizer.json", "tokenizer.json"),
        ("tokenizer_config.json", "tokenizer_config.json"),
        ("preprocessor_config.json", "processor_config.json"),
        ("generation_config.json", "generation_config.json"),
        ("hf_quant_config.json", "hf_quant_config.json"),
    ] {
        let identity = GEMMA4_MOE_SUPPORT_FILES
            .iter()
            .find(|identity| identity.file_name == source_name)
            .ok_or_else(|| invalid("reviewed support-file identity is absent"))?;
        exact_gguf_asset(gguf, asset_name, identity.size, identity.sha256)?;
    }
    let template = match gguf.metadata_value("tokenizer.chat_template") {
        Some(GgufValue::String(template)) => template.as_bytes(),
        _ => return Err(invalid("GGUF Gemma 4 MoE chat template is absent")),
    };
    let identity = GEMMA4_MOE_SUPPORT_FILES
        .iter()
        .find(|identity| identity.file_name == "chat_template.jinja")
        .expect("reviewed chat template identity");
    if template.len() as u64 != identity.size
        || sha256(template) != identity.sha256
        || gguf.metadata_value("tokenizer.chat_template.sha256")
            != Some(&GgufValue::String(format!("sha256:{}", identity.sha256)))
    {
        return Err(invalid("GGUF Gemma 4 MoE chat template identity differs"));
    }
    Ok((config, recipe))
}

fn exact_gguf_asset(
    gguf: &VerifiedGguf,
    name: &str,
    expected_size: u64,
    expected_sha256: &str,
) -> Result<Vec<u8>, Gemma4MoeModelError> {
    let bytes = gguf
        .frontend_asset(name)
        .ok_or_else(|| invalid(format!("GGUF Gemma 4 MoE asset is absent: {name}")))?;
    if bytes.len() as u64 != expected_size || sha256(bytes) != expected_sha256 {
        return Err(invalid(format!(
            "GGUF Gemma 4 MoE asset identity differs: {name}"
        )));
    }
    Ok(bytes.to_vec())
}

fn validate_gguf_identity_recipe(
    gguf: &VerifiedGguf,
    source_recipe: &Gemma4MoeRecipe,
) -> Result<(), Gemma4MoeModelError> {
    if source_recipe.encoding != QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer
        || source_recipe.block_size != 16
        || source_recipe.kv_cache_scale_source != "modelopt-fp8-cast-constant-amax-448"
        || source_recipe.kv_cache_dequant_scale_f32_bits != 1.0_f32.to_bits()
        || source_recipe.kv_cache_scale_tensor_count != 0
    {
        return Err(invalid("GGUF Gemma 4 MoE source recipe differs"));
    }
    let extension = gguf
        .extension()
        .ok_or_else(|| invalid("GGUF Gemma 4 MoE tensor recipe is absent"))?;
    let recipe = &extension.recipe;
    if recipe.semantic_model_id != format!("gemma4moe:{GEMMA4_MOE_MODEL_FINGERPRINT}")
        || recipe.source_lock_fingerprints.as_slice() != [GEMMA4_MOE_MODEL_FINGERPRINT]
        || !recipe.logical_shapes.is_empty()
        || recipe.bindings.len() != GEMMA4_MOE_EXPERT_PROJECTION_COUNT
        || recipe.static_fp8_kv.len() != 30
    {
        return Err(invalid("GGUF Gemma 4 MoE tensor recipe identity differs"));
    }
    for (layer, binding) in recipe.static_fp8_kv.iter().enumerate() {
        if binding.layer != layer as u32
            || binding.key_decode_scale_bf16 != 0x3f80
            || binding.value_decode_scale_bf16 != 0x3f80
        {
            return Err(invalid("GGUF Gemma 4 MoE static FP8 KV recipe differs"));
        }
    }
    Ok(())
}

fn validate_gemma4_moe_gguf_catalog(
    gguf: &VerifiedGguf,
    config: &Gemma4MoeConfig,
) -> Result<GgufGemma4MoeCatalog, Gemma4MoeModelError> {
    let extension = gguf.extension().expect("recipe was validated");
    let known: BTreeSet<&str> = extension
        .recipe
        .known_unconsumed_tensors
        .iter()
        .map(String::as_str)
        .collect();
    if known.len() != GEMMA4_MOE_VISION_TENSOR_COUNT
        || known.iter().any(|name| {
            !name.starts_with("model.embed_vision.") && !name.starts_with("model.vision_tower.")
        })
    {
        return Err(invalid("GGUF Gemma 4 MoE known vision tensor set differs"));
    }
    let mut experts = BTreeMap::new();
    let mut expert_physical_names = BTreeSet::new();
    for binding in &extension.recipe.bindings {
        if binding.encoding != GgufRecipeEncoding::Nvfp4E2m1Block16E4m3fnF32Outer
            || binding.value_tensor != binding.logical_tensor
            || binding.role != "routed-expert-projection"
            || binding.scope != GgufTensorScope::Consumed
            || binding.scales.len() != 2
            || binding.scales[0].role != GgufScaleRole::Outer
            || binding.scales[1].role != GgufScaleRole::Input
        {
            return Err(invalid("GGUF Gemma 4 MoE expert recipe differs"));
        }
        let (layer, expert, projection) =
            parse_gguf_gemma4_moe_expert_name(&binding.logical_tensor)?;
        let expected_outer = format!(
            "model.language_model.layers.{layer}.experts.{expert}.{}.weight_scale_2",
            projection.source_stem()
        );
        let expected_input = format!(
            "model.language_model.layers.{layer}.experts.{expert}.{}.input_scale",
            projection.source_stem()
        );
        if binding.logical_shape.as_slice() != projection.logical_shape()
            || binding.scales[0].tensor != expected_outer
            || binding.scales[1].tensor != expected_input
        {
            return Err(invalid("GGUF Gemma 4 MoE expert scale binding differs"));
        }
        let value = gguf
            .tensor(&binding.value_tensor)
            .ok_or_else(|| invalid("GGUF Gemma 4 MoE expert value is absent"))?;
        let outer = gguf
            .tensor(&expected_outer)
            .ok_or_else(|| invalid("GGUF Gemma 4 MoE expert outer scale is absent"))?;
        let input = gguf
            .tensor(&expected_input)
            .ok_or_else(|| invalid("GGUF Gemma 4 MoE expert input scale is absent"))?;
        let [rows, columns] = projection.logical_shape();
        if value.tensor_type != GgufTensorType::Nvfp4
            || value.dimensions.as_slice() != [columns, rows]
            || outer.tensor_type != GgufTensorType::F32
            || outer.dimensions.as_slice() != [1]
            || input.tensor_type != GgufTensorType::F32
            || input.dimensions.as_slice() != [1]
            || experts
                .insert(
                    (layer, expert, projection),
                    GgufGemma4MoeExpertBinding {
                        value_tensor: binding.value_tensor.clone(),
                        outer_scale_tensor: expected_outer.clone(),
                        input_scale_tensor: expected_input.clone(),
                    },
                )
                .is_some()
            || !expert_physical_names.insert(binding.value_tensor.as_str())
            || !expert_physical_names.insert(binding.scales[0].tensor.as_str())
            || !expert_physical_names.insert(binding.scales[1].tensor.as_str())
        {
            return Err(invalid("GGUF Gemma 4 MoE expert binding is not one-to-one"));
        }
    }
    if experts.len() != GEMMA4_MOE_EXPERT_PROJECTION_COUNT
        || expert_physical_names.len() != GEMMA4_MOE_EXPERT_PROJECTION_COUNT * 3
    {
        return Err(invalid("GGUF Gemma 4 MoE expert coverage differs"));
    }

    let expected_catalog = crate::gemma4_moe_graph::expected_gemma4_moe_text_tensor_catalog(config)
        .map_err(|error| invalid(error.to_string()))?;
    let mut expected_direct: BTreeMap<&str, &crate::Gemma4MoeGraphTensorSpec> = expected_catalog
        .iter()
        .filter(|spec| spec.encoding == crate::Gemma4MoeGraphTensorEncoding::Plain)
        .map(|spec| (spec.name.as_str(), spec))
        .collect();
    let source_file = gguf.path().display().to_string();
    let mut direct_planes = Vec::with_capacity(expected_direct.len());
    let mut vision_bytes = 0_u64;
    let mut payload_bytes = 0_u64;
    for tensor in gguf.tensors() {
        payload_bytes = payload_bytes
            .checked_add(tensor.byte_length())
            .ok_or_else(|| invalid("GGUF Gemma 4 MoE payload byte count overflows"))?;
        if expert_physical_names.contains(tensor.name.as_str()) {
            continue;
        }
        if known.contains(tensor.name.as_str()) {
            if tensor.tensor_type != GgufTensorType::Bf16 {
                return Err(invalid("GGUF Gemma 4 MoE vision tensor type differs"));
            }
            vision_bytes = vision_bytes
                .checked_add(tensor.byte_length())
                .ok_or_else(|| invalid("GGUF Gemma 4 MoE vision byte count overflows"))?;
            continue;
        }
        let expected = expected_direct
            .remove(tensor.name.as_str())
            .ok_or_else(|| {
                invalid(format!(
                    "unexpected GGUF Gemma 4 MoE tensor: {}",
                    tensor.name
                ))
            })?;
        let expected_type = match expected.dtype {
            crate::Gemma4MoeGraphTensorDtype::Bf16 => GgufTensorType::Bf16,
            crate::Gemma4MoeGraphTensorDtype::F32 => GgufTensorType::F32,
            _ => return Err(invalid("GGUF Gemma 4 MoE direct catalog encoding differs")),
        };
        let mut shape = tensor.dimensions.clone();
        shape.reverse();
        if tensor.tensor_type != expected_type || shape != expected.stored_shape {
            return Err(invalid(format!(
                "GGUF Gemma 4 MoE direct tensor metadata differs: {}",
                tensor.name
            )));
        }
        direct_planes.push(Gemma4MoeTensorPlane {
            source_file: source_file.clone(),
            source_name: tensor.name.clone(),
            dtype: match tensor.tensor_type {
                GgufTensorType::Bf16 => "BF16",
                GgufTensorType::F32 => "F32",
                _ => unreachable!("direct type was validated"),
            }
            .to_owned(),
            shape,
            absolute_byte_range: tensor.absolute_range,
        });
    }
    let expected_vision_bytes = GEMMA4_MOE_ADVERTISED_PAYLOAD_BYTES
        .checked_sub(GEMMA4_MOE_TEXT_RESIDENT_BYTES)
        .expect("reviewed payload partition");
    if !expected_direct.is_empty()
        || direct_planes.len() + GEMMA4_MOE_EXPERT_PROJECTION_COUNT * 4
            != GEMMA4_MOE_TEXT_TENSOR_COUNT
        || vision_bytes != expected_vision_bytes
        || payload_bytes != GEMMA4_MOE_ADVERTISED_PAYLOAD_BYTES
        || gguf.tensors().len() != GEMMA4_MOE_TENSOR_COUNT - GEMMA4_MOE_EXPERT_PROJECTION_COUNT
    {
        return Err(invalid("GGUF Gemma 4 MoE physical/logical catalog differs"));
    }
    direct_planes.sort_by(|left, right| left.source_name.cmp(&right.source_name));
    Ok((direct_planes, experts))
}

fn parse_gguf_gemma4_moe_expert_name(
    name: &str,
) -> Result<(u16, u16, Gemma4MoeExpertProjection), Gemma4MoeModelError> {
    const PREFIX: &str = "model.language_model.layers.";
    let parts: Vec<_> = name
        .strip_prefix(PREFIX)
        .ok_or_else(|| invalid("GGUF Gemma 4 MoE expert prefix differs"))?
        .split('.')
        .collect();
    if parts.len() != 5 || parts[1] != "experts" || parts[4] != "weight" {
        return Err(invalid("GGUF Gemma 4 MoE expert name is malformed"));
    }
    let layer = parts[0]
        .parse::<u16>()
        .map_err(|_| invalid("GGUF Gemma 4 MoE layer is invalid"))?;
    let expert = parts[2]
        .parse::<u16>()
        .map_err(|_| invalid("GGUF Gemma 4 MoE expert is invalid"))?;
    let projection = match parts[3] {
        "gate_proj" => Gemma4MoeExpertProjection::Gate,
        "up_proj" => Gemma4MoeExpertProjection::Up,
        "down_proj" => Gemma4MoeExpertProjection::Down,
        _ => return Err(invalid("GGUF Gemma 4 MoE projection is invalid")),
    };
    if layer >= 30 || expert >= 128 {
        return Err(invalid(
            "GGUF Gemma 4 MoE expert coordinate is out of range",
        ));
    }
    Ok((layer, expert, projection))
}

pub fn build_gemma4_moe_weight_load_plan(
    artifact: &VerifiedGemma4Moe,
) -> Result<WeightLoadPlan, Gemma4MoeModelError> {
    let direct_planes = validate_source_gemma4_moe_catalog(artifact)?;
    let mut entries = Vec::with_capacity(direct_planes.len() + 30);
    let mut destination = 0_u64;
    let mut consumers = BTreeSet::new();
    for plane in &direct_planes {
        let consumer = classify_gemma4_moe_direct_plane(plane, artifact.config())?;
        if !consumers.insert(consumer) {
            return Err(invalid("source Gemma 4 MoE direct consumer is duplicated"));
        }
        // The router's per-expert BF16 scale is verified and classified as a
        // direct source tensor, but its only resident copy is the final 256
        // bytes of the synthetic layer blob. Keeping a second allocation here
        // would silently violate the runtime's no-double-residency contract.
        if consumer.role == WeightConsumer::Gemma4MoeRouterPerExpertScale {
            continue;
        }
        let dtype = parse_gemma4_moe_tensor_dtype(&plane.dtype)?;
        let byte_length = u64::try_from(plane_byte_length(plane)?)
            .map_err(|_| invalid("source direct byte length exceeds u64"))?;
        let locked = artifact
            .locked_shard(&plane.source_file)
            .ok_or_else(|| invalid("source direct tensor shard is not locked"))?;
        entries.push(WeightLoadEntry {
            tensor_name: plane.source_name.clone(),
            classification: WeightClassification::Required,
            consumer: Some(consumer),
            dtype,
            shape: plane.shape.clone(),
            source_file: plane.source_file.clone(),
            locked_file_size: locked.size,
            locked_file_sha256: locked.sha256.to_owned(),
            source_range: plane.absolute_byte_range,
            destination_start: Some(destination),
            chunks: gemma4_moe_load_chunks(plane.absolute_byte_range[0], destination, byte_length)?,
        });
        destination = destination
            .checked_add(byte_length)
            .ok_or_else(|| invalid("source direct destination overflows"))?;
    }
    append_gemma4_moe_layer_blob_entries(
        &mut entries,
        &mut destination,
        false,
        "sllm://gemma4-moe/source",
        GEMMA4_MOE_TEXT_RESIDENT_BYTES,
        TEXT_CATALOG_SHA256,
    )?;
    entries.sort_by(|left, right| left.tensor_name.cmp(&right.tensor_name));
    crate::weights::WeightLoadPlan::from_verified_entries(
        crate::weights::VerifiedWeightPlanMetadata {
            schema_version: "gemma4-moe-nvfp4-load-plan-v1".to_owned(),
            repo_id: GEMMA4_MOE_REPOSITORY.to_owned(),
            resolved_revision: GEMMA4_MOE_REVISION.to_owned(),
            lock_fingerprint: GEMMA4_MOE_MODEL_FINGERPRINT.to_owned(),
            tied_embeddings: true,
            chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
            total_destination_bytes: destination,
        },
        entries,
    )
    .map_err(|error| invalid(error.to_string()))
}

pub fn build_gguf_gemma4_moe_weight_load_plan(
    source: &VerifiedGgufGemma4Moe,
) -> Result<WeightLoadPlan, Gemma4MoeModelError> {
    let source_file = source.gguf.path().display().to_string();
    let source_sha = source
        .file_sha256
        .strip_prefix("sha256:")
        .ok_or_else(|| invalid("GGUF Gemma 4 MoE SHA-256 prefix differs"))?;
    let mut entries = Vec::with_capacity(source.direct_planes.len() + 30);
    let mut destination = 0_u64;
    let mut consumers = BTreeSet::new();
    for plane in &source.direct_planes {
        let consumer = classify_gemma4_moe_direct_plane(plane, source.config())?;
        if !consumers.insert(consumer) {
            return Err(invalid("GGUF Gemma 4 MoE direct consumer is duplicated"));
        }
        if consumer.role == WeightConsumer::Gemma4MoeRouterPerExpertScale {
            continue;
        }
        let dtype = parse_gemma4_moe_tensor_dtype(&plane.dtype)?;
        let byte_length = u64::try_from(plane_byte_length(plane)?)
            .map_err(|_| invalid("GGUF direct byte length exceeds u64"))?;
        entries.push(WeightLoadEntry {
            tensor_name: plane.source_name.clone(),
            classification: WeightClassification::Required,
            consumer: Some(consumer),
            dtype,
            shape: plane.shape.clone(),
            source_file: source_file.clone(),
            locked_file_size: source.gguf.file_size(),
            locked_file_sha256: source_sha.to_owned(),
            source_range: plane.absolute_byte_range,
            destination_start: Some(destination),
            chunks: gemma4_moe_load_chunks(plane.absolute_byte_range[0], destination, byte_length)?,
        });
        destination = destination
            .checked_add(byte_length)
            .ok_or_else(|| invalid("GGUF direct destination overflows"))?;
    }
    append_gemma4_moe_layer_blob_entries(
        &mut entries,
        &mut destination,
        true,
        &source_file,
        source.gguf.file_size(),
        source_sha,
    )?;
    entries.sort_by(|left, right| left.tensor_name.cmp(&right.tensor_name));
    crate::weights::WeightLoadPlan::from_verified_entries(
        crate::weights::VerifiedWeightPlanMetadata {
            schema_version: "gemma4-moe-nvfp4-gguf-load-plan-v1".to_owned(),
            repo_id: GEMMA4_MOE_REPOSITORY.to_owned(),
            resolved_revision: GEMMA4_MOE_REVISION.to_owned(),
            lock_fingerprint: GEMMA4_MOE_MODEL_FINGERPRINT.to_owned(),
            tied_embeddings: true,
            chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
            total_destination_bytes: destination,
        },
        entries,
    )
    .map_err(|error| invalid(error.to_string()))
}

fn validate_source_gemma4_moe_catalog(
    artifact: &VerifiedGemma4Moe,
) -> Result<Vec<Gemma4MoeTensorPlane>, Gemma4MoeModelError> {
    let expected =
        crate::gemma4_moe_graph::expected_gemma4_moe_text_tensor_catalog(artifact.config())
            .map_err(|error| invalid(error.to_string()))?;
    let observed: BTreeMap<&str, &Gemma4MoeTensorPlane> = artifact
        .text_planes()
        .iter()
        .map(|plane| (plane.source_name.as_str(), plane))
        .collect();
    if observed.len() != expected.len() || expected.len() != GEMMA4_MOE_TEXT_TENSOR_COUNT {
        return Err(invalid("source Gemma 4 MoE text catalog count differs"));
    }
    let mut direct = Vec::new();
    for spec in expected {
        let plane = observed
            .get(spec.name.as_str())
            .copied()
            .ok_or_else(|| invalid(format!("source tensor is absent: {}", spec.name)))?;
        let expected_dtype = match spec.dtype {
            crate::Gemma4MoeGraphTensorDtype::Bf16 => "BF16",
            crate::Gemma4MoeGraphTensorDtype::F32 => "F32",
            crate::Gemma4MoeGraphTensorDtype::Fp8E4M3 => "F8_E4M3",
            crate::Gemma4MoeGraphTensorDtype::U8 => "U8",
        };
        if plane.dtype != expected_dtype || plane.shape != spec.stored_shape {
            return Err(invalid(format!(
                "source tensor metadata differs: {}",
                spec.name
            )));
        }
        if spec.encoding == crate::Gemma4MoeGraphTensorEncoding::Plain {
            direct.push(plane.clone());
        }
    }
    direct.sort_by(|left, right| left.source_name.cmp(&right.source_name));
    Ok(direct)
}

fn classify_gemma4_moe_direct_plane(
    plane: &Gemma4MoeTensorPlane,
    config: &Gemma4MoeConfig,
) -> Result<WeightConsumerKey, Gemma4MoeModelError> {
    let top = match plane.source_name.as_str() {
        "model.language_model.embed_tokens.weight" => Some(WeightConsumer::EmbeddingAndTiedOutput),
        "model.language_model.norm.weight" => Some(WeightConsumer::FinalNorm),
        _ => None,
    };
    if let Some(role) = top {
        return Ok(WeightConsumerKey { layer: None, role });
    }
    const PREFIX: &str = "model.language_model.layers.";
    let remainder = plane.source_name.strip_prefix(PREFIX).ok_or_else(|| {
        invalid(format!(
            "unknown Gemma 4 MoE direct tensor: {}",
            plane.source_name
        ))
    })?;
    let (layer_text, suffix) = remainder
        .split_once('.')
        .ok_or_else(|| invalid("malformed Gemma 4 MoE layer tensor"))?;
    let layer = layer_text
        .parse::<u32>()
        .map_err(|_| invalid("malformed Gemma 4 MoE layer index"))?;
    let layer_type = config
        .layer_types
        .get(layer as usize)
        .ok_or_else(|| invalid("Gemma 4 MoE direct layer is out of range"))?;
    let role = match suffix {
        "input_layernorm.weight" => WeightConsumer::InputNorm,
        "post_attention_layernorm.weight" => WeightConsumer::PostAttentionNorm,
        "pre_feedforward_layernorm.weight" => WeightConsumer::PreFeedforwardNorm,
        "pre_feedforward_layernorm_2.weight" => WeightConsumer::Gemma4MoePreFeedforwardNorm2,
        "post_feedforward_layernorm.weight" => WeightConsumer::PostFeedforwardNorm,
        "post_feedforward_layernorm_1.weight" => WeightConsumer::Gemma4MoePostFeedforwardNorm1,
        "post_feedforward_layernorm_2.weight" => WeightConsumer::Gemma4MoePostFeedforwardNorm2,
        "layer_scalar" => WeightConsumer::LayerScalar,
        "self_attn.q_proj.weight" => WeightConsumer::AttentionQ,
        "self_attn.k_proj.weight" if *layer_type == Gemma4LayerType::FullAttention => {
            WeightConsumer::AttentionKAndV
        }
        "self_attn.k_proj.weight" => WeightConsumer::AttentionK,
        "self_attn.v_proj.weight" if *layer_type == Gemma4LayerType::SlidingAttention => {
            WeightConsumer::AttentionV
        }
        "self_attn.o_proj.weight" => WeightConsumer::AttentionO,
        "self_attn.q_norm.weight" => WeightConsumer::AttentionQNorm,
        "self_attn.k_norm.weight" => WeightConsumer::AttentionKNorm,
        "mlp.gate_proj.weight" => WeightConsumer::MlpGate,
        "mlp.up_proj.weight" => WeightConsumer::MlpUp,
        "mlp.down_proj.weight" => WeightConsumer::MlpDown,
        "router.proj.weight" => WeightConsumer::MoeRouter,
        "router.scale" => WeightConsumer::Gemma4MoeRouterScale,
        "router.per_expert_scale" => WeightConsumer::Gemma4MoeRouterPerExpertScale,
        _ => {
            return Err(invalid(format!(
                "unknown Gemma 4 MoE direct tensor: {}",
                plane.source_name
            )));
        }
    };
    Ok(WeightConsumerKey {
        layer: Some(u64::from(layer)),
        role,
    })
}

fn parse_gemma4_moe_tensor_dtype(dtype: &str) -> Result<TensorDType, Gemma4MoeModelError> {
    match dtype {
        "BF16" => Ok(TensorDType::Bf16),
        "F32" => Ok(TensorDType::F32),
        _ => Err(invalid(format!(
            "unsupported Gemma 4 MoE direct dtype: {dtype}"
        ))),
    }
}

fn append_gemma4_moe_layer_blob_entries(
    entries: &mut Vec<WeightLoadEntry>,
    destination: &mut u64,
    gguf: bool,
    source_file: &str,
    locked_file_size: u64,
    locked_file_sha256: &str,
) -> Result<(), Gemma4MoeModelError> {
    for layer in 0..30_u32 {
        let name = gemma4_moe_layer_blob_name(layer);
        let chunks = if gguf {
            Vec::new()
        } else {
            gemma4_moe_load_chunks(0, *destination, GEMMA4_MOE_LAYER_BLOB_BYTES)?
        };
        entries.push(WeightLoadEntry {
            tensor_name: name,
            classification: WeightClassification::Required,
            consumer: Some(WeightConsumerKey {
                layer: Some(u64::from(layer)),
                role: WeightConsumer::Gemma4MoeLayerBlob,
            }),
            dtype: TensorDType::U8,
            shape: vec![GEMMA4_MOE_LAYER_BLOB_BYTES],
            source_file: if gguf {
                source_file.to_owned()
            } else {
                format!("{source_file}/layer/{layer}")
            },
            locked_file_size,
            locked_file_sha256: locked_file_sha256.to_owned(),
            source_range: [0, GEMMA4_MOE_LAYER_BLOB_BYTES],
            destination_start: Some(*destination),
            chunks,
        });
        *destination = destination
            .checked_add(GEMMA4_MOE_LAYER_BLOB_BYTES)
            .ok_or_else(|| invalid("Gemma 4 MoE layer blob destination overflows"))?;
    }
    Ok(())
}

fn gemma4_moe_load_chunks(
    source_start: u64,
    destination_start: u64,
    byte_length: u64,
) -> Result<Vec<WeightLoadChunk>, Gemma4MoeModelError> {
    let mut chunks = Vec::new();
    let mut relative = 0_u64;
    while relative < byte_length {
        let length = (byte_length - relative).min(WEIGHT_LOAD_CHUNK_BYTES);
        chunks.push(WeightLoadChunk {
            source_offset: source_start
                .checked_add(relative)
                .ok_or_else(|| invalid("Gemma 4 MoE source chunk offset overflows"))?,
            destination_offset: destination_start
                .checked_add(relative)
                .ok_or_else(|| invalid("Gemma 4 MoE destination chunk offset overflows"))?,
            byte_length: length,
        });
        relative += length;
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn synthetic_config() -> Value {
        let layer_types: Vec<_> = (0..30)
            .map(|layer| {
                if (layer + 1) % 6 == 0 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect();
        let mut ignore = vec![
            "lm_head".to_owned(),
            "model.embed_vision*".to_owned(),
            "model.vision_tower*".to_owned(),
        ];
        for layer in 0..30 {
            for component in ["mlp", "router", "self_attn"] {
                ignore.push(format!("model.language_model.layers.{layer}.{component}*"));
            }
        }
        ignore.sort();
        json!({
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 2816,
                "num_hidden_layers": 30,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 2,
                "head_dim": 256,
                "global_head_dim": 512,
                "sliding_window": 1024,
                "max_position_embeddings": 262144,
                "vocab_size": 262144,
                "intermediate_size": 2112,
                "num_experts": 128,
                "top_k_experts": 8,
                "moe_intermediate_size": 704,
                "dtype": "bfloat16",
                "hidden_activation": "gelu_pytorch_tanh",
                "enable_moe_block": true,
                "tie_word_embeddings": true,
                "use_cache": true,
                "attention_k_eq_v": true,
                "rms_norm_eps": 1.0e-6,
                "layer_types": layer_types,
                "rope_parameters": {
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                    "full_attention": {
                        "rope_type": "proportional",
                        "rope_theta": 1000000.0,
                        "partial_rotary_factor": 0.25
                    }
                }
            },
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "producer": {"name": "modelopt", "version": "0.43.0rc2.dev91+gc79ebc014"},
                "config_groups": {"group_0": {
                    "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                    "input_activations": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                    "targets": ["Linear"]
                }},
                "kv_cache_scheme": {"dynamic": false, "num_bits": 8, "type": "float"},
                "ignore": ignore
            }
        })
    }

    #[test]
    fn reviewed_config_contract_preserves_topology_and_nvfp4_block_16() {
        let (config, recipe) = validate_config_document(&synthetic_config()).unwrap();
        assert_eq!(config.hidden_size, 2_816);
        assert_eq!(config.layer_count, 30);
        assert_eq!(config.expert_count, 128);
        assert_eq!(config.selected_expert_count, 8);
        assert_eq!(config.layer_types[0], Gemma4LayerType::SlidingAttention);
        assert_eq!(config.layer_types[29], Gemma4LayerType::FullAttention);
        assert_eq!(recipe.block_size, 16);
        assert_eq!(recipe.kv_cache_scale_tensor_count, 0);
        assert_eq!(recipe.kv_cache_dequant_scale_f32_bits, 1.0_f32.to_bits());
        assert_eq!(
            recipe.encoding,
            QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer
        );
    }

    #[test]
    fn config_semantic_mutations_are_rejected() {
        let mut config = synthetic_config();
        config["text_config"]["top_k_experts"] = json!(7);
        assert!(validate_config_document(&config).is_err());
        let mut config = synthetic_config();
        config["quantization_config"]["config_groups"]["group_0"]["weights"]["group_size"] =
            json!(32);
        assert!(validate_config_document(&config).is_err());
    }

    #[test]
    fn index_metadata_and_shard_distribution_are_fail_closed() {
        let root = json!({
            "metadata": {"total_parameters": 3, "total_size": 7},
            "weight_map": {
                "a": "left.safetensors",
                "b": "right.safetensors"
            }
        });
        let index = validate_index_document(
            &root,
            2,
            &[("left.safetensors", 1), ("right.safetensors", 1)],
        )
        .unwrap();
        assert_eq!(index.tensor_count(), 2);
        assert_eq!(index.source_file("b"), Some("right.safetensors"));

        let mut wrong = root;
        wrong["weight_map"]["b"] = json!("left.safetensors");
        assert!(
            validate_index_document(
                &wrong,
                2,
                &[("left.safetensors", 1), ("right.safetensors", 1)]
            )
            .is_err()
        );
    }

    #[test]
    fn expert_boundary_names_and_plane_shapes_are_exact() {
        for (layer, expert) in [(0, 0), (0, 127), (29, 0), (29, 127)] {
            for projection in [
                Gemma4MoeExpertProjection::Gate,
                Gemma4MoeExpertProjection::Up,
                Gemma4MoeExpertProjection::Down,
            ] {
                let name = format!(
                    "model.language_model.layers.{layer}.experts.{expert}.{}.weight_scale",
                    projection.source_stem()
                );
                let parsed = parse_expert_plane_name(&name).unwrap();
                assert_eq!(
                    parsed,
                    (layer, expert, projection, ExpertPlaneRole::BlockScale)
                );
                let entry = SafeTensorEntry {
                    dtype: "F8_E4M3".to_owned(),
                    shape: projection.block_scale_shape().to_vec(),
                    data_offsets: [0, projection.block_scale_shape().iter().product()],
                };
                validate_expert_plane_metadata(
                    &name,
                    &entry,
                    projection,
                    ExpertPlaneRole::BlockScale,
                )
                .unwrap();
            }
        }
        assert!(
            parse_expert_plane_name("model.language_model.layers.30.experts.0.gate_proj.weight")
                .is_err()
        );
        assert!(
            parse_expert_plane_name("model.language_model.layers.0.experts.128.gate_proj.weight")
                .is_err()
        );
    }

    #[test]
    fn expert_plane_shape_and_dtype_mismatches_are_rejected() {
        let name = "model.language_model.layers.0.experts.0.down_proj.weight";
        let mut entry = SafeTensorEntry {
            dtype: "U8".to_owned(),
            shape: vec![2_816, 352],
            data_offsets: [0, 2_816 * 352],
        };
        validate_expert_plane_metadata(
            name,
            &entry,
            Gemma4MoeExpertProjection::Down,
            ExpertPlaneRole::Value,
        )
        .unwrap();
        entry.shape[1] = 351;
        assert!(
            validate_expert_plane_metadata(
                name,
                &entry,
                Gemma4MoeExpertProjection::Down,
                ExpertPlaneRole::Value
            )
            .is_err()
        );
        entry.shape[1] = 352;
        entry.dtype = "BF16".to_owned();
        assert!(
            validate_expert_plane_metadata(
                name,
                &entry,
                Gemma4MoeExpertProjection::Down,
                ExpertPlaneRole::Value
            )
            .is_err()
        );
    }

    #[test]
    fn layer_blob_pack_layout_is_exact_contiguous_and_boundary_checked() {
        let inputs = gemma4_moe_layer_blob_pack_inputs(0).unwrap();
        assert_eq!(inputs.len(), 128 * 3);
        assert_eq!(
            gemma4_moe_layer_blob_name(0),
            "__sllm_gemma4_moe_layer_blob.0"
        );
        assert_eq!(
            gemma4_moe_layer_blob_name(29),
            "__sllm_gemma4_moe_layer_blob.29"
        );
        assert_eq!(
            gemma4_moe_per_expert_scale_destination(),
            [
                GEMMA4_MOE_PER_EXPERT_SCALES_OFFSET,
                GEMMA4_MOE_LAYER_BLOB_BYTES
            ]
        );

        let gate_first = &inputs[0];
        assert_eq!(gate_first.expert, 0);
        assert_eq!(gate_first.projection, Gemma4MoeExpertProjection::Gate);
        assert_eq!(
            gate_first.value_destination,
            [0, GEMMA4_MOE_EXPERT_VALUE_BYTES]
        );
        assert_eq!(
            gate_first.block_scale_destination,
            [
                GEMMA4_MOE_GATE_SCALES_OFFSET,
                GEMMA4_MOE_GATE_SCALES_OFFSET + GEMMA4_MOE_EXPERT_BLOCK_SCALE_BYTES,
            ]
        );
        assert_eq!(
            inputs[127].value_destination[1],
            GEMMA4_MOE_GATE_SCALES_OFFSET
        );
        assert_eq!(
            inputs[128].value_destination[0],
            GEMMA4_MOE_UP_VALUES_OFFSET
        );
        assert_eq!(
            inputs[255].input_scale_destination[1],
            GEMMA4_MOE_DOWN_VALUES_OFFSET
        );
        let down_last = inputs.last().unwrap();
        assert_eq!(down_last.expert, 127);
        assert_eq!(down_last.projection, Gemma4MoeExpertProjection::Down);
        assert_eq!(
            down_last.input_scale_destination[1],
            GEMMA4_MOE_PER_EXPERT_SCALES_OFFSET
        );

        let mut ranges = Vec::with_capacity(inputs.len() * 4 + 1);
        for input in &inputs {
            ranges.extend([
                input.value_destination,
                input.block_scale_destination,
                input.outer_scale_destination,
                input.input_scale_destination,
            ]);
        }
        ranges.push(gemma4_moe_per_expert_scale_destination());
        ranges.sort_unstable();
        assert_eq!(
            ranges.first().copied(),
            Some([0, GEMMA4_MOE_EXPERT_VALUE_BYTES])
        );
        assert_eq!(
            ranges.last().map(|range| range[1]),
            Some(GEMMA4_MOE_LAYER_BLOB_BYTES)
        );
        for adjacent in ranges.windows(2) {
            assert_eq!(adjacent[0][1], adjacent[1][0]);
        }
        assert!(gemma4_moe_layer_blob_pack_inputs(29).is_ok());
        assert!(gemma4_moe_layer_blob_pack_inputs(30).is_err());
    }

    fn direct_plane(name: &str, dtype: &str) -> Gemma4MoeTensorPlane {
        Gemma4MoeTensorPlane {
            source_file: "fixture.safetensors".to_owned(),
            source_name: name.to_owned(),
            dtype: dtype.to_owned(),
            shape: vec![1],
            absolute_byte_range: [8, 10],
        }
    }

    #[test]
    fn direct_tensor_classification_preserves_gemma_specific_roles() {
        let (config, _) = validate_config_document(&synthetic_config()).unwrap();
        for (name, layer, role) in [
            (
                "model.language_model.embed_tokens.weight",
                None,
                WeightConsumer::EmbeddingAndTiedOutput,
            ),
            (
                "model.language_model.layers.0.self_attn.k_proj.weight",
                Some(0),
                WeightConsumer::AttentionK,
            ),
            (
                "model.language_model.layers.0.self_attn.v_proj.weight",
                Some(0),
                WeightConsumer::AttentionV,
            ),
            (
                "model.language_model.layers.5.self_attn.k_proj.weight",
                Some(5),
                WeightConsumer::AttentionKAndV,
            ),
            (
                "model.language_model.layers.0.pre_feedforward_layernorm_2.weight",
                Some(0),
                WeightConsumer::Gemma4MoePreFeedforwardNorm2,
            ),
            (
                "model.language_model.layers.0.post_feedforward_layernorm_1.weight",
                Some(0),
                WeightConsumer::Gemma4MoePostFeedforwardNorm1,
            ),
            (
                "model.language_model.layers.0.post_feedforward_layernorm_2.weight",
                Some(0),
                WeightConsumer::Gemma4MoePostFeedforwardNorm2,
            ),
            (
                "model.language_model.layers.0.router.scale",
                Some(0),
                WeightConsumer::Gemma4MoeRouterScale,
            ),
            (
                "model.language_model.layers.0.router.per_expert_scale",
                Some(0),
                WeightConsumer::Gemma4MoeRouterPerExpertScale,
            ),
        ] {
            assert_eq!(
                classify_gemma4_moe_direct_plane(&direct_plane(name, "BF16"), &config).unwrap(),
                WeightConsumerKey { layer, role }
            );
        }
        assert!(
            classify_gemma4_moe_direct_plane(
                &direct_plane(
                    "model.language_model.layers.5.self_attn.v_proj.weight",
                    "BF16"
                ),
                &config
            )
            .is_err()
        );
        assert!(
            classify_gemma4_moe_direct_plane(
                &direct_plane("model.language_model.layers.30.router.scale", "BF16"),
                &config
            )
            .is_err()
        );
    }

    #[test]
    fn gguf_expert_names_and_standard_nvfp4_unpack_are_exact() {
        for (layer, expert, projection) in [
            (0, 0, Gemma4MoeExpertProjection::Gate),
            (29, 127, Gemma4MoeExpertProjection::Down),
        ] {
            let name = format!(
                "model.language_model.layers.{layer}.experts.{expert}.{}.weight",
                projection.source_stem()
            );
            assert_eq!(
                parse_gguf_gemma4_moe_expert_name(&name).unwrap(),
                (layer, expert, projection)
            );
        }
        assert!(
            parse_gguf_gemma4_moe_expert_name(
                "model.language_model.layers.30.experts.0.gate_proj.weight"
            )
            .is_err()
        );
        assert!(
            parse_gguf_gemma4_moe_expert_name(
                "model.language_model.layers.0.experts.128.gate_proj.weight"
            )
            .is_err()
        );

        let standard = [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe];
        let mut adjacent = Vec::new();
        append_adjacent_nvfp4(&standard, &mut adjacent);
        assert_eq!(adjacent, [0x20, 0x64, 0xa8, 0xec, 0x31, 0x75, 0xb9, 0xfd]);
        let standard = [0x80, 0x91, 0xa2, 0xb3, 0xc4, 0xd5, 0xe6, 0xf7];
        adjacent.clear();
        append_adjacent_nvfp4(&standard, &mut adjacent);
        assert_eq!(adjacent, [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]);
    }

    #[test]
    fn generation_stop_policy_uses_all_reviewed_gemma_turn_boundaries() {
        let policy = gemma4_moe_generation_stop_policy();
        assert_eq!(policy.stop_token_ids, [1, 106, 50]);
        assert!(!policy.stop_token.visible_output);
        assert!(!policy.stop_token.subsequent_decode_input);
    }

    #[test]
    fn metadata_cache_validates_when_explicitly_available() {
        let Some(root) = std::env::var_os("SLLM_GEMMA4_MOE_METADATA_CACHE") else {
            return;
        };
        let root = PathBuf::from(root);
        let config = fs::read(root.join("config.json")).unwrap();
        let index = fs::read(root.join("model.safetensors.index.json")).unwrap();
        let (config, recipe) = validate_gemma4_moe_config(&config).unwrap();
        let index = validate_gemma4_moe_index(&index).unwrap();
        assert_eq!(config.selected_expert_count, 8);
        assert_eq!(recipe.block_size, 16);
        assert_eq!(index.tensor_count(), GEMMA4_MOE_TENSOR_COUNT);
    }

    #[test]
    #[ignore = "requires both immutable 9-10 GB source shards via SLLM_GEMMA4_MOE_CACHE"]
    fn reviewed_external_artifact_passes_full_identity_and_inventory() {
        let root = std::env::var_os("SLLM_GEMMA4_MOE_CACHE")
            .expect("SLLM_GEMMA4_MOE_CACHE must name the immutable snapshot");
        let artifact = verify_gemma4_moe_artifact(root).unwrap();
        assert_eq!(artifact.config().layer_count, 30);
        assert_eq!(artifact.config().selected_expert_count, 8);
        assert_eq!(artifact.text_planes().len(), GEMMA4_MOE_TEXT_TENSOR_COUNT);
        assert_eq!(artifact.experts().len(), GEMMA4_MOE_EXPERT_PROJECTION_COUNT);
        assert!(
            artifact
                .expert(0, 0, Gemma4MoeExpertProjection::Gate)
                .is_some()
        );
        assert!(
            artifact
                .expert(29, 127, Gemma4MoeExpertProjection::Down)
                .is_some()
        );
        let expert = artifact
            .read_expert_planes(29, 127, Gemma4MoeExpertProjection::Down)
            .unwrap();
        assert_eq!(expert.values.len() as u64, GEMMA4_MOE_EXPERT_VALUE_BYTES);
        assert_eq!(
            expert.block_scales.len() as u64,
            GEMMA4_MOE_EXPERT_BLOCK_SCALE_BYTES
        );
        let plan = build_gemma4_moe_weight_load_plan(&artifact).unwrap();
        assert_eq!(plan.entries.len(), 597);
        assert_eq!(
            plan.entries
                .iter()
                .filter(|entry| entry.consumer.is_some_and(|consumer| {
                    consumer.role == WeightConsumer::Gemma4MoeLayerBlob
                }))
                .count(),
            30
        );
        assert!(
            plan.entries
                .iter()
                .all(|entry| !entry.tensor_name.contains(".experts."))
        );
        assert!(
            plan.entries
                .iter()
                .all(|entry| { !entry.tensor_name.ends_with("router.per_expert_scale") })
        );
        assert_eq!(plan.total_destination_bytes, GEMMA4_MOE_TEXT_RESIDENT_BYTES);
        assert!(plan.has_valid_digest().unwrap());
    }

    #[test]
    #[ignore = "requires source cache plus canonical GGUF/lock via SLLM_GEMMA4_MOE_{CACHE,GGUF,GGUF_LOCK}"]
    fn reviewed_external_gguf_passes_identity_reads_and_load_plan() {
        let source_root = std::env::var_os("SLLM_GEMMA4_MOE_CACHE")
            .expect("SLLM_GEMMA4_MOE_CACHE must name the immutable source snapshot");
        let path = std::env::var_os("SLLM_GEMMA4_MOE_GGUF")
            .expect("SLLM_GEMMA4_MOE_GGUF must name the canonical GGUF");
        let lock_path = std::env::var_os("SLLM_GEMMA4_MOE_GGUF_LOCK")
            .expect("SLLM_GEMMA4_MOE_GGUF_LOCK must name its derived lock");
        let lock = crate::read_derived_gguf_lock(lock_path).unwrap();
        let derived = crate::verify_derived_gguf(lock, &path).unwrap();
        let artifact = verify_gguf_gemma4_moe(derived).unwrap();
        let source = verify_gemma4_moe_artifact(source_root).unwrap();
        assert_eq!(artifact.config().layer_count, 30);
        assert_eq!(artifact.direct_planes().len(), 597);
        assert_eq!(
            artifact.expert_tensor_name(0, 0, Gemma4MoeExpertProjection::Gate),
            Some("model.language_model.layers.0.experts.0.gate_proj.weight")
        );
        let expert = artifact
            .read_expert_planes(29, 127, Gemma4MoeExpertProjection::Down)
            .unwrap();
        assert_eq!(expert.values.len() as u64, GEMMA4_MOE_EXPERT_VALUE_BYTES);
        assert_eq!(
            expert.block_scales.len() as u64,
            GEMMA4_MOE_EXPERT_BLOCK_SCALE_BYTES
        );
        for name in [
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.layers.0.router.proj.weight",
            "model.language_model.layers.29.post_feedforward_layernorm_2.weight",
            "model.language_model.layers.29.mlp.down_proj.weight",
            "model.language_model.norm.weight",
        ] {
            assert_eq!(
                source.read_tensor(name).unwrap(),
                artifact.read_tensor(name).unwrap()
            );
        }
        for (layer, expert) in [(0, 0), (29, 127)] {
            for projection in [
                Gemma4MoeExpertProjection::Gate,
                Gemma4MoeExpertProjection::Up,
                Gemma4MoeExpertProjection::Down,
            ] {
                assert_eq!(
                    source
                        .read_expert_planes(layer, expert, projection)
                        .unwrap(),
                    artifact
                        .read_expert_planes(layer, expert, projection)
                        .unwrap()
                );
            }
        }
        let plan = build_gguf_gemma4_moe_weight_load_plan(&artifact).unwrap();
        assert_eq!(plan.entries.len(), 597);
        assert_eq!(
            plan.entries
                .iter()
                .filter(|entry| entry.consumer.is_some_and(|consumer| {
                    consumer.role == WeightConsumer::Gemma4MoeLayerBlob
                }))
                .count(),
            30
        );
        assert!(
            plan.entries
                .iter()
                .all(|entry| !entry.tensor_name.contains(".experts."))
        );
        assert!(
            plan.entries
                .iter()
                .all(|entry| { !entry.tensor_name.ends_with("router.per_expert_scale") })
        );
        assert_eq!(plan.total_destination_bytes, GEMMA4_MOE_TEXT_RESIDENT_BYTES);
        assert!(plan.has_valid_digest().unwrap());
    }

    #[test]
    #[ignore = "diagnostic contract probe requiring SLLM_GEMMA4_MOE_GGUF{,_LOCK}"]
    fn reviewed_external_gguf_container_contract_passes_without_payload_rehash() {
        let path = std::env::var_os("SLLM_GEMMA4_MOE_GGUF")
            .expect("SLLM_GEMMA4_MOE_GGUF must name the canonical GGUF");
        let lock_path = std::env::var_os("SLLM_GEMMA4_MOE_GGUF_LOCK")
            .expect("SLLM_GEMMA4_MOE_GGUF_LOCK must name its derived lock");
        let lock = crate::read_derived_gguf_lock(lock_path).unwrap();
        let gguf = crate::VerifiedGguf::open(&path).unwrap();
        assert_eq!(gguf.file_size(), lock.output.size_bytes);
        assert_eq!(gguf.metadata_sha256(), lock.output.metadata_sha256);
        assert_eq!(
            gguf.tensor_catalog_sha256(),
            lock.output.tensor_catalog_sha256
        );
        let artifact = verify_gguf_gemma4_moe(VerifiedDerivedGguf { lock, gguf }).unwrap();
        assert_eq!(artifact.direct_planes().len(), 597);
        assert_eq!(artifact.config().layer_count, 30);
        for name in [
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.layers.29.post_feedforward_layernorm_2.weight",
            "model.language_model.norm.weight",
        ] {
            assert!(!artifact.read_tensor(name).unwrap().is_empty());
        }
        for (layer, expert) in [(0, 0), (29, 127)] {
            for projection in [
                Gemma4MoeExpertProjection::Gate,
                Gemma4MoeExpertProjection::Up,
                Gemma4MoeExpertProjection::Down,
            ] {
                let planes = artifact
                    .read_expert_planes(layer, expert, projection)
                    .unwrap();
                assert_eq!(planes.values.len() as u64, GEMMA4_MOE_EXPERT_VALUE_BYTES);
                assert_eq!(
                    planes.block_scales.len() as u64,
                    GEMMA4_MOE_EXPERT_BLOCK_SCALE_BYTES
                );
            }
        }
        let plan = build_gguf_gemma4_moe_weight_load_plan(&artifact).unwrap();
        assert_eq!(plan.entries.len(), 597);
        assert_eq!(plan.total_destination_bytes, GEMMA4_MOE_TEXT_RESIDENT_BYTES);
        assert!(plan.has_valid_digest().unwrap());
    }
}
