//! Exact Qwen3.5-4B vision component and processor manifest.

use crate::{
    FrontendAssetKind, ModelLock, QWEN35_4B_FINGERPRINT, QWEN35_4B_REPO_ID, QWEN35_4B_REVISION,
    TensorDType, TensorDescriptor, VerifiedCache,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const QWEN35_VISION_TENSOR_COUNT: usize = 297;
pub const QWEN35_VISION_DEPTH: u32 = 24;
pub const QWEN35_VISION_HIDDEN_SIZE: u64 = 1_024;
pub const QWEN35_VISION_INTERMEDIATE_SIZE: u64 = 4_096;
pub const QWEN35_VISION_OUTPUT_SIZE: u64 = 2_560;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenVisionProcessorContract {
    pub minimum_pixels: u64,
    pub maximum_pixels: u64,
    pub patch_size: u32,
    pub temporal_patch_size: u32,
    pub merge_size: u32,
    pub image_mean_bits: [u64; 3],
    pub image_std_bits: [u64; 3],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenVisionTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub source_file: String,
    pub source_range: [u64; 2],
    pub byte_size: u64,
    pub source_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenVisionManifest {
    pub repo_id: String,
    pub resolved_revision: String,
    pub model_fingerprint: String,
    pub processor: QwenVisionProcessorContract,
    pub tensors: Vec<QwenVisionTensor>,
    pub resident_bytes: u64,
    pub vision_start_token: u32,
    pub vision_end_token: u32,
    pub vision_pad_token: u32,
    pub image_pad_token: u32,
    digest: [u8; 32],
}

impl QwenVisionManifest {
    pub fn digest_hex(&self) -> String {
        format!("sha256:{}", hex(&self.digest))
    }
}

pub fn build_verified_qwen35_vision_manifest(
    lock: &ModelLock,
    cache: &VerifiedCache,
) -> Result<QwenVisionManifest, QwenVisionError> {
    if cache.lock_fingerprint != lock.fingerprint() {
        return Err(QwenVisionError::Invalid(
            "verified cache fingerprint differs from model lock".to_owned(),
        ));
    }
    let processor_bytes = cache
        .read_frontend_asset(FrontendAssetKind::PreprocessorConfigJson)
        .map_err(|error| QwenVisionError::Invalid(error.to_string()))?;
    build_qwen35_vision_manifest(lock, cache.tensors(), &processor_bytes)
}

pub fn build_verified_gguf_qwen35_vision_manifest(
    lock: &ModelLock,
    source: &crate::VerifiedGgufWeightSource,
) -> Result<QwenVisionManifest, QwenVisionError> {
    if source.lock_fingerprint() != lock.fingerprint() {
        return Err(QwenVisionError::Invalid(
            "verified GGUF fingerprint differs from model lock".to_owned(),
        ));
    }
    let processor_bytes = source
        .gguf()
        .extension()
        .and_then(|extension| extension.frontend_assets.get("preprocessor_config.json"))
        .ok_or_else(|| {
            QwenVisionError::Invalid("GGUF preprocessor configuration is absent".to_owned())
        })?;
    let source_path = source.gguf().path().to_string_lossy().into_owned();
    build_qwen35_vision_manifest_with_source(
        lock,
        source.tensors(),
        processor_bytes,
        Some((source_path.as_str(), source.file_sha256())),
    )
}

pub fn build_qwen35_vision_manifest<'a>(
    lock: &ModelLock,
    descriptors: impl IntoIterator<Item = &'a TensorDescriptor>,
    processor_bytes: &[u8],
) -> Result<QwenVisionManifest, QwenVisionError> {
    build_qwen35_vision_manifest_with_source(lock, descriptors, processor_bytes, None)
}

fn build_qwen35_vision_manifest_with_source<'a>(
    lock: &ModelLock,
    descriptors: impl IntoIterator<Item = &'a TensorDescriptor>,
    processor_bytes: &[u8],
    source_override: Option<(&str, &str)>,
) -> Result<QwenVisionManifest, QwenVisionError> {
    validate_lock(lock)?;
    let processor = parse_processor(processor_bytes)?;
    let expected = expected_shapes();
    let locked_files = lock
        .model
        .files
        .iter()
        .map(|file| (file.path.as_str(), file.sha256.as_str()))
        .collect::<BTreeMap<_, _>>();
    let mut observed = BTreeSet::new();
    let mut tensors = Vec::with_capacity(QWEN35_VISION_TENSOR_COUNT);
    let mut resident_bytes = 0_u64;
    for descriptor in descriptors {
        if !descriptor.tensor_name.starts_with("model.visual.") {
            continue;
        }
        let shape = expected
            .get(descriptor.tensor_name.as_str())
            .ok_or_else(|| {
                QwenVisionError::Invalid(format!(
                    "unknown vision tensor: {}",
                    descriptor.tensor_name
                ))
            })?;
        if !observed.insert(descriptor.tensor_name.as_str()) {
            return Err(QwenVisionError::Invalid(format!(
                "duplicate vision tensor: {}",
                descriptor.tensor_name
            )));
        }
        if descriptor.dtype != TensorDType::Bf16 || descriptor.shape.as_slice() != shape.as_slice()
        {
            return Err(QwenVisionError::Invalid(format!(
                "vision tensor shape or dtype differs: {}",
                descriptor.tensor_name
            )));
        }
        if descriptor.absolute_byte_range[0] >= descriptor.absolute_byte_range[1]
            || descriptor.absolute_byte_range[1] - descriptor.absolute_byte_range[0]
                != descriptor.byte_size
        {
            return Err(QwenVisionError::Invalid(format!(
                "vision tensor range differs: {}",
                descriptor.tensor_name
            )));
        }
        let source_sha256 = source_override
            .filter(|(path, _)| *path == descriptor.source_file)
            .map(|(_, digest)| digest)
            .or_else(|| locked_files.get(descriptor.source_file.as_str()).copied())
            .ok_or_else(|| {
                QwenVisionError::Invalid("vision source file is not locked".to_owned())
            })?;
        resident_bytes = resident_bytes
            .checked_add(descriptor.byte_size)
            .ok_or_else(|| {
                QwenVisionError::Invalid("vision resident bytes overflowed".to_owned())
            })?;
        tensors.push(QwenVisionTensor {
            name: descriptor.tensor_name.clone(),
            shape: descriptor.shape.clone(),
            source_file: descriptor.source_file.clone(),
            source_range: descriptor.absolute_byte_range,
            byte_size: descriptor.byte_size,
            source_sha256: source_sha256.to_owned(),
        });
    }
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    if tensors.len() != QWEN35_VISION_TENSOR_COUNT || observed.len() != expected.len() {
        return Err(QwenVisionError::Invalid(format!(
            "vision tensor set differs: observed={}, expected={QWEN35_VISION_TENSOR_COUNT}",
            tensors.len()
        )));
    }
    let special = &lock.model.tokenizer_contract.special_token_ids;
    let token = |name: &str, expected: u32| -> Result<u32, QwenVisionError> {
        let actual = special
            .get(name)
            .copied()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| QwenVisionError::Invalid(format!("missing visual token {name}")))?;
        if actual != expected {
            return Err(QwenVisionError::Invalid(format!(
                "visual token {name} differs"
            )));
        }
        Ok(actual)
    };
    let vision_start_token = token("vision_start", 248_053)?;
    let vision_end_token = token("vision_end", 248_054)?;
    let vision_pad_token = token("vision_pad", 248_055)?;
    let image_pad_token = token("image_pad", 248_056)?;
    let digest = digest_manifest(lock, &processor, resident_bytes, &tensors);
    Ok(QwenVisionManifest {
        repo_id: lock.model.repo_id.clone(),
        resolved_revision: lock.model.resolved_revision.clone(),
        model_fingerprint: lock.fingerprint().to_owned(),
        processor,
        tensors,
        resident_bytes,
        vision_start_token,
        vision_end_token,
        vision_pad_token,
        image_pad_token,
        digest,
    })
}

fn validate_lock(lock: &ModelLock) -> Result<(), QwenVisionError> {
    let vision = &lock.model.architecture.vision;
    if lock.schema_version != "model-lock-v1"
        || lock.model.repo_id != QWEN35_4B_REPO_ID
        || lock.model.resolved_revision != QWEN35_4B_REVISION
        || lock.fingerprint() != QWEN35_4B_FINGERPRINT
        || vision.tensor_prefix != "model.visual."
        || vision.tensor_count != QWEN35_VISION_TENSOR_COUNT as u64
    {
        return Err(QwenVisionError::Invalid(
            "model is not the fixed Qwen3.5-4B vision contract".to_owned(),
        ));
    }
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProcessorWire {
    size: ProcessorSizeWire,
    patch_size: u32,
    temporal_patch_size: u32,
    merge_size: u32,
    image_mean: [f64; 3],
    image_std: [f64; 3],
    processor_class: String,
    image_processor_type: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProcessorSizeWire {
    longest_edge: u64,
    shortest_edge: u64,
}

fn parse_processor(bytes: &[u8]) -> Result<QwenVisionProcessorContract, QwenVisionError> {
    let wire: ProcessorWire = serde_json::from_slice(bytes)
        .map_err(|_| QwenVisionError::Invalid("processor config is malformed".to_owned()))?;
    if wire.size.shortest_edge != 65_536
        || wire.size.longest_edge != 16_777_216
        || wire.patch_size != 16
        || wire.temporal_patch_size != 2
        || wire.merge_size != 2
        || wire.image_mean != [0.5; 3]
        || wire.image_std != [0.5; 3]
        || wire.processor_class != "Qwen3VLProcessor"
        || wire.image_processor_type != "Qwen2VLImageProcessorFast"
    {
        return Err(QwenVisionError::Invalid(
            "processor config differs from the fixed contract".to_owned(),
        ));
    }
    Ok(QwenVisionProcessorContract {
        minimum_pixels: wire.size.shortest_edge,
        maximum_pixels: wire.size.longest_edge,
        patch_size: wire.patch_size,
        temporal_patch_size: wire.temporal_patch_size,
        merge_size: wire.merge_size,
        image_mean_bits: wire.image_mean.map(f64::to_bits),
        image_std_bits: wire.image_std.map(f64::to_bits),
    })
}

fn expected_shapes() -> BTreeMap<String, Vec<u64>> {
    let mut expected = BTreeMap::new();
    for layer in 0..QWEN35_VISION_DEPTH {
        let prefix = format!("model.visual.blocks.{layer}.");
        for (suffix, shape) in [
            ("attn.proj.bias", vec![1_024]),
            ("attn.proj.weight", vec![1_024, 1_024]),
            ("attn.qkv.bias", vec![3_072]),
            ("attn.qkv.weight", vec![3_072, 1_024]),
            ("mlp.linear_fc1.bias", vec![4_096]),
            ("mlp.linear_fc1.weight", vec![4_096, 1_024]),
            ("mlp.linear_fc2.bias", vec![1_024]),
            ("mlp.linear_fc2.weight", vec![1_024, 4_096]),
            ("norm1.bias", vec![1_024]),
            ("norm1.weight", vec![1_024]),
            ("norm2.bias", vec![1_024]),
            ("norm2.weight", vec![1_024]),
        ] {
            expected.insert(format!("{prefix}{suffix}"), shape);
        }
    }
    for (name, shape) in [
        ("model.visual.merger.linear_fc1.bias", vec![4_096]),
        ("model.visual.merger.linear_fc1.weight", vec![4_096, 4_096]),
        ("model.visual.merger.linear_fc2.bias", vec![2_560]),
        ("model.visual.merger.linear_fc2.weight", vec![2_560, 4_096]),
        ("model.visual.merger.norm.bias", vec![1_024]),
        ("model.visual.merger.norm.weight", vec![1_024]),
        (
            "model.visual.patch_embed.proj.weight",
            vec![1_024, 3, 2, 16, 16],
        ),
        ("model.visual.patch_embed.proj.bias", vec![1_024]),
        ("model.visual.pos_embed.weight", vec![2_304, 1_024]),
    ] {
        expected.insert(name.to_owned(), shape);
    }
    expected
}

fn digest_manifest(
    lock: &ModelLock,
    processor: &QwenVisionProcessorContract,
    resident_bytes: u64,
    tensors: &[QwenVisionTensor],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"sLLM-qwen35-vision-manifest-v1\0");
    hash.update(lock.model.repo_id.as_bytes());
    hash.update(lock.model.resolved_revision.as_bytes());
    hash.update(lock.fingerprint().as_bytes());
    hash.update(processor.minimum_pixels.to_le_bytes());
    hash.update(processor.maximum_pixels.to_le_bytes());
    hash.update(processor.patch_size.to_le_bytes());
    hash.update(processor.temporal_patch_size.to_le_bytes());
    hash.update(processor.merge_size.to_le_bytes());
    hash.update(resident_bytes.to_le_bytes());
    for tensor in tensors {
        hash.update(tensor.name.as_bytes());
        hash.update([0]);
        for dimension in &tensor.shape {
            hash.update(dimension.to_le_bytes());
        }
        hash.update(tensor.source_file.as_bytes());
        hash.update(tensor.source_sha256.as_bytes());
        hash.update(tensor.source_range[0].to_le_bytes());
        hash.update(tensor.source_range[1].to_le_bytes());
    }
    hash.finalize().into()
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QwenVisionError {
    Invalid(String),
}

impl fmt::Display for QwenVisionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => {
                write!(formatter, "invalid Qwen3.5 vision manifest: {message}")
            }
        }
    }
}

impl std::error::Error for QwenVisionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_is_exact_at_first_mid_last_and_top_level() {
        let catalog = expected_shapes();
        assert_eq!(catalog.len(), QWEN35_VISION_TENSOR_COUNT);
        assert_eq!(
            catalog["model.visual.blocks.0.attn.qkv.weight"],
            [3_072, 1_024]
        );
        assert_eq!(catalog["model.visual.blocks.12.norm2.bias"], [1_024]);
        assert_eq!(
            catalog["model.visual.blocks.23.mlp.linear_fc2.weight"],
            [1_024, 4_096]
        );
        assert_eq!(
            catalog["model.visual.patch_embed.proj.weight"],
            [1_024, 3, 2, 16, 16]
        );
    }
}
