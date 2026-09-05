//! Container-neutral low-bit model descriptors and the reviewed Unsloth
//! Gemma 4 NVFP4 importer.
//!
//! Safetensors is only the source container here. Execution consumes the
//! typed value/scale planes below, which is also the boundary a future GGUF
//! loader must produce.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{File, Metadata};
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

pub const UNSLOTH_GEMMA4_NVFP4_REPOSITORY: &str = "unsloth/gemma-4-12b-it-NVFP4";
pub const UNSLOTH_GEMMA4_NVFP4_REVISION: &str = "b1f649734b34aa5575b03d186abd1b9be3d0d5c4";
pub const UNSLOTH_GEMMA4_NVFP4_MODEL_SIZE: u64 = 9_304_966_064;
pub const UNSLOTH_GEMMA4_NVFP4_MODEL_SHA256: &str =
    "7c2ee23298e7c3a9247e8947597dca5a38f8b791a0322487466d2bfad8ce704b";
pub const UNSLOTH_GEMMA4_NVFP4_HEADER_BYTES: u64 = 179_720;
pub const UNSLOTH_GEMMA4_NVFP4_HEADER_SHA256: &str =
    "23a75ce46a6f005f9e53d84b7b6b5d015cc9840620f1f44bdce7071603bd5d55";

/// Immutable identity for the exact Unsloth Qwen3.8-27B mixed artifact used
/// by the Phase 76 integration lane.  The model and MTP companion are two
/// independent safetensors containers; their identities are kept separate so
/// an accidental BF16 or differently quantized companion cannot be accepted.
pub const UNSLOTH_QWEN38_NVFP4_REPOSITORY: &str = "unsloth/Qwen3.8-27B-NVFP4";
pub const UNSLOTH_QWEN38_NVFP4_REVISION: &str = "57926baca9a82b4d6906b43f2750d55315f5b10f";
pub const UNSLOTH_QWEN38_NVFP4_MODEL_SIZE: u64 = 22_568_192_096;
pub const UNSLOTH_QWEN38_NVFP4_MODEL_SHA256: &str =
    "c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05";
pub const UNSLOTH_QWEN38_NVFP4_HEADER_BYTES: u64 = 251_128;
pub const UNSLOTH_QWEN38_NVFP4_HEADER_SHA256: &str =
    "4b84e0212253cc57dc5d3a6ac51fec8eb4775c7c0c35d1d59b375822be22f9e4";
pub const UNSLOTH_QWEN38_NVFP4_MTP_SIZE: u64 = 849_400_392;
pub const UNSLOTH_QWEN38_NVFP4_MTP_SHA256: &str =
    "1d8268aa85ace093a561e3e7b63b9d390dac1cd55a90cd55b5ec509c3c9da9fe";
pub const UNSLOTH_QWEN38_NVFP4_MTP_HEADER_BYTES: u64 = 1_600;
pub const UNSLOTH_QWEN38_NVFP4_MTP_HEADER_SHA256: &str =
    "4aa005d948a3fb3d6a3e135a24da2a76c894898b68b21296bfb88adc32524d7b";
pub const UNSLOTH_QWEN38_NVFP4_INDEX_SIZE: u64 = 164_371;
pub const UNSLOTH_QWEN38_NVFP4_INDEX_SHA256: &str =
    "429430e1b9e65b2cb98eff8cd10a06e70a09cee89c48487a3914684aeb6df57f";
pub const UNSLOTH_QWEN38_NVFP4_CONFIG_SIZE: u64 = 22_564;
pub const UNSLOTH_QWEN38_NVFP4_CONFIG_SHA256: &str =
    "1b3c71868d1299e52df6fc907deb202d5132b1ef0f72aae0ef6d15185dd53a5c";

const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;
const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
// Qwen3.8's BF16 token embedding plane is just over 2 GiB.  Keep the
// importer bounded while allowing that single verified tensor to be read in
// one resident upload operation.
const MAX_RANGE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum QuantizedTensorEncoding {
    UnquantizedBf16,
    OcpFp8E4M3FnChannelBf16Scale,
    Nvfp4E2M1Block16E4M3FnF32Outer,
    Mxfp4E2M1Block32E8M0,
    Mxfp8E4M3Block32E8M0,
    Mxfp6E3M2Block32E8M0,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum QuantizedTensorRole {
    Embedding,
    AttentionProjection,
    MlpProjection,
    Normalization,
    Scalar,
    KnownUnconsumed,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ScalePlaneRole {
    WeightBlock,
    WeightOuter,
    InputOuter,
    WeightChannel,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizedScalePlane {
    pub role: ScalePlaneRole,
    pub source_name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub source_range: [u64; 2],
    pub reciprocal: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizedTensorDescriptor {
    pub logical_name: String,
    pub source_name: String,
    pub role: QuantizedTensorRole,
    pub encoding: QuantizedTensorEncoding,
    pub logical_shape: Vec<u64>,
    pub value_dtype: String,
    pub value_shape: Vec<u64>,
    pub value_range: [u64; 2],
    pub scale_planes: Vec<QuantizedScalePlane>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StaticFp8KvScale {
    pub key_decode_scale_bf16: u16,
    pub value_decode_scale_bf16: u16,
}

impl StaticFp8KvScale {
    pub fn key_decode_scale(self) -> f32 {
        f32::from_bits(u32::from(self.key_decode_scale_bf16) << 16)
    }

    pub fn value_decode_scale(self) -> f32 {
        f32::from_bits(u32::from(self.value_decode_scale_bf16) << 16)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MixedPrecisionRecipe {
    pub schema_version: &'static str,
    pub attention_weight: QuantizedTensorEncoding,
    pub attention_input: &'static str,
    pub mlp_weight: QuantizedTensorEncoding,
    pub mlp_input: QuantizedTensorEncoding,
    pub kv_cache: &'static str,
    pub ignored: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
}

impl FileIdentity {
    fn from_metadata(metadata: &Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            size: metadata.len(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
        }
    }
}

#[derive(Debug)]
pub struct VerifiedUnslothGemma4Nvfp4 {
    root: PathBuf,
    model_path: PathBuf,
    model_identity: FileIdentity,
    data_start: u64,
    tensors: BTreeMap<String, QuantizedTensorDescriptor>,
    kv_scales: BTreeMap<u32, StaticFp8KvScale>,
    recipe: MixedPrecisionRecipe,
    recipe_digest: String,
}

/// Verified source inventory for the exact Unsloth Qwen3.8-27B NVFP4
/// artifact.  `tensors` contains logical tensors: physical `weight_packed`,
/// block-scale, global-scale, and FP8 channel-scale planes are represented by
/// one descriptor plus its scale planes.  MTP tensors are retained as
/// known-unconsumed BF16 descriptors so a later companion path can consume
/// them without reopening or re-verifying the source files.
#[derive(Debug)]
pub struct VerifiedUnslothQwen38Nvfp4 {
    root: PathBuf,
    model_path: PathBuf,
    model_identity: FileIdentity,
    model_data_start: u64,
    mtp_path: PathBuf,
    mtp_identity: FileIdentity,
    mtp_data_start: u64,
    main_names: BTreeSet<String>,
    mtp_names: BTreeSet<String>,
    tensors: BTreeMap<String, QuantizedTensorDescriptor>,
    kv_scales: BTreeMap<u32, StaticFp8KvScale>,
    nvfp4_input_global_scale_f32_bits: BTreeMap<String, u32>,
    recipe: MixedPrecisionRecipe,
    recipe_digest: String,
}

impl VerifiedUnslothQwen38Nvfp4 {
    pub fn repository(&self) -> &'static str {
        UNSLOTH_QWEN38_NVFP4_REPOSITORY
    }

    pub fn resolved_revision(&self) -> &'static str {
        UNSLOTH_QWEN38_NVFP4_REVISION
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn recipe(&self) -> &MixedPrecisionRecipe {
        &self.recipe
    }

    pub fn recipe_digest(&self) -> &str {
        &self.recipe_digest
    }

    pub fn tensor(&self, logical_name: &str) -> Option<&QuantizedTensorDescriptor> {
        self.tensors.get(logical_name)
    }

    pub fn tensors(&self) -> impl ExactSizeIterator<Item = &QuantizedTensorDescriptor> {
        self.tensors.values()
    }

    pub fn kv_scale(&self, layer: u32) -> Option<StaticFp8KvScale> {
        self.kv_scales.get(&layer).copied()
    }

    /// Return the exact stored FP32 bits for an NVFP4 projection's dynamic
    /// input-global scale.  The importer validates these values while the
    /// immutable source file is open and keeps the raw representation so a
    /// projection-pack planner can require bit-identical activation variants
    /// without rounding through an arithmetic conversion.
    pub fn nvfp4_input_global_scale_f32_bits(&self, logical_name: &str) -> Option<u32> {
        self.nvfp4_input_global_scale_f32_bits
            .get(logical_name)
            .copied()
    }

    /// Read a verified physical source plane by its safetensors name.  Main
    /// model names and MTP names have disjoint namespaces (`mtp.` is reserved
    /// for the companion), so the file choice is deterministic and does not
    /// rely on a caller-provided path.
    pub fn read_source_range(
        &self,
        source_name: &str,
        range: [u64; 2],
    ) -> Result<Vec<u8>, QuantizedModelError> {
        let (path, identity, data_start, names) = if self.mtp_names.contains(source_name) {
            (
                &self.mtp_path,
                self.mtp_identity,
                self.mtp_data_start,
                &self.mtp_names,
            )
        } else {
            (
                &self.model_path,
                self.model_identity,
                self.model_data_start,
                &self.main_names,
            )
        };
        if !names.contains(source_name) {
            return Err(QuantizedModelError::invalid(format!(
                "source tensor is absent: {source_name}"
            )));
        }
        read_verified_source_range(path, identity, data_start, range)
    }

    /// Read a complete physical tensor plane after checking its exact
    /// descriptor range.  This is the convenient source boundary for a
    /// converter; the descriptor's `source_name` and scale-plane names are
    /// accepted alike.
    pub fn read_tensor_range(
        &self,
        source_name: &str,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, QuantizedModelError> {
        let source_range = self
            .source_range(source_name)
            .ok_or_else(|| QuantizedModelError::invalid("source tensor is absent"))?;
        let end =
            offset
                .checked_add(u64::try_from(length).map_err(|_| {
                    QuantizedModelError::invalid("source read length does not fit u64")
                })?)
                .ok_or_else(|| QuantizedModelError::invalid("source read range overflows"))?;
        if end > source_range[1].saturating_sub(source_range[0]) {
            return Err(QuantizedModelError::invalid(
                "source read exceeds the verified tensor plane",
            ));
        }
        let range = [
            source_range[0]
                .checked_add(offset)
                .ok_or_else(|| QuantizedModelError::invalid("source read offset overflows"))?,
            source_range[0]
                .checked_add(end)
                .ok_or_else(|| QuantizedModelError::invalid("source read end overflows"))?,
        ];
        self.read_source_range(source_name, range)
    }

    pub fn read_f32_reciprocal(
        &self,
        plane: &QuantizedScalePlane,
    ) -> Result<f32, QuantizedModelError> {
        if plane.dtype != "F32" || plane.shape != [1] || !plane.reciprocal {
            return Err(QuantizedModelError::invalid(
                "requested scale is not a reciprocal FP32 scalar",
            ));
        }
        let bytes: [u8; 4] = self
            .read_source_range(&plane.source_name, plane.source_range)?
            .try_into()
            .map_err(|_| QuantizedModelError::invalid("FP32 scale is not four bytes"))?;
        let encoded = f32::from_le_bytes(bytes);
        if !encoded.is_finite() || encoded <= 0.0 {
            return Err(QuantizedModelError::invalid(
                "FP32 reciprocal scale is non-positive or non-finite",
            ));
        }
        Ok(1.0 / encoded)
    }

    fn source_range(&self, source_name: &str) -> Option<[u64; 2]> {
        // The descriptor table intentionally stores logical names, so resolve
        // both logical/value names and every bound scale plane here.
        self.tensors.values().find_map(|descriptor| {
            if descriptor.source_name == source_name {
                return Some(descriptor.value_range);
            }
            descriptor
                .scale_planes
                .iter()
                .find(|plane| plane.source_name == source_name)
                .map(|plane| plane.source_range)
        })
    }
}

impl VerifiedUnslothGemma4Nvfp4 {
    pub fn repository(&self) -> &'static str {
        UNSLOTH_GEMMA4_NVFP4_REPOSITORY
    }

    pub fn resolved_revision(&self) -> &'static str {
        UNSLOTH_GEMMA4_NVFP4_REVISION
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn recipe(&self) -> &MixedPrecisionRecipe {
        &self.recipe
    }

    pub fn recipe_digest(&self) -> &str {
        &self.recipe_digest
    }

    pub fn tensor(&self, logical_name: &str) -> Option<&QuantizedTensorDescriptor> {
        self.tensors.get(logical_name)
    }

    pub fn tensors(&self) -> impl ExactSizeIterator<Item = &QuantizedTensorDescriptor> {
        self.tensors.values()
    }

    pub fn kv_scale(&self, layer: u32) -> Option<StaticFp8KvScale> {
        self.kv_scales.get(&layer).copied()
    }

    pub fn read_frontend_asset(
        &self,
        kind: crate::FrontendAssetKind,
    ) -> Result<Vec<u8>, QuantizedModelError> {
        let (relative, size, digest, maximum) = match kind {
            crate::FrontendAssetKind::ConfigJson => (
                "config.json",
                7_292,
                "bcc0ec0398a9dd0b09586f835f17c05ed2cce99d958dd59ef629ce77e618ee49",
                1024 * 1024,
            ),
            crate::FrontendAssetKind::PreprocessorConfigJson => (
                "processor_config.json",
                1_382,
                "6b938e76555b3e9946890770e1abcd442a4718f34041a58e8139dc8ad34545c9",
                1024 * 1024,
            ),
            crate::FrontendAssetKind::TokenizerJson => (
                "tokenizer.json",
                32_169_726,
                "adbaa8175acf7609b4359724f40eff359ec4fac1a8647eeb99d4422be708e1cf",
                64 * 1024 * 1024,
            ),
            crate::FrontendAssetKind::TokenizerConfigJson => (
                "tokenizer_config.json",
                2_725,
                "de3d45511a4ab7320083c0a4b65ec834fcf3d6c1027002714fceead12d2a6b86",
                1024 * 1024,
            ),
            crate::FrontendAssetKind::ChatTemplateJinja => (
                "chat_template.jinja",
                18_924,
                "845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b",
                64 * 1024,
            ),
        };
        let bytes = bounded_file(self.root.join(relative), maximum)?;
        if bytes.len() as u64 != size || format!("{:x}", Sha256::digest(&bytes)) != digest {
            return Err(QuantizedModelError::invalid(format!(
                "frontend asset identity changed: {relative}"
            )));
        }
        Ok(bytes)
    }

    pub fn read_source_range(&self, range: [u64; 2]) -> Result<Vec<u8>, QuantizedModelError> {
        let length = range[1]
            .checked_sub(range[0])
            .ok_or_else(|| QuantizedModelError::invalid("source range underflow"))?;
        if length > MAX_RANGE_BYTES {
            return Err(QuantizedModelError::invalid(
                "one source range exceeds the bounded importer limit",
            ));
        }
        let mut file = File::open(&self.model_path)
            .map_err(|error| QuantizedModelError::io("open model", error))?;
        let identity = FileIdentity::from_metadata(
            &file
                .metadata()
                .map_err(|error| QuantizedModelError::io("stat model", error))?,
        );
        if identity != self.model_identity {
            return Err(QuantizedModelError::invalid(
                "model identity changed after verification",
            ));
        }
        let absolute = self
            .data_start
            .checked_add(range[0])
            .ok_or_else(|| QuantizedModelError::invalid("absolute source range overflow"))?;
        file.seek(SeekFrom::Start(absolute))
            .map_err(|error| QuantizedModelError::io("seek model", error))?;
        let mut bytes = vec![
            0_u8;
            usize::try_from(length).map_err(|_| {
                QuantizedModelError::invalid("source range does not fit address space")
            })?
        ];
        file.read_exact(&mut bytes)
            .map_err(|error| QuantizedModelError::io("read model range", error))?;
        Ok(bytes)
    }

    pub fn read_f32_reciprocal(
        &self,
        plane: &QuantizedScalePlane,
    ) -> Result<f32, QuantizedModelError> {
        if plane.dtype != "F32" || plane.shape != [1] || !plane.reciprocal {
            return Err(QuantizedModelError::invalid(
                "requested scale is not a reciprocal FP32 scalar",
            ));
        }
        let bytes: [u8; 4] = self
            .read_source_range(plane.source_range)?
            .try_into()
            .map_err(|_| QuantizedModelError::invalid("FP32 scale is not four bytes"))?;
        let encode_scale = f32::from_le_bytes(bytes);
        if !encode_scale.is_finite() || encode_scale <= 0.0 {
            return Err(QuantizedModelError::invalid(
                "FP32 reciprocal scale is non-positive or non-finite",
            ));
        }
        Ok(1.0 / encode_scale)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizedModelError(String);

impl QuantizedModelError {
    fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    fn io(scope: &str, error: std::io::Error) -> Self {
        Self(format!("{scope}: {error}"))
    }
}

impl fmt::Display for QuantizedModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid quantized model: {}", self.0)
    }
}

impl std::error::Error for QuantizedModelError {}

#[derive(Clone, Debug)]
struct SafeTensorEntry {
    dtype: String,
    shape: Vec<u64>,
    range: [u64; 2],
}

type QuantizedInventory = (
    BTreeMap<String, QuantizedTensorDescriptor>,
    BTreeMap<u32, StaticFp8KvScale>,
);

pub fn verify_unsloth_gemma4_nvfp4(
    root: impl AsRef<Path>,
) -> Result<VerifiedUnslothGemma4Nvfp4, QuantizedModelError> {
    let root = root.as_ref();
    let root_metadata = root
        .symlink_metadata()
        .map_err(|error| QuantizedModelError::io("stat model directory", error))?;
    if !root_metadata.is_dir() || root_metadata.file_type().is_symlink() {
        return Err(QuantizedModelError::invalid(
            "model root must be a real directory",
        ));
    }
    verify_exact_file(
        root,
        "config.json",
        7_292,
        "bcc0ec0398a9dd0b09586f835f17c05ed2cce99d958dd59ef629ce77e618ee49",
    )?;
    verify_exact_file(
        root,
        "generation_config.json",
        255,
        "801ecff5b38d5a5f5072cd1fb0ee03afabed577fb754518266fea69453acfe6b",
    )?;
    verify_exact_file(
        root,
        "chat_template.jinja",
        18_924,
        "845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b",
    )?;
    verify_exact_file(
        root,
        "tokenizer.json",
        32_169_726,
        "adbaa8175acf7609b4359724f40eff359ec4fac1a8647eeb99d4422be708e1cf",
    )?;
    verify_exact_file(
        root,
        "tokenizer_config.json",
        2_725,
        "de3d45511a4ab7320083c0a4b65ec834fcf3d6c1027002714fceead12d2a6b86",
    )?;
    verify_exact_file(
        root,
        "recipe.yaml",
        661,
        "bb19ac4f3bf8c4ffb9b88701bef8faf06c679ae9461751cc849375bb1731c17b",
    )?;

    let config_bytes = bounded_file(root.join("config.json"), MAX_CONFIG_BYTES)?;
    let recipe = validate_recipe(&config_bytes)?;
    let model_path = root.join("model.safetensors");
    let mut model =
        File::open(&model_path).map_err(|error| QuantizedModelError::io("open model", error))?;
    let metadata = model
        .metadata()
        .map_err(|error| QuantizedModelError::io("stat model", error))?;
    if !metadata.is_file() || metadata.len() != UNSLOTH_GEMMA4_NVFP4_MODEL_SIZE {
        return Err(QuantizedModelError::invalid(
            "model size or file type differs",
        ));
    }
    if sha256_reader(&mut model)? != UNSLOTH_GEMMA4_NVFP4_MODEL_SHA256 {
        return Err(QuantizedModelError::invalid("model SHA-256 differs"));
    }
    model
        .seek(SeekFrom::Start(0))
        .map_err(|error| QuantizedModelError::io("rewind model", error))?;
    let (data_start, header) = read_header(&mut model)?;
    let (tensors, kv_scales) = build_inventory(&header, &mut model, data_start)?;
    let recipe_digest = recipe_digest(&recipe, &tensors, &kv_scales);
    Ok(VerifiedUnslothGemma4Nvfp4 {
        root: root.to_path_buf(),
        model_path,
        model_identity: FileIdentity::from_metadata(&metadata),
        data_start,
        tensors,
        kv_scales,
        recipe,
        recipe_digest,
    })
}

/// Verify the pinned Unsloth Qwen3.8-27B mixed-precision source artifact.
///
/// This importer deliberately consumes the exact compressed-tensors layout,
/// rather than treating the file as an ordinary BF16 safetensors cache.  The
/// model file contains 168 NVFP4 MLP value/scale groups, 233 FP8 projection
/// value/scale pairs, 16 full-attention static KV scale pairs, and the
/// remaining BF16 text/vision tensors.  The MTP companion is kept in the same
/// verified source object but remains known-unconsumed for the text target.
pub fn verify_unsloth_qwen38_nvfp4(
    root: impl AsRef<Path>,
) -> Result<VerifiedUnslothQwen38Nvfp4, QuantizedModelError> {
    let root = root.as_ref();
    let root_metadata = root
        .symlink_metadata()
        .map_err(|error| QuantizedModelError::io("stat model directory", error))?;
    if !root_metadata.is_dir() || root_metadata.file_type().is_symlink() {
        return Err(QuantizedModelError::invalid(
            "model root must be a real directory",
        ));
    }
    verify_exact_file(
        root,
        "config.json",
        UNSLOTH_QWEN38_NVFP4_CONFIG_SIZE,
        UNSLOTH_QWEN38_NVFP4_CONFIG_SHA256,
    )?;
    verify_exact_file(
        root,
        "model.safetensors.index.json",
        UNSLOTH_QWEN38_NVFP4_INDEX_SIZE,
        UNSLOTH_QWEN38_NVFP4_INDEX_SHA256,
    )?;

    let config_bytes = bounded_file(root.join("config.json"), MAX_CONFIG_BYTES)?;
    let recipe = validate_qwen38_recipe(&config_bytes)?;
    let index_bytes = bounded_file(root.join("model.safetensors.index.json"), MAX_INDEX_BYTES)?;
    let index: Value = serde_json::from_slice(&index_bytes)
        .map_err(|error| QuantizedModelError::invalid(format!("Qwen index JSON: {error}")))?;

    let model_path = root.join("model.safetensors");
    let (mut model, model_identity) = open_verified_blob(
        &model_path,
        UNSLOTH_QWEN38_NVFP4_MODEL_SIZE,
        UNSLOTH_QWEN38_NVFP4_MODEL_SHA256,
    )?;
    let (model_data_start, model_header) = read_qwen_header(
        &mut model,
        model_identity.size,
        UNSLOTH_QWEN38_NVFP4_HEADER_BYTES,
        UNSLOTH_QWEN38_NVFP4_HEADER_SHA256,
        1_953,
    )?;

    let mtp_path = root.join("model_mtp.safetensors");
    let (mut mtp, mtp_identity) = open_verified_blob(
        &mtp_path,
        UNSLOTH_QWEN38_NVFP4_MTP_SIZE,
        UNSLOTH_QWEN38_NVFP4_MTP_SHA256,
    )?;
    let (mtp_data_start, mtp_header) = read_qwen_header(
        &mut mtp,
        mtp_identity.size,
        UNSLOTH_QWEN38_NVFP4_MTP_HEADER_BYTES,
        UNSLOTH_QWEN38_NVFP4_MTP_HEADER_SHA256,
        15,
    )?;
    validate_qwen38_index(&index, &model_header, &mtp_header)?;

    let (tensors, kv_scales) = build_qwen38_inventory(
        &model_header,
        &mtp_header,
        &mut model,
        model_data_start,
        mtp_data_start,
    )?;
    let nvfp4_input_global_scale_f32_bits =
        read_qwen38_nvfp4_input_global_scale_f32_bits(&model_header, &mut model, model_data_start)?;
    let recipe_digest = recipe_digest(&recipe, &tensors, &kv_scales);
    Ok(VerifiedUnslothQwen38Nvfp4 {
        root: root.to_path_buf(),
        model_path,
        model_identity,
        model_data_start,
        mtp_path,
        mtp_identity,
        mtp_data_start,
        main_names: model_header.keys().cloned().collect(),
        mtp_names: mtp_header.keys().cloned().collect(),
        tensors,
        kv_scales,
        nvfp4_input_global_scale_f32_bits,
        recipe,
        recipe_digest,
    })
}

fn validate_qwen38_recipe(bytes: &[u8]) -> Result<MixedPrecisionRecipe, QuantizedModelError> {
    let root: Value = serde_json::from_slice(bytes)
        .map_err(|error| QuantizedModelError::invalid(format!("Qwen3.8 config JSON: {error}")))?;
    let object = root
        .as_object()
        .ok_or_else(|| QuantizedModelError::invalid("Qwen3.8 config is not an object"))?;
    let architecture_ok = object
        .get("architectures")
        .and_then(Value::as_array)
        .map(|values| {
            values.len() == 1 && values[0].as_str() == Some("Qwen3_5ForConditionalGeneration")
        })
        .unwrap_or(false);
    if !architecture_ok
        || object.get("model_type").and_then(Value::as_str) != Some("qwen3_5")
        || object.get("dtype").and_then(Value::as_str) != Some("bfloat16")
        || object.get("tie_word_embeddings").and_then(Value::as_bool) != Some(false)
    {
        return Err(QuantizedModelError::invalid(
            "Qwen3.8 semantic architecture/config differs",
        ));
    }
    let text = object
        .get("text_config")
        .and_then(Value::as_object)
        .ok_or_else(|| QuantizedModelError::invalid("Qwen3.8 text_config is absent"))?;
    for (key, expected) in [
        ("hidden_size", 5120),
        ("num_hidden_layers", 64),
        ("num_attention_heads", 24),
        ("num_key_value_heads", 4),
        ("head_dim", 256),
        ("intermediate_size", 17408),
        ("vocab_size", 248320),
        ("full_attention_interval", 4),
        ("linear_num_key_heads", 16),
        ("linear_num_value_heads", 48),
        ("linear_key_head_dim", 128),
        ("linear_value_head_dim", 128),
        ("linear_conv_kernel_dim", 4),
        ("mtp_num_hidden_layers", 1),
    ] {
        if text.get(key).and_then(Value::as_u64) != Some(expected) {
            return Err(QuantizedModelError::invalid(format!(
                "Qwen3.8 text_config field {key} differs"
            )));
        }
    }
    let quant = object
        .get("quantization_config")
        .and_then(Value::as_object)
        .ok_or_else(|| QuantizedModelError::invalid("Qwen3.8 quantization_config is absent"))?;
    if quant.get("format").and_then(Value::as_str) != Some("mixed-precision")
        || quant.get("quant_method").and_then(Value::as_str) != Some("compressed-tensors")
    {
        return Err(QuantizedModelError::invalid(
            "Qwen3.8 mixed quantization identity differs",
        ));
    }
    let groups = quant
        .get("config_groups")
        .and_then(Value::as_object)
        .ok_or_else(|| QuantizedModelError::invalid("Qwen3.8 config_groups is absent"))?;
    if groups.len() != 2 {
        return Err(QuantizedModelError::invalid(
            "Qwen3.8 quantization group count differs",
        ));
    }
    let group0 = groups
        .get("group_0")
        .and_then(Value::as_object)
        .ok_or_else(|| QuantizedModelError::invalid("Qwen3.8 FP8 group is absent"))?;
    let group1 = groups
        .get("group_1")
        .and_then(Value::as_object)
        .ok_or_else(|| QuantizedModelError::invalid("Qwen3.8 NVFP4 group is absent"))?;
    if group0.get("format").and_then(Value::as_str) != Some("float-quantized")
        || group1.get("format").and_then(Value::as_str) != Some("nvfp4-pack-quantized")
    {
        return Err(QuantizedModelError::invalid(
            "Qwen3.8 quantization group format differs",
        ));
    }
    validate_quant_spec(group0.get("weights"), 8, "channel", None, Some(false))?;
    validate_quant_spec(
        group0.get("input_activations"),
        8,
        "token",
        None,
        Some(true),
    )?;
    validate_quant_spec(
        group1.get("weights"),
        4,
        "tensor_group",
        Some(16),
        Some(false),
    )?;
    validate_quant_spec(
        group1.get("input_activations"),
        4,
        "tensor_group",
        Some(16),
        None,
    )?;
    Ok(MixedPrecisionRecipe {
        schema_version: "sllm-qwen38-mixed-precision-recipe-v1",
        attention_weight: QuantizedTensorEncoding::OcpFp8E4M3FnChannelBf16Scale,
        attention_input: "dynamic-per-token-e4m3fn",
        mlp_weight: QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer,
        mlp_input: QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer,
        kv_cache: "static-per-layer-tensor-e4m3fn-bf16-decode-scale",
        ignored: Vec::new(),
    })
}

fn build_qwen38_inventory(
    model_header: &BTreeMap<String, SafeTensorEntry>,
    mtp_header: &BTreeMap<String, SafeTensorEntry>,
    model: &mut File,
    model_data_start: u64,
    _mtp_data_start: u64,
) -> Result<QuantizedInventory, QuantizedModelError> {
    let mut result = BTreeMap::new();
    let mut consumed = BTreeSet::new();
    for layer in 0..64_u32 {
        let prefix = format!("model.language_model.layers.{layer}");
        for projection in ["gate", "up", "down"] {
            let (rows, columns) = if projection == "down" {
                (5120_u64, 17408_u64)
            } else {
                (17408_u64, 5120_u64)
            };
            let base = format!("{prefix}.mlp.{projection}_proj");
            if layer < 56 {
                let value_name = format!("{base}.weight_packed");
                let block_name = format!("{base}.weight_scale");
                let weight_outer = format!("{base}.weight_global_scale");
                let input_outer = format!("{base}.input_global_scale");
                let value = require(model_header, &value_name, "U8", &[rows, columns / 2])?;
                let block = require(
                    model_header,
                    &block_name,
                    "F8_E4M3",
                    &[rows, columns.div_ceil(16)],
                )?;
                let weight = require(model_header, &weight_outer, "F32", &[1])?;
                let input = require(model_header, &input_outer, "F32", &[1])?;
                consumed.extend([
                    value_name.clone(),
                    block_name.clone(),
                    weight_outer.clone(),
                    input_outer.clone(),
                ]);
                add_qwen38_descriptor(
                    &mut result,
                    QuantizedTensorDescriptor {
                        logical_name: format!("{base}.weight"),
                        source_name: value_name,
                        role: QuantizedTensorRole::MlpProjection,
                        encoding: QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer,
                        logical_shape: vec![rows, columns],
                        value_dtype: value.dtype.clone(),
                        value_shape: value.shape.clone(),
                        value_range: value.range,
                        scale_planes: vec![
                            QuantizedScalePlane {
                                role: ScalePlaneRole::WeightBlock,
                                source_name: block_name,
                                dtype: block.dtype.clone(),
                                shape: block.shape.clone(),
                                source_range: block.range,
                                reciprocal: false,
                            },
                            QuantizedScalePlane {
                                role: ScalePlaneRole::WeightOuter,
                                source_name: weight_outer,
                                dtype: weight.dtype.clone(),
                                shape: weight.shape.clone(),
                                source_range: weight.range,
                                reciprocal: true,
                            },
                            QuantizedScalePlane {
                                role: ScalePlaneRole::InputOuter,
                                source_name: input_outer,
                                dtype: input.dtype.clone(),
                                shape: input.shape.clone(),
                                source_range: input.range,
                                reciprocal: true,
                            },
                        ],
                    },
                )?;
            } else {
                add_qwen38_fp8(
                    model_header,
                    &mut result,
                    &mut consumed,
                    format!("{base}.weight"),
                    vec![rows, columns],
                    QuantizedTensorRole::MlpProjection,
                )?;
            }
        }
        if (layer + 1) % 4 == 0 {
            for (projection, shape) in [
                ("q", vec![12288, 5120]),
                ("k", vec![1024, 5120]),
                ("v", vec![1024, 5120]),
                ("o", vec![5120, 6144]),
            ] {
                add_qwen38_fp8(
                    model_header,
                    &mut result,
                    &mut consumed,
                    format!("{prefix}.self_attn.{projection}_proj.weight"),
                    shape,
                    QuantizedTensorRole::AttentionProjection,
                )?;
            }
            let key_name = format!("{prefix}.self_attn.k_scale");
            let value_name = format!("{prefix}.self_attn.v_scale");
            let key = require(model_header, &key_name, "BF16", &[1])?;
            let value = require(model_header, &value_name, "BF16", &[1])?;
            consumed.extend([key_name, value_name]);
            // These are read below after all range validation, preserving the
            // exact BF16 bit pattern in the recipe identity.
            let _ = (key, value);
        } else {
            for (projection, shape) in [
                ("in_proj_qkv", vec![10240, 5120]),
                ("in_proj_z", vec![6144, 5120]),
                ("out_proj", vec![5120, 6144]),
            ] {
                add_qwen38_fp8(
                    model_header,
                    &mut result,
                    &mut consumed,
                    format!("{prefix}.linear_attn.{projection}.weight"),
                    shape,
                    QuantizedTensorRole::AttentionProjection,
                )?;
            }
        }
    }
    add_qwen38_fp8(
        model_header,
        &mut result,
        &mut consumed,
        "lm_head.weight".to_owned(),
        vec![248320, 5120],
        QuantizedTensorRole::AttentionProjection,
    )?;

    // Every remaining main-file BF16 tensor is a semantic BF16 plane (vision,
    // norms, GDN auxiliaries, embeddings, and scalars).  Quantization helper
    // planes were consumed above and are intentionally not exposed as model
    // tensors.
    for (name, entry) in model_header {
        if consumed.contains(name)
            || name.ends_with(".weight_scale")
            || name.ends_with(".weight_global_scale")
            || name.ends_with(".input_global_scale")
            || name.ends_with(".k_scale")
            || name.ends_with(".v_scale")
        {
            continue;
        }
        if entry.dtype != "BF16" {
            return Err(QuantizedModelError::invalid(format!(
                "unconsumed Qwen3.8 tensor is not BF16: {name}"
            )));
        }
        let role = if name.starts_with("model.visual.") {
            QuantizedTensorRole::KnownUnconsumed
        } else if name.contains("embed_tokens") {
            QuantizedTensorRole::Embedding
        } else if name.ends_with("norm.weight") || name.contains("layernorm.weight") {
            QuantizedTensorRole::Normalization
        } else {
            QuantizedTensorRole::Scalar
        };
        add_qwen38_descriptor(
            &mut result,
            QuantizedTensorDescriptor {
                logical_name: name.clone(),
                source_name: name.clone(),
                role,
                encoding: QuantizedTensorEncoding::UnquantizedBf16,
                logical_shape: entry.shape.clone(),
                value_dtype: entry.dtype.clone(),
                value_shape: entry.shape.clone(),
                value_range: entry.range,
                scale_planes: Vec::new(),
            },
        )?;
    }
    for (name, entry) in mtp_header {
        add_qwen38_descriptor(
            &mut result,
            QuantizedTensorDescriptor {
                logical_name: name.clone(),
                source_name: name.clone(),
                role: QuantizedTensorRole::KnownUnconsumed,
                encoding: QuantizedTensorEncoding::UnquantizedBf16,
                logical_shape: entry.shape.clone(),
                value_dtype: entry.dtype.clone(),
                value_shape: entry.shape.clone(),
                value_range: entry.range,
                scale_planes: Vec::new(),
            },
        )?;
    }
    let nvfp4_count = result
        .values()
        .filter(|tensor| tensor.encoding == QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer)
        .count();
    let fp8_count = result
        .values()
        .filter(|tensor| tensor.encoding == QuantizedTensorEncoding::OcpFp8E4M3FnChannelBf16Scale)
        .count();
    if nvfp4_count != 168 || fp8_count != 233 || result.len() != 1_199 {
        return Err(QuantizedModelError::invalid(format!(
            "Qwen3.8 inventory counts differ: NVFP4={nvfp4_count}, FP8={fp8_count}, logical={} (expected 168/233/1199)",
            result.len()
        )));
    }
    let mut kv_scales = BTreeMap::new();
    for layer in (3..64_u32).step_by(4) {
        let key_name = format!("model.language_model.layers.{layer}.self_attn.k_scale");
        let value_name = format!("model.language_model.layers.{layer}.self_attn.v_scale");
        let key = require(model_header, &key_name, "BF16", &[1])?;
        let value = require(model_header, &value_name, "BF16", &[1])?;
        kv_scales.insert(
            layer,
            StaticFp8KvScale {
                key_decode_scale_bf16: read_bf16_scalar(model, model_data_start, key.range)?,
                value_decode_scale_bf16: read_bf16_scalar(model, model_data_start, value.range)?,
            },
        );
    }
    Ok((result, kv_scales))
}

fn add_qwen38_descriptor(
    result: &mut BTreeMap<String, QuantizedTensorDescriptor>,
    descriptor: QuantizedTensorDescriptor,
) -> Result<(), QuantizedModelError> {
    if result
        .insert(descriptor.logical_name.clone(), descriptor)
        .is_some()
    {
        Err(QuantizedModelError::invalid(
            "duplicate Qwen3.8 logical tensor",
        ))
    } else {
        Ok(())
    }
}

fn add_qwen38_fp8(
    model_header: &BTreeMap<String, SafeTensorEntry>,
    result: &mut BTreeMap<String, QuantizedTensorDescriptor>,
    consumed: &mut BTreeSet<String>,
    base: String,
    shape: Vec<u64>,
    role: QuantizedTensorRole,
) -> Result<(), QuantizedModelError> {
    let value = require(model_header, &base, "F8_E4M3", &shape)?;
    let scale_name = format!("{base}_scale");
    let scale = require(model_header, &scale_name, "BF16", &[shape[0], 1])?;
    consumed.extend([base.clone(), scale_name.clone()]);
    add_qwen38_descriptor(
        result,
        QuantizedTensorDescriptor {
            logical_name: base.clone(),
            source_name: base,
            role,
            encoding: QuantizedTensorEncoding::OcpFp8E4M3FnChannelBf16Scale,
            logical_shape: shape.clone(),
            value_dtype: value.dtype.clone(),
            value_shape: value.shape.clone(),
            value_range: value.range,
            scale_planes: vec![QuantizedScalePlane {
                role: ScalePlaneRole::WeightChannel,
                source_name: scale_name,
                dtype: scale.dtype.clone(),
                shape: scale.shape.clone(),
                source_range: scale.range,
                reciprocal: false,
            }],
        },
    )
}

fn read_verified_source_range(
    path: &Path,
    expected_identity: FileIdentity,
    data_start: u64,
    range: [u64; 2],
) -> Result<Vec<u8>, QuantizedModelError> {
    let length = range[1]
        .checked_sub(range[0])
        .ok_or_else(|| QuantizedModelError::invalid("source range underflows"))?;
    if length > MAX_RANGE_BYTES {
        return Err(QuantizedModelError::invalid(
            "source range exceeds the bounded read limit",
        ));
    }
    let mut file =
        File::open(path).map_err(|error| QuantizedModelError::io("open source", error))?;
    let identity = FileIdentity::from_metadata(
        &file
            .metadata()
            .map_err(|error| QuantizedModelError::io("stat source", error))?,
    );
    if identity != expected_identity {
        return Err(QuantizedModelError::invalid(
            "source identity changed after verification",
        ));
    }
    file.seek(SeekFrom::Start(
        data_start
            .checked_add(range[0])
            .ok_or_else(|| QuantizedModelError::invalid("source offset overflows"))?,
    ))
    .map_err(|error| QuantizedModelError::io("seek source", error))?;
    let mut bytes = vec![
        0_u8;
        usize::try_from(length).map_err(|_| QuantizedModelError::invalid(
            "source range does not fit usize"
        ))?
    ];
    file.read_exact(&mut bytes)
        .map_err(|error| QuantizedModelError::io("read source", error))?;
    Ok(bytes)
}

const MAX_INDEX_BYTES: u64 = 4 * 1024 * 1024;

fn open_verified_blob(
    path: &Path,
    expected_size: u64,
    expected_sha256: &str,
) -> Result<(File, FileIdentity), QuantizedModelError> {
    let metadata = path
        .symlink_metadata()
        .map_err(|error| QuantizedModelError::io("stat safetensors", error))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() != expected_size {
        return Err(QuantizedModelError::invalid(format!(
            "safetensors file identity differs: {}",
            path.display()
        )));
    }
    let mut file =
        File::open(path).map_err(|error| QuantizedModelError::io("open safetensors", error))?;
    let identity = FileIdentity::from_metadata(
        &file
            .metadata()
            .map_err(|error| QuantizedModelError::io("stat safetensors", error))?,
    );
    if identity.size != expected_size || sha256_reader(&mut file)? != expected_sha256 {
        return Err(QuantizedModelError::invalid(format!(
            "safetensors SHA-256 differs: {}",
            path.display()
        )));
    }
    Ok((file, identity))
}

fn read_qwen_header(
    file: &mut File,
    file_size: u64,
    expected_length: u64,
    expected_sha256: &str,
    expected_count: usize,
) -> Result<(u64, BTreeMap<String, SafeTensorEntry>), QuantizedModelError> {
    file.seek(SeekFrom::Start(0))
        .map_err(|error| QuantizedModelError::io("rewind Qwen safetensors", error))?;
    let mut length_bytes = [0_u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|error| QuantizedModelError::io("read Qwen header length", error))?;
    let length = u64::from_le_bytes(length_bytes);
    if length != expected_length || length > MAX_HEADER_BYTES {
        return Err(QuantizedModelError::invalid(
            "Qwen safetensors header length differs",
        ));
    }
    let data_start = length
        .checked_add(8)
        .ok_or_else(|| QuantizedModelError::invalid("Qwen header offset overflows"))?;
    if data_start > file_size {
        return Err(QuantizedModelError::invalid(
            "Qwen safetensors header exceeds file",
        ));
    }
    let mut bytes = vec![
        0_u8;
        usize::try_from(length).map_err(|_| {
            QuantizedModelError::invalid("Qwen header length does not fit address space")
        })?
    ];
    file.read_exact(&mut bytes)
        .map_err(|error| QuantizedModelError::io("read Qwen header", error))?;
    if format!("{:x}", Sha256::digest(&bytes)) != expected_sha256 {
        return Err(QuantizedModelError::invalid(
            "Qwen safetensors header SHA-256 differs",
        ));
    }
    let value: Value = serde_json::from_slice(&bytes).map_err(|error| {
        QuantizedModelError::invalid(format!("Qwen safetensors header JSON: {error}"))
    })?;
    let object = value
        .as_object()
        .ok_or_else(|| QuantizedModelError::invalid("Qwen safetensors header is not an object"))?;
    let mut result = BTreeMap::new();
    let mut spans = Vec::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let entry = value
            .as_object()
            .ok_or_else(|| QuantizedModelError::invalid("Qwen tensor metadata is not an object"))?;
        if entry.len() != 3
            || !entry.contains_key("dtype")
            || !entry.contains_key("shape")
            || !entry.contains_key("data_offsets")
        {
            return Err(QuantizedModelError::invalid(
                "Qwen tensor metadata keys are not exact",
            ));
        }
        let dtype = entry
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| QuantizedModelError::invalid("Qwen tensor dtype is absent"))?
            .to_owned();
        let shape = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| QuantizedModelError::invalid("Qwen tensor shape is absent"))?
            .iter()
            .map(|extent| {
                extent
                    .as_u64()
                    .filter(|extent| *extent != 0)
                    .ok_or_else(|| QuantizedModelError::invalid("Qwen tensor shape is invalid"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|offsets| offsets.len() == 2)
            .ok_or_else(|| QuantizedModelError::invalid("Qwen tensor offsets are absent"))?;
        let range = [
            offsets[0]
                .as_u64()
                .ok_or_else(|| QuantizedModelError::invalid("Qwen tensor start is invalid"))?,
            offsets[1]
                .as_u64()
                .ok_or_else(|| QuantizedModelError::invalid("Qwen tensor end is invalid"))?,
        ];
        if range[0] >= range[1]
            || result
                .insert(
                    name.clone(),
                    SafeTensorEntry {
                        dtype,
                        shape,
                        range,
                    },
                )
                .is_some()
        {
            return Err(QuantizedModelError::invalid(
                "Qwen tensor range or name is invalid",
            ));
        }
        spans.push((range[0], range[1], name.clone()));
    }
    if result.len() != expected_count {
        return Err(QuantizedModelError::invalid(
            "Qwen safetensors tensor count differs",
        ));
    }
    spans.sort_by_key(|span| (span.0, span.1, span.2.clone()));
    let payload_size = file_size - data_start;
    let mut cursor = 0_u64;
    for (start, end, name) in spans {
        if start != cursor || end > payload_size {
            return Err(QuantizedModelError::invalid(format!(
                "Qwen safetensors payload has a gap/overlap: {name}"
            )));
        }
        cursor = end;
    }
    if cursor != payload_size {
        return Err(QuantizedModelError::invalid(
            "Qwen safetensors payload is not fully covered",
        ));
    }
    Ok((data_start, result))
}

fn validate_qwen38_index(
    index: &Value,
    model_header: &BTreeMap<String, SafeTensorEntry>,
    mtp_header: &BTreeMap<String, SafeTensorEntry>,
) -> Result<(), QuantizedModelError> {
    let root = index
        .as_object()
        .ok_or_else(|| QuantizedModelError::invalid("Qwen index is not an object"))?;
    if root.len() != 2 || !root.contains_key("metadata") || !root.contains_key("weight_map") {
        return Err(QuantizedModelError::invalid(
            "Qwen index keys are not exact",
        ));
    }
    let metadata = root
        .get("metadata")
        .and_then(Value::as_object)
        .ok_or_else(|| QuantizedModelError::invalid("Qwen index metadata is absent"))?;
    if metadata.len() != 1
        || metadata.get("total_size").and_then(Value::as_u64)
            != Some(UNSLOTH_QWEN38_NVFP4_MODEL_SIZE + UNSLOTH_QWEN38_NVFP4_MTP_SIZE)
    {
        return Err(QuantizedModelError::invalid(
            "Qwen index total_size differs",
        ));
    }
    let weight_map = root
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| QuantizedModelError::invalid("Qwen index weight_map is absent"))?;
    if weight_map.len() != 1_968 {
        return Err(QuantizedModelError::invalid(
            "Qwen index tensor count differs",
        ));
    }
    let mut model_names = BTreeSet::new();
    let mut mtp_names = BTreeSet::new();
    for (name, file) in weight_map {
        let file = file
            .as_str()
            .ok_or_else(|| QuantizedModelError::invalid("Qwen index shard is not a string"))?;
        let set = match file {
            "model.safetensors" => &mut model_names,
            "model_mtp.safetensors" => &mut mtp_names,
            _ => {
                return Err(QuantizedModelError::invalid(
                    "Qwen index contains an unknown shard",
                ));
            }
        };
        set.insert(name.clone());
    }
    let expected_model: BTreeSet<_> = model_header.keys().cloned().collect();
    let expected_mtp: BTreeSet<_> = mtp_header.keys().cloned().collect();
    if model_names != expected_model || mtp_names != expected_mtp {
        return Err(QuantizedModelError::invalid(
            "Qwen index and safetensors headers differ",
        ));
    }
    Ok(())
}

fn verify_exact_file(
    root: &Path,
    relative: &str,
    size: u64,
    digest: &str,
) -> Result<(), QuantizedModelError> {
    let path = root.join(relative);
    let metadata = path
        .symlink_metadata()
        .map_err(|error| QuantizedModelError::io("stat locked file", error))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() != size {
        return Err(QuantizedModelError::invalid(format!(
            "locked file identity differs: {relative}"
        )));
    }
    let mut file =
        File::open(path).map_err(|error| QuantizedModelError::io("open locked file", error))?;
    if sha256_reader(&mut file)? != digest {
        return Err(QuantizedModelError::invalid(format!(
            "locked file SHA-256 differs: {relative}"
        )));
    }
    Ok(())
}

fn bounded_file(path: PathBuf, maximum: u64) -> Result<Vec<u8>, QuantizedModelError> {
    let metadata = path
        .metadata()
        .map_err(|error| QuantizedModelError::io("stat bounded file", error))?;
    if metadata.len() == 0 || metadata.len() > maximum {
        return Err(QuantizedModelError::invalid("bounded file size is invalid"));
    }
    let mut file =
        File::open(path).map_err(|error| QuantizedModelError::io("open bounded file", error))?;
    let mut bytes = Vec::with_capacity(
        usize::try_from(metadata.len())
            .map_err(|_| QuantizedModelError::invalid("bounded file size does not fit usize"))?,
    );
    file.read_to_end(&mut bytes)
        .map_err(|error| QuantizedModelError::io("read bounded file", error))?;
    Ok(bytes)
}

fn sha256_reader(file: &mut File) -> Result<String, QuantizedModelError> {
    file.seek(SeekFrom::Start(0))
        .map_err(|error| QuantizedModelError::io("rewind hashed file", error))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 16 * 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| QuantizedModelError::io("hash file", error))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn read_header(
    file: &mut File,
) -> Result<(u64, BTreeMap<String, SafeTensorEntry>), QuantizedModelError> {
    let mut length_bytes = [0_u8; 8];
    file.read_exact(&mut length_bytes)
        .map_err(|error| QuantizedModelError::io("read header length", error))?;
    let length = u64::from_le_bytes(length_bytes);
    if length != UNSLOTH_GEMMA4_NVFP4_HEADER_BYTES || length > MAX_HEADER_BYTES {
        return Err(QuantizedModelError::invalid(
            "safetensors header length differs",
        ));
    }
    let mut bytes = vec![
        0_u8;
        usize::try_from(length).map_err(|_| QuantizedModelError::invalid(
            "header length does not fit usize"
        ))?
    ];
    file.read_exact(&mut bytes)
        .map_err(|error| QuantizedModelError::io("read header", error))?;
    if format!("{:x}", Sha256::digest(&bytes)) != UNSLOTH_GEMMA4_NVFP4_HEADER_SHA256 {
        return Err(QuantizedModelError::invalid(
            "safetensors header SHA-256 differs",
        ));
    }
    let value: Value = serde_json::from_slice(&bytes).map_err(|error| {
        QuantizedModelError::invalid(format!("safetensors header JSON: {error}"))
    })?;
    let object = value
        .as_object()
        .ok_or_else(|| QuantizedModelError::invalid("safetensors header is not an object"))?;
    let mut result = BTreeMap::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let entry = value
            .as_object()
            .ok_or_else(|| QuantizedModelError::invalid("tensor metadata is not an object"))?;
        let dtype = entry
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| QuantizedModelError::invalid("tensor dtype is absent"))?
            .to_owned();
        let shape = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| QuantizedModelError::invalid("tensor shape is absent"))?
            .iter()
            .map(|extent| {
                extent
                    .as_u64()
                    .filter(|extent| *extent != 0)
                    .ok_or_else(|| QuantizedModelError::invalid("tensor shape is invalid"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|offsets| offsets.len() == 2)
            .ok_or_else(|| QuantizedModelError::invalid("tensor offsets are absent"))?;
        let range = [
            offsets[0]
                .as_u64()
                .ok_or_else(|| QuantizedModelError::invalid("tensor start is invalid"))?,
            offsets[1]
                .as_u64()
                .ok_or_else(|| QuantizedModelError::invalid("tensor end is invalid"))?,
        ];
        if range[1] < range[0]
            || result
                .insert(
                    name.clone(),
                    SafeTensorEntry {
                        dtype,
                        shape,
                        range,
                    },
                )
                .is_some()
        {
            return Err(QuantizedModelError::invalid(
                "tensor range or name is invalid",
            ));
        }
    }
    if result.len() != 1_389 {
        return Err(QuantizedModelError::invalid(
            "safetensors tensor count differs",
        ));
    }
    Ok((length + 8, result))
}

fn validate_recipe(bytes: &[u8]) -> Result<MixedPrecisionRecipe, QuantizedModelError> {
    let root: Value = serde_json::from_slice(bytes)
        .map_err(|error| QuantizedModelError::invalid(format!("config JSON: {error}")))?;
    let quant = root
        .get("quantization_config")
        .and_then(Value::as_object)
        .ok_or_else(|| QuantizedModelError::invalid("quantization_config is absent"))?;
    if quant.get("format").and_then(Value::as_str) != Some("mixed-precision")
        || quant.get("quant_method").and_then(Value::as_str) != Some("compressed-tensors")
        || quant.get("quantization_status").and_then(Value::as_str) != Some("compressed")
    {
        return Err(QuantizedModelError::invalid(
            "mixed quantization identity differs",
        ));
    }
    let groups = quant
        .get("config_groups")
        .and_then(Value::as_object)
        .ok_or_else(|| QuantizedModelError::invalid("config_groups is absent"))?;
    let group0 = groups
        .get("group_0")
        .and_then(Value::as_object)
        .ok_or_else(|| QuantizedModelError::invalid("attention group is absent"))?;
    let group1 = groups
        .get("group_1")
        .and_then(Value::as_object)
        .ok_or_else(|| QuantizedModelError::invalid("MLP group is absent"))?;
    if groups.len() != 2
        || group0.get("format").and_then(Value::as_str) != Some("float-quantized")
        || group1.get("format").and_then(Value::as_str) != Some("nvfp4-pack-quantized")
    {
        return Err(QuantizedModelError::invalid(
            "quantization group format differs",
        ));
    }
    validate_quant_spec(group0.get("weights"), 8, "channel", None, Some(false))?;
    validate_quant_spec(
        group0.get("input_activations"),
        8,
        "token",
        None,
        Some(true),
    )?;
    validate_quant_spec(
        group1.get("weights"),
        4,
        "tensor_group",
        Some(16),
        Some(false),
    )?;
    validate_quant_spec(
        group1.get("input_activations"),
        4,
        "tensor_group",
        Some(16),
        None,
    )?;
    validate_quant_spec(quant.get("kv_cache_scheme"), 8, "tensor", None, Some(false))?;
    let ignored = quant
        .get("ignore")
        .and_then(Value::as_array)
        .ok_or_else(|| QuantizedModelError::invalid("ignore selector is absent"))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| QuantizedModelError::invalid("ignore selector is not a string"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expected = [
        "model.vision_embedder.patch_dense",
        "model.embed_vision.embedding_projection",
        "model.embed_audio.embedding_projection",
        "lm_head",
    ];
    if ignored != expected {
        return Err(QuantizedModelError::invalid("ignore selector set differs"));
    }
    Ok(MixedPrecisionRecipe {
        schema_version: "sllm-mixed-precision-recipe-v1",
        attention_weight: QuantizedTensorEncoding::OcpFp8E4M3FnChannelBf16Scale,
        attention_input: "dynamic-per-token-e4m3fn",
        mlp_weight: QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer,
        mlp_input: QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer,
        kv_cache: "static-per-layer-tensor-e4m3fn-bf16-decode-scale",
        ignored,
    })
}

fn validate_quant_spec(
    value: Option<&Value>,
    bits: u64,
    strategy: &str,
    group_size: Option<u64>,
    dynamic: Option<bool>,
) -> Result<(), QuantizedModelError> {
    let spec = value
        .and_then(Value::as_object)
        .ok_or_else(|| QuantizedModelError::invalid("quantization spec is absent"))?;
    if spec.get("num_bits").and_then(Value::as_u64) != Some(bits)
        || spec.get("type").and_then(Value::as_str) != Some("float")
        || spec.get("strategy").and_then(Value::as_str) != Some(strategy)
        || spec.get("group_size").and_then(Value::as_u64) != group_size
    {
        return Err(QuantizedModelError::invalid("quantization spec differs"));
    }
    if let Some(dynamic) = dynamic {
        if spec.get("dynamic").and_then(Value::as_bool) != Some(dynamic) {
            return Err(QuantizedModelError::invalid(
                "quantization dynamic policy differs",
            ));
        }
    } else if spec.get("dynamic").and_then(Value::as_str) != Some("local") {
        return Err(QuantizedModelError::invalid("local dynamic policy differs"));
    }
    Ok(())
}

fn require<'a>(
    header: &'a BTreeMap<String, SafeTensorEntry>,
    name: &str,
    dtype: &str,
    shape: &[u64],
) -> Result<&'a SafeTensorEntry, QuantizedModelError> {
    let value = header.get(name).ok_or_else(|| {
        QuantizedModelError::invalid(format!("required tensor is absent: {name}"))
    })?;
    if value.dtype != dtype || value.shape != shape {
        return Err(QuantizedModelError::invalid(format!(
            "required tensor metadata differs: {name}"
        )));
    }
    Ok(value)
}

fn build_inventory(
    header: &BTreeMap<String, SafeTensorEntry>,
    model: &mut File,
    data_start: u64,
) -> Result<QuantizedInventory, QuantizedModelError> {
    let mut result = BTreeMap::new();
    let mut consumed = BTreeSet::new();
    for layer in 0..48_u32 {
        for projection in ["down", "gate", "up"] {
            let base = format!("model.language_model.layers.{layer}.mlp.{projection}_proj");
            let logical = format!("{base}.weight");
            let (rows, columns): (u64, u64) = if projection == "down" {
                (3_840, 15_360)
            } else {
                (15_360, 3_840)
            };
            let value_name = format!("{base}.weight_packed");
            let block_name = format!("{base}.weight_scale");
            let weight_outer = format!("{base}.weight_global_scale");
            let input_outer = format!("{base}.input_global_scale");
            let value = require(header, &value_name, "U8", &[rows, columns / 2])?;
            let block = require(
                header,
                &block_name,
                "F8_E4M3",
                &[rows, columns.div_ceil(16)],
            )?;
            let weight = require(header, &weight_outer, "F32", &[1])?;
            let input = require(header, &input_outer, "F32", &[1])?;
            consumed.extend([
                value_name.clone(),
                block_name.clone(),
                weight_outer.clone(),
                input_outer.clone(),
            ]);
            result.insert(
                logical.clone(),
                QuantizedTensorDescriptor {
                    logical_name: logical,
                    source_name: value_name,
                    role: QuantizedTensorRole::MlpProjection,
                    encoding: QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer,
                    logical_shape: vec![rows, columns],
                    value_dtype: "U8".to_owned(),
                    value_shape: value.shape.clone(),
                    value_range: value.range,
                    scale_planes: vec![
                        QuantizedScalePlane {
                            role: ScalePlaneRole::WeightBlock,
                            source_name: block_name,
                            dtype: block.dtype.clone(),
                            shape: block.shape.clone(),
                            source_range: block.range,
                            reciprocal: false,
                        },
                        QuantizedScalePlane {
                            role: ScalePlaneRole::WeightOuter,
                            source_name: weight_outer,
                            dtype: weight.dtype.clone(),
                            shape: weight.shape.clone(),
                            source_range: weight.range,
                            reciprocal: true,
                        },
                        QuantizedScalePlane {
                            role: ScalePlaneRole::InputOuter,
                            source_name: input_outer,
                            dtype: input.dtype.clone(),
                            shape: input.shape.clone(),
                            source_range: input.range,
                            reciprocal: true,
                        },
                    ],
                },
            );
        }
        let full = (layer + 1) % 6 == 0;
        for projection in ["q", "k", "o", "v"] {
            if full && projection == "v" {
                continue;
            }
            let base = format!("model.language_model.layers.{layer}.self_attn.{projection}_proj");
            let logical = format!("{base}.weight");
            let shape = match projection {
                "q" => vec![if full { 8_192 } else { 4_096 }, 3_840],
                "k" | "v" => vec![if full { 512 } else { 2_048 }, 3_840],
                "o" => vec![3_840, if full { 8_192 } else { 4_096 }],
                _ => unreachable!(),
            };
            let scale_name = format!("{base}.weight_scale");
            let value = require(header, &logical, "F8_E4M3", &shape)?;
            let scale = require(header, &scale_name, "BF16", &[shape[0], 1])?;
            consumed.extend([logical.clone(), scale_name.clone()]);
            result.insert(
                logical.clone(),
                QuantizedTensorDescriptor {
                    logical_name: logical.clone(),
                    source_name: logical,
                    role: QuantizedTensorRole::AttentionProjection,
                    encoding: QuantizedTensorEncoding::OcpFp8E4M3FnChannelBf16Scale,
                    logical_shape: shape,
                    value_dtype: "F8_E4M3".to_owned(),
                    value_shape: value.shape.clone(),
                    value_range: value.range,
                    scale_planes: vec![QuantizedScalePlane {
                        role: ScalePlaneRole::WeightChannel,
                        source_name: scale_name,
                        dtype: scale.dtype.clone(),
                        shape: scale.shape.clone(),
                        source_range: scale.range,
                        reciprocal: false,
                    }],
                },
            );
        }
    }
    if result
        .values()
        .filter(|tensor| tensor.role == QuantizedTensorRole::MlpProjection)
        .count()
        != 144
        || result
            .values()
            .filter(|tensor| tensor.role == QuantizedTensorRole::AttentionProjection)
            .count()
            != 184
    {
        return Err(QuantizedModelError::invalid(
            "mixed projection count differs",
        ));
    }
    let mut kv_scales = BTreeMap::new();
    for layer in 0..48_u32 {
        let key_name = format!("model.language_model.layers.{layer}.self_attn.k_scale");
        let value_name = format!("model.language_model.layers.{layer}.self_attn.v_scale");
        let key = require(header, &key_name, "BF16", &[1])?;
        let value = require(header, &value_name, "BF16", &[1])?;
        consumed.extend([key_name, value_name]);
        kv_scales.insert(
            layer,
            StaticFp8KvScale {
                key_decode_scale_bf16: read_bf16_scalar(model, data_start, key.range)?,
                value_decode_scale_bf16: read_bf16_scalar(model, data_start, value.range)?,
            },
        );
    }
    for (name, entry) in header {
        if consumed.contains(name) || name.ends_with(".input_scale") || result.contains_key(name) {
            continue;
        }
        if entry.dtype == "BF16" && !name.ends_with("_scale") && !name.ends_with("_global_scale") {
            let role = if name.contains("embed_tokens") {
                QuantizedTensorRole::Embedding
            } else if name.ends_with("norm.weight") || name.contains("layernorm.weight") {
                QuantizedTensorRole::Normalization
            } else if name.starts_with("model.vision_embedder.")
                || name.starts_with("model.embed_vision.")
                || name.starts_with("model.embed_audio.")
            {
                QuantizedTensorRole::KnownUnconsumed
            } else {
                QuantizedTensorRole::Scalar
            };
            result.insert(
                name.clone(),
                QuantizedTensorDescriptor {
                    logical_name: name.clone(),
                    source_name: name.clone(),
                    role,
                    encoding: QuantizedTensorEncoding::UnquantizedBf16,
                    logical_shape: entry.shape.clone(),
                    value_dtype: entry.dtype.clone(),
                    value_shape: entry.shape.clone(),
                    value_range: entry.range,
                    scale_planes: Vec::new(),
                },
            );
        }
    }
    Ok((result, kv_scales))
}

fn read_bf16_scalar(
    file: &mut File,
    data_start: u64,
    range: [u64; 2],
) -> Result<u16, QuantizedModelError> {
    if range[1].checked_sub(range[0]) != Some(2) {
        return Err(QuantizedModelError::invalid("BF16 scale is not two bytes"));
    }
    file.seek(SeekFrom::Start(data_start + range[0]))
        .map_err(|error| QuantizedModelError::io("seek BF16 scale", error))?;
    let mut bytes = [0_u8; 2];
    file.read_exact(&mut bytes)
        .map_err(|error| QuantizedModelError::io("read BF16 scale", error))?;
    let bits = u16::from_le_bytes(bytes);
    let value = f32::from_bits(u32::from(bits) << 16);
    if !value.is_finite() || value <= 0.0 {
        return Err(QuantizedModelError::invalid("BF16 KV scale is invalid"));
    }
    Ok(bits)
}

fn read_qwen38_nvfp4_input_global_scale_f32_bits(
    model_header: &BTreeMap<String, SafeTensorEntry>,
    file: &mut File,
    data_start: u64,
) -> Result<BTreeMap<String, u32>, QuantizedModelError> {
    let mut result = BTreeMap::new();
    for layer in 0..56_u32 {
        for projection in ["gate", "up", "down"] {
            let base = format!("model.language_model.layers.{layer}.mlp.{projection}_proj");
            let source_name = format!("{base}.input_global_scale");
            let entry = require(model_header, &source_name, "F32", &[1])?;
            let bits = read_f32_scalar_bits(file, data_start, entry.range)?;
            if result.insert(format!("{base}.weight"), bits).is_some() {
                return Err(QuantizedModelError::invalid(
                    "duplicate Qwen3.8 NVFP4 input-global scale",
                ));
            }
        }
    }
    if result.len() != 168 {
        return Err(QuantizedModelError::invalid(format!(
            "Qwen3.8 NVFP4 input-global scale count differs: {}",
            result.len()
        )));
    }
    Ok(result)
}

fn read_f32_scalar_bits(
    file: &mut File,
    data_start: u64,
    range: [u64; 2],
) -> Result<u32, QuantizedModelError> {
    if range[1].checked_sub(range[0]) != Some(4) {
        return Err(QuantizedModelError::invalid("FP32 scale is not four bytes"));
    }
    let absolute = data_start
        .checked_add(range[0])
        .ok_or_else(|| QuantizedModelError::invalid("FP32 scale offset overflows"))?;
    file.seek(SeekFrom::Start(absolute))
        .map_err(|error| QuantizedModelError::io("seek FP32 scale", error))?;
    let mut bytes = [0_u8; 4];
    file.read_exact(&mut bytes)
        .map_err(|error| QuantizedModelError::io("read FP32 scale", error))?;
    let bits = u32::from_le_bytes(bytes);
    let value = f32::from_bits(bits);
    if !value.is_finite() || value <= 0.0 {
        return Err(QuantizedModelError::invalid("FP32 scale is invalid"));
    }
    Ok(bits)
}

fn recipe_digest(
    recipe: &MixedPrecisionRecipe,
    tensors: &BTreeMap<String, QuantizedTensorDescriptor>,
    kv: &BTreeMap<u32, StaticFp8KvScale>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(recipe.schema_version.as_bytes());
    for tensor in tensors.values() {
        hasher.update(tensor.logical_name.as_bytes());
        hasher.update(format!("{:?}", tensor.encoding).as_bytes());
        for extent in &tensor.logical_shape {
            hasher.update(extent.to_le_bytes());
        }
    }
    for (layer, scales) in kv {
        hasher.update(layer.to_le_bytes());
        hasher.update(scales.key_decode_scale_bf16.to_le_bytes());
        hasher.update(scales.value_decode_scale_bf16.to_le_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_families_do_not_collapse_nvfp4_and_mxfp4() {
        assert_ne!(
            QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer,
            QuantizedTensorEncoding::Mxfp4E2M1Block32E8M0
        );
        assert_ne!(
            QuantizedTensorEncoding::Mxfp4E2M1Block32E8M0,
            QuantizedTensorEncoding::Mxfp8E4M3Block32E8M0
        );
        assert_ne!(
            QuantizedTensorEncoding::Mxfp8E4M3Block32E8M0,
            QuantizedTensorEncoding::Mxfp6E3M2Block32E8M0
        );
    }

    #[test]
    fn static_kv_scale_preserves_bf16_identity() {
        let scale = StaticFp8KvScale {
            key_decode_scale_bf16: 0x3f80,
            value_decode_scale_bf16: 0x4000,
        };
        assert_eq!(scale.key_decode_scale(), 1.0);
        assert_eq!(scale.value_decode_scale(), 2.0);
    }

    #[test]
    fn qwen38_recipe_keeps_fp8_and_nvfp4_groups_distinct() {
        let spec = |bits: u64,
                    strategy: &str,
                    group_size: serde_json::Value,
                    dynamic: serde_json::Value| {
            serde_json::json!({
                "num_bits": bits,
                "type": "float",
                "strategy": strategy,
                "group_size": group_size,
                "dynamic": dynamic,
            })
        };
        let config = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "dtype": "bfloat16",
            "tie_word_embeddings": false,
            "text_config": {
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "intermediate_size": 17408,
                "vocab_size": 248320,
                "full_attention_interval": 4,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "mtp_num_hidden_layers": 1,
            },
            "quantization_config": {
                "format": "mixed-precision",
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "format": "float-quantized",
                        "weights": spec(8, "channel", serde_json::Value::Null, serde_json::json!(false)),
                        "input_activations": spec(8, "token", serde_json::Value::Null, serde_json::json!(true)),
                    },
                    "group_1": {
                        "format": "nvfp4-pack-quantized",
                        "weights": spec(4, "tensor_group", serde_json::json!(16), serde_json::json!(false)),
                        "input_activations": spec(4, "tensor_group", serde_json::json!(16), serde_json::json!("local")),
                    },
                },
            },
        });
        let recipe = validate_qwen38_recipe(config.to_string().as_bytes()).unwrap();
        assert_eq!(
            recipe.attention_weight,
            QuantizedTensorEncoding::OcpFp8E4M3FnChannelBf16Scale
        );
        assert_eq!(
            recipe.mlp_weight,
            QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer
        );
        assert_eq!(
            recipe.kv_cache,
            "static-per-layer-tensor-e4m3fn-bf16-decode-scale"
        );

        let mut mutated = config;
        mutated["quantization_config"]["config_groups"]["group_1"]["weights"]["num_bits"] =
            serde_json::Value::from(8);
        assert!(validate_qwen38_recipe(mutated.to_string().as_bytes()).is_err());
    }

    #[test]
    #[ignore = "requires the immutable Unsloth Qwen3.8-27B-NVFP4 snapshot via SLLM_QWEN38_NVFP4_CACHE"]
    fn reviewed_external_qwen38_artifact_builds_mixed_plan_and_graph() {
        let root = std::env::var_os("SLLM_QWEN38_NVFP4_CACHE")
            .expect("SLLM_QWEN38_NVFP4_CACHE must name the immutable snapshot");
        let artifact = verify_unsloth_qwen38_nvfp4(root).expect("Qwen3.8 artifact verifies");
        let counts =
            artifact
                .tensors()
                .fold(
                    (0_usize, 0_usize, 0_usize),
                    |(nvfp4, fp8, bf16), tensor| match tensor.encoding {
                        QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer => {
                            (nvfp4 + 1, fp8, bf16)
                        }
                        QuantizedTensorEncoding::OcpFp8E4M3FnChannelBf16Scale => {
                            (nvfp4, fp8 + 1, bf16)
                        }
                        QuantizedTensorEncoding::UnquantizedBf16 => (nvfp4, fp8, bf16 + 1),
                        _ => (nvfp4, fp8, bf16),
                    },
                );
        assert_eq!(counts, (168, 233, 798));
        assert_eq!(artifact.tensors().len(), 1_199);
        assert_eq!(artifact.kv_scale(3).unwrap().key_decode_scale_bf16, 0x3ce1);
        assert_eq!(
            artifact.kv_scale(3).unwrap().value_decode_scale_bf16,
            0x3cc9
        );

        let lock_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/models/locks/qwen3.5-27b-bf16.json");
        let lock = crate::read_model_lock(lock_path).expect("reviewed Qwen3.5-27B lock parses");
        let plan = crate::build_qwen38_nvfp4_weight_load_plan(&lock, &artifact)
            .expect("Qwen3.8 mixed load plan builds");
        assert_eq!(plan.entries.len(), 1_199);
        assert_eq!(
            plan.entries
                .iter()
                .filter(|entry| entry.classification == crate::WeightClassification::Required)
                .count(),
            851
        );
        let graph = crate::build_qwen35_unsloth_qwen38_nvfp4_graph(
            &lock,
            &plan,
            &artifact,
            1,
            17,
            crate::KvCacheEncoding::Fp16,
        )
        .expect("Qwen3.8 mixed graph builds");
        assert!(!graph.is_mtp());
        assert!(!graph.is_multimodal());
        assert_eq!(graph.weight_bindings().len(), 851);
        let packs = graph
            .plan_unsloth_qwen38_projection_pack_reuse(&artifact)
            .expect("Qwen3.8 projection-pack planner succeeds")
            .expect("exact Qwen3.8 projection-pack plan applies");
        let counts = packs.packs().iter().fold(
            (0_usize, 0_usize, 0_usize, 0_usize),
            |(nv_mlp, fp8_mlp, fp8_qkv, fp8_gdn), pack| match pack.kind() {
                crate::qwen_graph::Qwen38ProjectionPackKind::Nvfp4MlpGateUp => {
                    (nv_mlp + 1, fp8_mlp, fp8_qkv, fp8_gdn)
                }
                crate::qwen_graph::Qwen38ProjectionPackKind::Fp8MlpGateUp => {
                    (nv_mlp, fp8_mlp + 1, fp8_qkv, fp8_gdn)
                }
                crate::qwen_graph::Qwen38ProjectionPackKind::Fp8FullAttentionQkv => {
                    (nv_mlp, fp8_mlp, fp8_qkv + 1, fp8_gdn)
                }
                crate::qwen_graph::Qwen38ProjectionPackKind::Fp8GdnQkvZ => {
                    (nv_mlp, fp8_mlp, fp8_qkv, fp8_gdn + 1)
                }
            },
        );
        assert_eq!(counts, (56, 8, 16, 48));
        for layer in 0..56_u32 {
            let prefix = format!("model.language_model.layers.{layer}.mlp");
            assert_eq!(
                artifact.nvfp4_input_global_scale_f32_bits(&format!("{prefix}.gate_proj.weight")),
                artifact.nvfp4_input_global_scale_f32_bits(&format!("{prefix}.up_proj.weight"))
            );
        }
    }
}
