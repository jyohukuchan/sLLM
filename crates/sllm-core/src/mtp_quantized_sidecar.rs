//! Verified MXFP8 and NVFP4 sidecars for the Qwen3.8 MTP companion.
//!
//! The sidecar owns only the eight matrix weights of the one-layer MTP
//! companion.  Norms, the shared embedding/output, and the original weight
//! load plan remain BF16/FP8 source bindings.  The source model artifact is
//! therefore still the provenance authority; this module only supplies the
//! replacement value and format-specific scale planes.  NVFP4 uses packed
//! E2M1 values, block16 E4M3FN scales, one FP32 weight tensor scale, and a
//! separately authenticated calibration manifest containing resident input `g`.

use crate::{
    ModelLock, QuantizedMx, QuantizedTensorEncoding, UNSLOTH_QWEN38_NVFP4_MTP_SHA256,
    UNSLOTH_QWEN38_NVFP4_REPOSITORY, UNSLOTH_QWEN38_NVFP4_REVISION, VerifiedUnslothQwen38Nvfp4,
    quantize_mxfp8_e4m3, quantize_mxfp8_e4m3_no_clipping_scale, quantize_nvfp4_weights,
    validate_qwen38_mtp_artifact,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{FileExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const SCHEMA: &str = "sllm-qwen38-mtp-mx-sidecar-v1";
const NVFP4_SCHEMA: &str = "sllm-qwen38-mtp-nvfp4-sidecar-v1";
const NVFP4_ACTIVATION_SCALE_SCHEMA: &str = "qwen38-mtp-nvfp4-activation-scale-v1";
const NVFP4_CALIBRATION_SCALE_RULE: &str = "f32(max_abs_bf16_activation / (6 * 448))";
const PAYLOAD_FILE: &str = "payload.safetensors";
const MANIFEST_FILE: &str = "manifest.json";
const SCALE_SUFFIX: &str = ".sllm_mxfp_scale";
const NVFP4_BLOCK_SCALE_SUFFIX: &str = ".sllm_nvfp4_block_scale";
const NVFP4_TENSOR_SCALE_SUFFIX: &str = ".sllm_nvfp4_tensor_scale";
const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;
const MAX_TENSOR_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const DIGEST_DOMAIN: &[u8] = b"sLLM-qwen38-mtp-combined-recipe-v1\0";
const NVFP4_INPUT_SCALE_CONVENTION: &str = "raw-calibrated-g; resident activation scale is g";

const NVFP4_ACTIVATION_SITES: [&str; 5] = [
    "mtp.concat.output",
    "layer.64.input_rmsnorm.output",
    "layer.64.full.sigmoid_mul.output",
    "layer.64.post_attention_rmsnorm.output",
    "layer.64.mlp.silu_mul.output",
];

const MTP_MATRIX_NAMES: [&str; 8] = [
    "mtp.fc.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
];

fn expected_matrix_shape(name: &str) -> Option<[u64; 2]> {
    match name {
        "mtp.fc.weight" => Some([5_120, 10_240]),
        "mtp.layers.0.mlp.down_proj.weight" => Some([5_120, 17_408]),
        "mtp.layers.0.mlp.gate_proj.weight" | "mtp.layers.0.mlp.up_proj.weight" => {
            Some([17_408, 5_120])
        }
        "mtp.layers.0.self_attn.k_proj.weight" | "mtp.layers.0.self_attn.v_proj.weight" => {
            Some([1_024, 5_120])
        }
        "mtp.layers.0.self_attn.o_proj.weight" => Some([5_120, 6_144]),
        "mtp.layers.0.self_attn.q_proj.weight" => Some([12_288, 5_120]),
        _ => None,
    }
}

/// The physical MX format selected for the MTP matrix weights.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MtpWeightEncoding {
    Bf16,
    Mxfp8W8A8Block32E8M0,
    Mxfp8W8A8Block32E8M0NoClippingScale,
    Nvfp4W4A4Block16E2M1E4M3FnF32,
}

/// Stage-0 MXFP8 fake-quant recipe. The resident payload is BF16.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MtpBf16RoundtripEncoding {
    Mxfp8,
}

impl MtpBf16RoundtripEncoding {
    pub const fn manifest_name(self) -> &'static str {
        match self {
            Self::Mxfp8 => "bf16-roundtrip-mxfp8",
        }
    }

    fn quantize(
        self,
        input: &[f32],
        rows: usize,
        columns: usize,
    ) -> Result<QuantizedMx, MtpQuantizedSidecarError> {
        match self {
            Self::Mxfp8 => quantize_mxfp8_e4m3(input, rows, columns),
        }
        .map_err(|error| MtpQuantizedSidecarError::invalid(error.to_string()))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bf16RoundtripDiagnostics {
    pub recipe: String,
    pub element_count: u64,
    pub bit_exact: bool,
    pub source_nonfinite_count: u64,
    pub dequant_nonfinite_count: u64,
    pub roundtrip_nonfinite_count: u64,
    pub underflow_count: u64,
    pub overflow_count: u64,
    pub bit_mismatch_count: u64,
    pub first_source_nonfinite_positions: Vec<u64>,
    pub first_dequant_nonfinite_positions: Vec<u64>,
    pub first_roundtrip_nonfinite_positions: Vec<u64>,
    pub first_underflow_positions: Vec<u64>,
    pub first_overflow_positions: Vec<u64>,
    pub first_bit_mismatch_positions: Vec<u64>,
}

impl MtpWeightEncoding {
    pub const fn manifest_name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Mxfp8W8A8Block32E8M0 => "mxfp8-w8a8-e4m3-block32-e8m0",
            Self::Mxfp8W8A8Block32E8M0NoClippingScale => {
                "mxfp8-w8a8-e4m3-block32-e8m0:mx-scale=no-clipping"
            }
            Self::Nvfp4W4A4Block16E2M1E4M3FnF32 => "nvfp4-w4a4-e2m1-block16-e4m3fn-f32",
        }
    }

    fn parse(value: &str) -> Result<Self, MtpQuantizedSidecarError> {
        match value {
            "bf16" => Ok(Self::Bf16),
            "mxfp8-w8a8-e4m3-block32-e8m0" => Ok(Self::Mxfp8W8A8Block32E8M0),
            "mxfp8-w8a8-e4m3-block32-e8m0:mx-scale=no-clipping" => {
                Ok(Self::Mxfp8W8A8Block32E8M0NoClippingScale)
            }
            "nvfp4-w4a4-e2m1-block16-e4m3fn-f32" => Ok(Self::Nvfp4W4A4Block16E2M1E4M3FnF32),
            "mxfp6-w6a6-e3m2-block32-e8m0"
            | "mxfp6-w6a6-e3m2-block32-e8m0:mx-scale=no-clipping" => Err(
                MtpQuantizedSidecarError::invalid("retired MTP MXFP6 sidecar encoding"),
            ),
            _ => Err(MtpQuantizedSidecarError::invalid(
                "unsupported MTP sidecar encoding",
            )),
        }
    }

    const fn value_dtype(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::Mxfp8W8A8Block32E8M0 | Self::Mxfp8W8A8Block32E8M0NoClippingScale => "F8_E4M3",
            Self::Nvfp4W4A4Block16E2M1E4M3FnF32 => "U8",
        }
    }

    const fn scale_dtype(self) -> &'static str {
        "U8"
    }
}

fn parse_manifest_encoding(
    value: &str,
) -> Result<(MtpWeightEncoding, Option<MtpBf16RoundtripEncoding>), MtpQuantizedSidecarError> {
    match value {
        "bf16-roundtrip-mxfp8" => Ok((
            MtpWeightEncoding::Bf16,
            Some(MtpBf16RoundtripEncoding::Mxfp8),
        )),
        "bf16-roundtrip-mxfp6" => Err(MtpQuantizedSidecarError::invalid(
            "retired MTP MXFP6 roundtrip sidecar encoding",
        )),
        "bf16" => Err(MtpQuantizedSidecarError::invalid(
            "plain BF16 MTP sidecars are not a verified roundtrip recipe",
        )),
        value => Ok((MtpWeightEncoding::parse(value)?, None)),
    }
}

impl fmt::Display for MtpWeightEncoding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.manifest_name())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MtpQuantizedSidecarError(String);

impl MtpQuantizedSidecarError {
    fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
    fn io(operation: &str, error: impl fmt::Display) -> Self {
        Self::invalid(format!("{operation}: {error}"))
    }
}

impl fmt::Display for MtpQuantizedSidecarError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid Qwen3.8 MTP quantized sidecar: {}",
            self.0
        )
    }
}

impl std::error::Error for MtpQuantizedSidecarError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MtpQuantizedSidecarTensor {
    pub name: String,
    pub logical_shape: [u64; 2],
    pub value_range: [u64; 2],
    pub scale_range: [u64; 2],
    /// NVFP4 has a separate one-element FP32 tensor-scale plane.  MX/BF16
    /// sidecars leave this absent, preserving their original contract.
    pub tensor_scale_range: Option<[u64; 2]>,
    pub source_sha256: String,
    pub value_sha256: String,
    pub scale_sha256: String,
    pub tensor_scale_sha256: Option<String>,
    /// The verified NVFP4 weight tensor-scale bits and calibrated input
    /// scale `g`. The resident activation quantizer uses this `g` unchanged.
    pub nvfp4_weight_tensor_scale_f32_bits: Option<u32>,
    pub nvfp4_input_global_scale_f32_bits: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct VerifiedQwen38MtpQuantizedSidecar {
    manifest_path: PathBuf,
    payload_path: PathBuf,
    payload: Arc<File>,
    source_lock_fingerprint: String,
    source_mtp_sha256: String,
    base_recipe_digest: String,
    manifest_fingerprint: String,
    manifest_encoding: String,
    roundtrip_encoding: Option<MtpBf16RoundtripEncoding>,
    encoding: MtpWeightEncoding,
    data_start: u64,
    tensors: BTreeMap<String, MtpQuantizedSidecarTensor>,
}

impl VerifiedQwen38MtpQuantizedSidecar {
    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    pub fn payload_path(&self) -> &Path {
        &self.payload_path
    }

    pub fn encoding(&self) -> MtpWeightEncoding {
        self.encoding
    }

    pub fn manifest_encoding(&self) -> &str {
        &self.manifest_encoding
    }

    pub fn roundtrip_encoding(&self) -> Option<MtpBf16RoundtripEncoding> {
        self.roundtrip_encoding
    }

    pub fn source_lock_fingerprint(&self) -> &str {
        &self.source_lock_fingerprint
    }

    pub fn source_mtp_sha256(&self) -> &str {
        &self.source_mtp_sha256
    }

    pub fn base_recipe_digest(&self) -> &str {
        &self.base_recipe_digest
    }

    pub fn manifest_fingerprint(&self) -> &str {
        &self.manifest_fingerprint
    }

    /// Identity used by graph and resident constructors when a quantized
    /// companion is paired with the unchanged target artifact.
    pub fn combined_recipe_digest(&self, base_recipe_digest: &str) -> String {
        combined_recipe_digest(
            base_recipe_digest,
            &self.manifest_fingerprint,
            &self.manifest_encoding,
        )
    }

    pub fn tensor(&self, name: &str) -> Option<&MtpQuantizedSidecarTensor> {
        self.tensors.get(name)
    }

    pub fn tensors(&self) -> impl ExactSizeIterator<Item = &MtpQuantizedSidecarTensor> {
        self.tensors.values()
    }

    /// Return `(weight_tensor_scale_bits, raw_calibrated_input_g_bits)` for
    /// an NVFP4 tensor. The second value is calibrated `g` itself and is
    /// uploaded unchanged as the resident activation scale.
    pub fn nvfp4_scale_bits(&self, name: &str) -> Option<(u32, u32)> {
        if self.encoding != MtpWeightEncoding::Nvfp4W4A4Block16E2M1E4M3FnF32 {
            return None;
        }
        let tensor = self.tensors.get(name)?;
        Some((
            tensor.nvfp4_weight_tensor_scale_f32_bits?,
            tensor.nvfp4_input_global_scale_f32_bits?,
        ))
    }

    /// The input scale stored by this sidecar is the resident `g` itself.
    pub const fn nvfp4_input_scale_convention() -> &'static str {
        NVFP4_INPUT_SCALE_CONVENTION
    }

    /// Read and hash-check the three NVFP4 planes: packed E2M1 values,
    /// block16 E4M3FN scales, and the one-element FP32 tensor scale.
    #[allow(clippy::type_complexity)] // The tuple mirrors the three on-disk planes.
    pub fn read_nvfp4_tensor_bytes(
        &self,
        name: &str,
    ) -> Result<(Vec<u8>, Vec<u8>, [u8; 4]), MtpQuantizedSidecarError> {
        if self.encoding != MtpWeightEncoding::Nvfp4W4A4Block16E2M1E4M3FnF32 {
            return Err(MtpQuantizedSidecarError::invalid(
                "MTP sidecar is not an NVFP4 sidecar",
            ));
        }
        let tensor = self
            .tensor(name)
            .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP sidecar tensor is absent"))?;
        let tensor_scale_range = tensor.tensor_scale_range.ok_or_else(|| {
            MtpQuantizedSidecarError::invalid("NVFP4 tensor scale plane is absent")
        })?;
        let tensor_scale_sha256 = tensor.tensor_scale_sha256.as_deref().ok_or_else(|| {
            MtpQuantizedSidecarError::invalid("NVFP4 tensor scale hash is absent")
        })?;
        let values = read_range(&self.payload, self.data_start, tensor.value_range)?;
        let block_scales = read_range(&self.payload, self.data_start, tensor.scale_range)?;
        let tensor_scale: [u8; 4] = read_range(&self.payload, self.data_start, tensor_scale_range)?
            .try_into()
            .map_err(|_| {
                MtpQuantizedSidecarError::invalid("NVFP4 tensor scale is not four bytes")
            })?;
        if sha256_bytes(&values) != tensor.value_sha256
            || sha256_bytes(&block_scales) != tensor.scale_sha256
            || sha256_bytes(&tensor_scale) != tensor_scale_sha256
        {
            return Err(MtpQuantizedSidecarError::invalid(
                "MTP sidecar NVFP4 payload changed after verification",
            ));
        }
        let scale = f32::from_le_bytes(tensor_scale);
        if !scale.is_finite() || scale <= 0.0 {
            return Err(MtpQuantizedSidecarError::invalid(
                "MTP sidecar NVFP4 tensor scale is non-positive or non-finite",
            ));
        }
        Ok((values, block_scales, tensor_scale))
    }

    /// Read and hash-check one value/scale pair.  This catches payload
    /// mutation after verification before bytes reach the resident upload.
    pub fn read_tensor_bytes(
        &self,
        name: &str,
    ) -> Result<(Vec<u8>, Vec<u8>), MtpQuantizedSidecarError> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP sidecar tensor is absent"))?;
        let values = read_range(&self.payload, self.data_start, tensor.value_range)?;
        let scales = read_range(&self.payload, self.data_start, tensor.scale_range)?;
        if sha256_bytes(&values) != tensor.value_sha256
            || sha256_bytes(&scales) != tensor.scale_sha256
        {
            return Err(MtpQuantizedSidecarError::invalid(
                "MTP sidecar value or scale payload changed after verification",
            ));
        }
        Ok((values, scales))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: String,
    source: Source,
    converter: String,
    encoding: String,
    base_recipe_digest: String,
    payload: Payload,
    tensors: Vec<TensorRecord>,
    #[serde(default)]
    activation_scales: Option<OutputActivationScales>,
    fingerprint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    repository: String,
    resolved_revision: String,
    lock_fingerprint: String,
    mtp_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    path: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorRecord {
    name: String,
    logical_shape: Vec<u64>,
    source_sha256: String,
    value_sha256: String,
    scale_sha256: String,
    #[serde(default)]
    roundtrip: Option<Bf16RoundtripDiagnostics>,
    #[serde(default)]
    tensor_scale_sha256: Option<String>,
    #[serde(default)]
    input_global_scale_f32_bits: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputActivationScales {
    schema: String,
    scale_rule: String,
    input_manifest_sha256: String,
    suite_sha256: String,
    source_report_sha256: String,
    source_reports: Vec<CalibrationSourceReport>,
    activation_amax: BTreeMap<String, f64>,
    input_scale_convention: String,
    scales: BTreeMap<String, u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationSourceReport {
    pub target: String,
    pub report_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputActivationScaleManifest {
    schema: String,
    scale_rule: String,
    input_manifest_sha256: String,
    suite_sha256: String,
    source_report_sha256: String,
    source_reports: Vec<CalibrationSourceReport>,
    activation_amax: BTreeMap<String, Value>,
    scales: BTreeMap<String, Value>,
}

/// Calibration identity and exact FP32 bit patterns used by an NVFP4 MTP
/// companion. `scales` contains calibrated resident activation scale `g` bits.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpNvfp4ActivationScaleManifest {
    pub schema: String,
    pub scale_rule: String,
    pub input_manifest_sha256: String,
    pub suite_sha256: String,
    pub source_report_sha256: String,
    pub source_reports: Vec<CalibrationSourceReport>,
    pub activation_amax: BTreeMap<String, f64>,
    pub scales: BTreeMap<String, u32>,
}

#[derive(Clone, Deserialize)]
struct SafeTensorMetadata {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Serialize)]
struct OutputManifest<'a> {
    schema_version: &'static str,
    source: OutputSource<'a>,
    converter: &'static str,
    encoding: &'static str,
    base_recipe_digest: &'a str,
    payload: OutputPayload<'a>,
    tensors: &'a [OutputTensor],
    fingerprint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    activation_scales: Option<&'a OutputActivationScales>,
}

#[derive(Serialize)]
struct OutputSource<'a> {
    repository: &'static str,
    resolved_revision: &'static str,
    lock_fingerprint: &'a str,
    mtp_sha256: &'static str,
}

#[derive(Serialize)]
struct OutputPayload<'a> {
    path: &'a str,
    size_bytes: u64,
    sha256: String,
}

#[derive(Clone, Serialize)]
struct OutputTensor {
    name: String,
    logical_shape: [u64; 2],
    source_sha256: String,
    value_sha256: String,
    scale_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    roundtrip: Option<Bf16RoundtripDiagnostics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tensor_scale_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_global_scale_f32_bits: Option<u32>,
}

struct ConvertedTensor {
    record: OutputTensor,
    value_dtype: &'static str,
    value_shape: Vec<u64>,
    scale_shape: Vec<u64>,
    values: Vec<u8>,
    scales: Vec<u8>,
    tensor_scale: Vec<u8>,
}

/// Read the strict activation calibration manifest consumed by the NVFP4
/// MTP converter.  The target artifact is never consulted for these values.
/// The JSON contract is:
///
/// ```json
/// {
///   "schema": "qwen38-mtp-nvfp4-activation-scale-v1",
///   "scale_rule": "f32(max_abs_bf16_activation / (6 * 448))",
///   "input_manifest_sha256": "sha256:<64 hex digits>",
///   "suite_sha256": "sha256:<64 hex digits>",
///   "source_report_sha256": "sha256:<64 hex digits>",
///   "source_reports": [{"target":"gfx1030","report_sha256":"sha256:<64 hex digits>"}],
///   "activation_amax": {"mtp.concat.output": 1.0 /* exactly five sites */},
///   "scales": { "mtp.fc.weight": 1.0 /* exactly eight matrix names */ }
/// }
/// ```
pub fn read_qwen38_mtp_nvfp4_activation_scale_manifest(
    path: &Path,
) -> Result<MtpNvfp4ActivationScaleManifest, MtpQuantizedSidecarError> {
    if !path.is_absolute() {
        return Err(MtpQuantizedSidecarError::invalid(
            "NVFP4 activation-scale manifest path must be absolute",
        ));
    }
    let bytes = bounded_read(path, MAX_MANIFEST_BYTES, "activation-scale manifest")?;
    let manifest: InputActivationScaleManifest =
        serde_json::from_slice(&bytes).map_err(|error| {
            MtpQuantizedSidecarError::invalid(format!("activation-scale manifest schema: {error}"))
        })?;
    if manifest.schema != NVFP4_ACTIVATION_SCALE_SCHEMA
        || manifest.scale_rule != NVFP4_CALIBRATION_SCALE_RULE
        || !is_sha256(&manifest.input_manifest_sha256)
        || !is_sha256(&manifest.suite_sha256)
        || !is_sha256(&manifest.source_report_sha256)
    {
        return Err(MtpQuantizedSidecarError::invalid(
            "NVFP4 activation-scale manifest identity or schema differs",
        ));
    }
    if manifest.source_reports.is_empty()
        || manifest.source_reports.len() > 2
        || manifest.source_reports.iter().any(|report| {
            !matches!(report.target.as_str(), "gfx1030" | "gfx1201")
                || !is_sha256(&report.report_sha256)
        })
        || manifest
            .source_reports
            .iter()
            .map(|report| report.target.as_str())
            .collect::<BTreeSet<_>>()
            .len()
            != manifest.source_reports.len()
    {
        return Err(MtpQuantizedSidecarError::invalid(
            "NVFP4 activation-scale source reports are malformed",
        ));
    }
    if source_report_digest(&manifest.source_reports) != manifest.source_report_sha256 {
        return Err(MtpQuantizedSidecarError::invalid(
            "NVFP4 activation-scale source report digest differs",
        ));
    }
    let mut activation_amax = BTreeMap::new();
    let expected_sites: BTreeSet<&str> = NVFP4_ACTIVATION_SITES.into_iter().collect();
    if manifest
        .activation_amax
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != expected_sites
    {
        return Err(MtpQuantizedSidecarError::invalid(
            "NVFP4 activation-amax site set differs",
        ));
    }
    for site in NVFP4_ACTIVATION_SITES {
        let value = manifest.activation_amax.get(site).ok_or_else(|| {
            MtpQuantizedSidecarError::invalid("NVFP4 activation-amax site is absent")
        })?;
        let value = value.as_f64().ok_or_else(|| {
            MtpQuantizedSidecarError::invalid(format!(
                "NVFP4 activation-amax is not a JSON number: {site}"
            ))
        })?;
        if !value.is_finite() || value <= 0.0 {
            return Err(MtpQuantizedSidecarError::invalid(format!(
                "NVFP4 activation-amax is not finite and positive: {site}"
            )));
        }
        activation_amax.insert(site.to_owned(), value);
    }
    let expected_names: BTreeSet<&str> = MTP_MATRIX_NAMES.into_iter().collect();
    if manifest
        .scales
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != expected_names
    {
        return Err(MtpQuantizedSidecarError::invalid(
            "NVFP4 activation-scale manifest matrix set differs",
        ));
    }
    let mut scales = BTreeMap::new();
    for name in MTP_MATRIX_NAMES {
        let value = manifest
            .scales
            .get(name)
            .ok_or_else(|| MtpQuantizedSidecarError::invalid("NVFP4 activation scale is absent"))?;
        let value = value.as_f64().ok_or_else(|| {
            MtpQuantizedSidecarError::invalid(format!(
                "NVFP4 activation scale is not a JSON number: {name}"
            ))
        })?;
        let scale = value as f32;
        if !value.is_finite() || !scale.is_finite() || scale <= 0.0 {
            return Err(MtpQuantizedSidecarError::invalid(format!(
                "NVFP4 activation scale is not finite and positive: {name}"
            )));
        }
        scales.insert(name.to_owned(), scale.to_bits());
    }
    validate_nvfp4_activation_scale_groups(&scales)?;
    for (name, bits) in &scales {
        let site = nvfp4_activation_site(name).ok_or_else(|| {
            MtpQuantizedSidecarError::invalid("NVFP4 activation scale matrix is unknown")
        })?;
        let expected = (activation_amax[site] / (6.0_f64 * 448.0_f64)) as f32;
        let expected = expected.to_bits();
        if *bits != expected {
            return Err(MtpQuantizedSidecarError::invalid(format!(
                "NVFP4 activation scale does not match activation-amax: {name}"
            )));
        }
    }
    Ok(MtpNvfp4ActivationScaleManifest {
        schema: manifest.schema,
        scale_rule: manifest.scale_rule,
        input_manifest_sha256: manifest.input_manifest_sha256,
        suite_sha256: manifest.suite_sha256,
        source_report_sha256: manifest.source_report_sha256,
        source_reports: manifest.source_reports,
        activation_amax,
        scales,
    })
}

fn nvfp4_activation_site(name: &str) -> Option<&'static str> {
    match name {
        "mtp.fc.weight" => Some("mtp.concat.output"),
        "mtp.layers.0.self_attn.q_proj.weight"
        | "mtp.layers.0.self_attn.k_proj.weight"
        | "mtp.layers.0.self_attn.v_proj.weight" => Some("layer.64.input_rmsnorm.output"),
        "mtp.layers.0.self_attn.o_proj.weight" => Some("layer.64.full.sigmoid_mul.output"),
        "mtp.layers.0.mlp.gate_proj.weight" | "mtp.layers.0.mlp.up_proj.weight" => {
            Some("layer.64.post_attention_rmsnorm.output")
        }
        "mtp.layers.0.mlp.down_proj.weight" => Some("layer.64.mlp.silu_mul.output"),
        _ => None,
    }
}

fn source_report_digest(reports: &[CalibrationSourceReport]) -> String {
    let mut hashes = reports
        .iter()
        .map(|report| report.report_sha256.as_str())
        .collect::<Vec<_>>();
    hashes.sort_unstable();
    sha256_bytes(hashes.join("\n").as_bytes())
}

fn validate_nvfp4_activation_scale_groups(
    scales: &BTreeMap<String, u32>,
) -> Result<(), MtpQuantizedSidecarError> {
    let equal = |names: &[&str], label: &str| {
        let first = scales
            .get(names[0])
            .copied()
            .ok_or_else(|| MtpQuantizedSidecarError::invalid("NVFP4 activation scale is absent"))?;
        if names
            .iter()
            .skip(1)
            .any(|name| scales.get(*name).copied() != Some(first))
        {
            return Err(MtpQuantizedSidecarError::invalid(format!(
                "NVFP4 activation scales for {label} must be identical"
            )));
        }
        Ok(())
    };
    equal(
        &[
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
        ],
        "Q/K/V",
    )?;
    equal(
        &[
            "mtp.layers.0.mlp.gate_proj.weight",
            "mtp.layers.0.mlp.up_proj.weight",
        ],
        "gate/up",
    )?;
    Ok(())
}

/// Verify a generated Qwen3.8 MTP MXFP sidecar against the immutable source
/// artifact and the model lock.
pub fn verify_qwen38_mtp_quantized_sidecar(
    lock: &ModelLock,
    artifact: &VerifiedUnslothQwen38Nvfp4,
    manifest_path: &Path,
    payload_path: &Path,
) -> Result<VerifiedQwen38MtpQuantizedSidecar, MtpQuantizedSidecarError> {
    validate_qwen38_mtp_artifact(lock, artifact)
        .map_err(|error| MtpQuantizedSidecarError::invalid(error.to_string()))?;
    let manifest_bytes = bounded_read(manifest_path, MAX_MANIFEST_BYTES, "manifest")?;
    let mut manifest_value: Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| MtpQuantizedSidecarError::invalid(format!("manifest JSON: {error}")))?;
    let claimed_fingerprint = manifest_value
        .as_object_mut()
        .and_then(|object| object.remove("fingerprint"))
        .and_then(|value| value.as_str().map(str::to_owned))
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("manifest fingerprint is absent"))?;
    let canonical = serde_json::to_vec(&manifest_value).map_err(|error| {
        MtpQuantizedSidecarError::invalid(format!("manifest canonicalization: {error}"))
    })?;
    if claimed_fingerprint != sha256_bytes(&canonical) {
        return Err(MtpQuantizedSidecarError::invalid(
            "manifest fingerprint differs",
        ));
    }
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| MtpQuantizedSidecarError::invalid(format!("manifest schema: {error}")))?;
    if manifest.fingerprint != claimed_fingerprint {
        return Err(MtpQuantizedSidecarError::invalid(
            "manifest fingerprint fields differ",
        ));
    }
    let (encoding, roundtrip_encoding) = parse_manifest_encoding(&manifest.encoding)?;
    let is_nvfp4 = encoding == MtpWeightEncoding::Nvfp4W4A4Block16E2M1E4M3FnF32;
    if (is_nvfp4 && roundtrip_encoding.is_some())
        || (!is_nvfp4 && (encoding == MtpWeightEncoding::Bf16 && roundtrip_encoding.is_none()))
        || ((is_nvfp4 && manifest.schema_version != NVFP4_SCHEMA)
            || (!is_nvfp4 && manifest.schema_version != SCHEMA))
        || (roundtrip_encoding.is_none() && manifest.converter != "sllm-qwen38-mtp-quantizer-v1")
        || (roundtrip_encoding.is_some()
            && manifest.converter != "sllm-qwen38-mtp-bf16-roundtrip-v1")
        || manifest.source.repository != UNSLOTH_QWEN38_NVFP4_REPOSITORY
        || manifest.source.resolved_revision != UNSLOTH_QWEN38_NVFP4_REVISION
        || manifest.source.lock_fingerprint != lock.fingerprint()
        || manifest.source.mtp_sha256 != UNSLOTH_QWEN38_NVFP4_MTP_SHA256
        || manifest.base_recipe_digest != artifact.recipe_digest()
        || !is_sha256(&manifest.base_recipe_digest)
        || !is_sha256(&manifest.source.lock_fingerprint)
    {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar source or recipe identity differs",
        ));
    }
    let output_activation_scales = if is_nvfp4 {
        let scales = manifest.activation_scales.as_ref().ok_or_else(|| {
            MtpQuantizedSidecarError::invalid(
                "NVFP4 sidecar activation calibration manifest is absent",
            )
        })?;
        validate_output_activation_scales(scales)?;
        Some(scales)
    } else {
        if manifest.activation_scales.is_some() {
            return Err(MtpQuantizedSidecarError::invalid(
                "MX/BF16 sidecar unexpectedly contains activation calibration",
            ));
        }
        None
    };
    let payload_name = payload_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("payload has no UTF-8 basename"))?;
    let metadata = payload_path
        .metadata()
        .map_err(|error| MtpQuantizedSidecarError::io("stat payload", error))?;
    if !metadata.is_file()
        || payload_name != manifest.payload.path
        || metadata.len() != manifest.payload.size_bytes
        || manifest.payload.size_bytes < 8
        || manifest.payload.sha256 != sha256_file(payload_path)?
    {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar payload identity differs",
        ));
    }
    let (data_start, header) = read_safetensors_header(payload_path)?;
    validate_payload_ranges(&header, metadata.len(), data_start)?;
    if manifest.tensors.len() != MTP_MATRIX_NAMES.len() {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar tensor count differs",
        ));
    }
    let mut records = BTreeMap::new();
    let expected_names: BTreeSet<&str> = MTP_MATRIX_NAMES.into_iter().collect();
    for record in manifest.tensors {
        let shape: [u64; 2] =
            record.logical_shape.as_slice().try_into().map_err(|_| {
                MtpQuantizedSidecarError::invalid("MTP sidecar shape is not rank two")
            })?;
        if expected_matrix_shape(&record.name) != Some(shape)
            || shape[0] == 0
            || shape[1] == 0
            || shape[1] % 16 != 0
        {
            return Err(MtpQuantizedSidecarError::invalid(
                "MTP sidecar matrix shape differs from the Qwen3.8 contract",
            ));
        }
        if !expected_names.contains(record.name.as_str())
            || records.insert(record.name.clone(), record).is_some()
        {
            return Err(MtpQuantizedSidecarError::invalid(
                "MTP sidecar matrix name set differs",
            ));
        }
    }
    if records.keys().map(String::as_str).collect::<BTreeSet<_>>() != expected_names {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar matrix coverage differs",
        ));
    }
    let mut payload_file = File::open(payload_path)
        .map_err(|error| MtpQuantizedSidecarError::io("open payload", error))?;
    let mut tensors = BTreeMap::new();
    let mut header_names = BTreeSet::new();
    for (name, record) in records {
        let shape = record
            .logical_shape
            .as_slice()
            .try_into()
            .map_err(|_| MtpQuantizedSidecarError::invalid("MTP sidecar shape is invalid"))?;
        let value_name = name.as_str();
        let scale_name = if is_nvfp4 {
            format!("{name}{NVFP4_BLOCK_SCALE_SUFFIX}")
        } else {
            format!("{name}{SCALE_SUFFIX}")
        };
        let value = header
            .get(value_name)
            .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP value tensor is absent"))?;
        let scale = header
            .get(scale_name.as_str())
            .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP scale tensor is absent"))?;
        let tensor_scale = if is_nvfp4 {
            let tensor_scale_name = format!("{name}{NVFP4_TENSOR_SCALE_SUFFIX}");
            let tensor_scale = header.get(tensor_scale_name.as_str()).ok_or_else(|| {
                MtpQuantizedSidecarError::invalid("MTP NVFP4 tensor scale is absent")
            })?;
            validate_nvfp4_record(
                &record,
                value,
                scale,
                tensor_scale,
                shape,
                &mut payload_file,
                data_start,
            )?;
            Some(tensor_scale)
        } else {
            validate_record(
                &record,
                value,
                scale,
                shape,
                encoding,
                roundtrip_encoding.is_some(),
                &mut payload_file,
                data_start,
            )?;
            None
        };
        if let Some(roundtrip) = &roundtrip_encoding {
            let diagnostic = record.roundtrip.as_ref().ok_or_else(|| {
                MtpQuantizedSidecarError::invalid(
                    "BF16 roundtrip diagnostic is absent from a roundtrip tensor",
                )
            })?;
            if diagnostic.recipe != roundtrip.manifest_name() {
                return Err(MtpQuantizedSidecarError::invalid(
                    "BF16 roundtrip diagnostic recipe differs",
                ));
            }
        }
        let descriptor = artifact
            .tensor(name.as_str())
            .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP source tensor is absent"))?;
        let source = artifact
            .read_source_range(&descriptor.source_name, descriptor.value_range)
            .map_err(|error| MtpQuantizedSidecarError::invalid(error.to_string()))?;
        if sha256_bytes(&source) != record.source_sha256 {
            return Err(MtpQuantizedSidecarError::invalid(
                "MTP sidecar source tensor differs from the locked BF16 artifact",
            ));
        }
        let (tensor_scale_range, tensor_scale_sha256, nvfp4_weight_tensor_scale_f32_bits) =
            if let Some(tensor_scale) = tensor_scale {
                let bytes = read_range(&payload_file, data_start, tensor_scale.data_offsets)?;
                let bits: [u8; 4] = bytes.clone().try_into().map_err(|_| {
                    MtpQuantizedSidecarError::invalid("MTP NVFP4 tensor scale is not four bytes")
                })?;
                (
                    Some(tensor_scale.data_offsets),
                    Some(record.tensor_scale_sha256.clone().ok_or_else(|| {
                        MtpQuantizedSidecarError::invalid("MTP NVFP4 tensor scale hash is absent")
                    })?),
                    Some(f32::from_le_bytes(bits).to_bits()),
                )
            } else {
                (None, None, None)
            };
        let nvfp4_input_global_scale_f32_bits = if is_nvfp4 {
            let expected = output_activation_scales
                .and_then(|scales| scales.scales.get(&name).copied())
                .ok_or_else(|| {
                    MtpQuantizedSidecarError::invalid("MTP NVFP4 input activation scale is absent")
                })?;
            if record.input_global_scale_f32_bits != Some(expected) {
                return Err(MtpQuantizedSidecarError::invalid(
                    "MTP NVFP4 input activation scale differs from calibration manifest",
                ));
            }
            Some(expected)
        } else {
            if record.tensor_scale_sha256.is_some() || record.input_global_scale_f32_bits.is_some()
            {
                return Err(MtpQuantizedSidecarError::invalid(
                    "MX/BF16 sidecar unexpectedly contains NVFP4 scale metadata",
                ));
            }
            None
        };
        let tensor = MtpQuantizedSidecarTensor {
            name: name.clone(),
            logical_shape: shape,
            value_range: value.data_offsets,
            scale_range: scale.data_offsets,
            tensor_scale_range,
            source_sha256: record.source_sha256.clone(),
            value_sha256: record.value_sha256.clone(),
            scale_sha256: record.scale_sha256.clone(),
            tensor_scale_sha256,
            nvfp4_weight_tensor_scale_f32_bits,
            nvfp4_input_global_scale_f32_bits,
        };
        tensors.insert(name.clone(), tensor);
        header_names.insert(name.clone());
        header_names.insert(scale_name);
        if let Some(tensor_scale) = tensor_scale {
            header_names.insert(format!("{name}{NVFP4_TENSOR_SCALE_SUFFIX}"));
            let _ = tensor_scale;
        }
    }
    let expected_header_names: BTreeSet<String> = MTP_MATRIX_NAMES
        .into_iter()
        .flat_map(|name| {
            if is_nvfp4 {
                vec![
                    name.to_owned(),
                    format!("{name}{NVFP4_BLOCK_SCALE_SUFFIX}"),
                    format!("{name}{NVFP4_TENSOR_SCALE_SUFFIX}"),
                ]
            } else {
                vec![name.to_owned(), format!("{name}{SCALE_SUFFIX}")]
            }
        })
        .collect();
    if header_names != expected_header_names
        || header
            .keys()
            .filter(|name| name.as_str() != "__metadata__")
            .cloned()
            .collect::<BTreeSet<_>>()
            != expected_header_names
    {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar payload tensor set differs",
        ));
    }
    Ok(VerifiedQwen38MtpQuantizedSidecar {
        manifest_path: manifest_path.to_path_buf(),
        payload_path: payload_path.to_path_buf(),
        payload: Arc::new(
            File::open(payload_path)
                .map_err(|error| MtpQuantizedSidecarError::io("open payload", error))?,
        ),
        source_lock_fingerprint: manifest.source.lock_fingerprint,
        source_mtp_sha256: manifest.source.mtp_sha256,
        base_recipe_digest: manifest.base_recipe_digest,
        manifest_fingerprint: claimed_fingerprint,
        manifest_encoding: manifest.encoding,
        roundtrip_encoding,
        encoding,
        data_start,
        tensors,
    })
}

fn validate_output_activation_scales(
    scales: &OutputActivationScales,
) -> Result<(), MtpQuantizedSidecarError> {
    if scales.schema != NVFP4_ACTIVATION_SCALE_SCHEMA
        || scales.scale_rule != NVFP4_CALIBRATION_SCALE_RULE
        || scales.input_scale_convention != NVFP4_INPUT_SCALE_CONVENTION
        || !is_sha256(&scales.input_manifest_sha256)
        || !is_sha256(&scales.suite_sha256)
        || !is_sha256(&scales.source_report_sha256)
        || scales.source_reports.is_empty()
        || scales.source_reports.len() > 2
        || scales.source_reports.iter().any(|report| {
            !matches!(report.target.as_str(), "gfx1030" | "gfx1201")
                || !is_sha256(&report.report_sha256)
        })
        || scales
            .source_reports
            .iter()
            .map(|report| report.target.as_str())
            .collect::<BTreeSet<_>>()
            .len()
            != scales.source_reports.len()
        || source_report_digest(&scales.source_reports) != scales.source_report_sha256
    {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP NVFP4 activation calibration identity differs",
        ));
    }
    let expected_names: BTreeSet<&str> = MTP_MATRIX_NAMES.into_iter().collect();
    if scales
        .scales
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != expected_names
    {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP NVFP4 activation calibration matrix set differs",
        ));
    }
    for (name, bits) in &scales.scales {
        let value = f32::from_bits(*bits);
        if !value.is_finite() || value <= 0.0 {
            return Err(MtpQuantizedSidecarError::invalid(format!(
                "MTP NVFP4 activation scale is non-positive or non-finite: {name}"
            )));
        }
        let site = nvfp4_activation_site(name).ok_or_else(|| {
            MtpQuantizedSidecarError::invalid("MTP NVFP4 activation scale matrix is unknown")
        })?;
        let amax = scales.activation_amax.get(site).copied().ok_or_else(|| {
            MtpQuantizedSidecarError::invalid("MTP NVFP4 activation-amax site is absent")
        })?;
        let expected = (amax / (6.0_f64 * 448.0_f64)) as f32;
        let expected = expected.to_bits();
        if *bits != expected {
            return Err(MtpQuantizedSidecarError::invalid(format!(
                "MTP NVFP4 activation scale does not match activation-amax: {name}"
            )));
        }
    }
    let expected_sites: BTreeSet<&str> = NVFP4_ACTIVATION_SITES.into_iter().collect();
    if scales
        .activation_amax
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != expected_sites
        || scales
            .activation_amax
            .values()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP NVFP4 activation-amax site set or values differ",
        ));
    }
    validate_nvfp4_activation_scale_groups(&scales.scales)
}

/// Convert the eight verified BF16 MTP matrices and publish a complete
/// sidecar directory atomically.  The returned object has already been
/// verified by rereading the published payload and manifest.
pub fn convert_qwen38_mtp_quantized_sidecar(
    lock: &ModelLock,
    artifact: &VerifiedUnslothQwen38Nvfp4,
    encoding: MtpWeightEncoding,
    output_dir: &Path,
) -> Result<VerifiedQwen38MtpQuantizedSidecar, MtpQuantizedSidecarError> {
    if encoding == MtpWeightEncoding::Bf16 {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar conversion requires MXFP8",
        ));
    }
    if encoding == MtpWeightEncoding::Nvfp4W4A4Block16E2M1E4M3FnF32 {
        return Err(MtpQuantizedSidecarError::invalid(
            "NVFP4 MTP sidecar conversion requires an activation-scale manifest",
        ));
    }
    validate_qwen38_mtp_artifact(lock, artifact)
        .map_err(|error| MtpQuantizedSidecarError::invalid(error.to_string()))?;
    if output_dir.exists() {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar output directory already exists",
        ));
    }
    let parent = output_dir
        .parent()
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP sidecar output has no parent"))?;
    fs::create_dir_all(parent)
        .map_err(|error| MtpQuantizedSidecarError::io("create output parent", error))?;
    let name = output_dir
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP sidecar output has no basename"))?;
    let temporary = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    if temporary.exists() {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar temporary output already exists",
        ));
    }
    fs::create_dir(&temporary)
        .map_err(|error| MtpQuantizedSidecarError::io("create temporary output", error))?;
    let result =
        convert_into_directory(lock, artifact, encoding, None, None, &temporary).and_then(|()| {
            verify_qwen38_mtp_quantized_sidecar(
                lock,
                artifact,
                &temporary.join(MANIFEST_FILE),
                &temporary.join(PAYLOAD_FILE),
            )
            .map(|_| ())?;
            fs::rename(&temporary, output_dir)
                .map_err(|error| MtpQuantizedSidecarError::io("publish MTP sidecar", error))
        });
    if result.is_err() {
        let _ = fs::remove_dir_all(&temporary);
    }
    result?;
    let verified = verify_qwen38_mtp_quantized_sidecar(
        lock,
        artifact,
        &output_dir.join(MANIFEST_FILE),
        &output_dir.join(PAYLOAD_FILE),
    );
    if verified.is_err() {
        let _ = fs::remove_dir_all(output_dir);
    }
    verified
}

/// Convert the eight verified BF16 MTP matrices to NVFP4 W4A4.  The
/// activation calibration manifest is mandatory and supplies resident `g` values
/// for all eight matrices; no scale is inferred from the target artifact.
pub fn convert_qwen38_mtp_nvfp4_sidecar(
    lock: &ModelLock,
    artifact: &VerifiedUnslothQwen38Nvfp4,
    activation_scale_manifest_path: &Path,
    output_dir: &Path,
) -> Result<VerifiedQwen38MtpQuantizedSidecar, MtpQuantizedSidecarError> {
    let activation_scales =
        read_qwen38_mtp_nvfp4_activation_scale_manifest(activation_scale_manifest_path)?;
    validate_qwen38_mtp_artifact(lock, artifact)
        .map_err(|error| MtpQuantizedSidecarError::invalid(error.to_string()))?;
    if output_dir.exists() {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar output directory already exists",
        ));
    }
    let parent = output_dir
        .parent()
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP sidecar output has no parent"))?;
    fs::create_dir_all(parent)
        .map_err(|error| MtpQuantizedSidecarError::io("create output parent", error))?;
    let name = output_dir
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP sidecar output has no basename"))?;
    let temporary = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    if temporary.exists() {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar temporary output already exists",
        ));
    }
    fs::create_dir(&temporary)
        .map_err(|error| MtpQuantizedSidecarError::io("create temporary output", error))?;
    let result = convert_into_directory(
        lock,
        artifact,
        MtpWeightEncoding::Nvfp4W4A4Block16E2M1E4M3FnF32,
        None,
        Some(&activation_scales),
        &temporary,
    )
    .and_then(|()| {
        verify_qwen38_mtp_quantized_sidecar(
            lock,
            artifact,
            &temporary.join(MANIFEST_FILE),
            &temporary.join(PAYLOAD_FILE),
        )
        .map(|_| ())?;
        fs::rename(&temporary, output_dir)
            .map_err(|error| MtpQuantizedSidecarError::io("publish MTP sidecar", error))
    });
    if result.is_err() {
        let _ = fs::remove_dir_all(&temporary);
    }
    result?;
    let verified = verify_qwen38_mtp_quantized_sidecar(
        lock,
        artifact,
        &output_dir.join(MANIFEST_FILE),
        &output_dir.join(PAYLOAD_FILE),
    );
    if verified.is_err() {
        let _ = fs::remove_dir_all(output_dir);
    }
    verified
}

/// Convert the verified BF16 MTP matrices through an MXFP8 fake-quant
/// path and write the BF16 dequantized values.  The sidecar keeps an empty
/// scale plane so the existing value+scale upload contract remains intact;
/// its manifest encoding and recipe digest are independent of the normal MX
/// sidecars.
pub fn convert_qwen38_mtp_bf16_roundtrip_sidecar(
    lock: &ModelLock,
    artifact: &VerifiedUnslothQwen38Nvfp4,
    roundtrip: MtpBf16RoundtripEncoding,
    output_dir: &Path,
) -> Result<VerifiedQwen38MtpQuantizedSidecar, MtpQuantizedSidecarError> {
    validate_qwen38_mtp_artifact(lock, artifact)
        .map_err(|error| MtpQuantizedSidecarError::invalid(error.to_string()))?;
    if output_dir.exists() {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar output directory already exists",
        ));
    }
    let parent = output_dir
        .parent()
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP sidecar output has no parent"))?;
    fs::create_dir_all(parent)
        .map_err(|error| MtpQuantizedSidecarError::io("create output parent", error))?;
    let name = output_dir
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP sidecar output has no basename"))?;
    let temporary = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    if temporary.exists() {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar temporary output already exists",
        ));
    }
    fs::create_dir(&temporary)
        .map_err(|error| MtpQuantizedSidecarError::io("create temporary output", error))?;
    let result = convert_into_directory(
        lock,
        artifact,
        MtpWeightEncoding::Bf16,
        Some(roundtrip),
        None,
        &temporary,
    )
    .and_then(|()| {
        verify_qwen38_mtp_quantized_sidecar(
            lock,
            artifact,
            &temporary.join(MANIFEST_FILE),
            &temporary.join(PAYLOAD_FILE),
        )
        .map(|_| ())?;
        fs::rename(&temporary, output_dir)
            .map_err(|error| MtpQuantizedSidecarError::io("publish MTP sidecar", error))
    });
    if result.is_err() {
        let _ = fs::remove_dir_all(&temporary);
    }
    result?;
    let verified = verify_qwen38_mtp_quantized_sidecar(
        lock,
        artifact,
        &output_dir.join(MANIFEST_FILE),
        &output_dir.join(PAYLOAD_FILE),
    );
    if verified.is_err() {
        let _ = fs::remove_dir_all(output_dir);
    }
    verified
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff_u32 + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding)) >> 16) as u16
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn push_position(positions: &mut Vec<u64>, index: usize) {
    if positions.len() < 16 {
        positions.push(index as u64);
    }
}

fn bf16_roundtrip_values(
    quantized: &QuantizedMx,
    source: &[f32],
    recipe: MtpBf16RoundtripEncoding,
) -> Result<(Vec<u8>, Bf16RoundtripDiagnostics), MtpQuantizedSidecarError> {
    let dequantized = quantized
        .dequantize()
        .map_err(|error| MtpQuantizedSidecarError::invalid(error.to_string()))?;
    if dequantized.len() != source.len() {
        return Err(MtpQuantizedSidecarError::invalid(
            "BF16 roundtrip dequantized element count differs",
        ));
    }
    let mut output = Vec::with_capacity(source.len() * 2);
    let mut diagnostics = Bf16RoundtripDiagnostics {
        recipe: recipe.manifest_name().to_owned(),
        element_count: source.len() as u64,
        bit_exact: true,
        source_nonfinite_count: 0,
        dequant_nonfinite_count: 0,
        roundtrip_nonfinite_count: 0,
        underflow_count: 0,
        overflow_count: 0,
        bit_mismatch_count: 0,
        first_source_nonfinite_positions: Vec::new(),
        first_dequant_nonfinite_positions: Vec::new(),
        first_roundtrip_nonfinite_positions: Vec::new(),
        first_underflow_positions: Vec::new(),
        first_overflow_positions: Vec::new(),
        first_bit_mismatch_positions: Vec::new(),
    };
    for (index, (&source_value, &dequantized_value)) in source.iter().zip(&dequantized).enumerate()
    {
        if !source_value.is_finite() {
            diagnostics.source_nonfinite_count += 1;
            push_position(&mut diagnostics.first_source_nonfinite_positions, index);
        }
        if !dequantized_value.is_finite() {
            diagnostics.dequant_nonfinite_count += 1;
            push_position(&mut diagnostics.first_dequant_nonfinite_positions, index);
        }
        let bits = f32_to_bf16_rne(dequantized_value);
        let roundtrip_value = bf16_to_f32(bits);
        output.extend_from_slice(&bits.to_le_bytes());
        if !roundtrip_value.is_finite() {
            diagnostics.roundtrip_nonfinite_count += 1;
            push_position(&mut diagnostics.first_roundtrip_nonfinite_positions, index);
        }
        if dequantized_value.is_finite() && dequantized_value != 0.0 && roundtrip_value == 0.0 {
            diagnostics.underflow_count += 1;
            push_position(&mut diagnostics.first_underflow_positions, index);
        }
        if dequantized_value.is_finite() && !roundtrip_value.is_finite() {
            diagnostics.overflow_count += 1;
            push_position(&mut diagnostics.first_overflow_positions, index);
        }
        if dequantized_value.is_finite() && roundtrip_value.to_bits() != dequantized_value.to_bits()
        {
            diagnostics.bit_mismatch_count += 1;
            push_position(&mut diagnostics.first_bit_mismatch_positions, index);
        }
    }
    diagnostics.bit_exact = diagnostics.source_nonfinite_count == 0
        && diagnostics.dequant_nonfinite_count == 0
        && diagnostics.roundtrip_nonfinite_count == 0
        && diagnostics.underflow_count == 0
        && diagnostics.overflow_count == 0
        && diagnostics.bit_mismatch_count == 0;
    Ok((output, diagnostics))
}

fn quantize_mtp_matrix(
    encoding: MtpWeightEncoding,
    values: &[f32],
    rows: usize,
    columns: usize,
) -> Result<QuantizedMx, MtpQuantizedSidecarError> {
    let quantized = match encoding {
        MtpWeightEncoding::Mxfp8W8A8Block32E8M0 => quantize_mxfp8_e4m3(values, rows, columns),
        MtpWeightEncoding::Mxfp8W8A8Block32E8M0NoClippingScale => {
            quantize_mxfp8_e4m3_no_clipping_scale(values, rows, columns)
        }
        MtpWeightEncoding::Bf16 => {
            return Err(MtpQuantizedSidecarError::invalid(
                "BF16 MTP sidecars do not use MX quantization",
            ));
        }
        MtpWeightEncoding::Nvfp4W4A4Block16E2M1E4M3FnF32 => {
            return Err(MtpQuantizedSidecarError::invalid(
                "NVFP4 MTP sidecars use the NVFP4 quantizer",
            ));
        }
    };
    quantized.map_err(|error| MtpQuantizedSidecarError::invalid(error.to_string()))
}

fn convert_into_directory(
    lock: &ModelLock,
    artifact: &VerifiedUnslothQwen38Nvfp4,
    encoding: MtpWeightEncoding,
    roundtrip: Option<MtpBf16RoundtripEncoding>,
    activation_scales: Option<&MtpNvfp4ActivationScaleManifest>,
    directory: &Path,
) -> Result<(), MtpQuantizedSidecarError> {
    if (encoding == MtpWeightEncoding::Bf16) != roundtrip.is_some() {
        return Err(MtpQuantizedSidecarError::invalid(
            "BF16 payload requires a roundtrip recipe and MX payloads do not accept one",
        ));
    }
    let is_nvfp4 = encoding == MtpWeightEncoding::Nvfp4W4A4Block16E2M1E4M3FnF32;
    if is_nvfp4 != activation_scales.is_some() {
        return Err(MtpQuantizedSidecarError::invalid(
            "NVFP4 sidecars require exactly one activation-scale manifest",
        ));
    }
    if let Some(activation_scales) = activation_scales {
        validate_nvfp4_activation_scale_groups(&activation_scales.scales)?;
    }
    let mut converted = Vec::with_capacity(MTP_MATRIX_NAMES.len());
    for name in MTP_MATRIX_NAMES {
        let descriptor = artifact.tensor(name).ok_or_else(|| {
            MtpQuantizedSidecarError::invalid(format!("MTP tensor is absent: {name}"))
        })?;
        if descriptor.encoding != QuantizedTensorEncoding::UnquantizedBf16
            || descriptor.logical_shape.len() != 2
            || descriptor.logical_shape[1] % 16 != 0
        {
            return Err(MtpQuantizedSidecarError::invalid(format!(
                "MTP tensor is not a block32 BF16 matrix: {name}"
            )));
        }
        let rows = usize::try_from(descriptor.logical_shape[0])
            .map_err(|_| MtpQuantizedSidecarError::invalid("MTP row count overflows usize"))?;
        let columns = usize::try_from(descriptor.logical_shape[1])
            .map_err(|_| MtpQuantizedSidecarError::invalid("MTP column count overflows usize"))?;
        let source = artifact
            .read_source_range(&descriptor.source_name, descriptor.value_range)
            .map_err(|error| MtpQuantizedSidecarError::invalid(error.to_string()))?;
        let expected_bytes = rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP source size overflows"))?;
        if source.len() != expected_bytes {
            return Err(MtpQuantizedSidecarError::invalid(format!(
                "MTP source byte size differs: {name}"
            )));
        }
        let values = source
            .chunks_exact(2)
            .map(|bytes| f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16))
            .collect::<Vec<_>>();
        let (
            values,
            scales,
            value_dtype,
            value_shape,
            scale_shape,
            roundtrip_diagnostics,
            tensor_scale,
            input_global_scale_f32_bits,
        ) = if let Some(roundtrip) = roundtrip {
            let quantized = roundtrip.quantize(&values, rows, columns)?;
            let (values, diagnostics) = bf16_roundtrip_values(&quantized, &values, roundtrip)?;
            (
                values,
                Vec::new(),
                "BF16",
                vec![rows as u64, columns as u64],
                vec![rows as u64, 0],
                Some(diagnostics),
                Vec::new(),
                None,
            )
        } else if is_nvfp4 {
            let activation_scales = activation_scales.expect("NVFP4 activation scales");
            let input_global_scale_f32_bits =
                activation_scales.scales.get(name).copied().ok_or_else(|| {
                    MtpQuantizedSidecarError::invalid(format!(
                        "NVFP4 activation scale is absent: {name}"
                    ))
                })?;
            let quantized = quantize_nvfp4_weights(&values, rows, columns)
                .map_err(|error| MtpQuantizedSidecarError::invalid(error.to_string()))?;
            (
                quantized.packed_values,
                quantized.block_scales,
                encoding.value_dtype(),
                vec![rows as u64, (columns / 2) as u64],
                vec![rows as u64, (columns / 16) as u64],
                None,
                quantized.tensor_scale.to_le_bytes().to_vec(),
                Some(input_global_scale_f32_bits),
            )
        } else {
            let quantized = quantize_mtp_matrix(encoding, &values, rows, columns)?;
            let value_shape = match encoding {
                MtpWeightEncoding::Mxfp8W8A8Block32E8M0
                | MtpWeightEncoding::Mxfp8W8A8Block32E8M0NoClippingScale => {
                    vec![rows as u64, columns as u64]
                }
                MtpWeightEncoding::Bf16 => unreachable!(),
                MtpWeightEncoding::Nvfp4W4A4Block16E2M1E4M3FnF32 => unreachable!(),
            };
            (
                quantized.values().to_vec(),
                quantized.scales().to_vec(),
                encoding.value_dtype(),
                value_shape,
                vec![rows as u64, (columns / 32) as u64],
                None,
                Vec::new(),
                None,
            )
        };
        converted.push(ConvertedTensor {
            record: OutputTensor {
                name: name.to_owned(),
                logical_shape: [rows as u64, columns as u64],
                source_sha256: sha256_bytes(&source),
                value_sha256: sha256_bytes(&values),
                scale_sha256: sha256_bytes(&scales),
                roundtrip: roundtrip_diagnostics,
                tensor_scale_sha256: if is_nvfp4 {
                    Some(sha256_bytes(&tensor_scale))
                } else {
                    None
                },
                input_global_scale_f32_bits,
            },
            value_dtype,
            value_shape,
            scale_shape,
            values,
            scales,
            tensor_scale,
        });
    }
    let mut data = Vec::new();
    let mut header = Map::new();
    let manifest_encoding = roundtrip.map_or_else(
        || encoding.manifest_name(),
        MtpBf16RoundtripEncoding::manifest_name,
    );
    for tensor in &converted {
        let value_start = u64::try_from(data.len())
            .map_err(|_| MtpQuantizedSidecarError::invalid("MTP payload offset overflows"))?;
        data.extend_from_slice(&tensor.values);
        let value_end = u64::try_from(data.len())
            .map_err(|_| MtpQuantizedSidecarError::invalid("MTP payload size overflows"))?;
        let scale_start = value_end;
        data.extend_from_slice(&tensor.scales);
        let scale_end = u64::try_from(data.len())
            .map_err(|_| MtpQuantizedSidecarError::invalid("MTP payload size overflows"))?;
        header.insert(
            tensor.record.name.clone(),
            serde_json::json!({
                "dtype": tensor.value_dtype,
                "shape": tensor.value_shape,
                "data_offsets": [value_start, value_end],
            }),
        );
        if is_nvfp4 {
            header.insert(
                format!("{}{NVFP4_BLOCK_SCALE_SUFFIX}", tensor.record.name),
                serde_json::json!({
                    "dtype": encoding.scale_dtype(),
                    "shape": tensor.scale_shape,
                    "data_offsets": [scale_start, scale_end],
                }),
            );
            let tensor_scale_start = scale_end;
            data.extend_from_slice(&tensor.tensor_scale);
            let tensor_scale_end = u64::try_from(data.len())
                .map_err(|_| MtpQuantizedSidecarError::invalid("MTP payload size overflows"))?;
            header.insert(
                format!("{}{NVFP4_TENSOR_SCALE_SUFFIX}", tensor.record.name),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [tensor_scale_start, tensor_scale_end],
                }),
            );
        } else {
            header.insert(
                format!("{}{SCALE_SUFFIX}", tensor.record.name),
                serde_json::json!({
                    "dtype": encoding.scale_dtype(),
                    "shape": tensor.scale_shape,
                    "data_offsets": [scale_start, scale_end],
                }),
            );
        }
    }
    header.insert(
        "__metadata__".to_owned(),
        serde_json::json!({
            "format": "pt",
            "sllm_schema": if is_nvfp4 { NVFP4_SCHEMA } else { SCHEMA },
            "sllm_encoding": manifest_encoding,
        }),
    );
    let header_bytes = serde_json::to_vec(&header).map_err(|error| {
        MtpQuantizedSidecarError::invalid(format!("payload header JSON: {error}"))
    })?;
    let header_length = u64::try_from(header_bytes.len())
        .map_err(|_| MtpQuantizedSidecarError::invalid("payload header size overflows"))?;
    if header_length > MAX_HEADER_BYTES {
        return Err(MtpQuantizedSidecarError::invalid(
            "payload header exceeds the bounded size",
        ));
    }
    let mut payload_bytes = Vec::with_capacity(8 + header_bytes.len() + data.len());
    payload_bytes.extend_from_slice(&header_length.to_le_bytes());
    payload_bytes.extend_from_slice(&header_bytes);
    payload_bytes.extend_from_slice(&data);
    let payload_path = directory.join(PAYLOAD_FILE);
    let manifest_path = directory.join(MANIFEST_FILE);
    write_new_file(&payload_path, &payload_bytes)?;
    let payload_digest = sha256_bytes(&payload_bytes);
    let tensors = converted
        .into_iter()
        .map(|tensor| OutputTensor {
            name: tensor.record.name,
            logical_shape: tensor.record.logical_shape,
            source_sha256: tensor.record.source_sha256,
            value_sha256: tensor.record.value_sha256,
            scale_sha256: tensor.record.scale_sha256,
            roundtrip: tensor.record.roundtrip,
            tensor_scale_sha256: tensor.record.tensor_scale_sha256,
            input_global_scale_f32_bits: tensor.record.input_global_scale_f32_bits,
        })
        .collect::<Vec<_>>();
    let base_recipe = artifact.recipe_digest();
    let lock_fingerprint = lock.fingerprint().to_owned();
    let payload_size = u64::try_from(payload_bytes.len())
        .map_err(|_| MtpQuantizedSidecarError::invalid("payload size overflows"))?;
    let output_activation_scales = activation_scales.map(|scales| OutputActivationScales {
        schema: scales.schema.clone(),
        scale_rule: scales.scale_rule.clone(),
        input_manifest_sha256: scales.input_manifest_sha256.clone(),
        suite_sha256: scales.suite_sha256.clone(),
        source_report_sha256: scales.source_report_sha256.clone(),
        source_reports: scales.source_reports.clone(),
        activation_amax: scales.activation_amax.clone(),
        input_scale_convention: NVFP4_INPUT_SCALE_CONVENTION.to_owned(),
        scales: scales.scales.clone(),
    });
    let manifest_without_fingerprint = OutputManifest {
        schema_version: if is_nvfp4 { NVFP4_SCHEMA } else { SCHEMA },
        source: OutputSource {
            repository: UNSLOTH_QWEN38_NVFP4_REPOSITORY,
            resolved_revision: UNSLOTH_QWEN38_NVFP4_REVISION,
            lock_fingerprint: &lock_fingerprint,
            mtp_sha256: UNSLOTH_QWEN38_NVFP4_MTP_SHA256,
        },
        converter: if roundtrip.is_some() {
            "sllm-qwen38-mtp-bf16-roundtrip-v1"
        } else {
            "sllm-qwen38-mtp-quantizer-v1"
        },
        encoding: manifest_encoding,
        base_recipe_digest: base_recipe,
        payload: OutputPayload {
            path: PAYLOAD_FILE,
            size_bytes: payload_size,
            sha256: payload_digest,
        },
        tensors: &tensors,
        fingerprint: String::new(),
        activation_scales: output_activation_scales.as_ref(),
    };
    let mut value = serde_json::to_value(&manifest_without_fingerprint).map_err(|error| {
        MtpQuantizedSidecarError::invalid(format!("manifest serialization: {error}"))
    })?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("manifest is not an object"))?;
    object.remove("fingerprint");
    let fingerprint = sha256_bytes(&serde_json::to_vec(&value).map_err(|error| {
        MtpQuantizedSidecarError::invalid(format!("manifest canonicalization: {error}"))
    })?);
    let mut final_value = value;
    final_value["fingerprint"] = Value::String(fingerprint);
    let manifest_bytes = serde_json::to_vec(&final_value)
        .map_err(|error| MtpQuantizedSidecarError::invalid(format!("manifest JSON: {error}")))?;
    write_new_file(&manifest_path, &manifest_bytes)
}

// Keep the explicit record, tensor-layout and payload bounds together.
#[allow(clippy::too_many_arguments)]
fn validate_record(
    record: &TensorRecord,
    value: &SafeTensorMetadata,
    scale: &SafeTensorMetadata,
    logical_shape: [u64; 2],
    encoding: MtpWeightEncoding,
    roundtrip: bool,
    file: &mut File,
    data_start: u64,
) -> Result<(), MtpQuantizedSidecarError> {
    if logical_shape[0] == 0 || logical_shape[1] == 0 || logical_shape[1] % 32 != 0 {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP sidecar matrix shape is not block32 aligned",
        ));
    }
    let values_len = value.data_offsets[1]
        .checked_sub(value.data_offsets[0])
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP value range is reversed"))?;
    let scales_len = scale.data_offsets[1]
        .checked_sub(scale.data_offsets[0])
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP scale range is reversed"))?;
    let expected_values = match encoding {
        MtpWeightEncoding::Mxfp8W8A8Block32E8M0
        | MtpWeightEncoding::Mxfp8W8A8Block32E8M0NoClippingScale => logical_shape[0]
            .checked_mul(logical_shape[1])
            .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP value shape overflows"))?,
        MtpWeightEncoding::Bf16 if roundtrip => logical_shape[0]
            .checked_mul(logical_shape[1])
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP value shape overflows"))?,
        MtpWeightEncoding::Bf16 => {
            return Err(MtpQuantizedSidecarError::invalid(
                "BF16 sidecar is unsupported",
            ));
        }
        MtpWeightEncoding::Nvfp4W4A4Block16E2M1E4M3FnF32 => {
            return Err(MtpQuantizedSidecarError::invalid(
                "NVFP4 sidecars use the three-plane validator",
            ));
        }
    };
    let expected_scales = if roundtrip {
        0
    } else {
        logical_shape[0]
            .checked_mul(logical_shape[1] / 32)
            .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP scale shape overflows"))?
    };
    if !is_sha256(&record.source_sha256)
        || !is_sha256(&record.value_sha256)
        || !is_sha256(&record.scale_sha256)
        || value.dtype != encoding.value_dtype()
        || scale.dtype != encoding.scale_dtype()
        || value.shape
            != match encoding {
                MtpWeightEncoding::Mxfp8W8A8Block32E8M0
                | MtpWeightEncoding::Mxfp8W8A8Block32E8M0NoClippingScale => {
                    vec![logical_shape[0], logical_shape[1]]
                }
                MtpWeightEncoding::Bf16 if roundtrip => {
                    vec![logical_shape[0], logical_shape[1]]
                }
                MtpWeightEncoding::Bf16 => unreachable!(),
                MtpWeightEncoding::Nvfp4W4A4Block16E2M1E4M3FnF32 => unreachable!(),
            }
        || scale.shape
            != if roundtrip {
                vec![logical_shape[0], 0]
            } else {
                vec![logical_shape[0], logical_shape[1] / 32]
            }
        || values_len != expected_values
        || scales_len != expected_scales
        || data_start.checked_add(value.data_offsets[1]).is_none()
        || data_start.checked_add(scale.data_offsets[1]).is_none()
        || sha256_range(file, data_start, value.data_offsets)? != record.value_sha256
        || sha256_range(file, data_start, scale.data_offsets)? != record.scale_sha256
    {
        return Err(MtpQuantizedSidecarError::invalid(format!(
            "MTP sidecar value/scale contract differs: {}",
            record.name
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_nvfp4_record(
    record: &TensorRecord,
    value: &SafeTensorMetadata,
    block_scale: &SafeTensorMetadata,
    tensor_scale: &SafeTensorMetadata,
    logical_shape: [u64; 2],
    file: &mut File,
    data_start: u64,
) -> Result<(), MtpQuantizedSidecarError> {
    if logical_shape[0] == 0 || logical_shape[1] == 0 || logical_shape[1] % 16 != 0 {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP NVFP4 matrix shape is not block16 aligned",
        ));
    }
    let values_len = value.data_offsets[1]
        .checked_sub(value.data_offsets[0])
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP NVFP4 value range is reversed"))?;
    let block_scales_len = block_scale.data_offsets[1]
        .checked_sub(block_scale.data_offsets[0])
        .ok_or_else(|| {
            MtpQuantizedSidecarError::invalid("MTP NVFP4 block-scale range is reversed")
        })?;
    let tensor_scale_len = tensor_scale.data_offsets[1]
        .checked_sub(tensor_scale.data_offsets[0])
        .ok_or_else(|| {
            MtpQuantizedSidecarError::invalid("MTP NVFP4 tensor-scale range is reversed")
        })?;
    let expected_values = logical_shape[0]
        .checked_mul(logical_shape[1] / 2)
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP NVFP4 value shape overflows"))?;
    let expected_block_scales = logical_shape[0]
        .checked_mul(logical_shape[1] / 16)
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("MTP NVFP4 scale shape overflows"))?;
    if record.roundtrip.is_some()
        || record.tensor_scale_sha256.is_none()
        || !is_sha256(&record.source_sha256)
        || !is_sha256(&record.value_sha256)
        || !is_sha256(&record.scale_sha256)
        || !is_sha256(record.tensor_scale_sha256.as_deref().unwrap_or_default())
        || value.dtype != "U8"
        || value.shape != vec![logical_shape[0], logical_shape[1] / 2]
        || block_scale.dtype != "U8"
        || block_scale.shape != vec![logical_shape[0], logical_shape[1] / 16]
        || tensor_scale.dtype != "F32"
        || tensor_scale.shape != vec![1]
        || values_len != expected_values
        || block_scales_len != expected_block_scales
        || tensor_scale_len != 4
        || data_start.checked_add(value.data_offsets[1]).is_none()
        || data_start
            .checked_add(block_scale.data_offsets[1])
            .is_none()
        || data_start
            .checked_add(tensor_scale.data_offsets[1])
            .is_none()
        || sha256_range(file, data_start, value.data_offsets)? != record.value_sha256
        || sha256_range(file, data_start, block_scale.data_offsets)? != record.scale_sha256
        || sha256_range(file, data_start, tensor_scale.data_offsets)?
            != record.tensor_scale_sha256.as_deref().unwrap_or_default()
    {
        return Err(MtpQuantizedSidecarError::invalid(format!(
            "MTP NVFP4 value/scale contract differs: {}",
            record.name
        )));
    }
    let tensor_scale_bytes = read_range(file, data_start, tensor_scale.data_offsets)?;
    let bits: [u8; 4] = tensor_scale_bytes.try_into().map_err(|_| {
        MtpQuantizedSidecarError::invalid("MTP NVFP4 tensor scale is not four bytes")
    })?;
    let tensor_scale_value = f32::from_le_bytes(bits);
    if !tensor_scale_value.is_finite() || tensor_scale_value <= 0.0 {
        return Err(MtpQuantizedSidecarError::invalid(
            "MTP NVFP4 tensor scale is non-positive or non-finite",
        ));
    }
    Ok(())
}

fn combined_recipe_digest(base: &str, manifest: &str, encoding: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(DIGEST_DOMAIN);
    digest.update(base.as_bytes());
    digest.update([0]);
    digest.update(manifest.as_bytes());
    digest.update([0]);
    digest.update(encoding.as_bytes());
    format!("sha256:{:x}", digest.finalize())
}

fn write_new_file(path: &Path, bytes: &[u8]) -> Result<(), MtpQuantizedSidecarError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| MtpQuantizedSidecarError::io("create sidecar file", error))?;
    file.write_all(bytes)
        .map_err(|error| MtpQuantizedSidecarError::io("write sidecar file", error))?;
    file.sync_all()
        .map_err(|error| MtpQuantizedSidecarError::io("sync sidecar file", error))?;
    let mut permissions = file
        .metadata()
        .map_err(|error| MtpQuantizedSidecarError::io("stat sidecar file", error))?
        .permissions();
    permissions.set_mode(0o600);
    fs::set_permissions(path, permissions)
        .map_err(|error| MtpQuantizedSidecarError::io("set sidecar permissions", error))
}

fn read_safetensors_header(
    path: &Path,
) -> Result<(u64, BTreeMap<String, SafeTensorMetadata>), MtpQuantizedSidecarError> {
    let mut file =
        File::open(path).map_err(|error| MtpQuantizedSidecarError::io("open payload", error))?;
    let mut raw_length = [0_u8; 8];
    file.read_exact(&mut raw_length)
        .map_err(|error| MtpQuantizedSidecarError::io("read payload header length", error))?;
    let length = u64::from_le_bytes(raw_length);
    if length == 0 || length > MAX_HEADER_BYTES {
        return Err(MtpQuantizedSidecarError::invalid(
            "payload header length is invalid",
        ));
    }
    let mut bytes = vec![
        0_u8;
        usize::try_from(length).map_err(|_| MtpQuantizedSidecarError::invalid(
            "payload header is too large"
        ))?
    ];
    file.read_exact(&mut bytes)
        .map_err(|error| MtpQuantizedSidecarError::io("read payload header", error))?;
    let mut value: Value = serde_json::from_slice(&bytes).map_err(|error| {
        MtpQuantizedSidecarError::invalid(format!("payload header JSON: {error}"))
    })?;
    value
        .as_object_mut()
        .and_then(|object| object.remove("__metadata__"));
    let header = serde_json::from_value(value).map_err(|error| {
        MtpQuantizedSidecarError::invalid(format!("payload header schema: {error}"))
    })?;
    Ok((8 + length, header))
}

fn validate_payload_ranges(
    header: &BTreeMap<String, SafeTensorMetadata>,
    payload_size: u64,
    data_start: u64,
) -> Result<(), MtpQuantizedSidecarError> {
    let data_size = payload_size
        .checked_sub(data_start)
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("payload data starts past EOF"))?;
    let mut ranges = Vec::with_capacity(header.len());
    for (name, metadata) in header {
        let [start, end] = metadata.data_offsets;
        if end < start || end > data_size {
            return Err(MtpQuantizedSidecarError::invalid(format!(
                "payload range is outside the file: {name}"
            )));
        }
        ranges.push((start, end, name));
    }
    ranges.sort_unstable_by_key(|(start, _, _)| *start);
    let mut cursor = 0_u64;
    for (start, end, name) in ranges {
        if start != cursor {
            return Err(MtpQuantizedSidecarError::invalid(format!(
                "payload ranges have a gap or overlap before {name}"
            )));
        }
        cursor = end;
    }
    if cursor != data_size {
        return Err(MtpQuantizedSidecarError::invalid(
            "payload ranges do not cover the complete data section",
        ));
    }
    Ok(())
}

fn bounded_read(
    path: &Path,
    maximum: u64,
    description: &str,
) -> Result<Vec<u8>, MtpQuantizedSidecarError> {
    let metadata = path
        .metadata()
        .map_err(|error| MtpQuantizedSidecarError::io(&format!("stat {description}"), error))?;
    if !metadata.is_file() || metadata.len() > maximum {
        return Err(MtpQuantizedSidecarError::invalid(format!(
            "{description} is absent or too large"
        )));
    }
    let mut file = File::open(path)
        .map_err(|error| MtpQuantizedSidecarError::io(&format!("open {description}"), error))?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.read_to_end(&mut bytes)
        .map_err(|error| MtpQuantizedSidecarError::io(&format!("read {description}"), error))?;
    Ok(bytes)
}

fn read_range(
    file: &File,
    data_start: u64,
    range: [u64; 2],
) -> Result<Vec<u8>, MtpQuantizedSidecarError> {
    let length = range[1]
        .checked_sub(range[0])
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("payload range is reversed"))?;
    if length > MAX_TENSOR_BYTES {
        return Err(MtpQuantizedSidecarError::invalid(
            "payload tensor is too large",
        ));
    }
    let absolute = data_start
        .checked_add(range[0])
        .ok_or_else(|| MtpQuantizedSidecarError::invalid("payload range overflows"))?;
    let mut bytes = vec![
        0_u8;
        usize::try_from(length).map_err(|_| MtpQuantizedSidecarError::invalid(
            "payload tensor does not fit usize"
        ))?
    ];
    read_exact_at(file, &mut bytes, absolute)?;
    Ok(bytes)
}

fn sha256_range(
    file: &File,
    data_start: u64,
    range: [u64; 2],
) -> Result<String, MtpQuantizedSidecarError> {
    let bytes = read_range(file, data_start, range)?;
    Ok(sha256_bytes(&bytes))
}

fn read_exact_at(
    file: &File,
    mut output: &mut [u8],
    mut offset: u64,
) -> Result<(), MtpQuantizedSidecarError> {
    while !output.is_empty() {
        let count = file
            .read_at(output, offset)
            .map_err(|error| MtpQuantizedSidecarError::io("read payload tensor", error))?;
        if count == 0 {
            return Err(MtpQuantizedSidecarError::invalid(
                "payload tensor is truncated",
            ));
        }
        output = &mut output[count..];
        offset = offset
            .checked_add(
                u64::try_from(count).map_err(|_| {
                    MtpQuantizedSidecarError::invalid("payload read offset overflows")
                })?,
            )
            .ok_or_else(|| MtpQuantizedSidecarError::invalid("payload read offset overflows"))?;
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, MtpQuantizedSidecarError> {
    let mut file = File::open(path)
        .map_err(|error| MtpQuantizedSidecarError::io("open hashed file", error))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| MtpQuantizedSidecarError::io("hash file", error))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("sha256:{:x}", digest.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.strip_prefix("sha256:").is_some_and(|digest| {
        digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn encoded_value_len(encoding: MtpWeightEncoding, rows: usize, columns: usize) -> usize {
        match encoding {
            MtpWeightEncoding::Bf16 => rows * columns * 2,
            MtpWeightEncoding::Mxfp8W8A8Block32E8M0
            | MtpWeightEncoding::Mxfp8W8A8Block32E8M0NoClippingScale => rows * columns,
            MtpWeightEncoding::Nvfp4W4A4Block16E2M1E4M3FnF32 => rows * columns / 2,
        }
    }

    fn temporary_test_path(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "sllm-mtp-sidecar-{label}-{}-{nanos}",
            std::process::id()
        ))
    }

    fn validate_fixture(
        encoding: MtpWeightEncoding,
        columns: usize,
        record: TensorRecord,
        value: SafeTensorMetadata,
        scale: SafeTensorMetadata,
        data: &[u8],
    ) -> Result<(), MtpQuantizedSidecarError> {
        let path = temporary_test_path("validate");
        {
            let mut output = File::create(&path).expect("create fixture");
            output.write_all(&[0_u8; 8]).expect("write fixture prefix");
            output.write_all(data).expect("write fixture payload");
            output.sync_all().expect("sync fixture");
        }
        let mut input = File::open(&path).expect("open fixture");
        let result = validate_record(
            &record,
            &value,
            &scale,
            [3, columns as u64],
            encoding,
            false,
            &mut input,
            8,
        );
        drop(input);
        std::fs::remove_file(path).expect("remove fixture");
        result
    }

    fn make_validate_fixture(
        encoding: MtpWeightEncoding,
        columns: usize,
    ) -> (
        TensorRecord,
        SafeTensorMetadata,
        SafeTensorMetadata,
        Vec<u8>,
    ) {
        let rows = 3_usize;
        let value_len = encoded_value_len(encoding, rows, columns);
        let scale_len = rows * (columns / 32);
        let values = vec![0x31_u8; value_len];
        let scales = vec![0x7f_u8; scale_len];
        let mut data = values.clone();
        data.extend_from_slice(&scales);
        let record = TensorRecord {
            name: "mtp.test.weight".to_owned(),
            logical_shape: vec![rows as u64, columns as u64],
            source_sha256: sha256_bytes(b"source"),
            value_sha256: sha256_bytes(&values),
            scale_sha256: sha256_bytes(&scales),
            roundtrip: None,
            tensor_scale_sha256: None,
            input_global_scale_f32_bits: None,
        };
        let value = SafeTensorMetadata {
            dtype: encoding.value_dtype().to_owned(),
            shape: match encoding {
                MtpWeightEncoding::Mxfp8W8A8Block32E8M0
                | MtpWeightEncoding::Mxfp8W8A8Block32E8M0NoClippingScale => {
                    vec![rows as u64, columns as u64]
                }
                MtpWeightEncoding::Bf16 => vec![rows as u64, columns as u64],
                MtpWeightEncoding::Nvfp4W4A4Block16E2M1E4M3FnF32 => {
                    vec![rows as u64, (columns / 2) as u64]
                }
            },
            data_offsets: [0, value_len as u64],
        };
        let scale = SafeTensorMetadata {
            dtype: encoding.scale_dtype().to_owned(),
            shape: vec![rows as u64, (columns / 32) as u64],
            data_offsets: [value_len as u64, (value_len + scale_len) as u64],
        };
        (record, value, scale, data)
    }

    #[test]
    fn matrix_contract_has_exact_eight_names_and_shapes() {
        assert_eq!(MTP_MATRIX_NAMES.len(), 8);
        for name in MTP_MATRIX_NAMES {
            let shape = expected_matrix_shape(name).expect("matrix shape");
            assert!(shape[0] > 0 && shape[1] > 0);
            assert_eq!(shape[1] % 32, 0);
        }
        assert_eq!(expected_matrix_shape("mtp.norm.weight"), None);
    }

    #[test]
    fn encoding_names_and_combined_recipe_identity_are_stable() {
        assert_eq!(MtpWeightEncoding::Bf16.manifest_name(), "bf16");
        assert_eq!(
            MtpWeightEncoding::Mxfp8W8A8Block32E8M0.manifest_name(),
            "mxfp8-w8a8-e4m3-block32-e8m0"
        );
        assert_eq!(
            MtpWeightEncoding::Mxfp8W8A8Block32E8M0NoClippingScale.manifest_name(),
            "mxfp8-w8a8-e4m3-block32-e8m0:mx-scale=no-clipping"
        );
        let base = "sha256:base";
        let manifest = "sha256:manifest";
        assert_ne!(
            combined_recipe_digest(
                base,
                manifest,
                MtpWeightEncoding::Mxfp8W8A8Block32E8M0.manifest_name(),
            ),
            combined_recipe_digest(
                base,
                manifest,
                MtpWeightEncoding::Nvfp4W4A4Block16E2M1E4M3FnF32.manifest_name(),
            )
        );
        assert_ne!(
            combined_recipe_digest(
                base,
                manifest,
                MtpWeightEncoding::Mxfp8W8A8Block32E8M0.manifest_name(),
            ),
            combined_recipe_digest(
                "sha256:other",
                manifest,
                MtpWeightEncoding::Mxfp8W8A8Block32E8M0.manifest_name()
            )
        );
    }

    #[test]
    fn no_clipping_encoding_names_round_trip_and_unknown_names_are_rejected() {
        let name = "mxfp8-w8a8-e4m3-block32-e8m0:mx-scale=no-clipping";
        let encoding = MtpWeightEncoding::Mxfp8W8A8Block32E8M0NoClippingScale;
        assert_eq!(
            MtpWeightEncoding::parse(name).expect("parse encoding"),
            encoding
        );
        assert_eq!(
            parse_manifest_encoding(name).expect("parse manifest encoding"),
            (encoding, None)
        );
        assert_eq!(encoding.manifest_name(), name);

        for unknown in [
            "mxfp8-w8a8-e4m3-block32-e8m0:mx-scale=clipping",
            "mxfp6-w6a6-e3m2-block32-e8m0:mx-scale=no-clipping-extra",
        ] {
            assert!(MtpWeightEncoding::parse(unknown).is_err());
            assert!(parse_manifest_encoding(unknown).is_err());
        }
        for retired in [
            "mxfp6-w6a6-e3m2-block32-e8m0",
            "mxfp6-w6a6-e3m2-block32-e8m0:mx-scale=no-clipping",
            "bf16-roundtrip-mxfp6",
        ] {
            let error = parse_manifest_encoding(retired).expect_err("MXFP6 MTP is retired");
            assert!(error.to_string().contains("retired MTP MXFP6"));
        }
    }

    #[test]
    fn nvfp4_weight_roundtrip_has_block16_planes_and_positive_tensor_scale() {
        let input = (0..64)
            .map(|index| (index as f32 - 31.0) * 0.125)
            .collect::<Vec<_>>();
        let quantized = quantize_nvfp4_weights(&input, 2, 32).expect("NVFP4 quantize");
        assert_eq!(quantized.packed_values.len(), 2 * 32 / 2);
        assert_eq!(quantized.block_scales.len(), 2 * 32 / 16);
        assert!(quantized.tensor_scale.is_finite());
        assert!(quantized.tensor_scale > 0.0);
        let decoded = quantized.dequantize();
        assert!(decoded.iter().all(|value| value.is_finite()));
        assert!(
            decoded
                .iter()
                .zip(input)
                .all(|(decoded, source)| (decoded - source).abs() <= 0.8)
        );
    }

    fn valid_activation_scale_manifest_value() -> Value {
        let mut scales = Map::new();
        for name in MTP_MATRIX_NAMES {
            scales.insert(name.to_owned(), Value::from(0.5_f64));
        }
        let mut activation_amax = Map::new();
        for site in NVFP4_ACTIVATION_SITES {
            activation_amax.insert(site.to_owned(), Value::from(1344.0_f64));
        }
        let source_reports = vec![
            CalibrationSourceReport {
                target: "gfx1030".to_owned(),
                report_sha256: sha256_bytes(b"gfx1030"),
            },
            CalibrationSourceReport {
                target: "gfx1201".to_owned(),
                report_sha256: sha256_bytes(b"gfx1201"),
            },
        ];
        let source_report_sha256 = source_report_digest(&source_reports);
        serde_json::json!({
            "schema": NVFP4_ACTIVATION_SCALE_SCHEMA,
            "scale_rule": NVFP4_CALIBRATION_SCALE_RULE,
            "input_manifest_sha256": sha256_bytes(b"held-out-input"),
            "suite_sha256": sha256_bytes(b"suite"),
            "source_report_sha256": source_report_sha256,
            "source_reports": source_reports,
            "activation_amax": activation_amax,
            "scales": scales,
        })
    }

    fn write_activation_scale_manifest(value: &Value) -> PathBuf {
        let path = temporary_test_path("activation-scale");
        let mut file = File::create(&path).expect("create activation manifest");
        file.write_all(serde_json::to_string(value).unwrap().as_bytes())
            .expect("write activation manifest");
        file.sync_all().expect("sync activation manifest");
        path
    }

    #[test]
    fn activation_scale_manifest_rejects_missing_extra_nonpositive_and_unequal_scales() {
        let valid_path = write_activation_scale_manifest(&valid_activation_scale_manifest_value());
        let valid = read_qwen38_mtp_nvfp4_activation_scale_manifest(&valid_path)
            .expect("valid activation manifest");
        assert_eq!(valid.scales.len(), 8);
        assert_eq!(valid.source_reports.len(), 2);
        std::fs::remove_file(&valid_path).expect("remove valid activation manifest");

        let mut invalid_cases = Vec::<(&str, Box<dyn Fn(&mut Value)>)>::new();
        invalid_cases.push((
            "missing",
            Box::new(|value: &mut Value| {
                value
                    .get_mut("scales")
                    .and_then(Value::as_object_mut)
                    .expect("scales")
                    .remove("mtp.fc.weight");
            }),
        ));
        invalid_cases.push((
            "extra",
            Box::new(|value: &mut Value| {
                value
                    .get_mut("scales")
                    .and_then(Value::as_object_mut)
                    .expect("scales")
                    .insert("mtp.extra.weight".to_owned(), Value::from(0.5_f64));
            }),
        ));
        invalid_cases.push((
            "nonpositive",
            Box::new(|value: &mut Value| {
                value["scales"]["mtp.fc.weight"] = Value::from(0.0_f64);
            }),
        ));
        invalid_cases.push((
            "unequal-gate-up",
            Box::new(|value: &mut Value| {
                value["scales"]["mtp.layers.0.mlp.up_proj.weight"] = Value::from(0.25_f64);
            }),
        ));
        for (label, mutate) in invalid_cases {
            let mut value = valid_activation_scale_manifest_value();
            mutate(&mut value);
            let path = write_activation_scale_manifest(&value);
            assert!(
                read_qwen38_mtp_nvfp4_activation_scale_manifest(&path).is_err(),
                "{label} manifest was accepted"
            );
            std::fs::remove_file(path).expect("remove invalid activation manifest");
        }
    }

    #[test]
    fn no_clipping_encoding_selects_the_unclipped_scale_rule() {
        let mut mxfp8_values = vec![1.0_f32; 32];
        mxfp8_values[0] = 449.0;
        let mxfp8_clipped = quantize_mtp_matrix(
            MtpWeightEncoding::Mxfp8W8A8Block32E8M0,
            &mxfp8_values,
            1,
            32,
        )
        .expect("clipped mxfp8");
        let mxfp8_no_clipping = quantize_mtp_matrix(
            MtpWeightEncoding::Mxfp8W8A8Block32E8M0NoClippingScale,
            &mxfp8_values,
            1,
            32,
        )
        .expect("no-clipping mxfp8");
        assert_ne!(mxfp8_clipped.scales(), mxfp8_no_clipping.scales());
    }

    #[test]
    fn bf16_roundtrip_records_bit_exact_finite_mxfp8_values() {
        let input = (0..64)
            .map(|index| (index as f32 - 31.0) * 0.25)
            .collect::<Vec<_>>();
        let recipe = MtpBf16RoundtripEncoding::Mxfp8;
        let quantized = recipe.quantize(&input, 1, 64).expect("quantize");
        let (bytes, diagnostics) =
            bf16_roundtrip_values(&quantized, &input, recipe).expect("roundtrip");
        assert_eq!(bytes.len(), 64 * 2);
        assert!(diagnostics.bit_exact, "{recipe:?}: {diagnostics:?}");
        assert_eq!(diagnostics.source_nonfinite_count, 0);
        assert_eq!(diagnostics.dequant_nonfinite_count, 0);
        assert_eq!(diagnostics.underflow_count, 0);
        assert_eq!(diagnostics.overflow_count, 0);
        assert_eq!(diagnostics.bit_mismatch_count, 0);
    }

    #[test]
    fn bf16_roundtrip_records_nonfinite_positions_and_withdraws_exact_claim() {
        let input = vec![f32::NAN; 32];
        let quantized = MtpBf16RoundtripEncoding::Mxfp8
            .quantize(&input, 1, 32)
            .expect("quantize");
        let (_bytes, diagnostics) =
            bf16_roundtrip_values(&quantized, &input, MtpBf16RoundtripEncoding::Mxfp8)
                .expect("roundtrip");
        assert!(!diagnostics.bit_exact);
        assert_eq!(diagnostics.source_nonfinite_count, 32);
        assert_eq!(diagnostics.dequant_nonfinite_count, 32);
        assert_eq!(diagnostics.roundtrip_nonfinite_count, 32);
        assert_eq!(
            diagnostics.first_source_nonfinite_positions,
            (0..16).collect::<Vec<_>>()
        );
    }

    #[test]
    fn payload_ranges_require_contiguous_complete_data() {
        let mut contiguous = BTreeMap::new();
        contiguous.insert(
            "values".to_owned(),
            SafeTensorMetadata {
                dtype: "U8".to_owned(),
                shape: vec![4],
                data_offsets: [0, 4],
            },
        );
        contiguous.insert(
            "scales".to_owned(),
            SafeTensorMetadata {
                dtype: "U8".to_owned(),
                shape: vec![4],
                data_offsets: [4, 8],
            },
        );
        assert!(validate_payload_ranges(&contiguous, 16, 8).is_ok());

        contiguous.get_mut("scales").expect("scales").data_offsets = [5, 9];
        assert!(validate_payload_ranges(&contiguous, 17, 8).is_err());
    }

    #[test]
    fn sha256_contract_rejects_non_digest_strings() {
        assert!(is_sha256(&sha256_bytes(b"sidecar")));
        assert!(!is_sha256("sha256:short"));
        assert!(!is_sha256(
            "sha256:GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
        ));
    }

    #[test]
    fn validate_record_accepts_n3_for_mxfp8_at_block_boundaries() {
        let encoding = MtpWeightEncoding::Mxfp8W8A8Block32E8M0;
        for columns in [32, 64] {
            let (record, value, scale, data) = make_validate_fixture(encoding, columns);
            assert!(
                validate_fixture(encoding, columns, record, value, scale, &data).is_ok(),
                "valid {encoding} N=3 K={columns} fixture rejected"
            );
        }
    }

    #[test]
    fn validate_record_accepts_bf16_roundtrip_with_empty_scale_plane() {
        let rows = 3_usize;
        let columns = 32_usize;
        let values = vec![0x31_u8; rows * columns * 2];
        let record = TensorRecord {
            name: "mtp.test.weight".to_owned(),
            logical_shape: vec![rows as u64, columns as u64],
            source_sha256: sha256_bytes(b"source"),
            value_sha256: sha256_bytes(&values),
            scale_sha256: sha256_bytes(&[]),
            roundtrip: Some(Bf16RoundtripDiagnostics {
                recipe: MtpBf16RoundtripEncoding::Mxfp8.manifest_name().to_owned(),
                element_count: (rows * columns) as u64,
                bit_exact: true,
                source_nonfinite_count: 0,
                dequant_nonfinite_count: 0,
                roundtrip_nonfinite_count: 0,
                underflow_count: 0,
                overflow_count: 0,
                bit_mismatch_count: 0,
                first_source_nonfinite_positions: Vec::new(),
                first_dequant_nonfinite_positions: Vec::new(),
                first_roundtrip_nonfinite_positions: Vec::new(),
                first_underflow_positions: Vec::new(),
                first_overflow_positions: Vec::new(),
                first_bit_mismatch_positions: Vec::new(),
            }),
            tensor_scale_sha256: None,
            input_global_scale_f32_bits: None,
        };
        let value = SafeTensorMetadata {
            dtype: "BF16".to_owned(),
            shape: vec![rows as u64, columns as u64],
            data_offsets: [0, values.len() as u64],
        };
        let scale = SafeTensorMetadata {
            dtype: "U8".to_owned(),
            shape: vec![rows as u64, 0],
            data_offsets: [values.len() as u64, values.len() as u64],
        };
        let path = temporary_test_path("bf16-roundtrip");
        {
            let mut output = File::create(&path).expect("create fixture");
            output.write_all(&[0_u8; 8]).expect("write prefix");
            output.write_all(&values).expect("write values");
            output.sync_all().expect("sync fixture");
        }
        let mut input = File::open(&path).expect("open fixture");
        assert!(
            validate_record(
                &record,
                &value,
                &scale,
                [rows as u64, columns as u64],
                MtpWeightEncoding::Bf16,
                true,
                &mut input,
                8,
            )
            .is_ok()
        );
        drop(input);
        std::fs::remove_file(path).expect("remove fixture");
    }

    #[test]
    fn validate_record_rejects_nonaligned_k_dtype_scale_length_and_truncation() {
        let encoding = MtpWeightEncoding::Mxfp8W8A8Block32E8M0;

        let (mut record, value, scale, data) = make_validate_fixture(encoding, 32);
        record.logical_shape = vec![3, 33];
        assert!(validate_fixture(encoding, 33, record, value, scale, &data).is_err());

        let (record, mut value, scale, data) = make_validate_fixture(encoding, 32);
        value.dtype = "BF16".to_owned();
        assert!(validate_fixture(encoding, 32, record, value, scale, &data).is_err());

        let (record, value, mut scale, data) = make_validate_fixture(encoding, 32);
        scale.shape = vec![3, 2];
        assert!(validate_fixture(encoding, 32, record, value, scale, &data).is_err());

        let (record, value, mut scale, data) = make_validate_fixture(encoding, 32);
        scale.data_offsets[1] -= 1;
        assert!(validate_fixture(encoding, 32, record, value, scale, &data).is_err());

        let (record, value, scale, mut data) = make_validate_fixture(encoding, 32);
        data.pop();
        assert!(validate_fixture(encoding, 32, record, value, scale, &data).is_err());
    }

    #[test]
    fn read_tensor_bytes_rejects_payload_mutation_after_verification() {
        let path = temporary_test_path("mutation");
        let values = vec![0x31_u8; 96];
        let scales = vec![0x7f_u8; 3];
        let mut payload_bytes = values.clone();
        payload_bytes.extend_from_slice(&scales);
        {
            let mut output = File::create(&path).expect("create mutation fixture");
            output
                .write_all(&payload_bytes)
                .expect("write mutation fixture");
            output.sync_all().expect("sync mutation fixture");
        }
        let payload = Arc::new(File::open(&path).expect("open mutation fixture"));
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "mtp.test.weight".to_owned(),
            MtpQuantizedSidecarTensor {
                name: "mtp.test.weight".to_owned(),
                logical_shape: [3, 32],
                value_range: [0, 96],
                scale_range: [96, 99],
                tensor_scale_range: None,
                source_sha256: sha256_bytes(b"source"),
                value_sha256: sha256_bytes(&values),
                scale_sha256: sha256_bytes(&scales),
                tensor_scale_sha256: None,
                nvfp4_weight_tensor_scale_f32_bits: None,
                nvfp4_input_global_scale_f32_bits: None,
            },
        );
        let sidecar = VerifiedQwen38MtpQuantizedSidecar {
            manifest_path: PathBuf::from("manifest.json"),
            payload_path: path.clone(),
            payload,
            source_lock_fingerprint: "sha256:lock".to_owned(),
            source_mtp_sha256: "mtp".to_owned(),
            base_recipe_digest: "sha256:recipe".to_owned(),
            manifest_fingerprint: "sha256:manifest".to_owned(),
            manifest_encoding: MtpWeightEncoding::Mxfp8W8A8Block32E8M0
                .manifest_name()
                .to_owned(),
            roundtrip_encoding: None,
            encoding: MtpWeightEncoding::Mxfp8W8A8Block32E8M0,
            data_start: 0,
            tensors,
        };
        assert_eq!(
            sidecar
                .read_tensor_bytes("mtp.test.weight")
                .expect("unchanged payload")
                .0,
            values
        );

        {
            let mut input = OpenOptions::new()
                .write(true)
                .open(&path)
                .expect("open mutation fixture for write");
            input.write_all(&[0x32]).expect("mutate value plane");
            input.sync_all().expect("sync mutation");
        }
        assert!(sidecar.read_tensor_bytes("mtp.test.weight").is_err());
        drop(sidecar);
        std::fs::remove_file(path).expect("remove mutation fixture");
    }
}
