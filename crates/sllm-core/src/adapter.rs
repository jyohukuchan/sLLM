//! Host-only contracts for preconverted BF16 LoRA and control-vector
//! artifacts.
//!
//! This module deliberately stops at verification and bounded CPU oracles.
//! It does not read files, perform conversion, or submit work to a backend.

use core::fmt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::sync::Arc;

use crate::{TensorDType, WeightClassification, WeightLoadPlan};

pub const ADAPTER_LOCK_SCHEMA_VERSION_V1: &str = "sllm-adapter-lock-v1";
pub const MAX_LORA_TARGETS_V1: usize = 256;
pub const MAX_REQUEST_ADAPTERS_V1: usize = 4;
pub const MAX_REQUEST_CONTROLS_V1: usize = 4;
pub const MAX_LORA_RANK_V1: u64 = 256;
pub const MAX_ALIAS_BYTES_V1: usize = 128;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterErrorV1(String);

impl AdapterErrorV1 {
    fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for AdapterErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for AdapterErrorV1 {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdapterModelDimsV1 {
    hidden_size: u64,
    num_hidden_layers: u64,
}

impl AdapterModelDimsV1 {
    pub fn new(hidden_size: u64, num_hidden_layers: u64) -> Result<Self, AdapterErrorV1> {
        if hidden_size == 0 || num_hidden_layers == 0 {
            return Err(AdapterErrorV1::invalid(
                "model dimensions must have nonzero hidden size and layer count",
            ));
        }
        Ok(Self {
            hidden_size,
            num_hidden_layers,
        })
    }

    pub const fn hidden_size(self) -> u64 {
        self.hidden_size
    }

    pub const fn num_hidden_layers(self) -> u64 {
        self.num_hidden_layers
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterArtifactIdentityV1 {
    kind: &'static str,
    artifact_id: String,
    lock_sha256: String,
    payload_sha256: String,
    payload_size: u64,
}

impl AdapterArtifactIdentityV1 {
    pub fn kind(&self) -> &str {
        self.kind
    }

    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }

    pub fn payload_sha256(&self) -> &str {
        &self.payload_sha256
    }

    pub fn lock_sha256(&self) -> &str {
        &self.lock_sha256
    }

    pub const fn payload_size(&self) -> u64 {
        self.payload_size
    }

    /// A deterministic identity containing metadata only; no pointer address
    /// or allocation identity is included.
    pub fn canonical_string(&self) -> String {
        format!(
            "{}:{}:{}:{}:{}:{}",
            self.kind,
            self.artifact_id,
            self.lock_sha256,
            self.payload_sha256,
            self.payload_size,
            "v1"
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LoraAdapterLockV1 {
    pub schema_version: String,
    pub kind: String,
    pub artifact_id: String,
    pub alpha: f32,
    pub base_model_fingerprint: String,
    pub base_weight_plan_digest: String,
    pub payload_sha256: String,
    pub payload_size: u64,
    pub targets: Vec<LoraTargetLockV1>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LoraTargetLockV1 {
    pub tensor_name: String,
    pub dtype: String,
    pub target_shape: Vec<u64>,
    pub rank: u64,
    pub a_offset: u64,
    pub a_size: u64,
    pub b_offset: u64,
    pub b_size: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ControlVectorLockV1 {
    pub schema_version: String,
    pub kind: String,
    pub artifact_id: String,
    pub dtype: String,
    pub base_model_fingerprint: String,
    pub base_weight_plan_digest: String,
    pub payload_sha256: String,
    pub payload_size: u64,
    pub hidden_size: u64,
    pub layer_start: u64,
    pub layer_end: u64,
    pub vector_offset: u64,
    pub vector_size: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedLoraTargetV1 {
    tensor_name: String,
    target_shape: [u64; 2],
    rank: u64,
    a_offset: u64,
    b_offset: u64,
}

impl VerifiedLoraTargetV1 {
    pub fn tensor_name(&self) -> &str {
        &self.tensor_name
    }

    pub fn target_shape(&self) -> [u64; 2] {
        self.target_shape
    }

    pub const fn rank(&self) -> u64 {
        self.rank
    }

    pub const fn a_offset(&self) -> u64 {
        self.a_offset
    }

    pub const fn b_offset(&self) -> u64 {
        self.b_offset
    }

    pub const fn input_size(&self) -> u64 {
        self.target_shape[1]
    }

    pub const fn output_size(&self) -> u64 {
        self.target_shape[0]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedLoraPayloadV1 {
    lock: LoraAdapterLockV1,
    payload: Arc<[u8]>,
    identity: AdapterArtifactIdentityV1,
    targets: Vec<VerifiedLoraTargetV1>,
}

impl VerifiedLoraPayloadV1 {
    pub fn verify(
        lock_json: &[u8],
        payload: Arc<[u8]>,
        model_fingerprint: &str,
        plan: &WeightLoadPlan,
    ) -> Result<Self, AdapterErrorV1> {
        let lock = parse_lora_lock_v1(lock_json)?;
        verify_lora_lock(lock, payload, model_fingerprint, plan)
    }

    pub fn from_bytes(
        lock_json: &[u8],
        payload: impl Into<Arc<[u8]>>,
        model_fingerprint: &str,
        plan: &WeightLoadPlan,
    ) -> Result<Self, AdapterErrorV1> {
        Self::verify(lock_json, payload.into(), model_fingerprint, plan)
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    pub fn payload_owner(&self) -> Arc<[u8]> {
        Arc::clone(&self.payload)
    }

    pub fn identity(&self) -> &AdapterArtifactIdentityV1 {
        &self.identity
    }

    pub fn lock(&self) -> &LoraAdapterLockV1 {
        &self.lock
    }

    pub fn targets(&self) -> &[VerifiedLoraTargetV1] {
        &self.targets
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedControlVectorPayloadV1 {
    lock: ControlVectorLockV1,
    payload: Arc<[u8]>,
    identity: AdapterArtifactIdentityV1,
}

impl VerifiedControlVectorPayloadV1 {
    pub fn verify(
        lock_json: &[u8],
        payload: Arc<[u8]>,
        model_fingerprint: &str,
        plan: &WeightLoadPlan,
        dims: AdapterModelDimsV1,
    ) -> Result<Self, AdapterErrorV1> {
        let lock = parse_control_vector_lock_v1(lock_json)?;
        verify_control_vector_lock(lock, payload, model_fingerprint, plan, dims)
    }

    pub fn from_bytes(
        lock_json: &[u8],
        payload: impl Into<Arc<[u8]>>,
        model_fingerprint: &str,
        plan: &WeightLoadPlan,
        dims: AdapterModelDimsV1,
    ) -> Result<Self, AdapterErrorV1> {
        Self::verify(lock_json, payload.into(), model_fingerprint, plan, dims)
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    pub fn payload_owner(&self) -> Arc<[u8]> {
        Arc::clone(&self.payload)
    }

    pub fn identity(&self) -> &AdapterArtifactIdentityV1 {
        &self.identity
    }

    pub fn lock(&self) -> &ControlVectorLockV1 {
        &self.lock
    }

    pub const fn layer_range(&self) -> (u64, u64) {
        (self.lock.layer_start, self.lock.layer_end)
    }
}

pub fn parse_lora_lock_v1(bytes: &[u8]) -> Result<LoraAdapterLockV1, AdapterErrorV1> {
    let lock: LoraAdapterLockV1 = serde_json::from_slice(bytes)
        .map_err(|_| AdapterErrorV1::invalid("LoRA lock JSON is invalid"))?;
    validate_lora_lock_shape(&lock)?;
    Ok(lock)
}

pub fn parse_control_vector_lock_v1(bytes: &[u8]) -> Result<ControlVectorLockV1, AdapterErrorV1> {
    let lock: ControlVectorLockV1 = serde_json::from_slice(bytes)
        .map_err(|_| AdapterErrorV1::invalid("control-vector lock JSON is invalid"))?;
    validate_control_vector_lock_shape(&lock)?;
    Ok(lock)
}

fn verify_lora_lock(
    lock: LoraAdapterLockV1,
    payload: Arc<[u8]>,
    model_fingerprint: &str,
    plan: &WeightLoadPlan,
) -> Result<VerifiedLoraPayloadV1, AdapterErrorV1> {
    let lock_sha256 = canonical_lock_sha256(&lock)?;
    validate_common(
        &lock.schema_version,
        &lock.kind,
        &lock.artifact_id,
        &lock.base_model_fingerprint,
        &lock.base_weight_plan_digest,
        &lock.payload_sha256,
        lock.payload_size,
        payload.len(),
        model_fingerprint,
        &plan.digest_hex(),
        "lora",
        &payload,
    )?;
    let mut names = BTreeSet::new();
    let mut ranges = Vec::with_capacity(lock.targets.len() * 2);
    let mut targets = Vec::with_capacity(lock.targets.len());
    for target in &lock.targets {
        if !names.insert(target.tensor_name.clone()) {
            return Err(AdapterErrorV1::invalid("LoRA target names must be unique"));
        }
        let entry = plan
            .entries
            .iter()
            .find(|entry| entry.tensor_name == target.tensor_name)
            .ok_or_else(|| AdapterErrorV1::invalid("LoRA target is absent from weight plan"))?;
        if entry.classification != WeightClassification::Required
            || entry.dtype != TensorDType::Bf16
        {
            return Err(AdapterErrorV1::invalid(
                "LoRA target must bind a required BF16 tensor",
            ));
        }
        if target.target_shape.as_slice() != entry.shape.as_slice()
            || target.target_shape.len() != 2
        {
            return Err(AdapterErrorV1::invalid(
                "LoRA target shape differs from weight plan",
            ));
        }
        if !(1..=MAX_LORA_RANK_V1).contains(&target.rank) {
            return Err(AdapterErrorV1::invalid("LoRA rank is outside 1..=256"));
        }
        let output = target.target_shape[0];
        let input = target.target_shape[1];
        let a_size = input
            .checked_mul(target.rank)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| AdapterErrorV1::invalid("LoRA A size overflowed"))?;
        let b_size = target
            .rank
            .checked_mul(output)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| AdapterErrorV1::invalid("LoRA B size overflowed"))?;
        if target.dtype != "BF16"
            || target.a_offset % 2 != 0
            || target.b_offset % 2 != 0
            || target.a_size != a_size
            || target.b_size != b_size
        {
            return Err(AdapterErrorV1::invalid(
                "LoRA payload matrix dtype, alignment, or size is invalid",
            ));
        }
        ranges.push((target.a_offset, a_size));
        ranges.push((target.b_offset, b_size));
        targets.push(VerifiedLoraTargetV1 {
            tensor_name: target.tensor_name.clone(),
            target_shape: [output, input],
            rank: target.rank,
            a_offset: target.a_offset,
            b_offset: target.b_offset,
        });
    }
    if targets.is_empty() || targets.len() > MAX_LORA_TARGETS_V1 {
        return Err(AdapterErrorV1::invalid(
            "LoRA target count must be between 1 and 256",
        ));
    }
    if lock
        .targets
        .windows(2)
        .any(|window| window[0].tensor_name >= window[1].tensor_name)
    {
        return Err(AdapterErrorV1::invalid(
            "LoRA targets must be sorted by tensor name",
        ));
    }
    validate_ranges(&ranges, payload.len())?;
    Ok(VerifiedLoraPayloadV1 {
        identity: AdapterArtifactIdentityV1 {
            kind: "lora",
            artifact_id: lock.artifact_id.clone(),
            lock_sha256,
            payload_sha256: lock.payload_sha256.clone(),
            payload_size: lock.payload_size,
        },
        lock,
        payload,
        targets,
    })
}

fn verify_control_vector_lock(
    lock: ControlVectorLockV1,
    payload: Arc<[u8]>,
    model_fingerprint: &str,
    plan: &WeightLoadPlan,
    dims: AdapterModelDimsV1,
) -> Result<VerifiedControlVectorPayloadV1, AdapterErrorV1> {
    let lock_sha256 = canonical_lock_sha256(&lock)?;
    validate_common(
        &lock.schema_version,
        &lock.kind,
        &lock.artifact_id,
        &lock.base_model_fingerprint,
        &lock.base_weight_plan_digest,
        &lock.payload_sha256,
        lock.payload_size,
        payload.len(),
        model_fingerprint,
        &plan.digest_hex(),
        "control-vector",
        &payload,
    )?;
    if lock.dtype != "bf16"
        || lock.hidden_size != dims.hidden_size
        || lock.layer_start >= lock.layer_end
        || lock.layer_end > dims.num_hidden_layers
    {
        return Err(AdapterErrorV1::invalid(
            "control-vector hidden size or half-open layer range differs from model",
        ));
    }
    let layers = lock.layer_end - lock.layer_start;
    let expected_size = layers
        .checked_mul(lock.hidden_size)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| AdapterErrorV1::invalid("control-vector size overflowed"))?;
    if lock.vector_offset % 2 != 0
        || lock.vector_size != expected_size
        || lock.vector_offset.checked_add(lock.vector_size).is_none()
    {
        return Err(AdapterErrorV1::invalid(
            "control-vector payload alignment or size is invalid",
        ));
    }
    validate_ranges(&[(lock.vector_offset, lock.vector_size)], payload.len())?;
    Ok(VerifiedControlVectorPayloadV1 {
        identity: AdapterArtifactIdentityV1 {
            kind: "control-vector",
            artifact_id: lock.artifact_id.clone(),
            lock_sha256,
            payload_sha256: lock.payload_sha256.clone(),
            payload_size: lock.payload_size,
        },
        lock,
        payload,
    })
}

fn validate_lora_lock_shape(lock: &LoraAdapterLockV1) -> Result<(), AdapterErrorV1> {
    validate_common_metadata(
        &lock.schema_version,
        &lock.kind,
        &lock.artifact_id,
        &lock.base_model_fingerprint,
        &lock.base_weight_plan_digest,
        &lock.payload_sha256,
        lock.payload_size,
        "lora",
    )?;
    if lock.targets.is_empty() || lock.targets.len() > MAX_LORA_TARGETS_V1 {
        return Err(AdapterErrorV1::invalid("invalid LoRA target count"));
    }
    if !lock.alpha.is_finite() || lock.alpha <= 0.0 {
        return Err(AdapterErrorV1::invalid(
            "LoRA alpha must be finite and positive",
        ));
    }
    Ok(())
}

fn validate_control_vector_lock_shape(lock: &ControlVectorLockV1) -> Result<(), AdapterErrorV1> {
    validate_common_metadata(
        &lock.schema_version,
        &lock.kind,
        &lock.artifact_id,
        &lock.base_model_fingerprint,
        &lock.base_weight_plan_digest,
        &lock.payload_sha256,
        lock.payload_size,
        "control-vector",
    )?;
    if lock.layer_start >= lock.layer_end || lock.hidden_size == 0 {
        return Err(AdapterErrorV1::invalid(
            "control-vector layer range or hidden size is invalid",
        ));
    }
    if lock.dtype != "bf16" {
        return Err(AdapterErrorV1::invalid("control-vector dtype must be bf16"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_common_metadata(
    schema_version: &str,
    kind: &str,
    artifact_id: &str,
    model_fingerprint: &str,
    plan_digest: &str,
    payload_sha256: &str,
    payload_size: u64,
    expected_kind: &str,
) -> Result<(), AdapterErrorV1> {
    if schema_version != ADAPTER_LOCK_SCHEMA_VERSION_V1 || kind != expected_kind {
        return Err(AdapterErrorV1::invalid(
            "adapter lock schema or kind is invalid",
        ));
    }
    if artifact_id.is_empty() || artifact_id.len() > MAX_ALIAS_BYTES_V1 {
        return Err(AdapterErrorV1::invalid("adapter artifact ID is invalid"));
    }
    parse_sha256(model_fingerprint, "base model fingerprint")?;
    parse_sha256(plan_digest, "base weight-plan digest")?;
    parse_sha256(payload_sha256, "payload SHA-256")?;
    if payload_size == 0 {
        return Err(AdapterErrorV1::invalid("payload size must be nonzero"));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_common(
    schema_version: &str,
    kind: &str,
    artifact_id: &str,
    base_model_fingerprint: &str,
    base_plan_digest: &str,
    payload_sha256: &str,
    payload_size: u64,
    actual_size: usize,
    expected_model: &str,
    expected_plan: &str,
    expected_kind: &str,
    payload: &[u8],
) -> Result<(), AdapterErrorV1> {
    validate_common_metadata(
        schema_version,
        kind,
        artifact_id,
        base_model_fingerprint,
        base_plan_digest,
        payload_sha256,
        payload_size,
        expected_kind,
    )?;
    if base_model_fingerprint != expected_model || base_plan_digest != expected_plan {
        return Err(AdapterErrorV1::invalid(
            "adapter base model or weight-plan identity differs",
        ));
    }
    if usize::try_from(payload_size).ok() != Some(actual_size) {
        return Err(AdapterErrorV1::invalid("adapter payload size differs"));
    }
    let expected_hash = parse_sha256(payload_sha256, "payload SHA-256")?;
    if canonical_sha256(payload) != expected_hash {
        return Err(AdapterErrorV1::invalid("adapter payload hash differs"));
    }
    Ok(())
}

fn validate_ranges(ranges: &[(u64, u64)], payload_size: usize) -> Result<(), AdapterErrorV1> {
    let payload_size = u64::try_from(payload_size)
        .map_err(|_| AdapterErrorV1::invalid("payload size exceeds u64"))?;
    let mut ordered = ranges.to_vec();
    ordered.sort_unstable_by_key(|range| range.0);
    for (index, &(start, length)) in ordered.iter().enumerate() {
        if start % 2 != 0
            || length == 0
            || start
                .checked_add(length)
                .is_none_or(|end| end > payload_size)
        {
            return Err(AdapterErrorV1::invalid(
                "adapter payload range is unaligned, empty, or out of bounds",
            ));
        }
        let overlaps_previous = index
            .checked_sub(1)
            .and_then(|previous| ordered.get(previous))
            .is_some_and(|&(previous_start, previous_length)| {
                previous_start
                    .checked_add(previous_length)
                    .is_some_and(|end| end > start)
            });
        if overlaps_previous {
            return Err(AdapterErrorV1::invalid("adapter payload ranges overlap"));
        }
    }
    Ok(())
}

fn parse_sha256(value: &str, field: &str) -> Result<[u8; 32], AdapterErrorV1> {
    let encoded = value
        .strip_prefix("sha256:")
        .ok_or_else(|| AdapterErrorV1::invalid(format!("{field} must use sha256: identity")))?;
    if encoded.len() != 64 || !encoded.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(AdapterErrorV1::invalid(format!(
            "{field} is not a SHA-256 identity"
        )));
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in encoded.as_bytes().chunks_exact(2).enumerate() {
        digest[index] = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
    }
    Ok(digest)
}

fn hex_nibble(byte: u8) -> Result<u8, AdapterErrorV1> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(AdapterErrorV1::invalid("invalid SHA-256 hexadecimal digit")),
    }
}

fn canonical_sha256(payload: &[u8]) -> [u8; 32] {
    Sha256::digest(payload).into()
}

fn canonical_lock_sha256<T: Serialize>(lock: &T) -> Result<String, AdapterErrorV1> {
    let bytes = serde_json::to_vec(lock)
        .map_err(|_| AdapterErrorV1::invalid("adapter lock cannot be canonicalized"))?;
    Ok(format!("sha256:{:x}", Sha256::digest(bytes)))
}

fn checked_scale(scale: f32) -> Result<f32, AdapterErrorV1> {
    if !scale.is_finite() || !(-16.0..=16.0).contains(&scale) {
        return Err(AdapterErrorV1::invalid("adapter scale is outside [-16,16]"));
    }
    Ok(scale)
}

fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff_u32 + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding)) >> 16) as u16
}

pub fn apply_lora_bf16(
    payload: &VerifiedLoraPayloadV1,
    target: &VerifiedLoraTargetV1,
    input_bf16: &[u16],
    output_bf16: &mut [u16],
    scale: f32,
) -> Result<(), AdapterErrorV1> {
    let effective_scale = checked_lora_scale(payload.lock.alpha, target.rank, scale)?;
    let input = usize::try_from(target.input_size())
        .map_err(|_| AdapterErrorV1::invalid("LoRA input size overflows usize"))?;
    let output = usize::try_from(target.output_size())
        .map_err(|_| AdapterErrorV1::invalid("LoRA output size overflows usize"))?;
    let rank = usize::try_from(target.rank)
        .map_err(|_| AdapterErrorV1::invalid("LoRA rank overflows usize"))?;
    if input_bf16.len() != input || output_bf16.len() != output {
        return Err(AdapterErrorV1::invalid(
            "LoRA oracle slice shape is invalid",
        ));
    }
    let a_start = usize::try_from(target.a_offset)
        .map_err(|_| AdapterErrorV1::invalid("LoRA A offset overflows usize"))?;
    let b_start = usize::try_from(target.b_offset)
        .map_err(|_| AdapterErrorV1::invalid("LoRA B offset overflows usize"))?;
    let a_bytes = input
        .checked_mul(rank)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| AdapterErrorV1::invalid("LoRA A range overflows usize"))?;
    let b_bytes = rank
        .checked_mul(output)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| AdapterErrorV1::invalid("LoRA B range overflows usize"))?;
    let a_end = a_start
        .checked_add(a_bytes)
        .ok_or_else(|| AdapterErrorV1::invalid("LoRA A range overflows usize"))?;
    let b_end = b_start
        .checked_add(b_bytes)
        .ok_or_else(|| AdapterErrorV1::invalid("LoRA B range overflows usize"))?;
    let a = payload
        .payload
        .get(a_start..a_end)
        .ok_or_else(|| AdapterErrorV1::invalid("LoRA A range is truncated"))?;
    let b = payload
        .payload
        .get(b_start..b_end)
        .ok_or_else(|| AdapterErrorV1::invalid("LoRA B range is truncated"))?;
    for (output_index, output_value) in output_bf16.iter_mut().enumerate() {
        let mut delta = 0.0_f32;
        for rank_index in 0..rank {
            let mut projected = 0.0_f32;
            for (input_index, &input_value) in input_bf16.iter().enumerate() {
                let offset = (rank_index * input + input_index) * 2;
                let value = u16::from_le_bytes([a[offset], a[offset + 1]]);
                projected += bf16_to_f32(input_value) * bf16_to_f32(value);
            }
            let offset = (output_index * rank + rank_index) * 2;
            let value = u16::from_le_bytes([b[offset], b[offset + 1]]);
            delta += projected * bf16_to_f32(value);
        }
        *output_value = f32_to_bf16(bf16_to_f32(*output_value) + effective_scale * delta);
    }
    Ok(())
}

pub fn apply_control_vector_bf16(
    payload: &VerifiedControlVectorPayloadV1,
    hidden_bf16: &mut [u16],
    model_dims: AdapterModelDimsV1,
    layer: u64,
    scale: f32,
) -> Result<(), AdapterErrorV1> {
    let scale = checked_scale(scale)?;
    if payload.lock.hidden_size != model_dims.hidden_size
        || layer < payload.lock.layer_start
        || layer >= payload.lock.layer_end
        || hidden_bf16.len() != usize::try_from(model_dims.hidden_size).unwrap_or(usize::MAX)
    {
        return Err(AdapterErrorV1::invalid(
            "control-vector oracle layer or hidden slice is invalid",
        ));
    }
    let hidden = usize::try_from(payload.lock.hidden_size)
        .map_err(|_| AdapterErrorV1::invalid("control-vector hidden size overflows usize"))?;
    let layer_index = usize::try_from(layer - payload.lock.layer_start)
        .map_err(|_| AdapterErrorV1::invalid("control-vector layer overflows usize"))?;
    let layer_offset = layer_index
        .checked_mul(hidden)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| AdapterErrorV1::invalid("control-vector offset overflows usize"))?;
    let offset = usize::try_from(payload.lock.vector_offset)
        .map_err(|_| AdapterErrorV1::invalid("control-vector offset overflows usize"))?
        .checked_add(layer_offset)
        .ok_or_else(|| AdapterErrorV1::invalid("control-vector offset overflows usize"))?;
    let end = offset
        .checked_add(
            hidden
                .checked_mul(2)
                .ok_or_else(|| AdapterErrorV1::invalid("control-vector range overflows usize"))?,
        )
        .ok_or_else(|| AdapterErrorV1::invalid("control-vector range overflows usize"))?;
    let bytes = payload
        .payload
        .get(offset..end)
        .ok_or_else(|| AdapterErrorV1::invalid("control-vector range is truncated"))?;
    for index in 0..hidden {
        let value = u16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]]);
        hidden_bf16[index] =
            f32_to_bf16(bf16_to_f32(hidden_bf16[index]) + scale * bf16_to_f32(value));
    }
    Ok(())
}

fn checked_lora_scale(alpha: f32, rank: u64, request_scale: f32) -> Result<f32, AdapterErrorV1> {
    let request_scale = checked_scale(request_scale)?;
    if !alpha.is_finite() || alpha <= 0.0 || rank == 0 {
        return Err(AdapterErrorV1::invalid("LoRA alpha or rank is invalid"));
    }
    let effective = request_scale * (alpha / rank as f32);
    if !effective.is_finite() {
        return Err(AdapterErrorV1::invalid(
            "LoRA effective scale is non-finite",
        ));
    }
    Ok(effective)
}

#[derive(Clone, Debug)]
pub struct LoraAdapterSelectionV1 {
    pub alias: String,
    pub artifact: Arc<VerifiedLoraPayloadV1>,
    pub scale: f32,
}

#[derive(Clone, Debug)]
pub struct ControlVectorSelectionV1 {
    pub alias: String,
    pub artifact: Arc<VerifiedControlVectorPayloadV1>,
    pub scale: f32,
}

#[derive(Clone, Debug)]
pub struct AdapterRequestSetV1 {
    adapters: Vec<LoraAdapterSelectionV1>,
    controls: Vec<ControlVectorSelectionV1>,
    identity: String,
}

impl AdapterRequestSetV1 {
    pub fn new(
        adapters: Vec<LoraAdapterSelectionV1>,
        controls: Vec<ControlVectorSelectionV1>,
    ) -> Result<Self, AdapterErrorV1> {
        if adapters.len() > MAX_REQUEST_ADAPTERS_V1 {
            return Err(AdapterErrorV1::invalid(
                "request contains too many LoRA adapters",
            ));
        }
        if controls.len() > MAX_REQUEST_CONTROLS_V1 {
            return Err(AdapterErrorV1::invalid(
                "request contains too many control vectors",
            ));
        }
        validate_selection_aliases(adapters.iter().map(|item| item.alias.as_str()))?;
        validate_selection_aliases(controls.iter().map(|item| item.alias.as_str()))?;
        let mut all_aliases = BTreeSet::new();
        for alias in adapters.iter().map(|item| item.alias.as_str()) {
            if !all_aliases.insert(alias) {
                return Err(AdapterErrorV1::invalid(
                    "adapter aliases must be globally unique",
                ));
            }
        }
        for alias in controls.iter().map(|item| item.alias.as_str()) {
            if !all_aliases.insert(alias) {
                return Err(AdapterErrorV1::invalid(
                    "adapter aliases must be globally unique",
                ));
            }
        }
        for (index, left) in controls.iter().enumerate() {
            let (left_start, left_end) = left.artifact.layer_range();
            for right in controls.iter().skip(index + 1) {
                let (right_start, right_end) = right.artifact.layer_range();
                if left_start < right_end && right_start < left_end {
                    return Err(AdapterErrorV1::invalid(
                        "control-vector layer ranges must not overlap",
                    ));
                }
            }
        }
        for item in &adapters {
            checked_scale(item.scale)?;
        }
        for item in &controls {
            checked_scale(item.scale)?;
        }
        let identity = request_identity(&adapters, &controls);
        Ok(Self {
            adapters,
            controls,
            identity,
        })
    }

    pub fn disabled() -> Self {
        Self {
            adapters: Vec::new(),
            controls: Vec::new(),
            identity: "adapter:none-v1".to_owned(),
        }
    }

    pub fn adapters(&self) -> &[LoraAdapterSelectionV1] {
        &self.adapters
    }

    pub fn controls(&self) -> &[ControlVectorSelectionV1] {
        &self.controls
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }
}

fn validate_selection_aliases<'a>(
    aliases: impl Iterator<Item = &'a str>,
) -> Result<(), AdapterErrorV1> {
    let mut previous = None;
    for alias in aliases {
        if alias.is_empty()
            || alias.len() > MAX_ALIAS_BYTES_V1
            || !alias
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        {
            return Err(AdapterErrorV1::invalid("adapter alias is invalid"));
        }
        if previous.is_some_and(|value| value >= alias) {
            return Err(AdapterErrorV1::invalid(
                "adapter aliases must be unique and sorted",
            ));
        }
        previous = Some(alias);
    }
    Ok(())
}

fn request_identity(
    adapters: &[LoraAdapterSelectionV1],
    controls: &[ControlVectorSelectionV1],
) -> String {
    if adapters.is_empty() && controls.is_empty() {
        return "adapter:none-v1".to_owned();
    }
    let mut identity = String::from("adapter:set-v1");
    for item in adapters {
        identity.push_str("|lora:");
        identity.push_str(&item.alias);
        identity.push(':');
        identity.push_str(item.artifact.identity().artifact_id());
        identity.push(':');
        identity.push_str(item.artifact.identity().lock_sha256());
        identity.push(':');
        identity.push_str(&format!("{:08x}", item.scale.to_bits()));
    }
    for item in controls {
        identity.push_str("|control:");
        identity.push_str(&item.alias);
        identity.push(':');
        identity.push_str(item.artifact.identity().artifact_id());
        identity.push(':');
        identity.push_str(item.artifact.identity().lock_sha256());
        identity.push(':');
        identity.push_str(&format!("{:08x}", item.scale.to_bits()));
    }
    identity
}
