//! Host-only Qwen3.5 weight registry and deterministic load-plan construction.
//!
//! The builder consumes descriptors that have already passed model-lock and
//! safetensors validation. It never reads tensor payloads and deliberately does
//! not duplicate the model parser, shape catalog, cache hasher, or range reader.

use crate::model::{
    LayerType, LockedFile, ModelLock, TensorDType, TensorDescriptor, VerifiedCache,
    reviewed_qwen35_spec,
};
use crate::{BufferRange, ExecutionQueue, ExecutionSession, ExecutionState};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

pub const WEIGHT_LOAD_CHUNK_BYTES: u64 = 16 * 1024 * 1024;

const PLAN_DOMAIN: &[u8] = b"sLLM-weight-load-plan-v1\0";
const QWEN_SCHEMA_VERSION: &str = "model-lock-v1";
#[cfg(test)]
const QWEN_REPO_ID: &str = crate::model::QWEN35_4B_REPO_ID;
#[cfg(test)]
const QWEN_REVISION: &str = crate::model::QWEN35_4B_REVISION;
#[cfg(test)]
const QWEN_FINGERPRINT: &str = crate::model::QWEN35_4B_FINGERPRINT;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightPlanError(String);

impl WeightPlanError {
    fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for WeightPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid weight load plan: {}", self.0)
    }
}

impl std::error::Error for WeightPlanError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightUploadError(String);

impl WeightUploadError {
    fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for WeightUploadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid verified weight upload: {}", self.0)
    }
}

impl std::error::Error for WeightUploadError {}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum WeightClassification {
    Required,
    ConfigConditional,
    KnownUnconsumed,
}

impl WeightClassification {
    fn tag(self) -> u8 {
        match self {
            Self::Required => 1,
            Self::ConfigConditional => 2,
            Self::KnownUnconsumed => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum WeightConsumer {
    EmbeddingAndTiedOutput,
    Embedding,
    OutputProjection,
    FinalNorm,
    InputNorm,
    PostAttentionNorm,
    MlpGate,
    MlpUp,
    MlpDown,
    GdnInProjQkv,
    GdnInProjZ,
    GdnInProjB,
    GdnInProjA,
    GdnConv1d,
    GdnALog,
    GdnDtBias,
    GdnNorm,
    GdnOutProj,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionO,
    AttentionQNorm,
    AttentionKNorm,
}

impl WeightConsumer {
    fn tag(self) -> u8 {
        match self {
            Self::EmbeddingAndTiedOutput => 1,
            Self::FinalNorm => 2,
            Self::InputNorm => 3,
            Self::PostAttentionNorm => 4,
            Self::MlpGate => 5,
            Self::MlpUp => 6,
            Self::MlpDown => 7,
            Self::GdnInProjQkv => 8,
            Self::GdnInProjZ => 9,
            Self::GdnInProjB => 10,
            Self::GdnInProjA => 11,
            Self::GdnConv1d => 12,
            Self::GdnALog => 13,
            Self::GdnDtBias => 14,
            Self::GdnNorm => 15,
            Self::GdnOutProj => 16,
            Self::AttentionQ => 17,
            Self::AttentionK => 18,
            Self::AttentionV => 19,
            Self::AttentionO => 20,
            Self::AttentionQNorm => 21,
            Self::AttentionKNorm => 22,
            Self::Embedding => 23,
            Self::OutputProjection => 24,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct WeightConsumerKey {
    pub layer: Option<u64>,
    pub role: WeightConsumer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WeightLoadChunk {
    pub source_offset: u64,
    pub destination_offset: u64,
    pub byte_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightLoadEntry {
    pub tensor_name: String,
    pub classification: WeightClassification,
    pub consumer: Option<WeightConsumerKey>,
    pub dtype: TensorDType,
    pub shape: Vec<u64>,
    pub source_file: String,
    pub locked_file_size: u64,
    pub locked_file_sha256: String,
    pub source_range: [u64; 2],
    pub destination_start: Option<u64>,
    pub chunks: Vec<WeightLoadChunk>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightLoadPlan {
    pub schema_version: String,
    pub repo_id: String,
    pub resolved_revision: String,
    pub lock_fingerprint: String,
    pub tied_embeddings: bool,
    pub chunk_size: u64,
    pub total_destination_bytes: u64,
    pub entries: Vec<WeightLoadEntry>,
    digest: [u8; 32],
}

impl WeightLoadPlan {
    pub fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    pub fn digest_hex(&self) -> String {
        let mut output = String::with_capacity(7 + self.digest.len() * 2);
        output.push_str("sha256:");
        for byte in self.digest {
            use std::fmt::Write as _;
            write!(output, "{byte:02x}").expect("writing to String cannot fail");
        }
        output
    }

    fn recompute_digest(&self) -> Result<[u8; 32], WeightPlanError> {
        digest_plan(
            &PlanDigestHeader {
                schema_version: &self.schema_version,
                repo_id: &self.repo_id,
                resolved_revision: &self.resolved_revision,
                fingerprint: &self.lock_fingerprint,
                tied_embeddings: self.tied_embeddings,
                chunk_size: self.chunk_size,
                total_destination_bytes: self.total_destination_bytes,
            },
            &self.entries,
        )
    }
}

/// A single verified tensor upload request.
///
/// `destination` is the exact tensor-sized target range. Plan-global offsets
/// are translated relative to that range, so callers may use either a packed
/// model allocation or a tensor-specific allocation without changing the
/// source plan.
pub struct WeightUploadRequest<'a> {
    pub plan: &'a WeightLoadPlan,
    pub expected_plan_digest: [u8; 32],
    pub cache: &'a VerifiedCache,
    pub tensor_name: &'a str,
    pub expected_dtype: TensorDType,
    pub session: &'a ExecutionSession,
    pub queue: &'a ExecutionQueue,
    pub destination: BufferRange,
    pub completion_timeout: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightUploadReceipt {
    pub plan_digest: [u8; 32],
    pub tensor_name: String,
    pub dtype: TensorDType,
    pub source_range: [u64; 2],
    pub destination_offset: u64,
    pub byte_length: u64,
    pub chunks_uploaded: usize,
    pub peak_host_staging_bytes: u64,
}

trait WeightRangeSource {
    fn lock_fingerprint(&self) -> &str;
    fn tensor(&self, tensor_name: &str) -> Option<&TensorDescriptor>;
    fn read_tensor_range(
        &self,
        tensor_name: &str,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, WeightUploadError>;
}

impl WeightRangeSource for VerifiedCache {
    fn lock_fingerprint(&self) -> &str {
        &self.lock_fingerprint
    }

    fn tensor(&self, tensor_name: &str) -> Option<&TensorDescriptor> {
        self.tensor(tensor_name)
    }

    fn read_tensor_range(
        &self,
        tensor_name: &str,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, WeightUploadError> {
        self.read_tensor_range(tensor_name, offset, length)
            .map_err(|error| WeightUploadError::invalid(error.to_string()))
    }
}

/// Upload one load-plan entry through the backend-neutral transfer API.
///
/// Only one plan chunk is resident in host staging memory at a time. A failed
/// call yields no receipt; callers must discard the partially written target.
pub fn upload_verified_weight(
    request: WeightUploadRequest<'_>,
) -> Result<WeightUploadReceipt, WeightUploadError> {
    if request.completion_timeout.is_zero() {
        return Err(WeightUploadError::invalid(
            "completion timeout must be non-zero",
        ));
    }
    if request.queue.session_id() != request.session.id()
        || request.destination.buffer().session_id() != request.session.id()
    {
        return Err(WeightUploadError::invalid(
            "queue and destination must belong to the upload session",
        ));
    }
    let max_transfer_bytes = request
        .session
        .max_transfer_bytes()
        .map_err(|error| WeightUploadError::invalid(error.to_string()))?;
    let destination = request.destination.clone();
    upload_weight_from_source(
        request.plan,
        request.expected_plan_digest,
        request.cache,
        request.tensor_name,
        request.expected_dtype,
        destination.offset_bytes(),
        destination.size_bytes(),
        max_transfer_bytes,
        |relative_offset, bytes| {
            let absolute_offset = destination
                .offset_bytes()
                .checked_add(relative_offset)
                .ok_or_else(|| WeightUploadError::invalid("destination offset overflow"))?;
            let range = destination
                .buffer()
                .range(
                    absolute_offset,
                    u64::try_from(bytes.len()).map_err(|_| {
                        WeightUploadError::invalid("upload byte length does not fit u64")
                    })?,
                )
                .map_err(|error| WeightUploadError::invalid(error.to_string()))?;
            let mut transfer = request
                .session
                .upload(request.queue, range, bytes)
                .map_err(|error| WeightUploadError::invalid(error.to_string()))?;
            match transfer
                .wait(request.completion_timeout)
                .map_err(|error| WeightUploadError::invalid(error.to_string()))?
            {
                ExecutionState::Success => Ok(()),
                ExecutionState::Pending => Err(WeightUploadError::invalid(
                    "weight upload remained pending after wait",
                )),
                ExecutionState::Failure => {
                    Err(WeightUploadError::invalid("weight upload reported failure"))
                }
            }
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn upload_weight_from_source<S, F>(
    plan: &WeightLoadPlan,
    expected_plan_digest: [u8; 32],
    source: &S,
    tensor_name: &str,
    expected_dtype: TensorDType,
    destination_offset: u64,
    destination_size: u64,
    max_transfer_bytes: u64,
    mut upload: F,
) -> Result<WeightUploadReceipt, WeightUploadError>
where
    S: WeightRangeSource,
    F: FnMut(u64, Arc<[u8]>) -> Result<(), WeightUploadError>,
{
    let recomputed = plan
        .recompute_digest()
        .map_err(|error| WeightUploadError::invalid(error.to_string()))?;
    if recomputed != *plan.digest() || expected_plan_digest != *plan.digest() {
        return Err(WeightUploadError::invalid(
            "weight plan identity or content digest differs",
        ));
    }
    if source.lock_fingerprint() != plan.lock_fingerprint {
        return Err(WeightUploadError::invalid(
            "verified cache fingerprint differs from the weight plan",
        ));
    }
    if max_transfer_bytes == 0 {
        return Err(WeightUploadError::invalid(
            "backend transfer limit must be non-zero",
        ));
    }
    let entry = plan
        .entries
        .binary_search_by(|entry| entry.tensor_name.as_str().cmp(tensor_name))
        .ok()
        .and_then(|index| plan.entries.get(index))
        .ok_or_else(|| WeightUploadError::invalid("tensor target is absent from the load plan"))?;
    if entry.classification == WeightClassification::KnownUnconsumed
        || entry.destination_start.is_none()
        || entry.chunks.is_empty()
    {
        return Err(WeightUploadError::invalid(
            "tensor target is not a loadable main-text weight",
        ));
    }
    if entry.dtype != expected_dtype {
        return Err(WeightUploadError::invalid(
            "tensor dtype differs from the requested upload dtype",
        ));
    }
    let descriptor = source
        .tensor(tensor_name)
        .ok_or_else(|| WeightUploadError::invalid("tensor is absent from the verified cache"))?;
    if descriptor.tensor_name != entry.tensor_name
        || descriptor.source_file != entry.source_file
        || descriptor.dtype != entry.dtype
        || descriptor.shape != entry.shape
        || descriptor.absolute_byte_range != entry.source_range
    {
        return Err(WeightUploadError::invalid(
            "verified tensor descriptor differs from the load-plan entry",
        ));
    }
    let source_size = entry.source_range[1]
        .checked_sub(entry.source_range[0])
        .ok_or_else(|| WeightUploadError::invalid("tensor source range underflow"))?;
    if source_size == 0 || destination_size != source_size {
        return Err(WeightUploadError::invalid(
            "destination range must exactly match the tensor byte range",
        ));
    }
    let plan_destination = entry
        .destination_start
        .expect("loadable entry has destination");
    let mut expected_relative = 0_u64;
    let mut peak_host_staging_bytes = 0_u64;
    for chunk in &entry.chunks {
        if chunk.byte_length == 0
            || chunk.byte_length > plan.chunk_size
            || chunk.byte_length > max_transfer_bytes
        {
            return Err(WeightUploadError::invalid(
                "weight chunk exceeds a non-zero plan/backend transfer bound",
            ));
        }
        let source_relative = chunk
            .source_offset
            .checked_sub(entry.source_range[0])
            .ok_or_else(|| WeightUploadError::invalid("chunk source precedes tensor range"))?;
        let destination_relative = chunk
            .destination_offset
            .checked_sub(plan_destination)
            .ok_or_else(|| WeightUploadError::invalid("chunk destination precedes tensor range"))?;
        if source_relative != expected_relative || destination_relative != expected_relative {
            return Err(WeightUploadError::invalid(
                "weight chunks are not contiguous source/destination peers",
            ));
        }
        let next = expected_relative
            .checked_add(chunk.byte_length)
            .ok_or_else(|| WeightUploadError::invalid("weight chunk range overflow"))?;
        if next > source_size {
            return Err(WeightUploadError::invalid(
                "weight chunk exceeds the tensor range",
            ));
        }
        let length = usize::try_from(chunk.byte_length)
            .map_err(|_| WeightUploadError::invalid("weight chunk length does not fit usize"))?;
        let bytes = source.read_tensor_range(tensor_name, source_relative, length)?;
        if bytes.len() != length {
            return Err(WeightUploadError::invalid(
                "verified range reader returned a short or long chunk",
            ));
        }
        peak_host_staging_bytes = peak_host_staging_bytes.max(chunk.byte_length);
        upload(destination_relative, Arc::<[u8]>::from(bytes))?;
        expected_relative = next;
    }
    if expected_relative != source_size {
        return Err(WeightUploadError::invalid(
            "weight chunks do not cover the exact tensor range",
        ));
    }
    Ok(WeightUploadReceipt {
        plan_digest: expected_plan_digest,
        tensor_name: entry.tensor_name.clone(),
        dtype: entry.dtype,
        source_range: entry.source_range,
        destination_offset,
        byte_length: source_size,
        chunks_uploaded: entry.chunks.len(),
        peak_host_staging_bytes,
    })
}

/// Build the fixed Qwen3.5-4B load plan from already-verified descriptors.
///
/// Input order is intentionally ignored. Entries are emitted in Rust `Ord`
/// tensor-name order, and only loadable main-text entries consume destination
/// space. Vision and MTP descriptors remain represented as known-unconsumed.
pub fn build_weight_load_plan<'a>(
    lock: &ModelLock,
    descriptors: impl IntoIterator<Item = &'a TensorDescriptor>,
) -> Result<WeightLoadPlan, WeightPlanError> {
    validate_fixed_lock(lock)?;
    let locked_files = locked_file_map(lock)?;
    let mut by_name = BTreeMap::new();
    for descriptor in descriptors {
        if by_name
            .insert(descriptor.tensor_name.as_str(), descriptor)
            .is_some()
        {
            return Err(WeightPlanError::invalid(format!(
                "duplicate tensor descriptor: {}",
                descriptor.tensor_name
            )));
        }
    }

    let architecture = &lock.model.architecture;
    let config = &architecture.text_config;
    let expected_consumers = expected_consumers(&config.layer_types, config.tie_word_embeddings);
    let mut observed_consumers = BTreeSet::new();
    let mut vision_count = 0_u64;
    let mut mtp_count = 0_u64;
    let mut entries = Vec::with_capacity(by_name.len());
    let mut destination_cursor = 0_u64;

    for descriptor in by_name.values() {
        validate_descriptor(descriptor, &locked_files)?;
        let (classification, consumer) = classify_descriptor(
            descriptor,
            &config.layer_types,
            &architecture.vision.tensor_prefix,
            &architecture.mtp.tensor_prefix,
            config.tie_word_embeddings,
        )?;
        if let Some(key) = consumer {
            if !observed_consumers.insert(key) {
                return Err(WeightPlanError::invalid(format!(
                    "duplicate weight consumer: {key:?}"
                )));
            }
        }
        match classification {
            WeightClassification::KnownUnconsumed => {
                if descriptor
                    .tensor_name
                    .starts_with(&architecture.vision.tensor_prefix)
                {
                    vision_count = vision_count
                        .checked_add(1)
                        .ok_or_else(|| WeightPlanError::invalid("vision tensor count overflow"))?;
                } else {
                    mtp_count = mtp_count
                        .checked_add(1)
                        .ok_or_else(|| WeightPlanError::invalid("MTP tensor count overflow"))?;
                }
            }
            WeightClassification::Required | WeightClassification::ConfigConditional => {}
        }

        let locked = locked_files
            .get(descriptor.source_file.as_str())
            .expect("descriptor source was validated");
        let (destination_start, chunks) = if classification == WeightClassification::KnownUnconsumed
        {
            (None, Vec::new())
        } else {
            let destination_start = destination_cursor;
            let chunks = split_chunks(descriptor, destination_start)?;
            destination_cursor = destination_cursor
                .checked_add(descriptor.byte_size)
                .ok_or_else(|| WeightPlanError::invalid("destination size overflow"))?;
            (Some(destination_start), chunks)
        };
        entries.push(WeightLoadEntry {
            tensor_name: descriptor.tensor_name.clone(),
            classification,
            consumer,
            dtype: descriptor.dtype,
            shape: descriptor.shape.clone(),
            source_file: descriptor.source_file.clone(),
            locked_file_size: locked.size_bytes,
            locked_file_sha256: locked.sha256.clone(),
            source_range: descriptor.absolute_byte_range,
            destination_start,
            chunks,
        });
    }

    if observed_consumers != expected_consumers {
        let missing: Vec<_> = expected_consumers
            .difference(&observed_consumers)
            .copied()
            .collect();
        let unexpected: Vec<_> = observed_consumers
            .difference(&expected_consumers)
            .copied()
            .collect();
        return Err(WeightPlanError::invalid(format!(
            "weight consumer set differs: missing={missing:?}, unexpected={unexpected:?}"
        )));
    }
    if vision_count != architecture.vision.tensor_count
        || mtp_count != architecture.mtp.tensor_count
    {
        return Err(WeightPlanError::invalid(format!(
            "known-unconsumed tensor count differs: vision={vision_count}/{}, mtp={mtp_count}/{}",
            architecture.vision.tensor_count, architecture.mtp.tensor_count
        )));
    }

    let digest = digest_plan(
        &PlanDigestHeader {
            schema_version: &lock.schema_version,
            repo_id: &lock.model.repo_id,
            resolved_revision: &lock.model.resolved_revision,
            fingerprint: lock.fingerprint(),
            tied_embeddings: config.tie_word_embeddings,
            chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
            total_destination_bytes: destination_cursor,
        },
        &entries,
    )?;
    Ok(WeightLoadPlan {
        schema_version: lock.schema_version.clone(),
        repo_id: lock.model.repo_id.clone(),
        resolved_revision: lock.model.resolved_revision.clone(),
        lock_fingerprint: lock.fingerprint().to_owned(),
        tied_embeddings: config.tie_word_embeddings,
        chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
        total_destination_bytes: destination_cursor,
        entries,
        digest,
    })
}

/// Thin wrapper that binds the plan to the verified cache fingerprint.
pub fn build_verified_weight_load_plan(
    lock: &ModelLock,
    cache: &VerifiedCache,
) -> Result<WeightLoadPlan, WeightPlanError> {
    if cache.lock_fingerprint != lock.fingerprint() {
        return Err(WeightPlanError::invalid(
            "verified cache fingerprint differs from the model lock",
        ));
    }
    build_weight_load_plan(lock, cache.tensors())
}

fn validate_fixed_lock(lock: &ModelLock) -> Result<(), WeightPlanError> {
    let config = &lock.model.architecture.text_config;
    if lock.schema_version != QWEN_SCHEMA_VERSION || reviewed_qwen35_spec(lock).is_none() {
        return Err(WeightPlanError::invalid(
            "model identity is not a reviewed Qwen3.5 dense contract",
        ));
    }
    if config.num_hidden_layers
        != u64::try_from(config.layer_types.len())
            .map_err(|_| WeightPlanError::invalid("layer count does not fit u64"))?
    {
        return Err(WeightPlanError::invalid(
            "layer schedule length differs from num_hidden_layers",
        ));
    }
    Ok(())
}

fn locked_file_map(lock: &ModelLock) -> Result<BTreeMap<&str, &LockedFile>, WeightPlanError> {
    let mut files = BTreeMap::new();
    for file in &lock.model.files {
        if file.sha256.len() != 64
            || !file
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(WeightPlanError::invalid(format!(
                "locked file SHA-256 is not lowercase hex: {}",
                file.path
            )));
        }
        if files.insert(file.path.as_str(), file).is_some() {
            return Err(WeightPlanError::invalid(format!(
                "duplicate locked file: {}",
                file.path
            )));
        }
    }
    Ok(files)
}

fn validate_descriptor(
    descriptor: &TensorDescriptor,
    locked_files: &BTreeMap<&str, &LockedFile>,
) -> Result<(), WeightPlanError> {
    let [start, end] = descriptor.absolute_byte_range;
    if start >= end || descriptor.byte_size == 0 {
        return Err(WeightPlanError::invalid(format!(
            "tensor has an empty or reversed range: {}",
            descriptor.tensor_name
        )));
    }
    let span = end
        .checked_sub(start)
        .ok_or_else(|| WeightPlanError::invalid("tensor source range underflow"))?;
    if span != descriptor.byte_size {
        return Err(WeightPlanError::invalid(format!(
            "tensor range size differs from byte_size: {}",
            descriptor.tensor_name
        )));
    }
    let locked = locked_files
        .get(descriptor.source_file.as_str())
        .ok_or_else(|| {
            WeightPlanError::invalid(format!(
                "tensor source is not a locked file: {}",
                descriptor.source_file
            ))
        })?;
    if end > locked.size_bytes {
        return Err(WeightPlanError::invalid(format!(
            "tensor range exceeds locked source file: {}",
            descriptor.tensor_name
        )));
    }
    Ok(())
}

fn classify_descriptor(
    descriptor: &TensorDescriptor,
    layer_types: &[LayerType],
    vision_prefix: &str,
    mtp_prefix: &str,
    tied_embeddings: bool,
) -> Result<(WeightClassification, Option<WeightConsumerKey>), WeightPlanError> {
    let name = descriptor.tensor_name.as_str();
    let top_level = match name {
        "model.language_model.embed_tokens.weight" => Some(if tied_embeddings {
            WeightConsumer::EmbeddingAndTiedOutput
        } else {
            WeightConsumer::Embedding
        }),
        "model.language_model.norm.weight" => Some(WeightConsumer::FinalNorm),
        "lm_head.weight" if !tied_embeddings => Some(WeightConsumer::OutputProjection),
        "lm_head.weight" | "model.language_model.lm_head.weight" => {
            return Err(WeightPlanError::invalid(
                "independent lm_head contradicts tied embeddings",
            ));
        }
        _ => None,
    };
    if let Some(role) = top_level {
        return Ok((
            WeightClassification::Required,
            Some(WeightConsumerKey { layer: None, role }),
        ));
    }
    if name.starts_with(vision_prefix) || name.starts_with(mtp_prefix) {
        return Ok((WeightClassification::KnownUnconsumed, None));
    }

    const LAYER_PREFIX: &str = "model.language_model.layers.";
    let remainder = name.strip_prefix(LAYER_PREFIX).ok_or_else(|| {
        WeightPlanError::invalid(format!("unknown tensor name: {}", descriptor.tensor_name))
    })?;
    let (layer_text, suffix) = remainder.split_once('.').ok_or_else(|| {
        WeightPlanError::invalid(format!(
            "malformed layer tensor: {}",
            descriptor.tensor_name
        ))
    })?;
    let layer = layer_text.parse::<u64>().map_err(|_| {
        WeightPlanError::invalid(format!("invalid layer index: {}", descriptor.tensor_name))
    })?;
    if layer.to_string() != layer_text {
        return Err(WeightPlanError::invalid(format!(
            "layer index is not canonical decimal: {}",
            descriptor.tensor_name
        )));
    }
    let layer_index = usize::try_from(layer)
        .map_err(|_| WeightPlanError::invalid("layer index does not fit usize"))?;
    let layer_type = layer_types
        .get(layer_index)
        .ok_or_else(|| WeightPlanError::invalid(format!("layer index is out of range: {layer}")))?;
    let common = match suffix {
        "input_layernorm.weight" => Some(WeightConsumer::InputNorm),
        "post_attention_layernorm.weight" => Some(WeightConsumer::PostAttentionNorm),
        "mlp.gate_proj.weight" => Some(WeightConsumer::MlpGate),
        "mlp.up_proj.weight" => Some(WeightConsumer::MlpUp),
        "mlp.down_proj.weight" => Some(WeightConsumer::MlpDown),
        _ => None,
    };
    let role = if let Some(role) = common {
        role
    } else {
        match (layer_type, suffix) {
            (LayerType::LinearAttention, "linear_attn.in_proj_qkv.weight") => {
                WeightConsumer::GdnInProjQkv
            }
            (LayerType::LinearAttention, "linear_attn.in_proj_z.weight") => {
                WeightConsumer::GdnInProjZ
            }
            (LayerType::LinearAttention, "linear_attn.in_proj_b.weight") => {
                WeightConsumer::GdnInProjB
            }
            (LayerType::LinearAttention, "linear_attn.in_proj_a.weight") => {
                WeightConsumer::GdnInProjA
            }
            (LayerType::LinearAttention, "linear_attn.conv1d.weight") => WeightConsumer::GdnConv1d,
            (LayerType::LinearAttention, "linear_attn.A_log") => WeightConsumer::GdnALog,
            (LayerType::LinearAttention, "linear_attn.dt_bias") => WeightConsumer::GdnDtBias,
            (LayerType::LinearAttention, "linear_attn.norm.weight") => WeightConsumer::GdnNorm,
            (LayerType::LinearAttention, "linear_attn.out_proj.weight") => {
                WeightConsumer::GdnOutProj
            }
            (LayerType::FullAttention, "self_attn.q_proj.weight") => WeightConsumer::AttentionQ,
            (LayerType::FullAttention, "self_attn.k_proj.weight") => WeightConsumer::AttentionK,
            (LayerType::FullAttention, "self_attn.v_proj.weight") => WeightConsumer::AttentionV,
            (LayerType::FullAttention, "self_attn.o_proj.weight") => WeightConsumer::AttentionO,
            (LayerType::FullAttention, "self_attn.q_norm.weight") => WeightConsumer::AttentionQNorm,
            (LayerType::FullAttention, "self_attn.k_norm.weight") => WeightConsumer::AttentionKNorm,
            _ => {
                return Err(WeightPlanError::invalid(format!(
                    "tensor suffix is invalid for its layer class: {}",
                    descriptor.tensor_name
                )));
            }
        }
    };
    Ok((
        WeightClassification::Required,
        Some(WeightConsumerKey {
            layer: Some(layer),
            role,
        }),
    ))
}

fn expected_consumers(
    layer_types: &[LayerType],
    tied_embeddings: bool,
) -> BTreeSet<WeightConsumerKey> {
    let embedding_role = if tied_embeddings {
        WeightConsumer::EmbeddingAndTiedOutput
    } else {
        WeightConsumer::Embedding
    };
    let mut expected = BTreeSet::from([
        WeightConsumerKey {
            layer: None,
            role: embedding_role,
        },
        WeightConsumerKey {
            layer: None,
            role: WeightConsumer::FinalNorm,
        },
    ]);
    if !tied_embeddings {
        expected.insert(WeightConsumerKey {
            layer: None,
            role: WeightConsumer::OutputProjection,
        });
    }
    for (layer, layer_type) in layer_types.iter().enumerate() {
        let layer = u64::try_from(layer).expect("Qwen layer index fits u64");
        for role in [
            WeightConsumer::InputNorm,
            WeightConsumer::PostAttentionNorm,
            WeightConsumer::MlpGate,
            WeightConsumer::MlpUp,
            WeightConsumer::MlpDown,
        ] {
            expected.insert(WeightConsumerKey {
                layer: Some(layer),
                role,
            });
        }
        let roles: &[WeightConsumer] = match layer_type {
            LayerType::LinearAttention => &[
                WeightConsumer::GdnInProjQkv,
                WeightConsumer::GdnInProjZ,
                WeightConsumer::GdnInProjB,
                WeightConsumer::GdnInProjA,
                WeightConsumer::GdnConv1d,
                WeightConsumer::GdnALog,
                WeightConsumer::GdnDtBias,
                WeightConsumer::GdnNorm,
                WeightConsumer::GdnOutProj,
            ],
            LayerType::FullAttention => &[
                WeightConsumer::AttentionQ,
                WeightConsumer::AttentionK,
                WeightConsumer::AttentionV,
                WeightConsumer::AttentionO,
                WeightConsumer::AttentionQNorm,
                WeightConsumer::AttentionKNorm,
            ],
        };
        for &role in roles {
            expected.insert(WeightConsumerKey {
                layer: Some(layer),
                role,
            });
        }
    }
    expected
}

fn split_chunks(
    descriptor: &TensorDescriptor,
    destination_start: u64,
) -> Result<Vec<WeightLoadChunk>, WeightPlanError> {
    let mut chunks = Vec::new();
    let mut relative = 0_u64;
    while relative < descriptor.byte_size {
        let remaining = descriptor
            .byte_size
            .checked_sub(relative)
            .ok_or_else(|| WeightPlanError::invalid("chunk remaining underflow"))?;
        let byte_length = remaining.min(WEIGHT_LOAD_CHUNK_BYTES);
        let source_offset = descriptor
            .absolute_start()
            .checked_add(relative)
            .ok_or_else(|| WeightPlanError::invalid("chunk source offset overflow"))?;
        let destination_offset = destination_start
            .checked_add(relative)
            .ok_or_else(|| WeightPlanError::invalid("chunk destination offset overflow"))?;
        chunks.push(WeightLoadChunk {
            source_offset,
            destination_offset,
            byte_length,
        });
        relative = relative
            .checked_add(byte_length)
            .ok_or_else(|| WeightPlanError::invalid("chunk cursor overflow"))?;
    }
    Ok(chunks)
}

struct PlanDigestHeader<'a> {
    schema_version: &'a str,
    repo_id: &'a str,
    resolved_revision: &'a str,
    fingerprint: &'a str,
    tied_embeddings: bool,
    chunk_size: u64,
    total_destination_bytes: u64,
}

fn digest_plan(
    header: &PlanDigestHeader<'_>,
    entries: &[WeightLoadEntry],
) -> Result<[u8; 32], WeightPlanError> {
    let mut encoded = Vec::new();
    encoded.extend_from_slice(PLAN_DOMAIN);
    put_string(&mut encoded, header.schema_version)?;
    put_string(&mut encoded, header.repo_id)?;
    put_string(&mut encoded, header.resolved_revision)?;
    put_string(&mut encoded, header.fingerprint)?;
    encoded.push(u8::from(header.tied_embeddings));
    put_u64(&mut encoded, header.chunk_size);
    put_u64(
        &mut encoded,
        u64::try_from(entries.len())
            .map_err(|_| WeightPlanError::invalid("entry count does not fit u64"))?,
    );
    put_u64(&mut encoded, header.total_destination_bytes);
    for entry in entries {
        put_string(&mut encoded, &entry.tensor_name)?;
        encoded.push(entry.classification.tag());
        match entry.consumer.and_then(|consumer| consumer.layer) {
            None => encoded.push(0),
            Some(layer) => {
                encoded.push(1);
                put_u64(&mut encoded, layer);
            }
        }
        encoded.push(entry.consumer.map_or(0, |consumer| consumer.role.tag()));
        encoded.push(dtype_tag(entry.dtype));
        put_u64(
            &mut encoded,
            u64::try_from(entry.shape.len())
                .map_err(|_| WeightPlanError::invalid("shape rank does not fit u64"))?,
        );
        for &dimension in &entry.shape {
            put_u64(&mut encoded, dimension);
        }
        put_string(&mut encoded, &entry.source_file)?;
        put_u64(&mut encoded, entry.locked_file_size);
        put_string(&mut encoded, &entry.locked_file_sha256)?;
        put_u64(&mut encoded, entry.source_range[0]);
        put_u64(&mut encoded, entry.source_range[1]);
        put_u64(&mut encoded, entry.destination_start.unwrap_or(0));
        put_u64(
            &mut encoded,
            u64::try_from(entry.chunks.len())
                .map_err(|_| WeightPlanError::invalid("chunk count does not fit u64"))?,
        );
        for chunk in &entry.chunks {
            put_u64(&mut encoded, chunk.source_offset);
            put_u64(&mut encoded, chunk.destination_offset);
            put_u64(&mut encoded, chunk.byte_length);
        }
    }
    Ok(Sha256::digest(encoded).into())
}

fn put_string(output: &mut Vec<u8>, value: &str) -> Result<(), WeightPlanError> {
    put_u64(
        output,
        u64::try_from(value.len())
            .map_err(|_| WeightPlanError::invalid("string length does not fit u64"))?,
    );
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn put_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn dtype_tag(dtype: TensorDType) -> u8 {
    match dtype {
        TensorDType::Bf16 => 1,
        TensorDType::F16 => 2,
        TensorDType::F32 => 3,
        TensorDType::I32 => 4,
        TensorDType::I64 => 5,
        TensorDType::U8 => 6,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    struct FakeWeightSource {
        fingerprint: String,
        descriptor: TensorDescriptor,
        bytes: Vec<u8>,
        reads: Cell<usize>,
    }

    impl WeightRangeSource for FakeWeightSource {
        fn lock_fingerprint(&self) -> &str {
            &self.fingerprint
        }

        fn tensor(&self, tensor_name: &str) -> Option<&TensorDescriptor> {
            (tensor_name == self.descriptor.tensor_name).then_some(&self.descriptor)
        }

        fn read_tensor_range(
            &self,
            tensor_name: &str,
            offset: u64,
            length: usize,
        ) -> Result<Vec<u8>, WeightUploadError> {
            self.reads.set(self.reads.get() + 1);
            if tensor_name != self.descriptor.tensor_name {
                return Err(WeightUploadError::invalid("unexpected fake tensor"));
            }
            let start = usize::try_from(offset)
                .map_err(|_| WeightUploadError::invalid("fake offset overflow"))?;
            let end = start
                .checked_add(length)
                .ok_or_else(|| WeightUploadError::invalid("fake range overflow"))?;
            self.bytes
                .get(start..end)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| WeightUploadError::invalid("fake range is out of bounds"))
        }
    }

    fn upload_fixture() -> (WeightLoadPlan, FakeWeightSource) {
        let byte_size = WEIGHT_LOAD_CHUNK_BYTES + 1;
        let tensor_name = "model.language_model.norm.weight";
        let source_file = "model-00001-of-00002.safetensors";
        let descriptor = TensorDescriptor {
            tensor_name: tensor_name.to_owned(),
            source_file: source_file.to_owned(),
            dtype: TensorDType::Bf16,
            shape: vec![byte_size],
            header_length_field_bytes: 8,
            header_length_bytes: 9,
            data_buffer_start: 17,
            data_offset_basis: "safetensors-data-buffer".to_owned(),
            data_offsets: [0, byte_size],
            absolute_byte_range: [17, 17 + byte_size],
            byte_size,
        };
        let chunks = split_chunks(&descriptor, 91).unwrap();
        let entry = WeightLoadEntry {
            tensor_name: tensor_name.to_owned(),
            classification: WeightClassification::Required,
            consumer: Some(WeightConsumerKey {
                layer: None,
                role: WeightConsumer::FinalNorm,
            }),
            dtype: TensorDType::Bf16,
            shape: descriptor.shape.clone(),
            source_file: source_file.to_owned(),
            locked_file_size: descriptor.absolute_end(),
            locked_file_sha256: "0".repeat(64),
            source_range: descriptor.absolute_byte_range,
            destination_start: Some(91),
            chunks,
        };
        let mut plan = WeightLoadPlan {
            schema_version: QWEN_SCHEMA_VERSION.to_owned(),
            repo_id: QWEN_REPO_ID.to_owned(),
            resolved_revision: QWEN_REVISION.to_owned(),
            lock_fingerprint: QWEN_FINGERPRINT.to_owned(),
            tied_embeddings: true,
            chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
            total_destination_bytes: 91 + byte_size,
            entries: vec![entry],
            digest: [0; 32],
        };
        plan.digest = plan.recompute_digest().unwrap();
        let bytes = (0..byte_size)
            .map(|offset| ((offset * 37 + 11) % 251) as u8)
            .collect();
        (
            plan,
            FakeWeightSource {
                fingerprint: QWEN_FINGERPRINT.to_owned(),
                descriptor,
                bytes,
                reads: Cell::new(0),
            },
        )
    }

    #[test]
    fn canonical_wire_vector_matches_the_reader_contract() {
        let entry = WeightLoadEntry {
            tensor_name: "x".to_owned(),
            classification: WeightClassification::Required,
            consumer: Some(WeightConsumerKey {
                layer: None,
                role: WeightConsumer::EmbeddingAndTiedOutput,
            }),
            dtype: TensorDType::Bf16,
            shape: vec![3],
            source_file: "model-00001-of-00002.safetensors".to_owned(),
            locked_file_size: 20,
            locked_file_sha256: "0".repeat(64),
            source_range: [17, 20],
            destination_start: Some(0),
            chunks: vec![WeightLoadChunk {
                source_offset: 17,
                destination_offset: 0,
                byte_length: 3,
            }],
        };
        let mut encoded = Vec::new();
        encoded.extend_from_slice(PLAN_DOMAIN);
        put_string(&mut encoded, QWEN_SCHEMA_VERSION).unwrap();
        put_string(&mut encoded, QWEN_REPO_ID).unwrap();
        put_string(&mut encoded, QWEN_REVISION).unwrap();
        put_string(&mut encoded, QWEN_FINGERPRINT).unwrap();
        encoded.push(1);
        put_u64(&mut encoded, WEIGHT_LOAD_CHUNK_BYTES);
        put_u64(&mut encoded, 1);
        put_u64(&mut encoded, 3);
        put_string(&mut encoded, &entry.tensor_name).unwrap();
        encoded.push(entry.classification.tag());
        encoded.push(0);
        encoded.push(entry.consumer.unwrap().role.tag());
        encoded.push(dtype_tag(entry.dtype));
        put_u64(&mut encoded, 1);
        put_u64(&mut encoded, 3);
        put_string(&mut encoded, &entry.source_file).unwrap();
        put_u64(&mut encoded, entry.locked_file_size);
        put_string(&mut encoded, &entry.locked_file_sha256).unwrap();
        put_u64(&mut encoded, 17);
        put_u64(&mut encoded, 20);
        put_u64(&mut encoded, 0);
        put_u64(&mut encoded, 1);
        put_u64(&mut encoded, 17);
        put_u64(&mut encoded, 0);
        put_u64(&mut encoded, 3);
        assert_eq!(encoded.len(), 426);
        let digest = Sha256::digest(encoded);
        assert_eq!(
            format!("{digest:x}"),
            "a8ee5d60d9dffeac3020d1b3833677eb66e50958a53412f1ccb1e9c09c174fef"
        );
        let expected_digest: [u8; 32] = digest.into();
        assert_eq!(
            digest_plan(
                &PlanDigestHeader {
                    schema_version: QWEN_SCHEMA_VERSION,
                    repo_id: QWEN_REPO_ID,
                    resolved_revision: QWEN_REVISION,
                    fingerprint: QWEN_FINGERPRINT,
                    tied_embeddings: true,
                    chunk_size: WEIGHT_LOAD_CHUNK_BYTES,
                    total_destination_bytes: 3,
                },
                &[entry],
            )
            .unwrap(),
            expected_digest
        );
    }

    #[test]
    fn chunk_boundaries_are_exact_and_nonzero() {
        for (bytes, expected) in [
            (1, vec![1]),
            (3, vec![3]),
            (17, vec![17]),
            (
                WEIGHT_LOAD_CHUNK_BYTES - 1,
                vec![WEIGHT_LOAD_CHUNK_BYTES - 1],
            ),
            (WEIGHT_LOAD_CHUNK_BYTES, vec![WEIGHT_LOAD_CHUNK_BYTES]),
            (
                WEIGHT_LOAD_CHUNK_BYTES + 1,
                vec![WEIGHT_LOAD_CHUNK_BYTES, 1],
            ),
        ] {
            let descriptor = TensorDescriptor {
                tensor_name: "x".to_owned(),
                source_file: "x.safetensors".to_owned(),
                dtype: TensorDType::Bf16,
                shape: vec![bytes],
                header_length_field_bytes: 8,
                header_length_bytes: 1,
                data_buffer_start: 9,
                data_offset_basis: "safetensors-data-buffer".to_owned(),
                data_offsets: [0, bytes],
                absolute_byte_range: [17, 17 + bytes],
                byte_size: bytes,
            };
            let chunks = split_chunks(&descriptor, 3).unwrap();
            assert_eq!(
                chunks
                    .iter()
                    .map(|chunk| chunk.byte_length)
                    .collect::<Vec<_>>(),
                expected
            );
            assert!(chunks.iter().all(|chunk| chunk.byte_length > 0));
        }
    }

    #[test]
    fn upload_bridge_reads_and_submits_one_exact_chunk_at_a_time() {
        let (plan, source) = upload_fixture();
        let mut calls = 0_usize;
        let receipt = upload_weight_from_source(
            &plan,
            *plan.digest(),
            &source,
            "model.language_model.norm.weight",
            TensorDType::Bf16,
            7,
            WEIGHT_LOAD_CHUNK_BYTES + 1,
            WEIGHT_LOAD_CHUNK_BYTES,
            |relative_offset, bytes| {
                let start = usize::try_from(relative_offset).unwrap();
                assert_eq!(bytes.as_ref(), &source.bytes[start..start + bytes.len()]);
                assert_eq!(relative_offset, calls as u64 * WEIGHT_LOAD_CHUNK_BYTES);
                calls += 1;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(source.reads.get(), 2);
        assert_eq!(receipt.chunks_uploaded, 2);
        assert_eq!(receipt.byte_length, WEIGHT_LOAD_CHUNK_BYTES + 1);
        assert_eq!(receipt.destination_offset, 7);
        assert_eq!(receipt.peak_host_staging_bytes, WEIGHT_LOAD_CHUNK_BYTES);
    }

    #[test]
    fn upload_bridge_rejects_identity_target_dtype_and_range_before_read() {
        let (plan, source) = upload_fixture();
        let invoke = |plan: &WeightLoadPlan,
                      expected_digest: [u8; 32],
                      source: &FakeWeightSource,
                      tensor_name: &str,
                      dtype: TensorDType,
                      destination_size: u64| {
            upload_weight_from_source(
                plan,
                expected_digest,
                source,
                tensor_name,
                dtype,
                7,
                destination_size,
                WEIGHT_LOAD_CHUNK_BYTES,
                |_, _| panic!("invalid request must not submit a transfer"),
            )
        };

        let mut wrong_digest = *plan.digest();
        wrong_digest[0] ^= 1;
        assert!(
            invoke(
                &plan,
                wrong_digest,
                &source,
                "model.language_model.norm.weight",
                TensorDType::Bf16,
                WEIGHT_LOAD_CHUNK_BYTES + 1,
            )
            .is_err()
        );
        assert!(
            invoke(
                &plan,
                *plan.digest(),
                &source,
                "model.language_model.missing.weight",
                TensorDType::Bf16,
                WEIGHT_LOAD_CHUNK_BYTES + 1,
            )
            .is_err()
        );
        assert!(
            invoke(
                &plan,
                *plan.digest(),
                &source,
                "model.language_model.norm.weight",
                TensorDType::F32,
                WEIGHT_LOAD_CHUNK_BYTES + 1,
            )
            .is_err()
        );
        assert!(
            invoke(
                &plan,
                *plan.digest(),
                &source,
                "model.language_model.norm.weight",
                TensorDType::Bf16,
                WEIGHT_LOAD_CHUNK_BYTES,
            )
            .is_err()
        );
        assert_eq!(source.reads.get(), 0);
    }

    #[test]
    fn upload_bridge_rejects_mutated_plan_cache_and_descriptor_before_read() {
        let (plan, source) = upload_fixture();
        let mut mutated_plan = plan.clone();
        mutated_plan.entries[0].chunks[0].byte_length -= 1;
        assert!(
            upload_weight_from_source(
                &mutated_plan,
                *plan.digest(),
                &source,
                "model.language_model.norm.weight",
                TensorDType::Bf16,
                7,
                WEIGHT_LOAD_CHUNK_BYTES + 1,
                WEIGHT_LOAD_CHUNK_BYTES,
                |_, _| panic!("mutated plan must not submit"),
            )
            .is_err()
        );

        let (_, mut wrong_cache) = upload_fixture();
        wrong_cache.fingerprint = format!("sha256:{}", "1".repeat(64));
        assert!(
            upload_weight_from_source(
                &plan,
                *plan.digest(),
                &wrong_cache,
                "model.language_model.norm.weight",
                TensorDType::Bf16,
                7,
                WEIGHT_LOAD_CHUNK_BYTES + 1,
                WEIGHT_LOAD_CHUNK_BYTES,
                |_, _| panic!("wrong cache must not submit"),
            )
            .is_err()
        );

        let (_, mut wrong_descriptor) = upload_fixture();
        wrong_descriptor.descriptor.absolute_byte_range[0] += 1;
        assert!(
            upload_weight_from_source(
                &plan,
                *plan.digest(),
                &wrong_descriptor,
                "model.language_model.norm.weight",
                TensorDType::Bf16,
                7,
                WEIGHT_LOAD_CHUNK_BYTES + 1,
                WEIGHT_LOAD_CHUNK_BYTES,
                |_, _| panic!("wrong descriptor must not submit"),
            )
            .is_err()
        );
        assert_eq!(source.reads.get(), 0);
        assert_eq!(wrong_cache.reads.get(), 0);
        assert_eq!(wrong_descriptor.reads.get(), 0);
    }
}
