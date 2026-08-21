//! Request-local execution ownership for the fixed Qwen3.5-4B text graph.
//!
//! This is a host-side orchestration layer. It owns the checked device
//! buffers, Stage C state objects, and completion ordering for one request;
//! it neither implements numerical operators nor offers a CPU fallback.
//! Every operation reaches a backend only through the existing owned
//! execution/session contracts.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::StateForkAuditV1;
use crate::adapter::{
    AdapterRequestSetV1, ControlVectorSelectionV1, LoraAdapterSelectionV1, VerifiedLoraTargetV1,
};
use crate::context_window::{ContextShiftDecisionV1, ContextWindowStateV1};
use crate::execution::{
    ExecutionBuffer, ExecutionError, ExecutionQueue, ExecutionSession, ExecutionStateImageV1,
    KvState, KvStateAppendSubmission, LinearAttentionBindings, LinearAttentionState,
    OwnedTensorBinding, PrepareSupport, Submission,
};
use crate::final_output::QWEN35_VOCAB_SIZE;
use crate::kv_state::{
    CausalAttentionDescriptor, KvCacheEncoding, KvPhysicalMemorySnapshot, KvStateDescriptor,
};
use crate::linear_attention::{LinearAttentionDescriptor, LinearAttentionStateDescriptor};
use crate::model::{QWEN35_4B_FINGERPRINT, TensorDType, VerifiedCache};
use crate::op::{
    AttentionPreprocessContract, AttentionPreprocessPositionMode, OpError, SemanticOpDescriptor,
    SemanticOpKind, TokenSelectorContractV1,
};
use crate::prepared_execution::{
    ExecutionAuditAccumulator, ExecutionBoundaryKind, ExecutionSegment, ExecutionTransaction,
    PreparedCachePolicy, PreparedDynamicIdentity, PreparedExecutionError, PreparedExecutionPlan,
    PreparedPlanNode, PreparedSemanticCache, PreparedTransition, require_terminal_success,
};
use crate::qwen_graph::{
    QWEN_RUNTIME_MAX_CONTEXT_TOKENS, QwenGraph, QwenGraphNode, QwenGraphNodeKind,
    QwenGraphStateDescriptor, QwenGraphStateKind, QwenGraphTensorBacking, QwenGraphWeightBinding,
};
use crate::session_checkpoint::{
    CheckpointIdentity, CheckpointPayload, SessionCheckpoint, StateOwnerKindV1, StatePlaneKindV1,
};
use crate::tensor::{TensorError, TensorView};
use crate::weights::{
    GgufWeightUploadRequest, VerifiedGgufWeightSource, WeightClassification, WeightLoadEntry,
    WeightLoadPlan, WeightUploadError, WeightUploadReceipt, WeightUploadRequest,
    upload_verified_gguf_weight, upload_verified_weight,
};
use crate::{
    AccessMode, DType, Encoding, Fp8ResidentRepresentation, Fp8ScaleGranularity,
    QWEN35_MOE_LAYER_BLOB_BYTES, QWEN35_MOE_LAYER_BLOB_PREFIX, Qwen35MoeExpertProjection,
    Qwen35MoeTensorPlane, VerifiedFp8Sidecar, VerifiedGgufQwen35Moe, VerifiedNvfp4Sidecar,
    VerifiedQwen35Moe, decode_e4m3fn,
};
use crate::{DeviceTokenSelectorRequestV1, SamplingSelectionV1};

/// Output published by a fully completed Qwen request transition.
#[derive(Clone, Debug, PartialEq)]
pub struct QwenExecutionOutput {
    token_ids: Vec<i32>,
    last_logits: Option<Vec<f32>>,
    selection: Option<SamplingSelectionV1>,
    logits_bf16: Option<Vec<u16>>,
    hidden_states_bf16: Option<Vec<u16>>,
    /// Final-RMSNorm output rows used by the explicit embedding execution
    /// mode.  This is deliberately separate from the MTP pre-final hidden
    /// rows above; callers must not confuse the two representations.
    embeddings_bf16: Option<Vec<u16>>,
    committed_length: u64,
}

/// Evidence-only `(layer, key bytes, value bytes)` semantic KV payload.
pub type QwenKvPayloadEvidence = (u32, Vec<u8>, Vec<u8>);

/// Aggregated, redacted accounting for every state fork that made up one
/// immutable Qwen prefix owner.  The accounting never exposes a native
/// handle, pointer, page table, token id, or payload byte.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QwenPrefixForkAuditV1 {
    kv_states: u32,
    linear_states: u32,
    shared_pages: u64,
    copied_bytes: u64,
    destination_owned_bytes: u64,
    cache_resident_bytes: u64,
}

impl QwenPrefixForkAuditV1 {
    pub const fn kv_states(self) -> u32 {
        self.kv_states
    }

    pub const fn linear_states(self) -> u32 {
        self.linear_states
    }

    pub const fn shared_pages(self) -> u64 {
        self.shared_pages
    }

    pub const fn copied_bytes(self) -> u64 {
        self.copied_bytes
    }

    pub const fn destination_owned_bytes(self) -> u64 {
        self.destination_owned_bytes
    }

    /// Resident bytes attributable to this immutable prefix owner. Shared
    /// VMM KV pages are charged from backend physical-memory metadata; a
    /// backend without that optional metadata is charged the full descriptor
    /// footprint as a conservative compatibility fallback. Device-copy KV
    /// and all linear state use their destination-owned audit bytes.
    pub const fn cache_resident_bytes(self) -> u64 {
        self.cache_resident_bytes
    }

    fn add(
        &mut self,
        audit: StateForkAuditV1,
        linear: bool,
        kv_physical: Option<KvPhysicalMemorySnapshot>,
        kv_fallback_resident_bytes: u64,
    ) -> Result<(), QwenExecutionError> {
        if linear {
            self.linear_states = self.linear_states.checked_add(1).ok_or_else(|| {
                QwenExecutionError::InvalidRequest("linear fork count overflowed".to_owned())
            })?;
        } else {
            self.kv_states = self.kv_states.checked_add(1).ok_or_else(|| {
                QwenExecutionError::InvalidRequest("KV fork count overflowed".to_owned())
            })?;
        }
        self.shared_pages = self
            .shared_pages
            .checked_add(audit.shared_pages())
            .ok_or_else(|| {
                QwenExecutionError::InvalidRequest("shared page count overflowed".to_owned())
            })?;
        self.copied_bytes = self
            .copied_bytes
            .checked_add(audit.copied_bytes())
            .ok_or_else(|| {
                QwenExecutionError::InvalidRequest("fork copied-byte count overflowed".to_owned())
            })?;
        self.destination_owned_bytes = self
            .destination_owned_bytes
            .checked_add(audit.destination_owned_bytes())
            .ok_or_else(|| {
                QwenExecutionError::InvalidRequest("fork owned-byte count overflowed".to_owned())
            })?;
        let resident_bytes = if linear {
            audit.destination_owned_bytes()
        } else {
            match audit.mode() {
                crate::StateForkModeV1::SharedReadOnlyPages => kv_physical
                    .map(|physical| physical.committed_bytes_per_plane())
                    .and_then(|bytes| bytes.checked_mul(2))
                    .unwrap_or(kv_fallback_resident_bytes),
                crate::StateForkModeV1::DeviceCopy => audit.destination_owned_bytes(),
            }
        };
        self.cache_resident_bytes = self
            .cache_resident_bytes
            .checked_add(resident_bytes)
            .ok_or_else(|| {
                QwenExecutionError::InvalidRequest(
                    "prefix resident-byte count overflowed".to_owned(),
                )
            })?;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct QwenPrefixIdentityV1 {
    model_fingerprint: String,
    plan_digest: [u8; 32],
    graph_semantics_digest: [u8; 32],
    adapter_identity: String,
    state_capacity: u64,
    is_mtp: bool,
    is_multimodal: bool,
}

/// Immutable, request-independent Qwen prefix owner.  It owns only quiescent
/// KV and linear/GDN state forks plus identity and terminal metadata; request
/// workspace, queue, prepared plans, completions, and selector output are not
/// retained or shared.
pub struct QwenPrefixStateV1 {
    inner: Arc<QwenPrefixStateInner>,
}

/// One KV layer in a serialized Qwen state image. The descriptor is retained
/// alongside the opaque bytes so a restore cannot silently change encoding,
/// layout, capacity, or layer identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenKvStateImageV1 {
    descriptor: KvStateDescriptor,
    image: ExecutionStateImageV1,
}

impl QwenKvStateImageV1 {
    pub fn descriptor(&self) -> KvStateDescriptor {
        self.descriptor
    }

    pub fn image(&self) -> &ExecutionStateImageV1 {
        &self.image
    }
}

/// One linear/GDN layer in a serialized Qwen state image.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenLinearStateImageV1 {
    descriptor: LinearAttentionStateDescriptor,
    image: ExecutionStateImageV1,
}

impl QwenLinearStateImageV1 {
    pub fn descriptor(&self) -> LinearAttentionStateDescriptor {
        self.descriptor
    }

    pub fn image(&self) -> &ExecutionStateImageV1 {
        &self.image
    }
}

/// Complete, backend-neutral Qwen request state image. Native state handles
/// are never persisted; each layer carries only its validated descriptor and
/// exact opaque planes. Terminal output is optional because a checkpoint can
/// restore state for a non-empty suffix without retaining visible output.
#[derive(Clone, PartialEq)]
pub struct QwenStateImageV1 {
    session_id: crate::ExecutionSessionId,
    identity: QwenPrefixIdentityV1,
    committed_length: u64,
    rope_position_delta: i64,
    kv_layers: BTreeMap<u32, QwenKvStateImageV1>,
    linear_layers: BTreeMap<u32, QwenLinearStateImageV1>,
    cached_terminal_output: Option<QwenExecutionOutput>,
}

impl fmt::Debug for QwenStateImageV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QwenStateImageV1")
            .field("session_id", &self.session_id)
            .field("identity", &"<redacted>")
            .field("committed_length", &self.committed_length)
            .field("kv_layer_count", &self.kv_layers.len())
            .field("linear_layer_count", &self.linear_layers.len())
            .field(
                "has_cached_terminal_output",
                &self.cached_terminal_output.is_some(),
            )
            .finish()
    }
}

impl QwenStateImageV1 {
    pub fn session_id(&self) -> crate::ExecutionSessionId {
        self.session_id
    }

    pub fn committed_length(&self) -> u64 {
        self.committed_length
    }

    pub fn rope_position_delta(&self) -> i64 {
        self.rope_position_delta
    }

    pub fn model_fingerprint(&self) -> &str {
        &self.identity.model_fingerprint
    }

    pub fn plan_digest(&self) -> &[u8; 32] {
        &self.identity.plan_digest
    }

    pub fn graph_semantics_digest(&self) -> &[u8; 32] {
        &self.identity.graph_semantics_digest
    }

    pub fn adapter_identity(&self) -> &str {
        &self.identity.adapter_identity
    }

    pub fn state_capacity(&self) -> u64 {
        self.identity.state_capacity
    }

    pub fn kv_layers(&self) -> &BTreeMap<u32, QwenKvStateImageV1> {
        &self.kv_layers
    }

    pub fn linear_layers(&self) -> &BTreeMap<u32, QwenLinearStateImageV1> {
        &self.linear_layers
    }

    pub fn cached_terminal_output(&self) -> Option<&QwenExecutionOutput> {
        self.cached_terminal_output.as_ref()
    }

    /// Returns a copy suitable for checkpoint restore when terminal output is
    /// intentionally not persisted. Non-empty suffix continuation remains
    /// valid, while an empty suffix fails closed without a cached output.
    pub fn without_terminal_output(mut self) -> Self {
        self.cached_terminal_output = None;
        self
    }

    /// Returns the canonical digest of the KV descriptors carried by this
    /// image.  Checkpoint callers should copy this value into
    /// [`CheckpointIdentity::kv_descriptor_digest`]; restore recomputes it
    /// from the fresh graph before importing any opaque bytes.
    pub fn kv_descriptor_digest(&self) -> [u8; 32] {
        qwen_kv_descriptor_digest(
            self.kv_layers
                .iter()
                .map(|(layer, image)| (*layer, image.descriptor)),
        )
    }

    /// Flattens a quiescent Qwen image into the backend-neutral checkpoint
    /// envelope.  Native handles and terminal output are intentionally not
    /// retained.  `absolute_position - logical_position` is the checked RoPE
    /// position delta represented by the image.
    #[allow(clippy::too_many_arguments)]
    pub fn to_checkpoint(
        &self,
        identity: CheckpointIdentity,
        token_history: &[u32],
        conversation: &[u8],
        sampler_state: &[u8],
        grammar_state: &[u8],
        stop_state: &[u8],
        absolute_position: u64,
        logical_position: u64,
        generation_state_version: u32,
    ) -> Result<SessionCheckpoint, QwenExecutionError> {
        if token_history.len() as u64 != self.committed_length {
            return Err(QwenExecutionError::InvalidRequest(
                "checkpoint token history length differs from Qwen state length".to_owned(),
            ));
        }
        if logical_position != self.committed_length {
            return Err(QwenExecutionError::InvalidRequest(
                "checkpoint logical position differs from Qwen state length".to_owned(),
            ));
        }
        let rope_delta = absolute_position
            .checked_sub(logical_position)
            .ok_or_else(|| {
                QwenExecutionError::InvalidRequest(
                    "checkpoint absolute position precedes logical position".to_owned(),
                )
            })?;
        let rope_delta = i64::try_from(rope_delta).map_err(|_| {
            QwenExecutionError::InvalidRequest(
                "checkpoint RoPE position delta exceeds i64".to_owned(),
            )
        })?;
        if rope_delta != self.rope_position_delta {
            return Err(QwenExecutionError::InvalidRequest(
                "checkpoint RoPE position delta differs from Qwen state".to_owned(),
            ));
        }
        if identity.model_lock_fingerprint != self.identity.model_fingerprint
            || identity.plan_digest != qwen_hex_digest(&self.identity.plan_digest)
            || identity.adapter_identity != self.identity.adapter_identity
            || identity.kv_descriptor_digest != self.kv_descriptor_digest()
            || self
                .kv_layers
                .values()
                .any(|layer| layer.descriptor.cache_encoding() != identity.kv_encoding)
        {
            return Err(QwenExecutionError::InvalidRequest(
                "checkpoint identity does not match Qwen model, plan, or KV encoding/descriptors"
                    .to_owned(),
            ));
        }

        let mut state_layers = Vec::with_capacity(self.kv_layers.len() + self.linear_layers.len());
        let mut state_planes = Vec::new();
        for image in self.kv_layers.values() {
            let metadata = image.image.metadata().clone();
            if metadata.published_length != logical_position {
                return Err(QwenExecutionError::InvalidRequest(
                    "checkpoint KV published length differs from logical position".to_owned(),
                ));
            }
            state_layers.push(metadata);
            state_planes.extend(image.image.planes().iter().cloned());
        }
        for image in self.linear_layers.values() {
            let metadata = image.image.metadata().clone();
            if metadata.published_length != logical_position {
                return Err(QwenExecutionError::InvalidRequest(
                    "checkpoint linear published length differs from logical position".to_owned(),
                ));
            }
            state_layers.push(metadata);
            state_planes.extend(image.image.planes().iter().cloned());
        }
        let payload = CheckpointPayload {
            token_history: token_history.to_vec(),
            conversation: conversation.to_vec(),
            state_layers,
            state_planes,
            sampler_state: sampler_state.to_vec(),
            grammar_state: grammar_state.to_vec(),
            stop_state: stop_state.to_vec(),
        };
        SessionCheckpoint::new(
            identity,
            absolute_position,
            logical_position,
            generation_state_version,
            payload,
        )
        .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))
    }

    /// Alias retained for checkpoint-oriented callers.
    #[allow(clippy::too_many_arguments)]
    pub fn checkpoint(
        &self,
        identity: CheckpointIdentity,
        token_history: &[u32],
        conversation: &[u8],
        sampler_state: &[u8],
        grammar_state: &[u8],
        stop_state: &[u8],
        absolute_position: u64,
        logical_position: u64,
        generation_state_version: u32,
    ) -> Result<SessionCheckpoint, QwenExecutionError> {
        self.to_checkpoint(
            identity,
            token_history,
            conversation,
            sampler_state,
            grammar_state,
            stop_state,
            absolute_position,
            logical_position,
            generation_state_version,
        )
    }
}

struct QwenPrefixStateInner {
    session: Arc<ExecutionSession>,
    identity: QwenPrefixIdentityV1,
    committed_length: u64,
    rope_position_delta: i64,
    kv_states: BTreeMap<u32, KvState>,
    linear_states: BTreeMap<u32, LinearAttentionState>,
    cached_terminal_output: QwenExecutionOutput,
    fork_audit: QwenPrefixForkAuditV1,
}

impl fmt::Debug for QwenPrefixStateV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QwenPrefixStateV1")
            .field("session", &self.inner.session.id())
            .field("committed_length", &self.inner.committed_length)
            .field("state_capacity", &self.inner.identity.state_capacity)
            .field("fork_audit", &self.inner.fork_audit)
            .finish_non_exhaustive()
    }
}

impl Clone for QwenPrefixStateV1 {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl QwenPrefixStateV1 {
    pub fn committed_length(&self) -> u64 {
        self.inner.committed_length
    }

    pub fn rope_position_delta(&self) -> i64 {
        self.inner.rope_position_delta
    }

    pub fn state_capacity(&self) -> u64 {
        self.inner.identity.state_capacity
    }

    pub fn fork_audit(&self) -> QwenPrefixForkAuditV1 {
        self.inner.fork_audit
    }

    pub fn model_fingerprint(&self) -> &str {
        &self.inner.identity.model_fingerprint
    }

    pub fn plan_digest(&self) -> &[u8; 32] {
        &self.inner.identity.plan_digest
    }

    pub fn graph_semantics_digest(&self) -> &[u8; 32] {
        &self.inner.identity.graph_semantics_digest
    }

    pub fn adapter_identity(&self) -> &str {
        &self.inner.identity.adapter_identity
    }

    pub fn cached_terminal_output(&self) -> &QwenExecutionOutput {
        &self.inner.cached_terminal_output
    }
}

pub const QWEN_PREFILL_SMALL_DEVICE_CHUNK_TOKENS: u64 = 512;
pub const QWEN_PREFILL_SMALL_DEVICE_MAX_BYTES: u64 = 16 * 1024 * 1024 * 1024;
pub const QWEN_PREFILL_CHUNK_BUCKETS: [u64; 4] = [16_384, 8_192, 4_096, 2_048];

/// Deterministic placement estimate for one candidate Qwen graph capacity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenGraphMemoryEstimate {
    model_resident_bytes: u64,
    workspace_baseline_bytes: u64,
    workspace_arena_bytes: u64,
    request_state_bytes: u64,
    safety_reserve_bytes: u64,
    required_bytes: u64,
}

impl QwenGraphMemoryEstimate {
    pub const fn model_resident_bytes(self) -> u64 {
        self.model_resident_bytes
    }

    pub const fn workspace_baseline_bytes(self) -> u64 {
        self.workspace_baseline_bytes
    }

    pub const fn workspace_arena_bytes(self) -> u64 {
        self.workspace_arena_bytes
    }

    pub const fn request_state_bytes(self) -> u64 {
        self.request_state_bytes
    }

    pub const fn safety_reserve_bytes(self) -> u64 {
        self.safety_reserve_bytes
    }

    pub const fn required_bytes(self) -> u64 {
        self.required_bytes
    }
}

/// Returns the stable, descending candidate capacities for one prompt. Short
/// prompts use their actual row count; a 512-row floor is evaluated before a
/// fail-closed rejection on devices larger than 16 GiB.
pub fn qwen_prefill_chunk_candidates(
    total_memory_bytes: u64,
    prompt_tokens: u64,
) -> Result<Vec<u64>, QwenExecutionError> {
    if total_memory_bytes == 0 || prompt_tokens == 0 {
        return Err(QwenExecutionError::InvalidRequest(
            "prefill chunk selection requires non-zero device memory and prompt tokens".to_owned(),
        ));
    }
    let buckets: &[u64] = if total_memory_bytes <= QWEN_PREFILL_SMALL_DEVICE_MAX_BYTES {
        &[QWEN_PREFILL_SMALL_DEVICE_CHUNK_TOKENS]
    } else {
        &QWEN_PREFILL_CHUNK_BUCKETS
    };
    let mut candidates = Vec::with_capacity(buckets.len() + 1);
    for bucket in buckets {
        let rows = prompt_tokens.min(*bucket);
        if candidates.last().copied() != Some(rows) {
            candidates.push(rows);
        }
    }
    if total_memory_bytes > QWEN_PREFILL_SMALL_DEVICE_MAX_BYTES {
        let floor = prompt_tokens.min(QWEN_PREFILL_SMALL_DEVICE_CHUNK_TOKENS);
        if candidates.last().copied() != Some(floor) {
            candidates.push(floor);
        }
    }
    Ok(candidates)
}

struct AttentionPreprocessExecution {
    layer: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    token_count: u64,
    start_position: u64,
    position_mode: AttentionPreprocessPositionMode,
}

#[derive(Clone, Copy)]
struct StatefulExecution {
    token_count: u64,
    start_position: u64,
    expected_length: u64,
}

/// Immutable, request-local dispatch audit published after a successful
/// transition.  The counters are accumulated from accepted backend evidence;
/// they are not estimates of graph size.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenExecutionAudit {
    selected_backend: &'static str,
    target: String,
    submission_count: u64,
    kernel_dispatch_count: u64,
    fallback_used: bool,
    all_dispatches_hip: bool,
    segment_count: u64,
    boundary_count: u64,
    sparse_moe_submission_count: u64,
    sparse_moe_active_pair_count: u64,
}

impl QwenExecutionAudit {
    pub const fn selected_backend(&self) -> &'static str {
        self.selected_backend
    }

    pub fn target(&self) -> &str {
        &self.target
    }

    pub const fn submission_count(&self) -> u64 {
        self.submission_count
    }

    pub const fn kernel_dispatch_count(&self) -> u64 {
        self.kernel_dispatch_count
    }

    pub const fn fallback_used(&self) -> bool {
        self.fallback_used
    }

    pub const fn all_dispatches_hip(&self) -> bool {
        self.all_dispatches_hip
    }

    pub const fn segment_count(&self) -> u64 {
        self.segment_count
    }

    pub const fn boundary_count(&self) -> u64 {
        self.boundary_count
    }

    pub const fn sparse_moe_submission_count(&self) -> u64 {
        self.sparse_moe_submission_count
    }

    pub const fn sparse_moe_active_pair_count(&self) -> u64 {
        self.sparse_moe_active_pair_count
    }
}

/// One full-attention layer's logical and physical KV state at an exact
/// request boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QwenKvLayerMemoryAudit {
    layer: u32,
    logical_capacity_tokens: u64,
    observed_length_tokens: u64,
    physical: KvPhysicalMemorySnapshot,
}

impl QwenKvLayerMemoryAudit {
    pub const fn layer(self) -> u32 {
        self.layer
    }

    pub const fn logical_capacity_tokens(self) -> u64 {
        self.logical_capacity_tokens
    }

    pub const fn observed_length_tokens(self) -> u64 {
        self.observed_length_tokens
    }

    pub const fn physical(self) -> KvPhysicalMemorySnapshot {
        self.physical
    }
}

/// Request-local state inventory used by service and GPU evidence runners.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QwenRequestMemoryAudit {
    kv_layers: Vec<QwenKvLayerMemoryAudit>,
    linear_attention_layers: usize,
    linear_attention_capacity_tokens: Option<u64>,
    linear_attention_observed_length_tokens: Option<u64>,
}

impl QwenRequestMemoryAudit {
    pub fn kv_layers(&self) -> &[QwenKvLayerMemoryAudit] {
        &self.kv_layers
    }

    pub const fn linear_attention_layers(&self) -> usize {
        self.linear_attention_layers
    }

    pub const fn linear_attention_capacity_tokens(&self) -> Option<u64> {
        self.linear_attention_capacity_tokens
    }

    pub const fn linear_attention_observed_length_tokens(&self) -> Option<u64> {
        self.linear_attention_observed_length_tokens
    }

    pub fn committed_kv_bytes(&self) -> Result<u64, QwenExecutionError> {
        self.kv_layers.iter().try_fold(0_u64, |total, layer| {
            layer
                .physical
                .committed_bytes_per_plane()
                .checked_mul(2)
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or_else(|| {
                    QwenExecutionError::InvalidRequest(
                        "request KV committed-byte audit overflowed u64".to_string(),
                    )
                })
        })
    }
}

impl QwenExecutionOutput {
    pub fn token_ids(&self) -> &[i32] {
        &self.token_ids
    }

    /// Last-token, full-vocabulary logits when explicitly requested.
    ///
    /// The default greedy path leaves this absent and retains the device
    /// Argmax behavior used by Phase 3. Sampling requests read back exactly
    /// one BF16 row and convert it to finite-width F32 host values.
    pub fn last_logits(&self) -> Option<&[f32]> {
        self.last_logits.as_deref()
    }

    /// Selection metadata returned by the device token-selector route. The
    /// route reads back only the fixed 16-byte selected-record; `None` denotes
    /// the legacy Argmax or host-logits paths.
    pub fn selection(&self) -> Option<&SamplingSelectionV1> {
        self.selection.as_ref()
    }

    /// Explicitly named alias for transport adapters that use the sampling
    /// terminology in their response contract.
    pub fn sampling_selection(&self) -> Option<&SamplingSelectionV1> {
        self.selection()
    }

    /// All target-logit rows in row-major BF16, published only by the
    /// explicit Phase 18 exactness hooks.
    pub fn logits_bf16(&self) -> Option<&[u16]> {
        self.logits_bf16.as_deref()
    }

    /// Target hidden rows before the final output RMSNorm, in row-major BF16.
    /// This is published only by the explicit MTP hook methods.
    pub fn hidden_states_bf16(&self) -> Option<&[u16]> {
        self.hidden_states_bf16.as_deref()
    }

    /// Final-normalized hidden rows in row-major BF16, published only by the
    /// explicit embedding prefill route.  These rows are the input to the LM
    /// head and are distinct from the pre-final MTP hidden rows.
    pub fn embeddings_bf16(&self) -> Option<&[u16]> {
        self.embeddings_bf16.as_deref()
    }

    /// Descriptive alias for callers that name the representation by its
    /// graph boundary rather than by the embedding endpoint.
    pub fn final_hidden_states_bf16(&self) -> Option<&[u16]> {
        self.embeddings_bf16()
    }

    pub const fn committed_length(&self) -> u64 {
        self.committed_length
    }
}

/// Fail-closed errors for one Qwen request owner.
#[derive(Debug)]
pub enum QwenExecutionError {
    InvalidGraph(String),
    InvalidRequest(String),
    Poisoned,
    Busy,
    CompletionPending {
        stage: String,
    },
    CompletionFailure {
        stage: String,
    },
    StateLength {
        layer: u32,
        state: &'static str,
        expected: u64,
        actual: u64,
    },
    ArgmaxSentinel {
        index: usize,
    },
    NodeExecution {
        node: String,
        error: Box<QwenExecutionError>,
    },
    Execution(ExecutionError),
    WeightUpload(WeightUploadError),
    Tensor(TensorError),
    Operation(OpError),
}

impl fmt::Display for QwenExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGraph(reason) => {
                write!(formatter, "invalid Qwen execution graph: {reason}")
            }
            Self::InvalidRequest(reason) => {
                write!(formatter, "invalid Qwen execution request: {reason}")
            }
            Self::Poisoned => {
                formatter.write_str("Qwen request is poisoned after a partial transition")
            }
            Self::Busy => formatter.write_str("Qwen request already has a transition in flight"),
            Self::CompletionPending { stage } => {
                write!(
                    formatter,
                    "{stage} remained pending after the completion wait"
                )
            }
            Self::CompletionFailure { stage } => write!(formatter, "{stage} reported failure"),
            Self::StateLength {
                layer,
                state,
                expected,
                actual,
            } => write!(
                formatter,
                "layer {layer} {state} state length is {actual}, expected {expected}"
            ),
            Self::ArgmaxSentinel { index } => {
                write!(
                    formatter,
                    "argmax returned the NaN sentinel at output index {index}"
                )
            }
            Self::NodeExecution { node, error } => {
                write!(formatter, "Qwen node {node} failed: {error}")
            }
            Self::Execution(error) => error.fmt(formatter),
            Self::WeightUpload(error) => error.fmt(formatter),
            Self::Tensor(error) => error.fmt(formatter),
            Self::Operation(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for QwenExecutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Execution(error) => Some(error),
            Self::NodeExecution { error, .. } => Some(error.as_ref()),
            Self::WeightUpload(error) => Some(error),
            Self::Tensor(error) => Some(error),
            Self::Operation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ExecutionError> for QwenExecutionError {
    fn from(error: ExecutionError) -> Self {
        Self::Execution(error)
    }
}

impl From<PreparedExecutionError> for QwenExecutionError {
    fn from(error: PreparedExecutionError) -> Self {
        match error {
            PreparedExecutionError::Poisoned => Self::Poisoned,
            PreparedExecutionError::Busy => Self::Busy,
            PreparedExecutionError::CompletionPending { stage } => {
                Self::CompletionPending { stage }
            }
            PreparedExecutionError::CompletionFailure { stage } => {
                Self::CompletionFailure { stage }
            }
            PreparedExecutionError::Execution(error) => Self::Execution(error),
            PreparedExecutionError::InvalidPlan(reason)
            | PreparedExecutionError::InvalidTransition(reason)
            | PreparedExecutionError::InvalidAudit(reason) => Self::InvalidRequest(reason),
        }
    }
}

impl From<WeightUploadError> for QwenExecutionError {
    fn from(error: WeightUploadError) -> Self {
        Self::WeightUpload(error)
    }
}

impl From<TensorError> for QwenExecutionError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

impl From<OpError> for QwenExecutionError {
    fn from(error: OpError) -> Self {
        Self::Operation(error)
    }
}

/// A validated model whose required weights are uploaded once for one HIP
/// execution session. It owns only model-resident buffers and the shared queue;
/// every call to [`Self::new_request`] creates fresh graph-local workspace and
/// fresh KV/linear-attention state.
#[derive(Clone)]
pub struct QwenResidentModel {
    inner: Arc<QwenResidentInner>,
}

impl fmt::Debug for QwenResidentModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QwenResidentModel")
            .field("session", &self.inner.session.id())
            .field("model_fingerprint", &self.inner.model_fingerprint)
            .field("plan_digest", &self.inner.plan.digest_hex())
            .finish_non_exhaustive()
    }
}

impl QwenResidentModel {
    /// Validates and uploads all required model weights and derived attention
    /// scale tensors exactly once for `session`.
    pub fn new(
        session: Arc<ExecutionSession>,
        graph: QwenGraph,
        plan: WeightLoadPlan,
        cache: Arc<VerifiedCache>,
        completion_timeout: Duration,
    ) -> Result<Self, QwenExecutionError> {
        if completion_timeout.is_zero() {
            return Err(QwenExecutionError::InvalidRequest(
                "completion timeout must be non-zero".to_owned(),
            ));
        }
        if cache.lock_fingerprint != plan.lock_fingerprint {
            return Err(QwenExecutionError::InvalidRequest(
                "verified cache fingerprint differs from the weight plan".to_owned(),
            ));
        }
        let source = VerifiedProvisionSource {
            cache: Arc::clone(&cache),
        };
        let inner =
            QwenResidentInner::provision(session, graph, plan, completion_timeout, &source)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    pub fn new_gguf(
        session: Arc<ExecutionSession>,
        graph: QwenGraph,
        plan: WeightLoadPlan,
        source: Arc<VerifiedGgufWeightSource>,
        completion_timeout: Duration,
    ) -> Result<Self, QwenExecutionError> {
        if completion_timeout.is_zero() {
            return Err(QwenExecutionError::InvalidRequest(
                "completion timeout must be non-zero".to_owned(),
            ));
        }
        if source.lock_fingerprint() != plan.lock_fingerprint {
            return Err(QwenExecutionError::InvalidRequest(
                "verified GGUF identity differs from the weight plan".to_owned(),
            ));
        }
        let provision = GgufProvisionSource { source };
        let inner =
            QwenResidentInner::provision(session, graph, plan, completion_timeout, &provision)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Provisions the exact reviewed Qwen3.5-35B-A3B MXFP4 artifact. Each
    /// layer's expert planes are packed once into the native immutable blob;
    /// routing and activation quantization remain request-time GPU work.
    pub fn new_moe(
        session: Arc<ExecutionSession>,
        graph: QwenGraph,
        plan: WeightLoadPlan,
        artifact: Arc<VerifiedQwen35Moe>,
        completion_timeout: Duration,
    ) -> Result<Self, QwenExecutionError> {
        if completion_timeout.is_zero() {
            return Err(QwenExecutionError::InvalidRequest(
                "completion timeout must be non-zero".to_owned(),
            ));
        }
        if plan.lock_fingerprint != crate::QWEN35_MOE_MODEL_FINGERPRINT
            || graph.model_fingerprint() != crate::QWEN35_MOE_MODEL_FINGERPRINT
        {
            return Err(QwenExecutionError::InvalidRequest(
                "MoE artifact, graph, and load-plan identities differ".to_owned(),
            ));
        }
        let source = Qwen35MoeProvisionSource { artifact };
        let inner =
            QwenResidentInner::provision(session, graph, plan, completion_timeout, &source)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    pub fn new_gguf_moe(
        session: Arc<ExecutionSession>,
        graph: QwenGraph,
        plan: WeightLoadPlan,
        source: Arc<VerifiedGgufQwen35Moe>,
        completion_timeout: Duration,
    ) -> Result<Self, QwenExecutionError> {
        if completion_timeout.is_zero()
            || plan.lock_fingerprint != crate::QWEN35_MOE_MODEL_FINGERPRINT
            || graph.model_fingerprint() != crate::QWEN35_MOE_MODEL_FINGERPRINT
        {
            return Err(QwenExecutionError::InvalidRequest(
                "MoE GGUF, graph, and load-plan identities differ".to_owned(),
            ));
        }
        let provision = GgufMoeProvisionSource { source };
        let inner =
            QwenResidentInner::provision(session, graph, plan, completion_timeout, &provision)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Provision a graph built by `build_qwen35_fp8_graph`. Non-linear
    /// tensors continue to come from the verified BF16 cache; text-linear
    /// value/scale pairs come from the independently verified sidecar.
    pub fn new_fp8(
        session: Arc<ExecutionSession>,
        graph: QwenGraph,
        plan: WeightLoadPlan,
        cache: Arc<VerifiedCache>,
        sidecar: Arc<VerifiedFp8Sidecar>,
        completion_timeout: Duration,
    ) -> Result<Self, QwenExecutionError> {
        if graph.fp8_sidecar_fingerprint() != Some(sidecar.manifest_fingerprint()) {
            return Err(QwenExecutionError::InvalidRequest(
                "FP8 graph and sidecar identities differ".to_owned(),
            ));
        }
        if cache.lock_fingerprint != plan.lock_fingerprint
            || sidecar.source_lock_fingerprint() != plan.lock_fingerprint
        {
            return Err(QwenExecutionError::InvalidRequest(
                "FP8 source cache, sidecar, and plan identities differ".to_owned(),
            ));
        }
        let source = Fp8ProvisionSource { cache, sidecar };
        let inner =
            QwenResidentInner::provision(session, graph, plan, completion_timeout, &source)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Provision packed weight-only NVFP4 residency. Text-linear values and
    /// their block/tensor scales come from one independently verified sidecar;
    /// all other tensors remain verified BF16 source weights.
    pub fn new_nvfp4(
        session: Arc<ExecutionSession>,
        graph: QwenGraph,
        plan: WeightLoadPlan,
        cache: Arc<VerifiedCache>,
        sidecar: Arc<VerifiedNvfp4Sidecar>,
        completion_timeout: Duration,
    ) -> Result<Self, QwenExecutionError> {
        if graph.fp8_sidecar_fingerprint() != Some(sidecar.manifest_fingerprint()) {
            return Err(QwenExecutionError::InvalidRequest(
                "NVFP4 graph and sidecar identities differ".to_owned(),
            ));
        }
        if cache.lock_fingerprint != plan.lock_fingerprint
            || sidecar.source_lock_fingerprint() != plan.lock_fingerprint
        {
            return Err(QwenExecutionError::InvalidRequest(
                "NVFP4 source cache, sidecar, and plan identities differ".to_owned(),
            ));
        }
        let source = Nvfp4ProvisionSource { cache, sidecar };
        let inner =
            QwenResidentInner::provision(session, graph, plan, completion_timeout, &source)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// CDNA3 provider: numerically converts the verified OCP value bytes to
    /// E4M3FNUZ once while provisioning model-resident storage. Scales remain
    /// FP32 and the source sidecar identity remains part of the model identity.
    pub fn new_fp8_fnuz(
        session: Arc<ExecutionSession>,
        graph: QwenGraph,
        plan: WeightLoadPlan,
        cache: Arc<VerifiedCache>,
        sidecar: Arc<VerifiedFp8Sidecar>,
        completion_timeout: Duration,
    ) -> Result<Self, QwenExecutionError> {
        if graph.fp8_sidecar_fingerprint() != Some(sidecar.manifest_fingerprint()) {
            return Err(QwenExecutionError::InvalidRequest(
                "FNUZ graph and OCP sidecar identities differ".to_owned(),
            ));
        }
        if cache.lock_fingerprint != plan.lock_fingerprint
            || sidecar.source_lock_fingerprint() != plan.lock_fingerprint
        {
            return Err(QwenExecutionError::InvalidRequest(
                "FNUZ source cache, sidecar, and plan identities differ".to_owned(),
            ));
        }
        let source = Fp8FnuzProvisionSource { cache, sidecar };
        let inner =
            QwenResidentInner::provision(session, graph, plan, completion_timeout, &source)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Explicit RDNA2 compatibility provider. It verifies the FP8 sidecar,
    /// converts each text-linear value/scale pair to BF16 once during model
    /// load, and then executes the existing BF16 graph. This is never labeled
    /// native FP8 and is never selected after an execution failure.
    pub fn new_fp8_converted_bf16(
        session: Arc<ExecutionSession>,
        graph: QwenGraph,
        plan: WeightLoadPlan,
        cache: Arc<VerifiedCache>,
        sidecar: Arc<VerifiedFp8Sidecar>,
        completion_timeout: Duration,
    ) -> Result<Self, QwenExecutionError> {
        if graph.fp8_sidecar_fingerprint().is_some() {
            return Err(QwenExecutionError::InvalidRequest(
                "converted BF16 provider requires an unquantized graph".to_owned(),
            ));
        }
        if cache.lock_fingerprint != plan.lock_fingerprint
            || sidecar.source_lock_fingerprint() != plan.lock_fingerprint
        {
            return Err(QwenExecutionError::InvalidRequest(
                "converted BF16 source cache, sidecar, and plan identities differ".to_owned(),
            ));
        }
        let source = Fp8ConvertedProvisionSource { cache, sidecar };
        let inner =
            QwenResidentInner::provision(session, graph, plan, completion_timeout, &source)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Creates a fresh request graph/state owner against this resident model.
    pub fn new_request(
        &self,
        graph: QwenGraph,
    ) -> Result<QwenExecutionRequest, QwenExecutionError> {
        self.new_request_with_adapters(graph, AdapterRequestSetV1::disabled())
    }

    /// Creates a request-local owner with verified dense-BF16 adapter effects.
    /// Adapter payloads are uploaded into request-state buffers; the resident
    /// model, tokenizer, and graph metadata remain shared and immutable.
    pub fn new_request_with_adapters(
        &self,
        graph: QwenGraph,
        adapters: AdapterRequestSetV1,
    ) -> Result<QwenExecutionRequest, QwenExecutionError> {
        QwenExecutionCore::from_resident(Arc::clone(&self.inner), graph, adapters).map(|core| {
            QwenExecutionRequest {
                _resident: Arc::clone(&self.inner),
                core,
            }
        })
    }

    /// Builds a fresh explicit-position Qwen request from retained prefix and
    /// recent token ranges. The source history is only read; state publication
    /// occurs on the fresh request after the retained prefill succeeds.
    pub fn new_request_from_context_shift(
        &self,
        graph: QwenGraph,
        decision: ContextShiftDecisionV1,
        state: ContextWindowStateV1,
        token_history: &[i32],
    ) -> Result<(QwenExecutionRequest, QwenExecutionOutput), QwenExecutionError> {
        if !decision.requires_shift() || decision.old_state() != state {
            return Err(QwenExecutionError::InvalidRequest(
                "Qwen context shift decision is stale or does not require a shift".to_owned(),
            ));
        }
        let retained = decision
            .retained_token_ids(token_history)
            .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
        let positions = decision
            .retained_absolute_positions(state.logical_length())
            .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
        let retained_len = u64::try_from(retained.len()).map_err(|_| {
            QwenExecutionError::InvalidRequest("retained length overflowed".to_owned())
        })?;
        if retained_len == 0 || retained_len != decision.proposed_state().logical_length() {
            return Err(QwenExecutionError::InvalidRequest(
                "Qwen retained state length differs from the shift decision".to_owned(),
            ));
        }
        if graph.position_payload_mode()
            != crate::AttentionPreprocessPositionPayloadModeV1::Explicit
            || graph.token_count() < retained_len
        {
            return Err(QwenExecutionError::InvalidRequest(
                "Qwen context shift requires an explicit-position graph with sufficient token capacity"
                    .to_owned(),
            ));
        }
        let mut request = self.new_request(graph)?;
        let output = request.prefill_with_absolute_positions(&retained, &positions)?;
        let delta = state
            .absolute_position()
            .checked_sub(retained_len)
            .and_then(|value| i64::try_from(value).ok())
            .ok_or_else(|| {
                QwenExecutionError::InvalidRequest("Qwen RoPE delta overflowed".to_owned())
            })?;
        request.set_rope_position_delta(delta)?;
        Ok((request, output))
    }

    /// Creates a fresh request workspace and transactionally forks every
    /// published KV and linear/GDN state from an immutable prefix owner.  The
    /// prefix owner is never reused as a mutable request state.
    pub fn new_request_from_prefix(
        &self,
        prefix: &QwenPrefixStateV1,
        graph: QwenGraph,
    ) -> Result<QwenExecutionRequest, QwenExecutionError> {
        self.new_request_from_prefix_with_adapters(prefix, graph, AdapterRequestSetV1::disabled())
    }

    pub fn new_request_from_prefix_with_adapters(
        &self,
        prefix: &QwenPrefixStateV1,
        graph: QwenGraph,
        adapters: AdapterRequestSetV1,
    ) -> Result<QwenExecutionRequest, QwenExecutionError> {
        let mut request = self.new_request_with_adapters(graph, adapters)?;
        request.core.install_prefix(prefix)?;
        Ok(request)
    }

    /// Creates a fresh request and transactionally imports every state layer
    /// from a backend-neutral Qwen image. The source image is never mutated;
    /// any partial destination import is dropped with the fresh request.
    pub fn new_request_from_state_image(
        &self,
        image: &QwenStateImageV1,
        graph: QwenGraph,
    ) -> Result<QwenExecutionRequest, QwenExecutionError> {
        let mut request = self.new_request(graph)?;
        request.core.restore_state_image(image)?;
        Ok(request)
    }

    /// Restores a backend-neutral checkpoint into a fresh request.  Unlike a
    /// raw [`QwenStateImageV1`], the checkpoint carries no source session ID;
    /// therefore this is the only state-image path that may cross process or
    /// execution-session boundaries.  The caller supplies the complete
    /// frontend identity so renderer/tokenizer/sampler policy cannot be
    /// silently changed during restore.
    pub fn new_request_from_checkpoint(
        &self,
        checkpoint: &SessionCheckpoint,
        graph: QwenGraph,
        expected_identity: &CheckpointIdentity,
    ) -> Result<QwenExecutionRequest, QwenExecutionError> {
        self.new_request_from_checkpoint_with_adapters(
            checkpoint,
            graph,
            expected_identity,
            AdapterRequestSetV1::disabled(),
        )
    }

    pub fn new_request_from_checkpoint_with_adapters(
        &self,
        checkpoint: &SessionCheckpoint,
        graph: QwenGraph,
        expected_identity: &CheckpointIdentity,
        adapters: AdapterRequestSetV1,
    ) -> Result<QwenExecutionRequest, QwenExecutionError> {
        let mut request = self.new_request_with_adapters(graph, adapters)?;
        request
            .core
            .restore_checkpoint(checkpoint, expected_identity)?;
        Ok(request)
    }

    /// Compatibility spelling for persistent-session factories.
    pub fn restore_request_from_checkpoint(
        &self,
        checkpoint: &SessionCheckpoint,
        graph: QwenGraph,
        expected_identity: &CheckpointIdentity,
    ) -> Result<QwenExecutionRequest, QwenExecutionError> {
        self.new_request_from_checkpoint(checkpoint, graph, expected_identity)
    }

    /// Alias used by checkpoint restore callers.
    pub fn restore_request_from_state_image(
        &self,
        image: &QwenStateImageV1,
        graph: QwenGraph,
    ) -> Result<QwenExecutionRequest, QwenExecutionError> {
        self.new_request_from_state_image(image, graph)
    }

    /// Short alias for service factories that use request-oriented naming.
    pub fn request_from_state_image(
        &self,
        image: &QwenStateImageV1,
        graph: QwenGraph,
    ) -> Result<QwenExecutionRequest, QwenExecutionError> {
        self.new_request_from_state_image(image, graph)
    }

    /// Compatibility spelling for callers that use a factory-oriented name.
    pub fn request_from_prefix(
        &self,
        prefix: &QwenPrefixStateV1,
        graph: QwenGraph,
    ) -> Result<QwenExecutionRequest, QwenExecutionError> {
        self.new_request_from_prefix(prefix, graph)
    }

    /// Argument-order compatibility alias for adapters that pass the graph
    /// before the reusable prefix owner.
    pub fn new_request_with_prefix(
        &self,
        graph: QwenGraph,
        prefix: &QwenPrefixStateV1,
    ) -> Result<QwenExecutionRequest, QwenExecutionError> {
        self.new_request_from_prefix(prefix, graph)
    }

    /// Runs an explicit suffix continuation from an immutable prefix. An
    /// empty suffix returns the cached terminal output; a non-empty suffix is
    /// lowered as chunked `DecodeContinuation` transitions.
    pub fn generate_from_prefix(
        &self,
        prefix: &QwenPrefixStateV1,
        graph: QwenGraph,
        suffix: &[i32],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        let mut request = self.new_request_from_prefix(prefix, graph)?;
        request.decode_continuation(suffix)
    }

    /// Explicit-session variant used by service owners that keep the session
    /// handle separately. It rejects even a same-backend session with a
    /// different identity before graph/state allocation.
    pub fn new_request_for_session(
        &self,
        session: Arc<ExecutionSession>,
        graph: QwenGraph,
    ) -> Result<QwenExecutionRequest, QwenExecutionError> {
        if session.id() != self.inner.session.id() {
            return Err(QwenExecutionError::InvalidRequest(
                "request session differs from the resident model session".to_owned(),
            ));
        }
        self.new_request(graph)
    }

    pub fn new_request_for_session_with_adapters(
        &self,
        session: Arc<ExecutionSession>,
        graph: QwenGraph,
        adapters: AdapterRequestSetV1,
    ) -> Result<QwenExecutionRequest, QwenExecutionError> {
        if session.id() != self.inner.session.id() {
            return Err(QwenExecutionError::InvalidRequest(
                "request session differs from the resident model session".to_owned(),
            ));
        }
        self.new_request_with_adapters(graph, adapters)
    }

    /// Compatibility spelling for callers that prefer factory terminology.
    pub fn create_request(
        &self,
        graph: QwenGraph,
    ) -> Result<QwenExecutionRequest, QwenExecutionError> {
        self.new_request(graph)
    }

    /// Short factory spelling retained for service and benchmark callers.
    pub fn request(&self, graph: QwenGraph) -> Result<QwenExecutionRequest, QwenExecutionError> {
        self.new_request(graph)
    }

    pub fn session_id(&self) -> crate::ExecutionSessionId {
        self.inner.session.id()
    }

    pub fn model_fingerprint(&self) -> &str {
        &self.inner.model_fingerprint
    }

    pub fn plan_digest(&self) -> &[u8; 32] {
        self.inner.plan.digest()
    }

    pub fn memory_snapshot(&self) -> crate::AllocationSnapshot {
        self.inner.session.memory_snapshot()
    }
}

/// One fully provisioned, request-local Qwen execution owner.
///
/// `new` remains source-compatible with the original API. It now builds a
/// short-lived resident owner and then creates one request, while callers that
/// repeat requests should retain [`QwenResidentModel`] and call
/// [`QwenResidentModel::new_request`].
pub struct QwenExecutionRequest {
    _resident: Arc<QwenResidentInner>,
    core: QwenExecutionCore,
}

impl fmt::Debug for QwenExecutionRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QwenExecutionRequest")
            .field("session", &self.core.session.id())
            .field("committed_length", &self.core.committed_length)
            .field("poisoned", &self.is_poisoned())
            .finish_non_exhaustive()
    }
}

impl QwenExecutionRequest {
    /// Builds a request owner without allocating a packed model arena or
    /// providing a numerical fallback. `completion_timeout` bounds every D1
    /// upload, submit, and readback wait and must be non-zero.
    pub fn new(
        session: Arc<ExecutionSession>,
        graph: QwenGraph,
        plan: WeightLoadPlan,
        cache: Arc<VerifiedCache>,
        completion_timeout: Duration,
    ) -> Result<Self, QwenExecutionError> {
        let resident =
            QwenResidentModel::new(session, graph.clone(), plan, cache, completion_timeout)?;
        resident.new_request(graph)
    }

    /// Runs the graph from position zero. A request accepts exactly the D0
    /// graph token count for prefill and cannot prefill a second time.
    pub fn prefill(
        &mut self,
        token_ids: &[i32],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.core.prefill(token_ids)
    }

    /// Prefills an explicit-position graph with compact logical rows and
    /// caller-supplied absolute RoPE positions.
    pub fn prefill_with_absolute_positions(
        &mut self,
        token_ids: &[i32],
        positions: &[u64],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.core
            .prefill_with_absolute_positions(token_ids, positions)
    }

    /// Publishes a quiescent immutable prefix owner. All KV and linear/GDN
    /// layers are forked through the execution-session contract; a failure in
    /// any layer leaves this request unchanged and drops already-created
    /// destination owners.
    pub fn prefix_state(&self) -> Result<QwenPrefixStateV1, QwenExecutionError> {
        self.core.publish_prefix()
    }

    /// Compatibility spelling for the explicit prefix publication API.
    pub fn create_prefix_state(&self) -> Result<QwenPrefixStateV1, QwenExecutionError> {
        self.prefix_state()
    }

    /// Updates the checked absolute-minus-logical RoPE delta used by later
    /// compacted decode transitions. This metadata is changed only on a fresh
    /// successfully prefilled request by the context-shift factory.
    pub fn set_rope_position_delta(&mut self, delta: i64) -> Result<(), QwenExecutionError> {
        self.core.set_rope_position_delta(delta)
    }

    /// Exports every quiescent KV and linear/GDN layer plus model/plan/graph
    /// identity and the current RoPE position delta.
    pub fn state_image(&self) -> Result<QwenStateImageV1, QwenExecutionError> {
        self.core.export_state_image()
    }

    /// Compatibility spelling for checkpoint writers.
    pub fn export_state_image(&self) -> Result<QwenStateImageV1, QwenExecutionError> {
        self.state_image()
    }

    /// Compatibility spelling for persistent-session callers.
    pub fn save_state_image(&self) -> Result<QwenStateImageV1, QwenExecutionError> {
        self.state_image()
    }

    /// Captures this request as a backend-neutral checkpoint. Terminal output
    /// is deliberately omitted, so restore callers must provide a non-empty
    /// suffix before requesting continuation.
    #[allow(clippy::too_many_arguments)]
    pub fn checkpoint(
        &self,
        identity: CheckpointIdentity,
        token_history: &[u32],
        conversation: &[u8],
        sampler_state: &[u8],
        grammar_state: &[u8],
        stop_state: &[u8],
        absolute_position: u64,
        logical_position: u64,
        generation_state_version: u32,
    ) -> Result<SessionCheckpoint, QwenExecutionError> {
        self.state_image()?.to_checkpoint(
            identity,
            token_history,
            conversation,
            sampler_state,
            grammar_state,
            stop_state,
            absolute_position,
            logical_position,
            generation_state_version,
        )
    }

    /// Continues a prefix-derived request using DecodeContinuation chunks.
    /// Empty suffixes are resolved from the immutable cached terminal output.
    pub fn decode_continuation(
        &mut self,
        suffix: &[i32],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.core.decode_continuation(suffix)
    }

    /// Explicit alias emphasizing that this API consumes a prefix suffix.
    pub fn continue_from_prefix(
        &mut self,
        suffix: &[i32],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.decode_continuation(suffix)
    }

    /// Compatibility alias for generation adapters that call the suffix a
    /// prefill even though its position mode is DecodeContinuation.
    pub fn prefill_from_prefix(
        &mut self,
        suffix: &[i32],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.decode_continuation(suffix)
    }

    /// Runs prefill and publishes the final row of full-vocabulary logits in
    /// addition to the existing device Argmax output.
    pub fn prefill_with_last_logits(
        &mut self,
        token_ids: &[i32],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.core.prefill_with_last_logits(token_ids)
    }

    /// Runs text prefill through the additive device token-selector subset.
    /// The selector is ordered after the terminal projection on the same
    /// execution queue and returns only a validated 16-byte selected record.
    /// MTP graphs are intentionally unsupported for this route.
    pub fn prefill_with_device_selector(
        &mut self,
        token_ids: &[i32],
        selector: &DeviceTokenSelectorRequestV1,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.core.prefill_with_device_selector(token_ids, selector)
    }

    pub fn prefill_with_mtp_state(
        &mut self,
        token_ids: &[i32],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.core.prefill_with_mtp_state(token_ids)
    }

    /// Runs the verified Qwen prefill route in explicit embedding mode.  The
    /// final normalized hidden rows are read back in BF16; LM-head/Argmax
    /// output is not read back and no generation token is published.
    pub fn prefill_with_embeddings(
        &mut self,
        token_ids: &[i32],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.core.prefill_with_embeddings(token_ids)
    }

    /// Runs a typed multimodal prefill with complete prompt embeddings and
    /// three-axis `[temporal,height,width]` mRoPE positions.
    pub fn prefill_multimodal_with_last_logits(
        &mut self,
        token_ids: &[i32],
        embeddings_bf16: &[u16],
        positions: &[[i32; 3]],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.core
            .prefill_multimodal(token_ids, embeddings_bf16, positions)
    }

    /// Runs exactly one decode token at the current committed position.
    pub fn decode(&mut self, token_id: i32) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.core.decode(token_id)
    }

    /// Runs one decode transition and publishes its full-vocabulary logits.
    pub fn decode_with_last_logits(
        &mut self,
        token_id: i32,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.core.decode_with_last_logits(token_id)
    }

    /// Runs one decode transition through the additive device token-selector
    /// subset. Unsupported selector/graph combinations fail closed; there is
    /// no implicit host-logits or Argmax fallback.
    pub fn decode_with_device_selector(
        &mut self,
        token_id: i32,
        selector: &DeviceTokenSelectorRequestV1,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.core.decode_with_device_selector(token_id, selector)
    }

    pub fn decode_with_mtp_state(
        &mut self,
        token_id: i32,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.core.decode_with_mtp_state(token_id)
    }

    /// Evidence-only exactness hook: returns the raw BF16 target-logit row in
    /// addition to the MTP hidden row.
    pub fn decode_with_mtp_state_and_logits(
        &mut self,
        token_id: i32,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.core.decode_with_mtp_state_and_logits(token_id)
    }

    /// Verifies one speculative target block in a single causal transition.
    ///
    /// The first input is the pending token already selected by the preceding
    /// target step; subsequent inputs are draft tokens. The returned Argmax
    /// and hidden-state rows preserve input order. Publication/rollback is
    /// deliberately handled by the Phase 18 speculative transaction owner,
    /// not by this numerical primitive.
    pub fn decode_block_with_mtp_state(
        &mut self,
        token_ids: &[i32],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.core.decode_block_with_mtp_state(token_ids)
    }

    /// Evidence-only exactness hook: returns every raw BF16 target-logit row
    /// for the speculative block.
    pub fn decode_block_with_mtp_state_and_logits(
        &mut self,
        token_ids: &[i32],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.core.decode_block_with_mtp_state_and_logits(token_ids)
    }

    /// Resolves the immediately preceding speculative block. Keeping the full
    /// block is metadata-only; a partial prefix restores the pre-block state
    /// and deterministically replays only the committed input rows.
    pub fn resolve_decode_block(
        &mut self,
        committed_input_rows: usize,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.core.resolve_decode_block(committed_input_rows)
    }

    /// Rewinds the immediately preceding single-row decode transition. This
    /// is used by the MTP owner when a later draft is discarded; stale or
    /// repeated rewinds fail closed in the backend.
    pub fn rewind_last_decode_transition(&mut self) -> Result<(), QwenExecutionError> {
        self.core.rewind_last_decode_transition()
    }

    /// Evidence-only semantic KV readback as `(layer, key_bytes, value_bytes)`.
    pub fn kv_payload_bytes_for_evidence(
        &self,
    ) -> Result<Vec<QwenKvPayloadEvidence>, QwenExecutionError> {
        self.core.kv_payload_bytes_for_evidence()
    }

    /// Runs one MTP row. `target_hidden_bf16` must contain exactly 2560 BF16
    /// values and the graph must expose the typed MTP hidden input.
    pub fn prefill_mtp(
        &mut self,
        token_id: i32,
        target_hidden_bf16: &[u16],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.core.prefill_mtp(token_id, target_hidden_bf16)
    }

    pub fn decode_mtp(
        &mut self,
        token_id: i32,
        target_hidden_bf16: &[u16],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.core.decode_mtp(token_id, target_hidden_bf16)
    }

    pub const fn committed_length(&self) -> u64 {
        self.core.committed_length
    }

    pub const fn prefill_chunk_capacity(&self) -> u64 {
        self.core.graph.token_count()
    }

    pub const fn prefill_chunk_count(&self) -> u64 {
        self.core.prefill_chunk_count
    }

    pub fn last_output(&self) -> Option<&QwenExecutionOutput> {
        self.core.last_output.as_ref()
    }

    pub fn is_poisoned(&self) -> bool {
        self.core.lifecycle.is_poisoned()
    }

    /// Idempotently invalidates this request owner without affecting the
    /// resident model. Synchronous callers use it between transitions when a
    /// transport cancellation or host-side sampling/decoding failure occurs.
    pub fn cancel(&mut self) {
        self.core.lifecycle.cancel();
    }

    pub fn model_fingerprint(&self) -> &str {
        self.core.graph.model_fingerprint()
    }

    pub fn plan_digest(&self) -> &[u8; 32] {
        self.core.plan.digest()
    }

    pub fn session_id(&self) -> crate::ExecutionSessionId {
        self.core.session.id()
    }

    pub fn adapter_identity(&self) -> &str {
        &self.core.adapters.identity
    }

    /// Returns the immutable audit accumulated by successful compute
    /// submissions. An empty audit is never a successful request audit.
    pub fn audit_snapshot(&self) -> Result<QwenExecutionAudit, QwenExecutionError> {
        self.core.audit_snapshot()
    }

    /// Reconciles post-COW ownership for every KV destination in a prefix
    /// continuation and returns the aggregate redacted fork audit. A fresh
    /// request has no fork destinations and therefore returns an explicit
    /// unsupported error instead of silently omitting accounting.
    pub fn refresh_prefix_fork_audit(&self) -> Result<QwenPrefixForkAuditV1, QwenExecutionError> {
        self.core.refresh_prefix_fork_audit()
    }

    /// Captures all request-local state at one quiescent boundary. HIP-backed
    /// KV states must include physical virtual-memory metadata; a backend that
    /// omits it fails closed instead of producing inferred evidence.
    pub fn memory_audit_snapshot(&self) -> Result<QwenRequestMemoryAudit, QwenExecutionError> {
        let mut kv_layers = Vec::with_capacity(self.core.kv_states.len());
        for (&layer, state) in &self.core.kv_states {
            let snapshot = state.snapshot(self.core.session.as_ref())?;
            let physical = snapshot.physical_memory().ok_or_else(|| {
                QwenExecutionError::InvalidRequest(format!(
                    "KV layer {layer} did not report physical-memory metadata"
                ))
            })?;
            kv_layers.push(QwenKvLayerMemoryAudit {
                layer,
                logical_capacity_tokens: snapshot.capacity(),
                observed_length_tokens: snapshot.length(),
                physical,
            });
        }
        if let Some(reference) = kv_layers.first().copied() {
            for layer in &kv_layers[1..] {
                if layer.logical_capacity_tokens != reference.logical_capacity_tokens
                    || layer.observed_length_tokens != reference.observed_length_tokens
                    || layer.physical != reference.physical
                {
                    return Err(QwenExecutionError::InvalidRequest(format!(
                        "KV layer {} memory metadata differs from layer {}",
                        layer.layer, reference.layer
                    )));
                }
            }
        }

        let mut linear_capacity = None;
        let mut linear_length = None;
        for (&layer, state) in &self.core.linear_states {
            let snapshot = state.snapshot(self.core.session.as_ref())?;
            match (linear_capacity, linear_length) {
                (None, None) => {
                    linear_capacity = Some(snapshot.descriptor().capacity());
                    linear_length = Some(snapshot.length());
                }
                (Some(capacity), Some(length))
                    if capacity == snapshot.descriptor().capacity()
                        && length == snapshot.length() => {}
                _ => {
                    return Err(QwenExecutionError::InvalidRequest(format!(
                        "linear-attention layer {layer} state differs from its peers"
                    )));
                }
            }
        }

        Ok(QwenRequestMemoryAudit {
            kv_layers,
            linear_attention_layers: self.core.linear_states.len(),
            linear_attention_capacity_tokens: linear_capacity,
            linear_attention_observed_length_tokens: linear_length,
        })
    }
}

// There is intentionally no `Drop` implementation. Destroying a request
// never shuts down its shared session; any active transition guard poisons its
// request before the owner is released.

struct QwenResidentInner {
    session: Arc<ExecutionSession>,
    model_fingerprint: String,
    fp8_sidecar_fingerprint: Option<String>,
    plan: WeightLoadPlan,
    queue: ExecutionQueue,
    static_tensors: BTreeMap<String, TensorAllocation>,
    scales: BTreeMap<String, CachedScale>,
    completion_timeout: Duration,
}

struct QwenExecutionCore {
    session: Arc<ExecutionSession>,
    graph: QwenGraph,
    execution_plan: PreparedExecutionPlan<QwenGraphNode>,
    plan: WeightLoadPlan,
    queue: ExecutionQueue,
    tensors: Vec<TensorAllocation>,
    tensor_ids: BTreeMap<String, usize>,
    dynamic_tensors: Vec<bool>,
    kv_states: BTreeMap<u32, KvState>,
    linear_states: BTreeMap<u32, LinearAttentionState>,
    scales: BTreeMap<usize, CachedScale>,
    completion_timeout: Duration,
    audit: Mutex<ExecutionAuditAccumulator>,
    prepared_semantics: PreparedSemanticCache,
    lifecycle: ExecutionTransaction,
    committed_length: u64,
    prefill_chunk_count: u64,
    rope_position_delta: i64,
    last_output: Option<QwenExecutionOutput>,
    pending_speculative: Option<PendingSpeculativeBlock>,
    adapters: QwenAdapterRuntime,
}

struct QwenAdapterRuntime {
    identity: String,
    lora: Vec<QwenLoraDeviceArtifact>,
    controls: Vec<QwenControlDeviceArtifact>,
}

struct QwenLoraDeviceArtifact {
    selection: LoraAdapterSelectionV1,
    buffer: ExecutionBuffer,
}

struct QwenControlDeviceArtifact {
    selection: ControlVectorSelectionV1,
    buffer: ExecutionBuffer,
}

impl QwenAdapterRuntime {
    fn disabled() -> Self {
        Self {
            identity: AdapterRequestSetV1::disabled().identity().to_owned(),
            lora: Vec::new(),
            controls: Vec::new(),
        }
    }

    fn has_lora_target(&self, tensor_name: &str) -> bool {
        self.lora.iter().any(|artifact| {
            artifact
                .selection
                .artifact
                .targets()
                .iter()
                .any(|target| target.tensor_name() == tensor_name)
        })
    }

    fn controls_for_layer(&self, layer: u32) -> impl Iterator<Item = &QwenControlDeviceArtifact> {
        self.controls.iter().filter(move |artifact| {
            let (start, end) = artifact.selection.artifact.layer_range();
            u64::from(layer) >= start && u64::from(layer) < end
        })
    }
}

struct PendingSpeculativeBlock {
    start_length: u64,
    token_ids: Vec<i32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TerminalOutputRows {
    Last,
    All,
}

struct TerminalSelection {
    token_ids: Vec<i32>,
    selection: Option<SamplingSelectionV1>,
}

const TERMINAL_ROW_MIN_TOKENS: u64 = 255;

#[derive(Clone)]
struct TensorAllocation {
    buffer: ExecutionBuffer,
    graph_view: TensorView,
}

#[derive(Clone)]
struct CachedScale {
    raw_tensor_id: usize,
    raw_bytes: Arc<[u8]>,
    expanded_bytes: Arc<[u8]>,
}

#[derive(Clone, Copy)]
struct ScaleMaterialization {
    raw_tensor_id: usize,
    output_tensor_id: usize,
    heads: u32,
    head_dim: u32,
}

struct GraphLayout {
    tensor_ids: BTreeMap<String, usize>,
    _weight_tensor_ids: BTreeMap<String, usize>,
    dynamic_tensors: Vec<bool>,
    scales: Vec<ScaleMaterialization>,
    workspace: WorkspaceArenaLayout,
}

#[derive(Clone, Debug)]
struct WorkspaceArenaLayout {
    tensor_offsets: Vec<Option<u64>>,
    slot_sizes: BTreeMap<u64, u64>,
    baseline_bytes: u64,
    high_water_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
struct WorkspaceInterval {
    tensor_id: usize,
    first_node: usize,
    last_node: usize,
    size_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
struct WorkspaceSlot {
    offset_bytes: u64,
    size_bytes: u64,
    last_node: usize,
}

type StateMaps = (BTreeMap<u32, KvState>, BTreeMap<u32, LinearAttentionState>);

fn qwen_prepared_execution_plan(
    graph: &QwenGraph,
) -> Result<PreparedExecutionPlan<QwenGraphNode>, QwenExecutionError> {
    let nodes = graph
        .nodes()
        .iter()
        .cloned()
        .map(|node| {
            let boundary_after = match node.kind() {
                QwenGraphNodeKind::FullKvAppend { .. } => {
                    Some(ExecutionBoundaryKind::StatePublication)
                }
                QwenGraphNodeKind::Semantic(SemanticOpKind::Argmax) => {
                    Some(ExecutionBoundaryKind::TerminalReadback)
                }
                _ => None,
            };
            PreparedPlanNode::new(node, boundary_after)
        })
        .collect();
    PreparedExecutionPlan::new(nodes).map_err(Into::into)
}

impl QwenAdapterRuntime {
    fn provision(
        session: &ExecutionSession,
        queue: &ExecutionQueue,
        graph: &QwenGraph,
        plan: &WeightLoadPlan,
        adapters: AdapterRequestSetV1,
        completion_timeout: Duration,
    ) -> Result<Self, QwenExecutionError> {
        let identity = adapters.identity().to_owned();
        if adapters.adapters().is_empty() && adapters.controls().is_empty() {
            return Ok(Self::disabled());
        }
        validate_dense_bf16_adapter_graph(graph, plan)?;
        let hidden_size = graph
            .tensor_metadata()
            .iter()
            .find(|tensor| tensor.name() == "embedding.output")
            .and_then(|tensor| tensor.view().shape().get(1).copied())
            .ok_or_else(|| {
                QwenExecutionError::InvalidGraph(
                    "adapter provisioning cannot resolve Qwen hidden size".to_owned(),
                )
            })?;
        let layer_count = u64::try_from(graph.layer_types().len()).map_err(|_| {
            QwenExecutionError::InvalidGraph("Qwen layer count does not fit u64".to_owned())
        })?;
        let mut lora = Vec::with_capacity(adapters.adapters().len());
        for selection in adapters.adapters().iter().cloned() {
            let lock = selection.artifact.lock();
            if lock.base_model_fingerprint != graph.model_fingerprint()
                || lock.base_weight_plan_digest != qwen_plan_digest_string(graph.plan_digest())
            {
                return Err(QwenExecutionError::InvalidRequest(
                    "adapter identity differs from the request graph".to_owned(),
                ));
            }
            for target in selection.artifact.targets() {
                validate_lora_target_graph(graph, target)?;
            }
            let buffer = session.allocate_with_category(
                u64::try_from(selection.artifact.payload().len()).map_err(|_| {
                    QwenExecutionError::InvalidRequest(
                        "LoRA payload length does not fit u64".to_owned(),
                    )
                })?,
                crate::AllocationCategory::RequestState,
            )?;
            let view = TensorView::contiguous(DType::U8, &[selection.artifact.payload().len()])?;
            upload_exact_bytes(
                session,
                queue,
                &buffer,
                &view,
                selection.artifact.payload(),
                completion_timeout,
                "LoRA adapter payload upload",
            )?;
            lora.push(QwenLoraDeviceArtifact { selection, buffer });
        }
        let mut controls = Vec::with_capacity(adapters.controls().len());
        for selection in adapters.controls().iter().cloned() {
            let lock = selection.artifact.lock();
            if lock.base_model_fingerprint != graph.model_fingerprint()
                || lock.base_weight_plan_digest != qwen_plan_digest_string(graph.plan_digest())
                || lock.layer_end > layer_count
                || lock.hidden_size != u64::try_from(hidden_size).unwrap_or(u64::MAX)
            {
                return Err(QwenExecutionError::InvalidRequest(
                    "control-vector identity differs from the request graph".to_owned(),
                ));
            }
            let buffer = session.allocate_with_category(
                u64::try_from(selection.artifact.payload().len()).map_err(|_| {
                    QwenExecutionError::InvalidRequest(
                        "control-vector payload length does not fit u64".to_owned(),
                    )
                })?,
                crate::AllocationCategory::RequestState,
            )?;
            let view = TensorView::contiguous(DType::U8, &[selection.artifact.payload().len()])?;
            upload_exact_bytes(
                session,
                queue,
                &buffer,
                &view,
                selection.artifact.payload(),
                completion_timeout,
                "control-vector payload upload",
            )?;
            controls.push(QwenControlDeviceArtifact { selection, buffer });
        }
        Ok(Self {
            identity,
            lora,
            controls,
        })
    }
}

fn validate_dense_bf16_adapter_graph(
    graph: &QwenGraph,
    plan: &WeightLoadPlan,
) -> Result<(), QwenExecutionError> {
    if graph.model_fingerprint() != QWEN35_4B_FINGERPRINT
        || graph.fp8_sidecar_fingerprint().is_some()
        || graph.is_mtp()
        || graph.is_multimodal()
        || graph.layer_types().len() != 32
    {
        return Err(QwenExecutionError::InvalidRequest(
            "Phase45 adapters require the reviewed dense BF16 Qwen3.5-4B text graph".to_owned(),
        ));
    }
    if graph.nodes().iter().any(|node| {
        matches!(
            node.kind(),
            QwenGraphNodeKind::Semantic(SemanticOpKind::SparseMoe)
        )
    }) {
        return Err(QwenExecutionError::InvalidRequest(
            "Phase45 adapters reject sparse-MoE Qwen graphs".to_owned(),
        ));
    }
    if graph
        .weight_bindings()
        .iter()
        .any(|binding| !is_reviewed_dense_adapter_weight(binding.tensor_name(), binding.dtype()))
    {
        return Err(QwenExecutionError::InvalidRequest(
            "Phase45 adapters require reviewed BF16 weights plus fixed F32 GDN state weights"
                .to_owned(),
        ));
    }
    if !matches!(
        plan.schema_version.as_str(),
        "model-lock-v1" | "gguf-model-plan-v1"
    ) || plan.entries.iter().any(|entry| {
        entry.classification == WeightClassification::Required
            && !is_reviewed_dense_adapter_weight(&entry.tensor_name, entry.dtype)
    }) {
        return Err(QwenExecutionError::InvalidRequest(
            "Phase45 adapters require a reviewed dense BF16 weight plan".to_owned(),
        ));
    }
    Ok(())
}

fn is_reviewed_dense_adapter_weight(name: &str, dtype: TensorDType) -> bool {
    match dtype {
        TensorDType::Bf16 => true,
        // Qwen3.5's reviewed dense text graph keeps only these two GDN
        // state parameters in F32; all other adapter-visible weights remain
        // BF16.
        TensorDType::F32 => {
            let Some(rest) = name.strip_prefix("model.language_model.layers.") else {
                return false;
            };
            let Some((layer, suffix)) = rest.split_once(".linear_attn.") else {
                return false;
            };
            layer.parse::<u32>().is_ok() && matches!(suffix, "A_log" | "norm.weight")
        }
        TensorDType::F16 | TensorDType::I32 | TensorDType::I64 | TensorDType::U8 => false,
    }
}

fn qwen_plan_digest_string(digest: &[u8; 32]) -> String {
    let mut output = String::from("sha256:");
    for byte in digest {
        use std::fmt::Write as _;
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn validate_lora_target_graph(
    graph: &QwenGraph,
    target: &VerifiedLoraTargetV1,
) -> Result<(), QwenExecutionError> {
    let binding = graph
        .weight_binding(target.tensor_name())
        .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
    if binding.dtype() != TensorDType::Bf16 || binding.shape() != target.target_shape().as_slice() {
        return Err(QwenExecutionError::InvalidRequest(format!(
            "LoRA target {} differs from the dense BF16 graph",
            target.tensor_name()
        )));
    }
    let is_matmul_target = graph.nodes().iter().any(|node| {
        node.inputs().get(1).is_some_and(|&weight_id| {
            graph
                .tensor_metadata()
                .get(weight_id)
                .is_some_and(|tensor| tensor.name() == target.tensor_name())
        }) && matches!(
            node.kind(),
            QwenGraphNodeKind::Semantic(SemanticOpKind::Matmul)
        )
    });
    if !is_matmul_target {
        return Err(QwenExecutionError::InvalidRequest(format!(
            "LoRA target {} is not a Qwen matmul weight",
            target.tensor_name()
        )));
    }
    Ok(())
}

trait QwenProvisionSource {
    fn upload_weight(
        &self,
        plan: &WeightLoadPlan,
        binding: &QwenGraphWeightBinding,
        session: &ExecutionSession,
        queue: &ExecutionQueue,
        destination: crate::BufferRange,
        completion_timeout: Duration,
    ) -> Result<(), QwenExecutionError>;

    #[allow(clippy::too_many_arguments)]
    fn upload_weight_for_resident_dtype(
        &self,
        plan: &WeightLoadPlan,
        binding: &QwenGraphWeightBinding,
        session: &ExecutionSession,
        queue: &ExecutionQueue,
        destination: crate::BufferRange,
        resident_dtype: DType,
        completion_timeout: Duration,
    ) -> Result<(), QwenExecutionError> {
        let _ = resident_dtype;
        self.upload_weight(
            plan,
            binding,
            session,
            queue,
            destination,
            completion_timeout,
        )
    }

    fn read_scale_bytes(
        &self,
        tensor_name: &str,
        expected_length: usize,
    ) -> Result<Arc<[u8]>, QwenExecutionError>;
}

struct VerifiedProvisionSource {
    cache: Arc<VerifiedCache>,
}

struct GgufProvisionSource {
    source: Arc<VerifiedGgufWeightSource>,
}

struct Fp8ProvisionSource {
    cache: Arc<VerifiedCache>,
    sidecar: Arc<VerifiedFp8Sidecar>,
}

struct Fp8FnuzProvisionSource {
    cache: Arc<VerifiedCache>,
    sidecar: Arc<VerifiedFp8Sidecar>,
}

struct Fp8ConvertedProvisionSource {
    cache: Arc<VerifiedCache>,
    sidecar: Arc<VerifiedFp8Sidecar>,
}

struct Nvfp4ProvisionSource {
    cache: Arc<VerifiedCache>,
    sidecar: Arc<VerifiedNvfp4Sidecar>,
}

struct Qwen35MoeProvisionSource {
    artifact: Arc<VerifiedQwen35Moe>,
}

struct GgufMoeProvisionSource {
    source: Arc<VerifiedGgufQwen35Moe>,
}

impl QwenProvisionSource for GgufMoeProvisionSource {
    fn upload_weight(
        &self,
        _plan: &WeightLoadPlan,
        binding: &QwenGraphWeightBinding,
        session: &ExecutionSession,
        queue: &ExecutionQueue,
        destination: crate::BufferRange,
        completion_timeout: Duration,
    ) -> Result<(), QwenExecutionError> {
        let bytes = if let Some(layer) = binding
            .tensor_name()
            .strip_prefix(QWEN35_MOE_LAYER_BLOB_PREFIX)
            .and_then(|value| value.parse::<u16>().ok())
        {
            self.pack_layer_blob(layer)?
        } else {
            self.source
                .read_tensor(binding.tensor_name())
                .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?
        };
        if u64::try_from(bytes.len()).ok() != Some(destination.size_bytes()) {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "GGUF MoE resident allocation differs for {}",
                binding.tensor_name()
            )));
        }
        upload_buffer_bytes(
            session,
            queue,
            &destination,
            &bytes,
            completion_timeout,
            "GGUF MoE weight upload",
        )
    }

    fn read_scale_bytes(
        &self,
        tensor_name: &str,
        expected_length: usize,
    ) -> Result<Arc<[u8]>, QwenExecutionError> {
        let bytes = self
            .source
            .read_tensor(tensor_name)
            .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
        if bytes.len() != expected_length {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "GGUF MoE attention scale length differs: {tensor_name}"
            )));
        }
        Ok(Arc::from(bytes))
    }
}

impl GgufMoeProvisionSource {
    fn pack_layer_blob(&self, layer: u16) -> Result<Vec<u8>, QwenExecutionError> {
        const GATE_VALUES: usize = 0;
        const GATE_SCALES: usize = 134_217_728;
        const UP_VALUES: usize = 142_606_336;
        const UP_SCALES: usize = 276_824_064;
        const DOWN_VALUES: usize = 285_212_672;
        const DOWN_SCALES: usize = 419_430_400;
        const SHARED_GATE: usize = 427_819_008;
        const SHARED_UP: usize = 429_916_160;
        const SHARED_DOWN: usize = 432_013_312;
        const SHARED_EXPERT_GATE: usize = 434_110_464;
        let mut blob = vec![0_u8; QWEN35_MOE_LAYER_BLOB_BYTES as usize];
        for expert in 0..256_u16 {
            for (projection, value_base, scale_base, value_stride, scale_stride) in [
                (
                    Qwen35MoeExpertProjection::Gate,
                    GATE_VALUES,
                    GATE_SCALES,
                    524_288,
                    32_768,
                ),
                (
                    Qwen35MoeExpertProjection::Up,
                    UP_VALUES,
                    UP_SCALES,
                    524_288,
                    32_768,
                ),
                (
                    Qwen35MoeExpertProjection::Down,
                    DOWN_VALUES,
                    DOWN_SCALES,
                    524_288,
                    32_768,
                ),
            ] {
                let (values, scales) = self
                    .source
                    .read_expert_planes(layer, expert, projection)
                    .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
                copy_bytes_into(
                    &values,
                    &mut blob,
                    value_base + usize::from(expert) * value_stride,
                )?;
                copy_bytes_into(
                    &scales,
                    &mut blob,
                    scale_base + usize::from(expert) * scale_stride,
                )?;
            }
        }
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        for (name, offset) in [
            (
                format!("{prefix}.shared_expert.gate_proj.weight"),
                SHARED_GATE,
            ),
            (format!("{prefix}.shared_expert.up_proj.weight"), SHARED_UP),
            (
                format!("{prefix}.shared_expert.down_proj.weight"),
                SHARED_DOWN,
            ),
            (
                format!("{prefix}.shared_expert_gate.weight"),
                SHARED_EXPERT_GATE,
            ),
        ] {
            let bytes = self
                .source
                .read_tensor(&name)
                .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
            copy_bytes_into(&bytes, &mut blob, offset)?;
        }
        Ok(blob)
    }
}

fn copy_bytes_into(
    source: &[u8],
    output: &mut [u8],
    offset: usize,
) -> Result<(), QwenExecutionError> {
    let end = offset.checked_add(source.len()).ok_or_else(|| {
        QwenExecutionError::InvalidRequest("MoE blob byte offset overflowed".to_owned())
    })?;
    let destination = output.get_mut(offset..end).ok_or_else(|| {
        QwenExecutionError::InvalidRequest("MoE blob bytes exceed layout".to_owned())
    })?;
    destination.copy_from_slice(source);
    Ok(())
}

impl QwenProvisionSource for Qwen35MoeProvisionSource {
    fn upload_weight(
        &self,
        plan: &WeightLoadPlan,
        binding: &QwenGraphWeightBinding,
        session: &ExecutionSession,
        queue: &ExecutionQueue,
        destination: crate::BufferRange,
        completion_timeout: Duration,
    ) -> Result<(), QwenExecutionError> {
        if plan.lock_fingerprint != crate::QWEN35_MOE_MODEL_FINGERPRINT {
            return Err(QwenExecutionError::InvalidRequest(
                "MoE load plan identity differs".to_owned(),
            ));
        }
        if let Some(layer_text) = binding
            .tensor_name()
            .strip_prefix(QWEN35_MOE_LAYER_BLOB_PREFIX)
        {
            let layer = layer_text.parse::<u16>().map_err(|_| {
                QwenExecutionError::InvalidRequest("MoE layer blob name is malformed".to_owned())
            })?;
            let bytes = self.pack_layer_blob(layer)?;
            return upload_buffer_bytes(
                session,
                queue,
                &destination,
                &bytes,
                completion_timeout,
                "Qwen3.5 MoE layer blob upload",
            );
        }
        let plane = self.artifact.plane(binding.tensor_name()).ok_or_else(|| {
            QwenExecutionError::InvalidRequest(format!(
                "MoE execution plane is absent: {}",
                binding.tensor_name()
            ))
        })?;
        upload_moe_plane(
            self.artifact.as_ref(),
            plane,
            session,
            queue,
            &destination,
            completion_timeout,
        )
    }

    fn read_scale_bytes(
        &self,
        tensor_name: &str,
        expected_length: usize,
    ) -> Result<Arc<[u8]>, QwenExecutionError> {
        let plane = self.artifact.plane(tensor_name).ok_or_else(|| {
            QwenExecutionError::InvalidRequest(format!(
                "MoE attention scale plane is absent: {tensor_name}"
            ))
        })?;
        let bytes = read_moe_plane(self.artifact.as_ref(), plane)?;
        if bytes.len() != expected_length {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "MoE attention scale length differs: {tensor_name}"
            )));
        }
        Ok(Arc::from(bytes))
    }
}

impl Qwen35MoeProvisionSource {
    fn pack_layer_blob(&self, layer: u16) -> Result<Vec<u8>, QwenExecutionError> {
        const GATE_VALUES: usize = 0;
        const GATE_SCALES: usize = 134_217_728;
        const UP_VALUES: usize = 142_606_336;
        const UP_SCALES: usize = 276_824_064;
        const DOWN_VALUES: usize = 285_212_672;
        const DOWN_SCALES: usize = 419_430_400;
        const SHARED_GATE: usize = 427_819_008;
        const SHARED_UP: usize = 429_916_160;
        const SHARED_DOWN: usize = 432_013_312;
        const SHARED_EXPERT_GATE: usize = 434_110_464;
        let mut blob = vec![0_u8; QWEN35_MOE_LAYER_BLOB_BYTES as usize];
        for expert in 0..256_u16 {
            for (projection, value_base, scale_base, value_stride, scale_stride) in [
                (
                    Qwen35MoeExpertProjection::Gate,
                    GATE_VALUES,
                    GATE_SCALES,
                    524_288,
                    32_768,
                ),
                (
                    Qwen35MoeExpertProjection::Up,
                    UP_VALUES,
                    UP_SCALES,
                    524_288,
                    32_768,
                ),
                (
                    Qwen35MoeExpertProjection::Down,
                    DOWN_VALUES,
                    DOWN_SCALES,
                    524_288,
                    32_768,
                ),
            ] {
                let descriptor =
                    self.artifact
                        .expert(layer, expert, projection)
                        .ok_or_else(|| {
                            QwenExecutionError::InvalidRequest(
                                "MoE expert descriptor is absent".to_owned(),
                            )
                        })?;
                copy_plane_into(
                    self.artifact.as_ref(),
                    &descriptor.value,
                    &mut blob,
                    value_base + usize::from(expert) * value_stride,
                )?;
                copy_plane_into(
                    self.artifact.as_ref(),
                    &descriptor.scale,
                    &mut blob,
                    scale_base + usize::from(expert) * scale_stride,
                )?;
            }
        }
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        for (name, offset) in [
            (
                format!("{prefix}.shared_expert.gate_proj.weight"),
                SHARED_GATE,
            ),
            (format!("{prefix}.shared_expert.up_proj.weight"), SHARED_UP),
            (
                format!("{prefix}.shared_expert.down_proj.weight"),
                SHARED_DOWN,
            ),
            (
                format!("{prefix}.shared_expert_gate.weight"),
                SHARED_EXPERT_GATE,
            ),
        ] {
            let plane = self.artifact.plane(&name).ok_or_else(|| {
                QwenExecutionError::InvalidRequest(format!("MoE shared plane is absent: {name}"))
            })?;
            copy_plane_into(self.artifact.as_ref(), plane, &mut blob, offset)?;
        }
        Ok(blob)
    }
}

fn copy_plane_into(
    artifact: &VerifiedQwen35Moe,
    plane: &Qwen35MoeTensorPlane,
    output: &mut [u8],
    offset: usize,
) -> Result<(), QwenExecutionError> {
    let bytes = read_moe_plane(artifact, plane)?;
    let end = offset.checked_add(bytes.len()).ok_or_else(|| {
        QwenExecutionError::InvalidRequest("MoE blob offset overflowed".to_owned())
    })?;
    let destination = output.get_mut(offset..end).ok_or_else(|| {
        QwenExecutionError::InvalidRequest("MoE blob plane exceeds layout".to_owned())
    })?;
    destination.copy_from_slice(&bytes);
    Ok(())
}

fn read_moe_plane(
    artifact: &VerifiedQwen35Moe,
    plane: &Qwen35MoeTensorPlane,
) -> Result<Vec<u8>, QwenExecutionError> {
    let length = usize::try_from(plane.absolute_byte_range[1] - plane.absolute_byte_range[0])
        .map_err(|_| QwenExecutionError::InvalidRequest("MoE plane is too large".to_owned()))?;
    artifact
        .read_plane_range(plane, 0, length)
        .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))
}

fn upload_moe_plane(
    artifact: &VerifiedQwen35Moe,
    plane: &Qwen35MoeTensorPlane,
    session: &ExecutionSession,
    queue: &ExecutionQueue,
    destination: &crate::BufferRange,
    completion_timeout: Duration,
) -> Result<(), QwenExecutionError> {
    let total = plane.absolute_byte_range[1]
        .checked_sub(plane.absolute_byte_range[0])
        .ok_or_else(|| {
            QwenExecutionError::InvalidRequest("MoE plane range underflow".to_owned())
        })?;
    if total != destination.size_bytes() || total == 0 {
        return Err(QwenExecutionError::InvalidRequest(
            "MoE plane does not exactly match its destination".to_owned(),
        ));
    }
    let maximum = session
        .max_transfer_bytes()?
        .min(crate::WEIGHT_LOAD_CHUNK_BYTES);
    if maximum == 0 {
        return Err(QwenExecutionError::InvalidRequest(
            "MoE backend transfer limit is zero".to_owned(),
        ));
    }
    let mut relative = 0_u64;
    while relative < total {
        let length = (total - relative).min(maximum);
        let length_usize = usize::try_from(length).map_err(|_| {
            QwenExecutionError::InvalidRequest("MoE transfer length is too large".to_owned())
        })?;
        let bytes = artifact
            .read_plane_range(plane, relative, length_usize)
            .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
        let absolute = destination
            .offset_bytes()
            .checked_add(relative)
            .ok_or_else(|| QwenExecutionError::InvalidRequest("MoE upload overflow".to_owned()))?;
        let range = destination.buffer().range(absolute, length)?;
        let mut transfer = session.upload(queue, range, Arc::from(bytes))?;
        require_terminal_success(
            "Qwen3.5 MoE tensor upload",
            transfer.wait(completion_timeout)?,
        )?;
        relative += length;
    }
    Ok(())
}

impl QwenProvisionSource for VerifiedProvisionSource {
    fn upload_weight(
        &self,
        plan: &WeightLoadPlan,
        binding: &QwenGraphWeightBinding,
        session: &ExecutionSession,
        queue: &ExecutionQueue,
        destination: crate::BufferRange,
        completion_timeout: Duration,
    ) -> Result<(), QwenExecutionError> {
        let receipt = upload_verified_weight(WeightUploadRequest {
            plan,
            expected_plan_digest: *plan.digest(),
            cache: self.cache.as_ref(),
            tensor_name: binding.tensor_name(),
            expected_dtype: binding.dtype(),
            session,
            queue,
            destination,
            completion_timeout,
        })?;
        validate_upload_receipt(&receipt, plan, binding)
    }

    fn read_scale_bytes(
        &self,
        tensor_name: &str,
        expected_length: usize,
    ) -> Result<Arc<[u8]>, QwenExecutionError> {
        let bytes = self
            .cache
            .read_tensor_range(tensor_name, 0, expected_length)
            .map_err(|error| {
                QwenExecutionError::InvalidRequest(format!(
                    "could not read verified attention scale {tensor_name}: {error}"
                ))
            })?;
        if bytes.len() != expected_length {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "verified attention scale {tensor_name} has a short or long byte read"
            )));
        }
        Ok(Arc::from(bytes))
    }
}

impl QwenProvisionSource for GgufProvisionSource {
    fn upload_weight(
        &self,
        plan: &WeightLoadPlan,
        binding: &QwenGraphWeightBinding,
        session: &ExecutionSession,
        queue: &ExecutionQueue,
        destination: crate::BufferRange,
        completion_timeout: Duration,
    ) -> Result<(), QwenExecutionError> {
        self.upload_weight_for_resident_dtype(
            plan,
            binding,
            session,
            queue,
            destination,
            DType::F8E4M3Fn,
            completion_timeout,
        )
    }

    fn upload_weight_for_resident_dtype(
        &self,
        plan: &WeightLoadPlan,
        binding: &QwenGraphWeightBinding,
        session: &ExecutionSession,
        queue: &ExecutionQueue,
        destination: crate::BufferRange,
        resident_dtype: DType,
        completion_timeout: Duration,
    ) -> Result<(), QwenExecutionError> {
        if let Some(recipe) = self.source.recipe_binding(binding.tensor_name()) {
            if !matches!(resident_dtype, DType::F8E4M3Fn | DType::F8E4M3FnuZ) {
                return Err(QwenExecutionError::InvalidRequest(
                    "GGUF FP8 recipe requires an OCP E4M3FN or FNUZ resident dtype".to_owned(),
                ));
            }
            let value = self
                .source
                .gguf()
                .tensor(&recipe.value_tensor)
                .ok_or_else(|| {
                    QwenExecutionError::InvalidRequest(
                        "GGUF FP8 value tensor is absent after verification".to_owned(),
                    )
                })?;
            let value_len = usize::try_from(value.byte_length()).map_err(|_| {
                QwenExecutionError::InvalidRequest("GGUF FP8 value is too large".to_owned())
            })?;
            let values = self
                .source
                .gguf()
                .read_tensor_range(&recipe.value_tensor, 0, value_len)
                .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
            let scale = recipe.scales.first().ok_or_else(|| {
                QwenExecutionError::InvalidRequest("GGUF FP8 scale binding is absent".to_owned())
            })?;
            if recipe.scales.len() != 1 || scale.role != crate::GgufScaleRole::Channel {
                return Err(QwenExecutionError::InvalidRequest(
                    "GGUF FP8 scale recipe is not exact channel scaling".to_owned(),
                ));
            }
            let scale_info = self.source.gguf().tensor(&scale.tensor).ok_or_else(|| {
                QwenExecutionError::InvalidRequest(
                    "GGUF FP8 scale tensor is absent after verification".to_owned(),
                )
            })?;
            let scale_len = usize::try_from(scale_info.byte_length()).map_err(|_| {
                QwenExecutionError::InvalidRequest("GGUF FP8 scale is too large".to_owned())
            })?;
            let scale_bytes = self
                .source
                .gguf()
                .read_tensor_range(&scale.tensor, 0, scale_len)
                .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
            let normalized_scales = normalize_gguf_fp8_scales(recipe.encoding, &scale_bytes)?;
            let combined = gguf_fp8_resident_payload(&values, &normalized_scales, resident_dtype)?;
            if u64::try_from(combined.len()).ok() != Some(destination.size_bytes()) {
                return Err(QwenExecutionError::InvalidRequest(format!(
                    "GGUF FP8 resident allocation differs for {}",
                    binding.tensor_name()
                )));
            }
            return upload_buffer_bytes(
                session,
                queue,
                &destination,
                &combined,
                completion_timeout,
                "GGUF FP8 weight/scale upload",
            );
        }
        if self.source.has_fp8_recipe() {
            let tensor = self
                .source
                .gguf()
                .tensor(binding.tensor_name())
                .ok_or_else(|| {
                    QwenExecutionError::InvalidRequest(
                        "GGUF BF16 tensor is absent after verification".to_owned(),
                    )
                })?;
            let length = usize::try_from(tensor.byte_length()).map_err(|_| {
                QwenExecutionError::InvalidRequest("GGUF BF16 tensor is too large".to_owned())
            })?;
            let bytes = self
                .source
                .gguf()
                .read_tensor_range(binding.tensor_name(), 0, length)
                .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
            if u64::try_from(bytes.len()).ok() != Some(destination.size_bytes()) {
                return Err(QwenExecutionError::InvalidRequest(format!(
                    "GGUF BF16 resident allocation differs for {}",
                    binding.tensor_name()
                )));
            }
            return upload_buffer_bytes(
                session,
                queue,
                &destination,
                &bytes,
                completion_timeout,
                "GGUF BF16 weight upload",
            );
        }
        let receipt = upload_verified_gguf_weight(GgufWeightUploadRequest {
            plan,
            expected_plan_digest: *plan.digest(),
            source: self.source.as_ref(),
            tensor_name: binding.tensor_name(),
            expected_dtype: binding.dtype(),
            session,
            queue,
            destination,
            completion_timeout,
        })?;
        validate_upload_receipt(&receipt, plan, binding)
    }

    fn read_scale_bytes(
        &self,
        tensor_name: &str,
        expected_length: usize,
    ) -> Result<Arc<[u8]>, QwenExecutionError> {
        let bytes = self
            .source
            .gguf()
            .read_tensor_range(tensor_name, 0, expected_length)
            .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
        if bytes.len() != expected_length {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "verified GGUF tensor {tensor_name} returned a short scale read"
            )));
        }
        Ok(Arc::from(bytes))
    }
}

fn gguf_fp8_resident_payload(
    values: &[u8],
    normalized_scales: &[u8],
    resident_dtype: DType,
) -> Result<Vec<u8>, QwenExecutionError> {
    if !matches!(resident_dtype, DType::F8E4M3Fn | DType::F8E4M3FnuZ) {
        return Err(QwenExecutionError::InvalidRequest(
            "GGUF FP8 recipe requires an OCP E4M3FN or FNUZ resident dtype".to_owned(),
        ));
    }
    let (values, scales) = if resident_dtype == DType::F8E4M3FnuZ {
        rebase_e4m3fn_outer_rows_to_fnuz(values, normalized_scales)?
    } else {
        (values.to_vec(), normalized_scales.to_vec())
    };
    let mut combined = values;
    combined.extend_from_slice(&scales);
    Ok(combined)
}

fn normalize_gguf_fp8_scales(
    encoding: crate::GgufRecipeEncoding,
    scale_bytes: &[u8],
) -> Result<Vec<u8>, QwenExecutionError> {
    match encoding {
        crate::GgufRecipeEncoding::Fp8E4m3fnChannelF32Scale => Ok(scale_bytes.to_vec()),
        crate::GgufRecipeEncoding::Fp8E4m3fnChannelBf16Scale => {
            if scale_bytes.len() % 2 != 0 {
                return Err(QwenExecutionError::InvalidRequest(
                    "GGUF FP8 BF16 scale byte count is odd".to_owned(),
                ));
            }
            let mut normalized = Vec::with_capacity(scale_bytes.len() * 2);
            for chunk in scale_bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                normalized.extend_from_slice(&f32::from_bits(u32::from(bits) << 16).to_le_bytes());
            }
            Ok(normalized)
        }
        _ => Err(QwenExecutionError::InvalidRequest(
            "GGUF Qwen recipe is not FP8".to_owned(),
        )),
    }
}

impl QwenProvisionSource for Fp8ProvisionSource {
    fn upload_weight(
        &self,
        plan: &WeightLoadPlan,
        binding: &QwenGraphWeightBinding,
        session: &ExecutionSession,
        queue: &ExecutionQueue,
        destination: crate::BufferRange,
        completion_timeout: Duration,
    ) -> Result<(), QwenExecutionError> {
        if let Some(tensor) = self.sidecar.tensor(binding.tensor_name()) {
            if tensor.shape.as_slice() != binding.shape() {
                return Err(QwenExecutionError::InvalidRequest(format!(
                    "FP8 sidecar shape differs for {}",
                    binding.tensor_name()
                )));
            }
            let (values, scales) = self
                .sidecar
                .read_tensor_bytes(binding.tensor_name())
                .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
            let expected = values.len().checked_add(scales.len()).ok_or_else(|| {
                QwenExecutionError::InvalidRequest("FP8 resident upload size overflowed".to_owned())
            })?;
            if u64::try_from(expected).ok() != Some(destination.size_bytes()) {
                return Err(QwenExecutionError::InvalidRequest(format!(
                    "FP8 resident allocation differs for {}",
                    binding.tensor_name()
                )));
            }
            let mut combined = values;
            combined.extend_from_slice(&scales);
            upload_buffer_bytes(
                session,
                queue,
                &destination,
                &combined,
                completion_timeout,
                "FP8 weight/scale upload",
            )
        } else {
            VerifiedProvisionSource {
                cache: Arc::clone(&self.cache),
            }
            .upload_weight(
                plan,
                binding,
                session,
                queue,
                destination,
                completion_timeout,
            )
        }
    }

    fn read_scale_bytes(
        &self,
        tensor_name: &str,
        expected_length: usize,
    ) -> Result<Arc<[u8]>, QwenExecutionError> {
        VerifiedProvisionSource {
            cache: Arc::clone(&self.cache),
        }
        .read_scale_bytes(tensor_name, expected_length)
    }
}

impl QwenProvisionSource for Fp8FnuzProvisionSource {
    fn upload_weight(
        &self,
        plan: &WeightLoadPlan,
        binding: &QwenGraphWeightBinding,
        session: &ExecutionSession,
        queue: &ExecutionQueue,
        destination: crate::BufferRange,
        completion_timeout: Duration,
    ) -> Result<(), QwenExecutionError> {
        let Some(tensor) = self.sidecar.tensor(binding.tensor_name()) else {
            return VerifiedProvisionSource {
                cache: Arc::clone(&self.cache),
            }
            .upload_weight(
                plan,
                binding,
                session,
                queue,
                destination,
                completion_timeout,
            );
        };
        if tensor.shape.as_slice() != binding.shape() {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "FNUZ resident contract differs for {}",
                binding.tensor_name()
            )));
        }
        let (values, scales) = self
            .sidecar
            .read_tensor_bytes(binding.tensor_name())
            .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
        let (mut combined, rebased_scales) = rebase_e4m3fn_outer_rows_to_fnuz(&values, &scales)?;
        combined.extend_from_slice(&rebased_scales);
        if u64::try_from(combined.len()).ok() != Some(destination.size_bytes()) {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "FNUZ resident allocation differs for {}",
                binding.tensor_name()
            )));
        }
        upload_buffer_bytes(
            session,
            queue,
            &destination,
            &combined,
            completion_timeout,
            "FNUZ weight/scale upload",
        )
    }

    fn read_scale_bytes(
        &self,
        tensor_name: &str,
        expected_length: usize,
    ) -> Result<Arc<[u8]>, QwenExecutionError> {
        VerifiedProvisionSource {
            cache: Arc::clone(&self.cache),
        }
        .read_scale_bytes(tensor_name, expected_length)
    }
}

fn rebase_e4m3fn_outer_rows_to_fnuz(
    values: &[u8],
    scales: &[u8],
) -> Result<(Vec<u8>, Vec<u8>), QwenExecutionError> {
    if scales.is_empty() || scales.len() % 4 != 0 {
        return Err(QwenExecutionError::InvalidRequest(
            "FNUZ resident scale payload is not a non-empty FP32 array".to_owned(),
        ));
    }
    let mut rebased_values = Vec::with_capacity(values.len());
    for &bits in values {
        if bits & 0x7f == 0x7f {
            return Err(QwenExecutionError::InvalidRequest(
                "FNUZ resident conversion refuses an OCP NaN byte".to_owned(),
            ));
        }
        // For every finite nonzero code, FNUZ decodes the same byte to
        // exactly half the OCP value. Preserve the byte and compensate once
        // in the outer-row scale; map OCP negative zero away from FNUZ NaN.
        rebased_values.push(if bits == 0x80 { 0 } else { bits });
    }
    let mut rebased_scales = Vec::with_capacity(scales.len());
    for bytes in scales.chunks_exact(4) {
        let scale = f32::from_le_bytes(bytes.try_into().expect("four-byte scale chunk"));
        let rebased = scale * 2.0;
        if !scale.is_finite() || scale <= 0.0 || !rebased.is_finite() {
            return Err(QwenExecutionError::InvalidRequest(
                "FNUZ resident conversion requires positive finite FP32 scales".to_owned(),
            ));
        }
        rebased_scales.extend_from_slice(&rebased.to_le_bytes());
    }
    Ok((rebased_values, rebased_scales))
}

impl QwenProvisionSource for Nvfp4ProvisionSource {
    fn upload_weight(
        &self,
        plan: &WeightLoadPlan,
        binding: &QwenGraphWeightBinding,
        session: &ExecutionSession,
        queue: &ExecutionQueue,
        destination: crate::BufferRange,
        completion_timeout: Duration,
    ) -> Result<(), QwenExecutionError> {
        let Some(tensor) = self.sidecar.tensor(binding.tensor_name()) else {
            return VerifiedProvisionSource {
                cache: Arc::clone(&self.cache),
            }
            .upload_weight(
                plan,
                binding,
                session,
                queue,
                destination,
                completion_timeout,
            );
        };
        if tensor.shape.as_slice() != binding.shape() {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "NVFP4 sidecar shape differs for {}",
                binding.tensor_name()
            )));
        }
        let (values, block_scales, tensor_scale) = self
            .sidecar
            .read_tensor_bytes(binding.tensor_name())
            .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
        let unaligned = values
            .len()
            .checked_add(block_scales.len())
            .ok_or_else(|| {
                QwenExecutionError::InvalidRequest("NVFP4 resident size overflowed".to_owned())
            })?;
        let tensor_scale_offset = unaligned.checked_add(3).ok_or_else(|| {
            QwenExecutionError::InvalidRequest("NVFP4 scale alignment overflowed".to_owned())
        })? & !3;
        let expected = tensor_scale_offset.checked_add(4).ok_or_else(|| {
            QwenExecutionError::InvalidRequest("NVFP4 resident size overflowed".to_owned())
        })?;
        if u64::try_from(expected).ok() != Some(destination.size_bytes()) {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "NVFP4 resident allocation differs for {}",
                binding.tensor_name()
            )));
        }
        let mut combined = values;
        combined.extend_from_slice(&block_scales);
        combined.resize(tensor_scale_offset, 0);
        combined.extend_from_slice(&tensor_scale);
        upload_buffer_bytes(
            session,
            queue,
            &destination,
            &combined,
            completion_timeout,
            "NVFP4 value/block/tensor-scale upload",
        )
    }

    fn read_scale_bytes(
        &self,
        tensor_name: &str,
        expected_length: usize,
    ) -> Result<Arc<[u8]>, QwenExecutionError> {
        VerifiedProvisionSource {
            cache: Arc::clone(&self.cache),
        }
        .read_scale_bytes(tensor_name, expected_length)
    }
}

impl QwenProvisionSource for Fp8ConvertedProvisionSource {
    fn upload_weight(
        &self,
        plan: &WeightLoadPlan,
        binding: &QwenGraphWeightBinding,
        session: &ExecutionSession,
        queue: &ExecutionQueue,
        destination: crate::BufferRange,
        completion_timeout: Duration,
    ) -> Result<(), QwenExecutionError> {
        let Some(tensor) = self.sidecar.tensor(binding.tensor_name()) else {
            return VerifiedProvisionSource {
                cache: Arc::clone(&self.cache),
            }
            .upload_weight(
                plan,
                binding,
                session,
                queue,
                destination,
                completion_timeout,
            );
        };
        if tensor.shape.as_slice() != binding.shape() {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "FP8 conversion shape differs for {}",
                binding.tensor_name()
            )));
        }
        let (values, scale_bytes) = self
            .sidecar
            .read_tensor_bytes(binding.tensor_name())
            .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
        let rows = usize::try_from(tensor.shape[0]).map_err(|_| {
            QwenExecutionError::InvalidRequest("FP8 conversion row count is too large".to_owned())
        })?;
        let columns = usize::try_from(tensor.shape[1]).map_err(|_| {
            QwenExecutionError::InvalidRequest(
                "FP8 conversion column count is too large".to_owned(),
            )
        })?;
        if scale_bytes.len() != rows * 4 || values.len() != rows * columns {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "FP8 conversion payload length differs for {}",
                binding.tensor_name()
            )));
        }
        let scales = scale_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect::<Vec<_>>();
        if scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "FP8 conversion scale is invalid for {}",
                binding.tensor_name()
            )));
        }
        let lookup = std::array::from_fn::<_, 256, _>(|bits| decode_e4m3fn(bits as u8));
        let mut converted = vec![0_u8; values.len() * 2];
        let workers = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(rows.max(1));
        let rows_per_worker = rows.div_ceil(workers);
        let input_chunk_bytes = rows_per_worker * columns;
        let output_chunk_bytes = input_chunk_bytes * 2;
        std::thread::scope(|scope| {
            for (chunk_index, output_chunk) in converted.chunks_mut(output_chunk_bytes).enumerate()
            {
                let start_row = chunk_index * rows_per_worker;
                let end_row = (start_row + rows_per_worker).min(rows);
                let input = &values[start_row * columns..end_row * columns];
                let row_scales = &scales[start_row..end_row];
                let lookup = &lookup;
                scope.spawn(move || {
                    for (local_row, row) in input.chunks_exact(columns).enumerate() {
                        let scale = row_scales[local_row];
                        let output_row = &mut output_chunk
                            [local_row * columns * 2..(local_row + 1) * columns * 2];
                        for (index, value) in row.iter().copied().enumerate() {
                            let word = f32_to_bf16_rne(lookup[usize::from(value)] * scale);
                            output_row[index * 2..index * 2 + 2]
                                .copy_from_slice(&word.to_le_bytes());
                        }
                    }
                });
            }
        });
        if u64::try_from(converted.len()).ok() != Some(destination.size_bytes()) {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "converted BF16 resident allocation differs for {}",
                binding.tensor_name()
            )));
        }
        upload_buffer_bytes(
            session,
            queue,
            &destination,
            &converted,
            completion_timeout,
            "FP8-to-BF16 converted weight upload",
        )
    }

    fn read_scale_bytes(
        &self,
        tensor_name: &str,
        expected_length: usize,
    ) -> Result<Arc<[u8]>, QwenExecutionError> {
        VerifiedProvisionSource {
            cache: Arc::clone(&self.cache),
        }
        .read_scale_bytes(tensor_name, expected_length)
    }
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return if bits & 0x007f_ffff == 0 {
            (bits >> 16) as u16
        } else {
            ((bits >> 16) as u16) | 0x0040
        };
    }
    let upper = bits >> 16;
    let lower = bits & 0xffff;
    (upper + u32::from(lower > 0x8000 || (lower == 0x8000 && upper & 1 != 0))) as u16
}

impl QwenResidentInner {
    fn provision<S: QwenProvisionSource>(
        session: Arc<ExecutionSession>,
        graph: QwenGraph,
        plan: WeightLoadPlan,
        completion_timeout: Duration,
        source: &S,
    ) -> Result<Self, QwenExecutionError> {
        let layout = validate_graph_plan(&graph, &plan)?;
        preflight_semantic_support(session.as_ref(), &graph)?;
        preflight_device_memory(session.as_ref(), &graph, &layout, false)?;
        let queue = session.create_queue()?;
        let static_tensors = allocate_resident_tensors(&session, &graph, &layout)?;

        let mut uploaded = BTreeSet::new();
        for binding in graph.weight_bindings() {
            if !uploaded.insert(binding.tensor_name().to_owned()) {
                return Err(QwenExecutionError::InvalidGraph(format!(
                    "required weight is represented more than once: {}",
                    binding.tensor_name()
                )));
            }
            let allocation = static_tensors.get(binding.tensor_name()).ok_or_else(|| {
                QwenExecutionError::InvalidGraph("resident weight allocation is absent".to_owned())
            })?;
            let destination = allocation.buffer.range(
                allocation.graph_view.byte_offset(),
                resident_weight_bytes(&allocation.graph_view)?,
            )?;
            source.upload_weight_for_resident_dtype(
                &plan,
                binding,
                session.as_ref(),
                &queue,
                destination,
                allocation.graph_view.dtype(),
                completion_timeout,
            )?;
        }
        if uploaded.len() != graph.weight_bindings().len() {
            return Err(QwenExecutionError::InvalidGraph(
                "required weight identities are not one-to-one".to_owned(),
            ));
        }

        let scales = provision_resident_scales(
            source,
            &session,
            &queue,
            &graph,
            &static_tensors,
            &layout.scales,
            completion_timeout,
        )?;
        let scales = scales
            .into_iter()
            .map(|(tensor_id, scale)| (graph.tensor_metadata()[tensor_id].name().to_owned(), scale))
            .collect();

        Ok(Self {
            session,
            model_fingerprint: graph.model_fingerprint().to_owned(),
            fp8_sidecar_fingerprint: graph.fp8_sidecar_fingerprint().map(str::to_owned),
            plan,
            queue,
            static_tensors,
            scales,
            completion_timeout,
        })
    }
}

impl QwenExecutionCore {
    fn from_resident(
        resident: Arc<QwenResidentInner>,
        graph: QwenGraph,
        adapters: AdapterRequestSetV1,
    ) -> Result<Self, QwenExecutionError> {
        let layout = validate_graph_plan(&graph, &resident.plan)?;
        if graph.model_fingerprint() != resident.model_fingerprint {
            return Err(QwenExecutionError::InvalidRequest(
                "resident model identity or backend differs from the request graph".to_owned(),
            ));
        }
        if graph.fp8_sidecar_fingerprint() != resident.fp8_sidecar_fingerprint.as_deref() {
            return Err(QwenExecutionError::InvalidRequest(
                "resident FP8 sidecar identity differs from the request graph".to_owned(),
            ));
        }
        validate_resident_graph(&graph, &layout, &resident.static_tensors, &resident.scales)?;
        preflight_device_memory(resident.session.as_ref(), &graph, &layout, true)?;
        let tensors =
            allocate_request_tensors(&resident.session, &graph, &layout, &resident.static_tensors)?;
        let scales = graph
            .nodes()
            .iter()
            .filter_map(|node| match node.kind() {
                QwenGraphNodeKind::AttentionScaleMaterialization { .. } => Some(node.outputs()[0]),
                _ => None,
            })
            .map(|tensor_id| {
                let name = graph.tensor_metadata()[tensor_id].name().to_owned();
                let scale = resident.scales.get(&name).ok_or_else(|| {
                    QwenExecutionError::InvalidGraph(format!(
                        "resident attention scale is absent: {name}"
                    ))
                })?;
                Ok((
                    tensor_id,
                    CachedScale {
                        raw_tensor_id: graph
                            .nodes()
                            .iter()
                            .find(|node| node.outputs().contains(&tensor_id))
                            .and_then(|node| node.inputs().first().copied())
                            .ok_or_else(|| {
                                QwenExecutionError::InvalidGraph(
                                    "resident attention scale input is absent".to_owned(),
                                )
                            })?,
                        raw_bytes: Arc::clone(&scale.raw_bytes),
                        expanded_bytes: Arc::clone(&scale.expanded_bytes),
                    },
                ))
            })
            .collect::<Result<BTreeMap<usize, CachedScale>, QwenExecutionError>>()?;
        let (kv_states, linear_states) = create_states(resident.session.as_ref(), &graph)?;
        let adapters = QwenAdapterRuntime::provision(
            resident.session.as_ref(),
            &resident.queue,
            &graph,
            &resident.plan,
            adapters,
            resident.completion_timeout,
        )?;
        let execution_plan = qwen_prepared_execution_plan(&graph)?;
        let core = Self {
            session: Arc::clone(&resident.session),
            graph,
            execution_plan,
            plan: resident.plan.clone(),
            queue: resident.queue.clone(),
            tensors,
            tensor_ids: layout.tensor_ids,
            dynamic_tensors: layout.dynamic_tensors,
            kv_states,
            linear_states,
            scales,
            completion_timeout: resident.completion_timeout,
            audit: Mutex::new(ExecutionAuditAccumulator::new(1)),
            prepared_semantics: PreparedSemanticCache::default(),
            lifecycle: ExecutionTransaction::new(),
            committed_length: 0,
            prefill_chunk_count: 0,
            rope_position_delta: 0,
            last_output: None,
            pending_speculative: None,
            adapters,
        };
        core.ensure_state_lengths(0)?;
        Ok(core)
    }

    fn export_state_image(&self) -> Result<QwenStateImageV1, QwenExecutionError> {
        if self.lifecycle.is_poisoned() {
            return Err(QwenExecutionError::Poisoned);
        }
        if self.pending_speculative.is_some() {
            return Err(QwenExecutionError::Busy);
        }
        if self.committed_length == 0 {
            return Err(QwenExecutionError::InvalidRequest(
                "state image export requires a non-empty committed request".to_owned(),
            ));
        }
        self.ensure_state_lengths(self.committed_length)?;
        let mut kv_layers = BTreeMap::new();
        for (&layer, state) in &self.kv_states {
            let image = self.session.export_kv_state_image(state)?;
            validate_qwen_layer_image(
                &image,
                StateOwnerKindV1::Kv,
                layer,
                state.descriptor(),
                self.committed_length,
            )?;
            kv_layers.insert(
                layer,
                QwenKvStateImageV1 {
                    descriptor: state.descriptor(),
                    image,
                },
            );
        }
        let mut linear_layers = BTreeMap::new();
        for (&layer, state) in &self.linear_states {
            let image = self.session.export_linear_attention_state_image(state)?;
            validate_qwen_layer_image(
                &image,
                StateOwnerKindV1::LinearAttention,
                layer,
                state.descriptor(),
                self.committed_length,
            )?;
            linear_layers.insert(
                layer,
                QwenLinearStateImageV1 {
                    descriptor: state.descriptor(),
                    image,
                },
            );
        }
        Ok(QwenStateImageV1 {
            session_id: self.session.id(),
            identity: qwen_prefix_identity(&self.graph, &self.plan, &self.adapters.identity),
            committed_length: self.committed_length,
            rope_position_delta: self.rope_position_delta,
            kv_layers,
            linear_layers,
            cached_terminal_output: self.last_output.clone(),
        })
    }

    fn restore_state_image(&mut self, image: &QwenStateImageV1) -> Result<(), QwenExecutionError> {
        if self.lifecycle.is_poisoned() {
            return Err(QwenExecutionError::Poisoned);
        }
        if self.pending_speculative.is_some() {
            return Err(QwenExecutionError::Busy);
        }
        self.validate_state_image_identity(image)?;

        // Keep the fresh request's original maps and publication scalars
        // untouched until every layer has imported and snapshotted exactly.
        let mut imported_kv = BTreeMap::new();
        for (&layer, destination) in &self.kv_states {
            let entry = image.kv_layers.get(&layer).ok_or_else(|| {
                QwenExecutionError::InvalidRequest(format!(
                    "state image KV layer {layer} is absent"
                ))
            })?;
            self.session
                .import_kv_state_image(destination, &entry.image)?;
            let snapshot = self.session.kv_state_snapshot(destination)?;
            if snapshot.length() != image.committed_length {
                return Err(QwenExecutionError::StateLength {
                    layer,
                    state: "restored KV",
                    expected: image.committed_length,
                    actual: snapshot.length(),
                });
            }
            imported_kv.insert(layer, destination.clone());
        }
        let mut imported_linear = BTreeMap::new();
        for (&layer, destination) in &self.linear_states {
            let entry = image.linear_layers.get(&layer).ok_or_else(|| {
                QwenExecutionError::InvalidRequest(format!(
                    "state image linear layer {layer} is absent"
                ))
            })?;
            self.session
                .import_linear_attention_state_image(destination, &entry.image)?;
            let snapshot = self.session.linear_attention_state_snapshot(destination)?;
            if snapshot.length() != image.committed_length {
                return Err(QwenExecutionError::StateLength {
                    layer,
                    state: "restored linear",
                    expected: image.committed_length,
                    actual: snapshot.length(),
                });
            }
            imported_linear.insert(layer, destination.clone());
        }

        self.kv_states = imported_kv;
        self.linear_states = imported_linear;
        self.committed_length = image.committed_length;
        self.rope_position_delta = image.rope_position_delta;
        self.last_output = image.cached_terminal_output.clone();
        Ok(())
    }

    fn restore_checkpoint(
        &mut self,
        checkpoint: &SessionCheckpoint,
        expected_identity: &CheckpointIdentity,
    ) -> Result<(), QwenExecutionError> {
        if self.lifecycle.is_poisoned() {
            return Err(QwenExecutionError::Poisoned);
        }
        if self.pending_speculative.is_some() {
            return Err(QwenExecutionError::Busy);
        }

        // Revalidate before consuming opaque payload bytes without encoding
        // (and therefore duplicating) a potentially large checkpoint.
        checkpoint
            .validate()
            .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
        if checkpoint.header.identity != *expected_identity {
            return Err(QwenExecutionError::InvalidRequest(
                "checkpoint frontend identity differs from the restore caller".to_owned(),
            ));
        }
        let identity = &checkpoint.header.identity;
        if identity.model_lock_fingerprint != self.graph.model_fingerprint()
            || identity.plan_digest != self.plan.digest_hex()
            || identity.adapter_identity != self.adapters.identity
            || identity.kv_descriptor_digest
                != qwen_kv_descriptor_digest(
                    self.kv_states
                        .iter()
                        .map(|(layer, state)| (*layer, state.descriptor())),
                )
            || self
                .kv_states
                .values()
                .any(|state| state.descriptor().cache_encoding() != identity.kv_encoding)
        {
            return Err(QwenExecutionError::InvalidRequest(
                "checkpoint model, plan, or KV encoding/descriptor identity differs".to_owned(),
            ));
        }
        let logical_position = checkpoint.header.logical_position;
        if logical_position == 0
            || logical_position != checkpoint.header.token_count
            || logical_position > self.graph.state_capacity()
        {
            return Err(QwenExecutionError::InvalidRequest(
                "checkpoint logical position is inconsistent with token history or capacity"
                    .to_owned(),
            ));
        }
        let rope_delta = checkpoint
            .header
            .absolute_position
            .checked_sub(logical_position)
            .ok_or_else(|| {
                QwenExecutionError::InvalidRequest(
                    "checkpoint absolute position precedes logical position".to_owned(),
                )
            })?;
        let rope_position_delta = i64::try_from(rope_delta).map_err(|_| {
            QwenExecutionError::InvalidRequest(
                "checkpoint RoPE position delta exceeds i64".to_owned(),
            )
        })?;

        let layers = &checkpoint.payload.state_layers;
        let planes = &checkpoint.payload.state_planes;
        let mut layer_keys = BTreeSet::new();
        for layer in layers {
            if !layer_keys.insert((layer.owner, layer.layer_id)) {
                return Err(QwenExecutionError::InvalidRequest(
                    "checkpoint contains duplicate layer metadata".to_owned(),
                ));
            }
            if layer.published_length != logical_position {
                return Err(QwenExecutionError::InvalidRequest(
                    "checkpoint layer length differs from logical position".to_owned(),
                ));
            }
        }
        let expected_keys = self
            .kv_states
            .keys()
            .copied()
            .map(|layer| (StateOwnerKindV1::Kv, layer))
            .chain(
                self.linear_states
                    .keys()
                    .copied()
                    .map(|layer| (StateOwnerKindV1::LinearAttention, layer)),
            )
            .collect::<BTreeSet<_>>();
        if layer_keys != expected_keys {
            return Err(QwenExecutionError::InvalidRequest(
                "checkpoint layer topology differs from the fresh Qwen graph".to_owned(),
            ));
        }

        let mut kv_layers = BTreeMap::new();
        for (&layer, destination) in &self.kv_states {
            let metadata = layers
                .iter()
                .find(|metadata| {
                    metadata.owner == StateOwnerKindV1::Kv && metadata.layer_id == layer
                })
                .ok_or_else(|| {
                    QwenExecutionError::InvalidRequest(format!(
                        "checkpoint KV layer {layer} is absent"
                    ))
                })?;
            let layer_planes = planes
                .iter()
                .filter(|plane| plane.owner == StateOwnerKindV1::Kv && plane.layer_id == layer)
                .cloned()
                .collect::<Vec<_>>();
            let state_image = ExecutionStateImageV1::new(metadata.clone(), layer_planes);
            validate_qwen_layer_image(
                &state_image,
                StateOwnerKindV1::Kv,
                layer,
                destination.descriptor(),
                logical_position,
            )?;
            kv_layers.insert(
                layer,
                QwenKvStateImageV1 {
                    descriptor: destination.descriptor(),
                    image: state_image,
                },
            );
        }
        let mut linear_layers = BTreeMap::new();
        for (&layer, destination) in &self.linear_states {
            let metadata = layers
                .iter()
                .find(|metadata| {
                    metadata.owner == StateOwnerKindV1::LinearAttention
                        && metadata.layer_id == layer
                })
                .ok_or_else(|| {
                    QwenExecutionError::InvalidRequest(format!(
                        "checkpoint linear layer {layer} is absent"
                    ))
                })?;
            let layer_planes = planes
                .iter()
                .filter(|plane| {
                    plane.owner == StateOwnerKindV1::LinearAttention && plane.layer_id == layer
                })
                .cloned()
                .collect::<Vec<_>>();
            let state_image = ExecutionStateImageV1::new(metadata.clone(), layer_planes);
            validate_qwen_layer_image(
                &state_image,
                StateOwnerKindV1::LinearAttention,
                layer,
                destination.descriptor(),
                logical_position,
            )?;
            linear_layers.insert(
                layer,
                QwenLinearStateImageV1 {
                    descriptor: destination.descriptor(),
                    image: state_image,
                },
            );
        }
        self.restore_state_image(&QwenStateImageV1 {
            session_id: self.session.id(),
            identity: qwen_prefix_identity(&self.graph, &self.plan, &self.adapters.identity),
            committed_length: logical_position,
            rope_position_delta,
            kv_layers,
            linear_layers,
            cached_terminal_output: None,
        })
    }

    fn validate_state_image_identity(
        &self,
        image: &QwenStateImageV1,
    ) -> Result<(), QwenExecutionError> {
        if image.session_id != self.session.id() {
            return Err(QwenExecutionError::InvalidRequest(
                "state image belongs to a different execution session".to_owned(),
            ));
        }
        let expected_identity =
            qwen_prefix_identity(&self.graph, &self.plan, &self.adapters.identity);
        if image.identity != expected_identity {
            return Err(QwenExecutionError::InvalidRequest(
                "state image model, plan, graph, or capacity identity differs".to_owned(),
            ));
        }
        if image.committed_length == 0 || image.committed_length > self.graph.state_capacity() {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "state image length {} exceeds request capacity {}",
                image.committed_length,
                self.graph.state_capacity()
            )));
        }
        if let Some(output) = &image.cached_terminal_output {
            if output.committed_length() != image.committed_length {
                return Err(QwenExecutionError::InvalidRequest(
                    "cached state-image output length differs from committed length".to_owned(),
                ));
            }
        }
        if image.kv_layers.len() != self.kv_states.len()
            || image.linear_layers.len() != self.linear_states.len()
            || image.kv_layers.keys().ne(self.kv_states.keys())
            || image.linear_layers.keys().ne(self.linear_states.keys())
        {
            return Err(QwenExecutionError::InvalidRequest(
                "state image layer set differs from the request graph".to_owned(),
            ));
        }
        for (&layer, destination) in &self.kv_states {
            let entry = image.kv_layers.get(&layer).ok_or_else(|| {
                QwenExecutionError::InvalidRequest(format!(
                    "state image KV layer {layer} is absent"
                ))
            })?;
            if entry.descriptor != destination.descriptor() {
                return Err(QwenExecutionError::InvalidRequest(format!(
                    "state image KV layer {layer} descriptor or encoding differs"
                )));
            }
            validate_qwen_layer_image(
                &entry.image,
                StateOwnerKindV1::Kv,
                layer,
                destination.descriptor(),
                image.committed_length,
            )?;
        }
        for (&layer, destination) in &self.linear_states {
            let entry = image.linear_layers.get(&layer).ok_or_else(|| {
                QwenExecutionError::InvalidRequest(format!(
                    "state image linear layer {layer} is absent"
                ))
            })?;
            if entry.descriptor != destination.descriptor() {
                return Err(QwenExecutionError::InvalidRequest(format!(
                    "state image linear layer {layer} descriptor or capacity differs"
                )));
            }
            validate_qwen_layer_image(
                &entry.image,
                StateOwnerKindV1::LinearAttention,
                layer,
                destination.descriptor(),
                image.committed_length,
            )?;
        }
        Ok(())
    }

    fn publish_prefix(&self) -> Result<QwenPrefixStateV1, QwenExecutionError> {
        if self.lifecycle.is_poisoned() {
            return Err(QwenExecutionError::Poisoned);
        }
        if self.pending_speculative.is_some() {
            return Err(QwenExecutionError::Busy);
        }
        if self.committed_length == 0 || self.last_output.is_none() {
            return Err(QwenExecutionError::InvalidRequest(
                "prefix publication requires a completed non-empty transition".to_owned(),
            ));
        }
        self.ensure_state_lengths(self.committed_length)?;

        let identity = qwen_prefix_identity(&self.graph, &self.plan, &self.adapters.identity);
        let mut kv_states = BTreeMap::new();
        let mut linear_states = BTreeMap::new();
        let mut audit = QwenPrefixForkAuditV1::default();

        // Fork every state into a new owner before exposing the prefix. If a
        // later layer fails, these local maps drop and the source request is
        // left untouched.
        for (&layer, source) in &self.kv_states {
            let descriptor = source.descriptor();
            let (forked, fork_audit) = self.session.fork_kv_state(source, descriptor)?;
            let snapshot = self.session.kv_state_snapshot(&forked)?;
            if snapshot.length() != self.committed_length {
                return Err(QwenExecutionError::StateLength {
                    layer,
                    state: "forked kv",
                    expected: self.committed_length,
                    actual: snapshot.length(),
                });
            }
            let fallback_resident_bytes = descriptor
                .resident_bytes_per_plane()
                .and_then(|bytes| bytes.checked_mul(2))
                .ok_or_else(|| {
                    QwenExecutionError::InvalidRequest(
                        "KV prefix resident-byte footprint overflowed".to_owned(),
                    )
                })?;
            audit.add(
                fork_audit,
                false,
                snapshot.physical_memory(),
                fallback_resident_bytes,
            )?;
            kv_states.insert(layer, forked);
        }
        for (&layer, source) in &self.linear_states {
            let descriptor = source.descriptor();
            let (forked, fork_audit) = self
                .session
                .fork_linear_attention_state(source, descriptor)?;
            let snapshot = self.session.linear_attention_state_snapshot(&forked)?;
            if snapshot.length() != self.committed_length {
                return Err(QwenExecutionError::StateLength {
                    layer,
                    state: "forked linear",
                    expected: self.committed_length,
                    actual: snapshot.length(),
                });
            }
            audit.add(fork_audit, true, None, 0)?;
            linear_states.insert(layer, forked);
        }

        Ok(QwenPrefixStateV1 {
            inner: Arc::new(QwenPrefixStateInner {
                session: Arc::clone(&self.session),
                identity,
                committed_length: self.committed_length,
                rope_position_delta: self.rope_position_delta,
                kv_states,
                linear_states,
                cached_terminal_output: self
                    .last_output
                    .clone()
                    .expect("checked cached terminal output"),
                fork_audit: audit,
            }),
        })
    }

    fn install_prefix(&mut self, prefix: &QwenPrefixStateV1) -> Result<(), QwenExecutionError> {
        if self.lifecycle.is_poisoned() {
            return Err(QwenExecutionError::Poisoned);
        }
        if self.pending_speculative.is_some() {
            return Err(QwenExecutionError::Busy);
        }
        let expected_identity =
            qwen_prefix_identity(&self.graph, &self.plan, &self.adapters.identity);
        if prefix.inner.session.id() != self.session.id() {
            return Err(QwenExecutionError::InvalidRequest(
                "prefix state belongs to a different execution session".to_owned(),
            ));
        }
        if prefix.inner.identity.model_fingerprint != expected_identity.model_fingerprint
            || prefix.inner.identity.plan_digest != expected_identity.plan_digest
            || prefix.inner.identity.graph_semantics_digest
                != expected_identity.graph_semantics_digest
            || prefix.inner.identity.adapter_identity != expected_identity.adapter_identity
            || prefix.inner.identity.is_mtp != expected_identity.is_mtp
            || prefix.inner.identity.is_multimodal != expected_identity.is_multimodal
        {
            return Err(QwenExecutionError::InvalidRequest(
                "prefix graph/model/plan state identity differs from the request graph".to_owned(),
            ));
        }
        if prefix.committed_length() == 0 || prefix.committed_length() > self.graph.state_capacity()
        {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "prefix length {} exceeds request capacity {}",
                prefix.committed_length(),
                self.graph.state_capacity()
            )));
        }
        if prefix.cached_terminal_output().committed_length() != prefix.committed_length() {
            return Err(QwenExecutionError::InvalidRequest(
                "cached prefix terminal output length differs from prefix state".to_owned(),
            ));
        }
        if prefix.inner.kv_states.len() != self.kv_states.len()
            || prefix.inner.linear_states.len() != self.linear_states.len()
            || prefix.inner.kv_states.keys().ne(self.kv_states.keys())
            || prefix
                .inner
                .linear_states
                .keys()
                .ne(self.linear_states.keys())
        {
            return Err(QwenExecutionError::InvalidRequest(
                "prefix state layer set differs from the request graph".to_owned(),
            ));
        }

        let mut forked_kv = BTreeMap::new();
        let mut forked_linear = BTreeMap::new();
        // Keep old request states in place until every destination fork and
        // snapshot has passed validation.
        for (&layer, destination) in &self.kv_states {
            let source = prefix.inner.kv_states.get(&layer).ok_or_else(|| {
                QwenExecutionError::InvalidRequest(format!("prefix KV layer {layer} is absent"))
            })?;
            if destination.descriptor().layer_id() != source.descriptor().layer_id()
                || destination.descriptor().layout() != source.descriptor().layout()
                || destination.descriptor().cache_encoding() != source.descriptor().cache_encoding()
                || destination.descriptor().static_fp8_scales()
                    != source.descriptor().static_fp8_scales()
            {
                return Err(QwenExecutionError::InvalidRequest(format!(
                    "prefix KV layer {layer} descriptor differs"
                )));
            }
            let (forked, _) = self
                .session
                .fork_kv_state(source, destination.descriptor())?;
            let snapshot = self.session.kv_state_snapshot(&forked)?;
            if snapshot.length() != prefix.committed_length() {
                return Err(QwenExecutionError::StateLength {
                    layer,
                    state: "installed kv",
                    expected: prefix.committed_length(),
                    actual: snapshot.length(),
                });
            }
            forked_kv.insert(layer, forked);
        }
        for (&layer, destination) in &self.linear_states {
            let source = prefix.inner.linear_states.get(&layer).ok_or_else(|| {
                QwenExecutionError::InvalidRequest(format!("prefix linear layer {layer} is absent"))
            })?;
            if destination.descriptor().layer_id() != source.descriptor().layer_id()
                || destination.descriptor().layout() != source.descriptor().layout()
            {
                return Err(QwenExecutionError::InvalidRequest(format!(
                    "prefix linear layer {layer} descriptor differs"
                )));
            }
            let (forked, _) = self
                .session
                .fork_linear_attention_state(source, destination.descriptor())?;
            let snapshot = self.session.linear_attention_state_snapshot(&forked)?;
            if snapshot.length() != prefix.committed_length() {
                return Err(QwenExecutionError::StateLength {
                    layer,
                    state: "installed linear",
                    expected: prefix.committed_length(),
                    actual: snapshot.length(),
                });
            }
            forked_linear.insert(layer, forked);
        }

        self.kv_states = forked_kv;
        self.linear_states = forked_linear;
        self.committed_length = prefix.committed_length();
        self.rope_position_delta = prefix.rope_position_delta();
        self.last_output = Some(prefix.cached_terminal_output().clone());
        self.ensure_state_lengths(self.committed_length)
    }

    fn decode_continuation(
        &mut self,
        suffix: &[i32],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        if self.lifecycle.is_poisoned() {
            return Err(QwenExecutionError::Poisoned);
        }
        if suffix.is_empty() {
            return self.last_output.clone().ok_or_else(|| {
                QwenExecutionError::InvalidRequest(
                    "empty continuation has no cached terminal output".to_owned(),
                )
            });
        }
        if self.committed_length == 0 {
            return Err(QwenExecutionError::InvalidRequest(
                "continuation requires an installed non-empty prefix".to_owned(),
            ));
        }
        validate_input_token_ids(suffix)?;
        let suffix_len = u64::try_from(suffix.len()).map_err(|_| {
            QwenExecutionError::InvalidRequest("continuation length does not fit u64".to_owned())
        })?;
        let end = self
            .committed_length
            .checked_add(suffix_len)
            .ok_or_else(|| {
                QwenExecutionError::InvalidRequest("continuation length overflowed u64".to_owned())
            })?;
        if end > self.graph.state_capacity() {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "continuation end {end} exceeds request capacity {}",
                self.graph.state_capacity()
            )));
        }
        let chunk_capacity = usize::try_from(self.graph.token_count()).map_err(|_| {
            QwenExecutionError::InvalidGraph("graph token count does not fit usize".to_owned())
        })?;
        if chunk_capacity == 0 {
            return Err(QwenExecutionError::InvalidGraph(
                "graph token count is zero".to_owned(),
            ));
        }
        let chunk_count = suffix.len().div_ceil(chunk_capacity);
        let mut final_output = None;
        for (index, chunk) in suffix.chunks(chunk_capacity).enumerate() {
            let final_chunk = index + 1 == chunk_count;
            let output = self.run_transition(
                chunk,
                AttentionPreprocessPositionMode::DecodeContinuation,
                false,
                false,
                false,
                false,
                None,
                None,
                final_chunk,
                None,
            )?;
            if final_chunk {
                final_output = Some(output);
            }
        }
        self.prefill_chunk_count = u64::try_from(chunk_count).map_err(|_| {
            QwenExecutionError::InvalidRequest(
                "continuation chunk count does not fit u64".to_owned(),
            )
        })?;
        final_output.ok_or_else(|| {
            QwenExecutionError::InvalidRequest(
                "continuation produced no terminal output".to_owned(),
            )
        })
    }

    #[cfg(test)]
    fn provision<S: QwenProvisionSource>(
        session: Arc<ExecutionSession>,
        graph: QwenGraph,
        plan: WeightLoadPlan,
        completion_timeout: Duration,
        source: &S,
    ) -> Result<Self, QwenExecutionError> {
        let layout = validate_graph_plan(&graph, &plan)?;
        preflight_device_memory(session.as_ref(), &graph, &layout, false)?;
        let queue = session.create_queue()?;
        let tensors = allocate_tensors(&session, &graph)?;

        let mut uploaded = BTreeSet::new();
        for binding in graph.weight_bindings() {
            if !uploaded.insert(binding.tensor_name().to_owned()) {
                return Err(QwenExecutionError::InvalidGraph(format!(
                    "required weight is represented more than once: {}",
                    binding.tensor_name()
                )));
            }
            let tensor_id = *layout
                ._weight_tensor_ids
                .get(binding.tensor_name())
                .ok_or_else(|| {
                    QwenExecutionError::InvalidGraph(format!(
                        "required weight tensor is absent: {}",
                        binding.tensor_name()
                    ))
                })?;
            let allocation = tensors.get(tensor_id).ok_or_else(|| {
                QwenExecutionError::InvalidGraph("weight tensor allocation is absent".to_owned())
            })?;
            let destination = allocation.buffer.range(
                allocation.graph_view.byte_offset(),
                resident_weight_bytes(&allocation.graph_view)?,
            )?;
            source.upload_weight_for_resident_dtype(
                &plan,
                binding,
                session.as_ref(),
                &queue,
                destination,
                allocation.graph_view.dtype(),
                completion_timeout,
            )?;
        }
        if uploaded.len() != graph.weight_bindings().len() {
            return Err(QwenExecutionError::InvalidGraph(
                "required weight identities are not one-to-one".to_owned(),
            ));
        }

        let scales = provision_scales(
            source,
            &session,
            &queue,
            &graph,
            &tensors,
            &layout.scales,
            completion_timeout,
        )?;
        let (kv_states, linear_states) = create_states(&session, &graph)?;
        let execution_plan = qwen_prepared_execution_plan(&graph)?;

        let core = Self {
            session,
            graph,
            execution_plan,
            plan,
            queue,
            tensors,
            tensor_ids: layout.tensor_ids,
            dynamic_tensors: layout.dynamic_tensors,
            kv_states,
            linear_states,
            scales,
            completion_timeout,
            audit: Mutex::new(ExecutionAuditAccumulator::new(1)),
            prepared_semantics: PreparedSemanticCache::default(),
            lifecycle: ExecutionTransaction::new(),
            committed_length: 0,
            prefill_chunk_count: 0,
            rope_position_delta: 0,
            last_output: None,
            pending_speculative: None,
            adapters: QwenAdapterRuntime::disabled(),
        };
        core.ensure_state_lengths(0)?;
        Ok(core)
    }

    fn prefill(&mut self, token_ids: &[i32]) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.prefill_impl(token_ids, false, false, None, None, None)
    }

    fn prefill_with_absolute_positions(
        &mut self,
        token_ids: &[i32],
        positions: &[u64],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        if self.graph.position_payload_mode()
            != crate::AttentionPreprocessPositionPayloadModeV1::Explicit
            || positions.len() != token_ids.len()
            || token_ids.is_empty()
            || u64::try_from(token_ids.len()).ok().is_none_or(|count| {
                count > self.graph.state_capacity() || count > self.graph.token_count()
            })
        {
            return Err(QwenExecutionError::InvalidRequest(
                "explicit-position prefill graph or input length is invalid".to_owned(),
            ));
        }
        self.run_transition_with_positions(
            token_ids,
            AttentionPreprocessPositionMode::Prefill,
            false,
            false,
            false,
            false,
            None,
            None,
            true,
            None,
            Some(positions),
        )
    }

    fn prefill_with_last_logits(
        &mut self,
        token_ids: &[i32],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.prefill_impl(token_ids, true, false, None, None, None)
    }

    fn prefill_with_device_selector(
        &mut self,
        token_ids: &[i32],
        selector: &DeviceTokenSelectorRequestV1,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        if self.graph.is_mtp() {
            return Err(QwenExecutionError::InvalidRequest(
                "device token selector is unsupported for MTP graphs".to_owned(),
            ));
        }
        self.prefill_impl(token_ids, false, false, None, None, Some(selector))
    }

    fn prefill_with_mtp_state(
        &mut self,
        token_ids: &[i32],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.prefill_impl(token_ids, true, true, None, None, None)
    }

    fn prefill_with_embeddings(
        &mut self,
        token_ids: &[i32],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.prefill_embeddings_impl(token_ids)
    }

    fn prefill_embeddings_impl(
        &mut self,
        token_ids: &[i32],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        if self.lifecycle.is_poisoned() {
            return Err(QwenExecutionError::Poisoned);
        }
        if self.committed_length != 0 {
            return Err(QwenExecutionError::InvalidRequest(
                "prefill is only valid before the first committed transition".to_owned(),
            ));
        }
        let chunk_capacity = usize::try_from(self.graph.token_count()).map_err(|_| {
            QwenExecutionError::InvalidGraph("graph token count does not fit usize".to_owned())
        })?;
        if chunk_capacity == 0 {
            return Err(QwenExecutionError::InvalidGraph(
                "embedding prefill graph token capacity is zero".to_owned(),
            ));
        }
        if token_ids.is_empty() {
            return Err(QwenExecutionError::InvalidRequest(
                "prefill requires at least one token".to_owned(),
            ));
        }
        if u64::try_from(token_ids.len())
            .ok()
            .is_none_or(|tokens| tokens > self.graph.state_capacity())
        {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "prefill token count {} exceeds request state capacity {}",
                token_ids.len(),
                self.graph.state_capacity()
            )));
        }

        let chunk_count = token_ids.len().div_ceil(chunk_capacity);
        let mut all_embeddings = Vec::new();
        let expected_words = token_ids.len().checked_mul(2_560).ok_or_else(|| {
            QwenExecutionError::InvalidRequest("embedding output size overflowed".to_owned())
        })?;
        all_embeddings.try_reserve(expected_words).map_err(|_| {
            QwenExecutionError::InvalidRequest(
                "embedding output allocation is too large".to_owned(),
            )
        })?;
        let mut final_output = None;
        for (index, chunk) in token_ids.chunks(chunk_capacity).enumerate() {
            let output = self.run_transition(
                chunk,
                if index == 0 {
                    AttentionPreprocessPositionMode::Prefill
                } else {
                    AttentionPreprocessPositionMode::DecodeContinuation
                },
                false,
                false,
                false,
                false,
                None,
                None,
                false,
                None,
            )?;
            let rows =
                self.read_final_hidden_states(u64::try_from(chunk.len()).map_err(|_| {
                    QwenExecutionError::InvalidRequest(
                        "embedding chunk token count does not fit u64".to_owned(),
                    )
                })?)?;
            all_embeddings.extend(rows);
            final_output = Some(output);
        }
        let mut output = final_output.ok_or_else(|| {
            QwenExecutionError::InvalidRequest("embedding prefill produced no output".to_owned())
        })?;
        output.embeddings_bf16 = Some(all_embeddings);
        self.last_output = Some(output.clone());
        self.prefill_chunk_count = u64::try_from(chunk_count).map_err(|_| {
            QwenExecutionError::InvalidRequest("prefill chunk count does not fit u64".to_owned())
        })?;
        Ok(output)
    }

    fn prefill_multimodal(
        &mut self,
        token_ids: &[i32],
        embeddings_bf16: &[u16],
        positions: &[[i32; 3]],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.prefill_impl(
            token_ids,
            true,
            false,
            None,
            Some((embeddings_bf16, positions)),
            None,
        )
    }

    fn prefill_mtp(
        &mut self,
        token_id: i32,
        target_hidden_bf16: &[u16],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.prefill_impl(
            &[token_id],
            true,
            true,
            Some(target_hidden_bf16),
            None,
            None,
        )
    }

    fn prefill_impl(
        &mut self,
        token_ids: &[i32],
        include_last_logits: bool,
        include_hidden_states: bool,
        target_hidden_bf16: Option<&[u16]>,
        multimodal: Option<(&[u16], &[[i32; 3]])>,
        device_selector: Option<&DeviceTokenSelectorRequestV1>,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        if self.lifecycle.is_poisoned() {
            return Err(QwenExecutionError::Poisoned);
        }
        if self.committed_length != 0 {
            return Err(QwenExecutionError::InvalidRequest(
                "prefill is only valid before the first committed transition".to_owned(),
            ));
        }
        let chunk_capacity = usize::try_from(self.graph.token_count()).map_err(|_| {
            QwenExecutionError::InvalidGraph("graph token count does not fit usize".to_owned())
        })?;
        if token_ids.is_empty() {
            return Err(QwenExecutionError::InvalidRequest(
                "prefill requires at least one token".to_owned(),
            ));
        }
        if u64::try_from(token_ids.len())
            .ok()
            .is_none_or(|tokens| tokens > self.graph.state_capacity())
        {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "prefill token count {} exceeds request state capacity {}",
                token_ids.len(),
                self.graph.state_capacity()
            )));
        }
        if token_ids.len() <= chunk_capacity {
            let output = self.run_transition(
                token_ids,
                AttentionPreprocessPositionMode::Prefill,
                include_last_logits,
                false,
                false,
                include_hidden_states,
                target_hidden_bf16,
                multimodal,
                true,
                device_selector,
            )?;
            self.prefill_chunk_count = 1;
            return Ok(output);
        }
        if multimodal.is_some() || target_hidden_bf16.is_some() {
            return Err(QwenExecutionError::InvalidRequest(
                "chunked prefill currently requires text input without an MTP component row"
                    .to_owned(),
            ));
        }

        validate_input_token_ids(token_ids)?;
        let chunk_count = token_ids.len().div_ceil(chunk_capacity);
        let mut hidden_states = include_hidden_states.then(Vec::new);
        let mut final_output = None;
        for (index, chunk) in token_ids.chunks(chunk_capacity).enumerate() {
            let final_chunk = index + 1 == chunk_count;
            let mut output = self.run_transition(
                chunk,
                if index == 0 {
                    AttentionPreprocessPositionMode::Prefill
                } else {
                    AttentionPreprocessPositionMode::DecodeContinuation
                },
                final_chunk && include_last_logits,
                false,
                false,
                include_hidden_states,
                None,
                None,
                final_chunk,
                final_chunk.then_some(device_selector).flatten(),
            )?;
            if let Some(all_hidden) = &mut hidden_states {
                all_hidden.extend(output.hidden_states_bf16.take().ok_or_else(|| {
                    QwenExecutionError::InvalidRequest(
                        "chunked prefill omitted requested hidden states".to_owned(),
                    )
                })?);
            }
            if final_chunk {
                final_output = Some(output);
            }
        }
        let mut output = final_output.ok_or_else(|| {
            QwenExecutionError::InvalidRequest(
                "chunked prefill produced no final output".to_owned(),
            )
        })?;
        output.hidden_states_bf16 = hidden_states;
        self.prefill_chunk_count = u64::try_from(chunk_count).map_err(|_| {
            QwenExecutionError::InvalidRequest("prefill chunk count does not fit u64".to_owned())
        })?;
        Ok(output)
    }

    fn decode(&mut self, token_id: i32) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.decode_impl(token_id, false, false, false, None, None)
    }

    fn decode_with_last_logits(
        &mut self,
        token_id: i32,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.decode_impl(token_id, true, false, false, None, None)
    }

    fn decode_with_device_selector(
        &mut self,
        token_id: i32,
        selector: &DeviceTokenSelectorRequestV1,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        if self.graph.is_mtp() {
            return Err(QwenExecutionError::InvalidRequest(
                "device token selector is unsupported for MTP graphs".to_owned(),
            ));
        }
        self.decode_impl(token_id, false, false, false, None, Some(selector))
    }

    fn decode_with_mtp_state(
        &mut self,
        token_id: i32,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.decode_impl(token_id, true, false, true, None, None)
    }

    fn decode_with_mtp_state_and_logits(
        &mut self,
        token_id: i32,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.decode_impl(token_id, false, true, true, None, None)
    }

    fn decode_block_with_mtp_state(
        &mut self,
        token_ids: &[i32],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        if self.lifecycle.is_poisoned() {
            return Err(QwenExecutionError::Poisoned);
        }
        if self.committed_length == 0 {
            return Err(QwenExecutionError::InvalidRequest(
                "decode block requires a committed prefill".to_owned(),
            ));
        }
        let maximum = usize::try_from(self.graph.token_count()).map_err(|_| {
            QwenExecutionError::InvalidGraph("graph token count does not fit usize".to_owned())
        })?;
        if token_ids.is_empty() || token_ids.len() > maximum {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "decode block token count is {}, graph capacity is {maximum}",
                token_ids.len()
            )));
        }
        let start_length = self.committed_length;
        let output = self.run_transition(
            token_ids,
            AttentionPreprocessPositionMode::DecodeContinuation,
            false,
            false,
            true,
            true,
            None,
            None,
            true,
            None,
        )?;
        self.pending_speculative = Some(PendingSpeculativeBlock {
            start_length,
            token_ids: token_ids.to_vec(),
        });
        Ok(output)
    }

    fn decode_block_with_mtp_state_and_logits(
        &mut self,
        token_ids: &[i32],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        if self.lifecycle.is_poisoned() {
            return Err(QwenExecutionError::Poisoned);
        }
        if self.committed_length == 0 {
            return Err(QwenExecutionError::InvalidRequest(
                "decode block requires a committed prefill".to_owned(),
            ));
        }
        let maximum = usize::try_from(self.graph.token_count()).map_err(|_| {
            QwenExecutionError::InvalidGraph("graph token count does not fit usize".to_owned())
        })?;
        if token_ids.is_empty() || token_ids.len() > maximum {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "decode block token count is {}, graph capacity is {maximum}",
                token_ids.len()
            )));
        }
        let start_length = self.committed_length;
        let output = self.run_transition(
            token_ids,
            AttentionPreprocessPositionMode::DecodeContinuation,
            false,
            true,
            true,
            true,
            None,
            None,
            true,
            None,
        )?;
        self.pending_speculative = Some(PendingSpeculativeBlock {
            start_length,
            token_ids: token_ids.to_vec(),
        });
        Ok(output)
    }

    fn resolve_decode_block(
        &mut self,
        committed_input_rows: usize,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        let pending = self.pending_speculative.take().ok_or_else(|| {
            QwenExecutionError::InvalidRequest(
                "no speculative decode block is pending resolution".to_owned(),
            )
        })?;
        if committed_input_rows == 0 || committed_input_rows > pending.token_ids.len() {
            self.pending_speculative = Some(pending);
            return Err(QwenExecutionError::InvalidRequest(format!(
                "committed speculative input rows {committed_input_rows} are outside 1..={}",
                self.pending_speculative
                    .as_ref()
                    .map_or(0, |block| block.token_ids.len())
            )));
        }
        if committed_input_rows == pending.token_ids.len() {
            return self.last_output.clone().ok_or_else(|| {
                QwenExecutionError::InvalidRequest(
                    "resolved speculative block has no completed output".to_owned(),
                )
            });
        }
        let expected_length = self.committed_length;
        if let Err(error) = self.rewind_last_transition(expected_length, pending.start_length) {
            self.lifecycle.cancel();
            return Err(error);
        }
        self.committed_length = pending.start_length;
        self.last_output = None;
        self.run_transition(
            &pending.token_ids[..committed_input_rows],
            AttentionPreprocessPositionMode::DecodeContinuation,
            false,
            false,
            false,
            true,
            None,
            None,
            true,
            None,
        )
    }

    fn rewind_last_decode_transition(&mut self) -> Result<(), QwenExecutionError> {
        if self.lifecycle.is_poisoned() {
            return Err(QwenExecutionError::Poisoned);
        }
        if self.pending_speculative.is_some() {
            return Err(QwenExecutionError::Busy);
        }
        let rewind_length = self.committed_length.checked_sub(1).ok_or_else(|| {
            QwenExecutionError::InvalidRequest("cannot rewind an empty request".to_owned())
        })?;
        if let Err(error) = self.rewind_last_transition(self.committed_length, rewind_length) {
            self.lifecycle.cancel();
            return Err(error);
        }
        self.committed_length = rewind_length;
        self.last_output = None;
        Ok(())
    }

    fn kv_payload_bytes_for_evidence(
        &self,
    ) -> Result<Vec<QwenKvPayloadEvidence>, QwenExecutionError> {
        let mut payloads = Vec::with_capacity(self.kv_states.len());
        for (&layer, state) in &self.kv_states {
            let snapshot = self.session.kv_state_snapshot(state)?;
            if snapshot.length() != self.committed_length {
                return Err(QwenExecutionError::StateLength {
                    layer,
                    state: "kv",
                    expected: self.committed_length,
                    actual: snapshot.length(),
                });
            }
            let resident = state
                .descriptor()
                .resident_bytes_per_plane()
                .ok_or_else(|| {
                    QwenExecutionError::InvalidGraph("KV resident byte count overflowed".to_owned())
                })?;
            let bytes_per_token = resident / state.capacity();
            let semantic_bytes =
                bytes_per_token
                    .checked_mul(snapshot.length())
                    .ok_or_else(|| {
                        QwenExecutionError::InvalidGraph(
                            "KV semantic byte count overflowed".to_owned(),
                        )
                    })?;
            let length = usize::try_from(semantic_bytes).map_err(|_| {
                QwenExecutionError::InvalidGraph(
                    "KV semantic byte count does not fit usize".to_owned(),
                )
            })?;
            let mut key = vec![0_u8; length];
            let mut value = vec![0_u8; length];
            self.session.readback_kv_state(state, 0, 0, &mut key)?;
            self.session.readback_kv_state(state, 1, 0, &mut value)?;
            payloads.push((layer, key, value));
        }
        Ok(payloads)
    }

    fn decode_mtp(
        &mut self,
        token_id: i32,
        target_hidden_bf16: &[u16],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.decode_impl(token_id, true, false, true, Some(target_hidden_bf16), None)
    }

    fn decode_impl(
        &mut self,
        token_id: i32,
        include_last_logits: bool,
        include_all_logits_bf16: bool,
        include_hidden_states: bool,
        target_hidden_bf16: Option<&[u16]>,
        device_selector: Option<&DeviceTokenSelectorRequestV1>,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        if self.lifecycle.is_poisoned() {
            return Err(QwenExecutionError::Poisoned);
        }
        if self.committed_length == 0 {
            return Err(QwenExecutionError::InvalidRequest(
                "decode requires a committed prefill".to_owned(),
            ));
        }
        self.run_transition(
            &[token_id],
            AttentionPreprocessPositionMode::DecodeContinuation,
            include_last_logits,
            include_all_logits_bf16,
            false,
            include_hidden_states,
            target_hidden_bf16,
            None,
            true,
            device_selector,
        )
    }

    fn set_rope_position_delta(&mut self, delta: i64) -> Result<(), QwenExecutionError> {
        if delta < 0 {
            return Err(QwenExecutionError::InvalidRequest(
                "Qwen RoPE position delta must be non-negative".to_owned(),
            ));
        }
        if self.lifecycle.is_poisoned() {
            return Err(QwenExecutionError::Poisoned);
        }
        self.rope_position_delta = delta;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_transition(
        &mut self,
        token_ids: &[i32],
        position_mode: AttentionPreprocessPositionMode,
        include_last_logits: bool,
        include_all_logits_bf16: bool,
        force_all_terminal_rows: bool,
        include_hidden_states: bool,
        target_hidden_bf16: Option<&[u16]>,
        multimodal: Option<(&[u16], &[[i32; 3]])>,
        emit_terminal: bool,
        device_selector: Option<&DeviceTokenSelectorRequestV1>,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.run_transition_with_positions(
            token_ids,
            position_mode,
            include_last_logits,
            include_all_logits_bf16,
            force_all_terminal_rows,
            include_hidden_states,
            target_hidden_bf16,
            multimodal,
            emit_terminal,
            device_selector,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_transition_with_positions(
        &mut self,
        token_ids: &[i32],
        position_mode: AttentionPreprocessPositionMode,
        include_last_logits: bool,
        include_all_logits_bf16: bool,
        force_all_terminal_rows: bool,
        include_hidden_states: bool,
        target_hidden_bf16: Option<&[u16]>,
        multimodal: Option<(&[u16], &[[i32; 3]])>,
        emit_terminal: bool,
        device_selector: Option<&DeviceTokenSelectorRequestV1>,
        explicit_positions: Option<&[u64]>,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        if self.pending_speculative.is_some() {
            return Err(QwenExecutionError::Busy);
        }
        let token_count = u64::try_from(token_ids.len()).map_err(|_| {
            QwenExecutionError::InvalidRequest("token count does not fit u64".to_owned())
        })?;
        if token_count == 0 {
            return Err(QwenExecutionError::InvalidRequest(
                "a Qwen transition requires at least one token".to_owned(),
            ));
        }
        let start_position = self.committed_length;
        match position_mode {
            AttentionPreprocessPositionMode::Prefill if start_position != 0 => {
                return Err(QwenExecutionError::InvalidRequest(
                    "prefill must start at position zero".to_owned(),
                ));
            }
            AttentionPreprocessPositionMode::DecodeContinuation if start_position == 0 => {
                return Err(QwenExecutionError::InvalidRequest(
                    "decode continuation requires a non-zero position".to_owned(),
                ));
            }
            _ => {}
        }
        let expected_length = start_position.checked_add(token_count).ok_or_else(|| {
            QwenExecutionError::InvalidRequest("committed length overflowed u64".to_owned())
        })?;
        if expected_length > self.graph.state_capacity() {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "transition end {expected_length} exceeds request capacity {}",
                self.graph.state_capacity()
            )));
        }
        validate_input_token_ids(token_ids)?;
        let expects_multimodal = self.graph.is_multimodal();
        if (start_position == 0 && expects_multimodal) != multimodal.is_some()
            || (!expects_multimodal && multimodal.is_some())
        {
            return Err(QwenExecutionError::InvalidRequest(
                "multimodal graph/input mode differs".to_owned(),
            ));
        }
        if explicit_positions.is_some_and(|positions| {
            positions.len() != token_ids.len()
                || positions
                    .iter()
                    .any(|position| *position >= QWEN_RUNTIME_MAX_CONTEXT_TOKENS)
        }) {
            return Err(QwenExecutionError::InvalidRequest(
                "explicit Qwen position payload differs from token input or runtime range"
                    .to_owned(),
            ));
        }
        let rope_start_position = if let Some(positions) = explicit_positions {
            *positions.first().ok_or_else(|| {
                QwenExecutionError::InvalidRequest("explicit position payload is empty".to_owned())
            })?
        } else if start_position == 0 {
            0
        } else {
            let committed = i64::try_from(start_position).map_err(|_| {
                QwenExecutionError::InvalidRequest("committed position does not fit i64".to_owned())
            })?;
            u64::try_from(
                committed
                    .checked_add(self.rope_position_delta)
                    .ok_or_else(|| {
                        QwenExecutionError::InvalidRequest(
                            "decode mRoPE position overflowed".to_owned(),
                        )
                    })?,
            )
            .map_err(|_| {
                QwenExecutionError::InvalidRequest("decode mRoPE position is negative".to_owned())
            })?
        };
        let next_rope_delta = multimodal
            .map(|(_, positions)| {
                let maximum = positions
                    .iter()
                    .flat_map(|position| position.iter())
                    .copied()
                    .max()
                    .ok_or_else(|| {
                        QwenExecutionError::InvalidRequest(
                            "multimodal positions are empty".to_owned(),
                        )
                    })?;
                i64::from(maximum)
                    .checked_add(1)
                    .and_then(|next| next.checked_sub(i64::try_from(expected_length).ok()?))
                    .ok_or_else(|| {
                        QwenExecutionError::InvalidRequest(
                            "multimodal position delta overflowed".to_owned(),
                        )
                    })
            })
            .transpose()?;

        let terminal_rows = if device_selector.is_some() {
            // The selector consumes exactly the final BF16 row.  This override
            // also covers small/chunked graphs whose legacy path retains all
            // rows for Argmax/readback.
            TerminalOutputRows::Last
        } else if self.graph.token_count() < TERMINAL_ROW_MIN_TOKENS
            || include_all_logits_bf16
            || force_all_terminal_rows
            || self.graph.is_mtp()
        {
            TerminalOutputRows::All
        } else {
            TerminalOutputRows::Last
        };
        if terminal_rows == TerminalOutputRows::All
            && self.graph.token_count() >= TERMINAL_ROW_MIN_TOKENS
            && !self.graph.is_mtp()
        {
            self.ensure_terminal_output_capacity(token_count)?;
        }

        // A stale state before any new upload/dispatch is an admission error,
        // not a graph-wide partial mutation. Once the guard begins, every
        // error path poisons the request.
        self.ensure_state_lengths(start_position)?;
        let mut guard = self.lifecycle.begin()?;

        self.upload_runtime_inputs(
            token_ids,
            rope_start_position,
            token_count,
            target_hidden_bf16,
            multimodal,
            explicit_positions,
        )?;
        let output = self.lower_graph(
            token_count,
            start_position,
            rope_start_position,
            expected_length,
            position_mode,
            terminal_rows,
            emit_terminal,
            device_selector,
        )?;
        let last_logits = include_last_logits
            .then(|| self.read_last_logits(token_count, terminal_rows))
            .transpose()?;
        let logits_bf16 = include_all_logits_bf16
            .then(|| self.read_logits_bf16(token_count))
            .transpose()?;
        let hidden_states_bf16 = include_hidden_states
            .then(|| self.read_hidden_states(token_count))
            .transpose()?;
        self.ensure_state_lengths(expected_length)?;

        let output = QwenExecutionOutput {
            token_ids: output.token_ids,
            last_logits,
            selection: output.selection,
            logits_bf16,
            hidden_states_bf16,
            embeddings_bf16: None,
            committed_length: expected_length,
        };
        guard.commit()?;
        self.committed_length = expected_length;
        if let Some(delta) = next_rope_delta {
            self.rope_position_delta = delta;
        }
        self.last_output = Some(output.clone());
        Ok(output)
    }

    fn rewind_last_transition(
        &self,
        expected_length: u64,
        rewind_length: u64,
    ) -> Result<(), QwenExecutionError> {
        for state in self.linear_states.values() {
            self.session.rewind_last_linear_attention_transition(
                state,
                expected_length,
                rewind_length,
            )?;
        }
        for state in self.kv_states.values() {
            self.session
                .rewind_last_kv_state_transition(state, expected_length, rewind_length)?;
        }
        self.ensure_state_lengths(rewind_length)
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_graph(
        &self,
        token_count: u64,
        start_position: u64,
        rope_start_position: u64,
        expected_length: u64,
        position_mode: AttentionPreprocessPositionMode,
        terminal_rows: TerminalOutputRows,
        emit_terminal: bool,
        device_selector: Option<&DeviceTokenSelectorRequestV1>,
    ) -> Result<TerminalSelection, QwenExecutionError> {
        let plan = self.execution_plan.clone();
        let transition = PreparedTransition::new(token_count, start_position, 0, start_position)?;
        if transition.expected_length() != expected_length {
            return Err(QwenExecutionError::InvalidRequest(
                "prepared transition length differs from the Qwen admission result".to_owned(),
            ));
        }
        let mut argmax: Option<TerminalSelection> = None;
        let mut pending = ExecutionSegment::profiled(self.completion_timeout);
        plan.execute(transition, |planned, transition| {
            let node = planned.operation();
            (|| -> Result<(), QwenExecutionError> {
                match node.kind() {
                    QwenGraphNodeKind::Semantic(_) => {
                        if !emit_terminal
                            && (matches!(
                                node.kind(),
                                QwenGraphNodeKind::Semantic(SemanticOpKind::Argmax)
                            ) || self.is_terminal_projection(node)?)
                        {
                            return Ok(());
                        }
                        let output = self.execute_semantic(
                            node,
                            transition.token_count(),
                            planned.boundary_after(),
                            terminal_rows,
                            device_selector,
                            &mut pending,
                        )?;
                        if let Some(output) = output {
                            if argmax.replace(output).is_some() {
                                return Err(QwenExecutionError::InvalidGraph(
                                    "graph contains more than one argmax result".to_owned(),
                                ));
                            }
                        }
                    }
                    QwenGraphNodeKind::AttentionScaleMaterialization {
                        heads, head_dim, ..
                    } => self.validate_cached_scale(node, heads, head_dim)?,
                    QwenGraphNodeKind::AttentionPreprocess {
                        layer,
                        q_heads,
                        kv_heads,
                        head_dim,
                        ..
                    } => self.execute_attention_preprocess(
                        node,
                        AttentionPreprocessExecution {
                            layer,
                            q_heads,
                            kv_heads,
                            head_dim,
                            token_count: transition.token_count(),
                            start_position: if self.graph.position_payload_mode()
                                == crate::AttentionPreprocessPositionPayloadModeV1::Explicit
                                && matches!(position_mode, AttentionPreprocessPositionMode::Prefill)
                            {
                                0
                            } else {
                                rope_start_position
                            },
                            position_mode,
                        },
                        &mut pending,
                    )?,
                    QwenGraphNodeKind::MultimodalEmbeddingSelect => self
                        .execute_multimodal_embedding_select(
                            node,
                            transition.token_count(),
                            position_mode,
                            &mut pending,
                        )?,
                    QwenGraphNodeKind::FullKvAppend { layer, state } => self.execute_kv_append(
                        node,
                        layer,
                        state,
                        StatefulExecution {
                            token_count: transition.token_count(),
                            start_position: transition.start_position(),
                            expected_length: transition.expected_length(),
                        },
                        planned.boundary_after(),
                        &mut pending,
                    )?,
                    QwenGraphNodeKind::FullCausalAttention { layer, state, .. } => self
                        .execute_causal_attention(
                            node,
                            layer,
                            state,
                            StatefulExecution {
                                token_count: transition.token_count(),
                                start_position: transition.start_position(),
                                expected_length: transition.expected_length(),
                            },
                            &mut pending,
                        )?,
                    QwenGraphNodeKind::LinearAttentionState { layer, state, .. } => self
                        .execute_linear_attention(
                            node,
                            layer,
                            state,
                            StatefulExecution {
                                token_count: transition.token_count(),
                                start_position: transition.start_position(),
                                expected_length: transition.expected_length(),
                            },
                            &mut pending,
                        )?,
                }
                Ok(())
            })()
            .map_err(|error| {
                if matches!(
                    &error,
                    QwenExecutionError::Execution(
                        ExecutionError::Busy
                            | ExecutionError::BackendStatus { .. }
                            | ExecutionError::AsyncFailure { .. }
                    )
                ) {
                    QwenExecutionError::NodeExecution {
                        node: node.label().to_owned(),
                        error: Box::new(error),
                    }
                } else {
                    error
                }
            })
        })?;
        if !emit_terminal {
            if pending.is_empty() {
                return Err(QwenExecutionError::InvalidGraph(
                    "chunked prefill has no work before terminal projection".to_owned(),
                ));
            }
            self.close_boundary(&mut pending, ExecutionBoundaryKind::PrefillChunkCompletion)?;
            return Ok(TerminalSelection {
                token_ids: Vec::new(),
                selection: None,
            });
        }
        if let Some(argmax) = argmax {
            return Ok(argmax);
        }
        if !pending.is_empty() {
            self.close_boundary(&mut pending, ExecutionBoundaryKind::Error)?;
        }
        Err(QwenExecutionError::InvalidGraph(
            "graph has no argmax output node".to_owned(),
        ))
    }

    fn read_last_logits(
        &self,
        token_count: u64,
        terminal_rows: TerminalOutputRows,
    ) -> Result<Vec<f32>, QwenExecutionError> {
        let argmax = self.graph.nodes().last().ok_or_else(|| {
            QwenExecutionError::InvalidGraph("graph has no terminal Argmax node".to_owned())
        })?;
        if !matches!(
            argmax.kind(),
            QwenGraphNodeKind::Semantic(SemanticOpKind::Argmax)
        ) || argmax.inputs().len() != 1
        {
            return Err(QwenExecutionError::InvalidGraph(
                "terminal Argmax does not have one logits input".to_owned(),
            ));
        }
        let logits_id = argmax.inputs()[0];
        let full_view = self.view(logits_id, token_count)?;
        let view = match terminal_rows {
            TerminalOutputRows::Last => first_row_view(&full_view)?,
            TerminalOutputRows::All => full_view,
        };
        if view.dtype() != DType::Bf16
            || view.shape()
                != [
                    if terminal_rows == TerminalOutputRows::Last {
                        1
                    } else {
                        usize::try_from(token_count).map_err(|_| {
                            QwenExecutionError::InvalidRequest(
                                "token count does not fit usize".to_owned(),
                            )
                        })?
                    },
                    QWEN35_VOCAB_SIZE,
                ]
        {
            return Err(QwenExecutionError::InvalidGraph(
                "terminal logits do not have the fixed BF16 [tokens,vocab] shape".to_owned(),
            ));
        }
        let row_bytes = u64::try_from(QWEN35_VOCAB_SIZE)
            .expect("fixed vocabulary fits u64")
            .checked_mul(2)
            .ok_or_else(|| {
                QwenExecutionError::InvalidGraph("logits row bytes overflowed".to_owned())
            })?;
        let row_index = match terminal_rows {
            TerminalOutputRows::Last => 0,
            TerminalOutputRows::All => token_count.checked_sub(1).ok_or_else(|| {
                QwenExecutionError::InvalidRequest("cannot read logits for zero tokens".to_owned())
            })?,
        };
        let row_offset = view
            .byte_offset()
            .checked_add(row_index.checked_mul(row_bytes).ok_or_else(|| {
                QwenExecutionError::InvalidGraph("logits row offset overflowed".to_owned())
            })?)
            .ok_or_else(|| {
                QwenExecutionError::InvalidGraph("logits row offset overflowed".to_owned())
            })?;
        let allocation = self.tensors.get(logits_id).ok_or_else(|| {
            QwenExecutionError::InvalidGraph("terminal logits allocation is absent".to_owned())
        })?;
        let maximum = self.session.max_transfer_bytes()?;
        if maximum == 0 {
            return Err(QwenExecutionError::InvalidRequest(
                "backend transfer limit must be non-zero".to_owned(),
            ));
        }
        let mut bytes =
            Vec::with_capacity(usize::try_from(row_bytes).expect("logits row fits usize"));
        let mut relative = 0_u64;
        while relative < row_bytes {
            let length = (row_bytes - relative).min(maximum);
            let offset = row_offset.checked_add(relative).ok_or_else(|| {
                QwenExecutionError::InvalidGraph("logits chunk offset overflowed".to_owned())
            })?;
            let range = allocation.buffer.range(offset, length)?;
            let mut readback = self.session.readback(&self.queue, range)?;
            require_terminal_success(
                "last-logits-readback",
                readback.wait(self.completion_timeout)?,
            )?;
            let start = bytes.len();
            bytes.resize(
                start + usize::try_from(length).expect("transfer length fits usize"),
                0,
            );
            readback.read_into(&mut bytes[start..])?;
            relative = relative.checked_add(length).ok_or_else(|| {
                QwenExecutionError::InvalidGraph("logits chunk progress overflowed".to_owned())
            })?;
        }
        decode_bf16_logits(&bytes)
    }

    fn read_logits_bf16(&self, token_count: u64) -> Result<Vec<u16>, QwenExecutionError> {
        let argmax = self.graph.nodes().last().ok_or_else(|| {
            QwenExecutionError::InvalidGraph("graph has no terminal Argmax node".to_owned())
        })?;
        if !matches!(
            argmax.kind(),
            QwenGraphNodeKind::Semantic(SemanticOpKind::Argmax)
        ) || argmax.inputs().len() != 1
        {
            return Err(QwenExecutionError::InvalidGraph(
                "terminal Argmax does not have one logits input".to_owned(),
            ));
        }
        let logits_id = argmax.inputs()[0];
        let view = self.view(logits_id, token_count)?;
        let rows = usize::try_from(token_count).map_err(|_| {
            QwenExecutionError::InvalidRequest("token count does not fit usize".to_owned())
        })?;
        if view.dtype() != DType::Bf16 || view.shape() != [rows, QWEN35_VOCAB_SIZE] {
            return Err(QwenExecutionError::InvalidGraph(
                "terminal logits do not have the fixed BF16 [tokens,vocab] shape".to_owned(),
            ));
        }
        let word_count = rows.checked_mul(QWEN35_VOCAB_SIZE).ok_or_else(|| {
            QwenExecutionError::InvalidGraph("logits word count overflowed".to_owned())
        })?;
        let total_bytes = u64::try_from(word_count)
            .map_err(|_| {
                QwenExecutionError::InvalidGraph("logits word count does not fit u64".to_owned())
            })?
            .checked_mul(2)
            .ok_or_else(|| {
                QwenExecutionError::InvalidGraph("logits byte count overflowed".to_owned())
            })?;
        let allocation = self.tensors.get(logits_id).ok_or_else(|| {
            QwenExecutionError::InvalidGraph("terminal logits allocation is absent".to_owned())
        })?;
        let maximum = self.session.max_transfer_bytes()?;
        if maximum == 0 {
            return Err(QwenExecutionError::InvalidRequest(
                "backend transfer limit must be non-zero".to_owned(),
            ));
        }
        let mut bytes = Vec::with_capacity(word_count * 2);
        let mut relative = 0_u64;
        while relative < total_bytes {
            let length = (total_bytes - relative).min(maximum);
            let offset = view.byte_offset().checked_add(relative).ok_or_else(|| {
                QwenExecutionError::InvalidGraph("logits chunk offset overflowed".to_owned())
            })?;
            let range = allocation.buffer.range(offset, length)?;
            let mut readback = self.session.readback(&self.queue, range)?;
            require_terminal_success(
                "all-logits-readback",
                readback.wait(self.completion_timeout)?,
            )?;
            let start = bytes.len();
            bytes.resize(
                start + usize::try_from(length).expect("transfer length fits usize"),
                0,
            );
            readback.read_into(&mut bytes[start..])?;
            relative = relative.checked_add(length).ok_or_else(|| {
                QwenExecutionError::InvalidGraph("logits chunk progress overflowed".to_owned())
            })?;
        }
        if bytes.len() != word_count * 2 {
            return Err(QwenExecutionError::InvalidGraph(
                "all-logits readback byte count differs".to_owned(),
            ));
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect())
    }

    fn read_hidden_states(&self, token_count: u64) -> Result<Vec<u16>, QwenExecutionError> {
        let final_norm = self
            .graph
            .nodes()
            .iter()
            .find(|node| node.label() == "final_rmsnorm")
            .ok_or_else(|| {
                QwenExecutionError::InvalidGraph("final RMSNorm node is absent".to_owned())
            })?;
        let hidden_id = *final_norm.inputs().first().ok_or_else(|| {
            QwenExecutionError::InvalidGraph("final RMSNorm hidden input is absent".to_owned())
        })?;
        let view = self.view(hidden_id, token_count)?;
        if view.dtype() != DType::Bf16
            || view.shape()
                != [
                    usize::try_from(token_count).map_err(|_| {
                        QwenExecutionError::InvalidRequest(
                            "hidden token count does not fit usize".to_owned(),
                        )
                    })?,
                    2_560,
                ]
        {
            return Err(QwenExecutionError::InvalidGraph(
                "MTP hidden hook requires BF16 [tokens,2560] before final RMSNorm".to_owned(),
            ));
        }
        let allocation = self.tensors.get(hidden_id).ok_or_else(|| {
            QwenExecutionError::InvalidGraph("hidden state allocation is absent".to_owned())
        })?;
        let bytes = read_exact_bytes(
            self.session.as_ref(),
            &self.queue,
            &allocation.buffer,
            &view,
            self.completion_timeout,
            "MTP hidden-state readback",
        )?;
        if bytes.len() % 2 != 0 {
            return Err(QwenExecutionError::InvalidGraph(
                "MTP hidden-state byte count is odd".to_owned(),
            ));
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect())
    }

    fn read_final_hidden_states(&self, token_count: u64) -> Result<Vec<u16>, QwenExecutionError> {
        if token_count == 0 {
            return Err(QwenExecutionError::InvalidRequest(
                "embedding readback requires at least one token".to_owned(),
            ));
        }
        let mut matches = self
            .graph
            .nodes()
            .iter()
            .filter(|node| node.label() == "final_rmsnorm")
            .collect::<Vec<_>>();
        let final_norm = matches.pop().ok_or_else(|| {
            QwenExecutionError::InvalidGraph("final RMSNorm node is absent".to_owned())
        })?;
        if !matches.is_empty()
            || !matches!(
                final_norm.kind(),
                QwenGraphNodeKind::Semantic(SemanticOpKind::RmsNorm)
            )
            || final_norm.inputs().len() != 2
            || final_norm.outputs().len() != 1
        {
            return Err(QwenExecutionError::InvalidGraph(
                "embedding final RMSNorm node identity is invalid".to_owned(),
            ));
        }
        let output_id = final_norm.outputs()[0];
        let rows = usize::try_from(token_count).map_err(|_| {
            QwenExecutionError::InvalidRequest(
                "embedding token count does not fit usize".to_owned(),
            )
        })?;
        let view = self.view(output_id, token_count)?;
        if view.dtype() != DType::Bf16
            || view.encoding() != Encoding::Unquantized
            || view.shape() != [rows, 2_560]
            || !view.is_contiguous()
        {
            return Err(QwenExecutionError::InvalidGraph(
                "embedding final RMSNorm output must be contiguous BF16 [tokens,2560]".to_owned(),
            ));
        }
        let expected_bytes = rows
            .checked_mul(2_560)
            .and_then(|words| words.checked_mul(2))
            .ok_or_else(|| {
                QwenExecutionError::InvalidRequest("embedding readback size overflowed".to_owned())
            })?;
        let allocation = self.tensors.get(output_id).ok_or_else(|| {
            QwenExecutionError::InvalidGraph("embedding output allocation is absent".to_owned())
        })?;
        let bytes = read_exact_bytes(
            self.session.as_ref(),
            &self.queue,
            &allocation.buffer,
            &view,
            self.completion_timeout,
            "embedding final-hidden readback",
        )?;
        if bytes.len() != expected_bytes || bytes.len() % 2 != 0 {
            return Err(QwenExecutionError::InvalidGraph(
                "embedding final-hidden readback size differs".to_owned(),
            ));
        }
        let values = bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>();
        if values
            .iter()
            .any(|bits| !f32::from_bits(u32::from(*bits) << 16).is_finite())
        {
            return Err(QwenExecutionError::InvalidGraph(
                "embedding final-hidden readback contains non-finite BF16".to_owned(),
            ));
        }
        Ok(values)
    }

    fn execute_semantic(
        &self,
        node: &QwenGraphNode,
        token_count: u64,
        boundary_after: Option<ExecutionBoundaryKind>,
        terminal_rows: TerminalOutputRows,
        device_selector: Option<&DeviceTokenSelectorRequestV1>,
        pending: &mut ExecutionSegment,
    ) -> Result<Option<TerminalSelection>, QwenExecutionError> {
        if terminal_rows == TerminalOutputRows::All {
            if device_selector.is_some() {
                return Err(QwenExecutionError::InvalidRequest(
                    "device token selector requires a final BF16 row".to_owned(),
                ));
            }
            return self.execute_semantic_all_rows(node, token_count, boundary_after, pending);
        }
        let operation = node.operation().ok_or_else(|| {
            QwenExecutionError::InvalidGraph(format!(
                "semantic node {} has no descriptor",
                node.label()
            ))
        })?;
        if operation.kind() == SemanticOpKind::Matmul
            && self
                .node_weight_name(node)
                .is_some_and(|name| self.adapters.has_lora_target(name))
        {
            return self.execute_lora_matmul(
                node,
                token_count,
                boundary_after,
                terminal_rows,
                pending,
            );
        }
        if operation.kind() == SemanticOpKind::AttentionPreprocess {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "attention preprocess must use the typed D0 node: {}",
                node.label()
            )));
        }
        let is_terminal_projection = self.is_terminal_projection(node)?;
        let is_terminal_argmax = operation.kind() == SemanticOpKind::Argmax;
        let (inputs, outputs, input_bindings, output_bindings) = if is_terminal_projection {
            if node.inputs().len() != 2 || node.outputs().len() != 1 {
                return Err(QwenExecutionError::InvalidGraph(
                    "terminal projection does not have two inputs and one output".to_owned(),
                ));
            }
            let activation = last_row_view(&self.view(node.inputs()[0], token_count)?)?;
            let weight = self.view(node.inputs()[1], token_count)?;
            let logits = first_row_view(&self.view(node.outputs()[0], token_count)?)?;
            let input_bindings = vec![
                self.bind_view(node.inputs()[0], activation.clone(), AccessMode::Read)?,
                self.bind(node.inputs()[1], token_count, AccessMode::Read)?,
            ];
            let output_bindings =
                vec![self.bind_view(node.outputs()[0], logits.clone(), AccessMode::Write)?];
            (
                vec![activation, weight],
                vec![logits],
                input_bindings,
                output_bindings,
            )
        } else if is_terminal_argmax {
            if node.inputs().len() != 1 || node.outputs().len() != 1 {
                return Err(QwenExecutionError::InvalidGraph(
                    "terminal Argmax does not have one input and one output".to_owned(),
                ));
            }
            let logits = first_row_view(&self.view(node.inputs()[0], token_count)?)?;
            let output = first_row_view(&self.view(node.outputs()[0], token_count)?)?;
            let input_bindings =
                vec![self.bind_view(node.inputs()[0], logits.clone(), AccessMode::Read)?];
            let output_bindings =
                vec![self.bind_view(node.outputs()[0], output.clone(), AccessMode::Write)?];
            (vec![logits], vec![output], input_bindings, output_bindings)
        } else {
            (
                self.views(node.inputs(), token_count)?,
                self.views(node.outputs(), token_count)?,
                self.bind_many(node.inputs(), token_count, AccessMode::Read)?,
                self.bind_many(node.outputs(), token_count, AccessMode::Write)?,
            )
        };
        if let Some(selector) = device_selector.filter(|_| is_terminal_argmax) {
            return self.execute_device_token_selector(
                node,
                token_count,
                boundary_after,
                selector,
                pending,
            );
        }
        let descriptor = match operation.kind() {
            SemanticOpKind::RmsNorm => SemanticOpDescriptor::new_rms_norm_with_contract(
                inputs,
                outputs,
                operation.rms_norm_contract().ok_or_else(|| {
                    QwenExecutionError::InvalidGraph(format!(
                        "RMSNorm node {} has no contract",
                        node.label()
                    ))
                })?,
            )?,
            SemanticOpKind::SparseMoe => SemanticOpDescriptor::new_sparse_moe(
                inputs,
                outputs,
                operation.sparse_moe_contract().ok_or_else(|| {
                    QwenExecutionError::InvalidGraph(format!(
                        "SparseMoe node {} has no contract",
                        node.label()
                    ))
                })?,
            )?,
            kind => SemanticOpDescriptor::new(kind, inputs, outputs)?,
        };
        let kind = descriptor.kind();
        let mut submission = self.submit_semantic(
            descriptor,
            input_bindings,
            output_bindings,
            PreparedCachePolicy::Reusable(PreparedDynamicIdentity::stateless(token_count, 0)),
        )?;
        if kind != SemanticOpKind::Argmax {
            if boundary_after.is_some() {
                return Err(QwenExecutionError::InvalidGraph(format!(
                    "non-terminal semantic node {} declares a boundary",
                    node.label()
                )));
            }
            pending.retain_semantic(node.label(), submission);
            if let Some(layer) = Self::control_layer(node) {
                self.execute_control_after_node(node, token_count, layer, pending)?;
            }
            return Ok(None);
        }
        // The terminal argmax completion is the final stream-ordered segment
        // boundary. Once it succeeds, every earlier completion can be checked
        // without serially sleeping between individual semantic operations.
        if boundary_after != Some(ExecutionBoundaryKind::TerminalReadback) {
            return Err(QwenExecutionError::InvalidGraph(
                "terminal Argmax lacks its readback boundary".to_owned(),
            ));
        }
        self.close_boundary_with_semantic(
            pending,
            node.label(),
            &mut submission,
            ExecutionBoundaryKind::TerminalReadback,
        )?;
        if node.outputs().len() != 1 {
            return Err(QwenExecutionError::InvalidGraph(
                "argmax node must have exactly one output".to_owned(),
            ));
        }
        let output_view = first_row_view(&self.view(node.outputs()[0], token_count)?)?;
        let byte_length = usize::try_from(output_view.payload_bytes()).map_err(|_| {
            QwenExecutionError::InvalidGraph(
                "argmax output byte length does not fit usize".to_owned(),
            )
        })?;
        let mut readback = submission.start_output_readback(0)?;
        require_terminal_success(node.label(), readback.wait(self.completion_timeout)?)?;
        let mut bytes = vec![0_u8; byte_length];
        let copied = readback.read_into(&mut bytes)?;
        if copied != u64::try_from(bytes.len()).expect("usize always fits u64") {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "argmax readback for {} returned a short or long byte count",
                node.label()
            )));
        }
        Ok(Some(TerminalSelection {
            token_ids: decode_argmax_bytes(&bytes)?,
            selection: None,
        }))
    }

    fn execute_device_token_selector(
        &self,
        node: &QwenGraphNode,
        token_count: u64,
        boundary_after: Option<ExecutionBoundaryKind>,
        selector: &DeviceTokenSelectorRequestV1,
        pending: &mut ExecutionSegment,
    ) -> Result<Option<TerminalSelection>, QwenExecutionError> {
        if boundary_after != Some(ExecutionBoundaryKind::TerminalReadback) {
            return Err(QwenExecutionError::InvalidGraph(
                "terminal token selector lacks its readback boundary".to_owned(),
            ));
        }
        if node.inputs().len() != 1 || node.outputs().len() != 1 {
            return Err(QwenExecutionError::InvalidGraph(
                "terminal token selector requires one logits input and one output".to_owned(),
            ));
        }
        let vocab = QWEN35_VOCAB_SIZE;
        if selector.additive_logits().len() != vocab || selector.valid_mask().len() != vocab {
            return Err(QwenExecutionError::InvalidRequest(
                "device selector additive logits/mask must match Qwen vocabulary".to_owned(),
            ));
        }
        if !selector.valid_mask().iter().any(|&value| value != 0) {
            return Err(QwenExecutionError::InvalidRequest(
                "device selector valid mask rejects every token".to_owned(),
            ));
        }
        let logits = first_row_view(&self.view(node.inputs()[0], token_count)?)?;
        if logits.dtype() != DType::Bf16 || logits.shape() != [1, vocab] {
            return Err(QwenExecutionError::InvalidGraph(
                "terminal selector logits must be BF16 [1,vocab]".to_owned(),
            ));
        }
        let additive_view = TensorView::contiguous(DType::F32, &[1, vocab]).map_err(|error| {
            QwenExecutionError::InvalidGraph(format!(
                "device selector additive tensor view is invalid: {error}"
            ))
        })?;
        let mask_view = TensorView::contiguous(DType::U8, &[1, vocab]).map_err(|error| {
            QwenExecutionError::InvalidGraph(format!(
                "device selector mask tensor view is invalid: {error}"
            ))
        })?;
        let output_view = TensorView::contiguous(DType::U8, &[16]).map_err(|error| {
            QwenExecutionError::InvalidGraph(format!(
                "device selector output tensor view is invalid: {error}"
            ))
        })?;
        let additive = self.session.allocate_with_category(
            additive_view.payload_bytes(),
            crate::AllocationCategory::RequestState,
        )?;
        let valid_mask = self.session.allocate_with_category(
            mask_view.payload_bytes(),
            crate::AllocationCategory::RequestState,
        )?;
        let output = self.session.allocate_with_category(
            output_view.payload_bytes(),
            crate::AllocationCategory::RequestState,
        )?;
        let additive_bytes = selector
            .additive_logits()
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let mask_bytes = selector.valid_mask().to_vec();
        upload_exact_bytes(
            self.session.as_ref(),
            &self.queue,
            &additive,
            &additive_view,
            &additive_bytes,
            self.completion_timeout,
            "device selector additive-logit upload",
        )?;
        upload_exact_bytes(
            self.session.as_ref(),
            &self.queue,
            &valid_mask,
            &mask_view,
            &mask_bytes,
            self.completion_timeout,
            "device selector valid-mask upload",
        )?;
        let contract = TokenSelectorContractV1::new(
            u64::try_from(vocab).expect("Qwen vocabulary fits u64"),
            selector.temperature(),
            selector.seed(),
            selector.counter(),
        )?;
        let descriptor = SemanticOpDescriptor::new_token_select(
            vec![logits.clone(), additive_view.clone(), mask_view.clone()],
            vec![output_view.clone()],
            contract,
        )?;
        let input_bindings = vec![
            self.bind_view(node.inputs()[0], logits, AccessMode::Read)?,
            self.session
                .bind(&additive, additive_view, AccessMode::Read)?,
            self.session
                .bind(&valid_mask, mask_view, AccessMode::Read)?,
        ];
        let output_binding = self.session.bind(&output, output_view, AccessMode::Write)?;
        let mut submission = self.submit_semantic(
            descriptor,
            input_bindings,
            vec![output_binding],
            PreparedCachePolicy::Transient,
        )?;
        self.close_boundary_with_semantic(
            pending,
            node.label(),
            &mut submission,
            ExecutionBoundaryKind::TerminalReadback,
        )?;
        let mut readback = submission.start_output_readback(0)?;
        require_terminal_success(node.label(), readback.wait(self.completion_timeout)?)?;
        let mut bytes = [0_u8; 16];
        let copied = readback.read_into(&mut bytes)?;
        if copied != 16 {
            return Err(QwenExecutionError::InvalidRequest(
                "device selector returned a short or long selected record".to_owned(),
            ));
        }
        let token_id = i32::from_le_bytes(bytes[0..4].try_into().expect("record token bytes"));
        let status = u32::from_le_bytes(bytes[4..8].try_into().expect("record status bytes"));
        let logprob = f32::from_le_bytes(bytes[8..12].try_into().expect("record logprob bytes"));
        let reserved = u32::from_le_bytes(bytes[12..16].try_into().expect("record reserved bytes"));
        if status != 0 {
            return Err(QwenExecutionError::Execution(
                ExecutionError::BackendStatus {
                    status,
                    diagnostic: format!("{} token-selector record status", node.label()),
                },
            ));
        }
        if reserved != 0 {
            return Err(QwenExecutionError::InvalidRequest(
                "device selector selected record reserved field is non-zero".to_owned(),
            ));
        }
        if token_id < 0 || usize::try_from(token_id).map_or(true, |id| id >= vocab) {
            return Err(QwenExecutionError::InvalidRequest(
                "device selector returned an out-of-range token ID".to_owned(),
            ));
        }
        let token_index = usize::try_from(token_id).expect("non-negative token ID");
        if selector.valid_mask()[token_index] == 0 {
            return Err(QwenExecutionError::InvalidRequest(
                "device selector returned a masked token ID".to_owned(),
            ));
        }
        if !logprob.is_finite() {
            return Err(QwenExecutionError::InvalidRequest(
                "device selector returned a non-finite logprob".to_owned(),
            ));
        }
        Ok(Some(TerminalSelection {
            token_ids: vec![token_id],
            selection: Some(SamplingSelectionV1 {
                token_id: u32::try_from(token_id).expect("validated token ID fits u32"),
                logprob: f64::from(logprob),
                top_logprobs: Vec::new(),
            }),
        }))
    }

    fn node_weight_name<'a>(&'a self, node: &'a QwenGraphNode) -> Option<&'a str> {
        node.inputs()
            .get(1)
            .and_then(|&tensor_id| self.graph.tensor_metadata().get(tensor_id))
            .map(|tensor| tensor.name())
    }

    fn allocate_adapter_temp(
        &self,
        view: &TensorView,
    ) -> Result<(ExecutionBuffer, TensorView), QwenExecutionError> {
        let temp_view = TensorView::contiguous(view.dtype(), view.shape())?;
        let buffer = self.session.allocate_with_category(
            temp_view.payload_bytes(),
            crate::AllocationCategory::Workspace,
        )?;
        Ok((buffer, temp_view))
    }

    fn retain_adapter_semantic(
        &self,
        label: &str,
        descriptor: SemanticOpDescriptor,
        inputs: Vec<OwnedTensorBinding>,
        outputs: Vec<OwnedTensorBinding>,
        pending: &mut ExecutionSegment,
    ) -> Result<(), QwenExecutionError> {
        let submission = self.submit_semantic(
            descriptor,
            inputs,
            outputs,
            PreparedCachePolicy::Reusable(PreparedDynamicIdentity::stateless(0, 0)),
        )?;
        pending.retain_semantic(label, submission);
        Ok(())
    }

    fn execute_lora_matmul(
        &self,
        node: &QwenGraphNode,
        token_count: u64,
        boundary_after: Option<ExecutionBoundaryKind>,
        terminal_rows: TerminalOutputRows,
        pending: &mut ExecutionSegment,
    ) -> Result<Option<TerminalSelection>, QwenExecutionError> {
        if boundary_after.is_some() {
            return Err(QwenExecutionError::InvalidGraph(
                "LoRA matmul cannot own an execution boundary".to_owned(),
            ));
        }
        let weight_name = self.node_weight_name(node).ok_or_else(|| {
            QwenExecutionError::InvalidGraph("LoRA matmul has no weight input".to_owned())
        })?;
        let is_terminal_projection = self.is_terminal_projection(node)?;
        let activation = if is_terminal_projection && terminal_rows == TerminalOutputRows::Last {
            last_row_view(&self.view(node.inputs()[0], token_count)?)?
        } else {
            self.view(node.inputs()[0], token_count)?
        };
        let weight = self.view(node.inputs()[1], token_count)?;
        let output = if is_terminal_projection && terminal_rows == TerminalOutputRows::Last {
            first_row_view(&self.view(node.outputs()[0], token_count)?)?
        } else {
            self.view(node.outputs()[0], token_count)?
        };
        if activation.shape().len() != 2
            || weight.shape().len() != 2
            || output.shape().len() != 2
            || output.shape()[1] != weight.shape()[0]
        {
            return Err(QwenExecutionError::InvalidGraph(
                "LoRA matmul has an invalid runtime shape".to_owned(),
            ));
        }
        let rows = activation.shape()[0];
        let input = activation.shape()[1];
        let output_width = output.shape()[1];
        let (base_buffer, base_view) = self.allocate_adapter_temp(&output)?;
        let base_descriptor = SemanticOpDescriptor::new(
            SemanticOpKind::Matmul,
            vec![activation.clone(), weight.clone()],
            vec![base_view.clone()],
        )?;
        let base_inputs = vec![
            self.bind_view(node.inputs()[0], activation.clone(), AccessMode::Read)?,
            self.bind(node.inputs()[1], token_count, AccessMode::Read)?,
        ];
        let base_outputs =
            vec![
                self.session
                    .bind(&base_buffer, base_view.clone(), AccessMode::Write)?,
            ];
        let base_submission = self.submit_semantic(
            base_descriptor,
            base_inputs,
            base_outputs,
            PreparedCachePolicy::Reusable(PreparedDynamicIdentity::stateless(token_count, 0)),
        )?;
        pending.retain_semantic(node.label(), base_submission);
        let mut accumulator = (base_buffer, base_view);
        let matching = self
            .adapters
            .lora
            .iter()
            .filter_map(|artifact| {
                artifact
                    .selection
                    .artifact
                    .targets()
                    .iter()
                    .find(|target| target.tensor_name() == weight_name)
                    .map(|target| (artifact, target))
            })
            .collect::<Vec<_>>();
        if matching.is_empty() {
            return Err(QwenExecutionError::InvalidRequest(
                "LoRA runtime target disappeared during execution".to_owned(),
            ));
        }
        for (adapter, target) in matching {
            let rank = usize::try_from(target.rank()).map_err(|_| {
                QwenExecutionError::InvalidRequest("LoRA rank does not fit usize".to_owned())
            })?;
            let a_view = TensorView::new(
                DType::Bf16,
                Encoding::Unquantized,
                &[rank, input],
                &[input, 1],
                target.a_offset(),
            )?;
            let b_view = TensorView::new(
                DType::Bf16,
                Encoding::Unquantized,
                &[output_width, rank],
                &[rank, 1],
                target.b_offset(),
            )?;
            let low_template = TensorView::contiguous(DType::Bf16, &[rows, rank])?;
            let (low_buffer, low_view) = self.allocate_adapter_temp(&low_template)?;
            let low_descriptor = SemanticOpDescriptor::new(
                SemanticOpKind::Matmul,
                vec![activation.clone(), a_view.clone()],
                vec![low_view.clone()],
            )?;
            self.retain_adapter_semantic(
                "adapter.lora.project",
                low_descriptor,
                vec![
                    self.bind_view(node.inputs()[0], activation.clone(), AccessMode::Read)?,
                    self.session
                        .bind(&adapter.buffer, a_view, AccessMode::Read)?,
                ],
                vec![
                    self.session
                        .bind(&low_buffer, low_view.clone(), AccessMode::Write)?,
                ],
                pending,
            )?;
            let delta_template = TensorView::contiguous(DType::Bf16, &[rows, output_width])?;
            let (delta_buffer, delta_view) = self.allocate_adapter_temp(&delta_template)?;
            let delta_descriptor = SemanticOpDescriptor::new(
                SemanticOpKind::Matmul,
                vec![low_view.clone(), b_view.clone()],
                vec![delta_view.clone()],
            )?;
            self.retain_adapter_semantic(
                "adapter.lora.expand",
                delta_descriptor,
                vec![
                    self.session.bind(&low_buffer, low_view, AccessMode::Read)?,
                    self.session
                        .bind(&adapter.buffer, b_view, AccessMode::Read)?,
                ],
                vec![
                    self.session
                        .bind(&delta_buffer, delta_view.clone(), AccessMode::Write)?,
                ],
                pending,
            )?;
            let effective_scale = adapter.selection.scale
                * (adapter.selection.artifact.lock().alpha / target.rank() as f32);
            if !effective_scale.is_finite() {
                return Err(QwenExecutionError::InvalidRequest(
                    "LoRA effective scale is non-finite".to_owned(),
                ));
            }
            let scalar_view = TensorView::contiguous(DType::Bf16, &[1])?;
            let scalar_buffer = self.session.allocate_with_category(
                scalar_view.payload_bytes(),
                crate::AllocationCategory::Workspace,
            )?;
            let scalar_bits = bf16_scalar_from_f32(effective_scale)?;
            upload_exact_bytes(
                self.session.as_ref(),
                &self.queue,
                &scalar_buffer,
                &scalar_view,
                &scalar_bits.to_le_bytes(),
                self.completion_timeout,
                "LoRA scale upload",
            )?;
            let scaled_template = TensorView::contiguous(DType::Bf16, &[rows, output_width])?;
            let (scaled_buffer, scaled_view) = self.allocate_adapter_temp(&scaled_template)?;
            let scale_descriptor = SemanticOpDescriptor::new(
                SemanticOpKind::ScalarMul,
                vec![delta_view.clone(), scalar_view.clone()],
                vec![scaled_view.clone()],
            )?;
            self.retain_adapter_semantic(
                "adapter.lora.scale",
                scale_descriptor,
                vec![
                    self.session
                        .bind(&delta_buffer, delta_view, AccessMode::Read)?,
                    self.session
                        .bind(&scalar_buffer, scalar_view, AccessMode::Read)?,
                ],
                vec![
                    self.session
                        .bind(&scaled_buffer, scaled_view.clone(), AccessMode::Write)?,
                ],
                pending,
            )?;
            let (sum_buffer, sum_view) = self.allocate_adapter_temp(&output)?;
            let add_descriptor = SemanticOpDescriptor::new(
                SemanticOpKind::Add,
                vec![accumulator.1.clone(), scaled_view.clone()],
                vec![sum_view.clone()],
            )?;
            self.retain_adapter_semantic(
                "adapter.lora.add",
                add_descriptor,
                vec![
                    self.session
                        .bind(&accumulator.0, accumulator.1, AccessMode::Read)?,
                    self.session
                        .bind(&scaled_buffer, scaled_view, AccessMode::Read)?,
                ],
                vec![
                    self.session
                        .bind(&sum_buffer, sum_view.clone(), AccessMode::Write)?,
                ],
                pending,
            )?;
            accumulator = (sum_buffer, sum_view);
        }
        let copy_descriptor = SemanticOpDescriptor::new(
            SemanticOpKind::Copy,
            vec![accumulator.1.clone()],
            vec![output.clone()],
        )?;
        self.retain_adapter_semantic(
            "adapter.lora.commit",
            copy_descriptor,
            vec![
                self.session
                    .bind(&accumulator.0, accumulator.1, AccessMode::Read)?,
            ],
            vec![self.bind_view(node.outputs()[0], output, AccessMode::Write)?],
            pending,
        )?;
        Ok(None)
    }

    fn control_layer(node: &QwenGraphNode) -> Option<u32> {
        let rest = node.label().strip_prefix("layer.")?;
        let (layer, suffix) = rest.split_once('.')?;
        if !suffix.ends_with("mlp_residual_add") {
            return None;
        }
        layer.parse().ok()
    }

    fn execute_control_after_node(
        &self,
        node: &QwenGraphNode,
        token_count: u64,
        layer: u32,
        pending: &mut ExecutionSegment,
    ) -> Result<(), QwenExecutionError> {
        let controls = self.adapters.controls_for_layer(layer).collect::<Vec<_>>();
        if controls.is_empty() {
            return Ok(());
        }
        if node.outputs().len() != 1 {
            return Err(QwenExecutionError::InvalidGraph(
                "control-vector residual node must have one output".to_owned(),
            ));
        }
        for control in controls {
            let input = self.view(node.outputs()[0], token_count)?;
            if input.shape().len() != 2 {
                return Err(QwenExecutionError::InvalidGraph(
                    "control-vector residual must be a rank-2 hidden tensor".to_owned(),
                ));
            }
            let hidden = input.shape()[1];
            let lock = control.selection.artifact.lock();
            if u64::try_from(hidden).ok() != Some(lock.hidden_size) {
                return Err(QwenExecutionError::InvalidRequest(
                    "control-vector hidden size differs from residual".to_owned(),
                ));
            }
            let layer_offset = u64::from(layer)
                .checked_sub(lock.layer_start)
                .and_then(|index| index.checked_mul(lock.hidden_size))
                .and_then(|elements| elements.checked_mul(2))
                .and_then(|offset| lock.vector_offset.checked_add(offset))
                .ok_or_else(|| {
                    QwenExecutionError::InvalidRequest(
                        "control-vector layer payload offset overflowed".to_owned(),
                    )
                })?;
            let vector = TensorView::new(
                DType::Bf16,
                Encoding::Unquantized,
                &[hidden],
                &[1],
                layer_offset,
            )?;
            let effective_scale = control.selection.scale;
            if !effective_scale.is_finite() {
                return Err(QwenExecutionError::InvalidRequest(
                    "control-vector scale is non-finite".to_owned(),
                ));
            }
            let scalar_view = TensorView::contiguous(DType::Bf16, &[1])?;
            let scalar_buffer = self.session.allocate_with_category(
                scalar_view.payload_bytes(),
                crate::AllocationCategory::Workspace,
            )?;
            let scalar_bits = bf16_scalar_from_f32(effective_scale)?;
            upload_exact_bytes(
                self.session.as_ref(),
                &self.queue,
                &scalar_buffer,
                &scalar_view,
                &scalar_bits.to_le_bytes(),
                self.completion_timeout,
                "control-vector scale upload",
            )?;
            let (scaled_buffer, scaled_view) = self.allocate_adapter_temp(&vector)?;
            let scale_descriptor = SemanticOpDescriptor::new(
                SemanticOpKind::ScalarMul,
                vec![vector.clone(), scalar_view.clone()],
                vec![scaled_view.clone()],
            )?;
            self.retain_adapter_semantic(
                "adapter.control.scale",
                scale_descriptor,
                vec![
                    self.session
                        .bind(&control.buffer, vector.clone(), AccessMode::Read)?,
                    self.session
                        .bind(&scalar_buffer, scalar_view, AccessMode::Read)?,
                ],
                vec![
                    self.session
                        .bind(&scaled_buffer, scaled_view.clone(), AccessMode::Write)?,
                ],
                pending,
            )?;
            let (output_buffer, output_view) = self.allocate_adapter_temp(&input)?;
            let descriptor = SemanticOpDescriptor::new(
                SemanticOpKind::BroadcastAdd,
                vec![input.clone(), scaled_view.clone()],
                vec![output_view.clone()],
            )?;
            self.retain_adapter_semantic(
                "adapter.control.broadcast_add",
                descriptor,
                vec![
                    self.bind(node.outputs()[0], token_count, AccessMode::Read)?,
                    self.session
                        .bind(&scaled_buffer, scaled_view, AccessMode::Read)?,
                ],
                vec![
                    self.session
                        .bind(&output_buffer, output_view.clone(), AccessMode::Write)?,
                ],
                pending,
            )?;
            let copy_descriptor = SemanticOpDescriptor::new(
                SemanticOpKind::Copy,
                vec![output_view.clone()],
                vec![input.clone()],
            )?;
            self.retain_adapter_semantic(
                "adapter.control.commit",
                copy_descriptor,
                vec![
                    self.session
                        .bind(&output_buffer, output_view, AccessMode::Read)?,
                ],
                vec![self.bind(node.outputs()[0], token_count, AccessMode::Write)?],
                pending,
            )?;
        }
        Ok(())
    }

    fn execute_semantic_all_rows(
        &self,
        node: &QwenGraphNode,
        token_count: u64,
        boundary_after: Option<ExecutionBoundaryKind>,
        pending: &mut ExecutionSegment,
    ) -> Result<Option<TerminalSelection>, QwenExecutionError> {
        let operation = node.operation().ok_or_else(|| {
            QwenExecutionError::InvalidGraph(format!(
                "semantic node {} has no descriptor",
                node.label()
            ))
        })?;
        if operation.kind() == SemanticOpKind::Matmul
            && self
                .node_weight_name(node)
                .is_some_and(|name| self.adapters.has_lora_target(name))
        {
            return self.execute_lora_matmul(
                node,
                token_count,
                boundary_after,
                TerminalOutputRows::All,
                pending,
            );
        }
        if operation.kind() == SemanticOpKind::AttentionPreprocess {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "attention preprocess must use the typed D0 node: {}",
                node.label()
            )));
        }
        let inputs = self.views(node.inputs(), token_count)?;
        let outputs = self.views(node.outputs(), token_count)?;
        let descriptor = match operation.kind() {
            SemanticOpKind::RmsNorm => SemanticOpDescriptor::new_rms_norm_with_contract(
                inputs,
                outputs,
                operation.rms_norm_contract().ok_or_else(|| {
                    QwenExecutionError::InvalidGraph(format!(
                        "RMSNorm node {} has no contract",
                        node.label()
                    ))
                })?,
            )?,
            SemanticOpKind::SparseMoe => SemanticOpDescriptor::new_sparse_moe(
                inputs,
                outputs,
                operation.sparse_moe_contract().ok_or_else(|| {
                    QwenExecutionError::InvalidGraph(format!(
                        "SparseMoe node {} has no contract",
                        node.label()
                    ))
                })?,
            )?,
            kind => SemanticOpDescriptor::new(kind, inputs, outputs)?,
        };
        let input_bindings = self.bind_many(node.inputs(), token_count, AccessMode::Read)?;
        let output_bindings = self.bind_many(node.outputs(), token_count, AccessMode::Write)?;
        let kind = descriptor.kind();
        let mut submission = self.submit_semantic(
            descriptor,
            input_bindings,
            output_bindings,
            PreparedCachePolicy::Reusable(PreparedDynamicIdentity::stateless(token_count, 0)),
        )?;
        if kind != SemanticOpKind::Argmax {
            if boundary_after.is_some() {
                return Err(QwenExecutionError::InvalidGraph(format!(
                    "non-terminal semantic node {} declares a boundary",
                    node.label()
                )));
            }
            pending.retain_semantic(node.label(), submission);
            if let Some(layer) = Self::control_layer(node) {
                self.execute_control_after_node(node, token_count, layer, pending)?;
            }
            return Ok(None);
        }
        if boundary_after != Some(ExecutionBoundaryKind::TerminalReadback) {
            return Err(QwenExecutionError::InvalidGraph(
                "terminal Argmax lacks its readback boundary".to_owned(),
            ));
        }
        self.close_boundary_with_semantic(
            pending,
            node.label(),
            &mut submission,
            ExecutionBoundaryKind::TerminalReadback,
        )?;
        if node.outputs().len() != 1 {
            return Err(QwenExecutionError::InvalidGraph(
                "argmax node must have exactly one output".to_owned(),
            ));
        }
        let output_view = self.view(node.outputs()[0], token_count)?;
        let byte_length = usize::try_from(output_view.payload_bytes()).map_err(|_| {
            QwenExecutionError::InvalidGraph(
                "argmax output byte length does not fit usize".to_owned(),
            )
        })?;
        let mut readback = submission.start_output_readback(0)?;
        require_terminal_success(node.label(), readback.wait(self.completion_timeout)?)?;
        let mut bytes = vec![0_u8; byte_length];
        let copied = readback.read_into(&mut bytes)?;
        if copied != u64::try_from(bytes.len()).expect("usize always fits u64") {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "argmax readback for {} returned a short or long byte count",
                node.label()
            )));
        }
        Ok(Some(TerminalSelection {
            token_ids: decode_argmax_bytes(&bytes)?,
            selection: None,
        }))
    }

    fn execute_multimodal_embedding_select(
        &self,
        node: &QwenGraphNode,
        token_count: u64,
        position_mode: AttentionPreprocessPositionMode,
        pending: &mut ExecutionSegment,
    ) -> Result<(), QwenExecutionError> {
        if node.inputs().len() != 2 || node.outputs().len() != 1 {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "multimodal embedding selector {} has the wrong arity",
                node.label()
            )));
        }
        let selected_input = match position_mode {
            AttentionPreprocessPositionMode::Prefill => node.inputs()[1],
            AttentionPreprocessPositionMode::DecodeContinuation => node.inputs()[0],
        };
        let output = node.outputs()[0];
        let input_view = self.view(selected_input, token_count)?;
        let output_view = self.view(output, token_count)?;
        let descriptor =
            SemanticOpDescriptor::new(SemanticOpKind::Copy, vec![input_view], vec![output_view])?;
        let submission = self.submit_semantic(
            descriptor,
            vec![self.bind(selected_input, token_count, AccessMode::Read)?],
            vec![self.bind(output, token_count, AccessMode::Write)?],
            PreparedCachePolicy::Transient,
        )?;
        pending.retain_semantic(node.label(), submission);
        Ok(())
    }

    fn execute_attention_preprocess(
        &self,
        node: &QwenGraphNode,
        execution: AttentionPreprocessExecution,
        pending: &mut ExecutionSegment,
    ) -> Result<(), QwenExecutionError> {
        if node.inputs().len() != 5 || node.outputs().len() != 3 {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "attention preprocess node {} has the wrong binding arity",
                node.label()
            )));
        }
        let contract =
            AttentionPreprocessContract::new_qwen3_5_with_layout_and_context_and_position_payload_mode(
            execution.position_mode,
            i64::try_from(execution.start_position).map_err(|_| {
                QwenExecutionError::InvalidRequest("position does not fit i64".to_owned())
            })?,
            execution.token_count,
            execution.q_heads,
            execution.kv_heads,
            execution.head_dim,
            u32::try_from(if self.graph.position_payload_mode()
                == crate::AttentionPreprocessPositionPayloadModeV1::Explicit
            {
                QWEN_RUNTIME_MAX_CONTEXT_TOKENS
            } else {
                self.graph.state_capacity()
            })
            .map_err(|_| {
                QwenExecutionError::InvalidRequest(
                    "request context exceeds the u32 execution ABI".to_owned(),
                )
            })?,
            self.graph.position_payload_mode(),
        )?;
        let descriptor = SemanticOpDescriptor::new_attention_preprocess(
            self.views(node.inputs(), execution.token_count)?,
            self.views(node.outputs(), execution.token_count)?,
            contract,
        )?;
        let inputs = self.bind_many(node.inputs(), execution.token_count, AccessMode::Read)?;
        let outputs = self.bind_many(node.outputs(), execution.token_count, AccessMode::Write)?;
        // Position is part of this descriptor, so decode steps with the same
        // token count are not interchangeable prepared operations.
        let submission =
            self.submit_semantic(descriptor, inputs, outputs, PreparedCachePolicy::Transient)?;
        pending.retain_semantic(node.label(), submission);
        let _ = execution.layer;
        Ok(())
    }

    fn execute_kv_append(
        &self,
        node: &QwenGraphNode,
        layer: u32,
        descriptor: KvStateDescriptor,
        execution: StatefulExecution,
        boundary_after: Option<ExecutionBoundaryKind>,
        pending: &mut ExecutionSegment,
    ) -> Result<(), QwenExecutionError> {
        if node.inputs().len() != 2 || !node.outputs().is_empty() {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "KV append node {} has the wrong binding arity",
                node.label()
            )));
        }
        let state = self.kv_state(layer, descriptor)?;
        let key = self.bind(node.inputs()[0], execution.token_count, AccessMode::Read)?;
        let value = self.bind(node.inputs()[1], execution.token_count, AccessMode::Read)?;
        let mut submission = self.session.append_kv_state(
            state,
            &self.queue,
            key,
            value,
            execution.start_position,
            execution.start_position,
        )?;
        // KV publication is an unavoidable state boundary. The append event
        // is after all earlier work on the stream, so drain the preceding
        // segment now and then publish this state transition.
        if boundary_after != Some(ExecutionBoundaryKind::StatePublication) {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "KV append node {} lacks its state-publication boundary",
                node.label()
            )));
        }
        self.close_boundary_with_kv_append(
            pending,
            node.label(),
            &mut submission,
            ExecutionBoundaryKind::StatePublication,
        )
    }

    fn execute_causal_attention(
        &self,
        node: &QwenGraphNode,
        layer: u32,
        state_descriptor: KvStateDescriptor,
        execution: StatefulExecution,
        pending: &mut ExecutionSegment,
    ) -> Result<(), QwenExecutionError> {
        if node.inputs().len() != 1 || node.outputs().len() != 1 {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "causal attention node {} has the wrong binding arity",
                node.label()
            )));
        }
        let state = self.kv_state(layer, state_descriptor)?;
        let descriptor = CausalAttentionDescriptor::new(
            execution.start_position,
            execution.token_count,
            execution.expected_length,
        )
        .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
        let query = self.bind(node.inputs()[0], execution.token_count, AccessMode::Read)?;
        let output = self.bind(node.outputs()[0], execution.token_count, AccessMode::Write)?;
        let submission =
            self.session
                .causal_attention(state, &self.queue, query, output, descriptor)?;
        pending.retain_causal_attention(node.label(), submission);
        Ok(())
    }

    fn execute_linear_attention(
        &self,
        node: &QwenGraphNode,
        layer: u32,
        state_descriptor: LinearAttentionStateDescriptor,
        execution: StatefulExecution,
        pending: &mut ExecutionSegment,
    ) -> Result<(), QwenExecutionError> {
        if node.inputs().len() != 8 || node.outputs().len() != 1 {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "linear-attention node {} has the wrong binding arity",
                node.label()
            )));
        }
        let state = self.linear_state(layer, state_descriptor)?;
        let inputs = self.bind_many(node.inputs(), execution.token_count, AccessMode::Read)?;
        let output = self.bind(node.outputs()[0], execution.token_count, AccessMode::Write)?;
        let bindings = LinearAttentionBindings::new(
            inputs[0].clone(),
            inputs[1].clone(),
            inputs[2].clone(),
            inputs[3].clone(),
            inputs[4].clone(),
            inputs[5].clone(),
            inputs[6].clone(),
            inputs[7].clone(),
            output,
        );
        let descriptor = LinearAttentionDescriptor::new(
            execution.start_position,
            execution.token_count,
            execution.expected_length,
        )
        .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
        let submission = self
            .session
            .linear_attention(state, &self.queue, bindings, descriptor)?;
        pending.retain_linear_attention(node.label(), submission);
        Ok(())
    }

    fn close_boundary(
        &self,
        pending: &mut ExecutionSegment,
        boundary: ExecutionBoundaryKind,
    ) -> Result<(), QwenExecutionError> {
        if boundary == ExecutionBoundaryKind::PrefillChunkCompletion && !pending.is_empty() {
            let mut fence = self.session.create_queue_fence(&self.queue)?;
            require_terminal_success(
                "prefill chunk completion fence",
                fence.wait(self.completion_timeout)?,
            )?;
        }
        let mut audit = self
            .audit
            .lock()
            .map_err(|_| QwenExecutionError::Poisoned)?;
        pending.flush(boundary, &mut audit).map_err(Into::into)
    }

    fn close_boundary_with_semantic(
        &self,
        pending: &mut ExecutionSegment,
        label: &str,
        terminal: &mut Submission,
        boundary: ExecutionBoundaryKind,
    ) -> Result<(), QwenExecutionError> {
        let mut audit = self
            .audit
            .lock()
            .map_err(|_| QwenExecutionError::Poisoned)?;
        pending
            .flush_with_semantic(label, terminal, boundary, &mut audit)
            .map_err(Into::into)
    }

    fn close_boundary_with_kv_append(
        &self,
        pending: &mut ExecutionSegment,
        label: &str,
        terminal: &mut KvStateAppendSubmission,
        boundary: ExecutionBoundaryKind,
    ) -> Result<(), QwenExecutionError> {
        let mut audit = self
            .audit
            .lock()
            .map_err(|_| QwenExecutionError::Poisoned)?;
        pending
            .flush_with_kv_append(label, terminal, Some(boundary), &mut audit)
            .map_err(Into::into)
    }

    fn audit_snapshot(&self) -> Result<QwenExecutionAudit, QwenExecutionError> {
        let audit = self
            .audit
            .lock()
            .map_err(|_| QwenExecutionError::Poisoned)?
            .snapshot()?;
        if audit.backend() != 1 || audit.fallback_used() {
            return Err(QwenExecutionError::InvalidRequest(
                "Qwen dispatch audit is not HIP-only and fallback-free".to_owned(),
            ));
        }
        Ok(QwenExecutionAudit {
            selected_backend: "hip",
            target: audit.target().to_owned(),
            submission_count: audit.submission_count(),
            kernel_dispatch_count: audit.kernel_dispatch_count(),
            fallback_used: audit.fallback_used(),
            all_dispatches_hip: true,
            segment_count: audit.segment_count(),
            boundary_count: audit.boundary_count(),
            sparse_moe_submission_count: audit.sparse_moe_submission_count(),
            sparse_moe_active_pair_count: audit.sparse_moe_active_pair_count(),
        })
    }

    fn refresh_prefix_fork_audit(&self) -> Result<QwenPrefixForkAuditV1, QwenExecutionError> {
        let states = self.kv_states.values().collect::<Vec<_>>();
        let queried = self
            .session
            .kv_state_fork_query_all(states.iter().copied())?;
        let mut aggregate = QwenPrefixForkAuditV1::default();
        for (state, audit) in states.into_iter().zip(queried) {
            let descriptor = state.descriptor();
            let fallback_resident_bytes = descriptor
                .resident_bytes_per_plane()
                .and_then(|bytes| bytes.checked_mul(2))
                .ok_or_else(|| {
                    QwenExecutionError::InvalidRequest(
                        "KV fork resident-byte footprint overflowed".to_owned(),
                    )
                })?;
            let physical = self.session.kv_state_snapshot(state)?.physical_memory();
            aggregate.add(audit, false, physical, fallback_resident_bytes)?;
        }
        Ok(aggregate)
    }

    fn validate_cached_scale(
        &self,
        node: &QwenGraphNode,
        heads: u32,
        head_dim: u32,
    ) -> Result<(), QwenExecutionError> {
        if node.inputs().len() != 1 || node.outputs().len() != 1 || head_dim != 256 {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "attention scale node {} has an invalid contract",
                node.label()
            )));
        }
        let cached = self.scales.get(&node.outputs()[0]).ok_or_else(|| {
            QwenExecutionError::InvalidGraph(format!(
                "attention scale node {} was not provisioned",
                node.label()
            ))
        })?;
        if cached.raw_tensor_id != node.inputs()[0]
            || cached.raw_bytes.len() != 512
            || cached.expanded_bytes.len() != 512_usize.saturating_mul(heads as usize)
        {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "attention scale node {} does not retain the expected raw BF16 bytes",
                node.label()
            )));
        }
        Ok(())
    }

    fn submit_semantic(
        &self,
        descriptor: SemanticOpDescriptor,
        inputs: Vec<OwnedTensorBinding>,
        outputs: Vec<OwnedTensorBinding>,
        cache_policy: PreparedCachePolicy,
    ) -> Result<Submission, QwenExecutionError> {
        match self.session.supports(&descriptor) {
            PrepareSupport::Supported => {}
            PrepareSupport::Unsupported { reason } => {
                return Err(QwenExecutionError::Execution(ExecutionError::Unsupported {
                    reason: format!("{:?} is unsupported: {reason}", descriptor.kind()),
                }));
            }
        }
        let prepared = self.prepared_semantics.prepare(
            self.session.as_ref(),
            descriptor,
            inputs,
            outputs,
            cache_policy,
        )?;
        Ok(self.session.submit(&prepared, &self.queue)?)
    }

    fn upload_runtime_inputs(
        &self,
        token_ids: &[i32],
        start_position: u64,
        token_count: u64,
        target_hidden_bf16: Option<&[u16]>,
        multimodal: Option<(&[u16], &[[i32; 3]])>,
        explicit_positions: Option<&[u64]>,
    ) -> Result<(), QwenExecutionError> {
        let token_tensor = self.tensor_id("input.token_ids")?;
        let position_tensor = self.tensor_id("input.positions")?;
        let token_view = self.view(token_tensor, token_count)?;
        let position_view = self.view(position_tensor, token_count)?;
        let token_bytes = i32_bytes(token_ids);
        let positions = if self.graph.is_multimodal() {
            if let Some((_, positions)) = multimodal {
                if positions.len() != token_ids.len()
                    || positions.iter().flatten().any(|position| {
                        *position < 0
                            || u64::try_from(*position)
                                .map_or(true, |position| position >= self.graph.state_capacity())
                    })
                {
                    return Err(QwenExecutionError::InvalidRequest(
                        "multimodal position payload differs".to_owned(),
                    ));
                }
                mrope_position_bytes(positions)
            } else {
                let position = i32::try_from(start_position).map_err(|_| {
                    QwenExecutionError::InvalidRequest(
                        "decode mRoPE position does not fit i32".to_owned(),
                    )
                })?;
                let positions = vec![[position; 3]; token_ids.len()];
                mrope_position_bytes(&positions)
            }
        } else {
            if let Some(positions) = explicit_positions {
                position_values_bytes(positions)?
            } else {
                position_bytes(start_position, token_count)?
            }
        };
        upload_exact_bytes(
            self.session.as_ref(),
            &self.queue,
            &self.tensors[token_tensor].buffer,
            &token_view,
            &token_bytes,
            self.completion_timeout,
            "token input upload",
        )?;
        upload_exact_bytes(
            self.session.as_ref(),
            &self.queue,
            &self.tensors[position_tensor].buffer,
            &position_view,
            &positions,
            self.completion_timeout,
            "position input upload",
        )?;
        match (
            self.tensor_ids.get("input.multimodal_embeddings").copied(),
            multimodal,
        ) {
            (Some(tensor_id), Some((words, _))) => {
                let expected = usize::try_from(token_count)
                    .ok()
                    .and_then(|count| count.checked_mul(2_560))
                    .ok_or_else(|| {
                        QwenExecutionError::InvalidRequest(
                            "multimodal embedding length overflowed".to_owned(),
                        )
                    })?;
                if words.len() != expected {
                    return Err(QwenExecutionError::InvalidRequest(format!(
                        "multimodal embedding words are {}, expected {expected}",
                        words.len()
                    )));
                }
                let bytes = words
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>();
                let view = self.view(tensor_id, token_count)?;
                upload_exact_bytes(
                    self.session.as_ref(),
                    &self.queue,
                    &self.tensors[tensor_id].buffer,
                    &view,
                    &bytes,
                    self.completion_timeout,
                    "multimodal embedding upload",
                )?;
            }
            (Some(_), None) if self.committed_length == 0 => {
                return Err(QwenExecutionError::InvalidRequest(
                    "multimodal prefill embeddings are absent".to_owned(),
                ));
            }
            (None, Some(_)) => {
                return Err(QwenExecutionError::InvalidRequest(
                    "text graph does not accept multimodal embeddings".to_owned(),
                ));
            }
            _ => {}
        }
        match (
            self.tensor_ids.get("input.target_hidden").copied(),
            target_hidden_bf16,
        ) {
            (Some(tensor_id), Some(words)) => {
                let expected_words = usize::try_from(token_count)
                    .ok()
                    .and_then(|tokens| tokens.checked_mul(2_560))
                    .ok_or_else(|| {
                        QwenExecutionError::InvalidRequest(
                            "MTP hidden-state length overflowed".to_owned(),
                        )
                    })?;
                if words.len() != expected_words {
                    return Err(QwenExecutionError::InvalidRequest(format!(
                        "MTP hidden-state words are {}, expected {expected_words}",
                        words.len()
                    )));
                }
                let mut bytes = Vec::with_capacity(words.len() * 2);
                for word in words {
                    bytes.extend_from_slice(&word.to_le_bytes());
                }
                let view = self.view(tensor_id, token_count)?;
                upload_exact_bytes(
                    self.session.as_ref(),
                    &self.queue,
                    &self.tensors[tensor_id].buffer,
                    &view,
                    &bytes,
                    self.completion_timeout,
                    "MTP target-hidden upload",
                )
            }
            (Some(_), None) => Err(QwenExecutionError::InvalidRequest(
                "MTP graph requires a target hidden-state row".to_owned(),
            )),
            (None, Some(_)) => Err(QwenExecutionError::InvalidRequest(
                "text graph does not accept an MTP target hidden-state row".to_owned(),
            )),
            (None, None) => Ok(()),
        }
    }

    fn ensure_state_lengths(&self, expected: u64) -> Result<(), QwenExecutionError> {
        for (&layer, state) in &self.kv_states {
            let snapshot = self.session.kv_state_snapshot(state)?;
            if snapshot.length() != expected {
                return Err(QwenExecutionError::StateLength {
                    layer,
                    state: "KV",
                    expected,
                    actual: snapshot.length(),
                });
            }
        }
        for (&layer, state) in &self.linear_states {
            let snapshot = self.session.linear_attention_state_snapshot(state)?;
            if snapshot.length() != expected {
                return Err(QwenExecutionError::StateLength {
                    layer,
                    state: "linear-attention",
                    expected,
                    actual: snapshot.length(),
                });
            }
        }
        Ok(())
    }

    fn kv_state(
        &self,
        layer: u32,
        descriptor: KvStateDescriptor,
    ) -> Result<&KvState, QwenExecutionError> {
        let state = self.kv_states.get(&layer).ok_or_else(|| {
            QwenExecutionError::InvalidGraph(format!(
                "full-attention layer {layer} has no KV state"
            ))
        })?;
        if state.descriptor() != descriptor {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "full-attention layer {layer} state descriptor differs from the graph"
            )));
        }
        Ok(state)
    }

    fn linear_state(
        &self,
        layer: u32,
        descriptor: LinearAttentionStateDescriptor,
    ) -> Result<&LinearAttentionState, QwenExecutionError> {
        let state = self.linear_states.get(&layer).ok_or_else(|| {
            QwenExecutionError::InvalidGraph(format!("linear-attention layer {layer} has no state"))
        })?;
        if state.descriptor() != descriptor {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "linear-attention layer {layer} state descriptor differs from the graph"
            )));
        }
        Ok(state)
    }

    fn tensor_id(&self, name: &str) -> Result<usize, QwenExecutionError> {
        self.tensor_ids.get(name).copied().ok_or_else(|| {
            QwenExecutionError::InvalidGraph(format!("required graph tensor is absent: {name}"))
        })
    }

    fn views(
        &self,
        tensor_ids: &[usize],
        token_count: u64,
    ) -> Result<Vec<TensorView>, QwenExecutionError> {
        tensor_ids
            .iter()
            .map(|&tensor_id| self.view(tensor_id, token_count))
            .collect()
    }

    fn bind_many(
        &self,
        tensor_ids: &[usize],
        token_count: u64,
        access: AccessMode,
    ) -> Result<Vec<OwnedTensorBinding>, QwenExecutionError> {
        tensor_ids
            .iter()
            .map(|&tensor_id| self.bind(tensor_id, token_count, access))
            .collect()
    }

    fn bind(
        &self,
        tensor_id: usize,
        token_count: u64,
        access: AccessMode,
    ) -> Result<OwnedTensorBinding, QwenExecutionError> {
        let allocation = self.tensors.get(tensor_id).ok_or_else(|| {
            QwenExecutionError::InvalidGraph(format!("tensor allocation {tensor_id} is absent"))
        })?;
        Ok(self.session.bind(
            &allocation.buffer,
            self.view(tensor_id, token_count)?,
            access,
        )?)
    }

    fn bind_view(
        &self,
        tensor_id: usize,
        view: TensorView,
        access: AccessMode,
    ) -> Result<OwnedTensorBinding, QwenExecutionError> {
        let allocation = self.tensors.get(tensor_id).ok_or_else(|| {
            QwenExecutionError::InvalidGraph(format!("tensor allocation {tensor_id} is absent"))
        })?;
        Ok(self.session.bind(&allocation.buffer, view, access)?)
    }

    fn ensure_terminal_output_capacity(
        &mut self,
        token_count: u64,
    ) -> Result<(), QwenExecutionError> {
        for tensor_id in terminal_output_tensor_ids(&self.graph)? {
            let required = self.view(tensor_id, token_count)?.end_offset();
            let allocation = self.tensors.get_mut(tensor_id).ok_or_else(|| {
                QwenExecutionError::InvalidGraph(format!(
                    "terminal tensor allocation {tensor_id} is absent"
                ))
            })?;
            if allocation.buffer.size_bytes() < required {
                allocation.buffer = self
                    .session
                    .allocate_with_category(required, crate::AllocationCategory::Workspace)?;
            }
        }
        Ok(())
    }

    fn is_terminal_projection(&self, node: &QwenGraphNode) -> Result<bool, QwenExecutionError> {
        let argmax = self.graph.nodes().last().ok_or_else(|| {
            QwenExecutionError::InvalidGraph("graph has no terminal Argmax node".to_owned())
        })?;
        let logits_id = *argmax.inputs().first().ok_or_else(|| {
            QwenExecutionError::InvalidGraph("terminal Argmax has no logits input".to_owned())
        })?;
        Ok(node.outputs() == [logits_id]
            && matches!(
                node.kind(),
                QwenGraphNodeKind::Semantic(SemanticOpKind::Matmul)
            ))
    }

    fn view(&self, tensor_id: usize, token_count: u64) -> Result<TensorView, QwenExecutionError> {
        let allocation = self.tensors.get(tensor_id).ok_or_else(|| {
            QwenExecutionError::InvalidGraph(format!("tensor allocation {tensor_id} is absent"))
        })?;
        if !self
            .dynamic_tensors
            .get(tensor_id)
            .copied()
            .ok_or_else(|| {
                QwenExecutionError::InvalidGraph("dynamic tensor table is short".to_owned())
            })?
        {
            return Ok(allocation.graph_view.clone());
        }
        runtime_view(
            &allocation.graph_view,
            self.graph.token_count(),
            token_count,
        )
    }
}

fn model_resident_bytes(
    graph: &QwenGraph,
    layout: &GraphLayout,
) -> Result<u64, QwenExecutionError> {
    graph
        .tensor_metadata()
        .iter()
        .try_fold(0_u64, |total, tensor| {
            if tensor.backing() == QwenGraphTensorBacking::Owned
                && !layout.dynamic_tensors[tensor.id()]
            {
                total
                    .checked_add(
                        tensor
                            .view()
                            .byte_offset()
                            .checked_add(resident_weight_bytes(tensor.view())?)
                            .ok_or_else(|| {
                                QwenExecutionError::InvalidGraph(
                                    "owned tensor allocation byte count overflowed".to_owned(),
                                )
                            })?,
                    )
                    .ok_or_else(|| {
                        QwenExecutionError::InvalidGraph(
                            "model-resident byte count overflowed".to_owned(),
                        )
                    })
            } else {
                Ok(total)
            }
        })
}

fn memory_estimate_from_layout(
    graph: &QwenGraph,
    layout: &GraphLayout,
    total_memory_bytes: u64,
) -> Result<QwenGraphMemoryEstimate, QwenExecutionError> {
    if total_memory_bytes == 0 {
        return Err(QwenExecutionError::InvalidRequest(
            "device total memory is zero".to_owned(),
        ));
    }
    let model_resident_bytes = model_resident_bytes(graph, layout)?;
    let safety_reserve_bytes = (total_memory_bytes / 20).max(1024 * 1024 * 1024);
    let required_bytes = model_resident_bytes
        .checked_add(layout.workspace.high_water_bytes)
        .and_then(|bytes| bytes.checked_add(graph.total_state_bytes()))
        .and_then(|bytes| bytes.checked_add(safety_reserve_bytes))
        .ok_or_else(|| {
            QwenExecutionError::InvalidGraph(
                "model, workspace, and request-state byte count overflowed".to_owned(),
            )
        })?;
    Ok(QwenGraphMemoryEstimate {
        model_resident_bytes,
        workspace_baseline_bytes: layout.workspace.baseline_bytes,
        workspace_arena_bytes: layout.workspace.high_water_bytes,
        request_state_bytes: graph.total_state_bytes(),
        safety_reserve_bytes,
        required_bytes,
    })
}

pub fn qwen_graph_memory_estimate(
    graph: &QwenGraph,
    plan: &WeightLoadPlan,
    total_memory_bytes: u64,
) -> Result<QwenGraphMemoryEstimate, QwenExecutionError> {
    let layout = validate_graph_plan(graph, plan)?;
    memory_estimate_from_layout(graph, &layout, total_memory_bytes)
}

fn preflight_device_memory(
    session: &ExecutionSession,
    graph: &QwenGraph,
    layout: &GraphLayout,
    model_already_resident: bool,
) -> Result<(), QwenExecutionError> {
    let available = session.available_memory_bytes()?.ok_or_else(|| {
        QwenExecutionError::InvalidRequest(
            "backend did not report available device memory for placement preflight".to_owned(),
        )
    })?;
    let total = session.total_memory_bytes()?.unwrap_or(available);
    let estimate = memory_estimate_from_layout(graph, layout, total)?;
    let placement_required = if model_already_resident {
        estimate
            .required_bytes
            .checked_sub(estimate.model_resident_bytes)
            .ok_or_else(|| {
                QwenExecutionError::InvalidGraph(
                    "request placement byte count underflowed model-resident bytes".to_owned(),
                )
            })?
    } else {
        estimate.required_bytes
    };
    if placement_required > available {
        return Err(QwenExecutionError::InvalidRequest(format!(
            "device memory preflight requires {placement_required} placement bytes (full layout {}, model-resident {}{}, workspace arena {} from separate-allocation baseline {}, request-state {}, safety-reserve {}), but only {available} bytes are available",
            estimate.required_bytes,
            estimate.model_resident_bytes,
            if model_already_resident {
                " already allocated"
            } else {
                ""
            },
            estimate.workspace_arena_bytes,
            estimate.workspace_baseline_bytes,
            estimate.request_state_bytes,
            estimate.safety_reserve_bytes,
        )));
    }
    Ok(())
}

fn preflight_semantic_support(
    session: &ExecutionSession,
    graph: &QwenGraph,
) -> Result<(), QwenExecutionError> {
    for node in graph.nodes() {
        let Some(operation) = node.operation() else {
            continue;
        };
        match session.supports(operation) {
            PrepareSupport::Supported => {}
            PrepareSupport::Unsupported { reason } => {
                return Err(QwenExecutionError::InvalidRequest(format!(
                    "semantic node {} is unsupported before model upload: {reason}",
                    node.label()
                )));
            }
        }
    }
    Ok(())
}

fn validate_upload_receipt(
    receipt: &WeightUploadReceipt,
    plan: &WeightLoadPlan,
    binding: &QwenGraphWeightBinding,
) -> Result<(), QwenExecutionError> {
    if receipt.plan_digest != *plan.digest()
        || receipt.tensor_name != binding.tensor_name()
        || receipt.dtype != binding.dtype()
        || receipt.source_range != binding.source_range()
    {
        return Err(QwenExecutionError::InvalidRequest(format!(
            "verified upload receipt does not match graph weight {}",
            binding.tensor_name()
        )));
    }
    Ok(())
}

const QWEN_WORKSPACE_ALIGNMENT: u64 = 256;

fn align_workspace(value: u64) -> Result<u64, QwenExecutionError> {
    value
        .checked_add(QWEN_WORKSPACE_ALIGNMENT - 1)
        .map(|value| value & !(QWEN_WORKSPACE_ALIGNMENT - 1))
        .ok_or_else(|| {
            QwenExecutionError::InvalidGraph("workspace alignment overflowed".to_owned())
        })
}

fn workspace_root(graph: &QwenGraph, tensor_id: usize) -> Result<usize, QwenExecutionError> {
    let mut current = tensor_id;
    let mut remaining = graph.tensor_metadata().len();
    loop {
        let tensor = graph.tensor_metadata().get(current).ok_or_else(|| {
            QwenExecutionError::InvalidGraph("workspace tensor ID is out of range".to_owned())
        })?;
        match tensor.backing() {
            QwenGraphTensorBacking::Owned => return Ok(current),
            QwenGraphTensorBacking::Alias { tensor_id } => {
                if tensor_id >= current || remaining == 0 {
                    return Err(QwenExecutionError::InvalidGraph(format!(
                        "alias tensor {} does not refer to a prior owned tensor",
                        tensor.name()
                    )));
                }
                current = tensor_id;
                remaining -= 1;
            }
        }
    }
}

fn workspace_allocation_bytes(
    graph: &QwenGraph,
    tensor_id: usize,
    terminal_outputs: [usize; 2],
) -> Result<u64, QwenExecutionError> {
    let tensor = graph.tensor_metadata().get(tensor_id).ok_or_else(|| {
        QwenExecutionError::InvalidGraph("workspace tensor ID is out of range".to_owned())
    })?;
    compact_terminal_allocation_end(
        graph.token_count(),
        graph.is_mtp(),
        tensor_id,
        terminal_outputs,
        tensor.view(),
    )
    .map(|value| value.unwrap_or_else(|| tensor.view().end_offset()))
}

type WorkspaceAllocationPlan = (BTreeMap<usize, u64>, BTreeMap<u64, u64>, u64);

fn allocate_workspace_intervals(
    intervals: &mut [WorkspaceInterval],
) -> Result<WorkspaceAllocationPlan, QwenExecutionError> {
    intervals.sort_by_key(|interval| (interval.first_node, interval.tensor_id));
    let mut slots: Vec<WorkspaceSlot> = Vec::new();
    let mut offsets = BTreeMap::new();
    let mut high_water_bytes = 0_u64;
    for interval in intervals {
        let reusable = slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| {
                slot.last_node < interval.first_node && slot.size_bytes >= interval.size_bytes
            })
            .min_by_key(|(_, slot)| slot.size_bytes)
            .map(|(index, _)| index);
        let slot_index = if let Some(index) = reusable {
            index
        } else {
            let offset_bytes = align_workspace(high_water_bytes)?;
            high_water_bytes = offset_bytes
                .checked_add(interval.size_bytes)
                .ok_or_else(|| {
                    QwenExecutionError::InvalidGraph(
                        "workspace arena high-water overflowed".to_owned(),
                    )
                })?;
            slots.push(WorkspaceSlot {
                offset_bytes,
                size_bytes: interval.size_bytes,
                last_node: interval.last_node,
            });
            slots.len() - 1
        };
        slots[slot_index].last_node = interval.last_node;
        if offsets
            .insert(interval.tensor_id, slots[slot_index].offset_bytes)
            .is_some()
        {
            return Err(QwenExecutionError::InvalidGraph(
                "workspace interval tensor ID is duplicated".to_owned(),
            ));
        }
    }
    let slot_sizes = slots
        .into_iter()
        .map(|slot| (slot.offset_bytes, slot.size_bytes))
        .collect();
    Ok((offsets, slot_sizes, high_water_bytes))
}

fn plan_workspace_arena(
    graph: &QwenGraph,
    dynamic_tensors: &[bool],
) -> Result<WorkspaceArenaLayout, QwenExecutionError> {
    if dynamic_tensors.len() != graph.tensor_metadata().len() {
        return Err(QwenExecutionError::InvalidGraph(
            "dynamic tensor table differs from graph metadata".to_owned(),
        ));
    }
    let terminal_outputs = terminal_output_tensor_ids(graph)?;
    let mut roots = Vec::with_capacity(graph.tensor_metadata().len());
    for tensor_id in 0..graph.tensor_metadata().len() {
        roots.push(workspace_root(graph, tensor_id)?);
    }

    for (tensor_id, tensor) in graph.tensor_metadata().iter().enumerate() {
        let root = roots[tensor_id];
        if dynamic_tensors[tensor_id]
            && tensor.view().end_offset() > graph.tensor_metadata()[root].view().end_offset()
        {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "dynamic alias tensor {} exceeds its owned workspace tensor",
                tensor.name()
            )));
        }
    }

    let mut first_nodes = vec![usize::MAX; graph.tensor_metadata().len()];
    let mut last_nodes = vec![0_usize; graph.tensor_metadata().len()];
    for (node_index, node) in graph.nodes().iter().enumerate() {
        for &tensor_id in node.inputs().iter().chain(node.outputs()) {
            let root = roots[tensor_id];
            if dynamic_tensors[root] {
                first_nodes[root] = first_nodes[root].min(node_index);
                last_nodes[root] = last_nodes[root].max(node_index);
            }
        }
    }
    let mut completion_boundaries = graph
        .nodes()
        .iter()
        .enumerate()
        .filter_map(|(node_index, node)| {
            matches!(node.kind(), QwenGraphNodeKind::FullKvAppend { .. }).then_some(node_index)
        })
        .collect::<Vec<_>>();
    let terminal_node = graph.nodes().len().checked_sub(1).ok_or_else(|| {
        QwenExecutionError::InvalidGraph("workspace graph has no completion boundary".to_owned())
    })?;
    if completion_boundaries.last().copied() != Some(terminal_node) {
        completion_boundaries.push(terminal_node);
    }

    let mut baseline_bytes = 0_u64;
    let mut intervals = Vec::new();
    for (tensor_id, tensor) in graph.tensor_metadata().iter().enumerate() {
        if !dynamic_tensors[tensor_id] || tensor.backing() != QwenGraphTensorBacking::Owned {
            continue;
        }
        let size_bytes = align_workspace(workspace_allocation_bytes(
            graph,
            tensor_id,
            terminal_outputs,
        )?)?;
        baseline_bytes = baseline_bytes.checked_add(size_bytes).ok_or_else(|| {
            QwenExecutionError::InvalidGraph("workspace baseline byte count overflowed".to_owned())
        })?;
        let first_node = if first_nodes[tensor_id] == usize::MAX {
            0
        } else {
            first_nodes[tensor_id]
        };
        let logical_last = last_nodes[tensor_id].max(first_node);
        let last_node = completion_boundaries
            .iter()
            .copied()
            .find(|boundary| *boundary >= logical_last)
            .ok_or_else(|| {
                QwenExecutionError::InvalidGraph(
                    "workspace tensor lifetime exceeds the terminal completion boundary".to_owned(),
                )
            })?;
        intervals.push(WorkspaceInterval {
            tensor_id,
            first_node,
            // Submission owners retain every buffer until their segment's
            // completion boundary. Reuse inside one still-pending segment
            // would violate the backend's buffer-level busy contract even
            // though the queued kernels are stream ordered.
            last_node,
            size_bytes,
        });
    }
    let mut root_offsets = vec![None; graph.tensor_metadata().len()];
    let (offsets, slot_sizes, high_water_bytes) = allocate_workspace_intervals(&mut intervals)?;
    for (tensor_id, offset) in offsets {
        root_offsets[tensor_id] = Some(offset);
    }

    let mut tensor_offsets = vec![None; graph.tensor_metadata().len()];
    for tensor_id in 0..graph.tensor_metadata().len() {
        if dynamic_tensors[tensor_id] {
            tensor_offsets[tensor_id] = root_offsets[roots[tensor_id]];
            if tensor_offsets[tensor_id].is_none() {
                return Err(QwenExecutionError::InvalidGraph(format!(
                    "dynamic tensor {} has no workspace arena slot",
                    graph.tensor_metadata()[tensor_id].name()
                )));
            }
        }
    }
    if high_water_bytes > baseline_bytes {
        return Err(QwenExecutionError::InvalidGraph(
            "workspace arena exceeds the separate-allocation baseline".to_owned(),
        ));
    }
    Ok(WorkspaceArenaLayout {
        tensor_offsets,
        slot_sizes,
        baseline_bytes,
        high_water_bytes,
    })
}

fn validate_graph_plan(
    graph: &QwenGraph,
    plan: &WeightLoadPlan,
) -> Result<GraphLayout, QwenExecutionError> {
    if graph.model_fingerprint() != plan.lock_fingerprint || graph.plan_digest() != plan.digest() {
        return Err(QwenExecutionError::InvalidGraph(
            "graph/load-plan identity or tied-embedding condition differs".to_owned(),
        ));
    }
    if graph.token_count() == 0 || graph.state_capacity() == 0 {
        return Err(QwenExecutionError::InvalidGraph(
            "graph has a zero token count or state capacity".to_owned(),
        ));
    }

    let mut tensor_ids = BTreeMap::new();
    for (index, tensor) in graph.tensor_metadata().iter().enumerate() {
        if tensor.id() != index || tensor_ids.insert(tensor.name().to_owned(), index).is_some() {
            return Err(QwenExecutionError::InvalidGraph(
                "graph tensor IDs or names are not one-to-one".to_owned(),
            ));
        }
    }

    let mut plan_entries = BTreeMap::<&str, &WeightLoadEntry>::new();
    for entry in &plan.entries {
        if plan_entries
            .insert(entry.tensor_name.as_str(), entry)
            .is_some()
        {
            return Err(QwenExecutionError::InvalidGraph(
                "weight plan tensor names are not one-to-one".to_owned(),
            ));
        }
    }
    let mut weight_tensor_ids = BTreeMap::new();
    let mut consumers = BTreeSet::new();
    for binding in graph.weight_bindings() {
        if binding.classification() != WeightClassification::Required
            || !consumers.insert(binding.consumer())
        {
            return Err(QwenExecutionError::InvalidGraph(
                "graph required weights are not one-to-one".to_owned(),
            ));
        }
        let entry = plan_entries.get(binding.tensor_name()).ok_or_else(|| {
            QwenExecutionError::InvalidGraph(format!(
                "graph weight is absent from the load plan: {}",
                binding.tensor_name()
            ))
        })?;
        if entry.classification != WeightClassification::Required
            || entry.consumer != Some(binding.consumer())
            || entry.dtype != binding.dtype()
            || entry.shape != binding.shape()
            || entry.source_range != binding.source_range()
            || entry.destination_start != Some(binding.destination_start())
        {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "graph weight metadata differs from load plan: {}",
                binding.tensor_name()
            )));
        }
        let tensor_id = *tensor_ids.get(binding.tensor_name()).ok_or_else(|| {
            QwenExecutionError::InvalidGraph(format!(
                "graph weight tensor is absent: {}",
                binding.tensor_name()
            ))
        })?;
        let tensor = &graph.tensor_metadata()[tensor_id];
        let source_dtype = tensor.view().dtype() == model_dtype(binding.dtype())?
            && tensor.view().encoding() == Encoding::Unquantized;
        let fp8_dtype = is_fp8_weight_view(tensor.view());
        let nvfp4_dtype = is_nvfp4_weight_view(tensor.view());
        if tensor.backing() != QwenGraphTensorBacking::Owned
            || (!source_dtype && !fp8_dtype && !nvfp4_dtype)
            || !shape_matches(tensor.view().shape(), binding.shape())?
        {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "graph weight tensor view differs from its binding: {}",
                binding.tensor_name()
            )));
        }
        if weight_tensor_ids
            .insert(binding.tensor_name().to_owned(), tensor_id)
            .is_some()
        {
            return Err(QwenExecutionError::InvalidGraph(
                "graph required weight names are not one-to-one".to_owned(),
            ));
        }
    }
    let plan_required = plan
        .entries
        .iter()
        .filter(|entry| entry.classification == WeightClassification::Required)
        .count();
    if plan_required != graph.weight_bindings().len()
        || weight_tensor_ids.len() != graph.weight_bindings().len()
    {
        return Err(QwenExecutionError::InvalidGraph(
            "graph/load-plan required weight coverage differs".to_owned(),
        ));
    }

    validate_output_projection_identity(graph, &weight_tensor_ids, plan.tied_embeddings)?;
    for node in graph.nodes() {
        for consumer in node.weight_consumers() {
            if !consumers.contains(consumer) {
                return Err(QwenExecutionError::InvalidGraph(format!(
                    "node {} references an unbound weight consumer",
                    node.label()
                )));
            }
        }
    }

    let mut dynamic_tensors = vec![false; graph.tensor_metadata().len()];
    let mut scales = Vec::new();
    let mut argmax_nodes = 0_usize;
    for (node_index, node) in graph.nodes().iter().enumerate() {
        if node
            .dependencies()
            .iter()
            .any(|dependency| *dependency >= node_index)
        {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "node {} has a non-prior dependency",
                node.label()
            )));
        }
        match node.kind() {
            QwenGraphNodeKind::AttentionScaleMaterialization {
                heads, head_dim, ..
            } => {
                if node.inputs().len() != 1 || node.outputs().len() != 1 {
                    return Err(QwenExecutionError::InvalidGraph(format!(
                        "scale node {} has the wrong arity",
                        node.label()
                    )));
                }
                scales.push(ScaleMaterialization {
                    raw_tensor_id: node.inputs()[0],
                    output_tensor_id: node.outputs()[0],
                    heads,
                    head_dim,
                });
            }
            QwenGraphNodeKind::Semantic(SemanticOpKind::Argmax) => {
                argmax_nodes = argmax_nodes.checked_add(1).ok_or_else(|| {
                    QwenExecutionError::InvalidGraph("argmax node count overflowed".to_owned())
                })?;
                mark_dynamic(&mut dynamic_tensors, node.inputs())?;
                mark_dynamic(&mut dynamic_tensors, node.outputs())?;
            }
            _ => {
                mark_dynamic(&mut dynamic_tensors, node.inputs())?;
                mark_dynamic(&mut dynamic_tensors, node.outputs())?;
            }
        }
    }
    if argmax_nodes != 1
        || !matches!(
            graph.nodes().last().map(QwenGraphNode::kind),
            Some(QwenGraphNodeKind::Semantic(SemanticOpKind::Argmax))
        )
    {
        return Err(QwenExecutionError::InvalidGraph(
            "graph must end in exactly one argmax node".to_owned(),
        ));
    }
    for &tensor_id in weight_tensor_ids.values() {
        dynamic_tensors[tensor_id] = false;
    }
    for scale in &scales {
        dynamic_tensors[scale.raw_tensor_id] = false;
        dynamic_tensors[scale.output_tensor_id] = false;
        validate_scale_metadata(graph, *scale)?;
    }
    let graph_token_count = usize::try_from(graph.token_count()).map_err(|_| {
        QwenExecutionError::InvalidGraph("graph token count does not fit usize".to_owned())
    })?;
    for (tensor_id, dynamic) in dynamic_tensors.iter().copied().enumerate() {
        if dynamic
            && graph.tensor_metadata()[tensor_id]
                .view()
                .shape()
                .first()
                .copied()
                != Some(graph_token_count)
        {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "dynamic tensor {} does not have graph token extent",
                graph.tensor_metadata()[tensor_id].name()
            )));
        }
    }

    let workspace = plan_workspace_arena(graph, &dynamic_tensors)?;
    Ok(GraphLayout {
        tensor_ids,
        _weight_tensor_ids: weight_tensor_ids,
        dynamic_tensors,
        scales,
        workspace,
    })
}

fn validate_output_projection_identity(
    graph: &QwenGraph,
    weight_tensor_ids: &BTreeMap<String, usize>,
    tied_embeddings: bool,
) -> Result<(), QwenExecutionError> {
    let embedding_role = if tied_embeddings {
        crate::WeightConsumer::EmbeddingAndTiedOutput
    } else {
        crate::WeightConsumer::Embedding
    };
    let output_role = if tied_embeddings {
        crate::WeightConsumer::EmbeddingAndTiedOutput
    } else {
        crate::WeightConsumer::OutputProjection
    };
    let embedding_binding = graph
        .weight_bindings()
        .iter()
        .find(|binding| {
            binding.consumer().layer.is_none() && binding.consumer().role == embedding_role
        })
        .ok_or_else(|| {
            QwenExecutionError::InvalidGraph("embedding binding is absent".to_owned())
        })?;
    let output_binding = graph
        .weight_bindings()
        .iter()
        .find(|binding| {
            binding.consumer().layer.is_none() && binding.consumer().role == output_role
        })
        .ok_or_else(|| {
            QwenExecutionError::InvalidGraph("output projection binding is absent".to_owned())
        })?;
    let embedding_id = *weight_tensor_ids
        .get(embedding_binding.tensor_name())
        .ok_or_else(|| QwenExecutionError::InvalidGraph("embedding tensor is absent".to_owned()))?;
    let output_id = *weight_tensor_ids
        .get(output_binding.tensor_name())
        .ok_or_else(|| {
            QwenExecutionError::InvalidGraph("output projection tensor is absent".to_owned())
        })?;
    let embedding = graph
        .nodes()
        .iter()
        .find(|node| {
            matches!(
                node.kind(),
                QwenGraphNodeKind::Semantic(SemanticOpKind::Embedding)
            )
        })
        .ok_or_else(|| QwenExecutionError::InvalidGraph("embedding node is absent".to_owned()))?;
    let output = graph
        .nodes()
        .iter()
        .find(|node| {
            node.label()
                == if tied_embeddings {
                    "tied_lm_head_matmul"
                } else {
                    "lm_head_matmul"
                }
        })
        .ok_or_else(|| {
            QwenExecutionError::InvalidGraph("output projection node is absent".to_owned())
        })?;
    if embedding.inputs().first() != Some(&embedding_id)
        || output.inputs().get(1) != Some(&output_id)
        || (tied_embeddings && embedding_id != output_id)
        || (!tied_embeddings && embedding_id == output_id)
    {
        return Err(QwenExecutionError::InvalidGraph(
            "embedding/output projection alias contract differs from the model".to_owned(),
        ));
    }
    Ok(())
}

fn validate_scale_metadata(
    graph: &QwenGraph,
    scale: ScaleMaterialization,
) -> Result<(), QwenExecutionError> {
    let raw = graph
        .tensor_metadata()
        .get(scale.raw_tensor_id)
        .ok_or_else(|| QwenExecutionError::InvalidGraph("scale raw tensor is absent".to_owned()))?;
    let output = graph
        .tensor_metadata()
        .get(scale.output_tensor_id)
        .ok_or_else(|| {
            QwenExecutionError::InvalidGraph("scale output tensor is absent".to_owned())
        })?;
    let heads = usize::try_from(scale.heads).map_err(|_| {
        QwenExecutionError::InvalidGraph("scale head count does not fit usize".to_owned())
    })?;
    if scale.head_dim != 256
        || raw.view().dtype() != DType::Bf16
        || output.view().dtype() != DType::Bf16
        || raw.view().shape() != [256]
        || output.view().shape() != [heads, 256]
        || raw.view().payload_bytes() != 512
        || output.view().payload_bytes()
            != 512_u64.checked_mul(u64::from(scale.heads)).ok_or_else(|| {
                QwenExecutionError::InvalidGraph("scale byte count overflowed".to_owned())
            })?
    {
        return Err(QwenExecutionError::InvalidGraph(
            "attention scale materialization does not have the fixed BF16 shape".to_owned(),
        ));
    }
    Ok(())
}

fn mark_dynamic(dynamic: &mut [bool], tensor_ids: &[usize]) -> Result<(), QwenExecutionError> {
    for &tensor_id in tensor_ids {
        let entry = dynamic.get_mut(tensor_id).ok_or_else(|| {
            QwenExecutionError::InvalidGraph("node references an absent tensor".to_owned())
        })?;
        *entry = true;
    }
    Ok(())
}

fn model_dtype(dtype: TensorDType) -> Result<DType, QwenExecutionError> {
    match dtype {
        TensorDType::Bf16 => Ok(DType::Bf16),
        TensorDType::F16 => Ok(DType::F16),
        TensorDType::F32 => Ok(DType::F32),
        TensorDType::I32 => Ok(DType::I32),
        TensorDType::U8 => Ok(DType::U8),
        TensorDType::I64 => Err(QwenExecutionError::InvalidGraph(
            "Qwen D1 does not accept I64 graph tensor bindings".to_owned(),
        )),
    }
}

fn shape_matches(shape: &[usize], expected: &[u64]) -> Result<bool, QwenExecutionError> {
    if shape.len() != expected.len() {
        return Ok(false);
    }
    shape
        .iter()
        .zip(expected)
        .try_fold(true, |matches, (&actual, &expected)| {
            Ok(matches
                && u64::try_from(actual).map_err(|_| {
                    QwenExecutionError::InvalidGraph("tensor shape does not fit u64".to_owned())
                })? == expected)
        })
}

fn is_fp8_weight_view(view: &TensorView) -> bool {
    matches!(view.dtype(), DType::F8E4M3Fn | DType::F8E4M3FnuZ)
        && view.encoding()
            == Encoding::Fp8Scaled {
                granularity: Fp8ScaleGranularity::OuterDimension,
                scale_dtype: DType::F32,
                resident: Fp8ResidentRepresentation::PackedBytes,
            }
        && view.shape().len() == 2
}

fn is_nvfp4_weight_view(view: &TensorView) -> bool {
    view.dtype() == DType::U8
        && view.encoding()
            == Encoding::Nvfp4 {
                block_size: 16,
                scale_dtype: DType::F8E4M3Fn,
            }
        && view.shape().len() == 2
}

fn resident_weight_bytes(view: &TensorView) -> Result<u64, QwenExecutionError> {
    if is_nvfp4_weight_view(view) {
        let rows = u64::try_from(view.shape()[0]).map_err(|_| {
            QwenExecutionError::InvalidGraph("NVFP4 row count does not fit u64".to_owned())
        })?;
        let columns = u64::try_from(view.shape()[1]).map_err(|_| {
            QwenExecutionError::InvalidGraph("NVFP4 column count does not fit u64".to_owned())
        })?;
        let blocks = rows.checked_mul(columns.div_ceil(16)).ok_or_else(|| {
            QwenExecutionError::InvalidGraph("NVFP4 block count overflowed".to_owned())
        })?;
        let unaligned = view.payload_bytes().checked_add(blocks).ok_or_else(|| {
            QwenExecutionError::InvalidGraph("NVFP4 resident bytes overflowed".to_owned())
        })?;
        return unaligned
            .checked_add(3)
            .map(|bytes| bytes & !3)
            .and_then(|bytes| bytes.checked_add(4))
            .ok_or_else(|| {
                QwenExecutionError::InvalidGraph("NVFP4 resident bytes overflowed".to_owned())
            });
    }
    if !is_fp8_weight_view(view) {
        return Ok(view.payload_bytes());
    }
    let rows = u64::try_from(view.shape()[0]).map_err(|_| {
        QwenExecutionError::InvalidGraph("FP8 weight row count does not fit u64".to_owned())
    })?;
    view.payload_bytes()
        .checked_add(rows.checked_mul(4).ok_or_else(|| {
            QwenExecutionError::InvalidGraph("FP8 scale byte count overflowed".to_owned())
        })?)
        .ok_or_else(|| {
            QwenExecutionError::InvalidGraph("FP8 resident byte count overflowed".to_owned())
        })
}

fn allocate_resident_tensors(
    session: &ExecutionSession,
    graph: &QwenGraph,
    layout: &GraphLayout,
) -> Result<BTreeMap<String, TensorAllocation>, QwenExecutionError> {
    let mut allocations = BTreeMap::new();
    for tensor in graph.tensor_metadata() {
        if !layout.dynamic_tensors[tensor.id()] && tensor.backing() == QwenGraphTensorBacking::Owned
        {
            let buffer = session.allocate_with_category(
                tensor
                    .view()
                    .byte_offset()
                    .checked_add(resident_weight_bytes(tensor.view())?)
                    .ok_or_else(|| {
                        QwenExecutionError::InvalidGraph(
                            "resident allocation size overflowed".to_owned(),
                        )
                    })?,
                crate::AllocationCategory::ModelResident,
            )?;
            allocations.insert(
                tensor.name().to_owned(),
                TensorAllocation {
                    buffer,
                    graph_view: tensor.view().clone(),
                },
            );
        }
    }
    Ok(allocations)
}

fn allocate_request_tensors(
    session: &ExecutionSession,
    graph: &QwenGraph,
    layout: &GraphLayout,
    resident_tensors: &BTreeMap<String, TensorAllocation>,
) -> Result<Vec<TensorAllocation>, QwenExecutionError> {
    let workspace = layout
        .workspace
        .slot_sizes
        .iter()
        .map(|(&offset, &size)| {
            session
                .allocate_with_category(size, crate::AllocationCategory::Workspace)
                .map(|buffer| (offset, buffer))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let mut allocations: Vec<TensorAllocation> = Vec::with_capacity(graph.tensor_metadata().len());
    for tensor in graph.tensor_metadata() {
        let (buffer, graph_view) = match tensor.backing() {
            QwenGraphTensorBacking::Owned => {
                if layout.dynamic_tensors[tensor.id()] {
                    let offset = layout.workspace.tensor_offsets[tensor.id()].ok_or_else(|| {
                        QwenExecutionError::InvalidGraph(format!(
                            "dynamic tensor {} has no workspace arena offset",
                            tensor.name()
                        ))
                    })?;
                    (
                        workspace
                            .get(&offset)
                            .ok_or_else(|| {
                                QwenExecutionError::InvalidGraph(
                                    "dynamic tensor requires an absent workspace slot".to_owned(),
                                )
                            })?
                            .clone(),
                        tensor.view().clone(),
                    )
                } else {
                    let resident = resident_tensors.get(tensor.name()).ok_or_else(|| {
                        QwenExecutionError::InvalidGraph(format!(
                            "resident tensor is absent: {}",
                            tensor.name()
                        ))
                    })?;
                    (resident.buffer.clone(), tensor.view().clone())
                }
            }
            QwenGraphTensorBacking::Alias { tensor_id } => {
                let source = allocations.get(tensor_id).ok_or_else(|| {
                    QwenExecutionError::InvalidGraph(format!(
                        "alias tensor {} precedes its backing tensor",
                        tensor.name()
                    ))
                })?;
                (source.buffer.clone(), tensor.view().clone())
            }
        };
        allocations.push(TensorAllocation { buffer, graph_view });
    }
    Ok(allocations)
}

fn validate_resident_graph(
    graph: &QwenGraph,
    layout: &GraphLayout,
    resident_tensors: &BTreeMap<String, TensorAllocation>,
    resident_scales: &BTreeMap<String, CachedScale>,
) -> Result<(), QwenExecutionError> {
    for (tensor_id, tensor) in graph.tensor_metadata().iter().enumerate() {
        if !layout.dynamic_tensors[tensor_id] && tensor.backing() == QwenGraphTensorBacking::Owned {
            let resident = resident_tensors.get(tensor.name()).ok_or_else(|| {
                QwenExecutionError::InvalidGraph(format!(
                    "request graph requires a model tensor that is not resident: {}",
                    tensor.name()
                ))
            })?;
            if resident.graph_view.dtype() != tensor.view().dtype()
                || resident.graph_view.encoding() != tensor.view().encoding()
                || resident.graph_view.shape() != tensor.view().shape()
                || resident.graph_view.strides() != tensor.view().strides()
                || resident.graph_view.byte_offset() != tensor.view().byte_offset()
            {
                return Err(QwenExecutionError::InvalidGraph(format!(
                    "request graph model tensor binding differs from resident: {}",
                    tensor.name()
                )));
            }
        }
    }
    for node in graph.nodes() {
        if let QwenGraphNodeKind::AttentionScaleMaterialization { .. } = node.kind() {
            let output = graph
                .tensor_metadata()
                .get(node.outputs()[0])
                .ok_or_else(|| {
                    QwenExecutionError::InvalidGraph("resident scale output is absent".to_owned())
                })?;
            if !resident_scales.contains_key(output.name()) {
                return Err(QwenExecutionError::InvalidGraph(format!(
                    "request graph attention scale is not resident: {}",
                    output.name()
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn allocate_tensors(
    session: &ExecutionSession,
    graph: &QwenGraph,
) -> Result<Vec<TensorAllocation>, QwenExecutionError> {
    let terminal_outputs = terminal_output_tensor_ids(graph)?;
    let mut allocations: Vec<TensorAllocation> = Vec::with_capacity(graph.tensor_metadata().len());
    for tensor in graph.tensor_metadata() {
        let buffer = match tensor.backing() {
            QwenGraphTensorBacking::Owned => {
                let allocation_end = compact_terminal_allocation_end(
                    graph.token_count(),
                    graph.is_mtp(),
                    tensor.id(),
                    terminal_outputs,
                    tensor.view(),
                )?
                .unwrap_or(
                    tensor
                        .view()
                        .byte_offset()
                        .checked_add(resident_weight_bytes(tensor.view())?)
                        .ok_or_else(|| {
                            QwenExecutionError::InvalidGraph(
                                "test tensor allocation size overflowed".to_owned(),
                            )
                        })?,
                );
                session.allocate(allocation_end)?
            }
            QwenGraphTensorBacking::Alias { tensor_id } => {
                let source = allocations.get(tensor_id).ok_or_else(|| {
                    QwenExecutionError::InvalidGraph(format!(
                        "alias tensor {} precedes its backing tensor",
                        tensor.name()
                    ))
                })?;
                if source.graph_view.dtype() != tensor.view().dtype()
                    || source.graph_view.encoding() != tensor.view().encoding()
                    || tensor.view().end_offset() > source.buffer.size_bytes()
                {
                    return Err(QwenExecutionError::InvalidGraph(format!(
                        "alias tensor {} does not match its backing allocation",
                        tensor.name()
                    )));
                }
                source.buffer.clone()
            }
        };
        allocations.push(TensorAllocation {
            buffer,
            graph_view: tensor.view().clone(),
        });
    }
    Ok(allocations)
}

fn provision_resident_scales<S: QwenProvisionSource>(
    source: &S,
    session: &ExecutionSession,
    queue: &ExecutionQueue,
    graph: &QwenGraph,
    tensors: &BTreeMap<String, TensorAllocation>,
    scales: &[ScaleMaterialization],
    completion_timeout: Duration,
) -> Result<BTreeMap<usize, CachedScale>, QwenExecutionError> {
    let mut raw_cache = BTreeMap::<usize, Arc<[u8]>>::new();
    let mut result = BTreeMap::new();
    for scale in scales {
        if result.contains_key(&scale.output_tensor_id) {
            return Err(QwenExecutionError::InvalidGraph(
                "scale materialization output occurs more than once".to_owned(),
            ));
        }
        let raw_tensor = graph
            .tensor_metadata()
            .get(scale.raw_tensor_id)
            .ok_or_else(|| {
                QwenExecutionError::InvalidGraph("scale raw tensor is absent".to_owned())
            })?;
        let output = graph
            .tensor_metadata()
            .get(scale.output_tensor_id)
            .ok_or_else(|| {
                QwenExecutionError::InvalidGraph("scale output tensor is absent".to_owned())
            })?;
        let output_allocation = tensors.get(output.name()).ok_or_else(|| {
            QwenExecutionError::InvalidGraph(format!(
                "resident scale output allocation is absent: {}",
                output.name()
            ))
        })?;
        let raw = match raw_cache.get(&scale.raw_tensor_id) {
            Some(bytes) => Arc::clone(bytes),
            None => {
                let bytes = source.read_scale_bytes(raw_tensor.name(), 512)?;
                if bytes.len() != 512 {
                    return Err(QwenExecutionError::InvalidRequest(format!(
                        "attention scale {} is not exactly 256 BF16 values",
                        raw_tensor.name()
                    )));
                }
                raw_cache.insert(scale.raw_tensor_id, Arc::clone(&bytes));
                bytes
            }
        };
        let repeat = usize::try_from(scale.heads).map_err(|_| {
            QwenExecutionError::InvalidGraph("scale head count does not fit usize".to_owned())
        })?;
        let expected = raw.len().checked_mul(repeat).ok_or_else(|| {
            QwenExecutionError::InvalidGraph("expanded scale byte length overflowed".to_owned())
        })?;
        let mut expanded = Vec::with_capacity(expected);
        for _ in 0..repeat {
            expanded.extend_from_slice(&raw);
        }
        if expanded.len() != expected
            || u64::try_from(expected).map_err(|_| {
                QwenExecutionError::InvalidGraph(
                    "expanded scale byte length does not fit u64".to_owned(),
                )
            })? != output.view().payload_bytes()
        {
            return Err(QwenExecutionError::InvalidGraph(
                "expanded scale bytes do not exactly match the output tensor".to_owned(),
            ));
        }
        let expanded: Arc<[u8]> = Arc::from(expanded);
        upload_exact_bytes(
            session,
            queue,
            &output_allocation.buffer,
            &output_allocation.graph_view,
            &expanded,
            completion_timeout,
            "attention scale upload",
        )?;
        result.insert(
            scale.output_tensor_id,
            CachedScale {
                raw_tensor_id: scale.raw_tensor_id,
                raw_bytes: raw,
                expanded_bytes: expanded,
            },
        );
    }
    Ok(result)
}

#[cfg(test)]
fn provision_scales<S: QwenProvisionSource>(
    source: &S,
    session: &ExecutionSession,
    queue: &ExecutionQueue,
    graph: &QwenGraph,
    tensors: &[TensorAllocation],
    scales: &[ScaleMaterialization],
    completion_timeout: Duration,
) -> Result<BTreeMap<usize, CachedScale>, QwenExecutionError> {
    let mut raw_cache = BTreeMap::<usize, Arc<[u8]>>::new();
    let mut result = BTreeMap::new();
    for scale in scales {
        if result.contains_key(&scale.output_tensor_id) {
            return Err(QwenExecutionError::InvalidGraph(
                "scale materialization output occurs more than once".to_owned(),
            ));
        }
        let raw_tensor = graph
            .tensor_metadata()
            .get(scale.raw_tensor_id)
            .ok_or_else(|| {
                QwenExecutionError::InvalidGraph("scale raw tensor is absent".to_owned())
            })?;
        let output = tensors.get(scale.output_tensor_id).ok_or_else(|| {
            QwenExecutionError::InvalidGraph("scale output allocation is absent".to_owned())
        })?;
        let raw = match raw_cache.get(&scale.raw_tensor_id) {
            Some(bytes) => Arc::clone(bytes),
            None => {
                let bytes = source.read_scale_bytes(raw_tensor.name(), 512)?;
                if bytes.len() != 512 {
                    return Err(QwenExecutionError::InvalidRequest(format!(
                        "attention scale {} is not exactly 256 BF16 values",
                        raw_tensor.name()
                    )));
                }
                raw_cache.insert(scale.raw_tensor_id, Arc::clone(&bytes));
                bytes
            }
        };
        let repeat = usize::try_from(scale.heads).map_err(|_| {
            QwenExecutionError::InvalidGraph("scale head count does not fit usize".to_owned())
        })?;
        let expected = raw.len().checked_mul(repeat).ok_or_else(|| {
            QwenExecutionError::InvalidGraph("expanded scale byte length overflowed".to_owned())
        })?;
        let mut expanded = Vec::with_capacity(expected);
        for _ in 0..repeat {
            expanded.extend_from_slice(&raw);
        }
        if expanded.len() != expected
            || u64::try_from(expected).map_err(|_| {
                QwenExecutionError::InvalidGraph(
                    "expanded scale byte length does not fit u64".to_owned(),
                )
            })? != output.graph_view.payload_bytes()
        {
            return Err(QwenExecutionError::InvalidGraph(
                "expanded scale bytes do not exactly match the output tensor".to_owned(),
            ));
        }
        let expanded: Arc<[u8]> = Arc::from(expanded);
        upload_exact_bytes(
            session,
            queue,
            &output.buffer,
            &output.graph_view,
            &expanded,
            completion_timeout,
            "attention scale upload",
        )?;
        result.insert(
            scale.output_tensor_id,
            CachedScale {
                raw_tensor_id: scale.raw_tensor_id,
                raw_bytes: raw,
                expanded_bytes: expanded,
            },
        );
    }
    Ok(result)
}

fn create_states(
    session: &ExecutionSession,
    graph: &QwenGraph,
) -> Result<StateMaps, QwenExecutionError> {
    let mut kv_states: BTreeMap<u32, KvState> = BTreeMap::new();
    let mut linear_states: BTreeMap<u32, LinearAttentionState> = BTreeMap::new();
    for state in graph.states() {
        match state.descriptor() {
            QwenGraphStateDescriptor::Kv(descriptor) => {
                if !matches!(
                    state.kind(),
                    crate::QwenGraphStateKind::FullKey | crate::QwenGraphStateKind::FullValue
                ) || descriptor.layer_id() != state.layer()
                {
                    return Err(QwenExecutionError::InvalidGraph(
                        "full-attention state metadata is malformed".to_owned(),
                    ));
                }
                match kv_states.get(&state.layer()) {
                    Some(existing) if existing.descriptor() == descriptor => {}
                    Some(_) => {
                        return Err(QwenExecutionError::InvalidGraph(
                            "one full-attention layer has conflicting state descriptors".to_owned(),
                        ));
                    }
                    None => {
                        let created = session.create_kv_state(descriptor)?;
                        if created.snapshot(session)?.length() != 0 {
                            return Err(QwenExecutionError::InvalidGraph(
                                "new KV state is not empty".to_owned(),
                            ));
                        }
                        kv_states.insert(state.layer(), created);
                    }
                }
            }
            QwenGraphStateDescriptor::Linear(descriptor) => {
                if !matches!(
                    state.kind(),
                    crate::QwenGraphStateKind::LinearConvolution
                        | crate::QwenGraphStateKind::LinearRecurrent
                ) || descriptor.layer_id() != state.layer()
                {
                    return Err(QwenExecutionError::InvalidGraph(
                        "linear-attention state metadata is malformed".to_owned(),
                    ));
                }
                match linear_states.get(&state.layer()) {
                    Some(existing) if existing.descriptor() == descriptor => {}
                    Some(_) => {
                        return Err(QwenExecutionError::InvalidGraph(
                            "one linear-attention layer has conflicting state descriptors"
                                .to_owned(),
                        ));
                    }
                    None => {
                        let created = session.create_linear_attention_state(descriptor)?;
                        if created.snapshot(session)?.length() != 0 {
                            return Err(QwenExecutionError::InvalidGraph(
                                "new linear-attention state is not empty".to_owned(),
                            ));
                        }
                        linear_states.insert(state.layer(), created);
                    }
                }
            }
        }
    }
    let full_layers: BTreeSet<u32> = if graph.is_mtp() {
        BTreeSet::from([crate::weights::QWEN35_MTP_CONSUMER_LAYER as u32])
    } else {
        graph
            .layer_types()
            .iter()
            .enumerate()
            .filter_map(|(layer, ty)| {
                (*ty == crate::LayerType::FullAttention).then_some(layer as u32)
            })
            .collect()
    };
    let linear_layers: BTreeSet<u32> = if graph.is_mtp() {
        BTreeSet::new()
    } else {
        graph
            .layer_types()
            .iter()
            .enumerate()
            .filter_map(|(layer, ty)| {
                (*ty == crate::LayerType::LinearAttention).then_some(layer as u32)
            })
            .collect()
    };
    if kv_states.keys().copied().collect::<BTreeSet<_>>() != full_layers
        || linear_states.keys().copied().collect::<BTreeSet<_>>() != linear_layers
    {
        return Err(QwenExecutionError::InvalidGraph(
            "graph state coverage does not match the explicit layer schedule".to_owned(),
        ));
    }
    Ok((kv_states, linear_states))
}

fn runtime_view(
    graph_view: &TensorView,
    graph_token_count: u64,
    token_count: u64,
) -> Result<TensorView, QwenExecutionError> {
    if token_count == 0 {
        return Err(QwenExecutionError::InvalidRequest(
            "runtime tensor view requires a non-zero token count".to_owned(),
        ));
    }
    let graph_tokens = usize::try_from(graph_token_count).map_err(|_| {
        QwenExecutionError::InvalidGraph("graph token count does not fit usize".to_owned())
    })?;
    let runtime_tokens = usize::try_from(token_count).map_err(|_| {
        QwenExecutionError::InvalidRequest("runtime token count does not fit usize".to_owned())
    })?;
    let mut shape = graph_view.shape().to_vec();
    if shape.first().copied() != Some(graph_tokens) {
        return Err(QwenExecutionError::InvalidGraph(
            "dynamic graph tensor does not start with the graph token count".to_owned(),
        ));
    }
    shape[0] = runtime_tokens;
    let strides = contiguous_strides(&shape)?;
    Ok(TensorView::new(
        graph_view.dtype(),
        graph_view.encoding(),
        &shape,
        &strides,
        graph_view.byte_offset(),
    )?)
}

fn first_row_view(view: &TensorView) -> Result<TensorView, QwenExecutionError> {
    row_view(view, 0)
}

fn terminal_output_tensor_ids(graph: &QwenGraph) -> Result<[usize; 2], QwenExecutionError> {
    let argmax = graph.nodes().last().ok_or_else(|| {
        QwenExecutionError::InvalidGraph("graph has no terminal Argmax node".to_owned())
    })?;
    if !matches!(
        argmax.kind(),
        QwenGraphNodeKind::Semantic(SemanticOpKind::Argmax)
    ) || argmax.inputs().len() != 1
        || argmax.outputs().len() != 1
    {
        return Err(QwenExecutionError::InvalidGraph(
            "terminal Argmax does not have one logits input and one output".to_owned(),
        ));
    }
    Ok([argmax.inputs()[0], argmax.outputs()[0]])
}

fn compact_terminal_allocation_end(
    graph_token_count: u64,
    is_mtp: bool,
    tensor_id: usize,
    terminal_outputs: [usize; 2],
    view: &TensorView,
) -> Result<Option<u64>, QwenExecutionError> {
    if graph_token_count < TERMINAL_ROW_MIN_TOKENS
        || is_mtp
        || !terminal_outputs.contains(&tensor_id)
    {
        return Ok(None);
    }
    Ok(Some(first_row_view(view)?.end_offset()))
}

fn last_row_view(view: &TensorView) -> Result<TensorView, QwenExecutionError> {
    let rows = view.shape().first().copied().ok_or_else(|| {
        QwenExecutionError::InvalidGraph("terminal tensor is not row-shaped".to_owned())
    })?;
    row_view(
        view,
        rows.checked_sub(1).ok_or_else(|| {
            QwenExecutionError::InvalidRequest("terminal tensor has zero rows".to_owned())
        })?,
    )
}

fn row_view(view: &TensorView, row: usize) -> Result<TensorView, QwenExecutionError> {
    let rows = view.shape().first().copied().ok_or_else(|| {
        QwenExecutionError::InvalidGraph("terminal tensor is not row-shaped".to_owned())
    })?;
    if rows == 0 || row >= rows || !(1..=2).contains(&view.shape().len()) || !view.is_contiguous() {
        return Err(QwenExecutionError::InvalidGraph(
            "terminal tensor row view is empty, out of range, unsupported-rank, or non-contiguous"
                .to_owned(),
        ));
    }
    let row_count = u64::try_from(rows).map_err(|_| {
        QwenExecutionError::InvalidGraph("terminal row count does not fit u64".to_owned())
    })?;
    if view.payload_bytes() % row_count != 0 {
        return Err(QwenExecutionError::InvalidGraph(
            "terminal tensor payload is not row divisible".to_owned(),
        ));
    }
    let row_bytes = view.payload_bytes() / row_count;
    let row_offset = u64::try_from(row)
        .ok()
        .and_then(|index| index.checked_mul(row_bytes))
        .and_then(|offset| view.byte_offset().checked_add(offset))
        .ok_or_else(|| {
            QwenExecutionError::InvalidGraph("terminal row offset overflowed".to_owned())
        })?;
    let mut shape = view.shape().to_vec();
    shape[0] = 1;
    Ok(TensorView::new(
        view.dtype(),
        view.encoding(),
        &shape,
        view.strides(),
        row_offset,
    )?)
}

fn contiguous_strides(shape: &[usize]) -> Result<Vec<usize>, QwenExecutionError> {
    let mut strides = vec![0_usize; shape.len()];
    let mut stride = 1_usize;
    for (&dimension, slot) in shape.iter().zip(strides.iter_mut()).rev() {
        *slot = stride;
        stride = stride.checked_mul(dimension).ok_or_else(|| {
            QwenExecutionError::InvalidGraph("runtime tensor stride overflowed".to_owned())
        })?;
    }
    Ok(strides)
}

fn upload_exact_bytes(
    session: &ExecutionSession,
    queue: &ExecutionQueue,
    buffer: &ExecutionBuffer,
    view: &TensorView,
    bytes: &[u8],
    completion_timeout: Duration,
    stage: &str,
) -> Result<(), QwenExecutionError> {
    if view.payload_bytes()
        != u64::try_from(bytes.len()).map_err(|_| {
            QwenExecutionError::InvalidRequest("upload byte length does not fit u64".to_owned())
        })?
        || bytes.is_empty()
    {
        return Err(QwenExecutionError::InvalidRequest(format!(
            "{stage} bytes do not exactly match the tensor view"
        )));
    }
    let maximum = usize::try_from(session.max_transfer_bytes()?).unwrap_or(usize::MAX);
    if maximum == 0 {
        return Err(QwenExecutionError::InvalidRequest(
            "backend transfer limit must be non-zero".to_owned(),
        ));
    }
    let mut offset = 0_usize;
    while offset < bytes.len() {
        let remaining = bytes.len() - offset;
        let length = remaining.min(maximum);
        let relative = u64::try_from(offset).map_err(|_| {
            QwenExecutionError::InvalidRequest("upload offset does not fit u64".to_owned())
        })?;
        let destination_offset = view.byte_offset().checked_add(relative).ok_or_else(|| {
            QwenExecutionError::InvalidRequest("upload destination offset overflowed".to_owned())
        })?;
        let range = buffer.range(
            destination_offset,
            u64::try_from(length).map_err(|_| {
                QwenExecutionError::InvalidRequest(
                    "upload chunk length does not fit u64".to_owned(),
                )
            })?,
        )?;
        let mut transfer = session.upload(
            queue,
            range,
            Arc::from(bytes[offset..offset + length].to_vec()),
        )?;
        require_terminal_success(stage, transfer.wait(completion_timeout)?)?;
        offset = offset.checked_add(length).ok_or_else(|| {
            QwenExecutionError::InvalidRequest("upload offset overflowed".to_owned())
        })?;
    }
    Ok(())
}

fn read_exact_bytes(
    session: &ExecutionSession,
    queue: &ExecutionQueue,
    buffer: &ExecutionBuffer,
    view: &TensorView,
    timeout: Duration,
    stage: &str,
) -> Result<Vec<u8>, QwenExecutionError> {
    let total = view.payload_bytes();
    let maximum = session.max_transfer_bytes()?;
    if total == 0 {
        return Err(QwenExecutionError::InvalidRequest(format!(
            "{stage} source is empty"
        )));
    }
    if maximum == 0 {
        return Err(QwenExecutionError::InvalidRequest(
            "backend transfer limit must be non-zero".to_owned(),
        ));
    }
    let total_usize = usize::try_from(total).map_err(|_| {
        QwenExecutionError::InvalidRequest(format!("{stage} byte count does not fit usize"))
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve(total_usize).map_err(|_| {
        QwenExecutionError::InvalidRequest(format!("{stage} allocation is too large"))
    })?;
    let mut relative = 0_u64;
    while relative < total {
        let length = (total - relative).min(maximum);
        let offset = view.byte_offset().checked_add(relative).ok_or_else(|| {
            QwenExecutionError::InvalidRequest(format!("{stage} offset overflowed"))
        })?;
        let mut transfer = session.readback(queue, buffer.range(offset, length)?)?;
        require_terminal_success(stage, transfer.wait(timeout)?)?;
        let start = bytes.len();
        bytes.resize(
            start
                + usize::try_from(length).map_err(|_| {
                    QwenExecutionError::InvalidRequest(format!(
                        "{stage} chunk length does not fit usize"
                    ))
                })?,
            0,
        );
        let copied = transfer.read_into(&mut bytes[start..])?;
        if copied != length {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "{stage} returned a short or long read"
            )));
        }
        relative = relative.checked_add(length).ok_or_else(|| {
            QwenExecutionError::InvalidRequest(format!("{stage} progress overflowed"))
        })?;
    }
    Ok(bytes)
}

fn upload_buffer_bytes(
    session: &ExecutionSession,
    queue: &ExecutionQueue,
    destination: &crate::BufferRange,
    bytes: &[u8],
    completion_timeout: Duration,
    stage: &str,
) -> Result<(), QwenExecutionError> {
    if bytes.is_empty() || u64::try_from(bytes.len()).ok() != Some(destination.size_bytes()) {
        return Err(QwenExecutionError::InvalidRequest(format!(
            "{stage} bytes do not exactly match the destination"
        )));
    }
    let maximum = usize::try_from(session.max_transfer_bytes()?).unwrap_or(usize::MAX);
    if maximum == 0 {
        return Err(QwenExecutionError::InvalidRequest(
            "backend transfer limit must be non-zero".to_owned(),
        ));
    }
    let mut offset = 0_usize;
    while offset < bytes.len() {
        let length = (bytes.len() - offset).min(maximum);
        let absolute = destination
            .offset_bytes()
            .checked_add(u64::try_from(offset).map_err(|_| {
                QwenExecutionError::InvalidRequest("FP8 upload offset does not fit u64".to_owned())
            })?)
            .ok_or_else(|| {
                QwenExecutionError::InvalidRequest("FP8 upload offset overflowed".to_owned())
            })?;
        let range = destination.buffer().range(
            absolute,
            u64::try_from(length).map_err(|_| {
                QwenExecutionError::InvalidRequest("FP8 upload length does not fit u64".to_owned())
            })?,
        )?;
        let mut transfer = session.upload(
            queue,
            range,
            Arc::from(bytes[offset..offset + length].to_vec()),
        )?;
        require_terminal_success(stage, transfer.wait(completion_timeout)?)?;
        offset = offset.checked_add(length).ok_or_else(|| {
            QwenExecutionError::InvalidRequest("FP8 upload offset overflowed".to_owned())
        })?;
    }
    Ok(())
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(std::mem::size_of::<i32>()));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn bf16_from_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff_u32 + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding)) >> 16) as u16
}

fn bf16_scalar_from_f32(value: f32) -> Result<u16, QwenExecutionError> {
    if !value.is_finite() {
        return Err(QwenExecutionError::InvalidRequest(
            "adapter scalar is non-finite".to_owned(),
        ));
    }
    let bits = bf16_from_f32(value);
    if bits & 0x7f80 == 0x7f80 {
        return Err(QwenExecutionError::InvalidRequest(
            "adapter scalar overflows BF16".to_owned(),
        ));
    }
    Ok(bits)
}

fn position_bytes(start_position: u64, token_count: u64) -> Result<Vec<u8>, QwenExecutionError> {
    let count = usize::try_from(token_count).map_err(|_| {
        QwenExecutionError::InvalidRequest("position token count does not fit usize".to_owned())
    })?;
    let mut positions = Vec::with_capacity(count.saturating_mul(std::mem::size_of::<i32>()));
    for relative in 0..token_count {
        let position = start_position.checked_add(relative).ok_or_else(|| {
            QwenExecutionError::InvalidRequest("position overflowed u64".to_owned())
        })?;
        let position = i32::try_from(position).map_err(|_| {
            QwenExecutionError::InvalidRequest(
                "position does not fit the I32 input contract".to_owned(),
            )
        })?;
        positions.extend_from_slice(&position.to_le_bytes());
    }
    Ok(positions)
}

fn position_values_bytes(positions: &[u64]) -> Result<Vec<u8>, QwenExecutionError> {
    let mut bytes = Vec::with_capacity(positions.len().saturating_mul(std::mem::size_of::<i32>()));
    for position in positions {
        let position = i32::try_from(*position).map_err(|_| {
            QwenExecutionError::InvalidRequest(
                "position does not fit the I32 input contract".to_owned(),
            )
        })?;
        bytes.extend_from_slice(&position.to_le_bytes());
    }
    Ok(bytes)
}

fn mrope_position_bytes(positions: &[[i32; 3]]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(positions.len().saturating_mul(12));
    for position in positions {
        for component in position {
            bytes.extend_from_slice(&component.to_le_bytes());
        }
    }
    bytes
}

fn validate_input_token_ids(token_ids: &[i32]) -> Result<(), QwenExecutionError> {
    let vocab = i32::try_from(QWEN35_VOCAB_SIZE).expect("fixed Qwen vocabulary fits I32");
    if let Some((index, token)) = token_ids
        .iter()
        .copied()
        .enumerate()
        .find(|(_, token)| *token < 0 || *token >= vocab)
    {
        return Err(QwenExecutionError::InvalidRequest(format!(
            "input token ID {token} at index {index} is outside [0, {QWEN35_VOCAB_SIZE})"
        )));
    }
    Ok(())
}

fn qwen_prefix_identity(
    graph: &QwenGraph,
    plan: &WeightLoadPlan,
    adapter_identity: &str,
) -> QwenPrefixIdentityV1 {
    let mut digest = Sha256::new();
    digest.update((graph.states().len() as u64).to_le_bytes());
    for state in graph.states() {
        digest.update(state.layer().to_le_bytes());
        let kind = match state.kind() {
            QwenGraphStateKind::FullKey => 1_u8,
            QwenGraphStateKind::FullValue => 2,
            QwenGraphStateKind::LinearConvolution => 3,
            QwenGraphStateKind::LinearRecurrent => 4,
        };
        digest.update([kind]);
        match state.descriptor() {
            QwenGraphStateDescriptor::Kv(descriptor) => {
                digest.update([1]);
                digest.update(descriptor.layer_id().to_le_bytes());
                digest.update((descriptor.layout().heads() as u64).to_le_bytes());
                digest.update((descriptor.layout().head_dim() as u64).to_le_bytes());
                let encoding = match descriptor.cache_encoding() {
                    crate::KvCacheEncoding::Fp16 => 1_u8,
                    crate::KvCacheEncoding::Fp8E4M3Fn => 2,
                    crate::KvCacheEncoding::Fp8E4M3FnStatic => 3,
                    crate::KvCacheEncoding::Nvfp4 => 4,
                };
                digest.update([encoding]);
                if let Some((key, value)) = descriptor.static_fp8_scales() {
                    digest.update([1]);
                    digest.update(key.to_bits().to_le_bytes());
                    digest.update(value.to_bits().to_le_bytes());
                } else {
                    digest.update([0]);
                }
            }
            QwenGraphStateDescriptor::Linear(descriptor) => {
                digest.update([2]);
                digest.update(descriptor.layer_id().to_le_bytes());
                let layout = descriptor.layout();
                digest.update((layout.qk_heads() as u64).to_le_bytes());
                digest.update((layout.value_heads() as u64).to_le_bytes());
                digest.update((layout.head_dim() as u64).to_le_bytes());
                digest.update((layout.conv_kernel_size() as u64).to_le_bytes());
            }
        }
        digest.update(format!("{:?}|{:?}", state.dtype(), state.encoding()).as_bytes());
        digest.update((state.shape().len() as u64).to_le_bytes());
        for extent in state.shape() {
            digest.update(extent.to_le_bytes());
        }
        digest.update((state.strides().len() as u64).to_le_bytes());
        for stride in state.strides() {
            digest.update(stride.to_le_bytes());
        }
        digest.update(state.byte_size().to_le_bytes());
    }
    QwenPrefixIdentityV1 {
        model_fingerprint: graph.model_fingerprint().to_owned(),
        plan_digest: *plan.digest(),
        graph_semantics_digest: digest.finalize().into(),
        adapter_identity: adapter_identity.to_owned(),
        state_capacity: graph.state_capacity(),
        is_mtp: graph.is_mtp(),
        is_multimodal: graph.is_multimodal(),
    }
}

fn qwen_hex_digest(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(71);
    output.push_str("sha256:");
    for byte in bytes {
        use std::fmt::Write as _;
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

/// Canonical descriptor identity shared by checkpoint export and fresh-graph
/// restore.  It covers every field that changes the physical KV ABI,
/// including static FP8 decode scales and layout geometry.
fn qwen_kv_descriptor_digest(
    descriptors: impl IntoIterator<Item = (u32, KvStateDescriptor)>,
) -> [u8; 32] {
    let mut ordered = descriptors.into_iter().collect::<Vec<_>>();
    ordered.sort_by_key(|(layer, _)| *layer);
    let mut digest = Sha256::new();
    digest.update(b"sllm-qwen-kv-descriptor-v1");
    digest.update((ordered.len() as u64).to_le_bytes());
    for (layer, descriptor) in ordered {
        digest.update(layer.to_le_bytes());
        digest.update(descriptor.layer_id().to_le_bytes());
        digest.update(descriptor.capacity().to_le_bytes());
        digest.update((descriptor.layout().heads() as u64).to_le_bytes());
        digest.update((descriptor.layout().head_dim() as u64).to_le_bytes());
        let encoding = match descriptor.cache_encoding() {
            KvCacheEncoding::Fp16 => 0_u8,
            KvCacheEncoding::Fp8E4M3Fn => 1,
            KvCacheEncoding::Fp8E4M3FnStatic => 2,
            KvCacheEncoding::Nvfp4 => 3,
        };
        digest.update([encoding]);
        if let Some((key, value)) = descriptor.static_fp8_scales() {
            digest.update([1]);
            digest.update(key.to_bits().to_le_bytes());
            digest.update(value.to_bits().to_le_bytes());
        } else {
            digest.update([0]);
        }
    }
    digest.finalize().into()
}

trait QwenStateImageDescriptor {
    fn capacity(&self) -> u64;
    fn plane_kinds(&self) -> Vec<StatePlaneKindV1>;
}

impl QwenStateImageDescriptor for KvStateDescriptor {
    fn capacity(&self) -> u64 {
        KvStateDescriptor::capacity(*self)
    }

    fn plane_kinds(&self) -> Vec<StatePlaneKindV1> {
        let mut planes = vec![StatePlaneKindV1::KvKey, StatePlaneKindV1::KvValue];
        match self.cache_encoding() {
            KvCacheEncoding::Fp16 | KvCacheEncoding::Fp8E4M3FnStatic => {}
            KvCacheEncoding::Fp8E4M3Fn => {
                planes.extend([StatePlaneKindV1::KvKeyScale, StatePlaneKindV1::KvValueScale]);
            }
            KvCacheEncoding::Nvfp4 => {
                planes.extend([
                    StatePlaneKindV1::KvKeyScale,
                    StatePlaneKindV1::KvValueScale,
                    StatePlaneKindV1::KvKeyOuterScale,
                    StatePlaneKindV1::KvValueOuterScale,
                ]);
            }
        }
        planes
    }
}

impl QwenStateImageDescriptor for LinearAttentionStateDescriptor {
    fn capacity(&self) -> u64 {
        LinearAttentionStateDescriptor::capacity(*self)
    }

    fn plane_kinds(&self) -> Vec<StatePlaneKindV1> {
        vec![
            StatePlaneKindV1::LinearConvSlot0,
            StatePlaneKindV1::LinearConvSlot1,
            StatePlaneKindV1::LinearRecurrentSlot0,
            StatePlaneKindV1::LinearRecurrentSlot1,
            StatePlaneKindV1::LinearScratch,
        ]
    }
}

fn validate_qwen_layer_image<D: QwenStateImageDescriptor>(
    image: &ExecutionStateImageV1,
    owner: StateOwnerKindV1,
    layer: u32,
    descriptor: D,
    expected_length: u64,
) -> Result<(), QwenExecutionError> {
    let metadata = image.metadata();
    if metadata.owner != owner || metadata.layer_id != layer {
        return Err(QwenExecutionError::InvalidRequest(format!(
            "state image layer {layer} owner or layer identity differs"
        )));
    }
    if metadata.published_length != expected_length || expected_length > descriptor.capacity() {
        return Err(QwenExecutionError::InvalidRequest(format!(
            "state image layer {layer} published length differs"
        )));
    }
    match owner {
        StateOwnerKindV1::Kv if metadata.active_slot.is_some() => {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "KV state image layer {layer} has an active slot"
            )));
        }
        StateOwnerKindV1::LinearAttention if !matches!(metadata.active_slot, Some(0 | 1)) => {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "linear state image layer {layer} has an invalid active slot"
            )));
        }
        _ => {}
    }
    let expected_planes = descriptor.plane_kinds();
    if image.planes().len() != expected_planes.len() {
        return Err(QwenExecutionError::InvalidRequest(format!(
            "state image layer {layer} is missing or has unexpected planes"
        )));
    }
    let mut seen = Vec::with_capacity(image.planes().len());
    for plane in image.planes() {
        if plane.owner != owner || plane.layer_id != layer {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "state image layer {layer} plane identity differs"
            )));
        }
        if plane.bytes.is_empty() {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "state image layer {layer} contains an empty plane"
            )));
        }
        if !expected_planes.contains(&plane.plane) {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "state image layer {layer} contains an unexpected plane"
            )));
        }
        if seen.contains(&plane.plane) {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "state image layer {layer} contains a duplicate plane"
            )));
        }
        seen.push(plane.plane);
    }
    if expected_planes.iter().any(|plane| !seen.contains(plane)) {
        return Err(QwenExecutionError::InvalidRequest(format!(
            "state image layer {layer} is missing a required plane"
        )));
    }
    Ok(())
}

fn decode_argmax_bytes(bytes: &[u8]) -> Result<Vec<i32>, QwenExecutionError> {
    if bytes.is_empty() || bytes.len() % std::mem::size_of::<i32>() != 0 {
        return Err(QwenExecutionError::InvalidRequest(
            "argmax readback is not an exact I32 tensor".to_owned(),
        ));
    }
    let mut token_ids = Vec::with_capacity(bytes.len() / std::mem::size_of::<i32>());
    for (index, chunk) in bytes.chunks_exact(std::mem::size_of::<i32>()).enumerate() {
        let token = i32::from_le_bytes(chunk.try_into().expect("exact I32 chunk"));
        if token < 0 {
            return Err(QwenExecutionError::ArgmaxSentinel { index });
        }
        if usize::try_from(token).expect("non-negative I32 fits usize") >= QWEN35_VOCAB_SIZE {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "argmax token ID {token} at output index {index} exceeds the fixed vocabulary"
            )));
        }
        token_ids.push(token);
    }
    Ok(token_ids)
}

fn decode_bf16_logits(bytes: &[u8]) -> Result<Vec<f32>, QwenExecutionError> {
    if bytes.len() != QWEN35_VOCAB_SIZE * std::mem::size_of::<u16>() {
        return Err(QwenExecutionError::InvalidRequest(
            "last-logits readback is not exactly one BF16 vocabulary row".to_owned(),
        ));
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<u16>())
        .map(|chunk| {
            let bits = u16::from_le_bytes(chunk.try_into().expect("exact BF16 chunk"));
            f32::from_bits(u32::from(bits) << 16)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

    use crate::execution::{
        AdapterResource, BoundSemanticOp, ExecutionAdapterAccess,
        ExecutionCausalAttentionSubmissionAdapter, ExecutionKvStateSubmissionAdapter,
        ExecutionLinearAttentionSubmissionAdapter, ExecutionReadbackAdapter,
        ExecutionSessionAdapter, ExecutionState, ExecutionSubmissionAdapter,
        ExecutionTransferAdapter, PreparedOperation, ShutdownReport,
    };
    use crate::kv_state::{KvStateAppendRequest, KvStateSnapshot};
    use crate::linear_attention::{LinearAttentionRequest, LinearAttentionStateSnapshot};

    #[test]
    fn resident_fp8_weight_view_accepts_ocp_and_fnuz_storage() {
        let encoding = Encoding::Fp8Scaled {
            granularity: Fp8ScaleGranularity::OuterDimension,
            scale_dtype: DType::F32,
            resident: Fp8ResidentRepresentation::PackedBytes,
        };
        for dtype in [DType::F8E4M3Fn, DType::F8E4M3FnuZ] {
            let view = TensorView::with_encoding(dtype, encoding, &[3, 17]).unwrap();
            assert!(is_fp8_weight_view(&view));
        }
        assert!(!is_fp8_weight_view(
            &TensorView::contiguous(DType::Bf16, &[3, 17]).unwrap()
        ));
    }

    #[test]
    fn fnuz_resident_rebase_preserves_every_finite_ocp_value() {
        let values = (0_u8..=u8::MAX)
            .filter(|bits| bits & 0x7f != 0x7f)
            .collect::<Vec<_>>();
        let source_scale = 1.25_f32;
        let (rebased, scales) =
            rebase_e4m3fn_outer_rows_to_fnuz(&values, &source_scale.to_le_bytes()).unwrap();
        let rebased_scale = f32::from_le_bytes(scales.try_into().unwrap());
        assert_eq!(rebased_scale, 2.5);
        for (&source, &destination) in values.iter().zip(&rebased) {
            assert_eq!(
                decode_e4m3fn(source) * source_scale,
                crate::decode_e4m3fnuz(destination) * rebased_scale,
                "OCP byte 0x{source:02x}",
            );
        }
        assert!(rebase_e4m3fn_outer_rows_to_fnuz(&[0x7f], &1.0_f32.to_le_bytes()).is_err());
        assert!(rebase_e4m3fn_outer_rows_to_fnuz(&[0], &(-1.0_f32).to_le_bytes()).is_err());
    }

    #[test]
    fn gguf_fp8_resident_payload_selects_ocp_or_fnuz_by_resident_dtype() {
        let values = [0x00, 0x01, 0x7e, 0x80, 0xfe];
        let source_scale = 1.25_f32;
        let f32_scales = normalize_gguf_fp8_scales(
            crate::GgufRecipeEncoding::Fp8E4m3fnChannelF32Scale,
            &source_scale.to_le_bytes(),
        )
        .unwrap();
        let bf16_bits = (source_scale.to_bits() >> 16) as u16;
        let bf16_scales = normalize_gguf_fp8_scales(
            crate::GgufRecipeEncoding::Fp8E4m3fnChannelBf16Scale,
            &bf16_bits.to_le_bytes(),
        )
        .unwrap();
        assert_eq!(f32_scales, bf16_scales);

        let ocp = gguf_fp8_resident_payload(&values, &f32_scales, DType::F8E4M3Fn).unwrap();
        assert_eq!(ocp, [&values[..], &f32_scales[..]].concat());

        let fnuz = gguf_fp8_resident_payload(&values, &bf16_scales, DType::F8E4M3FnuZ).unwrap();
        let (fnuz_values, fnuz_scales) = fnuz.split_at(values.len());
        let fnuz_scale = f32::from_le_bytes(fnuz_scales.try_into().unwrap());
        assert_eq!(fnuz_scale, 2.5);
        for (&source, &destination) in values.iter().zip(fnuz_values) {
            assert_eq!(
                decode_e4m3fn(source) * source_scale,
                crate::decode_e4m3fnuz(destination) * fnuz_scale,
                "OCP byte 0x{source:02x}",
            );
        }
        assert!(
            gguf_fp8_resident_payload(&values, &f32_scales, DType::Bf16).is_err(),
            "GGUF FP8 must reject non-FP8 resident dtypes"
        );
    }

    /// Host-only source for the structural fixture. It deliberately does not
    /// expose a `VerifiedCache` or pretend to validate model bytes; production
    /// provisioning uses `VerifiedProvisionSource` and `upload_verified_weight`.
    #[derive(Default)]
    struct TestProvisionSource {
        uploaded: Mutex<Vec<String>>,
        scale_reads: Mutex<BTreeMap<String, Arc<[u8]>>>,
    }

    impl TestProvisionSource {
        fn uploaded(&self) -> Vec<String> {
            self.uploaded.lock().expect("uploads lock").clone()
        }
    }

    impl QwenProvisionSource for TestProvisionSource {
        fn upload_weight(
            &self,
            plan: &WeightLoadPlan,
            binding: &QwenGraphWeightBinding,
            _session: &ExecutionSession,
            _queue: &ExecutionQueue,
            destination: crate::BufferRange,
            _completion_timeout: Duration,
        ) -> Result<(), QwenExecutionError> {
            let entry = plan
                .entries
                .iter()
                .find(|entry| entry.tensor_name == binding.tensor_name())
                .ok_or_else(|| {
                    QwenExecutionError::InvalidGraph("fixture plan weight missing".to_owned())
                })?;
            let byte_length = entry.source_range[1]
                .checked_sub(entry.source_range[0])
                .ok_or_else(|| {
                    QwenExecutionError::InvalidGraph("fixture source range underflow".to_owned())
                })?;
            if entry.classification != WeightClassification::Required
                || entry.consumer != Some(binding.consumer())
                || destination.size_bytes() != byte_length
            {
                return Err(QwenExecutionError::InvalidGraph(
                    "fixture weight upload does not match the load plan".to_owned(),
                ));
            }
            self.uploaded
                .lock()
                .expect("uploads lock")
                .push(binding.tensor_name().to_owned());
            Ok(())
        }

        fn read_scale_bytes(
            &self,
            tensor_name: &str,
            expected_length: usize,
        ) -> Result<Arc<[u8]>, QwenExecutionError> {
            if expected_length != 512 {
                return Err(QwenExecutionError::InvalidGraph(
                    "fixture scale has an unexpected byte length".to_owned(),
                ));
            }
            let mut cache = self.scale_reads.lock().expect("scale reads lock");
            if let Some(bytes) = cache.get(tensor_name) {
                return Ok(Arc::clone(bytes));
            }
            let seed = tensor_name
                .bytes()
                .fold(0_u8, |value, byte| value.wrapping_add(byte));
            let bytes: Arc<[u8]> = Arc::from(
                (0..expected_length)
                    .map(|index| seed.wrapping_add(index as u8))
                    .collect::<Vec<_>>(),
            );
            cache.insert(tensor_name.to_owned(), Arc::clone(&bytes));
            Ok(bytes)
        }
    }

    #[derive(Default)]
    struct RecorderState {
        events: Vec<String>,
        kv_lengths: BTreeMap<u64, u64>,
        linear_lengths: BTreeMap<u64, u64>,
        argmax_sequences: VecDeque<Vec<i32>>,
        preprocess: Vec<(AttentionPreprocessPositionMode, u32, u32)>,
        uploads: Vec<Vec<u8>>,
    }

    #[derive(Clone)]
    struct ExecutionRecorder {
        state: Arc<Mutex<RecorderState>>,
        failure_kind: Arc<Mutex<Option<SemanticOpKind>>>,
        pending_kind: Arc<Mutex<Option<SemanticOpKind>>>,
        shutdown_calls: Arc<AtomicUsize>,
        total_memory_bytes: Arc<AtomicU64>,
        available_memory_bytes: Arc<AtomicU64>,
        state_image_import_calls: Arc<AtomicUsize>,
        state_image_failure: Arc<AtomicBool>,
    }

    impl Default for ExecutionRecorder {
        fn default() -> Self {
            Self {
                state: Arc::new(Mutex::new(RecorderState::default())),
                failure_kind: Arc::new(Mutex::new(None)),
                pending_kind: Arc::new(Mutex::new(None)),
                shutdown_calls: Arc::new(AtomicUsize::new(0)),
                total_memory_bytes: Arc::new(AtomicU64::new(u64::MAX)),
                available_memory_bytes: Arc::new(AtomicU64::new(u64::MAX)),
                state_image_import_calls: Arc::new(AtomicUsize::new(0)),
                state_image_failure: Arc::new(AtomicBool::new(false)),
            }
        }
    }

    impl ExecutionRecorder {
        fn with_argmax_sequences(sequences: impl IntoIterator<Item = Vec<i32>>) -> Self {
            let state = RecorderState {
                argmax_sequences: sequences.into_iter().collect(),
                ..RecorderState::default()
            };
            Self {
                state: Arc::new(Mutex::new(state)),
                ..Self::default()
            }
        }

        fn event(&self, event: impl Into<String>) {
            self.state
                .lock()
                .expect("recorder lock")
                .events
                .push(event.into());
        }

        fn events(&self) -> Vec<String> {
            self.state.lock().expect("recorder lock").events.clone()
        }

        fn preprocess(&self) -> Vec<(AttentionPreprocessPositionMode, u32, u32)> {
            self.state.lock().expect("recorder lock").preprocess.clone()
        }

        fn uploads(&self) -> Vec<Vec<u8>> {
            self.state.lock().expect("recorder lock").uploads.clone()
        }

        fn set_failure(&self, kind: SemanticOpKind) {
            *self.failure_kind.lock().expect("failure lock") = Some(kind);
        }

        fn set_pending(&self, kind: SemanticOpKind) {
            *self.pending_kind.lock().expect("pending lock") = Some(kind);
        }

        fn set_state_image_failure(&self, enabled: bool) {
            self.state_image_failure.store(enabled, Ordering::Relaxed);
        }

        fn state_image_import_calls(&self) -> usize {
            self.state_image_import_calls.load(Ordering::Relaxed)
        }

        fn semantic_completion(&self, kind: SemanticOpKind) -> ExecutionState {
            if *self.pending_kind.lock().expect("pending lock") == Some(kind) {
                ExecutionState::Pending
            } else if *self.failure_kind.lock().expect("failure lock") == Some(kind) {
                ExecutionState::Failure
            } else {
                ExecutionState::Success
            }
        }

        fn linear_length(&self, state: &LinearAttentionState) -> u64 {
            *self
                .state
                .lock()
                .expect("recorder lock")
                .linear_lengths
                .get(&state.id().raw())
                .expect("created linear state")
        }
    }

    impl ExecutionSessionAdapter for ExecutionRecorder {
        fn max_transfer_bytes(&self) -> u64 {
            1_048_576
        }

        fn available_memory_bytes(&self) -> Option<u64> {
            Some(self.available_memory_bytes.load(Ordering::Relaxed))
        }

        fn total_memory_bytes(&self) -> Option<u64> {
            Some(self.total_memory_bytes.load(Ordering::Relaxed))
        }

        fn supports(&self, _descriptor: &SemanticOpDescriptor) -> PrepareSupport {
            PrepareSupport::Supported
        }

        fn create_queue(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
        ) -> Result<AdapterResource, ExecutionError> {
            self.event("queue");
            Ok(AdapterResource::new(()))
        }

        fn create_queue_fence(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _queue: &ExecutionQueue,
        ) -> Result<Box<dyn crate::execution::ExecutionQueueFenceAdapter>, ExecutionError> {
            self.event("queue-fence");
            Ok(Box::new(RecorderFence))
        }

        fn allocate(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            size_bytes: u64,
        ) -> Result<AdapterResource, ExecutionError> {
            self.event(format!("allocate:{size_bytes}"));
            Ok(AdapterResource::new(()))
        }

        fn prepare(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            operation: &BoundSemanticOp,
        ) -> Result<AdapterResource, ExecutionError> {
            if let Some(contract) = operation.descriptor().attention_preprocess_contract() {
                self.state.lock().expect("recorder lock").preprocess.push((
                    contract.position_mode(),
                    contract.start_position(),
                    contract.token_count(),
                ));
            }
            self.event(format!("prepare:{:?}", operation.descriptor().kind()));
            Ok(AdapterResource::new(()))
        }

        fn submit(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            prepared: &PreparedOperation,
            _queue: &ExecutionQueue,
        ) -> Result<(Box<dyn ExecutionSubmissionAdapter>, crate::DispatchEvidence), ExecutionError>
        {
            let kind = prepared.operation().descriptor().kind();
            self.event(format!("submit:{kind:?}"));
            Ok((
                Box::new(RecorderSubmission {
                    recorder: Arc::new(self.clone_for_submission()),
                    kind,
                }),
                dispatch_evidence(),
            ))
        }

        fn upload(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _queue: &ExecutionQueue,
            _destination: &crate::BufferRange,
            bytes: Arc<[u8]>,
        ) -> Result<Box<dyn ExecutionTransferAdapter>, ExecutionError> {
            self.state
                .lock()
                .expect("recorder lock")
                .uploads
                .push(bytes.to_vec());
            self.event(format!("upload:{}", bytes.len()));
            Ok(Box::new(RecorderTransfer {
                recorder: Arc::new(self.clone_for_submission()),
            }))
        }

        fn readback(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _queue: &ExecutionQueue,
            source: &crate::BufferRange,
        ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError> {
            let length =
                usize::try_from(source.size_bytes()).map_err(|_| ExecutionError::InvalidRange {
                    reason: "recorder readback length does not fit usize".to_owned(),
                })?;
            Ok(Box::new(RecorderReadback {
                recorder: Arc::new(self.clone_for_submission()),
                bytes: vec![0; length],
            }))
        }

        fn shutdown(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _deadline: Duration,
        ) -> Result<ShutdownReport, ExecutionError> {
            self.shutdown_calls.fetch_add(1, Ordering::Relaxed);
            Ok(ShutdownReport {
                retryable_cleanup: 0,
                durable_quarantine: 0,
            })
        }

        fn create_kv_state(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            state_id: crate::KvStateId,
            descriptor: KvStateDescriptor,
        ) -> Result<AdapterResource, ExecutionError> {
            self.state
                .lock()
                .expect("recorder lock")
                .kv_lengths
                .insert(state_id.raw(), 0);
            self.event(format!("create-kv:{}", descriptor.layer_id()));
            Ok(AdapterResource::new(()))
        }

        fn fork_kv_state(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            source: &KvState,
            destination_id: crate::KvStateId,
            _destination_descriptor: KvStateDescriptor,
        ) -> Result<(AdapterResource, crate::StateForkAuditV1), ExecutionError> {
            let length = *self
                .state
                .lock()
                .expect("recorder lock")
                .kv_lengths
                .get(&source.id().raw())
                .expect("created source KV state");
            self.state
                .lock()
                .expect("recorder lock")
                .kv_lengths
                .insert(destination_id.raw(), length);
            self.event(format!("fork-kv:{}:{length}", source.layer_id()));
            let audit = crate::StateForkAuditV1::new(
                crate::StateForkModeV1::SharedReadOnlyPages,
                length,
                1,
                0,
                0,
            )
            .map_err(|error| ExecutionError::InvalidRequest {
                reason: error.to_string(),
            })?;
            Ok((AdapterResource::new(()), audit))
        }

        fn kv_state_snapshot(
            &self,
            access: &ExecutionAdapterAccess<'_>,
            state: &KvState,
        ) -> Result<KvStateSnapshot, ExecutionError> {
            let length = *self
                .state
                .lock()
                .expect("recorder lock")
                .kv_lengths
                .get(&state.id().raw())
                .expect("created KV state");
            self.event(format!("kv-snapshot:{}:{length}", state.layer_id()));
            let physical = KvPhysicalMemorySnapshot::new(
                state.descriptor().capacity(),
                length,
                1,
                1,
                state.descriptor().capacity(),
                length,
            )
            .map_err(|error| ExecutionError::InvalidRequest {
                reason: error.to_string(),
            })?;
            KvStateSnapshot::new_with_physical_memory(
                access.session_id(),
                state.id(),
                state.descriptor(),
                length,
                physical,
            )
            .map_err(|error| ExecutionError::InvalidRequest {
                reason: error.to_string(),
            })
        }

        fn export_kv_state_image(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            state: &KvState,
        ) -> Result<ExecutionStateImageV1, ExecutionError> {
            let length = *self
                .state
                .lock()
                .expect("recorder lock")
                .kv_lengths
                .get(&state.id().raw())
                .expect("created KV state");
            let mut kinds = vec![StatePlaneKindV1::KvKey, StatePlaneKindV1::KvValue];
            match state.descriptor().cache_encoding() {
                KvCacheEncoding::Fp16 | KvCacheEncoding::Fp8E4M3FnStatic => {}
                KvCacheEncoding::Fp8E4M3Fn => {
                    kinds.extend([StatePlaneKindV1::KvKeyScale, StatePlaneKindV1::KvValueScale]);
                }
                KvCacheEncoding::Nvfp4 => {
                    kinds.extend([
                        StatePlaneKindV1::KvKeyScale,
                        StatePlaneKindV1::KvValueScale,
                        StatePlaneKindV1::KvKeyOuterScale,
                        StatePlaneKindV1::KvValueOuterScale,
                    ]);
                }
            }
            Ok(ExecutionStateImageV1::new(
                crate::StateLayerMetadataV1 {
                    owner: StateOwnerKindV1::Kv,
                    layer_id: state.layer_id(),
                    published_length: length,
                    generation: 1,
                    active_slot: None,
                },
                kinds
                    .into_iter()
                    .map(|plane| crate::OpaqueStatePlane {
                        owner: StateOwnerKindV1::Kv,
                        layer_id: state.layer_id(),
                        plane,
                        bytes: vec![state.layer_id() as u8, plane as u8],
                    })
                    .collect(),
            ))
        }

        fn import_kv_state_image(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            state: &KvState,
            image: &ExecutionStateImageV1,
        ) -> Result<(), ExecutionError> {
            self.state_image_import_calls
                .fetch_add(1, Ordering::Relaxed);
            self.event(format!("import-kv-image:{}", state.layer_id()));
            if self.state_image_failure.load(Ordering::Relaxed) {
                return Err(ExecutionError::BackendStatus {
                    status: 91,
                    diagnostic: "recorder KV image import failure".to_owned(),
                });
            }
            self.state
                .lock()
                .expect("recorder lock")
                .kv_lengths
                .insert(state.id().raw(), image.metadata().published_length);
            Ok(())
        }

        fn append_kv_state(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            state: &KvState,
            _queue: &ExecutionQueue,
            _key: &OwnedTensorBinding,
            _value: &OwnedTensorBinding,
            request: &KvStateAppendRequest,
        ) -> Result<
            (
                Box<dyn ExecutionKvStateSubmissionAdapter>,
                crate::DispatchEvidence,
            ),
            ExecutionError,
        > {
            self.event(format!(
                "kv-append:{}:{}:{}",
                state.layer_id(),
                request.start_position(),
                request.token_count()
            ));
            Ok((
                Box::new(RecorderKvCompletion {
                    recorder: Arc::new(self.clone_for_submission()),
                    state_id: state.id().raw(),
                    expected_length: request.end_position(),
                    completed: false,
                }),
                dispatch_evidence(),
            ))
        }

        fn execute_causal_attention(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            state: &KvState,
            _queue: &ExecutionQueue,
            _query: &OwnedTensorBinding,
            _output: &OwnedTensorBinding,
            descriptor: CausalAttentionDescriptor,
        ) -> Result<
            (
                Box<dyn ExecutionCausalAttentionSubmissionAdapter>,
                crate::DispatchEvidence,
            ),
            ExecutionError,
        > {
            Ok((
                Box::new(RecorderCausalCompletion {
                    recorder: Arc::new(self.clone_for_submission()),
                    event: format!(
                        "causal:{}:{}:{}",
                        state.layer_id(),
                        descriptor.start_position(),
                        descriptor.query_count()
                    ),
                }),
                dispatch_evidence(),
            ))
        }

        fn create_linear_attention_state(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            state_id: crate::LinearAttentionStateId,
            descriptor: LinearAttentionStateDescriptor,
        ) -> Result<AdapterResource, ExecutionError> {
            self.state
                .lock()
                .expect("recorder lock")
                .linear_lengths
                .insert(state_id.raw(), 0);
            self.event(format!("create-linear:{}", descriptor.layer_id()));
            Ok(AdapterResource::new(()))
        }

        fn fork_linear_attention_state(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            source: &LinearAttentionState,
            destination_id: crate::LinearAttentionStateId,
            _destination_descriptor: LinearAttentionStateDescriptor,
        ) -> Result<(AdapterResource, crate::StateForkAuditV1), ExecutionError> {
            let length = self.linear_length(source);
            self.state
                .lock()
                .expect("recorder lock")
                .linear_lengths
                .insert(destination_id.raw(), length);
            self.event(format!("fork-linear:{}:{length}", source.layer_id()));
            let audit =
                crate::StateForkAuditV1::new(crate::StateForkModeV1::DeviceCopy, length, 0, 1, 1)
                    .map_err(|error| ExecutionError::InvalidRequest {
                    reason: error.to_string(),
                })?;
            Ok((AdapterResource::new(()), audit))
        }

        fn linear_attention_state_snapshot(
            &self,
            access: &ExecutionAdapterAccess<'_>,
            state: &LinearAttentionState,
        ) -> Result<LinearAttentionStateSnapshot, ExecutionError> {
            let length = self.linear_length(state);
            self.event(format!("linear-snapshot:{}:{length}", state.layer_id()));
            LinearAttentionStateSnapshot::new(
                access.session_id(),
                state.id(),
                state.descriptor(),
                length,
            )
            .map_err(|error| ExecutionError::InvalidRequest {
                reason: error.to_string(),
            })
        }

        fn export_linear_attention_state_image(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            state: &LinearAttentionState,
        ) -> Result<ExecutionStateImageV1, ExecutionError> {
            let length = self.linear_length(state);
            let kinds = [
                StatePlaneKindV1::LinearConvSlot0,
                StatePlaneKindV1::LinearConvSlot1,
                StatePlaneKindV1::LinearRecurrentSlot0,
                StatePlaneKindV1::LinearRecurrentSlot1,
                StatePlaneKindV1::LinearScratch,
            ];
            Ok(ExecutionStateImageV1::new(
                crate::StateLayerMetadataV1 {
                    owner: StateOwnerKindV1::LinearAttention,
                    layer_id: state.layer_id(),
                    published_length: length,
                    generation: 1,
                    active_slot: Some((state.layer_id() % 2) as u8),
                },
                kinds
                    .into_iter()
                    .map(|plane| crate::OpaqueStatePlane {
                        owner: StateOwnerKindV1::LinearAttention,
                        layer_id: state.layer_id(),
                        plane,
                        bytes: vec![state.layer_id() as u8, plane as u8],
                    })
                    .collect(),
            ))
        }

        fn import_linear_attention_state_image(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            state: &LinearAttentionState,
            image: &ExecutionStateImageV1,
        ) -> Result<(), ExecutionError> {
            self.state_image_import_calls
                .fetch_add(1, Ordering::Relaxed);
            self.event(format!("import-linear-image:{}", state.layer_id()));
            if self.state_image_failure.load(Ordering::Relaxed) {
                return Err(ExecutionError::BackendStatus {
                    status: 92,
                    diagnostic: "recorder linear image import failure".to_owned(),
                });
            }
            self.state
                .lock()
                .expect("recorder lock")
                .linear_lengths
                .insert(state.id().raw(), image.metadata().published_length);
            Ok(())
        }

        fn execute_linear_attention(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            state: &LinearAttentionState,
            _queue: &ExecutionQueue,
            _bindings: &LinearAttentionBindings,
            request: LinearAttentionRequest,
        ) -> Result<
            (
                Box<dyn ExecutionLinearAttentionSubmissionAdapter>,
                crate::DispatchEvidence,
            ),
            ExecutionError,
        > {
            let descriptor = request.descriptor();
            self.event(format!(
                "linear:{}:{}:{}",
                state.layer_id(),
                descriptor.start_position(),
                descriptor.token_count()
            ));
            let mut dispatch = dispatch_evidence();
            dispatch.dispatch_count = 2;
            Ok((
                Box::new(RecorderLinearCompletion {
                    recorder: Arc::new(self.clone_for_submission()),
                    state_id: state.id().raw(),
                    expected_length: descriptor.expected_length(),
                    completed: false,
                }),
                dispatch,
            ))
        }
    }

    impl ExecutionRecorder {
        fn clone_for_submission(&self) -> Self {
            self.clone()
        }
    }

    struct RecorderSubmission {
        recorder: Arc<ExecutionRecorder>,
        kind: SemanticOpKind,
    }

    struct RecorderFence;

    impl crate::execution::ExecutionQueueFenceAdapter for RecorderFence {
        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }

        fn token(&self) -> Result<u64, ExecutionError> {
            Ok(1)
        }
    }

    impl ExecutionSubmissionAdapter for RecorderSubmission {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            self.wait(Duration::ZERO)
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            self.recorder.event(format!("semantic:{:?}", self.kind));
            Ok(self.recorder.semantic_completion(self.kind))
        }

        fn start_output_readback(
            &mut self,
            _access: &ExecutionAdapterAccess<'_>,
            output: &OwnedTensorBinding,
        ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError> {
            if self.kind == SemanticOpKind::TokenSelect {
                if output.view().shape() != [16] {
                    return Err(ExecutionError::InvalidRequest {
                        reason: "recorder token-selector output has the wrong shape".to_owned(),
                    });
                }
                self.recorder.event("token-selector-readback-start");
                return Ok(Box::new(RecorderReadback {
                    recorder: Arc::clone(&self.recorder),
                    bytes: vec![0_u8; 16],
                }));
            }
            if self.kind != SemanticOpKind::Argmax {
                return Err(ExecutionError::InvalidRequest {
                    reason: "recorder only permits argmax output readback".to_owned(),
                });
            }
            let count =
                *output
                    .view()
                    .shape()
                    .first()
                    .ok_or_else(|| ExecutionError::InvalidRequest {
                        reason: "recorder argmax output has no token extent".to_owned(),
                    })?;
            let tokens = self
                .recorder
                .state
                .lock()
                .expect("recorder lock")
                .argmax_sequences
                .pop_front()
                .unwrap_or_else(|| vec![7; count]);
            if tokens.len() != count {
                return Err(ExecutionError::InvalidRequest {
                    reason: "recorder argmax token count differs from output view".to_owned(),
                });
            }
            self.recorder.event("argmax-readback-start");
            Ok(Box::new(RecorderReadback {
                recorder: Arc::clone(&self.recorder),
                bytes: i32_bytes(&tokens),
            }))
        }
    }

    struct RecorderTransfer {
        recorder: Arc<ExecutionRecorder>,
    }

    impl ExecutionTransferAdapter for RecorderTransfer {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            self.wait(Duration::ZERO)
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            self.recorder.event("transfer-success");
            Ok(ExecutionState::Success)
        }
    }

    struct RecorderReadback {
        recorder: Arc<ExecutionRecorder>,
        bytes: Vec<u8>,
    }

    impl ExecutionReadbackAdapter for RecorderReadback {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            self.wait(Duration::ZERO)
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            self.recorder.event("readback-success");
            Ok(ExecutionState::Success)
        }

        fn read_into(&mut self, destination: &mut [u8]) -> Result<u64, ExecutionError> {
            if destination.len() != self.bytes.len() {
                return Err(ExecutionError::InvalidRange {
                    reason: "recorder readback size mismatch".to_owned(),
                });
            }
            destination.copy_from_slice(&self.bytes);
            self.recorder.event("readback-bytes");
            Ok(self.bytes.len() as u64)
        }
    }

    struct RecorderKvCompletion {
        recorder: Arc<ExecutionRecorder>,
        state_id: u64,
        expected_length: u64,
        completed: bool,
    }

    impl ExecutionKvStateSubmissionAdapter for RecorderKvCompletion {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            self.wait(Duration::ZERO)
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            if !self.completed {
                self.recorder
                    .state
                    .lock()
                    .expect("recorder lock")
                    .kv_lengths
                    .insert(self.state_id, self.expected_length);
                self.completed = true;
            }
            self.recorder.event("kv-success");
            Ok(ExecutionState::Success)
        }
    }

    struct RecorderCausalCompletion {
        recorder: Arc<ExecutionRecorder>,
        event: String,
    }

    impl ExecutionCausalAttentionSubmissionAdapter for RecorderCausalCompletion {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            self.wait(Duration::ZERO)
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            self.recorder.event(self.event.clone());
            Ok(ExecutionState::Success)
        }
    }

    struct RecorderLinearCompletion {
        recorder: Arc<ExecutionRecorder>,
        state_id: u64,
        expected_length: u64,
        completed: bool,
    }

    impl ExecutionLinearAttentionSubmissionAdapter for RecorderLinearCompletion {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            self.wait(Duration::ZERO)
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            if !self.completed {
                self.recorder
                    .state
                    .lock()
                    .expect("recorder lock")
                    .linear_lengths
                    .insert(self.state_id, self.expected_length);
                self.completed = true;
            }
            self.recorder.event("linear-success");
            Ok(ExecutionState::Success)
        }
    }

    fn dispatch_evidence() -> crate::DispatchEvidence {
        crate::DispatchEvidence {
            abi_version: 1,
            info_version: 1,
            dispatch_id: 1,
            dispatch_count: 1,
            kernel_id: 1,
            workgroup_size_x: 1,
            grid_size_x: 1,
            row_count: 1,
            normalized_size: 1,
            backend: 1,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: "recorder".to_owned(),
            device_symbol: "recorder".to_owned(),
            target: "recorder".to_owned(),
        }
    }

    fn provisioned_core(
        recorder: Arc<ExecutionRecorder>,
    ) -> (QwenExecutionCore, TestProvisionSource) {
        let (graph, plan) = crate::qwen_graph::qwen35_execution_fixture();
        let session = Arc::new(ExecutionSession::new("recorder", recorder));
        let source = TestProvisionSource::default();
        let core =
            QwenExecutionCore::provision(session, graph, plan, Duration::from_millis(1), &source)
                .expect("structural fixture provisions");
        (core, source)
    }

    fn control_fixture(
        graph: &QwenGraph,
        plan: &WeightLoadPlan,
    ) -> Arc<crate::adapter::VerifiedControlVectorPayloadV1> {
        let hidden_size = graph
            .tensor_metadata()
            .iter()
            .find(|tensor| tensor.name() == "embedding.output")
            .and_then(|tensor| tensor.view().shape().get(1).copied())
            .expect("fixture hidden size");
        let layer_count = u64::try_from(graph.layer_types().len()).expect("fixture layer count");
        let payload = [0x00_u8, 0x3f_u8].repeat(hidden_size);
        let payload_sha256 = format!("sha256:{:x}", Sha256::digest(&payload));
        let lock = serde_json::json!({
            "schema_version": "sllm-adapter-lock-v1",
            "kind": "control-vector",
            "artifact_id": "control-runtime-fixture-v1",
            "dtype": "bf16",
            "base_model_fingerprint": graph.model_fingerprint(),
            "base_weight_plan_digest": plan.digest_hex(),
            "payload_sha256": payload_sha256,
            "payload_size": payload.len(),
            "hidden_size": hidden_size,
            "layer_start": 0,
            "layer_end": 1,
            "vector_offset": 0,
            "vector_size": payload.len(),
        });
        let dims = crate::adapter::AdapterModelDimsV1::new(
            u64::try_from(hidden_size).expect("fixture hidden size"),
            layer_count,
        )
        .expect("fixture dimensions");
        let lock_json = serde_json::to_vec(&lock).expect("fixture lock serializes");
        Arc::new(
            crate::adapter::VerifiedControlVectorPayloadV1::from_bytes(
                &lock_json,
                Arc::<[u8]>::from(payload),
                graph.model_fingerprint(),
                plan,
                dims,
            )
            .expect("fixture control vector verifies"),
        )
    }

    #[test]
    fn adapter_scalar_bf16_conversion_is_finite_and_signed_at_boundaries() {
        for value in [0.0_f32, -2.0, 0.5, 16.0, -16.0] {
            let bits = bf16_scalar_from_f32(value).expect("finite BF16 scalar");
            assert_ne!(bits & 0x7f80, 0x7f80, "scalar {value} became BF16 Inf/NaN");
            assert_eq!(bits, bf16_from_f32(value));
        }
        assert!(bf16_scalar_from_f32(f32::INFINITY).is_err());
        assert!(bf16_scalar_from_f32(f32::NAN).is_err());
        // The largest finite f32 rounds to the BF16 exponent reserved for
        // Inf/NaN; reject it instead of submitting a non-finite multiplier.
        assert!(bf16_scalar_from_f32(f32::from_bits(0x7f7f_ffff)).is_err());
    }

    #[test]
    fn control_scale_is_uploaded_and_precedes_broadcast_add_in_submission_graph() {
        let recorder = Arc::new(ExecutionRecorder::default());
        let (graph, plan) = crate::qwen_graph::qwen35_execution_fixture();
        let session = Arc::new(ExecutionSession::new("recorder", recorder.clone()));
        let source = TestProvisionSource::default();
        let resident = Arc::new(
            QwenResidentInner::provision(
                Arc::clone(&session),
                graph.clone(),
                plan.clone(),
                Duration::from_millis(1),
                &source,
            )
            .expect("resident fixture provisions"),
        );
        let artifact = control_fixture(&graph, &plan);
        let mut event_start = recorder.events().len();
        for (alias, scale) in [("zero", 0.0_f32), ("negative", -2.0), ("fraction", 0.5)] {
            let request = AdapterRequestSetV1::new(
                Vec::new(),
                vec![ControlVectorSelectionV1 {
                    alias: alias.to_owned(),
                    artifact: Arc::clone(&artifact),
                    scale,
                }],
            )
            .expect("control request validates");
            let mut core =
                QwenExecutionCore::from_resident(Arc::clone(&resident), graph.clone(), request)
                    .expect("adapter request provisions");
            core.prefill(&[1]).expect("control request executes");

            let events = recorder.events();
            let request_events = &events[event_start..];
            let scalar = request_events
                .iter()
                .position(|event| event == "prepare:ScalarMul")
                .expect("control scale scalar-mul is prepared");
            let broadcast = request_events
                .iter()
                .position(|event| event == "prepare:BroadcastAdd")
                .expect("control broadcast-add is prepared");
            assert!(scalar < broadcast, "scale must precede residual add");
            let scalar_upload = recorder
                .uploads()
                .into_iter()
                .rev()
                .find(|bytes| bytes.len() == std::mem::size_of::<u16>())
                .expect("control scale upload is recorded");
            let actual = u16::from_le_bytes(scalar_upload.try_into().expect("BF16 scalar"));
            assert_eq!(actual, bf16_scalar_from_f32(scale).unwrap());
            event_start = events.len();
            drop(core);
        }
    }

    #[test]
    fn terminal_row_views_cover_non_aligned_boundaries() {
        for rows in [1_usize, 2, 3, 17, 255, 256, 257, 2_047, 2_049] {
            let matrix = TensorView::contiguous(DType::Bf16, &[rows, 7]).unwrap();
            let last = last_row_view(&matrix).unwrap();
            assert_eq!(last.shape(), [1, 7]);
            assert_eq!(last.strides(), [7, 1]);
            assert_eq!(last.byte_offset(), ((rows - 1) * 7 * 2) as u64);
            assert_eq!(last.payload_bytes(), 14);

            let vector = TensorView::contiguous(DType::I32, &[rows]).unwrap();
            let last = last_row_view(&vector).unwrap();
            assert_eq!(last.shape(), [1]);
            assert_eq!(last.byte_offset(), ((rows - 1) * 4) as u64);
            assert_eq!(last.payload_bytes(), 4);
        }
    }

    #[test]
    fn workspace_interval_allocator_respects_inclusive_lifetimes_and_alignment() {
        assert_eq!(align_workspace(255).unwrap(), 256);
        assert_eq!(align_workspace(256).unwrap(), 256);
        assert_eq!(align_workspace(257).unwrap(), 512);
        assert!(align_workspace(u64::MAX).is_err());

        let mut intervals = [
            WorkspaceInterval {
                tensor_id: 0,
                first_node: 0,
                last_node: 2,
                size_bytes: 256,
            },
            WorkspaceInterval {
                tensor_id: 1,
                first_node: 0,
                last_node: 1,
                size_bytes: 512,
            },
            WorkspaceInterval {
                tensor_id: 2,
                first_node: 1,
                last_node: 3,
                size_bytes: 256,
            },
            WorkspaceInterval {
                tensor_id: 3,
                first_node: 2,
                last_node: 2,
                size_bytes: 256,
            },
        ];
        let (offsets, slots, high_water) = allocate_workspace_intervals(&mut intervals).unwrap();
        assert_eq!(offsets[&0], 0);
        assert_eq!(offsets[&1], 256);
        assert_eq!(offsets[&2], 768);
        assert_eq!(offsets[&3], 256);
        assert_eq!(slots, BTreeMap::from([(0, 256), (256, 512), (768, 256)]));
        assert_eq!(high_water, 1_024);

        let mut duplicate = [
            WorkspaceInterval {
                tensor_id: 7,
                first_node: 0,
                last_node: 0,
                size_bytes: 256,
            },
            WorkspaceInterval {
                tensor_id: 7,
                first_node: 1,
                last_node: 1,
                size_bytes: 256,
            },
        ];
        assert!(allocate_workspace_intervals(&mut duplicate).is_err());

        let mut overflow = [
            WorkspaceInterval {
                tensor_id: 0,
                first_node: 0,
                last_node: 1,
                size_bytes: u64::MAX - 255,
            },
            WorkspaceInterval {
                tensor_id: 1,
                first_node: 0,
                last_node: 1,
                size_bytes: 256,
            },
        ];
        assert!(allocate_workspace_intervals(&mut overflow).is_err());
    }

    #[test]
    fn terminal_allocation_compaction_starts_at_the_measured_crossover() {
        let view = TensorView::contiguous(DType::Bf16, &[257, 7]).unwrap();
        let outputs = [4, 5];
        assert_eq!(
            compact_terminal_allocation_end(254, false, 4, outputs, &view).unwrap(),
            None
        );
        assert_eq!(
            compact_terminal_allocation_end(255, false, 4, outputs, &view).unwrap(),
            Some(first_row_view(&view).unwrap().end_offset())
        );
        assert_eq!(
            compact_terminal_allocation_end(255, true, 4, outputs, &view).unwrap(),
            None
        );
        assert_eq!(
            compact_terminal_allocation_end(255, false, 3, outputs, &view).unwrap(),
            None
        );
    }

    #[test]
    fn chunk_candidates_cover_the_16_gib_boundary_and_non_aligned_prompts() {
        let threshold = QWEN_PREFILL_SMALL_DEVICE_MAX_BYTES;
        assert_eq!(
            qwen_prefill_chunk_candidates(threshold - 1, 10_001).unwrap(),
            [512]
        );
        assert_eq!(
            qwen_prefill_chunk_candidates(threshold, 10_001).unwrap(),
            [512]
        );
        assert_eq!(
            qwen_prefill_chunk_candidates(threshold + 1, 10_001).unwrap(),
            [10_001, 8_192, 4_096, 2_048, 512]
        );
        assert_eq!(
            qwen_prefill_chunk_candidates(threshold + 1, 511).unwrap(),
            [511]
        );
        for prompt in [
            1_u64, 3, 511, 512, 513, 2_047, 2_048, 2_049, 4_095, 4_096, 4_097, 8_191, 8_192, 8_193,
            16_383, 16_384, 16_385, 65_535, 65_536, 65_537,
        ] {
            let small = qwen_prefill_chunk_candidates(threshold, prompt).unwrap();
            assert_eq!(small, [prompt.min(512)], "small-device prompt {prompt}");

            let large = qwen_prefill_chunk_candidates(threshold + 1, prompt).unwrap();
            assert_eq!(large.first().copied(), Some(prompt.min(16_384)));
            assert_eq!(large.last().copied(), Some(prompt.min(512)));
            assert!(large.windows(2).all(|pair| pair[0] > pair[1]));
        }
        assert!(qwen_prefill_chunk_candidates(0, 1).is_err());
        assert!(qwen_prefill_chunk_candidates(threshold, 0).is_err());
    }

    #[test]
    fn device_memory_preflight_accepts_exact_required_boundary() {
        let (graph, plan) = crate::qwen_graph::qwen35_execution_fixture();
        let layout = validate_graph_plan(&graph, &plan).expect("fixture layout validates");
        let total = 32 * 1024 * 1024 * 1024;
        let required = memory_estimate_from_layout(&graph, &layout, total)
            .expect("fixture estimate")
            .required_bytes();
        let recorder = Arc::new(ExecutionRecorder::default());
        recorder.total_memory_bytes.store(total, Ordering::Relaxed);
        let session = ExecutionSession::new("recorder", recorder.clone());

        recorder
            .available_memory_bytes
            .store(required - 1, Ordering::Relaxed);
        assert!(preflight_device_memory(&session, &graph, &layout, false).is_err());
        recorder
            .available_memory_bytes
            .store(required, Ordering::Relaxed);
        preflight_device_memory(&session, &graph, &layout, false)
            .expect("exact required bytes fit");
        recorder
            .available_memory_bytes
            .store(required + 1, Ordering::Relaxed);
        preflight_device_memory(&session, &graph, &layout, false).expect("one byte over fits");

        let incremental = required
            .checked_sub(model_resident_bytes(&graph, &layout).unwrap())
            .unwrap();
        recorder
            .available_memory_bytes
            .store(incremental - 1, Ordering::Relaxed);
        assert!(preflight_device_memory(&session, &graph, &layout, true).is_err());
        recorder
            .available_memory_bytes
            .store(incremental, Ordering::Relaxed);
        preflight_device_memory(&session, &graph, &layout, true)
            .expect("exact incremental request bytes fit after resident allocation");
    }

    #[test]
    fn explicit_all_logits_block_preserves_every_row() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([
            vec![11],
            vec![21, 22, 23],
        ]));
        let (mut core, _) = provisioned_core(recorder);
        assert_eq!(core.prefill(&[1]).unwrap().token_ids(), [11]);
        let block = core
            .decode_block_with_mtp_state_and_logits(&[2, 3, 4])
            .expect("all-logits block succeeds");
        assert_eq!(block.token_ids(), [21, 22, 23]);
        assert_eq!(
            block.logits_bf16().expect("all logits are published").len(),
            3 * QWEN35_VOCAB_SIZE
        );
        assert_eq!(
            block
                .hidden_states_bf16()
                .expect("all hidden rows are published")
                .len(),
            3 * 2_560
        );
    }

    #[test]
    fn mtp_target_prefill_preserves_every_argmax_and_hidden_row() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([vec![11, 12, 13]]));
        let (mut core, _) = provisioned_core(recorder);

        let output = core
            .prefill_with_mtp_state(&[1, 2, 3])
            .expect("MTP target prefill succeeds");
        assert_eq!(output.token_ids(), [11, 12, 13]);
        assert_eq!(
            output
                .hidden_states_bf16()
                .expect("MTP target hidden rows are published")
                .len(),
            3 * 2_560
        );
    }

    #[test]
    fn embedding_prefill_reads_final_normalized_rows_for_boundary_token_counts() {
        for token_count in [1_u64, 3, 17] {
            let (graph, plan) =
                crate::qwen_graph::qwen35_execution_fixture_with_token_count(token_count);
            let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([
                vec![31; usize::try_from(token_count).unwrap()],
            ]));
            let session = Arc::new(ExecutionSession::new("recorder", recorder.clone()));
            let mut core = QwenExecutionCore::provision(
                session,
                graph.clone(),
                plan.clone(),
                Duration::from_millis(1),
                &TestProvisionSource::default(),
            )
            .expect("embedding fixture provisions");
            let output = core
                .prefill_with_embeddings(&vec![1; usize::try_from(token_count).unwrap()])
                .expect("embedding prefill succeeds");
            assert!(output.token_ids().is_empty());
            assert_eq!(output.last_logits(), None);
            assert_eq!(output.hidden_states_bf16(), None);
            assert_eq!(
                output.embeddings_bf16().map(|rows| rows.len()),
                Some(usize::try_from(token_count).unwrap() * 2_560)
            );
            assert!(
                !recorder
                    .events()
                    .iter()
                    .any(|event| event == "argmax-readback-start"),
                "embedding mode must not read back generation tokens"
            );
            let final_nodes = graph
                .nodes()
                .iter()
                .filter(|node| node.label() == "final_rmsnorm")
                .collect::<Vec<_>>();
            assert_eq!(final_nodes.len(), 1);
            assert_eq!(final_nodes[0].outputs().len(), 1);

            let normal_recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([
                vec![31; usize::try_from(token_count).unwrap()],
            ]));
            let normal_session =
                Arc::new(ExecutionSession::new("recorder", normal_recorder.clone()));
            let mut normal = QwenExecutionCore::provision(
                normal_session,
                graph,
                plan,
                Duration::from_millis(1),
                &TestProvisionSource::default(),
            )
            .expect("normal fixture provisions");
            let normal_output = normal
                .prefill(&vec![1; usize::try_from(token_count).unwrap()])
                .expect("normal prefill succeeds");
            assert_eq!(
                normal_output.token_ids(),
                vec![31; usize::try_from(token_count).unwrap()].as_slice()
            );
        }
    }

    #[test]
    fn large_mtp_target_prefill_compacts_argmax_and_preserves_hidden_rows() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([vec![13]]));
        let (graph, plan) = crate::qwen_graph::qwen35_execution_fixture_with_token_count(256);
        let session = Arc::new(ExecutionSession::new("recorder", recorder));
        let mut core = QwenExecutionCore::provision(
            session,
            graph,
            plan,
            Duration::from_millis(1),
            &TestProvisionSource::default(),
        )
        .expect("large structural fixture provisions");

        let output = core
            .prefill_with_mtp_state(&vec![1; 256])
            .expect("large MTP target prefill succeeds");
        assert_eq!(output.token_ids(), [13]);
        assert_eq!(
            output
                .hidden_states_bf16()
                .expect("all large MTP target hidden rows are published")
                .len(),
            256 * 2_560
        );
    }

    #[test]
    fn large_target_graph_preserves_every_speculative_verify_row() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([
            vec![13],
            vec![21, 22],
        ]));
        let (graph, plan) = crate::qwen_graph::qwen35_execution_fixture_with_token_count(256);
        let session = Arc::new(ExecutionSession::new("recorder", recorder));
        let mut core = QwenExecutionCore::provision(
            session,
            graph,
            plan,
            Duration::from_millis(1),
            &TestProvisionSource::default(),
        )
        .expect("large structural fixture provisions");

        let prefill = core
            .prefill_with_mtp_state(&vec![1; 254])
            .expect("large MTP target prefill succeeds");
        assert_eq!(prefill.token_ids(), [13]);
        let block = core
            .decode_block_with_mtp_state(&[2, 3])
            .expect("large target verify block succeeds");
        assert_eq!(block.token_ids(), [21, 22]);
        assert_eq!(
            block
                .hidden_states_bf16()
                .expect("target verify hidden rows are published")
                .len(),
            2 * 2_560
        );
        core.resolve_decode_block(2)
            .expect("complete target verify block resolves");
    }

    #[test]
    fn mtp_graph_requires_hidden_row_and_advances_opaque_state() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([
            vec![11],
            vec![12],
        ]));
        let (graph, plan) = crate::qwen_graph::qwen35_mtp_execution_fixture();
        let session = Arc::new(ExecutionSession::new("recorder", recorder.clone()));
        let mut core = QwenExecutionCore::provision(
            session,
            graph,
            plan,
            Duration::from_millis(1),
            &TestProvisionSource::default(),
        )
        .expect("MTP fixture provisions");
        let first = core.prefill_mtp(7, &[0; 2_560]).expect("MTP prefill");
        assert_eq!(first.token_ids(), &[11]);
        assert_eq!(first.hidden_states_bf16().unwrap().len(), 2_560);
        let second = core.decode_mtp(11, &[0; 2_560]).expect("MTP decode");
        assert_eq!(second.token_ids(), &[12]);
        assert_eq!(second.committed_length(), 2);
        assert!(
            recorder
                .events()
                .iter()
                .any(|event| event == "create-kv:32")
        );
        assert!(
            recorder
                .events()
                .iter()
                .any(|event| event == "kv-append:32:1:1")
        );
    }

    #[test]
    fn device_memory_preflight_rejects_before_queue_or_allocation() {
        let recorder = Arc::new(ExecutionRecorder::default());
        recorder.available_memory_bytes.store(1, Ordering::Relaxed);
        let (graph, plan) = crate::qwen_graph::qwen35_execution_fixture();
        let session = Arc::new(ExecutionSession::new("recorder", recorder.clone()));
        let result = QwenExecutionCore::provision(
            session,
            graph,
            plan,
            Duration::from_millis(1),
            &TestProvisionSource::default(),
        );
        let error = match result {
            Ok(_) => panic!("insufficient device memory must fail closed"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("device memory preflight requires")
        );
        assert!(error.to_string().contains("but only 1 bytes are available"));
        assert!(recorder.events().is_empty());
    }

    #[test]
    fn audit_snapshot_rejects_empty_and_poisoned_audits() {
        let recorder = Arc::new(ExecutionRecorder::default());
        let (core, _) = provisioned_core(recorder);
        assert!(matches!(
            core.audit_snapshot(),
            Err(QwenExecutionError::InvalidRequest(_))
        ));
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = core.audit.lock().unwrap();
            panic!("poison audit mutex for fail-closed test");
        }));
        assert!(matches!(
            core.audit_snapshot(),
            Err(QwenExecutionError::Poisoned)
        ));
    }

    #[test]
    fn prefix_owner_forks_all_state_layers_and_continues_without_mutating_source() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([
            vec![101, 102, 103],
            vec![201],
        ]));
        let (graph, plan) = crate::qwen_graph::qwen35_execution_fixture();
        let session = Arc::new(ExecutionSession::new("recorder", recorder));
        let source = TestProvisionSource::default();
        let mut source_core = QwenExecutionCore::provision(
            Arc::clone(&session),
            graph.clone(),
            plan.clone(),
            Duration::from_millis(1),
            &source,
        )
        .expect("source provisions");
        let source_output = source_core.prefill(&[1, 2, 3]).expect("source prefill");
        let prefix = source_core.publish_prefix().expect("prefix publishes");

        assert_eq!(prefix.committed_length(), 3);
        assert_eq!(prefix.cached_terminal_output(), &source_output);
        assert_eq!(prefix.fork_audit().kv_states(), 8);
        assert_eq!(prefix.fork_audit().linear_states(), 24);
        assert_eq!(prefix.fork_audit().shared_pages(), 8);
        assert_eq!(prefix.fork_audit().copied_bytes(), 24);
        assert_eq!(prefix.fork_audit().cache_resident_bytes(), 72);
        assert_eq!(source_core.committed_length, 3);
        source_core
            .ensure_state_lengths(3)
            .expect("source remains exact");

        let mut continuation =
            QwenExecutionCore::provision(session, graph, plan, Duration::from_millis(1), &source)
                .expect("continuation provisions");
        continuation
            .install_prefix(&prefix)
            .expect("prefix installs transactionally");
        assert_eq!(continuation.committed_length, 3);
        assert_eq!(continuation.last_output.as_ref(), Some(&source_output));

        let empty = continuation
            .decode_continuation(&[])
            .expect("empty suffix uses cached output");
        assert_eq!(empty, source_output);
        let suffix = continuation
            .decode_continuation(&[4, 5, 6, 7])
            .expect("non-empty suffix uses continuation chunks");
        assert_eq!(suffix.token_ids(), [201]);
        assert_eq!(suffix.committed_length(), 7);
        continuation
            .ensure_state_lengths(7)
            .expect("continuation states remain synchronized");
        assert_eq!(prefix.committed_length(), 3);
        assert_eq!(prefix.cached_terminal_output().committed_length(), 3);
        source_core
            .ensure_state_lengths(3)
            .expect("source is isolated");
    }

    #[test]
    fn qwen_state_image_exports_all_layers_and_restores_transactionally() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([
            vec![101, 102, 103],
            vec![201],
        ]));
        let (graph, plan) = crate::qwen_graph::qwen35_execution_fixture();
        let session = Arc::new(ExecutionSession::new("recorder", recorder.clone()));
        let source = TestProvisionSource::default();
        let mut source_core = QwenExecutionCore::provision(
            Arc::clone(&session),
            graph.clone(),
            plan.clone(),
            Duration::from_millis(1),
            &source,
        )
        .expect("source provisions");
        let output = source_core.prefill(&[1, 2, 3]).expect("source prefill");
        let image = source_core
            .export_state_image()
            .expect("state image export");
        assert_eq!(image.session_id(), session.id());
        assert_eq!(image.committed_length(), 3);
        assert_eq!(image.kv_layers().len(), 8);
        assert_eq!(image.linear_layers().len(), 24);
        assert_eq!(image.cached_terminal_output(), Some(&output));
        assert!(
            image
                .kv_layers()
                .values()
                .all(|layer| layer.image().metadata().published_length == 3)
        );
        assert!(
            image
                .linear_layers()
                .values()
                .all(|layer| matches!(layer.image().metadata().active_slot, Some(0 | 1)))
        );

        let mut wrong_adapter = image.clone();
        wrong_adapter.identity.adapter_identity = "adapter:other-v1".to_owned();

        let mut restored = QwenExecutionCore::provision(
            Arc::clone(&session),
            graph.clone(),
            plan.clone(),
            Duration::from_millis(1),
            &source,
        )
        .expect("restore destination provisions");
        assert!(matches!(
            restored.restore_state_image(&wrong_adapter),
            Err(QwenExecutionError::InvalidRequest(_))
        ));
        assert_eq!(recorder.state_image_import_calls(), 0);
        restored
            .restore_state_image(&image)
            .expect("all state image layers restore");
        assert_eq!(restored.committed_length, 3);
        assert_eq!(restored.rope_position_delta, image.rope_position_delta());
        assert_eq!(restored.last_output.as_ref(), Some(&output));
        restored
            .ensure_state_lengths(3)
            .expect("restored layer lengths match");
        assert!(recorder.state_image_import_calls() >= 32);
    }

    #[test]
    fn qwen_resident_model_state_image_factory_restores_fresh_request() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([
            vec![101, 102, 103],
            vec![201],
        ]));
        let (graph, plan) = crate::qwen_graph::qwen35_execution_fixture();
        let session = Arc::new(ExecutionSession::new("recorder", recorder));
        let source = TestProvisionSource::default();
        let mut source_core = QwenExecutionCore::provision(
            Arc::clone(&session),
            graph.clone(),
            plan.clone(),
            Duration::from_millis(1),
            &source,
        )
        .expect("source provisions");
        source_core.prefill(&[1, 2, 3]).expect("source prefill");
        let image = source_core
            .export_state_image()
            .expect("state image export");
        let resident_inner = QwenResidentInner::provision(
            session,
            graph.clone(),
            plan.clone(),
            Duration::from_millis(1),
            &source,
        )
        .expect("resident provisions");
        let resident = QwenResidentModel {
            inner: Arc::new(resident_inner),
        };
        let request = resident
            .new_request_from_state_image(&image, graph)
            .expect("resident state-image factory restores");
        assert_eq!(request.committed_length(), 3);
        assert_eq!(request.last_output(), image.cached_terminal_output());
    }

    #[test]
    fn qwen_checkpoint_round_trip_cross_session_checks_identity_positions_and_suffix() {
        let source_recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([vec![
            101, 102, 103,
        ]]));
        let (graph, plan) = crate::qwen_graph::qwen35_execution_fixture();
        let source_session = Arc::new(ExecutionSession::new("source", source_recorder));
        let source = TestProvisionSource::default();
        let mut source_core = QwenExecutionCore::provision(
            Arc::clone(&source_session),
            graph.clone(),
            plan.clone(),
            Duration::from_millis(1),
            &source,
        )
        .expect("source provisions");
        source_core.prefill(&[1, 2, 3]).expect("source prefill");
        let image = source_core.export_state_image().expect("source image");
        let kv_encoding = image
            .kv_layers()
            .values()
            .next()
            .expect("KV layers")
            .descriptor()
            .cache_encoding();
        let identity = CheckpointIdentity::for_tokens(
            image.model_fingerprint().to_owned(),
            "artifact-v1",
            image.adapter_identity().to_owned(),
            "renderer-v1",
            "tokenizer-v1",
            "gfx942",
            qwen_hex_digest(image.plan_digest()),
            &[1, 2, 3],
            kv_encoding,
            image.kv_descriptor_digest(),
            [7; 32],
        )
        .expect("checkpoint identity");
        let mut wrong_encoding = identity.clone();
        wrong_encoding.kv_encoding = KvCacheEncoding::Fp8E4M3FnStatic;
        assert!(matches!(
            image.to_checkpoint(
                wrong_encoding,
                &[1, 2, 3],
                b"conversation",
                b"sampler",
                b"grammar",
                b"stop",
                3,
                3,
                1,
            ),
            Err(QwenExecutionError::InvalidRequest(reason))
                if reason.contains("KV encoding/descriptors")
        ));
        let checkpoint = image
            .to_checkpoint(
                identity.clone(),
                &[1, 2, 3],
                b"conversation",
                b"sampler",
                b"grammar",
                b"stop",
                3,
                3,
                1,
            )
            .expect("checkpoint flatten");
        let encoded = checkpoint.encode().expect("checkpoint encoding");
        let decoded = SessionCheckpoint::decode_with_identity(&encoded, Some(&identity))
            .expect("checkpoint decoding");
        assert_eq!(decoded.payload.token_history, [1, 2, 3]);
        assert_eq!(
            decoded.header.absolute_position - decoded.header.logical_position,
            0
        );
        assert!(image.cached_terminal_output().is_some());

        // The destination has a different execution session. Checkpoint
        // restore must succeed while raw QwenStateImage restore rejects it.
        let destination_recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([vec![201]]));
        let destination_session = Arc::new(ExecutionSession::new(
            "destination",
            destination_recorder.clone(),
        ));
        let mut destination = QwenExecutionCore::provision(
            Arc::clone(&destination_session),
            graph.clone(),
            plan.clone(),
            Duration::from_millis(1),
            &source,
        )
        .expect("destination provisions");
        assert!(matches!(
            destination.restore_state_image(&image),
            Err(QwenExecutionError::InvalidRequest(reason)) if reason.contains("different execution session")
        ));
        destination
            .restore_checkpoint(&decoded, &identity)
            .expect("cross-session checkpoint restore");
        assert_eq!(destination.committed_length, 3);
        let suffix = destination
            .decode_continuation(&[4])
            .expect("non-empty suffix continuation");
        assert_eq!(suffix.committed_length(), 4);

        let mut wrong_identity = identity.clone();
        wrong_identity.adapter_identity = "different-adapter".to_owned();
        assert!(matches!(
            destination.restore_checkpoint(&decoded, &wrong_identity),
            Err(QwenExecutionError::InvalidRequest(reason)) if reason.contains("frontend identity")
        ));
        assert_eq!(destination_recorder.state_image_import_calls(), 32);

        let mut malformed = decoded.clone();
        malformed.header.absolute_position = 2;
        assert!(matches!(
            destination.restore_checkpoint(&malformed, &identity),
            Err(QwenExecutionError::InvalidRequest(reason)) if reason.contains("absolute position") || reason.contains("checkpoint")
        ));
        assert_eq!(destination.committed_length, 4);
    }

    #[test]
    fn qwen_checkpoint_rejects_missing_topology_before_import() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([vec![
            101, 102, 103,
        ]]));
        let (graph, plan) = crate::qwen_graph::qwen35_execution_fixture();
        let session = Arc::new(ExecutionSession::new("recorder", recorder.clone()));
        let source = TestProvisionSource::default();
        let mut core = QwenExecutionCore::provision(
            Arc::clone(&session),
            graph.clone(),
            plan.clone(),
            Duration::from_millis(1),
            &source,
        )
        .expect("provision");
        core.prefill(&[1, 2, 3]).expect("prefill");
        let image = core.export_state_image().expect("image");
        let encoding = image
            .kv_layers()
            .values()
            .next()
            .unwrap()
            .descriptor()
            .cache_encoding();
        let identity = CheckpointIdentity::for_tokens(
            image.model_fingerprint(),
            "artifact-v1",
            image.adapter_identity(),
            "renderer-v1",
            "tokenizer-v1",
            "recorder",
            qwen_hex_digest(image.plan_digest()),
            &[1, 2, 3],
            encoding,
            image.kv_descriptor_digest(),
            [0; 32],
        )
        .unwrap();
        let mut checkpoint = image
            .to_checkpoint(identity.clone(), &[1, 2, 3], &[], &[], &[], &[], 3, 3, 1)
            .unwrap();
        checkpoint.payload.state_layers.pop();
        let mut destination =
            QwenExecutionCore::provision(session, graph, plan, Duration::from_millis(1), &source)
                .expect("destination provision");
        assert!(
            destination
                .restore_checkpoint(&checkpoint, &identity)
                .is_err()
        );
        assert_eq!(recorder.state_image_import_calls(), 0);
        assert_eq!(destination.committed_length, 0);
    }

    #[test]
    fn qwen_state_image_without_terminal_output_rejects_empty_suffix_but_continues_nonempty() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([
            vec![101, 102, 103],
            vec![201],
        ]));
        let (graph, plan) = crate::qwen_graph::qwen35_execution_fixture();
        let session = Arc::new(ExecutionSession::new("recorder", recorder));
        let source = TestProvisionSource::default();
        let mut source_core = QwenExecutionCore::provision(
            Arc::clone(&session),
            graph.clone(),
            plan.clone(),
            Duration::from_millis(1),
            &source,
        )
        .expect("source provisions");
        source_core.prefill(&[1, 2, 3]).expect("source prefill");
        let image = source_core
            .export_state_image()
            .expect("state image export")
            .without_terminal_output();

        let mut restored =
            QwenExecutionCore::provision(session, graph, plan, Duration::from_millis(1), &source)
                .expect("restore destination provisions");
        restored
            .restore_state_image(&image)
            .expect("state image without output restores");
        assert!(matches!(
            restored.decode_continuation(&[]),
            Err(QwenExecutionError::InvalidRequest(reason)) if reason.contains("cached terminal")
        ));
        let output = restored
            .decode_continuation(&[4])
            .expect("non-empty suffix resumes restored state");
        assert_eq!(output.committed_length(), 4);
    }

    #[test]
    fn qwen_state_image_rejects_wrong_missing_duplicate_and_mixed_lengths_before_import() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([vec![
            101, 102, 103,
        ]]));
        let (graph, plan) = crate::qwen_graph::qwen35_execution_fixture();
        let session = Arc::new(ExecutionSession::new("recorder", recorder.clone()));
        let source = TestProvisionSource::default();
        let mut source_core = QwenExecutionCore::provision(
            Arc::clone(&session),
            graph.clone(),
            plan.clone(),
            Duration::from_millis(1),
            &source,
        )
        .expect("source provisions");
        source_core.prefill(&[1, 2, 3]).expect("source prefill");
        let valid = source_core
            .export_state_image()
            .expect("state image export");

        let mut wrong_layer = valid.clone();
        let first_layer = *wrong_layer.kv_layers.keys().next().unwrap();
        let entry = wrong_layer.kv_layers.get_mut(&first_layer).unwrap();
        entry.image = ExecutionStateImageV1::new(
            crate::StateLayerMetadataV1 {
                owner: StateOwnerKindV1::Kv,
                layer_id: first_layer + 1,
                published_length: 3,
                generation: 1,
                active_slot: None,
            },
            entry.image.planes().to_vec(),
        );
        let mut destination = QwenExecutionCore::provision(
            Arc::clone(&session),
            graph.clone(),
            plan.clone(),
            Duration::from_millis(1),
            &source,
        )
        .expect("destination provisions");
        assert!(matches!(
            destination.restore_state_image(&wrong_layer),
            Err(QwenExecutionError::InvalidRequest(reason)) if reason.contains("owner or layer")
        ));
        assert_eq!(recorder.state_image_import_calls(), 0);

        let mut missing = valid.clone();
        missing.kv_layers.remove(&first_layer);
        assert!(matches!(
            destination.restore_state_image(&missing),
            Err(QwenExecutionError::InvalidRequest(reason)) if reason.contains("layer set")
        ));
        assert_eq!(recorder.state_image_import_calls(), 0);

        let mut duplicate = valid.clone();
        let duplicate_layer = *duplicate.kv_layers.keys().next().unwrap();
        let duplicate_image = duplicate
            .kv_layers
            .get(&duplicate_layer)
            .unwrap()
            .image
            .clone();
        let mut duplicate_planes = duplicate_image.planes().to_vec();
        duplicate_planes[1].plane = duplicate_planes[0].plane;
        duplicate.kv_layers.get_mut(&duplicate_layer).unwrap().image =
            ExecutionStateImageV1::new(duplicate_image.metadata().clone(), duplicate_planes);
        assert!(matches!(
            destination.restore_state_image(&duplicate),
            Err(QwenExecutionError::InvalidRequest(reason)) if reason.contains("duplicate")
        ));
        assert_eq!(recorder.state_image_import_calls(), 0);

        let mut mixed = valid.clone();
        let second_layer = *mixed.kv_layers.keys().nth(1).unwrap();
        let second_image = mixed.kv_layers.get(&second_layer).unwrap().image.clone();
        let mut metadata = second_image.metadata().clone();
        metadata.published_length = 2;
        mixed.kv_layers.get_mut(&second_layer).unwrap().image =
            ExecutionStateImageV1::new(metadata, second_image.planes().to_vec());
        assert!(matches!(
            destination.restore_state_image(&mixed),
            Err(QwenExecutionError::InvalidRequest(reason)) if reason.contains("published length")
        ));
        assert_eq!(recorder.state_image_import_calls(), 0);
    }

    #[test]
    fn qwen_state_image_adapter_failure_does_not_publish_destination() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([vec![
            101, 102, 103,
        ]]));
        let (graph, plan) = crate::qwen_graph::qwen35_execution_fixture();
        let session = Arc::new(ExecutionSession::new("recorder", recorder.clone()));
        let source = TestProvisionSource::default();
        let mut source_core = QwenExecutionCore::provision(
            Arc::clone(&session),
            graph.clone(),
            plan.clone(),
            Duration::from_millis(1),
            &source,
        )
        .expect("source provisions");
        source_core.prefill(&[1, 2, 3]).expect("source prefill");
        let image = source_core
            .export_state_image()
            .expect("state image export");
        recorder.set_state_image_failure(true);

        let mut destination =
            QwenExecutionCore::provision(session, graph, plan, Duration::from_millis(1), &source)
                .expect("destination provisions");
        assert!(matches!(
            destination.restore_state_image(&image),
            Err(QwenExecutionError::Execution(
                ExecutionError::BackendStatus { status: 91, .. }
            ))
        ));
        assert_eq!(destination.committed_length, 0);
        assert!(destination.last_output.is_none());
        assert!(recorder.state_image_import_calls() > 0);
    }

    #[test]
    fn qwen_state_image_encoding_plane_contract_covers_fp16_fp8_static_and_nvfp4() {
        let expected = [
            (KvCacheEncoding::Fp16, 2),
            (KvCacheEncoding::Fp8E4M3Fn, 4),
            (KvCacheEncoding::Fp8E4M3FnStatic, 2),
            (KvCacheEncoding::Nvfp4, 6),
        ];
        for (encoding, count) in expected {
            let descriptor = KvStateDescriptor::new_with_storage(0, 3, 4, 256, encoding).unwrap();
            assert_eq!(descriptor.plane_kinds().len(), count);
        }
        assert_eq!(
            LinearAttentionStateDescriptor::new(0, 3)
                .unwrap()
                .plane_kinds()
                .len(),
            5
        );
    }

    #[test]
    fn full_prefill_decode_records_graph_order_and_publishes_after_state_snapshots() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([
            vec![101, 102, 103],
            vec![104],
            vec![105],
        ]));
        let (mut core, source) = provisioned_core(Arc::clone(&recorder));

        let uploaded = source.uploaded();
        assert_eq!(uploaded.len(), core.graph.weight_bindings().len());
        assert_eq!(
            uploaded.iter().collect::<BTreeSet<_>>().len(),
            core.graph.weight_bindings().len()
        );
        assert_eq!(core.kv_states.len(), 8);
        assert_eq!(core.linear_states.len(), 24);
        assert_eq!(core.scales.len(), 16);
        for cached in core.scales.values() {
            assert_eq!(cached.raw_bytes.len(), 512);
            assert!(
                cached
                    .expanded_bytes
                    .chunks_exact(cached.raw_bytes.len())
                    .all(|chunk| chunk == cached.raw_bytes.as_ref())
            );
        }

        let q = core.tensor_id("layer.3.full.q.output").unwrap();
        let q_gate = core.tensor_id("layer.3.full.q_gate.packed").unwrap();
        assert_eq!(
            core.tensors[q].buffer.id(),
            core.tensors[q_gate].buffer.id()
        );
        let embedding = core
            .graph
            .nodes()
            .iter()
            .find(|node| {
                matches!(
                    node.kind(),
                    QwenGraphNodeKind::Semantic(SemanticOpKind::Embedding)
                )
            })
            .unwrap();
        let tied = core
            .graph
            .nodes()
            .iter()
            .find(|node| node.label() == "tied_lm_head_matmul")
            .unwrap();
        assert_eq!(embedding.inputs()[0], tied.inputs()[1]);
        assert_eq!(
            core.tensors[embedding.inputs()[0]].buffer.id(),
            core.tensors[tied.inputs()[1]].buffer.id()
        );

        let prefill = core.prefill(&[7, 8, 9]).expect("prefill succeeds");
        assert_eq!(prefill.token_ids(), &[101, 102, 103]);
        assert_eq!(prefill.committed_length(), 3);
        assert_eq!(core.committed_length, 3);
        for state in core.kv_states.values() {
            assert_eq!(state.snapshot(core.session.as_ref()).unwrap().length(), 3);
        }
        for state in core.linear_states.values() {
            assert_eq!(state.snapshot(core.session.as_ref()).unwrap().length(), 3);
        }

        let decode = core.decode(10).expect("decode succeeds");
        assert_eq!(decode.token_ids(), &[104]);
        assert_eq!(decode.committed_length(), 4);
        assert_eq!(core.last_output.as_ref(), Some(&decode));
        let audit = core
            .audit_snapshot()
            .expect("successful transition is audited");
        assert_eq!(audit.selected_backend(), "hip");
        assert_eq!(audit.target(), "recorder");
        assert!(audit.submission_count() > 0);
        assert!(audit.kernel_dispatch_count() >= audit.submission_count());
        assert!(!audit.fallback_used());
        assert!(audit.all_dispatches_hip());
        assert!(audit.segment_count() > 0);
        assert!(audit.boundary_count() >= audit.segment_count());
        for state in core.kv_states.values() {
            assert_eq!(state.snapshot(core.session.as_ref()).unwrap().length(), 4);
        }
        for state in core.linear_states.values() {
            assert_eq!(state.snapshot(core.session.as_ref()).unwrap().length(), 4);
        }

        let prepares_before_second_decode = recorder
            .events()
            .iter()
            .filter(|event| event.starts_with("prepare:"))
            .count();
        let second_decode = core.decode(11).expect("second decode succeeds");
        assert_eq!(second_decode.token_ids(), &[105]);
        assert_eq!(second_decode.committed_length(), 5);
        let prepares_after_second_decode = recorder
            .events()
            .iter()
            .filter(|event| event.starts_with("prepare:"))
            .count();
        assert_eq!(
            prepares_after_second_decode - prepares_before_second_decode,
            8,
            "only the position-dependent attention preprocess nodes are re-prepared"
        );

        let preprocess = recorder.preprocess();
        assert_eq!(preprocess.len(), 24);
        assert!(
            preprocess[..8]
                .iter()
                .all(|entry| { *entry == (AttentionPreprocessPositionMode::Prefill, 0, 3) })
        );
        assert!(preprocess[8..16].iter().all(|entry| {
            *entry == (AttentionPreprocessPositionMode::DecodeContinuation, 3, 1)
        }));
        assert!(preprocess[16..].iter().all(|entry| {
            *entry == (AttentionPreprocessPositionMode::DecodeContinuation, 4, 1)
        }));
        let events = recorder.events();
        let linear = events
            .iter()
            .position(|event| event == "linear:0:0:3")
            .unwrap();
        let full_preprocess = events
            .iter()
            .position(|event| event == "prepare:AttentionPreprocess")
            .unwrap();
        let kv_append = events
            .iter()
            .position(|event| event == "kv-append:3:0:3")
            .unwrap();
        let causal = events
            .iter()
            .position(|event| event == "causal:3:0:3")
            .unwrap();
        assert!(linear < full_preprocess && full_preprocess < kv_append && kv_append < causal);
        let readback = events
            .iter()
            .rposition(|event| event == "readback-bytes")
            .unwrap();
        assert!(
            events[readback + 1..]
                .iter()
                .any(|event| event.starts_with("kv-snapshot:"))
        );
        assert_eq!(recorder.shutdown_calls.load(Ordering::Relaxed), 0);
        drop(core);
        assert_eq!(recorder.shutdown_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn optional_last_logits_readback_is_one_full_bf16_vocab_row() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([vec![4, 5, 6]]));
        let (mut core, _) = provisioned_core(Arc::clone(&recorder));
        let output = core
            .prefill_with_last_logits(&[1, 2, 3])
            .expect("prefill with logits succeeds");
        assert_eq!(output.token_ids(), [4, 5, 6]);
        let logits = output.last_logits().expect("last logits are published");
        assert_eq!(logits.len(), QWEN35_VOCAB_SIZE);
        assert!(logits.iter().all(|value| value.to_bits() == 0));
        assert!(
            recorder
                .events()
                .iter()
                .any(|event| event == "argmax-readback-start")
        );
        assert_eq!(
            recorder
                .events()
                .iter()
                .filter(|event| event.as_str() == "readback-bytes")
                .count(),
            2,
        );
    }

    #[test]
    fn device_selector_reads_only_the_fixed_record_and_preserves_metadata() {
        let recorder = Arc::new(ExecutionRecorder::default());
        let (mut core, _) = provisioned_core(Arc::clone(&recorder));
        let parameters = crate::SamplingParametersV1::new(1.0, 1.0, 0.0, 0.0).unwrap();
        let chain =
            crate::SamplerChainV1::new(crate::SamplerChainConfigV1::new(parameters), &[]).unwrap();
        let selector = chain
            .prepare_device_selector(QWEN35_VOCAB_SIZE, None, 17, 3)
            .unwrap();
        let output = core
            .prefill_with_device_selector(&[1], &selector)
            .expect("device selector route succeeds on recorder");
        assert_eq!(output.token_ids(), [0]);
        assert!(output.last_logits().is_none());
        assert_eq!(
            output.selection().map(|selection| selection.token_id),
            Some(0)
        );
        let events = recorder.events();
        assert!(events.iter().any(|event| event == "submit:TokenSelect"));
        assert!(
            events
                .iter()
                .any(|event| event == "token-selector-readback-start")
        );
        assert!(!events.iter().any(|event| event == "argmax-readback-start"));

        let decode_recorder = Arc::new(ExecutionRecorder::default());
        let (mut decode_core, _) = provisioned_core(Arc::clone(&decode_recorder));
        decode_core
            .prefill(&[1])
            .expect("legacy prefill before selector decode");
        let argmax_readbacks_before_decode = decode_recorder
            .events()
            .iter()
            .filter(|event| event.as_str() == "argmax-readback-start")
            .count();
        let decode_output = decode_core
            .decode_with_device_selector(2, &selector)
            .expect("device selector decode succeeds on recorder");
        assert_eq!(decode_output.token_ids(), [0]);
        assert_eq!(
            decode_output
                .sampling_selection()
                .map(|selection| selection.token_id),
            Some(0)
        );
        assert_eq!(
            decode_recorder
                .events()
                .iter()
                .filter(|event| event.as_str() == "argmax-readback-start")
                .count(),
            argmax_readbacks_before_decode
        );
    }

    #[test]
    fn device_selector_rejects_mtp_without_fallback() {
        let recorder = Arc::new(ExecutionRecorder::default());
        let (graph, plan) = crate::qwen_graph::qwen35_mtp_execution_fixture();
        let session = Arc::new(ExecutionSession::new("recorder", recorder));
        let mut core = QwenExecutionCore::provision(
            session,
            graph,
            plan,
            Duration::from_millis(1),
            &TestProvisionSource::default(),
        )
        .expect("MTP fixture provisions");
        let parameters = crate::SamplingParametersV1::new(1.0, 1.0, 0.0, 0.0).unwrap();
        let chain =
            crate::SamplerChainV1::new(crate::SamplerChainConfigV1::new(parameters), &[]).unwrap();
        let selector = chain
            .prepare_device_selector(QWEN35_VOCAB_SIZE, None, 1, 0)
            .unwrap();
        assert!(matches!(
            core.prefill_with_device_selector(&[1], &selector),
            Err(QwenExecutionError::InvalidRequest(reason))
                if reason.contains("unsupported for MTP")
        ));
    }

    #[test]
    fn failure_after_state_mutation_poison_rejects_reuse_and_never_publishes() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([vec![1, 2, 3]]));
        recorder.set_failure(SemanticOpKind::Add);
        let (mut core, _) = provisioned_core(Arc::clone(&recorder));

        assert!(matches!(
            core.prefill(&[1, 2, 3]),
            Err(QwenExecutionError::CompletionFailure { .. })
        ));
        assert!(core.lifecycle.is_poisoned());
        assert_eq!(core.committed_length, 0);
        assert!(core.last_output.is_none());
        let first_linear = core.linear_states.get(&0).unwrap();
        assert_eq!(recorder.linear_length(first_linear), 3);
        assert!(matches!(
            core.prefill(&[1, 2, 3]),
            Err(QwenExecutionError::Poisoned)
        ));
        assert!(matches!(core.decode(4), Err(QwenExecutionError::Poisoned)));
        assert_eq!(recorder.shutdown_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn pending_completion_and_guard_drop_poison_without_output_publication() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([vec![1, 2, 3]]));
        recorder.set_pending(SemanticOpKind::RmsNorm);
        let (mut core, _) = provisioned_core(Arc::clone(&recorder));
        assert!(matches!(
            core.prefill(&[1, 2, 3]),
            Err(QwenExecutionError::CompletionPending { .. })
        ));
        assert!(core.lifecycle.is_poisoned());
        assert_eq!(core.committed_length, 0);
        assert!(core.last_output.is_none());

        let lifecycle = ExecutionTransaction::new();
        let guard = lifecycle.begin().unwrap();
        drop(guard);
        assert!(lifecycle.is_poisoned());
    }

    #[test]
    fn argmax_sentinel_is_not_published_after_state_updates() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([vec![1, 2, -1]]));
        let (mut core, _) = provisioned_core(Arc::clone(&recorder));
        assert!(matches!(
            core.prefill(&[1, 2, 3]),
            Err(QwenExecutionError::ArgmaxSentinel { index: 2 })
        ));
        assert!(core.lifecycle.is_poisoned());
        assert_eq!(core.committed_length, 0);
        assert!(core.last_output.is_none());
        assert_eq!(
            recorder.linear_length(core.linear_states.get(&0).unwrap()),
            3
        );
    }

    #[test]
    fn validation_and_position_bytes_cover_request_boundaries() {
        let recorder = Arc::new(ExecutionRecorder::default());
        let (mut core, _) = provisioned_core(recorder);
        assert!(matches!(
            core.decode(1),
            Err(QwenExecutionError::InvalidRequest(_))
        ));
        assert!(matches!(
            core.prefill(&[1; 18]),
            Err(QwenExecutionError::InvalidRequest(_))
        ));
        assert!(matches!(
            core.prefill(&[-1, 1, 2]),
            Err(QwenExecutionError::InvalidRequest(_))
        ));
        assert!(matches!(
            core.prefill(&[QWEN35_VOCAB_SIZE as i32, 1, 2]),
            Err(QwenExecutionError::InvalidRequest(_))
        ));
        assert!(!core.lifecycle.is_poisoned());
        assert_eq!(position_bytes(0, 3).unwrap(), i32_bytes(&[0, 1, 2]));
        assert_eq!(position_bytes(3, 1).unwrap(), i32_bytes(&[3]));
        assert!(position_bytes(i32::MAX as u64, 2).is_err());

        let (mut short, _) = provisioned_core(Arc::new(ExecutionRecorder::default()));
        assert_eq!(short.prefill(&[1]).unwrap().committed_length(), 1);
    }

    #[test]
    fn chunked_prefill_preserves_absolute_positions_and_publishes_only_the_final_chunk() {
        let recorder = Arc::new(ExecutionRecorder::default());
        let (mut core, _) = provisioned_core(Arc::clone(&recorder));
        let output = core
            .prefill(&[1, 2, 3, 4])
            .expect("chunked prefill succeeds");
        assert_eq!(output.committed_length(), 4);
        assert_eq!(output.token_ids(), [7]);
        assert_eq!(core.prefill_chunk_count, 2);
        assert_eq!(core.committed_length, 4);
        let preprocess = recorder.preprocess();
        assert!(
            preprocess
                .iter()
                .any(|entry| { *entry == (AttentionPreprocessPositionMode::Prefill, 0, 3) })
        );
        assert!(preprocess.iter().any(|entry| {
            *entry == (AttentionPreprocessPositionMode::DecodeContinuation, 3, 1)
        }));
        assert!(
            recorder
                .events()
                .iter()
                .any(|event| event == "linear:0:3:1")
        );
    }

    #[test]
    fn argmax_token_outside_vocabulary_poison_rejects_publication() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([vec![
            1,
            2,
            QWEN35_VOCAB_SIZE as i32,
        ]]));
        let (mut core, _) = provisioned_core(recorder);
        assert!(matches!(
            core.prefill(&[1, 2, 3]),
            Err(QwenExecutionError::InvalidRequest(_))
        ));
        assert!(core.lifecycle.is_poisoned());
        assert_eq!(core.committed_length, 0);
        assert!(core.last_output.is_none());
    }

    #[test]
    fn graph_identity_and_capacity_reject_before_a_transition_can_mutate_state() {
        let (graph, mut plan) = crate::qwen_graph::qwen35_execution_fixture();
        plan.lock_fingerprint = "tampered".to_owned();
        let recorder = Arc::new(ExecutionRecorder::default());
        let session = Arc::new(ExecutionSession::new("recorder", recorder.clone()));
        let source = TestProvisionSource::default();
        assert!(matches!(
            QwenExecutionCore::provision(session, graph, plan, Duration::from_millis(1), &source),
            Err(QwenExecutionError::InvalidGraph(_))
        ));
        assert!(!recorder.events().iter().any(|event| event == "queue"));

        let recorder = Arc::new(ExecutionRecorder::default());
        let (mut core, _) = provisioned_core(recorder);
        core.committed_length = core.graph.state_capacity();
        assert!(matches!(
            core.decode(7),
            Err(QwenExecutionError::InvalidRequest(_))
        ));
        assert!(!core.lifecycle.is_poisoned());
    }

    #[test]
    fn dispatch_audit_counts_kernel_dispatches_and_rejects_non_hip_evidence() {
        let mut audit = ExecutionAuditAccumulator::new(1);
        let first = dispatch_evidence();
        audit.record(&first).unwrap();
        let mut two_kernel = first.clone();
        two_kernel.dispatch_id = 2;
        two_kernel.dispatch_count = 2;
        audit.record(&two_kernel).unwrap();
        audit
            .record_boundary(ExecutionBoundaryKind::TerminalReadback, true)
            .unwrap();
        let snapshot = audit.snapshot().unwrap();
        assert_eq!(snapshot.target(), "recorder");
        assert_eq!(snapshot.submission_count(), 2);
        assert_eq!(snapshot.kernel_dispatch_count(), 3);
        assert_eq!(snapshot.backend(), 1);
        assert!(!snapshot.fallback_used());

        for invalid in [
            {
                let mut value = first.clone();
                value.backend = 2;
                value
            },
            {
                let mut value = first.clone();
                value.fallback_allowed = true;
                value
            },
            {
                let mut value = first.clone();
                value.fallback_used = true;
                value
            },
            {
                let mut value = first.clone();
                value.dispatch_count = 0;
                value
            },
        ] {
            assert!(ExecutionAuditAccumulator::new(1).record(&invalid).is_err());
        }

        let mut mixed_target = ExecutionAuditAccumulator::new(1);
        mixed_target.record(&first).unwrap();
        let mut wrong_target = first;
        wrong_target.target = "other".to_owned();
        assert!(mixed_target.record(&wrong_target).is_err());
    }

    #[test]
    fn resident_uploads_once_and_request_state_returns_to_resident() {
        let (graph, plan) = crate::qwen_graph::qwen35_execution_fixture();
        let layout = validate_graph_plan(&graph, &plan).expect("fixture layout validates");
        assert!(layout.workspace.baseline_bytes >= layout.workspace.high_water_bytes);
        assert!(layout.workspace.high_water_bytes > 0);
        let lock = crate::parse_model_lock(include_bytes!(
            "../../../docs/models/locks/qwen3.5-4b-bf16.json"
        ))
        .expect("fixed lock parses");
        let graph_with_different_shape =
            crate::build_qwen35_graph(&lock, &plan, 5, 19).expect("different request graph builds");
        let recorder = Arc::new(ExecutionRecorder::default());
        let session = Arc::new(ExecutionSession::new("recorder", recorder));
        let source = TestProvisionSource::default();
        let resident = Arc::new(
            QwenResidentInner::provision(
                Arc::clone(&session),
                graph.clone(),
                plan.clone(),
                Duration::from_millis(1),
                &source,
            )
            .expect("resident fixture provisions"),
        );
        let upload_count = source.uploaded().len();
        let resident_memory = session.memory_snapshot();
        assert!(resident_memory.model_resident().current_bytes() > 0);
        assert_eq!(resident_memory.request_state().current_bytes(), 0);
        assert_eq!(resident_memory.workspace().current_bytes(), 0);

        let request = QwenExecutionCore::from_resident(
            Arc::clone(&resident),
            graph.clone(),
            AdapterRequestSetV1::disabled(),
        )
        .expect("first fresh request provisions");
        let request_memory = session.memory_snapshot();
        assert!(request_memory.request_state().current_bytes() > 0);
        assert_eq!(
            request_memory.workspace().current_bytes(),
            layout.workspace.high_water_bytes
        );
        drop(request);
        let returned = session.memory_snapshot();
        assert_eq!(
            returned.model_resident().current_bytes(),
            resident_memory.model_resident().current_bytes()
        );
        assert_eq!(returned.request_state().current_bytes(), 0);
        assert_eq!(returned.workspace().current_bytes(), 0);

        let request = QwenExecutionCore::from_resident(
            resident.clone(),
            graph_with_different_shape,
            AdapterRequestSetV1::disabled(),
        )
        .expect("second fresh request provisions");
        drop(request);
        assert_eq!(source.uploaded().len(), upload_count);
        drop(resident);
        assert_eq!(session.memory_snapshot().current_bytes(), 0);
    }
}
