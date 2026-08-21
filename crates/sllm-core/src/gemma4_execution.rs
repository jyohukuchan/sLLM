//! Concrete tensor and buffer layout for the reviewed Gemma 4 graph.
//!
//! The structural graph deliberately does not own backend resources. This
//! module turns that graph into exact semantic descriptors and backing
//! identities before a backend is allowed to allocate or prepare anything.
//! In particular, decode K/V tails are represented as checked subviews of the
//! same request-state buffers later consumed as committed attention prefixes.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::context_window::{ContextShiftDecisionV1, ContextWindowStateV1};
use crate::gemma4::Gemma4LayerType;
use crate::gemma4_graph::{
    GEMMA4_HIDDEN_SIZE, Gemma4Graph, Gemma4GraphBindingClass, Gemma4GraphNodeKind, Gemma4NormRole,
};
use crate::kv_state::{KvCacheEncoding, KvStateDescriptor};
use crate::op::TokenSelectorContractV1;
use crate::op::{RmsNormContract, SemanticOpDescriptor, SemanticOpKind};
use crate::prepared_execution::{
    ExecutionAuditAccumulator, ExecutionBoundaryKind, ExecutionSegment, PreparedCachePolicy,
    PreparedDynamicIdentity, PreparedExecutionAudit, PreparedSemanticCache,
    require_terminal_success,
};
use crate::session_checkpoint::{
    CheckpointIdentity, CheckpointPayload, OpaqueStatePlane, SessionCheckpoint,
    StateLayerMetadataV1, StateOwnerKindV1, StatePlaneKindV1,
};
use crate::weights::{WeightClassification, WeightLoadEntry, WeightLoadPlan};
use crate::{
    AccessMode, AllocationCategory, BoundSemanticOp, DType, DeviceTokenSelectorRequestV1, Encoding,
    ExecutionBuffer, ExecutionQueue, ExecutionSession, ExecutionState, ExecutionStateImageV1,
    KvState, OwnedTensorBinding, PrepareSupport, QuantizedTensorEncoding, SamplingSelectionV1,
    ScalePlaneRole, StateForkAuditV1, TensorDType, TensorView, VerifiedCache,
    VerifiedGgufGemmaSource, VerifiedNvfp4Sidecar, VerifiedUnslothGemma4Nvfp4, WeightUploadRequest,
    upload_verified_weight,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Gemma4KvPlane {
    Key,
    Value,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Gemma4TensorBacking {
    ModelWeight { tensor_name: String },
    TokenIds,
    Positions,
    ConstantBf16 { bits: u16 },
    Workspace,
    RequestKv { layer: u32, plane: Gemma4KvPlane },
    Alias { tensor_id: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4ExecutionTensor {
    id: usize,
    name: String,
    view: TensorView,
    backing: Gemma4TensorBacking,
}

impl Gemma4ExecutionTensor {
    pub const fn id(&self) -> usize {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn view(&self) -> &TensorView {
        &self.view
    }

    pub fn backing(&self) -> &Gemma4TensorBacking {
        &self.backing
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gemma4KvAppendLayout {
    source_tensor: usize,
    state_tensor: usize,
    destination_view: usize,
}

impl Gemma4KvAppendLayout {
    pub const fn source_tensor(self) -> usize {
        self.source_tensor
    }

    pub const fn state_tensor(self) -> usize {
        self.state_tensor
    }

    pub const fn destination_view(self) -> usize {
        self.destination_view
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4ExecutionNode {
    graph_node_id: usize,
    descriptor: SemanticOpDescriptor,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
    kv_appends: Vec<Gemma4KvAppendLayout>,
}

impl Gemma4ExecutionNode {
    pub const fn graph_node_id(&self) -> usize {
        self.graph_node_id
    }

    pub fn descriptor(&self) -> &SemanticOpDescriptor {
        &self.descriptor
    }

    pub fn inputs(&self) -> &[usize] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[usize] {
        &self.outputs
    }

    pub fn kv_appends(&self) -> &[Gemma4KvAppendLayout] {
        &self.kv_appends
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4ExecutionLayout {
    model_fingerprint: String,
    nvfp4_sidecar_fingerprint: Option<String>,
    plan_digest: [u8; 32],
    tensors: Vec<Gemma4ExecutionTensor>,
    nodes: Vec<Gemma4ExecutionNode>,
    model_weight_bytes: u64,
    workspace_bytes: u64,
    request_state_bytes: u64,
}

impl Gemma4ExecutionLayout {
    pub fn model_fingerprint(&self) -> &str {
        &self.model_fingerprint
    }

    pub const fn plan_digest(&self) -> &[u8; 32] {
        &self.plan_digest
    }

    pub fn tensors(&self) -> &[Gemma4ExecutionTensor] {
        &self.tensors
    }

    pub fn nodes(&self) -> &[Gemma4ExecutionNode] {
        &self.nodes
    }

    pub const fn model_weight_bytes(&self) -> u64 {
        self.model_weight_bytes
    }

    pub const fn workspace_bytes(&self) -> u64 {
        self.workspace_bytes
    }

    pub const fn request_state_bytes(&self) -> u64 {
        self.request_state_bytes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4ExecutionLayoutError(String);

impl Gemma4ExecutionLayoutError {
    fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Gemma4ExecutionLayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid Gemma 4 execution layout: {}", self.0)
    }
}

impl std::error::Error for Gemma4ExecutionLayoutError {}

/// Backend-owned buffers for one exact layout.
///
/// Model weights use tensor-sized allocations so no single allocation exceeds
/// the bounded public-runtime range. `upload_verified_weight` still validates
/// each tensor against the packed destination identity in `WeightLoadPlan`.
/// Alias tensors clone the source buffer identity and only change the checked
/// view used by a semantic descriptor.
pub struct Gemma4ProvisionedBuffers {
    session: Arc<ExecutionSession>,
    buffers: Vec<ExecutionBuffer>,
    prepared_semantics: Arc<PreparedSemanticCache>,
}

/// Immutable Gemma weights and constants retained across request owners.
#[derive(Clone)]
pub struct Gemma4ResidentModel {
    inner: Arc<Gemma4ResidentInner>,
}

struct Gemma4ResidentInner {
    session: Arc<ExecutionSession>,
    queue: ExecutionQueue,
    lock: crate::Gemma4ModelLock,
    plan: WeightLoadPlan,
    nvfp4_sidecar: Option<Arc<VerifiedNvfp4Sidecar>>,
    quantized_model: Option<Arc<dyn GemmaQuantizedSource>>,
    immutable: BTreeMap<String, (Gemma4TensorBacking, ExecutionBuffer)>,
    completion_timeout: Duration,
}

trait GemmaQuantizedSource: Send + Sync {
    fn repository(&self) -> &str;
    fn resolved_revision(&self) -> &str;
    fn recipe_digest(&self) -> &str;
    fn tensor(&self, name: &str) -> Option<&crate::QuantizedTensorDescriptor>;
    fn kv_scale(&self, layer: u32) -> Option<crate::StaticFp8KvScale>;
    fn resident_bytes(
        &self,
        descriptor: &crate::QuantizedTensorDescriptor,
    ) -> Result<Vec<u8>, Gemma4ExecutionLayoutError>;
}

impl GemmaQuantizedSource for VerifiedUnslothGemma4Nvfp4 {
    fn repository(&self) -> &str {
        self.repository()
    }

    fn resolved_revision(&self) -> &str {
        self.resolved_revision()
    }

    fn recipe_digest(&self) -> &str {
        self.recipe_digest()
    }

    fn tensor(&self, name: &str) -> Option<&crate::QuantizedTensorDescriptor> {
        self.tensor(name)
    }

    fn kv_scale(&self, layer: u32) -> Option<crate::StaticFp8KvScale> {
        self.kv_scale(layer)
    }

    fn resident_bytes(
        &self,
        descriptor: &crate::QuantizedTensorDescriptor,
    ) -> Result<Vec<u8>, Gemma4ExecutionLayoutError> {
        build_unsloth_gemma_resident_bytes(self, descriptor)
    }
}

impl GemmaQuantizedSource for VerifiedGgufGemmaSource {
    fn repository(&self) -> &str {
        self.repository()
    }

    fn resolved_revision(&self) -> &str {
        self.resolved_revision()
    }

    fn recipe_digest(&self) -> &str {
        self.recipe_digest()
    }

    fn tensor(&self, name: &str) -> Option<&crate::QuantizedTensorDescriptor> {
        self.tensor(name)
    }

    fn kv_scale(&self, layer: u32) -> Option<crate::StaticFp8KvScale> {
        self.kv_scale(layer)
    }

    fn resident_bytes(
        &self,
        descriptor: &crate::QuantizedTensorDescriptor,
    ) -> Result<Vec<u8>, Gemma4ExecutionLayoutError> {
        self.resident_bytes(&descriptor.logical_name)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4ExecutionOutput {
    token_ids: Vec<i32>,
    last_logits: Option<Vec<f32>>,
    selection: Option<SamplingSelectionV1>,
    /// Final-RMSNorm output rows used by the explicit embedding execution
    /// mode. This remains separate from generation logits and token output.
    embeddings_bf16: Option<Vec<u16>>,
    state: crate::Gemma4RequestStateSnapshot,
    audit: PreparedExecutionAudit,
}

/// Redacted accounting for the opaque full-attention state forks that make up
/// one immutable Gemma prefix owner.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Gemma4PrefixForkAuditV1 {
    kv_states: u32,
    sliding_layers: u32,
    sliding_planes: u32,
    shared_pages: u64,
    copied_bytes: u64,
    destination_owned_bytes: u64,
    cache_resident_bytes: u64,
}

impl Gemma4PrefixForkAuditV1 {
    pub const fn kv_states(self) -> u32 {
        self.kv_states
    }

    pub const fn shared_pages(self) -> u64 {
        self.shared_pages
    }

    /// Number of sliding-attention layers cloned into the immutable owner.
    pub const fn sliding_layers(self) -> u32 {
        self.sliding_layers
    }

    /// Number of sliding-attention K/V planes cloned into the immutable owner.
    pub const fn sliding_planes(self) -> u32 {
        self.sliding_planes
    }

    pub const fn copied_bytes(self) -> u64 {
        self.copied_bytes
    }

    pub const fn destination_owned_bytes(self) -> u64 {
        self.destination_owned_bytes
    }

    /// Resident bytes attributable to this immutable prefix owner. Shared
    /// VMM KV pages are charged from backend physical-memory metadata; the
    /// destination-owned count alone is zero for a read-only fork and is
    /// therefore not a sufficient cache quota signal.
    pub const fn cache_resident_bytes(self) -> u64 {
        self.cache_resident_bytes
    }

    fn add(
        &mut self,
        audit: StateForkAuditV1,
        cache_resident_bytes: u64,
    ) -> Result<(), Gemma4ExecutionLayoutError> {
        self.kv_states = self
            .kv_states
            .checked_add(1)
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("Gemma KV fork count overflowed"))?;
        self.shared_pages = self
            .shared_pages
            .checked_add(audit.shared_pages())
            .ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("Gemma shared page count overflowed")
            })?;
        self.copied_bytes = self
            .copied_bytes
            .checked_add(audit.copied_bytes())
            .ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("Gemma fork copied-byte count overflowed")
            })?;
        self.destination_owned_bytes = self
            .destination_owned_bytes
            .checked_add(audit.destination_owned_bytes())
            .ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("Gemma fork owned-byte count overflowed")
            })?;
        self.cache_resident_bytes = self
            .cache_resident_bytes
            .checked_add(cache_resident_bytes)
            .ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("Gemma prefix resident-byte count overflowed")
            })?;
        Ok(())
    }

    fn add_sliding_plane(
        &mut self,
        is_first_plane_for_layer: bool,
        audit: crate::DeviceCopyAuditV1,
    ) -> Result<(), Gemma4ExecutionLayoutError> {
        self.sliding_planes = self.sliding_planes.checked_add(1).ok_or_else(|| {
            Gemma4ExecutionLayoutError::invalid("Gemma sliding KV plane count overflowed")
        })?;
        if is_first_plane_for_layer {
            self.sliding_layers = self.sliding_layers.checked_add(1).ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("Gemma sliding KV layer count overflowed")
            })?;
        }
        self.copied_bytes = self
            .copied_bytes
            .checked_add(audit.copied_bytes())
            .ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("Gemma sliding copied-byte count overflowed")
            })?;
        self.destination_owned_bytes = self
            .destination_owned_bytes
            .checked_add(audit.destination_owned_bytes())
            .ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("Gemma sliding owned-byte count overflowed")
            })?;
        self.cache_resident_bytes = self
            .cache_resident_bytes
            .checked_add(audit.destination_owned_bytes())
            .ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("Gemma sliding resident-byte count overflowed")
            })?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Gemma4SlidingLayerIdentity {
    heads: u32,
    head_dim: u32,
    capacity: u64,
    retention_window: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Gemma4PrefixIdentityV1 {
    model_fingerprint: String,
    plan_digest: [u8; 32],
    state_capacity: u64,
    kv_descriptors: BTreeMap<u32, KvStateDescriptor>,
    sliding_layers: BTreeMap<u32, Gemma4SlidingLayerIdentity>,
}

/// One encoding-native full-attention KV layer in a Gemma state image.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4KvStateImageV1 {
    descriptor: KvStateDescriptor,
    image: ExecutionStateImageV1,
}

impl Gemma4KvStateImageV1 {
    pub const fn descriptor(&self) -> KvStateDescriptor {
        self.descriptor
    }

    pub const fn image(&self) -> &ExecutionStateImageV1 {
        &self.image
    }
}

/// One exact BF16 sliding-attention K/V layer in a Gemma state image.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4SlidingStateImageV1 {
    heads: u32,
    head_dim: u32,
    capacity: u64,
    retention_window: u64,
    image: ExecutionStateImageV1,
}

impl Gemma4SlidingStateImageV1 {
    pub const fn heads(&self) -> u32 {
        self.heads
    }

    pub const fn head_dim(&self) -> u32 {
        self.head_dim
    }

    pub const fn capacity(&self) -> u64 {
        self.capacity
    }

    pub const fn retention_window(&self) -> u64 {
        self.retention_window
    }

    pub const fn image(&self) -> &ExecutionStateImageV1 {
        &self.image
    }

    const fn identity(&self) -> Gemma4SlidingLayerIdentity {
        Gemma4SlidingLayerIdentity {
            heads: self.heads,
            head_dim: self.head_dim,
            capacity: self.capacity,
            retention_window: self.retention_window,
        }
    }
}

/// Complete backend-neutral Gemma request state. Full-attention layers retain
/// their exact KV encoding planes while sliding-attention layers retain exact
/// BF16 K/V bytes. Terminal output is optional and is deliberately removed
/// when flattened into a persistent checkpoint.
#[derive(Clone, PartialEq)]
pub struct Gemma4StateImageV1 {
    session_id: crate::ExecutionSessionId,
    model_fingerprint: String,
    plan_digest: [u8; 32],
    state_capacity: u64,
    committed_length: u64,
    rope_position_delta: i64,
    full_kv_layers: BTreeMap<u32, Gemma4KvStateImageV1>,
    sliding_layers: BTreeMap<u32, Gemma4SlidingStateImageV1>,
    cached_terminal_output: Option<Gemma4ExecutionOutput>,
}

impl fmt::Debug for Gemma4StateImageV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Gemma4StateImageV1")
            .field("session_id", &self.session_id)
            .field("model_fingerprint", &"<redacted>")
            .field("state_capacity", &self.state_capacity)
            .field("committed_length", &self.committed_length)
            .field("full_attention_layers", &self.full_kv_layers.len())
            .field("sliding_attention_layers", &self.sliding_layers.len())
            .field(
                "has_cached_terminal_output",
                &self.cached_terminal_output.is_some(),
            )
            .finish()
    }
}

impl Gemma4StateImageV1 {
    pub const fn session_id(&self) -> crate::ExecutionSessionId {
        self.session_id
    }

    pub fn model_fingerprint(&self) -> &str {
        &self.model_fingerprint
    }

    pub const fn plan_digest(&self) -> &[u8; 32] {
        &self.plan_digest
    }

    pub const fn state_capacity(&self) -> u64 {
        self.state_capacity
    }

    pub const fn committed_length(&self) -> u64 {
        self.committed_length
    }

    pub const fn rope_position_delta(&self) -> i64 {
        self.rope_position_delta
    }

    pub fn full_kv_layers(&self) -> &BTreeMap<u32, Gemma4KvStateImageV1> {
        &self.full_kv_layers
    }

    pub fn sliding_layers(&self) -> &BTreeMap<u32, Gemma4SlidingStateImageV1> {
        &self.sliding_layers
    }

    pub fn cached_terminal_output(&self) -> Option<&Gemma4ExecutionOutput> {
        self.cached_terminal_output.as_ref()
    }

    pub fn without_terminal_output(mut self) -> Self {
        self.cached_terminal_output = None;
        self
    }

    pub fn kv_encoding(&self) -> Result<KvCacheEncoding, Gemma4ExecutionLayoutError> {
        gemma_image_kv_encoding(&self.full_kv_layers)
    }

    pub fn kv_descriptor_digest(&self) -> Result<[u8; 32], Gemma4ExecutionLayoutError> {
        gemma_checkpoint_descriptor_digest(&self.full_kv_layers, &self.sliding_layers)
    }

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
    ) -> Result<SessionCheckpoint, Gemma4ExecutionLayoutError> {
        validate_gemma_state_image(self, false)?;
        if token_history.len() as u64 != self.committed_length
            || logical_position != self.committed_length
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "checkpoint token history or logical position differs from Gemma state",
            ));
        }
        let rope_delta = absolute_position
            .checked_sub(logical_position)
            .and_then(|value| i64::try_from(value).ok())
            .ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid(
                    "checkpoint absolute/logical position delta is invalid",
                )
            })?;
        if rope_delta != self.rope_position_delta
            || identity.model_lock_fingerprint != self.model_fingerprint
            || identity.plan_digest != gemma_hex_digest(&self.plan_digest)
            || identity.kv_encoding != self.kv_encoding()?
            || identity.kv_descriptor_digest != self.kv_descriptor_digest()?
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "checkpoint identity or position differs from Gemma state",
            ));
        }
        let mut state_layers = Vec::with_capacity(
            self.full_kv_layers
                .len()
                .saturating_add(self.sliding_layers.len()),
        );
        let mut state_planes = Vec::new();
        for entry in self.full_kv_layers.values() {
            state_layers.push(entry.image.metadata().clone());
            state_planes.extend(entry.image.planes().iter().cloned());
        }
        for entry in self.sliding_layers.values() {
            state_layers.push(entry.image.metadata().clone());
            state_planes.extend(entry.image.planes().iter().cloned());
        }
        SessionCheckpoint::new(
            identity,
            absolute_position,
            logical_position,
            generation_state_version,
            CheckpointPayload {
                token_history: token_history.to_vec(),
                conversation: conversation.to_vec(),
                state_layers,
                state_planes,
                sampler_state: sampler_state.to_vec(),
                grammar_state: grammar_state.to_vec(),
                stop_state: stop_state.to_vec(),
            },
        )
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))
    }
}

/// Immutable, request-independent Gemma prefix owner. It contains only
/// quiescent full-attention KV forks, exact device-owned sliding K/V ranges,
/// and terminal metadata; request workspace, queue, prepared operations, and
/// in-flight completions are not retained.
pub struct Gemma4PrefixStateV1 {
    inner: Arc<Gemma4PrefixStateInner>,
}

struct Gemma4PrefixStateInner {
    session: Arc<ExecutionSession>,
    identity: Gemma4PrefixIdentityV1,
    committed_length: u64,
    rope_position_delta: i64,
    kv_states: BTreeMap<u32, KvState>,
    sliding_buffers: BTreeMap<(u32, Gemma4KvPlane), ExecutionBuffer>,
    sliding_bytes: BTreeMap<(u32, Gemma4KvPlane), u64>,
    cached_terminal_output: Gemma4ExecutionOutput,
    fork_audit: Gemma4PrefixForkAuditV1,
}

impl fmt::Debug for Gemma4PrefixStateV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Gemma4PrefixStateV1")
            .field("session", &self.inner.session.id())
            .field("committed_length", &self.inner.committed_length)
            .field("state_capacity", &self.inner.identity.state_capacity)
            .field("fork_audit", &self.inner.fork_audit)
            .finish_non_exhaustive()
    }
}

impl Clone for Gemma4PrefixStateV1 {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Gemma4PrefixStateV1 {
    pub fn committed_length(&self) -> u64 {
        self.inner.committed_length
    }

    pub fn state_capacity(&self) -> u64 {
        self.inner.identity.state_capacity
    }

    /// Absolute RoPE position minus compact logical position for the next
    /// token in this prefix owner.
    pub fn rope_position_delta(&self) -> i64 {
        self.inner.rope_position_delta
    }

    pub fn fork_audit(&self) -> Gemma4PrefixForkAuditV1 {
        self.inner.fork_audit
    }

    pub fn model_fingerprint(&self) -> &str {
        &self.inner.identity.model_fingerprint
    }

    pub fn plan_digest(&self) -> &[u8; 32] {
        &self.inner.identity.plan_digest
    }

    pub fn cached_terminal_output(&self) -> &Gemma4ExecutionOutput {
        &self.inner.cached_terminal_output
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gemma4ExecutionOptions {
    pub binding_generation: u64,
    pub completion_timeout: Duration,
    pub expected_backend: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4ExecutionAudit {
    target: String,
    submission_count: u64,
    kernel_dispatch_count: u64,
    segment_count: u64,
    boundary_count: u64,
    fallback_used: bool,
}

impl Gemma4ExecutionAudit {
    pub fn target(&self) -> &str {
        &self.target
    }

    pub const fn submission_count(&self) -> u64 {
        self.submission_count
    }

    pub const fn kernel_dispatch_count(&self) -> u64 {
        self.kernel_dispatch_count
    }

    pub const fn segment_count(&self) -> u64 {
        self.segment_count
    }

    pub const fn boundary_count(&self) -> u64 {
        self.boundary_count
    }

    pub const fn fallback_used(&self) -> bool {
        self.fallback_used
    }
}

/// One exact-HIP Gemma request owner used by the shared generation service.
/// The model is uploaded once for this owner; every decode transition reuses
/// the same weight, constant, and K/V allocations.
pub struct Gemma4ExecutionRequest {
    _resident: Arc<Gemma4ResidentInner>,
    session: Arc<ExecutionSession>,
    queue: ExecutionQueue,
    lock: crate::Gemma4ModelLock,
    plan: WeightLoadPlan,
    layout: Gemma4ExecutionLayout,
    buffers: Gemma4ProvisionedBuffers,
    state: crate::Gemma4RequestState,
    completion_timeout: Duration,
    committed_length: u64,
    rope_position_delta: i64,
    rotary_position_mode: crate::RotaryPositionModeV1,
    binding_generation: u64,
    audit: Option<Gemma4ExecutionAudit>,
    opaque_kv_states: Option<BTreeMap<u32, crate::KvState>>,
    last_output: Option<Gemma4ExecutionOutput>,
}

impl fmt::Debug for Gemma4ResidentModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Gemma4ResidentModel")
            .field("session", &self.inner.session.id())
            .field("model_fingerprint", &self.inner.lock.fingerprint())
            .field("plan_digest", &self.inner.plan.digest_hex())
            .finish_non_exhaustive()
    }
}

fn gemma_full_kv_state_descriptor(
    descriptor: &crate::Gemma4KvDescriptor,
    quantized: Option<&dyn GemmaQuantizedSource>,
) -> Result<KvStateDescriptor, Gemma4ExecutionLayoutError> {
    if descriptor.retention_window.is_some() {
        return Err(Gemma4ExecutionLayoutError::invalid(
            "sliding-attention layer cannot use opaque full-attention KV state",
        ));
    }
    match quantized {
        Some(artifact) => {
            let scales = artifact.kv_scale(descriptor.layer).ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("quantized Gemma static KV scale is absent")
            })?;
            KvStateDescriptor::new_with_static_fp8(
                descriptor.layer,
                descriptor.capacity,
                descriptor.heads as usize,
                descriptor.head_dim as usize,
                scales.key_decode_scale(),
                scales.value_decode_scale(),
            )
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))
        }
        None => KvStateDescriptor::new_with_storage(
            descriptor.layer,
            descriptor.capacity,
            descriptor.heads as usize,
            descriptor.head_dim as usize,
            KvCacheEncoding::Fp16,
        )
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string())),
    }
}

fn gemma_full_attention_layers(graph: &Gemma4Graph) -> BTreeSet<u32> {
    graph
        .kv_descriptors()
        .iter()
        .filter(|descriptor| descriptor.retention_window.is_none())
        .map(|descriptor| descriptor.layer)
        .collect()
}

fn gemma_sliding_layer_identities(
    graph: &Gemma4Graph,
) -> BTreeMap<u32, Gemma4SlidingLayerIdentity> {
    graph
        .kv_descriptors()
        .iter()
        .filter_map(|descriptor| {
            descriptor.retention_window.map(|retention_window| {
                (
                    descriptor.layer,
                    Gemma4SlidingLayerIdentity {
                        heads: descriptor.heads,
                        head_dim: descriptor.head_dim,
                        capacity: descriptor.capacity,
                        retention_window,
                    },
                )
            })
        })
        .collect()
}

fn gemma_sliding_plane_bytes(
    identity: Gemma4SlidingLayerIdentity,
    committed_length: u64,
) -> Result<u64, Gemma4ExecutionLayoutError> {
    committed_length
        .checked_mul(u64::from(identity.heads))
        .and_then(|bytes| bytes.checked_mul(u64::from(identity.head_dim)))
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or_else(|| {
            Gemma4ExecutionLayoutError::invalid("Gemma sliding KV byte range overflowed")
        })
}

fn gemma_request_kv_tensor(
    layout: &Gemma4ExecutionLayout,
    layer: u32,
    plane: Gemma4KvPlane,
) -> Result<&Gemma4ExecutionTensor, Gemma4ExecutionLayoutError> {
    layout
        .tensors
        .iter()
        .find(|tensor| tensor.backing == Gemma4TensorBacking::RequestKv { layer, plane })
        .ok_or_else(|| {
            Gemma4ExecutionLayoutError::invalid(format!(
                "Gemma sliding request K/V tensor is absent for layer {layer} plane {plane:?}"
            ))
        })
}

fn validate_gemma_sliding_tensor(
    tensor: &Gemma4ExecutionTensor,
    identity: Gemma4SlidingLayerIdentity,
    committed_length: u64,
    expected_bytes: u64,
) -> Result<(), Gemma4ExecutionLayoutError> {
    let shape = tensor.view.shape();
    let capacity = usize::try_from(identity.capacity).map_err(|_| {
        Gemma4ExecutionLayoutError::invalid("Gemma sliding capacity does not fit usize")
    })?;
    let committed = usize::try_from(committed_length).map_err(|_| {
        Gemma4ExecutionLayoutError::invalid("Gemma prefix length does not fit usize")
    })?;
    if tensor.view.dtype() != DType::Bf16
        || tensor.view.encoding() != Encoding::Unquantized
        || !tensor.view.is_contiguous()
        || shape.len() != 3
        || shape[0] != capacity
        || shape[1] != identity.heads as usize
        || shape[2] != identity.head_dim as usize
        || committed > shape[0]
        || expected_bytes > tensor.view.payload_bytes()
    {
        return Err(Gemma4ExecutionLayoutError::invalid(format!(
            "Gemma sliding K/V tensor layout differs for layer with {} heads",
            identity.heads
        )));
    }
    Ok(())
}

fn gemma_hex_digest(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(71);
    output.push_str("sha256:");
    for byte in bytes {
        use std::fmt::Write as _;
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn gemma_kv_plane_kinds(encoding: KvCacheEncoding) -> &'static [StatePlaneKindV1] {
    use StatePlaneKindV1::*;
    match encoding {
        KvCacheEncoding::Fp16 | KvCacheEncoding::Fp8E4M3FnStatic => &[KvKey, KvValue],
        KvCacheEncoding::Fp8E4M3Fn => &[KvKey, KvValue, KvKeyScale, KvValueScale],
        KvCacheEncoding::Nvfp4 => &[
            KvKey,
            KvValue,
            KvKeyScale,
            KvValueScale,
            KvKeyOuterScale,
            KvValueOuterScale,
        ],
    }
}

fn gemma_image_kv_encoding(
    full_layers: &BTreeMap<u32, Gemma4KvStateImageV1>,
) -> Result<KvCacheEncoding, Gemma4ExecutionLayoutError> {
    let encoding = full_layers
        .values()
        .next()
        .map(|entry| entry.descriptor.cache_encoding())
        .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("Gemma state image has no full KV"))?;
    if full_layers
        .values()
        .any(|entry| entry.descriptor.cache_encoding() != encoding)
    {
        return Err(Gemma4ExecutionLayoutError::invalid(
            "Gemma full-attention KV encodings are not uniform",
        ));
    }
    Ok(encoding)
}

fn gemma_checkpoint_descriptor_digest(
    full_layers: &BTreeMap<u32, Gemma4KvStateImageV1>,
    sliding_layers: &BTreeMap<u32, Gemma4SlidingStateImageV1>,
) -> Result<[u8; 32], Gemma4ExecutionLayoutError> {
    let encoding = gemma_image_kv_encoding(full_layers)?;
    let full_descriptors = full_layers
        .iter()
        .map(|(&layer, entry)| (layer, entry.descriptor))
        .collect::<BTreeMap<_, _>>();
    let sliding_identities = sliding_layers
        .iter()
        .map(|(&layer, entry)| (layer, entry.identity()))
        .collect::<BTreeMap<_, _>>();
    Ok(gemma_checkpoint_descriptor_digest_from_parts(
        encoding,
        &full_descriptors,
        &sliding_identities,
    ))
}

fn gemma_checkpoint_descriptor_digest_from_parts(
    encoding: KvCacheEncoding,
    full_layers: &BTreeMap<u32, KvStateDescriptor>,
    sliding_layers: &BTreeMap<u32, Gemma4SlidingLayerIdentity>,
) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"sllm-gemma4-checkpoint-descriptors-v1");
    digest.update([match encoding {
        KvCacheEncoding::Fp16 => 1,
        KvCacheEncoding::Fp8E4M3Fn => 2,
        KvCacheEncoding::Fp8E4M3FnStatic => 3,
        KvCacheEncoding::Nvfp4 => 4,
    }]);
    digest.update((full_layers.len() as u64).to_le_bytes());
    for (&layer, &descriptor) in full_layers {
        digest.update([1]);
        digest.update(layer.to_le_bytes());
        digest.update(descriptor.layer_id().to_le_bytes());
        digest.update(descriptor.capacity().to_le_bytes());
        digest.update((descriptor.layout().heads() as u64).to_le_bytes());
        digest.update((descriptor.layout().head_dim() as u64).to_le_bytes());
        match descriptor.static_fp8_scales() {
            Some((key, value)) => {
                digest.update([1]);
                digest.update(key.to_bits().to_le_bytes());
                digest.update(value.to_bits().to_le_bytes());
            }
            None => digest.update([0]),
        }
    }
    digest.update((sliding_layers.len() as u64).to_le_bytes());
    for (&layer, &entry) in sliding_layers {
        digest.update([2]);
        digest.update(layer.to_le_bytes());
        digest.update(entry.heads.to_le_bytes());
        digest.update(entry.head_dim.to_le_bytes());
        digest.update(entry.capacity.to_le_bytes());
        digest.update(entry.retention_window.to_le_bytes());
        digest.update(b"bf16-unquantized-key-value");
    }
    digest.finalize().into()
}

fn validate_gemma_layer_image(
    image: &ExecutionStateImageV1,
    layer: u32,
    committed_length: u64,
    expected_planes: &[StatePlaneKindV1],
    allow_empty_supplemental_planes: bool,
) -> Result<(), Gemma4ExecutionLayoutError> {
    let metadata = image.metadata();
    if metadata.owner != StateOwnerKindV1::Kv
        || metadata.layer_id != layer
        || metadata.published_length != committed_length
        || metadata.active_slot.is_some()
    {
        return Err(Gemma4ExecutionLayoutError::invalid(
            "Gemma state image layer metadata differs",
        ));
    }
    let mut actual = BTreeSet::new();
    for plane in image.planes() {
        if plane.owner != StateOwnerKindV1::Kv
            || plane.layer_id != layer
            || !actual.insert(plane.plane)
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma state image contains a duplicate or mismatched plane",
            ));
        }
        if allow_empty_supplemental_planes
            && !matches!(
                plane.plane,
                StatePlaneKindV1::KvKey | StatePlaneKindV1::KvValue
            )
            && !plane.bytes.is_empty()
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma sliding supplemental plane must be empty",
            ));
        }
    }
    if actual != expected_planes.iter().copied().collect() {
        return Err(Gemma4ExecutionLayoutError::invalid(
            "Gemma state image plane topology differs",
        ));
    }
    Ok(())
}

fn validate_gemma_state_image(
    image: &Gemma4StateImageV1,
    require_terminal_output: bool,
) -> Result<(), Gemma4ExecutionLayoutError> {
    if image.committed_length == 0 || image.committed_length > image.state_capacity {
        return Err(Gemma4ExecutionLayoutError::invalid(
            "Gemma state image length or capacity is invalid",
        ));
    }
    let encoding = image.kv_encoding()?;
    let expected_full_planes = gemma_kv_plane_kinds(encoding);
    let full_keys = image
        .full_kv_layers
        .keys()
        .copied()
        .collect::<BTreeSet<_>>();
    let sliding_keys = image
        .sliding_layers
        .keys()
        .copied()
        .collect::<BTreeSet<_>>();
    let (expected_full_keys, expected_sliding_keys) = crate::reviewed_layer_schedule()
        .into_iter()
        .enumerate()
        .try_fold(
            (BTreeSet::new(), BTreeSet::new()),
            |(mut full, mut sliding), (layer, kind)| {
                let layer = u32::try_from(layer).map_err(|_| {
                    Gemma4ExecutionLayoutError::invalid("Gemma layer index does not fit u32")
                })?;
                match kind {
                    Gemma4LayerType::FullAttention => {
                        full.insert(layer);
                    }
                    Gemma4LayerType::SlidingAttention => {
                        sliding.insert(layer);
                    }
                }
                Ok((full, sliding))
            },
        )?;
    if full_keys != expected_full_keys
        || sliding_keys != expected_sliding_keys
        || !full_keys.is_disjoint(&sliding_keys)
    {
        return Err(Gemma4ExecutionLayoutError::invalid(
            "Gemma state image layer topology is invalid",
        ));
    }
    for (&layer, entry) in &image.full_kv_layers {
        if entry.descriptor.layer_id() != layer
            || entry.descriptor.capacity() != image.state_capacity
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma full-attention state descriptor differs",
            ));
        }
        validate_gemma_layer_image(
            &entry.image,
            layer,
            image.committed_length,
            expected_full_planes,
            false,
        )?;
    }
    let expected_sliding_planes = gemma_kv_plane_kinds(encoding);
    for (&layer, entry) in &image.sliding_layers {
        let identity = entry.identity();
        if identity.capacity != image.state_capacity
            || identity.heads == 0
            || identity.head_dim == 0
            || identity.retention_window == 0
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma sliding-attention state identity differs",
            ));
        }
        validate_gemma_layer_image(
            &entry.image,
            layer,
            image.committed_length,
            expected_sliding_planes,
            true,
        )?;
        let expected_bytes = gemma_sliding_plane_bytes(identity, image.committed_length)?;
        for kind in [StatePlaneKindV1::KvKey, StatePlaneKindV1::KvValue] {
            if entry
                .image
                .planes()
                .iter()
                .find(|plane| plane.plane == kind)
                .map(|plane| plane.bytes.len() as u64)
                != Some(expected_bytes)
            {
                return Err(Gemma4ExecutionLayoutError::invalid(
                    "Gemma sliding-attention state byte length differs",
                ));
            }
        }
    }
    if let Some(output) = image.cached_terminal_output.as_ref() {
        if output.state().committed_length != image.committed_length {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma cached terminal output length differs from state image",
            ));
        }
    }
    if require_terminal_output && image.cached_terminal_output.is_none() {
        return Err(Gemma4ExecutionLayoutError::invalid(
            "Gemma raw state image has no terminal output",
        ));
    }
    Ok(())
}

fn gemma_checkpoint_layer_image(
    checkpoint: &SessionCheckpoint,
    layer: u32,
) -> Result<ExecutionStateImageV1, Gemma4ExecutionLayoutError> {
    let metadata = checkpoint
        .payload
        .state_layers
        .iter()
        .find(|metadata| metadata.owner == StateOwnerKindV1::Kv && metadata.layer_id == layer)
        .cloned()
        .ok_or_else(|| {
            Gemma4ExecutionLayoutError::invalid(format!(
                "checkpoint Gemma layer {layer} metadata is absent"
            ))
        })?;
    let planes = checkpoint
        .payload
        .state_planes
        .iter()
        .filter(|plane| plane.owner == StateOwnerKindV1::Kv && plane.layer_id == layer)
        .cloned()
        .collect();
    Ok(ExecutionStateImageV1::new(metadata, planes))
}

impl Gemma4ResidentModel {
    /// Capabilities exposed to the backend-neutral context-window policy.
    /// The fresh explicit-position factory below is the sole publication
    /// boundary for a compacted Gemma owner.
    pub const fn context_adapter_capabilities() -> crate::ContextAdapterCapabilitiesV1 {
        crate::ContextAdapterCapabilitiesV1::new(1, 1, 1, true, true)
    }

    /// Uploads the immutable BF16 model and derived constants exactly once.
    pub fn new(
        session: Arc<ExecutionSession>,
        lock: crate::Gemma4ModelLock,
        plan: WeightLoadPlan,
        cache: &VerifiedCache,
        completion_timeout: Duration,
    ) -> Result<Self, Gemma4ExecutionLayoutError> {
        Self::new_with_nvfp4(session, lock, plan, cache, None, completion_timeout)
    }

    pub fn new_nvfp4(
        session: Arc<ExecutionSession>,
        lock: crate::Gemma4ModelLock,
        plan: WeightLoadPlan,
        cache: &VerifiedCache,
        sidecar: Arc<VerifiedNvfp4Sidecar>,
        completion_timeout: Duration,
    ) -> Result<Self, Gemma4ExecutionLayoutError> {
        Self::new_with_nvfp4(
            session,
            lock,
            plan,
            cache,
            Some(sidecar),
            completion_timeout,
        )
    }

    /// Uploads the provider artifact directly and binds its complete mixed
    /// recipe to the existing Gemma graph. No BF16 source cache or sidecar is
    /// involved.
    pub fn new_quantized(
        session: Arc<ExecutionSession>,
        lock: crate::Gemma4ModelLock,
        plan: WeightLoadPlan,
        artifact: Arc<VerifiedUnslothGemma4Nvfp4>,
        completion_timeout: Duration,
    ) -> Result<Self, Gemma4ExecutionLayoutError> {
        Self::new_quantized_source(session, lock, plan, artifact, completion_timeout)
    }

    pub fn new_gguf_quantized(
        session: Arc<ExecutionSession>,
        lock: crate::Gemma4ModelLock,
        plan: WeightLoadPlan,
        source: Arc<VerifiedGgufGemmaSource>,
        completion_timeout: Duration,
    ) -> Result<Self, Gemma4ExecutionLayoutError> {
        if source.lock_fingerprint() != lock.fingerprint() {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "GGUF Gemma source and lock identities differ",
            ));
        }
        Self::new_quantized_source(session, lock, plan, source, completion_timeout)
    }

    fn new_quantized_source(
        session: Arc<ExecutionSession>,
        lock: crate::Gemma4ModelLock,
        plan: WeightLoadPlan,
        source: Arc<dyn GemmaQuantizedSource>,
        completion_timeout: Duration,
    ) -> Result<Self, Gemma4ExecutionLayoutError> {
        if completion_timeout.is_zero()
            || lock.fingerprint() != plan.lock_fingerprint
            || plan.repo_id != source.repository()
            || plan.resolved_revision != source.resolved_revision()
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "quantized Gemma resident identity or completion timeout differs",
            ));
        }
        let graph = crate::build_gemma4_graph(&lock, &plan, 1, 0, 1)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let layout =
            build_gemma4_quantized_execution_layout_source(&graph, &plan, source.as_ref())?;
        let queue = session
            .create_queue()
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let buffers = provision_gemma4_execution_buffers(Arc::clone(&session), &layout)?;
        buffers.upload_immutable_quantized(
            &layout,
            &plan,
            source.as_ref(),
            &queue,
            completion_timeout,
        )?;
        let immutable = buffers.immutable_buffers(&layout)?;
        drop(buffers);
        Ok(Self {
            inner: Arc::new(Gemma4ResidentInner {
                session,
                queue,
                lock,
                plan,
                nvfp4_sidecar: None,
                quantized_model: Some(source),
                immutable,
                completion_timeout,
            }),
        })
    }

    fn new_with_nvfp4(
        session: Arc<ExecutionSession>,
        lock: crate::Gemma4ModelLock,
        plan: WeightLoadPlan,
        cache: &VerifiedCache,
        sidecar: Option<Arc<VerifiedNvfp4Sidecar>>,
        completion_timeout: Duration,
    ) -> Result<Self, Gemma4ExecutionLayoutError> {
        if completion_timeout.is_zero()
            || lock.fingerprint() != plan.lock_fingerprint
            || cache.lock_fingerprint != plan.lock_fingerprint
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma resident identity or completion timeout differs",
            ));
        }
        let graph = crate::build_gemma4_graph(&lock, &plan, 1, 0, 1)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let layout = match sidecar.as_deref() {
            Some(sidecar) => build_gemma4_nvfp4_execution_layout(&graph, &plan, sidecar)?,
            None => build_gemma4_execution_layout(&graph, &plan)?,
        };
        let queue = session
            .create_queue()
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let buffers = provision_gemma4_execution_buffers(Arc::clone(&session), &layout)?;
        match sidecar.as_deref() {
            Some(sidecar) => buffers.upload_immutable_nvfp4(
                &layout,
                &plan,
                cache,
                sidecar,
                &queue,
                completion_timeout,
            )?,
            None => buffers.upload_immutable(&layout, &plan, cache, &queue, completion_timeout)?,
        }
        let immutable = buffers.immutable_buffers(&layout)?;
        drop(buffers);
        Ok(Self {
            inner: Arc::new(Gemma4ResidentInner {
                session,
                queue,
                lock,
                plan,
                nvfp4_sidecar: sidecar,
                quantized_model: None,
                immutable,
                completion_timeout,
            }),
        })
    }

    /// Creates a fresh request-local workspace and KV owner while sharing the
    /// immutable model allocations and ordered execution queue.
    pub fn new_request(
        &self,
        prefill_token_count: u64,
        state_capacity: u64,
    ) -> Result<Gemma4ExecutionRequest, Gemma4ExecutionLayoutError> {
        self.new_request_with_position_mode(
            prefill_token_count,
            state_capacity,
            crate::RotaryPositionModeV1::Contiguous,
        )
    }

    fn new_request_with_position_mode(
        &self,
        prefill_token_count: u64,
        state_capacity: u64,
        position_mode: crate::RotaryPositionModeV1,
    ) -> Result<Gemma4ExecutionRequest, Gemma4ExecutionLayoutError> {
        if prefill_token_count == 0 || state_capacity < prefill_token_count {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma request token count or capacity is invalid",
            ));
        }
        let graph = crate::build_gemma4_graph_with_position_mode(
            &self.inner.lock,
            &self.inner.plan,
            prefill_token_count,
            0,
            state_capacity,
            position_mode,
        )
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let layout = if let Some(artifact) = self.inner.quantized_model.as_deref() {
            build_gemma4_quantized_execution_layout_source(&graph, &self.inner.plan, artifact)?
        } else {
            match self.inner.nvfp4_sidecar.as_deref() {
                Some(sidecar) => {
                    build_gemma4_nvfp4_execution_layout(&graph, &self.inner.plan, sidecar)?
                }
                None => build_gemma4_execution_layout(&graph, &self.inner.plan)?,
            }
        };
        // A context shift temporarily owns the old request while the fresh
        // retained owner is materialized. Reject an obviously impossible
        // second-owner allocation before touching buffers; opaque KV state
        // creation below remains transactional if the backend reports no
        // placement telemetry.
        if let Some(available) = self
            .inner
            .session
            .available_memory_bytes()
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?
        {
            let required = layout
                .workspace_bytes()
                .checked_add(layout.request_state_bytes())
                .ok_or_else(|| {
                    Gemma4ExecutionLayoutError::invalid(
                        "Gemma request placement byte count overflowed",
                    )
                })?;
            if required > available {
                return Err(Gemma4ExecutionLayoutError::invalid(format!(
                    "Gemma request placement requires {required} bytes but only {available} bytes are available",
                )));
            }
        }
        let buffers = provision_gemma4_request_buffers(
            Arc::clone(&self.inner.session),
            &layout,
            &self.inner.immutable,
        )?;
        let opaque_kv_states = graph
            .kv_descriptors()
            .iter()
            .filter(|descriptor| descriptor.retention_window.is_none())
            .map(|descriptor| {
                let state_descriptor = gemma_full_kv_state_descriptor(
                    descriptor,
                    self.inner.quantized_model.as_deref(),
                )?;
                let state = self
                    .inner
                    .session
                    .create_kv_state(state_descriptor)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
                Ok((descriptor.layer, state))
            })
            .collect::<Result<BTreeMap<_, _>, Gemma4ExecutionLayoutError>>()?;
        if opaque_kv_states.keys().copied().collect::<BTreeSet<_>>()
            != gemma_full_attention_layers(&graph)
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma opaque KV layer set differs from full-attention graph layers",
            ));
        }
        Ok(Gemma4ExecutionRequest {
            _resident: Arc::clone(&self.inner),
            session: Arc::clone(&self.inner.session),
            queue: self.inner.queue.clone(),
            lock: self.inner.lock.clone(),
            plan: self.inner.plan.clone(),
            layout,
            buffers,
            state: crate::Gemma4RequestState::new(state_capacity)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
            completion_timeout: self.inner.completion_timeout,
            committed_length: 0,
            rope_position_delta: 0,
            rotary_position_mode: position_mode,
            binding_generation: 0,
            audit: None,
            opaque_kv_states: Some(opaque_kv_states),
            last_output: None,
        })
    }

    pub fn new_request_for_session(
        &self,
        session: Arc<ExecutionSession>,
        prefill_token_count: u64,
        state_capacity: u64,
    ) -> Result<Gemma4ExecutionRequest, Gemma4ExecutionLayoutError> {
        if session.id() != self.inner.session.id() {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma request session differs from resident model session",
            ));
        }
        self.new_request(prefill_token_count, state_capacity)
    }

    /// Builds a fresh request by transactionally materializing the retained
    /// prefix/recent token ranges into new device state. The source history is
    /// only read and remains unchanged; publication is visible only after the
    /// fresh prefill completes successfully.
    pub fn new_request_from_context_shift(
        &self,
        decision: ContextShiftDecisionV1,
        state: ContextWindowStateV1,
        token_history: &[i32],
        state_capacity: u64,
    ) -> Result<(Gemma4ExecutionRequest, Gemma4ExecutionOutput), Gemma4ExecutionLayoutError> {
        if !decision.requires_shift() || decision.old_state() != state {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma context shift decision is stale or does not require a shift",
            ));
        }
        let retained = decision
            .retained_token_ids(token_history)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let positions = decision
            .retained_absolute_positions(state.logical_length())
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let retained_len = u64::try_from(retained.len())
            .map_err(|_| Gemma4ExecutionLayoutError::invalid("retained length overflowed"))?;
        if retained_len != decision.proposed_state().logical_length()
            || retained_len == 0
            || retained_len > state_capacity
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma retained state length differs from the shift decision",
            ));
        }
        let mut request = self.new_request_with_position_mode(
            retained_len,
            state_capacity,
            crate::RotaryPositionModeV1::Explicit,
        )?;
        let output = request.prefill_with_absolute_positions(&retained, &positions)?;
        let delta = state
            .absolute_position()
            .checked_sub(retained_len)
            .and_then(|value| i64::try_from(value).ok())
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("Gemma RoPE delta overflowed"))?;
        request.set_rope_position_delta(delta)?;
        Ok((request, output))
    }

    /// Creates a fresh request workspace and transactionally forks every
    /// published full-attention state from an immutable prefix owner.
    pub fn new_request_from_prefix(
        &self,
        prefix: &Gemma4PrefixStateV1,
        suffix_token_count: u64,
        state_capacity: u64,
    ) -> Result<Gemma4ExecutionRequest, Gemma4ExecutionLayoutError> {
        if suffix_token_count == 0 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "prefix request token count must be non-zero",
            ));
        }
        let mut request = self.new_request(suffix_token_count, state_capacity)?;
        request.install_prefix(prefix)?;
        Ok(request)
    }

    pub fn request_from_prefix(
        &self,
        prefix: &Gemma4PrefixStateV1,
        suffix_token_count: u64,
        state_capacity: u64,
    ) -> Result<Gemma4ExecutionRequest, Gemma4ExecutionLayoutError> {
        self.new_request_from_prefix(prefix, suffix_token_count, state_capacity)
    }

    /// Creates a fresh request and transactionally imports every full- and
    /// sliding-attention layer from a same-session state image. A non-empty
    /// suffix shape is required so the restored owner can only be consumed by
    /// a continuation transition.
    pub fn new_request_from_state_image(
        &self,
        image: &Gemma4StateImageV1,
        suffix_token_count: u64,
        state_capacity: u64,
    ) -> Result<Gemma4ExecutionRequest, Gemma4ExecutionLayoutError> {
        if suffix_token_count == 0 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "state-image restore requires a non-empty suffix",
            ));
        }
        let mut request = self.new_request(suffix_token_count, state_capacity)?;
        request.restore_state_image(image)?;
        Ok(request)
    }

    pub fn restore_request_from_state_image(
        &self,
        image: &Gemma4StateImageV1,
        suffix_token_count: u64,
        state_capacity: u64,
    ) -> Result<Gemma4ExecutionRequest, Gemma4ExecutionLayoutError> {
        self.new_request_from_state_image(image, suffix_token_count, state_capacity)
    }

    /// Restores an authenticated backend-neutral checkpoint into a fresh
    /// request owner. Unlike a raw state image, this path intentionally
    /// permits a different execution-session owner; native handles are never
    /// carried across the boundary. Terminal output is not checkpointed, so a
    /// non-empty suffix is mandatory.
    pub fn new_request_from_checkpoint(
        &self,
        checkpoint: &SessionCheckpoint,
        expected_identity: &CheckpointIdentity,
        suffix_token_count: u64,
        state_capacity: u64,
    ) -> Result<Gemma4ExecutionRequest, Gemma4ExecutionLayoutError> {
        if suffix_token_count == 0 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "checkpoint restore requires a non-empty suffix",
            ));
        }
        let mut request = self.new_request(suffix_token_count, state_capacity)?;
        request.restore_checkpoint(checkpoint, expected_identity)?;
        Ok(request)
    }

    pub fn restore_request_from_checkpoint(
        &self,
        checkpoint: &SessionCheckpoint,
        expected_identity: &CheckpointIdentity,
        suffix_token_count: u64,
        state_capacity: u64,
    ) -> Result<Gemma4ExecutionRequest, Gemma4ExecutionLayoutError> {
        self.new_request_from_checkpoint(
            checkpoint,
            expected_identity,
            suffix_token_count,
            state_capacity,
        )
    }

    /// Runs an exact decode continuation from an immutable prefix. An empty
    /// suffix is a pure cached-terminal-output lookup and performs no work.
    pub fn generate_from_prefix(
        &self,
        prefix: &Gemma4PrefixStateV1,
        suffix: &[i32],
        state_capacity: u64,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        if suffix.is_empty() {
            return Ok(prefix.cached_terminal_output().clone());
        }
        let mut request = self.new_request_from_prefix(prefix, 1, state_capacity)?;
        request.decode_continuation(suffix)
    }

    pub fn session_id(&self) -> crate::ExecutionSessionId {
        self.inner.session.id()
    }

    pub fn model_fingerprint(&self) -> &str {
        self.inner.lock.fingerprint()
    }

    pub fn plan_digest(&self) -> &[u8; 32] {
        self.inner.plan.digest()
    }

    pub fn memory_snapshot(&self) -> crate::AllocationSnapshot {
        self.inner.session.memory_snapshot()
    }
}

impl Gemma4ExecutionRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session: Arc<ExecutionSession>,
        lock: crate::Gemma4ModelLock,
        plan: WeightLoadPlan,
        cache: &VerifiedCache,
        prefill_token_count: u64,
        state_capacity: u64,
        completion_timeout: Duration,
    ) -> Result<Self, Gemma4ExecutionLayoutError> {
        let resident = Gemma4ResidentModel::new(session, lock, plan, cache, completion_timeout)?;
        resident.new_request(prefill_token_count, state_capacity)
    }

    pub fn prefill(
        &mut self,
        token_ids: &[i32],
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.prefill_impl(token_ids, false, None)
    }

    /// Prefills a fresh explicit-position graph with compact logical rows and
    /// caller-supplied absolute RoPE positions.
    pub fn prefill_with_absolute_positions(
        &mut self,
        token_ids: &[i32],
        positions: &[u64],
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        if self.rotary_position_mode != crate::RotaryPositionModeV1::Explicit {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "absolute positions require an explicit rotary request",
            ));
        }
        let positions = positions
            .iter()
            .map(|position| {
                i32::try_from(*position)
                    .map_err(|_| Gemma4ExecutionLayoutError::invalid("position does not fit i32"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.prefill_impl_with_positions(token_ids, false, None, Some(&positions), false)
    }

    pub fn prefill_with_last_logits(
        &mut self,
        token_ids: &[i32],
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.prefill_impl(token_ids, true, None)
    }

    /// Runs the verified Gemma prefill route in explicit embedding mode. The
    /// final normalized hidden rows are read back in BF16; no generation token
    /// or full-logit row is read back as the embedding result.
    pub fn prefill_with_embeddings(
        &mut self,
        token_ids: &[i32],
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.prefill_impl_with_positions(token_ids, false, None, None, true)
    }

    /// Runs prefill with the bounded device token-selector subset.  The
    /// terminal Argmax is replaced by TokenSelect on the same queue and only
    /// its fixed selected-record is read back.
    pub fn prefill_with_device_selector(
        &mut self,
        token_ids: &[i32],
        selector: &DeviceTokenSelectorRequestV1,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.prefill_impl(token_ids, false, Some(selector))
    }

    fn prefill_impl(
        &mut self,
        token_ids: &[i32],
        include_last_logits: bool,
        selector: Option<&DeviceTokenSelectorRequestV1>,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.prefill_impl_with_positions(token_ids, include_last_logits, selector, None, false)
    }

    fn prefill_impl_with_positions(
        &mut self,
        token_ids: &[i32],
        include_last_logits: bool,
        selector: Option<&DeviceTokenSelectorRequestV1>,
        positions: Option<&[i32]>,
        include_embeddings: bool,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        if self.committed_length != 0
            || token_ids.len() as u64 != layout_token_count(&self.layout)?
            || positions.is_some_and(|positions| positions.len() != token_ids.len())
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma prefill length or lifecycle differs",
            ));
        }
        if let Some(positions) = positions {
            self.buffers.upload_transition_inputs_with_positions(
                &self.layout,
                &self.queue,
                token_ids,
                positions,
                self.completion_timeout,
            )?;
        } else {
            self.buffers.upload_transition_inputs(
                &self.layout,
                &self.queue,
                token_ids,
                self.completion_timeout,
            )?;
        }
        let state_capacity = self.state_capacity()?;
        let graph = crate::build_gemma4_graph_with_position_mode(
            &self.lock,
            &self.plan,
            token_ids.len() as u64,
            0,
            state_capacity,
            self.rotary_position_mode,
        )
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        self.run_graph(graph, include_last_logits, selector, include_embeddings)
    }

    pub fn decode(
        &mut self,
        token_id: i32,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.decode_impl(token_id, false, None)
    }

    pub fn decode_with_last_logits(
        &mut self,
        token_id: i32,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.decode_impl(token_id, true, None)
    }

    /// Runs decode with the bounded device token-selector subset.  No full
    /// logits row is copied to the host on this path.
    pub fn decode_with_device_selector(
        &mut self,
        token_id: i32,
        selector: &DeviceTokenSelectorRequestV1,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.decode_impl(token_id, false, Some(selector))
    }

    fn decode_impl(
        &mut self,
        token_id: i32,
        include_last_logits: bool,
        selector: Option<&DeviceTokenSelectorRequestV1>,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        if self.committed_length == 0 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma decode cannot precede prefill",
            ));
        }
        let position_mode = if self.rope_position_delta == 0 {
            crate::RotaryPositionModeV1::Contiguous
        } else {
            crate::RotaryPositionModeV1::Explicit
        };
        let graph = crate::build_gemma4_graph_with_position_mode(
            &self.lock,
            &self.plan,
            1,
            self.committed_length,
            self.state_capacity()?,
            position_mode,
        )
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let layout = if let Some(artifact) = self._resident.quantized_model.as_deref() {
            build_gemma4_quantized_execution_layout_source(&graph, &self.plan, artifact)?
        } else {
            match self._resident.nvfp4_sidecar.as_deref() {
                Some(sidecar) => build_gemma4_nvfp4_execution_layout(&graph, &self.plan, sidecar)?,
                None => build_gemma4_execution_layout(&graph, &self.plan)?,
            }
        };
        let buffers = self.buffers.rebind_transition(&self.layout, &layout)?;
        if self.rope_position_delta == 0 {
            buffers.upload_transition_inputs(
                &layout,
                &self.queue,
                &[token_id],
                self.completion_timeout,
            )?;
        } else {
            let absolute = i64::try_from(self.committed_length)
                .ok()
                .and_then(|position| position.checked_add(self.rope_position_delta))
                .and_then(|position| i32::try_from(position).ok())
                .ok_or_else(|| {
                    Gemma4ExecutionLayoutError::invalid(
                        "Gemma absolute RoPE position does not fit I32",
                    )
                })?;
            buffers.upload_transition_inputs_with_positions(
                &layout,
                &self.queue,
                &[token_id],
                &[absolute],
                self.completion_timeout,
            )?;
        }
        self.layout = layout;
        self.buffers = buffers;
        self.run_graph(graph, include_last_logits, selector, false)
    }

    pub fn cancel(&self) {
        self.state.cancel();
    }

    /// Exports every quiescent full- and sliding-attention state layer into a
    /// backend-neutral, encoding-native image. No workspace, queue, prepared
    /// operation, or native handle is retained.
    pub fn state_image(&self) -> Result<Gemma4StateImageV1, Gemma4ExecutionLayoutError> {
        let state_snapshot = self
            .state
            .snapshot()
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        if state_snapshot.poisoned
            || self.committed_length == 0
            || state_snapshot.committed_length != self.committed_length
            || self.last_output.is_none()
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma state export requires a completed quiescent transition",
            ));
        }
        let state_capacity = self.state_capacity()?;
        let graph = crate::build_gemma4_graph(&self.lock, &self.plan, 1, 0, state_capacity)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let expected_full = gemma_full_attention_layers(&graph);
        let states = self.opaque_kv_states.as_ref().ok_or_else(|| {
            Gemma4ExecutionLayoutError::invalid("opaque full-attention KV states are absent")
        })?;
        if states.keys().copied().collect::<BTreeSet<_>>() != expected_full {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma export full-attention topology differs",
            ));
        }
        let mut full_kv_layers = BTreeMap::new();
        for (&layer, state) in states {
            let image = self
                .session
                .export_kv_state_image(state)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
            if image.metadata().published_length != self.committed_length {
                return Err(Gemma4ExecutionLayoutError::invalid(format!(
                    "Gemma exported full-attention layer {layer} length differs"
                )));
            }
            full_kv_layers.insert(
                layer,
                Gemma4KvStateImageV1 {
                    descriptor: state.descriptor(),
                    image,
                },
            );
        }
        let encoding = gemma_image_kv_encoding(&full_kv_layers)?;
        let sliding_identities = gemma_sliding_layer_identities(&graph);
        let mut sliding_layers = BTreeMap::new();
        for (&layer, &identity) in &sliding_identities {
            let byte_count = gemma_sliding_plane_bytes(identity, self.committed_length)?;
            let mut planes = Vec::with_capacity(gemma_kv_plane_kinds(encoding).len());
            for (plane, kind) in [
                (Gemma4KvPlane::Key, StatePlaneKindV1::KvKey),
                (Gemma4KvPlane::Value, StatePlaneKindV1::KvValue),
            ] {
                let tensor = gemma_request_kv_tensor(&self.layout, layer, plane)?;
                validate_gemma_sliding_tensor(tensor, identity, self.committed_length, byte_count)?;
                let range = self
                    .buffers
                    .buffer(tensor.id())?
                    .range(0, byte_count)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
                planes.push(OpaqueStatePlane {
                    owner: StateOwnerKindV1::Kv,
                    layer_id: layer,
                    plane: kind,
                    bytes: read_gemma_buffer_bytes(
                        &self.session,
                        &self.queue,
                        &range,
                        self.completion_timeout,
                        "Gemma sliding checkpoint export",
                    )?,
                });
            }
            for &kind in gemma_kv_plane_kinds(encoding) {
                if !matches!(kind, StatePlaneKindV1::KvKey | StatePlaneKindV1::KvValue) {
                    planes.push(OpaqueStatePlane {
                        owner: StateOwnerKindV1::Kv,
                        layer_id: layer,
                        plane: kind,
                        bytes: Vec::new(),
                    });
                }
            }
            sliding_layers.insert(
                layer,
                Gemma4SlidingStateImageV1 {
                    heads: identity.heads,
                    head_dim: identity.head_dim,
                    capacity: identity.capacity,
                    retention_window: identity.retention_window,
                    image: ExecutionStateImageV1::new(
                        StateLayerMetadataV1 {
                            owner: StateOwnerKindV1::Kv,
                            layer_id: layer,
                            published_length: self.committed_length,
                            generation: self.binding_generation,
                            active_slot: None,
                        },
                        planes,
                    ),
                },
            );
        }
        let image = Gemma4StateImageV1 {
            session_id: self.session.id(),
            model_fingerprint: self.lock.fingerprint().to_owned(),
            plan_digest: *self.plan.digest(),
            state_capacity,
            committed_length: self.committed_length,
            rope_position_delta: self.rope_position_delta,
            full_kv_layers,
            sliding_layers,
            cached_terminal_output: self.last_output.clone(),
        };
        validate_gemma_state_image(&image, true)?;
        Ok(image)
    }

    pub fn export_state_image(&self) -> Result<Gemma4StateImageV1, Gemma4ExecutionLayoutError> {
        self.state_image()
    }

    pub fn save_state_image(&self) -> Result<Gemma4StateImageV1, Gemma4ExecutionLayoutError> {
        self.state_image()
    }

    /// Captures this request as a persistent checkpoint. Terminal output is
    /// intentionally omitted, so restore requires a non-empty suffix.
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
    ) -> Result<SessionCheckpoint, Gemma4ExecutionLayoutError> {
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

    fn restore_state_image(
        &mut self,
        image: &Gemma4StateImageV1,
    ) -> Result<(), Gemma4ExecutionLayoutError> {
        let state_snapshot = self
            .state
            .snapshot()
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        if state_snapshot.poisoned || self.committed_length != 0 || self.last_output.is_some() {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma state restore requires a fresh, quiescent request",
            ));
        }
        if image.session_id != self.session.id() {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "raw Gemma state image belongs to a different execution session",
            ));
        }
        validate_gemma_state_image(image, true)?;
        self.restore_validated_state_image(image)
    }

    fn restore_checkpoint(
        &mut self,
        checkpoint: &SessionCheckpoint,
        expected_identity: &CheckpointIdentity,
    ) -> Result<(), Gemma4ExecutionLayoutError> {
        let state_snapshot = self
            .state
            .snapshot()
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        if state_snapshot.poisoned || self.committed_length != 0 || self.last_output.is_some() {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma checkpoint restore requires a fresh, quiescent request",
            ));
        }
        checkpoint
            .validate()
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        if checkpoint.header.identity != *expected_identity {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "checkpoint frontend or target identity differs from restore caller",
            ));
        }

        let state_capacity = self.state_capacity()?;
        let logical_position = checkpoint.header.logical_position;
        if logical_position == 0
            || logical_position != checkpoint.header.token_count
            || logical_position > state_capacity
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "checkpoint logical position differs from token history or request capacity",
            ));
        }
        let rope_position_delta = checkpoint
            .header
            .absolute_position
            .checked_sub(logical_position)
            .and_then(|delta| i64::try_from(delta).ok())
            .ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid(
                    "checkpoint absolute/logical position delta is invalid",
                )
            })?;

        let graph = crate::build_gemma4_graph(&self.lock, &self.plan, 1, 0, state_capacity)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let expected_sliding = gemma_sliding_layer_identities(&graph);
        let destination_states = self.opaque_kv_states.as_ref().ok_or_else(|| {
            Gemma4ExecutionLayoutError::invalid("opaque full-attention KV states are absent")
        })?;
        let expected_full = destination_states
            .iter()
            .map(|(&layer, state)| (layer, state.descriptor()))
            .collect::<BTreeMap<_, _>>();
        if expected_full.keys().copied().collect::<BTreeSet<_>>()
            != gemma_full_attention_layers(&graph)
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "fresh Gemma full-attention topology differs from graph",
            ));
        }
        let encoding = expected_full
            .values()
            .next()
            .map(|descriptor| descriptor.cache_encoding())
            .ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid(
                    "fresh Gemma request has no full-attention state",
                )
            })?;
        if expected_full
            .values()
            .any(|descriptor| descriptor.cache_encoding() != encoding)
            || expected_identity.model_lock_fingerprint != self.lock.fingerprint()
            || expected_identity.plan_digest != self.plan.digest_hex()
            || expected_identity.kv_encoding != encoding
            || expected_identity.kv_descriptor_digest
                != gemma_checkpoint_descriptor_digest_from_parts(
                    encoding,
                    &expected_full,
                    &expected_sliding,
                )
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "checkpoint model, plan, encoding, or descriptor identity differs",
            ));
        }

        let expected_layer_keys = expected_full
            .keys()
            .chain(expected_sliding.keys())
            .copied()
            .map(|layer| (StateOwnerKindV1::Kv, layer))
            .collect::<BTreeSet<_>>();
        let actual_layer_keys = checkpoint
            .payload
            .state_layers
            .iter()
            .map(|metadata| (metadata.owner, metadata.layer_id))
            .collect::<BTreeSet<_>>();
        if checkpoint.payload.state_layers.len() != expected_layer_keys.len()
            || actual_layer_keys != expected_layer_keys
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "checkpoint layer topology differs from the fresh Gemma graph",
            ));
        }

        let mut full_kv_layers = BTreeMap::new();
        for (&layer, &descriptor) in &expected_full {
            let state_image = gemma_checkpoint_layer_image(checkpoint, layer)?;
            validate_gemma_layer_image(
                &state_image,
                layer,
                logical_position,
                gemma_kv_plane_kinds(encoding),
                false,
            )?;
            full_kv_layers.insert(
                layer,
                Gemma4KvStateImageV1 {
                    descriptor,
                    image: state_image,
                },
            );
        }
        let mut sliding_layers = BTreeMap::new();
        for (&layer, &identity) in &expected_sliding {
            let state_image = gemma_checkpoint_layer_image(checkpoint, layer)?;
            validate_gemma_layer_image(
                &state_image,
                layer,
                logical_position,
                gemma_kv_plane_kinds(encoding),
                true,
            )?;
            let expected_bytes = gemma_sliding_plane_bytes(identity, logical_position)?;
            for kind in [StatePlaneKindV1::KvKey, StatePlaneKindV1::KvValue] {
                if state_image
                    .planes()
                    .iter()
                    .find(|plane| plane.plane == kind)
                    .and_then(|plane| u64::try_from(plane.bytes.len()).ok())
                    != Some(expected_bytes)
                {
                    return Err(Gemma4ExecutionLayoutError::invalid(format!(
                        "checkpoint sliding layer {layer} byte length differs"
                    )));
                }
            }
            sliding_layers.insert(
                layer,
                Gemma4SlidingStateImageV1 {
                    heads: identity.heads,
                    head_dim: identity.head_dim,
                    capacity: identity.capacity,
                    retention_window: identity.retention_window,
                    image: state_image,
                },
            );
        }
        let image = Gemma4StateImageV1 {
            session_id: self.session.id(),
            model_fingerprint: self.lock.fingerprint().to_owned(),
            plan_digest: *self.plan.digest(),
            state_capacity,
            committed_length: logical_position,
            rope_position_delta,
            full_kv_layers,
            sliding_layers,
            cached_terminal_output: None,
        };
        validate_gemma_state_image(&image, false)?;
        self.restore_validated_state_image(&image)
    }

    /// Imports a fully validated image into this fresh request. Publication
    /// scalars change only after every opaque state and sliding plane succeeds.
    fn restore_validated_state_image(
        &mut self,
        image: &Gemma4StateImageV1,
    ) -> Result<(), Gemma4ExecutionLayoutError> {
        if image.model_fingerprint != self.lock.fingerprint()
            || image.plan_digest != *self.plan.digest()
            || image.state_capacity != self.state_capacity()?
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma state image model, plan, or capacity differs",
            ));
        }
        let graph = crate::build_gemma4_graph(&self.lock, &self.plan, 1, 0, image.state_capacity)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let expected_sliding = gemma_sliding_layer_identities(&graph);
        let destination_states = self.opaque_kv_states.as_ref().ok_or_else(|| {
            Gemma4ExecutionLayoutError::invalid("opaque full-attention KV states are absent")
        })?;
        if image.full_kv_layers.keys().ne(destination_states.keys())
            || image.sliding_layers.keys().ne(expected_sliding.keys())
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma state image topology differs from the fresh graph",
            ));
        }

        // Complete every descriptor, layout, and byte-length check before the
        // first device import. This makes malformed input a no-write failure.
        for (&layer, destination) in destination_states {
            let entry = image.full_kv_layers.get(&layer).ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid(format!(
                    "Gemma full-attention image layer {layer} is absent"
                ))
            })?;
            if entry.descriptor != destination.descriptor() {
                return Err(Gemma4ExecutionLayoutError::invalid(format!(
                    "Gemma full-attention image layer {layer} descriptor differs"
                )));
            }
        }
        for (&layer, &identity) in &expected_sliding {
            let entry = image.sliding_layers.get(&layer).ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid(format!(
                    "Gemma sliding image layer {layer} is absent"
                ))
            })?;
            if entry.identity() != identity {
                return Err(Gemma4ExecutionLayoutError::invalid(format!(
                    "Gemma sliding image layer {layer} identity differs"
                )));
            }
            let bytes = gemma_sliding_plane_bytes(identity, image.committed_length)?;
            for (plane, kind) in [
                (Gemma4KvPlane::Key, StatePlaneKindV1::KvKey),
                (Gemma4KvPlane::Value, StatePlaneKindV1::KvValue),
            ] {
                let tensor = gemma_request_kv_tensor(&self.layout, layer, plane)?;
                validate_gemma_sliding_tensor(tensor, identity, image.committed_length, bytes)?;
                let plane_bytes = entry
                    .image
                    .planes()
                    .iter()
                    .find(|candidate| candidate.plane == kind)
                    .ok_or_else(|| {
                        Gemma4ExecutionLayoutError::invalid(format!(
                            "Gemma sliding image layer {layer} plane is absent"
                        ))
                    })?;
                if u64::try_from(plane_bytes.bytes.len()).ok() != Some(bytes) {
                    return Err(Gemma4ExecutionLayoutError::invalid(format!(
                        "Gemma sliding image layer {layer} plane length differs"
                    )));
                }
            }
        }

        for (&layer, destination) in destination_states {
            let entry = &image.full_kv_layers[&layer];
            self.session
                .import_kv_state_image(destination, &entry.image)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
            let snapshot = self
                .session
                .kv_state_snapshot(destination)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
            if snapshot.length() != image.committed_length {
                return Err(Gemma4ExecutionLayoutError::invalid(format!(
                    "restored Gemma full-attention layer {layer} length differs"
                )));
            }
        }
        for (&layer, &identity) in &expected_sliding {
            let entry = &image.sliding_layers[&layer];
            let bytes = gemma_sliding_plane_bytes(identity, image.committed_length)?;
            for (plane, kind) in [
                (Gemma4KvPlane::Key, StatePlaneKindV1::KvKey),
                (Gemma4KvPlane::Value, StatePlaneKindV1::KvValue),
            ] {
                let tensor = gemma_request_kv_tensor(&self.layout, layer, plane)?;
                let destination = self
                    .buffers
                    .buffer(tensor.id())?
                    .range(0, bytes)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
                let plane_bytes = entry
                    .image
                    .planes()
                    .iter()
                    .find(|candidate| candidate.plane == kind)
                    .expect("validated Gemma sliding plane remains present");
                upload_gemma_buffer_bytes(
                    &self.session,
                    &self.queue,
                    &destination,
                    &plane_bytes.bytes,
                    self.completion_timeout,
                    "Gemma sliding checkpoint restore",
                )?;
            }
        }

        self.state
            .restore_prefix(image.committed_length)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        self.committed_length = image.committed_length;
        self.rope_position_delta = image.rope_position_delta;
        self.last_output = image.cached_terminal_output.clone();
        Ok(())
    }

    /// Publishes a quiescent immutable owner by forking every full-attention
    /// opaque state. Local destination ownership keeps the source untouched if
    /// any layer or snapshot validation fails.
    pub fn publish_prefix(&self) -> Result<Gemma4PrefixStateV1, Gemma4ExecutionLayoutError> {
        let state_snapshot = self
            .state
            .snapshot()
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        if state_snapshot.poisoned {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "cannot publish a poisoned Gemma request",
            ));
        }
        if self.committed_length == 0 || self.last_output.is_none() {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "prefix publication requires a completed non-empty transition",
            ));
        }
        if state_snapshot.committed_length != self.committed_length {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma request and state publication lengths differ",
            ));
        }
        let states = self.opaque_kv_states.as_ref().ok_or_else(|| {
            Gemma4ExecutionLayoutError::invalid("opaque full-attention KV states are absent")
        })?;
        let graph = crate::build_gemma4_graph(&self.lock, &self.plan, 1, 0, self.state_capacity()?)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let expected_full_layers = gemma_full_attention_layers(&graph);
        if states.keys().copied().collect::<BTreeSet<_>>() != expected_full_layers {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma opaque KV layer set differs from full-attention graph layers",
            ));
        }
        let sliding_layers = gemma_sliding_layer_identities(&graph);
        let mut kv_states = BTreeMap::new();
        let mut audit = Gemma4PrefixForkAuditV1::default();
        for (&layer, source) in states {
            let descriptor = source.descriptor();
            if descriptor.layer_id() != layer {
                return Err(Gemma4ExecutionLayoutError::invalid(
                    "Gemma opaque KV state layer identity differs",
                ));
            }
            let snapshot = self
                .session
                .kv_state_snapshot(source)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
            if snapshot.length() != self.committed_length {
                return Err(Gemma4ExecutionLayoutError::invalid(format!(
                    "Gemma source KV layer {layer} length {} differs from committed length {}",
                    snapshot.length(),
                    self.committed_length
                )));
            }
            let (forked, fork_audit) = self
                .session
                .fork_kv_state(source, descriptor)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
            let fork_snapshot = self
                .session
                .kv_state_snapshot(&forked)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
            if fork_snapshot.length() != self.committed_length {
                return Err(Gemma4ExecutionLayoutError::invalid(format!(
                    "Gemma forked KV layer {layer} length {} differs from committed length {}",
                    fork_snapshot.length(),
                    self.committed_length
                )));
            }
            let fallback_resident_bytes = descriptor
                .resident_bytes_per_plane()
                .and_then(|bytes| bytes.checked_mul(2))
                .ok_or_else(|| {
                    Gemma4ExecutionLayoutError::invalid(
                        "Gemma KV prefix resident-byte footprint overflowed",
                    )
                })?;
            let cache_resident_bytes = fork_snapshot
                .physical_memory()
                .map(|physical| physical.committed_bytes_per_plane())
                .and_then(|bytes| bytes.checked_mul(2))
                .unwrap_or(fallback_resident_bytes);
            audit.add(fork_audit, cache_resident_bytes)?;
            kv_states.insert(layer, forked);
        }
        if kv_states.is_empty() {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma prefix has no full-attention KV states",
            ));
        }
        let mut sliding_buffers = BTreeMap::new();
        let mut sliding_bytes = BTreeMap::new();
        for (&layer, identity) in &sliding_layers {
            if identity.capacity < self.committed_length {
                return Err(Gemma4ExecutionLayoutError::invalid(format!(
                    "Gemma sliding layer {layer} capacity is below committed length"
                )));
            }
            for (plane_index, plane) in [Gemma4KvPlane::Key, Gemma4KvPlane::Value]
                .into_iter()
                .enumerate()
            {
                let tensor = gemma_request_kv_tensor(&self.layout, layer, plane)?;
                let bytes = gemma_sliding_plane_bytes(*identity, self.committed_length)?;
                validate_gemma_sliding_tensor(tensor, *identity, self.committed_length, bytes)?;
                let source = self
                    .buffers
                    .buffer(tensor.id())?
                    .range(0, bytes)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
                let (destination, mut copy) = self
                    .session
                    .clone_device_buffer_range_to_request_state(&self.queue, source)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
                let completion = copy
                    .wait(self.completion_timeout)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
                if completion != ExecutionState::Success {
                    return Err(Gemma4ExecutionLayoutError::invalid(format!(
                        "Gemma sliding layer {layer} plane {plane:?} D2D clone did not complete successfully"
                    )));
                }
                audit.add_sliding_plane(plane_index == 0, copy.audit())?;
                sliding_bytes.insert((layer, plane), bytes);
                sliding_buffers.insert((layer, plane), destination);
            }
        }
        if sliding_buffers.len() != sliding_layers.len().saturating_mul(2)
            || sliding_bytes.len() != sliding_buffers.len()
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma sliding prefix clone topology is incomplete",
            ));
        }
        let cached_terminal_output = self
            .last_output
            .clone()
            .expect("checked cached terminal output");
        if cached_terminal_output.state().committed_length != self.committed_length {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "cached Gemma terminal output length differs from request",
            ));
        }
        let identity = Gemma4PrefixIdentityV1 {
            model_fingerprint: self.lock.fingerprint().to_owned(),
            plan_digest: *self.plan.digest(),
            state_capacity: self.state_capacity()?,
            kv_descriptors: states
                .iter()
                .map(|(&layer, state)| (layer, state.descriptor()))
                .collect(),
            sliding_layers,
        };
        Ok(Gemma4PrefixStateV1 {
            inner: Arc::new(Gemma4PrefixStateInner {
                session: Arc::clone(&self.session),
                identity,
                committed_length: self.committed_length,
                rope_position_delta: self.rope_position_delta,
                kv_states,
                sliding_buffers,
                sliding_bytes,
                cached_terminal_output,
                fork_audit: audit,
            }),
        })
    }

    pub fn prefix_state(&self) -> Result<Gemma4PrefixStateV1, Gemma4ExecutionLayoutError> {
        self.publish_prefix()
    }

    pub fn create_prefix_state(&self) -> Result<Gemma4PrefixStateV1, Gemma4ExecutionLayoutError> {
        self.publish_prefix()
    }

    /// Decodes a suffix after a prefix fork. Single-token transitions are used
    /// as the exact continuation fallback; each step preserves the same
    /// transactional state and device-side opaque KV owner.
    pub fn decode_continuation(
        &mut self,
        suffix: &[i32],
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        if suffix.is_empty() {
            return self.last_output.clone().ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid(
                    "empty continuation has no cached terminal output",
                )
            });
        }
        if self.committed_length == 0 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "continuation requires an installed non-empty prefix",
            ));
        }
        validate_gemma_input_token_ids(suffix)?;
        let suffix_len = u64::try_from(suffix.len())
            .map_err(|_| Gemma4ExecutionLayoutError::invalid("continuation length overflowed"))?;
        let end = self
            .committed_length
            .checked_add(suffix_len)
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("continuation length overflowed"))?;
        if end > self.state_capacity()? {
            return Err(Gemma4ExecutionLayoutError::invalid(format!(
                "continuation end {end} exceeds request capacity {}",
                self.state_capacity()?
            )));
        }
        let mut final_output = None;
        for &token_id in suffix {
            final_output = Some(self.decode(token_id)?);
        }
        final_output.ok_or_else(|| {
            Gemma4ExecutionLayoutError::invalid("continuation produced no terminal output")
        })
    }

    pub fn continue_from_prefix(
        &mut self,
        suffix: &[i32],
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.decode_continuation(suffix)
    }

    fn install_prefix(
        &mut self,
        prefix: &Gemma4PrefixStateV1,
    ) -> Result<(), Gemma4ExecutionLayoutError> {
        let state_snapshot = self
            .state
            .snapshot()
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        if state_snapshot.poisoned || self.committed_length != 0 || self.last_output.is_some() {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma prefix installation requires a fresh, quiescent request",
            ));
        }
        if prefix.inner.session.id() != self.session.id() {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "prefix state belongs to a different execution session",
            ));
        }
        if prefix.inner.identity.model_fingerprint != self.lock.fingerprint()
            || prefix.inner.identity.plan_digest != *self.plan.digest()
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "prefix model or weight-plan identity differs from request",
            ));
        }
        let destination_states = self.opaque_kv_states.as_ref().ok_or_else(|| {
            Gemma4ExecutionLayoutError::invalid("opaque full-attention KV states are absent")
        })?;
        let expected_layers = destination_states.keys().copied().collect::<BTreeSet<_>>();
        let prefix_layers = prefix
            .inner
            .kv_states
            .keys()
            .copied()
            .collect::<BTreeSet<_>>();
        if expected_layers != prefix_layers
            || expected_layers
                != prefix
                    .inner
                    .identity
                    .kv_descriptors
                    .keys()
                    .copied()
                    .collect()
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "prefix full-attention KV layer set differs from request",
            ));
        }
        if prefix.committed_length() == 0
            || prefix.committed_length() > self.state_capacity()?
            || prefix.committed_length() > prefix.inner.identity.state_capacity
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "prefix length exceeds request or source capacity",
            ));
        }
        if prefix.cached_terminal_output().state().committed_length != prefix.committed_length() {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "cached prefix terminal output length differs from prefix",
            ));
        }

        let graph = crate::build_gemma4_graph(&self.lock, &self.plan, 1, 0, self.state_capacity()?)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let expected_sliding_layers = gemma_sliding_layer_identities(&graph);
        if prefix.inner.identity.sliding_layers.len() != expected_sliding_layers.len()
            || prefix.inner.sliding_buffers.len() != expected_sliding_layers.len().saturating_mul(2)
            || prefix.inner.sliding_bytes.len() != prefix.inner.sliding_buffers.len()
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "prefix sliding-attention K/V topology differs from request",
            ));
        }
        for (&layer, expected) in &expected_sliding_layers {
            let source = prefix
                .inner
                .identity
                .sliding_layers
                .get(&layer)
                .ok_or_else(|| {
                    Gemma4ExecutionLayoutError::invalid(format!(
                        "prefix sliding layer {layer} identity is absent"
                    ))
                })?;
            if source.heads != expected.heads
                || source.head_dim != expected.head_dim
                || source.retention_window != expected.retention_window
                || source.capacity < prefix.committed_length()
                || expected.capacity < prefix.committed_length()
            {
                return Err(Gemma4ExecutionLayoutError::invalid(format!(
                    "prefix sliding layer {layer} identity differs"
                )));
            }
            for plane in [Gemma4KvPlane::Key, Gemma4KvPlane::Value] {
                let key = (layer, plane);
                let bytes = prefix
                    .inner
                    .sliding_bytes
                    .get(&key)
                    .copied()
                    .ok_or_else(|| {
                        Gemma4ExecutionLayoutError::invalid(format!(
                            "prefix sliding layer {layer} plane {plane:?} byte range is absent"
                        ))
                    })?;
                let expected_bytes =
                    gemma_sliding_plane_bytes(*expected, prefix.committed_length())?;
                if bytes != expected_bytes {
                    return Err(Gemma4ExecutionLayoutError::invalid(format!(
                        "prefix sliding layer {layer} plane {plane:?} byte range differs"
                    )));
                }
                let source_buffer = prefix.inner.sliding_buffers.get(&key).ok_or_else(|| {
                    Gemma4ExecutionLayoutError::invalid(format!(
                        "prefix sliding layer {layer} plane {plane:?} buffer is absent"
                    ))
                })?;
                if source_buffer.session_id() != self.session.id()
                    || source_buffer.size_bytes() != bytes
                {
                    return Err(Gemma4ExecutionLayoutError::invalid(format!(
                        "prefix sliding layer {layer} plane {plane:?} buffer identity differs"
                    )));
                }
                let tensor = gemma_request_kv_tensor(&self.layout, layer, plane)?;
                validate_gemma_sliding_tensor(tensor, *expected, prefix.committed_length(), bytes)?;
            }
        }

        let mut forked_states = BTreeMap::new();
        for (&layer, destination) in destination_states {
            let source = prefix.inner.kv_states.get(&layer).ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid(format!("prefix KV layer {layer} is absent"))
            })?;
            let source_descriptor = source.descriptor();
            if prefix.inner.identity.kv_descriptors.get(&layer).copied() != Some(source_descriptor)
            {
                return Err(Gemma4ExecutionLayoutError::invalid(format!(
                    "prefix KV layer {layer} identity descriptor differs"
                )));
            }
            let destination_descriptor = destination.descriptor();
            if source_descriptor.layer_id() != layer
                || destination_descriptor.layer_id() != layer
                || source_descriptor.layout() != destination_descriptor.layout()
                || source_descriptor.cache_encoding() != destination_descriptor.cache_encoding()
                || source_descriptor.static_fp8_scales()
                    != destination_descriptor.static_fp8_scales()
                || source_descriptor.capacity() < prefix.committed_length()
                || destination_descriptor.capacity() < prefix.committed_length()
            {
                return Err(Gemma4ExecutionLayoutError::invalid(format!(
                    "prefix KV layer {layer} descriptor differs"
                )));
            }
            let (forked, _) = self
                .session
                .fork_kv_state(source, destination_descriptor)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
            let snapshot = self
                .session
                .kv_state_snapshot(&forked)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
            if snapshot.length() != prefix.committed_length() {
                return Err(Gemma4ExecutionLayoutError::invalid(format!(
                    "installed KV layer {layer} length {} differs from prefix length {}",
                    snapshot.length(),
                    prefix.committed_length()
                )));
            }
            forked_states.insert(layer, forked);
        }

        // Install the immutable sliding ranges through the core D2D contract.
        // No host staging is permitted, and publication below is delayed until
        // every plane reports terminal success.
        for (&layer, identity) in &expected_sliding_layers {
            for plane in [Gemma4KvPlane::Key, Gemma4KvPlane::Value] {
                let key = (layer, plane);
                let bytes = prefix.inner.sliding_bytes[&key];
                let source = prefix.inner.sliding_buffers[&key]
                    .range(0, bytes)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
                let tensor = gemma_request_kv_tensor(&self.layout, layer, plane)?;
                validate_gemma_sliding_tensor(tensor, *identity, prefix.committed_length(), bytes)?;
                let destination = self
                    .buffers
                    .buffer(tensor.id())?
                    .range(0, bytes)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
                let mut copy = self
                    .session
                    .copy_device_to_device(&self.queue, source, destination)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
                let completion = copy
                    .wait(self.completion_timeout)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
                if completion != ExecutionState::Success {
                    return Err(Gemma4ExecutionLayoutError::invalid(format!(
                        "Gemma sliding layer {layer} plane {plane:?} D2D install did not complete successfully"
                    )));
                }
            }
        }

        self.state
            .restore_prefix(prefix.committed_length())
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        self.opaque_kv_states = Some(forked_states);
        self.committed_length = prefix.committed_length();
        self.rope_position_delta = prefix.rope_position_delta();
        self.last_output = Some(prefix.cached_terminal_output().clone());
        Ok(())
    }

    pub const fn committed_length(&self) -> u64 {
        self.committed_length
    }

    pub const fn rope_position_delta(&self) -> i64 {
        self.rope_position_delta
    }

    /// Sets the checked absolute-minus-logical RoPE delta used by subsequent
    /// compacted transitions. This only changes request metadata; the caller
    /// must have installed a state owner whose cached K/V rows use the same
    /// absolute position origin.
    pub fn set_rope_position_delta(
        &mut self,
        delta: i64,
    ) -> Result<(), Gemma4ExecutionLayoutError> {
        if delta < 0 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma RoPE position delta must be non-negative",
            ));
        }
        if self
            .state
            .snapshot()
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?
            .poisoned
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "cannot change RoPE position delta on a poisoned request",
            ));
        }
        self.rope_position_delta = delta;
        Ok(())
    }

    pub fn audit_snapshot(&self) -> Result<Gemma4ExecutionAudit, Gemma4ExecutionLayoutError> {
        self.audit
            .clone()
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("Gemma execution audit is empty"))
    }

    /// Reconciles post-COW ownership for every KV destination in a prefix
    /// continuation and returns the aggregate redacted fork audit. A fresh
    /// request has no fork destinations and therefore returns an explicit
    /// unsupported error instead of silently omitting accounting.
    pub fn refresh_prefix_fork_audit(
        &self,
    ) -> Result<Gemma4PrefixForkAuditV1, Gemma4ExecutionLayoutError> {
        let states = self
            .opaque_kv_states
            .as_ref()
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("Gemma KV states are absent"))?
            .values()
            .collect::<Vec<_>>();
        let queried = self
            .session
            .kv_state_fork_query_all(states.iter().copied())
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let mut aggregate = Gemma4PrefixForkAuditV1::default();
        for (state, audit) in states.into_iter().zip(queried) {
            let descriptor = state.descriptor();
            let fallback_resident_bytes = descriptor
                .resident_bytes_per_plane()
                .and_then(|bytes| bytes.checked_mul(2))
                .ok_or_else(|| {
                    Gemma4ExecutionLayoutError::invalid(
                        "Gemma KV fork resident-byte footprint overflowed",
                    )
                })?;
            let cache_resident_bytes = self
                .session
                .kv_state_snapshot(state)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?
                .physical_memory()
                .map(|physical| physical.committed_bytes_per_plane())
                .and_then(|bytes| bytes.checked_mul(2))
                .unwrap_or(fallback_resident_bytes);
            aggregate.add(audit, cache_resident_bytes)?;
        }
        Ok(aggregate)
    }

    pub fn memory_snapshot(&self) -> crate::AllocationSnapshot {
        self.session.memory_snapshot()
    }

    fn run_graph(
        &mut self,
        graph: Gemma4Graph,
        include_last_logits: bool,
        selector: Option<&DeviceTokenSelectorRequestV1>,
        include_embeddings: bool,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.binding_generation = self
            .binding_generation
            .checked_add(1)
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("binding generation overflowed"))?;
        let options = Gemma4ExecutionOptions {
            binding_generation: self.binding_generation,
            completion_timeout: self.completion_timeout,
            expected_backend: 1,
        };
        if include_last_logits && selector.is_some() {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "device token selector cannot be combined with full logits readback",
            ));
        }
        let output = if include_embeddings {
            self.buffers.execute_transition_with_embeddings_and_kv(
                &graph,
                &self.layout,
                &self.queue,
                &self.state,
                self.opaque_kv_states.as_ref(),
                options,
            )
        } else if include_last_logits {
            self.buffers.execute_transition_with_last_logits_and_kv(
                &graph,
                &self.layout,
                &self.queue,
                &self.state,
                self.opaque_kv_states.as_ref(),
                options,
            )
        } else {
            self.buffers.execute_transition_with_selector_and_kv(
                &graph,
                &self.layout,
                &self.queue,
                &self.state,
                self.opaque_kv_states.as_ref(),
                options,
                selector,
            )
        }?;
        self.committed_length = output.state().committed_length;
        self.record_audit(output.audit())?;
        self.last_output = Some(output.clone());
        Ok(output)
    }

    fn record_audit(
        &mut self,
        audit: &PreparedExecutionAudit,
    ) -> Result<(), Gemma4ExecutionLayoutError> {
        if audit.backend() != 1 || audit.fallback_used() {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma transition is not exact HIP/no-fallback",
            ));
        }
        match &mut self.audit {
            Some(total) => {
                if total.target != audit.target() {
                    return Err(Gemma4ExecutionLayoutError::invalid(
                        "Gemma transition targets differ",
                    ));
                }
                total.submission_count = total
                    .submission_count
                    .checked_add(audit.submission_count())
                    .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("audit count overflow"))?;
                total.kernel_dispatch_count = total
                    .kernel_dispatch_count
                    .checked_add(audit.kernel_dispatch_count())
                    .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("audit count overflow"))?;
                total.segment_count = total
                    .segment_count
                    .checked_add(audit.segment_count())
                    .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("audit count overflow"))?;
                total.boundary_count = total
                    .boundary_count
                    .checked_add(audit.boundary_count())
                    .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("audit count overflow"))?;
                total.fallback_used |= audit.fallback_used();
            }
            None => {
                self.audit = Some(Gemma4ExecutionAudit {
                    target: audit.target().to_owned(),
                    submission_count: audit.submission_count(),
                    kernel_dispatch_count: audit.kernel_dispatch_count(),
                    segment_count: audit.segment_count(),
                    boundary_count: audit.boundary_count(),
                    fallback_used: audit.fallback_used(),
                });
            }
        }
        Ok(())
    }

    fn state_capacity(&self) -> Result<u64, Gemma4ExecutionLayoutError> {
        self.layout
            .tensors
            .iter()
            .find_map(|tensor| match tensor.backing {
                Gemma4TensorBacking::RequestKv { .. } => tensor.view.shape().first().copied(),
                _ => None,
            })
            .and_then(|capacity| u64::try_from(capacity).ok())
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("Gemma state capacity is absent"))
    }
}

impl Gemma4ExecutionOutput {
    pub fn token_ids(&self) -> &[i32] {
        &self.token_ids
    }

    pub fn last_logits(&self) -> Option<&[f32]> {
        self.last_logits.as_deref()
    }

    /// Selection metadata returned by the device token-selector route. The
    /// legacy Argmax and explicit host-logits paths leave this absent.
    pub fn selection(&self) -> Option<&SamplingSelectionV1> {
        self.selection.as_ref()
    }

    /// Final-normalized hidden rows in row-major BF16, published only by the
    /// explicit embedding prefill route.
    pub fn embeddings_bf16(&self) -> Option<&[u16]> {
        self.embeddings_bf16.as_deref()
    }

    /// Descriptive alias for callers that name the representation by its
    /// graph boundary.
    pub fn final_hidden_states_bf16(&self) -> Option<&[u16]> {
        self.embeddings_bf16()
    }

    pub const fn state(&self) -> crate::Gemma4RequestStateSnapshot {
        self.state
    }

    pub fn audit(&self) -> &PreparedExecutionAudit {
        &self.audit
    }
}

impl Gemma4ProvisionedBuffers {
    pub fn session_id(&self) -> crate::ExecutionSessionId {
        self.session.id()
    }

    pub fn buffer(&self, tensor_id: usize) -> Result<&ExecutionBuffer, Gemma4ExecutionLayoutError> {
        self.buffers
            .get(tensor_id)
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("provisioned tensor id is absent"))
    }

    pub fn bind(
        &self,
        layout: &Gemma4ExecutionLayout,
        tensor_id: usize,
        access: AccessMode,
    ) -> Result<OwnedTensorBinding, Gemma4ExecutionLayoutError> {
        let tensor = layout
            .tensors
            .get(tensor_id)
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("layout tensor id is absent"))?;
        self.session
            .bind(self.buffer(tensor_id)?, tensor.view.clone(), access)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))
    }

    fn immutable_buffers(
        &self,
        layout: &Gemma4ExecutionLayout,
    ) -> Result<BTreeMap<String, (Gemma4TensorBacking, ExecutionBuffer)>, Gemma4ExecutionLayoutError>
    {
        let mut immutable = BTreeMap::new();
        for tensor in &layout.tensors {
            if !matches!(
                tensor.backing,
                Gemma4TensorBacking::ModelWeight { .. } | Gemma4TensorBacking::ConstantBf16 { .. }
            ) {
                continue;
            }
            if immutable
                .insert(
                    tensor.name.clone(),
                    (tensor.backing.clone(), self.buffer(tensor.id)?.clone()),
                )
                .is_some()
            {
                return Err(Gemma4ExecutionLayoutError::invalid(
                    "immutable tensor name is duplicated",
                ));
            }
        }
        Ok(immutable)
    }

    pub fn bind_node(
        &self,
        layout: &Gemma4ExecutionLayout,
        node: &Gemma4ExecutionNode,
    ) -> Result<Arc<BoundSemanticOp>, Gemma4ExecutionLayoutError> {
        let inputs = node
            .inputs
            .iter()
            .map(|tensor| self.bind(layout, *tensor, AccessMode::Read))
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = node
            .outputs
            .iter()
            .map(|tensor| self.bind(layout, *tensor, AccessMode::Write))
            .collect::<Result<Vec<_>, _>>()?;
        BoundSemanticOp::new(Arc::new(node.descriptor.clone()), inputs, outputs)
            .map(Arc::new)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))
    }

    /// Builds the ordered K/V tail copies that must complete before the
    /// corresponding attention node is submitted. Destination views retain
    /// their non-zero state-buffer offsets; no host readback or repack occurs.
    pub fn bind_kv_appends(
        &self,
        layout: &Gemma4ExecutionLayout,
        node: &Gemma4ExecutionNode,
    ) -> Result<Vec<Arc<BoundSemanticOp>>, Gemma4ExecutionLayoutError> {
        node.kv_appends
            .iter()
            .map(|append| {
                let source = layout.tensors.get(append.source_tensor).ok_or_else(|| {
                    Gemma4ExecutionLayoutError::invalid("KV append source tensor is absent")
                })?;
                let destination = layout.tensors.get(append.destination_view).ok_or_else(|| {
                    Gemma4ExecutionLayoutError::invalid("KV append destination view is absent")
                })?;
                let descriptor = Arc::new(
                    SemanticOpDescriptor::new(
                        SemanticOpKind::Copy,
                        vec![source.view.clone()],
                        vec![destination.view.clone()],
                    )
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
                );
                BoundSemanticOp::new(
                    descriptor,
                    vec![self.bind(layout, source.id, AccessMode::Read)?],
                    vec![self.bind(layout, destination.id, AccessMode::Write)?],
                )
                .map(Arc::new)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))
            })
            .collect()
    }

    /// Reuses immutable model allocations, request K/V state, and compatible
    /// request-local workspace for a new shape/position layout. The initial
    /// prefill allocation is the capacity owner; decode views may narrow it
    /// but never replace it with per-token device allocations. Weights and
    /// published K/V are not uploaded or copied through the host.
    pub fn rebind_transition(
        &self,
        current_layout: &Gemma4ExecutionLayout,
        next_layout: &Gemma4ExecutionLayout,
    ) -> Result<Self, Gemma4ExecutionLayoutError> {
        if current_layout.model_fingerprint != next_layout.model_fingerprint
            || current_layout.nvfp4_sidecar_fingerprint != next_layout.nvfp4_sidecar_fingerprint
            || current_layout.plan_digest != next_layout.plan_digest
            || current_layout.model_weight_bytes != next_layout.model_weight_bytes
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "transition rebind changes model or weight-plan identity",
            ));
        }
        let mut buffers = Vec::with_capacity(next_layout.tensors.len());
        for tensor in &next_layout.tensors {
            let buffer = match &tensor.backing {
                Gemma4TensorBacking::ModelWeight { .. }
                | Gemma4TensorBacking::ConstantBf16 { .. }
                | Gemma4TensorBacking::RequestKv { .. }
                | Gemma4TensorBacking::Workspace
                | Gemma4TensorBacking::TokenIds
                | Gemma4TensorBacking::Positions => {
                    let current = current_layout
                        .tensors
                        .iter()
                        .find(|candidate| {
                            candidate.name == tensor.name && candidate.backing == tensor.backing
                        })
                        .ok_or_else(|| {
                            Gemma4ExecutionLayoutError::invalid(
                                "reusable transition tensor is absent",
                            )
                        })?;
                    self.buffer(current.id)?.clone()
                }
                Gemma4TensorBacking::Alias { tensor_id } => buffers
                    .get(*tensor_id)
                    .cloned()
                    .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("alias source is absent"))?,
            };
            if tensor.view.end_offset() > buffer.size_bytes() {
                return Err(Gemma4ExecutionLayoutError::invalid(
                    "rebound tensor view exceeds its reused buffer",
                ));
            }
            buffers.push(buffer);
        }
        Ok(Self {
            session: Arc::clone(&self.session),
            buffers,
            prepared_semantics: Arc::clone(&self.prepared_semantics),
        })
    }

    /// Uploads all immutable model weights and scalar/vector constants.
    /// Request token IDs and positions remain explicit per-transition inputs.
    pub fn upload_immutable(
        &self,
        layout: &Gemma4ExecutionLayout,
        plan: &WeightLoadPlan,
        cache: &VerifiedCache,
        queue: &ExecutionQueue,
        completion_timeout: Duration,
    ) -> Result<(), Gemma4ExecutionLayoutError> {
        self.upload_immutable_source(layout, plan, cache, None, queue, completion_timeout)
    }

    pub fn upload_immutable_nvfp4(
        &self,
        layout: &Gemma4ExecutionLayout,
        plan: &WeightLoadPlan,
        cache: &VerifiedCache,
        sidecar: &VerifiedNvfp4Sidecar,
        queue: &ExecutionQueue,
        completion_timeout: Duration,
    ) -> Result<(), Gemma4ExecutionLayoutError> {
        self.upload_immutable_source(
            layout,
            plan,
            cache,
            Some(sidecar),
            queue,
            completion_timeout,
        )
    }

    fn upload_immutable_quantized(
        &self,
        layout: &Gemma4ExecutionLayout,
        plan: &WeightLoadPlan,
        artifact: &dyn GemmaQuantizedSource,
        queue: &ExecutionQueue,
        completion_timeout: Duration,
    ) -> Result<(), Gemma4ExecutionLayoutError> {
        if completion_timeout.is_zero()
            || queue.session_id() != self.session.id()
            || layout.model_fingerprint != plan.lock_fingerprint
            || layout.plan_digest != *plan.digest()
            || plan.repo_id != artifact.repository()
            || plan.resolved_revision != artifact.resolved_revision()
            || layout.nvfp4_sidecar_fingerprint.as_deref() != Some(artifact.recipe_digest())
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "quantized immutable upload identity, queue, or timeout differs",
            ));
        }
        for entry in &plan.entries {
            if entry.classification == WeightClassification::KnownUnconsumed {
                continue;
            }
            let tensor = layout
                .tensors
                .iter()
                .find(|tensor| {
                    matches!(&tensor.backing, Gemma4TensorBacking::ModelWeight { tensor_name } if tensor_name == &entry.tensor_name)
                })
                .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("quantized weight buffer is absent"))?;
            let descriptor = artifact.tensor(&entry.tensor_name).ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("quantized tensor descriptor is absent")
            })?;
            let resident_bytes = gemma_resident_weight_bytes(&tensor.view)?;
            let destination = self
                .buffer(tensor.id)?
                .range(0, resident_bytes)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
            upload_gemma_quantized_weight(
                self.session.as_ref(),
                queue,
                &destination,
                artifact,
                descriptor,
                completion_timeout,
            )?;
        }
        self.upload_constants(layout, queue, completion_timeout)
    }

    fn upload_immutable_source(
        &self,
        layout: &Gemma4ExecutionLayout,
        plan: &WeightLoadPlan,
        cache: &VerifiedCache,
        sidecar: Option<&VerifiedNvfp4Sidecar>,
        queue: &ExecutionQueue,
        completion_timeout: Duration,
    ) -> Result<(), Gemma4ExecutionLayoutError> {
        if completion_timeout.is_zero()
            || queue.session_id() != self.session.id()
            || layout.model_fingerprint != plan.lock_fingerprint
            || layout.plan_digest != *plan.digest()
            || cache.lock_fingerprint != layout.model_fingerprint
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "immutable upload identity, queue, or timeout differs",
            ));
        }
        for entry in &plan.entries {
            let Some(_start) = entry.destination_start else {
                continue;
            };
            let tensor = layout
                .tensors
                .iter()
                .find(|tensor| {
                    matches!(
                        &tensor.backing,
                        Gemma4TensorBacking::ModelWeight { tensor_name }
                            if tensor_name == &entry.tensor_name
                    )
                })
                .ok_or_else(|| {
                    Gemma4ExecutionLayoutError::invalid("loadable weight buffer is absent")
                })?;
            let resident_bytes = gemma_resident_weight_bytes(&tensor.view)?;
            let destination = self
                .buffer(tensor.id)?
                .range(0, resident_bytes)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
            if let Some(sidecar_tensor) =
                sidecar.and_then(|sidecar| sidecar.tensor(&entry.tensor_name))
            {
                if sidecar_tensor.shape.as_slice() != entry.shape.as_slice() {
                    return Err(Gemma4ExecutionLayoutError::invalid(
                        "NVFP4 sidecar shape differs from the weight plan",
                    ));
                }
                upload_gemma_nvfp4_weight(
                    self.session.as_ref(),
                    queue,
                    &destination,
                    sidecar.expect("sidecar tensor requires a sidecar"),
                    &entry.tensor_name,
                    completion_timeout,
                )?;
                continue;
            }
            let length = entry.source_range[1]
                .checked_sub(entry.source_range[0])
                .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("weight range underflow"))?;
            if resident_bytes != length {
                return Err(Gemma4ExecutionLayoutError::invalid(
                    "BF16 resident weight length differs from its source range",
                ));
            }
            upload_verified_weight(WeightUploadRequest {
                plan,
                expected_plan_digest: layout.plan_digest,
                cache,
                tensor_name: &entry.tensor_name,
                expected_dtype: entry.dtype,
                session: self.session.as_ref(),
                queue,
                destination,
                completion_timeout,
            })
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        }
        self.upload_constants(layout, queue, completion_timeout)
    }

    fn upload_constants(
        &self,
        layout: &Gemma4ExecutionLayout,
        queue: &ExecutionQueue,
        completion_timeout: Duration,
    ) -> Result<(), Gemma4ExecutionLayoutError> {
        for tensor in &layout.tensors {
            let Gemma4TensorBacking::ConstantBf16 { bits } = tensor.backing else {
                continue;
            };
            let elements = usize::try_from(tensor.view.element_count()).map_err(|_| {
                Gemma4ExecutionLayoutError::invalid("constant length does not fit usize")
            })?;
            let bytes: Arc<[u8]> = std::iter::repeat_n(bits, elements)
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>()
                .into();
            let mut transfer = self
                .session
                .upload(
                    queue,
                    self.buffer(tensor.id)?
                        .range(0, bytes.len() as u64)
                        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
                    bytes,
                )
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
            require_transfer_success(
                transfer.wait(completion_timeout),
                "Gemma immutable constant upload",
            )?;
        }
        Ok(())
    }

    pub fn upload_transition_inputs(
        &self,
        layout: &Gemma4ExecutionLayout,
        queue: &ExecutionQueue,
        token_ids: &[i32],
        completion_timeout: Duration,
    ) -> Result<(), Gemma4ExecutionLayoutError> {
        let start_position = layout
            .nodes
            .iter()
            .find_map(|node| node.descriptor.rotary_contract())
            .map(|contract| contract.start_position())
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("rotary contract is absent"))?;
        let positions = (0..token_ids.len())
            .map(|offset| {
                let offset = u32::try_from(offset).map_err(|_| {
                    Gemma4ExecutionLayoutError::invalid("position offset does not fit u32")
                })?;
                start_position
                    .checked_add(offset)
                    .and_then(|position| i32::try_from(position).ok())
                    .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("position does not fit i32"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.upload_transition_inputs_with_positions_inner(
            layout,
            queue,
            token_ids,
            &positions,
            completion_timeout,
        )
    }

    /// Uploads request inputs with caller-supplied absolute positions.  This
    /// is the context-window compaction path: state remains compactly indexed
    /// while RoPE consumes the original absolute position for each row.
    pub fn upload_transition_inputs_with_positions(
        &self,
        layout: &Gemma4ExecutionLayout,
        queue: &ExecutionQueue,
        token_ids: &[i32],
        positions: &[i32],
        completion_timeout: Duration,
    ) -> Result<(), Gemma4ExecutionLayoutError> {
        let rotary_contract = layout
            .nodes
            .iter()
            .find_map(|node| node.descriptor.rotary_contract())
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("rotary contract is absent"))?;
        if rotary_contract.position_mode() != crate::RotaryPositionModeV1::Explicit {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "explicit positions require an explicit rotary graph",
            ));
        }
        self.upload_transition_inputs_with_positions_inner(
            layout,
            queue,
            token_ids,
            positions,
            completion_timeout,
        )
    }

    fn upload_transition_inputs_with_positions_inner(
        &self,
        layout: &Gemma4ExecutionLayout,
        queue: &ExecutionQueue,
        token_ids: &[i32],
        positions: &[i32],
        completion_timeout: Duration,
    ) -> Result<(), Gemma4ExecutionLayoutError> {
        if token_ids.len() as u64 != layout_token_count(layout)?
            || positions.len() != token_ids.len()
            || completion_timeout.is_zero()
            || queue.session_id() != self.session.id()
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "transition input length, queue, or timeout differs",
            ));
        }
        let token_tensor = layout
            .tensors
            .iter()
            .find(|tensor| tensor.backing == Gemma4TensorBacking::TokenIds)
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("token tensor is absent"))?;
        let position_tensor = layout
            .tensors
            .iter()
            .find(|tensor| tensor.backing == Gemma4TensorBacking::Positions)
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("position tensor is absent"))?;
        if positions.iter().any(|&position| position < 0) {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "explicit positions must be non-negative",
            ));
        }
        let max_position = layout
            .nodes
            .iter()
            .find_map(|node| node.descriptor.rotary_contract())
            .map(|contract| contract.max_position_embeddings())
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("rotary contract is absent"))?;
        if positions.iter().any(|&position| {
            u32::try_from(position).map_or(true, |position| position >= max_position)
        }) {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "explicit position exceeds rotary context range",
            ));
        }
        for (label, tensor, bytes) in [
            (
                "Gemma token upload",
                token_tensor,
                token_ids
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            (
                "Gemma position upload",
                position_tensor,
                positions
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        ] {
            let mut transfer = self
                .session
                .upload(
                    queue,
                    self.buffer(tensor.id)?
                        .range(0, bytes.len() as u64)
                        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
                    Arc::from(bytes),
                )
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
            require_transfer_success(transfer.wait(completion_timeout), label)?;
        }
        Ok(())
    }

    /// Executes one complete immutable graph transition on the supplied
    /// ordered queue. Submissions are retained without per-op waits; only the
    /// graph's state-publication and terminal-readback boundaries wait.
    pub fn execute_transition(
        &self,
        graph: &Gemma4Graph,
        layout: &Gemma4ExecutionLayout,
        queue: &ExecutionQueue,
        request_state: &crate::Gemma4RequestState,
        options: Gemma4ExecutionOptions,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.execute_transition_with_kv(graph, layout, queue, request_state, None, options)
    }

    fn execute_transition_with_kv(
        &self,
        graph: &Gemma4Graph,
        layout: &Gemma4ExecutionLayout,
        queue: &ExecutionQueue,
        request_state: &crate::Gemma4RequestState,
        opaque_kv_states: Option<&BTreeMap<u32, crate::KvState>>,
        options: Gemma4ExecutionOptions,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.execute_transition_with_selector_and_kv(
            graph,
            layout,
            queue,
            request_state,
            opaque_kv_states,
            options,
            None,
        )
    }

    fn execute_transition_with_embeddings_and_kv(
        &self,
        graph: &Gemma4Graph,
        layout: &Gemma4ExecutionLayout,
        queue: &ExecutionQueue,
        request_state: &crate::Gemma4RequestState,
        opaque_kv_states: Option<&BTreeMap<u32, crate::KvState>>,
        options: Gemma4ExecutionOptions,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.execute_transition_impl(
            graph,
            layout,
            queue,
            request_state,
            opaque_kv_states,
            options,
            false,
            None,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_transition_with_selector_and_kv(
        &self,
        graph: &Gemma4Graph,
        layout: &Gemma4ExecutionLayout,
        queue: &ExecutionQueue,
        request_state: &crate::Gemma4RequestState,
        opaque_kv_states: Option<&BTreeMap<u32, crate::KvState>>,
        options: Gemma4ExecutionOptions,
        selector: Option<&DeviceTokenSelectorRequestV1>,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.execute_transition_impl(
            graph,
            layout,
            queue,
            request_state,
            opaque_kv_states,
            options,
            false,
            selector,
            false,
        )
    }

    pub fn execute_transition_with_last_logits(
        &self,
        graph: &Gemma4Graph,
        layout: &Gemma4ExecutionLayout,
        queue: &ExecutionQueue,
        request_state: &crate::Gemma4RequestState,
        options: Gemma4ExecutionOptions,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.execute_transition_with_last_logits_and_kv(
            graph,
            layout,
            queue,
            request_state,
            None,
            options,
        )
    }

    fn execute_transition_with_last_logits_and_kv(
        &self,
        graph: &Gemma4Graph,
        layout: &Gemma4ExecutionLayout,
        queue: &ExecutionQueue,
        request_state: &crate::Gemma4RequestState,
        opaque_kv_states: Option<&BTreeMap<u32, crate::KvState>>,
        options: Gemma4ExecutionOptions,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.execute_transition_impl(
            graph,
            layout,
            queue,
            request_state,
            opaque_kv_states,
            options,
            true,
            None,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_transition_impl(
        &self,
        graph: &Gemma4Graph,
        layout: &Gemma4ExecutionLayout,
        queue: &ExecutionQueue,
        request_state: &crate::Gemma4RequestState,
        opaque_kv_states: Option<&BTreeMap<u32, crate::KvState>>,
        options: Gemma4ExecutionOptions,
        include_last_logits: bool,
        selector: Option<&DeviceTokenSelectorRequestV1>,
        include_embeddings: bool,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        if options.completion_timeout.is_zero()
            || queue.session_id() != self.session.id()
            || graph.lock_fingerprint() != layout.model_fingerprint
            || graph.weight_plan_digest() != &layout.plan_digest
            || graph.nodes().len() != layout.nodes.len()
            || (include_embeddings && (include_last_logits || selector.is_some()))
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "execution graph, layout, queue, or timeout differs",
            ));
        }
        let mut transition = request_state
            .begin(
                graph.token_count(),
                graph.start_position(),
                options.binding_generation,
            )
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let execution_plan = graph
            .prepared_execution_plan()
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let prepared_transition = transition.transition();
        let mut pending = ExecutionSegment::profiled(options.completion_timeout);
        let mut audit = ExecutionAuditAccumulator::new(options.expected_backend);
        let mut terminal_bytes = None;
        let mut selector_logits = None;
        let mut selector_terminal_seen = false;
        let mut embedding_terminal = None;

        execution_plan
            .execute(prepared_transition, |planned, current| {
                let graph_node = planned.operation();
                let node = layout.nodes.get(graph_node.id()).ok_or_else(|| {
                    Gemma4ExecutionLayoutError::invalid("execution node is absent")
                })?;
                if current != prepared_transition
                    || node.graph_node_id != graph_node.id()
                    || node.descriptor.kind()
                        != graph_node.kind().semantic_kind().ok_or_else(|| {
                            Gemma4ExecutionLayoutError::invalid("graph node has no semantic kind")
                        })?
                {
                    return Err(Gemma4ExecutionLayoutError::invalid(
                        "prepared graph and execution layout differ",
                    ));
                }

                // Embedding mode terminates immediately after the final RMSNorm
                // and intentionally does not submit LM-head, softcap, or
                // Argmax nodes. The pending segment is closed below with the
                // same audited terminal boundary, so no generation token or
                // full-vocabulary row is copied to the host.
                if include_embeddings
                    && graph_node.binding_class() == Gemma4GraphBindingClass::TerminalOutput
                {
                    return Ok(());
                }

                let full_attention = matches!(
                    graph_node.kind(),
                    Gemma4GraphNodeKind::CausalAttention(contract)
                        if contract.sliding_window.is_none()
                );
                if let (true, Some(states)) = (
                    node.descriptor.kind() == SemanticOpKind::CausalAttention && full_attention,
                    opaque_kv_states,
                ) {
                    if planned.boundary_after().is_some()
                        || node.kv_appends.len() != 2
                        || node.inputs.is_empty()
                        || node.outputs.len() != 1
                    {
                        return Err(Gemma4ExecutionLayoutError::invalid(
                            "opaque Gemma attention layout or boundary differs",
                        ));
                    }
                    let layer = graph_node.layer().ok_or_else(|| {
                        Gemma4ExecutionLayoutError::invalid(
                            "opaque Gemma attention node has no layer",
                        )
                    })?;
                    let state = states.get(&layer).ok_or_else(|| {
                        Gemma4ExecutionLayoutError::invalid("opaque Gemma KV state is absent")
                    })?;
                    let key =
                        self.bind(layout, node.kv_appends[0].source_tensor, AccessMode::Read)?;
                    let value =
                        self.bind(layout, node.kv_appends[1].source_tensor, AccessMode::Read)?;
                    let mut append = self
                        .session
                        .append_kv_state(
                            state,
                            queue,
                            key,
                            value,
                            graph.start_position(),
                            graph.start_position(),
                        )
                        .map_err(|error| {
                            Gemma4ExecutionLayoutError::invalid(format!(
                                "{} static KV append failed: {error}",
                                graph_node.label()
                            ))
                        })?;
                    pending
                        .flush_with_kv_append(
                            &format!("{}.static_kv_append", graph_node.label()),
                            &mut append,
                            None,
                            &mut audit,
                        )
                        .map_err(|error| {
                            Gemma4ExecutionLayoutError::invalid(format!(
                                "{} static KV boundary flush failed: {error}",
                                graph_node.label()
                            ))
                        })?;
                    drop(append);
                    let query = self.bind(layout, node.inputs[0], AccessMode::Read)?;
                    let output = self.bind(layout, node.outputs[0], AccessMode::Write)?;
                    let descriptor = crate::CausalAttentionDescriptor::new(
                        graph.start_position(),
                        graph.token_count(),
                        graph.expected_length(),
                    )
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
                    let attention = self
                        .session
                        .causal_attention(state, queue, query, output, descriptor)
                        .map_err(|error| {
                            Gemma4ExecutionLayoutError::invalid(format!(
                                "{} causal attention failed: {error}",
                                graph_node.label()
                            ))
                        })?;
                    pending.retain_causal_attention(graph_node.label(), attention);
                    return Ok(());
                }

                // Device selection replaces the graph's terminal Argmax. The
                // preceding projection remains queued, and the selector is
                // submitted below on this same queue so no full logits row is
                // copied to the host.
                if selector.is_some() && node.descriptor.kind() == SemanticOpKind::Argmax {
                    if planned.boundary_after() != Some(ExecutionBoundaryKind::TerminalReadback)
                        || node.inputs.len() != 1
                        || node.outputs.len() != 1
                    {
                        return Err(Gemma4ExecutionLayoutError::invalid(
                            "terminal selector replacement requires one Argmax input/output",
                        ));
                    }
                    selector_logits = Some((
                        node.inputs[0],
                        layout
                            .tensors
                            .get(node.inputs[0])
                            .ok_or_else(|| {
                                Gemma4ExecutionLayoutError::invalid(
                                    "terminal selector logits tensor is absent",
                                )
                            })?
                            .view
                            .clone(),
                    ));
                    selector_terminal_seen = true;
                    return Ok(());
                }

                for append in self.bind_kv_appends(layout, node)? {
                    let submission = self
                        .submit_bound(append, queue, PreparedCachePolicy::Transient)
                        .map_err(|error| {
                            Gemma4ExecutionLayoutError::invalid(format!(
                                "{} KV append submit failed: {error}",
                                graph_node.label()
                            ))
                        })?;
                    pending
                        .retain_semantic(format!("{}.kv_append", graph_node.label()), submission);
                }

                let operation = self.bind_node(layout, node)?;
                let cache_policy = if node.descriptor.kind() == SemanticOpKind::CausalAttention {
                    PreparedCachePolicy::Transient
                } else {
                    PreparedCachePolicy::Reusable(PreparedDynamicIdentity::stateless(
                        graph.token_count(),
                        0,
                    ))
                };
                let mut submission =
                    self.submit_bound(operation, queue, cache_policy)
                        .map_err(|error| {
                            Gemma4ExecutionLayoutError::invalid(format!(
                                "{} submit failed: {error}",
                                graph_node.label()
                            ))
                        })?;
                let boundary = planned.boundary_after();
                if boundary.is_none() {
                    let final_embedding_norm = include_embeddings
                        && matches!(
                            graph_node.kind(),
                            Gemma4GraphNodeKind::RmsNorm {
                                role: Gemma4NormRole::Final,
                                ..
                            }
                        );
                    if final_embedding_norm {
                        if embedding_terminal
                            .replace((graph_node.label().to_owned(), submission))
                            .is_some()
                        {
                            return Err(Gemma4ExecutionLayoutError::invalid(
                                "embedding execution found duplicate final RMSNorm work",
                            ));
                        }
                    } else {
                        pending.retain_semantic(graph_node.label(), submission);
                    }
                    return Ok(());
                }
                let boundary = boundary.expect("checked boundary presence");
                pending
                    .flush_with_semantic(graph_node.label(), &mut submission, boundary, &mut audit)
                    .map_err(|error| {
                        Gemma4ExecutionLayoutError::invalid(format!(
                            "{} boundary flush failed: {error}",
                            graph_node.label()
                        ))
                    })?;

                if boundary == ExecutionBoundaryKind::TerminalReadback && !include_embeddings {
                    if node.descriptor.kind() != SemanticOpKind::Argmax || node.outputs.len() != 1 {
                        return Err(Gemma4ExecutionLayoutError::invalid(
                            "terminal boundary is not one Argmax output",
                        ));
                    }
                    let output = &layout.tensors[node.outputs[0]];
                    let mut readback = submission
                        .start_output_readback(0)
                        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
                    require_terminal_success(
                        graph_node.label(),
                        readback.wait(options.completion_timeout).map_err(|error| {
                            Gemma4ExecutionLayoutError::invalid(error.to_string())
                        })?,
                    )
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
                    let mut bytes =
                        vec![
                            0_u8;
                            usize::try_from(output.view.payload_bytes()).map_err(|_| {
                                Gemma4ExecutionLayoutError::invalid(
                                    "terminal output byte size does not fit usize",
                                )
                            })?
                        ];
                    let copied = readback
                        .read_into(&mut bytes)
                        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
                    if copied != bytes.len() as u64 || terminal_bytes.replace(bytes).is_some() {
                        return Err(Gemma4ExecutionLayoutError::invalid(
                            "terminal output is duplicate or has a short read",
                        ));
                    }
                }
                transition
                    .complete_boundary(boundary)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
                Ok(())
            })
            .map_err(|error: Gemma4ExecutionLayoutError| error)?;

        let selection = if let Some(selector) = selector {
            if !selector_terminal_seen {
                return Err(Gemma4ExecutionLayoutError::invalid(
                    "device selector did not replace the terminal Argmax",
                ));
            }
            let (logits_tensor_id, logits) = selector_logits.ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("device selector logits are absent")
            })?;
            let selection = self.execute_device_token_selector(
                logits_tensor_id,
                queue,
                &logits,
                selector,
                &mut pending,
                &mut audit,
                options.completion_timeout,
            )?;
            transition
                .complete_boundary(ExecutionBoundaryKind::TerminalReadback)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
            Some(selection)
        } else {
            None
        };

        if include_embeddings {
            let (label, mut terminal) = embedding_terminal.ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid(
                    "embedding execution ended before final RMSNorm work",
                )
            })?;
            pending
                .flush_with_semantic(
                    &label,
                    &mut terminal,
                    ExecutionBoundaryKind::TerminalReadback,
                    &mut audit,
                )
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
            transition
                .complete_boundary(ExecutionBoundaryKind::TerminalReadback)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        } else if !pending.is_empty() {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "execution ended with an unclosed segment",
            ));
        }
        let token_ids = if include_embeddings {
            Vec::new()
        } else if let Some(selection) = &selection {
            vec![i32::try_from(selection.token_id).map_err(|_| {
                Gemma4ExecutionLayoutError::invalid("selected token ID does not fit i32")
            })?]
        } else {
            let bytes = terminal_bytes.ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("execution did not publish Argmax output")
            })?;
            if bytes.len() % 4 != 0 {
                return Err(Gemma4ExecutionLayoutError::invalid(
                    "Argmax output byte length is not i32 aligned",
                ));
            }
            bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        };
        let audit = audit
            .snapshot()
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let last_logits = if include_last_logits {
            Some(self.read_last_logits(layout, queue, options.completion_timeout)?)
        } else {
            None
        };
        let embeddings_bf16 = if include_embeddings {
            Some(self.read_final_hidden_states(graph, layout, queue, options.completion_timeout)?)
        } else {
            None
        };
        let state = transition
            .commit()
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        Ok(Gemma4ExecutionOutput {
            token_ids,
            last_logits,
            selection,
            embeddings_bf16,
            state,
            audit,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_device_token_selector(
        &self,
        logits_tensor_id: usize,
        queue: &ExecutionQueue,
        logits: &TensorView,
        selector: &DeviceTokenSelectorRequestV1,
        pending: &mut ExecutionSegment,
        audit: &mut ExecutionAuditAccumulator,
        completion_timeout: Duration,
    ) -> Result<SamplingSelectionV1, Gemma4ExecutionLayoutError> {
        let vocab = selector.additive_logits().len();
        if vocab == 0 || selector.valid_mask().len() != vocab {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "device selector additive logits/mask must be non-empty and equal-sized",
            ));
        }
        if !selector.valid_mask().iter().any(|&value| value != 0) {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "device selector valid mask rejects every token",
            ));
        }
        if logits.dtype() != DType::Bf16
            || logits.encoding() != Encoding::Unquantized
            || logits.shape().len() != 2
            || logits.shape()[0] == 0
            || logits.shape()[1] != vocab
            || !logits.is_contiguous()
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "terminal selector logits must be contiguous BF16 [tokens,vocab]",
            ));
        }
        let row_bytes = logits
            .payload_bytes()
            .checked_div(u64::try_from(logits.shape()[0]).map_err(|_| {
                Gemma4ExecutionLayoutError::invalid("terminal logits row count is too large")
            })?)
            .ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("terminal logits row size overflowed")
            })?;
        let row_offset = logits
            .byte_offset()
            .checked_add(
                row_bytes
                    .checked_mul(u64::try_from(logits.shape()[0] - 1).map_err(|_| {
                        Gemma4ExecutionLayoutError::invalid(
                            "terminal logits row index is too large",
                        )
                    })?)
                    .ok_or_else(|| {
                        Gemma4ExecutionLayoutError::invalid("terminal logits row offset overflowed")
                    })?,
            )
            .ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("terminal logits row offset overflowed")
            })?;
        let logits_row = TensorView::new(
            DType::Bf16,
            Encoding::Unquantized,
            &[1, vocab],
            &[vocab, 1],
            row_offset,
        )
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let additive_view = TensorView::contiguous(DType::F32, &[1, vocab])
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let mask_view = TensorView::contiguous(DType::U8, &[1, vocab])
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let output_view = TensorView::contiguous(DType::U8, &[16])
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let additive = self
            .session
            .allocate_with_category(
                additive_view.payload_bytes(),
                AllocationCategory::RequestState,
            )
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let valid_mask = self
            .session
            .allocate_with_category(mask_view.payload_bytes(), AllocationCategory::RequestState)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let output = self
            .session
            .allocate_with_category(
                output_view.payload_bytes(),
                AllocationCategory::RequestState,
            )
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let additive_bytes = selector
            .additive_logits()
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        upload_selector_bytes(
            self.session.as_ref(),
            queue,
            &additive,
            &additive_view,
            &additive_bytes,
            completion_timeout,
            "Gemma device selector additive-logit upload",
        )?;
        upload_selector_bytes(
            self.session.as_ref(),
            queue,
            &valid_mask,
            &mask_view,
            selector.valid_mask(),
            completion_timeout,
            "Gemma device selector valid-mask upload",
        )?;
        let contract = TokenSelectorContractV1::new(
            u64::try_from(vocab).map_err(|_| {
                Gemma4ExecutionLayoutError::invalid("selector vocabulary is too large")
            })?,
            selector.temperature(),
            selector.seed(),
            selector.counter(),
        )
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let descriptor = Arc::new(
            SemanticOpDescriptor::new_token_select(
                vec![logits_row.clone(), additive_view.clone(), mask_view.clone()],
                vec![output_view.clone()],
                contract,
            )
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
        );
        let operation = BoundSemanticOp::new(
            descriptor,
            vec![
                self.session
                    .bind(self.buffer(logits_tensor_id)?, logits_row, AccessMode::Read)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
                self.session
                    .bind(&additive, additive_view, AccessMode::Read)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
                self.session
                    .bind(&valid_mask, mask_view, AccessMode::Read)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
            ],
            vec![
                self.session
                    .bind(&output, output_view, AccessMode::Write)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
            ],
        )
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let mut submission =
            self.submit_bound(Arc::new(operation), queue, PreparedCachePolicy::Transient)?;
        pending
            .flush_with_semantic(
                "gemma.token_select",
                &mut submission,
                ExecutionBoundaryKind::TerminalReadback,
                audit,
            )
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let mut readback = submission
            .start_output_readback(0)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        require_terminal_success(
            "gemma.token_select",
            readback
                .wait(completion_timeout)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
        )
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let mut bytes = [0_u8; 16];
        let copied = readback
            .read_into(&mut bytes)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        if copied != 16 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "device selector returned a short or long selected record",
            ));
        }
        let token_id = i32::from_le_bytes(bytes[0..4].try_into().expect("token ID record bytes"));
        let status = u32::from_le_bytes(bytes[4..8].try_into().expect("status record bytes"));
        let logprob = f32::from_le_bytes(bytes[8..12].try_into().expect("logprob record bytes"));
        let reserved = u32::from_le_bytes(bytes[12..16].try_into().expect("reserved record bytes"));
        if status != 0 {
            return Err(Gemma4ExecutionLayoutError::invalid(format!(
                "device selector record status is {status}"
            )));
        }
        if reserved != 0 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "device selector selected record reserved field is non-zero",
            ));
        }
        if token_id < 0
            || usize::try_from(token_id).map_or(true, |id| id >= vocab)
            || selector.valid_mask()[usize::try_from(token_id).unwrap_or(0)] == 0
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "device selector returned an out-of-range or masked token ID",
            ));
        }
        if !logprob.is_finite() {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "device selector returned a non-finite logprob",
            ));
        }
        Ok(SamplingSelectionV1 {
            token_id: u32::try_from(token_id).expect("validated token ID fits u32"),
            logprob: f64::from(logprob),
            top_logprobs: Vec::new(),
        })
    }

    fn read_last_logits(
        &self,
        layout: &Gemma4ExecutionLayout,
        queue: &ExecutionQueue,
        completion_timeout: Duration,
    ) -> Result<Vec<f32>, Gemma4ExecutionLayoutError> {
        let argmax = layout
            .nodes
            .last()
            .filter(|node| node.descriptor.kind() == SemanticOpKind::Argmax)
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("terminal Argmax node is absent"))?;
        if argmax.inputs.len() != 1 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "terminal Argmax does not have one logits input",
            ));
        }
        let logits_id = argmax.inputs[0];
        let logits = layout
            .tensors
            .get(logits_id)
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("terminal logits are absent"))?;
        let shape = logits.view.shape();
        if logits.view.dtype() != DType::Bf16 || shape.len() != 2 || shape[1] != 262_144 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "terminal logits are not BF16 [tokens,262144]",
            ));
        }
        let row_bytes = u64::try_from(shape[1])
            .ok()
            .and_then(|width| width.checked_mul(2))
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("logits row bytes overflowed"))?;
        let row_index =
            u64::try_from(shape[0].checked_sub(1).ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("terminal logits have no rows")
            })?)
            .map_err(|_| {
                Gemma4ExecutionLayoutError::invalid("logits row index does not fit u64")
            })?;
        let row_offset = logits
            .view
            .byte_offset()
            .checked_add(row_index.checked_mul(row_bytes).ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("logits row offset overflowed")
            })?)
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("logits row offset overflowed"))?;
        let maximum = self
            .session
            .max_transfer_bytes()
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        if maximum == 0 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "backend transfer limit is zero",
            ));
        }
        let mut bytes = Vec::with_capacity(
            usize::try_from(row_bytes)
                .map_err(|_| Gemma4ExecutionLayoutError::invalid("logits row is too large"))?,
        );
        let mut relative = 0_u64;
        while relative < row_bytes {
            let length = (row_bytes - relative).min(maximum);
            let offset = row_offset.checked_add(relative).ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("logits chunk offset overflowed")
            })?;
            let range = self
                .buffer(logits_id)?
                .range(offset, length)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
            let mut readback = self
                .session
                .readback(queue, range)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
            require_transfer_success(
                readback.wait(completion_timeout),
                "Gemma terminal logits readback",
            )?;
            let start = bytes.len();
            bytes.resize(
                start
                    .checked_add(usize::try_from(length).map_err(|_| {
                        Gemma4ExecutionLayoutError::invalid("logits chunk is too large")
                    })?)
                    .ok_or_else(|| {
                        Gemma4ExecutionLayoutError::invalid("logits readback size overflowed")
                    })?,
                0,
            );
            let copied = readback
                .read_into(&mut bytes[start..])
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
            if copied != length {
                return Err(Gemma4ExecutionLayoutError::invalid(
                    "terminal logits readback was short",
                ));
            }
            relative = relative.checked_add(length).ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("logits readback progress overflowed")
            })?;
        }
        if bytes.len() % 2 != 0 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "terminal logits bytes are not BF16 aligned",
            ));
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f32::from_bits(u32::from(bits) << 16)
            })
            .collect())
    }

    fn read_final_hidden_states(
        &self,
        graph: &Gemma4Graph,
        layout: &Gemma4ExecutionLayout,
        queue: &ExecutionQueue,
        completion_timeout: Duration,
    ) -> Result<Vec<u16>, Gemma4ExecutionLayoutError> {
        let mut final_nodes = graph
            .nodes()
            .iter()
            .filter(|node| {
                matches!(
                    node.kind(),
                    Gemma4GraphNodeKind::RmsNorm {
                        role: Gemma4NormRole::Final,
                        ..
                    }
                )
            })
            .collect::<Vec<_>>();
        let final_node = final_nodes.pop().ok_or_else(|| {
            Gemma4ExecutionLayoutError::invalid("final Gemma RMSNorm node is absent")
        })?;
        if !final_nodes.is_empty() || final_node.label() != "final_norm" {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "embedding final Gemma RMSNorm node identity is invalid",
            ));
        }
        let execution_node = layout.nodes.get(final_node.id()).ok_or_else(|| {
            Gemma4ExecutionLayoutError::invalid("final norm layout node is absent")
        })?;
        if execution_node.graph_node_id != final_node.id()
            || execution_node.descriptor.kind() != SemanticOpKind::RmsNorm
            || execution_node.outputs.len() != 1
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "embedding final norm execution node identity is invalid",
            ));
        }
        let output_id = execution_node.outputs[0];
        let output = layout
            .tensors
            .get(output_id)
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("final norm output is absent"))?;
        let rows = usize::try_from(graph.token_count()).map_err(|_| {
            Gemma4ExecutionLayoutError::invalid("embedding token count is too large")
        })?;
        if output.view.dtype() != DType::Bf16
            || output.view.encoding() != Encoding::Unquantized
            || output.view.shape() != [rows, GEMMA4_HIDDEN_SIZE as usize]
            || !output.view.is_contiguous()
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "embedding final norm output must be contiguous BF16 [tokens,3840]",
            ));
        }
        let expected_bytes = rows
            .checked_mul(GEMMA4_HIDDEN_SIZE as usize)
            .and_then(|words| words.checked_mul(2))
            .ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("embedding readback size overflowed")
            })?;
        let buffer = self.buffer(output_id)?;
        let source = buffer
            .range(output.view.byte_offset(), output.view.payload_bytes())
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let bytes = read_gemma_buffer_bytes(
            self.session.as_ref(),
            queue,
            &source,
            completion_timeout,
            "embedding final-hidden readback",
        )?;
        if bytes.len() != expected_bytes || bytes.len() % 2 != 0 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "embedding final-hidden readback size differs",
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
            return Err(Gemma4ExecutionLayoutError::invalid(
                "embedding final-hidden readback contains non-finite BF16",
            ));
        }
        Ok(values)
    }

    fn submit_bound(
        &self,
        operation: Arc<BoundSemanticOp>,
        queue: &ExecutionQueue,
        cache_policy: PreparedCachePolicy,
    ) -> Result<crate::Submission, Gemma4ExecutionLayoutError> {
        match self.session.supports(operation.descriptor()) {
            PrepareSupport::Supported => {}
            PrepareSupport::Unsupported { reason } => {
                return Err(Gemma4ExecutionLayoutError::invalid(format!(
                    "{:?} is unsupported: {reason}",
                    operation.descriptor().kind()
                )));
            }
        }
        let prepared = self
            .prepared_semantics
            .prepare(
                self.session.as_ref(),
                operation.descriptor().as_ref().clone(),
                operation.inputs().to_vec(),
                operation.outputs().to_vec(),
                cache_policy,
            )
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        self.session
            .submit(&prepared, queue)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))
    }
}

pub fn provision_gemma4_execution_buffers(
    session: Arc<ExecutionSession>,
    layout: &Gemma4ExecutionLayout,
) -> Result<Gemma4ProvisionedBuffers, Gemma4ExecutionLayoutError> {
    let mut buffers = Vec::with_capacity(layout.tensors.len());
    for tensor in &layout.tensors {
        let buffer = match tensor.backing {
            Gemma4TensorBacking::ModelWeight { .. } => session
                .allocate_with_category(
                    gemma_resident_weight_bytes(&tensor.view)?,
                    AllocationCategory::ModelResident,
                )
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
            Gemma4TensorBacking::Alias { tensor_id } => buffers
                .get(tensor_id)
                .cloned()
                .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("alias source is absent"))?,
            Gemma4TensorBacking::RequestKv { .. } => session
                .allocate_with_category(tensor.view.end_offset(), AllocationCategory::RequestState)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
            Gemma4TensorBacking::Workspace
            | Gemma4TensorBacking::TokenIds
            | Gemma4TensorBacking::Positions => session
                .allocate_with_category(tensor.view.end_offset(), AllocationCategory::Workspace)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
            Gemma4TensorBacking::ConstantBf16 { .. } => session
                .allocate_with_category(tensor.view.end_offset(), AllocationCategory::ModelResident)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
        };
        if tensor.view.end_offset() > buffer.size_bytes() {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "tensor view exceeds its provisioned buffer",
            ));
        }
        buffers.push(buffer);
    }
    Ok(Gemma4ProvisionedBuffers {
        session,
        buffers,
        prepared_semantics: Arc::new(PreparedSemanticCache::default()),
    })
}

fn provision_gemma4_request_buffers(
    session: Arc<ExecutionSession>,
    layout: &Gemma4ExecutionLayout,
    immutable: &BTreeMap<String, (Gemma4TensorBacking, ExecutionBuffer)>,
) -> Result<Gemma4ProvisionedBuffers, Gemma4ExecutionLayoutError> {
    let mut buffers = Vec::with_capacity(layout.tensors.len());
    for tensor in &layout.tensors {
        let buffer = match &tensor.backing {
            Gemma4TensorBacking::ModelWeight { .. } | Gemma4TensorBacking::ConstantBf16 { .. } => {
                let (backing, buffer) = immutable.get(&tensor.name).ok_or_else(|| {
                    Gemma4ExecutionLayoutError::invalid("resident immutable tensor is absent")
                })?;
                if backing != &tensor.backing {
                    return Err(Gemma4ExecutionLayoutError::invalid(
                        "resident immutable tensor backing differs",
                    ));
                }
                buffer.clone()
            }
            Gemma4TensorBacking::Alias { tensor_id } => buffers
                .get(*tensor_id)
                .cloned()
                .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("alias source is absent"))?,
            Gemma4TensorBacking::RequestKv { .. } => session
                .allocate_with_category(tensor.view.end_offset(), AllocationCategory::RequestState)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
            Gemma4TensorBacking::Workspace
            | Gemma4TensorBacking::TokenIds
            | Gemma4TensorBacking::Positions => session
                .allocate_with_category(tensor.view.end_offset(), AllocationCategory::Workspace)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
        };
        if tensor.view.end_offset() > buffer.size_bytes() {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "request tensor view exceeds its provisioned buffer",
            ));
        }
        buffers.push(buffer);
    }
    Ok(Gemma4ProvisionedBuffers {
        session,
        buffers,
        prepared_semantics: Arc::new(PreparedSemanticCache::default()),
    })
}

fn require_transfer_success(
    state: Result<ExecutionState, crate::ExecutionError>,
    label: &str,
) -> Result<(), Gemma4ExecutionLayoutError> {
    match state.map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))? {
        ExecutionState::Success => Ok(()),
        ExecutionState::Pending => Err(Gemma4ExecutionLayoutError::invalid(format!(
            "{label} remained pending"
        ))),
        ExecutionState::Failure => Err(Gemma4ExecutionLayoutError::invalid(format!(
            "{label} reported failure"
        ))),
    }
}

fn upload_selector_bytes(
    session: &ExecutionSession,
    queue: &ExecutionQueue,
    buffer: &ExecutionBuffer,
    view: &TensorView,
    bytes: &[u8],
    completion_timeout: Duration,
    stage: &str,
) -> Result<(), Gemma4ExecutionLayoutError> {
    if bytes.is_empty()
        || view.payload_bytes()
            != u64::try_from(bytes.len()).map_err(|_| {
                Gemma4ExecutionLayoutError::invalid("selector upload length does not fit u64")
            })?
    {
        return Err(Gemma4ExecutionLayoutError::invalid(format!(
            "{stage} bytes do not exactly match the tensor view"
        )));
    }
    let maximum = session
        .max_transfer_bytes()
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
    if maximum == 0 {
        return Err(Gemma4ExecutionLayoutError::invalid(
            "backend transfer limit must be non-zero",
        ));
    }
    let mut offset = 0_u64;
    let total = u64::try_from(bytes.len())
        .map_err(|_| Gemma4ExecutionLayoutError::invalid("selector upload is too large"))?;
    while offset < total {
        let length = (total - offset).min(maximum);
        let start = usize::try_from(offset).map_err(|_| {
            Gemma4ExecutionLayoutError::invalid("selector upload offset is too large")
        })?;
        let end = start
            .checked_add(usize::try_from(length).map_err(|_| {
                Gemma4ExecutionLayoutError::invalid("selector upload chunk is too large")
            })?)
            .ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("selector upload range overflowed")
            })?;
        let destination = buffer
            .range(
                view.byte_offset().checked_add(offset).ok_or_else(|| {
                    Gemma4ExecutionLayoutError::invalid("selector upload offset overflowed")
                })?,
                length,
            )
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let mut transfer = session
            .upload(queue, destination, Arc::from(bytes[start..end].to_vec()))
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        require_transfer_success(transfer.wait(completion_timeout), stage)?;
        offset = offset.checked_add(length).ok_or_else(|| {
            Gemma4ExecutionLayoutError::invalid("selector upload progress overflowed")
        })?;
    }
    Ok(())
}

fn read_gemma_buffer_bytes(
    session: &ExecutionSession,
    queue: &ExecutionQueue,
    source: &crate::BufferRange,
    completion_timeout: Duration,
    stage: &str,
) -> Result<Vec<u8>, Gemma4ExecutionLayoutError> {
    if source.size_bytes() == 0 {
        return Err(Gemma4ExecutionLayoutError::invalid(format!(
            "{stage} source is empty"
        )));
    }
    let maximum = session
        .max_transfer_bytes()
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
    if maximum == 0 {
        return Err(Gemma4ExecutionLayoutError::invalid(
            "backend transfer limit must be non-zero",
        ));
    }
    let total = usize::try_from(source.size_bytes()).map_err(|_| {
        Gemma4ExecutionLayoutError::invalid(format!("{stage} size does not fit usize"))
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve(total).map_err(|_| {
        Gemma4ExecutionLayoutError::invalid(format!("{stage} allocation is too large"))
    })?;
    let mut offset = 0_u64;
    while offset < source.size_bytes() {
        let length = (source.size_bytes() - offset).min(maximum);
        let absolute = source.offset_bytes().checked_add(offset).ok_or_else(|| {
            Gemma4ExecutionLayoutError::invalid(format!("{stage} offset overflowed"))
        })?;
        let range = source
            .buffer()
            .range(absolute, length)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let mut transfer = session
            .readback(queue, range)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        require_terminal_success(
            stage,
            transfer
                .wait(completion_timeout)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
        )
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let chunk = usize::try_from(length).map_err(|_| {
            Gemma4ExecutionLayoutError::invalid(format!("{stage} chunk does not fit usize"))
        })?;
        let start = bytes.len();
        bytes.resize(start.saturating_add(chunk), 0);
        let copied = transfer
            .read_into(&mut bytes[start..])
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        if copied != length {
            return Err(Gemma4ExecutionLayoutError::invalid(format!(
                "{stage} returned a short or long read"
            )));
        }
        offset = offset.checked_add(length).ok_or_else(|| {
            Gemma4ExecutionLayoutError::invalid(format!("{stage} progress overflowed"))
        })?;
    }
    Ok(bytes)
}

fn upload_gemma_buffer_bytes(
    session: &ExecutionSession,
    queue: &ExecutionQueue,
    destination: &crate::BufferRange,
    bytes: &[u8],
    completion_timeout: Duration,
    stage: &str,
) -> Result<(), Gemma4ExecutionLayoutError> {
    if bytes.is_empty() || bytes.len() as u64 != destination.size_bytes() {
        return Err(Gemma4ExecutionLayoutError::invalid(format!(
            "{stage} bytes do not exactly match destination"
        )));
    }
    let maximum = usize::try_from(
        session
            .max_transfer_bytes()
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
    )
    .unwrap_or(usize::MAX);
    if maximum == 0 {
        return Err(Gemma4ExecutionLayoutError::invalid(
            "backend transfer limit must be non-zero",
        ));
    }
    let mut offset = 0_usize;
    while offset < bytes.len() {
        let length = (bytes.len() - offset).min(maximum);
        let absolute = destination
            .offset_bytes()
            .checked_add(offset as u64)
            .ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid(format!("{stage} offset overflowed"))
            })?;
        let range = destination
            .buffer()
            .range(absolute, length as u64)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let mut transfer = session
            .upload(
                queue,
                range,
                Arc::from(bytes[offset..offset + length].to_vec()),
            )
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        require_terminal_success(
            stage,
            transfer
                .wait(completion_timeout)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
        )
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        offset = offset.checked_add(length).ok_or_else(|| {
            Gemma4ExecutionLayoutError::invalid(format!("{stage} progress overflowed"))
        })?;
    }
    Ok(())
}

fn gemma_is_nvfp4_weight(view: &TensorView) -> bool {
    view.dtype() == DType::U8
        && matches!(
            view.encoding(),
            Encoding::Nvfp4 {
                block_size: 16,
                scale_dtype: DType::F8E4M3Fn,
            } | Encoding::Nvfp4W4A4 {
                block_size: 16,
                scale_dtype: DType::F8E4M3Fn,
            }
        )
        && view.shape().len() == 2
}

fn gemma_is_w4a4_weight(view: &TensorView) -> bool {
    matches!(view.encoding(), Encoding::Nvfp4W4A4 { .. })
}

fn gemma_is_fp8_weight(view: &TensorView) -> bool {
    matches!(view.dtype(), DType::F8E4M3Fn | DType::F8E4M3FnuZ)
        && view.encoding()
            == Encoding::Fp8Scaled {
                granularity: crate::Fp8ScaleGranularity::OuterDimension,
                scale_dtype: DType::F32,
                resident: crate::Fp8ResidentRepresentation::PackedBytes,
            }
        && view.shape().len() == 2
}

fn gemma_resident_weight_bytes(view: &TensorView) -> Result<u64, Gemma4ExecutionLayoutError> {
    if gemma_is_fp8_weight(view) {
        let rows = u64::try_from(view.shape()[0])
            .map_err(|_| Gemma4ExecutionLayoutError::invalid("FP8 rows do not fit u64"))?;
        return view
            .payload_bytes()
            .checked_add(
                rows.checked_mul(4).ok_or_else(|| {
                    Gemma4ExecutionLayoutError::invalid("FP8 scale bytes overflowed")
                })?,
            )
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("FP8 resident bytes overflowed"));
    }
    if !gemma_is_nvfp4_weight(view) {
        return Ok(view.payload_bytes());
    }
    let rows = u64::try_from(view.shape()[0])
        .map_err(|_| Gemma4ExecutionLayoutError::invalid("NVFP4 rows do not fit u64"))?;
    let columns = u64::try_from(view.shape()[1])
        .map_err(|_| Gemma4ExecutionLayoutError::invalid("NVFP4 columns do not fit u64"))?;
    let block_scales = rows
        .checked_mul(columns.div_ceil(16))
        .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("NVFP4 block count overflowed"))?;
    view.payload_bytes()
        .checked_add(block_scales)
        .and_then(|bytes| bytes.checked_add(3))
        .map(|bytes| bytes & !3)
        .and_then(|bytes| bytes.checked_add(if gemma_is_w4a4_weight(view) { 8 } else { 4 }))
        .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("NVFP4 resident bytes overflowed"))
}

fn build_unsloth_gemma_resident_bytes(
    artifact: &VerifiedUnslothGemma4Nvfp4,
    descriptor: &crate::QuantizedTensorDescriptor,
) -> Result<Vec<u8>, Gemma4ExecutionLayoutError> {
    let mut bytes = artifact
        .read_source_range(descriptor.value_range)
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
    match descriptor.encoding {
        QuantizedTensorEncoding::UnquantizedBf16 => {}
        QuantizedTensorEncoding::OcpFp8E4M3FnChannelBf16Scale => {
            let plane = descriptor
                .scale_planes
                .iter()
                .find(|plane| plane.role == ScalePlaneRole::WeightChannel)
                .ok_or_else(|| {
                    Gemma4ExecutionLayoutError::invalid("FP8 channel scale is absent")
                })?;
            let source = artifact
                .read_source_range(plane.source_range)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
            if source.len() % 2 != 0 {
                return Err(Gemma4ExecutionLayoutError::invalid(
                    "FP8 BF16 scale plane has an odd byte length",
                ));
            }
            for chunk in source.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let value = f32::from_bits(u32::from(bits) << 16);
                if !value.is_finite() || value <= 0.0 {
                    return Err(Gemma4ExecutionLayoutError::invalid(
                        "FP8 channel scale is non-positive or non-finite",
                    ));
                }
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer => {
            let block = descriptor
                .scale_planes
                .iter()
                .find(|plane| plane.role == ScalePlaneRole::WeightBlock)
                .ok_or_else(|| {
                    Gemma4ExecutionLayoutError::invalid("NVFP4 block scale is absent")
                })?;
            bytes.extend_from_slice(
                &artifact
                    .read_source_range(block.source_range)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
            );
            while bytes.len() & 3 != 0 {
                bytes.push(0);
            }
            for role in [ScalePlaneRole::WeightOuter, ScalePlaneRole::InputOuter] {
                let plane = descriptor
                    .scale_planes
                    .iter()
                    .find(|plane| plane.role == role)
                    .ok_or_else(|| {
                        Gemma4ExecutionLayoutError::invalid("NVFP4 outer scale is absent")
                    })?;
                let decode = artifact
                    .read_f32_reciprocal(plane)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
                bytes.extend_from_slice(&decode.to_le_bytes());
            }
        }
        QuantizedTensorEncoding::Mxfp4E2M1Block32E8M0
        | QuantizedTensorEncoding::Mxfp8E4M3Block32E8M0 => {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "MX encoding is not part of the reviewed Gemma recipe",
            ));
        }
    }
    Ok(bytes)
}

fn upload_gemma_quantized_weight(
    session: &ExecutionSession,
    queue: &ExecutionQueue,
    destination: &crate::BufferRange,
    artifact: &dyn GemmaQuantizedSource,
    descriptor: &crate::QuantizedTensorDescriptor,
    completion_timeout: Duration,
) -> Result<(), Gemma4ExecutionLayoutError> {
    let bytes = artifact.resident_bytes(descriptor)?;
    if u64::try_from(bytes.len()).ok() != Some(destination.size_bytes()) {
        return Err(Gemma4ExecutionLayoutError::invalid(format!(
            "quantized resident bytes differ for {}: expected {}, got {}",
            descriptor.logical_name,
            destination.size_bytes(),
            bytes.len()
        )));
    }
    const MAX_TRANSFER_BYTES: usize = 1_073_741_824;
    let mut relative = 0_u64;
    for chunk in bytes.chunks(MAX_TRANSFER_BYTES) {
        let chunk_len = u64::try_from(chunk.len()).map_err(|_| {
            Gemma4ExecutionLayoutError::invalid("quantized upload chunk does not fit u64")
        })?;
        let offset = destination
            .offset_bytes()
            .checked_add(relative)
            .ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("quantized upload offset overflowed")
            })?;
        let range = crate::BufferRange::new(destination.buffer().clone(), offset, chunk_len)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let mut transfer = session
            .upload(queue, range, Arc::from(chunk))
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        require_transfer_success(
            transfer.wait(completion_timeout),
            "Gemma quantized weight upload",
        )?;
        relative = relative.checked_add(chunk_len).ok_or_else(|| {
            Gemma4ExecutionLayoutError::invalid("quantized upload length overflowed")
        })?;
    }
    Ok(())
}

fn upload_gemma_nvfp4_weight(
    session: &ExecutionSession,
    queue: &ExecutionQueue,
    destination: &crate::BufferRange,
    sidecar: &VerifiedNvfp4Sidecar,
    tensor_name: &str,
    completion_timeout: Duration,
) -> Result<(), Gemma4ExecutionLayoutError> {
    let (values, block_scales, tensor_scale) = sidecar
        .read_tensor_bytes(tensor_name)
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
    let unaligned = values
        .len()
        .checked_add(block_scales.len())
        .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("NVFP4 upload size overflowed"))?;
    let tensor_scale_offset = unaligned
        .checked_add(3)
        .map(|bytes| bytes & !3)
        .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("NVFP4 scale offset overflowed"))?;
    let expected = tensor_scale_offset
        .checked_add(4)
        .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("NVFP4 upload size overflowed"))?;
    if u64::try_from(expected).ok() != Some(destination.size_bytes()) {
        return Err(Gemma4ExecutionLayoutError::invalid(
            "NVFP4 upload bytes differ from the resident allocation",
        ));
    }
    let mut bytes = values;
    bytes.extend_from_slice(&block_scales);
    bytes.resize(tensor_scale_offset, 0);
    bytes.extend_from_slice(&tensor_scale);
    let mut transfer = session
        .upload(queue, destination.clone(), Arc::from(bytes))
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
    require_transfer_success(
        transfer.wait(completion_timeout),
        "Gemma NVFP4 weight upload",
    )
}

fn layout_token_count(layout: &Gemma4ExecutionLayout) -> Result<u64, Gemma4ExecutionLayoutError> {
    layout
        .tensors
        .iter()
        .find(|tensor| tensor.backing == Gemma4TensorBacking::TokenIds)
        .and_then(|tensor| tensor.view.shape().first().copied())
        .and_then(|count| u64::try_from(count).ok())
        .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("token tensor shape is absent"))
}

fn validate_gemma_input_token_ids(token_ids: &[i32]) -> Result<(), Gemma4ExecutionLayoutError> {
    let vocab = i32::try_from(crate::GEMMA4_VOCAB_SIZE)
        .map_err(|_| Gemma4ExecutionLayoutError::invalid("Gemma vocabulary exceeds i32"))?;
    if let Some((index, token)) = token_ids
        .iter()
        .copied()
        .enumerate()
        .find(|(_, token)| *token < 0 || *token >= vocab)
    {
        return Err(Gemma4ExecutionLayoutError::invalid(format!(
            "input token ID {token} at index {index} is outside [0, {})",
            crate::GEMMA4_VOCAB_SIZE
        )));
    }
    Ok(())
}

pub fn build_gemma4_execution_layout(
    graph: &Gemma4Graph,
    plan: &WeightLoadPlan,
) -> Result<Gemma4ExecutionLayout, Gemma4ExecutionLayoutError> {
    build_gemma4_execution_layout_source(graph, plan, None, None)
}

pub fn build_gemma4_nvfp4_execution_layout(
    graph: &Gemma4Graph,
    plan: &WeightLoadPlan,
    sidecar: &VerifiedNvfp4Sidecar,
) -> Result<Gemma4ExecutionLayout, Gemma4ExecutionLayoutError> {
    build_gemma4_execution_layout_source(graph, plan, Some(sidecar), None)
}

pub fn build_gemma4_quantized_execution_layout(
    graph: &Gemma4Graph,
    plan: &WeightLoadPlan,
    artifact: &VerifiedUnslothGemma4Nvfp4,
) -> Result<Gemma4ExecutionLayout, Gemma4ExecutionLayoutError> {
    build_gemma4_quantized_execution_layout_source(graph, plan, artifact)
}

fn build_gemma4_quantized_execution_layout_source(
    graph: &Gemma4Graph,
    plan: &WeightLoadPlan,
    source: &dyn GemmaQuantizedSource,
) -> Result<Gemma4ExecutionLayout, Gemma4ExecutionLayoutError> {
    build_gemma4_execution_layout_source(graph, plan, None, Some(source))
}

fn build_gemma4_execution_layout_source(
    graph: &Gemma4Graph,
    plan: &WeightLoadPlan,
    sidecar: Option<&VerifiedNvfp4Sidecar>,
    artifact: Option<&dyn GemmaQuantizedSource>,
) -> Result<Gemma4ExecutionLayout, Gemma4ExecutionLayoutError> {
    if graph.lock_fingerprint() != plan.lock_fingerprint
        || graph.weight_plan_digest() != plan.digest()
    {
        return Err(Gemma4ExecutionLayoutError::invalid(
            "graph and weight-plan identity differ",
        ));
    }
    if sidecar.is_some() && artifact.is_some() {
        return Err(Gemma4ExecutionLayoutError::invalid(
            "sidecar and first-class quantized artifact are mutually exclusive",
        ));
    }
    let mut builder = LayoutBuilder::new(graph, plan, sidecar, artifact)?;
    builder.build()?;
    builder.finish()
}

struct LayoutBuilder<'a> {
    graph: &'a Gemma4Graph,
    plan: &'a WeightLoadPlan,
    nvfp4_sidecar: Option<&'a VerifiedNvfp4Sidecar>,
    quantized_model: Option<&'a dyn GemmaQuantizedSource>,
    tensors: Vec<Gemma4ExecutionTensor>,
    nodes: Vec<Gemma4ExecutionNode>,
    weights: BTreeMap<String, usize>,
    weight_entries: BTreeMap<String, &'a WeightLoadEntry>,
    node_outputs: Vec<Vec<usize>>,
    token_ids: usize,
    positions: usize,
    workspace_bytes: u64,
    request_state_bytes: u64,
}

impl<'a> LayoutBuilder<'a> {
    fn new(
        graph: &'a Gemma4Graph,
        plan: &'a WeightLoadPlan,
        nvfp4_sidecar: Option<&'a VerifiedNvfp4Sidecar>,
        quantized_model: Option<&'a dyn GemmaQuantizedSource>,
    ) -> Result<Self, Gemma4ExecutionLayoutError> {
        if let Some(sidecar) = nvfp4_sidecar {
            if sidecar.source_lock_fingerprint() != graph.lock_fingerprint()
                || sidecar.tensors().len() > 144
            {
                return Err(Gemma4ExecutionLayoutError::invalid(
                    "NVFP4 sidecar source identity or tensor count differs",
                ));
            }
        }
        let mut builder = Self {
            graph,
            plan,
            nvfp4_sidecar,
            quantized_model,
            tensors: Vec::new(),
            nodes: Vec::with_capacity(graph.nodes().len()),
            weights: BTreeMap::new(),
            weight_entries: BTreeMap::new(),
            node_outputs: vec![Vec::new(); graph.nodes().len()],
            token_ids: usize::MAX,
            positions: usize::MAX,
            workspace_bytes: 0,
            request_state_bytes: 0,
        };
        let mut matched_nvfp4 = 0_usize;
        for entry in &plan.entries {
            if builder
                .weight_entries
                .insert(entry.tensor_name.clone(), entry)
                .is_some()
            {
                return Err(Gemma4ExecutionLayoutError::invalid(
                    "weight plan contains a duplicate tensor name",
                ));
            }
            if entry.classification == WeightClassification::KnownUnconsumed {
                continue;
            }
            entry.destination_start.ok_or_else(|| {
                Gemma4ExecutionLayoutError::invalid("loadable weight has no destination offset")
            })?;
            let view = if let Some(descriptor) =
                quantized_model.and_then(|artifact| artifact.tensor(&entry.tensor_name))
            {
                if descriptor.logical_shape != entry.shape {
                    return Err(Gemma4ExecutionLayoutError::invalid(
                        "quantized tensor shape differs from the topology plan",
                    ));
                }
                quantized_gemma_tensor_view(descriptor)?
            } else {
                match nvfp4_sidecar.and_then(|sidecar| sidecar.tensor(&entry.tensor_name)) {
                    Some(tensor) => {
                        matched_nvfp4 += 1;
                        if tensor.shape.as_slice() != entry.shape.as_slice()
                            || !entry.tensor_name.contains(".mlp.")
                            || !entry.tensor_name.ends_with("_proj.weight")
                        {
                            return Err(Gemma4ExecutionLayoutError::invalid(
                                "NVFP4 tensor does not match one Gemma MLP weight",
                            ));
                        }
                        nvfp4_tensor_view(&entry.shape)?
                    }
                    None => tensor_view(entry.dtype, &entry.shape, 0)?,
                }
            };
            let id = builder.push_tensor(
                entry.tensor_name.clone(),
                view,
                Gemma4TensorBacking::ModelWeight {
                    tensor_name: entry.tensor_name.clone(),
                },
            )?;
            builder.weights.insert(entry.tensor_name.clone(), id);
        }
        if let Some(sidecar) = nvfp4_sidecar {
            if matched_nvfp4 != sidecar.tensors().len() {
                return Err(Gemma4ExecutionLayoutError::invalid(
                    "NVFP4 sidecar contains a tensor outside the loadable Gemma MLP set",
                ));
            }
        }
        builder.token_ids = builder.push_tensor(
            "request.token_ids",
            contiguous(DType::I32, &[graph.token_count()])?,
            Gemma4TensorBacking::TokenIds,
        )?;
        builder.positions = builder.push_tensor(
            "request.positions",
            contiguous(DType::I32, &[graph.token_count()])?,
            Gemma4TensorBacking::Positions,
        )?;
        Ok(builder)
    }

    fn build(&mut self) -> Result<(), Gemma4ExecutionLayoutError> {
        for node in self.graph.nodes() {
            let node_id = node.id();
            let result = match node.kind() {
                Gemma4GraphNodeKind::Embedding { weight } => {
                    let weight = self.weight(weight)?;
                    let output = self.workspace(
                        format!("{}.output", node.label()),
                        contiguous(DType::Bf16, &[self.graph.token_count(), GEMMA4_HIDDEN_SIZE])?,
                    )?;
                    self.semantic(
                        node_id,
                        SemanticOpKind::Embedding,
                        vec![weight, self.token_ids],
                        vec![output],
                    )
                }
                Gemma4GraphNodeKind::ScaleConstant { value_bits } => {
                    let scalar =
                        self.constant(node.label(), f32_to_bf16_rne(f32::from_bits(*value_bits)))?;
                    self.elementwise(node_id, SemanticOpKind::ScalarMul, scalar)
                }
                Gemma4GraphNodeKind::ScaleWeight { weight } => {
                    self.elementwise(node_id, SemanticOpKind::ScalarMul, self.weight(weight)?)
                }
                Gemma4GraphNodeKind::RmsNorm {
                    role,
                    scale_mode,
                    epsilon_bits,
                    weight,
                } => self.rms_norm(
                    node_id,
                    *role,
                    *scale_mode,
                    *epsilon_bits,
                    weight.as_deref(),
                ),
                Gemma4GraphNodeKind::Matmul { weight, .. } => self.matmul(node_id, weight),
                Gemma4GraphNodeKind::Rotary(contract) => self.rotary(node_id, *contract),
                Gemma4GraphNodeKind::CausalAttention(contract) => {
                    self.attention(node_id, *contract)
                }
                Gemma4GraphNodeKind::GeluTanhMul => {
                    self.binary(node_id, SemanticOpKind::GeluTanhMul)
                }
                Gemma4GraphNodeKind::Add => self.binary(node_id, SemanticOpKind::Add),
                Gemma4GraphNodeKind::LogitSoftcap { cap_bits } => {
                    let scalar =
                        self.constant(node.label(), f32_to_bf16_rne(f32::from_bits(*cap_bits)))?;
                    self.elementwise(node_id, SemanticOpKind::TanhSoftcap, scalar)
                }
                Gemma4GraphNodeKind::Argmax => self.argmax(node_id),
            }?;
            self.node_outputs[node_id] = result;
        }
        Ok(())
    }

    fn finish(self) -> Result<Gemma4ExecutionLayout, Gemma4ExecutionLayoutError> {
        if self.nodes.len() != self.graph.nodes().len()
            || self
                .nodes
                .iter()
                .enumerate()
                .any(|(index, node)| node.graph_node_id != index)
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "execution nodes do not map one-to-one to graph nodes",
            ));
        }
        let model_weight_bytes = self
            .tensors
            .iter()
            .filter(|tensor| matches!(tensor.backing, Gemma4TensorBacking::ModelWeight { .. }))
            .try_fold(0_u64, |total, tensor| {
                total
                    .checked_add(gemma_resident_weight_bytes(&tensor.view)?)
                    .ok_or_else(|| {
                        Gemma4ExecutionLayoutError::invalid("model resident bytes overflowed")
                    })
            })?;
        Ok(Gemma4ExecutionLayout {
            model_fingerprint: self.graph.lock_fingerprint().to_owned(),
            nvfp4_sidecar_fingerprint: self
                .quantized_model
                .map(|artifact| artifact.recipe_digest().to_owned())
                .or_else(|| {
                    self.nvfp4_sidecar
                        .map(|sidecar| sidecar.manifest_fingerprint().to_owned())
                }),
            plan_digest: *self.plan.digest(),
            tensors: self.tensors,
            nodes: self.nodes,
            model_weight_bytes,
            workspace_bytes: self.workspace_bytes,
            request_state_bytes: self.request_state_bytes,
        })
    }

    fn semantic(
        &mut self,
        node_id: usize,
        kind: SemanticOpKind,
        inputs: Vec<usize>,
        outputs: Vec<usize>,
    ) -> Result<Vec<usize>, Gemma4ExecutionLayoutError> {
        let descriptor =
            SemanticOpDescriptor::new(kind, self.views(&inputs)?, self.views(&outputs)?)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        self.nodes.push(Gemma4ExecutionNode {
            graph_node_id: node_id,
            descriptor,
            inputs,
            outputs: outputs.clone(),
            kv_appends: Vec::new(),
        });
        Ok(outputs)
    }

    fn elementwise(
        &mut self,
        node_id: usize,
        kind: SemanticOpKind,
        scalar: usize,
    ) -> Result<Vec<usize>, Gemma4ExecutionLayoutError> {
        let input = self.predecessor_output(node_id, 0, 0)?;
        let output = self.workspace_like(node_id, input)?;
        self.semantic(node_id, kind, vec![input, scalar], vec![output])
    }

    fn binary(
        &mut self,
        node_id: usize,
        kind: SemanticOpKind,
    ) -> Result<Vec<usize>, Gemma4ExecutionLayoutError> {
        let left = self.predecessor_output(node_id, 0, 0)?;
        let right = self.predecessor_output(node_id, 1, 0)?;
        let output = self.workspace_like(node_id, left)?;
        self.semantic(node_id, kind, vec![left, right], vec![output])
    }

    fn rms_norm(
        &mut self,
        node_id: usize,
        role: Gemma4NormRole,
        scale_mode: crate::RmsNormScaleMode,
        epsilon_bits: u32,
        weight: Option<&str>,
    ) -> Result<Vec<usize>, Gemma4ExecutionLayoutError> {
        let source = self.predecessor_output(node_id, 0, 0)?;
        let width = match (role, weight) {
            (Gemma4NormRole::Query | Gemma4NormRole::Key, Some(name)) => {
                let scale = self.weight(name)?;
                let shape = self.tensor(scale)?.view.shape();
                if shape.len() != 1 {
                    return Err(Gemma4ExecutionLayoutError::invalid(
                        "attention norm scale is not rank one",
                    ));
                }
                shape[0]
            }
            (Gemma4NormRole::ValueUnitScale, None) => {
                let layer = self.graph.nodes()[node_id].layer().ok_or_else(|| {
                    Gemma4ExecutionLayoutError::invalid("value norm has no layer")
                })?;
                match crate::reviewed_layer_schedule()[layer as usize] {
                    Gemma4LayerType::SlidingAttention => 256,
                    Gemma4LayerType::FullAttention => 512,
                }
            }
            _ => usize::try_from(GEMMA4_HIDDEN_SIZE).map_err(|_| {
                Gemma4ExecutionLayoutError::invalid("hidden size does not fit usize")
            })?,
        };
        let elements = usize::try_from(self.tensor(source)?.view.element_count())
            .map_err(|_| Gemma4ExecutionLayoutError::invalid("norm elements do not fit usize"))?;
        if elements % width != 0 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "RMSNorm input cannot be partitioned by its scale width",
            ));
        }
        let input = self.alias(
            format!("{}.norm_input", self.graph.nodes()[node_id].label()),
            source,
            contiguous_usize(DType::Bf16, &[elements / width, width])?,
        )?;
        let scale = match weight {
            Some(name) => self.weight(name)?,
            None => self.constant_vector(
                self.graph.nodes()[node_id].label(),
                f32_to_bf16_rne(1.0),
                width,
            )?,
        };
        if self.tensor(scale)?.view.shape() != [width] && self.tensor(scale)?.view.shape() != [1] {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "RMSNorm scale shape differs from normalized width",
            ));
        }
        let output = self.workspace(
            format!("{}.output", self.graph.nodes()[node_id].label()),
            contiguous_usize(DType::Bf16, &[elements / width, width])?,
        )?;
        let contract = RmsNormContract::new(f32::from_bits(epsilon_bits), scale_mode)
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let descriptor = SemanticOpDescriptor::new_rms_norm_with_contract(
            self.views(&[input, scale])?,
            self.views(&[output])?,
            contract,
        )
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        self.nodes.push(Gemma4ExecutionNode {
            graph_node_id: node_id,
            descriptor,
            inputs: vec![input, scale],
            outputs: vec![output],
            kv_appends: Vec::new(),
        });
        Ok(vec![output])
    }

    fn matmul(
        &mut self,
        node_id: usize,
        weight_name: &str,
    ) -> Result<Vec<usize>, Gemma4ExecutionLayoutError> {
        let source = self.predecessor_output(node_id, 0, 0)?;
        let weight = self.weight(weight_name)?;
        let weight_shape = self.tensor(weight)?.view.shape();
        if weight_shape.len() != 2 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "matmul weight is not rank two",
            ));
        }
        let n = weight_shape[0];
        let k = weight_shape[1];
        let elements = usize::try_from(self.tensor(source)?.view.element_count())
            .map_err(|_| Gemma4ExecutionLayoutError::invalid("matmul input is too large"))?;
        if elements % k != 0 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "matmul input width differs from weight K",
            ));
        }
        let m = elements / k;
        let input = self.alias(
            format!("{}.matmul_input", self.graph.nodes()[node_id].label()),
            source,
            contiguous_usize(DType::Bf16, &[m, k])?,
        )?;
        let output = self.workspace(
            format!("{}.output", self.graph.nodes()[node_id].label()),
            contiguous_usize(DType::Bf16, &[m, n])?,
        )?;
        self.semantic(
            node_id,
            SemanticOpKind::Matmul,
            vec![input, weight],
            vec![output],
        )
    }

    fn rotary(
        &mut self,
        node_id: usize,
        rotary: crate::Gemma4RopeDescriptor,
    ) -> Result<Vec<usize>, Gemma4ExecutionLayoutError> {
        let query_source = self.predecessor_output(node_id, 0, 0)?;
        let key_source = self.predecessor_output(node_id, 1, 0)?;
        let m = usize::try_from(self.graph.token_count())
            .map_err(|_| Gemma4ExecutionLayoutError::invalid("token count does not fit usize"))?;
        let query_view = contiguous_usize(
            DType::Bf16,
            &[m, rotary.q_heads as usize, rotary.head_dim as usize],
        )?;
        let key_view = contiguous_usize(
            DType::Bf16,
            &[m, rotary.kv_heads as usize, rotary.head_dim as usize],
        )?;
        let query = self.alias(
            format!("{}.query", self.graph.nodes()[node_id].label()),
            query_source,
            query_view.clone(),
        )?;
        let key = self.alias(
            format!("{}.key", self.graph.nodes()[node_id].label()),
            key_source,
            key_view.clone(),
        )?;
        let query_output = self.workspace(
            format!("{}.query_output", self.graph.nodes()[node_id].label()),
            query_view,
        )?;
        let key_output = self.workspace(
            format!("{}.key_output", self.graph.nodes()[node_id].label()),
            key_view,
        )?;
        let descriptor = SemanticOpDescriptor::new_rotary(
            self.views(&[query, key, self.positions])?,
            self.views(&[query_output, key_output])?,
            rotary
                .semantic_contract_with_position_mode(
                    self.graph.start_position(),
                    self.graph.token_count(),
                    self.graph.rotary_position_mode(),
                )
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
        )
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        self.nodes.push(Gemma4ExecutionNode {
            graph_node_id: node_id,
            descriptor,
            inputs: vec![query, key, self.positions],
            outputs: vec![query_output, key_output],
            kv_appends: Vec::new(),
        });
        Ok(vec![query_output, key_output])
    }

    fn attention(
        &mut self,
        node_id: usize,
        attention: crate::Gemma4AttentionDescriptor,
    ) -> Result<Vec<usize>, Gemma4ExecutionLayoutError> {
        let graph_node = &self.graph.nodes()[node_id];
        let layer = graph_node
            .layer()
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("attention node has no layer"))?;
        let query = self.predecessor_output(node_id, 0, 0)?;
        let key_tail = self.predecessor_output(node_id, 0, 1)?;
        let value_source = self.predecessor_output(node_id, 1, 0)?;
        let m = usize::try_from(self.graph.token_count())
            .map_err(|_| Gemma4ExecutionLayoutError::invalid("token count does not fit usize"))?;
        let kv_shape = [m, attention.kv_heads as usize, attention.head_dim as usize];
        let value_tail = self.alias(
            format!("{}.value_tail", graph_node.label()),
            value_source,
            contiguous_usize(DType::Bf16, &kv_shape)?,
        )?;
        let key_state = self.kv_state(layer, Gemma4KvPlane::Key, attention)?;
        let value_state = self.kv_state(layer, Gemma4KvPlane::Value, attention)?;
        let key_destination = self.kv_tail_view(layer, Gemma4KvPlane::Key, key_state, attention)?;
        let value_destination =
            self.kv_tail_view(layer, Gemma4KvPlane::Value, value_state, attention)?;
        let prefix_shape = [
            usize::try_from(self.graph.expected_length())
                .map_err(|_| Gemma4ExecutionLayoutError::invalid("KV length does not fit usize"))?,
            attention.kv_heads as usize,
            attention.head_dim as usize,
        ];
        let key_prefix = self.alias(
            format!("{}.key_prefix", graph_node.label()),
            key_state,
            contiguous_usize(DType::Bf16, &prefix_shape)?,
        )?;
        let value_prefix = self.alias(
            format!("{}.value_prefix", graph_node.label()),
            value_state,
            contiguous_usize(DType::Bf16, &prefix_shape)?,
        )?;
        let output = self.workspace(
            format!("{}.output", graph_node.label()),
            self.tensor(query)?.view.clone(),
        )?;
        let descriptor = SemanticOpDescriptor::new_causal_attention(
            self.views(&[query, key_prefix, value_prefix])?,
            self.views(&[output])?,
            attention
                .semantic_contract(self.graph.start_position(), self.graph.token_count())
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
        )
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        self.nodes.push(Gemma4ExecutionNode {
            graph_node_id: node_id,
            descriptor,
            inputs: vec![query, key_prefix, value_prefix],
            outputs: vec![output],
            kv_appends: vec![
                Gemma4KvAppendLayout {
                    source_tensor: key_tail,
                    state_tensor: key_state,
                    destination_view: key_destination,
                },
                Gemma4KvAppendLayout {
                    source_tensor: value_tail,
                    state_tensor: value_state,
                    destination_view: value_destination,
                },
            ],
        });
        Ok(vec![output])
    }

    fn argmax(&mut self, node_id: usize) -> Result<Vec<usize>, Gemma4ExecutionLayoutError> {
        let input = self.predecessor_output(node_id, 0, 0)?;
        let shape = self.tensor(input)?.view.shape();
        if shape.len() != 2 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "argmax logits are not rank two",
            ));
        }
        let output = self.workspace(
            format!("{}.output", self.graph.nodes()[node_id].label()),
            contiguous_usize(DType::I32, &[shape[0]])?,
        )?;
        self.semantic(node_id, SemanticOpKind::Argmax, vec![input], vec![output])
    }

    fn kv_state(
        &mut self,
        layer: u32,
        plane: Gemma4KvPlane,
        attention: crate::Gemma4AttentionDescriptor,
    ) -> Result<usize, Gemma4ExecutionLayoutError> {
        if let Some(tensor) = self
            .tensors
            .iter()
            .find(|tensor| tensor.backing == Gemma4TensorBacking::RequestKv { layer, plane })
        {
            return Ok(tensor.id);
        }
        let view = contiguous(
            DType::Bf16,
            &[
                self.graph.state_capacity(),
                u64::from(attention.kv_heads),
                u64::from(attention.head_dim),
            ],
        )?;
        self.push_tensor(
            format!("layer.{layer}.kv.{plane:?}"),
            view,
            Gemma4TensorBacking::RequestKv { layer, plane },
        )
    }

    fn kv_tail_view(
        &mut self,
        layer: u32,
        plane: Gemma4KvPlane,
        state: usize,
        attention: crate::Gemma4AttentionDescriptor,
    ) -> Result<usize, Gemma4ExecutionLayoutError> {
        let row_elements = u64::from(attention.kv_heads)
            .checked_mul(u64::from(attention.head_dim))
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("KV row elements overflow"))?;
        let offset = self
            .graph
            .start_position()
            .checked_mul(row_elements)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("KV tail offset overflow"))?;
        let m = usize::try_from(self.graph.token_count())
            .map_err(|_| Gemma4ExecutionLayoutError::invalid("token count does not fit usize"))?;
        let heads = attention.kv_heads as usize;
        let head_dim = attention.head_dim as usize;
        let view = TensorView::new(
            DType::Bf16,
            Encoding::Unquantized,
            &[m, heads, head_dim],
            &[heads * head_dim, head_dim, 1],
            offset,
        )
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        self.alias(format!("layer.{layer}.kv.{plane:?}.tail"), state, view)
    }

    fn predecessor_output(
        &self,
        node_id: usize,
        predecessor_index: usize,
        output_index: usize,
    ) -> Result<usize, Gemma4ExecutionLayoutError> {
        let predecessor = *self.graph.nodes()[node_id]
            .predecessors()
            .get(predecessor_index)
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("node predecessor is absent"))?;
        self.node_outputs
            .get(predecessor)
            .and_then(|outputs| outputs.get(output_index))
            .copied()
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("predecessor output is absent"))
    }

    fn weight(&self, name: &str) -> Result<usize, Gemma4ExecutionLayoutError> {
        self.weights.get(name).copied().ok_or_else(|| {
            Gemma4ExecutionLayoutError::invalid(format!("weight is not dispatchable: {name}"))
        })
    }

    fn constant(&mut self, label: &str, bits: u16) -> Result<usize, Gemma4ExecutionLayoutError> {
        self.constant_vector(label, bits, 1)
    }

    fn constant_vector(
        &mut self,
        label: &str,
        bits: u16,
        width: usize,
    ) -> Result<usize, Gemma4ExecutionLayoutError> {
        self.push_tensor(
            format!("{label}.scalar"),
            contiguous_usize(DType::Bf16, &[width])?,
            Gemma4TensorBacking::ConstantBf16 { bits },
        )
    }

    fn workspace_like(
        &mut self,
        node_id: usize,
        source: usize,
    ) -> Result<usize, Gemma4ExecutionLayoutError> {
        self.workspace(
            format!("{}.output", self.graph.nodes()[node_id].label()),
            self.tensor(source)?.view.clone(),
        )
    }

    fn workspace(
        &mut self,
        name: impl Into<String>,
        view: TensorView,
    ) -> Result<usize, Gemma4ExecutionLayoutError> {
        self.push_tensor(name, view, Gemma4TensorBacking::Workspace)
    }

    fn alias(
        &mut self,
        name: impl Into<String>,
        source: usize,
        view: TensorView,
    ) -> Result<usize, Gemma4ExecutionLayoutError> {
        let source_tensor = self.tensor(source)?;
        if source_tensor.view.dtype() != view.dtype()
            || source_tensor.view.encoding() != view.encoding()
            || view.end_offset() > source_tensor.view.payload_bytes()
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "alias exceeds or changes its backing tensor",
            ));
        }
        self.push_tensor(name, view, Gemma4TensorBacking::Alias { tensor_id: source })
    }

    fn push_tensor(
        &mut self,
        name: impl Into<String>,
        view: TensorView,
        backing: Gemma4TensorBacking,
    ) -> Result<usize, Gemma4ExecutionLayoutError> {
        if view.payload_bytes() == 0 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "zero-byte tensor is not executable",
            ));
        }
        let id = self.tensors.len();
        match backing {
            Gemma4TensorBacking::Workspace => {
                self.workspace_bytes = self
                    .workspace_bytes
                    .checked_add(view.payload_bytes())
                    .ok_or_else(|| {
                        Gemma4ExecutionLayoutError::invalid("workspace bytes overflow")
                    })?;
            }
            Gemma4TensorBacking::RequestKv { .. } => {
                self.request_state_bytes = self
                    .request_state_bytes
                    .checked_add(view.payload_bytes())
                    .ok_or_else(|| {
                        Gemma4ExecutionLayoutError::invalid("request-state bytes overflow")
                    })?;
            }
            _ => {}
        }
        self.tensors.push(Gemma4ExecutionTensor {
            id,
            name: name.into(),
            view,
            backing,
        });
        Ok(id)
    }

    fn tensor(&self, id: usize) -> Result<&Gemma4ExecutionTensor, Gemma4ExecutionLayoutError> {
        self.tensors
            .get(id)
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("tensor id is absent"))
    }

    fn views(&self, ids: &[usize]) -> Result<Vec<TensorView>, Gemma4ExecutionLayoutError> {
        ids.iter()
            .map(|id| Ok(self.tensor(*id)?.view.clone()))
            .collect()
    }
}

fn nvfp4_tensor_view(shape: &[u64]) -> Result<TensorView, Gemma4ExecutionLayoutError> {
    let shape = shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension).map_err(|_| {
                Gemma4ExecutionLayoutError::invalid("NVFP4 tensor extent does not fit usize")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    TensorView::with_encoding(
        DType::U8,
        Encoding::Nvfp4 {
            block_size: 16,
            scale_dtype: DType::F8E4M3Fn,
        },
        &shape,
    )
    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))
}

fn quantized_gemma_tensor_view(
    descriptor: &crate::QuantizedTensorDescriptor,
) -> Result<TensorView, Gemma4ExecutionLayoutError> {
    let shape = descriptor
        .logical_shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension).map_err(|_| {
                Gemma4ExecutionLayoutError::invalid("quantized tensor extent does not fit usize")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (dtype, encoding) = match descriptor.encoding {
        QuantizedTensorEncoding::UnquantizedBf16 => (DType::Bf16, Encoding::Unquantized),
        QuantizedTensorEncoding::OcpFp8E4M3FnChannelBf16Scale => (
            DType::F8E4M3Fn,
            Encoding::Fp8Scaled {
                granularity: crate::Fp8ScaleGranularity::OuterDimension,
                scale_dtype: DType::F32,
                resident: crate::Fp8ResidentRepresentation::PackedBytes,
            },
        ),
        QuantizedTensorEncoding::Nvfp4E2M1Block16E4M3FnF32Outer => (
            DType::U8,
            Encoding::Nvfp4W4A4 {
                block_size: 16,
                scale_dtype: DType::F8E4M3Fn,
            },
        ),
        QuantizedTensorEncoding::Mxfp4E2M1Block32E8M0
        | QuantizedTensorEncoding::Mxfp8E4M3Block32E8M0 => {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "MX encoding is not part of the reviewed Gemma recipe",
            ));
        }
    };
    TensorView::with_encoding(dtype, encoding, &shape)
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))
}

fn tensor_view(
    dtype: TensorDType,
    shape: &[u64],
    byte_offset: u64,
) -> Result<TensorView, Gemma4ExecutionLayoutError> {
    let dtype = match dtype {
        TensorDType::Bf16 => DType::Bf16,
        TensorDType::F16 => DType::F16,
        TensorDType::F32 => DType::F32,
        TensorDType::I32 => DType::I32,
        TensorDType::I64 | TensorDType::U8 => {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "reviewed Gemma execution tensor has an unsupported dtype",
            ));
        }
    };
    let shape = shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension).map_err(|_| {
                Gemma4ExecutionLayoutError::invalid("tensor extent does not fit usize")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut strides = vec![0; shape.len()];
    let mut stride = 1usize;
    for (dimension, output) in shape.iter().zip(strides.iter_mut()).rev() {
        *output = stride;
        stride = stride
            .checked_mul(*dimension)
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("tensor stride overflow"))?;
    }
    TensorView::new(dtype, Encoding::Unquantized, &shape, &strides, byte_offset)
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))
}

fn contiguous(dtype: DType, shape: &[u64]) -> Result<TensorView, Gemma4ExecutionLayoutError> {
    let shape = shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension).map_err(|_| {
                Gemma4ExecutionLayoutError::invalid("tensor extent does not fit usize")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    contiguous_usize(dtype, &shape)
}

fn contiguous_usize(
    dtype: DType,
    shape: &[usize],
) -> Result<TensorView, Gemma4ExecutionLayoutError> {
    TensorView::contiguous(dtype, shape)
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        if bits & 0x007f_ffff != 0 {
            return ((bits >> 16) as u16 & 0x803f) | 0x7fc0;
        }
        return (bits >> 16) as u16;
    }
    let upper = bits >> 16;
    let lower = bits & 0xffff;
    (upper + u32::from(lower > 0x8000 || (lower == 0x8000 && upper & 1 != 0))) as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_gemma4_graph, build_gemma4_weight_load_plan, parse_gemma4_model_lock};

    fn fixture(
        token_count: u64,
        start_position: u64,
        capacity: u64,
    ) -> (Gemma4Graph, WeightLoadPlan) {
        let lock = parse_gemma4_model_lock(include_bytes!(
            "../../../docs/models/locks/gemma4-12b-bf16.json"
        ))
        .unwrap();
        let catalog = crate::gemma4::expected_gemma4_tensor_catalog().unwrap();
        let plan = build_gemma4_weight_load_plan(&lock, catalog.values()).unwrap();
        let graph =
            build_gemma4_graph(&lock, &plan, token_count, start_position, capacity).unwrap();
        (graph, plan)
    }

    fn synthetic_state_image(encoding: KvCacheEncoding) -> Gemma4StateImageV1 {
        let committed_length = 1;
        let state_capacity = 2;
        let mut full_kv_layers = BTreeMap::new();
        let mut sliding_layers = BTreeMap::new();
        for (layer, kind) in crate::reviewed_layer_schedule().into_iter().enumerate() {
            let layer = u32::try_from(layer).unwrap();
            let metadata = StateLayerMetadataV1 {
                owner: StateOwnerKindV1::Kv,
                layer_id: layer,
                published_length: committed_length,
                generation: u64::from(layer) + 1,
                active_slot: None,
            };
            let planes = gemma_kv_plane_kinds(encoding)
                .iter()
                .copied()
                .map(|plane| OpaqueStatePlane {
                    owner: StateOwnerKindV1::Kv,
                    layer_id: layer,
                    plane,
                    bytes: if kind == Gemma4LayerType::SlidingAttention
                        && !matches!(plane, StatePlaneKindV1::KvKey | StatePlaneKindV1::KvValue)
                    {
                        Vec::new()
                    } else {
                        vec![layer as u8, plane as u8]
                    },
                })
                .collect();
            let image = ExecutionStateImageV1::new(metadata, planes);
            match kind {
                Gemma4LayerType::FullAttention => {
                    let descriptor =
                        KvStateDescriptor::new_with_storage(layer, state_capacity, 1, 1, encoding)
                            .unwrap();
                    full_kv_layers.insert(layer, Gemma4KvStateImageV1 { descriptor, image });
                }
                Gemma4LayerType::SlidingAttention => {
                    sliding_layers.insert(
                        layer,
                        Gemma4SlidingStateImageV1 {
                            heads: 1,
                            head_dim: 1,
                            capacity: state_capacity,
                            retention_window: 128,
                            image,
                        },
                    );
                }
            }
        }
        Gemma4StateImageV1 {
            session_id: crate::ExecutionSessionId::new(91),
            model_fingerprint: format!("sha256:{}", "1".repeat(64)),
            plan_digest: [2; 32],
            state_capacity,
            committed_length,
            rope_position_delta: 4,
            full_kv_layers,
            sliding_layers,
            cached_terminal_output: None,
        }
    }

    fn synthetic_checkpoint_identity(
        image: &Gemma4StateImageV1,
        tokens: &[u32],
    ) -> CheckpointIdentity {
        CheckpointIdentity::for_tokens(
            image.model_fingerprint(),
            "derived",
            "adapter",
            "renderer",
            "tokenizer",
            "gfx942:wave64",
            gemma_hex_digest(image.plan_digest()),
            tokens,
            image.kv_encoding().unwrap(),
            image.kv_descriptor_digest().unwrap(),
            [3; 32],
        )
        .unwrap()
    }

    #[test]
    fn layout_materializes_every_semantic_descriptor_and_exact_weight_identity() {
        let (graph, plan) = fixture(3, 0, 17);
        let layout = build_gemma4_execution_layout(&graph, &plan).unwrap();
        assert_eq!(layout.nodes().len(), graph.nodes().len());
        assert_eq!(layout.model_weight_bytes(), plan.total_destination_bytes);
        assert!(layout.workspace_bytes() > 0);
        assert!(layout.request_state_bytes() > 0);
        assert!(layout.nodes().iter().enumerate().all(|(index, node)| {
            node.graph_node_id() == index
                && node.descriptor().kind() == graph.nodes()[index].kind().semantic_kind().unwrap()
        }));
        let tied = layout
            .tensors()
            .iter()
            .filter(|tensor| {
                matches!(
                    tensor.backing(),
                    Gemma4TensorBacking::ModelWeight { tensor_name }
                        if tensor_name == "model.language_model.embed_tokens.weight"
                )
            })
            .count();
        assert_eq!(tied, 1);
    }

    #[test]
    fn embedding_layout_keeps_final_norm_output_rows_for_boundaries() {
        for token_count in [1_u64, 3, 17] {
            let (graph, plan) = fixture(token_count, 0, token_count);
            let layout = build_gemma4_execution_layout(&graph, &plan).unwrap();
            let final_nodes = graph
                .nodes()
                .iter()
                .filter(|node| {
                    matches!(
                        node.kind(),
                        Gemma4GraphNodeKind::RmsNorm {
                            role: Gemma4NormRole::Final,
                            ..
                        }
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(final_nodes.len(), 1);
            assert_eq!(final_nodes[0].label(), "final_norm");
            let execution_node = &layout.nodes()[final_nodes[0].id()];
            assert_eq!(execution_node.descriptor().kind(), SemanticOpKind::RmsNorm);
            assert_eq!(execution_node.outputs().len(), 1);
            let output = &layout.tensors()[execution_node.outputs()[0]];
            assert_eq!(
                output.view().shape(),
                [token_count as usize, GEMMA4_HIDDEN_SIZE as usize]
            );
            assert_eq!(output.view().dtype(), DType::Bf16);
            assert_eq!(output.view().encoding(), Encoding::Unquantized);
            assert!(output.view().is_contiguous());
        }
    }

    #[test]
    fn decode_kv_tail_and_attention_prefix_share_checked_backing() {
        let (graph, plan) = fixture(3, 17, 257);
        let layout = build_gemma4_execution_layout(&graph, &plan).unwrap();
        let attention = layout
            .nodes()
            .iter()
            .find(|node| node.descriptor().kind() == SemanticOpKind::CausalAttention)
            .unwrap();
        assert_eq!(attention.kv_appends().len(), 2);
        for append in attention.kv_appends() {
            let state = &layout.tensors()[append.state_tensor()];
            let tail = &layout.tensors()[append.destination_view()];
            assert!(
                matches!(tail.backing(), Gemma4TensorBacking::Alias { tensor_id } if *tensor_id == state.id())
            );
            assert!(tail.view().byte_offset() > 0);
            assert!(tail.view().end_offset() <= state.view().payload_bytes());
        }
        let contract = attention.descriptor().causal_attention_contract().unwrap();
        assert_eq!(contract.start_position(), 17);
        assert_eq!(contract.query_count(), 3);
        assert_eq!(contract.expected_kv_length(), 20);
        assert_eq!(attention.descriptor().inputs()[1].shape()[0], 20);
        assert_eq!(attention.descriptor().inputs()[2].shape()[0], 20);
    }

    #[test]
    fn layout_rejects_mismatched_graph_and_plan_identity() {
        let (graph, mut plan) = fixture(1, 0, 1);
        plan.lock_fingerprint.push_str("-different");
        assert!(build_gemma4_execution_layout(&graph, &plan).is_err());
    }

    #[test]
    fn full_attention_states_use_opaque_fp16_geometry_only() {
        let (graph, _) = fixture(1, 0, 17);
        let full = graph
            .kv_descriptors()
            .iter()
            .filter(|descriptor| descriptor.retention_window.is_none())
            .collect::<Vec<_>>();
        assert_eq!(full.len(), 8);
        assert_eq!(gemma_full_attention_layers(&graph).len(), full.len());
        for descriptor in full {
            let state = gemma_full_kv_state_descriptor(descriptor, None).unwrap();
            assert_eq!(state.cache_encoding(), KvCacheEncoding::Fp16);
            assert_eq!(state.layout().heads(), descriptor.heads as usize);
            assert_eq!(state.layout().head_dim(), descriptor.head_dim as usize);
            assert_eq!(state.capacity(), descriptor.capacity);
        }
    }

    #[test]
    fn sliding_prefix_plane_ranges_are_exact_for_non_aligned_window_tail() {
        let (graph, _) = fixture(1, 0, 257);
        let sliding = gemma_sliding_layer_identities(&graph);
        assert!(!sliding.is_empty());
        assert!(sliding.values().all(|identity| identity.capacity == 257));
        for identity in sliding.values().copied() {
            let bytes = gemma_sliding_plane_bytes(identity, 65).unwrap();
            assert_eq!(
                bytes,
                65 * u64::from(identity.heads) * u64::from(identity.head_dim) * 2
            );
            assert!(bytes < gemma_sliding_plane_bytes(identity, identity.capacity).unwrap());
        }
    }

    #[test]
    fn prefix_fork_audit_charges_shared_physical_and_sliding_resident_bytes() {
        let mut audit = Gemma4PrefixForkAuditV1::default();
        let shared =
            StateForkAuditV1::new(crate::StateForkModeV1::SharedReadOnlyPages, 257, 2, 0, 0)
                .unwrap();
        audit.add(shared, 8192).unwrap();
        assert_eq!(audit.shared_pages(), 2);
        assert_eq!(audit.destination_owned_bytes(), 0);
        assert_eq!(audit.cache_resident_bytes(), 8192);
    }

    #[test]
    fn state_image_checkpoint_round_trip_covers_the_real_layer_topology() {
        let image = synthetic_state_image(KvCacheEncoding::Fp16);
        assert_eq!(image.full_kv_layers().len(), 8);
        assert_eq!(image.sliding_layers().len(), 40);
        let tokens = [7];
        let identity = synthetic_checkpoint_identity(&image, &tokens);
        let checkpoint = image
            .to_checkpoint(
                identity,
                &tokens,
                b"conversation",
                b"sampler",
                b"grammar",
                b"stop",
                5,
                1,
                9,
            )
            .unwrap();
        assert_eq!(checkpoint.payload.state_layers.len(), 48);
        assert_eq!(checkpoint.payload.state_planes.len(), 96);
        assert_eq!(checkpoint.payload.token_history, tokens);
        assert_eq!(checkpoint.payload.conversation, b"conversation");
        assert_eq!(checkpoint.header.absolute_position, 5);
        assert_eq!(checkpoint.header.logical_position, 1);
        let encoded = checkpoint.encode().unwrap();
        assert_eq!(SessionCheckpoint::decode(&encoded).unwrap(), checkpoint);
    }

    #[test]
    fn state_image_keeps_every_encoding_native_plane_generic() {
        let image = synthetic_state_image(KvCacheEncoding::Nvfp4);
        let tokens = [11];
        let checkpoint = image
            .to_checkpoint(
                synthetic_checkpoint_identity(&image, &tokens),
                &tokens,
                &[],
                &[],
                &[],
                &[],
                5,
                1,
                1,
            )
            .unwrap();
        assert_eq!(checkpoint.payload.state_planes.len(), 48 * 6);
        assert!(image.sliding_layers().values().all(|layer| {
            layer.image().planes().iter().all(|plane| {
                matches!(
                    plane.plane,
                    StatePlaneKindV1::KvKey | StatePlaneKindV1::KvValue
                ) || plane.bytes.is_empty()
            })
        }));
    }

    #[test]
    fn state_image_rejects_missing_duplicate_and_wrong_identity() {
        let mut missing = synthetic_state_image(KvCacheEncoding::Fp16);
        let removed = *missing.full_kv_layers.keys().next().unwrap();
        missing.full_kv_layers.remove(&removed);
        assert!(validate_gemma_state_image(&missing, false).is_err());

        let mut duplicate = synthetic_state_image(KvCacheEncoding::Fp16);
        let layer = *duplicate.full_kv_layers.keys().next().unwrap();
        let duplicate_plane = duplicate.full_kv_layers[&layer].image.planes()[0].clone();
        let entry = duplicate.full_kv_layers.get_mut(&layer).unwrap();
        let mut planes = entry.image.planes().to_vec();
        planes.push(duplicate_plane);
        entry.image = ExecutionStateImageV1::new(entry.image.metadata().clone(), planes);
        assert!(validate_gemma_state_image(&duplicate, false).is_err());

        let image = synthetic_state_image(KvCacheEncoding::Fp16);
        let tokens = [19];
        let mut identity = synthetic_checkpoint_identity(&image, &tokens);
        identity.target_semantics = "gfx1100:wave32".to_owned();
        identity.kv_descriptor_digest[0] ^= 0xff;
        assert!(
            image
                .to_checkpoint(identity, &tokens, &[], &[], &[], &[], 5, 1, 1)
                .is_err()
        );
    }

    #[test]
    fn descriptor_digest_commits_sliding_layout_and_retention() {
        let image = synthetic_state_image(KvCacheEncoding::Fp16);
        let baseline = image.kv_descriptor_digest().unwrap();
        let mut changed = image.clone();
        changed
            .sliding_layers
            .values_mut()
            .next()
            .unwrap()
            .retention_window += 1;
        assert_ne!(changed.kv_descriptor_digest().unwrap(), baseline);
    }
}
