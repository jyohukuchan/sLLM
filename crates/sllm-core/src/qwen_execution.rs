//! Request-local execution ownership for the fixed Qwen3.5-4B text graph.
//!
//! This is a host-side orchestration layer. It owns the checked device
//! buffers, Stage C state objects, and completion ordering for one request;
//! it neither implements numerical operators nor offers a CPU fallback.
//! Every operation reaches a backend only through the existing owned
//! execution/session contracts.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::execution::{
    BoundSemanticOp, CausalAttentionSubmission, ExecutionBuffer, ExecutionError, ExecutionQueue,
    ExecutionSession, ExecutionState, KvState, KvStateAppendSubmission, LinearAttentionBindings,
    LinearAttentionState, LinearAttentionSubmission, OwnedTensorBinding, PrepareSupport,
    Submission,
};
use crate::final_output::QWEN35_VOCAB_SIZE;
use crate::kv_state::{CausalAttentionDescriptor, KvPhysicalMemorySnapshot, KvStateDescriptor};
use crate::linear_attention::{LinearAttentionDescriptor, LinearAttentionStateDescriptor};
use crate::model::{TensorDType, VerifiedCache};
use crate::op::{
    AttentionPreprocessContract, AttentionPreprocessPositionMode, OpError, SemanticOpDescriptor,
    SemanticOpKind,
};
use crate::qwen_graph::{
    QwenGraph, QwenGraphNode, QwenGraphNodeKind, QwenGraphStateDescriptor, QwenGraphTensorBacking,
    QwenGraphWeightBinding,
};
use crate::tensor::{TensorError, TensorView};
use crate::weights::{
    WeightClassification, WeightLoadEntry, WeightLoadPlan, WeightUploadError, WeightUploadReceipt,
    WeightUploadRequest, upload_verified_weight,
};
use crate::{AccessMode, DType, DispatchEvidence};

/// Output published by a fully completed Qwen request transition.
#[derive(Clone, Debug, PartialEq)]
pub struct QwenExecutionOutput {
    token_ids: Vec<i32>,
    last_logits: Option<Vec<f32>>,
    committed_length: u64,
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

    /// Creates a fresh request graph/state owner against this resident model.
    pub fn new_request(
        &self,
        graph: QwenGraph,
    ) -> Result<QwenExecutionRequest, QwenExecutionError> {
        QwenExecutionCore::from_resident(Arc::clone(&self.inner), graph).map(|core| {
            QwenExecutionRequest {
                _resident: Arc::clone(&self.inner),
                core,
            }
        })
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

    /// Runs prefill and publishes the final row of full-vocabulary logits in
    /// addition to the existing device Argmax output.
    pub fn prefill_with_last_logits(
        &mut self,
        token_ids: &[i32],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.core.prefill_with_last_logits(token_ids)
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

    pub const fn committed_length(&self) -> u64 {
        self.core.committed_length
    }

    pub fn last_output(&self) -> Option<&QwenExecutionOutput> {
        self.core.last_output.as_ref()
    }

    pub fn is_poisoned(&self) -> bool {
        self.core.lifecycle.poisoned.load(Ordering::Acquire)
    }

    /// Idempotently invalidates this request owner without affecting the
    /// resident model. Synchronous callers use it between transitions when a
    /// transport cancellation or host-side sampling/decoding failure occurs.
    pub fn cancel(&mut self) {
        self.core.lifecycle.poisoned.store(true, Ordering::Release);
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

    /// Returns the immutable audit accumulated by successful compute
    /// submissions. An empty audit is never a successful request audit.
    pub fn audit_snapshot(&self) -> Result<QwenExecutionAudit, QwenExecutionError> {
        self.core.audit_snapshot()
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
    plan: WeightLoadPlan,
    queue: ExecutionQueue,
    static_tensors: BTreeMap<String, TensorAllocation>,
    scales: BTreeMap<String, CachedScale>,
    completion_timeout: Duration,
}

struct QwenExecutionCore {
    session: Arc<ExecutionSession>,
    graph: QwenGraph,
    plan: WeightLoadPlan,
    queue: ExecutionQueue,
    tensors: Vec<TensorAllocation>,
    tensor_ids: BTreeMap<String, usize>,
    dynamic_tensors: Vec<bool>,
    kv_states: BTreeMap<u32, KvState>,
    linear_states: BTreeMap<u32, LinearAttentionState>,
    scales: BTreeMap<usize, CachedScale>,
    completion_timeout: Duration,
    audit: Mutex<DispatchAuditAccumulator>,
    lifecycle: Arc<RequestLifecycle>,
    committed_length: u64,
    last_output: Option<QwenExecutionOutput>,
}

#[derive(Debug, Default)]
struct DispatchAuditAccumulator {
    target: Option<String>,
    submission_count: u64,
    kernel_dispatch_count: u64,
    fallback_used: bool,
    all_dispatches_hip: bool,
}

impl DispatchAuditAccumulator {
    fn record_evidence(&mut self, evidence: &DispatchEvidence) -> Result<(), QwenExecutionError> {
        if evidence.backend != 1
            || evidence.fallback_allowed
            || evidence.fallback_used
            || evidence.dispatch_count == 0
            || evidence.target.is_empty()
            || !evidence.target.is_ascii()
            || evidence.target.as_bytes().contains(&0)
        {
            return Err(QwenExecutionError::InvalidRequest(
                "accepted dispatch evidence is not an exact HIP, no-fallback dispatch".to_owned(),
            ));
        }
        if let Some(target) = &self.target {
            if target != &evidence.target {
                return Err(QwenExecutionError::InvalidRequest(
                    "dispatch evidence targets differ within one Qwen request".to_owned(),
                ));
            }
        } else {
            self.target = Some(evidence.target.clone());
        }
        self.submission_count = self.submission_count.checked_add(1).ok_or_else(|| {
            QwenExecutionError::InvalidRequest("dispatch submission count overflowed".to_owned())
        })?;
        self.kernel_dispatch_count = self
            .kernel_dispatch_count
            .checked_add(u64::from(evidence.dispatch_count))
            .ok_or_else(|| {
                QwenExecutionError::InvalidRequest("kernel dispatch count overflowed".to_owned())
            })?;
        self.all_dispatches_hip |= evidence.backend == 1;
        self.fallback_used |= evidence.fallback_used;
        Ok(())
    }
}

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
}

type StateMaps = (BTreeMap<u32, KvState>, BTreeMap<u32, LinearAttentionState>);

struct RequestLifecycle {
    poisoned: AtomicBool,
    in_flight: AtomicBool,
}

impl RequestLifecycle {
    fn new() -> Self {
        Self {
            poisoned: AtomicBool::new(false),
            in_flight: AtomicBool::new(false),
        }
    }
}

/// Marks the request unusable unless output publication disarms it. This is
/// deliberately separate from Stage C completion owners: they release their
/// own backend admission before this guard makes a graph-wide request usable.
struct TransitionGuard {
    lifecycle: Arc<RequestLifecycle>,
    published: bool,
}

impl TransitionGuard {
    fn begin(lifecycle: Arc<RequestLifecycle>) -> Result<Self, QwenExecutionError> {
        if lifecycle.poisoned.load(Ordering::Acquire) {
            return Err(QwenExecutionError::Poisoned);
        }
        if lifecycle
            .in_flight
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(QwenExecutionError::Busy);
        }
        if lifecycle.poisoned.load(Ordering::Acquire) {
            lifecycle.in_flight.store(false, Ordering::Release);
            return Err(QwenExecutionError::Poisoned);
        }
        Ok(Self {
            lifecycle,
            published: false,
        })
    }

    fn publish(&mut self) {
        self.published = true;
        self.lifecycle.in_flight.store(false, Ordering::Release);
    }
}

impl Drop for TransitionGuard {
    fn drop(&mut self) {
        if !self.published {
            self.lifecycle.poisoned.store(true, Ordering::Release);
        }
        self.lifecycle.in_flight.store(false, Ordering::Release);
    }
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

    fn read_scale_bytes(
        &self,
        tensor_name: &str,
        expected_length: usize,
    ) -> Result<Arc<[u8]>, QwenExecutionError>;
}

struct VerifiedProvisionSource {
    cache: Arc<VerifiedCache>,
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

impl QwenResidentInner {
    fn provision<S: QwenProvisionSource>(
        session: Arc<ExecutionSession>,
        graph: QwenGraph,
        plan: WeightLoadPlan,
        completion_timeout: Duration,
        source: &S,
    ) -> Result<Self, QwenExecutionError> {
        let layout = validate_graph_plan(&graph, &plan)?;
        preflight_device_memory(session.as_ref(), &graph, &plan)?;
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
                allocation.graph_view.payload_bytes(),
            )?;
            source.upload_weight(
                &plan,
                binding,
                session.as_ref(),
                &queue,
                destination,
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
    ) -> Result<Self, QwenExecutionError> {
        let layout = validate_graph_plan(&graph, &resident.plan)?;
        if graph.model_fingerprint() != resident.model_fingerprint {
            return Err(QwenExecutionError::InvalidRequest(
                "resident model identity or backend differs from the request graph".to_owned(),
            ));
        }
        validate_resident_graph(&graph, &layout, &resident.static_tensors, &resident.scales)?;
        preflight_device_memory(resident.session.as_ref(), &graph, &resident.plan)?;
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
        let core = Self {
            session: Arc::clone(&resident.session),
            graph,
            plan: resident.plan.clone(),
            queue: resident.queue.clone(),
            tensors,
            tensor_ids: layout.tensor_ids,
            dynamic_tensors: layout.dynamic_tensors,
            kv_states,
            linear_states,
            scales,
            completion_timeout: resident.completion_timeout,
            audit: Mutex::new(DispatchAuditAccumulator::default()),
            lifecycle: Arc::new(RequestLifecycle::new()),
            committed_length: 0,
            last_output: None,
        };
        core.ensure_state_lengths(0)?;
        Ok(core)
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
        preflight_device_memory(session.as_ref(), &graph, &plan)?;
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
                allocation.graph_view.payload_bytes(),
            )?;
            source.upload_weight(
                &plan,
                binding,
                session.as_ref(),
                &queue,
                destination,
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

        let core = Self {
            session,
            graph,
            plan,
            queue,
            tensors,
            tensor_ids: layout.tensor_ids,
            dynamic_tensors: layout.dynamic_tensors,
            kv_states,
            linear_states,
            scales,
            completion_timeout,
            audit: Mutex::new(DispatchAuditAccumulator::default()),
            lifecycle: Arc::new(RequestLifecycle::new()),
            committed_length: 0,
            last_output: None,
        };
        core.ensure_state_lengths(0)?;
        Ok(core)
    }

    fn prefill(&mut self, token_ids: &[i32]) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.prefill_impl(token_ids, false)
    }

    fn prefill_with_last_logits(
        &mut self,
        token_ids: &[i32],
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.prefill_impl(token_ids, true)
    }

    fn prefill_impl(
        &mut self,
        token_ids: &[i32],
        include_last_logits: bool,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        if self.lifecycle.poisoned.load(Ordering::Acquire) {
            return Err(QwenExecutionError::Poisoned);
        }
        if self.committed_length != 0 {
            return Err(QwenExecutionError::InvalidRequest(
                "prefill is only valid before the first committed transition".to_owned(),
            ));
        }
        let expected = usize::try_from(self.graph.token_count()).map_err(|_| {
            QwenExecutionError::InvalidGraph("graph token count does not fit usize".to_owned())
        })?;
        if token_ids.len() != expected {
            return Err(QwenExecutionError::InvalidRequest(format!(
                "prefill token count is {}, expected {expected}",
                token_ids.len()
            )));
        }
        self.run_transition(
            token_ids,
            AttentionPreprocessPositionMode::Prefill,
            include_last_logits,
        )
    }

    fn decode(&mut self, token_id: i32) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.decode_impl(token_id, false)
    }

    fn decode_with_last_logits(
        &mut self,
        token_id: i32,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        self.decode_impl(token_id, true)
    }

    fn decode_impl(
        &mut self,
        token_id: i32,
        include_last_logits: bool,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
        if self.lifecycle.poisoned.load(Ordering::Acquire) {
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
        )
    }

    fn run_transition(
        &mut self,
        token_ids: &[i32],
        position_mode: AttentionPreprocessPositionMode,
        include_last_logits: bool,
    ) -> Result<QwenExecutionOutput, QwenExecutionError> {
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
        if expected_length > u64::from(AttentionPreprocessContract::MAX_POSITION_EMBEDDINGS) {
            return Err(QwenExecutionError::InvalidRequest(
                "transition exceeds the attention position contract".to_owned(),
            ));
        }
        validate_input_token_ids(token_ids)?;

        // A stale state before any new upload/dispatch is an admission error,
        // not a graph-wide partial mutation. Once the guard begins, every
        // error path poisons the request.
        self.ensure_state_lengths(start_position)?;
        let mut guard = TransitionGuard::begin(Arc::clone(&self.lifecycle))?;

        self.upload_runtime_inputs(token_ids, start_position, token_count)?;
        let output =
            self.lower_graph(token_count, start_position, expected_length, position_mode)?;
        let last_logits = include_last_logits
            .then(|| self.read_last_logits(token_count))
            .transpose()?;
        self.ensure_state_lengths(expected_length)?;

        let output = QwenExecutionOutput {
            token_ids: output,
            last_logits,
            committed_length: expected_length,
        };
        self.committed_length = expected_length;
        self.last_output = Some(output.clone());
        guard.publish();
        Ok(output)
    }

    fn lower_graph(
        &self,
        token_count: u64,
        start_position: u64,
        expected_length: u64,
        position_mode: AttentionPreprocessPositionMode,
    ) -> Result<Vec<i32>, QwenExecutionError> {
        let nodes = self.graph.nodes().to_vec();
        let mut argmax = None;
        for node in &nodes {
            match node.kind() {
                QwenGraphNodeKind::Semantic(_) => {
                    let output = self.execute_semantic(node, token_count)?;
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
                        token_count,
                        start_position,
                        position_mode,
                    },
                )?,
                QwenGraphNodeKind::FullKvAppend { layer, state } => {
                    self.execute_kv_append(node, layer, state, token_count, start_position)?
                }
                QwenGraphNodeKind::FullCausalAttention { layer, state, .. } => self
                    .execute_causal_attention(
                        node,
                        layer,
                        state,
                        token_count,
                        start_position,
                        expected_length,
                    )?,
                QwenGraphNodeKind::LinearAttentionState { layer, state, .. } => self
                    .execute_linear_attention(
                        node,
                        layer,
                        state,
                        token_count,
                        start_position,
                        expected_length,
                    )?,
            }
        }
        argmax.ok_or_else(|| {
            QwenExecutionError::InvalidGraph("graph has no argmax output node".to_owned())
        })
    }

    fn read_last_logits(&self, token_count: u64) -> Result<Vec<f32>, QwenExecutionError> {
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
        if view.dtype() != DType::Bf16
            || view.shape()
                != [
                    usize::try_from(token_count).map_err(|_| {
                        QwenExecutionError::InvalidRequest(
                            "token count does not fit usize".to_owned(),
                        )
                    })?,
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
        let row_index = token_count.checked_sub(1).ok_or_else(|| {
            QwenExecutionError::InvalidRequest("cannot read logits for zero tokens".to_owned())
        })?;
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

    fn execute_semantic(
        &self,
        node: &QwenGraphNode,
        token_count: u64,
    ) -> Result<Option<Vec<i32>>, QwenExecutionError> {
        let operation = node.operation().ok_or_else(|| {
            QwenExecutionError::InvalidGraph(format!(
                "semantic node {} has no descriptor",
                node.label()
            ))
        })?;
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
            kind => SemanticOpDescriptor::new(kind, inputs, outputs)?,
        };
        let input_bindings = self.bind_many(node.inputs(), token_count, AccessMode::Read)?;
        let output_bindings = self.bind_many(node.outputs(), token_count, AccessMode::Write)?;
        let kind = descriptor.kind();
        let mut submission =
            self.submit_semantic(node.label(), descriptor, input_bindings, output_bindings)?;
        wait_submission_success(&mut submission, self.completion_timeout, node.label())?;
        self.record_dispatch(submission.dispatch())?;
        if kind != SemanticOpKind::Argmax {
            return Ok(None);
        }
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
        Ok(Some(decode_argmax_bytes(&bytes)?))
    }

    fn execute_attention_preprocess(
        &self,
        node: &QwenGraphNode,
        execution: AttentionPreprocessExecution,
    ) -> Result<(), QwenExecutionError> {
        if node.inputs().len() != 5 || node.outputs().len() != 3 {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "attention preprocess node {} has the wrong binding arity",
                node.label()
            )));
        }
        let contract = AttentionPreprocessContract::new_qwen3_5_with_layout(
            execution.position_mode,
            i64::try_from(execution.start_position).map_err(|_| {
                QwenExecutionError::InvalidRequest("position does not fit i64".to_owned())
            })?,
            execution.token_count,
            execution.q_heads,
            execution.kv_heads,
            execution.head_dim,
        )?;
        let descriptor = SemanticOpDescriptor::new_attention_preprocess(
            self.views(node.inputs(), execution.token_count)?,
            self.views(node.outputs(), execution.token_count)?,
            contract,
        )?;
        let inputs = self.bind_many(node.inputs(), execution.token_count, AccessMode::Read)?;
        let outputs = self.bind_many(node.outputs(), execution.token_count, AccessMode::Write)?;
        let mut submission = self.submit_semantic(node.label(), descriptor, inputs, outputs)?;
        wait_submission_success(&mut submission, self.completion_timeout, node.label())?;
        self.record_dispatch(submission.dispatch())?;
        let _ = execution.layer;
        Ok(())
    }

    fn execute_kv_append(
        &self,
        node: &QwenGraphNode,
        layer: u32,
        descriptor: KvStateDescriptor,
        token_count: u64,
        start_position: u64,
    ) -> Result<(), QwenExecutionError> {
        if node.inputs().len() != 2 || !node.outputs().is_empty() {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "KV append node {} has the wrong binding arity",
                node.label()
            )));
        }
        let state = self.kv_state(layer, descriptor)?;
        let key = self.bind(node.inputs()[0], token_count, AccessMode::Read)?;
        let value = self.bind(node.inputs()[1], token_count, AccessMode::Read)?;
        let mut submission = self.session.append_kv_state(
            state,
            &self.queue,
            key,
            value,
            start_position,
            start_position,
        )?;
        wait_kv_append_success(&mut submission, self.completion_timeout, node.label())?;
        self.record_dispatch(submission.dispatch())
    }

    fn execute_causal_attention(
        &self,
        node: &QwenGraphNode,
        layer: u32,
        state_descriptor: KvStateDescriptor,
        token_count: u64,
        start_position: u64,
        expected_length: u64,
    ) -> Result<(), QwenExecutionError> {
        if node.inputs().len() != 1 || node.outputs().len() != 1 {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "causal attention node {} has the wrong binding arity",
                node.label()
            )));
        }
        let state = self.kv_state(layer, state_descriptor)?;
        let descriptor =
            CausalAttentionDescriptor::new(start_position, token_count, expected_length)
                .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
        let query = self.bind(node.inputs()[0], token_count, AccessMode::Read)?;
        let output = self.bind(node.outputs()[0], token_count, AccessMode::Write)?;
        let mut submission =
            self.session
                .causal_attention(state, &self.queue, query, output, descriptor)?;
        wait_causal_attention_success(&mut submission, self.completion_timeout, node.label())?;
        self.record_dispatch(submission.dispatch())
    }

    fn execute_linear_attention(
        &self,
        node: &QwenGraphNode,
        layer: u32,
        state_descriptor: LinearAttentionStateDescriptor,
        token_count: u64,
        start_position: u64,
        expected_length: u64,
    ) -> Result<(), QwenExecutionError> {
        if node.inputs().len() != 8 || node.outputs().len() != 1 {
            return Err(QwenExecutionError::InvalidGraph(format!(
                "linear-attention node {} has the wrong binding arity",
                node.label()
            )));
        }
        let state = self.linear_state(layer, state_descriptor)?;
        let inputs = self.bind_many(node.inputs(), token_count, AccessMode::Read)?;
        let output = self.bind(node.outputs()[0], token_count, AccessMode::Write)?;
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
        let descriptor =
            LinearAttentionDescriptor::new(start_position, token_count, expected_length)
                .map_err(|error| QwenExecutionError::InvalidRequest(error.to_string()))?;
        let mut submission =
            self.session
                .linear_attention(state, &self.queue, bindings, descriptor)?;
        wait_linear_attention_success(&mut submission, self.completion_timeout, node.label())?;
        self.record_dispatch(submission.dispatch())
    }

    fn record_dispatch(&self, evidence: &DispatchEvidence) -> Result<(), QwenExecutionError> {
        self.audit
            .lock()
            .map_err(|_| QwenExecutionError::Poisoned)?
            .record_evidence(evidence)
    }

    fn audit_snapshot(&self) -> Result<QwenExecutionAudit, QwenExecutionError> {
        let audit = self
            .audit
            .lock()
            .map_err(|_| QwenExecutionError::Poisoned)?;
        let target = audit.target.clone().ok_or_else(|| {
            QwenExecutionError::InvalidRequest(
                "successful Qwen transition has an empty dispatch audit".to_owned(),
            )
        })?;
        if audit.submission_count == 0 || audit.kernel_dispatch_count == 0 {
            return Err(QwenExecutionError::InvalidRequest(
                "successful Qwen transition has an empty dispatch audit".to_owned(),
            ));
        }
        if !audit.all_dispatches_hip || audit.fallback_used {
            return Err(QwenExecutionError::InvalidRequest(
                "Qwen dispatch audit is not HIP-only and fallback-free".to_owned(),
            ));
        }
        Ok(QwenExecutionAudit {
            selected_backend: if audit.all_dispatches_hip { "hip" } else { "" },
            target,
            submission_count: audit.submission_count,
            kernel_dispatch_count: audit.kernel_dispatch_count,
            fallback_used: audit.fallback_used,
            all_dispatches_hip: audit.all_dispatches_hip,
        })
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
        label: &str,
        descriptor: SemanticOpDescriptor,
        inputs: Vec<OwnedTensorBinding>,
        outputs: Vec<OwnedTensorBinding>,
    ) -> Result<Submission, QwenExecutionError> {
        match self.session.supports(&descriptor) {
            PrepareSupport::Supported => {}
            PrepareSupport::Unsupported { reason } => {
                return Err(QwenExecutionError::Execution(ExecutionError::Unsupported {
                    reason: format!("{} is unsupported: {reason}", label),
                }));
            }
        }
        let operation = Arc::new(BoundSemanticOp::new(Arc::new(descriptor), inputs, outputs)?);
        let prepared = self.session.prepare(operation)?;
        Ok(self.session.submit(&prepared, &self.queue)?)
    }

    fn upload_runtime_inputs(
        &self,
        token_ids: &[i32],
        start_position: u64,
        token_count: u64,
    ) -> Result<(), QwenExecutionError> {
        let token_tensor = self.tensor_id("input.token_ids")?;
        let position_tensor = self.tensor_id("input.positions")?;
        let token_view = self.view(token_tensor, token_count)?;
        let position_view = self.view(position_tensor, token_count)?;
        let token_bytes = i32_bytes(token_ids);
        let positions = position_bytes(start_position, token_count)?;
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
        )
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

fn preflight_device_memory(
    session: &ExecutionSession,
    graph: &QwenGraph,
    plan: &WeightLoadPlan,
) -> Result<(), QwenExecutionError> {
    let owned_tensor_bytes = graph
        .tensor_metadata()
        .iter()
        .try_fold(0_u64, |total, tensor| match tensor.backing() {
            QwenGraphTensorBacking::Owned => total
                .checked_add(tensor.view().end_offset())
                .ok_or_else(|| {
                    QwenExecutionError::InvalidGraph(
                        "owned tensor allocation byte count overflowed".to_owned(),
                    )
                }),
            QwenGraphTensorBacking::Alias { .. } => Ok(total),
        })?;
    if plan.total_destination_bytes > owned_tensor_bytes {
        return Err(QwenExecutionError::InvalidGraph(format!(
            "model-resident bytes {} exceed owned tensor bytes {owned_tensor_bytes}",
            plan.total_destination_bytes
        )));
    }
    let required = owned_tensor_bytes
        .checked_add(graph.total_state_bytes())
        .ok_or_else(|| {
            QwenExecutionError::InvalidGraph(
                "model, workspace, and request-state byte count overflowed".to_owned(),
            )
        })?;
    let available = session.available_memory_bytes()?.ok_or_else(|| {
        QwenExecutionError::InvalidRequest(
            "backend did not report available device memory for placement preflight".to_owned(),
        )
    })?;
    if required > available {
        return Err(QwenExecutionError::InvalidRequest(format!(
            "device memory preflight requires {required} bytes (model-resident {}, owned tensor/workspace {owned_tensor_bytes}, request-state {}), but only {available} bytes are available",
            plan.total_destination_bytes,
            graph.total_state_bytes()
        )));
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
        if tensor.backing() != QwenGraphTensorBacking::Owned
            || tensor.view().dtype() != model_dtype(binding.dtype())?
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

    Ok(GraphLayout {
        tensor_ids,
        _weight_tensor_ids: weight_tensor_ids,
        dynamic_tensors,
        scales,
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
                tensor.view().end_offset(),
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
    let mut allocations: Vec<TensorAllocation> = Vec::with_capacity(graph.tensor_metadata().len());
    for tensor in graph.tensor_metadata() {
        let buffer = match tensor.backing() {
            QwenGraphTensorBacking::Owned => {
                if layout.dynamic_tensors[tensor.id()] {
                    session.allocate_with_category(
                        tensor.view().end_offset(),
                        crate::AllocationCategory::Workspace,
                    )?
                } else {
                    let resident = resident_tensors.get(tensor.name()).ok_or_else(|| {
                        QwenExecutionError::InvalidGraph(format!(
                            "resident tensor is absent: {}",
                            tensor.name()
                        ))
                    })?;
                    resident.buffer.clone()
                }
            }
            QwenGraphTensorBacking::Alias { tensor_id } => allocations
                .get(tensor_id)
                .ok_or_else(|| {
                    QwenExecutionError::InvalidGraph(format!(
                        "alias tensor {} precedes its backing tensor",
                        tensor.name()
                    ))
                })?
                .buffer
                .clone(),
        };
        allocations.push(TensorAllocation {
            buffer,
            graph_view: tensor.view().clone(),
        });
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
    let mut allocations: Vec<TensorAllocation> = Vec::with_capacity(graph.tensor_metadata().len());
    for tensor in graph.tensor_metadata() {
        let buffer = match tensor.backing() {
            QwenGraphTensorBacking::Owned => session.allocate(tensor.view().end_offset())?,
            QwenGraphTensorBacking::Alias { tensor_id } => {
                let source = allocations.get(tensor_id).ok_or_else(|| {
                    QwenExecutionError::InvalidGraph(format!(
                        "alias tensor {} precedes its backing tensor",
                        tensor.name()
                    ))
                })?;
                if source.graph_view.dtype() != tensor.view().dtype()
                    || source.graph_view.encoding() != tensor.view().encoding()
                    || source.graph_view.payload_bytes() != tensor.view().payload_bytes()
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
    let full_layers: BTreeSet<u32> = graph
        .layer_types()
        .iter()
        .enumerate()
        .filter_map(|(layer, ty)| (*ty == crate::LayerType::FullAttention).then_some(layer as u32))
        .collect();
    let linear_layers: BTreeSet<u32> = graph
        .layer_types()
        .iter()
        .enumerate()
        .filter_map(|(layer, ty)| {
            (*ty == crate::LayerType::LinearAttention).then_some(layer as u32)
        })
        .collect();
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

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(std::mem::size_of::<i32>()));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
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

fn require_terminal_success(stage: &str, state: ExecutionState) -> Result<(), QwenExecutionError> {
    match state {
        ExecutionState::Success => Ok(()),
        ExecutionState::Pending => Err(QwenExecutionError::CompletionPending {
            stage: stage.to_owned(),
        }),
        ExecutionState::Failure => Err(QwenExecutionError::CompletionFailure {
            stage: stage.to_owned(),
        }),
    }
}

fn wait_submission_success(
    submission: &mut Submission,
    timeout: Duration,
    stage: &str,
) -> Result<(), QwenExecutionError> {
    require_terminal_success(stage, submission.wait(timeout)?)
}

fn wait_kv_append_success(
    submission: &mut KvStateAppendSubmission,
    timeout: Duration,
    stage: &str,
) -> Result<(), QwenExecutionError> {
    require_terminal_success(stage, submission.wait(timeout)?)
}

fn wait_causal_attention_success(
    submission: &mut CausalAttentionSubmission,
    timeout: Duration,
    stage: &str,
) -> Result<(), QwenExecutionError> {
    require_terminal_success(stage, submission.wait(timeout)?)
}

fn wait_linear_attention_success(
    submission: &mut LinearAttentionSubmission,
    timeout: Duration,
    stage: &str,
) -> Result<(), QwenExecutionError> {
    require_terminal_success(stage, submission.wait(timeout)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    use crate::execution::{
        AdapterResource, ExecutionAdapterAccess, ExecutionCausalAttentionSubmissionAdapter,
        ExecutionKvStateSubmissionAdapter, ExecutionLinearAttentionSubmissionAdapter,
        ExecutionReadbackAdapter, ExecutionSessionAdapter, ExecutionSubmissionAdapter,
        ExecutionTransferAdapter, PreparedOperation, ShutdownReport,
    };
    use crate::kv_state::{KvStateAppendRequest, KvStateSnapshot};
    use crate::linear_attention::{LinearAttentionRequest, LinearAttentionStateSnapshot};

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
        available_memory_bytes: Arc<AtomicU64>,
    }

    impl Default for ExecutionRecorder {
        fn default() -> Self {
            Self {
                state: Arc::new(Mutex::new(RecorderState::default())),
                failure_kind: Arc::new(Mutex::new(None)),
                pending_kind: Arc::new(Mutex::new(None)),
                shutdown_calls: Arc::new(AtomicUsize::new(0)),
                available_memory_bytes: Arc::new(AtomicU64::new(u64::MAX)),
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

        fn set_failure(&self, kind: SemanticOpKind) {
            *self.failure_kind.lock().expect("failure lock") = Some(kind);
        }

        fn set_pending(&self, kind: SemanticOpKind) {
            *self.pending_kind.lock().expect("pending lock") = Some(kind);
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
            KvStateSnapshot::new(access.session_id(), state.id(), state.descriptor(), length)
                .map_err(|error| ExecutionError::InvalidRequest {
                    reason: error.to_string(),
                })
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
    fn dispatch_audit_counts_typed_multi_dispatch_and_kv_submission() {
        let mut audit = DispatchAuditAccumulator::default();
        let semantic = dispatch_evidence();
        audit.record_evidence(&semantic).unwrap();
        let mut linear = semantic.clone();
        linear.dispatch_count = 2;
        audit.record_evidence(&linear).unwrap();
        let mut kv_append = semantic;
        kv_append.dispatch_id = 3;
        audit.record_evidence(&kv_append).unwrap();
        assert_eq!(audit.submission_count, 3);
        assert_eq!(audit.kernel_dispatch_count, 4);
        assert_eq!(audit.target.as_deref(), Some("recorder"));
    }

    #[test]
    fn dispatch_audit_rejects_mixed_target_backend_and_fallback_evidence() {
        let mut audit = DispatchAuditAccumulator::default();
        audit.record_evidence(&dispatch_evidence()).unwrap();

        let mut wrong_target = dispatch_evidence();
        wrong_target.target = "other".to_owned();
        assert!(matches!(
            audit.record_evidence(&wrong_target),
            Err(QwenExecutionError::InvalidRequest(_))
        ));

        let mut wrong_backend = dispatch_evidence();
        wrong_backend.backend = 0;
        assert!(matches!(
            audit.record_evidence(&wrong_backend),
            Err(QwenExecutionError::InvalidRequest(_))
        ));

        let mut fallback_allowed = dispatch_evidence();
        fallback_allowed.fallback_allowed = true;
        assert!(matches!(
            audit.record_evidence(&fallback_allowed),
            Err(QwenExecutionError::InvalidRequest(_))
        ));

        let mut fallback_used = dispatch_evidence();
        fallback_used.fallback_used = true;
        assert!(matches!(
            audit.record_evidence(&fallback_used),
            Err(QwenExecutionError::InvalidRequest(_))
        ));
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
    fn full_prefill_decode_records_graph_order_and_publishes_after_state_snapshots() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([
            vec![101, 102, 103],
            vec![104],
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
        for state in core.kv_states.values() {
            assert_eq!(state.snapshot(core.session.as_ref()).unwrap().length(), 4);
        }
        for state in core.linear_states.values() {
            assert_eq!(state.snapshot(core.session.as_ref()).unwrap().length(), 4);
        }

        let preprocess = recorder.preprocess();
        assert_eq!(preprocess.len(), 16);
        assert!(
            preprocess[..8]
                .iter()
                .all(|entry| { *entry == (AttentionPreprocessPositionMode::Prefill, 0, 3) })
        );
        assert!(preprocess[8..].iter().all(|entry| {
            *entry == (AttentionPreprocessPositionMode::DecodeContinuation, 3, 1)
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
    fn failure_after_state_mutation_poison_rejects_reuse_and_never_publishes() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([vec![1, 2, 3]]));
        recorder.set_failure(SemanticOpKind::Add);
        let (mut core, _) = provisioned_core(Arc::clone(&recorder));

        assert!(matches!(
            core.prefill(&[1, 2, 3]),
            Err(QwenExecutionError::CompletionFailure { .. })
        ));
        assert!(core.lifecycle.poisoned.load(Ordering::Acquire));
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
        assert!(core.lifecycle.poisoned.load(Ordering::Acquire));
        assert_eq!(core.committed_length, 0);
        assert!(core.last_output.is_none());

        let lifecycle = Arc::new(RequestLifecycle::new());
        let guard = TransitionGuard::begin(Arc::clone(&lifecycle)).unwrap();
        drop(guard);
        assert!(lifecycle.poisoned.load(Ordering::Acquire));
        assert!(!lifecycle.in_flight.load(Ordering::Acquire));
    }

    #[test]
    fn argmax_sentinel_is_not_published_after_state_updates() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([vec![-1, 2, 3]]));
        let (mut core, _) = provisioned_core(Arc::clone(&recorder));
        assert!(matches!(
            core.prefill(&[1, 2, 3]),
            Err(QwenExecutionError::ArgmaxSentinel { index: 0 })
        ));
        assert!(core.lifecycle.poisoned.load(Ordering::Acquire));
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
            core.prefill(&[1]),
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
        assert!(!core.lifecycle.poisoned.load(Ordering::Acquire));
        assert_eq!(position_bytes(0, 3).unwrap(), i32_bytes(&[0, 1, 2]));
        assert_eq!(position_bytes(3, 1).unwrap(), i32_bytes(&[3]));
        assert!(position_bytes(i32::MAX as u64, 2).is_err());
    }

    #[test]
    fn argmax_token_outside_vocabulary_poison_rejects_publication() {
        let recorder = Arc::new(ExecutionRecorder::with_argmax_sequences([vec![
            QWEN35_VOCAB_SIZE as i32,
            2,
            3,
        ]]));
        let (mut core, _) = provisioned_core(recorder);
        assert!(matches!(
            core.prefill(&[1, 2, 3]),
            Err(QwenExecutionError::InvalidRequest(_))
        ));
        assert!(core.lifecycle.poisoned.load(Ordering::Acquire));
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
        assert!(!core.lifecycle.poisoned.load(Ordering::Acquire));
    }

    #[test]
    fn dispatch_audit_counts_kernel_dispatches_and_rejects_non_hip_evidence() {
        let mut audit = DispatchAuditAccumulator::default();
        let first = dispatch_evidence();
        audit.record_evidence(&first).unwrap();
        let mut two_kernel = first.clone();
        two_kernel.dispatch_id = 2;
        two_kernel.dispatch_count = 2;
        audit.record_evidence(&two_kernel).unwrap();
        assert_eq!(audit.target.as_deref(), Some("recorder"));
        assert_eq!(audit.submission_count, 2);
        assert_eq!(audit.kernel_dispatch_count, 3);
        assert!(audit.all_dispatches_hip);
        assert!(!audit.fallback_used);

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
            assert!(
                DispatchAuditAccumulator::default()
                    .record_evidence(&invalid)
                    .is_err()
            );
        }

        let mut mixed_target = DispatchAuditAccumulator::default();
        mixed_target.record_evidence(&first).unwrap();
        let mut wrong_target = first;
        wrong_target.target = "other".to_owned();
        assert!(mixed_target.record_evidence(&wrong_target).is_err());
    }

    #[test]
    fn resident_uploads_once_and_request_state_returns_to_resident() {
        let (graph, plan) = crate::qwen_graph::qwen35_execution_fixture();
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
                plan,
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

        let request = QwenExecutionCore::from_resident(Arc::clone(&resident), graph.clone())
            .expect("first fresh request provisions");
        let request_memory = session.memory_snapshot();
        assert!(request_memory.request_state().current_bytes() > 0);
        assert!(request_memory.workspace().current_bytes() > 0);
        drop(request);
        let returned = session.memory_snapshot();
        assert_eq!(
            returned.model_resident().current_bytes(),
            resident_memory.model_resident().current_bytes()
        );
        assert_eq!(returned.request_state().current_bytes(), 0);
        assert_eq!(returned.workspace().current_bytes(), 0);

        let request =
            QwenExecutionCore::from_resident(resident.clone(), graph_with_different_shape)
                .expect("second fresh request provisions");
        drop(request);
        assert_eq!(source.uploaded().len(), upload_count);
        drop(resident);
        assert_eq!(session.memory_snapshot().current_bytes(), 0);
    }
}
