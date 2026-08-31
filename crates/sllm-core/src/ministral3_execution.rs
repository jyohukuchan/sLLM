//! Resident, text-only execution for the reviewed Ministral 3 GGUF.
//!
//! The graph and weight modules deliberately stop at backend-neutral
//! contracts.  This module owns the small amount of orchestration needed to
//! turn those contracts into one ordered request: model tensors are resident
//! for the lifetime of a [`Ministral3ResidentModel`], while activation buffers
//! and full-attention KV state belong to each request.  Numerical work remains
//! entirely in the [`ExecutionSession`] adapter; this module never emulates a
//! backend or falls back to a host implementation.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use crate::execution::{
    BoundSemanticOp, DispatchEvidence, ExecutionBuffer, ExecutionError, ExecutionQueue,
    ExecutionSession, ExecutionState, KvState, OwnedTensorBinding, Submission,
};
use crate::kv_state::{CausalAttentionDescriptor, KvCacheEncoding, KvStateDescriptor};
use crate::ministral3_graph::{
    MINISTRAL3_GRAPH_HEAD_DIM, MINISTRAL3_GRAPH_KV_HEADS, MINISTRAL3_GRAPH_LAYER_COUNT,
    MINISTRAL3_GRAPH_MAX_CONTEXT, MINISTRAL3_GRAPH_Q_HEADS, MINISTRAL3_GRAPH_VOCAB_SIZE,
    Ministral3GraphError, Ministral3GraphNodeKind, Ministral3GraphTensor, Ministral3TensorClass,
    Ministral3TextGraph, build_ministral3_text_graph,
};
use crate::ministral3_weights::{
    MINISTRAL3_WEIGHT_BF16_TENSOR_COUNT, MINISTRAL3_WEIGHT_PLAN_SCHEMA,
    MINISTRAL3_WEIGHT_RESIDENT_BYTES, MINISTRAL3_WEIGHT_TENSOR_COUNT,
    VerifiedMinistral3WeightSource,
};
use crate::tensor::{TensorError, TensorView};
use crate::weights::{
    WeightClassification, WeightLoadPlan, WeightUploadError, upload_weight_from_source,
};
use crate::{AccessMode, AllocationCategory, DType, Encoding};

/// Errors raised while admitting or executing a Ministral 3 request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Ministral3ExecutionError {
    Invalid(String),
    Graph(Ministral3GraphError),
    Tensor(TensorError),
    Execution(ExecutionError),
    Weight(WeightUploadError),
}

impl Ministral3ExecutionError {
    fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}

impl fmt::Display for Ministral3ExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(formatter, "invalid Ministral 3 execution: {message}"),
            Self::Graph(error) => write!(formatter, "Ministral 3 graph error: {error}"),
            Self::Tensor(error) => write!(formatter, "Ministral 3 tensor error: {error}"),
            Self::Execution(error) => write!(formatter, "Ministral 3 execution error: {error}"),
            Self::Weight(error) => write!(formatter, "Ministral 3 weight upload error: {error}"),
        }
    }
}

impl std::error::Error for Ministral3ExecutionError {}

impl From<Ministral3GraphError> for Ministral3ExecutionError {
    fn from(error: Ministral3GraphError) -> Self {
        Self::Graph(error)
    }
}

impl From<TensorError> for Ministral3ExecutionError {
    fn from(error: TensorError) -> Self {
        Self::Tensor(error)
    }
}

impl From<ExecutionError> for Ministral3ExecutionError {
    fn from(error: ExecutionError) -> Self {
        Self::Execution(error)
    }
}

impl From<WeightUploadError> for Ministral3ExecutionError {
    fn from(error: WeightUploadError) -> Self {
        Self::Weight(error)
    }
}

/// Redacted, deterministic dispatch evidence for one completed transition.
///
/// The individual backend records remain available for diagnostics, but the
/// aggregate counters are the stable API used by callers and tests.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ministral3DispatchAudit {
    backend: u32,
    target: String,
    submission_count: u64,
    kernel_dispatch_count: u64,
    fallback_used: bool,
    dispatches: Vec<DispatchEvidence>,
}

impl Ministral3DispatchAudit {
    pub fn backend(&self) -> u32 {
        self.backend
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

    pub fn dispatches(&self) -> &[DispatchEvidence] {
        &self.dispatches
    }
}

#[derive(Default)]
struct DispatchAuditBuilder {
    backend: Option<u32>,
    target: Option<String>,
    submission_count: u64,
    kernel_dispatch_count: u64,
    fallback_used: bool,
    dispatches: Vec<DispatchEvidence>,
}

impl DispatchAuditBuilder {
    fn record(
        &mut self,
        evidence: &DispatchEvidence,
        expected_target: Option<&str>,
    ) -> Result<(), Ministral3ExecutionError> {
        if evidence.dispatch_count == 0
            || evidence.target.is_empty()
            || evidence.kernel_symbol.is_empty()
            || evidence.device_symbol.is_empty()
        {
            return Err(Ministral3ExecutionError::invalid(
                "successful dispatch evidence is empty or invalid",
            ));
        }
        if evidence.fallback_used {
            return Err(Ministral3ExecutionError::invalid(
                "fallback_used dispatch evidence is rejected",
            ));
        }
        if let Some(expected) = expected_target {
            if evidence.target != expected {
                return Err(Ministral3ExecutionError::invalid(
                    "dispatch target differs from the execution session target",
                ));
            }
        }
        if let Some(backend) = self.backend {
            if backend != evidence.backend {
                return Err(Ministral3ExecutionError::invalid(
                    "dispatch backend identity changed during a transition",
                ));
            }
        } else {
            self.backend = Some(evidence.backend);
        }
        if let Some(target) = self.target.as_deref() {
            if target != evidence.target {
                return Err(Ministral3ExecutionError::invalid(
                    "dispatch target changed during a transition",
                ));
            }
        } else {
            self.target = Some(evidence.target.clone());
        }
        self.submission_count = self.submission_count.checked_add(1).ok_or_else(|| {
            Ministral3ExecutionError::invalid("dispatch submission count overflowed")
        })?;
        self.kernel_dispatch_count = self
            .kernel_dispatch_count
            .checked_add(u64::from(evidence.dispatch_count))
            .ok_or_else(|| Ministral3ExecutionError::invalid("kernel dispatch count overflowed"))?;
        self.fallback_used |= evidence.fallback_used;
        self.dispatches.push(evidence.clone());
        Ok(())
    }

    fn finish(self) -> Result<Ministral3DispatchAudit, Ministral3ExecutionError> {
        Ok(Ministral3DispatchAudit {
            backend: self.backend.ok_or_else(|| {
                Ministral3ExecutionError::invalid("transition produced no dispatch")
            })?,
            target: self.target.ok_or_else(|| {
                Ministral3ExecutionError::invalid("transition produced no target")
            })?,
            submission_count: self.submission_count,
            kernel_dispatch_count: self.kernel_dispatch_count,
            fallback_used: self.fallback_used,
            dispatches: self.dispatches,
        })
    }
}

/// Output from a completed prefill or decode transition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Ministral3ExecutionOutput {
    token_ids: Vec<i32>,
    committed_length: u64,
    audit: Ministral3DispatchAudit,
}

impl Ministral3ExecutionOutput {
    pub fn token_ids(&self) -> &[i32] {
        &self.token_ids
    }

    pub const fn committed_length(&self) -> u64 {
        self.committed_length
    }

    pub fn audit(&self) -> &Ministral3DispatchAudit {
        &self.audit
    }
}

#[derive(Clone)]
struct ResidentTensor {
    buffer: ExecutionBuffer,
}

struct ResidentInner {
    session: Arc<ExecutionSession>,
    queue: ExecutionQueue,
    plan: WeightLoadPlan,
    completion_timeout: Duration,
    weights: BTreeMap<crate::WeightConsumerKey, ResidentTensor>,
    model_fingerprint: String,
}

/// Immutable model-resident Ministral 3 weights and its upload queue.
#[derive(Clone)]
pub struct Ministral3ResidentModel {
    inner: Arc<ResidentInner>,
}

impl fmt::Debug for Ministral3ResidentModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Ministral3ResidentModel")
            .field("session", &self.inner.session.id())
            .field("plan_digest", &self.inner.plan.digest_hex())
            .field("model_fingerprint", &self.inner.model_fingerprint)
            .finish_non_exhaustive()
    }
}

impl Ministral3ResidentModel {
    /// Upload all 236 BF16 resident tensors exactly once into tensor-sized
    /// model allocations.  The generic upload helper performs plan digest,
    /// descriptor, source range, and chunk-contiguity validation for every
    /// tensor before invoking the session upload callback.
    pub fn new_gguf(
        session: Arc<ExecutionSession>,
        plan: WeightLoadPlan,
        source: Arc<VerifiedMinistral3WeightSource>,
        completion_timeout: Duration,
    ) -> Result<Self, Ministral3ExecutionError> {
        validate_resident_identity(&session, &plan, source.as_ref(), completion_timeout)?;
        let available = session.available_memory_bytes()?.ok_or_else(|| {
            Ministral3ExecutionError::invalid("available VRAM telemetry is required")
        })?;
        if available < plan.total_destination_bytes {
            return Err(Ministral3ExecutionError::invalid(format!(
                "resident weights require {} bytes but only {available} bytes are available",
                plan.total_destination_bytes
            )));
        }
        let queue = session.create_queue()?;
        let mut weights = BTreeMap::new();
        for entry in &plan.entries {
            let key = entry.consumer.ok_or_else(|| {
                Ministral3ExecutionError::invalid("resident weight entry has no consumer")
            })?;
            let size = entry
                .source_range
                .get(1)
                .copied()
                .and_then(|end| {
                    entry
                        .source_range
                        .first()
                        .copied()
                        .and_then(|start| end.checked_sub(start))
                })
                .ok_or_else(|| {
                    Ministral3ExecutionError::invalid("resident source range underflowed")
                })?;
            let buffer = session.allocate_with_category(size, AllocationCategory::ModelResident)?;
            let view = TensorView::contiguous(
                DType::Bf16,
                &entry
                    .shape
                    .iter()
                    .map(|dimension| usize::try_from(*dimension))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| Ministral3ExecutionError::invalid("weight shape exceeds usize"))?,
            )?;
            if view.payload_bytes() != size {
                return Err(Ministral3ExecutionError::invalid(format!(
                    "resident tensor {} view size differs from source range",
                    entry.tensor_name
                )));
            }
            let destination = buffer.clone();
            let destination_size = size;
            let expected_digest = *plan.digest();
            upload_weight_from_source(
                &plan,
                expected_digest,
                source.as_ref(),
                &entry.tensor_name,
                crate::model::TensorDType::Bf16,
                0,
                destination_size,
                session.max_transfer_bytes()?,
                |relative_offset, bytes| {
                    let end = relative_offset
                        .checked_add(u64::try_from(bytes.len()).map_err(|_| {
                            WeightUploadError::invalid("weight upload length exceeds u64")
                        })?)
                        .ok_or_else(|| {
                            WeightUploadError::invalid("weight upload range overflowed")
                        })?;
                    if end > destination_size {
                        return Err(WeightUploadError::invalid(
                            "weight upload exceeds destination",
                        ));
                    }
                    let range = destination
                        .range(
                            relative_offset,
                            u64::try_from(bytes.len()).map_err(|_| {
                                WeightUploadError::invalid("weight upload length exceeds u64")
                            })?,
                        )
                        .map_err(|error| WeightUploadError::invalid(error.to_string()))?;
                    let mut transfer = session
                        .upload(&queue, range, bytes)
                        .map_err(|error| WeightUploadError::invalid(error.to_string()))?;
                    match transfer
                        .wait(completion_timeout)
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
            )?;
            if weights.insert(key, ResidentTensor { buffer }).is_some() {
                return Err(Ministral3ExecutionError::invalid(
                    "resident weight consumer is duplicated",
                ));
            }
        }
        if weights.len() != MINISTRAL3_WEIGHT_TENSOR_COUNT {
            return Err(Ministral3ExecutionError::invalid(
                "resident weight consumer count differs from 236",
            ));
        }
        Ok(Self {
            inner: Arc::new(ResidentInner {
                session,
                queue,
                plan,
                completion_timeout,
                weights,
                model_fingerprint: source.lock_fingerprint().to_owned(),
            }),
        })
    }

    pub fn session_id(&self) -> crate::ExecutionSessionId {
        self.inner.session.id()
    }

    pub fn plan_digest(&self) -> &[u8; 32] {
        self.inner.plan.digest()
    }

    pub fn resident_bytes(&self) -> u64 {
        self.inner.plan.total_destination_bytes
    }

    /// Allocate one request workspace at the initial max shape and 26 native
    /// FP16 KV states.  Subsequent transitions only bind narrower views into
    /// these same allocations.
    pub fn new_request(
        &self,
        prefill_token_count: u64,
        state_capacity: u64,
    ) -> Result<Ministral3ExecutionRequest, Ministral3ExecutionError> {
        let initial_rows = prefill_token_count.max(1);
        validate_request_admission(initial_rows, state_capacity)?;
        let graph = build_ministral3_text_graph(initial_rows, 0, state_capacity)?;
        validate_graph_contract(&graph, &self.inner.plan)?;
        let workspace_bytes = graph_workspace_bytes(&graph)?;
        let kv_descriptor = KvStateDescriptor::new_with_storage(
            0,
            state_capacity,
            MINISTRAL3_GRAPH_KV_HEADS as usize,
            MINISTRAL3_GRAPH_HEAD_DIM as usize,
            KvCacheEncoding::Fp16,
        )
        .map_err(|error| Ministral3ExecutionError::invalid(error.to_string()))?;
        let kv_per_layer = kv_descriptor
            .resident_bytes_per_plane()
            .ok_or_else(|| Ministral3ExecutionError::invalid("KV resident byte count overflowed"))?
            .checked_mul(2)
            .ok_or_else(|| {
                Ministral3ExecutionError::invalid("KV resident byte count overflowed")
            })?;
        let kv_bytes = kv_per_layer
            .checked_mul(u64::from(MINISTRAL3_GRAPH_LAYER_COUNT))
            .ok_or_else(|| {
                Ministral3ExecutionError::invalid("KV allocation byte count overflowed")
            })?;
        let required = workspace_bytes.checked_add(kv_bytes).ok_or_else(|| {
            Ministral3ExecutionError::invalid("request allocation byte count overflowed")
        })?;
        let total_required = self
            .inner
            .plan
            .total_destination_bytes
            .checked_add(required)
            .ok_or_else(|| {
                Ministral3ExecutionError::invalid("total request placement byte count overflowed")
            })?;
        let available = self
            .inner
            .session
            .available_memory_bytes()?
            .ok_or_else(|| {
                Ministral3ExecutionError::invalid("available VRAM telemetry is required")
            })?;
        if total_required > available {
            return Err(Ministral3ExecutionError::invalid(format!(
                "resident plus request allocation requires {total_required} bytes but only {available} bytes are available"
            )));
        }

        let mut buffers = Vec::with_capacity(graph.tensors().len());
        for tensor in graph.tensors() {
            if tensor.class() == Ministral3TensorClass::Weight {
                let key = tensor.weight().ok_or_else(|| {
                    Ministral3ExecutionError::invalid("weight graph tensor has no consumer")
                })?;
                buffers.push(
                    self.inner
                        .weights
                        .get(&key)
                        .ok_or_else(|| {
                            Ministral3ExecutionError::invalid(
                                "graph weight is absent from resident model",
                            )
                        })?
                        .buffer
                        .clone(),
                );
            } else if let Some(source_id) = tensor.alias_of() {
                let source = buffers.get(source_id).ok_or_else(|| {
                    Ministral3ExecutionError::invalid("graph alias source allocation is absent")
                })?;
                buffers.push(source.clone());
            } else {
                let bytes = tensor.view().span_bytes();
                buffers.push(
                    self.inner
                        .session
                        .allocate_with_category(bytes, AllocationCategory::Workspace)?,
                );
            }
        }
        validate_alias_buffers(&graph, &buffers)?;

        let mut kv_states = Vec::with_capacity(MINISTRAL3_GRAPH_LAYER_COUNT as usize);
        for layer in 0..MINISTRAL3_GRAPH_LAYER_COUNT {
            let descriptor = KvStateDescriptor::new_with_storage(
                layer,
                state_capacity,
                MINISTRAL3_GRAPH_KV_HEADS as usize,
                MINISTRAL3_GRAPH_HEAD_DIM as usize,
                KvCacheEncoding::Fp16,
            )
            .map_err(|error| Ministral3ExecutionError::invalid(error.to_string()))?;
            let state = self.inner.session.create_kv_state(descriptor)?;
            if state.session_id() != self.inner.session.id()
                || state.backend_name() != self.inner.session.backend_name()
                || state.descriptor() != descriptor
            {
                return Err(Ministral3ExecutionError::invalid(
                    "KV state identity or descriptor differs from the request session",
                ));
            }
            let snapshot = self.inner.session.kv_state_snapshot(&state)?;
            if snapshot.session_id() != self.inner.session.id()
                || snapshot.length() != 0
                || snapshot.descriptor() != descriptor
            {
                return Err(Ministral3ExecutionError::invalid(
                    "new KV state did not report an empty matching snapshot",
                ));
            }
            kv_states.push(state);
        }
        Ok(Ministral3ExecutionRequest {
            resident: Arc::clone(&self.inner),
            buffers,
            initial_graph: graph,
            kv_states,
            state_capacity,
            committed_length: 0,
            poisoned: false,
            workspace_bytes,
            last_audit: None,
        })
    }
}

/// One request-local Ministral 3 owner.
pub struct Ministral3ExecutionRequest {
    resident: Arc<ResidentInner>,
    buffers: Vec<ExecutionBuffer>,
    initial_graph: Ministral3TextGraph,
    kv_states: Vec<KvState>,
    state_capacity: u64,
    committed_length: u64,
    poisoned: bool,
    workspace_bytes: u64,
    last_audit: Option<Ministral3DispatchAudit>,
}

impl fmt::Debug for Ministral3ExecutionRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Ministral3ExecutionRequest")
            .field("session", &self.resident.session.id())
            .field("committed_length", &self.committed_length)
            .field("state_capacity", &self.state_capacity)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

impl Ministral3ExecutionRequest {
    pub const fn committed_length(&self) -> u64 {
        self.committed_length
    }

    pub const fn state_capacity(&self) -> u64 {
        self.state_capacity
    }

    pub const fn workspace_bytes(&self) -> u64 {
        self.workspace_bytes
    }

    pub fn last_audit(&self) -> Option<&Ministral3DispatchAudit> {
        self.last_audit.as_ref()
    }

    pub fn prefill(
        &mut self,
        token_ids: &[i32],
    ) -> Result<Ministral3ExecutionOutput, Ministral3ExecutionError> {
        if self.poisoned {
            return Err(Ministral3ExecutionError::invalid(
                "request is poisoned after a failed transition",
            ));
        }
        if self.committed_length != 0 {
            return Err(Ministral3ExecutionError::invalid(
                "prefill must start at position zero",
            ));
        }
        self.transition(token_ids)
    }

    pub fn decode(
        &mut self,
        token_id: i32,
    ) -> Result<Ministral3ExecutionOutput, Ministral3ExecutionError> {
        if self.poisoned {
            return Err(Ministral3ExecutionError::invalid(
                "request is poisoned after a failed transition",
            ));
        }
        if self.committed_length == 0 {
            return Err(Ministral3ExecutionError::invalid(
                "decode requires a committed prefill",
            ));
        }
        self.transition(&[token_id])
    }

    fn transition(
        &mut self,
        token_ids: &[i32],
    ) -> Result<Ministral3ExecutionOutput, Ministral3ExecutionError> {
        if self.poisoned {
            return Err(Ministral3ExecutionError::invalid(
                "request is poisoned after a failed transition",
            ));
        }
        if token_ids.is_empty() {
            return Err(Ministral3ExecutionError::invalid(
                "transition requires at least one token",
            ));
        }
        for &token in token_ids {
            if token < 0
                || u64::try_from(token)
                    .ok()
                    .is_none_or(|id| id >= MINISTRAL3_GRAPH_VOCAB_SIZE as u64)
            {
                return Err(Ministral3ExecutionError::invalid(
                    "token id is outside 0..131072",
                ));
            }
        }
        let token_count = u64::try_from(token_ids.len())
            .map_err(|_| Ministral3ExecutionError::invalid("token count does not fit u64"))?;
        let start = self.committed_length;
        let end = start
            .checked_add(token_count)
            .ok_or_else(|| Ministral3ExecutionError::invalid("committed length overflowed"))?;
        if end > self.state_capacity || end > MINISTRAL3_GRAPH_MAX_CONTEXT {
            return Err(Ministral3ExecutionError::invalid(
                "transition exceeds state capacity or context",
            ));
        }
        if let Err(error) = self.ensure_state_lengths(start) {
            // A state mismatch is an external consistency failure, not a
            // caller validation error.  Do not admit a later transition
            // against a state whose authoritative length is unknown.
            self.poisoned = true;
            return Err(error);
        }
        let graph = build_ministral3_text_graph(token_count, start, self.state_capacity)?;
        validate_graph_contract(&graph, &self.resident.plan)?;
        self.upload_runtime_inputs(&graph, token_ids, start)?;
        let result = self.execute_graph(&graph);
        let (selected, audit) = match result {
            Ok(value) => value,
            Err(error) => {
                self.poisoned = true;
                return Err(error);
            }
        };
        if let Err(error) = self.ensure_state_lengths(end) {
            self.poisoned = true;
            return Err(error);
        }
        self.committed_length = end;
        self.last_audit = Some(audit.clone());
        Ok(Ministral3ExecutionOutput {
            token_ids: selected,
            committed_length: end,
            audit,
        })
    }

    fn upload_runtime_inputs(
        &self,
        graph: &Ministral3TextGraph,
        token_ids: &[i32],
        start: u64,
    ) -> Result<(), Ministral3ExecutionError> {
        let rows = token_ids.len();
        let token_tensor = graph
            .tensors()
            .iter()
            .find(|tensor| tensor.label() == "input.token_ids")
            .ok_or_else(|| Ministral3ExecutionError::invalid("token input tensor is absent"))?;
        let position_tensor = graph
            .tensors()
            .iter()
            .find(|tensor| tensor.label() == "input.positions")
            .ok_or_else(|| Ministral3ExecutionError::invalid("position input tensor is absent"))?;
        let token_buffer = self.buffer_for_tensor(graph, token_tensor)?;
        let position_buffer = self.buffer_for_tensor(graph, position_tensor)?;
        let token_view = TensorView::contiguous(DType::I32, &[rows])?;
        let position_view = TensorView::contiguous(DType::I32, &[rows])?;
        if token_view != token_tensor.view().clone()
            || position_view != position_tensor.view().clone()
        {
            return Err(Ministral3ExecutionError::invalid(
                "runtime input view differs from graph contract",
            ));
        }
        let mut positions = Vec::with_capacity(rows);
        for index in 0..rows {
            let position =
                start
                    .checked_add(u64::try_from(index).map_err(|_| {
                        Ministral3ExecutionError::invalid("position index overflowed")
                    })?)
                    .ok_or_else(|| Ministral3ExecutionError::invalid("position overflowed"))?;
            positions.push(
                i32::try_from(position)
                    .map_err(|_| Ministral3ExecutionError::invalid("position exceeds i32"))?,
            );
        }
        upload_i32(
            &self.resident.session,
            &self.resident.queue,
            token_buffer,
            &token_view,
            token_ids,
            self.resident.completion_timeout,
        )?;
        upload_i32(
            &self.resident.session,
            &self.resident.queue,
            position_buffer,
            &position_view,
            &positions,
            self.resident.completion_timeout,
        )?;
        Ok(())
    }

    fn execute_graph(
        &self,
        graph: &Ministral3TextGraph,
    ) -> Result<(Vec<i32>, Ministral3DispatchAudit), Ministral3ExecutionError> {
        let expected_target = self.resident.session.expected_target();
        let mut audit = DispatchAuditBuilder::default();
        let mut selected = None;
        for node in graph.nodes() {
            let execute = match node.kind() {
                Ministral3GraphNodeKind::View | Ministral3GraphNodeKind::Reshape => Ok(()),
                Ministral3GraphNodeKind::YarnRopeQueryScale(stage) => {
                    let query = self.bind_tensor(
                        graph,
                        node.inputs().first().copied().ok_or_else(|| {
                            Ministral3ExecutionError::invalid("YaRN query input is absent")
                        })?,
                        AccessMode::Read,
                    )?;
                    let key = self.bind_tensor(
                        graph,
                        node.inputs().get(1).copied().ok_or_else(|| {
                            Ministral3ExecutionError::invalid("YaRN key input is absent")
                        })?,
                        AccessMode::Read,
                    )?;
                    let positions = self.bind_tensor(
                        graph,
                        node.inputs().get(2).copied().ok_or_else(|| {
                            Ministral3ExecutionError::invalid("YaRN positions input is absent")
                        })?,
                        AccessMode::Read,
                    )?;
                    let query_output = self.bind_tensor(
                        graph,
                        node.outputs().first().copied().ok_or_else(|| {
                            Ministral3ExecutionError::invalid("YaRN query output is absent")
                        })?,
                        AccessMode::Write,
                    )?;
                    let key_output = self.bind_tensor(
                        graph,
                        node.outputs().get(1).copied().ok_or_else(|| {
                            Ministral3ExecutionError::invalid("YaRN key output is absent")
                        })?,
                        AccessMode::Write,
                    )?;
                    let mut submission = self.resident.session.ministral3_yarn(
                        &self.resident.queue,
                        query,
                        key,
                        positions,
                        query_output,
                        key_output,
                        *stage,
                    )?;
                    require_success(
                        node.label(),
                        submission.wait(self.resident.completion_timeout)?,
                    )?;
                    audit.record(submission.dispatch(), expected_target.as_deref())
                }
                Ministral3GraphNodeKind::KvAppend(contract) => {
                    let state = self
                        .kv_states
                        .get(contract.layer() as usize)
                        .ok_or_else(|| {
                            Ministral3ExecutionError::invalid("KV layer state is absent")
                        })?;
                    let key = self.bind_tensor(
                        graph,
                        node.inputs().first().copied().ok_or_else(|| {
                            Ministral3ExecutionError::invalid("KV key input is absent")
                        })?,
                        AccessMode::Read,
                    )?;
                    let value = self.bind_tensor(
                        graph,
                        node.inputs().get(1).copied().ok_or_else(|| {
                            Ministral3ExecutionError::invalid("KV value input is absent")
                        })?,
                        AccessMode::Read,
                    )?;
                    let mut submission = self.resident.session.append_kv_state(
                        state,
                        &self.resident.queue,
                        key,
                        value,
                        contract.previous_length(),
                        contract.previous_length(),
                    )?;
                    require_success(
                        node.label(),
                        submission.wait(self.resident.completion_timeout)?,
                    )?;
                    audit.record(submission.dispatch(), expected_target.as_deref())?;
                    let snapshot = self.resident.session.kv_state_snapshot(state)?;
                    if snapshot.length() != contract.published_length() {
                        return Err(Ministral3ExecutionError::invalid(format!(
                            "KV layer {} published length differs",
                            contract.layer()
                        )));
                    }
                    Ok(())
                }
                Ministral3GraphNodeKind::CausalGqa(contract) => {
                    let state = self
                        .kv_states
                        .get(node.layer().ok_or_else(|| {
                            Ministral3ExecutionError::invalid("causal node has no layer")
                        })? as usize)
                        .ok_or_else(|| {
                            Ministral3ExecutionError::invalid("causal layer state is absent")
                        })?;
                    let query = self.bind_tensor(
                        graph,
                        node.inputs().first().copied().ok_or_else(|| {
                            Ministral3ExecutionError::invalid("causal query input is absent")
                        })?,
                        AccessMode::Read,
                    )?;
                    let output = self.bind_tensor(
                        graph,
                        node.outputs().first().copied().ok_or_else(|| {
                            Ministral3ExecutionError::invalid("causal output is absent")
                        })?,
                        AccessMode::Write,
                    )?;
                    let descriptor = CausalAttentionDescriptor::new(
                        contract.start_position(),
                        u64::from(contract.query_count()),
                        contract.expected_kv_length(),
                    )
                    .map_err(|error| Ministral3ExecutionError::invalid(error.to_string()))?;
                    let mut submission = self.resident.session.causal_attention(
                        state,
                        &self.resident.queue,
                        query,
                        output,
                        descriptor,
                    )?;
                    require_success(
                        node.label(),
                        submission.wait(self.resident.completion_timeout)?,
                    )?;
                    audit.record(submission.dispatch(), expected_target.as_deref())
                }
                _ => {
                    let operation = node.operation().ok_or_else(|| {
                        Ministral3ExecutionError::invalid(format!(
                            "semantic node {} has no operation",
                            node.label()
                        ))
                    })?;
                    let inputs = node
                        .inputs()
                        .iter()
                        .map(|id| self.bind_tensor(graph, *id, AccessMode::Read))
                        .collect::<Result<Vec<_>, _>>()?;
                    let outputs = node
                        .outputs()
                        .iter()
                        .map(|id| self.bind_tensor(graph, *id, AccessMode::Write))
                        .collect::<Result<Vec<_>, _>>()?;
                    let bound = BoundSemanticOp::new(Arc::new(operation.clone()), inputs, outputs)?;
                    let prepared = self.resident.session.prepare(Arc::new(bound))?;
                    let mut submission = self
                        .resident
                        .session
                        .submit(&prepared, &self.resident.queue)?;
                    require_success(
                        node.label(),
                        submission.wait(self.resident.completion_timeout)?,
                    )?;
                    if node.kind() == &Ministral3GraphNodeKind::Argmax {
                        selected = Some(read_selected(
                            &mut submission,
                            graph,
                            node,
                            self.resident.completion_timeout,
                        )?);
                    }
                    audit.record(submission.dispatch(), expected_target.as_deref())
                }
            };
            execute?;
        }
        let selected = selected.ok_or_else(|| {
            Ministral3ExecutionError::invalid("graph did not publish terminal selected tokens")
        })?;
        if selected.len() != 1 {
            return Err(Ministral3ExecutionError::invalid(
                "terminal selected token count differs from one",
            ));
        }
        Ok((selected, audit.finish()?))
    }

    fn ensure_state_lengths(&self, expected: u64) -> Result<(), Ministral3ExecutionError> {
        for state in &self.kv_states {
            if state.session_id() != self.resident.session.id() {
                return Err(Ministral3ExecutionError::invalid(
                    "KV state session identity differs",
                ));
            }
            let snapshot = self.resident.session.kv_state_snapshot(state)?;
            if snapshot.length() != expected {
                return Err(Ministral3ExecutionError::invalid(format!(
                    "KV layer {} length is {}, expected {expected}",
                    state.layer_id(),
                    snapshot.length()
                )));
            }
        }
        Ok(())
    }

    fn buffer_for_tensor(
        &self,
        graph: &Ministral3TextGraph,
        tensor: &Ministral3GraphTensor,
    ) -> Result<&ExecutionBuffer, Ministral3ExecutionError> {
        let index = self
            .initial_graph
            .tensors()
            .iter()
            .position(|candidate| {
                candidate.label() == tensor.label() && candidate.class() == tensor.class()
            })
            .ok_or_else(|| {
                Ministral3ExecutionError::invalid(format!(
                    "tensor label {} is absent from initial workspace",
                    tensor.label()
                ))
            })?;
        let buffer = self
            .buffers
            .get(index)
            .ok_or_else(|| Ministral3ExecutionError::invalid("tensor buffer index is absent"))?;
        if tensor.view().end_offset() > buffer.size_bytes() {
            return Err(Ministral3ExecutionError::invalid(format!(
                "tensor {} view exceeds workspace buffer",
                tensor.label()
            )));
        }
        let _ = graph;
        Ok(buffer)
    }

    fn bind_tensor(
        &self,
        graph: &Ministral3TextGraph,
        tensor_id: usize,
        access: AccessMode,
    ) -> Result<OwnedTensorBinding, Ministral3ExecutionError> {
        let tensor = graph
            .tensors()
            .get(tensor_id)
            .ok_or_else(|| Ministral3ExecutionError::invalid("node tensor id is absent"))?;
        let buffer = self.buffer_for_tensor(graph, tensor)?.clone();
        Ok(self
            .resident
            .session
            .bind(&buffer, tensor.view().clone(), access)?)
    }
}

fn validate_resident_identity(
    session: &ExecutionSession,
    plan: &WeightLoadPlan,
    source: &VerifiedMinistral3WeightSource,
    completion_timeout: Duration,
) -> Result<(), Ministral3ExecutionError> {
    if completion_timeout.is_zero()
        || plan.schema_version != MINISTRAL3_WEIGHT_PLAN_SCHEMA
        || plan.repo_id != source.repository()
        || plan.resolved_revision != source.revision()
        || plan.lock_fingerprint != source.lock_fingerprint()
        || !plan.tied_embeddings
        || plan.total_destination_bytes != MINISTRAL3_WEIGHT_RESIDENT_BYTES
        || plan.entries.len() != MINISTRAL3_WEIGHT_TENSOR_COUNT
        || source.tensors().len() != MINISTRAL3_WEIGHT_TENSOR_COUNT
    {
        return Err(Ministral3ExecutionError::invalid(
            "GGUF source and resident plan identities differ",
        ));
    }
    if !plan
        .has_valid_digest()
        .map_err(|error| Ministral3ExecutionError::invalid(error.to_string()))?
    {
        return Err(Ministral3ExecutionError::invalid(
            "resident weight plan digest is invalid",
        ));
    }
    if session.backend_name().is_empty() {
        return Err(Ministral3ExecutionError::invalid(
            "execution session backend identity is empty",
        ));
    }
    let mut names = BTreeSet::new();
    let mut consumers = BTreeSet::new();
    let source_file = source.gguf().path().display().to_string();
    for entry in &plan.entries {
        if entry.classification != WeightClassification::Required
            || entry.dtype != crate::model::TensorDType::Bf16
            || entry.consumer.is_none()
            || entry.destination_start.is_none()
            || entry.chunks.is_empty()
            || entry.source_file != source_file
            || entry.locked_file_size != source.gguf().file_size()
            || entry.locked_file_sha256 != source.file_sha256()
            || !names.insert(entry.tensor_name.as_str())
            || !consumers.insert(entry.consumer)
        {
            return Err(Ministral3ExecutionError::invalid(
                "resident plan contains a non-canonical weight entry",
            ));
        }
    }
    if consumers.len() != MINISTRAL3_WEIGHT_TENSOR_COUNT {
        return Err(Ministral3ExecutionError::invalid(
            "resident plan consumer count differs from 236",
        ));
    }
    Ok(())
}

fn validate_request_admission(
    initial_rows: u64,
    state_capacity: u64,
) -> Result<(), Ministral3ExecutionError> {
    if initial_rows == 0
        || state_capacity == 0
        || initial_rows > state_capacity
        || state_capacity > MINISTRAL3_GRAPH_MAX_CONTEXT
    {
        return Err(Ministral3ExecutionError::invalid(
            "request token count or state capacity is invalid",
        ));
    }
    Ok(())
}

fn graph_workspace_bytes(graph: &Ministral3TextGraph) -> Result<u64, Ministral3ExecutionError> {
    graph
        .tensors()
        .iter()
        .filter(|tensor| {
            tensor.class() != Ministral3TensorClass::Weight && !tensor.is_zero_copy_alias()
        })
        .try_fold(0_u64, |total, tensor| {
            total
                .checked_add(tensor.view().span_bytes())
                .ok_or_else(|| Ministral3ExecutionError::invalid("workspace byte count overflowed"))
        })
}

fn validate_alias_buffers(
    graph: &Ministral3TextGraph,
    buffers: &[ExecutionBuffer],
) -> Result<(), Ministral3ExecutionError> {
    let aliases = graph
        .tensors()
        .iter()
        .filter(|tensor| tensor.is_zero_copy_alias())
        .collect::<Vec<_>>();
    if aliases.len() != 4 * MINISTRAL3_GRAPH_LAYER_COUNT as usize + 1 {
        return Err(Ministral3ExecutionError::invalid(
            "Ministral 3 graph alias count differs from 105",
        ));
    }
    for tensor in aliases {
        let source = tensor
            .alias_of()
            .ok_or_else(|| Ministral3ExecutionError::invalid("alias source is absent"))?;
        if source >= tensor.id() {
            return Err(Ministral3ExecutionError::invalid(
                "alias source must precede the alias tensor",
            ));
        }
        let source_tensor = graph
            .tensors()
            .get(source)
            .ok_or_else(|| Ministral3ExecutionError::invalid("alias source tensor is absent"))?;
        if source_tensor.class() != Ministral3TensorClass::Activation
            || source_tensor.is_zero_copy_alias()
        {
            return Err(Ministral3ExecutionError::invalid(
                "alias source must be a non-alias activation",
            ));
        }
        let alias_buffer = buffers.get(tensor.id()).ok_or_else(|| {
            Ministral3ExecutionError::invalid("alias buffer allocation is absent")
        })?;
        let source_buffer = buffers.get(source).ok_or_else(|| {
            Ministral3ExecutionError::invalid("alias source buffer allocation is absent")
        })?;
        if alias_buffer.id() != source_buffer.id() {
            return Err(Ministral3ExecutionError::invalid(format!(
                "alias {} does not share its exact source buffer",
                tensor.label()
            )));
        }
        let source_view = graph.tensors()[source].view();
        if tensor.view().byte_offset() < source_view.byte_offset()
            || tensor.view().span_bytes() > source_view.span_bytes()
        {
            return Err(Ministral3ExecutionError::invalid(
                "alias view exceeds its source buffer range",
            ));
        }
    }
    Ok(())
}

fn validate_graph_contract(
    graph: &Ministral3TextGraph,
    plan: &WeightLoadPlan,
) -> Result<(), Ministral3ExecutionError> {
    if graph
        .tensors()
        .iter()
        .filter(|tensor| tensor.class() == Ministral3TensorClass::Weight)
        .count()
        != MINISTRAL3_WEIGHT_TENSOR_COUNT
    {
        return Err(Ministral3ExecutionError::invalid(
            "graph weight tensor count differs from 236",
        ));
    }
    if graph.nodes().len() != 499 {
        return Err(Ministral3ExecutionError::invalid(
            "graph node count differs from the reviewed topology",
        ));
    }
    let mut tensor_labels = BTreeSet::new();
    if graph.tensors().iter().any(|tensor| {
        !tensor_labels.insert(tensor.label())
            || (tensor.class() == Ministral3TensorClass::Weight) != tensor.weight().is_some()
            || (tensor.class() == Ministral3TensorClass::Activation) != tensor.writer().is_some()
            || (tensor.class() != Ministral3TensorClass::Activation && tensor.alias_of().is_some())
    }) {
        return Err(Ministral3ExecutionError::invalid(
            "graph tensor labels or class/consumer/writer mapping is invalid",
        ));
    }
    let mut node_labels = BTreeSet::new();
    if graph
        .nodes()
        .iter()
        .any(|node| !node_labels.insert(node.label()))
    {
        return Err(Ministral3ExecutionError::invalid(
            "graph node labels are not unique",
        ));
    }
    if graph
        .tensors()
        .iter()
        .filter(|tensor| tensor.is_zero_copy_alias())
        .count()
        != 4 * MINISTRAL3_GRAPH_LAYER_COUNT as usize + 1
    {
        return Err(Ministral3ExecutionError::invalid(
            "graph alias count differs from 105",
        ));
    }
    let mut consumers = BTreeSet::new();
    for tensor in graph.tensors() {
        if tensor.class() == Ministral3TensorClass::Weight {
            let key = tensor.weight().ok_or_else(|| {
                Ministral3ExecutionError::invalid("weight tensor has no consumer")
            })?;
            if !consumers.insert(key) {
                return Err(Ministral3ExecutionError::invalid(
                    "graph has duplicate weight consumer tensor",
                ));
            }
            let entry = plan
                .entries
                .iter()
                .find(|entry| entry.consumer == Some(key))
                .ok_or_else(|| {
                    Ministral3ExecutionError::invalid("graph consumer is absent from plan")
                })?;
            if entry.shape
                != tensor
                    .view()
                    .shape()
                    .iter()
                    .map(|value| u64::try_from(*value).unwrap_or(u64::MAX))
                    .collect::<Vec<_>>()
                || entry.dtype != crate::model::TensorDType::Bf16
            {
                return Err(Ministral3ExecutionError::invalid(
                    "graph weight shape or dtype differs from plan",
                ));
            }
        }
    }
    if consumers.len() != MINISTRAL3_WEIGHT_BF16_TENSOR_COUNT {
        return Err(Ministral3ExecutionError::invalid(
            "graph consumer count differs from resident catalog",
        ));
    }
    for (index, node) in graph.nodes().iter().enumerate() {
        if node.id() != index
            || node
                .dependencies()
                .iter()
                .any(|dependency| *dependency >= index)
        {
            return Err(Ministral3ExecutionError::invalid(
                "graph node order or dependency mapping is invalid",
            ));
        }
        for tensor_id in node.inputs().iter().chain(node.outputs()) {
            if *tensor_id >= graph.tensors().len() {
                return Err(Ministral3ExecutionError::invalid(
                    "graph node references an absent tensor",
                ));
            }
        }
        let expected_label = if index == 0 {
            "embedding".to_owned()
        } else if index == 499 - 3 {
            "final_norm.terminal".to_owned()
        } else if index == 499 - 2 {
            "tied_logits".to_owned()
        } else if index == 499 - 1 {
            "argmax".to_owned()
        } else if index == 1 + 26 * 19 {
            "final_norm".to_owned()
        } else {
            let layer_index = (index - 1) / 19;
            let layer_node = (index - 1) % 19;
            let stage = [
                "input_norm",
                "q_proj",
                "k_proj",
                "v_proj",
                "q_proj.reshape",
                "k_proj.reshape",
                "v_proj.reshape",
                "yarn_rope_query_scale",
                "kv_append",
                "causal_gqa",
                "attention.output.view",
                "o_proj",
                "attention_residual",
                "post_attention_norm",
                "mlp_gate",
                "mlp_up",
                "mlp_silu_mul",
                "mlp_down",
                "mlp_residual",
            ]
            .get(layer_node)
            .ok_or_else(|| {
                Ministral3ExecutionError::invalid("graph layer node index is invalid")
            })?;
            format!("layer.{layer_index}.{stage}")
        };
        if node.label() != expected_label {
            return Err(Ministral3ExecutionError::invalid(
                "graph node label differs from the reviewed topology",
            ));
        }
        match node.kind() {
            Ministral3GraphNodeKind::View | Ministral3GraphNodeKind::Reshape => {
                if node.operation().is_some()
                    || node.inputs().len() != 1
                    || node.outputs().len() != 1
                {
                    return Err(Ministral3ExecutionError::invalid(
                        "graph alias node contract is invalid",
                    ));
                }
            }
            Ministral3GraphNodeKind::YarnRopeQueryScale(stage) => {
                if node.operation().is_some()
                    || node.inputs().len() != 3
                    || node.outputs().len() != 2
                    || stage.start_position() != graph.start_position()
                    || u64::from(stage.token_count()) != graph.token_count()
                {
                    return Err(Ministral3ExecutionError::invalid(
                        "YaRN node contract differs from transition",
                    ));
                }
            }
            Ministral3GraphNodeKind::KvAppend(contract) => {
                if node.operation().is_some()
                    || contract.previous_length() != graph.start_position()
                    || u64::from(contract.append_count()) != graph.token_count()
                    || contract.published_length() != graph.expected_length()
                    || contract.capacity() != graph.state_capacity()
                {
                    return Err(Ministral3ExecutionError::invalid(
                        "KV append contract differs from transition",
                    ));
                }
            }
            Ministral3GraphNodeKind::CausalGqa(contract) => {
                if node.operation().is_some()
                    || contract.q_heads() != MINISTRAL3_GRAPH_Q_HEADS
                    || contract.kv_heads() != MINISTRAL3_GRAPH_KV_HEADS
                    || contract.head_dim() != MINISTRAL3_GRAPH_HEAD_DIM
                    || contract.start_position() != graph.start_position()
                    || u64::from(contract.query_count()) != graph.token_count()
                    || contract.expected_kv_length() != graph.expected_length()
                    || contract.scaling().to_bits()
                        != (1.0 / (MINISTRAL3_GRAPH_HEAD_DIM as f32).sqrt()).to_bits()
                {
                    return Err(Ministral3ExecutionError::invalid(
                        "causal attention contract differs from transition",
                    ));
                }
            }
            _ => {
                let operation = node.operation().ok_or_else(|| {
                    Ministral3ExecutionError::invalid(format!(
                        "semantic node {} has no operation",
                        node.label()
                    ))
                })?;
                operation
                    .validate()
                    .map_err(|error| Ministral3ExecutionError::invalid(error.to_string()))?;
                if node.kind().reused_semantic_kind() != Some(operation.kind()) {
                    return Err(Ministral3ExecutionError::invalid(
                        "semantic node kind differs from operation kind",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn upload_i32(
    session: &ExecutionSession,
    queue: &ExecutionQueue,
    buffer: &ExecutionBuffer,
    view: &TensorView,
    values: &[i32],
    timeout: Duration,
) -> Result<(), Ministral3ExecutionError> {
    if view.dtype() != DType::I32
        || view.encoding() != Encoding::Unquantized
        || view.shape() != [values.len()]
    {
        return Err(Ministral3ExecutionError::invalid(
            "I32 runtime input view is invalid",
        ));
    }
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let range = buffer.range(
        0,
        u64::try_from(bytes.len())
            .map_err(|_| Ministral3ExecutionError::invalid("I32 input byte count overflowed"))?,
    )?;
    let mut transfer = session.upload(queue, range, Arc::<[u8]>::from(bytes))?;
    require_success("runtime input upload", transfer.wait(timeout)?)
}

fn read_selected(
    submission: &mut Submission,
    graph: &Ministral3TextGraph,
    node: &crate::ministral3_graph::Ministral3GraphNode,
    timeout: Duration,
) -> Result<Vec<i32>, Ministral3ExecutionError> {
    if node.outputs().len() != 1 || node.outputs()[0] >= graph.tensors().len() {
        return Err(Ministral3ExecutionError::invalid(
            "argmax output tensor is invalid",
        ));
    }
    let output = graph.tensors()[node.outputs()[0]].view();
    let rows = 1_usize;
    if output.dtype() != DType::I32
        || output.encoding() != Encoding::Unquantized
        || output.shape() != [rows]
    {
        return Err(Ministral3ExecutionError::invalid(
            "argmax output is not terminal I32 [tokens]",
        ));
    }
    let mut readback = submission.start_output_readback(0)?;
    require_success("argmax readback", readback.wait(timeout)?)?;
    let mut bytes = vec![
        0_u8;
        rows.checked_mul(4)
            .ok_or_else(|| Ministral3ExecutionError::invalid(
                "argmax readback size overflowed"
            ))?
    ];
    readback.read_into(&mut bytes)?;
    let mut selected = Vec::with_capacity(rows);
    for chunk in bytes.chunks_exact(4) {
        let value = i32::from_le_bytes(chunk.try_into().expect("chunks_exact gives four bytes"));
        if value < 0
            || u64::try_from(value)
                .ok()
                .is_none_or(|id| id >= MINISTRAL3_GRAPH_VOCAB_SIZE as u64)
        {
            return Err(Ministral3ExecutionError::invalid(
                "argmax selected token is outside vocabulary",
            ));
        }
        selected.push(value);
    }
    Ok(selected)
}

fn require_success(label: &str, state: ExecutionState) -> Result<(), Ministral3ExecutionError> {
    match state {
        ExecutionState::Success => Ok(()),
        ExecutionState::Pending => Err(Ministral3ExecutionError::invalid(format!(
            "{label} remained pending"
        ))),
        ExecutionState::Failure => {
            Err(Ministral3ExecutionError::invalid(format!("{label} failed")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use crate::{
        AdapterResource, BoundSemanticOp, BufferRange, DispatchEvidence, ExecutionAdapterAccess,
        ExecutionCausalAttentionSubmissionAdapter, ExecutionError,
        ExecutionMinistral3YarnSubmissionAdapter, ExecutionReadbackAdapter,
        ExecutionSessionAdapter, ExecutionSubmissionAdapter, ExecutionTransferAdapter,
        KvStateAppendRequest, KvStateDescriptor, KvStateSnapshot, MINISTRAL3_GRAPH_HIDDEN_SIZE,
        PrepareSupport, PreparedOperation, SemanticOpDescriptor, ShutdownReport, TensorDType,
        WeightClassification, WeightLoadChunk, WeightLoadEntry, WeightLoadPlan,
    };

    #[derive(Clone, Copy)]
    struct TestState {
        id: crate::KvStateId,
        descriptor: KvStateDescriptor,
        length: u64,
    }

    #[derive(Default)]
    struct RecordingAdapter {
        states: Arc<Mutex<Vec<TestState>>>,
        fail_submit: AtomicBool,
    }

    struct RecordingSubmission;
    struct RecordingTransfer;
    struct RecordingReadback {
        bytes: Vec<u8>,
    }
    struct RecordingAttention;
    struct RecordingYarn;
    struct RecordingKvAppend {
        states: Arc<Mutex<Vec<TestState>>>,
        request: KvStateAppendRequest,
        complete: bool,
    }

    impl RecordingAdapter {
        fn evidence(symbol: &str) -> DispatchEvidence {
            DispatchEvidence {
                abi_version: 1,
                info_version: 1,
                dispatch_id: 1,
                dispatch_count: 1,
                kernel_id: 1,
                workgroup_size_x: 1,
                grid_size_x: 1,
                row_count: 1,
                normalized_size: 1,
                backend: 7,
                fallback_allowed: false,
                fallback_used: false,
                kernel_symbol: symbol.to_owned(),
                device_symbol: "ministral3-host-test".to_owned(),
                target: "ministral3-host-test".to_owned(),
            }
        }
    }

    impl ExecutionSubmissionAdapter for RecordingSubmission {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }

        fn start_output_readback(
            &mut self,
            _access: &ExecutionAdapterAccess<'_>,
            output: &OwnedTensorBinding,
        ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError> {
            Ok(Box::new(RecordingReadback {
                bytes: vec![
                    0;
                    usize::try_from(output.view().payload_bytes()).map_err(|_| {
                        ExecutionError::InvalidRange {
                            reason: "test output size does not fit usize".to_owned(),
                        }
                    })?
                ],
            }))
        }
    }

    impl ExecutionTransferAdapter for RecordingTransfer {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }
    }

    impl ExecutionReadbackAdapter for RecordingReadback {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }

        fn read_into(&mut self, destination: &mut [u8]) -> Result<u64, ExecutionError> {
            if destination.len() != self.bytes.len() {
                return Err(ExecutionError::InvalidRange {
                    reason: "test readback size differs".to_owned(),
                });
            }
            destination.copy_from_slice(&self.bytes);
            u64::try_from(destination.len()).map_err(|_| ExecutionError::InvalidRange {
                reason: "test readback size exceeds u64".to_owned(),
            })
        }
    }

    impl ExecutionCausalAttentionSubmissionAdapter for RecordingAttention {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }
    }

    impl ExecutionMinistral3YarnSubmissionAdapter for RecordingYarn {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }
    }

    impl RecordingKvAppend {
        fn finish(&mut self) -> Result<ExecutionState, ExecutionError> {
            if !self.complete {
                let mut states = self.states.lock().map_err(|_| ExecutionError::Busy)?;
                let state = states
                    .iter_mut()
                    .find(|state| state.id == self.request.state_id())
                    .ok_or(ExecutionError::WrongKvState {
                        expected: self.request.state_id(),
                        actual: crate::KvStateId::new(0),
                    })?;
                if state.length != self.request.expected_length() {
                    return Err(ExecutionError::StaleKvLength {
                        expected: self.request.expected_length(),
                        actual: state.length,
                    });
                }
                state.length = self.request.end_position();
                self.complete = true;
            }
            Ok(ExecutionState::Success)
        }
    }

    impl crate::ExecutionKvStateSubmissionAdapter for RecordingKvAppend {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            self.finish()
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            self.finish()
        }
    }

    impl ExecutionSessionAdapter for RecordingAdapter {
        fn expected_target(&self) -> Option<String> {
            Some("ministral3-host-test".to_owned())
        }

        fn max_transfer_bytes(&self) -> u64 {
            1 << 30
        }

        fn available_memory_bytes(&self) -> Option<u64> {
            Some(u64::MAX)
        }

        fn supports(&self, _descriptor: &SemanticOpDescriptor) -> PrepareSupport {
            PrepareSupport::Supported
        }

        fn create_queue(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
        ) -> Result<AdapterResource, ExecutionError> {
            Ok(AdapterResource::new(()))
        }

        fn allocate(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _size_bytes: u64,
        ) -> Result<AdapterResource, ExecutionError> {
            Ok(AdapterResource::new(()))
        }

        fn prepare(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _operation: &BoundSemanticOp,
        ) -> Result<AdapterResource, ExecutionError> {
            Ok(AdapterResource::new(()))
        }

        fn submit(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _prepared: &PreparedOperation,
            _queue: &ExecutionQueue,
        ) -> Result<(Box<dyn ExecutionSubmissionAdapter>, DispatchEvidence), ExecutionError>
        {
            if self.fail_submit.load(Ordering::Relaxed) {
                return Err(ExecutionError::BackendStatus {
                    status: 99,
                    diagnostic: "injected semantic failure".to_owned(),
                });
            }
            Ok((
                Box::new(RecordingSubmission),
                Self::evidence("test.semantic"),
            ))
        }

        fn upload(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _queue: &ExecutionQueue,
            _destination: &BufferRange,
            _bytes: Arc<[u8]>,
        ) -> Result<Box<dyn ExecutionTransferAdapter>, ExecutionError> {
            Ok(Box::new(RecordingTransfer))
        }

        fn readback(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _queue: &ExecutionQueue,
            source: &BufferRange,
        ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError> {
            Ok(Box::new(RecordingReadback {
                bytes: vec![
                    0;
                    usize::try_from(source.size_bytes()).map_err(|_| {
                        ExecutionError::InvalidRange {
                            reason: "test readback size does not fit usize".to_owned(),
                        }
                    })?
                ],
            }))
        }

        fn shutdown(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _deadline: Duration,
        ) -> Result<ShutdownReport, ExecutionError> {
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
            self.states
                .lock()
                .map_err(|_| ExecutionError::Busy)?
                .push(TestState {
                    id: state_id,
                    descriptor,
                    length: 0,
                });
            Ok(AdapterResource::new(()))
        }

        fn kv_state_snapshot(
            &self,
            access: &ExecutionAdapterAccess<'_>,
            state: &KvState,
        ) -> Result<KvStateSnapshot, ExecutionError> {
            let states = self.states.lock().map_err(|_| ExecutionError::Busy)?;
            let entry = states.iter().find(|entry| entry.id == state.id()).ok_or(
                ExecutionError::WrongKvState {
                    expected: state.id(),
                    actual: crate::KvStateId::new(0),
                },
            )?;
            KvStateSnapshot::new(
                access.session_id(),
                entry.id,
                entry.descriptor,
                entry.length,
            )
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
                Box<dyn crate::ExecutionKvStateSubmissionAdapter>,
                DispatchEvidence,
            ),
            ExecutionError,
        > {
            if state.id() != request.state_id() {
                return Err(ExecutionError::WrongKvState {
                    expected: state.id(),
                    actual: request.state_id(),
                });
            }
            Ok((
                Box::new(RecordingKvAppend {
                    states: Arc::clone(&self.states),
                    request: *request,
                    complete: false,
                }),
                Self::evidence("test.kv_append"),
            ))
        }

        fn execute_causal_attention(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _state: &KvState,
            _queue: &ExecutionQueue,
            _query: &OwnedTensorBinding,
            _output: &OwnedTensorBinding,
            _descriptor: CausalAttentionDescriptor,
        ) -> Result<
            (
                Box<dyn ExecutionCausalAttentionSubmissionAdapter>,
                DispatchEvidence,
            ),
            ExecutionError,
        > {
            Ok((
                Box::new(RecordingAttention),
                Self::evidence("test.causal_attention"),
            ))
        }

        fn execute_ministral3_yarn(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _queue: &ExecutionQueue,
            _query: &OwnedTensorBinding,
            _key: &OwnedTensorBinding,
            _positions: &OwnedTensorBinding,
            _query_output: &OwnedTensorBinding,
            _key_output: &OwnedTensorBinding,
            _stage: crate::Ministral3YarnQueryScaleStage,
        ) -> Result<
            (
                Box<dyn ExecutionMinistral3YarnSubmissionAdapter>,
                DispatchEvidence,
            ),
            ExecutionError,
        > {
            Ok((Box::new(RecordingYarn), Self::evidence("test.yarn")))
        }
    }

    fn synthetic_plan(graph: &Ministral3TextGraph) -> WeightLoadPlan {
        let entries = graph
            .tensors()
            .iter()
            .filter_map(|tensor| {
                let consumer = tensor.weight()?;
                let bytes = tensor.view().payload_bytes();
                Some(WeightLoadEntry {
                    tensor_name: tensor.label().to_owned(),
                    classification: WeightClassification::Required,
                    consumer: Some(consumer),
                    dtype: TensorDType::Bf16,
                    shape: tensor
                        .view()
                        .shape()
                        .iter()
                        .map(|dimension| u64::try_from(*dimension).expect("test shape fits u64"))
                        .collect(),
                    source_file: "test.gguf".to_owned(),
                    locked_file_size: bytes,
                    locked_file_sha256: "test".to_owned(),
                    source_range: [0, bytes],
                    destination_start: Some(0),
                    chunks: vec![WeightLoadChunk {
                        source_offset: 0,
                        destination_offset: 0,
                        byte_length: bytes,
                    }],
                })
            })
            .collect();
        WeightLoadPlan::from_verified_entries(
            crate::weights::VerifiedWeightPlanMetadata {
                schema_version: MINISTRAL3_WEIGHT_PLAN_SCHEMA.to_owned(),
                repo_id: "test".to_owned(),
                resolved_revision: "test".to_owned(),
                lock_fingerprint: "test".to_owned(),
                tied_embeddings: true,
                chunk_size: 1 << 20,
                total_destination_bytes: 0,
            },
            entries,
        )
        .expect("synthetic plan digest")
    }

    fn graph_buffers(
        session: &ExecutionSession,
        graph: &Ministral3TextGraph,
    ) -> Vec<ExecutionBuffer> {
        let mut buffers: Vec<ExecutionBuffer> = Vec::with_capacity(graph.tensors().len());
        for tensor in graph.tensors() {
            if let Some(source) = tensor.alias_of() {
                buffers.push(buffers[source].clone());
            } else {
                buffers.push(
                    session
                        .allocate_with_category(
                            tensor.view().span_bytes(),
                            AllocationCategory::Workspace,
                        )
                        .expect("test buffer allocation"),
                );
            }
        }
        buffers
    }

    fn test_request(
        token_count: u64,
        state_capacity: u64,
    ) -> (Ministral3ExecutionRequest, Arc<RecordingAdapter>) {
        let graph =
            build_ministral3_text_graph(token_count, 0, state_capacity).expect("test graph builds");
        let adapter = Arc::new(RecordingAdapter::default());
        let session = Arc::new(ExecutionSession::new(
            "ministral3-host-test",
            adapter.clone(),
        ));
        let queue = session.create_queue().expect("test queue");
        let buffers = graph_buffers(&session, &graph);
        let kv_states = (0..MINISTRAL3_GRAPH_LAYER_COUNT)
            .map(|layer| {
                let descriptor = KvStateDescriptor::new_with_storage(
                    layer,
                    state_capacity,
                    MINISTRAL3_GRAPH_KV_HEADS as usize,
                    MINISTRAL3_GRAPH_HEAD_DIM as usize,
                    KvCacheEncoding::Fp16,
                )
                .expect("test KV descriptor");
                session.create_kv_state(descriptor).expect("test KV state")
            })
            .collect();
        let workspace_bytes = graph_workspace_bytes(&graph).expect("test workspace bytes");
        let plan = synthetic_plan(&graph);
        let inner = Arc::new(ResidentInner {
            session,
            queue,
            plan,
            completion_timeout: Duration::from_secs(1),
            weights: BTreeMap::new(),
            model_fingerprint: "test".to_owned(),
        });
        (
            Ministral3ExecutionRequest {
                resident: inner,
                buffers,
                initial_graph: graph,
                kv_states,
                state_capacity,
                committed_length: 0,
                poisoned: false,
                workspace_bytes,
                last_audit: None,
            },
            adapter,
        )
    }

    #[test]
    fn admission_and_workspace_contract_cover_non_aligned_rows() {
        assert!(validate_request_admission(3, 17).is_ok());
        assert!(validate_request_admission(17, 17).is_ok());
        for (rows, capacity) in [
            (0, 1),
            (1, 0),
            (17, 16),
            (1, MINISTRAL3_GRAPH_MAX_CONTEXT + 1),
        ] {
            assert!(validate_request_admission(rows, capacity).is_err());
        }

        for token_count in [1, 3, 17] {
            let graph = build_ministral3_text_graph(token_count, 0, token_count)
                .expect("non-aligned graph");
            assert_eq!(graph.nodes().len(), 499);
            let terminal = graph
                .tensors()
                .iter()
                .find(|tensor| tensor.label() == "final_norm.terminal")
                .expect("terminal view");
            let logits = graph
                .tensors()
                .iter()
                .find(|tensor| tensor.label() == "logits")
                .expect("terminal logits");
            let argmax = graph
                .tensors()
                .iter()
                .find(|tensor| tensor.label() == "argmax.output")
                .expect("terminal argmax");
            assert_eq!(terminal.view().shape(), [1, MINISTRAL3_GRAPH_HIDDEN_SIZE]);
            assert_eq!(logits.view().shape(), [1, MINISTRAL3_GRAPH_VOCAB_SIZE]);
            assert_eq!(argmax.view().shape(), [1]);
            assert_eq!(
                graph_workspace_bytes(&graph).unwrap(),
                graph
                    .tensors()
                    .iter()
                    .filter(|tensor| {
                        tensor.class() != Ministral3TensorClass::Weight
                            && !tensor.is_zero_copy_alias()
                    })
                    .map(|tensor| tensor.view().span_bytes())
                    .sum::<u64>()
            );
        }
    }

    #[test]
    fn aliases_reuse_exact_buffers_and_terminal_subview_has_nonzero_offset() {
        let graph = build_ministral3_text_graph(17, 0, 17).expect("graph builds");
        let adapter = Arc::new(RecordingAdapter::default());
        let session = ExecutionSession::new("ministral3-host-test", adapter);
        let buffers = graph_buffers(&session, &graph);
        let terminal = graph
            .tensors()
            .iter()
            .find(|tensor| tensor.label() == "final_norm.terminal")
            .expect("terminal view");
        assert!(terminal.view().byte_offset() > 0);
        validate_alias_buffers(&graph, &buffers).expect("exact alias layout");

        let mut missing = buffers.clone();
        let last_alias = graph
            .tensors()
            .iter()
            .filter(|tensor| tensor.is_zero_copy_alias())
            .map(Ministral3GraphTensor::id)
            .max()
            .expect("alias tensor");
        missing.truncate(last_alias);
        assert!(validate_alias_buffers(&graph, &missing).is_err());
    }

    #[test]
    fn host_executor_returns_one_token_for_prefill_and_decode_and_poison_is_sticky() {
        let (mut request, _adapter) = test_request(17, 32);
        let prefill = request.prefill(&[0; 17]).expect("host contract prefill");
        assert_eq!(prefill.token_ids(), [0]);
        assert_eq!(prefill.committed_length(), 17);
        assert_eq!(prefill.audit().target(), "ministral3-host-test");

        let decode = request.decode(1).expect("host contract decode");
        assert_eq!(decode.token_ids(), [0]);
        assert_eq!(decode.committed_length(), 18);

        let (mut failing, adapter) = test_request(3, 8);
        adapter.fail_submit.store(true, Ordering::Relaxed);
        assert!(failing.prefill(&[0; 3]).is_err());
        assert!(matches!(
            failing.decode(1),
            Err(Ministral3ExecutionError::Invalid(message))
                if message.contains("poisoned")
        ));
        assert_eq!(failing.committed_length(), 0);
    }
}
