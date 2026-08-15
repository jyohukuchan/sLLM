//! Concrete tensor and buffer layout for the reviewed Gemma 4 graph.
//!
//! The structural graph deliberately does not own backend resources. This
//! module turns that graph into exact semantic descriptors and backing
//! identities before a backend is allowed to allocate or prepare anything.
//! In particular, decode K/V tails are represented as checked subviews of the
//! same request-state buffers later consumed as committed attention prefixes.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use crate::gemma4::Gemma4LayerType;
use crate::gemma4_graph::{GEMMA4_HIDDEN_SIZE, Gemma4Graph, Gemma4GraphNodeKind, Gemma4NormRole};
use crate::op::{RmsNormContract, SemanticOpDescriptor, SemanticOpKind};
use crate::prepared_execution::{
    ExecutionAuditAccumulator, ExecutionBoundaryKind, ExecutionSegment, PreparedExecutionAudit,
    require_terminal_success,
};
use crate::weights::{WeightClassification, WeightLoadEntry, WeightLoadPlan};
use crate::{
    AccessMode, AllocationCategory, BoundSemanticOp, DType, Encoding, ExecutionBuffer,
    ExecutionQueue, ExecutionSession, ExecutionState, OwnedTensorBinding, PrepareSupport,
    TensorDType, TensorView, VerifiedCache, WeightUploadRequest, upload_verified_weight,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
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
}

/// Immutable Gemma weights and constants retained across request owners.
pub struct Gemma4ResidentModel {
    inner: Arc<Gemma4ResidentInner>,
}

struct Gemma4ResidentInner {
    session: Arc<ExecutionSession>,
    queue: ExecutionQueue,
    lock: crate::Gemma4ModelLock,
    plan: WeightLoadPlan,
    immutable: BTreeMap<String, (Gemma4TensorBacking, ExecutionBuffer)>,
    completion_timeout: Duration,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4ExecutionOutput {
    token_ids: Vec<i32>,
    last_logits: Option<Vec<f32>>,
    state: crate::Gemma4RequestStateSnapshot,
    audit: PreparedExecutionAudit,
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
    binding_generation: u64,
    audit: Option<Gemma4ExecutionAudit>,
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

impl Gemma4ResidentModel {
    /// Uploads the immutable BF16 model and derived constants exactly once.
    pub fn new(
        session: Arc<ExecutionSession>,
        lock: crate::Gemma4ModelLock,
        plan: WeightLoadPlan,
        cache: &VerifiedCache,
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
        let layout = build_gemma4_execution_layout(&graph, &plan)?;
        let queue = session
            .create_queue()
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let buffers = provision_gemma4_execution_buffers(Arc::clone(&session), &layout)?;
        buffers.upload_immutable(&layout, &plan, cache, &queue, completion_timeout)?;
        let immutable = buffers.immutable_buffers(&layout)?;
        drop(buffers);
        Ok(Self {
            inner: Arc::new(Gemma4ResidentInner {
                session,
                queue,
                lock,
                plan,
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
        if prefill_token_count == 0 || state_capacity < prefill_token_count {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma request token count or capacity is invalid",
            ));
        }
        let graph = crate::build_gemma4_graph(
            &self.inner.lock,
            &self.inner.plan,
            prefill_token_count,
            0,
            state_capacity,
        )
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let layout = build_gemma4_execution_layout(&graph, &self.inner.plan)?;
        let buffers = provision_gemma4_request_buffers(
            Arc::clone(&self.inner.session),
            &layout,
            &self.inner.immutable,
        )?;
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
            binding_generation: 0,
            audit: None,
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
        self.prefill_impl(token_ids, false)
    }

    pub fn prefill_with_last_logits(
        &mut self,
        token_ids: &[i32],
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.prefill_impl(token_ids, true)
    }

    fn prefill_impl(
        &mut self,
        token_ids: &[i32],
        include_last_logits: bool,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        if self.committed_length != 0 || token_ids.len() as u64 != layout_token_count(&self.layout)?
        {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma prefill length or lifecycle differs",
            ));
        }
        self.buffers.upload_transition_inputs(
            &self.layout,
            &self.queue,
            token_ids,
            self.completion_timeout,
        )?;
        let state_capacity = self.state_capacity()?;
        let graph = crate::build_gemma4_graph(
            &self.lock,
            &self.plan,
            token_ids.len() as u64,
            0,
            state_capacity,
        )
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        self.run_graph(graph, include_last_logits)
    }

    pub fn decode(
        &mut self,
        token_id: i32,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.decode_impl(token_id, false)
    }

    pub fn decode_with_last_logits(
        &mut self,
        token_id: i32,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.decode_impl(token_id, true)
    }

    fn decode_impl(
        &mut self,
        token_id: i32,
        include_last_logits: bool,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        if self.committed_length == 0 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Gemma decode cannot precede prefill",
            ));
        }
        let graph = crate::build_gemma4_graph(
            &self.lock,
            &self.plan,
            1,
            self.committed_length,
            self.state_capacity()?,
        )
        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let layout = build_gemma4_execution_layout(&graph, &self.plan)?;
        let buffers = self.buffers.rebind_transition(&self.layout, &layout)?;
        buffers.upload_transition_inputs(
            &layout,
            &self.queue,
            &[token_id],
            self.completion_timeout,
        )?;
        self.layout = layout;
        self.buffers = buffers;
        self.run_graph(graph, include_last_logits)
    }

    pub fn cancel(&self) {
        self.state.cancel();
    }

    pub const fn committed_length(&self) -> u64 {
        self.committed_length
    }

    pub fn audit_snapshot(&self) -> Result<Gemma4ExecutionAudit, Gemma4ExecutionLayoutError> {
        self.audit
            .clone()
            .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("Gemma execution audit is empty"))
    }

    pub fn memory_snapshot(&self) -> crate::AllocationSnapshot {
        self.session.memory_snapshot()
    }

    fn run_graph(
        &mut self,
        graph: Gemma4Graph,
        include_last_logits: bool,
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
        let output = if include_last_logits {
            self.buffers.execute_transition_with_last_logits(
                &graph,
                &self.layout,
                &self.queue,
                &self.state,
                options,
            )
        } else {
            self.buffers
                .execute_transition(&graph, &self.layout, &self.queue, &self.state, options)
        }?;
        self.committed_length = output.state().committed_length;
        self.record_audit(output.audit())?;
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

    /// Reuses immutable model allocations, constants, and request K/V state
    /// for a new shape/position layout. Transition-local workspace, token,
    /// and position buffers are freshly allocated. This is the production
    /// prefill-to-decode ownership boundary; weights and published K/V are not
    /// uploaded or copied through the host.
    pub fn rebind_transition(
        &self,
        current_layout: &Gemma4ExecutionLayout,
        next_layout: &Gemma4ExecutionLayout,
    ) -> Result<Self, Gemma4ExecutionLayoutError> {
        if current_layout.model_fingerprint != next_layout.model_fingerprint
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
                | Gemma4TensorBacking::RequestKv { .. } => {
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
                Gemma4TensorBacking::Workspace
                | Gemma4TensorBacking::TokenIds
                | Gemma4TensorBacking::Positions => self
                    .session
                    .allocate_with_category(tensor.view.end_offset(), AllocationCategory::Workspace)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
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
            let length = entry.source_range[1]
                .checked_sub(entry.source_range[0])
                .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("weight range underflow"))?;
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
            let destination = self
                .buffer(tensor.id)?
                .range(0, length)
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
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
        if token_ids.len() as u64 != layout_token_count(layout)?
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
        self.execute_transition_impl(graph, layout, queue, request_state, options, false)
    }

    pub fn execute_transition_with_last_logits(
        &self,
        graph: &Gemma4Graph,
        layout: &Gemma4ExecutionLayout,
        queue: &ExecutionQueue,
        request_state: &crate::Gemma4RequestState,
        options: Gemma4ExecutionOptions,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        self.execute_transition_impl(graph, layout, queue, request_state, options, true)
    }

    fn execute_transition_impl(
        &self,
        graph: &Gemma4Graph,
        layout: &Gemma4ExecutionLayout,
        queue: &ExecutionQueue,
        request_state: &crate::Gemma4RequestState,
        options: Gemma4ExecutionOptions,
        include_last_logits: bool,
    ) -> Result<Gemma4ExecutionOutput, Gemma4ExecutionLayoutError> {
        if options.completion_timeout.is_zero()
            || queue.session_id() != self.session.id()
            || graph.lock_fingerprint() != layout.model_fingerprint
            || graph.weight_plan_digest() != &layout.plan_digest
            || graph.nodes().len() != layout.nodes.len()
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
        let mut pending = ExecutionSegment::default();
        let mut audit = ExecutionAuditAccumulator::new(options.expected_backend);
        let mut terminal_bytes = None;

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

                for append in self.bind_kv_appends(layout, node)? {
                    let submission = self.submit_bound(append, queue)?;
                    pending
                        .retain_semantic(format!("{}.kv_append", graph_node.label()), submission);
                }

                let operation = self.bind_node(layout, node)?;
                let mut submission = self.submit_bound(operation, queue)?;
                let boundary = planned.boundary_after();
                if boundary.is_none() {
                    pending.retain_semantic(graph_node.label(), submission);
                    return Ok(());
                }
                require_terminal_success(
                    graph_node.label(),
                    submission
                        .wait(options.completion_timeout)
                        .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?,
                )
                .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;

                if boundary == Some(ExecutionBoundaryKind::TerminalReadback) {
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
                pending.retain_semantic(graph_node.label(), submission);
                let boundary = boundary.expect("checked boundary presence");
                pending
                    .flush(boundary, &mut audit)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
                transition
                    .complete_boundary(boundary)
                    .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
                Ok(())
            })
            .map_err(|error: Gemma4ExecutionLayoutError| error)?;

        if !pending.is_empty() {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "execution ended with an unclosed segment",
            ));
        }
        let bytes = terminal_bytes.ok_or_else(|| {
            Gemma4ExecutionLayoutError::invalid("execution did not publish Argmax output")
        })?;
        if bytes.len() % 4 != 0 {
            return Err(Gemma4ExecutionLayoutError::invalid(
                "Argmax output byte length is not i32 aligned",
            ));
        }
        let token_ids = bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let audit = audit
            .snapshot()
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        let last_logits = if include_last_logits {
            Some(self.read_last_logits(layout, queue, options.completion_timeout)?)
        } else {
            None
        };
        let state = transition
            .commit()
            .map_err(|error| Gemma4ExecutionLayoutError::invalid(error.to_string()))?;
        Ok(Gemma4ExecutionOutput {
            token_ids,
            last_logits,
            state,
            audit,
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

    fn submit_bound(
        &self,
        operation: Arc<BoundSemanticOp>,
        queue: &ExecutionQueue,
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
            .session
            .prepare(operation)
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
                    tensor.view.payload_bytes(),
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
    Ok(Gemma4ProvisionedBuffers { session, buffers })
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
    Ok(Gemma4ProvisionedBuffers { session, buffers })
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

fn layout_token_count(layout: &Gemma4ExecutionLayout) -> Result<u64, Gemma4ExecutionLayoutError> {
    layout
        .tensors
        .iter()
        .find(|tensor| tensor.backing == Gemma4TensorBacking::TokenIds)
        .and_then(|tensor| tensor.view.shape().first().copied())
        .and_then(|count| u64::try_from(count).ok())
        .ok_or_else(|| Gemma4ExecutionLayoutError::invalid("token tensor shape is absent"))
}

pub fn build_gemma4_execution_layout(
    graph: &Gemma4Graph,
    plan: &WeightLoadPlan,
) -> Result<Gemma4ExecutionLayout, Gemma4ExecutionLayoutError> {
    if graph.lock_fingerprint() != plan.lock_fingerprint
        || graph.weight_plan_digest() != plan.digest()
    {
        return Err(Gemma4ExecutionLayoutError::invalid(
            "graph and weight-plan identity differ",
        ));
    }
    let mut builder = LayoutBuilder::new(graph, plan)?;
    builder.build()?;
    builder.finish()
}

struct LayoutBuilder<'a> {
    graph: &'a Gemma4Graph,
    plan: &'a WeightLoadPlan,
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
    ) -> Result<Self, Gemma4ExecutionLayoutError> {
        let mut builder = Self {
            graph,
            plan,
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
            let view = tensor_view(entry.dtype, &entry.shape, 0)?;
            let id = builder.push_tensor(
                entry.tensor_name.clone(),
                view,
                Gemma4TensorBacking::ModelWeight {
                    tensor_name: entry.tensor_name.clone(),
                },
            )?;
            builder.weights.insert(entry.tensor_name.clone(), id);
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
        Ok(Gemma4ExecutionLayout {
            model_fingerprint: self.graph.lock_fingerprint().to_owned(),
            plan_digest: *self.plan.digest(),
            tensors: self.tensors,
            nodes: self.nodes,
            model_weight_bytes: self.plan.total_destination_bytes,
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
                .semantic_contract(self.graph.start_position(), self.graph.token_count())
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
}
