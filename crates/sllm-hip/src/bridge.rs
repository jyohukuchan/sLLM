//! Additive `sllm-core` owned-execution adapter for the typed public HIP
//! BF16 copy/add, matmul, and RMSNorm paths. It does not alter the legacy `Backend`
//! control-plane methods.
//!
//! The adapter contains no alternate ABI or kernel path.  It only lowers core
//! owned bindings into the existing `Context`/`Queue`/`Buffer`/
//! typed prepared-operation/submission wrappers.

use std::sync::Arc;
#[cfg(test)]
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use sllm_hip_sys as sys;

use sllm_core::{
    AdapterResource, BoundSemanticOp, BufferRange, CausalAttentionDescriptor, DispatchEvidence,
    ExecutionAdapterAccess, ExecutionCausalAttentionSubmissionAdapter, ExecutionError,
    ExecutionKvStateSubmissionAdapter, ExecutionLinearAttentionSubmissionAdapter,
    ExecutionQueueFenceAdapter, ExecutionReadbackAdapter, ExecutionSession,
    ExecutionSessionAdapter, ExecutionSessionRequest, ExecutionState, ExecutionSubmissionAdapter,
    ExecutionTransferAdapter, OwnedTensorBinding, PrepareSupport, PreparedOperation,
    QueueCompletionMode as CoreQueueCompletionMode, ShutdownReport,
};

use crate::argmax::{ArgmaxDispatchInfo, ArgmaxSubmission, PreparedArgmax};
use crate::kv_state::{
    CausalAttentionCompletion, CausalAttentionEvidence, KvAppendCompletion, KvAppendEvidence,
    KvStateResource,
};
use crate::linear_attention::{
    LinearAttentionCompletion, LinearAttentionEvidence, LinearAttentionStateResource,
};
use crate::runtime::logical_gcn_arch_name;
use crate::{
    ArgmaxDescriptor, AttentionPreprocessDescriptor, AttentionPreprocessDispatchInfo,
    AttentionPreprocessSubmission, Buffer, Completion, CompletionState, Context,
    ElementwiseDescriptor, ElementwiseDispatchInfo, ElementwiseSubmission, EmbeddingDescriptor,
    EmbeddingDispatchInfo, EmbeddingSubmission, HipBackend, MatmulDescriptor, MatmulDispatchInfo,
    MatmulSubmission, MoeExpertDescriptor, MoeExpertDispatchInfo, MoeExpertSubmission,
    MoeRouteDescriptor, MoeRouteLayout, MoeRouteSubmission, PreparedAttentionPreprocess,
    PreparedElementwise, PreparedEmbedding, PreparedMatmul, PreparedMoeExpert, PreparedMoeRoute,
    PreparedRmsNorm, PreparedRotary, PreparedWindowedAttention, Queue,
    QueueCompletionMode as HipQueueCompletionMode, RmsNormDescriptor, RmsNormDispatchInfo,
    RmsNormSubmission, RotaryDescriptor, RotaryDispatchInfo, RotarySubmission, RuntimeError,
    RuntimeStatus, WindowedAttentionDescriptor, WindowedAttentionDispatchInfo,
    WindowedAttentionSubmission, moe_expert_workspace_bytes,
};

const HIP_BACKEND_NAME: &str = "hip";
const CLEANUP_ATTEMPT_CAP: usize = 16;

pub(crate) fn open_execution_session(
    backend: HipBackend,
    request: ExecutionSessionRequest,
) -> Result<Arc<ExecutionSession>, ExecutionError> {
    let device = Context::query_device(request.device_index()).map_err(map_backend_error)?;
    let context = Context::create(request.device_index(), request.expected_target())
        .map_err(map_backend_error)?;
    let adapter = Arc::new(HipExecutionSession {
        state: Arc::new(HipSessionState::new()),
        backend,
        context,
        available_memory_bytes: device.available_memory_bytes,
    });
    Ok(Arc::new(ExecutionSession::new(HIP_BACKEND_NAME, adapter)))
}

struct HipSessionState {
    activity: ActiveSessionState,
}

impl HipSessionState {
    fn new() -> Self {
        Self {
            activity: ActiveSessionState::new(),
        }
    }

    fn ensure_open(&self) -> Result<(), ExecutionError> {
        self.activity.ensure_open()
    }

    fn acquire_active(self: &Arc<Self>) -> Result<ActiveOperation, ExecutionError> {
        self.activity.acquire_active()?;
        Ok(ActiveOperation {
            state: Arc::clone(self),
            active: true,
        })
    }

    fn begin_shutdown(&self) -> Result<(), ExecutionError> {
        self.activity.begin_shutdown()
    }

    #[cfg(test)]
    fn active_count(&self) -> usize {
        self.activity.active_count()
    }
}

struct ActiveSessionState {
    // The high bit is the closing state; the remaining bits are the active
    // operation count.  Admission, closing, and release all update this one
    // word, so shutdown cannot observe zero and then race a stale admission.
    lifecycle: AtomicUsize,
    #[cfg(test)]
    admission_gate: Mutex<Option<AdmissionGate>>,
}

impl ActiveSessionState {
    fn new() -> Self {
        Self {
            lifecycle: AtomicUsize::new(0),
            #[cfg(test)]
            admission_gate: Mutex::new(None),
        }
    }

    const CLOSING_BIT: usize = 1usize << (usize::BITS - 1);
    const ACTIVE_MASK: usize = Self::CLOSING_BIT - 1;

    fn ensure_open(&self) -> Result<(), ExecutionError> {
        if self.lifecycle.load(Ordering::Acquire) & Self::CLOSING_BIT != 0 {
            Err(ExecutionError::Closing)
        } else {
            Ok(())
        }
    }

    fn acquire_active(&self) -> Result<(), ExecutionError> {
        let mut observed = self.lifecycle.load(Ordering::Acquire);
        loop {
            if observed & Self::CLOSING_BIT != 0 {
                return Err(ExecutionError::Closing);
            }
            let active = observed & Self::ACTIVE_MASK;
            if active == Self::ACTIVE_MASK {
                return Err(ExecutionError::Busy);
            }
            #[cfg(test)]
            self.pause_before_admission_cas();
            match self.lifecycle.compare_exchange(
                observed,
                observed + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(next) => observed = next,
            }
        }
    }

    fn begin_shutdown(&self) -> Result<(), ExecutionError> {
        let mut observed = self.lifecycle.load(Ordering::Acquire);
        loop {
            if observed & Self::CLOSING_BIT != 0 {
                return if observed & Self::ACTIVE_MASK == 0 {
                    Ok(())
                } else {
                    Err(ExecutionError::Busy)
                };
            }
            let active = observed & Self::ACTIVE_MASK;
            match self.lifecycle.compare_exchange(
                observed,
                observed | Self::CLOSING_BIT,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return if active == 0 {
                        Ok(())
                    } else {
                        Err(ExecutionError::Busy)
                    };
                }
                Err(next) => observed = next,
            }
        }
    }

    fn release_active(&self) {
        let mut observed = self.lifecycle.load(Ordering::Acquire);
        loop {
            let active = observed & Self::ACTIVE_MASK;
            if active == 0 {
                // A ticket is single-owner and should make this path
                // unreachable.  If an invariant is ever violated, close the
                // state instead of wrapping the counter and admitting work.
                self.lifecycle.fetch_or(Self::CLOSING_BIT, Ordering::AcqRel);
                return;
            }
            match self.lifecycle.compare_exchange(
                observed,
                observed - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(next) => observed = next,
            }
        }
    }

    #[cfg(test)]
    fn active_count(&self) -> usize {
        self.lifecycle.load(Ordering::Acquire) & Self::ACTIVE_MASK
    }

    #[cfg(test)]
    fn pause_next_admission(
        &self,
        reached: Arc<std::sync::Barrier>,
        proceed: Arc<std::sync::Barrier>,
    ) {
        *self.admission_gate.lock().expect("admission gate lock") =
            Some(AdmissionGate { reached, proceed });
    }

    #[cfg(test)]
    fn pause_before_admission_cas(&self) {
        let gate = self
            .admission_gate
            .lock()
            .expect("admission gate lock")
            .take();
        if let Some(gate) = gate {
            gate.reached.wait();
            gate.proceed.wait();
        }
    }
}

#[cfg(test)]
struct AdmissionGate {
    reached: Arc<std::sync::Barrier>,
    proceed: Arc<std::sync::Barrier>,
}

struct ActiveOperation {
    state: Arc<HipSessionState>,
    active: bool,
}

impl Drop for ActiveOperation {
    fn drop(&mut self) {
        if self.active {
            self.state.activity.release_active();
        }
    }
}

struct HipExecutionSession {
    state: Arc<HipSessionState>,
    backend: HipBackend,
    context: Context,
    available_memory_bytes: u64,
}

impl ExecutionSessionAdapter for HipExecutionSession {
    fn max_transfer_bytes(&self) -> u64 {
        crate::sys::SLLM_HIP_MAX_TRANSFER_BYTES
    }

    fn available_memory_bytes(&self) -> Option<u64> {
        Some(self.available_memory_bytes)
    }

    fn supports(&self, descriptor: &sllm_core::SemanticOpDescriptor) -> PrepareSupport {
        if let Err(error) = self.state.ensure_open() {
            return PrepareSupport::Unsupported {
                reason: error.to_string(),
            };
        }
        if let Err(error) = descriptor.validate() {
            return PrepareSupport::Unsupported {
                reason: format!("invalid semantic descriptor: {error}"),
            };
        }
        if !matches!(
            descriptor.kind(),
            sllm_core::SemanticOpKind::Copy
                | sllm_core::SemanticOpKind::Add
                | sllm_core::SemanticOpKind::ScalarMul
                | sllm_core::SemanticOpKind::SiluMul
                | sllm_core::SemanticOpKind::GeluTanhMul
                | sllm_core::SemanticOpKind::SigmoidMul
                | sllm_core::SemanticOpKind::TanhSoftcap
                | sllm_core::SemanticOpKind::Embedding
                | sllm_core::SemanticOpKind::Matmul
                | sllm_core::SemanticOpKind::RmsNorm
                | sllm_core::SemanticOpKind::Argmax
                | sllm_core::SemanticOpKind::AttentionPreprocess
                | sllm_core::SemanticOpKind::Rotary
                | sllm_core::SemanticOpKind::CausalAttention
                | sllm_core::SemanticOpKind::SparseMoe
        ) {
            return PrepareSupport::Unsupported {
                reason: "the HIP owned execution bridge does not support this semantic operation"
                    .to_owned(),
            };
        }
        PrepareSupport::Supported
    }

    fn create_queue(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
    ) -> Result<AdapterResource, ExecutionError> {
        self.state.ensure_open()?;
        Queue::create(&self.context)
            .map(AdapterResource::new)
            .map_err(map_backend_error)
    }

    fn set_queue_completion_mode(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        queue: &sllm_core::ExecutionQueue,
        mode: CoreQueueCompletionMode,
    ) -> Result<(), ExecutionError> {
        self.state.ensure_open()?;
        let queue = access.downcast_queue_payload::<Queue>(queue)?;
        let mode = match mode {
            CoreQueueCompletionMode::Profiled => HipQueueCompletionMode::Profiled,
            CoreQueueCompletionMode::Deferred => HipQueueCompletionMode::Deferred,
        };
        queue.set_completion_mode(mode).map_err(map_backend_error)
    }

    fn create_queue_fence(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        queue: &sllm_core::ExecutionQueue,
    ) -> Result<Box<dyn ExecutionQueueFenceAdapter>, ExecutionError> {
        self.state.ensure_open()?;
        let ticket = self.state.acquire_active()?;
        let completion = access
            .downcast_queue_payload::<Queue>(queue)?
            .fence()
            .map_err(map_backend_error)?;
        Ok(Box::new(HipQueueFence {
            completion,
            _ticket: ticket,
        }))
    }

    fn allocate(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
        size_bytes: u64,
    ) -> Result<AdapterResource, ExecutionError> {
        self.state.ensure_open()?;
        Buffer::allocate(&self.context, size_bytes)
            .map(AdapterResource::new)
            .map_err(map_backend_error)
    }

    fn create_kv_state(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state_id: sllm_core::KvStateId,
        descriptor: sllm_core::KvStateDescriptor,
    ) -> Result<AdapterResource, ExecutionError> {
        self.state.ensure_open()?;
        KvStateResource::create(&self.context, access.session_id(), state_id, descriptor)
            .map(AdapterResource::new)
            .map_err(map_backend_error)
    }

    fn kv_state_snapshot(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::KvState,
    ) -> Result<sllm_core::KvStateSnapshot, ExecutionError> {
        self.state.ensure_open()?;
        access
            .downcast_kv_state_payload::<KvStateResource>(state)?
            .snapshot()
            .map_err(map_backend_error)
    }

    fn readback_kv_state(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::KvState,
        plane: u32,
        byte_offset: u64,
        destination: &mut [u8],
    ) -> Result<(), ExecutionError> {
        self.state.ensure_open()?;
        access
            .downcast_kv_state_payload::<KvStateResource>(state)?
            .readback(plane, byte_offset, destination)
            .map_err(map_backend_error)
    }

    fn rewind_last_kv_state_transition(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::KvState,
        expected_length: u64,
        rewind_length: u64,
    ) -> Result<(), ExecutionError> {
        self.state.ensure_open()?;
        access
            .downcast_kv_state_payload::<KvStateResource>(state)?
            .rewind_last(expected_length, rewind_length)
            .map_err(map_backend_error)
    }

    fn append_kv_state(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::KvState,
        queue: &sllm_core::ExecutionQueue,
        key: &OwnedTensorBinding,
        value: &OwnedTensorBinding,
        request: &sllm_core::KvStateAppendRequest,
    ) -> Result<(Box<dyn ExecutionKvStateSubmissionAdapter>, DispatchEvidence), ExecutionError>
    {
        self.state.ensure_open()?;
        let state_resource = access
            .downcast_kv_state_payload::<KvStateResource>(state)?
            .clone();
        let queue = access.downcast_queue_payload::<Queue>(queue)?.clone();
        let key_buffer = access
            .downcast_buffer_payload::<Buffer>(key.buffer())?
            .clone();
        let value_buffer = access
            .downcast_buffer_payload::<Buffer>(value.buffer())?
            .clone();
        let key = key_buffer.binding(key.view().clone());
        let value = value_buffer.binding(value.view().clone());
        let ticket = self.state.acquire_active()?;
        let (completion, evidence) = match state_resource.append(&queue, &key, &value, *request) {
            Ok(result) => result,
            Err(error) => {
                drop(ticket);
                return Err(map_backend_error(error));
            }
        };
        Ok((
            Box::new(HipKvSubmission {
                completion,
                _ticket: ticket,
            }),
            dispatch_from_kv_append(evidence),
        ))
    }

    fn execute_causal_attention(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::KvState,
        queue: &sllm_core::ExecutionQueue,
        query: &OwnedTensorBinding,
        output: &OwnedTensorBinding,
        descriptor: CausalAttentionDescriptor,
    ) -> Result<
        (
            Box<dyn ExecutionCausalAttentionSubmissionAdapter>,
            DispatchEvidence,
        ),
        ExecutionError,
    > {
        self.state.ensure_open()?;
        let state_resource = access
            .downcast_kv_state_payload::<KvStateResource>(state)?
            .clone();
        let queue = access.downcast_queue_payload::<Queue>(queue)?.clone();
        let query_buffer = access
            .downcast_buffer_payload::<Buffer>(query.buffer())?
            .clone();
        let output_buffer = access
            .downcast_buffer_payload::<Buffer>(output.buffer())?
            .clone();
        let query = query_buffer.binding(query.view().clone());
        let output = output_buffer.binding(output.view().clone());
        let ticket = self.state.acquire_active()?;
        let (completion, evidence) = match state_resource.causal_attention(
            &queue,
            &query,
            &output,
            descriptor.start_position(),
            descriptor.expected_kv_length(),
        ) {
            Ok(value) => value,
            Err(error) => {
                drop(ticket);
                return Err(map_backend_error(error));
            }
        };
        Ok((
            Box::new(HipCausalAttentionSubmission {
                completion,
                _evidence: evidence.clone(),
                _ticket: ticket,
            }),
            dispatch_from_causal_attention(evidence),
        ))
    }

    fn create_linear_attention_state(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state_id: sllm_core::LinearAttentionStateId,
        descriptor: sllm_core::LinearAttentionStateDescriptor,
    ) -> Result<AdapterResource, ExecutionError> {
        self.state.ensure_open()?;
        LinearAttentionStateResource::create(
            &self.context,
            access.session_id(),
            state_id,
            descriptor,
        )
        .map(AdapterResource::new)
        .map_err(map_backend_error)
    }

    fn linear_attention_state_snapshot(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::LinearAttentionState,
    ) -> Result<sllm_core::LinearAttentionStateSnapshot, ExecutionError> {
        self.state.ensure_open()?;
        access
            .downcast_linear_attention_state_payload::<LinearAttentionStateResource>(state)?
            .snapshot()
            .map_err(map_backend_error)
    }

    fn rewind_last_linear_attention_transition(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::LinearAttentionState,
        expected_length: u64,
        rewind_length: u64,
    ) -> Result<(), ExecutionError> {
        self.state.ensure_open()?;
        access
            .downcast_linear_attention_state_payload::<LinearAttentionStateResource>(state)?
            .rewind_last(expected_length, rewind_length)
            .map_err(map_backend_error)
    }

    fn execute_linear_attention(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        state: &sllm_core::LinearAttentionState,
        queue: &sllm_core::ExecutionQueue,
        bindings: &sllm_core::LinearAttentionBindings,
        request: sllm_core::LinearAttentionRequest,
    ) -> Result<
        (
            Box<dyn ExecutionLinearAttentionSubmissionAdapter>,
            DispatchEvidence,
        ),
        ExecutionError,
    > {
        self.state.ensure_open()?;
        let state_resource = access
            .downcast_linear_attention_state_payload::<LinearAttentionStateResource>(state)?
            .clone();
        let queue = access.downcast_queue_payload::<Queue>(queue)?.clone();
        let owned = [
            bindings.qkv(),
            bindings.z(),
            bindings.b_input(),
            bindings.a_input(),
            bindings.conv_weight(),
            bindings.a_log(),
            bindings.dt_bias(),
            bindings.norm_weight(),
            bindings.output(),
        ];
        let buffers = owned
            .map(|binding| {
                access
                    .downcast_buffer_payload::<Buffer>(binding.buffer())
                    .cloned()
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let native: [crate::TensorBinding; 9] =
            std::array::from_fn(|index| buffers[index].binding(owned[index].view().clone()));
        let references: [&crate::TensorBinding; 9] = std::array::from_fn(|index| &native[index]);
        let ticket = self.state.acquire_active()?;
        let (completion, evidence) = match state_resource.execute(&queue, references, request) {
            Ok(result) => result,
            Err(error) => {
                drop(ticket);
                return Err(map_backend_error(error));
            }
        };
        Ok((
            Box::new(HipLinearAttentionSubmission {
                completion,
                _evidence: evidence.clone(),
                _ticket: ticket,
            }),
            dispatch_from_linear_attention(evidence),
        ))
    }

    fn prepare(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        operation: &BoundSemanticOp,
    ) -> Result<AdapterResource, ExecutionError> {
        self.state.ensure_open()?;
        let prepared = match operation.descriptor().kind() {
            sllm_core::SemanticOpKind::RmsNorm => {
                let activation = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[0].buffer())?
                    .clone();
                let raw_scale = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[1].buffer())?
                    .clone();
                let output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[0].buffer())?
                    .clone();
                let descriptor = RmsNormDescriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    activation.binding(operation.inputs()[0].view().clone()),
                    raw_scale.binding(operation.inputs()[1].view().clone()),
                    output.binding(operation.outputs()[0].view().clone()),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::RmsNorm(
                    self.backend
                        .prepare_rms_norm(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::Copy
            | sllm_core::SemanticOpKind::Add
            | sllm_core::SemanticOpKind::ScalarMul
            | sllm_core::SemanticOpKind::SiluMul
            | sllm_core::SemanticOpKind::GeluTanhMul
            | sllm_core::SemanticOpKind::SigmoidMul
            | sllm_core::SemanticOpKind::TanhSoftcap => {
                let mut inputs = Vec::with_capacity(operation.inputs().len());
                for input in operation.inputs() {
                    let buffer = access
                        .downcast_buffer_payload::<Buffer>(input.buffer())?
                        .clone();
                    inputs.push(buffer.binding(input.view().clone()));
                }
                let output_binding = &operation.outputs()[0];
                let output = access
                    .downcast_buffer_payload::<Buffer>(output_binding.buffer())?
                    .clone();
                let descriptor = ElementwiseDescriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    inputs,
                    output.binding(output_binding.view().clone()),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::Elementwise(
                    self.backend
                        .prepare_elementwise(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::Embedding => {
                let weight = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[0].buffer())?
                    .clone();
                let token_ids = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[1].buffer())?
                    .clone();
                let output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[0].buffer())?
                    .clone();
                let descriptor = EmbeddingDescriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    weight.binding(operation.inputs()[0].view().clone()),
                    token_ids.binding(operation.inputs()[1].view().clone()),
                    output.binding(operation.outputs()[0].view().clone()),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::Embedding(
                    self.backend
                        .prepare_embedding(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::Matmul => {
                let activation = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[0].buffer())?
                    .clone();
                let weight = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[1].buffer())?
                    .clone();
                let output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[0].buffer())?
                    .clone();
                let descriptor = MatmulDescriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    activation.binding(operation.inputs()[0].view().clone()),
                    weight.binding(operation.inputs()[1].view().clone()),
                    output.binding(operation.outputs()[0].view().clone()),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::Matmul(
                    self.backend
                        .prepare_matmul(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::Argmax => {
                let logits = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[0].buffer())?
                    .clone();
                let output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[0].buffer())?
                    .clone();
                let descriptor = ArgmaxDescriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    logits.binding(operation.inputs()[0].view().clone()),
                    output.binding(operation.outputs()[0].view().clone()),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::Argmax(
                    self.backend
                        .prepare_argmax(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::AttentionPreprocess => {
                let packed_q_gate = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[0].buffer())?
                    .clone();
                let k = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[1].buffer())?
                    .clone();
                let q_raw_scale = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[2].buffer())?
                    .clone();
                let k_raw_scale = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[3].buffer())?
                    .clone();
                let positions_binding = &operation.inputs()[4];
                let positions = access
                    .downcast_buffer_payload::<Buffer>(positions_binding.buffer())?
                    .clone();
                let q_output_binding = &operation.outputs()[0];
                let gate_output_binding = &operation.outputs()[1];
                let k_output_binding = &operation.outputs()[2];
                let q_output = access
                    .downcast_buffer_payload::<Buffer>(q_output_binding.buffer())?
                    .clone();
                let gate_output = access
                    .downcast_buffer_payload::<Buffer>(gate_output_binding.buffer())?
                    .clone();
                let k_output = access
                    .downcast_buffer_payload::<Buffer>(k_output_binding.buffer())?
                    .clone();
                let descriptor = AttentionPreprocessDescriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    packed_q_gate.binding(operation.inputs()[0].view().clone()),
                    k.binding(operation.inputs()[1].view().clone()),
                    q_raw_scale.binding(operation.inputs()[2].view().clone()),
                    k_raw_scale.binding(operation.inputs()[3].view().clone()),
                    positions.binding(positions_binding.view().clone()),
                    q_output.binding(q_output_binding.view().clone()),
                    gate_output.binding(gate_output_binding.view().clone()),
                    k_output.binding(k_output_binding.view().clone()),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::AttentionPreprocess(
                    self.backend
                        .prepare_attention_preprocess(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::Rotary => {
                let query = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[0].buffer())?
                    .clone();
                let key = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[1].buffer())?
                    .clone();
                let positions = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[2].buffer())?
                    .clone();
                let query_output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[0].buffer())?
                    .clone();
                let key_output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[1].buffer())?
                    .clone();
                let descriptor = RotaryDescriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    query.binding(operation.inputs()[0].view().clone()),
                    key.binding(operation.inputs()[1].view().clone()),
                    positions.binding(operation.inputs()[2].view().clone()),
                    query_output.binding(operation.outputs()[0].view().clone()),
                    key_output.binding(operation.outputs()[1].view().clone()),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::Rotary(
                    self.backend
                        .prepare_rotary(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::CausalAttention => {
                let query = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[0].buffer())?
                    .clone();
                let key = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[1].buffer())?
                    .clone();
                let value = access
                    .downcast_buffer_payload::<Buffer>(operation.inputs()[2].buffer())?
                    .clone();
                let output = access
                    .downcast_buffer_payload::<Buffer>(operation.outputs()[0].buffer())?
                    .clone();
                let descriptor = WindowedAttentionDescriptor::from_validated_semantic(
                    Arc::clone(operation.descriptor()),
                    query.binding(operation.inputs()[0].view().clone()),
                    key.binding(operation.inputs()[1].view().clone()),
                    value.binding(operation.inputs()[2].view().clone()),
                    output.binding(operation.outputs()[0].view().clone()),
                )
                .map_err(map_backend_error)?;
                self.state.ensure_open()?;
                HipPreparedPlan::WindowedAttention(
                    self.backend
                        .prepare_windowed_attention(&self.context, descriptor)
                        .map_err(map_backend_error)?,
                )
            }
            sllm_core::SemanticOpKind::SparseMoe => {
                let hidden_owned = &operation.inputs()[0];
                let router_owned = &operation.inputs()[1];
                let blob_owned = &operation.inputs()[2];
                let output_owned = &operation.outputs()[0];
                let hidden = access
                    .downcast_buffer_payload::<Buffer>(hidden_owned.buffer())?
                    .clone();
                let router_weight = access
                    .downcast_buffer_payload::<Buffer>(router_owned.buffer())?
                    .clone();
                let layer_blob = access
                    .downcast_buffer_payload::<Buffer>(blob_owned.buffer())?
                    .clone();
                let output = access
                    .downcast_buffer_payload::<Buffer>(output_owned.buffer())?
                    .clone();
                let token_count = hidden_owned.view().shape()[0] as u64;
                let logits_view = sllm_core::TensorView::contiguous(
                    sllm_core::DType::Bf16,
                    &[token_count as usize, 256],
                )
                .map_err(|error| ExecutionError::ExecutionUnavailable {
                    backend: HIP_BACKEND_NAME,
                    reason: format!("SparseMoe logits layout: {error}"),
                })?;
                let route_layout =
                    MoeRouteLayout::new(token_count, 256, 8).map_err(map_backend_error)?;
                let route_view = sllm_core::TensorView::contiguous(
                    sllm_core::DType::U8,
                    &[route_layout.metadata_bytes as usize],
                )
                .map_err(|error| ExecutionError::ExecutionUnavailable {
                    backend: HIP_BACKEND_NAME,
                    reason: format!("SparseMoe route layout: {error}"),
                })?;
                let workspace_bytes = moe_expert_workspace_bytes(token_count).ok_or_else(|| {
                    ExecutionError::ExecutionUnavailable {
                        backend: HIP_BACKEND_NAME,
                        reason: "SparseMoe workspace size overflow".to_owned(),
                    }
                })?;
                let workspace_view = sllm_core::TensorView::contiguous(
                    sllm_core::DType::U8,
                    &[workspace_bytes as usize],
                )
                .map_err(|error| ExecutionError::ExecutionUnavailable {
                    backend: HIP_BACKEND_NAME,
                    reason: format!("SparseMoe workspace layout: {error}"),
                })?;
                let logits = Buffer::allocate(&self.context, logits_view.payload_bytes())
                    .map_err(map_backend_error)?;
                let route_metadata = Buffer::allocate(&self.context, route_view.payload_bytes())
                    .map_err(map_backend_error)?;
                let workspace = Buffer::allocate(&self.context, workspace_view.payload_bytes())
                    .map_err(map_backend_error)?;
                let matmul = MatmulDescriptor::new(
                    hidden.binding(hidden_owned.view().clone()),
                    router_weight.binding(router_owned.view().clone()),
                    logits.binding(logits_view.clone()),
                )
                .map_err(|error| ExecutionError::ExecutionUnavailable {
                    backend: HIP_BACKEND_NAME,
                    reason: format!("SparseMoe router matmul: {error}"),
                })?;
                let route = MoeRouteDescriptor::new(
                    logits.binding(logits_view),
                    route_metadata.binding(route_view.clone()),
                    8,
                )
                .map_err(map_backend_error)?;
                let expert = MoeExpertDescriptor::new(
                    hidden.binding(hidden_owned.view().clone()),
                    route_metadata.binding(route_view),
                    layer_blob.binding(blob_owned.view().clone()),
                    workspace.binding(workspace_view),
                    output.binding(output_owned.view().clone()),
                )
                .map_err(map_backend_error)?;
                HipPreparedPlan::SparseMoe(PreparedSparseMoe {
                    router: self
                        .backend
                        .prepare_matmul(&self.context, matmul)
                        .map_err(map_backend_error)?,
                    route: self
                        .backend
                        .prepare_moe_route(&self.context, route)
                        .map_err(map_backend_error)?,
                    expert: self
                        .backend
                        .prepare_moe_expert(&self.context, expert)
                        .map_err(map_backend_error)?,
                })
            }
        };
        Ok(AdapterResource::new(prepared))
    }

    fn submit(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        prepared: &PreparedOperation,
        queue: &sllm_core::ExecutionQueue,
    ) -> Result<(Box<dyn ExecutionSubmissionAdapter>, DispatchEvidence), ExecutionError> {
        self.state.ensure_open()?;
        let plan = access
            .downcast_prepared_payload::<HipPreparedPlan>(prepared)?
            .clone();
        let queue = access.downcast_queue_payload::<Queue>(queue)?.clone();
        let ticket = self.state.acquire_active()?;
        let (submission, dispatch) = match plan {
            HipPreparedPlan::RmsNorm(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::RmsNorm(submission),
                    dispatch_from_rmsnorm(dispatch),
                )
            }
            HipPreparedPlan::Elementwise(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::Elementwise(submission),
                    dispatch_from_elementwise(dispatch),
                )
            }
            HipPreparedPlan::Embedding(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::Embedding(submission),
                    dispatch_from_embedding(dispatch),
                )
            }
            HipPreparedPlan::Matmul(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::Matmul(submission),
                    dispatch_from_matmul(dispatch),
                )
            }
            HipPreparedPlan::Argmax(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::Argmax(submission),
                    dispatch_from_argmax(dispatch),
                )
            }
            HipPreparedPlan::AttentionPreprocess(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::AttentionPreprocess(submission),
                    dispatch_from_attention_preprocess(dispatch),
                )
            }
            HipPreparedPlan::Rotary(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::Rotary(submission),
                    dispatch_from_rotary(dispatch),
                )
            }
            HipPreparedPlan::WindowedAttention(plan) => {
                let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::WindowedAttention(submission),
                    dispatch_from_windowed_attention(dispatch),
                )
            }
            HipPreparedPlan::SparseMoe(plan) => {
                let (router, _) = plan.router.execute(&queue).map_err(map_backend_error)?;
                let (route, _) = plan.route.execute(&queue).map_err(map_backend_error)?;
                let (expert, dispatch) = plan.expert.execute(&queue).map_err(map_backend_error)?;
                (
                    HipSemanticSubmission::SparseMoe(SparseMoeSubmission {
                        router,
                        route,
                        expert,
                    }),
                    dispatch_from_moe_expert(dispatch),
                )
            }
        };
        Ok((
            Box::new(HipSubmission {
                submission,
                queue,
                _ticket: ticket,
            }),
            dispatch,
        ))
    }

    fn upload(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        queue: &sllm_core::ExecutionQueue,
        destination: &BufferRange,
        bytes: Arc<[u8]>,
    ) -> Result<Box<dyn ExecutionTransferAdapter>, ExecutionError> {
        self.state.ensure_open()?;
        let queue = access.downcast_queue_payload::<Queue>(queue)?.clone();
        let buffer = access
            .downcast_buffer_payload::<Buffer>(destination.buffer())?
            .clone();
        let ticket = self.state.acquire_active()?;
        let completion = queue
            .copy_to_device(&buffer, bytes.as_ref(), destination.offset_bytes())
            .map_err(map_backend_error)?;
        Ok(Box::new(HipTransfer {
            completion,
            _ticket: ticket,
        }))
    }

    fn readback(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        queue: &sllm_core::ExecutionQueue,
        source: &BufferRange,
    ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError> {
        self.state.ensure_open()?;
        let queue = access.downcast_queue_payload::<Queue>(queue)?.clone();
        let buffer = access
            .downcast_buffer_payload::<Buffer>(source.buffer())?
            .clone();
        let ticket = self.state.acquire_active()?;
        let completion = queue
            .copy_to_host(&buffer, source.size_bytes(), source.offset_bytes())
            .map_err(map_backend_error)?;
        Ok(Box::new(HipReadback {
            completion,
            _ticket: ticket,
        }))
    }

    fn shutdown(
        &self,
        _access: &ExecutionAdapterAccess<'_>,
        deadline: Duration,
    ) -> Result<ShutdownReport, ExecutionError> {
        self.state.begin_shutdown()?;
        // The native cleanup API is nonblocking and bounded.  The supplied
        // deadline chooses only its bounded retry budget; it never creates a
        // CPU fallback or releases unresolved native ownership speculatively.
        let attempts = usize::try_from(deadline.as_millis())
            .unwrap_or(CLEANUP_ATTEMPT_CAP)
            .clamp(1, CLEANUP_ATTEMPT_CAP);
        let (retryable_cleanup, durable_quarantine) =
            Context::shutdown_cleanup(attempts).map_err(map_backend_error)?;
        if durable_quarantine != 0 || Context::cleanup_accounting_error_count() != 0 {
            return Err(ExecutionError::CleanupQuarantined);
        }
        Ok(ShutdownReport {
            retryable_cleanup,
            durable_quarantine,
        })
    }
}

#[derive(Clone)]
enum HipPreparedPlan {
    RmsNorm(PreparedRmsNorm),
    Elementwise(PreparedElementwise),
    Embedding(PreparedEmbedding),
    Matmul(PreparedMatmul),
    Argmax(PreparedArgmax),
    AttentionPreprocess(PreparedAttentionPreprocess),
    Rotary(PreparedRotary),
    WindowedAttention(PreparedWindowedAttention),
    SparseMoe(PreparedSparseMoe),
}

#[derive(Clone)]
struct PreparedSparseMoe {
    router: PreparedMatmul,
    route: PreparedMoeRoute,
    expert: PreparedMoeExpert,
}

enum HipSemanticSubmission {
    RmsNorm(RmsNormSubmission),
    Elementwise(ElementwiseSubmission),
    Embedding(EmbeddingSubmission),
    Matmul(MatmulSubmission),
    Argmax(ArgmaxSubmission),
    AttentionPreprocess(AttentionPreprocessSubmission),
    Rotary(RotarySubmission),
    WindowedAttention(WindowedAttentionSubmission),
    SparseMoe(SparseMoeSubmission),
}

struct SparseMoeSubmission {
    router: MatmulSubmission,
    route: MoeRouteSubmission,
    expert: MoeExpertSubmission,
}

impl HipSemanticSubmission {
    fn query(&mut self) -> Result<CompletionState, RuntimeError> {
        match self {
            Self::RmsNorm(submission) => submission.query(),
            Self::Elementwise(submission) => submission.query(),
            Self::Embedding(submission) => submission.query(),
            Self::Matmul(submission) => submission.query(),
            Self::Argmax(submission) => submission.query(),
            Self::AttentionPreprocess(submission) => submission.query(),
            Self::Rotary(submission) => submission.query(),
            Self::WindowedAttention(submission) => submission.query(),
            Self::SparseMoe(submission) => submission.expert.query(),
        }
    }

    fn wait(&mut self, timeout: Duration) -> Result<CompletionState, RuntimeError> {
        match self {
            Self::RmsNorm(submission) => submission.wait(timeout),
            Self::Elementwise(submission) => submission.wait(timeout),
            Self::Embedding(submission) => submission.wait(timeout),
            Self::Matmul(submission) => submission.wait(timeout),
            Self::Argmax(submission) => submission.wait(timeout),
            Self::AttentionPreprocess(submission) => submission.wait(timeout),
            Self::Rotary(submission) => submission.wait(timeout),
            Self::WindowedAttention(submission) => submission.wait(timeout),
            Self::SparseMoe(submission) => submission.expert.wait(timeout),
        }
    }

    fn finalize_after_token(&mut self, fence_token: u64) -> Result<CompletionState, RuntimeError> {
        match self {
            Self::RmsNorm(submission) => submission.finalize_after_token(fence_token),
            Self::Elementwise(submission) => submission.finalize_after_token(fence_token),
            Self::Embedding(submission) => submission.finalize_after_token(fence_token),
            Self::Matmul(submission) => submission.finalize_after_token(fence_token),
            Self::Argmax(submission) => submission.finalize_after_token(fence_token),
            Self::AttentionPreprocess(submission) => submission.finalize_after_token(fence_token),
            Self::Rotary(submission) => submission.finalize_after_token(fence_token),
            Self::WindowedAttention(submission) => submission.finalize_after_token(fence_token),
            Self::SparseMoe(submission) => {
                submission.router.finalize_after_token(fence_token)?;
                submission.route.finalize_after_token(fence_token)?;
                submission.expert.finalize_after_token(fence_token)
            }
        }
    }

    fn kernel_elapsed_ns(&mut self) -> Result<u64, RuntimeError> {
        match self {
            Self::RmsNorm(submission) => submission.kernel_elapsed_ns(),
            Self::Elementwise(submission) => submission.kernel_elapsed_ns(),
            Self::Embedding(submission) => submission.kernel_elapsed_ns(),
            Self::Matmul(submission) => submission.kernel_elapsed_ns(),
            Self::Argmax(submission) => submission.kernel_elapsed_ns(),
            Self::AttentionPreprocess(submission) => submission.kernel_elapsed_ns(),
            Self::Rotary(submission) => submission.kernel_elapsed_ns(),
            Self::WindowedAttention(submission) => submission.kernel_elapsed_ns(),
            Self::SparseMoe(submission) => {
                let expert = submission.expert.kernel_elapsed_ns()?;
                let route = submission.route.kernel_elapsed_ns()?;
                let router = submission.router.kernel_elapsed_ns()?;
                Ok(router + route + expert)
            }
        }
    }
}

struct HipSubmission {
    submission: HipSemanticSubmission,
    queue: Queue,
    _ticket: ActiveOperation,
}

impl ExecutionSubmissionAdapter for HipSubmission {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        self.submission
            .query()
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        self.submission
            .wait(timeout)
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn finalize_after_fence(&mut self, fence_token: u64) -> Result<ExecutionState, ExecutionError> {
        self.submission
            .finalize_after_token(fence_token)
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn kernel_elapsed_ns(&mut self) -> Result<Option<u64>, ExecutionError> {
        self.submission
            .kernel_elapsed_ns()
            .map(Some)
            .map_err(map_async_error)
    }

    fn start_output_readback(
        &mut self,
        access: &ExecutionAdapterAccess<'_>,
        output: &OwnedTensorBinding,
    ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError> {
        let buffer = access
            .downcast_buffer_payload::<Buffer>(output.buffer())?
            .clone();
        let ticket = self._ticket.state.acquire_active()?;
        let completion = self
            .queue
            .copy_to_host(
                &buffer,
                output.view().payload_bytes(),
                output.view().byte_offset(),
            )
            .map_err(map_backend_error)?;
        Ok(Box::new(HipReadback {
            completion,
            _ticket: ticket,
        }))
    }
}

struct HipTransfer {
    completion: Completion,
    _ticket: ActiveOperation,
}

struct HipQueueFence {
    completion: Completion,
    _ticket: ActiveOperation,
}

impl ExecutionQueueFenceAdapter for HipQueueFence {
    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .wait(timeout)
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn token(&self) -> Result<u64, ExecutionError> {
        self.completion.opaque_token().map_err(map_backend_error)
    }
}

struct HipKvSubmission {
    completion: KvAppendCompletion,
    _ticket: ActiveOperation,
}

struct HipCausalAttentionSubmission {
    completion: CausalAttentionCompletion,
    _evidence: CausalAttentionEvidence,
    _ticket: ActiveOperation,
}

struct HipLinearAttentionSubmission {
    completion: LinearAttentionCompletion,
    _evidence: LinearAttentionEvidence,
    _ticket: ActiveOperation,
}

impl ExecutionLinearAttentionSubmissionAdapter for HipLinearAttentionSubmission {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .query()
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .wait(timeout)
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn finalize_after_fence(&mut self, fence_token: u64) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .finalize_after_token(fence_token)
            .map(map_completion_state)
            .map_err(map_async_error)
    }
}

impl ExecutionCausalAttentionSubmissionAdapter for HipCausalAttentionSubmission {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .query()
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .wait(timeout)
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn finalize_after_fence(&mut self, fence_token: u64) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .finalize_after_token(fence_token)
            .map(map_completion_state)
            .map_err(map_async_error)
    }
}

impl ExecutionKvStateSubmissionAdapter for HipKvSubmission {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .query()
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .wait(timeout)
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn finalize_after_fence(&mut self, fence_token: u64) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .finalize_after_token(fence_token)
            .map(map_completion_state)
            .map_err(map_async_error)
    }
}

impl ExecutionTransferAdapter for HipTransfer {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .query()
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .wait(timeout)
            .map(map_completion_state)
            .map_err(map_async_error)
    }
}

struct HipReadback {
    completion: Completion,
    _ticket: ActiveOperation,
}

impl ExecutionReadbackAdapter for HipReadback {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .query()
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn wait(&mut self, timeout: Duration) -> Result<ExecutionState, ExecutionError> {
        self.completion
            .wait(timeout)
            .map(map_completion_state)
            .map_err(map_async_error)
    }

    fn read_into(&mut self, destination: &mut [u8]) -> Result<u64, ExecutionError> {
        self.completion
            .read_into(destination)
            .map_err(map_async_error)
    }
}

fn map_completion_state(state: CompletionState) -> ExecutionState {
    match state {
        CompletionState::Pending => ExecutionState::Pending,
        CompletionState::Success => ExecutionState::Success,
        CompletionState::Failure => ExecutionState::Failure,
    }
}

fn map_backend_error(error: RuntimeError) -> ExecutionError {
    match error.status() {
        RuntimeStatus::HipUnavailable => ExecutionError::ExecutionUnavailable {
            backend: HIP_BACKEND_NAME,
            reason: error.message().to_owned(),
        },
        RuntimeStatus::Busy
        | RuntimeStatus::CausalAttentionStateBusy
        | RuntimeStatus::LinearAttentionStateBusy => ExecutionError::Busy,
        RuntimeStatus::NotReady => ExecutionError::NotReady,
        _ => ExecutionError::BackendStatus {
            status: error.status().raw(),
            diagnostic: error.message().to_owned(),
        },
    }
}

fn map_async_error(error: RuntimeError) -> ExecutionError {
    match error.status() {
        RuntimeStatus::Busy => ExecutionError::Busy,
        RuntimeStatus::NotReady => ExecutionError::NotReady,
        _ => ExecutionError::AsyncFailure {
            status: error.status().raw(),
            diagnostic: error.message().to_owned(),
        },
    }
}

fn logical_dispatch_target(target: String) -> String {
    logical_gcn_arch_name(&target).to_owned()
}

fn dispatch_from_rmsnorm(dispatch: RmsNormDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.row_count,
        normalized_size: dispatch.normalized_size,
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_elementwise(dispatch: ElementwiseDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: 1,
        normalized_size: dispatch.element_count,
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_embedding(dispatch: EmbeddingDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.token_count,
        normalized_size: dispatch.hidden_size,
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_matmul(dispatch: MatmulDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.m,
        normalized_size: dispatch.output_elements,
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_argmax(dispatch: ArgmaxDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.row_count,
        normalized_size: dispatch.vocab_size,
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_attention_preprocess(
    dispatch: AttentionPreprocessDispatchInfo,
) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.m,
        normalized_size: 256,
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_rotary(dispatch: RotaryDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.token_count,
        normalized_size: u64::from(dispatch.head_dim),
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_windowed_attention(dispatch: WindowedAttentionDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.query_count,
        normalized_size: u64::from(dispatch.head_dim),
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_moe_expert(dispatch: MoeExpertDispatchInfo) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: dispatch.abi_version,
        info_version: dispatch.info_version,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count + 3,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.token_count,
        normalized_size: dispatch.active_pair_count,
        backend: dispatch.backend,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.gcn_arch_name),
    }
}

fn dispatch_from_kv_append(dispatch: KvAppendEvidence) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        info_version: sys::SLLM_HIP_KV_APPEND_INFO_VERSION,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.token_count,
        normalized_size: 4 * 256,
        backend: sys::SLLM_BACKEND_HIP,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.target),
    }
}

fn dispatch_from_causal_attention(dispatch: CausalAttentionEvidence) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        info_version: sys::SLLM_HIP_CAUSAL_ATTENTION_DISPATCH_INFO_VERSION,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.grid_size_x,
        row_count: dispatch.query_count,
        normalized_size: dispatch.head_dim as u64,
        backend: sys::SLLM_BACKEND_HIP,
        fallback_allowed: dispatch.fallback_allowed,
        fallback_used: dispatch.fallback_used,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.device_symbol,
        target: logical_dispatch_target(dispatch.target),
    }
}

fn dispatch_from_linear_attention(dispatch: LinearAttentionEvidence) -> DispatchEvidence {
    DispatchEvidence {
        abi_version: sys::SLLM_HIP_ABI_VERSION,
        info_version: sys::SLLM_HIP_LINEAR_ATTENTION_DISPATCH_INFO_VERSION,
        dispatch_id: dispatch.dispatch_id,
        dispatch_count: dispatch.dispatch_count,
        kernel_id: dispatch.recurrent_kernel_id,
        workgroup_size_x: dispatch.workgroup_size_x,
        grid_size_x: dispatch.recurrent_grid_size_x,
        row_count: dispatch.token_count,
        normalized_size: sys::SLLM_HIP_LINEAR_ATTENTION_HEAD_DIM as u64,
        backend: sys::SLLM_BACKEND_HIP,
        fallback_allowed: false,
        fallback_used: false,
        kernel_symbol: dispatch.kernel_symbol,
        device_symbol: dispatch.recurrent_device_symbol,
        target: logical_dispatch_target(dispatch.target),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sllm_core::{
        AttentionPreprocessContract, AttentionPreprocessPositionMode, Backend, DType,
        ExecutionSessionRequest, SemanticOpDescriptor, SplitHalfRotaryContract, TensorView,
        WindowedCausalAttentionContract,
    };

    #[test]
    fn mi300x_feature_tuple_has_one_fail_closed_logical_normalization() {
        assert_eq!(
            logical_dispatch_target("gfx942:sramecc+:xnack-".to_owned()),
            "gfx942"
        );
        assert_eq!(
            logical_dispatch_target("gfx942:sramecc+:xnack+".to_owned()),
            "gfx942:sramecc+:xnack+"
        );
    }
    use std::sync::Barrier;
    use std::thread;

    #[test]
    fn generic_backend_bridge_is_additive_and_stub_stays_unavailable() {
        let backend = HipBackend { _private: () };
        let request = ExecutionSessionRequest::new(0, "gfx1201").unwrap();
        assert!(matches!(
            backend.open_execution_session(request),
            Err(ExecutionError::ExecutionUnavailable { .. })
        ));
        assert!(!backend.capabilities().numerical_execution);
    }

    #[test]
    fn owned_bridge_advertises_the_existing_public_transfer_limit_without_gpu() {
        let adapter = HipExecutionSession {
            state: Arc::new(HipSessionState::new()),
            backend: HipBackend { _private: () },
            context: Context::test_without_native(),
            available_memory_bytes: u64::MAX,
        };
        assert_eq!(
            adapter.max_transfer_bytes(),
            crate::sys::SLLM_HIP_MAX_TRANSFER_BYTES
        );
        assert_eq!(adapter.max_transfer_bytes(), 1_073_741_824);
    }

    #[test]
    fn closing_rejects_new_active_operations_without_gpu() {
        let state = Arc::new(HipSessionState::new());
        assert_eq!(state.begin_shutdown(), Ok(()));
        assert!(matches!(
            state.acquire_active(),
            Err(ExecutionError::Closing)
        ));
        assert_eq!(state.active_count(), 0);
    }

    #[test]
    fn active_operation_ticket_changes_count_once_without_gpu() {
        let state = Arc::new(HipSessionState::new());
        assert_eq!(state.active_count(), 0);
        let ticket = state.acquire_active().expect("open state accepts ticket");
        assert_eq!(state.active_count(), 1);
        drop(ticket);
        assert_eq!(state.active_count(), 0);
    }

    #[test]
    fn shutdown_cas_wins_over_a_paused_admission_without_count_corruption() {
        let state = Arc::new(HipSessionState::new());
        let reached = Arc::new(Barrier::new(2));
        let proceed = Arc::new(Barrier::new(2));
        state
            .activity
            .pause_next_admission(Arc::clone(&reached), Arc::clone(&proceed));

        let admission_state = Arc::clone(&state);
        let admission_thread = thread::spawn(move || admission_state.acquire_active().map(|_| ()));
        reached.wait();

        assert_eq!(state.begin_shutdown(), Ok(()));
        proceed.wait();

        assert_eq!(
            admission_thread.join().expect("admission thread completed"),
            Err(ExecutionError::Closing)
        );
        assert_eq!(state.active_count(), 0);
    }

    #[test]
    fn closing_with_live_active_operation_is_busy_without_gpu() {
        let state = Arc::new(HipSessionState::new());
        let ticket = state.acquire_active().expect("open state accepts ticket");
        assert_eq!(state.active_count(), 1);
        assert_eq!(state.begin_shutdown(), Err(ExecutionError::Busy));
        assert_eq!(state.active_count(), 1);
        assert!(matches!(
            state.acquire_active(),
            Err(ExecutionError::Closing)
        ));
        drop(ticket);
        assert_eq!(state.active_count(), 0);
        assert_eq!(state.begin_shutdown(), Ok(()));
    }

    fn attention_descriptor() -> SemanticOpDescriptor {
        let contract = AttentionPreprocessContract::new_qwen3_5(
            AttentionPreprocessPositionMode::DecodeContinuation,
            3,
            17,
        )
        .expect("valid attention preprocess contract");
        SemanticOpDescriptor::new_attention_preprocess(
            vec![
                TensorView::contiguous(DType::Bf16, &[17, 16, 512]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[17, 4, 256]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[16, 256]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[4, 256]).unwrap(),
                TensorView::contiguous(DType::I32, &[17]).unwrap(),
            ],
            vec![
                TensorView::contiguous(DType::Bf16, &[17, 16, 256]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[17, 16, 256]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[17, 4, 256]).unwrap(),
            ],
            contract,
        )
        .expect("valid attention preprocess descriptor")
    }

    #[test]
    fn supports_attention_preprocess_after_owned_path_is_registered() {
        let adapter = HipExecutionSession {
            state: Arc::new(HipSessionState::new()),
            backend: HipBackend { _private: () },
            context: Context::test_without_native(),
            available_memory_bytes: u64::MAX,
        };
        assert_eq!(
            adapter.supports(&attention_descriptor()),
            PrepareSupport::Supported
        );
    }

    #[test]
    fn supports_split_half_rotary_after_owned_path_is_registered() {
        let adapter = HipExecutionSession {
            state: Arc::new(HipSessionState::new()),
            backend: HipBackend { _private: () },
            context: Context::test_without_native(),
            available_memory_bytes: u64::MAX,
        };
        let contract = SplitHalfRotaryContract::new(3, 1, 6, 4, 10_000.0, 255, 3, 262_144)
            .expect("valid rotary contract");
        let descriptor = SemanticOpDescriptor::new_rotary(
            vec![
                TensorView::contiguous(DType::Bf16, &[3, 3, 6]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[3, 1, 6]).unwrap(),
                TensorView::contiguous(DType::I32, &[3]).unwrap(),
            ],
            vec![
                TensorView::contiguous(DType::Bf16, &[3, 3, 6]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[3, 1, 6]).unwrap(),
            ],
            contract,
        )
        .expect("valid rotary descriptor");
        assert_eq!(adapter.supports(&descriptor), PrepareSupport::Supported);
    }

    #[test]
    fn supports_windowed_attention_after_owned_path_is_registered() {
        let adapter = HipExecutionSession {
            state: Arc::new(HipSessionState::new()),
            backend: HipBackend { _private: () },
            context: Context::test_without_native(),
            available_memory_bytes: u64::MAX,
        };
        let contract = WindowedCausalAttentionContract::new(3, 1, 6, 2, 3, 5, Some(4), 1.0)
            .expect("valid windowed attention contract");
        let descriptor = SemanticOpDescriptor::new_causal_attention(
            vec![
                TensorView::contiguous(DType::Bf16, &[3, 3, 6]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[5, 1, 6]).unwrap(),
                TensorView::contiguous(DType::Bf16, &[5, 1, 6]).unwrap(),
            ],
            vec![TensorView::contiguous(DType::Bf16, &[3, 3, 6]).unwrap()],
            contract,
        )
        .expect("valid windowed attention descriptor");
        assert_eq!(adapter.supports(&descriptor), PrepareSupport::Supported);
    }

    #[test]
    fn rotary_dispatch_mapping_preserves_non_aligned_shape_and_target() {
        let dispatch = dispatch_from_rotary(RotaryDispatchInfo {
            abi_version: 1,
            info_version: 1,
            dispatch_id: 12,
            dispatch_count: 1,
            kernel_id: 1,
            workgroup_size_x: 256,
            grid_size_x: 1,
            token_count: 3,
            q_heads: 3,
            kv_heads: 1,
            head_dim: 6,
            rotary_dim: 4,
            start_position: 255,
            max_position: 262_144,
            backend: crate::sys::SLLM_BACKEND_HIP,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: "rotary.split_half.bf16_fp32.v1".to_owned(),
            device_symbol: "sllm_rotary_split_half_bf16_fp32_v1".to_owned(),
            gcn_arch_name: "gfx1030".to_owned(),
        });
        assert_eq!(dispatch.row_count, 3);
        assert_eq!(dispatch.normalized_size, 6);
        assert_eq!(dispatch.target, "gfx1030");
        assert!(!dispatch.fallback_allowed);
        assert!(!dispatch.fallback_used);
    }

    #[test]
    fn windowed_attention_dispatch_mapping_preserves_shape_window_and_target() {
        let dispatch = dispatch_from_windowed_attention(WindowedAttentionDispatchInfo {
            abi_version: 1,
            info_version: 1,
            dispatch_id: 13,
            dispatch_count: 1,
            kernel_id: 1,
            workgroup_size_x: 256,
            grid_size_x: 1,
            query_count: 3,
            start_position: 2,
            committed_kv_length: 5,
            sliding_window: 4,
            q_heads: 3,
            kv_heads: 1,
            head_dim: 6,
            scaling_bits: 1.0_f32.to_bits(),
            backend: crate::sys::SLLM_BACKEND_HIP,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: "attention.windowed.bf16_fp32.v1".to_owned(),
            device_symbol: "sllm_gemma_attention_bf16_fp32_v1".to_owned(),
            gcn_arch_name: "gfx1201".to_owned(),
        });
        assert_eq!(dispatch.row_count, 3);
        assert_eq!(dispatch.normalized_size, 6);
        assert_eq!(dispatch.target, "gfx1201");
        assert!(!dispatch.fallback_allowed);
        assert!(!dispatch.fallback_used);
    }

    #[test]
    fn attention_dispatch_mapping_uses_m_rows_and_fixed_256_normalized_size() {
        let dispatch = dispatch_from_attention_preprocess(AttentionPreprocessDispatchInfo {
            abi_version: 1,
            info_version: 1,
            dispatch_id: 11,
            dispatch_count: 1,
            kernel_id: 1,
            workgroup_size_x: 1,
            grid_size_x: 340,
            m: 17,
            q_heads: 16,
            k_heads: 4,
            q_head_dim: 256,
            k_head_dim: 256,
            rotary_dim: 64,
            start_position: 255,
            backend: crate::sys::SLLM_BACKEND_HIP,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: "attention_preprocess.headwise_norm_rope.v1".to_owned(),
            device_symbol: "sllm_attention_preprocess_headwise_norm_rope_v1".to_owned(),
            gcn_arch_name: "gfx1201".to_owned(),
        });
        assert_eq!(dispatch.row_count, 17);
        assert_eq!(dispatch.normalized_size, 256);
        assert_eq!(dispatch.target, "gfx1201");
        assert!(!dispatch.fallback_allowed);
        assert!(!dispatch.fallback_used);
        assert_eq!(
            dispatch.kernel_symbol,
            "attention_preprocess.headwise_norm_rope.v1"
        );
    }
}
