//! Additive `sllm-core` owned-execution adapter for the typed public HIP
//! RMSNorm path.  It does not alter the legacy `Backend` control-plane methods.
//!
//! The adapter contains no alternate ABI or kernel path.  It only lowers core
//! owned bindings into the existing `Context`/`Queue`/`Buffer`/
//! `PreparedRmsNorm`/`RmsNormSubmission` wrappers.

use std::sync::Arc;
#[cfg(test)]
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use sllm_core::{
    AdapterResource, BoundSemanticOp, BufferRange, DispatchEvidence, ExecutionAdapterAccess,
    ExecutionError, ExecutionReadbackAdapter, ExecutionSession, ExecutionSessionAdapter,
    ExecutionSessionRequest, ExecutionState, ExecutionSubmissionAdapter, ExecutionTransferAdapter,
    OwnedTensorBinding, PrepareSupport, PreparedOperation, ShutdownReport,
};

use crate::{
    Buffer, Completion, CompletionState, Context, HipBackend, PreparedRmsNorm, Queue,
    RmsNormDescriptor, RmsNormDispatchInfo, RmsNormSubmission, RuntimeError, RuntimeStatus,
};

const HIP_BACKEND_NAME: &str = "hip";
const CLEANUP_ATTEMPT_CAP: usize = 16;

pub(crate) fn open_execution_session(
    backend: HipBackend,
    request: ExecutionSessionRequest,
) -> Result<Arc<ExecutionSession>, ExecutionError> {
    let context = Context::create(request.device_index(), request.expected_target())
        .map_err(map_backend_error)?;
    let adapter = Arc::new(HipExecutionSession {
        state: Arc::new(HipSessionState::new()),
        backend,
        context,
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
}

impl ExecutionSessionAdapter for HipExecutionSession {
    fn max_transfer_bytes(&self) -> u64 {
        crate::sys::SLLM_HIP_MAX_TRANSFER_BYTES
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
        if descriptor.kind() != sllm_core::SemanticOpKind::RmsNorm {
            return PrepareSupport::Unsupported {
                reason: "the HIP owned execution bridge currently prepares RMSNorm only".to_owned(),
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

    fn prepare(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        operation: &BoundSemanticOp,
    ) -> Result<AdapterResource, ExecutionError> {
        self.state.ensure_open()?;
        if operation.descriptor().kind() != sllm_core::SemanticOpKind::RmsNorm {
            return Err(ExecutionError::Unsupported {
                reason: "the HIP owned execution bridge currently prepares RMSNorm only".to_owned(),
            });
        }
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
        self.backend
            .prepare_rms_norm(&self.context, descriptor)
            .map(AdapterResource::new)
            .map_err(map_backend_error)
    }

    fn submit(
        &self,
        access: &ExecutionAdapterAccess<'_>,
        prepared: &PreparedOperation,
        queue: &sllm_core::ExecutionQueue,
    ) -> Result<(Box<dyn ExecutionSubmissionAdapter>, DispatchEvidence), ExecutionError> {
        self.state.ensure_open()?;
        let plan = access
            .downcast_prepared_payload::<PreparedRmsNorm>(prepared)?
            .clone();
        let queue = access.downcast_queue_payload::<Queue>(queue)?.clone();
        let ticket = self.state.acquire_active()?;
        let (submission, dispatch) = plan.execute(&queue).map_err(map_backend_error)?;
        Ok((
            Box::new(HipSubmission {
                submission,
                queue,
                _ticket: ticket,
            }),
            dispatch_from_hip(dispatch),
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

struct HipSubmission {
    submission: RmsNormSubmission,
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
        RuntimeStatus::Busy => ExecutionError::Busy,
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

fn dispatch_from_hip(dispatch: RmsNormDispatchInfo) -> DispatchEvidence {
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
        target: dispatch.gcn_arch_name,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sllm_core::{Backend, ExecutionSessionRequest};
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
}
