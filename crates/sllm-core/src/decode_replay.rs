//! One-ahead replay of a captured decode graph.
//!
//! The controller owns the result ring and every in-flight graph submission,
//! fence, and D2H readback. It submits the next generation before observing
//! the current result, then validates the current slot with the backend-
//! neutral little-endian decoder. The only exception is a replay with exactly
//! one output-budget row remaining: every valid graph result must consume that
//! row and halt, so no useful successor exists. No token is uploaded by this
//! path.

use std::sync::Arc;
use std::time::Duration;

use crate::decode_control::{
    DECODE_RESULT_BYTES_V1, DECODE_RESULT_RING_SLOTS_V1, DecodeControlV1, DecodeResultV1,
    decode_result_le,
};
use crate::{
    BufferReadback, ExecutionBuffer, ExecutionError, ExecutionGraphSpan, ExecutionQueueFence,
    ExecutionSession, ExecutionState, Submission,
};

const RESULT_RING_BYTES: u64 = (DECODE_RESULT_BYTES_V1 * DECODE_RESULT_RING_SLOTS_V1) as u64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeReplayResultState {
    NotStarted,
    Running,
    Finished,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeReplayAuditV1 {
    completed_replays: u64,
    discarded_replays: u64,
    executed_replays: u64,
    native_nodes_per_replay: u64,
    native_nodes_total: u64,
    native_kernel_nodes_per_replay: u64,
    native_kernel_nodes_total: u64,
    executed_native_nodes_total: u64,
    executed_native_kernel_nodes_total: u64,
    logical_dispatch_count: u64,
    logical_kernel_dispatch_count: u64,
    provider_dispatch_count: u64,
    provider_kernel_dispatch_count: u64,
    startup_ns: u64,
}

impl DecodeReplayAuditV1 {
    pub const fn completed_replays(self) -> u64 {
        self.completed_replays
    }

    pub const fn discarded_replays(self) -> u64 {
        self.discarded_replays
    }

    pub const fn executed_replays(self) -> u64 {
        self.executed_replays
    }

    pub const fn native_nodes_per_replay(self) -> u64 {
        self.native_nodes_per_replay
    }

    pub const fn native_nodes_total(self) -> u64 {
        self.native_nodes_total
    }

    pub const fn native_kernel_nodes_per_replay(self) -> u64 {
        self.native_kernel_nodes_per_replay
    }

    pub const fn native_kernel_nodes_total(self) -> u64 {
        self.native_kernel_nodes_total
    }

    pub const fn executed_native_nodes_total(self) -> u64 {
        self.executed_native_nodes_total
    }

    pub const fn executed_native_kernel_nodes_total(self) -> u64 {
        self.executed_native_kernel_nodes_total
    }

    pub const fn logical_dispatch_count(self) -> u64 {
        self.logical_dispatch_count
    }

    pub const fn logical_kernel_dispatch_count(self) -> u64 {
        self.logical_kernel_dispatch_count
    }

    pub const fn provider_dispatch_count(self) -> u64 {
        self.provider_dispatch_count
    }

    pub const fn provider_kernel_dispatch_count(self) -> u64 {
        self.provider_kernel_dispatch_count
    }

    pub const fn startup_ns(self) -> u64 {
        self.startup_ns
    }
}

struct PendingReplay {
    generation: u64,
    ring_slot: usize,
    submission: Option<Submission>,
    fence: Option<ExecutionQueueFence>,
    readback: Option<BufferReadback>,
}

impl PendingReplay {
    fn drain(mut self, timeout: Duration) -> Result<(), ExecutionError> {
        let mut first_error = None;
        if self.readback.is_none() || self.fence.is_none() || self.submission.is_none() {
            first_error = Some(ExecutionError::InvalidRequest {
                reason: "decode replay discard lost an owned completion resource".to_owned(),
            });
        }
        if let Some(mut readback) = self.readback.take() {
            match readback.wait(timeout) {
                Ok(ExecutionState::Success) => {}
                Ok(state) => {
                    first_error.get_or_insert(ExecutionError::AsyncFailure {
                        status: 0,
                        diagnostic: format!("decode result discard readback ended in {state:?}"),
                    });
                }
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            };
        }
        let token = self.fence.as_mut().and_then(|fence| {
            let state = match fence.query() {
                Ok(ExecutionState::Pending) => fence.wait(timeout),
                other => other,
            };
            if !matches!(state, Ok(ExecutionState::Success)) {
                if let Ok(state) = state {
                    first_error.get_or_insert(ExecutionError::AsyncFailure {
                        status: 0,
                        diagnostic: format!("decode discard fence ended in {state:?}"),
                    });
                } else if let Err(error) = state {
                    first_error.get_or_insert(error);
                }
                return None;
            }
            fence.token().ok()
        });
        if let (Some(mut submission), Some(fence)) = (self.submission.take(), self.fence.as_ref()) {
            if token.is_some() {
                match submission.finalize_after_fence(fence) {
                    Ok(ExecutionState::Success) => {}
                    Ok(state) => {
                        first_error.get_or_insert(ExecutionError::AsyncFailure {
                            status: 0,
                            diagnostic: format!("decode discard finalization ended in {state:?}"),
                        });
                    }
                    Err(error) => {
                        first_error.get_or_insert(error);
                    }
                };
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn settle(mut self, timeout: Duration) -> Result<[u8; DECODE_RESULT_BYTES_V1], ExecutionError> {
        let mut first_error = None;
        if self.readback.is_none() || self.fence.is_none() || self.submission.is_none() {
            first_error = Some(ExecutionError::InvalidRequest {
                reason: "decode replay lost an owned completion resource".to_owned(),
            });
        }
        let mut bytes = [0_u8; DECODE_RESULT_BYTES_V1];
        if let Some(mut readback) = self.readback.take() {
            match readback.wait(timeout) {
                Ok(ExecutionState::Success) => match readback.read_into(&mut bytes) {
                    Ok(copied) if copied == DECODE_RESULT_BYTES_V1 as u64 => {}
                    Ok(copied) => {
                        first_error = Some(ExecutionError::InvalidRange {
                            reason: format!(
                                "decode result readback returned {copied} bytes, expected {DECODE_RESULT_BYTES_V1}"
                            ),
                        })
                    }
                    Err(error) => first_error = Some(error),
                },
                Ok(state) => {
                    first_error = Some(ExecutionError::AsyncFailure {
                        status: 0,
                        diagnostic: format!("decode result readback ended in {state:?}"),
                    });
                }
                Err(error) => first_error = Some(error),
            }
        }

        if let Some(fence) = self.fence.as_mut() {
            let queried = fence.query();
            let completed = match queried {
                Ok(ExecutionState::Pending) => fence.wait(timeout),
                other => other,
            };
            match completed {
                Ok(ExecutionState::Success) => {}
                Ok(state) => {
                    first_error.get_or_insert(ExecutionError::AsyncFailure {
                        status: 0,
                        diagnostic: format!("decode graph fence ended in {state:?}"),
                    });
                }
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }

        if let (Some(submission), Some(fence)) = (self.submission.as_mut(), self.fence.as_ref()) {
            let finalized = submission.finalize_after_fence(fence);
            match finalized {
                Ok(ExecutionState::Success) => {}
                Ok(state) => {
                    first_error.get_or_insert(ExecutionError::AsyncFailure {
                        status: 0,
                        diagnostic: format!("decode graph finalization ended in {state:?}"),
                    });
                }
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            };
            /*
             * The fence is retained until finalization above. Its token is
             * intentionally not used as a substitute for terminal state.
             */
        } else if self.submission.is_some() {
            first_error.get_or_insert(ExecutionError::InvalidRequest {
                reason: "decode replay submission lost its fence".to_owned(),
            });
        }
        first_error.map_or(Ok(bytes), Err)
    }
}

pub struct DecodeReplayController {
    session: Arc<ExecutionSession>,
    graph: ExecutionGraphSpan,
    result_ring: ExecutionBuffer,
    control: DecodeControlV1,
    initial_model_position: u64,
    vocabulary_size: u32,
    timeout: Duration,
    pending: Option<PendingReplay>,
    last_result: Option<DecodeResultV1>,
    completed_replays: u64,
    finished: bool,
    discarded_replays: u64,
    startup_ns: u64,
}

impl DecodeReplayController {
    pub fn new(
        session: Arc<ExecutionSession>,
        graph: ExecutionGraphSpan,
        result_ring: ExecutionBuffer,
        initial_control: DecodeControlV1,
        vocabulary_size: u32,
        timeout: Duration,
    ) -> Result<Self, ExecutionError> {
        if graph.queue().session_id() != session.id() || result_ring.session_id() != session.id() {
            return Err(ExecutionError::WrongSession {
                expected: session.id(),
                actual: graph.queue().session_id(),
            });
        }
        if result_ring.size_bytes() != RESULT_RING_BYTES {
            return Err(ExecutionError::InvalidRange {
                reason: format!("decode result ring must be exactly {RESULT_RING_BYTES} bytes"),
            });
        }
        initial_control.validate(vocabulary_size)?;
        Ok(Self {
            session,
            graph,
            result_ring,
            initial_model_position: initial_control.model_position,
            control: initial_control,
            vocabulary_size,
            timeout,
            pending: None,
            last_result: None,
            completed_replays: 0,
            finished: false,
            discarded_replays: 0,
            startup_ns: 0,
        })
    }

    pub fn start(&mut self) -> Result<(), ExecutionError> {
        if self.pending.is_some() || self.finished {
            return Err(ExecutionError::InvalidRequest {
                reason: "decode replay controller was already started or finished".to_owned(),
            });
        }
        self.pending = Some(self.enqueue_generation(1)?);
        Ok(())
    }

    pub(crate) fn set_startup_ns(&mut self, startup_ns: u64) {
        self.startup_ns = startup_ns;
    }

    // This is an effectful fallible cursor rather than a standard Iterator:
    // errors must remain observable and terminal cleanup must run before the
    // caller can advance again.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<DecodeResultV1>, ExecutionError> {
        if self.finished {
            return Ok(None);
        }
        let current = self
            .pending
            .take()
            .ok_or_else(|| ExecutionError::InvalidRequest {
                reason: "decode replay controller must be started before next".to_owned(),
            })?;
        let current_generation = current.generation;
        let current_ring_slot = current.ring_slot;
        let expected_model_position = self.control.model_position;
        let expected_sampler_counter = self.control.sampler_counter;
        let expected_output_count = self.control.output_count;
        let known_budget_terminal =
            expected_output_count.checked_add(1) == Some(self.control.output_limit);

        // Enqueue the next graph/fence/readback before observing the current
        // D2H result whenever another useful replay can exist. The two ring
        // slots remain distinct by generation parity. A valid replay with one
        // budget row left must consume that row and halt, so submitting its
        // successor would only execute a discarded Noop graph.
        if !known_budget_terminal {
            let next_generation = match current_generation.checked_add(1) {
                Some(generation) => generation,
                None => {
                    let _ = current.drain(self.timeout);
                    self.finished = true;
                    return Err(ExecutionError::InvalidRequest {
                        reason: "decode replay generation overflowed".to_owned(),
                    });
                }
            };
            let next = match self.enqueue_generation(next_generation) {
                Ok(next) => next,
                Err(error) => {
                    let _ = current.drain(self.timeout);
                    self.finished = true;
                    return Err(error);
                }
            };
            self.pending = Some(next);
        }

        let bytes = match current.settle(self.timeout) {
            Ok(bytes) => bytes,
            Err(error) => {
                let _ = self.drain_pending();
                self.finished = true;
                return Err(error);
            }
        };
        let result = match decode_result_le(
            &bytes,
            current_ring_slot,
            current_generation,
            self.control.mode,
            self.control.width,
            expected_model_position,
            expected_sampler_counter,
            expected_output_count,
            self.control.output_limit,
            self.vocabulary_size,
        ) {
            Ok(result) => result,
            Err(error) => {
                let _ = self.drain_pending();
                self.finished = true;
                return Err(error);
            }
        };
        if known_budget_terminal && !result.halted() {
            self.finished = true;
            return Err(ExecutionError::InvalidRequest {
                reason: "decode replay did not halt after consuming its final budget row"
                    .to_owned(),
            });
        }
        if let Err(error) = self.apply_result(&result) {
            let _ = self.drain_pending();
            self.finished = true;
            return Err(error);
        }
        self.completed_replays = self.completed_replays.saturating_add(1);
        self.last_result = Some(result.clone());

        if result.halted() {
            // A speculative successor remains queued for an unpredictable
            // stop. Its result slot belongs to the old generation and is
            // never parsed, even if the device wrote a valid Noop record.
            // Known budget termination deliberately has no successor.
            if self.pending.is_some() {
                if let Err(error) = self.drain_pending() {
                    self.finished = true;
                    return Err(error);
                }
                self.discarded_replays = self.discarded_replays.saturating_add(1);
            }
            self.finished = true;
            self.session.publish_graph_state_metadata(
                &self.graph,
                self.initial_model_position,
                result.model_position_after,
                result.generation,
            )?;
        }
        Ok(Some(result))
    }

    pub fn result(&self) -> Option<&DecodeResultV1> {
        self.last_result.as_ref()
    }

    pub fn audit(&self) -> DecodeReplayAuditV1 {
        let executed_replays = self
            .completed_replays
            .saturating_add(self.discarded_replays);
        let native_nodes_per_replay = self.graph.native_node_count();
        let native_kernel_nodes_per_replay = self.graph.native_kernel_nodes();
        DecodeReplayAuditV1 {
            completed_replays: self.completed_replays,
            discarded_replays: self.discarded_replays,
            executed_replays,
            native_nodes_per_replay,
            native_nodes_total: native_nodes_per_replay.saturating_mul(self.completed_replays),
            native_kernel_nodes_per_replay,
            native_kernel_nodes_total: native_kernel_nodes_per_replay
                .saturating_mul(self.completed_replays),
            executed_native_nodes_total: native_nodes_per_replay.saturating_mul(executed_replays),
            executed_native_kernel_nodes_total: native_kernel_nodes_per_replay
                .saturating_mul(executed_replays),
            logical_dispatch_count: self.graph.logical_dispatch_count(),
            logical_kernel_dispatch_count: self.graph.logical_kernel_dispatch_count(),
            provider_dispatch_count: self.graph.provider_dispatch_count(),
            provider_kernel_dispatch_count: self.graph.provider_kernel_dispatch_count(),
            startup_ns: self.startup_ns,
        }
    }

    pub fn control(&self) -> &DecodeControlV1 {
        &self.control
    }

    pub fn state(&self) -> DecodeReplayResultState {
        if self.finished {
            DecodeReplayResultState::Finished
        } else if self.pending.is_some() {
            DecodeReplayResultState::Running
        } else {
            DecodeReplayResultState::NotStarted
        }
    }

    pub const fn finished(&self) -> bool {
        self.finished
    }

    pub fn graph_node_count(&self) -> u64 {
        self.graph.native_kernel_nodes()
    }

    fn enqueue_generation(&self, generation: u64) -> Result<PendingReplay, ExecutionError> {
        // The next replay is queued before the current result is read back.
        // Native graph append admits at most nine rows per replay, so reserve
        // two replay windows without synchronizing on device control.
        let conservative_end = self.control.model_position.saturating_add(18);
        self.session
            .prepare_graph_paged_kv(&self.graph, conservative_end)?;
        let submission = self.session.submit_graph_span(&self.graph)?;
        let fence = self.session.create_queue_fence(self.graph.queue())?;
        let slot = generation as usize % DECODE_RESULT_RING_SLOTS_V1;
        let range = self.result_ring.range(
            (slot * DECODE_RESULT_BYTES_V1) as u64,
            DECODE_RESULT_BYTES_V1 as u64,
        )?;
        let readback = self.session.readback(self.graph.queue(), range)?;
        // The fence is deliberately retained even though its completion is
        // observed after the D2H event; deferred submissions require its token
        // to finalize their state owner.
        Ok(PendingReplay {
            generation,
            ring_slot: slot,
            submission: Some(submission),
            fence: Some(fence),
            readback: Some(readback),
        })
    }

    fn apply_result(&mut self, result: &DecodeResultV1) -> Result<(), ExecutionError> {
        self.control.model_position = result.model_position_after;
        self.control.sampler_counter = result.counter_after;
        self.control.generation = result.generation;
        self.control.status = result.status;
        self.control.halted = result.halted();
        self.control.commit_rows = result.commit_rows;
        self.control.publish_count = result.count;
        self.control.accepted_count = result.accepted;
        self.control.stop_row = result.stop_row;
        self.control.phase_active = false;
        self.control.phase_rows = 0;
        self.control.output_count = self
            .control
            .output_count
            .checked_add(u64::from(result.count))
            .ok_or_else(|| ExecutionError::InvalidRequest {
                reason: "decode output count overflowed".to_owned(),
            })?;
        if let Some(selection) = result.selections.last() {
            self.control.pending_token = selection.token_id;
        }
        self.control.validate(self.vocabulary_size)
    }

    fn drain_pending(&mut self) -> Result<(), ExecutionError> {
        if let Some(pending) = self.pending.take() {
            pending.drain(self.timeout)
        } else {
            Ok(())
        }
    }
}

impl Drop for DecodeReplayController {
    fn drop(&mut self) {
        let _ = self.drain_pending();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AccessMode, AdapterResource, BoundSemanticOp, DType, DecodeControlStatusV1,
        DispatchEvidence, ExecutionAdapterAccess, ExecutionQueue, ExecutionQueueFenceAdapter,
        ExecutionReadbackAdapter, ExecutionSessionAdapter, ExecutionSubmissionAdapter,
        ExecutionTransferAdapter, PrepareSupport, PreparedOperation, ShutdownReport, TensorView,
    };
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    #[derive(Default)]
    struct FakeReplayState {
        next_generation: AtomicU64,
        records: Mutex<[Vec<u8>; 2]>,
        log: Mutex<Vec<String>>,
        published: Mutex<Vec<(u64, u64, u64)>>,
        fail_generation: AtomicU64,
        budget_halt_generation: AtomicU64,
        fence_failure_mode: AtomicU64,
        truncate_readback: AtomicBool,
    }

    impl FakeReplayState {
        fn new() -> Self {
            Self {
                records: Mutex::new(std::array::from_fn(|_| vec![0; DECODE_RESULT_BYTES_V1])),
                ..Default::default()
            }
        }

        fn log(&self, event: impl Into<String>) {
            self.log.lock().unwrap().push(event.into());
        }
    }

    struct FakeGraph {
        state: Arc<FakeReplayState>,
    }

    struct FakeSubmission {
        state: Arc<FakeReplayState>,
        generation: u64,
    }

    struct FakeFence {
        state: Arc<FakeReplayState>,
    }

    struct FakeReadback {
        state: Arc<FakeReplayState>,
        generation: u64,
        bytes: Vec<u8>,
    }

    struct FakeTransfer;

    #[derive(Clone)]
    struct FakeAdapter {
        state: Arc<FakeReplayState>,
    }

    fn dispatch() -> DispatchEvidence {
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
            backend: 0,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: "fake".to_owned(),
            device_symbol: "fake".to_owned(),
            target: "fake".to_owned(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn result_bytes(
        generation: u64,
        count: u32,
        accepted: u32,
        width: u32,
        before: u64,
        after: u64,
        counter_before: u64,
        counter_after: u64,
        halt: u32,
    ) -> Vec<u8> {
        let mut bytes = vec![0_u8; DECODE_RESULT_BYTES_V1];
        bytes[0..8].copy_from_slice(&generation.to_le_bytes());
        bytes[8..12].copy_from_slice(&(DecodeControlStatusV1::Ok as u32).to_le_bytes());
        bytes[12..16].copy_from_slice(&count.to_le_bytes());
        bytes[16..20].copy_from_slice(&count.to_le_bytes());
        bytes[20..24].copy_from_slice(&accepted.to_le_bytes());
        bytes[24..28].copy_from_slice(&width.to_le_bytes());
        bytes[28..32].copy_from_slice(&halt.to_le_bytes());
        bytes[32..36].copy_from_slice(&(if halt == 1 { 1_u32 } else { 0 }).to_le_bytes());
        bytes[40..48].copy_from_slice(&before.to_le_bytes());
        bytes[48..56].copy_from_slice(&after.to_le_bytes());
        bytes[56..64].copy_from_slice(&counter_before.to_le_bytes());
        bytes[64..72].copy_from_slice(&counter_after.to_le_bytes());
        for index in 0..count as usize {
            let token = 10 + index as u32;
            bytes[72 + index * 4..76 + index * 4].copy_from_slice(&token.to_le_bytes());
            bytes[112 + index * 8..120 + index * 8]
                .copy_from_slice(&(-0.5_f64).to_bits().to_le_bytes());
        }
        bytes
    }

    impl ExecutionSessionAdapter for FakeAdapter {
        fn max_transfer_bytes(&self) -> u64 {
            4096
        }

        fn supports(&self, _descriptor: &crate::SemanticOpDescriptor) -> PrepareSupport {
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
            size_bytes: u64,
        ) -> Result<AdapterResource, ExecutionError> {
            Ok(AdapterResource::new(Mutex::new(vec![
                0_u8;
                usize::try_from(
                    size_bytes
                )
                .unwrap()
            ])))
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
            Ok((
                Box::new(FakeSubmission {
                    state: Arc::clone(&self.state),
                    generation: 0,
                }),
                dispatch(),
            ))
        }

        fn create_graph_span(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _queue: &ExecutionQueue,
            _operations: &[PreparedOperation],
        ) -> Result<(AdapterResource, u64), ExecutionError> {
            Ok((
                AdapterResource::new(FakeGraph {
                    state: Arc::clone(&self.state),
                }),
                1,
            ))
        }

        fn submit_graph_span(
            &self,
            access: &ExecutionAdapterAccess<'_>,
            span: &ExecutionGraphSpan,
        ) -> Result<Box<dyn ExecutionSubmissionAdapter>, ExecutionError> {
            let graph = access.downcast_graph_span_payload::<FakeGraph>(span)?;
            let generation = graph
                .state
                .next_generation
                .fetch_add(1, Ordering::AcqRel)
                .checked_add(1)
                .unwrap();
            graph.state.log(format!("submit{generation}"));
            let (count, accepted, width, before, after, cb, ca, mut halt) = match generation {
                1 => (2, 2, 2, 100, 102, 7, 9, 0),
                2 => (1, 1, 1, 102, 103, 9, 10, 0),
                3 => (2, 2, 2, 103, 105, 10, 12, 1),
                _ => (0, 0, 2, 105, 105, 12, 12, 1 << 3),
            };
            if graph.state.budget_halt_generation.load(Ordering::Acquire) == generation {
                halt = 1 << 1;
            }
            graph.state.records.lock().unwrap()[generation as usize % 2] = result_bytes(
                generation, count, accepted, width, before, after, cb, ca, halt,
            );
            Ok(Box::new(FakeSubmission {
                state: Arc::clone(&graph.state),
                generation,
            }))
        }

        fn create_queue_fence(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _queue: &ExecutionQueue,
        ) -> Result<Box<dyn ExecutionQueueFenceAdapter>, ExecutionError> {
            self.state.log("fence");
            Ok(Box::new(FakeFence {
                state: Arc::clone(&self.state),
            }))
        }

        fn readback(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _queue: &ExecutionQueue,
            source: &crate::BufferRange,
        ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError> {
            let slot = usize::try_from(source.offset_bytes()).unwrap() / DECODE_RESULT_BYTES_V1;
            let bytes = self.state.records.lock().unwrap()[slot].clone();
            let generation = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
            self.state.log(format!("readback{generation}"));
            Ok(Box::new(FakeReadback {
                state: Arc::clone(&self.state),
                generation,
                bytes,
            }))
        }

        fn upload(
            &self,
            _access: &ExecutionAdapterAccess<'_>,
            _queue: &ExecutionQueue,
            _destination: &crate::BufferRange,
            _bytes: Arc<[u8]>,
        ) -> Result<Box<dyn ExecutionTransferAdapter>, ExecutionError> {
            Ok(Box::new(FakeTransfer))
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

        fn publish_graph_state_metadata(
            &self,
            access: &ExecutionAdapterAccess<'_>,
            span: &ExecutionGraphSpan,
            initial: u64,
            final_position: u64,
            generations: u64,
        ) -> Result<(), ExecutionError> {
            let graph = access.downcast_graph_span_payload::<FakeGraph>(span)?;
            graph
                .state
                .published
                .lock()
                .unwrap()
                .push((initial, final_position, generations));
            graph.state.log("publish");
            Ok(())
        }

        fn prepare_graph_paged_kv(
            &self,
            access: &ExecutionAdapterAccess<'_>,
            span: &ExecutionGraphSpan,
            conservative_end: u64,
        ) -> Result<(), ExecutionError> {
            let graph = access.downcast_graph_span_payload::<FakeGraph>(span)?;
            graph.state.log(format!("prepare_paged:{conservative_end}"));
            Ok(())
        }
    }

    impl ExecutionQueueFenceAdapter for FakeFence {
        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            self.state.log("fence_wait");
            match self.state.fence_failure_mode.load(Ordering::Acquire) {
                1 => Ok(ExecutionState::Failure),
                2 => Ok(ExecutionState::Pending),
                3 => Err(ExecutionError::AsyncFailure {
                    status: 99,
                    diagnostic: "injected fence wait failure".to_owned(),
                }),
                _ => Ok(ExecutionState::Success),
            }
        }

        fn token(&self) -> Result<u64, ExecutionError> {
            Ok(1)
        }
    }

    impl ExecutionSubmissionAdapter for FakeSubmission {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }

        fn finalize_after_fence(
            &mut self,
            _fence_token: u64,
        ) -> Result<ExecutionState, ExecutionError> {
            self.state.log(format!("finalize{}", self.generation));
            Ok(ExecutionState::Success)
        }

        fn start_output_readback(
            &mut self,
            _access: &ExecutionAdapterAccess<'_>,
            _output: &crate::OwnedTensorBinding,
        ) -> Result<Box<dyn ExecutionReadbackAdapter>, ExecutionError> {
            Err(ExecutionError::Unsupported {
                reason: "fake graph submission has no output readback".to_owned(),
            })
        }
    }

    impl ExecutionReadbackAdapter for FakeReadback {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }

        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            self.state.log(format!("readback_wait{}", self.generation));
            if self.state.fail_generation.load(Ordering::Acquire) == self.generation {
                return Err(ExecutionError::AsyncFailure {
                    status: 1,
                    diagnostic: "fake readback failure".to_owned(),
                });
            }
            Ok(ExecutionState::Success)
        }

        fn read_into(&mut self, destination: &mut [u8]) -> Result<u64, ExecutionError> {
            destination.copy_from_slice(&self.bytes);
            if self.state.truncate_readback.load(Ordering::Acquire) {
                Ok((destination.len() - 1) as u64)
            } else {
                Ok(destination.len() as u64)
            }
        }
    }

    impl ExecutionTransferAdapter for FakeTransfer {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }
        fn wait(&mut self, _timeout: Duration) -> Result<ExecutionState, ExecutionError> {
            Ok(ExecutionState::Success)
        }
    }

    fn fixture() -> (
        Arc<FakeReplayState>,
        Arc<ExecutionSession>,
        ExecutionGraphSpan,
        ExecutionBuffer,
    ) {
        let state = Arc::new(FakeReplayState::new());
        let session = Arc::new(ExecutionSession::new(
            "fake",
            Arc::new(FakeAdapter {
                state: Arc::clone(&state),
            }),
        ));
        let queue = session.create_queue().unwrap();
        let buffer = session.allocate(RESULT_RING_BYTES).unwrap();
        let input = TensorView::contiguous(DType::Bf16, &[1, 1]).unwrap();
        let output = TensorView::contiguous(DType::Bf16, &[1, 1]).unwrap();
        let input_buffer = session.allocate(input.payload_bytes()).unwrap();
        let output_buffer = session.allocate(output.payload_bytes()).unwrap();
        let operation = BoundSemanticOp::new(
            Arc::new(
                crate::SemanticOpDescriptor::new(
                    crate::SemanticOpKind::Copy,
                    vec![input.clone()],
                    vec![output.clone()],
                )
                .unwrap(),
            ),
            vec![
                session
                    .bind(&input_buffer, input, AccessMode::Read)
                    .unwrap(),
            ],
            vec![
                session
                    .bind(&output_buffer, output, AccessMode::Write)
                    .unwrap(),
            ],
        )
        .unwrap();
        let prepared = session.prepare(Arc::new(operation)).unwrap();
        let graph = session
            .create_graph_span(&queue, &[prepared], &[("fake".to_owned(), dispatch())])
            .unwrap();
        (state, session, graph, buffer)
    }

    #[test]
    fn replay_is_one_ahead_wraps_ring_and_publishes_after_discard() {
        let (state, session, graph, ring) = fixture();
        let control = DecodeControlV1::new(
            crate::DecodeControlModeV1::Mtp,
            2,
            1000,
            100,
            7,
            0x1234,
            32,
            3,
            64,
        )
        .unwrap();
        let mut replay = DecodeReplayController::new(
            session,
            graph,
            ring,
            control,
            64,
            Duration::from_millis(1),
        )
        .unwrap();
        replay.start().unwrap();
        assert_eq!(replay.next().unwrap().unwrap().count, 2);
        assert_eq!(replay.next().unwrap().unwrap().count, 1);
        assert!(replay.next().unwrap().unwrap().stopped());
        assert!(replay.finished());
        let audit = replay.audit();
        assert_eq!(audit.completed_replays(), 3);
        assert_eq!(audit.discarded_replays(), 1);
        assert_eq!(audit.executed_replays(), 4);
        assert_eq!(audit.executed_native_kernel_nodes_total(), 4);
        assert_eq!(audit.provider_dispatch_count(), 1);
        assert_eq!(state.published.lock().unwrap().as_slice(), &[(100, 105, 3)]);
        let log = state.log.lock().unwrap().join(",");
        assert!(log.find("prepare_paged:118").unwrap() < log.find("submit1").unwrap());
        assert!(log.find("submit2").unwrap() < log.find("readback_wait1").unwrap());
        assert!(log.contains("submit4"));
    }

    #[test]
    fn final_budget_replay_skips_successor_and_publishes_without_discard() {
        let (state, session, graph, ring) = fixture();
        state.budget_halt_generation.store(2, Ordering::Release);
        let control = DecodeControlV1::new(
            crate::DecodeControlModeV1::Mtp,
            2,
            1000,
            100,
            7,
            0x1234,
            4,
            3,
            64,
        )
        .unwrap();
        let mut replay = DecodeReplayController::new(
            session,
            graph,
            ring,
            control,
            64,
            Duration::from_millis(1),
        )
        .unwrap();
        replay.start().unwrap();
        assert_eq!(replay.next().unwrap().unwrap().count, 2);
        let terminal = replay.next().unwrap().unwrap();
        assert_eq!(terminal.count, 1);
        assert!(terminal.halted());
        assert!(replay.finished());
        let audit = replay.audit();
        assert_eq!(audit.completed_replays(), 2);
        assert_eq!(audit.discarded_replays(), 0);
        assert_eq!(audit.executed_replays(), 2);
        assert_eq!(audit.executed_native_kernel_nodes_total(), 2);
        assert_eq!(state.published.lock().unwrap().as_slice(), &[(100, 103, 2)]);
        let log = state.log.lock().unwrap().join(",");
        assert!(log.find("submit2").unwrap() < log.find("readback_wait1").unwrap());
        assert!(!log.contains("submit3"));
    }

    #[test]
    fn final_budget_replay_without_terminal_flag_fails_closed() {
        let (state, session, graph, ring) = fixture();
        let control = DecodeControlV1::new(
            crate::DecodeControlModeV1::Mtp,
            2,
            1000,
            100,
            7,
            0x1234,
            4,
            3,
            64,
        )
        .unwrap();
        let mut replay = DecodeReplayController::new(
            session,
            graph,
            ring,
            control,
            64,
            Duration::from_millis(1),
        )
        .unwrap();
        replay.start().unwrap();
        replay.next().unwrap();
        assert!(replay.next().is_err());
        assert!(replay.finished());
        assert!(state.published.lock().unwrap().is_empty());
        assert!(
            !state
                .log
                .lock()
                .unwrap()
                .iter()
                .any(|entry| entry == "submit3")
        );
    }

    #[test]
    fn unsuccessful_fence_never_finalizes_or_publishes_replay() {
        for failure_mode in 1..=3 {
            let (state, session, graph, ring) = fixture();
            state
                .fence_failure_mode
                .store(failure_mode, Ordering::Release);
            let control = DecodeControlV1::new(
                crate::DecodeControlModeV1::Mtp,
                2,
                1000,
                100,
                7,
                0x1234,
                32,
                3,
                64,
            )
            .unwrap();
            let mut replay = DecodeReplayController::new(
                session,
                graph,
                ring,
                control,
                64,
                Duration::from_millis(1),
            )
            .unwrap();
            replay.start().unwrap();
            assert!(replay.next().is_err());
            assert!(replay.finished());
            assert!(state.published.lock().unwrap().is_empty());
            assert!(
                !state
                    .log
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|entry| entry.starts_with("finalize"))
            );
        }
    }

    #[test]
    fn replay_does_not_publish_when_discard_drain_fails_or_readback_is_truncated() {
        let (state, session, graph, ring) = fixture();
        state.fail_generation.store(4, Ordering::Release);
        let control = DecodeControlV1::new(
            crate::DecodeControlModeV1::Mtp,
            2,
            1000,
            100,
            7,
            0x1234,
            32,
            3,
            64,
        )
        .unwrap();
        let mut replay = DecodeReplayController::new(
            Arc::clone(&session),
            graph,
            ring,
            control,
            64,
            Duration::from_millis(1),
        )
        .unwrap();
        replay.start().unwrap();
        replay.next().unwrap();
        replay.next().unwrap();
        assert!(replay.next().is_err());
        assert!(state.published.lock().unwrap().is_empty());

        let (state, session, graph, ring) = fixture();
        state.truncate_readback.store(true, Ordering::Release);
        let control = DecodeControlV1::new(
            crate::DecodeControlModeV1::Mtp,
            2,
            1000,
            100,
            7,
            0x1234,
            32,
            3,
            64,
        )
        .unwrap();
        let mut replay = DecodeReplayController::new(
            session,
            graph,
            ring,
            control,
            64,
            Duration::from_millis(1),
        )
        .unwrap();
        replay.start().unwrap();
        assert!(replay.next().is_err());
        assert!(state.published.lock().unwrap().is_empty());
    }
}
