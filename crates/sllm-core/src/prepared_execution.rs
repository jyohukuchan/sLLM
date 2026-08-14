//! Model-neutral prepared execution control.
//!
//! Model adapters lower their graph nodes into a [`PreparedExecutionPlan`]
//! and retain model-specific descriptor construction and state declarations.
//! This module owns cache identity, asynchronous segment ownership, boundary
//! accounting, and fail-closed request transaction state.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::execution::{
    BoundSemanticOp, CausalAttentionSubmission, ExecutionError, ExecutionSession, ExecutionState,
    KvStateAppendSubmission, LinearAttentionSubmission, OwnedTensorBinding, PreparedOperation,
    Submission,
};
use crate::{AccessMode, DispatchEvidence, SemanticOpDescriptor, TensorView};

/// A point at which ordered work must be terminal before externally visible
/// progress or host readback may continue.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExecutionBoundaryKind {
    StatePublication,
    TerminalReadback,
    Cancellation,
    Error,
}

/// Request-local values that may make otherwise identical prepared metadata
/// non-interchangeable.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PreparedDynamicIdentity {
    token_count: u64,
    start_position: Option<u64>,
    expected_length: Option<u64>,
    binding_generation: u64,
    state_generation: Option<u64>,
}

impl PreparedDynamicIdentity {
    pub const fn stateless(token_count: u64, binding_generation: u64) -> Self {
        Self {
            token_count,
            start_position: None,
            expected_length: None,
            binding_generation,
            state_generation: None,
        }
    }

    pub const fn stateful(
        token_count: u64,
        start_position: u64,
        expected_length: u64,
        binding_generation: u64,
        state_generation: u64,
    ) -> Self {
        Self {
            token_count,
            start_position: Some(start_position),
            expected_length: Some(expected_length),
            binding_generation,
            state_generation: Some(state_generation),
        }
    }

    pub const fn token_count(self) -> u64 {
        self.token_count
    }

    pub const fn start_position(self) -> Option<u64> {
        self.start_position
    }

    pub const fn expected_length(self) -> Option<u64> {
        self.expected_length
    }

    pub const fn binding_generation(self) -> u64 {
        self.binding_generation
    }

    pub const fn state_generation(self) -> Option<u64> {
        self.state_generation
    }
}

/// Whether a prepared operation may enter the request-local cache.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PreparedCachePolicy {
    Transient,
    Reusable(PreparedDynamicIdentity),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct PreparedBindingIdentity {
    buffer_id: crate::ExecutionBufferId,
    view: TensorView,
    access: AccessMode,
}

impl PreparedBindingIdentity {
    fn from_binding(binding: &OwnedTensorBinding) -> Self {
        Self {
            buffer_id: binding.buffer().id(),
            view: binding.view().clone(),
            access: binding.access(),
        }
    }
}

/// Exact cache identity. Labels are intentionally absent: a human-readable
/// graph label cannot prove descriptor, layout, storage, or dynamic equality.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct PreparedCacheKey {
    descriptor: SemanticOpDescriptor,
    inputs: Vec<PreparedBindingIdentity>,
    outputs: Vec<PreparedBindingIdentity>,
    dynamic: PreparedDynamicIdentity,
}

impl PreparedCacheKey {
    fn new(
        descriptor: &SemanticOpDescriptor,
        inputs: &[OwnedTensorBinding],
        outputs: &[OwnedTensorBinding],
        dynamic: PreparedDynamicIdentity,
    ) -> Self {
        Self {
            descriptor: descriptor.clone(),
            inputs: inputs
                .iter()
                .map(PreparedBindingIdentity::from_binding)
                .collect(),
            outputs: outputs
                .iter()
                .map(PreparedBindingIdentity::from_binding)
                .collect(),
            dynamic,
        }
    }
}

/// Request-owned semantic prepared-operation cache.
#[derive(Default)]
pub(crate) struct PreparedSemanticCache {
    entries: Mutex<HashMap<PreparedCacheKey, PreparedOperation>>,
}

impl PreparedSemanticCache {
    pub(crate) fn prepare(
        &self,
        session: &ExecutionSession,
        descriptor: SemanticOpDescriptor,
        inputs: Vec<OwnedTensorBinding>,
        outputs: Vec<OwnedTensorBinding>,
        policy: PreparedCachePolicy,
    ) -> Result<PreparedOperation, PreparedExecutionError> {
        let key = match policy {
            PreparedCachePolicy::Transient => None,
            PreparedCachePolicy::Reusable(dynamic) => Some(PreparedCacheKey::new(
                &descriptor,
                &inputs,
                &outputs,
                dynamic,
            )),
        };
        if let Some(key) = &key {
            if let Some(prepared) = self
                .entries
                .lock()
                .map_err(|_| PreparedExecutionError::Poisoned)?
                .get(key)
                .cloned()
            {
                return Ok(prepared);
            }
        }

        let operation = Arc::new(BoundSemanticOp::new(Arc::new(descriptor), inputs, outputs)?);
        let prepared = session.prepare(operation)?;
        if let Some(key) = key {
            self.entries
                .lock()
                .map_err(|_| PreparedExecutionError::Poisoned)?
                .insert(key, prepared.clone());
        }
        Ok(prepared)
    }
}

/// Immutable request transition values shared by model adapter nodes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PreparedTransition {
    token_count: u64,
    start_position: u64,
    expected_length: u64,
    binding_generation: u64,
    state_generation: u64,
}

impl PreparedTransition {
    pub fn new(
        token_count: u64,
        start_position: u64,
        binding_generation: u64,
        state_generation: u64,
    ) -> Result<Self, PreparedExecutionError> {
        if token_count == 0 {
            return Err(PreparedExecutionError::InvalidTransition(
                "transition token count must be non-zero".to_owned(),
            ));
        }
        let expected_length = start_position.checked_add(token_count).ok_or_else(|| {
            PreparedExecutionError::InvalidTransition(
                "transition expected length overflowed u64".to_owned(),
            )
        })?;
        Ok(Self {
            token_count,
            start_position,
            expected_length,
            binding_generation,
            state_generation,
        })
    }

    pub const fn token_count(self) -> u64 {
        self.token_count
    }

    pub const fn start_position(self) -> u64 {
        self.start_position
    }

    pub const fn expected_length(self) -> u64 {
        self.expected_length
    }

    pub const fn dynamic_identity(self) -> PreparedDynamicIdentity {
        PreparedDynamicIdentity::stateful(
            self.token_count,
            self.start_position,
            self.expected_length,
            self.binding_generation,
            self.state_generation,
        )
    }

    pub const fn stateless_identity(self) -> PreparedDynamicIdentity {
        PreparedDynamicIdentity::stateless(self.token_count, self.binding_generation)
    }
}

/// One adapter-owned operation and its declared boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedPlanNode<N> {
    operation: N,
    boundary_after: Option<ExecutionBoundaryKind>,
}

impl<N> PreparedPlanNode<N> {
    pub const fn new(operation: N, boundary_after: Option<ExecutionBoundaryKind>) -> Self {
        Self {
            operation,
            boundary_after,
        }
    }

    pub const fn operation(&self) -> &N {
        &self.operation
    }

    pub const fn boundary_after(&self) -> Option<ExecutionBoundaryKind> {
        self.boundary_after
    }
}

/// Immutable operation order produced by a model adapter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedExecutionPlan<N> {
    nodes: Arc<[PreparedPlanNode<N>]>,
}

impl<N> PreparedExecutionPlan<N> {
    pub fn new(nodes: Vec<PreparedPlanNode<N>>) -> Result<Self, PreparedExecutionError> {
        if nodes.is_empty() {
            return Err(PreparedExecutionError::InvalidPlan(
                "prepared execution plan must contain at least one node".to_owned(),
            ));
        }
        Ok(Self {
            nodes: nodes.into(),
        })
    }

    pub fn nodes(&self) -> &[PreparedPlanNode<N>] {
        &self.nodes
    }

    /// Visits every immutable node in plan order through a model adapter.
    /// The common layer owns ordering; the callback owns model-specific
    /// descriptor construction and output interpretation.
    pub fn execute<E>(
        &self,
        transition: PreparedTransition,
        mut adapter: impl FnMut(&PreparedPlanNode<N>, PreparedTransition) -> Result<(), E>,
    ) -> Result<(), E> {
        for node in self.nodes.iter() {
            adapter(node, transition)?;
        }
        Ok(())
    }
}

trait SegmentCompletionOwner: Send {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError>;
    fn dispatch(&self) -> &DispatchEvidence;
}

impl SegmentCompletionOwner for Submission {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        self.query()
    }

    fn dispatch(&self) -> &DispatchEvidence {
        self.dispatch()
    }
}

impl SegmentCompletionOwner for CausalAttentionSubmission {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        self.query()
    }

    fn dispatch(&self) -> &DispatchEvidence {
        self.dispatch()
    }
}

impl SegmentCompletionOwner for LinearAttentionSubmission {
    fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
        self.query()
    }

    fn dispatch(&self) -> &DispatchEvidence {
        self.dispatch()
    }
}

struct RetainedSubmission {
    label: String,
    owner: Box<dyn SegmentCompletionOwner>,
}

/// Completion owners retained until a declared boundary on one ordered queue.
#[derive(Default)]
pub(crate) struct ExecutionSegment {
    pending: Vec<RetainedSubmission>,
}

impl ExecutionSegment {
    pub(crate) fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    pub(crate) fn retain_semantic(&mut self, label: impl Into<String>, owner: Submission) {
        self.retain(label, owner);
    }

    pub(crate) fn retain_causal_attention(
        &mut self,
        label: impl Into<String>,
        owner: CausalAttentionSubmission,
    ) {
        self.retain(label, owner);
    }

    pub(crate) fn retain_linear_attention(
        &mut self,
        label: impl Into<String>,
        owner: LinearAttentionSubmission,
    ) {
        self.retain(label, owner);
    }

    fn retain(&mut self, label: impl Into<String>, owner: impl SegmentCompletionOwner + 'static) {
        self.pending.push(RetainedSubmission {
            label: label.into(),
            owner: Box::new(owner),
        });
    }

    pub(crate) fn flush(
        &mut self,
        boundary: ExecutionBoundaryKind,
        audit: &mut ExecutionAuditAccumulator,
    ) -> Result<(), PreparedExecutionError> {
        let had_work = !self.pending.is_empty();
        for mut retained in self.pending.drain(..) {
            require_terminal_success(&retained.label, retained.owner.query()?)?;
            audit.record(retained.owner.dispatch())?;
        }
        audit.record_boundary(boundary, had_work)?;
        Ok(())
    }

    #[cfg(test)]
    fn retain_test_owner(
        &mut self,
        label: impl Into<String>,
        owner: impl SegmentCompletionOwner + 'static,
    ) {
        self.retain(label, owner);
    }
}

/// Immutable common dispatch and boundary audit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedExecutionAudit {
    backend: u32,
    target: String,
    submission_count: u64,
    kernel_dispatch_count: u64,
    fallback_used: bool,
    segment_count: u64,
    boundary_count: u64,
}

impl PreparedExecutionAudit {
    pub const fn backend(&self) -> u32 {
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

    pub const fn segment_count(&self) -> u64 {
        self.segment_count
    }

    pub const fn boundary_count(&self) -> u64 {
        self.boundary_count
    }
}

pub(crate) struct ExecutionAuditAccumulator {
    expected_backend: u32,
    target: Option<String>,
    submission_count: u64,
    kernel_dispatch_count: u64,
    fallback_used: bool,
    segment_count: u64,
    boundary_count: u64,
}

impl ExecutionAuditAccumulator {
    pub(crate) const fn new(expected_backend: u32) -> Self {
        Self {
            expected_backend,
            target: None,
            submission_count: 0,
            kernel_dispatch_count: 0,
            fallback_used: false,
            segment_count: 0,
            boundary_count: 0,
        }
    }

    pub(crate) fn record(
        &mut self,
        evidence: &DispatchEvidence,
    ) -> Result<(), PreparedExecutionError> {
        if evidence.backend != self.expected_backend
            || evidence.fallback_allowed
            || evidence.fallback_used
            || evidence.dispatch_count == 0
            || evidence.target.is_empty()
            || !evidence.target.is_ascii()
            || evidence.target.as_bytes().contains(&0)
        {
            return Err(PreparedExecutionError::InvalidAudit(
                "dispatch evidence violates the exact backend/no-fallback policy".to_owned(),
            ));
        }
        if let Some(target) = &self.target {
            if target != &evidence.target {
                return Err(PreparedExecutionError::InvalidAudit(
                    "dispatch evidence targets differ within one request".to_owned(),
                ));
            }
        } else {
            self.target = Some(evidence.target.clone());
        }
        self.submission_count = self.submission_count.checked_add(1).ok_or_else(|| {
            PreparedExecutionError::InvalidAudit("submission count overflowed u64".to_owned())
        })?;
        self.kernel_dispatch_count = self
            .kernel_dispatch_count
            .checked_add(u64::from(evidence.dispatch_count))
            .ok_or_else(|| {
                PreparedExecutionError::InvalidAudit(
                    "kernel dispatch count overflowed u64".to_owned(),
                )
            })?;
        self.fallback_used |= evidence.fallback_used;
        Ok(())
    }

    pub(crate) fn record_boundary(
        &mut self,
        _kind: ExecutionBoundaryKind,
        closed_segment: bool,
    ) -> Result<(), PreparedExecutionError> {
        self.boundary_count = self.boundary_count.checked_add(1).ok_or_else(|| {
            PreparedExecutionError::InvalidAudit("boundary count overflowed u64".to_owned())
        })?;
        if closed_segment {
            self.segment_count = self.segment_count.checked_add(1).ok_or_else(|| {
                PreparedExecutionError::InvalidAudit("segment count overflowed u64".to_owned())
            })?;
        }
        Ok(())
    }

    pub(crate) fn snapshot(&self) -> Result<PreparedExecutionAudit, PreparedExecutionError> {
        let target = self.target.clone().ok_or_else(|| {
            PreparedExecutionError::InvalidAudit(
                "successful transition has an empty dispatch audit".to_owned(),
            )
        })?;
        if self.submission_count == 0
            || self.kernel_dispatch_count == 0
            || self.boundary_count == 0
            || self.fallback_used
        {
            return Err(PreparedExecutionError::InvalidAudit(
                "successful transition has an incomplete dispatch/boundary audit".to_owned(),
            ));
        }
        Ok(PreparedExecutionAudit {
            backend: self.expected_backend,
            target,
            submission_count: self.submission_count,
            kernel_dispatch_count: self.kernel_dispatch_count,
            fallback_used: self.fallback_used,
            segment_count: self.segment_count,
            boundary_count: self.boundary_count,
        })
    }
}

struct TransactionState {
    poisoned: AtomicBool,
    in_flight: AtomicBool,
}

/// Cloneable owner for one request's transaction lifecycle.
#[derive(Clone)]
pub(crate) struct ExecutionTransaction {
    state: Arc<TransactionState>,
}

impl ExecutionTransaction {
    pub(crate) fn new() -> Self {
        Self {
            state: Arc::new(TransactionState {
                poisoned: AtomicBool::new(false),
                in_flight: AtomicBool::new(false),
            }),
        }
    }

    pub(crate) fn begin(&self) -> Result<ExecutionTransactionGuard, PreparedExecutionError> {
        if self.state.poisoned.load(Ordering::Acquire) {
            return Err(PreparedExecutionError::Poisoned);
        }
        if self
            .state
            .in_flight
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(PreparedExecutionError::Busy);
        }
        if self.state.poisoned.load(Ordering::Acquire) {
            self.state.in_flight.store(false, Ordering::Release);
            return Err(PreparedExecutionError::Poisoned);
        }
        Ok(ExecutionTransactionGuard {
            state: Arc::clone(&self.state),
            committed: false,
        })
    }

    pub(crate) fn cancel(&self) {
        self.state.poisoned.store(true, Ordering::Release);
    }

    pub(crate) fn is_poisoned(&self) -> bool {
        self.state.poisoned.load(Ordering::Acquire)
    }

    #[cfg(test)]
    fn is_in_flight(&self) -> bool {
        self.state.in_flight.load(Ordering::Acquire)
    }
}

pub(crate) struct ExecutionTransactionGuard {
    state: Arc<TransactionState>,
    committed: bool,
}

impl ExecutionTransactionGuard {
    pub(crate) fn commit(&mut self) -> Result<(), PreparedExecutionError> {
        if self.state.poisoned.load(Ordering::Acquire) {
            return Err(PreparedExecutionError::Poisoned);
        }
        self.committed = true;
        self.state.in_flight.store(false, Ordering::Release);
        Ok(())
    }
}

impl Drop for ExecutionTransactionGuard {
    fn drop(&mut self) {
        if !self.committed {
            self.state.poisoned.store(true, Ordering::Release);
        }
        self.state.in_flight.store(false, Ordering::Release);
    }
}

/// Fail-closed errors from model-neutral execution control.
#[derive(Debug)]
pub enum PreparedExecutionError {
    InvalidPlan(String),
    InvalidTransition(String),
    InvalidAudit(String),
    Poisoned,
    Busy,
    CompletionPending { stage: String },
    CompletionFailure { stage: String },
    Execution(ExecutionError),
}

impl fmt::Display for PreparedExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan(reason) => write!(formatter, "invalid prepared plan: {reason}"),
            Self::InvalidTransition(reason) => {
                write!(formatter, "invalid prepared transition: {reason}")
            }
            Self::InvalidAudit(reason) => write!(formatter, "invalid execution audit: {reason}"),
            Self::Poisoned => formatter.write_str("request transaction is poisoned"),
            Self::Busy => formatter.write_str("request transaction is already in flight"),
            Self::CompletionPending { stage } => {
                write!(
                    formatter,
                    "{stage} remained pending at an execution boundary"
                )
            }
            Self::CompletionFailure { stage } => {
                write!(formatter, "{stage} failed at an execution boundary")
            }
            Self::Execution(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for PreparedExecutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Execution(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ExecutionError> for PreparedExecutionError {
    fn from(error: ExecutionError) -> Self {
        Self::Execution(error)
    }
}

pub(crate) fn require_terminal_success(
    stage: &str,
    state: ExecutionState,
) -> Result<(), PreparedExecutionError> {
    match state {
        ExecutionState::Success => Ok(()),
        ExecutionState::Pending => Err(PreparedExecutionError::CompletionPending {
            stage: stage.to_owned(),
        }),
        ExecutionState::Failure => Err(PreparedExecutionError::CompletionFailure {
            stage: stage.to_owned(),
        }),
    }
}

pub(crate) fn wait_terminal_submission(
    stage: &str,
    submission: &mut Submission,
    timeout: Duration,
) -> Result<(), PreparedExecutionError> {
    require_terminal_success(stage, submission.wait(timeout)?)
}

pub(crate) fn wait_terminal_kv_append(
    stage: &str,
    submission: &mut KvStateAppendSubmission,
    timeout: Duration,
) -> Result<(), PreparedExecutionError> {
    require_terminal_success(stage, submission.wait(timeout)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct SyntheticOwner {
        state: ExecutionState,
        query_error: bool,
        evidence: DispatchEvidence,
        drops: Arc<AtomicUsize>,
    }

    impl SegmentCompletionOwner for SyntheticOwner {
        fn query(&mut self) -> Result<ExecutionState, ExecutionError> {
            if self.query_error {
                Err(ExecutionError::AsyncFailure {
                    status: 7,
                    diagnostic: "synthetic query failure".to_owned(),
                })
            } else {
                Ok(self.state)
            }
        }

        fn dispatch(&self) -> &DispatchEvidence {
            &self.evidence
        }
    }

    impl Drop for SyntheticOwner {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn evidence(dispatch_count: u32) -> DispatchEvidence {
        DispatchEvidence {
            abi_version: 1,
            info_version: 1,
            dispatch_id: 1,
            dispatch_count,
            kernel_id: 1,
            workgroup_size_x: 1,
            grid_size_x: 1,
            row_count: 1,
            normalized_size: 1,
            backend: 1,
            fallback_allowed: false,
            fallback_used: false,
            kernel_symbol: "synthetic".to_owned(),
            device_symbol: "synthetic".to_owned(),
            target: "synthetic".to_owned(),
        }
    }

    #[test]
    fn plan_and_transition_cover_nonaligned_and_boundary_values() {
        for token_count in [3, 17, 255, 256, 257] {
            let transition = PreparedTransition::new(token_count, 11, 7, 5).unwrap();
            assert_eq!(transition.expected_length(), 11 + token_count);
            assert_eq!(transition.dynamic_identity().token_count(), token_count);
            assert_eq!(transition.dynamic_identity().start_position(), Some(11));
        }
        assert!(PreparedTransition::new(0, 0, 0, 0).is_err());
        assert!(PreparedTransition::new(1, u64::MAX, 0, 0).is_err());
        assert!(PreparedExecutionPlan::<u8>::new(Vec::new()).is_err());
        let plan = PreparedExecutionPlan::new(vec![
            PreparedPlanNode::new(3_u16, None),
            PreparedPlanNode::new(17_u16, Some(ExecutionBoundaryKind::TerminalReadback)),
        ])
        .unwrap();
        assert_eq!(plan.nodes()[0].operation(), &3);
        assert_eq!(
            plan.nodes()[1].boundary_after(),
            Some(ExecutionBoundaryKind::TerminalReadback)
        );
        let transition = PreparedTransition::new(3, 0, 1, 0).unwrap();
        let mut visited = Vec::new();
        plan.execute(transition, |node, current| {
            visited.push((*node.operation(), current.expected_length()));
            Ok::<(), ()>(())
        })
        .unwrap();
        assert_eq!(visited, vec![(3, 3), (17, 3)]);
    }

    #[test]
    fn dynamic_identity_invalidates_position_binding_and_state_changes() {
        let baseline = PreparedDynamicIdentity::stateful(17, 255, 272, 3, 9);
        assert_ne!(
            baseline,
            PreparedDynamicIdentity::stateful(17, 256, 273, 3, 9)
        );
        assert_ne!(
            baseline,
            PreparedDynamicIdentity::stateful(17, 255, 272, 4, 9)
        );
        assert_ne!(
            baseline,
            PreparedDynamicIdentity::stateful(17, 255, 272, 3, 10)
        );
        assert_ne!(
            PreparedDynamicIdentity::stateless(17, 3),
            PreparedDynamicIdentity::stateless(18, 3)
        );
    }

    #[test]
    fn cache_key_rejects_descriptor_binding_access_and_dynamic_staleness() {
        let view = TensorView::contiguous(crate::DType::Bf16, &[3, 17]).unwrap();
        let descriptor = SemanticOpDescriptor::new(
            crate::SemanticOpKind::Copy,
            vec![view.clone()],
            vec![view.clone()],
        )
        .unwrap();
        let binding = |id, access| PreparedBindingIdentity {
            buffer_id: crate::ExecutionBufferId::new(id),
            view: view.clone(),
            access,
        };
        let baseline = PreparedCacheKey {
            descriptor: descriptor.clone(),
            inputs: vec![binding(11, AccessMode::Read)],
            outputs: vec![binding(12, AccessMode::Write)],
            dynamic: PreparedDynamicIdentity::stateless(3, 7),
        };

        let mut changed_descriptor = baseline.clone();
        changed_descriptor.descriptor = SemanticOpDescriptor::new(
            crate::SemanticOpKind::Copy,
            vec![TensorView::contiguous(crate::DType::Bf16, &[3, 18]).unwrap()],
            vec![TensorView::contiguous(crate::DType::Bf16, &[3, 18]).unwrap()],
        )
        .unwrap();
        assert_ne!(baseline, changed_descriptor);

        let mut changed_buffer = baseline.clone();
        changed_buffer.inputs[0].buffer_id = crate::ExecutionBufferId::new(13);
        assert_ne!(baseline, changed_buffer);

        let mut changed_access = baseline.clone();
        changed_access.inputs[0].access = AccessMode::ReadWrite;
        assert_ne!(baseline, changed_access);

        let mut changed_generation = baseline.clone();
        changed_generation.dynamic = PreparedDynamicIdentity::stateless(3, 8);
        assert_ne!(baseline, changed_generation);
    }

    #[test]
    fn segment_holds_heterogeneous_owners_until_terminal_boundary() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut segment = ExecutionSegment::default();
        for dispatch_count in [1, 2, 1] {
            segment.retain_test_owner(
                "synthetic",
                SyntheticOwner {
                    state: ExecutionState::Success,
                    query_error: false,
                    evidence: evidence(dispatch_count),
                    drops: Arc::clone(&drops),
                },
            );
        }
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        let mut audit = ExecutionAuditAccumulator::new(1);
        segment
            .flush(ExecutionBoundaryKind::TerminalReadback, &mut audit)
            .unwrap();
        assert_eq!(drops.load(Ordering::Relaxed), 3);
        let snapshot = audit.snapshot().unwrap();
        assert_eq!(snapshot.submission_count(), 3);
        assert_eq!(snapshot.kernel_dispatch_count(), 4);
        assert_eq!(snapshot.segment_count(), 1);
        assert_eq!(snapshot.boundary_count(), 1);
    }

    #[test]
    fn segment_failure_drops_all_owners_and_never_records_success() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut segment = ExecutionSegment::default();
        for state in [
            ExecutionState::Success,
            ExecutionState::Failure,
            ExecutionState::Success,
        ] {
            segment.retain_test_owner(
                "synthetic",
                SyntheticOwner {
                    state,
                    query_error: false,
                    evidence: evidence(1),
                    drops: Arc::clone(&drops),
                },
            );
        }
        let mut audit = ExecutionAuditAccumulator::new(1);
        assert!(matches!(
            segment.flush(ExecutionBoundaryKind::Error, &mut audit),
            Err(PreparedExecutionError::CompletionFailure { .. })
        ));
        assert_eq!(drops.load(Ordering::Relaxed), 3);
        assert!(audit.snapshot().is_err());
    }

    #[test]
    fn pending_and_query_failure_drop_owners_without_boundary_publication() {
        for (label, state, query_error) in [
            ("pending", ExecutionState::Pending, false),
            ("query-error", ExecutionState::Success, true),
        ] {
            let drops = Arc::new(AtomicUsize::new(0));
            let mut segment = ExecutionSegment::default();
            segment.retain_test_owner(
                label,
                SyntheticOwner {
                    state,
                    query_error,
                    evidence: evidence(1),
                    drops: Arc::clone(&drops),
                },
            );
            let mut audit = ExecutionAuditAccumulator::new(1);
            let result = segment.flush(ExecutionBoundaryKind::Error, &mut audit);
            if query_error {
                assert!(matches!(result, Err(PreparedExecutionError::Execution(_))));
            } else {
                assert!(matches!(
                    result,
                    Err(PreparedExecutionError::CompletionPending { .. })
                ));
            }
            assert_eq!(drops.load(Ordering::Relaxed), 1);
            assert!(audit.snapshot().is_err());
        }
    }

    #[test]
    fn transaction_commit_drop_failure_and_cancel_are_fail_closed() {
        let committed = ExecutionTransaction::new();
        let mut guard = committed.begin().unwrap();
        assert!(committed.is_in_flight());
        guard.commit().unwrap();
        drop(guard);
        assert!(!committed.is_poisoned());
        assert!(!committed.is_in_flight());

        let cancelled_in_flight = ExecutionTransaction::new();
        let mut guard = cancelled_in_flight.begin().unwrap();
        cancelled_in_flight.cancel();
        assert!(matches!(
            guard.commit(),
            Err(PreparedExecutionError::Poisoned)
        ));
        drop(guard);
        assert!(cancelled_in_flight.is_poisoned());
        assert!(!cancelled_in_flight.is_in_flight());

        let dropped = ExecutionTransaction::new();
        drop(dropped.begin().unwrap());
        assert!(dropped.is_poisoned());
        assert!(!dropped.is_in_flight());
        assert!(matches!(
            dropped.begin(),
            Err(PreparedExecutionError::Poisoned)
        ));

        let cancelled = ExecutionTransaction::new();
        cancelled.cancel();
        assert!(cancelled.is_poisoned());
        assert!(matches!(
            cancelled.begin(),
            Err(PreparedExecutionError::Poisoned)
        ));
    }

    #[test]
    fn audit_rejects_backend_target_fallback_and_empty_boundary_contracts() {
        let mut wrong_backend = evidence(1);
        wrong_backend.backend = 2;
        assert!(
            ExecutionAuditAccumulator::new(1)
                .record(&wrong_backend)
                .is_err()
        );

        let mut fallback = evidence(1);
        fallback.fallback_used = true;
        assert!(ExecutionAuditAccumulator::new(1).record(&fallback).is_err());

        let mut audit = ExecutionAuditAccumulator::new(1);
        audit.record(&evidence(1)).unwrap();
        let mut other = evidence(1);
        other.target = "other".to_owned();
        assert!(audit.record(&other).is_err());
        assert!(audit.snapshot().is_err());
    }
}
