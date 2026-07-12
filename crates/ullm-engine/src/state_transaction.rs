// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Backend-independent request-state transaction boundaries.
//!
//! The transaction types in this module intentionally carry metadata and an
//! opaque payload type only.  A persistent-state owner creates snapshots and is
//! the only component that commits or aborts a prepared delta.  An executor
//! can read a snapshot through `&P` and return a delta, but this API never gives
//! an executor `&mut P` to persistent state.  Rust cannot prevent interior
//! mutability inside an arbitrary `P`; a payload implementation that mutates
//! its root through shared references violates this immutable snapshot/lease
//! contract and must be rejected by the owner adapter.

use std::{cmp::Ordering, fmt, num::NonZeroU64};

use crate::{
    execution_batch::{ExecutionBatch, MAX_EXECUTION_BATCH_STATE_BINDINGS, StateHandle},
    model_graph::StateId,
};

/// Maximum UTF-8 byte length of a state transaction error message.
pub const MAX_STATE_TRANSACTION_ERROR_MESSAGE_BYTES: usize = 1_024;

/// Low-cardinality location of a state transaction failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StateTransactionErrorStage {
    /// Validation of caller-provided metadata.
    Input,
    /// Construction or validation of a snapshot.
    Snapshot,
    /// Construction or validation of a prepared delta.
    Delta,
    /// Comparison of a delta against its snapshot.
    Validation,
    /// Owner-side begin operation.
    Begin,
    /// Owner-side commit operation.
    Commit,
    /// Owner-side abort operation.
    Abort,
    /// Checked progress arithmetic.
    Progress,
}

/// Low-cardinality category of a state transaction failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StateTransactionErrorKind {
    /// Caller metadata is malformed or inconsistent.
    InvalidInput,
    /// A bounded resource or collection limit was exceeded.
    Resource,
    /// The prepared batch nonce is no longer current.
    StaleNonce,
    /// A committed generation is no longer current.
    StaleGeneration,
    /// A lease generation does not identify the same lease.
    LeaseMismatch,
    /// A snapshot or delta belongs to another persistent owner instance.
    OwnerMismatch,
    /// The owner or backend encountered an internal failure.
    Internal,
}

/// A bounded, typed state transaction error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateTransactionError {
    stage: StateTransactionErrorStage,
    kind: StateTransactionErrorKind,
    message: String,
}

impl StateTransactionError {
    /// Constructs an error and bounds its UTF-8 message.
    ///
    /// This public escape hatch accepts arbitrary text, so callers are
    /// responsible for not placing prompt, token, state-payload, or sensitive
    /// owner data in the message.  Internal contract errors use static,
    /// low-cardinality text instead.
    pub fn new(
        stage: StateTransactionErrorStage,
        kind: StateTransactionErrorKind,
        message: impl Into<String>,
    ) -> Self {
        Self {
            stage,
            kind,
            message: truncate_utf8(message.into(), MAX_STATE_TRANSACTION_ERROR_MESSAGE_BYTES),
        }
    }

    /// Returns the failure stage.
    pub const fn stage(&self) -> StateTransactionErrorStage {
        self.stage
    }

    /// Returns the failure category.
    pub const fn kind(&self) -> StateTransactionErrorKind {
        self.kind
    }

    /// Returns the bounded diagnostic message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for StateTransactionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:?}/{:?}: {}",
            self.stage, self.kind, self.message
        )
    }
}

impl std::error::Error for StateTransactionError {}

/// A validated nonzero lease generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LeaseGeneration(NonZeroU64);

impl LeaseGeneration {
    /// Constructs a lease generation, rejecting zero at the boundary.
    pub fn new(value: u64) -> Result<Self, StateTransactionError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or_else(|| invalid_input("lease generation must be nonzero"))
    }

    /// Constructs a lease generation from an already validated nonzero value.
    pub const fn from_nonzero(value: NonZeroU64) -> Self {
        Self(value)
    }

    /// Returns the stable numeric lease generation.
    pub const fn get(self) -> u64 {
        self.0.get()
    }

    /// Returns the underlying nonzero integer.
    pub const fn as_nonzero(self) -> NonZeroU64 {
        self.0
    }
}

/// A validated identity for one persistent owner instance and its lifetime.
///
/// Owners must rotate this value when a registry instance is reset or
/// recreated.  It prevents a prepared delta from an old owner instance from
/// being applied to a new instance even when all handles happen to be reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateOwnerEpoch(NonZeroU64);

impl StateOwnerEpoch {
    /// Constructs an owner epoch, rejecting zero at the boundary.
    pub fn new(value: u64) -> Result<Self, StateTransactionError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or_else(|| invalid_input("state owner epoch must be nonzero"))
    }

    /// Constructs an owner epoch from an already validated nonzero value.
    pub const fn from_nonzero(value: NonZeroU64) -> Self {
        Self(value)
    }

    /// Returns the stable numeric owner epoch.
    pub const fn get(self) -> u64 {
        self.0.get()
    }

    /// Returns the underlying nonzero integer.
    pub const fn as_nonzero(self) -> NonZeroU64 {
        self.0
    }
}

/// One request-owned logical state identity and lease.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateKey {
    /// Scheduler/request identity.  Zero remains a valid generic identity.
    pub request_id: u64,
    /// Backend-independent graph state identity.
    pub state_id: StateId,
    /// Opaque physical state handle.
    pub handle: StateHandle,
    /// Lease generation used to reject stale handle reuse.
    pub lease_generation: LeaseGeneration,
}

impl StateKey {
    /// Creates and validates one state key.
    pub fn new(
        request_id: u64,
        state_id: StateId,
        handle: StateHandle,
        lease_generation: LeaseGeneration,
    ) -> Result<Self, StateTransactionError> {
        let key = Self {
            request_id,
            state_id,
            handle,
            lease_generation,
        };
        key.validate()?;
        Ok(key)
    }

    /// Revalidates the public key fields before an owner operation.
    pub fn validate(&self) -> Result<(), StateTransactionError> {
        self.state_id
            .validate()
            .map_err(|_| invalid_input("state key contains an invalid state ID"))?;
        if self.handle.get() == 0 {
            return Err(invalid_input("state key contains a zero state handle"));
        }
        if self.lease_generation.get() == 0 {
            return Err(invalid_input("state key contains a zero lease generation"));
        }
        Ok(())
    }
}

/// Committed length and absolute position tracked independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateProgress {
    committed_len: u64,
    absolute_position: u64,
}

impl StateProgress {
    /// Constructs progress metadata.  Both counters are independently valid.
    pub const fn new(committed_len: u64, absolute_position: u64) -> Self {
        Self {
            committed_len,
            absolute_position,
        }
    }

    /// Returns the committed retained length.
    pub const fn committed_len(self) -> u64 {
        self.committed_len
    }

    /// Returns the absolute position, which may differ from the retained len.
    pub const fn absolute_position(self) -> u64 {
        self.absolute_position
    }

    /// Advances both counters by `amount` with checked arithmetic.
    pub fn checked_advance(self, amount: u64) -> Result<Self, StateTransactionError> {
        self.checked_advance_by(amount, amount)
    }

    /// Advances retained length and absolute position independently.
    pub fn checked_advance_by(
        self,
        committed_len_delta: u64,
        absolute_position_delta: u64,
    ) -> Result<Self, StateTransactionError> {
        let committed_len = self
            .committed_len
            .checked_add(committed_len_delta)
            .ok_or_else(|| progress_overflow("committed length advance overflows"))?;
        let absolute_position = self
            .absolute_position
            .checked_add(absolute_position_delta)
            .ok_or_else(|| progress_overflow("absolute position advance overflows"))?;
        Ok(Self {
            committed_len,
            absolute_position,
        })
    }

    /// Alias that makes the independent sliding-window operation explicit.
    pub fn checked_advance_independent(
        self,
        committed_len_delta: u64,
        absolute_position_delta: u64,
    ) -> Result<Self, StateTransactionError> {
        self.checked_advance_by(committed_len_delta, absolute_position_delta)
    }
}

/// A state version that must still be current when a delta is committed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateBaseVersion {
    key: StateKey,
    committed_generation: u64,
}

impl StateBaseVersion {
    /// Creates and validates a base version.
    pub fn new(key: StateKey, committed_generation: u64) -> Result<Self, StateTransactionError> {
        key.validate()?;
        Ok(Self {
            key,
            committed_generation,
        })
    }

    /// Returns the complete leased state key.
    pub fn key(&self) -> &StateKey {
        &self.key
    }

    /// Returns the generation read by the executor.
    pub const fn committed_generation(&self) -> u64 {
        self.committed_generation
    }
}

/// One immutable snapshot entry with an opaque backend payload.
///
/// The accessor exposes only `&P`.  Interior mutability is a property of the
/// payload type and cannot be forbidden by this generic boundary; using it to
/// mutate a persistent root through a shared reference violates the owner
/// lease/root contract.
pub struct StateSnapshot<P> {
    base: StateBaseVersion,
    progress: StateProgress,
    payload: P,
}

impl<P> StateSnapshot<P> {
    /// Creates and validates one snapshot entry.
    pub fn new(
        base: StateBaseVersion,
        progress: StateProgress,
        payload: P,
    ) -> Result<Self, StateTransactionError> {
        base.key.validate()?;
        Ok(Self {
            base,
            progress,
            payload,
        })
    }

    /// Returns the base version without exposing payload mutation.
    pub fn base(&self) -> &StateBaseVersion {
        &self.base
    }

    /// Returns committed length and absolute position.
    pub const fn progress(&self) -> StateProgress {
        self.progress
    }

    /// Returns a read-only view of the opaque payload.
    pub fn payload(&self) -> &P {
        &self.payload
    }
}

/// An owner epoch, batch nonce, and all snapshots read for one execution batch.
pub struct StateSnapshotSet<P> {
    owner_epoch: StateOwnerEpoch,
    batch_nonce: NonZeroU64,
    entries: Vec<StateSnapshot<P>>,
}

impl<P> StateSnapshotSet<P> {
    /// Creates and validates a snapshot set without allocating by payload size.
    pub fn new(
        owner_epoch: StateOwnerEpoch,
        batch_nonce: NonZeroU64,
        entries: Vec<StateSnapshot<P>>,
    ) -> Result<Self, StateTransactionError> {
        validate_snapshot_entries(&entries)?;
        Ok(Self {
            owner_epoch,
            batch_nonce,
            entries,
        })
    }

    /// Revalidates bounded metadata and uniqueness constraints.
    pub fn validate(&self) -> Result<(), StateTransactionError> {
        validate_snapshot_entries(&self.entries)
    }

    /// Returns the persistent owner epoch associated with this snapshot.
    pub const fn owner_epoch(&self) -> StateOwnerEpoch {
        self.owner_epoch
    }

    /// Returns the nonce associated with this snapshot batch.
    pub const fn batch_nonce(&self) -> NonZeroU64 {
        self.batch_nonce
    }

    /// Returns immutable snapshot entries.
    pub fn entries(&self) -> &[StateSnapshot<P>] {
        &self.entries
    }

    /// Returns a read-only payload view for one snapshot.
    pub fn payload(&self, index: usize) -> Option<&P> {
        self.entries.get(index).map(StateSnapshot::payload)
    }
}

/// A backend-prepared state mutation that can be consumed only by commit/abort.
#[must_use = "a prepared state delta must be committed or aborted"]
pub struct PreparedStateDelta<P> {
    owner_epoch: StateOwnerEpoch,
    batch_nonce: NonZeroU64,
    bases: Vec<StateBaseVersion>,
    payload: P,
}

impl<P> PreparedStateDelta<P> {
    /// Creates and validates a prepared delta's metadata.
    pub fn new(
        owner_epoch: StateOwnerEpoch,
        batch_nonce: NonZeroU64,
        bases: Vec<StateBaseVersion>,
        payload: P,
    ) -> Result<Self, StateTransactionError> {
        validate_base_versions(&bases, StateTransactionErrorStage::Delta)?;
        Ok(Self {
            owner_epoch,
            batch_nonce,
            bases,
            payload,
        })
    }

    /// Revalidates nonce-independent delta metadata.
    pub fn validate(&self) -> Result<(), StateTransactionError> {
        validate_base_versions(&self.bases, StateTransactionErrorStage::Delta)
    }

    /// Returns the persistent owner epoch associated with this delta.
    pub const fn owner_epoch(&self) -> StateOwnerEpoch {
        self.owner_epoch
    }

    /// Returns the nonce associated with this prepared delta.
    pub const fn batch_nonce(&self) -> NonZeroU64 {
        self.batch_nonce
    }

    /// Returns immutable base versions read by the executor.
    pub fn bases(&self) -> &[StateBaseVersion] {
        &self.bases
    }

    /// Returns an immutable view of the opaque prepared payload.
    pub fn payload(&self) -> &P {
        &self.payload
    }

    /// Validates nonce, key set, lease generations, and committed generations
    /// against a snapshot.  No payload is cloned or inspected.
    pub fn validate_against_snapshot<Q>(
        &self,
        snapshots: &StateSnapshotSet<Q>,
    ) -> Result<(), StateTransactionError> {
        self.validate()?;
        snapshots.validate()?;
        if self.owner_epoch != snapshots.owner_epoch {
            return Err(StateTransactionError::new(
                StateTransactionErrorStage::Validation,
                StateTransactionErrorKind::OwnerMismatch,
                "prepared delta owner epoch does not match snapshot owner",
            ));
        }
        if self.batch_nonce != snapshots.batch_nonce {
            return Err(StateTransactionError::new(
                StateTransactionErrorStage::Validation,
                StateTransactionErrorKind::StaleNonce,
                "prepared delta nonce does not match snapshot nonce",
            ));
        }
        if self.bases.len() != snapshots.entries.len() {
            return Err(invalid_at(
                StateTransactionErrorStage::Validation,
                "prepared delta base count does not match snapshot count",
            ));
        }

        let snapshots_by_key = try_sorted_refs(
            &snapshots.entries,
            StateTransactionErrorStage::Validation,
            |left, right| left.base.key.cmp(&right.base.key),
        )?;
        let snapshots_by_request_state = try_sorted_refs(
            &snapshots.entries,
            StateTransactionErrorStage::Validation,
            |left, right| {
                left.base
                    .key
                    .request_id
                    .cmp(&right.base.key.request_id)
                    .then_with(|| left.base.key.state_id.cmp(&right.base.key.state_id))
            },
        )?;
        for base in &self.bases {
            match snapshots_by_key.binary_search_by(|snapshot| snapshot.base.key.cmp(base.key())) {
                Ok(index)
                    if snapshots_by_key[index].base.committed_generation
                        == base.committed_generation => {}
                Ok(_) => {
                    return Err(StateTransactionError::new(
                        StateTransactionErrorStage::Validation,
                        StateTransactionErrorKind::StaleGeneration,
                        "prepared delta committed generation is stale",
                    ));
                }
                Err(_)
                    if snapshots_by_request_state
                        .binary_search_by(|snapshot| {
                            snapshot
                                .base
                                .key
                                .request_id
                                .cmp(&base.key.request_id)
                                .then_with(|| snapshot.base.key.state_id.cmp(&base.key.state_id))
                        })
                        .is_ok() =>
                {
                    return Err(StateTransactionError::new(
                        StateTransactionErrorStage::Validation,
                        StateTransactionErrorKind::LeaseMismatch,
                        "prepared delta lease generation does not match snapshot",
                    ));
                }
                Err(_) => {
                    return Err(invalid_at(
                        StateTransactionErrorStage::Validation,
                        "prepared delta contains a state base absent from snapshot",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Consumes the delta into parts for an owner adapter's atomic commit or
    /// abort implementation.
    ///
    /// This is an ownership/protocol boundary, not a security capability.  An
    /// arbitrary caller can always drop a delta or decompose it, and doing so
    /// does not commit anything.  The method deliberately does not perform
    /// owner epoch, nonce, lease, generation, readiness-fence, quarantine, or
    /// root-swap checks; a caller that is not an audited
    /// [`StateTransactionOwner`] implementation is violating the contract.
    /// Production integrations must recheck every owner epoch, nonce, current
    /// request, state key/lease, and committed generation before mutating a
    /// root.  A future external backend adapter should encapsulate those checks
    /// rather than treating this method as a standalone commit API.
    pub fn into_parts(self) -> (StateOwnerEpoch, NonZeroU64, Vec<StateBaseVersion>, P) {
        (self.owner_epoch, self.batch_nonce, self.bases, self.payload)
    }
}

/// Persistent owner boundary for backend-independent state transactions.
pub trait StateTransactionOwner {
    /// Returns the owner instance/lifetime epoch.  A reset or recreation must
    /// rotate this value before any new transaction is admitted.
    fn owner_epoch(&self) -> StateOwnerEpoch;

    /// Opaque backend payload returned by [`Self::begin`].
    type SnapshotPayload;
    /// Opaque backend payload prepared by the executor and consumed by commit/abort.
    type DeltaPayload;
    /// Owner-defined receipt returned after an atomic commit.
    type CommitReceipt;

    /// Reads all request leases and creates one batch snapshot.
    ///
    /// The owner must validate the complete batch lease set before returning,
    /// and the returned snapshot must carry [`Self::owner_epoch`].
    /// The executor may inspect snapshot payloads through shared references only;
    /// no persistent payload is handed out mutably.
    fn begin(
        &mut self,
        batch: &ExecutionBatch,
    ) -> Result<StateSnapshotSet<Self::SnapshotPayload>, StateTransactionError>;

    /// Atomically commits a consumed delta after rechecking the current owner
    /// epoch, request,
    /// batch nonce, lease generations, and committed generations.  A successful
    /// implementation performs one root swap (copy-on-write or equivalent),
    /// never a partial in-place persistent mutation.  A multi-request batch is
    /// atomic as one unit.  Backend readiness fences remain the delta payload's
    /// responsibility.  Unknown synchronization state must be quarantined.
    /// The implementation may consume the opaque non-`Clone` payload through
    /// [`PreparedStateDelta::into_parts`], but decomposing it alone is never a
    /// commit and the Rust type system cannot force callers to invoke this
    /// method only from an owner implementation.
    fn commit(
        &mut self,
        delta: PreparedStateDelta<Self::DeltaPayload>,
    ) -> Result<Self::CommitReceipt, StateTransactionError>;

    /// Aborts a consumed delta while leaving committed state unchanged.
    ///
    /// The owner must recheck its epoch and all relevant lease metadata before
    /// accepting the abort.  The trait shape cannot enforce that every backend
    /// adapter performs these semantic checks; adapters are responsible for
    /// preserving this contract.
    ///
    /// Abort must be safe after a failed execution or synchronization fence;
    /// unknown backend state is quarantined rather than guessed to be ready.
    fn abort(
        &mut self,
        delta: PreparedStateDelta<Self::DeltaPayload>,
    ) -> Result<(), StateTransactionError>;
}

fn validate_snapshot_entries<P>(entries: &[StateSnapshot<P>]) -> Result<(), StateTransactionError> {
    if entries.len() > MAX_EXECUTION_BATCH_STATE_BINDINGS {
        return Err(StateTransactionError::new(
            StateTransactionErrorStage::Snapshot,
            StateTransactionErrorKind::Resource,
            "snapshot state entry count exceeds execution batch limit",
        ));
    }
    for snapshot in entries {
        snapshot.base.key.validate()?;
    }
    let by_key = try_sorted_refs(
        entries,
        StateTransactionErrorStage::Snapshot,
        |left, right| left.base.key.cmp(&right.base.key),
    )?;
    if by_key
        .windows(2)
        .any(|window| window[0].base.key == window[1].base.key)
    {
        return Err(invalid_at(
            StateTransactionErrorStage::Snapshot,
            "snapshot set contains a duplicate state key",
        ));
    }
    let by_handle = try_sorted_refs(
        entries,
        StateTransactionErrorStage::Snapshot,
        |left, right| left.base.key.handle.cmp(&right.base.key.handle),
    )?;
    if by_handle
        .windows(2)
        .any(|window| window[0].base.key.handle == window[1].base.key.handle)
    {
        return Err(invalid_at(
            StateTransactionErrorStage::Snapshot,
            "snapshot set contains a duplicate state handle",
        ));
    }
    let by_request_state = try_sorted_refs(
        entries,
        StateTransactionErrorStage::Snapshot,
        |left, right| {
            left.base
                .key
                .request_id
                .cmp(&right.base.key.request_id)
                .then_with(|| left.base.key.state_id.cmp(&right.base.key.state_id))
        },
    )?;
    if by_request_state.windows(2).any(|window| {
        window[0].base.key.request_id == window[1].base.key.request_id
            && window[0].base.key.state_id == window[1].base.key.state_id
    }) {
        return Err(invalid_at(
            StateTransactionErrorStage::Snapshot,
            "snapshot set contains a duplicate request state ID",
        ));
    }
    Ok(())
}

fn validate_base_versions(
    bases: &[StateBaseVersion],
    stage: StateTransactionErrorStage,
) -> Result<(), StateTransactionError> {
    if bases.len() > MAX_EXECUTION_BATCH_STATE_BINDINGS {
        return Err(StateTransactionError::new(
            stage,
            StateTransactionErrorKind::Resource,
            "prepared state base count exceeds execution batch limit",
        ));
    }
    for base in bases {
        base.key.validate()?;
    }
    let by_key = try_sorted_refs(bases, stage, |left, right| left.key.cmp(&right.key))?;
    if by_key
        .windows(2)
        .any(|window| window[0].key == window[1].key)
    {
        return Err(invalid_at(
            stage,
            "prepared state bases contain a duplicate key",
        ));
    }
    let by_handle = try_sorted_refs(bases, stage, |left, right| {
        left.key.handle.cmp(&right.key.handle)
    })?;
    if by_handle
        .windows(2)
        .any(|window| window[0].key.handle == window[1].key.handle)
    {
        return Err(invalid_at(
            stage,
            "prepared state bases contain a duplicate state handle",
        ));
    }
    let by_request_state = try_sorted_refs(bases, stage, |left, right| {
        left.key
            .request_id
            .cmp(&right.key.request_id)
            .then_with(|| left.key.state_id.cmp(&right.key.state_id))
    })?;
    if by_request_state.windows(2).any(|window| {
        window[0].key.request_id == window[1].key.request_id
            && window[0].key.state_id == window[1].key.state_id
    }) {
        return Err(invalid_at(
            stage,
            "prepared state bases contain a duplicate request state ID",
        ));
    }
    Ok(())
}

fn try_sorted_refs<'a, T, F>(
    values: &'a [T],
    stage: StateTransactionErrorStage,
    compare: F,
) -> Result<Vec<&'a T>, StateTransactionError>
where
    F: Fn(&T, &T) -> Ordering,
{
    let mut refs = Vec::new();
    refs.try_reserve_exact(values.len()).map_err(|_| {
        StateTransactionError::new(
            stage,
            StateTransactionErrorKind::Resource,
            "state metadata validation workspace allocation failed",
        )
    })?;
    refs.extend(values.iter());
    refs.sort_unstable_by(|left, right| compare(left, right));
    Ok(refs)
}

fn truncate_utf8(mut value: String, limit: usize) -> String {
    if value.len() <= limit {
        return value;
    }
    let mut boundary = limit;
    while !value.is_char_boundary(boundary) {
        boundary -= 1;
    }
    value.truncate(boundary);
    value
}

fn invalid_input(message: &'static str) -> StateTransactionError {
    invalid_at(StateTransactionErrorStage::Input, message)
}

fn invalid_at(stage: StateTransactionErrorStage, message: &'static str) -> StateTransactionError {
    StateTransactionError::new(stage, StateTransactionErrorKind::InvalidInput, message)
}

fn progress_overflow(message: &'static str) -> StateTransactionError {
    StateTransactionError::new(
        StateTransactionErrorStage::Progress,
        StateTransactionErrorKind::InvalidInput,
        message,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_batch::{
        BatchStateBinding, ExecutionBatchItem, ExecutionPhase, TokenRange, WorkspacePlan,
    };

    fn nz(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).expect("test nonce must be nonzero")
    }

    fn epoch(value: u64) -> StateOwnerEpoch {
        StateOwnerEpoch::new(value).expect("test owner epoch must be nonzero")
    }

    fn state_id(value: &str) -> StateId {
        StateId::new(value).expect("test state ID must be valid")
    }

    fn key(request_id: u64, state_name: &str, handle: u64, lease: u64) -> StateKey {
        StateKey::new(
            request_id,
            state_id(state_name),
            StateHandle::new(handle).expect("test handle must be nonzero"),
            LeaseGeneration::new(lease).expect("test lease must be nonzero"),
        )
        .expect("test state key must be valid")
    }

    fn base(
        request_id: u64,
        state_name: &str,
        handle: u64,
        lease: u64,
        generation: u64,
    ) -> StateBaseVersion {
        StateBaseVersion::new(key(request_id, state_name, handle, lease), generation)
            .expect("test base must be valid")
    }

    fn snapshot(
        request_id: u64,
        state_name: &str,
        handle: u64,
        lease: u64,
        generation: u64,
        progress: StateProgress,
    ) -> StateSnapshot<Vec<u8>> {
        StateSnapshot::new(
            base(request_id, state_name, handle, lease, generation),
            progress,
            vec![1, 2, 3],
        )
        .expect("test snapshot must be valid")
    }

    fn batch_fixture() -> ExecutionBatch {
        ExecutionBatch {
            phase: ExecutionPhase::ColdPrefill,
            compatibility_key_sha256: "a".repeat(64),
            commit_nonce: 7,
            common_chunk_width: 1,
            packed_token_count: 1,
            items: vec![ExecutionBatchItem {
                request_id: 11,
                packed: TokenRange::new(0, 1),
                prefix_len: 0,
                absolute_start_position: 0,
                source: TokenRange::new(0, 1),
                destination: TokenRange::new(0, 1),
                state_bindings: vec![BatchStateBinding {
                    state_id: state_id("state"),
                    handle: StateHandle::new(9).expect("test handle"),
                    uses_paged_kv: false,
                }],
                block_table: Vec::new(),
            }],
            workspace: WorkspacePlan {
                capacity_bytes: 1_000,
                resident_bytes: 100,
                persistent_state_bytes: 100,
                temporary_activation_bytes: 100,
                operator_workspace_bytes: 100,
                required_headroom_bytes: 100,
            },
        }
    }

    #[test]
    fn valid_snapshot_delta_and_validation() {
        let set = StateSnapshotSet::new(
            epoch(1),
            nz(7),
            vec![snapshot(1, "layer", 10, 3, 4, StateProgress::new(6, 9))],
        )
        .expect("valid snapshot set");
        let delta = PreparedStateDelta::new(
            epoch(1),
            nz(7),
            vec![base(1, "layer", 10, 3, 4)],
            vec![4_u8],
        )
        .expect("valid delta");
        assert_eq!(set.batch_nonce(), nz(7));
        assert_eq!(set.owner_epoch(), epoch(1));
        assert_eq!(set.payload(0), Some(&vec![1, 2, 3]));
        delta
            .validate_against_snapshot(&set)
            .expect("delta should match snapshot");
    }

    #[test]
    fn duplicate_handle_key_and_request_state_id_are_rejected() {
        let duplicate_handle = StateSnapshotSet::new(
            epoch(1),
            nz(1),
            vec![
                snapshot(1, "a", 10, 1, 1, StateProgress::new(0, 0)),
                snapshot(1, "b", 10, 1, 1, StateProgress::new(0, 0)),
            ],
        );
        assert_eq!(
            duplicate_handle
                .err()
                .expect("duplicate handle must reject")
                .kind(),
            StateTransactionErrorKind::InvalidInput
        );

        let duplicate_state_id = StateSnapshotSet::new(
            epoch(1),
            nz(1),
            vec![
                snapshot(1, "a", 10, 1, 1, StateProgress::new(0, 0)),
                snapshot(1, "a", 11, 1, 1, StateProgress::new(0, 0)),
            ],
        );
        assert!(duplicate_state_id.is_err());

        let duplicate_key = PreparedStateDelta::new(
            epoch(1),
            nz(1),
            vec![base(1, "a", 10, 1, 1), base(1, "a", 10, 1, 1)],
            vec![0_u8],
        );
        assert!(duplicate_key.is_err());
    }

    #[test]
    fn same_state_id_is_allowed_across_distinct_requests() {
        let set = StateSnapshotSet::new(
            epoch(1),
            nz(1),
            vec![
                snapshot(1, "shared", 10, 1, 2, StateProgress::new(0, 0)),
                snapshot(2, "shared", 11, 1, 3, StateProgress::new(4, 7)),
            ],
        )
        .expect("state IDs are request-scoped");
        let delta = PreparedStateDelta::new(
            epoch(1),
            nz(1),
            vec![base(2, "shared", 11, 1, 3), base(1, "shared", 10, 1, 2)],
            vec![0_u8],
        )
        .expect("multi-request delta");
        delta
            .validate_against_snapshot(&set)
            .expect("request-scoped IDs should validate");
    }

    #[test]
    fn zero_lease_is_rejected_and_nonce_is_nonzero_at_boundary() {
        assert!(LeaseGeneration::new(0).is_err());
        assert!(StateOwnerEpoch::new(0).is_err());
        assert!(NonZeroU64::new(0).is_none());
        assert!(NonZeroU64::new(1).is_some());
    }

    #[test]
    fn stale_nonce_generation_lease_and_missing_extra_bases_fail_closed() {
        let set = StateSnapshotSet::new(
            epoch(1),
            nz(7),
            vec![snapshot(1, "layer", 10, 3, 4, StateProgress::new(6, 9))],
        )
        .expect("valid snapshot");

        let stale_nonce = PreparedStateDelta::new(
            epoch(1),
            nz(8),
            vec![base(1, "layer", 10, 3, 4)],
            vec![0_u8],
        )
        .expect("delta metadata");
        assert_eq!(
            stale_nonce
                .validate_against_snapshot(&set)
                .unwrap_err()
                .kind(),
            StateTransactionErrorKind::StaleNonce
        );

        let owner_mismatch = PreparedStateDelta::new(
            epoch(2),
            nz(7),
            vec![base(1, "layer", 10, 3, 4)],
            vec![0_u8],
        )
        .expect("delta metadata");
        assert_eq!(
            owner_mismatch
                .validate_against_snapshot(&set)
                .unwrap_err()
                .kind(),
            StateTransactionErrorKind::OwnerMismatch
        );

        let stale_generation = PreparedStateDelta::new(
            epoch(1),
            nz(7),
            vec![base(1, "layer", 10, 3, 5)],
            vec![0_u8],
        )
        .expect("delta metadata");
        assert_eq!(
            stale_generation
                .validate_against_snapshot(&set)
                .unwrap_err()
                .kind(),
            StateTransactionErrorKind::StaleGeneration
        );

        let lease_mismatch = PreparedStateDelta::new(
            epoch(1),
            nz(7),
            vec![base(1, "layer", 10, 4, 4)],
            vec![0_u8],
        )
        .expect("delta metadata");
        assert_eq!(
            lease_mismatch
                .validate_against_snapshot(&set)
                .unwrap_err()
                .kind(),
            StateTransactionErrorKind::LeaseMismatch
        );

        let missing = PreparedStateDelta::new(
            epoch(1),
            nz(7),
            vec![base(1, "other", 11, 3, 4)],
            vec![0_u8],
        )
        .expect("delta metadata");
        assert!(missing.validate_against_snapshot(&set).is_err());

        let extra = PreparedStateDelta::new(
            epoch(1),
            nz(7),
            vec![base(1, "layer", 10, 3, 4), base(2, "other", 11, 3, 4)],
            vec![0_u8],
        )
        .expect("delta metadata");
        assert!(extra.validate_against_snapshot(&set).is_err());
    }

    #[test]
    fn sliding_window_progress_is_independent_and_checked() {
        let progress = StateProgress::new(8, 100)
            .checked_advance(4)
            .expect("ordinary progress advance");
        assert_eq!(progress, StateProgress::new(12, 104));
        let sliding = progress
            .checked_advance_independent(2, 10)
            .expect("sliding progress advance");
        assert_eq!(sliding, StateProgress::new(14, 114));
        assert!(StateProgress::new(u64::MAX, 0).checked_advance(1).is_err());
        assert!(StateProgress::new(0, u64::MAX).checked_advance(1).is_err());
    }

    #[test]
    fn error_messages_are_bounded_at_utf8_boundaries() {
        let error = StateTransactionError::new(
            StateTransactionErrorStage::Input,
            StateTransactionErrorKind::InvalidInput,
            "あ".repeat(1_000),
        );
        assert!(error.message().len() <= MAX_STATE_TRANSACTION_ERROR_MESSAGE_BYTES);
        assert!(std::str::from_utf8(error.message().as_bytes()).is_ok());
    }

    struct FakeOwner {
        owner_epoch: StateOwnerEpoch,
        current_nonce: NonZeroU64,
        current_key: StateKey,
        generation: u64,
        committed: Vec<u8>,
    }

    impl StateTransactionOwner for FakeOwner {
        type SnapshotPayload = Vec<u8>;
        type DeltaPayload = Vec<u8>;
        type CommitReceipt = usize;

        fn owner_epoch(&self) -> StateOwnerEpoch {
            self.owner_epoch
        }

        fn begin(
            &mut self,
            batch: &ExecutionBatch,
        ) -> Result<StateSnapshotSet<Self::SnapshotPayload>, StateTransactionError> {
            batch.validate().map_err(|_| {
                invalid_at(
                    StateTransactionErrorStage::Begin,
                    "execution batch is invalid",
                )
            })?;
            let [item] = batch.items.as_slice() else {
                return Err(invalid_at(
                    StateTransactionErrorStage::Begin,
                    "reference owner requires exactly one batch item",
                ));
            };
            let [binding] = item.state_bindings.as_slice() else {
                return Err(invalid_at(
                    StateTransactionErrorStage::Begin,
                    "reference owner requires exactly one state binding",
                ));
            };
            let batch_nonce = NonZeroU64::new(batch.commit_nonce).ok_or_else(|| {
                StateTransactionError::new(
                    StateTransactionErrorStage::Begin,
                    StateTransactionErrorKind::StaleNonce,
                    "execution batch nonce is zero",
                )
            })?;
            if batch_nonce != self.current_nonce {
                return Err(StateTransactionError::new(
                    StateTransactionErrorStage::Begin,
                    StateTransactionErrorKind::StaleNonce,
                    "execution batch nonce is stale",
                ));
            }
            let state_key = StateKey::new(
                item.request_id,
                binding.state_id.clone(),
                binding.handle,
                LeaseGeneration::new(1)?,
            )?;
            if state_key != self.current_key {
                return Err(StateTransactionError::new(
                    StateTransactionErrorStage::Begin,
                    StateTransactionErrorKind::LeaseMismatch,
                    "execution batch state lease is not current",
                ));
            }
            let base = StateBaseVersion::new(state_key, self.generation)?;
            StateSnapshotSet::new(
                self.owner_epoch,
                batch_nonce,
                vec![StateSnapshot::new(
                    base,
                    StateProgress::new(item.prefix_len, item.absolute_start_position),
                    self.committed.clone(),
                )?],
            )
        }

        fn commit(
            &mut self,
            delta: PreparedStateDelta<Self::DeltaPayload>,
        ) -> Result<Self::CommitReceipt, StateTransactionError> {
            let (owner_epoch, nonce, bases, payload) = delta.into_parts();
            if owner_epoch != self.owner_epoch {
                return Err(StateTransactionError::new(
                    StateTransactionErrorStage::Commit,
                    StateTransactionErrorKind::OwnerMismatch,
                    "prepared delta owner epoch is stale",
                ));
            }
            if nonce != self.current_nonce {
                return Err(StateTransactionError::new(
                    StateTransactionErrorStage::Commit,
                    StateTransactionErrorKind::StaleNonce,
                    "prepared delta nonce is stale",
                ));
            }
            if bases.len() != 1 {
                return Err(invalid_at(
                    StateTransactionErrorStage::Commit,
                    "reference owner requires exactly one state base",
                ));
            }
            let base = &bases[0];
            if base.key() != &self.current_key {
                return Err(StateTransactionError::new(
                    StateTransactionErrorStage::Commit,
                    StateTransactionErrorKind::LeaseMismatch,
                    "prepared delta state lease is not current",
                ));
            }
            if base.committed_generation() != self.generation {
                return Err(StateTransactionError::new(
                    StateTransactionErrorStage::Commit,
                    StateTransactionErrorKind::StaleGeneration,
                    "current committed generation differs",
                ));
            }
            let next_generation = self.generation.checked_add(1).ok_or_else(|| {
                StateTransactionError::new(
                    StateTransactionErrorStage::Commit,
                    StateTransactionErrorKind::Internal,
                    "committed generation overflow",
                )
            })?;
            self.committed = payload;
            self.generation = next_generation;
            Ok(self.committed.len())
        }

        fn abort(
            &mut self,
            delta: PreparedStateDelta<Self::DeltaPayload>,
        ) -> Result<(), StateTransactionError> {
            let (owner_epoch, nonce, bases, _payload) = delta.into_parts();
            if owner_epoch != self.owner_epoch {
                return Err(StateTransactionError::new(
                    StateTransactionErrorStage::Abort,
                    StateTransactionErrorKind::OwnerMismatch,
                    "prepared delta owner epoch is stale",
                ));
            }
            if nonce != self.current_nonce {
                return Err(StateTransactionError::new(
                    StateTransactionErrorStage::Abort,
                    StateTransactionErrorKind::StaleNonce,
                    "prepared delta nonce is stale",
                ));
            }
            if bases.len() != 1 {
                return Err(invalid_at(
                    StateTransactionErrorStage::Abort,
                    "reference owner requires exactly one state base",
                ));
            }
            let base = &bases[0];
            if base.key() != &self.current_key {
                return Err(StateTransactionError::new(
                    StateTransactionErrorStage::Abort,
                    StateTransactionErrorKind::LeaseMismatch,
                    "prepared delta state lease is not current",
                ));
            }
            if base.committed_generation() != self.generation {
                return Err(StateTransactionError::new(
                    StateTransactionErrorStage::Abort,
                    StateTransactionErrorKind::StaleGeneration,
                    "current committed generation differs",
                ));
            }
            Ok(())
        }
    }

    fn fake_owner() -> FakeOwner {
        FakeOwner {
            owner_epoch: epoch(1),
            current_nonce: nz(7),
            current_key: key(11, "state", 9, 1),
            generation: 1,
            committed: vec![1],
        }
    }

    fn fake_delta(
        owner_epoch: StateOwnerEpoch,
        nonce: u64,
        request_id: u64,
        state_name: &str,
        handle: u64,
        lease: u64,
        generation: u64,
    ) -> PreparedStateDelta<Vec<u8>> {
        PreparedStateDelta::new(
            owner_epoch,
            nz(nonce),
            vec![base(request_id, state_name, handle, lease, generation)],
            vec![9, 9],
        )
        .expect("fake delta metadata")
    }

    #[test]
    fn fake_owner_commit_consumes_delta_and_stale_generation_preserves_state() {
        let mut owner = fake_owner();
        let snapshots = owner.begin(&batch_fixture()).expect("begin");
        let valid = PreparedStateDelta::new(
            owner.owner_epoch(),
            snapshots.batch_nonce(),
            snapshots
                .entries()
                .iter()
                .map(|entry| entry.base().clone())
                .collect(),
            vec![2, 3],
        )
        .expect("valid delta");
        valid
            .validate_against_snapshot(&snapshots)
            .expect("valid delta matches");
        assert_eq!(owner.commit(valid).expect("commit"), 2);
        assert_eq!(owner.committed, vec![2, 3]);

        let before = owner.committed.clone();
        let stale = PreparedStateDelta::new(
            owner.owner_epoch(),
            nz(7),
            vec![base(11, "state", 9, 1, 1)],
            vec![9, 9],
        )
        .expect("stale delta metadata");
        assert_eq!(
            owner.commit(stale).unwrap_err().kind(),
            StateTransactionErrorKind::StaleGeneration
        );
        assert_eq!(owner.committed, before);

        let abort_delta = PreparedStateDelta::new(
            owner.owner_epoch(),
            nz(7),
            vec![base(11, "state", 9, 1, 2)],
            vec![8],
        )
        .expect("abort delta metadata");
        owner.abort(abort_delta).expect("abort");
        assert_eq!(owner.committed, before);
    }

    #[test]
    fn fake_owner_rejects_stale_owner_nonce_handle_lease_and_generation_without_mutation() {
        let cases = [
            (
                fake_delta(epoch(2), 7, 11, "state", 9, 1, 1),
                StateTransactionErrorKind::OwnerMismatch,
            ),
            (
                fake_delta(epoch(1), 8, 11, "state", 9, 1, 1),
                StateTransactionErrorKind::StaleNonce,
            ),
            (
                fake_delta(epoch(1), 7, 11, "state", 10, 1, 1),
                StateTransactionErrorKind::LeaseMismatch,
            ),
            (
                fake_delta(epoch(1), 7, 11, "state", 9, 2, 1),
                StateTransactionErrorKind::LeaseMismatch,
            ),
            (
                fake_delta(epoch(1), 7, 11, "state", 9, 1, 0),
                StateTransactionErrorKind::StaleGeneration,
            ),
        ];
        for (delta, expected_kind) in cases {
            let mut owner = fake_owner();
            let before_payload = owner.committed.clone();
            let before_generation = owner.generation;
            assert_eq!(owner.commit(delta).unwrap_err().kind(), expected_kind);
            assert_eq!(owner.committed, before_payload);
            assert_eq!(owner.generation, before_generation);
        }

        let mut overflow_owner = fake_owner();
        overflow_owner.generation = u64::MAX;
        let before_payload = overflow_owner.committed.clone();
        assert_eq!(
            overflow_owner
                .commit(fake_delta(epoch(1), 7, 11, "state", 9, 1, u64::MAX))
                .unwrap_err()
                .kind(),
            StateTransactionErrorKind::Internal
        );
        assert_eq!(overflow_owner.committed, before_payload);
        assert_eq!(overflow_owner.generation, u64::MAX);
    }

    #[test]
    fn fake_owner_begin_rejects_non_singleton_fixture_without_panicking() {
        let mut owner = fake_owner();
        let mut batch = batch_fixture();
        batch.items.clear();
        assert!(owner.begin(&batch).is_err());
    }

    /// Models an external backend crate: its payload is deliberately not
    /// `Clone`, yet an owner implementation can consume it through the public
    /// protocol boundary.
    mod external_style_owner {
        use super::*;

        struct NonClonePayload(Vec<u8>);

        struct ExternalOwner {
            epoch: StateOwnerEpoch,
        }

        impl StateTransactionOwner for ExternalOwner {
            type SnapshotPayload = NonClonePayload;
            type DeltaPayload = NonClonePayload;
            type CommitReceipt = usize;

            fn owner_epoch(&self) -> StateOwnerEpoch {
                self.epoch
            }

            fn begin(
                &mut self,
                _batch: &ExecutionBatch,
            ) -> Result<StateSnapshotSet<Self::SnapshotPayload>, StateTransactionError>
            {
                Err(invalid_at(
                    StateTransactionErrorStage::Begin,
                    "external-style compile fixture does not begin",
                ))
            }

            fn commit(
                &mut self,
                delta: PreparedStateDelta<Self::DeltaPayload>,
            ) -> Result<Self::CommitReceipt, StateTransactionError> {
                let (_, _, _, payload) = delta.into_parts();
                Ok(payload.0.len())
            }

            fn abort(
                &mut self,
                delta: PreparedStateDelta<Self::DeltaPayload>,
            ) -> Result<(), StateTransactionError> {
                let (_, _, _, payload) = delta.into_parts();
                drop(payload);
                Ok(())
            }
        }

        #[test]
        fn public_owner_can_consume_non_clone_delta_payload() {
            let mut owner = ExternalOwner { epoch: epoch(9) };
            let delta = PreparedStateDelta::new(
                owner.owner_epoch(),
                nz(3),
                Vec::new(),
                NonClonePayload(vec![1, 2, 3]),
            )
            .expect("external-style delta metadata");
            assert_eq!(owner.commit(delta).expect("owner consumes delta"), 3);

            let abort_delta = PreparedStateDelta::new(
                owner.owner_epoch(),
                nz(3),
                Vec::new(),
                NonClonePayload(vec![4]),
            )
            .expect("external-style abort metadata");
            owner
                .abort(abort_delta)
                .expect("owner consumes abort delta");
        }
    }
}
