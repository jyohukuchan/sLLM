// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Prepared, rectangular execution work for the generic model executor.
//!
//! An [`ExecutionBatch`] describes metadata only. It never allocates a token
//! buffer proportional to a declared token count and it does not advance
//! scheduler-visible request or state progress. The owner validates the batch,
//! executes it, and commits it only while its [`ExecutionBatch::commit_nonce`]
//! remains current.

use std::collections::{HashMap, HashSet};

use crate::model_graph::StateId;

/// The largest number of requests in one prepared execution batch.
pub const MAX_EXECUTION_BATCH_ITEMS: usize = 4_096;

/// The largest number of logical state bindings on one request item.
pub const MAX_STATE_BINDINGS_PER_ITEM: usize = 1_024;

/// The largest block-table length accepted on one request item.
pub const MAX_BLOCK_TABLE_ENTRIES_PER_ITEM: usize = 4_096;

/// The largest total state-binding count tracked during one batch validation.
pub const MAX_EXECUTION_BATCH_STATE_BINDINGS: usize = 65_536;

/// The largest common rectangular chunk width accepted before backend planning.
pub const MAX_COMMON_CHUNK_WIDTH: u64 = 1_024;

/// The largest packed input or output span accepted before backend planning.
pub const MAX_PACKED_TOKENS: u64 = 65_536;

/// A phase selected by the batch planner rather than by a model-specific loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPhase {
    /// Process a prompt that has no request-owned cached prefix.
    ColdPrefill,
    /// Process new prompt tokens after a committed request-owned prefix.
    CachedPrefixPrefill,
    /// Process one decode input token per request item.
    Decode,
}

/// A checked half-open range in a packed token or value buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenRange {
    /// The first element in the range.
    pub start: u64,
    /// The number of elements in the range. It is always positive when valid.
    pub len: u64,
}

impl TokenRange {
    /// Creates a range without allocating storage for the declared elements.
    pub const fn new(start: u64, len: u64) -> Self {
        Self { start, len }
    }

    /// Returns the exclusive end after rejecting empty and overflowing ranges.
    pub fn end_checked(self) -> Result<u64, String> {
        if self.len == 0 {
            return Err("token range length must be nonzero".to_string());
        }
        self.start
            .checked_add(self.len)
            .ok_or_else(|| "token range end overflows u64".to_string())
    }
}

/// An opaque Rust-owned reference to backend-owned request state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateHandle(u64);

impl StateHandle {
    /// Constructs a nonzero opaque state handle.
    pub fn new(value: u64) -> Result<Self, String> {
        if value == 0 {
            return Err("state handle must be nonzero".to_string());
        }
        Ok(Self(value))
    }

    /// Returns the stable numeric handle for backend/state ownership checks.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Binds one logical graph-state entry to one request-owned opaque handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchStateBinding {
    /// The backend-independent state declared by the model graph.
    pub state_id: StateId,
    /// The request-owned opaque handle that identifies the physical payload.
    pub handle: StateHandle,
    /// Whether this state requires the item's paged-KV block table.
    pub uses_paged_kv: bool,
}

/// One request's rectangular contribution to an [`ExecutionBatch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionBatchItem {
    /// The scheduler request identity. All u64 values are valid identities.
    pub request_id: u64,
    /// The item's token range in the shared packed input buffer.
    pub packed: TokenRange,
    /// The number of tokens already committed for this request.
    pub prefix_len: u64,
    /// The absolute position of the first token in this execution unit.
    pub absolute_start_position: u64,
    /// The request-local absolute input-token range consumed by the graph.
    pub source: TokenRange,
    /// The shared packed-output range produced by the graph execution.
    pub destination: TokenRange,
    /// Request-owned graph-state bindings used by this item.
    pub state_bindings: Vec<BatchStateBinding>,
    /// Paged-KV physical block IDs for this request when required by a binding.
    pub block_table: Vec<u32>,
}

/// A bounded workspace admission plan for one execution batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspacePlan {
    /// Total capacity available to this batch on the selected backend device.
    pub capacity_bytes: u64,
    /// Weights and other resident buffers already required by the execution.
    pub resident_bytes: u64,
    /// Request-owned persistent state, including KV and recurrent state.
    pub persistent_state_bytes: u64,
    /// Batch activation/intermediate storage.
    pub temporary_activation_bytes: u64,
    /// Temporary storage required by selected operator implementations.
    pub operator_workspace_bytes: u64,
    /// Capacity that must remain available after the planned allocation.
    pub required_headroom_bytes: u64,
}

impl WorkspacePlan {
    /// Returns the checked sum of the buffers required by this batch.
    pub fn planned_total_bytes(&self) -> Result<u64, String> {
        self.resident_bytes
            .checked_add(self.persistent_state_bytes)
            .and_then(|value| value.checked_add(self.temporary_activation_bytes))
            .and_then(|value| value.checked_add(self.operator_workspace_bytes))
            .ok_or_else(|| "workspace planned total overflows u64".to_string())
    }

    /// Returns the checked capacity required by the plan plus its safety headroom.
    pub fn required_capacity_bytes(&self) -> Result<u64, String> {
        self.planned_total_bytes()?
            .checked_add(self.required_headroom_bytes)
            .ok_or_else(|| "workspace required capacity overflows u64".to_string())
    }

    /// Returns headroom left after planned buffers are reserved.
    pub fn planned_headroom_bytes(&self) -> Result<u64, String> {
        self.capacity_bytes
            .checked_sub(self.planned_total_bytes()?)
            .ok_or_else(|| "workspace planned total exceeds capacity".to_string())
    }

    /// Rejects overflow, capacity exhaustion, and insufficient safety headroom.
    pub fn validate(&self) -> Result<(), String> {
        let required_capacity = self.required_capacity_bytes()?;
        if required_capacity > self.capacity_bytes {
            return Err(format!(
                "workspace capacity is insufficient: required={required_capacity} capacity={}",
                self.capacity_bytes
            ));
        }
        let headroom = self.planned_headroom_bytes()?;
        if headroom < self.required_headroom_bytes {
            return Err(format!(
                "workspace planned headroom is insufficient: available={headroom} required={}",
                self.required_headroom_bytes
            ));
        }
        Ok(())
    }
}

/// One validated, compatible unit of graph work prepared by the batch planner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionBatch {
    /// The requested cold-prefill, cached-prefix-prefill, or decode phase.
    pub phase: ExecutionPhase,
    /// Lowercase SHA-256 over graph/state/format/backend compatibility inputs.
    pub compatibility_key_sha256: String,
    /// Nonzero scheduler/state version validated before execution starts.
    pub commit_nonce: u64,
    /// Required shared execution width for every request item.
    pub common_chunk_width: u64,
    /// The number of elements available in the packed input buffer.
    pub packed_token_count: u64,
    /// Request contributions. Validation bounds and scans this vector only.
    pub items: Vec<ExecutionBatchItem>,
    /// The workspace reservation validated before backend allocation.
    pub workspace: WorkspacePlan,
}

impl ExecutionBatch {
    /// Validates this metadata before backend allocation or execution.
    pub fn validate(&self) -> Result<(), String> {
        validate_lowercase_sha256(&self.compatibility_key_sha256)?;
        if self.commit_nonce == 0 {
            return Err("execution batch commit nonce must be nonzero and current".to_string());
        }
        if self.common_chunk_width == 0 {
            return Err("execution batch common chunk width must be nonzero".to_string());
        }
        if self.common_chunk_width > MAX_COMMON_CHUNK_WIDTH {
            return Err(format!(
                "execution batch common chunk width {} exceeds maximum {MAX_COMMON_CHUNK_WIDTH}",
                self.common_chunk_width
            ));
        }
        if self.packed_token_count == 0 {
            return Err("execution batch packed token count must be nonzero".to_string());
        }
        if self.packed_token_count > MAX_PACKED_TOKENS {
            return Err(format!(
                "execution batch packed token count {} exceeds maximum {MAX_PACKED_TOKENS}",
                self.packed_token_count
            ));
        }
        if self.items.is_empty() {
            return Err("execution batch must contain at least one item".to_string());
        }
        if self.items.len() > MAX_EXECUTION_BATCH_ITEMS {
            return Err(format!(
                "execution batch item count {} exceeds maximum {MAX_EXECUTION_BATCH_ITEMS}",
                self.items.len()
            ));
        }

        let item_count = u64::try_from(self.items.len())
            .map_err(|_| "execution batch item count does not fit u64".to_string())?;
        let expected_packed_token_count = item_count
            .checked_mul(self.common_chunk_width)
            .ok_or_else(|| {
                "execution batch rectangular packed token count overflows u64".to_string()
            })?;
        if self.packed_token_count != expected_packed_token_count {
            return Err(format!(
                "execution batch packed token count {} does not equal rectangular item_count * common_chunk_width {}",
                self.packed_token_count, expected_packed_token_count
            ));
        }
        let total_state_bindings = self.items.iter().try_fold(0_usize, |total, item| {
            total.checked_add(item.state_bindings.len()).ok_or_else(|| {
                "execution batch total state binding count overflows usize".to_string()
            })
        })?;
        if total_state_bindings > MAX_EXECUTION_BATCH_STATE_BINDINGS {
            return Err(format!(
                "execution batch total state binding count {total_state_bindings} exceeds maximum {MAX_EXECUTION_BATCH_STATE_BINDINGS}"
            ));
        }

        self.workspace.validate()?;
        self.validate_phase_width()?;

        let mut request_ids = HashSet::with_capacity(self.items.len());
        let mut handles = HashMap::new();
        let mut expected_packed_start = 0_u64;
        for item in &self.items {
            self.validate_item(item, &mut request_ids, &mut handles, expected_packed_start)?;
            expected_packed_start = expected_packed_start
                .checked_add(self.common_chunk_width)
                .ok_or_else(|| {
                    "execution batch rectangular packed offset overflows u64".to_string()
                })?;
        }
        Ok(())
    }

    fn validate_phase_width(&self) -> Result<(), String> {
        match self.phase {
            ExecutionPhase::ColdPrefill | ExecutionPhase::CachedPrefixPrefill => Ok(()),
            ExecutionPhase::Decode if self.common_chunk_width == 1 => Ok(()),
            ExecutionPhase::Decode => Err(format!(
                "decode execution batch common chunk width must be 1, got {}",
                self.common_chunk_width
            )),
        }
    }

    fn validate_item(
        &self,
        item: &ExecutionBatchItem,
        request_ids: &mut HashSet<u64>,
        handles: &mut HashMap<u64, (u64, String)>,
        expected_packed_start: u64,
    ) -> Result<(), String> {
        if !request_ids.insert(item.request_id) {
            return Err(format!(
                "execution batch contains duplicate request ID {}",
                item.request_id
            ));
        }
        if item.state_bindings.len() > MAX_STATE_BINDINGS_PER_ITEM {
            return Err(format!(
                "request {} state binding count {} exceeds maximum {MAX_STATE_BINDINGS_PER_ITEM}",
                item.request_id,
                item.state_bindings.len()
            ));
        }
        if item.block_table.len() > MAX_BLOCK_TABLE_ENTRIES_PER_ITEM {
            return Err(format!(
                "request {} block table length {} exceeds maximum {MAX_BLOCK_TABLE_ENTRIES_PER_ITEM}",
                item.request_id,
                item.block_table.len()
            ));
        }

        let packed_end = item.packed.end_checked()?;
        item.source.end_checked()?;
        item.destination.end_checked()?;
        if packed_end > self.packed_token_count {
            return Err(format!(
                "request {} packed range end {packed_end} exceeds packed token count {}",
                item.request_id, self.packed_token_count
            ));
        }
        if item.packed.start != expected_packed_start {
            return Err(format!(
                "request {} packed range starts at {} but rectangular packing requires {}",
                item.request_id, item.packed.start, expected_packed_start
            ));
        }
        for (label, range) in [
            ("packed", item.packed),
            ("source", item.source),
            ("destination", item.destination),
        ] {
            if range.len != self.common_chunk_width {
                return Err(format!(
                    "request {} {label} range width {} differs from common chunk width {}",
                    item.request_id, range.len, self.common_chunk_width
                ));
            }
        }

        match self.phase {
            ExecutionPhase::ColdPrefill if item.prefix_len != 0 => {
                return Err(format!(
                    "cold-prefill request {} must have prefix length zero",
                    item.request_id
                ));
            }
            ExecutionPhase::CachedPrefixPrefill if item.prefix_len == 0 => {
                return Err(format!(
                    "cached-prefix-prefill request {} must have a nonzero prefix length",
                    item.request_id
                ));
            }
            _ => {}
        }
        item.prefix_len
            .checked_add(self.common_chunk_width)
            .ok_or_else(|| {
                format!(
                    "request {} context arithmetic overflows at prefix/chunk boundary",
                    item.request_id
                )
            })?;
        item.absolute_start_position
            .checked_add(item.source.len)
            .ok_or_else(|| {
                format!(
                    "request {} source position arithmetic overflows",
                    item.request_id
                )
            })?;
        item.absolute_start_position
            .checked_add(item.destination.len)
            .ok_or_else(|| {
                format!(
                    "request {} destination position arithmetic overflows",
                    item.request_id
                )
            })?;

        if item.source.start != item.absolute_start_position {
            return Err(format!(
                "request {} source range start {} does not match absolute start position {}",
                item.request_id, item.source.start, item.absolute_start_position
            ));
        }
        if item.destination != item.packed {
            return Err(format!(
                "request {} destination range must equal its packed output range",
                item.request_id
            ));
        }

        self.validate_state_bindings(item, handles)?;
        Ok(())
    }

    fn validate_state_bindings(
        &self,
        item: &ExecutionBatchItem,
        batch_handles: &mut HashMap<u64, (u64, String)>,
    ) -> Result<(), String> {
        let mut state_ids = HashSet::with_capacity(item.state_bindings.len());
        let mut item_handles = HashSet::with_capacity(item.state_bindings.len());
        let mut needs_paged_kv = false;
        for binding in &item.state_bindings {
            binding.state_id.validate()?;
            let state_key = binding.state_id.as_str();
            if !state_ids.insert(state_key) {
                return Err(format!(
                    "request {} contains duplicate state ID {state_key}",
                    item.request_id
                ));
            }
            if binding.handle.get() == 0 {
                return Err(format!(
                    "request {} contains a zero state handle",
                    item.request_id
                ));
            }
            if !item_handles.insert(binding.handle.get()) {
                return Err(format!(
                    "request {} contains duplicate state handle {}",
                    item.request_id,
                    binding.handle.get()
                ));
            }
            if let Some((other_request_id, other_state_id)) = batch_handles.insert(
                binding.handle.get(),
                (item.request_id, state_key.to_string()),
            ) {
                return Err(format!(
                    "state handle {} is shared by request {} state {} and request {} state {}; shared handles are not allowed in the initial contract",
                    binding.handle.get(),
                    other_request_id,
                    other_state_id,
                    item.request_id,
                    state_key
                ));
            }
            needs_paged_kv |= binding.uses_paged_kv;
        }
        if needs_paged_kv && item.block_table.is_empty() {
            return Err(format!(
                "request {} uses paged KV state without a block table",
                item.request_id
            ));
        }
        let mut block_ids = HashSet::with_capacity(item.block_table.len());
        for block_id in &item.block_table {
            if !block_ids.insert(*block_id) {
                return Err(format!(
                    "request {} block table contains duplicate block ID {block_id}",
                    item.request_id
                ));
            }
        }
        Ok(())
    }
}

fn validate_lowercase_sha256(value: &str) -> Result<(), String> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err("execution batch compatibility key must be a lowercase SHA-256".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_id(value: &str) -> StateId {
        StateId::new(value).expect("test state ID must be valid")
    }

    fn state_binding(state_name: &str, handle: u64, uses_paged_kv: bool) -> BatchStateBinding {
        BatchStateBinding {
            state_id: state_id(state_name),
            handle: StateHandle::new(handle).expect("test state handle must be nonzero"),
            uses_paged_kv,
        }
    }

    fn workspace() -> WorkspacePlan {
        WorkspacePlan {
            capacity_bytes: 1_000,
            resident_bytes: 400,
            persistent_state_bytes: 200,
            temporary_activation_bytes: 100,
            operator_workspace_bytes: 100,
            required_headroom_bytes: 100,
        }
    }

    fn item(
        request_id: u64,
        packed_start: u64,
        width: u64,
        prefix_len: u64,
        state_bindings: Vec<BatchStateBinding>,
    ) -> ExecutionBatchItem {
        ExecutionBatchItem {
            request_id,
            packed: TokenRange::new(packed_start, width),
            prefix_len,
            absolute_start_position: prefix_len,
            source: TokenRange::new(prefix_len, width),
            destination: TokenRange::new(packed_start, width),
            state_bindings,
            block_table: vec![0, 1],
        }
    }

    fn batch(phase: ExecutionPhase, width: u64, prefix_len: u64) -> ExecutionBatch {
        ExecutionBatch {
            phase,
            compatibility_key_sha256: "a".repeat(64),
            commit_nonce: 1,
            common_chunk_width: width,
            packed_token_count: width,
            items: vec![item(
                1,
                0,
                width,
                prefix_len,
                vec![state_binding("layer-0-kv", 1, true)],
            )],
            workspace: workspace(),
        }
    }

    #[test]
    fn valid_cold_prefill_m16_validates() {
        batch(ExecutionPhase::ColdPrefill, 16, 0)
            .validate()
            .expect("valid cold M16 batch must validate");
    }

    #[test]
    fn valid_cached_prefix_prefill_m32_validates() {
        batch(ExecutionPhase::CachedPrefixPrefill, 32, 128)
            .validate()
            .expect("valid cached-prefix M32 batch must validate");
    }

    #[test]
    fn valid_decode_m1_validates() {
        batch(ExecutionPhase::Decode, 1, 32)
            .validate()
            .expect("valid decode M1 batch must validate");
    }

    #[test]
    fn zero_commit_nonce_is_rejected_as_stale() {
        let mut value = batch(ExecutionPhase::ColdPrefill, 16, 0);
        value.commit_nonce = 0;
        assert!(
            value
                .validate()
                .expect_err("zero nonce must be rejected")
                .contains("nonce")
        );
    }

    #[test]
    fn nonrectangular_packed_ranges_are_rejected() {
        let mut value = batch(ExecutionPhase::ColdPrefill, 16, 0);
        value.items.push(item(
            2,
            8,
            16,
            0,
            vec![state_binding("layer-0-kv", 1, true)],
        ));
        value.packed_token_count = 32;
        assert!(
            value
                .validate()
                .expect_err("nonrectangular ranges must be rejected")
                .contains("rectangular packing")
        );
    }

    #[test]
    fn wrong_width_and_phase_prefix_are_rejected() {
        let mut wrong_width = batch(ExecutionPhase::ColdPrefill, 16, 0);
        wrong_width.items[0].source.len = 8;
        assert!(
            wrong_width
                .validate()
                .expect_err("mixed widths must be rejected")
                .contains("width")
        );

        let cold_prefix = batch(ExecutionPhase::ColdPrefill, 16, 1);
        assert!(
            cold_prefix
                .validate()
                .expect_err("cold prefill must not have a prefix")
                .contains("prefix")
        );

        let cached_no_prefix = batch(ExecutionPhase::CachedPrefixPrefill, 16, 0);
        assert!(
            cached_no_prefix
                .validate()
                .expect_err("cached prefill must have a prefix")
                .contains("prefix")
        );

        let decode_width = batch(ExecutionPhase::Decode, 2, 32);
        assert!(
            decode_width
                .validate()
                .expect_err("decode must use width one")
                .contains("must be 1")
        );
    }

    #[test]
    fn duplicate_request_state_and_handle_are_rejected() {
        let mut duplicate_request = batch(ExecutionPhase::ColdPrefill, 16, 0);
        duplicate_request.items.push(item(
            1,
            16,
            16,
            0,
            vec![state_binding("layer-1-kv", 2, true)],
        ));
        duplicate_request.packed_token_count = 32;
        assert!(
            duplicate_request
                .validate()
                .expect_err("duplicate requests must be rejected")
                .contains("duplicate request")
        );

        let mut duplicate_state = batch(ExecutionPhase::ColdPrefill, 16, 0);
        duplicate_state.items[0]
            .state_bindings
            .push(state_binding("layer-0-kv", 2, false));
        assert!(
            duplicate_state
                .validate()
                .expect_err("duplicate state IDs must be rejected")
                .contains("duplicate state ID")
        );

        let mut duplicate_handle = batch(ExecutionPhase::ColdPrefill, 16, 0);
        duplicate_handle.items[0]
            .state_bindings
            .push(state_binding("layer-1-position", 1, false));
        assert!(
            duplicate_handle
                .validate()
                .expect_err("duplicate state handles must be rejected")
                .contains("duplicate state handle")
        );
    }

    #[test]
    fn state_handle_shared_by_two_requests_is_rejected() {
        let mut value = batch(ExecutionPhase::ColdPrefill, 16, 0);
        value.items.push(item(
            2,
            16,
            16,
            0,
            vec![state_binding("layer-0-kv", 1, true)],
        ));
        value.packed_token_count = 32;
        let error = value
            .validate()
            .expect_err("request-owned handles must not be shared");
        assert!(error.contains("state handle 1 is shared by request 1 state layer-0-kv"));
        assert!(error.contains("request 2 state layer-0-kv"));
    }

    #[test]
    fn packed_metadata_must_be_bounded_and_exactly_rectangular() {
        let mut mismatched_count = batch(ExecutionPhase::ColdPrefill, 16, 0);
        mismatched_count.packed_token_count = 32;
        assert!(
            mismatched_count
                .validate()
                .expect_err("packed count must exactly cover rectangular items")
                .contains("item_count * common_chunk_width")
        );

        let mut oversized_count = batch(ExecutionPhase::ColdPrefill, 16, 0);
        oversized_count.packed_token_count = MAX_PACKED_TOKENS + 1;
        assert!(
            oversized_count
                .validate()
                .expect_err("oversized packed metadata must be rejected")
                .contains("exceeds maximum")
        );

        let oversized_width = MAX_COMMON_CHUNK_WIDTH + 1;
        let oversized_width_batch = batch(ExecutionPhase::ColdPrefill, oversized_width, 0);
        assert!(
            oversized_width_batch
                .validate()
                .expect_err("oversized common width must be rejected")
                .contains("common chunk width")
        );
    }

    #[test]
    fn source_destination_and_absolute_position_contract_is_enforced() {
        let mut offset = batch(ExecutionPhase::CachedPrefixPrefill, 32, 128);
        offset.items[0].absolute_start_position = 1_024;
        assert!(
            offset
                .validate()
                .expect_err("source must follow absolute position")
                .contains("source range start")
        );
        offset.items[0].source.start = 1_024;
        offset
            .validate()
            .expect("absolute position offset must be independent from retained prefix length");

        let mut destination = batch(ExecutionPhase::ColdPrefill, 16, 0);
        destination.items[0].destination.start = 1;
        assert!(
            destination
                .validate()
                .expect_err("destination must identify the packed output span")
                .contains("destination range")
        );
    }

    #[test]
    fn zero_request_id_remains_a_valid_scheduler_identity() {
        let mut value = batch(ExecutionPhase::ColdPrefill, 16, 0);
        value.items[0].request_id = 0;
        value
            .validate()
            .expect("generic batch request ID zero must remain valid");
    }

    #[test]
    fn bad_compatibility_sha256_is_rejected() {
        let mut value = batch(ExecutionPhase::ColdPrefill, 16, 0);
        value.compatibility_key_sha256 = "A".repeat(64);
        assert!(
            value
                .validate()
                .expect_err("uppercase SHA-256 must be rejected")
                .contains("SHA-256")
        );
    }

    #[test]
    fn workspace_overflow_and_insufficient_capacity_are_rejected() {
        let overflow = WorkspacePlan {
            capacity_bytes: u64::MAX,
            resident_bytes: u64::MAX,
            persistent_state_bytes: 1,
            temporary_activation_bytes: 0,
            operator_workspace_bytes: 0,
            required_headroom_bytes: 0,
        };
        assert!(
            overflow
                .validate()
                .expect_err("overflowing workspace must be rejected")
                .contains("overflows")
        );

        let insufficient = WorkspacePlan {
            capacity_bytes: 100,
            resident_bytes: 90,
            persistent_state_bytes: 0,
            temporary_activation_bytes: 5,
            operator_workspace_bytes: 0,
            required_headroom_bytes: 10,
        };
        assert!(
            insufficient
                .validate()
                .expect_err("insufficient workspace must be rejected")
                .contains("insufficient")
        );
    }

    #[test]
    fn state_handle_constructor_rejects_zero() {
        assert!(StateHandle::new(0).is_err());
    }
}
