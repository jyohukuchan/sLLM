// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Backend-independent logical state declarations for a model graph.
//!
//! A [`StateSchema`] describes state ownership, layout, numerical format, and
//! transaction semantics. It deliberately contains no runtime handle, device
//! pointer, allocation, or model-name-specific branch.

use std::collections::{BTreeMap, BTreeSet};

use crate::model_graph::{GraphNode, GraphNodeKind, ModelGraph, NumericalFormat, StateId};

/// Maximum number of logical state entries in one schema.
pub const MAX_STATE_SCHEMA_ENTRIES: usize = 65_536;

/// A validated extension identifier for a custom state kind.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CustomStateKindId(String);

impl CustomStateKindId {
    /// Creates a custom-state kind identifier accepted by schema validation.
    pub fn new(value: impl Into<String>) -> Result<Self, String> {
        let value = value.into();
        validate_safe_id(&value, "custom state kind")?;
        Ok(Self(value))
    }

    /// Returns the stable identifier text.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn validate(&self) -> Result<(), String> {
        validate_safe_id(&self.0, "custom state kind")
    }
}

/// Semantic category of logical execution state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateKind {
    /// Block-addressable key/value attention cache.
    PagedKv,
    /// Bounded key/value cache that retains a sliding context window.
    SlidingWindowKv,
    /// Recurrent matrix or vector state.
    Recurrent,
    /// Causal convolution history.
    ConvolutionHistory,
    /// Per-request absolute position and cache-length counters.
    PositionCacheLength,
    /// An explicitly named extension with a matching custom layout.
    Custom(CustomStateKindId),
}

/// Logical owner and lifetime scope for a state entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateOwnership {
    /// One instance exists for each request.
    Request,
    /// One instance exists for each request and decoder layer.
    RequestLayer {
        /// Index in the backend-independent ordered model graph.
        layer_index: usize,
    },
    /// Immutable state shared by the loaded model.
    ModelSharedReadOnly,
}

/// Typed logical state layout. No variant owns physical storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateLayout {
    /// Paged K/V cache with separate K and V dimensions.
    PagedKv {
        /// Tokens in each physical cache block.
        block_size: usize,
        /// Number of physical cache blocks allocated for one request/layer.
        cache_blocks: usize,
        /// Number of query heads.
        q_heads: usize,
        /// Number of key/value heads.
        kv_heads: usize,
        /// Elements in each key head.
        head_dim: usize,
        /// Elements in each value head.
        value_dim: usize,
    },
    /// Bounded paged K/V cache with a logical sliding-window limit.
    SlidingWindowKv {
        /// Logical context retained by this state.
        window_tokens: usize,
        /// Tokens in each physical cache block.
        block_size: usize,
        /// Number of physical blocks allocated for the window.
        cache_blocks: usize,
        /// Number of query heads.
        q_heads: usize,
        /// Number of key/value heads.
        kv_heads: usize,
        /// Elements in each key head.
        head_dim: usize,
        /// Elements in each value head.
        value_dim: usize,
    },
    /// Dense recurrent state matrix.
    Recurrent {
        /// Matrix rows.
        rows: usize,
        /// Matrix columns.
        cols: usize,
    },
    /// Per-channel causal convolution history.
    ConvolutionHistory {
        /// Number of channels.
        channels: usize,
        /// Stored history tokens per channel.
        history_tokens: usize,
    },
    /// Two integer counters: absolute position and cache length.
    PositionCacheLength,
    /// Explicit element count for a custom state kind.
    Custom {
        /// Logical elements represented by the custom state.
        logical_elements: usize,
    },
}

/// How a transactional state is initialized before its first execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateInitialization {
    /// Initialize every logical element to zero.
    Zeroed,
    /// Initialize from validated request-owned input.
    FromRequest,
    /// Initialize by restoring a validated committed snapshot.
    FromSnapshot,
}

/// Required visibility boundary for mutation of a transactional state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateExecutionProtocol {
    /// Prepare backend state, execute, then atomically commit a matching nonce.
    PrepareExecuteCommit,
}

/// Required post-request handling for a transactional state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateResetProtocol {
    /// The state must return to its reusable baseline before release.
    Required,
    /// The state is discarded with its request and has no reusable baseline.
    DiscardWithRequest,
}

/// Snapshot and restore capability required by a transaction contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotRestorePolicy {
    /// A failed execution must be recoverable from a snapshot.
    Required,
    /// Snapshot/restore may be used by a planner but is not required.
    Optional,
    /// The planner must discard rather than restore a failed prepared state.
    Unsupported,
}

/// State mutation semantics exposed to the planner and executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateTransactionContract {
    /// Immutable shared state; it has no request transaction.
    ReadOnly,
    /// Request-owned state with explicit initialization, commit, and reset semantics.
    Transactional {
        /// Initialization contract before first use.
        initialization: StateInitialization,
        /// Boundary that makes prepared mutations visible.
        execution: StateExecutionProtocol,
        /// Cleanup contract after terminal release.
        reset: StateResetProtocol,
        /// Snapshot/restore support for failed prepared work.
        snapshot_restore: SnapshotRestorePolicy,
    },
}

/// One logical state entry used by graph nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateSpec {
    /// Stable logical identifier shared with the model graph.
    pub id: StateId,
    /// Semantic state kind.
    pub kind: StateKind,
    /// Request, request-layer, or immutable-model ownership.
    pub ownership: StateOwnership,
    /// Numerical representation of logical state values.
    pub format: NumericalFormat,
    /// Logical dimensions and cache layout.
    pub layout: StateLayout,
    /// Initialization, mutation, commit, and reset contract.
    pub transaction: StateTransactionContract,
}

impl StateSpec {
    /// Validates this entry before any backend state allocation.
    pub fn validate(&self) -> Result<(), String> {
        self.id.validate()?;
        self.format.validate()?;
        validate_kind_layout(&self.kind, &self.layout)?;
        validate_layout(&self.layout)?;
        validate_kind_format(&self.kind, &self.format)?;
        validate_ownership_and_transaction(&self.kind, self.ownership, self.transaction)?;
        Ok(())
    }

    /// Returns the exact logical element count after validating the layout.
    pub fn logical_element_count(&self) -> Result<u64, String> {
        self.validate()?;
        layout_logical_element_count(&self.layout)
    }

    /// Returns known logical value bytes per request when the format is exact.
    ///
    /// Quantized and custom numerical formats intentionally return `None`: their
    /// physical storage can include packed payloads, scale tables, or other
    /// metadata, so inferring bytes from nominal bit width would be unsound.
    /// This excludes backend allocation metadata, alignment, block tables, and
    /// workspace; it MUST NOT be the sole input to OOM admission control.
    pub fn known_logical_value_bytes_per_request(&self) -> Result<Option<u64>, String> {
        self.validate()?;
        if self.ownership == StateOwnership::ModelSharedReadOnly {
            return Ok(None);
        }
        let Some(bytes_per_element) = exact_format_bytes(&self.format) else {
            return Ok(None);
        };
        self.logical_element_count()?
            .checked_mul(bytes_per_element)
            .map(Some)
            .ok_or_else(|| format!("state {} byte estimate overflows u64", self.id.as_str()))
    }

    /// Returns known logical value bytes per request when the format is exact.
    #[deprecated(
        note = "use known_logical_value_bytes_per_request; this excludes physical allocation metadata and workspace"
    )]
    pub fn estimated_bytes_per_request(&self) -> Result<Option<u64>, String> {
        self.known_logical_value_bytes_per_request()
    }
}

/// Immutable logical state schema for one model graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateSchema {
    /// Stable schema identifier.
    pub schema_id: String,
    /// Logical state entries referenced by graph nodes.
    pub entries: Vec<StateSpec>,
}

impl StateSchema {
    /// Creates and validates a schema.
    pub fn new(schema_id: impl Into<String>, entries: Vec<StateSpec>) -> Result<Self, String> {
        let schema = Self {
            schema_id: schema_id.into(),
            entries,
        };
        schema.validate()?;
        Ok(schema)
    }

    /// Validates identifiers, ownership, layout dimensions, and transaction semantics.
    pub fn validate(&self) -> Result<(), String> {
        validate_safe_id(&self.schema_id, "state schema")?;
        if self.entries.len() > MAX_STATE_SCHEMA_ENTRIES {
            return Err(format!(
                "state schema has {} entries, exceeding maximum {MAX_STATE_SCHEMA_ENTRIES}",
                self.entries.len()
            ));
        }
        let mut seen = BTreeSet::new();
        for entry in &self.entries {
            entry.validate()?;
            if !seen.insert(entry.id.as_str()) {
                return Err(format!(
                    "state schema has duplicate state ID {}",
                    entry.id.as_str()
                ));
            }
        }
        Ok(())
    }

    /// Validates this schema's request state references against one model graph.
    ///
    /// The graph does not yet expose an explicit graph-layer index, so this
    /// method intentionally does not infer or validate
    /// [`StateOwnership::RequestLayer`] `layer_index` values. A later adapter
    /// mapping contract must validate that relationship explicitly.
    ///
    /// [`StateKind::PositionCacheLength`] is request-owned graph-level execution
    /// state. It is consumed by [`crate::execution_batch::ExecutionBatch`]
    /// absolute-position and commit handling rather than by a graph node, so it
    /// is the one request-owned orphan explicitly permitted here. Immutable
    /// model-shared state is also permitted to be unreferenced.
    pub fn validate_against_graph(&self, graph: &ModelGraph) -> Result<(), String> {
        self.validate()?;
        graph.validate()?;

        let declared = self
            .entries
            .iter()
            .map(|entry| (entry.id.as_str(), entry))
            .collect::<BTreeMap<_, _>>();
        let mut referenced = BTreeSet::new();
        for node in &graph.nodes {
            let mut node_states = Vec::with_capacity(node.states.len());
            for state_id in &node.states {
                let entry = declared.get(state_id.as_str()).ok_or_else(|| {
                    format!(
                        "graph node {} references state {} absent from state schema {}",
                        node.id.as_str(),
                        state_id.as_str(),
                        self.schema_id
                    )
                })?;
                node_states.push(*entry);
                referenced.insert(state_id.as_str());
            }
            validate_node_state_contract(node, &node_states)?;
        }
        for entry in &self.entries {
            if matches!(
                entry.ownership,
                StateOwnership::Request | StateOwnership::RequestLayer { .. }
            ) && !matches!(entry.kind, StateKind::PositionCacheLength)
                && !referenced.contains(entry.id.as_str())
            {
                return Err(format!(
                    "request-owned state {} is not referenced by model graph {}",
                    entry.id.as_str(),
                    graph.graph_id
                ));
            }
        }
        Ok(())
    }

    /// Returns the sum of known logical per-request state value bytes, or `None` when
    /// any request-owned entry uses a format whose physical byte size is not exact.
    ///
    /// The result excludes backend allocation metadata, alignment, block tables,
    /// and workspace; it MUST NOT be the sole input to OOM admission control.
    pub fn known_logical_value_bytes_per_request(&self) -> Result<Option<u64>, String> {
        self.validate()?;
        let mut total = 0_u64;
        for entry in &self.entries {
            let Some(bytes) = entry.known_logical_value_bytes_per_request()? else {
                if entry.ownership != StateOwnership::ModelSharedReadOnly {
                    return Ok(None);
                }
                continue;
            };
            total = total.checked_add(bytes).ok_or_else(|| {
                format!(
                    "state schema {} per-request byte estimate overflows u64",
                    self.schema_id
                )
            })?;
        }
        Ok(Some(total))
    }

    /// Returns known logical value bytes per request when every format is exact.
    #[deprecated(
        note = "use known_logical_value_bytes_per_request; this excludes physical allocation metadata and workspace"
    )]
    pub fn estimated_bytes_per_request(&self) -> Result<Option<u64>, String> {
        self.known_logical_value_bytes_per_request()
    }
}

fn validate_node_state_contract(node: &GraphNode, states: &[&StateSpec]) -> Result<(), String> {
    match &node.kind {
        GraphNodeKind::DenseAttention { .. } => match states {
            [] => Ok(()),
            [state] => {
                if !matches!(state.kind, StateKind::PagedKv | StateKind::SlidingWindowKv) {
                    return Err(format!(
                        "dense-attention node {} state {} must be PagedKv or SlidingWindowKv",
                        node.id.as_str(),
                        state.id.as_str()
                    ));
                }
                require_request_layer_ownership(node, state, "dense-attention")
            }
            _ => Err(format!(
                "dense-attention node {} must use zero or one state",
                node.id.as_str()
            )),
        },
        GraphNodeKind::RecurrentAttention { .. } => {
            let Some((first, rest)) = states.split_first() else {
                return Err(format!(
                    "recurrent-attention node {} must have a recurrent state",
                    node.id.as_str()
                ));
            };
            if first.kind != StateKind::Recurrent {
                return Err(format!(
                    "recurrent-attention node {} first state {} must be Recurrent",
                    node.id.as_str(),
                    first.id.as_str()
                ));
            }
            require_request_layer_ownership(node, first, "recurrent-attention")?;
            if let Some(second) = rest.first() {
                if second.kind != StateKind::ConvolutionHistory {
                    return Err(format!(
                        "recurrent-attention node {} second state {} must be ConvolutionHistory",
                        node.id.as_str(),
                        second.id.as_str()
                    ));
                }
                require_request_layer_ownership(node, second, "recurrent-attention")?;
            }
            if rest.len() > 1 {
                return Err(format!(
                    "recurrent-attention node {} must use one or two states",
                    node.id.as_str()
                ));
            }
            Ok(())
        }
        _ if states.is_empty() => Ok(()),
        _ => Err(format!(
            "node {} kind does not permit state bindings",
            node.id.as_str()
        )),
    }
}

fn require_request_layer_ownership(
    node: &GraphNode,
    state: &StateSpec,
    node_kind: &str,
) -> Result<(), String> {
    if matches!(state.ownership, StateOwnership::RequestLayer { .. }) {
        Ok(())
    } else {
        Err(format!(
            "{node_kind} node {} state {} must use RequestLayer ownership",
            node.id.as_str(),
            state.id.as_str()
        ))
    }
}

fn validate_kind_layout(kind: &StateKind, layout: &StateLayout) -> Result<(), String> {
    match (kind, layout) {
        (StateKind::PagedKv, StateLayout::PagedKv { .. })
        | (StateKind::SlidingWindowKv, StateLayout::SlidingWindowKv { .. })
        | (StateKind::Recurrent, StateLayout::Recurrent { .. })
        | (StateKind::ConvolutionHistory, StateLayout::ConvolutionHistory { .. })
        | (StateKind::PositionCacheLength, StateLayout::PositionCacheLength) => Ok(()),
        (StateKind::Custom(id), StateLayout::Custom { .. }) => id.validate(),
        _ => Err(format!(
            "state kind {} is incompatible with layout {}",
            state_kind_name(kind),
            state_layout_name(layout)
        )),
    }
}

fn validate_layout(layout: &StateLayout) -> Result<(), String> {
    match layout {
        StateLayout::PagedKv {
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
        } => validate_kv_layout(
            *block_size,
            *cache_blocks,
            *q_heads,
            *kv_heads,
            *head_dim,
            *value_dim,
            None,
            "paged KV",
        ),
        StateLayout::SlidingWindowKv {
            window_tokens,
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
        } => validate_kv_layout(
            *block_size,
            *cache_blocks,
            *q_heads,
            *kv_heads,
            *head_dim,
            *value_dim,
            Some(*window_tokens),
            "sliding-window KV",
        ),
        StateLayout::Recurrent { rows, cols } => {
            checked_positive_product(*rows, *cols, "recurrent state")?;
            Ok(())
        }
        StateLayout::ConvolutionHistory {
            channels,
            history_tokens,
        } => {
            checked_positive_product(*channels, *history_tokens, "convolution history")?;
            Ok(())
        }
        StateLayout::PositionCacheLength => Ok(()),
        StateLayout::Custom { logical_elements } => {
            if *logical_elements == 0 {
                return Err("custom state logical_elements must be positive".to_string());
            }
            let _ = u64::try_from(*logical_elements)
                .map_err(|_| "custom state logical_elements exceed u64".to_string())?;
            Ok(())
        }
    }
}

fn validate_kind_format(kind: &StateKind, format: &NumericalFormat) -> Result<(), String> {
    match kind {
        StateKind::PositionCacheLength
            if !matches!(format, NumericalFormat::U32 | NumericalFormat::U64) =>
        {
            Err("position/cache-length state requires U32 or U64 format".to_string())
        }
        StateKind::PagedKv
        | StateKind::SlidingWindowKv
        | StateKind::Recurrent
        | StateKind::ConvolutionHistory
            if matches!(format, NumericalFormat::U32 | NumericalFormat::U64) =>
        {
            Err(format!(
                "{} state does not support U32 or U64 counter formats",
                state_kind_name(kind)
            ))
        }
        _ => Ok(()),
    }
}

fn validate_kv_layout(
    block_size: usize,
    cache_blocks: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    window_tokens: Option<usize>,
    label: &str,
) -> Result<(), String> {
    if block_size == 0
        || cache_blocks == 0
        || q_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || value_dim == 0
    {
        return Err(format!("{label} dimensions must be positive"));
    }
    if q_heads % kv_heads != 0 {
        return Err(format!("{label} q_heads must be divisible by kv_heads"));
    }
    let capacity = checked_positive_product(block_size, cache_blocks, label)?;
    if capacity > u64::from(u32::MAX) {
        return Err(format!("{label} capacity exceeds u32 index capacity"));
    }
    if let Some(window_tokens) = window_tokens {
        if window_tokens == 0 {
            return Err("sliding-window KV window_tokens must be positive".to_string());
        }
        let window_tokens = u64::try_from(window_tokens)
            .map_err(|_| "sliding-window KV window_tokens exceed u64".to_string())?;
        if window_tokens > capacity {
            return Err(
                "sliding-window KV window_tokens exceed physical cache capacity".to_string(),
            );
        }
    }
    let kv_heads = u64::try_from(kv_heads).map_err(|_| format!("{label} kv_heads exceed u64"))?;
    let head_dim = u64::try_from(head_dim).map_err(|_| format!("{label} head_dim exceed u64"))?;
    let value_dim =
        u64::try_from(value_dim).map_err(|_| format!("{label} value_dim exceed u64"))?;
    let _ = capacity
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| format!("{label} K element count overflows u64"))?;
    let _ = capacity
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(value_dim))
        .ok_or_else(|| format!("{label} V element count overflows u64"))?;
    Ok(())
}

fn validate_ownership_and_transaction(
    kind: &StateKind,
    ownership: StateOwnership,
    transaction: StateTransactionContract,
) -> Result<(), String> {
    match ownership {
        StateOwnership::ModelSharedReadOnly => {
            if transaction != StateTransactionContract::ReadOnly {
                return Err(
                    "model-shared read-only state must use ReadOnly transaction".to_string()
                );
            }
            if !matches!(kind, StateKind::Custom(_)) {
                return Err("only custom state may be model-shared read-only".to_string());
            }
        }
        StateOwnership::Request | StateOwnership::RequestLayer { .. } => {
            if !matches!(transaction, StateTransactionContract::Transactional { .. }) {
                return Err("request-owned state must use a Transactional contract".to_string());
            }
        }
    }

    match kind {
        StateKind::PagedKv
        | StateKind::SlidingWindowKv
        | StateKind::Recurrent
        | StateKind::ConvolutionHistory => {
            if !matches!(ownership, StateOwnership::RequestLayer { .. }) {
                return Err(format!(
                    "{} state must be owned by a request layer",
                    state_kind_name(kind)
                ));
            }
        }
        StateKind::PositionCacheLength => {
            if ownership != StateOwnership::Request {
                return Err("position/cache-length state must be request-owned".to_string());
            }
        }
        StateKind::Custom(_) => {}
    }
    Ok(())
}

fn layout_logical_element_count(layout: &StateLayout) -> Result<u64, String> {
    match layout {
        StateLayout::PagedKv {
            block_size,
            cache_blocks,
            kv_heads,
            head_dim,
            value_dim,
            ..
        }
        | StateLayout::SlidingWindowKv {
            block_size,
            cache_blocks,
            kv_heads,
            head_dim,
            value_dim,
            ..
        } => {
            let capacity = checked_positive_product(*block_size, *cache_blocks, "KV state")?;
            let kv_heads =
                u64::try_from(*kv_heads).map_err(|_| "KV state kv_heads exceed u64".to_string())?;
            let head_dim =
                u64::try_from(*head_dim).map_err(|_| "KV state head_dim exceed u64".to_string())?;
            let value_dim = u64::try_from(*value_dim)
                .map_err(|_| "KV state value_dim exceed u64".to_string())?;
            let k = capacity
                .checked_mul(kv_heads)
                .and_then(|value| value.checked_mul(head_dim))
                .ok_or_else(|| "KV state K element count overflows u64".to_string())?;
            let v = capacity
                .checked_mul(kv_heads)
                .and_then(|value| value.checked_mul(value_dim))
                .ok_or_else(|| "KV state V element count overflows u64".to_string())?;
            k.checked_add(v)
                .ok_or_else(|| "KV state element count overflows u64".to_string())
        }
        StateLayout::Recurrent { rows, cols } => {
            checked_positive_product(*rows, *cols, "recurrent state")
        }
        StateLayout::ConvolutionHistory {
            channels,
            history_tokens,
        } => checked_positive_product(*channels, *history_tokens, "convolution history"),
        StateLayout::PositionCacheLength => Ok(2),
        StateLayout::Custom { logical_elements } => u64::try_from(*logical_elements)
            .map_err(|_| "custom state logical_elements exceed u64".to_string()),
    }
}

fn exact_format_bytes(format: &NumericalFormat) -> Option<u64> {
    match format {
        NumericalFormat::F32 | NumericalFormat::U32 => Some(4),
        NumericalFormat::Bf16 | NumericalFormat::Fp16 => Some(2),
        NumericalFormat::U64 => Some(8),
        NumericalFormat::Aq4_0 | NumericalFormat::Sq8_0 | NumericalFormat::Custom(_) => None,
    }
}

fn checked_positive_product(left: usize, right: usize, label: &str) -> Result<u64, String> {
    if left == 0 || right == 0 {
        return Err(format!("{label} dimensions must be positive"));
    }
    let left = u64::try_from(left).map_err(|_| format!("{label} dimension exceeds u64"))?;
    let right = u64::try_from(right).map_err(|_| format!("{label} dimension exceeds u64"))?;
    left.checked_mul(right)
        .ok_or_else(|| format!("{label} element count overflows u64"))
}

fn validate_safe_id(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty() || value.len() > 128 {
        return Err(format!("{label} ID must contain 1..=128 bytes"));
    }
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        return Err(format!("{label} ID must be nonempty"));
    };
    if !first.is_ascii_alphanumeric() {
        return Err(format!(
            "{label} ID must start with an ASCII alphanumeric byte"
        ));
    }
    if bytes.any(|byte| !byte.is_ascii_alphanumeric() && !matches!(byte, b'_' | b'-' | b'.')) {
        return Err(format!("{label} ID contains an unsafe byte"));
    }
    Ok(())
}

fn state_kind_name(kind: &StateKind) -> &'static str {
    match kind {
        StateKind::PagedKv => "paged KV",
        StateKind::SlidingWindowKv => "sliding-window KV",
        StateKind::Recurrent => "recurrent",
        StateKind::ConvolutionHistory => "convolution history",
        StateKind::PositionCacheLength => "position/cache length",
        StateKind::Custom(_) => "custom",
    }
}

fn state_layout_name(layout: &StateLayout) -> &'static str {
    match layout {
        StateLayout::PagedKv { .. } => "paged KV",
        StateLayout::SlidingWindowKv { .. } => "sliding-window KV",
        StateLayout::Recurrent { .. } => "recurrent",
        StateLayout::ConvolutionHistory { .. } => "convolution history",
        StateLayout::PositionCacheLength => "position/cache length",
        StateLayout::Custom { .. } => "custom",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_graph::{
        GraphNode, GraphNodeKind, GraphValue, NodeId, PositiveF32, TensorLayout, TensorSpec,
        ValueId, WeightId, WeightSpec,
    };

    fn state_id(value: &str) -> StateId {
        StateId::new(value).unwrap()
    }

    fn transactional() -> StateTransactionContract {
        StateTransactionContract::Transactional {
            initialization: StateInitialization::Zeroed,
            execution: StateExecutionProtocol::PrepareExecuteCommit,
            reset: StateResetProtocol::Required,
            snapshot_restore: SnapshotRestorePolicy::Optional,
        }
    }

    fn paged_kv(id: &str, layer_index: usize) -> StateSpec {
        StateSpec {
            id: state_id(id),
            kind: StateKind::PagedKv,
            ownership: StateOwnership::RequestLayer { layer_index },
            format: NumericalFormat::F32,
            layout: StateLayout::PagedKv {
                block_size: 16,
                cache_blocks: 32,
                q_heads: 8,
                kv_heads: 2,
                head_dim: 64,
                value_dim: 64,
            },
            transaction: transactional(),
        }
    }

    fn position(id: &str) -> StateSpec {
        StateSpec {
            id: state_id(id),
            kind: StateKind::PositionCacheLength,
            ownership: StateOwnership::Request,
            format: NumericalFormat::U32,
            layout: StateLayout::PositionCacheLength,
            transaction: transactional(),
        }
    }

    fn recurrent(id: &str) -> StateSpec {
        StateSpec {
            id: state_id(id),
            kind: StateKind::Recurrent,
            ownership: StateOwnership::RequestLayer { layer_index: 1 },
            format: NumericalFormat::F32,
            layout: StateLayout::Recurrent { rows: 4, cols: 8 },
            transaction: transactional(),
        }
    }

    fn convolution(id: &str) -> StateSpec {
        StateSpec {
            id: state_id(id),
            kind: StateKind::ConvolutionHistory,
            ownership: StateOwnership::RequestLayer { layer_index: 1 },
            format: NumericalFormat::F32,
            layout: StateLayout::ConvolutionHistory {
                channels: 8,
                history_tokens: 4,
            },
            transaction: transactional(),
        }
    }

    fn sliding_window_kv(id: &str, window_tokens: usize) -> StateSpec {
        StateSpec {
            id: state_id(id),
            kind: StateKind::SlidingWindowKv,
            ownership: StateOwnership::RequestLayer { layer_index: 2 },
            format: NumericalFormat::F32,
            layout: StateLayout::SlidingWindowKv {
                window_tokens,
                block_size: 16,
                cache_blocks: 2,
                q_heads: 8,
                kv_heads: 2,
                head_dim: 64,
                value_dim: 64,
            },
            transaction: transactional(),
        }
    }

    fn shared_custom(id: &str) -> StateSpec {
        StateSpec {
            id: state_id(id),
            kind: StateKind::Custom(CustomStateKindId::new("lookup-cache").unwrap()),
            ownership: StateOwnership::ModelSharedReadOnly,
            format: NumericalFormat::F32,
            layout: StateLayout::Custom {
                logical_elements: 16,
            },
            transaction: StateTransactionContract::ReadOnly,
        }
    }

    fn graph_with_state(state: &str) -> ModelGraph {
        graph_with_recurrent_states(&[state])
    }

    fn graph_with_recurrent_states(states: &[&str]) -> ModelGraph {
        let input = ValueId::new("input").unwrap();
        let output = ValueId::new("output").unwrap();
        let weight = WeightId::new("weight").unwrap();
        let value_tensor =
            TensorSpec::new(vec![1], NumericalFormat::F32, TensorLayout::RowMajor).unwrap();
        let weight_tensor =
            TensorSpec::new(vec![1, 1], NumericalFormat::F32, TensorLayout::RowMajor).unwrap();
        ModelGraph {
            graph_id: "state-composition-graph".to_string(),
            inputs: vec![input.clone()],
            outputs: vec![output.clone()],
            values: vec![
                GraphValue {
                    id: input.clone(),
                    tensor: value_tensor.clone(),
                },
                GraphValue {
                    id: output.clone(),
                    tensor: value_tensor,
                },
            ],
            weights: vec![WeightSpec {
                id: weight.clone(),
                tensor: weight_tensor,
            }],
            nodes: vec![GraphNode {
                id: NodeId::new("recurrent").unwrap(),
                inputs: vec![input],
                outputs: vec![output],
                weights: vec![weight],
                states: states.iter().map(|state| state_id(state)).collect(),
                kind: GraphNodeKind::RecurrentAttention { state_width: 1 },
            }],
        }
    }

    fn dense_graph_with_state(state: &str) -> ModelGraph {
        let input = ValueId::new("dense-input").unwrap();
        let output = ValueId::new("dense-output").unwrap();
        let value_tensor =
            TensorSpec::new(vec![1, 1], NumericalFormat::F32, TensorLayout::RowMajor).unwrap();
        let weights = (0..4)
            .map(|index| WeightSpec {
                id: WeightId::new(format!("dense-weight-{index}")).unwrap(),
                tensor: TensorSpec::new(vec![1, 1], NumericalFormat::F32, TensorLayout::RowMajor)
                    .unwrap(),
            })
            .collect::<Vec<_>>();
        ModelGraph {
            graph_id: "dense-state-composition-graph".to_string(),
            inputs: vec![input.clone()],
            outputs: vec![output.clone()],
            values: vec![
                GraphValue {
                    id: input.clone(),
                    tensor: value_tensor.clone(),
                },
                GraphValue {
                    id: output.clone(),
                    tensor: value_tensor,
                },
            ],
            weights: weights.clone(),
            nodes: vec![GraphNode {
                id: NodeId::new("dense-attention").unwrap(),
                inputs: vec![input],
                outputs: vec![output],
                weights: weights.into_iter().map(|weight| weight.id).collect(),
                states: vec![state_id(state)],
                kind: GraphNodeKind::DenseAttention {
                    q_heads: 1,
                    kv_heads: 1,
                    head_dim: 1,
                    value_dim: 1,
                    softmax_scale: PositiveF32::new(1.0, "dense scale").unwrap(),
                },
            }],
        }
    }

    #[test]
    fn valid_dense_paged_kv_and_position_schema() {
        let schema = StateSchema::new(
            "dense-state-v1",
            vec![paged_kv("kv-0", 0), position("position")],
        )
        .unwrap();
        assert_eq!(schema.entries.len(), 2);
        assert_eq!(
            schema.known_logical_value_bytes_per_request().unwrap(),
            Some(524_296)
        );
    }

    #[test]
    fn valid_hybrid_paged_kv_recurrent_and_convolution_schema() {
        let recurrent = StateSpec {
            id: state_id("recurrent-1"),
            kind: StateKind::Recurrent,
            ownership: StateOwnership::RequestLayer { layer_index: 1 },
            format: NumericalFormat::Bf16,
            layout: StateLayout::Recurrent { rows: 64, cols: 64 },
            transaction: transactional(),
        };
        let convolution = StateSpec {
            id: state_id("conv-1"),
            kind: StateKind::ConvolutionHistory,
            ownership: StateOwnership::RequestLayer { layer_index: 1 },
            format: NumericalFormat::Fp16,
            layout: StateLayout::ConvolutionHistory {
                channels: 256,
                history_tokens: 4,
            },
            transaction: transactional(),
        };
        let schema = StateSchema::new(
            "hybrid-state-v1",
            vec![
                paged_kv("kv-0", 0),
                recurrent,
                convolution,
                position("position"),
            ],
        )
        .unwrap();
        assert_eq!(schema.entries.len(), 4);
        assert!(
            schema
                .known_logical_value_bytes_per_request()
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn duplicate_state_ids_are_rejected() {
        let error = StateSchema::new(
            "duplicate-state-v1",
            vec![paged_kv("kv", 0), paged_kv("kv", 1)],
        )
        .unwrap_err();
        assert!(error.contains("duplicate state ID"));
    }

    #[test]
    fn zero_paged_kv_block_is_rejected() {
        let mut state = paged_kv("kv", 0);
        let StateLayout::PagedKv { block_size, .. } = &mut state.layout else {
            panic!("test state must be paged KV");
        };
        *block_size = 0;
        let error = StateSchema::new("zero-block-v1", vec![state]).unwrap_err();
        assert!(error.contains("dimensions must be positive"));
    }

    #[test]
    fn q_head_kv_head_mismatch_is_rejected() {
        let mut state = paged_kv("kv", 0);
        let StateLayout::PagedKv {
            q_heads, kv_heads, ..
        } = &mut state.layout
        else {
            panic!("test state must be paged KV");
        };
        *q_heads = 3;
        *kv_heads = 2;
        let error = StateSchema::new("head-mismatch-v1", vec![state]).unwrap_err();
        assert!(error.contains("q_heads must be divisible by kv_heads"));
    }

    #[test]
    fn cache_capacity_overflow_is_rejected() {
        let mut state = paged_kv("kv", 0);
        let StateLayout::PagedKv {
            block_size,
            cache_blocks,
            ..
        } = &mut state.layout
        else {
            panic!("test state must be paged KV");
        };
        *block_size = usize::MAX;
        *cache_blocks = 2;
        let error = StateSchema::new("overflow-v1", vec![state]).unwrap_err();
        assert!(error.contains("overflows") || error.contains("u32 index capacity"));
    }

    #[test]
    fn kind_layout_mismatch_is_rejected() {
        let mut state = paged_kv("state", 0);
        state.kind = StateKind::Recurrent;
        let error = StateSchema::new("kind-layout-v1", vec![state]).unwrap_err();
        assert!(error.contains("incompatible"));
    }

    #[test]
    fn ownership_transaction_mismatch_is_rejected() {
        let state = StateSpec {
            id: state_id("shared-custom"),
            kind: StateKind::Custom(CustomStateKindId::new("lookup-cache").unwrap()),
            ownership: StateOwnership::ModelSharedReadOnly,
            format: NumericalFormat::F32,
            layout: StateLayout::Custom {
                logical_elements: 16,
            },
            transaction: transactional(),
        };
        let error = StateSchema::new("ownership-transaction-v1", vec![state]).unwrap_err();
        assert!(error.contains("read-only state must use ReadOnly"));
    }

    #[test]
    fn position_cache_length_requires_an_integer_format() {
        let mut state = position("position");
        state.format = NumericalFormat::F32;
        let error = StateSchema::new("position-format-v1", vec![state]).unwrap_err();
        assert!(error.contains("requires U32 or U64"));
    }

    #[test]
    fn quantized_state_does_not_claim_an_exact_byte_estimate() {
        let mut state = paged_kv("kv", 0);
        state.format = NumericalFormat::Aq4_0;
        assert_eq!(state.known_logical_value_bytes_per_request().unwrap(), None);
    }

    #[test]
    fn builtin_value_states_reject_u32_and_u64_counter_formats() {
        let mut paged = paged_kv("paged", 0);
        paged.format = NumericalFormat::U32;
        assert!(
            StateSchema::new("paged-u32", vec![paged])
                .unwrap_err()
                .contains("does not support U32 or U64")
        );

        let mut sliding = sliding_window_kv("sliding", 32);
        sliding.format = NumericalFormat::U64;
        assert!(
            StateSchema::new("sliding-u64", vec![sliding])
                .unwrap_err()
                .contains("does not support U32 or U64")
        );

        let mut recurrent = recurrent("recurrent");
        recurrent.format = NumericalFormat::U64;
        assert!(
            StateSchema::new("recurrent-u64", vec![recurrent])
                .unwrap_err()
                .contains("does not support U32 or U64")
        );

        let mut convolution = convolution("convolution");
        convolution.format = NumericalFormat::U32;
        assert!(
            StateSchema::new("convolution-u32", vec![convolution])
                .unwrap_err()
                .contains("does not support U32 or U64")
        );
    }

    #[test]
    fn state_id_uses_the_shared_model_graph_validator() {
        let schema = StateSchema::new("shared-id", vec![position("layer:0")])
            .expect("StateId accepted by ModelGraph must be accepted by StateSchema");
        assert_eq!(schema.entries[0].id.as_str(), "layer:0");
    }

    #[test]
    fn graph_composition_rejects_missing_and_orphan_request_state() {
        let missing_schema = StateSchema::new("missing-state", vec![paged_kv("kv", 0)]).unwrap();
        let missing_error = missing_schema
            .validate_against_graph(&graph_with_state("missing"))
            .unwrap_err();
        assert!(missing_error.contains("absent from state schema"));

        let orphan_schema = StateSchema::new(
            "orphan-state",
            vec![recurrent("recurrent-state"), recurrent("orphan")],
        )
        .unwrap();
        let orphan_error = orphan_schema
            .validate_against_graph(&graph_with_state("recurrent-state"))
            .unwrap_err();
        assert!(orphan_error.contains("not referenced"));
    }

    #[test]
    fn graph_composition_accepts_referenced_state_and_shared_orphan() {
        let schema = StateSchema::new(
            "composition-ok",
            vec![
                recurrent("recurrent-state"),
                position("position"),
                shared_custom("shared-cache"),
            ],
        )
        .unwrap();
        schema
            .validate_against_graph(&graph_with_state("recurrent-state"))
            .expect("referenced request state and shared read-only state must validate");

        let mut stateless_dense = dense_graph_with_state("unused-state");
        stateless_dense.nodes[0].states.clear();
        StateSchema::new("stateless-dense", vec![])
            .unwrap()
            .validate_against_graph(&stateless_dense)
            .expect("stateless dense attention fixture must be permitted");
    }

    #[test]
    fn graph_composition_rejects_node_state_kind_and_ownership_mismatches() {
        let dense_kind_schema =
            StateSchema::new("dense-kind", vec![recurrent("wrong-kind")]).unwrap();
        let dense_kind_error = dense_kind_schema
            .validate_against_graph(&dense_graph_with_state("wrong-kind"))
            .unwrap_err();
        assert!(dense_kind_error.contains("must be PagedKv or SlidingWindowKv"));

        let recurrent_kind_schema =
            StateSchema::new("recurrent-kind", vec![paged_kv("wrong-kind", 0)]).unwrap();
        let recurrent_kind_error = recurrent_kind_schema
            .validate_against_graph(&graph_with_state("wrong-kind"))
            .unwrap_err();
        assert!(recurrent_kind_error.contains("first state"));

        let second_kind_schema = StateSchema::new(
            "second-kind",
            vec![recurrent("recurrent"), paged_kv("wrong-second", 0)],
        )
        .unwrap();
        let second_kind_error = second_kind_schema
            .validate_against_graph(&graph_with_recurrent_states(&["recurrent", "wrong-second"]))
            .unwrap_err();
        assert!(second_kind_error.contains("second state"));

        let mut wrong_ownership = paged_kv("ownership", 0);
        wrong_ownership.ownership = StateOwnership::Request;
        let ownership_error = StateSchema::new("ownership", vec![wrong_ownership]).unwrap_err();
        assert!(ownership_error.contains("owned by a request layer"));
    }

    #[test]
    fn sliding_window_geometry_accepts_boundary_and_rejects_excess() {
        StateSchema::new("sliding-boundary", vec![sliding_window_kv("sliding", 32)])
            .expect("window equal to physical capacity must validate");
        let error =
            StateSchema::new("sliding-excess", vec![sliding_window_kv("sliding", 33)]).unwrap_err();
        assert!(error.contains("exceed physical cache capacity"));
    }

    #[test]
    fn paged_kv_u32_capacity_boundary_is_checked() {
        let mut exact = paged_kv("exact", 0);
        let StateLayout::PagedKv {
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
        } = &mut exact.layout
        else {
            panic!("test state must be paged KV");
        };
        *block_size = u32::MAX as usize;
        *cache_blocks = 1;
        *q_heads = 1;
        *kv_heads = 1;
        *head_dim = 1;
        *value_dim = 1;
        StateSchema::new("u32-capacity-exact", vec![exact])
            .expect("u32 maximum token capacity must remain representable");

        let mut excessive = paged_kv("excessive", 0);
        let StateLayout::PagedKv {
            block_size,
            cache_blocks,
            q_heads,
            kv_heads,
            head_dim,
            value_dim,
        } = &mut excessive.layout
        else {
            panic!("test state must be paged KV");
        };
        *block_size = u32::MAX as usize;
        *cache_blocks = 2;
        *q_heads = 1;
        *kv_heads = 1;
        *head_dim = 1;
        *value_dim = 1;
        assert!(
            StateSchema::new("u32-capacity-excess", vec![excessive])
                .unwrap_err()
                .contains("u32 index capacity")
        );
    }

    #[test]
    fn u64_position_and_sq8_or_custom_bytes_are_handled_without_guessing() {
        let mut counter = position("position");
        counter.format = NumericalFormat::U64;
        assert_eq!(
            counter.known_logical_value_bytes_per_request().unwrap(),
            Some(16)
        );

        let mut sq8 = paged_kv("sq8", 0);
        sq8.format = NumericalFormat::Sq8_0;
        assert_eq!(sq8.known_logical_value_bytes_per_request().unwrap(), None);

        let mut custom = paged_kv("custom", 0);
        custom.format = NumericalFormat::custom("opaque-state-v1").unwrap();
        assert_eq!(
            custom.known_logical_value_bytes_per_request().unwrap(),
            None
        );
    }
}
