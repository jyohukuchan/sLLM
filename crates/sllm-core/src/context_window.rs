//! Backend-neutral context-window shift decisions.
//!
//! This module does not move KV/GDN bytes and does not implement a model's
//! RoPE, mRoPE, or attention mask.  It computes a checked, pure decision that
//! an adapter can validate, build into a new opaque state, and publish only
//! after that build succeeds.

use std::fmt;

/// Version of the context-position policy contract implemented here.
pub const CONTEXT_POSITION_POLICY_VERSION_V1: u32 = 1;

/// A half-open logical token interval retained by a context shift.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ContextTokenRangeV1 {
    start: u64,
    end: u64,
}

impl ContextTokenRangeV1 {
    pub fn new(start: u64, end: u64) -> Result<Self, ContextShiftError> {
        if end < start {
            return Err(ContextShiftError::InvalidRetainedRange);
        }
        Ok(Self { start, end })
    }

    pub const fn start(self) -> u64 {
        self.start
    }

    pub const fn end(self) -> u64 {
        self.end
    }

    pub const fn len(self) -> u64 {
        self.end - self.start
    }

    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// Logical/absolute position state for the next token to be appended.
///
/// `logical_length` is the number of tokens currently represented by the
/// request-local state. `absolute_position` is the monotonically increasing
/// position assigned to the next token in the original conversation. A shift
/// compacts logical positions but never rewinds the absolute position.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ContextWindowStateV1 {
    logical_length: u64,
    absolute_position: u64,
    shift_count: u64,
}

impl ContextWindowStateV1 {
    pub const fn new(logical_length: u64, absolute_position: u64, shift_count: u64) -> Self {
        Self {
            logical_length,
            absolute_position,
            shift_count,
        }
    }

    pub const fn logical_length(self) -> u64 {
        self.logical_length
    }

    pub const fn logical_position(self) -> u64 {
        self.logical_length
    }

    pub const fn absolute_position(self) -> u64 {
        self.absolute_position
    }

    pub const fn shift_count(self) -> u64 {
        self.shift_count
    }

    fn validate(self) -> Result<(), ContextShiftError> {
        let delta = self
            .absolute_position
            .checked_sub(self.logical_length)
            .ok_or(ContextShiftError::PositionOverflow)?;
        i64::try_from(delta).map_err(|_| ContextShiftError::PositionOverflow)?;
        Ok(())
    }

    /// Advance the state after a successful append. This is pure and does not
    /// mutate an opaque backend owner.
    pub fn after_append(
        self,
        appended_tokens: u64,
        capacity: u64,
    ) -> Result<Self, ContextShiftError> {
        self.validate()?;
        if capacity == 0 {
            return Err(ContextShiftError::InvalidCapacity);
        }
        if self.logical_length > capacity {
            return Err(ContextShiftError::StateExceedsCapacity {
                logical_length: self.logical_length,
                capacity,
            });
        }
        let logical_length = self
            .logical_length
            .checked_add(appended_tokens)
            .ok_or(ContextShiftError::PositionOverflow)?;
        if logical_length > capacity {
            return Err(ContextShiftError::CapacityExceeded {
                required: logical_length,
                capacity,
            });
        }
        let absolute_position = self
            .absolute_position
            .checked_add(appended_tokens)
            .ok_or(ContextShiftError::PositionOverflow)?;
        Ok(Self {
            logical_length,
            absolute_position,
            shift_count: self.shift_count,
        })
    }
}

/// The policy used to decide whether a new state must be built.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ContextPositionPolicyV1 {
    Disabled {
        version: u32,
    },
    KeepPrefixRecentV1 {
        version: u32,
        keep_prefix: u64,
        keep_recent: u64,
    },
    /// Represents a version/tag received from a newer producer. It is always
    /// rejected by this implementation rather than silently approximated.
    Unsupported {
        version: u32,
        tag: u32,
    },
}

impl ContextPositionPolicyV1 {
    pub const fn disabled() -> Self {
        Self::Disabled {
            version: CONTEXT_POSITION_POLICY_VERSION_V1,
        }
    }

    pub fn keep_prefix_recent_v1(
        keep_prefix: u64,
        keep_recent: u64,
    ) -> Result<Self, ContextShiftError> {
        let retained = keep_prefix
            .checked_add(keep_recent)
            .ok_or(ContextShiftError::PolicyArithmeticOverflow)?;
        if retained == 0 {
            return Err(ContextShiftError::EmptyRetainedWindow);
        }
        Ok(Self::KeepPrefixRecentV1 {
            version: CONTEXT_POSITION_POLICY_VERSION_V1,
            keep_prefix,
            keep_recent,
        })
    }

    pub const fn unsupported(version: u32, tag: u32) -> Self {
        Self::Unsupported { version, tag }
    }

    pub const fn version(self) -> u32 {
        match self {
            Self::Disabled { version }
            | Self::KeepPrefixRecentV1 { version, .. }
            | Self::Unsupported { version, .. } => version,
        }
    }

    pub const fn is_disabled(self) -> bool {
        matches!(self, Self::Disabled { .. })
    }

    pub const fn keep_prefix(self) -> Option<u64> {
        match self {
            Self::KeepPrefixRecentV1 { keep_prefix, .. } => Some(keep_prefix),
            _ => None,
        }
    }

    pub const fn keep_recent(self) -> Option<u64> {
        match self {
            Self::KeepPrefixRecentV1 { keep_recent, .. } => Some(keep_recent),
            _ => None,
        }
    }

    fn validate(self) -> Result<(), ContextShiftError> {
        match self {
            Self::Disabled { version } if version == CONTEXT_POSITION_POLICY_VERSION_V1 => Ok(()),
            Self::KeepPrefixRecentV1 {
                version,
                keep_prefix,
                keep_recent,
            } if version == CONTEXT_POSITION_POLICY_VERSION_V1 => {
                let retained = keep_prefix
                    .checked_add(keep_recent)
                    .ok_or(ContextShiftError::PolicyArithmeticOverflow)?;
                if retained == 0 {
                    return Err(ContextShiftError::EmptyRetainedWindow);
                }
                Ok(())
            }
            Self::Disabled { version } | Self::KeepPrefixRecentV1 { version, .. } => {
                Err(ContextShiftError::UnsupportedPolicyVersion { version })
            }
            Self::Unsupported { version, tag } => {
                Err(ContextShiftError::UnsupportedPolicy { version, tag })
            }
        }
    }

    pub const fn adapter_requirements(self) -> ContextAdapterRequirementsV1 {
        match self {
            Self::Disabled { version } => ContextAdapterRequirementsV1 {
                policy_version: version,
                requires_rope: false,
                requires_attention_mask: false,
                requires_absolute_positions: false,
                requires_discontiguous_ranges: false,
            },
            Self::KeepPrefixRecentV1 {
                version,
                keep_prefix,
                keep_recent,
            } => ContextAdapterRequirementsV1 {
                policy_version: version,
                requires_rope: true,
                requires_attention_mask: true,
                requires_absolute_positions: true,
                requires_discontiguous_ranges: keep_prefix != 0 && keep_recent != 0,
            },
            Self::Unsupported { version, .. } => ContextAdapterRequirementsV1 {
                policy_version: version,
                requires_rope: true,
                requires_attention_mask: true,
                requires_absolute_positions: true,
                requires_discontiguous_ranges: true,
            },
        }
    }

    pub fn validate_adapter(
        self,
        capabilities: ContextAdapterCapabilitiesV1,
    ) -> Result<(), ContextShiftError> {
        self.validate()?;
        if self.is_disabled() {
            return Ok(());
        }
        capabilities.validate(self.adapter_requirements())
    }

    /// Compute a no-op or shift-required decision without touching backend
    /// state. A shifting policy may compact an owner that is already exactly
    /// at capacity before admitting the next token; disabled policy retains
    /// its legacy exact-fill behavior and fails closed on overflow.
    pub fn plan(
        self,
        state: ContextWindowStateV1,
        capacity: u64,
        incoming_tokens: u64,
    ) -> Result<ContextShiftDecisionV1, ContextShiftError> {
        self.validate()?;
        state.validate()?;
        if capacity == 0 {
            return Err(ContextShiftError::InvalidCapacity);
        }
        if state.logical_length > capacity {
            return Err(ContextShiftError::StateExceedsCapacity {
                logical_length: state.logical_length,
                capacity,
            });
        }
        let required = state
            .logical_length
            .checked_add(incoming_tokens)
            .ok_or(ContextShiftError::PositionOverflow)?;
        state
            .absolute_position
            .checked_add(incoming_tokens)
            .ok_or(ContextShiftError::PositionOverflow)?;
        // Disabled policy retains its legacy ability to fill the final slot
        // exactly. A shifting policy compacts whenever the incoming append
        // would fill or exceed capacity, including an owner already at the
        // capacity boundary.
        if incoming_tokens == 0
            || required < capacity
            || (required == capacity && self.is_disabled())
        {
            return Ok(ContextShiftDecisionV1::no_shift(state, incoming_tokens));
        }
        let Self::KeepPrefixRecentV1 {
            keep_prefix,
            keep_recent,
            ..
        } = self
        else {
            return Err(ContextShiftError::ShiftDisabled);
        };
        if keep_prefix > state.logical_length || keep_recent > state.logical_length {
            return Err(ContextShiftError::RetainedRangeExceedsState {
                logical_length: state.logical_length,
                keep_prefix,
                keep_recent,
            });
        }
        let retained = keep_prefix
            .checked_add(keep_recent)
            .ok_or(ContextShiftError::PolicyArithmeticOverflow)?;
        if retained > state.logical_length {
            return Err(ContextShiftError::RetainedRangesOverlap {
                logical_length: state.logical_length,
                keep_prefix,
                keep_recent,
            });
        }
        let retained_with_incoming = retained
            .checked_add(incoming_tokens)
            .ok_or(ContextShiftError::PositionOverflow)?;
        if retained_with_incoming > capacity {
            return Err(ContextShiftError::RetainedWindowTooLarge {
                retained,
                incoming_tokens,
                capacity,
            });
        }
        let recent_start = state
            .logical_length
            .checked_sub(keep_recent)
            .ok_or(ContextShiftError::PolicyArithmeticOverflow)?;
        let prefix = ContextTokenRangeV1::new(0, keep_prefix)?;
        let recent = ContextTokenRangeV1::new(recent_start, state.logical_length)?;
        let shift_count = state
            .shift_count
            .checked_add(1)
            .ok_or(ContextShiftError::ShiftCountOverflow)?;
        let proposed_state = ContextWindowStateV1 {
            logical_length: retained,
            absolute_position: state.absolute_position,
            shift_count,
        };
        Ok(ContextShiftDecisionV1::shift_required(
            state,
            proposed_state,
            incoming_tokens,
            ContextRetainedRangesV1 { prefix, recent },
        ))
    }

    /// Plans the initial rebuild for a prompt that already exceeds the
    /// compact state capacity. This is the prompt-side counterpart to
    /// [`Self::plan`]; the old owner is only a token history, so no backend
    /// state has been touched yet.
    pub fn plan_initial(
        self,
        token_count: u64,
        capacity: u64,
    ) -> Result<ContextShiftDecisionV1, ContextShiftError> {
        self.validate()?;
        if capacity == 0 {
            return Err(ContextShiftError::InvalidCapacity);
        }
        if token_count == 0 {
            return Err(ContextShiftError::EmptyRetainedWindow);
        }
        let state = ContextWindowStateV1::new(token_count, token_count, 0);
        state.validate()?;
        if token_count <= capacity {
            return Ok(ContextShiftDecisionV1::no_shift(state, 0));
        }
        let Self::KeepPrefixRecentV1 {
            keep_prefix,
            keep_recent,
            ..
        } = self
        else {
            return Err(ContextShiftError::ShiftDisabled);
        };
        if keep_prefix > token_count || keep_recent > token_count {
            return Err(ContextShiftError::RetainedRangeExceedsState {
                logical_length: token_count,
                keep_prefix,
                keep_recent,
            });
        }
        let retained = keep_prefix
            .checked_add(keep_recent)
            .ok_or(ContextShiftError::PolicyArithmeticOverflow)?;
        if retained == 0 || retained > capacity {
            return Err(ContextShiftError::RetainedWindowTooLarge {
                retained,
                incoming_tokens: 0,
                capacity,
            });
        }
        if retained > token_count {
            return Err(ContextShiftError::RetainedRangesOverlap {
                logical_length: token_count,
                keep_prefix,
                keep_recent,
            });
        }
        let recent_start = token_count
            .checked_sub(keep_recent)
            .ok_or(ContextShiftError::PolicyArithmeticOverflow)?;
        let prefix = ContextTokenRangeV1::new(0, keep_prefix)?;
        let recent = ContextTokenRangeV1::new(recent_start, token_count)?;
        Ok(ContextShiftDecisionV1::shift_required(
            state,
            ContextWindowStateV1::new(retained, token_count, 1),
            0,
            ContextRetainedRangesV1 { prefix, recent },
        ))
    }
}

/// Adapter-side requirements for a policy. The model adapter, not this core
/// module, supplies the actual RoPE/mRoPE and attention-mask implementation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ContextAdapterRequirementsV1 {
    policy_version: u32,
    requires_rope: bool,
    requires_attention_mask: bool,
    requires_absolute_positions: bool,
    requires_discontiguous_ranges: bool,
}

impl ContextAdapterRequirementsV1 {
    pub const fn policy_version(self) -> u32 {
        self.policy_version
    }

    pub const fn requires_rope(self) -> bool {
        self.requires_rope
    }

    pub const fn requires_attention_mask(self) -> bool {
        self.requires_attention_mask
    }

    pub const fn requires_absolute_positions(self) -> bool {
        self.requires_absolute_positions
    }

    pub const fn requires_discontiguous_ranges(self) -> bool {
        self.requires_discontiguous_ranges
    }
}

/// Capabilities reported by a model adapter before a shift is admitted.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ContextAdapterCapabilitiesV1 {
    policy_version: u32,
    rope_policy_version: u32,
    attention_mask_policy_version: u32,
    supports_absolute_positions: bool,
    supports_discontiguous_ranges: bool,
}

impl ContextAdapterCapabilitiesV1 {
    pub const fn unsupported() -> Self {
        Self {
            policy_version: 0,
            rope_policy_version: 0,
            attention_mask_policy_version: 0,
            supports_absolute_positions: false,
            supports_discontiguous_ranges: false,
        }
    }

    pub const fn new(
        policy_version: u32,
        rope_policy_version: u32,
        attention_mask_policy_version: u32,
        supports_absolute_positions: bool,
        supports_discontiguous_ranges: bool,
    ) -> Self {
        Self {
            policy_version,
            rope_policy_version,
            attention_mask_policy_version,
            supports_absolute_positions,
            supports_discontiguous_ranges,
        }
    }

    pub const fn policy_version(self) -> u32 {
        self.policy_version
    }

    pub const fn rope_policy_version(self) -> u32 {
        self.rope_policy_version
    }

    pub const fn attention_mask_policy_version(self) -> u32 {
        self.attention_mask_policy_version
    }

    pub const fn supports_absolute_positions(self) -> bool {
        self.supports_absolute_positions
    }

    pub const fn supports_discontiguous_ranges(self) -> bool {
        self.supports_discontiguous_ranges
    }

    fn validate(self, requirements: ContextAdapterRequirementsV1) -> Result<(), ContextShiftError> {
        if self.policy_version == 0
            || self.rope_policy_version == 0
            || self.attention_mask_policy_version == 0
        {
            return Err(ContextShiftError::AdapterValidationRequired {
                policy_version: requirements.policy_version,
            });
        }
        if self.policy_version != requirements.policy_version {
            return Err(ContextShiftError::AdapterPolicyVersionMismatch {
                expected: requirements.policy_version,
                actual: self.policy_version,
            });
        }
        if requirements.requires_rope && self.rope_policy_version != requirements.policy_version {
            return Err(ContextShiftError::AdapterRopePolicyMismatch {
                expected: requirements.policy_version,
                actual: self.rope_policy_version,
            });
        }
        if requirements.requires_attention_mask
            && self.attention_mask_policy_version != requirements.policy_version
        {
            return Err(ContextShiftError::AdapterAttentionMaskPolicyMismatch {
                expected: requirements.policy_version,
                actual: self.attention_mask_policy_version,
            });
        }
        if requirements.requires_absolute_positions && !self.supports_absolute_positions {
            return Err(ContextShiftError::AdapterAbsolutePositionUnsupported);
        }
        if requirements.requires_discontiguous_ranges && !self.supports_discontiguous_ranges {
            return Err(ContextShiftError::AdapterDiscontiguousRetentionUnsupported);
        }
        Ok(())
    }
}

/// Retained prefix and recent ranges in the old logical state.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ContextRetainedRangesV1 {
    prefix: ContextTokenRangeV1,
    recent: ContextTokenRangeV1,
}

impl ContextRetainedRangesV1 {
    pub const fn prefix(self) -> ContextTokenRangeV1 {
        self.prefix
    }

    pub const fn recent(self) -> ContextTokenRangeV1 {
        self.recent
    }

    pub fn retained_tokens(self) -> Result<u64, ContextShiftError> {
        self.prefix
            .len()
            .checked_add(self.recent.len())
            .ok_or(ContextShiftError::PolicyArithmeticOverflow)
    }
}

/// Whether an append can proceed without rebuilding the state.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ContextShiftKindV1 {
    NoShift,
    ShiftRequired,
}

/// Pure decision and proposed old/new positions. No backend state has been
/// mutated when this value is returned.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ContextShiftDecisionV1 {
    kind: ContextShiftKindV1,
    old_state: ContextWindowStateV1,
    proposed_state: ContextWindowStateV1,
    incoming_tokens: u64,
    retained_ranges: Option<ContextRetainedRangesV1>,
}

impl ContextShiftDecisionV1 {
    const fn no_shift(state: ContextWindowStateV1, incoming_tokens: u64) -> Self {
        Self {
            kind: ContextShiftKindV1::NoShift,
            old_state: state,
            proposed_state: state,
            incoming_tokens,
            retained_ranges: None,
        }
    }

    const fn shift_required(
        old_state: ContextWindowStateV1,
        proposed_state: ContextWindowStateV1,
        incoming_tokens: u64,
        retained_ranges: ContextRetainedRangesV1,
    ) -> Self {
        Self {
            kind: ContextShiftKindV1::ShiftRequired,
            old_state,
            proposed_state,
            incoming_tokens,
            retained_ranges: Some(retained_ranges),
        }
    }

    pub const fn kind(self) -> ContextShiftKindV1 {
        self.kind
    }

    pub const fn requires_shift(self) -> bool {
        matches!(self.kind, ContextShiftKindV1::ShiftRequired)
    }

    pub const fn old_state(self) -> ContextWindowStateV1 {
        self.old_state
    }

    pub const fn proposed_state(self) -> ContextWindowStateV1 {
        self.proposed_state
    }

    pub const fn incoming_tokens(self) -> u64 {
        self.incoming_tokens
    }

    pub const fn retained_ranges(self) -> Option<ContextRetainedRangesV1> {
        self.retained_ranges
    }

    /// Materializes the retained token sequence for a transactional rebuild.
    /// The returned vector is newly owned; the caller's history is never
    /// modified. For a no-shift decision the full history is copied.
    pub fn retained_token_ids(self, token_history: &[i32]) -> Result<Vec<i32>, ContextShiftError> {
        if self.old_state.logical_length
            != u64::try_from(token_history.len())
                .map_err(|_| ContextShiftError::PositionOverflow)?
        {
            return Err(ContextShiftError::StateExceedsCapacity {
                logical_length: self.old_state.logical_length,
                capacity: u64::try_from(token_history.len()).unwrap_or(u64::MAX),
            });
        }
        let Some(ranges) = self.retained_ranges else {
            return Ok(token_history.to_vec());
        };
        let prefix_start = usize::try_from(ranges.prefix.start())
            .map_err(|_| ContextShiftError::PositionOverflow)?;
        let prefix_end = usize::try_from(ranges.prefix.end())
            .map_err(|_| ContextShiftError::PositionOverflow)?;
        let recent_start = usize::try_from(ranges.recent.start())
            .map_err(|_| ContextShiftError::PositionOverflow)?;
        let recent_end = usize::try_from(ranges.recent.end())
            .map_err(|_| ContextShiftError::PositionOverflow)?;
        let retained = prefix_end
            .checked_sub(prefix_start)
            .and_then(|prefix_len| {
                recent_end
                    .checked_sub(recent_start)
                    .and_then(|recent_len| prefix_len.checked_add(recent_len))
            })
            .ok_or(ContextShiftError::PolicyArithmeticOverflow)?;
        let mut tokens = Vec::with_capacity(retained);
        tokens.extend_from_slice(
            token_history
                .get(prefix_start..prefix_end)
                .ok_or(ContextShiftError::InvalidRetainedRange)?,
        );
        tokens.extend_from_slice(
            token_history
                .get(recent_start..recent_end)
                .ok_or(ContextShiftError::InvalidRetainedRange)?,
        );
        Ok(tokens)
    }

    /// Materializes the absolute RoPE position for each retained token. The
    /// sequence has compact logical order but preserves the original absolute
    /// positions, including the gap between retained prefix and recent tail.
    pub fn retained_absolute_positions(
        self,
        token_history_len: u64,
    ) -> Result<Vec<u64>, ContextShiftError> {
        if self.old_state.logical_length != token_history_len {
            return Err(ContextShiftError::StateExceedsCapacity {
                logical_length: self.old_state.logical_length,
                capacity: token_history_len,
            });
        }
        let base = self
            .old_state
            .absolute_position
            .checked_sub(self.old_state.logical_length)
            .ok_or(ContextShiftError::PositionOverflow)?;
        let mut positions = Vec::new();
        let ranges = self.retained_ranges.unwrap_or(ContextRetainedRangesV1 {
            prefix: ContextTokenRangeV1::new(0, token_history_len)?,
            recent: ContextTokenRangeV1::new(token_history_len, token_history_len)?,
        });
        for range in [ranges.prefix, ranges.recent] {
            for logical in range.start()..range.end() {
                positions.push(
                    base.checked_add(logical)
                        .ok_or(ContextShiftError::PositionOverflow)?,
                );
            }
        }
        Ok(positions)
    }

    pub fn transaction(self) -> Option<ContextShiftTransactionV1> {
        self.requires_shift().then_some(ContextShiftTransactionV1 {
            old_state: self.old_state,
            proposed_state: self.proposed_state,
        })
    }
}

/// Transaction boundary for old/new state publication. The caller constructs
/// the new opaque state from `proposed_state`; only `commit` should replace the
/// old owner after that construction succeeds. Dropping this value publishes
/// nothing and leaves the old state valid.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ContextShiftTransactionV1 {
    old_state: ContextWindowStateV1,
    proposed_state: ContextWindowStateV1,
}

impl ContextShiftTransactionV1 {
    pub const fn old_state(self) -> ContextWindowStateV1 {
        self.old_state
    }

    pub const fn proposed_state(self) -> ContextWindowStateV1 {
        self.proposed_state
    }

    pub const fn commit(self) -> ContextWindowStateV1 {
        self.proposed_state
    }
}

/// Fail-closed errors for policy, arithmetic, and adapter validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContextShiftError {
    InvalidCapacity,
    InvalidRetainedRange,
    EmptyRetainedWindow,
    PolicyArithmeticOverflow,
    PositionOverflow,
    ShiftCountOverflow,
    UnsupportedPolicyVersion {
        version: u32,
    },
    UnsupportedPolicy {
        version: u32,
        tag: u32,
    },
    ShiftDisabled,
    StateExceedsCapacity {
        logical_length: u64,
        capacity: u64,
    },
    CapacityReached {
        capacity: u64,
    },
    CapacityExceeded {
        required: u64,
        capacity: u64,
    },
    RetainedRangeExceedsState {
        logical_length: u64,
        keep_prefix: u64,
        keep_recent: u64,
    },
    RetainedRangesOverlap {
        logical_length: u64,
        keep_prefix: u64,
        keep_recent: u64,
    },
    RetainedWindowTooLarge {
        retained: u64,
        incoming_tokens: u64,
        capacity: u64,
    },
    AdapterValidationRequired {
        policy_version: u32,
    },
    AdapterPolicyVersionMismatch {
        expected: u32,
        actual: u32,
    },
    AdapterRopePolicyMismatch {
        expected: u32,
        actual: u32,
    },
    AdapterAttentionMaskPolicyMismatch {
        expected: u32,
        actual: u32,
    },
    AdapterAbsolutePositionUnsupported,
    AdapterDiscontiguousRetentionUnsupported,
}

impl fmt::Display for ContextShiftError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCapacity => formatter.write_str("context capacity must be non-zero"),
            Self::InvalidRetainedRange => formatter.write_str("context retained range is inverted"),
            Self::EmptyRetainedWindow => {
                formatter.write_str("context retained window must keep at least one token")
            }
            Self::PolicyArithmeticOverflow => {
                formatter.write_str("context policy arithmetic overflowed")
            }
            Self::PositionOverflow => {
                formatter.write_str("context logical or absolute position overflowed")
            }
            Self::ShiftCountOverflow => formatter.write_str("context shift count overflowed"),
            Self::UnsupportedPolicyVersion { version } => {
                write!(formatter, "unsupported context policy version {version}")
            }
            Self::UnsupportedPolicy { version, tag } => write!(
                formatter,
                "unsupported context policy version {version} tag {tag}"
            ),
            Self::ShiftDisabled => formatter.write_str("context shift is disabled"),
            Self::StateExceedsCapacity {
                logical_length,
                capacity,
            } => write!(
                formatter,
                "logical context length {logical_length} exceeds capacity {capacity}"
            ),
            Self::CapacityReached { capacity } => write!(
                formatter,
                "context capacity {capacity} was reached before shift admission"
            ),
            Self::CapacityExceeded { required, capacity } => write!(
                formatter,
                "context append requires {required} tokens over capacity {capacity}"
            ),
            Self::RetainedRangeExceedsState {
                logical_length,
                keep_prefix,
                keep_recent,
            } => write!(
                formatter,
                "retained prefix/recent ({keep_prefix}+{keep_recent}) exceeds logical state {logical_length}"
            ),
            Self::RetainedRangesOverlap {
                logical_length,
                keep_prefix,
                keep_recent,
            } => write!(
                formatter,
                "retained prefix/recent ({keep_prefix}+{keep_recent}) overlap in logical state {logical_length}"
            ),
            Self::RetainedWindowTooLarge {
                retained,
                incoming_tokens,
                capacity,
            } => write!(
                formatter,
                "retained window {retained} plus incoming {incoming_tokens} exceeds capacity {capacity}"
            ),
            Self::AdapterValidationRequired { policy_version } => write!(
                formatter,
                "adapter must validate context policy version {policy_version}"
            ),
            Self::AdapterPolicyVersionMismatch { expected, actual } => write!(
                formatter,
                "adapter context policy version {actual}, expected {expected}"
            ),
            Self::AdapterRopePolicyMismatch { expected, actual } => write!(
                formatter,
                "adapter RoPE policy version {actual}, expected {expected}"
            ),
            Self::AdapterAttentionMaskPolicyMismatch { expected, actual } => write!(
                formatter,
                "adapter attention-mask policy version {actual}, expected {expected}"
            ),
            Self::AdapterAbsolutePositionUnsupported => {
                formatter.write_str("adapter does not support absolute context positions")
            }
            Self::AdapterDiscontiguousRetentionUnsupported => formatter
                .write_str("adapter does not support discontiguous prefix/recent retention"),
        }
    }
}

impl std::error::Error for ContextShiftError {}
