use sllm_core::{
    CONTEXT_POSITION_POLICY_VERSION_V1, ContextAdapterCapabilitiesV1, ContextPositionPolicyV1,
    ContextShiftError, ContextShiftKindV1, ContextWindowStateV1,
};

fn keep(prefix: u64, recent: u64) -> ContextPositionPolicyV1 {
    ContextPositionPolicyV1::keep_prefix_recent_v1(prefix, recent).unwrap()
}

#[test]
fn disabled_policy_is_noop_until_overflow_and_fails_closed_at_capacity() {
    let policy = ContextPositionPolicyV1::disabled();
    let state = ContextWindowStateV1::new(63, 1000, 0);
    let no_shift = policy.plan(state, 64, 1).unwrap();
    assert_eq!(no_shift.kind(), ContextShiftKindV1::NoShift);
    assert_eq!(no_shift.proposed_state(), state);
    assert_eq!(
        policy.plan(state, 64, 2),
        Err(ContextShiftError::ShiftDisabled)
    );
    assert_eq!(
        policy.plan(ContextWindowStateV1::new(64, 1000, 0), 64, 1),
        Err(ContextShiftError::ShiftDisabled)
    );
    // Disabled means no model-specific adapter validation is required.
    assert!(
        policy
            .validate_adapter(ContextAdapterCapabilitiesV1::unsupported())
            .is_ok()
    );
}

#[test]
fn non_aligned_capacity_boundaries_plan_before_overflow() {
    for capacity in [63_u64, 64, 65, 127, 128, 129] {
        let state = ContextWindowStateV1::new(capacity - 1, 10_000, 0);
        let policy = keep(1, 1);
        assert_eq!(
            policy.plan(state, capacity, 1).unwrap().kind(),
            ContextShiftKindV1::ShiftRequired,
            "capacity {capacity} shifts before the exact-fill append"
        );
        let decision = policy.plan(state, capacity, 2).unwrap();
        assert_eq!(decision.kind(), ContextShiftKindV1::ShiftRequired);
        assert_eq!(decision.old_state(), state);
        assert_eq!(decision.proposed_state().logical_length(), 2);
        assert_eq!(decision.proposed_state().absolute_position(), 10_000);
        assert_eq!(decision.proposed_state().shift_count(), 1);

        let at_capacity = ContextWindowStateV1::new(capacity, 10_000, 0);
        assert_eq!(
            policy.plan(at_capacity, capacity, 0).unwrap().kind(),
            ContextShiftKindV1::NoShift
        );
        let at_capacity_shift = policy.plan(at_capacity, capacity, 1).unwrap();
        assert_eq!(at_capacity_shift.kind(), ContextShiftKindV1::ShiftRequired);
        assert_eq!(at_capacity_shift.proposed_state().logical_length(), 2);
        assert_eq!(
            at_capacity_shift.proposed_state().absolute_position(),
            10_000
        );
        assert_eq!(
            policy.plan(
                ContextWindowStateV1::new(capacity + 1, 10_000, 0),
                capacity,
                0
            ),
            Err(ContextShiftError::StateExceedsCapacity {
                logical_length: capacity + 1,
                capacity,
            })
        );
    }
}

#[test]
fn retained_prefix_recent_ranges_keep_absolute_position_and_transaction_is_pure() {
    let policy = keep(5, 3);
    let old = ContextWindowStateV1::new(100, 1_000, 4);
    let decision = policy.plan(old, 128, 29).unwrap();
    assert!(decision.requires_shift());
    let ranges = decision.retained_ranges().unwrap();
    assert_eq!(ranges.prefix().start(), 0);
    assert_eq!(ranges.prefix().end(), 5);
    assert_eq!(ranges.recent().start(), 97);
    assert_eq!(ranges.recent().end(), 100);
    assert_eq!(ranges.retained_tokens().unwrap(), 8);
    assert_eq!(decision.proposed_state().logical_length(), 8);
    assert_eq!(decision.proposed_state().absolute_position(), 1_000);
    assert_eq!(decision.proposed_state().shift_count(), 5);

    let transaction = decision.transaction().unwrap();
    assert_eq!(transaction.old_state(), old);
    assert_eq!(transaction.proposed_state(), decision.proposed_state());
    // No owner was changed by planning. The caller publishes only this value
    // after rebuilding the opaque state successfully.
    assert_eq!(transaction.commit(), decision.proposed_state());
    assert_eq!(decision.old_state(), old);
}

#[test]
fn retained_materialization_preserves_tokens_and_absolute_rope_positions() {
    let policy = keep(2, 3);
    let old = ContextWindowStateV1::new(10, 1_010, 0);
    let decision = policy.plan(old, 12, 2).unwrap();
    let history: Vec<i32> = (0..10).collect();
    assert_eq!(
        decision.retained_token_ids(&history).unwrap(),
        [0, 1, 7, 8, 9]
    );
    assert_eq!(
        decision
            .retained_absolute_positions(history.len() as u64)
            .unwrap(),
        [1_000, 1_001, 1_007, 1_008, 1_009]
    );
    assert_eq!(history, (0..10).collect::<Vec<_>>());
}

#[test]
fn keep_zero_one_and_max_boundaries_are_explicit() {
    let state = ContextWindowStateV1::new(63, 63, 0);
    let prefix_zero = keep(0, 1).plan(state, 64, 2).unwrap();
    let ranges = prefix_zero.retained_ranges().unwrap();
    assert!(ranges.prefix().is_empty());
    assert_eq!(ranges.recent().len(), 1);

    let recent_zero = keep(1, 0).plan(state, 64, 2).unwrap();
    let ranges = recent_zero.retained_ranges().unwrap();
    assert_eq!(ranges.prefix().len(), 1);
    assert!(ranges.recent().is_empty());

    assert_eq!(
        ContextPositionPolicyV1::keep_prefix_recent_v1(0, 0),
        Err(ContextShiftError::EmptyRetainedWindow)
    );
    // The maximum useful prefix for this state is 63, but retaining it leaves
    // insufficient room for the incoming block; fail closed rather than
    // silently dropping more tokens.
    assert_eq!(
        keep(63, 0).plan(state, 64, 2),
        Err(ContextShiftError::RetainedWindowTooLarge {
            retained: 63,
            incoming_tokens: 2,
            capacity: 64,
        })
    );
    assert_eq!(
        keep(64, 0).plan(state, 64, 2),
        Err(ContextShiftError::RetainedRangeExceedsState {
            logical_length: 63,
            keep_prefix: 64,
            keep_recent: 0,
        })
    );
}

#[test]
fn repeated_shifts_increment_count_without_rewinding_absolute_position() {
    let policy = keep(1, 1);
    let first_old = ContextWindowStateV1::new(63, 63, 0);
    let first = policy.plan(first_old, 64, 2).unwrap();
    let compacted = first.transaction().unwrap().commit();
    assert_eq!(compacted.shift_count(), 1);
    let after_first = compacted.after_append(61, 64).unwrap();
    assert_eq!(after_first.logical_length(), 63);
    assert_eq!(after_first.absolute_position(), 124);

    let second = policy.plan(after_first, 64, 2).unwrap();
    assert_eq!(second.proposed_state().logical_length(), 2);
    assert_eq!(second.proposed_state().absolute_position(), 124);
    assert_eq!(second.proposed_state().shift_count(), 2);
}

#[test]
fn repeated_shift_materialization_compacts_history_before_next_append() {
    let policy = keep(1, 1);
    let original = vec![10, 11, 12, 13];
    let state = ContextWindowStateV1::new(4, 40, 0);
    let first = policy.plan(state, 4, 1).unwrap();
    assert!(first.requires_shift());
    let mut history = first.retained_token_ids(&original).unwrap();
    assert_eq!(history, vec![10, 13]);
    let mut compacted = first.transaction().unwrap().commit();
    assert_eq!(compacted.logical_length(), history.len() as u64);
    history.push(99);
    compacted = compacted.after_append(1, 4).unwrap();

    let second = policy.plan(compacted, 4, 1).unwrap();
    let source_before_second = history.clone();
    let second_history = second.retained_token_ids(&history).unwrap();
    assert_eq!(second_history, vec![10, 99]);
    assert_eq!(history, source_before_second);
    assert_eq!(second.proposed_state().shift_count(), 2);
}

#[test]
fn overflow_and_unsupported_policy_errors_are_fail_closed() {
    assert_eq!(
        ContextPositionPolicyV1::keep_prefix_recent_v1(u64::MAX, 1),
        Err(ContextShiftError::PolicyArithmeticOverflow)
    );
    let policy = keep(1, 1);
    assert_eq!(
        policy.plan(ContextWindowStateV1::new(u64::MAX - 1, 0, 0), u64::MAX, 2),
        Err(ContextShiftError::PositionOverflow)
    );
    assert_eq!(
        ContextWindowStateV1::new(1, u64::MAX, 0).after_append(1, 64),
        Err(ContextShiftError::PositionOverflow)
    );
    assert_eq!(
        ContextPositionPolicyV1::disabled().plan(
            ContextWindowStateV1::new(u64::MAX - 1, u64::MAX, 0),
            u64::MAX,
            1,
        ),
        Err(ContextShiftError::PositionOverflow)
    );
    assert_eq!(
        policy.plan(ContextWindowStateV1::new(63, 63, u64::MAX), 64, 2),
        Err(ContextShiftError::ShiftCountOverflow)
    );
    assert_eq!(
        ContextPositionPolicyV1::unsupported(9, 42).plan(
            ContextWindowStateV1::new(63, 0, 0),
            64,
            2,
        ),
        Err(ContextShiftError::UnsupportedPolicy {
            version: 9,
            tag: 42
        })
    );
    assert_eq!(
        ContextPositionPolicyV1::KeepPrefixRecentV1 {
            version: CONTEXT_POSITION_POLICY_VERSION_V1 + 1,
            keep_prefix: 1,
            keep_recent: 1,
        }
        .plan(ContextWindowStateV1::new(63, 0, 0), 64, 2),
        Err(ContextShiftError::UnsupportedPolicyVersion {
            version: CONTEXT_POSITION_POLICY_VERSION_V1 + 1,
        })
    );
}

#[test]
fn adapter_must_validate_rope_and_attention_contracts() {
    let policy = keep(1, 1);
    assert_eq!(
        policy.validate_adapter(ContextAdapterCapabilitiesV1::unsupported()),
        Err(ContextShiftError::AdapterValidationRequired {
            policy_version: CONTEXT_POSITION_POLICY_VERSION_V1,
        })
    );
    let supported = ContextAdapterCapabilitiesV1::new(1, 1, 1, true, true);
    assert!(policy.validate_adapter(supported).is_ok());
    assert_eq!(
        policy.validate_adapter(ContextAdapterCapabilitiesV1::new(1, 2, 1, true, true)),
        Err(ContextShiftError::AdapterRopePolicyMismatch {
            expected: 1,
            actual: 2,
        })
    );
    assert_eq!(
        policy.validate_adapter(ContextAdapterCapabilitiesV1::new(1, 1, 2, true, true)),
        Err(ContextShiftError::AdapterAttentionMaskPolicyMismatch {
            expected: 1,
            actual: 2,
        })
    );
    assert_eq!(
        policy.validate_adapter(ContextAdapterCapabilitiesV1::new(1, 1, 1, false, true)),
        Err(ContextShiftError::AdapterAbsolutePositionUnsupported)
    );
    assert_eq!(
        policy.validate_adapter(ContextAdapterCapabilitiesV1::new(1, 1, 1, true, false)),
        Err(ContextShiftError::AdapterDiscontiguousRetentionUnsupported)
    );
}
