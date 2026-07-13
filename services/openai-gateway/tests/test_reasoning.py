from __future__ import annotations

import pytest

from ullm_openai_gateway.reasoning import (
    EmissionKind,
    ReasoningDialect,
    ReasoningError,
    ReasoningPhase,
    ReasoningState,
    split_reasoning_completion,
)


@pytest.fixture
def dialect() -> ReasoningDialect:
    return ReasoningDialect(
        identity="synthetic.multi-token.v1",
        start_sequence=(10, 11),
        end_sequence=(20, 21, 22),
        forced_end_sequence=(20, 21, 22),
        max_budget_tokens=128,
        reserved_answer_tokens=1,
        effort_budgets=(("low", 32), ("medium", 64), ("high", 128)),
    )


def test_natural_multi_token_end_is_suppressed(dialect: ReasoningDialect) -> None:
    state = ReasoningState(dialect, enabled=True, budget_tokens=None, vocab_size=100)
    assert [item.token_id for item in state.accept_sampled(1).emissions] == [1]
    assert [item.token_id for item in state.accept_sampled(2).emissions] == [2]
    assert [state.accept_sampled(token).emissions for token in (20, 21, 22)] == [
        (),
        (),
        (),
    ]
    assert state.phase == ReasoningPhase.ANSWER
    answer = state.accept_sampled(30)
    assert answer.emissions[0].kind == EmissionKind.ANSWER
    assert answer.emissions[0].token_id == 30


def test_budget_forces_close_without_overshoot(dialect: ReasoningDialect) -> None:
    state = ReasoningState(dialect, enabled=True, budget_tokens=2, vocab_size=100)
    first = state.accept_sampled(1)
    second = state.accept_sampled(2)
    assert [item.token_id for item in first.emissions + second.emissions] == [1, 2]
    assert all(
        item.kind == EmissionKind.REASONING
        for item in first.emissions + second.emissions
    )
    assert second.request_forced_close
    assert state.reasoning_tokens == 2
    assert state.phase == ReasoningPhase.FORCING_END_SEQUENCE
    for token in (20, 21, 22):
        state.accept_forced(token)
    assert state.phase == ReasoningPhase.ANSWER
    assert state.accept_sampled(31).emissions[0].token_id == 31


def test_budget_zero_forces_close_immediately(dialect: ReasoningDialect) -> None:
    state = ReasoningState(dialect, enabled=True, budget_tokens=0, vocab_size=100)
    assert state.phase == ReasoningPhase.FORCING_END_SEQUENCE
    assert state.accept_forced(20).phase == ReasoningPhase.FORCING_END_SEQUENCE
    assert state.accept_forced(21).phase == ReasoningPhase.FORCING_END_SEQUENCE
    assert state.accept_forced(22).phase == ReasoningPhase.ANSWER
    assert state.reasoning_tokens == 0


def test_eos_uses_declared_close_policy(dialect: ReasoningDialect) -> None:
    state = ReasoningState(dialect, enabled=True, budget_tokens=None, vocab_size=100)
    step = state.on_eos()
    assert step.request_forced_close
    assert state.phase == ReasoningPhase.FORCING_END_SEQUENCE


def test_invalid_forced_token_is_rejected(dialect: ReasoningDialect) -> None:
    state = ReasoningState(dialect, enabled=True, budget_tokens=0, vocab_size=100)
    with pytest.raises(ReasoningError):
        state.accept_forced(99)


def test_raw_completion_split_handles_natural_and_forced_close(dialect: ReasoningDialect) -> None:
    natural = split_reasoning_completion(
        (1, 2, 20, 21, 22, 30, 99),
        dialect=dialect,
        enabled=True,
        budget_tokens=None,
        eos_token_ids=(99,),
        vocab_size=100,
    )
    assert natural.reasoning_token_ids == (1, 2)
    assert natural.answer_token_ids == (30,)
    forced = split_reasoning_completion(
        (1, 2, 20, 21, 22, 30),
        dialect=dialect,
        enabled=True,
        budget_tokens=2,
        vocab_size=100,
    )
    assert forced.reasoning_token_ids == (1, 2)
    assert forced.answer_token_ids == (30,)
    assert forced.forced_end_tokens == 3


def test_dialect_rejects_empty_or_oversized_configuration() -> None:
    dialect = ReasoningDialect(
        identity="bad",
        start_sequence=(),
        end_sequence=(2,),
        forced_end_sequence=(2,),
        max_budget_tokens=1,
        reserved_answer_tokens=1,
    )
    with pytest.raises(ReasoningError):
        ReasoningState(dialect, enabled=True, budget_tokens=1, vocab_size=10)
