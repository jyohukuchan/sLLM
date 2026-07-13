"""Model-independent reasoning dialect validation and token segmentation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ReasoningError(ValueError):
    """Raised when a reasoning dialect or token transition is invalid."""


class ReasoningPhase(str, Enum):
    DISABLED = "disabled"
    REASONING = "reasoning"
    FORCING_END_SEQUENCE = "forcing-end-sequence"
    ANSWER = "answer"
    FINISHED = "finished"
    CANCELLED = "cancelled"


class EmissionKind(str, Enum):
    REASONING = "reasoning"
    ANSWER = "answer"


@dataclass(frozen=True, slots=True)
class ReasoningDialect:
    """Declared model-specific reasoning token contract."""

    identity: str
    start_sequence: tuple[int, ...]
    end_sequence: tuple[int, ...]
    forced_end_sequence: tuple[int, ...]
    max_budget_tokens: int
    reserved_answer_tokens: int
    enabled_by_default: bool = False
    effort_budgets: tuple[tuple[str, int], ...] = ()
    history_reasoning_policy: str = "omit"
    initial_phase: str = "reasoning"
    eos_policy: str = "close"

    def validate(self, *, vocab_size: int) -> None:
        if not self.identity or len(self.identity.encode("utf-8")) > 256:
            raise ReasoningError("dialect identity must be nonempty and bounded")
        for name, sequence in (
            ("start_sequence", self.start_sequence),
            ("end_sequence", self.end_sequence),
            ("forced_end_sequence", self.forced_end_sequence),
        ):
            if not sequence:
                raise ReasoningError(f"{name} must not be empty")
            if len(sequence) != len(set(sequence)):
                raise ReasoningError(f"{name} contains duplicate tokens")
            if any(token < 0 or token >= vocab_size for token in sequence):
                raise ReasoningError(f"{name} contains a token outside the vocabulary")
        if self.end_sequence != self.forced_end_sequence:
            raise ReasoningError(
                "natural and forced end sequences must match until divergent policies exist"
            )
        if _has_prefix_collision(self.start_sequence, self.end_sequence):
            raise ReasoningError("start and end sequences must not have a prefix collision")
        if self.max_budget_tokens < 0:
            raise ReasoningError("max_budget_tokens must be nonnegative")
        if self.reserved_answer_tokens < 1:
            raise ReasoningError("reserved_answer_tokens must be positive")
        effort_names = {name for name, _ in self.effort_budgets}
        if effort_names != {"low", "medium", "high"}:
            raise ReasoningError("effort_budgets must declare low, medium, and high")
        if any(
            not isinstance(name, str)
            or not isinstance(budget, int)
            or budget < 1
            or budget > self.max_budget_tokens
            for name, budget in self.effort_budgets
        ):
            raise ReasoningError("effort_budgets contains an invalid budget")
        if self.history_reasoning_policy not in {"omit", "preserve"}:
            raise ReasoningError("history_reasoning_policy is invalid")
        if self.initial_phase not in {"reasoning", "answer"}:
            raise ReasoningError("initial_phase is invalid")
        if self.eos_policy not in {"close", "finish", "continue"}:
            raise ReasoningError("eos_policy is invalid")

    def budget_for_effort(self, effort: str) -> int:
        for name, budget in self.effort_budgets:
            if name == effort:
                return budget
        raise ReasoningError("reasoning_effort is not declared by the dialect")


@dataclass(frozen=True, slots=True)
class ReasoningRequest:
    enabled: bool
    budget_tokens: int | None
    history_reasoning_policy: str
    dialect_id: str


@dataclass(frozen=True, slots=True)
class ReasoningOutput:
    reasoning_token_ids: tuple[int, ...]
    answer_token_ids: tuple[int, ...]
    reasoning_tokens: int
    forced_end_tokens: int


def split_reasoning_completion(
    token_ids: Iterable[int],
    *,
    dialect: ReasoningDialect,
    enabled: bool,
    budget_tokens: int | None,
    eos_token_ids: Iterable[int] = (),
    vocab_size: int,
) -> ReasoningOutput:
    """Split raw generated IDs without searching decoded text for delimiters."""

    state = ReasoningState(
        dialect,
        enabled=enabled,
        budget_tokens=budget_tokens,
        vocab_size=vocab_size,
    )
    reasoning: list[int] = []
    answer: list[int] = []
    eos = set(eos_token_ids)
    forced_end_tokens = 0
    for token_id in token_ids:
        if token_id in eos:
            step = state.on_eos()
        elif state.phase == ReasoningPhase.FORCING_END_SEQUENCE:
            step = state.accept_forced(token_id)
            forced_end_tokens += 1
        else:
            step = state.accept_sampled(token_id)
        for emission in step.emissions:
            if emission.kind == EmissionKind.REASONING:
                reasoning.append(emission.token_id)
            else:
                answer.append(emission.token_id)
    if state.phase == ReasoningPhase.FORCING_END_SEQUENCE:
        raise ReasoningError("raw completion ended before forced end sequence")
    return ReasoningOutput(
        reasoning_token_ids=tuple(reasoning),
        answer_token_ids=tuple(answer),
        reasoning_tokens=state.reasoning_tokens,
        forced_end_tokens=forced_end_tokens,
    )


@dataclass(frozen=True, slots=True)
class Emission:
    kind: EmissionKind
    token_id: int


@dataclass(frozen=True, slots=True)
class ReasoningStep:
    emissions: tuple[Emission, ...] = ()
    request_forced_close: bool = False
    phase: ReasoningPhase = ReasoningPhase.REASONING


class ReasoningState:
    """Token-level state machine shared by gateway and worker adapters."""

    def __init__(
        self,
        dialect: ReasoningDialect,
        *,
        enabled: bool,
        budget_tokens: int | None,
        vocab_size: int,
    ) -> None:
        dialect.validate(vocab_size=vocab_size)
        if budget_tokens is not None and budget_tokens < 0:
            raise ReasoningError("budget_tokens must be nonnegative or None")
        if budget_tokens is not None and budget_tokens > dialect.max_budget_tokens:
            raise ReasoningError("budget_tokens exceeds dialect maximum")
        self.dialect = dialect
        self.budget_tokens = budget_tokens
        self.reasoning_tokens = 0
        self.answer_tokens = 0
        self._pending: deque[int] = deque()
        self._forced_index = 0
        if not enabled:
            self.phase = ReasoningPhase.DISABLED
        elif dialect.initial_phase == "answer":
            self.phase = ReasoningPhase.ANSWER
        elif budget_tokens == 0:
            self.phase = ReasoningPhase.FORCING_END_SEQUENCE
        else:
            self.phase = ReasoningPhase.REASONING

    def accept_sampled(self, token_id: int) -> ReasoningStep:
        self._validate_token(token_id)
        if self.phase in {ReasoningPhase.DISABLED, ReasoningPhase.ANSWER}:
            self.answer_tokens += 1
            self.phase = ReasoningPhase.ANSWER
            return ReasoningStep(
                (Emission(EmissionKind.ANSWER, token_id),), phase=self.phase
            )
        if self.phase != ReasoningPhase.REASONING:
            raise ReasoningError(f"cannot accept sampled token in phase {self.phase.value}")
        self._pending.append(token_id)
        if tuple(self._pending) == self.dialect.end_sequence:
            self._pending.clear()
            self.phase = ReasoningPhase.ANSWER
            return ReasoningStep(phase=self.phase)

        emissions: list[Emission] = []
        while self._pending and not _is_prefix(
            tuple(self._pending), self.dialect.end_sequence
        ):
            body_token = self._pending.popleft()
            if self.budget_tokens is not None and self.reasoning_tokens >= self.budget_tokens:
                self._pending.clear()
                self.phase = ReasoningPhase.FORCING_END_SEQUENCE
                return ReasoningStep(
                    tuple(emissions), request_forced_close=True, phase=self.phase
                )
            self.reasoning_tokens += 1
            emissions.append(Emission(EmissionKind.REASONING, body_token))
            if (
                self.budget_tokens is not None
                and self.reasoning_tokens >= self.budget_tokens
            ):
                self._pending.clear()
                self.phase = ReasoningPhase.FORCING_END_SEQUENCE
                return ReasoningStep(
                    tuple(emissions), request_forced_close=True, phase=self.phase
                )
        return ReasoningStep(tuple(emissions), phase=self.phase)

    def accept_forced(self, token_id: int) -> ReasoningStep:
        self._validate_token(token_id)
        if self.phase != ReasoningPhase.FORCING_END_SEQUENCE:
            raise ReasoningError(f"forced token is invalid in phase {self.phase.value}")
        expected = self.dialect.forced_end_sequence[self._forced_index]
        if token_id != expected:
            raise ReasoningError("forced token does not match the declared end sequence")
        self._forced_index += 1
        if self._forced_index == len(self.dialect.forced_end_sequence):
            self.phase = ReasoningPhase.ANSWER
        return ReasoningStep(phase=self.phase)

    def on_eos(self) -> ReasoningStep:
        if self.phase == ReasoningPhase.REASONING and self.dialect.eos_policy == "close":
            self._pending.clear()
            self.phase = ReasoningPhase.FORCING_END_SEQUENCE
            return ReasoningStep(request_forced_close=True, phase=self.phase)
        if self.phase in {ReasoningPhase.ANSWER, ReasoningPhase.DISABLED} or self.dialect.eos_policy == "finish":
            self.phase = ReasoningPhase.FINISHED
        return ReasoningStep(phase=self.phase)

    def finish(self) -> None:
        if self.phase not in {ReasoningPhase.ANSWER, ReasoningPhase.DISABLED}:
            raise ReasoningError(f"cannot finish in phase {self.phase.value}")
        self.phase = ReasoningPhase.FINISHED

    def cancel(self) -> None:
        self.phase = ReasoningPhase.CANCELLED
        self._pending.clear()

    def _validate_token(self, token_id: int) -> None:
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ReasoningError("token ID must be a nonnegative integer")


def _is_prefix(candidate: Iterable[int], sequence: tuple[int, ...]) -> bool:
    value = tuple(candidate)
    return len(value) < len(sequence) and sequence[: len(value)] == value


def _has_prefix_collision(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return (
        len(left) <= len(right)
        and right[: len(left)] == left
    ) or (
        len(right) <= len(left)
        and left[: len(right)] == right
    )
