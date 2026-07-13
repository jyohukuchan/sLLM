// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Quantization-independent reasoning phase and thinking-budget state machine.

use std::collections::{HashSet, VecDeque};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningPhase {
    Disabled,
    Reasoning,
    ForcingEndSequence,
    Answer,
    Finished,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEmissionKind {
    Reasoning,
    Answer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningEmission {
    pub kind: ReasoningEmissionKind,
    pub token_id: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningExecution {
    pub enabled: bool,
    pub budget_tokens: Option<usize>,
    pub dialect_id: String,
    pub end_sequence: Vec<usize>,
    pub forced_end_sequence: Vec<usize>,
    pub reserved_answer_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningDialect {
    pub identity: String,
    pub start_sequence: Vec<usize>,
    pub end_sequence: Vec<usize>,
    pub forced_end_sequence: Vec<usize>,
    pub max_budget_tokens: usize,
    pub reserved_answer_tokens: usize,
    pub enabled_by_default: bool,
    pub effort_budgets: Vec<(String, usize)>,
    pub history_reasoning_policy: HistoryReasoningPolicy,
    pub initial_phase: InitialReasoningPhase,
    pub eos_policy: ReasoningEosPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryReasoningPolicy {
    Omit,
    Preserve,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialReasoningPhase {
    Reasoning,
    Answer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEosPolicy {
    Close,
    Finish,
    Continue,
}

impl ReasoningDialect {
    pub fn validate(&self, vocab_size: usize) -> Result<(), ReasoningError> {
        if self.identity.is_empty() || self.identity.len() > 256 {
            return Err(ReasoningError::InvalidDialect(
                "dialect identity must be nonempty and bounded",
            ));
        }
        for (name, sequence) in [
            ("start_sequence", &self.start_sequence),
            ("end_sequence", &self.end_sequence),
            ("forced_end_sequence", &self.forced_end_sequence),
        ] {
            if sequence.is_empty() {
                return Err(ReasoningError::InvalidDialect(name));
            }
            if sequence.iter().any(|token| *token >= vocab_size) {
                return Err(ReasoningError::InvalidDialect(name));
            }
        }
        if self.end_sequence != self.forced_end_sequence {
            return Err(ReasoningError::InvalidDialect(
                "natural and forced end sequences must match",
            ));
        }
        if self.reserved_answer_tokens == 0 {
            return Err(ReasoningError::InvalidDialect(
                "reserved_answer_tokens must be positive",
            ));
        }
        let effort_names = self
            .effort_budgets
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<HashSet<_>>();
        if effort_names != HashSet::from(["low", "medium", "high"])
            || self
                .effort_budgets
                .iter()
                .any(|(_, budget)| *budget == 0 || *budget > self.max_budget_tokens)
        {
            return Err(ReasoningError::InvalidDialect(
                "effort_budgets must declare valid low, medium, and high budgets",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasoningError {
    InvalidDialect(&'static str),
    InvalidBudget,
    InvalidPhase(ReasoningPhase),
    ForcedTokenMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningStep {
    pub emissions: Vec<ReasoningEmission>,
    pub request_forced_close: bool,
    pub phase: ReasoningPhase,
}

#[derive(Debug, Clone)]
pub struct ReasoningState {
    dialect: ReasoningDialect,
    budget_tokens: Option<usize>,
    pub reasoning_tokens: usize,
    pub answer_tokens: usize,
    pending: VecDeque<usize>,
    forced_index: usize,
    pub phase: ReasoningPhase,
}

impl ReasoningState {
    pub fn new(
        dialect: ReasoningDialect,
        enabled: bool,
        budget_tokens: Option<usize>,
        vocab_size: usize,
    ) -> Result<Self, ReasoningError> {
        dialect.validate(vocab_size)?;
        if budget_tokens.is_some_and(|budget| budget > dialect.max_budget_tokens) {
            return Err(ReasoningError::InvalidBudget);
        }
        let phase = if !enabled {
            ReasoningPhase::Disabled
        } else if dialect.initial_phase == InitialReasoningPhase::Answer {
            ReasoningPhase::Answer
        } else if budget_tokens == Some(0) {
            ReasoningPhase::ForcingEndSequence
        } else {
            ReasoningPhase::Reasoning
        };
        Ok(Self {
            dialect,
            budget_tokens,
            reasoning_tokens: 0,
            answer_tokens: 0,
            pending: VecDeque::new(),
            forced_index: 0,
            phase,
        })
    }

    pub fn accept_sampled(&mut self, token_id: usize) -> Result<ReasoningStep, ReasoningError> {
        if self.phase == ReasoningPhase::Disabled || self.phase == ReasoningPhase::Answer {
            self.answer_tokens = self.answer_tokens.saturating_add(1);
            self.phase = ReasoningPhase::Answer;
            return Ok(ReasoningStep {
                emissions: vec![ReasoningEmission {
                    kind: ReasoningEmissionKind::Answer,
                    token_id,
                }],
                request_forced_close: false,
                phase: self.phase,
            });
        }
        if self.phase != ReasoningPhase::Reasoning {
            return Err(ReasoningError::InvalidPhase(self.phase));
        }
        self.pending.push_back(token_id);
        if self.pending.iter().eq(self.dialect.end_sequence.iter()) {
            self.pending.clear();
            self.phase = ReasoningPhase::Answer;
            return Ok(ReasoningStep {
                emissions: Vec::new(),
                request_forced_close: false,
                phase: self.phase,
            });
        }
        let mut emissions = Vec::new();
        while !self.pending.is_empty() && !is_prefix(&self.pending, &self.dialect.end_sequence) {
            let body_token = self.pending.pop_front().expect("pending is nonempty");
            if self
                .budget_tokens
                .is_some_and(|budget| self.reasoning_tokens >= budget)
            {
                self.pending.clear();
                self.phase = ReasoningPhase::ForcingEndSequence;
                return Ok(ReasoningStep {
                    emissions,
                    request_forced_close: true,
                    phase: self.phase,
                });
            }
            self.reasoning_tokens = self.reasoning_tokens.saturating_add(1);
            emissions.push(ReasoningEmission {
                kind: ReasoningEmissionKind::Reasoning,
                token_id: body_token,
            });
            if self
                .budget_tokens
                .is_some_and(|budget| self.reasoning_tokens >= budget)
            {
                self.pending.clear();
                self.phase = ReasoningPhase::ForcingEndSequence;
                return Ok(ReasoningStep {
                    emissions,
                    request_forced_close: true,
                    phase: self.phase,
                });
            }
        }
        Ok(ReasoningStep {
            emissions,
            request_forced_close: false,
            phase: self.phase,
        })
    }

    pub fn accept_forced(&mut self, token_id: usize) -> Result<ReasoningStep, ReasoningError> {
        if self.phase != ReasoningPhase::ForcingEndSequence {
            return Err(ReasoningError::InvalidPhase(self.phase));
        }
        if self.dialect.forced_end_sequence[self.forced_index] != token_id {
            return Err(ReasoningError::ForcedTokenMismatch);
        }
        self.forced_index += 1;
        if self.forced_index == self.dialect.forced_end_sequence.len() {
            self.phase = ReasoningPhase::Answer;
        }
        Ok(ReasoningStep {
            emissions: Vec::new(),
            request_forced_close: false,
            phase: self.phase,
        })
    }

    pub fn next_forced_token(&self) -> Option<usize> {
        (self.phase == ReasoningPhase::ForcingEndSequence)
            .then(|| {
                self.dialect
                    .forced_end_sequence
                    .get(self.forced_index)
                    .copied()
            })
            .flatten()
    }

    pub fn force_close(&mut self) -> Result<(), ReasoningError> {
        if self.phase != ReasoningPhase::Reasoning {
            return Err(ReasoningError::InvalidPhase(self.phase));
        }
        self.pending.clear();
        self.phase = ReasoningPhase::ForcingEndSequence;
        Ok(())
    }

    pub fn on_eos(&mut self) -> ReasoningStep {
        if self.phase == ReasoningPhase::Reasoning
            && self.dialect.eos_policy == ReasoningEosPolicy::Close
        {
            self.pending.clear();
            self.phase = ReasoningPhase::ForcingEndSequence;
            return ReasoningStep {
                emissions: Vec::new(),
                request_forced_close: true,
                phase: self.phase,
            };
        }
        if matches!(
            self.phase,
            ReasoningPhase::Answer | ReasoningPhase::Disabled
        ) || self.dialect.eos_policy == ReasoningEosPolicy::Finish
        {
            self.phase = ReasoningPhase::Finished;
        }
        ReasoningStep {
            emissions: Vec::new(),
            request_forced_close: false,
            phase: self.phase,
        }
    }

    pub fn finish(&mut self) -> Result<(), ReasoningError> {
        if !matches!(
            self.phase,
            ReasoningPhase::Answer | ReasoningPhase::Disabled
        ) {
            return Err(ReasoningError::InvalidPhase(self.phase));
        }
        self.phase = ReasoningPhase::Finished;
        Ok(())
    }

    pub fn cancel(&mut self) {
        self.phase = ReasoningPhase::Cancelled;
        self.pending.clear();
    }
}

fn is_prefix(candidate: &VecDeque<usize>, sequence: &[usize]) -> bool {
    candidate.len() < sequence.len() && candidate.iter().eq(sequence.iter().take(candidate.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dialect() -> ReasoningDialect {
        ReasoningDialect {
            identity: "synthetic.multi-token.v1".into(),
            start_sequence: vec![10, 11],
            end_sequence: vec![20, 21, 22],
            forced_end_sequence: vec![20, 21, 22],
            max_budget_tokens: 128,
            reserved_answer_tokens: 1,
            enabled_by_default: false,
            effort_budgets: vec![
                ("low".into(), 32),
                ("medium".into(), 64),
                ("high".into(), 128),
            ],
            history_reasoning_policy: HistoryReasoningPolicy::Omit,
            initial_phase: InitialReasoningPhase::Reasoning,
            eos_policy: ReasoningEosPolicy::Close,
        }
    }

    #[test]
    fn natural_multi_token_end_is_suppressed() {
        let mut state = ReasoningState::new(dialect(), true, None, 100).unwrap();
        assert_eq!(state.accept_sampled(1).unwrap().emissions[0].token_id, 1);
        assert_eq!(state.accept_sampled(2).unwrap().emissions[0].token_id, 2);
        for token in [20, 21, 22] {
            assert!(state.accept_sampled(token).unwrap().emissions.is_empty());
        }
        assert_eq!(state.phase, ReasoningPhase::Answer);
        assert_eq!(
            state.accept_sampled(30).unwrap().emissions[0].kind,
            ReasoningEmissionKind::Answer
        );
    }

    #[test]
    fn hard_budget_forces_close_without_overshoot() {
        let mut state = ReasoningState::new(dialect(), true, Some(2), 100).unwrap();
        assert_eq!(state.accept_sampled(1).unwrap().emissions.len(), 1);
        let step = state.accept_sampled(2).unwrap();
        assert!(step.request_forced_close);
        assert_eq!(state.reasoning_tokens, 2);
        for token in [20, 21, 22] {
            state.accept_forced(token).unwrap();
        }
        assert_eq!(state.phase, ReasoningPhase::Answer);
    }

    #[test]
    fn zero_budget_starts_forced_close() {
        let mut state = ReasoningState::new(dialect(), true, Some(0), 100).unwrap();
        assert_eq!(state.phase, ReasoningPhase::ForcingEndSequence);
        for token in [20, 21, 22] {
            state.accept_forced(token).unwrap();
        }
        assert_eq!(state.phase, ReasoningPhase::Answer);
        assert_eq!(state.reasoning_tokens, 0);
    }

    #[test]
    fn eos_requests_declared_close() {
        let mut state = ReasoningState::new(dialect(), true, None, 100).unwrap();
        assert!(state.on_eos().request_forced_close);
        assert_eq!(state.phase, ReasoningPhase::ForcingEndSequence);
    }
}
