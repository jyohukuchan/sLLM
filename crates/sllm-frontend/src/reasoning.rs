//! Request-local reasoning control for the generation frontend.
//!
//! Reasoning is deliberately kept at the same token-selection boundary as
//! grammar and stop-token handling.  The controller does not rewrite model
//! output after decode and it never submits a second decode loop: a forced
//! closing marker is selected by intersecting the existing candidate mask.

use crate::chat::ThinkingModeV1;

/// Maximum number of generated reasoning tokens accepted by Phase 44.
pub const MAX_REASONING_TOKENS_V1: u32 = 4_096;

/// Maximum length of a model-specific closing marker in token IDs.
///
/// Qwen's marker is short, but this bound keeps a malformed profile from
/// turning a request-local controller into an unbounded buffer.
pub const MAX_REASONING_CLOSE_TOKENS_V1: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReasoningModeV1 {
    Disabled,
    Enabled,
    TemplateDefault,
}

impl From<ThinkingModeV1> for ReasoningModeV1 {
    fn from(mode: ThinkingModeV1) -> Self {
        match mode {
            ThinkingModeV1::Disabled => Self::Disabled,
            ThinkingModeV1::Enabled => Self::Enabled,
            ThinkingModeV1::TemplateDefault => Self::TemplateDefault,
        }
    }
}

impl From<ReasoningModeV1> for ThinkingModeV1 {
    fn from(mode: ReasoningModeV1) -> Self {
        match mode {
            ReasoningModeV1::Disabled => Self::Disabled,
            ReasoningModeV1::Enabled => Self::Enabled,
            ReasoningModeV1::TemplateDefault => Self::TemplateDefault,
        }
    }
}

impl ReasoningModeV1 {
    pub const fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }

    /// Template-default is resolved by the renderer.  The generation
    /// frontend has no model lock here, so an unresolved default is inactive
    /// rather than silently enabling hidden reasoning.
    pub const fn active(self) -> bool {
        self.is_enabled()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReasoningErrorV1 {
    BudgetOutOfRange,
    EmptyClosingSequence,
    ClosingSequenceTooLong,
    MaxTokensTooSmall,
    TokenOutsideVocabulary,
    CandidateMaskUnavailable,
    CandidateMaskEmpty,
    ForcedTokenMismatch,
    BudgetExceeded,
    CountOverflow,
}

impl core::fmt::Display for ReasoningErrorV1 {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(match self {
            Self::BudgetOutOfRange => "reasoning budget must be in 1..=4096",
            Self::EmptyClosingSequence => "reasoning closing token sequence is empty",
            Self::ClosingSequenceTooLong => "reasoning closing token sequence is too long",
            Self::MaxTokensTooSmall => {
                "max_new_tokens must include the reasoning budget and closing sequence"
            }
            Self::TokenOutsideVocabulary => "reasoning forced token is outside the vocabulary",
            Self::CandidateMaskUnavailable => {
                "reasoning forced close requires a candidate mask or logits"
            }
            Self::CandidateMaskEmpty => "reasoning mask intersection disabled every candidate",
            Self::ForcedTokenMismatch => "selected token differs from the forced reasoning close",
            Self::BudgetExceeded => "reasoning token budget was exceeded",
            Self::CountOverflow => "reasoning token accounting overflowed",
        })
    }
}

impl std::error::Error for ReasoningErrorV1 {}

/// Immutable request policy.  `None` for `max_reasoning_tokens` retains the
/// legacy enabled-thinking behavior: a closing sequence is recognized, but no
/// forced close is introduced.  A disabled policy is intentionally accepted so
/// API adapters can lower a typed mode without creating a second path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReasoningPolicyV1 {
    mode: ReasoningModeV1,
    max_reasoning_tokens: Option<u32>,
    closing_token_ids: Vec<u32>,
}

impl ReasoningPolicyV1 {
    pub fn new(
        mode: ReasoningModeV1,
        max_reasoning_tokens: Option<u32>,
        closing_token_ids: impl Into<Vec<u32>>,
    ) -> Result<Self, ReasoningErrorV1> {
        if let Some(budget) = max_reasoning_tokens {
            if !(1..=MAX_REASONING_TOKENS_V1).contains(&budget) {
                return Err(ReasoningErrorV1::BudgetOutOfRange);
            }
        }
        let closing_token_ids = closing_token_ids.into();
        if mode.active() && closing_token_ids.is_empty() {
            return Err(ReasoningErrorV1::EmptyClosingSequence);
        }
        if closing_token_ids.len() > MAX_REASONING_CLOSE_TOKENS_V1 {
            return Err(ReasoningErrorV1::ClosingSequenceTooLong);
        }
        Ok(Self {
            mode,
            max_reasoning_tokens,
            closing_token_ids,
        })
    }

    pub fn disabled() -> Self {
        Self {
            mode: ReasoningModeV1::Disabled,
            max_reasoning_tokens: None,
            closing_token_ids: Vec::new(),
        }
    }

    pub fn enabled(
        max_reasoning_tokens: Option<u32>,
        closing_token_ids: impl Into<Vec<u32>>,
    ) -> Result<Self, ReasoningErrorV1> {
        Self::new(
            ReasoningModeV1::Enabled,
            max_reasoning_tokens,
            closing_token_ids,
        )
    }

    pub fn from_thinking(
        thinking: ThinkingModeV1,
        max_reasoning_tokens: Option<u32>,
        closing_token_ids: impl Into<Vec<u32>>,
    ) -> Result<Self, ReasoningErrorV1> {
        Self::new(thinking.into(), max_reasoning_tokens, closing_token_ids)
    }

    pub const fn mode(&self) -> ReasoningModeV1 {
        self.mode
    }

    pub const fn max_reasoning_tokens(&self) -> Option<u32> {
        self.max_reasoning_tokens
    }

    pub fn closing_token_ids(&self) -> &[u32] {
        &self.closing_token_ids
    }

    pub const fn is_enabled(&self) -> bool {
        self.mode.active()
    }

    /// Admission check for the combined output budget.  The closing sequence
    /// is counted even though it is not part of visible answer tokens.
    pub fn validate_max_new_tokens(&self, max_new_tokens: u32) -> Result<(), ReasoningErrorV1> {
        if self.is_enabled() {
            if let Some(budget) = self.max_reasoning_tokens {
                if u64::from(budget) + self.closing_token_ids.len() as u64
                    > u64::from(max_new_tokens)
                {
                    return Err(ReasoningErrorV1::MaxTokensTooSmall);
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReasoningPhase {
    Inactive,
    Reasoning,
    MatchingClose,
    ForcingClose,
    Answer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReasoningObservationV1 {
    visible: bool,
    entered_answer: bool,
    forced: bool,
}

impl ReasoningObservationV1 {
    pub const fn visible(self) -> bool {
        self.visible
    }

    pub const fn entered_answer(self) -> bool {
        self.entered_answer
    }

    pub const fn forced(self) -> bool {
        self.forced
    }
}

/// Mutable state owned by exactly one generation request.
#[derive(Clone, Debug)]
pub struct ReasoningControllerV1 {
    policy: ReasoningPolicyV1,
    phase: ReasoningPhase,
    close_progress: usize,
    close_buffer: Vec<u32>,
    reasoning_tokens: u32,
    visible_tokens: u32,
    generated_tokens: u32,
}

impl ReasoningControllerV1 {
    pub fn new(policy: ReasoningPolicyV1) -> Self {
        let phase = if policy.is_enabled() {
            ReasoningPhase::Reasoning
        } else {
            ReasoningPhase::Inactive
        };
        Self {
            policy,
            phase,
            close_progress: 0,
            close_buffer: Vec::new(),
            reasoning_tokens: 0,
            visible_tokens: 0,
            generated_tokens: 0,
        }
    }

    pub const fn is_enabled(&self) -> bool {
        self.policy.is_enabled()
    }

    pub const fn is_reasoning(&self) -> bool {
        matches!(
            self.phase,
            ReasoningPhase::Reasoning | ReasoningPhase::MatchingClose
        )
    }

    pub const fn is_answer(&self) -> bool {
        matches!(self.phase, ReasoningPhase::Answer)
    }

    pub const fn is_forcing_close(&self) -> bool {
        matches!(self.phase, ReasoningPhase::ForcingClose)
    }

    pub const fn reasoning_tokens(&self) -> u32 {
        self.reasoning_tokens
    }

    pub const fn visible_tokens(&self) -> u32 {
        self.visible_tokens
    }

    pub const fn generated_tokens(&self) -> u32 {
        self.generated_tokens
    }

    pub const fn close_progress(&self) -> usize {
        self.close_progress
    }

    pub fn policy(&self) -> &ReasoningPolicyV1 {
        &self.policy
    }

    /// Intersects the existing grammar/stop mask with the next forced close
    /// token.  When the host path has no prior mask, logits provide the
    /// vocabulary size for the new singleton mask.  An empty intersection is
    /// a hard error rather than a fallback to host-only token insertion.
    pub fn apply_mask(
        &self,
        base_mask: Option<&[bool]>,
        vocab_size: Option<usize>,
    ) -> Result<Option<Vec<bool>>, ReasoningErrorV1> {
        let Some(expected) = self.expected_forced_token() else {
            return Ok(base_mask.map(ToOwned::to_owned));
        };
        let size = base_mask
            .map(<[bool]>::len)
            .or(vocab_size)
            .ok_or(ReasoningErrorV1::CandidateMaskUnavailable)?;
        let expected_index =
            usize::try_from(expected).map_err(|_| ReasoningErrorV1::TokenOutsideVocabulary)?;
        if expected_index >= size {
            return Err(ReasoningErrorV1::TokenOutsideVocabulary);
        }
        if base_mask.is_some_and(|mask| !mask.get(expected_index).copied().unwrap_or(false)) {
            return Err(ReasoningErrorV1::CandidateMaskEmpty);
        }
        let mut mask = vec![false; size];
        mask[expected_index] = true;
        Ok(Some(mask))
    }

    /// Returns the next token required by a forced close, if any.
    pub fn expected_forced_token(&self) -> Option<u32> {
        if let ReasoningPhase::ForcingClose = self.phase {
            self.policy
                .closing_token_ids
                .get(self.close_progress)
                .copied()
        } else {
            None
        }
    }

    /// Records one selected token.  Closing marker candidates are held until
    /// the complete fixed sequence is observed, so a partial marker never
    /// leaks into visible output.  A mismatch is treated as ordinary hidden
    /// reasoning and is still charged to the bounded budget.
    pub fn observe(&mut self, token_id: u32) -> Result<ReasoningObservationV1, ReasoningErrorV1> {
        self.generated_tokens = self
            .generated_tokens
            .checked_add(1)
            .ok_or(ReasoningErrorV1::CountOverflow)?;
        if !self.is_enabled() {
            self.visible_tokens = self
                .visible_tokens
                .checked_add(1)
                .ok_or(ReasoningErrorV1::CountOverflow)?;
            return Ok(ReasoningObservationV1 {
                visible: true,
                entered_answer: false,
                forced: false,
            });
        }

        if let ReasoningPhase::Answer = self.phase {
            self.visible_tokens = self
                .visible_tokens
                .checked_add(1)
                .ok_or(ReasoningErrorV1::CountOverflow)?;
            return Ok(ReasoningObservationV1 {
                visible: true,
                entered_answer: false,
                forced: false,
            });
        }

        if let ReasoningPhase::ForcingClose = self.phase {
            let expected = self
                .expected_forced_token()
                .ok_or(ReasoningErrorV1::ForcedTokenMismatch)?;
            if token_id != expected {
                return Err(ReasoningErrorV1::ForcedTokenMismatch);
            }
            self.close_progress += 1;
            if self.close_progress == self.policy.closing_token_ids.len() {
                self.phase = ReasoningPhase::Answer;
                self.close_progress = 0;
                self.close_buffer.clear();
                return Ok(ReasoningObservationV1 {
                    visible: false,
                    entered_answer: true,
                    forced: true,
                });
            }
            return Ok(ReasoningObservationV1 {
                visible: false,
                entered_answer: false,
                forced: true,
            });
        }

        if matches!(
            self.phase,
            ReasoningPhase::Reasoning | ReasoningPhase::MatchingClose
        ) {
            let close = &self.policy.closing_token_ids;
            let expected = close.get(self.close_progress).copied();
            if expected == Some(token_id) {
                self.close_buffer.push(token_id);
                self.close_progress += 1;
                self.phase = ReasoningPhase::MatchingClose;
                if self.close_progress == close.len() {
                    self.phase = ReasoningPhase::Answer;
                    self.close_progress = 0;
                    self.close_buffer.clear();
                    return Ok(ReasoningObservationV1 {
                        visible: false,
                        entered_answer: true,
                        forced: false,
                    });
                }
                return Ok(ReasoningObservationV1 {
                    visible: false,
                    entered_answer: false,
                    forced: false,
                });
            }

            // A partial marker is only a marker candidate.  Once it differs,
            // account for all held IDs as hidden reasoning and then account
            // for the current token as well.
            let held = self.close_buffer.len();
            self.close_buffer.clear();
            self.close_progress = 0;
            self.phase = ReasoningPhase::Reasoning;
            let charge = held.checked_add(1).ok_or(ReasoningErrorV1::CountOverflow)?;
            self.charge_reasoning(charge)?;
            return Ok(ReasoningObservationV1 {
                visible: false,
                entered_answer: false,
                forced: false,
            });
        }

        self.charge_reasoning(1)?;
        Ok(ReasoningObservationV1 {
            visible: false,
            entered_answer: false,
            forced: false,
        })
    }

    fn charge_reasoning(&mut self, count: usize) -> Result<(), ReasoningErrorV1> {
        let count = u32::try_from(count).map_err(|_| ReasoningErrorV1::CountOverflow)?;
        self.reasoning_tokens = self
            .reasoning_tokens
            .checked_add(count)
            .ok_or(ReasoningErrorV1::CountOverflow)?;
        if let Some(budget) = self.policy.max_reasoning_tokens {
            if self.reasoning_tokens > budget {
                return Err(ReasoningErrorV1::BudgetExceeded);
            }
            if self.reasoning_tokens == budget {
                self.phase = ReasoningPhase::ForcingClose;
                self.close_progress = 0;
                self.close_buffer.clear();
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_mode_mapping_is_lossless() {
        for mode in [
            ThinkingModeV1::Disabled,
            ThinkingModeV1::Enabled,
            ThinkingModeV1::TemplateDefault,
        ] {
            assert_eq!(ThinkingModeV1::from(ReasoningModeV1::from(mode)), mode);
        }
    }

    #[test]
    fn boundaries_and_forced_close_are_bounded() {
        assert!(ReasoningPolicyV1::enabled(Some(0), [9]).is_err());
        assert!(ReasoningPolicyV1::enabled(Some(MAX_REASONING_TOKENS_V1 + 1), [9]).is_err());
        let policy = ReasoningPolicyV1::enabled(Some(3), [9, 10]).unwrap();
        assert!(policy.validate_max_new_tokens(4).is_err());
        assert!(policy.validate_max_new_tokens(5).is_ok());

        let mut controller = ReasoningControllerV1::new(policy);
        assert!(!controller.observe(1).unwrap().visible());
        assert!(!controller.observe(2).unwrap().visible());
        assert!(!controller.observe(3).unwrap().visible());
        assert_eq!(controller.expected_forced_token(), Some(9));
        let mask = controller
            .apply_mask(
                Some(&[
                    true, true, true, true, true, true, true, true, true, true, true,
                ]),
                Some(11),
            )
            .unwrap();
        assert_eq!(mask.unwrap().iter().filter(|value| **value).count(), 1);
        assert!(!controller.observe(9).unwrap().visible());
        assert!(controller.observe(10).unwrap().entered_answer());
        assert!(controller.observe(11).unwrap().visible());
        assert_eq!(controller.reasoning_tokens(), 3);
        assert_eq!(controller.visible_tokens(), 1);
        assert_eq!(controller.generated_tokens(), 6);
    }

    #[test]
    fn early_close_is_not_visible_and_partial_close_is_charged() {
        let policy = ReasoningPolicyV1::enabled(Some(4), [9, 10]).unwrap();
        let mut controller = ReasoningControllerV1::new(policy);
        assert!(!controller.observe(9).unwrap().visible());
        assert!(controller.observe(10).unwrap().entered_answer());
        assert!(controller.observe(7).unwrap().visible());
        assert_eq!(controller.reasoning_tokens(), 0);

        let policy = ReasoningPolicyV1::enabled(Some(3), [9, 10]).unwrap();
        let mut controller = ReasoningControllerV1::new(policy);
        assert!(!controller.observe(9).unwrap().visible());
        assert!(!controller.observe(8).unwrap().visible());
        assert_eq!(controller.reasoning_tokens(), 2);
        assert!(!controller.observe(7).unwrap().visible());
        assert_eq!(controller.reasoning_tokens(), 3);
        assert!(controller.is_forcing_close());
    }
}
