//! Model-neutral speculative decoding decisions and publication transaction.
//!
//! This module deliberately knows nothing about a model family or KV-cache
//! encoding. A provider stages work behind an opaque checkpoint and publishes
//! only the prefix returned by verification.

use crate::{SamplingError, SamplingRandomSource};
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub struct TokenDistribution {
    probabilities: Vec<f64>,
}

impl TokenDistribution {
    pub fn new(probabilities: Vec<f64>) -> Result<Self, SpeculativeError> {
        if probabilities.is_empty() {
            return Err(SpeculativeError::EmptyDistribution);
        }
        if probabilities.len() > u32::MAX as usize {
            return Err(SpeculativeError::VocabularyOverflow);
        }
        let mut total = 0.0_f64;
        for &value in &probabilities {
            if !value.is_finite() || value < 0.0 {
                return Err(SpeculativeError::InvalidProbability);
            }
            total += value;
        }
        if !total.is_finite() || total <= 0.0 {
            return Err(SpeculativeError::EmptyDistribution);
        }
        Ok(Self {
            probabilities: probabilities
                .into_iter()
                .map(|value| value / total)
                .collect(),
        })
    }

    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    pub fn probability(&self, token: u32) -> Result<f64, SpeculativeError> {
        self.probabilities
            .get(token as usize)
            .copied()
            .ok_or(SpeculativeError::TokenOutOfVocabulary(token))
    }

    pub fn argmax(&self) -> u32 {
        self.probabilities
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index as u32)
            .expect("validated distributions are non-empty")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DraftToken {
    pub token_id: u32,
    pub distribution: TokenDistribution,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpeculativeDecision {
    accepted_draft_tokens: usize,
    emitted_tokens: Vec<u32>,
    rejected_at: Option<usize>,
    random_draws: usize,
}

impl SpeculativeDecision {
    pub const fn accepted_draft_tokens(&self) -> usize {
        self.accepted_draft_tokens
    }

    pub fn emitted_tokens(&self) -> &[u32] {
        &self.emitted_tokens
    }

    pub const fn rejected_at(&self) -> Option<usize> {
        self.rejected_at
    }

    pub const fn random_draws(&self) -> usize {
        self.random_draws
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpaqueStateCheckpoint {
    pub generation: u64,
    pub committed_tokens: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpeculativeTransaction {
    checkpoint: OpaqueStateCheckpoint,
    draft_width: usize,
    state: TransactionState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TransactionState {
    Staging,
    Verified { accepted: usize },
    Committed,
    Aborted,
}

impl SpeculativeTransaction {
    pub fn new(
        checkpoint: OpaqueStateCheckpoint,
        draft_width: usize,
    ) -> Result<Self, SpeculativeError> {
        if checkpoint.generation == 0 {
            return Err(SpeculativeError::StaleGeneration);
        }
        if draft_width == 0 {
            return Err(SpeculativeError::ZeroDraftWidth);
        }
        Ok(Self {
            checkpoint,
            draft_width,
            state: TransactionState::Staging,
        })
    }

    pub const fn checkpoint(&self) -> OpaqueStateCheckpoint {
        self.checkpoint
    }

    pub fn verify(&mut self, decision: &SpeculativeDecision) -> Result<(), SpeculativeError> {
        if self.state != TransactionState::Staging {
            return Err(SpeculativeError::TransactionNotStaging);
        }
        if decision.accepted_draft_tokens > self.draft_width
            || decision.emitted_tokens.is_empty()
            || decision.emitted_tokens.len() > self.draft_width + 1
        {
            return Err(SpeculativeError::InvalidDecision);
        }
        self.state = TransactionState::Verified {
            accepted: decision.accepted_draft_tokens,
        };
        Ok(())
    }

    pub fn commit(&mut self, current: OpaqueStateCheckpoint) -> Result<usize, SpeculativeError> {
        if current != self.checkpoint {
            self.state = TransactionState::Aborted;
            return Err(SpeculativeError::StaleGeneration);
        }
        let TransactionState::Verified { accepted } = self.state else {
            return Err(SpeculativeError::TransactionNotVerified);
        };
        self.state = TransactionState::Committed;
        Ok(accepted)
    }

    pub fn abort(&mut self) {
        if self.state != TransactionState::Committed {
            self.state = TransactionState::Aborted;
        }
    }

    pub const fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            TransactionState::Committed | TransactionState::Aborted
        )
    }
}

pub fn verify_greedy(
    draft_tokens: &[u32],
    target_argmax: &[u32],
) -> Result<SpeculativeDecision, SpeculativeError> {
    validate_widths(draft_tokens.len(), target_argmax.len())?;
    let mut emitted_tokens = Vec::with_capacity(draft_tokens.len() + 1);
    for (index, (&draft, &target)) in draft_tokens.iter().zip(target_argmax).enumerate() {
        if draft != target {
            emitted_tokens.push(target);
            return Ok(SpeculativeDecision {
                accepted_draft_tokens: index,
                emitted_tokens,
                rejected_at: Some(index),
                random_draws: 0,
            });
        }
        emitted_tokens.push(draft);
    }
    emitted_tokens.push(target_argmax[draft_tokens.len()]);
    Ok(SpeculativeDecision {
        accepted_draft_tokens: draft_tokens.len(),
        emitted_tokens,
        rejected_at: None,
        random_draws: 0,
    })
}

/// Sequentially accepts proposals against tokens already selected by the
/// canonical target sampler. This is the production exact path for both
/// greedy and stochastic generation: it consumes no RNG and never applies a
/// residual/rejection distribution.
pub fn verify_target_selected(
    draft_tokens: &[u32],
    target_selected: &[u32],
) -> Result<SpeculativeDecision, SpeculativeError> {
    verify_greedy(draft_tokens, target_selected)
}

pub fn verify_stochastic(
    drafts: &[DraftToken],
    target: &[TokenDistribution],
    random: &mut impl SamplingRandomSource,
) -> Result<SpeculativeDecision, SpeculativeError> {
    validate_widths(drafts.len(), target.len())?;
    let vocabulary = target[0].probabilities.len();
    if target
        .iter()
        .any(|entry| entry.probabilities.len() != vocabulary)
        || drafts
            .iter()
            .any(|entry| entry.distribution.probabilities.len() != vocabulary)
    {
        return Err(SpeculativeError::VocabularyMismatch);
    }
    let mut emitted_tokens = Vec::with_capacity(drafts.len() + 1);
    let mut random_draws = 0_usize;
    for (index, draft) in drafts.iter().enumerate() {
        let p = target[index].probability(draft.token_id)?;
        let q = draft.distribution.probability(draft.token_id)?;
        if q <= 0.0 {
            return Err(SpeculativeError::ImpossibleDraftToken(draft.token_id));
        }
        let threshold = (p / q).min(1.0);
        let draw = next_draw(random)?;
        random_draws += 1;
        if draw < threshold {
            emitted_tokens.push(draft.token_id);
            continue;
        }

        let residual = target[index]
            .probabilities
            .iter()
            .zip(&draft.distribution.probabilities)
            .map(|(&target_probability, &draft_probability)| {
                (target_probability - draft_probability).max(0.0)
            })
            .collect::<Vec<_>>();
        let residual = TokenDistribution::new(residual)
            .map_err(|_| SpeculativeError::EmptyResidualDistribution)?;
        let replacement = sample_distribution(&residual, random)?;
        random_draws += 1;
        emitted_tokens.push(replacement);
        return Ok(SpeculativeDecision {
            accepted_draft_tokens: index,
            emitted_tokens,
            rejected_at: Some(index),
            random_draws,
        });
    }

    emitted_tokens.push(sample_distribution(&target[drafts.len()], random)?);
    random_draws += 1;
    Ok(SpeculativeDecision {
        accepted_draft_tokens: drafts.len(),
        emitted_tokens,
        rejected_at: None,
        random_draws,
    })
}

fn validate_widths(draft_width: usize, target_width: usize) -> Result<(), SpeculativeError> {
    if draft_width == 0 {
        return Err(SpeculativeError::ZeroDraftWidth);
    }
    if target_width != draft_width + 1 {
        return Err(SpeculativeError::TargetWidthMismatch {
            draft_width,
            target_width,
        });
    }
    Ok(())
}

fn next_draw(random: &mut impl SamplingRandomSource) -> Result<f64, SpeculativeError> {
    let value = random.next_unit_f64().map_err(SpeculativeError::Sampling)?;
    if !value.is_finite() || !(0.0..1.0).contains(&value) {
        return Err(SpeculativeError::InvalidRandomDraw);
    }
    Ok(value)
}

fn sample_distribution(
    distribution: &TokenDistribution,
    random: &mut impl SamplingRandomSource,
) -> Result<u32, SpeculativeError> {
    let draw = next_draw(random)?;
    let mut cumulative = 0.0_f64;
    for (index, probability) in distribution.probabilities.iter().enumerate() {
        cumulative += probability;
        if draw < cumulative || index + 1 == distribution.probabilities.len() {
            return Ok(index as u32);
        }
    }
    unreachable!("validated distribution always selects a token")
}

#[derive(Debug)]
pub enum SpeculativeError {
    EmptyDistribution,
    InvalidProbability,
    VocabularyOverflow,
    VocabularyMismatch,
    TokenOutOfVocabulary(u32),
    ImpossibleDraftToken(u32),
    EmptyResidualDistribution,
    InvalidRandomDraw,
    ZeroDraftWidth,
    TargetWidthMismatch {
        draft_width: usize,
        target_width: usize,
    },
    InvalidDecision,
    TransactionNotStaging,
    TransactionNotVerified,
    StaleGeneration,
    Sampling(SamplingError),
}

impl fmt::Display for SpeculativeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDistribution => formatter.write_str("probability distribution is empty"),
            Self::InvalidProbability => {
                formatter.write_str("probability distribution contains an invalid value")
            }
            Self::VocabularyOverflow => formatter.write_str("vocabulary does not fit u32"),
            Self::VocabularyMismatch => {
                formatter.write_str("draft and target vocabulary sizes differ")
            }
            Self::TokenOutOfVocabulary(token) => {
                write!(formatter, "token {token} is out of vocabulary")
            }
            Self::ImpossibleDraftToken(token) => {
                write!(formatter, "draft token {token} has zero draft probability")
            }
            Self::EmptyResidualDistribution => {
                formatter.write_str("rejected proposal has no residual target mass")
            }
            Self::InvalidRandomDraw => formatter.write_str("random draw is outside [0, 1)"),
            Self::ZeroDraftWidth => formatter.write_str("draft width must be non-zero"),
            Self::TargetWidthMismatch {
                draft_width,
                target_width,
            } => write!(
                formatter,
                "target width {target_width} must equal draft width {draft_width} plus one"
            ),
            Self::InvalidDecision => formatter.write_str("speculative decision is inconsistent"),
            Self::TransactionNotStaging => {
                formatter.write_str("speculative transaction is not staging")
            }
            Self::TransactionNotVerified => {
                formatter.write_str("speculative transaction is not verified")
            }
            Self::StaleGeneration => formatter.write_str("speculative checkpoint is stale"),
            Self::Sampling(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SpeculativeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    struct FixedRandom(VecDeque<f64>);

    impl SamplingRandomSource for FixedRandom {
        fn next_unit_f64(&mut self) -> Result<f64, SamplingError> {
            self.0
                .pop_front()
                .ok_or(SamplingError::RandomSourceUnavailable)
        }
    }

    fn distribution(values: &[f64]) -> TokenDistribution {
        TokenDistribution::new(values.to_vec()).unwrap()
    }

    #[test]
    fn greedy_covers_all_accept_first_and_mid_reject() {
        let all = verify_greedy(&[1, 2, 3], &[1, 2, 3, 4]).unwrap();
        assert_eq!(all.accepted_draft_tokens(), 3);
        assert_eq!(all.emitted_tokens(), [1, 2, 3, 4]);
        assert_eq!(all.rejected_at(), None);

        let first = verify_greedy(&[1, 2, 3], &[9, 2, 3, 4]).unwrap();
        assert_eq!(first.accepted_draft_tokens(), 0);
        assert_eq!(first.emitted_tokens(), [9]);
        assert_eq!(first.rejected_at(), Some(0));

        let mid = verify_greedy(&[1, 2, 3], &[1, 8, 3, 4]).unwrap();
        assert_eq!(mid.accepted_draft_tokens(), 1);
        assert_eq!(mid.emitted_tokens(), [1, 8]);
        assert_eq!(mid.rejected_at(), Some(1));
    }

    #[test]
    fn target_selected_verification_is_rng_free_for_sampled_tokens() {
        let decision = verify_target_selected(&[4, 5, 6], &[4, 9, 6, 7]).unwrap();
        assert_eq!(decision.accepted_draft_tokens(), 1);
        assert_eq!(decision.emitted_tokens(), [4, 9]);
        assert_eq!(decision.random_draws(), 0);
    }

    #[test]
    fn stochastic_acceptance_and_residual_follow_fixed_rng_order() {
        let drafts = [
            DraftToken {
                token_id: 0,
                distribution: distribution(&[0.5, 0.5, 0.0]),
            },
            DraftToken {
                token_id: 1,
                distribution: distribution(&[0.1, 0.8, 0.1]),
            },
        ];
        let target = [
            distribution(&[0.5, 0.4, 0.1]),
            distribution(&[0.7, 0.1, 0.2]),
            distribution(&[0.0, 0.0, 1.0]),
        ];
        let mut random = FixedRandom(VecDeque::from([0.99, 0.5, 0.9]));
        let decision = verify_stochastic(&drafts, &target, &mut random).unwrap();
        assert_eq!(decision.accepted_draft_tokens(), 1);
        assert_eq!(decision.emitted_tokens(), [0, 2]);
        assert_eq!(decision.rejected_at(), Some(1));
        assert_eq!(decision.random_draws(), 3);
    }

    #[test]
    fn stochastic_all_accept_samples_bonus_and_normalizes_inputs() {
        let drafts = [DraftToken {
            token_id: 1,
            distribution: distribution(&[1.0, 3.0]),
        }];
        let target = [distribution(&[0.0, 1.0]), distribution(&[0.25, 0.75])];
        let mut random = FixedRandom(VecDeque::from([0.5, 0.2]));
        let decision = verify_stochastic(&drafts, &target, &mut random).unwrap();
        assert_eq!(decision.emitted_tokens(), [1, 0]);
        assert_eq!(decision.random_draws(), 2);
    }

    #[test]
    fn transaction_publishes_only_verified_prefix_and_rejects_stale_state() {
        for draft_width in [1, 2, 3, 7] {
            let checkpoint = OpaqueStateCheckpoint {
                generation: 17,
                committed_tokens: 257,
            };
            let mut transaction = SpeculativeTransaction::new(checkpoint, draft_width).unwrap();
            let decision = SpeculativeDecision {
                accepted_draft_tokens: draft_width - 1,
                emitted_tokens: vec![3; draft_width],
                rejected_at: Some(draft_width - 1),
                random_draws: 0,
            };
            transaction.verify(&decision).unwrap();
            assert_eq!(transaction.commit(checkpoint).unwrap(), draft_width - 1);
            assert!(transaction.is_terminal());
        }

        let checkpoint = OpaqueStateCheckpoint {
            generation: 1,
            committed_tokens: 255,
        };
        let mut transaction = SpeculativeTransaction::new(checkpoint, 1).unwrap();
        transaction
            .verify(&verify_greedy(&[1], &[1, 2]).unwrap())
            .unwrap();
        assert!(matches!(
            transaction.commit(OpaqueStateCheckpoint {
                generation: 2,
                committed_tokens: 255,
            }),
            Err(SpeculativeError::StaleGeneration)
        ));
        assert!(transaction.is_terminal());
    }

    #[test]
    fn invalid_boundaries_fail_closed() {
        assert!(matches!(
            verify_greedy(&[], &[1]),
            Err(SpeculativeError::ZeroDraftWidth)
        ));
        assert!(matches!(
            verify_greedy(&[1, 2], &[1, 2]),
            Err(SpeculativeError::TargetWidthMismatch { .. })
        ));
        assert!(TokenDistribution::new(vec![f64::NAN]).is_err());
        assert!(TokenDistribution::new(vec![0.0, 0.0]).is_err());
    }
}
