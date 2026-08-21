//! Model-neutral speculative decoding decisions and publication transaction.
//!
//! This module deliberately knows nothing about a model family or KV-cache
//! encoding. A provider stages work behind an opaque checkpoint and publishes
//! only the prefix returned by verification.

use crate::{SamplingError, SamplingRandomSource};
use std::fmt;

pub const MAX_SPECULATIVE_DRAFT_WIDTH_V1: usize = 8;
pub const MAX_NGRAM_ORDER_V1: usize = 16;
pub const MAX_SPECULATIVE_HISTORY_TOKENS_V1: usize = 1_048_576;

/// Stable provider identity used for accounting and compatibility checks.
/// Model-specific state remains behind the provider implementation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DraftProviderKindV1 {
    QwenMtp,
    ExternalModel,
    Ngram,
}

/// One bounded proposal. Proposals never imply publication: the target model
/// remains the sole owner of verification, sampling, and visible output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DraftProposalV1 {
    provider: DraftProviderKindV1,
    token_ids: Vec<u32>,
}

impl DraftProposalV1 {
    pub fn new(
        provider: DraftProviderKindV1,
        token_ids: Vec<u32>,
    ) -> Result<Self, SpeculativeError> {
        if token_ids.is_empty() {
            return Err(SpeculativeError::ZeroDraftWidth);
        }
        if token_ids.len() > MAX_SPECULATIVE_DRAFT_WIDTH_V1 {
            return Err(SpeculativeError::DraftWidthExceeded);
        }
        Ok(Self {
            provider,
            token_ids,
        })
    }

    pub const fn provider(&self) -> DraftProviderKindV1 {
        self.provider
    }

    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
}

/// Model-neutral proposal seam. Implementations own their private state and
/// RNG, but receive only already-committed target tokens.
pub trait DraftProviderV1 {
    fn kind(&self) -> DraftProviderKindV1;

    fn propose(
        &mut self,
        committed_target_tokens: &[u32],
        max_width: usize,
    ) -> Result<Option<DraftProposalV1>, SpeculativeError>;

    fn reset(&mut self) -> Result<(), SpeculativeError> {
        Ok(())
    }
}

/// External draft execution boundary. A concrete model adapter must keep its
/// KV/RNG/state separate from the target and return tokens in target-vocabulary
/// IDs only after [`validate_external_draft_compatibility_v1`] succeeds.
pub trait ExternalDraftModelV1 {
    fn model_fingerprint(&self) -> &str;
    fn tokenizer_fingerprint(&self) -> &str;
    fn vocabulary_size(&self) -> u32;
    fn reset_to_prefix(&mut self, committed_target_tokens: &[u32]) -> Result<(), SpeculativeError>;
    fn propose_next(&mut self, pending_token: Option<u32>) -> Result<u32, SpeculativeError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalDraftCompatibilityV1 {
    target_tokenizer_fingerprint: String,
    draft_tokenizer_fingerprint: String,
    target_vocabulary_size: u32,
    draft_vocabulary_size: u32,
}

impl ExternalDraftCompatibilityV1 {
    pub fn new(
        target_tokenizer_fingerprint: impl Into<String>,
        draft_tokenizer_fingerprint: impl Into<String>,
        target_vocabulary_size: u32,
        draft_vocabulary_size: u32,
    ) -> Result<Self, SpeculativeError> {
        let value = Self {
            target_tokenizer_fingerprint: target_tokenizer_fingerprint.into(),
            draft_tokenizer_fingerprint: draft_tokenizer_fingerprint.into(),
            target_vocabulary_size,
            draft_vocabulary_size,
        };
        validate_external_draft_compatibility_v1(&value)?;
        Ok(value)
    }

    pub fn target_tokenizer_fingerprint(&self) -> &str {
        &self.target_tokenizer_fingerprint
    }

    pub fn draft_tokenizer_fingerprint(&self) -> &str {
        &self.draft_tokenizer_fingerprint
    }

    pub const fn target_vocabulary_size(&self) -> u32 {
        self.target_vocabulary_size
    }

    pub const fn draft_vocabulary_size(&self) -> u32 {
        self.draft_vocabulary_size
    }
}

pub fn validate_external_draft_compatibility_v1(
    compatibility: &ExternalDraftCompatibilityV1,
) -> Result<(), SpeculativeError> {
    if compatibility.target_tokenizer_fingerprint.is_empty()
        || compatibility.target_tokenizer_fingerprint != compatibility.draft_tokenizer_fingerprint
        || compatibility.target_vocabulary_size == 0
        || compatibility.target_vocabulary_size != compatibility.draft_vocabulary_size
    {
        return Err(SpeculativeError::IncompatibleDraftModel);
    }
    Ok(())
}

pub struct ExternalDraftProviderV1<M> {
    model: M,
    compatibility: ExternalDraftCompatibilityV1,
}

impl<M: ExternalDraftModelV1> ExternalDraftProviderV1<M> {
    pub fn new(
        model: M,
        compatibility: ExternalDraftCompatibilityV1,
    ) -> Result<Self, SpeculativeError> {
        validate_external_draft_compatibility_v1(&compatibility)?;
        if model.tokenizer_fingerprint() != compatibility.draft_tokenizer_fingerprint()
            || model.vocabulary_size() != compatibility.draft_vocabulary_size()
            || model.model_fingerprint().is_empty()
        {
            return Err(SpeculativeError::IncompatibleDraftModel);
        }
        Ok(Self {
            model,
            compatibility,
        })
    }

    pub const fn model(&self) -> &M {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

impl<M: ExternalDraftModelV1> DraftProviderV1 for ExternalDraftProviderV1<M> {
    fn kind(&self) -> DraftProviderKindV1 {
        DraftProviderKindV1::ExternalModel
    }

    fn propose(
        &mut self,
        committed_target_tokens: &[u32],
        max_width: usize,
    ) -> Result<Option<DraftProposalV1>, SpeculativeError> {
        validate_draft_request(committed_target_tokens, max_width)?;
        validate_external_draft_compatibility_v1(&self.compatibility)?;
        self.model.reset_to_prefix(committed_target_tokens)?;
        let mut tokens = Vec::with_capacity(max_width);
        let mut pending = committed_target_tokens.last().copied();
        for _ in 0..max_width {
            let token = self.model.propose_next(pending)?;
            if token >= self.compatibility.target_vocabulary_size() {
                return Err(SpeculativeError::TokenOutOfVocabulary(token));
            }
            tokens.push(token);
            pending = Some(token);
        }
        Ok(Some(DraftProposalV1::new(self.kind(), tokens)?))
    }
}

/// Deterministic request-local n-gram proposer. It uses the longest suffix
/// with a prior continuation and resolves equal matches toward the most recent
/// occurrence. No sampler or proposal RNG is consumed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NgramDraftProviderV1 {
    min_order: usize,
    max_order: usize,
}

impl NgramDraftProviderV1 {
    pub fn new(min_order: usize, max_order: usize) -> Result<Self, SpeculativeError> {
        if min_order == 0 || min_order > max_order || max_order > MAX_NGRAM_ORDER_V1 {
            return Err(SpeculativeError::InvalidNgramOrder);
        }
        Ok(Self {
            min_order,
            max_order,
        })
    }

    pub const fn min_order(&self) -> usize {
        self.min_order
    }

    pub const fn max_order(&self) -> usize {
        self.max_order
    }
}

impl DraftProviderV1 for NgramDraftProviderV1 {
    fn kind(&self) -> DraftProviderKindV1 {
        DraftProviderKindV1::Ngram
    }

    fn propose(
        &mut self,
        committed_target_tokens: &[u32],
        max_width: usize,
    ) -> Result<Option<DraftProposalV1>, SpeculativeError> {
        validate_draft_request(committed_target_tokens, max_width)?;
        let history_len = committed_target_tokens.len();
        let maximum_order = self.max_order.min(history_len.saturating_sub(1));
        for order in (self.min_order..=maximum_order).rev() {
            let suffix_start = history_len - order;
            let suffix = &committed_target_tokens[suffix_start..];
            let latest_start = suffix_start.saturating_sub(1);
            for candidate_start in (0..=latest_start).rev() {
                let candidate_end = candidate_start + order;
                if candidate_end >= suffix_start
                    || &committed_target_tokens[candidate_start..candidate_end] != suffix
                {
                    continue;
                }
                let proposal_end = candidate_end.saturating_add(max_width).min(history_len);
                let tokens = committed_target_tokens[candidate_end..proposal_end].to_vec();
                if !tokens.is_empty() {
                    return Ok(Some(DraftProposalV1::new(self.kind(), tokens)?));
                }
            }
        }
        Ok(None)
    }
}

fn validate_draft_request(history: &[u32], max_width: usize) -> Result<(), SpeculativeError> {
    if max_width == 0 {
        return Err(SpeculativeError::ZeroDraftWidth);
    }
    if max_width > MAX_SPECULATIVE_DRAFT_WIDTH_V1 {
        return Err(SpeculativeError::DraftWidthExceeded);
    }
    if history.len() > MAX_SPECULATIVE_HISTORY_TOKENS_V1 {
        return Err(SpeculativeError::HistoryLimitExceeded);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SpeculativeAccountingV1 {
    proposal_blocks: u64,
    proposed_tokens: u64,
    accepted_tokens: u64,
    rejected_tokens: u64,
    emitted_target_tokens: u64,
}

impl SpeculativeAccountingV1 {
    pub fn record(
        &mut self,
        proposal: &DraftProposalV1,
        decision: &SpeculativeDecision,
    ) -> Result<(), SpeculativeError> {
        let proposed = u64::try_from(proposal.token_ids.len())
            .map_err(|_| SpeculativeError::AccountingOverflow)?;
        let accepted = u64::try_from(decision.accepted_draft_tokens)
            .map_err(|_| SpeculativeError::AccountingOverflow)?;
        let emitted = u64::try_from(decision.emitted_tokens.len())
            .map_err(|_| SpeculativeError::AccountingOverflow)?;
        if accepted > proposed
            || decision.accepted_draft_tokens > proposal.token_ids.len()
            || decision.emitted_tokens.is_empty()
            || decision.emitted_tokens.len() > proposal.token_ids.len() + 1
        {
            return Err(SpeculativeError::InvalidDecision);
        }
        self.proposal_blocks = self
            .proposal_blocks
            .checked_add(1)
            .ok_or(SpeculativeError::AccountingOverflow)?;
        self.proposed_tokens = self
            .proposed_tokens
            .checked_add(proposed)
            .ok_or(SpeculativeError::AccountingOverflow)?;
        self.accepted_tokens = self
            .accepted_tokens
            .checked_add(accepted)
            .ok_or(SpeculativeError::AccountingOverflow)?;
        self.rejected_tokens = self
            .rejected_tokens
            .checked_add(proposed - accepted)
            .ok_or(SpeculativeError::AccountingOverflow)?;
        self.emitted_target_tokens = self
            .emitted_target_tokens
            .checked_add(emitted)
            .ok_or(SpeculativeError::AccountingOverflow)?;
        Ok(())
    }

    pub const fn proposal_blocks(self) -> u64 {
        self.proposal_blocks
    }

    pub const fn proposed_tokens(self) -> u64 {
        self.proposed_tokens
    }

    pub const fn accepted_tokens(self) -> u64 {
        self.accepted_tokens
    }

    pub const fn rejected_tokens(self) -> u64 {
        self.rejected_tokens
    }

    pub const fn emitted_target_tokens(self) -> u64 {
        self.emitted_target_tokens
    }
}

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
        if draft_width > MAX_SPECULATIVE_DRAFT_WIDTH_V1 {
            return Err(SpeculativeError::DraftWidthExceeded);
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
        let maximum_emitted = self
            .draft_width
            .checked_add(1)
            .ok_or(SpeculativeError::DraftWidthExceeded)?;
        if decision.accepted_draft_tokens > self.draft_width
            || decision.emitted_tokens.is_empty()
            || decision.emitted_tokens.len() > maximum_emitted
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
    let output_width = draft_tokens
        .len()
        .checked_add(1)
        .ok_or(SpeculativeError::DraftWidthExceeded)?;
    let mut emitted_tokens = Vec::with_capacity(output_width);
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
    let output_width = drafts
        .len()
        .checked_add(1)
        .ok_or(SpeculativeError::DraftWidthExceeded)?;
    let mut emitted_tokens = Vec::with_capacity(output_width);
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
    if draft_width > MAX_SPECULATIVE_DRAFT_WIDTH_V1 {
        return Err(SpeculativeError::DraftWidthExceeded);
    }
    let expected_target_width = draft_width
        .checked_add(1)
        .ok_or(SpeculativeError::DraftWidthExceeded)?;
    if target_width != expected_target_width {
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

#[derive(Clone, Debug, Eq, PartialEq)]
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
    DraftWidthExceeded,
    InvalidNgramOrder,
    HistoryLimitExceeded,
    IncompatibleDraftModel,
    AccountingOverflow,
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
            Self::DraftWidthExceeded => {
                formatter.write_str("draft width exceeds the bounded limit")
            }
            Self::InvalidNgramOrder => {
                formatter.write_str("n-gram order is outside the bounded range")
            }
            Self::HistoryLimitExceeded => {
                formatter.write_str("speculative history exceeds the bounded limit")
            }
            Self::IncompatibleDraftModel => formatter.write_str(
                "external draft model tokenizer or vocabulary is incompatible with the target",
            ),
            Self::AccountingOverflow => formatter.write_str("speculative accounting overflowed"),
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
    fn ngram_uses_longest_then_latest_match_without_rng() {
        let mut provider = NgramDraftProviderV1::new(1, 4).unwrap();
        let history = [8, 1, 2, 3, 7, 1, 2, 3, 9, 1, 2, 3];
        let proposal = provider.propose(&history, 3).unwrap().unwrap();
        assert_eq!(proposal.provider(), DraftProviderKindV1::Ngram);
        assert_eq!(proposal.token_ids(), [9, 1, 2]);

        assert_eq!(provider.propose(&[1], 1).unwrap(), None);
        assert_eq!(
            provider.propose(&history, 0),
            Err(SpeculativeError::ZeroDraftWidth)
        );
        assert_eq!(
            provider.propose(&history, MAX_SPECULATIVE_DRAFT_WIDTH_V1 + 1),
            Err(SpeculativeError::DraftWidthExceeded)
        );
    }

    struct FixedExternalDraft {
        model: String,
        tokenizer: String,
        vocabulary: u32,
        proposals: VecDeque<u32>,
        reset_prefixes: Vec<Vec<u32>>,
    }

    impl ExternalDraftModelV1 for FixedExternalDraft {
        fn model_fingerprint(&self) -> &str {
            &self.model
        }

        fn tokenizer_fingerprint(&self) -> &str {
            &self.tokenizer
        }

        fn vocabulary_size(&self) -> u32 {
            self.vocabulary
        }

        fn reset_to_prefix(
            &mut self,
            committed_target_tokens: &[u32],
        ) -> Result<(), SpeculativeError> {
            self.reset_prefixes.push(committed_target_tokens.to_vec());
            Ok(())
        }

        fn propose_next(&mut self, _: Option<u32>) -> Result<u32, SpeculativeError> {
            self.proposals
                .pop_front()
                .ok_or(SpeculativeError::InvalidDecision)
        }
    }

    #[test]
    fn external_draft_requires_exact_tokenizer_and_vocabulary_identity() {
        let compatibility = ExternalDraftCompatibilityV1::new("tok-a", "tok-a", 17, 17)
            .expect("matching tokenizer contract");
        let model = FixedExternalDraft {
            model: "sha256:model".to_owned(),
            tokenizer: "tok-a".to_owned(),
            vocabulary: 17,
            proposals: VecDeque::from([4, 5, 6]),
            reset_prefixes: Vec::new(),
        };
        let mut provider = ExternalDraftProviderV1::new(model, compatibility).unwrap();
        let proposal = provider.propose(&[1, 2, 3], 3).unwrap().unwrap();
        assert_eq!(proposal.token_ids(), [4, 5, 6]);
        assert_eq!(provider.model().reset_prefixes, [vec![1, 2, 3]]);

        assert_eq!(
            ExternalDraftCompatibilityV1::new("tok-a", "tok-b", 17, 17),
            Err(SpeculativeError::IncompatibleDraftModel)
        );
        assert_eq!(
            ExternalDraftCompatibilityV1::new("tok-a", "tok-a", 17, 18),
            Err(SpeculativeError::IncompatibleDraftModel)
        );
    }

    #[test]
    fn accounting_separates_proposed_accepted_rejected_and_emitted() {
        let proposal = DraftProposalV1::new(DraftProviderKindV1::Ngram, vec![1, 2, 3])
            .expect("bounded proposal");
        let partial = verify_target_selected(proposal.token_ids(), &[1, 9, 3, 4]).unwrap();
        let mut accounting = SpeculativeAccountingV1::default();
        accounting.record(&proposal, &partial).unwrap();
        assert_eq!(accounting.proposal_blocks(), 1);
        assert_eq!(accounting.proposed_tokens(), 3);
        assert_eq!(accounting.accepted_tokens(), 1);
        assert_eq!(accounting.rejected_tokens(), 2);
        assert_eq!(accounting.emitted_target_tokens(), 2);
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
        let bounded_checkpoint = OpaqueStateCheckpoint {
            generation: 17,
            committed_tokens: 257,
        };
        assert_eq!(
            SpeculativeTransaction::new(bounded_checkpoint, MAX_SPECULATIVE_DRAFT_WIDTH_V1 + 1,),
            Err(SpeculativeError::DraftWidthExceeded)
        );
        assert_eq!(
            verify_greedy(
                &[1; MAX_SPECULATIVE_DRAFT_WIDTH_V1 + 1],
                &[1; MAX_SPECULATIVE_DRAFT_WIDTH_V1 + 2],
            ),
            Err(SpeculativeError::DraftWidthExceeded)
        );
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
