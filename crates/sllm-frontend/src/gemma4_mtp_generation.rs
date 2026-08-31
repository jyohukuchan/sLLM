//! Gemma 4 target + MTP assistant generation owner.
//!
//! This module is intentionally a small adapter around the model-neutral
//! speculative transaction in [`crate::generation`].  The target remains the
//! only source of visible tokens: the assistant can propose one token, but a
//! target block verifies it before the adapter publishes anything.  In
//! particular, this owner never appends assistant K/V to the target request.
//!
//! The initial production contract is deliberately narrow.  Sampling, draft
//! widths greater than one, and prefix-cache construction are not silently
//! approximated; callers receive a fail-closed error instead.

use sllm_core::{
    DraftProposalV1, DraftProviderKindV1, DraftProviderV1, Gemma4ExecutionOutput,
    Gemma4ExecutionRequest, Gemma4MtpExecutionRequest, SpeculativeError,
};

use crate::generation::{
    GenerationExecutorV1, GenerationServiceError, GenerationStepV1, SpeculativeGenerationExecutorV1,
};

/// Target hidden width consumed by the Gemma 4 assistant pre-projection.
///
/// The assistant's `[3_840 + 3_840] -> 1_024` input is formed in the core
/// execution graph.  The frontend therefore transports exactly one normalized
/// target row and does not attempt to reproduce that projection here.
pub const GEMMA4_MTP_TARGET_HIDDEN_WIDTH: usize = 3_840;

/// Gemma 4 12B MTP's first production executor.
///
/// `target` owns the canonical Gemma request and its target KV state.
/// `assistant` owns the separately resident assistant weights and stateless
/// proposal workspace. Proposal execution borrows an owner-bound read-only
/// target-KV lease; verification is always performed by
/// `target.decode_block_with_mtp_state` and publication is performed by the
/// model-neutral `SpeculativeGenerationAdapterV1`.
pub struct Gemma4MtpGenerationExecutorV1 {
    target: Gemma4ExecutionRequest,
    assistant: Gemma4MtpExecutionRequest,
    last_target_hidden_bf16: Vec<u16>,
    pending_speculative_block: Option<PendingGemma4MtpBlockV1>,
}

struct PendingGemma4MtpBlockV1 {
    target_hidden_rows_bf16: Vec<u16>,
    target_input_rows: usize,
}

impl Gemma4MtpGenerationExecutorV1 {
    /// Gemma MTP currently has exactly one assistant proposal per target
    /// transition.  Keeping the constant public makes the boundary explicit
    /// to lifecycle and transport adapters without introducing a user flag.
    pub const MAX_DRAFT_WIDTH: usize = 1;

    pub fn new(target: Gemma4ExecutionRequest, assistant: Gemma4MtpExecutionRequest) -> Self {
        Self {
            target,
            assistant,
            last_target_hidden_bf16: Vec::new(),
            pending_speculative_block: None,
        }
    }

    /// Constructing a width other than the reviewed width-one route is an
    /// explicit error.  Do not clamp a requested width: doing so would make a
    /// caller believe that a different speculative contract was active.
    pub fn new_with_draft_width(
        target: Gemma4ExecutionRequest,
        assistant: Gemma4MtpExecutionRequest,
        draft_width: usize,
    ) -> Result<Self, GenerationServiceError> {
        Self::validate_draft_width(draft_width)?;
        Ok(Self::new(target, assistant))
    }

    fn validate_draft_width(draft_width: usize) -> Result<(), GenerationServiceError> {
        if draft_width != Self::MAX_DRAFT_WIDTH {
            return Err(GenerationServiceError::Execution(
                "Gemma 4 MTP supports draft width 1 only".to_owned(),
            ));
        }
        Ok(())
    }

    /// Number of target rows needed for one proposal.  `None` is used by
    /// admission/tests to reject width zero and future width expansion until
    /// its transaction semantics are implemented.
    pub const fn target_block_rows_for_draft_width(draft_width: usize) -> Option<usize> {
        if draft_width == Self::MAX_DRAFT_WIDTH {
            Some(2)
        } else {
            None
        }
    }

    pub const fn draft_width(&self) -> usize {
        Self::MAX_DRAFT_WIDTH
    }

    pub fn target(&self) -> &Gemma4ExecutionRequest {
        &self.target
    }

    pub fn assistant(&self) -> &Gemma4MtpExecutionRequest {
        &self.assistant
    }

    fn reject_sampling(include_last_logits: bool) -> Result<(), GenerationServiceError> {
        if include_last_logits {
            return Err(GenerationServiceError::Execution(
                "Gemma 4 MTP supports greedy generation only".to_owned(),
            ));
        }
        Ok(())
    }

    fn step_from_output(
        output: &Gemma4ExecutionOutput,
        row: usize,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        let token = *output
            .token_ids()
            .get(row)
            .ok_or(GenerationServiceError::MissingDeviceArgmax)?;
        Ok(GenerationStepV1::new(
            u32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow)?,
            None,
        ))
    }

    fn target_hidden_rows(
        output: &Gemma4ExecutionOutput,
        expected_rows: usize,
    ) -> Result<&[u16], GenerationServiceError> {
        let hidden = output.final_hidden_states_bf16().ok_or_else(|| {
            GenerationServiceError::Execution(
                "Gemma 4 target MTP route omitted normalized hidden rows".to_owned(),
            )
        })?;
        let expected_words = expected_rows
            .checked_mul(GEMMA4_MTP_TARGET_HIDDEN_WIDTH)
            .ok_or(GenerationServiceError::CountOverflow)?;
        if hidden.len() != expected_words {
            return Err(GenerationServiceError::Execution(format!(
                "Gemma 4 target hidden row count differs: expected {expected_words}, got {}",
                hidden.len()
            )));
        }
        Ok(hidden)
    }

    fn target_terminal_hidden(
        output: &Gemma4ExecutionOutput,
        row_count: usize,
    ) -> Result<Vec<u16>, GenerationServiceError> {
        let hidden = Self::target_hidden_rows(output, row_count)?;
        let start = (row_count - 1)
            .checked_mul(GEMMA4_MTP_TARGET_HIDDEN_WIDTH)
            .ok_or(GenerationServiceError::CountOverflow)?;
        Ok(hidden[start..start + GEMMA4_MTP_TARGET_HIDDEN_WIDTH].to_vec())
    }

    fn propose_mtp_draft(
        &mut self,
        pending_token: u32,
    ) -> Result<Option<DraftProposalV1>, SpeculativeError> {
        if self.last_target_hidden_bf16.len() != GEMMA4_MTP_TARGET_HIDDEN_WIDTH {
            return Err(SpeculativeError::HistoryLimitExceeded);
        }
        let token = i32::try_from(pending_token)
            .map_err(|_| SpeculativeError::TokenOutOfVocabulary(pending_token))?;
        let lease = self
            .target
            .mtp_target_kv_lease()
            .map_err(|_| SpeculativeError::InvalidDecision)?;
        let output = self
            .assistant
            .propose(token, &self.last_target_hidden_bf16, &lease)
            .map_err(|_| SpeculativeError::InvalidDecision)?;
        let proposal_token = output.token_id();
        Ok(Some(DraftProposalV1::new(
            DraftProviderKindV1::Gemma4Mtp,
            vec![
                u32::try_from(proposal_token)
                    .map_err(|_| SpeculativeError::TokenOutOfVocabulary(proposal_token as u32))?,
            ],
        )?))
    }

    fn verify_mtp_draft(
        &mut self,
        pending_token: u32,
        proposal: &DraftProposalV1,
    ) -> Result<Vec<GenerationStepV1>, GenerationServiceError> {
        if self.pending_speculative_block.is_some() {
            return Err(GenerationServiceError::Execution(
                "previous Gemma MTP target block is still pending".to_owned(),
            ));
        }
        if proposal.provider() != DraftProviderKindV1::Gemma4Mtp {
            return Err(GenerationServiceError::Speculative(
                "Gemma 4 MTP cannot verify a foreign draft provider".to_owned(),
            ));
        }
        if proposal.token_ids().len() != Self::MAX_DRAFT_WIDTH {
            return Err(GenerationServiceError::Speculative(
                "Gemma 4 MTP accepts draft width 1 only".to_owned(),
            ));
        }
        let pending =
            i32::try_from(pending_token).map_err(|_| GenerationServiceError::TokenIdOverflow)?;
        let draft = i32::try_from(proposal.token_ids()[0])
            .map_err(|_| GenerationServiceError::TokenIdOverflow)?;
        let block_inputs = [pending, draft];
        let block = self
            .target
            .decode_block_with_mtp_state(&block_inputs)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let hidden = Self::target_hidden_rows(&block, block_inputs.len())?;
        if block.token_ids().len() != block_inputs.len() {
            return Err(GenerationServiceError::Execution(
                "Gemma 4 target verify row count differs from draft width".to_owned(),
            ));
        }
        let accepted = accepted_draft_prefix_len(proposal.token_ids(), block.token_ids())?;
        let committed_rows = if accepted == Self::MAX_DRAFT_WIDTH {
            Self::MAX_DRAFT_WIDTH + 1
        } else {
            accepted + 1
        };
        let steps = (0..committed_rows)
            .map(|row| Self::step_from_output(&block, row))
            .collect::<Result<Vec<_>, _>>()?;

        // The assistant has no private KV timeline to commit or rewind. The
        // next proposal borrows the target lease after this block is resolved.
        self.pending_speculative_block = Some(PendingGemma4MtpBlockV1 {
            target_hidden_rows_bf16: hidden.to_vec(),
            target_input_rows: committed_rows,
        });
        Ok(steps)
    }

    fn finalize_mtp_draft(
        &mut self,
        committed_input_rows: usize,
    ) -> Result<(), GenerationServiceError> {
        let pending = self.pending_speculative_block.take().ok_or_else(|| {
            GenerationServiceError::Execution(
                "no Gemma MTP target block is pending finalization".to_owned(),
            )
        })?;
        if committed_input_rows == 0 || committed_input_rows > pending.target_input_rows {
            self.pending_speculative_block = Some(pending);
            return Err(GenerationServiceError::Speculative(
                SpeculativeError::InvalidDecision.to_string(),
            ));
        }
        // `resolve_decode_block` is the target transaction boundary.  It
        // retains the committed prefix and discards/replays any unconsumed
        // verification row; no frontend rewind of published target state is
        // attempted here.
        self.target
            .resolve_decode_block(committed_input_rows)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let hidden_start = (committed_input_rows - 1)
            .checked_mul(GEMMA4_MTP_TARGET_HIDDEN_WIDTH)
            .ok_or(GenerationServiceError::CountOverflow)?;
        self.last_target_hidden_bf16 = pending.target_hidden_rows_bf16
            [hidden_start..hidden_start + GEMMA4_MTP_TARGET_HIDDEN_WIDTH]
            .to_vec();
        Ok(())
    }
}

impl GenerationExecutorV1 for Gemma4MtpGenerationExecutorV1 {
    fn prefill(
        &mut self,
        input_token_ids: &[u32],
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        Self::reject_sampling(include_last_logits)?;
        if input_token_ids.is_empty() {
            return Err(GenerationServiceError::Execution(
                "Gemma 4 MTP prefill requires at least one token".to_owned(),
            ));
        }
        if self.pending_speculative_block.is_some() {
            return Err(GenerationServiceError::Execution(
                "speculative Gemma target block must be finalized before prefill".to_owned(),
            ));
        }
        let input = input_token_ids
            .iter()
            .map(|&token| i32::try_from(token).map_err(|_| GenerationServiceError::TokenIdOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        let output = self
            .target
            .prefill_with_mtp_state(&input)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        let terminal_hidden = Self::target_terminal_hidden(&output, input.len())?;
        let final_row = output
            .token_ids()
            .len()
            .checked_sub(1)
            .ok_or(GenerationServiceError::MissingDeviceArgmax)?;
        let step = Self::step_from_output(&output, final_row)?;
        // The assistant is stateless across proposals. Its first execution
        // borrows this request's read-only KV lease after target prefill.
        self.last_target_hidden_bf16 = terminal_hidden;
        Ok(step)
    }

    fn decode(
        &mut self,
        token_id: u32,
        include_last_logits: bool,
    ) -> Result<GenerationStepV1, GenerationServiceError> {
        Self::reject_sampling(include_last_logits)?;
        if self.pending_speculative_block.is_some() {
            return Err(GenerationServiceError::Execution(
                "speculative Gemma target block must be finalized before ordinary decode"
                    .to_owned(),
            ));
        }
        let token = i32::try_from(token_id).map_err(|_| GenerationServiceError::TokenIdOverflow)?;
        let output = self
            .target
            .decode(token)
            .map_err(|error| GenerationServiceError::Execution(error.to_string()))?;
        Self::step_from_output(&output, 0)
    }

    fn cancel(&mut self) {
        // Cancellation is request poison/drop.  It does not rewind a target
        // transition which may already have been published by the adapter.
        self.target.cancel();
        self.pending_speculative_block = None;
        self.last_target_hidden_bf16.clear();
    }
}

impl SpeculativeGenerationExecutorV1 for Gemma4MtpGenerationExecutorV1 {
    fn draft_provider(&mut self) -> Option<&mut dyn DraftProviderV1> {
        Some(self)
    }

    fn has_draft_provider(&self) -> bool {
        true
    }

    fn speculative_draft_width(&self) -> usize {
        Self::MAX_DRAFT_WIDTH
    }

    fn speculative_decode_greedy(
        &mut self,
        pending_token: u32,
    ) -> Result<Vec<GenerationStepV1>, GenerationServiceError> {
        let proposal = self
            .propose_mtp_draft(pending_token)
            .map_err(GenerationServiceError::from)?
            .ok_or_else(|| {
                GenerationServiceError::Execution("Gemma MTP provider returned no draft".to_owned())
            })?;
        self.speculative_decode_with_proposal(pending_token, &proposal)
    }

    fn speculative_decode_with_proposal(
        &mut self,
        pending_token: u32,
        proposal: &DraftProposalV1,
    ) -> Result<Vec<GenerationStepV1>, GenerationServiceError> {
        self.verify_mtp_draft(pending_token, proposal)
    }

    fn finalize_speculative_decode(
        &mut self,
        committed_input_rows: usize,
    ) -> Result<(), GenerationServiceError> {
        self.finalize_mtp_draft(committed_input_rows)
    }
}

impl DraftProviderV1 for Gemma4MtpGenerationExecutorV1 {
    fn kind(&self) -> DraftProviderKindV1 {
        DraftProviderKindV1::Gemma4Mtp
    }

    fn propose(
        &mut self,
        committed_target_tokens: &[u32],
        max_width: usize,
    ) -> Result<Option<DraftProposalV1>, SpeculativeError> {
        if committed_target_tokens.len() > sllm_core::MAX_SPECULATIVE_HISTORY_TOKENS_V1 {
            return Err(SpeculativeError::HistoryLimitExceeded);
        }
        if max_width != Self::MAX_DRAFT_WIDTH {
            return Err(if max_width == 0 {
                SpeculativeError::ZeroDraftWidth
            } else {
                SpeculativeError::DraftWidthExceeded
            });
        }
        let pending_token = committed_target_tokens
            .last()
            .copied()
            .ok_or(SpeculativeError::HistoryLimitExceeded)?;
        self.propose_mtp_draft(pending_token)
    }
}

/// Returns the contiguous accepted draft prefix for target argmax rows.
///
/// Keeping this decision pure makes the width-one contract testable without a
/// GPU request and keeps acceptance semantics in one place.  The target rows
/// include the replacement/final row, hence `target_tokens` may contain one
/// more item than `draft_tokens`.
fn accepted_draft_prefix_len(
    draft_tokens: &[u32],
    target_tokens: &[i32],
) -> Result<usize, GenerationServiceError> {
    if draft_tokens.is_empty()
        || draft_tokens.len() != 1
        || target_tokens.len() != draft_tokens.len() + 1
    {
        return Err(GenerationServiceError::Speculative(
            SpeculativeError::InvalidDecision.to_string(),
        ));
    }
    let target =
        u32::try_from(target_tokens[0]).map_err(|_| GenerationServiceError::TokenIdOverflow)?;
    Ok(usize::from(draft_tokens[0] == target))
}

#[cfg(test)]
mod tests {
    use super::{Gemma4MtpGenerationExecutorV1, accepted_draft_prefix_len};

    #[test]
    fn gemma_mtp_width_is_fixed_to_one() {
        assert_eq!(Gemma4MtpGenerationExecutorV1::MAX_DRAFT_WIDTH, 1);
        assert_eq!(
            Gemma4MtpGenerationExecutorV1::target_block_rows_for_draft_width(1),
            Some(2)
        );
        assert_eq!(
            Gemma4MtpGenerationExecutorV1::target_block_rows_for_draft_width(0),
            None
        );
        assert_eq!(
            Gemma4MtpGenerationExecutorV1::target_block_rows_for_draft_width(2),
            None
        );
        assert_eq!(
            Gemma4MtpGenerationExecutorV1::target_block_rows_for_draft_width(
                sllm_core::MAX_SPECULATIVE_DRAFT_WIDTH_V1
            ),
            None
        );
    }

    #[test]
    fn fixture_target_accepts_or_rejects_the_single_draft() {
        assert_eq!(accepted_draft_prefix_len(&[7], &[7, 8]).unwrap(), 1);
        assert_eq!(accepted_draft_prefix_len(&[7], &[9, 8]).unwrap(), 0);
    }

    #[test]
    fn fixture_target_row_contract_is_fail_closed() {
        assert!(accepted_draft_prefix_len(&[], &[1, 2]).is_err());
        assert!(accepted_draft_prefix_len(&[1, 2], &[1, 2, 3]).is_err());
        assert!(accepted_draft_prefix_len(&[1], &[1]).is_err());
        assert!(accepted_draft_prefix_len(&[1], &[-1, 2]).is_err());
    }

    #[test]
    fn fixture_sampling_and_width_expansion_are_fail_closed() {
        assert!(Gemma4MtpGenerationExecutorV1::validate_draft_width(0).is_err());
        assert!(Gemma4MtpGenerationExecutorV1::validate_draft_width(2).is_err());
        assert!(Gemma4MtpGenerationExecutorV1::validate_draft_width(1).is_ok());
        assert!(Gemma4MtpGenerationExecutorV1::reject_sampling(true).is_err());
        assert!(Gemma4MtpGenerationExecutorV1::reject_sampling(false).is_ok());
    }
}
