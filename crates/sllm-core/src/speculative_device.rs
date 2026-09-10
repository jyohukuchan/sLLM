//! Model-independent GPU speculative verification and decision decoding.
//!
//! Proposal/target support remains on the device. Only the bounded decision
//! record is read back; model adapters retain ownership of state rollback.

use crate::device_sampling::FIXED_K20_DECISION_BYTES;
use crate::{
    BufferRange, ExecutionError, ExecutionQueue, ExecutionSession, ExecutionState,
    SamplingSelectionV1,
};
use std::time::Duration;

fn invalid_decision(reason: String) -> ExecutionError {
    ExecutionError::InvalidRequest { reason }
}

/// Private fixed-K20 GPU verification result consumed by speculative generation.
/// The record is converted to ordinary generation steps only after its
/// version, status, counts, IDs, and target log probabilities are checked.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedK20SpeculativeDecisionV1 {
    width: usize,
    accepted_draft_tokens: usize,
    selections: Vec<SamplingSelectionV1>,
}

impl FixedK20SpeculativeDecisionV1 {
    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn accepted_draft_tokens(&self) -> usize {
        self.accepted_draft_tokens
    }

    pub fn selections(&self) -> &[SamplingSelectionV1] {
        &self.selections
    }
}

pub fn decode_fixed_k20_speculative_decision(
    bytes: &[u8; FIXED_K20_DECISION_BYTES as usize],
    width: usize,
    draft_ids: &[u32],
    vocabulary_size: u32,
) -> Result<FixedK20SpeculativeDecisionV1, ExecutionError> {
    if vocabulary_size == 0 || draft_ids.iter().any(|&id| id >= vocabulary_size) {
        return Err(invalid_decision(
            "fixed-K20 draft IDs require a nonempty matching vocabulary".to_owned(),
        ));
    }
    const VERSION: u32 = 1;
    const NO_REJECTION: u32 = u32::MAX;
    if !(1..=8).contains(&width) || draft_ids.len() != width {
        return Err(invalid_decision(
            "fixed-K20 decision width is outside 1 through 8".to_owned(),
        ));
    }
    let read_u32 = |offset: usize| -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("decision u32"))
    };
    let read_f64 = |offset: usize| -> f64 {
        f64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("decision f64"))
    };
    let version = read_u32(0);
    let status = read_u32(4);
    let accepted = usize::try_from(read_u32(8))
        .map_err(|_| invalid_decision("fixed-K20 accepted count overflowed usize".to_owned()))?;
    let emitted = usize::try_from(read_u32(12))
        .map_err(|_| invalid_decision("fixed-K20 emitted count overflowed usize".to_owned()))?;
    let rejected_at = read_u32(16);
    let record_width = usize::try_from(read_u32(20))
        .map_err(|_| invalid_decision("fixed-K20 decision width overflowed usize".to_owned()))?;
    let draws_used = read_u32(24);
    let reserved0 = read_u32(28);
    let reserved1 = read_u32(68);
    let expected_emitted = accepted
        .checked_add(1)
        .ok_or_else(|| invalid_decision("fixed-K20 emitted count overflowed".to_owned()))?;
    if version != VERSION
        || status != 0
        || record_width != width
        || accepted > width
        || emitted != expected_emitted
        || emitted > 9
        || reserved0 != 0
        || reserved1 != 0
    {
        return Err(invalid_decision(
            "fixed-K20 decision header is invalid".to_owned(),
        ));
    }
    let expected_draws = if accepted == width {
        width + 1
    } else {
        accepted + 2
    };
    if usize::try_from(draws_used).ok() != Some(expected_draws) {
        return Err(invalid_decision(
            "fixed-K20 decision draw count is invalid".to_owned(),
        ));
    }
    if (accepted == width && rejected_at != NO_REJECTION)
        || (accepted < width && rejected_at != u32::try_from(accepted).expect("bounded count"))
    {
        return Err(invalid_decision(
            "fixed-K20 decision rejection index is invalid".to_owned(),
        ));
    }
    let mut selections = Vec::with_capacity(emitted);
    for index in 0..emitted {
        let token_offset = 32 + index * 4;
        let token_id = read_u32(token_offset);
        let logprob = read_f64(72 + index * 8);
        if token_id >= vocabulary_size || !logprob.is_finite() || logprob > 0.0 {
            return Err(invalid_decision(
                "fixed-K20 decision token or target log probability is invalid".to_owned(),
            ));
        }
        if index < accepted && draft_ids.get(index).copied() != Some(token_id) {
            return Err(invalid_decision(
                "fixed-K20 decision accepted prefix differs from draft IDs".to_owned(),
            ));
        }
        selections.push(SamplingSelectionV1 {
            token_id,
            logprob,
            top_logprobs: Vec::new(),
        });
    }
    for index in emitted..9 {
        if read_u32(32 + index * 4) != 0 || read_f64(72 + index * 8) != 0.0 {
            return Err(invalid_decision(
                "fixed-K20 decision has nonzero unused entries".to_owned(),
            ));
        }
    }
    Ok(FixedK20SpeculativeDecisionV1 {
        width,
        accepted_draft_tokens: accepted,
        selections,
    })
}

impl ExecutionSession {
    /// Verifies device-resident target/draft supports and validates the compact
    /// result against the caller's vocabulary. No model architecture is needed.
    /// The adapter must discard staged model state if this operation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_fixed_k20_speculative_decision(
        &self,
        queue: &ExecutionQueue,
        target: BufferRange,
        draft: BufferRange,
        draft_ids: &[u32],
        vocabulary_size: u32,
        seed: u64,
        absolute_position: u64,
        output: BufferRange,
        completion_timeout: Duration,
    ) -> Result<FixedK20SpeculativeDecisionV1, ExecutionError> {
        if vocabulary_size == 0 || draft_ids.iter().any(|&id| id >= vocabulary_size) {
            return Err(invalid_decision(
                "fixed-K20 draft IDs require a nonempty matching vocabulary".to_owned(),
            ));
        }
        // The backend hook keeps its original name for adapter compatibility;
        // its support/decision ABI is independent of how proposals are produced.
        self.verify_fixed_k20_mtp(
            queue,
            target,
            draft,
            draft_ids,
            seed,
            absolute_position,
            output.clone(),
        )?;
        let mut readback = self.readback(queue, output)?;
        if readback.wait(completion_timeout)? != ExecutionState::Success {
            return Err(invalid_decision(
                "fixed-K20 speculative decision readback did not complete successfully".to_owned(),
            ));
        }
        let mut bytes = [0_u8; FIXED_K20_DECISION_BYTES as usize];
        if readback.read_into(&mut bytes)? != FIXED_K20_DECISION_BYTES {
            return Err(invalid_decision(
                "fixed-K20 decision readback returned an unexpected byte count".to_owned(),
            ));
        }
        decode_fixed_k20_speculative_decision(&bytes, draft_ids.len(), draft_ids, vocabulary_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(width: usize, accepted: usize, ids: &[u32]) -> [u8; 144] {
        let mut bytes = [0; 144];
        for (offset, value) in [
            (0, 1),
            (8, accepted as u32),
            (12, accepted as u32 + 1),
            (
                16,
                if accepted == width {
                    u32::MAX
                } else {
                    accepted as u32
                },
            ),
            (20, width as u32),
            (24, (accepted + 1 + usize::from(accepted < width)) as u32),
        ] {
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        for (i, id) in ids.iter().enumerate() {
            bytes[32 + i * 4..36 + i * 4].copy_from_slice(&id.to_le_bytes());
            bytes[72 + i * 8..80 + i * 8].copy_from_slice(&(-0.5_f64).to_le_bytes());
        }
        bytes
    }

    #[test]
    fn decisions_use_caller_vocabulary_including_non_qwen_boundaries() {
        for vocabulary in [31, 32, 33, 151_936, 262_144] {
            let drafts = [vocabulary - 2, vocabulary - 1];
            let bytes = record(2, 2, &[drafts[0], drafts[1], vocabulary - 1]);
            assert!(decode_fixed_k20_speculative_decision(&bytes, 2, &drafts, vocabulary).is_ok());
            assert!(
                decode_fixed_k20_speculative_decision(&bytes, 2, &drafts, vocabulary - 1).is_err()
            );
        }
    }

    #[test]
    fn decision_prefix_and_width_contract_covers_acceptance_boundaries() {
        for width in [1, 2, 3, 7, 8] {
            let drafts: Vec<u32> = (0..width as u32).collect();
            for accepted in 0..=width {
                let mut emitted = drafts[..accepted].to_vec();
                emitted.push(30);
                let bytes = record(width, accepted, &emitted);
                let result =
                    decode_fixed_k20_speculative_decision(&bytes, width, &drafts, 31).unwrap();
                assert_eq!(result.accepted_draft_tokens(), accepted);
                assert_eq!(result.selections().len(), accepted + 1);
            }
        }
        assert!(decode_fixed_k20_speculative_decision(&[0; 144], 0, &[], 31).is_err());
        assert!(decode_fixed_k20_speculative_decision(&[0; 144], 9, &[0; 9], 31).is_err());
        assert!(decode_fixed_k20_speculative_decision(&[0; 144], 1, &[0], 0).is_err());
        let mut bytes = record(2, 1, &[0, 30]);
        bytes[72..80].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(decode_fixed_k20_speculative_decision(&bytes, 2, &[0, 1], 31).is_err());
        let bytes = record(2, 1, &[1, 30]);
        assert!(decode_fixed_k20_speculative_decision(&bytes, 2, &[0, 1], 31).is_err());
    }
}
