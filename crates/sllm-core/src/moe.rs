//! Backend-neutral sparse-MoE routing contracts and a host reference oracle.
//!
//! The reference implementation in this module exists for focused tests and
//! numerical evidence. Production Qwen3.5 MoE execution must make routing
//! decisions on the GPU and must not use this implementation as a fallback.

use std::fmt;

pub const QWEN35_MOE_EXPERT_COUNT: u32 = 256;
pub const QWEN35_MOE_SELECTED_EXPERT_COUNT: u32 = 8;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SparseMoeRoutingContract {
    expert_count: u32,
    selected_expert_count: u32,
    renormalize_selected_weights: bool,
}

impl SparseMoeRoutingContract {
    pub fn new(
        expert_count: u32,
        selected_expert_count: u32,
        renormalize_selected_weights: bool,
    ) -> Result<Self, SparseMoeRoutingError> {
        if expert_count == 0 || expert_count > u16::MAX as u32 {
            return Err(SparseMoeRoutingError::InvalidExpertCount(expert_count));
        }
        if selected_expert_count == 0 || selected_expert_count > expert_count {
            return Err(SparseMoeRoutingError::InvalidSelectedExpertCount {
                expert_count,
                selected_expert_count,
            });
        }
        Ok(Self {
            expert_count,
            selected_expert_count,
            renormalize_selected_weights,
        })
    }

    pub fn qwen35() -> Self {
        Self::new(
            QWEN35_MOE_EXPERT_COUNT,
            QWEN35_MOE_SELECTED_EXPERT_COUNT,
            true,
        )
        .expect("the reviewed Qwen3.5 MoE routing contract is valid")
    }

    pub const fn expert_count(self) -> u32 {
        self.expert_count
    }

    pub const fn selected_expert_count(self) -> u32 {
        self.selected_expert_count
    }

    pub const fn renormalize_selected_weights(self) -> bool {
        self.renormalize_selected_weights
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SparseMoeRouting {
    token_count: u32,
    contract: SparseMoeRoutingContract,
    expert_ids: Vec<u16>,
    expert_weights: Vec<f32>,
    expert_counts: Vec<u32>,
    expert_offsets: Vec<u32>,
    grouped_token_ids: Vec<u32>,
    grouped_topk_slots: Vec<u16>,
}

impl SparseMoeRouting {
    pub const fn token_count(&self) -> u32 {
        self.token_count
    }

    pub const fn contract(&self) -> SparseMoeRoutingContract {
        self.contract
    }

    /// Row-major `[token_count, selected_expert_count]` expert IDs.
    pub fn expert_ids(&self) -> &[u16] {
        &self.expert_ids
    }

    /// Row-major weights corresponding one-to-one with `expert_ids`.
    pub fn expert_weights(&self) -> &[f32] {
        &self.expert_weights
    }

    /// Pair counts for every expert, including inactive experts.
    pub fn expert_counts(&self) -> &[u32] {
        &self.expert_counts
    }

    /// Exclusive prefix sum with `expert_count + 1` entries.
    pub fn expert_offsets(&self) -> &[u32] {
        &self.expert_offsets
    }

    /// Stable expert-major pair list. Within an expert, token order and then
    /// top-k slot order are preserved.
    pub fn grouped_token_ids(&self) -> &[u32] {
        &self.grouped_token_ids
    }

    pub fn grouped_topk_slots(&self) -> &[u16] {
        &self.grouped_topk_slots
    }

    pub fn validate(&self) -> Result<(), SparseMoeRoutingError> {
        let experts = usize::try_from(self.contract.expert_count).unwrap();
        let topk = usize::try_from(self.contract.selected_expert_count).unwrap();
        let tokens = usize::try_from(self.token_count).unwrap();
        let pairs = tokens
            .checked_mul(topk)
            .ok_or(SparseMoeRoutingError::PairCountOverflow)?;
        if self.expert_ids.len() != pairs
            || self.expert_weights.len() != pairs
            || self.expert_counts.len() != experts
            || self.expert_offsets.len() != experts + 1
            || self.grouped_token_ids.len() != pairs
            || self.grouped_topk_slots.len() != pairs
        {
            return Err(SparseMoeRoutingError::LayoutMismatch);
        }
        if self.expert_offsets.first().copied() != Some(0)
            || self.expert_offsets.last().copied() != u32::try_from(pairs).ok()
        {
            return Err(SparseMoeRoutingError::OffsetMismatch);
        }
        for expert in 0..experts {
            let begin = usize::try_from(self.expert_offsets[expert]).unwrap();
            let end = usize::try_from(self.expert_offsets[expert + 1]).unwrap();
            if end < begin || end - begin != usize::try_from(self.expert_counts[expert]).unwrap() {
                return Err(SparseMoeRoutingError::OffsetMismatch);
            }
            let mut previous: Option<(u32, u16)> = None;
            for pair in begin..end {
                let token = self.grouped_token_ids[pair];
                let slot = self.grouped_topk_slots[pair];
                if token >= self.token_count || usize::from(slot) >= topk {
                    return Err(SparseMoeRoutingError::GroupedPairOutOfRange);
                }
                let flat = usize::try_from(token).unwrap() * topk + usize::from(slot);
                if usize::from(self.expert_ids[flat]) != expert {
                    return Err(SparseMoeRoutingError::GroupedPairExpertMismatch);
                }
                if previous.is_some_and(|value| value >= (token, slot)) {
                    return Err(SparseMoeRoutingError::GroupedPairOrderMismatch);
                }
                previous = Some((token, slot));
            }
        }
        for token in 0..tokens {
            let begin = token * topk;
            let end = begin + topk;
            let mut seen = vec![false; experts];
            let mut sum = 0.0_f32;
            for index in begin..end {
                let expert = usize::from(self.expert_ids[index]);
                let weight = self.expert_weights[index];
                if expert >= experts || seen[expert] {
                    return Err(SparseMoeRoutingError::DuplicateOrOutOfRangeExpert);
                }
                if !weight.is_finite() || weight < 0.0 {
                    return Err(SparseMoeRoutingError::InvalidWeight);
                }
                seen[expert] = true;
                sum += weight;
            }
            if self.contract.renormalize_selected_weights && (sum - 1.0).abs() > 2.0e-6 {
                return Err(SparseMoeRoutingError::WeightSumMismatch {
                    token: token as u32,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SparseMoeRoutingError {
    InvalidExpertCount(u32),
    InvalidSelectedExpertCount {
        expert_count: u32,
        selected_expert_count: u32,
    },
    TokenCountZero,
    LogitCountMismatch {
        expected: usize,
        actual: usize,
    },
    NonFiniteLogit {
        token: u32,
        expert: u32,
    },
    InvalidSoftmax {
        token: u32,
    },
    PairCountOverflow,
    LayoutMismatch,
    OffsetMismatch,
    GroupedPairOutOfRange,
    GroupedPairExpertMismatch,
    GroupedPairOrderMismatch,
    DuplicateOrOutOfRangeExpert,
    InvalidWeight,
    WeightSumMismatch {
        token: u32,
    },
}

impl fmt::Display for SparseMoeRoutingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "sparse MoE routing error: {self:?}")
    }
}

impl std::error::Error for SparseMoeRoutingError {}

/// FP32 softmax/top-k oracle for the reviewed routing contract.
///
/// Ties are resolved by smaller expert ID. This function is test-only in
/// purpose even though it is public so integration evidence tools can use it.
pub fn reference_sparse_moe_route(
    contract: SparseMoeRoutingContract,
    token_count: u32,
    logits: &[f32],
) -> Result<SparseMoeRouting, SparseMoeRoutingError> {
    if token_count == 0 {
        return Err(SparseMoeRoutingError::TokenCountZero);
    }
    let experts = usize::try_from(contract.expert_count).unwrap();
    let topk = usize::try_from(contract.selected_expert_count).unwrap();
    let tokens = usize::try_from(token_count).unwrap();
    let expected = tokens
        .checked_mul(experts)
        .ok_or(SparseMoeRoutingError::PairCountOverflow)?;
    if logits.len() != expected {
        return Err(SparseMoeRoutingError::LogitCountMismatch {
            expected,
            actual: logits.len(),
        });
    }

    let pairs = tokens
        .checked_mul(topk)
        .ok_or(SparseMoeRoutingError::PairCountOverflow)?;
    let mut expert_ids = Vec::with_capacity(pairs);
    let mut expert_weights = Vec::with_capacity(pairs);
    let mut expert_counts = vec![0_u32; experts];

    for token in 0..tokens {
        let row = &logits[token * experts..(token + 1) * experts];
        for (expert, logit) in row.iter().copied().enumerate() {
            if !logit.is_finite() {
                return Err(SparseMoeRoutingError::NonFiniteLogit {
                    token: token as u32,
                    expert: expert as u32,
                });
            }
        }
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let probabilities: Vec<f32> = row.iter().map(|value| (*value - maximum).exp()).collect();
        let denominator: f32 = probabilities.iter().sum();
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(SparseMoeRoutingError::InvalidSoftmax {
                token: token as u32,
            });
        }
        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_unstable_by(|left, right| {
            probabilities[*right]
                .total_cmp(&probabilities[*left])
                .then_with(|| left.cmp(right))
        });
        let selected = &order[..topk];
        let selected_sum: f32 = selected.iter().map(|expert| probabilities[*expert]).sum();
        if !selected_sum.is_finite() || selected_sum <= 0.0 {
            return Err(SparseMoeRoutingError::InvalidSoftmax {
                token: token as u32,
            });
        }
        for expert in selected {
            expert_ids.push(*expert as u16);
            let probability = probabilities[*expert] / denominator;
            expert_weights.push(if contract.renormalize_selected_weights {
                probability / (selected_sum / denominator)
            } else {
                probability
            });
            expert_counts[*expert] += 1;
        }
    }

    let mut expert_offsets = Vec::with_capacity(experts + 1);
    expert_offsets.push(0);
    for count in &expert_counts {
        let next = expert_offsets.last().copied().unwrap() + count;
        expert_offsets.push(next);
    }
    let mut write_positions = expert_offsets[..experts].to_vec();
    let mut grouped_token_ids = vec![0_u32; pairs];
    let mut grouped_topk_slots = vec![0_u16; pairs];
    for token in 0..tokens {
        for slot in 0..topk {
            let expert = usize::from(expert_ids[token * topk + slot]);
            let position = usize::try_from(write_positions[expert]).unwrap();
            grouped_token_ids[position] = token as u32;
            grouped_topk_slots[position] = slot as u16;
            write_positions[expert] += 1;
        }
    }

    let routing = SparseMoeRouting {
        token_count,
        contract,
        expert_ids,
        expert_weights,
        expert_counts,
        expert_offsets,
        grouped_token_ids,
        grouped_topk_slots,
    };
    routing.validate()?;
    Ok(routing)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_contract_is_exact() {
        let contract = SparseMoeRoutingContract::qwen35();
        assert_eq!(contract.expert_count(), 256);
        assert_eq!(contract.selected_expert_count(), 8);
        assert!(contract.renormalize_selected_weights());
    }

    #[test]
    fn reference_route_has_stable_ties_and_grouping() {
        let contract = SparseMoeRoutingContract::new(10, 3, true).unwrap();
        let logits = vec![0.0_f32; 20];
        let route = reference_sparse_moe_route(contract, 2, &logits).unwrap();
        assert_eq!(route.expert_ids(), &[0, 1, 2, 0, 1, 2]);
        assert_eq!(route.expert_counts(), &[2, 2, 2, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(route.expert_offsets(), &[0, 2, 4, 6, 6, 6, 6, 6, 6, 6, 6]);
        assert_eq!(route.grouped_token_ids(), &[0, 1, 0, 1, 0, 1]);
        assert_eq!(route.grouped_topk_slots(), &[0, 0, 1, 1, 2, 2]);
        for token in 0..2 {
            let sum: f32 = route.expert_weights()[token * 3..token * 3 + 3]
                .iter()
                .sum();
            assert!((sum - 1.0).abs() <= 2.0e-6);
        }
    }

    #[test]
    fn reference_route_handles_non_aligned_token_counts_and_extreme_skew() {
        let contract = SparseMoeRoutingContract::qwen35();
        for tokens in [1_usize, 2, 3, 7, 8, 31, 32, 33] {
            let mut logits = vec![-20.0_f32; tokens * 256];
            for token in 0..tokens {
                for rank in 0..8 {
                    logits[token * 256 + (255 - rank)] = 20.0 - rank as f32;
                }
            }
            let route = reference_sparse_moe_route(contract, tokens as u32, &logits).unwrap();
            assert_eq!(
                &route.expert_ids()[..8],
                &[255, 254, 253, 252, 251, 250, 249, 248]
            );
            assert_eq!(route.expert_counts()[255], tokens as u32);
            assert_eq!(route.expert_offsets()[256], (tokens * 8) as u32);
        }
    }

    #[test]
    fn reference_route_rejects_nonfinite_or_malformed_input() {
        let contract = SparseMoeRoutingContract::new(4, 2, true).unwrap();
        assert_eq!(
            reference_sparse_moe_route(contract, 1, &[0.0, f32::NAN, 0.0, 0.0]),
            Err(SparseMoeRoutingError::NonFiniteLogit {
                token: 0,
                expert: 1
            })
        );
        assert_eq!(
            reference_sparse_moe_route(contract, 1, &[0.0; 3]),
            Err(SparseMoeRoutingError::LogitCountMismatch {
                expected: 4,
                actual: 3
            })
        );
    }
}
