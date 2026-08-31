//! Pure host semantics for the reviewed MiniMax M3 foundation.
//!
//! This module freezes the official sigmoid router, selection-only correction
//! bias, stable top-4 policy, normalized routed weights, routed-branch scale,
//! shared-expert merge, text-layer schedule, and MTP topology boundaries. It
//! contains no container or backend behavior and is not a CPU production
//! fallback. In particular, the MTP contract records topology only and does
//! not admit speculative generation.

use std::fmt;

pub const MINIMAX_M3_SEMANTIC_REPOSITORY: &str = "MiniMaxAI/MiniMax-M3";
pub const MINIMAX_M3_SEMANTIC_REVISION: &str = "f0e1c1e04d40177e4673a22097036854f536e9c0";
pub const MINIMAX_M3_CONFIG_URL: &str = "https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/f0e1c1e04d40177e4673a22097036854f536e9c0/config.json";
pub const MINIMAX_M3_TRANSFORMERS_REFERENCE_REVISION: &str =
    "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c";
pub const MINIMAX_M3_TRANSFORMERS_REFERENCE_URL: &str = "https://github.com/huggingface/transformers/blob/42ca97014c85d71a88ad60d55f08cb9fb4d26e2c/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py";

pub const MINIMAX_M3_TEXT_LAYER_COUNT: u32 = 60;
pub const MINIMAX_M3_DENSE_LAYER_COUNT: u32 = 3;
pub const MINIMAX_M3_EXPERT_COUNT: u32 = 128;
pub const MINIMAX_M3_SELECTED_EXPERT_COUNT: u32 = 4;
pub const MINIMAX_M3_SHARED_EXPERT_COUNT: u32 = 1;
pub const MINIMAX_M3_ROUTED_SCALING_FACTOR: f32 = 2.0;
pub const MINIMAX_M3_MTP_MODULE_COUNT: u32 = 7;
pub const MINIMAX_M3_NEXTN_PREDICT_LAYER_COUNT: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MiniMaxM3SemanticRole {
    RouterLogit,
    RouterSelectionBias,
    SelectedExpertId,
    SelectedRouterScore,
    RoutedExpertOutput,
    SharedExpertOutput,
    CombinedOutput,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MiniMaxM3SemanticStage {
    RouterScore,
    RouterSelection,
    RouterNormalization,
    RoutedExpertSum,
    RoutedScaling,
    SharedExpertMerge,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MiniMaxM3LayerKind {
    Dense,
    SparseMoe,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MiniMaxM3MtpStage {
    Module(u32),
    NextNPredictLayer(u32),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MiniMaxM3UnsupportedFeature {
    FullMtpSpeculativeGeneration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MiniMaxM3SemanticError {
    InvalidTokenCount(u64),
    InvalidHiddenSize(u32),
    TextLayerOutOfRange(u32),
    RoutingNotAllowedOnDenseLayer(u32),
    MtpTopologyMismatch {
        module_count: u32,
        nextn_predict_layer_count: u32,
    },
    MtpModuleOutOfRange(u32),
    NextNPredictLayerOutOfRange(u32),
    UnsupportedFeature(MiniMaxM3UnsupportedFeature),
    ElementCountOverflow {
        role: MiniMaxM3SemanticRole,
    },
    AllocationFailed {
        role: MiniMaxM3SemanticRole,
    },
    ElementCountMismatch {
        role: MiniMaxM3SemanticRole,
        expected: usize,
        actual: usize,
    },
    NonFiniteInput {
        role: MiniMaxM3SemanticRole,
        index: usize,
    },
    NonFiniteIntermediate {
        stage: MiniMaxM3SemanticStage,
        index: usize,
    },
    DuplicateExpert {
        token: u64,
        expert: u32,
    },
    ExpertOutOfRange {
        token: u64,
        expert: u32,
    },
    InvalidSelectedRouterScore {
        token: u64,
        slot: u32,
        score_bits: u32,
    },
    ZeroNormalizer {
        token: u64,
    },
    InvalidNormalizedWeightSum {
        token: u64,
        sum_bits: u32,
    },
}

impl fmt::Display for MiniMaxM3SemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "MiniMax M3 semantic error: {self:?}")
    }
}

impl std::error::Error for MiniMaxM3SemanticError {}

/// Return the reviewed dense/MoE schedule for one text layer.
pub const fn minimax_m3_layer_kind(
    layer: u32,
) -> Result<MiniMaxM3LayerKind, MiniMaxM3SemanticError> {
    match layer {
        0..MINIMAX_M3_DENSE_LAYER_COUNT => Ok(MiniMaxM3LayerKind::Dense),
        MINIMAX_M3_DENSE_LAYER_COUNT..MINIMAX_M3_TEXT_LAYER_COUNT => {
            Ok(MiniMaxM3LayerKind::SparseMoe)
        }
        _ => Err(MiniMaxM3SemanticError::TextLayerOutOfRange(layer)),
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MiniMaxM3MtpTopology {
    module_count: u32,
    nextn_predict_layer_count: u32,
}

impl MiniMaxM3MtpTopology {
    /// Validate the two independent identity fields from the reviewed config.
    ///
    /// This contract deliberately does not invent a module-to-next-N mapping:
    /// the built-in Transformers implementation ignores checkpoint `mtp.*`
    /// keys and does not provide full speculative-generation semantics.
    pub const fn new(
        module_count: u32,
        nextn_predict_layer_count: u32,
    ) -> Result<Self, MiniMaxM3SemanticError> {
        if module_count != MINIMAX_M3_MTP_MODULE_COUNT
            || nextn_predict_layer_count != MINIMAX_M3_NEXTN_PREDICT_LAYER_COUNT
        {
            return Err(MiniMaxM3SemanticError::MtpTopologyMismatch {
                module_count,
                nextn_predict_layer_count,
            });
        }
        Ok(Self {
            module_count,
            nextn_predict_layer_count,
        })
    }

    pub const fn reviewed() -> Self {
        Self {
            module_count: MINIMAX_M3_MTP_MODULE_COUNT,
            nextn_predict_layer_count: MINIMAX_M3_NEXTN_PREDICT_LAYER_COUNT,
        }
    }

    pub const fn module_count(self) -> u32 {
        self.module_count
    }

    pub const fn nextn_predict_layer_count(self) -> u32 {
        self.nextn_predict_layer_count
    }

    pub const fn module_stage(
        self,
        index: u32,
    ) -> Result<MiniMaxM3MtpStage, MiniMaxM3SemanticError> {
        if index >= self.module_count {
            return Err(MiniMaxM3SemanticError::MtpModuleOutOfRange(index));
        }
        Ok(MiniMaxM3MtpStage::Module(index))
    }

    pub const fn nextn_predict_layer_stage(
        self,
        index: u32,
    ) -> Result<MiniMaxM3MtpStage, MiniMaxM3SemanticError> {
        if index >= self.nextn_predict_layer_count {
            return Err(MiniMaxM3SemanticError::NextNPredictLayerOutOfRange(index));
        }
        Ok(MiniMaxM3MtpStage::NextNPredictLayer(index))
    }

    pub const fn admit_full_speculative_generation(self) -> Result<(), MiniMaxM3SemanticError> {
        let _ = self;
        Err(MiniMaxM3SemanticError::UnsupportedFeature(
            MiniMaxM3UnsupportedFeature::FullMtpSpeculativeGeneration,
        ))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MiniMaxM3Routing {
    layer: u32,
    token_count: u64,
    expert_ids: Vec<u16>,
    normalized_expert_weights: Vec<f32>,
}

impl MiniMaxM3Routing {
    pub const fn layer(&self) -> u32 {
        self.layer
    }

    pub const fn token_count(&self) -> u64 {
        self.token_count
    }

    pub const fn expert_count(&self) -> u32 {
        MINIMAX_M3_EXPERT_COUNT
    }

    pub const fn selected_expert_count(&self) -> u32 {
        MINIMAX_M3_SELECTED_EXPERT_COUNT
    }

    pub const fn routed_scaling_factor(&self) -> f32 {
        MINIMAX_M3_ROUTED_SCALING_FACTOR
    }

    /// Stable selected IDs in row-major `[tokens, 4]` order.
    pub fn expert_ids(&self) -> &[u16] {
        &self.expert_ids
    }

    /// Bias-free sigmoid scores normalized over the selected four experts.
    /// The routed branch scale is applied later, after the weighted expert sum.
    pub fn normalized_expert_weights(&self) -> &[f32] {
        &self.normalized_expert_weights
    }
}

fn checked_usize(value: u64, role: MiniMaxM3SemanticRole) -> Result<usize, MiniMaxM3SemanticError> {
    usize::try_from(value).map_err(|_| MiniMaxM3SemanticError::ElementCountOverflow { role })
}

fn checked_product(
    left: usize,
    right: usize,
    role: MiniMaxM3SemanticRole,
) -> Result<usize, MiniMaxM3SemanticError> {
    left.checked_mul(right)
        .ok_or(MiniMaxM3SemanticError::ElementCountOverflow { role })
}

fn empty_vec<T>(
    capacity: usize,
    role: MiniMaxM3SemanticRole,
) -> Result<Vec<T>, MiniMaxM3SemanticError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| MiniMaxM3SemanticError::AllocationFailed { role })?;
    Ok(values)
}

fn validate_finite(
    values: &[f32],
    expected: usize,
    role: MiniMaxM3SemanticRole,
) -> Result<(), MiniMaxM3SemanticError> {
    if values.len() != expected {
        return Err(MiniMaxM3SemanticError::ElementCountMismatch {
            role,
            expected,
            actual: values.len(),
        });
    }
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(MiniMaxM3SemanticError::NonFiniteInput { role, index });
        }
    }
    Ok(())
}

fn stable_sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn route_from_selected_scores(
    layer: u32,
    token_count: u64,
    selected_expert_ids: &[u16],
    selected_unbiased_scores: &[f32],
) -> Result<MiniMaxM3Routing, MiniMaxM3SemanticError> {
    if minimax_m3_layer_kind(layer)? != MiniMaxM3LayerKind::SparseMoe {
        return Err(MiniMaxM3SemanticError::RoutingNotAllowedOnDenseLayer(layer));
    }
    if token_count == 0 {
        return Err(MiniMaxM3SemanticError::InvalidTokenCount(token_count));
    }
    let tokens = checked_usize(token_count, MiniMaxM3SemanticRole::SelectedExpertId)?;
    let top_k = MINIMAX_M3_SELECTED_EXPERT_COUNT as usize;
    let pair_count = checked_product(tokens, top_k, MiniMaxM3SemanticRole::SelectedExpertId)?;
    if selected_expert_ids.len() != pair_count {
        return Err(MiniMaxM3SemanticError::ElementCountMismatch {
            role: MiniMaxM3SemanticRole::SelectedExpertId,
            expected: pair_count,
            actual: selected_expert_ids.len(),
        });
    }
    validate_finite(
        selected_unbiased_scores,
        pair_count,
        MiniMaxM3SemanticRole::SelectedRouterScore,
    )?;

    let mut normalized_expert_weights =
        empty_vec(pair_count, MiniMaxM3SemanticRole::SelectedRouterScore)?;
    for token in 0..tokens {
        let start = token * top_k;
        let ids = &selected_expert_ids[start..start + top_k];
        let scores = &selected_unbiased_scores[start..start + top_k];
        for (slot, expert) in ids.iter().copied().enumerate() {
            if u32::from(expert) >= MINIMAX_M3_EXPERT_COUNT {
                return Err(MiniMaxM3SemanticError::ExpertOutOfRange {
                    token: token as u64,
                    expert: u32::from(expert),
                });
            }
            if ids[..slot].contains(&expert) {
                return Err(MiniMaxM3SemanticError::DuplicateExpert {
                    token: token as u64,
                    expert: u32::from(expert),
                });
            }
            let score = scores[slot];
            if !(0.0..=1.0).contains(&score) {
                return Err(MiniMaxM3SemanticError::InvalidSelectedRouterScore {
                    token: token as u64,
                    slot: slot as u32,
                    score_bits: score.to_bits(),
                });
            }
        }
        let normalizer = scores.iter().copied().try_fold(0.0_f32, |sum, score| {
            let next = sum + score;
            next.is_finite()
                .then_some(next)
                .ok_or(MiniMaxM3SemanticError::NonFiniteIntermediate {
                    stage: MiniMaxM3SemanticStage::RouterNormalization,
                    index: token,
                })
        })?;
        if normalizer <= 0.0 {
            return Err(MiniMaxM3SemanticError::ZeroNormalizer {
                token: token as u64,
            });
        }
        for score in scores {
            let weight = *score / normalizer;
            if !weight.is_finite() {
                return Err(MiniMaxM3SemanticError::NonFiniteIntermediate {
                    stage: MiniMaxM3SemanticStage::RouterNormalization,
                    index: normalized_expert_weights.len(),
                });
            }
            normalized_expert_weights.push(weight);
        }
    }

    Ok(MiniMaxM3Routing {
        layer,
        token_count,
        expert_ids: selected_expert_ids.to_vec(),
        normalized_expert_weights,
    })
}

/// Validate and normalize already-selected route metadata.
///
/// This boundary is useful for checking native top-k output independently of
/// the selector. IDs and unbiased sigmoid scores are row-major `[tokens, 4]`.
pub fn reference_minimax_m3_route_from_selection(
    layer: u32,
    token_count: u64,
    selected_expert_ids: &[u16],
    selected_unbiased_scores: &[f32],
) -> Result<MiniMaxM3Routing, MiniMaxM3SemanticError> {
    route_from_selected_scores(
        layer,
        token_count,
        selected_expert_ids,
        selected_unbiased_scores,
    )
}

/// Independent FP32 MiniMax M3 score-router reference.
///
/// The official router computes sigmoid scores, adds the correction bias only
/// for expert choice, gathers the original sigmoid scores, and normalizes the
/// selected four. Exact selection ties are made deterministic here by the
/// smaller expert ID, independent of backend top-k ordering.
pub fn reference_minimax_m3_route(
    layer: u32,
    token_count: u64,
    router_logits: &[f32],
    selection_bias: &[f32],
) -> Result<MiniMaxM3Routing, MiniMaxM3SemanticError> {
    if minimax_m3_layer_kind(layer)? != MiniMaxM3LayerKind::SparseMoe {
        return Err(MiniMaxM3SemanticError::RoutingNotAllowedOnDenseLayer(layer));
    }
    if token_count == 0 {
        return Err(MiniMaxM3SemanticError::InvalidTokenCount(token_count));
    }
    let tokens = checked_usize(token_count, MiniMaxM3SemanticRole::RouterLogit)?;
    let experts = MINIMAX_M3_EXPERT_COUNT as usize;
    let top_k = MINIMAX_M3_SELECTED_EXPERT_COUNT as usize;
    let logit_count = checked_product(tokens, experts, MiniMaxM3SemanticRole::RouterLogit)?;
    validate_finite(
        router_logits,
        logit_count,
        MiniMaxM3SemanticRole::RouterLogit,
    )?;
    validate_finite(
        selection_bias,
        experts,
        MiniMaxM3SemanticRole::RouterSelectionBias,
    )?;
    let pair_count = checked_product(tokens, top_k, MiniMaxM3SemanticRole::SelectedExpertId)?;
    let mut selected_ids = empty_vec(pair_count, MiniMaxM3SemanticRole::SelectedExpertId)?;
    let mut selected_scores = empty_vec(pair_count, MiniMaxM3SemanticRole::SelectedRouterScore)?;
    let mut unbiased_scores = empty_vec(experts, MiniMaxM3SemanticRole::RouterLogit)?;
    let mut selection_scores = empty_vec(experts, MiniMaxM3SemanticRole::RouterSelectionBias)?;
    let mut order = empty_vec(experts, MiniMaxM3SemanticRole::SelectedExpertId)?;

    for token in 0..tokens {
        unbiased_scores.clear();
        selection_scores.clear();
        order.clear();
        for (expert, bias) in selection_bias.iter().copied().enumerate() {
            let flat_index = token * experts + expert;
            let score = stable_sigmoid(router_logits[flat_index]);
            if !score.is_finite() {
                return Err(MiniMaxM3SemanticError::NonFiniteIntermediate {
                    stage: MiniMaxM3SemanticStage::RouterScore,
                    index: flat_index,
                });
            }
            let selection_score = score + bias;
            if !selection_score.is_finite() {
                return Err(MiniMaxM3SemanticError::NonFiniteIntermediate {
                    stage: MiniMaxM3SemanticStage::RouterSelection,
                    index: flat_index,
                });
            }
            unbiased_scores.push(score);
            selection_scores.push(selection_score);
            order.push(expert);
        }
        order.sort_unstable_by(|left, right| {
            selection_scores[*right]
                .total_cmp(&selection_scores[*left])
                .then_with(|| left.cmp(right))
        });
        for expert in order.iter().copied().take(top_k) {
            selected_ids.push(expert as u16);
            selected_scores.push(unbiased_scores[expert]);
        }
    }

    route_from_selected_scores(layer, token_count, &selected_ids, &selected_scores)
}

fn validate_routing_invariants(
    routing: &MiniMaxM3Routing,
) -> Result<(usize, usize), MiniMaxM3SemanticError> {
    if minimax_m3_layer_kind(routing.layer)? != MiniMaxM3LayerKind::SparseMoe {
        return Err(MiniMaxM3SemanticError::RoutingNotAllowedOnDenseLayer(
            routing.layer,
        ));
    }
    let tokens = checked_usize(routing.token_count, MiniMaxM3SemanticRole::SelectedExpertId)?;
    let top_k = MINIMAX_M3_SELECTED_EXPERT_COUNT as usize;
    let pairs = checked_product(tokens, top_k, MiniMaxM3SemanticRole::SelectedExpertId)?;
    if routing.expert_ids.len() != pairs {
        return Err(MiniMaxM3SemanticError::ElementCountMismatch {
            role: MiniMaxM3SemanticRole::SelectedExpertId,
            expected: pairs,
            actual: routing.expert_ids.len(),
        });
    }
    validate_finite(
        &routing.normalized_expert_weights,
        pairs,
        MiniMaxM3SemanticRole::SelectedRouterScore,
    )?;
    for token in 0..tokens {
        let start = token * top_k;
        let ids = &routing.expert_ids[start..start + top_k];
        for (slot, expert) in ids.iter().copied().enumerate() {
            if u32::from(expert) >= MINIMAX_M3_EXPERT_COUNT {
                return Err(MiniMaxM3SemanticError::ExpertOutOfRange {
                    token: token as u64,
                    expert: u32::from(expert),
                });
            }
            if ids[..slot].contains(&expert) {
                return Err(MiniMaxM3SemanticError::DuplicateExpert {
                    token: token as u64,
                    expert: u32::from(expert),
                });
            }
        }
        let weights = &routing.normalized_expert_weights[start..start + top_k];
        let sum: f32 = weights.iter().copied().sum();
        if weights.iter().any(|weight| *weight < 0.0)
            || !sum.is_finite()
            || (sum - 1.0).abs() > 2.0e-5
        {
            return Err(MiniMaxM3SemanticError::InvalidNormalizedWeightSum {
                token: token as u64,
                sum_bits: sum.to_bits(),
            });
        }
    }
    Ok((tokens, top_k))
}

/// Combine already-evaluated routed experts and the always-active shared expert.
///
/// `routed_expert_outputs` is row-major `[tokens, 4, hidden]` in the selected
/// slot order. `shared_expert_output` is `[tokens, hidden]`. The weighted routed
/// sum is multiplied by exactly 2.0 before the unscaled shared result is added.
pub fn reference_minimax_m3_sparse_moe_combine(
    routing: &MiniMaxM3Routing,
    hidden_size: u32,
    routed_expert_outputs: &[f32],
    shared_expert_output: &[f32],
) -> Result<Vec<f32>, MiniMaxM3SemanticError> {
    if hidden_size == 0 {
        return Err(MiniMaxM3SemanticError::InvalidHiddenSize(hidden_size));
    }
    let (tokens, top_k) = validate_routing_invariants(routing)?;
    let hidden = hidden_size as usize;
    let token_hidden = checked_product(tokens, hidden, MiniMaxM3SemanticRole::SharedExpertOutput)?;
    let routed_elements = checked_product(
        checked_product(tokens, top_k, MiniMaxM3SemanticRole::RoutedExpertOutput)?,
        hidden,
        MiniMaxM3SemanticRole::RoutedExpertOutput,
    )?;
    validate_finite(
        routed_expert_outputs,
        routed_elements,
        MiniMaxM3SemanticRole::RoutedExpertOutput,
    )?;
    validate_finite(
        shared_expert_output,
        token_hidden,
        MiniMaxM3SemanticRole::SharedExpertOutput,
    )?;
    let mut output = empty_vec(token_hidden, MiniMaxM3SemanticRole::CombinedOutput)?;

    for token in 0..tokens {
        let weight_start = token * top_k;
        for channel in 0..hidden {
            let mut routed = 0.0_f32;
            for slot in 0..top_k {
                let expert_index = (weight_start + slot) * hidden + channel;
                routed += routing.normalized_expert_weights[weight_start + slot]
                    * routed_expert_outputs[expert_index];
                if !routed.is_finite() {
                    return Err(MiniMaxM3SemanticError::NonFiniteIntermediate {
                        stage: MiniMaxM3SemanticStage::RoutedExpertSum,
                        index: token * hidden + channel,
                    });
                }
            }
            let scaled = routed * MINIMAX_M3_ROUTED_SCALING_FACTOR;
            if !scaled.is_finite() {
                return Err(MiniMaxM3SemanticError::NonFiniteIntermediate {
                    stage: MiniMaxM3SemanticStage::RoutedScaling,
                    index: token * hidden + channel,
                });
            }
            let merged = scaled + shared_expert_output[token * hidden + channel];
            if !merged.is_finite() {
                return Err(MiniMaxM3SemanticError::NonFiniteIntermediate {
                    stage: MiniMaxM3SemanticStage::SharedExpertMerge,
                    index: token * hidden + channel,
                });
            }
            output.push(merged);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 2.0e-5,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn exact_layer_schedule_covers_both_boundaries() {
        assert_eq!(minimax_m3_layer_kind(0), Ok(MiniMaxM3LayerKind::Dense));
        assert_eq!(minimax_m3_layer_kind(2), Ok(MiniMaxM3LayerKind::Dense));
        assert_eq!(minimax_m3_layer_kind(3), Ok(MiniMaxM3LayerKind::SparseMoe));
        assert_eq!(minimax_m3_layer_kind(59), Ok(MiniMaxM3LayerKind::SparseMoe));
        assert_eq!(
            minimax_m3_layer_kind(60),
            Err(MiniMaxM3SemanticError::TextLayerOutOfRange(60))
        );
        assert!(matches!(
            reference_minimax_m3_route(2, 1, &[0.0; 128], &[0.0; 128]),
            Err(MiniMaxM3SemanticError::RoutingNotAllowedOnDenseLayer(2))
        ));
    }

    #[test]
    fn stable_top4_and_selection_only_bias_match_official_router_ordering() {
        let tied = reference_minimax_m3_route(3, 1, &[0.0; 128], &[0.0; 128]).unwrap();
        assert_eq!(tied.expert_ids(), &[0, 1, 2, 3]);
        for weight in tied.normalized_expert_weights() {
            assert_close(*weight, 0.25);
        }

        let mut bias = [0.0_f32; 128];
        bias[127] = 1.0;
        let biased = reference_minimax_m3_route(59, 1, &[0.0; 128], &bias).unwrap();
        assert_eq!(biased.expert_ids(), &[127, 0, 1, 2]);
        for weight in biased.normalized_expert_weights() {
            assert_close(*weight, 0.25);
        }
        assert_close(biased.normalized_expert_weights().iter().sum(), 1.0);
    }

    #[test]
    fn router_selects_expert_zero_and_127_and_normalizes_unbiased_sigmoid() {
        let mut logits = [-10.0_f32; 128];
        logits[0] = 10.0;
        logits[127] = 9.0;
        logits[63] = 8.0;
        logits[64] = 7.0;
        let route = reference_minimax_m3_route(3, 1, &logits, &[0.0; 128]).unwrap();
        assert_eq!(route.expert_ids(), &[0, 127, 63, 64]);
        assert_close(route.normalized_expert_weights().iter().sum(), 1.0);
        assert_eq!(route.routed_scaling_factor(), 2.0);
    }

    #[test]
    fn selected_route_rejects_duplicate_out_of_range_nonfinite_and_zero_sum() {
        assert!(matches!(
            reference_minimax_m3_route_from_selection(3, 1, &[0, 0, 1, 2], &[0.5; 4]),
            Err(MiniMaxM3SemanticError::DuplicateExpert {
                token: 0,
                expert: 0
            })
        ));
        assert!(matches!(
            reference_minimax_m3_route_from_selection(3, 1, &[0, 1, 2, 128], &[0.5; 4]),
            Err(MiniMaxM3SemanticError::ExpertOutOfRange {
                token: 0,
                expert: 128
            })
        ));
        assert!(matches!(
            reference_minimax_m3_route_from_selection(
                3,
                1,
                &[0, 1, 2, 3],
                &[0.5, f32::NAN, 0.5, 0.5],
            ),
            Err(MiniMaxM3SemanticError::NonFiniteInput {
                role: MiniMaxM3SemanticRole::SelectedRouterScore,
                index: 1
            })
        ));
        assert!(matches!(
            reference_minimax_m3_route_from_selection(3, 1, &[0, 1, 2, 3], &[0.0; 4]),
            Err(MiniMaxM3SemanticError::ZeroNormalizer { token: 0 })
        ));
        assert!(matches!(
            reference_minimax_m3_route(3, 1, &[-f32::MAX; 128], &[0.0; 128]),
            Err(MiniMaxM3SemanticError::ZeroNormalizer { token: 0 })
        ));
    }

    #[test]
    fn router_rejects_nonfinite_inputs() {
        let mut logits = [0.0_f32; 128];
        logits[17] = f32::INFINITY;
        assert!(matches!(
            reference_minimax_m3_route(3, 1, &logits, &[0.0; 128]),
            Err(MiniMaxM3SemanticError::NonFiniteInput {
                role: MiniMaxM3SemanticRole::RouterLogit,
                index: 17
            })
        ));
        let mut bias = [0.0_f32; 128];
        bias[3] = f32::NAN;
        assert!(matches!(
            reference_minimax_m3_route(3, 1, &[0.0; 128], &bias),
            Err(MiniMaxM3SemanticError::NonFiniteInput {
                role: MiniMaxM3SemanticRole::RouterSelectionBias,
                index: 3
            })
        ));
    }

    #[test]
    fn routed_branch_is_scaled_before_unscaled_shared_expert_merge() {
        let route =
            reference_minimax_m3_route_from_selection(3, 1, &[0, 1, 2, 127], &[0.25; 4]).unwrap();
        let output = reference_minimax_m3_sparse_moe_combine(
            &route,
            2,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[10.0, 20.0],
        )
        .unwrap();
        assert_close(output[0], 18.0);
        assert_close(output[1], 30.0);
    }

    #[test]
    fn mtp_contract_fixes_identity_boundaries_and_refuses_full_generation() {
        let topology = MiniMaxM3MtpTopology::new(7, 1).unwrap();
        assert_eq!(topology.module_stage(0), Ok(MiniMaxM3MtpStage::Module(0)));
        assert_eq!(topology.module_stage(6), Ok(MiniMaxM3MtpStage::Module(6)));
        assert_eq!(
            topology.module_stage(7),
            Err(MiniMaxM3SemanticError::MtpModuleOutOfRange(7))
        );
        assert_eq!(
            topology.nextn_predict_layer_stage(0),
            Ok(MiniMaxM3MtpStage::NextNPredictLayer(0))
        );
        assert_eq!(
            topology.nextn_predict_layer_stage(1),
            Err(MiniMaxM3SemanticError::NextNPredictLayerOutOfRange(1))
        );
        assert!(matches!(
            MiniMaxM3MtpTopology::new(1, 1),
            Err(MiniMaxM3SemanticError::MtpTopologyMismatch { .. })
        ));
        assert_eq!(
            topology.admit_full_speculative_generation(),
            Err(MiniMaxM3SemanticError::UnsupportedFeature(
                MiniMaxM3UnsupportedFeature::FullMtpSpeculativeGeneration
            ))
        );
    }

    #[test]
    fn element_count_arithmetic_fails_closed_before_allocation() {
        assert!(matches!(
            reference_minimax_m3_route(3, u64::MAX, &[], &[0.0; 128]),
            Err(MiniMaxM3SemanticError::ElementCountOverflow {
                role: MiniMaxM3SemanticRole::RouterLogit
            })
        ));
        assert!(matches!(
            reference_minimax_m3_route_from_selection(3, u64::MAX, &[], &[]),
            Err(MiniMaxM3SemanticError::ElementCountOverflow {
                role: MiniMaxM3SemanticRole::SelectedExpertId
            })
        ));
    }
}
