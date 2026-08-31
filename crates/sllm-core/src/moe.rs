//! Backend-neutral sparse-MoE routing contracts and a host reference oracle.
//!
//! The reference implementation in this module exists for focused tests and
//! numerical evidence. Production Qwen3.5 MoE execution must make routing
//! decisions on the GPU and must not use this implementation as a fallback.

use std::fmt;

pub const QWEN35_MOE_EXPERT_COUNT: u32 = 256;
pub const QWEN35_MOE_SELECTED_EXPERT_COUNT: u32 = 8;

pub const GEMMA4_MOE_HIDDEN_SIZE: u32 = 2_816;
pub const GEMMA4_MOE_EXPERT_COUNT: u32 = 128;
pub const GEMMA4_MOE_SELECTED_EXPERT_COUNT: u32 = 8;
pub const GEMMA4_MOE_ROUTER_EPSILON: f32 = 1.0e-6;

/// Accumulator type fixed by the Gemma 4 router semantic contract.
///
/// RMSNorm square/sum, the root-hidden-size multiplication, the BF16 router
/// projection dot products, and the full-expert softmax reductions all use
/// FP32 arithmetic. BF16 stage boundaries are represented explicitly by the
/// host oracle rather than being inferred from this value.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Gemma4MoeRouterAccumulation {
    F32,
}

/// Gemma 4 sparse-MoE router semantics, independent of any GPU provider.
///
/// The reviewed 26B-A4B descriptor is available through
/// [`Self::gemma4_26b_a4b`]. `new` remains dimension-parameterized so focused
/// host oracles can exercise malformed dimensions and small numerical cases.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Gemma4MoeRouterDescriptor {
    hidden_size: u32,
    expert_count: u32,
    selected_expert_count: u32,
    rms_norm_epsilon_bits: u32,
    root_hidden_scale_bits: u32,
    accumulation: Gemma4MoeRouterAccumulation,
}

impl Gemma4MoeRouterDescriptor {
    pub fn new(
        hidden_size: u32,
        expert_count: u32,
        selected_expert_count: u32,
        rms_norm_epsilon: f32,
    ) -> Result<Self, Gemma4MoeRouterError> {
        if hidden_size == 0 {
            return Err(Gemma4MoeRouterError::InvalidHiddenSize(hidden_size));
        }
        if expert_count == 0 || expert_count > u16::MAX as u32 {
            return Err(Gemma4MoeRouterError::InvalidExpertCount(expert_count));
        }
        if selected_expert_count == 0 || selected_expert_count > expert_count {
            return Err(Gemma4MoeRouterError::InvalidSelectedExpertCount {
                expert_count,
                selected_expert_count,
            });
        }
        if !rms_norm_epsilon.is_finite() || rms_norm_epsilon <= 0.0 {
            return Err(Gemma4MoeRouterError::InvalidRmsNormEpsilon(
                rms_norm_epsilon.to_bits(),
            ));
        }
        let root_hidden_scale = (hidden_size as f32).powf(-0.5);
        if !root_hidden_scale.is_finite() || root_hidden_scale <= 0.0 {
            return Err(Gemma4MoeRouterError::InvalidRootHiddenScale);
        }
        Ok(Self {
            hidden_size,
            expert_count,
            selected_expert_count,
            rms_norm_epsilon_bits: rms_norm_epsilon.to_bits(),
            root_hidden_scale_bits: root_hidden_scale.to_bits(),
            accumulation: Gemma4MoeRouterAccumulation::F32,
        })
    }

    pub fn gemma4_26b_a4b() -> Self {
        Self::new(
            GEMMA4_MOE_HIDDEN_SIZE,
            GEMMA4_MOE_EXPERT_COUNT,
            GEMMA4_MOE_SELECTED_EXPERT_COUNT,
            GEMMA4_MOE_ROUTER_EPSILON,
        )
        .expect("the reviewed Gemma 4 26B-A4B router descriptor is valid")
    }

    pub const fn hidden_size(self) -> u32 {
        self.hidden_size
    }

    pub const fn expert_count(self) -> u32 {
        self.expert_count
    }

    pub const fn selected_expert_count(self) -> u32 {
        self.selected_expert_count
    }

    pub const fn rms_norm_has_learned_scale(self) -> bool {
        false
    }

    pub const fn rms_norm_epsilon(self) -> f32 {
        f32::from_bits(self.rms_norm_epsilon_bits)
    }

    pub const fn root_hidden_scale(self) -> f32 {
        f32::from_bits(self.root_hidden_scale_bits)
    }

    pub const fn rms_norm_accumulation(self) -> Gemma4MoeRouterAccumulation {
        self.accumulation
    }

    pub const fn root_scale_accumulation(self) -> Gemma4MoeRouterAccumulation {
        self.accumulation
    }

    pub const fn projection_accumulation(self) -> Gemma4MoeRouterAccumulation {
        self.accumulation
    }

    pub const fn softmax_accumulation(self) -> Gemma4MoeRouterAccumulation {
        self.accumulation
    }

    pub const fn renormalizes_selected_weights(self) -> bool {
        true
    }

    pub const fn applies_expert_scale_after_renormalization(self) -> bool {
        true
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gemma4MoeRouterTensor {
    Hidden,
    ElementwiseScale,
    Projection,
    PerExpertScale,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gemma4MoeRouterStage {
    RmsNorm,
    ElementwiseScale,
    RootHiddenScale,
    Projection,
    Softmax,
    TopKRenormalization,
    PerExpertScale,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4MoeRouterReference {
    token_count: u32,
    descriptor: Gemma4MoeRouterDescriptor,
    rms_normalized_hidden_bf16: Vec<u16>,
    elementwise_scaled_hidden_bf16: Vec<u16>,
    projection_input_bf16: Vec<u16>,
    router_logits_bf16: Vec<u16>,
    router_probabilities: Vec<f32>,
    expert_ids: Vec<u16>,
    normalized_topk_weights: Vec<f32>,
    expert_scaled_topk_weights: Vec<f32>,
}

impl Gemma4MoeRouterReference {
    pub const fn token_count(&self) -> u32 {
        self.token_count
    }

    pub const fn descriptor(&self) -> Gemma4MoeRouterDescriptor {
        self.descriptor
    }

    /// Scale-free RMSNorm output, rounded to BF16-RNE.
    pub fn rms_normalized_hidden_bf16(&self) -> &[u16] {
        &self.rms_normalized_hidden_bf16
    }

    /// Output after multiplication by the learned elementwise BF16 scale,
    /// rounded to BF16-RNE before applying the root-hidden-size scale.
    pub fn elementwise_scaled_hidden_bf16(&self) -> &[u16] {
        &self.elementwise_scaled_hidden_bf16
    }

    /// BF16-RNE input to the router projection after the FP32
    /// `hidden_size^-0.5` multiplication.
    pub fn projection_input_bf16(&self) -> &[u16] {
        &self.projection_input_bf16
    }

    /// BF16-RNE router projection output in row-major `[tokens, experts]`.
    pub fn router_logits_bf16(&self) -> &[u16] {
        &self.router_logits_bf16
    }

    /// Full-expert FP32 softmax probabilities in row-major
    /// `[tokens, experts]`.
    pub fn router_probabilities(&self) -> &[f32] {
        &self.router_probabilities
    }

    /// Stable probability-descending expert IDs with smaller IDs breaking
    /// exact ties, row-major `[tokens, top_k]`.
    pub fn expert_ids(&self) -> &[u16] {
        &self.expert_ids
    }

    /// Selected softmax probabilities renormalized to sum to one per token.
    pub fn normalized_topk_weights(&self) -> &[f32] {
        &self.normalized_topk_weights
    }

    /// Final routing weights after multiplying each normalized top-k weight by
    /// the corresponding BF16 per-expert scale. These do not generally sum to
    /// one and therefore intentionally do not use `SparseMoeRouting`'s final
    /// weight invariant.
    pub fn expert_scaled_topk_weights(&self) -> &[f32] {
        &self.expert_scaled_topk_weights
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Gemma4MoeRouterError {
    InvalidHiddenSize(u32),
    InvalidExpertCount(u32),
    InvalidSelectedExpertCount {
        expert_count: u32,
        selected_expert_count: u32,
    },
    InvalidRmsNormEpsilon(u32),
    InvalidRootHiddenScale,
    TokenCountZero,
    ElementCountOverflow {
        tensor: Gemma4MoeRouterTensor,
    },
    PairCountOverflow,
    ElementCountMismatch {
        tensor: Gemma4MoeRouterTensor,
        expected: usize,
        actual: usize,
    },
    NonFiniteInput {
        tensor: Gemma4MoeRouterTensor,
        index: usize,
    },
    NonFiniteIntermediate {
        stage: Gemma4MoeRouterStage,
        token: u32,
        index: u32,
    },
    InvalidSoftmax {
        token: u32,
    },
}

impl fmt::Display for Gemma4MoeRouterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Gemma 4 MoE router error: {self:?}")
    }
}

impl std::error::Error for Gemma4MoeRouterError {}

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

fn gemma4_bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn gemma4_f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let round_bias = 0x7fff_u32 + ((bits >> 16) & 1);
    bits.wrapping_add(round_bias).wrapping_shr(16) as u16
}

fn gemma4_checked_element_count(
    left: u32,
    right: u32,
    tensor: Gemma4MoeRouterTensor,
) -> Result<usize, Gemma4MoeRouterError> {
    let left =
        usize::try_from(left).map_err(|_| Gemma4MoeRouterError::ElementCountOverflow { tensor })?;
    let right = usize::try_from(right)
        .map_err(|_| Gemma4MoeRouterError::ElementCountOverflow { tensor })?;
    left.checked_mul(right)
        .ok_or(Gemma4MoeRouterError::ElementCountOverflow { tensor })
}

fn gemma4_validate_bf16_input(
    tensor: Gemma4MoeRouterTensor,
    values: &[u16],
    expected: usize,
) -> Result<(), Gemma4MoeRouterError> {
    if values.len() != expected {
        return Err(Gemma4MoeRouterError::ElementCountMismatch {
            tensor,
            expected,
            actual: values.len(),
        });
    }
    for (index, bits) in values.iter().copied().enumerate() {
        if !gemma4_bf16_to_f32(bits).is_finite() {
            return Err(Gemma4MoeRouterError::NonFiniteInput { tensor, index });
        }
    }
    Ok(())
}

fn gemma4_checked_bf16_round(
    value: f32,
    stage: Gemma4MoeRouterStage,
    token: usize,
    index: usize,
) -> Result<u16, Gemma4MoeRouterError> {
    if !value.is_finite() {
        return Err(Gemma4MoeRouterError::NonFiniteIntermediate {
            stage,
            token: token as u32,
            index: index as u32,
        });
    }
    let rounded = gemma4_f32_to_bf16_rne(value);
    if !gemma4_bf16_to_f32(rounded).is_finite() {
        return Err(Gemma4MoeRouterError::NonFiniteIntermediate {
            stage,
            token: token as u32,
            index: index as u32,
        });
    }
    Ok(rounded)
}

/// Independent host reference for the Gemma 4 sparse-MoE router.
///
/// Input layouts are BF16-bit row-major `hidden[tokens, hidden]`, learned
/// elementwise `scale[hidden]`, projection `weight[experts, hidden]`, and
/// `per_expert_scale[experts]`. The exact stage order is:
///
/// 1. scale-free RMSNorm with epsilon and an ordered FP32 square/sum;
/// 2. BF16-RNE, learned BF16 elementwise scale in FP32, then BF16-RNE;
/// 3. FP32 `hidden_size^-0.5` multiplication, then BF16-RNE;
/// 4. BF16 projection operands with ordered FP32 dot-product accumulation,
///    followed by BF16-RNE logits;
/// 5. stable all-expert softmax with FP32 exponent/sum/divide;
/// 6. stable top-k, selected-weight FP32 renormalization to one;
/// 7. multiplication by the selected expert's BF16 scale.
///
/// Per-expert scale is deliberately after selection and renormalization; it
/// cannot affect expert identity and final weights need not sum to one.
pub fn reference_gemma4_moe_route(
    descriptor: Gemma4MoeRouterDescriptor,
    token_count: u32,
    hidden_bf16: &[u16],
    elementwise_scale_bf16: &[u16],
    projection_bf16: &[u16],
    per_expert_scale_bf16: &[u16],
) -> Result<Gemma4MoeRouterReference, Gemma4MoeRouterError> {
    if token_count == 0 {
        return Err(Gemma4MoeRouterError::TokenCountZero);
    }
    let hidden_size = usize::try_from(descriptor.hidden_size).unwrap();
    let experts = usize::try_from(descriptor.expert_count).unwrap();
    let topk = usize::try_from(descriptor.selected_expert_count).unwrap();
    let tokens =
        usize::try_from(token_count).map_err(|_| Gemma4MoeRouterError::ElementCountOverflow {
            tensor: Gemma4MoeRouterTensor::Hidden,
        })?;

    token_count
        .checked_mul(descriptor.selected_expert_count)
        .ok_or(Gemma4MoeRouterError::PairCountOverflow)?;
    let hidden_elements = gemma4_checked_element_count(
        token_count,
        descriptor.hidden_size,
        Gemma4MoeRouterTensor::Hidden,
    )?;
    let projection_elements = gemma4_checked_element_count(
        descriptor.expert_count,
        descriptor.hidden_size,
        Gemma4MoeRouterTensor::Projection,
    )?;
    let logit_elements = gemma4_checked_element_count(
        token_count,
        descriptor.expert_count,
        Gemma4MoeRouterTensor::Projection,
    )?;
    let pair_elements = tokens
        .checked_mul(topk)
        .ok_or(Gemma4MoeRouterError::PairCountOverflow)?;

    gemma4_validate_bf16_input(Gemma4MoeRouterTensor::Hidden, hidden_bf16, hidden_elements)?;
    gemma4_validate_bf16_input(
        Gemma4MoeRouterTensor::ElementwiseScale,
        elementwise_scale_bf16,
        hidden_size,
    )?;
    gemma4_validate_bf16_input(
        Gemma4MoeRouterTensor::Projection,
        projection_bf16,
        projection_elements,
    )?;
    gemma4_validate_bf16_input(
        Gemma4MoeRouterTensor::PerExpertScale,
        per_expert_scale_bf16,
        experts,
    )?;

    let mut rms_normalized_hidden_bf16 = Vec::with_capacity(hidden_elements);
    let mut elementwise_scaled_hidden_bf16 = Vec::with_capacity(hidden_elements);
    let mut projection_input_bf16 = Vec::with_capacity(hidden_elements);
    for token in 0..tokens {
        let row = &hidden_bf16[token * hidden_size..(token + 1) * hidden_size];
        let mut square_sum = 0.0_f32;
        for (lane, bits) in row.iter().copied().enumerate() {
            let value = gemma4_bf16_to_f32(bits);
            square_sum += value * value;
            if !square_sum.is_finite() {
                return Err(Gemma4MoeRouterError::NonFiniteIntermediate {
                    stage: Gemma4MoeRouterStage::RmsNorm,
                    token: token as u32,
                    index: lane as u32,
                });
            }
        }
        let mean_square = square_sum / hidden_size as f32;
        let inverse_root_mean_square = (mean_square + descriptor.rms_norm_epsilon()).powf(-0.5);
        if !inverse_root_mean_square.is_finite() || inverse_root_mean_square <= 0.0 {
            return Err(Gemma4MoeRouterError::NonFiniteIntermediate {
                stage: Gemma4MoeRouterStage::RmsNorm,
                token: token as u32,
                index: 0,
            });
        }
        for (lane, bits) in row.iter().copied().enumerate() {
            let normalized = gemma4_checked_bf16_round(
                gemma4_bf16_to_f32(bits) * inverse_root_mean_square,
                Gemma4MoeRouterStage::RmsNorm,
                token,
                lane,
            )?;
            rms_normalized_hidden_bf16.push(normalized);

            let elementwise_scaled = gemma4_checked_bf16_round(
                gemma4_bf16_to_f32(normalized) * gemma4_bf16_to_f32(elementwise_scale_bf16[lane]),
                Gemma4MoeRouterStage::ElementwiseScale,
                token,
                lane,
            )?;
            elementwise_scaled_hidden_bf16.push(elementwise_scaled);

            let projection_input = gemma4_checked_bf16_round(
                gemma4_bf16_to_f32(elementwise_scaled) * descriptor.root_hidden_scale(),
                Gemma4MoeRouterStage::RootHiddenScale,
                token,
                lane,
            )?;
            projection_input_bf16.push(projection_input);
        }
    }

    let mut router_logits_bf16 = Vec::with_capacity(logit_elements);
    for token in 0..tokens {
        let activation = &projection_input_bf16[token * hidden_size..(token + 1) * hidden_size];
        for expert in 0..experts {
            let weight = &projection_bf16[expert * hidden_size..(expert + 1) * hidden_size];
            let mut logit = 0.0_f32;
            for lane in 0..hidden_size {
                logit += gemma4_bf16_to_f32(activation[lane]) * gemma4_bf16_to_f32(weight[lane]);
                if !logit.is_finite() {
                    return Err(Gemma4MoeRouterError::NonFiniteIntermediate {
                        stage: Gemma4MoeRouterStage::Projection,
                        token: token as u32,
                        index: expert as u32,
                    });
                }
            }
            router_logits_bf16.push(gemma4_checked_bf16_round(
                logit,
                Gemma4MoeRouterStage::Projection,
                token,
                expert,
            )?);
        }
    }

    let mut router_probabilities = Vec::with_capacity(logit_elements);
    let mut expert_ids = Vec::with_capacity(pair_elements);
    let mut normalized_topk_weights = Vec::with_capacity(pair_elements);
    let mut expert_scaled_topk_weights = Vec::with_capacity(pair_elements);
    for token in 0..tokens {
        let logits = &router_logits_bf16[token * experts..(token + 1) * experts];
        let maximum = logits
            .iter()
            .copied()
            .map(gemma4_bf16_to_f32)
            .fold(f32::NEG_INFINITY, f32::max);
        let probability_start = router_probabilities.len();
        let mut denominator = 0.0_f32;
        for (expert, bits) in logits.iter().copied().enumerate() {
            let exponential = (gemma4_bf16_to_f32(bits) - maximum).exp();
            denominator += exponential;
            if !exponential.is_finite() || !denominator.is_finite() {
                return Err(Gemma4MoeRouterError::NonFiniteIntermediate {
                    stage: Gemma4MoeRouterStage::Softmax,
                    token: token as u32,
                    index: expert as u32,
                });
            }
            router_probabilities.push(exponential);
        }
        if denominator <= 0.0 {
            return Err(Gemma4MoeRouterError::InvalidSoftmax {
                token: token as u32,
            });
        }
        let probabilities =
            &mut router_probabilities[probability_start..probability_start + experts];
        for (expert, probability) in probabilities.iter_mut().enumerate() {
            *probability /= denominator;
            if !probability.is_finite() {
                return Err(Gemma4MoeRouterError::NonFiniteIntermediate {
                    stage: Gemma4MoeRouterStage::Softmax,
                    token: token as u32,
                    index: expert as u32,
                });
            }
        }

        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_unstable_by(|left, right| {
            probabilities[*right]
                .total_cmp(&probabilities[*left])
                .then_with(|| left.cmp(right))
        });
        let selected = &order[..topk];
        let mut selected_sum = 0.0_f32;
        for expert in selected {
            selected_sum += probabilities[*expert];
        }
        if !selected_sum.is_finite() || selected_sum <= 0.0 {
            return Err(Gemma4MoeRouterError::NonFiniteIntermediate {
                stage: Gemma4MoeRouterStage::TopKRenormalization,
                token: token as u32,
                index: 0,
            });
        }
        for expert in selected {
            let normalized = probabilities[*expert] / selected_sum;
            if !normalized.is_finite() {
                return Err(Gemma4MoeRouterError::NonFiniteIntermediate {
                    stage: Gemma4MoeRouterStage::TopKRenormalization,
                    token: token as u32,
                    index: *expert as u32,
                });
            }
            let scaled = normalized * gemma4_bf16_to_f32(per_expert_scale_bf16[*expert]);
            if !scaled.is_finite() {
                return Err(Gemma4MoeRouterError::NonFiniteIntermediate {
                    stage: Gemma4MoeRouterStage::PerExpertScale,
                    token: token as u32,
                    index: *expert as u32,
                });
            }
            expert_ids.push(*expert as u16);
            normalized_topk_weights.push(normalized);
            expert_scaled_topk_weights.push(scaled);
        }
    }

    Ok(Gemma4MoeRouterReference {
        token_count,
        descriptor,
        rms_normalized_hidden_bf16,
        elementwise_scaled_hidden_bf16,
        projection_input_bf16,
        router_logits_bf16,
        router_probabilities,
        expert_ids,
        normalized_topk_weights,
        expert_scaled_topk_weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(value: f32) -> u16 {
        gemma4_f32_to_bf16_rne(value)
    }

    fn gemma4_inputs(
        descriptor: Gemma4MoeRouterDescriptor,
        token_count: usize,
    ) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let hidden = vec![bf16(0.0); token_count * descriptor.hidden_size() as usize];
        let scale = vec![bf16(1.0); descriptor.hidden_size() as usize];
        let projection =
            vec![bf16(0.0); descriptor.expert_count() as usize * descriptor.hidden_size() as usize];
        let per_expert_scale = vec![bf16(1.0); descriptor.expert_count() as usize];
        (hidden, scale, projection, per_expert_scale)
    }

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

    #[test]
    fn gemma4_moe_descriptor_fixes_reviewed_semantics_and_f32_accumulation() {
        let descriptor = Gemma4MoeRouterDescriptor::gemma4_26b_a4b();
        assert_eq!(descriptor.hidden_size(), 2_816);
        assert_eq!(descriptor.expert_count(), 128);
        assert_eq!(descriptor.selected_expert_count(), 8);
        assert_eq!(descriptor.rms_norm_epsilon(), 1.0e-6);
        assert_eq!(descriptor.root_hidden_scale(), (2_816.0_f32).powf(-0.5));
        assert!(!descriptor.rms_norm_has_learned_scale());
        assert!(descriptor.renormalizes_selected_weights());
        assert!(descriptor.applies_expert_scale_after_renormalization());
        assert_eq!(
            descriptor.rms_norm_accumulation(),
            Gemma4MoeRouterAccumulation::F32
        );
        assert_eq!(
            descriptor.root_scale_accumulation(),
            Gemma4MoeRouterAccumulation::F32
        );
        assert_eq!(
            descriptor.projection_accumulation(),
            Gemma4MoeRouterAccumulation::F32
        );
        assert_eq!(
            descriptor.softmax_accumulation(),
            Gemma4MoeRouterAccumulation::F32
        );
    }

    #[test]
    fn gemma4_moe_reference_exposes_bf16_boundaries_and_full_softmax() {
        let descriptor = Gemma4MoeRouterDescriptor::new(4, 128, 8, 1.0e-6).unwrap();
        let (mut hidden, mut scale, mut projection, per_expert_scale) =
            gemma4_inputs(descriptor, 1);
        hidden.copy_from_slice(&[bf16(3.0), bf16(4.0), bf16(0.0), bf16(0.0)]);
        scale[0] = bf16(2.0);
        projection[0] = bf16(1.0);
        projection[127 * 4 + 1] = bf16(1.0);

        let route = reference_gemma4_moe_route(
            descriptor,
            1,
            &hidden,
            &scale,
            &projection,
            &per_expert_scale,
        )
        .unwrap();

        let inverse_rms = ((3.0_f32 * 3.0 + 4.0 * 4.0) / 4.0 + 1.0e-6).powf(-0.5);
        let normalized_lane_0 = bf16(3.0 * inverse_rms);
        let normalized_lane_1 = bf16(4.0 * inverse_rms);
        assert_eq!(route.rms_normalized_hidden_bf16()[0], normalized_lane_0);
        assert_eq!(route.rms_normalized_hidden_bf16()[1], normalized_lane_1);
        let scaled_lane_0 = bf16(gemma4_bf16_to_f32(normalized_lane_0) * 2.0);
        assert_eq!(route.elementwise_scaled_hidden_bf16()[0], scaled_lane_0);
        assert_eq!(
            route.projection_input_bf16()[0],
            bf16(gemma4_bf16_to_f32(scaled_lane_0) * 0.5)
        );
        assert_eq!(route.router_logits_bf16().len(), 128);
        assert_eq!(route.router_probabilities().len(), 128);
        let probability_sum: f32 = route.router_probabilities().iter().sum();
        assert!(
            (probability_sum - 1.0).abs() <= 1.0e-5,
            "full softmax sum was {probability_sum}"
        );
        assert_eq!(route.expert_ids()[0], 0);
        assert!(route.expert_ids().contains(&127));
    }

    #[test]
    fn gemma4_moe_stable_top8_covers_aligned_and_non_aligned_token_counts() {
        let descriptor = Gemma4MoeRouterDescriptor::gemma4_26b_a4b();
        for token_count in [1_usize, 3, 7, 8, 17, 31, 32, 33] {
            let (hidden, scale, projection, per_expert_scale) =
                gemma4_inputs(descriptor, token_count);
            let route = reference_gemma4_moe_route(
                descriptor,
                token_count as u32,
                &hidden,
                &scale,
                &projection,
                &per_expert_scale,
            )
            .unwrap();
            assert_eq!(route.token_count(), token_count as u32);
            assert_eq!(route.descriptor(), descriptor);
            assert_eq!(route.expert_ids().len(), token_count * 8);
            for token in 0..token_count {
                let begin = token * 8;
                assert_eq!(
                    &route.expert_ids()[begin..begin + 8],
                    &[0, 1, 2, 3, 4, 5, 6, 7]
                );
                let sum: f32 = route.normalized_topk_weights()[begin..begin + 8]
                    .iter()
                    .sum();
                assert!((sum - 1.0).abs() <= 2.0e-6);
            }
        }
    }

    #[test]
    fn gemma4_moe_stable_routing_reaches_expert_zero_and_127() {
        let descriptor = Gemma4MoeRouterDescriptor::new(4, 128, 8, 1.0e-6).unwrap();
        let (mut hidden, scale, mut projection, per_expert_scale) = gemma4_inputs(descriptor, 2);
        hidden[0] = bf16(1.0);
        hidden[4] = bf16(-1.0);
        projection[0] = bf16(-1.0);
        projection[127 * 4] = bf16(1.0);

        let route = reference_gemma4_moe_route(
            descriptor,
            2,
            &hidden,
            &scale,
            &projection,
            &per_expert_scale,
        )
        .unwrap();
        assert_eq!(&route.expert_ids()[..8], &[127, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(&route.expert_ids()[8..], &[0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn gemma4_moe_expert_scale_is_after_topk_renormalization() {
        let descriptor = Gemma4MoeRouterDescriptor::new(4, 128, 8, 1.0e-6).unwrap();
        let (hidden, scale, projection, mut per_expert_scale) = gemma4_inputs(descriptor, 1);
        per_expert_scale[0] = bf16(2.0);
        per_expert_scale[1] = bf16(4.0);
        per_expert_scale[127] = bf16(256.0);

        let route = reference_gemma4_moe_route(
            descriptor,
            1,
            &hidden,
            &scale,
            &projection,
            &per_expert_scale,
        )
        .unwrap();
        assert_eq!(route.expert_ids(), &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(
            route
                .normalized_topk_weights()
                .iter()
                .all(|weight| (*weight - 0.125).abs() <= f32::EPSILON)
        );
        assert_eq!(
            route.expert_scaled_topk_weights(),
            &[0.25, 0.5, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
        );
        let normalized_sum: f32 = route.normalized_topk_weights().iter().sum();
        let scaled_sum: f32 = route.expert_scaled_topk_weights().iter().sum();
        assert_eq!(normalized_sum, 1.0);
        assert_eq!(scaled_sum, 1.5);
    }

    #[test]
    fn gemma4_moe_rejects_nan_and_infinity_in_every_input_role() {
        let descriptor = Gemma4MoeRouterDescriptor::new(4, 128, 8, 1.0e-6).unwrap();
        let (hidden, scale, projection, per_expert_scale) = gemma4_inputs(descriptor, 1);

        let mut invalid = hidden.clone();
        invalid[3] = 0x7fc0;
        assert_eq!(
            reference_gemma4_moe_route(
                descriptor,
                1,
                &invalid,
                &scale,
                &projection,
                &per_expert_scale
            ),
            Err(Gemma4MoeRouterError::NonFiniteInput {
                tensor: Gemma4MoeRouterTensor::Hidden,
                index: 3
            })
        );

        let mut invalid = scale.clone();
        invalid[1] = 0x7f80;
        assert_eq!(
            reference_gemma4_moe_route(
                descriptor,
                1,
                &hidden,
                &invalid,
                &projection,
                &per_expert_scale
            ),
            Err(Gemma4MoeRouterError::NonFiniteInput {
                tensor: Gemma4MoeRouterTensor::ElementwiseScale,
                index: 1
            })
        );

        let mut invalid = projection.clone();
        invalid[127 * 4 + 2] = 0xff80;
        assert_eq!(
            reference_gemma4_moe_route(descriptor, 1, &hidden, &scale, &invalid, &per_expert_scale),
            Err(Gemma4MoeRouterError::NonFiniteInput {
                tensor: Gemma4MoeRouterTensor::Projection,
                index: 127 * 4 + 2
            })
        );

        let mut invalid = per_expert_scale.clone();
        invalid[127] = 0x7fc0;
        assert_eq!(
            reference_gemma4_moe_route(descriptor, 1, &hidden, &scale, &projection, &invalid),
            Err(Gemma4MoeRouterError::NonFiniteInput {
                tensor: Gemma4MoeRouterTensor::PerExpertScale,
                index: 127
            })
        );
    }

    #[test]
    fn gemma4_moe_rejects_zero_overflow_malformed_and_intermediate_nonfinite() {
        assert_eq!(
            Gemma4MoeRouterDescriptor::new(0, 128, 8, 1.0e-6),
            Err(Gemma4MoeRouterError::InvalidHiddenSize(0))
        );
        assert_eq!(
            Gemma4MoeRouterDescriptor::new(4, 0, 8, 1.0e-6),
            Err(Gemma4MoeRouterError::InvalidExpertCount(0))
        );
        assert_eq!(
            Gemma4MoeRouterDescriptor::new(4, u32::MAX, 8, 1.0e-6),
            Err(Gemma4MoeRouterError::InvalidExpertCount(u32::MAX))
        );
        assert_eq!(
            Gemma4MoeRouterDescriptor::new(4, 128, 0, 1.0e-6),
            Err(Gemma4MoeRouterError::InvalidSelectedExpertCount {
                expert_count: 128,
                selected_expert_count: 0
            })
        );
        assert_eq!(
            Gemma4MoeRouterDescriptor::new(4, 128, 129, 1.0e-6),
            Err(Gemma4MoeRouterError::InvalidSelectedExpertCount {
                expert_count: 128,
                selected_expert_count: 129
            })
        );
        assert_eq!(
            Gemma4MoeRouterDescriptor::new(4, 128, 8, 0.0),
            Err(Gemma4MoeRouterError::InvalidRmsNormEpsilon(
                0.0_f32.to_bits()
            ))
        );

        let descriptor = Gemma4MoeRouterDescriptor::new(4, 128, 8, 1.0e-6).unwrap();
        let (hidden, scale, projection, per_expert_scale) = gemma4_inputs(descriptor, 1);
        assert_eq!(
            reference_gemma4_moe_route(descriptor, 0, &[], &scale, &projection, &per_expert_scale),
            Err(Gemma4MoeRouterError::TokenCountZero)
        );
        assert_eq!(
            reference_gemma4_moe_route(
                descriptor,
                u32::MAX,
                &[],
                &scale,
                &projection,
                &per_expert_scale
            ),
            Err(Gemma4MoeRouterError::PairCountOverflow)
        );
        assert_eq!(
            reference_gemma4_moe_route(
                descriptor,
                1,
                &hidden[..3],
                &scale,
                &projection,
                &per_expert_scale
            ),
            Err(Gemma4MoeRouterError::ElementCountMismatch {
                tensor: Gemma4MoeRouterTensor::Hidden,
                expected: 4,
                actual: 3
            })
        );

        let hidden = vec![bf16(1.0); 4];
        let scale = vec![0x7f7f; 4];
        let projection = vec![0x7f7f; 128 * 4];
        assert!(matches!(
            reference_gemma4_moe_route(
                descriptor,
                1,
                &hidden,
                &scale,
                &projection,
                &per_expert_scale
            ),
            Err(Gemma4MoeRouterError::NonFiniteIntermediate {
                stage: Gemma4MoeRouterStage::Projection,
                token: 0,
                index: 0
            })
        ));
    }
}
