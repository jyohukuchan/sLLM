//! Pure host semantics for the reviewed DeepSeek V4 Flash foundation.
//!
//! These references deliberately contain no container or backend behavior.
//! They freeze routing, mHC mixing, and compressed-attention boundary
//! semantics for focused tests and numerical evidence; they are not a CPU
//! fallback for production inference.

use std::fmt;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeepSeekV4SemanticRole {
    RouterLogit,
    RouterSelectionBias,
    HashTokenId,
    HashExpertId,
    Stream,
    PreGateLogit,
    PostGateLogit,
    MixingLogit,
    OperatorOutput,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeepSeekV4SemanticStage {
    RouterScore,
    RouterSelection,
    RouterWeight,
    PreGate,
    PreCollapse,
    MixingSoftmax,
    SinkhornRow,
    SinkhornColumn,
    PostGate,
    RecurrentMix,
}

/// Architectural location of a router invocation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DeepSeekV4RouterLocation {
    MainLayer(u32),
    DSparkStage(u32),
    NextN,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeepSeekV4SemanticError {
    InvalidTokenCount(u32),
    InvalidExpertCount(u32),
    InvalidSelectedExpertCount {
        expert_count: u32,
        selected_expert_count: u32,
    },
    InvalidStreamCount(u32),
    InvalidHiddenSize(u32),
    InvalidEpsilon(u32),
    InvalidSinkhornIterations(u32),
    InvalidRoutedScale(u32),
    InvalidCompressionRatio(u32),
    ElementCountOverflow {
        role: DeepSeekV4SemanticRole,
    },
    AllocationFailed {
        role: DeepSeekV4SemanticRole,
    },
    ElementCountMismatch {
        role: DeepSeekV4SemanticRole,
        expected: usize,
        actual: usize,
    },
    NonFiniteInput {
        role: DeepSeekV4SemanticRole,
        index: usize,
    },
    NonFiniteIntermediate {
        stage: DeepSeekV4SemanticStage,
        index: usize,
    },
    InvalidWeightSum {
        token: u32,
    },
    DuplicateHashExpert {
        token: u32,
        expert: u32,
    },
    HashExpertOutOfRange {
        token: u32,
        expert: u32,
        expert_count: u32,
    },
    HashRoutingNotAllowed {
        location: DeepSeekV4RouterLocation,
    },
    PositionOverflow,
}

impl fmt::Display for DeepSeekV4SemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "DeepSeek V4 semantic error: {self:?}")
    }
}

impl std::error::Error for DeepSeekV4SemanticError {}

#[derive(Clone, Debug, PartialEq)]
pub struct DeepSeekV4Routing {
    token_count: u32,
    expert_count: u32,
    selected_expert_count: u32,
    expert_ids: Vec<u16>,
    expert_weights: Vec<f32>,
}

impl DeepSeekV4Routing {
    pub const fn token_count(&self) -> u32 {
        self.token_count
    }

    pub const fn expert_count(&self) -> u32 {
        self.expert_count
    }

    pub const fn selected_expert_count(&self) -> u32 {
        self.selected_expert_count
    }

    /// Stable selected IDs in row-major `[tokens, top_k]` order.
    pub fn expert_ids(&self) -> &[u16] {
        &self.expert_ids
    }

    /// Weights made from unbiased scores, with optional selected-score
    /// renormalization followed by the routed scale.
    pub fn expert_weights(&self) -> &[f32] {
        &self.expert_weights
    }
}

fn checked_product(
    left: usize,
    right: usize,
    role: DeepSeekV4SemanticRole,
) -> Result<usize, DeepSeekV4SemanticError> {
    left.checked_mul(right)
        .ok_or(DeepSeekV4SemanticError::ElementCountOverflow { role })
}

fn empty_vec<T>(
    capacity: usize,
    role: DeepSeekV4SemanticRole,
) -> Result<Vec<T>, DeepSeekV4SemanticError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| DeepSeekV4SemanticError::AllocationFailed { role })?;
    Ok(values)
}

fn validate_count(
    values: &[f32],
    expected: usize,
    role: DeepSeekV4SemanticRole,
) -> Result<(), DeepSeekV4SemanticError> {
    if values.len() != expected {
        return Err(DeepSeekV4SemanticError::ElementCountMismatch {
            role,
            expected,
            actual: values.len(),
        });
    }
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(DeepSeekV4SemanticError::NonFiniteInput { role, index });
        }
    }
    Ok(())
}

fn validate_routing_shape(
    token_count: u32,
    expert_count: u32,
    selected_expert_count: u32,
) -> Result<(usize, usize, usize), DeepSeekV4SemanticError> {
    if token_count == 0 {
        return Err(DeepSeekV4SemanticError::InvalidTokenCount(token_count));
    }
    if expert_count == 0 || expert_count > u16::MAX as u32 {
        return Err(DeepSeekV4SemanticError::InvalidExpertCount(expert_count));
    }
    if selected_expert_count == 0 || selected_expert_count > expert_count {
        return Err(DeepSeekV4SemanticError::InvalidSelectedExpertCount {
            expert_count,
            selected_expert_count,
        });
    }
    Ok((
        usize::try_from(token_count).expect("u32 token count fits usize"),
        usize::try_from(expert_count).expect("validated expert count fits usize"),
        usize::try_from(selected_expert_count).expect("validated selected expert count fits usize"),
    ))
}

/// Numerically stable FP32 `sqrt(softplus(logit))` used by both routers.
fn router_score(logit: f32, index: usize) -> Result<f32, DeepSeekV4SemanticError> {
    let softplus = if logit > 0.0 {
        logit + (-logit).exp().ln_1p()
    } else {
        logit.exp().ln_1p()
    };
    let score = softplus.sqrt();
    if !score.is_finite() {
        return Err(DeepSeekV4SemanticError::NonFiniteIntermediate {
            stage: DeepSeekV4SemanticStage::RouterScore,
            index,
        });
    }
    Ok(score)
}

fn selected_weights(
    token: usize,
    expert_ids: &[u16],
    unbiased_scores: &[f32],
    renormalize: bool,
    routed_scale: f32,
    output: &mut Vec<f32>,
) -> Result<(), DeepSeekV4SemanticError> {
    let denominator = if renormalize {
        let mut sum = 0.0_f32;
        for expert in expert_ids {
            sum += unbiased_scores[usize::from(*expert)];
            if !sum.is_finite() {
                return Err(DeepSeekV4SemanticError::NonFiniteIntermediate {
                    stage: DeepSeekV4SemanticStage::RouterWeight,
                    index: token,
                });
            }
        }
        if sum <= 0.0 {
            return Err(DeepSeekV4SemanticError::InvalidWeightSum {
                token: token as u32,
            });
        }
        sum
    } else {
        1.0
    };

    for expert in expert_ids {
        let weight = unbiased_scores[usize::from(*expert)] / denominator * routed_scale;
        if !weight.is_finite() {
            return Err(DeepSeekV4SemanticError::NonFiniteIntermediate {
                stage: DeepSeekV4SemanticStage::RouterWeight,
                index: output.len(),
            });
        }
        output.push(weight);
    }
    Ok(())
}

/// Independent score-router reference.
///
/// Selection uses `sqrt(softplus(logit)) + selection_bias`; returned weights
/// always use the unbiased `sqrt(softplus(logit))` score. Exact selection ties
/// are resolved by the smaller expert ID.
pub fn reference_deepseek_v4_score_route(
    token_count: u32,
    expert_count: u32,
    selected_expert_count: u32,
    logits: &[f32],
    selection_bias: &[f32],
    renormalize: bool,
    routed_scale: f32,
) -> Result<DeepSeekV4Routing, DeepSeekV4SemanticError> {
    let (tokens, experts, top_k) =
        validate_routing_shape(token_count, expert_count, selected_expert_count)?;
    if !routed_scale.is_finite() || routed_scale <= 0.0 {
        return Err(DeepSeekV4SemanticError::InvalidRoutedScale(
            routed_scale.to_bits(),
        ));
    }
    let logit_count = checked_product(tokens, experts, DeepSeekV4SemanticRole::RouterLogit)?;
    validate_count(logits, logit_count, DeepSeekV4SemanticRole::RouterLogit)?;
    validate_count(
        selection_bias,
        experts,
        DeepSeekV4SemanticRole::RouterSelectionBias,
    )?;
    let pair_count = checked_product(tokens, top_k, DeepSeekV4SemanticRole::HashExpertId)?;
    let mut expert_ids = empty_vec(pair_count, DeepSeekV4SemanticRole::HashExpertId)?;
    let mut expert_weights = empty_vec(pair_count, DeepSeekV4SemanticRole::RouterLogit)?;
    let mut unbiased_scores = empty_vec(experts, DeepSeekV4SemanticRole::RouterLogit)?;
    let mut selection_scores = empty_vec(experts, DeepSeekV4SemanticRole::RouterSelectionBias)?;
    let mut order = empty_vec(experts, DeepSeekV4SemanticRole::HashExpertId)?;

    for token in 0..tokens {
        unbiased_scores.clear();
        selection_scores.clear();
        order.clear();
        for (expert, selection_bias) in selection_bias.iter().copied().enumerate().take(experts) {
            let flat_index = token * experts + expert;
            let score = router_score(logits[flat_index], flat_index)?;
            let selection_score = score + selection_bias;
            if !selection_score.is_finite() {
                return Err(DeepSeekV4SemanticError::NonFiniteIntermediate {
                    stage: DeepSeekV4SemanticStage::RouterSelection,
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
        let selected_start = expert_ids.len();
        for expert in order.iter().copied().take(top_k) {
            expert_ids.push(expert as u16);
        }
        selected_weights(
            token,
            &expert_ids[selected_start..],
            &unbiased_scores,
            renormalize,
            routed_scale,
            &mut expert_weights,
        )?;
    }

    Ok(DeepSeekV4Routing {
        token_count,
        expert_count,
        selected_expert_count,
        expert_ids,
        expert_weights,
    })
}

/// Independent token-ID hash-router validation and weighting reference.
///
/// `hashed_expert_ids` is the already derived row-major `[tokens, top_k]`
/// result of the reviewed token-ID hash. This function intentionally does not
/// invent or substitute a hash algorithm: it validates the identity-bearing
/// token rows, rejects duplicate/out-of-range experts, and computes weights
/// from the same unbiased score as the score router. Hash routing is valid
/// only for main layers `0..=2` (the first three main layers); DSpark stages,
/// NextN, and later main layers fail closed.
#[allow(clippy::too_many_arguments)] // Mirrors the score-route oracle's explicit contract inputs.
pub fn reference_deepseek_v4_hash_route(
    token_ids: &[u64],
    location: DeepSeekV4RouterLocation,
    expert_count: u32,
    selected_expert_count: u32,
    logits: &[f32],
    hashed_expert_ids: &[u16],
    renormalize: bool,
    routed_scale: f32,
) -> Result<DeepSeekV4Routing, DeepSeekV4SemanticError> {
    if !matches!(location, DeepSeekV4RouterLocation::MainLayer(0..=2)) {
        return Err(DeepSeekV4SemanticError::HashRoutingNotAllowed { location });
    }
    let token_count = u32::try_from(token_ids.len()).map_err(|_| {
        DeepSeekV4SemanticError::ElementCountOverflow {
            role: DeepSeekV4SemanticRole::HashTokenId,
        }
    })?;
    let (tokens, experts, top_k) =
        validate_routing_shape(token_count, expert_count, selected_expert_count)?;
    if !routed_scale.is_finite() || routed_scale <= 0.0 {
        return Err(DeepSeekV4SemanticError::InvalidRoutedScale(
            routed_scale.to_bits(),
        ));
    }
    let logit_count = checked_product(tokens, experts, DeepSeekV4SemanticRole::RouterLogit)?;
    validate_count(logits, logit_count, DeepSeekV4SemanticRole::RouterLogit)?;
    let pair_count = checked_product(tokens, top_k, DeepSeekV4SemanticRole::HashExpertId)?;
    if hashed_expert_ids.len() != pair_count {
        return Err(DeepSeekV4SemanticError::ElementCountMismatch {
            role: DeepSeekV4SemanticRole::HashExpertId,
            expected: pair_count,
            actual: hashed_expert_ids.len(),
        });
    }
    let mut expert_ids = empty_vec(pair_count, DeepSeekV4SemanticRole::HashExpertId)?;
    let mut expert_weights = empty_vec(pair_count, DeepSeekV4SemanticRole::RouterLogit)?;
    let mut unbiased_scores = empty_vec(experts, DeepSeekV4SemanticRole::RouterLogit)?;

    for token in 0..tokens {
        unbiased_scores.clear();
        for expert in 0..experts {
            let flat_index = token * experts + expert;
            unbiased_scores.push(router_score(logits[flat_index], flat_index)?);
        }
        let selected = &hashed_expert_ids[token * top_k..(token + 1) * top_k];
        for (slot, expert) in selected.iter().copied().enumerate() {
            if u32::from(expert) >= expert_count {
                return Err(DeepSeekV4SemanticError::HashExpertOutOfRange {
                    token: token as u32,
                    expert: u32::from(expert),
                    expert_count,
                });
            }
            if selected[..slot].contains(&expert) {
                return Err(DeepSeekV4SemanticError::DuplicateHashExpert {
                    token: token as u32,
                    expert: u32::from(expert),
                });
            }
        }
        expert_ids.extend_from_slice(selected);
        selected_weights(
            token,
            selected,
            &unbiased_scores,
            renormalize,
            routed_scale,
            &mut expert_weights,
        )?;
    }

    Ok(DeepSeekV4Routing {
        token_count,
        expert_count,
        selected_expert_count,
        expert_ids,
        expert_weights,
    })
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DeepSeekV4MhcDescriptor {
    stream_count: u32,
    hidden_size: u32,
    epsilon_bits: u32,
    sinkhorn_iterations: u32,
}

impl DeepSeekV4MhcDescriptor {
    pub fn new(
        stream_count: u32,
        hidden_size: u32,
        epsilon: f32,
        sinkhorn_iterations: u32,
    ) -> Result<Self, DeepSeekV4SemanticError> {
        if stream_count == 0 {
            return Err(DeepSeekV4SemanticError::InvalidStreamCount(stream_count));
        }
        if hidden_size == 0 {
            return Err(DeepSeekV4SemanticError::InvalidHiddenSize(hidden_size));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(DeepSeekV4SemanticError::InvalidEpsilon(epsilon.to_bits()));
        }
        if sinkhorn_iterations == 0 {
            return Err(DeepSeekV4SemanticError::InvalidSinkhornIterations(
                sinkhorn_iterations,
            ));
        }
        Ok(Self {
            stream_count,
            hidden_size,
            epsilon_bits: epsilon.to_bits(),
            sinkhorn_iterations,
        })
    }

    pub const fn stream_count(self) -> u32 {
        self.stream_count
    }

    pub const fn hidden_size(self) -> u32 {
        self.hidden_size
    }

    pub const fn epsilon(self) -> f32 {
        f32::from_bits(self.epsilon_bits)
    }

    pub const fn sinkhorn_iterations(self) -> u32 {
        self.sinkhorn_iterations
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeepSeekV4MhcReference {
    descriptor: DeepSeekV4MhcDescriptor,
    pre_gates: Vec<f32>,
    post_gates: Vec<f32>,
    mixing_matrix: Vec<f32>,
    pre_collapse: Vec<f32>,
    operator_output: Vec<f32>,
    output_streams: Vec<f32>,
}

impl DeepSeekV4MhcReference {
    pub const fn descriptor(&self) -> DeepSeekV4MhcDescriptor {
        self.descriptor
    }

    pub fn pre_gates(&self) -> &[f32] {
        &self.pre_gates
    }

    pub fn post_gates(&self) -> &[f32] {
        &self.post_gates
    }

    /// Row-major recurrent matrix `C[source, destination]`.
    pub fn mixing_matrix(&self) -> &[f32] {
        &self.mixing_matrix
    }

    pub fn pre_collapse(&self) -> &[f32] {
        &self.pre_collapse
    }

    pub fn operator_output(&self) -> &[f32] {
        &self.operator_output
    }

    /// Row-major `[streams, hidden]` post/recurrent mixture.
    pub fn output_streams(&self) -> &[f32] {
        &self.output_streams
    }
}

fn stable_sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

/// Pure FP32 mHC reference.
///
/// The operator closure receives the pre-collapse gated by
/// `sigmoid(pre_logit) + epsilon`. The post gate is
/// `2 * sigmoid(post_logit)`. Recurrent mixing applies a row-wise stable
/// softmax, adds epsilon to every element, normalizes columns once, then runs
/// the requested number of alternating row/column Sinkhorn normalizations.
/// With row-major `C[source, destination]`, the final destination stream is
/// `sum_source(C[source, destination] * input[source])
/// + post_gate[destination] * operator_output`.
pub fn reference_deepseek_v4_mhc<F>(
    descriptor: DeepSeekV4MhcDescriptor,
    streams: &[f32],
    pre_gate_logits: &[f32],
    post_gate_logits: &[f32],
    mixing_logits: &[f32],
    operator: F,
) -> Result<DeepSeekV4MhcReference, DeepSeekV4SemanticError>
where
    F: FnOnce(&[f32]) -> Vec<f32>,
{
    let stream_count =
        usize::try_from(descriptor.stream_count).expect("u32 stream count fits usize");
    let hidden_size = usize::try_from(descriptor.hidden_size).expect("u32 hidden size fits usize");
    let stream_elements =
        checked_product(stream_count, hidden_size, DeepSeekV4SemanticRole::Stream)?;
    let mixing_elements = checked_product(
        stream_count,
        stream_count,
        DeepSeekV4SemanticRole::MixingLogit,
    )?;
    validate_count(streams, stream_elements, DeepSeekV4SemanticRole::Stream)?;
    validate_count(
        pre_gate_logits,
        stream_count,
        DeepSeekV4SemanticRole::PreGateLogit,
    )?;
    validate_count(
        post_gate_logits,
        stream_count,
        DeepSeekV4SemanticRole::PostGateLogit,
    )?;
    validate_count(
        mixing_logits,
        mixing_elements,
        DeepSeekV4SemanticRole::MixingLogit,
    )?;

    let mut pre_gates = empty_vec(stream_count, DeepSeekV4SemanticRole::PreGateLogit)?;
    let mut post_gates = empty_vec(stream_count, DeepSeekV4SemanticRole::PostGateLogit)?;
    for (index, value) in pre_gate_logits.iter().copied().enumerate() {
        let gate = stable_sigmoid(value) + descriptor.epsilon();
        if !gate.is_finite() {
            return Err(DeepSeekV4SemanticError::NonFiniteIntermediate {
                stage: DeepSeekV4SemanticStage::PreGate,
                index,
            });
        }
        pre_gates.push(gate);
    }
    for (index, value) in post_gate_logits.iter().copied().enumerate() {
        let gate = 2.0 * stable_sigmoid(value);
        if !gate.is_finite() {
            return Err(DeepSeekV4SemanticError::NonFiniteIntermediate {
                stage: DeepSeekV4SemanticStage::PostGate,
                index,
            });
        }
        post_gates.push(gate);
    }

    let mut pre_collapse = empty_vec(hidden_size, DeepSeekV4SemanticRole::Stream)?;
    for hidden in 0..hidden_size {
        let mut value = 0.0_f32;
        for stream in 0..stream_count {
            value += pre_gates[stream] * streams[stream * hidden_size + hidden];
            if !value.is_finite() {
                return Err(DeepSeekV4SemanticError::NonFiniteIntermediate {
                    stage: DeepSeekV4SemanticStage::PreCollapse,
                    index: hidden,
                });
            }
        }
        pre_collapse.push(value);
    }

    let operator_output = operator(&pre_collapse);
    validate_count(
        &operator_output,
        hidden_size,
        DeepSeekV4SemanticRole::OperatorOutput,
    )?;

    let mut mixing_matrix = empty_vec(mixing_elements, DeepSeekV4SemanticRole::MixingLogit)?;
    for row in 0..stream_count {
        let begin = row * stream_count;
        let row_logits = &mixing_logits[begin..begin + stream_count];
        let maximum = row_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denominator = 0.0_f32;
        for (column, logit) in row_logits.iter().copied().enumerate() {
            let value = (logit - maximum).exp();
            if !value.is_finite() {
                return Err(DeepSeekV4SemanticError::NonFiniteIntermediate {
                    stage: DeepSeekV4SemanticStage::MixingSoftmax,
                    index: begin + column,
                });
            }
            denominator += value;
            mixing_matrix.push(value);
        }
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(DeepSeekV4SemanticError::NonFiniteIntermediate {
                stage: DeepSeekV4SemanticStage::MixingSoftmax,
                index: row,
            });
        }
        for value in &mut mixing_matrix[begin..begin + stream_count] {
            *value = *value / denominator + descriptor.epsilon();
        }
    }

    let normalize_columns = |matrix: &mut [f32]| -> Result<(), DeepSeekV4SemanticError> {
        for column in 0..stream_count {
            let mut sum = 0.0_f32;
            for row in 0..stream_count {
                sum += matrix[row * stream_count + column];
            }
            if !sum.is_finite() || sum <= 0.0 {
                return Err(DeepSeekV4SemanticError::NonFiniteIntermediate {
                    stage: DeepSeekV4SemanticStage::SinkhornColumn,
                    index: column,
                });
            }
            for row in 0..stream_count {
                matrix[row * stream_count + column] /= sum;
            }
        }
        Ok(())
    };
    normalize_columns(&mut mixing_matrix)?;
    for _ in 0..descriptor.sinkhorn_iterations {
        for row in 0..stream_count {
            let begin = row * stream_count;
            let sum: f32 = mixing_matrix[begin..begin + stream_count]
                .iter()
                .copied()
                .sum();
            if !sum.is_finite() || sum <= 0.0 {
                return Err(DeepSeekV4SemanticError::NonFiniteIntermediate {
                    stage: DeepSeekV4SemanticStage::SinkhornRow,
                    index: row,
                });
            }
            for value in &mut mixing_matrix[begin..begin + stream_count] {
                *value /= sum;
            }
        }
        normalize_columns(&mut mixing_matrix)?;
    }

    let mut output_streams = empty_vec(stream_elements, DeepSeekV4SemanticRole::Stream)?;
    for destination in 0..stream_count {
        for hidden in 0..hidden_size {
            let mut value = post_gates[destination] * operator_output[hidden];
            if !value.is_finite() {
                return Err(DeepSeekV4SemanticError::NonFiniteIntermediate {
                    stage: DeepSeekV4SemanticStage::RecurrentMix,
                    index: destination * hidden_size + hidden,
                });
            }
            for source in 0..stream_count {
                value += mixing_matrix[source * stream_count + destination]
                    * streams[source * hidden_size + hidden];
                if !value.is_finite() {
                    return Err(DeepSeekV4SemanticError::NonFiniteIntermediate {
                        stage: DeepSeekV4SemanticStage::RecurrentMix,
                        index: destination * hidden_size + hidden,
                    });
                }
            }
            output_streams.push(value);
        }
    }

    Ok(DeepSeekV4MhcReference {
        descriptor,
        pre_gates,
        post_gates,
        mixing_matrix,
        pre_collapse,
        operator_output,
        output_streams,
    })
}

/// Validate the reviewed uncompressed/CSA/HCA compression schedule value.
pub const fn validate_deepseek_v4_compression_ratio(
    ratio: u32,
) -> Result<u32, DeepSeekV4SemanticError> {
    match ratio {
        0 | 4 | 128 => Ok(ratio),
        _ => Err(DeepSeekV4SemanticError::InvalidCompressionRatio(ratio)),
    }
}

/// Number of complete units visible at a zero-based position.
///
/// For compressed layers this is exactly `floor((position + 1) / ratio)`.
/// Ratio zero denotes uncompressed attention, where each token is one visible
/// unit and therefore returns `position + 1`.
pub const fn deepseek_v4_completed_visible_blocks(
    position: u64,
    ratio: u32,
) -> Result<u64, DeepSeekV4SemanticError> {
    let tokens = match position.checked_add(1) {
        Some(tokens) => tokens,
        None => return Err(DeepSeekV4SemanticError::PositionOverflow),
    };
    match ratio {
        0 => Ok(tokens),
        4 | 128 => Ok(tokens / ratio as u64),
        _ => Err(DeepSeekV4SemanticError::InvalidCompressionRatio(ratio)),
    }
}

/// Ceil capacity in visible units for a token capacity.
///
/// Ratio zero is uncompressed, so its visible-unit capacity equals the token
/// capacity. The division form avoids overflow from `tokens + ratio - 1`.
pub const fn deepseek_v4_compression_capacity(
    token_capacity: u64,
    ratio: u32,
) -> Result<u64, DeepSeekV4SemanticError> {
    match ratio {
        0 => Ok(token_capacity),
        4 | 128 => {
            let divisor = ratio as u64;
            Ok(token_capacity / divisor + (token_capacity % divisor != 0) as u64)
        }
        _ => Err(DeepSeekV4SemanticError::InvalidCompressionRatio(ratio)),
    }
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
    fn score_router_covers_token_and_expert_boundaries() {
        for token_count in [1_u32, 3, 5, 6, 7] {
            for expert_count in [1_u32, 3, 5, 6, 7] {
                let top_k = expert_count.min(6);
                let logits = vec![0.0; token_count as usize * expert_count as usize];
                let bias = vec![0.0; expert_count as usize];
                let route = reference_deepseek_v4_score_route(
                    token_count,
                    expert_count,
                    top_k,
                    &logits,
                    &bias,
                    true,
                    1.0,
                )
                .unwrap();
                assert_eq!(
                    route.expert_ids().len(),
                    token_count as usize * top_k as usize
                );
                for token in 0..token_count as usize {
                    let begin = token * top_k as usize;
                    assert_eq!(
                        &route.expert_ids()[begin..begin + top_k as usize],
                        &(0..top_k as u16).collect::<Vec<_>>()
                    );
                    assert_close(
                        route.expert_weights()[begin..begin + top_k as usize]
                            .iter()
                            .sum(),
                        1.0,
                    );
                }
            }
        }
    }

    #[test]
    fn score_router_bias_selects_but_does_not_weight_and_ties_are_stable() {
        let logits = [0.0, 0.0, 4.0, -4.0, 0.0, 0.0, 0.0];
        let bias = [0.0, 10.0, -10.0, 0.0, 0.0, 0.0, 0.0];
        let route = reference_deepseek_v4_score_route(1, 7, 6, &logits, &bias, true, 2.5).unwrap();
        assert_eq!(route.expert_ids(), &[1, 0, 4, 5, 6, 3]);
        assert!(route.expert_weights()[0] > route.expert_weights()[5]);
        assert_close(route.expert_weights().iter().sum(), 2.5);

        let no_bias =
            reference_deepseek_v4_score_route(1, 7, 3, &logits, &[0.0; 7], false, 1.0).unwrap();
        assert_eq!(no_bias.expert_ids(), &[2, 0, 1]);
        assert_eq!(no_bias.expert_weights()[1], no_bias.expert_weights()[2]);
    }

    #[test]
    fn score_router_rejects_nonfinite_input_and_intermediate() {
        assert!(matches!(
            reference_deepseek_v4_score_route(1, 3, 1, &[0.0, f32::NAN, 0.0], &[0.0; 3], true, 1.0),
            Err(DeepSeekV4SemanticError::NonFiniteInput {
                role: DeepSeekV4SemanticRole::RouterLogit,
                index: 1
            })
        ));
        assert!(matches!(
            reference_deepseek_v4_score_route(1, 3, 1, &[1.0; 3], &[0.0; 3], false, f32::MAX,),
            Err(DeepSeekV4SemanticError::NonFiniteIntermediate {
                stage: DeepSeekV4SemanticStage::RouterWeight,
                ..
            })
        ));
        for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                reference_deepseek_v4_score_route(1, 3, 1, &[0.0; 3], &[0.0; 3], true, scale,),
                Err(DeepSeekV4SemanticError::InvalidRoutedScale(_))
            ));
        }

        let skew = reference_deepseek_v4_score_route(
            1,
            3,
            1,
            &[-f32::MAX, 0.0, f32::MAX],
            &[0.0; 3],
            true,
            1.0,
        )
        .unwrap();
        assert_eq!(skew.expert_ids(), &[2]);
        assert_eq!(skew.expert_weights(), &[1.0]);
    }

    #[test]
    fn hash_router_preserves_hash_order_and_uses_unbiased_scores() {
        let route = reference_deepseek_v4_hash_route(
            &[11, 29],
            DeepSeekV4RouterLocation::MainLayer(2),
            7,
            3,
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0,
            ],
            &[6, 2, 4, 0, 5, 1],
            true,
            1.0,
        )
        .unwrap();
        assert_eq!(route.expert_ids(), &[6, 2, 4, 0, 5, 1]);
        assert_close(route.expert_weights()[..3].iter().sum(), 1.0);
        assert_close(route.expert_weights()[3..].iter().sum(), 1.0);
        assert!(route.expert_weights()[0] > route.expert_weights()[1]);
    }

    #[test]
    fn hash_router_rejects_duplicate_and_out_of_range_experts() {
        let logits = [0.0; 7];
        assert!(matches!(
            reference_deepseek_v4_hash_route(
                &[1],
                DeepSeekV4RouterLocation::MainLayer(0),
                7,
                3,
                &logits,
                &[1, 1, 2],
                true,
                1.0,
            ),
            Err(DeepSeekV4SemanticError::DuplicateHashExpert {
                token: 0,
                expert: 1
            })
        ));
        assert!(matches!(
            reference_deepseek_v4_hash_route(
                &[1],
                DeepSeekV4RouterLocation::MainLayer(1),
                7,
                3,
                &logits,
                &[1, 2, 7],
                true,
                1.0,
            ),
            Err(DeepSeekV4SemanticError::HashExpertOutOfRange {
                token: 0,
                expert: 7,
                expert_count: 7
            })
        ));
    }

    #[test]
    fn hash_router_is_limited_to_first_three_main_layers() {
        let logits = [0.0; 7];
        for location in [
            DeepSeekV4RouterLocation::MainLayer(3),
            DeepSeekV4RouterLocation::DSparkStage(0),
            DeepSeekV4RouterLocation::NextN,
        ] {
            assert_eq!(
                reference_deepseek_v4_hash_route(
                    &[1],
                    location,
                    7,
                    3,
                    &logits,
                    &[1, 2, 3],
                    true,
                    1.0,
                ),
                Err(DeepSeekV4SemanticError::HashRoutingNotAllowed { location })
            );
        }
    }

    #[test]
    fn mhc_identity_operator_is_finite_and_preserves_stage_order() {
        let descriptor = DeepSeekV4MhcDescriptor::new(3, 5, 1.0e-6, 8).unwrap();
        let streams = [
            1.0, 2.0, 3.0, 4.0, 5.0, // stream 0
            6.0, 7.0, 8.0, 9.0, 10.0, // stream 1
            -1.0, -2.0, -3.0, -4.0, -5.0, // stream 2
        ];
        let reference = reference_deepseek_v4_mhc(
            descriptor,
            &streams,
            &[0.0; 3],
            &[0.0; 3],
            &[12.0, -12.0, -12.0, -12.0, 12.0, -12.0, -12.0, -12.0, 12.0],
            |pre_collapse| pre_collapse.to_vec(),
        )
        .unwrap();
        assert_eq!(reference.descriptor(), descriptor);
        assert_eq!(reference.pre_gates(), &[0.5 + 1.0e-6; 3]);
        assert_eq!(reference.post_gates(), &[1.0; 3]);
        for (actual, expected) in reference
            .pre_collapse()
            .iter()
            .copied()
            .zip([3.0, 3.5, 4.0, 4.5, 5.0])
        {
            assert_close(actual, expected);
        }
        assert_eq!(reference.operator_output(), reference.pre_collapse());
        assert!(
            reference
                .mixing_matrix()
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            reference
                .output_streams()
                .iter()
                .all(|value| value.is_finite())
        );
        for row in 0..3 {
            assert_close(
                reference.mixing_matrix()[row * 3..row * 3 + 3].iter().sum(),
                1.0,
            );
        }
        for column in 0..3 {
            assert_close(
                (0..3)
                    .map(|row| reference.mixing_matrix()[row * 3 + column])
                    .sum(),
                1.0,
            );
        }
    }

    #[test]
    fn mhc_rejects_shape_nonfinite_and_overflowing_arithmetic() {
        let descriptor = DeepSeekV4MhcDescriptor::new(3, 5, 1.0e-6, 2).unwrap();
        assert!(matches!(
            reference_deepseek_v4_mhc(
                descriptor,
                &[0.0; 14],
                &[0.0; 3],
                &[0.0; 3],
                &[0.0; 9],
                |values| values.to_vec(),
            ),
            Err(DeepSeekV4SemanticError::ElementCountMismatch {
                role: DeepSeekV4SemanticRole::Stream,
                expected: 15,
                actual: 14,
            })
        ));
        let mut streams = [0.0; 15];
        streams[4] = f32::INFINITY;
        assert!(matches!(
            reference_deepseek_v4_mhc(
                descriptor,
                &streams,
                &[0.0; 3],
                &[0.0; 3],
                &[0.0; 9],
                |values| values.to_vec(),
            ),
            Err(DeepSeekV4SemanticError::NonFiniteInput {
                role: DeepSeekV4SemanticRole::Stream,
                index: 4,
            })
        ));
        assert!(matches!(
            reference_deepseek_v4_mhc(
                DeepSeekV4MhcDescriptor::new(1, 1, 1.0e-6, 1).unwrap(),
                &[f32::MAX],
                &[0.0],
                &[f32::MAX],
                &[0.0],
                |_| vec![f32::MAX],
            ),
            Err(DeepSeekV4SemanticError::NonFiniteIntermediate {
                stage: DeepSeekV4SemanticStage::RecurrentMix,
                ..
            })
        ));
    }

    #[test]
    fn compression_boundaries_are_exact_before_and_after_blocks() {
        assert_eq!(validate_deepseek_v4_compression_ratio(0), Ok(0));
        assert_eq!(validate_deepseek_v4_compression_ratio(4), Ok(4));
        assert_eq!(validate_deepseek_v4_compression_ratio(128), Ok(128));
        assert!(validate_deepseek_v4_compression_ratio(1).is_err());

        let ratio4 = [
            (0, 0),
            (1, 0),
            (3, 1),
            (4, 1),
            (5, 1),
            (127, 32),
            (128, 32),
            (129, 32),
        ];
        for (position, expected) in ratio4 {
            assert_eq!(
                deepseek_v4_completed_visible_blocks(position, 4),
                Ok(expected)
            );
        }
        let ratio128 = [(0, 0), (1, 0), (127, 1), (128, 1), (129, 1)];
        for (position, expected) in ratio128 {
            assert_eq!(
                deepseek_v4_completed_visible_blocks(position, 128),
                Ok(expected)
            );
        }
        assert_eq!(deepseek_v4_completed_visible_blocks(0, 0), Ok(1));
        assert_eq!(deepseek_v4_completed_visible_blocks(5, 0), Ok(6));
        assert_eq!(deepseek_v4_compression_capacity(0, 4), Ok(0));
        assert_eq!(deepseek_v4_compression_capacity(1, 4), Ok(1));
        assert_eq!(deepseek_v4_compression_capacity(3, 4), Ok(1));
        assert_eq!(deepseek_v4_compression_capacity(4, 4), Ok(1));
        assert_eq!(deepseek_v4_compression_capacity(5, 4), Ok(2));
        assert_eq!(deepseek_v4_compression_capacity(127, 128), Ok(1));
        assert_eq!(deepseek_v4_compression_capacity(128, 128), Ok(1));
        assert_eq!(deepseek_v4_compression_capacity(129, 128), Ok(2));
        assert_eq!(deepseek_v4_compression_capacity(u64::MAX, 128), Ok(1 << 57));
        assert_eq!(
            deepseek_v4_completed_visible_blocks(u64::MAX, 4),
            Err(DeepSeekV4SemanticError::PositionOverflow)
        );
    }
}
