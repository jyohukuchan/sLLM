//! Bounded host-side semantics for DiffusionGemma denoising.
//!
//! This is a small, model-independent oracle.  It deliberately does not load
//! weights or retain a canvas-sized tensor of logits.  In particular, the
//! self-conditioning value below is a shape/step descriptor for the processed
//! logits supplied by a caller; it is not a previous-token shortcut.

use std::fmt;

pub const DIFFUSION_GEMMA_CANVAS_LENGTH: usize = 256;
pub const DIFFUSION_GEMMA_CONTEXT_LENGTH: usize = 262_144;
pub const DIFFUSION_GEMMA_MAX_DENOISING_STEPS: u32 = 48;
pub const DIFFUSION_GEMMA_TEMPERATURE_START: f32 = 0.8;
pub const DIFFUSION_GEMMA_TEMPERATURE_END: f32 = 0.4;
pub const DIFFUSION_GEMMA_ENTROPY_BOUND: f32 = 0.1;
pub const DIFFUSION_GEMMA_CONFIDENCE_THRESHOLD: f32 = 0.005;
pub const DIFFUSION_GEMMA_STABILITY_THRESHOLD: u32 = 1;
pub const DIFFUSION_GEMMA_BLOCK_LENGTH: usize = 32;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DiffusionGemmaAttentionMode {
    /// Incremental prefix attention over the encoder context.
    CausalEncoder,
    /// Bidirectional attention over one denoising canvas.
    BidirectionalDecoder,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DiffusionGemmaRefinementAction {
    Accepted,
    Renoised,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DiffusionGemmaSemanticError {
    InvalidCanvasLength(usize),
    InvalidVocabularySize(u32),
    InvalidStep {
        step: u32,
        max_steps: u32,
    },
    InvalidMaxSteps(u32),
    InvalidAttentionSequenceLength {
        length: usize,
        maximum: usize,
    },
    InvalidAttentionPosition {
        query: usize,
        key: usize,
        length: usize,
    },
    InvalidSelfConditioningStep(u32),
    SelfConditioningStepMismatch {
        expected: u32,
        actual: u32,
    },
    SelfConditioningLengthMismatch {
        expected: usize,
        actual: usize,
    },
    SelfConditioningVocabularyMismatch {
        expected: u32,
        actual: u32,
    },
    MissingSelfConditioning {
        step: u32,
    },
    UnexpectedSelfConditioning {
        step: u32,
    },
    VectorLengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    TokenOutOfRange {
        index: usize,
        token: u32,
        vocabulary_size: u32,
    },
    NonFiniteInput {
        field: &'static str,
        index: usize,
    },
    NegativeProbability {
        index: usize,
    },
    ProbabilitySumInvalid,
    EntropyOutOfRange {
        index: usize,
        value_bits: u32,
    },
    ConfidenceOutOfRange {
        index: usize,
        value_bits: u32,
    },
    ArithmeticOverflow,
    StepOverflow,
    RandomSourceExhausted,
    InvalidRandomDraw(u64),
    NotReadyToCommit,
    AlreadyCommitted,
    AlreadyFinished,
    CanvasCountMismatch {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for DiffusionGemmaSemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "DiffusionGemma semantic error: {self:?}")
    }
}

impl std::error::Error for DiffusionGemmaSemanticError {}

/// Explicit deterministic randomness seam.  The caller owns the RNG and the
/// oracle never falls back to ambient or operating-system randomness.
pub trait DiffusionGemmaRandomSource {
    fn next_u64(&mut self) -> Result<u64, DiffusionGemmaSemanticError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiffusionGemmaSequenceRandom {
    values: Vec<u64>,
    position: usize,
}

impl DiffusionGemmaSequenceRandom {
    pub fn new(values: &[u64]) -> Self {
        Self {
            values: values.to_vec(),
            position: 0,
        }
    }

    pub const fn position(&self) -> usize {
        self.position
    }
}

impl DiffusionGemmaRandomSource for DiffusionGemmaSequenceRandom {
    fn next_u64(&mut self) -> Result<u64, DiffusionGemmaSemanticError> {
        let value = self
            .values
            .get(self.position)
            .copied()
            .ok_or(DiffusionGemmaSemanticError::RandomSourceExhausted)?;
        self.position = self
            .position
            .checked_add(1)
            .ok_or(DiffusionGemmaSemanticError::StepOverflow)?;
        Ok(value)
    }
}

fn validate_canvas_length(length: usize) -> Result<(), DiffusionGemmaSemanticError> {
    if length == 0 || length > DIFFUSION_GEMMA_CANVAS_LENGTH {
        return Err(DiffusionGemmaSemanticError::InvalidCanvasLength(length));
    }
    Ok(())
}

fn validate_max_steps(max_steps: u32) -> Result<(), DiffusionGemmaSemanticError> {
    if max_steps == 0 || max_steps > DIFFUSION_GEMMA_MAX_DENOISING_STEPS {
        return Err(DiffusionGemmaSemanticError::InvalidMaxSteps(max_steps));
    }
    Ok(())
}

fn validate_probability(
    value: f32,
    field: &'static str,
    index: usize,
) -> Result<(), DiffusionGemmaSemanticError> {
    if !value.is_finite() {
        return Err(DiffusionGemmaSemanticError::NonFiniteInput { field, index });
    }
    if value < 0.0 {
        return Err(DiffusionGemmaSemanticError::NegativeProbability { index });
    }
    Ok(())
}

fn validate_entropy(value: f32, index: usize) -> Result<(), DiffusionGemmaSemanticError> {
    if !value.is_finite() {
        return Err(DiffusionGemmaSemanticError::NonFiniteInput {
            field: "entropy",
            index,
        });
    }
    // Categorical entropy is not normalized: values may be larger than one.
    if value < 0.0 {
        return Err(DiffusionGemmaSemanticError::EntropyOutOfRange {
            index,
            value_bits: value.to_bits(),
        });
    }
    Ok(())
}

fn validate_token(
    token: u32,
    index: usize,
    vocabulary_size: u32,
) -> Result<(), DiffusionGemmaSemanticError> {
    if token >= vocabulary_size {
        return Err(DiffusionGemmaSemanticError::TokenOutOfRange {
            index,
            token,
            vocabulary_size,
        });
    }
    Ok(())
}

/// The zero-based denoising-index view of the schedule.
///
/// The fixed Transformers processor receives `cur_step`, the number of
/// denoising steps remaining.  Thus zero-based index `step` maps to
/// `cur_step = max_steps - step` and the final index deliberately does not
/// reach `t_min`: for 48 steps it is 0.4083333, not 0.4.
pub fn diffusion_gemma_temperature(
    step: u32,
    max_steps: u32,
) -> Result<f32, DiffusionGemmaSemanticError> {
    validate_max_steps(max_steps)?;
    if step >= max_steps {
        return Err(DiffusionGemmaSemanticError::InvalidStep { step, max_steps });
    }
    diffusion_gemma_temperature_for_cur_step(max_steps - step, max_steps)
}

/// Exact source-side schedule.  `cur_step` is the number of steps remaining,
/// as passed to `LinearTemperatureScheduleLogitsProcessor`.
pub fn diffusion_gemma_temperature_for_cur_step(
    cur_step: u32,
    max_denoising_steps: u32,
) -> Result<f32, DiffusionGemmaSemanticError> {
    validate_max_steps(max_denoising_steps)?;
    if cur_step == 0 || cur_step > max_denoising_steps {
        return Err(DiffusionGemmaSemanticError::InvalidStep {
            step: cur_step,
            max_steps: max_denoising_steps,
        });
    }
    let fraction = cur_step as f32 / max_denoising_steps as f32;
    let temperature = DIFFUSION_GEMMA_TEMPERATURE_END
        + (DIFFUSION_GEMMA_TEMPERATURE_START - DIFFUSION_GEMMA_TEMPERATURE_END) * fraction;
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(DiffusionGemmaSemanticError::NonFiniteInput {
            field: "temperature",
            index: cur_step as usize,
        });
    }
    Ok(temperature)
}

/// Return whether a query/key pair is visible under the selected branch.
/// Encoder positions use the full 262144-token context; decoder positions
/// are independently restricted to the 256-token canvas.
pub fn diffusion_gemma_attention_allowed(
    mode: DiffusionGemmaAttentionMode,
    query_position: usize,
    key_position: usize,
    sequence_length: usize,
) -> Result<bool, DiffusionGemmaSemanticError> {
    let maximum = match mode {
        DiffusionGemmaAttentionMode::CausalEncoder => DIFFUSION_GEMMA_CONTEXT_LENGTH,
        DiffusionGemmaAttentionMode::BidirectionalDecoder => DIFFUSION_GEMMA_CANVAS_LENGTH,
    };
    if sequence_length == 0 || sequence_length > maximum {
        return Err(
            DiffusionGemmaSemanticError::InvalidAttentionSequenceLength {
                length: sequence_length,
                maximum,
            },
        );
    }
    if query_position >= sequence_length || key_position >= sequence_length {
        return Err(DiffusionGemmaSemanticError::InvalidAttentionPosition {
            query: query_position,
            key: key_position,
            length: sequence_length,
        });
    }
    Ok(match mode {
        DiffusionGemmaAttentionMode::CausalEncoder => key_position <= query_position,
        DiffusionGemmaAttentionMode::BidirectionalDecoder => true,
    })
}

/// A bounded identity for one processed logits tensor.  It intentionally
/// carries shape and step only; the potentially huge logits are held by the
/// model/runtime, not by this host oracle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiffusionGemmaProcessedLogits {
    source_step: u32,
    canvas_length: usize,
    vocabulary_size: u32,
}

impl DiffusionGemmaProcessedLogits {
    pub fn new(
        source_step: u32,
        canvas_length: usize,
        vocabulary_size: u32,
    ) -> Result<Self, DiffusionGemmaSemanticError> {
        validate_canvas_length(canvas_length)?;
        if vocabulary_size == 0 {
            return Err(DiffusionGemmaSemanticError::InvalidVocabularySize(
                vocabulary_size,
            ));
        }
        Ok(Self {
            source_step,
            canvas_length,
            vocabulary_size,
        })
    }

    pub const fn source_step(self) -> u32 {
        self.source_step
    }

    pub const fn canvas_length(self) -> usize {
        self.canvas_length
    }

    pub const fn vocabulary_size(self) -> u32 {
        self.vocabulary_size
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiffusionGemmaSelfConditioning {
    previous_logits: DiffusionGemmaProcessedLogits,
}

impl DiffusionGemmaSelfConditioning {
    pub const fn source_step(self) -> u32 {
        self.previous_logits.source_step
    }

    pub const fn previous_logits(self) -> DiffusionGemmaProcessedLogits {
        self.previous_logits
    }
}

/// Describe the previous processed logits input for a denoising step.  Step
/// zero has no previous logits.  Later steps require the immediately previous
/// processed logits shape and reject token-id-shaped substitutes.
pub fn diffusion_gemma_self_conditioning(
    step: u32,
    canvas_length: usize,
    vocabulary_size: u32,
    previous_logits: Option<DiffusionGemmaProcessedLogits>,
) -> Result<Option<DiffusionGemmaSelfConditioning>, DiffusionGemmaSemanticError> {
    validate_canvas_length(canvas_length)?;
    if vocabulary_size == 0 {
        return Err(DiffusionGemmaSemanticError::InvalidVocabularySize(
            vocabulary_size,
        ));
    }
    match (step, previous_logits) {
        (0, None) => Ok(None),
        (0, Some(_)) => Err(DiffusionGemmaSemanticError::UnexpectedSelfConditioning { step }),
        (_, None) => Err(DiffusionGemmaSemanticError::MissingSelfConditioning { step }),
        (step, Some(logits)) => {
            let expected_source_step = step.checked_sub(1).ok_or(
                DiffusionGemmaSemanticError::InvalidSelfConditioningStep(step),
            )?;
            if logits.source_step != expected_source_step {
                return Err(DiffusionGemmaSemanticError::SelfConditioningStepMismatch {
                    expected: expected_source_step,
                    actual: logits.source_step,
                });
            }
            if logits.canvas_length != canvas_length {
                return Err(
                    DiffusionGemmaSemanticError::SelfConditioningLengthMismatch {
                        expected: canvas_length,
                        actual: logits.canvas_length,
                    },
                );
            }
            if logits.vocabulary_size != vocabulary_size {
                return Err(
                    DiffusionGemmaSemanticError::SelfConditioningVocabularyMismatch {
                        expected: vocabulary_size,
                        actual: logits.vocabulary_size,
                    },
                );
            }
            Ok(Some(DiffusionGemmaSelfConditioning {
                previous_logits: logits,
            }))
        }
    }
}

/// Stable normalized Shannon entropy for a probability-weight vector.  This
/// helper is separate from the sampler, whose source contract uses raw
/// categorical entropy after logits processors have run.
pub fn diffusion_gemma_normalized_entropy(
    weights: &[f32],
) -> Result<f32, DiffusionGemmaSemanticError> {
    if weights.is_empty() {
        return Err(DiffusionGemmaSemanticError::ProbabilitySumInvalid);
    }
    let mut total = 0.0_f64;
    for (index, value) in weights.iter().copied().enumerate() {
        validate_probability(value, "probability", index)?;
        total += f64::from(value);
        if !total.is_finite() {
            return Err(DiffusionGemmaSemanticError::ProbabilitySumInvalid);
        }
    }
    if total <= 0.0 {
        return Err(DiffusionGemmaSemanticError::ProbabilitySumInvalid);
    }
    if weights.len() == 1 {
        return Ok(0.0);
    }
    let mut entropy = 0.0_f64;
    for value in weights.iter().copied() {
        if value != 0.0 {
            let probability = f64::from(value) / total;
            entropy -= probability * probability.ln();
        }
    }
    let normalized = entropy / (weights.len() as f64).ln();
    if !normalized.is_finite() || !(0.0..=1.0).contains(&normalized) {
        return Err(DiffusionGemmaSemanticError::ProbabilitySumInvalid);
    }
    Ok(normalized as f32)
}

/// Compute raw categorical entropy from a processed logits row.  This is the
/// same quantity used by the stopping criterion and by the entropy-bound
/// sampler (before rows are reduced to a mean).
pub fn diffusion_gemma_categorical_entropy(
    logits: &[f32],
) -> Result<f32, DiffusionGemmaSemanticError> {
    if logits.is_empty() {
        return Err(DiffusionGemmaSemanticError::ProbabilitySumInvalid);
    }
    let mut maximum = f64::NEG_INFINITY;
    for (index, value) in logits.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(DiffusionGemmaSemanticError::NonFiniteInput {
                field: "logits",
                index,
            });
        }
        maximum = maximum.max(f64::from(value));
    }
    let mut exp_sum = 0.0_f64;
    let mut weighted_logit_sum = 0.0_f64;
    for value in logits.iter().copied() {
        let weight = (f64::from(value) - maximum).exp();
        exp_sum += weight;
        weighted_logit_sum += weight * f64::from(value);
    }
    if !exp_sum.is_finite() || exp_sum <= 0.0 || !weighted_logit_sum.is_finite() {
        return Err(DiffusionGemmaSemanticError::ArithmeticOverflow);
    }
    let logsumexp = maximum + exp_sum.ln();
    let entropy = logsumexp - weighted_logit_sum / exp_sum;
    if !entropy.is_finite() || entropy < 0.0 || entropy > f64::from(f32::MAX) {
        return Err(DiffusionGemmaSemanticError::ArithmeticOverflow);
    }
    Ok(entropy as f32)
}

/// A diagnostic for one entropy value.  It is not the sampler's acceptance
/// rule; acceptance is the cumulative sorted rule in the function below.
pub fn diffusion_gemma_entropy_within_bound(
    entropy: f32,
) -> Result<bool, DiffusionGemmaSemanticError> {
    validate_entropy(entropy, 0)?;
    Ok(entropy <= DIFFUSION_GEMMA_ENTROPY_BOUND)
}

/// Select the k lowest-entropy positions under the source sampler contract:
/// after ascending sort, position k is accepted iff
/// `cumulative_entropy[k] - sorted_entropy[k] <= entropy_bound`.
///
/// The subtraction intentionally excludes the current (largest selected)
/// entropy.  Entropies are raw values from temperature-scaled logits, not
/// normalized values and not confidence scores.  Equal values use the lower
/// original index for deterministic host-oracle behavior.
pub fn diffusion_gemma_entropy_bound_selection(
    entropies: &[f32],
    entropy_bound: f32,
) -> Result<Vec<bool>, DiffusionGemmaSemanticError> {
    if !entropy_bound.is_finite() || entropy_bound <= 0.0 {
        return Err(DiffusionGemmaSemanticError::EntropyOutOfRange {
            index: 0,
            value_bits: entropy_bound.to_bits(),
        });
    }
    let mut order = Vec::with_capacity(entropies.len());
    for (index, entropy) in entropies.iter().copied().enumerate() {
        validate_entropy(entropy, index)?;
        order.push((entropy, index));
    }
    order.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });

    let mut selected = vec![false; entropies.len()];
    let mut cumulative = 0.0_f32;
    for (entropy, index) in order {
        cumulative += entropy;
        if !cumulative.is_finite() {
            return Err(DiffusionGemmaSemanticError::ArithmeticOverflow);
        }
        let previous_sum = cumulative - entropy;
        if !previous_sum.is_finite() {
            return Err(DiffusionGemmaSemanticError::ArithmeticOverflow);
        }
        selected[index] = previous_sum <= entropy_bound;
    }
    Ok(selected)
}

fn uniform_token<R: DiffusionGemmaRandomSource>(
    vocabulary_size: u32,
    random: &mut R,
) -> Result<u32, DiffusionGemmaSemanticError> {
    if vocabulary_size == 0 {
        return Err(DiffusionGemmaSemanticError::InvalidVocabularySize(
            vocabulary_size,
        ));
    }
    if vocabulary_size == 1 {
        let _ = random.next_u64()?;
        return Ok(0);
    }
    let divisor = u64::from(vocabulary_size);
    // Keep only a complete number of residue classes to avoid modulo bias.
    let limit = u64::MAX - (u64::MAX % divisor);
    loop {
        let draw = random.next_u64()?;
        if draw < limit {
            return Ok((draw % divisor) as u32);
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiffusionGemmaCanvas {
    vocabulary_size: u32,
    tokens: Vec<u32>,
    accepted: Vec<bool>,
    step: u32,
    committed: bool,
    finished: bool,
    last_logits: Option<DiffusionGemmaProcessedLogits>,
    last_argmax_canvas: Option<Vec<u32>>,
}

impl DiffusionGemmaCanvas {
    /// Initialize a uniform canvas using only the caller-provided RNG.
    pub fn uniform_state<R: DiffusionGemmaRandomSource>(
        canvas_length: usize,
        vocabulary_size: u32,
        random: &mut R,
    ) -> Result<Self, DiffusionGemmaSemanticError> {
        validate_canvas_length(canvas_length)?;
        if vocabulary_size == 0 {
            return Err(DiffusionGemmaSemanticError::InvalidVocabularySize(
                vocabulary_size,
            ));
        }
        let mut tokens = Vec::with_capacity(canvas_length);
        for _ in 0..canvas_length {
            tokens.push(uniform_token(vocabulary_size, random)?);
        }
        Ok(Self {
            vocabulary_size,
            tokens,
            accepted: vec![false; canvas_length],
            step: 0,
            committed: false,
            finished: false,
            last_logits: None,
            last_argmax_canvas: None,
        })
    }

    pub const fn vocabulary_size(&self) -> u32 {
        self.vocabulary_size
    }

    pub const fn step(&self) -> u32 {
        self.step
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The accepted mask from the most recent sampler step.  It is replaced
    /// on each step and is never used as a stopping/confidence signal.
    pub fn accepted_mask(&self) -> &[bool] {
        &self.accepted
    }

    pub const fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn is_ready_to_commit(&self) -> bool {
        self.finished && !self.committed
    }

    pub const fn is_committed(&self) -> bool {
        self.committed
    }

    pub fn self_conditioning(
        &self,
    ) -> Result<Option<DiffusionGemmaSelfConditioning>, DiffusionGemmaSemanticError> {
        diffusion_gemma_self_conditioning(
            self.step,
            self.tokens.len(),
            self.vocabulary_size,
            self.last_logits,
        )
    }

    /// Update the separate adaptive-stopping state.  Callers pass argmax
    /// tokens from the processed logits, not the accepted/renoised canvas.
    pub fn update_stopping(
        &mut self,
        argmax_canvas: &[u32],
        token_entropies: &[f32],
        stopping: &mut DiffusionGemmaStoppingCriteria,
    ) -> Result<bool, DiffusionGemmaSemanticError> {
        if self.committed {
            return Err(DiffusionGemmaSemanticError::AlreadyCommitted);
        }
        if argmax_canvas.len() != self.tokens.len() {
            return Err(DiffusionGemmaSemanticError::VectorLengthMismatch {
                field: "argmax_canvas",
                expected: self.tokens.len(),
                actual: argmax_canvas.len(),
            });
        }
        for (index, token) in argmax_canvas.iter().copied().enumerate() {
            validate_token(token, index, self.vocabulary_size)?;
        }
        let stop = stopping.should_stop(argmax_canvas, token_entropies)?;
        self.last_argmax_canvas = Some(argmax_canvas.to_vec());
        if stop {
            self.finished = true;
        }
        Ok(stop)
    }

    /// Mark the canvas final at the maximum-step boundary. The official loop
    /// publishes the latest argmax prediction, not the sampled/renoised
    /// working canvas, so that prediction must be supplied explicitly.
    pub fn mark_finished(
        &mut self,
        argmax_canvas: &[u32],
    ) -> Result<(), DiffusionGemmaSemanticError> {
        if self.committed {
            return Err(DiffusionGemmaSemanticError::AlreadyCommitted);
        }
        if argmax_canvas.len() != self.tokens.len() {
            return Err(DiffusionGemmaSemanticError::VectorLengthMismatch {
                field: "argmax_canvas",
                expected: self.tokens.len(),
                actual: argmax_canvas.len(),
            });
        }
        for (index, token) in argmax_canvas.iter().copied().enumerate() {
            validate_token(token, index, self.vocabulary_size)?;
        }
        self.last_argmax_canvas = Some(argmax_canvas.to_vec());
        self.finished = true;
        Ok(())
    }

    /// Refine one denoising step.  Entropies are already computed from the
    /// temperature-scaled logits. The source creates a complete uniform random
    /// canvas every step before selecting its non-accepted positions, so the
    /// explicit RNG advances once per canvas position even when a proposal is
    /// accepted. Accepted positions retain the denoiser token.
    pub fn refine<R: DiffusionGemmaRandomSource>(
        &mut self,
        proposed_tokens: &[u32],
        token_entropies: &[f32],
        max_steps: u32,
        random: &mut R,
    ) -> Result<DiffusionGemmaRefinementReport, DiffusionGemmaSemanticError> {
        if self.committed {
            return Err(DiffusionGemmaSemanticError::AlreadyCommitted);
        }
        if self.finished {
            return Err(DiffusionGemmaSemanticError::AlreadyFinished);
        }
        validate_max_steps(max_steps)?;
        if self.step >= max_steps {
            return Err(DiffusionGemmaSemanticError::AlreadyFinished);
        }
        let expected = self.tokens.len();
        if proposed_tokens.len() != expected {
            return Err(DiffusionGemmaSemanticError::VectorLengthMismatch {
                field: "proposed_tokens",
                expected,
                actual: proposed_tokens.len(),
            });
        }
        if token_entropies.len() != expected {
            return Err(DiffusionGemmaSemanticError::VectorLengthMismatch {
                field: "token_entropies",
                expected,
                actual: token_entropies.len(),
            });
        }
        for (index, token) in proposed_tokens.iter().copied().enumerate() {
            validate_token(token, index, self.vocabulary_size)?;
        }
        // Selection validates all entropy rows before any state or RNG event.
        let accepted = diffusion_gemma_entropy_bound_selection(
            token_entropies,
            DIFFUSION_GEMMA_ENTROPY_BOUND,
        )?;
        let mut random_canvas = Vec::with_capacity(expected);
        for _ in 0..expected {
            random_canvas.push(uniform_token(self.vocabulary_size, random)?);
        }
        let mut next_tokens = Vec::with_capacity(expected);
        let mut actions = Vec::with_capacity(expected);
        for (index, proposed) in proposed_tokens.iter().copied().enumerate() {
            if accepted[index] {
                next_tokens.push(proposed);
                actions.push(DiffusionGemmaRefinementAction::Accepted);
            } else {
                next_tokens.push(random_canvas[index]);
                actions.push(DiffusionGemmaRefinementAction::Renoised);
            }
        }

        let step = self.step;
        let cur_step = max_steps - step;
        let temperature = diffusion_gemma_temperature_for_cur_step(cur_step, max_steps)?;
        self.tokens = next_tokens;
        self.accepted = accepted.clone();
        self.last_logits = Some(DiffusionGemmaProcessedLogits {
            source_step: step,
            canvas_length: expected,
            vocabulary_size: self.vocabulary_size,
        });
        self.step = self
            .step
            .checked_add(1)
            .ok_or(DiffusionGemmaSemanticError::StepOverflow)?;
        let accepted_count = accepted.iter().filter(|value| **value).count();
        Ok(DiffusionGemmaRefinementReport {
            step,
            cur_step,
            temperature,
            actions,
            accepted_mask: accepted,
            accepted_count,
            renoised_count: expected - accepted_count,
            // Adaptive stopping is deliberately separate and is updated by
            // `update_stopping`; acceptance never implies early stop.
            early_stop: false,
        })
    }

    pub fn commit(&mut self) -> Result<DiffusionGemmaCanvasCommit, DiffusionGemmaSemanticError> {
        if self.committed {
            return Err(DiffusionGemmaSemanticError::AlreadyCommitted);
        }
        if !self.finished {
            return Err(DiffusionGemmaSemanticError::NotReadyToCommit);
        }
        let tokens = self
            .last_argmax_canvas
            .clone()
            .ok_or(DiffusionGemmaSemanticError::NotReadyToCommit)?;
        self.committed = true;
        Ok(DiffusionGemmaCanvasCommit {
            step: self.step,
            tokens,
        })
    }
}

/// Stateful, independent adaptive stopping.  It uses mean raw categorical
/// entropy and argmax-canvas history only; no accepted mask or confidence
/// score participates in this decision.
#[derive(Clone, Debug, PartialEq)]
pub struct DiffusionGemmaStoppingCriteria {
    confidence_threshold: f32,
    stability_threshold: usize,
    history: Vec<Vec<u32>>,
}

impl DiffusionGemmaStoppingCriteria {
    pub fn new(
        stability_threshold: u32,
        confidence_threshold: f32,
    ) -> Result<Self, DiffusionGemmaSemanticError> {
        if !confidence_threshold.is_finite() || confidence_threshold <= 0.0 {
            return Err(DiffusionGemmaSemanticError::ConfidenceOutOfRange {
                index: 0,
                value_bits: confidence_threshold.to_bits(),
            });
        }
        let stability_threshold = usize::try_from(stability_threshold)
            .map_err(|_| DiffusionGemmaSemanticError::ArithmeticOverflow)?;
        Ok(Self {
            confidence_threshold,
            stability_threshold,
            history: Vec::new(),
        })
    }

    pub fn default_contract() -> Self {
        Self {
            confidence_threshold: DIFFUSION_GEMMA_CONFIDENCE_THRESHOLD,
            stability_threshold: DIFFUSION_GEMMA_STABILITY_THRESHOLD as usize,
            history: Vec::new(),
        }
    }

    pub const fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }

    pub const fn stability_threshold(&self) -> usize {
        self.stability_threshold
    }

    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    pub fn reset(&mut self) {
        self.history.clear();
    }

    pub fn should_stop(
        &mut self,
        argmax_canvas: &[u32],
        token_entropies: &[f32],
    ) -> Result<bool, DiffusionGemmaSemanticError> {
        if argmax_canvas.is_empty() || argmax_canvas.len() > DIFFUSION_GEMMA_CANVAS_LENGTH {
            return Err(DiffusionGemmaSemanticError::InvalidCanvasLength(
                argmax_canvas.len(),
            ));
        }
        if token_entropies.len() != argmax_canvas.len() {
            return Err(DiffusionGemmaSemanticError::VectorLengthMismatch {
                field: "token_entropies",
                expected: argmax_canvas.len(),
                actual: token_entropies.len(),
            });
        }
        let mut sum = 0.0_f64;
        for (index, entropy) in token_entropies.iter().copied().enumerate() {
            validate_entropy(entropy, index)?;
            sum += f64::from(entropy);
        }
        if !sum.is_finite() {
            return Err(DiffusionGemmaSemanticError::ArithmeticOverflow);
        }
        let mean = sum / token_entropies.len() as f64;
        if !mean.is_finite() {
            return Err(DiffusionGemmaSemanticError::ArithmeticOverflow);
        }
        // The source initializes history with impossible -1 sentinels.  A
        // bounded history with the same observable behavior is cleaner.
        let stable = self.history.len() >= self.stability_threshold
            && self
                .history
                .iter()
                .rev()
                .take(self.stability_threshold)
                .all(|previous| previous.as_slice() == argmax_canvas);
        self.history.push(argmax_canvas.to_vec());
        if self.history.len() > self.stability_threshold {
            let remove = self.history.len() - self.stability_threshold;
            self.history.drain(0..remove);
        }
        // The fixed stopping contract uses strict inequality; exactly 0.005
        // does not stop.
        Ok(stable && mean < f64::from(self.confidence_threshold))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiffusionGemmaCanvasCommit {
    pub step: u32,
    pub tokens: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiffusionGemmaRefinementReport {
    pub step: u32,
    pub cur_step: u32,
    pub temperature: f32,
    pub actions: Vec<DiffusionGemmaRefinementAction>,
    pub accepted_mask: Vec<bool>,
    pub accepted_count: usize,
    pub renoised_count: usize,
    /// Always false for `refine`; use `update_stopping` for this separate
    /// state transition.
    pub early_stop: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiffusionGemmaCanvasBatch {
    canvases: Vec<DiffusionGemmaCanvas>,
}

impl DiffusionGemmaCanvasBatch {
    pub fn new(canvases: Vec<DiffusionGemmaCanvas>) -> Result<Self, DiffusionGemmaSemanticError> {
        if canvases.is_empty() {
            return Err(DiffusionGemmaSemanticError::CanvasCountMismatch {
                expected: 1,
                actual: 0,
            });
        }
        Ok(Self { canvases })
    }

    pub fn canvases(&self) -> &[DiffusionGemmaCanvas] {
        &self.canvases
    }

    pub fn refine_all<R: DiffusionGemmaRandomSource>(
        &mut self,
        proposed_tokens: &[Vec<u32>],
        entropies: &[Vec<f32>],
        max_steps: u32,
        random: &mut R,
    ) -> Result<Vec<Option<DiffusionGemmaRefinementReport>>, DiffusionGemmaSemanticError> {
        let expected = self.canvases.len();
        for (field, actual) in [
            ("proposed_tokens", proposed_tokens.len()),
            ("entropies", entropies.len()),
        ] {
            if actual != expected {
                return Err(DiffusionGemmaSemanticError::CanvasCountMismatch { expected, actual });
            }
            let _ = field;
        }
        let mut reports = Vec::with_capacity(expected);
        for index in 0..expected {
            if self.canvases[index].committed {
                reports.push(None);
            } else {
                reports.push(Some(self.canvases[index].refine(
                    &proposed_tokens[index],
                    &entropies[index],
                    max_steps,
                    random,
                )?));
            }
        }
        Ok(reports)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_temperature_uses_remaining_steps_and_not_endpoint_47() {
        assert_eq!(diffusion_gemma_temperature(0, 48).unwrap(), 0.8);
        assert!((diffusion_gemma_temperature(47, 48).unwrap() - 0.40833333).abs() < 1e-6);
        assert_eq!(
            diffusion_gemma_temperature_for_cur_step(48, 48).unwrap(),
            0.8
        );
        assert!(
            (diffusion_gemma_temperature_for_cur_step(1, 48).unwrap() - 0.40833333).abs() < 1e-6
        );
        assert_eq!(diffusion_gemma_temperature(0, 1).unwrap(), 0.8);
        for (step, max_steps) in [(48, 48), (0, 0), (0, 49), (u32::MAX, 48)] {
            assert!(diffusion_gemma_temperature(step, max_steps).is_err());
        }
        assert!(diffusion_gemma_temperature_for_cur_step(0, 48).is_err());
    }

    #[test]
    fn attention_modes_have_independent_context_boundaries() {
        for length in [1, 31, 32, 33, 255, 256, 257, DIFFUSION_GEMMA_CONTEXT_LENGTH] {
            assert!(
                diffusion_gemma_attention_allowed(
                    DiffusionGemmaAttentionMode::CausalEncoder,
                    length - 1,
                    length - 1,
                    length,
                )
                .is_ok()
            );
        }
        assert!(
            diffusion_gemma_attention_allowed(
                DiffusionGemmaAttentionMode::CausalEncoder,
                0,
                0,
                DIFFUSION_GEMMA_CONTEXT_LENGTH + 1,
            )
            .is_err()
        );
        assert!(
            diffusion_gemma_attention_allowed(
                DiffusionGemmaAttentionMode::BidirectionalDecoder,
                256,
                256,
                257,
            )
            .is_err()
        );
        assert!(
            !diffusion_gemma_attention_allowed(
                DiffusionGemmaAttentionMode::CausalEncoder,
                2,
                3,
                32,
            )
            .unwrap()
        );
        assert!(
            diffusion_gemma_attention_allowed(
                DiffusionGemmaAttentionMode::BidirectionalDecoder,
                2,
                3,
                32,
            )
            .unwrap()
        );
    }

    #[test]
    fn canvas_boundaries_and_explicit_uniform_rng_are_checked() {
        for length in [1, 31, 32, 33, 255, 256] {
            let mut random = DiffusionGemmaSequenceRandom::new(&[0; 256]);
            let canvas = DiffusionGemmaCanvas::uniform_state(length, 17, &mut random).unwrap();
            assert_eq!(canvas.tokens().len(), length);
            assert!(canvas.tokens().iter().all(|token| *token < 17));
            assert_eq!(random.position(), length);
        }
        let mut random = DiffusionGemmaSequenceRandom::new(&[0; 257]);
        assert!(DiffusionGemmaCanvas::uniform_state(257, 17, &mut random).is_err());
        assert!(DiffusionGemmaCanvas::uniform_state(1, 0, &mut random).is_err());
        let mut exhausted = DiffusionGemmaSequenceRandom::new(&[]);
        assert!(DiffusionGemmaCanvas::uniform_state(1, 17, &mut exhausted).is_err());
        // Rejection is explicit and deterministic; the max draw is not a
        // modulo-biased sample.
        let mut rejected = DiffusionGemmaSequenceRandom::new(&[u64::MAX, 4]);
        let canvas = DiffusionGemmaCanvas::uniform_state(1, 4, &mut rejected).unwrap();
        assert_eq!(canvas.tokens(), &[0]);
        assert_eq!(rejected.position(), 2);
    }

    #[test]
    fn self_conditioning_describes_previous_processed_logits_only() {
        assert!(
            diffusion_gemma_self_conditioning(0, 32, 128, None)
                .unwrap()
                .is_none()
        );
        let logits = DiffusionGemmaProcessedLogits {
            source_step: 0,
            canvas_length: 32,
            vocabulary_size: 128,
        };
        assert!(diffusion_gemma_self_conditioning(0, 32, 128, Some(logits)).is_err());
        assert!(diffusion_gemma_self_conditioning(1, 32, 128, None).is_err());
        let condition = diffusion_gemma_self_conditioning(1, 32, 128, Some(logits))
            .unwrap()
            .unwrap();
        assert_eq!(condition.source_step(), 0);
        assert_eq!(condition.previous_logits().canvas_length(), 32);
        assert!(diffusion_gemma_self_conditioning(1, 31, 128, Some(logits)).is_err());
        assert!(diffusion_gemma_self_conditioning(2, 32, 128, Some(logits)).is_err());
    }

    #[test]
    fn entropy_selection_is_cumulative_tie_deterministic_and_fail_closed() {
        assert_eq!(
            diffusion_gemma_entropy_bound_selection(&[0.06, 0.06, 0.06], 0.1).unwrap(),
            vec![true, true, false]
        );
        // Equal values are ordered by source position, but the mask remains
        // permutation-invariant for this exact tie.
        assert_eq!(
            diffusion_gemma_entropy_bound_selection(&[0.05, 0.05, 0.2], 0.05).unwrap(),
            vec![true, true, false]
        );
        // Match the official FP32 cumsum/subtract boundary exactly. An f64
        // accumulator changes the second decision for these representable
        // values.
        assert_eq!(
            diffusion_gemma_entropy_bound_selection(&[0.10000001_f32, 0.15322891], 0.1).unwrap(),
            vec![true, true]
        );
        assert!(diffusion_gemma_entropy_bound_selection(&[f32::NAN], 0.1).is_err());
        assert!(diffusion_gemma_entropy_bound_selection(&[-1.0], 0.1).is_err());
        assert!(diffusion_gemma_entropy_bound_selection(&[f32::MAX, f32::MAX], 0.1).is_err());
        assert!(diffusion_gemma_entropy_bound_selection(&[0.1], 0.0).is_err());
        assert!(diffusion_gemma_normalized_entropy(&[f32::NAN]).is_err());
        assert!(diffusion_gemma_normalized_entropy(&[-1.0, 2.0]).is_err());
        assert!(diffusion_gemma_entropy_within_bound(f32::INFINITY).is_err());
        assert!(diffusion_gemma_categorical_entropy(&[0.0, 0.0]).unwrap() > 0.69);
        assert!(diffusion_gemma_categorical_entropy(&[f32::NAN]).is_err());
    }

    #[test]
    fn renoise_changes_only_rejected_positions_but_consumes_a_full_canvas() {
        let mut random = DiffusionGemmaSequenceRandom::new(&[0, 0, 0]);
        let mut canvas = DiffusionGemmaCanvas::uniform_state(3, 100, &mut random).unwrap();
        let before = canvas.tokens().to_vec();
        let mut random = DiffusionGemmaSequenceRandom::new(&[77, 88, 99]);
        let report = canvas
            .refine(&[10, 11, 12], &[0.06, 0.06, 0.06], 48, &mut random)
            .unwrap();
        assert_eq!(report.accepted_mask, vec![true, true, false]);
        assert_eq!(report.accepted_count, 2);
        assert_eq!(report.renoised_count, 1);
        assert_eq!(&canvas.tokens()[..2], &[10, 11]);
        assert_eq!(canvas.tokens()[2], 99);
        assert_ne!(canvas.tokens()[2], before[2]);
        assert_eq!(random.position(), 3);
        assert!(!report.early_stop);
        assert_eq!(canvas.accepted_mask(), &[true, true, false]);
        assert!(canvas.commit().is_err());
        canvas.mark_finished(&[21, 22, 23]).unwrap();
        assert_eq!(canvas.commit().unwrap().tokens, vec![21, 22, 23]);
        assert!(canvas.commit().is_err());
    }

    #[test]
    fn stopping_is_strict_and_separate_from_acceptance() {
        let mut stopping = DiffusionGemmaStoppingCriteria::new(1, 0.005).unwrap();
        assert!(!stopping.should_stop(&[1], &[0.005]).unwrap());
        assert!(stopping.should_stop(&[1], &[0.004]).unwrap());
        assert!(!stopping.should_stop(&[2], &[0.004]).unwrap());
        stopping.reset();
        assert_eq!(stopping.history_len(), 0);
        let default = DiffusionGemmaStoppingCriteria::default_contract();
        assert_eq!(default.confidence_threshold(), 0.005);
        assert_eq!(default.stability_threshold(), 1);
        assert!(DiffusionGemmaStoppingCriteria::new(1, 0.0).is_err());
    }

    #[test]
    fn canvas_self_conditioning_advances_with_processed_logits_descriptor() {
        let mut random = DiffusionGemmaSequenceRandom::new(&[0]);
        let mut canvas = DiffusionGemmaCanvas::uniform_state(1, 10, &mut random).unwrap();
        assert!(canvas.self_conditioning().unwrap().is_none());
        let mut random = DiffusionGemmaSequenceRandom::new(&[3]);
        canvas.refine(&[1], &[0.0], 48, &mut random).unwrap();
        assert_eq!(random.position(), 1);
        let condition = canvas.self_conditioning().unwrap().unwrap();
        assert_eq!(condition.source_step(), 0);
        assert_eq!(condition.previous_logits().vocabulary_size(), 10);
    }

    #[test]
    fn multicanvas_count_mismatch_is_explicit() {
        let mut random = DiffusionGemmaSequenceRandom::new(&[1, 2]);
        let a = DiffusionGemmaCanvas::uniform_state(1, 10, &mut random).unwrap();
        let b = DiffusionGemmaCanvas::uniform_state(1, 10, &mut random).unwrap();
        let mut batch = DiffusionGemmaCanvasBatch::new(vec![a, b]).unwrap();
        let mut refine_random = DiffusionGemmaSequenceRandom::new(&[3, 4]);
        assert!(
            batch
                .refine_all(&[vec![1]], &[vec![0.0]], 48, &mut refine_random)
                .is_err()
        );
        let reports = batch
            .refine_all(
                &[vec![1], vec![2]],
                &[vec![0.0], vec![0.0]],
                48,
                &mut refine_random,
            )
            .unwrap();
        assert_eq!(reports.len(), 2);
        assert!(reports.iter().all(|report| report.is_some()));
        assert_eq!(refine_random.position(), 2);
    }
}
