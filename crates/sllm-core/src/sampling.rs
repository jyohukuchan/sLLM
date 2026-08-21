// Portions derived from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-profile-v1-sampling-001 and
// THIRD_PARTY_NOTICES.md#llama-cpp-profile-v1-sampling-tests-001
// Upstream: https://github.com/ggml-org/llama.cpp @ f5919bf458ef190468b5c329bb293f8a54a1e69c, src/llama-sampler.cpp
// SPDX-License-Identifier: MIT

//! Profile-v1 CPU sampling over one bounded full-vocabulary logits row.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::Read;

/// Validated OpenAI profile-v1 sampling parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParametersV1 {
    temperature: f32,
    top_p: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
}

impl SamplingParametersV1 {
    pub fn new(
        temperature: f32,
        top_p: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
    ) -> Result<Self, SamplingError> {
        if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
            return Err(SamplingError::InvalidTemperature);
        }
        if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
            return Err(SamplingError::InvalidTopP);
        }
        if !presence_penalty.is_finite() || !(-2.0..=2.0).contains(&presence_penalty) {
            return Err(SamplingError::InvalidPresencePenalty);
        }
        if !frequency_penalty.is_finite() || !(-2.0..=2.0).contains(&frequency_penalty) {
            return Err(SamplingError::InvalidFrequencyPenalty);
        }
        Ok(Self {
            temperature,
            top_p,
            presence_penalty,
            frequency_penalty,
        })
    }

    pub const fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }

    pub const fn temperature(self) -> f32 {
        self.temperature
    }

    pub const fn top_p(self) -> f32 {
        self.top_p
    }

    pub const fn presence_penalty(self) -> f32 {
        self.presence_penalty
    }

    pub const fn frequency_penalty(self) -> f32 {
        self.frequency_penalty
    }

    pub const fn requires_logits(self) -> bool {
        self.temperature > 0.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SamplingError {
    InvalidTemperature,
    InvalidTopP,
    InvalidPresencePenalty,
    InvalidFrequencyPenalty,
    EmptyLogits,
    MissingLogits,
    InvalidGreedyToken,
    TokenIdOverflow,
    CountOverflow,
    NanLogit { token_id: u32 },
    EmptyDistribution,
    RandomSourceUnavailable,
    InvalidRandomValue,
    InvalidTopK,
    InvalidMinP,
    InvalidTypicalP,
    InvalidRepeatPenalty,
    InvalidRepeatWindow,
    InvalidDynamicTemperature,
    InvalidLogitBias,
    DuplicateLogitBias { token_id: u32 },
    InvalidDrySampling,
    InvalidXtcSampling,
    InvalidMirostat,
    ConflictingTerminalSamplers,
    InvalidTopLogprobs,
    TokenIdOutOfRange { token_id: u32 },
    InvalidMaskLength,
    UnsupportedDeviceSelector,
}

impl fmt::Display for SamplingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTemperature => {
                formatter.write_str("temperature must be finite and in [0,2]")
            }
            Self::InvalidTopP => formatter.write_str("top_p must be finite and in [0,1]"),
            Self::InvalidPresencePenalty => {
                formatter.write_str("presence_penalty must be finite and in [-2,2]")
            }
            Self::InvalidFrequencyPenalty => {
                formatter.write_str("frequency_penalty must be finite and in [-2,2]")
            }
            Self::EmptyLogits => formatter.write_str("sampling requires a nonempty logits row"),
            Self::MissingLogits => {
                formatter.write_str("non-greedy sampling requires full-vocabulary logits")
            }
            Self::InvalidGreedyToken => {
                formatter.write_str("device Argmax token is outside the logits vocabulary")
            }
            Self::TokenIdOverflow => formatter.write_str("vocabulary index does not fit u32"),
            Self::CountOverflow => formatter.write_str("token frequency count overflowed"),
            Self::NanLogit { token_id } => write!(formatter, "logit for token {token_id} is NaN"),
            Self::EmptyDistribution => {
                formatter.write_str("sampling distribution has no finite probability mass")
            }
            Self::RandomSourceUnavailable => {
                formatter.write_str("operating-system random source is unavailable")
            }
            Self::InvalidRandomValue => {
                formatter.write_str("random source returned a value outside [0,1)")
            }
            Self::InvalidTopK => {
                formatter.write_str("top_k must be zero or a positive bounded value")
            }
            Self::InvalidMinP => formatter.write_str("min_p must be finite and in [0,1]"),
            Self::InvalidTypicalP => formatter.write_str("typical_p must be finite and in (0,1]"),
            Self::InvalidRepeatPenalty => {
                formatter.write_str("repeat penalty must be finite and strictly positive")
            }
            Self::InvalidRepeatWindow => {
                formatter.write_str("repeat window is outside the bounded range")
            }
            Self::InvalidDynamicTemperature => {
                formatter.write_str("dynamic temperature bounds/exponent are invalid")
            }
            Self::InvalidLogitBias => {
                formatter.write_str("logit bias must be finite and in [-100,100]")
            }
            Self::DuplicateLogitBias { token_id } => {
                write!(formatter, "duplicate logit bias for token {token_id}")
            }
            Self::InvalidDrySampling => {
                formatter.write_str("DRY sampling configuration is invalid")
            }
            Self::InvalidXtcSampling => {
                formatter.write_str("XTC sampling configuration is invalid")
            }
            Self::InvalidMirostat => formatter.write_str("Mirostat configuration is invalid"),
            Self::ConflictingTerminalSamplers => {
                formatter.write_str("Mirostat cannot be combined with another terminal sampler")
            }
            Self::InvalidTopLogprobs => formatter.write_str("top_logprobs must be in [0,20]"),
            Self::TokenIdOutOfRange { token_id } => {
                write!(
                    formatter,
                    "token {token_id} is outside the logits vocabulary"
                )
            }
            Self::InvalidMaskLength => {
                formatter.write_str("sampling mask length must equal the logits vocabulary")
            }
            Self::UnsupportedDeviceSelector => {
                formatter.write_str("sampler chain cannot use the prepared device selector subset")
            }
        }
    }
}

impl std::error::Error for SamplingError {}

/// Internal randomness seam shared by OS-seeded and explicitly seeded requests.
pub trait SamplingRandomSource {
    fn next_unit_f64(&mut self) -> Result<f64, SamplingError>;
}

/// Linux OS-seeded per-request generator.
#[derive(Debug)]
pub struct OsSamplingRandom {
    state: u64,
}

impl OsSamplingRandom {
    pub fn new() -> Result<Self, SamplingError> {
        Ok(Self {
            state: Self::resolve_seed(None)?,
        })
    }

    pub fn resolve_seed(seed: Option<u64>) -> Result<u64, SamplingError> {
        if let Some(seed) = seed {
            return Ok(seed);
        }
        let mut seed = [0_u8; 8];
        File::open("/dev/urandom")
            .and_then(|mut file| file.read_exact(&mut seed))
            .map_err(|_| SamplingError::RandomSourceUnavailable)?;
        Ok(u64::from_le_bytes(seed))
    }

    /// Avoids touching the OS random source for the strict greedy path, where
    /// [`ProfileSamplerV1`] guarantees that randomness is never observed.
    pub fn for_parameters(parameters: SamplingParametersV1) -> Result<Self, SamplingError> {
        Self::for_parameters_and_seed(parameters, None)
    }

    pub fn for_parameters_and_seed(
        parameters: SamplingParametersV1,
        seed: Option<u64>,
    ) -> Result<Self, SamplingError> {
        Self::for_randomness_and_seed(parameters.requires_logits(), seed)
    }

    /// Creates a request-local stream when an extended sampler consumes
    /// randomness even though the legacy temperature is zero.
    pub fn for_randomness_and_seed(
        requires_randomness: bool,
        seed: Option<u64>,
    ) -> Result<Self, SamplingError> {
        if requires_randomness {
            match seed {
                Some(state) => Ok(Self { state }),
                None => Self::new(),
            }
        } else {
            Ok(Self { state: 0 })
        }
    }

    fn next_u64(&mut self) -> u64 {
        // SplitMix64 is used after the state is seeded by the OS or supplied
        // explicitly by the request. It is adequate for categorical sampling
        // and keeps the reproducible RNG seam tiny.
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

impl SamplingRandomSource for OsSamplingRandom {
    fn next_unit_f64(&mut self) -> Result<f64, SamplingError> {
        Ok(((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64)))
    }
}

#[derive(Clone, Debug)]
pub struct ProfileSamplerV1 {
    parameters: SamplingParametersV1,
    token_counts: BTreeMap<u32, u32>,
}

impl ProfileSamplerV1 {
    pub fn new(
        parameters: SamplingParametersV1,
        prior_tokens: &[u32],
    ) -> Result<Self, SamplingError> {
        let mut sampler = Self {
            parameters,
            token_counts: BTreeMap::new(),
        };
        for &token in prior_tokens {
            sampler.accept(token)?;
        }
        Ok(sampler)
    }

    pub const fn parameters(&self) -> SamplingParametersV1 {
        self.parameters
    }

    pub fn accept(&mut self, token_id: u32) -> Result<(), SamplingError> {
        let count = self.token_counts.entry(token_id).or_default();
        *count = count.checked_add(1).ok_or(SamplingError::CountOverflow)?;
        Ok(())
    }

    /// Selects one token. `temperature=0` returns the device Argmax without
    /// inspecting or allocating a host logits row.
    pub fn select(
        &self,
        device_argmax: u32,
        logits: Option<&[f32]>,
        random: &mut impl SamplingRandomSource,
    ) -> Result<u32, SamplingError> {
        if !self.parameters.requires_logits() {
            if let Some(logits) = logits {
                if usize::try_from(device_argmax).map_or(true, |id| id >= logits.len()) {
                    return Err(SamplingError::InvalidGreedyToken);
                }
            }
            return Ok(device_argmax);
        }
        let logits = logits.ok_or(SamplingError::MissingLogits)?;
        if logits.is_empty() {
            return Err(SamplingError::EmptyLogits);
        }

        let mut candidates = Vec::with_capacity(logits.len());
        let mut positive_infinity = Vec::new();
        for (index, &raw) in logits.iter().enumerate() {
            let token_id = u32::try_from(index).map_err(|_| SamplingError::TokenIdOverflow)?;
            if raw.is_nan() {
                return Err(SamplingError::NanLogit { token_id });
            }
            let count = self.token_counts.get(&token_id).copied().unwrap_or(0);
            let penalized = raw
                - (count as f32) * self.parameters.frequency_penalty
                - if count > 0 {
                    self.parameters.presence_penalty
                } else {
                    0.0
                };
            let scaled = penalized / self.parameters.temperature;
            if scaled == f32::INFINITY {
                positive_infinity.push(token_id);
            } else if scaled != f32::NEG_INFINITY {
                candidates.push(Candidate {
                    token_id,
                    scaled_logit: scaled,
                    probability: 0.0,
                });
            }
        }
        if !positive_infinity.is_empty() {
            return select_uniform(&positive_infinity, random);
        }
        if candidates.is_empty() {
            return Err(SamplingError::EmptyDistribution);
        }

        candidates.sort_by(|left, right| {
            right
                .scaled_logit
                .total_cmp(&left.scaled_logit)
                .then_with(|| left.token_id.cmp(&right.token_id))
        });
        let max = f64::from(candidates[0].scaled_logit);
        let mut total = 0.0_f64;
        for candidate in &mut candidates {
            candidate.probability = (f64::from(candidate.scaled_logit) - max).exp();
            total += candidate.probability;
        }
        if !total.is_finite() || total <= 0.0 {
            return Err(SamplingError::EmptyDistribution);
        }
        for candidate in &mut candidates {
            candidate.probability /= total;
        }

        let mut keep = 1_usize;
        let mut cumulative = 0.0_f64;
        for (index, candidate) in candidates.iter().enumerate() {
            cumulative += candidate.probability;
            keep = index + 1;
            if cumulative >= f64::from(self.parameters.top_p) {
                break;
            }
        }
        candidates.truncate(keep);
        let retained_total: f64 = candidates
            .iter()
            .map(|candidate| candidate.probability)
            .sum();
        if retained_total <= 0.0 || !retained_total.is_finite() {
            return Err(SamplingError::EmptyDistribution);
        }
        let sample = random.next_unit_f64()?;
        if !(0.0..1.0).contains(&sample) {
            return Err(SamplingError::InvalidRandomValue);
        }
        let threshold = sample * retained_total;
        let mut cumulative = 0.0;
        for candidate in &candidates {
            cumulative += candidate.probability;
            if threshold < cumulative {
                return Ok(candidate.token_id);
            }
        }
        Ok(candidates
            .last()
            .expect("nonempty retained candidates")
            .token_id)
    }
}

/// A sparse additive logit bias. Biases are applied before temperature and
/// candidate filters. The chain sorts these entries by token id at creation
/// time so result ordering is deterministic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogitBiasV1 {
    pub token_id: u32,
    pub bias: f32,
}

/// Dynamic temperature parameters. The temperature is interpolated from
/// `min_temperature` to `max_temperature` using the normalized entropy of the
/// current row. This is deliberately bounded and request-local.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicTemperatureV1 {
    pub min_temperature: f32,
    pub max_temperature: f32,
    pub exponent: f32,
}

impl DynamicTemperatureV1 {
    pub fn new(
        min_temperature: f32,
        max_temperature: f32,
        exponent: f32,
    ) -> Result<Self, SamplingError> {
        if !min_temperature.is_finite()
            || !max_temperature.is_finite()
            || !exponent.is_finite()
            || min_temperature <= 0.0
            || max_temperature < min_temperature
            || exponent <= 0.0
            || max_temperature > 2.0
        {
            return Err(SamplingError::InvalidDynamicTemperature);
        }
        Ok(Self {
            min_temperature,
            max_temperature,
            exponent,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DrySamplingConfigV1 {
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: usize,
    /// Maximum recent history considered by DRY.  This is independent from
    /// the full frequency/presence counts retained by the sampler.
    pub penalty_last_n: usize,
    pub sequence_breakers: Vec<Vec<u32>>,
}

impl DrySamplingConfigV1 {
    pub fn new(
        multiplier: f32,
        base: f32,
        allowed_length: usize,
        sequence_breakers: Vec<Vec<u32>>,
    ) -> Result<Self, SamplingError> {
        if !multiplier.is_finite()
            || !(0.0..=100.0).contains(&multiplier)
            || !base.is_finite()
            || !(1.0..=4.0).contains(&base)
            || allowed_length > MAX_SAMPLING_HISTORY
            || sequence_breakers.len() > MAX_SEQUENCE_BREAKERS
            || sequence_breakers.iter().any(Vec::is_empty)
            || sequence_breakers
                .iter()
                .try_fold(0_usize, |total, breaker| total.checked_add(breaker.len()))
                .is_none_or(|total| total > MAX_SEQUENCE_BREAKER_TOKENS)
        {
            return Err(SamplingError::InvalidDrySampling);
        }
        Ok(Self {
            multiplier,
            base,
            allowed_length,
            penalty_last_n: MAX_SAMPLING_HISTORY,
            sequence_breakers,
        })
    }

    pub fn with_penalty_last_n(mut self, penalty_last_n: usize) -> Result<Self, SamplingError> {
        if penalty_last_n > MAX_SAMPLING_HISTORY {
            return Err(SamplingError::InvalidDrySampling);
        }
        self.penalty_last_n = penalty_last_n;
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct XtcSamplingConfigV1 {
    pub probability: f32,
    pub threshold: f32,
    pub min_keep: usize,
}

impl XtcSamplingConfigV1 {
    pub fn new(probability: f32, threshold: f32, min_keep: usize) -> Result<Self, SamplingError> {
        if !probability.is_finite()
            || !(0.0..=1.0).contains(&probability)
            || !threshold.is_finite()
            || !(0.0..=1.0).contains(&threshold)
            || min_keep == 0
            || min_keep > MAX_CANDIDATES
        {
            return Err(SamplingError::InvalidXtcSampling);
        }
        Ok(Self {
            probability,
            threshold,
            min_keep,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MirostatModeV1 {
    V1,
    V2,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MirostatSamplingConfigV1 {
    pub mode: MirostatModeV1,
    pub tau: f32,
    pub eta: f32,
    pub mu: f32,
}

impl MirostatSamplingConfigV1 {
    pub fn new(mode: MirostatModeV1, tau: f32, eta: f32, mu: f32) -> Result<Self, SamplingError> {
        if !tau.is_finite()
            || tau <= 0.0
            || !eta.is_finite()
            || !(0.0..=1.0).contains(&eta)
            || !mu.is_finite()
            || mu <= 0.0
        {
            return Err(SamplingError::InvalidMirostat);
        }
        Ok(Self { mode, tau, eta, mu })
    }
}

pub const MAX_SAMPLING_HISTORY: usize = 16_384;
pub const MAX_SEQUENCE_BREAKERS: usize = 256;
pub const MAX_SEQUENCE_BREAKER_TOKENS: usize = 4_096;
pub const MAX_CANDIDATES: usize = 1_048_576;
pub const SAMPLER_CHAIN_SCHEMA_V1: &str = "sampler-chain-v1";

/// Stable stage identifiers. The selector applies these in this exact order;
/// backends consume the same ordered contract rather than reimplementing
/// policy-specific stage ordering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplerStageV1 {
    LogitBias,
    LegacyPenalty,
    RepeatPenalty,
    DryPenalty,
    GrammarMask,
    IgnoreEos,
    Temperature,
    TopK,
    MinP,
    Typical,
    TopP,
    Xtc,
    Mirostat,
    Logprobs,
}

pub const SAMPLER_STAGE_ORDER_V1: &[SamplerStageV1] = &[
    SamplerStageV1::LogitBias,
    SamplerStageV1::LegacyPenalty,
    SamplerStageV1::RepeatPenalty,
    SamplerStageV1::DryPenalty,
    SamplerStageV1::GrammarMask,
    SamplerStageV1::IgnoreEos,
    SamplerStageV1::Temperature,
    SamplerStageV1::TopK,
    SamplerStageV1::MinP,
    SamplerStageV1::Typical,
    SamplerStageV1::TopP,
    SamplerStageV1::Xtc,
    SamplerStageV1::Mirostat,
    SamplerStageV1::Logprobs,
];

/// Backend-neutral, versioned sampler-chain configuration. All optional
/// stages are disabled by default; `legacy` therefore retains the exact
/// profile-v1 path, including greedy Argmax's no-logits/no-RNG behavior.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplerChainConfigV1 {
    pub parameters: SamplingParametersV1,
    pub top_k: Option<usize>,
    pub min_p: f32,
    pub typical_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub dynamic_temperature: Option<DynamicTemperatureV1>,
    pub ignore_eos_token: Option<u32>,
    pub logit_bias: Vec<LogitBiasV1>,
    pub dry: Option<DrySamplingConfigV1>,
    pub xtc: Option<XtcSamplingConfigV1>,
    pub mirostat: Option<MirostatSamplingConfigV1>,
    /// Return selected logprob metadata even when `top_logprobs` is zero.
    pub return_logprobs: bool,
    pub top_logprobs: usize,
}

impl SamplerChainConfigV1 {
    pub fn new(parameters: SamplingParametersV1) -> Self {
        Self {
            parameters,
            top_k: None,
            min_p: 0.0,
            typical_p: 1.0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            dynamic_temperature: None,
            ignore_eos_token: None,
            logit_bias: Vec::new(),
            dry: None,
            xtc: None,
            mirostat: None,
            return_logprobs: false,
            top_logprobs: 0,
        }
    }

    pub fn legacy(parameters: SamplingParametersV1) -> Self {
        Self::new(parameters)
    }

    pub const fn schema_version() -> &'static str {
        SAMPLER_CHAIN_SCHEMA_V1
    }

    pub const fn stage_order() -> &'static [SamplerStageV1] {
        SAMPLER_STAGE_ORDER_V1
    }

    pub fn requires_logits(&self) -> bool {
        self.parameters.requires_logits() || self.has_extensions()
    }

    pub fn requires_randomness(&self) -> bool {
        self.parameters.temperature() > 0.0
            || self.dynamic_temperature.is_some()
            || self.xtc.is_some_and(|xtc| xtc.probability > 0.0)
            || self.mirostat.is_some()
    }

    pub fn validate(&self) -> Result<(), SamplingError> {
        if let Some(k) = self.top_k {
            if k > MAX_CANDIDATES {
                return Err(SamplingError::InvalidTopK);
            }
        }
        if !self.min_p.is_finite() || !(0.0..=1.0).contains(&self.min_p) {
            return Err(SamplingError::InvalidMinP);
        }
        if !self.typical_p.is_finite() || self.typical_p <= 0.0 || self.typical_p > 1.0 {
            return Err(SamplingError::InvalidTypicalP);
        }
        if !self.repeat_penalty.is_finite() || self.repeat_penalty <= 0.0 {
            return Err(SamplingError::InvalidRepeatPenalty);
        }
        if self.repeat_last_n > MAX_SAMPLING_HISTORY {
            return Err(SamplingError::InvalidRepeatWindow);
        }
        if self.top_logprobs > 20 {
            return Err(SamplingError::InvalidTopLogprobs);
        }
        if let Some(dynamic) = self.dynamic_temperature {
            DynamicTemperatureV1::new(
                dynamic.min_temperature,
                dynamic.max_temperature,
                dynamic.exponent,
            )?;
        }
        if let Some(dry) = &self.dry {
            DrySamplingConfigV1::new(
                dry.multiplier,
                dry.base,
                dry.allowed_length,
                dry.sequence_breakers.clone(),
            )?
            .with_penalty_last_n(dry.penalty_last_n)?;
        }
        if let Some(xtc) = self.xtc {
            XtcSamplingConfigV1::new(xtc.probability, xtc.threshold, xtc.min_keep)?;
        }
        if let Some(mirostat) = self.mirostat {
            MirostatSamplingConfigV1::new(mirostat.mode, mirostat.tau, mirostat.eta, mirostat.mu)?;
        }
        let mut previous = None;
        for entry in &self.logit_bias {
            if !entry.bias.is_finite() || !(-100.0..=100.0).contains(&entry.bias) {
                return Err(SamplingError::InvalidLogitBias);
            }
            if previous == Some(entry.token_id) {
                return Err(SamplingError::DuplicateLogitBias {
                    token_id: entry.token_id,
                });
            }
            if previous.is_some_and(|token| token > entry.token_id) {
                return Err(SamplingError::InvalidLogitBias);
            }
            previous = Some(entry.token_id);
        }
        if self.mirostat.is_some()
            && (self.parameters.top_p() < 1.0
                || self.top_k.is_some_and(|k| k > 0)
                || self.min_p > 0.0
                || self.typical_p < 1.0
                || self.xtc.is_some())
        {
            return Err(SamplingError::ConflictingTerminalSamplers);
        }
        Ok(())
    }

    pub fn with_top_k(mut self, top_k: usize) -> Result<Self, SamplingError> {
        self.top_k = Some(top_k);
        self.validate()?;
        Ok(self)
    }

    pub fn with_min_p(mut self, min_p: f32) -> Result<Self, SamplingError> {
        self.min_p = min_p;
        self.validate()?;
        Ok(self)
    }

    pub fn with_typical_p(mut self, typical_p: f32) -> Result<Self, SamplingError> {
        self.typical_p = typical_p;
        self.validate()?;
        Ok(self)
    }

    pub fn with_repeat_penalty(
        mut self,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Result<Self, SamplingError> {
        self.repeat_penalty = repeat_penalty;
        self.repeat_last_n = repeat_last_n;
        self.validate()?;
        Ok(self)
    }

    pub fn with_logit_bias(mut self, mut entries: Vec<LogitBiasV1>) -> Result<Self, SamplingError> {
        entries.sort_by_key(|entry| entry.token_id);
        self.logit_bias = entries;
        self.validate()?;
        Ok(self)
    }

    pub fn with_top_logprobs(mut self, count: usize) -> Result<Self, SamplingError> {
        self.top_logprobs = count;
        self.validate()?;
        Ok(self)
    }

    pub fn with_return_logprobs(mut self, enabled: bool) -> Self {
        self.return_logprobs = enabled;
        self
    }

    pub fn with_dynamic_temperature(
        mut self,
        dynamic: DynamicTemperatureV1,
    ) -> Result<Self, SamplingError> {
        self.dynamic_temperature = Some(dynamic);
        self.validate()?;
        Ok(self)
    }

    pub fn with_ignore_eos(mut self, token_id: u32) -> Self {
        self.ignore_eos_token = Some(token_id);
        self
    }

    pub fn with_dry(mut self, dry: DrySamplingConfigV1) -> Result<Self, SamplingError> {
        self.dry = Some(dry);
        self.validate()?;
        Ok(self)
    }

    pub fn with_xtc(mut self, xtc: XtcSamplingConfigV1) -> Result<Self, SamplingError> {
        self.xtc = Some(xtc);
        self.validate()?;
        Ok(self)
    }

    pub fn with_mirostat(
        mut self,
        mirostat: MirostatSamplingConfigV1,
    ) -> Result<Self, SamplingError> {
        self.mirostat = Some(mirostat);
        self.validate()?;
        Ok(self)
    }

    fn has_extensions(&self) -> bool {
        self.top_k.is_some_and(|k| k > 0)
            || self.min_p > 0.0
            || self.typical_p < 1.0
            || (self.repeat_penalty != 1.0 && self.repeat_last_n > 0)
            || self.dynamic_temperature.is_some()
            || self.ignore_eos_token.is_some()
            || !self.logit_bias.is_empty()
            || self.dry.as_ref().is_some_and(|dry| {
                dry.multiplier > 0.0 && dry.allowed_length > 0 && dry.penalty_last_n > 0
            })
            || self.xtc.is_some_and(|xtc| xtc.probability > 0.0)
            || self.mirostat.is_some()
            || self.return_logprobs
            || self.top_logprobs > 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingLogprobV1 {
    pub token_id: u32,
    pub logprob: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamplingSelectionV1 {
    pub token_id: u32,
    pub logprob: f64,
    pub top_logprobs: Vec<SamplingLogprobV1>,
}

/// Inputs for the bounded M=1 prepared device selector subset.  The vectors
/// are vocabulary-sized and are uploaded on the model execution queue; only
/// the fixed-size selected record is read back.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceTokenSelectorRequestV1 {
    additive_logits: Vec<f32>,
    valid_mask: Vec<u8>,
    temperature: f32,
    seed: u64,
    counter: u64,
    return_logprob: bool,
}

impl DeviceTokenSelectorRequestV1 {
    pub fn additive_logits(&self) -> &[f32] {
        &self.additive_logits
    }

    pub fn valid_mask(&self) -> &[u8] {
        &self.valid_mask
    }

    pub const fn temperature(&self) -> f32 {
        self.temperature
    }

    pub const fn seed(&self) -> u64 {
        self.seed
    }

    pub const fn counter(&self) -> u64 {
        self.counter
    }

    pub const fn return_logprob(&self) -> bool {
        self.return_logprob
    }
}

#[derive(Clone, Debug)]
pub struct SamplerChainV1 {
    config: SamplerChainConfigV1,
    token_counts: BTreeMap<u32, u32>,
    history: Vec<u32>,
    mirostat_mu: f64,
}

impl SamplerChainV1 {
    pub fn new(config: SamplerChainConfigV1, prior_tokens: &[u32]) -> Result<Self, SamplingError> {
        config.validate()?;
        let mut token_counts: BTreeMap<u32, u32> = BTreeMap::new();
        for &token in prior_tokens {
            let count = token_counts.entry(token).or_insert(0);
            *count = (*count)
                .checked_add(1)
                .ok_or(SamplingError::CountOverflow)?;
        }
        let mirostat_mu = config
            .mirostat
            .map_or(0.0, |settings| f64::from(settings.mu));
        let history = if prior_tokens.len() > MAX_SAMPLING_HISTORY {
            prior_tokens[prior_tokens.len() - MAX_SAMPLING_HISTORY..].to_vec()
        } else {
            prior_tokens.to_vec()
        };
        Ok(Self {
            config,
            token_counts,
            history,
            mirostat_mu,
        })
    }

    pub fn config(&self) -> &SamplerChainConfigV1 {
        &self.config
    }

    pub fn accept(&mut self, token_id: u32) -> Result<(), SamplingError> {
        let count = self.token_counts.entry(token_id).or_default();
        *count = count.checked_add(1).ok_or(SamplingError::CountOverflow)?;
        self.history.push(token_id);
        if self.history.len() > MAX_SAMPLING_HISTORY {
            self.history.remove(0);
        }
        Ok(())
    }

    pub fn requires_logits(&self) -> bool {
        self.config.parameters.requires_logits() || self.config.has_extensions()
    }

    pub fn supports_device_selector(&self) -> bool {
        self.config.parameters.temperature() > 0.0
            && self.config.parameters.top_p() == 1.0
            && !self.config.top_k.is_some_and(|top_k| top_k > 0)
            && self.config.min_p == 0.0
            && self.config.typical_p == 1.0
            && (self.config.repeat_penalty == 1.0 || self.config.repeat_last_n == 0)
            && self.config.dynamic_temperature.is_none()
            && !self.config.xtc.is_some_and(|xtc| xtc.probability > 0.0)
            && self.config.mirostat.is_none()
            && self.config.top_logprobs == 0
    }

    pub fn prepare_device_selector(
        &self,
        vocab_size: usize,
        valid_mask: Option<&[bool]>,
        seed: u64,
        counter: u64,
    ) -> Result<DeviceTokenSelectorRequestV1, SamplingError> {
        if !self.supports_device_selector() || vocab_size == 0 || vocab_size > MAX_CANDIDATES {
            return Err(SamplingError::UnsupportedDeviceSelector);
        }
        if valid_mask.is_some_and(|mask| mask.len() != vocab_size) {
            return Err(SamplingError::InvalidMaskLength);
        }
        let mut additive_logits = vec![0.0_f32; vocab_size];
        let mut output_mask = valid_mask.map_or_else(
            || vec![1_u8; vocab_size],
            |mask| mask.iter().map(|&valid| u8::from(valid)).collect(),
        );
        for entry in &self.config.logit_bias {
            let index =
                usize::try_from(entry.token_id).map_err(|_| SamplingError::TokenIdOutOfRange {
                    token_id: entry.token_id,
                })?;
            let Some(value) = additive_logits.get_mut(index) else {
                return Err(SamplingError::TokenIdOutOfRange {
                    token_id: entry.token_id,
                });
            };
            *value += entry.bias;
        }
        if let Some(token_id) = self.config.ignore_eos_token {
            let index = usize::try_from(token_id)
                .map_err(|_| SamplingError::TokenIdOutOfRange { token_id })?;
            let Some(valid) = output_mask.get_mut(index) else {
                return Err(SamplingError::TokenIdOutOfRange { token_id });
            };
            *valid = 0;
        }
        for (index, additive) in additive_logits.iter_mut().enumerate() {
            let token_id = u32::try_from(index).map_err(|_| SamplingError::TokenIdOverflow)?;
            let count = self.token_counts.get(&token_id).copied().unwrap_or(0);
            *additive -= count as f32 * self.config.parameters.frequency_penalty();
            if count > 0 {
                *additive -= self.config.parameters.presence_penalty();
            }
            if let Some(dry) = &self.config.dry {
                let repeated = dry_repetition(&self.history, token_id, dry);
                if repeated > 0 {
                    let penalty = dry.multiplier * (dry.base.powi(repeated as i32) - 1.0);
                    if !penalty.is_finite() {
                        return Err(SamplingError::InvalidDrySampling);
                    }
                    *additive -= penalty;
                }
            }
            if !additive.is_finite() {
                return Err(SamplingError::EmptyDistribution);
            }
        }
        if !output_mask.iter().any(|&value| value != 0) {
            return Err(SamplingError::EmptyDistribution);
        }
        Ok(DeviceTokenSelectorRequestV1 {
            additive_logits,
            valid_mask: output_mask,
            temperature: self.config.parameters.temperature(),
            seed,
            counter,
            return_logprob: self.config.return_logprobs,
        })
    }

    pub fn select_token(
        &mut self,
        device_argmax: u32,
        logits: Option<&[f32]>,
        random: &mut impl SamplingRandomSource,
    ) -> Result<u32, SamplingError> {
        Ok(self.select(device_argmax, logits, random)?.token_id)
    }

    /// Select one token and return metadata computed from the exact post-filter
    /// distribution. The legacy configuration delegates to ProfileSamplerV1;
    /// this is the compatibility seam that preserves its token and RNG stream.
    pub fn select(
        &mut self,
        device_argmax: u32,
        logits: Option<&[f32]>,
        random: &mut impl SamplingRandomSource,
    ) -> Result<SamplingSelectionV1, SamplingError> {
        self.select_with_mask(device_argmax, logits, None, random)
    }

    /// Applies an optional grammar-produced valid-token mask before candidate
    /// filters. Returned logprobs therefore describe the exact masked
    /// distribution used for selection.
    pub fn select_with_mask(
        &mut self,
        device_argmax: u32,
        logits: Option<&[f32]>,
        valid_mask: Option<&[bool]>,
        random: &mut impl SamplingRandomSource,
    ) -> Result<SamplingSelectionV1, SamplingError> {
        if let (Some(values), Some(mask)) = (logits, valid_mask)
            && values.len() != mask.len()
        {
            return Err(SamplingError::InvalidMaskLength);
        }
        if !self.config.has_extensions() && valid_mask.is_none() {
            let legacy = ProfileSamplerV1 {
                parameters: self.config.parameters,
                token_counts: self.token_counts.clone(),
            };
            let token_id = legacy.select(device_argmax, logits, random)?;
            let logprob = if self.config.parameters.requires_logits() {
                let values = logits.ok_or(SamplingError::MissingLogits)?;
                let distribution =
                    legacy_distribution(&self.config.parameters, &self.token_counts, values)?;
                distribution
                    .into_iter()
                    .find(|candidate| candidate.token_id == token_id)
                    .map_or(f64::NEG_INFINITY, |candidate| candidate.logprob)
            } else {
                0.0
            };
            return Ok(SamplingSelectionV1 {
                token_id,
                logprob,
                top_logprobs: Vec::new(),
            });
        }
        let values = logits.ok_or(SamplingError::MissingLogits)?;
        if values.is_empty() {
            return Err(SamplingError::EmptyLogits);
        }
        let mut candidates = self.prepare_candidates(values, valid_mask)?;
        if candidates.is_empty() {
            return Err(SamplingError::EmptyDistribution);
        }
        let deterministic = self.config.parameters.temperature() == 0.0
            && self.config.dynamic_temperature.is_none()
            && self.config.mirostat.is_none();
        let temperature = self.effective_temperature(&candidates);
        for candidate in &mut candidates {
            candidate.logit /= temperature;
        }
        candidates = softmax_candidates(candidates)?;
        apply_filters(&mut candidates, &self.config)?;
        if candidates.is_empty() {
            return Err(SamplingError::EmptyDistribution);
        }
        let token_id = if let Some(settings) = self.config.mirostat {
            self.select_mirostat(&mut candidates, settings, random)?
        } else {
            if let Some(xtc) = self.config.xtc {
                apply_xtc(&mut candidates, xtc, random)?;
            }
            if deterministic {
                candidates[0].token_id
            } else {
                sample_candidates(&candidates, random)?
            }
        };
        let selected_logprob = candidates
            .iter()
            .find(|candidate| candidate.token_id == token_id)
            .map_or(f64::NEG_INFINITY, |candidate| candidate.logprob);
        let mut top_logprobs = candidates
            .iter()
            .map(|candidate| SamplingLogprobV1 {
                token_id: candidate.token_id,
                logprob: candidate.logprob,
            })
            .collect::<Vec<_>>();
        top_logprobs.sort_by(|left, right| {
            right
                .logprob
                .total_cmp(&left.logprob)
                .then_with(|| left.token_id.cmp(&right.token_id))
        });
        top_logprobs.truncate(self.config.top_logprobs);
        Ok(SamplingSelectionV1 {
            token_id,
            logprob: selected_logprob,
            top_logprobs,
        })
    }

    fn prepare_candidates(
        &self,
        logits: &[f32],
        valid_mask: Option<&[bool]>,
    ) -> Result<Vec<ChainCandidate>, SamplingError> {
        for entry in &self.config.logit_bias {
            if usize::try_from(entry.token_id).map_or(true, |index| index >= logits.len()) {
                return Err(SamplingError::TokenIdOutOfRange {
                    token_id: entry.token_id,
                });
            }
        }
        let mut candidates = Vec::with_capacity(logits.len());
        for (index, &raw) in logits.iter().enumerate() {
            let token_id = u32::try_from(index).map_err(|_| SamplingError::TokenIdOverflow)?;
            if raw.is_nan() {
                return Err(SamplingError::NanLogit { token_id });
            }
            if self.config.ignore_eos_token == Some(token_id)
                || valid_mask.is_some_and(|mask| !mask[index])
            {
                continue;
            }
            let mut logit = raw;
            if let Some(entry) = self
                .config
                .logit_bias
                .binary_search_by_key(&token_id, |entry| entry.token_id)
                .ok()
                .map(|index| self.config.logit_bias[index])
            {
                logit += entry.bias;
            }
            let count = self.token_counts.get(&token_id).copied().unwrap_or(0);
            logit -= (count as f32) * self.config.parameters.frequency_penalty();
            if count > 0 {
                logit -= self.config.parameters.presence_penalty();
            }
            let recent_count = self
                .history
                .iter()
                .rev()
                .take(self.config.repeat_last_n)
                .filter(|&&value| value == token_id)
                .count() as u32;
            if recent_count > 0 && self.config.repeat_penalty != 1.0 {
                if logit >= 0.0 {
                    logit /= self.config.repeat_penalty;
                } else {
                    logit *= self.config.repeat_penalty;
                }
            }
            if let Some(dry) = &self.config.dry {
                let repeated = dry_repetition(&self.history, token_id, dry);
                if repeated > 0 {
                    let penalty = dry.multiplier * (dry.base.powi(repeated as i32) - 1.0);
                    if !penalty.is_finite() {
                        return Err(SamplingError::InvalidDrySampling);
                    }
                    logit -= penalty;
                }
            }
            if logit.is_nan() {
                return Err(SamplingError::NanLogit { token_id });
            }
            if logit != f32::NEG_INFINITY {
                candidates.push(ChainCandidate {
                    token_id,
                    logit,
                    probability: 0.0,
                    logprob: 0.0,
                });
            }
        }
        Ok(candidates)
    }

    fn effective_temperature(&self, candidates: &[ChainCandidate]) -> f32 {
        if let Some(dynamic) = self.config.dynamic_temperature {
            let max = candidates
                .iter()
                .map(|candidate| candidate.logit)
                .fold(f32::NEG_INFINITY, f32::max);
            let mut entropy = 0.0_f64;
            let mut total = 0.0_f64;
            for candidate in candidates {
                let weight = (f64::from(candidate.logit) - f64::from(max)).exp();
                total += weight;
                entropy -= weight * weight.max(f64::MIN_POSITIVE).ln();
            }
            let normalized = if total > 0.0 && total.is_finite() {
                let normalizer = if candidates.len() <= 1 {
                    1.0
                } else {
                    (candidates.len() as f64).ln()
                };
                (entropy / total + total.ln()) / normalizer
            } else {
                0.0
            };
            let fraction = normalized.clamp(0.0, 1.0).powf(f64::from(dynamic.exponent));
            dynamic.min_temperature
                + (dynamic.max_temperature - dynamic.min_temperature) * fraction as f32
        } else if self.config.parameters.temperature() > 0.0 {
            self.config.parameters.temperature()
        } else {
            1.0
        }
    }

    fn select_mirostat(
        &mut self,
        candidates: &mut Vec<ChainCandidate>,
        settings: MirostatSamplingConfigV1,
        random: &mut impl SamplingRandomSource,
    ) -> Result<u32, SamplingError> {
        let cutoff = (-self.mirostat_mu).exp();
        let mut keep = candidates
            .iter()
            .filter(|candidate| candidate.probability >= cutoff)
            .count();
        if keep == 0 {
            keep = 1;
        }
        keep = keep.min(candidates.len());
        candidates.truncate(keep);
        renormalize(candidates)?;
        let chosen = sample_candidates(candidates, random)?;
        let surprise = candidates
            .iter()
            .find(|candidate| candidate.token_id == chosen)
            .map_or(f64::INFINITY, |candidate| -candidate.logprob);
        let error = surprise - f64::from(settings.tau);
        self.mirostat_mu = match settings.mode {
            MirostatModeV1::V1 => self.mirostat_mu + f64::from(settings.eta) * error,
            MirostatModeV1::V2 => (self.mirostat_mu + f64::from(settings.eta) * error).max(0.0),
        };
        if !self.mirostat_mu.is_finite() {
            return Err(SamplingError::EmptyDistribution);
        }
        Ok(chosen)
    }
}

#[derive(Clone, Copy, Debug)]
struct ChainCandidate {
    token_id: u32,
    logit: f32,
    probability: f64,
    logprob: f64,
}

fn legacy_distribution(
    parameters: &SamplingParametersV1,
    token_counts: &BTreeMap<u32, u32>,
    logits: &[f32],
) -> Result<Vec<SamplingLogprobV1>, SamplingError> {
    let mut candidates = Vec::with_capacity(logits.len());
    for (index, &raw) in logits.iter().enumerate() {
        let token_id = u32::try_from(index).map_err(|_| SamplingError::TokenIdOverflow)?;
        if raw.is_nan() {
            return Err(SamplingError::NanLogit { token_id });
        }
        let count = token_counts.get(&token_id).copied().unwrap_or(0);
        let penalized = raw
            - count as f32 * parameters.frequency_penalty()
            - if count > 0 {
                parameters.presence_penalty()
            } else {
                0.0
            };
        let scaled = penalized / parameters.temperature();
        if scaled != f32::NEG_INFINITY {
            candidates.push(ChainCandidate {
                token_id,
                logit: scaled,
                probability: 0.0,
                logprob: 0.0,
            });
        }
    }
    let mut candidates = softmax_candidates(candidates)?;
    if candidates
        .first()
        .is_some_and(|candidate| candidate.logit == f32::INFINITY)
    {
        return Ok(candidates
            .into_iter()
            .map(|candidate| SamplingLogprobV1 {
                token_id: candidate.token_id,
                logprob: candidate.logprob,
            })
            .collect());
    }
    cumulative_retain(&mut candidates, f64::from(parameters.top_p()));
    renormalize(&mut candidates)?;
    Ok(candidates
        .into_iter()
        .map(|candidate| SamplingLogprobV1 {
            token_id: candidate.token_id,
            logprob: candidate.logprob,
        })
        .collect())
}

fn softmax_candidates(
    mut candidates: Vec<ChainCandidate>,
) -> Result<Vec<ChainCandidate>, SamplingError> {
    if candidates.is_empty() {
        return Err(SamplingError::EmptyDistribution);
    }
    candidates.sort_by(|left, right| {
        right
            .logit
            .total_cmp(&left.logit)
            .then_with(|| left.token_id.cmp(&right.token_id))
    });
    if candidates[0].logit == f32::INFINITY {
        let count = candidates
            .iter()
            .take_while(|candidate| candidate.logit == f32::INFINITY)
            .count();
        candidates.truncate(count);
        let probability = 1.0 / count as f64;
        for candidate in &mut candidates {
            candidate.probability = probability;
            candidate.logprob = probability.ln();
        }
        return Ok(candidates);
    }
    let max = f64::from(candidates[0].logit);
    let mut total = 0.0;
    for candidate in &mut candidates {
        let weight = (f64::from(candidate.logit) - max).exp();
        if !weight.is_finite() {
            return Err(SamplingError::EmptyDistribution);
        }
        candidate.probability = weight;
        total += weight;
    }
    candidates.retain(|candidate| candidate.probability > 0.0);
    if !total.is_finite() || total <= 0.0 {
        return Err(SamplingError::EmptyDistribution);
    }
    for candidate in &mut candidates {
        candidate.probability /= total;
        candidate.logprob = candidate.probability.ln();
    }
    Ok(candidates)
}

fn apply_filters(
    candidates: &mut Vec<ChainCandidate>,
    config: &SamplerChainConfigV1,
) -> Result<(), SamplingError> {
    if let Some(k) = config.top_k {
        if k > 0 {
            candidates.truncate(k.min(candidates.len()));
            renormalize(candidates)?;
        }
    }
    if config.min_p > 0.0 {
        let maximum = candidates[0].probability;
        candidates.retain(|candidate| candidate.probability >= maximum * f64::from(config.min_p));
        renormalize(candidates)?;
    }
    if config.typical_p < 1.0 {
        let entropy = candidates
            .iter()
            .map(|candidate| -candidate.probability * candidate.logprob)
            .sum::<f64>();
        candidates.sort_by(|left, right| {
            (-(left.logprob) - entropy)
                .abs()
                .total_cmp(&(-right.logprob - entropy).abs())
                .then_with(|| left.token_id.cmp(&right.token_id))
        });
        cumulative_retain(candidates, f64::from(config.typical_p));
        renormalize(candidates)?;
    }
    candidates.sort_by(|left, right| {
        right
            .probability
            .total_cmp(&left.probability)
            .then_with(|| left.token_id.cmp(&right.token_id))
    });
    let top_p = f64::from(config.parameters.top_p());
    cumulative_retain(candidates, top_p);
    renormalize(candidates)?;
    candidates.sort_by(|left, right| {
        right
            .logit
            .total_cmp(&left.logit)
            .then_with(|| left.token_id.cmp(&right.token_id))
    });
    Ok(())
}

fn cumulative_retain(candidates: &mut Vec<ChainCandidate>, probability: f64) {
    let mut cumulative = 0.0;
    let mut keep = 1_usize.min(candidates.len());
    for (index, candidate) in candidates.iter().enumerate() {
        cumulative += candidate.probability;
        keep = index + 1;
        if cumulative >= probability {
            break;
        }
    }
    candidates.truncate(keep);
}

fn renormalize(candidates: &mut [ChainCandidate]) -> Result<(), SamplingError> {
    if candidates.is_empty() {
        return Err(SamplingError::EmptyDistribution);
    }
    let total = candidates
        .iter()
        .map(|candidate| candidate.probability)
        .sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(SamplingError::EmptyDistribution);
    }
    for candidate in candidates {
        candidate.probability /= total;
        candidate.logprob = candidate.probability.ln();
    }
    Ok(())
}

fn apply_xtc(
    candidates: &mut Vec<ChainCandidate>,
    settings: XtcSamplingConfigV1,
    random: &mut impl SamplingRandomSource,
) -> Result<(), SamplingError> {
    if settings.probability == 0.0 || candidates.len() <= settings.min_keep {
        return Ok(());
    }
    let threshold = f64::from(settings.threshold);
    let mut retained = Vec::with_capacity(candidates.len());
    for candidate in candidates.iter().copied() {
        let draw = random.next_unit_f64()?;
        if !(0.0..1.0).contains(&draw) {
            return Err(SamplingError::InvalidRandomValue);
        }
        if candidate.probability >= threshold
            || draw >= f64::from(settings.probability)
            || retained.len() < settings.min_keep
        {
            retained.push(candidate);
        }
    }
    if retained.is_empty() {
        retained.push(candidates[0]);
    }
    *candidates = retained;
    renormalize(candidates)
}

fn sample_candidates(
    candidates: &[ChainCandidate],
    random: &mut impl SamplingRandomSource,
) -> Result<u32, SamplingError> {
    let sample = random.next_unit_f64()?;
    if !(0.0..1.0).contains(&sample) {
        return Err(SamplingError::InvalidRandomValue);
    }
    let threshold = sample;
    let mut cumulative = 0.0;
    for candidate in candidates {
        cumulative += candidate.probability;
        if threshold < cumulative {
            return Ok(candidate.token_id);
        }
    }
    candidates
        .last()
        .map_or(Err(SamplingError::EmptyDistribution), |candidate| {
            Ok(candidate.token_id)
        })
}

fn dry_repetition(history: &[u32], token_id: u32, config: &DrySamplingConfigV1) -> usize {
    if config.allowed_length == 0 || config.penalty_last_n == 0 || history.is_empty() {
        return 0;
    }
    let history = if history.len() > config.penalty_last_n {
        &history[history.len() - config.penalty_last_n..]
    } else {
        history
    };
    if config
        .sequence_breakers
        .iter()
        .any(|breaker| breaker.len() == 1 && breaker[0] == token_id)
    {
        return 0;
    }
    let max_length = history.len().min(config.allowed_length);
    // Compare the current suffix with an earlier bounded occurrence and
    // penalize a token that would continue that earlier sequence.
    for length in (1..=max_length).rev() {
        let suffix_start = history.len() - length;
        let suffix = &history[suffix_start..];
        if contains_sequence_breaker(suffix, &config.sequence_breakers) {
            continue;
        }
        let search_end = suffix_start.saturating_sub(length);
        for start in 0..=search_end {
            let end = start + length;
            if history[start..end] == *suffix
                && end < history.len()
                && history[end] == token_id
                && !candidate_completes_sequence_breaker(
                    &history[..end],
                    token_id,
                    &config.sequence_breakers,
                )
            {
                return length;
            }
        }
    }
    0
}

fn contains_sequence_breaker(tokens: &[u32], breakers: &[Vec<u32>]) -> bool {
    breakers.iter().any(|breaker| {
        breaker.len() <= tokens.len()
            && tokens
                .windows(breaker.len())
                .any(|window| window == breaker.as_slice())
    })
}

fn candidate_completes_sequence_breaker(
    prefix: &[u32],
    token_id: u32,
    breakers: &[Vec<u32>],
) -> bool {
    breakers.iter().any(|breaker| {
        let Some((&last, head)) = breaker.split_last() else {
            return false;
        };
        last == token_id && head.len() <= prefix.len() && prefix.ends_with(head)
    })
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    token_id: u32,
    scaled_logit: f32,
    probability: f64,
}

fn select_uniform(
    token_ids: &[u32],
    random: &mut impl SamplingRandomSource,
) -> Result<u32, SamplingError> {
    let sample = random.next_unit_f64()?;
    if !(0.0..1.0).contains(&sample) {
        return Err(SamplingError::InvalidRandomValue);
    }
    let index = ((sample * token_ids.len() as f64) as usize).min(token_ids.len() - 1);
    Ok(token_ids[index])
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixed(f64);

    impl SamplingRandomSource for Fixed {
        fn next_unit_f64(&mut self) -> Result<f64, SamplingError> {
            Ok(self.0)
        }
    }

    struct SequenceRandom {
        values: Vec<f64>,
        position: usize,
    }

    impl SequenceRandom {
        fn new(values: &[f64]) -> Self {
            Self {
                values: values.to_vec(),
                position: 0,
            }
        }
    }

    impl SamplingRandomSource for SequenceRandom {
        fn next_unit_f64(&mut self) -> Result<f64, SamplingError> {
            let value = self
                .values
                .get(self.position)
                .copied()
                .ok_or(SamplingError::InvalidRandomValue)?;
            self.position += 1;
            Ok(value)
        }
    }

    fn params(temp: f32, top_p: f32, presence: f32, frequency: f32) -> SamplingParametersV1 {
        SamplingParametersV1::new(temp, top_p, presence, frequency).unwrap()
    }

    #[test]
    fn greedy_keeps_device_argmax_and_needs_no_logits_or_rng() {
        let sampler = ProfileSamplerV1::new(SamplingParametersV1::greedy(), &[]).unwrap();
        assert_eq!(sampler.select(17, None, &mut Fixed(f64::NAN)).unwrap(), 17);
    }

    #[test]
    fn explicit_seed_replays_the_same_sampling_stream() {
        let parameters = params(0.7, 0.9, 0.0, 0.0);
        let mut first =
            OsSamplingRandom::for_parameters_and_seed(parameters, Some(u64::MAX)).unwrap();
        let mut replay =
            OsSamplingRandom::for_parameters_and_seed(parameters, Some(u64::MAX)).unwrap();
        for _ in 0..17 {
            assert_eq!(
                first.next_unit_f64().unwrap(),
                replay.next_unit_f64().unwrap()
            );
        }
    }

    #[test]
    fn explicit_seed_first_draw_is_splitmix_counter_zero() {
        // The HIP token selector uses this same state transition.  Keep the
        // first draw explicit so a device-side seed^counter shortcut cannot
        // silently diverge from the CPU reference stream.
        let parameters = params(0.7, 0.9, 0.0, 0.0);
        let seed = 7_u64;
        let mut random = OsSamplingRandom::for_parameters_and_seed(parameters, Some(seed)).unwrap();
        let actual = random.next_unit_f64().unwrap();
        let gamma = 0x9e37_79b9_7f4a_7c15_u64;
        let mut value = seed.wrapping_add(gamma);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        let expected = ((value ^ (value >> 31)) >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64));
        assert_eq!(actual, expected);
    }

    #[test]
    fn non_aligned_vocab_temperature_and_top_p_boundaries_are_exact() {
        let logits = [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0];
        let top_one = ProfileSamplerV1::new(params(1.0, 0.0, 0.0, 0.0), &[]).unwrap();
        assert_eq!(
            top_one.select(3, Some(&logits), &mut Fixed(0.999)).unwrap(),
            3
        );
        let full = ProfileSamplerV1::new(params(2.0, 1.0, 0.0, 0.0), &[]).unwrap();
        assert!(full.select(3, Some(&logits), &mut Fixed(0.5)).unwrap() < 7);
    }

    #[test]
    fn presence_and_frequency_penalties_change_selection() {
        let sampler = ProfileSamplerV1::new(params(1.0, 0.0, 1.0, 1.0), &[1, 1]).unwrap();
        assert_eq!(
            sampler
                .select(1, Some(&[0.0, 2.5, 1.0]), &mut Fixed(0.0))
                .unwrap(),
            2
        );
    }

    #[test]
    fn nan_and_empty_mass_fail_closed_while_positive_inf_is_explicit() {
        let sampler = ProfileSamplerV1::new(params(1.0, 1.0, 0.0, 0.0), &[]).unwrap();
        assert_eq!(
            sampler.select(0, Some(&[f32::NAN]), &mut Fixed(0.0)),
            Err(SamplingError::NanLogit { token_id: 0 })
        );
        assert_eq!(
            sampler.select(0, Some(&[f32::NEG_INFINITY]), &mut Fixed(0.0)),
            Err(SamplingError::EmptyDistribution)
        );
        assert_eq!(
            sampler
                .select(
                    0,
                    Some(&[f32::INFINITY, 0.0, f32::INFINITY]),
                    &mut Fixed(0.75)
                )
                .unwrap(),
            2
        );
    }

    #[test]
    fn ties_choose_lowest_token_at_top_p_zero() {
        let sampler = ProfileSamplerV1::new(params(1.0, 0.0, 0.0, 0.0), &[]).unwrap();
        assert_eq!(
            sampler
                .select(0, Some(&[2.0, 2.0, 1.0]), &mut Fixed(0.9))
                .unwrap(),
            0
        );
    }

    #[test]
    fn parameter_ranges_reject_nonfinite_and_both_sides() {
        for invalid in [-0.1, 2.1, f32::NAN, f32::INFINITY] {
            assert_eq!(
                SamplingParametersV1::new(invalid, 1.0, 0.0, 0.0),
                Err(SamplingError::InvalidTemperature)
            );
        }
        assert!(SamplingParametersV1::new(0.0, 0.0, -2.0, 2.0).is_ok());
        assert!(SamplingParametersV1::new(2.0, 1.0, 2.0, -2.0).is_ok());
    }

    #[test]
    fn legacy_chain_is_an_exact_additive_adapter() {
        let parameters = params(0.7, 0.9, 0.2, -0.1);
        let mut chain =
            SamplerChainV1::new(SamplerChainConfigV1::legacy(parameters), &[1, 1]).unwrap();
        let expected = ProfileSamplerV1::new(parameters, &[1, 1])
            .unwrap()
            .select(0, Some(&[0.2, 1.3, 0.8, -0.4]), &mut Fixed(0.123))
            .unwrap();
        let actual = chain
            .select(0, Some(&[0.2, 1.3, 0.8, -0.4]), &mut Fixed(0.123))
            .unwrap();
        assert_eq!(actual.token_id, expected);
    }

    #[test]
    fn chain_filters_have_stable_tie_and_boundary_behavior() {
        let parameters = params(1.0, 1.0, 0.0, 0.0);
        let config = SamplerChainConfigV1::new(parameters)
            .with_top_k(1)
            .unwrap()
            .with_top_logprobs(1)
            .unwrap();
        let mut chain = SamplerChainV1::new(config, &[]).unwrap();
        let selected = chain
            .select(0, Some(&[2.0, 2.0, 1.0]), &mut Fixed(0.999))
            .unwrap();
        assert_eq!(selected.token_id, 0);
        assert_eq!(selected.top_logprobs.len(), 1);
        assert_eq!(selected.top_logprobs[0].token_id, 0);
        assert!((selected.logprob - 0.0).abs() < 1e-12);
    }

    #[test]
    fn greedy_logprobs_request_uses_logits_without_rng() {
        let config =
            SamplerChainConfigV1::new(SamplingParametersV1::greedy()).with_return_logprobs(true);
        let mut chain = SamplerChainV1::new(config, &[]).unwrap();
        let selected = chain
            .select(0, Some(&[1.0, 2.0]), &mut Fixed(f64::NAN))
            .unwrap();
        assert_eq!(selected.token_id, 1);
        assert!(selected.logprob.is_finite());
    }

    #[test]
    fn chain_masks_eos_and_reports_all_masked_without_fallback() {
        let config = SamplerChainConfigV1::new(params(1.0, 1.0, 0.0, 0.0)).with_ignore_eos(7);
        let mut chain = SamplerChainV1::new(config, &[]).unwrap();
        assert_eq!(
            chain.select(7, Some(&[f32::NEG_INFINITY; 8]), &mut Fixed(0.0)),
            Err(SamplingError::EmptyDistribution)
        );
        let mut finite = SamplerChainV1::new(
            SamplerChainConfigV1::new(params(1.0, 1.0, 0.0, 0.0)).with_ignore_eos(1),
            &[],
        )
        .unwrap();
        assert_eq!(
            finite
                .select(1, Some(&[0.0, 10.0, 0.0]), &mut Fixed(0.0))
                .unwrap()
                .token_id,
            0
        );
    }

    #[test]
    fn chain_handles_infinity_explicitly_and_rejects_nan() {
        let config = SamplerChainConfigV1::new(params(1.0, 1.0, 0.0, 0.0))
            .with_top_logprobs(2)
            .unwrap();
        let mut chain = SamplerChainV1::new(config, &[]).unwrap();
        let selected = chain
            .select(
                0,
                Some(&[f32::INFINITY, 0.0, f32::INFINITY]),
                &mut Fixed(0.75),
            )
            .unwrap();
        assert_eq!(selected.token_id, 2);
        assert_eq!(selected.top_logprobs.len(), 2);
        assert_eq!(
            chain.select(0, Some(&[f32::NAN]), &mut Fixed(0.0)),
            Err(SamplingError::NanLogit { token_id: 0 })
        );
        assert_eq!(
            chain.select(0, Some(&[f32::NEG_INFINITY]), &mut Fixed(0.0)),
            Err(SamplingError::EmptyDistribution)
        );
    }

    #[test]
    fn chain_mirostat_and_seed_are_bounded_and_reproducible() {
        let mirostat = MirostatSamplingConfigV1::new(MirostatModeV1::V2, 2.0, 0.2, 4.0).unwrap();
        let config = SamplerChainConfigV1::new(params(1.0, 1.0, 0.0, 0.0))
            .with_mirostat(mirostat)
            .unwrap();
        let mut first = SamplerChainV1::new(config.clone(), &[]).unwrap();
        let mut replay = SamplerChainV1::new(config, &[]).unwrap();
        let parameters = params(1.0, 1.0, 0.0, 0.0);
        let mut random_a = OsSamplingRandom::for_parameters_and_seed(parameters, Some(9)).unwrap();
        let mut random_b = OsSamplingRandom::for_parameters_and_seed(parameters, Some(9)).unwrap();
        for _ in 0..8 {
            assert_eq!(
                first
                    .select(0, Some(&[0.1, 0.7, 1.2, -0.2]), &mut random_a)
                    .unwrap()
                    .token_id,
                replay
                    .select(0, Some(&[0.1, 0.7, 1.2, -0.2]), &mut random_b)
                    .unwrap()
                    .token_id
            );
        }
        assert!(
            SamplerChainConfigV1::new(parameters)
                .with_top_k(1)
                .unwrap()
                .with_mirostat(mirostat)
                .is_err()
        );
    }

    #[test]
    fn chain_bias_and_extended_validation_are_fail_closed() {
        let parameters = params(1.0, 1.0, 0.0, 0.0);
        assert!(
            SamplerChainConfigV1::new(parameters)
                .with_logit_bias(vec![
                    LogitBiasV1 {
                        token_id: 2,
                        bias: 1.0,
                    },
                    LogitBiasV1 {
                        token_id: 2,
                        bias: 2.0,
                    },
                ])
                .is_err()
        );
        assert!(
            SamplerChainConfigV1::new(parameters)
                .with_min_p(f32::NAN)
                .is_err()
        );
        assert!(
            SamplerChainConfigV1::new(parameters)
                .with_typical_p(0.0)
                .is_err()
        );
        assert!(
            SamplerChainConfigV1::new(parameters)
                .with_top_logprobs(21)
                .is_err()
        );
        let config = SamplerChainConfigV1::new(parameters)
            .with_logit_bias(vec![LogitBiasV1 {
                token_id: 99,
                bias: 1.0,
            }])
            .unwrap();
        let mut chain = SamplerChainV1::new(config, &[]).unwrap();
        assert_eq!(
            chain.select(0, Some(&[0.0, 1.0]), &mut Fixed(0.0)),
            Err(SamplingError::TokenIdOutOfRange { token_id: 99 })
        );
    }

    #[test]
    fn dry_penalizes_a_repeated_multi_token_suffix() {
        let dry = DrySamplingConfigV1::new(1.0, 2.0, 4, vec![]).unwrap();
        let config = SamplerChainConfigV1::new(params(1.0, 1.0, 0.0, 0.0))
            .with_dry(dry)
            .unwrap();
        let mut chain = SamplerChainV1::new(config, &[1, 2, 1, 2]).unwrap();
        // Token 1 would continue the earlier [1,2] sequence and receives a
        // three-logit penalty; token 2 remains the deterministic winner.
        let selected = chain
            .select(0, Some(&[0.0, 3.0, 2.0]), &mut Fixed(0.0))
            .unwrap();
        assert_eq!(selected.token_id, 2);
    }

    #[test]
    fn dry_multi_token_breaker_and_history_window_are_independent() {
        let dry = DrySamplingConfigV1::new(10.0, 2.0, 8, vec![vec![9, 1]])
            .unwrap()
            .with_penalty_last_n(4)
            .unwrap();
        let config = SamplerChainConfigV1::new(params(1.0, 1.0, 0.0, 0.0))
            .with_dry(dry)
            .unwrap();
        // The candidate 1 completes breaker [9,1], so DRY must not penalize
        // it even though the recent suffix otherwise repeats.
        let mut chain = SamplerChainV1::new(config, &[9, 1, 9]).unwrap();
        let selected = chain
            .select(0, Some(&[0.0, 3.0, 2.0]), &mut Fixed(0.0))
            .unwrap();
        assert_eq!(selected.token_id, 1);
    }

    #[test]
    fn xtc_consumption_order_is_bounded_and_deterministic() {
        let xtc = XtcSamplingConfigV1::new(1.0, 1.0, 1).unwrap();
        let config = SamplerChainConfigV1::new(params(1.0, 1.0, 0.0, 0.0))
            .with_xtc(xtc)
            .unwrap();
        let mut chain = SamplerChainV1::new(config, &[]).unwrap();
        let mut random = SequenceRandom::new(&[0.1, 0.2, 0.9]);
        let selected = chain.select(0, Some(&[2.0, 1.0]), &mut random).unwrap();
        assert_eq!(selected.token_id, 0);
        // One draw per candidate for XTC, followed by the terminal draw. A
        // future dedicated node stream must preserve this documented order.
        assert_eq!(random.position, 3);

        let disabled = XtcSamplingConfigV1::new(0.0, 1.0, 1).unwrap();
        let config = SamplerChainConfigV1::new(params(0.0, 1.0, 0.0, 0.0))
            .with_xtc(disabled)
            .unwrap();
        assert!(!config.requires_logits());
        assert!(!config.requires_randomness());
        let mut chain = SamplerChainV1::new(config, &[]).unwrap();
        let mut random = SequenceRandom::new(&[]);
        assert_eq!(chain.select(7, None, &mut random).unwrap().token_id, 7);
        assert_eq!(random.position, 0);
    }

    #[test]
    fn long_prompt_counts_are_kept_while_repeat_history_is_bounded() {
        let prior = vec![6_u32; MAX_SAMPLING_HISTORY + 1];
        let config = SamplerChainConfigV1::new(params(1.0, 1.0, 0.0, 1.0));
        let mut chain = SamplerChainV1::new(config, &prior).unwrap();
        assert_eq!(
            chain
                .select(
                    0,
                    Some(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 1.0]),
                    &mut Fixed(0.0)
                )
                .unwrap()
                .token_id,
            7
        );
    }

    #[test]
    fn device_selector_subset_materializes_additive_and_mask_without_logits() {
        let parameters = params(1.0, 1.0, 0.5, 0.25);
        let config = SamplerChainConfigV1::new(parameters)
            .with_logit_bias(vec![LogitBiasV1 {
                token_id: 2,
                bias: 3.0,
            }])
            .unwrap();
        let chain = SamplerChainV1::new(config, &[1, 1]).unwrap();
        assert!(chain.supports_device_selector());
        let request = chain
            .prepare_device_selector(3, Some(&[true, false, true]), 7, 9)
            .unwrap();
        assert_eq!(request.valid_mask(), &[1, 0, 1]);
        assert_eq!(request.additive_logits(), &[0.0, -1.0, 3.0]);
        assert_eq!(request.temperature(), 1.0);
        assert_eq!(request.seed(), 7);
        assert_eq!(request.counter(), 9);
        assert!(!request.return_logprob());

        let filtered = SamplerChainConfigV1::new(parameters).with_top_k(2).unwrap();
        let chain = SamplerChainV1::new(filtered, &[]).unwrap();
        assert!(!chain.supports_device_selector());
        assert_eq!(
            chain.prepare_device_selector(3, None, 0, 0),
            Err(SamplingError::UnsupportedDeviceSelector)
        );
    }

    #[test]
    fn device_selector_categorical_order_matches_legacy_logit_order() {
        let parameters = params(1.0, 1.0, 0.0, 0.0);
        let mut chain = SamplerChainV1::new(SamplerChainConfigV1::legacy(parameters), &[]).unwrap();
        let mut random = OsSamplingRandom::for_parameters_and_seed(parameters, Some(0)).unwrap();

        // The device selector contract accumulates probability in descending
        // effective-logit order and breaks equal-logit ties by token ID. Keep
        // this cross-layer fixture aligned with the native selector tests.
        assert_eq!(
            chain
                .select(2, Some(&[0.0, 1.0, 2.0]), &mut random)
                .unwrap()
                .token_id,
            1
        );
        assert!(chain.prepare_device_selector(3, None, 0, 0).is_ok());
    }
}
