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
        }
    }
}

impl std::error::Error for SamplingError {}

/// Internal randomness seam. HTTP request schemas never expose a seed.
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
        let mut seed = [0_u8; 8];
        File::open("/dev/urandom")
            .and_then(|mut file| file.read_exact(&mut seed))
            .map_err(|_| SamplingError::RandomSourceUnavailable)?;
        let state = u64::from_le_bytes(seed);
        Ok(Self { state })
    }

    /// Avoids touching the OS random source for the strict greedy path, where
    /// [`ProfileSamplerV1`] guarantees that randomness is never observed.
    pub fn for_parameters(parameters: SamplingParametersV1) -> Result<Self, SamplingError> {
        if parameters.requires_logits() {
            Self::new()
        } else {
            Ok(Self { state: 0 })
        }
    }

    fn next_u64(&mut self) -> u64 {
        // SplitMix64 is used only after the state is seeded by the OS. It is
        // adequate for categorical sampling and keeps the RNG seam tiny.
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

    fn params(temp: f32, top_p: f32, presence: f32, frequency: f32) -> SamplingParametersV1 {
        SamplingParametersV1::new(temp, top_p, presence, frequency).unwrap()
    }

    #[test]
    fn greedy_keeps_device_argmax_and_needs_no_logits_or_rng() {
        let sampler = ProfileSamplerV1::new(SamplingParametersV1::greedy(), &[]).unwrap();
        assert_eq!(sampler.select(17, None, &mut Fixed(f64::NAN)).unwrap(), 17);
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
}
