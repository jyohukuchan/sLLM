// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Deterministic CPU sampling for the single-request SQ8 serving path.

use crate::sq8_generation_runtime::greedy_top1_finite;
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};

const F64_UNIT_SCALE: f64 = 1.0 / 9_007_199_254_740_992.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sq8SampledToken {
    pub token_id: usize,
    pub logit: f32,
}

#[derive(Debug, Clone)]
pub struct Sq8SamplingProposal {
    sampled: Sq8SampledToken,
    seed_bits: u64,
    expected_draws: u64,
    next_rng: Option<ChaCha8Rng>,
}

impl Sq8SamplingProposal {
    pub fn sampled(&self) -> Sq8SampledToken {
        self.sampled
    }

    pub fn consumes_rng(&self) -> bool {
        self.next_rng.is_some()
    }
}

/// Request-local sampler whose RNG advances only when a proposal is committed.
#[derive(Debug, Clone)]
pub struct Sq8CpuSampler {
    seed_bits: u64,
    draws: u64,
    rng: ChaCha8Rng,
}

impl Sq8CpuSampler {
    pub fn new(seed: i64) -> Self {
        let seed_bits = seed as u64;
        Self {
            seed_bits,
            draws: 0,
            rng: ChaCha8Rng::seed_from_u64(seed_bits),
        }
    }

    pub fn draws(&self) -> u64 {
        self.draws
    }

    /// Computes a token without mutating request RNG state.
    ///
    /// Dropping the proposal consumes no draw. This lets the serving session perform its final
    /// cancellation check before publishing a token and committing the corresponding RNG state.
    pub fn propose(
        &self,
        logits: &[f32],
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<Sq8SamplingProposal, String> {
        validate_sampling_inputs(logits, temperature, top_k, top_p)?;
        if temperature == 0.0 {
            let top1 = greedy_top1_finite(logits)?;
            return Ok(Sq8SamplingProposal {
                sampled: Sq8SampledToken {
                    token_id: top1.token_id,
                    logit: top1.logit,
                },
                seed_bits: self.seed_bits,
                expected_draws: self.draws,
                next_rng: None,
            });
        }

        let candidates = top_candidates(logits, temperature, top_k)?;
        let probabilities = nucleus_probabilities(&candidates, top_p)?;
        let mut next_rng = self.rng.clone();
        let uniform = uniform_f64(&mut next_rng);
        let selected_index = select_probability_index(&probabilities, uniform)?;
        let selected = candidates[selected_index];
        Ok(Sq8SamplingProposal {
            sampled: Sq8SampledToken {
                token_id: selected.token_id,
                logit: selected.logit,
            },
            seed_bits: self.seed_bits,
            expected_draws: self.draws,
            next_rng: Some(next_rng),
        })
    }

    pub fn commit(&mut self, proposal: Sq8SamplingProposal) -> Result<Sq8SampledToken, String> {
        if proposal.seed_bits != self.seed_bits || proposal.expected_draws != self.draws {
            return Err(format!(
                "SQ8 sampling proposal is stale or belongs to another RNG stream: proposal_seed={} sampler_seed={} proposal_draws={} sampler_draws={}",
                proposal.seed_bits, self.seed_bits, proposal.expected_draws, self.draws
            ));
        }
        if let Some(next_rng) = proposal.next_rng {
            self.draws = self
                .draws
                .checked_add(1)
                .ok_or_else(|| "SQ8 sampling draw counter overflows".to_string())?;
            self.rng = next_rng;
        }
        Ok(proposal.sampled)
    }
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    token_id: usize,
    logit: f32,
    scaled_logit: f64,
}

fn validate_sampling_inputs(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> Result<(), String> {
    if logits.is_empty() {
        return Err("SQ8 sampling logits must not be empty".into());
    }
    if logits.iter().any(|logit| !logit.is_finite()) {
        let (index, value) = logits
            .iter()
            .copied()
            .enumerate()
            .find(|(_, logit)| !logit.is_finite())
            .expect("non-finite logit was detected above");
        return Err(format!(
            "SQ8 sampling logits contain non-finite value {value} at index {index}"
        ));
    }
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        return Err(format!(
            "SQ8 sampling temperature must be finite and in 0..=2, got {temperature}"
        ));
    }
    if top_k == 0 || top_k > logits.len() {
        return Err(format!(
            "SQ8 sampling top_k must be in 1..={}, got {top_k}",
            logits.len()
        ));
    }
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err(format!(
            "SQ8 sampling top_p must be finite and in 0<top_p<=1, got {top_p}"
        ));
    }
    Ok(())
}

fn top_candidates(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
) -> Result<Vec<Candidate>, String> {
    debug_assert!(temperature > 0.0);
    let inverse_temperature = 1.0_f64 / f64::from(temperature);
    let mut top = Vec::with_capacity(top_k);
    for (token_id, logit) in logits.iter().copied().enumerate() {
        let candidate = Candidate {
            token_id,
            logit,
            scaled_logit: f64::from(logit) * inverse_temperature,
        };
        let insertion = top
            .iter()
            .position(|current| candidate_precedes(candidate, *current))
            .unwrap_or(top.len());
        if insertion < top_k {
            top.insert(insertion, candidate);
            if top.len() > top_k {
                top.pop();
            }
        } else if top.len() < top_k {
            top.push(candidate);
        }
    }
    if top.len() != top_k
        || top
            .iter()
            .any(|candidate| !candidate.scaled_logit.is_finite())
    {
        return Err("SQ8 sampling failed to construct finite top-k candidates".into());
    }
    Ok(top)
}

fn candidate_precedes(lhs: Candidate, rhs: Candidate) -> bool {
    lhs.scaled_logit > rhs.scaled_logit
        || (lhs.scaled_logit == rhs.scaled_logit && lhs.token_id < rhs.token_id)
}

fn nucleus_probabilities(candidates: &[Candidate], top_p: f32) -> Result<Vec<f64>, String> {
    let maximum = candidates
        .first()
        .ok_or_else(|| "SQ8 sampling top-k candidate set is empty".to_string())?
        .scaled_logit;
    let mut weights = Vec::with_capacity(candidates.len());
    let mut total = 0.0_f64;
    for candidate in candidates {
        let weight = (candidate.scaled_logit - maximum).exp();
        if !weight.is_finite() || weight < 0.0 {
            return Err("SQ8 sampling softmax produced an invalid weight".into());
        }
        total += weight;
        weights.push(weight);
    }
    if !total.is_finite() || total <= 0.0 {
        return Err("SQ8 sampling softmax has invalid probability mass".into());
    }

    let threshold = f64::from(top_p);
    let mut cumulative = 0.0_f64;
    let mut keep = 0_usize;
    for weight in &weights {
        cumulative += *weight / total;
        keep += 1;
        if cumulative >= threshold {
            break;
        }
    }
    keep = keep.max(1).min(weights.len());
    weights.truncate(keep);
    let kept_total = weights.iter().sum::<f64>();
    if !kept_total.is_finite() || kept_total <= 0.0 {
        return Err("SQ8 sampling nucleus has invalid probability mass".into());
    }
    for probability in &mut weights {
        *probability /= kept_total;
    }
    let normalized_total = weights.iter().sum::<f64>();
    if !normalized_total.is_finite() || normalized_total <= 0.0 {
        return Err("SQ8 sampling nucleus normalization failed".into());
    }
    Ok(weights)
}

fn uniform_f64(rng: &mut ChaCha8Rng) -> f64 {
    ((rng.next_u64() >> 11) as f64) * F64_UNIT_SCALE
}

fn select_probability_index(probabilities: &[f64], uniform: f64) -> Result<usize, String> {
    if probabilities.is_empty() || !uniform.is_finite() || !(0.0..1.0).contains(&uniform) {
        return Err("SQ8 sampling probability draw is invalid".into());
    }
    let mut cumulative = 0.0_f64;
    for (index, probability) in probabilities.iter().copied().enumerate() {
        if !probability.is_finite() || probability < 0.0 {
            return Err("SQ8 sampling probability is invalid".into());
        }
        cumulative += probability;
        if uniform < cumulative {
            return Ok(index);
        }
    }
    if cumulative.is_finite() && cumulative > 0.0 {
        return Ok(probabilities.len() - 1);
    }
    Err("SQ8 sampling cumulative probability is invalid".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn commit_one(
        sampler: &mut Sq8CpuSampler,
        logits: &[f32],
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Sq8SampledToken {
        let proposal = sampler.propose(logits, temperature, top_k, top_p).unwrap();
        sampler.commit(proposal).unwrap()
    }

    #[test]
    fn temperature_zero_matches_full_vocabulary_greedy_and_consumes_no_draw() {
        let mut sampler = Sq8CpuSampler::new(99);
        let sampled = commit_one(&mut sampler, &[-3.0, 4.5, 4.5, 2.0], 0.0, 1, 0.01);
        assert_eq!(sampled.token_id, 1);
        assert_eq!(sampled.logit, 4.5);
        assert_eq!(sampler.draws(), 0);
    }

    #[test]
    fn proposal_is_transactional_and_stale_commit_is_rejected() {
        let logits = [1.0, 0.5, -1.0];
        let mut sampler = Sq8CpuSampler::new(7);
        let dropped = sampler.propose(&logits, 1.0, 3, 1.0).unwrap();
        assert!(dropped.consumes_rng());
        drop(dropped);
        assert_eq!(sampler.draws(), 0);

        let proposal = sampler.propose(&logits, 1.0, 3, 1.0).unwrap();
        let stale = proposal.clone();
        sampler.commit(proposal).unwrap();
        assert_eq!(sampler.draws(), 1);
        assert!(sampler.commit(stale).is_err());
    }

    #[test]
    fn same_seed_and_parameters_produce_the_same_sequence() {
        let logits = [2.0, 1.5, 1.0, 0.5, 0.0];
        let mut first = Sq8CpuSampler::new(-17);
        let mut second = Sq8CpuSampler::new(-17);
        let first_tokens = (0..32)
            .map(|_| commit_one(&mut first, &logits, 0.6, 5, 0.95).token_id)
            .collect::<Vec<_>>();
        let second_tokens = (0..32)
            .map(|_| commit_one(&mut second, &logits, 0.6, 5, 0.95).token_id)
            .collect::<Vec<_>>();
        assert_eq!(first_tokens, second_tokens);
        assert_eq!(
            first_tokens,
            vec![
                3, 0, 1, 2, 2, 0, 1, 0, 0, 2, 3, 0, 1, 0, 1, 2, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1,
                1, 2, 2, 1,
            ]
        );
        assert_eq!(first.draws(), 32);
        assert_eq!(second.draws(), 32);
    }

    #[test]
    fn different_seeds_reach_multiple_valid_tokens() {
        let logits = [0.0; 4];
        let observed = (0..64)
            .map(|seed| {
                let mut sampler = Sq8CpuSampler::new(seed);
                commit_one(&mut sampler, &logits, 1.0, 4, 1.0).token_id
            })
            .collect::<BTreeSet<_>>();
        assert!(observed.len() > 1, "observed={observed:?}");
        assert!(observed.iter().all(|token_id| *token_id < logits.len()));
    }

    #[test]
    fn top_k_excludes_lower_ranked_tokens_and_ties_use_token_order() {
        let logits = [5.0, 5.0, 4.0, 100.0];
        let candidates = top_candidates(&logits, 1.0, 3).unwrap();
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.token_id)
                .collect::<Vec<_>>(),
            vec![3, 0, 1]
        );
        for seed in 0..64 {
            let mut sampler = Sq8CpuSampler::new(seed);
            let token = commit_one(&mut sampler, &logits, 1.0, 2, 1.0).token_id;
            assert!(matches!(token, 0 | 3));
        }
    }

    #[test]
    fn top_p_keeps_the_shortest_probability_prefix_and_at_least_one_token() {
        let candidates = top_candidates(&[8.0, 1.0, 0.0], 1.0, 3).unwrap();
        assert_eq!(nucleus_probabilities(&candidates, 0.5).unwrap().len(), 1);
        assert_eq!(
            nucleus_probabilities(&candidates, f32::MIN_POSITIVE)
                .unwrap()
                .len(),
            1
        );
        assert_eq!(nucleus_probabilities(&candidates, 1.0).unwrap().len(), 3);
    }

    #[test]
    fn nonfinite_and_invalid_probability_inputs_are_rejected() {
        for logits in [
            &[][..],
            &[f32::NAN][..],
            &[f32::INFINITY][..],
            &[f32::NEG_INFINITY][..],
        ] {
            assert!(Sq8CpuSampler::new(0).propose(logits, 1.0, 1, 1.0).is_err());
        }
        let logits = [0.0, 1.0];
        for temperature in [f32::NAN, -0.1, f32::INFINITY, 2.1] {
            assert!(
                Sq8CpuSampler::new(0)
                    .propose(&logits, temperature, 2, 1.0)
                    .is_err()
            );
        }
        for top_p in [f32::NAN, 0.0, -0.1, 1.1, f32::INFINITY] {
            assert!(
                Sq8CpuSampler::new(0)
                    .propose(&logits, 1.0, 2, top_p)
                    .is_err()
            );
        }
        for top_k in [0, 3] {
            assert!(
                Sq8CpuSampler::new(0)
                    .propose(&logits, 1.0, top_k, 1.0)
                    .is_err()
            );
        }
    }

    #[test]
    fn signed_seed_uses_twos_complement_u64_bits() {
        let signed = Sq8CpuSampler::new(-1);
        let direct = Sq8CpuSampler {
            seed_bits: u64::MAX,
            draws: 0,
            rng: ChaCha8Rng::seed_from_u64(u64::MAX),
        };
        let logits = [1.0, 0.0];
        assert_eq!(
            signed.propose(&logits, 1.0, 2, 1.0).unwrap().sampled(),
            direct.propose(&logits, 1.0, 2, 1.0).unwrap().sampled()
        );
    }
}
