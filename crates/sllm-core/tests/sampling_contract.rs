// Test portions adapted from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-profile-v1-sampling-tests-001
// Upstream: https://github.com/ggml-org/llama.cpp @ f5919bf458ef190468b5c329bb293f8a54a1e69c, tests/test-sampling.cpp
// SPDX-License-Identifier: MIT

use std::fs;
use std::path::PathBuf;

use serde::Deserialize;
use sllm_core::{
    LogitBiasV1, ProfileSamplerV1, SamplerChainConfigV1, SamplerChainV1, SamplingError,
    SamplingParametersV1, SamplingRandomSource,
};

#[derive(Deserialize)]
struct Fixture {
    schema_version: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    logits: Vec<f32>,
    temperature: f32,
    top_p: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    history: Vec<u32>,
    uniform: f64,
    expected_token: u32,
}

struct FixedRandom(f64);

impl SamplingRandomSource for FixedRandom {
    fn next_unit_f64(&mut self) -> Result<f64, SamplingError> {
        Ok(self.0)
    }
}

#[test]
fn rust_sampler_matches_independent_numpy_fixture() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/profile_sampling_cases.json");
    let fixture: Fixture = serde_json::from_slice(&fs::read(path).expect("sampling fixture"))
        .expect("valid sampling fixture");
    assert_eq!(fixture.schema_version, "profile-sampling-cases-v1");
    for case in fixture.cases {
        let parameters = SamplingParametersV1::new(
            case.temperature,
            case.top_p,
            case.presence_penalty,
            case.frequency_penalty,
        )
        .expect("valid fixture parameters");
        let sampler = ProfileSamplerV1::new(parameters, &case.history).expect("valid history");
        let actual = sampler
            .select(0, Some(&case.logits), &mut FixedRandom(case.uniform))
            .expect("fixture distribution samples");
        assert_eq!(actual, case.expected_token, "{}", case.id);
    }
}

#[test]
fn sampler_chain_contract_keeps_greedy_no_logits_no_rng() {
    let config = SamplerChainConfigV1::legacy(SamplingParametersV1::greedy());
    let mut chain = SamplerChainV1::new(config, &[]).expect("legacy chain");
    assert!(!chain.requires_logits());
    let selected = chain
        .select(17, None, &mut FixedRandom(f64::NAN))
        .expect("device argmax");
    assert_eq!(selected.token_id, 17);
}

#[test]
fn sampler_chain_contract_reports_post_filter_logprobs() {
    let parameters = SamplingParametersV1::new(1.0, 1.0, 0.0, 0.0).unwrap();
    let config = SamplerChainConfigV1::new(parameters)
        .with_top_k(1)
        .unwrap()
        .with_return_logprobs(true)
        .with_top_logprobs(1)
        .unwrap();
    let mut chain = SamplerChainV1::new(config, &[]).unwrap();
    let selected = chain
        .select(0, Some(&[2.0, 2.0, 1.0]), &mut FixedRandom(0.999))
        .unwrap();
    assert_eq!(selected.token_id, 0);
    assert_eq!(selected.top_logprobs.len(), 1);
    assert_eq!(selected.top_logprobs[0].token_id, 0);
    assert!((selected.logprob - 0.0).abs() < 1e-12);
}

#[test]
fn sampler_chain_contract_rejects_invalid_bias_and_all_mask() {
    let parameters = SamplingParametersV1::new(1.0, 1.0, 0.0, 0.0).unwrap();
    let config = SamplerChainConfigV1::new(parameters)
        .with_logit_bias(vec![LogitBiasV1 {
            token_id: 99,
            bias: 1.0,
        }])
        .unwrap()
        .with_ignore_eos(1);
    let mut chain = SamplerChainV1::new(config, &[]).unwrap();
    assert_eq!(
        chain.select(
            1,
            Some(&[f32::NEG_INFINITY, f32::NEG_INFINITY]),
            &mut FixedRandom(0.0)
        ),
        Err(SamplingError::TokenIdOutOfRange { token_id: 99 })
    );
    let config = SamplerChainConfigV1::new(parameters).with_ignore_eos(1);
    let mut chain = SamplerChainV1::new(config, &[]).unwrap();
    assert_eq!(
        chain.select(1, Some(&[f32::NEG_INFINITY, 0.0]), &mut FixedRandom(0.0)),
        Err(SamplingError::EmptyDistribution)
    );
}
