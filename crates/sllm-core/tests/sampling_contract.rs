// Test portions adapted from llama.cpp.
// Provenance: THIRD_PARTY_NOTICES.md#llama-cpp-profile-v1-sampling-tests-001
// Upstream: https://github.com/ggml-org/llama.cpp @ f5919bf458ef190468b5c329bb293f8a54a1e69c, tests/test-sampling.cpp
// SPDX-License-Identifier: MIT

use std::fs;
use std::path::PathBuf;

use serde::Deserialize;
use sllm_core::{ProfileSamplerV1, SamplingError, SamplingParametersV1, SamplingRandomSource};

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
