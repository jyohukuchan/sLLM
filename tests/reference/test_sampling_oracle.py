from __future__ import annotations

import pytest

from tests.reference.oracles import MAX_ORACLE_ELEMENTS, sample_token


@pytest.mark.tier_h2
def test_sampling_helper_is_deterministic_for_tiny_fixed_seed_cases(load_json_fixture) -> None:
    fixture = load_json_fixture("sampling_cases.json")
    for case in fixture["cases"]:
        kwargs = {
            "temperature": case["temperature"],
            "top_p": case["top_p"],
            "seed": fixture["seed"],
        }
        first = sample_token(case["logits"], **kwargs)
        second = sample_token(case["logits"], **kwargs)
        assert first == second == case["expected_token"], case["id"]
        assert 0 <= first < len(case["logits"])


@pytest.mark.tier_h2
def test_sampling_helper_rejects_invalid_or_oversized_inputs() -> None:
    with pytest.raises(ValueError, match="temperature"):
        sample_token([0.0, 1.0], temperature=0.0, top_p=1.0, seed=1)
    with pytest.raises(ValueError, match="top_p"):
        sample_token([0.0, 1.0], temperature=1.0, top_p=0.0, seed=1)
    with pytest.raises(ValueError, match="finite"):
        sample_token([0.0, float("nan")], temperature=1.0, top_p=1.0, seed=1)
    with pytest.raises(ValueError, match="tiny"):
        sample_token([0.0] * (MAX_ORACLE_ELEMENTS + 1), temperature=1.0, top_p=1.0, seed=1)
