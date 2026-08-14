from __future__ import annotations

import numpy as np
import pytest


def reference_select(case: dict[str, object]) -> int:
    logits = np.asarray(case["logits"], dtype=np.float64)
    history = [int(token) for token in case["history"]]
    presence = float(case["presence_penalty"])
    frequency = float(case["frequency_penalty"])
    for token in range(logits.size):
        count = history.count(token)
        logits[token] -= count * frequency + (presence if count else 0.0)
    logits /= float(case["temperature"])
    token_ids = np.arange(logits.size)
    order = np.lexsort((token_ids, -logits))
    shifted = logits[order] - np.max(logits[order])
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities)
    top_p = float(case["top_p"])
    keep = int(np.searchsorted(np.cumsum(probabilities), top_p, side="left")) + 1
    keep = max(1, min(keep, order.size))
    order = order[:keep]
    probabilities = probabilities[:keep]
    threshold = float(case["uniform"]) * float(np.sum(probabilities))
    selected = int(np.searchsorted(np.cumsum(probabilities), threshold, side="right"))
    return int(order[min(selected, keep - 1)])


@pytest.mark.tier_h2
def test_profile_sampler_cases_have_independent_numpy_answers(load_json_fixture) -> None:
    fixture = load_json_fixture("profile_sampling_cases.json")
    assert fixture["schema_version"] == "profile-sampling-cases-v1"
    for case in fixture["cases"]:
        assert len(case["logits"]) not in {2, 4, 8}
        assert reference_select(case) == case["expected_token"], case["id"]
