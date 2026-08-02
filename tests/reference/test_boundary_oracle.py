from __future__ import annotations

import pytest

from tests.reference.oracles import DEFAULT_SEED, MAX_CASES, boundary_cases


@pytest.mark.tier_h2
def test_boundary_generation_is_seeded_bounded_and_complete(load_json_fixture) -> None:
    fixture = load_json_fixture("boundary_cases.json")
    generated = boundary_cases(fixture["seed"])
    repeated = boundary_cases(fixture["seed"])

    assert fixture["seed"] == DEFAULT_SEED
    assert generated == repeated
    assert len(generated) == len(fixture["cases"]) <= MAX_CASES
    assert {case.case_id for case in generated} == {case["id"] for case in fixture["cases"]}

    values = {value for case in generated for value in case.values}
    assert {0, 1, 3, 7, 15, 16, 17, 37, 73, 255, 256, 257} <= values
    assert all(isinstance(value, int) and value >= 0 for value in values)


@pytest.mark.tier_h2
def test_boundary_generator_rejects_non_integer_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        boundary_cases("20260803")
