from __future__ import annotations

import pytest

from tests.contracts.api_contract import validate_chat_request


@pytest.mark.tier_h1
def test_valid_fixture_requests_are_accepted(load_json_fixture) -> None:
    fixture = load_json_fixture("api_cases.json")
    for case in fixture["valid"]:
        result = validate_chat_request(
            case["request"], served_models=(fixture["served_model"],)
        )
        assert result.accepted, case["id"]
        assert result.error is None, case["id"]


@pytest.mark.tier_h1
def test_invalid_fixture_requests_match_profile_error_mapping(load_json_fixture) -> None:
    fixture = load_json_fixture("api_cases.json")
    for case in fixture["invalid"]:
        result = validate_chat_request(
            case["request"], served_models=(fixture["served_model"],)
        )
        assert not result.accepted, case["id"]
        assert result.error is not None, case["id"]
        assert (result.error.status, result.error.code, result.error.param) == (
            case["status"],
            case["code"],
            case["param"],
        ), case["id"]


@pytest.mark.tier_h1
def test_error_envelope_has_no_unregistered_or_missing_members(load_json_fixture) -> None:
    fixture = load_json_fixture("api_cases.json")
    result = validate_chat_request(
        fixture["invalid"][0]["request"], served_models=(fixture["served_model"],)
    )

    assert result.error is not None
    assert set(result.error.envelope()) == {"error"}
    assert set(result.error.envelope()["error"]) == {
        "message",
        "type",
        "param",
        "code",
    }


@pytest.mark.tier_h1
def test_numeric_and_boolean_types_are_not_silently_coerced() -> None:
    base = {"model": "fixture-model", "messages": [{"role": "user", "content": "hello"}]}
    invalid = (
        {**base, "temperature": "0.5"},
        {**base, "stream": "false"},
        {**base, "n": True},
    )

    for request in invalid:
        result = validate_chat_request(request)
        assert not result.accepted
        assert result.error is not None
        assert result.error.code == "invalid_value"


@pytest.mark.tier_h1
def test_phase40_logprob_schema_and_sampler_bounds_are_fail_closed() -> None:
    base = {
        "model": "fixture-model",
        "messages": [{"role": "user", "content": "Return JSON."}],
    }
    accepted = validate_chat_request(
        {
            **base,
            "n": 8,
            "logit_bias": {"0": -100, "4294967295": 100},
            "logprobs": True,
            "top_logprobs": 0,
            "response_format": {"type": "json_object"},
            "sllm": {
                "sampling": {
                    "chain_version": 1,
                    "top_k": 0,
                    "typical_p": 1.0,
                    "repeat_penalty": 100.0,
                }
            },
        }
    )
    assert accepted.accepted

    invalid_requests = (
        {**base, "top_logprobs": 1},
        {**base, "logit_bias": {"0": 100.1}},
        {**base, "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": "hello"}]},
        {
            **base,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "string", "pattern": "x"},
                },
            },
        },
        {**base, "sllm": {"sampling": {"mirostat": {"version": 2}, "top_k": 4}}},
    )
    for request in invalid_requests:
        result = validate_chat_request(request)
        assert not result.accepted
        assert result.error is not None
        assert result.error.code == "invalid_value"
