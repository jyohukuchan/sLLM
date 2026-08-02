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
