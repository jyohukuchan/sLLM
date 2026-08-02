from __future__ import annotations

import pytest

from tests.contracts.api_contract import validate_chat_request


@pytest.mark.tier_h1
def test_unknown_request_members_are_rejected_without_coercion() -> None:
    result = validate_chat_request(
        {
            "model": "fixture-model",
            "messages": [{"role": "user", "content": "hello"}],
            "unknown": 1,
        }
    )

    assert not result.accepted
    assert result.error is not None
    assert (result.error.status, result.error.code, result.error.param) == (
        400,
        "unsupported_parameter",
        "unknown",
    )


@pytest.mark.tier_h1
def test_malformed_root_is_an_error_not_an_empty_request() -> None:
    for payload in (
        [{"role": "user", "content": "hello"}],
        {1: "non-string member name"},
    ):
        result = validate_chat_request(payload)

        assert not result.accepted
        assert result.error is not None
        assert (result.error.status, result.error.code, result.error.param) == (
            400,
            "invalid_json",
            None,
        )


@pytest.mark.tier_h1
def test_unknown_model_is_not_reinterpreted_as_the_fixture_model() -> None:
    result = validate_chat_request(
        {
            "model": "fixture-model ",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    assert not result.accepted
    assert result.error is not None
    assert (result.error.status, result.error.code, result.error.param) == (
        404,
        "model_not_found",
        "model",
    )
