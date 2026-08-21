from __future__ import annotations

import pytest

from tests.contracts.api_contract import validate_chat_request


@pytest.mark.tier_h1
def test_pinned_openai_profile_fixture_is_accepted_and_negative_cases_reject(
    load_json_fixture,
) -> None:
    fixture = load_json_fixture("openai_chat_profile_v1.json")
    positive_requests = [fixture["positive"]["request"]]
    positive_requests.extend(
        case["request"] for case in fixture.get("positive_variants", [])
    )
    for request in positive_requests:
        positive = validate_chat_request(
            request, served_models=(fixture["served_model"],)
        )
        assert positive.accepted, request
    for case in fixture["negative"]:
        if isinstance(case["body"], str):
            continue
        result = validate_chat_request(
            case["body"], served_models=(fixture["served_model"],)
        )
        assert not result.accepted, case["case_id"]
        assert result.error is not None, case["case_id"]
        assert (result.error.status, result.error.code, result.error.param) == (
            case["status"],
            case["code"],
            case["param"],
        ), case["case_id"]
