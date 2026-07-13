from __future__ import annotations

import json

import pytest

from ullm_openai_gateway.errors import ApiError
from ullm_openai_gateway.reasoning import ReasoningDialect
from ullm_openai_gateway.schemas import (
    MODEL_ID,
    decode_json_object,
    normalize_chat_request,
)


def request(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "hello"}],
    }
    value.update(updates)
    return value


def reasoning_dialect() -> ReasoningDialect:
    return ReasoningDialect(
        identity="synthetic.multi-token.v1",
        start_sequence=(10, 11),
        end_sequence=(20, 21, 22),
        forced_end_sequence=(20, 21, 22),
        max_budget_tokens=128,
        reserved_answer_tokens=1,
        effort_budgets=(("low", 32), ("medium", 64), ("high", 128)),
    )


def assert_error(
    value: dict[str, object], code: str, param: str | None = None
) -> ApiError:
    with pytest.raises(ApiError) as captured:
        normalize_chat_request(value)
    assert captured.value.code == code
    if param is not None:
        assert captured.value.param == param
    return captured.value


def test_defaults_and_complete_history_are_normalized() -> None:
    value = request(
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "c"},
                    {"type": "text", "text": "d"},
                ],
            },
        ]
    )
    normalized = normalize_chat_request(value)
    assert [(item.role, item.content) for item in normalized.messages] == [
        ("system", "system"),
        ("user", "a"),
        ("assistant", "b"),
        ("user", "cd"),
    ]
    assert normalized.max_completion_tokens == 256
    assert normalized.temperature == 0.6
    assert normalized.top_p == 0.95
    assert normalized.stream is False
    assert -(2**63) <= normalized.seed < 2**63


@pytest.mark.parametrize(
    "messages,param",
    [
        ([{"role": "assistant", "content": "x"}], "messages[0].role"),
        ([{"role": "system", "content": "x"}], "messages"),
        (
            [
                {"role": "user", "content": "x"},
                {"role": "user", "content": "y"},
            ],
            "messages[1].role",
        ),
        (
            [
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": "y"},
            ],
            "messages",
        ),
    ],
)
def test_invalid_role_orders_are_rejected(
    messages: list[dict[str, object]], param: str
) -> None:
    assert_error(request(messages=messages), "invalid_request_error", param)


def test_unknown_null_is_ignored_but_nonnull_is_unsupported_at_each_level() -> None:
    accepted = request(
        future=None,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "x", "future": None}],
                "future": None,
            }
        ],
        stream_options={"include_usage": None, "future": None},
    )
    assert normalize_chat_request(accepted).messages[0].content == "x"
    assert_error(request(future=True), "unsupported_parameter", "future")
    assert_error(
        request(messages=[{"role": "user", "content": "x", "name": "n"}]),
        "unsupported_parameter",
        "messages[0].name",
    )


@pytest.mark.parametrize(
    "updates,code,param",
    [
        ({"model": "other"}, "model_not_found", "model"),
        (
            {"max_tokens": 1, "max_completion_tokens": 1},
            "unsupported_parameter",
            "max_completion_tokens",
        ),
        ({"max_tokens": 0}, "invalid_request_error", "max_tokens"),
        ({"max_tokens": 513}, "invalid_request_error", "max_tokens"),
        ({"max_tokens": True}, "invalid_request_error", "max_tokens"),
        ({"temperature": False}, "invalid_request_error", "temperature"),
        ({"temperature": 2.1}, "invalid_request_error", "temperature"),
        ({"top_p": 0}, "invalid_request_error", "top_p"),
        ({"seed": True}, "invalid_request_error", "seed"),
        ({"seed": 2**63}, "invalid_request_error", "seed"),
        ({"n": 2}, "unsupported_parameter", "n"),
        ({"stop": ["x"]}, "unsupported_parameter", "stop"),
        ({"frequency_penalty": 0.1}, "unsupported_parameter", "frequency_penalty"),
        ({"logit_bias": {"1": 2}}, "unsupported_parameter", "logit_bias"),
        ({"logprobs": True}, "unsupported_parameter", "logprobs"),
        ({"top_logprobs": 1}, "unsupported_parameter", "top_logprobs"),
        ({"tools": []}, "unsupported_parameter", "tools"),
        (
            {"stream_options": {"include_usage": True}},
            "invalid_request_error",
            "stream_options.include_usage",
        ),
    ],
)
def test_parameter_contract(updates: dict[str, object], code: str, param: str) -> None:
    assert_error(request(**updates), code, param)


def test_reasoning_fields_require_a_v2_dialect() -> None:
    assert_error(
        request(reasoning_effort="low"),
        "unsupported_parameter",
        "reasoning_effort",
    )
    assert_error(
        request(thinking_budget_tokens=8),
        "unsupported_parameter",
        "thinking_budget_tokens",
    )


def test_reasoning_effort_and_exact_budget_normalize_with_reservation() -> None:
    dialect = reasoning_dialect()
    effort = normalize_chat_request(
        request(reasoning_effort="medium", max_tokens=128),
        reasoning_dialect=dialect,
    )
    assert effort.reasoning is not None
    assert effort.reasoning.enabled is True
    assert effort.reasoning.budget_tokens == 64
    exact = normalize_chat_request(
        request(thinking_budget_tokens=0, max_tokens=8),
        reasoning_dialect=dialect,
    )
    assert exact.reasoning is not None
    assert exact.reasoning.budget_tokens == 0
    none = normalize_chat_request(
        request(reasoning_effort="none"),
        reasoning_dialect=dialect,
    )
    assert none.reasoning is not None
    assert none.reasoning.enabled is False


def test_reasoning_budget_reservation_is_not_silently_clamped() -> None:
    with pytest.raises(ApiError) as captured:
        normalize_chat_request(
            request(thinking_budget_tokens=8, max_tokens=11),
            reasoning_dialect=reasoning_dialect(),
        )
    assert captured.value.code == "invalid_request_error"
    assert captured.value.param == "thinking_budget_tokens"


def test_unbounded_reasoning_still_reserves_forced_end_and_answer() -> None:
    normalized = normalize_chat_request(
        request(thinking_budget_tokens=-1, max_tokens=4),
        reasoning_dialect=reasoning_dialect(),
    )
    assert normalized.reasoning is not None
    assert normalized.reasoning.budget_tokens is None
    with pytest.raises(ApiError) as captured:
        normalize_chat_request(
            request(thinking_budget_tokens=-1, max_tokens=3),
            reasoning_dialect=reasoning_dialect(),
        )
    assert captured.value.code == "invalid_request_error"
    assert captured.value.param == "thinking_budget_tokens"


def test_neutral_parameters_and_null_optionals_are_accepted() -> None:
    normalized = normalize_chat_request(
        request(
            max_tokens=None,
            max_completion_tokens=512,
            n=1,
            stop=[],
            frequency_penalty=0,
            presence_penalty=0.0,
            logit_bias={},
            logprobs=False,
            top_logprobs=None,
            user="opaque",
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assert normalized.max_completion_tokens == 512
    assert normalized.stream is True
    assert normalized.include_usage is True


def test_non_text_content_part_is_unsupported() -> None:
    assert_error(
        request(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image_url", "text": "x"}],
                }
            ]
        ),
        "unsupported_parameter",
        "messages[0].content[0].type",
    )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"model":"a","model":"b"}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b"[]",
        b"{",
        b"\xff",
    ],
)
def test_strict_json_rejects_duplicate_nonfinite_nonobject_and_invalid_utf8(
    raw: bytes,
) -> None:
    with pytest.raises(ApiError) as captured:
        decode_json_object(raw)
    assert captured.value.status_code == 400
    assert captured.value.code == "invalid_request_error"


def test_json_round_trip_preserves_unicode_content() -> None:
    raw = json.dumps(
        request(messages=[{"role": "user", "content": "日本語"}]), ensure_ascii=False
    ).encode()
    normalized = normalize_chat_request(decode_json_object(raw))
    assert normalized.messages[0].content == "日本語"


def test_model_selection_precedes_parameter_validation() -> None:
    error = assert_error(
        request(model="missing", unknown_control=True), "model_not_found", "model"
    )
    assert error.status_code == 404


def test_deep_json_and_unpaired_surrogate_text_are_rejected() -> None:
    with pytest.raises(ApiError) as captured:
        decode_json_object(("[" * 2_000 + "]" * 2_000).encode())
    assert captured.value.status_code == 400
    assert_error(
        request(messages=[{"role": "user", "content": "\ud800"}]),
        "invalid_request_error",
        "messages[0].content",
    )


def test_huge_integer_and_overflowing_json_float_are_400_errors() -> None:
    huge = 10**4_000
    assert_error(
        request(temperature=huge),
        "invalid_request_error",
        "temperature",
    )
    assert_error(
        request(frequency_penalty=huge),
        "unsupported_parameter",
        "frequency_penalty",
    )
    raw = (
        b'{"model":"ullm-qwen3-14b-sq8","messages":'
        b'[{"role":"user","content":"x"}],"temperature":1e999}'
    )
    with pytest.raises(ApiError) as captured:
        decode_json_object(raw)
    assert captured.value.status_code == 400
