"""Strict OpenAI request normalization without framework-native coercion."""

from __future__ import annotations

import json
import math
import secrets
from dataclasses import dataclass
from typing import Any

from .errors import invalid_request, model_not_found, unsupported_parameter
from .reasoning import ReasoningDialect, ReasoningError, ReasoningRequest


MODEL_ID = "ullm-qwen3-14b-sq8"
CONTEXT_LENGTH = 4_096
DEFAULT_MAX_COMPLETION_TOKENS = 256
MAX_COMPLETION_TOKENS = 512
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
TOP_K = 20
EOS_TOKEN_IDS = (151_645, 151_643)

ROOT_FIELDS = frozenset(
    {
        "model",
        "messages",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "seed",
        "n",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "user",
        "top_k",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "functions",
        "function_call",
        "response_format",
        "modalities",
        "audio",
        "reasoning_effort",
        "thinking_budget_tokens",
        "store",
        "metadata",
        "service_tier",
        "prediction",
    }
)
ALWAYS_UNSUPPORTED = frozenset(
    {
        "top_k",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "functions",
        "function_call",
        "response_format",
        "modalities",
        "audio",
        "store",
        "metadata",
        "service_tier",
        "prediction",
    }
)


class DuplicateKeyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NormalizedMessage:
    role: str
    content: str

    reasoning_content: str | None = None

    def as_template_value(self, *, include_reasoning_content: bool = False) -> dict[str, str]:
        value = {"role": self.role, "content": self.content}
        if include_reasoning_content and self.reasoning_content is not None:
            value["reasoning_content"] = self.reasoning_content
        return value


@dataclass(frozen=True, slots=True)
class NormalizedChatRequest:
    model_id: str
    messages: tuple[NormalizedMessage, ...]
    stream: bool
    include_usage: bool
    max_completion_tokens: int
    temperature: float
    top_p: float
    seed: int
    reasoning: ReasoningRequest | None = None


def decode_json_object(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise invalid_request("The request body is not valid UTF-8.") from error

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateKeyError(key)
            result[key] = value
        return result

    def reject_constant(_: str) -> Any:
        raise ValueError("non-finite JSON number")

    def finite_float(raw: str) -> float:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        return value

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except (
        json.JSONDecodeError,
        DuplicateKeyError,
        ValueError,
        RecursionError,
    ) as error:
        raise invalid_request("The request body is not valid JSON.") from error
    if not isinstance(value, dict):
        raise invalid_request("The request body root must be an object.")
    return value


def normalize_chat_request(
    value: dict[str, Any],
    *,
    model_id: str = MODEL_ID,
    max_completion_tokens: int = MAX_COMPLETION_TOKENS,
    temperature_supported: bool = True,
    top_p_supported: bool = True,
    reasoning_dialect: ReasoningDialect | None = None,
) -> NormalizedChatRequest:
    model = _required(value, "model")
    if not isinstance(model, str):
        raise invalid_request("model must be a string.", "model")
    if model != model_id:
        raise model_not_found()

    raw_messages = _required(value, "messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise invalid_request("messages must be a nonempty array.", "messages")

    _validate_unknown_fields(value, ROOT_FIELDS, "")
    for field, item in value.items():
        if field in ALWAYS_UNSUPPORTED and item is not None:
            raise unsupported_parameter(field)

    messages = _normalize_messages(raw_messages)
    stream = _optional_bool(value, "stream", False)
    include_usage = _normalize_stream_options(value.get("stream_options"), stream)
    maximum = _normalize_maximum(value, max_completion_tokens=max_completion_tokens)
    temperature = _optional_number(
        value,
        "temperature",
        DEFAULT_TEMPERATURE if temperature_supported else 0.0,
        minimum=0.0,
        maximum=2.0,
    )
    top_p = _optional_number(
        value,
        "top_p",
        DEFAULT_TOP_P if top_p_supported else 1.0,
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    if not temperature_supported and temperature != 0.0:
        raise unsupported_parameter("temperature")
    if not top_p_supported and top_p != 1.0:
        raise unsupported_parameter("top_p")
    seed = _normalize_seed(value.get("seed"))
    _normalize_neutral_fields(value)
    reasoning = _normalize_reasoning(value, reasoning_dialect, maximum)
    if value.get("user") is not None and not isinstance(value["user"], str):
        raise invalid_request("user must be a string.", "user")
    return NormalizedChatRequest(
        model_id=model_id,
        messages=messages,
        stream=stream,
        include_usage=include_usage,
        max_completion_tokens=maximum,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        reasoning=reasoning,
    )


def _normalize_reasoning(
    value: dict[str, Any],
    dialect: ReasoningDialect | None,
    max_completion_tokens: int,
) -> ReasoningRequest | None:
    effort = value.get("reasoning_effort")
    raw_budget = value.get("thinking_budget_tokens")
    if effort is None and raw_budget is None:
        if dialect is None:
            return None
        enabled = dialect.enabled_by_default
        budget = None
        if enabled:
            budget = dialect.max_budget_tokens
        return _build_reasoning_request(dialect, enabled, budget, max_completion_tokens)
    if dialect is None:
        field = "reasoning_effort" if effort is not None else "thinking_budget_tokens"
        raise unsupported_parameter(field)
    if effort is not None and raw_budget is not None:
        raise unsupported_parameter("thinking_budget_tokens")
    if effort is not None:
        if not isinstance(effort, str) or effort not in {"none", "low", "medium", "high"}:
            raise invalid_request(
                "reasoning_effort must be one of none, low, medium, or high.",
                "reasoning_effort",
            )
        if effort == "none":
            return _build_reasoning_request(dialect, False, None, max_completion_tokens)
        try:
            budget = dialect.budget_for_effort(effort)
        except ReasoningError as error:
            raise invalid_request("reasoning_effort is not supported by this model.", "reasoning_effort") from error
        return _build_reasoning_request(dialect, True, budget, max_completion_tokens)
    if isinstance(raw_budget, bool) or not isinstance(raw_budget, int):
        raise invalid_request("thinking_budget_tokens must be an integer.", "thinking_budget_tokens")
    if raw_budget < -1 or raw_budget > dialect.max_budget_tokens:
        raise invalid_request("thinking_budget_tokens is outside the model budget.", "thinking_budget_tokens")
    budget = None if raw_budget == -1 else raw_budget
    return _build_reasoning_request(dialect, True, budget, max_completion_tokens)


def _build_reasoning_request(
    dialect: ReasoningDialect,
    enabled: bool,
    budget: int | None,
    max_completion_tokens: int,
) -> ReasoningRequest:
    if enabled and budget is not None:
        reserved = len(dialect.forced_end_sequence) + dialect.reserved_answer_tokens
        if budget + reserved > max_completion_tokens:
            raise invalid_request(
                "thinking budget and reserved answer tokens exceed max_completion_tokens.",
                "thinking_budget_tokens",
            )
    elif enabled:
        reserved = len(dialect.forced_end_sequence) + dialect.reserved_answer_tokens
        if reserved > max_completion_tokens:
            raise invalid_request(
                "reasoning end sequence and reserved answer tokens exceed max_completion_tokens.",
                "thinking_budget_tokens",
            )
    return ReasoningRequest(
        enabled=enabled,
        budget_tokens=budget,
        history_reasoning_policy=dialect.history_reasoning_policy,
        dialect_id=dialect.identity,
    )


def _required(value: dict[str, Any], field: str) -> Any:
    if field not in value or value[field] is None:
        raise invalid_request(f"{field} is required.", field)
    return value[field]


def _validate_unknown_fields(
    value: dict[str, Any], allowed: frozenset[str], prefix: str
) -> None:
    for field, item in value.items():
        if field not in allowed and item is not None:
            path = f"{prefix}.{field}" if prefix else field
            raise unsupported_parameter(path)


def _normalize_messages(raw_messages: list[Any]) -> tuple[NormalizedMessage, ...]:
    result: list[NormalizedMessage] = []
    for index, item in enumerate(raw_messages):
        path = f"messages[{index}]"
        if not isinstance(item, dict):
            raise invalid_request("Each message must be an object.", path)
        allowed = {"role", "content"}
        if item.get("role") == "assistant":
            allowed.add("reasoning_content")
        _validate_unknown_fields(item, frozenset(allowed), path)
        if "role" not in item or item["role"] is None:
            raise invalid_request("role is required.", f"{path}.role")
        if "content" not in item or item["content"] is None:
            raise invalid_request("content is required.", f"{path}.content")
        role = item["role"]
        content = item["content"]
        if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
            raise invalid_request("Message role is invalid.", f"{path}.role")
        reasoning_content = item.get("reasoning_content")
        if reasoning_content is not None:
            if role != "assistant" or not isinstance(reasoning_content, str):
                raise invalid_request(
                    "reasoning_content is only a string on assistant messages.",
                    f"{path}.reasoning_content",
                )
            _require_scalar_text(reasoning_content, f"{path}.reasoning_content")
        result.append(
            NormalizedMessage(
                role=role,
                content=_normalize_content(content, path),
                reasoning_content=reasoning_content,
            )
        )
    _validate_role_order(result)
    return tuple(result)


def _normalize_content(value: Any, message_path: str) -> str:
    if isinstance(value, str):
        _require_scalar_text(value, f"{message_path}.content")
        return value
    if not isinstance(value, list) or not value:
        raise invalid_request(
            "Message content must be a string or nonempty text-part array.",
            f"{message_path}.content",
        )
    parts: list[str] = []
    for index, part in enumerate(value):
        path = f"{message_path}.content[{index}]"
        if not isinstance(part, dict):
            raise invalid_request("Content parts must be objects.", path)
        _validate_unknown_fields(part, frozenset({"type", "text"}), path)
        if "type" not in part or part["type"] is None:
            raise invalid_request("type is required.", f"{path}.type")
        part_type = part["type"]
        if not isinstance(part_type, str):
            raise invalid_request("Content-part type must be a string.", f"{path}.type")
        if part_type != "text":
            raise unsupported_parameter(f"{path}.type")
        if "text" not in part or part["text"] is None:
            raise invalid_request("text is required.", f"{path}.text")
        text = part["text"]
        if not isinstance(text, str):
            raise invalid_request("Content-part text must be a string.", f"{path}.text")
        _require_scalar_text(text, f"{path}.text")
        parts.append(text)
    return "".join(parts)


def _validate_role_order(messages: list[NormalizedMessage]) -> None:
    offset = 1 if messages[0].role == "system" else 0
    if offset == len(messages):
        raise invalid_request(
            "The conversation must contain a user message.", "messages"
        )
    for index in range(offset, len(messages)):
        expected = "user" if (index - offset) % 2 == 0 else "assistant"
        if messages[index].role != expected:
            raise invalid_request(
                "Message roles are not in the required order.",
                f"messages[{index}].role",
            )
    if messages[-1].role != "user":
        raise invalid_request(
            "The conversation must end with a user message.", "messages"
        )


def _normalize_stream_options(value: Any, stream: bool) -> bool:
    if value is None:
        return False
    if not isinstance(value, dict):
        raise invalid_request("stream_options must be an object.", "stream_options")
    _validate_unknown_fields(value, frozenset({"include_usage"}), "stream_options")
    include_usage = _optional_bool(value, "include_usage", False)
    if include_usage and not stream:
        raise invalid_request(
            "stream_options.include_usage requires stream=true.",
            "stream_options.include_usage",
        )
    return include_usage


def _normalize_maximum(value: dict[str, Any], *, max_completion_tokens: int) -> int:
    first = value.get("max_tokens")
    second = value.get("max_completion_tokens")
    if first is not None and second is not None:
        raise unsupported_parameter("max_completion_tokens")
    maximum = second if second is not None else first
    if maximum is None:
        return min(DEFAULT_MAX_COMPLETION_TOKENS, max_completion_tokens)
    if isinstance(maximum, bool) or not isinstance(maximum, int):
        field = "max_completion_tokens" if second is not None else "max_tokens"
        raise invalid_request("The completion maximum must be an integer.", field)
    if not 1 <= maximum <= max_completion_tokens:
        field = "max_completion_tokens" if second is not None else "max_tokens"
        raise invalid_request(
            f"The completion maximum is outside 1..{max_completion_tokens}.", field
        )
    return maximum


def _optional_bool(value: dict[str, Any], field: str, default: bool) -> bool:
    item = value.get(field)
    if item is None:
        return default
    if not isinstance(item, bool):
        raise invalid_request(f"{field} must be a boolean.", field)
    return item


def _optional_number(
    value: dict[str, Any],
    field: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
) -> float:
    item = value.get(field)
    if item is None:
        return default
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise invalid_request(f"{field} must be a number.", field)
    if isinstance(item, int):
        lower_ok = item >= minimum if minimum_inclusive else item > minimum
        if not lower_ok or item > maximum:
            raise invalid_request(f"{field} is outside the supported range.", field)
        return float(item)
    lower_ok = item >= minimum if minimum_inclusive else item > minimum
    if not math.isfinite(item) or not lower_ok or item > maximum:
        raise invalid_request(f"{field} is outside the supported range.", field)
    return float(item)


def _normalize_seed(value: Any) -> int:
    if value is None:
        unsigned = secrets.randbits(64)
        return unsigned if unsigned < 2**63 else unsigned - 2**64
    if isinstance(value, bool) or not isinstance(value, int):
        raise invalid_request("seed must be a signed 64-bit integer.", "seed")
    if not -(2**63) <= value < 2**63:
        raise invalid_request("seed must be a signed 64-bit integer.", "seed")
    return value


def _normalize_neutral_fields(value: dict[str, Any]) -> None:
    n = value.get("n")
    if n is not None:
        if isinstance(n, bool) or not isinstance(n, int):
            raise invalid_request("n must be an integer.", "n")
        if n != 1:
            raise unsupported_parameter("n")

    for field in ("frequency_penalty", "presence_penalty"):
        item = value.get(field)
        if item is None:
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise invalid_request(f"{field} must be a finite number.", field)
        if isinstance(item, float) and not math.isfinite(item):
            raise invalid_request(f"{field} must be a finite number.", field)
        if item != 0:
            raise unsupported_parameter(field)

    stop = value.get("stop")
    if stop is not None:
        if stop == "" or (isinstance(stop, list) and not stop):
            pass
        elif not isinstance(stop, (str, list)):
            raise invalid_request("stop must be a string or array.", "stop")
        else:
            raise unsupported_parameter("stop")

    bias = value.get("logit_bias")
    if bias is not None:
        if not isinstance(bias, dict):
            raise invalid_request("logit_bias must be an object.", "logit_bias")
        if bias:
            raise unsupported_parameter("logit_bias")

    logprobs = value.get("logprobs")
    if logprobs is not None:
        if not isinstance(logprobs, bool):
            raise invalid_request("logprobs must be a boolean.", "logprobs")
        if logprobs:
            raise unsupported_parameter("logprobs")
    if value.get("top_logprobs") is not None:
        raise unsupported_parameter("top_logprobs")


def _require_scalar_text(value: str, param: str) -> None:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise invalid_request(
            "Text contains an invalid Unicode scalar value.", param
        ) from error
