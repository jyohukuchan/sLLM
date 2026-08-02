"""Python-side v1 request contract used until the Rust API adapter is exposed.

This module intentionally validates only request shape and fail-closed policy. It
does not start a server, call a model, or implement generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping


SUPPORTED_REQUEST_FIELDS = frozenset(
    {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_completion_tokens",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "stream",
        "n",
    }
)
SUPPORTED_MESSAGE_FIELDS = frozenset({"role", "content"})
SUPPORTED_ROLES = frozenset({"system", "user", "assistant"})


@dataclass(frozen=True)
class ApiError:
    status: int
    error_type: str
    code: str
    param: str | None
    message: str

    def envelope(self) -> dict[str, dict[str, str | None]]:
        """Return the exact non-streaming error envelope shape."""

        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    error: ApiError | None = None

    @property
    def status(self) -> int:
        return 200 if self.accepted else self.error.status  # type: ignore[union-attr]


def _reject(code: str, param: str | None, message: str, status: int = 400) -> ValidationResult:
    return ValidationResult(
        accepted=False,
        error=ApiError(
            status=status,
            error_type="invalid_request_error" if status != 429 else "rate_limit_error",
            code=code,
            param=param,
            message=message,
        ),
    )


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def validate_chat_request(
    payload: Any,
    *,
    served_models: tuple[str, ...] = ("fixture-model",),
) -> ValidationResult:
    """Validate the supported subset and reject unknown input explicitly."""

    if not isinstance(payload, Mapping):
        return _reject("invalid_json", None, "request body must be a JSON object")
    if any(not isinstance(key, str) for key in payload):
        return _reject("invalid_json", None, "object member names must be strings")

    unknown = sorted(set(payload) - SUPPORTED_REQUEST_FIELDS)
    if unknown:
        return _reject("unsupported_parameter", unknown[0], f"unsupported field: {unknown[0]}")

    model = payload.get("model")
    if not isinstance(model, str) or not model:
        return _reject("invalid_value", "model", "model must be a non-empty string")
    if model not in served_models:
        return _reject("model_not_found", "model", f"unknown model: {model}", status=404)

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return _reject("invalid_value", "messages", "messages must be a non-empty array")
    for message in messages:
        if not isinstance(message, Mapping):
            return _reject("invalid_value", "messages", "each message must be an object")
        if set(message) - SUPPORTED_MESSAGE_FIELDS:
            return _reject("unsupported_parameter", "messages", "message field is unsupported")
        role = message.get("role")
        if role not in SUPPORTED_ROLES:
            return _reject("unsupported_parameter", "messages", "message role is unsupported")
        if not isinstance(message.get("content"), str):
            return _reject("unsupported_parameter", "messages", "only text content is supported")

    if "temperature" in payload and (
        not _is_finite_number(payload["temperature"])
        or not 0.0 <= payload["temperature"] <= 2.0
    ):
        return _reject("invalid_value", "temperature", "temperature must be in [0, 2]")
    if "top_p" in payload and (
        not _is_finite_number(payload["top_p"]) or not 0.0 <= payload["top_p"] <= 1.0
    ):
        return _reject("invalid_value", "top_p", "top_p must be in [0, 1]")
    if "max_completion_tokens" in payload and (
        isinstance(payload["max_completion_tokens"], bool)
        or not isinstance(payload["max_completion_tokens"], int)
        or payload["max_completion_tokens"] < 1
    ):
        return _reject(
            "invalid_value",
            "max_completion_tokens",
            "max_completion_tokens must be a positive integer",
        )
    if "stop" in payload:
        stop = payload["stop"]
        if isinstance(stop, str):
            pass
        elif isinstance(stop, list) and all(isinstance(item, str) for item in stop):
            pass
        else:
            return _reject("invalid_value", "stop", "stop must be a string or string array")
    for field in ("presence_penalty", "frequency_penalty"):
        if field in payload and (
            not _is_finite_number(payload[field]) or not -2.0 <= payload[field] <= 2.0
        ):
            return _reject("invalid_value", field, f"{field} must be in [-2, 2]")
    if "stream" in payload and not isinstance(payload["stream"], bool):
        return _reject("invalid_value", "stream", "stream must be boolean")
    if "n" in payload and (
        isinstance(payload["n"], bool) or not isinstance(payload["n"], int) or payload["n"] != 1
    ):
        return _reject("invalid_value", "n", "n must equal 1")

    return ValidationResult(accepted=True)
