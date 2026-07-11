"""Stable public and private gateway errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ApiError(Exception):
    status_code: int
    error_type: str
    code: str
    message: str
    param: str | None = None
    headers: dict[str, str] | None = None

    def envelope(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


def invalid_request(message: str, param: str | None = None) -> ApiError:
    return ApiError(
        400, "invalid_request_error", "invalid_request_error", message, param
    )


def unsupported_parameter(param: str) -> ApiError:
    return ApiError(
        400,
        "invalid_request_error",
        "unsupported_parameter",
        "The requested parameter is not supported.",
        param,
    )


def context_length_exceeded() -> ApiError:
    return ApiError(
        400,
        "invalid_request_error",
        "context_length_exceeded",
        "The prompt and requested completion exceed the model context length.",
        "messages",
    )


def model_not_found() -> ApiError:
    return ApiError(
        404,
        "invalid_request_error",
        "model_not_found",
        "The requested model does not exist.",
        "model",
    )
