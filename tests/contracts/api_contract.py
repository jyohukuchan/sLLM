"""Python-side v1 request contract used until the Rust API adapter is exposed.

This module intentionally validates only request shape and fail-closed policy. It
does not start a server, call a model, or implement generation.
"""

from __future__ import annotations

import json
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
        "seed",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "response_format",
        "sllm",
    }
)
SUPPORTED_MESSAGE_FIELDS = frozenset({"role", "content", "reasoning_content"})
SUPPORTED_ROLES = frozenset({"system", "user", "assistant"})
SUPPORTED_SLLM_FIELDS = frozenset({"thinking", "separate_reasoning", "resumable", "sampling"})
SUPPORTED_SAMPLER_FIELDS = frozenset(
    {
        "chain_version",
        "top_k",
        "min_p",
        "typical_p",
        "repeat_penalty",
        "repeat_last_n",
        "ignore_eos",
        "dry",
        "xtc",
        "mirostat",
        "dynamic_temperature",
    }
)
SUPPORTED_DRY_FIELDS = frozenset(
    {"multiplier", "base", "allowed_length", "penalty_last_n", "sequence_breakers"}
)
SUPPORTED_XTC_FIELDS = frozenset({"probability", "threshold", "min_keep"})
SUPPORTED_MIROSTAT_FIELDS = frozenset({"version", "tau", "eta"})
SUPPORTED_DYNAMIC_TEMPERATURE_FIELDS = frozenset({"range", "exponent"})
SUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "$ref",
        "$defs",
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "anyOf",
    }
)
MAX_LOGIT_BIAS_ENTRIES = 4_096
MAX_SCHEMA_BYTES = 64 * 1024
MAX_SCHEMA_DEPTH = 32
MAX_SCHEMA_ENUM = 256
MAX_SCHEMA_PROPERTIES = 1_024
MAX_SAMPLER_TOP_K = 1_000_000
MAX_SAMPLER_HISTORY = 4_096
MAX_SEQUENCE_BREAKERS = 16
MAX_SEQUENCE_BREAKER_BYTES = 1_024


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


def _unknown_fields(value: Mapping[str, Any], supported: frozenset[str]) -> str | None:
    unknown = sorted(set(value) - supported)
    return unknown[0] if unknown else None


def _validate_schema_node(
    value: Any,
    *,
    depth: int = 0,
    refs: frozenset[str] = frozenset(),
    definitions: Mapping[str, Any] | None = None,
) -> str | None:
    """Return a fail-closed reason for the bounded JSON-Schema subset."""

    if not isinstance(value, Mapping):
        return "schema nodes must be objects"
    if depth > MAX_SCHEMA_DEPTH:
        return "schema nesting exceeds 32"
    unknown = _unknown_fields(value, SUPPORTED_SCHEMA_KEYWORDS)
    if unknown:
        return f"unsupported JSON Schema keyword {unknown}"

    defs = value.get("$defs")
    if defs is not None:
        if not isinstance(defs, Mapping):
            return "$defs must be an object"
        for name, definition in defs.items():
            if not isinstance(name, str):
                return "$defs names must be strings"
            reason = _validate_schema_node(
                definition,
                depth=depth + 1,
                refs=refs,
                definitions=defs,
            )
            if reason:
                return reason
        definitions = defs

    reference = value.get("$ref")
    if reference is not None:
        if not isinstance(reference, str):
            return "$ref must be a string"
        if not reference.startswith("#/$defs/") or reference == "#/$defs/":
            return f"remote JSON Schema reference is unsupported: {reference}"
        name = reference[len("#/$defs/") :]
        if name in refs:
            return f"recursive JSON Schema reference is unsupported: {reference}"
        if not isinstance(definitions, Mapping) or name not in definitions:
            return f"unknown $ref {reference}"
        return _validate_schema_node(
            definitions[name],
            depth=depth + 1,
            refs=refs | {name},
            definitions=definitions,
        )

    enum = value.get("enum")
    if enum is not None:
        if not isinstance(enum, list) or not enum or len(enum) > MAX_SCHEMA_ENUM:
            return "JSON enum exceeds 256 values"
        return None
    if "const" in value:
        return None
    any_of = value.get("anyOf")
    if any_of is not None:
        if not isinstance(any_of, list) or not any_of or len(any_of) > 256:
            return "anyOf exceeds 256 alternatives"
        for branch in any_of:
            reason = _validate_schema_node(
                branch,
                depth=depth + 1,
                refs=refs,
                definitions=definitions,
            )
            if reason:
                return reason
        return None

    type_name = value.get("type")
    if not isinstance(type_name, str):
        return "schema requires a supported string type"
    if type_name == "object":
        properties = value.get("properties")
        if not isinstance(properties, Mapping):
            return "object requires properties"
        if len(properties) > MAX_SCHEMA_PROPERTIES:
            return "JSON property count exceeds 1024"
        if value.get("additionalProperties") is not False:
            return "additionalProperties (must be false)"
        required = value.get("required", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            return "required must be an array of strings"
        if any(item not in properties for item in required):
            return "required property is not declared"
        for child in properties.values():
            reason = _validate_schema_node(
                child,
                depth=depth + 1,
                refs=refs,
                definitions=definitions,
            )
            if reason:
                return reason
        return None
    if type_name == "array":
        if "items" not in value:
            return "array requires items"
        return _validate_schema_node(
            value["items"],
            depth=depth + 1,
            refs=refs,
            definitions=definitions,
        )
    if type_name not in {"string", "number", "integer", "boolean", "null"}:
        return f"unsupported type {type_name}"
    return None


def _validate_response_format(
    value: Any, messages: list[Mapping[str, Any]]
) -> tuple[str, str] | None:
    if not isinstance(value, Mapping):
        return ("response_format", "response_format must be an object")
    format_type = value.get("type")
    if format_type == "text":
        if set(value) != {"type"}:
            return ("response_format", "response_format.text does not accept extra fields")
        return None
    if format_type == "json_object":
        if set(value) != {"type"}:
            return (
                "response_format",
                "response_format.json_object does not accept extra fields",
            )
        if not any("json" in str(message.get("content", "")).lower() for message in messages):
            return (
                "response_format",
                "response_format=json_object requires a message to mention JSON",
            )
        return None
    if format_type != "json_schema":
        return (
            "response_format.type",
            "response_format.type must be text, json_object, or json_schema",
        )
    if set(value) != {"type", "json_schema"} or not isinstance(value.get("json_schema"), Mapping):
        return ("response_format", "json_schema response format has an invalid envelope")
    schema = value["json_schema"]
    unknown = _unknown_fields(schema, frozenset({"name", "description", "schema", "strict"}))
    if unknown:
        return (
            "response_format.json_schema",
            f"unsupported response_format.json_schema field: {unknown}",
        )
    name = schema.get("name")
    if not isinstance(name, str) or not 1 <= len(name.encode()) <= 256:
        return (
            "response_format.json_schema.name",
            "schema name must contain 1..=256 bytes",
        )
    if "description" in schema and (
        not isinstance(schema["description"], str) or len(schema["description"].encode()) > 4_096
    ):
        return (
            "response_format.json_schema.description",
            "schema description must contain at most 4096 bytes",
        )
    if "strict" in schema and not isinstance(schema["strict"], bool):
        return ("response_format.json_schema.strict", "schema strict must be boolean")
    reason = _validate_schema_node(schema.get("schema"))
    if reason:
        return ("response_format.json_schema.schema", reason)
    if len(
        json.dumps(schema["schema"], separators=(",", ":"), ensure_ascii=False).encode()
    ) > MAX_SCHEMA_BYTES:
        return (
            "response_format.json_schema.schema",
            "schema must be at most 65536 bytes",
        )
    return None


def _validate_sampler(value: Any, *, stream: bool) -> str | None:
    if not isinstance(value, Mapping):
        return "sllm must be an object"
    unknown = _unknown_fields(value, SUPPORTED_SLLM_FIELDS)
    if unknown:
        return f"unsupported sllm field: {unknown}"
    thinking = value.get("thinking", "disabled")
    if thinking not in {"enabled", "disabled"}:
        return "sllm.thinking must be enabled or disabled"
    if "separate_reasoning" in value and not isinstance(value["separate_reasoning"], bool):
        return "sllm.separate_reasoning must be boolean"
    if value.get("separate_reasoning", False) and thinking != "enabled":
        return "separate_reasoning requires sllm.thinking=enabled"
    if "resumable" in value and not isinstance(value["resumable"], bool):
        return "sllm.resumable must be boolean"
    if value.get("resumable", False) and not stream:
        return "resumable requires stream=true"
    sampling = value.get("sampling")
    if sampling is None:
        return None
    if not isinstance(sampling, Mapping):
        return "sllm.sampling must be an object"
    unknown = _unknown_fields(sampling, SUPPORTED_SAMPLER_FIELDS)
    if unknown:
        return f"unsupported sllm.sampling field: {unknown}"
    if sampling.get("chain_version", 1) != 1:
        return "only sampler chain version 1 is supported"
    if "top_k" in sampling and (
        isinstance(sampling["top_k"], bool)
        or not isinstance(sampling["top_k"], int)
        or not 0 <= sampling["top_k"] <= MAX_SAMPLER_TOP_K
    ):
        return "top_k must be in [0,1000000]"
    for field, bounds in (("min_p", (0.0, 1.0, True)), ("typical_p", (0.0, 1.0, False)), ("repeat_penalty", (0.0, 100.0, False))):
        if field not in sampling:
            continue
        value_number = sampling[field]
        lower, upper, include_lower = bounds
        if not _is_finite_number(value_number) or not (lower < value_number <= upper if not include_lower else lower <= value_number <= upper):
            return f"{field} is outside its supported range"
    if "repeat_last_n" in sampling and (
        isinstance(sampling["repeat_last_n"], bool)
        or not isinstance(sampling["repeat_last_n"], int)
        or not 0 <= sampling["repeat_last_n"] <= MAX_SAMPLER_HISTORY
    ):
        return "repeat_last_n must be in [0,4096]"
    if "ignore_eos" in sampling and not isinstance(sampling["ignore_eos"], bool):
        return "ignore_eos must be boolean"

    dry = sampling.get("dry")
    if dry is not None:
        if not isinstance(dry, Mapping):
            return "dry must be an object"
        unknown = _unknown_fields(dry, SUPPORTED_DRY_FIELDS)
        if unknown:
            return f"unsupported dry field: {unknown}"
        for field, lower, upper in (("multiplier", 0.0, 100.0), ("base", 1.0, 4.0)):
            if field in dry and (not _is_finite_number(dry[field]) or not lower <= dry[field] <= upper):
                return f"dry.{field} is outside its supported range"
        for field in ("allowed_length", "penalty_last_n"):
            if field in dry and (isinstance(dry[field], bool) or not isinstance(dry[field], int) or not 0 <= dry[field] <= MAX_SAMPLER_HISTORY):
                return f"dry.{field} must be in [0,4096]"
        if "sequence_breakers" in dry:
            breakers = dry["sequence_breakers"]
            if not isinstance(breakers, list) or len(breakers) > MAX_SEQUENCE_BREAKERS or any(not isinstance(item, str) or not item or len(item.encode()) > 128 for item in breakers) or len(set(breakers)) != len(breakers) or sum(len(item.encode()) for item in breakers) > MAX_SEQUENCE_BREAKER_BYTES:
                return "dry.sequence_breakers exceeds its bounded unique string limits"

    xtc = sampling.get("xtc")
    if xtc is not None:
        if not isinstance(xtc, Mapping):
            return "xtc must be an object"
        unknown = _unknown_fields(xtc, SUPPORTED_XTC_FIELDS)
        if unknown:
            return f"unsupported xtc field: {unknown}"
        for field in ("probability", "threshold"):
            if field in xtc and (not _is_finite_number(xtc[field]) or not 0.0 <= xtc[field] <= 1.0):
                return f"xtc.{field} is outside its supported range"
        if "min_keep" in xtc and (isinstance(xtc["min_keep"], bool) or not isinstance(xtc["min_keep"], int) or not 1 <= xtc["min_keep"] <= MAX_SAMPLER_HISTORY):
            return "xtc.min_keep must be in [1,4096]"

    mirostat = sampling.get("mirostat")
    if mirostat is not None:
        if not isinstance(mirostat, Mapping):
            return "mirostat must be an object"
        unknown = _unknown_fields(mirostat, SUPPORTED_MIROSTAT_FIELDS)
        if unknown:
            return f"unsupported mirostat field: {unknown}"
        if mirostat.get("version", 2) not in {1, 2}:
            return "mirostat.version must be 1 or 2"
        if "tau" in mirostat and (not _is_finite_number(mirostat["tau"]) or not 0.0 < mirostat["tau"] <= 100.0):
            return "mirostat.tau is outside its supported range"
        if "eta" in mirostat and (not _is_finite_number(mirostat["eta"]) or not 0.0 < mirostat["eta"] <= 1.0):
            return "mirostat.eta is outside its supported range"
        if any(field in sampling for field in ("top_k", "min_p", "typical_p", "xtc", "dynamic_temperature")):
            return "mirostat cannot be combined with top_k, min_p, typical_p, xtc, or dynamic_temperature"

    dynamic = sampling.get("dynamic_temperature")
    if dynamic is not None:
        if not isinstance(dynamic, Mapping):
            return "dynamic_temperature must be an object"
        unknown = _unknown_fields(dynamic, SUPPORTED_DYNAMIC_TEMPERATURE_FIELDS)
        if unknown:
            return f"unsupported dynamic_temperature field: {unknown}"
        if "range" in dynamic and (not _is_finite_number(dynamic["range"]) or not 0.0 <= dynamic["range"] <= 10.0):
            return "dynamic_temperature.range is outside its supported range"
        if "exponent" in dynamic and (not _is_finite_number(dynamic["exponent"]) or not 0.0 < dynamic["exponent"] <= 10.0):
            return "dynamic_temperature.exponent is outside its supported range"
    return None


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
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            return _reject("invalid_value", "messages", "each message must be an object")
        if set(message) - SUPPORTED_MESSAGE_FIELDS:
            return _reject("unsupported_parameter", "messages", "message field is unsupported")
        role = message.get("role")
        if role not in SUPPORTED_ROLES:
            return _reject(
                "unsupported_parameter",
                f"messages[{index}].role",
                "message role is unsupported",
            )
        if not isinstance(message.get("content"), str):
            return _reject(
                "unsupported_parameter",
                f"messages[{index}].content",
                "only text content is supported",
            )
        if "reasoning_content" in message:
            if role != "assistant" or not isinstance(message["reasoning_content"], str):
                return _reject(
                    "unsupported_parameter",
                    f"messages[{index}].reasoning_content",
                    "reasoning_content is supported only on assistant messages",
                )

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
        isinstance(payload["n"], bool) or not isinstance(payload["n"], int)
    ):
        return _reject("invalid_value", "n", "n must be an integer in [1, 8]")
    if "n" in payload and not 1 <= payload["n"] <= 8:
        return _reject("invalid_value", "n", "n must be an integer in [1, 8]")

    if "seed" in payload and (
        isinstance(payload["seed"], bool)
        or not isinstance(payload["seed"], int)
        or not -(1 << 63) <= payload["seed"] <= (1 << 63) - 1
    ):
        return _reject("invalid_value", "seed", "seed must be a signed 64-bit integer")

    if "logit_bias" in payload:
        bias = payload["logit_bias"]
        if not isinstance(bias, Mapping):
            return _reject("invalid_value", "logit_bias", "logit_bias must be an object")
        if len(bias) > MAX_LOGIT_BIAS_ENTRIES:
            return _reject("invalid_value", "logit_bias", "logit_bias must contain at most 4096 entries")
        for token_id, value in bias.items():
            if not isinstance(token_id, str):
                return _reject("invalid_value", "logit_bias", "token IDs must be strings")
            try:
                parsed = int(token_id, 10)
            except ValueError:
                parsed = -1
            if parsed < 0 or parsed > (1 << 32) - 1 or not token_id.isascii() or not token_id.isdigit():
                return _reject("invalid_value", f"logit_bias.{token_id}", "token ID must be an unsigned 32-bit integer")
            if not _is_finite_number(value) or not -100.0 <= value <= 100.0:
                return _reject("invalid_value", f"logit_bias.{token_id}", "logit bias must be finite and in [-100, 100]")

    if "logprobs" in payload and not isinstance(payload["logprobs"], bool):
        return _reject("invalid_value", "logprobs", "logprobs must be boolean")
    if "top_logprobs" in payload:
        top_logprobs = payload["top_logprobs"]
        if (
            isinstance(top_logprobs, bool)
            or not isinstance(top_logprobs, int)
            or not 0 <= top_logprobs <= 20
        ):
            return _reject("invalid_value", "top_logprobs", "top_logprobs must be in [0, 20]")
        if payload.get("logprobs") is not True:
            return _reject("invalid_value", "top_logprobs", "top_logprobs requires logprobs=true")

    if "response_format" in payload:
        response_format_error = _validate_response_format(payload["response_format"], messages)
        if response_format_error:
            param, reason = response_format_error
            return _reject("invalid_value", param, reason)

    if "sllm" in payload:
        reason = _validate_sampler(payload["sllm"], stream=payload.get("stream", False))
        if reason:
            return _reject("invalid_value", "sllm", reason)

    return ValidationResult(accepted=True)
