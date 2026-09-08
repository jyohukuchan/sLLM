#!/usr/bin/env python3
"""Probe a running localhost sLLM API for the Phase 81 fixed profile.

This is an opt-in, live HTTP probe.  It never starts a server or a GPU
process.  The probe keeps unrestricted text and grammar-mask timing in
separate records and records output lengths beside timings; it does not turn
different output lengths into a performance conclusion.

Example:

    scripts/dev/phase81_api_probe.py \
      --base-url http://127.0.0.1:8080 --model qwen \
      --output /tmp/sllm-phase81-api-probe.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener


SCHEMA = "sllm-phase81-api-probe-v1"
DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_OUTPUT = "/tmp/sllm-phase81-api-probe.json"
DEFAULT_SEED = 81095
DEFAULT_MAX_COMPLETION_TOKENS = 8
FORCED_STOP = "<PHASE81_STOP>"
# Existing 256x256 PNG fixture from the API image contract test.
IMAGE_PNG_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAIAAADTED8xAAAA1UlEQVR42u3BMQEAAADCoPVP7WULoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAGwEtAAGey8LtAAAAAElFTkSuQmCC"


class ProbeFailure(RuntimeError):
    """One expected API assertion failed."""


class ProbeResponse:
    def __init__(
        self,
        call_index: int,
        status: int,
        body: bytes,
        value: Any,
        elapsed_ms: float,
        headers: dict[str, str],
    ) -> None:
        self.call_index = call_index
        self.status = status
        self.body = body
        self.value = value
        self.elapsed_ms = elapsed_ms
        self.headers = headers


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def infer_top_k(model: str, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    name = model.lower()
    if "ministral" in name:
        return 0
    if "gemma" in name:
        return 64
    if "qwen" in name:
        return 20
    raise ValueError(
        "--top-k is required for an unknown model alias; do not infer a profile"
    )


def json_body(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "lookup",
        "description": "Read-only lookup fixture.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "enum": ["Tokyo"]}},
            "required": ["query"],
            "additionalProperties": False,
        },
    }


class ApiProbe:
    def __init__(
        self,
        base_url: str,
        model: str,
        seed: int,
        timeout: float,
        api_key: str | None,
        top_k: int,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("--base-url must be an absolute http(s) URL")
        self.base_url = base_url.rstrip("/")
        self._parsed = parsed
        self.model = model
        self.seed = seed
        self.timeout = timeout
        self.api_key = api_key
        self.top_k = top_k
        self.calls: list[dict[str, Any]] = []
        self.cases: list[dict[str, Any]] = []
        self._assertions: list[dict[str, Any]] | None = None
        self._opener = build_opener()

    def _headers(self, content_type: bool = False) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if content_type:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _record_call(
        self,
        method: str,
        path: str,
        body: bytes,
        status: int,
        response_body: bytes,
        value: Any,
        elapsed_ms: float,
        headers: dict[str, str],
    ) -> int:
        record: dict[str, Any] = {
            "method": method,
            "path": path,
            "request_bytes": len(body),
            "status": status,
            "elapsed_ms": round(elapsed_ms, 3),
            "response_bytes": len(response_body),
            "response_sha256": sha256(response_body),
            "content_type": headers.get("content-type", ""),
        }
        transport_error = headers.get("error")
        if transport_error:
            record["transport_error"] = transport_error
        if value is not None:
            if isinstance(value, dict):
                record["response_object"] = value.get("object", value.get("type"))
                error = value.get("error")
                if isinstance(error, dict):
                    record["error"] = {
                        key: error[key]
                        for key in ("message", "type", "param", "code")
                        if key in error
                    }
                elif isinstance(value.get("message"), str):
                    record["message"] = value["message"]
                usage = value.get("usage")
                if isinstance(usage, dict):
                    record["usage"] = {
                        key: usage.get(key)
                        for key in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens", "total_tokens")
                        if key in usage
                    }
                if isinstance(value.get("choices"), list):
                    choices = value["choices"]
                    record["choice_count"] = len(choices)
                    if choices and isinstance(choices[0], dict):
                        message = choices[0].get("message")
                        if isinstance(message, dict):
                            text = message.get("content")
                            if isinstance(text, str):
                                record["output_chars"] = len(text)
                            record["reasoning_chars"] = len(message.get("reasoning_content", ""))
                        record["finish_reason"] = choices[0].get("finish_reason")
                if isinstance(value.get("output_text"), str):
                    record["output_chars"] = len(value["output_text"])
                if isinstance(value.get("output"), list):
                    record["output_item_count"] = len(value["output"])
            elif isinstance(value, list):
                record["response_array_length"] = len(value)
        if response_body and value is None:
            record["body_preview"] = response_body[:256].decode("utf-8", "replace")
        self.calls.append(record)
        return len(self.calls) - 1

    def post(self, path: str, payload: dict[str, Any]) -> ProbeResponse:
        body = json_body(payload)
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=self._headers(content_type=True),
            method="POST",
        )
        started = time.perf_counter()
        status = 0
        response_body = b""
        headers: dict[str, str] = {}
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                status = response.status
                headers = {key.lower(): value for key, value in response.headers.items()}
                response_body = response.read()
        except HTTPError as error:
            status = error.code
            headers = {key.lower(): value for key, value in error.headers.items()}
            response_body = error.read()
        except (URLError, TimeoutError, OSError) as error:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            index = self._record_call(
                "POST", path, body, 0, b"", None, elapsed_ms, {"error": str(error)}
            )
            raise ProbeFailure(f"POST {path} transport error: {error}") from error
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        try:
            value = json.loads(response_body.decode("utf-8")) if response_body else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            value = None
        index = self._record_call(
            "POST", path, body, status, response_body, value, elapsed_ms, headers
        )
        return ProbeResponse(index, status, response_body, value, elapsed_ms, headers)

    def get(self, path: str) -> ProbeResponse:
        request = Request(
            f"{self.base_url}{path}", headers=self._headers(), method="GET"
        )
        started = time.perf_counter()
        status = 0
        response_body = b""
        headers: dict[str, str] = {}
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                status = response.status
                headers = {key.lower(): value for key, value in response.headers.items()}
                response_body = response.read()
        except HTTPError as error:
            status = error.code
            headers = {key.lower(): value for key, value in error.headers.items()}
            response_body = error.read()
        except (URLError, TimeoutError, OSError) as error:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._record_call("GET", path, b"", 0, b"", None, elapsed_ms, {"error": str(error)})
            raise ProbeFailure(f"GET {path} transport error: {error}") from error
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        try:
            value = json.loads(response_body.decode("utf-8")) if response_body else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            value = None
        index = self._record_call(
            "GET", path, b"", status, response_body, value, elapsed_ms, headers
        )
        return ProbeResponse(index, status, response_body, value, elapsed_ms, headers)

    def stream_disconnect(self, path: str, payload: dict[str, Any]) -> ProbeResponse:
        """Disconnect after a generated token, rather than the initial role event."""
        if self._parsed.scheme != "http":
            raise ProbeFailure("stream disconnect probe currently requires an http localhost URL")
        host = self._parsed.hostname
        if not host:
            raise ProbeFailure("stream disconnect URL has no host")
        port = self._parsed.port or 80
        body = json_body(payload)
        headers = self._headers(content_type=True)
        headers.update({"Connection": "close", "Content-Length": str(len(body))})
        started = time.perf_counter()
        status = 0
        received = b""
        content_type = ""
        observed_token_content = False
        connection = None
        try:
            connection = http.client.HTTPConnection(host, port, timeout=self.timeout)
            connection.request("POST", path, body, headers)
            response = connection.getresponse()
            status = response.status
            content_type = response.getheader("Content-Type", "")
            while len(received) < 65_536:
                line = response.readline(16_384)
                if not line:
                    break
                received += line
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    break
                try:
                    event = json.loads(data)
                except (ValueError, UnicodeDecodeError):
                    continue
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content") or delta.get("reasoning_content"):
                        observed_token_content = True
                if observed_token_content:
                    break
        except (socket.timeout, OSError, http.client.HTTPException) as error:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            index = self._record_call(
                "POST_DISCONNECT",
                path,
                body,
                status,
                received,
                None,
                elapsed_ms,
                {"content-type": content_type, "error": str(error)},
            )
            raise ProbeFailure(f"stream disconnect transport error: {error}") from error
        finally:
            if connection is not None:
                connection.close()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        index = self._record_call(
            "POST_DISCONNECT",
            path,
            body,
            status,
            received,
            None,
            elapsed_ms,
            {"content-type": content_type},
        )
        self.calls[index]["disconnect_after_bytes"] = len(received)
        self.calls[index]["stream_content_type"] = content_type
        self.calls[index]["observed_token_content"] = observed_token_content
        return ProbeResponse(index, status, received, None, elapsed_ms, {"content-type": content_type})

    def assertion(self, condition: bool, message: str) -> None:
        if self._assertions is None:
            raise RuntimeError("assertion outside case")
        self._assertions.append({"check": message, "ok": bool(condition)})
        if not condition:
            raise ProbeFailure(message)

    def case(self, name: str, function: Callable[[], None]) -> None:
        started = time.perf_counter()
        assertions: list[dict[str, Any]] = []
        self._assertions = assertions
        case: dict[str, Any] = {"name": name, "state": "PASS", "assertions": assertions}
        first_call = len(self.calls)
        try:
            function()
        except Exception as error:  # Evidence must survive one failed probe.
            case["state"] = "FAIL"
            case["error"] = str(error)
        finally:
            self._assertions = None
        case["call_indices"] = list(range(first_call, len(self.calls)))
        case["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        self.cases.append(case)


def require_response_object(probe: ApiProbe, response: ProbeResponse, object_name: str) -> dict[str, Any]:
    probe.assertion(response.status == 200, f"HTTP status is 200 (got {response.status})")
    probe.assertion(isinstance(response.value, dict), "response is a JSON object")
    value = response.value
    probe.assertion(value.get("object") == object_name, f"response.object is {object_name}")
    return value


def require_chat(probe: ApiProbe, response: ProbeResponse) -> dict[str, Any]:
    value = require_response_object(probe, response, "chat.completion")
    choices = value.get("choices")
    probe.assertion(isinstance(choices, list) and len(choices) >= 1, "chat response has a choice")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else None
    probe.assertion(isinstance(message, dict), "chat choice has a message")
    probe.assertion(message.get("role") == "assistant", "chat message role is assistant")
    probe.assertion(isinstance(message.get("content"), str), "chat message content is text")
    probe.assertion(choice.get("finish_reason") in {"stop", "length"}, "chat finish reason is bounded")
    usage = value.get("usage")
    probe.assertion(isinstance(usage, dict), "chat response has usage")
    return value


def require_responses(probe: ApiProbe, response: ProbeResponse) -> dict[str, Any]:
    value = require_response_object(probe, response, "response")
    probe.assertion(isinstance(value.get("output"), list), "Responses output is an array")
    probe.assertion(isinstance(value.get("output_text"), str), "Responses output_text is text")
    probe.assertion(isinstance(value.get("usage"), dict), "Responses response has usage")
    return value


def require_error(probe: ApiProbe, response: ProbeResponse, parameter: str) -> None:
    probe.assertion(response.status == 400, f"unsupported setting returns HTTP 400 (got {response.status})")
    value = response.value
    probe.assertion(isinstance(value, dict), "error response is JSON")
    error = value.get("error") if isinstance(value, dict) else None
    probe.assertion(isinstance(error, dict), "error envelope has error object")
    probe.assertion(error.get("type") == "invalid_request_error", "error type is invalid_request_error")
    probe.assertion(error.get("param") == parameter, f"error parameter is {parameter}")
    probe.assertion(
        error.get("code") in {"invalid_value", "unsupported_parameter"},
        "error code is a fixed-profile rejection code",
    )


def chat_payload(model: str, seed: int, max_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with one short word: ok."}],
        "max_completion_tokens": max_tokens,
        "seed": seed,
    }


def explicit_fixed_payload(base: dict[str, Any], top_k: int) -> dict[str, Any]:
    payload = copy.deepcopy(base)
    payload.update(
        {
            "temperature": 1.0,
            "top_p": 0.95,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "logprobs": False,
            "top_logprobs": 0,
            "sllm": {
                "sampling": {
                    "chain_version": 1,
                    "top_k": top_k,
                    "min_p": 0.0,
                    "typical_p": 1.0,
                    "repeat_penalty": 1.0,
                    "repeat_last_n": 0,
                    "ignore_eos": False,
                }
            },
        }
    )
    return payload


def chat_text(value: dict[str, Any]) -> tuple[str, Any, Any]:
    choice = value["choices"][0]
    message = choice["message"]
    return message["content"], message.get("reasoning_content"), choice.get("finish_reason")


def run(probe: ApiProbe, image_case: bool = False, tools: str = "supported") -> None:
    def models() -> None:
        response = probe.get("/v1/models")
        value = require_response_object(probe, response, "list")
        models_value = value.get("data")
        probe.assertion(isinstance(models_value, list), "models.data is an array")
        aliases = {entry.get("id") for entry in models_value if isinstance(entry, dict)}
        probe.assertion(probe.model in aliases, f"requested model {probe.model!r} is listed")

    probe.case("models_schema", models)

    base = chat_payload(probe.model, probe.seed)

    def fixed_replay() -> None:
        first = require_chat(probe, probe.post("/v1/chat/completions", base))
        explicit = require_chat(
            probe, probe.post("/v1/chat/completions", explicit_fixed_payload(base, probe.top_k))
        )
        probe.assertion(
            chat_text(first) == chat_text(explicit),
            "omitted fixed profile and explicit equal profile replay identically with the same seed",
        )

    probe.case("fixed_profile_default_and_explicit_replay", fixed_replay)

    invalid_cases: list[tuple[str, str, Callable[[dict[str, Any]], None]]] = [
        ("temperature", "temperature", lambda payload: payload.update({"temperature": 0.7})),
        ("top_p", "top_p", lambda payload: payload.update({"top_p": 0.9})),
        (
            "presence_penalty",
            "presence_penalty",
            lambda payload: payload.update({"presence_penalty": 0.1}),
        ),
        (
            "frequency_penalty",
            "frequency_penalty",
            lambda payload: payload.update({"frequency_penalty": 0.1}),
        ),
        ("logprobs", "logprobs", lambda payload: payload.update({"logprobs": True})),
        (
            "top_k",
            "sllm.sampling.top_k",
            lambda payload: payload.update({"sllm": {"sampling": {"top_k": 17}}}),
        ),
    ]
    for name, parameter, mutate in invalid_cases:
        def rejected(name: str = name, parameter: str = parameter, mutate: Callable[[dict[str, Any]], None] = mutate) -> None:
            payload = copy.deepcopy(base)
            mutate(payload)
            require_error(probe, probe.post("/v1/chat/completions", payload), parameter)

        probe.case(f"reject_{name}", rejected)

    def json_schema_mask(include_image: bool = False) -> None:
        payload = chat_payload(probe.model, probe.seed, max_tokens=16)
        payload["messages"] = [
            {
                "role": "user",
                "content": 'Return exactly JSON with answer set to "ok". Include the word JSON.',
            }
        ]
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "phase81_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string", "enum": ["ok"]}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        }
        if include_image:
            payload["messages"][0]["content"] = [
                {"type": "text", "text": 'Inspect this image and return JSON with answer "ok".'},
                {"type": "image_url", "image_url": {"url": IMAGE_PNG_URL}},
            ]
        value = require_chat(probe, probe.post("/v1/chat/completions", payload))
        content, _, _ = chat_text(value)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as error:
            raise ProbeFailure(f"JSON schema output is not JSON: {error}") from error
        probe.assertion(isinstance(parsed, dict), "JSON schema output is an object")
        probe.assertion(set(parsed) == {"answer"}, "JSON schema output has exactly the declared key")
        probe.assertion(parsed.get("answer") == "ok", "JSON schema answer satisfies the declared enum")

        if include_image:
            probe.assertion(value["usage"]["prompt_tokens"] > 32, "image request accounts for visual prompt tokens")

    probe.case("json_schema_mask", json_schema_mask)
    if image_case:
        probe.case("image_json_schema_mask", lambda: json_schema_mask(True))

    def timing_unconstrained_text() -> None:
        require_chat(probe, probe.post("/v1/chat/completions", base))

    probe.case("timing_unconstrained_text", timing_unconstrained_text)

    def tool_call() -> None:
        payload = {
            "model": probe.model,
            "input": "Call lookup with query Tokyo. Do not answer with prose.",
            "max_output_tokens": 128,
            "temperature": 1.0,
            "top_p": 0.95,
            "store": False,
            "tools": [tool_definition()],
            "tool_choice": {"type": "function", "name": "lookup"},
            "parallel_tool_calls": False,
        }
        response = probe.post("/v1/responses", payload)
        if tools == "unsupported":
            require_error(probe, response, "tools")
            probe.assertion(response.value["error"]["code"] == "unsupported_parameter", "profile explicitly rejects unsupported tools")
            return
        value = require_responses(probe, response)
        calls = [item for item in value["output"] if isinstance(item, dict) and item.get("type") == "function_call"]
        probe.assertion(len(calls) == 1, "Responses tool request returns exactly one function call")
        call = calls[0]
        probe.assertion(call.get("name") == "lookup", "function call uses the declared tool name")
        arguments = json.loads(call.get("arguments", ""))
        probe.assertion(isinstance(arguments, dict), "function call arguments are a JSON object")
        probe.assertion(arguments == {"query": "Tokyo"}, "function call preserves the constrained argument")

    probe.case("reject_unsupported_tools" if tools == "unsupported" else "responses_tool_call", tool_call)

    def reasoning_controls() -> None:
        payload = chat_payload(probe.model, probe.seed, max_tokens=2)
        payload["messages"] = [{"role": "user", "content": "Answer briefly: ok."}]
        payload["sllm"] = {
            "thinking": "enabled",
            "separate_reasoning": True,
            "max_reasoning_tokens": 1,
        }
        value = require_chat(probe, probe.post("/v1/chat/completions", payload))
        message = value["choices"][0]["message"]
        probe.assertion("reasoning_content" in message, "separate reasoning control is reflected in the response schema")

    probe.case("reasoning_controls", reasoning_controls)

    def stop_control() -> None:
        payload = chat_payload(probe.model, probe.seed, max_tokens=128)
        payload["messages"] = [
            {
                "role": "user",
                "content": "Return exactly the requested JSON. Include the word JSON.",
            }
        ]
        payload["stop"] = [FORCED_STOP]
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "phase81_stop",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "enum": [FORCED_STOP]}
                    },
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        }
        value = require_chat(probe, probe.post("/v1/chat/completions", payload))
        content, _, finish_reason = chat_text(value)
        probe.assertion(finish_reason == "stop", "schema-constrained stop request finishes with stop")
        probe.assertion(FORCED_STOP not in content, "stop sequence is excluded from returned content")
        probe.assertion(bool(content), "stop interaction returns a bounded content fragment")

    probe.case("stop_control", stop_control)

    def stream_disconnect_recovery() -> None:
        payload = chat_payload(probe.model, probe.seed, max_tokens=64)
        payload["messages"] = [{"role": "user", "content": "Write a complete Python LRU cache class and five unit tests. Include all code."}]
        payload["stream"] = True
        response = probe.stream_disconnect("/v1/chat/completions", payload)
        probe.assertion(response.status == 200, f"stream headers return HTTP 200 (got {response.status})")
        probe.assertion(
            response.headers.get("content-type", "").startswith("text/event-stream"),
            "stream response uses text/event-stream",
        )
        probe.assertion(probe.calls[response.call_index]["observed_token_content"], "stream yielded generated token content before disconnect")
        recovery = require_chat(probe, probe.post("/v1/chat/completions", base))
        probe.assertion(isinstance(chat_text(recovery)[0], str), "request recovers after stream disconnect")

    probe.case("stream_disconnect_then_recovery", stream_disconnect_recovery)


def build_evidence(probe: ApiProbe, started_at: str) -> dict[str, Any]:
    failures = [case for case in probe.cases if case["state"] != "PASS"]
    return {
        "schema": SCHEMA,
        "state": "PASS" if not failures else "FAIL",
        "started_at": started_at,
        "base_url": probe.base_url,
        "model": probe.model,
        "seed": probe.seed,
        "fixed_profile": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": probe.top_k,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "logprobs": False,
            "top_logprobs": 0,
        },
        "timing_scope": {
            "unconstrained_case": "timing_unconstrained_text",
            "masked_case": "json_schema_mask",
            "comparison": "descriptive records only; output length and mask work are not converted into a speed claim",
        },
        "cases": probe.cases,
        "calls": probe.calls,
        "failed_cases": [case["name"] for case in failures],
        "notes": [
            "localhost HTTP is assumed; the script does not start a server or a GPU process",
            "tool and reasoning cases are actual API assertions and therefore fail for a model that does not advertise those capabilities",
            "stream disconnect recovery is checked after a generated content/reasoning token, excluding the role-only event",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="candidate localhost API base URL")
    parser.add_argument("--model", required=True, help="served model alias from GET /v1/models")
    parser.add_argument(
        "--top-k",
        type=int,
        choices=(0, 20, 64),
        default=None,
        help="fixed model profile top_k; inferred only for qwen/gemma/ministral aliases",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--api-key", default=os.environ.get("SLLM_API_KEY"))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--tools", choices=("supported", "unsupported"), default="supported", help="expected tool capability of the selected server profile")
    parser.add_argument("--image-case", action="store_true", help="verify the existing Qwen image route with a bounded JSON mask")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        top_k = infer_top_k(args.model, args.top_k)
        probe = ApiProbe(args.base_url, args.model, args.seed, args.timeout, args.api_key, top_k)
    except (TypeError, ValueError) as error:
        print(f"phase81_api_probe: argument error: {error}", file=sys.stderr)
        return 2
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        run(probe, image_case=args.image_case, tools=args.tools)
    except Exception as error:
        probe.cases.append(
            {
                "name": "harness",
                "state": "FAIL",
                "assertions": [],
                "call_indices": [],
                "elapsed_ms": 0.0,
                "error": str(error),
            }
        )
    evidence = build_evidence(probe, started_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": evidence["state"], "output": str(args.output), "failed_cases": evidence["failed_cases"]}))
    return 0 if evidence["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
