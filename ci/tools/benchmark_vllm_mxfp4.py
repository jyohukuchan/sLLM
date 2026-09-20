#!/usr/bin/env python3
"""Benchmark a vLLM-compatible ``/v1/completions`` endpoint.

The benchmark uses the reviewed Phase 83 coding prompt as pre-tokenized input
and keeps the request shape close to the existing 8,192/128 measurements:
one warmup, three measured requests, temperature 1, top-p .95, top-k 20, and
seed 123.  The prompt is sent as token IDs, so a second tokenizer or chat
template cannot silently change the input length.

The endpoint is streamed with Server-Sent Events.  Each row records request
wall time, time to the first generated token (TTFT), decode time, TPOT, and
the final ``usage`` object.  A row is accepted only when the server reports
exactly the expected prompt and completion token counts.  ``cache_salt`` is
unique for every request when the endpoint accepts it; that prevents prefix
blocks from being reused by servers that implement the vLLM cache-salt
contract.  Older endpoints may reject the field.  In that case the runner
retries without it and records that prefill may have been cached.

Only Python's standard library is used.  The report is JSON and does not
contain the prompt token payload or generated text, keeping large benchmark
inputs out of Git and making the output suitable for a small evidence file.

Example::

    python ci/tools/benchmark_vllm_mxfp4.py \
      --url http://127.0.0.1:8000 \
      --model GGZ14/vllm-mxfp4 \
      --prompt-token-ids /tmp/phase83-token-ids.json \
      --output /tmp/vllm-mxfp4-benchmark.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import secrets
import statistics
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, BinaryIO


SCHEMA_VERSION = "vllm-mxfp4-benchmark-v1"
DEFAULT_PROMPT_TOKENS = 8192
DEFAULT_OUTPUT_TOKENS = 128
DEFAULT_WARMUPS = 1
DEFAULT_MEASURED = 3
SEED = 123
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20


class RequestFailure(RuntimeError):
    """An HTTP request failed before a valid SSE response was received."""

    def __init__(self, status: int | None, body: str):
        self.status = status
        self.body = body
        message = f"HTTP {status}: {body}" if status is not None else body
        super().__init__(message)


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def token_ids_sha256(token_ids: list[int]) -> str:
    payload = b"".join(struct.pack("<i", token) for token in token_ids)
    return sha256_bytes(payload)


def read_prompt_token_ids(path: Path, expected_count: int) -> list[int]:
    """Read a token list or a small metadata object containing one."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read prompt token IDs from {path}: {error}") from error

    if isinstance(value, list):
        token_ids = value
    elif isinstance(value, dict):
        token_ids = None
        for key in ("token_ids", "prompt_token_ids", "tokens"):
            if key in value:
                token_ids = value[key]
                break
        if token_ids is None:
            raise ValueError(
                f"{path} must be a JSON list or contain token_ids/prompt_token_ids"
            )
    else:
        raise ValueError(f"{path} must contain a JSON list or object")

    if not isinstance(token_ids, list):
        raise ValueError(f"prompt token IDs in {path} must be a list")
    if len(token_ids) != expected_count:
        raise ValueError(
            f"prompt token count is {len(token_ids)}, expected {expected_count}"
        )
    for index, token in enumerate(token_ids):
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError(f"prompt token {index} is not a non-negative integer")
        # The JSON/OpenAI request uses an integer token ID.  Keep the bound
        # explicit so struct.pack and a malformed huge input cannot disagree.
        if token > 2**31 - 1:
            raise ValueError(f"prompt token {index} does not fit the locked ID format")
    return [int(token) for token in token_ids]


def completion_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("--url must be an http(s) URL with a host")
    if parsed.query or parsed.fragment:
        raise ValueError("--url must not contain a query or fragment")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1/completions"):
        endpoint_path = path
    elif path.endswith("/v1"):
        endpoint_path = path + "/completions"
    else:
        endpoint_path = path + "/v1/completions"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, endpoint_path, "", ""))


def finite_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def json_error_body(error: urllib.error.HTTPError) -> str:
    try:
        body = error.read(16 * 1024)
    except OSError:
        body = b""
    return body.decode("utf-8", errors="replace").strip() or error.reason


def likely_unsupported_field(body: str, features: dict[str, bool]) -> str | None:
    """Find a request extension that a compatible server explicitly rejected."""

    text = body.lower()
    markers = ("extra", "unknown", "unexpected", "unsupported", "invalid", "forbid")
    for field in ("skip_reading_prefix_cache", "return_token_ids", "cache_salt", "stream_options"):
        if features.get(field, False) and field in text and any(marker in text for marker in markers):
            return field
    return None


def build_request(
    *,
    model: str,
    token_ids: list[int],
    output_tokens: int,
    cache_salt: str | None,
    features: dict[str, bool],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "prompt": token_ids,
        "max_tokens": output_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "seed": SEED,
        "ignore_eos": True,
        "n": 1,
        "stream": True,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "add_special_tokens": False,
        "skip_special_tokens": True,
    }
    if features.get("stream_options", False):
        body["stream_options"] = {"include_usage": True}
    if features.get("return_token_ids", False):
        body["return_token_ids"] = True
    if features.get("cache_salt", False) and cache_salt is not None:
        body["cache_salt"] = cache_salt
    # Some vLLM forks expose the engine-level switch as an extra request field.
    # It is harmless on OpenAIBaseModel implementations that retain unknown
    # fields, while cache_salt remains the portable cache-isolation mechanism.
    if features.get("skip_reading_prefix_cache", False):
        body["skip_reading_prefix_cache"] = True
    return body


def _parse_sse_event(data_lines: list[str]) -> dict[str, Any] | str | None:
    if not data_lines:
        return None
    payload = "\n".join(data_lines).strip()
    if payload == "[DONE]":
        return "[DONE]"
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RequestFailure(None, f"invalid SSE JSON: {error}: {payload[:200]}") from error
    if not isinstance(value, dict):
        raise RequestFailure(None, "SSE event JSON must be an object")
    if isinstance(value.get("error"), dict):
        raise RequestFailure(None, json.dumps(value["error"], ensure_ascii=False))
    return value


def consume_sse(
    response: BinaryIO,
    *,
    request_started: float,
    response_started: float,
) -> dict[str, Any]:
    """Consume one completion stream and return timing/usage metadata."""

    data_lines: list[str] = []
    done_marker = False
    first_token_time: float | None = None
    first_candidate_time: float | None = None
    first_token_detection: str | None = None
    first_event_token_count: int | None = None
    observed_output_tokens = 0
    output_text: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reasons: list[Any] = []

    def handle_event(event: dict[str, Any] | str | None, now: float) -> None:
        nonlocal done_marker, first_token_time, first_candidate_time
        nonlocal first_token_detection, first_event_token_count
        nonlocal observed_output_tokens, usage
        if event is None:
            return
        if event == "[DONE]":
            done_marker = True
            return
        assert isinstance(event, dict)
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        choices = event.get("choices")
        if not isinstance(choices, list):
            return
        event_token_count = 0
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                finish_reasons.append(finish_reason)
            token_ids = choice.get("token_ids")
            text = choice.get("text")
            token_count = 0
            if isinstance(token_ids, list):
                token_count = len(token_ids)
            elif isinstance(text, str) and text:
                # Completion streaming normally emits one token per choice.
                # The final usage object remains the authoritative count.
                token_count = 1
            observed_output_tokens += token_count
            event_token_count += token_count
            if isinstance(text, str):
                output_text.append(text)
        if event_token_count > 0:
            if first_event_token_count is None:
                first_event_token_count = event_token_count
            if first_token_time is None:
                first_token_time = now
                first_token_detection = (
                    "token_ids" if any(
                        isinstance(choice, dict) and isinstance(choice.get("token_ids"), list)
                        and len(choice["token_ids"]) > 0 for choice in choices
                    ) else "text"
                )
        elif first_candidate_time is None and any(
            isinstance(choice, dict) and choice.get("finish_reason") is None for choice in choices
        ):
            # Keep a fallback only for an empty, non-terminal initial event.
            # An empty terminal event must not become the TTFT origin.
            first_candidate_time = now

    while True:
        try:
            line = response.readline()
        except OSError as error:
            raise RequestFailure(None, f"SSE read failed: {error}") from error
        if not line:
            break
        decoded = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if decoded == "":
            event = _parse_sse_event(data_lines)
            data_lines = []
            handle_event(event, time.perf_counter())
            if done_marker:
                break
        elif decoded.startswith("data:"):
            data_lines.append(decoded[5:].lstrip())
        # event/id/retry fields are intentionally ignored; this endpoint's
        # contract uses data-only JSON events.

    if data_lines and not done_marker:
        handle_event(_parse_sse_event(data_lines), time.perf_counter())
    if not done_marker:
        raise RequestFailure(None, "SSE stream ended without data: [DONE]")
    if first_token_time is None and first_candidate_time is not None:
        # A token can decode to an empty string.  Preserve the event arrival
        # time while making the ambiguity visible in the report.
        first_token_time = first_candidate_time
        first_token_detection = "empty-choice-fallback"
    if first_token_time is None:
        raise RequestFailure(None, "SSE stream contained no generated token event")
    if usage is None:
        raise RequestFailure(None, "stream did not provide final usage; exact count is unverified")

    ended = time.perf_counter()
    return {
        "response_started": response_started,
        "ended": ended,
        "wall_seconds": ended - request_started,
        "ttft_seconds": first_token_time - request_started,
        "decode_seconds": ended - first_token_time,
        "first_token_detection": first_token_detection,
        "first_event_token_count": first_event_token_count,
        "grouped_stream_timing": bool(first_event_token_count and first_event_token_count > 1),
        "finish_reasons": finish_reasons,
        "observed_output_tokens": observed_output_tokens,
        "usage": usage,
        "output_text_sha256": sha256_bytes("".join(output_text).encode("utf-8")),
    }


def request_once(
    *,
    endpoint: str,
    api_key: str | None,
    body: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
    started = time.perf_counter()
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        raise RequestFailure(error.code, json_error_body(error)) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RequestFailure(None, f"HTTP request failed: {error}") from error
    with response:
        response_started = time.perf_counter()
        status = getattr(response, "status", response.getcode())
        if status != 200:
            body_text = response.read(16 * 1024).decode("utf-8", errors="replace")
            raise RequestFailure(status, body_text.strip() or "empty response")
        return request_result_with_status(
            consume_sse(response, request_started=started, response_started=response_started),
            int(status),
        )


def request_result_with_status(result: dict[str, Any], status: int) -> dict[str, Any]:
    result["http_status"] = status
    return result


def run_request(
    *,
    endpoint: str,
    api_key: str | None,
    model: str,
    token_ids: list[int],
    output_tokens: int,
    timeout: float,
    features: dict[str, bool],
    request_number: int,
) -> dict[str, Any]:
    """Run one request, learning optional-field support from HTTP 400 errors."""

    attempts: list[dict[str, Any]] = []
    cache_salt = secrets.token_urlsafe(32) if features.get("cache_salt", False) else None
    while True:
        body = build_request(
            model=model,
            token_ids=token_ids,
            output_tokens=output_tokens,
            cache_salt=cache_salt,
            features=features,
        )
        try:
            result = request_once(
                endpoint=endpoint,
                api_key=api_key,
                body=body,
                timeout=timeout,
            )
        except RequestFailure as error:
            field = likely_unsupported_field(error.body, features) if error.status == 400 else None
            attempts.append({"status": error.status, "error": error.body[:512], "removed_field": field})
            if field is None:
                raise RequestFailure(
                    error.status,
                    f"request {request_number} failed after {len(attempts)} attempt(s): {error.body}",
                ) from error
            features[field] = False
            continue
        result["request_number"] = request_number
        result["optional_features"] = dict(features)
        result["attempts"] = attempts
        result["cache_salt_sha256"] = (
            sha256_bytes(cache_salt.encode()) if cache_salt and features.get("cache_salt", False) else None
        )
        usage_details = result.get("usage", {}).get("prompt_tokens_details")
        if isinstance(usage_details, dict) and isinstance(usage_details.get("cached_tokens"), int):
            result["cached_prompt_tokens"] = usage_details["cached_tokens"]
        else:
            result["cached_prompt_tokens"] = None
        return result


def validate_result(result: dict[str, Any], expected_prompt: int, expected_output: int) -> None:
    usage = result.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("response usage is missing")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"response usage.{key} is not an integer")
    if usage["prompt_tokens"] != expected_prompt:
        raise ValueError(
            f"server reported prompt_tokens={usage['prompt_tokens']}, expected {expected_prompt}"
        )
    if usage["completion_tokens"] != expected_output:
        raise ValueError(
            f"server reported completion_tokens={usage['completion_tokens']}, expected {expected_output}"
        )
    if usage["total_tokens"] != expected_prompt + expected_output:
        raise ValueError("server usage.total_tokens does not equal prompt + completion tokens")
    for key in ("wall_seconds", "ttft_seconds", "decode_seconds"):
        value = result.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"invalid timing value: {key}={value!r}")


def metric_samples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def median_ms(key: str) -> float:
        return float(statistics.median(float(row[key]) * 1000.0 for row in rows))

    output_tokens = rows[0]["usage"]["completion_tokens"]
    decode_seconds = [float(row["decode_seconds"]) for row in rows]
    wall_seconds = [float(row["wall_seconds"]) for row in rows]
    grouped_rows = [row for row in rows if row.get("grouped_stream_timing")]
    return {
        "count": len(rows),
        "grouped_stream_timing": bool(grouped_rows),
        "grouped_stream_rows": len(grouped_rows),
        "first_event_token_counts": [row.get("first_event_token_count") for row in rows],
        "timing_note": (
            "TTFT is the arrival time of the first non-empty SSE token event; TPOT uses the "
            "conventional decode_seconds/(completion_tokens-1), decode throughput uses "
            "(completion_tokens-1)/decode_seconds, and E2E throughput uses "
            "completion_tokens/wall_seconds. Grouped SSE or speculative events therefore "
            "do not provide per-token inter-arrival timing."
        ),
        "ttft_ms_median": median_ms("ttft_seconds"),
        "end_to_end_ms_median": median_ms("wall_seconds"),
        "decode_ms_median": median_ms("decode_seconds"),
        "tpot_ms_median": float(statistics.median(
            seconds * 1000.0 / max(1, output_tokens - 1) for seconds in decode_seconds
        )),
        "end_to_end_output_tok_per_s_median": float(statistics.median(
            output_tokens / seconds for seconds in wall_seconds
        )),
        "decode_output_tok_per_s_median": float(statistics.median(
            max(0, output_tokens - 1) / seconds for seconds in decode_seconds
        )),
        "ttft_ms_samples": [float(row["ttft_seconds"]) * 1000.0 for row in rows],
        "end_to_end_ms_samples": [float(row["wall_seconds"]) * 1000.0 for row in rows],
        "decode_ms_samples": [float(row["decode_seconds"]) * 1000.0 for row in rows],
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="vLLM base URL or /v1/completions URL")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--prompt-token-ids", type=Path, required=True, help="JSON token list or object containing token_ids")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--expected-prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--measured", type=int, default=DEFAULT_MEASURED)
    parser.add_argument("--timeout", type=float, default=1800.0, help="per-request socket timeout in seconds")
    parser.add_argument("--api-key", default=None, help="Bearer token; defaults to VLLM_API_KEY")
    parser.add_argument(
        "--server-prefix-cache-disabled",
        action="store_true",
        help=(
            "record that the caller started the server with prefix caching disabled; "
            "this is metadata, not an independent server check"
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "state": "RUNNING",
        "endpoint": None,
        "model": args.model,
        "conditions": {
            "prompt_tokens_expected": args.expected_prompt_tokens,
            "output_tokens_expected": args.max_tokens,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "seed": SEED,
            "ignore_eos": True,
            "warmups": args.warmups,
            "measured": args.measured,
            "sampling_profile": "temperature=1.0, top_p=0.95, top_k=20, seed=123",
        },
        "runs": [],
    }
    try:
        if args.expected_prompt_tokens <= 0 or args.max_tokens <= 0:
            raise ValueError("expected prompt and max token counts must be positive")
        if args.warmups < 0 or args.measured <= 0:
            raise ValueError("warmups must be non-negative and measured must be positive")
        finite_positive(args.timeout, "--timeout")
        endpoint = completion_url(args.url)
        report["endpoint"] = endpoint
        token_ids = read_prompt_token_ids(args.prompt_token_ids, args.expected_prompt_tokens)
        report["prompt_token_ids"] = {
            "path": str(args.prompt_token_ids),
            "sha256": token_ids_sha256(token_ids),
            "count": len(token_ids),
        }
        api_key = args.api_key
        if api_key is None:
            import os

            api_key = os.environ.get("VLLM_API_KEY")
        features = {
            # vLLM forks that implement the sampling flag may disable cache
            # reads directly.  Unique cache_salt is the portable fallback.
            "skip_reading_prefix_cache": not args.server_prefix_cache_disabled,
            "cache_salt": not args.server_prefix_cache_disabled,
            "return_token_ids": True,
            "stream_options": True,
        }
        report["server_prefix_cache_disabled"] = args.server_prefix_cache_disabled
        report["prefix_cache_policy"] = (
            {
                "requested": "caller-declared server prefix caching disabled; no per-request cache control fields",
                "verification": "caller argument only; endpoint configuration is not queried",
            }
            if args.server_prefix_cache_disabled
            else {
                "requested": "skip_reading_prefix_cache plus unique cache_salt per request",
                "fallback": "if cache controls are rejected, cached prefill is possible and recorded",
            }
        )
        report["optional_features_initial"] = dict(features)

        for phase, count in (("warmup", args.warmups), ("measured", args.measured)):
            for index in range(count):
                request_number = len(report["runs"])
                result = run_request(
                    endpoint=endpoint,
                    api_key=api_key,
                    model=args.model,
                    token_ids=token_ids,
                    output_tokens=args.max_tokens,
                    timeout=args.timeout,
                    features=features,
                    request_number=request_number,
                )
                validate_result(result, args.expected_prompt_tokens, args.max_tokens)
                result["phase"] = phase
                result["index"] = index
                report["runs"].append(result)
                print(
                    f"{phase} {index}: TTFT {result['ttft_seconds'] * 1000:.3f} ms, "
                    f"E2E {result['wall_seconds'] * 1000:.3f} ms, "
                    f"usage={result['usage']}",
                    flush=True,
                )

        measured_rows = [row for row in report["runs"] if row["phase"] == "measured"]
        report["optional_features_final"] = dict(features)
        cached_values = [row["cached_prompt_tokens"] for row in report["runs"]]
        if args.server_prefix_cache_disabled:
            prefix_cache_mode = "server_config_explicitly_disabled"
        elif cached_values and any(value is not None for value in cached_values) and all(
            value == 0 for value in cached_values if value is not None
        ):
            prefix_cache_mode = "verified_zero_cached_prompt_tokens"
        elif any(isinstance(value, int) and value > 0 for value in cached_values):
            prefix_cache_mode = "prefix_cache_hit_observed"
        elif features["cache_salt"]:
            prefix_cache_mode = "unique_cache_salt_per_request_unverified"
        else:
            prefix_cache_mode = "cached_prefill_possible"
        report["prefix_cache_mode"] = prefix_cache_mode
        report["prefix_cache_observation"] = {
            "usage_prompt_tokens_details_present": all(value is not None for value in cached_values),
            "direct_skip_reading_prefix_cache_requested": features["skip_reading_prefix_cache"],
            "unique_cache_salt_active": features["cache_salt"],
            "note": (
                "Caller configuration is the declared source of truth."
                if args.server_prefix_cache_disabled
                else "The endpoint may ignore unknown extra fields; cached_tokens is the only direct "
                "server-side observation."
            ),
        }
        report["summary"] = metric_samples(measured_rows)
        report["state"] = "PASS"
    except (OSError, ValueError, RequestFailure) as error:
        report["state"] = "FAIL"
        report["error"] = str(error)
        print(f"benchmark failed: {error}", file=sys.stderr)
    finally:
        write_report(args.output, report)
    return 0 if report["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
