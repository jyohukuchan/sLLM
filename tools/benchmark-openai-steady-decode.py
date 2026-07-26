#!/usr/bin/env python3
"""Capture five one-stream OpenAI-compatible decode requests as raw JSON.

This is a server-side fallback for engines that cannot express a prefilled
decode-only benchmark natively.  It uses a plain `/v1/completions` request,
not a chat template, and creates a deterministic Qwen prompt whose local
tokenizer count is exactly 1,028.  Each streamed request asks for 16 output
tokens.  The report separates TTFT from the inter-token intervals; the latter
are the client-visible steady-decode evidence and must not be conflated with a
kernel-only timer such as llama-bench.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_prompt(tokenizer_path: Path, tokens: int) -> tuple[str, dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    prompt = " hello" * tokens
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) != tokens:
        raise RuntimeError(f"prompt tokenization mismatch: wanted {tokens}, got {len(ids)}")
    return prompt, {
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_class": type(tokenizer).__name__,
        "requested_prompt_tokens": tokens,
        "observed_prompt_tokens_local": len(ids),
        "first_token_ids": ids[:16],
        "last_token_ids": ids[-16:],
        "add_special_tokens": False,
    }


def decode_one(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request_body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=request_body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started_monotonic = time.perf_counter()
    started_utc = utc_now()
    events: list[dict[str, Any]] = []
    content_events: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    error: str | None = None
    status: int | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            for raw_line in response:
                received = time.perf_counter()
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                event: dict[str, Any] = {
                    "received_offset_seconds": received - started_monotonic,
                    "raw": data,
                }
                if data == "[DONE]":
                    event["done"] = True
                    events.append(event)
                    break
                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    event["json"] = None
                    events.append(event)
                    continue
                event["json"] = parsed
                events.append(event)
                if isinstance(parsed, dict) and isinstance(parsed.get("usage"), dict):
                    usage = parsed["usage"]
                choices = parsed.get("choices") if isinstance(parsed, dict) else None
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    continue
                text = choices[0].get("text")
                if isinstance(text, str) and text:
                    content_events.append(
                        {
                            "received_offset_seconds": received - started_monotonic,
                            "text": text,
                            "finish_reason": choices[0].get("finish_reason"),
                        }
                    )
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = f"HTTPError: {exc}"
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - preserve the original error if body reading fails.
            body = ""
        events.append({"received_offset_seconds": time.perf_counter() - started_monotonic, "error_body": body})
    except Exception as exc:  # noqa: BLE001 - raw failures are part of benchmark evidence.
        error = f"{type(exc).__name__}: {exc}"

    finished_monotonic = time.perf_counter()
    response_text = "".join(event["text"] for event in content_events)
    token_times = [event["received_offset_seconds"] for event in content_events]
    intervals = [later - earlier for earlier, later in zip(token_times, token_times[1:])]
    return {
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "http_status": status,
        "error": error,
        "wall_seconds": finished_monotonic - started_monotonic,
        "first_content_seconds": token_times[0] if token_times else None,
        "last_content_seconds": token_times[-1] if token_times else None,
        "content_event_count": len(content_events),
        "content_events": content_events,
        "inter_content_intervals_seconds": intervals,
        "response_text": response_text,
        "usage": usage,
        "events": events,
    }


def run_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("error") is None and row.get("first_content_seconds") is not None]
    ttft = [float(row["first_content_seconds"]) for row in ok_rows]
    itls = [interval for row in ok_rows for interval in row.get("inter_content_intervals_seconds", []) if interval > 0]
    return {
        "successful_repetitions": len(ok_rows),
        "ttft_seconds": ttft,
        "ttft_mean_seconds": statistics.mean(ttft) if ttft else None,
        "ttft_median_seconds": statistics.median(ttft) if ttft else None,
        "inter_content_intervals_seconds": itls,
        "mean_itl_seconds": statistics.mean(itls) if itls else None,
        "median_itl_seconds": statistics.median(itls) if itls else None,
        "sample_variance_itl_seconds": statistics.variance(itls) if len(itls) > 1 else None,
        "client_visible_steady_tokens_per_second": 1.0 / statistics.mean(itls) if itls else None,
        "note": "This rate is derived from stream event intervals after the first content event; it excludes TTFT but includes server scheduling and local HTTP/SSE overhead.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="OpenAI-compatible /v1/completions URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=1028)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions != 5 or args.prompt_tokens != 1028 or args.max_tokens != 16:
        raise SystemExit("the controlled R9700 comparison requires five repeats, 1028 prompt tokens, and 16 output tokens")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing result directory: {args.output}")
    args.output.mkdir(parents=True)

    prompt, tokenization = load_prompt(args.tokenizer, args.prompt_tokens)
    (args.output / "prompt.txt").write_text(prompt, encoding="utf-8")
    write_json(args.output / "prompt-tokenization.json", tokenization)
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "min_tokens": args.max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    write_json(args.output / "request.json", payload)

    warmups = [decode_one(args.url, payload, args.timeout) for _ in range(args.warmup)]
    write_json(args.output / "warmups.json", warmups)
    repetitions = [decode_one(args.url, payload, args.timeout) for _ in range(args.repetitions)]
    write_json(args.output / "repetitions.json", repetitions)
    summary = {
        "schema_version": "ullm.r9700.external-engine.openai-decode.v1",
        "status": "ok" if all(row.get("error") is None for row in repetitions) else "failed",
        "engine": "openai-compatible-server",
        "url": args.url,
        "model": args.model,
        "prompt_tokenization": tokenization,
        "measurement": {
            "profiled": False,
            "single_stream": True,
            "prompt_tokens": args.prompt_tokens,
            "cache_length_start": 1028,
            "cache_length_end_nominal": 1044,
            "max_tokens": args.max_tokens,
            "warmup_requests": args.warmup,
            "timing_method": "client stream SSE",
        },
        "statistics": run_statistics(repetitions),
    }
    write_json(args.output / "summary.json", summary)
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
