#!/usr/bin/env python3
"""Compare common Chat Completions response shape without treating peers as oracle."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PINNED = {
    "vllm": "568afb3a13806beb53bb2e6bd518269357b237c0",
    "sglang": "fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1",
    "llama_cpp": "f5919bf458ef190468b5c329bb293f8a54a1e69c",
}


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def serializer_snapshot(python: Path, source: Path, engine: str) -> dict[str, object]:
    expected = PINNED[engine]
    if git_head(source) != expected:
        raise ValueError(f"{engine} checkout does not match pinned commit")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source / ("python" if engine == "sglang" else ""))
    result = subprocess.run(
        [
            str(python),
            str(ROOT / "ci/tools/openai_differential_serializer_probe.py"),
            "--engine",
            engine,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def http_snapshot(base_url: str, model: str) -> dict[str, object]:
    common = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with one word."}],
        "temperature": 0,
        "max_completion_tokens": 1,
    }
    non_stream = post_json(base_url, {**common, "stream": False}, "application/json")
    raw_stream = post_json(base_url, {**common, "stream": True}, "text/event-stream")
    events = []
    terminal = None
    for block in raw_stream.decode("utf-8").split("\n\n"):
        if not block:
            continue
        lines = [line[6:] for line in block.splitlines() if line.startswith("data: ")]
        if len(lines) != 1:
            raise ValueError("SSE event does not contain exactly one data line")
        if lines[0] == "[DONE]":
            terminal = "[DONE]"
        else:
            events.append(json.loads(lines[0]))
    return {"non_stream": json.loads(non_stream), "stream": events, "terminal": terminal}


def post_json(base_url: str, payload: dict[str, object], expected_type: str) -> bytes:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        content_type = response.headers.get_content_type()
        if content_type != expected_type:
            raise ValueError(f"unexpected response content type {content_type}")
        return response.read()


def validate_common(name: str, snapshot: dict[str, object]) -> dict[str, object]:
    response = snapshot.get("non_stream")
    chunks = snapshot.get("stream")
    if not isinstance(response, dict) or not isinstance(chunks, list) or not chunks:
        raise ValueError(f"{name} produced an incomplete snapshot")
    if response.get("object") != "chat.completion" or snapshot.get("terminal") != "[DONE]":
        raise ValueError(f"{name} differs on response object or terminal")
    choices = response.get("choices")
    usage = response.get("usage")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(usage, dict):
        raise ValueError(f"{name} differs on non-stream choices/usage shape")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("index") != 0:
        raise ValueError(f"{name} differs on choice index")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError(f"{name} differs on assistant message role")
    if choice.get("finish_reason") not in {"stop", "length"}:
        raise ValueError(f"{name} differs on profile finish reason")
    ids = set()
    first_role = None
    final_reason = None
    for chunk in chunks:
        if not isinstance(chunk, dict) or chunk.get("object") != "chat.completion.chunk":
            raise ValueError(f"{name} differs on stream chunk object")
        ids.add(chunk.get("id"))
        stream_choices = chunk.get("choices")
        if not isinstance(stream_choices, list) or len(stream_choices) != 1:
            raise ValueError(f"{name} differs on stream choice shape")
        stream_choice = stream_choices[0]
        if not isinstance(stream_choice, dict):
            raise ValueError(f"{name} has an invalid stream choice")
        delta = stream_choice.get("delta")
        if isinstance(delta, dict) and first_role is None and "role" in delta:
            first_role = delta["role"]
        if stream_choice.get("finish_reason") is not None:
            final_reason = stream_choice["finish_reason"]
    if ids == {None} or len(ids) != 1 or first_role != "assistant":
        raise ValueError(f"{name} differs on stable stream identity/first role")
    if final_reason not in {"stop", "length"}:
        raise ValueError(f"{name} has no profile terminal finish reason")
    required_top = {"id", "object", "created", "model", "choices", "usage"}
    return {
        "result": "PASS",
        "mode": "runtime_http" if name in {"sllm", "llama_cpp"} else "serializer_schema",
        "non_stream_extra_fields": sorted(set(response) - required_top),
        "finish_reason": choice["finish_reason"],
        "stream_finish_reason": final_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, default=ROOT / "reference/vLLM")
    parser.add_argument("--sglang-source", type=Path, default=ROOT / "reference/SGLang")
    parser.add_argument("--llama-source", type=Path, default=ROOT / "reference/llama.cpp")
    parser.add_argument("--llama-url", required=True)
    parser.add_argument("--llama-model", default="qwen-diff")
    parser.add_argument("--sllm-url")
    parser.add_argument("--sllm-model", default="qwen3.5-4b")
    args = parser.parse_args()
    if git_head(args.llama_source) != PINNED["llama_cpp"]:
        raise ValueError("llama.cpp checkout does not match pinned commit")
    snapshots = {
        "vllm": serializer_snapshot(args.python, args.vllm_source, "vllm"),
        "sglang": serializer_snapshot(args.python, args.sglang_source, "sglang"),
        "llama_cpp": http_snapshot(args.llama_url, args.llama_model),
    }
    if args.sllm_url:
        snapshots["sllm"] = http_snapshot(args.sllm_url, args.sllm_model)
    report = {
        "schema_version": "openai-profile-differential-v1",
        "specification_oracle": "pinned-openai-profile-v1-not-peer-engines",
        "peer_commits": PINNED,
        "results": {name: validate_common(name, value) for name, value in snapshots.items()},
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
