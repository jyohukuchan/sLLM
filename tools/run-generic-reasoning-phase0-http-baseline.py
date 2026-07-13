#!/usr/bin/env python3
"""Capture a credential-free, hash-only Phase 0 HTTP/SSE baseline.

The collector uses a short-lived Docker client on the OpenWebUI network. It
never writes prompt or response bodies; only request/response hashes and
bounded protocol metadata are published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("/etc/ullm/served-models/active.json")
DEFAULT_TOKENIZER = Path(
    "/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B"
)
DEFAULT_IMAGE = "ullm/open-webui:0.9.4-ullm.1"
DEFAULT_NETWORK = "open-webui-network"
DEFAULT_KEY_FILE = Path("/etc/ullm/openai-api-key")
TARGETS = (18, 1024, 2048, 3072)


class BaselineError(RuntimeError):
    """Raised when a baseline record cannot be completed safely."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tokenizer(root: Path) -> Any:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(root, local_files_only=True)
    except Exception as error:
        raise BaselineError("failed to load the local tokenizer") from error


def _prompt_for_target(tokenizer: Any, target: int) -> tuple[str, int]:
    if target < 1:
        raise BaselineError("prompt target must be positive")
    repetitions = max(0, target - 12)
    for _ in range(64):
        content = " a" * repetitions
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        token_ids = encoded["input_ids"] if hasattr(encoded, "__getitem__") else encoded
        count = len(token_ids)
        if count == target:
            return content, count
        repetitions = max(0, repetitions + target - count)
    raise BaselineError(f"could not construct an exact {target}-token prompt")


def _docker_curl(
    body: bytes,
    *,
    endpoint: str,
    key_file: Path,
    image: str,
    network: str,
    timeout_seconds: float,
    stream: bool,
) -> tuple[int, bytes]:
    curl_script = (
        "set -eu; "
        "key=$(cat /run/secrets/ullm-api-key); "
        "config=$(mktemp); "
        "trap 'rm -f \"$config\"' EXIT; "
        "umask 077; "
        "printf 'header = \"Authorization: Bearer %s\"\\n' \"$key\" > \"$config\"; "
        "exec curl --config \"$config\" --silent --show-error --no-buffer "
        "-H 'Content-Type: application/json' "
        + ("-H 'Accept: text/event-stream' " if stream else "")
        + "-w '%{http_code}' --data-binary @- "
        + endpoint
    )
    key_path = key_file.resolve(strict=True)
    command = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--group-add",
        str(key_path.stat().st_gid),
        "--network",
        network,
        "-v",
        f"{key_path}:/run/secrets/ullm-api-key:ro",
        "--entrypoint",
        "sh",
        image,
        "-c",
        curl_script,
    ]
    try:
        result = subprocess.run(
            command,
            input=body,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BaselineError("Docker HTTP probe failed or timed out") from error
    if result.returncode != 0 or len(result.stdout) < 3:
        raise BaselineError("Docker HTTP probe returned no complete HTTP status")
    status_bytes = result.stdout[-3:]
    if not status_bytes.isdigit():
        raise BaselineError("Docker HTTP probe returned a malformed HTTP status")
    return int(status_bytes), result.stdout[:-3]


def _nonstream_metadata(status: int, response: bytes) -> dict[str, Any]:
    try:
        value = json.loads(response)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BaselineError("non-stream response was not JSON") from error
    if not isinstance(value, dict):
        raise BaselineError("non-stream response was not an object")
    usage = value.get("usage")
    if not isinstance(usage, dict):
        raise BaselineError("non-stream response has no usage object")
    return {
        "http_status": status,
        "response_body_sha256": _sha256(response),
        "response_bytes": len(response),
        "finish_reason": (
            value.get("choices", [{}])[0].get("finish_reason")
            if isinstance(value.get("choices"), list) and value["choices"]
            else None
        ),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _stream_metadata(status: int, response: bytes) -> dict[str, Any]:
    event_sequence: list[str] = []
    delta_keys: list[list[str]] = []
    usage: dict[str, Any] | None = None
    chunks = 0
    invalid_data_lines = 0
    for line in response.splitlines():
        if not line.startswith(b"data: "):
            if line.strip():
                invalid_data_lines += 1
            continue
        payload = line[6:]
        if payload == b"[DONE]":
            event_sequence.append("done")
            continue
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BaselineError("stream contained invalid JSON data") from error
        if not isinstance(value, dict):
            raise BaselineError("stream data was not an object")
        chunks += 1
        if isinstance(value.get("usage"), dict):
            usage = value["usage"]
            event_sequence.append("usage")
        choices = value.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict) and choice.get("finish_reason") is not None:
                event_sequence.append("stop")
            delta = choice.get("delta")
            if isinstance(delta, dict):
                delta_keys.append(sorted(str(key) for key in delta))
                if "role" in delta:
                    event_sequence.append("role")
                if "reasoning_content" in delta:
                    event_sequence.append("reasoning_token")
                if "content" in delta:
                    event_sequence.append("token")
    if usage is None:
        raise BaselineError("stream contained no usage chunk")
    return {
        "http_status": status,
        "response_body_sha256": _sha256(response),
        "response_bytes": len(response),
        "chunks": chunks,
        "event_sequence": event_sequence,
        "delta_keys": delta_keys,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
        "invalid_data_lines": invalid_data_lines,
    }


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    if path.is_symlink():
        raise BaselineError("output path must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(raw_path)
        with os.fdopen(descriptor, "w", encoding="ascii") as destination:
            json.dump(value, destination, ensure_ascii=True, allow_nan=False, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def capture(
    output: Path,
    *,
    manifest: Path,
    tokenizer_root: Path,
    key_file: Path,
    image: str,
    network: str,
    endpoint: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    if not manifest.is_file() or manifest.is_symlink():
        raise BaselineError("active manifest must be a regular non-symlink file")
    if not key_file.is_file() or key_file.is_symlink():
        raise BaselineError("API key file must be a regular non-symlink file")
    active = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(active, dict):
        raise BaselineError("active manifest is not an object")
    tokenizer = _load_tokenizer(tokenizer_root)
    model = active.get("public", {}).get("id")
    if not isinstance(model, str) or not model:
        raise BaselineError("active manifest has no public model ID")
    cases: list[dict[str, Any]] = []
    for target in TARGETS:
        prompt, prompt_tokens = _prompt_for_target(tokenizer, target)
        request = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2,
            "temperature": 0,
            "stream": False,
        }
        request_body = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        status, response = _docker_curl(
            request_body,
            endpoint=endpoint,
            key_file=key_file,
            image=image,
            network=network,
            timeout_seconds=timeout_seconds,
            stream=False,
        )
        nonstream = _nonstream_metadata(status, response)
        if status != 200 or nonstream["prompt_tokens"] != prompt_tokens:
            raise BaselineError(f"non-stream baseline case {target} did not match its contract")
        cases.append(
            {
                "id": f"phase0-v1-target-{target}",
                "target_prompt_tokens": target,
                "prompt_sha256": _sha256(prompt.encode("utf-8")),
                "prompt_tokens": prompt_tokens,
                "request_body_sha256": _sha256(request_body),
                "request_max_tokens": 2,
                "nonstream": nonstream,
            }
        )

    prompt, prompt_tokens = _prompt_for_target(tokenizer, TARGETS[0])
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request_body = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    status, response = _docker_curl(
        request_body,
        endpoint=endpoint,
        key_file=key_file,
        image=image,
        network=network,
        timeout_seconds=timeout_seconds,
        stream=True,
    )
    stream = _stream_metadata(status, response)
    if status != 200 or stream["usage"]["prompt_tokens"] != prompt_tokens:
        raise BaselineError("stream baseline did not match its prompt usage")
    cases[0]["stream"] = {
        **stream,
        "request_body_sha256": _sha256(request_body),
    }

    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ).stdout.strip()
    promotion = active.get("promotion", {})
    document = {
        "schema_version": "ullm.generic_reasoning_phase0_http_baseline.v1",
        "status": "partial",
        "captured_at_unix_ns": time.time_ns(),
        "production_activation_performed": False,
        "source_commit": source_commit,
        "active_manifest": {
            "path": os.fspath(manifest),
            "sha256": _sha256_file(manifest),
            "schema_version": active.get("schema_version"),
        },
        "active_promotion_source_commit": promotion.get("source_commit"),
        "source_commit_aligned": source_commit == promotion.get("source_commit"),
        "worker": {
            "protocol": active.get("worker", {}).get("protocol"),
            "binary": active.get("worker", {}).get("binary"),
            "binary_sha256": (
                _sha256_file(Path(active["worker"]["binary"]))
                if isinstance(active.get("worker", {}).get("binary"), str)
                and Path(active["worker"]["binary"]).is_file()
                else None
            ),
        },
        "endpoint": endpoint,
        "image": image,
        "cases": cases,
        "raw_bodies_stored": False,
        "missing": [
            "same-HEAD AQ4 source/token alignment",
            "AQ4 generated token IDs",
        ],
    }
    _atomic_write(output, document)
    return document


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tokenizer-root", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--network", default=DEFAULT_NETWORK)
    parser.add_argument("--endpoint", default="http://172.20.0.1:8000/v1/chat/completions")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        document = capture(
            args.output,
            manifest=args.manifest,
            tokenizer_root=args.tokenizer_root,
            key_file=args.key_file,
            image=args.image,
            network=args.network,
            endpoint=args.endpoint,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as error:
        print(f"Phase 0 HTTP baseline failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": document["schema_version"],
                "output": os.fspath(args.output.resolve()),
                "source_commit_aligned": document["source_commit_aligned"],
                "case_count": len(document["cases"]),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
