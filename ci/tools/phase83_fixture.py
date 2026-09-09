#!/usr/bin/env python3
"""Capture the Phase83 fixture from the reviewed Rust benchmark frontend.

The benchmark's fixture-only mode verifies the locked artifact, applies the
reviewed Qwen3.8 chat template, tokenizes it, and exits before HIP connection.
This helper only writes compact metadata and explicitly local token JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path


TARGET_TOKENS = 8192


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def token_digest(tokens: list[int]) -> str:
    payload = b"".join(struct.pack("<i", token) for token in tokens)
    return sha256_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="compact metadata JSON")
    parser.add_argument(
        "--tokens-output",
        type=Path,
        help="optional local token-ID JSON; keep under .local-artifacts",
    )
    args = parser.parse_args()
    if args.tokens_output and ".local-artifacts" not in args.tokens_output.parts:
        raise SystemExit("--tokens-output must be under .local-artifacts")

    environment = os.environ.copy()
    environment.update(
        {
            "SLLM_PHASE78_MODEL_PATH": str(args.model_root),
            "SLLM_PHASE83_MODE": "coding8192",
            "SLLM_PHASE83_FIXTURE_ONLY": "1",
        }
    )
    result = subprocess.run(
        [str(args.benchmark)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"fixture-only benchmark failed ({result.returncode}): {result.stderr.strip()}"
        )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit(f"fixture-only benchmark did not emit JSON: {error}")
    if report.get("schema_version") != "phase83-qwen38-fixture-only-v1":
        raise SystemExit("fixture-only benchmark schema mismatch")
    if report.get("state") != "PASS":
        raise SystemExit("fixture-only benchmark did not PASS")
    tokens = report.get("token_ids")
    fixture = report.get("fixture")
    if not isinstance(tokens, list) or len(tokens) != TARGET_TOKENS:
        raise SystemExit(f"fixture token count is not {TARGET_TOKENS}")
    if not isinstance(fixture, dict) or fixture.get("total_tokens") != TARGET_TOKENS:
        raise SystemExit("fixture metadata token count mismatch")
    token_ids = [int(token) for token in tokens]
    if token_digest(token_ids) != fixture.get("sha256"):
        raise SystemExit("fixture token digest mismatch")

    metadata = {
        "schema_version": "phase83-coding-fixture-metadata-v1",
        "generator": "ci/tools/phase83_fixture.py",
        "source_schema": report["schema_version"],
        "target_tokens": TARGET_TOKENS,
        "message_sha256": sha256_bytes(report["message"].encode()),
        "rendered_prompt_sha256": sha256_bytes(report["rendered_prompt"].encode()),
        "token_ids_sha256": fixture["sha256"],
        "benchmark_sha256": sha256_bytes(args.benchmark.read_bytes()),
        "model_root": str(args.model_root),
        "payload_policy": "token JSON is local scratch under .local-artifacts; do not track model or fixture payload",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.tokens_output:
        args.tokens_output.parent.mkdir(parents=True, exist_ok=True)
        args.tokens_output.write_text(
            json.dumps(token_ids, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
