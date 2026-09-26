#!/usr/bin/env python3
"""Build the fixed Tier A prompt suite for Phase 87 Stage 3 M3 timing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "ci/matrix/mtp-bench-v1.json"
FIXTURE = REPO / "ci/fixtures/mtp-bench-v1"


def build() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = []
    for condition in manifest["conditions"]:
        if condition["tier"] != "A":
            continue
        relative = Path(condition["prompt_file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("prompt path escapes the frozen fixture")
        message = (FIXTURE / relative).read_text(encoding="utf-8")
        digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
        if digest != condition["prompt_sha256"] or condition["output_tokens"] != 256:
            raise ValueError(f"frozen prompt or output length differs: {condition['cond_id']}")
        cases.append({
            "id": condition["cond_id"],
            "task": condition.get("task", "translation"),
            "language": condition.get("instruction_language", condition.get("target_language")),
            "message": message,
        })
    if len(cases) != 26 or len({case["id"] for case in cases}) != 26:
        raise ValueError("Tier A must contain exactly 26 unique conditions")
    return {
        "schema": "phase85-a16-mtp-suite-v1",
        "source_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "seeds": [123],
        "max_completion_tokens": 256,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute():
        parser.error("--output must be absolute")
    if args.output.exists():
        parser.error("--output must not already exist")
    suite = build()
    content = json.dumps(suite, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(content, encoding="utf-8")
    print(f"{len(suite['cases'])} cases sha256={hashlib.sha256(content.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
