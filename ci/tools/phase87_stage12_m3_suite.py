#!/usr/bin/env python3
"""Build the fixed Tier A MTP width 2/3/4 timing suite from mtp-bench-v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from phase87_stage12_acceptance import load_identity


def build() -> dict:
    identity = load_identity()
    cases = []
    for case_id, condition in identity["conditions"].items():
        relative = Path(condition["prompt_file"])
        message = (identity["fixture_root"] / relative).read_text(encoding="utf-8")
        if hashlib.sha256(message.encode("utf-8")).hexdigest() != condition["prompt_sha256"]:
            raise ValueError(f"frozen prompt changed: {case_id}")
        if condition["output_tokens"] != 256:
            raise ValueError(f"frozen output length changed: {case_id}")
        cases.append(
            {
                "id": case_id,
                "task": condition.get("task", "translation"),
                "language": condition.get(
                    "instruction_language", condition.get("target_language")
                ),
                "message": message,
            }
        )
    if len(cases) != 26:
        raise ValueError("Tier A must contain exactly 26 conditions")
    return {
        "schema": "phase87-stage12-mtp-suite-v2",
        "source_manifest_sha256": identity["manifest_sha256"],
        "source_fixture_tree_sha256": identity["manifest"]["fixture_tree_sha256"],
        "supported_widths": [2, 3, 4],
        "default_width": 2,
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
