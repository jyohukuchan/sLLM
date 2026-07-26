#!/usr/bin/env python3
"""Decode paired isolated SQ8 runtime outputs for the fixed prompt suite.

The script only turns saved token IDs into text and writes a side-by-side
review document.  It does not make a numerical pass/fail decision and never
changes a served-model manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


INPUT_SCHEMA = "ullm.qwen_lightweight_token_suite.v1"
OUTPUT_SCHEMA = "ullm.sq8_isolated_lightweight_output_comparison.v1"


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} {path} is not an object")
    return value


def generated_ids(path: Path) -> tuple[list[int], dict[str, Any]]:
    document = load_object(path, "runtime result")
    if document.get("passed") is not True:
        raise ValueError(f"runtime result did not pass: {path}")
    requests = document.get("requests")
    if not isinstance(requests, list) or len(requests) != 1 or not isinstance(requests[0], dict):
        raise ValueError(f"runtime result does not have one request: {path}")
    request = requests[0]
    ids = request.get("generated_token_ids")
    if not isinstance(ids, list):
        raise ValueError(f"runtime result has no generated token IDs: {path}")
    normalized: list[int] = []
    for index, token_id in enumerate(ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"runtime result has invalid token {index}: {path}")
        normalized.append(token_id)
    return normalized, request


def markdown_escape(value: str) -> str:
    return value.replace("```", "``\\`")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prepared-suite-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    prepared = args.prepared_suite_dir.expanduser().resolve()
    baseline_dir = args.baseline_dir.expanduser().resolve()
    candidate_dir = args.candidate_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output_dir}")
    suite = load_object(prepared / "suite.json", "prepared suite")
    if suite.get("schema_version") != INPUT_SCHEMA:
        raise SystemExit("prepared suite has an unexpected schema")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("prepared suite has no cases")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    output_dir.mkdir(parents=True)
    case_dir = output_dir / "cases"
    case_dir.mkdir()

    comparison_cases: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise SystemExit("prepared suite has an invalid case")
        case_id = case["case_id"]
        baseline_ids, baseline_request = generated_ids(baseline_dir / f"{case_id}.json")
        candidate_ids, candidate_request = generated_ids(candidate_dir / f"{case_id}.json")
        baseline_text = tokenizer.decode(
            baseline_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        candidate_text = tokenizer.decode(
            candidate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        record = {
            "case": case,
            "baseline": {
                "result_json": str(baseline_dir / f"{case_id}.json"),
                "generated_token_ids": baseline_ids,
                "generated_text": baseline_text,
                "terminal_reason": baseline_request.get("terminal_reason"),
                "terminal_status": baseline_request.get("terminal_status"),
            },
            "candidate": {
                "result_json": str(candidate_dir / f"{case_id}.json"),
                "generated_token_ids": candidate_ids,
                "generated_text": candidate_text,
                "terminal_reason": candidate_request.get("terminal_reason"),
                "terminal_status": candidate_request.get("terminal_status"),
            },
            "automatic_observations": {
                "baseline_text_empty": not baseline_text.strip(),
                "candidate_text_empty": not candidate_text.strip(),
                "same_generated_token_ids": baseline_ids == candidate_ids,
                "same_decoded_text": baseline_text == candidate_text,
                "quality_decision": "manual review required; no logits or top-1 threshold is used",
            },
        }
        (case_dir / f"{case_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        comparison_cases.append(record)

    summary = {
        "schema_version": OUTPUT_SCHEMA,
        "scope": "isolated SQ8 CK and handwritten output side-by-side decoding",
        "prepared_suite": str(prepared),
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "case_count": len(comparison_cases),
        "cases": [
            {
                "case_id": record["case"]["case_id"],
                "baseline_text_empty": record["automatic_observations"]["baseline_text_empty"],
                "candidate_text_empty": record["automatic_observations"]["candidate_text_empty"],
                "same_generated_token_ids": record["automatic_observations"]["same_generated_token_ids"],
                "same_decoded_text": record["automatic_observations"]["same_decoded_text"],
            }
            for record in comparison_cases
        ],
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Isolated SQ8 lightweight output comparison",
        "",
        "This is a content-review artifact. Token/logit equality is descriptive only, not a quality gate.",
    ]
    for record in comparison_cases:
        case = record["case"]
        lines.extend(
            [
                "",
                f"## {case['case_id']} ({case['category']})",
                "",
                "### Prompt",
                "",
                "```text",
                markdown_escape((prepared / case["rendered_prompt_file"]).read_text(encoding="utf-8")),
                "```",
                "",
                "### CK baseline",
                "",
                "```text",
                markdown_escape(record["baseline"]["generated_text"]),
                "```",
                "",
                "### Handwritten WMMA candidate",
                "",
                "```text",
                markdown_escape(record["candidate"]["generated_text"]),
                "```",
            ]
        )
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
