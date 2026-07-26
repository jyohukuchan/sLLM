#!/usr/bin/env python3
"""Decode and compare isolated wide-M SQ8_0 generation evidence.

This is deliberately an evidence summarizer, not a promotion tool or a
numerical gate.  It uses the same fixed prompt-suite text checks as the
lightweight-promotion policy, while retaining the actual completions for
human review.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any


def repository_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "tools" / "lightweight_promotion.py").is_file():
            return parent
    raise RuntimeError("could not locate repository root")


ROOT = repository_root()
sys.path.insert(0, str(ROOT / "tools"))
import lightweight_promotion as promotion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--generation-input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-suite",
        type=Path,
        default=ROOT / "docs" / "plans" / "lightweight-promotion-prompt-suite-v0.1.json",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def decode_token_ids(tokenizer: Any, value: Any, path: Path) -> tuple[list[int], str]:
    if not isinstance(value, list) or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0
        for token_id in value
    ):
        raise ValueError(f"{path}: generated_token_ids are invalid")
    token_ids = list(value)
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(text, str):
        raise ValueError(f"{path}: tokenizer returned non-text completion")
    text.encode("utf-8", errors="strict")
    return token_ids, text


def load_case(
    *,
    result_path: Path,
    definition: promotion.SuiteCase,
    tokenizer: Any,
) -> dict[str, Any]:
    result = load_json(result_path)
    requests = result.get("requests")
    if not isinstance(requests, list) or len(requests) != 1 or not isinstance(requests[0], dict):
        raise ValueError(f"{result_path}: expected one serving request")
    request = requests[0]
    token_ids, content = decode_token_ids(tokenizer, request.get("generated_token_ids"), result_path)
    return {
        "case_id": definition.case_id,
        "category": definition.category,
        "source_result": str(result_path),
        "prefill_mode": result.get("prefill_mode"),
        "prefill_chunk_tokens": result.get("prefill_chunk_tokens"),
        "runner_binary_sha256": result.get("runner_binary_sha256"),
        "request_terminal_reason": request.get("terminal_reason"),
        "generated_token_ids": token_ids,
        "generated_token_count": len(token_ids),
        "content": content,
        "character_count": len(content),
        "analysis": promotion.analyze_text(content, definition),
    }


def widths(run_root: Path) -> list[int]:
    root = run_root / "generation"
    result: list[int] = []
    if not root.is_dir():
        raise ValueError(f"missing generation directory: {root}")
    for path in root.glob("m*"):
        if not path.is_dir() or not path.name[1:].isdigit():
            continue
        result.append(int(path.name[1:]))
    if 128 not in result:
        raise ValueError("generation evidence must include M=128 baseline")
    return sorted(set(result))


def comparison(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    before = {str(record["case_id"]): record for record in baseline}
    after = {str(record["case_id"]): record for record in candidate}
    if set(before) != set(after):
        raise ValueError("baseline/candidate case coverage differs")
    cases: list[dict[str, Any]] = []
    exact = 0
    findings: list[str] = []
    for case_id in sorted(before):
        left = before[case_id]
        right = after[case_id]
        left_text = str(left["content"])
        right_text = str(right["content"])
        blocking = list(right["analysis"]["blocking"])
        # These are policy-defined obvious-collapse diagnostics, not a scalar
        # numerical threshold and not a release decision in this local run.
        if len(left_text) >= 40 and len(right_text) * 10 < len(left_text):
            blocking.append("extreme_shortening_vs_m128")
        if len(right_text) > max(2_000, len(left_text) * 10):
            blocking.append("extreme_lengthening_vs_m128")
        blocking = sorted(set(blocking))
        findings.extend(f"{case_id}:{item}" for item in blocking)
        matched = left_text == right_text
        exact += int(matched)
        cases.append(
            {
                "case_id": case_id,
                "m128_characters": len(left_text),
                "candidate_characters": len(right_text),
                "output_exact_match": matched,
                "generated_token_ids_exact_match": left["generated_token_ids"]
                == right["generated_token_ids"],
                "blocking": blocking,
                "attention": right["analysis"]["attention"],
            }
        )
    return {
        "case_count": len(cases),
        "output_exact_match_rate": exact / len(cases),
        "blocking_findings": findings,
        "cases": cases,
        "interpretation": "diagnostic evidence only; no numerical threshold decides acceptance",
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Wide-M isolated generation evidence",
        "",
        "These are direct SQ8_0 local-serving completions, not gateway activation or promotion evidence. They use the fixed lightweight-promotion prompt suite and retain the actual text. Automated findings are diagnostic obvious-collapse checks, not a scalar numerical acceptance gate.",
        "",
    ]
    baseline = summary["runs"]["128"]
    for width_text, records in summary["runs"].items():
        width = int(width_text)
        if width == 128:
            continue
        compared = summary["comparisons"][width_text]
        lines.extend(
            [
                f"## M={width} compared with M=128",
                "",
                f"- Exact-text rate (diagnostic): {compared['output_exact_match_rate']:.3f}",
                f"- Automated findings: `{json.dumps(compared['blocking_findings'], ensure_ascii=False)}`",
                "",
            ]
        )
        by_id = {str(record["case_id"]): record for record in records}
        baseline_by_id = {str(record["case_id"]): record for record in baseline}
        comparison_by_id = {str(record["case_id"]): record for record in compared["cases"]}
        for case_id in sorted(baseline_by_id):
            left = baseline_by_id[case_id]
            right = by_id[case_id]
            details = comparison_by_id[case_id]
            lines.extend(
                [
                    f"### {case_id}",
                    "",
                    "M=128:",
                    "",
                    "<pre>",
                    html.escape(str(left["content"])),
                    "</pre>",
                    "",
                    f"M={width}:",
                    "",
                    "<pre>",
                    html.escape(str(right["content"])),
                    "</pre>",
                    "",
                    "Diagnostic observations:",
                    "",
                    "<pre>",
                    html.escape(json.dumps(details, ensure_ascii=False, indent=2)),
                    "</pre>",
                    "",
                ]
            )
    if len(summary["runs"]) == 1:
        lines.extend(
            [
                "Only the M=128 baseline was collected because every wide-M numerical comparison was byte-identical. The baseline completions remain available in the JSON evidence.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    generation_input = load_json(args.generation_input / "suite.json")
    case_ids = generation_input.get("cases")
    if not isinstance(case_ids, list):
        raise ValueError("generation input manifest has no cases")
    suite = promotion.load_suite(args.prompt_suite)
    if {case.case_id for case in suite} != {
        str(item.get("case_id")) for item in case_ids if isinstance(item, dict)
    }:
        raise ValueError("generation input and fixed prompt suite coverage differ")
    from transformers import AutoTokenizer, __version__ as transformers_version

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    collected: dict[str, list[dict[str, Any]]] = {}
    for width in widths(args.run_root):
        collected[str(width)] = [
            load_case(
                result_path=args.run_root / "generation" / f"m{width}" / f"{case.case_id}.json",
                definition=case,
                tokenizer=tokenizer,
            )
            for case in suite
        ]
    baseline = collected["128"]
    comparisons = {
        width: comparison(baseline, records)
        for width, records in collected.items()
        if width != "128"
    }
    summary = {
        "schema_version": "ullm.sq8.prefill_chunk_width.generation.v1",
        "scope": "isolated direct SQ8_0 serving evidence; no service activation or promotion",
        "generation_input": str(args.generation_input),
        "model_dir": str(args.model_dir),
        "transformers_version": transformers_version,
        "not_a_numerical_gate": True,
        "runs": collected,
        "comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as destination:
        json.dump(summary, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")
    with args.output_markdown.open("x", encoding="utf-8") as destination:
        destination.write(markdown(summary))
        destination.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
