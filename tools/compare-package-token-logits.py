#!/usr/bin/env python3
"""Compare two package-token-ids-logits-smoke JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare final top logits from two package-token-ids-logits-smoke reports."
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--logit-atol", type=float, default=1e-6)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"failed to read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} {path}: expected JSON object")
    return value


def as_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SystemExit(f"{label} must be a list")
    return value


def token_list(value: Any, label: str) -> list[int]:
    try:
        return [int(item) for item in as_list(value, label)]
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must contain integer token IDs") from exc


def top_logits(value: Any, label: str) -> list[dict[str, Any]]:
    rows = as_list(value, label)
    parsed = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"{label}[{index}] must be an object")
        try:
            parsed.append({"token_id": int(row["token_id"]), "logit": float(row["logit"])})
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"{label}[{index}] must contain token_id and logit") from exc
    return parsed


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    reference = load_json_object(args.reference, "reference")
    candidate = load_json_object(args.candidate, "candidate")
    reference_tokens = token_list(reference.get("token_ids"), "reference.token_ids")
    candidate_tokens = token_list(candidate.get("token_ids"), "candidate.token_ids")
    reference_top = top_logits(reference.get("top_logits"), "reference.top_logits")
    candidate_top = top_logits(candidate.get("top_logits"), "candidate.top_logits")
    top_count = min(len(reference_top), len(candidate_top))
    rows = []
    for index in range(top_count):
        left = reference_top[index]
        right = candidate_top[index]
        diff = abs(left["logit"] - right["logit"])
        rows.append(
            {
                "rank": index,
                "reference_token_id": left["token_id"],
                "candidate_token_id": right["token_id"],
                "token_id_match": left["token_id"] == right["token_id"],
                "reference_logit": left["logit"],
                "candidate_logit": right["logit"],
                "abs_logit_diff": diff,
                "logit_within_atol": diff <= args.logit_atol,
            }
        )

    prompt_match = reference_tokens == candidate_tokens
    top_count_match = len(reference_top) == len(candidate_top)
    top_token_ids_match = top_count_match and all(item["token_id_match"] for item in rows)
    max_abs_diff = max((item["abs_logit_diff"] for item in rows), default=None)
    logits_within_atol = top_count_match and all(item["logit_within_atol"] for item in rows)
    both_verified = reference.get("verified") is True and candidate.get("verified") is True
    passed = prompt_match and top_token_ids_match and logits_within_atol and both_verified

    return {
        "schema_version": "package-token-ids-logits-guard-v0.1",
        "reference": {
            "label": args.reference_label,
            "path": str(args.reference),
            "device_index": reference.get("device_index"),
            "verified": reference.get("verified"),
            "timing_ms_total": reference.get("timing_ms", {}).get("total"),
        },
        "candidate": {
            "label": args.candidate_label,
            "path": str(args.candidate),
            "device_index": candidate.get("device_index"),
            "verified": candidate.get("verified"),
            "timing_ms_total": candidate.get("timing_ms", {}).get("total"),
        },
        "metrics": {
            "prompt_tokens": len(reference_tokens),
            "prompt_tokens_match": prompt_match,
            "top_count": len(reference_top),
            "top_count_match": top_count_match,
            "top_token_ids_match": top_token_ids_match,
            "max_abs_logit_diff": max_abs_diff,
            "logit_atol": args.logit_atol,
            "logits_within_atol": logits_within_atol,
            "both_verified": both_verified,
            "passed": passed,
        },
        "top_logits": rows,
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fmt_bool(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return str(value)


def write_md(path: Path, json_path: Path | None, report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    lines = ["# Package Token Logits Guard", ""]
    if json_path is not None:
        lines.append(f"- JSON: `{json_path}`")
    lines.extend(
        [
            f"- Reference: `{report['reference']['label']}`",
            f"- Candidate: `{report['candidate']['label']}`",
            f"- Passed: `{fmt_bool(metrics['passed'])}`",
            f"- Max abs logit diff: `{metrics['max_abs_logit_diff']}`",
            "",
            "| rank | reference token | candidate token | token match | reference logit | candidate logit | abs diff | within atol |",
            "| ---: | ---: | ---: | :---: | ---: | ---: | ---: | :---: |",
        ]
    )
    for item in report["top_logits"]:
        lines.append(
            "| {rank} | {ref_token} | {cand_token} | {match} | {ref_logit:.9f} | {cand_logit:.9f} | {diff:.9f} | {within} |".format(
                rank=item["rank"],
                ref_token=item["reference_token_id"],
                cand_token=item["candidate_token_id"],
                match=fmt_bool(item["token_id_match"]),
                ref_logit=item["reference_logit"],
                cand_logit=item["candidate_logit"],
                diff=item["abs_logit_diff"],
                within=fmt_bool(item["logit_within_atol"]),
            )
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    report = build_report(args)
    if args.output_json is not None:
        write_json(args.output_json, report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_md is not None:
        write_md(args.output_md, args.output_json, report)
    return 0 if report["metrics"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
