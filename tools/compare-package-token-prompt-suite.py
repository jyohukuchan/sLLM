#!/usr/bin/env python3
"""Compare generated token IDs between two package prompt suite runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two run-package-token-prompt-suite.py summary outputs and "
            "verify exact generated-token agreement for every shared case."
        )
    )
    parser.add_argument("--reference-summary", required=True, type=Path)
    parser.add_argument("--candidate-summary", required=True, type=Path)
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-label", default="candidate")
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


def as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be an object")
    return value


def as_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SystemExit(f"{label} must be a list")
    return value


def token_list(value: Any, label: str) -> list[int]:
    raw = as_list(value, label)
    try:
        return [int(item) for item in raw]
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must contain integer token IDs") from exc


def case_map(summary: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    cases = as_list(summary.get("cases"), f"{label}.cases")
    mapped: dict[str, dict[str, Any]] = {}
    for index, raw_case in enumerate(cases):
        case = as_mapping(raw_case, f"{label}.cases[{index}]")
        case_id = str(case.get("id", "")).strip()
        if not case_id:
            raise SystemExit(f"{label}.cases[{index}] has no id")
        if case_id in mapped:
            raise SystemExit(f"{label}.cases has duplicate id: {case_id}")
        mapped[case_id] = case
    return mapped


def resolve_report_path(summary_path: Path, case: dict[str, Any], label: str) -> Path:
    raw = case.get("report")
    if not isinstance(raw, str) or not raw.strip():
        raise SystemExit(f"{label} case {case.get('id')}: report must be a path string")
    path = Path(raw)
    if path.exists():
        return path
    sibling = summary_path.parent / path.name
    if sibling.exists():
        return sibling
    raise SystemExit(f"{label} case {case.get('id')}: report not found: {raw}")


def token_hash(tokens: list[int]) -> str:
    payload = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def first_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def stop_signature(report: dict[str, Any], label: str) -> dict[str, Any]:
    stop = as_mapping(report.get("stop"), f"{label}.stop")
    return {
        "reason": stop.get("reason"),
        "stopped": stop.get("stopped"),
        "stopped_on_token_id": stop.get("stopped_on_token_id"),
        "stopped_on_token_sequence": stop.get("stopped_on_token_sequence"),
    }


def compare_case(
    reference_summary_path: Path,
    candidate_summary_path: Path,
    reference_case: dict[str, Any],
    candidate_case: dict[str, Any],
    reference_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    case_id = str(reference_case["id"])
    reference_report = load_json_object(
        resolve_report_path(reference_summary_path, reference_case, reference_label),
        f"{reference_label} report {case_id}",
    )
    candidate_report = load_json_object(
        resolve_report_path(candidate_summary_path, candidate_case, candidate_label),
        f"{candidate_label} report {case_id}",
    )
    reference_prompt = token_list(reference_report.get("prompt_token_ids"), f"{reference_label}.{case_id}.prompt_token_ids")
    candidate_prompt = token_list(candidate_report.get("prompt_token_ids"), f"{candidate_label}.{case_id}.prompt_token_ids")
    reference_generated = token_list(
        reference_report.get("generated_token_ids"),
        f"{reference_label}.{case_id}.generated_token_ids",
    )
    candidate_generated = token_list(
        candidate_report.get("generated_token_ids"),
        f"{candidate_label}.{case_id}.generated_token_ids",
    )
    prompt_match = reference_prompt == candidate_prompt
    generated_match = reference_generated == candidate_generated
    reference_stop = stop_signature(reference_report, f"{reference_label}.{case_id}")
    candidate_stop = stop_signature(candidate_report, f"{candidate_label}.{case_id}")
    stop_match = reference_stop == candidate_stop

    return {
        "id": case_id,
        "category": reference_case.get("category"),
        "prompt_tokens": len(reference_prompt),
        "reference_generated_tokens": len(reference_generated),
        "candidate_generated_tokens": len(candidate_generated),
        "prompt_tokens_match": prompt_match,
        "generated_tokens_match": generated_match,
        "generated_first_mismatch_index": first_mismatch(reference_generated, candidate_generated),
        "generated_token_sha256": token_hash(reference_generated) if generated_match else None,
        "stop_match": stop_match,
        "reference_stop": reference_stop,
        "candidate_stop": candidate_stop,
        "reference_verified": reference_report.get("verified"),
        "candidate_verified": candidate_report.get("verified"),
        "both_verified": reference_report.get("verified") is True and candidate_report.get("verified") is True,
        "reference_output_status": reference_case.get("output_status"),
        "candidate_output_status": candidate_case.get("output_status"),
        "output_status_match": reference_case.get("output_status") == candidate_case.get("output_status"),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    reference_summary = load_json_object(args.reference_summary, "reference summary")
    candidate_summary = load_json_object(args.candidate_summary, "candidate summary")
    reference_cases = case_map(reference_summary, "reference summary")
    candidate_cases = case_map(candidate_summary, "candidate summary")
    reference_ids = set(reference_cases)
    candidate_ids = set(candidate_cases)
    common_ids = sorted(reference_ids & candidate_ids)
    missing_in_candidate = sorted(reference_ids - candidate_ids)
    extra_in_candidate = sorted(candidate_ids - reference_ids)

    case_reports = [
        compare_case(
            args.reference_summary,
            args.candidate_summary,
            reference_cases[case_id],
            candidate_cases[case_id],
            args.reference_label,
            args.candidate_label,
        )
        for case_id in common_ids
    ]
    metrics = {
        "reference_case_count": len(reference_cases),
        "candidate_case_count": len(candidate_cases),
        "compared_case_count": len(case_reports),
        "missing_in_candidate_count": len(missing_in_candidate),
        "extra_in_candidate_count": len(extra_in_candidate),
        "prompt_token_match_count": sum(1 for item in case_reports if item["prompt_tokens_match"]),
        "generated_token_match_count": sum(1 for item in case_reports if item["generated_tokens_match"]),
        "stop_match_count": sum(1 for item in case_reports if item["stop_match"]),
        "both_verified_count": sum(1 for item in case_reports if item["both_verified"]),
        "output_status_match_count": sum(1 for item in case_reports if item["output_status_match"]),
    }
    passed = (
        not missing_in_candidate
        and not extra_in_candidate
        and all(item["prompt_tokens_match"] for item in case_reports)
        and all(item["generated_tokens_match"] for item in case_reports)
        and all(item["stop_match"] for item in case_reports)
        and all(item["both_verified"] for item in case_reports)
    )
    metrics["passed"] = passed
    return {
        "schema_version": "package-token-prompt-suite-generated-token-guard-v0.1",
        "reference": {
            "label": args.reference_label,
            "summary": str(args.reference_summary),
            "suite": reference_summary.get("suite"),
            "device_index": reference_summary.get("device_index"),
        },
        "candidate": {
            "label": args.candidate_label,
            "summary": str(args.candidate_summary),
            "suite": candidate_summary.get("suite"),
            "device_index": candidate_summary.get("device_index"),
        },
        "metrics": metrics,
        "missing_in_candidate": missing_in_candidate,
        "extra_in_candidate": extra_in_candidate,
        "cases": case_reports,
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
    metrics = as_mapping(report.get("metrics"), "report.metrics")
    lines = [
        "# Package Prompt Suite Generated-Token Guard",
        "",
    ]
    if json_path is not None:
        lines.append(f"- JSON: `{json_path}`")
    lines.extend(
        [
            f"- Reference: `{report['reference']['label']}`",
            f"- Candidate: `{report['candidate']['label']}`",
            f"- Passed: `{fmt_bool(metrics.get('passed'))}`",
            f"- Compared cases: `{metrics.get('compared_case_count')}`",
            "",
            "| case | category | prompt match | generated match | stop match | both verified | output status match | generated tokens | sha256 |",
            "| --- | --- | :---: | :---: | :---: | :---: | :---: | ---: | --- |",
        ]
    )
    for item in report["cases"]:
        lines.append(
            "| {case} | {category} | {prompt} | {generated} | {stop} | {verified} | {status} | {tokens} | {sha} |".format(
                case=item["id"],
                category=item.get("category", ""),
                prompt=fmt_bool(item.get("prompt_tokens_match")),
                generated=fmt_bool(item.get("generated_tokens_match")),
                stop=fmt_bool(item.get("stop_match")),
                verified=fmt_bool(item.get("both_verified")),
                status=fmt_bool(item.get("output_status_match")),
                tokens=item.get("reference_generated_tokens"),
                sha=(item.get("generated_token_sha256") or "")[:16],
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
