#!/usr/bin/env python3
"""Compare generated token IDs and decoded text between two package prompt suite runs."""

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
    parser.add_argument("--logit-atol", type=float, default=1e-6)
    parser.add_argument(
        "--acceptance-mode",
        choices=("strict", "behavioral"),
        default="strict",
        help=(
            "strict requires exact generated token/text and top-logit agreement; "
            "behavioral records exact-match diagnostics but only gates on prompt match, "
            "candidate verification, non-empty generated text, and accepted output health"
        ),
    )
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


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def decoded_text(report: dict[str, Any], key: str, label: str) -> str:
    decoded = as_mapping(report.get("decoded_text"), f"{label}.decoded_text")
    value = decoded.get(key)
    if not isinstance(value, str):
        raise SystemExit(f"{label}.decoded_text.{key} must be a string")
    return value


def nested_top_logits(report: dict[str, Any], path: tuple[str, str], label: str) -> list[dict[str, Any]]:
    parent = as_mapping(report.get(path[0]), f"{label}.{path[0]}")
    return top_logits(parent.get(path[1]), f"{label}.{path[0]}.{path[1]}")


def compare_top_logits(
    reference_top: list[dict[str, Any]],
    candidate_top: list[dict[str, Any]],
    logit_atol: float,
) -> dict[str, Any]:
    count = min(len(reference_top), len(candidate_top))
    rows = []
    for index in range(count):
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
                "logit_within_atol": diff <= logit_atol,
            }
        )
    count_match = len(reference_top) == len(candidate_top)
    token_ids_match = count_match and all(item["token_id_match"] for item in rows)
    max_abs_diff = max((item["abs_logit_diff"] for item in rows), default=None)
    logits_within_atol = count_match and all(item["logit_within_atol"] for item in rows)
    return {
        "count": len(reference_top),
        "count_match": count_match,
        "token_ids_match": token_ids_match,
        "max_abs_logit_diff": max_abs_diff,
        "logit_atol": logit_atol,
        "logits_within_atol": logits_within_atol,
        "rows": rows,
    }


def output_status_accepted(value: Any) -> bool:
    return value in ("ok", "not_evaluated")


def compare_case(
    reference_summary_path: Path,
    candidate_summary_path: Path,
    reference_case: dict[str, Any],
    candidate_case: dict[str, Any],
    reference_label: str,
    candidate_label: str,
    logit_atol: float,
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
    reference_text = decoded_text(reference_report, "generated", f"{reference_label}.{case_id}")
    candidate_text = decoded_text(candidate_report, "generated", f"{candidate_label}.{case_id}")
    reference_text_without_stop = decoded_text(
        reference_report,
        "generated_without_stop_sequence",
        f"{reference_label}.{case_id}",
    )
    candidate_text_without_stop = decoded_text(
        candidate_report,
        "generated_without_stop_sequence",
        f"{candidate_label}.{case_id}",
    )
    prompt_match = reference_prompt == candidate_prompt
    generated_match = reference_generated == candidate_generated
    generated_text_match = reference_text == candidate_text
    generated_text_without_stop_match = reference_text_without_stop == candidate_text_without_stop
    candidate_generated_text_nonempty = bool(
        (candidate_text_without_stop or candidate_text).strip()
    )
    reference_stop = stop_signature(reference_report, f"{reference_label}.{case_id}")
    candidate_stop = stop_signature(candidate_report, f"{candidate_label}.{case_id}")
    stop_match = reference_stop == candidate_stop
    prefill_top_logits = compare_top_logits(
        nested_top_logits(reference_report, ("prefill", "top_logits"), f"{reference_label}.{case_id}"),
        nested_top_logits(candidate_report, ("prefill", "top_logits"), f"{candidate_label}.{case_id}"),
        logit_atol,
    )
    decode_last_top_logits = compare_top_logits(
        nested_top_logits(reference_report, ("decode", "last_top_logits"), f"{reference_label}.{case_id}"),
        nested_top_logits(candidate_report, ("decode", "last_top_logits"), f"{candidate_label}.{case_id}"),
        logit_atol,
    )

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
        "reference_generated_text_chars": len(reference_text),
        "candidate_generated_text_chars": len(candidate_text),
        "generated_text_match": generated_text_match,
        "generated_text_sha256": text_hash(reference_text) if generated_text_match else None,
        "reference_generated_without_stop_text_chars": len(reference_text_without_stop),
        "candidate_generated_without_stop_text_chars": len(candidate_text_without_stop),
        "generated_without_stop_text_match": generated_text_without_stop_match,
        "generated_without_stop_text_sha256": (
            text_hash(reference_text_without_stop) if generated_text_without_stop_match else None
        ),
        "stop_match": stop_match,
        "reference_stop": reference_stop,
        "candidate_stop": candidate_stop,
        "reference_verified": reference_report.get("verified"),
        "candidate_verified": candidate_report.get("verified"),
        "both_verified": reference_report.get("verified") is True and candidate_report.get("verified") is True,
        "reference_output_status": reference_case.get("output_status"),
        "candidate_output_status": candidate_case.get("output_status"),
        "output_status_match": reference_case.get("output_status") == candidate_case.get("output_status"),
        "candidate_output_status_accepted": output_status_accepted(candidate_case.get("output_status")),
        "candidate_generated_text_nonempty": candidate_generated_text_nonempty,
        "prefill_top_logits": prefill_top_logits,
        "decode_last_top_logits": decode_last_top_logits,
        "top_logits_match": prefill_top_logits["token_ids_match"]
        and prefill_top_logits["logits_within_atol"]
        and decode_last_top_logits["token_ids_match"]
        and decode_last_top_logits["logits_within_atol"],
        "behavioral_accept": prompt_match
        and candidate_report.get("verified") is True
        and candidate_generated_text_nonempty
        and output_status_accepted(candidate_case.get("output_status")),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    acceptance_mode = getattr(args, "acceptance_mode", "strict")
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
            args.logit_atol,
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
        "generated_text_match_count": sum(1 for item in case_reports if item["generated_text_match"]),
        "generated_without_stop_text_match_count": sum(
            1 for item in case_reports if item["generated_without_stop_text_match"]
        ),
        "stop_match_count": sum(1 for item in case_reports if item["stop_match"]),
        "both_verified_count": sum(1 for item in case_reports if item["both_verified"]),
        "output_status_match_count": sum(1 for item in case_reports if item["output_status_match"]),
        "candidate_verified_count": sum(1 for item in case_reports if item["candidate_verified"] is True),
        "candidate_output_status_accepted_count": sum(
            1 for item in case_reports if item["candidate_output_status_accepted"]
        ),
        "candidate_generated_text_nonempty_count": sum(
            1 for item in case_reports if item["candidate_generated_text_nonempty"]
        ),
        "behavioral_accept_count": sum(1 for item in case_reports if item["behavioral_accept"]),
        "top_logits_match_count": sum(1 for item in case_reports if item["top_logits_match"]),
        "max_prefill_top_logit_abs_diff": max(
            (
                item["prefill_top_logits"]["max_abs_logit_diff"]
                for item in case_reports
                if item["prefill_top_logits"]["max_abs_logit_diff"] is not None
            ),
            default=None,
        ),
        "max_decode_last_top_logit_abs_diff": max(
            (
                item["decode_last_top_logits"]["max_abs_logit_diff"]
                for item in case_reports
                if item["decode_last_top_logits"]["max_abs_logit_diff"] is not None
            ),
            default=None,
        ),
        "logit_atol": args.logit_atol,
        "acceptance_mode": acceptance_mode,
    }
    strict_passed = (
        not missing_in_candidate
        and not extra_in_candidate
        and all(item["prompt_tokens_match"] for item in case_reports)
        and all(item["generated_tokens_match"] for item in case_reports)
        and all(item["generated_text_match"] for item in case_reports)
        and all(item["generated_without_stop_text_match"] for item in case_reports)
        and all(item["stop_match"] for item in case_reports)
        and all(item["both_verified"] for item in case_reports)
        and all(item["top_logits_match"] for item in case_reports)
    )
    behavioral_passed = (
        not missing_in_candidate
        and not extra_in_candidate
        and all(item["behavioral_accept"] for item in case_reports)
    )
    metrics["strict_passed"] = strict_passed
    metrics["behavioral_passed"] = behavioral_passed
    metrics["passed"] = strict_passed if acceptance_mode == "strict" else behavioral_passed
    return {
        "schema_version": "package-token-prompt-suite-generated-text-guard-v0.3",
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
        "# Package Prompt Suite Generated-Text Guard",
        "",
    ]
    if json_path is not None:
        lines.append(f"- JSON: `{json_path}`")
    lines.extend(
        [
            f"- Reference: `{report['reference']['label']}`",
            f"- Candidate: `{report['candidate']['label']}`",
            f"- Acceptance mode: `{metrics.get('acceptance_mode')}`",
            f"- Passed: `{fmt_bool(metrics.get('passed'))}`",
            f"- Strict passed: `{fmt_bool(metrics.get('strict_passed'))}`",
            f"- Behavioral passed: `{fmt_bool(metrics.get('behavioral_passed'))}`",
            f"- Compared cases: `{metrics.get('compared_case_count')}`",
            "",
            "| case | category | prompt match | token match | text match | no-stop text match | logits match | behavioral accept | output status | generated tokens | token sha256 | text sha256 |",
            "| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | --- | ---: | --- | --- |",
        ]
    )
    for item in report["cases"]:
        lines.append(
            "| {case} | {category} | {prompt} | {tokens_match} | {text_match} | {no_stop_text_match} | {logits} | {behavioral} | {status} | {tokens} | {token_sha} | {text_sha} |".format(
                case=item["id"],
                category=item.get("category", ""),
                prompt=fmt_bool(item.get("prompt_tokens_match")),
                tokens_match=fmt_bool(item.get("generated_tokens_match")),
                text_match=fmt_bool(item.get("generated_text_match")),
                no_stop_text_match=fmt_bool(item.get("generated_without_stop_text_match")),
                logits=fmt_bool(item.get("top_logits_match")),
                behavioral=fmt_bool(item.get("behavioral_accept")),
                status=item.get("candidate_output_status"),
                tokens=item.get("reference_generated_tokens"),
                token_sha=(item.get("generated_token_sha256") or "")[:16],
                text_sha=(item.get("generated_without_stop_text_sha256") or "")[:16],
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
