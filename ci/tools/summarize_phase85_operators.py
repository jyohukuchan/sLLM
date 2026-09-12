#!/usr/bin/env python3
"""Summarize Phase85 per-shape MXFP operator evidence.

The runner writes one compact JSON report below ``ROOT/CASE_ID/stdout.json``.
This tool keeps failed and incomplete attempts in the output while comparing
only reports that satisfy the shape, dispatch, oracle, repeat, and resource
contracts.  It intentionally does not make a fixed performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "phase85-mxfp-shapes-v1"
SUMMARY_SCHEMA_VERSION = "phase85-mxfp-operator-summary-v1"
EXPECTED_CASE_COUNT = 87
FORMAT_PREFIX = {
    "mxfp8": "mxfp8-",
    "mxfp6": "mxfp6-",
}


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label} is not finite and positive")
    return result


def _integer(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} is not an integer")
    if positive and value <= 0:
        raise ValueError(f"{label} is not positive")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _median_mad(values: Iterable[float]) -> dict[str, float | int]:
    ordered = list(values)
    if not ordered:
        raise ValueError("empty timing sample")
    center = float(statistics.median(ordered))
    return {
        "median_ns": center,
        "mad_ns": float(statistics.median(abs(value - center) for value in ordered)),
        "sample_count": len(ordered),
    }


def _geometric_mean(values: Iterable[float]) -> float | None:
    checked = [value for value in values if math.isfinite(value) and value > 0]
    if not checked:
        return None
    return math.exp(sum(math.log(value) for value in checked) / len(checked))


def _load_json(path: Path) -> tuple[Any | None, str | None, str]:
    try:
        data = path.read_bytes()
    except OSError as error:
        return None, f"read_error:{error}", ""
    source_hash = _sha256(data)
    if not data.strip():
        return None, "empty_stdout", source_hash
    try:
        return json.loads(data), None, source_hash
    except json.JSONDecodeError as error:
        return None, f"invalid_json:{error.msg}", source_hash


def _load_manifest(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    data = path.read_bytes()
    manifest_hash = _sha256(data)
    manifest = json.loads(data)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"manifest schema is not {SCHEMA_VERSION}: {path}")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest cases must be a non-empty list")
    result: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("manifest case is not an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in result:
            raise ValueError(f"manifest case_id is empty or duplicated: {case_id!r}")
        if case.get("format") not in FORMAT_PREFIX:
            raise ValueError(f"manifest case {case_id} has an unsupported format")
        for key in ("m", "k", "n"):
            _integer(case.get(key), f"manifest {case_id}.{key}", positive=True)
        result[case_id] = case
    return result, manifest_hash


def _report_level_error(report: dict[str, Any], target: str) -> str | None:
    if report.get("schema_version") != SCHEMA_VERSION:
        return "schema_mismatch"
    if report.get("state") != "PASS":
        return f"report_state:{report.get('state')!r}"
    if report.get("target") != target:
        return f"target_mismatch:{report.get('target')!r}"
    if report.get("fallback_allowed") is not False:
        return "fallback_allowed"
    if report.get("fallback_used") is not False:
        return "fallback_used"
    if report.get("retryable_cleanup") != 0:
        return "retryable_cleanup_nonzero"
    if report.get("durable_quarantine") != 0:
        return "durable_quarantine_nonzero"
    if not isinstance(report.get("cases"), list):
        return "cases_missing_or_not_list"
    if not report["cases"]:
        return "zero_cases"
    return None


def _case_attempt(
    case: dict[str, Any],
    spec: dict[str, Any],
    report: dict[str, Any],
    target: str,
    source_hash: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    case_id = spec["case_id"]
    if case.get("case_id") != case_id:
        errors.append("case_id_mismatch")
    if not isinstance(case.get("format"), str) or not case["format"].startswith(
        FORMAT_PREFIX[spec["format"]]
    ):
        errors.append("format_mismatch")
    for key in ("m", "k", "n"):
        if case.get(key) != spec.get(key):
            errors.append(f"{key}_mismatch")
    if case.get("oracle_mode") != ("full" if spec["oracle"] == "full" else "boundary-sample"):
        errors.append("oracle_mode_mismatch")
    try:
        oracle_points = _integer(case.get("oracle_point_count"), f"{case_id}.oracle_point_count", positive=True)
    except ValueError as error:
        errors.append(str(error))
        oracle_points = 0
    if case.get("oracle_sampled_output_indices") is not None:
        sampled = case.get("oracle_sampled_output_indices")
        if not isinstance(sampled, list) or len(sampled) != oracle_points:
            errors.append("oracle_inspection_range_mismatch")
    try:
        repeat_count = _integer(case.get("repeat_count"), f"{case_id}.repeat_count", positive=True)
    except ValueError as error:
        errors.append(str(error))
        repeat_count = 0
    samples = case.get("repeat_kernel_elapsed_ns")
    if not isinstance(samples, list) or len(samples) != repeat_count:
        errors.append("repeat_timing_count_mismatch")
        samples = []
    try:
        timing = [_number(value, f"{case_id}.repeat_kernel_elapsed_ns", positive=True) for value in samples]
    except ValueError as error:
        errors.append(str(error))
        timing = []
    if report.get("repeat_count") != repeat_count:
        errors.append("report_case_repeat_count_mismatch")
    output_digests = case.get("repeat_output_bf16_sha256")
    output_digest = case.get("output_bf16_sha256")
    if (
        not isinstance(output_digest, str)
        or not output_digest
        or not isinstance(output_digests, list)
        or len(output_digests) != repeat_count
        or any(value != output_digest for value in output_digests)
    ):
        errors.append("output_digest_instability")
    for key in ("weight_value_sha256", "weight_scale_sha256"):
        if not isinstance(case.get(key), str) or not case[key]:
            errors.append(f"{key}_missing")
    try:
        kernel_id = _integer(case.get("kernel_id"), f"{case_id}.kernel_id", positive=True)
        actual_dispatch_count = _integer(
            case.get("actual_dispatch_count"), f"{case_id}.actual_dispatch_count", positive=True
        )
    except ValueError as error:
        errors.append(str(error))
        kernel_id = 0
        actual_dispatch_count = 0
    if actual_dispatch_count != 2:
        errors.append("dispatch_count_mismatch")
    for key in ("kernel_symbol", "device_symbol"):
        if not isinstance(case.get(key), str) or not case[key]:
            errors.append(f"{key}_missing")
    if case.get("nonfinite_mismatch_count") not in (0, None):
        errors.append("nonfinite_mismatch")
    if errors:
        return None, sorted(set(errors))
    return {
        "case_id": case_id,
        "source_report_sha256": source_hash,
        "target": target,
        "format": spec["format"],
        "m": spec["m"],
        "k": spec["k"],
        "n": spec["n"],
        "repeat_count": repeat_count,
        "timing_ns": timing,
        "kernel_id": kernel_id,
        "kernel_symbol": case["kernel_symbol"],
        "device_symbol": case["device_symbol"],
        "actual_dispatch_count": actual_dispatch_count,
        "weight_value_sha256": case["weight_value_sha256"],
        "weight_scale_sha256": case["weight_scale_sha256"],
        "output_bf16_sha256": output_digest,
        "oracle_mode": case["oracle_mode"],
        "oracle_point_count": oracle_points,
        "max_abs_error": case.get("max_abs_error"),
        "max_relative_error": case.get("max_relative_error"),
        "provider_role": report.get("provider_role"),
    }, []


def _case_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    direct = root / "stdout.json"
    if direct.is_file():
        return [direct]
    return sorted(path for path in root.glob("*/stdout.json") if path.is_file())


def _root_format_hint(root: Path) -> str | None:
    name = root.name.lower()
    if "mx6" in name or "mxfp6" in name:
        return "mxfp6"
    if "mx8" in name or "mxfp8" in name:
        return "mxfp8"
    observed = {
        "mxfp8" if path.parent.name.lower().startswith("mx8-") else "mxfp6"
        for path in root.glob("*/stdout.json")
        if path.parent.name.lower().startswith(("mx8-", "mx6-"))
    }
    if len(observed) == 1:
        return observed.pop()
    return None


def _collect_side(
    side: str,
    roots: list[Path],
    specs: dict[str, dict[str, Any]],
    target: str,
) -> dict[str, Any]:
    valid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    attempts: list[dict[str, Any]] = []
    scan: list[dict[str, Any]] = []
    seen_roots: Counter[str] = Counter()
    for root in roots:
        root_key = str(root.resolve())
        seen_roots[root_key] += 1
        files = _case_files(root)
        format_hint = _root_format_hint(root)
        expected_in_root = [
            case_id
            for case_id, spec in specs.items()
            if format_hint is None or spec["format"] == format_hint
        ]
        scan_entry: dict[str, Any] = {
            "root": str(root),
            "format_hint": format_hint,
            "file_count": len(files),
            "unexpected_case_ids": [],
        }
        emitted: Counter[str] = Counter()
        if not files:
            attempts.append({"side": side, "root": str(root), "path": str(root), "case_id": None, "status": "failed", "reason": "root_missing_or_no_stdout"})
            scan_entry["status"] = "missing"
        for path in files:
            path_case_hint = path.parent.name if path.parent.name in specs else None
            report, load_error, source_hash = _load_json(path)
            if load_error:
                if path_case_hint is not None:
                    emitted[path_case_hint] = max(emitted[path_case_hint], 1)
                attempts.append({"side": side, "root": str(root), "path": str(path), "case_id": path_case_hint, "status": "failed", "reason": load_error, "source_report_sha256": source_hash})
                continue
            if not isinstance(report, dict):
                if path_case_hint is not None:
                    emitted[path_case_hint] = max(emitted[path_case_hint], 1)
                attempts.append({"side": side, "root": str(root), "path": str(path), "case_id": path_case_hint, "status": "failed", "reason": "report_not_object", "source_report_sha256": source_hash})
                continue
            level_error = _report_level_error(report, target)
            if level_error:
                if path_case_hint is not None:
                    emitted[path_case_hint] = max(emitted[path_case_hint], 1)
                attempts.append({"side": side, "root": str(root), "path": str(path), "case_id": path_case_hint, "status": "failed", "reason": level_error, "source_report_sha256": source_hash, "report_state": report.get("state"), "report_target": report.get("target")})
                continue
            for case in report["cases"]:
                case_id = case.get("case_id") if isinstance(case, dict) else None
                if case_id not in specs:
                    scan_entry["unexpected_case_ids"].append(case_id)
                    attempts.append({"side": side, "root": str(root), "path": str(path), "case_id": case_id, "status": "failed", "reason": "unexpected_case_id", "source_report_sha256": source_hash})
                    continue
                emitted[case_id] += 1
                if emitted[case_id] > 1:
                    attempts.append({"side": side, "root": str(root), "path": str(path), "case_id": case_id, "status": "failed", "reason": "duplicate_case_id_in_root", "source_report_sha256": source_hash})
                    continue
                attempt, errors = _case_attempt(case, specs[case_id], report, target, source_hash)
                if attempt is None:
                    attempts.append({"side": side, "root": str(root), "path": str(path), "case_id": case_id, "status": "failed", "reason": "case_contract:" + ",".join(errors), "source_report_sha256": source_hash})
                    continue
                valid[case_id].append(attempt)
                attempts.append({"side": side, "root": str(root), "path": str(path), "case_id": case_id, "status": "valid", "source_report_sha256": source_hash, "repeat_count": attempt["repeat_count"]})
        for case_id in expected_in_root:
            if emitted[case_id] == 0:
                attempts.append({"side": side, "root": str(root), "path": str(root / case_id / "stdout.json"), "case_id": case_id, "status": "missing", "reason": "case_missing"})
        scan_entry["missing_case_count"] = sum(emitted[case_id] == 0 for case_id in expected_in_root)
        scan_entry["duplicate_case_ids"] = sorted(case_id for case_id, count in emitted.items() if count > 1)
        scan.append(scan_entry)
    duplicate_roots = sorted(root for root, count in seen_roots.items() if count > 1)
    duplicate_case_ids = sorted(
        {
            attempt["case_id"]
            for attempt in attempts
            if attempt.get("reason") == "duplicate_case_id_in_root" and attempt.get("case_id")
        }
    )
    return {
        "valid": valid,
        "attempts": attempts,
        "scan": scan,
        "duplicate_roots": duplicate_roots,
        "duplicate_case_ids": duplicate_case_ids,
    }


def _compare_case(
    case_id: str,
    spec: dict[str, Any],
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    baseline_duplicates: set[str],
    candidate_duplicates: set[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case_id,
        "format": spec["format"],
        "m": spec["m"],
        "k": spec["k"],
        "n": spec["n"],
        "tags": spec.get("tags", []),
        "role": spec.get("role", "matmul"),
        "oracle_expected": spec["oracle"],
        "baseline_attempt_count": len(baseline),
        "candidate_attempt_count": len(candidate),
        "status": "missing_baseline" if not baseline else "missing_candidate" if not candidate else "comparable",
    }
    if not baseline or not candidate:
        row["classification"] = "unmeasured"
        return row
    duplicate_sides = []
    if case_id in baseline_duplicates:
        duplicate_sides.append("baseline")
    if case_id in candidate_duplicates:
        duplicate_sides.append("candidate")
    if duplicate_sides:
        row["status"] = "invalid_comparison"
        row["classification"] = "unmeasured"
        row["invalid_reasons"] = [f"duplicate_case_id:{side}" for side in duplicate_sides]
        return row
    all_attempts = {"baseline": baseline, "candidate": candidate}
    side_summary: dict[str, Any] = {}
    invalid_reasons: list[str] = []
    repeat_counts: dict[str, list[int]] = {}
    identities: dict[str, list[dict[str, Any]]] = {}
    for side, attempts in all_attempts.items():
        timings = [value for attempt in attempts for value in attempt["timing_ns"]]
        side_summary[side] = _median_mad(timings)
        side_summary[side]["source_report_sha256"] = sorted({attempt["source_report_sha256"] for attempt in attempts})
        side_summary[side]["weight_value_sha256"] = sorted({attempt["weight_value_sha256"] for attempt in attempts})
        side_summary[side]["weight_scale_sha256"] = sorted({attempt["weight_scale_sha256"] for attempt in attempts})
        side_summary[side]["output_bf16_sha256"] = sorted({attempt["output_bf16_sha256"] for attempt in attempts})
        side_summary[side]["oracle_inspection"] = sorted({(attempt["oracle_mode"], attempt["oracle_point_count"]) for attempt in attempts})
        repeat_counts[side] = sorted({attempt["repeat_count"] for attempt in attempts})
        identities[side] = sorted({(attempt["kernel_id"], attempt["kernel_symbol"], attempt["device_symbol"]) for attempt in attempts})
        if len(repeat_counts[side]) != 1:
            invalid_reasons.append(f"{side}_repeat_count_variance")
        if len(identities[side]) != 1:
            invalid_reasons.append(f"{side}_dispatch_identity_variance")
        if len(side_summary[side]["weight_value_sha256"]) != 1 or len(side_summary[side]["weight_scale_sha256"]) != 1:
            invalid_reasons.append(f"{side}_weight_hash_variance")
        if len(side_summary[side]["output_bf16_sha256"]) != 1:
            invalid_reasons.append(f"{side}_output_digest_variance")
    if repeat_counts["baseline"] != repeat_counts["candidate"]:
        invalid_reasons.append("baseline_candidate_repeat_count_mismatch")
    if side_summary["baseline"]["weight_value_sha256"] != side_summary["candidate"]["weight_value_sha256"]:
        invalid_reasons.append("weight_value_hash_mismatch")
    if side_summary["baseline"]["weight_scale_sha256"] != side_summary["candidate"]["weight_scale_sha256"]:
        invalid_reasons.append("weight_scale_hash_mismatch")
    if side_summary["baseline"]["oracle_inspection"] != side_summary["candidate"]["oracle_inspection"]:
        invalid_reasons.append("oracle_inspection_range_mismatch")
    row["baseline"] = side_summary["baseline"]
    row["candidate"] = side_summary["candidate"]
    row["dispatch_identity"] = {
        "baseline": [list(identity) for identity in identities["baseline"]],
        "candidate": [list(identity) for identity in identities["candidate"]],
    }
    output_equal = side_summary["baseline"]["output_bf16_sha256"] == side_summary["candidate"]["output_bf16_sha256"]
    row["output_digest_equal"] = output_equal
    row["numeric_change"] = not output_equal
    if invalid_reasons:
        row["status"] = "invalid_comparison"
        row["invalid_reasons"] = sorted(set(invalid_reasons))
        row["classification"] = "unmeasured"
        return row
    baseline_median = side_summary["baseline"]["median_ns"]
    candidate_median = side_summary["candidate"]["median_ns"]
    spread = max(side_summary["baseline"]["mad_ns"], side_summary["candidate"]["mad_ns"])
    if candidate_median < baseline_median - spread:
        classification = "improved"
    elif candidate_median > baseline_median + spread:
        classification = "regressed"
    else:
        classification = "within_variation"
    row["speedup_baseline_over_candidate"] = baseline_median / candidate_median
    row["candidate_over_baseline_ratio"] = candidate_median / baseline_median
    row["classification"] = classification
    row["noise_threshold_ns"] = spread
    return row


def _groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "comparable":
            continue
        grouped[f"format:{row['format']}"].append(row)
        for tag in row.get("tags", []):
            grouped[f"format:{row['format']}/tag:{tag}"].append(row)
        grouped[f"format:{row['format']}/role:{row.get('role', 'matmul')}"].append(row)
    summaries = []
    for name, selected in sorted(grouped.items()):
        worst = max(selected, key=lambda item: item["candidate_over_baseline_ratio"])
        summaries.append({
            "group": name,
            "row_count": len(selected),
            "geometric_mean_speedup": _geometric_mean(row["speedup_baseline_over_candidate"] for row in selected),
            "worst_regression": {
                "case_id": worst["case_id"],
                "candidate_over_baseline_ratio": worst["candidate_over_baseline_ratio"],
                "classification": worst["classification"],
            },
            "classification_counts": dict(Counter(row["classification"] for row in selected)),
        })
    return summaries


def summarize(manifest_path: Path, target: str, baseline_roots: list[Path], candidate_roots: list[Path]) -> dict[str, Any]:
    specs, manifest_hash = _load_manifest(manifest_path)
    baseline = _collect_side("baseline", baseline_roots, specs, target)
    candidate = _collect_side("candidate", candidate_roots, specs, target)
    rows = [
        _compare_case(
            case_id,
            specs[case_id],
            baseline["valid"].get(case_id, []),
            candidate["valid"].get(case_id, []),
            set(baseline["duplicate_case_ids"]),
            set(candidate["duplicate_case_ids"]),
        )
        for case_id in specs
    ]
    comparable = [row for row in rows if row["status"] == "comparable"]
    classification_counts = Counter(row["classification"] for row in rows)
    numeric_change_count = sum(row.get("numeric_change", False) for row in comparable)
    worst = max(comparable, key=lambda row: row["candidate_over_baseline_ratio"], default=None)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "state": "COMPLETE" if len(comparable) == len(specs) else "PARTIAL",
        "target": target,
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_hash,
            "schema_version": SCHEMA_VERSION,
            "case_count": len(specs),
            "expected_case_count": EXPECTED_CASE_COUNT,
            "case_count_matches_contract": len(specs) == EXPECTED_CASE_COUNT,
        },
        "inputs": {
            "baseline_roots": [str(root) for root in baseline_roots],
            "candidate_roots": [str(root) for root in candidate_roots],
            "kernel_timing_semantics": "repeat_kernel_elapsed_ns from the runner; preserve the source report wording and do not relabel it as standalone matmul time",
        },
        "timing_metric": "reported_kernel_elapsed_ns",
        "coverage": {
            "expected_case_count": len(specs),
            "baseline_valid_case_count": sum(bool(baseline["valid"].get(case_id)) for case_id in specs),
            "candidate_valid_case_count": sum(bool(candidate["valid"].get(case_id)) for case_id in specs),
            "comparable_case_count": len(comparable),
            "missing_baseline_case_ids": [case_id for case_id in specs if not baseline["valid"].get(case_id)],
            "missing_candidate_case_ids": [case_id for case_id in specs if not candidate["valid"].get(case_id)],
            "failed_attempt_count": sum(attempt.get("status") == "failed" for attempt in baseline["attempts"] + candidate["attempts"]),
            "missing_attempt_count": sum(attempt.get("status") == "missing" for attempt in baseline["attempts"] + candidate["attempts"]),
            "duplicate_case_ids": {
                "baseline": baseline["duplicate_case_ids"],
                "candidate": candidate["duplicate_case_ids"],
            },
            "failed_attempts_preserved": True,
        },
        "classification_counts": dict(classification_counts),
        "numeric_change_count": numeric_change_count,
        "worst_regression": None if worst is None else {
            "case_id": worst["case_id"],
            "format": worst["format"],
            "candidate_over_baseline_ratio": worst["candidate_over_baseline_ratio"],
            "classification": worst["classification"],
        },
        "groups": _groups(rows),
        "rows": rows,
        "attempts": baseline["attempts"] + candidate["attempts"],
        "root_scan": {"baseline": baseline["scan"], "candidate": candidate["scan"]},
        "duplicate_roots": {"baseline": baseline["duplicate_roots"], "candidate": candidate["duplicate_roots"]},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--baseline-root", type=Path, action="append", nargs="+", required=True)
    parser.add_argument("--candidate-root", type=Path, action="append", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="return success for an incomplete measurement set; the report remains PARTIAL",
    )
    args = parser.parse_args(argv)
    baseline_roots = [root for group in args.baseline_root for root in group]
    candidate_roots = [root for group in args.candidate_root for root in group]
    try:
        report = summarize(args.manifest, args.target, baseline_roots, candidate_roots)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"phase85 operator summary failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"state": report["state"], "comparable_case_count": report["coverage"]["comparable_case_count"], "expected_case_count": report["coverage"]["expected_case_count"]}, sort_keys=True))
    if report["state"] == "PARTIAL" and not args.allow_partial:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
