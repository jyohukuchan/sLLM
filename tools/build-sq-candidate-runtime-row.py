#!/usr/bin/env python3
"""Build one sq-candidate-runtime-result-v0.1 JSONL row from suite artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ullm_format_ids import FORMAT_AQ4_0, canonical_or_original, is_legacy_alias


SCHEMA_VERSION = "sq-candidate-runtime-result-v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one sq-candidate-runtime-result-v0.1 row from a prompt-suite summary."
    )
    parser.add_argument("--suite-summary", required=True, type=Path)
    parser.add_argument("--guard-bundle", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--format-version", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--package-or-runtime-artifact", required=True)
    parser.add_argument("--source-aq-policy", default="qwen35_9b_p4p46_hidden3994_v1")
    parser.add_argument("--row-scale-override-policy", default="preserved")
    parser.add_argument("--host", default="WRX80")
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--golden-prefix-artifact")
    parser.add_argument("--golden-prefix-verified", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compact-resident-bytes", type=parse_optional_int)
    parser.add_argument("--materialized-working-set-bytes", type=parse_optional_int)
    parser.add_argument("--materialization-granularity", required=True)
    parser.add_argument("--materialization-wall-ms", type=parse_optional_float)
    parser.add_argument("--whole-model-f32-resident", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--baseline-anchor", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--note", action="append", default=[])
    return parser.parse_args()


def parse_optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null", "unknown"}:
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_optional_float(value: str) -> float | None:
    if value.lower() in {"none", "null", "unknown"}:
        return None
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


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


def load_case_report(summary_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    cases = summary.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("suite summary has no cases")
    first = cases[0]
    if not isinstance(first, dict):
        raise SystemExit("suite summary first case must be an object")
    raw_report = first.get("report")
    if not isinstance(raw_report, str) or not raw_report.strip():
        raise SystemExit("suite summary first case has no report path")
    report_path = Path(raw_report)
    if not report_path.exists():
        report_path = summary_path.parent / report_path.name
    return load_json_object(report_path, "case report")


def metric(summary: dict[str, Any], key: str) -> Any:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return metrics.get(key)


def prompt_suite_token_logits_check(guard_bundle: dict[str, Any]) -> dict[str, Any] | None:
    checks = guard_bundle.get("checks")
    if not isinstance(checks, list):
        return None
    for check in checks:
        if isinstance(check, dict) and check.get("name") == "prompt_suite_token_logits":
            return check
    return None


def prompt_suite_regression_status(check: dict[str, Any] | None) -> str:
    if check is None:
        return "not_attached"
    return "passed" if check.get("passed") is True else "failed"


def prompt_suite_token_logits_metrics(check: dict[str, Any] | None) -> dict[str, Any]:
    if check is None:
        return {}
    metrics = check.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def case_p50_mean(summary: dict[str, Any]) -> float | None:
    cases = summary.get("cases")
    if not isinstance(cases, list):
        return None
    values = []
    for case in cases:
        if isinstance(case, dict) and case.get("p50_ms") is not None:
            values.append(float(case["p50_ms"]))
    return mean(values)


def max_kv_cache_bytes(summary_path: Path, summary: dict[str, Any]) -> int | None:
    cases = summary.get("cases")
    if not isinstance(cases, list):
        return None
    values: list[int] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        raw_report = case.get("report")
        if not isinstance(raw_report, str):
            continue
        report_path = Path(raw_report)
        if not report_path.exists():
            report_path = summary_path.parent / report_path.name
        report = load_json_object(report_path, f"case report {case.get('id')}")
        memory = report.get("memory")
        if isinstance(memory, dict) and memory.get("kv_cache_bytes") is not None:
            values.append(int(memory["kv_cache_bytes"]))
    return max(values) if values else None


def comparable(args: argparse.Namespace, summary: dict[str, Any], guard: dict[str, Any]) -> tuple[bool, str]:
    if args.baseline_anchor:
        return True, "baseline anchor row for the current AQ4 prototype"
    missing = []
    if args.compact_resident_bytes is None:
        missing.append("storage.compact_resident_bytes")
    if args.materialized_working_set_bytes is None:
        missing.append("storage.materialized_working_set_bytes")
    if args.materialization_wall_ms is None:
        missing.append("timing.materialization_wall_ms")
    if metric(summary, "verified_all") is not True:
        missing.append("quality.verified_all")
    if guard.get("passed") is not True:
        missing.append("guards.prompt_guard_bundle.passed")
    if missing:
        return False, "missing or failed gate fields: " + ", ".join(missing)
    return True, "all required comparison gates are present"


def build_row(args: argparse.Namespace) -> dict[str, Any]:
    summary = load_json_object(args.suite_summary, "suite summary")
    guard = load_json_object(args.guard_bundle, "guard bundle")
    first_report = load_case_report(args.suite_summary, summary)
    is_comparable, reason = comparable(args, summary, guard)
    prompt_suite_check = prompt_suite_token_logits_check(guard)
    prompt_suite_metrics = prompt_suite_token_logits_metrics(prompt_suite_check)
    suite = summary.get("suite") if isinstance(summary.get("suite"), dict) else {}
    tokenizer_dir = summary.get("tokenizer_dir")
    candidate_id = canonical_or_original(args.candidate_id)
    format_version = canonical_or_original(args.format_version)
    legacy_candidate_id = args.candidate_id if args.candidate_id != candidate_id else None
    legacy_format_version = args.format_version if args.format_version != format_version else None
    row = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "case_id": args.case_id,
        "status": "ok",
        "candidate": {
            "id": candidate_id,
            "format_version": format_version,
            "description": args.description,
            "package_or_runtime_artifact": args.package_or_runtime_artifact,
            "source_aq_policy": args.source_aq_policy,
            "row_scale_override_policy": args.row_scale_override_policy,
        },
        "model": {
            "name": "Qwen3.5-9B",
            "format": "ullm-package",
            "tokenizer": tokenizer_dir,
        },
        "hardware": {
            "host": args.host,
            "device_index": summary.get("device_index"),
            "gpu_name": args.gpu_name,
            "architecture": args.architecture,
            "backend": first_report.get("backend"),
        },
        "workload": {
            "suite": suite.get("suite_id", str(args.suite_summary)),
            "suite_artifact": str(args.suite_summary),
            "batch_size": 1,
            "tensor_parallel": 1,
            "sampling": "greedy",
            "kv_cache_dtype": first_report.get("memory", {}).get("kv_cache_value_dtype"),
        },
        "storage": {
            "compact_resident_bytes": args.compact_resident_bytes,
            "materialized_working_set_bytes": args.materialized_working_set_bytes,
            "materialization_granularity": args.materialization_granularity,
            "whole_model_f32_resident": args.whole_model_f32_resident,
            "kv_cache_bytes": max_kv_cache_bytes(args.suite_summary, summary),
        },
        "timing": {
            "materialization_wall_ms": args.materialization_wall_ms,
            "prefill_tps_mean": metric(summary, "prefill_tps_mean"),
            "decode_tps_mean": metric(summary, "decode_tps_mean"),
            "decode_tps_min": metric(summary, "decode_tps_min"),
            "decode_tps_max": metric(summary, "decode_tps_max"),
            "decode_p50_ms_mean": case_p50_mean(summary),
        },
        "quality": {
            "output_ok_count": metric(summary, "output_ok_count"),
            "output_warn_count": metric(summary, "output_warn_count"),
            "output_not_evaluated_count": metric(summary, "output_not_evaluated_count"),
            "verified_all": metric(summary, "verified_all"),
            "prompt_suite_regression_status": prompt_suite_regression_status(prompt_suite_check),
        },
        "guards": {
            "golden_prefix": {
                "status": "ok" if args.golden_prefix_artifact else "not_attached",
                "artifact": args.golden_prefix_artifact,
                "verified": args.golden_prefix_verified if args.golden_prefix_artifact else None,
            },
            "prompt_guard_bundle": {
                "status": "ok",
                "artifact": str(args.guard_bundle),
                "passed": guard.get("passed"),
                "acceptance_mode": prompt_suite_metrics.get("acceptance_mode"),
                "strict_passed": prompt_suite_metrics.get("strict_passed"),
                "behavioral_passed": prompt_suite_metrics.get("behavioral_passed"),
                "compared_case_count": prompt_suite_metrics.get("compared_case_count"),
                "generated_token_match_count": prompt_suite_metrics.get(
                    "generated_token_match_count"
                ),
                "generated_text_match_count": prompt_suite_metrics.get("generated_text_match_count"),
                "generated_without_stop_text_match_count": prompt_suite_metrics.get(
                    "generated_without_stop_text_match_count"
                ),
                "top_logits_match_count": prompt_suite_metrics.get("top_logits_match_count"),
                "max_prefill_top_logit_abs_diff": prompt_suite_metrics.get(
                    "max_prefill_top_logit_abs_diff"
                ),
                "max_decode_last_top_logit_abs_diff": prompt_suite_metrics.get(
                    "max_decode_last_top_logit_abs_diff"
                ),
            },
            "external_logits": {
                "status": "deferred",
                "artifact": None,
                "passed": None,
            },
        },
        "artifacts": {
            "suite_summary_json": str(args.suite_summary),
            "suite_summary_md": str(args.suite_summary.with_suffix(".md")),
            "guard_bundle_json": str(args.guard_bundle),
            "command_log": None,
        },
        "baseline": {
            "id": "aq4-rdna-prototype-2026-07-06",
            "r9700_decode_tps_mean": 19.796,
            "v620_decode_tps_mean": 15.434,
            "guard_bundle_artifact": "benchmarks/results/2026-07-06/engine/prompt-suite-aq4-pagedattn-r9700-v620-v0.3-guard-bundle/guard-bundle-summary.json",
        },
        "decision": {
            "comparable_to_baseline": is_comparable,
            "accepted_for_next_iteration": args.baseline_anchor,
            "reason": reason,
        },
        "notes": args.note,
    }
    if legacy_candidate_id is not None:
        row["candidate"]["legacy_candidate_id"] = legacy_candidate_id
    if legacy_format_version is not None:
        row["candidate"]["legacy_format_version"] = legacy_format_version
    if is_legacy_alias(args.candidate_id, FORMAT_AQ4_0):
        row["baseline"]["legacy_id"] = args.candidate_id
    return row


def main() -> int:
    args = parse_args()
    row = build_row(args)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with args.output_jsonl.open(mode, encoding="utf-8") as output_file:
        output_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
