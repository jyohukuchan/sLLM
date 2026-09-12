#!/usr/bin/env python3
"""Compact before/after summary for Phase85 body and MTP inference runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

SUMMARY_SCHEMA = "phase85-inference-summary-v1"


def stats(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    middle = float(statistics.median(values))
    return {"median": middle, "mad": float(statistics.median(abs(x - middle) for x in values)), "count": len(values)}


def sha(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(samples: list[dict[str, Any]], key: str) -> dict[str, float | int] | None:
    return stats([float(sample[key]) for sample in samples if isinstance(sample.get(key), (int, float))])


def infer_format(case_id: str, label: str | None = None) -> str:
    if label in {"off", "bf16", "mxfp8", "mxfp6"}:
        return label
    # The first format names the weights; a later MXFP8 token can name KV.
    match = re.search(r"(?:^|-)(mxfp8|mxfp6|bf16)(?:-|$)", case_id)
    return match.group(1) if match else "unknown"


def load_attempt(path: Path, kind: str) -> tuple[str | None, dict[str, Any] | None, dict[str, Any]]:
    attempt: dict[str, Any] = {"path": str(path), "source_report_sha256": None, "status": "failed"}
    try:
        raw = path.read_bytes()
        attempt["source_report_sha256"] = hashlib.sha256(raw).hexdigest()
        if not raw.strip():
            attempt["reason"] = "empty_stdout"
            return None, None, attempt
        report = json.loads(raw)
    except OSError as error:
        attempt["reason"] = f"read_error:{error}"
        return None, None, attempt
    except json.JSONDecodeError as error:
        attempt["reason"] = f"invalid_json:{error.msg}"
        return None, None, attempt
    if not isinstance(report, dict) or report.get("state") != "PASS":
        attempt["reason"] = f"report_state:{report.get('state') if isinstance(report, dict) else None!r}"
        return None, None, attempt
    label = path.parent.name
    if kind == "body":
        case = report.get("row", {}).get("case_id")
        key = case
    else:
        rows = report.get("rows")
        case = rows[0].get("row_scope") if isinstance(rows, list) and len(rows) == 1 else None
        key = f"{label}:{case}" if case else None
    if not isinstance(case, str) or not key:
        attempt["reason"] = "case_id_missing_or_multiple_rows"
        return None, None, attempt
    attempt["case_id"] = key
    attempt["label"] = label
    attempt["target"] = report.get("target") or report.get("identities", {}).get("target")
    try:
        parsed = parse_body(report, key, label) if kind == "body" else parse_mtp(report, key, label)
    except (KeyError, TypeError, ValueError) as error:
        attempt["reason"] = f"contract:{error}"
        return key, None, attempt
    attempt["status"] = "valid" if not parsed["issues"] else "failed"
    if parsed["issues"]:
        attempt["reason"] = ",".join(parsed["issues"])
        return key, None, attempt
    return key, parsed, attempt


def common_checks(report: dict[str, Any], target: str) -> list[str]:
    issues: list[str] = []
    observed = report.get("target") or report.get("identities", {}).get("target")
    if observed != target:
        issues.append(f"target_mismatch:{observed!r}")
    audits = [report.get("audit", {})]
    if report.get("rows"):
        runs = report["rows"][0].get("runs", [])
        audits = [run.get("audit", {}) for run in runs]
        if report.get("mtp", {}).get("requested") is True:
            for run in runs:
                draft = run.get("mtp", {})
                if draft.get("draft_fallback_used") is not False:
                    issues.append("draft_fallback_or_evidence_missing")
                if draft.get("draft_all_dispatches_hip") is not True:
                    issues.append("draft_not_hip_or_evidence_missing")
                if not isinstance(draft.get("draft_kernel_dispatch_count"), int) or draft["draft_kernel_dispatch_count"] <= 0:
                    issues.append("draft_zero_or_missing_dispatch")
    else:
        session_cleanup = report.get("session_cleanup", {})
        if session_cleanup.get("retryable_cleanup") != 0 or session_cleanup.get("durable_quarantine") != 0:
            issues.append("session_cleanup_nonzero_or_missing")
    if not audits or any(item.get("selected_backend") != "hip" for item in audits):
        issues.append("backend_not_hip")
    if any(item.get("fallback_used") is not False for item in audits):
        issues.append("fallback_used")
    if any(item.get("all_dispatches_hip") is not True for item in audits):
        issues.append("non_hip_dispatch")
    if any(not isinstance(item.get("kernel_dispatch_count"), int) or item["kernel_dispatch_count"] <= 0 for item in audits):
        issues.append("zero_dispatch")
    cleanup = report.get("cleanup", {})
    if cleanup.get("retryable_cleanup") not in (None, 0):
        issues.append("retryable_cleanup_nonzero")
    if cleanup.get("durable_quarantine") not in (None, 0):
        issues.append("durable_quarantine_nonzero")
    if cleanup.get("zero") is False:
        issues.append("cleanup_zero_flag_false")
    if cleanup.get("all_requests_dropped") is False:
        issues.append("request_cleanup_incomplete")
    return issues


def parse_body(report: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    issues = common_checks(report, report.get("identities", {}).get("target"))
    row, config, measured = report["row"], report["config"], report["measured"]
    samples = [sample.get("derived", {}) for sample in measured.get("samples", [])]
    if not samples or measured.get("count") != len(samples):
        issues.append("missing_or_zero_measured_samples")
    kv = config.get("kv_cache_selection", {}).get("resolved", config.get("kv_cache_encoding"))
    input_hash = sha(row.get("input_token_ids", config.get("input_token_ids", [])))
    output_hashes = [sha(sample.get("tokens", {}).get("generated_token_ids", [])) for sample in measured.get("samples", [])]
    output_counts = [
        len(sample.get("tokens", {}).get("generated_token_ids", []))
        for sample in measured.get("samples", [])
        if sample.get("tokens", {}).get("generated_token_ids") is not None
    ]
    if not output_counts:
        output_counts = [int(sample.get("decode_tokens", 0)) for sample in samples if sample.get("decode_tokens") is not None]
    early_eos = any(count < int(config.get("max_new_tokens", count)) for count in output_counts)
    identity = {
        "model": report.get("identities", {}).get("model"),
        "model_fingerprint": report.get("identities", {}).get("binding", {}).get("model_fingerprint"),
        "target": report.get("identities", {}).get("target"),
        "input_token_count": config.get("input_token_count"),
        "input_fixture_sha256": input_hash,
        "sampling": {"greedy": config.get("greedy"), "max_new_tokens": config.get("max_new_tokens"), "stop_policy": config.get("stop_policy")},
        "kv": kv,
        "format": infer_format(key),
        "warmups": config.get("warmups"),
        "measured": config.get("measured"),
        "execution_config": {name: config.get(name) for name in ("lane", "effective_context_length", "effective_prefill_chunk_tokens")},
    }
    metrics = {name: metric(samples, name) for name in ("prefill_ns", "ttft_ns", "e2e_ns", "decode_tokens_per_second", "prefill_tokens_per_second")}
    metrics["output_tokens"] = stats([float(x) for x in output_counts])
    return {"issues": issues, "key": key, "format": identity["format"], "identity": identity, "metrics": metrics, "output_hashes": output_hashes, "numeric_change": False, "early_eos": early_eos, "memory": {k: report.get("memory", {}).get(k) for k in ("resident_vram_bytes", "peak_vram_bytes")}, "provider": report.get("identities", {}).get("engine")}


def parse_mtp(report: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    issues = common_checks(report, report.get("target"))
    row = report["rows"][0]
    summary = row.get("measured_summary", {})
    measured_runs = [run for run in row.get("runs", []) if run.get("sample_kind") == "measured"]
    if not measured_runs or summary.get("measured_samples") != len(measured_runs):
        issues.append("missing_or_zero_measured_samples")
    fixture = report.get("fixture", {})
    protocol = report.get("protocol", {})
    mtp = report.get("mtp", {})
    output_hashes = [run.get("generated_tokens_sha256") for run in measured_runs]
    output_counts = [run.get("decoded_token_count") for run in measured_runs if isinstance(run.get("decoded_token_count"), int)]
    early_eos = any(count < row.get("output_tokens", count) for count in output_counts)
    identity = {
        "model": report.get("model", {"revision": report.get("model", {}).get("revision")}),
        "target": report.get("target"),
        "fixture_sha256": fixture.get("sha256"),
        "prompt_prefix_sha256": row.get("prompt_prefix_sha256"),
        "sampling": report.get("sampling"),
        "kv": protocol.get("kv_cache"),
        "format": label,
        "companion_encoding": mtp.get("companion_encoding"),
        "companion_digest": mtp.get("companion_digest"),
        "warmups": report.get("repetitions", {}).get("warmups_per_row"),
        "measured": report.get("repetitions", {}).get("measured_per_row"),
        "draft_width": mtp.get("draft_width"),
        "execution_config": {"protocol": protocol, "selector_environment": report.get("selector_environment")},
    }
    names = ("request_setup_ms", "prefill_ms", "ttft_ms", "decode_ms", "e2e_ms", "tpot_ms", "prefill_tokens_per_second", "decode_tokens_per_second")
    metrics = {name: summary.get(name) for name in names}
    metrics["output_tokens"] = stats([float(x) for x in output_counts])
    inner: dict[str, Any] = {}
    for name in ("proposal_blocks", "proposed_draft_tokens", "accepted_draft_tokens", "rejected_draft_tokens", "committed_target_rows", "mtp_prefix_priming_wall_ns", "mtp_decode_proposal_wall_ns"):
        values = [run.get("mtp", {}).get(name) for run in measured_runs if isinstance(run.get("mtp", {}).get(name), (int, float))]
        inner[name] = stats([float(x) for x in values])
    inner["verify"] = {"reported": False, "source": "no separate verify duration in report"}
    inner["acceptance_rate"] = (sum(run.get("mtp", {}).get("accepted_draft_tokens", 0) for run in measured_runs) / sum(run.get("mtp", {}).get("proposed_draft_tokens", 0) for run in measured_runs) if sum(run.get("mtp", {}).get("proposed_draft_tokens", 0) for run in measured_runs) else None)
    metrics["mtp"] = inner
    ready = report.get("resident_ready_memory", {})
    allocations = [ready, report.get("cleanup", {}).get("allocation_after_resident_drop_before_shutdown", {})]
    allocations.extend(run.get("allocation_while_request_alive", {}) for run in row.get("runs", []))
    high_water = [item["high_water_bytes"] for item in allocations if isinstance(item.get("high_water_bytes"), int)]
    return {"issues": issues, "key": key, "format": label, "identity": identity, "metrics": metrics, "output_hashes": output_hashes, "numeric_change": False, "early_eos": early_eos, "memory": {"resident_vram_bytes": ready.get("model_resident", {}).get("current_bytes"), "peak_vram_bytes": max(high_water) if high_water else None, "peak_vram_source": "cumulative execution-session allocation accounting"}, "provider": "sllm"}


def classify(base: dict[str, float | int] | None, cand: dict[str, float | int] | None, higher: bool) -> dict[str, Any] | None:
    if not base or not cand:
        return None
    b, c, spread = float(base["median"]), float(cand["median"]), max(float(base["mad"]), float(cand["mad"]))
    improved = c > b + spread if higher else c < b - spread
    regressed = c < b - spread if higher else c > b + spread
    return {"baseline": base, "candidate": cand, "candidate_over_baseline": c / b if b else None, "classification": "improved" if improved else "regressed" if regressed else "within_variation"}


def collect(root: Path, kind: str, target: str) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    valid: dict[str, dict[str, Any]] = {}
    attempts: list[dict[str, Any]] = []
    paths = sorted(root.glob("*/stdout.json")) if root.is_dir() else [root]
    for path in paths:
        key, parsed, attempt = load_attempt(path, kind)
        if key and key in valid:
            attempt.update(status="failed", reason="duplicate_case_id")
            parsed = None
        if parsed is not None and parsed["issues"]:
            attempt.update(status="failed", reason=",".join(parsed["issues"]))
            parsed = None
        if parsed is not None:
            if parsed["identity"].get("target") != target:
                attempt.update(status="failed", reason="target_mismatch")
                parsed = None
            else:
                valid[key] = parsed
        attempts.append(attempt)
    if not paths:
        attempts.append({"path": str(root), "status": "failed", "reason": "root_missing_or_no_stdout"})
    return valid, attempts


def summarize(kind: str, baseline_root: Path, candidate_root: Path) -> dict[str, Any]:
    target_hint = next(
        (
            part
            for name in (baseline_root.name, candidate_root.name)
            for part in name.split("-")
            if part.startswith("gfx")
        ),
        None,
    )
    target = target_hint or "unknown"
    baseline, base_attempts = collect(baseline_root, kind, target)
    candidate, cand_attempts = collect(candidate_root, kind, target)
    keys = sorted(set(baseline) | set(candidate))
    rows: list[dict[str, Any]] = []
    for key in keys:
        b, c = baseline.get(key), candidate.get(key)
        row: dict[str, Any] = {"case_id": key, "format": (b or c)["format"], "matched": bool(b and c), "status": "missing_baseline" if not b else "missing_candidate" if not c else "matched"}
        if not b or not c:
            row["classification"] = "unmeasured"
            rows.append(row)
            continue
        identity_fields = ("model", "model_fingerprint", "target", "input_token_count", "input_fixture_sha256", "fixture_sha256", "prompt_prefix_sha256", "sampling", "kv", "format", "companion_encoding", "companion_digest", "warmups", "measured", "draft_width", "execution_config")
        mismatches = [field for field in identity_fields if b["identity"].get(field) != c["identity"].get(field)]
        if mismatches:
            row.update(status="identity_mismatch", mismatches=mismatches, identity={"baseline": b["identity"], "candidate": c["identity"]}, classification="unmeasured")
            rows.append(row)
            continue
        row["identity"] = {"baseline": b["identity"], "candidate": c["identity"]}
        row["numeric_change"] = sorted(set(b["output_hashes"])) != sorted(set(c["output_hashes"]))
        row["output_digest_equal"] = not row["numeric_change"]
        row["output_digest"] = {"baseline": sorted(set(b["output_hashes"])), "candidate": sorted(set(c["output_hashes"]))}
        row["early_eos"] = {"baseline": b["early_eos"], "candidate": c["early_eos"]}
        row["memory"] = {"baseline": b["memory"], "candidate": c["memory"]}
        row["performance"] = {}
        timing_names = ("prefill_ns", "ttft_ns", "e2e_ns") if kind == "body" else ("request_setup_ms", "prefill_ms", "ttft_ms", "decode_ms", "e2e_ms", "tpot_ms")
        for name in (*timing_names, "decode_tokens_per_second", "prefill_tokens_per_second", "output_tokens"):
            result = classify(b["metrics"].get(name), c["metrics"].get(name), "tokens_per_second" in name)
            if result is not None:
                row["performance"][name] = result
        if kind == "mtp":
            row["mtp_inner"] = {"baseline": b["metrics"].get("mtp"), "candidate": c["metrics"].get("mtp")}
        row["classification"] = "numeric_change" if row["numeric_change"] else "matched"
        rows.append(row)
    comparable = [row for row in rows if row["status"] == "matched"]
    return {"schema_version": SUMMARY_SCHEMA, "state": "COMPLETE" if keys and len(comparable) == len(keys) else "PARTIAL", "kind": kind, "target": target, "baseline_root": str(baseline_root), "candidate_root": str(candidate_root), "expected_case_count": len(keys), "comparable_case_count": len(comparable), "missing_case_ids": [row["case_id"] for row in rows if row["status"].startswith("missing")], "attempts": base_attempts + cand_attempts, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("body", "mtp"), required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = summarize(args.kind, args.baseline_root, args.candidate_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"phase85 inference summary failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"state": report["state"], "comparable_case_count": report["comparable_case_count"], "expected_case_count": report["expected_case_count"]}, sort_keys=True))
    return 0 if report["state"] == "COMPLETE" or args.allow_partial else 3


if __name__ == "__main__":
    raise SystemExit(main())
