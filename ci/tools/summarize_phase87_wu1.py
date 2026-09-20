#!/usr/bin/env python3
"""Create a compact, source-addressed Phase 87 WU1 report.

The operator JSONL remains the detailed evidence.  This joiner keeps oracle
metrics, performance medians/MADs, identity hashes, and the optional full
model summaries while dropping the 27-value timing arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


TARGETS = ("gfx1030", "gfx1201")
OPERATOR_EXEC_SCHEMA = "phase87-wu1-execution-v1"
RUNNER = Path(__file__).with_name("run_phase87_wu1.py")
FULLMODEL_EXEC_SCHEMA = "phase86-execution-v1"


class SummaryError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise SummaryError(message)


def digest(path: Path) -> str:
    if not path.is_file():
        fail(f"missing file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    if not path.is_file():
        fail(f"missing JSON: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        fail(f"invalid JSON {path}: {error}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        fail(f"missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            fail(f"invalid JSONL {path}:{number}: {error}")
        if not isinstance(row, dict):
            fail(f"JSONL row is not an object: {path}:{number}")
        rows.append(row)
    return rows


def finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        fail(f"{label} is not numeric: {value!r}")
        raise AssertionError from error
    if not math.isfinite(number):
        fail(f"{label} is not finite: {value!r}")
    return number


def mad(values: list[float]) -> float:
    median = statistics.median(values)
    return statistics.median([abs(value - median) for value in values])


def execution_summary(root: Path, target: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    directory = root / f"operator-{target}"
    execution_path = directory / "execution.json"
    execution = read_json(execution_path)
    if execution.get("schema") != OPERATOR_EXEC_SCHEMA or execution.get("target") != target:
        fail(f"operator execution identity mismatch: {execution_path}")
    if execution.get("state") != "complete":
        fail(f"operator execution is not complete: {execution_path}")
    jobs = execution.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 24:
        fail(f"operator execution must contain 24 jobs: {execution_path}")
    hashes = {"execution": digest(execution_path)}
    rows_by_case: list[dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict) or int(job.get("exit_code", -1)) != 0:
            fail(f"operator job did not exit successfully: {execution_path}")
        case = job.get("case")
        raw = Path(str(job.get("raw_jsonl", "")))
        if not isinstance(case, dict) or not raw.is_file():
            fail(f"operator job evidence is incomplete: {job}")
        observed_hash = digest(raw)
        if observed_hash != job.get("raw_sha256"):
            fail(f"operator raw hash mismatch: {raw}")
        hashes[str(job["name"])] = observed_hash
        rows_by_case.append({"case": case, "job": job, "rows": read_jsonl(raw)})
    binary = Path(str(execution.get("binary", "")))
    binary_hash = digest(binary)
    if binary_hash != execution.get("binary_sha256"):
        fail(f"operator binary hash mismatch: {binary}")
    summary = {
        "path": str(execution_path),
        "sha256": digest(execution_path),
        "schema": execution.get("schema"),
        "state": execution.get("state"),
        "target": target,
        "uuid": execution.get("uuid"),
        "binary": str(binary),
        "binary_sha256": binary_hash,
        "source_sha256_before": execution.get("source_sha256_before", execution.get("source_sha256")),
        "source_sha256_after": execution.get("source_sha256_after"),
        "run_inputs_unchanged": execution.get("run_inputs_unchanged"),
        "performance_level_restored": execution.get("performance_level_restored"),
        "service_restored": execution.get("service_restored") if target == "gfx1201" else None,
        "raw_sha256": hashes,
    }
    return summary, execution, rows_by_case


def oracle_summary(rows_by_case: list[dict[str, Any]], target: str) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    aggregate = {
        str(variant): {"count": 0, "max_ulp": 0, "max_abs": 0.0, "max_relative": 0.0,
                       "repeat": True, "finite": True, "guard": True,
                       "control_bitwise": True, "control_bitwise_count": 0}
        for variant in range(5)
    }
    seen_cases: set[tuple[int, int, int, int]] = set()
    for item in rows_by_case:
        case = item["case"]
        identity = (int(case["length"]), int(case["m"]), int(case["pattern"]), int(case["bench"]))
        if identity in seen_cases:
            fail(f"duplicate operator case for {target}: {identity}")
        seen_cases.add(identity)
        oracle_rows = [row for row in item["rows"] if row.get("kind") == "oracle"]
        if len(oracle_rows) != 5 or sorted(int(row.get("variant", -1)) for row in oracle_rows) != list(range(5)):
            fail(f"oracle variant matrix incomplete for {target}: {identity}")
        case_variants: dict[str, Any] = {}
        for row in oracle_rows:
            variant = str(int(row["variant"]))
            if (row.get("target") != target or int(row.get("length", -1)) != identity[0] or
                    int(row.get("m", -1)) != identity[1] or int(row.get("pattern", -1)) != identity[2] or
                    row.get("state") != "PASS" or row.get("repeat") is not True or
                    row.get("finite") is not True or row.get("guard") is not True):
                fail(f"oracle explicit PASS/identity check failed for {target}: {identity}")
            values = {
                "state": row.get("state"),
                "max_ulp": int(row["max_ulp"]),
                "max_abs": finite(row["max_abs"], "max_abs"),
                "max_relative": finite(row["max_relative"], "max_relative"),
                "repeat": row.get("repeat") is True,
                "finite": row.get("finite") is True,
                "guard": row.get("guard") is True,
                "control_bitwise": row.get("control_bitwise") is True,
            }
            if values["max_ulp"] > 4 or values["max_abs"] > 0.03125 or values["max_relative"] > 0.04:
                fail(f"oracle threshold check failed for {target}: {identity} variant {variant}")
            case_variants[variant] = values
            total = aggregate[variant]
            total["count"] += 1
            total["max_ulp"] = max(total["max_ulp"], values["max_ulp"])
            total["max_abs"] = max(total["max_abs"], values["max_abs"])
            total["max_relative"] = max(total["max_relative"], values["max_relative"])
            for key in ("repeat", "finite", "guard", "control_bitwise"):
                total[key] = total[key] and values[key]
            total["control_bitwise_count"] += int(values["control_bitwise"])
        cases.append({"length": identity[0], "m": identity[1], "pattern": identity[2],
                      "bench": identity[3], "variants": case_variants})
    if len(cases) != 24:
        fail(f"expected 24 operator cases for {target}, got {len(cases)}")
    return {"case_count": len(cases), "variants": aggregate, "cases": cases}


def performance_summary(rows_by_case: list[dict[str, Any]], target: str) -> dict[str, Any]:
    compact: list[dict[str, Any]] = []
    for item in rows_by_case:
        case = item["case"]
        if int(case["bench"]) != 1:
            continue
        for row in item["rows"]:
            if row.get("kind") != "performance":
                continue
            variant = int(row["variant"])
            fields: dict[str, Any] = {
                "length": int(case["length"]), "m": int(case["m"]),
                "pattern": int(case["pattern"]), "bench": int(case["bench"]),
                "variant": variant,
            }
            for prefix in ("stage1", "total"):
                control = [finite(value, prefix) for value in row[f"control_{prefix}_samples_ms"]]
                candidate = [finite(value, prefix) for value in row[f"candidate_{prefix}_samples_ms"]]
                control_median = finite(row[f"control_{prefix}_ms"], prefix)
                candidate_median = finite(row[f"candidate_{prefix}_ms"], prefix)
                fields[f"control_{prefix}_median_ms"] = control_median
                fields[f"candidate_{prefix}_median_ms"] = candidate_median
                fields[f"control_{prefix}_mad_ms"] = mad(control)
                fields[f"candidate_{prefix}_mad_ms"] = mad(candidate)
                fields[f"{prefix}_reduction_ms"] = control_median - candidate_median
                if prefix == "stage1":
                    fields["stage1_reduction_16_layers_ms"] = (control_median - candidate_median) * 16.0
            compact.append(fields)
    if len(compact) != 64:
        fail(f"expected 16 performance cases x 4 variants for {target}, got {len(compact)}")
    variants: dict[str, Any] = {}
    for variant in range(1, 5):
        selected = [row for row in compact if row["variant"] == variant]
        reductions = [row["stage1_reduction_16_layers_ms"] for row in selected]
        variants[str(variant)] = {
            "case_count": len(selected),
            "stage1_reduction_16_layers_median_ms": statistics.median(reductions),
            "stage1_reduction_16_layers_mad_ms": mad(reductions),
        }
    return {"case_count": 16, "row_count": len(compact), "rows": compact, "variants": variants}


def cutoff_summary(performance: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cutoff = {"gfx1030": 6.2072, "gfx1201": 1.3935}
    result: dict[str, Any] = {"context": 8256, "m": 1, "threshold_ms_per_token": cutoff, "targets": {}}
    for target in TARGETS:
        rows = [row for row in performance[target]["rows"]
                if row["length"] == 8256 and row["m"] == 1 and row["pattern"] == 0]
        if len(rows) != 4:
            fail(f"missing 8256/M1 performance rows for cutoff: {target}")
        candidates = {}
        for row in rows:
            observed = row["stage1_reduction_16_layers_ms"]
            candidates[str(row["variant"])] = {
                "stage1_reduction_16_layers_ms": observed,
                "cutoff_ms_per_token": cutoff[target],
                "meets_cutoff": observed >= cutoff[target],
            }
        result["targets"][target] = {
            "candidates": candidates,
            "any_candidate_meets_cutoff": any(row["meets_cutoff"] for row in candidates.values()),
        }
    return result


def build_manifest(root: Path, executions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    path = root / "probe-final-build-identity.json"
    if not path.is_file():
        fail(f"missing final build manifest: {path}")
    manifest = read_json(path)
    binaries = manifest.get("binaries", {})
    checks = {}
    for target in TARGETS:
        entry = binaries.get(target, {})
        checks[target] = {
            "manifest_sha256": entry.get("sha256"),
            "execution_sha256": executions[target].get("binary_sha256"),
            "match": entry.get("sha256") == executions[target].get("binary_sha256"),
        }
        if checks[target]["match"] is not True:
            fail(f"build manifest binary hash mismatch: {target}")
    return {"path": str(path), "sha256": digest(path), "manifest": manifest, "binary_checks": checks}


def _mtp_mode(name: str) -> str:
    if "mtp-on" in name:
        return "on"
    if "mtp-off" in name:
        return "off"
    return name


def fullmodel_summary(root: Path, target: str, flavor: str) -> dict[str, Any]:
    directory = root / f"fullmodel-{flavor}-{target}"
    if not directory.is_dir():
        return {"status": "absent", "flavor": flavor, "target": target}
    execution_path = directory / "execution.json"
    if not execution_path.is_file():
        return {"status": "missing_execution", "flavor": flavor, "target": target}
    execution = read_json(execution_path)
    if execution.get("state") != "complete":
        # A running full-model benchmark is intentionally represented by status
        # only; the operator evidence is not a full-model completion claim.
        return {"status": execution.get("state"), "flavor": flavor, "target": target}
    jobs = execution.get("jobs", [])
    base: dict[str, Any] = {
        "status": "complete", "flavor": flavor, "target": target,
        "execution": str(execution_path), "execution_sha256": digest(execution_path),
        "uuid": execution.get("uuid"), "jobs": [], "validation_pass": True,
        "binary": execution.get("binary"), "binary_sha256": execution.get("binary_sha256"),
        "performance_level_restored": execution.get("performance_level_restored"),
        "service_restored": execution.get("service_restored"),
    }
    if execution.get("schema") != FULLMODEL_EXEC_SCHEMA or not isinstance(jobs, list):
        base["validation_pass"] = False
        return base
    if digest(Path(str(execution.get("binary", "")))) != execution.get("binary_sha256"):
        fail(f"fullmodel binary hash mismatch: {directory}")
    for job in jobs:
        if not isinstance(job, dict):
            base["validation_pass"] = False
            continue
        name = str(job.get("name"))
        report_path = directory / name / "report.json"
        item: dict[str, Any] = {
            "name": name, "mtp_mode": _mtp_mode(name), "exit_code": job.get("exit_code"),
            "report": str(report_path), "validation": {},
        }
        if job.get("exit_code") != 0 or not report_path.is_file() or report_path.stat().st_size == 0:
            item["status"] = "invalid"
            item["validation"]["exit_and_report"] = False
            base["validation_pass"] = False
            base["jobs"].append(item)
            continue
        report = read_json(report_path)
        report_hash = digest(report_path)
        item["status"] = report.get("state")
        item["report_sha256"] = report_hash
        item["report_sha256_matches"] = report_hash == job.get("report_sha256")
        rows = report.get("rows") if isinstance(report.get("rows"), list) else []
        row = rows[0] if rows else {}
        repetitions = report.get("repetitions") if isinstance(report.get("repetitions"), dict) else {}
        item["measured_summary"] = row.get("measured_summary")
        item["token_hashes"] = {key: row.get(key) for key in (
            "prompt_prefix_sha256", "generated_tokens_sha256", "visible_tokens_sha256",
            "generated_text_sha256")}
        item["determinism"] = {
            "deterministic_generated_tokens": row.get("deterministic_generated_tokens"),
            "visible_token_count": row.get("visible_token_count"),
            "decoded_token_count": row.get("decoded_token_count"),
        }
        measured_runs = sorted(
            [run for run in row.get("runs", []) if isinstance(run, dict) and run.get("sample_kind") == "measured"],
            key=lambda run: int(run.get("sample_index", 0)))
        first_measured = measured_runs[0] if measured_runs else {}
        item["first_measured_tokens"] = {
            "sample_index": first_measured.get("sample_index"),
            "generated_tokens": first_measured.get("generated_tokens"),
            "generated_tokens_sha256": first_measured.get("generated_tokens_sha256"),
            "visible_tokens": first_measured.get("visible_tokens"),
            "visible_tokens_sha256": first_measured.get("visible_tokens_sha256"),
        }
        item["first_measured_mtp"] = first_measured.get("mtp")
        cleanup = report.get("cleanup")
        item["cleanup"] = {key: cleanup.get(key) for key in (
            "zero", "retryable_cleanup", "durable_quarantine")} if isinstance(cleanup, dict) else None
        audit_summary = []
        id93 = []
        audit_valid = True
        for run in row.get("runs", []) if isinstance(row.get("runs"), list) else []:
            audit = run.get("audit") if isinstance(run, dict) else None
            selected = audit.get("selected_kernel_counts", []) if isinstance(audit, dict) else []
            entries = [entry for entry in selected if isinstance(entry, dict) and entry.get("kernel_id") == 93]
            id93_dispatch_count = sum(int(entry.get("dispatch_count", 0)) for entry in entries)
            run_audit_valid = (isinstance(audit, dict) and audit.get("selected_backend") == "hip" and
                               audit.get("fallback_used") is False and audit.get("all_dispatches_hip") is True and
                               int(audit.get("terminal_logit_non_finite_count", -1)) == 0 and
                               id93_dispatch_count > 0)
            audit_valid = audit_valid and run_audit_valid
            if isinstance(audit, dict):
                audit_summary.append({
                    "sample_kind": run.get("sample_kind"), "sample_index": run.get("sample_index"),
                    "selected_backend": audit.get("selected_backend"), "target": audit.get("target"),
                    "fallback_used": audit.get("fallback_used"), "all_dispatches_hip": audit.get("all_dispatches_hip"),
                    "terminal_logit_non_finite_count": audit.get("terminal_logit_non_finite_count"),
                    "kernel_dispatch_count": audit.get("kernel_dispatch_count"),
                    "graph_replay_count": audit.get("graph_replay_count"),
                    "graph_span_count": audit.get("graph_span_count"),
                    "kv_append_attention_chain_count": audit.get("kv_append_attention_chain_count"),
                    "id93_dispatch_count": id93_dispatch_count,
                    "validation_pass": run_audit_valid,
                })
            if entries:
                id93.append({"sample_kind": run.get("sample_kind"), "sample_index": run.get("sample_index"),
                             "entries": [{"kernel_id": entry.get("kernel_id"),
                                          "kernel_symbol": entry.get("kernel_symbol"),
                                          "dispatch_count": entry.get("dispatch_count")} for entry in entries]})
        item["id93_audit"] = id93
        item["audit_summary"] = audit_summary
        item["validation"].update({
            "state_pass": report.get("state") == "PASS",
            "exit_and_report": True,
            "report_sha256_matches": item["report_sha256_matches"],
            "repetitions_1_warmup_3_measured": repetitions.get("warmups_per_row") == 1 and repetitions.get("measured_per_row") == 3,
            "deterministic": row.get("deterministic_generated_tokens") is True,
            "cleanup_zero": isinstance(cleanup, dict) and cleanup.get("zero") is True,
            "audit_hip_no_fallback_finite_id93": audit_valid,
        })
        item["validation_pass"] = all(item["validation"].values()) and len(measured_runs) == 3
        base["validation_pass"] = base["validation_pass"] and item["validation_pass"]
        base["jobs"].append(item)
    return base


def _first_divergence(lhs: list[Any] | None, rhs: list[Any] | None) -> int | None:
    if lhs is None or rhs is None:
        return None
    for index, (left, right) in enumerate(zip(lhs, rhs)):
        if left != right:
            return index
    return len(lhs) if len(lhs) != len(rhs) else None


def compare_fullmodel(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "pending", "modes": {}}
    if baseline.get("status") != "complete" or candidate.get("status") != "complete":
        return result
    for mode in ("off", "on"):
        base_job = next((job for job in baseline.get("jobs", []) if job.get("mtp_mode") == mode), None)
        cand_job = next((job for job in candidate.get("jobs", []) if job.get("mtp_mode") == mode), None)
        if base_job is None or cand_job is None:
            result["modes"][mode] = {"status": "missing"}
            continue
        base_tokens = (base_job.get("first_measured_tokens") or {}).get("generated_tokens")
        cand_tokens = (cand_job.get("first_measured_tokens") or {}).get("generated_tokens")
        base_first_hash = (base_job.get("first_measured_tokens") or {}).get("generated_tokens_sha256")
        cand_first_hash = (cand_job.get("first_measured_tokens") or {}).get("generated_tokens_sha256")
        token_hashes = {"baseline": base_job.get("token_hashes"), "candidate": cand_job.get("token_hashes")}
        token_hashes["equal"] = token_hashes["baseline"] == token_hashes["candidate"]
        speed_delta: dict[str, Any] = {}
        base_summary = base_job.get("measured_summary") or {}
        cand_summary = cand_job.get("measured_summary") or {}
        for metric in sorted(set(base_summary) & set(cand_summary)):
            if not isinstance(base_summary[metric], dict) or not isinstance(cand_summary[metric], dict):
                continue
            baseline_median = finite(base_summary[metric].get("median"), metric)
            candidate_median = finite(cand_summary[metric].get("median"), metric)
            speed_delta[metric] = {"baseline_median": baseline_median, "candidate_median": candidate_median,
                                   "candidate_minus_baseline": candidate_median - baseline_median}
        result["modes"][mode] = {
            "status": "complete",
            "token_hashes": token_hashes,
            "first_measured_token_hashes": {"baseline": base_first_hash, "candidate": cand_first_hash,
                                            "equal": base_first_hash == cand_first_hash},
            "mtp": {"baseline": base_job.get("first_measured_mtp"),
                    "candidate": cand_job.get("first_measured_mtp")},
            "first_measured_tokens": {"baseline": base_tokens, "candidate": cand_tokens,
                                      "baseline_length": len(base_tokens or []), "candidate_length": len(cand_tokens or []),
                                      "first_divergence": _first_divergence(base_tokens, cand_tokens)},
            "speed_delta": speed_delta,
        }
    result["status"] = "complete" if all(item.get("status") == "complete" for item in result["modes"].values()) else "pending"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".local-artifacts/phase87/wu1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    executions: dict[str, dict[str, Any]] = {}
    raw_cases: dict[str, list[dict[str, Any]]] = {}
    execution_evidence: dict[str, Any] = {}
    for target in TARGETS:
        evidence, execution, cases = execution_summary(root, target)
        execution_evidence[target] = evidence
        executions[target] = execution
        raw_cases[target] = cases
    oracle = {target: oracle_summary(raw_cases[target], target) for target in TARGETS}
    performance = {target: performance_summary(raw_cases[target], target) for target in TARGETS}
    manifests = build_manifest(root, execution_evidence)
    source_identity = {"build_manifest_sources": manifests["manifest"].get("sources")}
    source_identity.update({
        target: {"before": evidence.get("source_sha256_before"),
                 "after": evidence.get("source_sha256_after"),
                 "unchanged": evidence.get("run_inputs_unchanged")}
        for target, evidence in execution_evidence.items()
    })
    fullmodel = {
        target: {
            "baseline": fullmodel_summary(root, target, "baseline"),
            "candidate": fullmodel_summary(root, target, "candidate"),
        }
        for target in TARGETS
    }
    for target in TARGETS:
        fullmodel[target]["comparison"] = compare_fullmodel(
            fullmodel[target]["baseline"], fullmodel[target]["candidate"])
    # Keep token arrays in the ignored model reports; the tracked summary
    # needs only hashes, lengths and the first divergence computed above.
    for target in TARGETS:
        for flavor in ("baseline", "candidate"):
            for job in fullmodel[target][flavor].get("jobs", []):
                tokens = job.get("first_measured_tokens", {})
                generated = tokens.pop("generated_tokens", None)
                tokens.pop("visible_tokens", None)
                tokens["generated_count"] = len(generated) if isinstance(generated, list) else None
        for comparison in fullmodel[target]["comparison"].get("modes", {}).values():
            comparison.get("first_measured_tokens", {}).pop("baseline", None)
            comparison.get("first_measured_tokens", {}).pop("candidate", None)
    all_fullmodel_complete = all(
        fullmodel[target][flavor].get("status") == "complete" and
        fullmodel[target][flavor].get("validation_pass") is True
        for target in TARGETS for flavor in ("baseline", "candidate")
    )
    if all_fullmodel_complete:
        state = "PASS"
    elif any(fullmodel[target][flavor].get("status") == "complete" and
             fullmodel[target][flavor].get("validation_pass") is False
             for target in TARGETS for flavor in ("baseline", "candidate")):
        state = "operator_complete_model_validation_failed"
    else:
        state = "operator_complete_model_pending"
    report = {
        "schema": "phase87-wu1-summary-v1",
        "state": state,
        "root": str(root),
        "identity": {
            "summarizer": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__).resolve())},
            "runner": {"path": str(RUNNER.resolve()), "sha256": digest(RUNNER.resolve())},
            "build_manifest": manifests,
            "production_build": {
                "sha256": digest(root / "production-build-identity.json"),
                "manifest": read_json(root / "production-build-identity.json"),
            },
            "public_gpu": {
                "sha256": digest(root / "public-gpu-results.json"),
                "results": read_json(root / "public-gpu-results.json"),
            },
            "execution": execution_evidence,
            "binary": {target: {"path": evidence["binary"], "sha256": evidence["binary_sha256"]}
                       for target, evidence in execution_evidence.items()},
        },
        "source_identity": source_identity,
        "oracle": oracle,
        "performance": performance,
        "cutoff": cutoff_summary(performance),
        "fullmodel": fullmodel,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
