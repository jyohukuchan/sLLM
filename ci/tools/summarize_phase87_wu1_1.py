#!/usr/bin/env python3
"""Summarize the Phase 87 WU1.1 operator matrix without copying raw samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


TARGETS = ("gfx1030", "gfx1201")
VARIANTS = {"gfx1030": (0, 1, 2), "gfx1201": (0, 1, 2, 3)}
EXEC_SCHEMA = "phase87-wu1-1-execution-v1"


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
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            fail(f"invalid JSONL {path}:{number}: {error}")
        if not isinstance(value, dict):
            fail(f"JSONL row is not an object: {path}:{number}")
        rows.append(value)
    return rows


def finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        fail(f"{label} is not numeric: {value!r}")
        raise AssertionError from error
    if not math.isfinite(result):
        fail(f"{label} is not finite: {value!r}")
    return result


def median_absolute_deviation(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median([abs(value - center) for value in values])


def compact_build_identity(root: Path, executions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    manifest_path = root / "probe-build-identity.json"
    manifest = read_json(manifest_path)
    checks = {}
    for target in TARGETS:
        manifest_entry = (manifest.get("binaries") or {}).get(target, {})
        execution_hash = executions[target].get("binary_sha256")
        checks[target] = {
            "manifest_sha256": manifest_entry.get("sha256"),
            "execution_sha256": execution_hash,
            "match": manifest_entry.get("sha256") == execution_hash,
        }
        if checks[target]["match"] is not True:
            fail(f"build manifest binary mismatch: {target}")
    return {"path": str(manifest_path), "sha256": digest(manifest_path),
            "flags": manifest.get("flags"), "sources": manifest.get("sources"), "binary_checks": checks}


def initial_tpots(root: Path) -> tuple[dict[str, float], dict[str, Any]]:
    path = root / "initial-state.json"
    document = read_json(path)
    values = {}
    for target in TARGETS:
        entry = (document.get("binaries") or {}).get(target, {})
        values[target] = finite(entry.get("tpot_ms"), f"{target} initial tpot_ms")
    return values, {"path": str(path), "sha256": digest(path), "tpot_ms": values}


def execution_evidence(root: Path, target: str) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
    directory = root / f"operator-{target}"
    path = directory / "execution.json"
    if not path.is_file():
        return {"status": "absent", "target": target}, None, []
    execution = read_json(path)
    state = execution.get("state")
    base = {"status": state, "target": target, "path": str(path), "sha256": digest(path),
            "completed_jobs": sum(int(job.get("exit_code") == 0) for job in execution.get("jobs", [])
                                   if isinstance(job, dict)),
            "job_count": len(execution.get("jobs", [])) if isinstance(execution.get("jobs"), list) else 0}
    if state != "complete":
        return base, execution, []
    if execution.get("schema") != EXEC_SCHEMA or base["job_count"] != 26:
        fail(f"operator execution identity/matrix mismatch: {path}")
    binary = Path(str(execution.get("binary", "")))
    binary_hash = digest(binary)
    if binary_hash != execution.get("binary_sha256"):
        fail(f"operator binary hash mismatch: {binary}")
    base.update({"uuid": execution.get("uuid"), "binary": str(binary),
                 "binary_sha256": binary_hash, "run_inputs_unchanged": execution.get("run_inputs_unchanged"),
                 "performance_level_restored": execution.get("performance_level_restored"),
                 "service_restored": execution.get("service_restored"),
                 "raw_sha256": {}})
    cases = []
    for job in execution["jobs"]:
        if not isinstance(job, dict) or job.get("exit_code") != 0:
            fail(f"operator job exit failure: {path}")
        raw_path = Path(str(job.get("raw_jsonl", "")))
        raw_hash = digest(raw_path)
        if raw_hash != job.get("raw_sha256"):
            fail(f"operator raw hash mismatch: {raw_path}")
        base["raw_sha256"][job["name"]] = raw_hash
        cases.append({"case": job["case"], "job": job, "rows": read_jsonl(raw_path)})
    return base, execution, cases


def oracle_and_cleanup(cases: list[dict[str, Any]], target: str) -> dict[str, Any]:
    aggregates = {
        str(variant): {"count": 0, "max_ulp": 0, "max_abs": 0.0, "max_relative": 0.0,
                       "repeat": True, "finite": True, "guard": True, "control_bitwise": True}
        for variant in VARIANTS[target]
    }
    compact_cases = []
    for item in cases:
        case = item["case"]
        identity = {key: int(case[key]) for key in ("length", "m", "pattern", "bench")}
        rows = item["rows"]
        oracle = [row for row in rows if row.get("kind") == "oracle"]
        cleanup = [row for row in rows if row.get("kind") == "cleanup"]
        performance = [row for row in rows if row.get("kind") == "performance"]
        variants = sorted(int(row.get("variant", -1)) for row in oracle)
        if variants != list(VARIANTS[target]) or len(cleanup) != 1:
            fail(f"oracle/cleanup matrix mismatch: {target} {identity}")
        if cleanup[0].get("state") != "PASS" or cleanup[0].get("live_allocations") != 0:
            fail(f"cleanup failed: {target} {identity}")
        case_oracle = {}
        for row in oracle:
            variant = str(int(row["variant"]))
            values = {"state": row.get("state"), "max_ulp": int(row["max_ulp"]),
                      "max_abs": finite(row["max_abs"], "max_abs"),
                      "max_relative": finite(row["max_relative"], "max_relative"),
                      "repeat": row.get("repeat") is True, "finite": row.get("finite") is True,
                      "guard": row.get("guard") is True, "control_bitwise": row.get("control_bitwise") is True}
            if values["state"] != "PASS" or not all(values[key] for key in ("repeat", "finite", "guard")):
                fail(f"oracle PASS check failed: {target} {identity} variant {variant}")
            if values["max_ulp"] > 4 or values["max_abs"] > 0.03125 or values["max_relative"] > 0.04:
                fail(f"oracle bound failed: {target} {identity} variant {variant}")
            case_oracle[variant] = values
            total = aggregates[variant]
            total["count"] += 1
            total["max_ulp"] = max(total["max_ulp"], values["max_ulp"])
            total["max_abs"] = max(total["max_abs"], values["max_abs"])
            total["max_relative"] = max(total["max_relative"], values["max_relative"])
            for key in ("repeat", "finite", "guard", "control_bitwise"):
                total[key] = total[key] and values[key]
        compact_cases.append({**identity, "oracle": case_oracle, "performance_row_count": len(performance)})
    if len(compact_cases) != 26:
        fail(f"expected 26 cases for {target}, got {len(compact_cases)}")
    return {"case_count": 26, "variants": aggregates, "cases": compact_cases}


def performance_summary(cases: list[dict[str, Any]], target: str, tpot_ms: float) -> dict[str, Any]:
    rows = []
    for item in cases:
        case = item["case"]
        if int(case["bench"]) != 1:
            continue
        for row in item["rows"]:
            if row.get("kind") != "performance":
                continue
            compact = {key: int(case[key]) for key in ("length", "m", "pattern", "bench")}
            compact["variant"] = int(row["variant"])
            for metric in ("stage1", "stage2", "total"):
                control_samples = [finite(v, metric) for v in row[f"control_{metric}_samples_ms"]]
                candidate_samples = [finite(v, metric) for v in row[f"candidate_{metric}_samples_ms"]]
                control_median = finite(row[f"control_{metric}_ms"], metric)
                candidate_median = finite(row[f"candidate_{metric}_ms"], metric)
                compact[f"control_{metric}_median_ms"] = control_median
                compact[f"candidate_{metric}_median_ms"] = candidate_median
                compact[f"control_{metric}_mad_ms"] = median_absolute_deviation(control_samples)
                compact[f"candidate_{metric}_mad_ms"] = median_absolute_deviation(candidate_samples)
                compact[f"{metric}_delta_ms"] = control_median - candidate_median
                if metric == "total":
                    deltas = []
                    for round_number in range(3):
                        begin, end = round_number * 9, (round_number + 1) * 9
                        deltas.append(statistics.median(control_samples[begin:end]) -
                                      statistics.median(candidate_samples[begin:end]))
                    compact["total_delta_ms_by_round"] = deltas
                    compact["all_rounds_positive"] = all(delta > 0.0 for delta in deltas)
            rows.append(compact)
    # Variant zero is the control and emits no performance row.
    expected = 18 * (len(VARIANTS[target]) - 1)
    if len(rows) != expected:
        fail(f"performance row count mismatch for {target}: {len(rows)} != {expected}")
    at_8256 = [row for row in rows if row["length"] == 8256 and row["m"] == 1 and row["pattern"] == 0]
    if len(at_8256) != len(VARIANTS[target]) - 1:
        fail(f"missing 8256/M1 rows for {target}")
    cutoff = 2.31 if target == "gfx1030" else None
    cutoff_rows = {}
    adoption_rows = {}
    for row in at_8256:
        reduction_16 = row["total_delta_ms"] * 16.0
        fraction = reduction_16 / tpot_ms
        key = str(row["variant"])
        cutoff_rows[key] = {"total_reduction_16_layers_ms": reduction_16,
                            "stage1_reduction_16_layers_ms": row["stage1_delta_ms"] * 16.0,
                            "cutoff_ms_per_token": cutoff,
                            "meets_cutoff": cutoff is not None and row["stage1_delta_ms"] * 16.0 >= cutoff,
                            "exploration_only": True}
        adoption_rows[key] = {"total_reduction_16_layers_ms": reduction_16,
                              "initial_tpot_ms": tpot_ms,
                              "improvement_fraction": fraction,
                              "performance_condition_ge_1_percent": fraction >= 0.01,
                              "all_rounds_positive": row["all_rounds_positive"],
                              "performance_condition_met": fraction >= 0.01 and row["all_rounds_positive"],
                              "adoption_boolean_scope": "performance condition only; N classification and target scope remain main-agent decisions"}
    return {"performance_case_count": 18, "row_count": len(rows), "rows": rows,
            "cutoff_exploration": {"threshold_ms_per_token": cutoff, "variants": cutoff_rows},
            "adoption_performance_only": {"variants": adoption_rows}}


def compact_fullmodel(root: Path) -> dict[str, Any]:
    from summarize_phase87_wu1 import fullmodel_summary, compare_fullmodel
    result = {}
    for target in TARGETS:
        result[target] = {flavor: fullmodel_summary(root, target, flavor)
                          for flavor in ("baseline", "candidate")}
        result[target]["comparison"] = compare_fullmodel(result[target]["baseline"], result[target]["candidate"])
        for flavor in ("baseline", "candidate"):
            for job in result[target][flavor].get("jobs", []):
                first = job.get("first_measured_tokens", {})
                tokens = first.pop("generated_tokens", None)
                first.pop("visible_tokens", None)
                first["generated_count"] = len(tokens) if isinstance(tokens, list) else None
        for row in result[target]["comparison"].get("modes", {}).values():
            row.get("first_measured_tokens", {}).pop("baseline", None)
            row.get("first_measured_tokens", {}).pop("candidate", None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".local-artifacts/phase87/wu1-1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    initial_tpot, initial_identity = initial_tpots(root)
    executions = {}
    raw_cases = {}
    operator = {}
    for target in TARGETS:
        evidence, execution, cases = execution_evidence(root, target)
        operator[target] = evidence
        executions[target] = execution or {}
        raw_cases[target] = cases
    manifest = compact_build_identity(root, executions) if all(executions.values()) else None
    complete = all(operator[target].get("status") == "complete" for target in TARGETS)
    fullmodel_preview = compact_fullmodel(root)
    fullmodel_complete = all(
        fullmodel_preview.get(target, {}).get(flavor, {}).get("status") == "complete" and
        fullmodel_preview.get(target, {}).get(flavor, {}).get("validation_pass") is True
        for target in TARGETS for flavor in ("baseline", "candidate")
    )
    extra_identity = {}
    for name in ("baseline-reuse.json", "production-build-identity.json", "public-gpu-results.json", "decision.json"):
        path = root / name
        if path.is_file():
            extra_identity[name] = {"sha256": digest(path), "contents": read_json(path)}
    report = {
        "schema": "phase87-wu1-1-summary-v1",
        "state": "PASS" if complete and fullmodel_complete else
                 ("operator_complete_model_pending" if complete else "operator_pending"),
        "root": str(root),
        "identity": {"initial_state": initial_identity, "build_manifest": manifest,
                     "tools": {n: digest(Path(__file__).with_name(n)) for n in ("summarize_phase87_wu1_1.py", "run_phase87_wu1_1.py", "summarize_phase87_wu1.py")},
                     "execution": operator, "integration": extra_identity},
        "operator": {},
        "fullmodel": fullmodel_preview,
        "claims": {"operator_complete": complete,
                   "model_complete": fullmodel_complete},
    }
    if complete:
        for target in TARGETS:
            report["operator"][target] = {
                "execution": operator[target],
                "oracle": oracle_and_cleanup(raw_cases[target], target),
                "performance": performance_summary(raw_cases[target], target, initial_tpot[target]),
            }
    else:
        report["operator"] = {target: {"status": operator[target].get("status"),
                                       "completed_jobs": operator[target].get("completed_jobs"),
                                       "job_count": operator[target].get("job_count")}
                               for target in TARGETS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
