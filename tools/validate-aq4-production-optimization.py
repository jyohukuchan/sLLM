#!/usr/bin/env python3
"""Validate append-only AQ4 matrix evidence and optional baseline thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import re
from pathlib import Path
from typing import Any


class ValidationError(ValueError):
    pass


HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"not a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid strict JSON: {path}") from error


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValidationError("duplicate JSON key")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValidationError(f"non-finite number: {value}")


def validate(run_dir: Path, baseline: Path | None = None, max_regression: float = 0.05) -> dict[str, Any]:
    if (run_dir / ".incomplete").exists() or list(run_dir.glob("*.incomplete")):
        raise ValidationError("incomplete evidence is present")
    summary = load(run_dir / "summary.json")
    if summary.get("schema_version") != "ullm.aq4_production_matrix.v1" or summary.get("status") != "complete":
        raise ValidationError("run summary is not complete")
    mode = summary.get("execution_mode")
    if mode not in {"mechanics_smoke", "production"}:
        raise ValidationError("run summary execution_mode is missing")
    if summary.get("performance_eligible") is not (mode == "production"):
        raise ValidationError("performance eligibility does not match execution mode")
    if mode == "mechanics_smoke" and summary.get("p1_mechanics_smoke") is not True:
        raise ValidationError("mechanics smoke is not labeled")
    if mode == "production" and summary.get("p1_mechanics_smoke") is not False:
        raise ValidationError("production run has mechanics-smoke label")
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != summary.get("case_count"):
        raise ValidationError("case count does not reconcile")
    reports = []
    for row in rows:
        if row.get("schema_version") != "inference-benchmark-result-v0.1" or row.get("run_id") != summary.get("run_id"):
            raise ValidationError("benchmark row identity differs")
        measurement = row.get("measurement", {})
        samples = measurement.get("raw", [])
        if measurement.get("measured_runs") != summary.get("measured_runs") or measurement.get("successful_runs") != len(samples):
            raise ValidationError(f"measurement counts do not reconcile for {row.get('case_id')}")
        values = []
        for item in samples:
            relative = item.get("path")
            if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts or not HASH_RE.fullmatch(str(item.get("sha256", ""))):
                raise ValidationError(f"raw evidence path/hash is invalid for {row.get('case_id')}")
            path = run_dir / relative
            if path.is_symlink() or not path.is_file() or digest(path) != item["sha256"]:
                raise ValidationError(f"raw evidence hash differs for {row.get('case_id')}")
            raw = load(path)
            if raw.get("status") != "ok":
                raise ValidationError(f"failed measured sample is present for {row.get('case_id')}")
            elapsed = raw.get("elapsed_ms")
            if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValidationError(f"invalid elapsed sample for {row.get('case_id')}")
            values.append(float(elapsed))
        if row.get("status") != "ok" or len(values) != summary.get("measured_runs"):
            raise ValidationError(f"case is not complete: {row.get('case_id')}")
        reports.append({"case_id": row["case_id"], "elapsed_ms_p50": measurement["elapsed_ms_p50"], "elapsed_ms_p95": measurement["elapsed_ms_p95"]})
    baseline_report = None
    if baseline:
        base = validate(baseline, None, max_regression)
        base_by_case = {item["case_id"]: item for item in base["cases"]}
        regressions = []
        for item in reports:
            old = base_by_case.get(item["case_id"])
            if old is None:
                continue
            limit = old["elapsed_ms_p50"] * (1.0 + max_regression)
            if item["elapsed_ms_p50"] > limit:
                regressions.append({"case_id": item["case_id"], "baseline_p50": old["elapsed_ms_p50"], "candidate_p50": item["elapsed_ms_p50"], "limit": limit})
        baseline_report = {"status": "passed" if not regressions else "failed", "max_regression": max_regression, "regressions": regressions}
        if regressions:
            raise ValidationError(f"performance regression exceeds {max_regression:.1%}")
    return {"schema_version": "ullm.aq4_production_optimization_validator.v1", "status": "valid", "run_id": summary["run_id"], "cases": reports, "baseline": baseline_report, "execution_mode": mode, "performance_eligible": mode == "production", "p1_mechanics_smoke": mode == "mechanics_smoke"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-regression", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        raw = json.dumps(validate(args.run_dir, args.baseline, args.max_regression), ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(raw, encoding="utf-8")
        else:
            print(raw, end="")
        return 0
    except (ValidationError, OSError, ValueError) as error:
        print(f"AQ4 matrix validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
