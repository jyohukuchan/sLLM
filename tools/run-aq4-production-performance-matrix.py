#!/usr/bin/env python3
"""Run an append-only AQ4 performance matrix with bounded raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class RunnerError(ValueError):
    pass


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise RunnerError("cannot calculate a percentile from no samples")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lower, upper = int(rank), min(int(rank) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_case(command_template: str, case: dict[str, Any], run_dir: Path, sample_index: int, phase: str, timeout: float) -> dict[str, Any]:
    case_id = str(case["case_id"])
    rendered = command_template.format(case_id=case_id, phase=phase, sample_index=sample_index, run_dir=str(run_dir), case_json=json.dumps(case, ensure_ascii=True, sort_keys=True))
    started = time.perf_counter()
    try:
        completed = subprocess.run(rendered, shell=True, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)
        status = "ok" if completed.returncode == 0 else "failed"
        error = None if status == "ok" else {"type": "command_failed", "message": completed.stderr[-4096:]}
    except subprocess.TimeoutExpired as exc:
        completed = None
        status = "failed"
        error = {"type": "timeout", "message": str(exc)}
    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    raw_path = run_dir / "raw" / case_id / f"{phase}-{sample_index:03d}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "schema_version": "ullm.aq4_production_matrix_raw.v1",
        "case_id": case_id,
        "phase": phase,
        "sample_index": sample_index,
        "command": rendered,
        "status": status,
        "elapsed_ms": elapsed_ms,
        "returncode": completed.returncode if completed else None,
        "stdout": completed.stdout[-65_536:] if completed else "",
        "stderr": completed.stderr[-65_536:] if completed else "",
        "error": error,
    }
    raw_path.write_text(json.dumps(raw, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(raw_path.relative_to(run_dir)), "sha256": digest(raw_path), "status": status, "elapsed_ms": elapsed_ms, "raw": raw}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode not in {"mechanics_smoke", "production"}:
        raise RunnerError("mode must be mechanics_smoke or production")
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    cases = matrix.get("cases") if isinstance(matrix, dict) else None
    if not isinstance(cases, list) or not cases:
        raise RunnerError("matrix must contain a nonempty cases array")
    case_ids = [case.get("case_id") for case in cases]
    if any(not isinstance(value, str) or not value for value in case_ids) or len(set(case_ids)) != len(case_ids):
        raise RunnerError("case IDs must be unique nonempty strings")
    if args.output.exists():
        raise RunnerError(f"refusing to overwrite existing run: {args.output}")
    incomplete = args.output.with_name(f".{args.output.name}.incomplete")
    incomplete.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    try:
        for case in cases:
            for sample_index in range(args.warmups):
                run_case(args.command, case, incomplete, sample_index, "warmup", args.timeout)
            measured = [run_case(args.command, case, incomplete, index, "measured", args.timeout) for index in range(args.measured)]
            valid = [item for item in measured if item["status"] == "ok"]
            row = {
                "schema_version": "inference-benchmark-result-v0.1",
                "run_id": args.run_id,
                "case_id": case["case_id"],
                "status": "ok" if len(valid) == args.measured else "failed",
                "case": case,
                "measurement": {
                    "warmup_runs": args.warmups,
                    "measured_runs": args.measured,
                    "successful_runs": len(valid),
                    "percentile_method": "linear_interpolation_rank_(n-1)*p",
                    "elapsed_ms_p50": percentile([item["elapsed_ms"] for item in valid], 0.50) if valid else None,
                    "elapsed_ms_p95": percentile([item["elapsed_ms"] for item in valid], 0.95) if valid else None,
                    "raw": [{"path": item["path"], "sha256": item["sha256"]} for item in measured],
                },
                "error": None if len(valid) == args.measured else {"type": "incomplete_measurements", "message": "one or more measured samples failed"},
            }
            rows.append(row)
        (incomplete / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        (incomplete / "matrix.json").write_text(json.dumps(matrix, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = {
            "schema_version": "ullm.aq4_production_matrix.v1",
            "run_id": args.run_id,
            "status": "complete",
            "case_count": len(rows),
            "warmup_runs": args.warmups,
            "measured_runs": args.measured,
            "command_template": args.command,
            "execution_mode": args.mode,
            "performance_eligible": args.mode == "production",
            "p1_mechanics_smoke": args.mode == "mechanics_smoke",
            "p2_handoff": None if args.mode == "production" else "repeat with --mode production against the real active worker/server boundary; P1 smoke is not performance evidence",
        }
        (incomplete / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (incomplete / "SHA256SUMS").write_text("".join(f"{digest(path)}  {path.relative_to(incomplete)}\n" for path in sorted(incomplete.rglob("*") ) if path.is_file() and path.name != "SHA256SUMS"), encoding="utf-8")
        incomplete.rename(args.output)
        return summary
    except Exception:
        # Leaving the .incomplete directory is intentional: a failed schedule
        # must remain visible and cannot be mistaken for a complete run.
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--command", required=True, help="shell template with {case_id}, {phase}, {sample_index}, {run_dir}, {case_json}")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--measured", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--mode", choices=("mechanics_smoke", "production"), default="mechanics_smoke", help="P1 mechanics smoke (non-performance) or P2 real production command")
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.measured <= 0:
        parser.error("warmups must be nonnegative and measured must be positive")
    try:
        print(json.dumps(run(args), ensure_ascii=True, sort_keys=True))
        return 0
    except (RunnerError, OSError, ValueError) as error:
        print(f"AQ4 matrix failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
