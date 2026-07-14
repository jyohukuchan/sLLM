#!/usr/bin/env python3
"""Rank read-only AQ4 bottleneck families from trace and executor evidence."""

from __future__ import annotations

import argparse
import json
import sys
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, dict[str, float]] = defaultdict(lambda: {"wall_time_ms": 0.0, "launches": 0.0, "h2d_bytes": 0.0, "d2h_bytes": 0.0, "syncs": 0.0, "workspace_bytes": 0.0, "fallbacks": 0.0, "actual_m": 0.0, "observations": 0.0})
    for record in records:
        for phase in record.get("phases", []):
            kind = phase.get("kind", "unknown")
            row = totals[kind]
            wall = float(phase.get("wall_time_ms", 0.0)); actual_m = float(phase.get("actual_token_batch_width", 0))
            if not math.isfinite(wall) or wall < 0 or not math.isfinite(actual_m) or actual_m < 0:
                raise ValueError("phase timing/width is non-finite or negative")
            row["wall_time_ms"] += wall
            row["actual_m"] = max(row["actual_m"], actual_m)
            row["observations"] += 1
        for operator in record.get("operator_resolutions", []):
            kind = operator.get("op_kind", "unknown")
            row = totals[kind]
            launches = float(operator.get("invocation_count", 0)); workspace = float(operator.get("workspace", {}).get("planned_bytes", 0))
            if not math.isfinite(launches) or launches < 0 or not math.isfinite(workspace) or workspace < 0:
                raise ValueError("operator count/workspace is non-finite or negative")
            row["launches"] += launches
            row["workspace_bytes"] += workspace
            if operator.get("resolution_status") != "selected":
                row["fallbacks"] += 1
        for event in record.get("fallback", {}).get("events", []):
            if event.get("classification") not in {"expected", "unexpected", "unsupported", "fail_closed"}:
                raise ValueError("unknown fallback classification")
            totals[event.get("op_kind", "unknown")]["fallbacks"] += 1
        memory = record.get("memory", {})
        observed = float(memory.get("observed_peak_bytes") or 0)
        if not math.isfinite(observed) or observed < 0:
            raise ValueError("memory observed peak is invalid")
        totals["memory"]["workspace_bytes"] = max(totals["memory"]["workspace_bytes"], observed)
    ranked = sorted(({"family": name, **values} for name, values in totals.items()), key=lambda row: (-row["wall_time_ms"], -row["launches"], row["family"]))
    return {"schema_version": "ullm.aq4_bottleneck_audit.v1", "status": "ok", "records": len(records), "ranked": ranked, "selection_rule": "wall_time_ms_then_launches_then_fallbacks", "performance_eligible": False, "p1_d_handoff": "P2 must rerun this read-only audit from an independently valid production-server trace; P1 audit is diagnostic only", "promotion_note": "diagnostic only; no candidate may be promoted without a validated production trace"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        records = []
        for path in args.inputs:
            value = json.loads(path.read_text(encoding="utf-8"))
            records.append(value.get("executor_record", value))
        result = audit(records)
        raw = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(raw, encoding="utf-8")
        else:
            print(raw, end="")
        return 0
    except (OSError, ValueError, TypeError) as error:
        print(f"AQ4 bottleneck audit failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
