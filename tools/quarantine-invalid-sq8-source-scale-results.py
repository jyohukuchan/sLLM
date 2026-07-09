#!/usr/bin/env python3
"""Mark invalid Qwen3-14B-FP8 uLLM rows as connection diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


TARGET_DATES = ("2026-07-09", "2026-07-10")
TARGET_ENGINE = "uLLM"
TARGET_MODEL = "Qwen3-14B-FP8"
TARGET_ARTIFACT = "/tmp/ullm-qwen3-14b-fp8-full-sq8-artifact"
EXPECTED_MATCHES = 21
EXPECTED_FILES = 8
REASON_CODE = "source_fp8_weight_scale_inv_not_applied"
EVIDENCE = (
    "benchmarks/results/2026-07-10/"
    "sq8-qwen3-14b-invalid-sidecar-quarantine.md"
)

QUARANTINE_VALIDITY = {
    "state": "quarantined",
    "classification": "connection_diagnostic",
    "implementation_valid": False,
    "quality_comparison_valid": False,
    "performance_comparison_valid": False,
    "reason_codes": [REASON_CODE],
    "evidence": [EVIDENCE],
}


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def matches_invalid_source_scale_row(row: dict[str, Any]) -> bool:
    return (
        as_dict(row.get("engine")).get("name") == TARGET_ENGINE
        and as_dict(row.get("model")).get("name") == TARGET_MODEL
        and as_dict(row.get("workload")).get("sq_artifact") == TARGET_ARTIFACT
    )


def quarantine_row(row: dict[str, Any]) -> bool:
    if not matches_invalid_source_scale_row(row):
        return False
    existing = row.get("result_validity")
    if existing is not None and existing != QUARANTINE_VALIDITY:
        raise ValueError(
            f"row {row.get('case_id', '<unknown>')} has conflicting result_validity"
        )
    if existing == QUARANTINE_VALIDITY:
        return False
    row["result_validity"] = dict(QUARANTINE_VALIDITY)
    return True


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            rows.append(row)
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def result_paths(results_root: Path) -> list[Path]:
    paths: list[Path] = []
    for date in TARGET_DATES:
        date_root = results_root / date
        if date_root.is_dir():
            paths.extend(date_root.rglob("results.jsonl"))
    return sorted(paths)


def migrate(results_root: Path, *, apply: bool) -> dict[str, Any]:
    changes: list[tuple[Path, list[dict[str, Any]], int, int]] = []
    matched_total = 0
    modified_total = 0
    matched_files = 0
    for path in result_paths(results_root):
        rows = load_jsonl(path)
        matched = sum(matches_invalid_source_scale_row(row) for row in rows)
        if matched == 0:
            continue
        matched_files += 1
        modified = sum(quarantine_row(row) for row in rows)
        matched_total += matched
        modified_total += modified
        changes.append((path, rows, matched, modified))

    if matched_total != EXPECTED_MATCHES or matched_files != EXPECTED_FILES:
        raise ValueError(
            "invalid SQ8 quarantine inventory mismatch: "
            f"expected {EXPECTED_MATCHES} rows/{EXPECTED_FILES} files, "
            f"found {matched_total} rows/{matched_files} files"
        )

    if apply:
        for path, rows, _matched, modified in changes:
            if modified > 0:
                write_jsonl_atomic(path, rows)

    return {
        "applied": apply,
        "matched_rows": matched_total,
        "matched_files": matched_files,
        "modified_rows": modified_total,
        "files": [
            {
                "path": str(path),
                "matched_rows": matched,
                "modified_rows": modified,
            }
            for path, _rows, matched, modified in changes
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("benchmarks/results"),
    )
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = migrate(args.results_root, apply=args.apply)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
