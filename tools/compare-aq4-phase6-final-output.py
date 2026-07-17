#!/usr/bin/env python3
"""Bind the Phase 6 final-output comparison to the 07/14 metric definition.

The historical `0.6151289249` is the maximum, over the fixed three M=1 rows,
of relative L2 on the intersection of stored source/path logit coordinates.
It is not a full-vocabulary metric.  This tool validates each oracle root with
the existing P2 validator, recomputes that exact bounded metric from payloads,
and writes a create-new immutable side-by-side report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
EXPECTED_BASELINE_LOGIT_RELATIVE_L2 = 0.6151289249025698
BASELINE_TOLERANCE = 1e-12
OUTPUT_DIRECTORY_MODE = 0o555
OUTPUT_FILE_MODE = 0o444


class ComparisonError(RuntimeError):
    """The fixed-fixture metric could not be validated or recorded."""


def load_oracle_module():
    spec = importlib.util.spec_from_file_location("qwen35_aq4_p2_oracle_phase6", TOOLS / "qwen35_aq4_p2_oracle.py")
    if spec is None or spec.loader is None:
        raise ComparisonError("cannot load qwen35_aq4_p2_oracle.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ORACLE = load_oracle_module()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


def lstat_directory(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ComparisonError(f"{label} is unavailable: {path}: {error}") from error
    require(not stat.S_ISLNK(metadata.st_mode), f"{label} must not be a symlink: {path}")
    require(stat.S_ISDIR(metadata.st_mode), f"{label} must be a directory: {path}")
    return metadata


def assert_absent(path: Path, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise ComparisonError(f"could not inspect {label}: {path}: {error}") from error
    raise ComparisonError(f"refusing to overwrite existing {label}: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def metric(source_root: Path, path_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_context = ORACLE.ValidationContext()
    path_context = ORACLE.ValidationContext()
    source = ORACLE.validate_manifest(source_root, expected_kind="independent_source", context=source_context)
    path = ORACLE.validate_manifest(path_root, expected_kind="same_artifact_all_m1", context=path_context)
    result = ORACLE.compare_payloads(
        source_root,
        source,
        path_root,
        path,
        source_context=source_context,
        path_context=path_context,
    )
    source_context.verify_all()
    path_context.verify_all()
    require(result["record_count"] == 3, f"fixed fixture must have exactly three rows, got {result['record_count']}")
    require(result["bounded_metric_scope"] == "intersection_of_stored_indices", "bounded metric scope differs")
    return result, source, path


def load_cases(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonError(f"could not load cases JSON: {error}") from error
    require(isinstance(value, dict) and isinstance(value.get("cases"), list), "cases JSON differs from the fixed contract")
    expected = [("fixture-prompt-0", 2), ("fixture-prompt-1", 1)]
    actual = [(entry.get("case_id"), entry.get("step_count")) for entry in value["cases"] if isinstance(entry, dict)]
    require(actual == expected, f"cases JSON differs from the fixed three-row fixture: {actual!r}")
    return value


def write_exclusive(path: Path, payload: bytes, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(path, flags, mode)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ComparisonError(f"short write: {path}")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def report(args: argparse.Namespace) -> dict[str, Any]:
    lstat_directory(args.output.parent, "comparison output parent")
    assert_absent(args.output, "comparison output directory")
    cases = load_cases(args.cases)
    baseline, source, baseline_path = metric(args.source_oracle, args.baseline_path_oracle)
    baseline_value = float(baseline["logit_sample_bounded_relative_l2_max"])
    require(math.isclose(baseline_value, EXPECTED_BASELINE_LOGIT_RELATIVE_L2, rel_tol=0.0, abs_tol=BASELINE_TOLERANCE), "legacy bounded logit relative L2 no longer reproduces 0.6151289249025698")
    result: dict[str, Any] = {
        "schema_version": "ullm.aq4_phase6_final_output_comparison.v1",
        "status": "valid",
        "metric_definition": {
            "name": "logit_sample_bounded_relative_l2_max",
            "formula": "sqrt(sum((aq4-source)^2)) / max(sqrt(sum(source^2)), 1e-12)",
            "aggregation": "maximum across the three fixed M=1 rows",
            "scope": "intersection_of_stored_indices",
            "not_full_vocabulary": True,
        },
        "fixture": {
            "cases_path": str(args.cases.resolve(strict=True)),
            "cases_sha256": sha256(args.cases),
            "cases": cases["cases"],
            "source_oracle": str(args.source_oracle.resolve(strict=True)),
            "source_manifest_sha256": sha256(args.source_oracle / "manifest.json"),
        },
        "before_fix": {
            "path_oracle": str(args.baseline_path_oracle.resolve(strict=True)),
            "path_manifest_sha256": sha256(args.baseline_path_oracle / "manifest.json"),
            "agreement": baseline,
            "expected_logit_sample_bounded_relative_l2_max": EXPECTED_BASELINE_LOGIT_RELATIVE_L2,
        },
        "gate_note": "diagnostic comparison only; it is not the independent full-vocabulary P2 fidelity gate",
    }
    if args.after_path_oracle is not None:
        after, after_source, after_path = metric(args.source_oracle, args.after_path_oracle)
        require(after_source["payload"]["sha256"] == source["payload"]["sha256"], "post-fix source oracle differs from fixed source oracle")
        after_value = float(after["logit_sample_bounded_relative_l2_max"])
        result["after_fix"] = {
            "path_oracle": str(args.after_path_oracle.resolve(strict=True)),
            "path_manifest_sha256": sha256(args.after_path_oracle / "manifest.json"),
            "agreement": after,
        }
        result["delta"] = {
            "logit_sample_bounded_relative_l2_absolute": after_value - baseline_value,
            "logit_sample_bounded_relative_l2_percent_reduction": (baseline_value - after_value) / baseline_value * 100.0,
            "strictly_improved": after_value < baseline_value,
            "hidden_sample_bounded_relative_l2_absolute": float(after["hidden_sample_bounded_relative_l2_max"]) - float(baseline["hidden_sample_bounded_relative_l2_max"]),
        }

    args.output.mkdir(mode=0o700)
    comparison = args.output / "comparison.json"
    payload = (json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_exclusive(comparison, payload, OUTPUT_FILE_MODE)
    write_exclusive(args.output / "SHA256SUMS", f"{sha256(comparison)}  comparison.json\n".encode("ascii"), OUTPUT_FILE_MODE)
    os.chmod(args.output, OUTPUT_DIRECTORY_MODE)
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-oracle", type=Path, required=True)
    parser.add_argument("--baseline-path-oracle", type=Path, required=True)
    parser.add_argument("--after-path-oracle", type=Path)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="create-new directory for comparison.json and SHA256SUMS")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    try:
        result = report(parse_args(argv))
    except (ComparisonError, ORACLE.OracleError, OSError, ValueError) as error:
        print(f"aq4-phase6-final-output-comparison: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
