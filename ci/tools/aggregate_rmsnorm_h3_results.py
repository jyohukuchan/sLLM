#!/usr/bin/env python3
"""Fail-closed aggregation for the two dedicated RMSNorm H3 rows."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from run_rmsnorm_h3_compile import (
        DEVICE_SYMBOL,
        LOGICAL_KERNEL,
        PINNED_CONFIG,
        PINNED_IMAGE,
        ROOT,
        ROWS,
        TARGETS,
        ContractError,
        git,
        iso_now,
        read_json,
        sha256_file,
        sha256_json,
    )
    from validate_rmsnorm_h3_contracts import SCHEMAS, validate_artifacts, validate_static
except ImportError:  # pragma: no cover - package import path
    from ci.tools.run_rmsnorm_h3_compile import (  # type: ignore[no-redef]
        DEVICE_SYMBOL,
        LOGICAL_KERNEL,
        PINNED_CONFIG,
        PINNED_IMAGE,
        ROOT,
        ROWS,
        TARGETS,
        ContractError,
        git,
        iso_now,
        read_json,
        sha256_file,
        sha256_json,
    )
    from ci.tools.validate_rmsnorm_h3_contracts import SCHEMAS, validate_artifacts, validate_static  # type: ignore[no-redef]


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _new_output_root(path: Path) -> Path:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ContractError(f"aggregate output contains a symlink component: {current}")
    if absolute.parent != Path("/tmp") or not absolute.name.startswith("sllm-rmsnorm-h3-"):
        raise ContractError("aggregate output must be a new direct child of /tmp with the dedicated prefix")
    if absolute.exists() or absolute.is_symlink():
        raise ContractError("aggregate output overwrite is forbidden")
    absolute.mkdir(mode=0o700)
    return absolute


def _write_json(path: Path, value: dict[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise ContractError(f"refusing to overwrite aggregate file: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return sha256_file(path)


def _write_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.exists() or sidecar.is_symlink():
        raise ContractError(f"refusing to overwrite aggregate sidecar: {sidecar}")
    sidecar.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="ascii")
    return sha256_file(sidecar)


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    repo = _absolute(args.repo)
    if bool(args.strict_ci) == bool(args.non_strict_local):
        raise ContractError("choose exactly one of --strict-ci or --non-strict-local")
    toolchain, matrix, rows = validate_static(repo)
    commit = git(repo, "rev-parse", "HEAD")
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    clean = not bool(git(repo, "status", "--porcelain=v1", "--untracked-files=all"))
    expected_sha = args.expected_reviewed_sha or commit
    expected_tree = args.tree_oid or tree
    if args.strict_ci:
        if args.expected_reviewed_sha != commit or args.expected_tested_sha != commit or args.expected_workflow_sha != commit or args.tree_oid != tree or not clean:
            raise ContractError("strict RMSNorm aggregation requires exact clean SHA/tree identity")
    elif any(value is not None for value in (args.expected_reviewed_sha, args.expected_tested_sha, args.expected_workflow_sha, args.tree_oid)):
        raise ContractError("local-nonstrict aggregation cannot claim strict identity arguments")
    reports = validate_artifacts(repo, _absolute(args.artifact_root), expected_sha=expected_sha, expected_tree=expected_tree, strict=args.strict_ci)
    reports_by_row = {report["row_id"]: report for report in reports}
    if set(reports_by_row) != set(ROWS) or len(reports_by_row) != 2:
        raise ContractError("aggregate collected zero, duplicate, missing, or unknown RMSNorm rows")
    first = reports_by_row[ROWS[0]]
    for row_id in ROWS:
        report = reports_by_row[row_id]
        for key in ("reviewed_sha", "tested_sha", "workflow_sha", "git_tree_oid", "matrix_manifest_sha256", "workflow_file_sha256"):
            if report[key] != first[key]:
                raise ContractError(f"aggregate identity or manifest mismatch between rows: {key}")
        if report["scope"] != first["scope"] or report["source_sets"] != first["source_sets"] or report["source_symbol_map"] != first["source_symbol_map"]:
            raise ContractError("aggregate source/scope evidence differs between rows")
    artifact_root = _absolute(args.artifact_root)
    row_records: list[dict[str, Any]] = []
    for row_id in ROWS:
        target = row_id.rsplit("-", 1)[-1]
        row_dir = artifact_root / row_id
        report_path = row_dir / "rmsnorm-h3-report.json"
        metadata_path = row_dir / "rmsnorm-h3-artifact.json"
        report_sidecar = row_dir / "rmsnorm-h3-report.json.sha256"
        metadata_sidecar = row_dir / "rmsnorm-h3-artifact.json.sha256"
        row_records.append({"row_id": row_id, "target": target, "state": "PASS", "report": report_path.name, "report_sha256": sha256_file(report_path), "report_sidecar_sha256": sha256_file(report_sidecar), "metadata_sha256": sha256_file(metadata_path), "metadata_sidecar_sha256": sha256_file(metadata_sidecar)})
    started = iso_now()
    aggregate_document = {
        "schema_version": "rmsnorm-h3-aggregate-v1",
        "aggregate_id": f"rmsnorm-h3-aggregate-{args.run_id}",
        "suite_id": matrix["suite_id"],
        "tier": matrix["tier"],
        "state": "PASS",
        "required": False,
        "evidence_mode": "required-ci" if args.strict_ci else "local-nonstrict",
        "run_id": str(args.run_id),
        "run_attempt": args.run_attempt,
        "reviewed_sha": first["reviewed_sha"],
        "tested_sha": first["tested_sha"],
        "workflow_sha": first["workflow_sha"],
        "git_tree_oid": first["git_tree_oid"],
        "matrix_id": matrix["matrix_id"],
        "matrix_manifest_sha256": first["matrix_manifest_sha256"],
        "workflow_file_sha256": first["workflow_file_sha256"],
        "expected_rows": list(ROWS),
        "rows": row_records,
        "source_sets": first["source_sets"],
        "source_symbol_map": first["source_symbol_map"],
        "toolchain": first["toolchain"],
        "container": first["container"],
        "codegen": {target: rows[f"h3-rmsnorm-{target}"]["codegen"] for target in TARGETS},
        "logical_kernel": LOGICAL_KERNEL,
        "device_symbol": DEVICE_SYMBOL,
        "case_manifest": {"id": "rmsnorm-h3-compile-link-extract-inspect-v1", "selected_count": 2, "collected_count": 2},
        "scope": {"compile_only": True, "execution_attempted": False, "gpu_execution": False, "model_used": False, "network_used": False, "fallback_allowed": False, "fallback_used": False, "cpu_fallback_used": False, "fake_hip": False, "emulation": False},
        "counts": {"expected_rows": 2, "selected_rows": 2, "collected_rows": 2, "passed_rows": 2, "failed_rows": 0},
        "errors": [],
        "timestamps": {"started_at": started, "finished_at": iso_now()},
    }
    from jsonschema import Draft202012Validator, FormatChecker
    schema = read_json(repo / SCHEMAS["aggregate"])
    errors = list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(aggregate_document))
    if errors:
        raise ContractError(f"aggregate document fails dedicated schema: {errors[0].message}")
    output_root = _new_output_root(args.output_dir)
    output_path = output_root / "rmsnorm-h3-aggregate.json"
    _write_json(output_path, aggregate_document)
    _write_sidecar(output_path)
    if {entry.name for entry in output_root.iterdir()} != {"rmsnorm-h3-aggregate.json", "rmsnorm-h3-aggregate.json.sha256"}:
        raise ContractError("aggregate output contains stale or unknown entries")
    return aggregate_document


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", type=Path, default=ROOT)
    result.add_argument("--artifact-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--expected-reviewed-sha")
    result.add_argument("--expected-tested-sha")
    result.add_argument("--expected-workflow-sha")
    result.add_argument("--tree-oid")
    result.add_argument("--run-id", default="local")
    result.add_argument("--run-attempt", type=int, default=1)
    result.add_argument("--strict-ci", action="store_true")
    result.add_argument("--non-strict-local", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        if args.run_attempt < 1 or not str(args.run_id) or any(char in str(args.run_id) for char in "/\\"):
            raise ContractError("aggregate run identity is invalid")
        aggregate(args)
        print("RMSNorm H3 aggregate: PASS (two exact compile-only rows)")
        return 0
    except (ContractError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"RMSNorm H3 aggregate: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
