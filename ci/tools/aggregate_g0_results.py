#!/usr/bin/env python3
"""Fail-closed aggregation of the two serial trusted-local G0 rows."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ContractError,
    ROOT,
    canonical_bytes,
    exact_sha,
    ensure_clean_worktree,
    identity,
    parse_time,
    read_json,
    sha256_bytes,
    sha256_file,
    sha256_json,
    validate_result_payload,
)
from validate_g0_contracts import (  # noqa: E402
    EXPECTED_ROWS,
    path_outside_repo,
    row_by_id,
    validate_g0_matrix,
    validate_g0_preflight,
    validate_schema,
)

ROW_IDS = ("g0-gfx1030", "g0-gfx1201")
PREFLIGHT_SCHEMA = "ci/schema/g0-preflight-v1.schema.json"
AGGREGATE_SCHEMA = "ci/schema/g0-aggregate-v1.schema.json"
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def validate_sidecar(path: Path, target: Path) -> str:
    if not path.is_file() or path.is_symlink() or not target.is_file() or target.is_symlink():
        raise ContractError(f"missing or unsafe report/sidecar: {target.parent.name}")
    expected = f"{sha256_file(target)}  {target.name}\n"
    try:
        actual = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"cannot read report sidecar: {exc}") from exc
    if actual != expected:
        raise ContractError(f"stale or malformed report sidecar: {target.parent.name}")
    return sha256_file(path)


def load_needs(path: Path) -> None:
    document = read_json(path)
    if not isinstance(document, dict) or list(document) != list(ROW_IDS):
        raise ContractError("G0 needs input must contain exactly the ordered two rows")
    for row_id in ROW_IDS:
        if document[row_id] != {"result": "success"}:
            raise ContractError(f"needs.{row_id} is missing, cancelled, skipped, or non-success")


def validate_row(
    row_dir: Path,
    row: dict[str, Any],
    matrix: dict[str, Any],
    repo: Path,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    if not row_dir.is_dir() or row_dir.is_symlink():
        raise ContractError(f"row directory is missing or unsafe: {row['row_id']}")
    if {path.name for path in row_dir.iterdir()} != {"report.json", "report.json.sha256"}:
        raise ContractError(f"row has missing, duplicate, or unknown files: {row['row_id']}")
    report_path = row_dir / "report.json"
    sidecar_hash = validate_sidecar(row_dir / "report.json.sha256", report_path)
    report = read_json(report_path)
    if not isinstance(report, dict):
        raise ContractError(f"report is not an object: {row['row_id']}")
    validate_result_payload(report)
    exact_values = {
        "result_id": f"{row['row_id']}.{expected_identity['run_id']}.{expected_identity['run_attempt']}",
        "suite_id": row["row_id"], "tier": "tier_g0", "state": "PASS", "required": True,
        "evidence_mode": "required-ci", "matrix_row_id": row["row_id"],
        "matrix_manifest_sha256": sha256_json(matrix), "tuple_digest": sha256_json(row),
        "run_id": expected_identity["run_id"], "run_attempt": expected_identity["run_attempt"],
        "reviewed_sha": expected_identity["reviewed_sha"], "tested_sha": expected_identity["tested_sha"],
        "workflow_sha": expected_identity["workflow_sha"], "git_tree_oid": expected_identity["git_tree_oid"],
        "worktree_clean": True,
        "seed": row["seed"],
    }
    for key, expected in exact_values.items():
        if report.get(key) != expected:
            raise ContractError(f"{row['row_id']} report {key} is stale or mismatched")
    created = parse_time(report["created_at"])
    started = parse_time(report["started_at"])
    finished = parse_time(report["finished_at"])
    if not created <= started <= finished or finished > datetime.now(timezone.utc):
        raise ContractError(f"{row['row_id']} report timestamps are stale, unordered, or future")
    if report.get("counts") != {"collected": 1, "selected": 1, "passed": 1, "failed": 0, "skipped": 0, "deselected": 0}:
        raise ContractError(f"{row['row_id']} report has zero, skip, or non-PASS counts")
    if report.get("g0", {}).get("kernel_dispatch_count") != 0:
        raise ContractError(f"{row['row_id']} G0 report claims a kernel dispatch")
    preflight = report.get("g0", {}).get("preflight")
    if not isinstance(preflight, dict) or report["g0"].get("preflight_sha256") != sha256_json(preflight):
        raise ContractError(f"{row['row_id']} preflight content hash is missing or stale")
    validate_g0_preflight(
        preflight,
        row["row_id"],
        repo,
        expected_sha=expected_identity["reviewed_sha"],
        expected_tree=expected_identity["git_tree_oid"],
        observation_window=(started, finished),
    )
    if report["g0"].get("preflight_schema_sha256") != sha256_file(repo / PREFLIGHT_SCHEMA):
        raise ContractError(f"{row['row_id']} G0 preflight schema hash is stale")
    expected_artifact = {
        "content_sha256": preflight["artifact_binding"]["artifact_sha256"],
        "manifest_sha256": preflight["artifact_binding"]["metadata_sha256"],
    }
    if report.get("artifact") != expected_artifact:
        raise ContractError(f"{row['row_id']} report artifact binding is stale or substituted")
    gpu = report.get("gpu")
    if gpu != {
        "uuid": row["uuid"], "bdf": row["bdf"], "exact_target": row["target"],
        "selected_backend": "hip-preflight", "dispatch_count": 0, "kernel_dispatch_count": 0,
        "dispatch_ids": [], "fallback_allowed": False, "fallback_used": False,
        "code_object": {"target": row["target"], "artifact_sha256": preflight["artifact_binding"]["artifact_sha256"]},
    }:
        raise ContractError(f"{row['row_id']} GPU summary is not exact non-execution preflight evidence")
    return {
        "row_id": row["row_id"], "target": row["target"], "bdf": row["bdf"], "uuid": row["uuid"],
        "state": "PASS", "report_sha256": sha256_file(report_path),
        "report_sidecar_sha256": sidecar_hash, "kernel_dispatch_count": 0,
    }


def aggregate(
    *, needs: Path, artifact_dir: Path, repo: Path, run_id: str, run_attempt: int,
    reviewed_sha: str, tested_sha: str, workflow_sha: str, tree_oid: str,
) -> dict[str, Any]:
    if not RUN_ID.fullmatch(run_id) or isinstance(run_attempt, bool) or not isinstance(run_attempt, int) or run_attempt < 1:
        raise ContractError("G0 run identity is invalid")
    shas = [exact_sha(value, name) for value, name in (
        (reviewed_sha, "reviewed_sha"), (tested_sha, "tested_sha"), (workflow_sha, "workflow_sha")
    )]
    if len(set(shas)) != 1:
        raise ContractError("reviewed/tested/workflow SHA values differ")
    exact_sha(tree_oid, "tree_oid")
    checked_out = identity(repo)
    ensure_clean_worktree(repo)
    if checked_out != {"commit": reviewed_sha, "tree": tree_oid}:
        raise ContractError("aggregate candidate does not match the checked-out immutable commit/tree")
    load_needs(needs)
    matrix = validate_g0_matrix(repo)
    if not artifact_dir.is_dir() or artifact_dir.is_symlink():
        raise ContractError("G0 artifact collection is missing or unsafe")
    path_outside_repo(artifact_dir, repo, "G0 artifact collection")
    resolved_artifact_dir = artifact_dir.resolve()
    if resolved_artifact_dir.parent != Path("/tmp") or not resolved_artifact_dir.name.startswith("sllm-g0-"):
        raise ContractError("G0 artifact collection must be a private /tmp/sllm-g0-* directory")
    if [path.name for path in sorted(artifact_dir.iterdir())] != list(ROW_IDS):
        raise ContractError("G0 artifact collection has missing, duplicate, or unknown rows")
    expected_identity = {
        "run_id": run_id, "run_attempt": run_attempt, "reviewed_sha": reviewed_sha,
        "tested_sha": tested_sha, "workflow_sha": workflow_sha, "git_tree_oid": tree_oid,
    }
    rows = [
        validate_row(artifact_dir / row_id, row_by_id(matrix, row_id), matrix, repo, expected_identity)
        for row_id in ROW_IDS
    ]
    return {
        "schema_version": "g0-aggregate-v1", "aggregate_id": f"g0-aggregate.{run_id}.{run_attempt}",
        "state": "PASS", "required": True, "run_id": run_id, "run_attempt": run_attempt,
        "reviewed_sha": reviewed_sha, "tested_sha": tested_sha, "workflow_sha": workflow_sha,
        "git_tree_oid": tree_oid, "matrix_manifest_sha256": sha256_json(matrix),
        "preflight_schema_sha256": sha256_file(repo / PREFLIGHT_SCHEMA),
        "aggregate_schema_sha256": sha256_file(repo / AGGREGATE_SCHEMA),
        "expected_rows": list(ROW_IDS), "rows": rows, "errors": [],
    }


def validate_aggregate_schema(document: dict[str, Any], repo: Path) -> None:
    schema = read_json(repo / "ci/schema/g0-aggregate-v1.schema.json")
    if not isinstance(schema, dict):
        raise ContractError("G0 aggregate schema must be an object")
    validate_schema(document, schema, "G0 aggregate")
    if document.get("matrix_manifest_sha256") != sha256_json(validate_g0_matrix(repo)):
        raise ContractError("G0 aggregate matrix hash is stale")
    if document.get("aggregate_schema_sha256") != sha256_file(repo / AGGREGATE_SCHEMA):
        raise ContractError("G0 aggregate schema hash is stale")
    if document.get("preflight_schema_sha256") != sha256_file(repo / PREFLIGHT_SCHEMA):
        raise ContractError("G0 preflight schema hash is stale")


def write_summary(output: Path, document: dict[str, Any]) -> None:
    if not output.is_absolute() or output.is_symlink():
        raise ContractError("G0 aggregate output must be an absolute non-symlink path")
    resolved = output.resolve(strict=False)
    if resolved.parent != Path("/tmp") or not resolved.name.startswith("sllm-g0-"):
        raise ContractError("G0 aggregate output must be a private /tmp/sllm-g0-* directory")
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() or not output.is_dir() or output.resolve() != resolved:
        raise ContractError("G0 aggregate output is not a regular directory")
    expected_names = {"aggregate.json", "aggregate.json.sha256"}
    if {path.name for path in output.iterdir()} - expected_names:
        raise ContractError("G0 aggregate output contains unknown or stale files")
    for name in expected_names:
        path = output / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ContractError(f"G0 aggregate output member is unsafe: {name}")
    data = canonical_bytes(document)
    path = output / "aggregate.json"
    path.write_bytes(data)
    path.with_name("aggregate.json.sha256").write_text(f"{sha256_bytes(data)}  aggregate.json\n", encoding="ascii")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--needs-json", type=Path, required=True)
    result.add_argument("--artifact-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--repo", type=Path, default=ROOT)
    result.add_argument("--run-id", required=True)
    result.add_argument("--run-attempt", type=int, required=True)
    result.add_argument("--expected-reviewed-sha", required=True)
    result.add_argument("--expected-tested-sha", required=True)
    result.add_argument("--expected-workflow-sha", required=True)
    result.add_argument("--expected-tree-oid", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        summary = aggregate(
            needs=args.needs_json, artifact_dir=args.artifact_dir, repo=args.repo.resolve(),
            run_id=args.run_id, run_attempt=args.run_attempt,
            reviewed_sha=args.expected_reviewed_sha, tested_sha=args.expected_tested_sha,
            workflow_sha=args.expected_workflow_sha, tree_oid=args.expected_tree_oid,
        )
        validate_aggregate_schema(summary, args.repo.resolve())
    except (ContractError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"G0 aggregate: FAIL: {exc}", file=sys.stderr)
        return 3
    output = args.output_dir
    if not output.is_absolute() or output.is_symlink():
        print("G0 aggregate: FAIL: output must be an absolute non-symlink path", file=sys.stderr)
        return 3
    try:
        path_outside_repo(output, args.repo.resolve(), "G0 aggregate output")
        write_summary(output.resolve(strict=False), summary)
    except (ContractError, OSError, ValueError) as exc:
        print(f"G0 aggregate: FAIL: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
