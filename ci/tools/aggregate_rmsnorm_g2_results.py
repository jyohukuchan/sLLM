#!/usr/bin/env python3
"""Fail-closed aggregation for exactly two canonical G2 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from common import ContractError, ROOT, read_json, sha256_file  # noqa: E402
from validate_rmsnorm_g2_contracts import (  # noqa: E402
    CASE_IDS, MODEL_LOCK_FINGERPRINT, MODEL_LOCK_PATH, RESOLVED_REVISION, ROWS,
    SCHEMAS, TARGETS, TOLERANCE_ID, TOLERANCE_PATH, candidate_sha256,
    validate_aggregate, validate_artifact, validate_candidate, validate_report, validate_tolerance,
)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def aggregate_reports(
    report_paths: Iterable[Path],
    artifact_paths: Iterable[Path],
    *,
    candidate: dict[str, Any],
    repo: Path = ROOT,
    strict_git: bool = False,
) -> dict[str, Any]:
    report_paths = tuple(report_paths)
    artifact_paths = tuple(artifact_paths)
    if len(report_paths) != 2 or len(artifact_paths) != 2:
        raise ContractError("G2 aggregate requires exactly two reports and two dedicated artifacts")
    validate_candidate(candidate, repo, strict_git=strict_git)
    for path in (*report_paths, *artifact_paths):
        if path.is_symlink() or not path.is_file():
            raise ContractError("G2 aggregate evidence input must be a regular non-symlink file")
    report_documents = [read_json(path) for path in report_paths]
    artifact_documents = [read_json(path) for path in artifact_paths]
    reports = [validate_report(document, repo) for document in report_documents]
    artifacts = [validate_artifact(document, repo, binary_path=path.parent / document["binary"]["path"]) for document, path in zip(artifact_documents, artifact_paths, strict=True)]
    if [report["target"] for report in reports] != list(TARGETS) or [report["row_id"] for report in reports] != list(ROWS):
        raise ContractError("G2 reports are not exactly ordered gfx1030/gfx1201 rows")
    if [artifact["target"] for artifact in artifacts] != list(TARGETS) or [artifact["row_id"] for artifact in artifacts] != list(ROWS):
        raise ContractError("G2 artifacts are not exactly ordered gfx1030/gfx1201 rows")
    if any(report["candidate"] != candidate for report in reports) or any(artifact["candidate"] != candidate for artifact in artifacts):
        raise ContractError("G2 aggregate contains a mixed or stale candidate")
    model_lock_sha = sha256_file(repo / MODEL_LOCK_PATH)
    tolerance_sha = sha256_file(repo / TOLERANCE_PATH)
    rows = []
    aggregate_prerequisites = []
    slice_identity: tuple[Any, ...] | None = None
    for order, (report, artifact) in enumerate(zip(reports, artifacts)):
        if report["model"]["lock_sha256"] != model_lock_sha or report["model"]["fingerprint"] != MODEL_LOCK_FINGERPRINT or report["model"]["resolved_revision"] != RESOLVED_REVISION:
            raise ContractError("G2 report model lock binding is stale")
        if report["tolerance"]["schema_sha256"] != tolerance_sha or report["tolerance"]["tolerance_id"] != TOLERANCE_ID:
            raise ContractError("G2 report tolerance binding is stale")
        if report["artifact"]["artifact_id"] != artifact["artifact_id"] or report["artifact"]["binary_sha256"] != artifact["binary"]["sha256"] or report["artifact"]["binary_sidecar_sha256"] != artifact["binary"]["sidecar_sha256"] or report["artifact"]["binary_source_sha256"] != artifact["binary"]["source_sha256"] or report["artifact"]["binary_source_set_sha256"] != artifact["binary"]["build_source_set"]["source_set_sha256"]:
            raise ContractError("G2 report and artifact binary identities are mixed")
        current_slice = (report["model"]["slice"]["tensor_name"], report["model"]["slice"]["source_shard"], report["model"]["slice"]["size_bytes"], report["model"]["slice"]["sha256"], report["model"]["slice"]["recipe_sha256"])
        if slice_identity is None:
            slice_identity = current_slice
        elif current_slice != slice_identity:
            raise ContractError("G2 aggregate mixes slice identities")
        aggregate_prerequisites.extend({
            "kind": item["kind"], "row_id": item["row_id"], "target": report["target"], "state": "bound-not-executed-by-g2-aggregate",
            "candidate_sha256": candidate_sha256(candidate), "artifact_sha256": item["artifact_sha256"], "report_sha256": item["report_sha256"],
        } for item in report["prerequisites"])
        report_digest = sha256_file(report_paths[order])
        artifact_digest = sha256_file(artifact_paths[order])
        if report_digest == "0" * 64 or artifact_digest == "0" * 64:
            raise ContractError("G2 aggregate evidence file hash is zero")
        tuple_digest = _sha_json(report["device"])
        rows.append({
            "order": order, "row_id": report["row_id"], "target": report["target"], "state": report["state"],
            "report_sha256": report_digest, "artifact_sha256": artifact_digest,
            "candidate_sha256": candidate_sha256(candidate), "tuple_sha256": tuple_digest,
            "collected_cases": report["collection"]["collected_cases"],
            "dispatch_count": report["scope"]["dispatch_count"], "fallback_used": report["scope"]["fallback_used"],
            "health_ok": all(report[name]["state"] == "OK" and report[name]["available"] and report[name]["reliable"] and report[name]["ras_uncorrectable_count"] == 0 for name in ("health_pre", "health_post")),
            "process_clean": all(report[name]["state"] == "CLEAN" and not report[name]["residual_runner_children"] and not report[name]["gpu_processes"] for name in ("process_pre", "process_post")),
        })
    passed = sum(row["state"] == "PASS" for row in rows)
    collected = sum(row["collected_cases"] == len(CASE_IDS) for row in rows)
    state = "PASS" if passed == 2 and collected == 2 and all(row["dispatch_count"] >= len(CASE_IDS) and row["health_ok"] and row["process_clean"] and not row["fallback_used"] for row in rows) else "FAIL"
    aggregate = {
        "schema_version": "rmsnorm-g2-aggregate-v1", "aggregate_id": "rmsnorm-g2-aggregate-" + candidate["reviewed_sha"], "state": state, "required": True,
        "candidate": candidate, "tree_oid": candidate["git_tree_oid"],
        "matrix": {"path": "ci/matrix/rmsnorm-g2-v1.json", "sha256": sha256_file(repo / "ci/matrix/rmsnorm-g2-v1.json")},
        "model_lock": {"path": MODEL_LOCK_PATH, "sha256": model_lock_sha, "fingerprint": MODEL_LOCK_FINGERPRINT, "resolved_revision": RESOLVED_REVISION},
        "tolerance": {"path": TOLERANCE_PATH, "sha256": tolerance_sha, "tolerance_id": TOLERANCE_ID, "atol": 0.0078125, "rtol": 0.015625},
        "rows": rows,
        "counts": {"expected_rows": 2, "selected_rows": 2, "collected_rows": 2, "passed_rows": passed, "failed_rows": 2 - passed, "expected_cases": 12, "collected_cases": 6 * collected},
        "prerequisites": aggregate_prerequisites,
        "raw_data_policy": {"raw_model_git": False, "raw_slice_git": False, "raw_model_artifact": False, "raw_slice_artifact": False, "report_contains_paths": False, "report_contains_bytes": False},
    }
    return validate_aggregate(aggregate, repo)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--reports", nargs=2, type=Path, required=True)
    parser.add_argument("--artifacts", nargs=2, type=Path, required=True)
    parser.add_argument("--reviewed-sha", required=True)
    parser.add_argument("--tested-sha", required=True)
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--tree-oid", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate = {"reviewed_sha": args.reviewed_sha, "tested_sha": args.tested_sha, "workflow_sha": args.workflow_sha, "git_tree_oid": args.tree_oid, "worktree_clean": True, "revision_input": "full-sha"}
    try:
        aggregate = aggregate_reports(args.reports, args.artifacts, candidate=candidate, repo=args.repo.resolve(), strict_git=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / "rmsnorm-g2-aggregate.json"
        output.write_text(json.dumps(aggregate, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        output.with_name(output.name + ".sha256").write_text(sha256_file(output) + "\n", encoding="ascii")
    except (ContractError, OSError, ValueError) as exc:
        print(f"G2 aggregate: FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"G2 aggregate: {aggregate['state']}")
    return 0 if aggregate["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
