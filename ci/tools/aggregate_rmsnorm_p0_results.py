#!/usr/bin/env python3
"""Fail-closed aggregation for exactly two canonical RMSNorm P0 rows."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from common import ContractError, ROOT, canonical_bytes, read_json, sha256_file, sha256_json  # noqa: E402
from validate_rmsnorm_p0_contracts import (  # noqa: E402
    CASE_IDS, MATRIX_PATH, MODEL_LOCK_FINGERPRINT, MODEL_LOCK_PATH, P0_BINARY,
    REVIEW_POLICY_PATH, RESOLVED_REVISION, ROWS, TARGETS, TOTAL_DISPATCHES,
    candidate_sha256, case_set_sha256, source_set, validate_aggregate,
    validate_artifact, validate_candidate, validate_disposition, validate_report,
)

MAX_REPORT_AGE = timedelta(hours=24)


def generate_review_disposition(
    report_paths: Iterable[Path],
    artifact_paths: Iterable[Path],
    reports: Iterable[dict[str, Any]],
    *,
    candidate: dict[str, Any],
    reviewer: str,
    reason: str,
    reviewed_at: str,
    repo: Path = ROOT,
) -> dict[str, Any]:
    """Generate only from explicit human review fields, then validate it."""

    report_paths = tuple(report_paths)
    artifact_paths = tuple(artifact_paths)
    reports = tuple(reports)
    if len(report_paths) != 2 or len(artifact_paths) != 2 or len(reports) != 2:
        raise ContractError("P0 review disposition generation requires exactly two rows")
    rows = []
    for order, (report_path, artifact_path, report) in enumerate(
        zip(report_paths, artifact_paths, reports, strict=True)
    ):
        rows.append({
            "order": order,
            "row_id": ROWS[order],
            "target": TARGETS[order],
            "report_sha256": sha256_file(report_path),
            "artifact_sha256": sha256_file(artifact_path),
            "measurement_sha256": report["measurement_sha256"],
            "complete_measurements": len(report["measurements"]) == 5,
            "cases": [
                {"order": case["order"], "id": case["id"], **case["summary"]}
                for case in report["measurements"]
            ],
        })
    document = {
        "schema_version": "rmsnorm-p0-review-disposition-v1",
        "disposition_id": f"rmsnorm-p0-review-{candidate['reviewed_sha']}",
        "performance_sanity_disposition": "review_required",
        "threshold": {"approved": False, "threshold_id": None, "metric_thresholds": []},
        "candidate": candidate,
        "tree_oid": candidate["git_tree_oid"],
        "matrix": {"path": MATRIX_PATH, "sha256": sha256_file(repo / MATRIX_PATH)},
        "review_policy": {"path": REVIEW_POLICY_PATH, "sha256": sha256_file(repo / REVIEW_POLICY_PATH)},
        "case_set_sha256": case_set_sha256(repo),
        "model_lock": {"path": MODEL_LOCK_PATH, "sha256": sha256_file(repo / MODEL_LOCK_PATH), "fingerprint": MODEL_LOCK_FINGERPRINT, "resolved_revision": RESOLVED_REVISION},
        "source_set_sha256": source_set(repo)["sha256"],
        "review": {"decision": "accept_observation_without_threshold", "reviewer": reviewer, "reason": reason, "reviewed_at": reviewed_at},
        "canonical_rows": rows,
        "claims": {"optimized": False, "faster_than_other_engine": False, "performance_hard_gate_established": False},
    }
    return validate_disposition(
        document,
        repo,
        reports=reports,
        report_sha256s=[sha256_file(path) for path in report_paths],
        artifact_sha256s=[sha256_file(path) for path in artifact_paths],
    )


def _parse_time(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ContractError("P0 aggregate received a non-UTC report timestamp")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise ContractError("P0 aggregate received an invalid report timestamp") from exc


def aggregate_reports(
    report_paths: Iterable[Path],
    artifact_paths: Iterable[Path],
    disposition_path: Path,
    *,
    candidate: dict[str, Any],
    repo: Path = ROOT,
    strict_git: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    report_paths = tuple(report_paths)
    artifact_paths = tuple(artifact_paths)
    if len(report_paths) != 2 or len(artifact_paths) != 2:
        raise ContractError("P0 aggregate requires exactly two reports and two artifacts")
    validate_candidate(candidate, repo, strict_git=strict_git)
    for path in (*report_paths, *artifact_paths, disposition_path):
        if path.is_symlink() or not path.is_file():
            raise ContractError("P0 aggregate inputs must be regular non-symlink files")
    report_documents = [read_json(path) for path in report_paths]
    artifact_documents = [read_json(path) for path in artifact_paths]
    reports = [validate_report(document, repo) for document in report_documents]
    artifacts = [
        validate_artifact(document, repo, binary_path=path.parent / P0_BINARY)
        for document, path in zip(artifact_documents, artifact_paths, strict=True)
    ]
    if [report["target"] for report in reports] != list(TARGETS) or [report["row_id"] for report in reports] != list(ROWS):
        raise ContractError("P0 reports are not the exact ordered canonical rows")
    if [artifact["target"] for artifact in artifacts] != list(TARGETS) or [artifact["row_id"] for artifact in artifacts] != list(ROWS):
        raise ContractError("P0 artifacts are not the exact ordered canonical rows")
    if any(report["candidate"] != candidate for report in reports) or any(artifact["candidate"] != candidate for artifact in artifacts):
        raise ContractError("P0 aggregate contains a stale or mixed candidate")
    run = reports[0]["run"]
    if any(report["run"] != run for report in reports):
        raise ContractError("P0 aggregate mixes run ID/attempt")
    report_digests = [sha256_file(path) for path in report_paths]
    artifact_digests = [sha256_file(path) for path in artifact_paths]
    disposition = validate_disposition(
        read_json(disposition_path), repo, reports=reports,
        report_sha256s=report_digests, artifact_sha256s=artifact_digests,
    )
    if disposition["candidate"] != candidate:
        raise ContractError("P0 review disposition names another candidate")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    rows: list[dict[str, Any]] = []
    for order, (report, artifact, report_digest, artifact_digest) in enumerate(
        zip(reports, artifacts, report_digests, artifact_digests, strict=True)
    ):
        if report["artifact"]["artifact_id"] != artifact["artifact_id"] or report["artifact"]["artifact_sha256"] != artifact_digest or report["artifact"]["binary_sha256"] != artifact["binary"]["sha256"] or report["artifact"]["source_set_sha256"] != artifact["source_set"]["sha256"]:
            raise ContractError("P0 report/artifact/source identity mismatch")
        finished = _parse_time(report["execution"]["finished_at"])
        if finished > current + timedelta(minutes=5) or current - finished > MAX_REPORT_AGE:
            raise ContractError("P0 report is stale or from the future")
        health_ok = (
            report["health_pre"]["state"] == "OK"
            and report["health_post"]["state"] == "OK"
            and report["health_pre"]["available"]
            and report["health_post"]["available"]
            and report["health_pre"]["reliable"]
            and report["health_post"]["reliable"]
            and report["health_post"]["ras_uncorrectable_count"] <= report["health_pre"]["ras_uncorrectable_count"]
        )
        process_clean = all(
            report[name]["state"] == "CLEAN"
            and not report[name]["residual_runner_children"]
            and not report[name]["gpu_processes"]
            for name in ("process_pre", "process_post")
        )
        rows.append({
            "order": order, "row_id": report["row_id"], "target": report["target"], "state": report["state"],
            "report_sha256": report_digest, "artifact_sha256": artifact_digest,
            "candidate_sha256": candidate_sha256(candidate),
            "tuple_sha256": sha256_json(report["device"]),
            "measurement_sha256": report["measurement_sha256"],
            "collected_cases": report["collection"]["collected_cases"],
            "dispatch_count": report["dispatch"]["dispatch_count"],
            "fallback_used": report["dispatch"]["fallback_used"],
            "health_ok": health_ok, "process_clean": process_clean,
        })
    passed = sum(row["state"] == "PASS" for row in rows)
    complete = sum(row["collected_cases"] == len(CASE_IDS) for row in rows)
    state = "PASS" if passed == 2 and complete == 2 and all(row["dispatch_count"] == TOTAL_DISPATCHES and row["health_ok"] and row["process_clean"] and not row["fallback_used"] for row in rows) else "FAIL"
    aggregate = {
        "schema_version": "rmsnorm-p0-aggregate-v1",
        "aggregate_id": f"rmsnorm-p0-aggregate-{candidate['reviewed_sha']}",
        "state": state, "required": True, "candidate": candidate,
        "tree_oid": candidate["git_tree_oid"], "run": run,
        "matrix": {"path": MATRIX_PATH, "sha256": sha256_file(repo / MATRIX_PATH)},
        "review_policy": {"path": REVIEW_POLICY_PATH, "sha256": sha256_file(repo / REVIEW_POLICY_PATH)},
        "case_set_sha256": case_set_sha256(repo),
        "model_lock": {"path": MODEL_LOCK_PATH, "sha256": sha256_file(repo / MODEL_LOCK_PATH), "fingerprint": MODEL_LOCK_FINGERPRINT, "resolved_revision": RESOLVED_REVISION},
        "source_set_sha256": source_set(repo)["sha256"],
        "rows": rows,
        "counts": {"expected_rows": 2, "selected_rows": 2, "collected_rows": complete, "passed_rows": passed, "failed_rows": 2 - passed, "expected_cases": 10, "collected_cases": sum(row["collected_cases"] for row in rows)},
        "review_disposition": {"path": disposition_path.name, "sha256": sha256_file(disposition_path), "disposition_id": disposition["disposition_id"], "performance_sanity_disposition": disposition["performance_sanity_disposition"], "reviewer": disposition["review"]["reviewer"], "reviewed_at": disposition["review"]["reviewed_at"]},
        "claims": {"optimized": False, "faster_than_other_engine": False, "performance_hard_gate_established": False},
    }
    return validate_aggregate(aggregate, repo)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--reports", nargs=2, type=Path, required=True)
    parser.add_argument("--artifacts", nargs=2, type=Path, required=True)
    parser.add_argument("--review-disposition", type=Path)
    parser.add_argument("--reviewer")
    parser.add_argument("--reason")
    parser.add_argument("--reviewed-at")
    parser.add_argument("--reviewed-sha", required=True)
    parser.add_argument("--tested-sha", required=True)
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--tree-oid", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate = {"reviewed_sha": args.reviewed_sha, "tested_sha": args.tested_sha, "workflow_sha": args.workflow_sha, "git_tree_oid": args.tree_oid, "worktree_clean": True, "revision_input": "full-sha"}
    try:
        disposition_path = args.review_disposition
        if disposition_path is None:
            if args.reviewer is None or args.reason is None or args.reviewed_at is None:
                raise ContractError("P0 disposition generation requires explicit reviewer, reason, and reviewed_at")
            reports = [validate_report(read_json(path), args.repo.resolve()) for path in args.reports]
            disposition = generate_review_disposition(
                args.reports,
                args.artifacts,
                reports,
                candidate=candidate,
                reviewer=args.reviewer,
                reason=args.reason,
                reviewed_at=args.reviewed_at,
                repo=args.repo.resolve(),
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            disposition_path = args.output_dir / "rmsnorm-p0-review-disposition.json"
            disposition_path.write_bytes(canonical_bytes(disposition))
        aggregate = aggregate_reports(
            args.reports, args.artifacts, disposition_path,
            candidate=candidate, repo=args.repo.resolve(), strict_git=True,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / "rmsnorm-p0-aggregate.json"
        output.write_bytes(canonical_bytes(aggregate))
        output.with_name(output.name + ".sha256").write_text(sha256_file(output) + "\n", encoding="ascii")
    except (ContractError, OSError, ValueError) as exc:
        print(f"P0 aggregate: FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"P0 aggregate: {aggregate['state']}")
    return 0 if aggregate["state"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
