#!/usr/bin/env python3
"""Fail-closed aggregation for the three required host rows.

The primary interface is::

    python3 ci/tools/aggregate_host_results.py \
        --needs-json FILE --artifact-dir DIR --output-dir DIR

Each input row must be a ``report.json`` with a matching
``report.json.sha256`` sidecar.  The old ``--results-dir``, ``--needs-file``
and ``--output`` spellings remain parsing aliases for local callers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ContractError,
    ROOT,
    command_content_hash,
    command_hash,
    identity,
    load_manifests,
    manifest_bundle_hash,
    matrix_manifest_hash,
    parse_sidecar,
    parse_time,
    read_json,
    registered_row_commands,
    result_report_bytes,
    sha256_bytes,
    sha256_json,
    tuple_digest,
    utc_now,
    validate_result_payload,
)

EXIT_PASS = 0
EXIT_NONPASS = 1
EXIT_HARNESS = 3
SHA40 = re.compile(r"^[0-9a-f]{40}$")


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--needs-json", "--needs-file", dest="needs_json")
    parser.add_argument("--needs", help="JSON object mapping h0/h1/h2 to GitHub needs conclusions")
    parser.add_argument("--artifact-dir", "--results-dir", dest="artifact_dir")
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument("--output", dest="legacy_output", help="legacy alias; report.json is still written to its parent")
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument("--run-attempt", type=int, default=int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")))
    parser.add_argument(
        "--reviewed-sha", "--expected-reviewed-sha", dest="reviewed_sha",
        default=os.environ.get("REVIEWED_SHA"),
    )
    parser.add_argument(
        "--tested-sha", "--expected-tested-sha", dest="tested_sha",
        default=os.environ.get("TESTED_SHA"),
    )
    parser.add_argument(
        "--workflow-sha", "--expected-workflow-sha", dest="workflow_sha",
        default=os.environ.get("WORKFLOW_SHA"),
    )
    parser.add_argument("--tree-oid", "--git-tree-oid", dest="tree_oid", default=os.environ.get("GIT_TREE_OID"))
    parser.add_argument("--matrix-manifest-sha256", dest="matrix_manifest_sha256")
    parser.add_argument("--run-started-at", default=os.environ.get("CI_RUN_STARTED_AT"))
    parser.add_argument("--run-finished-at", default=os.environ.get("CI_RUN_FINISHED_AT"))
    parser.add_argument(
        "--strict-ci",
        action="store_true",
        help="aggregate only immutable required-ci row reports",
    )
    parser.add_argument(
        "--allow-local-development",
        action="store_true",
        help="explicitly aggregate local-development reports as non-immutable output",
    )
    return parser


def write_summary(output_dir: Path, summary: dict[str, Any], legacy_output: str | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = result_report_bytes(summary)
    report = output_dir / "report.json"
    report.write_bytes(raw)
    (output_dir / "report.json.sha256").write_text(
        f"{sha256_bytes(raw)}  {report.name}\n", encoding="utf-8"
    )
    if legacy_output:
        legacy = Path(legacy_output).resolve()
        if legacy != report.resolve():
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_bytes(raw)
            legacy.with_name(legacy.name + ".sha256").write_text(
                f"{sha256_bytes(raw)}  {legacy.name}\n", encoding="utf-8"
            )


def fail(message: str, *, output_dir: Path | None = None, legacy_output: str | None = None, code: int = EXIT_HARNESS) -> int:
    summary = {"schema_version": "host-required-v1", "state": "HARNESS_ERROR", "errors": [message]}
    if output_dir is not None:
        try:
            write_summary(output_dir, summary, legacy_output)
        except OSError as exc:
            print(f"host-required: cannot write failure report: {exc}", file=sys.stderr)
            return EXIT_HARNESS
    print(f"host-required: HARNESS_ERROR: {message}", file=sys.stderr)
    return code


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate needs row: {key}")
        result[key] = value
    return result


def load_needs(args: argparse.Namespace, *, required_rows: set[str]) -> dict[str, str]:
    if args.needs_json and args.needs:
        raise ContractError("use only one of --needs-json/--needs")
    if args.needs_json:
        try:
            with Path(args.needs_json).open("r", encoding="utf-8") as stream:
                value = json.load(stream, object_pairs_hook=_pairs_without_duplicates)
        except (OSError, ValueError) as exc:
            raise ContractError(f"cannot read needs JSON: {exc}") from exc
    elif args.needs:
        try:
            value = json.loads(args.needs, object_pairs_hook=_pairs_without_duplicates)
        except ValueError as exc:
            raise ContractError(f"invalid needs JSON: {exc}") from exc
    else:
        raise ContractError("needs JSON is required")
    if not isinstance(value, dict) or set(value) != required_rows:
        present = sorted(value) if isinstance(value, dict) else []
        missing = sorted(required_rows - set(present))
        unknown = sorted(set(present) - required_rows)
        detail = f"missing={missing} unknown={unknown}" if isinstance(value, dict) else "needs is not an object"
        raise ContractError(f"needs rows are not exactly h0/h1/h2: {detail}")
    normalized: dict[str, str] = {}
    for row_id, raw in value.items():
        if isinstance(raw, dict):
            conclusion = raw.get("result")
        else:
            conclusion = raw
        if not isinstance(conclusion, str):
            raise ContractError(f"needs.{row_id} has no string result")
        normalized[row_id] = conclusion
    return normalized


def exact_sha(value: str, name: str) -> str:
    if not isinstance(value, str) or not SHA40.fullmatch(value):
        raise ContractError(f"{name} must be a 40-character lowercase SHA")
    return value


def discover_reports(artifact_dir: Path, output_dir: Path) -> list[Path]:
    if not artifact_dir.is_dir():
        raise ContractError(f"artifact directory does not exist: {artifact_dir}")
    reports: list[Path] = []
    for path in sorted(artifact_dir.rglob("report.json")):
        if not path.is_file():
            continue
        if path.resolve().is_relative_to(output_dir.resolve()):
            continue
        reports.append(path)
    if not reports:
        raise ContractError("zero result collection: no JSON row reports found")
    return reports


def main(argv: list[str] | None = None) -> int:
    args = args_parser().parse_args(argv)
    output_dir: Path | None = None
    try:
        if not args.artifact_dir:
            raise ContractError("--artifact-dir is required")
        if args.output_dir and args.legacy_output:
            raise ContractError("use only one of --output-dir/--output")
        output_dir = Path(args.output_dir).resolve() if args.output_dir else (
            Path(args.legacy_output).resolve().parent if args.legacy_output else None
        )
        if output_dir is None:
            raise ContractError("--output-dir is required")
        if args.strict_ci and args.allow_local_development:
            raise ContractError(
                "--allow-local-development is prohibited under --strict-ci"
            )
        if not args.strict_ci and not args.allow_local_development:
            raise ContractError(
                "local aggregation requires explicit --allow-local-development"
            )
        if args.strict_ci and not all(
            (args.reviewed_sha, args.tested_sha, args.workflow_sha)
        ):
            raise ContractError(
                "strict CI requires explicit reviewed/tested/workflow SHA values"
            )
        repo = args.repo.resolve()
        suites, host, _ = load_manifests(repo)
        expected_rows = {row["row_id"]: row for row in host["rows"]}
        if set(expected_rows) != {"h0", "h1", "h2"}:
            raise ContractError("host matrix is not exactly h0/h1/h2")
        needs = load_needs(args, required_rows=set(expected_rows))
        if any(value != "success" for value in needs.values()):
            bad = ",".join(f"{key}={value}" for key, value in sorted(needs.items()) if value != "success")
            summary = {"schema_version": "host-required-v1", "state": "FAIL", "errors": [f"needs non-success: {bad}"]}
            write_summary(output_dir, summary, args.legacy_output)
            print(f"host-required: needs contains non-success conclusion: {bad}", file=sys.stderr)
            return EXIT_NONPASS

        git_identity = identity(repo)
        expected_run_id = args.run_id or f"local-{git_identity['commit'][:12]}"
        if not isinstance(expected_run_id, str) or not (1 <= len(expected_run_id) <= 128):
            raise ContractError("invalid run id")
        if args.run_attempt < 1:
            raise ContractError("run attempt must be positive")
        expected = {
            "run_id": expected_run_id,
            "run_attempt": args.run_attempt,
            "reviewed_sha": exact_sha(args.reviewed_sha or git_identity["commit"], "reviewed_sha"),
            "tested_sha": exact_sha(args.tested_sha or git_identity["commit"], "tested_sha"),
            "workflow_sha": exact_sha(args.workflow_sha or git_identity["commit"], "workflow_sha"),
            "git_tree_oid": exact_sha(args.tree_oid or git_identity["tree"], "git_tree_oid"),
            "matrix_manifest_sha256": args.matrix_manifest_sha256 or matrix_manifest_hash(repo),
            "manifest_bundle_sha256": manifest_bundle_hash(repo),
        }
        sha_values = {
            expected["reviewed_sha"],
            expected["tested_sha"],
            expected["workflow_sha"],
        }
        if sha_values != {git_identity["commit"]}:
            raise ContractError(
                "reviewed/tested/workflow SHA values must all match checked-out HEAD"
            )
        if expected["git_tree_oid"] != git_identity["tree"]:
            raise ContractError("expected Git tree OID does not match checked-out HEAD")
        if not re.fullmatch(r"[0-9a-f]{64}", expected["matrix_manifest_sha256"]):
            raise ContractError("matrix manifest SHA must be 64 lowercase hex characters")
        expected_evidence_mode = (
            "required-ci" if args.strict_ci else "local-development"
        )

        artifact_dir = Path(args.artifact_dir).resolve()
        report_paths = discover_reports(artifact_dir, output_dir)
        reports: dict[str, tuple[dict[str, Any], Path]] = {}
        duplicate: list[str] = []
        unknown: list[str] = []
        for report_path in report_paths:
            try:
                payload = read_json(report_path)
                if not isinstance(payload, dict):
                    raise ContractError("result is not an object")
                validate_result_payload(payload)
                row_id = payload.get("matrix_row_id")
                if row_id not in expected_rows:
                    unknown.append(str(row_id))
                    continue
                if row_id in reports:
                    duplicate.append(str(row_id))
                    continue
                reports[str(row_id)] = (payload, report_path)
            except (ContractError, OSError, ValueError, TypeError) as exc:
                raise ContractError(f"{report_path.relative_to(artifact_dir)}: {exc}") from exc
        if unknown:
            raise ContractError(f"unknown result row(s): {','.join(sorted(unknown))}")
        if duplicate:
            raise ContractError(f"duplicate result row(s): {','.join(sorted(duplicate))}")
        missing = sorted(set(expected_rows) - set(reports))
        if missing:
            raise ContractError(f"missing result row(s): {','.join(missing)}")

        now = utc_now()
        start = parse_time(args.run_started_at) if args.run_started_at else now - timedelta(minutes=15)
        end = parse_time(args.run_finished_at) if args.run_finished_at else now
        if end < start:
            raise ContractError("run finished before run started")
        failures: list[str] = []
        for row_id, row in expected_rows.items():
            payload, report_path = reports[row_id]
            if payload.get("evidence_mode") != expected_evidence_mode:
                raise ContractError(
                    f"{row_id}: evidence_mode must be {expected_evidence_mode}"
                )
            for key in ("run_id", "run_attempt", "reviewed_sha", "tested_sha", "workflow_sha", "git_tree_oid", "matrix_manifest_sha256"):
                if payload.get(key) != expected[key]:
                    raise ContractError(f"{row_id}: {key} does not match current run")
            if payload.get("tier") != row["tier"] or payload.get("required") is not True or payload.get("suite_id") != f"host-{row_id}":
                raise ContractError(f"{row_id}: tier/required/suite identity mismatch")
            if payload.get("seed") != row["seed"]:
                raise ContractError(f"{row_id}: seed does not match matrix")
            if payload.get("tuple_digest") != tuple_digest(row):
                raise ContractError(f"{row_id}: tuple digest mismatch")
            artifact = payload.get("artifact")
            if not isinstance(artifact, dict) or artifact.get("manifest_sha256") != expected["manifest_bundle_sha256"]:
                raise ContractError(f"{row_id}: manifest hash mismatch")
            if artifact.get("content_sha256") != command_content_hash(payload["steps"]):
                raise ContractError(f"{row_id}: command content hash mismatch")
            expected_command_records = registered_row_commands(suites, row, repo)
            expected_command_ids = [
                command_id for command_id, _ in expected_command_records
            ]
            expected_commands = [
                command for _, command in expected_command_records
            ]
            if payload.get("command") != expected_commands:
                raise ContractError(
                    f"{row_id}: command sequence does not match suite registry"
                )
            if payload.get("command_sha256") != command_hash(expected_commands):
                raise ContractError(
                    f"{row_id}: command hash does not match suite registry"
                )
            resource = payload.get("resource")
            if (
                not isinstance(resource, dict)
                or resource.get("commands_expected") != len(expected_command_ids)
            ):
                raise ContractError(
                    f"{row_id}: expected command count does not match suite registry"
                )
            actual_step_ids = [step["step_id"] for step in payload["steps"]]
            if actual_step_ids != expected_command_ids[:len(actual_step_ids)]:
                raise ContractError(
                    f"{row_id}: executed command identities do not match suite registry"
                )
            if payload.get("toolchain_sha256") != sha256_json(payload["toolchain"]):
                raise ContractError(f"{row_id}: toolchain hash mismatch")
            sidecar = report_path.with_name(report_path.name + ".sha256")
            if not sidecar.exists():
                raise ContractError(f"{row_id}: missing report.json.sha256 sidecar")
            if parse_sidecar(sidecar) != sha256_bytes(report_path.read_bytes()):
                raise ContractError(f"{row_id}: report hash does not match sidecar")
            created = parse_time(payload["created_at"])
            started = parse_time(payload["started_at"])
            finished = parse_time(payload["finished_at"])
            if not (created <= started <= finished):
                raise ContractError(f"{row_id}: result timestamps are not ordered")
            if created < start or finished > end or created > now or finished > now:
                raise ContractError(f"{row_id}: stale or future result timestamp")
            if payload["state"] != "PASS":
                failures.append(f"{row_id}: {payload['state']}")

        summary = {
            "schema_version": "host-required-v1",
            "state": "PASS" if not failures else "FAIL",
            "run_id": expected["run_id"],
            "run_attempt": expected["run_attempt"],
            "matrix_manifest_sha256": expected["matrix_manifest_sha256"],
            "rows": {row_id: {"state": reports[row_id][0]["state"], "report": reports[row_id][1].relative_to(artifact_dir).as_posix()} for row_id in sorted(reports)},
            "errors": failures,
        }
        write_summary(output_dir, summary, args.legacy_output)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return EXIT_PASS if not failures else EXIT_NONPASS
    except (ContractError, OSError, ValueError, KeyError, TypeError) as exc:
        return fail(str(exc), output_dir=output_dir, legacy_output=args.legacy_output)


if __name__ == "__main__":
    raise SystemExit(main())
