#!/usr/bin/env python3
"""Fail-closed aggregate for the two canonical model-free G1 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ContractError,
    ROOT,
    canonical_bytes,
    ensure_clean_worktree,
    identity as _git_identity,
    read_json,
    sha256_file,
)
from validate_g1_contracts import (  # noqa: E402
    EXPECTED_ROWS,
    EXPECTED_TOOLCHAIN_ID,
    validate_g1_matrix,
    validate_row,
)

AGGREGATE_SCHEMA = "ci/schema/g1-aggregate-v1.schema.json"
TOOLCHAIN_MANIFEST = "ci/toolchains/rocm-7.14.0.json"
MATRIX_MANIFEST = "ci/matrix/g1-runtime-v1.json"
REPORT_SCHEMA = "ci/schema/g1-report-v1.schema.json"
ARTIFACT_SCHEMA = "ci/schema/g1-runtime-artifact-v1.schema.json"


def git_identity(repo: Path = ROOT) -> dict[str, str]:
    """Patchable identity hook used by host-only contract tests."""

    return _git_identity(repo)


def validate_aggregate_schema(document: dict[str, Any], repo: Path = ROOT) -> None:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:  # pragma: no cover - locked host dependency
        raise ContractError("jsonschema is required for G1 aggregate validation") from exc
    schema_path = repo / AGGREGATE_SCHEMA
    if schema_path.is_symlink() or not schema_path.is_file():
        raise ContractError("G1 aggregate schema is missing or unsafe")
    schema = read_json(schema_path)
    if not isinstance(schema, dict):
        raise ContractError("G1 aggregate schema must be an object")
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document), key=lambda error: list(error.path))
    if errors:
        raise ContractError("G1 aggregate schema validation failed: " + "; ".join(error.message for error in errors[:8]))
    for relative in (TOOLCHAIN_MANIFEST, MATRIX_MANIFEST, REPORT_SCHEMA, ARTIFACT_SCHEMA, AGGREGATE_SCHEMA):
        path = repo / relative
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"G1 aggregate manifest is missing or unsafe: {relative}")
    matrix = validate_g1_matrix(repo)
    expected_hashes = {
        "toolchain_manifest_sha256": sha256_file(repo / TOOLCHAIN_MANIFEST),
        "matrix_manifest_sha256": hashlib.sha256(canonical_bytes(matrix)).hexdigest(),
        "report_schema_sha256": sha256_file(repo / REPORT_SCHEMA),
        "artifact_schema_sha256": sha256_file(repo / ARTIFACT_SCHEMA),
        "aggregate_schema_sha256": sha256_file(repo / AGGREGATE_SCHEMA),
    }
    for key, expected in expected_hashes.items():
        if document.get(key) != expected:
            raise ContractError(f"G1 aggregate {key} is stale")


def load_needs(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContractError("G1 needs file is missing or unsafe")
    document = read_json(path)
    if not isinstance(document, dict) or list(document) != list(EXPECTED_ROWS):
        raise ContractError("G1 needs must contain exactly both ordered canonical rows")
    for row_id in EXPECTED_ROWS:
        value = document[row_id]
        if not isinstance(value, dict) or set(value) != {"result"} or value["result"] != "success":
            raise ContractError(f"G1 needs.{row_id} is not a successful required job")
    return document


def _validate_inputs(
    *, run_id: str, run_attempt: int, reviewed_sha: str, tested_sha: str,
    workflow_sha: str, tree_oid: str | None,
) -> dict[str, Any]:
    if not isinstance(run_id, str) or not re.fullmatch(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$", run_id):
        raise ContractError("G1 run_id is invalid")
    if isinstance(run_attempt, bool) or not isinstance(run_attempt, int) or run_attempt < 1:
        raise ContractError("G1 run_attempt is invalid")
    values = {"reviewed_sha": reviewed_sha, "tested_sha": tested_sha, "workflow_sha": workflow_sha}
    for name, value in values.items():
        if not isinstance(value, str) or len(value) != 40 or value != value.lower() or any(char not in "0123456789abcdef" for char in value):
            raise ContractError(f"{name} is not a complete lowercase SHA")
    if len(set(values.values())) != 1:
        raise ContractError("G1 reviewed/tested/workflow SHA values differ")
    if tree_oid is None:
        raise ContractError("G1 aggregate requires an explicit immutable tree OID")
    if not isinstance(tree_oid, str) or len(tree_oid) != 40 or tree_oid != tree_oid.lower() or any(char not in "0123456789abcdef" for char in tree_oid):
        raise ContractError("G1 tree OID is not a complete lowercase SHA")
    return {
        "run_id": run_id,
        "run_attempt": run_attempt,
        "reviewed_sha": reviewed_sha,
        "tested_sha": tested_sha,
        "workflow_sha": workflow_sha,
        "git_tree_oid": tree_oid,
    }


def aggregate_results(
    *, needs_path: Path, artifact_dir: Path, repo: Path, output_dir: Path,
    run_id: str, run_attempt: int, reviewed_sha: str, tested_sha: str,
    workflow_sha: str, tree_oid: str | None = None,
    tool_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Validate and return a PASS aggregate; every failure raises ContractError."""

    expected_identity = _validate_inputs(
        run_id=run_id, run_attempt=run_attempt, reviewed_sha=reviewed_sha,
        tested_sha=tested_sha, workflow_sha=workflow_sha, tree_oid=tree_oid,
    )
    checked_out = git_identity(repo)
    ensure_clean_worktree(repo)
    if checked_out != {"commit": reviewed_sha, "tree": tree_oid}:
        raise ContractError("G1 aggregate candidate does not match the checked-out immutable commit/tree")
    load_needs(needs_path)
    matrix = validate_g1_matrix(repo)
    if not artifact_dir.is_absolute() or artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise ContractError("G1 artifact collection is missing or unsafe")
    resolved_artifact_dir = artifact_dir.resolve(strict=False)
    if resolved_artifact_dir != artifact_dir or resolved_artifact_dir.parent != Path("/tmp") or not resolved_artifact_dir.name.startswith("ullm-g1-"):
        raise ContractError("G1 artifact collection must be a private /tmp/ullm-g1-* directory")
    if artifact_dir.stat().st_uid != os.getuid() or artifact_dir.stat().st_mode & 0o077:
        raise ContractError("G1 artifact collection must be owned by the current user with mode 0700")
    if [path.name for path in sorted(artifact_dir.iterdir())] != list(EXPECTED_ROWS):
        raise ContractError("G1 artifact collection must contain exactly the ordered two row directories")
    rows = []
    for row_id in EXPECTED_ROWS:
        expected = next(row for row in matrix["rows"] if row["row_id"] == row_id)
        rows.append(
            validate_row(
                artifact_dir / row_id,
                row_id,
                expected,
                expected_identity,
                matrix,
                repo,
                tool_runner=tool_runner,
            )
        )
    if len({row["bdf"] for row in rows}) != 2 or len({row["uuid"] for row in rows}) != 2:
        raise ContractError("G1 aggregate contains duplicate canonical GPU identities")
    if len({row["artifact_sha256"] for row in rows}) != 2:
        raise ContractError("G1 aggregate contains duplicate runtime artifacts")
    if len({row["artifact_path"] for row in rows}) != 2 or len({row["staged_artifact_path"] for row in rows}) != 2:
        raise ContractError("G1 aggregate contains duplicate source or staged artifact paths")
    if any(row["toolchain_id"] != EXPECTED_TOOLCHAIN_ID for row in rows):
        raise ContractError("G1 aggregate rows do not share the exact ROCm toolchain")
    for key in ("run_id", "run_attempt", "reviewed_sha", "tested_sha", "workflow_sha", "git_tree_oid", "toolchain_id", "toolchain_manifest_sha256", "matrix_manifest_sha256", "artifact_schema_sha256"):
        if len({row[key] for row in rows}) != 1:
            raise ContractError(f"G1 aggregate rows do not share {key}")
    result = {
        "schema_version": "g1-aggregate-v1",
        "aggregate_id": f"g1-aggregate.{run_id}.{run_attempt}",
        "state": "PASS",
        "required": True,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "reviewed_sha": reviewed_sha,
        "tested_sha": tested_sha,
        "workflow_sha": workflow_sha,
        "git_tree_oid": tree_oid,
        "toolchain_id": EXPECTED_TOOLCHAIN_ID,
        "toolchain_manifest_sha256": sha256_file(repo / TOOLCHAIN_MANIFEST),
        "matrix_manifest_sha256": hashlib.sha256(canonical_bytes(matrix)).hexdigest(),
        "report_schema_sha256": sha256_file(repo / REPORT_SCHEMA),
        "artifact_schema_sha256": sha256_file(repo / ARTIFACT_SCHEMA),
        "aggregate_schema_sha256": sha256_file(repo / AGGREGATE_SCHEMA),
        "expected_rows": list(EXPECTED_ROWS),
        "rows": rows,
        "errors": [],
    }
    validate_aggregate_schema(result, repo)
    return result


def write_summary(output_dir: Path, summary: dict[str, Any], repo: Path = ROOT) -> None:
    if not output_dir.is_absolute() or output_dir.is_symlink() or output_dir.parent != Path("/tmp") or not output_dir.name.startswith("ullm-g1-"):
        raise ContractError("G1 aggregate output must be a private /tmp/ullm-g1-* directory")
    validate_aggregate_schema(summary, repo)
    data = canonical_bytes(summary)
    report_digest = hashlib.sha256(data).hexdigest()
    sidecar = f"{report_digest}  aggregate.json\n".encode("ascii")

    # Resolve the one allowed path component through a /tmp directory fd.
    # O_NOFOLLOW protects the directory open; all output members are then
    # created relative to that stable descriptor, so a path replacement cannot
    # redirect either write to a symlink or another directory.
    try:
        tmp_fd = os.open(
            "/tmp",
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        )
    except OSError as exc:
        raise ContractError(f"cannot open /tmp for G1 aggregate output: {exc}") from exc
    try:
        try:
            os.mkdir(output_dir.name, mode=0o700, dir_fd=tmp_fd)
        except FileExistsError:
            pass
        try:
            output_fd = os.open(
                output_dir.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=tmp_fd,
            )
        except OSError as exc:
            raise ContractError(f"G1 aggregate output directory is missing or unsafe: {exc}") from exc
    finally:
        os.close(tmp_fd)

    try:
        output_stat = os.fstat(output_fd)
        if not stat.S_ISDIR(output_stat.st_mode):
            raise ContractError("G1 aggregate output is not a directory")
        if output_stat.st_uid != os.getuid() or output_stat.st_mode & 0o077:
            raise ContractError("G1 aggregate output must be owned by the current user with mode 0700")
        try:
            existing_names = set(os.listdir(output_fd))
        except OSError as exc:
            raise ContractError(f"cannot inspect G1 aggregate output directory: {exc}") from exc
        allowed_names = {"aggregate.json", "aggregate.json.sha256"}
        if existing_names - allowed_names:
            raise ContractError("G1 aggregate output contains unknown files")
        if existing_names & allowed_names:
            raise ContractError("refusing to overwrite existing G1 aggregate output")

        def create_regular(name: str, content: bytes) -> None:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
            try:
                fd = os.open(name, flags, 0o600, dir_fd=output_fd)
            except FileExistsError as exc:
                raise ContractError(f"refusing to overwrite existing G1 aggregate output: {name}") from exc
            except OSError as exc:
                raise ContractError(f"cannot create G1 aggregate output {name}: {exc}") from exc
            try:
                member_stat = os.fstat(fd)
                if not stat.S_ISREG(member_stat.st_mode):
                    raise ContractError(f"G1 aggregate output member is not a regular file: {name}")
                offset = 0
                while offset < len(content):
                    written = os.write(fd, content[offset:])
                    if written <= 0:
                        raise ContractError(f"cannot write G1 aggregate output member: {name}")
                    offset += written
                os.fchmod(fd, 0o600)
            except OSError as exc:
                raise ContractError(f"cannot write G1 aggregate output member {name}: {exc}") from exc
            finally:
                os.close(fd)

        create_regular("aggregate.json", data)
        create_regular("aggregate.json.sha256", sidecar)
    finally:
        os.close(output_fd)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--needs-json", type=Path, required=True)
    result.add_argument("--artifact-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--repo", type=Path, default=ROOT)
    result.add_argument("--run-id", required=True)
    result.add_argument("--run-attempt", type=int, required=True)
    result.add_argument("--expected-reviewed-sha", "--reviewed-sha", dest="reviewed_sha", required=True)
    result.add_argument("--expected-tested-sha", "--tested-sha", dest="tested_sha", required=True)
    result.add_argument("--expected-workflow-sha", "--workflow-sha", dest="workflow_sha", required=True)
    result.add_argument("--expected-tree-oid", "--tree-oid", dest="tree_oid", required=True)
    result.add_argument("--strict-ci", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if not args.strict_ci:
            raise ContractError("G1 aggregation requires --strict-ci")
        if not args.output_dir.is_absolute() or args.output_dir.is_symlink():
            raise ContractError("G1 aggregate output must be an absolute non-symlink path")
        summary = aggregate_results(
            needs_path=args.needs_json, artifact_dir=args.artifact_dir,
            repo=args.repo.resolve(), output_dir=args.output_dir,
            run_id=args.run_id, run_attempt=args.run_attempt,
            reviewed_sha=args.reviewed_sha, tested_sha=args.tested_sha,
            workflow_sha=args.workflow_sha, tree_oid=args.tree_oid,
        )
        write_summary(args.output_dir, summary, args.repo.resolve())
    except (ContractError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"G1 aggregate: FAIL: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
