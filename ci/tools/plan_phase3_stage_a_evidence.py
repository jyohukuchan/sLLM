#!/usr/bin/env python3
"""Plan Phase 3 Stage A evidence without executing any evidence activity.

The planner is a host-only contract boundary.  It reads reviewed matrices,
validators, workflow expectations, and builder layout helpers, then emits one
canonical JSON document.  It deliberately has no execute mode.
"""

from __future__ import annotations

import argparse
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ContractError,
    canonical_bytes,
    load_manifests,
    sha256_file,
    sha256_json,
)
import build_rmsnorm_g1_runtime as g1_builder  # noqa: E402
import validate_json_manifests as workflow_contracts  # noqa: E402
import validate_matrix as matrix_registry  # noqa: E402
import validate_rmsnorm_g1_contracts as g1_contracts  # noqa: E402
import validate_rmsnorm_g2_contracts as g2_contracts  # noqa: E402
import validate_rmsnorm_p0_contracts as p0_contracts  # noqa: E402


PLAN_SCHEMA = "ci/schema/phase3-stage-a-evidence-plan-v1.schema.json"
PLAN_ID = "phase3-stage-a-evidence-plan-v1"
EVIDENCE_STATE = "NOT_EXECUTED"
WORKFLOW_RUN_ID_RE = re.compile(r"^[1-9][0-9]*$")
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_RUN_ROOT_BYTES = 35
AF_UNIX_PATH_MAX_BYTES = 108

# These are the small, reviewed authority inputs used to derive the plan.  The
# list is intentionally data-only: no shell recipe or upload expansion is an
# authority of this planner.
AUTHORITY_FILES = tuple(sorted({
    PLAN_SCHEMA,
    "ci/tools/plan_phase3_stage_a_evidence.py",
    "ci/tools/common.py",
    "ci/matrix/suites-v1.json",
    "ci/matrix/host-v1.json",
    "ci/matrix/path-to-suite-v1.json",
    "ci/tools/build_rmsnorm_g1_runtime.py",
    "ci/tools/build_rmsnorm_g2_runtime.py",
    "ci/tools/build_rmsnorm_p0_runtime.py",
    "ci/tools/orchestrate_rmsnorm_g1_evidence.py",
    "ci/tools/validate_json_manifests.py",
    "ci/tools/validate_matrix.py",
    "ci/tools/validate_rmsnorm_g1_contracts.py",
    "ci/tools/validate_rmsnorm_g2_contracts.py",
    "ci/tools/validate_rmsnorm_p0_contracts.py",
    ".github/workflows/semantic-rmsnorm-g1.yml",
    ".github/workflows/rmsnorm-h3-compile.yml",
    g1_contracts.MATRIX_MANIFEST,
    g1_contracts.MATRIX_SCHEMA,
    g1_contracts.ARTIFACT_SCHEMA,
    g1_contracts.REPORT_SCHEMA,
    g1_contracts.AGGREGATE_SCHEMA,
    g2_contracts.MATRIX_PATH,
    g2_contracts.G2_BUILD_INPUTS_PATH,
    g2_contracts.TOLERANCE_PATH,
    *g2_contracts.SCHEMAS.values(),
    p0_contracts.MATRIX_PATH,
    p0_contracts.REVIEW_POLICY_PATH,
    p0_contracts.P0_PUBLIC_PATH_INPUTS_PATH,
    *p0_contracts.SCHEMAS.values(),
    workflow_contracts.H3_RMSNORM_MATRIX,
    *workflow_contracts.H3_RMSNORM_SCHEMA_FILES,
}))


class PlanError(ContractError):
    """A fail-closed Phase 3 Stage A plan violation."""


IdentityVerifier = Callable[[Path, Mapping[str, Any]], Mapping[str, Any]]


def _require_repo(repo: Path) -> Path:
    requested = Path(repo)
    if not requested.is_absolute() or "\x00" in str(requested):
        raise PlanError("repo must be an absolute path")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise PlanError("repo cannot be resolved") from exc
    if requested != resolved or resolved.is_symlink() or not resolved.is_dir():
        raise PlanError("repo must be an absolute resolved non-symlink directory")
    return resolved


def _require_run_root(run_root: Path) -> Path:
    requested = Path(run_root)
    if not requested.is_absolute() or "\x00" in str(requested):
        raise PlanError("run-root must be absolute")
    if len(str(requested).encode("utf-8")) > MAX_RUN_ROOT_BYTES:
        raise PlanError("run-root is not short enough for the evidence socket contract")
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as exc:
        raise PlanError("run-root parent must already exist") from exc
    if requested.parent != parent or not parent.is_dir() or parent.is_symlink():
        raise PlanError("run-root parent must be an existing non-symlink directory")
    if requested.exists() or requested.is_symlink():
        raise PlanError("run-root must be a non-existing path")
    return requested


def _require_run_identity(run_id: str, run_attempt: str | int) -> tuple[str, int]:
    if not isinstance(run_id, str) or WORKFLOW_RUN_ID_RE.fullmatch(run_id) is None:
        raise PlanError("run-id must match the workflow numeric identity contract")
    attempt_text = str(run_attempt)
    if WORKFLOW_RUN_ID_RE.fullmatch(attempt_text) is None:
        raise PlanError("run-attempt must be a positive numeric identity")
    return run_id, int(attempt_text)


def _candidate_input(
    reviewed_sha: str,
    tested_sha: str,
    workflow_sha: str,
    tree_oid: str,
) -> dict[str, Any]:
    candidate = {
        "reviewed_sha": reviewed_sha,
        "tested_sha": tested_sha,
        "workflow_sha": workflow_sha,
        "git_tree_oid": tree_oid,
        "worktree_clean": True,
        "revision_input": "full-sha",
    }
    for name in ("reviewed_sha", "tested_sha", "workflow_sha", "git_tree_oid"):
        value = candidate[name]
        if not isinstance(value, str) or SHA40_RE.fullmatch(value) is None or value == "0" * 40:
            raise PlanError(f"{name} must be a nonzero full lowercase SHA")
    return candidate


def _strict_identity_verifier(repo: Path, expected: Mapping[str, Any]) -> Mapping[str, Any]:
    """Verify the candidate against Git; this is the only CLI identity path."""

    return g2_contracts.validate_candidate(expected, repo, strict_git=True)


def api_only_identity_verifier(_repo: Path, expected: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a supplied identity without Git or subprocess access.

    This intentionally named verifier is an API-only test seam.  It is not a
    CLI option and is never selected by :func:`main`.
    """

    return g2_contracts.validate_candidate(expected, _repo, strict_git=False)


def _verify_identity(
    repo: Path,
    expected: Mapping[str, Any],
    verifier: IdentityVerifier,
) -> dict[str, Any]:
    try:
        observed = verifier(repo, expected)
    except (ContractError, OSError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        raise PlanError(f"candidate identity verification failed: {exc}") from exc
    if not isinstance(observed, Mapping):
        raise PlanError("candidate identity verifier returned a non-object")
    if dict(observed) != dict(expected):
        raise PlanError("candidate identity verifier returned a mismatched candidate")
    try:
        checked = g2_contracts.validate_candidate(dict(observed), repo, strict_git=False)
    except (ContractError, TypeError, ValueError) as exc:
        raise PlanError(f"candidate identity is malformed: {exc}") from exc
    if checked != dict(observed):
        raise PlanError("candidate identity verifier returned an unstable identity")
    return checked


def _ordered_target_rows(
    label: str,
    matrix: Mapping[str, Any],
    field: str,
) -> list[dict[str, Any]]:
    raw = matrix.get(field)
    if not isinstance(raw, list) or len(raw) != 2:
        raise PlanError(f"{label} matrix must contain exactly two ordered target rows")
    rows = [dict(row) for row in raw if isinstance(row, Mapping)]
    if len(rows) != 2:
        raise PlanError(f"{label} matrix target rows are malformed")
    if field == "rows" and all("order" not in row for row in rows):
        for order, row in enumerate(rows):
            row["order"] = order
    elif [row.get("order") for row in rows] != [0, 1]:
        raise PlanError(f"{label} target order drifted")
    targets = [row.get("target") for row in rows]
    if len(set(targets)) != 2 or targets != ["gfx1030", "gfx1201"]:
        raise PlanError(f"{label} target order is duplicated or non-canonical")
    return rows


def _validate_target_consistency(
    g1_matrix: Mapping[str, Any],
    g2_matrix: Mapping[str, Any],
    p0_matrix: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    g1_rows = _ordered_target_rows("G1", g1_matrix, "rows")
    g2_rows = _ordered_target_rows("G2", g2_matrix, "targets")
    p0_rows = _ordered_target_rows("P0", p0_matrix, "targets")
    orders = [[row["target"] for row in rows] for rows in (g1_rows, g2_rows, p0_rows)]
    if any(order != orders[0] for order in orders[1:]):
        raise PlanError("G1/G2/P0 target order or duplication drifted")
    return g1_rows, g2_rows, p0_rows


def _authority_records(repo: Path) -> list[dict[str, str]]:
    strict_repo = repo.resolve(strict=True)
    records: list[dict[str, str]] = []
    for relative in AUTHORITY_FILES:
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise PlanError(f"authority path is unsafe: {relative}")
        path = repo / relative
        absolute = Path(path.anchor) if path.is_absolute() else Path.cwd()
        for component in path.parts[1:] if path.is_absolute() else path.parts:
            absolute /= component
            try:
                if stat.S_ISLNK(absolute.lstat().st_mode):
                    raise PlanError(f"authority path contains a symlink component: {relative}")
            except OSError as exc:
                raise PlanError(f"authority file is unavailable: {relative}") from exc
        try:
            strict_path = path.resolve(strict=True)
        except OSError as exc:
            raise PlanError(f"authority file cannot be strictly resolved: {relative}") from exc
        if not strict_path.is_relative_to(strict_repo):
            raise PlanError(f"authority file resolves outside the repository: {relative}")
        try:
            details = strict_path.lstat()
        except OSError as exc:
            raise PlanError(f"authority file cannot be stated: {relative}") from exc
        if not stat.S_ISREG(details.st_mode) or stat.S_ISLNK(details.st_mode):
            raise PlanError(f"authority file is not a regular file: {relative}")
        records.append({"path": relative, "sha256": sha256_file(strict_path)})
    return records


def _validate_workflow_authorities(repo: Path) -> None:
    """Parse and validate only the two workflows that define this plan."""

    try:
        import yaml
    except ImportError as exc:
        raise PlanError(f"workflow YAML dependency missing: {exc}") from exc

    workflows = (
        ".github/workflows/rmsnorm-h3-compile.yml",
        ".github/workflows/semantic-rmsnorm-g1.yml",
    )
    for relative in workflows:
        path = repo / relative
        try:
            with path.open("r", encoding="utf-8") as stream:
                document = yaml.safe_load(stream)
            if not isinstance(document, dict) or not isinstance(document.get("jobs"), dict):
                raise PlanError(f"workflow has no jobs object: {relative}")
            workflow_contracts.validate_workflow(path, document)
        except PlanError:
            raise
        except Exception as exc:
            raise PlanError(f"workflow authority is invalid: {relative}: {exc}") from exc


def _g1_environment(target: str, layout: g1_builder.SemanticG1BuildLayout) -> dict[str, str]:
    # This calls the existing builder's environment derivation only.  It does
    # not invoke Cargo, CMake, HIP, a compiler, or any output-producing step.
    try:
        return g1_builder.build_environment(target, layout.cargo_target_dir, layout.native_hip_build_dir)
    except (ContractError, OSError, TypeError, ValueError) as exc:
        raise PlanError(f"G1 build environment cannot be derived: {exc}") from exc


def _validate_plan_schema(plan: Mapping[str, Any], repo: Path) -> None:
    try:
        from jsonschema import Draft202012Validator, FormatChecker

        schema_path = repo / PLAN_SCHEMA
        schema = __import__("json").loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(plan),
            key=lambda error: list(error.path),
        )
    except (ImportError, OSError, ValueError) as exc:
        raise PlanError(f"evidence-plan schema cannot be validated: {exc}") from exc
    if errors:
        detail = "; ".join(f"{'.'.join(map(str, error.path)) or '<root>'}: {error.message}" for error in errors[:5])
        raise PlanError(f"evidence-plan schema rejected the generated plan: {detail}")


def build_plan(
    *,
    repo: Path,
    run_root: Path,
    run_id: str,
    run_attempt: str | int,
    reviewed_sha: str,
    tested_sha: str,
    workflow_sha: str,
    tree_oid: str,
    identity_verifier: IdentityVerifier = _strict_identity_verifier,
) -> dict[str, Any]:
    """Validate contracts and construct one side-effect-free evidence plan."""

    resolved_repo = _require_repo(repo)
    planned_run_root = _require_run_root(run_root)
    canonical_run_id, canonical_attempt = _require_run_identity(run_id, run_attempt)
    expected_identity = _candidate_input(reviewed_sha, tested_sha, workflow_sha, tree_oid)
    candidate = _verify_identity(resolved_repo, expected_identity, identity_verifier)

    try:
        g1_contracts.validate_contracts(resolved_repo)
        g2_contracts.validate_contracts(resolved_repo)
        p0_contracts.validate_contracts(resolved_repo)
        g1_matrix = g1_contracts.validate_matrix(resolved_repo)
        g2_matrix = g2_contracts.validate_matrix(resolved_repo)
        p0_matrix = p0_contracts.validate_matrix(resolved_repo)
        suites, host, paths = load_manifests(resolved_repo)
        matrix_registry.validate_phase3_stage_a_registration(suites, host, paths)
        _validate_workflow_authorities(resolved_repo)
    except (ContractError, OSError, TypeError, ValueError) as exc:
        raise PlanError(f"existing Phase 3 Stage A contracts are invalid: {exc}") from exc

    g1_rows, g2_rows, p0_rows = _validate_target_consistency(g1_matrix, g2_matrix, p0_matrix)
    workspace = workflow_contracts.h3_workspace_expectations()
    authority = _authority_records(resolved_repo)
    artifact_root = planned_run_root / "artifacts"
    g1_aggregate_root = planned_run_root / f"rmsnorm-semantic-g1-aggregate-{canonical_run_id}-{canonical_attempt}"

    g1_plan_rows: list[dict[str, Any]] = []
    for row in g1_rows:
        layout = g1_builder.semantic_g1_build_layout(artifact_root, row)
        projected_socket = layout.socket_path_projection
        if len(str(projected_socket).encode("utf-8")) >= AF_UNIX_PATH_MAX_BYTES:
            raise PlanError("G1 projected socket path is not AF_UNIX-safe")
        g1_plan_rows.append({
            "order": row["order"],
            "row_id": row["row_id"],
            "target": row["target"],
            "cargo_target_dir": str(layout.cargo_target_dir),
            "native_hip_build_dir": str(layout.native_hip_build_dir),
            "row_output_dir": str(layout.row_output),
            "socket_root": str(layout.socket_root),
            "socket_path_projection": str(projected_socket),
            "build_environment": _g1_environment(str(row["target"]), layout),
        })

    g2_plan_rows = [
        {
            "order": row["order"],
            "row_id": row["row_id"],
            "target": row["target"],
            "builder_output_path": str(g2_contracts.builder_output_path(resolved_repo)),
            "builder_output_relative": g2_contracts.G2_BUILDER_OUTPUT_PATH,
            "build_environment": g2_contracts.g2_build_environment(str(row["target"])),
        }
        for row in g2_rows
    ]
    p0_plan_rows = [
        {
            "order": row["order"],
            "row_id": row["row_id"],
            "target": row["target"],
            "artifact_output_dir": str(planned_run_root / "artifacts" / "p0" / str(row["row_id"])),
            "builder": "ci/tools/build_rmsnorm_p0_runtime.py",
            "builder_owned_output": {
                "binary_name": p0_contracts.P0_BINARY,
                "relative_path": f"release/{p0_contracts.P0_BINARY}",
                "copied_output_name": p0_contracts.P0_BINARY,
            },
            "build_environment": p0_contracts.p0_build_environment(str(row["target"])),
        }
        for row in p0_rows
    ]

    h3_plan_rows: list[dict[str, Any]] = []
    for order, target in enumerate(("gfx1030", "gfx1201")):
        expectation = workflow_contracts.h3_rmsnorm_row_expectation(
            target, canonical_run_id, str(canonical_attempt)
        )
        container_output = expectation["container_output_dir"]
        host_relative = Path(container_output).relative_to("/tmp")
        h3_plan_rows.append({
            "order": order,
            "row_id": expectation["row_id"],
            "target": target,
            "container_output_dir": container_output,
            "host_output_dir": str(planned_run_root / host_relative),
        })

    plan: dict[str, Any] = {
        "$schema": "https://sllm-project.local/ci/schema/phase3-stage-a-evidence-plan-v1.schema.json",
        "schema_version": PLAN_ID,
        "plan_id": PLAN_ID,
        "evidence_state": EVIDENCE_STATE,
        "evidence_claim": {
            "execution_performed": False,
            "gpu_evidence": False,
            "model_evidence": False,
            "claim": "host-only plan; no GPU, model, cache, container, build, or network work was executed",
        },
        "repository": {"path": str(resolved_repo), "resolved": True},
        "candidate": candidate,
        "run": {
            "run_id": canonical_run_id,
            "run_attempt": canonical_attempt,
            "run_root": str(planned_run_root),
            "run_root_exists": False,
        },
        "authority_files": authority,
        "authority_files_sha256": sha256_json(authority),
        "target_order": ["gfx1030", "gfx1201"],
        "h3": {
            "workspace": workspace,
            "rows": h3_plan_rows,
        },
        "controller": {
            "execution_order": ["gfx1030", "gfx1201"],
            "artifact_root": str(artifact_root),
            "aggregate_output_dir": str(g1_aggregate_root),
            "artifact_row_dirs": [str(artifact_root / str(row["row_id"])) for row in g1_rows],
            "aggregate_row_dirs": [str(g1_aggregate_root / "rows" / str(row["row_id"])) for row in g1_rows],
        },
        "g1": {
            "matrix_id": g1_matrix["matrix_id"],
            "command": list(g1_matrix["command"]),
            "rows": g1_plan_rows,
        },
        "g2": {
            "matrix_id": g2_matrix["matrix_id"],
            "command": list(g2_contracts.G2_BUILD_COMMAND),
            "builder_root": str((resolved_repo / "target")),
            "rows": g2_plan_rows,
        },
        "p0": {
            "matrix_id": p0_matrix["matrix_id"],
            "command": list(p0_contracts.P0_BUILD_COMMAND),
            "rows": p0_plan_rows,
        },
    }
    _validate_plan_schema(plan, resolved_repo)
    return plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True)
    parser.add_argument("--reviewed-sha", required=True)
    parser.add_argument("--tested-sha", required=True)
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--tree-oid", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_plan(
            repo=args.repo,
            run_root=args.run_root,
            run_id=args.run_id,
            run_attempt=args.run_attempt,
            reviewed_sha=args.reviewed_sha,
            tested_sha=args.tested_sha,
            workflow_sha=args.workflow_sha,
            tree_oid=args.tree_oid,
        )
        sys.stdout.buffer.write(canonical_bytes(plan))
        sys.stdout.buffer.flush()
        return 0
    except (PlanError, ContractError, OSError, TypeError, ValueError) as exc:
        print(f"Phase 3 Stage A evidence plan: FAIL-CLOSED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
