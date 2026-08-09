#!/usr/bin/env python3
"""Create only the small G2 artifact manifest; never copy/upload a binary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from common import ContractError, ROOT, read_json, sha256_file  # noqa: E402
from validate_rmsnorm_g2_contracts import (  # noqa: E402
    G2_BINARY,
    G2_SIDECAR,
    G2_SOURCE_PATH,
    G2_BUILD_COMMAND,
    G2_BUILD_PROFILE,
    G2_BUILDER_OUTPUT_PATH,
    _canonical_sidecar,
    _stable_file_bytes,
    _validate_builder_owned_output,
    _validate_embedded_build_identity,
    _source_set,
    _validate_prerequisites,
    builder_output_path,
    candidate_sha256,
    expected_build_identity,
    g2_build_environment,
    query_build_identity,
    validate_artifact,
    validate_candidate,
)


def build_g2_binary(target: str, repo: Path = ROOT) -> Path:
    """Run the sole supported G2 build and return its discovered output."""

    build_environment = g2_build_environment(target)
    # Cargo's ambient target directory is not part of the G2 contract.  Force
    # the invocation and the subsequent discovery to share the repo-local
    # target root, otherwise a successful build can leave the expected profile
    # output stale.
    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith("CARGO_FEATURE_"):
            environment.pop(name, None)
    environment.update(build_environment)
    environment["CARGO_TARGET_DIR"] = str((repo / "target").resolve())
    try:
        completed = subprocess.run(
            list(G2_BUILD_COMMAND),
            cwd=repo,
            capture_output=True,
            check=False,
            env=environment,
            timeout=900,
            start_new_session=True,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ContractError(f"fixed G2 Cargo build failed to start or timed out: {exc}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", "replace") if isinstance(completed.stderr, bytes) else str(completed.stderr)
        raise ContractError(f"fixed G2 Cargo build failed: {stderr.strip()}")
    binary = builder_output_path(repo)
    binary_bytes = _stable_file_bytes(binary, "fresh builder-owned G2 output")
    _validate_builder_owned_output(binary_bytes, repo)
    query_build_identity(binary, repo)
    sidecar = binary.with_name(G2_SIDECAR)
    sidecar.write_bytes(_canonical_sidecar(sha256_file(binary), G2_BINARY))
    return binary


def build_artifact(
    target: str,
    binary: Path,
    candidate: dict[str, object],
    output: Path,
    *,
    prerequisites: list[dict[str, object]] | None = None,
    repo: Path = ROOT,
    strict_git: bool = False,
) -> dict[str, object]:
    """Build and manifest one owned G2 output.

    ``binary`` is retained as a narrow path assertion for callers that expose
    an override (including the CLI), but it is never the source of authority.
    The owned Cargo result is built first and is the only binary that reaches
    the manifest and validation logic.  This prevents a copied canonical
    output or any other caller-controlled executable from bypassing the build
    boundary.
    """

    if target not in ("gfx1030", "gfx1201"):
        raise ContractError("G2 artifact target is not canonical")
    owned_binary = build_g2_binary(target, repo)
    if binary.resolve() != owned_binary.resolve():
        raise ContractError("G2 artifact binary override is not the fresh owned Cargo output")
    validate_candidate(candidate, repo, strict_git=strict_git)
    if owned_binary.name != G2_BINARY:
        raise ContractError("G2 artifact builder accepts only the dedicated G2 binary")
    if owned_binary.absolute().parent != output.absolute().parent:
        raise ContractError("G2 artifact manifest must be emitted beside the actual binary and sidecar")
    if prerequisites is None:
        raise ContractError("G2 artifact requires explicit nonzero prerequisite evidence hashes")
    prereqs = [dict(item) for item in prerequisites]
    _validate_prerequisites(prereqs, candidate=candidate, target=target)
    source_set = _source_set(repo)
    source_sha = source_set["files"][0]["sha256"]
    binary_bytes = _stable_file_bytes(owned_binary, "G2 dedicated binary")
    binary_sha = sha256_file(owned_binary)
    expected_identity = expected_build_identity(repo)
    embedded_identity = _validate_embedded_build_identity(binary_bytes, repo)
    if embedded_identity["embedded"] != expected_identity["identity"]:
        raise ContractError("G2 binary was not produced with the dedicated build identity")
    sidecar = owned_binary.with_name(G2_SIDECAR)
    if _stable_file_bytes(sidecar, "G2 dedicated binary sidecar") != _canonical_sidecar(binary_sha, G2_BINARY):
        raise ContractError("G2 artifact builder requires the existing canonical binary sidecar")
    document = {"schema_version": "rmsnorm-g2-artifact-v1", "artifact_id": f"rmsnorm-g2-{target}-{binary_sha}", "row_id": f"rmsnorm-g2-{target}", "target": target, "artifact_kind": "rmsnorm-g2-dedicated-public-rmsnorm", "candidate": candidate, "binary": {"role": "dedicated-g2-runtime", "path": G2_BINARY, "sidecar_path": G2_SIDECAR, "size_bytes": owned_binary.stat().st_size, "sha256": binary_sha, "sidecar_sha256": sha256_file(sidecar), "source_path": G2_SOURCE_PATH, "source_sha256": source_sha, "build_source_set": source_set, "build_identity": {**expected_identity["identity"], "identity_sha256": expected_identity["identity_sha256"]}, "build_command": list(G2_BUILD_COMMAND), "build_profile": G2_BUILD_PROFILE, "build_environment": g2_build_environment(target), "builder_output_path": G2_BUILDER_OUTPUT_PATH, "g2_binary_name": G2_BINARY, "g1_substitution_rejected": True, "h3_substitution_rejected": True}, "scope": {"model_used": True, "full_model_used": False, "tokenizer_used": False, "generation_used": False, "hip_only": True, "fallback_allowed": False, "fallback_used": False, "cpu_fallback_used": False}, "backend": "hip", "dispatch_contract": {"backend": "hip", "kernel_id": 1, "kernel_symbol": "rmsnorm.baseline.wave32.v1", "device_symbol": "sllm_rmsnorm_baseline_wave32_v1", "dispatch_count": 1, "workgroup_size_x": 256, "fallback_allowed": False, "fallback_used": False}, "prerequisites": prereqs}
    return validate_artifact(document, repo, binary_path=owned_binary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--target", required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prerequisites", type=Path, required=True)
    parser.add_argument("--reviewed-sha", required=True)
    parser.add_argument("--tested-sha", required=True)
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--tree-oid", required=True)
    args = parser.parse_args()
    candidate = {"reviewed_sha": args.reviewed_sha, "tested_sha": args.tested_sha, "workflow_sha": args.workflow_sha, "git_tree_oid": args.tree_oid, "worktree_clean": True, "revision_input": "full-sha"}
    try:
        repo = args.repo.resolve()
        prerequisites = read_json(args.prerequisites)
        if not isinstance(prerequisites, list):
            raise ContractError("--prerequisites must contain a JSON array")
        document = build_artifact(args.target, args.binary, candidate, args.output, prerequisites=prerequisites, repo=repo, strict_git=True)
        args.output.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    except (ContractError, OSError, ValueError) as exc:
        print(f"G2 artifact: FAIL: {exc}", file=sys.stderr)
        return 1
    print("G2 dedicated artifact manifest: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
