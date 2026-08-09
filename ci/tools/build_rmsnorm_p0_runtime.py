#!/usr/bin/env python3
"""Build exactly the dedicated P0 public RMSNorm producer.

The builder never executes the binary.  It uses a fresh Cargo target
directory, copies only the canonical release binary into the artifact
directory, records the complete source/build identity, and validates the
result before returning.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from common import ContractError, ROOT, canonical_bytes, read_json, sha256_file  # noqa: E402
from validate_rmsnorm_p0_contracts import (  # noqa: E402
    P0_BINARY,
    P0_BINARY_ROLE,
    P0_BUILD_COMMAND,
    P0_SIDECAR,
    PRODUCER_STATUS,
    PUBLIC_PATH,
    DTYPE_CONTRACT,
    p0_build_environment,
    validate_artifact,
    validate_candidate,
    source_set,
)


def _regular(path: Path, label: str, *, executable: bool = False) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ContractError(f"{label} cannot be stated: {exc}") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"{label} must be a regular non-symlink file")
    if executable and metadata.st_mode & 0o111 == 0:
        raise ContractError(f"{label} must be executable")


def _candidate(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "reviewed_sha": args.reviewed_sha,
        "tested_sha": args.tested_sha,
        "workflow_sha": args.workflow_sha,
        "git_tree_oid": args.tree_oid,
        "worktree_clean": True,
        "revision_input": "full-sha",
    }


def build_artifact(
    *,
    repo: Path = ROOT,
    output_dir: Path,
    candidate: Mapping[str, Any],
    target: str,
    prerequisites: list[dict[str, Any]],
    run_build: bool = True,
) -> dict[str, Any]:
    if target not in ("gfx1030", "gfx1201"):
        raise ContractError("P0 builder target is not canonical")
    validate_candidate(candidate, repo)
    if len(prerequisites) != 5:
        raise ContractError("P0 builder requires exactly five prerequisite records")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    binary = output_dir / P0_BINARY
    sidecar = output_dir / P0_SIDECAR
    artifact_path = output_dir / "rmsnorm-p0-artifact.json"
    if any(path.exists() or path.is_symlink() for path in (binary, sidecar, artifact_path)):
        raise ContractError("P0 builder refuses to overwrite an existing artifact directory")

    build_root = Path(tempfile.mkdtemp(prefix="sllm-p0-cargo-"))
    try:
        if run_build:
            environment = os.environ.copy()
            for name in list(environment):
                if name.startswith("CARGO_FEATURE_"):
                    environment.pop(name, None)
            environment.update(p0_build_environment(target))
            environment["CARGO_TARGET_DIR"] = str(build_root)
            completed = subprocess.run(
                list(P0_BUILD_COMMAND),
                cwd=repo,
                env=environment,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).decode("utf-8", "replace")[-4096:]
                raise ContractError(f"dedicated P0 Cargo build failed: {detail}")
        built_binary = build_root / "release" / P0_BINARY
        _regular(built_binary, "fresh P0 Cargo output", executable=True)
        shutil.copyfile(built_binary, binary)
        binary.chmod(stat.S_IMODE(built_binary.stat().st_mode) | 0o111)
        _regular(binary, "P0 artifact binary", executable=True)
        binary_sha = sha256_file(binary)
        sidecar.write_bytes(f"{binary_sha}  {P0_BINARY}\n".encode("ascii"))
        _regular(sidecar, "P0 artifact sidecar")

        artifact = {
            "schema_version": "rmsnorm-p0-artifact-v1",
            "artifact_id": f"rmsnorm-p0-{target}-{binary_sha}",
            "row_id": f"rmsnorm-p0-{target}",
            "target": target,
            "candidate": dict(candidate),
            "binary": {
                "role": P0_BINARY_ROLE,
                "path": P0_BINARY,
                "sidecar_path": P0_SIDECAR,
                "size_bytes": binary.stat().st_size,
                "sha256": binary_sha,
                "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            },
            "build": {
                "builder": "ci/tools/build_rmsnorm_p0_runtime.py",
                "command": list(P0_BUILD_COMMAND),
                "profile": "release",
                "binary_name": P0_BINARY,
                "output_path": P0_BINARY,
                "fresh_output": True,
                "substitution_rejected": True,
                "environment": p0_build_environment(target),
            },
            "source_set": source_set(repo),
            "execution_contract": {
                "public_path": PUBLIC_PATH,
                "kernel_id": 1,
                "kernel_symbol": "rmsnorm.baseline.wave32.v1",
                "device_symbol": "sllm_rmsnorm_baseline_wave32_v1",
                "workgroup_size_x": 256,
                "timing_contract": "rmsnorm-p0-timing-v1",
                "dtype": dict(DTYPE_CONTRACT),
                "producer_status": PRODUCER_STATUS,
            },
            "scope": {
                "selected_backend": "hip",
                "public_rmsnorm_path": True,
                "semantic_op_used": True,
                "model_used": False,
                "hip_only": True,
                "fallback_allowed": False,
                "fallback_used": False,
                "cpu_fallback_used": False,
            },
            "prerequisites": prerequisites,
        }
        artifact_path.write_bytes(canonical_bytes(artifact))
        return validate_artifact(artifact, repo, binary_path=binary)
    finally:
        shutil.rmtree(build_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prerequisites", required=True, type=Path)
    parser.add_argument("--reviewed-sha", required=True)
    parser.add_argument("--tested-sha", required=True)
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--tree-oid", required=True)
    args = parser.parse_args()
    try:
        prerequisites = read_json(args.prerequisites)
        if not isinstance(prerequisites, list) or not all(isinstance(item, dict) for item in prerequisites):
            raise ContractError("P0 prerequisites must be a JSON array of objects")
        artifact = build_artifact(
            repo=args.repo.resolve(),
            output_dir=args.output_dir,
            candidate=_candidate(args),
            target=args.target,
            prerequisites=prerequisites,
        )
        print(f"P0 artifact: {artifact['artifact_id']}")
    except (ContractError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"P0 builder: FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
