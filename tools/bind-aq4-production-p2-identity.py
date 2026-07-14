#!/usr/bin/env python3
"""Bind every identity required by the AQ4 P2 evidence contract.

The binder is deliberately boring: it hashes bounded, regular files and
canonical metadata, and it refuses to manufacture an identity for a missing
artifact.  It does not execute a worker or inspect a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

HASH_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_PACKAGE_FILES = 100_000
REQUIRED_HASHES = (
    "model_identity_sha256", "tokenizer_sha256", "served_model_manifest_sha256",
    "worker_binary_sha256", "package_manifest_sha256", "package_content_sha256",
    "graph_identity_sha256", "state_schema_sha256", "source_oracle_sha256",
    "path_oracle_identity_sha256", "baseline_result_sha256", "power_capture_sha256",
    "policy_sha256", "bound_case_manifest_sha256",
)


class BindError(ValueError):
    pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise BindError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> Any:
    raise BindError(f"non-finite JSON constant: {value}")


def load_json(path: Path, label: str) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise BindError(f"{label} must be a bounded regular file")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, BindError) as error:
        raise BindError(f"cannot parse {label}: {error}") from error


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise BindError(f"{label} must be a regular file: {path}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise BindError(f"{label} is too large: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise BindError(f"{label} must be lowercase SHA-256")
    return value


def atomic_write(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise BindError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    raw = (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with temporary.open("xb") as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def package_content_sha256(root: Path) -> tuple[str, list[dict[str, str]]]:
    if root.is_symlink() or not root.is_dir():
        raise BindError(f"package root must be a regular directory: {root}")
    entries: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise BindError(f"package contains symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative.startswith(".") or ".." in Path(relative).parts:
                raise BindError(f"unsafe package relative path: {relative}")
            entries.append({"path": relative, "sha256": sha_file(path, "package file")})
            if len(entries) > MAX_PACKAGE_FILES:
                raise BindError("package has too many files")
    if not entries:
        raise BindError("package is empty")
    return sha_bytes(canonical(entries)), entries


def tokenizer_identity(path: Path) -> tuple[str, list[dict[str, str]]]:
    """Hash a tokenizer file or a deterministic tokenizer directory aggregate."""
    if path.is_dir() and not path.is_symlink():
        entries: list[dict[str, str]] = []
        for item in sorted(path.rglob("*")):
            if item.is_symlink():
                raise BindError(f"tokenizer contains symlink: {item}")
            if item.is_file():
                relative = item.relative_to(path).as_posix()
                entries.append({"path": relative, "sha256": sha_file(item, "tokenizer file")})
        if not entries:
            raise BindError("tokenizer directory is empty")
        return sha_bytes(canonical(entries)), entries
    return sha_file(path, "tokenizer"), [{"path": path.name, "sha256": sha_file(path, "tokenizer")}]


def git_commit(explicit: str | None, base: Path) -> str:
    if explicit:
        if not re.fullmatch(r"[0-9a-f]{40,64}", explicit):
            raise BindError("git commit must be hexadecimal")
        return explicit
    try:
        value = subprocess.check_output(["git", "-C", str(base), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise BindError("git commit is required") from error
    if not re.fullmatch(r"[0-9a-f]{40,64}", value):
        raise BindError("git returned an invalid commit")
    return value


def bind(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    manifest = load_json(args.manifest, "case manifest")
    policy = load_json(args.policy, "threshold policy")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "ullm.aq4_production_p2_case_manifest.v1":
        raise BindError("unexpected case manifest schema")
    if not isinstance(policy, dict) or policy.get("schema_version") != "ullm.aq4_production_p2_threshold_policy.v1":
        raise BindError("unexpected policy schema")
    if policy.get("status") not in {"unbound_template", "bound"}:
        raise BindError("policy status is not bindable")
    required_policy_hashes = policy.get("binding_contract", {}).get("required_hash_fields", [])
    if set(required_policy_hashes) != set(REQUIRED_HASHES):
        raise BindError("policy required_hash_fields do not match identity contract")
    expanded = load_json(args.expanded, "expanded cases") if args.expanded else None
    if expanded is not None:
        if expanded.get("manifest_sha256") != sha_file(args.manifest, "case manifest"):
            raise BindError("expanded manifest hash does not match case manifest")
    worker_sha = sha_file(args.worker, "worker binary")
    package_manifest_sha = sha_file(args.package_manifest, "package manifest")
    package_sha, package_files = package_content_sha256(args.package_root)
    tokenizer_sha, tokenizer_files = tokenizer_identity(args.tokenizer)
    served_manifest_sha = sha_file(args.served_model_manifest, "served model manifest")
    source_sha = sha_file(args.source_oracle, "source oracle")
    power_sha = sha_file(args.power_capture, "power capture")
    baseline_sha = sha_file(args.baseline_result, "baseline result")
    graph = load_json(args.graph, "graph identity")
    state = load_json(args.state, "state schema")
    model = load_json(args.model_identity, "model identity")
    model_sha = sha_bytes(canonical(model))
    graph_sha = sha_bytes(canonical(graph))
    state_sha = sha_bytes(canonical(state))
    path_oracle = {"mode": "all_m1", "contract": manifest.get("path_oracle_contract"), "source": "same-artifact-all-m1"}
    path_oracle_sha = sha_bytes(canonical(path_oracle))
    manifest_sha = sha_file(args.manifest, "case manifest")
    # The policy digest covers the bound policy with its own digest set to
    # null.  This keeps the attestation deterministic without a circular hash.
    bound_template = json.loads(json.dumps(policy))
    bound_template["status"] = "bound"
    bound_template["scope"] = "bound_execution"
    unself_hashes = {
        "model_identity_sha256": model_sha, "tokenizer_sha256": tokenizer_sha,
        "served_model_manifest_sha256": served_manifest_sha, "worker_binary_sha256": worker_sha,
        "package_manifest_sha256": package_manifest_sha, "package_content_sha256": package_sha,
        "graph_identity_sha256": graph_sha, "state_schema_sha256": state_sha,
        "source_oracle_sha256": source_sha, "path_oracle_identity_sha256": path_oracle_sha,
        "baseline_result_sha256": baseline_sha, "power_capture_sha256": power_sha,
        "policy_sha256": None, "bound_case_manifest_sha256": manifest_sha,
    }
    bound_template.setdefault("hash_binding", {}).update(unself_hashes)
    policy_sha = sha_bytes(canonical(bound_template))
    hashes = dict(unself_hashes)
    hashes["policy_sha256"] = policy_sha
    identity = {
        "schema_version": "ullm.aq4_production_p2_identity.v1", "status": "bound",
        "manifest_id": manifest.get("manifest_id"), "manifest_sha256": manifest_sha,
        "policy_id": policy.get("policy_id"), "policy_sha256": policy_sha,
        "build_git_commit": git_commit(args.git_commit, args.git_base),
        "hash_binding": hashes,
        "model_identity": model, "path_oracle_identity": path_oracle,
        "artifacts": {"worker": str(args.worker), "package_manifest": str(args.package_manifest), "package_root": str(args.package_root), "tokenizer": str(args.tokenizer), "served_model_manifest": str(args.served_model_manifest), "graph": str(args.graph), "state": str(args.state), "source_oracle": str(args.source_oracle), "power_capture": str(args.power_capture), "baseline_result": str(args.baseline_result)},
        "package_files": package_files, "tokenizer_files": tokenizer_files,
        "binding_contract": {"required_hash_fields": list(REQUIRED_HASHES), "production_requires_real_artifacts": True, "bound_policy_required": True},
    }
    bound_policy = None
    if args.bound_policy:
        bound_policy = bound_template
        bound_policy.setdefault("hash_binding", {}).update(hashes)
        bound_policy["bound_identity_sha256"] = sha_bytes(canonical(identity))
    return identity, bound_policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--expanded", type=Path); parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True); parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True); parser.add_argument("--served-model-manifest", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True); parser.add_argument("--graph", type=Path, required=True); parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--source-oracle", type=Path, required=True); parser.add_argument("--power-capture", type=Path, required=True); parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--git-commit"); parser.add_argument("--git-base", type=Path, default=Path.cwd()); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--bound-policy", type=Path)
    args = parser.parse_args(argv)
    try:
        identity, bound_policy = bind(args)
        atomic_write(args.output, identity)
        if args.bound_policy and bound_policy is not None:
            atomic_write(args.bound_policy, bound_policy)
        print(json.dumps({"status": "bound", "identity_sha256": sha_bytes(canonical(identity))}, sort_keys=True))
        return 0
    except (BindError, OSError, ValueError) as error:
        print(f"P2 identity binding failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
