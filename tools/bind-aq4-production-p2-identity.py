#!/usr/bin/env python3
"""Create a fully bound AQ4 P2 identity and threshold policy."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

HASH_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_TREE_FILES = 100_000
REQUIRED_HASHES = [
    "model_identity_sha256", "tokenizer_sha256", "served_model_manifest_sha256",
    "worker_binary_sha256", "package_manifest_sha256", "package_content_sha256",
    "graph_identity_sha256", "state_schema_sha256", "source_oracle_sha256",
    "path_oracle_identity_sha256", "baseline_result_sha256", "power_capture_sha256",
    "policy_sha256", "bound_case_manifest_sha256",
]
POWER_FIELDS = ["expected_power_limit_watts", "allowed_power_tolerance_watts", "maximum_temperature_c", "minimum_vram_headroom_bytes"]
CORRECTNESS_FIELDS = ["max_hidden_relative_l2", "max_hidden_max_abs", "max_logits_relative_l2", "max_logits_max_abs", "minimum_top_k_overlap"]


class BindError(ValueError):
    pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in items:
        if key in value: raise BindError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file(): raise BindError(f"{label} must be a regular non-symlink file")
    return path.resolve(strict=True)


def load_json(path: Path, label: str) -> dict[str, Any]:
    path = regular(path, label)
    if path.stat().st_size > MAX_JSON_BYTES: raise BindError(f"{label} JSON exceeds bounded size")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(BindError(f"non-finite JSON number: {item}")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BindError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict): raise BindError(f"{label} root must be an object")
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha_bytes(value: bytes) -> str: return hashlib.sha256(value).hexdigest()


def sha_file(path: Path, label: str) -> str:
    """Hash an arbitrarily large regular file without retaining its bytes."""
    path = regular(path, label)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path, label: str) -> tuple[str, int]:
    if root.is_symlink() or not root.is_dir(): raise BindError(f"{label} must be a non-symlink directory")
    root = root.resolve(strict=True)
    paths: list[Path] = []
    for item in root.rglob("*"):
        if item.is_symlink(): raise BindError(f"{label} contains a symlink: {item}")
        if item.is_file():
            paths.append(item)
            if len(paths) > MAX_TREE_FILES: raise BindError(f"{label} has too many files")
    if not paths: raise BindError(f"{label} is empty")
    digest = hashlib.sha256()
    for item in sorted(paths, key=lambda value: value.relative_to(root).as_posix()):
        relative = item.relative_to(root).as_posix()
        digest.update(relative.encode()); digest.update(b"\0"); digest.update(bytes.fromhex(sha_file(item, f"{label} file"))); digest.update(b"\n")
    return digest.hexdigest(), len(paths)


def tokenizer_hash(path: Path) -> tuple[str, int]:
    if path.is_dir(): return tree_hash(path, "tokenizer")
    return sha_file(path, "tokenizer"), 1


def validate_utc(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value): raise BindError("effective_at must be UTC second precision")
    try: dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error: raise BindError("effective_at is invalid") from error
    return value


def git_commit(value: str | None, root: Path) -> str:
    if value is None:
        try: value = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError) as error: raise BindError("Git commit is unavailable") from error
    if GIT_RE.fullmatch(value) is None: raise BindError("Git commit must be a 40-character lowercase SHA")
    return value


def case_self_hash(case: dict[str, Any]) -> str:
    value = json.loads(json.dumps(case)); value["case_sha256"] = None
    return sha_bytes(canonical(value))


def policy_self_hash(policy: dict[str, Any]) -> str:
    value = json.loads(json.dumps(policy)); value.setdefault("hash_binding", {})["policy_sha256"] = None
    return sha_bytes(canonical(value))


def validate_number_set(source: dict[str, Any], fields: list[str], label: str) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for field in fields:
        value = source.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0: raise BindError(f"{label}.{field} must be finite and non-negative")
        result[field] = value
    return result


def bind(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_json(args.manifest, "planning manifest")
    expanded = load_json(args.expanded, "expanded manifest")
    template = load_json(args.policy, "policy template")
    if manifest.get("schema_version") != "ullm.aq4_production_p2_case_manifest.v1" or manifest.get("status") != "planning_only": raise BindError("planning manifest contract differs")
    if expanded.get("schema_version") != "ullm.aq4_production_p2_expanded.v2": raise BindError("expanded manifest schema differs")
    if expanded.get("manifest_sha256") != sha_file(args.manifest, "planning manifest"): raise BindError("expanded planning-manifest binding differs")
    cases = expanded.get("cases")
    if not isinstance(cases, list) or len(cases) != expanded.get("case_count") or len({item.get("case_id") for item in cases if isinstance(item, dict)}) != len(cases): raise BindError("expanded case coverage differs")
    for case in cases:
        if not isinstance(case, dict) or case.get("case_sha256") != case_self_hash(case): raise BindError("expanded case self-hash differs")
    if expanded.get("canonical_case_sha256") != sha_bytes(canonical(cases)): raise BindError("expanded canonical case hash differs")
    if template.get("schema_version") != "ullm.aq4_production_p2_threshold_policy.v1" or template.get("status") != "unbound_template": raise BindError("only the planning policy template may be bound")
    contract = template.get("binding_contract", {})
    if contract.get("required_hash_fields") != REQUIRED_HASHES or contract.get("required_power_fields") != POWER_FIELDS or contract.get("required_correctness_thresholds") != CORRECTNESS_FIELDS: raise BindError("policy binding contract differs")
    power_capture = load_json(args.power_capture, "power capture")
    power_source = power_capture.get("policy_binding", power_capture)
    power_values = validate_number_set(power_source, POWER_FIELDS, "power")
    if power_values["expected_power_limit_watts"] <= 0 or power_values["maximum_temperature_c"] <= 0 or power_values["minimum_vram_headroom_bytes"] <= 0: raise BindError("power limit, temperature, and headroom must be positive")
    correctness_source = load_json(args.correctness_thresholds, "correctness thresholds")
    correctness_values = validate_number_set(correctness_source, CORRECTNESS_FIELDS, "correctness")
    if not isinstance(correctness_values["minimum_top_k_overlap"], int) or correctness_values["minimum_top_k_overlap"] > template.get("correctness_thresholds", {}).get("top_k", 0): raise BindError("minimum_top_k_overlap is invalid")
    model = load_json(args.model_identity, "model identity")
    graph = load_json(args.graph, "graph identity")
    state = load_json(args.state, "state schema")
    package_sha, package_files = tree_hash(args.package_root, "package")
    tokenizer_sha, tokenizer_files = tokenizer_hash(args.tokenizer)
    expanded_sha = sha_file(args.expanded, "expanded manifest")
    hashes: dict[str, Any] = {
        "model_identity_sha256": sha_bytes(canonical(model)), "tokenizer_sha256": tokenizer_sha,
        "served_model_manifest_sha256": sha_file(args.served_model_manifest, "served model manifest"),
        "worker_binary_sha256": sha_file(args.worker, "worker binary"),
        "package_manifest_sha256": sha_file(args.package_manifest, "package manifest"),
        "package_content_sha256": package_sha, "graph_identity_sha256": sha_bytes(canonical(graph)),
        "state_schema_sha256": sha_bytes(canonical(state)), "source_oracle_sha256": sha_file(args.source_oracle, "source oracle"),
        "path_oracle_identity_sha256": sha_bytes(canonical({"contract": expanded.get("path_oracle_contract"), "canonical_case_sha256": expanded.get("canonical_case_sha256")})),
        "baseline_result_sha256": sha_file(args.baseline_result, "baseline result"),
        "power_capture_sha256": sha_file(args.power_capture, "power capture"),
        "policy_sha256": None, "bound_case_manifest_sha256": expanded_sha,
    }
    bound_policy = json.loads(json.dumps(template))
    bound_policy["status"] = "bound"; bound_policy["scope"] = "bound_execution"; bound_policy["effective_at"] = validate_utc(args.effective_at)
    bound_policy["hash_binding"].update(hashes)
    bound_policy["power_condition"].update(power_values)
    bound_policy["correctness_thresholds"].update(correctness_values)
    bound_policy["hash_binding"]["policy_sha256"] = policy_self_hash(bound_policy)
    hashes["policy_sha256"] = bound_policy["hash_binding"]["policy_sha256"]
    identity = {
        "schema_version": "ullm.aq4_production_p2_identity.v2", "status": "bound", "identity_sha256": None,
        "manifest_id": manifest.get("manifest_id"), "planning_manifest_sha256": sha_file(args.manifest, "planning manifest"),
        "expanded_manifest_sha256": expanded_sha, "canonical_case_sha256": expanded.get("canonical_case_sha256"),
        "case_count": expanded.get("case_count"), "policy_id": bound_policy.get("policy_id"), "policy_sha256": hashes["policy_sha256"],
        "build_git_commit": git_commit(args.git_commit, args.git_base), "hash_binding": hashes,
        "model_identity": model,
        "artifacts": {key: str(value.resolve(strict=True)) for key, value in {
            "worker": args.worker, "package_root": args.package_root, "package_manifest": args.package_manifest,
            "tokenizer": args.tokenizer, "served_model_manifest": args.served_model_manifest, "graph": args.graph,
            "state": args.state, "source_oracle": args.source_oracle, "power_capture": args.power_capture,
            "baseline_result": args.baseline_result, "expanded_manifest": args.expanded,
        }.items()} | {"bound_policy": str(args.bound_policy.resolve(strict=False))},
        "package_file_count": package_files, "tokenizer_file_count": tokenizer_files,
        "execution_contract": {"worker_argv": [str(args.worker.resolve(strict=True))], "r9700_lock_name": manifest.get("execution_safety", {}).get("r9700_queue_lock"), "allowed_positive_vram_processes": manifest.get("r9700_queue", {}).get("positive_vram_processes_allowed", [])},
    }
    identity["identity_sha256"] = sha_bytes(canonical({**identity, "identity_sha256": None}))
    return identity, bound_policy


def atomic_write(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink(): raise BindError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("xb") as target:
        target.write((json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode()); target.flush(); os.fsync(target.fileno())
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "expanded", "policy", "worker", "package_root", "package_manifest", "tokenizer", "served_model_manifest", "model_identity", "graph", "state", "source_oracle", "power_capture", "correctness_thresholds", "baseline_result", "output", "bound_policy"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--effective-at", required=True); parser.add_argument("--git-commit"); parser.add_argument("--git-base", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        identity, policy = bind(args); atomic_write(args.bound_policy, policy); atomic_write(args.output, identity)
        print(json.dumps({"status": "bound", "identity_sha256": identity["identity_sha256"]}, sort_keys=True)); return 0
    except (BindError, OSError, ValueError) as error:
        print(f"P2 identity binding failed: {error}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
