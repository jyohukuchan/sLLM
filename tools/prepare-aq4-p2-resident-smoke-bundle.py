#!/usr/bin/env python3
"""Prepare and independently validate the AQ4 P2 resident one-case smoke bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v1"
SERVED_PATH = Path("/etc/ullm/served-models/active.json")
CASE_MANIFEST_PATH = ROOT / "benchmarks/workloads/aq4-production-opt-p2-case-manifest-v0.1.json"
DRIVER_BUILD_PATH = ROOT / "target/release/ullm-aq4-p2-resident-driver"
SOURCE_COMMIT = "0fd7993843d0d7f1096d89079ce06922871d9f1a"
SOURCE_TREE = "3b0956a39749c8741a7d1852b5bc8a07adbc557b"
DRIVER_SOURCE_SHA = "297f3e22397a3120f150b3a381253b4a852119171e0aabdb35f7514dff084d3e"
EXPANDER_SOURCE_SHA = "575cf80551ca09b681bc7b0e13b46f9259c5d4504f726647277fb0b828dc710e"
FIXTURE_SOURCE_SHA = "e20285669a87285803bc6f9714b8d1ebae8188551e01a68f645ab39893e6e32c"
EXPECTED_DRIVER_SHA = "cb81b05e6e3b80426843be0c63aa6f2beeb3686016f64a03b6af5fe019caa2b4"
EXPECTED_SERVED_SHA = "feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44"
EXPECTED_WORKER_SHA = "177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d"
EXPECTED_PACKAGE_MANIFEST_SHA = "a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad"
EXPECTED_PACKAGE_CONTENT_SHA = "a24774432d3f0b7f175dc761ef9a53df1fed901dd02f825e8542b17181f004b1"
EXPECTED_CASE_MANIFEST_SHA = "1fa264c6a7a485e36b1119ca13732ad88e052a8bd502c2addacdff14ff41cbea"
EXPECTED_GUARD_SHA = "4eafd9bc149792b9c9849fed07a70830a42cf8227b85431130eec8f41708abc0"
EXPECTED_PACKAGE_FILES = 1045
PROTOCOL = "ullm.aq4_p2_resident_driver.v2"
BUNDLE_SCHEMA = "ullm.aq4_p2_resident_smoke_binding_bundle.v2"
MAX_JSON = 64 * 1024 * 1024
CHUNK = 1024 * 1024
SHA_RE = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_FILES = {
    "official-case.json": (0o444, "trusted_official_expansion_case"),
    "case-binding.json": (0o444, "runtime_bound_case"),
    "fixture.json": (0o444, "fixture"),
    "fixture-index.json": (0o444, "fixture_index"),
    "identity.json": (0o444, "resident_identity"),
    "preflight.json": (0o444, "synthetic_preflight"),
    "policy.json": (0o444, "threshold_policy"),
    "served-model.json": (0o444, "served_model_snapshot"),
    "package-manifest.json": (0o444, "package_manifest_snapshot"),
    "trust-roots.json": (0o444, "independent_trust_roots"),
    "resident-driver": (0o555, "detached_resident_driver"),
    "fake-ready.json": (0o444, "synthetic_ready_event"),
    "dry-run.json": (0o444, "resident_batch_dry_run"),
}
_VALIDATION_HOOK: Callable[[Path], None] | None = None


class BundleError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pretty(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def strict_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise BundleError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=strict_pairs, parse_constant=lambda item: (_ for _ in ()).throw(BundleError(f"non-finite JSON: {item}")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BundleError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise BundleError(f"{label} root must be an object")
    return value


def fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def reject_symlink_components(path: Path, label: str) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise BundleError(f"{label} path must be absolute without parent traversal")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise BundleError(f"{label} path has a symlink component: {current}")
        except FileNotFoundError:
            continue


class Snapshot:
    def __init__(self) -> None:
        self.entries: dict[Path, tuple[int, ...]] = {}

    def capture(self, path: Path, value: os.stat_result | None = None) -> os.stat_result:
        reject_symlink_components(path, "trust root")
        observed = path.lstat() if value is None else value
        prior = self.entries.get(path)
        if prior is not None and prior != fingerprint(observed):
            raise BundleError(f"trust root changed during reconstruction: {path}")
        self.entries[path] = fingerprint(observed)
        return observed

    def verify(self) -> None:
        for path, expected in self.entries.items():
            if fingerprint(path.lstat()) != expected:
                raise BundleError(f"trust-root TOCTOU mutation detected: {path}")


def read_stable(path: Path, label: str, maximum: int | None = MAX_JSON, *, single_link: bool = True, snapshot: Snapshot | None = None) -> bytes:
    reject_symlink_components(path, label)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or (single_link and before.st_nlink != 1):
        raise BundleError(f"{label} must be a regular{' single-link' if single_link else ''} file")
    if maximum is not None and before.st_size > maximum:
        raise BundleError(f"{label} exceeds bounded size")
    if snapshot is not None:
        snapshot.capture(path, before)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    chunks: list[bytes] = []
    try:
        if fingerprint(os.fstat(descriptor)) != fingerprint(before):
            raise BundleError(f"{label} changed before open")
        while chunk := os.read(descriptor, CHUNK):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if fingerprint(after) != fingerprint(before) or fingerprint(path.lstat()) != fingerprint(before):
        raise BundleError(f"{label} changed while read")
    return b"".join(chunks)


def sha_file(path: Path, label: str, *, single_link: bool = True, snapshot: Snapshot | None = None) -> str:
    reject_symlink_components(path, label)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or (single_link and before.st_nlink != 1):
        raise BundleError(f"{label} must be a regular{' single-link' if single_link else ''} file")
    if snapshot is not None:
        snapshot.capture(path, before)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        if fingerprint(os.fstat(descriptor)) != fingerprint(before):
            raise BundleError(f"{label} changed before hash")
        while chunk := os.read(descriptor, CHUNK):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if fingerprint(after) != fingerprint(before) or fingerprint(path.lstat()) != fingerprint(before):
        raise BundleError(f"{label} changed while hashed")
    return digest.hexdigest()


def package_tree_sha256(root: Path, snapshot: Snapshot | None = None) -> tuple[str, int]:
    reject_symlink_components(root, "package root")
    if root.is_symlink() or not root.is_dir():
        raise BundleError("package root must be a non-symlink directory")
    pending = [(root, 0)]
    files: list[Path] = []
    while pending:
        directory, depth = pending.pop()
        if depth > 32:
            raise BundleError("package depth exceeds 32")
        if snapshot is not None:
            snapshot.capture(directory)
        for entry in os.scandir(directory):
            path = Path(entry.path)
            if entry.is_symlink():
                raise BundleError(f"package symlink rejected: {path}")
            if entry.is_dir(follow_symlinks=False):
                pending.append((path, depth + 1))
            elif entry.is_file(follow_symlinks=False):
                files.append(path)
                if len(files) > 65_536:
                    raise BundleError("package file count exceeds 65536")
            else:
                raise BundleError(f"package non-regular entry rejected: {path}")
    if not files:
        raise BundleError("package tree is empty")
    aggregate = hashlib.sha256()
    for path in sorted(files, key=str):
        relative = path.relative_to(root).as_posix()
        digest = bytes.fromhex(sha_file(path, f"package/{relative}", snapshot=snapshot))
        aggregate.update(relative.encode() + b"\0" + digest + b"\n")
    return aggregate.hexdigest(), len(files)


def git_blob(path: str, expected_sha: str) -> bytes:
    completed = subprocess.run(["git", "show", f"{SOURCE_COMMIT}:{path}"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0 or sha_bytes(completed.stdout) != expected_sha:
        raise BundleError(f"trusted Git blob differs: {path}")
    return completed.stdout


def case_hash(case: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(case))
    clone["case_sha256"] = None
    return sha_bytes(canonical(clone))


def self_hash(value: dict[str, Any], field: str) -> str:
    clone = json.loads(json.dumps(value))
    clone[field] = None
    return sha_bytes(canonical(clone))


def guard_sha(names: list[str]) -> str:
    digest = hashlib.sha256(b"ullm-aq4-p2-resident-guards-v1\0")
    for name in sorted(names):
        digest.update(f"{name}=1\n".encode())
    return digest.hexdigest()


def token_ids(case: dict[str, Any], count: int, vocab_size: int, reserved: set[int]) -> list[int]:
    result: list[int] = []
    index = 0
    seed = f"{case['case_id']}\0{case['case_sha256']}".encode()
    while len(result) < count:
        digest = hashlib.sha256(seed + index.to_bytes(8, "little")).digest()
        index += 1
        candidate = int.from_bytes(digest[:8], "little") % vocab_size
        if candidate not in reserved:
            result.append(candidate)
    return result


def trusted_official_case(manifest: dict[str, Any], expander_source: bytes) -> dict[str, Any]:
    module = types.ModuleType("trusted_aq4_p2_expander")
    exec(compile(expander_source, f"{SOURCE_COMMIT}:tools/expand-aq4-production-p2.py", "exec"), module.__dict__)
    expanded = module.expand(json.loads(json.dumps(manifest)), EXPECTED_CASE_MANIFEST_SHA)
    matches = [case for case in expanded["cases"] if case.get("stage_id") == "representative" and case.get("scope") == "full_model" and case.get("phase") == "cold_prefill" and case.get("mode") == "cold_batched" and case.get("prompt_tokens") == 128 and case.get("prefill_requested_m") == 128 and case.get("device", {}).get("device_id") == "r9700-rdna4" and case.get("control_id") == "aq4_0_target"]
    if len(matches) != 1 or matches[0].get("case_sha256") != case_hash(matches[0]):
        raise BundleError("trusted official expansion case differs")
    return matches[0]


def bind_runtime_case(source: dict[str, Any]) -> dict[str, Any]:
    case = json.loads(json.dumps(source))
    case["device"] = {"device_id": "r9700-rdna4", "runtime_device_index": 1, "backend": "hip", "name": "AMD Radeon Graphics", "architecture": "gfx1201"}
    case["case_sha256"] = case_hash(case)
    return case


class Reconstruction:
    def __init__(self, payloads: dict[str, bytes], bundle: dict[str, Any], sums: bytes, snapshot: Snapshot) -> None:
        self.payloads = payloads
        self.bundle = bundle
        self.sums = sums
        self.snapshot = snapshot


def reconstruct() -> Reconstruction:
    snapshot = Snapshot()
    tree = subprocess.run(["git", "rev-parse", f"{SOURCE_COMMIT}^{{tree}}"], cwd=ROOT, text=True, capture_output=True, check=False)
    if tree.returncode != 0 or tree.stdout.strip() != SOURCE_TREE:
        raise BundleError("trusted source commit/tree differs")
    driver_source = git_blob("crates/ullm-engine/src/bin/ullm-aq4-p2-resident-driver.rs", DRIVER_SOURCE_SHA)
    if f'const PROTOCOL: &str = "{PROTOCOL}";'.encode() not in driver_source:
        raise BundleError("trusted protocol differs")
    expander_source = git_blob("tools/expand-aq4-production-p2.py", EXPANDER_SOURCE_SHA)
    git_blob("tools/generate-aq4-p2-fixtures.py", FIXTURE_SOURCE_SHA)

    served_raw = read_stable(SERVED_PATH, "active served model", snapshot=snapshot)
    if sha_bytes(served_raw) != EXPECTED_SERVED_SHA:
        raise BundleError("active served model trust-root SHA differs")
    served = parse_json(served_raw, "active served model")
    public, generation, worker, product, format_value = (served.get(key) for key in ("public", "generation", "worker", "product", "format"))
    if served.get("schema_version") != "ullm.served_model.v2" or not all(isinstance(value, dict) for value in (public, generation, worker, product, format_value)):
        raise BundleError("active served model nested schema differs")
    if public.get("id") != "ullm-qwen3.5-9b-aq4" or public.get("revision") != "aq4-reasoning-v0.1-candidate" or worker.get("protocol") != "ullm.worker.v2" or worker.get("identity") != {"device": "gfx1201", "execution_profile": "rdna4_aq4_resident"} or format_value != {"format_id": "AQ4_0", "implementation_id": "qwen35_aq4_rdna4_v1"}:
        raise BundleError("active served model semantic trust root differs")
    worker_path = Path(worker.get("binary", ""))
    if worker_path != ROOT / "target/reasoning-v2/release/ullm-aq4-worker":
        raise BundleError("active worker path differs")
    if sha_file(worker_path, "active worker", single_link=False, snapshot=snapshot) != EXPECTED_WORKER_SHA or worker.get("binary_sha256") != EXPECTED_WORKER_SHA:
        raise BundleError("active worker SHA differs")
    guards = worker.get("required_environment")
    if not isinstance(guards, list) or len(guards) != len(set(guards)) or any(not isinstance(item, str) for item in guards) or guard_sha(guards) != EXPECTED_GUARD_SHA:
        raise BundleError("active guard trust root differs")

    package_info = product.get("package")
    if not isinstance(package_info, dict) or product.get("root") != "/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1":
        raise BundleError("active product root differs")
    package_manifest_path = Path(product["root"]) / package_info.get("manifest_path", "")
    package_manifest_raw = read_stable(package_manifest_path, "package manifest", snapshot=snapshot)
    if sha_bytes(package_manifest_raw) != EXPECTED_PACKAGE_MANIFEST_SHA or package_info.get("manifest_sha256") != EXPECTED_PACKAGE_MANIFEST_SHA:
        raise BundleError("package manifest trust root differs")
    parse_json(package_manifest_raw, "package manifest")
    package_sha, package_count = package_tree_sha256(package_manifest_path.parent, snapshot)
    if package_sha != EXPECTED_PACKAGE_CONTENT_SHA or package_count != EXPECTED_PACKAGE_FILES:
        raise BundleError("package content trust root differs")

    manifest_raw = read_stable(CASE_MANIFEST_PATH, "official case manifest", snapshot=snapshot)
    if sha_bytes(manifest_raw) != EXPECTED_CASE_MANIFEST_SHA:
        raise BundleError("official case manifest trust root differs")
    source_case = trusted_official_case(parse_json(manifest_raw, "official case manifest"), expander_source)
    case = bind_runtime_case(source_case)
    runtime_device = case["device"]
    source_case_raw = pretty({"schema_version": "ullm.aq4_p2_official_case.v1", "source_commit": SOURCE_COMMIT, "manifest_sha256": EXPECTED_CASE_MANIFEST_SHA, "case": source_case})
    case_binding = {
        "schema_version": "ullm.aq4_production_p2_expanded.v2",
        "status": "bound_one_case_smoke",
        "source_manifest_sha256": EXPECTED_CASE_MANIFEST_SHA,
        "official_case_sha256": source_case["case_sha256"],
        "runtime_binding": {"schema_version": "ullm.aq4_p2_r9700_host_binding.v1", "source_device": source_case["device"], "bound_device": runtime_device, "environment": {"HIP_VISIBLE_DEVICES": "1", "ULLM_HIP_VISIBLE_DEVICES": "1"}},
        "case_count": 1,
        "canonical_case_sha256": sha_bytes(canonical([case])),
        "cases": [case],
    }
    case_binding_raw = pretty(case_binding)
    case_binding_sha = sha_bytes(case_binding_raw)
    reasoning = served.get("reasoning")
    if not isinstance(reasoning, dict):
        raise BundleError("served reasoning contract differs")
    reserved = set(generation.get("eos_token_ids", []))
    for field in ("start_token_ids", "end_token_ids", "forced_end_token_ids"):
        values = reasoning.get(field)
        if not isinstance(values, list):
            raise BundleError(f"served reasoning {field} differs")
        reserved.update(values)
    ids = token_ids(case, case["prompt_tokens"], generation["vocab_size"], reserved)
    fixture = {"schema_version": "ullm.aq4_p2_case_fixture.v1", "cases": [{"case_id": case["case_id"], "prompt_token_ids": ids, "step_count": case["generated_tokens"]}]}
    fixture_raw = pretty(fixture)
    fixture_sha = sha_bytes(fixture_raw)
    fixture_index = {
        "schema_version": "ullm.aq4_p2_fixture_index.v1",
        "expanded_manifest_sha256": case_binding_sha,
        "served_model_manifest_sha256": EXPECTED_SERVED_SHA,
        "subset": "resident_one_case_smoke",
        "case_count": 1,
        "cases": [{"case_id": case["case_id"], "case_sha256": case["case_sha256"], "fixture_path": str(CANONICAL_ROOT / "fixture.json"), "fixture_sha256": fixture_sha, "prompt_tokens": case["prompt_tokens"], "context_tokens": case["context_tokens"], "generated_tokens": case["generated_tokens"], "prompt_token_ids_sha256": sha_bytes(canonical(ids))}],
    }
    fixture_index_raw = pretty(fixture_index)
    resident_identity = {
        "binary_sha256": EXPECTED_DRIVER_SHA, "build_git_commit": SOURCE_COMMIT, "protocol": PROTOCOL,
        "worker_binary_sha256": EXPECTED_WORKER_SHA, "package_manifest_sha256": EXPECTED_PACKAGE_MANIFEST_SHA,
        "package_content_sha256": EXPECTED_PACKAGE_CONTENT_SHA, "served_model_manifest_sha256": EXPECTED_SERVED_SHA,
        "model_id": public["id"], "model_revision": public["revision"], "format_id": format_value["format_id"],
        "implementation_id": format_value["implementation_id"], "runtime_device": runtime_device, "guard_set_sha256": EXPECTED_GUARD_SHA,
    }
    identity = {
        "schema_version": "ullm.aq4_production_p2_identity.v2", "status": "bound", "identity_sha256": None,
        "expanded_manifest_sha256": case_binding_sha, "build_git_commit": SOURCE_COMMIT, "resident_driver_identity": resident_identity,
        "hash_binding": {"bound_case_manifest_sha256": case_binding_sha, "worker_binary_sha256": EXPECTED_WORKER_SHA, "package_manifest_sha256": EXPECTED_PACKAGE_MANIFEST_SHA, "package_content_sha256": EXPECTED_PACKAGE_CONTENT_SHA, "served_model_manifest_sha256": EXPECTED_SERVED_SHA},
    }
    identity["identity_sha256"] = self_hash(identity, "identity_sha256")
    identity_raw = pretty(identity)
    preflight = {field: 0 for field in ("weights_bytes", "persistent_state_bytes", "kv_cache_bytes", "workspace_bytes", "temporary_bytes", "vram_headroom_bytes")}
    preflight["gpu_process_snapshot"] = []
    policy = {"schema_version": "ullm.aq4_production_p2_threshold_policy.v1", "status": "bound"}
    fake_ready = {"event": "ready", "schema_version": PROTOCOL, "model_loads": 1, "resident_session_id": "offline-fake-ready-not-executed", "driver_identity": resident_identity}
    trust_roots = {
        "schema_version": "ullm.aq4_p2_resident_smoke_trust_roots.v1",
        "source": {"commit": SOURCE_COMMIT, "tree": SOURCE_TREE, "driver_source_sha256": DRIVER_SOURCE_SHA, "expander_source_sha256": EXPANDER_SOURCE_SHA, "fixture_generator_source_sha256": FIXTURE_SOURCE_SHA, "protocol": PROTOCOL, "clean_build": {"command": "CARGO_BUILD_JOBS=1 cargo build --release -p ullm-engine --bin ullm-aq4-p2-resident-driver", "provenance": "detached clean Git worktree at source commit", "expected_binary_sha256": EXPECTED_DRIVER_SHA}},
        "external": {"served_model": {"path": str(SERVED_PATH), "sha256": EXPECTED_SERVED_SHA}, "worker": {"path": str(worker_path), "sha256": EXPECTED_WORKER_SHA, "observed_nlink": worker_path.lstat().st_nlink}, "package_manifest": {"path": str(package_manifest_path), "sha256": EXPECTED_PACKAGE_MANIFEST_SHA}, "package_tree": {"path": str(package_manifest_path.parent), "sha256": EXPECTED_PACKAGE_CONTENT_SHA, "file_count": EXPECTED_PACKAGE_FILES}, "case_manifest": {"path": str(CASE_MANIFEST_PATH), "sha256": EXPECTED_CASE_MANIFEST_SHA}, "guard_set_sha256": EXPECTED_GUARD_SHA},
    }
    policy_raw = pretty(policy)
    dry_run = {
        "schema_version": "ullm.aq4_p2_resident_batch.v1", "status": "dry_run", "scope": "full_model", "case_count": 1,
        "warmup_runs": 2, "measured_runs": 10, "transaction_count": 12, "prompt_tokens_across_transactions": case["prompt_tokens"] * 12, "resident_model_loads": 1,
        "baseline_identity": {"run_id": "p2-r9700-resident-one-case-smoke-prepared-v2", "kind": "active-production", "identity_file": {"path": str(CANONICAL_ROOT / "identity.json"), "sha256": sha_bytes(identity_raw)}, "served_model_manifest_sha256": EXPECTED_SERVED_SHA, "worker_binary_sha256": EXPECTED_WORKER_SHA, "build_git_commit": SOURCE_COMMIT},
        "links": {"expanded": {"path": str(CANONICAL_ROOT / "case-binding.json"), "sha256": case_binding_sha}, "fixture_index": {"path": str(CANONICAL_ROOT / "fixture-index.json"), "sha256": sha_bytes(fixture_index_raw)}, "policy": {"path": str(CANONICAL_ROOT / "policy.json"), "sha256": sha_bytes(policy_raw)}},
    }
    payloads = {
        "official-case.json": source_case_raw, "case-binding.json": case_binding_raw, "fixture.json": fixture_raw,
        "fixture-index.json": fixture_index_raw, "identity.json": identity_raw, "preflight.json": pretty(preflight),
        "policy.json": policy_raw, "served-model.json": served_raw, "package-manifest.json": package_manifest_raw,
        "trust-roots.json": pretty(trust_roots), "fake-ready.json": pretty(fake_ready), "dry-run.json": pretty(dry_run),
    }
    file_bindings = {name: {"sha256": EXPECTED_DRIVER_SHA if name == "resident-driver" else sha_bytes(payloads[name]), "mode": f"{mode:04o}", "role": role} for name, (mode, role) in sorted(REQUIRED_FILES.items())}
    bundle = {
        "schema_version": BUNDLE_SCHEMA, "status": "prepared_not_executed", "promotion": False,
        "run_id": "p2-r9700-resident-one-case-smoke-prepared-v2", "canonical_root": str(CANONICAL_ROOT),
        "resident_driver": {"source_commit": SOURCE_COMMIT, "source_tree": SOURCE_TREE, "binary_sha256": EXPECTED_DRIVER_SHA, "protocol": PROTOCOL},
        "expected_runtime": {"device": runtime_device, "environment": {"HIP_VISIBLE_DEVICES": "1", "ULLM_HIP_VISIBLE_DEVICES": "1"}, "required_guards": {name: "1" for name in sorted(guards)}},
        "bindings": {"official_case_sha256": source_case["case_sha256"], "case_sha256": case["case_sha256"], "case_binding_sha256": case_binding_sha, "fixture_sha256": fixture_sha, "identity_file_sha256": sha_bytes(identity_raw), "identity_self_sha256": identity["identity_sha256"], "preflight_sha256": sha_bytes(payloads["preflight.json"]), "policy_sha256": sha_bytes(policy_raw), "served_model_manifest_sha256": EXPECTED_SERVED_SHA, "worker_binary_sha256": EXPECTED_WORKER_SHA, "package_manifest_sha256": EXPECTED_PACKAGE_MANIFEST_SHA, "package_content_sha256": EXPECTED_PACKAGE_CONTENT_SHA, "guard_set_sha256": EXPECTED_GUARD_SHA},
        "offline_evidence": {"trust_root_reconstruction": "passed", "schema_hash_path_link_toctou_validation": "passed", "runner_dry_run": "passed", "synthetic_fake_ready_validation": "passed", "model_load_executed": False, "gpu_command_executed": False, "service_touched": False},
        "actual_live_observations": {"runtime_identity": None, "power": None, "vram": None, "reason": "not acquired; preparation intentionally performed no GPU model load or live service operation"},
        "files": file_bindings,
    }
    bundle_raw = pretty(bundle)
    sums_map = {name: entry["sha256"] for name, entry in file_bindings.items()}
    sums_map["bundle.json"] = sha_bytes(bundle_raw)
    sums = "".join(f"{digest}  {name}\n" for name, digest in sorted(sums_map.items())).encode()
    return Reconstruction(payloads, bundle, sums, snapshot)


def safe_member(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute() or Path(name).parts != (name,) or name in {".", ".."}:
        raise BundleError(f"unsafe bundle member path: {name!r}")
    return root / name


_safe_member = safe_member


def validate(root: Path, trusted: Reconstruction | None = None) -> dict[str, Any]:
    root = root.absolute()
    reject_symlink_components(root, "bundle root")
    if root.is_symlink() or not root.is_dir():
        raise BundleError("bundle root must be a non-symlink directory")
    expected = reconstruct() if trusted is None else trusted
    allowed = set(REQUIRED_FILES) | {"bundle.json", "SHA256SUMS"}
    if {entry.name for entry in root.iterdir()} != allowed:
        raise BundleError("bundle directory exact coverage differs")
    initial: dict[str, tuple[int, ...]] = {}
    for name in sorted(allowed):
        path = safe_member(root, name)
        observed = path.lstat()
        mode = REQUIRED_FILES.get(name, (0o444, ""))[0]
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1 or stat.S_IMODE(observed.st_mode) != mode:
            raise BundleError(f"bundle member type/link/mode differs: {name}")
        initial[name] = fingerprint(observed)
    for name, raw in expected.payloads.items():
        actual = read_stable(root / name, name, single_link=True)
        parse_json(actual, name)
        if actual != raw:
            raise BundleError(f"independent semantic reconstruction differs: {name}")
    if sha_file(root / "resident-driver", "resident driver") != EXPECTED_DRIVER_SHA:
        raise BundleError("detached resident driver expected SHA differs")
    bundle_raw = read_stable(root / "bundle.json", "bundle", single_link=True)
    bundle = parse_json(bundle_raw, "bundle")
    if bundle != expected.bundle or bundle_raw != pretty(expected.bundle):
        raise BundleError("independent semantic reconstruction differs: bundle.json")
    sums = read_stable(root / "SHA256SUMS", "SHA256SUMS", single_link=True)
    if sums != expected.sums:
        raise BundleError("SHA256SUMS independent exact coverage differs")
    for line in sums.decode("ascii").splitlines():
        digest, name = line.split("  ", 1)
        if SHA_RE.fullmatch(digest) is None or sha_file(safe_member(root, name), name) != digest:
            raise BundleError(f"SHA256SUMS differs: {name}")
    if _VALIDATION_HOOK is not None:
        _VALIDATION_HOOK(root)
    for name, before in initial.items():
        if fingerprint((root / name).lstat()) != before:
            raise BundleError(f"TOCTOU mutation detected: {name}")
    expected.snapshot.verify()
    return bundle


def prepare(output: Path, driver_path: Path) -> dict[str, Any]:
    if output.resolve() != CANONICAL_ROOT.resolve():
        raise BundleError(f"output must be the canonical run root: {CANONICAL_ROOT}")
    if output.exists() or output.is_symlink():
        raise BundleError(f"output already exists: {output}")
    expected = reconstruct()
    if sha_file(driver_path.resolve(), "clean release resident driver", single_link=False) != EXPECTED_DRIVER_SHA:
        raise BundleError("release driver differs from trusted clean-build SHA")
    output.mkdir(parents=True)
    for name, raw in expected.payloads.items():
        (output / name).write_bytes(raw)
    with driver_path.open("rb") as source, (output / "resident-driver").open("xb") as target:
        shutil.copyfileobj(source, target, CHUNK)
        target.flush()
        os.fsync(target.fileno())
    for name, (mode, _) in REQUIRED_FILES.items():
        os.chmod(output / name, mode)
    (output / "bundle.json").write_bytes(pretty(expected.bundle))
    (output / "SHA256SUMS").write_bytes(expected.sums)
    os.chmod(output / "bundle.json", 0o444)
    os.chmod(output / "SHA256SUMS", 0o444)
    return validate(output, expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=CANONICAL_ROOT)
    prepare_parser.add_argument("--resident-driver", type=Path, default=DRIVER_BUILD_PATH)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--bundle", type=Path, default=CANONICAL_ROOT)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.output, args.resident_driver) if args.command == "prepare" else validate(args.bundle)
        print(json.dumps({"status": result["status"], "promotion": result["promotion"], "run_id": result["run_id"]}, sort_keys=True))
        return 0
    except (BundleError, OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as error:
        print(f"AQ4 P2 resident smoke bundle {args.command} failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
