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
BINDING_ROOT = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-binding-v4"
BINDING_VALIDATOR_EXEC = ROOT / "tools/prepare-aq4-p2-resident-smoke-bundle.py"
SERVED_PATH = Path("/etc/ullm/served-models/active.json")
CASE_MANIFEST_PATH = ROOT / "benchmarks/workloads/aq4-production-opt-p2-case-manifest-v0.1.json"
DRIVER_BUILD_PATH = ROOT / "target/release/ullm-aq4-p2-resident-driver"
SOURCE_COMMIT = "3dc4aa612b6cfd87675d0bd9fe506426f43e64f9"
SOURCE_TREE = "bd46e713c658878e66fcab6d49ef863e43a06bd8"
DRIVER_COMMIT = "319d6187b29e877536aa5dfe80c02bde0c77ed7a"
DRIVER_TREE = "0b5e31e6a4d7f3fd0a7b6ce4002b9b67f7f4347e"
DRIVER_SOURCE_SHA = "d42e283d231dc177b929bcffb0f51acb0c13900be7bd040f6e24bd51aede95b7"
RUNNER_SOURCE_SHA = "e7dae31c64b3844a09fbba7ef36bbae7834e21d5d217bad679dd50bdf314ff02"
EXPANDER_SOURCE_SHA = "575cf80551ca09b681bc7b0e13b46f9259c5d4504f726647277fb0b828dc710e"
FIXTURE_SOURCE_SHA = "e20285669a87285803bc6f9714b8d1ebae8188551e01a68f645ab39893e6e32c"
EXPECTED_DRIVER_SHA = "62f720835de60a61bad0a9aab5b80d778624d4d97ef5c8998e179418dab730f1"
EXPECTED_SERVED_SHA = "feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44"
EXPECTED_WORKER_SHA = "177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d"
EXPECTED_PACKAGE_MANIFEST_SHA = "a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad"
EXPECTED_PACKAGE_CONTENT_SHA = "a24774432d3f0b7f175dc761ef9a53df1fed901dd02f825e8542b17181f004b1"
EXPECTED_CASE_MANIFEST_SHA = "1fa264c6a7a485e36b1119ca13732ad88e052a8bd502c2addacdff14ff41cbea"
EXPECTED_GUARD_SHA = "4eafd9bc149792b9c9849fed07a70830a42cf8227b85431130eec8f41708abc0"
EXPECTED_PACKAGE_FILES = 1045
PROTOCOL = "ullm.aq4_p2_resident_driver.v2"
BUNDLE_SCHEMA = "ullm.aq4_p2_resident_smoke_binding_bundle.v3"
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
    "launch-command.json": (0o444, "exact_resident_launch_command"),
    "trusted-runner.py": (0o444, "trusted_one_case_smoke_runner"),
    "SUPERSEDED-0fd7993.json": (0o444, "historical_non_executable_bundle_record"),
    "resident-driver": (0o555, "detached_resident_driver"),
    "fake-ready.json": (0o444, "synthetic_ready_event"),
    "dry-run.json": (0o444, "resident_batch_dry_run"),
    "runner-dry-run-evidence.json": (0o444, "trusted_runner_subprocess_evidence"),
}
POST_RUN_FILES = {"dry-run.json", "runner-dry-run-evidence.json"}
RUNNER_VALIDATE_OUTPUT = Path("/tmp/ullm-aq4-p2-resident-smoke-validate-only")
BINDING_RUNNER_OUTPUT = Path("/tmp/ullm-aq4-p2-resident-smoke-binding-v4-runner")
BINDING_SOURCE_COMMIT = "e9065925d7b5af0352cb8dfd454a7e106abd7172"
BINDING_SOURCE_TREE = "9f2ff38d06d5ea5724a6e84af1c00d2b8147f241"
BINDING_RUNNER_GIT_BLOB = "9c097d1a97af3e15ca695c6da08b1e2928d08df7"
BINDING_RUNNER_SHA = "3140574c4f50f9b09aeb3780e400cbf8020ecf1c4ff69da685622858128f33cc"
BINDING_DRIVER_GIT_BLOB = "0bed05e56a07807fa1338a80dfba2f72de64d5af"
BINDING_FILES = {
    "trusted-runner.py": (0o444, "ed67910_generic_runner_source"),
    "trusted-validator.py": (0o444, "ed67910_bundle_validator_source"),
    "runner-plan.json": (0o444, "actual_generic_runner_dry_run_plan"),
    "runner-subprocess-evidence.json": (0o444, "actual_runner_subprocess_evidence"),
    "validator-report.json": (0o444, "trusted_validator_report"),
    "binding-manifest.json": (0o444, "immutable_binding_manifest"),
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


def git_blob(path: str, expected_sha: str, commit: str = SOURCE_COMMIT) -> bytes:
    completed = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
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


def runner_validate_argv() -> list[str]:
    return [
        str(Path(sys.executable).resolve()), str(CANONICAL_ROOT / "trusted-runner.py"),
        "--expanded", str(CANONICAL_ROOT / "case-binding.json"),
        "--fixture-index", str(CANONICAL_ROOT / "fixture-index.json"),
        "--identity", str(CANONICAL_ROOT / "identity.json"),
        "--preflight", str(CANONICAL_ROOT / "preflight.json"),
        "--policy", str(CANONICAL_ROOT / "policy.json"),
        "--output-dir", str(RUNNER_VALIDATE_OUTPUT),
        "--run-id", "p2-r9700-resident-one-case-smoke-runner-validate-v3",
        "--baseline-kind", "active-production", "--one-case-smoke", "--dry-run",
    ]


def resident_driver_argv() -> list[str]:
    return [str(CANONICAL_ROOT / "resident-driver"), "--served-model-manifest", str(SERVED_PATH), "--device-index", "1", "--build-git-commit", DRIVER_COMMIT]


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
    driver_tree = subprocess.run(["git", "rev-parse", f"{DRIVER_COMMIT}^{{tree}}"], cwd=ROOT, text=True, capture_output=True, check=False)
    if driver_tree.returncode != 0 or driver_tree.stdout.strip() != DRIVER_TREE:
        raise BundleError("normative driver commit/tree differs")
    driver_source = git_blob("crates/ullm-engine/src/bin/ullm-aq4-p2-resident-driver.rs", DRIVER_SOURCE_SHA)
    normative_driver_source = git_blob("crates/ullm-engine/src/bin/ullm-aq4-p2-resident-driver.rs", DRIVER_SOURCE_SHA, DRIVER_COMMIT)
    if driver_source != normative_driver_source:
        raise BundleError("current source changed the normative resident driver blob")
    if f'const PROTOCOL: &str = "{PROTOCOL}";'.encode() not in driver_source:
        raise BundleError("trusted protocol differs")
    for contract in (
        b'require_absolute_normal_path(&args.served_model_manifest, "served model manifest")?',
        b'require_absolute_normal_path(path, label)?',
        b'Component::ParentDir',
        b'metadata.nlink() != 1',
    ):
        if contract not in driver_source:
            raise BundleError("trusted absolute/single-link protocol contract differs")
    runner_source = git_blob("tools/run-aq4-p2-resident-batch.py", RUNNER_SOURCE_SHA)
    for contract in (b"def validate_driver_command", b"expected_binary_sha256", b"LOCK_EX | fcntl.LOCK_NB"):
        if contract not in runner_source:
            raise BundleError("trusted runner launch contract differs")
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
        "binary_sha256": EXPECTED_DRIVER_SHA, "build_git_commit": DRIVER_COMMIT, "protocol": PROTOCOL,
        "worker_binary_sha256": EXPECTED_WORKER_SHA, "package_manifest_sha256": EXPECTED_PACKAGE_MANIFEST_SHA,
        "package_content_sha256": EXPECTED_PACKAGE_CONTENT_SHA, "served_model_manifest_sha256": EXPECTED_SERVED_SHA,
        "model_id": public["id"], "model_revision": public["revision"], "format_id": format_value["format_id"],
        "implementation_id": format_value["implementation_id"], "runtime_device": runtime_device, "guard_set_sha256": EXPECTED_GUARD_SHA,
    }
    identity = {
        "schema_version": "ullm.aq4_production_p2_identity.v2", "status": "bound", "identity_sha256": None,
        "expanded_manifest_sha256": case_binding_sha, "build_git_commit": DRIVER_COMMIT, "resident_driver_identity": resident_identity,
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
        "source": {"commit": SOURCE_COMMIT, "tree": SOURCE_TREE, "driver_source_sha256": DRIVER_SOURCE_SHA, "runner_source_sha256": RUNNER_SOURCE_SHA, "expander_source_sha256": EXPANDER_SOURCE_SHA, "fixture_generator_source_sha256": FIXTURE_SOURCE_SHA, "protocol": PROTOCOL, "one_case_smoke_runner": {"flag": "--one-case-smoke", "normal_case_count": 84, "smoke_case_count": 1, "smoke_transactions": 12, "warmup_runs": 2, "measured_runs": 10, "promotion_eligible": False}, "normative_driver": {"commit": DRIVER_COMMIT, "tree": DRIVER_TREE, "blob_unchanged_at_current_source": True, "clean_build": {"command": "CARGO_BUILD_JOBS=1 cargo build --release -p ullm-engine --bin ullm-aq4-p2-resident-driver", "provenance": "detached clean Git worktree at normative driver commit", "expected_binary_sha256": EXPECTED_DRIVER_SHA}}, "protocol_path_contract": {"served_model_manifest": "absolute_without_parent_traversal", "case_binding": "absolute_without_parent_traversal", "identity": "absolute_without_parent_traversal", "preflight": "absolute_without_parent_traversal", "policy": "absolute_without_parent_traversal", "fixture": "absolute_without_parent_traversal"}},
        "external": {"served_model": {"path": str(SERVED_PATH), "sha256": EXPECTED_SERVED_SHA}, "worker": {"path": str(worker_path), "sha256": EXPECTED_WORKER_SHA, "observed_nlink": worker_path.lstat().st_nlink}, "package_manifest": {"path": str(package_manifest_path), "sha256": EXPECTED_PACKAGE_MANIFEST_SHA}, "package_tree": {"path": str(package_manifest_path.parent), "sha256": EXPECTED_PACKAGE_CONTENT_SHA, "file_count": EXPECTED_PACKAGE_FILES}, "case_manifest": {"path": str(CASE_MANIFEST_PATH), "sha256": EXPECTED_CASE_MANIFEST_SHA}, "guard_set_sha256": EXPECTED_GUARD_SHA},
    }
    policy_raw = pretty(policy)
    launch_command = {
        "schema_version": "ullm.aq4_p2_resident_launch_command.v1",
        "runner_validate_only_argv": runner_validate_argv(),
        "resident_driver_argv": resident_driver_argv(),
        "bindings": {"python": {"path": str(Path(sys.executable).resolve()), "sha256": sha_file(Path(sys.executable).resolve(), "Python interpreter", single_link=False)}, "runner": {"path": str(CANONICAL_ROOT / "trusted-runner.py"), "sha256": RUNNER_SOURCE_SHA, "source_commit": SOURCE_COMMIT}, "driver": {"path": str(CANONICAL_ROOT / "resident-driver"), "sha256": EXPECTED_DRIVER_SHA, "source_commit": DRIVER_COMMIT}, "served_model_manifest": {"path": str(SERVED_PATH), "sha256": EXPECTED_SERVED_SHA}, "device_index": 1, "build_git_commit": DRIVER_COMMIT, "protocol": PROTOCOL, "one_case_smoke": True},
    }
    superseded = {
        "schema_version": "ullm.aq4_p2_resident_smoke_supersession.v1",
        "source_commit": "0fd7993843d0d7f1096d89079ce06922871d9f1a",
        "resident_binary_sha256": "cb81b05e6e3b80426843be0c63aa6f2beeb3686016f64a03b6af5fe019caa2b4",
        "status": "superseded_historical_prepared",
        "promotion": False,
        "execution_eligible": False,
        "superseded_by_source_commit": DRIVER_COMMIT,
        "reason": "source predates the normative resident launch-boundary hardening",
    }
    payloads = {
        "official-case.json": source_case_raw, "case-binding.json": case_binding_raw, "fixture.json": fixture_raw,
        "fixture-index.json": fixture_index_raw, "identity.json": identity_raw, "preflight.json": pretty(preflight),
        "policy.json": policy_raw, "served-model.json": served_raw, "package-manifest.json": package_manifest_raw,
        "trust-roots.json": pretty(trust_roots), "fake-ready.json": pretty(fake_ready), "trusted-runner.py": runner_source,
        "launch-command.json": pretty(launch_command), "SUPERSEDED-0fd7993.json": pretty(superseded),
    }
    file_bindings = {name: {"sha256": EXPECTED_DRIVER_SHA if name == "resident-driver" else sha_bytes(payloads[name]), "mode": f"{mode:04o}", "role": role} for name, (mode, role) in sorted(REQUIRED_FILES.items()) if name not in POST_RUN_FILES}
    bundle = {
        "schema_version": BUNDLE_SCHEMA, "status": "prepared_not_executed", "promotion": False,
        "run_id": "p2-r9700-resident-one-case-smoke-prepared-v3", "canonical_root": str(CANONICAL_ROOT),
        "historical_predecessor": {"source_commit": superseded["source_commit"], "status": superseded["status"], "execution_eligible": False},
        "resident_driver": {"source_commit": DRIVER_COMMIT, "source_tree": DRIVER_TREE, "blob_unchanged_at_source_commit": SOURCE_COMMIT, "binary_sha256": EXPECTED_DRIVER_SHA, "protocol": PROTOCOL},
        "runner": {"source_commit": SOURCE_COMMIT, "source_tree": SOURCE_TREE, "source_sha256": RUNNER_SOURCE_SHA, "one_case_smoke": True},
        "expected_runtime": {"device": runtime_device, "environment": {"HIP_VISIBLE_DEVICES": "1", "ULLM_HIP_VISIBLE_DEVICES": "1"}, "required_guards": {name: "1" for name in sorted(guards)}},
        "bindings": {"official_case_sha256": source_case["case_sha256"], "case_sha256": case["case_sha256"], "case_binding_sha256": case_binding_sha, "fixture_sha256": fixture_sha, "identity_file_sha256": sha_bytes(identity_raw), "identity_self_sha256": identity["identity_sha256"], "preflight_sha256": sha_bytes(payloads["preflight.json"]), "policy_sha256": sha_bytes(policy_raw), "served_model_manifest_sha256": EXPECTED_SERVED_SHA, "worker_binary_sha256": EXPECTED_WORKER_SHA, "package_manifest_sha256": EXPECTED_PACKAGE_MANIFEST_SHA, "package_content_sha256": EXPECTED_PACKAGE_CONTENT_SHA, "guard_set_sha256": EXPECTED_GUARD_SHA},
        "offline_evidence": {"trust_root_reconstruction": "passed", "schema_hash_path_link_toctou_validation": "passed", "trusted_runner_subprocess_required": True, "runner_dry_run": "passed", "synthetic_fake_ready_validation": "passed", "model_load_executed": False, "gpu_command_executed": False, "service_touched": False},
        "actual_live_observations": {"runtime_identity": None, "power": None, "vram": None, "reason": "not acquired; preparation intentionally performed no GPU model load or live service operation"},
        "files": file_bindings,
    }
    bundle_raw = pretty(bundle)
    return Reconstruction(payloads, bundle, b"", snapshot)


def safe_member(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute() or Path(name).parts != (name,) or name in {".", ".."}:
        raise BundleError(f"unsafe bundle member path: {name!r}")
    return root / name


_safe_member = safe_member


def validate_launch_command(value: dict[str, Any]) -> None:
    expected_keys = {"schema_version", "runner_validate_only_argv", "resident_driver_argv", "bindings"}
    if set(value) != expected_keys or value.get("schema_version") != "ullm.aq4_p2_resident_launch_command.v1":
        raise BundleError("launch command exact schema differs")
    runner_argv = runner_validate_argv()
    driver_argv = resident_driver_argv()
    if value.get("runner_validate_only_argv") != runner_argv or value.get("resident_driver_argv") != driver_argv:
        raise BundleError("launch command argv differs")
    bindings = value.get("bindings")
    expected_bindings = {
        "python": {"path": runner_argv[0], "sha256": sha_file(Path(runner_argv[0]), "Python interpreter", single_link=False)},
        "runner": {"path": runner_argv[1], "sha256": RUNNER_SOURCE_SHA, "source_commit": SOURCE_COMMIT},
        "driver": {"path": driver_argv[0], "sha256": EXPECTED_DRIVER_SHA, "source_commit": DRIVER_COMMIT},
        "served_model_manifest": {"path": driver_argv[2], "sha256": EXPECTED_SERVED_SHA},
        "device_index": 1,
        "build_git_commit": DRIVER_COMMIT,
        "protocol": PROTOCOL,
        "one_case_smoke": True,
    }
    if bindings != expected_bindings:
        raise BundleError("launch command path/SHA bindings differ")
    for path, label in ((Path(runner_argv[0]), "launch Python"), (Path(runner_argv[1]), "launch runner"), (Path(driver_argv[0]), "launch driver"), (Path(driver_argv[2]), "launch served manifest")):
        reject_symlink_components(path, label)
    if "--one-case-smoke" not in runner_argv or "--dry-run" not in runner_argv:
        raise BundleError("launch command does not select validate-only one-case smoke")
    if sha_file(Path(runner_argv[1]), "launch runner") != RUNNER_SOURCE_SHA:
        raise BundleError("launch runner path/SHA differs")
    if sha_file(Path(driver_argv[0]), "launch driver") != EXPECTED_DRIVER_SHA:
        raise BundleError("launch driver path/SHA differs")
    if sha_file(Path(driver_argv[2]), "launch served manifest") != EXPECTED_SERVED_SHA:
        raise BundleError("launch served manifest path/SHA differs")


def validate_runner_plan(raw: bytes, bundle_raw: bytes, expected: Reconstruction) -> dict[str, Any]:
    plan = parse_json(raw, "trusted runner dry-run plan")
    exact_keys = {
        "schema_version", "status", "scope", "case_count", "warmup_runs", "measured_runs",
        "transaction_count", "prompt_tokens_across_transactions", "resident_model_loads",
        "baseline_identity", "links", "execution_mode", "smoke_only", "promotion_eligible", "validation",
    }
    if set(plan) != exact_keys:
        raise BundleError("trusted runner dry-run exact schema differs")
    facts = {
        "schema_version": "ullm.aq4_p2_resident_batch.v1", "status": "dry_run", "scope": "full_model",
        "case_count": 1, "warmup_runs": 2, "measured_runs": 10, "transaction_count": 12,
        "prompt_tokens_across_transactions": 1536, "resident_model_loads": 1,
        "execution_mode": "one_case_smoke", "smoke_only": True, "promotion_eligible": False,
    }
    if any(plan.get(key) != value for key, value in facts.items()):
        raise BundleError("trusted runner one-case smoke facts differ")
    identity_raw = expected.payloads["identity.json"]
    identity = parse_json(identity_raw, "identity")
    policy_raw = expected.payloads["policy.json"]
    baseline = {
        "run_id": "p2-r9700-resident-one-case-smoke-runner-validate-v3",
        "kind": "active-production",
        "identity_file": {"path": str(CANONICAL_ROOT / "identity.json"), "sha256": sha_bytes(identity_raw)},
        "served_model_manifest_sha256": EXPECTED_SERVED_SHA,
        "worker_binary_sha256": EXPECTED_WORKER_SHA,
        "build_git_commit": DRIVER_COMMIT,
    }
    links = {
        "expanded": {"path": str(CANONICAL_ROOT / "case-binding.json"), "sha256": sha_bytes(expected.payloads["case-binding.json"])},
        "fixture_index": {"path": str(CANONICAL_ROOT / "fixture-index.json"), "sha256": sha_bytes(expected.payloads["fixture-index.json"])},
        "policy": {"path": str(CANONICAL_ROOT / "policy.json"), "sha256": sha_bytes(policy_raw)},
    }
    if plan.get("baseline_identity") != baseline or plan.get("links") != links:
        raise BundleError("trusted runner dry-run identity/links differ")
    validation = plan.get("validation")
    if not isinstance(validation, dict) or set(validation) != {"mode", "bundle", "fake_ready", "driver_fake_handshake", "resident_session_id", "driver_identity"}:
        raise BundleError("trusted runner validation exact schema differs")
    expected_validation = {
        "mode": "validate_only",
        "bundle": {"path": str(CANONICAL_ROOT / "bundle.json"), "sha256": sha_bytes(bundle_raw)},
        "fake_ready": {"path": str(CANONICAL_ROOT / "fake-ready.json"), "sha256": sha_bytes(expected.payloads["fake-ready.json"])},
        "driver_fake_handshake": "passed",
        "resident_session_id": "offline-fake-ready-not-executed",
        "driver_identity": identity["resident_driver_identity"],
    }
    if validation != expected_validation:
        raise BundleError("trusted runner validate-only/fake-ready handshake differs")
    return plan


def make_runner_evidence(plan_raw: bytes, stdout: bytes, stderr: bytes, exit_code: int) -> bytes:
    value = {
        "schema_version": "ullm.aq4_p2_resident_runner_subprocess_evidence.v1",
        "runner_source_commit": SOURCE_COMMIT,
        "runner_source_sha256": RUNNER_SOURCE_SHA,
        "runner_subprocess_count": 1,
        "command": runner_validate_argv(),
        "exit_code": exit_code,
        "stdout": {"sha256": sha_bytes(stdout), "utf8": stdout.decode("utf-8")},
        "stderr": {"sha256": sha_bytes(stderr), "utf8": stderr.decode("utf-8")},
        "plan": {"path": str(CANONICAL_ROOT / "dry-run.json"), "sha256": sha_bytes(plan_raw)},
        "facts": {"case_count": 1, "transaction_count": 12, "warmup_runs": 2, "measured_runs": 10, "smoke_only": True, "promotion_eligible": False, "validation_mode": "validate_only", "fake_handshake": "passed"},
        "normal_profile": {"case_count": 84, "separate": True},
    }
    return pretty(value)


def finalize(expected: Reconstruction, plan_raw: bytes, evidence_raw: bytes) -> Reconstruction:
    bundle_raw = pretty(expected.bundle)
    validate_runner_plan(plan_raw, bundle_raw, expected)
    evidence = parse_json(evidence_raw, "trusted runner subprocess evidence")
    expected_evidence = make_runner_evidence(plan_raw, b"", b"", 0)
    if evidence_raw != expected_evidence or evidence != parse_json(expected_evidence, "expected runner evidence"):
        raise BundleError("trusted runner subprocess evidence differs")
    payloads = dict(expected.payloads)
    payloads["dry-run.json"] = plan_raw
    payloads["runner-dry-run-evidence.json"] = evidence_raw
    lines = []
    for name in sorted([*REQUIRED_FILES, "bundle.json"]):
        raw = bundle_raw if name == "bundle.json" else (None if name == "resident-driver" else payloads[name])
        digest = EXPECTED_DRIVER_SHA if raw is None else sha_bytes(raw)
        lines.append(f"{digest}  {name}\n")
    return Reconstruction(payloads, expected.bundle, "".join(lines).encode("ascii"), expected.snapshot)


def validate(root: Path, trusted: Reconstruction | None = None) -> dict[str, Any]:
    root = root.absolute()
    reject_symlink_components(root, "bundle root")
    if root.is_symlink() or not root.is_dir():
        raise BundleError("bundle root must be a non-symlink directory")
    root_before = fingerprint(root.lstat())
    names_before = {entry.name for entry in root.iterdir()}
    expected = reconstruct() if trusted is None else trusted
    allowed = set(REQUIRED_FILES) | {"bundle.json", "SHA256SUMS"}
    if names_before != allowed:
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
        if name.endswith(".json"):
            parse_json(actual, name)
        if actual != raw:
            raise BundleError(f"independent semantic reconstruction differs: {name}")
    if sha_file(root / "resident-driver", "resident driver") != EXPECTED_DRIVER_SHA:
        raise BundleError("detached resident driver expected SHA differs")
    bundle_raw = read_stable(root / "bundle.json", "bundle", single_link=True)
    bundle = parse_json(bundle_raw, "bundle")
    if bundle != expected.bundle or bundle_raw != pretty(expected.bundle):
        raise BundleError("independent semantic reconstruction differs: bundle.json")
    plan_raw = read_stable(root / "dry-run.json", "dry-run.json", single_link=True)
    evidence_raw = read_stable(root / "runner-dry-run-evidence.json", "runner-dry-run-evidence.json", single_link=True)
    expected = finalize(expected, plan_raw, evidence_raw)
    validate_launch_command(parse_json(expected.payloads["launch-command.json"], "launch command"))
    sums = read_stable(root / "SHA256SUMS", "SHA256SUMS", single_link=True)
    if sums != expected.sums:
        raise BundleError("SHA256SUMS independent exact coverage differs")
    for line in sums.decode("ascii").splitlines():
        digest, name = line.split("  ", 1)
        if SHA_RE.fullmatch(digest) is None or sha_file(safe_member(root, name), name) != digest:
            raise BundleError(f"SHA256SUMS differs: {name}")
    if _VALIDATION_HOOK is not None:
        _VALIDATION_HOOK(root)
    if {entry.name for entry in root.iterdir()} != allowed or fingerprint(root.lstat()) != root_before:
        raise BundleError("late bundle directory mutation detected")
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
    (output / "bundle.json").write_bytes(pretty(expected.bundle))
    if RUNNER_VALIDATE_OUTPUT.exists() or RUNNER_VALIDATE_OUTPUT.is_symlink():
        raise BundleError(f"trusted runner validate-only output already exists: {RUNNER_VALIDATE_OUTPUT}")
    completed = None
    try:
        completed = subprocess.run(runner_validate_argv(), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        plan_path = RUNNER_VALIDATE_OUTPUT / "resident-batch.plan.json"
        if completed.returncode != 0:
            raise BundleError(f"trusted runner validate-only subprocess failed with exit {completed.returncode}")
        plan_raw = read_stable(plan_path, "trusted runner generated plan", single_link=True)
    finally:
        if RUNNER_VALIDATE_OUTPUT.exists() and not RUNNER_VALIDATE_OUTPUT.is_symlink():
            shutil.rmtree(RUNNER_VALIDATE_OUTPUT)
    if completed is None:
        raise BundleError("trusted runner validate-only subprocess did not run")
    (output / "dry-run.json").write_bytes(plan_raw)
    evidence_raw = make_runner_evidence(plan_raw, completed.stdout, completed.stderr, completed.returncode)
    (output / "runner-dry-run-evidence.json").write_bytes(evidence_raw)
    expected = finalize(expected, plan_raw, evidence_raw)
    (output / "SHA256SUMS").write_bytes(expected.sums)
    for name, (mode, _) in REQUIRED_FILES.items():
        os.chmod(output / name, mode)
    os.chmod(output / "bundle.json", 0o444)
    os.chmod(output / "SHA256SUMS", 0o444)
    return validate(output, expected)


def binding_sources(validator_commit: str, validator_sha: str) -> tuple[bytes, bytes, str, str]:
    tree = subprocess.run(["git", "rev-parse", f"{BINDING_SOURCE_COMMIT}^{{tree}}"], cwd=ROOT, text=True, capture_output=True, check=False)
    if tree.returncode != 0 or tree.stdout.strip() != BINDING_SOURCE_TREE:
        raise BundleError("binding source commit/tree differs")
    expected_objects = {
        "tools/run-aq4-p2-resident-batch.py": BINDING_RUNNER_GIT_BLOB,
        "crates/ullm-engine/src/bin/ullm-aq4-p2-resident-driver.rs": BINDING_DRIVER_GIT_BLOB,
    }
    for path, object_id in expected_objects.items():
        observed = subprocess.run(["git", "rev-parse", f"{BINDING_SOURCE_COMMIT}:{path}"], cwd=ROOT, text=True, capture_output=True, check=False)
        if observed.returncode != 0 or observed.stdout.strip() != object_id:
            raise BundleError(f"binding Git blob object differs: {path}")
    runner = git_blob("tools/run-aq4-p2-resident-batch.py", BINDING_RUNNER_SHA, BINDING_SOURCE_COMMIT)
    if re.fullmatch(r"[0-9a-f]{40}", validator_commit) is None or SHA_RE.fullmatch(validator_sha) is None:
        raise BundleError("binding validator commit/SHA is invalid")
    validator_tree_process = subprocess.run(["git", "rev-parse", f"{validator_commit}^{{tree}}"], cwd=ROOT, text=True, capture_output=True, check=False)
    validator_object_process = subprocess.run(["git", "rev-parse", f"{validator_commit}:tools/prepare-aq4-p2-resident-smoke-bundle.py"], cwd=ROOT, text=True, capture_output=True, check=False)
    if validator_tree_process.returncode != 0 or validator_object_process.returncode != 0:
        raise BundleError("binding validator Git commit/blob is unavailable")
    validator_tree = validator_tree_process.stdout.strip()
    validator_object = validator_object_process.stdout.strip()
    validator = git_blob("tools/prepare-aq4-p2-resident-smoke-bundle.py", validator_sha, validator_commit)
    current_driver = git_blob("crates/ullm-engine/src/bin/ullm-aq4-p2-resident-driver.rs", DRIVER_SOURCE_SHA, BINDING_SOURCE_COMMIT)
    normative_driver = git_blob("crates/ullm-engine/src/bin/ullm-aq4-p2-resident-driver.rs", DRIVER_SOURCE_SHA, DRIVER_COMMIT)
    if current_driver != normative_driver:
        raise BundleError("binding commit changed the normative resident driver blob")
    for contract in (b"ONE_CASE_ROOT_CONTRACT", b"def _run_bundle_validator", b'_require_absolute_nonsymlink_path(path, "trusted bundle validator")', b"fake_driver_subprocess_count", b"--bundle-root"):
        if contract not in runner:
            raise BundleError("binding runner generic root/validator contract differs")
    return runner, validator, validator_tree, validator_object


def binding_runner_argv(validator_sha: str, root: Path = BINDING_ROOT) -> list[str]:
    return [
        str(Path(sys.executable).resolve()), str(root / "trusted-runner.py"),
        "--expanded", str(CANONICAL_ROOT / "case-binding.json"),
        "--fixture-index", str(CANONICAL_ROOT / "fixture-index.json"),
        "--identity", str(CANONICAL_ROOT / "identity.json"),
        "--preflight", str(CANONICAL_ROOT / "preflight.json"),
        "--policy", str(CANONICAL_ROOT / "policy.json"),
        "--bundle-root", str(CANONICAL_ROOT),
        "--trusted-validator", str(BINDING_VALIDATOR_EXEC),
        "--trusted-validator-sha256", validator_sha,
        "--output-dir", str(BINDING_RUNNER_OUTPUT),
        "--run-id", "p2-r9700-resident-one-case-smoke-binding-v4-validate",
        "--baseline-kind", "active-production", "--one-case-smoke", "--dry-run",
    ]


def input_root_inventory() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root_metadata = CANONICAL_ROOT.lstat()
    allowed = set(REQUIRED_FILES) | {"bundle.json", "SHA256SUMS"}
    if {entry.name for entry in CANONICAL_ROOT.iterdir()} != allowed:
        raise BundleError("binding input root exact member coverage differs")
    members: dict[str, dict[str, Any]] = {}
    for name in sorted(allowed):
        path = CANONICAL_ROOT / name
        metadata = path.lstat()
        if name in REQUIRED_FILES:
            mode, role = REQUIRED_FILES[name]
        else:
            mode, role = 0o444, {"bundle.json": "bundle_manifest", "SHA256SUMS": "sha256_manifest"}[name]
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != mode:
            raise BundleError(f"binding input root member differs: {name}")
        members[name] = {"path": str(path), "sha256": sha_file(path, f"binding input {name}"), "role": role, "type": "regular_file", "nlink": 1, "mode": f"{mode:04o}"}
    directory = {"path": str(CANONICAL_ROOT), "device": root_metadata.st_dev, "inode": root_metadata.st_ino}
    return directory, members


def validator_report_stdout(report: dict[str, Any]) -> bytes:
    return (json.dumps(report, ensure_ascii=True, sort_keys=True) + "\n").encode()


def validate_binding_plan(raw: bytes, report: dict[str, Any], directory: dict[str, Any], members: dict[str, dict[str, Any]], validator_sha: str) -> dict[str, Any]:
    plan = parse_json(raw, "binding runner plan")
    exact_keys = {
        "schema_version", "status", "scope", "case_count", "warmup_runs", "measured_runs", "transaction_count",
        "prompt_tokens_across_transactions", "resident_model_loads", "baseline_identity", "links", "execution_mode",
        "smoke_only", "promotion_eligible", "validation",
    }
    if set(plan) != exact_keys:
        raise BundleError("binding runner plan exact schema differs")
    facts = {"schema_version": "ullm.aq4_p2_resident_batch.v1", "status": "dry_run", "scope": "full_model", "case_count": 1, "warmup_runs": 2, "measured_runs": 10, "transaction_count": 12, "prompt_tokens_across_transactions": 1536, "resident_model_loads": 1, "execution_mode": "one_case_smoke", "smoke_only": True, "promotion_eligible": False}
    if any(plan.get(key) != value for key, value in facts.items()):
        raise BundleError("binding runner plan one-case facts differ")
    identity = parse_json(read_stable(CANONICAL_ROOT / "identity.json", "binding identity"), "binding identity")
    baseline = {
        "run_id": "p2-r9700-resident-one-case-smoke-binding-v4-validate", "kind": "active-production",
        "identity_file": {"path": str(CANONICAL_ROOT / "identity.json"), "sha256": members["identity.json"]["sha256"]},
        "served_model_manifest_sha256": EXPECTED_SERVED_SHA, "worker_binary_sha256": EXPECTED_WORKER_SHA, "build_git_commit": DRIVER_COMMIT,
    }
    links = {
        "expanded": {"path": str(CANONICAL_ROOT / "case-binding.json"), "sha256": members["case-binding.json"]["sha256"]},
        "fixture_index": {"path": str(CANONICAL_ROOT / "fixture-index.json"), "sha256": members["fixture-index.json"]["sha256"]},
        "policy": {"path": str(CANONICAL_ROOT / "policy.json"), "sha256": members["policy.json"]["sha256"]},
    }
    if plan.get("baseline_identity") != baseline or plan.get("links") != links:
        raise BundleError("binding runner plan identity/links differ")
    validation = plan.get("validation")
    expected_validator = {
        "subprocess_count": 1,
        "source": {"path": str(BINDING_VALIDATOR_EXEC), "sha256": validator_sha},
        "stdout_sha256": sha_bytes(validator_report_stdout(report)),
        "report_sha256": sha_bytes(canonical(report)),
        "report": report,
    }
    expected_validation = {
        "mode": "validate_only", "root_contract": "ullm.aq4_p2_resident_smoke_bundle_root.v4",
        "bundle_root": directory, "members": members,
        "bundle": {"path": str(CANONICAL_ROOT / "bundle.json"), "sha256": members["bundle.json"]["sha256"]},
        "fake_ready": {"path": str(CANONICAL_ROOT / "fake-ready.json"), "sha256": members["fake-ready.json"]["sha256"]},
        "fake_driver_subprocess_count": 1, "driver_fake_handshake": "passed",
        "resident_session_id": "offline-fake-ready-not-executed", "driver_identity": identity["resident_driver_identity"],
        "trusted_bundle_validator": expected_validator,
    }
    if validation != expected_validation:
        raise BundleError("binding runner root/fake-ready/validator report differs")
    return plan


def binding_evidence(plan_raw: bytes, report: dict[str, Any], stdout: bytes, stderr: bytes, exit_code: int, validator_sha: str) -> bytes:
    return pretty({
        "schema_version": "ullm.aq4_p2_resident_binding_runner_evidence.v1", "runner_subprocess_count": 1,
        "command": binding_runner_argv(validator_sha), "exit_code": exit_code,
        "stdout": {"sha256": sha_bytes(stdout), "utf8": stdout.decode("utf-8")},
        "stderr": {"sha256": sha_bytes(stderr), "utf8": stderr.decode("utf-8")},
        "plan": {"path": str(BINDING_ROOT / "runner-plan.json"), "sha256": sha_bytes(plan_raw)},
        "trusted_validator": {"source_sha256": validator_sha, "subprocess_count": 1, "canonical_report_sha256": sha_bytes(canonical(report)), "report_file_sha256": sha_bytes(pretty(report))},
    })


def binding_manifest(plan_raw: bytes, evidence_raw: bytes, report_raw: bytes, directory: dict[str, Any], members: dict[str, dict[str, Any]], runner_raw: bytes, validator_raw: bytes, validator_commit: str, validator_tree: str, validator_object: str) -> dict[str, Any]:
    root_fingerprint = {"directory": directory, "members": members, "sha256": sha_bytes(canonical({"directory": directory, "members": members}))}
    return {
        "schema_version": "ullm.aq4_p2_resident_smoke_binding.v4", "status": "prepared_not_executed", "promotion": False,
        "launch_eligible": False, "requires_immutable_launcher": True,
        "predecessor": {"commit": "791a20c", "status": "SUPERSEDED", "execution_eligible": False},
        "trust_roots": {
            "source_commit": BINDING_SOURCE_COMMIT, "source_tree": BINDING_SOURCE_TREE,
            "runner": {"git_blob": BINDING_RUNNER_GIT_BLOB, "sha256": sha_bytes(runner_raw)},
            "validator": {"source_commit": validator_commit, "source_tree": validator_tree, "git_blob": validator_object, "sha256": sha_bytes(validator_raw), "archive_path": str(BINDING_ROOT / "trusted-validator.py"), "execution_path": str(BINDING_VALIDATOR_EXEC)},
            "resident_driver": {"normative_commit": DRIVER_COMMIT, "git_blob_at_binding_commit": BINDING_DRIVER_GIT_BLOB, "source_sha256": DRIVER_SOURCE_SHA, "blob_unchanged": True, "binary_sha256": EXPECTED_DRIVER_SHA},
        },
        "input_root": root_fingerprint,
        "outputs": {"runner_plan_sha256": sha_bytes(plan_raw), "runner_evidence_sha256": sha_bytes(evidence_raw), "validator_report_file_sha256": sha_bytes(report_raw), "validator_report_canonical_sha256": sha_bytes(canonical(parse_json(report_raw, "binding validator report")))},
        "execution": {"runner_subprocess_count": 1, "trusted_validator_subprocess_count": 1, "fake_driver_subprocess_count": 1, "model_load_executed": False, "gpu_command_executed": False, "service_touched": False},
        "cycle_control": {"input_root_unchanged_after_runner": True, "generated_outputs_outside_input_root": True, "input_root_dry_run_not_replaced": True, "generic_runner_schema_not_embedded_back_into_input_root": True},
        "next_stage": {"name": "L immutable launcher", "required": True, "must_pin": ["input root fingerprint", "binding manifest SHA-256", "runner SHA-256", "validator SHA-256"]},
    }


def validate_binding(validator_commit: str, validator_sha: str, root: Path = BINDING_ROOT) -> dict[str, Any]:
    root = root.absolute()
    reject_symlink_components(root, "binding root")
    if root.is_symlink() or not root.is_dir():
        raise BundleError("binding root must be a non-symlink directory")
    allowed = set(BINDING_FILES) | {"SHA256SUMS"}
    if {entry.name for entry in root.iterdir()} != allowed:
        raise BundleError("binding sidecar exact member coverage differs")
    for name in sorted(allowed):
        metadata = (root / name).lstat()
        mode = BINDING_FILES.get(name, (0o444, ""))[0]
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != mode:
            raise BundleError(f"binding sidecar member differs: {name}")
    runner_raw, validator_raw, validator_tree, validator_object = binding_sources(validator_commit, validator_sha)
    if read_stable(root / "trusted-runner.py", "binding runner") != runner_raw or read_stable(root / "trusted-validator.py", "binding validator") != validator_raw:
        raise BundleError("binding trusted source differs")
    if read_stable(BINDING_VALIDATOR_EXEC, "binding validator execution copy") != validator_raw:
        raise BundleError("binding validator execution copy differs")
    validate(CANONICAL_ROOT)
    directory, members = input_root_inventory()
    plan_raw = read_stable(root / "runner-plan.json", "binding runner plan")
    report_raw = read_stable(root / "validator-report.json", "binding validator report")
    report = parse_json(report_raw, "binding validator report")
    if report != {"status": "prepared_not_executed", "promotion": False, "run_id": "p2-r9700-resident-one-case-smoke-prepared-v3"} or report_raw != pretty(report):
        raise BundleError("binding validator report differs")
    validate_binding_plan(plan_raw, report, directory, members, validator_sha)
    evidence_raw = read_stable(root / "runner-subprocess-evidence.json", "binding runner evidence")
    if evidence_raw != binding_evidence(plan_raw, report, b"", b"", 0, validator_sha):
        raise BundleError("binding runner subprocess evidence differs")
    manifest = binding_manifest(plan_raw, evidence_raw, report_raw, directory, members, runner_raw, validator_raw, validator_commit, validator_tree, validator_object)
    manifest_raw = read_stable(root / "binding-manifest.json", "binding manifest")
    if manifest_raw != pretty(manifest) or parse_json(manifest_raw, "binding manifest") != manifest:
        raise BundleError("binding manifest differs")
    expected_sums = "".join(f"{sha_file(root / name, f'binding sum {name}')}  {name}\n" for name in sorted(BINDING_FILES)).encode("ascii")
    if read_stable(root / "SHA256SUMS", "binding SHA256SUMS") != expected_sums:
        raise BundleError("binding SHA256SUMS differs")
    return manifest


def prepare_binding(validator_commit: str, validator_sha: str, output: Path = BINDING_ROOT) -> dict[str, Any]:
    if output.resolve() != BINDING_ROOT.resolve():
        raise BundleError(f"binding output must be canonical: {BINDING_ROOT}")
    if output.exists() or output.is_symlink():
        raise BundleError(f"binding output already exists: {output}")
    validate(CANONICAL_ROOT)
    directory_before, members_before = input_root_inventory()
    runner_raw, validator_raw, validator_tree, validator_object = binding_sources(validator_commit, validator_sha)
    output.mkdir(parents=True)
    (output / "trusted-runner.py").write_bytes(runner_raw)
    (output / "trusted-validator.py").write_bytes(validator_raw)
    if read_stable(BINDING_VALIDATOR_EXEC, "binding validator execution copy") != validator_raw:
        raise BundleError(f"binding validator execution path differs: {BINDING_VALIDATOR_EXEC}")
    os.chmod(output / "trusted-runner.py", 0o444)
    os.chmod(output / "trusted-validator.py", 0o444)
    if BINDING_RUNNER_OUTPUT.exists() or BINDING_RUNNER_OUTPUT.is_symlink():
        raise BundleError(f"binding runner output already exists: {BINDING_RUNNER_OUTPUT}")
    completed = None
    try:
        completed = subprocess.run(binding_runner_argv(validator_sha, output), cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if completed.returncode != 0:
            raise BundleError(f"binding runner subprocess failed with exit {completed.returncode}: {completed.stderr.decode('utf-8', 'replace')}")
        plan_raw = read_stable(BINDING_RUNNER_OUTPUT / "resident-batch.plan.json", "actual binding runner plan")
    finally:
        if BINDING_RUNNER_OUTPUT.exists() and not BINDING_RUNNER_OUTPUT.is_symlink():
            shutil.rmtree(BINDING_RUNNER_OUTPUT)
    if completed is None:
        raise BundleError("binding runner subprocess did not run")
    plan = parse_json(plan_raw, "actual binding runner plan")
    report = plan.get("validation", {}).get("trusted_bundle_validator", {}).get("report")
    if not isinstance(report, dict):
        raise BundleError("actual binding runner omitted validator report")
    validate_binding_plan(plan_raw, report, directory_before, members_before, validator_sha)
    report_raw = pretty(report)
    evidence_raw = binding_evidence(plan_raw, report, completed.stdout, completed.stderr, completed.returncode, validator_sha)
    manifest = binding_manifest(plan_raw, evidence_raw, report_raw, directory_before, members_before, runner_raw, validator_raw, validator_commit, validator_tree, validator_object)
    generated = {"runner-plan.json": plan_raw, "runner-subprocess-evidence.json": evidence_raw, "validator-report.json": report_raw, "binding-manifest.json": pretty(manifest)}
    for name, raw in generated.items():
        (output / name).write_bytes(raw)
    sums = "".join(f"{sha_file(output / name, f'prepared binding {name}')}  {name}\n" for name in sorted(BINDING_FILES)).encode("ascii")
    (output / "SHA256SUMS").write_bytes(sums)
    for name, (mode, _) in BINDING_FILES.items():
        os.chmod(output / name, mode)
    os.chmod(output / "SHA256SUMS", 0o444)
    directory_after, members_after = input_root_inventory()
    if directory_after != directory_before or members_after != members_before:
        raise BundleError("binding input root changed during actual runner validation")
    return validate_binding(validator_commit, validator_sha, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=CANONICAL_ROOT)
    prepare_parser.add_argument("--resident-driver", type=Path, default=DRIVER_BUILD_PATH)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--bundle", type=Path, default=CANONICAL_ROOT)
    binding_prepare_parser = sub.add_parser("prepare-binding")
    binding_prepare_parser.add_argument("--output", type=Path, default=BINDING_ROOT)
    binding_prepare_parser.add_argument("--validator-source-commit", required=True)
    binding_prepare_parser.add_argument("--validator-sha256", required=True)
    binding_validate_parser = sub.add_parser("validate-binding")
    binding_validate_parser.add_argument("--binding", type=Path, default=BINDING_ROOT)
    binding_validate_parser.add_argument("--validator-source-commit", required=True)
    binding_validate_parser.add_argument("--validator-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.output, args.resident_driver)
        elif args.command == "validate":
            result = validate(args.bundle)
        elif args.command == "prepare-binding":
            result = prepare_binding(args.validator_source_commit, args.validator_sha256, args.output)
        else:
            result = validate_binding(args.validator_source_commit, args.validator_sha256, args.binding)
        summary = {"status": result["status"], "promotion": result["promotion"]}
        summary["run_id"] = result.get("run_id", "p2-r9700-resident-one-case-smoke-binding-v4")
        print(json.dumps(summary, sort_keys=True))
        return 0
    except (BundleError, OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as error:
        print(f"AQ4 P2 resident smoke bundle {args.command} failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
