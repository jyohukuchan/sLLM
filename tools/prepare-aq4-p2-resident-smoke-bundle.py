#!/usr/bin/env python3
"""Prepare and offline-validate one hash-bound AQ4 P2 resident smoke input bundle.

Preparation reads the active served-model contract and package once, but never starts a worker,
resident driver, service, or GPU operation.  Validation is deliberately offline: it trusts only
the detached, single-link files in the bundle and their exact SHA256SUMS coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import stat
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVED = Path("/etc/ullm/served-models/active.json")
DEFAULT_DRIVER = ROOT / "target/release/ullm-aq4-p2-resident-driver"
DEFAULT_CASE_MANIFEST = ROOT / "benchmarks/workloads/aq4-production-opt-p2-case-manifest-v0.1.json"
DRIVER_COMMIT = "0fd7993843d0d7f1096d89079ce06922871d9f1a"
DRIVER_PROTOCOL = "ullm.aq4_p2_resident_driver.v2"
BUNDLE_SCHEMA = "ullm.aq4_p2_resident_smoke_binding_bundle.v1"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_JSON = 64 * 1024 * 1024
CHUNK = 1024 * 1024

REQUIRED_FILES = {
    "case-binding.json": (0o444, "case_binding"),
    "fixture.json": (0o444, "fixture"),
    "fixture-index.json": (0o444, "fixture_index"),
    "identity.json": (0o444, "resident_identity"),
    "preflight.json": (0o444, "synthetic_preflight"),
    "policy.json": (0o444, "threshold_policy"),
    "served-model.json": (0o444, "served_model_snapshot"),
    "package-manifest.json": (0o444, "package_manifest_snapshot"),
    "resident-driver": (0o555, "resident_driver"),
    "fake-ready.json": (0o444, "synthetic_ready_event"),
    "dry-run.json": (0o444, "resident_batch_dry_run"),
}

# Tests replace this with a one-shot mutation between the first and final stat/hash passes.
_VALIDATION_HOOK: Callable[[Path], None] | None = None


class BundleError(ValueError):
    pass


def _module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BundleError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


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
        value = json.loads(
            raw,
            object_pairs_hook=strict_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(BundleError(f"non-finite JSON: {item}")),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BundleError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise BundleError(f"{label} root must be an object")
    return value


def _fingerprint(st: os.stat_result) -> tuple[int, ...]:
    return (st.st_dev, st.st_ino, st.st_mode, st.st_nlink, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def read_stable(path: Path, label: str, maximum: int | None = MAX_JSON, single_link: bool = False) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or (single_link and before.st_nlink != 1):
        raise BundleError(f"{label} must be a regular{' single-link' if single_link else ''} file")
    if maximum is not None and before.st_size > maximum:
        raise BundleError(f"{label} exceeds bounded size")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if _fingerprint(opened) != _fingerprint(before):
            raise BundleError(f"{label} changed before open")
        parts: list[bytes] = []
        while True:
            chunk = os.read(fd, CHUNK)
            if not chunk:
                break
            parts.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if _fingerprint(after) != _fingerprint(before):
        raise BundleError(f"{label} changed while read")
    return b"".join(parts)


def sha_file(path: Path, label: str, single_link: bool = False) -> str:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or (single_link and before.st_nlink != 1):
        raise BundleError(f"{label} must be a regular{' single-link' if single_link else ''} file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        opened = os.fstat(fd)
        if _fingerprint(opened) != _fingerprint(before):
            raise BundleError(f"{label} changed before hash")
        while chunk := os.read(fd, CHUNK):
            digest.update(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if _fingerprint(after) != _fingerprint(before):
        raise BundleError(f"{label} changed while hashed")
    return digest.hexdigest()


def package_tree_sha256(root: Path) -> tuple[str, int]:
    if root.is_symlink() or not root.is_dir():
        raise BundleError("package root must be a non-symlink directory")
    files: list[Path] = []
    pending = [(root, 0)]
    while pending:
        directory, depth = pending.pop()
        if depth > 32:
            raise BundleError("package depth exceeds 32")
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
    for path in sorted(files, key=lambda item: str(item)):
        relative = path.relative_to(root).as_posix()
        digest = bytes.fromhex(sha_file(path, f"package/{relative}"))
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(digest)
        aggregate.update(b"\n")
    return aggregate.hexdigest(), len(files)


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")


def case_hash(case: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(case))
    clone["case_sha256"] = None
    return sha_bytes(canonical(clone))


def self_hash(value: dict[str, Any], field: str) -> str:
    clone = json.loads(json.dumps(value))
    if field not in clone:
        raise BundleError(f"self hash field missing: {field}")
    clone[field] = None
    return sha_bytes(canonical(clone))


def guard_set_sha256(names: list[str]) -> str:
    digest = hashlib.sha256(b"ullm-aq4-p2-resident-guards-v1\0")
    for name in sorted(names):
        digest.update(f"{name}=1\n".encode())
    return digest.hexdigest()


def validate_fake_ready(runner: Any, event: dict[str, Any], identity: dict[str, Any], cases: list[dict[str, Any]], binary_sha256: str) -> tuple[str, dict[str, Any]]:
    """Use either resident-runner API while preserving the pre-spawn binary SHA assertion."""
    parameters = inspect.signature(runner.validate_ready).parameters
    if "expected_binary_sha256" in parameters:
        return runner.validate_ready(event, identity, cases, binary_sha256)
    session_id, ready_identity = runner.validate_ready(event, identity, cases)
    if ready_identity.get("binary_sha256") != binary_sha256:
        raise BundleError("fake-ready binary differs from detached driver")
    return session_id, ready_identity


def _copy_detached(source: Path, destination: Path, mode: int) -> str:
    if destination.exists() or destination.is_symlink():
        raise BundleError(f"refusing to overwrite {destination}")
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, CHUNK)
        dst.flush()
        os.fsync(dst.fileno())
    os.chmod(destination, mode)
    st = destination.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        raise BundleError(f"detached copy is not single-link: {destination}")
    return sha_file(destination, destination.name, single_link=True)


def _select_case(manifest: dict[str, Any], manifest_raw: bytes) -> dict[str, Any]:
    expander = _module("aq4_p2_expand_for_smoke", ROOT / "tools/expand-aq4-production-p2.py")
    expanded = expander.expand(manifest, sha_bytes(manifest_raw))
    matches = [
        case for case in expanded["cases"]
        if case.get("stage_id") == "representative"
        and case.get("scope") == "full_model"
        and case.get("phase") == "cold_prefill"
        and case.get("mode") == "cold_batched"
        and case.get("prompt_tokens") == 128
        and case.get("prefill_requested_m") == 128
        and case.get("device", {}).get("device_id") == "r9700-rdna4"
        and case.get("control_id") == "aq4_0_target"
    ]
    if len(matches) != 1:
        raise BundleError(f"expected one representative smoke case, got {len(matches)}")
    case = json.loads(json.dumps(matches[0]))
    # The active WRX80 binding exposes the R9700 at physical index 1.  The driver reports gfx1201.
    case["device"] = {
        "device_id": "r9700-rdna4",
        "backend": "hip",
        "name": "AMD Radeon Graphics",
        "architecture": "gfx1201",
        "runtime_device_index": 1,
    }
    case["case_sha256"] = case_hash(case)
    return case


def prepare(output: Path, served_path: Path, driver_path: Path, case_manifest_path: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise BundleError(f"output already exists: {output}")
    output.mkdir(parents=True)
    try:
        served_raw = read_stable(served_path, "active served manifest")
        served = parse_json(served_raw, "active served manifest")
        if served.get("schema_version") != "ullm.served_model.v2":
            raise BundleError("active served manifest is not v2")
        worker = served.get("worker")
        product = served.get("product")
        public = served.get("public")
        format_value = served.get("format")
        if not all(isinstance(value, dict) for value in (worker, product, public, format_value)):
            raise BundleError("served model binding is incomplete")
        if worker.get("protocol") != "ullm.worker.v2" or worker.get("identity") != {"device": "gfx1201", "execution_profile": "rdna4_aq4_resident"}:
            raise BundleError("served worker identity differs")
        required_environment = worker.get("required_environment")
        if not isinstance(required_environment, list) or not required_environment or any(not isinstance(item, str) for item in required_environment):
            raise BundleError("served guard set is invalid")

        worker_path = Path(worker["binary"])
        worker_sha = sha_file(worker_path, "served worker")
        if worker_sha != worker.get("binary_sha256"):
            raise BundleError("served worker binary hash differs")
        product_root = Path(product["root"])
        package_info = product.get("package")
        if not isinstance(package_info, dict):
            raise BundleError("served package binding is missing")
        package_manifest_path = product_root / package_info["manifest_path"]
        package_manifest_raw = read_stable(package_manifest_path, "package manifest")
        package_manifest_sha = sha_bytes(package_manifest_raw)
        if package_manifest_sha != package_info.get("manifest_sha256"):
            raise BundleError("package manifest hash differs")
        package_root = package_manifest_path.parent
        package_content_sha, package_file_count = package_tree_sha256(package_root)

        driver_sha = sha_file(driver_path, "release resident driver")
        manifest_raw = read_stable(case_manifest_path, "P2 case manifest")
        manifest = parse_json(manifest_raw, "P2 case manifest")
        case = _select_case(manifest, manifest_raw)
        case_binding = {
            "schema_version": "ullm.aq4_production_p2_expanded.v2",
            "status": "bound_one_case_smoke",
            "source_manifest_sha256": sha_bytes(manifest_raw),
            "case_count": 1,
            "canonical_case_sha256": sha_bytes(canonical([case])),
            "cases": [case],
        }
        write_json(output / "case-binding.json", case_binding)
        case_binding_sha = sha_file(output / "case-binding.json", "case binding")

        fixture_helper = _module("aq4_p2_fixture_for_smoke", ROOT / "tools/generate-aq4-p2-fixtures.py")
        generation = served.get("generation")
        reasoning = served.get("reasoning", {})
        if not isinstance(generation, dict) or not isinstance(reasoning, dict):
            raise BundleError("served token contract is incomplete")
        reserved = set(generation.get("eos_token_ids", []))
        for field in ("start_token_ids", "end_token_ids", "forced_end_token_ids"):
            reserved.update(reasoning.get(field, []))
        token_ids = fixture_helper.token_ids(case, case["prompt_tokens"], generation["vocab_size"], reserved)
        fixture = {"schema_version": "ullm.aq4_p2_case_fixture.v1", "cases": [{"case_id": case["case_id"], "prompt_token_ids": token_ids, "step_count": case["generated_tokens"]}]}
        write_json(output / "fixture.json", fixture)
        fixture_sha = sha_file(output / "fixture.json", "fixture")
        fixture_index = {
            "schema_version": "ullm.aq4_p2_fixture_index.v1",
            "expanded_manifest_sha256": case_binding_sha,
            "served_model_manifest_sha256": sha_bytes(served_raw),
            "subset": "resident_one_case_smoke",
            "case_count": 1,
            "cases": [{
                "case_id": case["case_id"], "case_sha256": case["case_sha256"],
                "fixture_path": "fixture.json", "fixture_sha256": fixture_sha,
                "prompt_tokens": case["prompt_tokens"], "context_tokens": case["context_tokens"],
                "generated_tokens": case["generated_tokens"], "prompt_token_ids_sha256": sha_bytes(canonical(token_ids)),
            }],
        }
        write_json(output / "fixture-index.json", fixture_index)

        served_sha = sha_bytes(served_raw)
        runtime_device = {"runtime_device_index": 1, "device_id": "r9700-rdna4", "backend": "hip", "name": "AMD Radeon Graphics", "architecture": "gfx1201"}
        resident_identity = {
            "binary_sha256": driver_sha, "build_git_commit": DRIVER_COMMIT, "protocol": DRIVER_PROTOCOL,
            "worker_binary_sha256": worker_sha, "package_manifest_sha256": package_manifest_sha,
            "package_content_sha256": package_content_sha, "served_model_manifest_sha256": served_sha,
            "model_id": public["id"], "model_revision": public["revision"],
            "format_id": format_value["format_id"], "implementation_id": format_value["implementation_id"],
            "runtime_device": runtime_device, "guard_set_sha256": guard_set_sha256(required_environment),
        }
        identity = {
            "schema_version": "ullm.aq4_production_p2_identity.v2", "status": "bound", "identity_sha256": None,
            "expanded_manifest_sha256": case_binding_sha, "build_git_commit": DRIVER_COMMIT,
            "resident_driver_identity": resident_identity,
            "hash_binding": {
                "bound_case_manifest_sha256": case_binding_sha, "worker_binary_sha256": worker_sha,
                "package_manifest_sha256": package_manifest_sha, "package_content_sha256": package_content_sha,
                "served_model_manifest_sha256": served_sha,
            },
        }
        identity["identity_sha256"] = self_hash(identity, "identity_sha256")
        write_json(output / "identity.json", identity)
        preflight = {field: 0 for field in ("weights_bytes", "persistent_state_bytes", "kv_cache_bytes", "workspace_bytes", "temporary_bytes", "vram_headroom_bytes")}
        preflight["gpu_process_snapshot"] = []
        write_json(output / "preflight.json", preflight)
        write_json(output / "policy.json", {"schema_version": "ullm.aq4_production_p2_threshold_policy.v1", "status": "bound"})
        (output / "served-model.json").write_bytes(served_raw)
        (output / "package-manifest.json").write_bytes(package_manifest_raw)
        detached_driver_sha = _copy_detached(driver_path, output / "resident-driver", 0o555)
        if detached_driver_sha != driver_sha:
            raise BundleError("detached driver hash differs")

        fake_ready = {
            "event": "ready", "schema_version": DRIVER_PROTOCOL, "model_loads": 1,
            "resident_session_id": "offline-fake-ready-not-executed", "driver_identity": resident_identity,
        }
        write_json(output / "fake-ready.json", fake_ready)
        runner = _module("aq4_p2_resident_runner_for_smoke", ROOT / "tools/run-aq4-p2-resident-batch.py")
        validate_fake_ready(runner, fake_ready, identity, [case], driver_sha)
        identity_for_plan = json.loads(json.dumps(identity))
        policy_for_plan = {"schema_version": "ullm.aq4_production_p2_threshold_policy.v1", "status": "bound"}
        identity_for_plan["_path"] = str((output / "identity.json").resolve())
        identity_for_plan["_sha256"] = sha_file(output / "identity.json", "identity")
        policy_for_plan["_path"] = str((output / "policy.json").resolve())
        policy_for_plan["_sha256"] = sha_file(output / "policy.json", "policy")
        dry_run = runner.build_plan([case], output / "case-binding.json", output / "fixture-index.json", "p2-r9700-resident-one-case-smoke-prepared-v1", "active-production", identity_for_plan, policy_for_plan)
        write_json(output / "dry-run.json", dry_run)

        for name, (mode, _) in REQUIRED_FILES.items():
            os.chmod(output / name, mode)
        file_bindings = {
            name: {"sha256": sha_file(output / name, name, single_link=True), "mode": f"{mode:04o}", "role": role}
            for name, (mode, role) in sorted(REQUIRED_FILES.items())
        }
        bundle = {
            "schema_version": BUNDLE_SCHEMA, "status": "prepared_not_executed", "promotion": False,
            "run_id": "p2-r9700-resident-one-case-smoke-prepared-v1",
            "resident_driver": {"implementation_commit": DRIVER_COMMIT, "build_git_commit": DRIVER_COMMIT, "protocol": DRIVER_PROTOCOL, "binary_sha256": driver_sha},
            "expected_runtime": {"device": runtime_device, "environment": {"HIP_VISIBLE_DEVICES": "1", "ULLM_HIP_VISIBLE_DEVICES": "1"}, "required_guards": {name: "1" for name in sorted(required_environment)}},
            "bindings": {
                "case_sha256": case["case_sha256"], "case_binding_sha256": case_binding_sha,
                "fixture_sha256": fixture_sha, "identity_sha256": sha_file(output / "identity.json", "identity"),
                "preflight_sha256": sha_file(output / "preflight.json", "preflight"), "policy_sha256": sha_file(output / "policy.json", "policy"),
                "served_model_manifest_sha256": served_sha, "worker_binary_sha256": worker_sha,
                "package_manifest_sha256": package_manifest_sha, "package_content_sha256": package_content_sha,
                "guard_set_sha256": resident_identity["guard_set_sha256"],
            },
            "source_bindings": {
                "active_served_model_manifest": {"path": str(served_path), "sha256": served_sha},
                "worker_binary": {"path": str(worker_path), "sha256": worker_sha},
                "package_manifest": {"path": str(package_manifest_path), "sha256": package_manifest_sha},
                "package_root": {"path": str(package_root), "content_sha256": package_content_sha, "file_count": package_file_count},
                "case_manifest": {"path": str(case_manifest_path), "sha256": sha_bytes(manifest_raw)},
            },
            "offline_evidence": {
                "schema_hash_path_link_toctou_validation": "passed", "runner_dry_run": "passed",
                "synthetic_fake_ready_validation": "passed", "model_load_executed": False,
                "gpu_command_executed": False, "service_touched": False,
            },
            "actual_live_observations": {
                "runtime_identity": None, "power": None, "vram": None,
                "reason": "not acquired; preparation intentionally performed no GPU model load or live service operation",
            },
            "files": file_bindings,
        }
        write_json(output / "bundle.json", bundle)
        os.chmod(output / "bundle.json", 0o444)
        checksummed = sorted([*REQUIRED_FILES, "bundle.json"])
        sums = "".join(f"{sha_file(output / name, name, single_link=True)}  {name}\n" for name in checksummed)
        (output / "SHA256SUMS").write_text(sums, encoding="ascii")
        os.chmod(output / "SHA256SUMS", 0o444)
        validate(output)
        return bundle
    except Exception:
        # A partial directory is evidence of a failed attempt and is intentionally retained.
        raise


def _safe_member(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute() or Path(name).parts != (name,) or name in {".", ".."}:
        raise BundleError(f"unsafe bundle member path: {name!r}")
    return root / name


def validate(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise BundleError("bundle root must be a non-symlink directory")
    bundle_path = root / "bundle.json"
    bundle = parse_json(read_stable(bundle_path, "bundle", single_link=True), "bundle")
    required_root = {"schema_version", "status", "promotion", "run_id", "resident_driver", "expected_runtime", "bindings", "source_bindings", "offline_evidence", "actual_live_observations", "files"}
    if set(bundle) != required_root or bundle.get("schema_version") != BUNDLE_SCHEMA or bundle.get("status") != "prepared_not_executed" or bundle.get("promotion") is not False:
        raise BundleError("bundle schema/status/promotion differs")
    files = bundle.get("files")
    if not isinstance(files, dict) or set(files) != set(REQUIRED_FILES):
        raise BundleError("bundle file coverage differs")
    allowed = set(REQUIRED_FILES) | {"bundle.json", "SHA256SUMS"}
    actual_names = {item.name for item in root.iterdir()}
    if actual_names != allowed:
        raise BundleError("bundle directory coverage differs")

    initial: dict[str, tuple[int, ...]] = {}
    for name in sorted(allowed):
        path = _safe_member(root, name)
        st = path.lstat()
        expected_mode = REQUIRED_FILES.get(name, (0o444, ""))[0]
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or stat.S_IMODE(st.st_mode) != expected_mode:
            raise BundleError(f"bundle member type/link/mode differs: {name}")
        initial[name] = _fingerprint(st)
    for name, (mode, role) in REQUIRED_FILES.items():
        entry = files.get(name)
        if not isinstance(entry, dict) or set(entry) != {"sha256", "mode", "role"} or entry.get("mode") != f"{mode:04o}" or entry.get("role") != role or not isinstance(entry.get("sha256"), str) or SHA_RE.fullmatch(entry["sha256"]) is None:
            raise BundleError(f"bundle file declaration differs: {name}")
        if sha_file(root / name, name, single_link=True) != entry["sha256"]:
            raise BundleError(f"bundle declared hash differs: {name}")

    sums_raw = read_stable(root / "SHA256SUMS", "SHA256SUMS", single_link=True).decode("ascii")
    sums: dict[str, str] = {}
    for line in sums_raw.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)", line)
        if match is None or match.group(2) in sums:
            raise BundleError("SHA256SUMS format differs")
        sums[match.group(2)] = match.group(1)
    expected_sums = set(REQUIRED_FILES) | {"bundle.json"}
    if set(sums) != expected_sums:
        raise BundleError("SHA256SUMS exact coverage differs")
    for name, expected in sums.items():
        if sha_file(root / name, name, single_link=True) != expected:
            raise BundleError(f"SHA256SUMS differs: {name}")

    case_binding = parse_json(read_stable(root / "case-binding.json", "case binding", single_link=True), "case binding")
    cases = case_binding.get("cases")
    if case_binding.get("schema_version") != "ullm.aq4_production_p2_expanded.v2" or case_binding.get("status") != "bound_one_case_smoke" or case_binding.get("case_count") != 1 or not isinstance(cases, list) or len(cases) != 1:
        raise BundleError("case binding schema/count differs")
    case = cases[0]
    if not isinstance(case, dict) or case.get("case_sha256") != case_hash(case) or case_binding.get("canonical_case_sha256") != sha_bytes(canonical(cases)):
        raise BundleError("case hash differs")
    bindings = bundle.get("bindings")
    if not isinstance(bindings, dict) or bindings.get("case_sha256") != case["case_sha256"] or bindings.get("case_binding_sha256") != sums["case-binding.json"]:
        raise BundleError("bundle case bindings differ")

    identity = parse_json(read_stable(root / "identity.json", "identity", single_link=True), "identity")
    if identity.get("schema_version") != "ullm.aq4_production_p2_identity.v2" or identity.get("status") != "bound" or identity.get("identity_sha256") != self_hash(identity, "identity_sha256"):
        raise BundleError("identity schema/self-hash differs")
    resident = identity.get("resident_driver_identity")
    expected_device = {"runtime_device_index": 1, "device_id": "r9700-rdna4", "backend": "hip", "name": "AMD Radeon Graphics", "architecture": "gfx1201"}
    expected_runtime = bundle.get("expected_runtime")
    if not isinstance(resident, dict) or resident.get("runtime_device") != expected_device or resident.get("protocol") != DRIVER_PROTOCOL or resident.get("build_git_commit") != DRIVER_COMMIT:
        raise BundleError("resident identity differs")
    if not isinstance(expected_runtime, dict) or expected_runtime.get("device") != expected_device or expected_runtime.get("environment") != {"HIP_VISIBLE_DEVICES": "1", "ULLM_HIP_VISIBLE_DEVICES": "1"}:
        raise BundleError("expected R9700 visibility binding differs")
    if case.get("device") != expected_device:
        raise BundleError("case device differs from resident identity")
    if identity.get("expanded_manifest_sha256") != sums["case-binding.json"] or identity.get("hash_binding", {}).get("bound_case_manifest_sha256") != sums["case-binding.json"]:
        raise BundleError("identity case-manifest binding differs")
    for field in ("worker_binary_sha256", "package_manifest_sha256", "package_content_sha256", "served_model_manifest_sha256", "guard_set_sha256"):
        if bindings.get(field) != resident.get(field):
            raise BundleError(f"resident/bundle {field} differs")
    if resident.get("binary_sha256") != sums["resident-driver"]:
        raise BundleError("resident binary binding differs")
    if bindings.get("served_model_manifest_sha256") != sums["served-model.json"] or bindings.get("package_manifest_sha256") != sums["package-manifest.json"]:
        raise BundleError("served/package snapshots differ")

    fixture = parse_json(read_stable(root / "fixture.json", "fixture", single_link=True), "fixture")
    index = parse_json(read_stable(root / "fixture-index.json", "fixture index", single_link=True), "fixture index")
    if fixture.get("schema_version") != "ullm.aq4_p2_case_fixture.v1" or index.get("schema_version") != "ullm.aq4_p2_fixture_index.v1" or index.get("case_count") != 1:
        raise BundleError("fixture schema/count differs")
    if fixture.get("cases", [{}])[0].get("case_id") != case.get("case_id") or len(fixture.get("cases", [{}])[0].get("prompt_token_ids", [])) != case.get("prompt_tokens"):
        raise BundleError("fixture case input differs")
    entry = index.get("cases", [{}])[0]
    if entry.get("case_sha256") != case.get("case_sha256") or entry.get("fixture_sha256") != sums["fixture.json"] or entry.get("fixture_path") != "fixture.json":
        raise BundleError("fixture index binding differs")

    preflight = parse_json(read_stable(root / "preflight.json", "preflight", single_link=True), "preflight")
    expected_preflight = {field: 0 for field in ("weights_bytes", "persistent_state_bytes", "kv_cache_bytes", "workspace_bytes", "temporary_bytes", "vram_headroom_bytes")}
    expected_preflight["gpu_process_snapshot"] = []
    policy = parse_json(read_stable(root / "policy.json", "policy", single_link=True), "policy")
    if preflight != expected_preflight or policy != {"schema_version": "ullm.aq4_production_p2_threshold_policy.v1", "status": "bound"}:
        raise BundleError("synthetic preflight/policy fixture differs")

    fake_ready = parse_json(read_stable(root / "fake-ready.json", "fake ready", single_link=True), "fake ready")
    runner = _module("aq4_p2_resident_runner_validate_smoke", ROOT / "tools/run-aq4-p2-resident-batch.py")
    validate_fake_ready(runner, fake_ready, identity, [case], sums["resident-driver"])
    dry = parse_json(read_stable(root / "dry-run.json", "dry run", single_link=True), "dry run")
    if dry.get("status") != "dry_run" or dry.get("case_count") != 1 or dry.get("transaction_count") != 12 or dry.get("resident_model_loads") != 1:
        raise BundleError("resident runner dry-run differs")
    actual = bundle.get("actual_live_observations")
    offline = bundle.get("offline_evidence")
    if not isinstance(actual, dict) or any(actual.get(field) is not None for field in ("runtime_identity", "power", "vram")) or not isinstance(offline, dict) or offline.get("model_load_executed") is not False or offline.get("gpu_command_executed") is not False:
        raise BundleError("prepared-not-executed evidence differs")

    if _VALIDATION_HOOK is not None:
        _VALIDATION_HOOK(root)
    for name, before in initial.items():
        path = root / name
        if _fingerprint(path.lstat()) != before:
            raise BundleError(f"TOCTOU mutation detected: {name}")
        if name in sums and sha_file(path, name, single_link=True) != sums[name]:
            raise BundleError(f"TOCTOU hash mutation detected: {name}")
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--served-model-manifest", type=Path, default=DEFAULT_SERVED)
    prepare_parser.add_argument("--resident-driver", type=Path, default=DEFAULT_DRIVER)
    prepare_parser.add_argument("--case-manifest", type=Path, default=DEFAULT_CASE_MANIFEST)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = prepare(args.output, args.served_model_manifest, args.resident_driver, args.case_manifest) if args.command == "prepare" else validate(args.bundle)
        print(json.dumps({"status": value["status"], "promotion": value["promotion"], "run_id": value["run_id"]}, sort_keys=True))
        return 0
    except (BundleError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"AQ4 P2 resident smoke bundle {args.command} failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
