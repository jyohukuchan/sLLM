#!/usr/bin/env python3
"""Materialize a create-new SQ8-overlay GPU promotion Gate without running it.

This builder performs only filesystem, Git, and hash validation.  It never calls
GPU tools, systemctl, or the product service.  The resulting Gate deliberately
keeps actual execution disabled until a separate audit binds an executor.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "deploy/served-models/qwen35-9b-aq4-sq8-linear-qkv-z-overlay.profile.json"
WORKER = ROOT / "target/release/ullm-aq4-worker"
GENERATOR = ROOT / "tools/generate-served-model.py"
RECEIPT_WRITER = ROOT / "tools/write-qwen35-aq4-sq8-overlay-promotion-receipt.py"
MAINTENANCE = ROOT / "tools/run-qwen35-aq4-sq8-overlay-gpu-promotion.py"
CAPTURE = ROOT / "tools/capture-aq4-resident-executor-record.py"
SCHEMA = "ullm.qwen35_aq4.sq8_overlay_gpu_promotion_gate.v1"
BUILD_SCHEMA = "ullm.qwen35_aq4.sq8_overlay_release_build.v1"
IMPLEMENTATION_ID = "qwen35_aq4_sq8_linear_qkv_z_overlay_v1"
EXECUTION_PROFILE = "rdna4_aq4_resident_sq8_linear_qkv_z_overlay"
REQUIRED_OVERLAY_ENV = (
    "ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL",
    "ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_PAIR_KERNEL",
    "ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL",
    "ULLM_DISABLE_AQ4_MATVEC_QKV_Z_GATE_BETA",
)
MAX_JSON_BYTES = 16 * 1024 * 1024
AUDIT_SCHEMA = "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUEST_ID_RE = re.compile(r"^sq8-promotion-[0-9a-f]{64}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
READY_CONTAINER = "open-webui"
READY_PATH = "/readyz"
READY_URL = "http://172.20.0.1:8000/readyz"
READY_BODY = '{"status":"ready"}'
READY_TIMEOUT_SECONDS = 5
READINESS_SCHEMA = "ullm.bridge_container_readiness.v1"


class GateError(RuntimeError):
    pass


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise GateError(f"{label} must be lowercase SHA-256")
    return value


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise GateError(f"{label} must be a bounded regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"cannot parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise GateError(f"{label} must be a JSON object")
    return value


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise GateError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def command_text(argv: list[str], *, cwd: Path = ROOT) -> str:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise GateError(f"command failed: {' '.join(argv)}")
    return completed.stdout.strip() or completed.stderr.strip()


def git_value(*args: str) -> str:
    return command_text(["git", *args])


def source_archive_sha256(commit: str) -> str:
    archive = subprocess.Popen(
        ["git", "archive", "--format=tar", commit],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert archive.stdout is not None
    digest = hashlib.sha256()
    while chunk := archive.stdout.read(1024 * 1024):
        digest.update(chunk)
    _, stderr = archive.communicate(timeout=30)
    if archive.returncode != 0:
        raise GateError(f"git archive failed: {stderr.decode(errors='replace')}")
    return digest.hexdigest()


def fixed_promotion_request_id(
    *,
    commit: str,
    tree: str,
    archive_sha256: str,
    worker_sha256: str,
    binding_sha256: str,
    content_sha256: str,
    tensor_set_sha256: str,
    package_sha256: str,
    readiness: dict[str, Any],
) -> str:
    identity = {
        "schema_version": "ullm.qwen35_aq4.sq8_overlay_promotion_request.v1",
        "source": {"commit": commit, "tree": tree, "archive_sha256": archive_sha256},
        "worker_sha256": worker_sha256,
        "overlay": {
            "binding_sha256": binding_sha256,
            "content_sha256": content_sha256,
            "tensor_set_sha256": tensor_set_sha256,
        },
        "package_sha256": package_sha256,
        "readiness": readiness,
    }
    encoded = json.dumps(
        identity, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return "sq8-promotion-" + hashlib.sha256(encoded).hexdigest()


def _docker_inspect(kind: str, identity: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["docker", "inspect", "--type", kind, identity],
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=5,
    )
    if completed.returncode != 0 or completed.stderr or len(completed.stdout) > MAX_JSON_BYTES:
        raise GateError(f"readiness {kind} inspect failed")
    try:
        values = json.loads(completed.stdout)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"readiness {kind} inspect JSON differs") from error
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise GateError(f"readiness {kind} inspect shape differs")
    return values[0]


def readiness_identity() -> dict[str, Any]:
    """Bind the one audited bridge-side readiness observation path."""

    container = _docker_inspect("container", READY_CONTAINER)
    container_id = container.get("Id")
    raw_name = container.get("Name")
    image_id = container.get("Image")
    config = container.get("Config")
    networks = container.get("NetworkSettings", {}).get("Networks")
    if (
        not isinstance(container_id, str)
        or HEX64_RE.fullmatch(container_id) is None
        or raw_name != f"/{READY_CONTAINER}"
        or not isinstance(image_id, str)
        or IMAGE_ID_RE.fullmatch(image_id) is None
        or not isinstance(config, dict)
        or not isinstance(config.get("Image"), str)
        or not config["Image"]
        or not isinstance(networks, dict)
        or len(networks) != 1
    ):
        raise GateError("readiness container identity differs")
    network_name, attachment = next(iter(networks.items()))
    if not isinstance(network_name, str) or not network_name or not isinstance(attachment, dict):
        raise GateError("readiness container network attachment differs")
    network_id = attachment.get("NetworkID")
    if not isinstance(network_id, str) or HEX64_RE.fullmatch(network_id) is None:
        raise GateError("readiness container network ID differs")
    network = _docker_inspect("network", network_id)
    bridge_interface = f"br-{network_id[:12]}"
    if (
        network.get("Id") != network_id
        or network.get("Name") != network_name
        or network.get("Driver") != "bridge"
        or not (Path("/sys/class/net") / bridge_interface).is_dir()
    ):
        raise GateError("readiness bridge network identity differs")
    expected_body_sha256 = hashlib.sha256(READY_BODY.encode("ascii")).hexdigest()
    return {
        "schema": READINESS_SCHEMA,
        "container": {
            "name": READY_CONTAINER,
            "id": container_id,
            "image_id": image_id,
            "config_image": config["Image"],
        },
        "network": {
            "name": network_name,
            "id": network_id,
            "driver": "bridge",
            "bridge_interface": bridge_interface,
        },
        "endpoint": {
            "url": READY_URL,
            "path": READY_PATH,
            "expected_status": 200,
            "expected_body": READY_BODY,
            "expected_body_sha256": expected_body_sha256,
            "timeout_seconds": READY_TIMEOUT_SECONDS,
        },
    }


def write_exclusive(path: Path, payload: bytes, mode: int = 0o444) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        os.close(descriptor)
    path.chmod(mode)


def write_json_exclusive(path: Path, value: dict[str, Any], mode: int = 0o444) -> None:
    raw = (json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("ascii")
    write_exclusive(path, raw, mode)


def copy_binary_exclusive(source: Path, destination: Path) -> dict[str, Any]:
    metadata = source.stat(follow_symlinks=False)
    if source.is_symlink() or not stat.S_ISREG(metadata.st_mode) or not os.access(source, os.X_OK):
        raise GateError("release worker must be an executable regular non-symlink file")
    source_sha = sha_file(source)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o555)
    try:
        with source.open("rb") as src, os.fdopen(descriptor, "wb", closefd=False) as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
    finally:
        os.close(descriptor)
    destination.chmod(0o555)
    copied = destination.stat(follow_symlinks=False)
    if copied.st_nlink != 1 or copied.st_size != metadata.st_size or sha_file(destination) != source_sha:
        raise GateError("immutable worker copy identity differs")
    return {
        "source_path": str(source.resolve()),
        "source_sha256": source_sha,
        "source_bytes": metadata.st_size,
        "source_nlink": metadata.st_nlink,
        "immutable_path": str(destination.resolve()),
        "immutable_sha256": source_sha,
        "immutable_bytes": copied.st_size,
        "immutable_mode": "0555",
        "immutable_nlink": copied.st_nlink,
    }


def validate_profile(profile: dict[str, Any]) -> None:
    worker = profile.get("worker")
    if profile.get("schema_version") != "ullm.served_model.profile.v1" or not isinstance(worker, dict):
        raise GateError("overlay served-model profile schema differs")
    if profile.get("format") != {"format_id": "AQ4_0", "implementation_id": IMPLEMENTATION_ID}:
        raise GateError("overlay implementation identity differs")
    identity = worker.get("identity")
    if identity != {"device": "gfx1201", "execution_profile": EXECUTION_PROFILE}:
        raise GateError("overlay worker identity differs")
    required = worker.get("required_environment")
    if not isinstance(required, list) or any(name not in required for name in REQUIRED_OVERLAY_ENV):
        raise GateError("overlay required environment is incomplete")


def validate_binding(binding: dict[str, Any], package_manifest: Path) -> None:
    exact = {
        "schema_version": "ullm.qwen35_aq4_sq8_qkv_z_overlay.v2",
        "format_id": "AQ4_0",
        "overlay_format_id": "SQ8_0",
        "implementation_id": IMPLEMENTATION_ID,
    }
    if any(binding.get(key) != value for key, value in exact.items()):
        raise GateError("overlay binding identity differs")
    names = binding.get("tensor_names")
    if not isinstance(names, list) or len(names) != 48 or len(set(names)) != 48:
        raise GateError("overlay binding tensor set is not exactly 48 unique tensors")
    if any(not isinstance(name, str) or not name.endswith(("in_proj_qkv.weight", "in_proj_z.weight")) for name in names):
        raise GateError("overlay binding contains a non-QKV/Z tensor")
    for field in ("content_sha256", "tensor_set_sha256"):
        value = binding.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise GateError(f"overlay binding {field} is invalid")
    package = binding.get("package")
    if not isinstance(package, dict) or package.get("manifest_sha256") != sha_file(package_manifest):
        raise GateError("overlay package manifest binding differs")


def _audit_reference(record: Any, expected_path: Path, label: str) -> str:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise GateError(f"independent audit {label} reference differs")
    path = Path(str(record["path"])).resolve()
    digest = require_sha256(record["sha256"], f"independent audit {label}")
    if path != expected_path.resolve() or path.is_symlink() or not path.is_file() or sha_file(path) != digest:
        raise GateError(f"independent audit {label} live identity differs")
    return digest


def validate_independent_audit(
    path: Path,
    *,
    commit: str,
    tree: str,
    archive_sha256: str,
) -> dict[str, Any]:
    if path.is_symlink():
        raise GateError("independent audit receipt must be immutable 0444 single-link non-symlink")
    path = path.resolve()
    metadata = path.stat(follow_symlinks=False)
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or metadata.st_nlink != 1
    ):
        raise GateError("independent audit receipt must be immutable 0444 single-link non-symlink")
    audit_sha = sha_file(path)
    audit = read_object(path, "independent audit receipt")
    if set(audit) != {
        "schema_version", "auditor_task_id", "audited_at_utc", "audited_source", "runtime",
        "fixed_request_id", "gate_state", "topology", "verdict", "actual", "tests",
    }:
        raise GateError("independent audit receipt shape differs")
    if audit.get("schema_version") != AUDIT_SCHEMA or audit.get("verdict") != "implementation_ready" or audit.get("actual") != "not_executed":
        raise GateError("independent audit verdict differs")
    source = audit.get("audited_source")
    if source != {"commit": commit, "tree_sha256": tree, "archive_sha256": archive_sha256}:
        raise GateError("independent audit source identity differs")
    require_sha256(source["archive_sha256"], "independent audit source archive")
    request_id = audit.get("fixed_request_id")
    if not isinstance(request_id, str) or REQUEST_ID_RE.fullmatch(request_id) is None:
        raise GateError("independent audit fixed request ID differs")
    runtime = audit.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
        "path", "gate", "worker", "profile", "served_model", "prepared_receipt",
        "binding", "package", "sha256sums",
    }:
        raise GateError("independent audit runtime shape differs")
    runtime_root = Path(str(runtime["path"])).resolve()
    if runtime_root.is_symlink() or not runtime_root.is_dir() or stat.S_IMODE(runtime_root.stat().st_mode) != 0o555:
        raise GateError("independent audit runtime topology differs")
    if {entry.name for entry in runtime_root.iterdir()} != {
        "gate.json", "ullm-aq4-worker", "profile.json", "served-model.json",
        "promotion-receipt.json", "build-receipt.json", "SHA256SUMS",
    }:
        raise GateError("independent audit runtime member set differs")
    for entry in runtime_root.iterdir():
        item = entry.stat(follow_symlinks=False)
        if entry.is_symlink() or not stat.S_ISREG(item.st_mode) or stat.S_IMODE(item.st_mode) not in {0o444, 0o555} or item.st_nlink != 1:
            raise GateError("independent audit runtime file topology differs")
    gate_path = runtime_root / "gate.json"
    worker_path = runtime_root / "ullm-aq4-worker"
    profile_path = runtime_root / "profile.json"
    manifest_path = runtime_root / "served-model.json"
    receipt_path = runtime_root / "promotion-receipt.json"
    sums_path = runtime_root / "SHA256SUMS"
    identities = {
        "gate_sha256": _audit_reference(runtime["gate"], gate_path, "Gate"),
        "worker_sha256": _audit_reference(runtime["worker"], worker_path, "worker"),
        "profile_sha256": _audit_reference(runtime["profile"], profile_path, "profile"),
        "manifest_sha256": _audit_reference(runtime["served_model"], manifest_path, "served model"),
        "prepared_receipt_sha256": _audit_reference(runtime["prepared_receipt"], receipt_path, "prepared receipt"),
        "sha256sums_sha256": _audit_reference(runtime["sha256sums"], sums_path, "SHA256SUMS"),
    }
    gate = read_object(gate_path, "audited Gate")
    prepared = read_object(receipt_path, "audited prepared receipt")
    build = read_object(runtime_root / "build-receipt.json", "audited build receipt")
    profile = read_object(profile_path, "audited profile")
    manifest = read_object(manifest_path, "audited served model")
    if (
        gate.get("status") != "ready_for_independent_audit"
        or gate.get("actual_run_allowed") is not False
        or gate.get("release_source_commit") != commit
        or gate.get("request", {}).get("actual", {}).get("request_id") != request_id
        or build.get("release_source_commit") != commit
        or build.get("release_source_tree") != tree
        or build.get("release_source_archive_sha256") != archive_sha256
        or build.get("promotion_request_id") != request_id
    ):
        raise GateError("independent audit Gate/build state differs")
    if (
        prepared.get("status") != "prepared_not_executed"
        or prepared.get("actual") != {"status": "pending", "required": True}
        or prepared.get("request_id") != request_id
        or prepared.get("source_commit") != commit
        or prepared.get("source_provenance") != {"tree_sha256": tree, "archive_sha256": archive_sha256}
        or manifest.get("promotion", {}).get("receipt_sha256") != identities["prepared_receipt_sha256"]
        or Path(str(profile.get("promotion", {}).get("receipt", ""))).resolve() != receipt_path
    ):
        raise GateError("independent audit prepared/profile/manifest state differs")
    product_root = Path(str(profile.get("product", {}).get("root", ""))).resolve()
    binding_path = product_root / str(profile.get("product", {}).get("artifact", {}).get("manifest_path", ""))
    package_path = product_root / str(profile.get("product", {}).get("package", {}).get("manifest_path", ""))
    binding = read_object(binding_path, "audited overlay binding")
    binding_ref = runtime.get("binding")
    if not isinstance(binding_ref, dict) or set(binding_ref) != {"path", "sha256", "content_sha256", "tensor_set_sha256", "tensor_count"}:
        raise GateError("independent audit binding reference differs")
    if (
        Path(str(binding_ref["path"])).resolve() != binding_path.resolve()
        or require_sha256(binding_ref["sha256"], "independent audit binding") != sha_file(binding_path)
        or binding_ref.get("content_sha256") != binding.get("content_sha256")
        or binding_ref.get("tensor_set_sha256") != binding.get("tensor_set_sha256")
        or binding_ref.get("tensor_count") != 48
        or prepared.get("overlay", {}).get("binding_manifest_sha256") != binding_ref["sha256"]
        or prepared.get("overlay", {}).get("content_sha256") != binding_ref["content_sha256"]
        or prepared.get("overlay", {}).get("tensor_set_sha256") != binding_ref["tensor_set_sha256"]
    ):
        raise GateError("independent audit binding identity differs")
    _audit_reference(runtime["package"], package_path, "package")
    if prepared.get("package", {}).get("manifest_sha256") != runtime["package"]["sha256"]:
        raise GateError("independent audit package identity differs")
    gate_state = audit.get("gate_state")
    if gate_state != {
        "status": "ready_for_independent_audit", "actual_run_allowed": False,
        "prepared_receipt_status": "prepared_not_executed",
        "prepared_receipt_actual": {"status": "pending", "required": True},
    }:
        raise GateError("independent audit declared Gate state differs")
    tests = audit.get("tests")
    if not isinstance(tests, dict) or tests.get("gpu_or_service_execution") is not False:
        raise GateError("independent audit execution boundary differs")
    return {
        "path": str(path),
        "sha256": audit_sha,
        "request_id": request_id,
        "runtime": str(runtime_root),
        "binding_sha256": binding_ref["sha256"],
        "package_sha256": runtime["package"]["sha256"],
        **identities,
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise GateError(f"refusing to reuse output directory: {output}")
    commit = git_value("rev-parse", f"{args.release_source_commit}^{{commit}}")
    if commit != args.release_source_commit:
        raise GateError("release source commit must be the full canonical commit id")
    profile_source = args.profile.resolve()
    worker_source = args.worker_binary.resolve()
    profile = read_object(profile_source, "overlay deployment profile")
    validate_profile(profile)
    product_root = Path(str(profile["product"]["root"])).resolve()
    binding_path = product_root / str(profile["product"]["artifact"]["manifest_path"])
    package_manifest = product_root / str(profile["product"]["package"]["manifest_path"])
    binding = read_object(binding_path, "overlay binding")
    validate_binding(binding, package_manifest)
    readiness = readiness_identity()
    source_tree = git_value("rev-parse", f"{commit}^{{tree}}")
    source_archive = source_archive_sha256(commit)
    authorize = bool(getattr(args, "authorize_actual_run", False))
    audit_path = getattr(args, "independent_audit_receipt", None)
    if authorize != (audit_path is not None):
        raise GateError("authorization flag and independent audit receipt are required together")
    audit = None
    if authorize:
        audit = validate_independent_audit(
            Path(audit_path), commit=commit, tree=source_tree, archive_sha256=source_archive
        )
        expected_output = Path(
            f"/tmp/ullm-sq8-overlay-gpu-promotion-gate-authorized-{audit['sha256'][:16]}"
        )
        if output != expected_output:
            raise GateError(f"authorized output path must be create-new {expected_output}")

    output.mkdir(mode=0o700, parents=False)
    try:
        immutable_worker = output / "ullm-aq4-worker"
        worker_identity = copy_binary_exclusive(worker_source, immutable_worker)
        request_id = fixed_promotion_request_id(
            commit=commit,
            tree=source_tree,
            archive_sha256=source_archive,
            worker_sha256=worker_identity["immutable_sha256"],
            binding_sha256=sha_file(binding_path),
            content_sha256=binding["content_sha256"],
            tensor_set_sha256=binding["tensor_set_sha256"],
            package_sha256=sha_file(package_manifest),
            readiness=readiness,
        )
        if audit is not None and (
            request_id != audit["request_id"]
            or worker_identity["immutable_sha256"] != audit["worker_sha256"]
            or sha_file(binding_path) != audit["binding_sha256"]
            or sha_file(package_manifest) != audit["package_sha256"]
        ):
            raise GateError("authorized candidate differs from independently audited identity")
        receipt_path = output / "promotion-receipt.json"
        candidate_profile = json.loads(json.dumps(profile))
        candidate_profile["worker"]["binary"] = str(immutable_worker)
        candidate_profile["promotion"] = {
            "receipt": str(receipt_path),
            "source_commit_from_receipt": ["source_commit"],
            "required_schema_version": "ullm.qwen35_aq4_sq8_overlay_promotion.v1",
            "overlay_from_receipt": ["overlay"],
            "release_from_receipt": ["release"],
            "package_from_receipt": ["package"],
            "actual_evidence_from_receipt": ["actual"],
            "request_id_from_receipt": ["request_id"],
            "authorization_audit_from_receipt": ["authorization_audit"],
            "readiness_from_receipt": ["readiness"],
            "readiness": readiness,
            "release_source_commit": commit,
        }
        profile_path = output / "profile.json"
        write_json_exclusive(profile_path, candidate_profile)

        receipt_writer = load_module("_ullm_sq8_gate_receipt_writer", RECEIPT_WRITER)
        manifest_path = output / "served-model.json"
        receipt_writer.write_receipt(
            profile_path=profile_path,
            output_path=receipt_path,
            source_tree_sha256=source_tree,
            source_archive_sha256=source_archive,
            served_model_path=manifest_path,
            request_id=request_id,
            authorization_audit_path=Path(audit["path"]) if audit is not None else None,
        )
        generator = load_module("_ullm_sq8_gate_generator", GENERATOR)
        generator.generate_prepared_candidate(profile_path, manifest_path)
        manifest_path.chmod(0o444)
        manifest = read_object(manifest_path, "candidate served-model manifest")

        build_receipt = {
            "schema_version": BUILD_SCHEMA,
            "promotion_request_id": request_id,
            "release_source_commit": commit,
            "release_source_tree": source_tree,
            "release_source_archive_sha256": source_archive,
            "build": {
                "command": ["cargo", "build", "--release", "-p", "ullm-engine", "--bin", "ullm-aq4-worker"],
                "jobs": 1,
                "environment": {"CARGO_BUILD_JOBS": "1"},
                "cargo_version": command_text(["cargo", "--version"]),
                "rustc_verbose_version": command_text(["rustc", "-vV"]),
                "cxx_version": command_text([os.environ.get("CXX", "c++"), "--version"]).splitlines()[0],
            },
            "worker": worker_identity,
            "inputs": {
                "profile_path": str(profile_source),
                "profile_sha256": sha_file(profile_source),
                "binding_path": str(binding_path),
                "binding_sha256": sha_file(binding_path),
                "artifact_content_sha256": binding["content_sha256"],
                "tensor_set_sha256": binding["tensor_set_sha256"],
                "package_manifest_path": str(package_manifest),
                "package_manifest_sha256": sha_file(package_manifest),
            },
        }
        build_receipt_path = output / "build-receipt.json"
        write_json_exclusive(build_receipt_path, build_receipt)

        gate = {
            "schema_version": SCHEMA,
            "status": "authorized_pending_execution" if authorize else "ready_for_independent_audit",
            "actual_run_allowed": authorize,
            "release_source_commit": commit,
            "classification": {
                "promotion": "unclassified",
                "fidelity": "unclassified",
                "holdout_used": False,
                "policy_relaxed": False,
            },
            "authorization": {
                "blocked_until": None if authorize else "independent_executor_and_gate_audit",
                "fresh_output_required": True,
                "maximum_actual_runs": 1,
                "max_attempts": 1 if authorize else 0,
                "service_or_gpu_commands_during_preparation": 0,
                "independent_audit_receipt": (
                    {"path": audit["path"], "sha256": audit["sha256"]}
                    if audit is not None else None
                ),
            },
            "readiness": readiness,
            "device": {
                "HIP_VISIBLE_DEVICES": "1",
                "ULLM_HIP_VISIBLE_DEVICES": "1",
                "runtime_device_index": 1,
                "amd_smi_index": 2,
                "architecture": "gfx1201",
                "exclusive_lock": "/run/ullm/device-1.lock",
            },
            "profile_identity": {
                "implementation_id": IMPLEMENTATION_ID,
                "execution_profile": EXECUTION_PROFILE,
                "artifact_binding_sha256": sha_file(binding_path),
                "artifact_content_sha256": binding["content_sha256"],
                "tensor_set_sha256": binding["tensor_set_sha256"],
                "tensor_count": 48,
                "package_manifest_sha256": sha_file(package_manifest),
                "worker_sha256": worker_identity["immutable_sha256"],
            },
            "required_environment": {name: "1" for name in REQUIRED_OVERLAY_ENV},
            "request": {
                "smoke": {"prompt_token_ids": [1], "max_new_tokens": 1, "telemetry_eligible": False},
                "actual": {
                    "request_id": request_id,
                    "prompt_token_ids": list(range(1, 129)),
                    "max_new_tokens": 1,
                    "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 0},
                    "telemetry_environment": {"ULLM_SQ8_PROMOTION_EVIDENCE_REQUEST_ID": request_id},
                },
            },
            "sequence": [
                "capture-service-prestate",
                "stop-default-service",
                "observe-two-stable-owner-free-polls",
                "prepare-candidate-runtime-directory-and-exclusive-lock",
                "verify-source-artifact-package-worker-pre-hashes",
                "load-overlay-worker-and-verify-ready-identity",
                "run-fixed-smoke-prefill-decode-without-telemetry-eligibility",
                "run-fixed-actual-request-with-request-scoped-telemetry",
                "shutdown-worker-and-verify-source-artifact-package-worker-post-hashes",
                "cleanup-candidate-runtime-and-lock",
                "restore-default-service-new-epoch-and-health",
            ],
            "actual_evidence_requirements": {
                "ready_identity_exact": True,
                "projection_counts": {
                    "batch_matvec_count": ">0",
                    "pair_matvec_count": ">0",
                    "single_matvec_count": 0,
                    "triple_matvec_count": 0,
                    "fallback_count": 0,
                },
                "diagnostic_host_staging": {"read_count": 0, "write_count": 0, "read_bytes": 0, "write_bytes": 0},
                "token_output_identity_sha256_required": True,
                "pre_post_hashes_equal": ["source", "artifact", "binding", "package", "worker"],
                "service_restore": {"new_epoch": True, "healthy": True, "lock_restored": True},
                "failure_cleanup_and_restore_required": True,
            },
            "trusted_components": {
                "maintenance_wrapper": {"path": str(MAINTENANCE), "sha256": sha_file(MAINTENANCE)},
                "executor_capture": {"path": str(CAPTURE), "sha256": sha_file(CAPTURE)},
                "served_model_generator": {"path": str(GENERATOR), "sha256": sha_file(GENERATOR)},
                "promotion_receipt_writer": {"path": str(RECEIPT_WRITER), "sha256": sha_file(RECEIPT_WRITER)},
            },
            "candidate": {
                "worker": str(immutable_worker),
                "profile": str(profile_path),
                "manifest": str(manifest_path),
                "build_receipt": str(build_receipt_path),
                "manifest_sha256": sha_file(manifest_path),
                "ready_expected": {
                    "model": manifest["public"]["id"],
                    "model_revision": manifest["public"]["revision"],
                    "artifact_content_sha256": manifest["product"]["artifact"]["content_sha256"],
                    "package_manifest_sha256": manifest["product"]["package"]["manifest_sha256"],
                    "device": "gfx1201",
                    "execution_profile": EXECUTION_PROFILE,
                },
            },
        }
        gate_path = output / "gate.json"
        write_json_exclusive(gate_path, gate)
        hashes = []
        for name in (
            "ullm-aq4-worker",
            "promotion-receipt.json",
            "profile.json",
            "served-model.json",
            "build-receipt.json",
            "gate.json",
        ):
            hashes.append(f"{sha_file(output / name)}  {name}\n")
        write_exclusive(output / "SHA256SUMS", "".join(hashes).encode("ascii"))
        directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        output.chmod(0o555)
        return {
            "output": str(output),
            "gate": str(gate_path),
            "gate_sha256": sha_file(gate_path),
            "worker_sha256": worker_identity["immutable_sha256"],
            "manifest_sha256": sha_file(manifest_path),
            "actual_run_allowed": authorize,
        }
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--worker-binary", type=Path, default=WORKER)
    parser.add_argument("--authorize-actual-run", action="store_true")
    parser.add_argument("--independent-audit-receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(materialize(args), sort_keys=True))
        return 0
    except (GateError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"SQ8 overlay GPU promotion Gate preparation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
