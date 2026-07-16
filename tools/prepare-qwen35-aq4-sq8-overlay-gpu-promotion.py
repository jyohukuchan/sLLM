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
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any

try:
    import qwen35_aq4_sq8_authorization_lineage as lineage_tool
except ModuleNotFoundError:
    from tools import qwen35_aq4_sq8_authorization_lineage as lineage_tool


ROOT = Path(__file__).resolve().parents[1]
PROFILE = (
    ROOT / "deploy/served-models/qwen35-9b-aq4-sq8-linear-qkv-z-overlay.profile.json"
)
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
EXECUTION_TIMEOUTS = {
    "ready_seconds": 900,
    "request_seconds": 240,
    "shutdown_seconds": 30,
    "outer_seconds": 1350,
}
SOURCE_ARCHIVE_TIMEOUT_SECONDS = 30.0
SOURCE_ARCHIVE_TERM_GRACE_SECONDS = 1.0
SOURCE_ARCHIVE_REAP_TIMEOUT_SECONDS = 5.0
SOURCE_ARCHIVE_DIAGNOSTIC_BYTES = 32 * 1024
AUTHORIZATION_LINEAGE_SCHEMA = "ullm.sq8_authorization_lineage.v1"
RUNTIME_MEMBERS = frozenset(
    {
        "gate.json",
        "ullm-aq4-worker",
        "profile.json",
        "served-model.json",
        "promotion-receipt.json",
        "build-receipt.json",
        "SHA256SUMS",
        "lineage-input-manifest.json",
    }
)


class GateError(RuntimeError):
    pass


class SourceArchiveError(GateError):
    def __init__(self, reason: str, cleanup_errors: tuple[str, ...] = ()) -> None:
        super().__init__(reason)
        self.reason = reason
        self.cleanup_errors = cleanup_errors


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
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        archive = subprocess.Popen(
            ["git", "archive", "--format=tar", commit],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            archive.wait(timeout=SOURCE_ARCHIVE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            cleanup = _terminate_archive_process_group(archive, archive.pid)
            raise SourceArchiveError("git archive timed out", tuple(cleanup)) from error
        if _archive_process_group_exists(archive.pid):
            cleanup = ["unexpected_process_group_descendants"]
            cleanup.extend(_terminate_archive_process_group(archive, archive.pid))
            raise SourceArchiveError("git archive cleanup failed", tuple(cleanup))
        stdout.seek(0)
        digest = hashlib.sha256()
        while chunk := stdout.read(1024 * 1024):
            digest.update(chunk)
        if archive.returncode != 0:
            stderr.seek(0)
            diagnostic = stderr.read(SOURCE_ARCHIVE_DIAGNOSTIC_BYTES).decode(errors="replace")
            raise SourceArchiveError(f"git archive failed: {diagnostic}")
        return digest.hexdigest()


def _archive_process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_archive_process_group(process: Any, pgid: int) -> list[str]:
    cleanup: list[str] = []
    try:
        owned = process.pid == pgid and os.getpgid(process.pid) == pgid
    except ProcessLookupError:
        owned = process.poll() is not None
    if not owned:
        cleanup.append("process_group_identity_invalid")
        if process.poll() is None:
            try:
                process.kill()
                process.wait(timeout=SOURCE_ARCHIVE_REAP_TIMEOUT_SECONDS)
            except (OSError, subprocess.TimeoutExpired):
                cleanup.append("process_reap_timeout")
        return cleanup
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError:
        cleanup.append("process_group_term_failed")
    try:
        process.wait(timeout=SOURCE_ARCHIVE_TERM_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    if _archive_process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            cleanup.append("process_group_kill_failed")
    if process.poll() is None:
        try:
            process.wait(timeout=SOURCE_ARCHIVE_REAP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            cleanup.append("process_reap_timeout")
    deadline = time.monotonic() + SOURCE_ARCHIVE_REAP_TIMEOUT_SECONDS
    while _archive_process_group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.01)
    if _archive_process_group_exists(pgid):
        cleanup.append("process_group_reap_timeout")
    return cleanup


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
    authorization_lineage: dict[str, Any] | None,
    authorization_lineage_manifest: dict[str, Any] | None = None,
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
        "authorization_lineage": authorization_lineage,
        "authorization_lineage_manifest": authorization_lineage_manifest,
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return "sq8-promotion-" + hashlib.sha256(encoded).hexdigest()


def authorized_output_path(audit_sha256: str) -> Path:
    digest = require_sha256(audit_sha256, "independent audit receipt SHA")
    return Path(f"/tmp/ullm-sq8-overlay-gpu-promotion-gate-authorized-{digest[:16]}")


def _prior_no_go_audit(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if path.is_symlink():
        raise GateError(
            "prior No-Go audit must be immutable 0444 single-link non-symlink"
        )
    path = path.resolve()
    metadata = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or metadata.st_nlink != 1
    ):
        raise GateError(
            "prior No-Go audit must be immutable 0444 single-link non-symlink"
        )
    receipt = read_object(path, "prior No-Go audit")
    source = receipt.get("audited_source")
    runtime = receipt.get("runtime")
    gate = runtime.get("gate") if isinstance(runtime, dict) else None
    if (
        receipt.get("schema_version") != AUDIT_SCHEMA
        or receipt.get("verdict") != "implementation_no_go"
        or receipt.get("actual") != "not_executed"
        or receipt.get("reason_code")
        != "restore_retry_terminal_identity_not_fail_closed"
        or not isinstance(source, dict)
        or not isinstance(source.get("commit"), str)
        or len(source["commit"]) != 40
        or not isinstance(gate, dict)
        or not isinstance(gate.get("sha256"), str)
        or SHA256_RE.fullmatch(gate["sha256"]) is None
    ):
        raise GateError("prior No-Go audit state differs")
    return {
        "path": str(path),
        "sha256": sha_file(path),
        "verdict": "implementation_no_go",
        "reason_code": receipt["reason_code"],
        "audited_source_commit": source["commit"],
        "audited_gate_sha256": gate["sha256"],
    }


def prior_failure_lineage(
    path: Path | None, prior_no_go_audit_path: Path | None = None
) -> dict[str, Any] | None:
    if path is None:
        return None
    if path.is_symlink():
        raise GateError(
            "prior failure receipt must be immutable 0444 single-link non-symlink"
        )
    path = path.resolve()
    value = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(value.st_mode)
        or stat.S_IMODE(value.st_mode) != 0o444
        or value.st_nlink != 1
    ):
        raise GateError(
            "prior failure receipt must be immutable 0444 single-link non-symlink"
        )
    receipt = read_object(path, "prior failure receipt")
    request_id = receipt.get("request_id")
    actual = receipt.get("actual")
    if (
        receipt.get("schema_version") != "ullm.qwen35_aq4_sq8_overlay_promotion.v1"
        or receipt.get("status") != "actual_failed"
        or not isinstance(request_id, str)
        or REQUEST_ID_RE.fullmatch(request_id) is None
        or not isinstance(actual, dict)
        or actual.get("status") != "failed"
        or actual.get("request_id") != request_id
    ):
        raise GateError("prior failure receipt state differs")
    return {
        "schema": AUTHORIZATION_LINEAGE_SCHEMA,
        "disposition": "consumed_failed_not_reusable",
        "prior_request_id": request_id,
        "prior_failure_receipt": {"path": str(path), "sha256": sha_file(path)},
        "prior_no_go_audit": _prior_no_go_audit(prior_no_go_audit_path),
    }


def _docker_inspect(kind: str, identity: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["docker", "inspect", "--type", kind, identity],
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=5,
    )
    if (
        completed.returncode != 0
        or completed.stderr
        or len(completed.stdout) > MAX_JSON_BYTES
    ):
        raise GateError(f"readiness {kind} inspect failed")
    try:
        values = json.loads(completed.stdout)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"readiness {kind} inspect JSON differs") from error
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], dict)
    ):
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
    if (
        not isinstance(network_name, str)
        or not network_name
        or not isinstance(attachment, dict)
    ):
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
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        os.close(descriptor)
    path.chmod(mode)


def write_json_exclusive(path: Path, value: dict[str, Any], mode: int = 0o444) -> None:
    raw = (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")
    write_exclusive(path, raw, mode)


def _worker_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def copy_binary_exclusive(source: Path, destination: Path) -> dict[str, Any]:
    source = source.resolve()
    source_fd = os.open(
        source, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    destination_fd: int | None = None
    digest = hashlib.sha256()
    try:
        before = os.fstat(source_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) not in {0o555, 0o755}
        ):
            raise GateError(
                "release worker must be an executable single-link regular non-symlink file"
            )
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o555,
        )
        while chunk := os.read(source_fd, 1024 * 1024):
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise OSError("worker copy made no progress")
                view = view[written:]
        os.fsync(destination_fd)
        os.fchmod(destination_fd, 0o555)
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        final = os.lstat(source)
        if _worker_fingerprint(before) != _worker_fingerprint(
            after
        ) or _worker_fingerprint(after) != _worker_fingerprint(final):
            raise GateError("release worker changed during copy")
    finally:
        os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)
    source_sha = digest.hexdigest()
    copied = destination.stat(follow_symlinks=False)
    if (
        destination.is_symlink()
        or not stat.S_ISREG(copied.st_mode)
        or stat.S_IMODE(copied.st_mode) != 0o555
        or copied.st_nlink != 1
        or copied.st_size != before.st_size
        or sha_file(destination) != source_sha
    ):
        raise GateError("immutable worker copy identity differs")
    return {
        "source_path": str(source),
        "source_sha256": source_sha,
        "source_bytes": before.st_size,
        "source_mode": f"{stat.S_IMODE(before.st_mode):04o}",
        "source_nlink": before.st_nlink,
        "immutable_path": str(destination.resolve()),
        "immutable_sha256": source_sha,
        "immutable_bytes": copied.st_size,
        "immutable_mode": "0555",
        "immutable_nlink": copied.st_nlink,
    }


def normalize_runtime_worker_identity(
    identity: dict[str, Any], worker_path: Path
) -> dict[str, Any]:
    metadata = worker_path.stat(follow_symlinks=False)
    digest = sha_file(worker_path)
    if (
        worker_path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o555
        or metadata.st_nlink != 1
        or digest != identity.get("immutable_sha256")
        or metadata.st_size != identity.get("immutable_bytes")
    ):
        raise GateError("runtime worker self identity differs")
    path = str(worker_path.resolve())
    return {
        "source_path": path,
        "source_sha256": digest,
        "source_bytes": metadata.st_size,
        "source_mode": "0555",
        "source_nlink": 1,
        "immutable_path": path,
        "immutable_sha256": digest,
        "immutable_bytes": metadata.st_size,
        "immutable_mode": "0555",
        "immutable_nlink": 1,
    }


def _string_values(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from _string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _string_values(child)
    elif isinstance(value, str):
        yield value


def reject_runtime_references(runtime: Path, forbidden_runtime: Path) -> None:
    runtime = runtime.resolve()
    forbidden_runtime = forbidden_runtime.resolve()
    if {entry.name for entry in runtime.iterdir()} != RUNTIME_MEMBERS:
        raise GateError("authorized runtime member set differs")
    forbidden_raw = str(forbidden_runtime).encode("utf-8")
    for member in sorted(runtime.iterdir()):
        raw = member.read_bytes()
        if forbidden_raw in raw:
            raise GateError(
                f"authorized runtime retains audited runtime path: {member.name}"
            )
        if member.suffix != ".json":
            continue
        value = json.loads(raw)
        for text in _string_values(value):
            candidate = text[7:] if text.startswith("file://") else text
            if not candidate.startswith("/"):
                continue
            resolved = Path(candidate).resolve(strict=False)
            if resolved == forbidden_runtime or forbidden_runtime in resolved.parents:
                raise GateError(
                    f"authorized runtime retains audited runtime path alias: {member.name}"
                )


def validate_profile(profile: dict[str, Any]) -> None:
    worker = profile.get("worker")
    if profile.get(
        "schema_version"
    ) != "ullm.served_model.profile.v1" or not isinstance(worker, dict):
        raise GateError("overlay served-model profile schema differs")
    if profile.get("format") != {
        "format_id": "AQ4_0",
        "implementation_id": IMPLEMENTATION_ID,
    }:
        raise GateError("overlay implementation identity differs")
    identity = worker.get("identity")
    if identity != {"device": "gfx1201", "execution_profile": EXECUTION_PROFILE}:
        raise GateError("overlay worker identity differs")
    required = worker.get("required_environment")
    if not isinstance(required, list) or any(
        name not in required for name in REQUIRED_OVERLAY_ENV
    ):
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
    if any(
        not isinstance(name, str)
        or not name.endswith(("in_proj_qkv.weight", "in_proj_z.weight"))
        for name in names
    ):
        raise GateError("overlay binding contains a non-QKV/Z tensor")
    for field in ("content_sha256", "tensor_set_sha256"):
        value = binding.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise GateError(f"overlay binding {field} is invalid")
    package = binding.get("package")
    if not isinstance(package, dict) or package.get("manifest_sha256") != sha_file(
        package_manifest
    ):
        raise GateError("overlay package manifest binding differs")


def _audit_reference(record: Any, expected_path: Path, label: str) -> str:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise GateError(f"independent audit {label} reference differs")
    path = Path(str(record["path"])).resolve()
    digest = require_sha256(record["sha256"], f"independent audit {label}")
    if (
        path != expected_path.resolve()
        or path.is_symlink()
        or not path.is_file()
        or sha_file(path) != digest
    ):
        raise GateError(f"independent audit {label} live identity differs")
    return digest


def validate_independent_audit(
    path: Path,
    *,
    commit: str,
    tree: str,
    archive_sha256: str,
    authorization_lineage_manifest: dict[str, Any],
) -> dict[str, Any]:
    if path.is_symlink():
        raise GateError(
            "independent audit receipt must be immutable 0444 single-link non-symlink"
        )
    path = path.resolve()
    metadata = path.stat(follow_symlinks=False)
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or metadata.st_nlink != 1
    ):
        raise GateError(
            "independent audit receipt must be immutable 0444 single-link non-symlink"
        )
    audit_sha = sha_file(path)
    audit = read_object(path, "independent audit receipt")
    if set(audit) != {
        "schema_version",
        "auditor_task_id",
        "audited_at_utc",
        "audited_source",
        "runtime",
        "fixed_request_id",
        "gate_state",
        "topology",
        "verdict",
        "actual",
        "tests",
    }:
        raise GateError("independent audit receipt shape differs")
    if (
        audit.get("schema_version") != AUDIT_SCHEMA
        or audit.get("verdict") != "implementation_ready"
        or audit.get("actual") != "not_executed"
    ):
        raise GateError("independent audit verdict differs")
    source = audit.get("audited_source")
    if source != {
        "commit": commit,
        "tree_sha256": tree,
        "archive_sha256": archive_sha256,
    }:
        raise GateError("independent audit source identity differs")
    require_sha256(source["archive_sha256"], "independent audit source archive")
    request_id = audit.get("fixed_request_id")
    if not isinstance(request_id, str) or REQUEST_ID_RE.fullmatch(request_id) is None:
        raise GateError("independent audit fixed request ID differs")
    runtime = audit.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
        "path",
        "gate",
        "worker",
        "profile",
        "served_model",
        "prepared_receipt",
        "binding",
        "package",
        "authorization_lineage_manifest",
        "sha256sums",
    }:
        raise GateError("independent audit runtime shape differs")
    runtime_root = Path(str(runtime["path"])).resolve()
    if (
        runtime_root.is_symlink()
        or not runtime_root.is_dir()
        or stat.S_IMODE(runtime_root.stat().st_mode) != 0o555
    ):
        raise GateError("independent audit runtime topology differs")
    if {entry.name for entry in runtime_root.iterdir()} != {
        "gate.json",
        "ullm-aq4-worker",
        "profile.json",
        "served-model.json",
        "promotion-receipt.json",
        "build-receipt.json",
        "SHA256SUMS",
        "lineage-input-manifest.json",
    }:
        raise GateError("independent audit runtime member set differs")
    for entry in runtime_root.iterdir():
        item = entry.stat(follow_symlinks=False)
        if (
            entry.is_symlink()
            or not stat.S_ISREG(item.st_mode)
            or stat.S_IMODE(item.st_mode) not in {0o444, 0o555}
            or item.st_nlink != 1
        ):
            raise GateError("independent audit runtime file topology differs")
    gate_path = runtime_root / "gate.json"
    worker_path = runtime_root / "ullm-aq4-worker"
    profile_path = runtime_root / "profile.json"
    manifest_path = runtime_root / "served-model.json"
    receipt_path = runtime_root / "promotion-receipt.json"
    sums_path = runtime_root / "SHA256SUMS"
    lineage_path = runtime_root / "lineage-input-manifest.json"
    identities = {
        "gate_sha256": _audit_reference(runtime["gate"], gate_path, "Gate"),
        "worker_sha256": _audit_reference(runtime["worker"], worker_path, "worker"),
        "profile_sha256": _audit_reference(runtime["profile"], profile_path, "profile"),
        "manifest_sha256": _audit_reference(
            runtime["served_model"], manifest_path, "served model"
        ),
        "prepared_receipt_sha256": _audit_reference(
            runtime["prepared_receipt"], receipt_path, "prepared receipt"
        ),
        "sha256sums_sha256": _audit_reference(
            runtime["sha256sums"], sums_path, "SHA256SUMS"
        ),
        "authorization_lineage_manifest_sha256": _audit_reference(
            runtime["authorization_lineage_manifest"],
            lineage_path,
            "authorization lineage manifest",
        ),
    }
    expected_lineage_ref = lineage_tool.make_reference(
        authorization_lineage_manifest, lineage_path
    )
    try:
        lineage_tool.validate_reference(
            expected_lineage_ref, expected_runtime_path=lineage_path
        )
    except lineage_tool.LineageError as error:
        raise GateError(
            f"independent audit authorization lineage differs: {error}"
        ) from error
    gate = read_object(gate_path, "audited Gate")
    prepared = read_object(receipt_path, "audited prepared receipt")
    build = read_object(runtime_root / "build-receipt.json", "audited build receipt")
    profile = read_object(profile_path, "audited profile")
    manifest = read_object(manifest_path, "audited served model")
    expected_actual_request = {
        "request_id": request_id,
        "prompt_token_ids": list(range(1, 129)),
        "max_new_tokens": 2,
        "eos_token_ids": [],
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "seed": 0,
        },
        "telemetry_environment": {
            "ULLM_SQ8_PROMOTION_EVIDENCE_REQUEST_ID": request_id
        },
        "timeouts": dict(EXECUTION_TIMEOUTS),
    }
    if (
        gate.get("status") != "ready_for_independent_audit"
        or gate.get("actual_run_allowed") is not False
        or gate.get("release_source_commit") != commit
        or gate.get("request", {}).get("actual") != expected_actual_request
        or build.get("release_source_commit") != commit
        or build.get("release_source_tree") != tree
        or build.get("release_source_archive_sha256") != archive_sha256
        or build.get("promotion_request_id") != request_id
        or gate.get("authorization", {}).get("lineage_manifest") != expected_lineage_ref
        or build.get("inputs", {}).get("authorization_lineage_manifest")
        != expected_lineage_ref
    ):
        raise GateError("independent audit Gate/build state differs")
    if (
        prepared.get("status") != "prepared_not_executed"
        or prepared.get("actual") != {"status": "pending", "required": True}
        or prepared.get("execution_timeouts") != EXECUTION_TIMEOUTS
        or prepared.get("request_id") != request_id
        or prepared.get("source_commit") != commit
        or prepared.get("source_provenance")
        != {"tree_sha256": tree, "archive_sha256": archive_sha256}
        or manifest.get("promotion", {}).get("receipt_sha256")
        != identities["prepared_receipt_sha256"]
        or Path(str(profile.get("promotion", {}).get("receipt", ""))).resolve()
        != receipt_path
        or profile.get("promotion", {}).get("authorization_lineage")
        != expected_lineage_ref
        or prepared.get("authorization_lineage") != expected_lineage_ref
        or manifest.get("promotion", {}).get("authorization_lineage")
        != expected_lineage_ref
    ):
        raise GateError("independent audit prepared/profile/manifest state differs")
    product_root = Path(str(profile.get("product", {}).get("root", ""))).resolve()
    binding_path = product_root / str(
        profile.get("product", {}).get("artifact", {}).get("manifest_path", "")
    )
    package_path = product_root / str(
        profile.get("product", {}).get("package", {}).get("manifest_path", "")
    )
    binding = read_object(binding_path, "audited overlay binding")
    binding_ref = runtime.get("binding")
    if not isinstance(binding_ref, dict) or set(binding_ref) != {
        "path",
        "sha256",
        "content_sha256",
        "tensor_set_sha256",
        "tensor_count",
    }:
        raise GateError("independent audit binding reference differs")
    if (
        Path(str(binding_ref["path"])).resolve() != binding_path.resolve()
        or require_sha256(binding_ref["sha256"], "independent audit binding")
        != sha_file(binding_path)
        or binding_ref.get("content_sha256") != binding.get("content_sha256")
        or binding_ref.get("tensor_set_sha256") != binding.get("tensor_set_sha256")
        or binding_ref.get("tensor_count") != 48
        or prepared.get("overlay", {}).get("binding_manifest_sha256")
        != binding_ref["sha256"]
        or prepared.get("overlay", {}).get("content_sha256")
        != binding_ref["content_sha256"]
        or prepared.get("overlay", {}).get("tensor_set_sha256")
        != binding_ref["tensor_set_sha256"]
    ):
        raise GateError("independent audit binding identity differs")
    _audit_reference(runtime["package"], package_path, "package")
    if (
        prepared.get("package", {}).get("manifest_sha256")
        != runtime["package"]["sha256"]
    ):
        raise GateError("independent audit package identity differs")
    gate_state = audit.get("gate_state")
    if gate_state != {
        "status": "ready_for_independent_audit",
        "actual_run_allowed": False,
        "prepared_receipt_status": "prepared_not_executed",
        "prepared_receipt_actual": {"status": "pending", "required": True},
    }:
        raise GateError("independent audit declared Gate state differs")
    tests = audit.get("tests")
    if (
        not isinstance(tests, dict)
        or tests.get("gpu_or_service_execution") is not False
    ):
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
    output_argument = Path(args.output)
    if not output_argument.is_absolute() or output_argument != output_argument.resolve(
        strict=False
    ):
        raise GateError("output path must be absolute and canonical")
    output = output_argument
    if output.exists() or output.is_symlink():
        raise GateError(f"refusing to reuse output directory: {output}")
    commit = git_value("rev-parse", f"{args.release_source_commit}^{{commit}}")
    if commit != args.release_source_commit:
        raise GateError("release source commit must be the full canonical commit id")
    profile_source = args.profile.resolve()
    worker_argument = Path(args.worker_binary)
    if (
        not worker_argument.is_absolute()
        or worker_argument.is_symlink()
        or worker_argument != worker_argument.resolve()
    ):
        raise GateError(
            "worker source path must be absolute, canonical, and non-symlink"
        )
    worker_source = worker_argument
    profile = read_object(profile_source, "overlay deployment profile")
    validate_profile(profile)
    product_root = Path(str(profile["product"]["root"])).resolve()
    binding_path = product_root / str(profile["product"]["artifact"]["manifest_path"])
    package_manifest = product_root / str(
        profile["product"]["package"]["manifest_path"]
    )
    binding = read_object(binding_path, "overlay binding")
    validate_binding(binding, package_manifest)
    readiness = readiness_identity()
    authorization_lineage = prior_failure_lineage(
        getattr(args, "prior_failure_receipt", None),
        getattr(args, "prior_no_go_audit_receipt", None),
    )
    source_tree = git_value("rev-parse", f"{commit}^{{tree}}")
    source_archive = source_archive_sha256(commit)
    lineage_path = getattr(args, "authorization_lineage_manifest", None)
    lineage_input = None
    lineage_request_identity = None
    current_audit_path = getattr(
        args, "current_implementation_audit_receipt", None
    )
    current_audit_sha = getattr(
        args, "current_implementation_audit_sha256", None
    )
    if (current_audit_path is None) != (current_audit_sha is None):
        raise GateError(
            "current implementation audit receipt path and SHA are required together"
        )
    expected_current_audit = None
    if current_audit_path is not None:
        expected_current_audit = {
            "path": str(Path(current_audit_path).resolve()),
            "sha256": require_sha256(
                current_audit_sha, "current implementation audit receipt SHA"
            ),
        }
    if lineage_path is not None:
        try:
            lineage_input = lineage_tool.validate_manifest(
                Path(lineage_path),
                expected_source={
                    "commit": commit,
                    "tree_oid": source_tree,
                    "archive_sha256": source_archive,
                },
                expected_current_implementation_audit=expected_current_audit,
            )
        except lineage_tool.LineageError as error:
            raise GateError(
                f"authorization lineage manifest differs: {error}"
            ) from error
        reference_schema = (
            lineage_tool.REFERENCE_SCHEMA
            if lineage_input["authorization_eligible"]
            else lineage_tool.REFERENCE_SCHEMA_V1
        )
        lineage_request_identity = {
            "schema_version": reference_schema,
            "input_path": lineage_input["path"],
            "sha256": lineage_input["sha256"],
            "entries_sha256": lineage_input["entries_sha256"],
        }
        if lineage_input["authorization_eligible"]:
            lineage_request_identity.update(
                entry_count=lineage_input["entry_count"],
                current_implementation_audit=lineage_input[
                    "current_implementation_audit"
                ],
            )
    authorize = bool(getattr(args, "authorize_actual_run", False))
    audit_path = getattr(args, "independent_audit_receipt", None)
    if authorize != (audit_path is not None):
        raise GateError(
            "authorization flag and independent audit receipt are required together"
        )
    if authorize and lineage_input is None:
        raise GateError(
            "actual authorization requires an authorization lineage manifest"
        )
    if authorize and (
        not lineage_input["authorization_eligible"]
        or expected_current_audit is None
    ):
        raise GateError(
            "actual authorization requires v2 lineage and an explicit current "
            "implementation audit receipt path and SHA"
        )
    audit = None
    if authorize:
        audit = validate_independent_audit(
            Path(audit_path),
            commit=commit,
            tree=source_tree,
            archive_sha256=source_archive,
            authorization_lineage_manifest=lineage_input,
        )
        expected_output = authorized_output_path(audit["sha256"])
        if output != expected_output:
            raise GateError(
                f"authorized output path must be create-new {expected_output}"
            )
        audited_worker = Path(audit["runtime"]) / "ullm-aq4-worker"
        if worker_source != audited_worker.resolve():
            raise GateError("authorized worker source differs from audited runtime")

    output.mkdir(mode=0o700, parents=False)
    try:
        lineage_runtime_path = output / "lineage-input-manifest.json"
        lineage_reference = None
        if lineage_input is not None:
            write_exclusive(lineage_runtime_path, lineage_input["raw"])
            lineage_runtime_path.chmod(0o444)
            lineage_reference = lineage_tool.make_reference(
                lineage_input, lineage_runtime_path
            )
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
            authorization_lineage=authorization_lineage,
            authorization_lineage_manifest=lineage_request_identity,
        )
        if audit is not None and (
            request_id != audit["request_id"]
            or worker_identity["immutable_sha256"] != audit["worker_sha256"]
            or sha_file(binding_path) != audit["binding_sha256"]
            or sha_file(package_manifest) != audit["package_sha256"]
        ):
            raise GateError(
                "authorized candidate differs from independently audited identity"
            )
        worker_identity = normalize_runtime_worker_identity(
            worker_identity, immutable_worker
        )
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
            "authorization_lineage_from_receipt": ["authorization_lineage"],
            "readiness_from_receipt": ["readiness"],
            "authorization_lineage": lineage_reference,
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
            authorization_lineage=lineage_reference,
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
                "command": [
                    "cargo",
                    "build",
                    "--release",
                    "-p",
                    "ullm-engine",
                    "--bin",
                    "ullm-aq4-worker",
                ],
                "jobs": 1,
                "environment": {"CARGO_BUILD_JOBS": "1"},
                "cargo_version": command_text(["cargo", "--version"]),
                "rustc_verbose_version": command_text(["rustc", "-vV"]),
                "cxx_version": command_text(
                    [os.environ.get("CXX", "c++"), "--version"]
                ).splitlines()[0],
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
                "prior_failure_receipt": (
                    authorization_lineage["prior_failure_receipt"]
                    if authorization_lineage is not None
                    else None
                ),
                "independent_audit_receipt": (
                    {"path": audit["path"], "sha256": audit["sha256"]}
                    if audit is not None
                    else None
                ),
                "authorization_lineage_manifest": lineage_reference,
            },
        }
        build_receipt_path = output / "build-receipt.json"
        write_json_exclusive(build_receipt_path, build_receipt)

        gate = {
            "schema_version": SCHEMA,
            "status": "authorized_pending_execution"
            if authorize
            else "ready_for_independent_audit",
            "actual_run_allowed": authorize,
            "release_source_commit": commit,
            "classification": {
                "promotion": "unclassified",
                "fidelity": "unclassified",
                "holdout_used": False,
                "policy_relaxed": False,
            },
            "authorization": {
                "blocked_until": None
                if authorize
                else "independent_executor_and_gate_audit",
                "fresh_output_required": True,
                "maximum_actual_runs": 1,
                "max_attempts": 1 if authorize else 0,
                "service_or_gpu_commands_during_preparation": 0,
                "independent_audit_receipt": (
                    {"path": audit["path"], "sha256": audit["sha256"]}
                    if audit is not None
                    else None
                ),
                "lineage": authorization_lineage,
                "lineage_manifest": lineage_reference,
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
                "smoke": {
                    "prompt_token_ids": [1],
                    "max_new_tokens": 1,
                    "telemetry_eligible": False,
                },
                "actual": {
                    "request_id": request_id,
                    "prompt_token_ids": list(range(1, 129)),
                    "max_new_tokens": 2,
                    "eos_token_ids": [],
                    "sampling": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": 1,
                        "seed": 0,
                    },
                    "telemetry_environment": {
                        "ULLM_SQ8_PROMOTION_EVIDENCE_REQUEST_ID": request_id
                    },
                    "timeouts": dict(EXECUTION_TIMEOUTS),
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
                "diagnostic_host_staging": {
                    "read_count": 0,
                    "write_count": 0,
                    "read_bytes": 0,
                    "write_bytes": 0,
                },
                "token_output_identity_sha256_required": True,
                "pre_post_hashes_equal": [
                    "source",
                    "artifact",
                    "binding",
                    "package",
                    "worker",
                ],
                "service_restore": {
                    "new_epoch": True,
                    "healthy": True,
                    "lock_restored": True,
                },
                "failure_cleanup_and_restore_required": True,
            },
            "trusted_components": {
                "maintenance_wrapper": {
                    "path": str(MAINTENANCE),
                    "sha256": sha_file(MAINTENANCE),
                },
                "executor_capture": {"path": str(CAPTURE), "sha256": sha_file(CAPTURE)},
                "served_model_generator": {
                    "path": str(GENERATOR),
                    "sha256": sha_file(GENERATOR),
                },
                "promotion_receipt_writer": {
                    "path": str(RECEIPT_WRITER),
                    "sha256": sha_file(RECEIPT_WRITER),
                },
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
                    "artifact_content_sha256": manifest["product"]["artifact"][
                        "content_sha256"
                    ],
                    "package_manifest_sha256": manifest["product"]["package"][
                        "manifest_sha256"
                    ],
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
        if lineage_reference is not None:
            hashes.append(
                f"{sha_file(lineage_runtime_path)}  lineage-input-manifest.json\n"
            )
            try:
                refreshed = lineage_tool.validate_manifest(
                    Path(lineage_reference["input_path"]),
                    expected_source={
                        "commit": commit,
                        "tree_oid": source_tree,
                        "archive_sha256": source_archive,
                    },
                )
                lineage_tool.validate_reference(
                    lineage_reference, expected_runtime_path=lineage_runtime_path
                )
            except lineage_tool.LineageError as error:
                raise GateError(
                    f"authorization lineage changed during materialization: {error}"
                ) from error
            if (
                refreshed["sha256"] != lineage_input["sha256"]
                or refreshed["entries_sha256"] != lineage_input["entries_sha256"]
            ):
                raise GateError("authorization lineage changed during materialization")
        if (
            normalize_runtime_worker_identity(worker_identity, immutable_worker)
            != worker_identity
        ):
            raise GateError("runtime worker changed during materialization")
        write_exclusive(output / "SHA256SUMS", "".join(hashes).encode("ascii"))
        if audit is not None:
            reject_runtime_references(output, Path(audit["runtime"]))
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
    parser.add_argument("--prior-failure-receipt", type=Path)
    parser.add_argument("--prior-no-go-audit-receipt", type=Path)
    parser.add_argument("--authorization-lineage-manifest", type=Path)
    parser.add_argument("--current-implementation-audit-receipt", type=Path)
    parser.add_argument("--current-implementation-audit-sha256")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(materialize(args), sort_keys=True))
        return 0
    except (GateError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(
            f"SQ8 overlay GPU promotion Gate preparation failed: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
