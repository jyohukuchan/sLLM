#!/usr/bin/env python3
"""Single-use maintenance wrapper for the Qwen3.5 AQ4 SQ8 overlay candidate.

The wrapper is candidate-specific.  It never invokes the production P2 launcher.
Real execution requires an explicit audited confirmation; tests inject all lifecycle
operations and therefore do not touch systemd or a GPU.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    import qwen35_aq4_sq8_authorization_lineage as lineage_tool
except ModuleNotFoundError:
    from tools import qwen35_aq4_sq8_authorization_lineage as lineage_tool


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "tools/capture-aq4-resident-executor-record.py"
RECEIPT_WRITER = ROOT / "tools/write-qwen35-aq4-sq8-overlay-promotion-receipt.py"
SCHEMA = "ullm.qwen35_aq4.sq8_overlay_gpu_promotion_maintenance.v1"
GATE_SCHEMA = "ullm.qwen35_aq4.sq8_overlay_gpu_promotion_gate.v1"
TELEMETRY_SCHEMA = "ullm.qwen35_aq4.sq8_promotion_telemetry.v1"
TELEMETRY_BINDING_SCHEMA = "ullm.qwen35_aq4.sq8_promotion_telemetry_binding.v1"
TELEMETRY_HASH_ENCODING = "canonical_json_ascii_sort_keys_compact_v1"
IMPLEMENTATION_ID = "qwen35_aq4_sq8_linear_qkv_z_overlay_v1"
EXECUTION_PROFILE = "rdna4_aq4_resident_sq8_linear_qkv_z_overlay"
SERVICE = "ullm-openai.service"
LOCK_PATH = Path("/run/ullm/device-1.lock")
LOCK_HELPER = ROOT / "tools/manage-qwen35-aq4-sq8-overlay-lock.py"
LOCK_UID = 1000
LOCK_GID = 1000
STOP_TIMEOUT_SECONDS = 30.0
RESTORE_TIMEOUT_SECONDS = 120.0
POLL_SECONDS = 0.25
MAX_JSON_BYTES = 16 * 1024 * 1024
CAPTURE_DIAGNOSTIC_MAX_BYTES = 32 * 1024
CAPTURE_ENVELOPE_MAX_BYTES = 512 * 1024
CAPTURE_READ_CHUNK_BYTES = 64 * 1024
CAPTURE_SUBPROCESS_TIMEOUT_SECONDS = 300.0
CAPTURE_ERROR_SCHEMA = "ullm.aq4_resident_capture_error.v3"
WORKER_STDERR_SCHEMA = "ullm.aq4_resident_worker_stderr.v1"
CAPTURE_ERROR_STAGES = frozenset(
    {
        "capture",
        "request",
        "shutdown",
        "cleanup",
        "worker",
        "worker_exit",
        "audit_missing",
        "resource_observation",
        "package_validation",
        "telemetry_validation",
        "validation",
    }
)
PROMOTION_REQUEST_ID_RE = re.compile(r"^sq8-promotion-[0-9a-f]{64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DOCKER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
READY_URL = "http://172.20.0.1:8000/readyz"
READY_BODY = b'{"status":"ready"}'
READINESS_SCHEMA = "ullm.bridge_container_readiness.v1"
AUTHORIZATION_LINEAGE_SCHEMA = "ullm.sq8_authorization_lineage.v1"
AUTHORIZED_RUNTIME_MEMBERS = frozenset(
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
READY_CONTAINER_NAME = "open-webui"
READY_PATH = "/readyz"
READY_TIMEOUT_SECONDS = 5
DOCKER_INSPECT_MAX_BYTES = 256 * 1024
PRODUCTION_LOCK_PATH = Path("/run/ullm/r9700.lock")
AMD_SMI = Path("/opt/rocm/bin/amd-smi")
AMD_SMI_INDEX = 2
KFD_PROC_ROOT = Path("/sys/class/kfd/kfd/proc")
KFD_ID = 51545
REQUIRED_OVERLAY_ENV = (
    "ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL",
    "ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_PAIR_KERNEL",
    "ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL",
    "ULLM_DISABLE_AQ4_MATVEC_QKV_Z_GATE_BETA",
)


class PromotionError(RuntimeError):
    pass


class TransientRestoreError(PromotionError):
    """A narrowly allowlisted startup state that may become ready."""


class TerminalRestoreError(PromotionError):
    """A restore invariant violation that must never be retried."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise PromotionError(f"{label} must be a bounded regular non-symlink file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PromotionError(f"cannot parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise PromotionError(f"{label} must be a JSON object")
    return value


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def sq8_telemetry_binding_valid(
    binding: Any, telemetry: Any, expected_request_id: str
) -> bool:
    if (
        PROMOTION_REQUEST_ID_RE.fullmatch(expected_request_id) is None
        or not isinstance(binding, dict)
        or set(binding) != {
            "schema_version",
            "request_id",
            "hash_encoding",
            "telemetry_sha256",
        }
        or binding.get("schema_version") != TELEMETRY_BINDING_SCHEMA
        or binding.get("request_id") != expected_request_id
        or binding.get("hash_encoding") != TELEMETRY_HASH_ENCODING
        or not isinstance(binding.get("telemetry_sha256"), str)
        or SHA256_RE.fullmatch(binding["telemetry_sha256"]) is None
    ):
        return False
    try:
        return binding["telemetry_sha256"] == canonical_sha(telemetry)
    except (TypeError, ValueError, UnicodeError):
        return False


def validate_readiness_contract(value: Any) -> dict[str, Any]:
    """Validate the exact Gate-bound Open WebUI readiness identity."""

    if not isinstance(value, dict) or set(value) != {
        "schema",
        "container",
        "network",
        "endpoint",
    }:
        raise PromotionError("candidate readiness contract shape differs")
    if value.get("schema") != READINESS_SCHEMA:
        raise PromotionError("candidate readiness contract schema differs")
    container = value.get("container")
    network = value.get("network")
    endpoint = value.get("endpoint")
    if not isinstance(container, dict) or set(container) != {
        "name",
        "id",
        "image_id",
        "config_image",
    }:
        raise PromotionError("candidate readiness container identity differs")
    if not isinstance(network, dict) or set(network) != {
        "name",
        "id",
        "driver",
        "bridge_interface",
    }:
        raise PromotionError("candidate readiness network identity differs")
    if not isinstance(endpoint, dict) or set(endpoint) != {
        "url",
        "path",
        "expected_status",
        "expected_body",
        "expected_body_sha256",
        "timeout_seconds",
    }:
        raise PromotionError("candidate readiness endpoint contract differs")

    container_id = container.get("id")
    image_id = container.get("image_id")
    config_image = container.get("config_image")
    if (
        container.get("name") != READY_CONTAINER_NAME
        or not isinstance(container_id, str)
        or SHA256_RE.fullmatch(container_id) is None
        or not isinstance(image_id, str)
        or not image_id.startswith("sha256:")
        or SHA256_RE.fullmatch(image_id.removeprefix("sha256:")) is None
        or not isinstance(config_image, str)
        or not config_image
        or len(config_image.encode("utf-8")) > 4096
        or any(ord(character) < 0x20 for character in config_image)
    ):
        raise PromotionError("candidate readiness container identity differs")

    network_id = network.get("id")
    network_name = network.get("name")
    bridge_interface = network.get("bridge_interface")
    if (
        not isinstance(network_name, str)
        or DOCKER_NAME_RE.fullmatch(network_name) is None
        or not isinstance(network_id, str)
        or SHA256_RE.fullmatch(network_id) is None
        or network.get("driver") != "bridge"
        or bridge_interface != f"br-{network_id[:12]}"
    ):
        raise PromotionError("candidate readiness network identity differs")

    expected_body = endpoint.get("expected_body")
    if (
        endpoint.get("url") != READY_URL
        or endpoint.get("path") != READY_PATH
        or endpoint.get("expected_status") != 200
        or type(endpoint.get("expected_status")) is not int
        or expected_body != READY_BODY.decode("ascii")
        or endpoint.get("expected_body_sha256")
        != hashlib.sha256(READY_BODY).hexdigest()
        or endpoint.get("timeout_seconds") != READY_TIMEOUT_SECONDS
        or type(endpoint.get("timeout_seconds")) is not int
    ):
        raise PromotionError("candidate readiness endpoint contract differs")
    return json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))


def validate_authorization_lineage(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "disposition",
        "prior_request_id",
        "prior_failure_receipt",
        "prior_no_go_audit",
    }:
        raise PromotionError("candidate authorization lineage shape differs")
    request_id = value.get("prior_request_id")
    reference = value.get("prior_failure_receipt")
    if (
        value.get("schema") != AUTHORIZATION_LINEAGE_SCHEMA
        or value.get("disposition") != "consumed_failed_not_reusable"
        or not isinstance(request_id, str)
        or PROMOTION_REQUEST_ID_RE.fullmatch(request_id) is None
        or not isinstance(reference, dict)
        or set(reference) != {"path", "sha256"}
    ):
        raise PromotionError("candidate authorization lineage identity differs")
    raw_path = reference.get("path")
    digest = reference.get("sha256")
    if (
        not isinstance(raw_path, str)
        or not Path(raw_path).is_absolute()
        or not isinstance(digest, str)
        or SHA256_RE.fullmatch(digest) is None
    ):
        raise PromotionError("candidate prior failure receipt identity differs")
    path = Path(raw_path)
    metadata = path.stat(follow_symlinks=False)
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o444
        or metadata.st_nlink != 1
        or path.resolve() != path
        or sha_file(path) != digest
    ):
        raise PromotionError("candidate prior failure receipt file differs")
    receipt = read_object(path, "prior failure receipt")
    actual = receipt.get("actual")
    if (
        receipt.get("schema_version") != "ullm.qwen35_aq4_sq8_overlay_promotion.v1"
        or receipt.get("status") != "actual_failed"
        or receipt.get("request_id") != request_id
        or not isinstance(actual, dict)
        or actual.get("status") != "failed"
        or actual.get("request_id") != request_id
    ):
        raise PromotionError("candidate prior failure receipt state differs")
    no_go = value.get("prior_no_go_audit")
    if no_go is not None:
        if not isinstance(no_go, dict) or set(no_go) != {
            "path",
            "sha256",
            "verdict",
            "reason_code",
            "audited_source_commit",
            "audited_gate_sha256",
        }:
            raise PromotionError("candidate prior No-Go audit identity differs")
        raw_audit_path = no_go.get("path")
        if (
            no_go.get("verdict") != "implementation_no_go"
            or no_go.get("reason_code")
            != "restore_retry_terminal_identity_not_fail_closed"
            or not isinstance(raw_audit_path, str)
            or not Path(raw_audit_path).is_absolute()
            or not isinstance(no_go.get("sha256"), str)
            or SHA256_RE.fullmatch(no_go["sha256"]) is None
            or not isinstance(no_go.get("audited_source_commit"), str)
            or len(no_go["audited_source_commit"]) != 40
            or not isinstance(no_go.get("audited_gate_sha256"), str)
            or SHA256_RE.fullmatch(no_go["audited_gate_sha256"]) is None
        ):
            raise PromotionError("candidate prior No-Go audit identity differs")
        audit_path = Path(raw_audit_path)
        audit_metadata = audit_path.stat(follow_symlinks=False)
        if (
            audit_path.is_symlink()
            or not stat.S_ISREG(audit_metadata.st_mode)
            or stat.S_IMODE(audit_metadata.st_mode) != 0o444
            or audit_metadata.st_nlink != 1
            or audit_path.resolve() != audit_path
            or sha_file(audit_path) != no_go["sha256"]
        ):
            raise PromotionError("candidate prior No-Go audit file differs")
        audit = read_object(audit_path, "prior No-Go audit")
        audit_source = audit.get("audited_source")
        audit_runtime = audit.get("runtime")
        audit_gate = (
            audit_runtime.get("gate") if isinstance(audit_runtime, dict) else None
        )
        if (
            audit.get("schema_version")
            != "ullm.qwen35_aq4_sq8_overlay_independent_audit.v1"
            or audit.get("verdict") != no_go["verdict"]
            or audit.get("actual") != "not_executed"
            or audit.get("reason_code") != no_go["reason_code"]
            or not isinstance(audit_source, dict)
            or audit_source.get("commit") != no_go["audited_source_commit"]
            or not isinstance(audit_gate, dict)
            or audit_gate.get("sha256") != no_go["audited_gate_sha256"]
        ):
            raise PromotionError("candidate prior No-Go audit state differs")
    return json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))


def metadata(path: Path, label: str, *, executable: bool = False) -> dict[str, Any]:
    value = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise PromotionError(f"{label} must be a single-link regular non-symlink file")
    if executable and not value.st_mode & stat.S_IXUSR:
        raise PromotionError(f"{label} must be executable")
    return {
        "path": str(path.resolve()),
        "sha256": sha_file(path),
        "bytes": value.st_size,
        "mode": f"{stat.S_IMODE(value.st_mode):04o}",
        "uid": value.st_uid,
        "gid": value.st_gid,
        "nlink": value.st_nlink,
        "device": value.st_dev,
        "inode": value.st_ino,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def _nested_strings(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from _nested_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_strings(child)
    elif isinstance(value, str):
        yield value


def validate_authorized_runtime_references(candidate: Path, audit_path: Path) -> None:
    audit = read_object(audit_path, "candidate independent audit")
    raw_runtime = audit.get("runtime", {}).get("path")
    if not isinstance(raw_runtime, str) or not Path(raw_runtime).is_absolute():
        raise PromotionError("candidate audited runtime path differs")
    audited_runtime = Path(raw_runtime).resolve()
    if {member.name for member in candidate.iterdir()} != AUTHORIZED_RUNTIME_MEMBERS:
        raise PromotionError("authorized candidate runtime member set differs")
    audited_raw = str(audited_runtime).encode("utf-8")
    for member in sorted(candidate.iterdir()):
        raw = member.read_bytes()
        if audited_raw in raw:
            raise PromotionError(
                f"authorized candidate retains audited runtime path: {member.name}"
            )
        if member.suffix != ".json":
            continue
        for text in _nested_strings(json.loads(raw)):
            candidate_path = text[7:] if text.startswith("file://") else text
            if not candidate_path.startswith("/"):
                continue
            resolved = Path(candidate_path).resolve(strict=False)
            if resolved == audited_runtime or audited_runtime in resolved.parents:
                raise PromotionError(
                    f"authorized candidate retains audited runtime path alias: {member.name}"
                )


def tree_inventory(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise PromotionError(
            "overlay artifact root must be a directory without symlinks"
        )
    entries: list[dict[str, Any]] = []
    directory_count = 0
    file_count = 0
    symlink_count = 0
    total_bytes = 0
    for path in sorted(
        (root, *root.rglob("*")),
        key=lambda item: item.relative_to(root).as_posix() if item != root else "",
    ):
        relative = "." if path == root else path.relative_to(root).as_posix()
        value = path.stat(follow_symlinks=False)
        if stat.S_ISLNK(value.st_mode):
            symlink_count += 1
            kind = "symlink"
        elif stat.S_ISDIR(value.st_mode):
            directory_count += 1
            kind = "directory"
        elif stat.S_ISREG(value.st_mode):
            file_count += 1
            total_bytes += value.st_size
            kind = "file"
        else:
            raise PromotionError(f"overlay artifact has unsupported entry: {relative}")
        entries.append(
            {
                "path": relative,
                "kind": kind,
                "bytes": value.st_size if kind == "file" else 0,
                "mode": f"{stat.S_IMODE(value.st_mode):04o}",
                "uid": value.st_uid,
                "gid": value.st_gid,
                "nlink": value.st_nlink,
                "device": value.st_dev,
                "inode": value.st_ino,
                "mtime_ns": value.st_mtime_ns,
                "ctime_ns": value.st_ctime_ns,
            }
        )
    if symlink_count != 0:
        raise PromotionError("overlay artifact contains a symlink")
    return {
        "directory_count": directory_count,
        "regular_file_count": file_count,
        "symlink_count": symlink_count,
        "bytes": total_bytes,
        "entries_sha256": canonical_sha(entries),
        "entries": entries,
    }


def source_archive_sha256(commit: str) -> str:
    process = subprocess.Popen(
        ["git", "archive", "--format=tar", commit],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    digest = hashlib.sha256()
    while chunk := process.stdout.read(1024 * 1024):
        digest.update(chunk)
    _, stderr = process.communicate(timeout=30)
    if process.returncode != 0:
        raise PromotionError(f"git archive failed: {stderr.decode(errors='replace')}")
    return digest.hexdigest()


def source_identity(candidate: Path, build: dict[str, Any]) -> dict[str, Any]:
    commit = str(build.get("release_source_commit", ""))
    tree = str(build.get("release_source_tree", ""))
    archive = str(build.get("release_source_archive_sha256", ""))
    if len(commit) != 40 or len(tree) != 40 or len(archive) != 64:
        raise PromotionError("candidate build receipt source identity is invalid")
    completed = subprocess.run(
        ["git", "rev-parse", f"{commit}^{{tree}}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0 or completed.stdout.strip() != tree:
        raise PromotionError("candidate release source tree differs")
    actual_archive = source_archive_sha256(commit)
    if actual_archive != archive:
        raise PromotionError("candidate release source archive differs")
    return {"commit": commit, "tree": tree, "archive_sha256": actual_archive}


def validate_build_worker_identity(
    candidate: Path, build: dict[str, Any], *, authorized: bool
) -> None:
    worker_path = (candidate / "ullm-aq4-worker").resolve()
    worker = build.get("worker")
    expected_keys = {
        "source_path",
        "source_sha256",
        "source_bytes",
        "source_mode",
        "source_nlink",
        "immutable_path",
        "immutable_sha256",
        "immutable_bytes",
        "immutable_mode",
        "immutable_nlink",
    }
    if not isinstance(worker, dict) or set(worker) != expected_keys:
        raise PromotionError("candidate build worker receipt shape differs")
    immutable = metadata(worker_path, "candidate build worker", executable=True)
    expected_immutable = {
        "immutable_path": str(worker_path),
        "immutable_sha256": immutable["sha256"],
        "immutable_bytes": immutable["bytes"],
        "immutable_mode": immutable["mode"],
        "immutable_nlink": 1,
    }
    if immutable["mode"] != "0555" or any(
        worker.get(key) != value for key, value in expected_immutable.items()
    ):
        raise PromotionError("candidate build immutable worker identity differs")
    source_path = Path(str(worker["source_path"]))
    if not source_path.is_absolute() or source_path != source_path.resolve():
        raise PromotionError("candidate build worker source path is not absolute")
    if authorized and source_path != worker_path:
        raise PromotionError(
            "authorized candidate worker source is not self-referential"
        )
    source = metadata(source_path, "candidate build worker source", executable=True)
    expected_source = {
        "source_sha256": source["sha256"],
        "source_bytes": source["bytes"],
        "source_mode": source["mode"],
        "source_nlink": source["nlink"],
    }
    if any(worker.get(key) != value for key, value in expected_source.items()):
        raise PromotionError("candidate build source worker identity differs")
    if authorized and any(
        worker.get(source_key) != worker.get(immutable_key)
        for source_key, immutable_key in (
            ("source_sha256", "immutable_sha256"),
            ("source_bytes", "immutable_bytes"),
            ("source_mode", "immutable_mode"),
            ("source_nlink", "immutable_nlink"),
        )
    ):
        raise PromotionError(
            "authorized candidate worker source/immutable identity differs"
        )


def candidate_runtime_fingerprint(candidate: Path) -> tuple[Any, ...]:
    root = candidate.stat(follow_symlinks=False)
    if candidate.is_symlink() or not stat.S_ISDIR(root.st_mode):
        raise PromotionError("candidate runtime must be a real directory")

    def fingerprint(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_uid,
            value.st_gid,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    members = []
    for member in sorted(candidate.iterdir()):
        value = member.stat(follow_symlinks=False)
        if (
            member.is_symlink()
            or not stat.S_ISREG(value.st_mode)
            or value.st_nlink != 1
        ):
            raise PromotionError("candidate runtime member topology differs")
        members.append((member.name, fingerprint(value)))
    return fingerprint(root), tuple(members)


def candidate_snapshot(candidate: Path) -> dict[str, Any]:
    initial_runtime_fingerprint = candidate_runtime_fingerprint(candidate)
    gate_path = candidate / "gate.json"
    build_path = candidate / "build-receipt.json"
    profile_path = candidate / "profile.json"
    manifest_path = candidate / "served-model.json"
    worker_path = candidate / "ullm-aq4-worker"
    gate = read_object(gate_path, "candidate Gate")
    build = read_object(build_path, "candidate build receipt")
    profile = read_object(profile_path, "candidate profile")
    manifest = read_object(manifest_path, "candidate manifest")
    validate_build_worker_identity(
        candidate, build, authorized=gate.get("actual_run_allowed") is True
    )
    promotion = profile.get("promotion")
    legacy_promotion_keys = {
        "receipt",
        "source_commit_from_receipt",
        "required_schema_version",
        "overlay_from_receipt",
        "release_from_receipt",
        "package_from_receipt",
        "actual_evidence_from_receipt",
        "request_id_from_receipt",
        "authorization_audit_from_receipt",
        "readiness_from_receipt",
        "readiness",
        "release_source_commit",
    }
    lineage_promotion_keys = legacy_promotion_keys | {
        "authorization_lineage_from_receipt",
        "authorization_lineage",
    }
    if not isinstance(promotion, dict) or (
        set(promotion) != legacy_promotion_keys
        and set(promotion) != lineage_promotion_keys
    ):
        raise PromotionError("candidate strict promotion profile differs")
    has_lineage_contract = set(promotion) == lineage_promotion_keys
    if promotion.get("authorization_audit_from_receipt") != ["authorization_audit"]:
        raise PromotionError("candidate authorization audit profile binding differs")
    if has_lineage_contract and promotion.get("authorization_lineage_from_receipt") != [
        "authorization_lineage"
    ]:
        raise PromotionError("candidate authorization lineage profile binding differs")
    if promotion.get("readiness_from_receipt") != ["readiness"]:
        raise PromotionError("candidate readiness profile binding differs")
    receipt_path = Path(str(promotion["receipt"])).resolve()
    receipt = read_object(receipt_path, "candidate promotion receipt")
    if gate.get("schema_version") != GATE_SCHEMA:
        raise PromotionError("candidate Gate schema differs")
    readiness = validate_readiness_contract(gate.get("readiness"))
    if promotion.get("readiness") != readiness:
        raise PromotionError("candidate readiness profile identity differs")
    if gate.get("release_source_commit") != build.get("release_source_commit"):
        raise PromotionError("candidate Gate and build source commits differ")
    identity = gate.get("profile_identity")
    worker_identity = manifest.get("worker", {}).get("identity")
    if (
        not isinstance(identity, dict)
        or identity.get("implementation_id") != IMPLEMENTATION_ID
    ):
        raise PromotionError("candidate implementation identity differs")
    if worker_identity != {"device": "gfx1201", "execution_profile": EXECUTION_PROFILE}:
        raise PromotionError("candidate worker identity differs")
    required = profile.get("worker", {}).get("required_environment")
    if not isinstance(required, list) or any(
        name not in required for name in REQUIRED_OVERLAY_ENV
    ):
        raise PromotionError("candidate overlay environment contract is incomplete")
    product_root = Path(str(profile.get("product", {}).get("root", ""))).resolve()
    binding = product_root / str(profile["product"]["artifact"]["manifest_path"])
    package = product_root / str(profile["product"]["package"]["manifest_path"])
    binding_value = read_object(binding, "overlay binding")
    artifact_root = binding.parent
    inventory = tree_inventory(artifact_root)
    package_inventory = tree_inventory(package.parent)
    expected = {
        "artifact_binding_sha256": sha_file(binding),
        "artifact_content_sha256": binding_value.get("content_sha256"),
        "tensor_set_sha256": binding_value.get("tensor_set_sha256"),
        "package_manifest_sha256": sha_file(package),
        "worker_sha256": sha_file(worker_path),
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise PromotionError("candidate Gate live identity differs")
    request_id = gate.get("request", {}).get("actual", {}).get("request_id")
    if (
        not isinstance(request_id, str)
        or PROMOTION_REQUEST_ID_RE.fullmatch(request_id) is None
    ):
        raise PromotionError("candidate Gate promotion request ID differs")
    receipt_keys = {
        "schema_version",
        "status",
        "request_id",
        "source_commit",
        "source_provenance",
        "release",
        "overlay",
        "package",
        "authorization_audit",
        "actual",
        "readiness",
    }
    if has_lineage_contract:
        receipt_keys.add("authorization_lineage")
    if set(receipt) != receipt_keys:
        raise PromotionError("candidate promotion receipt shape differs")
    source = receipt.get("source_provenance")
    if (
        receipt.get("schema_version") != "ullm.qwen35_aq4_sq8_overlay_promotion.v1"
        or receipt.get("status") != "prepared_not_executed"
        or receipt.get("actual") != {"status": "pending", "required": True}
        or receipt.get("request_id") != request_id
        or build.get("promotion_request_id") != request_id
        or receipt.get("source_commit") != build.get("release_source_commit")
        or source
        != {
            "tree_sha256": build.get("release_source_tree"),
            "archive_sha256": build.get("release_source_archive_sha256"),
        }
    ):
        raise PromotionError("candidate promotion receipt source/status differs")
    release = receipt.get("release")
    receipt_overlay = receipt.get("overlay")
    receipt_package = receipt.get("package")
    expected_entries = []
    for entry in inventory["entries"]:
        expected_entries.append(
            {
                "path": entry["path"],
                "kind": "regular" if entry["kind"] == "file" else entry["kind"],
                "mode": entry["mode"],
                "uid": entry["uid"],
                "gid": entry["gid"],
                "nlink": entry["nlink"],
                "bytes": entry["bytes"],
            }
        )
    receipt_inventory = (
        receipt_overlay.get("artifact_inventory")
        if isinstance(receipt_overlay, dict)
        else None
    )
    if not isinstance(release, dict) or release.get("worker") != {
        "path": str(worker_path.resolve()),
        "sha256": sha_file(worker_path),
        "bytes": worker_path.stat().st_size,
        "mode": "0555",
        "nlink": 1,
    }:
        raise PromotionError("candidate promotion receipt worker differs")
    if release.get("profile") != {
        "path": str(profile_path.resolve()),
        "sha256": sha_file(profile_path),
    }:
        raise PromotionError("candidate promotion receipt profile differs")
    served_release = release.get("served_model")
    semantic_manifest = json.loads(json.dumps(manifest))
    if isinstance(semantic_manifest.get("promotion"), dict):
        semantic_manifest["promotion"].pop("receipt_sha256", None)
    if served_release != {
        "path": str(manifest_path.resolve()),
        "semantic_sha256": canonical_sha(semantic_manifest),
    }:
        raise PromotionError("candidate promotion receipt served manifest differs")
    if not isinstance(receipt_overlay, dict) or any(
        receipt_overlay.get(key) != value
        for key, value in {
            "binding_manifest_path": str(binding.resolve()),
            "binding_manifest_sha256": sha_file(binding),
            "content_sha256": binding_value.get("content_sha256"),
            "tensor_set_sha256": binding_value.get("tensor_set_sha256"),
            "tensor_count": 48,
        }.items()
    ):
        raise PromotionError("candidate promotion receipt overlay differs")
    if (
        not isinstance(receipt_inventory, dict)
        or receipt_inventory.get("entries") != expected_entries
    ):
        raise PromotionError("candidate promotion receipt inventory differs")
    if receipt_package != {
        "manifest_path": str(package.resolve()),
        "manifest_sha256": sha_file(package),
    }:
        raise PromotionError("candidate promotion receipt package differs")
    if manifest.get("promotion", {}).get("receipt_sha256") != sha_file(receipt_path):
        raise PromotionError("candidate served manifest receipt SHA differs")
    authorization = gate.get("authorization")
    actual_run_allowed = gate.get("actual_run_allowed")
    authorization_audit = receipt.get("authorization_audit")
    if not isinstance(authorization, dict) or actual_run_allowed not in {False, True}:
        raise PromotionError("candidate Gate authorization differs")
    expected_status = (
        "authorized_pending_execution"
        if actual_run_allowed
        else "ready_for_independent_audit"
    )
    expected_attempts = 1 if actual_run_allowed else 0
    if (
        gate.get("status") != expected_status
        or authorization.get("fresh_output_required") is not True
        or authorization.get("maximum_actual_runs") != 1
        or authorization.get("max_attempts") != expected_attempts
        or authorization.get("service_or_gpu_commands_during_preparation") != 0
    ):
        raise PromotionError("candidate Gate authorization policy differs")
    manifest_audit = manifest.get("promotion", {}).get("authorization_audit")
    lineage_manifest = (
        receipt.get("authorization_lineage") if has_lineage_contract else None
    )
    if has_lineage_contract:
        if (
            promotion.get("authorization_lineage") != lineage_manifest
            or manifest.get("promotion", {}).get("authorization_lineage")
            != lineage_manifest
            or authorization.get("lineage_manifest") != lineage_manifest
            or build.get("inputs", {}).get("authorization_lineage_manifest")
            != lineage_manifest
        ):
            raise PromotionError(
                "candidate authorization lineage manifest propagation differs"
            )
        if lineage_manifest is not None:
            if (
                actual_run_allowed
                and lineage_manifest.get("schema_version")
                != lineage_tool.REFERENCE_SCHEMA
            ):
                raise PromotionError(
                    "authorized candidate requires authorization lineage v2"
                )
            try:
                lineage_tool.validate_reference(
                    lineage_manifest,
                    expected_runtime_path=candidate / "lineage-input-manifest.json",
                )
            except (lineage_tool.LineageError, OSError) as error:
                raise PromotionError(
                    f"candidate authorization lineage manifest differs: {error}"
                ) from error
    lineage = validate_authorization_lineage(authorization.get("lineage"))
    build_inputs = build.get("inputs")
    if not isinstance(build_inputs, dict) or build_inputs.get(
        "prior_failure_receipt"
    ) != (lineage["prior_failure_receipt"] if lineage is not None else None):
        raise PromotionError("candidate authorization lineage build binding differs")
    if build_inputs.get("independent_audit_receipt") != authorization_audit:
        raise PromotionError("candidate authorization audit build binding differs")
    manifest_readiness = manifest.get("promotion", {}).get("readiness")
    if receipt.get("readiness") != readiness or manifest_readiness != readiness:
        raise PromotionError("candidate readiness propagation differs")
    if actual_run_allowed:
        if not has_lineage_contract or lineage_manifest is None:
            raise PromotionError("authorized candidate lineage manifest is incomplete")
        if not isinstance(authorization_audit, dict) or set(authorization_audit) != {
            "path",
            "sha256",
        }:
            raise PromotionError("authorized candidate audit binding is incomplete")
        raw_audit_path = authorization_audit.get("path")
        audit_sha256 = authorization_audit.get("sha256")
        if (
            not isinstance(raw_audit_path, str)
            or not Path(raw_audit_path).is_absolute()
            or not isinstance(audit_sha256, str)
            or SHA256_RE.fullmatch(audit_sha256) is None
        ):
            raise PromotionError("authorized candidate audit identity differs")
        audit_path = Path(raw_audit_path)
        audit_stat = audit_path.stat(follow_symlinks=False)
        if (
            audit_path.is_symlink()
            or not stat.S_ISREG(audit_stat.st_mode)
            or stat.S_IMODE(audit_stat.st_mode) != 0o444
            or audit_stat.st_nlink != 1
            or audit_path.resolve() != audit_path
            or sha_file(audit_path) != audit_sha256
        ):
            raise PromotionError("authorized candidate audit file differs")
        if (
            authorization.get("blocked_until") is not None
            or authorization.get("independent_audit_receipt") != authorization_audit
            or manifest_audit != authorization_audit
        ):
            raise PromotionError("authorized candidate audit propagation differs")
        validate_authorized_runtime_references(candidate, audit_path)
    elif (
        authorization.get("blocked_until") != "independent_executor_and_gate_audit"
        or authorization.get("independent_audit_receipt") is not None
        or authorization_audit is not None
        or manifest_audit is not None
    ):
        raise PromotionError("unauthorized candidate audit state differs")
    result = {
        "source": source_identity(candidate, build),
        "files": {
            "gate": metadata(gate_path, "candidate Gate"),
            "build_receipt": metadata(build_path, "candidate build receipt"),
            "profile": metadata(profile_path, "candidate profile"),
            "manifest": metadata(manifest_path, "candidate manifest"),
            "worker": metadata(worker_path, "candidate worker", executable=True),
            "binding": metadata(binding, "overlay binding"),
            "package_manifest": metadata(package, "package manifest"),
            "promotion_receipt": metadata(receipt_path, "promotion receipt"),
        },
        "overlay": {
            "content_sha256": binding_value["content_sha256"],
            "tensor_set_sha256": binding_value["tensor_set_sha256"],
            "tensor_names": binding_value.get("tensor_names"),
            "inventory": inventory,
        },
        "package": {
            "inventory": package_inventory,
        },
        "source_provenance_sha256": canonical_sha(binding_value.get("source")),
        "authorization": {
            "actual_run_allowed": actual_run_allowed,
            "status": expected_status,
            "max_attempts": expected_attempts,
            "independent_audit_receipt": authorization_audit,
            "lineage": lineage,
            "lineage_manifest": lineage_manifest,
        },
        "readiness": readiness,
    }
    if lineage_manifest is not None:
        result["files"]["authorization_lineage_manifest"] = metadata(
            candidate / "lineage-input-manifest.json",
            "authorization lineage manifest",
        )
    if candidate_runtime_fingerprint(candidate) != initial_runtime_fingerprint:
        raise PromotionError("candidate runtime changed during validation")
    return result


def stable_identity(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Drop only evidence paths; every file and tree identity remains exact."""
    return snapshot


def validate_executor_record(
    path: Path, snapshot: dict[str, Any], expected_request_id: str
) -> dict[str, Any]:
    value = read_object(path, "SQ8 executor record")
    evidence = value.get("sq8_promotion_evidence")
    if value.get("status") != "ok" or not isinstance(evidence, dict):
        raise PromotionError("SQ8 executor evidence is incomplete")
    if set(evidence) != {
        "schema_version",
        "request_id",
        "manifest_identity",
        "telemetry",
        "telemetry_binding",
        "output_identity",
    }:
        raise PromotionError("SQ8 executor evidence shape differs")
    if evidence.get("schema_version") != "ullm.qwen35_aq4.sq8_promotion_executor.v1":
        raise PromotionError("SQ8 executor evidence schema differs")
    if evidence.get("request_id") != expected_request_id:
        raise PromotionError("SQ8 executor promotion request ID differs")
    manifest = evidence.get("manifest_identity")
    files = snapshot["files"]
    if manifest != {
        "implementation_id": IMPLEMENTATION_ID,
        "execution_profile": EXECUTION_PROFILE,
        "artifact_content_sha256": snapshot["overlay"]["content_sha256"],
        "artifact_manifest_sha256": files["binding"]["sha256"],
        "package_manifest_sha256": files["package_manifest"]["sha256"],
    }:
        raise PromotionError("SQ8 executor manifest identity differs")
    telemetry = evidence.get("telemetry")
    projection = telemetry.get("projection") if isinstance(telemetry, dict) else None
    staging = (
        telemetry.get("diagnostic_host_staging")
        if isinstance(telemetry, dict)
        else None
    )
    if (
        not isinstance(telemetry, dict)
        or telemetry.get("schema_version") != TELEMETRY_SCHEMA
    ):
        raise PromotionError("SQ8 executor telemetry schema differs")
    if (
        not isinstance(projection, dict)
        or projection.get("batch_matvec_count", 0) <= 0
        or projection.get("pair_matvec_count", 0) <= 0
    ):
        raise PromotionError("SQ8 executor lacks batch/pair calls")
    if any(
        projection.get(key) != 0
        for key in ("single_matvec_count", "triple_matvec_count", "fallback_count")
    ):
        raise PromotionError("SQ8 executor used an unexpected projection path")
    if not isinstance(staging, dict) or any(
        staging.get(key) != 0
        for key in ("read_count", "write_count", "read_bytes", "write_bytes")
    ):
        raise PromotionError("SQ8 executor used diagnostic host staging")
    if not sq8_telemetry_binding_valid(
        evidence.get("telemetry_binding"), telemetry, expected_request_id
    ):
        raise PromotionError("SQ8 executor telemetry binding differs")
    output = evidence.get("output_identity")
    if (
        not isinstance(output, dict)
        or output.get("token_count") != 2
        or output.get("token_ids_recorded") is not False
        or not isinstance(output.get("token_ids_sha256"), str)
        or SHA256_RE.fullmatch(output["token_ids_sha256"]) is None
    ):
        raise PromotionError("SQ8 executor output identity is incomplete")
    return value


@dataclass
class LockLease:
    path: Path
    descriptor: int
    device: int
    inode: int
    command_runner: Callable[..., subprocess.CompletedProcess[str]]

    def release(self) -> None:
        identity_error: BaseException | None = None
        try:
            before = os.fstat(self.descriptor)
            current = self.path.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(before.st_mode)
                or (before.st_dev, before.st_ino) != (self.device, self.inode)
                or (current.st_dev, current.st_ino) != (self.device, self.inode)
            ):
                identity_error = PromotionError(
                    "candidate promotion lock inode changed while held"
                )
        except (OSError, PromotionError) as error:
            identity_error = error
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self.descriptor)
        if identity_error is not None:
            if isinstance(identity_error, PromotionError):
                raise identity_error
            raise PromotionError(
                f"candidate promotion lock validation failed: {identity_error}"
            ) from identity_error
        _lock_helper_remove(self.device, self.inode, self.command_runner)

    def evidence(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "device": self.device,
            "inode": self.inode,
            "held": True,
        }


def _lock_helper_result(
    argv: list[str],
    command_runner: Callable[..., subprocess.CompletedProcess[str]],
    label: str,
) -> dict[str, Any]:
    allowed_prefix = ["sudo", "-n", str(LOCK_HELPER)]
    create_argv = argv == [*allowed_prefix, "create"]
    remove_argv = (
        len(argv) == 8
        and argv[:4] == [*allowed_prefix, "remove"]
        and argv[4] == "--device"
        and argv[6] == "--inode"
        and argv[5].isdigit()
        and argv[7].isdigit()
        and int(argv[5]) > 0
        and int(argv[7]) > 0
    )
    if not create_argv and not remove_argv:
        raise PromotionError("candidate lock helper argv is not whitelisted")
    try:
        completed = command_runner(argv, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PromotionError(f"candidate lock helper {label} failed") from error
    if (
        completed.returncode != 0
        or completed.stderr
        or len(completed.stdout.encode("utf-8")) > 16 * 1024
    ):
        raise PromotionError(f"candidate lock helper {label} failed")
    try:
        value = json.loads(completed.stdout)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PromotionError(f"candidate lock helper {label} output differs") from error
    if not isinstance(value, dict):
        raise PromotionError(f"candidate lock helper {label} output differs")
    return value


def _lock_helper_remove(
    device: int,
    inode: int,
    command_runner: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    argv = [
        "sudo",
        "-n",
        str(LOCK_HELPER),
        "remove",
        "--device",
        str(device),
        "--inode",
        str(inode),
    ]
    value = _lock_helper_result(argv, command_runner, "remove")
    if value != {
        "status": "removed",
        "device": device,
        "inode": inode,
        "runtime_directory_removed": True,
    }:
        raise PromotionError("candidate lock helper remove output differs")


def acquire_lock(
    path: Path = LOCK_PATH,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> LockLease:
    if path != LOCK_PATH:
        raise PromotionError("candidate promotion lock path differs")
    if command_runner is None:
        command_runner = _run
    created = _lock_helper_result(
        ["sudo", "-n", str(LOCK_HELPER), "create"], command_runner, "create"
    )
    lock = created.get("lock")
    directory = created.get("runtime_directory")
    if (
        set(created)
        != {"status", "runtime_directory_created", "runtime_directory", "lock"}
        or created.get("status") != "created"
        or created.get("runtime_directory_created") is not True
        or not isinstance(lock, dict)
        or not isinstance(directory, dict)
        or lock.get("path") != str(path)
        or lock.get("mode") != "0600"
        or lock.get("uid") != LOCK_UID
        or lock.get("gid") != LOCK_GID
        or lock.get("nlink") != 1
        or directory.get("path") != str(path.parent)
        or directory.get("mode") != "0750"
        or directory.get("uid") != LOCK_UID
        or directory.get("gid") != LOCK_GID
    ):
        raise PromotionError("candidate lock helper create output differs")
    device = lock.get("device")
    inode = lock.get("inode")
    if (
        not isinstance(device, int)
        or device <= 0
        or not isinstance(inode, int)
        or inode <= 0
    ):
        raise PromotionError("candidate lock helper inode differs")
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        value = os.fstat(descriptor)
        if (
            not stat.S_ISREG(value.st_mode)
            or value.st_nlink != 1
            or stat.S_IMODE(value.st_mode) != 0o600
            or value.st_uid != LOCK_UID
            or value.st_gid != LOCK_GID
            or (value.st_dev, value.st_ino) != (device, inode)
        ):
            raise PromotionError("candidate promotion lock substrate differs")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = path.stat(follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (device, inode):
            raise PromotionError("candidate promotion lock inode changed")
        return LockLease(path, descriptor, device, inode, command_runner)
    except BaseException as error:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            _lock_helper_remove(device, inode, command_runner)
        except PromotionError as cleanup_error:
            raise PromotionError(
                f"candidate lock acquire and cleanup failed: {cleanup_error}"
            ) from error
        if isinstance(error, PromotionError):
            raise
        raise PromotionError(
            f"candidate promotion lock acquire failed: {error}"
        ) from error


def _run(
    argv: list[str], *, env: dict[str, str] | None = None, timeout: float = 30.0
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv, check=False, capture_output=True, text=True, env=env, timeout=timeout
    )


def _bounded_bytes(path: Path, limit: int, label: str) -> bytes:
    with path.open("rb") as source:
        raw = source.read(limit + 1)
    if len(raw) > limit:
        raise PromotionError(f"{label} exceeds its bound")
    return raw


def _cgroup_pids(control_group: str) -> list[int]:
    if (
        not control_group.startswith("/system.slice/")
        or ".." in Path(control_group).parts
    ):
        raise PromotionError("service control group differs")
    path = Path("/sys/fs/cgroup") / control_group.lstrip("/") / "cgroup.procs"
    try:
        values = (
            _bounded_bytes(path, 64 * 1024, "service cgroup process list")
            .decode("ascii")
            .splitlines()
        )
    except FileNotFoundError as error:
        raise TransientRestoreError(
            "service cgroup process list is not ready"
        ) from error
    except (OSError, UnicodeError) as error:
        raise PromotionError("service cgroup process list is unavailable") from error
    try:
        pids = sorted({int(value) for value in values if value})
    except ValueError as error:
        raise PromotionError("service cgroup process list differs") from error
    if any(pid <= 0 for pid in pids):
        raise PromotionError("service cgroup process list differs")
    return pids


def _worker_pids(candidates: list[int] | None = None) -> list[int]:
    names = (
        [str(pid) for pid in candidates]
        if candidates is not None
        else [entry.name for entry in Path("/proc").iterdir() if entry.name.isdigit()]
    )
    result: list[int] = []
    for name in names:
        try:
            argv0 = _bounded_bytes(
                Path("/proc") / name / "cmdline", 64 * 1024, "worker command line"
            ).split(b"\0", 1)[0]
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as error:
            raise PromotionError("worker process scan failed") from error
        if Path(os.fsdecode(argv0)).name == "ullm-aq4-worker":
            result.append(int(name))
    return sorted(result)


def _lock_holder_pids(path: Path) -> list[int]:
    try:
        value = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return []
    if path.is_symlink() or not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise PromotionError("production lock topology differs")
    try:
        lines = (
            _bounded_bytes(Path("/proc/locks"), 4 * 1024 * 1024, "kernel lock table")
            .decode("ascii")
            .splitlines()
        )
    except (OSError, UnicodeError) as error:
        raise PromotionError("kernel lock table is unavailable") from error
    device = f"{os.major(value.st_dev):02x}:{os.minor(value.st_dev):x}:{value.st_ino}"
    holders: set[int] = set()
    for line in lines:
        fields = line.split()
        if len(fields) >= 6 and fields[1] == "FLOCK" and fields[5] == device:
            try:
                pid = int(fields[4])
            except ValueError as error:
                raise PromotionError("kernel lock owner schema differs") from error
            if pid <= 0:
                raise PromotionError("kernel lock owner schema differs")
            holders.add(pid)
    return sorted(holders)


def _inspect_object(
    completed: subprocess.CompletedProcess[str], label: str
) -> dict[str, Any]:
    if completed.returncode != 0 or completed.stderr:
        raise PromotionError(f"{label} failed")
    try:
        raw = completed.stdout.encode("utf-8")
    except UnicodeError as error:
        raise PromotionError(f"{label} output encoding differs") from error
    if len(raw) > DOCKER_INSPECT_MAX_BYTES:
        raise PromotionError(f"{label} output exceeds its bound")
    try:
        value = json.loads(completed.stdout)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise PromotionError(f"{label} output differs") from error
    if not isinstance(value, dict):
        raise PromotionError(f"{label} output differs")
    return value


def _ready(
    readiness: dict[str, Any],
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = _run,
    bridge_exists: Callable[[str], bool] = lambda name: (
        Path("/sys/class/net") / name
    ).is_dir(),
) -> bool:
    """Probe readiness only through the exact Gate-bound Docker identity."""

    contract = validate_readiness_contract(readiness)
    container = contract["container"]
    network = contract["network"]
    endpoint = contract["endpoint"]
    container_format = (
        '{"id":{{json .Id}},"name":{{json .Name}},'
        '"image_id":{{json .Image}},"config_image":{{json .Config.Image}},'
        '"networks":{{json .NetworkSettings.Networks}}}'
    )
    network_format = (
        '{"id":{{json .Id}},"name":{{json .Name}},'
        '"driver":{{json .Driver}},"options":{{json .Options}},'
        '"containers":{{json .Containers}}}'
    )
    try:
        container_result = command_runner(
            [
                "docker",
                "inspect",
                "--type",
                "container",
                "--format",
                container_format,
                container["id"],
            ],
            timeout=endpoint["timeout_seconds"],
        )
        network_result = command_runner(
            [
                "docker",
                "network",
                "inspect",
                "--format",
                network_format,
                network["id"],
            ],
            timeout=endpoint["timeout_seconds"],
        )
    except subprocess.TimeoutExpired as error:
        raise PromotionError(
            "Docker readiness identity inspection timed out"
        ) from error
    except OSError as error:
        raise PromotionError("Docker readiness identity inspection failed") from error

    observed_container = _inspect_object(container_result, "container inspect")
    if set(observed_container) != {
        "id",
        "name",
        "image_id",
        "config_image",
        "networks",
    }:
        raise PromotionError("readiness container inspect shape differs")
    observed_name = observed_container.get("name")
    if isinstance(observed_name, str):
        observed_name = observed_name.removeprefix("/")
    networks = observed_container.get("networks")
    if (
        observed_container.get("id") != container["id"]
        or observed_name != container["name"]
        or observed_container.get("image_id") != container["image_id"]
        or observed_container.get("config_image") != container["config_image"]
        or not isinstance(networks, dict)
        or set(networks) != {network["name"]}
    ):
        raise PromotionError("readiness container identity differs from Gate")
    attachment = networks[network["name"]]
    if not isinstance(attachment, dict) or attachment.get("NetworkID") != network["id"]:
        raise PromotionError("readiness container network attachment differs from Gate")

    observed_network = _inspect_object(network_result, "network inspect")
    if set(observed_network) != {"id", "name", "driver", "options", "containers"}:
        raise PromotionError("readiness network inspect shape differs")
    options = observed_network.get("options")
    containers = observed_network.get("containers")
    configured_bridge = (
        options.get("com.docker.network.bridge.name")
        if isinstance(options, dict)
        else None
    )
    network_member = (
        containers.get(container["id"]) if isinstance(containers, dict) else None
    )
    if (
        observed_network.get("id") != network["id"]
        or observed_network.get("name") != network["name"]
        or observed_network.get("driver") != network["driver"]
        or not isinstance(options, dict)
        or configured_bridge not in {None, network["bridge_interface"]}
        or not bridge_exists(network["bridge_interface"])
        or not isinstance(network_member, dict)
        or network_member.get("Name") not in {None, container["name"]}
    ):
        raise PromotionError("readiness network identity differs from Gate")

    expected_body = endpoint["expected_body"]
    curl_command = [
        "docker",
        "exec",
        container["id"],
        "curl",
        "--silent",
        "--show-error",
        "--request",
        "GET",
        "--header",
        "Accept: application/json",
        "--connect-timeout",
        str(endpoint["timeout_seconds"]),
        "--max-time",
        str(endpoint["timeout_seconds"]),
        "--max-filesize",
        str(len(expected_body.encode("ascii"))),
        "--output",
        "-",
        "--write-out",
        "\n%{http_code}",
        endpoint["url"],
    ]
    try:
        completed = command_runner(curl_command, timeout=endpoint["timeout_seconds"])
    except (subprocess.TimeoutExpired, OSError):
        return False
    expected = f"{expected_body}\n{endpoint['expected_status']}"
    return (
        completed.returncode == 0
        and completed.stderr == ""
        and len(completed.stdout.encode("utf-8")) <= len(expected.encode("utf-8"))
        and completed.stdout == expected
    )


def _amd_owner_snapshot(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PromotionError("AMD GPU owner JSON differs") from error
    if not isinstance(value, list) or len(value) != 1:
        raise PromotionError("AMD GPU owner root differs")
    root = value[0]
    if (
        not isinstance(root, dict)
        or set(root) != {"gpu", "process_list"}
        or root.get("gpu") != AMD_SMI_INDEX
    ):
        raise PromotionError("AMD GPU owner identity differs")
    processes = root["process_list"]
    if processes == [{"process_info": "No running processes detected"}]:
        return {
            "owners": [],
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "raw_bytes": len(raw),
        }
    if not isinstance(processes, list) or not processes:
        raise PromotionError("AMD GPU owner process list differs")
    owners: list[int] = []
    expected = {"name", "pid", "mem_usage", "cu_occupancy", "evicted_time"}
    for process in processes:
        if not isinstance(process, dict) or set(process) != {"process_info"}:
            raise PromotionError("AMD GPU owner entry differs")
        info = process["process_info"]
        if (
            not isinstance(info, dict)
            or set(info) != expected
            or not isinstance(info.get("pid"), int)
            or info["pid"] <= 0
        ):
            raise PromotionError("AMD GPU owner process information differs")
        owners.append(info["pid"])
    if len(owners) != len(set(owners)):
        raise PromotionError("AMD GPU owner PID is duplicated")
    return {
        "owners": sorted(owners),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_bytes": len(raw),
    }


def _kfd_owner_snapshot() -> dict[str, Any]:
    try:
        root_before = KFD_PROC_ROOT.stat()
        process_names = sorted(os.listdir(KFD_PROC_ROOT))
    except OSError as error:
        raise PromotionError("KFD owner root is unavailable") from error
    if not stat.S_ISDIR(root_before.st_mode) or any(
        not name.isdigit() or int(name) <= 0 for name in process_names
    ):
        raise PromotionError("KFD owner root schema differs")
    owners: set[int] = set()
    sources: list[dict[str, Any]] = []
    for process_name in process_names:
        pid = int(process_name)
        queues = KFD_PROC_ROOT / process_name / "queues"
        try:
            queues_before = queues.stat()
            queue_names = sorted(os.listdir(queues))
        except FileNotFoundError as error:
            raise PromotionError("KFD owner source changed during scan") from error
        except OSError as error:
            raise PromotionError("KFD queue source is unavailable") from error
        if not stat.S_ISDIR(queues_before.st_mode) or any(
            not name.isdigit() for name in queue_names
        ):
            raise PromotionError("KFD queue source schema differs")
        for queue_name in queue_names:
            path = queues / queue_name / "gpuid"
            try:
                before = path.stat()
                raw = _bounded_bytes(path, 64, "KFD GPU ID")
                after = path.stat()
            except FileNotFoundError as error:
                raise PromotionError("KFD owner source changed during scan") from error
            if not stat.S_ISREG(before.st_mode) or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise PromotionError("KFD GPU ID source changed during scan")
            payload = raw[:-1] if raw.endswith(b"\n") else raw
            if not payload.isdigit() or payload.startswith(b"0"):
                raise PromotionError("KFD GPU ID schema differs")
            gpuid = int(payload)
            sources.append(
                {
                    "pid": pid,
                    "queue": int(queue_name),
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "raw_bytes": len(raw),
                }
            )
            if gpuid == KFD_ID:
                owners.add(pid)
        try:
            if sorted(os.listdir(queues)) != queue_names or (
                queues.stat().st_dev,
                queues.stat().st_ino,
            ) != (queues_before.st_dev, queues_before.st_ino):
                raise PromotionError("KFD queue source changed during scan")
        except FileNotFoundError as error:
            raise PromotionError("KFD owner source changed during scan") from error
    try:
        root_after = KFD_PROC_ROOT.stat()
        final_names = sorted(os.listdir(KFD_PROC_ROOT))
    except OSError as error:
        raise PromotionError("KFD owner root changed during scan") from error
    if final_names != process_names or (root_after.st_dev, root_after.st_ino) != (
        root_before.st_dev,
        root_before.st_ino,
    ):
        raise PromotionError("KFD owner root changed during scan")
    return {
        "owners": sorted(owners),
        "enumerated_pids": [int(name) for name in process_names],
        "sources": sources,
        "root": {
            "path": str(KFD_PROC_ROOT),
            "device": root_before.st_dev,
            "inode": root_before.st_ino,
        },
    }


def default_service_snapshot(
    readiness: dict[str, Any],
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = _run,
) -> dict[str, Any]:
    fields = "ActiveState,SubState,MainPID,NRestarts,ControlGroup"
    result = _run(["systemctl", "show", SERVICE, f"--property={fields}"], timeout=5)
    if result.returncode != 0:
        raise PromotionError("service snapshot failed")
    values = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    if set(values) != {
        "ActiveState",
        "SubState",
        "MainPID",
        "NRestarts",
        "ControlGroup",
    }:
        raise PromotionError("service snapshot fields differ")
    try:
        main_pid = int(values.get("MainPID", "0"))
        nrestarts = int(values.get("NRestarts", "0"))
    except ValueError as error:
        raise PromotionError("service snapshot numeric fields differ") from error
    if main_pid < 0 or nrestarts < 0:
        raise PromotionError("service snapshot numeric fields differ")
    active = values.get("ActiveState") == "active"
    running = values.get("SubState") == "running"
    control_group = values.get("ControlGroup", "")
    cgroup_pids = _cgroup_pids(control_group) if active else []
    workers = _worker_pids(cgroup_pids) if active else []
    if active and (main_pid <= 0 or len(workers) != 1 or main_pid not in cgroup_pids):
        raise TransientRestoreError("active service process topology is not ready")
    lock_holders = _lock_holder_pids(PRODUCTION_LOCK_PATH)
    lock_owned = bool(lock_holders) and set(lock_holders).issubset(cgroup_pids)
    return {
        "active": active,
        "running": running,
        "main_pid": main_pid,
        "nrestarts": nrestarts,
        "control_group": control_group,
        "worker_pid": workers[0] if len(workers) == 1 else 0,
        "cgroup_pids": cgroup_pids,
        "healthy": active and running and _ready(readiness, command_runner),
        "lock_owned": lock_owned,
        "lock_holders": lock_holders,
    }


def default_owner_snapshot() -> dict[str, Any]:
    completed = subprocess.run(
        [str(AMD_SMI), "process", "--gpu", str(AMD_SMI_INDEX), "--general", "--json"],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
    )
    if completed.returncode != 0 or completed.stderr:
        raise PromotionError("AMD GPU owner probe failed")
    try:
        parsed = _amd_owner_snapshot(completed.stdout)
        kfd = _kfd_owner_snapshot()
    except PromotionError:
        raise
    return {
        "worker_pids": _worker_pids(),
        "amd_pids": parsed["owners"],
        "kfd_pids": kfd["owners"],
        "amd_diagnostic": {
            key: value for key, value in parsed.items() if key != "owners"
        },
        "kfd_source": kfd,
    }


def _bytes_value(value: Any) -> int | None:
    if isinstance(value, dict):
        number = value.get("value")
        unit = str(value.get("unit", "B")).lower()
    else:
        number = value
        unit = "b"
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        return None
    factors = {"b": 1, "kb": 1000, "kib": 1024, "mb": 1000**2, "mib": 1024**2, "gb": 1000**3, "gib": 1024**3}
    factor = factors.get(unit)
    if factor is None or number < 0 or int(number) != number:
        return None
    return int(number) * factor


def default_vram_headroom_bytes() -> int:
    """Read free VRAM from amd-smi; absence or ambiguity is fail-closed."""

    completed = _run(
        [str(AMD_SMI), "metric", "--mem-usage", "--gpu", str(AMD_SMI_INDEX), "--json"],
        timeout=5,
    )
    if completed.returncode != 0 or completed.stderr:
        raise PromotionError("AMD VRAM headroom probe failed")
    try:
        value = json.loads(completed.stdout)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PromotionError("AMD VRAM headroom JSON differs") from error
    free: list[int] = []
    total: list[int] = []
    used: list[int] = []

    def visit(node: Any, context: str = "") -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                lowered = f"{context}_{key}".lower().replace("-", "_")
                numeric = _bytes_value(item)
                if numeric is not None:
                    if "free" in lowered and ("mem" in lowered or "vram" in lowered):
                        free.append(numeric)
                    elif "total" in lowered and ("mem" in lowered or "vram" in lowered):
                        total.append(numeric)
                    elif "used" in lowered and ("mem" in lowered or "vram" in lowered):
                        used.append(numeric)
                visit(item, lowered)
        elif isinstance(node, list):
            for item in node:
                visit(item, context)

    visit(value)
    candidates = set(free)
    if not candidates and total and used:
        candidates = {total_value - used_value for total_value in total for used_value in used if total_value >= used_value}
    if len(candidates) != 1 or next(iter(candidates), 0) < 1:
        raise PromotionError("AMD VRAM headroom observation is ambiguous")
    return next(iter(candidates))


@dataclass
class CaptureStream:
    byte_count: int
    sha256: str
    prefix: bytes
    prefix_truncated: bool
    parse_buffer: bytes | None
    parse_buffer_truncated: bool
    complete: bool
    stream_error: str | None


@dataclass
class CaptureProcessResult:
    argv: list[str]
    returncode: int
    stdout: CaptureStream
    stderr: CaptureStream
    timed_out: bool
    timeout_seconds: float | None


class _CaptureStreamCollector:
    def __init__(self, *, retain_parse_buffer: bool) -> None:
        self._digest = hashlib.sha256()
        self._byte_count = 0
        self._prefix = bytearray()
        self._parse = bytearray() if retain_parse_buffer else None
        self._parse_truncated = False
        self._complete = False
        self._stream_error: str | None = None

    def feed(self, chunk: bytes) -> None:
        self._digest.update(chunk)
        self._byte_count += len(chunk)
        remaining = CAPTURE_DIAGNOSTIC_MAX_BYTES - len(self._prefix)
        if remaining > 0:
            self._prefix.extend(chunk[:remaining])
        if self._parse is not None:
            parse_remaining = CAPTURE_ENVELOPE_MAX_BYTES - len(self._parse)
            if parse_remaining > 0:
                self._parse.extend(chunk[:parse_remaining])
            if len(chunk) > parse_remaining:
                self._parse_truncated = True

    def drain(self, stream: Any) -> None:
        try:
            while True:
                chunk = stream.read(CAPTURE_READ_CHUNK_BYTES)
                if not chunk:
                    break
                if not isinstance(chunk, bytes):
                    chunk = bytes(chunk)
                self.feed(chunk)
        except BaseException as error:
            self._stream_error = type(error).__name__
        finally:
            self._complete = True

    def finish(self) -> None:
        self._complete = True

    def result(self, thread_alive: bool) -> CaptureStream:
        complete = self._complete and not thread_alive and self._stream_error is None
        return CaptureStream(
            byte_count=self._byte_count,
            sha256=self._digest.hexdigest(),
            prefix=bytes(self._prefix),
            prefix_truncated=self._byte_count > len(self._prefix),
            parse_buffer=bytes(self._parse) if self._parse is not None else None,
            parse_buffer_truncated=self._parse_truncated,
            complete=complete,
            stream_error=self._stream_error,
        )


def default_capture(
    argv: list[str], environment: dict[str, str]
) -> CaptureProcessResult:
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    assert proc.stdout is not None and proc.stderr is not None
    stdout_collector = _CaptureStreamCollector(retain_parse_buffer=True)
    stderr_collector = _CaptureStreamCollector(retain_parse_buffer=False)
    stdout_thread = threading.Thread(
        target=stdout_collector.drain,
        args=(proc.stdout,),
        name="sq8-capture-stdout",
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=stderr_collector.drain,
        args=(proc.stderr,),
        name="sq8-capture-stderr",
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    timed_out = False
    try:
        proc.wait(timeout=CAPTURE_SUBPROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait()
    finally:
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        for stream, thread in (
            (proc.stdout, stdout_thread),
            (proc.stderr, stderr_thread),
        ):
            if thread.is_alive():
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
                thread.join(timeout=1)
    return CaptureProcessResult(
        argv=list(argv),
        returncode=int(proc.returncode),
        stdout=stdout_collector.result(stdout_thread.is_alive()),
        stderr=stderr_collector.result(stderr_thread.is_alive()),
        timed_out=timed_out,
        timeout_seconds=CAPTURE_SUBPROCESS_TIMEOUT_SECONDS if timed_out else None,
    )


def _diagnostic_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8", errors="surrogatepass")
    return repr(value).encode("utf-8", errors="backslashreplace")


def _redact_diagnostic(raw: bytes) -> str:
    """Return bounded, display-only diagnostics without credential-bearing lines."""

    text = raw.decode("utf-8", errors="replace")
    sensitive = re.compile(
        r"(?i)(?:password|passwd|secret|api[_-]?key|authorization|"
        r"access[_-]?token|refresh[_-]?token|bearer\s+|"
        r"(?<![A-Za-z0-9])token\s*[:=]|https?://[^/\s:@]+:[^@\s/]+@)"
    )
    redacted: list[str] = []
    for line in text.splitlines(keepends=True):
        ending = "\n" if line.endswith("\n") else ""
        if sensitive.search(line):
            redacted.append("<redacted sensitive diagnostic line>" + ending)
        else:
            redacted.append(line)
    return "".join(redacted)


def _bounded_diagnostic(value: Any) -> dict[str, Any]:
    raw = _diagnostic_bytes(value)
    captured = raw[:CAPTURE_DIAGNOSTIC_MAX_BYTES]
    return _bounded_diagnostic_parts(
        byte_count=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        captured=captured,
        prefix_truncated=len(raw) > len(captured),
    )


def _bounded_diagnostic_parts(
    *, byte_count: int, sha256: str, captured: bytes, prefix_truncated: bool
) -> dict[str, Any]:
    redacted = _redact_diagnostic(captured)
    source = {
        "byte_count": byte_count,
        "sha256": sha256,
        "captured_prefix_bytes": len(captured),
        "prefix_truncated": prefix_truncated,
    }

    def document(text: str, truncated: bool, serialized_bytes: int) -> dict[str, Any]:
        return {
            "source": source,
            "display": {
                "serialized_byte_limit": CAPTURE_DIAGNOSTIC_MAX_BYTES,
                "serialized_byte_count": serialized_bytes,
                "truncated_after_redaction": truncated,
                "text": text,
            },
        }

    def serialized_size(result: dict[str, Any]) -> int:
        return len(
            json.dumps(
                result,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        )

    # Reserve the maximum five-digit count while choosing the largest prefix.
    if (
        serialized_size(document(redacted, False, CAPTURE_DIAGNOSTIC_MAX_BYTES))
        <= CAPTURE_DIAGNOSTIC_MAX_BYTES
    ):
        display = redacted
        display_truncated = False
    else:
        low = 0
        high = len(redacted)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = document(redacted[:middle], True, CAPTURE_DIAGNOSTIC_MAX_BYTES)
            if serialized_size(candidate) <= CAPTURE_DIAGNOSTIC_MAX_BYTES:
                low = middle
            else:
                high = middle - 1
        display = redacted[:low]
        display_truncated = True

    result = document(display, display_truncated, 0)
    for _ in range(3):
        size = serialized_size(result)
        result["display"]["serialized_byte_count"] = size
    if serialized_size(result) > CAPTURE_DIAGNOSTIC_MAX_BYTES:
        raise PromotionError("capture diagnostic serialization exceeds its bound")
    return result


def _stream_diagnostic(stream: CaptureStream) -> dict[str, Any]:
    return _bounded_diagnostic_parts(
        byte_count=stream.byte_count,
        sha256=stream.sha256,
        captured=stream.prefix,
        prefix_truncated=stream.prefix_truncated,
    )


def capture_failure_diagnostic(
    *,
    stage: str,
    returncode: int | None,
    stdout: Any,
    stderr: Any,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    signal_value: dict[str, Any] | None = None
    if isinstance(returncode, int) and returncode < 0:
        number = -returncode
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = None
        signal_value = {"number": number, "name": name}
    stdout_diagnostic = (
        _stream_diagnostic(stdout)
        if isinstance(stdout, CaptureStream)
        else _bounded_diagnostic(stdout)
    )
    stderr_diagnostic = (
        _stream_diagnostic(stderr)
        if isinstance(stderr, CaptureStream)
        else _bounded_diagnostic(stderr)
    )
    result = {
        "schema_version": "ullm.qwen35_aq4.sq8_overlay_capture_failure.v1",
        "stage": stage,
        "returncode": returncode,
        "signal": signal_value,
        "timeout_seconds": timeout_seconds,
        "stdout": stdout_diagnostic,
        "stderr": stderr_diagnostic,
    }
    if isinstance(stdout, CaptureStream) and isinstance(stderr, CaptureStream):
        result["outer_collection"] = {
            "stdout": {
                "complete": stdout.complete,
                "stream_error": stdout.stream_error,
                "parse_buffer_truncated": stdout.parse_buffer_truncated,
            },
            "stderr": {
                "complete": stderr.complete,
                "stream_error": stderr.stream_error,
            },
        }
    return result


class _DuplicateJsonKey(ValueError):
    pass


def _unique_json_object(raw: bytes) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateJsonKey(key)
            result[key] = value
        return result

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("capture error envelope is not an object")
    return value


def _capture_terminal_contract_valid(
    *,
    stage: str,
    timed_out: bool,
    returncode: int | None,
    worker_signal: int | None,
) -> bool:
    if returncode is None:
        if worker_signal is not None:
            return False
    elif returncode < 0:
        if worker_signal != -returncode:
            return False
    elif worker_signal is not None:
        return False

    if timed_out:
        return (
            stage in {"request", "shutdown"}
            and isinstance(returncode, int)
            and returncode < 0
            and worker_signal == -returncode
        )
    if stage == "capture":
        return returncode is None and worker_signal is None
    if stage == "shutdown":
        return False
    if stage == "worker_exit":
        return isinstance(returncode, int) and returncode != 0
    if stage in {
        "audit_missing",
        "resource_observation",
        "package_validation",
        "telemetry_validation",
        "validation",
    }:
        return returncode == 0
    if stage in {"request", "cleanup", "worker"}:
        return isinstance(returncode, int)
    return False


def _capture_error_envelope(
    stream: CaptureStream, expected_request_id: str
) -> dict[str, Any]:
    invalid: dict[str, Any] = {"validation": "invalid", "reason": None}
    if PROMOTION_REQUEST_ID_RE.fullmatch(expected_request_id) is None:
        invalid["reason"] = "expected_request_id_invalid"
        return invalid
    if not stream.complete or stream.stream_error is not None:
        invalid["reason"] = "outer_stdout_collection_incomplete"
        return invalid
    if stream.parse_buffer_truncated or stream.parse_buffer is None:
        invalid["reason"] = "capture_error_envelope_truncated"
        return invalid
    try:
        value = _unique_json_object(stream.parse_buffer)
    except _DuplicateJsonKey:
        invalid["reason"] = "capture_error_envelope_duplicate_key"
        return invalid
    except (UnicodeError, json.JSONDecodeError, ValueError):
        invalid["reason"] = "capture_error_envelope_invalid_json"
        return invalid
    expected = {
        "schema_version",
        "status",
        "stage",
        "reason",
        "timed_out",
        "worker_returncode",
        "worker_signal",
        "worker_stderr",
        "observed_sq8_promotion_telemetry",
        "observed_sq8_promotion_telemetry_binding",
        "worker_terminal",
    }
    if set(value) != expected:
        invalid["reason"] = "capture_error_envelope_keys_differ"
        return invalid
    stage = value.get("stage")
    reason = value.get("reason")
    timed_out = value.get("timed_out")
    returncode = value.get("worker_returncode")
    worker_signal = value.get("worker_signal")
    if (
        value.get("schema_version") != CAPTURE_ERROR_SCHEMA
        or value.get("status") != "failed"
        or not isinstance(stage, str)
        or stage not in CAPTURE_ERROR_STAGES
        or not isinstance(reason, str)
        or not reason
        or len(reason.encode("utf-8")) > 4096
        or type(timed_out) is not bool
        or not (
            returncode is None
            or (type(returncode) is int and -(2**31) <= returncode < 2**31)
        )
        or not (
            worker_signal is None
            or (type(worker_signal) is int and 0 < worker_signal < 128)
        )
    ):
        invalid["reason"] = "capture_error_envelope_type_or_value_differs"
        return invalid
    if not _capture_terminal_contract_valid(
        stage=stage,
        timed_out=timed_out,
        returncode=returncode,
        worker_signal=worker_signal,
    ):
        invalid["reason"] = "capture_error_envelope_stage_terminal_mismatch"
        return invalid
    worker = value.get("worker_stderr")
    worker_expected = {
        "schema_version",
        "byte_count",
        "sha256",
        "preview_text",
        "captured_bytes",
        "truncated",
        "utf8_replacement",
        "redacted_lines",
        "complete",
        "stream_error",
    }
    if not isinstance(worker, dict) or set(worker) != worker_expected:
        invalid["reason"] = "worker_stderr_keys_differ"
        return invalid
    byte_count = worker.get("byte_count")
    captured_bytes = worker.get("captured_bytes")
    redacted_lines = worker.get("redacted_lines")
    preview_text = worker.get("preview_text")
    complete = worker.get("complete")
    stream_error = worker.get("stream_error")
    if (
        worker.get("schema_version") != WORKER_STDERR_SCHEMA
        or type(byte_count) is not int
        or not 0 <= byte_count <= 9_007_199_254_740_991
        or not isinstance(worker.get("sha256"), str)
        or SHA256_RE.fullmatch(worker["sha256"]) is None
        or not isinstance(preview_text, str)
        or type(captured_bytes) is not int
        or not 0 <= captured_bytes <= CAPTURE_DIAGNOSTIC_MAX_BYTES
        or len(preview_text.encode("utf-8")) != captured_bytes
        or type(worker.get("truncated")) is not bool
        or type(worker.get("utf8_replacement")) is not bool
        or type(redacted_lines) is not int
        or redacted_lines < 0
        or type(complete) is not bool
        or not (
            stream_error is None
            or (
                isinstance(stream_error, str)
                and 0 < len(stream_error.encode("utf-8")) <= 1024
            )
        )
    ):
        invalid["reason"] = "worker_stderr_type_or_value_differs"
        return invalid
    normalized_worker = {
        key: worker[key]
        for key in (
            "schema_version",
            "byte_count",
            "sha256",
            "captured_bytes",
            "truncated",
            "utf8_replacement",
            "redacted_lines",
            "complete",
            "stream_error",
        )
    }
    if isinstance(stream_error, str):
        normalized_worker["stream_error"] = _redact_diagnostic(
            stream_error.encode("utf-8")
        )
    normalized_worker["preview"] = _bounded_diagnostic(preview_text)
    result = {
        "validation": "valid",
        "schema_version": value["schema_version"],
        "status": value["status"],
        "stage": stage,
        "reason": _redact_diagnostic(reason.encode("utf-8")),
        "timed_out": timed_out,
        "worker_returncode": returncode,
        "worker_signal": worker_signal,
        "worker_stderr": normalized_worker,
    }
    if complete is not True or stream_error is not None:
        result["validation"] = "invalid"
        result["validation_reason"] = "worker_stderr_incomplete"
    observed_telemetry = value.get("observed_sq8_promotion_telemetry")
    if observed_telemetry is not None:
        projection = observed_telemetry.get("projection") if isinstance(observed_telemetry, dict) else None
        staging = (
            observed_telemetry.get("diagnostic_host_staging")
            if isinstance(observed_telemetry, dict)
            else None
        )
        projection_keys = {
            "single_matvec_count",
            "batch_matvec_count",
            "pair_matvec_count",
            "triple_matvec_count",
            "fallback_count",
        }
        staging_keys = {"read_count", "write_count", "read_bytes", "write_bytes"}
        if (
            not isinstance(observed_telemetry, dict)
            or set(observed_telemetry) != {
                "schema_version",
                "projection",
                "diagnostic_host_staging",
            }
            or observed_telemetry.get("schema_version") != TELEMETRY_SCHEMA
            or not isinstance(projection, dict)
            or set(projection) != projection_keys
            or any(type(projection[key]) is not int or projection[key] < 0 for key in projection_keys)
            or not isinstance(staging, dict)
            or set(staging) != staging_keys
            or any(type(staging[key]) is not int or staging[key] < 0 for key in staging_keys)
        ):
            invalid["reason"] = "observed_sq8_promotion_telemetry_invalid"
            return invalid
    observed_binding = value.get("observed_sq8_promotion_telemetry_binding")
    if (observed_telemetry is None) != (observed_binding is None):
        invalid["reason"] = "observed_sq8_promotion_telemetry_binding_missing"
        return invalid
    if observed_telemetry is not None and not sq8_telemetry_binding_valid(
        observed_binding, observed_telemetry, expected_request_id
    ):
        invalid["reason"] = "observed_sq8_promotion_telemetry_binding_invalid"
        return invalid
    terminal = value.get("worker_terminal")
    if terminal is not None:
        if (
            not isinstance(terminal, dict)
            or set(terminal) != {
                "schema_version",
                "event",
                "request_id",
                "request_id_matches",
                "operation_execution_audit_observed",
                "request_execution_audit_observed",
            }
            or terminal.get("schema_version")
            != "ullm.aq4_resident_worker_terminal.v1"
            or terminal.get("event") != "request_released"
            or not isinstance(terminal.get("request_id"), str)
            or PROMOTION_REQUEST_ID_RE.fullmatch(terminal["request_id"]) is None
            or terminal["request_id"] != expected_request_id
            or any(
                type(terminal.get(key)) is not bool
                for key in (
                    "request_id_matches",
                    "operation_execution_audit_observed",
                    "request_execution_audit_observed",
                )
            )
        ):
            invalid["reason"] = "worker_terminal_invalid"
            return invalid
        if terminal["request_id_matches"] is not True:
            invalid["reason"] = "worker_terminal_request_id_mismatch"
            return invalid
    if observed_telemetry is not None and terminal is None:
        invalid["reason"] = "observed_sq8_promotion_worker_terminal_missing"
        return invalid
    if stage == "telemetry_validation" and (
        observed_telemetry is None
        or terminal is None
        or terminal["request_id_matches"] is not True
        or terminal["operation_execution_audit_observed"] is not True
        or terminal["request_execution_audit_observed"] is not True
    ):
        invalid["reason"] = "telemetry_validation_evidence_missing"
        return invalid
    result["observed_sq8_promotion_telemetry"] = observed_telemetry
    result["observed_sq8_promotion_telemetry_binding"] = observed_binding
    result["worker_terminal"] = terminal
    return result


def _coerce_capture_result(value: Any) -> CaptureProcessResult:
    if isinstance(value, CaptureProcessResult):
        return value
    if not isinstance(value, subprocess.CompletedProcess):
        raise PromotionError("capture dependency result type differs")
    stdout_collector = _CaptureStreamCollector(retain_parse_buffer=True)
    stderr_collector = _CaptureStreamCollector(retain_parse_buffer=False)
    stdout_collector.feed(_diagnostic_bytes(value.stdout))
    stderr_collector.feed(_diagnostic_bytes(value.stderr))
    stdout_collector.finish()
    stderr_collector.finish()
    raw_args = value.args
    argv = [raw_args] if isinstance(raw_args, str) else [str(item) for item in raw_args]
    return CaptureProcessResult(
        argv=argv,
        returncode=int(value.returncode),
        stdout=stdout_collector.result(False),
        stderr=stderr_collector.result(False),
        timed_out=False,
        timeout_seconds=None,
    )


@dataclass
class Dependencies:
    service_snapshot: Callable[[dict[str, Any]], dict[str, Any]]
    owner_snapshot: Callable[[], dict[str, Any]]
    stop_service: Callable[[], None]
    start_service: Callable[[], None]
    acquire_lock: Callable[[], Any]
    capture: Callable[[list[str], dict[str, str]], subprocess.CompletedProcess[Any]]
    vram_headroom_bytes: Callable[[], int] = default_vram_headroom_bytes
    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep


def default_dependencies() -> Dependencies:
    def control(action: str) -> None:
        result = _run(["sudo", "-n", "systemctl", action, SERVICE], timeout=10)
        if result.returncode != 0 or result.stdout or result.stderr:
            raise PromotionError(f"service {action} failed")

    return Dependencies(
        service_snapshot=default_service_snapshot,
        owner_snapshot=default_owner_snapshot,
        stop_service=lambda: control("stop"),
        start_service=lambda: control("start"),
        acquire_lock=acquire_lock,
        capture=default_capture,
    )


def stopped_decision(
    observation: dict[str, Any], old_worker_pid: int, seen_zero: bool
) -> tuple[bool, bool]:
    service = observation.get("service")
    owners = observation.get("owners")
    if not isinstance(service, dict) or not isinstance(owners, dict):
        raise PromotionError("stopped observation shape differs")
    worker = owners.get("worker_pids")
    amd = owners.get("amd_pids")
    kfd = owners.get("kfd_pids")
    if not all(
        isinstance(value, list)
        and all(isinstance(pid, int) and pid > 0 for pid in value)
        for value in (worker, amd, kfd)
    ):
        raise PromotionError("stopped owner observation is invalid")
    union = set(worker) | set(amd) | set(kfd)
    if union and union != {old_worker_pid}:
        raise PromotionError("foreign GPU or worker owner observed after service stop")
    zero = not union
    if seen_zero and not zero:
        raise PromotionError("GPU or worker owner reappeared after zero")
    stable = (
        service.get("active") is False
        and service.get("running") is False
        and service.get("main_pid") == 0
        and service.get("worker_pid") == 0
        and service.get("lock_owned") is False
        and zero
    )
    return stable, seen_zero or zero


def poll_stopped(
    deps: Dependencies, old_worker_pid: int, readiness: dict[str, Any]
) -> list[dict[str, Any]]:
    deadline = deps.monotonic() + STOP_TIMEOUT_SECONDS
    stable_count = 0
    seen_zero = False
    observations = []
    while deps.monotonic() < deadline:
        observation = {
            "service": deps.service_snapshot(readiness),
            "owners": deps.owner_snapshot(),
        }
        stable, seen_zero = stopped_decision(observation, old_worker_pid, seen_zero)
        observations.append(observation)
        stable_count = stable_count + 1 if stable else 0
        if stable_count == 2:
            return observations
        deps.sleep(POLL_SECONDS)
    raise PromotionError("stable stopped owner-free state timed out")


def poll_restored(
    deps: Dependencies, before: dict[str, Any], readiness: dict[str, Any]
) -> dict[str, Any]:
    started = deps.monotonic()
    deadline = started + RESTORE_TIMEOUT_SECONDS
    observations: list[dict[str, Any]] = []
    attempts = 0
    last_failure: str | None = None

    def details() -> dict[str, Any]:
        return {
            "passed": False,
            "attempts": attempts,
            "elapsed_seconds": max(0.0, deps.monotonic() - started),
            "last_failure": last_failure,
            "observations": observations,
        }

    def terminal(reason: str, cause: BaseException | None = None) -> None:
        nonlocal last_failure
        last_failure = reason
        observations.append({"terminal_failure": reason})
        error = TerminalRestoreError(reason, details())
        if cause is None:
            raise error
        raise error from cause

    while deps.monotonic() < deadline:
        attempts += 1
        try:
            current = deps.service_snapshot(readiness)
            owners = deps.owner_snapshot()
            observation = {"service": current, "owners": owners}
            observations.append(observation)
            if current.get("active") is not True or current.get("running") is not True:
                raise TransientRestoreError("service is not active/running yet")
            main_pid = current.get("main_pid")
            if not isinstance(main_pid, int) or main_pid <= 0:
                terminal("service main PID schema differs")
            if main_pid == before.get("main_pid"):
                terminal("service main PID epoch regressed")
            if current.get("nrestarts") != 0:
                terminal("service NRestarts differs")
            if current.get("control_group") != before.get("control_group"):
                terminal("service control group identity differs")
            worker_pid = current.get("worker_pid")
            if worker_pid in (None, 0):
                raise TransientRestoreError("service cgroup worker is not ready")
            if not isinstance(worker_pid, int) or worker_pid < 0:
                terminal("service worker PID schema differs")
            if worker_pid == before.get("worker_pid"):
                terminal("service worker PID epoch regressed")
            if current.get("lock_owned") is not True:
                raise TransientRestoreError("production lock is not owned yet")
            owner_lists = []
            for key in ("worker_pids", "amd_pids", "kfd_pids"):
                value = owners.get(key)
                if not isinstance(value, list) or any(
                    not isinstance(pid, int) or pid <= 0 for pid in value
                ):
                    terminal(f"{key} owner schema differs")
                owner_lists.append(value)
            foreign = (set().union(*map(set, owner_lists))) - {worker_pid}
            if foreign:
                terminal("foreign AMD/KFD/worker owner observed")
            if any(value != [worker_pid] for value in owner_lists):
                raise TransientRestoreError("AMD/KFD worker owner is not ready")
            if current.get("healthy") is not True:
                raise TransientRestoreError("bridge readiness is not healthy yet")
            return {
                "passed": True,
                "attempts": attempts,
                "elapsed_seconds": max(0.0, deps.monotonic() - started),
                "last_failure": None,
                "observations": observations,
            }
        except TransientRestoreError as error:
            last_failure = str(error)
            observations.append({"transient_failure": last_failure})
        except TerminalRestoreError:
            raise
        except PromotionError as error:
            terminal(str(error), error)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            terminal(
                f"unexpected restore exception: {type(error).__name__}: {error}", error
            )
        except Exception as error:
            terminal(
                f"unexpected restore exception: {type(error).__name__}: {error}", error
            )
        deps.sleep(POLL_SECONDS)
    return details()


def capture_environment(profile: dict[str, Any]) -> dict[str, str]:
    result = dict(os.environ)
    result["HIP_VISIBLE_DEVICES"] = "1"
    result["ULLM_HIP_VISIBLE_DEVICES"] = "1"
    result.pop("ROCR_VISIBLE_DEVICES", None)
    result.pop("ULLM_SQ8_PROMOTION_EVIDENCE_REQUEST_ID", None)
    required = profile.get("worker", {}).get("required_environment")
    if not isinstance(required, list):
        raise PromotionError("candidate profile required environment is invalid")
    for name in required:
        result[str(name)] = "1"
    return result


def capture_command(candidate: Path, output: Path, request_id: str) -> list[str]:
    if PROMOTION_REQUEST_ID_RE.fullmatch(request_id) is None:
        raise PromotionError("candidate promotion request ID differs")
    return [
        "python3",
        str(CAPTURE),
        "--manifest",
        str(candidate / "served-model.json"),
        "--output",
        str(output),
        "--prompt-tokens",
        "128",
        "--max-new-tokens",
        "2",
        "--timeout",
        "240",
        "--sq8-promotion-evidence",
        "--sq8-promotion-request-id",
        request_id,
    ]


def finalize_directory(
    output: Path,
    documents: dict[str, dict[str, Any]],
    receipt_factory: Callable[[Path], str] | None = None,
) -> None:
    if output.exists() or output.is_symlink():
        raise PromotionError("promotion evidence output must be create-new")
    staging = output.with_name(f".{output.name}.incomplete")
    if staging.exists() or staging.is_symlink():
        raise PromotionError("promotion evidence staging path already exists")
    staging.mkdir(mode=0o700)
    try:
        for name, value in documents.items():
            if Path(name).name != name or name in {"", ".", "..", "SHA256SUMS"}:
                raise PromotionError("promotion evidence document name is unsafe")
            path = staging / name
            raw = (
                json.dumps(
                    value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True
                )
                + "\n"
            ).encode("ascii")
            with path.open("xb") as destination:
                destination.write(raw)
                destination.flush()
                os.fsync(destination.fileno())
            path.chmod(0o444)
        receipt_name = receipt_factory(staging) if receipt_factory is not None else None
        names = sorted(
            path.name
            for path in staging.iterdir()
            if path.is_file() and not path.is_symlink()
        )
        expected_names = set(documents)
        if receipt_name is not None:
            expected_names.add(receipt_name)
        if set(names) != expected_names:
            raise PromotionError("promotion evidence receipt output set differs")
        for name in names:
            path = staging / name
            metadata = path.stat(follow_symlinks=False)
            if (
                path.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise PromotionError("promotion evidence member topology differs")
            path.chmod(0o444)
            immutable = path.stat(follow_symlinks=False)
            if (
                path.is_symlink()
                or not stat.S_ISREG(immutable.st_mode)
                or immutable.st_nlink != 1
                or stat.S_IMODE(immutable.st_mode) != 0o444
            ):
                raise PromotionError("promotion evidence member immutability differs")
        sums = [f"{sha_file(staging / name)}  {name}\n" for name in names]
        sums_path = staging / "SHA256SUMS"
        with sums_path.open("xb") as destination:
            destination.write("".join(sums).encode("ascii"))
            destination.flush()
            os.fsync(destination.fileno())
        sums_path.chmod(0o444)
        directory = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        staging.chmod(0o555)
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_receipt_writer() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_ullm_sq8_actual_receipt_writer", RECEIPT_WRITER
    )
    if spec is None or spec.loader is None:
        raise PromotionError("promotion receipt writer import failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def execute(
    candidate: Path, output: Path, deps: Dependencies
) -> tuple[int, dict[str, Any]]:
    before_candidate = candidate_snapshot(candidate)
    if before_candidate.get("authorization", {}).get("actual_run_allowed") is not True:
        raise PromotionError("candidate is not authorized for actual execution")
    profile = read_object(candidate / "profile.json", "candidate profile")
    readiness = validate_readiness_contract(before_candidate.get("readiness"))
    gate = read_object(candidate / "gate.json", "candidate Gate")
    request_id = gate.get("request", {}).get("actual", {}).get("request_id")
    if (
        not isinstance(request_id, str)
        or PROMOTION_REQUEST_ID_RE.fullmatch(request_id) is None
    ):
        raise PromotionError("candidate Gate promotion request ID differs")
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "running",
        "candidate": str(candidate),
        "promotion_request_id": request_id,
        "candidate_pre": before_candidate,
        "service_prestate": None,
        "stopped_observations": [],
        "lock": None,
        "capture": None,
        "candidate_post": None,
        "vram_headroom_bytes": None,
        "restore": {
            "attempted": False,
            "passed": False,
            "attempts": 0,
            "elapsed_seconds": 0.0,
            "last_failure": None,
            "observations": [],
        },
        "failure": None,
        "actual_run_count": 0,
    }
    service_touched = False
    lease = None
    capture_record: dict[str, Any] | None = None
    capture_temp = (
        Path(tempfile.mkdtemp(prefix="ullm-sq8-overlay-evidence-run-"))
        / "executor-record.json"
    )
    code = 1
    try:
        prestate = deps.service_snapshot(readiness)
        evidence["service_prestate"] = prestate
        if not (
            prestate.get("active") is True
            and prestate.get("running") is True
            and prestate.get("healthy") is True
            and prestate.get("lock_owned") is True
        ):
            raise PromotionError(
                "default service prestate is not active/running/healthy/lock-owner"
            )
        old_worker_pid = prestate.get("worker_pid")
        if not isinstance(old_worker_pid, int) or old_worker_pid <= 0:
            raise PromotionError("default service prestate worker PID is invalid")
        service_touched = True
        deps.stop_service()
        evidence["stopped_observations"] = poll_stopped(deps, old_worker_pid, readiness)
        evidence["vram_headroom_bytes"] = deps.vram_headroom_bytes()
        if type(evidence["vram_headroom_bytes"]) is not int or evidence["vram_headroom_bytes"] < 1:
            raise PromotionError("VRAM headroom observation is not positive")
        lease = deps.acquire_lock()
        evidence["lock"] = lease.evidence()
        command = capture_command(candidate, capture_temp, request_id)
        environment = capture_environment(profile)
        evidence["capture"] = {
            "argv": command,
            "environment": {
                name: environment[name]
                for name in (
                    "HIP_VISIBLE_DEVICES",
                    "ULLM_HIP_VISIBLE_DEVICES",
                    *REQUIRED_OVERLAY_ENV,
                )
            },
        }
        evidence["actual_run_count"] = 1
        try:
            completed = _coerce_capture_result(deps.capture(command, environment))
        except subprocess.TimeoutExpired as error:
            evidence["capture_failure"] = capture_failure_diagnostic(
                stage="capture_subprocess_timeout",
                returncode=None,
                stdout=error.stdout,
                stderr=error.stderr,
                timeout_seconds=float(error.timeout),
            )
            raise PromotionError("candidate SQ8 capture timed out") from error
        if completed.timed_out:
            evidence["capture_failure"] = capture_failure_diagnostic(
                stage="capture_subprocess_timeout",
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                timeout_seconds=completed.timeout_seconds,
            )
            raise PromotionError("candidate SQ8 capture timed out")
        if not completed.stdout.complete or not completed.stderr.complete:
            evidence["capture_failure"] = capture_failure_diagnostic(
                stage="capture_stream_collection",
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
            raise PromotionError("candidate SQ8 capture stream collection failed")
        if completed.returncode != 0:
            evidence["capture_failure"] = capture_failure_diagnostic(
                stage="capture_subprocess_completed",
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
            evidence["capture_failure"]["capture_tool_error"] = _capture_error_envelope(
                completed.stdout, request_id
            )
            raise PromotionError("candidate SQ8 capture failed")
        try:
            if (
                completed.stdout.parse_buffer is None
                or completed.stdout.parse_buffer_truncated
            ):
                raise ValueError("capture status output exceeds its bound")
            capture_status = (
                _unique_json_object(completed.stdout.parse_buffer)
                if completed.stdout.parse_buffer.strip()
                else None
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as error:
            evidence["capture_failure"] = capture_failure_diagnostic(
                stage="capture_status_parse",
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
            raise PromotionError("candidate SQ8 capture status JSON differs") from error
        if capture_status != {"status": "ok", "output": str(capture_temp)}:
            evidence["capture_failure"] = capture_failure_diagnostic(
                stage="capture_status_validation",
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
            raise PromotionError("candidate SQ8 capture failed")
        capture_record = validate_executor_record(
            capture_temp, before_candidate, request_id
        )
        after_candidate = candidate_snapshot(candidate)
        evidence["candidate_post"] = after_candidate
        if stable_identity(after_candidate) != stable_identity(before_candidate):
            raise PromotionError(
                "candidate/source/artifact/package identity changed during capture"
            )
        code = 0
    except (PromotionError, OSError, ValueError, subprocess.SubprocessError) as error:
        evidence["failure"] = {"reason": str(error)}
    finally:
        if lease is not None:
            try:
                lease.release()
                evidence["lock"]["released"] = True
            except (OSError, PromotionError) as error:
                evidence["lock"]["released"] = False
                evidence["failure"] = {"reason": f"lock release failed: {error}"}
                code = 1
        shutil.rmtree(capture_temp.parent, ignore_errors=True)
        if service_touched:
            evidence["restore"]["attempted"] = True
            try:
                deps.start_service()
                restored = poll_restored(deps, evidence["service_prestate"], readiness)
                evidence["restore"].update(restored)
                if not restored["passed"]:
                    raise PromotionError(
                        "default service restore/new epoch/health timed out"
                    )
            except (
                PromotionError,
                OSError,
                ValueError,
                subprocess.SubprocessError,
            ) as error:
                if (
                    isinstance(error, TerminalRestoreError)
                    and error.details is not None
                ):
                    evidence["restore"].update(error.details)
                evidence["restore"]["error"] = str(error)
                code = 1
    evidence["status"] = (
        "passed" if code == 0 and evidence["restore"]["passed"] else "failed"
    )
    documents = {"maintenance-evidence.json": evidence}
    if capture_record is not None:
        documents["executor-record.json"] = capture_record
    prepared_receipt = Path(str(profile["promotion"]["receipt"])).resolve()

    def receipt_factory(staging: Path) -> str:
        writer = load_receipt_writer()
        maintenance_path = staging / "maintenance-evidence.json"
        if evidence["status"] == "passed" and capture_record is not None:
            name = "promotion-actual-receipt.json"
            writer.write_actual_receipt(
                prepared_receipt_path=prepared_receipt,
                maintenance_evidence_path=maintenance_path,
                executor_record_path=staging / "executor-record.json",
                output_path=staging / name,
            )
            return name
        name = "promotion-failure-receipt.json"
        writer.write_failure_receipt(
            prepared_receipt_path=prepared_receipt,
            maintenance_evidence_path=maintenance_path,
            output_path=staging / name,
        )
        return name

    finalize_directory(output, documents, receipt_factory)
    return code, evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-independent-audit", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not args.execute:
            snapshot = candidate_snapshot(args.candidate.resolve())
            print(
                json.dumps(
                    {
                        "status": "dry-run-ready",
                        "candidate_sha256": canonical_sha(snapshot),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if not args.confirm_independent_audit:
            raise PromotionError(
                "actual execution requires --confirm-independent-audit"
            )
        code, evidence = execute(
            args.candidate.resolve(), args.output.resolve(), default_dependencies()
        )
        print(
            json.dumps(
                {"status": evidence["status"], "output": str(args.output.resolve())},
                sort_keys=True,
            )
        )
        return code
    except (PromotionError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"SQ8 overlay GPU promotion failed: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
