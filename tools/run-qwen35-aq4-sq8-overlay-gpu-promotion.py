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
import json
import os
import shutil
import stat
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "tools/capture-aq4-resident-executor-record.py"
SCHEMA = "ullm.qwen35_aq4.sq8_overlay_gpu_promotion_maintenance.v1"
GATE_SCHEMA = "ullm.qwen35_aq4.sq8_overlay_gpu_promotion_gate.v1"
TELEMETRY_SCHEMA = "ullm.qwen35_aq4.sq8_promotion_telemetry.v1"
IMPLEMENTATION_ID = "qwen35_aq4_sq8_linear_qkv_z_overlay_v1"
EXECUTION_PROFILE = "rdna4_aq4_resident_sq8_linear_qkv_z_overlay"
SERVICE = "ullm-openai.service"
LOCK_PATH = Path("/run/ullm/device-1.lock")
STOP_TIMEOUT_SECONDS = 30.0
RESTORE_TIMEOUT_SECONDS = 120.0
POLL_SECONDS = 0.25
MAX_JSON_BYTES = 16 * 1024 * 1024
READY_URL = "http://172.20.0.1:8000/readyz"
READY_BODY = b'{"status":"ready"}'
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
        json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("ascii")
    ).hexdigest()


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


def tree_inventory(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise PromotionError("overlay artifact root must be a directory without symlinks")
    entries: list[dict[str, Any]] = []
    directory_count = 0
    file_count = 0
    symlink_count = 0
    total_bytes = 0
    for path in sorted((root, *root.rglob("*")), key=lambda item: item.relative_to(root).as_posix() if item != root else ""):
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


def candidate_snapshot(candidate: Path) -> dict[str, Any]:
    gate_path = candidate / "gate.json"
    build_path = candidate / "build-receipt.json"
    profile_path = candidate / "profile.json"
    manifest_path = candidate / "served-model.json"
    worker_path = candidate / "ullm-aq4-worker"
    gate = read_object(gate_path, "candidate Gate")
    build = read_object(build_path, "candidate build receipt")
    profile = read_object(profile_path, "candidate profile")
    manifest = read_object(manifest_path, "candidate manifest")
    promotion = profile.get("promotion")
    if not isinstance(promotion, dict) or set(promotion) != {
        "receipt",
        "source_commit_from_receipt",
        "required_schema_version",
        "overlay_from_receipt",
        "release_from_receipt",
        "package_from_receipt",
        "actual_evidence_from_receipt",
        "release_source_commit",
    }:
        raise PromotionError("candidate strict promotion profile differs")
    receipt_path = Path(str(promotion["receipt"])).resolve()
    receipt = read_object(receipt_path, "candidate promotion receipt")
    if gate.get("schema_version") != GATE_SCHEMA or gate.get("actual_run_allowed") is not False:
        raise PromotionError("candidate Gate authorization differs")
    if gate.get("release_source_commit") != build.get("release_source_commit"):
        raise PromotionError("candidate Gate and build source commits differ")
    identity = gate.get("profile_identity")
    worker_identity = manifest.get("worker", {}).get("identity")
    if not isinstance(identity, dict) or identity.get("implementation_id") != IMPLEMENTATION_ID:
        raise PromotionError("candidate implementation identity differs")
    if worker_identity != {"device": "gfx1201", "execution_profile": EXECUTION_PROFILE}:
        raise PromotionError("candidate worker identity differs")
    required = profile.get("worker", {}).get("required_environment")
    if not isinstance(required, list) or any(name not in required for name in REQUIRED_OVERLAY_ENV):
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
    if set(receipt) != {"schema_version", "status", "source_commit", "source_provenance", "release", "overlay", "package", "actual"}:
        raise PromotionError("candidate promotion receipt shape differs")
    source = receipt.get("source_provenance")
    if (
        receipt.get("schema_version") != "ullm.qwen35_aq4_sq8_overlay_promotion.v1"
        or receipt.get("status") != "prepared_not_executed"
        or receipt.get("actual") != {"status": "pending", "required": True}
        or receipt.get("source_commit") != build.get("release_source_commit")
        or source != {
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
    receipt_inventory = receipt_overlay.get("artifact_inventory") if isinstance(receipt_overlay, dict) else None
    if not isinstance(release, dict) or release.get("worker") != {
        "path": str(worker_path.resolve()),
        "sha256": sha_file(worker_path),
        "bytes": worker_path.stat().st_size,
        "mode": "0555",
        "nlink": 1,
    }:
        raise PromotionError("candidate promotion receipt worker differs")
    if release.get("profile") != {"path": str(profile_path.resolve()), "sha256": sha_file(profile_path)}:
        raise PromotionError("candidate promotion receipt profile differs")
    served_release = release.get("served_model")
    semantic_manifest = json.loads(json.dumps(manifest))
    if isinstance(semantic_manifest.get("promotion"), dict):
        semantic_manifest["promotion"].pop("receipt_sha256", None)
    if served_release != {"path": str(manifest_path.resolve()), "semantic_sha256": canonical_sha(semantic_manifest)}:
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
    if not isinstance(receipt_inventory, dict) or receipt_inventory.get("entries") != expected_entries:
        raise PromotionError("candidate promotion receipt inventory differs")
    if receipt_package != {"manifest_path": str(package.resolve()), "manifest_sha256": sha_file(package)}:
        raise PromotionError("candidate promotion receipt package differs")
    if manifest.get("promotion", {}).get("receipt_sha256") != sha_file(receipt_path):
        raise PromotionError("candidate served manifest receipt SHA differs")
    return {
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
    }


def stable_identity(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Drop only evidence paths; every file and tree identity remains exact."""
    return snapshot


def validate_executor_record(path: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    value = read_object(path, "SQ8 executor record")
    evidence = value.get("sq8_promotion_evidence")
    if value.get("status") != "ok" or not isinstance(evidence, dict):
        raise PromotionError("SQ8 executor evidence is incomplete")
    if evidence.get("schema_version") != "ullm.qwen35_aq4.sq8_promotion_executor.v1":
        raise PromotionError("SQ8 executor evidence schema differs")
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
    staging = telemetry.get("diagnostic_host_staging") if isinstance(telemetry, dict) else None
    if not isinstance(telemetry, dict) or telemetry.get("schema_version") != TELEMETRY_SCHEMA:
        raise PromotionError("SQ8 executor telemetry schema differs")
    if not isinstance(projection, dict) or projection.get("batch_matvec_count", 0) <= 0 or projection.get("pair_matvec_count", 0) <= 0:
        raise PromotionError("SQ8 executor lacks batch/pair calls")
    if any(projection.get(key) != 0 for key in ("single_matvec_count", "triple_matvec_count", "fallback_count")):
        raise PromotionError("SQ8 executor used an unexpected projection path")
    if not isinstance(staging, dict) or any(staging.get(key) != 0 for key in ("read_count", "write_count", "read_bytes", "write_bytes")):
        raise PromotionError("SQ8 executor used diagnostic host staging")
    output = evidence.get("output_identity")
    if not isinstance(output, dict) or output.get("token_ids_recorded") is not False or not isinstance(output.get("token_ids_sha256"), str) or len(output["token_ids_sha256"]) != 64:
        raise PromotionError("SQ8 executor output identity is incomplete")
    return value


@dataclass
class LockLease:
    path: Path
    descriptor: int
    device: int
    inode: int

    def release(self) -> None:
        fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        os.close(self.descriptor)

    def evidence(self) -> dict[str, Any]:
        return {"path": str(self.path), "device": self.device, "inode": self.inode, "held": True}


def acquire_lock(path: Path = LOCK_PATH) -> LockLease:
    if path != LOCK_PATH:
        raise PromotionError("candidate promotion lock path differs")
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    value = os.fstat(descriptor)
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1 or stat.S_IMODE(value.st_mode) != 0o600:
        os.close(descriptor)
        raise PromotionError("candidate promotion lock substrate differs")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        os.close(descriptor)
        raise PromotionError("candidate promotion lock is busy") from error
    current = path.stat(follow_symlinks=False)
    if (current.st_dev, current.st_ino) != (value.st_dev, value.st_ino):
        os.close(descriptor)
        raise PromotionError("candidate promotion lock inode changed")
    return LockLease(path, descriptor, value.st_dev, value.st_ino)


def _run(argv: list[str], *, env: dict[str, str] | None = None, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=False, capture_output=True, text=True, env=env, timeout=timeout)


def _bounded_bytes(path: Path, limit: int, label: str) -> bytes:
    with path.open("rb") as source:
        raw = source.read(limit + 1)
    if len(raw) > limit:
        raise PromotionError(f"{label} exceeds its bound")
    return raw


def _cgroup_pids(control_group: str) -> list[int]:
    if not control_group.startswith("/system.slice/") or ".." in Path(control_group).parts:
        raise PromotionError("service control group differs")
    path = Path("/sys/fs/cgroup") / control_group.lstrip("/") / "cgroup.procs"
    try:
        values = _bounded_bytes(path, 64 * 1024, "service cgroup process list").decode("ascii").splitlines()
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
    names = [str(pid) for pid in candidates] if candidates is not None else [entry.name for entry in Path("/proc").iterdir() if entry.name.isdigit()]
    result: list[int] = []
    for name in names:
        try:
            argv0 = _bounded_bytes(Path("/proc") / name / "cmdline", 64 * 1024, "worker command line").split(b"\0", 1)[0]
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
        lines = _bounded_bytes(Path("/proc/locks"), 4 * 1024 * 1024, "kernel lock table").decode("ascii").splitlines()
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


def _ready() -> bool:
    request = urllib.request.Request(READY_URL, method="GET", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read(len(READY_BODY) + 1)
            return response.status == 200 and body == READY_BODY
    except OSError:
        return False


def _amd_owner_snapshot(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PromotionError("AMD GPU owner JSON differs") from error
    if not isinstance(value, list) or len(value) != 1:
        raise PromotionError("AMD GPU owner root differs")
    root = value[0]
    if not isinstance(root, dict) or set(root) != {"gpu", "process_list"} or root.get("gpu") != AMD_SMI_INDEX:
        raise PromotionError("AMD GPU owner identity differs")
    processes = root["process_list"]
    if processes == [{"process_info": "No running processes detected"}]:
        return {"owners": [], "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_bytes": len(raw)}
    if not isinstance(processes, list) or not processes:
        raise PromotionError("AMD GPU owner process list differs")
    owners: list[int] = []
    expected = {"name", "pid", "mem_usage", "cu_occupancy", "evicted_time"}
    for process in processes:
        if not isinstance(process, dict) or set(process) != {"process_info"}:
            raise PromotionError("AMD GPU owner entry differs")
        info = process["process_info"]
        if not isinstance(info, dict) or set(info) != expected or not isinstance(info.get("pid"), int) or info["pid"] <= 0:
            raise PromotionError("AMD GPU owner process information differs")
        owners.append(info["pid"])
    if len(owners) != len(set(owners)):
        raise PromotionError("AMD GPU owner PID is duplicated")
    return {"owners": sorted(owners), "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_bytes": len(raw)}


def _kfd_owner_snapshot() -> dict[str, Any]:
    try:
        root_before = KFD_PROC_ROOT.stat()
        process_names = sorted(os.listdir(KFD_PROC_ROOT))
    except OSError as error:
        raise PromotionError("KFD owner root is unavailable") from error
    if not stat.S_ISDIR(root_before.st_mode) or any(not name.isdigit() or int(name) <= 0 for name in process_names):
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
        if not stat.S_ISDIR(queues_before.st_mode) or any(not name.isdigit() for name in queue_names):
            raise PromotionError("KFD queue source schema differs")
        for queue_name in queue_names:
            path = queues / queue_name / "gpuid"
            try:
                before = path.stat()
                raw = _bounded_bytes(path, 64, "KFD GPU ID")
                after = path.stat()
            except FileNotFoundError as error:
                raise PromotionError("KFD owner source changed during scan") from error
            if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
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
            sources.append({"pid": pid, "queue": int(queue_name), "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_bytes": len(raw)})
            if gpuid == KFD_ID:
                owners.add(pid)
        try:
            if sorted(os.listdir(queues)) != queue_names or (queues.stat().st_dev, queues.stat().st_ino) != (queues_before.st_dev, queues_before.st_ino):
                raise PromotionError("KFD queue source changed during scan")
        except FileNotFoundError as error:
            raise PromotionError("KFD owner source changed during scan") from error
    try:
        root_after = KFD_PROC_ROOT.stat()
        final_names = sorted(os.listdir(KFD_PROC_ROOT))
    except OSError as error:
        raise PromotionError("KFD owner root changed during scan") from error
    if final_names != process_names or (root_after.st_dev, root_after.st_ino) != (root_before.st_dev, root_before.st_ino):
        raise PromotionError("KFD owner root changed during scan")
    return {
        "owners": sorted(owners),
        "enumerated_pids": [int(name) for name in process_names],
        "sources": sources,
        "root": {"path": str(KFD_PROC_ROOT), "device": root_before.st_dev, "inode": root_before.st_ino},
    }


def default_service_snapshot() -> dict[str, Any]:
    fields = "ActiveState,SubState,MainPID,NRestarts,ControlGroup"
    result = _run(["systemctl", "show", SERVICE, f"--property={fields}"], timeout=5)
    if result.returncode != 0:
        raise PromotionError("service snapshot failed")
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if set(values) != {"ActiveState", "SubState", "MainPID", "NRestarts", "ControlGroup"}:
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
        raise PromotionError("active service process topology differs")
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
        "healthy": active and running and _ready(),
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
        "amd_diagnostic": {key: value for key, value in parsed.items() if key != "owners"},
        "kfd_source": kfd,
    }


def default_capture(argv: list[str], environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return _run(argv, env=environment, timeout=300)


@dataclass
class Dependencies:
    service_snapshot: Callable[[], dict[str, Any]]
    owner_snapshot: Callable[[], dict[str, Any]]
    stop_service: Callable[[], None]
    start_service: Callable[[], None]
    acquire_lock: Callable[[], Any]
    capture: Callable[[list[str], dict[str, str]], subprocess.CompletedProcess[str]]
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


def stopped_decision(observation: dict[str, Any], old_worker_pid: int, seen_zero: bool) -> tuple[bool, bool]:
    service = observation.get("service")
    owners = observation.get("owners")
    if not isinstance(service, dict) or not isinstance(owners, dict):
        raise PromotionError("stopped observation shape differs")
    worker = owners.get("worker_pids")
    amd = owners.get("amd_pids")
    kfd = owners.get("kfd_pids")
    if not all(isinstance(value, list) and all(isinstance(pid, int) and pid > 0 for pid in value) for value in (worker, amd, kfd)):
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


def poll_stopped(deps: Dependencies, old_worker_pid: int) -> list[dict[str, Any]]:
    deadline = deps.monotonic() + STOP_TIMEOUT_SECONDS
    stable_count = 0
    seen_zero = False
    observations = []
    while deps.monotonic() < deadline:
        observation = {"service": deps.service_snapshot(), "owners": deps.owner_snapshot()}
        stable, seen_zero = stopped_decision(observation, old_worker_pid, seen_zero)
        observations.append(observation)
        stable_count = stable_count + 1 if stable else 0
        if stable_count == 2:
            return observations
        deps.sleep(POLL_SECONDS)
    raise PromotionError("stable stopped owner-free state timed out")


def poll_restored(deps: Dependencies, before: dict[str, Any]) -> list[dict[str, Any]]:
    deadline = deps.monotonic() + RESTORE_TIMEOUT_SECONDS
    observations = []
    while deps.monotonic() < deadline:
        current = deps.service_snapshot()
        owners = deps.owner_snapshot()
        observation = {"service": current, "owners": owners}
        observations.append(observation)
        worker_pid = current.get("worker_pid")
        if (
            current.get("active") is True
            and current.get("running") is True
            and current.get("healthy") is True
            and current.get("lock_owned") is True
            and isinstance(current.get("main_pid"), int)
            and current["main_pid"] > 0
            and current["main_pid"] != before.get("main_pid")
            and isinstance(worker_pid, int)
            and worker_pid > 0
            and worker_pid != before.get("worker_pid")
            and current.get("control_group") == before.get("control_group")
            and owners.get("worker_pids") == [worker_pid]
            and owners.get("amd_pids") == [worker_pid]
            and owners.get("kfd_pids") == [worker_pid]
        ):
            return observations
        deps.sleep(POLL_SECONDS)
    raise PromotionError("default service restore/new epoch/health timed out")


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


def capture_command(candidate: Path, output: Path) -> list[str]:
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
        "1",
        "--timeout",
        "240",
        "--sq8-promotion-evidence",
    ]


def finalize_directory(output: Path, documents: dict[str, dict[str, Any]]) -> None:
    if output.exists() or output.is_symlink():
        raise PromotionError("promotion evidence output must be create-new")
    staging = output.with_name(f".{output.name}.incomplete")
    if staging.exists() or staging.is_symlink():
        raise PromotionError("promotion evidence staging path already exists")
    staging.mkdir(mode=0o700)
    try:
        sums = []
        for name, value in documents.items():
            path = staging / name
            raw = (json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("ascii")
            with path.open("xb") as destination:
                destination.write(raw)
                destination.flush()
                os.fsync(destination.fileno())
            path.chmod(0o444)
            sums.append(f"{sha_file(path)}  {name}\n")
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


def execute(candidate: Path, output: Path, deps: Dependencies) -> tuple[int, dict[str, Any]]:
    before_candidate = candidate_snapshot(candidate)
    profile = read_object(candidate / "profile.json", "candidate profile")
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "running",
        "candidate": str(candidate),
        "candidate_pre": before_candidate,
        "service_prestate": None,
        "stopped_observations": [],
        "lock": None,
        "capture": None,
        "candidate_post": None,
        "restore": {"attempted": False, "passed": False, "observations": []},
        "failure": None,
        "actual_run_count": 0,
    }
    service_touched = False
    lease = None
    capture_record: dict[str, Any] | None = None
    capture_temp = Path(tempfile.mkdtemp(prefix="ullm-sq8-overlay-capture-")) / "executor-record.json"
    code = 1
    try:
        prestate = deps.service_snapshot()
        evidence["service_prestate"] = prestate
        if not (
            prestate.get("active") is True
            and prestate.get("running") is True
            and prestate.get("healthy") is True
            and prestate.get("lock_owned") is True
        ):
            raise PromotionError("default service prestate is not active/running/healthy/lock-owner")
        old_worker_pid = prestate.get("worker_pid")
        if not isinstance(old_worker_pid, int) or old_worker_pid <= 0:
            raise PromotionError("default service prestate worker PID is invalid")
        service_touched = True
        deps.stop_service()
        evidence["stopped_observations"] = poll_stopped(deps, old_worker_pid)
        lease = deps.acquire_lock()
        evidence["lock"] = lease.evidence()
        command = capture_command(candidate, capture_temp)
        environment = capture_environment(profile)
        evidence["capture"] = {
            "argv": command,
            "environment": {name: environment[name] for name in ("HIP_VISIBLE_DEVICES", "ULLM_HIP_VISIBLE_DEVICES", *REQUIRED_OVERLAY_ENV)},
        }
        completed = deps.capture(command, environment)
        evidence["actual_run_count"] = 1
        try:
            capture_status = json.loads(completed.stdout) if completed.stdout.strip() else None
        except json.JSONDecodeError as error:
            raise PromotionError("candidate SQ8 capture status JSON differs") from error
        if completed.returncode != 0 or capture_status != {"status": "ok", "output": str(capture_temp)}:
            raise PromotionError("candidate SQ8 capture failed")
        capture_record = validate_executor_record(capture_temp, before_candidate)
        after_candidate = candidate_snapshot(candidate)
        evidence["candidate_post"] = after_candidate
        if stable_identity(after_candidate) != stable_identity(before_candidate):
            raise PromotionError("candidate/source/artifact/package identity changed during capture")
        code = 0
    except (PromotionError, OSError, ValueError, subprocess.SubprocessError) as error:
        evidence["failure"] = {"reason": str(error)}
    finally:
        if lease is not None:
            try:
                lease.release()
                evidence["lock"]["released"] = True
            except OSError as error:
                evidence["lock"]["released"] = False
                evidence["failure"] = {"reason": f"lock release failed: {error}"}
                code = 1
        shutil.rmtree(capture_temp.parent, ignore_errors=True)
        if service_touched:
            evidence["restore"]["attempted"] = True
            try:
                deps.start_service()
                evidence["restore"]["observations"] = poll_restored(deps, evidence["service_prestate"])
                evidence["restore"]["passed"] = True
            except (PromotionError, OSError, ValueError, subprocess.SubprocessError) as error:
                evidence["restore"]["error"] = str(error)
                code = 1
    evidence["status"] = "passed" if code == 0 and evidence["restore"]["passed"] else "failed"
    documents = {"maintenance-evidence.json": evidence}
    if capture_record is not None:
        documents["executor-record.json"] = capture_record
    finalize_directory(output, documents)
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
            print(json.dumps({"status": "dry-run-ready", "candidate_sha256": canonical_sha(snapshot)}, sort_keys=True))
            return 0
        if not args.confirm_independent_audit:
            raise PromotionError("actual execution requires --confirm-independent-audit")
        code, evidence = execute(args.candidate.resolve(), args.output.resolve(), default_dependencies())
        print(json.dumps({"status": evidence["status"], "output": str(args.output.resolve())}, sort_keys=True))
        return code
    except (PromotionError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"SQ8 overlay GPU promotion failed: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
