#!/usr/bin/env python3
"""Run one serial trusted-local G0 row; identity and health observation only."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ContractError,
    ROOT,
    canonical_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    sha256_json,
    validate_result_payload,
)
from validate_g0_contracts import (  # noqa: E402
    AMD_SMI_EXECUTABLE,
    AMD_SMI_LIST_COMMAND,
    VISIBILITY_NAMES,
    amd_smi_uuid_to_hip_uuid,
    exact_target_from_gcn_arch_name,
    native_provider_source_contract,
    path_outside_repo,
    reject_inherited_visibility_selectors,
    regular_non_symlink,
    row_by_id,
    validate_g0_matrix,
    validate_g0_preflight,
    validate_sidecar,
    validate_visibility_environment,
)

ZERO_SHA = "0" * 64
ZERO_SHA40 = "0" * 40
G0_PREFLIGHT_SCHEMA = ROOT / "ci/schema/g0-preflight-v1.schema.json"
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
AMD_UUID = re.compile(r"^[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")
HEX16 = re.compile(r"^[0-9a-f]{16}$")
SYSFS_BDF = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
SYSFS_RAS_COUNTER_MAX = (1 << 64) - 1
SYSFS_RAS_COUNTER_MAX_TEXT = str(SYSFS_RAS_COUNTER_MAX)
SYSFS_RAS_COUNTERS = re.compile(
    r"ue: (0|[1-9][0-9]*)\n"
    r"ce: (0|[1-9][0-9]*)\n"
    r"de: (0|[1-9][0-9]*)\n?\Z"
)


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def git_output(repo: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=repo, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise ContractError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def git_candidate(repo: Path, reviewed: str | None, tested: str | None, workflow: str | None) -> dict[str, Any]:
    commit = git_output(repo, "rev-parse", "HEAD")
    tree = git_output(repo, "rev-parse", "HEAD^{tree}")
    tracked = git_output(repo, "status", "--porcelain=v1", "--untracked-files=all")
    candidate = {
        "reviewed_sha": reviewed or "",
        "tested_sha": tested or "",
        "workflow_sha": workflow or "",
        "git_tree_oid": tree,
        "worktree_clean": not bool(tracked),
        "revision_input": "full-sha",
    }
    if any(value != commit for value in (candidate["reviewed_sha"], candidate["tested_sha"], candidate["workflow_sha"])):
        raise ContractError("reviewed/tested/workflow inputs must be the checked-out full SHA")
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(r"[0-9a-f]{40}", tree):
        raise ContractError("checked-out candidate is not a full immutable SHA/tree")
    if not candidate["worktree_clean"]:
        raise ContractError("G0 rejects a dirty worktree")
    return candidate


def safe_output_directory(output: Path, repo: Path, row: dict[str, Any]) -> Path:
    if not output.is_absolute():
        raise ContractError("G0 output directory must be absolute")
    resolved = output.resolve(strict=False)
    path_outside_repo(resolved, repo, "G0 output directory")
    if resolved.name != f"g0-{row['target']}" or not resolved.parent.name.startswith("ullm-g0-") or resolved.parent.parent != Path("/tmp"):
        raise ContractError("G0 output must be /tmp/ullm-g0-*/g0-<exact-target>")
    if output.exists() and output.is_symlink():
        raise ContractError("G0 output directory must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    if output.resolve() != resolved:
        raise ContractError("G0 output directory changed during creation")
    return resolved


@contextmanager
def nonblocking_host_lock(path: Path) -> Iterator[None]:
    if path != Path("/tmp/ullm-g0.lock"):
        raise ContractError("G0 lock path is not canonical")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError("G0 host lock is busy; rows must run serially") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def artifact_binding(metadata_path: Path, row: dict[str, Any], repo: Path) -> dict[str, Any]:
    if not metadata_path.is_absolute():
        raise ContractError("H3 metadata path must be absolute")
    regular_non_symlink(metadata_path, "H3 metadata")
    path_outside_repo(metadata_path, repo, "H3 metadata")
    metadata_path = metadata_path.resolve()
    if metadata_path.name != "hip-artifact-metadata.json" or metadata_path.parent.name != row["h3_artifact_row_id"]:
        raise ContractError("H3 metadata must be staged in its exact target row directory")
    metadata = read_json(metadata_path)
    if not isinstance(metadata, dict) or not isinstance(metadata.get("artifact"), dict) or not isinstance(metadata["artifact"].get("path"), str):
        raise ContractError("H3 metadata artifact record is missing")
    artifact_path = metadata_path.parent / f"device-code-object-{row['target']}.elf"
    regular_non_symlink(artifact_path, "H3 artifact")
    path_outside_repo(artifact_path, repo, "H3 artifact")
    artifact_path = artifact_path.resolve()
    metadata_sidecar = metadata_path.with_name(metadata_path.name + ".sha256")
    artifact_sidecar = artifact_path.with_name(artifact_path.name + ".sha256")
    return {
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "metadata_sidecar_path": str(metadata_sidecar),
        "metadata_sidecar_sha256": validate_sidecar(metadata_sidecar, metadata_path, "H3 metadata sidecar"),
        "metadata_declared_artifact_path": metadata["artifact"]["path"],
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_sidecar_path": str(artifact_sidecar),
        "artifact_sidecar_sha256": validate_sidecar(artifact_sidecar, artifact_path, "H3 artifact sidecar"),
        "h3_matrix_row_id": row["h3_artifact_row_id"],
        "target": row["target"],
        "toolchain_id": "rocm-7.14.0",
        "toolchain_manifest_sha256": metadata["toolchain_manifest_sha256"],
    }


def unavailable_health() -> dict[str, Any]:
    return {
        "available": False,
        "reliable": False,
        "observed_at": None,
        "bdf": None,
        "uuid": None,
        "gcnArchName": None,
        "source": None,
        "facts": {
            "device_state": None,
            "amdgpu_driver_bound": None,
            "runtime_status": None,
            "ras_uncorrectable_count": None,
            "sysfs_ras_uncorrectable_count": None,
            "temperature_c": None,
        },
    }


def unavailable_process() -> dict[str, Any]:
    return {
        "available": False,
        "reliable": False,
        "observed_at": None,
        "bdf": None,
        "uuid": None,
        "gcnArchName": None,
        "source": None,
        "gpu_processes": [],
        "residual_runner_children": [],
    }


def unavailable_preflight(candidate: dict[str, Any], visibility: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "g0-preflight-v1",
        "candidate": candidate,
        "visibility": visibility,
        "routing": {
            "source": None,
            "amd_smi": None,
            "argv": [],
            "bdf": None,
            "uuid": None,
            "gpu": None,
            "hip_id": None,
        },
        "artifact_binding": {
            "metadata_path": "unavailable", "metadata_sha256": ZERO_SHA,
            "metadata_sidecar_path": "unavailable", "metadata_sidecar_sha256": ZERO_SHA,
            "metadata_declared_artifact_path": "unavailable", "artifact_path": "unavailable",
            "artifact_sha256": ZERO_SHA, "artifact_sidecar_path": "unavailable",
            "artifact_sidecar_sha256": ZERO_SHA, "h3_matrix_row_id": row["h3_artifact_row_id"],
            "target": row["target"], "toolchain_id": "rocm-7.14.0", "toolchain_manifest_sha256": ZERO_SHA,
        },
        "provider": {
            "provider_id": "g0-native-hip-observer-v1", "available": False,
            "source_path": None, "source_sha256": None, "binary_path": None,
            "binary_removed": False, "binary_sha256": None,
            "compiler_path": None, "compiler_version": None,
            "compile_command_sha256": None, "runtime_command_sha256": None,
        },
        "device": {
            "probe_kind": "hip-identity-only-v1", "observed": False,
            "visible_device_count": None, "ordinal": None, "bdf": None, "uuid": None,
            "hip_uuid_hex": None, "gcnArchName": None, "exact_target": None,
            "product": None, "wave_size": None, "total_global_memory_bytes": None,
            "rocm_root": None, "allocation_count": 0, "copy_count": 0,
            "kernel_dispatch_count": 0, "dispatch_count": 0,
        },
        "runtime": {
            "rocm_root": None, "release": None, "hip_runtime_api_version": None,
            "hip_runtime_library_path": None, "hsa_runtime_library_path": None,
            "bdf": None, "uuid": None, "gcnArchName": None,
        },
        "health_pre": unavailable_health(), "health_post": unavailable_health(),
        "process_pre": unavailable_process(), "process_post": unavailable_process(),
        "scope": {
            "selected_backend": "hip-preflight", "fallback_allowed": False,
            "fallback_used": False, "identity_probe_only": True,
            "native_hip_observation_provider": "native-hip-observer-v1",
            "execution_verified": False, "numerics_verified": False,
            "performance_verified": False, "support_claim": False,
        },
    }


def host_toolchain() -> dict[str, Any]:
    return {
        "python": platform.python_version(), "platform": platform.platform(aliased=True),
        "system": platform.system(), "machine": platform.machine(), "git": "observed-by-runner",
        "rustc_dev": "not-applicable", "cargo_dev": "not-applicable",
        "rustc_msrv": "not-applicable", "cargo_msrv": "not-applicable",
        "clang_format": "not-applicable", "cmake": "not-applicable",
        "host_packages": {"g0": "trusted-local-preflight"},
    }


def make_report(
    *, row: dict[str, Any], matrix: dict[str, Any], candidate: dict[str, Any], preflight: dict[str, Any],
    state: str, error: str | None, run_id: str, run_attempt: int, started: datetime, finished: datetime,
) -> dict[str, Any]:
    passed = 1 if state == "PASS" else 0
    toolchain = host_toolchain()
    runner_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    step = {
        "step_id": f"{row['row_id']}.preflight", "state": state,
        "started_at": iso(started), "finished_at": iso(finished),
        "duration_seconds": max(0.0, (finished - started).total_seconds()),
        "exit_code": 0 if state == "PASS" else 2,
        "stdout_sha256": sha256_bytes(b""), "stderr_sha256": sha256_bytes((error or "").encode()),
        "diagnostic": error or "", "selection_required": True, "count_source": "validator-command",
        "counts": {"collected": 1, "selected": 1, "passed": passed, "failed": 1 - passed, "skipped": 0, "deselected": 0},
        "resource": {
            "wall_time_limit_seconds": 300, "timed_out": False, "max_rss_bytes": 0,
            "max_rss_limit_bytes": 1073741824, "rss_breach": False, "cpu_user_seconds": 0,
            "cpu_system_seconds": 0, "stdout_bytes": 0, "stderr_bytes": len(error or ""),
            "output_bytes": len(error or ""), "stdout_captured_bytes": 0,
            "stderr_captured_bytes": len(error or ""), "captured_output_bytes": len(error or ""),
            "output_limit_bytes": 1048576, "output_breach": False, "network_isolated": False,
            "network_guard_strategy": "trusted-local-no-network-use", "address_space_limit_bytes": None,
            "address_space_limit_enforced": False,
        },
    }
    artifact = preflight["artifact_binding"]
    return {
        "schema_version": "test-result-v1", "result_id": f"{row['row_id']}.{run_id}.{run_attempt}",
        "suite_id": row["row_id"], "tier": "tier_g0", "state": state, "required": True,
        "evidence_mode": "required-ci", "run_id": run_id, "run_attempt": run_attempt,
        "reviewed_sha": candidate["reviewed_sha"], "tested_sha": candidate["tested_sha"],
        "workflow_sha": candidate["workflow_sha"], "git_tree_oid": candidate["git_tree_oid"],
        "worktree_clean": candidate["worktree_clean"], "matrix_manifest_sha256": sha256_json(matrix),
        "matrix_row_id": row["row_id"], "tuple_digest": sha256_json(row),
        "command": [["python3", "ci/tools/run_g0_preflight.py", "--row", row["row_id"], "--trusted-local"]],
        "command_sha256": sha256_json([["python3", "ci/tools/run_g0_preflight.py", "--row", row["row_id"], "--trusted-local"]]),
        "toolchain": toolchain, "toolchain_sha256": sha256_json(toolchain),
        "artifact": {"content_sha256": artifact["artifact_sha256"], "manifest_sha256": artifact["metadata_sha256"]},
        "g0": {
            "preflight": preflight, "preflight_sha256": sha256_json(preflight),
            "preflight_schema_sha256": sha256_file(G0_PREFLIGHT_SCHEMA), "kernel_dispatch_count": 0,
        },
        "created_at": iso(started), "started_at": iso(started), "finished_at": iso(finished),
        "duration_seconds": max(0.0, (finished - started).total_seconds()), "seed": row["seed"],
        "counts": {"collected": 1, "selected": 1, "passed": passed, "failed": 1 - passed, "skipped": 0, "deselected": 0},
        "resource": {
            "wall_time_limit_seconds": 300, "wall_time_breach": False, "max_rss_bytes": runner_rss,
            "max_rss_limit_bytes": 1073741824, "rss_breach": runner_rss > 1073741824,
            "runner_max_rss_bytes": runner_rss, "fixture_size_bytes": 0, "fixture_size_limit_bytes": 1,
            "fixture_size_breach": False, "output_bytes": len(error or ""), "captured_output_bytes": len(error or ""),
            "row_output_limit_bytes": 1048576, "output_breach": False, "address_space_limit_bytes": None,
            "commands_expected": 1, "commands_executed": 1, "commands_complete": True,
            "network_isolated": False, "network_guard_strategies": ["trusted-local-no-network-use"],
        },
        "cases": [{"case_id": f"{row['row_id']}.preflight", **{key: value for key, value in step.items() if key != "step_id"}}],
        "steps": [step],
        "diagnostic": {
            "message": "G0 native HIP identity/health/process observation accepted" if state == "PASS" else "G0 preflight infrastructure failed closed",
            "errors": [] if error is None else [error],
            "warnings": ["G0 does not prove execution, correctness, performance, or support"],
            "network_disabled": False, "model_disabled": True, "gpu_fallback_disabled": True,
            "network_guard_self_test": False,
        },
        "gpu": {
            "uuid": row["uuid"], "bdf": row["bdf"], "exact_target": row["target"],
            "selected_backend": "hip-preflight", "dispatch_count": 0, "kernel_dispatch_count": 0,
            "dispatch_ids": [], "fallback_allowed": False, "fallback_used": False,
            "code_object": {"target": row["target"], "artifact_sha256": artifact["artifact_sha256"]},
        },
    }


def run_command(argv: list[str], *, env: dict[str, str] | None, timeout_seconds: int, label: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ContractError(f"{label} timed out after {timeout_seconds}s") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise ContractError(f"{label} failed with exit {result.returncode}: {detail}")
    return result


def parse_json_output(result: subprocess.CompletedProcess[str], label: str) -> Any:
    try:
        value = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} returned malformed JSON") from exc
    if not isinstance(value, (dict, list)):
        raise ContractError(f"{label} returned a non-object/non-array JSON value")
    return value


def amd_smi_list_json(row: dict[str, Any], *, executable: str) -> dict[str, Any]:
    if executable != AMD_SMI_EXECUTABLE:
        raise ContractError("G0 routing AMD-SMI executable is not the fixed ROCm 7.14 binary")
    command = list(AMD_SMI_LIST_COMMAND)
    document = parse_json_output(
        run_command(
            command,
            env=None,
            timeout_seconds=30,
            label="AMD-SMI list",
        ),
        "AMD-SMI list",
    )
    if not isinstance(document, list) or not document:
        raise ContractError("AMD-SMI list JSON must contain a non-empty array")
    seen_bdf: set[str] = set()
    seen_uuid: set[str] = set()
    seen_hip_id: set[int] = set()
    matches: list[dict[str, Any]] = []
    for item in document:
        if not isinstance(item, dict):
            raise ContractError("AMD-SMI list JSON contains a malformed device")
        required = ("gpu", "bdf", "uuid", "hip_uuid", "hip_id")
        if any(key not in item for key in required):
            raise ContractError("AMD-SMI list JSON is missing a binding field")
        bdf = item["bdf"].lower() if isinstance(item["bdf"], str) else ""
        uuid = item["uuid"] if isinstance(item["uuid"], str) else ""
        hip_uuid = item["hip_uuid"] if isinstance(item["hip_uuid"], str) else ""
        if not SYSFS_BDF.fullmatch(bdf) or not AMD_UUID.fullmatch(uuid) or not HEX16.fullmatch(hip_uuid.removeprefix("GPU-").lower()):
            raise ContractError("AMD-SMI list JSON contains malformed BDF or UUID")
        if amd_smi_uuid_to_hip_uuid(uuid) != hip_uuid:
            raise ContractError("AMD-SMI UUID and HIP UUID do not match their observed format")
        gpu = item["gpu"]
        hip_id = item["hip_id"]
        if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0 or isinstance(hip_id, bool) or not isinstance(hip_id, int) or hip_id < 0:
            raise ContractError("AMD-SMI list JSON contains a malformed device index")
        if bdf in seen_bdf or hip_uuid.lower() in seen_uuid or hip_id in seen_hip_id:
            raise ContractError("AMD-SMI list JSON contains duplicate device bindings")
        seen_bdf.add(bdf)
        seen_uuid.add(hip_uuid.lower())
        seen_hip_id.add(hip_id)
        if bdf == row["bdf"]:
            matches.append({"gpu": gpu, "hip_id": hip_id, "uuid": uuid, "hip_uuid": hip_uuid})
    if len(matches) != 1:
        raise ContractError("AMD-SMI list did not resolve exactly one canonical BDF")
    match = matches[0]
    if match["hip_uuid"] != row["uuid"] or amd_smi_uuid_to_hip_uuid(match["uuid"]) != row["uuid"]:
        raise ContractError("AMD-SMI canonical BDF is bound to the wrong UUID")
    return {
        "source": "amd-smi-list-e-json-v1",
        "amd_smi": executable,
        "argv": command,
        "bdf": row["bdf"],
        "uuid": match["hip_uuid"],
        "gpu": match["gpu"],
        "hip_id": match["hip_id"],
    }


def selected_gpu_record(document: Any, gpu: int, label: str) -> dict[str, Any]:
    if not isinstance(document, dict) or not isinstance(document.get("gpu_data"), list) or len(document["gpu_data"]) != 1:
        raise ContractError(f"{label} JSON must contain exactly one gpu_data record")
    record = document["gpu_data"][0]
    if not isinstance(record, dict) or record.get("gpu") != gpu:
        raise ContractError(f"{label} JSON is not bound to the canonical AMD-SMI GPU index")
    return record


def parse_sysfs_ras_counters(text: str) -> dict[str, int]:
    """Parse the canonical three-line AMD UMC RAS counter format."""
    match = SYSFS_RAS_COUNTERS.fullmatch(text)
    if match is None:
        raise ContractError("canonical PCI sysfs RAS counter is malformed")

    counters: dict[str, int] = {}
    for key, digits in zip(("ue", "ce", "de"), match.groups()):
        if (
            len(digits) > len(SYSFS_RAS_COUNTER_MAX_TEXT)
            or (
                len(digits) == len(SYSFS_RAS_COUNTER_MAX_TEXT)
                and digits > SYSFS_RAS_COUNTER_MAX_TEXT
            )
        ):
            raise ContractError("canonical PCI sysfs RAS counter overflows uint64")
        counters[key] = int(digits, 10)
    return counters


def read_sysfs_health(row: dict[str, Any], sysfs_root: Path) -> tuple[str, bool, int]:
    device = sysfs_root / row["bdf"]
    if not device.is_dir():
        raise ContractError("canonical PCI sysfs device is unavailable")
    driver = (device / "driver").resolve()
    if driver.name != "amdgpu":
        raise ContractError("canonical PCI device is not bound to amdgpu")
    runtime_status_path = device / "power/runtime_status"
    if runtime_status_path.is_symlink() or not runtime_status_path.is_file():
        raise ContractError("canonical PCI runtime_status is unavailable or unsafe")
    runtime_status = runtime_status_path.read_text(encoding="ascii").strip()
    ras_path = device / "ras/umc_err_count"
    if ras_path.is_symlink() or not ras_path.is_file():
        raise ContractError("canonical PCI sysfs RAS counter is unavailable or unsafe")
    ras_text = ras_path.read_text(encoding="ascii")
    counters = parse_sysfs_ras_counters(ras_text)
    return runtime_status, True, counters["ue"]


def observe_health(row: dict[str, Any], binding: dict[str, Any], *, amd_smi: str, sysfs_root: Path) -> dict[str, Any]:
    static = parse_json_output(run_command([amd_smi, "static", "-a", "-d", "-b", "--json", "-g", row["bdf"]], env=None, timeout_seconds=30, label="AMD-SMI static"), "AMD-SMI static")
    static_record = selected_gpu_record(static, binding["gpu"], "AMD-SMI static")
    bus = static_record.get("bus")
    asic = static_record.get("asic")
    driver = static_record.get("driver")
    if not isinstance(bus, dict) or bus.get("bdf", "").lower() != row["bdf"] or not isinstance(asic, dict) or not isinstance(driver, dict):
        raise ContractError("AMD-SMI static JSON has malformed canonical identity")
    if asic.get("market_name") != row["product"] or asic.get("target_graphics_version") != row["target"] or driver.get("name") != "amdgpu":
        raise ContractError("AMD-SMI static JSON does not match the canonical product/target/driver")
    metric = parse_json_output(run_command([amd_smi, "metric", "-t", "-e", "--json", "-g", row["bdf"]], env=None, timeout_seconds=30, label="AMD-SMI metric"), "AMD-SMI metric")
    metric_record = selected_gpu_record(metric, binding["gpu"], "AMD-SMI metric")
    temperature = metric_record.get("temperature")
    ecc = metric_record.get("ecc")
    edge = temperature.get("edge") if isinstance(temperature, dict) else None
    temp_value = edge.get("value") if isinstance(edge, dict) else None
    ras_value = ecc.get("total_uncorrectable_count") if isinstance(ecc, dict) else None
    if isinstance(temp_value, bool) or not isinstance(temp_value, (int, float)) or not math.isfinite(float(temp_value)):
        raise ContractError("AMD-SMI temperature fact is malformed")
    if isinstance(ras_value, bool) or not isinstance(ras_value, int) or ras_value < 0:
        raise ContractError("AMD-SMI RAS fact is malformed")
    runtime_status, driver_bound, sysfs_ras = read_sysfs_health(row, sysfs_root)
    return {
        "available": True, "reliable": True, "observed_at": iso(now()),
        "bdf": row["bdf"], "uuid": row["uuid"], "gcnArchName": row["target"],
        "source": "amd-smi-sysfs-read-only-v1",
        "facts": {
            "device_state": runtime_status, "amdgpu_driver_bound": driver_bound,
            "runtime_status": runtime_status, "ras_uncorrectable_count": ras_value,
            "sysfs_ras_uncorrectable_count": sysfs_ras, "temperature_c": float(temp_value),
        },
    }


def child_process_ids(pid: int) -> list[int]:
    parent_to_children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="ascii").split()
            child_pid, parent_pid = int(fields[0]), int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
        parent_to_children.setdefault(parent_pid, []).append(child_pid)
    result: list[int] = []
    pending = list(parent_to_children.get(pid, []))
    while pending:
        child = pending.pop()
        result.append(child)
        pending.extend(parent_to_children.get(child, []))
    return sorted(set(result))


def observe_processes(row: dict[str, Any], binding: dict[str, Any], *, amd_smi: str) -> dict[str, Any]:
    document = parse_json_output(run_command([amd_smi, "process", "--json", "-g", row["bdf"]], env=None, timeout_seconds=30, label="AMD-SMI process"), "AMD-SMI process")
    if not isinstance(document, list) or len(document) != 1 or not isinstance(document[0], dict) or document[0].get("gpu") != binding["gpu"]:
        raise ContractError("AMD-SMI process JSON is not bound to the canonical GPU")
    process_list = document[0].get("process_list")
    if not isinstance(process_list, list):
        raise ContractError("AMD-SMI process JSON has no process_list")
    if process_list == [{"process_info": "No running processes detected"}]:
        processes: list[dict[str, Any]] = []
    else:
        if any(not isinstance(item, dict) or item.get("process_info") == "No running processes detected" for item in process_list):
            raise ContractError("AMD-SMI process JSON has a malformed sentinel")
        processes = process_list
    return {
        "available": True, "reliable": True, "observed_at": iso(now()),
        "bdf": row["bdf"], "uuid": row["uuid"], "gcnArchName": row["target"],
        "source": "amd-smi-sysfs-read-only-v1", "gpu_processes": processes,
        "residual_runner_children": child_process_ids(os.getpid()),
    }


def require_available_observation(observation: Any, label: str) -> None:
    if not isinstance(observation, dict):
        raise ContractError(f"{label} observation is missing or malformed")
    if observation.get("available") is not True or observation.get("reliable") is not True:
        raise ContractError(f"{label} observation is unavailable or unreliable")
    if observation.get("source") != "amd-smi-sysfs-read-only-v1":
        raise ContractError(f"{label} observation does not use the read-only AMD-SMI/sysfs provider")


def validate_native_provider_json(document: Any) -> dict[str, Any]:
    required = {"provider_id", "probe_kind", "rocm_root", "release", "hip_runtime_api_version", "hip_runtime_library_path", "hsa_runtime_library_path", "visible_device_count", "device", "scope"}
    if not isinstance(document, dict) or set(document) != required:
        raise ContractError("native HIP provider JSON has missing or unknown sections")
    if document["provider_id"] != "g0-native-hip-observer-v1" or document["probe_kind"] != "hip-identity-only-v1":
        raise ContractError("native HIP provider JSON has the wrong identity")
    if document["visible_device_count"] != 1 or document["rocm_root"] != "/opt/rocm/core-7.14" or document["release"] != "7.14.0":
        raise ContractError("native HIP provider JSON has the wrong visibility or ROCm root")
    device = document["device"]
    scope = document["scope"]
    if not isinstance(device, dict) or set(device) != {"ordinal", "bdf", "uuid", "hip_uuid_hex", "gcnArchName", "exact_target", "product", "wave_size", "total_global_memory_bytes"}:
        raise ContractError("native HIP provider JSON device section is malformed")
    if not isinstance(scope, dict) or set(scope) != {"allocation_count", "copy_count", "kernel_dispatch_count", "dispatch_count"} or any(scope[key] != 0 for key in scope):
        raise ContractError("native HIP provider JSON scope is not identity-only")
    if device["ordinal"] != 0 or not isinstance(device["bdf"], str) or not isinstance(device["uuid"], str) or not HEX16.fullmatch(device["hip_uuid_hex"]):
        raise ContractError("native HIP provider JSON device identity is malformed")
    if device["uuid"] != f"GPU-{device['hip_uuid_hex']}" or exact_target_from_gcn_arch_name(device["gcnArchName"]) != device["exact_target"]:
        raise ContractError("native HIP provider JSON UUID or target derivation is invalid")
    return document


def run_native_provider(repo: Path, row: dict[str, Any], *, hip_visible_devices: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source = repo / "ci/tools/g0_native_observer.cpp"
    source_contract = native_provider_source_contract(repo)
    root = Path("/opt/rocm/core-7.14")
    compiler = root / "bin/amdclang++"
    hip_library = root / "lib/libamdhip64.so.7.14.60850-0000000"
    hsa_library = root / "lib/libhsa-runtime64.so.1.21.0"
    if not compiler.is_file() or not hip_library.is_file() or not hsa_library.is_file():
        raise ContractError("ROCm native provider compiler/runtime library path is unavailable")
    provider_dir = Path(tempfile.mkdtemp(prefix="ullm-g0-provider-", dir="/tmp"))
    binary = provider_dir / "g0_native_observer"
    compile_argv = [
        str(compiler),
        "-std=c++17",
        "-D__HIP_PLATFORM_AMD__=1",
        "-I/opt/rocm/core-7.14/include",
        str(source),
        "-L/opt/rocm/core-7.14/lib",
        "-lamdhip64",
        "-Wl,--no-as-needed",
        "-lhsa-runtime64",
        "-Wl,--as-needed",
        "-ldl",
        "-Wl,-rpath,/opt/rocm/core-7.14/lib",
        "-o",
        str(binary),
    ]
    runtime_argv = [
        str(binary),
        "--rocm-root",
        str(root),
        "--expected-bdf",
        row["bdf"],
        "--expected-uuid",
        row["uuid"],
        "--expected-target",
        row["target"],
        "--expected-product",
        row["product"],
        "--expected-hip-library",
        str(hip_library),
        "--expected-hsa-library",
        str(hsa_library),
    ]
    provider: dict[str, Any] | None = None
    device: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    try:
        version = run_command(
            [str(compiler), "--version"],
            env=None,
            timeout_seconds=60,
            label="amdclang++ version",
        ).stdout
        if not re.search(r"(?:AMD )?clang version 23\.", version):
            raise ContractError("native provider compiler is not LLVM/clang 23")
        run_command(
            compile_argv,
            env=None,
            timeout_seconds=60,
            label="native HIP provider compile",
            cwd=repo,
        )
        regular_non_symlink(binary, "native HIP provider binary")
        environment = {"HIP_VISIBLE_DEVICES": hip_visible_devices}
        result = run_command(
            runtime_argv,
            env=environment,
            timeout_seconds=60,
            label="native HIP provider run",
        )
        if not result.stdout.strip() or result.stderr.strip():
            raise ContractError("native HIP provider emitted unexpected stderr or empty stdout")
        document = validate_native_provider_json(parse_json_output(result, "native HIP provider"))
        provider = {
            "provider_id": "g0-native-hip-observer-v1",
            "available": True,
            "source_path": source_contract["source_path"],
            "source_sha256": source_contract["source_sha256"],
            "binary_path": None,
            "binary_removed": False,
            "binary_sha256": sha256_file(binary),
            "compiler_path": str(compiler),
            "compiler_version": version.strip(),
            "compile_command_sha256": sha256_json(compile_argv),
            "runtime_command_sha256": sha256_json(
                {
                    "argv": runtime_argv,
                    "env": {"HIP_VISIBLE_DEVICES": hip_visible_devices},
                }
            ),
        }
        device = {
            "probe_kind": document["probe_kind"],
            "observed": True,
            **document["device"],
            "visible_device_count": document["visible_device_count"],
            "rocm_root": document["rocm_root"],
            **document["scope"],
        }
        runtime = {
            "rocm_root": document["rocm_root"],
            "release": document["release"],
            "hip_runtime_api_version": document["hip_runtime_api_version"],
            "hip_runtime_library_path": document["hip_runtime_library_path"],
            "hsa_runtime_library_path": document["hsa_runtime_library_path"],
            "bdf": document["device"]["bdf"],
            "uuid": document["device"]["uuid"],
            "gcnArchName": exact_target_from_gcn_arch_name(document["device"]["gcnArchName"]),
        }
    finally:
        try:
            shutil.rmtree(provider_dir)
        except OSError as exc:
            raise ContractError(f"native provider temporary cleanup failed: {exc}") from exc
    if provider is None or device is None or runtime is None:
        raise ContractError("native HIP provider completed without an observation")
    provider["binary_removed"] = True
    return provider, device, runtime


def write_report(output: Path, report: dict[str, Any]) -> None:
    if not output.is_dir() or output.is_symlink():
        raise ContractError("G0 report output is not a regular directory")
    expected_names = {"report.json", "report.json.sha256"}
    if {path.name for path in output.iterdir()} - expected_names:
        raise ContractError("G0 report output contains unknown or stale files")
    for name in expected_names:
        path = output / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ContractError(f"G0 report output member is unsafe: {name}")
    data = canonical_bytes(report)
    path = output / "report.json"
    path.write_bytes(data)
    path.with_name("report.json.sha256").write_text(f"{sha256_bytes(data)}  report.json\n", encoding="ascii")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--row", required=True, choices=("g0-gfx1030", "g0-gfx1201"))
    result.add_argument("--repo", type=Path, default=ROOT)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--trusted-local", action="store_true")
    result.add_argument("--artifact-metadata", type=Path)
    result.add_argument("--run-id", default="local-g0")
    result.add_argument("--run-attempt", type=int, default=1)
    result.add_argument("--reviewed-sha", required=True)
    result.add_argument("--tested-sha", required=True)
    result.add_argument("--workflow-sha", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    started = now()
    repo = args.repo.resolve()
    matrix = validate_g0_matrix(repo)
    row = row_by_id(matrix, args.row)
    output = safe_output_directory(args.output_dir, repo, row)
    visibility = {"HIP_VISIBLE_DEVICES": None, "CUDA_VISIBLE_DEVICES": None, "GPU_DEVICE_ORDINAL": None, "security_boundary": False}
    fallback_sha = args.reviewed_sha if re.fullmatch(r"[0-9a-f]{40}", args.reviewed_sha) else ZERO_SHA40
    candidate = {
        "reviewed_sha": fallback_sha, "tested_sha": fallback_sha, "workflow_sha": fallback_sha,
        "git_tree_oid": ZERO_SHA40, "worktree_clean": False, "revision_input": "full-sha",
    }
    report_run_id = args.run_id if RUN_ID.fullmatch(args.run_id) else "invalid-g0-run"
    report_attempt = args.run_attempt if args.run_attempt >= 1 else 1
    preflight = unavailable_preflight(candidate, visibility, row)
    state = "INFRA_ERROR"
    error: str | None = None
    try:
        if not RUN_ID.fullmatch(args.run_id):
            raise ContractError("run ID is malformed")
        if args.run_attempt < 1:
            raise ContractError("run attempt must be positive")
        if not args.trusted_local:
            raise ContractError("explicit --trusted-local execution mode is required")
        if args.artifact_metadata is None:
            raise ContractError("exact H3 artifact metadata is required")
        reject_inherited_visibility_selectors(
            {name: os.environ.get(name) for name in (*VISIBILITY_NAMES, "ROCR_VISIBLE_DEVICES")}
        )
        candidate = git_candidate(repo, args.reviewed_sha, args.tested_sha, args.workflow_sha)
        binding = artifact_binding(args.artifact_metadata, row, repo)
        execution = matrix["execution"]
        health_process = execution["health_process_observer"]
        with nonblocking_host_lock(Path(execution["host_lock"]["path"])):
            routing = amd_smi_list_json(row, executable=health_process["amd_smi"])
            visibility = validate_visibility_environment(
                {
                    "HIP_VISIBLE_DEVICES": str(routing["hip_id"]),
                    "CUDA_VISIBLE_DEVICES": None,
                    "GPU_DEVICE_ORDINAL": None,
                }
            )
            pre_health = observe_health(row, routing, amd_smi=health_process["amd_smi"], sysfs_root=Path(health_process["sysfs_pci_root"]))
            require_available_observation(pre_health, "pre-health")
            pre_process = observe_processes(row, routing, amd_smi=health_process["amd_smi"])
            require_available_observation(pre_process, "pre-process")
            if pre_process["gpu_processes"] or pre_process["residual_runner_children"]:
                raise ContractError("existing GPU process or runner child is present; G0 fails closed")
            provider, device, runtime = run_native_provider(repo, row, hip_visible_devices=str(routing["hip_id"]))
            post_health = observe_health(row, routing, amd_smi=health_process["amd_smi"], sysfs_root=Path(health_process["sysfs_pci_root"]))
            require_available_observation(post_health, "post-health")
            post_process = observe_processes(row, routing, amd_smi=health_process["amd_smi"])
            require_available_observation(post_process, "post-process")
            preflight = {
                "schema_version": "g0-preflight-v1", "candidate": candidate, "visibility": visibility,
                "routing": routing, "artifact_binding": binding, "provider": provider,
                "device": device, "runtime": runtime,
                "health_pre": pre_health, "health_post": post_health,
                "process_pre": pre_process, "process_post": post_process,
                "scope": {
                    "selected_backend": "hip-preflight", "fallback_allowed": False, "fallback_used": False,
                    "identity_probe_only": True, "native_hip_observation_provider": "native-hip-observer-v1",
                    "execution_verified": False, "numerics_verified": False, "performance_verified": False,
                    "support_claim": False,
                },
            }
            validate_g0_preflight(preflight, args.row, repo, expected_sha=candidate["reviewed_sha"], expected_tree=candidate["git_tree_oid"])
        state = "PASS"
    except (ContractError, KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        error = str(exc)
    finished = now()
    if state == "PASS":
        try:
            validate_g0_preflight(
                preflight,
                args.row,
                repo,
                expected_sha=candidate["reviewed_sha"],
                expected_tree=candidate["git_tree_oid"],
                observation_window=(started, finished),
            )
        except (ContractError, KeyError, OSError, TypeError, ValueError) as exc:
            state = "INFRA_ERROR"
            error = str(exc)
    report = make_report(row=row, matrix=matrix, candidate=candidate, preflight=preflight, state=state, error=error, run_id=report_run_id, run_attempt=report_attempt, started=started, finished=finished)
    try:
        validate_result_payload(report)
        write_report(output, report)
    except (ContractError, OSError, TypeError, ValueError) as exc:
        print(f"G0 preflight: result/output contract failure: {exc}", file=sys.stderr)
        return 3
    if state != "PASS":
        print(f"G0 preflight: INFRA_ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
