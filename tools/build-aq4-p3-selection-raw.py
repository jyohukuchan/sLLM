#!/usr/bin/env python3
"""Build hash-bound AQ4 P3 candidate-selection raw evidence from P2 traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import stat
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = ROOT / "tools/select-aq4-p3-candidate.py"
SELECTOR_SHA256 = "4a510c7351131072ed368e2ac8fffeb2daf10488edef94c37fe5dbcb729e9739"
PROFILER_PATH = ROOT / "tools/profile-aq4-p2-family-exclusive.py"
PROFILER_SHA256 = "ef26005a364511ab8d0f7ca2fa46ad2108cac083d0a2a24721f6cef577e16c92"
DEPENDENCY_MAX_BYTES = 2 * 1024 * 1024


def load_verified_tool(name: str, path: Path, expected_sha256: str) -> Any:
    if not path.is_absolute() or ".." in path.parts or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise RuntimeError("producer dependency binding is invalid")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if stat.S_ISLNK(current.lstat().st_mode):
            raise RuntimeError(f"producer dependency path contains a symlink: {current}")
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_size > DEPENDENCY_MAX_BYTES:
        raise RuntimeError("producer dependency must be a bounded single-link regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_nlink, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)
        opened_identity = (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        if opened_identity != identity:
            raise RuntimeError("producer dependency changed while opening")
        data = b""
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            if size > DEPENDENCY_MAX_BYTES:
                raise RuntimeError("producer dependency exceeds the input limit")
            chunks.append(chunk)
            digest.update(chunk)
        data = b"".join(chunks)
        final = os.fstat(descriptor)
        final_identity = (final.st_dev, final.st_ino, final.st_mode, final.st_nlink, final.st_size, final.st_mtime_ns, final.st_ctime_ns)
        if digest.hexdigest() != expected_sha256 or final_identity != identity:
            raise RuntimeError("producer dependency SHA-256 or identity differs")
    finally:
        os.close(descriptor)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(data, str(path), "exec", dont_inherit=True), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


_injected = globals().pop("_ULLM_VERIFIED_MODULES", None)
if _injected is None:
    SELECTOR = load_verified_tool("aq4_p3_selector_for_producer", SELECTOR_PATH, SELECTOR_SHA256)
    PROFILER = load_verified_tool("aq4_p2_profiler_for_producer", PROFILER_PATH, PROFILER_SHA256)
elif isinstance(_injected, dict) and set(_injected) == {"selector", "profiler"} and all(isinstance(item, types.ModuleType) for item in _injected.values()):
    SELECTOR = _injected["selector"]
    PROFILER = _injected["profiler"]
else:
    raise RuntimeError("verified producer dependency injection differs")

INPUT_SCHEMA = "ullm.aq4_p3_selection_raw_producer_input.v1"
PROFILE_BINDING_SCHEMA = "ullm.aq4_p3_rocprof_run_binding.v1"
CAPABILITY_SCHEMA = "ullm.aq4_p3_rocprof_capture_capabilities.v1"
RAW_SCHEMA = SELECTOR.RAW_SCHEMA
MAX_INPUT_BYTES = 128 * 1024 * 1024
MAX_TRACE_ROWS = 500_000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

ROOT_FIELDS = {
    "schema_version",
    "status",
    "measurement_eligible",
    "smoke_only",
    "promotion_eligible",
    "manifest_sha256",
    "candidate",
    "identity",
    "resident_summaries",
    "representative_cases",
    "full_model_pairs",
}
CANDIDATE_FIELDS = {"candidate_id", "family"}
REF_FIELDS = {"path", "sha256"}
CASE_FIELDS = {
    "prompt_id",
    "case_id",
    "case_sha256",
    "resolved_m",
    "resident_raw",
    "profile_runs",
}
PROFILE_RUN_FIELDS = {
    "schema_version",
    "case_id",
    "case_sha256",
    "identity_sha256",
    "resident_run_index",
    "measurement_eligible",
    "clock_domain",
    "kernel_trace_complete",
    "hip_api_trace_complete",
    "capture_capabilities",
    "kernel_trace",
    "hip_api_trace",
}
PAIR_FIELDS = {
    "pair_id",
    "case_id",
    "case_sha256",
    "run_index",
    "baseline_raw",
    "candidate_raw",
}
RAW_ROOT_FIELDS = {
    "schema_version",
    "case_id",
    "case_sha256",
    "status",
    "immutable_status",
    "baseline_identity",
    "resident",
    "device_lock",
    "workload",
    "schedule",
    "runs",
    "terminal",
    "failure_reason",
    "links",
}
RUN_FIELDS = {
    "event",
    "schema_version",
    "resident_session_id",
    "case_id",
    "run_index",
    "run_kind",
    "status",
    "elapsed_ms",
    "requested_m",
    "resolved_m",
    "actual_token_batch_width",
    "actual_request_batch_width",
    "timing",
    "audit",
    "state",
    "lifecycle",
    "reset",
    "resource",
    "terminal",
}
BASELINE_IDENTITY_FIELDS = {"run_id", "kind", "identity_file"}
RESIDENT_FIELDS = {"session_id", "model_loads", "driver_identity", "case_reset_count"}
DEVICE_LOCK_FIELDS = {
    "schema_version", "path", "pid", "hostname", "run_id", "acquired_unix_ns", "driver"
}
DEVICE_LOCK_DRIVER_FIELDS = {"path", "sha256", "device", "inode", "nlink"}
WORKLOAD_FIELDS = {
    "scope", "phase", "mode", "prompt_tokens", "cached_prefix_tokens", "context_tokens",
    "prefill_requested_m", "resolved_m", "request_count", "generated_tokens",
}
AUDIT_FIELDS = {"coverage_complete", "deterministic_digest_sha256", "physical_operation_invocations"}
STATE_FIELDS = {"baseline_before", "baseline_after", "request_state_sha256"}
LIFECYCLE_FIELDS = {"prepare", "commit", "discard", "error", "cancel", "reset"}
RESET_FIELDS = {"attempted", "complete", "failed"}
RESOURCE_FIELDS = {"samples", "peak"}
RESOURCE_SAMPLE_FIELDS = {"monotonic_ms"}
RESOURCE_PEAK_FIELDS = {"vram_used_bytes", "workspace_bytes", "temporary_bytes"}
TERMINAL_FIELDS = {"reuse_forbidden", "reason_code", "oom", "hip_fault"}
RAW_TERMINAL_FIELDS = {"audit_digests", "reset_count", "all_resets_complete"}
CAPABILITY_FIELDS = {
    "schema_version", "status", "measurement_eligible", "capability_sha256", "tool",
    "domains", "rocprof_config",
}
CAPABILITY_TOOL_FIELDS = {"name", "version"}
CAPABILITY_DOMAIN_FIELDS = {
    "kernel_dispatch", "hip_api", "memory_copy", "d2h_memcpy", "stream_synchronize",
    "device_synchronize",
}
CAPABILITY_CONFIG_FIELDS = {
    "kernel_trace", "hip_api_trace", "memory_copy_trace", "marker_trace", "api_filter"
}

D2H_APIS = {"hipmemcpydtoh", "hipmemcpydtohasync"}
KNOWN_OTHER_MEMCPY_APIS = {
    "hipmemcpyhtod",
    "hipmemcpyhtodasync",
    "hipmemcpydtod",
    "hipmemcpydtodasync",
    "hipmemcpyhtoh",
    "hipmemcpyhtohasync",
    "hipmemcpy2d",
    "hipmemcpy2dasync",
    "hipmemcpy3d",
    "hipmemcpy3dasync",
    "hipmemcpypeer",
    "hipmemcpypeerasync",
}
SYNC_APIS = {"hipstreamsynchronize", "hipdevicesynchronize"}
KNOWN_OTHER_SYNC_APIS = {"hipeventsynchronize", "hipexternal_semaphoresignal", "hipexternal_semaphorewait"}


class ProducerError(ValueError):
    pass


@dataclass(frozen=True)
class Snapshot:
    path: Path
    identity: tuple[int, ...]
    sha256: str
    data: bytes

    def verify(self) -> None:
        try:
            current = self.path.lstat()
        except OSError as error:
            raise ProducerError(f"input disappeared: {self.path}: {error}") from error
        if file_identity(current) != self.identity:
            raise ProducerError(f"input identity changed: {self.path}")


def file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def capture(path: Path, label: str) -> Snapshot:
    if not path.is_absolute():
        raise ProducerError(f"{label} path must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except OSError as error:
            raise ProducerError(f"cannot inspect {label} path: {error}") from error
        if stat.S_ISLNK(info.st_mode):
            raise ProducerError(f"{label} path contains a symlink: {current}")
    path = path.resolve(strict=True)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ProducerError(f"{label} must be a regular file")
    if before.st_size > MAX_INPUT_BYTES:
        raise ProducerError(f"{label} exceeds {MAX_INPUT_BYTES} bytes")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if file_identity(opened) != file_identity(before):
            raise ProducerError(f"{label} identity changed while opening")
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            if size > MAX_INPUT_BYTES:
                raise ProducerError(f"{label} exceeds {MAX_INPUT_BYTES} bytes")
            chunks.append(chunk)
            digest.update(chunk)
        after_fd = os.fstat(descriptor)
        after_path = path.lstat()
        if (
            file_identity(after_fd) != file_identity(before)
            or file_identity(after_path) != file_identity(before)
        ):
            raise ProducerError(f"{label} identity changed while reading")
    finally:
        os.close(descriptor)
    return Snapshot(path, file_identity(before), digest.hexdigest(), b"".join(chunks))


def parse_json(snapshot: Snapshot, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ProducerError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            snapshot.data,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ProducerError(f"non-finite JSON in {label}: {token}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProducerError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ProducerError(f"{label} root must be an object")
    ensure_finite_tree(value, label)
    return value


def ensure_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProducerError(f"{label} contains a non-finite number")
    elif isinstance(value, dict):
        for key, child in value.items():
            ensure_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            ensure_finite_tree(child, f"{label}[{index}]")


def exact(value: dict[str, Any], fields: set[str], label: str) -> None:
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise ProducerError(f"{label} fields differ: missing={missing}, unknown={unknown}")


def digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ProducerError(f"{label} must be a lowercase SHA-256 digest")
    return value


def finite_float(value: Any, label: str, *, positive: bool = False) -> float:
    if type(value) is not float:
        raise ProducerError(f"{label} must be a finite float")
    if not math.isfinite(value) or (positive and value <= 0.0) or (not positive and value < 0.0):
        raise ProducerError(f"{label} has an invalid value")
    return value


def count(value: Any, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0):
        raise ProducerError(f"{label} must be a {'positive' if positive else 'non-negative'} integer")
    return value


def text_value(value: Any, label: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise ProducerError(f"{label} must be a {'non-empty ' if nonempty else ''}string")
    return value


def boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ProducerError(f"{label} must be a boolean")
    return value


def strict_json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(strict_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(strict_json_equal(a, b) for a, b in zip(left, right))
    return left == right


def validate_ref_shape(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ProducerError(f"{label} must be an object")
    exact(value, REF_FIELDS, label)
    text_value(value.get("path"), f"{label}.path")
    digest(value.get("sha256"), f"{label}.sha256")


def validate_complete_reset(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ProducerError(f"{label} must be an object")
    exact(value, RESET_FIELDS, label)
    expected = {"attempted": 1, "complete": 1, "failed": 0}
    for field, wanted in expected.items():
        observed = count(value.get(field), f"{label}.{field}")
        if observed != wanted:
            raise ProducerError(f"{label}.{field} differs")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")


def normalized_manifest(value: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(value, allow_nan=False))
    clone["manifest_sha256"] = None
    clone["resident_summaries"] = sorted(
        clone["resident_summaries"], key=lambda item: (item.get("sha256", ""), item.get("path", ""))
    )
    for case in clone["representative_cases"]:
        case["profile_runs"] = sorted(
            case["profile_runs"], key=lambda item: item.get("resident_run_index", -1)
        )
    clone["representative_cases"] = sorted(
        clone["representative_cases"],
        key=lambda item: (item.get("prompt_id", ""), item.get("case_sha256", "")),
    )
    clone["full_model_pairs"] = sorted(
        clone["full_model_pairs"],
        key=lambda item: (item.get("pair_id", ""), item.get("case_sha256", "")),
    )
    return clone


def manifest_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(normalized_manifest(value))).hexdigest()


def load_ref(value: Any, label: str, snapshots: list[Snapshot]) -> tuple[Snapshot, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ProducerError(f"{label} reference must be an object")
    exact(value, REF_FIELDS, f"{label} reference")
    expected = digest(value["sha256"], f"{label}.sha256")
    path = value["path"]
    if not isinstance(path, str):
        raise ProducerError(f"{label}.path must be a string")
    snapshot = capture(Path(path), label)
    if snapshot.sha256 != expected:
        raise ProducerError(f"{label} SHA-256 differs")
    snapshots.append(snapshot)
    return snapshot, parse_json(snapshot, label)


def load_csv_ref(value: Any, label: str, snapshots: list[Snapshot]) -> Snapshot:
    if not isinstance(value, dict):
        raise ProducerError(f"{label} reference must be an object")
    exact(value, REF_FIELDS, f"{label} reference")
    expected = digest(value["sha256"], f"{label}.sha256")
    if not isinstance(value["path"], str):
        raise ProducerError(f"{label}.path must be a string")
    snapshot = capture(Path(value["path"]), label)
    if snapshot.sha256 != expected:
        raise ProducerError(f"{label} SHA-256 differs")
    snapshots.append(snapshot)
    return snapshot


def self_hash(value: dict[str, Any], field: str) -> str:
    clone = json.loads(json.dumps(value, allow_nan=False))
    clone[field] = None
    return hashlib.sha256(canonical(clone)).hexdigest()


def validate_identity(value: dict[str, Any], snapshot: Snapshot) -> dict[str, str]:
    if value.get("schema_version") != "ullm.aq4_production_p2_identity.v2" or value.get("status") != "bound":
        raise ProducerError("identity schema/status differs")
    identity_sha = digest(value.get("identity_sha256"), "identity.identity_sha256")
    if identity_sha != self_hash(value, "identity_sha256"):
        raise ProducerError("identity self-hash differs")
    resident = value.get("resident_driver_identity")
    binding = value.get("hash_binding")
    if not isinstance(resident, dict) or not isinstance(binding, dict):
        raise ProducerError("identity resident/hash binding is incomplete")
    binary_sha = digest(resident.get("binary_sha256"), "identity resident binary")
    package_sha = digest(binding.get("package_content_sha256"), "identity package content")
    case_manifest_sha = digest(
        binding.get("bound_case_manifest_sha256"), "identity case manifest"
    )
    if value.get("expanded_manifest_sha256") != case_manifest_sha:
        raise ProducerError("identity expanded/case manifest hash differs")
    return {
        "identity_sha256": identity_sha,
        "case_manifest_sha256": case_manifest_sha,
        "binary_sha256": binary_sha,
        "package_content_sha256": package_sha,
        "identity_file_sha256": snapshot.sha256,
        "identity_path": str(snapshot.path),
        "_resident_driver_identity": resident,
    }


def validate_summary(
    value: dict[str, Any], snapshot: Snapshot, identity: dict[str, str], mode: str
) -> str:
    if value.get("schema_version") != "ullm.aq4_p2_resident_batch.v1" or value.get("status") != "complete":
        raise ProducerError("resident summary schema/status differs")
    if (
        type(value.get("warmup_runs")) is not int
        or value["warmup_runs"] != 2
        or type(value.get("measured_runs")) is not int
        or value["measured_runs"] != 10
    ):
        raise ProducerError("resident summary schedule differs")
    case_count = count(value.get("case_count"), "resident summary case_count", positive=True)
    if count(value.get("completed_cases"), "resident summary completed_cases") != case_count:
        raise ProducerError("resident summary is incomplete")
    baseline = value.get("baseline_identity")
    if not isinstance(baseline, dict) or baseline.get("identity_file") != {
        "path": identity["identity_path"],
        "sha256": identity["identity_file_sha256"],
    }:
        raise ProducerError("resident summary identity link differs")
    run_id = baseline.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ProducerError("resident summary run_id is invalid")
    smoke = value.get("smoke_only") is True or value.get("execution_mode") == "one_case_smoke"
    explicitly_ineligible = value.get("measurement_eligible") is False
    promotion_false = value.get("promotion_eligible") is False
    if mode == "promotion" and (smoke or explicitly_ineligible or promotion_false):
        raise ProducerError("smoke/ineligible resident summary cannot produce promotion raw")
    if mode == "diagnostic" and not (smoke and promotion_false):
        raise ProducerError("diagnostic resident summary must be smoke-only and promotion-ineligible")
    return run_id


def validate_capture_capabilities(value: dict[str, Any], mode: str) -> None:
    exact(value, CAPABILITY_FIELDS, "capture capabilities")
    if value.get("schema_version") != CAPABILITY_SCHEMA or value.get("status") != "complete":
        raise ProducerError("capture capabilities schema/status differs")
    expected_eligible = mode == "promotion"
    if (
        boolean(
            value.get("measurement_eligible"),
            "capture capabilities.measurement_eligible",
        )
        is not expected_eligible
    ):
        raise ProducerError("capture capabilities measurement eligibility differs")
    declared = digest(value.get("capability_sha256"), "capture capabilities.capability_sha256")
    if declared != self_hash(value, "capability_sha256"):
        raise ProducerError("capture capabilities self-hash differs")
    tool = value.get("tool")
    if not isinstance(tool, dict):
        raise ProducerError("capture capabilities.tool must be an object")
    exact(tool, CAPABILITY_TOOL_FIELDS, "capture capabilities.tool")
    if tool.get("name") != "rocprofv3" or not text_value(
        tool.get("version"), "capture capabilities.tool.version"
    ):
        raise ProducerError("capture capabilities tool differs")
    domains = value.get("domains")
    if not isinstance(domains, dict):
        raise ProducerError("capture capabilities.domains must be an object")
    exact(domains, CAPABILITY_DOMAIN_FIELDS, "capture capabilities.domains")
    for field in sorted(CAPABILITY_DOMAIN_FIELDS):
        if boolean(domains.get(field), f"capture capabilities.domains.{field}") is not True:
            raise ProducerError(f"capture capabilities domain is incomplete: {field}")
    config = value.get("rocprof_config")
    if not isinstance(config, dict):
        raise ProducerError("capture capabilities.rocprof_config must be an object")
    exact(config, CAPABILITY_CONFIG_FIELDS, "capture capabilities.rocprof_config")
    if (
        boolean(
            config.get("kernel_trace"),
            "capture capabilities.rocprof_config.kernel_trace",
        )
        is not True
        or boolean(
            config.get("hip_api_trace"),
            "capture capabilities.rocprof_config.hip_api_trace",
        )
        is not True
        or boolean(
            config.get("memory_copy_trace"),
            "capture capabilities.rocprof_config.memory_copy_trace",
        )
        is not True
        or boolean(
            config.get("marker_trace"),
            "capture capabilities.rocprof_config.marker_trace",
        )
        is not True
        or config.get("api_filter") != "all_functions"
    ):
        raise ProducerError("capture capabilities rocprof configuration is incomplete")


def validate_raw(
    value: dict[str, Any], identity: dict[str, str], summaries: dict[str, Snapshot], mode: str
) -> tuple[str, list[dict[str, Any]]]:
    expected = set(RAW_ROOT_FIELDS)
    smoke_fields = {"execution_mode", "smoke_only", "promotion_eligible"}
    if smoke_fields & set(value):
        expected |= smoke_fields
    exact(value, expected, "resident raw")
    if value.get("schema_version") != "ullm.aq4_p2_resident_batch_raw.v1":
        raise ProducerError("resident raw schema differs")
    if (
        value.get("status") != "ok"
        or value.get("immutable_status") is not False
        or value.get("failure_reason") is not None
    ):
        raise ProducerError("resident raw is not a complete successful measurement")
    text_value(value.get("case_id"), "resident raw.case_id")
    digest(value.get("case_sha256"), "resident raw.case_sha256")
    boolean(value.get("immutable_status"), "resident raw.immutable_status")
    baseline = value.get("baseline_identity")
    if not isinstance(baseline, dict):
        raise ProducerError("resident raw baseline_identity must be an object")
    exact(baseline, BASELINE_IDENTITY_FIELDS, "resident raw baseline_identity")
    validate_ref_shape(
        baseline.get("identity_file"), "resident raw baseline_identity.identity_file"
    )
    if not strict_json_equal(
        baseline.get("identity_file"),
        {
            "path": identity["identity_path"],
            "sha256": identity["identity_file_sha256"],
        },
    ):
        raise ProducerError("resident raw identity link differs")
    run_id = text_value(baseline.get("run_id"), "resident raw baseline_identity.run_id")
    text_value(baseline.get("kind"), "resident raw baseline_identity.kind")
    if run_id not in summaries:
        raise ProducerError("resident raw has no bound complete summary")
    resident = value.get("resident")
    if not isinstance(resident, dict):
        raise ProducerError("resident raw resident must be an object")
    exact(resident, RESIDENT_FIELDS, "resident raw resident")
    text_value(resident.get("session_id"), "resident raw resident.session_id")
    if (
        count(
            resident.get("model_loads"),
            "resident raw resident.model_loads",
            positive=True,
        )
        != 1
        or count(resident.get("case_reset_count"), "resident raw resident.case_reset_count") != 12
        or not strict_json_equal(resident.get("driver_identity"), identity["_resident_driver_identity"])
    ):
        raise ProducerError("resident raw artifact identity differs")
    lock = value.get("device_lock")
    if not isinstance(lock, dict):
        raise ProducerError("resident raw device_lock must be an object")
    exact(lock, DEVICE_LOCK_FIELDS, "resident raw device_lock")
    if (
        lock.get("schema_version") != "ullm.aq4_p2_device_lock_owner.v1"
        or lock.get("run_id") != run_id
    ):
        raise ProducerError("resident raw device_lock binding differs")
    for field in ("path", "hostname"):
        text_value(lock.get(field), f"resident raw device_lock.{field}")
    for field in ("pid", "acquired_unix_ns"):
        count(lock.get(field), f"resident raw device_lock.{field}", positive=True)
    driver = lock.get("driver")
    if not isinstance(driver, dict):
        raise ProducerError("resident raw device_lock.driver must be an object")
    exact(driver, DEVICE_LOCK_DRIVER_FIELDS, "resident raw device_lock.driver")
    text_value(driver.get("path"), "resident raw device_lock.driver.path")
    digest(driver.get("sha256"), "resident raw device_lock.driver.sha256")
    for field in ("device", "inode", "nlink"):
        count(driver.get(field), f"resident raw device_lock.driver.{field}", positive=True)
    if driver["sha256"] != identity["binary_sha256"]:
        raise ProducerError("resident raw device_lock driver SHA differs")
    workload = value.get("workload")
    if not isinstance(workload, dict):
        raise ProducerError("resident raw workload must be an object")
    exact(workload, WORKLOAD_FIELDS, "resident raw workload")
    for field in ("scope", "phase", "mode"):
        text_value(workload.get(field), f"resident raw workload.{field}")
    for field in (
        "prompt_tokens",
        "context_tokens",
        "prefill_requested_m",
        "resolved_m",
        "request_count",
    ):
        count(workload.get(field), f"resident raw workload.{field}", positive=True)
    for field in ("cached_prefix_tokens", "generated_tokens"):
        count(workload.get(field), f"resident raw workload.{field}")
    if workload["scope"] != "full_model" or workload["phase"] != "cold_prefill":
        raise ProducerError("resident raw workload scope/phase differs")
    links = value.get("links")
    expected_links = {"fixture", "identity", "policy"} | (
        {"live_preflight"}
        if mode == "diagnostic" and "live_preflight" in (links or {})
        else set()
    )
    if not isinstance(links, dict):
        raise ProducerError("resident raw links must be an object")
    exact(links, expected_links, "resident raw links")
    for field in expected_links:
        validate_ref_shape(links.get(field), f"resident raw links.{field}")
    if not strict_json_equal(links["identity"], baseline["identity_file"]):
        raise ProducerError("resident raw links.identity differs")
    smoke = value.get("smoke_only") is True or value.get("execution_mode") == "one_case_smoke"
    for field in smoke_fields & set(value):
        if field == "execution_mode":
            text_value(value[field], "resident raw execution_mode")
        else:
            boolean(value[field], f"resident raw {field}")
    if mode == "promotion" and (smoke or value.get("promotion_eligible") is False):
        raise ProducerError("smoke/ineligible resident raw cannot produce promotion raw")
    if mode == "diagnostic" and not (smoke and value.get("promotion_eligible") is False):
        raise ProducerError("diagnostic resident raw must be smoke-only and promotion-ineligible")
    schedule = value.get("schedule")
    if (
        not isinstance(schedule, dict)
        or set(schedule) != {"warmup_runs", "measured_runs", "completed_runs"}
        or type(schedule["warmup_runs"]) is not int
        or schedule["warmup_runs"] != 2
        or type(schedule["measured_runs"]) is not int
        or schedule["measured_runs"] != 10
        or type(schedule["completed_runs"]) is not int
        or schedule["completed_runs"] != 12
    ):
        raise ProducerError("resident raw schedule differs")
    runs = value.get("runs")
    if not isinstance(runs, list) or len(runs) != 12:
        raise ProducerError("resident raw must contain 12 runs")
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ProducerError(f"resident raw run {index} must be an object")
        exact(run, RUN_FIELDS, f"resident raw run {index}")
        expected_kind = "warmup" if index < 2 else "measured"
        if (
            type(run.get("run_index")) is not int
            or run["run_index"] != index
            or run.get("run_kind") != expected_kind
            or run.get("status") != "ok"
            or run.get("case_id") != value.get("case_id")
        ):
            raise ProducerError(f"resident raw run order/status differs at {index}")
        for field in (
            "event",
            "schema_version",
            "resident_session_id",
            "case_id",
            "run_kind",
            "status",
        ):
            text_value(run.get(field), f"resident raw run {index}.{field}")
        if (
            run["event"] != "run_complete"
            or run["schema_version"] != "ullm.aq4_p2_resident_driver.v2"
            or run["resident_session_id"] != resident["session_id"]
        ):
            raise ProducerError(f"resident raw run protocol/session differs at {index}")
        finite_float(
            run.get("elapsed_ms"),
            f"resident raw run {index}.elapsed_ms",
            positive=True,
        )
        timing = run.get("timing")
        if not isinstance(timing, dict):
            raise ProducerError(f"resident raw run timing is missing at {index}")
        exact(
            timing,
            {"prefill_ms", "decode_ms", "end_to_end_ms", "generated_tokens"},
            f"resident raw run {index}.timing",
        )
        finite_float(
            timing.get("prefill_ms"),
            f"resident raw run {index} prefill_ms",
            positive=True,
        )
        finite_float(timing.get("decode_ms"), f"resident raw run {index} decode_ms")
        finite_float(
            timing.get("end_to_end_ms"),
            f"resident raw run {index} end_to_end_ms",
            positive=True,
        )
        count(timing.get("generated_tokens"), f"resident raw run {index} generated_tokens")
        for field in (
            "requested_m",
            "resolved_m",
            "actual_token_batch_width",
            "actual_request_batch_width",
        ):
            count(run.get(field), f"resident raw run {index}.{field}", positive=True)
        if (
            run["requested_m"] != workload["prefill_requested_m"]
            or run["resolved_m"] != workload["resolved_m"]
            or run["actual_token_batch_width"] != workload["resolved_m"]
            or run["actual_request_batch_width"] != workload["request_count"]
        ):
            raise ProducerError(f"resident raw run width differs at {index}")
        audit = run.get("audit")
        if not isinstance(audit, dict):
            raise ProducerError(f"resident raw run {index}.audit must be an object")
        exact(audit, AUDIT_FIELDS, f"resident raw run {index}.audit")
        if (
            boolean(
                audit.get("coverage_complete"),
                f"resident raw run {index}.audit.coverage_complete",
            )
            is not True
        ):
            raise ProducerError(f"resident raw run audit coverage differs at {index}")
        digest(audit.get("deterministic_digest_sha256"), f"resident raw run {index}.audit digest")
        count(
            audit.get("physical_operation_invocations"),
            f"resident raw run {index}.audit physical operations",
            positive=True,
        )
        state = run.get("state")
        if not isinstance(state, dict):
            raise ProducerError(f"resident raw run {index}.state must be an object")
        exact(state, STATE_FIELDS, f"resident raw run {index}.state")
        if any(
            boolean(state.get(field), f"resident raw run {index}.state.{field}")
            is not True
            for field in ("baseline_before", "baseline_after")
        ):
            raise ProducerError(f"resident raw run state baseline differs at {index}")
        digest(state.get("request_state_sha256"), f"resident raw run {index}.state digest")
        lifecycle = run.get("lifecycle")
        if not isinstance(lifecycle, dict):
            raise ProducerError(f"resident raw run {index}.lifecycle must be an object")
        exact(lifecycle, LIFECYCLE_FIELDS, f"resident raw run {index}.lifecycle")
        expected_lifecycle = {
            "prepare": 1,
            "commit": 1,
            "discard": 0,
            "error": 0,
            "cancel": 0,
        }
        for field, wanted in expected_lifecycle.items():
            if count(lifecycle.get(field), f"resident raw run {index}.lifecycle.{field}") != wanted:
                raise ProducerError(f"resident raw run lifecycle differs at {index}")
        validate_complete_reset(
            lifecycle.get("reset"), f"resident raw run {index}.lifecycle.reset"
        )
        validate_complete_reset(run.get("reset"), f"resident raw run {index}.reset")
        resource = run.get("resource")
        if not isinstance(resource, dict):
            raise ProducerError(f"resident raw run {index}.resource must be an object")
        exact(resource, RESOURCE_FIELDS, f"resident raw run {index}.resource")
        samples = resource.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ProducerError(f"resident raw run {index}.resource.samples must be non-empty")
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ProducerError(f"resident raw run {index}.resource sample must be an object")
            sample_label = f"resident raw run {index}.resource.samples[{sample_index}]"
            exact(sample, RESOURCE_SAMPLE_FIELDS, sample_label)
            finite_float(sample.get("monotonic_ms"), f"{sample_label}.monotonic_ms")
        peak = resource.get("peak")
        if not isinstance(peak, dict):
            raise ProducerError(f"resident raw run {index}.resource.peak must be an object")
        exact(peak, RESOURCE_PEAK_FIELDS, f"resident raw run {index}.resource.peak")
        for field in RESOURCE_PEAK_FIELDS:
            if peak[field] is not None:
                count(peak[field], f"resident raw run {index}.resource.peak.{field}")
        terminal = run.get("terminal")
        if not isinstance(terminal, dict):
            raise ProducerError(f"resident raw run {index}.terminal must be an object")
        exact(terminal, TERMINAL_FIELDS, f"resident raw run {index}.terminal")
        if any(
            boolean(terminal.get(field), f"resident raw run {index}.terminal.{field}")
            for field in ("reuse_forbidden", "oom", "hip_fault")
        ):
            raise ProducerError(f"resident raw run terminal flags differ at {index}")
        if terminal.get("reason_code") != "none":
            raise ProducerError(f"resident raw run terminal reason differs at {index}")
    raw_terminal = value.get("terminal")
    if not isinstance(raw_terminal, dict):
        raise ProducerError("resident raw terminal must be an object")
    exact(raw_terminal, RAW_TERMINAL_FIELDS, "resident raw terminal")
    digests = raw_terminal.get("audit_digests")
    if not isinstance(digests, list) or len(digests) != 12:
        raise ProducerError("resident raw terminal audit_digests differs")
    for index, item in enumerate(digests):
        digest(item, f"resident raw terminal.audit_digests[{index}]")
        if item != runs[index]["audit"]["deterministic_digest_sha256"]:
            raise ProducerError("resident raw terminal audit digest binding differs")
    if (
        count(raw_terminal.get("reset_count"), "resident raw terminal.reset_count") != 12
        or boolean(
            raw_terminal.get("all_resets_complete"),
            "resident raw terminal.all_resets_complete",
        )
        is not True
    ):
        raise ProducerError("resident raw terminal reset summary differs")
    return run_id, runs


def _column(fieldnames: list[str], aliases: tuple[str, ...], label: str) -> str:
    matches = [name for name in aliases if name in fieldnames]
    if len(matches) != 1:
        raise ProducerError(f"trace must have exactly one {label} column, got {matches}")
    return matches[0]


def _csv(snapshot: Snapshot, label: str) -> tuple[csv.DictReader, list[str]]:
    try:
        text = snapshot.data.decode("utf-8-sig")
    except UnicodeError as error:
        raise ProducerError(f"{label} is not UTF-8: {error}") from error
    reader = csv.DictReader(text.splitlines())
    fields = reader.fieldnames
    if not fields or len(fields) != len(set(fields)):
        raise ProducerError(f"{label} header is missing or duplicated")
    return reader, fields


def parse_kernel_trace(snapshot: Snapshot, candidate_id: str) -> dict[str, int]:
    reader, fields = _csv(snapshot, "kernel trace")
    dispatch_col = _column(fields, ("Dispatch_Id", "Dispatch_ID", "Index", "dispatch_id"), "dispatch id")
    name_col = _column(fields, ("Kernel_Name", "KernelName", "Name", "kernel_name"), "kernel name")
    start_col = _column(fields, ("Start_Timestamp", "BeginNs", "start_ns"), "start timestamp")
    end_col = _column(fields, ("End_Timestamp", "EndNs", "end_ns"), "end timestamp")
    phase_col = _column(fields, ("Phase", "phase"), "phase")
    intervals: list[Any] = []
    dispatches: set[str] = set()
    previous = -1
    for line, row in enumerate(reader, 2):
        if len(intervals) >= MAX_TRACE_ROWS:
            raise ProducerError("kernel trace row limit exceeded")
        if None in row:
            raise ProducerError(f"kernel trace row {line} has extra fields")
        dispatch = (row.get(dispatch_col) or "").strip()
        name = (row.get(name_col) or "").strip()
        phase = (row.get(phase_col) or "").strip().lower()
        if not dispatch or dispatch in dispatches or not name or phase != "prefill":
            raise ProducerError(f"kernel trace row {line} identity/phase differs")
        dispatches.add(dispatch)
        try:
            start = int((row.get(start_col) or "").strip())
            end = int((row.get(end_col) or "").strip())
        except ValueError as error:
            raise ProducerError(f"kernel trace row {line} clock is invalid") from error
        if start < 0 or end <= start or start < previous or end > (1 << 63) - 1:
            raise ProducerError(f"kernel trace row {line} interval/order is invalid")
        previous = start
        try:
            family = PROFILER.classify_kernel(name)
        except PROFILER.ProfileError as error:
            raise ProducerError(f"kernel family classification failed: {error}") from error
        if family is None:
            raise ProducerError(f"unknown kernel family: {name}")
        intervals.append(PROFILER.KernelInterval(dispatch, name, start, end, family, phase))
    if not intervals:
        raise ProducerError("kernel trace is empty")
    try:
        aggregate = PROFILER.aggregate(intervals)
    except PROFILER.ProfileError as error:
        raise ProducerError(f"kernel interval aggregation failed: {error}") from error
    family = SELECTOR.CANDIDATES[candidate_id]["family"]
    families = ("attention", "recurrent") if family == "attention_recurrent" else (family,)
    exclusive = sum(aggregate["families"][item]["exclusive_ns"] for item in families)
    return {
        "candidate_exclusive_ns": exclusive,
        "gpu_total_union_ns": aggregate["gpu_total_union_ns"],
        "cross_family_overlap_ns": aggregate["cross_family_overlap_ns"],
    }


def union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    total = 0
    left, right = ordered[0]
    for start, end in ordered[1:]:
        if start > right:
            total += right - left
            left, right = start, end
        else:
            right = max(right, end)
    return total + right - left


def normalized_api_name(value: str) -> str:
    return value.strip().split("(", 1)[0].split("::")[-1].replace("_", "").lower()


def parse_hip_api_trace(
    snapshot: Snapshot, capture_capabilities: dict[str, Any] | None = None
) -> dict[str, int]:
    if capture_capabilities is None:
        raise ProducerError("HIP API zero observations require complete capture capabilities")
    domains = capture_capabilities.get("domains")
    config = capture_capabilities.get("rocprof_config")
    if (
        not isinstance(domains, dict)
        or any(domains.get(field) is not True for field in CAPABILITY_DOMAIN_FIELDS)
        or not isinstance(config, dict)
        or config.get("hip_api_trace") is not True
        or config.get("api_filter") != "all_functions"
    ):
        raise ProducerError("HIP API trace lacks complete capture domain proof")
    reader, fields = _csv(snapshot, "HIP API trace")
    correlation_col = _column(
        fields, ("Correlation_Id", "Correlation_ID", "Index", "correlation_id"), "correlation id"
    )
    name_col = _column(fields, ("Function", "Api_Name", "API_Name", "Name", "function"), "API name")
    start_col = _column(fields, ("Start_Timestamp", "BeginNs", "start_ns"), "start timestamp")
    end_col = _column(fields, ("End_Timestamp", "EndNs", "end_ns"), "end timestamp")
    seen: set[str] = set()
    d2h: list[tuple[int, int]] = []
    sync: list[tuple[int, int]] = []
    previous = -1
    rows = 0
    for line, row in enumerate(reader, 2):
        rows += 1
        if rows > MAX_TRACE_ROWS or None in row:
            raise ProducerError(f"HIP API trace row {line} is invalid")
        correlation = (row.get(correlation_col) or "").strip()
        raw_name = (row.get(name_col) or "").strip()
        if not correlation or correlation in seen or not raw_name:
            raise ProducerError(f"HIP API trace row {line} identity differs")
        seen.add(correlation)
        try:
            start = int((row.get(start_col) or "").strip())
            end = int((row.get(end_col) or "").strip())
        except ValueError as error:
            raise ProducerError(f"HIP API trace row {line} clock is invalid") from error
        if start < 0 or end <= start or start < previous or end > (1 << 63) - 1:
            raise ProducerError(f"HIP API trace row {line} interval/order is invalid")
        previous = start
        name = normalized_api_name(raw_name)
        if name in D2H_APIS:
            d2h.append((start, end))
        elif name in SYNC_APIS:
            sync.append((start, end))
        elif name in KNOWN_OTHER_MEMCPY_APIS or name in KNOWN_OTHER_SYNC_APIS:
            continue
        elif "memcpy" in name or "synchron" in name:
            raise ProducerError(f"unknown transfer/synchronization HIP API: {raw_name}")
    if rows == 0:
        raise ProducerError("HIP API trace is empty; zero counts are not observable")
    return {
        "d2h_count": len(d2h),
        "d2h_union_ns": union_ns(d2h),
        "stream_sync_count": len(sync),
        "stream_sync_union_ns": union_ns(sync),
    }


def stable_mean(values: list[float]) -> float:
    return math.fsum(sorted(values)) / len(values)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return math.fsum((ordered[middle - 1], ordered[middle])) / 2.0


def stats(values: list[float]) -> tuple[float, float, float]:
    if len(values) != 10:
        raise ProducerError("measurement statistics require exactly 10 values")
    try:
        mean = stable_mean(values)
        squared = sorted((value - mean) ** 2 for value in values)
        variance = math.fsum(squared) / 9
        standard_deviation = math.sqrt(variance)
        cv = standard_deviation / mean
        halfwidth = SELECTOR.T_CRITICAL_975[9] * standard_deviation / math.sqrt(10)
    except (OverflowError, ValueError, ZeroDivisionError) as error:
        raise ProducerError("measurement statistics overflowed") from error
    if not all(math.isfinite(item) for item in (mean, variance, standard_deviation, cv, halfwidth)):
        raise ProducerError("measurement statistics are non-finite")
    return median(values), cv, halfwidth


def trace_measurement(
    case: dict[str, Any],
    raw_runs: list[dict[str, Any]],
    candidate_id: str,
    identity_sha: str,
    mode: str,
    snapshots: list[Snapshot],
    used_kernel_traces: set[str],
    used_api_traces: set[str],
) -> dict[str, Any]:
    profile_runs = case["profile_runs"]
    if not isinstance(profile_runs, list):
        raise ProducerError("representative profile_runs must be an array")
    required_indices = (
        list(range(2, 12)) if mode == "promotion" or len(profile_runs) == 10 else None
    )
    if mode == "promotion" and len(profile_runs) != 10:
        raise ProducerError("promotion representative case requires 10 profile runs")
    if mode == "diagnostic" and len(profile_runs) not in {1, 10}:
        raise ProducerError("diagnostic representative case requires one or 10 profile runs")
    observed_indices: list[int] = []
    exclusive_ms: list[float] = []
    d2h_count = 0
    d2h_times_ms: list[float] = []
    sync_count = 0
    sync_times_ms: list[float] = []
    for index, binding in enumerate(
        sorted(profile_runs, key=lambda item: item.get("resident_run_index", -1) if isinstance(item, dict) else -1)
    ):
        label = f"profile_runs[{index}]"
        if not isinstance(binding, dict):
            raise ProducerError(f"{label} must be an object")
        exact(binding, PROFILE_RUN_FIELDS, label)
        if binding["schema_version"] != PROFILE_BINDING_SCHEMA:
            raise ProducerError(f"{label} schema differs")
        if (
            binding["case_id"] != case["case_id"]
            or binding["case_sha256"] != case["case_sha256"]
            or binding["identity_sha256"] != identity_sha
            or binding["clock_domain"] != "rocprofv3_monotonic_ns"
            or binding["kernel_trace_complete"] is not True
            or binding["hip_api_trace_complete"] is not True
        ):
            raise ProducerError(f"{label} case/identity/clock binding differs")
        eligible = binding["measurement_eligible"]
        if (
            not isinstance(eligible, bool)
            or (mode == "promotion" and not eligible)
            or (mode == "diagnostic" and eligible)
        ):
            raise ProducerError(f"{label} measurement eligibility differs")
        run_index = count(binding["resident_run_index"], f"{label}.resident_run_index")
        if run_index < 2 or run_index > 11:
            raise ProducerError(f"{label} does not bind a measured resident run")
        observed_indices.append(run_index)
        if raw_runs[run_index]["case_id"] != case["case_id"]:
            raise ProducerError(f"{label} resident run pairing differs")
        _capability_snapshot, capability = load_ref(
            binding["capture_capabilities"], f"{label} capture capabilities", snapshots
        )
        validate_capture_capabilities(capability, mode)
        kernel = load_csv_ref(binding["kernel_trace"], f"{label} kernel trace", snapshots)
        api = load_csv_ref(binding["hip_api_trace"], f"{label} HIP API trace", snapshots)
        if kernel.sha256 in used_kernel_traces or api.sha256 in used_api_traces:
            raise ProducerError("rocprof kernel or HIP API trace was reused")
        used_kernel_traces.add(kernel.sha256)
        used_api_traces.add(api.sha256)
        kernel_value = parse_kernel_trace(kernel, candidate_id)
        api_value = parse_hip_api_trace(api, capability)
        exclusive_ms.append(kernel_value["candidate_exclusive_ns"] / 1_000_000.0)
        d2h_count += api_value["d2h_count"]
        d2h_times_ms.append(api_value["d2h_union_ns"] / 1_000_000.0)
        sync_count += api_value["stream_sync_count"]
        sync_times_ms.append(api_value["stream_sync_union_ns"] / 1_000_000.0)
    if len(observed_indices) != len(set(observed_indices)):
        raise ProducerError("profile run binding contains duplicate resident run indices")
    if required_indices is not None and sorted(observed_indices) != required_indices:
        raise ProducerError("profile runs do not cover measured resident indices 2..11")
    baseline_values = [
        finite_float(run["timing"]["prefill_ms"], "measured prefill_ms", positive=True)
        for run in raw_runs[2:]
    ]
    p50, cv, ci_halfwidth = stats(baseline_values)
    return {
        "baseline_p50_ms": p50,
        "baseline_cv": cv,
        "ci95_halfwidth_ms": ci_halfwidth,
        "recoverable_family_exclusive_ms": median(exclusive_ms),
        "d2h_count": d2h_count,
        "d2h_time_ms": math.fsum(sorted(d2h_times_ms)),
        "stream_sync_count": sync_count,
        "stream_sync_time_ms": math.fsum(sorted(sync_times_ms)),
    }


def build(manifest: dict[str, Any], manifest_snapshot: Snapshot) -> tuple[dict[str, Any], list[Snapshot]]:
    exact(manifest, ROOT_FIELDS, "producer manifest")
    if manifest["schema_version"] != INPUT_SCHEMA:
        raise ProducerError("producer manifest schema differs")
    declared_manifest_sha = digest(manifest["manifest_sha256"], "manifest_sha256")
    if declared_manifest_sha != manifest_sha256(manifest):
        raise ProducerError("producer manifest semantic SHA-256 differs")
    status = manifest["status"]
    if status == "promotion_ready":
        mode = "promotion"
        expected_flags = (True, False, True)
    elif status == "one_case_diagnostic":
        mode = "diagnostic"
        expected_flags = (False, True, False)
    else:
        raise ProducerError("producer manifest status differs")
    flag_names = ("measurement_eligible", "smoke_only", "promotion_eligible")
    if any(type(manifest[field]) is not bool for field in flag_names):
        raise ProducerError("producer manifest eligibility flags must be boolean")
    flags = tuple(manifest[field] for field in flag_names)
    if flags != expected_flags:
        raise ProducerError("producer manifest eligibility flags differ")

    candidate = manifest["candidate"]
    if not isinstance(candidate, dict):
        raise ProducerError("candidate must be an object")
    exact(candidate, CANDIDATE_FIELDS, "candidate")
    candidate_id = candidate["candidate_id"]
    if candidate_id not in SELECTOR.CANDIDATES or candidate["family"] != SELECTOR.CANDIDATES[candidate_id]["family"]:
        raise ProducerError("candidate ID/family differs from selector policy")

    snapshots: list[Snapshot] = [manifest_snapshot]
    identity_snapshot, identity_value = load_ref(manifest["identity"], "identity", snapshots)
    identity = validate_identity(identity_value, identity_snapshot)

    summary_refs = manifest["resident_summaries"]
    if not isinstance(summary_refs, list) or not summary_refs:
        raise ProducerError("resident_summaries must be a non-empty array")
    summaries: dict[str, Snapshot] = {}
    for index, ref in enumerate(summary_refs):
        snapshot, value = load_ref(ref, f"resident summary {index}", snapshots)
        run_id = validate_summary(value, snapshot, identity, mode)
        if run_id in summaries:
            raise ProducerError("duplicate resident summary run_id")
        summaries[run_id] = snapshot

    cases = manifest["representative_cases"]
    expected_case_count = 7 if mode == "promotion" else 1
    if not isinstance(cases, list) or len(cases) != expected_case_count:
        raise ProducerError(f"{mode} requires {expected_case_count} representative cases")
    prompt_ids: set[str] = set()
    case_hashes: set[str] = set()
    measurements: list[dict[str, Any]] = []
    raw_cache: dict[str, tuple[dict[str, Any], str, list[dict[str, Any]]]] = {}
    used_kernel_traces: set[str] = set()
    used_api_traces: set[str] = set()

    def resident_raw(ref: dict[str, Any], label: str) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        key = json.dumps(ref, sort_keys=True)
        if key not in raw_cache:
            snapshot, value = load_ref(ref, label, snapshots)
            run_id, runs = validate_raw(value, identity, summaries, mode)
            raw_cache[key] = (value, run_id, runs)
        return raw_cache[key]

    for index, case in enumerate(
        sorted(cases, key=lambda item: (item.get("prompt_id", ""), item.get("case_sha256", "")) if isinstance(item, dict) else ("", ""))
    ):
        label = f"representative_cases[{index}]"
        if not isinstance(case, dict):
            raise ProducerError(f"{label} must be an object")
        exact(case, CASE_FIELDS, label)
        prompt_id = case["prompt_id"]
        if not isinstance(prompt_id, str) or not prompt_id or prompt_id in prompt_ids:
            raise ProducerError("representative prompt IDs are invalid or duplicated")
        prompt_ids.add(prompt_id)
        case_sha = digest(case["case_sha256"], f"{label}.case_sha256")
        if case_sha in case_hashes:
            raise ProducerError("representative case SHA-256 values are duplicated")
        case_hashes.add(case_sha)
        resolved_m = count(case["resolved_m"], f"{label}.resolved_m", positive=True)
        raw, _run_id, runs = resident_raw(case["resident_raw"], f"{label} resident raw")
        workload = raw.get("workload")
        if (
            raw.get("case_id") != case["case_id"]
            or raw.get("case_sha256") != case_sha
            or not isinstance(workload, dict)
            or workload.get("resolved_m") != resolved_m
            or workload.get("phase") != "cold_prefill"
            or workload.get("scope") != "full_model"
        ):
            raise ProducerError(f"{label} resident raw case/workload differs")
        if type(workload.get("resolved_m")) is not int:
            raise ProducerError(f"{label} resident raw resolved_m type differs")
        observed = trace_measurement(
            case,
            runs,
            candidate_id,
            identity["identity_sha256"],
            mode,
            snapshots,
            used_kernel_traces,
            used_api_traces,
        )
        measurements.append(
            {
                "candidate_id": candidate_id,
                "family": candidate["family"],
                "prompt_id": prompt_id,
                "case_sha256": case_sha,
                "identity_sha256": identity["identity_sha256"],
                "resolved_m": resolved_m,
                **observed,
            }
        )
    if mode == "promotion" and (
        not any(row["resolved_m"] == 128 for row in measurements)
        or not any(row["resolved_m"] != 128 for row in measurements)
    ):
        raise ProducerError("promotion representative cases require M=128 and another M")

    pair_inputs = manifest["full_model_pairs"]
    if not isinstance(pair_inputs, list) or (mode == "promotion" and len(pair_inputs) < 2) or (mode == "diagnostic" and pair_inputs):
        raise ProducerError("full_model_pairs count differs for producer mode")
    pairs: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    for index, pair in enumerate(
        sorted(pair_inputs, key=lambda item: (item.get("pair_id", ""), item.get("case_sha256", "")) if isinstance(item, dict) else ("", ""))
    ):
        label = f"full_model_pairs[{index}]"
        if not isinstance(pair, dict):
            raise ProducerError(f"{label} must be an object")
        exact(pair, PAIR_FIELDS, label)
        pair_id = pair["pair_id"]
        if not isinstance(pair_id, str) or not pair_id or pair_id in pair_ids:
            raise ProducerError("full-model pair IDs are invalid or duplicated")
        pair_ids.add(pair_id)
        run_index = count(pair["run_index"], f"{label}.run_index")
        if run_index < 2 or run_index > 11:
            raise ProducerError(f"{label} does not bind a measured run")
        baseline, baseline_run_id, baseline_runs = resident_raw(pair["baseline_raw"], f"{label} baseline raw")
        contender, contender_run_id, contender_runs = resident_raw(pair["candidate_raw"], f"{label} candidate raw")
        case_sha = digest(pair["case_sha256"], f"{label}.case_sha256")
        if (
            baseline_run_id == contender_run_id
            or baseline.get("case_id") != pair["case_id"]
            or contender.get("case_id") != pair["case_id"]
            or baseline.get("case_sha256") != case_sha
            or contender.get("case_sha256") != case_sha
            or baseline.get("workload") != contender.get("workload")
            or baseline_runs[run_index]["run_index"] != contender_runs[run_index]["run_index"]
        ):
            raise ProducerError(f"{label} baseline/candidate run pairing differs")
        pairs.append(
            {
                "candidate_id": candidate_id,
                "pair_id": pair_id,
                "case_sha256": case_sha,
                "identity_sha256": identity["identity_sha256"],
                "baseline_ms": finite_float(
                    baseline_runs[run_index]["timing"]["prefill_ms"],
                    f"{label} baseline prefill_ms",
                    positive=True,
                ),
                "candidate_ms": finite_float(
                    contender_runs[run_index]["timing"]["prefill_ms"],
                    f"{label} candidate prefill_ms",
                    positive=True,
                ),
            }
        )

    output = {
        "schema_version": RAW_SCHEMA,
        "status": "complete" if mode == "promotion" else "one_case_diagnostic",
        "measurement_eligible": mode == "promotion",
        "smoke_only": mode == "diagnostic",
        "promotion_eligible": mode == "promotion",
        "evidence_sha256": None,
        "identity": {
            field: identity[field]
            for field in (
                "identity_sha256",
                "case_manifest_sha256",
                "binary_sha256",
                "package_content_sha256",
            )
        },
        "capabilities": {
            "family_exclusive_timing": True,
            "d2h_count": True,
            "stream_sync_count": True,
        },
        "representative_prompt_count": 7,
        "measurements": measurements,
        "full_model_pairs": pairs,
    }
    output["evidence_sha256"] = SELECTOR.semantic_sha256(output)
    ensure_finite_tree(output, "candidate selection raw output")
    if mode == "promotion":
        SELECTOR.validate_raw(output)
    return output, snapshots


def write_output(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ProducerError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode("ascii") + b"\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest_snapshot = capture(args.manifest.absolute(), "producer manifest")
        manifest = parse_json(manifest_snapshot, "producer manifest")
        output, snapshots = build(manifest, manifest_snapshot)
        for snapshot in snapshots:
            snapshot.verify()
        write_output(args.output, output)
        print(
            json.dumps(
                {
                    "status": output["status"],
                    "measurement_eligible": output["measurement_eligible"],
                    "promotion_eligible": output["promotion_eligible"],
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ProducerError, SELECTOR.SelectionError) as error:
        print(f"build-aq4-p3-selection-raw: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
