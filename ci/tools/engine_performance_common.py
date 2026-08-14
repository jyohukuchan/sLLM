#!/usr/bin/env python3
"""Shared fail-closed contract for the Phase 5 direct benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError as exc:  # pragma: no cover
    Draft202012Validator = None  # type: ignore[assignment,misc]
    FormatChecker = None  # type: ignore[assignment,misc]
    _JSONSCHEMA_IMPORT_ERROR = exc
else:
    _JSONSCHEMA_IMPORT_ERROR = None

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, ROOT, canonical_bytes  # noqa: E402


MATRIX_PATH = ROOT / "ci/matrix/engine-performance-direct-v1.json"
DIRECT_SCHEMA_PATH = ROOT / "ci/schema/engine-performance-direct-v1.schema.json"
AGGREGATE_SCHEMA_PATH = ROOT / "ci/schema/engine-performance-aggregate-v1.schema.json"
DIRECT_VERSION = "engine-performance-direct-v1"
AGGREGATE_VERSION = "engine-performance-aggregate-v1"
EVIDENCE_VERSION = "engine-performance-evidence-v1"
BUILD_IDENTITY_VERSION = "sllm-build-identity-v2"
ROCM_RELEASE = "7.14.0"
ROCM_ROOT = "/opt/rocm/core-7.14"
MATRIX_RELATIVE = "ci/matrix/engine-performance-direct-v1.json"
PROTOCOL = {
    "backend": "hip",
    "dtype": "BF16",
    "batch_size": 1,
    "warmup_requests": 3,
    "measured_requests": 10,
    "stop_token_ids": [248046, 248044],
    "visible_stop_tokens": False,
}
CLAIMS = {"baseline_only": True, "optimized": False, "faster": False, "hard_gate": False}
TARGETS = ("gfx1030", "gfx1201")
BUILD_CONFIGURATION_KEYS = (
    "cargo_command",
    "cargo_profile",
    "rust_toolchain",
    "ROCM_PATH",
    "HIP_PATH",
    "SLLM_HIP_COMPILER",
    "CMAKE_HIP_ARCHITECTURES",
    "SLLM_HIP_CODEGEN_FEATURES",
    "SLLM_ENABLE_HIP_RUNTIME",
    "SLLM_ENABLE_PUBLIC_HIP_RUNTIME",
    "SLLM_ENABLE_HIP_COMPILE_PROBE",
)
SOURCE_IDENTITY_KEYS = ("source_root", "source_base_revision", "semantic_tree")
AGGREGATE_BUILD_IDENTITY_KEYS = (
    "build_inputs_digest",
    "build_configuration",
    "target",
    "backend",
    "rocm_release",
    "rocm_root",
    "binary_sha256",
)
_FIXED_BUILD_CONFIGURATION = {
    "cargo_command": "cargo +1.97.1 build --locked --offline --release -p sllm-cli",
    "cargo_profile": "release",
    "rust_toolchain": "1.97.1",
    "ROCM_PATH": "/opt/rocm",
    "HIP_PATH": "/opt/rocm",
    "SLLM_HIP_COMPILER": "/opt/rocm/bin/amdclang++",
    "SLLM_HIP_CODEGEN_FEATURES": "co_v6,wave32,xnack=unsupported,sramecc=unsupported,generic_processor_version=0",
    "SLLM_ENABLE_HIP_RUNTIME": "1",
    "SLLM_ENABLE_PUBLIC_HIP_RUNTIME": "1",
    "SLLM_ENABLE_HIP_COMPILE_PROBE": "0",
}
TARGET_MAPPING: dict[str, dict[str, Any]] = {
    "gfx1030": {
        "target": "gfx1030", "backend": "hip", "gpu_uuid": "GPU-76a08c022586fed6",
        "gpu_bdf": "0000:03:00.0", "product": "AMD Radeon Pro V620",
        "physical_hip_index": 1, "logical_device_index": 0,
    },
    "gfx1201": {
        "target": "gfx1201", "backend": "hip", "gpu_uuid": "GPU-a8e9ddefa2d60f55",
        "gpu_bdf": "0000:07:00.0", "product": "AMD Radeon AI PRO R9700",
        "physical_hip_index": 2, "logical_device_index": 0,
    },
}
MODEL_MAPPING: dict[str, dict[str, str]] = {
    "4B": {
        "repo_id": "Qwen/Qwen3.5-4B", "resolved_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "lock_path": "docs/models/locks/qwen3.5-4b-bf16.json",
        "lock_fingerprint": "sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae",
    },
    "2B": {
        "repo_id": "Qwen/Qwen3.5-2B", "resolved_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "lock_path": "docs/models/locks/qwen3.5-2b-bf16.json",
        "lock_fingerprint": "sha256:304e19f8b8ef78bab1848a6cfb46ac619a8ca5c8fd052cac1c43fc3f4d6dcdb3",
    },
    "9B": {
        "repo_id": "Qwen/Qwen3.5-9B", "resolved_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "lock_path": "docs/models/locks/qwen3.5-9b-bf16.json",
        "lock_fingerprint": "sha256:2d2bc642540e97d4681f8c66140e09f305f487476bb9fe238ca82a298febf893",
    },
}
_TOKEN_SEED = (1, 3, 17, 37, 73, 255, 256, 257, 2, 5, 11, 19, 23, 29, 31, 41, 43)


def _make_tokens(length: int) -> tuple[int, ...]:
    if length < len(_TOKEN_SEED):
        return _TOKEN_SEED[:length]
    return _TOKEN_SEED + tuple((index * 7919 + 41) % 248000 for index in range(len(_TOKEN_SEED), length))


TOKEN_SEQUENCE_MAPPING: dict[str, tuple[int, ...]] = {
    "minimum": (1,),
    "short-odd": _TOKEN_SEED,
    "boundary-255": _make_tokens(255),
    "boundary-256": _make_tokens(256),
    "boundary-257": _make_tokens(257),
    "prefill-long": _make_tokens(1024),
    "decode-long": _make_tokens(32),
}
CASE_MAPPING: dict[str, tuple[tuple[str, str, int, int, int], ...]] = {
    "4B": (
        ("minimum", "minimum", 1, 1, 5400), ("short-odd", "short-odd", 17, 17, 5400),
        ("boundary-255", "boundary-255", 255, 64, 5400), ("boundary-256", "boundary-256", 256, 64, 5400),
        ("boundary-257", "boundary-257", 257, 64, 5400), ("prefill-long", "prefill-long", 1024, 128, 10800),
        ("decode-long", "decode-long", 32, 256, 5400),
    ),
    "2B": (("short-odd", "short-odd", 17, 17, 3600), ("boundary-257", "boundary-257", 257, 64, 3600)),
    "9B": (("minimum", "minimum", 1, 1, 7200), ("short-odd", "short-odd", 17, 17, 7200)),
}
MAX_NS = (1 << 63) - 1
MAX_TOKEN_ID = 248319
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ROW_ID_RE = re.compile(r"^engine-performance-direct-(2b|4b|9b)-gfx(1030|1201)-[a-z0-9-]+$")
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_CACHE_FILES = 100_000
BUNDLE_PROTOCOL = "sllm-artifact-bundle-v1"
BUNDLE_COMMIT_NAME = "bundle.complete.json"


def fail(message: str) -> None:
    raise ContractError(message)


def expected_build_configuration(target: str) -> dict[str, str]:
    if target not in TARGETS:
        fail(f"unsupported Phase 5 build target: {target}")
    values = dict(_FIXED_BUILD_CONFIGURATION)
    values["CMAKE_HIP_ARCHITECTURES"] = target
    return {key: values[key] for key in BUILD_CONFIGURATION_KEYS}


def validate_build_configuration(value: Any, target: str, label: str = "build configuration") -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(BUILD_CONFIGURATION_KEYS):
        fail(f"{label} is incomplete or has unexpected fields")
    expected = expected_build_configuration(target)
    for key in BUILD_CONFIGURATION_KEYS:
        item = value[key]
        if not isinstance(item, str) or not item:
            fail(f"{label} value must be a nonempty string: {key}")
        if item != item.strip() or any(ord(char) < 0x20 or ord(char) == 0x7f for char in item):
            fail(f"{label} value is unsafe: {key}")
        if item != expected[key]:
            fail(f"{label} value does not match the Phase 5 contract: {key}")
    return expected


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_json_bytes(data: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                fail(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        fail(f"non-finite JSON number in {label}: {value}")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicates, parse_constant=reject_constant)
    except ContractError:
        raise
    except (UnicodeError, ValueError) as exc:
        fail(f"cannot parse {label}: {exc}")


def read_json(path: Path, label: str, max_bytes: int = MAX_JSON_BYTES) -> tuple[Any, bytes, str]:
    try:
        if path.is_symlink() or not path.is_file():
            fail(f"{label} must be a regular non-symlink file: {path}")
        if path.stat().st_size > max_bytes:
            fail(f"{label} exceeds bounded size: {path}")
        data = path.read_bytes()
    except OSError as exc:
        fail(f"cannot read {label} {path}: {exc}")
    return parse_json_bytes(data, label), data, hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, label: str, *, max_bytes: int | None = None) -> str:
    try:
        if path.is_symlink() or not path.is_file():
            fail(f"{label} must be a regular non-symlink file: {path}")
        size = path.stat().st_size
        if max_bytes is not None and size > max_bytes:
            fail(f"{label} exceeds bounded size: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        fail(f"cannot hash {label} {path}: {exc}")
    return digest.hexdigest()


def cache_digest(path: Path) -> str:
    try:
        if path.is_symlink() or not path.is_dir():
            fail(f"model cache must be a regular non-symlink directory: {path}")
        root = path.resolve()
        records: list[tuple[str, int, str]] = []
        for item in sorted(path.rglob("*")):
            if item.is_symlink():
                fail(f"model cache contains a symlink: {item}")
            if not item.is_file():
                continue
            relative = item.relative_to(path).as_posix()
            if not item.resolve().is_relative_to(root):
                fail(f"model cache escapes its root: {item}")
            if len(records) >= MAX_CACHE_FILES:
                fail("model cache exceeds bounded file count")
            records.append((relative, item.stat().st_size, sha256_file(item, f"model cache file {relative}")))
        return hashlib.sha256(canonical_bytes(records)).hexdigest()
    except OSError as exc:
        fail(f"cannot inspect model cache {path}: {exc}")


def schema_validate(value: Any, path: Path, label: str, definition: str | None = None) -> None:
    if Draft202012Validator is None:
        fail(f"jsonschema is required for {label}: {_JSONSCHEMA_IMPORT_ERROR}")
    schema, _, _ = read_json(path, f"{label} schema", 4 * 1024 * 1024)
    target: Any = schema
    if definition is not None:
        target = {"$schema": schema["$schema"], "$ref": f"#/$defs/{definition}", "$defs": schema["$defs"]}
    errors = sorted(Draft202012Validator(target, format_checker=FormatChecker()).iter_errors(value), key=lambda item: list(item.path))
    if errors:
        fail(f"{label} schema validation failed: " + "; ".join(error.message for error in errors[:5]))
    if definition == "manifest":
        validate_manifest_evidence(value)


def aggregate_source_identity(build_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the source fields that must be common to every aggregate row."""
    return {key: build_identity[key] for key in SOURCE_IDENTITY_KEYS}


def aggregate_build_identity(build_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete target-specific build tuple used by an aggregate."""
    return {
        "schema_version": BUILD_IDENTITY_VERSION,
        "build_manifest_sha256": build_identity["sha256"],
        **{key: build_identity[key] for key in AGGREGATE_BUILD_IDENTITY_KEYS},
    }


def _bundle_commit(payloads: Mapping[str, bytes]) -> bytes:
    return canonical_bytes({
        "protocol": BUNDLE_PROTOCOL,
        "state": "COMMITTED",
        "members": {
            name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            for name, payload in sorted(payloads.items())
        },
    })


def verify_aggregate_bundle(output_dir: Path, label: str) -> dict[str, Any]:
    """Require the last-published commit record and verify every bound member."""
    expected_names = {"graph.csv", "summary.json", "graph.csv.sha256", "summary.json.sha256"}
    output_dir = output_dir.resolve()
    commit, _, _ = read_json(output_dir / BUNDLE_COMMIT_NAME, f"{label} completion record", 1024 * 1024)
    if not isinstance(commit, dict) or set(commit) != {"protocol", "state", "members"}:
        fail(f"{label} completion record is malformed")
    if commit["protocol"] != BUNDLE_PROTOCOL or commit["state"] != "COMMITTED":
        fail(f"{label} completion record is stale or incomplete")
    members = commit["members"]
    if not isinstance(members, dict) or set(members) != expected_names:
        fail(f"{label} completion record has an incomplete member set")
    for name in sorted(expected_names):
        identity = members[name]
        if not isinstance(identity, dict) or set(identity) != {"sha256", "bytes"}:
            fail(f"{label} completion member identity is malformed: {name}")
        path = output_dir / name
        digest = sha256_file(path, f"{label} member {name}")
        try:
            size = path.stat().st_size
        except OSError as exc:
            fail(f"cannot stat {label} member {path}: {exc}")
        if identity["sha256"] != digest or identity["bytes"] != size:
            fail(f"{label} completion member is stale or tampered: {name}")
    return commit


def publish_aggregate_bundle(output_dir: Path, payloads: Mapping[str, bytes], label: str) -> None:
    """Publish members no-replace, then publish a consumer-verified commit record."""
    expected_names = {"graph.csv", "summary.json", "graph.csv.sha256", "summary.json.sha256"}
    if set(payloads) != expected_names:
        fail(f"{label} publication bundle is incomplete")
    output_dir = output_dir.resolve()
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        fail(f"{label} output directory is not a regular directory: {output_dir}")
    commit_payload = _bundle_commit(payloads)
    destinations = {name: output_dir / name for name in expected_names}
    destinations[BUNDLE_COMMIT_NAME] = output_dir / BUNDLE_COMMIT_NAME
    if any(path.exists() or path.is_symlink() for path in destinations.values()):
        fail(f"refusing to overwrite existing {label} output")

    temporary: dict[str, Path] = {}
    published: list[tuple[Path, Path]] = []
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.is_symlink() or not output_dir.is_dir():
            fail(f"{label} output directory changed during publication: {output_dir}")
        if any(path.exists() or path.is_symlink() for path in destinations.values()):
            fail(f"refusing to overwrite existing {label} output")
        staged_payloads = dict(payloads)
        staged_payloads[BUNDLE_COMMIT_NAME] = commit_payload
        for name, payload in staged_payloads.items():
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=output_dir)
            temporary_path = Path(temporary_name)
            temporary[name] = temporary_path
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        for name in ("graph.csv", "summary.json", "graph.csv.sha256", "summary.json.sha256"):
            source = temporary[name]
            destination = destinations[name]
            os.link(source, destination)
            published.append((source, destination))
        directory_descriptor = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        source = temporary[BUNDLE_COMMIT_NAME]
        destination = destinations[BUNDLE_COMMIT_NAME]
        os.link(source, destination)
        published.append((source, destination))
        directory_descriptor = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except (OSError, ValueError) as exc:
        for source, destination in reversed(published):
            try:
                if destination.exists() and os.path.samefile(source, destination):
                    destination.unlink()
            except OSError:
                pass
        fail(f"cannot publish {label} bundle: {exc}")
    finally:
        for path in temporary.values():
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _performance_static_identity(static: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "target", "product", "gpu_bdf", "gpu_uuid", "physical_hip_index",
        "amd_smi_gpu_index", "driver_version", "kernel_version", "vram_total_mb",
    )
    return {key: static.get(key) for key in keys}


def _performance_limit_value(static: Mapping[str, Any], key: str, unit: str) -> float:
    limits = static.get("limits", {}).get("values")
    if not isinstance(limits, dict):
        fail("performance manifest static limits are missing")
    if key == "socket_power_limit":
        ppt0 = limits.get("ppt0")
        record = ppt0.get(key) if isinstance(ppt0, dict) else None
    else:
        record = limits.get(key)
    if not isinstance(record, dict) or set(record) != {"value", "unit"} or record.get("unit") != unit:
        fail(f"performance manifest {key} limit is malformed")
    value = record.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        fail(f"performance manifest {key} limit is malformed")
    return float(value)


def _validate_performance_metric_safety(static: Mapping[str, Any], metric: Mapping[str, Any], label: str) -> None:
    temperatures = metric.get("temperature_c")
    if not isinstance(temperatures, dict):
        fail(f"{label} temperature evidence is missing")
    thresholds = {
        "edge": _performance_limit_value(static, "slowdown_edge_temperature", "C"),
        "hotspot": _performance_limit_value(static, "slowdown_hotspot_temperature", "C"),
        "mem": _performance_limit_value(static, "slowdown_vram_temperature", "C"),
    }
    for sensor, threshold in thresholds.items():
        value = temperatures.get(sensor)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) >= threshold:
            fail(f"{label} {sensor} temperature reached or exceeded the published slowdown limit")
    power = metric.get("power_w")
    power_limit = _performance_limit_value(static, "socket_power_limit", "W")
    if not isinstance(power, (int, float)) or isinstance(power, bool) or not math.isfinite(float(power)) or float(power) > power_limit:
        fail(f"{label} socket power exceeded the published limit")


def validate_manifest_evidence(manifest: Any) -> None:
    """Validate cross-field evidence invariants that JSON Schema cannot express."""
    if not isinstance(manifest, dict) or not isinstance(manifest.get("row_id"), str):
        fail("performance manifest evidence has no row identity")
    row_parts = manifest["row_id"].split("-")
    if len(row_parts) < 6 or row_parts[4] not in TARGETS:
        fail("performance manifest evidence row identity is malformed")
    target = row_parts[4]
    device = expected_device(target)
    build = manifest.get("build_identity")
    if not isinstance(build, dict) or build.get("target") != target or build.get("backend") != "hip" or build.get("rocm_release") != ROCM_RELEASE or build.get("rocm_root") != ROCM_ROOT or build.get("binary_sha256") != manifest.get("binary", {}).get("sha256"):
        fail("performance manifest build identity is not bound to the row and binary")
    validate_build_configuration(build.get("build_configuration"), target, "performance manifest build configuration")
    evidence = manifest.get("evidence")
    if not isinstance(evidence, dict):
        fail("performance manifest evidence is missing")
    if evidence.get("version") != EVIDENCE_VERSION or evidence.get("cadence_seconds") != 1:
        fail("performance manifest evidence version/cadence is stale")
    expected_definitions = {
        "clock_variation": "Dynamic clock min/max is observational; no numeric threshold is a violation.",
        "violation": "When violation accumulators are unavailable, aggregate THROTTLED status is observational; ECC, published thermal/power limits, and exposed active violations remain fail-closed.",
        "process_ownership": "Every during sample must name only descendants of the benchmark process group.",
    }
    legacy_definitions = {
        **expected_definitions,
        "violation": "A non-UNTHROTTLED power status or exposed active violation is explicit evidence; all-N/A accumulators are a documented AMD-SMI limitation.",
    }
    if evidence.get("definitions") not in (expected_definitions, legacy_definitions):
        fail("performance manifest evidence definitions drifted")
    if evidence.get("visibility") != {"cleared": ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"], "selector": "ROCR_VISIBLE_DEVICES", "uuid": device["gpu_uuid"]}:
        fail("performance manifest visibility isolation evidence drifted")
    enforce_pass_safety = manifest.get("state") == "PASS"
    for phase_name in ("pre", "post"):
        phase = evidence.get(phase_name)
        if not isinstance(phase, dict) or phase.get("process_state") != "CLEAN":
            fail(f"performance manifest {phase_name} evidence is not clean")
        static = phase.get("static", {})
        metric = phase.get("metric", {})
        vram = phase.get("vram_auxiliary", {})
        if static.get("target") != target or static.get("gpu_bdf") != device["gpu_bdf"] or static.get("gpu_uuid") != device["gpu_uuid"] or static.get("product") != device["product"] or static.get("physical_hip_index") != device["physical_hip_index"]:
            fail(f"performance manifest {phase_name} evidence exact identity drifted")
        if metric.get("ecc_uncorrectable") != 0 or metric.get("throttle_status") not in {"UNTHROTTLED", "THROTTLED"} or vram.get("source") != "amd-smi monitor -v":
            fail(f"performance manifest {phase_name} evidence health/VRAM contract failed")
        if enforce_pass_safety:
            _validate_performance_metric_safety(static, metric, f"performance manifest {phase_name}")
    pre = evidence["pre"]
    post = evidence["post"]
    during = evidence.get("during")
    if not isinstance(during, dict) or during.get("sample_count") != during.get("summary", {}).get("sample_count") or during.get("sample_count", 0) < 1:
        fail("performance manifest during evidence sample count is invalid")
    first = during.get("first", {})
    last = during.get("last", {})
    loader = during.get("loader", {})
    loader_records = during.get("loaders")
    if not isinstance(first, dict) or not isinstance(last, dict) or not isinstance(loader, dict) or not isinstance(loader_records, list) or not loader_records:
        fail("performance manifest during evidence is incomplete")
    loaders_by_digest: dict[str, dict[str, Any]] = {}
    for loader_record in loader_records:
        if not isinstance(loader_record, dict):
            fail("performance manifest loader record is not an object")
        paths = loader_record.get("resolved_paths", [])
        path_digest = loader_record.get("path_digest")
        library_digests = loader_record.get("library_digests")
        process_ids = loader_record.get("process_ids")
        if (
            loader_record.get("required_rocm_release") != ROCM_RELEASE
            or loader_record.get("expected_root") != ROCM_ROOT
            or not isinstance(paths, list)
            or len(paths) < 2
            or any(not isinstance(path, str) for path in paths)
            or paths != sorted(set(paths))
            or path_digest != "sha256:" + hashlib.sha256(canonical_bytes(paths)).hexdigest()
            or not isinstance(library_digests, dict)
            or set(library_digests) != set(paths)
            or any(
                not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in library_digests.values()
            )
            or not isinstance(process_ids, list)
            or not process_ids
            or any(not isinstance(pid, int) or isinstance(pid, bool) or pid < 1 for pid in process_ids)
        ):
            fail("performance manifest loader path/digest/process evidence failed")
        if not any(Path(path).name.startswith("libamdhip64.so") for path in paths) or not any(Path(path).name.startswith("libhsa-runtime64.so") for path in paths) or any(not path.startswith(ROCM_ROOT + "/") for path in paths):
            fail("performance manifest loader root/library evidence failed")
        if path_digest in loaders_by_digest:
            fail("performance manifest loader evidence digest is duplicated")
        loaders_by_digest[path_digest] = loader_record
    if loaders_by_digest.get(loader.get("path_digest")) != loader:
        fail("performance manifest final loader has no exact provenance record")
    for sample in (first, last):
        if sample.get("process", {}).get("state") != "OWNED" or not sample.get("process", {}).get("pids") or sample.get("loader_path_digest") not in loaders_by_digest:
            fail("performance manifest during process/loader ownership evidence failed")
        if sample.get("metric", {}).get("throttle_status") not in {"UNTHROTTLED", "THROTTLED"} or sample.get("metric", {}).get("ecc_uncorrectable") != 0:
            fail("performance manifest during health evidence failed")
        if enforce_pass_safety:
            _validate_performance_metric_safety(pre["static"], sample["metric"], "performance manifest during")
    checks = evidence.get("checks", {})
    if manifest.get("state") == "PASS":
        perf_levels = during.get("summary", {}).get("perf_levels")
        expected_checks = {
            "exact_identity": _performance_static_identity(pre["static"]) == _performance_static_identity(post["static"]),
            "static_identity_unchanged": pre["static"].get("target") == post["static"].get("target") and pre["static"].get("product") == post["static"].get("product"),
            "profile_unchanged": pre["static"].get("profile") == post["static"].get("profile"),
            "limits_unchanged": pre["static"].get("limits") == post["static"].get("limits"),
            "performance_level_unchanged": pre["metric"].get("perf_level") == post["metric"].get("perf_level") and perf_levels == [pre["metric"].get("perf_level")],
            "explicit_violation": False,
            "vram_auxiliary_complete": all(sample.get("vram_auxiliary", {}).get("source") == "amd-smi monitor -v" for sample in (first, last)),
            "process_ownership": all(sample.get("process", {}).get("state") == "OWNED" and bool(sample.get("process", {}).get("pids")) for sample in (first, last)),
            "loader_paths_verified": True,
            "monitor_errors": 0,
            "process_group_cleanup": manifest.get("execution", {}).get("process_group_gone") is True and manifest.get("cleanup", {}).get("process_group_gone") is True,
        }
        if checks != expected_checks or during.get("violation", {}).get("explicit_violation") is not False:
            fail("PASS performance manifest has failed evidence checks")


def _closed_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        fail(f"{label} keys differ: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")


def _sequence_records() -> list[dict[str, Any]]:
    return [
        {"order": order, "sequence_id": sequence_id, "input_token_ids": list(ids), "input_tokens": len(ids)}
        for order, (sequence_id, ids) in enumerate(TOKEN_SEQUENCE_MAPPING.items())
    ]


def _case_records(model_size: str) -> list[dict[str, Any]]:
    return [
        {"order": order, "case_id": case_id, "input_token_sequence": sequence_id, "input_tokens": input_tokens,
         "requested_output_tokens": output_tokens, "timeout_seconds": timeout_seconds}
        for order, (case_id, sequence_id, input_tokens, output_tokens, timeout_seconds) in enumerate(CASE_MAPPING[model_size])
    ]


def expected_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    order = 0
    for model_size in ("4B", "2B", "9B"):
        for target in TARGETS:
            for case_id, sequence_id, input_tokens, output_tokens, timeout_seconds in CASE_MAPPING[model_size]:
                rows.append({
                    "order": order,
                    "row_id": f"engine-performance-direct-{model_size.lower()}-{target}-{case_id}",
                    "model_size": model_size,
                    "case_id": case_id,
                    "input_token_sequence": sequence_id,
                    "input_token_ids": list(TOKEN_SEQUENCE_MAPPING[sequence_id]),
                    "input_tokens": input_tokens,
                    "requested_output_tokens": output_tokens,
                    "target": target,
                    "timeout_seconds": timeout_seconds,
                })
                order += 1
    return rows


def resolved_row(row: Mapping[str, Any]) -> dict[str, Any]:
    sequence_id = row.get("input_token_sequence")
    if sequence_id not in TOKEN_SEQUENCE_MAPPING:
        fail(f"unknown input token sequence: {sequence_id}")
    input_ids = list(TOKEN_SEQUENCE_MAPPING[sequence_id])
    if row.get("input_tokens") != len(input_ids):
        fail(f"row input token count differs from its explicit token sequence: {row.get('row_id')}")
    if row.get("input_token_ids") != input_ids:
        fail(f"row input token IDs differ from its named token sequence: {row.get('row_id')}")
    resolved = dict(row)
    resolved["input_token_ids"] = input_ids
    return resolved


def validate_matrix_document(matrix: Any) -> dict[str, Any]:
    expected_top = {"schema_version", "matrix_id", "revision", "suite_id", "tier", "required", "claims", "protocol", "token_sequences", "targets", "models", "rows"}
    _closed_keys(matrix, expected_top, "performance matrix")
    if matrix["schema_version"] != DIRECT_VERSION or matrix["matrix_id"] != DIRECT_VERSION or matrix["revision"] != 4:
        fail("performance matrix version or identity is stale")
    if matrix["suite_id"] != "h0-engine-performance-direct-contract" or matrix["tier"] != "tier_p1" or matrix["required"] is not False:
        fail("performance matrix suite/tier/required contract drifted")
    if matrix["claims"] != CLAIMS or matrix["protocol"] != PROTOCOL:
        fail("performance matrix claims or protocol drifted")
    if matrix["token_sequences"] != _sequence_records():
        fail("performance matrix token sequences are missing, reordered, or changed")
    targets = matrix["targets"]
    target_keys = {"order", "target", "backend", "gpu_uuid", "gpu_bdf", "product", "physical_hip_index", "logical_device_index"}
    if not isinstance(targets, list) or len(targets) != len(TARGETS):
        fail("performance matrix must contain exactly the canonical targets")
    for expected_target, actual in zip(TARGETS, targets):
        _closed_keys(actual, target_keys, f"performance target {expected_target}")
        if actual != {"order": TARGETS.index(expected_target), **TARGET_MAPPING[expected_target]}:
            fail(f"performance target mapping is not canonical: {expected_target}")
    models = matrix["models"]
    if not isinstance(models, list) or [item.get("model_size") for item in models] != ["4B", "2B", "9B"]:
        fail("performance model order is not exactly 4B/2B/9B")
    model_keys = {"order", "model_size", "repo_id", "resolved_revision", "lock_path", "lock_fingerprint", "cases"}
    for index, model in enumerate(models):
        size = ("4B", "2B", "9B")[index]
        _closed_keys(model, model_keys, f"performance model {size}")
        expected_model = {"order": index, "model_size": size, **MODEL_MAPPING[size]}
        if {key: model[key] for key in expected_model} != expected_model:
            fail(f"performance model identity drifted: {size}")
        if model["cases"] != _case_records(size):
            fail(f"performance case set drifted: {size}")
    if matrix["rows"] != expected_rows():
        fail("performance matrix rows are missing, reordered, duplicated, or changed")
    return matrix


def load_matrix(path: Path = MATRIX_PATH) -> tuple[dict[str, Any], str]:
    matrix, _, digest = read_json(path, "performance matrix", 8 * 1024 * 1024)
    return validate_matrix_document(matrix), digest


def expected_device(target: str) -> dict[str, Any]:
    if target not in TARGET_MAPPING:
        fail(f"unknown canonical target: {target}")
    return dict(TARGET_MAPPING[target])


def expected_model(model_size: str) -> dict[str, str]:
    if model_size not in MODEL_MAPPING:
        fail(f"unknown model size: {model_size}")
    return dict(MODEL_MAPPING[model_size])


def safe_difference(end: Any, start: Any, label: str) -> int:
    if not is_int(end) or not is_int(start) or start < 0 or end < 0 or start > MAX_NS or end > MAX_NS:
        fail(f"{label} contains an out-of-range event timestamp")
    if end <= start:
        fail(f"{label} is not strictly increasing")
    difference = end - start
    if difference > MAX_NS:
        fail(f"{label} duration overflows the bounded integer range")
    return difference


def validate_stop_semantics(
    generated: list[int], stop: Mapping[str, Any], stop_policy: Mapping[str, Any],
    max_new_tokens: int, label: str,
) -> None:
    stop_ids = stop_policy["stop_token_ids"]
    if stop.get("version") != 1 or stop.get("reason_version") != 1:
        fail(f"{label} stop protocol version is stale")
    if stop["kind"] == "max_new_tokens":
        if stop["token_id"] is not None or len(generated) != max_new_tokens or any(token in stop_ids for token in generated):
            fail(f"{label} max-new-token stop evidence is invalid")
    elif stop["kind"] == "stop_token":
        if stop["token_id"] not in stop_ids or not generated or len(generated) > max_new_tokens or generated[-1] != stop["token_id"] or any(token in stop_ids for token in generated[:-1]):
            fail(f"{label} stop-token evidence is invalid")
    else:
        fail(f"{label} has an unknown generation stop reason")


def expected_sample(
    sample: Mapping[str, Any], input_ids: list[int], stop_policy: Mapping[str, Any],
    max_new_tokens: int,
) -> None:
    if sample["execution_path"] != "timed-production" or sample["timing_instrumentation"] != "on":
        fail("sample did not use the timed production path with timing instrumentation enabled")
    events = sample["events"]
    publications = events["later_token_publications_ns"]
    generated = sample["tokens"]["generated_token_ids"]
    if len(publications) + 1 != len(generated) or not generated:
        fail("sample token publication count does not match generated token count")
    ordered = [
        events["request_start_ns"], events["prefill_submit_ns"], events["prefill_complete_ns"],
        events["first_token_ns"], *publications, events["stop_ns"], events["cleanup_ns"],
    ]
    if any(not is_int(value) or value < 0 or value > MAX_NS for value in ordered):
        fail("sample event timestamp is outside the bounded integer range")
    if any(left > right for left, right in zip(ordered, ordered[1:])):
        fail("sample events are non-monotonic")
    if sample["tokens"]["input_token_ids"] != input_ids:
        fail("sample input token IDs differ from the matrix sequence")
    if sample["tokens"]["decode_input_token_ids"] != generated[:-1]:
        fail("decode input token IDs do not match the generated sequence")
    if sample["tokens"]["visible_token_ids"] != [token for token in generated if token not in stop_policy["stop_token_ids"]]:
        fail("visible token IDs do not match the locked stop policy")
    if any(not is_int(token) or token < 0 or token > MAX_TOKEN_ID for token in generated):
        fail("generated token ID is outside the locked tokenizer vocabulary")
    validate_stop_semantics(generated, sample["stop"], stop_policy, max_new_tokens, "sample")
    ttft = safe_difference(events["first_token_ns"], events["request_start_ns"], "TTFT")
    prefill = safe_difference(events["prefill_complete_ns"], events["prefill_submit_ns"], "prefill")
    e2e = safe_difference(events["cleanup_ns"], events["request_start_ns"], "E2E")
    tpot = [safe_difference(right, left, "TPOT") for left, right in zip([events["first_token_ns"], *publications[:-1]], publications)]
    decode_tokens = len(generated) - 1
    decode_rate = None
    if decode_tokens:
        decode_window = safe_difference(publications[-1], events["first_token_ns"], "decode")
        decode_rate = decode_tokens * 1_000_000_000 / decode_window
    expected_derived = {
        "ttft_ns": ttft,
        "prefill_ns": prefill,
        "prefill_tokens_per_second": len(input_ids) * 1_000_000_000 / prefill,
        "e2e_ns": e2e,
        "tpot_ns": tpot,
        "decode_tokens": decode_tokens,
        "decode_tokens_per_second": decode_rate,
    }
    if sample["derived"] != expected_derived:
        fail("sample derived timing arithmetic does not match its event trace")
    audit = sample["audit"]
    if audit["selected_backend"] != "hip" or audit["target"] != sample["_target"] or audit["device_index"] != sample["_device_index"]:
        fail("sample dispatch identity is stale")
    if audit["model_fingerprint"] != sample["_model_fingerprint"] or audit["plan_digest"] != sample["_plan_digest"]:
        fail("sample binding identity is stale")
    if audit["fallback_used"] is not False or audit["all_dispatches_hip"] is not True or audit["submission_count"] < 1 or audit["kernel_dispatch_count"] < 1:
        fail("sample audit is not HIP-only and fallback-free")
    validate_snapshot(sample["memory"]["request_start"], "sample request-start memory")
    validate_snapshot(sample["memory"]["after_cleanup"], "sample after-cleanup memory")
    start_values = _snapshot_values(sample["memory"]["request_start"], "sample request-start memory")
    cleanup_values = _snapshot_values(sample["memory"]["after_cleanup"], "sample after-cleanup memory")
    if cleanup_values["request_current_bytes"] != 0 or cleanup_values["workspace_current_bytes"] != 0:
        fail("sample request cleanup left request/workspace allocations")
    if cleanup_values["model_current_bytes"] != start_values["model_current_bytes"]:
        fail("sample request cleanup changed the resident model allocation")
    if sample["cleanup"]["request_dropped"] is not True or sample["cleanup"]["allocator_cleanup_validated"] is not True or sample["cleanup"]["retryable_cleanup"] != 0 or sample["cleanup"]["durable_quarantine"] != 0:
        fail("sample request cleanup is not empty")


def validate_snapshot(snapshot: Any, label: str) -> None:
    expected = {"model_resident", "request_state", "workspace", "current_bytes", "high_water_bytes", "poisoned"}
    _closed_keys(snapshot, expected, label)
    if not is_int(snapshot["current_bytes"]) or not is_int(snapshot["high_water_bytes"]) or snapshot["current_bytes"] < 0 or snapshot["high_water_bytes"] < snapshot["current_bytes"]:
        fail(f"{label} has invalid total allocation accounting")
    if snapshot["poisoned"] is not False:
        fail(f"{label} allocation accounting is poisoned")
    current_sum = 0
    for category in ("model_resident", "request_state", "workspace"):
        value = snapshot[category]
        _closed_keys(value, {"current_bytes", "high_water_bytes"}, f"{label}.{category}")
        if any(not is_int(value[key]) or value[key] < 0 or value["high_water_bytes"] < value["current_bytes"] for key in ("current_bytes", "high_water_bytes")):
            fail(f"{label}.{category} has invalid allocation accounting")
        if value["high_water_bytes"] > snapshot["high_water_bytes"]:
            fail(f"{label}.{category} high-water bytes exceed the total high-water bytes")
        current_sum += value["current_bytes"]
    if current_sum != snapshot["current_bytes"]:
        fail(f"{label} total current bytes do not equal the category current sum")


def _snapshot_values(snapshot: Mapping[str, Any], label: str) -> dict[str, int]:
    validate_snapshot(snapshot, label)
    return {
        "model_current_bytes": snapshot["model_resident"]["current_bytes"],
        "model_high_water_bytes": snapshot["model_resident"]["high_water_bytes"],
        "request_current_bytes": snapshot["request_state"]["current_bytes"],
        "workspace_current_bytes": snapshot["workspace"]["current_bytes"],
        "total_current_bytes": snapshot["current_bytes"],
        "total_high_water_bytes": snapshot["high_water_bytes"],
    }


def validate_cli_result(result: Any, row: Mapping[str, Any], *, schema: bool = True) -> dict[str, Any]:
    if schema:
        schema_validate(result, DIRECT_SCHEMA_PATH, "direct engine result")
    if not isinstance(result, dict):
        fail("direct engine result is not an object")
    row = resolved_row(row)
    model = expected_model(row["model_size"])
    device = expected_device(row["target"])
    expected_row_record = {
        "row_id": row["row_id"], "model_size": row["model_size"], "case_id": row["case_id"],
        "input_token_ids": row["input_token_ids"], "input_token_count": row["input_tokens"],
        "requested_output_tokens": row["requested_output_tokens"],
    }
    if result["row"] != expected_row_record:
        fail("direct result row/token identity does not match the matrix")
    expected_model_record = {
        "model_size": row["model_size"], "repo_id": model["repo_id"],
        "resolved_revision": model["resolved_revision"], "lock_fingerprint": model["lock_fingerprint"],
    }
    identities = result["identities"]
    if identities["engine"] != "sllm" or identities["backend"] != "hip" or identities["device_index"] != device["logical_device_index"] or identities["target"] != row["target"]:
        fail("direct result engine/device identity is stale")
    if identities["model"] != expected_model_record or identities["binding"]["model_fingerprint"] != model["lock_fingerprint"]:
        fail("direct result model/binding identity is stale")
    if result["config"] != {
        "input_token_ids": row["input_token_ids"], "input_token_count": row["input_tokens"],
        "max_new_tokens": row["requested_output_tokens"], "greedy": True,
        "warmups": PROTOCOL["warmup_requests"], "measured": PROTOCOL["measured_requests"],
        "tokenizer": False, "render": False,
        "stop_policy": {"stop_token_ids": PROTOCOL["stop_token_ids"], "visible_stop_tokens": False},
    }:
        fail("direct result config does not match the fixed matrix")
    load = result["model_load"]
    if load["event"] != "model_load" or load["start_ns"] != 0 or load["load_count"] != 1 or load["duration_ns"] != safe_difference(load["model_ready_ns"], load["start_ns"], "model load"):
        fail("model load arithmetic or count is invalid")
    ready = _snapshot_values(result["memory"]["model_ready"], "model-ready memory")
    after_drop = _snapshot_values(result["memory"]["after_model_drop"], "post-model memory")
    if ready["request_current_bytes"] != 0 or ready["workspace_current_bytes"] != 0 or ready["model_current_bytes"] == 0:
        fail("model-ready memory does not contain only the resident model")
    if any(after_drop[key] != 0 for key in ("model_current_bytes", "request_current_bytes", "workspace_current_bytes", "total_current_bytes")):
        fail("model drop left non-zero current allocation bytes")
    if result["memory"]["model_resident_high_water_bytes"] != ready["model_high_water_bytes"] or result["memory"]["resident_vram_bytes"] != ready["model_high_water_bytes"] or result["memory"]["resident_vram_source"] != "model_resident_allocator_high_water" or result["memory"]["peak_vram_bytes"] != after_drop["total_high_water_bytes"] or result["memory"]["peak_vram_bytes"] < result["memory"]["resident_vram_bytes"] or result["memory"]["peak_source"] != "runtime_allocator":
        fail("direct memory identity or high-water mark is invalid")
    audit = result["audit"]
    if audit["selected_backend"] != "hip" or audit["target"] != row["target"] or audit["device_index"] != device["logical_device_index"] or audit["fallback_used"] is not False or audit["all_dispatches_hip"] is not True or audit["submission_count"] < 1 or audit["kernel_dispatch_count"] < 1 or audit["segment_count"] < 1 or audit["boundary_count"] < 1:
        fail("direct aggregate audit is not HIP-only and fallback-free")
    if audit["model_load_count"] != 1 or audit["request_model_load_count"] != 0 or audit["model_reused"] is not True or audit["sample_count"] != 13 or audit["correctness_control_request_count"] != 1 or audit["total_request_count"] != 14:
        fail("model resident/request-local audit is invalid")
    if result["cleanup"] != {
        "correctness_control_request_count": 1, "warmup_request_count": 3, "measured_request_count": 10,
        "request_cleanup_count": 14, "performance_sample_count": 13, "all_requests_dropped": True,
        "correctness_control_dropped": True, "retryable_cleanup": 0, "durable_quarantine": 0,
    }:
        fail("request-local cleanup contract is invalid")
    if result["session_cleanup"] != {"retryable_cleanup": 0, "durable_quarantine": 0}:
        fail("HIP session cleanup is not empty")
    if result["warmups"]["count"] != 3 or len(result["warmups"]["samples"]) != 3 or result["measured"]["count"] != 10 or len(result["measured"]["samples"]) != 10:
        fail("warmup/measured counts are invalid")
    stop_policy = {"stop_token_ids": PROTOCOL["stop_token_ids"], "visible_stop_tokens": False}
    control = result["correctness_control"]
    expected_comparison = {
        "mode": "exact", "scope": "every_warmup_and_measured_sample",
        "token_fields": ["input_token_ids", "generated_token_ids", "visible_token_ids", "decode_input_token_ids"],
        "stop_fields": ["version", "reason_version", "kind", "token_id"],
        "dispatch_fields": ["selected_backend", "target", "device_index", "model_fingerprint", "plan_digest", "fallback_used", "all_dispatches_hip", "submission_count", "kernel_dispatch_count", "segment_count", "boundary_count"],
        "dispatch_count_rule": "exact_when_token_and_stop_fields_match",
    }
    if control["label"] != "correctness-only" or control["execution_path"] != "normal-untimed" or control["timing_instrumentation"] != "off" or control["included_in_performance_statistics"] is not False or control["comparison"] != expected_comparison:
        fail("correctness-control execution/comparison contract is invalid")
    control_tokens = control["tokens"]
    if control_tokens["input_token_ids"] != row["input_token_ids"] or not control_tokens["generated_token_ids"] or control_tokens["decode_input_token_ids"] != control_tokens["generated_token_ids"][:-1] or control_tokens["visible_token_ids"] != [token for token in control_tokens["generated_token_ids"] if token not in stop_policy["stop_token_ids"]]:
        fail("correctness-control token semantics are invalid")
    if any(not is_int(token) or token < 0 or token > MAX_TOKEN_ID for token in control_tokens["generated_token_ids"]):
        fail("correctness-control token ID is outside the locked tokenizer vocabulary")
    validate_stop_semantics(control_tokens["generated_token_ids"], control["stop"], stop_policy, row["requested_output_tokens"], "correctness-control")
    control_audit = control["audit"]
    if control_audit["selected_backend"] != "hip" or control_audit["target"] != row["target"] or control_audit["device_index"] != device["logical_device_index"] or control_audit["model_fingerprint"] != model["lock_fingerprint"] or control_audit["plan_digest"] != identities["binding"]["plan_digest"] or control_audit["fallback_used"] is not False or control_audit["all_dispatches_hip"] is not True or control_audit["submission_count"] < 1 or control_audit["kernel_dispatch_count"] < 1 or control_audit["segment_count"] < 1 or control_audit["boundary_count"] < 1:
        fail("correctness-control dispatch audit is invalid")
    control_start = _snapshot_values(control["memory"]["request_start"], "correctness-control request-start memory")
    control_cleanup = _snapshot_values(control["memory"]["after_cleanup"], "correctness-control after-cleanup memory")
    if control_cleanup["request_current_bytes"] != 0 or control_cleanup["workspace_current_bytes"] != 0 or control_cleanup["model_current_bytes"] != control_start["model_current_bytes"] or control["cleanup"] != {"request_dropped": True, "allocator_cleanup_validated": True}:
        fail("correctness-control allocator cleanup is invalid")
    sample_submission_count = 0
    sample_dispatch_count = 0
    for group in ("warmups", "measured"):
        for expected_index, sample in enumerate(result[group]["samples"]):
            if sample["cleanup"]["sample_index"] != expected_index:
                fail(f"{group} samples are missing, duplicated, or reordered")
            enriched = dict(sample)
            enriched["_target"] = row["target"]
            enriched["_device_index"] = device["logical_device_index"]
            enriched["_model_fingerprint"] = model["lock_fingerprint"]
            enriched["_plan_digest"] = identities["binding"]["plan_digest"]
            expected_sample(enriched, row["input_token_ids"], stop_policy, row["requested_output_tokens"])
            for section, fields in (("tokens", expected_comparison["token_fields"]), ("stop", expected_comparison["stop_fields"]), ("audit", expected_comparison["dispatch_fields"])):
                if any(control[section][field] != sample[section][field] for field in fields):
                    fail(f"{group} sample {section} semantic/dispatch signature differs from correctness control")
            sample_submission_count += sample["audit"]["submission_count"]
            sample_dispatch_count += sample["audit"]["kernel_dispatch_count"]
    if audit["submission_count"] != sample_submission_count or audit["kernel_dispatch_count"] != sample_dispatch_count:
        fail("aggregate audit counts do not match per-request dispatch evidence")
    return result


def percentile(values: list[int | float], fraction: float) -> int | float:
    if not values:
        fail("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary_stats(values: list[int | float]) -> dict[str, int | float]:
    if not values:
        return {"median": 0, "p10": 0, "p90": 0, "mad": 0, "min": 0, "max": 0, "count": 0}
    median = statistics_median(values)
    deviations = [abs(value - median) for value in values]
    return {
        "median": median, "p10": percentile(values, 0.10), "p90": percentile(values, 0.90),
        "mad": statistics_median(deviations), "min": min(values), "max": max(values), "count": len(values),
    }


def statistics_median(values: Iterable[int | float]) -> int | float:
    ordered = sorted(values)
    if not ordered:
        fail("median requires at least one value")
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def metric_values(result: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, list[int | float]]:
    values: dict[str, list[int | float]] = {
        "ttft_ns": [], "prefill_ns": [], "tpot_ns": [], "decode_token_per_s": [],
        "prefill_token_per_s": [], "e2e_ns": [], "resident_vram_bytes": [], "peak_vram_bytes": [],
    }
    for sample in result["measured"]["samples"]:
        derived = sample["derived"]
        values["ttft_ns"].append(derived["ttft_ns"])
        values["prefill_ns"].append(derived["prefill_ns"])
        if derived["tpot_ns"]:
            values["tpot_ns"].append(statistics_median(derived["tpot_ns"]))
        if derived["decode_tokens_per_second"] is not None:
            values["decode_token_per_s"].append(derived["decode_tokens_per_second"])
        values["prefill_token_per_s"].append(derived["prefill_tokens_per_second"])
        values["e2e_ns"].append(derived["e2e_ns"])
    values["resident_vram_bytes"].append(result["memory"]["resident_vram_bytes"])
    values["peak_vram_bytes"].append(result["memory"]["peak_vram_bytes"])
    return values


def claims() -> dict[str, bool]:
    return dict(CLAIMS)
