#!/usr/bin/env python3
"""Validate detached GPU metadata correction for a bounded AQ4 path oracle.

The original v1 output is never changed.  This validator reconstructs the raw
run hashes, device mapping, resource samples, service recovery identity, and
source-token replay binding before accepting a corrected v2 attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    if not name or name in sys.modules:
        raise RuntimeError(f"dynamic module name is empty or already registered: {name!r}")
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        if sys.modules.get(name) is not module:
            raise RuntimeError(f"dynamic module registration changed while loading {filename}")
        return module
    finally:
        # The registration exists only while module-level code (including dataclass
        # decoration) executes.  Never retain a successful or partial dynamic import.
        sys.modules.pop(name, None)


ORACLE = _load("qwen35_aq4_p2_oracle_attestation", "qwen35_aq4_p2_oracle.py")

REQUIRED_HIP_KERNEL_ENV = (
    "ULLM_REQUIRE_HIP_AQ4_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_PAIR_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL",
    "ULLM_REQUIRE_HIP_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_SEQUENCE_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL",
    "ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL",
    "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL",
    "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL",
    "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
    "ULLM_REQUIRE_HIP_ROPE_KERNEL",
    "ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL",
    "ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL",
    "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL",
    "ULLM_REQUIRE_HIP_TOP1_KERNEL",
)


def _sha(path: Path) -> str:
    return ORACLE.sha256_file(path)


def _sha_relaxed(path: Path) -> str:
    """Hash an active service binary without requiring its install-time nlink."""
    try:
        info = path.lstat()
    except OSError as error:
        raise ORACLE.OracleError(f"cannot stat active service binary: {path}") from error
    if path.is_symlink() or not path.is_file():
        raise ORACLE.OracleError(f"active service binary is not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ORACLE.OracleError(f"cannot read active service binary: {path}") from error
    if path.lstat() != info:
        raise ORACLE.OracleError(f"active service binary changed while reading: {path}")
    return digest.hexdigest()


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        actual = set(value) if isinstance(value, dict) else set()
        raise ORACLE.OracleError(f"{label} keys differ: missing={sorted(fields - actual)} extra={sorted(actual - fields)}")
    return value


def _validate_sha256s(root: Path, manifest: dict[str, Any]) -> None:
    sums = root / "SHA256SUMS"
    if sums.is_symlink() or not sums.is_file():
        raise ORACLE.OracleError("corrected path SHA256SUMS is not a regular file")
    expected_names = {"manifest.json", manifest["payload"]["file"], "runtime.json"}
    entries: dict[str, str] = {}
    for number, line in enumerate(sums.read_text(encoding="ascii").splitlines(), 1):
        parts = line.split("  ")
        if len(parts) != 2 or parts[1] in entries:
            raise ORACLE.OracleError(f"corrected path SHA256SUMS line {number} is invalid")
        ORACLE.ensure_sha256(parts[0], f"corrected path SHA256SUMS line {number}")
        entries[parts[1]] = parts[0]
    if set(entries) != expected_names:
        raise ORACLE.OracleError("corrected path SHA256SUMS coverage differs")
    actual_names = {path.name for path in root.iterdir() if not path.is_symlink() and path.is_file()}
    if actual_names != expected_names | {"SHA256SUMS"}:
        raise ORACLE.OracleError("corrected path root file coverage differs")
    for name, digest in entries.items():
        path = root / name
        if _sha(path) != digest:
            raise ORACLE.OracleError(f"corrected path SHA256SUMS digest differs: {name}")


def _validate_corrected_path(root: Path) -> dict[str, Any]:
    """Validate the v2 package-bound path contract without the mutable parent validator."""
    manifest = ORACLE.validate_manifest(root, expected_kind="path")
    runtime = _load_json(root / "runtime.json")
    runtime = _exact(
        runtime,
        {
            "all_m1", "artifact_manifest", "artifact_manifest_sha256", "binary", "device_index", "device_kind",
            "dtype", "evidence_scope", "execution_environment", "model_loads", "package_dir", "package_manifest",
            "package_manifest_sha256", "run", "runtime", "schema_version", "served_model_guard", "source_replay",
            "visible_devices",
        },
        "corrected path runtime",
    )
    if runtime["schema_version"] != "ullm.qwen35_aq4_path_oracle_runtime.v1" or runtime["runtime"] != "ullm-aq4-p2-path-oracle":
        raise ORACLE.OracleError("corrected path runtime schema differs")
    if runtime["evidence_scope"] != "production_gpu" or runtime["device_kind"] != "gpu" or runtime["device_index"] != 1 or runtime["visible_devices"] != "1" or runtime["dtype"] != "f32" or runtime["all_m1"] is not True or runtime["model_loads"] != 1:
        raise ORACLE.OracleError("corrected path runtime GPU contract differs")
    package_dir = Path(runtime["package_dir"])
    package_path = Path(runtime["package_manifest"])
    if not package_dir.is_absolute() or package_dir.is_symlink() or not package_dir.is_dir() or not package_path.is_absolute() or package_path.is_symlink() or not package_path.is_file():
        raise ORACLE.OracleError("corrected path package identity paths are invalid")
    try:
        package_path.resolve().relative_to(package_dir.resolve())
    except ValueError as error:
        raise ORACLE.OracleError("corrected path package manifest escapes package directory") from error
    package_sha = ORACLE.ensure_sha256(runtime["package_manifest_sha256"], "corrected path package hash")
    artifact = manifest["identity"]["artifact"]
    if package_sha != artifact["package_manifest_sha256"] or _sha(package_path) != package_sha:
        raise ORACLE.OracleError("corrected path package manifest hash differs")
    artifact_binding = artifact.get("artifact_binding_kind")
    if artifact_binding != "package_manifest" or artifact.get("artifact_manifest_sha256") is not None or runtime["artifact_manifest"] is not None or runtime["artifact_manifest_sha256"] is not None:
        raise ORACLE.OracleError("corrected path package-only artifact nullability differs")

    guard = _exact(runtime["served_model_guard"], {"evidence_scope", "manifest", "manifest_sha256", "package", "public", "required_environment", "required_environment_sha256", "worker"}, "corrected path served guard")
    if guard["evidence_scope"] != runtime["evidence_scope"] or tuple(guard["required_environment"]) != REQUIRED_HIP_KERNEL_ENV or guard["required_environment_sha256"] != ORACLE.canonical_sha256(list(REQUIRED_HIP_KERNEL_ENV)):
        raise ORACLE.OracleError("corrected path served guard environment differs")
    expected_environment = {name: "1" for name in REQUIRED_HIP_KERNEL_ENV} | {"HIP_VISIBLE_DEVICES": "1", "ULLM_HIP_VISIBLE_DEVICES": "1"}
    if runtime["execution_environment"] != expected_environment:
        raise ORACLE.OracleError("corrected path execution environment differs")
    served_path = Path(guard["manifest"])
    if not served_path.is_absolute() or served_path.is_symlink() or not served_path.is_file() or _sha_relaxed(served_path) != guard["manifest_sha256"]:
        raise ORACLE.OracleError("corrected path served-model manifest hash differs")
    if served_path.resolve() != Path(artifact["served_model_manifest_path"]).resolve() or guard["manifest_sha256"] != artifact["served_model_manifest_sha256"]:
        raise ORACLE.OracleError("corrected path served-model manifest binding differs")
    served = _load_json(served_path)
    if served.get("schema_version") != "ullm.served_model.v2" or not isinstance(served.get("product"), dict):
        raise ORACLE.OracleError("active served-model schema is invalid")
    product = served["product"]
    active_package = product.get("package")
    if not isinstance(active_package, dict) or product.get("artifact") is not None:
        raise ORACLE.OracleError("active served-model package-only identity differs")
    public = served.get("public")
    if not isinstance(public, dict) or not isinstance(public.get("upstream_id"), str) or not isinstance(public.get("revision"), str):
        raise ORACLE.OracleError("active served public identity is incomplete")
    if public["upstream_id"] != manifest["identity"]["model_id"] or _exact(guard["public"], {"model_id", "model_revision"}, "corrected path public binding") != {"model_id": public["upstream_id"], "model_revision": public["revision"]}:
        raise ORACLE.OracleError("corrected path public model identity differs")
    active_worker = served.get("worker")
    if not isinstance(active_worker, dict) or not isinstance(active_worker.get("binary"), str) or not isinstance(active_worker.get("binary_sha256"), str):
        raise ORACLE.OracleError("active served worker identity is incomplete")
    worker_binding = _exact(guard["worker"], {"binary_path", "binary_sha256", "device_architecture", "execution_profile"}, "corrected path worker binding")
    worker_path = Path(active_worker["binary"])
    worker_sha = ORACLE.ensure_sha256(active_worker["binary_sha256"], "active worker hash")
    if worker_path.is_symlink() or not worker_path.is_file() or _sha_relaxed(worker_path) != worker_sha or worker_binding["binary_path"] != str(worker_path.resolve()) or worker_binding["binary_sha256"] != worker_sha or worker_binding["device_architecture"] != "gfx1201" or worker_binding["execution_profile"] != "rdna4_aq4_resident":
        raise ORACLE.OracleError("corrected path active worker identity differs")
    package_binding = _exact(guard["package"], {"manifest_path", "manifest_sha256", "product_root"}, "corrected path package binding")
    product_root = Path(product.get("root", ""))
    active_package_path = product_root / str(active_package.get("manifest_path", ""))
    if not product_root.is_absolute() or product_root.resolve() != package_dir.resolve().parent or Path(package_binding["manifest_path"]).resolve() != package_path.resolve() or package_binding["manifest_sha256"] != package_sha or Path(package_binding["product_root"]).resolve() != product_root.resolve() or active_package_path.resolve() != package_path.resolve() or active_package.get("manifest_sha256") != package_sha:
        raise ORACLE.OracleError("corrected path served package binding differs")
    binary = _exact(runtime["binary"], {"path", "sha256"}, "corrected path binary")
    binary_path = Path(binary["path"])
    if not binary_path.is_absolute() or binary_path.is_symlink() or not binary_path.is_file() or binary_path.stat().st_nlink != 1 or binary_path.stat().st_mode & 0o002 or _sha(binary_path) != ORACLE.ensure_sha256(binary["sha256"], "corrected path binary hash"):
        raise ORACLE.OracleError("corrected path binary identity differs")
    run = _exact(runtime["run"], {"elapsed_seconds", "row_count"}, "corrected path run")
    if not isinstance(run["row_count"], int) or run["row_count"] != manifest["payload"]["record_count"] or not isinstance(run["elapsed_seconds"], (int, float)) or run["elapsed_seconds"] <= 0:
        raise ORACLE.OracleError("corrected path run summary differs")
    _validate_sha256s(root, manifest)
    return {"schema_version": "ullm.qwen35_aq4_p2_oracle_validator.v1", "status": "valid", "oracle_kind": manifest["oracle_kind"], "manifest_sha256": _sha(root / "manifest.json"), "payload_sha256": manifest["payload"]["sha256"], "record_count": manifest["payload"]["record_count"], "usable_as_path_evidence": manifest["usable_as_path_evidence"], "promotion_eligible": False, "blockers": []}


def _rust_replay_sha(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(b"ullm.qwen35_aq4.calibration_replay.v1\0")
    digest.update(len(token_ids).to_bytes(8, "little"))
    for token_id in token_ids:
        digest.update(token_id.to_bytes(8, "little"))
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = ORACLE.load_json(path)
    if not isinstance(value, dict):
        raise ORACLE.OracleError(f"{path} must contain an object")
    return value


def _validate_raw_root(raw_root: Path, expected: dict[str, str]) -> None:
    sums = raw_root / "SHA256SUMS"
    if _sha(sums) != expected.get("SHA256SUMS"):
        raise ORACLE.OracleError("raw evidence SHA256SUMS digest differs")
    for name, digest in expected.items():
        if name == "SHA256SUMS":
            continue
        path = raw_root / name
        if not path.is_file() or path.is_symlink() or _sha(path) != digest:
            raise ORACLE.OracleError(f"raw evidence hash differs: {name}")


def _validate_binary_binding(attestation: dict[str, Any], raw_root: Path) -> dict[str, Any]:
    execution = attestation.get("execution")
    if not isinstance(execution, dict):
        raise ORACLE.OracleError("attestation execution binding is missing")
    executed_path_value = execution.get("executed_path")
    executed_sha = execution.get("executed_binary_sha256")
    copy_path_value = execution.get("evidence_copy_path")
    copy_sha = execution.get("evidence_copy_sha256")
    if not all(isinstance(value, str) and value for value in (executed_path_value, executed_sha, copy_path_value, copy_sha)):
        raise ORACLE.OracleError("attestation must bind executed and detached binary paths and hashes")

    command = _load_json(raw_root / "command.json")
    command_binary = command.get("binary")
    if not isinstance(command_binary, str) or not command_binary:
        raise ORACLE.OracleError("raw command evidence has no binary path")
    executed_path = Path(executed_path_value)
    expected_executed_path = (ROOT / command_binary).resolve()
    if executed_path.is_symlink() or not executed_path.is_file() or executed_path.resolve() != expected_executed_path:
        raise ORACLE.OracleError("executed_path does not bind the raw command binary")
    observed_executed_sha = _sha_relaxed(executed_path)
    if observed_executed_sha != executed_sha:
        raise ORACLE.OracleError("executed binary SHA256 differs from attestation")

    copy_path = Path(copy_path_value)
    if copy_path.is_absolute():
        raise ORACLE.OracleError("evidence_copy_path must be relative to the raw evidence root")
    evidence_copy = (raw_root / copy_path).resolve()
    raw_resolved = raw_root.resolve()
    if not evidence_copy.is_relative_to(raw_resolved) or evidence_copy.is_symlink() or not evidence_copy.is_file():
        raise ORACLE.OracleError("evidence_copy_path is not a regular file in the raw evidence root")
    stat_result = evidence_copy.stat()
    if stat_result.st_nlink != 1:
        raise ORACLE.OracleError("evidence binary must be detached with nlink=1")
    if stat_result.st_mode & 0o002:
        raise ORACLE.OracleError("evidence binary must not be world-writable")
    observed_copy_sha = _sha(evidence_copy)
    if observed_copy_sha != copy_sha or observed_copy_sha != observed_executed_sha:
        raise ORACLE.OracleError("detached evidence binary SHA256 does not equal executed binary SHA256")
    worker_executed_value = execution.get("worker_executed_path")
    worker_executed_sha = execution.get("worker_executed_sha256")
    worker_copy_value = execution.get("worker_evidence_copy_path")
    worker_copy_sha = execution.get("worker_evidence_copy_sha256")
    if not all(isinstance(value, str) and value for value in (worker_executed_value, worker_executed_sha, worker_copy_value, worker_copy_sha)):
        raise ORACLE.OracleError("attestation must bind active and detached worker paths and hashes")
    worker_path = Path(worker_executed_value)
    if worker_path.is_symlink() or not worker_path.is_file() or _sha_relaxed(worker_path) != worker_executed_sha:
        raise ORACLE.OracleError("active served worker SHA256 differs from attestation")
    worker_copy_relative = Path(worker_copy_value)
    if worker_copy_relative.is_absolute():
        raise ORACLE.OracleError("worker_evidence_copy_path must be relative to the raw evidence root")
    worker_copy = (raw_root / worker_copy_relative).resolve()
    if not worker_copy.is_relative_to(raw_resolved) or worker_copy.is_symlink() or not worker_copy.is_file():
        raise ORACLE.OracleError("worker evidence copy is not a regular file in the raw evidence root")
    worker_stat = worker_copy.stat()
    if worker_stat.st_nlink != 1 or worker_stat.st_mode & 0o002:
        raise ORACLE.OracleError("worker evidence copy must be detached and not world-writable")
    if _sha(worker_copy) != worker_copy_sha or worker_copy_sha != worker_executed_sha:
        raise ORACLE.OracleError("detached worker evidence SHA256 does not equal active worker SHA256")
    return {
        "executed_path": str(executed_path.resolve()),
        "evidence_copy_path": str(evidence_copy),
        "sha256": observed_executed_sha,
        "evidence_copy_nlink": stat_result.st_nlink,
        "evidence_copy_mode": format(stat_result.st_mode & 0o777, "04o"),
        "worker_executed_path": str(worker_path.resolve()),
        "worker_evidence_copy_path": str(worker_copy),
        "worker_sha256": worker_executed_sha,
        "worker_evidence_copy_nlink": worker_stat.st_nlink,
        "worker_evidence_copy_mode": format(worker_stat.st_mode & 0o777, "04o"),
    }


def _validate_device(raw_root: Path, command: dict[str, Any]) -> dict[str, Any]:
    if command.get("device_index") != 1 or command.get("physical_gpu_index") != 2 or command.get("gfx_architecture", command.get("rocm_architecture")) != "gfx1201":
        raise ORACLE.OracleError("declared GPU mapping is not runtime-index 1 -> physical GPU 2 gfx1201")
    if command.get("environment") != {"HIP_VISIBLE_DEVICES": "1", "ULLM_HIP_VISIBLE_DEVICES": "1"}:
        raise ORACLE.OracleError("HIP visible-device environment is not exact")
    devices = (raw_root / "devices.txt").read_text(encoding="utf-8")
    if devices.count("gfx1201") != 1 or "GPU[2]" not in devices:
        raise ORACLE.OracleError("ROCm device evidence does not identify one GPU[2] gfx1201")
    monitor = (raw_root / "monitor.log").read_text(encoding="utf-8")
    if "--device-index 1" not in monitor or "total=34208743424" not in monitor:
        raise ORACLE.OracleError("monitor evidence does not bind runtime device index and VRAM total")
    samples = [(int(used), float(power)) for used, power in re.findall(r"used=(\d+) power=([0-9.]+)", monitor)]
    if not samples:
        raise ORACLE.OracleError("monitor evidence has no resource sample")
    used, power = max(samples, key=lambda item: item[0])
    if used != 7343022080 or power != 21.0:
        raise ORACLE.OracleError("monitor peak resource sample differs")
    return {"physical_gpu_index": 2, "runtime_device_index": 1, "gfx_architecture": "gfx1201", "vram_total_bytes": 34208743424, "vram_baseline_bytes": 87384064, "vram_peak_bytes": used, "power_peak_watts": power}


def _validate_replay(source_root: Path, path_root: Path, cases_path: Path) -> dict[str, Any]:
    source = ORACLE.validate_manifest(source_root, expected_kind="independent_source")
    path = ORACLE.validate_manifest(path_root, expected_kind="same_artifact_all_m1")
    cases_doc = _load_json(cases_path)
    cases = cases_doc.get("cases")
    if not isinstance(cases, list):
        raise ORACLE.OracleError("cases JSON has no cases list")
    source_rows: dict[str, list[int]] = {}
    for row in ORACLE.payload_records(source_root, source):
        source_rows.setdefault(row["case_id"], []).append(row["greedy_token_id"])
    runtime = _load_json(path_root / "runtime.json")
    binding = runtime.get("source_replay", {}).get("cases")
    if not isinstance(binding, list):
        raise ORACLE.OracleError("path runtime has no per-case source replay binding")
    observed = {item.get("case_id"): item for item in binding if isinstance(item, dict)}
    case_results = []
    for case in source["cases"]:
        case_id = case["case_id"]
        case_input = next((item for item in cases if item.get("case_id") == case_id), None)
        if case_input is None or ORACLE.canonical_token_ids_hash(case_input.get("prompt_token_ids", [])) != case["prompt_token_ids_sha256"]:
            raise ORACLE.OracleError(f"case prompt token hash differs for {case_id}")
        tokens = source_rows.get(case_id, [])
        item = observed.get(case_id)
        expected_sha = _rust_replay_sha(tokens)
        expected_contexts = [
            {"step": step, "length": len(case_input["prompt_token_ids"]) + step, "token_ids_sha256": ORACLE.canonical_token_ids_hash(case_input["prompt_token_ids"] + tokens[:step])}
            for step in range(case["step_count"])
        ]
        ok = bool(item and item.get("length") == len(tokens) == case["step_count"] and item.get("source_sequence_sha256") == expected_sha and item.get("contexts") == expected_contexts)
        case_results.append({"case_id": case_id, "length": len(tokens), "source_sequence_sha256": expected_sha, "contexts": expected_contexts, "position_semantics": "step_i_uses_prompt_plus_greedy_tokens_before_i", "exact": ok})
    return {"status": "valid" if all(item["exact"] for item in case_results) else "blocked", "cases": case_results}


def validate(args: argparse.Namespace) -> dict[str, Any]:
    attestation = _load_json(args.attestation)
    if attestation.get("schema_version") != "ullm.qwen35_aq4_path_oracle_attestation.v1":
        raise ORACLE.OracleError("attestation schema is invalid")
    raw = attestation.get("raw_evidence")
    if not isinstance(raw, dict) or raw.get("root") != args.raw_root.name:
        raise ORACLE.OracleError("attestation raw evidence root differs")
    _validate_raw_root(args.raw_root, raw["files"])
    binary = _validate_binary_binding(attestation, args.raw_root)
    device = _validate_device(args.raw_root, attestation["execution"])
    # v1 is intentionally retained as immutable raw evidence.  Validate its
    # manifest contract directly and record its legacy CPU metadata blocker;
    # do not delegate to the mutable parent validator.
    base_manifest = ORACLE.validate_manifest(args.base_path, expected_kind="path")
    base_report = {
        "schema_version": "ullm.qwen35_aq4_p2_oracle_validator.v1",
        "status": "valid",
        "oracle_kind": base_manifest["oracle_kind"],
        "manifest_sha256": _sha(args.base_path / "manifest.json"),
        "payload_sha256": base_manifest["payload"]["sha256"],
        "record_count": base_manifest["payload"]["record_count"],
        "usable_as_path_evidence": False,
        "promotion_eligible": False,
        "blockers": ["metadata_invalid: runtime.json declared cpu for the GPU execution"],
    }
    path_report = _validate_corrected_path(args.path)
    if base_report["manifest_sha256"] != attestation["base_path"]["manifest_sha256"] or base_report["payload_sha256"] != attestation["base_path"]["payload_sha256"] or _sha(args.base_path / "runtime.json") != attestation["base_path"]["runtime_sha256"]:
        raise ORACLE.OracleError("base path oracle hash binding differs")
    if path_report["manifest_sha256"] != attestation["corrected_path"]["manifest_sha256"] or path_report["payload_sha256"] != attestation["corrected_path"]["payload_sha256"] or _sha(args.path / "runtime.json") != attestation["corrected_path"]["runtime_sha256"]:
        raise ORACLE.OracleError("corrected path oracle hash binding differs")
    runtime = _load_json(args.path / "runtime.json")
    base_runtime = _load_json(args.base_path / "runtime.json")
    base_metadata_invalid = base_runtime.get("device") == "cpu"
    if runtime.get("device_kind") != "gpu" or runtime.get("device_index") != 1 or runtime.get("visible_devices") != "1" or runtime.get("evidence_scope") != "production_gpu":
        raise ORACLE.OracleError("corrected runtime device metadata differs")
    replay = _validate_replay(args.source_path, args.path, args.cases)
    comparison = ORACLE.compare_payloads(args.source_path, ORACLE.validate_manifest(args.source_path, expected_kind="source"), args.path, ORACLE.validate_manifest(args.path, expected_kind="path"))
    policy = attestation.get("policy_audit")
    if not isinstance(policy, dict) or policy.get("status") != "blocked_unbound" or policy.get("values") is not None:
        raise ORACLE.OracleError("AQ4 P2 threshold policy is unexpectedly bound or malformed")
    for key in ("template_path", "prefill_validation_spec_path", "threshold_audit_path"):
        policy_path = ROOT / policy[key]
        hash_key = {"template_path": "template_sha256", "prefill_validation_spec_path": "prefill_validation_spec_sha256", "threshold_audit_path": "threshold_audit_sha256"}[key]
        if _sha(policy_path) != policy[hash_key]:
            raise ORACLE.OracleError(f"policy audit source hash differs: {policy_path}")
    blockers = []
    if base_metadata_invalid:
        blockers.append("base_v1_metadata_invalid: runtime.json declared cpu for the GPU execution")
    if replay["status"] != "valid":
        blockers.append("replay_step_alignment_invalid")
    if not comparison["greedy_token_exact"] or not comparison["topk_exact"] or not comparison["hidden_sample_within_atol"] or not comparison["logit_sample_within_atol"]:
        blockers.append("source_comparison_bounded_agreement_failed")
    if not comparison["hidden_sample_shape_exact"] or not comparison["logit_sample_shape_exact"]:
        blockers.append("source_comparison_bounded_shape_failed")
    blockers.append("policy_missing: no hash-bound AQ4 P2 bounded relative-L2/cosine/top-k policy")
    blockers.append("path_regression_requires_exact_same_artifact_all_m1_contract")
    return {"schema_version": "ullm.qwen35_aq4_path_oracle_attestation_validator.v1", "status": "valid_with_blockers", "metadata_valid": not base_metadata_invalid, "base_path": base_report, "corrected_path": path_report, "execution": {"device": device, "binary": binary, "service_recovery": attestation["execution"]["service_recovery"]}, "step_alignment": replay, "source_comparison": comparison, "path_regression": {"status": "diagnostic_only", "all_m1": True, "exact_greedy": comparison["greedy_token_exact"], "exact_topk": comparison["topk_exact"]}, "policy_audit": policy, "blockers": blockers}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(validate(args), ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, ORACLE.OracleError) as error:
        print(f"Qwen3.5 AQ4 P2 path attestation validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
