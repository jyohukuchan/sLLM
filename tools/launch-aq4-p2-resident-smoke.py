#!/usr/bin/env python3
"""Immutable L launcher for the AQ4 P2 resident one-case smoke trust chain."""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v2"
BINDING_ROOT = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-binding-v7"
BINDING_MANIFEST = BINDING_ROOT / "binding-manifest.json"
RUNNER = BINDING_ROOT / "trusted-runner.py"
VALIDATOR = ROOT / "tools/prepare-aq4-p2-resident-smoke-bundle.py"
PYTHON = Path("/usr/bin/python3.12")
RESIDENT_DRIVER = INPUT_ROOT / "resident-driver"
SERVED_MANIFEST = Path("/etc/ullm/served-models/active.json")
LOCK_PATH = Path("/run/ullm/r9700.lock")
RUNNER_OUTPUT = Path("/tmp/ullm-aq4-p2-resident-smoke-L-dry-run")
EXECUTE_BINDING_ROOT = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-execute-binding-v11"
EXECUTE_BINDING_PATH = EXECUTE_BINDING_ROOT / "execute-binding.json"
EXECUTE_LAUNCHER_TRUST_PATH = EXECUTE_BINDING_ROOT / "launcher-trust.json"
EXECUTE_RUN_OUTPUT = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-execute-v11"
EXECUTE_EVIDENCE_OUTPUT = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-execute-evidence-v11"
LIVE_PREFLIGHT_PATH = EXECUTE_EVIDENCE_OUTPUT / "live-preflight.json"
AMD_SMI = Path("/opt/rocm/bin/amd-smi")
AMD_SMI_REAL = Path("/opt/rocm-7.2.1/libexec/amdsmi_cli/amdsmi_cli.py")
ROCM_LINK = Path("/opt/rocm")
ROCM_ALTERNATIVE_LINK = Path("/etc/alternatives/rocm")
ROCMINFO = Path("/usr/bin/rocminfo")
ROCMINFO_REAL = Path("/opt/rocm-7.2.1/bin/rocminfo")
ROCMINFO_ALTERNATIVE_LINK = Path("/etc/alternatives/rocminfo")
SYSTEMCTL = Path("/usr/bin/systemctl")
PGREP = Path("/usr/bin/pgrep")
SUDO = Path("/usr/bin/sudo")
WORKER = ROOT / "target/reasoning-v2/release/ullm-aq4-worker"
KFD_PROC_ROOT = Path("/sys/class/kfd/kfd/proc")

INPUT_ROOT_DEVICE = 66306
INPUT_ROOT_INODE = 10617640
INPUT_FINGERPRINT_SHA = "584ab574b6283b21f13216c265cd16c88abb836b1c94b2630127247215f633d1"
BINDING_ROOT_DEVICE = 66306
BINDING_ROOT_INODE = 10617667
BINDING_ROOT_MODE = 0o555
BINDING_MANIFEST_SHA = "3b99dcfd11f9c4726a8531f9f828ec62dd84fabe577b6b529636ee0b66918579"
BINDING_PLAN_SHA = "b0acd2da4ab5b4c6bfd338358ac95b55280cc5fb05f885675ff5a1b27429a280"
RUNNER_COMMIT = "d367b6da07393f55c720ded7250bda8cdc402a79"
RUNNER_TREE = "8fea6bf90e8ad99c7ed36c719b8b4ad204ce73df"
RUNNER_GIT_BLOB = "fed94b749790cdbf6a61e33f3f9e95ebd73502e0"
RUNNER_SHA = "98e324414d9e2d7e6db5b066209e6f7c6734e391502ae81ecd1809e8ec558e7f"
VALIDATOR_COMMIT = "e36a03ad423a0bb45cc1e4de67d3ca4fddfacdbc"
VALIDATOR_TREE = "9189252d996e2eda05761f650960224676867811"
VALIDATOR_GIT_BLOB = "5ee4278a58b18454cf714da6bfe540f5d2ff832c"
VALIDATOR_SHA = "15a65fed6d182e706473821f128fbae02214ab0bf988bb7c1f363f69233e9904"
PYTHON_SHA = "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
RESIDENT_COMMIT = "43ba16f2347a45caba8a60cac2189714118db280"
RESIDENT_TREE = "72392a7114f5968d6c2ad05e24762a6790000013"
RESIDENT_GIT_BLOB = "7e37119cc8b66dc0e0f7abcf49b896fcdad8315f"
RESIDENT_SOURCE_SHA = "0acb46d1ab8730267edf40b505224ff157760ec19aa40a07ee1b389860ec54bf"
RESIDENT_SHA = "d7458fcdf8553871cac00123413676625c61eff2fdee3be9a440e656f05bcc1e"
RESIDENT_BYTES = 3505000
RESIDENT_BUILD_ID = "033ce9b214e2149861a8fcf0381c27bbac5bf1d1"
RESIDENT_BUILD_INPUTS = {
    ".cargo/config.toml": {
        "git_blob": "6dee7973a174f5e45c5762d91522f5d6849a5b84",
        "role": "cargo_linker_configuration",
        "sha256": "41627bf0cfcb00817cd6bee0285a01d25c89614bb271798139c26539c525f67d",
    },
    "Cargo.lock": {
        "git_blob": "fb12cb0388ea1c6fc6368e7ea5d5100c11a20666",
        "role": "locked_dependency_graph",
        "sha256": "10df8371ae3a33ed792dc4e8c15dd6196a8a7e176e377ef275e75b3219aa157b",
    },
    "crates/ullm-runtime-sys/build.rs": {
        "git_blob": "bfd7a966b465e6f61189ae7cae8432065f102b6f",
        "role": "runtime_static_library_build",
        "sha256": "e2d29a16e4e6be98e8cc5f41f7350e8210d707229a204fb7c0b35a9ef0d096ea",
    },
    "runtime/src/ullm_runtime_parts/part_00.inc": {
        "git_blob": "316d3ae5c13f79678fb8256aa8c66ea7e154660f",
        "role": "directional_hip_copy_runtime",
        "sha256": "db138bfaf33f59708f24edbec8352a39fe809ff39422d5b742399752c8fa9f5f",
    },
}
RESIDENT_BUILD_METADATA = {
    "command": "CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/tmp/ullm-profile-v10-resident-target-a cargo build --locked --release -p ullm-engine --bin ullm-aq4-p2-resident-driver",
    "provenance": "clean Git worktree at the exact source commit with an initially absent independent target directory",
    "source_commit": RESIDENT_COMMIT,
    "source_tree": RESIDENT_TREE,
    "source_git_blob": RESIDENT_GIT_BLOB,
    "source_sha256": RESIDENT_SOURCE_SHA,
    "build_inputs": RESIDENT_BUILD_INPUTS,
    "cargo_version": "1.96.0 (30a34c682 2026-05-25)",
    "rustc_version": "1.96.0 (ac68faa20 2026-05-25)",
    "rustc_host": "x86_64-unknown-linux-gnu",
    "llvm_version": "22.1.2",
    "clang_version": "Ubuntu clang version 18.1.3 (1ubuntu1)",
    "cxx_version": "c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0",
    "mold_version": "mold 2.30.0 (compatible with GNU ld)",
    "rocm_resolved_path": "/opt/rocm-7.2.1",
    "cargo_build_jobs": 1,
    "cargo_incremental": False,
    "locked": True,
    "profile": "release",
    "binary_bytes": RESIDENT_BYTES,
    "binary_build_id_sha1": RESIDENT_BUILD_ID,
    "expected_binary_sha256": RESIDENT_SHA,
    "reproducibility": {
        "build_count": 2,
        "build_id_equal": True,
        "byte_identical": True,
        "bytes_equal": True,
        "commands": [
            "CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/tmp/ullm-profile-v10-resident-target-a cargo build --locked --release -p ullm-engine --bin ullm-aq4-p2-resident-driver",
            "CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/tmp/ullm-profile-v10-resident-target-b cargo build --locked --release -p ullm-engine --bin ullm-aq4-p2-resident-driver",
        ],
        "independent_initially_absent_target_directories": True,
        "sha256_equal": True,
    },
}
SERVED_SHA = "feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44"
DEVICE_INDEX = 1
AMD_SMI_INDEX = 2
GPU_BDF = "0000:47:00.0"
GPU_UUID = "a8ff7551-0000-1000-80e9-ddefa2d60f55"
KFD_ID = 51545
CASE_ID = "p2-representative-full_model-cold_prefill-cold_batched-n128-m128-r9700-rdna4-aq4_0_target"
CASE_SHA = "d83a420476bde889c7c8014d7982fd52e0f61ab09b888f66415d0ac9fb443ae7"
DRY_L_COMMIT = "2ff2e7c4172a2edee49dfce67b07009364a2f958"
DRY_L_TREE = "f7d553a0901af033c86faa09eb966dce8255e065"
DRY_L_GIT_BLOB = "9e9cee31e23559e440cbd1a074784eaae97fae57"
DRY_L_SHA = "8cd38aabc60eba5dfdcc3adc46421cbe7508bfd95bcb1d8b56b410f1a0f1fa81"
AMD_SMI_SHA = "c6185991e96dc45b3ae930eace23869f070fce2afab5e061a336c0a7e2e9fa4a"
ROCMINFO_SHA = "e22d9361a66797b4f5fc8ff1a305f1492e70d323f76b7bd89b7db2a981b567ed"
SYSTEMCTL_SHA = "7ba82b5ba146759c710e1b80fadaa3fdbc0f9b85c8fb2c8c3196b7b1a0037ef8"
PGREP_SHA = "8e1a7f00f33b9447e24835307cef71800677a2fe2975c8a1632b613109816b52"
SUDO_SHA = "136f2e48b0295b9fc595b8259cf2411ac43f27ddbfe02b956649ddaa2e92b9fa"
EXECUTE_RUN_ID = "p2-r9700-resident-one-case-smoke-execute-v11"
PROFILE_RUN_ID = "p2-r9700-resident-one-case-smoke-profile-diagnostic-v11"
PROFILE_RUN_OUTPUT = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-execute-v11"
PROFILE_EVIDENCE_OUTPUT = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-execute-evidence-v11"
PROFILE_LIVE_PREFLIGHT_PATH = PROFILE_EVIDENCE_OUTPUT / "live-preflight.json"
PROFILE_RUNNER_TARGET_MANIFEST_NAME = "runner-target-command-manifest.json"
PROFILE_CAPTURE_OUTPUT_DIRECTORY = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p3/aq4-p3-diagnostic-rocprof-capture-v11"
PROFILE_CAPTURE_ARTIFACT = PROFILE_CAPTURE_OUTPUT_DIRECTORY / "capture-artifact.json"
PROFILE_CAPTURE_FAILURE = PROFILE_CAPTURE_OUTPUT_DIRECTORY / "capture-failure.json"
ROCTX_LIBRARY = Path("/opt/rocm-7.2.1/lib/librocprofiler-sdk-roctx.so.1.1.0")
ROCTX_LIBRARY_RESOLVED = Path("/opt/rocm-7.2.1/lib/librocprofiler-sdk-roctx.so.1.1.0")
ROCTX_LIBRARY_SHA = "1a5831a3817eac29f63d1442dc348ba31b417202b7ce15f3aed9c09a8f4773c9"
ROCTX_LIBRARY_SONAME = "librocprofiler-sdk-roctx.so.1"
ROCTX_LIBRARY_MODE = 0o644
ROCTX_LIBRARY_BYTES = 456232
PROFILE_PRODUCER_HELPER = ROOT / "tools/build-aq4-p3-selection-raw.py"
PROFILE_PRODUCER_HELPER_SHA = "a589c3e644d36132fb6054afdb15b27543d8e8181e3c737dcbd071d7c52e3d20"
PROFILE_SELECTOR_HELPER = ROOT / "tools/select-aq4-p3-candidate.py"
PROFILE_SELECTOR_HELPER_SHA = "4a510c7351131072ed368e2ac8fffeb2daf10488edef94c37fe5dbcb729e9739"
PROFILE_FAMILY_HELPER = ROOT / "tools/profile-aq4-p2-family-exclusive.py"
PROFILE_FAMILY_HELPER_SHA = "f8d32c340231e329f004d9e16192c02378f1fd58b8ab713e8efbbd3029b052d6"
SUDO_KEEPALIVE_SECONDS = 30.0
SERVICE_UNITS = ("ullm-openai.service", "llama-qwen35-udq4.service")
GUARD_NAMES = (
    "ULLM_REQUIRE_HIP_ADD_KERNEL", "ULLM_REQUIRE_HIP_AQ4_KERNEL", "ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL", "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL", "ULLM_REQUIRE_HIP_AQ4_MATVEC_PAIR_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL", "ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL", "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL", "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL", "ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL", "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_SEQUENCE_KERNEL", "ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL", "ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL", "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL",
    "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL", "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL",
    "ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL", "ULLM_REQUIRE_HIP_RMSNORM_KERNEL", "ULLM_REQUIRE_HIP_ROPE_KERNEL",
    "ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL", "ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL",
    "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL", "ULLM_REQUIRE_HIP_TOP1_KERNEL",
)
EXECUTE_ENV = {"HIP_VISIBLE_DEVICES": "1", "ULLM_HIP_VISIBLE_DEVICES": "1", "ULLM_SERVED_MODEL_MANIFEST": str(SERVED_MANIFEST), "ULLM_BUILD_GIT_COMMIT": RESIDENT_COMMIT} | {name: "1" for name in GUARD_NAMES}

INPUT_MEMBER_SHA = {
    "SHA256SUMS": "27579326b1ba703585d4683ddcabf47676debeca58878e54a2ae2cccee6e99b9",
    "SUPERSEDED-0fd7993.json": "ef18c2289b56647ece390683af1e562aea6df071d5fc22cfbf5e63b0ed058a7b",
    "bundle.json": "b947e43d6aff609967ea9b0909aa14a689b4f35791a7903134453c42ede8fb38",
    "case-binding.json": "1c8cf17475c0840900ebcc5cd9334d4ebe76c1bd354aaa5106cb875efa1da8b5",
    "dry-run.json": "7a6738f6ebdc3b07b388753c8152d052d3e329e50a44e30e40b6fa97e2269037",
    "fake-ready.json": "b353d412fc11a7656ed1eb522cdc77bd29412ef9b0f702eb4615263a9273427c",
    "fixture-index.json": "9bde3a24bd8bb3ff017f5a370cee505e16cb632182fdbf3ad8c2cfb33c23bc4d",
    "fixture.json": "a61c977a7671e7e3d141b87fc84e20e9957be71706cface1988d03054f2dad50",
    "identity.json": "6c3e3562eaafbc357497153f46a12c726383130b8b7f4e630ba50c769d03fc37",
    "launch-command.json": "6a963811ccab739958bfbf6846eaed9cdb16e45c9e533e47c7a97bd307731383",
    "official-case.json": "916abe28bd0b82bde85f918d6a6b1a1f54a49d14149f4832f042b17d2eb15bc2",
    "package-manifest.json": "a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad",
    "policy.json": "21dff8ecdbc17a1cd86a458fe7f8771eed0cdd18577a5f0fb6c7b96310a2de16",
    "preflight.json": "294ddf1771251c4b1954ea663d73e85821749119da2a4f6c7528fdae510bbc6e",
    "resident-driver": RESIDENT_SHA,
    "runner-dry-run-evidence.json": "4f37d7bba1320ff2dd57c0249b0d18ba3e0f1ee45c68cbb5dfb373c54de9ddcf",
    "served-model.json": SERVED_SHA,
    "trust-roots.json": "b6b5d0975e45a40b2a42d1a8f07345caf510fc15b2c3ce39e23b1ba05badaacd",
    "trusted-runner.py": "62cf9cd77863d18158afad8955b7d09c2c0f8b09046869bacb25c91c789878e0",
}
BINDING_MEMBER_SHA = {
    "binding-manifest.json": BINDING_MANIFEST_SHA,
    "runner-plan.json": BINDING_PLAN_SHA,
    "runner-subprocess-evidence.json": "16c8a4ba60bdeb5debab7e4c70926414be9410fe915ed622353b90fb6f3effd3",
    "trusted-runner.py": RUNNER_SHA,
    "trusted-validator.py": VALIDATOR_SHA,
    "validator-report.json": "8df8debd89f1dbc9f866a8ec67a862d4daf538e264778a07607ea5189e830b62",
    "SHA256SUMS": "e922a38142380bbee3e7e4db18d195f93aad86f4b1123e0034f9149edf1f9918",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_BYTES = 64 * 1024 * 1024
CHUNK = 1024 * 1024
_AFTER_VALIDATOR_HOOK: Callable[[], None] | None = None
_BEFORE_FINAL_VERIFY_HOOK: Callable[[], None] | None = None


class LauncherError(ValueError):
    pass


class AmdProcessSchemaError(LauncherError):
    def __init__(self, diagnostic: dict[str, Any]) -> None:
        self.diagnostic = diagnostic
        super().__init__(f"amd-smi process schema rejected: {diagnostic['reason_code']}")


class KfdOwnerScanError(LauncherError):
    def __init__(self, diagnostic: dict[str, Any]) -> None:
        self.diagnostic = diagnostic
        super().__init__(f"KFD owner scan rejected: {diagnostic['reason_code']}")


class _KfdEntryDisappeared(Exception):
    def __init__(self, stage: str, pid: int | None = None) -> None:
        self.stage = stage
        self.pid = pid


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pretty(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise LauncherError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "bool"
    if type(value) is int:
        return "int"
    if type(value) is float:
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _amd_process_diagnostic(raw: bytes, value: Any = None) -> dict[str, Any]:
    top_is_list = isinstance(value, list)
    root = value[0] if top_is_list and len(value) == 1 and isinstance(value[0], dict) else None
    process_list = root.get("process_list") if isinstance(root, dict) else None
    entries = process_list if isinstance(process_list, list) else []
    return {
        "schema_version": "ullm.aq4_p2_amd_process_parse_diagnostic.v1",
        "status": "rejected",
        "reason_code": "unclassified",
        "raw_sha256": sha_bytes(raw),
        "raw_bytes": len(raw),
        "top_level_type": _json_type(value),
        "top_level_length": len(value) if top_is_list else None,
        "root_keys": sorted(root) if isinstance(root, dict) and all(isinstance(key, str) for key in root) else None,
        "process_list_type": _json_type(process_list),
        "process_list_length": len(process_list) if isinstance(process_list, list) else None,
        "entry_key_sets": [sorted(entry) if isinstance(entry, dict) and all(isinstance(key, str) for key in entry) else None for entry in entries],
        "process_info_types": [_json_type(entry.get("process_info")) if isinstance(entry, dict) else _json_type(entry) for entry in entries],
    }


def parse_amd_process_owners(raw: bytes) -> dict[str, Any]:
    value: Any = None
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(LauncherError(f"non-finite JSON: {item}")))
    except (UnicodeError, json.JSONDecodeError, LauncherError):
        diagnostic = _amd_process_diagnostic(raw)
        diagnostic["reason_code"] = "invalid_json"
        raise AmdProcessSchemaError(diagnostic) from None
    diagnostic = _amd_process_diagnostic(raw, value)

    def reject(reason_code: str) -> None:
        diagnostic["reason_code"] = reason_code
        raise AmdProcessSchemaError(diagnostic)

    if not isinstance(value, list):
        reject("top_level_not_list")
    if len(value) != 1:
        reject("top_level_length_not_one")
    root = value[0]
    if not isinstance(root, dict):
        reject("gpu_root_not_object")
    if set(root) != {"gpu", "process_list"}:
        reject("gpu_root_keys_differ")
    if type(root["gpu"]) is not int or root["gpu"] != AMD_SMI_INDEX:
        reject("gpu_index_differs")
    process_list = root["process_list"]
    if not isinstance(process_list, list) or not process_list:
        reject("process_list_not_nonempty_list")
    zero_sentinel = [{"process_info": "No running processes detected"}]
    if process_list == zero_sentinel:
        diagnostic.update({"status": "accepted", "reason_code": "accepted_zero_sentinel"})
        return {"owners": [], "diagnostic": diagnostic}
    if any(isinstance(entry, dict) and isinstance(entry.get("process_info"), str) for entry in process_list):
        reject("sentinel_mixed_or_unknown")
    owners: list[int] = []
    expected_info_keys = {"name", "pid", "mem_usage", "cu_occupancy", "evicted_time"}
    for entry in process_list:
        if not isinstance(entry, dict) or set(entry) != {"process_info"}:
            reject("process_entry_keys_differ")
        info = entry["process_info"]
        if not isinstance(info, dict) or set(info) != expected_info_keys:
            reject("process_info_keys_differ")
        name = info["name"]
        pid = info["pid"]
        memory = info["mem_usage"]
        occupancy = info["cu_occupancy"]
        evicted = info["evicted_time"]
        if not isinstance(name, str) or not name.startswith("/") or "\x00" in name:
            reject("process_name_differs")
        if type(pid) is not int or pid <= 0:
            reject("process_pid_differs")
        if not isinstance(memory, dict) or set(memory) != {"value", "unit"} or type(memory.get("value")) is not int or memory["value"] < 0 or memory.get("unit") != "B":
            reject("process_mem_usage_differs")
        if not ((type(occupancy) is int and occupancy >= 0) or occupancy == "N/A"):
            reject("process_cu_occupancy_differs")
        if not isinstance(evicted, dict) or set(evicted) != {"value", "unit"} or type(evicted.get("value")) is not int or evicted["value"] < 0 or evicted.get("unit") != "ms":
            reject("process_evicted_time_differs")
        owners.append(pid)
    if len(set(owners)) != len(owners):
        reject("duplicate_process_pid")
    diagnostic.update({"status": "accepted", "reason_code": "accepted_owner_records"})
    return {"owners": sorted(owners), "diagnostic": diagnostic}


def parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(LauncherError(f"non-finite JSON: {item}")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise LauncherError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise LauncherError(f"{label} root must be an object")
    return value


def file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def reject_symlink_components(path: Path, label: str, *, allow_missing_leaf: bool = False) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise LauncherError(f"{label} must be absolute without parent traversal")
    current = Path(path.anchor)
    for index, part in enumerate(path.parts[1:], 1):
        current /= part
        try:
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise LauncherError(f"{label} has a symlink component: {current}")
        except FileNotFoundError:
            if allow_missing_leaf and index == len(path.parts) - 1:
                return
            raise LauncherError(f"{label} component is missing: {current}")


def ensure_directory_chain(path: Path, label: str) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise LauncherError(f"{label} must be absolute without parent traversal")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir(mode=0o755)
            except FileExistsError:
                metadata = current.lstat()
            else:
                metadata = current.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise LauncherError(f"{label} has a non-directory or symlink component: {current}")


def read_regular(path: Path, label: str, *, maximum: int | None = MAX_BYTES) -> tuple[bytes, tuple[int, ...]]:
    reject_symlink_components(path, label)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise LauncherError(f"{label} must be a single-link regular file")
    if maximum is not None and before.st_size > maximum:
        raise LauncherError(f"{label} exceeds size bound")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        if file_identity(os.fstat(descriptor)) != file_identity(before):
            raise LauncherError(f"{label} changed while opening")
        while chunk := os.read(descriptor, CHUNK):
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if file_identity(after) != file_identity(before) or file_identity(path.lstat()) != file_identity(before):
        raise LauncherError(f"{label} changed while reading")
    return b"".join(chunks), file_identity(before)


def sha_file(path: Path, label: str) -> tuple[str, tuple[int, ...]]:
    raw, identity = read_regular(path, label, maximum=None)
    return sha_bytes(raw), identity


class Snapshot:
    def __init__(self) -> None:
        self.files: dict[Path, tuple[tuple[int, ...], str, str]] = {}
        self.directories: dict[Path, tuple[tuple[int, ...], str]] = {}
        self.symlinks: dict[Path, tuple[tuple[int, ...], str, str]] = {}

    def file(self, path: Path, expected_sha: str, label: str) -> bytes:
        raw, identity = read_regular(path, label, maximum=None)
        if sha_bytes(raw) != expected_sha:
            raise LauncherError(f"{label} SHA differs")
        self.files[path] = (identity, expected_sha, label)
        return raw

    def directory(self, path: Path, device: int, inode: int, label: str) -> None:
        reject_symlink_components(path, label)
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_dev != device or metadata.st_ino != inode:
            raise LauncherError(f"{label} identity differs")
        self.directories[path] = (file_identity(metadata), label)

    def symlink(self, path: Path, expected_target: str, label: str) -> None:
        metadata = path.lstat()
        if not stat.S_ISLNK(metadata.st_mode) or os.readlink(path) != expected_target:
            raise LauncherError(f"{label} symlink binding differs")
        self.symlinks[path] = (file_identity(metadata), expected_target, label)

    def verify(self) -> None:
        for path, (expected_identity, expected_sha, label) in self.files.items():
            try:
                raw, observed_identity = read_regular(path, f"late {label}", maximum=None)
            except (LauncherError, OSError) as error:
                raise LauncherError(f"late replacement detected: {path}") from error
            if observed_identity != expected_identity or sha_bytes(raw) != expected_sha:
                raise LauncherError(f"late replacement detected: {path}")
        for path, (expected_identity, label) in self.directories.items():
            metadata = path.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or file_identity(metadata) != expected_identity:
                raise LauncherError(f"late directory replacement detected: {label}")
        for path, (expected_identity, expected_target, label) in self.symlinks.items():
            metadata = path.lstat()
            if not stat.S_ISLNK(metadata.st_mode) or file_identity(metadata) != expected_identity or os.readlink(path) != expected_target:
                raise LauncherError(f"late symlink replacement detected: {label}")


def validate_binding_manifest(raw: bytes) -> dict[str, Any]:
    manifest = parse_json(raw, "B binding manifest")
    exact = {"schema_version", "status", "promotion", "launch_eligible", "requires_immutable_launcher", "predecessor", "trust_roots", "input_root", "outputs", "execution", "cycle_control", "next_stage", "runner_roles", "binding_root_contract"}
    if set(manifest) != exact:
        raise LauncherError("B binding manifest exact schema differs")
    if manifest.get("schema_version") != "ullm.aq4_p2_resident_smoke_binding.v7" or manifest.get("status") != "prepared_not_executed" or manifest.get("promotion") is not False:
        raise LauncherError("B binding status/promotion differs")
    if manifest.get("binding_root_contract") != {"type": "directory", "mode": "0555", "members_read_only": True, "members_single_link": True}:
        raise LauncherError("B binding root contract differs")
    if manifest.get("launch_eligible") is not False or manifest.get("requires_immutable_launcher") is not True:
        raise LauncherError("B binding L boundary differs")
    if manifest.get("runner_roles") != {
        "binding_actual": {
            "commit": RUNNER_COMMIT,
            "execution_closure": "code_execution/exec",
            "role": "actual_generic_runner",
            "sha256": RUNNER_SHA,
        },
        "prepared_bootstrap": {
            "commit": "410d6fa1876a6772215604ba765ae1d6a91d67b9",
            "execution_closure": "control_input/read",
            "role": "historical_control_member",
            "sha256": "62cf9cd77863d18158afad8955b7d09c2c0f8b09046869bacb25c91c789878e0",
        },
        "same_runner": False,
    }:
        raise LauncherError("B runner role boundary differs")
    roots = manifest.get("trust_roots")
    if not isinstance(roots, dict) or roots.get("source_commit") != RUNNER_COMMIT or roots.get("source_tree") != RUNNER_TREE or roots.get("runner") != {
        "archive_path": str(RUNNER),
        "git_blob": RUNNER_GIT_BLOB,
        "source_commit": RUNNER_COMMIT,
        "source_sha256": RUNNER_SHA,
        "source_tree": RUNNER_TREE,
    }:
        raise LauncherError("B runner trust root differs")
    validator = roots.get("validator")
    if validator != {"archive_path": str(BINDING_ROOT / "trusted-validator.py"), "execution_path": str(VALIDATOR), "git_blob": VALIDATOR_GIT_BLOB, "sha256": VALIDATOR_SHA, "source_commit": VALIDATOR_COMMIT, "source_tree": VALIDATOR_TREE}:
        raise LauncherError("B validator trust root differs")
    resident = roots.get("resident_driver")
    if resident != {
        "normative_commit": RESIDENT_COMMIT,
        "source_tree": RESIDENT_TREE,
        "git_blob_at_binding_commit": RESIDENT_GIT_BLOB,
        "source_sha256": RESIDENT_SOURCE_SHA,
        "blob_unchanged": True,
        "binary_sha256": RESIDENT_SHA,
        "binary_bytes": RESIDENT_BYTES,
        "binary_build_id_sha1": RESIDENT_BUILD_ID,
        "build_inputs": RESIDENT_BUILD_INPUTS,
        "build": RESIDENT_BUILD_METADATA,
    }:
        raise LauncherError("B resident trust root differs")
    input_root = manifest.get("input_root")
    if not isinstance(input_root, dict) or input_root.get("sha256") != INPUT_FINGERPRINT_SHA or input_root.get("directory") != {"path": str(INPUT_ROOT), "device": INPUT_ROOT_DEVICE, "inode": INPUT_ROOT_INODE}:
        raise LauncherError("B input root fingerprint differs")
    members = input_root.get("members")
    if not isinstance(members, dict) or set(members) != set(INPUT_MEMBER_SHA):
        raise LauncherError("B exact19 member coverage differs")
    for name, digest in INPUT_MEMBER_SHA.items():
        record = members.get(name)
        if not isinstance(record, dict) or record.get("path") != str(INPUT_ROOT / name) or record.get("sha256") != digest or record.get("type") != "regular_file" or record.get("nlink") != 1:
            raise LauncherError(f"B input member differs: {name}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or outputs.get("runner_plan_sha256") != BINDING_PLAN_SHA:
        raise LauncherError("B runner plan binding differs")
    if manifest.get("predecessor") != {"commit": "31eb65a644eae20a3be6cbeb36b04aaaabf69429", "status": "SUPERSEDED", "execution_eligible": False}:
        raise LauncherError("B predecessor differs")
    return manifest


def validator_argv() -> list[str]:
    return [str(PYTHON), str(VALIDATOR), "validate-binding", "--binding", str(BINDING_ROOT), "--validator-source-commit", VALIDATOR_COMMIT, "--validator-sha256", VALIDATOR_SHA]


def runner_argv() -> list[str]:
    return [
        str(PYTHON), str(RUNNER),
        "--expanded", str(INPUT_ROOT / "case-binding.json"), "--fixture-index", str(INPUT_ROOT / "fixture-index.json"),
        "--identity", str(INPUT_ROOT / "identity.json"), "--preflight", str(INPUT_ROOT / "preflight.json"),
        "--policy", str(INPUT_ROOT / "policy.json"), "--bundle-root", str(INPUT_ROOT),
        "--trusted-validator", str(VALIDATOR), "--trusted-validator-sha256", VALIDATOR_SHA,
        "--output-dir", str(RUNNER_OUTPUT), "--run-id", "p2-r9700-resident-one-case-smoke-binding-v7-validate",
        "--baseline-kind", "active-production", "--lock-path", str(LOCK_PATH), "--one-case-smoke", "--dry-run",
    ]


def validate_constants(snapshot: Snapshot) -> dict[str, Any]:
    snapshot.directory(INPUT_ROOT, INPUT_ROOT_DEVICE, INPUT_ROOT_INODE, "input root")
    snapshot.directory(BINDING_ROOT, BINDING_ROOT_DEVICE, BINDING_ROOT_INODE, "B root")
    if stat.S_IMODE(BINDING_ROOT.lstat().st_mode) != BINDING_ROOT_MODE:
        raise LauncherError("B root immutable mode differs")
    if {entry.name for entry in INPUT_ROOT.iterdir()} != set(INPUT_MEMBER_SHA):
        raise LauncherError("input root exact19 coverage differs")
    for name, digest in INPUT_MEMBER_SHA.items():
        snapshot.file(INPUT_ROOT / name, digest, f"input/{name}")
    if {entry.name for entry in BINDING_ROOT.iterdir()} != set(BINDING_MEMBER_SHA):
        raise LauncherError("B root exact member coverage differs")
    binding_raw = b""
    for name, digest in BINDING_MEMBER_SHA.items():
        raw = snapshot.file(BINDING_ROOT / name, digest, f"B/{name}")
        if name == "binding-manifest.json":
            binding_raw = raw
    snapshot.file(PYTHON, PYTHON_SHA, "Python")
    snapshot.file(VALIDATOR, VALIDATOR_SHA, "validator")
    snapshot.file(RUNNER, RUNNER_SHA, "runner")
    snapshot.file(RESIDENT_DRIVER, RESIDENT_SHA, "resident driver")
    snapshot.file(SERVED_MANIFEST, SERVED_SHA, "served manifest")
    reject_symlink_components(LOCK_PATH, "device lock", allow_missing_leaf=True)
    case = parse_json(snapshot.file(INPUT_ROOT / "case-binding.json", INPUT_MEMBER_SHA["case-binding.json"], "case binding"), "case binding")
    cases = case.get("cases")
    if not isinstance(cases, list) or len(cases) != 1 or cases[0].get("case_id") != CASE_ID or cases[0].get("case_sha256") != CASE_SHA or cases[0].get("device", {}).get("runtime_device_index") != DEVICE_INDEX:
        raise LauncherError("pinned case/device differs")
    return validate_binding_manifest(binding_raw)


def validate_execute_constants(snapshot: Snapshot, self_sha: str) -> dict[str, Any]:
    manifest = validate_constants(snapshot)
    snapshot.file(Path(__file__).resolve(), self_sha, "launcher self")
    snapshot.symlink(ROCM_LINK, "/etc/alternatives/rocm", "ROCm root link")
    snapshot.symlink(ROCM_ALTERNATIVE_LINK, "/opt/rocm-7.2.1", "ROCm alternative link")
    snapshot.symlink(AMD_SMI, "../libexec/amdsmi_cli/amdsmi_cli.py", "amd-smi invocation link")
    snapshot.file(AMD_SMI_REAL, AMD_SMI_SHA, "amd-smi resolved tool")
    snapshot.symlink(ROCMINFO, "/etc/alternatives/rocminfo", "rocminfo invocation link")
    snapshot.symlink(ROCMINFO_ALTERNATIVE_LINK, str(ROCMINFO_REAL), "rocminfo alternative link")
    snapshot.file(ROCMINFO_REAL, ROCMINFO_SHA, "rocminfo resolved tool")
    snapshot.file(SYSTEMCTL, SYSTEMCTL_SHA, "systemctl")
    snapshot.file(PGREP, PGREP_SHA, "pgrep")
    snapshot.file(SUDO, SUDO_SHA, "sudo")
    return manifest


def atomic_write(directory: Path, name: str, raw: bytes, mode: int = 0o444) -> None:
    path = directory / name
    temporary = directory / f".{name}.{os.getpid()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:offset + CHUNK])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(temporary, mode)
    try:
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def process_record(argv: list[str], completed: subprocess.CompletedProcess[bytes], prefix: str, output: Path) -> dict[str, Any]:
    atomic_write(output, f"{prefix}.stdout.bin", completed.stdout)
    atomic_write(output, f"{prefix}.stderr.bin", completed.stderr)
    return {
        "argv": argv, "exit_code": completed.returncode,
        "stdout": {"file": f"{prefix}.stdout.bin", "sha256": sha_bytes(completed.stdout)},
        "stderr": {"file": f"{prefix}.stderr.bin", "sha256": sha_bytes(completed.stderr)},
    }


def validate_validator_report(raw: bytes) -> dict[str, Any]:
    report = parse_json(raw, "validator report")
    expected = {"status": "prepared_not_executed", "promotion": False, "run_id": "p2-r9700-resident-one-case-smoke-binding-v7"}
    if report != expected:
        raise LauncherError("validator report/root/B binding differs")
    return report


def validate_runner_plan(raw: bytes) -> dict[str, Any]:
    if sha_bytes(raw) != BINDING_PLAN_SHA:
        raise LauncherError("runner dry-run plan differs from B")
    plan = parse_json(raw, "runner plan")
    expected = {"case_count": 1, "transaction_count": 12, "warmup_runs": 2, "measured_runs": 10, "smoke_only": True, "promotion_eligible": False}
    if any(plan.get(key) != value for key, value in expected.items()):
        raise LauncherError("runner one-case plan facts differ")
    validation = plan.get("validation")
    if not isinstance(validation, dict) or validation.get("root_contract") != "ullm.aq4_p2_resident_smoke_bundle_root.v4" or validation.get("trusted_bundle_validator", {}).get("source") != {"path": str(VALIDATOR), "sha256": VALIDATOR_SHA}:
        raise LauncherError("runner validator/root report differs")
    return plan


def execute_binding_document() -> dict[str, Any]:
    return {
        "schema_version": "ullm.aq4_p2_resident_smoke_execute_binding.v1",
        "status": "blocked_pending_live_preflight_and_qa", "actual_eligible": False, "promotion": False,
        "dry_launcher": {"commit": DRY_L_COMMIT, "tree": DRY_L_TREE, "git_blob": DRY_L_GIT_BLOB, "sha256": DRY_L_SHA},
        "input_root": {"path": str(INPUT_ROOT), "fingerprint_sha256": INPUT_FINGERPRINT_SHA, "member_count": 19},
        "B": {"path": str(BINDING_ROOT), "manifest_sha256": BINDING_MANIFEST_SHA},
        "R": {"path": str(RUNNER), "commit": RUNNER_COMMIT, "tree": RUNNER_TREE, "git_blob": RUNNER_GIT_BLOB, "sha256": RUNNER_SHA},
        "validator": {"path": str(VALIDATOR), "commit": VALIDATOR_COMMIT, "tree": VALIDATOR_TREE, "git_blob": VALIDATOR_GIT_BLOB, "sha256": VALIDATOR_SHA},
        "resident": {"path": str(RESIDENT_DRIVER), "commit": RESIDENT_COMMIT, "sha256": RESIDENT_SHA, "served_manifest": str(SERVED_MANIFEST), "served_sha256": SERVED_SHA},
        "runtime_mapping": {"runtime_device_index": DEVICE_INDEX, "visible_token": "1", "amd_smi_index": AMD_SMI_INDEX, "bdf": GPU_BDF, "uuid": GPU_UUID, "kfd_id": KFD_ID},
        "case": {"case_id": CASE_ID, "case_sha256": CASE_SHA},
        "lock_path": str(LOCK_PATH), "run_id": EXECUTE_RUN_ID, "runner_output": str(EXECUTE_RUN_OUTPUT), "evidence_output": str(EXECUTE_EVIDENCE_OUTPUT),
        "environment": EXECUTE_ENV,
        "tools": {
            "amd_smi": {"path": str(AMD_SMI), "resolved_path": str(AMD_SMI_REAL), "sha256": AMD_SMI_SHA, "symlink_chain": [[str(ROCM_LINK), "/etc/alternatives/rocm"], [str(ROCM_ALTERNATIVE_LINK), "/opt/rocm-7.2.1"], [str(AMD_SMI), "../libexec/amdsmi_cli/amdsmi_cli.py"]]}, "rocminfo": {"path": str(ROCMINFO), "resolved_path": str(ROCMINFO_REAL), "sha256": ROCMINFO_SHA, "symlink_chain": [[str(ROCMINFO), "/etc/alternatives/rocminfo"], [str(ROCMINFO_ALTERNATIVE_LINK), str(ROCMINFO_REAL)]]},
            "systemctl": {"path": str(SYSTEMCTL), "sha256": SYSTEMCTL_SHA}, "pgrep": {"path": str(PGREP), "sha256": PGREP_SHA},
            "sudo": {"path": str(SUDO), "sha256": SUDO_SHA, "prevalidate_argv": [str(SUDO), "-n", "-v"], "keepalive_seconds": SUDO_KEEPALIVE_SECONDS},
        },
        "services": list(SERVICE_UNITS), "worker_path": str(WORKER),
        "live_preflight": {"required": True, "path": str(LIVE_PREFLIGHT_PATH), "sha256": None, "replaces_synthetic_preflight": False},
        "execution_contract": {"resident_model_loads": 1, "case_count": 1, "warmup_runs": 2, "measured_runs": 10, "sequential": True, "oom_reuse_forbidden": True},
        "blocked_reasons": ["live preflight sidecar is absent", "independent execute-launcher QA is pending"],
    }


def profile_capture_helper_bindings() -> list[dict[str, Any]]:
    bindings = []
    for role, path, expected_sha in (
        ("selection_raw_producer", PROFILE_PRODUCER_HELPER, PROFILE_PRODUCER_HELPER_SHA),
        ("candidate_selector", PROFILE_SELECTOR_HELPER, PROFILE_SELECTOR_HELPER_SHA),
        ("profile_family_classifier", PROFILE_FAMILY_HELPER, PROFILE_FAMILY_HELPER_SHA),
    ):
        observed_sha, identity = sha_file(path, f"profile capture helper {role}")
        metadata = path.lstat()
        if observed_sha != expected_sha or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise LauncherError("profile capture helper trust differs")
        bindings.append({"role": role, "path": str(path), "identity": list(identity), "sha256": expected_sha})
    return bindings


def profile_execute_binding_document() -> dict[str, Any]:
    value = execute_binding_document()
    value["run_id"] = PROFILE_RUN_ID
    value["runner_output"] = str(PROFILE_RUN_OUTPUT)
    value["evidence_output"] = str(PROFILE_EVIDENCE_OUTPUT)
    value["live_preflight"] = {
        "required": True,
        "path": str(PROFILE_LIVE_PREFLIGHT_PATH),
        "sha256": None,
        "replaces_synthetic_preflight": False,
    }
    value["execution_contract"] = {
        **value["execution_contract"],
        "profile_diagnostic": True,
        "rocprof_wrapper_required": True,
        "execution_boundary": {
            "order": ["launcher-validator", "launcher-gates", "capture-tool", "rocprofv3", "runner"],
            "validator_profiled": False,
            "gates_profiled": False,
            "runner_profiled": True,
        },
        "target_manifest": {
            "schema_version": "ullm.aq4_p3_profile_target_command.v1",
            "generated_after_live_preflight": True,
            "file_name": PROFILE_RUNNER_TARGET_MANIFEST_NAME,
            "maximum_invocations": 1,
            "capture_helpers": profile_capture_helper_bindings(),
        },
        "measurement_eligible": False,
        "promotion_eligible": False,
    }
    value["profile_diagnostic"] = {
        "schema_version": "ullm.aq4_p2_resident_profile_diagnostic.v1",
        "enabled": True,
        "measurement_eligible": False,
        "promotion_eligible": False,
        "rocprof_wrapper_required": True,
        "output": {
            "directory": str(PROFILE_CAPTURE_OUTPUT_DIRECTORY),
            "artifact": str(PROFILE_CAPTURE_ARTIFACT),
        },
        "roctx_library": {
            "invocation_path": str(ROCTX_LIBRARY),
            "resolved_path": str(ROCTX_LIBRARY_RESOLVED),
            "sha256": ROCTX_LIBRARY_SHA,
            "symbols": ["roctxRangePushA", "roctxRangePop"],
        },
        "runner_arguments": [
            "--profile-roctx-ranges",
            "--roctx-library",
            str(ROCTX_LIBRARY),
            "--roctx-library-sha256",
            ROCTX_LIBRARY_SHA,
        ],
        "range_contract": {
            "schema_version": "ullm.aq4_p2_resident_roctx_ranges.v1",
            "range_count": 12,
            "warmup_indices": [0, 1],
            "measured_indices": list(range(2, 12)),
            "same_pid_thread": True,
            "balanced": True,
        },
    }
    return value


def ready_profile_execute_binding() -> dict[str, Any]:
    value = profile_execute_binding_document()
    value["status"] = "ready_for_explicit_execute"
    value["actual_eligible"] = True
    value["blocked_reasons"] = []
    value["live_preflight"] = {
        "required": True,
        "path": str(PROFILE_LIVE_PREFLIGHT_PATH),
        "sha256": None,
        "replaces_synthetic_preflight": True,
    }
    validate_execute_binding(value, permit_test_live_preflight=True)
    return value


def validate_execute_binding(value: dict[str, Any], *, permit_test_live_preflight: bool = False) -> dict[str, Any]:
    expected = profile_execute_binding_document() if "profile_diagnostic" in value else execute_binding_document()
    live = value.get("live_preflight")
    if permit_test_live_preflight:
        clone = json.loads(json.dumps(value))
        clone["runner_output"] = expected["runner_output"]
        clone["evidence_output"] = expected["evidence_output"]
        clone["run_id"] = expected["run_id"]
        clone["live_preflight"] = expected["live_preflight"]
        clone["status"] = expected["status"]
        clone["actual_eligible"] = expected["actual_eligible"]
        clone["blocked_reasons"] = expected["blocked_reasons"]
        if "profile_diagnostic" in clone:
            clone["profile_diagnostic"]["output"] = expected["profile_diagnostic"]["output"]
        if clone != expected:
            raise LauncherError("execute binding fixed trust roots differ")
        if not isinstance(live, dict) or live.get("required") is not True or live.get("replaces_synthetic_preflight") is not True or live.get("path") != str(Path(value["evidence_output"]) / "live-preflight.json") or live.get("sha256") is not None:
            raise LauncherError("execute live preflight binding differs")
        if "profile_diagnostic" in value:
            output = value["profile_diagnostic"].get("output")
            if not isinstance(output, dict) or set(output) != {"directory", "artifact"} or not isinstance(output.get("directory"), str) or not Path(output["directory"]).is_absolute() or output.get("artifact") != str(Path(output["directory"]) / "capture-artifact.json"):
                raise LauncherError("test profile capture output binding differs")
        return value
    if value != expected:
        raise LauncherError("execute binding exact document differs")
    return value


def prepare_execute_binding(output: Path = EXECUTE_BINDING_ROOT) -> dict[str, Any]:
    if output.absolute() != EXECUTE_BINDING_ROOT:
        raise LauncherError("execute binding output must be canonical")
    ensure_directory_chain(output.parent, "execute binding parent")
    reject_symlink_components(output, "execute binding output", allow_missing_leaf=True)
    if output.exists() or output.is_symlink():
        raise LauncherError("execute binding output already exists")
    output.mkdir(parents=True, mode=0o700)
    document = execute_binding_document()
    raw = pretty(document)
    launcher_path = Path(__file__).resolve()
    launcher_raw, _ = read_regular(launcher_path, "execute launcher source")
    relative = launcher_path.relative_to(ROOT)
    source_commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(relative)],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if source_commit.returncode != 0 or source_commit.stderr or not re.fullmatch(rb"[0-9a-f]{40}\n", source_commit.stdout):
        raise LauncherError("execute launcher Git authority lookup failed")
    source_revision = source_commit.stdout.decode("ascii").strip()
    git_values = [source_revision]
    for revision in (f"{source_revision}^{{tree}}", f"{source_revision}:{relative}"):
        completed = subprocess.run(["git", "rev-parse", revision], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if completed.returncode != 0 or completed.stderr:
            raise LauncherError("execute launcher Git identity lookup failed")
        git_values.append(completed.stdout.decode("ascii").strip())
    committed = subprocess.run(["git", "show", f"{source_revision}:{relative}"], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if committed.returncode != 0 or committed.stderr or committed.stdout != launcher_raw:
        raise LauncherError("execute launcher source is not the exact committed authority blob")
    launcher_trust = {
        "schema_version": "ullm.aq4_p2_resident_execute_launcher_trust.v1", "status": "qa_pending", "actual_eligible": False,
        "path": str(launcher_path), "commit": git_values[0], "tree": git_values[1], "git_blob": git_values[2], "sha256": sha_bytes(launcher_raw),
        "execute_binding": {"path": str(EXECUTE_BINDING_PATH), "sha256": sha_bytes(raw)},
    }
    trust_raw = pretty(launcher_trust)
    atomic_write(output, "execute-binding.json", raw)
    atomic_write(output, "launcher-trust.json", trust_raw)
    sums = f"{sha_bytes(raw)}  execute-binding.json\n{sha_bytes(trust_raw)}  launcher-trust.json\n".encode("ascii")
    atomic_write(output, "SHA256SUMS", sums)
    os.chmod(output, 0o555)
    return document


def load_execute_binding(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if path != EXECUTE_BINDING_PATH:
        raise LauncherError("execute binding path differs")
    if {item.name for item in EXECUTE_BINDING_ROOT.iterdir()} != {"execute-binding.json", "launcher-trust.json", "SHA256SUMS"}:
        raise LauncherError("execute binding artifact coverage differs")
    raw, _ = read_regular(path, "execute binding")
    value = parse_json(raw, "execute binding")
    validate_execute_binding(value)
    trust_raw, _ = read_regular(EXECUTE_LAUNCHER_TRUST_PATH, "execute launcher trust")
    trust = parse_json(trust_raw, "execute launcher trust")
    expected_keys = {"schema_version", "status", "actual_eligible", "path", "commit", "tree", "git_blob", "sha256", "execute_binding"}
    if set(trust) != expected_keys or trust.get("schema_version") != "ullm.aq4_p2_resident_execute_launcher_trust.v1" or trust.get("status") != "qa_pending" or trust.get("actual_eligible") is not False or trust.get("path") != str(Path(__file__).resolve()) or not isinstance(trust.get("commit"), str) or not re.fullmatch(r"[0-9a-f]{40}", trust["commit"]) or not isinstance(trust.get("tree"), str) or not re.fullmatch(r"[0-9a-f]{40}", trust["tree"]) or not isinstance(trust.get("git_blob"), str) or not re.fullmatch(r"[0-9a-f]{40}", trust["git_blob"]) or not isinstance(trust.get("sha256"), str) or not SHA_RE.fullmatch(trust["sha256"]) or trust.get("execute_binding") != {"path": str(path), "sha256": sha_bytes(raw)}:
        raise LauncherError("execute launcher trust contract differs")
    launcher_raw, _ = read_regular(Path(__file__).resolve(), "execute launcher trusted self")
    if sha_bytes(launcher_raw) != trust["sha256"]:
        raise LauncherError("execute launcher self differs from artifact trust")
    relative = Path(__file__).resolve().relative_to(ROOT)
    authority_revisions = (f'{trust["commit"]}^{{tree}}', f'{trust["commit"]}:{relative}')
    authority_values = []
    for revision in authority_revisions:
        completed = subprocess.run(["git", "rev-parse", revision], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if completed.returncode != 0 or completed.stderr or not re.fullmatch(rb"[0-9a-f]{40}\n", completed.stdout):
            raise LauncherError("execute launcher artifact Git identity lookup failed")
        authority_values.append(completed.stdout.decode("ascii").strip())
    if authority_values != [trust["tree"], trust["git_blob"]]:
        raise LauncherError("execute launcher artifact Git identity differs")
    committed = subprocess.run(["git", "show", authority_revisions[1]], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if committed.returncode != 0 or committed.stderr or committed.stdout != launcher_raw:
        raise LauncherError("execute launcher artifact authority blob differs")
    expected_sums = f"{sha_bytes(raw)}  execute-binding.json\n{sha_bytes(trust_raw)}  launcher-trust.json\n".encode("ascii")
    sums_raw, _ = read_regular(EXECUTE_BINDING_ROOT / "SHA256SUMS", "execute binding sums")
    if sums_raw != expected_sums:
        raise LauncherError("execute binding SHA256SUMS differs")
    return value, trust


def execute_runner_argv(binding: dict[str, Any]) -> list[str]:
    driver = [str(RESIDENT_DRIVER), "--served-model-manifest", str(SERVED_MANIFEST), "--device-index", str(DEVICE_INDEX), "--build-git-commit", RESIDENT_COMMIT]
    command = [
        str(PYTHON), str(RUNNER), "--expanded", str(INPUT_ROOT / "case-binding.json"), "--fixture-index", str(INPUT_ROOT / "fixture-index.json"),
        "--identity", str(INPUT_ROOT / "identity.json"), "--preflight", str(INPUT_ROOT / "preflight.json"), "--policy", str(INPUT_ROOT / "policy.json"),
        "--bundle-root", str(INPUT_ROOT), "--trusted-validator", str(VALIDATOR), "--trusted-validator-sha256", VALIDATOR_SHA,
        "--output-dir", binding["runner_output"], "--run-id", binding["run_id"], "--baseline-kind", "active-production", "--lock-path", str(LOCK_PATH),
        "--one-case-smoke", "--live-preflight", str(Path(binding["evidence_output"]) / "live-preflight.json"),
    ]
    if "profile_diagnostic" in binding:
        command.extend(binding["profile_diagnostic"]["runner_arguments"])
    return [*command, "--driver-command", *driver]


def profile_runner_target_manifest(
    binding: dict[str, Any],
    command: list[str],
    environment: dict[str, str],
    live_preflight: dict[str, Any],
) -> dict[str, Any]:
    if "profile_diagnostic" not in binding or command != execute_runner_argv(binding):
        raise LauncherError("profile runner target argv differs")
    if environment != EXECUTE_ENV:
        raise LauncherError("profile runner target base environment differs")
    live_path = Path(live_preflight["path"])
    live_sha = live_preflight.get("sha256")
    if command[28] != str(live_path) or not isinstance(live_sha, str) or not SHA_RE.fullmatch(live_sha):
        raise LauncherError("profile runner live preflight binding differs")
    file_bindings = {
        0: (PYTHON_SHA, True, "python_interpreter", "code_execution", "exec"),
        1: (RUNNER_SHA, False, "resident_runner", "code_execution", "exec"),
        3: (INPUT_MEMBER_SHA["case-binding.json"], False, "case_binding", "control_input", "read"),
        5: (INPUT_MEMBER_SHA["fixture-index.json"], False, "fixture_index", "control_input", "read"),
        7: (INPUT_MEMBER_SHA["identity.json"], False, "identity", "control_input", "read"),
        9: (INPUT_MEMBER_SHA["preflight.json"], False, "prepared_preflight", "control_input", "read"),
        11: (INPUT_MEMBER_SHA["policy.json"], False, "policy", "control_input", "read"),
        15: (VALIDATOR_SHA, False, "trusted_validator", "code_execution", "exec"),
        28: (live_sha, False, "live_preflight", "control_input", "read"),
        35: (RESIDENT_SHA, True, "resident_driver", "code_execution", "exec"),
        37: (SERVED_SHA, False, "served_manifest", "control_input", "read"),
    }
    for index, (expected_sha, _executable, _role, _closure, _method) in file_bindings.items():
        if sha_file(Path(command[index]), f"profile runner target argv[{index}]")[0] != expected_sha:
            raise LauncherError("profile runner target input SHA differs")
    bundle_metadata = INPUT_ROOT.lstat()
    lock_metadata = LOCK_PATH.lstat()
    resolved_roctx = ROCTX_LIBRARY.resolve(strict=True)
    if resolved_roctx != ROCTX_LIBRARY_RESOLVED:
        raise LauncherError("profile runner ROCTx resolution differs")
    value: dict[str, Any] = {
        "schema_version": "ullm.aq4_p3_profile_target_command.v1",
        "status": "bound",
        "manifest_sha256": None,
        "argv": command,
        "environment": environment,
        "input_files": [
            {
                "argument_index": index,
                "path": command[index],
                "sha256": digest,
                "executable": executable,
                "role": role,
                "closure": closure,
                "method": method,
            }
            for index, (digest, executable, role, closure, method) in sorted(file_bindings.items())
        ],
        "runtime_paths": [
            {
                "argument_index": 13,
                "path": command[13],
                "kind": "directory",
                "identity": list(file_identity(bundle_metadata)),
                "role": "bundle_root",
                "closure": "data_integrity",
                "method": "pre_post_guard",
            },
            {
                "argument_index": 25,
                "path": command[25],
                "kind": "regular_file",
                "identity": list(file_identity(lock_metadata)),
                "role": "device_lock",
                "closure": "device_lock",
                "method": "flock",
            },
            {
                "argument_index": 31,
                "path": command[31],
                "kind": "symlinked_file",
                "resolved_path": str(resolved_roctx),
                "sha256": ROCTX_LIBRARY_SHA,
                "role": "roctx_library",
                "closure": "code_execution",
                "method": "dlopen",
            },
        ],
        "control_files": [
            {
                "path": str(INPUT_ROOT / name),
                "sha256": digest,
                "role": f"bundle_{name.lower().replace('-', '_').replace('.', '_')}",
                "closure": "control_input",
                "method": "read",
            }
            for name, digest in sorted(INPUT_MEMBER_SHA.items())
            if name not in {
                "case-binding.json", "fixture-index.json", "identity.json",
                "preflight.json", "policy.json", "resident-driver",
            }
        ],
        "output_paths": [{"argument_index": 19, "path": command[19]}],
        "closure_contract": {
            "code_execution_closure": "pinned_fd",
            "control_input_closure": "pinned_fd",
            "device_lock_closure": "pinned_fd",
            "data_integrity": "trusted_pre_post_guarded",
        },
        "capture_helpers": profile_capture_helper_bindings(),
        "authorization": {
            "maximum_invocations": 1,
            "target_role": "profile_runner_only",
            "promotion_eligible": False,
        },
    }
    absolute_indices = {index for index, item in enumerate(command) if Path(item).is_absolute()}
    classified = set(file_bindings) | {13, 25, 31, 19}
    if classified != absolute_indices:
        raise LauncherError("profile runner absolute argv coverage differs")
    value["manifest_sha256"] = sha_bytes(canonical(value))
    return value


def _probe(command: list[str], label: str, run: Callable[..., subprocess.CompletedProcess[bytes]]) -> tuple[subprocess.CompletedProcess[bytes], dict[str, Any]]:
    completed = run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    record = {"label": label, "argv": command, "exit_code": completed.returncode, "stdout_sha256": sha_bytes(completed.stdout), "stderr_sha256": sha_bytes(completed.stderr), "captured_unix_ns": time.time_ns()}
    return completed, record


def _interrupt_runner(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_runner_with_sudo_keepalive(
    command: list[str],
    environment: dict[str, str],
    *,
    sudo_run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    interval: float = SUDO_KEEPALIVE_SECONDS,
    on_started: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if interval <= 0:
        raise LauncherError("sudo keepalive interval must be positive")
    records: list[dict[str, Any]] = []
    failed = False
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout_file, stderr=stderr_file,
            env=dict(environment), shell=False, start_new_session=True,
        )
        if on_started is not None:
            on_started()
        next_keepalive = time.monotonic() + interval
        while process.poll() is None:
            now = time.monotonic()
            if now >= next_keepalive:
                completed, record = _probe([str(SUDO), "-n", "-v"], f"sudo-keepalive-{len(records) + 1}", sudo_run)
                records.append(record)
                if completed.returncode != 0 or completed.stdout or completed.stderr:
                    failed = True
                    _interrupt_runner(process)
                    break
                while next_keepalive <= now:
                    next_keepalive += interval
            try:
                process.wait(timeout=min(1.0, max(0.01, next_keepalive - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
        return_code = process.wait()
        stdout_file.seek(0); stdout = stdout_file.read(MAX_BYTES + 1)
        stderr_file.seek(0); stderr = stderr_file.read(MAX_BYTES + 1)
    if len(stdout) > MAX_BYTES or len(stderr) > MAX_BYTES:
        raise LauncherError("execute runner output exceeds evidence size bound")
    completed = subprocess.CompletedProcess(command, return_code, stdout, stderr)
    completed_cleanly = return_code == 0 and not stderr and not failed
    execution_state: bool | str = True if completed_cleanly else "unknown"
    return {"completed": completed, "keepalives": records, "keepalive_failed": failed, "gpu_command_executed": execution_state, "model_load_executed": execution_state}


def _service_value(raw: bytes, unit: str) -> dict[str, Any]:
    try:
        pairs_value = dict(line.split("=", 1) for line in raw.decode().splitlines())
    except (UnicodeError, ValueError) as error:
        raise LauncherError(f"service probe is invalid: {unit}") from error
    if set(pairs_value) != {"ActiveState", "SubState", "MainPID"} or pairs_value != {"ActiveState": "inactive", "SubState": "dead", "MainPID": "0"}:
        raise LauncherError(f"service is not inactive: {unit}")
    return {"unit": unit, "active_state": "inactive", "sub_state": "dead", "main_pid": 0}


def _kfd_entry_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_nlink)


def _kfd_lstat(path: Path, stage: str, pid: int, *, directory: bool) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise _KfdEntryDisappeared(stage, pid) from None
    except OSError as error:
        diagnostic = {"reason_code": "source_os_error", "stage": stage, "pid": pid, "errno": error.errno, "errno_name": errno.errorcode.get(error.errno, "UNKNOWN")}
        raise KfdOwnerScanError(diagnostic) from error
    expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode) or not expected:
        raise KfdOwnerScanError({"reason_code": "source_type_or_symlink", "stage": stage, "pid": pid, "mode": stat.S_IFMT(metadata.st_mode)})
    return metadata


def _read_kfd_gpuid(path: Path, pid: int, queue: int) -> tuple[int, dict[str, Any]]:
    before = _kfd_lstat(path, "gpuid_lstat_before", pid, directory=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise _KfdEntryDisappeared("gpuid_open", pid) from None
    except OSError as error:
        raise KfdOwnerScanError({"reason_code": "source_os_error", "stage": "gpuid_open", "pid": pid, "errno": error.errno, "errno_name": errno.errorcode.get(error.errno, "UNKNOWN")}) from error
    raw = b""
    try:
        opened = os.fstat(descriptor)
        if _kfd_entry_identity(opened) != _kfd_entry_identity(before) or not stat.S_ISREG(opened.st_mode):
            raise KfdOwnerScanError({"reason_code": "source_identity_changed", "stage": "gpuid_open", "pid": pid, "queue": queue})
        try:
            raw = os.read(descriptor, 65)
            trailing = os.read(descriptor, 1)
        except FileNotFoundError:
            if raw:
                    raise KfdOwnerScanError({"reason_code": "source_disappeared_after_partial_read", "stage": "gpuid_read", "pid": pid, "queue": queue}) from None
            raise _KfdEntryDisappeared("gpuid_read", pid) from None
        except OSError as error:
            raise KfdOwnerScanError({"reason_code": "source_os_error", "stage": "gpuid_read", "pid": pid, "queue": queue, "errno": error.errno, "errno_name": errno.errorcode.get(error.errno, "UNKNOWN")}) from error
    finally:
        os.close(descriptor)
    if trailing or len(raw) > 64:
        raise KfdOwnerScanError({"reason_code": "gpuid_source_exceeds_bound", "stage": "gpuid_read", "pid": pid, "queue": queue})
    after = _kfd_lstat(path, "gpuid_lstat_after", pid, directory=False)
    if _kfd_entry_identity(after) != _kfd_entry_identity(before):
        raise KfdOwnerScanError({"reason_code": "source_identity_changed", "stage": "gpuid_lstat_after", "pid": pid, "queue": queue})
    payload = raw[:-1] if raw.endswith(b"\n") else raw
    try:
        if re.fullmatch(rb"[0-9]+\n?", raw) is None or payload.startswith(b"0"):
            raise ValueError
        gpuid = int(payload.decode("ascii"))
    except (UnicodeError, ValueError):
        raise KfdOwnerScanError({"reason_code": "gpuid_schema_differs", "stage": "gpuid_parse", "pid": pid, "queue": queue, "raw_sha256": sha_bytes(raw), "raw_bytes": len(raw)}) from None
    if gpuid <= 0:
        raise KfdOwnerScanError({"reason_code": "gpuid_schema_differs", "stage": "gpuid_parse", "pid": pid, "queue": queue, "raw_sha256": sha_bytes(raw), "raw_bytes": len(raw)})
    return gpuid, {"pid": pid, "queue": queue, "raw_sha256": sha_bytes(raw), "raw_bytes": len(raw), "line_ending": "lf" if raw.endswith(b"\n") else "none", "parsed_gpuid": gpuid}


def _scan_kfd_owners_once(root: Path, allowed_owners: set[int] | None) -> dict[str, Any]:
    try:
        root_metadata = root.stat()
        names = sorted(os.listdir(root))
    except OSError as error:
        raise KfdOwnerScanError({"reason_code": "root_unavailable", "stage": "root_enumeration", "pid": None, "errno": error.errno, "errno_name": errno.errorcode.get(error.errno, "UNKNOWN")}) from error
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise KfdOwnerScanError({"reason_code": "root_not_directory", "stage": "root_enumeration", "pid": None})
    if any(not name.isdigit() or int(name) <= 0 for name in names):
        raise KfdOwnerScanError({"reason_code": "pid_name_schema_differs", "stage": "root_enumeration", "pid": None, "entry_count": len(names)})
    owners: set[int] = set()
    sources: list[dict[str, Any]] = []
    for name in names:
        pid = int(name)
        process = root / name
        process_before = _kfd_lstat(process, "pid_lstat_before", pid, directory=True)
        queues = process / "queues"
        queues_before = _kfd_lstat(queues, "queues_lstat_before", pid, directory=True)
        try:
            queue_names = sorted(os.listdir(queues))
        except FileNotFoundError:
            raise _KfdEntryDisappeared("queues_enumeration", pid) from None
        except OSError as error:
            raise KfdOwnerScanError({"reason_code": "source_os_error", "stage": "queues_enumeration", "pid": pid, "errno": error.errno, "errno_name": errno.errorcode.get(error.errno, "UNKNOWN")}) from error
        if any(not queue.isdigit() for queue in queue_names):
            raise KfdOwnerScanError({"reason_code": "queue_name_schema_differs", "stage": "queues_enumeration", "pid": pid, "entry_count": len(queue_names)})
        for queue in queue_names:
            queue_path = queues / queue
            _kfd_lstat(queue_path, "queue_lstat", pid, directory=True)
            gpuid, source = _read_kfd_gpuid(queue_path / "gpuid", pid, int(queue))
            sources.append(source)
            if gpuid == KFD_ID:
                if allowed_owners is not None and pid not in allowed_owners:
                    raise KfdOwnerScanError({"reason_code": "foreign_owner", "stage": "gpuid_parse", "pid": pid, "raw_sha256": source["raw_sha256"], "raw_bytes": source["raw_bytes"]})
                owners.add(pid)
        try:
            final_queue_names = sorted(os.listdir(queues))
        except FileNotFoundError:
            raise _KfdEntryDisappeared("queues_revalidation", pid) from None
        except OSError as error:
            raise KfdOwnerScanError({"reason_code": "source_os_error", "stage": "queues_revalidation", "pid": pid, "errno": error.errno, "errno_name": errno.errorcode.get(error.errno, "UNKNOWN")}) from error
        if final_queue_names != queue_names:
            raise _KfdEntryDisappeared("queue_membership_changed", pid)
        queues_after = _kfd_lstat(queues, "queues_lstat_after", pid, directory=True)
        process_after = _kfd_lstat(process, "pid_lstat_after", pid, directory=True)
        if _kfd_entry_identity(queues_after) != _kfd_entry_identity(queues_before) or _kfd_entry_identity(process_after) != _kfd_entry_identity(process_before):
            raise KfdOwnerScanError({"reason_code": "source_identity_changed", "stage": "pid_revalidation", "pid": pid})
    try:
        final_names = sorted(os.listdir(root))
        root_after = root.stat()
    except OSError as error:
        raise KfdOwnerScanError({"reason_code": "root_unavailable", "stage": "root_revalidation", "pid": None, "errno": error.errno, "errno_name": errno.errorcode.get(error.errno, "UNKNOWN")}) from error
    if final_names != names:
        raise _KfdEntryDisappeared("root_membership_changed")
    if _kfd_entry_identity(root_after) != _kfd_entry_identity(root_metadata):
        raise KfdOwnerScanError({"reason_code": "source_identity_changed", "stage": "root_revalidation", "pid": None})
    return {
        "owners": sorted(owners),
        "enumerated_pids": [int(name) for name in names],
        "sources": sources,
        "root": {"path": str(root), "device": root_metadata.st_dev, "inode": root_metadata.st_ino},
    }


def _kfd_owner_snapshot(root: Path = KFD_PROC_ROOT, *, allowed_owners: set[int] | None = None) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for attempt in range(2):
        try:
            result = _scan_kfd_owners_once(root, allowed_owners)
        except _KfdEntryDisappeared as error:
            attempts.append({"attempt": attempt, "classification": "entry_disappeared", "stage": error.stage, "pid": error.pid})
            continue
        return {
            "schema_version": "ullm.aq4_p2_kfd_owner_snapshot.v1",
            "classification": "stable_after_disappearance" if attempts else "stable",
            "attempt_count": attempt + 1,
            "attempts": attempts + [{"attempt": attempt, "classification": "stable", "enumerated_pids_sha256": sha_bytes(canonical(result["enumerated_pids"])), "enumerated_pid_count": len(result["enumerated_pids"])}],
            "owners": result["owners"],
            "source_kind": "kernel_sysfs",
            "root": result["root"],
            "sources": result["sources"],
            "secret_material_recorded": False,
        }
    raise KfdOwnerScanError({"reason_code": "entries_unstable_after_retry", "stage": attempts[-1]["stage"], "pid": attempts[-1]["pid"], "attempts": attempts})


def _kfd_owners(root: Path = KFD_PROC_ROOT) -> list[int]:
    return _kfd_owner_snapshot(root)["owners"]


def _zero_kfd_owner_snapshot() -> dict[str, Any]:
    return _kfd_owner_snapshot(allowed_owners=set())


def validate_amd_smi_tool() -> None:
    links = ((ROCM_LINK, "/etc/alternatives/rocm"), (ROCM_ALTERNATIVE_LINK, "/opt/rocm-7.2.1"), (AMD_SMI, "../libexec/amdsmi_cli/amdsmi_cli.py"))
    for path, target in links:
        metadata = path.lstat()
        if not stat.S_ISLNK(metadata.st_mode) or os.readlink(path) != target:
            raise LauncherError(f"amd-smi symlink binding differs: {path}")
    if AMD_SMI.resolve(strict=True) != AMD_SMI_REAL or sha_file(AMD_SMI_REAL, "amd-smi resolved tool")[0] != AMD_SMI_SHA:
        raise LauncherError("amd-smi resolved tool binding differs")


def validate_rocminfo_tool() -> None:
    links = ((ROCMINFO, "/etc/alternatives/rocminfo"), (ROCMINFO_ALTERNATIVE_LINK, str(ROCMINFO_REAL)))
    for path, target in links:
        metadata = path.lstat()
        if not stat.S_ISLNK(metadata.st_mode) or os.readlink(path) != target:
            raise LauncherError(f"rocminfo symlink binding differs: {path}")
    if ROCMINFO.resolve(strict=True) != ROCMINFO_REAL or sha_file(ROCMINFO_REAL, "rocminfo resolved tool")[0] != ROCMINFO_SHA:
        raise LauncherError("rocminfo resolved tool binding differs")


def _lock_gate() -> dict[str, Any]:
    reject_symlink_components(LOCK_PATH, "device lock")
    lock_metadata = LOCK_PATH.lstat()
    if not stat.S_ISREG(lock_metadata.st_mode) or lock_metadata.st_nlink != 1 or stat.S_IMODE(lock_metadata.st_mode) != 0o600:
        raise LauncherError("device lock file contract differs")
    lock_before = file_identity(lock_metadata)
    descriptor = os.open(LOCK_PATH, os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    except BlockingIOError as error:
        raise LauncherError("device lock is busy") from error
    finally:
        os.close(descriptor)
    if file_identity(LOCK_PATH.lstat()) != lock_before:
        raise LauncherError("device lock identity changed")
    return {"path": str(LOCK_PATH), "free": True, "device": lock_before[0], "inode": lock_before[1]}


def collect_execute_gates(*, run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run, environment: dict[str, str] | None = None, kfd_owner_provider: Callable[[], list[int] | dict[str, Any]] = _zero_kfd_owner_snapshot, lock_provider: Callable[[], dict[str, Any]] = _lock_gate) -> dict[str, Any]:
    environment = dict(os.environ) if environment is None else environment
    controlled = {name: environment.get(name) for name in EXECUTE_ENV}
    if controlled != EXECUTE_ENV:
        raise LauncherError("execute environment differs from exact binding")
    probes: list[dict[str, Any]] = []
    validate_amd_smi_tool()
    validate_rocminfo_tool()
    for path, digest, label in ((SYSTEMCTL, SYSTEMCTL_SHA, "systemctl"), (PGREP, PGREP_SHA, "pgrep"), (SUDO, SUDO_SHA, "sudo")):
        if sha_file(path, label)[0] != digest:
            raise LauncherError(f"{label} tool SHA differs")
    sudo, record = _probe([str(SUDO), "-n", "-v"], "sudo-n", run); probes.append(record)
    if sudo.returncode != 0 or sudo.stdout or sudo.stderr:
        raise LauncherError("sudo -n prevalidation failed")
    services = []
    for unit in SERVICE_UNITS:
        completed, record = _probe([str(SYSTEMCTL), "show", unit, "--property=ActiveState", "--property=SubState", "--property=MainPID", "--no-pager"], f"service-{unit}", run); probes.append(record)
        if completed.returncode != 0 or completed.stderr:
            raise LauncherError(f"service probe failed: {unit}")
        services.append(_service_value(completed.stdout, unit))
    worker, record = _probe([str(PGREP), "-f", "-x", f"{WORKER}.*"], "old-worker", run); probes.append(record)
    if worker.returncode != 1 or worker.stdout or worker.stderr:
        raise LauncherError("old worker PID is present")
    listed, record = _probe([str(AMD_SMI), "list", "--json"], "amd-smi-list", run); probes.append(record)
    if listed.returncode != 0 or listed.stderr:
        raise LauncherError("amd-smi list failed")
    try:
        gpu_list = json.loads(listed.stdout, object_pairs_hook=pairs)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise LauncherError("amd-smi list schema differs") from error
    if not isinstance(gpu_list, list):
        raise LauncherError("amd-smi list root differs")
    valid_items = [item for item in gpu_list if isinstance(item, dict)]
    by_bdf = [item for item in valid_items if item.get("bdf") == GPU_BDF]
    by_uuid = [item for item in valid_items if item.get("uuid") == GPU_UUID]
    by_kfd = [item for item in valid_items if item.get("kfd_id") == KFD_ID]
    by_index = [item for item in valid_items if item.get("gpu") == AMD_SMI_INDEX]
    matches = [item for item in valid_items if item.get("bdf") == GPU_BDF and item.get("uuid") == GPU_UUID and item.get("kfd_id") == KFD_ID]
    if len(by_bdf) != 1 or len(by_uuid) != 1 or len(by_kfd) != 1 or len(by_index) != 1 or len(matches) != 1 or not (by_bdf[0] is by_uuid[0] is by_kfd[0] is by_index[0]) or matches[0].get("node_id") != 2 or matches[0].get("partition_id") != 0 or set(matches[0]) != {"gpu", "bdf", "uuid", "kfd_id", "node_id", "partition_id"}:
        raise LauncherError("target GPU unique identity/mapping differs")
    info, record = _probe([str(ROCMINFO)], "rocminfo", run); probes.append(record)
    if info.returncode != 0 or info.stderr or info.stdout.count(b"Name:                    gfx1201") != 1 or b"Uuid:                    GPU-a8e9ddefa2d60f55" not in info.stdout or b"Marketing Name:          AMD Radeon Graphics" not in info.stdout:
        raise LauncherError("rocminfo target schema differs")
    processes, record = _probe([str(AMD_SMI), "process", "--gpu", str(AMD_SMI_INDEX), "--general", "--json"], "amd-smi-process", run); probes.append(record)
    if processes.returncode != 0 or processes.stderr:
        raise LauncherError("amd-smi process probe failed")
    parsed_processes = parse_amd_process_owners(processes.stdout)
    if parsed_processes["owners"]:
        raise LauncherError("target GPU compute owners are not zero")
    static, record = _probe([str(AMD_SMI), "static", "--gpu", str(AMD_SMI_INDEX), "--vram", "--json"], "amd-smi-static-vram", run); probes.append(record)
    try:
        static_value = json.loads(static.stdout, object_pairs_hook=pairs)
        gpu_data = static_value["gpu_data"]
        vram_item = gpu_data[0]
        size = vram_item["vram"]["size"]
    except (UnicodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise LauncherError("amd-smi VRAM schema differs") from error
    if static.returncode != 0 or static.stderr or len(gpu_data) != 1 or vram_item.get("gpu") != AMD_SMI_INDEX or size.get("unit") != "MB" or type(size.get("value")) is not int or size["value"] <= 0:
        raise LauncherError("amd-smi VRAM identity differs")
    total_bytes = size["value"] * 1_000_000
    kfd_value = kfd_owner_provider()
    if isinstance(kfd_value, dict):
        kfd_owners = kfd_value.get("owners")
        kfd_source = kfd_value
    else:
        kfd_owners = kfd_value
        kfd_source = None
    if not isinstance(kfd_owners, list) or any(type(pid) is not int or pid <= 0 for pid in kfd_owners):
        raise LauncherError("KFD owner provider contract differs")
    if kfd_owners:
        raise LauncherError("target KFD compute owners are not zero")
    lock = lock_provider()
    if not isinstance(lock, dict) or set(lock) != {"path", "free", "device", "inode"} or lock.get("path") != str(LOCK_PATH) or lock.get("free") is not True or type(lock.get("device")) is not int or lock["device"] < 0 or type(lock.get("inode")) is not int or lock["inode"] < 0:
        raise LauncherError("device lock gate contract differs")
    return {"passed": True, "environment": EXECUTE_ENV, "services": services, "old_worker_pids": [], "runtime_mapping": {"runtime_device_index": DEVICE_INDEX, "visible_token": "1", "amd_smi_index": AMD_SMI_INDEX, "bdf": GPU_BDF, "uuid": GPU_UUID, "kfd_id": KFD_ID, "node_id": matches[0]["node_id"]}, "amd_smi_owners": [], "amd_smi_process": parsed_processes["diagnostic"], "kfd_owners": [], "kfd_source": kfd_source, "lock": lock, "vram": {"total_bytes": total_bytes, "used_bytes": 0, "free_bytes": total_bytes, "headroom_bytes": total_bytes}, "probes": probes}


def _result_inventory(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise LauncherError("runner result directory is missing")
    files = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        digest, _ = sha_file(path, f"runner result {path.name}")
        files[path.name] = digest
    return {"path": str(root), "files": files, "tree_sha256": sha_bytes(canonical(files))}


def validate_profile_constants(snapshot: Snapshot, binding: dict[str, Any]) -> None:
    expected = profile_execute_binding_document()["profile_diagnostic"]
    observed = json.loads(json.dumps(binding.get("profile_diagnostic")))
    if binding.get("run_id") != PROFILE_RUN_ID and isinstance(observed, dict):
        observed["output"] = expected["output"]
    if observed != expected:
        raise LauncherError("profile diagnostic binding differs")
    try:
        resolved = ROCTX_LIBRARY.resolve(strict=True)
    except OSError as error:
        raise LauncherError(f"ROCTx invocation path resolution failed: {error}") from error
    if resolved != ROCTX_LIBRARY_RESOLVED:
        raise LauncherError("ROCTx invocation resolved path differs")
    snapshot.file(ROCTX_LIBRARY_RESOLVED, ROCTX_LIBRARY_SHA, "ROCTx resolved library")


def validate_profile_result(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    sidecar_path = root / "resident-batch.roctx-ranges.json"
    raw_path = root / f"{CASE_ID}.raw.json"
    sidecar_raw, _ = read_regular(sidecar_path, "ROCTx range evidence")
    raw_bytes, _ = read_regular(raw_path, "profile resident raw")
    sidecar = parse_json(sidecar_raw, "ROCTx range evidence")
    raw = parse_json(raw_bytes, "profile resident raw")
    exact = {
        "schema_version", "status", "measurement_eligible", "promotion_eligible",
        "audit_sha256", "pid", "thread_id", "library", "ranges",
    }
    if set(sidecar) != exact or sidecar.get("schema_version") != "ullm.aq4_p2_resident_roctx_ranges.v1" or sidecar.get("status") != "complete_diagnostic" or sidecar.get("measurement_eligible") is not False or sidecar.get("promotion_eligible") is not False:
        raise LauncherError("ROCTx range evidence top-level contract differs")
    audit = sidecar.get("audit_sha256")
    clone = json.loads(json.dumps(sidecar)); clone["audit_sha256"] = None
    if not isinstance(audit, str) or not SHA_RE.fullmatch(audit) or audit != sha_bytes(canonical(clone)):
        raise LauncherError("ROCTx range evidence audit SHA differs")
    if type(sidecar.get("pid")) is not int or sidecar["pid"] <= 0 or type(sidecar.get("thread_id")) is not int or sidecar["thread_id"] <= 0:
        raise LauncherError("ROCTx range PID/thread differs")
    library = sidecar.get("library")
    expected_library = binding["profile_diagnostic"]["roctx_library"]
    if not isinstance(library, dict) or set(library) != {"invocation_path", "resolved_path", "sha256", "symbols", "components"} or any(library.get(key) != expected_library[key] for key in ("invocation_path", "resolved_path", "sha256", "symbols")) or not isinstance(library.get("components"), list):
        raise LauncherError("ROCTx range library binding differs")
    try:
        session_id = raw["resident"]["session_id"]
        raw_run_id = raw["baseline_identity"]["run_id"]
    except (KeyError, TypeError) as error:
        raise LauncherError("profile resident raw identity differs") from error
    if not isinstance(session_id, str) or not session_id or "/" in session_id or "=" in session_id or raw_run_id != binding["run_id"] or raw.get("case_id") != CASE_ID or raw.get("case_sha256") != CASE_SHA or raw.get("promotion_eligible") is not False or raw.get("execution_mode") != "one_case_smoke":
        raise LauncherError("profile resident raw run/session/case binding differs")
    ranges = sidecar.get("ranges")
    if not isinstance(ranges, list) or len(ranges) != 12:
        raise LauncherError("ROCTx range count differs")
    for index, item in enumerate(ranges):
        kind = "warmup" if index < 2 else "measured"
        expected_name = (
            f"ullm.aq4_p2.run.v1/run_id={binding['run_id']}/session_id={session_id}/"
            f"case_id={CASE_ID}/case_sha256={CASE_SHA}/run_index={index}/run_kind={kind}"
        )
        if not isinstance(item, dict) or set(item) != {"name", "run_index", "run_kind", "push_result", "pop_result"} or item.get("name") != expected_name or item.get("run_index") != index or item.get("run_kind") != kind or type(item.get("push_result")) is not int or item["push_result"] < 0 or type(item.get("pop_result")) is not int or item["pop_result"] < 0:
            raise LauncherError("ROCTx range order/name/balance differs")
    return {
        "mode": "profile_diagnostic",
        "measurement_eligible": False,
        "promotion_eligible": False,
        "run_id": binding["run_id"],
        "resident_session_id": session_id,
        "case_id": CASE_ID,
        "case_sha256": CASE_SHA,
        "ranges": {"path": str(sidecar_path), "sha256": sha_bytes(sidecar_raw), "audit_sha256": audit, "count": 12},
        "resident_raw": {"path": str(raw_path), "sha256": sha_bytes(raw_bytes)},
        "library": expected_library,
    }


def expected_live_probe_contracts() -> dict[str, tuple[list[str], int]]:
    return {
        "sudo-n": ([str(SUDO), "-n", "-v"], 0),
        **{f"service-{unit}": ([str(SYSTEMCTL), "show", unit, "--property=ActiveState", "--property=SubState", "--property=MainPID", "--no-pager"], 0) for unit in SERVICE_UNITS},
        "old-worker": ([str(PGREP), "-f", "-x", f"{WORKER}.*"], 1),
        "amd-smi-list": ([str(AMD_SMI), "list", "--json"], 0),
        "rocminfo": ([str(ROCMINFO)], 0),
        "amd-smi-process": ([str(AMD_SMI), "process", "--gpu", str(AMD_SMI_INDEX), "--general", "--json"], 0),
        "amd-smi-static-vram": ([str(AMD_SMI), "static", "--gpu", str(AMD_SMI_INDEX), "--vram", "--json"], 0),
    }


def validate_generated_live_preflight_content(value: dict[str, Any], binding: dict[str, Any]) -> None:
    exact = {"schema_version", "status", "run_id", "captured_unix_ns", "prepared_preflight", "runtime_mapping", "services", "worker_pids", "compute_owners", "lock", "environment", "vram", "commands"}
    if set(value) != exact or value.get("schema_version") != "ullm.aq4_p2_resident_live_preflight.v1" or value.get("status") != "passed" or value.get("run_id") != binding.get("run_id") or type(value.get("captured_unix_ns")) is not int or value["captured_unix_ns"] <= 0:
        raise LauncherError("generated live preflight top-level contract differs")
    if value.get("prepared_preflight") != {"path": str(INPUT_ROOT / "preflight.json"), "sha256": INPUT_MEMBER_SHA["preflight.json"], "role": "synthetic_bundle_contract_only"}:
        raise LauncherError("generated live preflight prepared member differs")
    expected_mapping = {"runtime_device_index": 1, "visible_token": "1", "amd_smi_index": 2, "bdf": GPU_BDF, "uuid": GPU_UUID, "kfd_id": KFD_ID, "node_id": 2}
    if value.get("runtime_mapping") != expected_mapping:
        raise LauncherError("generated live preflight mapping differs")
    expected_services = [{"unit": unit, "active_state": "inactive", "sub_state": "dead", "main_pid": 0} for unit in SERVICE_UNITS]
    if value.get("services") != expected_services or value.get("worker_pids") != [] or value.get("compute_owners") != {"amd_smi": [], "kfd": []}:
        raise LauncherError("generated live preflight owner/service state differs")
    lock = value.get("lock")
    if not isinstance(lock, dict) or set(lock) != {"path", "free", "device", "inode"} or lock.get("path") != str(LOCK_PATH) or lock.get("free") is not True or type(lock.get("device")) is not int or lock["device"] < 0 or type(lock.get("inode")) is not int or lock["inode"] < 0:
        raise LauncherError("generated live preflight lock differs")
    if value.get("environment") != EXECUTE_ENV:
        raise LauncherError("generated live preflight environment differs")
    vram = value.get("vram")
    if not isinstance(vram, dict) or set(vram) != {"total_bytes", "used_bytes", "free_bytes", "headroom_bytes"} or any(type(vram.get(name)) is not int for name in vram) or vram["total_bytes"] < 30_000_000_000 or vram["used_bytes"] != 0 or vram["free_bytes"] != vram["total_bytes"] or vram["headroom_bytes"] != vram["total_bytes"]:
        raise LauncherError("generated live preflight VRAM differs")
    commands = value.get("commands")
    expected = expected_live_probe_contracts()
    if not isinstance(commands, list) or len(commands) != len(expected) or any(not isinstance(item, dict) for item in commands):
        raise LauncherError("generated live preflight probe contract differs")
    observed: set[str] = set()
    for item in commands:
        label = item.get("label")
        if set(item) != {"label", "argv", "exit_code", "stdout_sha256", "stderr_sha256", "captured_unix_ns"} or label not in expected or label in observed:
            raise LauncherError("generated live preflight probe contract differs")
        argv, exit_code = expected[label]
        if item.get("argv") != argv or item.get("exit_code") != exit_code or not isinstance(item.get("stdout_sha256"), str) or not SHA_RE.fullmatch(item["stdout_sha256"]) or not isinstance(item.get("stderr_sha256"), str) or not SHA_RE.fullmatch(item["stderr_sha256"]) or type(item.get("captured_unix_ns")) is not int or item["captured_unix_ns"] < 0:
            raise LauncherError("generated live preflight probe contract differs")
        observed.add(label)
    if observed != set(expected):
        raise LauncherError("generated live preflight probe contract differs")


def make_live_preflight(binding: dict[str, Any], gates: dict[str, Any], evidence_output: Path) -> dict[str, Any]:
    value = {
        "schema_version": "ullm.aq4_p2_resident_live_preflight.v1", "status": "passed", "run_id": binding["run_id"], "captured_unix_ns": time.time_ns(),
        "prepared_preflight": {"path": str(INPUT_ROOT / "preflight.json"), "sha256": INPUT_MEMBER_SHA["preflight.json"], "role": "synthetic_bundle_contract_only"},
        "runtime_mapping": gates["runtime_mapping"], "services": gates["services"], "worker_pids": gates["old_worker_pids"],
        "compute_owners": {"amd_smi": gates["amd_smi_owners"], "kfd": gates["kfd_owners"]}, "lock": gates["lock"],
        "environment": gates["environment"], "vram": gates["vram"], "commands": gates["probes"],
    }
    validate_generated_live_preflight_content(value, binding)
    raw = pretty(value)
    atomic_write(evidence_output, "live-preflight.json", raw)
    path = evidence_output / "live-preflight.json"
    observed, identity = read_regular(path, "generated live preflight")
    if observed != raw or stat.S_IMODE(path.lstat().st_mode) != 0o444:
        raise LauncherError("generated live preflight differs")
    return {"path": str(path), "sha256": sha_bytes(raw), "identity": identity, "content": value}


def verify_generated_live_preflight(link: dict[str, Any]) -> None:
    path = Path(link["path"])
    raw, identity = read_regular(path, "generated live preflight final")
    if identity != tuple(link["identity"]) or sha_bytes(raw) != link["sha256"] or stat.S_IMODE(path.lstat().st_mode) != 0o444:
        raise LauncherError("generated live preflight changed after creation")


def execute_bound(
    binding: dict[str, Any], evidence_output: Path, runner_output: Path, run_id: str, *,
    trusted_launcher_sha: str,
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    gate_provider: Callable[[], dict[str, Any]] = collect_execute_gates,
    restore_provider: Callable[[], dict[str, Any]] | None = None,
    runner_executor: Callable[[list[str], dict[str, str], Callable[[], None]], dict[str, Any]] | None = None,
    profile_runner_executor: Callable[[list[str], dict[str, str], Callable[[], None], dict[str, Any]], dict[str, Any]] | None = None,
    verification_hook: Callable[[str], None] | None = None,
) -> tuple[int, dict[str, Any]]:
    validate_execute_binding(binding, permit_test_live_preflight=True)
    if binding.get("actual_eligible") is not True or binding.get("status") != "ready_for_explicit_execute":
        raise LauncherError("execute binding is not actual eligible")
    if not isinstance(trusted_launcher_sha, str) or not SHA_RE.fullmatch(trusted_launcher_sha):
        raise LauncherError("trusted execute launcher SHA differs")
    observed_self_sha = sha_file(Path(__file__).resolve(), "launcher self before execute")[0]
    if observed_self_sha != trusted_launcher_sha:
        raise LauncherError("execute launcher self differs from trusted artifact")
    if binding["runner_output"] != str(runner_output) or binding["evidence_output"] != str(evidence_output) or binding["run_id"] != run_id:
        raise LauncherError("execute output/run-id differs from binding")
    profile_enabled = "profile_diagnostic" in binding
    if profile_runner_executor is not None and runner_executor is not None:
        raise LauncherError("profile and generic runner executors are mutually exclusive")
    if not profile_enabled and profile_runner_executor is not None:
        raise LauncherError("profile runner executor is forbidden for normal execute")
    if profile_enabled and runner_executor is not None:
        raise LauncherError("generic runner executor is forbidden for profile execute")
    for path, label in ((evidence_output, "execute evidence"), (runner_output, "execute runner output")):
        reject_symlink_components(path, label, allow_missing_leaf=True)
        if path.exists() or path.is_symlink():
            raise LauncherError(f"{label} already exists")
    evidence_output.mkdir(mode=0o700)
    self_sha = observed_self_sha
    evidence = make_evidence("execute", self_sha)
    evidence.update({"execute_binding": binding, "profile_diagnostic": None, "profile_runner_target": None, "profile_capture": None, "profile_diagnostics": None, "gates": None, "gate_failure_diagnostic": None, "restore": None, "trust_verifications": []})
    evidence["safety"]["execution_state_source"] = "runner_not_started"
    snapshot = Snapshot()
    snapshot_ready = False
    runner_after_verified = False
    live_preflight: dict[str, Any] | None = None
    profile_target: dict[str, Any] | None = None
    profile_rocprof_started = False
    stage = "constants"

    def verify_trust(point: str) -> None:
        if verification_hook is not None:
            verification_hook(point)
        snapshot.verify()
        evidence["trust_verifications"].append(point)

    def mark_runner_started() -> None:
        if evidence["process_counts"]["runner"] != 0:
            raise LauncherError("execute runner start was reported more than once")
        evidence["process_counts"]["runner"] = 1
        evidence["sequence"].append("runner")
        evidence["safety"]["gpu_command_executed"] = "unknown"
        evidence["safety"]["model_load_executed"] = "unknown"
        evidence["safety"]["execution_state_source"] = "runner_started_completion_unknown"

    def mark_rocprof_started() -> None:
        nonlocal profile_rocprof_started
        if profile_rocprof_started:
            raise LauncherError("profile rocprof start was reported more than once")
        profile_rocprof_started = True

    def profile_lifecycle() -> dict[str, Any]:
        capture = evidence.get("profile_capture")
        if isinstance(capture, dict):
            return {
                key: capture[key]
                for key in (
                    "rocprof_started", "runner_start_known", "runner_started",
                    "runner_completed", "cleanup_passed", "children_state_known",
                    "children_remaining",
                )
            }
        return {
            "rocprof_started": profile_rocprof_started,
            "runner_start_known": not profile_rocprof_started,
            "runner_started": False,
            "runner_completed": False,
            "cleanup_passed": not profile_rocprof_started,
            "children_state_known": not profile_rocprof_started,
            "children_remaining": [],
        }

    def failure_record(failure_stage: str, reason: str, runner_started: bool) -> dict[str, Any]:
        value: dict[str, Any] = {"stage": failure_stage, "reason": reason, "runner_started": runner_started}
        if profile_enabled:
            value.update(profile_lifecycle())
        return value

    try:
        validate_execute_constants(snapshot, self_sha)
        if profile_enabled:
            validate_profile_constants(snapshot, binding)
        snapshot_ready = True
        verify_trust("validator-before")
        stage = "validator"
        validator_command = validator_argv()
        validated = run(validator_command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        evidence["process_counts"]["launcher_validator"] = 1; evidence["sequence"].append("validator")
        evidence["validator"] = process_record(validator_command, validated, "validator", evidence_output)
        if validated.returncode != 0 or validated.stderr:
            raise LauncherError("trusted validator subprocess rejected root/B")
        evidence["validator"]["report"] = validate_validator_report(validated.stdout)
        stage = "gates"
        evidence["gates"] = gate_provider()
        evidence["sequence"].append("pre-exec-gates")
        live_preflight = make_live_preflight(binding, evidence["gates"], evidence_output)
        evidence["live_preflight"] = {"path": live_preflight["path"], "sha256": live_preflight["sha256"], "identity": list(live_preflight["identity"])}
        if profile_enabled:
            stage = "profile-runner-target"
            target_value = profile_runner_target_manifest(
                binding, execute_runner_argv(binding), EXECUTE_ENV, live_preflight
            )
            target_raw = pretty(target_value)
            atomic_write(evidence_output, PROFILE_RUNNER_TARGET_MANIFEST_NAME, target_raw)
            target_path = evidence_output / PROFILE_RUNNER_TARGET_MANIFEST_NAME
            target_observed, target_identity = read_regular(target_path, "profile runner target manifest")
            if target_observed != target_raw or stat.S_IMODE(target_path.lstat().st_mode) != 0o444:
                raise LauncherError("profile runner target manifest creation differs")
            profile_target = {
                "path": str(target_path),
                "sha256": sha_bytes(target_raw),
                "manifest_sha256": target_value["manifest_sha256"],
                "identity": list(target_identity),
            }
            evidence["profile_runner_target"] = profile_target
        verify_trust("runner-before")
        stage = "runner"
        command = execute_runner_argv(binding)
        try:
            if profile_enabled and profile_runner_executor is not None:
                assert profile_target is not None
                outcome = profile_runner_executor(command, EXECUTE_ENV, mark_rocprof_started, profile_target)
            elif profile_enabled:
                raise LauncherError("profile runner executor is required")
            elif runner_executor is not None:
                outcome = runner_executor(command, EXECUTE_ENV, mark_runner_started)
            else:
                outcome = run_runner_with_sudo_keepalive(command, EXECUTE_ENV, sudo_run=run, on_started=mark_runner_started)
        except Exception:
            if evidence["process_counts"]["runner"] == 1:
                verify_trust("runner-after")
                runner_after_verified = True
            raise
        if not profile_enabled and evidence["process_counts"]["runner"] != 1:
            evidence["safety"]["gpu_command_executed"] = "unknown"
            evidence["safety"]["model_load_executed"] = "unknown"
            evidence["safety"]["execution_state_source"] = "runner_outcome_without_start_signal"
            raise LauncherError("execute runner outcome omitted start signal")
        gpu_state = outcome.get("gpu_command_executed") if isinstance(outcome, dict) else None
        load_state = outcome.get("model_load_executed") if isinstance(outcome, dict) else None
        valid_gpu_state = type(gpu_state) is bool or gpu_state == "unknown"
        valid_load_state = type(load_state) is bool or load_state == "unknown"
        expected_outcome_fields = {"completed", "keepalives", "keepalive_failed", "gpu_command_executed", "model_load_executed"}
        if profile_enabled and profile_runner_executor is not None:
            expected_outcome_fields.update({"profile_capture", "profile_diagnostics"})
        if not isinstance(outcome, dict) or set(outcome) != expected_outcome_fields or not isinstance(outcome.get("completed"), subprocess.CompletedProcess) or not isinstance(outcome.get("keepalives"), list) or type(outcome.get("keepalive_failed")) is not bool or not valid_gpu_state or not valid_load_state:
            raise LauncherError("execute runner outcome contract differs")
        if "profile_capture" in expected_outcome_fields:
            capture = outcome.get("profile_capture")
            capture_fields = {
                "status", "runner_profiled", "validator_profiled", "gates_profiled",
                "capture_tool_invocations", "rocprof_invocations", "rocprof_started",
                "runner_started", "target_manifest_sha256", "target_manifest_semantic_sha256",
                "target_argv_sha256", "environment_sha256", "capture_stdout_sha256",
                "capture_stderr_sha256", "timed_out", "cleanup_passed", "children_remaining",
                "children_state_known", "runner_start_known", "runner_completed",
            }
            if (
                not isinstance(capture, dict)
                or set(capture) != capture_fields
                or type(capture.get("rocprof_started")) is not bool
                or type(capture.get("runner_started")) is not bool
                or type(capture.get("runner_start_known")) is not bool
                or type(capture.get("runner_completed")) is not bool
                or capture.get("runner_profiled") is not True
                or capture.get("validator_profiled") is not False
                or capture.get("gates_profiled") is not False
                or capture.get("capture_tool_invocations") != 1
                or capture.get("rocprof_invocations") != int(capture.get("rocprof_started"))
                or capture.get("rocprof_started") is not profile_rocprof_started
                or capture.get("target_manifest_sha256") != profile_target["sha256"]
                or capture.get("target_manifest_semantic_sha256") != profile_target["manifest_sha256"]
                or capture.get("target_argv_sha256") != sha_bytes(canonical(command))
                or capture.get("environment_sha256") != sha_bytes(canonical(EXECUTE_ENV))
                or any(not isinstance(capture.get(key), str) or SHA_RE.fullmatch(capture[key]) is None for key in ("capture_stdout_sha256", "capture_stderr_sha256"))
                or type(capture.get("timed_out")) is not bool
                or type(capture.get("cleanup_passed")) is not bool
                or type(capture.get("children_state_known")) is not bool
                or not isinstance(capture.get("children_remaining"), list)
                or any(type(pid) is not int or pid <= 0 for pid in capture["children_remaining"])
                or len(capture["children_remaining"]) != len(set(capture["children_remaining"]))
                or capture["children_remaining"] != sorted(capture["children_remaining"])
                or capture["cleanup_passed"] is not (capture["children_state_known"] and capture["children_remaining"] == [])
                or (not capture["children_state_known"] and capture["children_remaining"] != [])
                or (capture["runner_started"] and (not capture["runner_start_known"] or not capture["rocprof_started"]))
                or (capture["runner_completed"] and (not capture["runner_start_known"] or not capture["runner_started"]))
                or (not capture["runner_start_known"] and (
                    not capture["rocprof_started"]
                    or capture["runner_started"]
                    or capture["runner_completed"]
                    or capture["status"] != "failed"
                ))
                or (not capture["rocprof_started"] and (
                    not capture["runner_start_known"]
                    or capture["runner_started"]
                    or capture["runner_completed"]
                ))
                or (capture["timed_out"] and not capture["rocprof_started"])
                or capture.get("status") not in {"complete_diagnostic", "failed"}
                or (capture["status"] == "complete_diagnostic" and (
                    not capture["rocprof_started"]
                    or not capture["runner_started"]
                    or not capture["runner_start_known"]
                    or not capture["runner_completed"]
                    or capture["timed_out"]
                    or not capture["cleanup_passed"]
                ))
            ):
                raise LauncherError("profile capture outcome contract differs")
            evidence["profile_capture"] = capture
            diagnostics = outcome.get("profile_diagnostics")
            diagnostic_fields = {
                "schema_version", "runner_finished", "capture_artifact", "failure_evidence",
                "validation_error", "executor_exception",
            }
            if not isinstance(diagnostics, dict) or set(diagnostics) != diagnostic_fields or diagnostics.get("schema_version") != "ullm.aq4_p3_profile_executor_diagnostics.v1" or type(diagnostics.get("runner_finished")) is not bool:
                raise LauncherError("profile capture diagnostics contract differs")
            for key in ("validation_error", "executor_exception"):
                if diagnostics[key] is not None and (not isinstance(diagnostics[key], str) or not diagnostics[key]):
                    raise LauncherError("profile capture diagnostics error differs")
            for key, fields in (
                ("capture_artifact", {"path", "sha256", "mode"}),
                ("failure_evidence", {"path", "sha256", "mode", "reason"}),
            ):
                item = diagnostics[key]
                expected_path = (
                    binding["profile_diagnostic"]["output"]["artifact"]
                    if key == "capture_artifact"
                    else str(Path(binding["profile_diagnostic"]["output"]["directory"]) / "capture-failure.json")
                )
                if item is not None and (
                    not isinstance(item, dict)
                    or set(item) != fields
                    or not isinstance(item.get("path"), str)
                    or item["path"] != expected_path
                    or not isinstance(item.get("sha256"), str)
                    or SHA_RE.fullmatch(item["sha256"]) is None
                    or item.get("mode") != 0o444
                    or (key == "failure_evidence" and (not isinstance(item.get("reason"), str) or not item["reason"]))
                ):
                    raise LauncherError("profile capture diagnostics evidence differs")
                if item is not None:
                    diagnostic_raw, _ = read_regular(Path(item["path"]), f"profile diagnostics {key}")
                    if sha_bytes(diagnostic_raw) != item["sha256"] or stat.S_IMODE(Path(item["path"]).lstat().st_mode) != 0o444:
                        raise LauncherError("profile capture diagnostics file binding differs")
            if diagnostics["runner_finished"] is not capture["runner_completed"] or (capture["status"] == "complete_diagnostic") is not (diagnostics["capture_artifact"] is not None) or (capture["status"] == "complete_diagnostic" and (diagnostics["failure_evidence"] is not None or diagnostics["validation_error"] is not None or diagnostics["executor_exception"] is not None)):
                raise LauncherError("profile capture diagnostics state differs")
            if capture["status"] == "failed" and diagnostics["failure_evidence"] is None and diagnostics["validation_error"] is None and diagnostics["executor_exception"] is None:
                raise LauncherError("profile capture diagnostics failure cause differs")
            evidence["profile_diagnostics"] = diagnostics
            if capture["runner_started"]:
                mark_runner_started()
        if profile_enabled and evidence["process_counts"]["runner"] != int(evidence["profile_capture"]["runner_started"]):
            raise LauncherError("profile runner lifecycle accounting differs")
        completed = outcome["completed"]
        keepalives = outcome["keepalives"]
        keepalive_failed = outcome["keepalive_failed"]
        evidence["safety"]["gpu_command_executed"] = outcome["gpu_command_executed"]
        evidence["safety"]["model_load_executed"] = outcome["model_load_executed"]
        evidence["safety"]["execution_state_source"] = "runner_executor_outcome"
        verify_trust("runner-after")
        runner_after_verified = True
        evidence["sudo_keepalive"] = {"interval_seconds": SUDO_KEEPALIVE_SECONDS, "records": keepalives, "failed": keepalive_failed}
        evidence["runner"] = process_record(command, completed, "runner", evidence_output)
        verify_generated_live_preflight(live_preflight)
        if keepalive_failed:
            raise LauncherError("sudo credential keepalive failed; execute runner was interrupted")
        if completed.returncode != 0 or completed.stderr:
            raise LauncherError("execute runner subprocess failed")
        if profile_enabled and profile_runner_executor is not None and evidence["profile_capture"]["status"] != "complete_diagnostic":
            raise LauncherError("profile capture did not complete")
        if outcome["gpu_command_executed"] is not True or outcome["model_load_executed"] is not True:
            raise LauncherError("successful execute runner did not prove GPU command and model load")
        evidence["result"] = _result_inventory(runner_output)
        roctx_path = runner_output / "resident-batch.roctx-ranges.json"
        if profile_enabled:
            evidence["profile_diagnostic"] = validate_profile_result(runner_output, binding)
        elif roctx_path.exists() or roctx_path.is_symlink():
            raise LauncherError("non-profile one-case emitted unexpected ROCTx evidence")
        evidence["process_counts"]["runner_internal_validator"] = 1
        evidence["status"] = "passed"
        code = 0
    except (LauncherError, OSError, ValueError, subprocess.SubprocessError) as error:
        if isinstance(error, (AmdProcessSchemaError, KfdOwnerScanError)):
            evidence["gate_failure_diagnostic"] = error.diagnostic
        evidence["failure"] = failure_record(stage, str(error), evidence["process_counts"]["runner"] == 1)
        if runner_output.exists() and not runner_output.is_symlink():
            try:
                evidence["result"] = _result_inventory(runner_output) | {"partial": True}
            except (LauncherError, OSError):
                pass
        code = 1
    finally:
        if evidence["process_counts"]["runner"] == 1 and not runner_after_verified and snapshot_ready:
            try:
                verify_trust("runner-after")
                runner_after_verified = True
            except (LauncherError, OSError, ValueError) as error:
                evidence["failure"] = failure_record("runner-after-verification", str(error), True)
                evidence["status"] = "failed"; code = 1
        try:
            evidence["restore"] = {"required": False, "service_stop_performed": False, "state_preserved": True} if restore_provider is None else restore_provider()
        except Exception as error:
            evidence["restore"] = {"state_preserved": False, "error": str(error)}
            evidence["status"] = "failed"; code = 1
        try:
            if live_preflight is not None:
                verify_generated_live_preflight(live_preflight)
            if profile_target is not None:
                target_path = Path(profile_target["path"])
                target_raw, target_identity = read_regular(target_path, "profile runner target manifest final")
                if (
                    target_identity != tuple(profile_target["identity"])
                    or sha_bytes(target_raw) != profile_target["sha256"]
                    or stat.S_IMODE(target_path.lstat().st_mode) != 0o444
                ):
                    raise LauncherError("profile runner target manifest changed after creation")
            if snapshot_ready:
                verify_trust("finalize-before")
        except (LauncherError, OSError, ValueError) as error:
            evidence["failure"] = failure_record("finalize-verification", str(error), evidence["process_counts"]["runner"] == 1)
            evidence["status"] = "failed"; code = 1
        finalize_output(evidence_output, evidence)
    return code, evidence


def make_evidence(mode: str, self_sha: str) -> dict[str, Any]:
    return {
        "schema_version": "ullm.aq4_p2_resident_smoke_immutable_launcher.v1", "status": "failed", "mode": mode,
        "promotion": False, "self": {"path": str(Path(__file__).resolve()), "sha256": self_sha},
        "constants": {
            "input_root": {"path": str(INPUT_ROOT), "fingerprint_sha256": INPUT_FINGERPRINT_SHA, "member_count": 19},
            "B": {"path": str(BINDING_ROOT), "manifest_sha256": BINDING_MANIFEST_SHA},
            "R": {"path": str(RUNNER), "commit": RUNNER_COMMIT, "tree": RUNNER_TREE, "git_blob": RUNNER_GIT_BLOB, "sha256": RUNNER_SHA},
            "validator": {"path": str(VALIDATOR), "commit": VALIDATOR_COMMIT, "tree": VALIDATOR_TREE, "git_blob": VALIDATOR_GIT_BLOB, "sha256": VALIDATOR_SHA},
            "resident": {"path": str(RESIDENT_DRIVER), "commit": RESIDENT_COMMIT, "sha256": RESIDENT_SHA, "served_manifest": str(SERVED_MANIFEST), "served_sha256": SERVED_SHA, "device_index": DEVICE_INDEX, "lock_path": str(LOCK_PATH)},
            "case": {"case_id": CASE_ID, "case_sha256": CASE_SHA},
        },
        "sequence": [], "process_counts": {"launcher_validator": 0, "runner": 0, "runner_internal_validator": 0, "fake_driver": 0},
        "validator": None, "runner": None, "result": None, "failure": None,
        "safety": {"gpu_command_executed": False, "model_load_executed": False, "service_touched": False, "service_stopped": False},
    }


def finalize_output(output: Path, evidence: dict[str, Any]) -> None:
    atomic_write(output, "launcher-evidence.json", pretty(evidence))
    names = sorted(entry.name for entry in output.iterdir() if entry.name != "SHA256SUMS")
    lines = []
    for name in names:
        digest, _ = sha_file(output / name, f"launcher output {name}")
        lines.append(f"{digest}  {name}\n")
    atomic_write(output, "SHA256SUMS", "".join(lines).encode("ascii"))
    os.chmod(output, 0o555)


def launch(mode: str, output: Path, *, run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run) -> tuple[int, dict[str, Any]]:
    reject_symlink_components(output, "launcher evidence output", allow_missing_leaf=True)
    if output.exists() or output.is_symlink():
        raise LauncherError(f"launcher evidence output already exists: {output}")
    output.mkdir(mode=0o700, parents=False)
    self_sha, _ = sha_file(Path(__file__).resolve(), "launcher self")
    evidence = make_evidence(mode, self_sha)
    snapshot = Snapshot()
    stage = "constants"
    runner_started = False
    try:
        if mode != "dry-run":
            raise LauncherError("actual execution is disabled; only dry-run is authorized")
        validate_constants(snapshot)
        snapshot.file(Path(__file__).resolve(), self_sha, "launcher self")
        stage = "validator"
        validator_command = validator_argv()
        validator_completed = run(validator_command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        evidence["process_counts"]["launcher_validator"] = 1
        evidence["sequence"].append("validator")
        evidence["validator"] = process_record(validator_command, validator_completed, "validator", output)
        if validator_completed.returncode != 0 or validator_completed.stderr:
            raise LauncherError("trusted validator subprocess rejected root/B")
        report = validate_validator_report(validator_completed.stdout)
        evidence["validator"]["report"] = report
        evidence["validator"]["report_sha256"] = sha_bytes(canonical(report))
        if _AFTER_VALIDATOR_HOOK is not None:
            _AFTER_VALIDATOR_HOOK()
        snapshot.verify()
        stage = "runner"
        if RUNNER_OUTPUT.exists() or RUNNER_OUTPUT.is_symlink():
            raise LauncherError(f"runner output already exists: {RUNNER_OUTPUT}")
        runner_command = runner_argv()
        runner_started = True
        runner_completed = run(runner_command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        evidence["process_counts"]["runner"] = 1
        evidence["sequence"].append("runner")
        evidence["runner"] = process_record(runner_command, runner_completed, "runner", output)
        if runner_completed.returncode != 0 or runner_completed.stderr:
            raise LauncherError("trusted runner subprocess failed")
        plan_path = RUNNER_OUTPUT / "resident-batch.plan.json"
        plan_raw, _ = read_regular(plan_path, "runner result plan")
        validate_runner_plan(plan_raw)
        atomic_write(output, "runner-plan.json", plan_raw)
        evidence["runner"]["plan"] = {"file": "runner-plan.json", "sha256": sha_bytes(plan_raw)}
        evidence["result"] = {"kind": "dry_run_plan", "sha256": sha_bytes(plan_raw), "B_plan_match": True}
        evidence["process_counts"]["runner_internal_validator"] = 1
        evidence["process_counts"]["fake_driver"] = 1
        if _BEFORE_FINAL_VERIFY_HOOK is not None:
            _BEFORE_FINAL_VERIFY_HOOK()
        snapshot.verify()
        evidence["status"] = "passed"
        return_code = 0
    except (LauncherError, OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as error:
        evidence["failure"] = {"stage": stage, "reason": str(error), "runner_started": runner_started}
        return_code = 1
    finally:
        if RUNNER_OUTPUT.exists() and not RUNNER_OUTPUT.is_symlink():
            shutil.rmtree(RUNNER_OUTPUT)
        finalize_output(output, evidence)
    return return_code, evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "execute", "profile-execute"), default="dry-run")
    parser.add_argument("--evidence-output", type=Path)
    parser.add_argument("--execute-binding", type=Path)
    parser.add_argument("--runner-output", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--trusted-launcher-sha")
    parser.add_argument("--prepare-execute-binding", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.prepare_execute_binding:
            value = prepare_execute_binding()
            print(json.dumps({"status": value["status"], "actual_eligible": value["actual_eligible"], "binding": str(EXECUTE_BINDING_PATH)}, sort_keys=True))
            return 0
        if args.evidence_output is None:
            raise LauncherError("--evidence-output is required")
        if args.mode == "profile-execute":
            if args.execute_binding is not None or args.runner_output != PROFILE_RUN_OUTPUT or args.evidence_output != PROFILE_EVIDENCE_OUTPUT or args.run_id != PROFILE_RUN_ID or args.trusted_launcher_sha is None:
                raise LauncherError("profile execute requires exact canonical output/run-id and no execute-binding")
            binding = ready_profile_execute_binding()
            code, evidence = execute_bound(binding, args.evidence_output, args.runner_output, args.run_id, trusted_launcher_sha=args.trusted_launcher_sha)
        elif args.mode == "execute":
            if args.execute_binding is None or args.runner_output is None or args.run_id is None:
                raise LauncherError("execute requires --execute-binding, --runner-output, and --run-id")
            if args.trusted_launcher_sha is not None:
                raise LauncherError("--trusted-launcher-sha is reserved for profile-execute")
            binding, launcher_trust = load_execute_binding(args.execute_binding)
            if binding.get("actual_eligible") is not True:
                reject_symlink_components(args.evidence_output, "execute evidence", allow_missing_leaf=True)
                if args.evidence_output.exists() or args.evidence_output.is_symlink():
                    raise LauncherError("execute evidence already exists")
                args.evidence_output.mkdir(mode=0o700)
                evidence = make_evidence("execute", sha_file(Path(__file__).resolve(), "launcher self")[0])
                evidence["execute_binding"] = binding
                evidence["failure"] = {"stage": "execute-binding", "reason": "execute binding is not actual eligible", "runner_started": False}
                finalize_output(args.evidence_output, evidence)
                code = 1
            else:
                code, evidence = execute_bound(binding, args.evidence_output, args.runner_output, args.run_id, trusted_launcher_sha=launcher_trust["sha256"])
        else:
            if args.trusted_launcher_sha is not None:
                raise LauncherError("--trusted-launcher-sha is reserved for profile-execute")
            code, evidence = launch(args.mode, args.evidence_output)
        print(json.dumps({"status": evidence["status"], "mode": evidence["mode"], "evidence": str(args.evidence_output / "launcher-evidence.json")}, sort_keys=True))
        return code
    except (LauncherError, OSError, ValueError) as error:
        print(f"AQ4 P2 immutable launcher failed before evidence creation: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
