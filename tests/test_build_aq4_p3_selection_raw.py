from __future__ import annotations

import collections
import csv
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_aq4_p3_selection_raw",
    ROOT / "tools/build-aq4-p3-selection-raw.py",
)
assert SPEC and SPEC.loader
PRODUCER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PRODUCER
try:
    SPEC.loader.exec_module(PRODUCER)
finally:
    sys.modules.pop(SPEC.name, None)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def sha(path: Path) -> str:
    return PRODUCER.hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha(path)}


def identity_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    resident = {
        "binary_sha256": "b" * 64,
        "build_git_commit": "c" * 40,
        "protocol": "ullm.aq4_p2_resident_driver.v2",
        "worker_binary_sha256": "d" * 64,
        "package_manifest_sha256": "e" * 64,
        "package_content_sha256": "f" * 64,
        "served_model_manifest_sha256": "1" * 64,
        "model_id": "Qwen3.5-9B-AQ4",
        "model_revision": "fixture",
        "format_id": "AQ4_0",
        "implementation_id": "fixture-v1",
        "runtime_device": {
            "runtime_device_index": 1,
            "device_id": "r9700-rdna4",
            "backend": "hip",
            "name": "AMD Radeon Graphics",
            "architecture": "gfx1201",
        },
        "guard_set_sha256": "2" * 64,
    }
    value = {
        "schema_version": "ullm.aq4_production_p2_identity.v2",
        "status": "bound",
        "identity_sha256": None,
        "expanded_manifest_sha256": "a" * 64,
        "build_git_commit": "c" * 40,
        "resident_driver_identity": resident,
        "hash_binding": {
            "bound_case_manifest_sha256": "a" * 64,
            "worker_binary_sha256": "d" * 64,
            "package_manifest_sha256": "e" * 64,
            "package_content_sha256": "f" * 64,
            "served_model_manifest_sha256": "1" * 64,
        },
    }
    value["identity_sha256"] = PRODUCER.self_hash(value, "identity_sha256")
    path = tmp_path / "identity.json"
    write_json(path, value)
    return path, value


def run_record(case_id: str, run_index: int, resolved_m: int, prefill_ms: float) -> dict:
    return {
        "event": "run_complete",
        "schema_version": "ullm.aq4_p2_resident_driver.v2",
        "resident_session_id": "fixture-session",
        "case_id": case_id,
        "run_index": run_index,
        "run_kind": "warmup" if run_index < 2 else "measured",
        "status": "ok",
        "elapsed_ms": prefill_ms,
        "requested_m": resolved_m,
        "resolved_m": resolved_m,
        "actual_token_batch_width": resolved_m,
        "actual_request_batch_width": 1,
        "timing": {
            "prefill_ms": prefill_ms,
            "decode_ms": 0.0,
            "end_to_end_ms": prefill_ms,
            "generated_tokens": 0,
        },
        "audit": {
            "coverage_complete": True,
            "deterministic_digest_sha256": "3" * 64,
            "physical_operation_invocations": 1,
        },
        "state": {"baseline_before": True, "baseline_after": True, "request_state_sha256": "4" * 64},
        "lifecycle": {
            "prepare": 1,
            "commit": 1,
            "discard": 0,
            "error": 0,
            "cancel": 0,
            "reset": {"attempted": 1, "complete": 1, "failed": 0},
        },
        "reset": {"attempted": 1, "complete": 1, "failed": 0},
        "resource": {
            "samples": [{"monotonic_ms": 1.0}],
            "peak": {"vram_used_bytes": 1, "workspace_bytes": 1, "temporary_bytes": 1},
        },
        "terminal": {"reuse_forbidden": False, "reason_code": "none", "oom": False, "hip_fault": False},
    }


def device_lock_fixture(run_id: str) -> dict[str, object]:
    return {
        "schema_version": "ullm.aq4_p2_device_lock_owner.v1",
        "path": "/tmp/fixture-device.lock",
        "device": 26,
        "inode": 123456,
        "pid": 123,
        "hostname": "fixture-host",
        "run_id": run_id,
        "acquired_unix_ns": 123456789,
        "driver": {
            "path": "/fixture/resident-driver",
            "sha256": "b" * 64,
            "device": 1,
            "inode": 2,
            "nlink": 1,
        },
    }


def live_preflight_fixture(
    path: Path,
    run_id: str,
    identity: dict[str, object],
) -> dict[str, object]:
    runtime_index = identity["resident_driver_identity"]["runtime_device"][
        "runtime_device_index"
    ]
    captured_unix_ns = 987654321
    runtime_mapping = {
        "amd_smi_index": 2,
        "bdf": "0000:47:00.0",
        "kfd_id": 51545,
        "node_id": 2,
        "runtime_device_index": runtime_index,
        "uuid": "a8ff7551-0000-1000-80e9-ddefa2d60f55",
        "visible_token": str(runtime_index),
    }
    lock = {
        "path": "/tmp/fixture-device.lock",
        "free": True,
        "device": 26,
        "inode": 123456,
    }
    vram = {
        "total_bytes": 32_624_000_000,
        "used_bytes": 0,
        "free_bytes": 32_624_000_000,
        "headroom_bytes": 32_624_000_000,
    }
    command_exits = {
        "sudo-n": 0,
        "service-ullm-openai.service": 0,
        "service-llama-qwen35-udq4.service": 0,
        "old-worker": 1,
        "amd-smi-list": 0,
        "rocminfo": 0,
        "amd-smi-process": 0,
        "amd-smi-static-vram": 0,
    }
    environment = {
        field: "1"
        for field in PRODUCER.LIVE_PREFLIGHT_ENVIRONMENT_FIELDS
        if field.startswith("ULLM_REQUIRE_")
    }
    environment.update(
        {
            "HIP_VISIBLE_DEVICES": str(runtime_index),
            "ULLM_BUILD_GIT_COMMIT": identity["resident_driver_identity"][
                "build_git_commit"
            ],
            "ULLM_HIP_VISIBLE_DEVICES": str(runtime_index),
            "ULLM_SERVED_MODEL_MANIFEST": "/etc/ullm/served-models/active.json",
        }
    )
    document = {
        "schema_version": "ullm.aq4_p2_resident_live_preflight.v1",
        "status": "passed",
        "run_id": run_id,
        "captured_unix_ns": captured_unix_ns,
        "runtime_mapping": runtime_mapping,
        "lock": lock,
        "vram": vram,
        "commands": [
            {
                "argv": ["/fixture/tool", label],
                "captured_unix_ns": captured_unix_ns - len(command_exits) + index,
                "exit_code": exit_code,
                "label": label,
                "stderr_sha256": "e" * 64,
                "stdout_sha256": "f" * 64,
            }
            for index, (label, exit_code) in enumerate(command_exits.items())
        ],
        "compute_owners": {"amd_smi": [], "kfd": []},
        "environment": environment,
        "prepared_preflight": {
            "path": "/fixture/preflight.json",
            "role": "synthetic_bundle_contract_only",
            "sha256": "a" * 64,
        },
        "services": [
            {
                "unit": unit,
                "active_state": "inactive",
                "sub_state": "dead",
                "main_pid": 0,
            }
            for unit in ("ullm-openai.service", "llama-qwen35-udq4.service")
        ],
        "worker_pids": [],
    }
    write_json(path, document)
    path.chmod(0o444)
    info = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": sha(path),
        "device": info.st_dev,
        "inode": info.st_ino,
        "captured_unix_ns": captured_unix_ns,
        "runtime_mapping": runtime_mapping,
        "lock": lock,
        "vram": vram,
    }


def raw_fixture(
    path: Path,
    identity_path: Path,
    identity: dict[str, object],
    run_id: str,
    case_id: str,
    case_sha: str,
    resolved_m: int,
    prefill_ms: float,
    *,
    diagnostic: bool = False,
    live_preflight: dict[str, object] | None = None,
) -> Path:
    runs = [
        run_record(case_id, index, resolved_m, prefill_ms + (index % 3) * 0.1)
        for index in range(12)
    ]
    value = {
        "schema_version": "ullm.aq4_p2_resident_batch_raw.v1",
        "case_id": case_id,
        "case_sha256": case_sha,
        "status": "ok",
        "immutable_status": False,
        "baseline_identity": {
            "run_id": run_id,
            "kind": "p3-current-head",
            "identity_file": {"path": str(identity_path.resolve()), "sha256": sha(identity_path)},
        },
        "resident": {
            "session_id": "fixture-session",
            "model_loads": 1,
            "driver_identity": identity["resident_driver_identity"],
            "case_reset_count": 12,
        },
        "device_lock": device_lock_fixture(run_id),
        "workload": {
            "scope": "full_model",
            "phase": "cold_prefill",
            "mode": "cold_batched",
            "prompt_tokens": 128,
            "cached_prefix_tokens": 0,
            "context_tokens": 128,
            "prefill_requested_m": resolved_m,
            "resolved_m": resolved_m,
            "request_count": 1,
            "generated_tokens": 0,
        },
        "schedule": {"warmup_runs": 2, "measured_runs": 10, "completed_runs": 12},
        "runs": runs,
        "terminal": {"audit_digests": ["3" * 64] * 12, "reset_count": 12, "all_resets_complete": True},
        "failure_reason": None,
        "links": {
            "fixture": {"path": "/fixture", "sha256": "5" * 64},
            "identity": {"path": str(identity_path.resolve()), "sha256": sha(identity_path)},
            "policy": {"path": "/policy", "sha256": "6" * 64},
        },
    }
    if diagnostic:
        assert live_preflight is not None
        value["links"]["live_preflight"] = live_preflight
        value.update(
            {
                "execution_mode": "one_case_smoke",
                "smoke_only": True,
                "promotion_eligible": False,
            }
        )
    write_json(path, value)
    return path


def summary_fixture(
    path: Path,
    identity_path: Path,
    run_id: str,
    *,
    diagnostic: bool = False,
    live_preflight: dict[str, object] | None = None,
) -> Path:
    value = {
        "schema_version": "ullm.aq4_p2_resident_batch.v1",
        "status": "complete",
        "scope": "full_model",
        "case_count": 1 if diagnostic else 7,
        "completed_cases": 1 if diagnostic else 7,
        "warmup_runs": 2,
        "measured_runs": 10,
        "baseline_identity": {
            "run_id": run_id,
            "kind": "p3-current-head",
            "identity_file": {"path": str(identity_path.resolve()), "sha256": sha(identity_path)},
        },
        "device_lock": device_lock_fixture(run_id),
    }
    if diagnostic:
        assert live_preflight is not None
        value["validation"] = {"live_preflight": live_preflight}
        value.update(
            {
                "execution_mode": "one_case_smoke",
                "smoke_only": True,
                "promotion_eligible": False,
            }
        )
    write_json(path, value)
    return path


def write_kernel(path: Path, token: int, *, overlap: bool = False, unknown: bool = False) -> None:
    name = "brand_new_kernel" if unknown else "hip_paged_kv_write_kernel"
    rows = [f"{token},{name},{token * 1000},{token * 1000 + 100},prefill"]
    if overlap:
        rows.append(f"{token + 1},{name},{token * 1000 + 50},{token * 1000 + 150},prefill")
    path.write_text(
        "Dispatch_Id,Kernel_Name,Start_Timestamp,End_Timestamp,Phase\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def write_api(path: Path, token: int, *, overlap: bool = False, unknown: bool = False) -> None:
    base = token * 1000
    name = "hipMemcpyAsync" if unknown else "hipMemcpyDtoHAsync"
    rows = [f"{token},{name},{base},{base + 100}"]
    if overlap:
        rows.append(f"{token + 1},hipMemcpyDtoH,{base + 50},{base + 150}")
        rows.append(f"{token + 2},hipStreamSynchronize,{base + 200},{base + 300}")
        rows.append(f"{token + 3},hipDeviceSynchronize,{base + 250},{base + 350}")
    else:
        rows.append(f"{token + 1},hipStreamSynchronize,{base + 200},{base + 300}")
    path.write_text(
        "Correlation_Id,Function,Start_Timestamp,End_Timestamp\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def capability_fixture(path: Path, *, diagnostic: bool = False) -> tuple[Path, dict[str, object]]:
    value = {
        "schema_version": PRODUCER.CAPABILITY_SCHEMA,
        "status": "complete",
        "measurement_eligible": not diagnostic,
        "capability_sha256": None,
        "tool": {"name": "rocprofv3", "version": "fixture-3.0"},
        "domains": {
            "kernel_dispatch": True,
            "hip_api": True,
            "memory_copy": True,
            "d2h_memcpy": True,
            "stream_synchronize": True,
            "device_synchronize": True,
        },
        "rocprof_config": {
            "kernel_trace": True,
            "hip_api_trace": True,
            "memory_copy_trace": True,
            "marker_trace": True,
            "api_filter": "all_functions",
        },
    }
    value["capability_sha256"] = PRODUCER.self_hash(value, "capability_sha256")
    write_json(path, value)
    return path, value


def promotion_manifest(tmp_path: Path, *, all_m128: bool = False) -> tuple[Path, dict[str, object]]:
    identity_path, identity = identity_fixture(tmp_path)
    capability_path, _capability = capability_fixture(tmp_path / "capture-capabilities.json")
    summaries = []
    for run_id in ("profile-run", "baseline-run", "candidate-run"):
        path = tmp_path / f"summary-{run_id}.json"
        summary_fixture(path, identity_path, run_id)
        summaries.append(ref(path))

    cases = []
    ms = [128] * 7 if all_m128 else [128, 64, 128, 32, 128, 16, 8]
    token = 1
    for index, resolved_m in enumerate(ms):
        case_id = f"representative-{index}"
        case_sha = f"{index + 7:x}" * 64
        raw_path = raw_fixture(
            tmp_path / f"raw-{case_id}.json",
            identity_path,
            identity,
            "profile-run",
            case_id,
            case_sha,
            resolved_m,
            100.0,
        )
        profile_runs = []
        for run_index in range(2, 12):
            kernel = tmp_path / f"kernel-{index}-{run_index}.csv"
            api = tmp_path / f"api-{index}-{run_index}.csv"
            write_kernel(kernel, token)
            write_api(api, token)
            token += 10
            profile_runs.append(
                {
                    "schema_version": PRODUCER.PROFILE_BINDING_SCHEMA,
                    "case_id": case_id,
                    "case_sha256": case_sha,
                    "identity_sha256": identity["identity_sha256"],
                    "resident_run_index": run_index,
                    "measurement_eligible": True,
                    "clock_domain": "rocprofv3_monotonic_ns",
                    "kernel_trace_complete": True,
                    "hip_api_trace_complete": True,
                    "capture_capabilities": ref(capability_path),
                    "kernel_trace": ref(kernel),
                    "hip_api_trace": ref(api),
                }
            )
        cases.append(
            {
                "prompt_id": f"prompt-{index}",
                "case_id": case_id,
                "case_sha256": case_sha,
                "resolved_m": resolved_m,
                "resident_raw": ref(raw_path),
                "profile_runs": profile_runs,
            }
        )

    pair_case = "paired-full-model"
    pair_sha = "9" * 64
    baseline = raw_fixture(
        tmp_path / "pair-baseline.json",
        identity_path,
        identity,
        "baseline-run",
        pair_case,
        pair_sha,
        128,
        100.0,
    )
    contender = raw_fixture(
        tmp_path / "pair-candidate.json",
        identity_path,
        identity,
        "candidate-run",
        pair_case,
        pair_sha,
        128,
        90.0,
    )
    pairs = [
        {
            "pair_id": f"pair-{run_index}",
            "case_id": pair_case,
            "case_sha256": pair_sha,
            "run_index": run_index,
            "baseline_raw": ref(baseline),
            "candidate_raw": ref(contender),
        }
        for run_index in (2, 3, 4, 5, 6)
    ]
    manifest = {
        "schema_version": PRODUCER.INPUT_SCHEMA,
        "status": "promotion_ready",
        "measurement_eligible": True,
        "smoke_only": False,
        "promotion_eligible": True,
        "manifest_sha256": None,
        "candidate": {
            "candidate_id": "paged-kv-table-validation-v1",
            "family": "paged_validation",
        },
        "identity": ref(identity_path),
        "resident_summaries": summaries,
        "representative_cases": cases,
        "full_model_pairs": pairs,
    }
    manifest["manifest_sha256"] = PRODUCER.manifest_sha256(manifest)
    path = tmp_path / "producer-manifest.json"
    write_json(path, manifest)
    return path, manifest


def build_manifest(path: Path) -> dict[str, object]:
    snapshot = PRODUCER.capture(path.resolve(), "manifest")
    value = PRODUCER.parse_json(snapshot, "manifest")
    output, _snapshots = PRODUCER.build(value, snapshot)
    return output


def resident_pair_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    identity_path, identity = identity_fixture(tmp_path)
    run_id = "device-lock-fixture-run"
    live_preflight = live_preflight_fixture(
        tmp_path / "live-preflight.json", run_id, identity
    )
    summary_path = summary_fixture(
        tmp_path / "resident-summary.json",
        identity_path,
        run_id,
        diagnostic=True,
        live_preflight=live_preflight,
    )
    raw_path = raw_fixture(
        tmp_path / "resident-raw.json",
        identity_path,
        identity,
        run_id,
        "device-lock-fixture-case",
        "8" * 64,
        128,
        100.0,
        diagnostic=True,
        live_preflight=live_preflight,
    )
    return identity_path, summary_path, raw_path


def validate_resident_pair(
    identity_path: Path,
    summary_path: Path,
    raw_path: Path,
) -> tuple[dict[str, object], dict[str, object], str, list[dict[str, object]]]:
    identity_snapshot = PRODUCER.capture(identity_path.resolve(), "identity")
    identity_value = PRODUCER.parse_json(identity_snapshot, "identity")
    identity = PRODUCER.validate_identity(identity_value, identity_snapshot)

    summary_snapshot = PRODUCER.capture(summary_path.resolve(), "resident summary")
    summary_value = PRODUCER.parse_json(summary_snapshot, "resident summary")
    run_id = PRODUCER.validate_summary(
        summary_value,
        summary_snapshot,
        identity,
        "diagnostic",
    )

    raw_snapshot = PRODUCER.capture(raw_path.resolve(), "resident raw")
    raw_value = PRODUCER.parse_json(raw_snapshot, "resident raw")
    validated_run_id, runs = PRODUCER.validate_raw(
        raw_value,
        identity,
        {run_id: summary_snapshot},
        "diagnostic",
    )
    return summary_value, raw_value, validated_run_id, runs


def test_sealed_actual_v8_full_resident_pair_is_accepted() -> None:
    result_root = (
        ROOT
        / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2"
    )
    identity_path = (
        ROOT
        / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2"
        / "resident-one-case-smoke-prepared-v1/identity.json"
    )
    execute_root = result_root / "resident-one-case-smoke-profile-execute-v7"
    summary_path = execute_root / "resident-batch.summary.json"
    raw_path = execute_root / (
        "p2-representative-full_model-cold_prefill-cold_batched-n128-m128-"
        "r9700-rdna4-aq4_0_target.raw.json"
    )

    assert sha(raw_path) == "397f02a2cd87e5d30eb9eb569b5d022351b1f994358e71535f2ce697af5df25c"
    assert sha(summary_path) == "b82409bf997e207df5576ba7e38ebefddff363440c256250ffc8f7b521dcb3f5"
    summary, raw, run_id, runs = validate_resident_pair(
        identity_path,
        summary_path,
        raw_path,
    )

    assert run_id == "p2-r9700-resident-one-case-smoke-profile-diagnostic-v7"
    assert len(runs) == 12
    assert raw["device_lock"] == summary["device_lock"]
    assert raw["links"]["live_preflight"] == summary["validation"]["live_preflight"]
    for field in ("device", "inode"):
        assert type(raw["device_lock"][field]) is int
        assert raw["device_lock"][field] > 0


def test_profiler_family_authority_matches_current_git_and_mapping() -> None:
    authority = PRODUCER.PROFILER_GIT_AUTHORITY
    assert authority == {
        "commit": "e4f8583a0fc710d2146f70d06b8b49eb42f04a16",
        "tree": "be5ac39ea05b0b79223d974487c6cddda8d84f0c",
        "blob": "8c318849838f85cf2f2a687aef260506bfa4097c",
    }
    commit_tree = subprocess.run(
        ["git", "show", "-s", "--format=%T", authority["commit"]],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    committed_blob = subprocess.run(
        [
            "git",
            "rev-parse",
            f'{authority["commit"]}:tools/profile-aq4-p2-family-exclusive.py',
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    current_blob = subprocess.run(
        ["git", "hash-object", str(PRODUCER.PROFILER_PATH)],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    assert commit_tree == authority["tree"]
    assert committed_blob == current_blob == authority["blob"]
    assert sha(PRODUCER.PROFILER_PATH) == PRODUCER.PROFILER_SHA256
    assert PRODUCER.PROFILER.mapping_sha256() == PRODUCER.PROFILER_MAPPING_SHA256


def test_sealed_actual_v9_full_resident_pair_and_kernel_families_are_accepted() -> None:
    p2 = (
        ROOT
        / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2"
    )
    identity_path = (
        ROOT
        / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2"
        / "resident-one-case-smoke-prepared-v1/identity.json"
    )
    execute_root = p2 / "resident-one-case-smoke-profile-execute-v8"
    summary_path = execute_root / "resident-batch.summary.json"
    raw_path = execute_root / (
        "p2-representative-full_model-cold_prefill-cold_batched-n128-m128-"
        "r9700-rdna4-aq4_0_target.raw.json"
    )
    trace_path = (
        ROOT
        / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p3"
        / "aq4-p3-diagnostic-rocprof-capture-v8/aq4-p3-diagnostic_kernel_trace.csv"
    )

    assert sha(raw_path) == "1b2effa4c0ab44159919e32691d08329dec632cf56a1b22a78efc4fc607bf6f2"
    assert sha(summary_path) == "7b122428ede8e7dd5cc8780386d2f1274ac679c4206990ba45d6a334c2e66c8e"
    assert sha(trace_path) == "a9833a65cffd6cbc3e974edcfb32fdf5657a17f6e90321085bae734c51a07131"
    summary, raw, run_id, runs = validate_resident_pair(
        identity_path,
        summary_path,
        raw_path,
    )

    assert run_id == "p2-r9700-resident-one-case-smoke-profile-diagnostic-v8"
    assert raw["resident"]["session_id"] == (
        "3fc38e24c47e904242a3d3f12c9bd3250e53097d62dababbaec5efc4af34e0dc"
    )
    assert len(runs) == 12
    assert raw["device_lock"] == summary["device_lock"]
    assert raw["links"]["live_preflight"] == summary["validation"]["live_preflight"]

    family_counts: collections.Counter[str] = collections.Counter()
    with trace_path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            family = PRODUCER.PROFILER.classify_kernel(row["Kernel_Name"].strip())
            assert family is not None
            family_counts[family] += 1
    assert family_counts == {
        "runtime_support": 4071,
        "embedding": 1537,
        "paged_validation": 197,
        "aq4_projection": 2986,
        "attention": 104,
        "recurrent": 1158,
        "normalization": 2209,
        "head": 1,
    }
    assert family_counts.total() == 12_263


def rewrite_live_preflight_document(
    summary_path: Path,
    raw_path: Path,
    mutation,
) -> None:
    summary = json.loads(summary_path.read_text())
    raw = json.loads(raw_path.read_text())
    document_path = Path(summary["validation"]["live_preflight"]["path"])
    document = json.loads(document_path.read_text())
    mutation(document)
    document_path.chmod(0o644)
    write_json(document_path, document)
    document_path.chmod(0o444)
    for link in (
        summary["validation"]["live_preflight"],
        raw["links"]["live_preflight"],
    ):
        link["sha256"] = sha(document_path)
    write_json(summary_path, summary)
    write_json(raw_path, raw)


@pytest.mark.parametrize("artifact", ["summary", "raw"])
def test_diagnostic_live_preflight_is_required(tmp_path: Path, artifact: str) -> None:
    identity_path, summary_path, raw_path = resident_pair_fixture(tmp_path)
    target_path = summary_path if artifact == "summary" else raw_path
    value = json.loads(target_path.read_text())
    if artifact == "summary":
        del value["validation"]["live_preflight"]
    else:
        del value["links"]["live_preflight"]
    write_json(target_path, value)

    with pytest.raises(PRODUCER.ProducerError, match="live_preflight|links fields differ"):
        validate_resident_pair(identity_path, summary_path, raw_path)


def test_diagnostic_live_preflight_rejects_legacy_ref_shape(tmp_path: Path) -> None:
    identity_path, summary_path, raw_path = resident_pair_fixture(tmp_path)
    summary = json.loads(summary_path.read_text())
    link = summary["validation"]["live_preflight"]
    summary["validation"]["live_preflight"] = {
        "path": link["path"],
        "sha256": link["sha256"],
    }
    write_json(summary_path, summary)

    with pytest.raises(PRODUCER.ProducerError, match="live_preflight fields differ"):
        validate_resident_pair(identity_path, summary_path, raw_path)


def test_promotion_summary_rejects_diagnostic_live_preflight(tmp_path: Path) -> None:
    _path, manifest = promotion_manifest(tmp_path)
    summary_path = Path(manifest["resident_summaries"][0]["path"])
    summary = json.loads(summary_path.read_text())
    summary["validation"] = {
        "live_preflight": {"path": "/fixture/live.json", "sha256": "1" * 64}
    }
    write_json(summary_path, summary)
    manifest["resident_summaries"][0] = ref(summary_path)
    manifest["manifest_sha256"] = PRODUCER.manifest_sha256(manifest)
    manifest_path = tmp_path / "promotion-with-diagnostic-live-preflight.json"
    write_json(manifest_path, manifest)

    with pytest.raises(PRODUCER.ProducerError, match="must not carry diagnostic"):
        build_manifest(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda link: link.pop("captured_unix_ns"), "fields differ"),
        (lambda link: link.__setitem__("unknown", 1), "fields differ"),
        (lambda link: link.__setitem__("path", "relative/live.json"), "absolute"),
        (lambda link: link.__setitem__("path", "/tmp/../live.json"), "traversal-free"),
        (lambda link: link.__setitem__("sha256", "invalid"), "lowercase SHA-256"),
        (lambda link: link.__setitem__("sha256", "0" * 64), "path/SHA differs"),
    ],
)
def test_live_preflight_link_shape_path_and_hash_fail_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    identity_path, summary_path, raw_path = resident_pair_fixture(tmp_path)
    summary = json.loads(summary_path.read_text())
    mutation(summary["validation"]["live_preflight"])
    write_json(summary_path, summary)

    with pytest.raises(PRODUCER.ProducerError, match=message):
        validate_resident_pair(identity_path, summary_path, raw_path)


@pytest.mark.parametrize("field", ["device", "inode"])
@pytest.mark.parametrize("replacement", [True, 0, -1, "1"])
def test_live_preflight_file_identity_fields_require_positive_integers(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    identity_path, summary_path, raw_path = resident_pair_fixture(tmp_path)
    summary = json.loads(summary_path.read_text())
    summary["validation"]["live_preflight"][field] = replacement
    write_json(summary_path, summary)

    with pytest.raises(PRODUCER.ProducerError, match=rf"{field} must be a positive integer"):
        validate_resident_pair(identity_path, summary_path, raw_path)


@pytest.mark.parametrize("field", ["device", "inode"])
def test_live_preflight_file_identity_must_match_stat(tmp_path: Path, field: str) -> None:
    identity_path, summary_path, raw_path = resident_pair_fixture(tmp_path)
    summary = json.loads(summary_path.read_text())
    summary["validation"]["live_preflight"][field] += 1
    write_json(summary_path, summary)

    with pytest.raises(PRODUCER.ProducerError, match="document identity/mode differs"):
        validate_resident_pair(identity_path, summary_path, raw_path)


def test_live_preflight_file_mode_must_be_0444(tmp_path: Path) -> None:
    identity_path, summary_path, raw_path = resident_pair_fixture(tmp_path)
    summary = json.loads(summary_path.read_text())
    document_path = Path(summary["validation"]["live_preflight"]["path"])
    document_path.chmod(0o644)

    with pytest.raises(PRODUCER.ProducerError, match="document identity/mode differs"):
        validate_resident_pair(identity_path, summary_path, raw_path)


def test_live_preflight_file_must_have_one_link(tmp_path: Path) -> None:
    identity_path, summary_path, raw_path = resident_pair_fixture(tmp_path)
    summary = json.loads(summary_path.read_text())
    document_path = Path(summary["validation"]["live_preflight"]["path"])
    os.link(document_path, tmp_path / "second-live-preflight-link.json")

    with pytest.raises(PRODUCER.ProducerError, match="document identity/mode differs"):
        validate_resident_pair(identity_path, summary_path, raw_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("commands"), "document fields differ"),
        (lambda value: value.__setitem__("unknown", 1), "document fields differ"),
        (lambda value: value.__setitem__("schema_version", "v0"), "schema/status/run"),
        (lambda value: value.__setitem__("status", "failed"), "schema/status/run"),
        (lambda value: value.__setitem__("run_id", "different"), "schema/status/run"),
        (lambda value: value.__setitem__("captured_unix_ns", True), "positive integer"),
        (lambda value: value["runtime_mapping"].pop("bdf"), "runtime_mapping fields differ"),
        (
            lambda value: value["runtime_mapping"].__setitem__("unknown", 1),
            "runtime_mapping fields differ",
        ),
        (
            lambda value: value["runtime_mapping"].__setitem__("runtime_device_index", True),
            "non-negative integer",
        ),
        (lambda value: value["lock"].__setitem__("free", False), "free lock"),
        (lambda value: value["lock"].__setitem__("inode", 1), "device_lock"),
        (lambda value: value["vram"].__setitem__("used_bytes", 1), "idle headroom"),
        (lambda value: value["commands"][0].pop("argv"), "commands\\[0\\] fields differ"),
        (lambda value: value["compute_owners"]["kfd"].append(1), "must be empty"),
        (lambda value: value["environment"].pop("HIP_VISIBLE_DEVICES"), "fields differ"),
        (
            lambda value: value["prepared_preflight"].__setitem__("role", "unknown"),
            "prepared_preflight.role differs",
        ),
        (lambda value: value["services"][0].__setitem__("main_pid", True), "integer"),
        (lambda value: value["worker_pids"].append(1), "must be empty"),
    ],
)
def test_live_preflight_document_and_nested_contract_fail_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    identity_path, summary_path, raw_path = resident_pair_fixture(tmp_path)
    rewrite_live_preflight_document(summary_path, raw_path, mutation)

    with pytest.raises(PRODUCER.ProducerError, match=message):
        validate_resident_pair(identity_path, summary_path, raw_path)


@pytest.mark.parametrize(
    ("field_path", "replacement", "message"),
    [
        (("captured_unix_ns",), True, "positive integer"),
        (("captured_unix_ns",), 0, "positive integer"),
        (("captured_unix_ns",), -1, "positive integer"),
        (("captured_unix_ns",), "1", "positive integer"),
        (("runtime_mapping", "node_id"), True, "non-negative integer"),
        (("runtime_mapping", "kfd_id"), 0, "positive integer"),
        (("lock", "device"), -1, "positive integer"),
        (("vram", "total_bytes"), "1", "positive integer"),
    ],
)
def test_live_preflight_embedded_metadata_type_matrix(
    tmp_path: Path,
    field_path: tuple[str, ...],
    replacement: object,
    message: str,
) -> None:
    identity_path, summary_path, raw_path = resident_pair_fixture(tmp_path)
    summary = json.loads(summary_path.read_text())
    target = summary["validation"]["live_preflight"]
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = replacement
    write_json(summary_path, summary)

    with pytest.raises(PRODUCER.ProducerError, match=message):
        validate_resident_pair(identity_path, summary_path, raw_path)


def test_live_preflight_lock_binds_to_device_lock(tmp_path: Path) -> None:
    identity_path, summary_path, raw_path = resident_pair_fixture(tmp_path)
    summary = json.loads(summary_path.read_text())
    summary["validation"]["live_preflight"]["lock"]["inode"] += 1
    write_json(summary_path, summary)

    with pytest.raises(PRODUCER.ProducerError, match="differs from device_lock"):
        validate_resident_pair(identity_path, summary_path, raw_path)


def test_live_preflight_raw_and_summary_must_match(tmp_path: Path) -> None:
    identity_path, summary_path, raw_path = resident_pair_fixture(tmp_path)
    identity = json.loads(identity_path.read_text())
    second_link = live_preflight_fixture(
        tmp_path / "second-live-preflight.json",
        "device-lock-fixture-run",
        identity,
    )
    raw = json.loads(raw_path.read_text())
    raw["links"]["live_preflight"] = second_link
    write_json(raw_path, raw)

    with pytest.raises(PRODUCER.ProducerError, match="raw/summary live_preflight differs"):
        validate_resident_pair(identity_path, summary_path, raw_path)


@pytest.mark.parametrize("artifact", ["raw", "summary"])
@pytest.mark.parametrize("field", ["device", "inode"])
def test_device_lock_requires_device_and_inode(
    tmp_path: Path,
    artifact: str,
    field: str,
) -> None:
    identity_path, summary_path, raw_path = resident_pair_fixture(tmp_path)
    target_path = raw_path if artifact == "raw" else summary_path
    value = json.loads(target_path.read_text())
    value["device_lock"].pop(field)
    write_json(target_path, value)

    with pytest.raises(PRODUCER.ProducerError, match="device_lock fields differ"):
        validate_resident_pair(identity_path, summary_path, raw_path)


@pytest.mark.parametrize("artifact", ["raw", "summary"])
@pytest.mark.parametrize("field", ["device", "inode"])
@pytest.mark.parametrize("replacement", [0, -1, "1", True])
def test_device_lock_device_and_inode_are_positive_nonboolean_integers(
    tmp_path: Path,
    artifact: str,
    field: str,
    replacement: object,
) -> None:
    identity_path, summary_path, raw_path = resident_pair_fixture(tmp_path)
    target_path = raw_path if artifact == "raw" else summary_path
    value = json.loads(target_path.read_text())
    value["device_lock"][field] = replacement
    write_json(target_path, value)

    with pytest.raises(
        PRODUCER.ProducerError,
        match=rf"resident {artifact} device_lock\.{field} must be a positive integer",
    ):
        validate_resident_pair(identity_path, summary_path, raw_path)


@pytest.mark.parametrize("artifact", ["raw", "summary"])
def test_device_lock_rejects_unknown_field(tmp_path: Path, artifact: str) -> None:
    identity_path, summary_path, raw_path = resident_pair_fixture(tmp_path)
    target_path = raw_path if artifact == "raw" else summary_path
    value = json.loads(target_path.read_text())
    value["device_lock"]["unknown"] = 1
    write_json(target_path, value)

    with pytest.raises(PRODUCER.ProducerError, match="device_lock fields differ"):
        validate_resident_pair(identity_path, summary_path, raw_path)


@pytest.mark.parametrize(
    ("artifact", "field", "message"),
    [
        ("raw", "device", "raw/summary device_lock differs"),
        ("summary", "inode", "live_preflight.lock differs from device_lock"),
    ],
)
def test_device_lock_raw_and_summary_must_match(
    tmp_path: Path,
    artifact: str,
    field: str,
    message: str,
) -> None:
    identity_path, summary_path, raw_path = resident_pair_fixture(tmp_path)
    target_path = raw_path if artifact == "raw" else summary_path
    value = json.loads(target_path.read_text())
    value["device_lock"][field] += 1
    write_json(target_path, value)

    with pytest.raises(PRODUCER.ProducerError, match=message):
        validate_resident_pair(identity_path, summary_path, raw_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda lock: lock["driver"].__setitem__("sha256", "0" * 64),
            "device_lock driver SHA differs",
        ),
        (
            lambda lock: lock.__setitem__("run_id", "different-run"),
            "device_lock binding differs",
        ),
    ],
)
def test_device_lock_identity_and_run_binding_fail_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    identity_path, summary_path, raw_path = resident_pair_fixture(tmp_path)
    raw = json.loads(raw_path.read_text())
    mutation(raw["device_lock"])
    write_json(raw_path, raw)

    with pytest.raises(PRODUCER.ProducerError, match=message):
        validate_resident_pair(identity_path, summary_path, raw_path)


def test_hip_api_parser_counts_union_time_and_rejects_unknown(tmp_path: Path) -> None:
    _capability_path, capability = capability_fixture(tmp_path / "capability.json")
    trace = tmp_path / "api.csv"
    write_api(trace, 1, overlap=True)
    result = PRODUCER.parse_hip_api_trace(PRODUCER.capture(trace.resolve(), "api"), capability)
    assert result == {
        "d2h_count": 2,
        "d2h_union_ns": 150,
        "stream_sync_count": 2,
        "stream_sync_union_ns": 150,
    }
    unknown = tmp_path / "unknown-api.csv"
    write_api(unknown, 2, unknown=True)
    with pytest.raises(PRODUCER.ProducerError, match="unknown transfer"):
        PRODUCER.parse_hip_api_trace(PRODUCER.capture(unknown.resolve(), "unknown"), capability)
    empty = tmp_path / "empty-api.csv"
    empty.write_text(
        "Correlation_Id,Function,Start_Timestamp,End_Timestamp\n", encoding="utf-8"
    )
    with pytest.raises(PRODUCER.ProducerError, match="zero counts are not observable"):
        PRODUCER.parse_hip_api_trace(PRODUCER.capture(empty.resolve(), "empty"), capability)


def test_hip_api_zero_requires_hash_bound_complete_domain_proof(tmp_path: Path) -> None:
    _capability_path, capability = capability_fixture(tmp_path / "capability.json")
    trace = tmp_path / "unrelated-api.csv"
    trace.write_text(
        "Correlation_Id,Function,Start_Timestamp,End_Timestamp\n"
        "1,hipLaunchKernel,100,200\n"
        "2,hipMemcpyHtoDAsync,300,400\n",
        encoding="utf-8",
    )
    snapshot = PRODUCER.capture(trace.resolve(), "unrelated API")
    with pytest.raises(PRODUCER.ProducerError, match="require complete capture capabilities"):
        PRODUCER.parse_hip_api_trace(snapshot)
    result = PRODUCER.parse_hip_api_trace(snapshot, capability)
    assert result == {
        "d2h_count": 0,
        "d2h_union_ns": 0,
        "stream_sync_count": 0,
        "stream_sync_union_ns": 0,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["domains"].__setitem__("d2h_memcpy", False), "domain is incomplete"),
        (lambda value: value["rocprof_config"].__setitem__("api_filter", "selected"), "configuration is incomplete"),
        (lambda value: value["domains"].__setitem__("unknown_domain", True), "fields differ"),
    ],
)
def test_build_rejects_incomplete_or_ambiguous_capture_capability(
    tmp_path: Path, mutation, message: str
) -> None:
    _path, manifest = promotion_manifest(tmp_path)
    capability_ref = manifest["representative_cases"][0]["profile_runs"][0]["capture_capabilities"]
    capability_path = Path(capability_ref["path"])
    capability = json.loads(capability_path.read_text())
    mutation(capability)
    capability["capability_sha256"] = PRODUCER.self_hash(capability, "capability_sha256")
    write_json(capability_path, capability)
    for case in manifest["representative_cases"]:
        for binding in case["profile_runs"]:
            binding["capture_capabilities"] = ref(capability_path)
    manifest["manifest_sha256"] = PRODUCER.manifest_sha256(manifest)
    bad_path = tmp_path / "invalid-capability-manifest.json"
    write_json(bad_path, manifest)
    with pytest.raises(PRODUCER.ProducerError, match=message):
        build_manifest(bad_path)


def test_build_rejects_missing_or_hash_swapped_capture_capability(tmp_path: Path) -> None:
    _path, manifest = promotion_manifest(tmp_path)
    del manifest["representative_cases"][0]["profile_runs"][0]["capture_capabilities"]
    manifest["manifest_sha256"] = PRODUCER.manifest_sha256(manifest)
    missing_path = tmp_path / "missing-capability.json"
    write_json(missing_path, manifest)
    with pytest.raises(PRODUCER.ProducerError, match="fields differ"):
        build_manifest(missing_path)

    swap_root = tmp_path / "swap"
    swap_root.mkdir()
    _path, swapped = promotion_manifest(swap_root)
    capability_path = Path(
        swapped["representative_cases"][0]["profile_runs"][0]["capture_capabilities"]["path"]
    )
    capability = json.loads(capability_path.read_text())
    capability["tool"]["version"] = "different"
    capability["capability_sha256"] = PRODUCER.self_hash(capability, "capability_sha256")
    write_json(capability_path, capability)
    swapped["manifest_sha256"] = PRODUCER.manifest_sha256(swapped)
    swapped_path = swap_root / "hash-swapped-capability.json"
    write_json(swapped_path, swapped)
    with pytest.raises(PRODUCER.ProducerError, match="SHA-256 differs"):
        build_manifest(swapped_path)


def test_kernel_parser_uses_union_for_same_family_overlap_and_rejects_unknown(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "kernel.csv"
    write_kernel(trace, 1, overlap=True)
    result = PRODUCER.parse_kernel_trace(
        PRODUCER.capture(trace.resolve(), "kernel"), "paged-kv-table-validation-v1"
    )
    assert result["candidate_exclusive_ns"] == 150
    assert result["gpu_total_union_ns"] == 150
    unknown = tmp_path / "unknown-kernel.csv"
    write_kernel(unknown, 3, unknown=True)
    with pytest.raises(PRODUCER.ProducerError, match="unknown kernel"):
        PRODUCER.parse_kernel_trace(
            PRODUCER.capture(unknown.resolve(), "unknown"),
            "paged-kv-table-validation-v1",
        )


def test_promotion_build_emits_selector_compatible_hash_bound_raw(tmp_path: Path) -> None:
    path, _manifest = promotion_manifest(tmp_path)
    output = build_manifest(path)
    assert output["status"] == "complete"
    assert output["measurement_eligible"] is True
    assert output["promotion_eligible"] is True
    assert len(output["measurements"]) == 7
    assert len(output["full_model_pairs"]) == 5
    assert output["measurements"][0]["d2h_count"] == 10
    assert output["measurements"][0]["stream_sync_count"] == 10
    assert output["measurements"][0]["d2h_time_ms"] == pytest.approx(0.001)
    assert output["measurements"][0]["stream_sync_time_ms"] == pytest.approx(0.001)
    PRODUCER.SELECTOR.validate_raw(output)


def test_cli_publishes_once_and_refuses_overwrite(tmp_path: Path) -> None:
    path, _manifest = promotion_manifest(tmp_path)
    output = tmp_path / "selection-raw.json"
    assert PRODUCER.main(["--manifest", str(path), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["schema_version"] == PRODUCER.RAW_SCHEMA
    assert value["promotion_eligible"] is True
    original = output.read_bytes()
    assert PRODUCER.main(["--manifest", str(path), "--output", str(output)]) == 2
    assert output.read_bytes() == original


def test_output_publish_is_atomic_no_replace_and_fsyncs_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _manifest = promotion_manifest(tmp_path)
    value = build_manifest(path)
    output = tmp_path / "selection-raw.json"
    observed_fsync_kinds: list[bool] = []
    original_fsync = PRODUCER.os.fsync

    def recording_fsync(descriptor: int) -> None:
        observed_fsync_kinds.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
        original_fsync(descriptor)

    monkeypatch.setattr(PRODUCER.os, "fsync", recording_fsync)
    PRODUCER.write_output(output, value)
    assert output.is_file()
    assert observed_fsync_kinds == [False, True]

    competing = tmp_path / "competing-selection-raw.json"
    competitor_raw = b"competitor-wins\n"
    original_link = PRODUCER.os.link

    def racing_link(source, destination, **kwargs):
        Path(destination).write_bytes(competitor_raw)
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(PRODUCER.os, "link", racing_link)
    with pytest.raises(PRODUCER.ProducerError, match="refusing to overwrite"):
        PRODUCER.write_output(competing, value)
    assert competing.read_bytes() == competitor_raw
    assert list(tmp_path.glob(".competing-selection-raw.json.tmp-*")) == []


def test_manifest_array_order_is_semantically_invariant(tmp_path: Path) -> None:
    path, manifest = promotion_manifest(tmp_path)
    first = build_manifest(path)
    manifest["resident_summaries"].reverse()
    manifest["representative_cases"].reverse()
    for case in manifest["representative_cases"]:
        case["profile_runs"].reverse()
    manifest["full_model_pairs"].reverse()
    manifest["manifest_sha256"] = PRODUCER.manifest_sha256(manifest)
    second_path = tmp_path / "producer-manifest-reordered.json"
    write_json(second_path, manifest)
    second = build_manifest(second_path)
    assert first == second


def test_hash_swap_missing_prompt_m_and_pairing_fail_closed(tmp_path: Path) -> None:
    path, manifest = promotion_manifest(tmp_path)
    raw_path = Path(manifest["representative_cases"][0]["resident_raw"]["path"])
    value = json.loads(raw_path.read_text())
    value["workload"]["prompt_tokens"] += 1
    write_json(raw_path, value)
    with pytest.raises(PRODUCER.ProducerError, match="SHA-256 differs"):
        build_manifest(path)

    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    _path, missing = promotion_manifest(missing_root)
    missing["representative_cases"].pop()
    missing["manifest_sha256"] = PRODUCER.manifest_sha256(missing)
    missing_path = missing_root / "missing-prompt.json"
    write_json(missing_path, missing)
    with pytest.raises(PRODUCER.ProducerError, match="requires 7"):
        build_manifest(missing_path)

    m_root = tmp_path / "m"
    m_root.mkdir()
    m_path, _ = promotion_manifest(m_root, all_m128=True)
    with pytest.raises(PRODUCER.ProducerError, match="M=128 and another M"):
        build_manifest(m_path)

    pair_root = tmp_path / "pair"
    pair_root.mkdir()
    _path, broken = promotion_manifest(pair_root)
    broken["full_model_pairs"][0]["candidate_raw"] = broken["full_model_pairs"][0]["baseline_raw"]
    broken["manifest_sha256"] = PRODUCER.manifest_sha256(broken)
    broken_path = pair_root / "broken-pair.json"
    write_json(broken_path, broken)
    with pytest.raises(PRODUCER.ProducerError, match="run pairing differs"):
        build_manifest(broken_path)

    duplicate_root = tmp_path / "duplicate-pair"
    duplicate_root.mkdir()
    _path, duplicate = promotion_manifest(duplicate_root)
    duplicate["full_model_pairs"][1]["run_index"] = duplicate["full_model_pairs"][0]["run_index"]
    duplicate["manifest_sha256"] = PRODUCER.manifest_sha256(duplicate)
    duplicate_path = duplicate_root / "duplicate-pair.json"
    write_json(duplicate_path, duplicate)
    with pytest.raises(PRODUCER.ProducerError, match="reuses an underlying measured run sample"):
        build_manifest(duplicate_path)

    distinct_root = tmp_path / "distinct-pair"
    distinct_root.mkdir()
    _path, distinct = promotion_manifest(distinct_root)
    identity_path = Path(distinct["identity"]["path"])
    identity = json.loads(identity_path.read_text())
    pair_case = distinct["full_model_pairs"][0]["case_id"]
    pair_sha = distinct["full_model_pairs"][0]["case_sha256"]
    replacement_refs = []
    for role, run_id, prefill_ms in (
        ("baseline", "baseline-run-second-session", 101.0),
        ("candidate", "candidate-run-second-session", 91.0),
    ):
        summary = summary_fixture(
            distinct_root / f"summary-{run_id}.json", identity_path, run_id
        )
        distinct["resident_summaries"].append(ref(summary))
        raw = raw_fixture(
            distinct_root / f"pair-{role}-second-session.json",
            identity_path,
            identity,
            run_id,
            pair_case,
            pair_sha,
            128,
            prefill_ms,
        )
        replacement_refs.append(ref(raw))
    distinct["full_model_pairs"][1].update(
        {
            "run_index": distinct["full_model_pairs"][0]["run_index"],
            "baseline_raw": replacement_refs[0],
            "candidate_raw": replacement_refs[1],
        }
    )
    distinct["manifest_sha256"] = PRODUCER.manifest_sha256(distinct)
    distinct_path = distinct_root / "distinct-session-pair.json"
    write_json(distinct_path, distinct)
    output = build_manifest(distinct_path)
    assert len(output["full_model_pairs"]) == 5


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("measurement_eligible", False, "measurement eligibility differs"),
        ("hip_api_trace_complete", False, "case/identity/clock binding differs"),
        ("kernel_trace_complete", False, "case/identity/clock binding differs"),
    ],
)
def test_promotion_profile_binding_must_be_eligible_and_complete(
    tmp_path: Path, field: str, replacement: bool, message: str
) -> None:
    _path, manifest = promotion_manifest(tmp_path)
    manifest["representative_cases"][0]["profile_runs"][0][field] = replacement
    manifest["manifest_sha256"] = PRODUCER.manifest_sha256(manifest)
    path = tmp_path / f"invalid-{field}.json"
    write_json(path, manifest)
    with pytest.raises(PRODUCER.ProducerError, match=message):
        build_manifest(path)


def test_one_case_diagnostic_is_explicitly_non_promotable(tmp_path: Path) -> None:
    identity_path, identity = identity_fixture(tmp_path)
    capability_path, _capability = capability_fixture(
        tmp_path / "capture-capabilities.json", diagnostic=True
    )
    live_preflight = live_preflight_fixture(
        tmp_path / "live-preflight.json", "diagnostic-run", identity
    )
    summary = summary_fixture(
        tmp_path / "summary.json",
        identity_path,
        "diagnostic-run",
        diagnostic=True,
        live_preflight=live_preflight,
    )
    case_id = "diagnostic-case"
    case_sha = "8" * 64
    raw = raw_fixture(
        tmp_path / "raw.json",
        identity_path,
        identity,
        "diagnostic-run",
        case_id,
        case_sha,
        128,
        100.0,
        diagnostic=True,
        live_preflight=live_preflight,
    )
    kernel = tmp_path / "kernel.csv"
    api = tmp_path / "api.csv"
    write_kernel(kernel, 1)
    write_api(api, 1)
    manifest = {
        "schema_version": PRODUCER.INPUT_SCHEMA,
        "status": "one_case_diagnostic",
        "measurement_eligible": False,
        "smoke_only": True,
        "promotion_eligible": False,
        "manifest_sha256": None,
        "candidate": {
            "candidate_id": "paged-kv-table-validation-v1",
            "family": "paged_validation",
        },
        "identity": ref(identity_path),
        "resident_summaries": [ref(summary)],
        "representative_cases": [
            {
                "prompt_id": "diagnostic",
                "case_id": case_id,
                "case_sha256": case_sha,
                "resolved_m": 128,
                "resident_raw": ref(raw),
                "profile_runs": [
                    {
                        "schema_version": PRODUCER.PROFILE_BINDING_SCHEMA,
                        "case_id": case_id,
                        "case_sha256": case_sha,
                        "identity_sha256": identity["identity_sha256"],
                        "resident_run_index": 2,
                        "measurement_eligible": False,
                        "clock_domain": "rocprofv3_monotonic_ns",
                        "kernel_trace_complete": True,
                        "hip_api_trace_complete": True,
                        "capture_capabilities": ref(capability_path),
                        "kernel_trace": ref(kernel),
                        "hip_api_trace": ref(api),
                    }
                ],
            }
        ],
        "full_model_pairs": [],
    }
    manifest["manifest_sha256"] = PRODUCER.manifest_sha256(manifest)
    path = tmp_path / "diagnostic-manifest.json"
    write_json(path, manifest)
    output = build_manifest(path)
    assert output["status"] == "one_case_diagnostic"
    assert output["measurement_eligible"] is False
    assert output["smoke_only"] is True
    assert output["promotion_eligible"] is False
    with pytest.raises(PRODUCER.SELECTOR.SelectionError):
        PRODUCER.SELECTOR.validate_raw(output)


def test_producer_rejects_bool_int_float_type_substitution(tmp_path: Path) -> None:
    _path, manifest = promotion_manifest(tmp_path)
    manifest["measurement_eligible"] = 1
    manifest["manifest_sha256"] = PRODUCER.manifest_sha256(manifest)
    flag_path = tmp_path / "bad-flag.json"
    write_json(flag_path, manifest)
    with pytest.raises(PRODUCER.ProducerError, match="flags must be boolean"):
        build_manifest(flag_path)

    summary_root = tmp_path / "summary-type"
    summary_root.mkdir()
    _path, summary_manifest = promotion_manifest(summary_root)
    summary_path = Path(summary_manifest["resident_summaries"][0]["path"])
    summary = json.loads(summary_path.read_text())
    summary["warmup_runs"] = 2.0
    write_json(summary_path, summary)
    summary_manifest["resident_summaries"][0] = ref(summary_path)
    summary_manifest["manifest_sha256"] = PRODUCER.manifest_sha256(summary_manifest)
    bad_summary = summary_root / "bad-summary-type.json"
    write_json(bad_summary, summary_manifest)
    with pytest.raises(PRODUCER.ProducerError, match="summary schedule differs"):
        build_manifest(bad_summary)

    raw_root = tmp_path / "raw-type"
    raw_root.mkdir()
    _path, raw_manifest = promotion_manifest(raw_root)
    raw_path = Path(raw_manifest["representative_cases"][0]["resident_raw"]["path"])
    raw = json.loads(raw_path.read_text())
    raw["runs"][2]["run_index"] = 2.0
    write_json(raw_path, raw)
    raw_manifest["representative_cases"][0]["resident_raw"] = ref(raw_path)
    raw_manifest["manifest_sha256"] = PRODUCER.manifest_sha256(raw_manifest)
    bad_raw = raw_root / "bad-raw-type.json"
    write_json(bad_raw, raw_manifest)
    with pytest.raises(PRODUCER.ProducerError, match="run order/status differs"):
        build_manifest(bad_raw)

    reset_root = tmp_path / "reset-bool"
    reset_root.mkdir()
    _path, reset_manifest = promotion_manifest(reset_root)
    reset_path = Path(reset_manifest["representative_cases"][0]["resident_raw"]["path"])
    reset_raw = json.loads(reset_path.read_text())
    reset_raw["runs"][2]["reset"]["attempted"] = True
    write_json(reset_path, reset_raw)
    reset_manifest["representative_cases"][0]["resident_raw"] = ref(reset_path)
    reset_manifest["manifest_sha256"] = PRODUCER.manifest_sha256(reset_manifest)
    bad_reset = reset_root / "bad-reset-bool.json"
    write_json(bad_reset, reset_manifest)
    with pytest.raises(PRODUCER.ProducerError, match="must be a non-negative integer"):
        build_manifest(bad_reset)


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("elapsed_ms",), 100),
        (("timing", "prefill_ms"), 100),
        (("timing", "decode_ms"), 0),
        (("timing", "end_to_end_ms"), 100),
        (("resource", "samples", 0, "monotonic_ms"), 1),
    ],
)
def test_resident_raw_float_field_matrix_rejects_integer_substitution(
    tmp_path: Path, field_path: tuple[object, ...], replacement: int
) -> None:
    _path, manifest = promotion_manifest(tmp_path)
    raw_path = Path(manifest["representative_cases"][0]["resident_raw"]["path"])
    raw = json.loads(raw_path.read_text())
    target = raw["runs"][2]
    for part in field_path[:-1]:
        target = target[part]
    target[field_path[-1]] = replacement
    write_json(raw_path, raw)
    manifest["representative_cases"][0]["resident_raw"] = ref(raw_path)
    manifest["manifest_sha256"] = PRODUCER.manifest_sha256(manifest)
    bad_path = tmp_path / "integer-for-float.json"
    write_json(bad_path, manifest)
    with pytest.raises(PRODUCER.ProducerError, match="must be a finite float"):
        build_manifest(bad_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["resident_driver_identity"].pop("runtime_device"), "runtime_device"),
        (lambda value: value["hash_binding"].pop("package_manifest_sha256"), "hash_binding"),
        (lambda value: value["resident_driver_identity"].pop("worker_binary_sha256"), "fields differ"),
    ],
)
def test_identity_requires_source_build_runtime_and_package_bindings(
    tmp_path: Path, mutation, message: str
) -> None:
    identity_path, identity = identity_fixture(tmp_path)
    mutation(identity)
    identity["identity_sha256"] = PRODUCER.self_hash(identity, "identity_sha256")
    write_json(identity_path, identity)
    snapshot = PRODUCER.capture(identity_path.resolve(), "identity")
    value = PRODUCER.parse_json(snapshot, "identity")
    with pytest.raises(PRODUCER.ProducerError, match=message):
        PRODUCER.validate_identity(value, snapshot)


def test_identity_runtime_device_accepts_zero_index_and_integer_id_but_rejects_bool(
    tmp_path: Path,
) -> None:
    identity_path, identity = identity_fixture(tmp_path)
    runtime = identity["resident_driver_identity"]["runtime_device"]
    runtime["runtime_device_index"] = 0
    runtime["device_id"] = 0
    identity["identity_sha256"] = PRODUCER.self_hash(identity, "identity_sha256")
    write_json(identity_path, identity)
    snapshot = PRODUCER.capture(identity_path.resolve(), "identity")
    value = PRODUCER.parse_json(snapshot, "identity")
    PRODUCER.validate_identity(value, snapshot)

    value["resident_driver_identity"]["runtime_device"]["device_id"] = False
    value["identity_sha256"] = PRODUCER.self_hash(value, "identity_sha256")
    write_json(identity_path, value)
    snapshot = PRODUCER.capture(identity_path.resolve(), "identity")
    value = PRODUCER.parse_json(snapshot, "identity")
    with pytest.raises(PRODUCER.ProducerError, match="device_id is invalid"):
        PRODUCER.validate_identity(value, snapshot)
