from __future__ import annotations

import importlib.util
import json
import math
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
            "package_content_sha256": "f" * 64,
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
        "device_lock": {
            "schema_version": "ullm.aq4_p2_device_lock_owner.v1",
            "path": "/tmp/fixture-device.lock",
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
        },
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
    }
    if diagnostic:
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


def direct_trace_fixture(
    path: Path,
    *,
    case_id: str = "case",
    case_sha: str = "2" * 64,
    identity_sha: str = "3" * 64,
    binding_kind: str = "run",
    binding_id: str = "2",
    duplicate: bool = False,
    baseline_full_model_ms: float = 100.0,
    candidate_full_model_ms: float = 90.0,
    baseline_overrides: dict[str, object] | None = None,
    candidate_overrides: dict[str, object] | None = None,
) -> Path:
    events = []
    for side, d2d_bytes, copy_count in (("baseline", 1000, 10), ("candidate", 400, 4)):
        values = {
            "d2d_bytes": d2d_bytes,
            "d2d_copy_count": copy_count,
            "launch_count": copy_count,
            "component_ms": 12.0 if side == "baseline" else 8.0,
            "full_model_ms": baseline_full_model_ms if side == "baseline" else candidate_full_model_ms,
            "workspace_bytes": 2000,
            "peak_vram_bytes": 3000,
            "fallback_count": 0,
            "fallback_reasons": [],
            "alias_safe": True,
            "size_safe": True,
            "admission_safe": True,
            "fidelity_binding_sha256": "d" * 64,
        }
        values.update(
            baseline_overrides if side == "baseline" and baseline_overrides else {}
        )
        values.update(
            candidate_overrides if side == "candidate" and candidate_overrides else {}
        )
        allowed = PRODUCER.DIRECT_RUN_METRICS if binding_kind == "run" else PRODUCER.DIRECT_PAIR_METRICS
        for metric, value in values.items():
            if metric not in allowed:
                continue
            event = {
                "event_id": f"{side}-{metric}",
                "event_sha256": None,
                "side": side,
                "metric": metric,
                "value": value,
            }
            event["event_sha256"] = PRODUCER.self_hash(event, "event_sha256")
            events.append(event)
    if duplicate:
        events.append(events[-1].copy())
    value = {
        "schema_version": PRODUCER.DIRECT_TRACE_SCHEMA,
        "status": "complete",
        "trace_sha256": None,
        "binding_kind": binding_kind,
        "binding_id": binding_id,
        "candidate_id": "sequence-output-direct-v1",
        "case_id": case_id,
        "case_sha256": case_sha,
        "identity_sha256": identity_sha,
        "events": events,
    }
    value["trace_sha256"] = PRODUCER.self_hash(value, "trace_sha256")
    write_json(path, value)
    return path


def reseal_direct_trace(value: dict[str, object]) -> dict[str, object]:
    for event in value["events"]:
        event["event_sha256"] = PRODUCER.self_hash(event, "event_sha256")
    value["trace_sha256"] = PRODUCER.self_hash(value, "trace_sha256")
    return value


def parse_run_direct(path: Path) -> dict[str, dict[str, object]]:
    return PRODUCER.parse_direct_sequence_output_trace(
        PRODUCER.capture(path.resolve(), "direct"),
        case_id="case",
        case_sha256="2" * 64,
        identity_sha256="3" * 64,
        candidate_id="sequence-output-direct-v1",
        binding_kind="run",
        binding_id="2",
    )


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


def candidate_a_promotion_manifest(
    tmp_path: Path,
    *,
    run_overrides=None,
) -> tuple[Path, dict[str, object]]:
    _path, manifest = promotion_manifest(tmp_path)
    identity_path = Path(manifest["identity"]["path"])
    identity = json.loads(identity_path.read_text())
    manifest["candidate"] = {
        "candidate_id": "sequence-output-direct-v1",
        "family": "attention_recurrent",
    }
    for index, case in enumerate(manifest["representative_cases"]):
        case_id = case["case_id"]
        case_sha = case["case_sha256"]
        for binding in case["profile_runs"]:
            run_index = binding["resident_run_index"]
            baseline_overrides: dict[str, object] = {}
            candidate_overrides: dict[str, object] = {}
            if run_overrides is not None:
                baseline_overrides, candidate_overrides = run_overrides(index, run_index)
            direct = direct_trace_fixture(
                tmp_path / f"direct-{index}-{run_index}.json",
                case_id=case_id,
                case_sha=case_sha,
                identity_sha=identity["identity_sha256"],
                binding_id=str(run_index),
                baseline_full_model_ms=100.0 + (run_index % 3) * 0.1,
                candidate_full_model_ms=90.0 + (run_index % 3) * 0.1,
                baseline_overrides=baseline_overrides,
                candidate_overrides=candidate_overrides,
            )
            binding["direct_sequence_output_trace"] = ref(direct)
    for index, pair in enumerate(manifest["full_model_pairs"]):
        direct = direct_trace_fixture(
            tmp_path / f"pair-direct-{index}.json",
            case_id=pair["case_id"],
            case_sha=pair["case_sha256"],
            identity_sha=identity["identity_sha256"],
            binding_kind="pair",
            binding_id=pair["pair_id"],
        )
        pair["direct_sequence_output_trace"] = ref(direct)
    manifest["manifest_sha256"] = PRODUCER.manifest_sha256(manifest)
    manifest_path = tmp_path / "candidate-a-manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path, manifest


def build_manifest(path: Path) -> dict[str, object]:
    snapshot = PRODUCER.capture(path.resolve(), "manifest")
    value = PRODUCER.parse_json(snapshot, "manifest")
    output, _snapshots = PRODUCER.build(value, snapshot)
    return output


def pair_trace_target(
    tmp_path: Path,
) -> tuple[Path, dict[str, object], Path, dict[str, object]]:
    manifest_path, manifest = candidate_a_promotion_manifest(tmp_path)
    pair = manifest["full_model_pairs"][0]
    trace_path = Path(pair["direct_sequence_output_trace"]["path"])
    trace = json.loads(trace_path.read_text())
    return manifest_path, manifest, trace_path, trace


def update_pair_trace_ref(
    manifest_path: Path,
    manifest: dict[str, object],
    trace_path: Path,
    trace: dict[str, object],
    *,
    reseal: bool = True,
) -> None:
    if reseal:
        reseal_direct_trace(trace)
    write_json(trace_path, trace)
    pair = manifest["full_model_pairs"][0]
    pair["direct_sequence_output_trace"] = ref(trace_path)
    manifest["manifest_sha256"] = PRODUCER.manifest_sha256(manifest)
    write_json(manifest_path, manifest)


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


def test_candidate_a_direct_trace_binds_events_and_rejects_duplicate_or_tamper(
    tmp_path: Path,
) -> None:
    path = direct_trace_fixture(tmp_path / "direct.json")
    parsed = PRODUCER.parse_direct_sequence_output_trace(
        PRODUCER.capture(path.resolve(), "direct"),
        case_id="case",
        case_sha256="2" * 64,
        identity_sha256="3" * 64,
        candidate_id="sequence-output-direct-v1",
        binding_kind="run",
        binding_id="2",
    )
    assert parsed["candidate"]["d2d_bytes"] == 400
    duplicate_path = direct_trace_fixture(tmp_path / "duplicate.json", duplicate=True)
    with pytest.raises(PRODUCER.ProducerError, match="event ID|metric is duplicated"):
        PRODUCER.parse_direct_sequence_output_trace(
            PRODUCER.capture(duplicate_path.resolve(), "duplicate"),
            case_id="case",
            case_sha256="2" * 64,
            identity_sha256="3" * 64,
            candidate_id="sequence-output-direct-v1",
            binding_kind="run",
            binding_id="2",
        )
    tampered = json.loads(path.read_text())
    tampered["events"][0]["value"] = 999
    tampered_path = tmp_path / "tampered.json"
    write_json(tampered_path, tampered)
    with pytest.raises(PRODUCER.ProducerError, match="self-hash differs"):
        parse_run_direct(tampered_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["events"].pop(), "metrics are missing"),
        (
            lambda value: value["events"][0].__setitem__("metric", "unknown_metric"),
            "side/metric is unknown",
        ),
        (
            lambda value: value["events"].append(
                {
                    **value["events"][0],
                    "event_id": "unique-duplicate-metric",
                }
            ),
            "metric is duplicated",
        ),
    ],
)
def test_candidate_a_direct_trace_missing_unknown_and_duplicate_metric_matrix(
    tmp_path: Path, mutation, message: str
) -> None:
    path = direct_trace_fixture(tmp_path / "direct.json")
    value = json.loads(path.read_text())
    mutation(value)
    reseal_direct_trace(value)
    write_json(path, value)
    with pytest.raises(PRODUCER.ProducerError, match=message):
        parse_run_direct(path)


def test_candidate_a_direct_trace_rejects_nonfinite_json(tmp_path: Path) -> None:
    path = direct_trace_fixture(tmp_path / "direct.json")
    value = json.loads(path.read_text())
    value["events"][0]["value"] = math.nan
    path.write_text(json.dumps(value, allow_nan=True), encoding="utf-8")
    with pytest.raises(PRODUCER.ProducerError, match="non-finite JSON"):
        parse_run_direct(path)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("schema_version", "unknown.schema", "schema/status differs"),
        ("status", "partial", "schema/status differs"),
        ("candidate_id", "other-candidate", "candidate differs"),
        ("identity_sha256", "4" * 64, "identity differs"),
        ("case_id", "other-case", "case differs"),
        ("case_sha256", "5" * 64, "case differs"),
        ("binding_id", "3", "binding differs"),
        ("binding_kind", "pair", "binding differs"),
    ],
)
def test_candidate_a_direct_trace_root_identity_case_run_pair_matrix(
    tmp_path: Path, field: str, replacement: str, message: str
) -> None:
    path = direct_trace_fixture(tmp_path / "direct.json")
    value = json.loads(path.read_text())
    value[field] = replacement
    reseal_direct_trace(value)
    write_json(path, value)
    with pytest.raises(PRODUCER.ProducerError, match=message):
        parse_run_direct(path)


def test_candidate_a_direct_trace_root_and_file_hash_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    path = direct_trace_fixture(tmp_path / "direct.json")
    value = json.loads(path.read_text())
    value["trace_sha256"] = "0" * 64
    write_json(path, value)
    with pytest.raises(PRODUCER.ProducerError, match="self-hash differs"):
        parse_run_direct(path)

    with pytest.raises(PRODUCER.ProducerError, match="SHA-256 differs"):
        PRODUCER.load_ref(
            {"path": str(path.resolve()), "sha256": "0" * 64},
            "direct sequence output trace",
            [],
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("schema_version", "unknown.schema", "schema/status differs"),
        ("identity_sha256", "4" * 64, "identity differs"),
        ("candidate_id", "other-candidate", "candidate differs"),
        ("case_id", "other-case", "case differs"),
        ("case_sha256", "5" * 64, "case differs"),
        ("binding_id", "pair-other", "binding differs"),
    ],
)
def test_candidate_a_pair_trace_binding_matrix_uses_build_path(
    tmp_path: Path, field: str, replacement: str, message: str
) -> None:
    manifest_path, manifest, trace_path, trace = pair_trace_target(tmp_path)
    trace[field] = replacement
    update_pair_trace_ref(manifest_path, manifest, trace_path, trace)
    with pytest.raises(PRODUCER.ProducerError, match=message):
        build_manifest(manifest_path)


def test_candidate_a_pair_trace_root_self_hash_tamper_uses_build_path(
    tmp_path: Path,
) -> None:
    manifest_path, manifest, trace_path, trace = pair_trace_target(tmp_path)
    trace["trace_sha256"] = "0" * 64
    update_pair_trace_ref(manifest_path, manifest, trace_path, trace, reseal=False)
    with pytest.raises(PRODUCER.ProducerError, match="self-hash differs"):
        build_manifest(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["events"].pop(), "metrics are missing"),
        (
            lambda value: value["events"][0].__setitem__("metric", "unknown_metric"),
            "side/metric is unknown",
        ),
        (
            lambda value: value["events"].append(
                {
                    **value["events"][0],
                    "event_id": "unique-duplicate-metric",
                }
            ),
            "metric is duplicated",
        ),
    ],
)
def test_candidate_a_pair_trace_event_matrix_uses_build_path(
    tmp_path: Path, mutation, message: str
) -> None:
    manifest_path, manifest, trace_path, trace = pair_trace_target(tmp_path)
    mutation(trace)
    update_pair_trace_ref(manifest_path, manifest, trace_path, trace)
    with pytest.raises(PRODUCER.ProducerError, match=message):
        build_manifest(manifest_path)


def test_candidate_a_pair_trace_nonfinite_uses_build_path(tmp_path: Path) -> None:
    manifest_path, manifest, trace_path, trace = pair_trace_target(tmp_path)
    trace["events"][0]["value"] = math.nan
    trace_bytes = json.dumps(trace, ensure_ascii=True, sort_keys=True, allow_nan=True) + "\n"
    trace_path.write_text(trace_bytes, encoding="utf-8")
    pair = manifest["full_model_pairs"][0]
    pair["direct_sequence_output_trace"] = ref(trace_path)
    manifest["manifest_sha256"] = PRODUCER.manifest_sha256(manifest)
    write_json(manifest_path, manifest)
    with pytest.raises(PRODUCER.ProducerError, match="non-finite JSON"):
        build_manifest(manifest_path)


def test_candidate_a_pair_trace_file_hash_tamper_uses_build_path(
    tmp_path: Path,
) -> None:
    manifest_path, _manifest, trace_path, trace = pair_trace_target(tmp_path)
    trace["events"][0]["value"] = 999
    write_json(trace_path, trace)
    with pytest.raises(PRODUCER.ProducerError, match="SHA-256 differs"):
        build_manifest(manifest_path)


def test_candidate_a_pair_trace_reuse_uses_build_path(tmp_path: Path) -> None:
    manifest_path, manifest = candidate_a_promotion_manifest(tmp_path)
    pairs = manifest["full_model_pairs"]
    pairs[1]["direct_sequence_output_trace"] = pairs[0][
        "direct_sequence_output_trace"
    ]
    manifest["manifest_sha256"] = PRODUCER.manifest_sha256(manifest)
    write_json(manifest_path, manifest)
    with pytest.raises(PRODUCER.ProducerError, match="direct sequence trace was reused"):
        build_manifest(manifest_path)


def test_candidate_a_direct_trace_rejects_producer_fidelity_mismatch(
    tmp_path: Path,
) -> None:
    path = direct_trace_fixture(
        tmp_path / "direct.json",
        candidate_overrides={"fidelity_binding_sha256": "e" * 64},
    )
    with pytest.raises(PRODUCER.ProducerError, match="fidelity binding differs"):
        parse_run_direct(path)


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


def test_candidate_a_promotion_build_emits_direct_metrics_and_contract(tmp_path: Path) -> None:
    manifest_path, _manifest = candidate_a_promotion_manifest(tmp_path)
    output = build_manifest(manifest_path)
    assert output["measurements"][0]["candidate_d2d_bytes"] == 400
    assert output["measurements"][0]["candidate_component_p50_ms"] == 8.0
    assert output["full_model_pairs"][0]["candidate_d2d_copy_count"] == 4
    PRODUCER.SELECTOR.validate_raw(output)


def test_candidate_a_nonuniform_ten_run_percentiles_preserve_half_medians(
    tmp_path: Path,
) -> None:
    def overrides(_case_index: int, run_index: int):
        sample = run_index - 1
        upper_half = sample > 5
        return (
            {
                "d2d_bytes": 1001 if upper_half else 1000,
                "workspace_bytes": 2001 if upper_half else 2000,
                "peak_vram_bytes": 3001 if upper_half else 3000,
                "component_ms": float(sample),
            },
            {
                "d2d_bytes": 401 if upper_half else 400,
                "workspace_bytes": 2001 if upper_half else 2000,
                "peak_vram_bytes": 3001 if upper_half else 3000,
                "component_ms": float(sample) - 0.5,
                "full_model_ms": 79.0 + float(sample),
            },
        )

    path, _manifest = candidate_a_promotion_manifest(
        tmp_path, run_overrides=overrides
    )
    output = build_manifest(path)
    row = output["measurements"][0]
    assert row["baseline_d2d_bytes"] == 1000.5
    assert row["candidate_d2d_bytes"] == 400.5
    assert row["baseline_workspace_bytes"] == 2000.5
    assert row["baseline_peak_vram_bytes"] == 3000.5
    assert row["baseline_component_p50_ms"] == 5.5
    assert row["baseline_component_p95_ms"] == pytest.approx(9.55)
    assert row["candidate_component_p50_ms"] == 5.0
    assert row["candidate_component_p95_ms"] == pytest.approx(9.05)
    assert row["candidate_full_model_p50_ms"] == 84.5
    assert row["candidate_full_model_p95_ms"] == pytest.approx(88.55)
    PRODUCER.SELECTOR.validate_raw(output)


def test_candidate_a_direct_trace_reuse_across_runs_fails_closed(
    tmp_path: Path,
) -> None:
    _path, manifest = candidate_a_promotion_manifest(tmp_path)
    bindings = manifest["representative_cases"][0]["profile_runs"]
    bindings[1]["direct_sequence_output_trace"] = bindings[0][
        "direct_sequence_output_trace"
    ]
    manifest["manifest_sha256"] = PRODUCER.manifest_sha256(manifest)
    path = tmp_path / "reused-direct-manifest.json"
    write_json(path, manifest)
    with pytest.raises(PRODUCER.ProducerError, match="direct sequence trace was reused"):
        build_manifest(path)


def test_candidate_a_fidelity_binding_must_be_uniform_across_ten_runs(
    tmp_path: Path,
) -> None:
    def overrides(_case_index: int, run_index: int):
        fidelity = "e" * 64 if run_index == 11 else "d" * 64
        values = {"fidelity_binding_sha256": fidelity}
        return values, values

    path, _manifest = candidate_a_promotion_manifest(
        tmp_path, run_overrides=overrides
    )
    with pytest.raises(PRODUCER.ProducerError, match="changed across measured runs"):
        build_manifest(path)


def test_cli_publishes_once_and_refuses_overwrite(tmp_path: Path) -> None:
    path, _manifest = promotion_manifest(tmp_path)
    output = tmp_path / "selection-raw.json"
    assert PRODUCER.main(["--manifest", str(path), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["schema_version"] == PRODUCER.RAW_SCHEMA
    assert value["promotion_eligible"] is True
    assert PRODUCER.main(["--manifest", str(path), "--output", str(output)]) == 2


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
    summary = summary_fixture(
        tmp_path / "summary.json", identity_path, "diagnostic-run", diagnostic=True
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
