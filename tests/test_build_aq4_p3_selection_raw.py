from __future__ import annotations

import importlib.util
import json
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
            "d2h_memcpy": True,
            "stream_synchronize": True,
            "device_synchronize": True,
        },
        "rocprof_config": {
            "kernel_trace": True,
            "hip_api_trace": True,
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
