from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "profile_aq4_p2_family_exclusive",
    ROOT / "tools/profile-aq4-p2-family-exclusive.py",
)
assert SPEC and SPEC.loader
PROFILE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROFILE
try:
    SPEC.loader.exec_module(PROFILE)
finally:
    sys.modules.pop(SPEC.name, None)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_trace(path: Path, rows: list[dict[str, object]], legacy: bool = False) -> None:
    fields = (
        ["Index", "KernelName", "BeginNs", "EndNs", "Phase"]
        if legacy
        else ["Dispatch_Id", "Kernel_Name", "Start_Timestamp", "End_Timestamp", "Phase"]
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def interval(
    dispatch: str,
    name: str,
    start: int,
    end: int,
    phase: str | None,
) -> object:
    return PROFILE.KernelInterval(
        dispatch,
        name,
        start,
        end,
        PROFILE.classify_kernel(name),
        phase,
    )


def bound_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    binary = tmp_path / "resident-driver"
    worker = tmp_path / "worker"
    product = tmp_path / "product"
    package = product / "package" / "manifest.json"
    package.parent.mkdir(parents=True)
    binary.write_bytes(b"resident-driver")
    binary.chmod(0o555)
    worker.write_bytes(b"worker")
    worker.chmod(0o555)
    os.link(worker, tmp_path / "worker-deps-hardlink")
    package.write_bytes(b"package-manifest")
    served = tmp_path / "served-model.json"
    guards = ["ULLM_REQUIRE_HIP_AQ4_KERNEL", "ULLM_REQUIRE_HIP_TOP1_KERNEL"]
    write_json(
        served,
        {
            "schema_version": "ullm.served_model.v2",
            "worker": {
                "binary": str(worker),
                "binary_sha256": sha256(worker),
                "required_environment": guards,
            },
            "product": {
                "root": str(product),
                "package": {
                    "manifest_path": "package/manifest.json",
                    "manifest_sha256": sha256(package),
                },
            },
            "public": {"id": "ullm-qwen3.5-9b-aq4", "revision": "candidate"},
            "format": {"format_id": "AQ4_0", "implementation_id": "rdna4-v1"},
        },
    )
    case = {
        "case_id": "representative-cold-prefill-m128",
        "case_sha256": None,
        "prefill_requested_m": 128,
        "resolved_m": 128,
        "device": PROFILE.EXPECTED_DEVICE,
    }
    case["case_sha256"] = PROFILE.case_sha256(case)
    cases = tmp_path / "case-binding.json"
    write_json(
        cases,
        {
            "schema_version": "ullm.aq4_production_p2_expanded.v2",
            "status": "bound_one_case_smoke",
            "case_count": 1,
            "canonical_case_sha256": PROFILE.canonical_sha256([case]),
            "cases": [case],
        },
    )
    identity = tmp_path / "identity.json"
    resident = {
        "binary_sha256": sha256(binary),
        "build_git_commit": "f" * 40,
        "protocol": "ullm.aq4_p2_resident_driver.v2",
        "package_manifest_sha256": sha256(package),
        "package_content_sha256": "b" * 64,
        "worker_binary_sha256": sha256(worker),
        "served_model_manifest_sha256": sha256(served),
        "guard_set_sha256": PROFILE.guard_set_sha256(guards),
        "model_id": "ullm-qwen3.5-9b-aq4",
        "model_revision": "candidate",
        "format_id": "AQ4_0",
        "implementation_id": "rdna4-v1",
        "runtime_device": PROFILE.EXPECTED_DEVICE,
    }
    identity_value = {
        "schema_version": "ullm.aq4_production_p2_identity.v2",
        "status": "bound",
        "identity_sha256": None,
        "build_git_commit": "f" * 40,
        "expanded_manifest_sha256": sha256(cases),
        "resident_driver_identity": resident,
        "hash_binding": {
            "bound_case_manifest_sha256": sha256(cases),
            "package_manifest_sha256": resident["package_manifest_sha256"],
            "package_content_sha256": resident["package_content_sha256"],
            "worker_binary_sha256": resident["worker_binary_sha256"],
            "served_model_manifest_sha256": resident["served_model_manifest_sha256"],
        },
    }
    identity_value["identity_sha256"] = PROFILE.self_sha256(
        identity_value, "identity_sha256"
    )
    write_json(identity, identity_value)
    policy = tmp_path / "policy.json"
    write_json(
        policy,
        {
            "schema_version": "ullm.aq4_production_p2_threshold_policy.v1",
            "status": "bound",
        },
    )
    return cases, identity, binary, package, policy, served


def snapshots(paths: tuple[Path, Path, Path, Path, Path, Path]) -> list[object]:
    return [
        PROFILE.capture(paths[0], "case", PROFILE.MAX_JSON_BYTES),
        PROFILE.capture(paths[1], "identity", PROFILE.MAX_JSON_BYTES),
        PROFILE.capture(paths[2], "binary"),
        PROFILE.capture(paths[3], "package"),
        PROFILE.capture(paths[4], "policy", PROFILE.MAX_JSON_BYTES),
        PROFILE.capture(paths[5], "served", PROFILE.MAX_JSON_BYTES),
    ]


def test_interval_union_and_family_exclusive_partition() -> None:
    values = [
        interval("1", "hip_aq4_matvec_kernel", 0, 10, "prefill"),
        interval("2", "hip_paged_decode_attention", 5, 15, "prefill"),
        interval("3", "hip_aq4_register_bm8", 8, 12, "prefill"),
        interval("4", "vendor_unknown_kernel", 14, 20, "prefill"),
        interval("5", "hip_rmsnorm_kernel", 30, 40, "decode"),
        interval("6", "hip_linear_attn_recurrent", 35, 42, "decode"),
        interval("7", "hip_top1_kernel", 40, 45, "decode"),
    ]
    total = PROFILE.aggregate(values)
    assert total["inclusive_sum_ns"] == 52
    assert total["gpu_total_union_ns"] == 35
    assert total["inclusive_overcount_ns"] == 17
    assert total["overlap_union_ns"] == 15
    assert total["cross_family_overlap_ns"] == 14
    assert total["unclassified_ns"] == 6
    assert total["families"]["aq4_projection"] == {
        "exclusive_ns": 5,
        "non_overlap_ns": 5,
        "active_union_ns": 12,
    }
    assert total["families"]["attention"]["exclusive_ns"] == 2
    assert total["families"]["normalization"]["exclusive_ns"] == 5
    assert total["families"]["head"]["exclusive_ns"] == 3


def test_build_artifact_keeps_prefill_decode_and_performance_separate(tmp_path: Path) -> None:
    trace = tmp_path / "trace.csv"
    write_trace(
        trace,
        [
            {
                "Dispatch_Id": "1",
                "Kernel_Name": "hip_aq4_matvec_kernel",
                "Start_Timestamp": 0,
                "End_Timestamp": 2_000_000,
                "Phase": "prefill",
            },
            {
                "Dispatch_Id": "2",
                "Kernel_Name": "hip_top1_kernel",
                "Start_Timestamp": 3_000_000,
                "End_Timestamp": 4_000_000,
                "Phase": "decode",
            },
        ],
    )
    trace_snapshot = PROFILE.capture(trace, "trace", PROFILE.MAX_TRACE_BYTES)
    values, schema = PROFILE.parse_trace(trace_snapshot)
    artifact = PROFILE.build_artifact(
        trace_snapshot=trace_snapshot,
        trace_schema=schema,
        intervals=values,
        binding_value={"case": "bound"},
        profiler_value={"tool": "rocprofv3", "version": "1.1.0"},
        command=["rocprofv3", "--", "resident"],
        maximum_unclassified_fraction=0.0,
    )
    assert artifact["timing_ns"]["prefill"]["gpu_total_union_ns"] == 2_000_000
    assert artifact["timing_ns"]["decode"]["gpu_total_union_ns"] == 1_000_000
    assert artifact["timing_ms"]["prefill"]["gpu_total_union_ms"] == 2.0
    assert artifact["timing_ms"]["prefill"]["kernel_count"] == 1
    assert artifact["measurement_eligible"] is False
    assert artifact["schedule_separation"] == {
        "warmup_runs": 2,
        "measured_runs": 10,
        "profile_aggregation_used_for_performance": False,
        "inclusive_kernel_sum_used_as_gpu_total": False,
    }


def test_parser_accepts_v3_and_legacy_timestamp_schemas(tmp_path: Path) -> None:
    v3 = tmp_path / "v3.csv"
    write_trace(
        v3,
        [{"Dispatch_Id": "7", "Kernel_Name": "hip_rmsnorm", "Start_Timestamp": 10, "End_Timestamp": 20, "Phase": "decode"}],
    )
    legacy = tmp_path / "legacy.csv"
    write_trace(
        legacy,
        [{"Index": "8", "KernelName": "hip_top1", "BeginNs": 20, "EndNs": 30, "Phase": "decode"}],
        legacy=True,
    )
    v3_values, v3_schema = PROFILE.parse_trace(PROFILE.capture(v3, "v3", 1024))
    legacy_values, legacy_schema = PROFILE.parse_trace(PROFILE.capture(legacy, "legacy", 1024))
    assert v3_values[0].family == "normalization"
    assert legacy_values[0].family == "head"
    assert PROFILE.classify_kernel("hip_qwen35_qk_norm_rope_batch_kernel") == "paged_validation"
    assert v3_schema["start_timestamp"] == "Start_Timestamp"
    assert legacy_schema["start_timestamp"] == "BeginNs"


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("__amd_rocclr_fillBufferAligned", "runtime_support"),
        ("__amd_rocclr_copyBuffer", "runtime_support"),
        ("ullm_bf16_row_f32_kernel", "embedding"),
        ("ullm_add_f32_kernel", "normalization"),
        ("ullm_aq4_matvec_silu_mul_f32_kernel", "aq4_projection"),
        ("ullm_aq4_matvec_add_f32_kernel", "aq4_projection"),
        ("hip_paged_kv_write_kernel", "paged_validation"),
        ("hip_paged_decode_attention", "attention"),
        ("hip_linear_attn_recurrent", "recurrent"),
        ("hip_rmsnorm_kernel", "normalization"),
        ("hip_top1_kernel", "head"),
    ],
)
def test_actual_runtime_names_and_existing_families_are_exclusive(
    name: str, family: str
) -> None:
    assert PROFILE.classify_kernel(name) == family


def test_unknown_and_multiple_family_names_remain_fail_closed() -> None:
    assert PROFILE.classify_kernel("vendor_new_unreviewed_kernel") is None
    with pytest.raises(PROFILE.ProfileError, match="matches multiple families"):
        PROFILE.classify_kernel("hip_aq4_matvec_rmsnorm_kernel")


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {"Dispatch_Id": "1", "Kernel_Name": "hip_top1", "Start_Timestamp": 10, "End_Timestamp": 20, "Phase": "decode"},
                {"Dispatch_Id": "1", "Kernel_Name": "hip_top1", "Start_Timestamp": 20, "End_Timestamp": 30, "Phase": "decode"},
            ],
            "duplicate dispatch",
        ),
        (
            [{"Dispatch_Id": "1", "Kernel_Name": "hip_top1", "Start_Timestamp": 20, "End_Timestamp": 10, "Phase": "decode"}],
            "invalid clock",
        ),
        (
            [
                {"Dispatch_Id": "1", "Kernel_Name": "hip_top1", "Start_Timestamp": 20, "End_Timestamp": 30, "Phase": "decode"},
                {"Dispatch_Id": "2", "Kernel_Name": "hip_top1", "Start_Timestamp": 10, "End_Timestamp": 15, "Phase": "decode"},
            ],
            "out of timestamp order",
        ),
        (
            [{"Dispatch_Id": "1", "Kernel_Name": "hip_top1", "Start_Timestamp": "", "End_Timestamp": 10, "Phase": "decode"}],
            "partial",
        ),
    ],
)
def test_parser_rejects_duplicate_clock_order_and_partial_rows(
    tmp_path: Path, rows: list[dict[str, object]], message: str
) -> None:
    trace = tmp_path / "bad.csv"
    write_trace(trace, rows)
    with pytest.raises(PROFILE.ProfileError, match=message):
        PROFILE.parse_trace(PROFILE.capture(trace, "bad trace", 4096))


def test_unknown_kernel_is_reported_and_thresholded(tmp_path: Path) -> None:
    trace = tmp_path / "unknown.csv"
    write_trace(
        trace,
        [{"Dispatch_Id": "1", "Kernel_Name": "brand_new_kernel", "Start_Timestamp": 0, "End_Timestamp": 10, "Phase": "decode"}],
    )
    snapshot = PROFILE.capture(trace, "trace", 4096)
    values, schema = PROFILE.parse_trace(snapshot)
    rejected = PROFILE.build_artifact(
        trace_snapshot=snapshot,
        trace_schema=schema,
        intervals=values,
        binding_value={},
        profiler_value={},
        command=["rocprofv3", "--", "resident"],
        maximum_unclassified_fraction=0.0,
    )
    accepted = PROFILE.build_artifact(
        trace_snapshot=snapshot,
        trace_schema=schema,
        intervals=values,
        binding_value={},
        profiler_value={},
        command=["rocprofv3", "--", "resident"],
        maximum_unclassified_fraction=1.0,
    )
    assert rejected["mapping"]["complete"] is False
    assert rejected["mapping"]["unknown_kernel_names"] == ["brand_new_kernel"]
    assert accepted["mapping"]["complete"] is True


def test_missing_phase_is_isolated_and_blocks_exact_phase_attribution(tmp_path: Path) -> None:
    trace = tmp_path / "no-phase.csv"
    trace.write_text(
        "Dispatch_Id,Kernel_Name,Start_Timestamp,End_Timestamp\n"
        "1,hip_top1,0,10\n",
        encoding="utf-8",
    )
    snapshot = PROFILE.capture(trace, "trace", 4096)
    values, schema = PROFILE.parse_trace(snapshot)
    artifact = PROFILE.build_artifact(
        trace_snapshot=snapshot,
        trace_schema=schema,
        intervals=values,
        binding_value={},
        profiler_value={},
        command=["rocprofv3", "--", "resident"],
        maximum_unclassified_fraction=0.0,
    )
    assert artifact["timing_ns"]["prefill"]["kernel_count"] == 0
    assert artifact["timing_ns"]["decode"]["kernel_count"] == 0
    assert artifact["timing_ns"]["unclassified_phase"]["gpu_total_union_ns"] == 10
    assert "trace lacks exact prefill/decode attribution" in artifact["eligibility_blockers"]


def test_binding_accepts_exact_hash_chain(tmp_path: Path) -> None:
    paths = bound_inputs(tmp_path)
    values = snapshots(paths)
    result, derived = PROFILE.binding(*values)
    assert result["case"]["prefill_requested_m"] == 128
    assert result["device"] == PROFILE.EXPECTED_DEVICE
    assert len(derived) == 2


def test_binding_rejects_malformed_bound_digest(tmp_path: Path) -> None:
    paths = bound_inputs(tmp_path)
    value = json.loads(paths[1].read_text(encoding="utf-8"))
    value["resident_driver_identity"]["package_content_sha256"] = "not-a-digest"
    value["hash_binding"]["package_content_sha256"] = "not-a-digest"
    value["identity_sha256"] = PROFILE.self_sha256(value, "identity_sha256")
    write_json(paths[1], value)
    with pytest.raises(PROFILE.ProfileError, match="not a SHA-256 digest"):
        PROFILE.binding(*snapshots(paths))


@pytest.mark.parametrize(
    ("variant", "message"),
    [
        ("binary", "binary hash"),
        ("package", "package manifest hash"),
        ("case", "M=128"),
        ("device", "exact R9700"),
        ("policy", "threshold policy"),
    ],
)
def test_binding_rejects_binary_package_case_device_and_policy_drift(
    tmp_path: Path, variant: str, message: str
) -> None:
    paths = bound_inputs(tmp_path)
    if variant == "binary":
        paths[2].chmod(0o644)
        paths[2].write_bytes(b"different-binary")
        paths[2].chmod(0o555)
    elif variant == "package":
        paths[3].write_bytes(b"different-package")
    elif variant in {"case", "device"}:
        value = json.loads(paths[0].read_text(encoding="utf-8"))
        case = value["cases"][0]
        if variant == "case":
            case["prefill_requested_m"] = 64
        else:
            case["device"] = {**PROFILE.EXPECTED_DEVICE, "runtime_device_index": 0}
        case["case_sha256"] = PROFILE.case_sha256(case)
        value["canonical_case_sha256"] = PROFILE.canonical_sha256(value["cases"])
        write_json(paths[0], value)
    else:
        write_json(paths[4], {"schema_version": "bad", "status": "bound"})
    with pytest.raises(PROFILE.ProfileError, match=message):
        PROFILE.binding(*snapshots(paths))


def test_snapshot_rejects_same_size_rewrite_and_rename(tmp_path: Path) -> None:
    path = tmp_path / "trace.csv"
    path.write_bytes(b"abcdef")
    snapshot = PROFILE.capture(path, "trace", 32)
    before = path.stat()
    path.write_bytes(b"ABCDEF")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(PROFILE.ProfileError, match="identity changed"):
        snapshot.verify()

    path.write_bytes(b"abcdef")
    snapshot = PROFILE.capture(path, "trace", 32)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"abcdef")
    os.chmod(replacement, stat.S_IMODE(path.stat().st_mode))
    os.replace(replacement, path)
    with pytest.raises(PROFILE.ProfileError, match="identity changed"):
        snapshot.verify()


def test_capture_rejects_file_and_ancestor_symlinks(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    path = real / "tool"
    path.write_bytes(b"tool")
    path.chmod(0o555)
    direct = tmp_path / "direct-link"
    direct.symlink_to(path)
    ancestor = tmp_path / "ancestor-link"
    ancestor.symlink_to(real, target_is_directory=True)
    with pytest.raises(PROFILE.ProfileError, match="symlink component"):
        PROFILE.capture(direct, "direct")
    with pytest.raises(PROFILE.ProfileError, match="symlink component"):
        PROFILE.capture(ancestor / "tool", "ancestor")


def test_resident_command_is_exact_and_rejects_substitution_or_argument_swap(
    tmp_path: Path,
) -> None:
    paths = bound_inputs(tmp_path)
    values = snapshots(paths)
    binding_value, _ = PROFILE.binding(*values)
    expected = [
        str(paths[2]),
        "--served-model-manifest",
        str(paths[5]),
        "--device-index",
        "1",
        "--build-git-commit",
        "f" * 40,
    ]
    assert PROFILE.validate_resident_command(
        expected, values[2], values[5], binding_value["identity"]["build_git_commit"]
    ) == expected
    other = tmp_path / "other-driver"
    other.write_bytes(paths[2].read_bytes())
    other.chmod(0o555)
    with pytest.raises(PROFILE.ProfileError, match="exactly match"):
        PROFILE.validate_resident_command(
            [str(other), *expected[1:]], values[2], values[5], "f" * 40
        )
    swapped = [expected[0], *expected[3:5], *expected[1:3], *expected[5:]]
    with pytest.raises(PROFILE.ProfileError, match="allowed argument schema"):
        PROFILE.validate_resident_command(swapped, values[2], values[5], "f" * 40)


def test_profiler_version_rejects_executable_swap(tmp_path: Path) -> None:
    profiler = tmp_path / "rocprofv3"
    profiler.write_bytes(b"original-profiler")
    profiler.chmod(0o555)
    snapshot = PROFILE.capture(profiler, "profiler", require_executable=True)

    def run(*_: object, **__: object) -> object:
        replacement = tmp_path / "replacement-profiler"
        replacement.write_bytes(b"replacement-tool!")
        replacement.chmod(0o555)
        os.replace(replacement, profiler)
        return type(
            "Result",
            (),
            {
                "returncode": 0,
                "stdout": b"version: 1.1.0\nrocm_version: 7.2.1\n",
            },
        )()

    with mock.patch.object(PROFILE.subprocess, "run", side_effect=run):
        with pytest.raises(PROFILE.ProfileError, match="identity changed"):
            PROFILE.profiler_version(snapshot)


@pytest.mark.parametrize("target", ["trace", "resident", "package", "case", "policy", "served"])
def test_main_rejects_input_toctou_after_attribution(tmp_path: Path, target: str) -> None:
    paths = bound_inputs(tmp_path)
    trace = tmp_path / "trace.csv"
    write_trace(
        trace,
        [{"Dispatch_Id": "1", "Kernel_Name": "hip_top1", "Start_Timestamp": 0, "End_Timestamp": 10, "Phase": "decode"}],
    )
    artifact = tmp_path / "artifact.json"

    targets = {
        "trace": trace,
        "resident": paths[2],
        "package": paths[3],
        "case": paths[0],
        "policy": paths[4],
        "served": paths[5],
    }

    def mutate() -> None:
        path = targets[target]
        before = path.stat()
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        path.chmod(stat.S_IMODE(before.st_mode))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

    PROFILE._TEST_HOOK = mutate
    try:
        with mock.patch.object(
            PROFILE,
            "profiler_version",
            return_value={"tool": "rocprofv3", "version": "1.1.0"},
        ):
            result = PROFILE.main(
                [
                    "parse",
                    "--trace",
                    str(trace),
                    "--case-binding",
                    str(paths[0]),
                    "--identity",
                    str(paths[1]),
                    "--resident-binary",
                    str(paths[2]),
                    "--package-manifest",
                    str(paths[3]),
                    "--policy",
                    str(paths[4]),
                    "--served-model-manifest",
                    str(paths[5]),
                    "--artifact",
                    str(artifact),
                    "--resident-command",
                    str(paths[2]),
                    "--served-model-manifest",
                    str(paths[5]),
                    "--device-index",
                    "1",
                    "--build-git-commit",
                    "f" * 40,
                ]
            )
    finally:
        PROFILE._TEST_HOOK = None
    assert result == 1
    assert not artifact.exists()


def test_profiler_wrapper_launches_resident_command_once(tmp_path: Path) -> None:
    output = tmp_path / "profile"
    profiler = tmp_path / "rocprofv3"
    profiler.write_bytes(b"profiler")
    profiler.chmod(0o555)
    profiler_snapshot = PROFILE.capture(profiler, "profiler", require_executable=True)
    command = PROFILE.profiler_command(
        profiler_snapshot, output, "one-case", ["resident-runner", "--one-case"]
    )
    calls: list[list[str]] = []

    def run(value: list[str], **_: object) -> object:
        calls.append(value)
        trace = output / "one-case_kernel_trace.csv"
        trace.write_text(
            "Dispatch_Id,Kernel_Name,Start_Timestamp,End_Timestamp,Phase\n"
            "1,hip_top1,0,10,decode\n",
            encoding="utf-8",
        )
        return type("Result", (), {"returncode": 0})()

    with mock.patch.object(PROFILE.subprocess, "run", side_effect=run):
        trace = PROFILE.run_profile(command, output, 30.0)
    assert trace.name == "one-case_kernel_trace.csv"
    assert calls == [command]
    assert command[-2:] == ["resident-runner", "--one-case"]
