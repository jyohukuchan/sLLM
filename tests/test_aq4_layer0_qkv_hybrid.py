from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-qkv-hybrid-fidelity-v0.1"
Z_ARTIFACT = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-z-hybrid-fidelity-v0.1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_report() -> dict:
    return json.loads((ARTIFACT / "diagnostic/report.json").read_text(encoding="utf-8"))


def test_qkv_hybrid_report_is_cpu_only_fail_closed_and_identity_bound() -> None:
    report = load_report()
    assert report["schema_version"] == "ullm.aq4_layer0_qkv_hybrid_fidelity.cpu.v1"
    assert report["status"] == "valid"
    assert report["promotion"] is False
    assert report["holdout"] == "not_run"
    assert report["policy_evaluation"] == "policy_not_evaluated"
    assert report["thresholds"] is None
    assert report["device"] == {"backend": "cpu", "requested_index": 0}
    assert report["input"]["rows"] == 3
    assert report["input"]["sha256"] == "c009a9bded30b1b9a7c704c622bd3106b3d17989c438f91eb20bb16817348e17"
    assert report["package"]["manifest_sha256"] == "a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad"
    assert report["state_reset"]["same_between_baseline_and_hybrid"] is True
    assert report["override"] == {
        "boundary": "after_production_qkv_matvec_before_production_depthwise_conv_split_norm_recurrent",
        "default_off": True,
        "family": "qkv",
        "promotion": False,
        "worker_reachable": False,
    }


def test_qkv_hybrid_metrics_replace_only_qkv_and_reach_layer_output() -> None:
    metrics = load_report()["metrics"]
    assert metrics["baseline_vs_source_qkv"]["aggregate"]["relative_l2"] > 0.025
    assert metrics["hybrid_vs_source_qkv"]["aggregate"]["relative_l2"] == 0.0
    assert metrics["hybrid_vs_source_qkv"]["aggregate"]["cosine"] == 1.0
    assert metrics["hybrid_vs_baseline_qkv"]["aggregate"]["relative_l2"] > 0.025
    assert metrics["hybrid_vs_baseline_recurrent"]["aggregate"]["relative_l2"] > 0.01
    assert metrics["hybrid_vs_baseline_attention_block"]["aggregate"]["relative_l2"] > 0.0
    qkv_layer_l2 = metrics["hybrid_vs_baseline_layer_output"]["aggregate"]["relative_l2"]
    z_report = json.loads((Z_ARTIFACT / "diagnostic/report.json").read_text(encoding="utf-8"))
    z_layer_l2 = z_report["metrics"]["hybrid_vs_baseline_layer_output"]["aggregate"]["relative_l2"]
    assert qkv_layer_l2 > z_layer_l2
    for item in metrics.values():
        assert len(item["per_step"]) == 3
        assert item["aggregate"]["nonfinite"] is False
        assert all(row["nonfinite"] is False for row in item["per_step"])


def test_qkv_source_metadata_and_recurrent_state_digests_are_bound() -> None:
    source = json.loads((ARTIFACT / "source-qkv.f32le.json").read_text(encoding="utf-8"))
    report = load_report()
    states = json.loads((ARTIFACT / "diagnostic/recurrent-state-digests.json").read_text(encoding="utf-8"))
    assert source["schema_version"] == "ullm.aq4_layer0_qkv_source_sidecar.v1"
    assert source["tensor_name"] == "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    assert source["shape"] == [3, 8192]
    assert source["source"]["tensor_shape"] == [8192, 4096]
    assert source["source"]["tensor_dtype"] == "BF16"
    assert source["source"]["tensor_payload_sha256"] == "21286bed37372105d182b20bc47c08e04e1fd3d3f0a968aa3af1f215ec7fb6a2"
    assert source["memory_policy"] == {"full_f32_weight_materialized": False, "output_row_chunk": 256}
    assert source["output"]["sha256"] == report["source_qkv"]["sha256"]
    assert states["schema_version"] == "ullm.aq4_layer0_qkv_hybrid_recurrent_state_digests.cpu.v1"
    assert states["state_shape"] == [32, 128, 128]
    assert states["changed_by_step"] == [True, True, True]
    assert [row["step"] for row in states["baseline"]] == [0, 1, 2]
    assert [row["step"] for row in states["hybrid"]] == [0, 1, 2]
    assert report["recurrent_state_digests"]["sha256"] == sha256(
        ARTIFACT / "diagnostic/recurrent-state-digests.json"
    )


def test_qkv_hybrid_sidecar_hashes_sizes_and_manifest_are_frozen() -> None:
    expected = {
        "source-qkv.f32le": ("d00333055f5a88d3d102fde62d351a1caa736579c9dc01871a942e6496a46018", 3 * 8192 * 4),
        "diagnostic/baseline-qkv.f32le": ("9683b8c5decd545c35e416da0b0f9568e6f51463ae5395fcd872dc9cbd82b473", 3 * 8192 * 4),
        "diagnostic/hybrid-qkv.f32le": ("d00333055f5a88d3d102fde62d351a1caa736579c9dc01871a942e6496a46018", 3 * 8192 * 4),
        "diagnostic/baseline-recurrent.f32le": ("e4a953aa4fe8af2dc57e6b8072ebfad42ba14d82f604e579e9e8a00247b5f581", 3 * 4096 * 4),
        "diagnostic/hybrid-recurrent.f32le": ("9f78dd6e195efeb14553d6fa65d3b13b1d57bf2ec86c9f5f5acd5f3bf69a1794", 3 * 4096 * 4),
        "diagnostic/baseline-attention-block.f32le": ("adee6df2dbd0c6215f0b11fcb4fd8e85a1bc06d7aa4ac156dc5993a7161dba0c", 3 * 4096 * 4),
        "diagnostic/hybrid-attention-block.f32le": ("3e99de5118ee5fad56436b47ab635c80cce0c34b14f96d4d201a6786d75d9b80", 3 * 4096 * 4),
        "diagnostic/baseline-layer-output.f32le": ("1de0f1aee1be8d376bffe44081195ff5dd09bbeb3e2b872e1c55db73385dccd8", 3 * 4096 * 4),
        "diagnostic/hybrid-layer-output.f32le": ("23a9c2662579736a965a7a8b0270240b29222ad94e603f1502a0ad8451feff01", 3 * 4096 * 4),
    }
    for relative, (digest, size) in expected.items():
        path = ARTIFACT / relative
        assert path.is_file()
        assert sha256(path) == digest
        assert path.stat().st_size == size

    sums = (ARTIFACT / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    assert len(sums) == 12
    for line in sums:
        digest, relative = line.split("  ", 1)
        assert sha256(ARTIFACT / relative) == digest
