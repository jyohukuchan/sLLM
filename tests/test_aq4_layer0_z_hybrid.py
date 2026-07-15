from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-z-hybrid-fidelity-v0.1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_z_hybrid_report_is_fail_closed_and_identity_bound() -> None:
    report = json.loads((ARTIFACT / "diagnostic/report.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "ullm.aq4_layer0_z_hybrid_fidelity.cpu.v1"
    assert report["status"] == "valid"
    assert report["promotion"] is False
    assert report["holdout"] == "not_run"
    assert report["policy_evaluation"] == "policy_not_evaluated"
    assert report["thresholds"] is None
    assert report["device"] == {"backend": "cpu", "requested_index": 0}
    assert report["input"]["rows"] == 3
    assert report["input"]["sha256"] == "c009a9bded30b1b9a7c704c622bd3106b3d17989c438f91eb20bb16817348e17"
    assert report["state_reset"]["same_between_baseline_and_hybrid"] is True
    assert report["override"]["boundary"] == "after_production_z_matvec_before_production_silu_mul"
    assert report["override"]["default_off"] is True
    assert report["override"]["worker_reachable"] is False


def test_z_hybrid_metrics_show_z_change_and_downstream_effect() -> None:
    report = json.loads((ARTIFACT / "diagnostic/report.json").read_text(encoding="utf-8"))
    metrics = report["metrics"]
    assert metrics["baseline_vs_source_z"]["aggregate"]["relative_l2"] > 0.02
    assert metrics["hybrid_vs_source_z"]["aggregate"]["relative_l2"] == 0.0
    assert metrics["hybrid_vs_source_z"]["aggregate"]["cosine"] == 1.0
    assert metrics["hybrid_vs_baseline_z"]["aggregate"]["relative_l2"] > 0.02
    assert metrics["hybrid_vs_baseline_attention_block"]["aggregate"]["relative_l2"] > 0.0
    assert metrics["hybrid_vs_baseline_layer_output"]["aggregate"]["relative_l2"] > 0.0
    for item in metrics.values():
        assert len(item["per_step"]) == 3
        assert all(row["nonfinite"] is False for row in item["per_step"])


def test_z_hybrid_sidecar_hashes_and_shapes_are_frozen() -> None:
    expected = {
        "source-z.f32le": "a087cb44d8c0f6831167a155fac67b1f8f991f5157fb33af89c51b0b4b497221",
        "diagnostic/baseline-z.f32le": "7ed98f1c7f8988958377b548f44afe3a2ddc5180150d1e3191c7d0e2a408b286",
        "diagnostic/hybrid-z.f32le": "a087cb44d8c0f6831167a155fac67b1f8f991f5157fb33af89c51b0b4b497221",
        "diagnostic/baseline-attention-block.f32le": "adee6df2dbd0c6215f0b11fcb4fd8e85a1bc06d7aa4ac156dc5993a7161dba0c",
        "diagnostic/hybrid-attention-block.f32le": "be4fd7b1d158fa53dc79caf4a945edb77f84a80ffa3c41caa803ec79592c846f",
        "diagnostic/baseline-layer-output.f32le": "1de0f1aee1be8d376bffe44081195ff5dd09bbeb3e2b872e1c55db73385dccd8",
        "diagnostic/hybrid-layer-output.f32le": "e2f2ac8f441ba9bd0bef1b906c13caafb0fb79612aeb22ad8612693e40d1fd37",
    }
    for relative, digest in expected.items():
        path = ARTIFACT / relative
        assert path.is_file()
        assert sha256(path) == digest
        assert path.stat().st_size == 3 * 4096 * 4

    sums = (ARTIFACT / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    assert len(sums) == 9
    for line in sums:
        digest, relative = line.split("  ", 1)
        assert sha256(ARTIFACT / relative) == digest
