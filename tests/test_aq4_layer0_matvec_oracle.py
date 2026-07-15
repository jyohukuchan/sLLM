from __future__ import annotations

import importlib.util
import hashlib
import struct
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("aq4_layer0_matvec_oracle", ROOT / "tools/aq4-layer0-matvec-oracle.py")
assert SPEC and SPEC.loader
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


def test_scale_table_is_the_pinned_e4m3_table() -> None:
    values = ORACLE.scale_values_e4m3()
    encoded = b"".join(struct.pack("<f", value) for value in values)
    assert len(values) == 119
    assert hashlib.sha256(encoded).hexdigest() == "c7f73f73756de5ddff4f96dab8d6d7ac5e5f383a270c3a889f94d6aa0279216e"


def test_streamed_dequant_matches_bound_cpu_reference(tmp_path: Path) -> None:
    package = tmp_path / "package"
    (package / "tensors").mkdir(parents=True)
    (package / "codebooks").mkdir()
    (package / "tensors/indices.idx4").write_bytes(bytes((0x21, 0x03)))
    (package / "tensors/scales.u8").write_bytes(bytes((0, 1)))
    (package / "codebooks/codebook.f32").write_bytes(struct.pack("<16f", *[float(value) for value in range(16)]))
    item = {
        "name": ORACLE.DEFAULT_TENSOR,
        "shape": [1, 4],
        "group_size": 2,
        "index_file": "tensors/indices.idx4",
        "scale_file": "tensors/scales.u8",
        "codebook_file": "codebooks/codebook.f32",
        "tensor_scale": 10.0,
    }
    vector = [0.5, -1.0, 2.0, 1.0]
    assert ORACLE.dequant_matvec(package, item, vector) == pytest.approx([0.205078125])
    assert ORACLE.dequant_matvec(package, item, vector, scalar_f32=True) == pytest.approx([0.205078125])


def test_cpu_formula_vs_runtime_f32_metric_is_finite() -> None:
    report = ORACLE.compare_vectors([112.5, 30.0], [112.5, 30.0])
    assert report["max_abs"] == 0.0
    assert report["relative_l2"] == 0.0
    assert report["cosine"] == pytest.approx(1.0)
    assert report["bit_mismatch_count"] == 0
    assert report["bit_mismatch_rate"] == 0.0
    assert report["finite"] is True


def test_runtime_f32_bit_metric_detects_single_lsb_difference() -> None:
    left = [struct.unpack("<f", bytes.fromhex("0000803f"))[0]]
    right = [struct.unpack("<f", bytes.fromhex("0100803f"))[0]]
    report = ORACLE.compare_vectors(left, right)
    assert report["bit_mismatch_count"] == 1
    assert report["bit_mismatch_rate"] == 1.0
    assert report["max_abs"] == pytest.approx(2.0**-23)
    assert report["finite"] is True


def test_runtime_identity_f32_encodings_are_explicit() -> None:
    assert ORACLE._f32(1.0e-6) == pytest.approx(9.999999974752427e-07)
    assert ORACLE._f32_bits_hex(1.0e-6) == "0x358637bd"
    assert ORACLE._f32_bits_hex(0.022842333) == "0x3cbb1fd8"


def test_candidate1_evidence_is_fail_closed_and_identity_bound() -> None:
    report_path = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-matvec-oracle-candidate1-v0.1/report-v5.json"
    report = __import__("json").loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked_missing_gpu_tensor_output"
    assert report["classification"] == "inconclusive_missing_gpu_tensor_output"
    assert report["input_norm_identity"]["raw_payload_sha256"] == "7d9dae62cc87d982dc1fa1b476eb3ee105ea096e663d4c2e749a0723f344301a"
    assert report["input_norm_identity"]["transform"] == "effective_rmsnorm_weight_values: raw_f32 + 1.0_f32"
    assert report["payload_identity"]["tensor_scale_f32_bits_hex"] == "0x3cbb1fd8"
    assert report["nonfinite_counts"] == {"cpu_outputs": 0, "input_vectors": 0, "runtime_cpu_outputs": 0, "source_outputs": 0}
    assert report["cpu_f32_reference_contract"]["execution"] == "offline_python_model; runtime API not invoked"
    assert report["cpu_f32_reference_contract"]["bit_exact_required"] is False
    for row in report["rows"]:
        assert "cosine" in row["cpu_vs_source"]
        assert "bit_mismatch_count" in row["cpu_formula_vs_f32_reference"]
        assert row["runtime_cpu_f32_api_invoked"] is False
        assert row["runtime_cpu_f32_bit_exact"] is False
        assert row["runtime_cpu_f32_bound_pass"] is True
