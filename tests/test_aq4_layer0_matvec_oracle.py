from __future__ import annotations

import importlib.util
import hashlib
import json
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


def test_rmsnorm_f32_uses_sequential_f32_operations() -> None:
    values = ORACLE.rmsnorm_f32_input([1.1, 2.2, 3.3], [1.111, 2.222, 3.333], 1.0e-6)
    assert [f"{struct.unpack('<I', struct.pack('<f', value))[0]:08x}" for value in values] == ["3f03a8b5", "4003a8b5", "40941dcc"]


def test_canonical_package_payload_rejects_escape(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    with pytest.raises(ORACLE.OracleError, match="relative package path"):
        ORACLE._canonical_package_payload(package, str(outside), "payload")
    with pytest.raises(ORACLE.OracleError, match="escapes canonical root"):
        ORACLE._canonical_package_payload(package, "../outside.bin", "payload")


def test_file_identity_detects_post_capture_change(tmp_path: Path) -> None:
    path = tmp_path / "identity.bin"
    path.write_bytes(b"before")
    identity = ORACLE._capture_file_identity(path, "identity")
    path.write_bytes(b"after")
    with pytest.raises(ORACLE.OracleError, match="changed after identity capture"):
        ORACLE._assert_file_identity(identity, "identity")


def test_trace_jsonl_streaming_rejects_blank_rows(tmp_path: Path) -> None:
    root = tmp_path / "trace"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema_version": ORACLE.TRACE_SCHEMA, "rows": 3}), encoding="utf-8")
    (root / "payload.jsonl").write_text("\n", encoding="utf-8")
    with pytest.raises(ORACLE.OracleError, match="blank"):
        ORACLE._load_trace(root, None, None)


def test_gpu_output_requires_full_identity_binding(tmp_path: Path) -> None:
    path = tmp_path / "gpu.json"
    path.write_text(json.dumps({"schema_version": ORACLE.TENSOR_OUTPUT_SCHEMA, "tensor_name": ORACLE.DEFAULT_TENSOR, "rows": []}), encoding="utf-8")
    with pytest.raises(ORACLE.OracleError, match="identity is missing"):
        ORACLE._load_tensor_outputs(path, {"name": ORACLE.DEFAULT_TENSOR}, {})


def test_gpu_output_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    identity = {
        "package_manifest_sha256": "a" * 64,
        "active_manifest_sha256": "b" * 64,
        "input_bindings_sha256": "c" * 64,
        "device": {"backend": "hip", "index": 1},
        "guard_set_sha256": "d" * 64,
        "operation": "wrong-operation",
        "effective_rpb": {"rows_per_block": 4, "threads_per_row": 64},
    }
    path = tmp_path / "gpu.json"
    path.write_text(json.dumps({"schema_version": ORACLE.TENSOR_OUTPUT_SCHEMA, "tensor_name": ORACLE.DEFAULT_TENSOR, "identity": identity, "rows": []}), encoding="utf-8")
    expected = dict(identity, operation=ORACLE.DEFAULT_GPU_OPERATION)
    with pytest.raises(ORACLE.OracleError, match="identity mismatch: operation"):
        ORACLE._load_tensor_outputs(path, {"name": ORACLE.DEFAULT_TENSOR}, expected)


def test_runtime_input_sidecar_is_atomic_and_probe_compatible(tmp_path: Path) -> None:
    vectors = {key: [float(index) for index in range(4096)] for key in ORACLE.EXPECTED_ROWS}
    bindings = {key: {"context_token_ids_sha256": "a" * 64, "context_length": 2, "input_sha256": ORACLE.vector_sha(vectors[key])} for key in ORACLE.EXPECTED_ROWS}
    path = tmp_path / "input.jsonl"
    identity = ORACLE.emit_runtime_input_jsonl(path, vectors, bindings)
    assert identity["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert header == {"dtype": "f32", "kind": "header", "schema_version": "ullm.aq4_layer0_input_normed_jsonl.v1", "shape": [4096], "tensor_name": ORACLE.DEFAULT_TENSOR}
    with pytest.raises(ORACLE.OracleError, match="refusing to overwrite"):
        ORACLE.emit_runtime_input_jsonl(path, vectors, bindings)


def test_candidate1_evidence_is_fail_closed_and_identity_bound() -> None:
    report_path = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-matvec-oracle-candidate1-v0.1/report-v7.json"
    report = __import__("json").loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked_missing_gpu_tensor_output"
    assert report["classification"] == "inconclusive_missing_gpu_tensor_output"
    assert report["input_norm_identity"]["raw_payload_sha256"] == "7d9dae62cc87d982dc1fa1b476eb3ee105ea096e663d4c2e749a0723f344301a"
    assert report["input_norm_identity"]["transform"] == "effective_rmsnorm_weight_values: raw_f32 + 1.0_f32"
    assert report["payload_identity"]["tensor_scale_f32_bits_hex"] == "0x3cbb1fd8"
    assert report["nonfinite_counts"] == {"cpu_outputs": 0, "input_vectors": 0, "runtime_cpu_outputs": 0, "source_outputs": 0}
    assert report["cpu_f32_reference_contract"]["execution"] == "offline_python_model; runtime API not invoked"
    assert report["cpu_f32_reference_contract"]["bit_exact_required"] is False
    assert report["file_identities"]["package manifest"]["sha256"] == report["package_manifest_sha256"]
    assert report["gpu_tensor_output_identity_contract"]["effective_rpb"] == {"rows_per_block": 4, "threads_per_row": 64}
    assert report["runtime_input_sidecar"]["sha256"] == "c009a9bded30b1b9a7c704c622bd3106b3d17989c438f91eb20bb16817348e17"
    assert report["gpu_comparison_tolerance"]["abs_tol"] == 1.0e-3
    assert report["promotion_eligible"] is False
    for row in report["rows"]:
        assert "cosine" in row["cpu_vs_source"]
        assert "bit_mismatch_count" in row["cpu_formula_vs_f32_reference"]
        assert row["runtime_cpu_f32_api_invoked"] is False
        assert row["runtime_cpu_f32_bit_exact"] is False
        assert row["runtime_cpu_f32_bound_pass"] is True
