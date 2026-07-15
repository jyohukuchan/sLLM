from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import struct
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "tools/build-aq4-layer0-ab-source-sidecars.py"
SPEC = importlib.util.spec_from_file_location("build_aq4_layer0_ab_source_sidecars", BUILDER_PATH)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)

ARTIFACT = ROOT / (
    "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/"
    "aq4-layer0-ab-hybrid-fidelity-v0.1"
)
INPUT = ROOT / (
    "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/"
    "aq4-layer0-family-isolation-v0.1/runtime-input.jsonl"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_report() -> dict:
    return json.loads((ARTIFACT / "report.json").read_text(encoding="utf-8"))


def _write_input(path: Path, rows: int = 3) -> None:
    header = {
        "dtype": "f32",
        "kind": "header",
        "schema_version": BUILDER.INPUT_SCHEMA,
        "shape": [BUILDER.INPUT_COLS],
        "tensor_name": BUILDER.INPUT_TENSOR,
    }
    lines = [json.dumps(header, sort_keys=True)]
    for step in range(rows):
        values = [((index + 1) * (step + 2) % 97) / 97.0 for index in range(BUILDER.INPUT_COLS)]
        payload = struct.pack(f"<{len(values)}f", *values)
        case = {
            "case_id": f"fixture-{step}",
            "input_sha256": hashlib.sha256(payload).hexdigest(),
            "kind": "case",
            "step": step,
            "values": values,
        }
        lines.append(json.dumps(case, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_source_model(path: Path) -> None:
    path.mkdir(parents=True)
    shard = path / "model.safetensors"
    a_name, _ = BUILDER.FAMILIES["a"]
    b_name, _ = BUILDER.FAMILIES["b"]
    a = torch.arange(BUILDER.OUTPUT_ROWS * BUILDER.INPUT_COLS, dtype=torch.float32).reshape(
        BUILDER.OUTPUT_ROWS, BUILDER.INPUT_COLS
    )
    b = torch.flip(a, dims=[1]) * 0.25
    save_file({a_name: a.to(torch.bfloat16), b_name: b.to(torch.bfloat16)}, str(shard))
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard.stat().st_size},
                "weight_map": {a_name: shard.name, b_name: shard.name},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_source_builder_emits_finite_three_row_a_and_b_raw_matmuls(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "out"
    _write_input(input_path)
    _write_source_model(source_dir)

    reports = BUILDER.build(
        Namespace(
            input=input_path,
            source_model=source_dir,
            output_a=None,
            output_b=None,
            output_dir=output_dir,
        )
    )
    assert set(reports) == {"a", "b"}
    for family, report in reports.items():
        output = output_dir / f"source-{family}.f32le"
        metadata = output.with_suffix(output.suffix + ".json")
        assert report["schema_version"] == BUILDER.SCHEMA
        assert report["family"] == family
        assert report["shape"] == [3, 32]
        assert report["source"]["tensor_shape"] == [32, 4096]
        assert report["source"]["tensor_dtype"] == "BF16"
        assert report["output"] == {
            "bytes": 3 * 32 * 4,
            "path": str(output),
            "sha256": sha256(output),
        }
        assert report["memory_policy"]["full_f32_weight_materialized"] is False
        assert report["promotion"] is False
        assert report["holdout"] == "not_run"
        assert report["policy_evaluation"] == "policy_not_evaluated"
        assert report["thresholds"] is None
        assert json.loads(metadata.read_text(encoding="utf-8")) == report
        values = struct.unpack("<96f", output.read_bytes())
        assert all(math.isfinite(value) for value in values)

    # The source-side output is the explicit f32 accumulation contract, not a
    # copy of the BF16 payload.  Check one row from both families end to end.
    input_values = json.loads(input_path.read_text(encoding="utf-8").splitlines()[1])["values"]
    input_tensor = torch.tensor(input_values, dtype=torch.float32)
    a_name, _ = BUILDER.FAMILIES["a"]
    b_name, _ = BUILDER.FAMILIES["b"]
    with BUILDER.safe_open(str(source_dir / "model.safetensors"), framework="pt", device="cpu") as handle:
        for family, name in (("a", a_name), ("b", b_name)):
            expected = torch.matmul(input_tensor, handle.get_tensor(name).to(torch.float32).T)
            actual = torch.tensor(
                struct.unpack("<32f", (output_dir / f"source-{family}.f32le").read_bytes()[: 32 * 4])
            )
            assert torch.equal(actual, expected)


def test_source_builder_requires_exactly_three_input_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    source_dir = tmp_path / "source"
    _write_input(input_path, rows=2)
    _write_source_model(source_dir)
    with pytest.raises(ValueError, match="exactly 3 rows"):
        BUILDER.build(
            Namespace(
                input=input_path,
                source_model=source_dir,
                output_a=tmp_path / "a.f32le",
                output_b=tmp_path / "b.f32le",
                output_dir=None,
            )
        )


def test_ab_hybrid_report_is_cpu_only_fail_closed_and_three_row_bound() -> None:
    report = load_report()
    assert report["schema_version"] == "ullm.aq4_layer0_ab_hybrid_fidelity.cpu.v1"
    assert report["status"] == "valid"
    assert report["promotion"] is False
    assert report["holdout"] == "not_run"
    assert report["policy_evaluation"] == "policy_not_evaluated"
    assert report["thresholds"] is None
    assert report["device"] == {"backend": "cpu", "requested_index": 0}
    assert report["input"]["rows"] == 3
    assert report["input"]["sha256"] == "c009a9bded30b1b9a7c704c622bd3106b3d17989c438f91eb20bb16817348e17"
    assert set(report) >= {"baseline", "a_only", "b_only", "combined", "metrics"}


def test_ab_variant_hash_fields_are_published_and_well_formed() -> None:
    report = load_report()
    hash_fields = (
        "a_sha256",
        "b_sha256",
        "gate_sha256",
        "beta_sha256",
        "recurrent_sha256",
        "attention_block_sha256",
        "layer_output_sha256",
    )
    prefixes = {"baseline": "baseline", "a_only": "a-only", "b_only": "b-only", "combined": "combined"}
    suffixes = {
        "a_sha256": "a",
        "b_sha256": "b",
        "gate_sha256": "gate",
        "beta_sha256": "beta",
        "recurrent_sha256": "recurrent",
        "attention_block_sha256": "attention-block",
        "layer_output_sha256": "layer-output",
    }
    for variant in ("baseline", "a_only", "b_only", "combined"):
        identity = report[variant].get("identity", report[variant])
        assert set(identity) >= set(hash_fields)
        for field in hash_fields:
            digest = identity[field]
            assert isinstance(digest, str) and len(digest) == 64
            assert digest == digest.lower()
            int(digest, 16)
            output = ARTIFACT / f"{prefixes[variant]}-{suffixes[field]}.f32le"
            assert output.is_file()
            assert sha256(output) == digest


def test_ab_hybrid_metrics_are_finite_and_expose_all_variants() -> None:
    report = load_report()
    metrics = report["metrics"]
    assert set(metrics) >= {
        "baseline_vs_source_a",
        "baseline_vs_source_b",
        "a_only_vs_source_a",
        "b_only_vs_source_b",
    }
    for name, item in metrics.items():
        assert len(item["per_step"]) == 3, name
        aggregate = item["aggregate"]
        assert aggregate["rows"] == 3, name
        assert aggregate["nonfinite"] is False, name
        assert all(row["nonfinite"] is False for row in item["per_step"]), name
        assert all(math.isfinite(float(row["relative_l2"])) for row in item["per_step"]), name
        assert math.isfinite(float(aggregate["relative_l2"])), name

    # Raw source replacement must exactly reproduce its BF16 source reference;
    # AQ4 baseline remains a measurable, finite comparison point.
    assert metrics["a_only_vs_source_a"]["aggregate"]["relative_l2"] == 0.0
    assert metrics["b_only_vs_source_b"]["aggregate"]["relative_l2"] == 0.0
    assert metrics["baseline_vs_source_a"]["aggregate"]["relative_l2"] > 0.0
    assert metrics["baseline_vs_source_b"]["aggregate"]["relative_l2"] > 0.0


def test_ab_hybrid_a_only_b_only_and_combined_reach_downstream() -> None:
    metrics = load_report()["metrics"]
    for variant in ("a_only", "b_only", "combined"):
        for boundary in ("attention_block", "layer_output"):
            key = f"{variant}_vs_baseline_{boundary}"
            assert key in metrics
            assert metrics[key]["aggregate"]["relative_l2"] > 0.0

    # Family-local projections must be separately observable, while the full
    # A+B override provides one combined downstream result.
    for variant, families in (("a_only", ("a",)), ("b_only", ("b",)), ("combined", ("a", "b"))):
        for family in families:
            key = f"{variant}_vs_baseline_{family}"
            assert key in metrics
            assert metrics[key]["aggregate"]["relative_l2"] > 0.0


def test_ab_source_sidecar_hashes_are_finite_and_bound_to_report() -> None:
    report = load_report()
    for family in ("a", "b"):
        source_meta = json.loads((ARTIFACT / f"source-{family}.f32le.json").read_text(encoding="utf-8"))
        output = ARTIFACT / f"source-{family}.f32le"
        assert source_meta["schema_version"] == "ullm.aq4_layer0_ab_source_sidecar.v1"
        assert source_meta["family"] == family
        assert source_meta["shape"] == [3, 32]
        assert source_meta["source"]["tensor_shape"] == [32, 4096]
        assert source_meta["source"]["tensor_dtype"] == "BF16"
        assert source_meta["output"]["sha256"] == sha256(output)
        assert output.stat().st_size == 3 * 32 * 4
        assert all(math.isfinite(value) for value in struct.unpack("<96f", output.read_bytes()))
        # The variant records carry the same immutable source output digests.
        identity = report["combined"].get("identity", report["combined"])
        assert identity[f"{family}_sha256"] == source_meta["output"]["sha256"]


def test_ab_comparison_to_qkv_and_z_is_present_when_published() -> None:
    comparison = load_report().get("comparison_to_existing_hybrids")
    if comparison is None:
        pytest.skip("comparison is appended by the artifact integration step")
    assert isinstance(comparison, dict)
    for family in ("qkv", "z"):
        assert family in comparison
        assert isinstance(comparison[family], dict)
        values = [value for value in comparison[family].values() if isinstance(value, (int, float))]
        assert values
        assert all(math.isfinite(float(value)) for value in values)
