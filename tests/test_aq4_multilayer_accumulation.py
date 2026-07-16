from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "benchmarks/results/2026-07-17/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-multilayer-accumulation-v0.1"
)
EXTENDED_ARTIFACT = ARTIFACT / "chain-0-11-v0.1"
HYBRID_SPEC = importlib.util.spec_from_file_location(
    "compare_aq4_layer0_hybrid",
    ROOT / "tools/compare-aq4-layer0-hybrid.py",
)
assert HYBRID_SPEC and HYBRID_SPEC.loader
HYBRID = importlib.util.module_from_spec(HYBRID_SPEC)
sys.modules[HYBRID_SPEC.name] = HYBRID
HYBRID_SPEC.loader.exec_module(HYBRID)

SPEC = importlib.util.spec_from_file_location(
    "compare_aq4_multilayer_accumulation",
    ROOT / "tools/compare-aq4-multilayer-accumulation.py",
)
assert SPEC and SPEC.loader
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def metric(layer_index: int, relative_l2: float) -> dict[str, object]:
    return {
        "layer_index": layer_index,
        "kind": "linear_attention",
        "aggregate": {"relative_l2": relative_l2, "cosine": 1.0, "max_abs": 0.0, "records": 1},
    }


def test_chain_range_requires_at_least_two_ascending_layers() -> None:
    assert TOOL.parse_layer_range("0:3") == (0, 3)
    for raw in ("0:0", "3:0", "-1:1", "invalid"):
        with pytest.raises(ValueError):
            TOOL.parse_layer_range(raw)


def test_extrapolation_detects_geometric_growth_and_reports_h8_fraction() -> None:
    result = TOOL.extrapolate([metric(0, 0.04), metric(1, 0.06), metric(2, 0.09), metric(3, 0.135)])
    assert result["shape"] == "superlinear"
    assert result["chosen_model"] == "geometric"
    assert result["geometric_mean_ratio"] == pytest.approx(1.5)
    assert result["chosen_fraction_of_production_final"] > 1.0
    assert result["verdict"] == "explains"


def test_extrapolation_uses_linear_model_for_linear_curve() -> None:
    result = TOOL.extrapolate([metric(0, 0.04), metric(1, 0.08), metric(2, 0.12), metric(3, 0.16)])
    assert result["shape"] == "approximately_linear_or_sublinear"
    assert result["chosen_model"] == "linear"
    assert result["linear_extrapolated_relative_l2_at_layer31"] == pytest.approx(1.28)


def test_rope_preserves_non_rotary_tail_and_is_finite() -> None:
    hidden = TOOL.torch.ones((2, 1, TOOL.SELF_HEAD_DIM), dtype=TOOL.torch.bfloat16)
    result = TOOL.source_rope(hidden)
    assert TOOL.torch.isfinite(result).all()
    assert TOOL.torch.equal(result[..., TOOL.ROTARY_DIM :], hidden[..., TOOL.ROTARY_DIM :])


def test_cpu_multilayer_artifact_binds_self_attention_and_h8_extrapolation() -> None:
    import json

    report = json.loads((ARTIFACT / "compare/comparison.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "ullm.aq4_multilayer_accumulation.source_compare.v1"
    assert report["status"] == "valid"
    assert report["device"] == "cpu-only"
    assert report["topology"]["source_config"]["self_attention_indices"] == [3, 7, 11, 15, 19, 23, 27, 31]
    assert report["topology"]["selected_layers"] == [
        {"kind": "linear_attention", "layer_index": 0},
        {"kind": "linear_attention", "layer_index": 1},
        {"kind": "linear_attention", "layer_index": 2},
        {"kind": "self_attention", "layer_index": 3},
    ]
    relative_l2 = [item["aggregate"]["relative_l2"] for item in report["layer_metrics"]]
    assert relative_l2 == pytest.approx([0.042451383744, 0.075075875044, 0.092594142713, 0.106253645855])
    growth = report["growth_curve"]
    assert growth["shape"] == "approximately_linear_or_sublinear"
    assert growth["chosen_model"] == "linear"
    assert growth["chosen_extrapolated_relative_l2_at_layer31"] == pytest.approx(0.850029166840)
    assert growth["verdict"] == "explains"


def test_epsilon_control_artifact_reports_small_layer_output_effect() -> None:
    import json

    default = json.loads(
        (ARTIFACT / "epsilon-control/runtime-default-compare/comparison.json").read_text(encoding="utf-8")
    )
    control = json.loads(
        (ARTIFACT / "epsilon-control/source-epsilon-compare/comparison.json").read_text(encoding="utf-8")
    )
    assert default["aq4_probe"]["post_rms_epsilon"] == 1e-5
    assert default["aq4_probe"]["post_rms_epsilon_mode"] == "aq4_runtime_default_1e-5"
    assert control["aq4_probe"]["post_rms_epsilon"] == 1e-6
    assert control["aq4_probe"]["post_rms_epsilon_mode"] == "source_1e-6_diagnostic_control"
    default_l2 = default["stages"]["layer_output"]["relative_l2"]
    control_l2 = control["stages"]["layer_output"]["relative_l2"]
    assert default_l2 == pytest.approx(0.042451383744)
    assert control_l2 == pytest.approx(0.042349396382)
    assert (default_l2 - control_l2) == pytest.approx(0.000101987362)


def test_extended_cpu_multilayer_artifact_records_nonmonotonic_h8_evidence() -> None:
    import json

    report = json.loads((EXTENDED_ARTIFACT / "compare/comparison.json").read_text(encoding="utf-8"))
    analysis = json.loads((EXTENDED_ARTIFACT / "analysis.json").read_text(encoding="utf-8"))
    assert report["status"] == "valid"
    assert report["device"] == "cpu-only"
    assert report["classification"] == "partially_explains"
    assert report["aq4_probe"]["binary_sha256"] == "e1139923fbd26d90f84b91aaa6e5449e595cdd46e04e013fc7c60a2d3e9b8fc1"
    assert [item["layer_index"] for item in report["topology"]["selected_layers"]] == list(range(12))
    assert [item["layer_index"] for item in report["layer_metrics"]] == list(range(12))
    assert [
        item["layer_index"] for item in report["topology"]["selected_layers"] if item["kind"] == "self_attention"
    ] == [3, 7, 11]
    relative_l2 = [item["aggregate"]["relative_l2"] for item in report["layer_metrics"]]
    assert relative_l2 == pytest.approx(
        [
            0.042451383744,
            0.075075875044,
            0.092594142713,
            0.106253645855,
            0.119418995374,
            0.125535704929,
            0.077142617728,
            0.094488065222,
            0.094775196394,
            0.092623375159,
            0.074961402054,
            0.080826992876,
        ]
    )
    assert relative_l2[5] == max(relative_l2)
    assert relative_l2[6] < relative_l2[5]
    growth = report["growth_curve"]
    assert growth["shape"] == "nonmonotonic_or_layer_jump"
    assert growth["chosen_model"] == "linear_conservative"
    assert growth["chosen_extrapolated_relative_l2_at_layer31"] == pytest.approx(0.215538647671)
    assert analysis["source_comparison"]["sha256"] == HYBRID.sha256_file(EXTENDED_ARTIFACT / "compare/comparison.json")
    assert analysis["scope"]["requested_layer_range"] == "0:11"
    assert analysis["scope"]["self_attention_indices_in_range"] == [3, 7, 11]
    assert analysis["resource_observation"]["exit_status"] == 0
    assert analysis["resource_observation"]["swap_operations"] == 0
    assert analysis["h8_assessment"]["verdict"] == "partially_explains"
    estimates = {item["name"]: item["estimate_at_layer31"] for item in analysis["extrapolations"]}
    assert estimates["full_window_signed_mean_delta"] == pytest.approx(0.150600827662)
    assert estimates["recent_four_signed_mean_delta"] == pytest.approx(0.012521631148)
    assert estimates["early_positive_delta_geometric_limit"] == pytest.approx(0.137305508365)
    assert estimates["self_attention_block_end_geometric_level"] == pytest.approx(0.040793209230)
