from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
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
