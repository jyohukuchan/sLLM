from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_aq4_layer0_family_isolation",
    ROOT / "tools/compare-aq4-layer0-family-isolation.py",
)
assert SPEC and SPEC.loader
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


ARTIFACT = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-family-isolation-v0.1"


def test_family_probe_source_comparison_artifact_is_identity_bound() -> None:
    aq4 = json.loads((ARTIFACT / "aq4-report.json").read_text(encoding="utf-8"))
    report = json.loads((ARTIFACT / "compare/comparison.json").read_text(encoding="utf-8"))
    assert aq4["schema_version"] == "ullm.aq4_layer0_family_isolation.aq4_cpu.v1"
    assert aq4["status"] == "valid"
    assert aq4["device"].startswith("cpu:")
    assert aq4["family_order"] == ["qkv", "z", "a", "b"]
    assert aq4["input"]["rows"] == 3
    assert aq4["input"]["consumed_sha256"] == "c009a9bded30b1b9a7c704c622bd3106b3d17989c438f91eb20bb16817348e17"
    assert aq4["promotion"] is False
    assert aq4["holdout"] == "not_run"
    assert aq4["policy_evaluation"] == "policy_not_evaluated"
    assert aq4["one_at_a_time_hybrid"]["attempted"] is False
    assert aq4["one_at_a_time_hybrid"]["status"] == "not_implemented"
    assert report["schema_version"] == "ullm.aq4_layer0_family_isolation.source_compare.v1"
    assert report["status"] == "valid"
    assert report["promotion"] is False
    assert report["holdout"] == "not_run"
    assert report["thresholds"] is None
    assert report["input"]["sha256"] == aq4["input"]["consumed_sha256"]
    assert {family["family"] for family in report["families"]} == {"qkv", "z", "a", "b"}


def test_family_probe_reports_finite_three_row_metrics_and_candidate() -> None:
    report = json.loads((ARTIFACT / "compare/comparison.json").read_text(encoding="utf-8"))
    candidates = report["dominant_family_candidate"]
    assert candidates["status"] == "diagnostic_candidate_only"
    assert candidates["family"] == "z"
    assert candidates["metric"] == "aggregate.relative_l2"
    assert candidates["max_abs_family"] == "qkv"
    for family in report["families"]:
        aggregate = family["aggregate"]
        assert aggregate["rows"] == 3
        assert aggregate["finite_rows"] == 3
        assert aggregate["nonfinite_rows"] == 0
        assert aggregate["thresholds"] is None
        assert aggregate["policy_evaluation"] == "policy_not_evaluated"
        assert len(family["per_row"]) == 3
        assert all(row["nonfinite"] is False for row in family["per_row"])
        assert all(row["max_abs"] is not None for row in family["per_row"])
        assert all(row["relative_l2"] is not None for row in family["per_row"])
        assert all(row["cosine"] is not None for row in family["per_row"])


def test_metric_row_rejects_no_nonfinite_values() -> None:
    actual = TOOL.torch.tensor([1.0, 2.0], dtype=TOOL.torch.float32)
    reference = TOOL.torch.tensor([1.0, 2.0], dtype=TOOL.torch.float32)
    metric = TOOL.metric_row(actual, reference)
    assert metric["max_abs"] == 0.0
    assert metric["relative_l2"] == 0.0
    assert metric["cosine"] == pytest.approx(1.0)
    assert metric["nonfinite"] is False


def test_hybrid_is_explicitly_not_inferred() -> None:
    aq4 = json.loads((ARTIFACT / "aq4-report.json").read_text(encoding="utf-8"))
    reason = aq4["one_at_a_time_hybrid"]["reason"]
    assert "recurrent-state" in reason
    assert "not inferred" in reason
