from __future__ import annotations

import hashlib
import json
import math
from array import array
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/"
    "aq4-layer0-qkv-z-combined-hybrid-fidelity-v0.1"
)
INPUT_SHA256 = "c009a9bded30b1b9a7c704c622bd3106b3d17989c438f91eb20bb16817348e17"
VARIANTS = ("baseline", "qkv_only", "z_only", "combined")
BOUNDARIES = ("qkv", "z", "recurrent", "attention_block", "layer_output")
HASH_FIELDS = tuple(f"{boundary}_sha256" for boundary in BOUNDARIES)
PREFIXES = {
    "baseline": "baseline",
    "qkv_only": "qkv-only",
    "z_only": "z-only",
    "combined": "combined",
}
ELEMENTS_PER_ROW = {"qkv": 8192, "z": 4096, "recurrent": 4096, "attention_block": 4096, "layer_output": 4096}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report_path() -> Path:
    for candidate in (ARTIFACT / "diagnostic/report.json", ARTIFACT / "report.json"):
        if candidate.is_file():
            return candidate
    raise AssertionError(f"combined isolation report is missing under {ARTIFACT}")


def load_report() -> dict[str, Any]:
    return json.loads(_report_path().read_text(encoding="utf-8"))


def _is_digest(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _variant_identity(report: dict[str, Any], variant: str) -> dict[str, Any]:
    value = report[variant]
    assert isinstance(value, dict), variant
    identity = value.get("identity", value)
    assert isinstance(identity, dict), variant
    return identity


def _state_rows(identity: dict[str, Any]) -> list[dict[str, Any]]:
    state = identity.get("recurrent_state_digests", identity.get("state_digests"))
    assert state is not None, "every variant must publish recurrent state digests"
    if isinstance(state, dict):
        state = state.get("rows", state.get("per_step", state.get("digests")))
    assert isinstance(state, list), "state digests must be a per-step list"
    return state


def _sidecar_path(variant: str, boundary: str, identity: dict[str, Any]) -> Path:
    filename = f"{PREFIXES[variant]}-{boundary.replace('_', '-')}.f32le"
    candidates = [ARTIFACT / filename, ARTIFACT / "diagnostic" / filename]
    paths = identity.get("paths", identity.get("files"))
    if isinstance(paths, dict):
        path_value = paths.get(boundary, paths.get(f"{boundary}_f32le"))
        if isinstance(path_value, str):
            path = Path(path_value)
            candidates.insert(0, path if path.is_absolute() else ARTIFACT / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise AssertionError(f"missing {variant}/{boundary} f32le sidecar")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _assert_metric_item(item: Any, name: str) -> None:
    assert isinstance(item, dict), name
    per_step = item.get("per_step")
    aggregate = item.get("aggregate")
    assert isinstance(per_step, list) and len(per_step) == 3, name
    assert isinstance(aggregate, dict) and aggregate.get("rows") == 3, name
    for row in [*per_step, aggregate]:
        assert isinstance(row, dict), name
        assert row.get("nonfinite") is False, name
        for field in ("relative_l2", "max_abs", "cosine"):
            if field in row:
                assert _finite_number(row[field]), f"{name}.{field}"


def _interaction_metric(metrics: dict[str, Any], boundary: str) -> Any:
    for key in (
        f"interaction_residual_{boundary}",
        f"combined_interaction_residual_{boundary}",
        f"combined_vs_additive_residual_{boundary}",
        f"combined_minus_additive_{boundary}",
    ):
        if key in metrics:
            return metrics[key]
    for container_name in ("interactions", "interaction_residuals", "interaction", "interaction_metrics"):
        container = metrics.get(container_name)
        if isinstance(container, dict) and boundary in container:
            value = container[boundary]
            if isinstance(value, dict):
                return value.get("combined_delta_vs_additive_prediction", value)
            return value
    raise AssertionError(f"missing interaction residual metric for {boundary}")


def _load_existing_layer_metric(relative: str, key: str) -> float:
    report = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    return float(report["metrics"][key]["aggregate"]["relative_l2"])


def test_qkv_z_combined_report_is_cpu_only_fail_closed_and_three_row_bound() -> None:
    report = load_report()
    assert report["schema_version"] == "ullm.aq4_layer0_qkv_z_combined_hybrid_fidelity.cpu.v1"
    assert report["status"] == "valid"
    assert report["promotion"] is False
    assert report["holdout"] == "not_run"
    assert report["policy_evaluation"] == "policy_not_evaluated"
    assert report["thresholds"] is None
    assert report["device"] == {"backend": "cpu", "requested_index": 0}
    assert report["input"]["rows"] == 3
    assert report["input"]["sha256"] == INPUT_SHA256
    assert set(report) >= set(VARIANTS) | {"metrics", "override", "state_reset"}

    override = report["override"]
    assert override["default_off"] is True
    assert override["promotion"] is False
    assert override["worker_reachable"] is False
    families = override.get("families", override.get("family", ()))
    if isinstance(families, str):
        families = (families,)
    assert set(families) >= {"qkv", "z"}
    assert "qkv" in override.get("qkv_boundary", "")
    assert "z" in override.get("z_boundary", "")

    state_reset = report["state_reset"]
    assert state_reset.get("same_between_all_variants", state_reset.get("same_between_baseline_and_hybrid")) is True


def test_qkv_z_combined_variant_hashes_are_finite_and_state_bound() -> None:
    report = load_report()
    for variant in VARIANTS:
        identity = _variant_identity(report, variant)
        assert identity.get("finite") is True, variant
        for field, boundary in zip(HASH_FIELDS, BOUNDARIES):
            digest = identity.get(field)
            assert _is_digest(digest), f"{variant}.{field}"
            sidecar = _sidecar_path(variant, boundary, identity)
            assert sha256(sidecar) == digest, f"{variant}.{field}"
            assert sidecar.stat().st_size == 3 * ELEMENTS_PER_ROW[boundary] * 4
            values = array("f")
            values.frombytes(sidecar.read_bytes())
            assert len(values) == 3 * ELEMENTS_PER_ROW[boundary]
            assert all(math.isfinite(float(value)) for value in values)

        states = _state_rows(identity)
        assert len(states) == 3, variant
        assert [row.get("step") for row in states] == [0, 1, 2]
        assert all(_is_digest(row.get("sha256")) for row in states)


def test_qkv_z_combined_metrics_cover_every_variant_boundary() -> None:
    metrics = load_report()["metrics"]
    for variant in VARIANTS[1:]:
        for boundary in BOUNDARIES:
            _assert_metric_item(metrics.get(f"{variant}_vs_baseline_{boundary}"), f"{variant}/{boundary}")

    # The two isolated direct projections and the combined projection must be
    # observable, and each override must reach the downstream state/output.
    assert metrics["qkv_only_vs_baseline_qkv"]["aggregate"]["relative_l2"] > 0.0
    assert metrics["qkv_only_vs_baseline_z"]["aggregate"]["relative_l2"] == 0.0
    for boundary in ("recurrent", "attention_block", "layer_output"):
        assert metrics[f"qkv_only_vs_baseline_{boundary}"]["aggregate"]["relative_l2"] > 0.0
    assert metrics["z_only_vs_baseline_qkv"]["aggregate"]["relative_l2"] == 0.0
    assert metrics["z_only_vs_baseline_z"]["aggregate"]["relative_l2"] > 0.0
    assert metrics["z_only_vs_baseline_recurrent"]["aggregate"]["relative_l2"] == 0.0
    for boundary in ("attention_block", "layer_output"):
        assert metrics[f"z_only_vs_baseline_{boundary}"]["aggregate"]["relative_l2"] > 0.0
    for boundary in ("qkv", "z", "recurrent", "attention_block", "layer_output"):
        assert metrics[f"combined_vs_baseline_{boundary}"]["aggregate"]["relative_l2"] > 0.0

    for key in ("qkv_only_vs_source_qkv", "combined_vs_source_qkv", "z_only_vs_source_z", "combined_vs_source_z"):
        _assert_metric_item(metrics.get(key), key)
        assert metrics[key]["aggregate"]["relative_l2"] == 0.0


def test_qkv_z_combined_interaction_residual_metrics_are_three_row_finite() -> None:
    report = load_report()
    metrics = dict(report["metrics"], interactions=report.get("interactions"))
    for boundary in ("recurrent", "attention_block", "layer_output"):
        item = _interaction_metric(metrics, boundary)
        _assert_metric_item(item, f"interaction/{boundary}")


def test_qkv_z_combined_has_numeric_comparison_to_existing_hybrids() -> None:
    report = load_report()
    expected = {
        "qkv": _load_existing_layer_metric(
            "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/"
            "aq4-layer0-qkv-hybrid-fidelity-v0.1/diagnostic/report.json",
            "hybrid_vs_baseline_layer_output",
        ),
        "z": _load_existing_layer_metric(
            "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/"
            "aq4-layer0-z-hybrid-fidelity-v0.1/diagnostic/report.json",
            "hybrid_vs_baseline_layer_output",
        ),
        "ab": _load_existing_layer_metric(
            "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/"
            "aq4-layer0-ab-hybrid-fidelity-v0.1/report.json",
            "combined_vs_baseline_layer_output",
        ),
    }
    observed = {
        "qkv": report["metrics"]["qkv_only_vs_baseline_layer_output"]["aggregate"]["relative_l2"],
        "z": report["metrics"]["z_only_vs_baseline_layer_output"]["aggregate"]["relative_l2"],
        "ab": expected["ab"],
    }
    for family, value in expected.items():
        assert _finite_number(value), family
        assert _finite_number(observed[family]), family
    assert math.isclose(observed["qkv"], expected["qkv"], rel_tol=1e-12, abs_tol=1e-15)
    assert math.isclose(observed["z"], expected["z"], rel_tol=1e-12, abs_tol=1e-15)
    # The combined QKV+Z effect is compared numerically with the existing
    # single-family and A+B diagnostics; this is descriptive, not a gate.
    combined = report["metrics"]["combined_vs_baseline_layer_output"]["aggregate"]["relative_l2"]
    assert _finite_number(combined)
    assert combined > max(expected.values())


def test_qkv_z_combined_interaction_formula_is_published() -> None:
    report = load_report()
    interactions = report.get("interactions")
    assert isinstance(interactions, dict)
    assert set(interactions) >= {"recurrent", "attention_block", "layer_output"}
    for boundary, item in interactions.items():
        assert item["definition"] == "combined_delta_minus_qkv_delta_plus_z_delta", boundary
        assert item["finite"] is True, boundary
        assert _finite_number(item["interaction_residual_l2"]), boundary
        assert _is_digest(item["interaction_residual_sha256"]), boundary
