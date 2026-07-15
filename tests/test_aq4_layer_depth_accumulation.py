from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / (
    "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/"
    "aq4-layer-depth-accumulation-v0.1"
)


def load_report() -> dict:
    return json.loads((ARTIFACT / "report.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_depth_report_is_cpu_fail_closed_and_explicitly_partial() -> None:
    report = load_report()
    assert report["schema_version"] == "ullm.aq4_layer_depth_accumulation.cpu.v1"
    assert report["status"] == "partial_valid_blocked_at_depth4"
    assert report["promotion"] is False
    assert report["holdout"] == "not_run"
    assert report["policy_evaluation"] == "policy_not_evaluated"
    assert report["thresholds"] is None
    assert report["device"] == {"backend": "cpu", "requested_index": 0}
    assert report["tracks"] == ["baseline", "qkv_only", "z_only", "combined"]


def test_topology_and_depth4_blocker_match_real_layer_order() -> None:
    report = load_report()
    assert report["layer_topology"]["layer_types_0_3"] == [
        "linear_attention", "linear_attention", "linear_attention", "full_attention"
    ]
    assert report["maximum_faithful_depth"] == 3
    blocker = report["requested_depth4"]
    assert blocker["status"] == "blocked_not_run"
    assert blocker["blocking_layer_index"] == 3
    assert blocker["layer_type"] == "full_attention"
    assert "monolithic" in blocker["exact_blocker"]
    assert "external residual" in blocker["required_hook"]
    assert "track-local state" in blocker["required_hook"]


def test_depth_checkpoints_bind_hidden_recurrent_and_finite_metrics() -> None:
    report = load_report()
    assert set(report["checkpoints"]) == {"1", "2", "3"}
    for depth, checkpoint in report["checkpoints"].items():
        assert checkpoint["depth"] == int(depth)
        for track, item in checkpoint["hidden"].items():
            path = Path(item["path"])
            assert path.is_file(), (depth, track)
            assert item["shape"] == [3, 4096]
            assert item["sha256"] == sha256(path)
        for track, item in checkpoint["linear_recurrent_states"].items():
            path = Path(item["path"])
            assert path.is_file(), (depth, track)
            assert item["sha256"] == sha256(path)
            assert len(item["state_digests"]) == 3
        for name, item in checkpoint["metrics"].items():
            assert item["aggregate"]["rows"] == 3, (depth, name)
            assert item["aggregate"]["nonfinite"] is False
            assert len(item["per_step"]) == 3
            assert math.isfinite(item["aggregate"]["relative_l2"])


def test_source_projection_is_from_each_track_live_normalized_input() -> None:
    report = load_report()
    for layer, layer_steps in report["steps"].items():
        for track, step in layer_steps.items():
            assert step["outputs"]["finite"] is True
            qkv_source = step["override"]["source_qkv_path"]
            z_source = step["override"]["source_z_path"]
            step_dir = ARTIFACT / "steps" / f"layer-{layer}" / track
            if qkv_source != "-":
                source = Path(qkv_source)
                metadata = json.loads(source.with_suffix(source.suffix + ".json").read_text())
                assert metadata["input_normed_sha256"] == step["input"]["input_normed_sha256"]
                assert sha256(step_dir / "qkv.f32le") == sha256(source)
            if z_source != "-":
                source = Path(z_source)
                metadata = json.loads(source.with_suffix(source.suffix + ".json").read_text())
                assert metadata["input_normed_sha256"] == step["input"]["input_normed_sha256"]
                assert sha256(step_dir / "z.f32le") == sha256(source)


def test_combined_depth_growth_and_interaction_are_observational() -> None:
    report = load_report()
    relative_l2 = []
    for depth in ("1", "2", "3"):
        checkpoint = report["checkpoints"][depth]
        relative_l2.append(checkpoint["metrics"]["combined_vs_baseline"]["aggregate"]["relative_l2"])
        interaction = checkpoint["interaction"]
        assert interaction["finite"] is True
        assert len(interaction["combined_delta_vs_additive_prediction"]["per_step"]) == 3
        assert math.isfinite(interaction["combined_delta_vs_additive_prediction"]["aggregate"]["relative_l2"])
    assert relative_l2[0] < relative_l2[1] < relative_l2[2]
    assert math.isclose(report["checkpoints"]["3"]["combined_amplification_vs_depth1"], relative_l2[2] / relative_l2[0])


def test_memory_policy_is_serial_and_bounded() -> None:
    memory = load_report()["memory"]
    assert memory["track_parallelism"] == 1
    assert memory["layer_parallelism"] == 1
    assert memory["torch_threads"] == 1
    assert memory["source_weight_row_chunk"] == 256
    assert 0 < memory["max_child_vm_hwm_kib"] < 2 * 1024 * 1024
    assert 0 < memory["python_ru_maxrss_kib"] < 2 * 1024 * 1024
