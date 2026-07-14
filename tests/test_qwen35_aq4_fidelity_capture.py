from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("aq4_fidelity_capture", ROOT / "tools" / "capture-qwen35-aq4-fidelity.py")
assert SPEC and SPEC.loader
CAPTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAPTURE)
PREPARE = ROOT / "tools" / "prepare-qwen35-aq4-fidelity-cases.py"
SPLIT_ROOT = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-holdout-split-v0.1"


def test_stream_stats_are_bounded_and_reproducible() -> None:
    stats = CAPTURE._stream_stats(iter([[1.0, 2.0], [3.0]]), iter([[2.0, 2.0], [1.0]]), 3)
    assert stats["elements"] == 3
    assert stats["reference_norm_sq"] == pytest.approx(14.0)
    assert stats["candidate_norm_sq"] == pytest.approx(9.0)
    assert stats["delta_norm_sq"] == pytest.approx(5.0)
    assert stats["max_abs"] == pytest.approx(2.0)
    assert stats["cosine"] == pytest.approx(9.0 / math.sqrt(126.0))


def test_stream_stats_rejects_short_or_nonfinite_rows() -> None:
    with pytest.raises(CAPTURE.CaptureError, match="element count"):
        CAPTURE._stream_stats(iter([[1.0]]), iter([[1.0]]), 2)
    with pytest.raises(CAPTURE.CaptureError, match="non-finite"):
        CAPTURE._stream_stats(iter([[float("nan")]]), iter([[1.0]]), 1)


def test_output_is_noreplace(tmp_path: Path) -> None:
    output = tmp_path / "metrics.json"
    output.write_text("existing\n", encoding="ascii")
    with pytest.raises(CAPTURE.CaptureError, match="overwrite"):
        CAPTURE._atomic_json(output, {"status": "bad"})


def test_cpu_fixture_source_active_to_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Exercise the source/active sidecar adapter without loading a model or GPU."""
    monkeypatch.setattr(CAPTURE, "HIDDEN_SIZE", 2)
    monkeypatch.setattr(CAPTURE, "VOCAB_SIZE", 3)
    rows = []
    source_rows = {}
    active_rows = {}
    source_hidden = tmp_path / "source-hidden.f32le"
    source_logits = tmp_path / "source-logits.f32le"
    active_hidden = tmp_path / "active-hidden.f32le"
    active_logits = tmp_path / "active-logits.f32le"
    with source_hidden.open("wb") as sh, source_logits.open("wb") as sl, active_hidden.open("wb") as ah, active_logits.open("wb") as al:
        for index in range(24):
            case_id = f"case-{index}"
            digest = f"{index:064x}"
            row = {"case_id": case_id, "case_sha256": digest, "fixture_sha256": digest, "prompt_token_ids_sha256": digest, "context_token_ids_sha256": digest, "prompt_tokens": 2, "context_tokens": 2, "baseline_mode": "all_m1", "prefill_requested_m": 1, "resolved_m": 1, "step": 0, "row_count": 1}
            rows.append(row)
            sh.write(struct.pack("<2f", 1.0, 2.0)); sl.write(struct.pack("<3f", 3.0, 1.0, 0.0))
            ah.write(struct.pack("<2f", 1.0, 2.0)); al.write(struct.pack("<3f", 3.0, 1.0, 0.0))
            base = {"case_id": case_id, "step": 0, "input_token_ids_sha256": digest, "greedy_token_id": 0, "topk": [{"token_id": i, "logit": float(3 - i)} for i in range(10)], "hidden": {"offset_bytes": index * 8}, "logits": {"offset_bytes": index * 12}}
            source_rows[(case_id, 0)] = base
            active_rows[(case_id, 0)] = json.loads(json.dumps(base))
    identity = {"model_id": "fixture", "model_revision": "fixture", "tokenizer": {"aggregate_sha256": "a" * 64}}
    artifacts = {
        "independent_source_full": {"rows": source_rows, "hidden": source_hidden, "logits": source_logits, "chunk_elements": 2, "manifest_sha256": "b" * 64, "manifest": {"identity": identity}},
        "aq4_target": {"rows": active_rows, "hidden": active_hidden, "logits": active_logits, "chunk_elements": 2, "manifest_sha256": "c" * 64, "manifest": {"identity": identity}},
    }
    monkeypatch.setattr(CAPTURE, "_artifact", lambda _root, kind: artifacts[kind])
    monkeypatch.setattr(CAPTURE, "_load_split", lambda _root: ({}, {"status": "ready_for_calibration"}, rows, "d" * 64, "e" * 64, "f" * 64))
    result = CAPTURE.capture(tmp_path, Path("source"), Path("active"), tmp_path / "metrics.json")
    assert result["row_count"] == 24
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 24
    assert all(row["metrics"]["token_agreement_rate"] == 1.0 for row in payload["rows"])


def test_prepare_binds_latest_24_row_split(tmp_path: Path) -> None:
    cases = tmp_path / "cases.json"
    plan = tmp_path / "plan.json"
    result = subprocess.run(["python3", str(PREPARE), "--split-root", str(SPLIT_ROOT), "--output", str(cases), "--plan-output", str(plan)], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    value = json.loads(plan.read_text(encoding="utf-8"))
    assert value["row_count"] == 24
    assert value["split_manifest_sha256"] == "966878f3d9eb13f5b485825208f8072521724f308f5ee3d8a003b0b051198887"
    assert value["calibration_cases_sha256"] == "20c09f22bb1ca4dfac907de09febddb01ed0228c3f4a17c01efd646491e0983f"
    assert len(json.loads(cases.read_text(encoding="utf-8"))["cases"]) == 24
