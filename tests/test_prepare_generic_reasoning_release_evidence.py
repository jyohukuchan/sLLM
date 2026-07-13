from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/prepare-generic-reasoning-release-evidence.py"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generic_release_preparer", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def measured_case(case_id: str, mode: str, reasoning: int = 0) -> dict:
    answer = 2
    return {
        "id": case_id,
        "mode": mode,
        "prompt_fixture_id": f"fixture-{case_id}",
        "prompt_sha256": "a" * 64,
        "stream": True,
        "http_status": 200,
        "sse_chunk_count": 3,
        "finish_reason": "stop",
        "raw": {
            "prompt_tokens": 8,
            "completion_tokens": reasoning + answer,
            "reasoning_tokens": reasoning,
            "forced_end_tokens": 0,
            "answer_tokens": answer,
            "budget_overshoot": 0,
            "empty_answer": False,
            "usage_completion_tokens": reasoning + answer,
        },
        "timing": {
            "prefill_tokens_per_second": 100.0,
            "first_reasoning_token_ms": 10.0 if reasoning else None,
            "first_answer_token_ms": 20.0,
            "reasoning_decode_tokens_per_second": 50.0 if reasoning else None,
            "answer_decode_tokens_per_second": 80.0,
            "decode_tokens_per_second": 70.0,
            "latency_ms": 100.0,
        },
        "resource": {
            "rss_delta_bytes": 0,
            "vram_delta_bytes": 0,
            "gpu_temperature_c": 60.0,
            "power_w": 200.0,
        },
        "quality": {"correct": True, "score": 1.0},
    }


def cases() -> list[dict]:
    return [
        measured_case("disabled", "disabled"),
        measured_case("budget-32", "budget-32", 8),
        measured_case("budget-128", "budget-128", 20),
        measured_case("budget-256", "budget-256", 24),
        measured_case("unbounded", "unbounded", 30),
    ]


def lifecycle_events(measured: list[dict]) -> list[dict]:
    events = []
    for item in measured:
        raw = item["raw"]
        enabled = item["mode"] != "disabled"
        events.append(
            {
                "case_id": item["id"],
                "stream": item["stream"],
                "outcome": item["finish_reason"],
                "prompt_tokens": raw["prompt_tokens"],
                "completion_tokens": raw["completion_tokens"],
                "reset_complete": True,
                "reasoning_tokens": raw["reasoning_tokens"] if enabled else None,
                "forced_end_tokens": raw["forced_end_tokens"] if enabled else None,
                "admit_to_start_ns": 1,
                "start_to_release_ns": 2,
                "admit_to_release_ns": 3,
            }
        )
    return events


def write_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    fixture = ROOT / "services/openai-gateway/tests/fixtures/served-model/aq4"
    candidate = root / "served-model"
    shutil.copytree(fixture, candidate)
    manifest = candidate / "served-model.json"
    worker = candidate / "worker"
    cases_path = root / "cases.json"
    measured = cases()
    cases_path.write_text(json.dumps(measured), encoding="ascii")
    lifecycle_path = root / "lifecycle.json"
    lifecycle_path.write_text(
        json.dumps(
            {
                "schema_version": "ullm.generic_reasoning_lifecycle_evidence.v1",
                "events": lifecycle_events(measured),
            }
        ),
        encoding="ascii",
    )
    return cases_path, manifest, worker, lifecycle_path


def test_prepare_writes_valid_complete_hash_only_evidence(tmp_path: Path, monkeypatch) -> None:
    cases_path, manifest, worker, lifecycle_path = write_inputs(tmp_path)
    commit = "1" * 40
    monkeypatch.setattr(TOOL, "_git_commit", lambda: commit)
    monkeypatch.setattr(TOOL, "_git_status", lambda: b"")
    output = tmp_path / "release.json"

    document = TOOL.prepare(
        cases_path,
        manifest,
        worker,
        "ullm/open-webui@sha256:" + "b" * 64,
        commit,
        output,
        lifecycle_path=lifecycle_path,
        status="complete",
    )

    assert json.loads(output.read_text(encoding="ascii")) == document
    assert document["git_worktree_clean"] is True
    assert document["git_worktree_status_sha256"] == hashlib.sha256(b"").hexdigest()
    assert TOOL._load_validator().validate(output)["gate_eligible"] is True


def test_prepare_keeps_dirty_incomplete_evidence_but_rejects_complete(tmp_path: Path, monkeypatch) -> None:
    cases_path, manifest, worker, lifecycle_path = write_inputs(tmp_path)
    commit = "1" * 40
    monkeypatch.setattr(TOOL, "_git_commit", lambda: commit)
    monkeypatch.setattr(TOOL, "_git_status", lambda: b" M source.py\n")

    incomplete = tmp_path / "incomplete.json"
    document = TOOL.prepare(
        cases_path,
        manifest,
        worker,
        "ullm/open-webui@sha256:" + "b" * 64,
        commit,
        incomplete,
        lifecycle_path=lifecycle_path,
    )
    assert document["git_worktree_clean"] is False
    assert TOOL._load_validator().validate(incomplete)["gate_eligible"] is False

    with pytest.raises(TOOL.EvidenceError, match="clean Git worktree"):
        TOOL.prepare(
            cases_path,
            manifest,
            worker,
            "ullm/open-webui@sha256:" + "b" * 64,
            commit,
            tmp_path / "complete.json",
            lifecycle_path=lifecycle_path,
            status="complete",
        )


def test_prepare_complete_rejects_unaligned_active_promotion(tmp_path: Path, monkeypatch) -> None:
    cases_path, manifest, worker, lifecycle_path = write_inputs(tmp_path)
    monkeypatch.setattr(TOOL, "_git_commit", lambda: "1" * 40)
    monkeypatch.setattr(TOOL, "_git_status", lambda: b"")

    with pytest.raises(TOOL.EvidenceError, match="not production-gate eligible"):
        TOOL.prepare(
            cases_path,
            manifest,
            worker,
            "ullm/open-webui@sha256:" + "b" * 64,
            "2" * 40,
            tmp_path / "unaligned.json",
            lifecycle_path=lifecycle_path,
            status="complete",
        )


def test_prepare_rejects_cleartext_case_fields(tmp_path: Path, monkeypatch) -> None:
    cases_path, manifest, worker, lifecycle_path = write_inputs(tmp_path)
    value = json.loads(cases_path.read_text(encoding="ascii"))
    value[0]["response"] = "secret"
    cases_path.write_text(json.dumps(value), encoding="ascii")
    monkeypatch.setattr(TOOL, "_git_commit", lambda: "1" * 40)
    monkeypatch.setattr(TOOL, "_git_status", lambda: b"")

    with pytest.raises(TOOL.EvidenceError, match="forbidden field"):
        TOOL.prepare(
            cases_path,
            manifest,
            worker,
            "ullm/open-webui@sha256:" + "b" * 64,
            "1" * 40,
            tmp_path / "release.json",
            lifecycle_path=lifecycle_path,
        )


def test_prepare_rejects_invalid_served_model_manifest(tmp_path: Path, monkeypatch) -> None:
    cases_path, manifest, worker, lifecycle_path = write_inputs(tmp_path)
    document = json.loads(manifest.read_text(encoding="ascii"))
    document["worker"]["protocol"] = "ullm.worker.v2"
    manifest.write_text(json.dumps(document), encoding="ascii")
    monkeypatch.setattr(TOOL, "_git_commit", lambda: "1" * 40)
    monkeypatch.setattr(TOOL, "_git_status", lambda: b"")

    with pytest.raises(TOOL.EvidenceError, match="served-model manifest failed validation"):
        TOOL.prepare(
            cases_path,
            manifest,
            worker,
            "ullm/open-webui@sha256:" + "b" * 64,
            "1" * 40,
            tmp_path / "release.json",
            lifecycle_path=lifecycle_path,
        )
