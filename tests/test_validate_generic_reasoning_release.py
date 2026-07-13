from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/validate-generic-reasoning-release.py"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generic_reasoning_release_validator", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def case(case_id: str, mode: str, *, reasoning: int = 0, forced: int = 0) -> dict:
    answer = 2
    return {
        "id": case_id,
        "mode": mode,
        "prompt_fixture_id": f"fixture-{case_id}",
        "prompt_sha256": "a" * 64,
        "stream": True,
        "http_status": 200,
        "sse_chunk_count": 8,
        "finish_reason": "stop",
        "raw": {
            "prompt_tokens": 64,
            "completion_tokens": reasoning + forced + answer,
            "reasoning_tokens": reasoning,
            "forced_end_tokens": forced,
            "answer_tokens": answer,
            "budget_overshoot": 0,
            "empty_answer": False,
            "usage_completion_tokens": reasoning + forced + answer,
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


def evidence() -> dict:
    cases = [
        case("disabled", "disabled"),
        case("budget-32", "budget-32", reasoning=8, forced=1),
        case("budget-128", "budget-128", reasoning=20, forced=1),
        case("budget-256", "budget-256", reasoning=24, forced=1),
        case("unbounded", "unbounded", reasoning=30, forced=1),
    ]
    return {
        "schema_version": TOOL.SCHEMA_VERSION,
        "status": "incomplete",
        "production_activation_performed": False,
        "source_commit": "1" * 40,
        "active_promotion_source_commit": "2" * 40,
        "source_commit_aligned": False,
        "git_worktree_clean": True,
        "git_worktree_status_sha256": "f" * 64,
        "identity": {
            "manifest_sha256": "b" * 64,
            "worker_binary_sha256": "c" * 64,
            "tokenizer_sha256": "d" * 64,
            "openwebui_image": "ullm/open-webui@sha256:" + "e" * 64,
        },
        "cases": cases,
    }


def test_validator_accepts_structure_but_not_incomplete_production_gate(tmp_path: Path) -> None:
    path = tmp_path / "release.json"
    path.write_text(json.dumps(evidence()), encoding="ascii")

    report = TOOL.validate(path)

    assert report["structurally_valid"] is True
    assert report["gate_eligible"] is False
    assert report["case_count"] == 5
    assert set(report["observed_modes"]) == TOOL.REQUIRED_MODES
    assert report["timing_percentiles"]["budget-128"]["latency_ms"] == {
        "count": 1,
        "p50": 100.0,
        "p95": 100.0,
        "p99": 100.0,
    }
    assert report["quality_summary"]["budget-128"] == {
        "total": 1,
        "correct": 1,
        "accuracy": 1.0,
    }
    assert report["resource_percentiles"]["budget-128"]["vram_delta_bytes"] == {
        "count": 1,
        "p50": 0.0,
        "p95": 0.0,
        "p99": 0.0,
        "maximum": 0.0,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["cases"][1]["raw"].__setitem__("budget_overshoot", 1),
        lambda value: value["cases"][2]["raw"].__setitem__("usage_completion_tokens", 1),
        lambda value: value["cases"][0].__setitem__("response", "secret"),
        lambda value: value["identity"].__setitem__("openwebui_image", "latest"),
        lambda value: value["identity"].__setitem__("openwebui_image", "@sha256:" + "e" * 64),
        lambda value: value.__setitem__("source_commit", "head"),
        lambda value: value.__setitem__("source_commit_aligned", True),
        lambda value: value.__setitem__("git_worktree_status_sha256", "dirty"),
        lambda value: value["cases"][0].__setitem__("sse_chunk_count", 0),
    ],
)
def test_validator_rejects_invalid_release_records(tmp_path: Path, mutation) -> None:
    value = evidence()
    mutation(value)
    path = tmp_path / "release.json"
    path.write_text(json.dumps(value), encoding="ascii")

    with pytest.raises(TOOL.ValidationError):
        TOOL.validate(path)


def test_validator_reports_missing_required_mode_as_incomplete_gate(tmp_path: Path) -> None:
    value = evidence()
    value["cases"].pop()
    path = tmp_path / "release.json"
    path.write_text(json.dumps(value), encoding="ascii")

    report = TOOL.validate(path)

    assert report["structurally_valid"] is True
    assert report["gate_eligible"] is False
    assert "required benchmark modes are missing: unbounded" in report["reasons"]


def test_validator_reports_quality_and_timing_gate_failures(tmp_path: Path) -> None:
    value = evidence()
    value["status"] = "complete"
    value["source_commit"] = value["active_promotion_source_commit"]
    value["source_commit_aligned"] = True
    value["cases"][1]["quality"]["correct"] = False
    value["cases"][2]["timing"]["latency_ms"] = None
    path = tmp_path / "release.json"
    path.write_text(json.dumps(value), encoding="ascii")

    report = TOOL.validate(path)

    assert report["structurally_valid"] is True
    assert report["gate_eligible"] is False
    assert "case quality is incorrect: budget-32" in report["reasons"]
    assert any("case timing is incomplete: budget-128" in reason for reason in report["reasons"])


def test_validator_reports_dirty_git_worktree_as_gate_failure(tmp_path: Path) -> None:
    value = evidence()
    value["git_worktree_clean"] = False
    path = tmp_path / "release.json"
    path.write_text(json.dumps(value), encoding="ascii")

    report = TOOL.validate(path)

    assert report["structurally_valid"] is True
    assert report["gate_eligible"] is False
    assert "Git worktree is not clean" in report["reasons"]


def test_validator_recomputes_percentiles_over_raw_cases(tmp_path: Path) -> None:
    value = evidence()
    for index, latency in enumerate((200.0, 300.0), start=2):
        extra = case(f"budget-128-{index}", "budget-128", reasoning=20, forced=1)
        extra["timing"]["latency_ms"] = latency
        value["cases"].append(extra)
    path = tmp_path / "release.json"
    path.write_text(json.dumps(value), encoding="ascii")

    report = TOOL.validate(path)

    assert report["timing_percentiles"]["budget-128"]["latency_ms"] == {
        "count": 3,
        "p50": 200.0,
        "p95": 290.0,
        "p99": 298.0,
    }
    assert report["quality_summary"]["budget-128"]["total"] == 3


def test_validator_rejects_oversized_evidence_and_sse_count(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"{" + b" " * TOOL.MAX_EVIDENCE_BYTES + b"}")
    with pytest.raises(TOOL.ValidationError, match="size bound"):
        TOOL.validate(oversized)

    value = evidence()
    value["cases"][0]["sse_chunk_count"] = TOOL.MAX_SSE_CHUNKS + 1
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(value), encoding="ascii")
    with pytest.raises(TOOL.ValidationError, match="chunk count"):
        TOOL.validate(path)


def test_validator_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":"x","schema_version":"y"}', encoding="ascii")

    with pytest.raises(TOOL.ValidationError, match="duplicate"):
        TOOL.validate(path)


def test_require_complete_returns_distinct_exit_code(tmp_path: Path) -> None:
    path = tmp_path / "release.json"
    path.write_text(json.dumps(evidence()), encoding="ascii")

    assert TOOL.main([str(path)]) == 0
    assert TOOL.main([str(path), "--require-complete"]) == 2
