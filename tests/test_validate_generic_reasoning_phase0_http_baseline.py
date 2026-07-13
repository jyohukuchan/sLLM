from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/validate-generic-reasoning-phase0-http-baseline.py"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_generic_reasoning_phase0_http_baseline_test_tool", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def baseline() -> dict:
    cases = []
    for target in TOOL.TARGETS:
        case = {
            "id": f"phase0-v1-target-{target}",
            "target_prompt_tokens": target,
            "prompt_sha256": "a" * 64,
            "prompt_tokens": target,
            "request_body_sha256": "b" * 64,
            "request_max_tokens": 2,
            "nonstream": {
                "http_status": 200,
                "response_body_sha256": "c" * 64,
                "response_bytes": 128,
                "prompt_tokens": target,
                "completion_tokens": 2,
                "total_tokens": target + 2,
            },
        }
        cases.append(case)
    cases[0]["stream"] = {
        "http_status": 200,
        "request_body_sha256": "d" * 64,
        "response_body_sha256": "e" * 64,
        "chunks": 4,
        "event_sequence": ["role", "token", "stop", "usage", "done"],
        "delta_keys": [["role"], ["content"]],
        "usage": {"prompt_tokens": 18, "completion_tokens": 2, "total_tokens": 20},
        "invalid_data_lines": 0,
    }
    return {
        "schema_version": TOOL.SCHEMA_VERSION,
        "status": "partial",
        "production_activation_performed": False,
        "source_commit": "head",
        "active_promotion_source_commit": "old",
        "source_commit_aligned": False,
        "active_manifest": {"sha256": "f" * 64},
        "worker": {"protocol": "ullm.worker.v1"},
        "endpoint": "http://fixture/v1/chat/completions",
        "image": "fixture-image",
        "cases": cases,
        "raw_bodies_stored": False,
        "missing": ["token IDs"],
    }


def test_validator_accepts_structure_but_keeps_gate_incomplete(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline()), encoding="ascii")

    report = TOOL.validate(path)

    assert report["structurally_valid"] is True
    assert report["gate_eligible"] is False
    assert report["case_count"] == 4


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["cases"][0]["nonstream"].__setitem__("total_tokens", 1),
        lambda value: value["cases"][0]["stream"].__setitem__("invalid_data_lines", 1),
        lambda value: value.__setitem__("raw_bodies_stored", True),
        lambda value: value["cases"].pop(),
    ],
)
def test_validator_rejects_inconsistent_or_unsafe_records(tmp_path: Path, mutation) -> None:
    value = baseline()
    mutation(value)
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(value), encoding="ascii")

    with pytest.raises(TOOL.ValidationError):
        TOOL.validate(path)


def test_require_complete_returns_distinct_exit_code_for_partial_gate(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline()), encoding="ascii")

    assert TOOL.main([str(path)]) == 0
    assert TOOL.main([str(path), "--require-complete"]) == 2
