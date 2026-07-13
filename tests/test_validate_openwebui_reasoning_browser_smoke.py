from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/validate-openwebui-reasoning-browser-smoke.py"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reasoning_browser_validator", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def evidence() -> dict:
    request = {
        "sha256": "a" * 64,
        "utf8_bytes": 128,
        "has_reasoning_content_key": True,
        "assistant_has_reasoning_content": False,
    }
    return {
        "schema_version": TOOL.SCHEMA_VERSION,
        "model_id_sha256": "b" * 64,
        "first_answer": {"utf8_bytes": 20, "sha256": "c" * 64},
        "second_answer": {"utf8_bytes": 21, "sha256": "d" * 64},
        "reasoning_details_expanded": True,
        "provider_request_count": 2,
        "provider_requests": [request, {**request, "sha256": "e" * 64}],
        "hidden_reasoning_reinserted": False,
        "page_error_count": 0,
        "page_error_digests": [],
    }


def test_validator_accepts_hash_only_browser_gate(tmp_path: Path) -> None:
    path = tmp_path / "browser.json"
    path.write_text(json.dumps(evidence()), encoding="ascii")

    report = TOOL.validate(path)

    assert report["structurally_valid"] is True
    assert report["gate_eligible"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("reasoning_details_expanded", False),
        lambda value: value.__setitem__("response", "secret"),
        lambda value: value.__setitem__("page_error_count", 1),
    ],
)
def test_validator_rejects_unsafe_or_failed_browser_records(tmp_path: Path, mutation) -> None:
    value = evidence()
    mutation(value)
    path = tmp_path / "browser.json"
    path.write_text(json.dumps(value), encoding="ascii")

    with pytest.raises(TOOL.ValidationError):
        TOOL.validate(path)


def test_validator_reports_reinserted_reasoning_as_gate_failure(tmp_path: Path) -> None:
    value = evidence()
    value["provider_requests"][-1]["assistant_has_reasoning_content"] = True
    path = tmp_path / "browser.json"
    path.write_text(json.dumps(value), encoding="ascii")

    report = TOOL.validate(path)

    assert report["structurally_valid"] is True
    assert report["gate_eligible"] is False
    assert report["reasons"] == [
        "last provider request contains assistant reasoning_content"
    ]


def test_validator_require_pass_uses_distinct_exit_code(tmp_path: Path) -> None:
    value = evidence()
    value["provider_requests"][-1]["has_reasoning_content_key"] = False
    value["provider_requests"][0]["has_reasoning_content_key"] = False
    path = tmp_path / "browser.json"
    path.write_text(json.dumps(value), encoding="ascii")

    assert TOOL.main([str(path)]) == 0
    assert TOOL.main([str(path), "--require-pass"]) == 2
