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
        "model_id_sha256": "b" * 64,
        "has_reasoning_content_key": True,
        "assistant_has_reasoning_content": False,
    }
    return {
        "schema_version": TOOL.SCHEMA_VERSION,
        "model_id_sha256": "b" * 64,
        "first_answer": {"utf8_bytes": 20, "sha256": "c" * 64},
        "expanded_view": {"utf8_bytes": 40, "sha256": "f" * 64},
        "second_answer": {"utf8_bytes": 21, "sha256": "d" * 64},
        "provider_switch_performed": True,
        "provider_switch_model_id_sha256": "2" * 64,
        "provider_switch_answer": {"utf8_bytes": 22, "sha256": "3" * 64},
        "provider_return_performed": True,
        "provider_return_model_id_sha256": "b" * 64,
        "provider_return_answer": {"utf8_bytes": 23, "sha256": "6" * 64},
        "reasoning_details_expanded": True,
        "provider_request_count": 4,
        "provider_requests": [
            request,
            {**request, "sha256": "e" * 64},
            {**request, "sha256": "4" * 64, "model_id_sha256": "2" * 64},
            {**request, "sha256": "5" * 64, "model_id_sha256": "b" * 64},
        ],
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
    value["provider_requests"][-1]["assistant_has_reasoning_content"] = True
    path = tmp_path / "browser.json"
    path.write_text(json.dumps(value), encoding="ascii")

    assert TOOL.main([str(path)]) == 0
    assert TOOL.main([str(path), "--require-pass"]) == 2


def test_validator_rejects_provider_switch_model_mismatch(tmp_path: Path) -> None:
    value = evidence()
    value["provider_requests"][-2]["model_id_sha256"] = "4" * 64
    path = tmp_path / "browser.json"
    path.write_text(json.dumps(value), encoding="ascii")

    with pytest.raises(TOOL.ValidationError, match="provider switch request model"):
        TOOL.validate(path)


def test_validator_rejects_initial_model_mismatch(tmp_path: Path) -> None:
    value = evidence()
    value["provider_requests"][0]["model_id_sha256"] = "4" * 64
    path = tmp_path / "browser.json"
    path.write_text(json.dumps(value), encoding="ascii")

    with pytest.raises(TOOL.ValidationError, match="initial provider request model"):
        TOOL.validate(path)


def test_validator_rejects_non_switching_provider(tmp_path: Path) -> None:
    value = evidence()
    value["provider_switch_model_id_sha256"] = value["model_id_sha256"]
    value["provider_requests"][-2]["model_id_sha256"] = value["model_id_sha256"]
    path = tmp_path / "browser.json"
    path.write_text(json.dumps(value), encoding="ascii")

    with pytest.raises(TOOL.ValidationError, match="provider switch model is not distinct"):
        TOOL.validate(path)


def test_validator_reads_v1_hash_only_record(tmp_path: Path) -> None:
    value = evidence()
    value["schema_version"] = TOOL.SCHEMA_VERSION_V1
    value.pop("provider_switch_performed")
    value.pop("provider_switch_model_id_sha256")
    value.pop("provider_switch_answer")
    value.pop("provider_return_performed")
    value.pop("provider_return_model_id_sha256")
    value.pop("provider_return_answer")
    for request in value["provider_requests"]:
        request.pop("model_id_sha256")
    value["provider_request_count"] = 2
    value["provider_requests"] = value["provider_requests"][:2]
    path = tmp_path / "browser-v1.json"
    path.write_text(json.dumps(value), encoding="ascii")

    report = TOOL.validate(path)

    assert report["input_schema_version"] == TOOL.SCHEMA_VERSION_V1
    assert report["gate_eligible"] is True
