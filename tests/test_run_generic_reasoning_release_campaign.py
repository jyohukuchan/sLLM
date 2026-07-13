from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/run-generic-reasoning-release-campaign.py"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generic_reasoning_release_campaign", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def test_modes_and_request_body_are_explicit_and_bounded() -> None:
    fixture = TOOL.Fixture("fixture", "hello", "ok")

    assert TOOL._mode_fields("disabled") == {"reasoning_effort": "none"}
    assert TOOL._mode_fields("budget-32") == {"thinking_budget_tokens": 32}
    assert TOOL._mode_fields("budget-128") == {"thinking_budget_tokens": 128}
    assert TOOL._mode_fields("budget-256") == {"thinking_budget_tokens": 256}
    assert TOOL._mode_fields("unbounded") == {"thinking_budget_tokens": -1}

    body = json.loads(TOOL._request_body("model", "budget-128", fixture))
    assert body["model"] == "model"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["max_completion_tokens"] == 512
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["thinking_budget_tokens"] == 128

    with pytest.raises(TOOL.CampaignError, match="unknown release mode"):
        TOOL._mode_fields("invalid")


def test_manifest_preflight_rejects_v1_before_external_validation(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": "ullm.served_model.v1"}), encoding="ascii")

    with pytest.raises(TOOL.CampaignError, match="not v2"):
        TOOL._validate_manifest(manifest)


def test_immutable_http_image_is_required(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_bytes(b"opaque-token")

    with pytest.raises(TOOL.CampaignError, match="immutable Docker"):
        TOOL.execute(
            output_dir=tmp_path / "out",
            manifest=tmp_path / "missing-manifest",
            fixture_suite=TOOL.DEFAULT_FIXTURES,
            token_file=token,
            http_image="curl:latest",
        )


def test_gpu_process_identity_is_bound_to_manifest_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    preflight = {
        "positive_vram_processes": [{"pid": "123", "process": "ullm-aq4-worker"}]
    }
    monkeypatch.setattr(TOOL, "_hash_process_executable", lambda _pid: "a" * 64)

    TOOL._bind_gpu_processes(preflight, "a" * 64)
    assert preflight["positive_vram_processes"][0]["binary_sha256"] == "a" * 64

    monkeypatch.setattr(TOOL, "_hash_process_executable", lambda _pid: "b" * 64)
    with pytest.raises(TOOL.CampaignError, match="differs from the v2 manifest"):
        TOOL._bind_gpu_processes(preflight, "a" * 64)


def test_lifecycle_observer_correlates_release_and_removes_socket(tmp_path: Path) -> None:
    path = tmp_path / "observer.sock"
    observer = TOOL.LifecycleObserver(path)
    observer.open()
    sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

    def send() -> None:
        sender.sendto(
            json.dumps(
                {
                    "schema_version": TOOL.LIFECYCLE_SCHEMA,
                    "event": "request_released",
                    "completion_id": "matching",
                }
            ).encode("ascii"),
            str(path),
        )

    thread = threading.Thread(target=send)
    thread.start()
    assert observer.wait_release("matching", 2.0)["completion_id"] == "matching"
    thread.join()
    sender.close()
    observer.close()
    assert not path.exists()
