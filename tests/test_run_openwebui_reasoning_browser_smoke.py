from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/run-openwebui-reasoning-browser-smoke.py"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reasoning_browser_runner", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def evidence(model_id: str, switch_model_id: str) -> dict:
    primary = digest(model_id)
    switched = digest(switch_model_id)

    def request(model_hash: str, suffix: str) -> dict:
        return {
            "sha256": digest(f"request-{suffix}"),
            "utf8_bytes": 128,
            "model_id_sha256": model_hash,
            "has_reasoning_content_key": True,
            "assistant_has_reasoning_content": False,
        }

    return {
        "schema_version": "ullm.openwebui.reasoning_browser_smoke.v2",
        "model_id_sha256": primary,
        "first_answer": {"utf8_bytes": 20, "sha256": "c" * 64},
        "expanded_view": {"utf8_bytes": 40, "sha256": "f" * 64},
        "second_answer": {"utf8_bytes": 21, "sha256": "d" * 64},
        "provider_switch_performed": True,
        "provider_switch_model_id_sha256": switched,
        "provider_switch_answer": {"utf8_bytes": 22, "sha256": "3" * 64},
        "provider_return_performed": True,
        "provider_return_model_id_sha256": primary,
        "provider_return_answer": {"utf8_bytes": 23, "sha256": "6" * 64},
        "reasoning_details_expanded": True,
        "provider_request_count": 4,
        "provider_requests": [
            request(primary, "one"),
            request(primary, "two"),
            request(switched, "three"),
            request(primary, "four"),
        ],
        "hidden_reasoning_reinserted": False,
        "page_error_count": 0,
        "page_error_digests": [],
    }


class FakeProcess:
    def __init__(self, output, payload: bytes) -> None:
        self.output = output
        self.payload = payload

    def wait(self, *, timeout: float) -> int:
        del timeout
        self.output.write(self.payload)
        self.output.flush()
        return 0


def test_runner_publishes_valid_hash_only_evidence_and_binds_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = tmp_path / "token"
    token.write_text("secret-token\n", encoding="ascii")
    script = tmp_path / "smoke.cjs"
    script.write_text("console.log('{}')\n", encoding="ascii")
    output = tmp_path / "browser.json"
    model_id = "ullm-qwen3.5-9b-aq4"
    switch_model_id = "llama-qwen3.5-9b-ud-q4"
    payload = (json.dumps(evidence(model_id, switch_model_id)) + "\n").encode("ascii")
    commands: list[list[str]] = []

    def fake_popen(command, *, stdout, **kwargs):
        del kwargs
        commands.append(command)
        return FakeProcess(stdout, payload)

    monkeypatch.setattr(TOOL.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        TOOL,
        "_validate_manifest_identity",
        lambda _path, _model: {"manifest_sha256": "m" * 64},
    )
    result = TOOL.execute(
        output=output,
        manifest=tmp_path / "manifest.json",
        token_file=token,
        browser_image="sha256:" + "a" * 64,
        openwebui_url="http://127.0.0.1:3000/",
        model_id=model_id,
        model_name="uLLM Qwen3.5 9B AQ4",
        switch_model_id=switch_model_id,
        switch_model_name="llama.cpp Qwen3.5 9B UD-Q4_K_XL",
        browser_script=script,
    )

    assert result["provider_request_count"] == 4
    assert json.loads(output.read_text(encoding="ascii"))["model_id_sha256"] == digest(
        model_id
    )
    assert commands
    assert "--network=host" in commands[0]
    assert f"ULLM_MODEL_ID={model_id}" in commands[0]
    assert f"OPENWEBUI_SWITCH_MODEL_ID={switch_model_id}" in commands[0]


def test_runner_rejects_external_model_binding_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    token = tmp_path / "token"
    token.write_text("secret-token", encoding="ascii")
    script = tmp_path / "smoke.cjs"
    script.write_text("console.log('{}')\n", encoding="ascii")
    output = tmp_path / "browser.json"
    payload = (json.dumps(evidence("candidate", "switch")) + "\n").encode("ascii")

    def fake_popen(command, *, stdout, **kwargs):
        del command, kwargs
        return FakeProcess(stdout, payload)

    monkeypatch.setattr(TOOL.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        TOOL,
        "_validate_manifest_identity",
        lambda _path, _model: {"manifest_sha256": "m" * 64},
    )
    with pytest.raises(TOOL.SmokeError, match="primary model identity"):
        TOOL.execute(
            output=output,
            manifest=tmp_path / "manifest.json",
            token_file=token,
            browser_image="sha256:" + "a" * 64,
            openwebui_url="http://127.0.0.1:3000",
            model_id="different",
            model_name="candidate",
            switch_model_id="switch",
            switch_model_name="switch",
            browser_script=script,
        )
    assert not output.exists()


def test_runner_rejects_the_current_v1_active_manifest() -> None:
    manifest = Path("/etc/ullm/served-models/active.json")
    if not manifest.is_file():
        pytest.skip("WRX80 active manifest is unavailable")
    with pytest.raises(TOOL.SmokeError, match="not v2"):
        TOOL._validate_manifest_identity(manifest, "ullm-qwen3.5-9b-aq4")


@pytest.mark.parametrize("value", ["browser:latest", "sha256:ABC" + "a" * 61])
def test_runner_requires_immutable_browser_image(value: str) -> None:
    with pytest.raises(TOOL.SmokeError, match="immutable Docker"):
        TOOL._validate_image(value)


def test_alternating_r9700_coordinator_serializes_provider_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions: list[tuple[str, str]] = []

    def service(_systemctl: str, action: str, name: str) -> None:
        actions.append((action, name))

    def wait(_rocm: str, expected: set[str], timeout_seconds: float = 60.0) -> None:
        del timeout_seconds
        actions.append(("gpu", ",".join(sorted(expected))))

    monkeypatch.setattr(TOOL, "_service_command", service)
    monkeypatch.setattr(TOOL, "_wait_for_gpu_owner", wait)
    coordinator = TOOL._AlternatingServiceCoordinator(
        "systemctl", "rocm-smi", "ullm-openai.service", "llama-qwen35-udq4.service"
    )

    coordinator.transition("before-switch")
    assert coordinator.owner == "llama"
    coordinator.transition("before-return")
    assert coordinator.owner == "ullm"
    assert actions == [
        ("stop", "ullm-openai.service"),
        ("gpu", ""),
        ("start", "llama-qwen35-udq4.service"),
        ("gpu", "llama-server"),
        ("stop", "llama-qwen35-udq4.service"),
        ("gpu", ""),
        ("start", "ullm-openai.service"),
        ("gpu", "ullm-aq4-worker"),
    ]
