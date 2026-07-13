from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/run-generic-reasoning-phase0-http-baseline.py"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "phase0_http_baseline_test_tool", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


class FakeTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        repetitions = len(messages[0]["content"].split())
        return {"input_ids": [0] * (12 + repetitions)}


def test_prompt_builder_hits_exact_targets() -> None:
    tokenizer = FakeTokenizer()
    for target in [18, 1024, 2048, 3072]:
        content, count = TOOL._prompt_for_target(tokenizer, target)
        assert count == target
        assert len(content.split()) == target - 12


def test_capture_publishes_hashes_and_usage_without_bodies(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = tmp_path / "active.json"
    worker = tmp_path / "worker"
    key = tmp_path / "key"
    worker.write_bytes(b"fixture-worker")
    key.write_text("secret-not-observed-by-test", encoding="ascii")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "ullm.served_model.v1",
                "public": {"id": "fixture-model"},
                "worker": {"protocol": "ullm.worker.v1", "binary": str(worker)},
                "promotion": {"source_commit": "old-commit"},
            }
        ),
        encoding="ascii",
    )
    monkeypatch.setattr(TOOL, "_load_tokenizer", lambda _root: FakeTokenizer())

    def fake_curl(body, *, stream, **_kwargs):
        request = json.loads(body)
        prompt_tokens = len(request["messages"][0]["content"].split()) + 12
        if stream:
            response = (
                b'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n'
                b'data: {"choices":[],"usage":{"prompt_tokens":18,"completion_tokens":2,"total_tokens":20}}\n'
                b"data: [DONE]\n"
            )
        else:
            response = json.dumps(
                {
                    "choices": [{"finish_reason": "length"}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": 2,
                        "total_tokens": prompt_tokens + 2,
                    },
                }
            ).encode("ascii")
        return 200, response

    monkeypatch.setattr(TOOL, "_docker_curl", fake_curl)
    output = tmp_path / "baseline.json"
    document = TOOL.capture(
        output,
        manifest=manifest,
        tokenizer_root=tmp_path / "tokenizer",
        key_file=key,
        image="fixture-image",
        network="fixture-network",
        endpoint="http://fixture/v1/chat/completions",
        timeout_seconds=1.0,
    )

    assert document["raw_bodies_stored"] is False
    assert [case["prompt_tokens"] for case in document["cases"]] == [
        18,
        1024,
        2048,
        3072,
    ]
    assert document["cases"][0]["stream"]["usage"] == {
        "prompt_tokens": 18,
        "completion_tokens": 2,
        "total_tokens": 20,
    }
    serialized = output.read_text(encoding="ascii")
    assert "secret-not-observed-by-test" not in serialized
    assert '"content"' not in serialized


def test_docker_probe_does_not_put_api_key_in_process_arguments(
    tmp_path: Path, monkeypatch
) -> None:
    key = tmp_path / "key"
    key.write_text("secret-not-observed-by-process-list", encoding="ascii")
    observed: dict[str, object] = {}

    def fake_run(command, **_kwargs):
        observed["command"] = command
        return type("Completed", (), {"returncode": 0, "stdout": b"{}200"})()

    monkeypatch.setattr(TOOL.subprocess, "run", fake_run)
    status, response = TOOL._docker_curl(
        b"{}",
        endpoint="http://fixture/v1/chat/completions",
        key_file=key,
        image="fixture-image",
        network="fixture-network",
        timeout_seconds=1.0,
        stream=False,
    )

    assert status == 200
    assert response == b"{}"
    command = observed["command"]
    assert isinstance(command, list)
    serialized = " ".join(str(part) for part in command)
    assert "secret-not-observed-by-process-list" not in serialized
    assert '--config "$config"' in serialized
    assert 'printf \'header = "Authorization: Bearer %s"' in serialized
    assert "--group-add" in command
    assert str(key.stat().st_gid) in command


def test_resident_token_evidence_is_reduced_to_safe_fields(tmp_path: Path) -> None:
    evidence = tmp_path / "resident.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "ullm.aq4_resident_promotion_evidence.v1",
                "verified": True,
                "production_receipt_written": False,
                "source_commit": "a" * 40,
                "worker_binary_sha256": "b" * 64,
                "resident": {
                    "cases": [
                        {
                            "id": "raw-p0001-g0004",
                            "prompt_token_ids": [1],
                            "tokens": [2, 3],
                            "outcome": "length",
                            "prompt_progress": [1],
                            "reset_complete": True,
                            "reasoning_usage": {
                                "reasoning_tokens": 0,
                                "forced_end_tokens": 0,
                            },
                        }
                    ]
                },
            }
        ),
        encoding="ascii",
    )

    reduced = TOOL._load_resident_token_evidence(
        evidence, source_commit="a" * 40, worker_binary_sha256="b" * 64
    )

    assert reduced["cases"][0]["generated_token_ids"] == [2, 3]
    assert "tokens" not in reduced["cases"][0]
    assert "response" not in json.dumps(reduced)
