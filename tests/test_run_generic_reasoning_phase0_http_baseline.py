from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/run-generic-reasoning-phase0-http-baseline.py"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase0_http_baseline_test_tool", TOOL_PATH)
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


def test_capture_publishes_hashes_and_usage_without_bodies(tmp_path: Path, monkeypatch) -> None:
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
    assert [case["prompt_tokens"] for case in document["cases"]] == [18, 1024, 2048, 3072]
    assert document["cases"][0]["stream"]["usage"] == {
        "prompt_tokens": 18,
        "completion_tokens": 2,
        "total_tokens": 20,
    }
    serialized = output.read_text(encoding="ascii")
    assert "secret-not-observed-by-test" not in serialized
    assert '"content"' not in serialized
