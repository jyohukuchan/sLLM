from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/run-aq4-resident-promotion-evidence.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_module("test_run_aq4_resident_promotion_evidence_tool", TOOL_PATH)


FAKE_WORKER = r'''#!/usr/bin/env python3
import json
import sys

print(json.dumps({
    "schema_version": "ullm.worker.v1",
    "type": "ready",
    "model": "fixture-aq4",
    "model_revision": "fixture-resident",
    "artifact_content_sha256": "a" * 64,
    "package_manifest_sha256": "a" * 64,
    "device": "cpu-test",
    "execution_profile": "fixture",
    "context_length": 32,
    "max_new_tokens": 4,
}), flush=True)
for line in sys.stdin:
    command = json.loads(line)
    if command["type"] == "shutdown":
        break
    request_id = command["request_id"]
    prompt = command["prompt_token_ids"]
    print(json.dumps({
        "schema_version": "ullm.worker.v1", "type": "started",
        "request_id": request_id, "prompt_tokens": len(prompt),
    }), flush=True)
    for processed in range(1, len(prompt) + 1):
        print(json.dumps({
            "schema_version": "ullm.worker.v1", "type": "progress",
            "request_id": request_id, "phase": "prompt",
            "processed_prompt_tokens": processed,
        }), flush=True)
    tokens = [prompt[-1] + offset for offset in range(1, command["max_new_tokens"] + 1)]
    for index, token in enumerate(tokens):
        print(json.dumps({
            "schema_version": "ullm.worker.v1", "type": "token",
            "request_id": request_id, "index": index, "token_id": token,
        }), flush=True)
    print(json.dumps({
        "schema_version": "ullm.worker.v1", "type": "released",
        "request_id": request_id, "outcome": "length",
        "prompt_tokens": len(prompt), "completion_tokens": len(tokens),
        "timings": {
            "cache_n": 0, "prompt_n": len(prompt), "prompt_ms": 1.0,
            "prompt_per_token_ms": 1.0 / len(prompt),
            "prompt_per_second": float(len(prompt) * 1000),
            "predicted_n": len(tokens), "predicted_ms": 2.0,
            "predicted_per_token_ms": 0.5,
            "predicted_per_second": 2000.0,
        },
        "reset_complete": True,
    }), flush=True)
'''


def write_fixture_profile(root: Path, worker: Path) -> tuple[Path, Path]:
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text('{"fixture":true}\n', encoding="ascii")
    (tokenizer / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}) + "\n", encoding="ascii"
    )

    product = root / "product"
    package = product / "package"
    package.mkdir(parents=True)
    (package / "manifest.json").write_text('{"package":"fixture"}\n', encoding="ascii")
    production_receipt = product / "promotion.json"
    profile = {
        "schema_version": "ullm.served_model.profile.v1",
        "public": {
            "id": "fixture-aq4",
            "name": "Fixture AQ4",
            "description": "Fixture resident worker.",
            "upstream_id": "fixture/Qwen3.5-9B",
            "revision": "fixture-resident",
            "context_length": 32,
        },
        "generation": {
            "max_completion_tokens": 4,
            "vocab_size": 100,
            "eos_token_ids": [98, 99],
            "sampling": {"top_k": 1, "temperature": False, "top_p": False},
        },
        "format": {"format_id": "AQ4_0", "implementation_id": "fixture-aq4-v1"},
        "tokenizer": {
            "root": os.fspath(tokenizer),
            "transformers_version": "5.12.1",
            "class": "Qwen2Tokenizer",
            "files": ["tokenizer.json", "tokenizer_config.json"],
            "template_options": {"add_generation_prompt": True, "enable_thinking": False},
        },
        "worker": {
            "protocol": "ullm.worker.v1",
            "binary": os.fspath(worker),
            "arguments": ["--served-model-manifest", "{manifest}"],
            "required_environment": [],
            "identity": {"device": "gfx1201", "execution_profile": "rdna4_aq4_resident"},
        },
        "product": {
            "root": os.fspath(product),
            "artifact": None,
            "package": {"manifest_path": "package/manifest.json"},
        },
        "promotion": {
            "receipt": os.fspath(production_receipt),
            "source_commit_from_receipt": ["source_commit"],
        },
    }
    path = root / "profile.json"
    path.write_text(json.dumps(profile) + "\n", encoding="ascii")
    return path, production_receipt


def test_evidence_uses_ephemeral_bundle_and_sequential_processes(tmp_path: Path) -> None:
    worker = tmp_path / "fake-worker"
    worker.write_text(FAKE_WORKER, encoding="utf-8")
    worker.chmod(0o755)
    profile, production_receipt = write_fixture_profile(tmp_path, worker)
    output = tmp_path / "evidence.json"

    document = TOOL.run_evidence(
        profile,
        output,
        worker,
        worker,
        ready_timeout_seconds=5.0,
        request_timeout_seconds=5.0,
        source_commit="fixture-commit",
    )

    assert output.is_file()
    assert json.loads(output.read_text(encoding="ascii")) == document
    assert not production_receipt.exists()
    assert document["production_receipt_written"] is False
    assert document["resident"]["clean_shutdown"] is True
    assert document["legacy"]["clean_shutdown"] is True
    assert document["resident"]["pid"] != document["legacy"]["pid"]
    assert all(item["tokens_exact_match"] for item in document["comparisons"])
    assert [case["id"] for case in document["resident"]["cases"]] == [
        "raw-p0001-g0004",
        "raw-p0008-g0004",
    ]
    assert all(case["reset_complete"] for case in document["resident"]["cases"])
    assert all(case["timings"] for case in document["legacy"]["cases"])
    assert all(
        check["sibling_engine_count"] == 0
        for check in document["resident"]["child_process_checks"]
    )


def test_atomic_output_refuses_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("unchanged", encoding="ascii")
    output = tmp_path / "evidence.json"
    output.symlink_to(target)

    with pytest.raises(TOOL.EvidenceError, match="must not be a symlink"):
        TOOL._atomic_write_json(output, {"verified": True})

    assert target.read_text(encoding="ascii") == "unchanged"
