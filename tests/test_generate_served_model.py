from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/generate-served-model.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GENERATOR = load_module("test_generate_served_model_tool", TOOL_PATH)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_profile(root: Path, *, receipt_exists: bool = True) -> Path:
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    tokenizer_json = b'{"model":"fixture"}\n'
    tokenizer_config = {
        "chat_template": "{{ messages }}",
        "tokenizer_class": "Qwen2Tokenizer",
    }
    (tokenizer / "tokenizer.json").write_bytes(tokenizer_json)
    (tokenizer / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config), encoding="utf-8"
    )

    worker = root / "worker"
    worker.write_bytes(b"#!/bin/sh\nexit 0\n")
    worker.chmod(0o755)

    product = root / "product"
    (product / "artifact").mkdir(parents=True)
    (product / "package").mkdir()
    artifact_manifest = b'{"artifact":true}\n'
    package_manifest = b'{"package":true}\n'
    (product / "artifact/sq_manifest.json").write_bytes(artifact_manifest)
    (product / "package/manifest.json").write_bytes(package_manifest)
    receipt = root / "promotion.json"
    if receipt_exists:
        receipt.write_text(
            json.dumps(
                {
                    "plan_commit": "abc1234",
                    "artifact": {"content_sha256": "1" * 64},
                }
            ),
            encoding="utf-8",
        )

    profile = {
        "schema_version": "ullm.served_model.profile.v1",
        "public": {
            "id": "fixture-model",
            "name": "Fixture model",
            "description": "Generator fixture.",
            "upstream_id": "fixture/upstream",
            "revision": "fixture-revision",
            "context_length": 128,
        },
        "generation": {
            "max_completion_tokens": 16,
            "vocab_size": 100,
            "eos_token_ids": [2],
            "sampling": {"top_k": 1, "temperature": False, "top_p": False},
        },
        "format": {"format_id": "SQ8_0", "implementation_id": "fixture-v1"},
        "tokenizer": {
            "root": os.fspath(tokenizer),
            "transformers_version": "5.12.1",
            "class": "Qwen2Tokenizer",
            "files": ["tokenizer.json", "tokenizer_config.json"],
            "template_options": {
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        },
        "worker": {
            "protocol": "ullm.worker.v1",
            "binary": os.fspath(worker),
            "arguments": ["--served-model-manifest", "{manifest}"],
            "required_environment": [],
            "identity": {"device": "cpu", "execution_profile": "fixture"},
        },
        "product": {
            "root": os.fspath(product),
            "artifact": {
                "manifest_path": "artifact/sq_manifest.json",
                "content_sha256_from_receipt": ["artifact", "content_sha256"],
            },
            "package": {"manifest_path": "package/manifest.json"},
        },
        "promotion": {
            "receipt": os.fspath(receipt),
            "source_commit_from_receipt": ["plan_commit"],
        },
    }
    path = root / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path


def test_generate_hashes_live_files_and_passes_strict_loader(tmp_path: Path) -> None:
    profile = write_profile(tmp_path)
    output = tmp_path / "served-model.json"

    digest = GENERATOR.generate(profile, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    loaded = GENERATOR._load_validator().load_served_model(output)

    assert digest == sha256(output.read_bytes()) == loaded.manifest_sha256
    assert document["worker"]["binary_sha256"] == sha256(
        (tmp_path / "worker").read_bytes()
    )
    assert document["tokenizer"]["chat_template_sha256"] == sha256(b"{{ messages }}")
    assert document["product"]["artifact"]["content_sha256"] == "1" * 64
    assert document["promotion"]["source_commit"] == "abc1234"
    assert output.stat().st_mode & 0o777 == 0o644


def test_missing_promotion_receipt_fails_without_output(tmp_path: Path) -> None:
    profile = write_profile(tmp_path, receipt_exists=False)
    output = tmp_path / "served-model.json"

    with pytest.raises(GENERATOR.GenerationError, match="promotion receipt"):
        GENERATOR.generate(profile, output)

    assert not output.exists()


def test_refuses_symlink_output(tmp_path: Path) -> None:
    profile = write_profile(tmp_path)
    target = tmp_path / "target.json"
    target.write_text("unchanged", encoding="utf-8")
    output = tmp_path / "served-model.json"
    output.symlink_to(target)

    with pytest.raises(GENERATOR.GenerationError, match="must not be a symlink"):
        GENERATOR.generate(profile, output)

    assert target.read_text(encoding="utf-8") == "unchanged"
