from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/assemble-gemma4-e2b-serving-package.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ASSEMBLER = load_module("test_gemma4_e2b_assembler", TOOL_PATH)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_source(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma4ForConditionalGeneration"],
                "model_type": "gemma4",
                "text_config": {"model_type": "gemma4_text", "vocab_size": 262144},
            }
        ),
        encoding="utf-8",
    )
    (source / "model.safetensors").write_bytes(b"fixture-bf16-tensors")
    (source / "tokenizer.json").write_bytes(b'{"model":"fixture"}\n')
    (source / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "GemmaTokenizer", "add_bos_token": True}),
        encoding="utf-8",
    )
    return source


def test_assembly_binds_model_and_explicit_template_overlay(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    template = tmp_path / "chat_template.jinja"
    template.write_text("{{ messages }}", encoding="utf-8")
    destination = tmp_path / "product"

    result = ASSEMBLER.assemble(
        source_model_dir=source,
        chat_template=template,
        chat_template_revision="a" * 40,
        destination=destination,
    )

    package = json.loads((destination / "package/manifest.json").read_text())
    config = json.loads((destination / "tokenizer/tokenizer_config.json").read_text())
    provenance = json.loads((destination / "tokenizer/provenance.json").read_text())
    assert package["format_id"] == "BF16_0"
    assert package["architecture"]["architectures"] == ["Gemma4ForConditionalGeneration"]
    assert package["model"]["files"]["model.safetensors"]["sha256"] == sha256(
        b"fixture-bf16-tensors"
    )
    assert config["chat_template"] == "{{ messages }}"
    assert provenance["base"]["upstream_id"] == "google/gemma-4-E2B"
    assert provenance["chat_template"] == {
        "upstream_id": "google/gemma-4-E2B-it",
        "revision": "a" * 40,
        "sha256": sha256(b"{{ messages }}"),
    }
    assert result["tokenizer"]["chat_template_sha256"] == sha256(b"{{ messages }}")
    assert (destination / "package/manifest.json").stat().st_mode & 0o777 == 0o444
    assert destination.stat().st_mode & 0o777 == 0o555


def test_assembly_refuses_to_replace_a_native_template(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    source_config = json.loads((source / "tokenizer_config.json").read_text())
    source_config["chat_template"] = "native"
    (source / "tokenizer_config.json").write_text(json.dumps(source_config))
    template = tmp_path / "chat_template.jinja"
    template.write_text("{{ messages }}", encoding="utf-8")

    with pytest.raises(ASSEMBLER.AssemblyError, match="native contract"):
        ASSEMBLER.assemble(
            source_model_dir=source,
            chat_template=template,
            chat_template_revision="a" * 40,
            destination=tmp_path / "product",
        )


def test_assembly_requires_a_pinned_template_revision(tmp_path: Path) -> None:
    source = write_source(tmp_path)
    template = tmp_path / "chat_template.jinja"
    template.write_text("{{ messages }}", encoding="utf-8")

    with pytest.raises(ASSEMBLER.AssemblyError, match="Git SHA-1"):
        ASSEMBLER.assemble(
            source_model_dir=source,
            chat_template=template,
            chat_template_revision="unverified",
            destination=tmp_path / "product",
        )
