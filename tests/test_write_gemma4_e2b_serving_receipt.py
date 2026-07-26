from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/write-gemma4-e2b-serving-receipt.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RECEIPT_TOOL = load_module("test_gemma4_e2b_receipt", TOOL_PATH)


def test_receipt_binds_worker_package_and_template(tmp_path: Path) -> None:
    worker = tmp_path / "worker"
    package = tmp_path / "package.json"
    tokenizer = tmp_path / "tokenizer_config.json"
    worker.write_bytes(b"worker")
    package.write_bytes(b"package")
    tokenizer.write_text(json.dumps({"chat_template": "{{ messages }}"}))
    output = tmp_path / "receipt.json"

    receipt = RECEIPT_TOOL.write_receipt(
        source_commit="a" * 40,
        worker_binary=worker,
        package_manifest=package,
        tokenizer_config=tokenizer,
        output=output,
    )

    assert json.loads(output.read_text()) == receipt
    assert receipt["worker_binary_sha256"] == hashlib.sha256(b"worker").hexdigest()
    assert receipt["package_manifest_sha256"] == hashlib.sha256(b"package").hexdigest()
    assert receipt["tokenizer_chat_template_sha256"] == hashlib.sha256(
        b"{{ messages }}"
    ).hexdigest()
    assert output.stat().st_mode & 0o777 == 0o444


def test_receipt_refuses_unpinned_commit_or_duplicate_output(tmp_path: Path) -> None:
    worker = tmp_path / "worker"
    package = tmp_path / "package.json"
    tokenizer = tmp_path / "tokenizer_config.json"
    worker.write_bytes(b"worker")
    package.write_bytes(b"package")
    tokenizer.write_text(json.dumps({"chat_template": "{{ messages }}"}))
    output = tmp_path / "receipt.json"

    with pytest.raises(RECEIPT_TOOL.ReceiptError, match="Git SHA-1"):
        RECEIPT_TOOL.write_receipt(
            source_commit="unverified",
            worker_binary=worker,
            package_manifest=package,
            tokenizer_config=tokenizer,
            output=output,
        )
    output.write_text("existing")
    with pytest.raises(RECEIPT_TOOL.ReceiptError, match="already exists"):
        RECEIPT_TOOL.write_receipt(
            source_commit="a" * 40,
            worker_binary=worker,
            package_manifest=package,
            tokenizer_config=tokenizer,
            output=output,
        )
