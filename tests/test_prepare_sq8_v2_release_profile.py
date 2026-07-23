from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/prepare-sq8-v2-release-profile.py"
BASE_PROFILE = ROOT / "deploy/served-models/qwen3-14b-sq8.profile.json"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_module("prepare_sq8_v2_release_profile_tool", TOOL_PATH)


def fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    product = tmp_path / "product"
    product.mkdir()
    base = json.loads(BASE_PROFILE.read_text(encoding="ascii"))
    base["product"]["root"] = os.fspath(product)
    profile = tmp_path / "base.json"
    profile.write_text(json.dumps(base), encoding="ascii")
    worker = tmp_path / "ullm-sq8-worker"
    worker.write_bytes(b"worker")
    worker.chmod(0o555)
    receipt = product / "sq8-serving-promotion-final.json"
    output = tmp_path / "profile-v2.json"
    return profile, worker, receipt, output


def test_prepare_publishes_exact_worker_v2_reasoning_profile(
    tmp_path: Path,
) -> None:
    base, worker, receipt, output = fixture(tmp_path)

    result = TOOL.prepare(
        base_profile=base,
        worker=worker,
        serving_receipt=receipt,
        output=output,
    )

    document = json.loads(output.read_text(encoding="ascii"))
    assert document["worker"]["protocol"] == "ullm.worker.v2"
    assert document["worker"]["binary"] == os.fspath(worker)
    assert document["reasoning"] == TOOL.promotion.REASONING_CONTRACT
    assert document["promotion"]["receipt"] == os.fspath(receipt)
    assert result["profile_sha256"]
    assert output.stat().st_mode & 0o777 == 0o444
    assert output.stat().st_nlink == 1


def test_prepare_rejects_existing_or_legacy_product_receipt(
    tmp_path: Path,
) -> None:
    base, worker, receipt, output = fixture(tmp_path)
    receipt.write_bytes(b"existing")
    with pytest.raises(TOOL.ProfileError, match="already exists"):
        TOOL.prepare(
            base_profile=base,
            worker=worker,
            serving_receipt=receipt,
            output=output,
        )
    receipt.unlink()
    with pytest.raises(TOOL.ProfileError, match="distinct file"):
        TOOL.prepare(
            base_profile=base,
            worker=worker,
            serving_receipt=receipt.with_name("promotion.json"),
            output=output,
        )


def test_prepare_rejects_mutable_or_misnamed_worker(tmp_path: Path) -> None:
    base, worker, receipt, output = fixture(tmp_path)
    worker.chmod(0o755)
    with pytest.raises(TOOL.ProfileError, match="immutable identity"):
        TOOL.prepare(
            base_profile=base,
            worker=worker,
            serving_receipt=receipt,
            output=output,
        )


def test_prepare_cli_failure_is_content_free(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = TOOL.main(
        [
            "--base-profile",
            os.fspath(tmp_path / "missing"),
            "--worker",
            os.fspath(tmp_path / "ullm-sq8-worker"),
            "--serving-receipt",
            os.fspath(tmp_path / "receipt"),
            "--output",
            os.fspath(tmp_path / "profile"),
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "SQ8 v2 release profile preparation failed\n"
