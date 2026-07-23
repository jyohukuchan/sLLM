from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import stat
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/freeze-served-model-manifest.py"
FIXTURE = (
    ROOT
    / "services/openai-gateway/tests/fixtures/served-model/aq4/served-model.json"
)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_module("freeze_served_model_manifest_tool", TOOL_PATH)


def test_freeze_publishes_exact_read_only_no_clobber_manifest(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    shutil.copytree(FIXTURE.parent, release)
    source = release / "candidate.json"
    (release / "served-model.json").rename(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = release / "frozen.json"

    result = TOOL.freeze(source, digest, output)

    assert output.read_bytes() == source.read_bytes()
    assert result["sha256"] == digest
    assert result["format_id"] == "AQ4_0"
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert output.stat().st_nlink == 1
    with pytest.raises(TOOL.FreezeError, match="already exists"):
        TOOL.freeze(source, digest, output)


def test_freeze_rejects_wrong_hash_without_publishing(tmp_path: Path) -> None:
    source = tmp_path / "candidate.json"
    source.write_bytes(FIXTURE.read_bytes())
    output = tmp_path / "frozen.json"

    with pytest.raises(TOOL.FreezeError, match="SHA-256 differs"):
        TOOL.freeze(source, "0" * 64, output)

    assert not output.exists()


def test_freeze_rejects_symlink_source_and_output_parent(tmp_path: Path) -> None:
    source = tmp_path / "candidate.json"
    source.write_bytes(FIXTURE.read_bytes())
    source_link = tmp_path / "candidate-link.json"
    source_link.symlink_to(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(TOOL.FreezeError, match="symlink"):
        TOOL.freeze(source_link, digest, tmp_path / "frozen.json")

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(TOOL.FreezeError, match="symlink"):
        TOOL.freeze(source, digest, linked_parent / "frozen.json")


def test_freeze_cli_failure_is_content_free(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = TOOL.main(
        [
            "--source",
            os.fspath(tmp_path / "secret-candidate.json"),
            "--expected-sha256",
            "0" * 64,
            "--output",
            os.fspath(tmp_path / "frozen.json"),
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "served-model manifest freeze failed\n"
