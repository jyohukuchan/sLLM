from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/validate-served-model.py"
FIXTURES = ROOT / "services/openai-gateway/tests/fixtures/served-model"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_module("test_validate_served_model_tool", TOOL_PATH)


@pytest.mark.parametrize(
    ("fixture", "model_id", "format_id", "has_artifact"),
    [
        ("sq8", "ullm-qwen3-14b-sq8", "SQ8_0", True),
        ("aq4", "ullm-qwen3.5-9b-aq4", "AQ4_0", False),
    ],
)
def test_summary_uses_gateway_loader_and_reports_non_secret_identity(
    fixture: str, model_id: str, format_id: str, has_artifact: bool
) -> None:
    manifest = FIXTURES / fixture / "served-model.json"
    summary = VALIDATOR.validation_summary(manifest)

    assert summary["schema_version"] == "ullm.served_model.validation.v1"
    assert summary["validated"] is True
    assert len(summary["manifest_sha256"]) == 64
    assert summary["model_id"] == model_id
    assert summary["format_id"] == format_id
    assert Path(summary["worker"]["binary"]).is_absolute()
    assert len(summary["worker"]["binary_sha256"]) == 64
    assert Path(summary["product"]["root"]).is_absolute()
    assert (summary["product"]["artifact"] is not None) is has_artifact
    assert len(summary["product"]["package"]["manifest_sha256"]) == 64
    assert "tokenizer" not in summary
    assert "promotion" not in summary


def test_cli_emits_one_canonical_json_summary() -> None:
    manifest = FIXTURES / "sq8/served-model.json"
    secret = "environment-api-key-must-not-leak"
    result = subprocess.run(
        [sys.executable, TOOL_PATH, "--manifest", manifest],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "ULLM_API_KEY": secret},
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.count("\n") == 1
    summary = json.loads(result.stdout)
    assert summary == VALIDATOR.validation_summary(manifest)
    assert result.stdout == (
        json.dumps(
            summary,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    assert secret not in result.stdout


def test_cli_failure_does_not_expose_manifest_content_or_loader_details(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "candidate"
    shutil.copytree(FIXTURES / "aq4", fixture)
    manifest = fixture / "served-model.json"
    secret = "super-secret-manifest-value"
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["secret"] = secret
    manifest.write_text(json.dumps(document), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, TOOL_PATH, "--manifest", manifest],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "served-model validation failed\n"
    assert secret not in result.stdout + result.stderr
    assert str(manifest) not in result.stdout + result.stderr
