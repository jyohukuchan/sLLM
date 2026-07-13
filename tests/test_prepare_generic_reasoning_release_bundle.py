from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
PREPARER_PATH = ROOT / "tools/prepare-generic-reasoning-release-bundle.py"
BUNDLE_TEST_PATH = ROOT / "tests/test_validate_generic_reasoning_release_bundle.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PREPARER = load_module("generic_reasoning_release_bundle_preparer", PREPARER_PATH)
FIXTURES = load_module("generic_reasoning_release_bundle_preparer_fixtures", BUNDLE_TEST_PATH)


def inputs(tmp_path: Path) -> tuple[dict[str, Path], Path]:
    bundle = FIXTURES.make_bundle(tmp_path)
    bundle.unlink()
    paths = {
        name: tmp_path / value["path"]
        for name, value in {
            "release_evidence": {"path": "release.json"},
            "release_validator": {"path": "release-validator.json"},
            "browser_evidence": {"path": "browser.json"},
            "browser_validator": {"path": "browser-validator.json"},
            "promotion_evidence": {"path": "promotion-evidence.json"},
            "promotion_receipt": {"path": "promotion-receipt.json"},
        }.items()
    }
    rollback = {
        "rollback_manifest": tmp_path / "active.json",
        "systemd_unit": tmp_path / "ullm-openai.service",
        "environment_file": tmp_path / "ullm-openai.env",
    }
    for path in rollback.values():
        path.write_bytes(b"rollback-fixture")
    return {**paths, **rollback}, bundle


def test_prepare_writes_valid_complete_bundle(tmp_path: Path) -> None:
    paths, _unused = inputs(tmp_path)
    output = tmp_path / "bundle.json"

    document = PREPARER.prepare(**paths, output=output, status="complete")

    assert output.is_file()
    assert document["status"] == "complete"
    assert PREPARER._load_validator().validate(output)["gate_eligible"] is True


def test_prepare_incomplete_status_preserves_gate_failure(tmp_path: Path) -> None:
    paths, _unused = inputs(tmp_path)
    output = tmp_path / "bundle.json"

    document = PREPARER.prepare(**paths, output=output, status="incomplete")

    assert document["status"] == "incomplete"
    report = PREPARER._load_validator().validate(output)
    assert report["structurally_valid"] is True
    assert report["gate_eligible"] is False
    assert "release bundle status is incomplete" in report["reasons"]


def test_prepare_rejects_artifact_outside_bundle_directory(tmp_path: Path) -> None:
    paths, _unused = inputs(tmp_path)
    output_dir = tmp_path / "nested"
    output = output_dir / "bundle.json"

    with pytest.raises(PREPARER.BundleError, match="below the bundle directory"):
        PREPARER.prepare(**paths, output=output)


def test_prepare_rejects_symlinked_rollback_input(tmp_path: Path) -> None:
    paths, _unused = inputs(tmp_path)
    target = paths["rollback_manifest"]
    linked = tmp_path / "active-link.json"
    linked.symlink_to(target)
    paths["rollback_manifest"] = linked

    with pytest.raises(PREPARER.BundleError, match="rollback manifest_sha256"):
        PREPARER.prepare(**paths, output=tmp_path / "bundle.json")
