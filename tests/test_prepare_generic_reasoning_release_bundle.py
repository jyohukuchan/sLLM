from __future__ import annotations

import importlib.util
import json
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


class InjectedV2Validator:
    @staticmethod
    def validate(path: Path) -> dict[str, object]:
        return FIXTURES.validate_v2_bundle(path)


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


def test_prepare_accepts_complete_bundle_with_v2_no_switch_browser_evidence(
    tmp_path: Path,
) -> None:
    paths, _unused = inputs(tmp_path)
    browser = FIXTURES.BROWSER_FIXTURE.evidence()
    for field in FIXTURES.BROWSER_FIXTURE.TOOL.SWITCH_EVIDENCE_FIELDS:
        browser.pop(field)
    browser["provider_request_count"] = 2
    browser["provider_requests"] = browser["provider_requests"][:2]
    FIXTURES.write_json(paths["browser_evidence"], browser)
    FIXTURES.write_json(
        paths["browser_validator"],
        FIXTURES.BROWSER_FIXTURE.TOOL.validate(paths["browser_evidence"]),
    )
    output = tmp_path / "bundle.json"

    document = PREPARER.prepare(**paths, output=output, status="complete")

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


def test_prepare_v2_publishes_immutable_nine_slot_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_bundle, fakes, _report = FIXTURES.make_v2_bundle(tmp_path)
    FIXTURES.install_v2_fakes(monkeypatch, fakes)
    fixture = json.loads(fixture_bundle.read_text(encoding="ascii"))
    component_paths = {
        name: tmp_path / value["path"]
        for name, value in fixture["artifacts"].items()
    }
    fixture_bundle.unlink()
    rollback = {
        "rollback_manifest": tmp_path / "active.json",
        "systemd_unit": tmp_path / "ullm-openai.service",
        "environment_file": tmp_path / "ullm-openai.env",
    }
    for path in rollback.values():
        path.write_bytes(b"rollback-fixture")
    monkeypatch.setattr(PREPARER, "_load_validator", lambda: InjectedV2Validator)

    document = PREPARER.prepare_v2(
        **component_paths,
        **rollback,
        output=fixture_bundle,
        status="complete",
    )

    assert document["schema_version"] == FIXTURES.BUNDLE.SCHEMA_VERSION_V2
    assert len(document["artifacts"]) == 9
    assert fixture_bundle.stat().st_mode & 0o777 == 0o444
    assert fixture_bundle.stat().st_nlink == 1
    assert FIXTURES.validate_v2_bundle(fixture_bundle)["gate_eligible"] is True
    with pytest.raises(PREPARER.BundleError, match="already exists"):
        PREPARER.prepare_v2(
            **component_paths,
            **rollback,
            output=fixture_bundle,
            status="complete",
        )


def test_prepare_v2_no_replace_race_preserves_existing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_bundle, fakes, _report = FIXTURES.make_v2_bundle(tmp_path)
    FIXTURES.install_v2_fakes(monkeypatch, fakes)
    fixture = json.loads(fixture_bundle.read_text(encoding="ascii"))
    component_paths = {
        name: tmp_path / value["path"]
        for name, value in fixture["artifacts"].items()
    }
    fixture_bundle.unlink()
    rollback = {
        "rollback_manifest": tmp_path / "active.json",
        "systemd_unit": tmp_path / "ullm-openai.service",
        "environment_file": tmp_path / "ullm-openai.env",
    }
    for path in rollback.values():
        path.write_bytes(b"rollback-fixture")
    monkeypatch.setattr(PREPARER, "_load_validator", lambda: InjectedV2Validator)
    original_rename = PREPARER._rename_noreplace_at
    attacker = b"attacker-owned-target\n"

    def race_rename(
        parent: Path,
        source_name: str,
        destination_name: str,
    ) -> None:
        destination = parent / destination_name
        if destination == fixture_bundle:
            fixture_bundle.write_bytes(attacker)
        original_rename(parent, source_name, destination_name)

    monkeypatch.setattr(PREPARER, "_rename_noreplace_at", race_rename)

    with pytest.raises(PREPARER.BundleError, match="already exists"):
        PREPARER.prepare_v2(
            **component_paths,
            **rollback,
            output=fixture_bundle,
            status="complete",
        )

    assert fixture_bundle.read_bytes() == attacker


def test_publish_fault_after_noreplace_leaves_single_link_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "bundle.json"
    original_rename = PREPARER._rename_noreplace_at
    observed: dict[str, object] = {}

    def publish_then_fail(
        parent: Path,
        source_name: str,
        destination_name: str,
    ) -> None:
        original_rename(parent, source_name, destination_name)
        destination = parent / destination_name
        observed["source_exists"] = (parent / source_name).exists()
        observed["links"] = destination.stat().st_nlink
        raise RuntimeError("simulated interruption after rename")

    monkeypatch.setattr(
        PREPARER,
        "_rename_noreplace_at",
        publish_then_fail,
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        PREPARER._publish_immutable(output, {"schema_version": "fixture.v1"})

    assert output.is_file()
    assert output.stat().st_mode & 0o777 == 0o444
    assert output.stat().st_nlink == 1
    assert observed == {"source_exists": False, "links": 1}
    assert not list(tmp_path.glob(".bundle.json.*"))
