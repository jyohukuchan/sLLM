from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "tools/validate-generic-reasoning-release-bundle.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUNDLE = load_module("generic_reasoning_release_bundle_validator", BUNDLE_PATH)
RELEASE_FIXTURE = load_module(
    "generic_reasoning_release_bundle_release_fixture",
    ROOT / "tests/test_validate_generic_reasoning_release.py",
)
BROWSER_FIXTURE = load_module(
    "generic_reasoning_release_bundle_browser_fixture",
    ROOT / "tests/test_validate_openwebui_reasoning_browser_smoke.py",
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="ascii")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_bundle(root: Path) -> Path:
    source = "1" * 40
    release = RELEASE_FIXTURE.evidence()
    release["status"] = "complete"
    release["active_promotion_source_commit"] = source
    release["source_commit_aligned"] = True
    release_path = root / "release.json"
    write_json(release_path, release)
    release_report_path = root / "release-validator.json"
    write_json(release_report_path, RELEASE_FIXTURE.TOOL.validate(release_path))

    browser_path = root / "browser.json"
    write_json(browser_path, BROWSER_FIXTURE.evidence())
    browser_report_path = root / "browser-validator.json"
    write_json(browser_report_path, BROWSER_FIXTURE.TOOL.validate(browser_path))

    identity = release["identity"]
    promotion_path = root / "promotion-evidence.json"
    write_json(
        promotion_path,
        {
            "schema_version": "ullm.aq4_resident_promotion_evidence.v1",
            "source_commit": source,
            "production_receipt_written": False,
            "verified": True,
            "worker_binary_sha256": identity["worker_binary_sha256"],
            "ephemeral_bundle": {"manifest_sha256": identity["manifest_sha256"]},
        },
    )
    receipt_path = root / "promotion-receipt.json"
    write_json(
        receipt_path,
        {
            "schema_version": "ullm.aq4_resident_promotion.v1",
            "source_commit": source,
            "evidence": {"path": promotion_path.name, "sha256": digest(promotion_path)},
        },
    )

    artifacts = {}
    for name, path in (
        ("release_evidence", release_path),
        ("release_validator", release_report_path),
        ("browser_evidence", browser_path),
        ("browser_validator", browser_report_path),
        ("promotion_evidence", promotion_path),
        ("promotion_receipt", receipt_path),
    ):
        artifacts[name] = {"path": path.name, "sha256": digest(path)}
    bundle_path = root / "bundle.json"
    write_json(
        bundle_path,
        {
            "schema_version": BUNDLE.SCHEMA_VERSION,
            "status": "complete",
            "production_activation_performed": False,
            "source_commit": source,
            "active_promotion_source_commit": source,
            "identity": identity,
            "artifacts": artifacts,
            "rollback_target": {
                "manifest_sha256": "f" * 64,
                "systemd_unit_sha256": "e" * 64,
                "environment_sha256": "d" * 64,
            },
        },
    )
    return bundle_path


def test_bundle_recomputes_component_validators_and_bindings(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)

    report = BUNDLE.validate(bundle)

    assert report["structurally_valid"] is True
    assert report["gate_eligible"] is True
    assert report["artifact_count"] == 6


def test_bundle_rejects_forged_validator_report(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    value = json.loads(bundle.read_text(encoding="ascii"))
    report_path = tmp_path / value["artifacts"]["release_validator"]["path"]
    report = json.loads(report_path.read_text(encoding="ascii"))
    report["gate_eligible"] = False
    write_json(report_path, report)
    value["artifacts"]["release_validator"]["sha256"] = digest(report_path)
    write_json(bundle, value)

    with pytest.raises(BUNDLE.ValidationError, match="validator report differs"):
        BUNDLE.validate(bundle)


def test_bundle_rejects_absolute_component_path(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    value = json.loads(bundle.read_text(encoding="ascii"))
    value["artifacts"]["release_evidence"]["path"] = "/etc/hosts"
    write_json(bundle, value)

    with pytest.raises(BUNDLE.ValidationError, match="path is unsafe"):
        BUNDLE.validate(bundle)


def test_bundle_preserves_incomplete_gate_result(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    value = json.loads(bundle.read_text(encoding="ascii"))
    release_path = tmp_path / value["artifacts"]["release_evidence"]["path"]
    release = json.loads(release_path.read_text(encoding="ascii"))
    release["status"] = "incomplete"
    write_json(release_path, release)
    value["artifacts"]["release_evidence"]["sha256"] = digest(release_path)
    validator_path = tmp_path / value["artifacts"]["release_validator"]["path"]
    write_json(validator_path, RELEASE_FIXTURE.TOOL.validate(release_path))
    value["artifacts"]["release_validator"]["sha256"] = digest(validator_path)
    write_json(bundle, value)

    report = BUNDLE.validate(bundle)

    assert report["structurally_valid"] is True
    assert report["gate_eligible"] is False
    assert "release validator gate is not eligible" in report["reasons"]


def test_bundle_rejects_symlink_component(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    target = tmp_path / "release.json"
    linked = tmp_path / "release-link.json"
    linked.symlink_to(target)
    value = json.loads(bundle.read_text(encoding="ascii"))
    value["artifacts"]["release_evidence"]["path"] = linked.name
    value["artifacts"]["release_evidence"]["sha256"] = digest(target)
    write_json(bundle, value)

    with pytest.raises(BUNDLE.ValidationError, match="path is a symlink"):
        BUNDLE.validate(bundle)
