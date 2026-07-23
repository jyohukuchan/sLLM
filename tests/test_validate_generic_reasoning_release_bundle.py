from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

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
            "gpu_exclusive_preflight": {
                "tool": "rocm-smi --showpids --json",
                "gpu_index": "1",
                "positive_vram_processes": [],
            },
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


def make_v2_bundle(
    root: Path,
) -> tuple[Path, dict[str, ModuleType | object], bytes]:
    source = "1" * 40
    worker_sha = "c" * 64
    tokenizer_sha = "d" * 64
    image = "registry.example/openwebui@sha256:" + "e" * 64
    promotion_path = root / "promotion-evidence.json"
    write_json(
        promotion_path,
        {"schema_version": "ullm.sq8_serving_promotion_evidence.v1"},
    )
    promotion_receipt_path = root / "promotion-receipt.json"
    write_json(
        promotion_receipt_path,
        {
            "schema_version": "ullm.sq8_serving_promotion.v1",
            "source_commit": source,
            "evidence": {
                "path": promotion_path.name,
                "sha256": digest(promotion_path),
            },
            "product": {},
        },
    )
    receipt_hash = digest(promotion_receipt_path)
    candidate_path = root / "campaign/candidate-served-model.json"
    candidate_path.parent.mkdir()
    write_json(
        candidate_path,
        {
            "schema_version": "ullm.served_model.v2",
            "public": {"id": "ullm-qwen3-14b-sq8"},
            "format": {"format_id": "SQ8_0"},
            "worker": {
                "protocol": "ullm.worker.v2",
                "binary_sha256": worker_sha,
            },
            "promotion": {
                "source_commit": source,
                "receipt": str(promotion_receipt_path),
                "receipt_sha256": receipt_hash,
            },
        },
    )
    identity = {
        "manifest_sha256": digest(candidate_path),
        "worker_binary_sha256": worker_sha,
        "tokenizer_sha256": tokenizer_sha,
        "openwebui_image": image,
    }
    release = RELEASE_FIXTURE.evidence()
    release.update(
        {
            "status": "complete",
            "source_commit": source,
            "active_promotion_source_commit": source,
            "source_commit_aligned": True,
            "identity": identity,
        }
    )
    release_path = root / "release.json"
    write_json(release_path, release)
    release_report_path = root / "release-validator.json"
    write_json(release_report_path, RELEASE_FIXTURE.TOOL.validate(release_path))

    browser = BROWSER_FIXTURE.evidence()
    browser.update(
        {
            "schema_version": "ullm.openwebui.reasoning_browser_smoke.v3",
            "source_commit": source,
            "identity": identity,
        }
    )
    browser_path = root / "browser.json"
    write_json(browser_path, browser)
    browser_report = {
        "schema_version": "ullm.openwebui.reasoning_browser_smoke_validator.v1",
        "input_schema_version": "ullm.openwebui.reasoning_browser_smoke.v3",
        "structurally_valid": True,
        "gate_eligible": True,
        "provider_request_count": 4,
        "reasons": [],
    }
    browser_report_path = root / "browser-validator.json"
    write_json(browser_report_path, browser_report)

    campaign_identity = {
        "schema_version": "ullm.sq8.full_campaign.model_identity.v2",
        "record_type": "fixture",
        "model": {},
        "promotion_validation": {},
        "product": {},
        "tokenizer": {},
        "oracle": {},
        "worker": {},
        "served_model_manifest": {
            "sha256": identity["manifest_sha256"],
            "worker_binary_sha256": worker_sha,
            "promotion_source_commit": source,
            "promotion_receipt_sha256": receipt_hash,
        },
        "campaign_authorization_claim": {
            "sha256": "7" * 64,
            "authorization_sha256": "8" * 64,
        },
    }
    campaign_identity_path = root / "campaign/model-identity.json"
    write_json(campaign_identity_path, campaign_identity)
    browser_dir = root / "campaign/browser"
    browser_dir.mkdir()
    (browser_dir / "proof.png").write_bytes(b"png")
    campaign_manifest_path = root / "campaign/SHA256SUMS"
    campaign_manifest_path.write_text("fixture\n", encoding="ascii")
    campaign_report_raw = (
        b'{"release_status":"complete","schema_version":'
        b'"ullm.sq8.openwebui_release.validation.v2"}\n'
    )
    campaign_report_path = root / "campaign/release-validation.json"
    campaign_report_path.write_bytes(campaign_report_raw)

    artifacts = {}
    for name, component in (
        ("release_evidence", release_path),
        ("release_validator", release_report_path),
        ("browser_evidence", browser_path),
        ("browser_validator", browser_report_path),
        ("promotion_evidence", promotion_path),
        ("promotion_receipt", promotion_receipt_path),
        ("model_campaign_manifest", campaign_manifest_path),
        ("model_campaign_evidence", campaign_identity_path),
        ("model_campaign_validator", campaign_report_path),
    ):
        artifacts[name] = {
            "path": component.relative_to(root).as_posix(),
            "sha256": digest(component),
        }
    bundle_path = root / "bundle-v2.json"
    write_json(
        bundle_path,
        {
            "schema_version": BUNDLE.SCHEMA_VERSION_V2,
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

    class BrowserValidator:
        @staticmethod
        def validate(_path: Path) -> dict[str, Any]:
            return browser_report

    class PromotionValidator:
        @staticmethod
        def validate_receipt(
            _path: Path,
            **_kwargs: object,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            return (
                {
                    "schema_version": "ullm.sq8_serving_promotion.v1",
                    "source_commit": source,
                },
                {
                    "schema_version": "ullm.sq8_serving_promotion_evidence.v1",
                    "worker": {"sha256": worker_sha},
                },
            )

    class ServedModelValidator:
        @staticmethod
        def validation_summary(_path: Path) -> dict[str, Any]:
            return {
                "manifest_sha256": identity["manifest_sha256"],
                "model_id": "ullm-qwen3-14b-sq8",
                "format_id": "SQ8_0",
                "worker": {
                    "protocol": "ullm.worker.v2",
                    "binary_sha256": worker_sha,
                },
            }

    class CampaignValidator:
        BUNDLE_FILES_V2 = {
            "SHA256SUMS",
            "model-identity.json",
            "candidate-served-model.json",
            "browser/proof.png",
        }

        @staticmethod
        def validate_full_release_no_publish(
            _path: Path,
            **_kwargs: object,
        ) -> bytes:
            return campaign_report_raw

    fakes: dict[str, ModuleType | object] = {
        "browser": BrowserValidator,
        "promotion": PromotionValidator,
        "served": ServedModelValidator,
        "campaign": CampaignValidator,
    }
    return bundle_path, fakes, campaign_report_raw


def install_v2_fakes(
    monkeypatch: pytest.MonkeyPatch,
    fakes: dict[str, ModuleType | object],
) -> None:
    original = BUNDLE._load_module

    def load(name: str, path: Path) -> ModuleType | object:
        if path == BUNDLE.BROWSER_VALIDATOR_PATH:
            return fakes["browser"]
        if path == BUNDLE.SQ8_PROMOTION_VALIDATOR_PATH:
            return fakes["promotion"]
        if path == BUNDLE.SERVED_MODEL_VALIDATOR_PATH:
            return fakes["served"]
        if path == BUNDLE.SQ8_CAMPAIGN_VALIDATOR_PATH:
            return fakes["campaign"]
        return original(name, path)

    monkeypatch.setattr(BUNDLE, "_load_module", load)


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


def test_bundle_rejects_missing_gpu_exclusivity_preflight(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    value = json.loads(bundle.read_text(encoding="ascii"))
    promotion_path = tmp_path / value["artifacts"]["promotion_evidence"]["path"]
    promotion = json.loads(promotion_path.read_text(encoding="ascii"))
    promotion.pop("gpu_exclusive_preflight")
    write_json(promotion_path, promotion)
    value["artifacts"]["promotion_evidence"]["sha256"] = digest(promotion_path)
    write_json(bundle, value)

    with pytest.raises(BUNDLE.ValidationError, match="GPU exclusivity preflight"):
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


def test_bundle_v2_recomputes_nine_slots_and_cross_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, fakes, _report = make_v2_bundle(tmp_path)
    install_v2_fakes(monkeypatch, fakes)

    result = BUNDLE.validate(bundle)

    assert result["input_schema_version"] == BUNDLE.SCHEMA_VERSION_V2
    assert result["schema_version"] == BUNDLE.VALIDATOR_SCHEMA_VERSION_V2
    assert result["gate_eligible"] is True
    assert result["artifact_count"] == 9
    assert (
        result["model_campaign_schema_version"]
        == "ullm.sq8.full_campaign.model_identity.v2"
    )


def test_bundle_v2_rejects_six_slot_v1_artifact_mix(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    value = json.loads(bundle.read_text(encoding="ascii"))
    value["schema_version"] = BUNDLE.SCHEMA_VERSION_V2
    write_json(bundle, value)

    with pytest.raises(BUNDLE.ValidationError, match="v2 artifacts differ"):
        BUNDLE.validate(bundle)


def test_bundle_v2_rejects_browser_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, fakes, _report = make_v2_bundle(tmp_path)
    install_v2_fakes(monkeypatch, fakes)
    value = json.loads(bundle.read_text(encoding="ascii"))
    browser_path = tmp_path / value["artifacts"]["browser_evidence"]["path"]
    browser = json.loads(browser_path.read_text(encoding="ascii"))
    browser["identity"]["tokenizer_sha256"] = "0" * 64
    write_json(browser_path, browser)
    value["artifacts"]["browser_evidence"]["sha256"] = digest(browser_path)
    write_json(bundle, value)

    with pytest.raises(BUNDLE.ValidationError, match="browser v3 identity"):
        BUNDLE.validate(bundle)


def test_bundle_v2_rejects_candidate_receipt_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, fakes, _report = make_v2_bundle(tmp_path)
    install_v2_fakes(monkeypatch, fakes)
    value = json.loads(bundle.read_text(encoding="ascii"))
    receipt_path = tmp_path / value["artifacts"]["promotion_receipt"]["path"]
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    receipt["product"] = {"mutated": True}
    write_json(receipt_path, receipt)
    value["artifacts"]["promotion_receipt"]["sha256"] = digest(receipt_path)
    write_json(bundle, value)

    with pytest.raises(BUNDLE.ValidationError, match="candidate promotion identity"):
        BUNDLE.validate(bundle)


def test_bundle_v2_rejects_forged_campaign_validator_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, fakes, _report = make_v2_bundle(tmp_path)
    install_v2_fakes(monkeypatch, fakes)
    value = json.loads(bundle.read_text(encoding="ascii"))
    report_path = tmp_path / value["artifacts"]["model_campaign_validator"]["path"]
    write_json(
        report_path,
        {
            "schema_version": "ullm.sq8.openwebui_release.validation.v2",
            "release_status": "complete",
            "forged": True,
        },
    )
    value["artifacts"]["model_campaign_validator"]["sha256"] = digest(report_path)
    write_json(bundle, value)

    with pytest.raises(BUNDLE.ValidationError, match="differs from recomputation"):
        BUNDLE.validate(bundle)


def test_bundle_v2_requires_exact_campaign_component_locations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, fakes, _report = make_v2_bundle(tmp_path)
    install_v2_fakes(monkeypatch, fakes)
    value = json.loads(bundle.read_text(encoding="ascii"))
    manifest_path = tmp_path / value["artifacts"]["model_campaign_manifest"]["path"]
    renamed = manifest_path.with_name("campaign-manifest.txt")
    manifest_path.rename(renamed)
    value["artifacts"]["model_campaign_manifest"]["path"] = renamed.relative_to(
        tmp_path
    ).as_posix()
    value["artifacts"]["model_campaign_manifest"]["sha256"] = digest(renamed)
    write_json(bundle, value)

    with pytest.raises(BUNDLE.ValidationError, match="artifact locations differ"):
        BUNDLE.validate(bundle)
