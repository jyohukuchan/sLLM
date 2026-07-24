from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
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
AUTH = load_module(
    "generic_reasoning_release_bundle_authorization_fixture",
    ROOT / "tools/served_model_campaign_authorization.py",
)
RELEASE_FIXTURE = load_module(
    "generic_reasoning_release_bundle_release_fixture",
    ROOT / "tests/test_validate_generic_reasoning_release.py",
)
BROWSER_FIXTURE = load_module(
    "generic_reasoning_release_bundle_browser_fixture",
    ROOT / "tests/test_validate_openwebui_reasoning_browser_smoke.py",
)
_REAL_VALIDATOR_FIXTURES: tuple[ModuleType, ModuleType] | None = None


def real_validator_fixtures() -> tuple[ModuleType, ModuleType]:
    global _REAL_VALIDATOR_FIXTURES
    if _REAL_VALIDATOR_FIXTURES is None:
        promotion = load_module(
            "generic_reasoning_release_bundle_real_sq8_promotion_fixture",
            ROOT / "tests/test_sq8_serving_promotion.py",
        )
        full_campaign = load_module(
            "generic_reasoning_release_bundle_real_full_campaign_fixture",
            ROOT / "tests/test_sq8_full_campaign_fake_integration.py",
        )
        _REAL_VALIDATOR_FIXTURES = promotion, full_campaign
    return _REAL_VALIDATOR_FIXTURES


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="ascii")


def write_canonical_json(path: Path, value: object) -> None:
    path.write_bytes(
        (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_openwebui_server_observation() -> dict[str, object]:
    return {
        "container_id": "1" * 64,
        "image_id": AUTH.FIXED_OPENWEBUI_IMAGE.rsplit("@", 1)[1],
        "config_image": AUTH.FIXED_OPENWEBUI_CONFIG_IMAGE,
        "name": f"/{AUTH.FIXED_OPENWEBUI_CONTAINER_NAME}",
        "running": True,
        "pid": 1234,
        "started_at": "2026-07-24T00:00:00.000000000Z",
    }


def add_aq4_authorization_v2_bindings(
    document: dict[str, Any],
    *,
    root: Path,
) -> None:
    """Populate the AQ4 half of the exact-six authorization fixture."""

    aq4_source = root / "aq4-source"
    aq4_output = root / "aq4-output"
    before = document["before"]
    before.update(
        {
            "worker_protocol": "ullm.worker.v2",
            "worker_binary_path": str(aq4_source / "ullm-worker"),
            "promotion_receipt_path": str(
                aq4_source / "promotion-receipt.json"
            ),
            "promotion_receipt_sha256": "4" * 64,
        }
    )
    document["aq4_release"] = {
        "source": {
            "root": str(aq4_source),
            "commit": before["promotion_source_commit"],
            "tree": "5" * 40,
        },
        "openwebui_image": AUTH.FIXED_OPENWEBUI_IMAGE,
        "promotion_evidence": {
            "source_path": str(aq4_source / "promotion-evidence.json"),
            "path": str(aq4_output / "promotion-evidence.json"),
            "sha256": "7" * 64,
        },
        "promotion_receipt": {
            "source_path": before["promotion_receipt_path"],
            "path": str(aq4_output / "promotion-receipt.json"),
            "sha256": before["promotion_receipt_sha256"],
        },
        "release_evidence_path": str(aq4_output / "release-evidence.json"),
        "release_validator_path": str(aq4_output / "release-validator.json"),
        "browser_validator_path": str(aq4_output / "browser-validator.json"),
    }
    document["campaigns"].update(
        {
            "aq4_reasoning_release": {
                "run_id": "aq4-reasoning-release-run",
                "final_path": str(aq4_output / "reasoning-release"),
            },
            "aq4_reasoning_browser": {
                "run_id": "aq4-reasoning-browser-run",
                "final_path": str(aq4_output / "browser-evidence.json"),
            },
            "aq4_bundle": {
                "run_id": "aq4-bundle-run",
                "final_path": str(aq4_output / "bundle.json"),
            },
        }
    )


def seal_bundle_v2(path: Path) -> None:
    path.chmod(0o444)


def rewrite_bundle_v2(path: Path, value: object) -> None:
    path.chmod(0o644)
    try:
        write_json(path, value)
    finally:
        path.chmod(0o444)


def active_binding_artifacts(
    *,
    output: Path,
    candidate_source: Path,
    candidate_raw: bytes,
    claim: dict[str, Any],
    campaign_name: str,
    run_id: str,
    stages: tuple[str, ...],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    candidate_sha256 = hashlib.sha256(candidate_raw).hexdigest()
    identity = {
        "device": 1,
        "inode": 2,
        "mode": 0o444,
        "links": 1,
        "uid": 1000,
        "gid": 1000,
        "bytes": len(candidate_raw),
        "mtime_ns": 3,
        "ctime_ns": 4,
    }
    rows = [
        {
            "schema_version": "ullm.served_model.active_manifest_observation.v1",
            "sequence": sequence,
            "stage": stage,
            "observed_unix_ns": sequence,
            "observed_monotonic_ns": sequence,
            "candidate": {
                "path": str(candidate_source),
                "sha256": candidate_sha256,
                "identity": identity,
            },
            "active": {
                "path": str(output.parent / "active.json"),
                "sha256": candidate_sha256,
                "identity": identity,
            },
            "bytes_equal": True,
            "claim": claim,
        }
        for sequence, stage in enumerate(stages)
    ]
    observations_raw = b"".join(
        (
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("ascii")
        for row in rows
    )
    binding = {
        "schema_version": "ullm.served_model.active_binding.v1",
        "status": "complete",
        "candidate": {
            "artifact": "candidate-served-model.json",
            "source_path": str(candidate_source),
            "sha256": candidate_sha256,
            "bytes": len(candidate_raw),
        },
        "actual_active_path": str(output.parent / "active.json"),
        "expected_stages": list(stages),
        "observation_count": len(stages),
        "observations": {
            "artifact": "active-manifest-observations.jsonl",
            "sha256": hashlib.sha256(observations_raw).hexdigest(),
            "bytes": len(observations_raw),
        },
        "claim": claim,
        "campaign": {
            "name": campaign_name,
            "run_id": run_id,
            "final_path": str(output),
        },
    }
    binding_raw = (
        json.dumps(binding, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("ascii")
    return (
        {
            "candidate-served-model.json": candidate_raw,
            "active-manifest-observations.jsonl": observations_raw,
            "active-manifest-binding.json": binding_raw,
        },
        binding,
    )


def campaign_lineage(
    *,
    output: Path,
    campaign_name: str,
    run_id: str,
    claim: dict[str, Any],
    artifacts: dict[str, bytes],
    files: set[str],
    stages: tuple[str, ...],
) -> dict[str, Any]:
    references = {
        name: {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        for name, raw in artifacts.items()
    }
    canonical = json.dumps(
        references, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    observation_lines = artifacts[
        "active-manifest-observations.jsonl"
    ].splitlines(keepends=True)
    return {
        "schema_version": "ullm.served_model.campaign_lineage.v2",
        "campaign": {
            "name": campaign_name,
            "run_id": run_id,
            "final_path": str(output),
            "final_kind": "directory",
            "files": sorted(files),
        },
        "claim": claim,
        "artifacts": references,
        "artifact_inventory_sha256": hashlib.sha256(
            b"ullm.served_model.campaign_lineage.v2\0" + canonical
        ).hexdigest(),
        "observations": {
            "count": len(stages),
            "stages": [
                {
                    "sequence": sequence,
                    "stage": stage,
                    "sha256": hashlib.sha256(
                        observation_lines[sequence]
                    ).hexdigest(),
                }
                for sequence, stage in enumerate(stages)
            ],
        },
    }


def write_immutable_directory(output: Path, values: dict[str, bytes]) -> None:
    output.mkdir()
    for name, raw in values.items():
        path = output / name
        path.write_bytes(raw)
        path.chmod(0o444)
    output.chmod(0o555)


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
    *,
    authorization_schema: str | None = None,
    claim_schema: str | None = None,
    include_aq4_campaigns: bool = True,
    authorized_sq8_run_id: str = "sq8-full-run",
    authorized_sq8_final_path: Path | None = None,
) -> tuple[Path, dict[str, ModuleType | object], bytes]:
    source = "1" * 40
    worker_sha = "c" * 64
    tokenizer_sha = "d" * 64
    image = AUTH.FIXED_OPENWEBUI_IMAGE
    rollback_sha = hashlib.sha256(b"rollback-fixture").hexdigest()
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
    authorization_path = root / "campaign-authorization.json"
    authorization_document = {
        "schema_version": (
            BUNDLE.AUTHORIZATION_SCHEMA
            if authorization_schema is None
            else authorization_schema
        ),
        "authorization_id": "bundle-v2-fixture-authorization",
        "issued_at": "2026-07-24T00:00:00Z",
        "expires_at": "2026-07-25T00:00:00Z",
        "max_attempts": 1,
        "authorization_note": "Bundle v2 cross-binding fixture.",
        "purpose": "temporary_candidate_active_evidence_collection_only",
        "required_final_route": "restore_exact_aq4_then_bundle_v2_activation",
        "source": {"commit": source, "tree": "2" * 40},
        "before": {
            "model_id": "ullm-qwen3.5-9b-aq4",
            "format_id": "AQ4_0",
            "manifest_sha256": rollback_sha,
            "worker_binary_sha256": "b" * 64,
            "promotion_source_commit": "3" * 40,
        },
        "candidate": {
            "model_id": "ullm-qwen3-14b-sq8",
            "format_id": "SQ8_0",
            "manifest_sha256": identity["manifest_sha256"],
            "worker_protocol": "ullm.worker.v2",
            "worker_binary_sha256": worker_sha,
            "promotion_source_commit": source,
            "promotion_receipt_sha256": receipt_hash,
        },
        "campaigns": {
            "sq8_full": {
                "run_id": authorized_sq8_run_id,
                "final_path": str(
                    root / "campaign"
                    if authorized_sq8_final_path is None
                    else authorized_sq8_final_path
                ),
            },
            "reasoning_release": {
                "run_id": "reasoning-release-run",
                "final_path": str(root / "reasoning-release"),
            },
            "reasoning_browser": {
                "run_id": "reasoning-browser-run",
                "final_path": str(root / "reasoning-browser"),
            },
        },
        "rollback": {
            "backup_path": str(root / "aq4-backup.json"),
            "systemd_unit_sha256": rollback_sha,
            "environment_sha256": rollback_sha,
        },
        "prior_outcome": None,
    }
    add_aq4_authorization_v2_bindings(
        authorization_document,
        root=root,
    )
    if not include_aq4_campaigns:
        for name in (
            "aq4_reasoning_release",
            "aq4_reasoning_browser",
            "aq4_bundle",
        ):
            del authorization_document["campaigns"][name]
    write_canonical_json(authorization_path, authorization_document)
    authorization_path.chmod(0o444)
    claim_path = root / "campaign-authorization.claimed.json"
    write_canonical_json(
        claim_path,
        {
            "schema_version": (
                BUNDLE.CLAIM_SCHEMA
                if claim_schema is None
                else claim_schema
            ),
            "authorization_id": authorization_document["authorization_id"],
            "authorization_path": str(authorization_path),
            "authorization_sha256": digest(authorization_path),
            "claimed_at": "2026-07-24T00:01:00Z",
            "attempt": 1,
            "max_attempts": 1,
        },
    )
    claim_path.chmod(0o444)
    claim = {
        "path": str(claim_path),
        "sha256": digest(claim_path),
        "bytes": len(claim_path.read_bytes()),
        "authorization_path": str(authorization_path),
        "authorization_sha256": digest(authorization_path),
    }

    release = RELEASE_FIXTURE.evidence()
    release.update(
        {
            "schema_version": "ullm.generic_reasoning_release_evidence.v2",
            "status": "complete",
            "source_commit": source,
            "active_promotion_source_commit": source,
            "source_commit_aligned": True,
            "identity": identity,
        }
    )
    reasoning_output = root / "reasoning-release"
    reasoning_stages = tuple(
        RELEASE_FIXTURE.TOOL.REASONING_CAMPAIGN_STAGES
    )
    reasoning_binding_artifacts, reasoning_binding = active_binding_artifacts(
        output=reasoning_output,
        candidate_source=candidate_path,
        candidate_raw=candidate_path.read_bytes(),
        claim=claim,
        campaign_name="reasoning_release",
        run_id="reasoning-release-run",
        stages=reasoning_stages,
    )
    reasoning_values = {
        "cases.json": (
            json.dumps(release["cases"], separators=(",", ":")).encode("ascii")
            + b"\n"
        ),
        "lifecycle.json": (
            json.dumps(release["lifecycle"], separators=(",", ":")).encode("ascii")
            + b"\n"
        ),
        "resource-samples.jsonl": b"{}\n",
        **reasoning_binding_artifacts,
    }
    reasoning_summary = {
        "schema_version": "ullm.generic_reasoning_release_campaign.v2",
        "status": "incomplete",
        "raw_bodies_stored": False,
        "case_count": len(release["cases"]),
        "stream_case_count": len(release["cases"]),
        "nonstream_case_count": 0,
        "modes": ["disabled", "budget-32", "budget-128", "budget-256", "unbounded"],
        "manifest_sha256": identity["manifest_sha256"],
        "model_id": "ullm-qwen3-14b-sq8",
        "worker_binary_sha256": worker_sha,
        "gpu_exclusive_preflight": {},
        "active_manifest_binding": reasoning_binding,
        "run_id": "reasoning-release-run",
    }
    reasoning_values["summary.json"] = (
        json.dumps(reasoning_summary, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    release["campaign_lineage"] = campaign_lineage(
        output=reasoning_output,
        campaign_name="reasoning_release",
        run_id="reasoning-release-run",
        claim=claim,
        artifacts=reasoning_values,
        files=set(reasoning_values),
        stages=reasoning_stages,
    )
    write_immutable_directory(reasoning_output, reasoning_values)
    release_path = root / "release.json"
    write_json(release_path, release)
    release_report_path = root / "release-validator.json"
    write_json(release_report_path, RELEASE_FIXTURE.TOOL.validate(release_path))

    browser = BROWSER_FIXTURE.evidence()
    browser.update(
        {
            "schema_version": "ullm.openwebui.reasoning_browser_smoke.v5",
            "source_commit": source,
            "identity": identity,
            "browser_image": AUTH.FIXED_BROWSER_IMAGE,
            "openwebui_server": {
                "before": fixed_openwebui_server_observation(),
                "after": fixed_openwebui_server_observation(),
            },
        }
    )
    browser_output = root / "reasoning-browser"
    browser_stages = tuple(BROWSER_FIXTURE.TOOL.ACTIVE_BINDING_STAGES)
    browser_artifacts, _browser_binding = active_binding_artifacts(
        output=browser_output,
        candidate_source=candidate_path,
        candidate_raw=candidate_path.read_bytes(),
        claim=claim,
        campaign_name="reasoning_browser",
        run_id="reasoning-browser-run",
        stages=browser_stages,
    )
    browser["campaign_lineage"] = campaign_lineage(
        output=browser_output,
        campaign_name="reasoning_browser",
        run_id="reasoning-browser-run",
        claim=claim,
        artifacts=browser_artifacts,
        files={*browser_artifacts, "browser-evidence.json"},
        stages=browser_stages,
    )
    browser_raw = (
        json.dumps(browser, separators=(",", ":"), sort_keys=True).encode("ascii")
        + b"\n"
    )
    write_immutable_directory(
        browser_output,
        {**browser_artifacts, "browser-evidence.json": browser_raw},
    )
    browser_path = browser_output / "browser-evidence.json"
    browser_report = BROWSER_FIXTURE.TOOL.validate(browser_path)
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
        "campaign_authorization_claim": claim,
    }
    campaign_identity_path = root / "campaign/model-identity.json"
    write_json(campaign_identity_path, campaign_identity)
    browser_dir = root / "campaign/browser"
    browser_dir.mkdir()
    (browser_dir / "proof.png").write_bytes(b"png")
    campaign_manifest_path = root / "campaign/SHA256SUMS"
    campaign_manifest_path.write_text("fixture\n", encoding="ascii")
    campaign_report_raw = (
        b'{"release_status":"complete","run_id":"sq8-full-run",'
        b'"schema_version":"ullm.sq8.openwebui_release.validation.v2"}\n'
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
                "manifest_sha256": rollback_sha,
                "systemd_unit_sha256": rollback_sha,
                "environment_sha256": rollback_sha,
            },
        },
    )
    seal_bundle_v2(bundle_path)

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


def _tokenizer_identity(candidate: dict[str, Any]) -> str:
    digest_value = hashlib.sha256()
    files = candidate["tokenizer"]["files"]
    for name in sorted(files):
        digest_value.update(name.encode("utf-8"))
        digest_value.update(b"\0")
        digest_value.update(bytes.fromhex(files[name]))
    return digest_value.hexdigest()


def _real_auxiliary_campaigns(
    *,
    root: Path,
    candidate_path: Path,
    claim: dict[str, Any],
    identity: dict[str, Any],
    source_commit: str,
) -> tuple[Path, Path, Path, Path]:
    release = RELEASE_FIXTURE.evidence()
    release.update(
        {
            "schema_version": "ullm.generic_reasoning_release_evidence.v2",
            "status": "complete",
            "source_commit": source_commit,
            "active_promotion_source_commit": source_commit,
            "source_commit_aligned": True,
            "identity": identity,
        }
    )
    release_output = root / "reasoning-release-real"
    release_stages = tuple(RELEASE_FIXTURE.TOOL.REASONING_CAMPAIGN_STAGES)
    release_binding_artifacts, release_binding = active_binding_artifacts(
        output=release_output,
        candidate_source=candidate_path,
        candidate_raw=candidate_path.read_bytes(),
        claim=claim,
        campaign_name="reasoning_release",
        run_id="reasoning-release-real-run",
        stages=release_stages,
    )
    release_values = {
        "cases.json": (
            json.dumps(release["cases"], separators=(",", ":")).encode("ascii")
            + b"\n"
        ),
        "lifecycle.json": (
            json.dumps(release["lifecycle"], separators=(",", ":")).encode("ascii")
            + b"\n"
        ),
        "resource-samples.jsonl": b"{}\n",
        **release_binding_artifacts,
    }
    release_summary = {
        "schema_version": "ullm.generic_reasoning_release_campaign.v2",
        "status": "complete",
        "raw_bodies_stored": False,
        "case_count": len(release["cases"]),
        "stream_case_count": len(release["cases"]),
        "nonstream_case_count": 0,
        "modes": [
            "disabled",
            "budget-32",
            "budget-128",
            "budget-256",
            "unbounded",
        ],
        "manifest_sha256": identity["manifest_sha256"],
        "model_id": "ullm-qwen3-14b-sq8",
        "worker_binary_sha256": identity["worker_binary_sha256"],
        "gpu_exclusive_preflight": {},
        "active_manifest_binding": release_binding,
        "run_id": "reasoning-release-real-run",
    }
    release_values["summary.json"] = (
        json.dumps(
            release_summary,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    release["campaign_lineage"] = campaign_lineage(
        output=release_output,
        campaign_name="reasoning_release",
        run_id="reasoning-release-real-run",
        claim=claim,
        artifacts=release_values,
        files=set(release_values),
        stages=release_stages,
    )
    write_immutable_directory(release_output, release_values)
    release_path = root / "release-real.json"
    write_json(release_path, release)
    release_report_path = root / "release-real-validator.json"
    write_json(
        release_report_path,
        RELEASE_FIXTURE.TOOL.validate(release_path),
    )

    browser = BROWSER_FIXTURE.evidence()
    browser.update(
        {
            "schema_version": "ullm.openwebui.reasoning_browser_smoke.v5",
            "source_commit": source_commit,
            "identity": identity,
            "browser_image": AUTH.FIXED_BROWSER_IMAGE,
            "openwebui_server": {
                "before": fixed_openwebui_server_observation(),
                "after": fixed_openwebui_server_observation(),
            },
        }
    )
    browser_output = root / "reasoning-browser-real"
    browser_stages = tuple(BROWSER_FIXTURE.TOOL.ACTIVE_BINDING_STAGES)
    browser_artifacts, _browser_binding = active_binding_artifacts(
        output=browser_output,
        candidate_source=candidate_path,
        candidate_raw=candidate_path.read_bytes(),
        claim=claim,
        campaign_name="reasoning_browser",
        run_id="reasoning-browser-real-run",
        stages=browser_stages,
    )
    browser["campaign_lineage"] = campaign_lineage(
        output=browser_output,
        campaign_name="reasoning_browser",
        run_id="reasoning-browser-real-run",
        claim=claim,
        artifacts=browser_artifacts,
        files={*browser_artifacts, "browser-evidence.json"},
        stages=browser_stages,
    )
    browser_raw = (
        json.dumps(browser, separators=(",", ":"), sort_keys=True).encode("ascii")
        + b"\n"
    )
    write_immutable_directory(
        browser_output,
        {**browser_artifacts, "browser-evidence.json": browser_raw},
    )
    browser_path = browser_output / "browser-evidence.json"
    browser_report_path = root / "browser-real-validator.json"
    write_json(
        browser_report_path,
        BROWSER_FIXTURE.TOOL.validate(browser_path),
    )
    return release_path, release_report_path, browser_path, browser_report_path


def make_real_validator_v2_bundle(
    root: Path,
) -> tuple[Path, Any]:
    promotion_fixtures, full_fixtures = real_validator_fixtures()

    class HeadPromotionFixture(promotion_fixtures.Fixture):
        def _write_source(self) -> None:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "-q",
                    "--no-hardlinks",
                    os.fspath(ROOT),
                    os.fspath(self.source),
                ],
                check=True,
            )

    promotion_root = root / "promotion-real"
    promotion_root.mkdir()
    promotion = HeadPromotionFixture(promotion_root)
    promotion.publish_evidence()
    promotion.publish_receipt()
    candidate_path = promotion.release / "served-model-final.json"
    promotion_fixtures.GENERATOR.generate(promotion.profile, candidate_path)
    candidate_path.chmod(0o444)
    candidate = json.loads(candidate_path.read_text(encoding="ascii"))
    source_commit = promotion.commit
    worker_sha256 = digest(promotion.worker)

    original_worker_sha256 = full_fixtures.WORKER_SHA256

    class RealV2FullCampaign(full_fixtures.FullFakeCampaign):
        def __init__(
            self,
            base: Path,
            *,
            claim_value: dict[str, Any],
        ) -> None:
            super().__init__(base)
            self.bundle.abort()
            self.bundle = full_fixtures.AtomicCampaignDirectory(
                self.final_path,
                uid=os.getuid(),
                gid=os.getgid(),
                layout_version="v2",
            )
            self.claim_value = claim_value

        def _identity_checkout(self) -> None:
            environment, model_identity = (
                full_fixtures.VALIDATOR_FIXTURES.build_identity_documents(
                    v2=True
                )
            )
            role_paths = full_fixtures.VALIDATOR.EXPECTED_SOURCE_ROLE_PATHS_V2
            source_groups = full_fixtures.VALIDATOR.EXPECTED_SOURCE_GROUPS_V2
            for source in environment["sources"]:
                relative = role_paths[source["role"]]
                raw = (promotion.source / relative).read_bytes()
                source.update(
                    path=relative,
                    bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(),
                )
            environment["sources"].sort(
                key=lambda item: item["path"].encode("utf-8")
            )
            by_role = {
                source["role"]: source for source in environment["sources"]
            }
            source_sets = {
                group: hashlib.sha256(
                    full_fixtures.VALIDATOR_FIXTURES.identity_canonical(
                        [by_role[role] for role in sorted(roles)]
                    )
                ).hexdigest()
                for group, roles in source_groups.items()
            }
            environment["git"]["commit"] = source_commit
            environment["source_sets"] = source_sets
            for target, role in (
                (
                    environment["deployment"]["service_unit_file"],
                    "systemd_service",
                ),
                (
                    environment["deployment"]["environment_file"],
                    "systemd_environment_contract",
                ),
            ):
                target["bytes"] = by_role[role]["bytes"]
                target["sha256"] = by_role[role]["sha256"]
            environment["openwebui"]["Dockerfile_sha256"] = by_role[
                "openwebui_dockerfile"
            ]["sha256"]
            environment["openwebui"]["patch_sha256"] = by_role[
                "openwebui_patch"
            ]["sha256"]
            environment["service"]["worker"]["executable_bytes"] = (
                promotion.worker.stat().st_size
            )
            environment["service"]["worker"]["executable_sha256"] = worker_sha256
            model_identity["promotion_validation"][
                "validator_source_sha256"
            ] = by_role["product_promotion_validator"]["sha256"]
            model_identity["promotion_validation"][
                "canonical_source_sha256"
            ] = by_role["product_promotion_canonical"]["sha256"]
            model_identity["worker"]["binary_bytes"] = promotion.worker.stat().st_size
            model_identity["worker"]["binary_sha256"] = worker_sha256
            model_identity["worker"]["source_sha256"] = source_sets["worker"]
            model_identity["served_model_manifest"] = {
                "artifact": "candidate-served-model.json",
                "source_path": str(candidate_path),
                "bytes": len(candidate_path.read_bytes()),
                "sha256": digest(candidate_path),
                "schema_version": "ullm.served_model.v2",
                "model_id": candidate["public"]["id"],
                "model_revision": candidate["public"]["revision"],
                "format_id": candidate["format"]["format_id"],
                "worker_protocol": candidate["worker"]["protocol"],
                "worker_binary_sha256": candidate["worker"]["binary_sha256"],
                "promotion_source_commit": candidate["promotion"][
                    "source_commit"
                ],
                "promotion_receipt_sha256": candidate["promotion"][
                    "receipt_sha256"
                ],
            }
            model_identity["campaign_authorization_claim"] = self.claim_value
            self.source_root = promotion.source
            self.commit = source_commit
            self.environment = environment
            self.model_identity = model_identity

        def _write_raw(self) -> None:
            super()._write_raw()
            stages = tuple(
                full_fixtures.VALIDATOR.ACTIVE_BINDING_PHASE_ORDER
            )
            artifacts, _binding = active_binding_artifacts(
                output=self.final_path,
                candidate_source=candidate_path,
                candidate_raw=candidate_path.read_bytes(),
                claim=self.claim_value,
                campaign_name="sq8_full",
                run_id=full_fixtures.RUN_ID,
                stages=stages,
            )
            for relative in (
                "candidate-served-model.json",
                "active-manifest-observations.jsonl",
            ):
                self.bundle.write_bytes(
                    relative,
                    artifacts[relative],
                    scan=lambda _raw, _label: None,
                )

        def _render(self) -> None:
            renderer = full_fixtures.FullCampaignRenderer

            class V2Renderer:
                def render(self, context: object) -> dict[str, bytes]:
                    context.bundle_layout_version = "v2"
                    return renderer().render(context)

            full_fixtures.FullCampaignRenderer = V2Renderer
            try:
                super()._render()
            finally:
                full_fixtures.FullCampaignRenderer = renderer

        def validator(self) -> Any:
            return full_fixtures.VALIDATOR.FullCampaignIndependentValidator(
                expected_commit=source_commit,
                expected_worker_binary_sha256=worker_sha256,
                repo_root=promotion.source,
                forbidden_values=(b"never-present-real-bundle-token",),
                expected_served_model_manifest_sha256=digest(candidate_path),
                expected_authorization_claim_sha256=self.claim_value["sha256"],
                expected_authorization_sha256=self.claim_value[
                    "authorization_sha256"
                ],
            )

    rollback_manifest = root / "active-aq4.json"
    systemd_unit = root / "ullm-openai.service"
    environment_file = root / "ullm-openai.env"
    rollback_manifest.write_bytes(b"exact aq4 active manifest\n")
    systemd_unit.write_bytes(b"[Service]\nExecStart=/usr/bin/ullm\n")
    environment_file.write_bytes(b"ULLM_TEST=1\n")

    full_base = root / "full-real"
    full_base.mkdir()
    placeholder_claim = {
        "path": str(root / "campaign-authorization.claimed.json"),
        "sha256": "0" * 64,
        "bytes": 1,
        "authorization_path": str(root / "campaign-authorization.json"),
        "authorization_sha256": "0" * 64,
    }
    full_campaign = RealV2FullCampaign(
        full_base,
        claim_value=placeholder_claim,
    )
    authorization_path = root / "campaign-authorization.json"
    authorization = {
        "schema_version": BUNDLE.AUTHORIZATION_SCHEMA,
        "authorization_id": "real-validator-bundle-authorization",
        "issued_at": "2026-07-24T00:00:00Z",
        "expires_at": "2026-07-25T00:00:00Z",
        "max_attempts": 1,
        "authorization_note": "Real-validator bundle integration fixture.",
        "purpose": "temporary_candidate_active_evidence_collection_only",
        "required_final_route": "restore_exact_aq4_then_bundle_v2_activation",
        "source": {"commit": source_commit, "tree": promotion.tree},
        "before": {
            "model_id": "ullm-qwen3.5-9b-aq4",
            "format_id": "AQ4_0",
            "manifest_sha256": digest(rollback_manifest),
            "worker_binary_sha256": "9" * 64,
            "promotion_source_commit": "8" * 40,
        },
        "candidate": {
            "model_id": candidate["public"]["id"],
            "format_id": candidate["format"]["format_id"],
            "manifest_sha256": digest(candidate_path),
            "worker_protocol": candidate["worker"]["protocol"],
            "worker_binary_sha256": worker_sha256,
            "promotion_source_commit": source_commit,
            "promotion_receipt_sha256": digest(promotion.receipt),
        },
        "campaigns": {
            "sq8_full": {
                "run_id": full_fixtures.RUN_ID,
                "final_path": str(full_campaign.final_path),
            },
            "reasoning_release": {
                "run_id": "reasoning-release-real-run",
                "final_path": str(root / "reasoning-release-real"),
            },
            "reasoning_browser": {
                "run_id": "reasoning-browser-real-run",
                "final_path": str(root / "reasoning-browser-real"),
            },
        },
        "rollback": {
            "backup_path": str(root / "aq4-backup.json"),
            "systemd_unit_sha256": digest(systemd_unit),
            "environment_sha256": digest(environment_file),
        },
        "prior_outcome": None,
    }
    add_aq4_authorization_v2_bindings(
        authorization,
        root=root,
    )
    write_canonical_json(authorization_path, authorization)
    authorization_path.chmod(0o444)
    claim_path = root / "campaign-authorization.claimed.json"
    write_canonical_json(
        claim_path,
        {
            "schema_version": BUNDLE.CLAIM_SCHEMA,
            "authorization_id": authorization["authorization_id"],
            "authorization_path": str(authorization_path),
            "authorization_sha256": digest(authorization_path),
            "claimed_at": "2026-07-24T00:01:00Z",
            "attempt": 1,
            "max_attempts": 1,
        },
    )
    claim_path.chmod(0o444)
    claim = {
        "path": str(claim_path),
        "sha256": digest(claim_path),
        "bytes": len(claim_path.read_bytes()),
        "authorization_path": str(authorization_path),
        "authorization_sha256": digest(authorization_path),
    }
    full_campaign.claim_value = claim
    full_fixtures.WORKER_SHA256 = worker_sha256
    try:
        full_campaign.prepare()
        full_campaign.publish()
    finally:
        full_fixtures.WORKER_SHA256 = original_worker_sha256

    identity = {
        "manifest_sha256": digest(candidate_path),
        "worker_binary_sha256": worker_sha256,
        "tokenizer_sha256": _tokenizer_identity(candidate),
        "openwebui_image": AUTH.FIXED_OPENWEBUI_IMAGE,
    }
    (
        release_path,
        release_report_path,
        browser_path,
        browser_report_path,
    ) = _real_auxiliary_campaigns(
        root=root,
        candidate_path=candidate_path,
        claim=claim,
        identity=identity,
        source_commit=source_commit,
    )
    components = {
        "release_evidence": release_path,
        "release_validator": release_report_path,
        "browser_evidence": browser_path,
        "browser_validator": browser_report_path,
        "promotion_evidence": promotion.evidence,
        "promotion_receipt": promotion.receipt,
        "model_campaign_manifest": full_campaign.final_path / "SHA256SUMS",
        "model_campaign_evidence": full_campaign.final_path / "model-identity.json",
        "model_campaign_validator": (
            full_campaign.final_path / "release-validation.json"
        ),
    }
    bundle_path = root / "real-validator-bundle-v2.json"
    write_json(
        bundle_path,
        {
            "schema_version": BUNDLE.SCHEMA_VERSION_V2,
            "status": "complete",
            "production_activation_performed": False,
            "source_commit": source_commit,
            "active_promotion_source_commit": source_commit,
            "identity": identity,
            "artifacts": {
                name: {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": digest(path),
                }
                for name, path in components.items()
            },
            "rollback_target": {
                "manifest_sha256": digest(rollback_manifest),
                "systemd_unit_sha256": digest(systemd_unit),
                "environment_sha256": digest(environment_file),
            },
        },
    )
    seal_bundle_v2(bundle_path)
    return bundle_path, promotion


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


def test_aq4_bundle_v1_report_and_cli_bytes_remain_unchanged(
    tmp_path: Path,
) -> None:
    bundle = make_bundle(tmp_path)
    expected = {
        "schema_version": BUNDLE.VALIDATOR_SCHEMA_VERSION,
        "input_schema_version": BUNDLE.SCHEMA_VERSION,
        "structurally_valid": True,
        "gate_eligible": True,
        "source_commit": "1" * 40,
        "artifact_count": 6,
        "reasons": [],
    }

    assert BUNDLE.validate(bundle) == expected
    completed = subprocess.run(
        [sys.executable, os.fspath(BUNDLE_PATH), os.fspath(bundle)],
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0
    assert completed.stderr == b""
    assert completed.stdout == (
        json.dumps(expected, separators=(",", ":"), sort_keys=True).encode(
            "ascii"
        )
        + b"\n"
    )


@pytest.mark.parametrize(
    ("authorization_schema", "claim_schema"),
    (
        (
            "ullm.served_model.v2_cross_model_campaign_authorization.v1",
            BUNDLE.CLAIM_SCHEMA,
        ),
        (
            BUNDLE.AUTHORIZATION_SCHEMA,
            "ullm.served_model.v2_cross_model_campaign_claim.v1",
        ),
    ),
)
def test_bundle_v2_rejects_mixed_v1_v2_authorization_and_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authorization_schema: str,
    claim_schema: str,
) -> None:
    bundle, fakes, _report = make_v2_bundle(
        tmp_path,
        authorization_schema=authorization_schema,
        claim_schema=claim_schema,
    )
    install_v2_fakes(monkeypatch, fakes)

    with pytest.raises(
        BUNDLE.ValidationError,
        match="loaded campaign claim identity differs",
    ):
        BUNDLE.validate(bundle)


def test_bundle_v2_rejects_three_campaign_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, fakes, _report = make_v2_bundle(
        tmp_path,
        include_aq4_campaigns=False,
    )
    install_v2_fakes(monkeypatch, fakes)

    with pytest.raises(
        BUNDLE.ValidationError,
        match="campaign authorization validation failed",
    ):
        BUNDLE.validate(bundle)


def test_bundle_v2_rejects_sq8_full_run_not_selected_by_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, fakes, _report = make_v2_bundle(
        tmp_path,
        authorized_sq8_run_id="alternate-sq8-full-run",
    )
    install_v2_fakes(monkeypatch, fakes)

    with pytest.raises(
        BUNDLE.ValidationError,
        match="lineage differs from its authorization",
    ):
        BUNDLE.validate(bundle)


def test_bundle_v2_rejects_sq8_full_output_not_selected_by_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, fakes, _report = make_v2_bundle(
        tmp_path,
        authorized_sq8_final_path=tmp_path / "alternate-sq8-campaign",
    )
    install_v2_fakes(monkeypatch, fakes)

    with pytest.raises(
        BUNDLE.ValidationError,
        match="lineage differs from its authorization",
    ):
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
    assert result["bundle_sha256"] == digest(bundle)
    reasoning_campaign = result["reasoning_release_campaign"]
    assert reasoning_campaign["campaign_name"] == "reasoning_release"
    assert reasoning_campaign["run_id"] == "reasoning-release-run"
    assert reasoning_campaign["final_path"] == str(tmp_path / "reasoning-release")
    assert reasoning_campaign["artifact_count"] == 7
    assert reasoning_campaign["claim_sha256"] == digest(
        tmp_path / "campaign-authorization.claimed.json"
    )
    assert reasoning_campaign["authorization_sha256"] == digest(
        tmp_path / "campaign-authorization.json"
    )
    release_root = tmp_path / "reasoning-release"
    inventory = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in sorted(
            release_root.iterdir(),
            key=lambda value: value.name.encode("utf-8"),
        )
    ]
    assert reasoning_campaign["total_bytes"] == sum(
        item["bytes"] for item in inventory
    )
    assert reasoning_campaign["sha256"] == hashlib.sha256(
        (
            json.dumps(
                {"files": inventory},
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    ).hexdigest()
    assert (
        result["model_campaign_schema_version"]
        == "ullm.sq8.full_campaign.model_identity.v2"
    )


def test_bundle_v2_root_must_be_mode_0444_single_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, fakes, _report = make_v2_bundle(tmp_path)
    install_v2_fakes(monkeypatch, fakes)
    bundle.chmod(0o644)

    with pytest.raises(BUNDLE.ValidationError, match="mode-0444 single-link"):
        BUNDLE.validate(bundle)


def test_bundle_v2_root_rejects_hard_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, fakes, _report = make_v2_bundle(tmp_path)
    install_v2_fakes(monkeypatch, fakes)
    os.link(bundle, tmp_path / "bundle-v2-alias.json")

    with pytest.raises(BUNDLE.ValidationError, match="mode-0444 single-link"):
        BUNDLE.validate(bundle)


def test_bundle_v2_root_rejects_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, fakes, _report = make_v2_bundle(tmp_path)
    install_v2_fakes(monkeypatch, fakes)
    target = bundle.with_name("bundle-v2-target.json")
    bundle.rename(target)
    bundle.symlink_to(target)

    with pytest.raises(BUNDLE.ValidationError, match="regular non-symlink"):
        BUNDLE.validate(bundle)


def test_bundle_v2_root_rejects_mutation_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, fakes, _report = make_v2_bundle(tmp_path)
    install_v2_fakes(monkeypatch, fakes)
    original = BUNDLE._validate_v2

    def mutate(path: Path, document: dict[str, Any]) -> dict[str, Any]:
        report = original(path, document)
        changed = dict(document)
        changed["status"] = "incomplete"
        rewrite_bundle_v2(path, changed)
        return report

    monkeypatch.setattr(BUNDLE, "_validate_v2", mutate)

    with pytest.raises(BUNDLE.ValidationError, match="changed during validation"):
        BUNDLE.validate(bundle)


def test_bundle_v2_rejects_six_slot_v1_artifact_mix(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    value = json.loads(bundle.read_text(encoding="ascii"))
    value["schema_version"] = BUNDLE.SCHEMA_VERSION_V2
    write_json(bundle, value)
    seal_bundle_v2(bundle)

    with pytest.raises(BUNDLE.ValidationError, match="v2 artifacts differ"):
        BUNDLE.validate(bundle)


def test_bundle_v2_does_not_reinterpret_release_v1_as_lineage_v2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, fakes, _report = make_v2_bundle(tmp_path)
    install_v2_fakes(monkeypatch, fakes)
    value = json.loads(bundle.read_text(encoding="ascii"))
    release_path = tmp_path / value["artifacts"]["release_evidence"]["path"]
    release = json.loads(release_path.read_text(encoding="ascii"))
    release["schema_version"] = "ullm.generic_reasoning_release_evidence.v1"
    release.pop("campaign_lineage")
    write_json(release_path, release)
    value["artifacts"]["release_evidence"]["sha256"] = digest(release_path)
    rewrite_bundle_v2(bundle, value)

    with pytest.raises(BUNDLE.ValidationError, match="release evidence identity"):
        BUNDLE.validate(bundle)


def test_bundle_v2_rejects_cross_campaign_claim_mix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, fakes, _report = make_v2_bundle(tmp_path)
    install_v2_fakes(monkeypatch, fakes)
    value = json.loads(bundle.read_text(encoding="ascii"))
    release_path = tmp_path / value["artifacts"]["release_evidence"]["path"]
    release = json.loads(release_path.read_text(encoding="ascii"))
    release["campaign_lineage"]["claim"]["sha256"] = "0" * 64
    write_json(release_path, release)
    value["artifacts"]["release_evidence"]["sha256"] = digest(release_path)
    rewrite_bundle_v2(bundle, value)

    with pytest.raises(BUNDLE.ValidationError):
        BUNDLE.validate(bundle)


def test_bundle_v2_rejects_alternate_valid_release_lineage_not_in_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, fakes, _report = make_v2_bundle(tmp_path)
    install_v2_fakes(monkeypatch, fakes)
    bundle_document = json.loads(bundle.read_text(encoding="ascii"))
    release_path = (
        tmp_path / bundle_document["artifacts"]["release_evidence"]["path"]
    )
    release_report_path = (
        tmp_path / bundle_document["artifacts"]["release_validator"]["path"]
    )
    release = json.loads(release_path.read_text(encoding="ascii"))
    original_output = Path(release["campaign_lineage"]["campaign"]["final_path"])
    alternate_output = tmp_path / "alternate-reasoning-release"
    claim = release["campaign_lineage"]["claim"]
    candidate_path = tmp_path / "campaign/candidate-served-model.json"
    stages = tuple(RELEASE_FIXTURE.TOOL.REASONING_CAMPAIGN_STAGES)
    binding_artifacts, binding = active_binding_artifacts(
        output=alternate_output,
        candidate_source=candidate_path,
        candidate_raw=candidate_path.read_bytes(),
        claim=claim,
        campaign_name="reasoning_release",
        run_id="alternate-reasoning-release-run",
        stages=stages,
    )
    values = {
        "cases.json": (original_output / "cases.json").read_bytes(),
        "lifecycle.json": (original_output / "lifecycle.json").read_bytes(),
        "resource-samples.jsonl": (
            original_output / "resource-samples.jsonl"
        ).read_bytes(),
        **binding_artifacts,
    }
    summary = json.loads(
        (original_output / "summary.json").read_text(encoding="ascii")
    )
    summary["active_manifest_binding"] = binding
    summary["run_id"] = "alternate-reasoning-release-run"
    values["summary.json"] = (
        json.dumps(summary, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("ascii")
    release["campaign_lineage"] = campaign_lineage(
        output=alternate_output,
        campaign_name="reasoning_release",
        run_id="alternate-reasoning-release-run",
        claim=claim,
        artifacts=values,
        files=set(values),
        stages=stages,
    )
    write_immutable_directory(alternate_output, values)
    write_json(release_path, release)
    write_json(
        release_report_path,
        RELEASE_FIXTURE.TOOL.validate(release_path),
    )
    bundle_document["artifacts"]["release_evidence"]["sha256"] = digest(
        release_path
    )
    bundle_document["artifacts"]["release_validator"]["sha256"] = digest(
        release_report_path
    )
    rewrite_bundle_v2(bundle, bundle_document)

    with pytest.raises(
        BUNDLE.ValidationError,
        match="lineage differs from its authorization",
    ):
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
    browser_path.chmod(0o644)
    write_json(browser_path, browser)
    value["artifacts"]["browser_evidence"]["sha256"] = digest(browser_path)
    rewrite_bundle_v2(bundle, value)

    with pytest.raises(BUNDLE.ValidationError, match="browser v5 identity"):
        BUNDLE.validate(bundle)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda browser: browser.__setitem__(
            "browser_image",
            "sha256:" + "0" * 64,
        ),
        lambda browser: browser.__setitem__(
            "openwebui_server",
            {
                "before": {
                    "container_id": "1" * 64,
                    "image_id": "sha256:" + "0" * 64,
                    "config_image": "attacker/open-webui:fixed-looking",
                    "name": "/open-webui",
                    "running": True,
                    "pid": 1234,
                    "started_at": "2026-07-24T00:00:00.000000000Z",
                },
                "after": {
                    "container_id": "1" * 64,
                    "image_id": "sha256:" + "0" * 64,
                    "config_image": "attacker/open-webui:fixed-looking",
                    "name": "/open-webui",
                    "running": True,
                    "pid": 1234,
                    "started_at": "2026-07-24T00:00:00.000000000Z",
                },
            },
        ),
    ),
)
def test_bundle_v2_rejects_unfixed_browser_or_server_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
) -> None:
    bundle, fakes, _report = make_v2_bundle(tmp_path)
    install_v2_fakes(monkeypatch, fakes)
    value = json.loads(bundle.read_text(encoding="ascii"))
    browser_path = tmp_path / value["artifacts"]["browser_evidence"]["path"]
    browser = json.loads(browser_path.read_text(encoding="ascii"))
    mutate(browser)
    browser_path.chmod(0o644)
    write_json(browser_path, browser)
    browser_path.chmod(0o444)
    value["artifacts"]["browser_evidence"]["sha256"] = digest(browser_path)
    rewrite_bundle_v2(bundle, value)

    with pytest.raises(
        BUNDLE.ValidationError,
        match="image identities differ from authorization",
    ):
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
    rewrite_bundle_v2(bundle, value)

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
    rewrite_bundle_v2(bundle, value)

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
    rewrite_bundle_v2(bundle, value)

    with pytest.raises(BUNDLE.ValidationError, match="artifact locations differ"):
        BUNDLE.validate(bundle)


def test_bundle_v2_real_validators_and_build_receipt_mutation_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, promotion = make_real_validator_v2_bundle(tmp_path)
    monkeypatch.setattr(BUNDLE, "ROOT", promotion.source)

    report = BUNDLE.validate(bundle)

    assert report["gate_eligible"] is True
    assert report["bundle_sha256"] == digest(bundle)
    assert report["reasoning_release_campaign"]["artifact_count"] == 7
    assert (
        report["model_campaign_schema_version"]
        == "ullm.sq8.full_campaign.model_identity.v2"
    )

    build_receipt = promotion.build_receipt
    original = build_receipt.read_bytes()
    alias = build_receipt.with_name("build-receipt-hardlink.json")
    mutations = (
        (
            "schema",
            lambda value: value.__setitem__(
                "schema_version",
                "ullm.sq8_worker_build_receipt.invalid",
            ),
        ),
        (
            "source-commit",
            lambda value: value["source"].__setitem__("commit", "0" * 40),
        ),
        (
            "build-argv",
            lambda value: value["build"]["argv"].append("--mutated"),
        ),
        (
            "build-environment",
            lambda value: value["build"]["environment"].__setitem__(
                "HIP_VISIBLE_DEVICES",
                "0",
            ),
        ),
        (
            "worker-sha256",
            lambda value: value["worker"].__setitem__("sha256", "0" * 64),
        ),
    )
    for _label, mutate in mutations:
        build_receipt.chmod(0o644)
        build_receipt.write_bytes(original)
        value = json.loads(original)
        mutate(value)
        write_canonical_json(build_receipt, value)
        build_receipt.chmod(0o444)
        with pytest.raises(
            BUNDLE.ValidationError,
            match="SQ8 promotion receipt validation failed",
        ):
            BUNDLE.validate(bundle)

    build_receipt.chmod(0o644)
    build_receipt.write_bytes(original)
    with pytest.raises(
        BUNDLE.ValidationError,
        match="SQ8 promotion receipt validation failed",
    ):
        BUNDLE.validate(bundle)
    build_receipt.chmod(0o444)

    os.link(build_receipt, alias)
    try:
        with pytest.raises(
            BUNDLE.ValidationError,
            match="SQ8 promotion receipt validation failed",
        ):
            BUNDLE.validate(bundle)
    finally:
        alias.unlink()

    assert stat.S_IMODE(build_receipt.stat().st_mode) == 0o444
    assert build_receipt.stat().st_nlink == 1
    assert build_receipt.read_bytes() == original
