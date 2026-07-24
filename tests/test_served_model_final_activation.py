from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/served_model_final_activation.py"
SPEC = importlib.util.spec_from_file_location(
    "test_served_model_final_activation_module",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
FINAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FINAL
SPEC.loader.exec_module(FINAL)
AUTH = FINAL.authorization

RUNNER_PATH = ROOT / "tools/run-served-model-final-activation.py"
ROLLBACK_PATH = ROOT / "tools/rollback-served-model.py"
PREPARE_PATH = ROOT / "tools/prepare-served-model-final-activation.py"
RUNBOOK_PATH = ROOT / "docs/plans/sq8-final-activation-operator-runbook-v0.1.md"

REAL_CAPTURE_SOURCE_ROOT = FINAL._capture_source_root
REAL_REQUIRE_PRODUCTION_ENTRYPOINT = FINAL.require_production_entrypoint

NOW = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)
SOURCE_COMMIT = "a" * 40
SOURCE_TREE = "b" * 40
AQ4_SOURCE_COMMIT = "c" * 40
AQ4_SOURCE_TREE = "d" * 40
AQ4_WORKER_RAW = b"fixture-aq4-worker\n"
SQ8_WORKER_RAW = b"fixture-sq8-worker\n"
AQ4_WORKER = hashlib.sha256(AQ4_WORKER_RAW).hexdigest()
SQ8_WORKER = hashlib.sha256(SQ8_WORKER_RAW).hexdigest()


@pytest.fixture(autouse=True)
def stub_execution_source_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Keep fixture tests hermetic; dedicated tests exercise the real Git seal."""

    sealed = SimpleNamespace(
        root=FINAL.ROOT,
        required_uid=os.geteuid(),
        entries=(),
        fingerprint_sha256="e" * 64,
    )

    def capture(
        *,
        expected_commit: str,
        expected_tree: str,
        required_uid: int,
    ) -> SimpleNamespace:
        assert len(expected_commit) == 40
        assert len(expected_tree) == 40
        sealed.required_uid = required_uid
        return sealed

    def require(
        expected: object,
        *,
        expected_commit: str,
        expected_tree: str,
        required_uid: int,
    ) -> None:
        assert expected is sealed
        assert len(expected_commit) == 40
        assert len(expected_tree) == 40
        assert sealed.required_uid == required_uid

    monkeypatch.setattr(FINAL, "_capture_execution_source", capture)
    monkeypatch.setattr(FINAL, "_require_execution_source", require)
    return sealed


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def load_cli(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Fixture:
    def __init__(
        self,
        tmp_path: Path,
        *,
        aq4_mutator: Callable[["Fixture"], None] | None = None,
    ) -> None:
        self.root = tmp_path
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.root.chmod(0o700)
        self.registry = tmp_path / "registry"
        self.claims = self.registry / "claims"
        self.outcomes = self.registry / "campaign-outcomes"
        self.claims.mkdir(parents=True, mode=0o700)
        self.registry.chmod(0o700)
        self.claims.chmod(0o700)
        self.outcomes.mkdir(mode=0o700)
        self.outcomes.chmod(0o700)
        self.slot = tmp_path / "slot"
        self.unit = self.slot / "ullm-openai.service"
        self.environment = self.slot / "ullm-openai.env"
        self.policy = AUTH.RegistryPolicy(
            claim_registry=self.claims,
            outcome_registry=self.outcomes,
            required_uid=os.geteuid(),
            active_manifest_path=tmp_path / "slot/active.json",
            systemd_unit_path=self.unit,
            environment_file_path=self.environment,
            service_unit="ullm-openai.service",
        )
        self.slot.mkdir(mode=0o700)
        self.release_root = tmp_path / "release"
        self.release_root.mkdir(mode=0o700)
        (self.release_root / "campaigns").mkdir(mode=0o700)
        self.aq4_bundle_root = self.release_root / "aq4-bundle"
        self.aq4_bundle_root.mkdir(mode=0o700)
        self.aq4_source_root = tmp_path / "aq4-source"
        self.aq4_source_root.mkdir(mode=0o700)
        self.final_outcomes = tmp_path / "final-outcomes"
        self.final_outcomes.mkdir(mode=0o700)

        self.aq4_worker = self.release_root / "aq4-worker"
        self.aq4_worker.write_bytes(AQ4_WORKER_RAW)
        self.sq8_worker = self.release_root / "sq8-worker"
        self.sq8_worker.write_bytes(SQ8_WORKER_RAW)
        self.aq4_tokenizer = tmp_path / "aq4-tokenizer"
        self.aq4_tokenizer.mkdir(mode=0o700)
        self.aq4_tokenizer_file = self.aq4_tokenizer / "tokenizer.json"
        self.aq4_tokenizer_file.write_bytes(b'{"tokenizer":"aq4"}\n')
        self.sq8_tokenizer = tmp_path / "sq8-tokenizer"
        self.sq8_tokenizer.mkdir(mode=0o700)
        self.sq8_tokenizer_file = self.sq8_tokenizer / "tokenizer.json"
        self.sq8_tokenizer_file.write_bytes(b'{"tokenizer":"sq8"}\n')

        self.aq4_product = tmp_path / "aq4-product"
        (self.aq4_product / "package").mkdir(parents=True, mode=0o700)
        self.aq4_package_manifest = self.aq4_product / "package/manifest.json"
        self.aq4_package_manifest.write_bytes(b'{"package":"aq4"}\n')
        self.sq8_product = self.release_root / "sq8-product"
        (self.sq8_product / "package").mkdir(parents=True, mode=0o700)
        (self.sq8_product / "artifact").mkdir(mode=0o700)
        self.sq8_package_manifest = self.sq8_product / "package/manifest.json"
        self.sq8_package_manifest.write_bytes(b'{"package":"sq8"}\n')
        self.sq8_artifact_manifest = self.sq8_product / "artifact/manifest.json"
        self.sq8_artifact_manifest.write_bytes(b'{"artifact":"sq8"}\n')

        self.aq4_promotion_source = self.aq4_product
        self.aq4_source_promotion_evidence = (
            self.aq4_promotion_source / "promotion-evidence.json"
        )
        self.aq4_source_promotion_evidence.write_bytes(
            canonical(
                {
                    "schema_version": "ullm.aq4_resident_promotion_evidence.v1",
                    "source_commit": AQ4_SOURCE_COMMIT,
                    "worker_binary": os.fspath(self.aq4_worker),
                    "worker_binary_sha256": AQ4_WORKER,
                }
            )
        )
        self.aq4_receipt = (
            self.aq4_promotion_source / "promotion-receipt.json"
        )
        self.aq4_receipt.write_bytes(
            canonical(
                {
                    "schema_version": "ullm.aq4_resident_promotion.v1",
                    "source_commit": AQ4_SOURCE_COMMIT,
                    "evidence": {
                        "path": self.aq4_source_promotion_evidence.name,
                        "sha256": digest(
                            self.aq4_source_promotion_evidence.read_bytes()
                        ),
                    },
                }
            )
        )
        self.aq4_promotion_evidence = (
            self.aq4_bundle_root / "promotion-evidence.json"
        )
        self.aq4_bundle_receipt = (
            self.aq4_bundle_root / "promotion-receipt.json"
        )
        self.sq8_promotion_evidence = (
            self.sq8_product / "sq8-serving-promotion-evidence.json"
        )
        self.sq8_promotion_evidence.write_bytes(
            canonical(
                {
                    "schema_version": "ullm.sq8_serving_promotion_evidence.v1",
                    "source_commit": SOURCE_COMMIT,
                    "worker_binary_sha256": SQ8_WORKER,
                }
            )
        )
        self.receipt = self.sq8_product / "sq8-promotion-receipt.json"
        self.receipt.write_bytes(
            canonical(
                {
                    "schema_version": "ullm.sq8_serving_promotion.v1",
                    "source_commit": SOURCE_COMMIT,
                    "evidence": {
                        "path": self.sq8_promotion_evidence.name,
                        "sha256": digest(self.sq8_promotion_evidence.read_bytes()),
                    },
                }
            )
        )
        self.aq4_raw = canonical(
            {
                "schema_version": FINAL.SERVED_MODEL_SCHEMA,
                "tokenizer": {
                    "root": os.fspath(self.aq4_tokenizer),
                    "files": {
                        self.aq4_tokenizer_file.name: digest(
                            self.aq4_tokenizer_file.read_bytes()
                        )
                    },
                },
                "worker": {
                    "protocol": FINAL.WORKER_PROTOCOL,
                    "binary": os.fspath(self.aq4_worker),
                    "binary_sha256": AQ4_WORKER,
                },
                "product": {
                    "root": os.fspath(self.aq4_product),
                    "artifact": None,
                    "package": {
                        "manifest_path": "package/manifest.json",
                        "manifest_sha256": digest(
                            self.aq4_package_manifest.read_bytes()
                        ),
                    },
                },
                "promotion": {
                    "source_commit": AQ4_SOURCE_COMMIT,
                    "receipt": os.fspath(self.aq4_receipt),
                    "receipt_sha256": digest(self.aq4_receipt.read_bytes()),
                },
            }
        )
        self.sq8_raw = canonical(
            {
                "schema_version": FINAL.SERVED_MODEL_SCHEMA,
                "tokenizer": {
                    "root": os.fspath(self.sq8_tokenizer),
                    "files": {
                        self.sq8_tokenizer_file.name: digest(
                            self.sq8_tokenizer_file.read_bytes()
                        )
                    },
                },
                "worker": {
                    "protocol": FINAL.WORKER_PROTOCOL,
                    "binary": os.fspath(self.sq8_worker),
                    "binary_sha256": SQ8_WORKER,
                },
                "product": {
                    "root": os.fspath(self.sq8_product),
                    "artifact": {
                        "manifest_path": "artifact/manifest.json",
                        "manifest_sha256": digest(
                            self.sq8_artifact_manifest.read_bytes()
                        ),
                    },
                    "package": {
                        "manifest_path": "package/manifest.json",
                        "manifest_sha256": digest(
                            self.sq8_package_manifest.read_bytes()
                        ),
                    },
                },
                "promotion": {
                    "source_commit": SOURCE_COMMIT,
                    "receipt": os.fspath(self.receipt),
                    "receipt_sha256": digest(self.receipt.read_bytes()),
                },
            }
        )
        self.active = self.slot / "active.json"
        self.active.write_bytes(self.aq4_raw)
        self.active.chmod(0o644)
        self.candidate = self.release_root / "candidate.json"
        self.candidate.write_bytes(self.sq8_raw)
        self.unit.write_bytes(b"[Service]\nExecStart=/opt/ullm/worker\n")
        self.environment.write_bytes(b"ULLM_TEST=1\n")
        self.rollback = self.release_root / "aq4-rollback.json"

        self.campaign_paths = {
            "aq4_reasoning_release": (
                self.release_root / "campaigns/aq4-reasoning-release"
            ),
            "aq4_reasoning_browser": (
                self.aq4_bundle_root / "browser-evidence.json"
            ),
            "aq4_bundle": self.aq4_bundle_root / "release-bundle-v1.json",
            "sq8_full": self.release_root / "campaigns/sq8-full",
            "reasoning_release": self.release_root / "campaigns/reasoning-release",
            "reasoning_browser": self.release_root / "campaigns/reasoning-browser",
        }
        self.aq4_release_evidence = (
            self.aq4_bundle_root / "release-evidence.json"
        )
        self.aq4_release_validator = (
            self.aq4_bundle_root / "release-validator.json"
        )
        self.aq4_browser_validator = (
            self.aq4_bundle_root / "browser-validator.json"
        )
        self.authorization_path = self.registry / "authorization.json"
        self.authorization_document = {
            "schema_version": AUTH.AUTHORIZATION_SCHEMA,
            "authorization_id": "sq8-final-route-test-001",
            "issued_at": AUTH.utc_timestamp(NOW - timedelta(minutes=1)),
            "expires_at": AUTH.utc_timestamp(NOW + timedelta(hours=1)),
            "max_attempts": 1,
            "authorization_note": "Private final activation fixture.",
            "purpose": "temporary_candidate_active_evidence_collection_only",
            "required_final_route": "restore_exact_aq4_then_bundle_v2_activation",
            "source": {"commit": SOURCE_COMMIT, "tree": SOURCE_TREE},
            "before": {
                "model_id": FINAL.AQ4_MODEL_ID,
                "format_id": FINAL.AQ4_FORMAT_ID,
                "manifest_sha256": digest(self.aq4_raw),
                "worker_protocol": FINAL.WORKER_PROTOCOL,
                "worker_binary_path": os.fspath(self.aq4_worker),
                "worker_binary_sha256": AQ4_WORKER,
                "promotion_source_commit": AQ4_SOURCE_COMMIT,
                "promotion_receipt_path": os.fspath(self.aq4_receipt),
                "promotion_receipt_sha256": digest(
                    self.aq4_receipt.read_bytes()
                ),
            },
            "aq4_release": {
                "source": {
                    "root": os.fspath(self.aq4_source_root),
                    "commit": AQ4_SOURCE_COMMIT,
                    "tree": AQ4_SOURCE_TREE,
                },
                "openwebui_image": AUTH.FIXED_OPENWEBUI_IMAGE,
                "promotion_evidence": {
                    "source_path": os.fspath(
                        self.aq4_source_promotion_evidence
                    ),
                    "path": os.fspath(self.aq4_promotion_evidence),
                    "sha256": digest(
                        self.aq4_source_promotion_evidence.read_bytes()
                    ),
                },
                "promotion_receipt": {
                    "source_path": os.fspath(self.aq4_receipt),
                    "path": os.fspath(self.aq4_bundle_receipt),
                    "sha256": digest(self.aq4_receipt.read_bytes()),
                },
                "release_evidence_path": os.fspath(
                    self.aq4_release_evidence
                ),
                "release_validator_path": os.fspath(
                    self.aq4_release_validator
                ),
                "browser_validator_path": os.fspath(
                    self.aq4_browser_validator
                ),
            },
            "candidate": {
                "model_id": FINAL.SQ8_MODEL_ID,
                "format_id": FINAL.SQ8_FORMAT_ID,
                "manifest_sha256": digest(self.sq8_raw),
                "worker_protocol": FINAL.WORKER_PROTOCOL,
                "worker_binary_sha256": SQ8_WORKER,
                "promotion_source_commit": SOURCE_COMMIT,
                "promotion_receipt_sha256": digest(self.receipt.read_bytes()),
            },
            "campaigns": {
                name: {
                    "run_id": f"{name}-run-001",
                    "final_path": os.fspath(path),
                }
                for name, path in self.campaign_paths.items()
            },
            "rollback": {
                "backup_path": os.fspath(self.rollback),
                "systemd_unit_sha256": digest(self.unit.read_bytes()),
                "environment_sha256": digest(self.environment.read_bytes()),
            },
            "prior_outcome": None,
        }
        AUTH.issue_authorization(
            self.authorization_document,
            self.authorization_path,
            now=NOW,
            policy=self.policy,
        )
        claim = AUTH.claim_authorization(
            self.authorization_path,
            now=NOW,
            policy=self.policy,
        )
        self.claim_reference = {
            "path": os.fspath(claim.snapshot.path),
            "sha256": claim.snapshot.sha256,
            "bytes": len(claim.snapshot.raw),
            "authorization_path": os.fspath(claim.authorization.snapshot.path),
            "authorization_sha256": claim.authorization.snapshot.sha256,
        }

        self.rollback.write_bytes(self.aq4_raw)
        self.rollback.chmod(0o444)
        self.aq4_promotion_evidence.write_bytes(
            self.aq4_source_promotion_evidence.read_bytes()
        )
        self.aq4_bundle_receipt.write_bytes(self.aq4_receipt.read_bytes())
        aq4_release = self.campaign_paths["aq4_reasoning_release"]
        aq4_release.mkdir(parents=True)
        self.aq4_cases = [
            {
                "case_id": "aq4-case-001",
                "mode": "disabled",
                "passed": True,
            }
        ]
        self.aq4_lifecycle = {
            "schema_version": "ullm.generic_reasoning_lifecycle_evidence.v1",
            "events": [{"case_id": "aq4-case-001", "passed": True}],
        }
        (aq4_release / "cases.json").write_bytes(canonical(self.aq4_cases))
        (aq4_release / "lifecycle.json").write_bytes(
            canonical(self.aq4_lifecycle)
        )
        aq4_browser = self.campaign_paths["aq4_reasoning_browser"]
        aq4_browser.write_bytes(
            canonical(
                {
                    "schema_version": (
                        "ullm.openwebui.reasoning_browser_smoke.v2"
                    ),
                    "status": "complete",
                    "model_id": FINAL.AQ4_MODEL_ID,
                }
            )
        )
        self.aq4_release_evidence.write_bytes(
            canonical(
                {
                    "schema_version": (
                        "ullm.generic_reasoning_release_evidence.v1"
                    ),
                    "status": "complete",
                    "production_activation_performed": False,
                    "source_commit": AQ4_SOURCE_COMMIT,
                    "active_promotion_source_commit": AQ4_SOURCE_COMMIT,
                    "identity": {
                        "manifest_sha256": digest(self.aq4_raw),
                        "worker_binary_sha256": AQ4_WORKER,
                        "tokenizer_sha256": "8" * 64,
                        "openwebui_image": AUTH.FIXED_OPENWEBUI_IMAGE,
                    },
                    "cases": self.aq4_cases,
                    "lifecycle": self.aq4_lifecycle,
                }
            )
        )
        self.aq4_release_validator.write_text(
            '{"gate_eligible":true}\n',
            encoding="ascii",
        )
        self.aq4_browser_validator.write_text(
            '{"gate_eligible":true}\n',
            encoding="ascii",
        )
        aq4_artifacts = {
            "release_evidence": self.aq4_release_evidence,
            "release_validator": self.aq4_release_validator,
            "browser_evidence": aq4_browser,
            "browser_validator": self.aq4_browser_validator,
            "promotion_evidence": self.aq4_promotion_evidence,
            "promotion_receipt": self.aq4_bundle_receipt,
        }
        self.aq4_bundle_document = {
            "schema_version": FINAL.AQ4_BUNDLE_SCHEMA,
            "status": "complete",
            "production_activation_performed": False,
            "source_commit": AQ4_SOURCE_COMMIT,
            "active_promotion_source_commit": AQ4_SOURCE_COMMIT,
            "identity": {
                "manifest_sha256": digest(self.aq4_raw),
                "worker_binary_sha256": AQ4_WORKER,
                "tokenizer_sha256": "8" * 64,
                "openwebui_image": AUTH.FIXED_OPENWEBUI_IMAGE,
            },
            "artifacts": {
                name: {
                    "path": path.relative_to(self.aq4_bundle_root).as_posix(),
                    "sha256": digest(path.read_bytes()),
                }
                for name, path in aq4_artifacts.items()
            },
            "rollback_target": {
                "manifest_sha256": "7" * 64,
                "systemd_unit_sha256": digest(self.unit.read_bytes()),
                "environment_sha256": digest(self.environment.read_bytes()),
            },
        }
        self.aq4_bundle = self.campaign_paths["aq4_bundle"]
        self.aq4_bundle.write_bytes(canonical(self.aq4_bundle_document))
        if aq4_mutator is not None:
            aq4_mutator(self)
        for path in (
            aq4_release / "cases.json",
            aq4_release / "lifecycle.json",
            aq4_browser,
            self.aq4_release_evidence,
            self.aq4_release_validator,
            self.aq4_browser_validator,
            self.aq4_promotion_evidence,
            self.aq4_bundle_receipt,
            self.aq4_bundle,
        ):
            path.chmod(0o444)
        aq4_release.chmod(0o555)

        full = self.campaign_paths["sq8_full"]
        full.mkdir(parents=True)
        (full / "SHA256SUMS").write_text("fixture checksums\n", encoding="ascii")
        (full / "model-identity.json").write_bytes(
            canonical({"campaign_authorization_claim": self.claim_reference})
        )
        (full / "release-validation.json").write_text("{}\n", encoding="ascii")
        release = self.campaign_paths["reasoning_release"]
        release.mkdir()
        (release / "summary.json").write_text("{}\n", encoding="ascii")
        browser = self.campaign_paths["reasoning_browser"]
        browser.mkdir()
        browser_evidence = browser / "browser-evidence.json"
        browser_evidence.write_bytes(
            canonical(
                {
                    "schema_version": "ullm.openwebui.reasoning_browser_smoke.v5",
                    "browser_image": AUTH.FIXED_BROWSER_IMAGE,
                    "openwebui_server": {
                        "before": {
                            "container_id": "1" * 64,
                            "image_id": AUTH.FIXED_OPENWEBUI_IMAGE.rsplit(
                                "@",
                                1,
                            )[1],
                            "config_image": AUTH.FIXED_OPENWEBUI_CONFIG_IMAGE,
                            "name": (
                                f"/{AUTH.FIXED_OPENWEBUI_CONTAINER_NAME}"
                            ),
                            "running": True,
                            "pid": 1234,
                            "started_at": (
                                "2026-07-24T00:00:00.000000000Z"
                            ),
                        },
                        "after": {
                            "container_id": "1" * 64,
                            "image_id": AUTH.FIXED_OPENWEBUI_IMAGE.rsplit(
                                "@",
                                1,
                            )[1],
                            "config_image": AUTH.FIXED_OPENWEBUI_CONFIG_IMAGE,
                            "name": (
                                f"/{AUTH.FIXED_OPENWEBUI_CONTAINER_NAME}"
                            ),
                            "running": True,
                            "pid": 1234,
                            "started_at": (
                                "2026-07-24T00:00:00.000000000Z"
                            ),
                        },
                    },
                    "campaign_lineage": {
                        "schema_version": "ullm.served_model.campaign_lineage.v2",
                        "claim": self.claim_reference,
                        "campaign": {
                            "name": "reasoning_browser",
                            "run_id": self.authorization_document["campaigns"][
                                "reasoning_browser"
                            ]["run_id"],
                            "final_path": os.fspath(browser),
                            "final_kind": "directory",
                        },
                    },
                }
            )
        )
        campaign_results = {
            name: FINAL._output_inventory(
                path,
                run_id=self.authorization_document["campaigns"][name]["run_id"],
            )
            for name, path in self.campaign_paths.items()
        }
        restoration_proof = {
            "schema_version": ("ullm.served_model.v2_cross_model_restoration_proof.v1"),
            "authorization_sha256": claim.authorization.snapshot.sha256,
            "claim_sha256": claim.snapshot.sha256,
            "captured_at": AUTH.utc_timestamp(NOW + timedelta(seconds=2)),
            "active_manifest": {
                "path": os.fspath(self.active),
                "expected_sha256": digest(self.aq4_raw),
                "observed_sha256": digest(self.aq4_raw),
                "bytes_equal": True,
            },
            "service": {
                "unit": "ullm-openai.service",
                "active_state": "active",
                "sub_state": "running",
                "boot_id": "11111111-2222-3333-4444-555555555555",
                "n_restarts": 0,
            },
            "gateway": {
                "pid": 100,
                "ppid": 0,
                "starttime_ticks": 1000,
                "executable_sha256": "7" * 64,
            },
            "worker": {
                "pid": 101,
                "ppid": 100,
                "starttime_ticks": 1001,
                "executable_sha256": AQ4_WORKER,
            },
            "endpoints": {
                "gateway_healthz": {"status": 200},
                "gateway_readyz": {"status": 200},
                "gateway_models": {
                    "status": 200,
                    "model_ids": [FINAL.AQ4_MODEL_ID],
                },
                "openwebui_health": {"status": 200},
                "openwebui_models": {
                    "status": 200,
                    "model_ids": [FINAL.AQ4_MODEL_ID],
                },
            },
            "epoch_stable": True,
            "passed": True,
        }
        outcome_document = {
            "schema_version": AUTH.OUTCOME_SCHEMA,
            "authorization_id": self.authorization_document["authorization_id"],
            "authorization_path": os.fspath(claim.authorization.snapshot.path),
            "authorization_sha256": claim.authorization.snapshot.sha256,
            "claim_path": os.fspath(claim.snapshot.path),
            "claim_sha256": claim.snapshot.sha256,
            "started_at": AUTH.utc_timestamp(NOW + timedelta(seconds=1)),
            "completed_at": AUTH.utc_timestamp(NOW + timedelta(seconds=2)),
            "status": "succeeded_restored",
            "failure_stage": None,
            "stages": {name: "passed" for name in AUTH.OUTCOME_STAGE_FIELDS},
            "aq4_observations": [
                {
                    "stage": stage,
                    "active_manifest_sha256": digest(self.aq4_raw),
                    "bytes_equal": True,
                }
                for stage in AUTH.AQ4_OBSERVATION_STAGES
            ],
            "candidate_observations": [
                {
                    "stage": stage,
                    "active_manifest_sha256": digest(self.sq8_raw),
                    "bytes_equal": True,
                }
                for stage in AUTH.CANDIDATE_OBSERVATION_STAGES
            ],
            "campaigns": campaign_results,
            "restoration": {
                "expected_manifest_sha256": digest(self.aq4_raw),
                "observed_manifest_sha256": digest(self.aq4_raw),
                "displaced_manifest_sha256": digest(self.sq8_raw),
                "bytes_equal": True,
                "reverse_reconciliation_passed": True,
                "final_checks_passed": True,
                "model_id": FINAL.AQ4_MODEL_ID,
                "format_id": FINAL.AQ4_FORMAT_ID,
                "worker_binary_sha256": AQ4_WORKER,
                "proof": restoration_proof,
            },
        }
        self.campaign_outcome = AUTH.publish_outcome(
            claim,
            outcome_document,
            policy=self.policy,
        )

        derived = self.release_root / "derived"
        derived.mkdir()
        component_contents = {
            "release-validator.json": b'{"release_validator":true}\n',
            "browser-validator.json": b'{"browser_validator":true}\n',
            "promotion-evidence.json": b'{"promotion":true}\n',
        }
        for name, raw in component_contents.items():
            (derived / name).write_bytes(raw)
        self.release_evidence = derived / "release-evidence.json"
        self.release_evidence.write_bytes(
            canonical(
                {
                    "campaign_lineage": {
                        "schema_version": "ullm.served_model.campaign_lineage.v2",
                        "claim": self.claim_reference,
                        "artifact_inventory_sha256": "8" * 64,
                        "campaign": {
                            "name": "reasoning_release",
                            "run_id": self.authorization_document["campaigns"][
                                "reasoning_release"
                            ]["run_id"],
                            "final_path": os.fspath(release),
                            "final_kind": "directory",
                        },
                    }
                }
            )
        )
        artifacts = {
            "release_evidence": self.release_evidence,
            "release_validator": derived / "release-validator.json",
            "browser_evidence": browser_evidence,
            "browser_validator": derived / "browser-validator.json",
            "promotion_evidence": derived / "promotion-evidence.json",
            "promotion_receipt": self.receipt,
            "model_campaign_manifest": full / "SHA256SUMS",
            "model_campaign_evidence": full / "model-identity.json",
            "model_campaign_validator": full / "release-validation.json",
        }
        self.bundle_document = {
            "schema_version": FINAL.BUNDLE_SCHEMA,
            "status": "complete",
            "production_activation_performed": False,
            "source_commit": SOURCE_COMMIT,
            "active_promotion_source_commit": SOURCE_COMMIT,
            "identity": {
                "manifest_sha256": digest(self.sq8_raw),
                "worker_binary_sha256": SQ8_WORKER,
                "tokenizer_sha256": "5" * 64,
                "openwebui_image": AUTH.FIXED_OPENWEBUI_IMAGE,
            },
            "artifacts": {
                name: {
                    "path": path.relative_to(self.release_root).as_posix(),
                    "sha256": digest(path.read_bytes()),
                }
                for name, path in artifacts.items()
            },
            "rollback_target": {
                "manifest_sha256": digest(self.aq4_raw),
                "systemd_unit_sha256": digest(self.unit.read_bytes()),
                "environment_sha256": digest(self.environment.read_bytes()),
            },
        }
        self.bundle = self.release_root / "release-bundle.json"
        self.bundle.write_bytes(canonical(self.bundle_document))

        self.executable = self.release_root / "reviewed-operation"
        self.executable.write_bytes(Path("/usr/bin/true").read_bytes())
        self.executable.chmod(0o755)
        executable_hash = digest(self.executable.read_bytes())
        self.operations_document = {
            "schema_version": FINAL.OPERATIONS_SCHEMA,
            "review_id": "final-ops-review-001",
            "reviewed_at": AUTH.utc_timestamp(NOW),
            "reviewed_by": "fixture-reviewer",
            "timeout_seconds": 30,
            "active_window_timeout_seconds": 300,
            "live_proofs": {
                "candidate_live_health": {
                    "path": os.fspath(
                        self.final_outcomes / "candidate-live-proof.json"
                    ),
                    "service_unit": "ullm-openai.service",
                    "gateway_executable_sha256": "7" * 64,
                    "endpoint_urls": {
                        name: f"http://127.0.0.1:19001/{name}"
                        for name in FINAL.ENDPOINT_NAMES
                    },
                },
                "rollback_live_health": {
                    "path": os.fspath(self.final_outcomes / "rollback-live-proof.json"),
                    "service_unit": "ullm-openai.service",
                    "gateway_executable_sha256": "7" * 64,
                    "endpoint_urls": {
                        name: f"http://127.0.0.1:19002/{name}"
                        for name in FINAL.ENDPOINT_NAMES
                    },
                },
            },
            "stages": {
                stage: [
                    {
                        "argv": [os.fspath(self.executable), stage],
                        "executable_sha256": executable_hash,
                    }
                ]
                for stage in FINAL.OPERATION_STAGES
            },
        }
        self.operations = self.release_root / "reviewed-operations.json"
        self.operations.write_bytes(canonical(self.operations_document))
        self.operations.chmod(0o444)
        self.plan = self.release_root / "final-activation-plan.json"
        self.activation_outcome = self.final_outcomes / "activation-outcome.json"
        self.rollback_outcome = self.final_outcomes / "rollback-outcome.json"
        for runtime_file in (
            self.aq4_worker,
            self.sq8_worker,
            self.aq4_tokenizer_file,
            self.sq8_tokenizer_file,
            self.aq4_package_manifest,
            self.sq8_package_manifest,
            self.sq8_artifact_manifest,
            self.aq4_source_promotion_evidence,
            self.aq4_receipt,
            self.sq8_promotion_evidence,
            self.receipt,
            self.candidate,
            self.unit,
            self.environment,
        ):
            runtime_file.chmod(0o644)
        for runtime_directory in (
            self.aq4_product,
            self.aq4_product / "package",
            self.sq8_product,
            self.sq8_product / "package",
            self.sq8_product / "artifact",
        ):
            runtime_directory.chmod(0o700)

    def manifest_validator(self, path: Path) -> dict[str, object]:
        raw = path.read_bytes()
        if raw == self.sq8_raw:
            return {
                "validated": True,
                "manifest_sha256": digest(raw),
                "model_id": FINAL.SQ8_MODEL_ID,
                "format_id": FINAL.SQ8_FORMAT_ID,
                "worker": {
                    "protocol": FINAL.WORKER_PROTOCOL,
                    "binary_sha256": SQ8_WORKER,
                },
            }
        if raw == self.aq4_raw:
            return {
                "validated": True,
                "manifest_sha256": digest(raw),
                "model_id": FINAL.AQ4_MODEL_ID,
                "format_id": FINAL.AQ4_FORMAT_ID,
                "worker": {
                    "protocol": FINAL.WORKER_PROTOCOL,
                    "binary_sha256": AQ4_WORKER,
                },
            }
        raise ValueError("unknown manifest")

    def bundle_validator(self, path: Path) -> dict[str, object]:
        if path == self.aq4_bundle:
            return {
                "schema_version": FINAL.AQ4_BUNDLE_VALIDATOR_SCHEMA,
                "input_schema_version": FINAL.AQ4_BUNDLE_SCHEMA,
                "structurally_valid": True,
                "gate_eligible": True,
                "source_commit": AQ4_SOURCE_COMMIT,
                "artifact_count": 6,
                "reasons": [],
            }
        if path != self.bundle:
            raise ValueError("unknown release bundle")
        reasoning_release = FINAL._output_inventory(
            self.campaign_paths["reasoning_release"],
            run_id=self.authorization_document["campaigns"]["reasoning_release"][
                "run_id"
            ],
        )
        return {
            "schema_version": FINAL.BUNDLE_VALIDATOR_SCHEMA,
            "input_schema_version": FINAL.BUNDLE_SCHEMA,
            "structurally_valid": True,
            "gate_eligible": True,
            "source_commit": SOURCE_COMMIT,
            "artifact_count": 9,
            "model_campaign_schema_version": (
                "ullm.sq8.full_campaign.model_identity.v2"
            ),
            "reasoning_release_campaign": {
                "campaign_name": "reasoning_release",
                "run_id": reasoning_release["run_id"],
                "final_path": reasoning_release["path"],
                "kind": reasoning_release["kind"],
                "sha256": reasoning_release["sha256"],
                "artifact_inventory_sha256": "8" * 64,
                "artifact_count": reasoning_release["artifact_count"],
                "total_bytes": reasoning_release["total_bytes"],
                "selected_artifacts": reasoning_release["selected_artifacts"],
                "claim_path": self.claim_reference["path"],
                "claim_sha256": self.claim_reference["sha256"],
                "authorization_path": self.claim_reference["authorization_path"],
                "authorization_sha256": self.claim_reference[
                    "authorization_sha256"
                ],
            },
            "reasons": [],
        }

    def prepare(self) -> dict[str, object]:
        return FINAL.prepare_plan(
            plan_id="sq8-final-test-001",
            authorization_path=self.authorization_path,
            candidate_manifest=self.candidate,
            active_manifest=self.active,
            rollback_manifest=self.rollback,
            release_bundle=self.bundle,
            systemd_unit=self.unit,
            environment_file=self.environment,
            operations_document=self.operations,
            activation_outcome=self.activation_outcome,
            rollback_outcome=self.rollback_outcome,
            output=self.plan,
            now=NOW + timedelta(minutes=1),
            policy=self.policy,
            manifest_validator=self.manifest_validator,
            bundle_validator=self.bundle_validator,
        )

    def load(self, action: str) -> object:
        return FINAL.load_plan(
            self.plan,
            action=action,
            now=NOW + timedelta(minutes=2),
            policy=self.policy,
            manifest_validator=self.manifest_validator,
            bundle_validator=self.bundle_validator,
        )


class Runner:
    def __init__(self, *, fail_stage: str | None = None) -> None:
        self.fail_stage = fail_stage
        self.stages: list[str] = []
        self.environments: list[dict[str, str]] = []

    def __call__(
        self,
        argv: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        stage = str(kwargs["env"]["ULLM_FINAL_ACTIVATION_STAGE"])
        self.stages.append(stage)
        self.environments.append(dict(kwargs["env"]))
        return subprocess.CompletedProcess(
            argv,
            17 if stage == self.fail_stage else 0,
            "",
            "",
        )


class LiveProofLoader:
    def __init__(
        self,
        captured_at: datetime = NOW + timedelta(minutes=3),
    ) -> None:
        self.captured_at = captured_at

    def __call__(
        self,
        record: object,
        stage: str,
        activation_epoch: str,
    ) -> dict[str, object]:
        identity = (
            record.document["candidate"]
            if stage == "candidate_live_health"
            else record.document["rollback"]
        )
        specification = record.document["live_proofs"][stage]
        document = {
            "schema_version": FINAL.LIVE_PROOF_SCHEMA,
            "plan_sha256": record.snapshot.sha256,
            "stage": stage,
            "activation_epoch": activation_epoch,
            "captured_at": AUTH.utc_timestamp(self.captured_at),
            "active_manifest": {
                "path": record.document["active_manifest"]["path"],
                "manifest_sha256": identity["manifest_sha256"],
                "model_id": identity["model_id"],
                "format_id": identity["format_id"],
                "worker_protocol": identity["worker_protocol"],
                "worker_binary_sha256": identity["worker_binary_sha256"],
            },
            "service": {
                "unit": specification["service_unit"],
                "active_state": "active",
                "sub_state": "running",
                "boot_id": "11111111-2222-3333-4444-555555555555",
                "n_restarts": 0,
                "main_pid": 200,
                "control_group": "/system.slice/ullm-openai.service",
                "fragment_path": os.fspath(record.unit.path),
                "environment_file_path": os.fspath(record.environment.path),
            },
            "gateway": {
                "pid": 200,
                "ppid": 0,
                "starttime_ticks": 2000,
                "executable_sha256": specification["gateway_executable_sha256"],
            },
            "worker": {
                "pid": 201,
                "ppid": 200,
                "starttime_ticks": 2001,
                "executable_sha256": identity["worker_binary_sha256"],
            },
            "endpoints": {
                "gateway_healthz": {"status": 200},
                "gateway_readyz": {"status": 200},
                "gateway_models": {
                    "status": 200,
                    "model_ids": [identity["model_id"]],
                },
                "openwebui_health": {"status": 200},
                "openwebui_models": {
                    "status": 200,
                    "model_ids": [identity["model_id"]],
                },
            },
            "epoch_stable": True,
            "passed": True,
        }
        path = Path(specification["path"])
        path.write_bytes(canonical(document))
        path.chmod(0o444)
        return document


def noop_live_state_verifier(
    _record: object,
    _stage: str,
    _document: dict[str, object],
    _stage_started: datetime,
    _verified_at: datetime,
    _timeout: float,
) -> None:
    return None


def execute_activation(fixture: Fixture, runner: Runner) -> object:
    return FINAL.execute_activation(
        fixture.plan,
        expected_plan_sha256=digest(fixture.plan.read_bytes()),
        confirmation=FINAL.ACTIVATION_CONFIRMATION,
        policy=fixture.policy,
        manifest_validator=fixture.manifest_validator,
        bundle_validator=fixture.bundle_validator,
        runner=runner,
        live_proof_loader=LiveProofLoader(),
        live_state_verifier=noop_live_state_verifier,
        clock=lambda: NOW + timedelta(minutes=3),
    )


def execute_rollback(fixture: Fixture, runner: Runner) -> object:
    return FINAL.execute_rollback(
        fixture.plan,
        expected_plan_sha256=digest(fixture.plan.read_bytes()),
        confirmation=FINAL.ROLLBACK_CONFIRMATION,
        policy=fixture.policy,
        manifest_validator=fixture.manifest_validator,
        bundle_validator=fixture.bundle_validator,
        runner=runner,
        live_proof_loader=LiveProofLoader(NOW + timedelta(minutes=4)),
        live_state_verifier=noop_live_state_verifier,
        clock=lambda: NOW + timedelta(minutes=4),
    )


def damage_campaign_authority(
    fixture: Fixture,
    *,
    target: str,
    fault: str,
) -> Path:
    if target == "authorization":
        path = fixture.authorization_path
    elif target == "outcome":
        path = fixture.campaign_outcome.path
    else:
        raise AssertionError(f"unknown campaign authority target: {target}")
    if fault == "delete":
        path.unlink()
    elif fault == "corrupt":
        path.chmod(0o644)
        path.write_bytes(canonical({"corrupted": True}))
        path.chmod(0o444)
    else:
        raise AssertionError(f"unknown campaign authority fault: {fault}")
    return path


def test_prepare_preflight_activate_and_manual_rollback(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    document = fixture.prepare()

    assert document["route"] == FINAL.ROUTE
    assert document["schema_version"] == "ullm.served_model.final_activation_plan.v2"
    assert document["aq4_release_bundle"] == {
        "path": os.fspath(fixture.aq4_bundle),
        "sha256": digest(fixture.aq4_bundle.read_bytes()),
        "schema_version": FINAL.AQ4_BUNDLE_SCHEMA,
        "validator_schema_version": FINAL.AQ4_BUNDLE_VALIDATOR_SCHEMA,
        "validator_report_sha256": digest(
            canonical(fixture.bundle_validator(fixture.aq4_bundle))
        ),
    }
    assert stat.S_IMODE(fixture.plan.stat().st_mode) == 0o444
    preflight = fixture.load("activate")
    assert preflight.active.raw == fixture.aq4_raw
    assert (
        FINAL.preflight_report(preflight, action="activate")[
            "aq4_release_bundle_sha256"
        ]
        == digest(fixture.aq4_bundle.read_bytes())
    )

    activation_runner = Runner()
    activated = execute_activation(fixture, activation_runner)
    assert activated.status == "activated"
    assert fixture.active.read_bytes() == fixture.sq8_raw
    assert activation_runner.stages == [
        "candidate_reconciliation",
        "candidate_live_health",
    ]
    activation_document = json.loads(
        fixture.activation_outcome.read_text(encoding="ascii")
    )
    assert activation_document["status"] == "activated"
    assert stat.S_IMODE(fixture.activation_outcome.stat().st_mode) == 0o444

    rollback_preflight = fixture.load("rollback")
    assert rollback_preflight.active.raw == fixture.sq8_raw
    rollback_runner = Runner()
    rolled_back = execute_rollback(fixture, rollback_runner)
    assert rolled_back.status == "rolled_back"
    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert rollback_runner.stages == [
        "reverse_reconciliation",
        "rollback_live_health",
    ]
    rollback_document = json.loads(fixture.rollback_outcome.read_text(encoding="ascii"))
    assert rollback_document["status"] == "rolled_back"
    assert rollback_document["bytes_equal"] is True
    assert stat.S_IMODE(fixture.rollback_outcome.stat().st_mode) == 0o444


@pytest.mark.parametrize("target", ["authorization", "outcome"])
@pytest.mark.parametrize("fault", ["delete", "corrupt"])
def test_activation_admission_still_requires_campaign_authority(
    tmp_path: Path,
    target: str,
    fault: str,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    damaged = damage_campaign_authority(
        fixture,
        target=target,
        fault=fault,
    )
    runner = Runner()

    with pytest.raises(FINAL.FinalActivationError):
        execute_activation(fixture, runner)

    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert runner.stages == []
    assert not fixture.activation_outcome.exists()
    if fault == "delete":
        assert not damaged.exists()
    else:
        assert damaged.read_bytes() == canonical({"corrupted": True})


@pytest.mark.parametrize("target", ["authorization", "outcome"])
@pytest.mark.parametrize("fault", ["delete", "corrupt"])
def test_manual_rollback_uses_pinned_plan_not_campaign_registry(
    tmp_path: Path,
    target: str,
    fault: str,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    execute_activation(fixture, Runner())
    damage_campaign_authority(
        fixture,
        target=target,
        fault=fault,
    )

    rollback_preflight = fixture.load("rollback")
    assert rollback_preflight.campaign_outcome is None
    assert rollback_preflight.campaign_outcome_document is None
    assert (
        FINAL.preflight_report(rollback_preflight, action="rollback")[
            "campaign_outcome_sha256"
        ]
        == json.loads(fixture.plan.read_text(encoding="ascii"))["campaign"][
            "outcome_sha256"
        ]
    )
    result = execute_rollback(fixture, Runner())

    assert result.status == "rolled_back"
    assert fixture.active.read_bytes() == fixture.aq4_raw
    outcome = json.loads(fixture.rollback_outcome.read_text(encoding="ascii"))
    assert outcome["status"] == "rolled_back"
    assert outcome["bytes_equal"] is True


def test_campaign_independent_rollback_rejects_unknown_active_bytes(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    execute_activation(fixture, Runner())
    damage_campaign_authority(
        fixture,
        target="authorization",
        fault="delete",
    )
    damage_campaign_authority(
        fixture,
        target="outcome",
        fault="delete",
    )
    unknown = b'{"unexpected-active":true}\n'
    fixture.active.write_bytes(unknown)
    runner = Runner()

    with pytest.raises(FINAL.FinalActivationError, match="input hash"):
        execute_rollback(fixture, runner)

    assert fixture.active.read_bytes() == unknown
    assert runner.stages == []
    assert not fixture.rollback_outcome.exists()


@pytest.mark.parametrize("target", ["authorization", "outcome"])
@pytest.mark.parametrize("fault", ["delete", "corrupt"])
def test_activation_failure_restore_ignores_lost_campaign_registry_after_swap(
    tmp_path: Path,
    target: str,
    fault: str,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()

    class CampaignAuthorityFaultRunner(Runner):
        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            completed = super().__call__(argv, **kwargs)
            if self.stages[-1] == "candidate_reconciliation":
                assert fixture.active.read_bytes() == fixture.sq8_raw
                damage_campaign_authority(
                    fixture,
                    target=target,
                    fault=fault,
                )
            return completed

    runner = CampaignAuthorityFaultRunner()
    with pytest.raises(FINAL.FinalActivationError):
        execute_activation(fixture, runner)

    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert runner.stages == [
        "candidate_reconciliation",
        "reverse_reconciliation",
        "rollback_live_health",
    ]
    outcome = json.loads(fixture.activation_outcome.read_text(encoding="ascii"))
    assert outcome["status"] == "failed_restored"
    assert outcome["failure_stage"] == "candidate_reconciliation"
    assert outcome["restoration"]["bytes_equal"] is True
    assert outcome["restoration"]["reverse_reconciliation_passed"] is True
    assert outcome["restoration"]["live_health_passed"] is True


def test_activation_health_failure_restores_aq4_and_records_outcome(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()

    with pytest.raises(FINAL.FinalActivationError):
        execute_activation(fixture, Runner(fail_stage="candidate_live_health"))

    assert fixture.active.read_bytes() == fixture.aq4_raw
    outcome = json.loads(fixture.activation_outcome.read_text(encoding="ascii"))
    assert outcome["status"] == "failed_restored"
    assert outcome["failure_stage"] == "candidate_live_health"
    assert outcome["restoration"] == {
        "attempted": True,
        "manifest_sha256": digest(fixture.aq4_raw),
        "bytes_equal": True,
        "reverse_reconciliation_passed": True,
        "live_health_passed": True,
    }


def test_reverse_reconciliation_failure_is_not_reported_as_safe(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()

    with pytest.raises(FINAL.FinalActivationError):
        execute_activation(
            fixture,
            Runner(
                fail_stage="candidate_live_health",
            ),
        )

    # The first fixture has consumed its immutable outcome; exercise a fresh
    # transaction whose rollback hook itself fails.
    second = Fixture(tmp_path / "second")
    second.prepare()
    runner = Runner(fail_stage="reverse_reconciliation")

    # Trigger candidate failure first, then fail the reverse hook.
    def fail_two_stages(
        argv: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        stage = str(kwargs["env"]["ULLM_FINAL_ACTIVATION_STAGE"])
        runner.stages.append(stage)
        code = 19 if stage in {"candidate_live_health", "reverse_reconciliation"} else 0
        return subprocess.CompletedProcess(argv, code, "", "")

    with pytest.raises(FINAL.FinalActivationError):
        FINAL.execute_activation(
            second.plan,
            expected_plan_sha256=digest(second.plan.read_bytes()),
            confirmation=FINAL.ACTIVATION_CONFIRMATION,
            policy=second.policy,
            manifest_validator=second.manifest_validator,
            bundle_validator=second.bundle_validator,
            runner=fail_two_stages,
            live_proof_loader=LiveProofLoader(),
            live_state_verifier=noop_live_state_verifier,
            clock=lambda: NOW + timedelta(minutes=3),
        )
    assert second.active.read_bytes() == second.aq4_raw
    outcome = json.loads(second.activation_outcome.read_text(encoding="ascii"))
    assert outcome["status"] == "failed_restore"
    assert outcome["restoration"]["bytes_equal"] is True
    assert outcome["restoration"]["reverse_reconciliation_passed"] is False


def test_changed_campaign_output_invalidates_plan(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    (fixture.campaign_paths["reasoning_release"] / "summary.json").write_text(
        '{"changed":true}\n',
        encoding="ascii",
    )

    with pytest.raises(
        FINAL.FinalActivationError,
        match="changed",
    ):
        fixture.load("activate")
    assert fixture.active.read_bytes() == fixture.aq4_raw


@pytest.mark.parametrize("mutation", ["missing", "stale"])
def test_aq4_bundle_campaign_output_must_remain_exact(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = Fixture(tmp_path)
    if mutation == "missing":
        fixture.aq4_bundle.unlink()
    else:
        changed = dict(fixture.aq4_bundle_document)
        changed["identity"] = dict(changed["identity"])
        changed["identity"]["tokenizer_sha256"] = "6" * 64
        fixture.aq4_bundle.chmod(0o644)
        fixture.aq4_bundle.write_bytes(canonical(changed))
        fixture.aq4_bundle.chmod(0o444)

    with pytest.raises(FINAL.FinalActivationError):
        fixture.prepare()


def test_aq4_bundle_must_be_v1_gate_eligible(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)

    def gate_false(path: Path) -> dict[str, object]:
        report = fixture.bundle_validator(path)
        if path == fixture.aq4_bundle:
            report["gate_eligible"] = False
            report["reasons"] = ["fixture rejection"]
        return report

    with pytest.raises(
        FINAL.FinalActivationError,
        match="AQ4 release bundle is not production-gate eligible",
    ):
        FINAL.prepare_plan(
            plan_id="sq8-final-test-001",
            authorization_path=fixture.authorization_path,
            candidate_manifest=fixture.candidate,
            active_manifest=fixture.active,
            rollback_manifest=fixture.rollback,
            release_bundle=fixture.bundle,
            systemd_unit=fixture.unit,
            environment_file=fixture.environment,
            operations_document=fixture.operations,
            activation_outcome=fixture.activation_outcome,
            rollback_outcome=fixture.rollback_outcome,
            output=fixture.plan,
            now=NOW + timedelta(minutes=1),
            policy=fixture.policy,
            manifest_validator=fixture.manifest_validator,
            bundle_validator=gate_false,
        )


def test_fresh_aq4_raw_output_mutation_is_rejected(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    cases_path = fixture.campaign_paths["aq4_reasoning_release"] / "cases.json"
    cases_path.chmod(0o644)
    cases_path.write_bytes(
        canonical([{"case_id": "mutated-after-outcome", "passed": True}])
    )

    with pytest.raises(FINAL.FinalActivationError, match="campaign output changed"):
        fixture.prepare()


def test_aq4_bundle_browser_component_must_be_the_fresh_output(
    tmp_path: Path,
) -> None:
    def mismatch(fixture: Fixture) -> None:
        alternate = fixture.aq4_bundle_root / "alternate-browser-evidence.json"
        alternate.write_bytes(
            fixture.campaign_paths["aq4_reasoning_browser"].read_bytes()
        )
        fixture.aq4_bundle_document["artifacts"]["browser_evidence"] = {
            "path": alternate.relative_to(fixture.aq4_bundle_root).as_posix(),
            "sha256": digest(alternate.read_bytes()),
        }
        fixture.aq4_bundle.write_bytes(canonical(fixture.aq4_bundle_document))

    fixture = Fixture(tmp_path, aq4_mutator=mismatch)

    with pytest.raises(
        FINAL.FinalActivationError,
        match="outside its fresh campaign output",
    ):
        fixture.prepare()


@pytest.mark.parametrize(
    ("artifact_name", "attribute"),
    (
        ("promotion_evidence", "aq4_promotion_evidence"),
        ("promotion_receipt", "aq4_bundle_receipt"),
    ),
)
def test_aq4_bundle_promotion_pair_must_be_exact_authorized_copies(
    tmp_path: Path,
    artifact_name: str,
    attribute: str,
) -> None:
    def mismatch(fixture: Fixture) -> None:
        target = getattr(fixture, attribute)
        target.write_bytes(canonical({"different-valid-copy": True}))
        fixture.aq4_bundle_document["artifacts"][artifact_name]["sha256"] = (
            digest(target.read_bytes())
        )
        fixture.aq4_bundle.write_bytes(canonical(fixture.aq4_bundle_document))

    fixture = Fixture(tmp_path, aq4_mutator=mismatch)

    with pytest.raises(
        FINAL.FinalActivationError,
        match=f"AQ4 bundle {artifact_name} differs from its authorization",
    ):
        fixture.prepare()


def test_aq4_release_evidence_must_embed_fresh_cases_and_lifecycle(
    tmp_path: Path,
) -> None:
    def mismatch(fixture: Fixture) -> None:
        release = json.loads(
            fixture.aq4_release_evidence.read_text(encoding="ascii")
        )
        release["cases"] = [{"case_id": "different-valid-case", "passed": True}]
        fixture.aq4_release_evidence.write_bytes(canonical(release))
        fixture.aq4_bundle_document["artifacts"]["release_evidence"][
            "sha256"
        ] = digest(fixture.aq4_release_evidence.read_bytes())
        fixture.aq4_bundle.write_bytes(canonical(fixture.aq4_bundle_document))

    fixture = Fixture(tmp_path, aq4_mutator=mismatch)

    with pytest.raises(
        FINAL.FinalActivationError,
        match="release evidence differs from fresh",
    ):
        fixture.prepare()


@pytest.mark.parametrize("mutation", ["source", "manifest", "worker"])
def test_aq4_bundle_source_and_identity_must_equal_authorized_before(
    tmp_path: Path,
    mutation: str,
) -> None:
    def mismatch(fixture: Fixture) -> None:
        if mutation == "source":
            fixture.aq4_bundle_document["source_commit"] = "e" * 40
            fixture.aq4_bundle_document["active_promotion_source_commit"] = (
                "e" * 40
            )
        else:
            identity = fixture.aq4_bundle_document["identity"]
            identity[f"{mutation}_sha256" if mutation == "manifest" else "worker_binary_sha256"] = (
                "e" * 64
            )
        fixture.aq4_bundle.write_bytes(canonical(fixture.aq4_bundle_document))

    fixture = Fixture(tmp_path, aq4_mutator=mismatch)

    with pytest.raises(FINAL.FinalActivationError, match="bundle identity differs"):
        fixture.prepare()


def test_plan_cannot_select_a_different_valid_aq4_bundle(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    alternate = fixture.aq4_bundle_root / "alternate-valid-bundle-v1.json"
    alternate.write_bytes(fixture.aq4_bundle.read_bytes())
    alternate.chmod(0o444)
    plan = json.loads(fixture.plan.read_text(encoding="ascii"))
    plan["aq4_release_bundle"]["path"] = os.fspath(alternate)
    fixture.plan.chmod(0o644)
    fixture.plan.write_bytes(canonical(plan))
    fixture.plan.chmod(0o444)

    with pytest.raises(
        FINAL.FinalActivationError,
        match="plan path differs from fresh campaign output",
    ):
        fixture.load("activate")


def test_plan_tampered_aq4_validator_report_hash_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    plan = json.loads(fixture.plan.read_text(encoding="ascii"))
    plan["aq4_release_bundle"]["validator_report_sha256"] = "0" * 64
    fixture.plan.chmod(0o644)
    fixture.plan.write_bytes(canonical(plan))
    fixture.plan.chmod(0o444)

    with pytest.raises(
        FINAL.FinalActivationError,
        match="AQ4 release bundle validator report changed",
    ):
        fixture.load("activate")


def test_changed_reviewed_executable_invalidates_plan(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    fixture.executable.write_bytes(b"#!/bin/sh\nexit 9\n")
    fixture.executable.chmod(0o755)

    with pytest.raises(FINAL.FinalActivationError, match="executable"):
        fixture.load("activate")
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_active_bytes_must_equal_exact_rollback_bytes(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    fixture.active.write_bytes(b'{"unexpected":true}\n')

    with pytest.raises(FINAL.FinalActivationError, match="exact activate precondition"):
        fixture.load("activate")


def test_complete_bundle_v2_is_required(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)

    with pytest.raises(FINAL.FinalActivationError, match="production-gate"):
        FINAL.prepare_plan(
            plan_id="sq8-final-test-001",
            authorization_path=fixture.authorization_path,
            candidate_manifest=fixture.candidate,
            active_manifest=fixture.active,
            rollback_manifest=fixture.rollback,
            release_bundle=fixture.bundle,
            systemd_unit=fixture.unit,
            environment_file=fixture.environment,
            operations_document=fixture.operations,
            activation_outcome=fixture.activation_outcome,
            rollback_outcome=fixture.rollback_outcome,
            output=fixture.plan,
            now=NOW + timedelta(minutes=1),
            policy=fixture.policy,
            manifest_validator=fixture.manifest_validator,
            bundle_validator=lambda _path: {
                "schema_version": FINAL.BUNDLE_VALIDATOR_SCHEMA,
                "input_schema_version": FINAL.BUNDLE_SCHEMA,
                "structurally_valid": True,
                "gate_eligible": False,
            },
        )
    assert not fixture.plan.exists()


def test_execute_confirmation_is_exact_and_cli_has_no_command_json(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    runner_cli = load_cli("test_final_activation_runner_cli", RUNNER_PATH)
    rollback_cli = load_cli("test_final_rollback_cli", ROLLBACK_PATH)
    observed = digest(fixture.plan.read_bytes())

    preflight_args = runner_cli.parse_args(["--plan", os.fspath(fixture.plan)])
    runner_cli._require_mode(preflight_args, observed)
    wrong = runner_cli.parse_args(
        [
            "--plan",
            os.fspath(fixture.plan),
            "--execute",
            "--confirm-plan-sha256",
            "0" * 64,
            "--confirmation",
            FINAL.ACTIVATION_CONFIRMATION,
        ]
    )
    with pytest.raises(Exception, match="exact plan"):
        runner_cli._require_mode(wrong, observed)
    confirmed = runner_cli.parse_args(
        [
            "--plan",
            os.fspath(fixture.plan),
            "--execute",
            "--confirm-plan-sha256",
            observed,
            "--confirmation",
            FINAL.ACTIVATION_CONFIRMATION,
        ]
    )
    runner_cli._require_mode(confirmed, observed)
    with pytest.raises(SystemExit):
        runner_cli.parse_args(
            [
                "--plan",
                os.fspath(fixture.plan),
                "--check-command-json",
                '["/bin/true"]',
            ]
        )

    rollback_confirmed = rollback_cli.parse_args(
        [
            "--plan",
            os.fspath(fixture.plan),
            "--execute",
            "--confirm-plan-sha256",
            observed,
            "--confirmation",
            FINAL.ROLLBACK_CONFIRMATION,
        ]
    )
    rollback_cli._require_mode(rollback_confirmed, observed)


def test_lock_is_shared_with_existing_activation_route(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    lock_path = fixture.active.parent / f".{fixture.active.name}.activation.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(FINAL.FinalActivationError, match="another activation"):
            execute_activation(fixture, Runner())
    finally:
        os.close(descriptor)
    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert not fixture.activation_outcome.exists()


def test_candidate_mutation_and_active_symlink_fail_before_commands(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    fixture.candidate.write_bytes(b'{"changed":true}\n')

    with pytest.raises(FINAL.FinalActivationError, match="input hash"):
        fixture.load("activate")

    second = Fixture(tmp_path / "symlink")
    second.prepare()
    original = second.slot / "active-original.json"
    second.active.rename(original)
    second.active.symlink_to(original)
    with pytest.raises(FINAL.FinalActivationError, match="unavailable or changed"):
        second.load("activate")


def test_shell_or_interpreter_wrapper_is_not_a_reviewed_direct_operation(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    operations = json.loads(fixture.operations.read_text(encoding="ascii"))
    operations["stages"]["candidate_reconciliation"][0] = {
        "argv": ["/bin/sh", "-c", "exit 0"],
        "executable_sha256": digest(Path("/bin/sh").read_bytes()),
    }
    fixture.operations.chmod(0o644)
    fixture.operations.write_bytes(canonical(operations))
    fixture.operations.chmod(0o444)

    with pytest.raises(
        FINAL.FinalActivationError,
        match="shell or interpreter",
    ):
        fixture.prepare()
    assert not fixture.plan.exists()


def test_manual_rollback_hook_failure_records_bytes_without_claiming_health(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    execute_activation(fixture, Runner())

    with pytest.raises(FINAL.FinalActivationError, match="live health failed"):
        execute_rollback(fixture, Runner(fail_stage="reverse_reconciliation"))

    assert fixture.active.read_bytes() == fixture.aq4_raw
    outcome = json.loads(fixture.rollback_outcome.read_text(encoding="ascii"))
    assert outcome["status"] == "rollback_incomplete"
    assert outcome["bytes_equal"] is True
    assert outcome["stages"]["reverse_reconciliation"] == "failed"
    assert outcome["stages"]["rollback_live_health"] == "skipped"


def test_signal_at_immutable_activation_commit_does_not_create_split_brain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    original_publish = FINAL._publish_immutable
    signalled = False

    def publish_then_signal(
        path: Path,
        document: dict[str, object],
        *,
        required_uid: int,
    ) -> object:
        nonlocal signalled
        snapshot = original_publish(
            path,
            document,
            required_uid=required_uid,
        )
        if path == fixture.activation_outcome and not signalled:
            signalled = True
            os.kill(os.getpid(), signal.SIGTERM)
        return snapshot

    import signal

    monkeypatch.setattr(FINAL, "_publish_immutable", publish_then_signal)
    result = execute_activation(fixture, Runner())

    assert signalled is True
    assert result.status == "activated"
    assert fixture.active.read_bytes() == fixture.sq8_raw
    outcome = json.loads(fixture.activation_outcome.read_text(encoding="ascii"))
    assert outcome["status"] == "activated"


def test_activation_cannot_be_replayed_after_immutable_success(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    execute_activation(fixture, Runner())

    with pytest.raises(FINAL.FinalActivationError):
        execute_activation(fixture, Runner())
    assert fixture.active.read_bytes() == fixture.sq8_raw


def test_reviewed_true_cannot_replace_structured_live_health(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    operations = json.loads(fixture.operations.read_text(encoding="ascii"))
    true_path = fixture.executable
    for commands in operations["stages"].values():
        commands[:] = [
            {
                "argv": [os.fspath(true_path)],
                "executable_sha256": digest(true_path.read_bytes()),
            }
        ]
    fixture.operations.chmod(0o644)
    fixture.operations.write_bytes(canonical(operations))
    fixture.operations.chmod(0o444)
    fixture.prepare()

    with pytest.raises(FINAL.FinalActivationError):
        FINAL.execute_activation(
            fixture.plan,
            expected_plan_sha256=digest(fixture.plan.read_bytes()),
            confirmation=FINAL.ACTIVATION_CONFIRMATION,
            policy=fixture.policy,
            manifest_validator=fixture.manifest_validator,
            bundle_validator=fixture.bundle_validator,
            runner=Runner(),
            live_state_verifier=noop_live_state_verifier,
            clock=lambda: NOW + timedelta(minutes=3),
        )

    assert fixture.active.read_bytes() == fixture.aq4_raw
    outcome = json.loads(fixture.activation_outcome.read_text(encoding="ascii"))
    assert outcome["status"] == "failed_restore"
    assert outcome["stages"]["candidate_live_health"] == "failed"
    assert outcome["live_proofs"] == {
        "candidate_live_health": None,
        "rollback_live_health": None,
    }


def test_plan_input_mutation_between_command_boundaries_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()

    class MutatingRunner(Runner):
        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            completed = super().__call__(argv, **kwargs)
            if self.stages[-1] == "candidate_reconciliation":
                fixture.unit.write_bytes(b"[Service]\nExecStart=/unexpected\n")
            return completed

    with pytest.raises(FINAL.FinalActivationError):
        execute_activation(fixture, MutatingRunner())

    assert fixture.active.read_bytes() == fixture.aq4_raw
    outcome = json.loads(fixture.activation_outcome.read_text(encoding="ascii"))
    assert outcome["status"] == "failed_restore"
    assert outcome["failure_stage"] == "candidate_reconciliation"


def test_aq4_bundle_mutation_during_activation_is_detected_and_restores_bytes(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()

    class MutatingRunner(Runner):
        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            completed = super().__call__(argv, **kwargs)
            if self.stages[-1] == "candidate_reconciliation":
                fixture.aq4_bundle.chmod(0o644)
                changed = dict(fixture.aq4_bundle_document)
                changed["identity"] = dict(changed["identity"])
                changed["identity"]["tokenizer_sha256"] = "6" * 64
                fixture.aq4_bundle.write_bytes(canonical(changed))
                fixture.aq4_bundle.chmod(0o444)
            return completed

    with pytest.raises(FINAL.FinalActivationError):
        execute_activation(fixture, MutatingRunner())

    assert fixture.active.read_bytes() == fixture.aq4_raw
    outcome = json.loads(fixture.activation_outcome.read_text(encoding="ascii"))
    assert outcome["status"] == "failed_restored"
    assert outcome["restoration"]["bytes_equal"] is True


def test_aq4_bundle_mutation_does_not_block_exact_runtime_rollback(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    execute_activation(fixture, Runner())
    fixture.aq4_bundle.chmod(0o644)
    changed = dict(fixture.aq4_bundle_document)
    changed["identity"] = dict(changed["identity"])
    changed["identity"]["tokenizer_sha256"] = "6" * 64
    fixture.aq4_bundle.write_bytes(canonical(changed))
    fixture.aq4_bundle.chmod(0o444)

    result = execute_rollback(fixture, Runner())

    assert result.status == "rolled_back"
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_active_mutation_during_command_is_detected_and_exact_aq4_is_restored(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()

    class ActiveMutatingRunner(Runner):
        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            completed = super().__call__(argv, **kwargs)
            if self.stages[-1] == "candidate_reconciliation":
                fixture.active.write_bytes(b'{"unexpected":true}\n')
            return completed

    with pytest.raises(FINAL.FinalActivationError):
        execute_activation(fixture, ActiveMutatingRunner())

    assert fixture.active.read_bytes() == fixture.aq4_raw
    outcome = json.loads(fixture.activation_outcome.read_text(encoding="ascii"))
    assert outcome["status"] == "failed_restored"
    assert outcome["restoration"]["bytes_equal"] is True
    assert outcome["live_proofs"]["rollback_live_health"] is not None


def test_live_proof_model_mutation_is_rejected(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    valid_loader = LiveProofLoader()

    def mutated_loader(
        record: object,
        stage: str,
        activation_epoch: str,
    ) -> dict[str, object]:
        document = valid_loader(record, stage, activation_epoch)
        if stage == "candidate_live_health":
            document["endpoints"]["gateway_models"]["model_ids"] = [FINAL.AQ4_MODEL_ID]
            path = Path(record.document["live_proofs"][stage]["path"])
            path.chmod(0o644)
            path.write_bytes(canonical(document))
            path.chmod(0o444)
        return document

    with pytest.raises(FINAL.FinalActivationError):
        FINAL.execute_activation(
            fixture.plan,
            expected_plan_sha256=digest(fixture.plan.read_bytes()),
            confirmation=FINAL.ACTIVATION_CONFIRMATION,
            policy=fixture.policy,
            manifest_validator=fixture.manifest_validator,
            bundle_validator=fixture.bundle_validator,
            runner=Runner(),
            live_proof_loader=mutated_loader,
            live_state_verifier=noop_live_state_verifier,
            clock=lambda: NOW + timedelta(minutes=3),
        )

    assert fixture.active.read_bytes() == fixture.aq4_raw
    outcome = json.loads(fixture.activation_outcome.read_text(encoding="ascii"))
    assert outcome["status"] == "failed_restored"
    assert outcome["live_proofs"]["candidate_live_health"] is None
    assert outcome["live_proofs"]["rollback_live_health"] is not None


def test_reviewed_operations_receive_only_minimal_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    monkeypatch.setenv("ULLM_AMBIENT_SECRET", "must-not-propagate")
    runner = Runner()

    execute_activation(fixture, runner)

    assert runner.environments
    for environment in runner.environments:
        assert "ULLM_AMBIENT_SECRET" not in environment
        assert "PATH" not in environment
        assert set(environment) <= {
            "LANG",
            "LC_ALL",
            "ULLM_FINAL_ACTIVATION_PLAN",
            "ULLM_FINAL_ACTIVATION_PLAN_SHA256",
            "ULLM_FINAL_ACTIVATION_STAGE",
            "ULLM_FINAL_ACTIVATION_EPOCH",
            "ULLM_FINAL_ACTIVATION_LIVE_PROOF",
            "ULLM_ACTIVE_MANIFEST",
            "ULLM_CANDIDATE_MANIFEST_SHA256",
            "ULLM_ROLLBACK_MANIFEST_SHA256",
        }


def test_renamed_interpreter_script_is_not_a_direct_reviewed_executable(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    raw = b"#!/bin/sh\nexit 0\n"
    fixture.executable.write_bytes(raw)
    fixture.executable.chmod(0o755)
    operations = json.loads(fixture.operations.read_text(encoding="ascii"))
    for commands in operations["stages"].values():
        commands[0]["executable_sha256"] = digest(raw)
    fixture.operations.chmod(0o644)
    fixture.operations.write_bytes(canonical(operations))
    fixture.operations.chmod(0o444)

    with pytest.raises(FINAL.FinalActivationError, match="interpreter script"):
        fixture.prepare()


def test_timeout_reaps_the_owned_reviewed_operation_process_group(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    program = (
        "import pathlib,subprocess,time;"
        "child=subprocess.Popen(['/usr/bin/sleep','30'],start_new_session=True);"
        f"pathlib.Path({os.fspath(child_pid_path)!r}).write_text(str(child.pid));"
        "time.sleep(30)"
    )
    python_path = Path("/usr/bin/python3").resolve(strict=True)
    executable_fd = FINAL._open_verified_executable(
        python_path,
        expected_sha256=digest(python_path.read_bytes()),
        required_uid=os.geteuid(),
        label="test Python",
    )

    try:
        with pytest.raises(FINAL.FinalActivationError, match="command failed"):
            FINAL._run_owned_process_group(
                [os.fspath(python_path), "-c", program],
                executable_fd=executable_fd,
                environment={"LANG": "C", "LC_ALL": "C"},
                timeout=0.2,
                stage="candidate_reconciliation",
            )
    finally:
        os.close(executable_fd)

    child_pid = int(child_pid_path.read_text(encoding="ascii"))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail("reviewed operation descendant survived process-group cleanup")


def test_signal_during_failure_restore_is_deferred_until_safe_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    original_replace = FINAL._atomic_replace_exact
    signalled = False

    def replace_then_signal(**kwargs: object) -> None:
        nonlocal signalled
        original_replace(**kwargs)
        if kwargs["replacement_raw"] == fixture.aq4_raw and not signalled:
            signalled = True
            os.kill(os.getpid(), signal.SIGTERM)

    import signal

    monkeypatch.setattr(FINAL, "_atomic_replace_exact", replace_then_signal)
    with pytest.raises(FINAL.FinalActivationError):
        execute_activation(fixture, Runner(fail_stage="candidate_live_health"))

    assert signalled is True
    assert fixture.active.read_bytes() == fixture.aq4_raw
    outcome = json.loads(fixture.activation_outcome.read_text(encoding="ascii"))
    assert outcome["status"] == "failed_restored"
    assert outcome["restoration"]["live_health_passed"] is True


def test_core_execution_authority_is_required_before_any_write(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    observed = digest(fixture.plan.read_bytes())

    with pytest.raises(FINAL.FinalActivationError, match="confirmation"):
        FINAL.execute_activation(
            fixture.plan,
            expected_plan_sha256=observed,
            confirmation="NOT_APPROVED",
            policy=fixture.policy,
            manifest_validator=fixture.manifest_validator,
            bundle_validator=fixture.bundle_validator,
            runner=Runner(),
            live_proof_loader=LiveProofLoader(),
            live_state_verifier=noop_live_state_verifier,
            clock=lambda: NOW + timedelta(minutes=3),
        )

    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert not fixture.activation_outcome.exists()


def test_plan_inode_swap_between_confirmation_and_lock_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    observed = digest(fixture.plan.read_bytes())
    original_open = FINAL._open_activation_lock
    swapped = False

    def swap_then_open(active: Path, *, required_uid: int) -> tuple[int, int]:
        nonlocal swapped
        if not swapped:
            raw = fixture.plan.read_bytes()
            fixture.plan.rename(fixture.plan.with_suffix(".reviewed.json"))
            fixture.plan.write_bytes(raw)
            fixture.plan.chmod(0o444)
            swapped = True
        return original_open(active, required_uid=required_uid)

    monkeypatch.setattr(FINAL, "_open_activation_lock", swap_then_open)
    with pytest.raises(FINAL.FinalActivationError, match="confirmed final activation plan"):
        FINAL.execute_activation(
            fixture.plan,
            expected_plan_sha256=observed,
            confirmation=FINAL.ACTIVATION_CONFIRMATION,
            policy=fixture.policy,
            manifest_validator=fixture.manifest_validator,
            bundle_validator=fixture.bundle_validator,
            runner=Runner(),
            live_proof_loader=LiveProofLoader(),
            live_state_verifier=noop_live_state_verifier,
            clock=lambda: NOW + timedelta(minutes=3),
        )

    assert swapped is True
    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert not fixture.activation_outcome.exists()


def test_verified_executable_fd_is_not_redirected_by_path_replacement(
    tmp_path: Path,
) -> None:
    reviewed = tmp_path / "reviewed-operation"
    reviewed.write_bytes(Path("/usr/bin/true").read_bytes())
    reviewed.chmod(0o755)
    executable_fd = FINAL._open_verified_executable(
        reviewed,
        expected_sha256=digest(reviewed.read_bytes()),
        required_uid=os.geteuid(),
        label="reviewed operation",
    )
    replacement = tmp_path / "replacement"
    replacement.write_bytes(Path("/usr/bin/false").read_bytes())
    replacement.chmod(0o755)
    os.replace(replacement, reviewed)
    try:
        FINAL._run_owned_process_group(
            [os.fspath(reviewed)],
            executable_fd=executable_fd,
            environment={"LANG": "C", "LC_ALL": "C"},
            timeout=2.0,
            stage="candidate_reconciliation",
        )
    finally:
        os.close(executable_fd)


def test_atomic_exchange_detects_racing_active_writer_and_restores_aq4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    original_exchange = FINAL._rename_exchange
    raced = False

    def race_once(parent_descriptor: int, left: str, right: str) -> None:
        nonlocal raced
        if not raced and right == fixture.active.name:
            racer = fixture.slot / "racing-active.json"
            racer.write_bytes(b'{"racing-writer":true}\n')
            racer.chmod(0o644)
            os.replace(racer, fixture.active)
            raced = True
        original_exchange(parent_descriptor, left, right)

    monkeypatch.setattr(FINAL, "_rename_exchange", race_once)
    with pytest.raises(FINAL.FinalActivationError):
        execute_activation(fixture, Runner())

    assert raced is True
    assert fixture.active.read_bytes() == fixture.aq4_raw
    outcome = json.loads(fixture.activation_outcome.read_text(encoding="ascii"))
    assert outcome["status"] == "failed_restored"


def test_reasoning_release_lineage_must_equal_the_authorized_outcome(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    release = json.loads(fixture.release_evidence.read_text(encoding="ascii"))
    release["campaign_lineage"]["campaign"]["run_id"] = "other-run"
    fixture.release_evidence.write_bytes(canonical(release))
    fixture.bundle_document["artifacts"]["release_evidence"]["sha256"] = digest(
        fixture.release_evidence.read_bytes()
    )
    fixture.bundle.write_bytes(canonical(fixture.bundle_document))

    with pytest.raises(FINAL.FinalActivationError, match="reasoning_release lineage"):
        fixture.prepare()


def test_bundle_validator_reasoning_release_report_must_equal_outcome(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)

    def wrong_report(path: Path) -> dict[str, object]:
        report = fixture.bundle_validator(path)
        if path == fixture.aq4_bundle:
            return report
        reasoning_campaign = dict(report["reasoning_release_campaign"])
        reasoning_campaign["run_id"] = "other-run"
        report["reasoning_release_campaign"] = reasoning_campaign
        return report

    with pytest.raises(
        FINAL.FinalActivationError,
        match="validator reasoning release campaign differs",
    ):
        FINAL.prepare_plan(
            plan_id="sq8-final-test-001",
            authorization_path=fixture.authorization_path,
            candidate_manifest=fixture.candidate,
            active_manifest=fixture.active,
            rollback_manifest=fixture.rollback,
            release_bundle=fixture.bundle,
            systemd_unit=fixture.unit,
            environment_file=fixture.environment,
            operations_document=fixture.operations,
            activation_outcome=fixture.activation_outcome,
            rollback_outcome=fixture.rollback_outcome,
            output=fixture.plan,
            now=NOW + timedelta(minutes=1),
            policy=fixture.policy,
            manifest_validator=fixture.manifest_validator,
            bundle_validator=wrong_report,
        )


def test_live_proof_is_fresh_and_independently_reobserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    record = fixture.load("activate")
    epoch = "8" * 64
    proof = LiveProofLoader()(record, "candidate_live_health", epoch)

    def process_identity(pid: int) -> dict[str, object]:
        if pid == proof["gateway"]["pid"]:
            return dict(proof["gateway"])
        return dict(proof["worker"])

    monkeypatch.setattr(FINAL, "_process_live_identity", process_identity)
    monkeypatch.setattr(FINAL, "_read_boot_id", lambda: proof["service"]["boot_id"])
    monkeypatch.setattr(
        FINAL,
        "_systemd_live_state",
        lambda *_args, **_kwargs: {
            key: value
            for key, value in proof["service"].items()
            if key != "boot_id"
        },
    )

    def endpoints(
        name: str,
        _url: str,
        *,
        timeout: float,
        required_uid: int,
    ) -> tuple[int, bytes]:
        del timeout, required_uid
        if name in {"gateway_models", "openwebui_models"}:
            return 200, canonical({"data": [{"id": FINAL.SQ8_MODEL_ID}]})
        return 200, b"{}\n"

    monkeypatch.setattr(FINAL, "_endpoint_live_state", endpoints)
    FINAL.default_live_state_verifier(
        record,
        "candidate_live_health",
        proof,
        NOW + timedelta(minutes=3),
        NOW + timedelta(minutes=3),
        10.0,
    )

    def wrong_models(
        name: str,
        _url: str,
        *,
        timeout: float,
        required_uid: int,
    ) -> tuple[int, bytes]:
        del timeout, required_uid
        if name in {"gateway_models", "openwebui_models"}:
            return 200, canonical({"data": [{"id": FINAL.AQ4_MODEL_ID}]})
        return 200, b"{}\n"

    monkeypatch.setattr(FINAL, "_endpoint_live_state", wrong_models)
    with pytest.raises(FINAL.FinalActivationError, match="different model"):
        FINAL.default_live_state_verifier(
            record,
            "candidate_live_health",
            proof,
            NOW + timedelta(minutes=3),
            NOW + timedelta(minutes=3),
            10.0,
        )


def test_stale_live_proof_fails_and_exact_aq4_is_restored(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()

    with pytest.raises(FINAL.FinalActivationError):
        FINAL.execute_activation(
            fixture.plan,
            expected_plan_sha256=digest(fixture.plan.read_bytes()),
            confirmation=FINAL.ACTIVATION_CONFIRMATION,
            policy=fixture.policy,
            manifest_validator=fixture.manifest_validator,
            bundle_validator=fixture.bundle_validator,
            runner=Runner(),
            live_proof_loader=LiveProofLoader(NOW),
            live_state_verifier=noop_live_state_verifier,
            clock=lambda: NOW + timedelta(minutes=3),
        )

    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_embedded_candidate_proof_keeps_manual_rollback_available(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    execute_activation(fixture, Runner())
    candidate_proof = Path(
        fixture.operations_document["live_proofs"]["candidate_live_health"]["path"]
    )
    candidate_proof.unlink()

    fixture.load("rollback")
    result = execute_rollback(fixture, Runner())
    assert result.status == "rolled_back"
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_active_window_deadline_forces_exact_aq4_restoration(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    operations = json.loads(fixture.operations.read_text(encoding="ascii"))
    operations["timeout_seconds"] = 0.01
    operations["active_window_timeout_seconds"] = 0.02
    fixture.operations.chmod(0o644)
    fixture.operations.write_bytes(canonical(operations))
    fixture.operations.chmod(0o444)
    fixture.prepare()

    class SlowOnceRunner(Runner):
        def __init__(self) -> None:
            super().__init__()
            self.slept = False

        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            completed = super().__call__(argv, **kwargs)
            if not self.slept:
                self.slept = True
                time.sleep(0.05)
            return completed

    with pytest.raises(FINAL.FinalActivationError):
        execute_activation(fixture, SlowOnceRunner())
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_wrong_service_and_renamed_interpreter_binary_are_rejected(
    tmp_path: Path,
) -> None:
    wrong_service = Fixture(tmp_path / "wrong-service")
    operations = json.loads(wrong_service.operations.read_text(encoding="ascii"))
    operations["live_proofs"]["candidate_live_health"]["service_unit"] = "other.service"
    wrong_service.operations.chmod(0o644)
    wrong_service.operations.write_bytes(canonical(operations))
    wrong_service.operations.chmod(0o444)
    with pytest.raises(FINAL.FinalActivationError, match="different service unit"):
        wrong_service.prepare()

    renamed = Fixture(tmp_path / "renamed-interpreter")
    python_path = Path("/usr/bin/python3").resolve(strict=True)
    raw = python_path.read_bytes()
    renamed.executable.write_bytes(raw)
    renamed.executable.chmod(0o755)
    operations = json.loads(renamed.operations.read_text(encoding="ascii"))
    for commands in operations["stages"].values():
        commands[0]["executable_sha256"] = digest(raw)
    renamed.operations.chmod(0o644)
    renamed.operations.write_bytes(canonical(operations))
    renamed.operations.chmod(0o444)
    with pytest.raises(FINAL.FinalActivationError, match="renamed command wrapper"):
        renamed.prepare()


@pytest.mark.parametrize(
    "target_attribute",
    [
        "candidate",
        "sq8_worker",
        "receipt",
        "sq8_artifact_manifest",
    ],
)
def test_candidate_runtime_swap_and_exact_restore_is_detected(
    tmp_path: Path,
    target_attribute: str,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    target = getattr(fixture, target_attribute)

    class SwapAndRestoreRunner(Runner):
        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            completed = super().__call__(argv, **kwargs)
            if self.stages[-1] == "candidate_reconciliation":
                held = target.with_name(f".{target.name}.held")
                target.rename(held)
                target.write_bytes(b"attacker-controlled-runtime\n")
                target.unlink()
                held.rename(target)
            return completed

    with pytest.raises(FINAL.FinalActivationError):
        execute_activation(fixture, SwapAndRestoreRunner())

    assert fixture.active.read_bytes() == fixture.aq4_raw
    outcome = json.loads(fixture.activation_outcome.read_text(encoding="ascii"))
    assert outcome["status"] == "failed_restored"
    assert outcome["failure_stage"] == "candidate_reconciliation"


def test_changed_aq4_worker_blocks_unsafe_manifest_restoration(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()

    class ReplaceAQ4AndFailRunner(Runner):
        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            completed = super().__call__(argv, **kwargs)
            if self.stages[-1] == "candidate_live_health":
                fixture.aq4_worker.write_bytes(b"attacker-aq4-worker\n")
                return subprocess.CompletedProcess(argv, 23, "", "")
            return completed

    with pytest.raises(FINAL.FinalActivationError):
        execute_activation(fixture, ReplaceAQ4AndFailRunner())

    assert fixture.active.read_bytes() == fixture.sq8_raw
    outcome = json.loads(fixture.activation_outcome.read_text(encoding="ascii"))
    assert outcome["status"] == "failed_restore"
    assert outcome["stages"]["aq4_restore"] == "failed"


def test_manual_rollback_survives_missing_sq8_runtime_and_evidence(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    candidate_executable = fixture.release_root / "candidate-operation"
    candidate_executable.write_bytes(fixture.executable.read_bytes())
    candidate_executable.chmod(0o755)
    operations = json.loads(fixture.operations.read_text(encoding="ascii"))
    for stage in ("candidate_reconciliation", "candidate_live_health"):
        operations["stages"][stage][0] = {
            "argv": [os.fspath(candidate_executable), stage],
            "executable_sha256": digest(candidate_executable.read_bytes()),
        }
    fixture.operations.chmod(0o644)
    fixture.operations.write_bytes(canonical(operations))
    fixture.operations.chmod(0o444)
    fixture.prepare()
    execute_activation(fixture, Runner())

    for path in (
        fixture.candidate,
        fixture.sq8_worker,
        fixture.receipt,
        fixture.sq8_promotion_evidence,
        fixture.bundle,
        candidate_executable,
    ):
        path.unlink()

    result = execute_rollback(fixture, Runner())
    assert result.status == "rolled_back"
    assert fixture.active.read_bytes() == fixture.aq4_raw


@pytest.mark.parametrize("unsafe_kind", ["ancestor", "setid"])
def test_reviewed_executable_runtime_metadata_is_sealed(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    fixture = Fixture(tmp_path)
    executable = fixture.executable
    if unsafe_kind == "ancestor":
        unsafe = fixture.root / "unsafe-executable-parent"
        unsafe.mkdir(mode=0o770)
        unsafe.chmod(0o770)
        executable = unsafe / "reviewed-operation"
        executable.write_bytes(fixture.executable.read_bytes())
        executable.chmod(0o755)
    else:
        executable.chmod(0o4755)
    operations = json.loads(fixture.operations.read_text(encoding="ascii"))
    for commands in operations["stages"].values():
        commands[0]["argv"][0] = os.fspath(executable)
        commands[0]["executable_sha256"] = digest(executable.read_bytes())
    fixture.operations.chmod(0o644)
    fixture.operations.write_bytes(canonical(operations))
    fixture.operations.chmod(0o444)

    with pytest.raises(FINAL.FinalActivationError, match="runtime artifact"):
        fixture.prepare()


def test_double_slash_runtime_destination_alias_is_rejected(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    operations = json.loads(fixture.operations.read_text(encoding="ascii"))
    original = operations["live_proofs"]["rollback_live_health"]["path"]
    operations["live_proofs"]["rollback_live_health"]["path"] = (
        f"//{original.lstrip('/')}"
    )
    fixture.operations.chmod(0o644)
    fixture.operations.write_bytes(canonical(operations))
    fixture.operations.chmod(0o444)

    with pytest.raises(FINAL.FinalActivationError, match="lexically canonical"):
        fixture.prepare()


@pytest.mark.parametrize(
    ("kind", "uid", "gid", "mode", "seal_uid"),
    [
        ("api", 0, 1000, 0o640, 0),
        ("jwt", 0, 1000, 0o640, 0),
    ],
)
def test_runtime_secret_fixed_private_metadata_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    uid: int,
    gid: int,
    mode: int,
    seal_uid: int,
) -> None:
    api_path = tmp_path / "api-key"
    jwt_path = tmp_path / "session.jwt"
    monkeypatch.setattr(FINAL.campaign_plan, "API_KEY_FILE", api_path)
    monkeypatch.setattr(
        FINAL.campaign_plan,
        "OPENWEBUI_SESSION_TOKEN_FILE",
        jwt_path,
    )
    selected = api_path if kind == "api" else jwt_path
    if kind == "jwt":
        monkeypatch.setattr(
            FINAL,
            "OPENWEBUI_SESSION_TOKEN_PARENT",
            jwt_path.parent,
        )

    def capture(
        path: Path,
        *,
        label: str,
        maximum: int,
        required_uid: int,
    ) -> object:
        del label, maximum
        assert path == selected
        assert required_uid == seal_uid
        identity = FINAL.runtime_seal.FileIdentity(
            1,
            2,
            stat.S_IFREG | mode,
            1,
            uid,
            gid,
            13,
            1,
            1,
        )
        return SimpleNamespace(
            snapshot=FINAL.StableFileSnapshot(
                selected,
                b"fixture-token\n",
                digest(b"fixture-token\n"),
                identity,
            ),
            ancestry=(
                SimpleNamespace(
                    path=selected.parent,
                    mode=stat.S_IFDIR | 0o750,
                    uid=0,
                    gid=1000,
                ),
            ),
        )

    monkeypatch.setattr(FINAL, "_capture_runtime_artifact", capture)
    secret = FINAL._read_runtime_secret(
        selected,
        "credential",
        required_uid=0,
    )
    assert bytes(secret) == b"fixture-token"


@pytest.mark.parametrize(
    ("kind", "uid", "gid", "mode"),
    [
        ("api", 0, 0, 0o640),
        ("api", 0, 1000, 0o600),
        ("jwt", 0, 1000, 0o600),
        ("jwt", 1000, 1000, 0o640),
    ],
)
def test_runtime_secret_wrong_metadata_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    uid: int,
    gid: int,
    mode: int,
) -> None:
    api_path = tmp_path / "api-key"
    jwt_path = tmp_path / "session.jwt"
    monkeypatch.setattr(FINAL.campaign_plan, "API_KEY_FILE", api_path)
    monkeypatch.setattr(
        FINAL.campaign_plan,
        "OPENWEBUI_SESSION_TOKEN_FILE",
        jwt_path,
    )
    selected = api_path if kind == "api" else jwt_path
    if kind == "jwt":
        monkeypatch.setattr(
            FINAL,
            "OPENWEBUI_SESSION_TOKEN_PARENT",
            jwt_path.parent,
        )

    def capture(*_args: object, **_kwargs: object) -> object:
        identity = FINAL.runtime_seal.FileIdentity(
            1,
            2,
            stat.S_IFREG | mode,
            1,
            uid,
            gid,
            6,
            1,
            1,
        )
        return SimpleNamespace(
            snapshot=FINAL.StableFileSnapshot(
                selected,
                b"token\n",
                digest(b"token\n"),
                identity,
            ),
            ancestry=(
                SimpleNamespace(
                    path=selected.parent,
                    mode=stat.S_IFDIR | 0o750,
                    uid=0,
                    gid=1000,
                ),
            ),
        )

    monkeypatch.setattr(FINAL, "_capture_runtime_artifact", capture)
    with pytest.raises(FINAL.FinalActivationError, match="private metadata"):
        FINAL._read_runtime_secret(
            selected,
            "credential",
            required_uid=0,
        )


def _git_for_source(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    environment = FINAL.source_seal.git_environment()
    environment.update(
        {
            "GIT_AUTHOR_EMAIL": "source-seal@example.invalid",
            "GIT_AUTHOR_NAME": "Source Seal Test",
            "GIT_COMMITTER_EMAIL": "source-seal@example.invalid",
            "GIT_COMMITTER_NAME": "Source Seal Test",
        }
    )
    return subprocess.run(
        FINAL.source_seal.git_argv(["-C", os.fspath(root), *arguments]),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        timeout=10.0,
        check=True,
    )


def _standalone_source(
    tmp_path: Path,
) -> tuple[Path, str, str]:
    root = tmp_path / "sealed-final-source"
    root.mkdir(mode=0o700)
    tools = root / "tools"
    tools.mkdir(mode=0o700)
    (tools / "served_model_final_activation.py").write_text(
        "FINAL_SOURCE_FIXTURE = True\n",
        encoding="ascii",
    )
    for name in FINAL.PRODUCTION_WRAPPER_NAMES:
        (tools / name).write_text(
            "#!/usr/bin/python3.12\n",
            encoding="ascii",
        )
    subprocess.run(
        ["/usr/bin/git", "init", "--quiet", os.fspath(root)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=FINAL.source_seal.git_environment(),
        timeout=10.0,
        check=True,
    )
    _git_for_source(root, "add", "--all")
    _git_for_source(root, "commit", "--quiet", "-m", "sealed fixture")
    commit = (
        _git_for_source(root, "rev-parse", "--verify", "HEAD^{commit}")
        .stdout.decode("ascii")
        .strip()
    )
    tree = (
        _git_for_source(root, "rev-parse", "--verify", "HEAD^{tree}")
        .stdout.decode("ascii")
        .strip()
    )
    _git_for_source(root, "checkout", "--quiet", "--detach", commit)
    _protect_source_tree(root)
    return root, commit, tree


def _protect_source_tree(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            path.chmod(0o700)
        elif stat.S_ISREG(metadata.st_mode):
            executable = bool(stat.S_IMODE(metadata.st_mode) & 0o100)
            path.chmod(0o700 if executable else 0o600)


def test_real_execution_source_seal_requires_detached_clean_standalone_git(
    tmp_path: Path,
) -> None:
    root, commit, tree = _standalone_source(tmp_path)
    marker = tmp_path / "fsmonitor-ran"
    monitor = tmp_path / "malicious-fsmonitor"
    monitor.write_text(
        f"#!/bin/sh\n/usr/bin/touch {marker}\n",
        encoding="ascii",
    )
    monitor.chmod(0o755)
    _git_for_source(root, "config", "core.fsmonitor", os.fspath(monitor))
    _protect_source_tree(root)

    sealed = REAL_CAPTURE_SOURCE_ROOT(
        root,
        expected_commit=commit,
        expected_tree=tree,
        required_uid=os.geteuid(),
    )

    assert sealed.root == root
    assert sealed.required_uid == os.geteuid()
    assert not marker.exists()

    _git_for_source(root, "switch", "--quiet", "-c", "unsafe-attached")
    _protect_source_tree(root)
    with pytest.raises(FINAL.FinalActivationError, match="detached"):
        REAL_CAPTURE_SOURCE_ROOT(
            root,
            expected_commit=commit,
            expected_tree=tree,
            required_uid=os.geteuid(),
        )


@pytest.mark.parametrize("fault", ["writable", "linked", "alternate"])
def test_real_execution_source_seal_rejects_unsafe_repository_forms(
    tmp_path: Path,
    fault: str,
) -> None:
    root, commit, tree = _standalone_source(tmp_path)
    selected = root
    if fault == "writable":
        root.chmod(0o770)
    elif fault == "linked":
        selected = tmp_path / "linked-worktree"
        _git_for_source(
            root,
            "worktree",
            "add",
            "--quiet",
            "--detach",
            os.fspath(selected),
            commit,
        )
    else:
        alternates = root / ".git/objects/info/alternates"
        alternates.write_text("/untrusted/object/store\n", encoding="ascii")

    with pytest.raises(
        FINAL.FinalActivationError,
        match="protected standalone",
    ):
        REAL_CAPTURE_SOURCE_ROOT(
            selected,
            expected_commit=commit,
            expected_tree=tree,
            required_uid=os.geteuid(),
        )


@pytest.mark.parametrize("field", ["commit", "tree"])
def test_real_execution_source_seal_rejects_plan_source_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    root, commit, tree = _standalone_source(tmp_path)
    expected_commit = "f" * 40 if field == "commit" else commit
    expected_tree = "f" * 40 if field == "tree" else tree

    with pytest.raises(FINAL.FinalActivationError, match=f"Git {field}"):
        REAL_CAPTURE_SOURCE_ROOT(
            root,
            expected_commit=expected_commit,
            expected_tree=expected_tree,
            required_uid=os.geteuid(),
        )


@pytest.mark.parametrize("phase", ["prepare", "load"])
def test_execution_source_is_admitted_before_local_validators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    fixture = Fixture(tmp_path)
    if phase == "load":
        fixture.prepare()
    validator_called = False

    def validator(_path: Path) -> dict[str, object]:
        nonlocal validator_called
        validator_called = True
        raise AssertionError("validator must not run")

    def reject_source(**_kwargs: object) -> object:
        raise FINAL.FinalActivationError("execution source rejected")

    monkeypatch.setattr(FINAL, "_capture_execution_source", reject_source)
    with pytest.raises(FINAL.FinalActivationError, match="execution source"):
        if phase == "prepare":
            FINAL.prepare_plan(
                plan_id="sq8-final-test-001",
                authorization_path=fixture.authorization_path,
                candidate_manifest=fixture.candidate,
                active_manifest=fixture.active,
                rollback_manifest=fixture.rollback,
                release_bundle=fixture.bundle,
                systemd_unit=fixture.unit,
                environment_file=fixture.environment,
                operations_document=fixture.operations,
                activation_outcome=fixture.activation_outcome,
                rollback_outcome=fixture.rollback_outcome,
                output=fixture.plan,
                now=NOW + timedelta(minutes=1),
                policy=fixture.policy,
                manifest_validator=validator,
                bundle_validator=fixture.bundle_validator,
            )
        else:
            FINAL.load_plan(
                fixture.plan,
                action="activate",
                now=NOW + timedelta(minutes=2),
                policy=fixture.policy,
                manifest_validator=validator,
                bundle_validator=fixture.bundle_validator,
            )
    assert validator_called is False


@pytest.mark.parametrize(
    ("mutation_point", "expected_prefix"),
    [
        ("before_candidate_command", []),
        ("after_candidate_command", ["candidate_reconciliation"]),
    ],
)
def test_execution_source_repin_failure_restores_exact_aq4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_point: str,
    expected_prefix: list[str],
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    runner = Runner()
    original = FINAL._require_execution_source
    raised = False

    def fail_once(
        expected: object,
        *,
        expected_commit: str,
        expected_tree: str,
        required_uid: int,
    ) -> None:
        nonlocal raised
        armed = fixture.active.read_bytes() == fixture.sq8_raw
        if mutation_point == "after_candidate_command":
            armed = armed and runner.stages == ["candidate_reconciliation"]
        else:
            armed = armed and not runner.stages
        if armed and not raised:
            raised = True
            raise FINAL.FinalActivationError("execution source seal changed")
        original(
            expected,
            expected_commit=expected_commit,
            expected_tree=expected_tree,
            required_uid=required_uid,
        )

    monkeypatch.setattr(FINAL, "_require_execution_source", fail_once)
    with pytest.raises(FINAL.FinalActivationError):
        execute_activation(fixture, runner)

    assert raised is True
    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert runner.stages[: len(expected_prefix)] == expected_prefix
    assert "reverse_reconciliation" in runner.stages
    assert "rollback_live_health" in runner.stages


def test_rollback_core_repins_source_without_campaign_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    execute_activation(fixture, Runner())
    record = fixture.load("rollback")
    damage_campaign_authority(fixture, target="authorization", fault="delete")
    damage_campaign_authority(fixture, target="outcome", fault="delete")
    original = FINAL._require_execution_source
    calls = 0

    def observe(
        expected: object,
        *,
        expected_commit: str,
        expected_tree: str,
        required_uid: int,
    ) -> None:
        nonlocal calls
        calls += 1
        original(
            expected,
            expected_commit=expected_commit,
            expected_tree=expected_tree,
            required_uid=required_uid,
        )

    monkeypatch.setattr(FINAL, "_require_execution_source", observe)
    FINAL._repin_rollback_inputs(
        record,
        policy=fixture.policy,
        manifest_validator=fixture.manifest_validator,
        include_shared=False,
    )

    assert calls == 2
    assert fixture.active.read_bytes() == fixture.sq8_raw


def test_production_entrypoint_requires_exact_canonical_python_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sealed-entrypoint"
    tools = root / "tools"
    tools.mkdir(parents=True)
    wrapper = tools / "run-served-model-final-activation.py"
    wrapper.write_text("#!/usr/bin/python3.12\n", encoding="ascii")
    sealed = SimpleNamespace(root=root, required_uid=0)
    monkeypatch.setattr(FINAL, "_module_execution_root", lambda: root)
    monkeypatch.setattr(FINAL.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        FINAL.sys,
        "flags",
        SimpleNamespace(
            isolated=1,
            no_site=1,
            dont_write_bytecode=1,
            safe_path=True,
        ),
    )
    monkeypatch.setattr(
        FINAL.sys,
        "orig_argv",
        [
            "/usr/bin/python3.12",
            "-I",
            "-S",
            "-B",
            os.fspath(wrapper),
            "--plan",
            "/run/plan.json",
        ],
    )
    monkeypatch.setattr(
        FINAL.source_seal,
        "capture_source_seal",
        lambda _root, required_uid: sealed,
    )
    monkeypatch.setattr(
        FINAL,
        "_source_git",
        lambda _root, arguments, _label: (
            b"a" * 40 + b"\n"
            if arguments[-1] == "HEAD^{commit}"
            else b"b" * 40 + b"\n"
        ),
    )
    monkeypatch.setattr(FINAL, "_require_execution_source", lambda *_args, **_kwargs: None)

    REAL_REQUIRE_PRODUCTION_ENTRYPOINT(wrapper)

    FINAL.sys.orig_argv[0] = "/usr/bin/python3"
    with pytest.raises(FINAL.FinalActivationError, match="exact"):
        REAL_REQUIRE_PRODUCTION_ENTRYPOINT(wrapper)
    FINAL.sys.orig_argv[0] = "/usr/bin/python3.12"
    with pytest.raises(FINAL.FinalActivationError, match="exact"):
        REAL_REQUIRE_PRODUCTION_ENTRYPOINT(Path(wrapper.name))


@pytest.mark.parametrize("path", [PREPARE_PATH, RUNNER_PATH, ROLLBACK_PATH])
def test_production_wrappers_are_guarded_and_have_fixed_python_shebang(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_cli(f"test_guarded_{path.stem}", path)
    called = False

    def reject(_path: Path) -> None:
        nonlocal called
        called = True
        raise module.final_activation.FinalActivationError("guarded")

    monkeypatch.setattr(
        module.final_activation,
        "require_production_entrypoint",
        reject,
    )
    assert path.read_text(encoding="utf-8").startswith("#!/usr/bin/python3.12\n")
    assert module.main([]) == 1
    assert called is True
    direct = subprocess.run(
        ["/usr/bin/python3.12", os.fspath(path), "--help"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10.0,
        check=False,
    )
    assert direct.returncode != 0
    assert b"production wrapper requires exact root" in direct.stderr


def test_final_activation_runbook_uses_only_sealed_absolute_python_entrypoints() -> None:
    raw = RUNBOOK_PATH.read_text(encoding="utf-8")
    prefix = (
        "sudo -- /usr/bin/python3.12 -I -S -B "
        "/ABSOLUTE/ROOT-OWNED-SEALED-SQ8-SOURCE/tools/"
    )

    assert "sudo tools/" not in raw
    assert "sudo -- tools/" not in raw
    assert raw.count(prefix) == 5
    assert f"{prefix}prepare-served-model-final-activation.py" in raw
    assert f"{prefix}run-served-model-final-activation.py" in raw
    assert f"{prefix}rollback-served-model.py" in raw


def test_openwebui_session_secret_requires_fixed_root_owned_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jwt_path = tmp_path / "openwebui-session.jwt"
    monkeypatch.setattr(
        FINAL.campaign_plan,
        "OPENWEBUI_SESSION_TOKEN_FILE",
        jwt_path,
    )
    monkeypatch.setattr(
        FINAL,
        "OPENWEBUI_SESSION_TOKEN_PARENT",
        jwt_path.parent,
    )
    identity = FINAL.runtime_seal.FileIdentity(
        1,
        2,
        stat.S_IFREG | 0o640,
        1,
        0,
        1000,
        6,
        1,
        1,
    )
    sealed = SimpleNamespace(
        snapshot=FINAL.StableFileSnapshot(
            jwt_path,
            b"token\n",
            digest(b"token\n"),
            identity,
        ),
        ancestry=(
            SimpleNamespace(
                path=jwt_path.parent,
                mode=stat.S_IFDIR | 0o770,
                uid=0,
                gid=1000,
            ),
        ),
    )
    monkeypatch.setattr(
        FINAL,
        "_capture_runtime_artifact",
        lambda *_args, **_kwargs: sealed,
    )

    with pytest.raises(FINAL.FinalActivationError, match="parent metadata"):
        FINAL._runtime_secret_seal(
            jwt_path,
            "OpenWebUI session token",
            required_uid=0,
        )
