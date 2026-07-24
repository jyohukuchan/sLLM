from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/served_model_campaign_transaction.py"
SPEC = importlib.util.spec_from_file_location(
    "test_served_model_campaign_transaction_module",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
TX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TX
SPEC.loader.exec_module(TX)
AUTH = TX.authorization

RECOVERY_PATH = ROOT / "tools" / "served_model_campaign_recovery.py"
RECOVERY_SPEC = importlib.util.spec_from_file_location(
    "test_served_model_campaign_recovery_module",
    RECOVERY_PATH,
)
assert RECOVERY_SPEC is not None and RECOVERY_SPEC.loader is not None
RECOVERY = importlib.util.module_from_spec(RECOVERY_SPEC)
sys.modules[RECOVERY_SPEC.name] = RECOVERY
RECOVERY_SPEC.loader.exec_module(RECOVERY)

NOW = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)
SOURCE_COMMIT = "a" * 40
SOURCE_TREE = "b" * 40
AQ4_SOURCE_COMMIT = "c" * 40
AQ4_SOURCE_TREE = "d" * 40
AQ4_WORKER_RAW = b"fixture AQ4 worker binary\n"
AQ4_WORKER = hashlib.sha256(AQ4_WORKER_RAW).hexdigest()
SQ8_WORKER_RAW = b"fixture SQ8 worker binary\n"
SQ8_WORKER = hashlib.sha256(SQ8_WORKER_RAW).hexdigest()
ORIGINAL_GIT_COMMAND_PREFIX = TX.GIT_COMMAND_PREFIX


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def is_git(argv: list[str]) -> bool:
    return argv[: len(TX.GIT_COMMAND_PREFIX)] == list(TX.GIT_COMMAND_PREFIX)


def git_arguments(argv: list[str]) -> tuple[str, ...]:
    assert is_git(argv)
    return tuple(argv[len(TX.GIT_COMMAND_PREFIX) :])


class Fixture:
    def __init__(
        self,
        tmp_path: Path,
        *,
        authorization_lifetime: timedelta = timedelta(hours=1),
    ) -> None:
        self.root = tmp_path
        self.source = tmp_path / "source"
        self.source.mkdir()
        self.source.chmod(0o755)
        (self.source / ".git").mkdir()
        (self.source / ".git").chmod(0o755)
        (self.source / "tools").mkdir()
        (self.source / "tools").chmod(0o755)
        source_stage = self.source / "tools" / "fixture-stage.py"
        source_stage.write_text(
            "raise SystemExit(0)\n",
            encoding="ascii",
        )
        source_stage.chmod(0o644)
        self.docker_wrapper = (
            self.source / TX.DOCKER_LEASE_WRAPPER_RELATIVE_PATH
        )
        self.docker_wrapper.write_bytes(b"#!/bin/sh\nexit 0\n")
        self.docker_wrapper.chmod(0o555)
        self.aq4_source = tmp_path / "aq4-source"
        self.aq4_source.mkdir()
        self.aq4_source.chmod(0o755)
        (self.aq4_source / ".git").mkdir()
        (self.aq4_source / ".git").chmod(0o755)
        (self.aq4_source / "tools").mkdir()
        (self.aq4_source / "tools").chmod(0o755)
        aq4_source_stage = self.aq4_source / "tools" / "fixture-stage.py"
        aq4_source_stage.write_text(
            "raise SystemExit(0)\n",
            encoding="ascii",
        )
        aq4_source_stage.chmod(0o644)
        self.slot = tmp_path / "slot"
        self.slot.mkdir()
        self.slot.chmod(0o700)
        self.command_root = self.slot / "commands"
        self.command_root.mkdir()
        self.command_root.chmod(0o700)
        self.candidate_command = self.command_root / "candidate-command"
        self.candidate_command.write_bytes(b"#!/bin/sh\nexit 0\n")
        self.candidate_command.chmod(0o555)
        self.aq4_command = self.command_root / "aq4-command"
        self.aq4_command.write_bytes(b"#!/bin/sh\nexit 0\n")
        self.aq4_command.chmod(0o555)
        self.shared_command = self.command_root / "shared-command"
        self.shared_command.write_bytes(b"#!/bin/sh\nexit 0\n")
        self.shared_command.chmod(0o555)
        TX.PYTHON_BINARY = str(self.shared_command)
        TX.DOCKER_BINARY = str(self.shared_command)
        TX.GIT_COMMAND_PREFIX = (
            str(self.shared_command),
            *ORIGINAL_GIT_COMMAND_PREFIX[1:],
        )
        TX.campaign_source_seal.GIT_COMMAND_PREFIX = TX.GIT_COMMAND_PREFIX
        self.outputs = tmp_path / "outputs"
        self.outputs.mkdir(mode=0o700)
        self.outputs.chmod(0o700)
        self.claims = tmp_path / "claims"
        self.claims.mkdir(mode=0o700)
        self.outcomes = tmp_path / "outcomes"
        self.outcomes.mkdir(mode=0o700)
        self.policy = AUTH.RegistryPolicy(
            claim_registry=self.claims,
            outcome_registry=self.outcomes,
            required_uid=os.geteuid(),
        )
        self.receipt = self.slot / "sq8-promotion-receipt.json"
        self.receipt.write_bytes(b'{"schema_version":"ullm.sq8_serving_promotion.v1"}\n')
        self.receipt.chmod(0o644)
        self.sq8_worker = self.slot / "ullm-sq8-worker"
        self.sq8_worker.write_bytes(SQ8_WORKER_RAW)
        self.sq8_worker.chmod(0o555)
        self.aq4_worker = self.slot / "ullm-aq4-worker"
        self.aq4_worker.write_bytes(AQ4_WORKER_RAW)
        self.aq4_worker.chmod(0o555)
        self.sq8_tokenizer = self.slot / "sq8-tokenizer"
        self.sq8_tokenizer.mkdir()
        self.sq8_tokenizer.chmod(0o755)
        (self.sq8_tokenizer / "tokenizer.json").write_bytes(
            b'{"fixture":"sq8-tokenizer"}\n'
        )
        (self.sq8_tokenizer / "tokenizer.json").chmod(0o644)
        self.aq4_tokenizer = self.slot / "aq4-tokenizer"
        self.aq4_tokenizer.mkdir()
        self.aq4_tokenizer.chmod(0o755)
        (self.aq4_tokenizer / "tokenizer.json").write_bytes(
            b'{"fixture":"aq4-tokenizer"}\n'
        )
        (self.aq4_tokenizer / "tokenizer.json").chmod(0o644)
        self.sq8_product = self.slot / "sq8-product"
        (self.sq8_product / "artifact").mkdir(parents=True)
        (self.sq8_product / "package").mkdir()
        (self.sq8_product / "artifact" / "weight.bin").write_bytes(
            b"sq8 artifact payload\n"
        )
        (self.sq8_product / "artifact" / "sq_manifest.json").write_bytes(
            b'{"schema_version":"sq-fp8-artifact-v0.2"}\n'
        )
        (self.sq8_product / "package" / "weight.bin").write_bytes(
            b"sq8 package payload\n"
        )
        (self.sq8_product / "package" / "manifest.json").write_bytes(
            b'{"schema_version":"ullm-prototype-manifest-v0.1"}\n'
        )
        self.sq8_product.chmod(0o755)
        (self.sq8_product / "artifact").chmod(0o755)
        (self.sq8_product / "package").chmod(0o755)
        for artifact in self.sq8_product.rglob("*"):
            if artifact.is_file():
                artifact.chmod(0o644)
        self.aq4_product = self.slot / "aq4-product"
        (self.aq4_product / "package").mkdir(parents=True)
        (self.aq4_product / "package" / "weight.bin").write_bytes(
            b"aq4 package payload\n"
        )
        (self.aq4_product / "package" / "manifest.json").write_bytes(
            b'{"schema_version":"ullm-prototype-manifest-v0.1"}\n'
        )
        self.aq4_product.chmod(0o755)
        (self.aq4_product / "package").chmod(0o755)
        for artifact in self.aq4_product.rglob("*"):
            if artifact.is_file():
                artifact.chmod(0o644)
        self.aq4_bundle_root = self.outputs / "aq4-bundle-components"
        self.aq4_bundle_root.mkdir(mode=0o700)
        self.aq4_promotion_evidence = (
            self.slot / "promotion-evidence.json"
        )
        self.aq4_promotion_evidence.write_text(
            json.dumps(
                {
                    "schema_version": (
                        "ullm.aq4_resident_promotion_evidence.v1"
                    ),
                    "source_commit": AQ4_SOURCE_COMMIT,
                    "worker_binary": str(self.aq4_worker),
                    "worker_binary_sha256": AQ4_WORKER,
                    "verified": True,
                    "production_receipt_written": False,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="ascii",
        )
        self.aq4_promotion_evidence.chmod(0o644)
        self.aq4_promotion_receipt = (
            self.slot / "promotion-receipt.json"
        )
        self.aq4_promotion_receipt.write_text(
            json.dumps(
                {
                    "schema_version": "ullm.aq4_resident_promotion.v1",
                    "source_commit": AQ4_SOURCE_COMMIT,
                    "evidence": {
                        "path": self.aq4_promotion_evidence.name,
                        "sha256": digest(
                            self.aq4_promotion_evidence.read_bytes()
                        ),
                    },
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="ascii",
        )
        self.aq4_promotion_receipt.chmod(0o644)
        self.active = self.slot / "active.json"
        self.candidate = self.slot / "candidate.json"
        self.aq4_raw = (
            json.dumps(
                {
                    "schema_version": "ullm.served_model.v2",
                    "tokenizer": {
                        "root": str(self.aq4_tokenizer),
                        "files": {
                            "tokenizer.json": digest(
                                (
                                    self.aq4_tokenizer / "tokenizer.json"
                                ).read_bytes()
                            )
                        },
                    },
                    "worker": {
                        "binary": str(self.aq4_worker),
                        "binary_sha256": AQ4_WORKER,
                    },
                    "product": {
                        "root": str(self.aq4_product),
                        "artifact": None,
                        "package": {
                            "manifest_path": "package/manifest.json",
                            "manifest_sha256": digest(
                                (
                                    self.aq4_product
                                    / "package"
                                    / "manifest.json"
                                ).read_bytes()
                            ),
                        },
                    },
                    "promotion": {
                        "source_commit": AQ4_SOURCE_COMMIT,
                        "receipt": str(self.aq4_promotion_receipt),
                        "receipt_sha256": digest(
                            self.aq4_promotion_receipt.read_bytes()
                        ),
                    },
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
        self.sq8_raw = (
            json.dumps(
                {
                    "schema_version": "ullm.served_model.v2",
                    "tokenizer": {
                        "root": str(self.sq8_tokenizer),
                        "files": {
                            "tokenizer.json": digest(
                                (
                                    self.sq8_tokenizer / "tokenizer.json"
                                ).read_bytes()
                            )
                        },
                    },
                    "worker": {
                        "binary": str(self.sq8_worker),
                        "binary_sha256": SQ8_WORKER,
                    },
                    "product": {
                        "root": str(self.sq8_product),
                        "artifact": {
                            "manifest_path": "artifact/sq_manifest.json",
                            "manifest_sha256": digest(
                                (
                                    self.sq8_product
                                    / "artifact"
                                    / "sq_manifest.json"
                                ).read_bytes()
                            ),
                        },
                        "package": {
                            "manifest_path": "package/manifest.json",
                            "manifest_sha256": digest(
                                (
                                    self.sq8_product
                                    / "package"
                                    / "manifest.json"
                                ).read_bytes()
                            ),
                        },
                    },
                    "promotion": {
                        "source_commit": SOURCE_COMMIT,
                        "receipt": str(self.receipt),
                        "receipt_sha256": digest(self.receipt.read_bytes()),
                    },
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
        self.active.write_bytes(self.aq4_raw)
        self.active.chmod(0o644)
        self.candidate.write_bytes(self.sq8_raw)
        self.candidate.chmod(0o644)
        self.policy = AUTH.RegistryPolicy(
            claim_registry=self.claims,
            outcome_registry=self.outcomes,
            required_uid=os.geteuid(),
            active_manifest_path=self.active,
            systemd_unit_path=self.slot / "ullm-openai.service",
            environment_file_path=self.slot / "ullm-openai.env",
            service_unit="ullm-openai.service",
        )
        self.unit = self.slot / "ullm-openai.service"
        self.environment = self.slot / "ullm-openai.env"
        self.unit.write_bytes(b"[Service]\nExecStart=/usr/bin/ullm\n")
        self.environment.write_bytes(b"ULLM_TEST=1\n")
        self.unit.chmod(0o644)
        self.environment.chmod(0o644)
        self.backup = self.outputs / "aq4-backup.json"
        self.campaign_paths = {
            "aq4_reasoning_release": self.outputs / "aq4-reasoning-release",
            "aq4_reasoning_browser": (
                self.aq4_bundle_root / "browser-evidence.json"
            ),
            "aq4_bundle": self.aq4_bundle_root / "bundle.json",
            "sq8_full": self.outputs / "sq8-full",
            "reasoning_release": self.outputs / "reasoning-release",
            "reasoning_browser": self.outputs / "reasoning-browser.json",
        }
        self.authorization_path = tmp_path / "authorization.json"
        self.authorization_document = {
            "schema_version": AUTH.AUTHORIZATION_SCHEMA,
            "authorization_id": "sq8-window-test-001",
            "issued_at": AUTH.utc_timestamp(NOW - timedelta(minutes=1)),
            "expires_at": AUTH.utc_timestamp(NOW + authorization_lifetime),
            "max_attempts": 1,
            "authorization_note": "Fixture-only private manifest transaction.",
            "purpose": "temporary_candidate_active_evidence_collection_only",
            "required_final_route": "restore_exact_aq4_then_bundle_v2_activation",
            "source": {"commit": SOURCE_COMMIT, "tree": SOURCE_TREE},
            "before": {
                "model_id": "ullm-qwen3.5-9b-aq4",
                "format_id": "AQ4_0",
                "manifest_sha256": digest(self.aq4_raw),
                "worker_protocol": "ullm.worker.v2",
                "worker_binary_path": str(self.aq4_worker),
                "worker_binary_sha256": AQ4_WORKER,
                "promotion_source_commit": AQ4_SOURCE_COMMIT,
                "promotion_receipt_path": str(
                    self.aq4_promotion_receipt
                ),
                "promotion_receipt_sha256": digest(
                    self.aq4_promotion_receipt.read_bytes()
                ),
            },
            "aq4_release": {
                "source": {
                    "root": str(self.aq4_source),
                    "commit": AQ4_SOURCE_COMMIT,
                    "tree": AQ4_SOURCE_TREE,
                },
                "openwebui_image": AUTH.FIXED_OPENWEBUI_IMAGE,
                "promotion_evidence": {
                    "source_path": str(self.aq4_promotion_evidence),
                    "path": str(
                        self.aq4_bundle_root
                        / self.aq4_promotion_evidence.name
                    ),
                    "sha256": digest(
                        self.aq4_promotion_evidence.read_bytes()
                    ),
                },
                "promotion_receipt": {
                    "source_path": str(self.aq4_promotion_receipt),
                    "path": str(
                        self.aq4_bundle_root
                        / self.aq4_promotion_receipt.name
                    ),
                    "sha256": digest(
                        self.aq4_promotion_receipt.read_bytes()
                    ),
                },
                "release_evidence_path": str(
                    self.aq4_bundle_root / "release-evidence.json"
                ),
                "release_validator_path": str(
                    self.aq4_bundle_root / "release-validator.json"
                ),
                "browser_validator_path": str(
                    self.aq4_bundle_root / "browser-validator.json"
                ),
            },
            "candidate": {
                "model_id": "ullm-qwen3-14b-sq8",
                "format_id": "SQ8_0",
                "manifest_sha256": digest(self.sq8_raw),
                "worker_protocol": "ullm.worker.v2",
                "worker_binary_sha256": SQ8_WORKER,
                "promotion_source_commit": SOURCE_COMMIT,
                "promotion_receipt_sha256": digest(self.receipt.read_bytes()),
            },
            "campaigns": {
                name: {
                    "run_id": f"{name}-run-001",
                    "final_path": str(path),
                }
                for name, path in self.campaign_paths.items()
            },
            "rollback": {
                "backup_path": str(self.backup),
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
        self.commands = TX.TransactionCommands(
            candidate_reconciliation=((str(self.candidate_command),),),
            candidate_checks=((str(self.candidate_command),),),
            sq8_full=(
                str(self.candidate_command),
                "--final-path",
                str(self.campaign_paths["sq8_full"]),
                "--docker",
                str(self.docker_wrapper),
            ),
            reasoning_release=(
                str(self.candidate_command),
                "--output-dir",
                str(self.campaign_paths["reasoning_release"]),
            ),
            reasoning_browser=(
                str(self.candidate_command),
                "--output",
                str(self.campaign_paths["reasoning_browser"]),
            ),
            reverse_reconciliation=((str(self.aq4_command),),),
            aq4_reasoning_release=(
                (
                    str(self.aq4_command),
                    "--output-dir",
                    str(self.campaign_paths["aq4_reasoning_release"]),
                ),
                (
                    str(self.aq4_command),
                    "--output",
                    self.authorization_document["aq4_release"][
                        "release_evidence_path"
                    ],
                ),
                (
                    str(self.aq4_command),
                    "--evidence",
                    self.authorization_document["aq4_release"][
                        "release_evidence_path"
                    ],
                    "--output",
                    self.authorization_document["aq4_release"][
                        "release_validator_path"
                    ],
                ),
            ),
            aq4_reasoning_browser=(
                (
                    str(self.aq4_command),
                    "--output",
                    str(self.campaign_paths["aq4_reasoning_browser"]),
                ),
                (
                    str(self.aq4_command),
                    "--evidence",
                    str(self.campaign_paths["aq4_reasoning_browser"]),
                    "--output",
                    self.authorization_document["aq4_release"][
                        "browser_validator_path"
                    ],
                ),
            ),
            aq4_bundle=(
                (
                    str(self.aq4_command),
                    "--output",
                    str(self.campaign_paths["aq4_bundle"]),
                ),
                (
                    str(self.aq4_command),
                    str(self.campaign_paths["aq4_bundle"]),
                ),
            ),
            final_checks=(
                (
                    str(self.aq4_command),
                    "--docker",
                    str(self.docker_wrapper),
                ),
            ),
        )
        self.request = TX.TransactionRequest(
            authorization_path=self.authorization_path,
            source_root=self.source,
            candidate_manifest=self.candidate,
            active_manifest=self.active,
            systemd_unit=self.unit,
            environment_file=self.environment,
            inactive_services=("ullm-openai.service",),
            commands=self.commands,
            command_timeout_seconds=10.0,
        )

    def validator(self, path: Path) -> dict[str, object]:
        raw = path.read_bytes()
        if raw == self.aq4_raw:
            return {
                "validated": True,
                "manifest_sha256": digest(raw),
                "model_id": "ullm-qwen3.5-9b-aq4",
                "format_id": "AQ4_0",
                "worker": {
                    "binary": str(self.aq4_worker),
                    "protocol": "ullm.worker.v2",
                    "binary_sha256": AQ4_WORKER,
                },
            }
        if raw == self.sq8_raw:
            return {
                "validated": True,
                "manifest_sha256": digest(raw),
                "model_id": "ullm-qwen3-14b-sq8",
                "format_id": "SQ8_0",
                "worker": {
                    "binary": str(self.sq8_worker),
                    "protocol": "ullm.worker.v2",
                    "binary_sha256": SQ8_WORKER,
                },
            }
        raise ValueError("unknown manifest")


class Runner:
    def __init__(
        self,
        fixture: Fixture,
        *,
        fail_stage: str | None = None,
        interrupt_stage: str | None = None,
        mutate_active_stage: str | None = None,
    ) -> None:
        self.fixture = fixture
        self.fail_stage = fail_stage
        self.interrupt_stage = interrupt_stage
        self.mutate_active_stage = mutate_active_stage
        self.stage_calls: list[str] = []
        self.stage_call_counts: dict[str, int] = {}
        self.docker_lease_clock = 0.0

    def docker_lease_monotonic(self) -> float:
        return self.docker_lease_clock

    def docker_lease_sleep(self, seconds: float) -> None:
        self.docker_lease_clock += seconds

    @staticmethod
    def _argument(argv: list[str], flag: str) -> str:
        assert argv.count(flag) == 1
        index = argv.index(flag)
        assert index + 1 < len(argv)
        return argv[index + 1]

    def _write_v2_artifacts(
        self,
        output: Path,
        *,
        campaign_name: str,
        evidence_name: str,
        evidence_schema: str,
        files: frozenset[str],
    ) -> None:
        campaign = self.fixture.authorization_document["campaigns"][
            campaign_name
        ]
        binding = {
            "schema_version": "ullm.served_model.active_binding.v1",
            "status": "complete",
            "campaign": {
                "name": campaign_name,
                "run_id": campaign["run_id"],
                "final_path": campaign["final_path"],
            },
            "candidate": {"sha256": digest(self.fixture.sq8_raw)},
        }
        for relative in sorted(files):
            artifact = output / relative
            if relative == "candidate-served-model.json":
                artifact.write_bytes(self.fixture.sq8_raw)
            elif relative == "active-manifest-binding.json":
                artifact.write_text(
                    json.dumps(binding, separators=(",", ":"), sort_keys=True)
                    + "\n",
                    encoding="ascii",
                )
            elif relative == evidence_name:
                artifact.write_text(
                    json.dumps(
                        {"schema_version": evidence_schema},
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="ascii",
                )
            else:
                artifact.write_text(f"{relative}\n", encoding="ascii")
            artifact.chmod(0o444)

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if is_git(argv):
            arguments = git_arguments(argv)
            selected_root = Path(str(kwargs["cwd"]))
            aq4 = selected_root == self.fixture.aq4_source
            values = {
                ("rev-parse", "--show-toplevel"): str(selected_root),
                ("rev-parse", "HEAD"): (
                    AQ4_SOURCE_COMMIT if aq4 else SOURCE_COMMIT
                ),
                ("rev-parse", "HEAD^{tree}"): (
                    AQ4_SOURCE_TREE if aq4 else SOURCE_TREE
                ),
                ("rev-parse", "--abbrev-ref", "HEAD"): (
                    "HEAD" if aq4 else "main"
                ),
                (
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    "--ignore-submodules=all",
                    "--no-renames",
                ): "",
            }
            return subprocess.CompletedProcess(
                argv,
                0,
                values[arguments] + ("\n" if values[arguments] else ""),
                "",
            )
        stage = str(kwargs["env"]["ULLM_CAMPAIGN_TRANSACTION_STAGE"])
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        self.stage_calls.append(stage)
        self.stage_call_counts[stage] = self.stage_call_counts.get(stage, 0) + 1
        stage_call = self.stage_call_counts[stage]
        if stage == self.interrupt_stage:
            raise TX.TransactionInterrupted("fixture interruption")
        if stage == self.fail_stage:
            return subprocess.CompletedProcess(argv, 19, "", "")
        if stage == "sq8_full":
            assert self._argument(argv, "--final-path") == str(
                self.fixture.campaign_paths["sq8_full"]
            )
            output = Path(
                environment[TX.CAMPAIGN_STAGING_OUTPUT_ENVIRONMENT]
            )
            output.mkdir(mode=0o700)
            browser = output / "browser"
            browser.mkdir(mode=0o700)
            for relative in sorted(TX.SQ8_FULL_V2_FILES):
                artifact = output / relative
                artifact.parent.mkdir(mode=0o700, exist_ok=True)
                if relative == "candidate-served-model.json":
                    artifact.write_bytes(self.fixture.sq8_raw)
                else:
                    artifact.write_text(f"{relative}\n", encoding="ascii")
                artifact.chmod(0o600)
            browser.chmod(0o700)
            output.chmod(0o700)
        elif stage == "reasoning_release":
            assert self._argument(argv, "--output-dir") == str(
                self.fixture.campaign_paths["reasoning_release"]
            )
            output = Path(
                environment[TX.CAMPAIGN_STAGING_OUTPUT_ENVIRONMENT]
            )
            output.mkdir(mode=0o700)
            self._write_v2_artifacts(
                output,
                campaign_name="reasoning_release",
                evidence_name="summary.json",
                evidence_schema="ullm.generic_reasoning_release_campaign.v2",
                files=TX.REASONING_RELEASE_V2_FILES,
            )
            output.chmod(0o555)
        elif stage == "reasoning_browser":
            assert self._argument(argv, "--output") == str(
                self.fixture.campaign_paths["reasoning_browser"]
            )
            output = Path(
                environment[TX.CAMPAIGN_STAGING_OUTPUT_ENVIRONMENT]
            )
            output.mkdir(mode=0o700)
            self._write_v2_artifacts(
                output,
                campaign_name="reasoning_browser",
                evidence_name="browser-evidence.json",
                evidence_schema="ullm.openwebui.reasoning_browser_smoke.v5",
                files=TX.REASONING_BROWSER_V2_FILES,
            )
            output.chmod(0o555)
        elif stage == "aq4_reasoning_release":
            if stage_call == 1:
                assert self._argument(argv, "--output-dir") == str(
                    self.fixture.campaign_paths["aq4_reasoning_release"]
                )
                output = Path(
                    environment[TX.CAMPAIGN_STAGING_OUTPUT_ENVIRONMENT]
                )
                output.mkdir(mode=0o700)
                for relative in sorted(TX.AQ4_REASONING_RELEASE_FILES):
                    artifact = output / relative
                    if relative == "summary.json":
                        value = {
                            "schema_version": (
                                "ullm.generic_reasoning_release_campaign.v1"
                            ),
                            "status": "incomplete",
                            "manifest_sha256": digest(self.fixture.aq4_raw),
                            "model_id": "ullm-qwen3.5-9b-aq4",
                        }
                        artifact.write_text(
                            json.dumps(
                                value,
                                separators=(",", ":"),
                                sort_keys=True,
                            )
                            + "\n",
                            encoding="ascii",
                        )
                    else:
                        artifact.write_text(
                            f"{relative}\n",
                            encoding="ascii",
                        )
                    artifact.chmod(0o600)
            elif stage_call == 2:
                output = Path(self._argument(argv, "--output"))
                evidence = {
                    "schema_version": (
                        "ullm.generic_reasoning_release_evidence.v1"
                    ),
                    "status": "complete",
                    "source_commit": AQ4_SOURCE_COMMIT,
                    "active_promotion_source_commit": AQ4_SOURCE_COMMIT,
                    "source_commit_aligned": True,
                    "git_worktree_clean": True,
                    "identity": {
                        "manifest_sha256": digest(self.fixture.aq4_raw),
                        "worker_binary_sha256": AQ4_WORKER,
                        "tokenizer_sha256": "9" * 64,
                        "openwebui_image": AUTH.FIXED_OPENWEBUI_IMAGE,
                    },
                }
                output.write_text(
                    json.dumps(
                        evidence,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="ascii",
                )
                output.chmod(0o600)
            elif stage_call == 3:
                assert self._argument(argv, "--evidence") == (
                    self.fixture.authorization_document["aq4_release"][
                        "release_evidence_path"
                    ]
                )
                report = {
                    "schema_version": (
                        "ullm.generic_reasoning_release_validator.v1"
                    ),
                    "input_schema_version": (
                        "ullm.generic_reasoning_release_evidence.v1"
                    ),
                    "structurally_valid": True,
                    "gate_eligible": True,
                }
                target = Path(self._argument(argv, "--output"))
                target.write_text(
                    json.dumps(
                        report,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="ascii",
                )
                target.chmod(0o444)
        elif stage == "aq4_reasoning_browser":
            if stage_call == 1:
                assert self._argument(argv, "--output") == str(
                    self.fixture.campaign_paths["aq4_reasoning_browser"]
                )
                output = Path(
                    environment[TX.CAMPAIGN_STAGING_OUTPUT_ENVIRONMENT]
                )
                output.write_text(
                    '{"schema_version":'
                    '"ullm.openwebui.reasoning_browser_smoke.v2"}\n',
                    encoding="ascii",
                )
                output.chmod(0o600)
            elif stage_call == 2:
                assert self._argument(argv, "--evidence") == str(
                    self.fixture.campaign_paths["aq4_reasoning_browser"]
                )
                report = {
                    "schema_version": (
                        "ullm.openwebui.reasoning_browser_smoke_validator.v1"
                    ),
                    "input_schema_version": (
                        "ullm.openwebui.reasoning_browser_smoke.v2"
                    ),
                    "structurally_valid": True,
                    "gate_eligible": True,
                }
                target = Path(self._argument(argv, "--output"))
                target.write_text(
                    json.dumps(
                        report,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="ascii",
                )
                target.chmod(0o444)
        elif stage == "aq4_bundle" and stage_call == 1:
            output = Path(self._argument(argv, "--output"))
            output.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "ullm.generic_reasoning_release_bundle.v1"
                        ),
                        "status": "complete",
                        "production_activation_performed": False,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n",
                encoding="ascii",
            )
            output.chmod(0o600)
        elif stage == "aq4_bundle" and stage_call == 2:
            assert argv[1:] == [
                str(self.fixture.campaign_paths["aq4_bundle"])
            ]
        if stage == self.mutate_active_stage:
            self.fixture.active.write_bytes(b'{"unexpected":true}\n')
        return subprocess.CompletedProcess(argv, 0, "", "")


class DockerLeaseRunner(Runner):
    def __init__(
        self,
        fixture: Fixture,
        *,
        container_ids: tuple[str, ...] = (),
        leak_stage: str | None = None,
        failed_rm_calls: int = 0,
        delayed_create_at: float | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(fixture, **kwargs)
        self.container_ids = list(container_ids)
        self.leak_stage = leak_stage
        self.failed_rm_calls = failed_rm_calls
        self.delayed_create_at = delayed_create_at
        self.delayed_create_done = False
        self.docker_calls: list[tuple[str, ...]] = []

    def docker_lease_sleep(self, seconds: float) -> None:
        super().docker_lease_sleep(seconds)
        if (
            self.delayed_create_at is not None
            and not self.delayed_create_done
            and self.docker_lease_clock >= self.delayed_create_at
        ):
            self.container_ids[:] = ["c" * 64]
            self.delayed_create_done = True

    def __call__(
        self,
        argv: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess:
        if (
            len(argv) >= 3
            and argv[1] == "container"
            and argv[2] in {"ls", "rm"}
        ):
            self.docker_calls.append(tuple(argv))
            if argv[2] == "ls":
                assert kwargs["timeout"] == TX.DOCKER_LEASE_CONTROL_TIMEOUT_SECONDS
                assert argv[-2] == "--filter"
                assert argv[-1].startswith(
                    f"label={TX.DOCKER_LEASE_LABEL_KEY}="
                )
                stdout = "".join(
                    f"{identifier}\n" for identifier in self.container_ids
                )
                return subprocess.CompletedProcess(argv, 0, stdout, "")
            assert argv[3] == "--force"
            if self.failed_rm_calls:
                self.failed_rm_calls -= 1
                return subprocess.CompletedProcess(argv, 19, "", "")
            requested = argv[4:]
            assert requested == self.container_ids
            self.container_ids.clear()
            return subprocess.CompletedProcess(argv, 0, "\n".join(requested), "")
        completed = super().__call__(argv, **kwargs)
        if (
            self.leak_stage is not None
            and not is_git(argv)
            and kwargs["env"]["ULLM_CAMPAIGN_TRANSACTION_STAGE"]
            == self.leak_stage
        ):
            self.container_ids[:] = ["e" * 64]
            self.leak_stage = None
        return completed


def live_aq4_proof(
    request: object,
    claim: object,
    preflight: object,
) -> dict[str, object]:
    return {
        "schema_version": TX.restoration_proof.SCHEMA_VERSION,
        "authorization_sha256": claim.authorization.snapshot.sha256,
        "claim_sha256": claim.snapshot.sha256,
        "captured_at": AUTH.utc_timestamp(NOW),
        "active_manifest": {
            "path": str(preflight.active.path),
            "expected_sha256": preflight.active.sha256,
            "observed_sha256": preflight.active.sha256,
            "bytes_equal": True,
        },
        "service": {
            "unit": request.service_unit,
            "active_state": "active",
            "sub_state": "running",
            "boot_id": "11111111-2222-3333-4444-555555555555",
            "n_restarts": 0,
        },
        "gateway": {
            "pid": 100,
            "ppid": 1,
            "starttime_ticks": 10,
            "executable_sha256": "6" * 64,
        },
        "worker": {
            "pid": 101,
            "ppid": 100,
            "starttime_ticks": 11,
            "executable_sha256": AQ4_WORKER,
        },
        "endpoints": {
            "gateway_healthz": {"status": 200},
            "gateway_readyz": {"status": 200},
            "gateway_models": {
                "status": 200,
                "model_ids": ["ullm-qwen3.5-9b-aq4"],
            },
            "openwebui_health": {"status": 200},
            "openwebui_models": {
                "status": 200,
                "model_ids": ["ullm-qwen3.5-9b-aq4"],
            },
        },
        "epoch_stable": True,
        "passed": True,
    }


def live_sq8_epoch(
    request: object,
    _claim: object,
    _preflight: object,
) -> dict[str, object]:
    return {
        "service": {
            "unit": request.service_unit,
            "active_state": "active",
            "sub_state": "running",
            "boot_id": "11111111-2222-3333-4444-555555555555",
            "n_restarts": 0,
        },
        "gateway": {
            "pid": 200,
            "ppid": 1,
            "starttime_ticks": 20,
            "executable_sha256": "6" * 64,
        },
        "worker": {
            "pid": 201,
            "ppid": 200,
            "starttime_ticks": 21,
            "executable_sha256": SQ8_WORKER,
        },
    }


class FastStabilizationTimer:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds


def stabilization_kwargs() -> dict[str, object]:
    timer = FastStabilizationTimer()
    return {
        "candidate_stabilization_probe": live_sq8_epoch,
        "stabilization_sleeper": timer.sleep,
        "stabilization_monotonic": timer.monotonic,
    }


def execute(fixture: Fixture, runner: Runner) -> object:

    return TX.execute_transaction(
        fixture.request,
        policy=fixture.policy,
        validator=fixture.validator,
        runner=runner,
        inactive_checker=lambda _services: None,
        clock=lambda: NOW,
        restoration_probe=live_aq4_proof,
        **stabilization_kwargs(),
    )


def load_outcome(fixture: Fixture) -> dict[str, object]:
    _snapshot, document = AUTH.load_outcome(
        fixture.authorization_path,
        now=NOW,
        policy=fixture.policy,
    )
    return document


def test_success_restores_exact_aq4_and_publishes_complete_outcome(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    result = execute(fixture, Runner(fixture))

    assert result.status == "succeeded_restored"
    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert fixture.backup.read_bytes() == fixture.aq4_raw
    assert stat.S_IMODE(fixture.backup.stat().st_mode) == 0o444
    outcome = load_outcome(fixture)
    assert outcome["status"] == "succeeded_restored"
    assert set(outcome["stages"].values()) == {"passed"}
    assert outcome["restoration"]["bytes_equal"] is True
    assert all(outcome["campaigns"].values())
    assert len(outcome["candidate_observations"]) == 9
    assert len(outcome["aq4_observations"]) == 6
    for name, source in (
        ("promotion_evidence", fixture.aq4_promotion_evidence),
        ("promotion_receipt", fixture.aq4_promotion_receipt),
    ):
        copied = Path(
            fixture.authorization_document["aq4_release"][name]["path"]
        )
        assert copied.read_bytes() == source.read_bytes()
        assert stat.S_IMODE(copied.stat().st_mode) == 0o444
        assert copied.stat().st_nlink == 1
        assert stat.S_IMODE(source.stat().st_mode) != 0o444


def test_docker_lease_preflight_rejects_and_cleans_stale_container(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    runner = DockerLeaseRunner(
        fixture,
        container_ids=("d" * 64,),
    )

    with pytest.raises(TX.TransactionFailed) as caught:
        execute(fixture, runner)

    assert caught.value.result.status == "failed_restore"
    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert runner.container_ids == []
    assert "candidate_reconciliation" not in runner.stage_calls
    assert load_outcome(fixture)["status"] == "failed_restore"


def test_producer_leaked_container_is_removed_before_transaction_continues(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    runner = DockerLeaseRunner(fixture, leak_stage="sq8_full")

    result = execute(fixture, runner)

    assert result.status == "succeeded_restored"
    assert runner.container_ids == []
    removals = [
        call
        for call in runner.docker_calls
        if call[1:4] == ("container", "rm", "--force")
    ]
    assert removals
    assert removals[0][4:] == ("e" * 64,)


def test_docker_cleanup_failure_forbids_restored_status_and_recovery_retries(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    failed_runner = DockerLeaseRunner(
        fixture,
        leak_stage="sq8_full",
        failed_rm_calls=1,
    )

    with pytest.raises(TX.TransactionFailed) as caught:
        execute(fixture, failed_runner)

    assert caught.value.result.status == "failed_restore"
    assert fixture.active.read_bytes() == fixture.sq8_raw
    assert failed_runner.container_ids == []
    assert load_outcome(fixture)["status"] == "failed_restore"

    recovery_runner = DockerLeaseRunner(fixture)
    recovered = RECOVERY.recover_transaction(
        recovery_request(fixture),
        policy=fixture.policy,
        validator=fixture.validator,
        runner=recovery_runner,
        clock=lambda: NOW + timedelta(hours=2),
        restoration_probe=live_aq4_proof,
    )
    assert recovered.status == "restored"
    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert recovery_runner.container_ids == []


def test_unproved_docker_cleanup_cannot_publish_failed_restored(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    runner = DockerLeaseRunner(
        fixture,
        leak_stage="sq8_full",
        failed_rm_calls=100,
    )

    with pytest.raises(TX.TransactionFailed) as caught:
        execute(fixture, runner)

    assert caught.value.result.status == "failed_restore"
    assert runner.container_ids == ["e" * 64]
    outcome = load_outcome(fixture)
    assert outcome["status"] == "failed_restore"
    assert outcome["restoration"]["bytes_equal"] is False


def test_docker_cleanup_is_one_bounded_batch_for_maximum_inventory(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    claim = AUTH.claim_authorization(
        fixture.authorization_path,
        now=NOW,
        policy=fixture.policy,
    )
    identifiers = tuple(f"{index:064x}" for index in range(256))
    runner = DockerLeaseRunner(fixture, container_ids=identifiers)

    TX._cleanup_docker_lease(
        fixture.request,
        claim,
        runner=runner,
        stage="bounded_cleanup_test",
    )

    removals = [
        call
        for call in runner.docker_calls
        if call[1:4] == ("container", "rm", "--force")
    ]
    assert len(removals) == 1
    assert len(runner.docker_calls) <= 16
    assert removals[0][1:4] == (
        "container",
        "rm",
        "--force",
    )
    assert removals[0][4:] == identifiers
    assert (
        runner.docker_lease_clock
        >= TX.DOCKER_LEASE_QUIESCENCE_SECONDS
    )
    assert runner.container_ids == []


def test_docker_cleanup_catches_daemon_delayed_create_during_quiescence(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    claim = AUTH.claim_authorization(
        fixture.authorization_path,
        now=NOW,
        policy=fixture.policy,
    )
    runner = DockerLeaseRunner(
        fixture,
        delayed_create_at=1.0,
    )

    TX._cleanup_docker_lease(
        fixture.request,
        claim,
        runner=runner,
        stage="delayed_create_cleanup_test",
    )

    removals = [
        call
        for call in runner.docker_calls
        if call[1:4] == ("container", "rm", "--force")
    ]
    assert len(removals) == 1
    assert removals[0][4:] == ("c" * 64,)
    assert runner.container_ids == []
    assert runner.docker_lease_clock >= (
        1.0 + TX.DOCKER_LEASE_QUIESCENCE_SECONDS
    )


def test_recovery_cleans_stale_lease_before_replacing_active_manifest(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    AUTH.claim_authorization(
        fixture.authorization_path,
        now=NOW,
        policy=fixture.policy,
    )
    fixture.backup.write_bytes(fixture.aq4_raw)
    fixture.backup.chmod(0o444)
    fixture.active.write_bytes(fixture.sq8_raw)
    fixture.active.chmod(0o644)
    runner = DockerLeaseRunner(
        fixture,
        container_ids=("f" * 64,),
    )

    result = RECOVERY.recover_transaction(
        recovery_request(fixture),
        policy=fixture.policy,
        validator=fixture.validator,
        runner=runner,
        clock=lambda: NOW + timedelta(hours=2),
        restoration_probe=live_aq4_proof,
    )

    assert result.status == "restored"
    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert runner.container_ids == []
    assert runner.docker_calls[1][1:4] == (
        "container",
        "rm",
        "--force",
    )


def test_fresh_aq4_campaigns_do_not_reopen_displaced_candidate(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)

    class CandidateLossAfterRestoreRunner(Runner):
        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            result = super().__call__(argv, **kwargs)
            environment = kwargs["env"]
            assert isinstance(environment, dict)
            stage = environment.get("ULLM_CAMPAIGN_TRANSACTION_STAGE")
            if (
                stage == "reverse_reconciliation"
                and self.stage_call_counts[stage]
                == len(fixture.request.commands.reverse_reconciliation)
            ):
                fixture.candidate.unlink()
            return result

    result = execute(fixture, CandidateLossAfterRestoreRunner(fixture))

    assert result.status == "succeeded_restored"
    assert not fixture.candidate.exists()
    outcome = load_outcome(fixture)
    assert all(
        outcome["stages"][name] == "passed"
        for name in (
            "aq4_reasoning_release",
            "aq4_reasoning_browser",
            "aq4_bundle",
        )
    )


def test_aq4_staging_rewrite_changes_only_exact_authorized_output_argument(
    tmp_path: Path,
) -> None:
    authorized = tmp_path / "authorized.json"
    staged = tmp_path / ".ullm-aq4-campaign-stage-fixture.json"
    command = (
        "legacy-producer",
        "--input",
        "immutable-input.json",
        "--output",
        str(authorized),
        "--status",
        "complete",
    )

    rewritten = TX._rewrite_authorized_output_argument(
        command,
        flag="--output",
        authorized_path=authorized,
        staging_path=staged,
    )

    assert rewritten == (
        "legacy-producer",
        "--input",
        "immutable-input.json",
        "--output",
        str(staged),
        "--status",
        "complete",
    )
    assert command[0:4] == rewritten[0:4]
    assert command[5:] == rewritten[5:]
    for invalid in (
        command + ("--output", str(authorized)),
        tuple(
            "different.json" if value == str(authorized) else value
            for value in command
        ),
        tuple(value for value in command if value != "--output"),
    ):
        with pytest.raises(TX.TransactionError, match="fixed output"):
            TX._rewrite_authorized_output_argument(
                invalid,
                flag="--output",
                authorized_path=authorized,
                staging_path=staged,
            )


@pytest.mark.parametrize(
    "failed_stage",
    (
        "candidate_reconciliation",
        "candidate_checks",
        "sq8_full",
        "reasoning_release",
        "reasoning_browser",
        "aq4_reasoning_release",
        "aq4_reasoning_browser",
        "aq4_bundle",
    ),
)
def test_candidate_window_failure_still_restores_and_reconciles_aq4(
    tmp_path: Path,
    failed_stage: str,
) -> None:
    fixture = Fixture(tmp_path)
    runner = Runner(fixture, fail_stage=failed_stage)
    with pytest.raises(TX.TransactionError, match="failed_restored"):
        execute(fixture, runner)

    outcome = load_outcome(fixture)
    assert outcome["status"] == "failed_restored"
    assert outcome["stages"][failed_stage] == "failed"
    assert outcome["stages"]["aq4_restore"] == "passed"
    assert outcome["stages"]["reverse_reconciliation"] == "passed"
    assert outcome["stages"]["final_checks"] == "passed"
    assert fixture.active.read_bytes() == fixture.aq4_raw
    if failed_stage in {
        "candidate_reconciliation",
        "candidate_checks",
        "sq8_full",
        "reasoning_release",
        "reasoning_browser",
    }:
        assert not any(
            stage.startswith("aq4_") for stage in runner.stage_calls
        )


@pytest.mark.parametrize(
    "failed_stage",
    ("reverse_reconciliation", "final_checks"),
)
def test_restore_verification_failure_is_not_misreported_as_restored(
    tmp_path: Path,
    failed_stage: str,
) -> None:
    fixture = Fixture(tmp_path)
    with pytest.raises(TX.TransactionError, match="failed_restore"):
        execute(fixture, Runner(fixture, fail_stage=failed_stage))

    outcome = load_outcome(fixture)
    assert outcome["status"] == "failed_restore"
    assert outcome["stages"][failed_stage] == "failed"
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_interruption_is_caught_across_campaign_and_restores_before_return(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    with pytest.raises(TX.TransactionError, match="failed_restored"):
        execute(
            fixture,
            Runner(fixture, interrupt_stage="reasoning_release"),
        )
    assert fixture.active.read_bytes() == fixture.aq4_raw
    outcome = load_outcome(fixture)
    assert outcome["failure_stage"] == "reasoning_release"
    assert outcome["status"] == "failed_restored"


def test_claim_is_consumed_and_cannot_be_replayed(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    execute(fixture, Runner(fixture))

    with pytest.raises(TX.TransactionError, match="claim failed"):
        execute(fixture, Runner(fixture))
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_expired_authorization_never_creates_claim_or_changes_active(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    with pytest.raises(TX.TransactionError, match="claim failed"):
        TX.execute_transaction(
            fixture.request,
            policy=fixture.policy,
            validator=fixture.validator,
            runner=Runner(fixture),
            inactive_checker=lambda _services: None,
            clock=lambda: NOW + timedelta(hours=2),
            restoration_probe=live_aq4_proof,
            **stabilization_kwargs(),
        )
    assert not list(fixture.claims.iterdir())
    assert not list(fixture.outcomes.iterdir())
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_read_only_preflight_does_not_claim_or_write_backup(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    report = TX.preflight(
        fixture.request,
        now=NOW,
        policy=fixture.policy,
        validator=fixture.validator,
        runner=Runner(fixture),
    )
    assert report.active.sha256 == digest(fixture.aq4_raw)
    assert report.candidate.sha256 == digest(fixture.sq8_raw)
    assert report.runtime_artifact_seals
    assert report.runtime_tree_seals
    labels = {sealed.label for sealed in report.runtime_artifact_seals}
    shared_paths = {
        sealed.snapshot.path
        for sealed in report.shared_runtime_artifact_seals
    }
    assert "candidate SQ8 worker binary" in labels
    assert "active AQ4 worker binary" in labels
    assert "frozen candidate served-model manifest" in labels
    assert fixture.docker_wrapper in shared_paths
    assert Path(TX.PYTHON_BINARY) in shared_paths
    assert Path(TX.DOCKER_BINARY) in shared_paths
    assert not list(fixture.claims.iterdir())
    assert not list(fixture.outcomes.iterdir())
    assert not fixture.backup.exists()


def test_docker_wrapper_command_binding_rejects_route_local_bypasses(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    wrapper = str(fixture.docker_wrapper)
    sq8_full = fixture.commands.sq8_full
    docker_index = sq8_full.index("--docker")
    wrong_value = list(sq8_full)
    wrong_value[docker_index + 1] = "/tmp/not-the-wrapper"
    known_producer = (
        TX.PYTHON_BINARY,
        "-I",
        "-S",
        "-B",
        str(
            fixture.source
            / "tools/run-sq8-full-openwebui-campaign.py"
        ),
        "--execute",
    )
    cases = (
        replace(
            fixture.commands,
            sq8_full=(*sq8_full, "--docker=/usr/bin/docker"),
        ),
        replace(
            fixture.commands,
            sq8_full=(*sq8_full, "--docker", wrapper),
        ),
        replace(fixture.commands, sq8_full=tuple(wrong_value)),
        replace(
            fixture.commands,
            reasoning_release=(
                *fixture.commands.reasoning_release,
                "/tmp/docker",
            ),
        ),
        replace(
            fixture.commands,
            candidate_reconciliation=(("/tmp/docker", "run", "image"),),
        ),
        replace(fixture.commands, sq8_full=known_producer),
        replace(
            fixture.commands,
            candidate_reconciliation=((wrapper, "run", "--rm", "image"),),
        ),
        replace(
            fixture.commands,
            candidate_reconciliation=(
                (
                    TX.PYTHON_BINARY,
                    "-S",
                    "-I",
                    "-B",
                    wrapper,
                    "run",
                    "--rm",
                    "image",
                ),
            ),
        ),
    )
    for commands in cases:
        with pytest.raises(TX.TransactionError, match="Docker"):
            TX._require_docker_lease_wrapper_commands(
                commands,
                fixture.source,
            )

    valid_direct = replace(
        fixture.commands,
        candidate_reconciliation=(
            (
                TX.PYTHON_BINARY,
                "-I",
                "-S",
                "-B",
                wrapper,
                "run",
                "--rm",
                "image",
            ),
        ),
    )
    TX._require_docker_lease_wrapper_commands(
        valid_direct,
        fixture.source,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "candidate_group_writable",
        "candidate_worker_hardlink",
        "product_payload_group_writable",
        "tokenizer_symlink",
        "artifact_parent_group_writable",
        "aq4_evidence_group_writable",
        "candidate_command_group_writable",
    ),
)
def test_preflight_rejects_unsealed_runtime_artifacts(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = Fixture(tmp_path)
    if mutation == "candidate_group_writable":
        fixture.candidate.chmod(0o664)
    elif mutation == "candidate_worker_hardlink":
        os.link(
            fixture.sq8_worker,
            fixture.slot / "ullm-sq8-worker-hardlink",
        )
    elif mutation == "product_payload_group_writable":
        (
            fixture.sq8_product / "artifact" / "weight.bin"
        ).chmod(0o664)
    elif mutation == "tokenizer_symlink":
        tokenizer = fixture.sq8_tokenizer / "tokenizer.json"
        held = fixture.slot / "held-tokenizer.json"
        tokenizer.rename(held)
        tokenizer.symlink_to(held)
    elif mutation == "artifact_parent_group_writable":
        fixture.slot.chmod(0o770)
    elif mutation == "candidate_command_group_writable":
        fixture.candidate_command.chmod(0o575)
    else:
        fixture.aq4_promotion_evidence.chmod(0o664)

    with pytest.raises(TX.TransactionError, match="runtime"):
        TX.preflight(
            fixture.request,
            now=NOW,
            policy=fixture.policy,
            validator=fixture.validator,
            runner=Runner(fixture),
        )


def test_runtime_artifact_seal_rejects_non_executor_owned_candidate(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "preserved-candidate-worker"
    candidate.write_bytes(b"candidate\n")
    candidate.chmod(0o555)
    with pytest.raises(
        TX.campaign_runtime_seal.RuntimeArtifactSealError,
        match="owner",
    ):
        TX.campaign_runtime_seal.capture_runtime_artifact_seal(
            candidate,
            label="preserved candidate worker",
            maximum=1024,
            required_uid=os.geteuid() + 1,
        )


def test_runtime_artifact_seal_rejects_posix_acl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "candidate-worker"
    target.write_bytes(b"candidate\n")
    target.chmod(0o555)
    target_inode = target.stat().st_ino
    real_getxattr = os.getxattr

    def injected_acl(
        selected: object,
        attribute: str,
        *args: object,
        **kwargs: object,
    ) -> bytes:
        if (
            isinstance(selected, int)
            and os.fstat(selected).st_ino == target_inode
            and attribute == "system.posix_acl_access"
        ):
            return b"fixture-acl"
        return real_getxattr(selected, attribute, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        TX.campaign_runtime_seal.os,
        "getxattr",
        injected_acl,
    )
    with pytest.raises(
        TX.campaign_runtime_seal.RuntimeArtifactSealError,
        match="POSIX ACL",
    ):
        TX.campaign_runtime_seal.capture_runtime_artifact_seal(
            target,
            label="candidate worker",
            maximum=1024,
            required_uid=os.geteuid(),
        )


def test_runtime_artifact_seal_rejects_file_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "candidate-worker"
    target.write_bytes(b"candidate\n")
    target.chmod(0o555)
    target_inode = target.stat().st_ino
    real_getxattr = os.getxattr

    def injected_capability(
        selected: object,
        attribute: str,
        *args: object,
        **kwargs: object,
    ) -> bytes:
        if (
            isinstance(selected, int)
            and os.fstat(selected).st_ino == target_inode
            and attribute == "security.capability"
        ):
            return b"fixture-file-capability"
        return real_getxattr(selected, attribute, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        TX.campaign_runtime_seal.os,
        "getxattr",
        injected_capability,
    )
    with pytest.raises(
        TX.campaign_runtime_seal.RuntimeArtifactSealError,
        match="file capability",
    ):
        TX.campaign_runtime_seal.capture_runtime_artifact_seal(
            target,
            label="candidate worker",
            maximum=1024,
            required_uid=os.geteuid(),
        )


def test_preflight_rejects_command_executable_file_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    target_inode = fixture.candidate_command.stat().st_ino
    real_getxattr = os.getxattr

    def injected_capability(
        selected: object,
        attribute: str,
        *args: object,
        **kwargs: object,
    ) -> bytes:
        if (
            isinstance(selected, int)
            and os.fstat(selected).st_ino == target_inode
            and attribute == "security.capability"
        ):
            return b"fixture-file-capability"
        return real_getxattr(selected, attribute, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        TX.campaign_runtime_seal.os,
        "getxattr",
        injected_capability,
    )
    with pytest.raises(TX.TransactionError, match="command executable"):
        TX.preflight(
            fixture.request,
            now=NOW,
            policy=fixture.policy,
            validator=fixture.validator,
            runner=Runner(fixture),
        )


@pytest.mark.parametrize(
    ("path", "uid", "gid", "mode"),
    (
        (TX.FIXED_GATEWAY_API_KEY_PATH, 0, 1000, 0o640),
        (
            TX.FIXED_OPENWEBUI_SESSION_TOKEN_PATH,
            0,
            1000,
            0o640,
        ),
    ),
)
def test_transaction_fixed_secret_metadata_contract_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    uid: int,
    gid: int,
    mode: int,
) -> None:
    fixture = Fixture(tmp_path)
    template = TX._read_input(
        fixture.unit,
        "fixture template",
        TX.MAX_INPUT_BYTES,
    )
    raw = b"fixture-private-secret\n"
    if path == TX.FIXED_OPENWEBUI_SESSION_TOKEN_PATH:
        monkeypatch.setattr(
            TX,
            "_validate_fixed_session_token_parent",
            lambda _path: None,
        )

    def snapshot_with(
        *,
        selected_uid: int,
        selected_gid: int,
        selected_mode: int,
    ) -> object:
        return replace(
            template,
            path=path,
            raw=raw,
            sha256=digest(raw),
            identity=replace(
                template.identity,
                uid=selected_uid,
                gid=selected_gid,
                mode=stat.S_IFREG | selected_mode,
                size=len(raw),
            ),
        )

    monkeypatch.setattr(
        TX,
        "_read_input",
        lambda *_args, **_kwargs: snapshot_with(
            selected_uid=uid,
            selected_gid=gid,
            selected_mode=mode,
        ),
    )
    assert TX._validate_private_secret(
        path,
        "fixture secret",
        required_uid=0,
    ) == digest(raw)

    for wrong in (
        snapshot_with(
            selected_uid=uid + 1,
            selected_gid=gid,
            selected_mode=mode,
        ),
        snapshot_with(
            selected_uid=uid,
            selected_gid=gid + 1,
            selected_mode=mode,
        ),
        snapshot_with(
            selected_uid=uid,
            selected_gid=gid,
            selected_mode=0o600 if mode == 0o640 else 0o640,
        ),
    ):
        monkeypatch.setattr(
            TX,
            "_read_input",
            lambda *_args, _wrong=wrong, **_kwargs: _wrong,
        )
        with pytest.raises(TX.TransactionError, match="metadata is unsafe"):
            TX._validate_private_secret(
                path,
                "fixture secret",
                required_uid=0,
            )


def test_session_token_parent_and_file_reject_uid1000_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template_path = tmp_path / "template"
    template_path.write_bytes(b"template\n")
    template = TX._read_input(
        template_path,
        "fixture",
        TX.MAX_INPUT_BYTES,
    )
    parent = tmp_path / "parent"
    parent.mkdir(mode=0o750)
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    real_fstat = TX.os.fstat

    def parent_metadata(uid: int) -> os.stat_result:
        return os.stat_result(
            (
                stat.S_IFDIR | 0o750,
                1,
                1,
                2,
                uid,
                1000,
                0,
                0,
                0,
                0,
            )
        )

    monkeypatch.setattr(
        TX,
        "_open_parent_descriptor",
        lambda *_args, **_kwargs: (descriptor, (1, 1)),
    )
    monkeypatch.setattr(
        TX.os,
        "fstat",
        lambda selected: (
            parent_metadata(1000)
            if selected == descriptor
            else real_fstat(selected)
        ),
    )
    monkeypatch.setattr(TX, "_require_service_entry_xattrs", lambda _fd: None)
    with pytest.raises(TX.TransactionError, match="parent metadata is unsafe"):
        TX._validate_fixed_session_token_parent(
            TX.FIXED_OPENWEBUI_SESSION_TOKEN_PATH
        )

    raw = b"uid1000-replacement\n"
    replaced = replace(
        template,
        path=TX.FIXED_OPENWEBUI_SESSION_TOKEN_PATH,
        raw=raw,
        sha256=digest(raw),
        identity=replace(
            template.identity,
            uid=1000,
            gid=1000,
            mode=stat.S_IFREG | 0o640,
            size=len(raw),
        ),
    )
    monkeypatch.setattr(
        TX,
        "_validate_fixed_session_token_parent",
        lambda _path: None,
    )
    monkeypatch.setattr(
        TX,
        "_read_input",
        lambda *_args, **_kwargs: replaced,
    )
    with pytest.raises(TX.TransactionError, match="metadata is unsafe"):
        TX._validate_private_secret(
            TX.FIXED_OPENWEBUI_SESSION_TOKEN_PATH,
            "OpenWebUI session token",
            required_uid=0,
        )


@pytest.mark.parametrize("mutation", ("rename", "metadata"))
def test_runtime_artifact_repin_rejects_leaf_change(
    tmp_path: Path,
    mutation: str,
) -> None:
    target = tmp_path / "candidate-worker"
    target.write_bytes(b"candidate\n")
    target.chmod(0o555)
    sealed = TX.campaign_runtime_seal.capture_runtime_artifact_seal(
        target,
        label="candidate worker",
        maximum=1024,
        required_uid=os.geteuid(),
    )
    if mutation == "rename":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(target.read_bytes())
        replacement.chmod(0o555)
        os.replace(replacement, target)
    else:
        target.chmod(0o500)
    with pytest.raises(
        TX.campaign_runtime_seal.RuntimeArtifactSealError,
        match="changed",
    ):
        TX.campaign_runtime_seal.require_runtime_artifact_seal(
            sealed,
            required_uid=os.geteuid(),
        )


def test_runtime_tree_repin_rejects_member_change(tmp_path: Path) -> None:
    tree = tmp_path / "product"
    tree.mkdir()
    tree.chmod(0o755)
    payload = tree / "weight.bin"
    payload.write_bytes(b"weights\n")
    payload.chmod(0o644)
    sealed = TX.campaign_runtime_seal.capture_runtime_tree_seal(
        tree,
        label="candidate product",
        required_uid=os.geteuid(),
    )
    added = tree / "added.bin"
    added.write_bytes(b"added\n")
    added.chmod(0o644)
    with pytest.raises(
        TX.campaign_runtime_seal.RuntimeArtifactSealError,
        match="changed",
    ):
        TX.campaign_runtime_seal.require_runtime_tree_seal(
            sealed,
            required_uid=os.geteuid(),
        )


def test_runtime_tree_seal_rejects_member_file_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = tmp_path / "product"
    tree.mkdir()
    tree.chmod(0o755)
    payload = tree / "weight.bin"
    payload.write_bytes(b"weights\n")
    payload.chmod(0o644)
    real_getxattr = os.getxattr

    def injected_capability(
        selected: object,
        attribute: str,
        *args: object,
        **kwargs: object,
    ) -> bytes:
        if (
            isinstance(selected, (str, bytes, os.PathLike))
            and Path(selected) == payload
            and attribute == "security.capability"
        ):
            return b"fixture-file-capability"
        return real_getxattr(selected, attribute, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        TX.campaign_runtime_seal.os,
        "getxattr",
        injected_capability,
    )
    with pytest.raises(
        TX.campaign_runtime_seal.RuntimeArtifactSealError,
        match="file capability",
    ):
        TX.campaign_runtime_seal.capture_runtime_tree_seal(
            tree,
            label="candidate product",
            required_uid=os.geteuid(),
        )


def test_worker_replacement_during_validator_is_rejected_by_preflight(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    replaced = False

    def replacing_validator(path: Path) -> dict[str, object]:
        nonlocal replaced
        result = fixture.validator(path)
        if path == fixture.candidate and not replaced:
            replaced = True
            replacement = fixture.slot / "worker-replacement"
            replacement.write_bytes(fixture.sq8_worker.read_bytes())
            replacement.chmod(0o555)
            os.replace(replacement, fixture.sq8_worker)
        return result

    with pytest.raises(TX.TransactionError, match="runtime artifact seal changed"):
        TX.preflight(
            fixture.request,
            now=NOW,
            policy=fixture.policy,
            validator=replacing_validator,
            runner=Runner(fixture),
        )


@pytest.mark.parametrize(
    ("declared_file", "expected_label"),
    (
        ("tokenizer", "tokenizer file tokenizer.json"),
        ("package", "package manifest"),
        ("artifact", "artifact manifest"),
    ),
)
def test_runtime_seal_binds_declared_manifest_hashes_immediately(
    tmp_path: Path,
    declared_file: str,
    expected_label: str,
) -> None:
    fixture = Fixture(tmp_path)
    document = json.loads(fixture.sq8_raw)
    if declared_file == "tokenizer":
        document["tokenizer"]["files"]["tokenizer.json"] = "0" * 64
    else:
        document["product"][declared_file]["manifest_sha256"] = "0" * 64

    with pytest.raises(
        TX.TransactionError,
        match=rf"{expected_label} runtime bytes differ",
    ):
        TX._manifest_runtime_seals(
            document,
            manifest_path=fixture.candidate,
            expected_receipt_path=fixture.receipt,
            label="candidate SQ8",
            required_uid=os.geteuid(),
        )


@pytest.mark.parametrize("declared_file", ("tokenizer", "package"))
def test_declared_file_replacement_during_validator_is_rejected(
    tmp_path: Path,
    declared_file: str,
) -> None:
    fixture = Fixture(tmp_path)
    replaced = False
    target = (
        fixture.sq8_tokenizer / "tokenizer.json"
        if declared_file == "tokenizer"
        else fixture.sq8_product / "package" / "manifest.json"
    )

    def replacing_validator(path: Path) -> dict[str, object]:
        nonlocal replaced
        result = fixture.validator(path)
        if path == fixture.candidate and not replaced:
            replaced = True
            replacement = fixture.slot / f"{declared_file}-replacement"
            replacement.write_bytes(target.read_bytes())
            replacement.chmod(0o644)
            os.replace(replacement, target)
        return result

    with pytest.raises(TX.TransactionError, match="runtime artifact seal changed"):
        TX.preflight(
            fixture.request,
            now=NOW,
            policy=fixture.policy,
            validator=replacing_validator,
            runner=Runner(fixture),
        )


def test_worker_replacement_is_rejected_before_next_command_exec(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.commands = replace(
        fixture.commands,
        candidate_reconciliation=(
            (str(fixture.candidate_command), "one"),
            (str(fixture.candidate_command), "two"),
        ),
    )
    fixture.request = replace(fixture.request, commands=fixture.commands)

    class ReplacingWorkerRunner(Runner):
        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            result = super().__call__(argv, **kwargs)
            if (
                not is_git(argv)
                and kwargs["env"]["ULLM_CAMPAIGN_TRANSACTION_STAGE"]
                == "candidate_reconciliation"
                and self.stage_call_counts["candidate_reconciliation"] == 1
            ):
                replacement = self.fixture.slot / "worker-replacement"
                replacement.write_bytes(self.fixture.sq8_worker.read_bytes())
                replacement.chmod(0o555)
                os.replace(replacement, self.fixture.sq8_worker)
            return result

    runner = ReplacingWorkerRunner(fixture)
    with pytest.raises(TX.TransactionFailed, match="failed_restored"):
        execute(fixture, runner)
    assert runner.stage_call_counts["candidate_reconciliation"] == 1
    outcome = load_outcome(fixture)
    assert outcome["stages"]["candidate_reconciliation"] == "failed"
    assert outcome["stages"]["reverse_reconciliation"] == "passed"
    assert outcome["stages"]["final_checks"] == "passed"
    assert outcome["failure_stage"] == "candidate_reconciliation"
    assert outcome["status"] == "failed_restored"
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_candidate_executable_replacement_fails_campaign_but_restores_aq4(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.commands = replace(
        fixture.commands,
        candidate_reconciliation=(
            (str(fixture.candidate_command), "one"),
            (str(fixture.candidate_command), "two"),
        ),
    )
    fixture.request = replace(fixture.request, commands=fixture.commands)

    class ReplacingExecutableRunner(Runner):
        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            result = super().__call__(argv, **kwargs)
            if (
                not is_git(argv)
                and kwargs["env"]["ULLM_CAMPAIGN_TRANSACTION_STAGE"]
                == "candidate_reconciliation"
                and self.stage_call_counts["candidate_reconciliation"] == 1
            ):
                replacement = (
                    self.fixture.command_root / "candidate-command-replacement"
                )
                replacement.write_bytes(
                    self.fixture.candidate_command.read_bytes()
                )
                replacement.chmod(0o555)
                os.replace(replacement, self.fixture.candidate_command)
            return result

    runner = ReplacingExecutableRunner(fixture)
    with pytest.raises(TX.TransactionFailed, match="failed_restored"):
        execute(fixture, runner)

    assert runner.stage_call_counts["candidate_reconciliation"] == 1
    outcome = load_outcome(fixture)
    assert outcome["failure_stage"] == "candidate_reconciliation"
    assert outcome["stages"]["reverse_reconciliation"] == "passed"
    assert outcome["stages"]["final_checks"] == "passed"
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_command_path_swap_after_descriptor_pin_never_executes_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    executable = fixture.command_root / "pinned-python"
    shutil.copyfile("/usr/bin/python3.12", executable)
    executable.chmod(0o555)
    marker = fixture.slot / "descriptor-pin-marker"
    code = (
        "from pathlib import Path;"
        "Path(__import__('sys').argv[1]).write_text('sealed')"
    )
    commands = replace(
        fixture.commands,
        candidate_checks=((str(executable), "-c", code, str(marker)),),
    )
    request = replace(fixture.request, commands=commands)
    claim = AUTH.claim_authorization(
        fixture.authorization_path,
        now=NOW,
        policy=fixture.policy,
    )
    pinned = TX.preflight(
        request,
        now=NOW,
        policy=fixture.policy,
        validator=fixture.validator,
        runner=Runner(fixture),
        claimed=claim,
    )
    malicious = fixture.command_root / "malicious-replacement"
    malicious.write_bytes(
        (
            "#!/bin/sh\n"
            f"printf malicious > {marker}\n"
            "exit 0\n"
        ).encode("ascii")
    )
    malicious.chmod(0o555)
    real_open = TX.campaign_runtime_seal.open_runtime_artifact_seal
    swapped = False

    def swap_after_open(
        sealed: object,
        *,
        required_uid: int,
    ) -> int:
        nonlocal swapped
        descriptor = real_open(sealed, required_uid=required_uid)
        if sealed.snapshot.path == executable and not swapped:
            swapped = True
            os.replace(malicious, executable)
        return descriptor

    monkeypatch.setattr(
        TX.campaign_runtime_seal,
        "open_runtime_artifact_seal",
        swap_after_open,
    )
    cleanup_stages: list[str] = []

    def fake_docker_cleanup(
        _request: object,
        _claim: object,
        *,
        runner: object,
        stage: str,
    ) -> None:
        assert runner is subprocess.run
        cleanup_stages.append(stage)

    # This test needs the real subprocess path only to exercise execution from
    # the pinned descriptor. Docker lease cleanup has its own fake-runner
    # coverage and must never contact the host daemon from this unit test.
    monkeypatch.setattr(TX, "_cleanup_docker_lease", fake_docker_cleanup)
    TX._run_commands(
        commands.candidate_checks,
        request=request,
        claim=claim,
        preflight_result=pinned,
        stage="candidate_checks",
        runner=subprocess.run,
    )

    assert swapped is True
    assert marker.read_text(encoding="ascii") == "sealed"
    assert cleanup_stages == ["candidate_checks:docker_lease_cleanup"]


def test_runtime_artifacts_are_repinned_after_restoration_probe(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)

    def mutating_probe(
        request: object,
        claim: object,
        preflight: object,
    ) -> dict[str, object]:
        proof = live_aq4_proof(request, claim, preflight)
        replacement = fixture.slot / "worker-replacement"
        replacement.write_bytes(fixture.aq4_worker.read_bytes())
        replacement.chmod(0o555)
        os.replace(replacement, fixture.aq4_worker)
        return proof

    with pytest.raises(TX.TransactionFailed, match="failed_restore"):
        TX.execute_transaction(
            fixture.request,
            policy=fixture.policy,
            validator=fixture.validator,
            runner=Runner(fixture),
            inactive_checker=lambda _services: None,
            clock=lambda: NOW,
            restoration_probe=mutating_probe,
            **stabilization_kwargs(),
        )
    assert load_outcome(fixture)["failure_stage"] == "final_checks"
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_browser_campaigns_are_bracketed_by_source_bound_image_checks(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)

    class VerifierBindingRunner(Runner):
        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            stage = (
                str(kwargs["env"]["ULLM_CAMPAIGN_TRANSACTION_STAGE"])
                if not is_git(argv)
                else ""
            )
            if stage in {
                "reasoning_browser_openwebui_image_before",
                "reasoning_browser_openwebui_image_after",
                "aq4_reasoning_browser_openwebui_image_before",
                "aq4_reasoning_browser_openwebui_image_after",
            }:
                assert argv == [
                    TX.PYTHON_BINARY,
                    "-I",
                    "-S",
                    "-B",
                    str(
                        fixture.source
                        / "tools"
                        / "verify-openwebui-container-image.py"
                    ),
                    "--docker",
                    str(fixture.docker_wrapper),
                ]
            return super().__call__(argv, **kwargs)

    runner = VerifierBindingRunner(fixture)
    assert execute(fixture, runner).status == "succeeded_restored"
    for browser_stage, prefix in (
        ("reasoning_browser", "reasoning_browser"),
        ("aq4_reasoning_browser", "aq4_reasoning_browser"),
    ):
        before = runner.stage_calls.index(
            f"{prefix}_openwebui_image_before"
        )
        browser = runner.stage_calls.index(browser_stage)
        after = runner.stage_calls.index(
            f"{prefix}_openwebui_image_after"
        )
        assert before < browser < after


@pytest.mark.parametrize(
    ("failed_check", "failure_stage"),
    (
        (
            "reasoning_browser_openwebui_image_after",
            "reasoning_browser",
        ),
        (
            "aq4_reasoning_browser_openwebui_image_after",
            "aq4_reasoning_browser",
        ),
    ),
)
def test_browser_image_change_after_execution_fails_closed(
    tmp_path: Path,
    failed_check: str,
    failure_stage: str,
) -> None:
    fixture = Fixture(tmp_path)
    with pytest.raises(TX.TransactionFailed, match="failed_restored"):
        execute(fixture, Runner(fixture, fail_stage=failed_check))
    assert load_outcome(fixture)["failure_stage"] == failure_stage
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_lock_failure_is_recorded_after_claim_without_touching_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    monkeypatch.setattr(
        TX.ActiveSlot,
        "acquire",
        classmethod(
            lambda _cls, _active, **_kwargs: (_ for _ in ()).throw(
                TX.TransactionError("busy")
            )
        ),
    )
    with pytest.raises(TX.TransactionError, match="failed_restore"):
        execute(fixture, Runner(fixture))

    assert fixture.active.read_bytes() == fixture.aq4_raw
    outcome = load_outcome(fixture)
    assert outcome["failure_stage"] == "lock"
    assert outcome["stages"]["lock"] == "failed"
    assert outcome["status"] == "failed_restore"


def test_inactive_preflight_failure_still_proves_aq4_identity(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    with pytest.raises(TX.TransactionError, match="failed_restored"):
        TX.execute_transaction(
            fixture.request,
            policy=fixture.policy,
            validator=fixture.validator,
            runner=Runner(fixture),
            inactive_checker=lambda _services: (_ for _ in ()).throw(
                TX.TransactionError("service active")
            ),
            clock=lambda: NOW,
            restoration_probe=live_aq4_proof,
            **stabilization_kwargs(),
        )
    outcome = load_outcome(fixture)
    assert outcome["failure_stage"] == "preflight"
    assert outcome["status"] == "failed_restored"
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_backup_publication_failure_is_restored_and_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    monkeypatch.setattr(
        TX,
        "_exclusive_publish",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            TX.TransactionError("backup failure")
        ),
    )
    with pytest.raises(TX.TransactionError, match="failed_restored"):
        execute(fixture, Runner(fixture))
    outcome = load_outcome(fixture)
    assert outcome["failure_stage"] == "backup"
    assert outcome["status"] == "failed_restored"
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_candidate_replace_failure_runs_exact_restore_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    real_replace = TX.ActiveSlot.replace
    calls = 0

    def fail_once(
        slot: object,
        raw: bytes,
        identity: object,
        **kwargs: object,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TX.TransactionError("candidate replace failure")
        real_replace(slot, raw, identity, **kwargs)

    monkeypatch.setattr(TX.ActiveSlot, "replace", fail_once)
    with pytest.raises(TX.TransactionError, match="failed_restored"):
        execute(fixture, Runner(fixture))
    outcome = load_outcome(fixture)
    assert outcome["failure_stage"] == "candidate_activation"
    assert outcome["status"] == "failed_restored"
    assert calls == 2
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_active_byte_mutation_during_campaign_fails_and_restores(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    with pytest.raises(TX.TransactionError, match="failed_restored"):
        execute(
            fixture,
            Runner(fixture, mutate_active_stage="reasoning_release"),
        )
    outcome = load_outcome(fixture)
    assert outcome["failure_stage"] == "reasoning_release"
    assert outcome["status"] == "failed_restored"
    assert outcome["restoration"]["displaced_manifest_sha256"] == digest(
        b'{"unexpected":true}\n'
    )
    assert fixture.active.read_bytes() == fixture.aq4_raw


@pytest.mark.parametrize(
    "stage",
    ("aq4_reasoning_release", "aq4_reasoning_browser", "aq4_bundle"),
)
def test_aq4_stage_active_byte_mutation_fails_without_sq8_reentry(
    tmp_path: Path,
    stage: str,
) -> None:
    fixture = Fixture(tmp_path)
    with pytest.raises(TX.TransactionFailed, match="failed_restored"):
        execute(
            fixture,
            Runner(fixture, mutate_active_stage=stage),
        )
    outcome = load_outcome(fixture)
    assert outcome["failure_stage"] == stage
    assert outcome["status"] == "failed_restored"
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_repeated_reverse_reconciliation_mutation_still_leaves_exact_aq4(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    with pytest.raises(TX.TransactionFailed, match="failed_restore"):
        execute(
            fixture,
            Runner(
                fixture,
                mutate_active_stage="reverse_reconciliation",
            ),
        )
    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert load_outcome(fixture)["status"] == "failed_restore"


def test_preflight_rejects_non_detached_aq4_source(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)

    class AttachedAq4Runner(Runner):
        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            if (
                is_git(argv)
                and Path(str(kwargs["cwd"])) == self.fixture.aq4_source
                and git_arguments(argv)
                == ("rev-parse", "--abbrev-ref", "HEAD")
            ):
                return subprocess.CompletedProcess(argv, 0, "release\n", "")
            return super().__call__(argv, **kwargs)

    with pytest.raises(TX.TransactionError, match="not detached"):
        TX.preflight(
            fixture.request,
            now=NOW,
            policy=fixture.policy,
            validator=fixture.validator,
            runner=AttachedAq4Runner(fixture),
        )


def test_aq4_promotion_copy_is_no_replace_and_never_overwrites_racer(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    raced = Path(
        fixture.authorization_document["aq4_release"][
            "promotion_evidence"
        ]["path"]
    )
    attacker = b'{"attacker":"owned"}\n'

    class RacingCopyRunner(Runner):
        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            result = super().__call__(argv, **kwargs)
            if (
                not is_git(argv)
                and kwargs["env"]["ULLM_CAMPAIGN_TRANSACTION_STAGE"]
                == "aq4_reasoning_browser"
                and self.stage_call_counts["aq4_reasoning_browser"] == 2
            ):
                raced.write_bytes(attacker)
                raced.chmod(0o444)
            return result

    with pytest.raises(TX.TransactionFailed, match="failed_restored"):
        execute(fixture, RacingCopyRunner(fixture))
    assert raced.read_bytes() == attacker
    outcome = load_outcome(fixture)
    assert outcome["failure_stage"] in {
        "aq4_reasoning_browser",
        "aq4_bundle",
    }
    assert outcome["status"] == "failed_restored"
    assert fixture.active.read_bytes() == fixture.aq4_raw


@pytest.mark.parametrize(
    ("producer", "stage", "stage_call"),
    (
        ("raw_release", "aq4_reasoning_release", 1),
        ("release_evidence", "aq4_reasoning_release", 2),
        ("browser", "aq4_reasoning_browser", 1),
        ("bundle", "aq4_bundle", 1),
    ),
)
def test_aq4_legacy_producer_publication_never_overwrites_final_racer(
    tmp_path: Path,
    producer: str,
    stage: str,
    stage_call: int,
) -> None:
    fixture = Fixture(tmp_path)
    attacker = b"attacker-owned-final\n"
    targets = {
        "raw_release": fixture.campaign_paths["aq4_reasoning_release"],
        "release_evidence": Path(
            fixture.authorization_document["aq4_release"][
                "release_evidence_path"
            ]
        ),
        "browser": fixture.campaign_paths["aq4_reasoning_browser"],
        "bundle": fixture.campaign_paths["aq4_bundle"],
    }
    target = targets[producer]

    class RacingLegacyProducerRunner(Runner):
        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            result = super().__call__(argv, **kwargs)
            if (
                not is_git(argv)
                and kwargs["env"]["ULLM_CAMPAIGN_TRANSACTION_STAGE"]
                == stage
                and self.stage_call_counts[stage] == stage_call
            ):
                output_flag = "--output-dir" if producer == "raw_release" else "--output"
                if producer in {"raw_release", "browser"}:
                    assert Path(self._argument(argv, output_flag)) == target
                    producer_output = Path(
                        kwargs["env"][
                            TX.CAMPAIGN_STAGING_OUTPUT_ENVIRONMENT
                        ]
                    )
                    assert producer_output != target
                    assert producer_output.parent.name.startswith(
                        TX.SERVICE_PRODUCER_STAGING_PREFIX
                    )
                else:
                    producer_output = Path(self._argument(argv, output_flag))
                    assert producer_output != target
                    staging_name = (
                        producer_output.parent.name
                        if producer == "release_evidence"
                        else producer_output.name
                    )
                    assert staging_name.startswith(TX.AQ4_STAGING_PREFIX)
                if producer == "raw_release":
                    target.mkdir(mode=0o700)
                    marker = target / "attacker-marker"
                    marker.write_bytes(attacker)
                    marker.chmod(0o444)
                else:
                    target.write_bytes(attacker)
                    target.chmod(0o444)
            return result

    with pytest.raises(TX.TransactionFailed, match="failed_restored"):
        execute(fixture, RacingLegacyProducerRunner(fixture))

    if producer == "raw_release":
        assert (target / "attacker-marker").read_bytes() == attacker
    else:
        assert target.read_bytes() == attacker
    outcome = load_outcome(fixture)
    assert outcome["failure_stage"] == stage
    assert outcome["status"] == "failed_restored"
    assert fixture.active.read_bytes() == fixture.aq4_raw
    for parent in (fixture.outputs, fixture.aq4_bundle_root):
        assert not any(
            entry.name.startswith(TX.AQ4_STAGING_PREFIX)
            or entry.name.startswith(TX.SERVICE_PRODUCER_STAGING_PREFIX)
            for entry in parent.iterdir()
        )


def test_aq4_promotion_source_mutation_is_repinned_and_restored(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)

    class MutatingPromotionRunner(Runner):
        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            result = super().__call__(argv, **kwargs)
            if (
                not is_git(argv)
                and kwargs["env"]["ULLM_CAMPAIGN_TRANSACTION_STAGE"]
                == "sq8_full"
            ):
                self.fixture.aq4_promotion_evidence.write_bytes(
                    b'{"mutated":true}\n'
                )
            return result

    with pytest.raises(TX.TransactionFailed, match="failed_restored"):
        execute(fixture, MutatingPromotionRunner(fixture))
    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert load_outcome(fixture)["failure_stage"] == "sq8_full"


def test_outcome_publication_failure_keeps_claim_consumed_after_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    monkeypatch.setattr(
        AUTH,
        "publish_outcome",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AUTH.AuthorizationError("outcome failure")
        ),
    )
    with pytest.raises(TX.TransactionError, match="outcome publication"):
        execute(fixture, Runner(fixture))
    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert len(list(fixture.claims.iterdir())) == 1
    assert not list(fixture.outcomes.iterdir())


@pytest.mark.parametrize("timeout", (float("nan"), float("inf"), 3_601.0, 0.0))
def test_command_timeout_must_be_finite_positive_and_bounded(
    tmp_path: Path,
    timeout: float,
) -> None:
    fixture = Fixture(tmp_path)
    request = replace(fixture.request, command_timeout_seconds=timeout)
    with pytest.raises(TX.TransactionError, match="runtime binding"):
        TX.preflight(
            request,
            now=NOW,
            policy=fixture.policy,
            validator=fixture.validator,
            runner=Runner(fixture),
        )


@pytest.mark.parametrize(
    ("lifetime", "expected_timeout"),
    (
        (timedelta(hours=8), 21_600.0),
        (timedelta(hours=2), 7_200.0),
    ),
)
def test_sq8_full_uses_fixed_six_hour_timeout_capped_by_authorization(
    tmp_path: Path,
    lifetime: timedelta,
    expected_timeout: float,
) -> None:
    fixture = Fixture(tmp_path, authorization_lifetime=lifetime)

    class TimeoutCapturingRunner(Runner):
        def __init__(self, selected_fixture: Fixture) -> None:
            super().__init__(selected_fixture)
            self.sq8_timeout: float | None = None

        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            if (
                not is_git(argv)
                and kwargs["env"]["ULLM_CAMPAIGN_TRANSACTION_STAGE"]
                == "sq8_full"
            ):
                self.sq8_timeout = float(kwargs["timeout"])
            return super().__call__(argv, **kwargs)

    runner = TimeoutCapturingRunner(fixture)
    assert execute(fixture, runner).status == "succeeded_restored"
    assert runner.sq8_timeout == expected_timeout
    assert runner.sq8_timeout > 1_800.0
    assert fixture.request.command_timeout_seconds == 10.0


def test_sq8_full_timeout_cannot_exceed_fixed_six_hour_ceiling(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    with pytest.raises(TX.TransactionError, match="timeout is invalid"):
        TX._run_owned_process_group(
            (sys.executable, "-c", "raise SystemExit(0)"),
            request=fixture.request,
            environment=dict(os.environ),
            stage="sq8_full",
            timeout_seconds=TX.SQ8_FULL_MAX_TIMEOUT_SECONDS + 1,
            maximum_timeout_seconds=TX.SQ8_FULL_MAX_TIMEOUT_SECONDS,
        )


def test_sq8_full_timeout_uses_deadline_remaining_after_stabilization(
    tmp_path: Path,
) -> None:
    fixture = Fixture(
        tmp_path,
        authorization_lifetime=timedelta(hours=2),
    )
    timer = FastStabilizationTimer()

    class TimeoutCapturingRunner(Runner):
        sq8_timeout: float | None = None

        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            environment = kwargs["env"]
            if (
                not is_git(argv)
                and isinstance(environment, dict)
                and environment.get("ULLM_CAMPAIGN_TRANSACTION_STAGE")
                == "sq8_full"
            ):
                self.sq8_timeout = float(kwargs["timeout"])
            return super().__call__(argv, **kwargs)

    runner = TimeoutCapturingRunner(fixture)
    result = TX.execute_transaction(
        fixture.request,
        policy=fixture.policy,
        validator=fixture.validator,
        runner=runner,
        inactive_checker=lambda _services: None,
        clock=lambda: NOW + timedelta(seconds=timer.value),
        restoration_probe=live_aq4_proof,
        candidate_stabilization_probe=live_sq8_epoch,
        stabilization_sleeper=timer.sleep,
        stabilization_monotonic=timer.monotonic,
    )

    assert result.status == "succeeded_restored"
    assert timer.value == TX.CANDIDATE_STABILIZATION_SECONDS
    assert runner.sq8_timeout == (
        2 * 60 * 60 - TX.CANDIDATE_STABILIZATION_SECONDS
    )


def test_candidate_stabilization_monitors_exactly_nine_hundred_seconds(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    timer = FastStabilizationTimer()
    probe_calls = 0

    def probe(*args: object) -> dict[str, object]:
        nonlocal probe_calls
        probe_calls += 1
        return live_sq8_epoch(*args)

    result = TX.execute_transaction(
        fixture.request,
        policy=fixture.policy,
        validator=fixture.validator,
        runner=Runner(fixture),
        inactive_checker=lambda _services: None,
        clock=lambda: NOW,
        restoration_probe=live_aq4_proof,
        candidate_stabilization_probe=probe,
        stabilization_sleeper=timer.sleep,
        stabilization_monotonic=timer.monotonic,
    )
    assert result.status == "succeeded_restored"
    assert sum(timer.sleeps) == TX.CANDIDATE_STABILIZATION_SECONDS
    assert all(
        value == TX.CANDIDATE_STABILIZATION_POLL_SECONDS
        for value in timer.sleeps
    )
    assert probe_calls == len(timer.sleeps) + 1


def test_candidate_stabilization_epoch_change_fails_checks_and_restores(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    timer = FastStabilizationTimer()
    calls = 0

    def changing_probe(*args: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        value = live_sq8_epoch(*args)
        if calls > 1:
            value["worker"]["pid"] = 202
        return value

    with pytest.raises(TX.TransactionFailed, match="failed_restored"):
        TX.execute_transaction(
            fixture.request,
            policy=fixture.policy,
            validator=fixture.validator,
            runner=Runner(fixture),
            inactive_checker=lambda _services: None,
            clock=lambda: NOW,
            restoration_probe=live_aq4_proof,
            candidate_stabilization_probe=changing_probe,
            stabilization_sleeper=timer.sleep,
            stabilization_monotonic=timer.monotonic,
        )
    assert load_outcome(fixture)["failure_stage"] == "candidate_checks"
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_candidate_stabilization_checks_deadline_after_each_sleep(
    tmp_path: Path,
) -> None:
    fixture = Fixture(
        tmp_path,
        authorization_lifetime=timedelta(seconds=950),
    )
    timer = FastStabilizationTimer()

    def wall_clock() -> datetime:
        return NOW + timedelta(seconds=timer.value * 2)

    with pytest.raises(TX.TransactionFailed, match="failed_restored"):
        TX.execute_transaction(
            fixture.request,
            policy=fixture.policy,
            validator=fixture.validator,
            runner=Runner(fixture),
            inactive_checker=lambda _services: None,
            clock=wall_clock,
            restoration_probe=live_aq4_proof,
            candidate_stabilization_probe=live_sq8_epoch,
            stabilization_sleeper=timer.sleep,
            stabilization_monotonic=timer.monotonic,
        )
    assert 0 < sum(timer.sleeps) < TX.CANDIDATE_STABILIZATION_SECONDS
    assert load_outcome(fixture)["failure_stage"] == "candidate_checks"
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_campaign_executor_privilege_drop_order_and_exact_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, object] = {
        "uid": 0,
        "gid": 0,
        "groups": (),
    }
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(TX.os, "geteuid", lambda: state["uid"])
    monkeypatch.setattr(TX.os, "getegid", lambda: state["gid"])
    monkeypatch.setattr(TX.os, "getgroups", lambda: list(state["groups"]))

    def setgroups(groups: list[int]) -> None:
        calls.append(("setgroups", tuple(groups)))
        state["groups"] = tuple(groups)

    def setgid(gid: int) -> None:
        calls.append(("setgid", gid))
        state["gid"] = gid

    def setuid(uid: int) -> None:
        calls.append(("setuid", uid))
        state["uid"] = uid

    monkeypatch.setattr(TX.os, "setgroups", setgroups)
    monkeypatch.setattr(TX.os, "setgid", setgid)
    monkeypatch.setattr(TX.os, "setuid", setuid)
    TX._drop_campaign_executor_privileges()
    assert calls == [
        ("setgroups", TX.CAMPAIGN_EXECUTOR_SUPPLEMENTARY_GROUPS),
        ("setgid", TX.CAMPAIGN_EXECUTOR_GID),
        ("setuid", TX.CAMPAIGN_EXECUTOR_UID),
    ]


def test_active_replace_refuses_unexpected_current_bytes(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    before = TX._read_input(
        fixture.active,
        "fixture active",
        TX.MAX_MANIFEST_BYTES,
    )
    slot = TX.ActiveSlot.acquire(
        fixture.active,
        required_uid=os.geteuid(),
    )
    try:
        unexpected = b'{"third-party":"replacement"}\n'
        fixture.active.write_bytes(unexpected)
        fixture.active.chmod(0o644)
        with pytest.raises(TX.TransactionError, match="expected-current"):
            slot.replace(
                fixture.sq8_raw,
                before.identity,
                expected_current=before,
            )
        assert fixture.active.read_bytes() == unexpected
    finally:
        slot.close()


def test_exchange_cas_restores_racing_version_and_durably_loses_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    unexpected = b'{"third-party":"exchange-race"}\n'
    racing = fixture.slot / "racing-active.json"
    racing.write_bytes(unexpected)
    racing.chmod(0o644)
    real_exchange = TX._rename_exchange
    injected = False

    def inject_before_exchange(
        source_name: str,
        destination_name: str,
        *,
        parent_descriptor: int,
    ) -> None:
        nonlocal injected
        if not injected:
            injected = True
            os.replace(
                racing.name,
                destination_name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
        real_exchange(
            source_name,
            destination_name,
            parent_descriptor=parent_descriptor,
        )

    monkeypatch.setattr(TX, "_rename_exchange", inject_before_exchange)
    with pytest.raises(TX.TransactionError, match="failed_restore"):
        execute(fixture, Runner(fixture))

    outcome = load_outcome(fixture)
    assert outcome["status"] == "failed_restore"
    assert outcome["failure_stage"] == "aq4_restore"
    assert outcome["restoration"]["displaced_manifest_sha256"] == digest(
        unexpected
    )
    assert fixture.active.read_bytes() == unexpected
    assert not list(fixture.slot.glob(".active.json.transaction.*.json"))


def test_exchange_cas_never_rolls_back_over_new_active_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    expected = TX._read_input(
        fixture.active,
        "fixture active",
        TX.MAX_MANIFEST_BYTES,
    )
    unexpected = b'{"third-party":"post-exchange-owner"}\n'
    racing = fixture.slot / "racing-active.json"
    racing.write_bytes(unexpected)
    racing.chmod(0o644)
    real_exchange = TX._rename_exchange
    injected = False

    def inject_after_exchange(
        source_name: str,
        destination_name: str,
        *,
        parent_descriptor: int,
    ) -> None:
        nonlocal injected
        real_exchange(
            source_name,
            destination_name,
            parent_descriptor=parent_descriptor,
        )
        if not injected:
            injected = True
            os.replace(
                racing.name,
                destination_name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )

    monkeypatch.setattr(TX, "_rename_exchange", inject_after_exchange)
    slot = TX.ActiveSlot.acquire(
        fixture.active,
        required_uid=os.geteuid(),
    )
    try:
        with pytest.raises(TX.ActiveSlotOwnershipLost):
            slot.replace(
                fixture.sq8_raw,
                expected.identity,
                expected_current=expected,
            )
        assert fixture.active.read_bytes() == unexpected
    finally:
        slot.close()


def test_active_lock_rejects_symlink_parent_component(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir(mode=0o700)
    active = real_parent / "active.json"
    active.write_bytes(b"{}\n")
    active.chmod(0o644)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises((TX.TransactionError, OSError)):
        TX.ActiveSlot.acquire(
            linked_parent / "active.json",
            required_uid=os.geteuid(),
        )


def test_output_inventory_detects_tree_mutation_during_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    runner = Runner(fixture)
    runner(
        ["reasoning-browser", "--output", str(fixture.campaign_paths["reasoning_browser"])],
        env={
            "ULLM_CAMPAIGN_TRANSACTION_STAGE": "reasoning_browser",
            TX.CAMPAIGN_STAGING_OUTPUT_ENVIRONMENT: str(
                fixture.campaign_paths["reasoning_browser"]
            ),
        },
    )
    output = fixture.campaign_paths["reasoning_browser"]
    target = output / "browser-evidence.json"
    real_inventory_file = TX._inventory_file
    mutated = False

    def mutate_once(path: Path, label: str) -> tuple[int, str]:
        nonlocal mutated
        result = real_inventory_file(path, label)
        if not mutated:
            mutated = True
            target.chmod(0o644)
            target.write_text("mutated\n", encoding="ascii")
            target.chmod(0o444)
        return result

    monkeypatch.setattr(TX, "_inventory_file", mutate_once)
    with pytest.raises(TX.TransactionError, match="changed during inventory"):
        TX._output_inventory(
            output,
            run_id=fixture.authorization_document["campaigns"][
                "reasoning_browser"
            ]["run_id"],
            campaign_name="reasoning_browser",
            required_uid=os.geteuid(),
            candidate_raw=fixture.sq8_raw,
        )


def test_browser_inventory_rejects_legacy_file_layout(tmp_path: Path) -> None:
    output = tmp_path / "browser.json"
    output.write_text("{}\n", encoding="ascii")
    output.chmod(0o444)
    with pytest.raises(TX.TransactionError, match="must be a directory"):
        TX._output_inventory(
            output,
            run_id="reasoning-browser-run",
            campaign_name="reasoning_browser",
            required_uid=os.geteuid(),
            candidate_raw=b"",
        )


def test_timeout_kills_owned_descendant_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    child_pid_path = tmp_path / "child.pid"
    request = replace(fixture.request, command_timeout_seconds=0.2)
    monkeypatch.setattr(TX, "COMMAND_TERMINATION_GRACE_SECONDS", 0.2)
    script = (
        "import pathlib,signal,subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "time.sleep(60)']);"
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid));"
        "time.sleep(60)"
    )
    with pytest.raises(TX.TransactionError, match="command failed"):
        TX._run_owned_process_group(
            (sys.executable, "-c", script, str(child_pid_path)),
            request=request,
            environment=dict(os.environ),
            stage="process-group-fixture",
        )
    child_pid = int(child_pid_path.read_text(encoding="ascii"))
    deadline = time.monotonic() + 2.0
    state: str | None = None
    while time.monotonic() < deadline:
        try:
            state = (Path("/proc") / str(child_pid) / "stat").read_text(
                encoding="ascii"
            ).split()[2]
        except (FileNotFoundError, IndexError):
            state = None
            break
        if state in {"Z", "X"}:
            break
        time.sleep(0.02)
    assert state is None or state in {"Z", "X"}


def test_successful_command_cannot_escape_with_double_fork_setsid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    daemon_pid_path = tmp_path / "daemon.pid"
    request = replace(fixture.request, command_timeout_seconds=2.0)
    monkeypatch.setattr(TX, "COMMAND_TERMINATION_GRACE_SECONDS", 0.2)
    daemon = (
        "import os,pathlib,signal,sys,time;"
        "pid=os.fork();"
        "pid and os._exit(0);"
        "os.setsid();"
        "pid=os.fork();"
        "pid and os._exit(0);"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));"
        "time.sleep(60)"
    )
    root = (
        "import pathlib,subprocess,sys,time;"
        "path=pathlib.Path(sys.argv[1]);"
        "subprocess.Popen([sys.executable,'-c',sys.argv[2],sys.argv[1]]);"
        "deadline=time.monotonic()+1;"
        "\nwhile not path.exists() and time.monotonic()<deadline: time.sleep(.01);"
        "\nsys.exit(0 if path.exists() else 9)"
    )
    with pytest.raises(TX.TransactionError, match="command failed"):
        TX._run_owned_process_group(
            (sys.executable, "-c", root, str(daemon_pid_path), daemon),
            request=request,
            environment=dict(os.environ),
            stage="double-fork-fixture",
        )
    daemon_pid = int(daemon_pid_path.read_text(encoding="ascii"))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not (Path("/proc") / str(daemon_pid)).exists():
            break
        time.sleep(0.02)
    assert not (Path("/proc") / str(daemon_pid)).exists()


def test_candidate_window_expiry_aborts_stage_and_restores_aq4(
    tmp_path: Path,
) -> None:
    fixture = Fixture(
        tmp_path,
        authorization_lifetime=timedelta(seconds=2),
    )

    class MutableClock:
        def __init__(self) -> None:
            self.value = NOW

        def __call__(self) -> datetime:
            return self.value

    selected_clock = MutableClock()

    class ExpiringRunner(Runner):
        def __init__(self, selected_fixture: Fixture) -> None:
            super().__init__(selected_fixture)
            self.candidate_timeout: float | None = None

        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            if not is_git(argv):
                stage = str(kwargs["env"]["ULLM_CAMPAIGN_TRANSACTION_STAGE"])
                if stage == "candidate_reconciliation":
                    self.candidate_timeout = float(kwargs["timeout"])
                    selected_clock.value = NOW + timedelta(seconds=3)
            return super().__call__(argv, **kwargs)

    runner = ExpiringRunner(fixture)
    with pytest.raises(TX.TransactionError, match="failed_restored"):
        TX.execute_transaction(
            fixture.request,
            policy=fixture.policy,
            validator=fixture.validator,
            runner=runner,
            inactive_checker=lambda _services: None,
            clock=selected_clock,
            restoration_probe=live_aq4_proof,
            **stabilization_kwargs(),
        )
    assert runner.candidate_timeout is not None
    assert 0 < runner.candidate_timeout <= 2.0
    assert fixture.active.read_bytes() == fixture.aq4_raw
    outcome = load_outcome(fixture)
    assert outcome["status"] == "failed_restored"
    assert outcome["failure_stage"] == "candidate_reconciliation"


@pytest.mark.parametrize("field", ("systemd_unit", "environment_file"))
def test_runtime_binding_rejects_same_byte_noncanonical_config_copy(
    tmp_path: Path,
    field: str,
) -> None:
    fixture = Fixture(tmp_path)
    original = getattr(fixture.request, field)
    copied = fixture.slot / f"copied-{original.name}"
    copied.write_bytes(original.read_bytes())
    request = replace(fixture.request, **{field: copied})
    with pytest.raises(TX.TransactionError, match="runtime binding"):
        TX.preflight(
            request,
            now=NOW,
            policy=fixture.policy,
            validator=fixture.validator,
            runner=Runner(fixture),
        )


def test_source_or_unit_mutation_is_repinned_and_fails_restore_proof(
    tmp_path: Path,
) -> None:
    class MutatingRunner(Runner):
        def __init__(self, fixture: Fixture, mutation: str) -> None:
            super().__init__(fixture)
            self.mutation = mutation
            self.source_dirty = False

        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            if (
                is_git(argv)
                and git_arguments(argv)
                == (
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    "--ignore-submodules=all",
                    "--no-renames",
                )
                and self.source_dirty
            ):
                return subprocess.CompletedProcess(argv, 0, " M tools/x.py\n", "")
            result = super().__call__(argv, **kwargs)
            stage = (
                str(kwargs["env"]["ULLM_CAMPAIGN_TRANSACTION_STAGE"])
                if not is_git(argv)
                else ""
            )
            if stage == "sq8_full":
                if self.mutation == "source":
                    self.source_dirty = True
                else:
                    self.fixture.unit.write_text(
                        "[Service]\nExecStart=/changed\n",
                        encoding="ascii",
                    )
            return result

    for mutation in ("source", "unit"):
        fixture_root = tmp_path / mutation
        fixture_root.mkdir()
        fixture_root.chmod(0o700)
        fixture = Fixture(fixture_root)
        with pytest.raises(TX.TransactionFailed, match="failed_restore"):
            execute(fixture, MutatingRunner(fixture, mutation))
        assert fixture.active.read_bytes() == fixture.aq4_raw
        assert load_outcome(fixture)["status"] == "failed_restore"


@pytest.mark.parametrize(
    ("source_name", "target_stage"),
    (
        ("source", "sq8_full"),
        ("aq4_source", "aq4_reasoning_release"),
    ),
)
def test_transient_source_tool_replacement_and_restoration_is_detected(
    tmp_path: Path,
    source_name: str,
    target_stage: str,
) -> None:
    fixture = Fixture(tmp_path)

    class TransientReplacementRunner(Runner):
        replaced = False

        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            stage = (
                str(kwargs["env"]["ULLM_CAMPAIGN_TRANSACTION_STAGE"])
                if not is_git(argv)
                else ""
            )
            if stage != target_stage or self.replaced:
                return super().__call__(argv, **kwargs)
            self.replaced = True
            source_root = Path(getattr(self.fixture, source_name))
            script = source_root / "tools" / "fixture-stage.py"
            held = source_root / "tools" / "fixture-stage.original"
            malicious = source_root / "tools" / "fixture-stage.replacement"
            script.rename(held)
            malicious.write_text("raise SystemExit(99)\n", encoding="ascii")
            malicious.chmod(0o644)
            malicious.rename(script)
            try:
                return super().__call__(argv, **kwargs)
            finally:
                script.unlink()
                held.rename(script)

    with pytest.raises(TX.TransactionError, match="failed_restore"):
        execute(fixture, TransientReplacementRunner(fixture))
    assert fixture.active.read_bytes() == fixture.aq4_raw
    outcome = load_outcome(fixture)
    assert outcome["stages"][target_stage] == "failed"
    assert outcome["status"] == (
        "failed_restore" if source_name == "source" else "failed_restored"
    )


def test_git_and_stage_environments_do_not_inherit_injection_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    monkeypatch.setenv("LD_PRELOAD", "/attacker/preload.so")
    monkeypatch.setenv("PYTHONPATH", "/attacker/python")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/attacker/gitconfig")

    class EnvironmentCheckingRunner(Runner):
        def __init__(self, selected_fixture: Fixture) -> None:
            super().__init__(selected_fixture)
            self.producer_stages: list[str] = []

        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            environment = kwargs["env"]
            assert isinstance(environment, dict)
            if is_git(argv):
                assert environment == TX.campaign_source_seal.GIT_ENVIRONMENT
            else:
                assert environment["PATH"] == TX.STAGE_BASE_ENVIRONMENT["PATH"]
                assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
                assert environment["PYTHONNOUSERSITE"] == "1"
                assert environment["PYTHONSAFEPATH"] == "1"
                assert "LD_PRELOAD" not in environment
                assert "PYTHONPATH" not in environment
                assert environment.get("GIT_CONFIG_GLOBAL") is None
                staging = environment.get(
                    TX.CAMPAIGN_STAGING_OUTPUT_ENVIRONMENT
                )
                source_root = environment.get(
                    TX.CAMPAIGN_SOURCE_ROOT_ENVIRONMENT
                )
                if staging is not None:
                    stage = environment[
                        "ULLM_CAMPAIGN_TRANSACTION_STAGE"
                    ]
                    self.producer_stages.append(stage)
                    assert Path(staging).name == (
                        TX.SERVICE_PRODUCER_OUTPUT_NAME
                    )
                    expected_source = (
                        fixture.aq4_source
                        if stage
                        in {
                            "aq4_reasoning_release",
                            "aq4_reasoning_browser",
                        }
                        else fixture.source
                    )
                    assert source_root == str(expected_source)
                else:
                    assert source_root is None
            return super().__call__(argv, **kwargs)

    runner = EnvironmentCheckingRunner(fixture)
    assert execute(fixture, runner).status == "succeeded_restored"
    assert runner.producer_stages == [
        "sq8_full",
        "reasoning_release",
        "reasoning_browser",
        "aq4_reasoning_release",
        "aq4_reasoning_browser",
    ]


def test_output_precreated_after_preflight_is_rejected_and_aq4_restored(
    tmp_path: Path,
) -> None:
    class PrecreatingRunner(Runner):
        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            result = super().__call__(argv, **kwargs)
            if not is_git(argv) and (
                kwargs["env"]["ULLM_CAMPAIGN_TRANSACTION_STAGE"]
                == "candidate_checks"
            ):
                self.fixture.campaign_paths["sq8_full"].mkdir()
            return result

    fixture = Fixture(tmp_path)
    with pytest.raises(TX.TransactionFailed, match="failed_restored"):
        execute(fixture, PrecreatingRunner(fixture))
    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert load_outcome(fixture)["failure_stage"] == "sq8_full"


@pytest.mark.parametrize("mutation", ("symlink", "hardlink", "path_leak"))
def test_service_producer_staging_attack_is_rejected_cleaned_and_restored(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = Fixture(tmp_path)

    class MaliciousProducerRunner(Runner):
        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            result = super().__call__(argv, **kwargs)
            if (
                not is_git(argv)
                and kwargs["env"]["ULLM_CAMPAIGN_TRANSACTION_STAGE"]
                == "sq8_full"
            ):
                actual = Path(
                    kwargs["env"][
                        TX.CAMPAIGN_STAGING_OUTPUT_ENVIRONMENT
                    ]
                )
                target = actual / "environment.json"
                target.unlink()
                if mutation == "symlink":
                    target.symlink_to(fixture.candidate)
                elif mutation == "hardlink":
                    os.link(actual / "candidate-served-model.json", target)
                else:
                    target.write_text(str(actual) + "\n", encoding="ascii")
                    target.chmod(0o600)
            return result

    with pytest.raises(TX.TransactionFailed, match="failed_restored"):
        execute(fixture, MaliciousProducerRunner(fixture))
    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert load_outcome(fixture)["failure_stage"] == "sq8_full"
    assert not fixture.campaign_paths["sq8_full"].exists()
    assert not any(
        entry.name.startswith(TX.SERVICE_PRODUCER_STAGING_PREFIX)
        for entry in fixture.outputs.iterdir()
    )


def test_service_producer_output_swap_after_adoption_is_rejected_and_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    real_capture = TX.campaign_runtime_seal.capture_runtime_tree_seal
    swapped = False

    def capture_and_swap(
        root: Path,
        *,
        label: str,
        required_uid: int,
    ) -> object:
        nonlocal swapped
        seal = real_capture(
            root,
            label=label,
            required_uid=required_uid,
        )
        if label == "adopted campaign producer output" and not swapped:
            swapped = True
            target = root / "environment.json"
            replacement = root / ".replacement"
            replacement.write_bytes(target.read_bytes())
            replacement.chmod(0o600)
            os.replace(replacement, target)
        return seal

    monkeypatch.setattr(
        TX.campaign_runtime_seal,
        "capture_runtime_tree_seal",
        capture_and_swap,
    )
    with pytest.raises(TX.TransactionFailed, match="failed_restored"):
        execute(fixture, Runner(fixture))
    assert swapped is True
    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert load_outcome(fixture)["failure_stage"] == "sq8_full"
    assert not any(
        entry.name.startswith(TX.SERVICE_PRODUCER_STAGING_PREFIX)
        for entry in fixture.outputs.iterdir()
    )


def test_service_producer_publication_noreplace_preserves_racer_and_restores(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    final = fixture.campaign_paths["sq8_full"]
    attacker = b"attacker-owned-final\n"

    class RacingFinalRunner(Runner):
        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            result = super().__call__(argv, **kwargs)
            if (
                not is_git(argv)
                and kwargs["env"]["ULLM_CAMPAIGN_TRANSACTION_STAGE"]
                == "sq8_full"
            ):
                final.mkdir(mode=0o700)
                marker = final / "attacker-marker"
                marker.write_bytes(attacker)
                marker.chmod(0o600)
            return result

    with pytest.raises(TX.TransactionFailed, match="failed_restored"):
        execute(fixture, RacingFinalRunner(fixture))
    assert (final / "attacker-marker").read_bytes() == attacker
    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert load_outcome(fixture)["failure_stage"] == "sq8_full"
    assert not any(
        entry.name.startswith(TX.SERVICE_PRODUCER_STAGING_PREFIX)
        for entry in fixture.outputs.iterdir()
    )


def test_service_producer_adoption_failure_cleans_staging_and_restores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    monkeypatch.setattr(
        TX,
        "_adopt_service_entry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            TX.TransactionError("fixture adoption failure")
        ),
    )
    with pytest.raises(TX.TransactionFailed, match="failed_restored"):
        execute(fixture, Runner(fixture))
    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert load_outcome(fixture)["failure_stage"] == "sq8_full"
    assert not any(
        entry.name.startswith(TX.SERVICE_PRODUCER_STAGING_PREFIX)
        for entry in fixture.outputs.iterdir()
    )


def test_service_producer_metadata_rejects_unknown_owner() -> None:
    metadata = os.stat(__file__)
    fields = list(metadata)
    fields[4] = 4242
    fields[5] = 4242
    foreign = os.stat_result(fields)
    assert not TX._service_entry_metadata_is_safe(
        foreign,
        root_device=metadata.st_dev,
        control_uid=0,
        control_gid=0,
        service_uid=1000,
        service_gid=1000,
    )


def recovery_request(fixture: Fixture) -> object:
    api_key = fixture.slot / "api-key"
    session = fixture.slot / "session.jwt"
    api_key.write_text("fixture-api-key\n", encoding="ascii")
    session.write_text("fixture-session-jwt\n", encoding="ascii")
    api_key.chmod(0o600)
    session.chmod(0o600)
    return replace(
        fixture.request,
        api_key_file=api_key,
        openwebui_session_token_file=session,
    )


def test_recovery_preflight_has_nonempty_runtime_seals(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    AUTH.claim_authorization(
        fixture.authorization_path,
        now=NOW,
        policy=fixture.policy,
    )
    fixture.backup.write_bytes(fixture.aq4_raw)
    fixture.backup.chmod(0o444)

    pinned = RECOVERY.preflight_recovery(
        recovery_request(fixture),
        now=NOW + timedelta(hours=2),
        policy=fixture.policy,
        validator=fixture.validator,
        runner=Runner(fixture),
    )

    runtime = pinned.transaction_preflight
    assert runtime.runtime_artifact_seals
    assert runtime.runtime_tree_seals
    labels = {sealed.label for sealed in runtime.runtime_artifact_seals}
    assert not runtime.candidate_runtime_artifact_seals
    assert not runtime.candidate_runtime_tree_seals
    assert "candidate SQ8 worker binary" not in labels
    assert "backup AQ4 worker binary" in labels
    assert "authorized immutable AQ4 backup" in labels
    assert fixture.docker_wrapper in {
        sealed.snapshot.path
        for sealed in runtime.aq4_runtime_artifact_seals
    }
    assert Path(RECOVERY.transaction.PYTHON_BINARY) in {
        sealed.snapshot.path
        for sealed in runtime.aq4_runtime_artifact_seals
    }
    assert Path(RECOVERY.transaction.DOCKER_BINARY) in {
        sealed.snapshot.path
        for sealed in runtime.aq4_runtime_artifact_seals
    }


def test_recovery_repin_rejects_empty_runtime_seals(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    AUTH.claim_authorization(
        fixture.authorization_path,
        now=NOW,
        policy=fixture.policy,
    )
    fixture.backup.write_bytes(fixture.aq4_raw)
    fixture.backup.chmod(0o444)
    request = recovery_request(fixture)
    pinned = RECOVERY.preflight_recovery(
        request,
        now=NOW + timedelta(hours=2),
        policy=fixture.policy,
        validator=fixture.validator,
        runner=Runner(fixture),
    )
    unsealed = replace(
        pinned,
        transaction_preflight=replace(
            pinned.transaction_preflight,
            runtime_artifact_seals=(),
            runtime_tree_seals=(),
        ),
    )

    with pytest.raises(RECOVERY.RecoveryError, match="runtime seals"):
        RECOVERY._repin(
            request,
            unsealed,
            now=NOW + timedelta(hours=2),
            policy=fixture.policy,
            runner=Runner(fixture),
        )


def test_recovery_worker_replacement_is_rejected_before_next_command_exec(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    AUTH.claim_authorization(
        fixture.authorization_path,
        now=NOW,
        policy=fixture.policy,
    )
    fixture.backup.write_bytes(fixture.aq4_raw)
    fixture.backup.chmod(0o444)
    fixture.active.write_bytes(fixture.sq8_raw)
    fixture.active.chmod(0o644)
    commands = replace(
        fixture.commands,
        reverse_reconciliation=(
            (str(fixture.aq4_command), "one"),
            (str(fixture.aq4_command), "two"),
        ),
    )
    request = replace(recovery_request(fixture), commands=commands)

    class ReplacingWorkerRunner(Runner):
        def __call__(
            self,
            argv: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess:
            result = super().__call__(argv, **kwargs)
            if (
                not is_git(argv)
                and kwargs["env"]["ULLM_CAMPAIGN_TRANSACTION_STAGE"]
                == "reverse_reconciliation"
                and self.stage_call_counts["reverse_reconciliation"] == 1
            ):
                replacement = self.fixture.slot / "aq4-worker-replacement"
                replacement.write_bytes(self.fixture.aq4_worker.read_bytes())
                replacement.chmod(0o555)
                os.replace(replacement, self.fixture.aq4_worker)
            return result

    runner = ReplacingWorkerRunner(fixture)
    with pytest.raises(RECOVERY.RecoveryFailed) as caught:
        RECOVERY.recover_transaction(
            request,
            policy=fixture.policy,
            validator=fixture.validator,
            runner=runner,
            clock=lambda: NOW + timedelta(hours=2),
            restoration_probe=live_aq4_proof,
        )

    assert runner.stage_call_counts["reverse_reconciliation"] == 1
    assert caught.value.result.status == "failed_restore"
    assert caught.value.result.failure_stage == "final_checks"
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_recovery_runtime_artifacts_are_repinned_after_restoration_probe(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    AUTH.claim_authorization(
        fixture.authorization_path,
        now=NOW,
        policy=fixture.policy,
    )
    fixture.backup.write_bytes(fixture.aq4_raw)
    fixture.backup.chmod(0o444)
    fixture.active.write_bytes(fixture.sq8_raw)
    fixture.active.chmod(0o644)

    def mutating_probe(
        request: object,
        claim: object,
        preflight: object,
    ) -> dict[str, object]:
        proof = live_aq4_proof(request, claim, preflight)
        replacement = fixture.slot / "aq4-worker-replacement"
        replacement.write_bytes(fixture.aq4_worker.read_bytes())
        replacement.chmod(0o555)
        os.replace(replacement, fixture.aq4_worker)
        return proof

    with pytest.raises(RECOVERY.RecoveryFailed) as caught:
        RECOVERY.recover_transaction(
            recovery_request(fixture),
            policy=fixture.policy,
            validator=fixture.validator,
            runner=Runner(fixture),
            clock=lambda: NOW + timedelta(hours=2),
            restoration_probe=mutating_probe,
        )

    assert caught.value.result.status == "failed_restore"
    assert caught.value.result.failure_stage == "final_checks"
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_recovery_does_not_require_candidate_manifest_or_runtime(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    AUTH.claim_authorization(
        fixture.authorization_path,
        now=NOW,
        policy=fixture.policy,
    )
    fixture.backup.write_bytes(fixture.aq4_raw)
    fixture.backup.chmod(0o444)
    fixture.active.write_bytes(fixture.sq8_raw)
    fixture.active.chmod(0o644)
    fixture.candidate.unlink()
    fixture.sq8_worker.unlink()
    for leaf in sorted(
        fixture.sq8_tokenizer.rglob("*"),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        leaf.unlink() if leaf.is_file() else leaf.rmdir()
    fixture.sq8_tokenizer.rmdir()
    for leaf in sorted(
        fixture.sq8_product.rglob("*"),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        leaf.unlink() if leaf.is_file() else leaf.rmdir()
    fixture.sq8_product.rmdir()

    result = RECOVERY.recover_transaction(
        recovery_request(fixture),
        policy=fixture.policy,
        validator=fixture.validator,
        runner=Runner(fixture),
        clock=lambda: NOW + timedelta(hours=2),
        restoration_probe=live_aq4_proof,
    )

    assert result.status == "restored"
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_failed_restore_outcome_can_use_locked_recovery_route(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    with pytest.raises(TX.TransactionFailed, match="failed_restore"):
        execute(fixture, Runner(fixture, fail_stage="final_checks"))
    assert load_outcome(fixture)["status"] == "failed_restore"

    result = RECOVERY.recover_transaction(
        recovery_request(fixture),
        policy=fixture.policy,
        validator=fixture.validator,
        runner=Runner(fixture),
        clock=lambda: NOW + timedelta(hours=2),
        restoration_probe=live_aq4_proof,
    )

    assert result.status == "restored"
    assert fixture.active.read_bytes() == fixture.aq4_raw
    _snapshot, receipt = AUTH.load_recovery(
        fixture.authorization_path,
        now=NOW + timedelta(hours=2),
        policy=fixture.policy,
    )
    assert receipt["status"] == "restored"
    assert receipt["active_before"]["state"] == "aq4"
    assert receipt["restoration"]["proof"]["passed"] is True


def test_crash_recovery_restores_unknown_safe_regular_current_state(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    AUTH.claim_authorization(
        fixture.authorization_path,
        now=NOW,
        policy=fixture.policy,
    )
    fixture.backup.write_bytes(fixture.aq4_raw)
    fixture.backup.chmod(0o444)
    fixture.active.write_bytes(b'{"unrelated":"manifest"}\n')
    fixture.active.chmod(0o644)

    result = RECOVERY.recover_transaction(
        recovery_request(fixture),
        policy=fixture.policy,
        validator=fixture.validator,
        runner=Runner(fixture),
        clock=lambda: NOW + timedelta(hours=2),
        restoration_probe=live_aq4_proof,
    )

    assert result.status == "restored"
    assert fixture.active.read_bytes() == fixture.aq4_raw
    _snapshot, receipt = AUTH.load_recovery(
        fixture.authorization_path,
        now=NOW + timedelta(hours=2),
        policy=fixture.policy,
    )
    assert receipt["active_before"]["state"] == "unknown"
    assert receipt["active_before"]["sha256"] == digest(
        b'{"unrelated":"manifest"}\n'
    )


def test_recovery_exchange_race_preserves_new_owner_and_publishes_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Fixture(tmp_path)
    AUTH.claim_authorization(
        fixture.authorization_path,
        now=NOW,
        policy=fixture.policy,
    )
    fixture.backup.write_bytes(fixture.aq4_raw)
    fixture.backup.chmod(0o444)
    fixture.active.write_bytes(fixture.sq8_raw)
    fixture.active.chmod(0o644)
    unexpected = b'{"third-party":"recovery-race"}\n'
    racing = fixture.slot / "recovery-racing-active.json"
    racing.write_bytes(unexpected)
    racing.chmod(0o644)
    real_exchange = RECOVERY.transaction._rename_exchange
    injected = False

    def inject_before_exchange(
        source_name: str,
        destination_name: str,
        *,
        parent_descriptor: int,
    ) -> None:
        nonlocal injected
        if not injected:
            injected = True
            os.replace(
                racing.name,
                destination_name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
        real_exchange(
            source_name,
            destination_name,
            parent_descriptor=parent_descriptor,
        )

    monkeypatch.setattr(
        RECOVERY.transaction,
        "_rename_exchange",
        inject_before_exchange,
    )
    with pytest.raises(RECOVERY.RecoveryFailed) as caught:
        RECOVERY.recover_transaction(
            recovery_request(fixture),
            policy=fixture.policy,
            validator=fixture.validator,
            runner=Runner(fixture),
            clock=lambda: NOW + timedelta(hours=2),
            restoration_probe=live_aq4_proof,
        )

    assert caught.value.result.status == "failed_restore"
    assert caught.value.result.failure_stage == "aq4_restore"
    assert fixture.active.read_bytes() == unexpected
    _snapshot, receipt = AUTH.load_recovery(
        fixture.authorization_path,
        now=NOW + timedelta(hours=2),
        policy=fixture.policy,
    )
    assert receipt["restoration"]["displaced_manifest_sha256"] == digest(
        unexpected
    )


def test_claim_only_crash_bootstraps_missing_backup_from_exact_aq4(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    AUTH.claim_authorization(
        fixture.authorization_path,
        now=NOW,
        policy=fixture.policy,
    )
    assert not fixture.backup.exists()

    result = RECOVERY.recover_transaction(
        recovery_request(fixture),
        policy=fixture.policy,
        validator=fixture.validator,
        runner=Runner(fixture),
        clock=lambda: NOW + timedelta(hours=2),
        restoration_probe=live_aq4_proof,
    )

    assert result.status == "restored"
    assert fixture.backup.read_bytes() == fixture.aq4_raw


def test_recovery_does_not_require_disappeared_aq4_campaign_source(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    AUTH.claim_authorization(
        fixture.authorization_path,
        now=NOW,
        policy=fixture.policy,
    )
    fixture.backup.write_bytes(fixture.aq4_raw)
    fixture.backup.chmod(0o444)
    fixture.active.write_bytes(fixture.sq8_raw)
    fixture.active.chmod(0o644)
    (fixture.aq4_source / "tools" / "fixture-stage.py").unlink()
    (fixture.aq4_source / "tools").rmdir()
    (fixture.aq4_source / ".git").rmdir()
    fixture.aq4_source.rmdir()
    fixture.aq4_bundle_root.rmdir()

    result = RECOVERY.recover_transaction(
        recovery_request(fixture),
        policy=fixture.policy,
        validator=fixture.validator,
        runner=Runner(fixture),
        clock=lambda: NOW + timedelta(hours=2),
        restoration_probe=live_aq4_proof,
    )
    assert result.status == "restored"
    assert fixture.active.read_bytes() == fixture.aq4_raw
    assert stat.S_IMODE(fixture.backup.stat().st_mode) == 0o444
    assert fixture.backup.stat().st_nlink == 1
    assert fixture.active.read_bytes() == fixture.aq4_raw


def test_recovery_still_requires_a_sealed_sq8_campaign_source(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    AUTH.claim_authorization(
        fixture.authorization_path,
        now=NOW,
        policy=fixture.policy,
    )
    fixture.backup.write_bytes(fixture.aq4_raw)
    fixture.backup.chmod(0o444)
    fixture.active.write_bytes(fixture.sq8_raw)
    fixture.active.chmod(0o644)
    fixture.source.chmod(0o775)

    with pytest.raises(RECOVERY.RecoveryError, match="preflight failed"):
        RECOVERY.recover_transaction(
            recovery_request(fixture),
            policy=fixture.policy,
            validator=fixture.validator,
            runner=Runner(fixture),
            clock=lambda: NOW + timedelta(hours=2),
            restoration_probe=live_aq4_proof,
        )
    assert fixture.active.read_bytes() == fixture.sq8_raw


def test_recovery_live_proof_mismatch_is_durable_failed_restore(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    AUTH.claim_authorization(
        fixture.authorization_path,
        now=NOW,
        policy=fixture.policy,
    )
    fixture.backup.write_bytes(fixture.aq4_raw)
    fixture.backup.chmod(0o444)
    fixture.active.write_bytes(fixture.sq8_raw)
    fixture.active.chmod(0o644)

    def wrong_worker_proof(
        request: object,
        claim: object,
        preflight: object,
    ) -> dict[str, object]:
        proof = live_aq4_proof(request, claim, preflight)
        proof["worker"]["executable_sha256"] = "9" * 64
        return proof

    with pytest.raises(RECOVERY.RecoveryFailed) as caught:
        RECOVERY.recover_transaction(
            recovery_request(fixture),
            policy=fixture.policy,
            validator=fixture.validator,
            runner=Runner(fixture),
            clock=lambda: NOW + timedelta(hours=2),
            restoration_probe=wrong_worker_proof,
        )

    assert caught.value.result.status == "failed_restore"
    assert caught.value.result.failure_stage == "final_checks"
    assert fixture.active.read_bytes() == fixture.aq4_raw
    _snapshot, receipt = AUTH.load_recovery(
        fixture.authorization_path,
        now=NOW + timedelta(hours=2),
        policy=fixture.policy,
    )
    assert receipt["status"] == "failed_restore"
    assert receipt["restoration"]["proof"] is None
