from __future__ import annotations

import hashlib
import importlib.util
import json
import os
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
AQ4_WORKER = "2" * 64
SQ8_WORKER = "4" * 64


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


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
        self.slot = tmp_path / "slot"
        self.slot.mkdir()
        self.slot.chmod(0o700)
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
        self.active = self.slot / "active.json"
        self.candidate = self.slot / "candidate.json"
        self.aq4_raw = (
            json.dumps(
                {
                    "schema_version": "ullm.served_model.v2",
                    "promotion": {
                        "source_commit": "c" * 40,
                        "receipt": "aq4-receipt.json",
                        "receipt_sha256": "1" * 64,
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
        self.backup = self.outputs / "aq4-backup.json"
        self.campaign_paths = {
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
                "worker_binary_sha256": AQ4_WORKER,
                "promotion_source_commit": "c" * 40,
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
            candidate_reconciliation=(("candidate-reconciliation",),),
            candidate_checks=(("candidate-checks",),),
            sq8_full=("sq8-full",),
            reasoning_release=("reasoning-release",),
            reasoning_browser=("reasoning-browser",),
            reverse_reconciliation=(("reverse-reconciliation",),),
            final_checks=(("final-checks",),),
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
                "final_path": str(output),
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
        if argv[0] == "git":
            arguments = argv[1:]
            values = {
                ("rev-parse", "--show-toplevel"): str(self.fixture.source),
                ("rev-parse", "HEAD"): SOURCE_COMMIT,
                ("rev-parse", "HEAD^{tree}"): SOURCE_TREE,
                ("status", "--porcelain=v1", "--untracked-files=all"): "",
            }
            return subprocess.CompletedProcess(
                argv,
                0,
                values[tuple(arguments)] + ("\n" if values[tuple(arguments)] else ""),
                "",
            )
        stage = str(kwargs["env"]["ULLM_CAMPAIGN_TRANSACTION_STAGE"])
        self.stage_calls.append(stage)
        if stage == self.interrupt_stage:
            raise TX.TransactionInterrupted("fixture interruption")
        if stage == self.fail_stage:
            return subprocess.CompletedProcess(argv, 19, "", "")
        if stage == "sq8_full":
            output = self.fixture.campaign_paths["sq8_full"]
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
            output = self.fixture.campaign_paths["reasoning_release"]
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
            output = self.fixture.campaign_paths["reasoning_browser"]
            output.mkdir(mode=0o700)
            self._write_v2_artifacts(
                output,
                campaign_name="reasoning_browser",
                evidence_name="browser-evidence.json",
                evidence_schema="ullm.openwebui.reasoning_browser_smoke.v4",
                files=TX.REASONING_BROWSER_V2_FILES,
            )
            output.chmod(0o555)
        if stage == self.mutate_active_stage:
            self.fixture.active.write_bytes(b'{"unexpected":true}\n')
        return subprocess.CompletedProcess(argv, 0, "", "")


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


def execute(fixture: Fixture, runner: Runner) -> object:

    return TX.execute_transaction(
        fixture.request,
        policy=fixture.policy,
        validator=fixture.validator,
        runner=runner,
        inactive_checker=lambda _services: None,
        clock=lambda: NOW,
        restoration_probe=live_aq4_proof,
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


@pytest.mark.parametrize(
    "failed_stage",
    (
        "candidate_reconciliation",
        "candidate_checks",
        "sq8_full",
        "reasoning_release",
        "reasoning_browser",
    ),
)
def test_candidate_window_failure_still_restores_and_reconciles_aq4(
    tmp_path: Path,
    failed_stage: str,
) -> None:
    fixture = Fixture(tmp_path)
    with pytest.raises(TX.TransactionError, match="failed_restored"):
        execute(fixture, Runner(fixture, fail_stage=failed_stage))

    outcome = load_outcome(fixture)
    assert outcome["status"] == "failed_restored"
    assert outcome["stages"][failed_stage] == "failed"
    assert outcome["stages"]["aq4_restore"] == "passed"
    assert outcome["stages"]["reverse_reconciliation"] == "passed"
    assert outcome["stages"]["final_checks"] == "passed"
    assert fixture.active.read_bytes() == fixture.aq4_raw


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
    assert not list(fixture.claims.iterdir())
    assert not list(fixture.outcomes.iterdir())
    assert not fixture.backup.exists()


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
        ["reasoning-browser"],
        env={"ULLM_CAMPAIGN_TRANSACTION_STAGE": "reasoning_browser"},
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
            if argv[0] != "git":
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
                argv[0] == "git"
                and tuple(argv[1:])
                == ("status", "--porcelain=v1", "--untracked-files=all")
                and self.source_dirty
            ):
                return subprocess.CompletedProcess(argv, 0, " M tools/x.py\n", "")
            result = super().__call__(argv, **kwargs)
            stage = (
                str(kwargs["env"]["ULLM_CAMPAIGN_TRANSACTION_STAGE"])
                if argv[0] != "git"
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
            if argv[0] != "git" and (
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
    assert stat.S_IMODE(fixture.backup.stat().st_mode) == 0o444
    assert fixture.backup.stat().st_nlink == 1
    assert fixture.active.read_bytes() == fixture.aq4_raw


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
