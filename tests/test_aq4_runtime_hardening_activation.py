from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/aq4_runtime_hardening_activation.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AQ4 = load_module("aq4_runtime_hardening_activation_test", TOOL)


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write(path: Path, raw: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)


def seal_tree(path: Path) -> None:
    for current, directories, files in os.walk(path, topdown=False):
        root = Path(current)
        for name in files:
            (root / name).chmod(0o444)
        for name in directories:
            (root / name).chmod(0o555)
    path.chmod(0o555)


class Fixture:
    def __init__(self, tmp_path: Path) -> None:
        self.uid = os.geteuid()
        self.root = tmp_path / "protected"
        self.root.mkdir(mode=0o755)
        self.activation = self.root / "activation"
        self.activation.mkdir(mode=0o755)
        self.recovery_audits = self.activation / "recovery-attempts"
        self.rollback_audits = self.activation / "rollback-attempts"
        self.live_proof_audits = self.activation / "live-proof-attempts"
        self.recovery_audits.mkdir(mode=0o755)
        self.rollback_audits.mkdir(mode=0o755)
        self.live_proof_audits.mkdir(mode=0o755)
        self.control_source, self.control_commit = self._source("control-source")
        self.promotion_source, self.commit = self._source("promotion-source")
        self.control_tools = [
            self.control_source / relative for relative in AQ4.CONTROL_TOOL_RELATIVE_PATHS
        ]

        self.legacy = tmp_path / "legacy"
        self.legacy.mkdir(mode=0o755)
        self.new_worker = self.root / "releases/aq4/ullm-aq4-worker"
        self.old_worker = self.legacy / "ullm-aq4-worker"
        worker_raw = b"AQ4 worker fixture bytes\n"
        write(self.new_worker, worker_raw, 0o555)
        write(self.old_worker, worker_raw, 0o755)
        self.worker_sha = digest(worker_raw)

        self.new_product, self.new_tokenizer, self.new_receipt = self._runtime_tree(
            self.root / "products/new", self.root / "tokenizers/new", self.root / "promotion/receipt.json"
        )
        self.old_product, self.old_tokenizer, self.old_receipt = self._runtime_tree(
            self.legacy / "product", self.legacy / "tokenizer", self.legacy / "receipt.json"
        )
        self.candidate_raw = self._manifest(
            worker=self.new_worker,
            product=self.new_product,
            tokenizer=self.new_tokenizer,
            receipt=self.new_receipt,
        )
        self.rollback_raw = self._manifest(
            worker=self.old_worker,
            product=self.old_product,
            tokenizer=self.old_tokenizer,
            receipt=self.old_receipt,
        )
        self.manifests = self.root / "manifests"
        self.manifests.mkdir(mode=0o755)
        self.candidate = self.manifests / "aq4-hardened-frozen.json"
        self.rollback = self.activation / "rollback-active.json"
        self.active = tmp_path / "active.json"
        write(self.candidate, self.candidate_raw, 0o444)
        write(self.rollback, self.rollback_raw, 0o444)
        write(self.active, self.rollback_raw, 0o644)

        self.systemd_unit = tmp_path / "ullm-openai.service"
        self.environment = tmp_path / "gateway.env"
        self.credential = tmp_path / "gateway.secret"
        write(self.systemd_unit, b"[Service]\nExecStart=/fixture\n", 0o644)
        write(self.environment, b"ULLM_FIXTURE=1\n", 0o644)
        write(self.credential, b"credential fixture\n", 0o400)
        self.lock = tmp_path / ".active.json.activation.lock"
        write(self.lock, b"", 0o600)
        self.executable = self.root / "control-bin/operation"
        write(self.executable, b"#!/bin/sh\nexit 0\n", 0o555)
        self.operations = self.root / "activation/reviewed-operations.json"
        operations = {
            "schema_version": AQ4.OPERATIONS_SCHEMA,
            "stages": {
                stage: {
                    "argv": [os.fspath(self.executable), stage],
                    "executable_sha256": digest(self.executable.read_bytes()),
                    "timeout_seconds": 10,
                }
                for stage in AQ4.OPERATION_STAGES
            },
        }
        write(self.operations, canonical(operations), 0o444)
        self.plan = self.activation / "activation-plan.json"
        self.intent = self.activation / "activation-intent.json"
        self.outcome = self.activation / "outcome.json"
        self.recovery = self.activation / "recovery.json"
        self.rollback_outcome = self.activation / "rollback-outcome.json"
        self.candidate_proof = self.activation / "candidate-live-proof.json"
        self.rollback_proof = self.activation / "rollback-live-proof.json"
        self.candidate_isolated_preflight = self.activation / "candidate-isolated-preflight.json"

    def _source(self, name: str) -> tuple[Path, str]:
        source = self.root / "sources" / name
        (source / "tools").mkdir(parents=True, mode=0o755)
        tools = [source / relative for relative in AQ4.CONTROL_TOOL_RELATIVE_PATHS]
        for tool in tools:
            write(tool, b"# sealed control fixture\n", 0o644)
        subprocess.run(["git", "init", "-q", os.fspath(source)], check=True)
        subprocess.run(["git", "-C", os.fspath(source), "config", "user.email", "fixture@example.test"], check=True)
        subprocess.run(["git", "-C", os.fspath(source), "config", "user.name", "Fixture"], check=True)
        subprocess.run(["git", "-C", os.fspath(source), "add", "tools"], check=True)
        subprocess.run(["git", "-C", os.fspath(source), "commit", "-q", "-m", "fixture"], check=True)
        commit = subprocess.run(
            ["git", "-C", os.fspath(source), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "-C", os.fspath(source), "checkout", "-q", "--detach", commit],
            check=True,
        )
        for current, directories, files in os.walk(source / ".git", topdown=False):
            root = Path(current)
            for item in files:
                (root / item).chmod(0o644)
            for item in directories:
                (root / item).chmod(0o755)
        (source / ".git").chmod(0o755)
        for tool in tools:
            tool.chmod(0o444)
        (source / "tools").chmod(0o555)
        source.chmod(0o555)
        return source, commit

    def _runtime_tree(self, product: Path, tokenizer: Path, receipt: Path) -> tuple[Path, Path, Path]:
        package = product / "package"
        package.mkdir(parents=True, mode=0o755)
        package_manifest = package / "manifest.json"
        write(package_manifest, b'{"package":"fixture"}\n', 0o444)
        tokenizer.mkdir(parents=True, mode=0o755)
        write(tokenizer / "tokenizer.json", b'{"tokenizer":"fixture"}\n', 0o444)
        write(receipt, b'{"receipt":"fixture"}\n', 0o444)
        seal_tree(product)
        seal_tree(tokenizer)
        return product, tokenizer, receipt

    def _manifest(self, *, worker: Path, product: Path, tokenizer: Path, receipt: Path) -> bytes:
        package = product / "package/manifest.json"
        token = tokenizer / "tokenizer.json"
        return canonical(
            {
                "schema_version": AQ4.SERVED_MODEL_SCHEMA,
                "public": {"id": AQ4.AQ4_MODEL_ID},
                "format": {"format_id": AQ4.AQ4_FORMAT_ID},
                "worker": {
                    "protocol": AQ4.WORKER_PROTOCOL,
                    "binary": os.fspath(worker),
                    "binary_sha256": self.worker_sha,
                },
                "product": {
                    "root": os.fspath(product),
                    "package": {
                        "manifest_path": "package/manifest.json",
                        "manifest_sha256": digest(package.read_bytes()),
                    },
                },
                "tokenizer": {
                    "root": os.fspath(tokenizer),
                    "files": {"tokenizer.json": digest(token.read_bytes())},
                },
                "promotion": {
                    "source_commit": self.commit,
                    "receipt": os.fspath(receipt),
                    "receipt_sha256": digest(receipt.read_bytes()),
                },
            }
        )

    def prepare(self) -> object:
        return AQ4.prepare_plan(
            plan_id="aq4-hardening-fixture-001",
            protected_root=self.root,
            control_source=self.control_source,
            control_source_commit=self.control_commit,
            control_tool_paths=self.control_tools,
            promotion_source=self.promotion_source,
            candidate_manifest=self.candidate,
            active_manifest=self.active,
            rollback_manifest=self.rollback,
            systemd_unit=self.systemd_unit,
            environment_file=self.environment,
            credential_files=[self.credential],
            operations_document=self.operations,
            lock_path=self.lock,
            activation_intent=self.intent,
            activation_outcome=self.outcome,
            activation_recovery=self.recovery,
            rollback_outcome=self.rollback_outcome,
            candidate_live_proof=self.candidate_proof,
            rollback_live_proof=self.rollback_proof,
            candidate_isolated_preflight=self.candidate_isolated_preflight,
            recovery_audit_directory=self.recovery_audits,
            rollback_audit_directory=self.rollback_audits,
            live_proof_audit_directory=self.live_proof_audits,
            output=self.plan,
            expected_worker_sha256=self.worker_sha,
            required_uid=self.uid,
        )

    def isolated_preflight(self, runner: "Runner") -> object:
        record = AQ4.load_plan(self.plan, required_uid=self.uid)
        return AQ4.run_isolated_candidate_preflight(
            record,
            required_uid=self.uid,
            runner=runner,
        )

    def execute(self, runner: "Runner", **kwargs: object) -> object:
        if not self.candidate_isolated_preflight.exists():
            self.isolated_preflight(runner)
        return AQ4.execute_activation(
            self.plan,
            expected_plan_sha256=digest(self.plan.read_bytes()),
            confirmation=AQ4.ACTIVATION_CONFIRMATION,
            required_uid=self.uid,
            runner=runner,
            **kwargs,
        )

    def recover(self, runner: "Runner") -> object:
        return AQ4.execute_activation_recovery(
            self.plan,
            expected_plan_sha256=digest(self.plan.read_bytes()),
            confirmation=AQ4.RECOVERY_CONFIRMATION,
            required_uid=self.uid,
            runner=runner,
        )

    def rollback_execute(self, runner: "Runner") -> object:
        return AQ4.execute_rollback(
            self.plan,
            expected_plan_sha256=digest(self.plan.read_bytes()),
            confirmation=AQ4.ROLLBACK_CONFIRMATION,
            required_uid=self.uid,
            runner=runner,
        )


class Runner:
    def __init__(
        self,
        *,
        fail_stage: str | None = None,
        failure_stderr: str = "fixture failure",
    ) -> None:
        self.fail_stage = fail_stage
        self.failure_stderr = failure_stderr
        self.stages: list[str] = []

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        stage = str(environment["ULLM_AQ4_RUNTIME_HARDENING_STAGE"])
        self.stages.append(stage)
        if stage == self.fail_stage:
            endpoints = {
                name: {"ok": name != "gateway_ready", "status": 200 if name != "gateway_ready" else None,
                       "cause": None if name != "gateway_ready" else "transport"}
                for name in AQ4.LIVE_ENDPOINTS
            }
            diagnostic = {
                "schema_version": "ullm.aq4_runtime_hardening_readiness_failure.v1",
                "cause": "endpoints_incoherent",
                "endpoints": endpoints,
            }
            return subprocess.CompletedProcess(argv, 1, json.dumps(diagnostic), self.failure_stderr)
        if stage == AQ4.ISOLATED_PREFLIGHT_STAGE:
            manifest = json.loads(
                Path(str(environment["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_MANIFEST"])).read_text(
                    encoding="ascii"
                )
            )
            observation = {
                "schema_version": "ullm.aq4_runtime_hardening_isolated_worker_observation.v1",
                "plan_sha256": environment["ULLM_AQ4_RUNTIME_HARDENING_PLAN_SHA256"],
                "operation_epoch": environment["ULLM_AQ4_RUNTIME_HARDENING_EPOCH"],
                "candidate_manifest_sha256": environment["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_SHA256"],
                "stage": AQ4.ISOLATED_PREFLIGHT_STAGE,
                "checked_at": "2026-07-26T00:00:00.000000Z",
                "status": "passed",
                "cause": None,
                "worker": {
                    "model_id": manifest["public"]["id"],
                    "package_manifest_sha256": manifest["product"]["package"]["manifest_sha256"],
                    "device": "fixture-device",
                    "execution_profile": "fixture-profile",
                },
                "operation": {
                    "argv_sha256": "a" * 64,
                    "stdout_sha256": "b" * 64,
                    "stderr_sha256": "c" * 64,
                    "stdout_bytes": 1,
                    "stderr_bytes": 0,
                    "returncode": -15,
                },
                "timing": {
                    "timeout_seconds": 120,
                    "ready_after_milliseconds": 10,
                    "elapsed_milliseconds": 11,
                },
                "cleanup": {"terminated": True, "returncode": -15},
                "production_activation_performed": False,
            }
            return subprocess.CompletedProcess(argv, 0, json.dumps(observation), "")
        if stage.endswith("live_proof"):
            manifest = json.loads(
                Path(str(environment["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_MANIFEST"])).read_text(
                    encoding="ascii"
                )
            )
            observation = {
                "schema_version": "ullm.aq4_runtime_hardening_live_observation.v2",
                "plan_sha256": environment["ULLM_AQ4_RUNTIME_HARDENING_PLAN_SHA256"],
                "operation_epoch": environment["ULLM_AQ4_RUNTIME_HARDENING_EPOCH"],
                "active_manifest_sha256": environment["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_SHA256"],
                "model_id": manifest["public"]["id"],
                "worker_binary_path": manifest["worker"]["binary"],
                "worker_binary_sha256": manifest["worker"]["binary_sha256"],
                "systemd": {
                    "unit": AQ4.DEFAULT_SERVICE_UNIT,
                    "active_state": "active",
                    "sub_state": "running",
                },
                "process": {
                    "boot_id": "fixture-boot",
                    "pid": 11,
                    "ppid": 1,
                    "starttime": 17,
                    "executable_sha256": "f" * 64,
                },
                "manifest": {
                    "active_path": environment["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_MANIFEST"],
                    "active_manifest_sha256": environment["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_SHA256"],
                    "file_match": True,
                    "service_environment_match": True,
                    "worker_command_match": True,
                },
                "endpoints": {
                    name: {"ok": True, "status": 200, "cause": None}
                    for name in AQ4.LIVE_ENDPOINTS
                },
                "readiness": {
                    "timeout_seconds": 120,
                    "max_attempts": 15,
                    "attempts": 2,
                    "stable_pid_observations": 2,
                    "elapsed_milliseconds": 1,
                },
            }
            return subprocess.CompletedProcess(argv, 0, json.dumps(observation), "")
        return subprocess.CompletedProcess(argv, 0, "", "")


def test_preflight_activate_and_manual_rollback(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    record = AQ4.load_plan(fixture.plan, required_uid=fixture.uid)
    report = AQ4.preflight_report(record, required_uid=fixture.uid)
    assert report["ready"] is False
    assert any("candidate_isolated_preflight" in blocker for blocker in report["blockers"])
    fixture.isolated_preflight(Runner())
    report = AQ4.preflight_report(record, required_uid=fixture.uid)
    assert report["ready"] is True
    assert fixture.active.read_bytes() == fixture.rollback_raw

    result = fixture.execute(Runner())

    assert result.status == "activated"
    assert fixture.active.read_bytes() == fixture.candidate_raw
    assert fixture.outcome.stat().st_mode & 0o777 == 0o444
    assert fixture.outcome.stat().st_nlink == 1
    assert fixture.candidate_proof.exists()

    rolled_back = fixture.rollback_execute(Runner())

    assert rolled_back.status == "rolled_back"
    assert fixture.active.read_bytes() == fixture.rollback_raw
    assert fixture.rollback_outcome.stat().st_mode & 0o777 == 0o444


def test_sigkill_after_swap_is_recovered_from_intent(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    pid = os.fork()
    if pid == 0:
        def kill_after_swap(point: str) -> None:
            if point == "after_swap":
                os.kill(os.getpid(), signal.SIGKILL)

        fixture.execute(Runner(), fault_hook=kill_after_swap)
        os._exit(99)
    _waited, status = os.waitpid(pid, 0)
    assert os.WIFSIGNALED(status)
    assert os.WTERMSIG(status) == signal.SIGKILL
    assert fixture.intent.exists()
    assert not fixture.outcome.exists()
    assert fixture.active.read_bytes() == fixture.candidate_raw

    result = fixture.recover(Runner())

    assert result.status == "recovered"
    assert fixture.active.read_bytes() == fixture.rollback_raw
    assert fixture.recovery.exists()


def test_sigkill_after_durable_intent_before_swap_is_recoverable(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    pid = os.fork()
    if pid == 0:
        def kill_after_intent(point: str) -> None:
            if point == "after_intent":
                os.kill(os.getpid(), signal.SIGKILL)

        fixture.execute(Runner(), fault_hook=kill_after_intent)
        os._exit(99)
    _waited, status = os.waitpid(pid, 0)
    assert os.WIFSIGNALED(status)
    assert os.WTERMSIG(status) == signal.SIGKILL
    assert fixture.intent.exists()
    assert not fixture.outcome.exists()
    assert fixture.active.read_bytes() == fixture.rollback_raw

    result = fixture.recover(Runner())

    assert result.status == "recovered"
    assert fixture.active.read_bytes() == fixture.rollback_raw


def test_post_rename_fault_is_not_mistaken_for_precommit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    original = AQ4._rename_exchange

    def exchange_then_fault(parent_fd: int, left: str, right: str) -> None:
        original(parent_fd, left, right)
        raise RuntimeError("fixture post-rename fault")

    monkeypatch.setattr(AQ4, "_rename_exchange", exchange_then_fault)
    with pytest.raises(AQ4.ActivationError):
        fixture.execute(Runner())

    assert fixture.active.read_bytes() == fixture.rollback_raw
    outcome = json.loads(fixture.outcome.read_text(encoding="ascii"))
    assert outcome["status"] == "failed_restored"
    assert outcome["restoration"]["attempted"] is True


def test_recovery_failure_audit_does_not_consume_success_receipt(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    pid = os.fork()
    if pid == 0:
        fixture.execute(
            Runner(),
            fault_hook=lambda point: os.kill(os.getpid(), signal.SIGKILL)
            if point == "after_swap"
            else None,
        )
        os._exit(99)
    os.waitpid(pid, 0)

    with pytest.raises(AQ4.ActivationError):
        fixture.recover(Runner(fail_stage="rollback_live_proof"))
    assert not fixture.recovery.exists()
    assert list(fixture.recovery_audits.glob("recovery-attempt-*.json"))
    assert fixture.active.read_bytes() == fixture.rollback_raw

    result = fixture.recover(Runner())
    assert result.status == "recovered"
    assert fixture.recovery.exists()


def test_successful_plan_cannot_be_replayed(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    fixture.execute(Runner())

    with pytest.raises(AQ4.ActivationError, match="activation intent is already consumed"):
        fixture.execute(Runner())


def test_candidate_live_proof_failure_restores_exact_aq4(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()

    with pytest.raises(AQ4.ActivationError):
        fixture.execute(Runner(fail_stage="candidate_live_proof"))

    outcome = json.loads(fixture.outcome.read_text(encoding="ascii"))
    assert outcome["status"] == "failed_restored"
    assert outcome["failure_stage"] == "candidate_live_proof"
    assert outcome["stages"]["rollback_reconcile"] == "passed"
    assert outcome["stages"]["rollback_live_proof"] == "passed"
    assert fixture.active.read_bytes() == fixture.rollback_raw


def test_live_proof_failure_audit_preserves_safe_diagnostics_and_redacts_credentials(
    tmp_path: Path,
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    leaked_api_key = "aq4-api-key-should-not-be-published"
    leaked_jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmaXh0dXJlIn0.signature"
    stderr = (
        "safe readiness diagnostic; "
        f"Authorization: Bearer {leaked_api_key}; "
        f"api_key={leaked_api_key}; session={leaked_jwt}"
    )

    with pytest.raises(AQ4.ActivationError):
        fixture.execute(Runner(fail_stage="candidate_live_proof", failure_stderr=stderr))

    audits = sorted(fixture.live_proof_audits.glob("candidate_live_proof-attempt-*.json"))
    assert len(audits) == 1
    audit_path = audits[0]
    audit = json.loads(audit_path.read_text(encoding="ascii"))
    serialized = audit_path.read_text(encoding="ascii")
    assert audit["stage"] == "candidate_live_proof"
    assert audit["stage_status"] == "failed"
    assert audit["operation"]["return_code"] == 1
    assert audit["operation"]["cause"] == "endpoints_incoherent"
    assert "safe readiness diagnostic" in audit["operation"]["stderr"]
    assert audit["endpoints"]["gateway_ready"] == {
        "ok": False,
        "status": None,
        "cause": "transport",
    }
    assert audit["endpoints"]["gateway_health"]["ok"] is True
    assert leaked_api_key not in serialized
    assert leaked_jwt not in serialized
    assert "[REDACTED]" in audit["operation"]["stderr"]
    assert audit_path.stat().st_mode & 0o777 == 0o444
    assert audit_path.stat().st_nlink == 1


def test_failed_candidate_readiness_enters_rollback_before_returning_failure(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    runner = Runner(fail_stage="candidate_live_proof")

    with pytest.raises(AQ4.ActivationError):
        fixture.execute(runner)

    outcome = json.loads(fixture.outcome.read_text(encoding="ascii"))
    assert outcome["failure_stage"] == "candidate_live_proof"
    assert outcome["stages"]["candidate_live_proof"] == "failed"
    assert outcome["stages"]["rollback_reconcile"] == "passed"
    assert outcome["stages"]["rollback_live_proof"] == "passed"
    assert runner.stages[-2:] == ["rollback_reconcile", "rollback_live_proof"]
    assert fixture.active.read_bytes() == fixture.rollback_raw


def test_unit_drift_fails_before_intent(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    fixture.systemd_unit.write_bytes(b"[Service]\nExecStart=/changed\n")
    fixture.systemd_unit.chmod(0o644)
    record = AQ4.load_plan(fixture.plan, required_uid=fixture.uid)
    report = AQ4.preflight_report(record, required_uid=fixture.uid)
    assert report["ready"] is False
    assert any("runtime_preconditions" in blocker for blocker in report["blockers"])

    with pytest.raises(AQ4.ActivationError):
        fixture.execute(Runner())
    assert not fixture.intent.exists()
    assert fixture.active.read_bytes() == fixture.rollback_raw


def test_concurrent_lock_fails_without_consuming_intent(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    descriptor = os.open(fixture.lock, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(AQ4.ActivationError, match="already held"):
            fixture.execute(Runner())
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert not fixture.intent.exists()
    assert fixture.active.read_bytes() == fixture.rollback_raw


def test_stale_plan_hash_fails_before_lock_or_intent(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()

    with pytest.raises(AQ4.ActivationError, match="confirmed AQ4 hardening plan SHA-256 differs"):
        AQ4.execute_activation(
            fixture.plan,
            expected_plan_sha256="0" * 64,
            confirmation=AQ4.ACTIVATION_CONFIRMATION,
            required_uid=fixture.uid,
            runner=Runner(),
        )

    assert not fixture.intent.exists()
    assert fixture.active.read_bytes() == fixture.rollback_raw


def test_outcome_publication_fault_after_commit_stays_activated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Fixture(tmp_path)
    fixture.prepare()
    original = AQ4._publish_immutable
    faulted = False

    def publish_then_fault(path: Path, document: dict[str, Any], *, required_uid: int) -> object:
        nonlocal faulted
        published = original(path, document, required_uid=required_uid)
        if path == fixture.outcome and not faulted:
            faulted = True
            raise AQ4.ImmutablePublicationCommittedError("fixture post-publication fault")
        return published

    monkeypatch.setattr(AQ4, "_publish_immutable", publish_then_fault)
    result = fixture.execute(Runner())

    assert faulted is True
    assert result.status == "activated"
    assert fixture.active.read_bytes() == fixture.candidate_raw


def test_route_has_no_sq8_final_route_or_disallowed_service_reference() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert "served_model_final_activation" not in source
    assert "llama-qwen35-udq4.service" not in source
    assert '"gdm3"' not in source


def test_preplan_preflight_is_read_only_and_lists_missing_sealed_inputs(tmp_path: Path) -> None:
    fixture = Fixture(tmp_path)
    missing_root = tmp_path / "not-created"
    before = fixture.active.read_bytes()

    report = AQ4.preplan_preflight_report(
        active_manifest=fixture.active,
        expected_active_sha256=digest(before),
        protected_root=missing_root,
        candidate_manifest=missing_root / "manifests/candidate.json",
        rollback_manifest=missing_root / "activation/rollback.json",
        plan_path=missing_root / "activation/plan.json",
        control_source_parent=missing_root / "control-source",
        operations_document=missing_root / "activation/operations.json",
        lock_path=missing_root / "lock",
        systemd_unit=fixture.systemd_unit,
        environment_file=fixture.environment,
        required_uid=fixture.uid,
    )

    assert report["ready"] is False
    assert report["mode"] == "read_only_preplan"
    assert report["paths"]["candidate_manifest"]["exists"] is False
    assert any("credential seal set" in blocker for blocker in report["blockers"])
    assert fixture.active.read_bytes() == before
