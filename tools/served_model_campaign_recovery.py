#!/usr/bin/env python3
"""Locked recovery of one consumed AQ4-to-SQ8 campaign authorization."""

from __future__ import annotations

import os
import math
import stat
import subprocess
import sys
from dataclasses import replace
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


TOOLS = Path(__file__).resolve().parent
if os.fspath(TOOLS) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS))

import served_model_aq4_restoration_proof as restoration_proof  # noqa: E402
import served_model_campaign_authorization as authorization  # noqa: E402
import served_model_campaign_transaction as transaction  # noqa: E402
from served_model_active_binding import StableFileSnapshot  # noqa: E402


class RecoveryError(transaction.TransactionError):
    """The locked recovery could not prove the live AQ4 route."""


class RecoveryFailed(RecoveryError):
    """A durable failed recovery receipt was published."""

    def __init__(self, message: str, *, result: "RecoveryResult") -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True, slots=True)
class RecoveryPreflight:
    claim: authorization.ClaimRecord
    source_commit: str
    source_tree: str
    active_before: StableFileSnapshot
    active_state: str
    candidate: StableFileSnapshot
    backup: StableFileSnapshot
    candidate_summary: dict[str, Any]
    backup_summary: dict[str, Any]
    systemd_unit_sha256: str
    environment_sha256: str
    candidate_promotion_receipt_sha256: str
    transaction_preflight: transaction.TransactionPreflight
    backup_requires_publication: bool


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    receipt_path: Path
    receipt_sha256: str
    status: str
    failure_stage: str | None


RecoveryProbe = transaction.RestorationProbe


def _require_fresh_recovery_receipt(
    claim: authorization.ClaimRecord,
    *,
    policy: authorization.RegistryPolicy,
) -> None:
    path = authorization.recovery_path(
        claim.authorization.snapshot.sha256,
        policy=policy,
    )
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise RecoveryError("recovery receipt path cannot be inspected") from error
    raise RecoveryError("campaign recovery receipt already exists")


def _require_recovery_eligible_outcome(
    claim: authorization.ClaimRecord,
    *,
    now: datetime,
    policy: authorization.RegistryPolicy,
) -> bool:
    path = authorization.outcome_path(
        claim.authorization.snapshot.sha256,
        policy=policy,
    )
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise RecoveryError("campaign outcome path cannot be inspected") from error
    try:
        _snapshot, document = authorization.load_outcome(
            claim.authorization.snapshot.path,
            now=now,
            policy=policy,
        )
    except authorization.AuthorizationError as error:
        raise RecoveryError("campaign outcome is unsafe") from error
    if document["status"] != "failed_restore":
        raise RecoveryError("campaign outcome does not require recovery")
    return True


def _active_state(
    active: StableFileSnapshot,
    claim: authorization.ClaimRecord,
) -> str:
    before_hash = claim.authorization.document["before"]["manifest_sha256"]
    candidate_hash = claim.authorization.document["candidate"]["manifest_sha256"]
    if active.sha256 == before_hash:
        return "aq4"
    if active.sha256 == candidate_hash:
        return "sq8"
    return "unknown"


def preflight_recovery(
    request: transaction.TransactionRequest,
    *,
    now: datetime,
    policy: authorization.RegistryPolicy = authorization.RegistryPolicy(),
    validator: transaction.ManifestValidator = (
        transaction.default_manifest_validator
    ),
    runner: transaction.CommandRunner = subprocess.run,
) -> RecoveryPreflight:
    """Pin a consumed claim, its immutable backup, and the only two safe states."""

    transaction._validate_commands(request.commands)
    transaction._lexical_absolute(
        request.candidate_manifest,
        "recovery candidate manifest",
    )
    if (
        request.service_unit not in request.inactive_services
        or transaction._lexical_absolute(
            request.active_manifest,
            "recovery active manifest",
        )
        != transaction._lexical_absolute(
            policy.active_manifest_path,
            "policy active manifest",
        )
        or transaction._lexical_absolute(
            request.systemd_unit,
            "recovery systemd unit",
        )
        != transaction._lexical_absolute(
            policy.systemd_unit_path,
            "policy systemd unit",
        )
        or transaction._lexical_absolute(
            request.environment_file,
            "recovery environment file",
        )
        != transaction._lexical_absolute(
            policy.environment_file_path,
            "policy systemd environment file",
        )
        or request.service_unit != policy.service_unit
        or request.api_key_file is None
        or request.openwebui_session_token_file is None
        or (
            policy.required_uid == 0
            and (
                request.api_key_file
                != transaction.FIXED_GATEWAY_API_KEY_PATH
                or request.openwebui_session_token_file
                != transaction.FIXED_OPENWEBUI_SESSION_TOKEN_PATH
            )
        )
        or not math.isfinite(request.command_timeout_seconds)
        or request.command_timeout_seconds <= 0
        or request.command_timeout_seconds
        > transaction.MAX_COMMAND_TIMEOUT_SECONDS
    ):
        raise RecoveryError("recovery runtime binding is incomplete")
    try:
        claim = authorization.load_claim(
            request.authorization_path,
            now=now,
            policy=policy,
        )
    except authorization.AuthorizationError as error:
        raise RecoveryError("consumed campaign claim is unavailable") from error
    try:
        authorization.validate_authorization_document(
            claim.authorization.document,
            now=now,
            required_uid=policy.required_uid,
            require_fresh_outputs=False,
            require_bound_inputs=False,
            enforce_current_window=False,
            policy=policy,
            source_root=request.source_root,
        )
    except authorization.AuthorizationError as error:
        raise RecoveryError("recovery authorization binding differs") from error
    _require_fresh_recovery_receipt(claim, policy=policy)
    outcome_present = _require_recovery_eligible_outcome(
        claim,
        now=now,
        policy=policy,
    )

    command_runtime = transaction._capture_command_runtime_seals(
        request.commands,
        required_uid=policy.required_uid,
        recovery_only=True,
    )
    source_commit, source_tree, source_seal = transaction._sealed_source_identity(
        request.source_root,
        runner=runner,
        required_uid=policy.required_uid,
    )
    active = transaction._read_input(
        request.active_manifest,
        "recovery active served-model manifest",
        transaction.MAX_MANIFEST_BYTES,
    )
    if (
        active.identity.uid != policy.required_uid
        or active.identity.links != 1
        or stat.S_IMODE(active.identity.mode) != 0o644
    ):
        raise RecoveryError("recovery active manifest metadata is unsafe")
    active_state = _active_state(active, claim)
    candidate_authorization = claim.authorization.document["candidate"]
    candidate = StableFileSnapshot(
        request.candidate_manifest,
        b"",
        candidate_authorization["manifest_sha256"],
        active.identity,
    )
    backup_path = Path(
        claim.authorization.document["rollback"]["backup_path"]
    )
    backup_requires_publication = False
    try:
        backup = transaction._read_input(
            backup_path,
            "authorized immutable AQ4 backup",
            transaction.MAX_MANIFEST_BYTES,
        )
    except transaction.TransactionError:
        try:
            backup_path.lstat()
        except FileNotFoundError:
            if outcome_present or active_state != "aq4":
                raise RecoveryError("authorized AQ4 backup is unavailable") from None
            backup_requires_publication = True
            backup = StableFileSnapshot(
                backup_path,
                active.raw,
                active.sha256,
                active.identity,
            )
        else:
            raise
    if (
        backup.sha256
        != claim.authorization.document["before"]["manifest_sha256"]
        or backup.identity.uid != policy.required_uid
        or (
            not backup_requires_publication
            and (
                backup.identity.links != 1
                or stat.S_IMODE(backup.identity.mode) != 0o444
            )
        )
    ):
        raise RecoveryError("authorized AQ4 backup identity differs")
    backup_document = transaction._strict_object(
        backup.raw,
        "backup served-model manifest",
    )
    if (
        backup_document.get("schema_version") != "ullm.served_model.v2"
        or not isinstance(backup_document.get("promotion"), dict)
        or backup_document["promotion"].get("source_commit")
        != claim.authorization.document["before"]["promotion_source_commit"]
    ):
        raise RecoveryError("recovery served-model document identity differs")
    receipt_sha256 = candidate_authorization["promotion_receipt_sha256"]
    before = claim.authorization.document["before"]
    aq4_receipt_path, aq4_receipt_sha256 = transaction._promotion_identity(
        backup_document,
        manifest_parent=active.path.parent,
        source_commit=before["promotion_source_commit"],
        label="backup AQ4",
    )
    if (
        aq4_receipt_path != Path(before["promotion_receipt_path"])
        or aq4_receipt_sha256 != before["promotion_receipt_sha256"]
    ):
        raise RecoveryError("backup AQ4 promotion identity differs")
    aq4_runtime = transaction._manifest_runtime_seals(
        backup_document,
        manifest_path=active.path,
        expected_receipt_path=aq4_receipt_path,
        label="backup AQ4",
        required_uid=policy.required_uid,
    )
    unit_seal = transaction._capture_runtime_artifact(
        request.systemd_unit,
        label="systemd unit",
        maximum=transaction.MAX_INPUT_BYTES,
        required_uid=policy.required_uid,
    )
    environment_seal = transaction._capture_runtime_artifact(
        request.environment_file,
        label="systemd environment file",
        maximum=transaction.MAX_INPUT_BYTES,
        required_uid=policy.required_uid,
    )
    backup_seals: tuple[
        transaction.campaign_runtime_seal.RuntimeArtifactSeal, ...
    ] = ()
    if not backup_requires_publication:
        backup_seals = (
            transaction._capture_runtime_artifact(
                backup.path,
                label="authorized immutable AQ4 backup",
                maximum=transaction.MAX_MANIFEST_BYTES,
                required_uid=policy.required_uid,
            ),
        )
    backup_summary = validator(
        active.path if backup_requires_publication else backup.path
    )
    backup_worker = transaction._summary_identity(
        backup_summary,
        model_id="ullm-qwen3.5-9b-aq4",
        format_id="AQ4_0",
        manifest_sha256=backup.sha256,
        worker_protocol="ullm.worker.v2",
        label="backup AQ4",
    )
    backup_worker_document = backup_summary.get("worker")
    if (
        not isinstance(backup_worker_document, dict)
        or backup_worker_document.get("binary")
        != os.fspath(aq4_runtime.worker.snapshot.path)
        or aq4_runtime.worker.snapshot.sha256 != backup_worker
    ):
        raise RecoveryError("recovery runtime worker identity differs")
    unit = unit_seal.snapshot
    environment = environment_seal.snapshot
    rollback = claim.authorization.document["rollback"]
    if (
        source_commit != claim.authorization.document["source"]["commit"]
        or source_tree != claim.authorization.document["source"]["tree"]
        or backup_worker
        != claim.authorization.document["before"]["worker_binary_sha256"]
        or receipt_sha256
        != claim.authorization.document["candidate"][
            "promotion_receipt_sha256"
        ]
        or (
            backup_seals
            and backup_seals[0].snapshot != backup
        )
        or unit.sha256 != rollback["systemd_unit_sha256"]
        or environment.sha256 != rollback["environment_sha256"]
    ):
        raise RecoveryError("recovery authorization identity differs")
    api_key_sha256 = transaction._validate_private_secret(
        request.api_key_file,
        "gateway API key",
        required_uid=policy.required_uid,
    )
    session_token_sha256 = transaction._validate_private_secret(
        request.openwebui_session_token_file,
        "OpenWebUI session token",
        required_uid=policy.required_uid,
    )
    api_key_seal = transaction._capture_runtime_artifact(
        request.api_key_file,
        label="gateway API key",
        maximum=65_536,
        required_uid=transaction._private_secret_seal_uid(
            request.api_key_file,
            required_uid=policy.required_uid,
        ),
    )
    session_token_seal = transaction._capture_runtime_artifact(
        request.openwebui_session_token_file,
        label="OpenWebUI session token",
        maximum=65_536,
        required_uid=transaction._private_secret_seal_uid(
            request.openwebui_session_token_file,
            required_uid=policy.required_uid,
        ),
    )
    if (
        api_key_seal.snapshot.sha256 != api_key_sha256
        or session_token_seal.snapshot.sha256 != session_token_sha256
    ):
        raise RecoveryError("recovery private credential changed while sealing")
    aq4_runtime_artifact_seals = (
        *aq4_runtime.artifacts,
        *backup_seals,
        *command_runtime.aq4,
    )
    aq4_runtime_tree_seals = aq4_runtime.trees
    shared_runtime_artifact_seals = (
        unit_seal,
        environment_seal,
        api_key_seal,
        session_token_seal,
        *command_runtime.shared,
    )
    shared_runtime_tree_seals: tuple[
        transaction.campaign_runtime_seal.RuntimeTreeSeal, ...
    ] = ()
    runtime_artifact_seals = (
        *aq4_runtime_artifact_seals,
        *shared_runtime_artifact_seals,
    )
    runtime_tree_seals = (
        *aq4_runtime_tree_seals,
        *shared_runtime_tree_seals,
    )
    candidate_summary = {
        "validated": False,
        "manifest_sha256": candidate.sha256,
        "model_id": candidate_authorization["model_id"],
        "format_id": candidate_authorization["format_id"],
        "worker": {
            "binary": None,
            "binary_sha256": candidate_authorization[
                "worker_binary_sha256"
            ],
            "protocol": candidate_authorization["worker_protocol"],
        },
    }
    aq4_source = claim.authorization.document["aq4_release"]["source"]
    pseudo = transaction.TransactionPreflight(
        authorization=claim.authorization,
        source_commit=source_commit,
        source_tree=source_tree,
        source_seal=source_seal,
        active=active,
        candidate=candidate,
        active_summary=backup_summary,
        candidate_summary=candidate_summary,
        systemd_unit_sha256=unit.sha256,
        environment_sha256=environment.sha256,
        candidate_promotion_receipt_sha256=receipt_sha256,
        aq4_source_root=Path(aq4_source["root"]),
        aq4_source_commit=aq4_source["commit"],
        aq4_source_tree=aq4_source["tree"],
        aq4_source_seal=None,
        aq4_worker_binary=aq4_runtime.worker.snapshot,
        aq4_promotion_receipt=aq4_runtime.promotion_receipt.snapshot,
        aq4_promotion_evidence=None,
        api_key_sha256=api_key_sha256,
        openwebui_session_token_sha256=session_token_sha256,
        runtime_artifact_seals=runtime_artifact_seals,
        runtime_tree_seals=runtime_tree_seals,
        aq4_runtime_artifact_seals=aq4_runtime_artifact_seals,
        aq4_runtime_tree_seals=aq4_runtime_tree_seals,
        shared_runtime_artifact_seals=shared_runtime_artifact_seals,
        shared_runtime_tree_seals=shared_runtime_tree_seals,
    )
    transaction._require_runtime_seals(
        pseudo,
        required_uid=policy.required_uid,
        scope="aq4",
    )
    return RecoveryPreflight(
        claim,
        source_commit,
        source_tree,
        active,
        active_state,
        candidate,
        backup,
        candidate_summary,
        backup_summary,
        unit.sha256,
        environment.sha256,
        receipt_sha256,
        pseudo,
        backup_requires_publication,
    )


def _repin(
    request: transaction.TransactionRequest,
    pinned: RecoveryPreflight,
    *,
    now: datetime,
    policy: authorization.RegistryPolicy,
    runner: transaction.CommandRunner,
) -> None:
    if (
        not pinned.transaction_preflight.runtime_artifact_seals
        or not pinned.transaction_preflight.runtime_tree_seals
    ):
        raise RecoveryError("recovery runtime seals are unavailable")
    transaction._repin_transaction_inputs(
        request,
        pinned.claim,
        pinned.transaction_preflight,
        policy=policy,
        runner=runner,
        now=now,
        repin_aq4_release=False,
        runtime_scope="aq4",
    )
    backup = transaction._read_input(
        pinned.backup.path,
        "authorized immutable AQ4 backup",
        transaction.MAX_MANIFEST_BYTES,
    )
    active = transaction._read_input(
        pinned.active_before.path,
        "recovery active served-model manifest",
        transaction.MAX_MANIFEST_BYTES,
    )
    if (
        backup.raw != pinned.backup.raw
        or backup.identity.uid != policy.required_uid
        or backup.identity.links != 1
        or stat.S_IMODE(backup.identity.mode) != 0o444
        or active.sha256
        not in {
            pinned.active_before.sha256,
            pinned.claim.authorization.document["before"]["manifest_sha256"],
        }
    ):
        raise RecoveryError("recovery input changed while locked")


def _materialize_missing_backup(
    pinned: RecoveryPreflight,
    *,
    policy: authorization.RegistryPolicy,
) -> RecoveryPreflight:
    if not pinned.backup_requires_publication:
        return pinned
    if pinned.active_state != "aq4":
        raise RecoveryError("only exact AQ4 can bootstrap a missing backup")
    transaction._exclusive_publish(
        pinned.backup.path,
        pinned.active_before.raw,
        mode=0o444,
        required_uid=policy.required_uid,
    )
    backup = transaction._read_input(
        pinned.backup.path,
        "authorized immutable AQ4 backup",
        transaction.MAX_MANIFEST_BYTES,
    )
    if (
        backup.raw != pinned.active_before.raw
        or backup.identity.uid != policy.required_uid
        or backup.identity.links != 1
        or stat.S_IMODE(backup.identity.mode) != 0o444
    ):
        raise RecoveryError("bootstrapped AQ4 backup identity differs")
    backup_seal = transaction._capture_runtime_artifact(
        backup.path,
        label="authorized immutable AQ4 backup",
        maximum=transaction.MAX_MANIFEST_BYTES,
        required_uid=policy.required_uid,
    )
    if backup_seal.snapshot != backup:
        raise RecoveryError("bootstrapped AQ4 backup seal differs")
    aq4_runtime_artifact_seals = (
        *pinned.transaction_preflight.aq4_runtime_artifact_seals,
        backup_seal,
    )
    transaction_preflight = replace(
        pinned.transaction_preflight,
        runtime_artifact_seals=(
            *aq4_runtime_artifact_seals,
            *pinned.transaction_preflight.shared_runtime_artifact_seals,
        ),
        aq4_runtime_artifact_seals=aq4_runtime_artifact_seals,
    )
    return replace(
        pinned,
        backup=backup,
        transaction_preflight=transaction_preflight,
        backup_requires_publication=False,
    )


def default_recovery_probe(
    request: transaction.TransactionRequest,
    claim: authorization.ClaimRecord,
    preflight_result: transaction.TransactionPreflight,
) -> dict[str, Any]:
    return transaction.default_restoration_probe(
        request,
        claim,
        preflight_result,
    )


def recover_transaction(
    request: transaction.TransactionRequest,
    *,
    policy: authorization.RegistryPolicy = authorization.RegistryPolicy(),
    validator: transaction.ManifestValidator = (
        transaction.default_manifest_validator
    ),
    runner: transaction.CommandRunner = subprocess.run,
    clock: transaction.Clock = transaction.utc_now,
    restoration_probe: RecoveryProbe = default_recovery_probe,
) -> RecoveryResult:
    """Restore AQ4 under the activation lock and publish one recovery receipt."""

    started_at = clock()
    pinned: RecoveryPreflight | None = None
    slot: transaction.ActiveSlot | None = None
    failure_stage: str | None = None
    primary_error: BaseException | None = None
    restoration: dict[str, Any] = {}
    receipt: authorization.FileSnapshot | None = None
    post_receipt_interrupt: transaction.TransactionInterrupted | None = None
    ownership_lost = False
    with transaction._termination_guard() as termination, ExitStack() as resources:
        try:
            try:
                slot = transaction.ActiveSlot.acquire(
                    request.active_manifest,
                    required_uid=policy.required_uid,
                )
                resources.callback(slot.close)
                selected = preflight_recovery(
                    request,
                    now=clock(),
                    policy=policy,
                    validator=validator,
                    runner=runner,
                )
                with termination.deferred():
                    selected = _materialize_missing_backup(
                        selected,
                        policy=policy,
                    )
                pinned = selected
                if slot.path != pinned.active_before.path:
                    raise RecoveryError("locked recovery active path differs")
                pending_signum = termination.take_pending()
                if pending_signum is not None:
                    post_receipt_interrupt = (
                        transaction.TransactionInterrupted(
                            f"termination signal {pending_signum}"
                        )
                    )
            except BaseException as error:
                failure_stage = "preflight"
                primary_error = error
                raise

            restoration = {
                "expected_manifest_sha256": pinned.backup.sha256,
                "displaced_manifest_sha256": pinned.active_before.sha256,
                "observed_manifest_sha256": None,
                "bytes_equal": False,
                "reverse_reconciliation_passed": False,
                "final_checks_passed": False,
                "model_id": None,
                "format_id": None,
                "worker_binary_sha256": None,
                "proof": None,
            }
            deferred_interrupt: transaction.TransactionInterrupted | None = None
            try:
                last_error: BaseException | None = None
                restore_expected = pinned.active_before
                with termination.deferred():
                    for _attempt in range(2):
                        try:
                            _repin(
                                request,
                                pinned,
                                now=clock(),
                                policy=policy,
                                runner=runner,
                            )
                            restore_current = slot.snapshot_current()
                            if restore_current != restore_expected:
                                raise transaction.ActiveSlotOwnershipLost(
                                    "active manifest changed before locked recovery",
                                    displaced_sha256=restore_current.sha256,
                                )
                            restore_expected = slot.replace(
                                pinned.backup.raw,
                                pinned.active_before.identity,
                                expected_current=restore_current,
                            )
                            restored = transaction._read_input(
                                pinned.active_before.path,
                                "recovered active served-model manifest",
                                transaction.MAX_MANIFEST_BYTES,
                            )
                            if (
                                restored != restore_expected
                                or restored.raw != pinned.backup.raw
                            ):
                                raise RecoveryError(
                                    "recovered active manifest identity differs"
                                )
                            last_error = None
                            break
                        except transaction.ActiveSlotOwnershipLost as error:
                            ownership_lost = True
                            if error.displaced_sha256 is not None:
                                restoration[
                                    "displaced_manifest_sha256"
                                ] = error.displaced_sha256
                            last_error = error
                            break
                        except BaseException as error:
                            last_error = error
                    if last_error is not None:
                        raise last_error
            except BaseException as error:
                failure_stage = "aq4_restore"
                primary_error = error
            pending_signum = termination.take_pending()
            if pending_signum is not None:
                deferred_interrupt = transaction.TransactionInterrupted(
                    f"termination signal {pending_signum}"
                )

            if failure_stage is None and not ownership_lost:
                final_commands_passed = False
                try:
                    _repin(
                        request,
                        pinned,
                        now=clock(),
                        policy=policy,
                        runner=runner,
                    )
                    transaction._run_commands(
                        request.commands.reverse_reconciliation,
                        request=request,
                        claim=pinned.claim,
                        preflight_result=pinned.transaction_preflight,
                        stage="reverse_reconciliation",
                        runner=runner,
                        runtime_scope="aq4",
                    )
                    _repin(
                        request,
                        pinned,
                        now=clock(),
                        policy=policy,
                        runner=runner,
                    )
                    restoration["reverse_reconciliation_passed"] = True
                except BaseException as error:
                    failure_stage = "reverse_reconciliation"
                    primary_error = error
                try:
                    _repin(
                        request,
                        pinned,
                        now=clock(),
                        policy=policy,
                        runner=runner,
                    )
                    transaction._run_commands(
                        request.commands.final_checks,
                        request=request,
                        claim=pinned.claim,
                        preflight_result=pinned.transaction_preflight,
                        stage="final_checks",
                        runner=runner,
                        runtime_scope="aq4",
                    )
                    _repin(
                        request,
                        pinned,
                        now=clock(),
                        policy=policy,
                        runner=runner,
                    )
                    final_commands_passed = True
                except BaseException as error:
                    failure_stage = "final_checks"
                    if primary_error is None:
                        primary_error = error
                try:
                    restored_snapshot = transaction._read_input(
                        pinned.active_before.path,
                        "recovered active served-model manifest",
                        transaction.MAX_MANIFEST_BYTES,
                    )
                    transaction_preflight = pinned.transaction_preflight
                    proof_preflight = transaction.TransactionPreflight(
                        authorization=pinned.claim.authorization,
                        source_commit=pinned.source_commit,
                        source_tree=pinned.source_tree,
                        source_seal=transaction_preflight.source_seal,
                        active=restored_snapshot,
                        candidate=pinned.candidate,
                        active_summary=pinned.backup_summary,
                        candidate_summary=pinned.candidate_summary,
                        systemd_unit_sha256=pinned.systemd_unit_sha256,
                        environment_sha256=pinned.environment_sha256,
                        candidate_promotion_receipt_sha256=(
                            pinned.candidate_promotion_receipt_sha256
                        ),
                        aq4_source_root=transaction_preflight.aq4_source_root,
                        aq4_source_commit=transaction_preflight.aq4_source_commit,
                        aq4_source_tree=transaction_preflight.aq4_source_tree,
                        aq4_source_seal=None,
                        aq4_worker_binary=None,
                        aq4_promotion_receipt=None,
                        aq4_promotion_evidence=None,
                        api_key_sha256=transaction_preflight.api_key_sha256,
                        openwebui_session_token_sha256=(
                            transaction_preflight.openwebui_session_token_sha256
                        ),
                        runtime_artifact_seals=(
                            transaction_preflight.runtime_artifact_seals
                        ),
                        runtime_tree_seals=(
                            transaction_preflight.runtime_tree_seals
                        ),
                        candidate_runtime_artifact_seals=(
                            transaction_preflight
                            .candidate_runtime_artifact_seals
                        ),
                        candidate_runtime_tree_seals=(
                            transaction_preflight.candidate_runtime_tree_seals
                        ),
                        aq4_runtime_artifact_seals=(
                            transaction_preflight.aq4_runtime_artifact_seals
                        ),
                        aq4_runtime_tree_seals=(
                            transaction_preflight.aq4_runtime_tree_seals
                        ),
                        shared_runtime_artifact_seals=(
                            transaction_preflight
                            .shared_runtime_artifact_seals
                        ),
                        shared_runtime_tree_seals=(
                            transaction_preflight.shared_runtime_tree_seals
                        ),
                    )
                    proof = restoration_probe(
                        request,
                        pinned.claim,
                        proof_preflight,
                    )
                    restoration_proof.validate_proof(
                        proof,
                        authorization_sha256=(
                            pinned.claim.authorization.snapshot.sha256
                        ),
                        claim_sha256=pinned.claim.snapshot.sha256,
                        active_manifest_path=pinned.active_before.path,
                        expected_manifest_sha256=pinned.backup.sha256,
                        expected_worker_sha256=pinned.claim.authorization.document[
                            "before"
                        ]["worker_binary_sha256"],
                        service_unit=request.service_unit,
                    )
                    _repin(
                        request,
                        pinned,
                        now=clock(),
                        policy=policy,
                        runner=runner,
                    )
                    restoration.update(
                        observed_manifest_sha256=pinned.backup.sha256,
                        bytes_equal=True,
                        final_checks_passed=True,
                        model_id="ullm-qwen3.5-9b-aq4",
                        format_id="AQ4_0",
                        worker_binary_sha256=proof["worker"][
                            "executable_sha256"
                        ],
                        proof=proof,
                    )
                    if not final_commands_passed:
                        restoration["final_checks_passed"] = False
                except BaseException as error:
                    failure_stage = "final_checks"
                    if primary_error is None:
                        primary_error = error
                if deferred_interrupt is not None:
                    post_receipt_interrupt = deferred_interrupt
        except BaseException:
            pass
        if pinned is None:
            assert primary_error is not None
            raise RecoveryError("campaign recovery preflight failed") from primary_error

        status = "restored" if failure_stage is None else "failed_restore"
        document = {
            "schema_version": authorization.RECOVERY_SCHEMA,
            "authorization_id": pinned.claim.authorization.document[
                "authorization_id"
            ],
            "authorization_path": os.fspath(
                pinned.claim.authorization.snapshot.path
            ),
            "authorization_sha256": (
                pinned.claim.authorization.snapshot.sha256
            ),
            "claim_path": os.fspath(pinned.claim.snapshot.path),
            "claim_sha256": pinned.claim.snapshot.sha256,
            "started_at": authorization.utc_timestamp(started_at),
            "completed_at": authorization.utc_timestamp(clock()),
            "status": status,
            "failure_stage": failure_stage,
            "source": pinned.claim.authorization.document["source"],
            "active_before": {
                "path": os.fspath(pinned.active_before.path),
                "sha256": pinned.active_before.sha256,
                "state": pinned.active_state,
            },
            "backup": {
                "path": os.fspath(pinned.backup.path),
                "sha256": pinned.backup.sha256,
            },
            "restoration": restoration,
        }
        try:
            with termination.deferred():
                receipt = authorization.publish_recovery(
                    pinned.claim,
                    document,
                    policy=policy,
                )
        except authorization.AuthorizationError as error:
            raise RecoveryError(
                "campaign recovery receipt publication failed"
            ) from error
        assert receipt is not None
        result = RecoveryResult(
            receipt.path,
            receipt.sha256,
            status,
            failure_stage,
        )
        pending_signum = termination.take_pending()
        if pending_signum is not None:
            post_receipt_interrupt = transaction.TransactionInterrupted(
                "termination signal received after durable recovery receipt "
                f"{result.receipt_path} ({result.receipt_sha256})"
            )
        if post_receipt_interrupt is not None:
            raise post_receipt_interrupt
        if status != "restored":
            raise RecoveryFailed(
                "campaign recovery ended as failed_restore",
                result=result,
            ) from primary_error
        return result
