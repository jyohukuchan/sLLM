#!/usr/bin/env python3
"""Locked AQ4-to-SQ8 campaign transaction with unconditional AQ4 restoration."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator


TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
if os.fspath(TOOLS) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS))

import served_model_campaign_authorization as authorization  # noqa: E402
import served_model_aq4_restoration_proof as restoration_proof  # noqa: E402
from served_model_active_binding import (  # noqa: E402
    FileIdentity,
    MAX_MANIFEST_BYTES,
    StableFileSnapshot,
    stable_read_regular,
)


VALIDATOR_PATH = TOOLS / "validate-served-model.py"
VALIDATOR_MODULE = "_ullm_campaign_transaction_served_model_validator"
MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_FILE_BYTES = 256 * 1024 * 1024
MAX_OUTPUT_FILES = 16_384
MAX_OUTPUT_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
MAX_COMMANDS = 64
MAX_ARGUMENTS = 128
MAX_ARGUMENT_BYTES = 65_536
MAX_COMMAND_TIMEOUT_SECONDS = 3_600.0
COMMAND_TERMINATION_GRACE_SECONDS = 2.0
GIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SELECTED_ARTIFACTS = {
    "SHA256SUMS",
    "active-manifest-binding.json",
    "browser-validator.json",
    "candidate-served-model.json",
    "model-identity.json",
    "release-validation.json",
    "summary.json",
    "validation.json",
    "active-manifest-binding.json",
    "active-manifest-observations.jsonl",
    "browser-evidence.json",
    "lifecycle.json",
    "resource-samples.jsonl",
}
SQ8_FULL_V2_ROOT_FILES = frozenset(
    {
        "environment.json",
        "model-identity.json",
        "raw-session-results.jsonl",
        "soak-resources.raw.jsonl",
        "service-journal.raw.jsonl",
        "amd-smi-metric-normal-before.json",
        "amd-smi-metric-normal-after.json",
        "amd-smi-metric-restart-before.json",
        "amd-smi-metric-restart-after.json",
        "sampling-results.json",
        "cancel-results.json",
        "prefill-latency-results.json",
        "api-contract-results.json",
        "openwebui-smoke.json",
        "soak-results.json",
        "release-matrix.json",
        "summary.md",
        "SHA256SUMS",
        "candidate-served-model.json",
        "active-manifest-observations.jsonl",
        "release-validation.json",
    }
)
SQ8_FULL_V2_FILES = SQ8_FULL_V2_ROOT_FILES | frozenset(
    {
        "browser/openwebui-stop-before.png",
        "browser/post-header-failure.png",
    }
)
REASONING_RELEASE_V2_FILES = frozenset(
    {
        "cases.json",
        "lifecycle.json",
        "resource-samples.jsonl",
        "summary.json",
        "candidate-served-model.json",
        "active-manifest-observations.jsonl",
        "active-manifest-binding.json",
    }
)
REASONING_BROWSER_V2_FILES = frozenset(
    {
        "browser-evidence.json",
        "candidate-served-model.json",
        "active-manifest-observations.jsonl",
        "active-manifest-binding.json",
    }
)
CAMPAIGN_OUTPUT_LAYOUTS = {
    "sq8_full": {
        "files": SQ8_FULL_V2_FILES,
        "directories": frozenset({"browser"}),
        "directory_mode": 0o700,
        "file_mode": 0o600,
    },
    "reasoning_release": {
        "files": REASONING_RELEASE_V2_FILES,
        "directories": frozenset(),
        "directory_mode": 0o555,
        "file_mode": 0o444,
    },
    "reasoning_browser": {
        "files": REASONING_BROWSER_V2_FILES,
        "directories": frozenset(),
        "directory_mode": 0o555,
        "file_mode": 0o444,
    },
}


class TransactionError(RuntimeError):
    """The temporary cross-model campaign transaction failed closed."""


class TransactionInterrupted(TransactionError):
    """A termination signal requested transactional restoration."""


class CandidateWindowExpired(TransactionError):
    """The authorization deadline ended while SQ8_0 was active."""


class CommandContainmentLost(TransactionError):
    """A command supervisor could not prove all descendants absent."""


class ActiveSlotOwnershipLost(TransactionError):
    """A non-cooperating writer won an active-manifest exchange boundary."""

    def __init__(
        self,
        message: str,
        *,
        displaced_sha256: str | None = None,
        exchange_rolled_back: bool = False,
    ) -> None:
        super().__init__(message)
        self.displaced_sha256 = displaced_sha256
        self.exchange_rolled_back = exchange_rolled_back


class TransactionFailed(TransactionError):
    """A durable non-success outcome was published."""

    def __init__(
        self,
        message: str,
        *,
        result: "TransactionResult",
        backup_path: Path,
        restoration: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.result = result
        self.backup_path = backup_path
        self.restoration = restoration


@dataclass(frozen=True, slots=True)
class TransactionCommands:
    candidate_reconciliation: tuple[tuple[str, ...], ...]
    candidate_checks: tuple[tuple[str, ...], ...]
    sq8_full: tuple[str, ...]
    reasoning_release: tuple[str, ...]
    reasoning_browser: tuple[str, ...]
    reverse_reconciliation: tuple[tuple[str, ...], ...]
    final_checks: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class TransactionRequest:
    authorization_path: Path
    source_root: Path
    candidate_manifest: Path
    active_manifest: Path
    systemd_unit: Path
    environment_file: Path
    inactive_services: tuple[str, ...]
    commands: TransactionCommands
    command_timeout_seconds: float = 1_800.0
    service_unit: str = "ullm-openai.service"
    api_key_file: Path | None = None
    openwebui_session_token_file: Path | None = None


@dataclass(frozen=True, slots=True)
class TransactionPreflight:
    authorization: authorization.AuthorizationRecord
    source_commit: str
    source_tree: str
    active: StableFileSnapshot
    candidate: StableFileSnapshot
    active_summary: dict[str, Any]
    candidate_summary: dict[str, Any]
    systemd_unit_sha256: str
    environment_sha256: str
    candidate_promotion_receipt_sha256: str
    api_key_sha256: str | None
    openwebui_session_token_sha256: str | None


@dataclass(frozen=True, slots=True)
class TransactionResult:
    outcome_path: Path
    outcome_sha256: str
    status: str


CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]
ManifestValidator = Callable[[Path], dict[str, Any]]
InactiveChecker = Callable[[Sequence[str]], None]
Clock = Callable[[], datetime]
RestorationProbe = Callable[
    [
        TransactionRequest,
        authorization.ClaimRecord,
        TransactionPreflight,
    ],
    dict[str, Any],
]


def _load_validator() -> ModuleType:
    existing = sys.modules.get(VALIDATOR_MODULE)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(VALIDATOR_MODULE, VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise TransactionError("served-model validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[VALIDATOR_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(VALIDATOR_MODULE, None)
        raise
    return module


def default_manifest_validator(path: Path) -> dict[str, Any]:
    try:
        return _load_validator().validation_summary(path)
    except Exception as error:
        raise TransactionError("served-model validation failed") from error


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
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
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise TransactionError("transaction evidence is not canonicalizable") from error


def _validate_commands(commands: TransactionCommands) -> None:
    groups: tuple[Sequence[Sequence[str]], ...] = (
        commands.candidate_reconciliation,
        commands.candidate_checks,
        (commands.sq8_full,),
        (commands.reasoning_release,),
        (commands.reasoning_browser,),
        commands.reverse_reconciliation,
        commands.final_checks,
    )
    if any(not group or len(group) > MAX_COMMANDS for group in groups):
        raise TransactionError("every transaction stage requires bounded commands")
    for group in groups:
        for command in group:
            if (
                not command
                or len(command) > MAX_ARGUMENTS
                or any(
                    not isinstance(argument, str)
                    or not argument
                    or "\x00" in argument
                    or len(argument.encode("utf-8")) > MAX_ARGUMENT_BYTES
                    for argument in command
                )
            ):
                raise TransactionError("transaction command is invalid")


def _run_git(
    source_root: Path,
    arguments: Sequence[str],
    *,
    runner: CommandRunner,
) -> str:
    try:
        completed = runner(
            ["git", *arguments],
            cwd=source_root,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise TransactionError("source identity check failed") from error
    if completed.returncode != 0:
        raise TransactionError("source identity check failed")
    value = completed.stdout.strip()
    if len(value.encode("utf-8")) > 1_048_576:
        raise TransactionError("source identity output is oversized")
    return value


def _source_identity(
    source_root: Path,
    *,
    runner: CommandRunner,
) -> tuple[str, str]:
    try:
        root = source_root.resolve(strict=True)
    except OSError as error:
        raise TransactionError("source root is unavailable") from error
    top = Path(
        _run_git(root, ("rev-parse", "--show-toplevel"), runner=runner)
    ).resolve(strict=True)
    if top != root:
        raise TransactionError("source root differs from Git top-level")
    commit = _run_git(root, ("rev-parse", "HEAD"), runner=runner)
    tree = _run_git(root, ("rev-parse", "HEAD^{tree}"), runner=runner)
    if GIT_RE.fullmatch(commit) is None or GIT_RE.fullmatch(tree) is None:
        raise TransactionError("source identity is not a full Git object ID")
    status = _run_git(
        root,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        runner=runner,
    )
    if status:
        raise TransactionError("campaign source worktree is not clean")
    return commit, tree


def _strict_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TransactionError(f"{label} is not JSON") from error
    if not isinstance(value, dict):
        raise TransactionError(f"{label} root is not an object")
    return value


def _summary_identity(
    summary: dict[str, Any],
    *,
    model_id: str,
    format_id: str,
    manifest_sha256: str,
    worker_protocol: str,
    label: str,
) -> str:
    worker = summary.get("worker")
    worker_hash = worker.get("binary_sha256") if isinstance(worker, dict) else None
    if (
        summary.get("validated") is not True
        or summary.get("manifest_sha256") != manifest_sha256
        or summary.get("model_id") != model_id
        or summary.get("format_id") != format_id
        or not isinstance(worker, dict)
        or worker.get("protocol") != worker_protocol
        or not isinstance(worker_hash, str)
        or authorization.HASH_RE.fullmatch(worker_hash) is None
    ):
        raise TransactionError(f"{label} served-model identity differs")
    return worker_hash


def _promotion_identity(
    document: dict[str, Any],
    *,
    manifest_parent: Path,
    source_commit: str,
    label: str,
) -> tuple[Path, str]:
    promotion = document.get("promotion")
    if not isinstance(promotion, dict) or set(promotion) != {
        "source_commit",
        "receipt",
        "receipt_sha256",
    }:
        raise TransactionError(f"{label} promotion identity differs")
    receipt = promotion["receipt"]
    receipt_sha256 = promotion["receipt_sha256"]
    if (
        promotion["source_commit"] != source_commit
        or not isinstance(receipt, str)
        or not isinstance(receipt_sha256, str)
        or authorization.HASH_RE.fullmatch(receipt_sha256) is None
    ):
        raise TransactionError(f"{label} promotion identity differs")
    receipt_path = Path(receipt)
    if not receipt_path.is_absolute():
        receipt_path = manifest_parent / receipt_path
    return receipt_path, receipt_sha256


def _read_input(path: Path, label: str, maximum: int) -> StableFileSnapshot:
    try:
        return stable_read_regular(path, label, maximum=maximum)
    except Exception as error:
        raise TransactionError(f"{label} is unavailable or changed") from error


def _validate_private_secret(
    path: Path,
    label: str,
    *,
    required_uid: int,
) -> str:
    snapshot = _read_input(path, label, 65_536)
    if (
        snapshot.identity.uid != required_uid
        or snapshot.identity.links != 1
        or stat.S_IMODE(snapshot.identity.mode) & 0o077
        or not snapshot.raw.rstrip(b"\r\n")
    ):
        raise TransactionError(f"{label} metadata is unsafe")
    return snapshot.sha256


def preflight(
    request: TransactionRequest,
    *,
    now: datetime | None = None,
    policy: authorization.RegistryPolicy = authorization.RegistryPolicy(),
    validator: ManifestValidator = default_manifest_validator,
    runner: CommandRunner = subprocess.run,
    require_fresh_outputs: bool = True,
    claimed: authorization.ClaimRecord | None = None,
) -> TransactionPreflight:
    """Perform a read-only exact-identity preflight."""

    _validate_commands(request.commands)
    if (
        not math.isfinite(request.command_timeout_seconds)
        or request.command_timeout_seconds <= 0
        or request.command_timeout_seconds > MAX_COMMAND_TIMEOUT_SECONDS
        or not request.inactive_services
        or len(set(request.inactive_services)) != len(request.inactive_services)
        or any(
            not isinstance(service, str)
            or not service
            or "\x00" in service
            or len(service.encode("utf-8")) > 512
            for service in request.inactive_services
        )
        or not isinstance(request.service_unit, str)
        or request.service_unit not in request.inactive_services
        or _lexical_absolute(
            request.active_manifest,
            "active manifest",
        )
        != _lexical_absolute(
            policy.active_manifest_path,
            "policy active manifest",
        )
        or _lexical_absolute(
            request.systemd_unit,
            "systemd unit",
        )
        != _lexical_absolute(
            policy.systemd_unit_path,
            "policy systemd unit",
        )
        or _lexical_absolute(
            request.environment_file,
            "systemd environment file",
        )
        != _lexical_absolute(
            policy.environment_file_path,
            "policy systemd environment file",
        )
        or request.service_unit != policy.service_unit
    ):
        raise TransactionError("transaction runtime binding is invalid")
    selected_now = utc_now() if now is None else now
    try:
        if claimed is None:
            auth = authorization.load_authorization(
                request.authorization_path,
                now=selected_now,
                policy=policy,
                require_fresh_outputs=require_fresh_outputs,
                source_root=request.source_root,
            )
        else:
            reloaded_claim = authorization.load_claim(
                request.authorization_path,
                now=selected_now,
                policy=policy,
            )
            if (
                reloaded_claim.snapshot.sha256 != claimed.snapshot.sha256
                or reloaded_claim.authorization.snapshot.sha256
                != claimed.authorization.snapshot.sha256
            ):
                raise authorization.AuthorizationError(
                    "campaign claim changed after consumption"
                )
            authorization.validate_authorization_document(
                reloaded_claim.authorization.document,
                now=selected_now,
                required_uid=policy.required_uid,
                require_fresh_outputs=require_fresh_outputs,
                enforce_current_window=False,
                policy=policy,
                source_root=request.source_root,
            )
            auth = reloaded_claim.authorization
    except authorization.AuthorizationError as error:
        raise TransactionError("campaign authorization preflight failed") from error
    source_commit, source_tree = _source_identity(request.source_root, runner=runner)
    active = _read_input(
        request.active_manifest,
        "actual active served-model manifest",
        MAX_MANIFEST_BYTES,
    )
    candidate = _read_input(
        request.candidate_manifest,
        "frozen candidate served-model manifest",
        MAX_MANIFEST_BYTES,
    )
    if (
        active.identity.uid != policy.required_uid
        or active.identity.links != 1
        or stat.S_IMODE(active.identity.mode) != 0o644
    ):
        raise TransactionError("active manifest metadata is unsafe")
    api_key_sha256: str | None = None
    session_token_sha256: str | None = None
    secret_paths = (
        request.api_key_file,
        request.openwebui_session_token_file,
    )
    if any(value is not None for value in secret_paths):
        if any(value is None for value in secret_paths):
            raise TransactionError("transaction private credential binding is incomplete")
        assert request.api_key_file is not None
        assert request.openwebui_session_token_file is not None
        api_key_sha256 = _validate_private_secret(
            request.api_key_file,
            "gateway API key",
            required_uid=policy.required_uid,
        )
        session_token_sha256 = _validate_private_secret(
            request.openwebui_session_token_file,
            "OpenWebUI session token",
            required_uid=policy.required_uid,
        )
    if active.path == candidate.path or active.raw == candidate.raw:
        raise TransactionError("AQ4 and SQ8 manifest inputs are not distinct")
    active_document = _strict_object(active.raw, "active served-model manifest")
    candidate_document = _strict_object(
        candidate.raw, "candidate served-model manifest"
    )
    if (
        active_document.get("schema_version") != "ullm.served_model.v2"
        or candidate_document.get("schema_version") != "ullm.served_model.v2"
    ):
        raise TransactionError("cross-model transaction requires two v2 manifests")
    active_summary = validator(active.path)
    candidate_summary = validator(candidate.path)
    active_worker = _summary_identity(
        active_summary,
        model_id="ullm-qwen3.5-9b-aq4",
        format_id="AQ4_0",
        manifest_sha256=active.sha256,
        worker_protocol="ullm.worker.v2",
        label="active AQ4",
    )
    candidate_worker = _summary_identity(
        candidate_summary,
        model_id="ullm-qwen3-14b-sq8",
        format_id="SQ8_0",
        manifest_sha256=candidate.sha256,
        worker_protocol="ullm.worker.v2",
        label="candidate SQ8",
    )
    active_promotion = active_document.get("promotion")
    if (
        not isinstance(active_promotion, dict)
        or active_promotion.get("source_commit")
        != auth.document["before"]["promotion_source_commit"]
    ):
        raise TransactionError("active AQ4 promotion identity differs")
    receipt_path, receipt_sha256 = _promotion_identity(
        candidate_document,
        manifest_parent=candidate.path.parent,
        source_commit=source_commit,
        label="candidate SQ8",
    )
    receipt = _read_input(receipt_path, "candidate promotion receipt", MAX_INPUT_BYTES)
    if receipt.sha256 != receipt_sha256:
        raise TransactionError("candidate promotion receipt bytes differ")
    unit = _read_input(request.systemd_unit, "systemd unit", MAX_INPUT_BYTES)
    environment = _read_input(
        request.environment_file, "systemd environment file", MAX_INPUT_BYTES
    )
    rollback = auth.document["rollback"]
    if (
        unit.sha256 != rollback["systemd_unit_sha256"]
        or environment.sha256 != rollback["environment_sha256"]
    ):
        raise TransactionError("rollback unit/environment identity differs")
    try:
        authorization.require_authorization_window_binding(
            auth,
            source_commit=source_commit,
            source_tree=source_tree,
            before_manifest_sha256=active.sha256,
            candidate_manifest_sha256=candidate.sha256,
            candidate_worker_binary_sha256=candidate_worker,
            candidate_promotion_receipt_sha256=receipt.sha256,
            rollback_backup_path=Path(rollback["backup_path"]),
        )
    except authorization.AuthorizationError as error:
        raise TransactionError("campaign authorization identity differs") from error
    if active_worker != auth.document["before"]["worker_binary_sha256"]:
        raise TransactionError("active AQ4 worker identity differs")
    return TransactionPreflight(
        auth,
        source_commit,
        source_tree,
        active,
        candidate,
        active_summary,
        candidate_summary,
        unit.sha256,
        environment.sha256,
        receipt.sha256,
        api_key_sha256,
        session_token_sha256,
    )


def default_inactive_checker(services: Sequence[str]) -> None:
    for service in services:
        try:
            completed = subprocess.run(
                [
                    "systemctl",
                    "show",
                    service,
                    "--property=LoadState",
                    "--property=ActiveState",
                    "--value",
                ],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10.0,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise TransactionError("inactive-service precondition failed") from error
        states = completed.stdout.splitlines()
        if (
            completed.returncode != 0
            or len(states) != 2
            or states[0] != "loaded"
            or states[1] not in {"inactive", "failed"}
        ):
            raise TransactionError("candidate switch requires inactive services")


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise TransactionError("safe directory descriptor flags are unavailable")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _lexical_absolute(path: Path, label: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise TransactionError(f"{label} path must be absolute")
    normalized = Path(os.path.abspath(path))
    if normalized != path or path.name in {"", ".", ".."} or ".." in path.parts:
        raise TransactionError(f"{label} path is not lexically canonical")
    return normalized


def _open_parent_descriptor(
    path: Path,
    label: str,
    *,
    required_uid: int,
) -> tuple[int, tuple[int, int]]:
    absolute = _lexical_absolute(path, label)
    descriptor = -1
    try:
        descriptor = os.open(absolute.anchor, _directory_flags())
        for component in absolute.parent.parts[1:]:
            next_descriptor = os.open(
                component,
                _directory_flags(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != required_uid
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise TransactionError(f"{label} parent metadata is unsafe")
        return descriptor, (metadata.st_dev, metadata.st_ino)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _verify_parent_descriptor(
    path: Path,
    expected: tuple[int, int],
    *,
    required_uid: int,
    label: str,
) -> None:
    descriptor, identity = _open_parent_descriptor(
        path, label, required_uid=required_uid
    )
    os.close(descriptor)
    if identity != expected:
        raise TransactionError(f"{label} parent changed")


RENAME_EXCHANGE = 2


def _rename_exchange(
    source_name: str,
    destination_name: str,
    *,
    parent_descriptor: int,
) -> None:
    """Atomically exchange two names in one pinned directory."""

    if (
        not isinstance(source_name, str)
        or not isinstance(destination_name, str)
        or not source_name
        or not destination_name
        or "/" in source_name
        or "/" in destination_name
        or "\x00" in source_name
        or "\x00" in destination_name
    ):
        raise TransactionError("active manifest exchange names are invalid")
    try:
        function = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as error:
        raise TransactionError(
            "renameat2(RENAME_EXCHANGE) is unavailable"
        ) from error
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = function(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(destination_name),
        RENAME_EXCHANGE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
            message = "renameat2(RENAME_EXCHANGE) is unsupported"
        else:
            message = "active manifest atomic exchange failed"
        raise TransactionError(message) from OSError(
            error_number,
            os.strerror(error_number),
        )


def _identity_after_exchange_matches(
    observed: FileIdentity,
    expected: FileIdentity,
) -> bool:
    """Match an inode across rename exchange, which legitimately advances ctime."""

    return (
        observed.device == expected.device
        and observed.inode == expected.inode
        and observed.mode == expected.mode
        and observed.links == expected.links
        and observed.uid == expected.uid
        and observed.gid == expected.gid
        and observed.size == expected.size
        and observed.mtime_ns == expected.mtime_ns
        and observed.ctime_ns >= expected.ctime_ns
    )


def _snapshot_after_exchange_matches(
    observed: StableFileSnapshot,
    expected: StableFileSnapshot,
) -> bool:
    return (
        observed.raw == expected.raw
        and _identity_after_exchange_matches(
            observed.identity,
            expected.identity,
        )
    )


@dataclass(slots=True)
class ActiveSlot:
    path: Path
    parent_descriptor: int
    parent_identity: tuple[int, int]
    lock_descriptor: int
    required_uid: int

    @classmethod
    def acquire(cls, active: Path, *, required_uid: int) -> "ActiveSlot":
        absolute = _lexical_absolute(active, "active manifest")
        parent, identity = _open_parent_descriptor(
            absolute,
            "active manifest",
            required_uid=required_uid,
        )
        lock_descriptor = -1
        try:
            flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
            lock_descriptor = os.open(
                f".{absolute.name}.activation.lock",
                flags,
                0o600,
                dir_fd=parent,
            )
            metadata = os.fstat(lock_descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
                or metadata.st_uid != required_uid
            ):
                raise TransactionError("activation lock metadata is unsafe")
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise TransactionError("another activation is in progress") from error
            return cls(
                absolute,
                parent,
                identity,
                lock_descriptor,
                required_uid,
            )
        except BaseException:
            if lock_descriptor >= 0:
                os.close(lock_descriptor)
            os.close(parent)
            raise

    def _read_named(
        self,
        name: str,
        label: str,
    ) -> tuple[bytes, os.stat_result]:
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\x00" in name
        ):
            raise TransactionError(f"{label} name is invalid")
        descriptor = -1
        try:
            entry_before = os.stat(
                name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=self.parent_descriptor,
            )
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_size <= 0
                or before.st_size > MAX_MANIFEST_BYTES
                or (
                    entry_before.st_dev,
                    entry_before.st_ino,
                    entry_before.st_mode,
                    entry_before.st_uid,
                    entry_before.st_gid,
                    entry_before.st_nlink,
                    entry_before.st_size,
                    entry_before.st_mtime_ns,
                    entry_before.st_ctime_ns,
                )
                != (
                    before.st_dev,
                    before.st_ino,
                    before.st_mode,
                    before.st_uid,
                    before.st_gid,
                    before.st_nlink,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
            ):
                raise TransactionError(f"{label} current identity is unsafe")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(
                    descriptor,
                    min(65_536, MAX_MANIFEST_BYTES - total + 1),
                )
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_MANIFEST_BYTES:
                    raise TransactionError(f"{label} exceeds its byte bound")
                chunks.append(chunk)
            after = os.fstat(descriptor)
            entry_after = os.stat(
                name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
            if (
                (
                    before.st_dev,
                    before.st_ino,
                    before.st_mode,
                    before.st_uid,
                    before.st_gid,
                    before.st_nlink,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                != (
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_uid,
                    after.st_gid,
                    after.st_nlink,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
                or (
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_uid,
                    after.st_gid,
                    after.st_nlink,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
                != (
                    entry_after.st_dev,
                    entry_after.st_ino,
                    entry_after.st_mode,
                    entry_after.st_uid,
                    entry_after.st_gid,
                    entry_after.st_nlink,
                    entry_after.st_size,
                    entry_after.st_mtime_ns,
                    entry_after.st_ctime_ns,
                )
                or total != before.st_size
            ):
                raise TransactionError(f"{label} changed at replace boundary")
            return b"".join(chunks), after
        except TransactionError:
            raise
        except OSError as error:
            raise TransactionError(
                f"{label} is unavailable at replace boundary"
            ) from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _snapshot_named(
        self,
        name: str,
        label: str,
    ) -> StableFileSnapshot:
        raw, metadata = self._read_named(name, label)
        path = self.path if name == self.path.name else self.path.with_name(name)
        return StableFileSnapshot(
            path,
            raw,
            _sha256(raw),
            FileIdentity.from_stat(metadata),
        )

    def snapshot_current(self) -> StableFileSnapshot:
        return self._snapshot_named(self.path.name, "active manifest")

    def _read_current(self) -> tuple[bytes, os.stat_result]:
        return self._read_named(self.path.name, "active manifest")

    def _rollback_exchange_if_owned(
        self,
        temporary_name: str,
        *,
        staged: StableFileSnapshot,
        displaced: StableFileSnapshot,
    ) -> None:
        try:
            active_now = self.snapshot_current()
            displaced_now = self._snapshot_named(
                temporary_name,
                "displaced active manifest",
            )
        except BaseException as error:
            raise ActiveSlotOwnershipLost(
                "active manifest ownership was lost after atomic exchange",
                displaced_sha256=displaced.sha256,
            ) from error
        if (
            not _snapshot_after_exchange_matches(active_now, staged)
            or displaced_now != displaced
        ):
            raise ActiveSlotOwnershipLost(
                "active manifest ownership was lost after atomic exchange",
                displaced_sha256=displaced.sha256,
            )
        try:
            _rename_exchange(
                temporary_name,
                self.path.name,
                parent_descriptor=self.parent_descriptor,
            )
            os.fsync(self.parent_descriptor)
            restored = self.snapshot_current()
            staged_now = self._snapshot_named(
                temporary_name,
                "rolled-back staged active manifest",
            )
            if (
                not _snapshot_after_exchange_matches(restored, displaced)
                or not _snapshot_after_exchange_matches(staged_now, staged)
            ):
                raise ActiveSlotOwnershipLost(
                    "active manifest ownership was lost during exchange rollback",
                    displaced_sha256=displaced.sha256,
                )
            os.unlink(temporary_name, dir_fd=self.parent_descriptor)
            os.fsync(self.parent_descriptor)
        except ActiveSlotOwnershipLost:
            raise
        except BaseException as error:
            raise ActiveSlotOwnershipLost(
                "active manifest exchange could not be rolled back safely",
                displaced_sha256=displaced.sha256,
            ) from error

    def replace(
        self,
        raw: bytes,
        identity: Any,
        *,
        expected_current: StableFileSnapshot,
    ) -> StableFileSnapshot:
        _verify_parent_descriptor(
            self.path,
            self.parent_identity,
            required_uid=self.required_uid,
            label="active manifest",
        )
        temporary_name = (
            f".{self.path.name}.transaction.{secrets.token_hex(16)}.json"
        )
        descriptor = -1
        cleanup_staging = False
        exchanged = False
        displaced: StableFileSnapshot | None = None
        staged: StableFileSnapshot | None = None
        try:
            if (
                not isinstance(expected_current, StableFileSnapshot)
                or expected_current.path != self.path
            ):
                raise TransactionError(
                    "active manifest expected-current boundary is invalid"
                )
            current = self.snapshot_current()
            if current != expected_current:
                raise ActiveSlotOwnershipLost(
                    "active manifest expected-current boundary differs"
                )
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | os.O_NOFOLLOW,
                0o600,
                dir_fd=self.parent_descriptor,
            )
            cleanup_staging = True
            os.fchmod(descriptor, stat.S_IMODE(identity.mode))
            os.fchown(descriptor, identity.uid, identity.gid)
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise TransactionError(
                        "active manifest staging made no progress"
                    )
                view = view[written:]
            os.fsync(descriptor)
            staged_metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(staged_metadata.st_mode)
                or staged_metadata.st_uid != identity.uid
                or staged_metadata.st_gid != identity.gid
                or stat.S_IMODE(staged_metadata.st_mode)
                != stat.S_IMODE(identity.mode)
                or staged_metadata.st_nlink != 1
                or staged_metadata.st_size != len(raw)
            ):
                raise TransactionError(
                    "active manifest staged identity differs"
                )
            staged = StableFileSnapshot(
                self.path.with_name(temporary_name),
                raw,
                _sha256(raw),
                FileIdentity.from_stat(staged_metadata),
            )
            os.close(descriptor)
            descriptor = -1
            current = self.snapshot_current()
            if current != expected_current:
                raise ActiveSlotOwnershipLost(
                    "active manifest changed before atomic replacement"
                )
            _rename_exchange(
                temporary_name,
                self.path.name,
                parent_descriptor=self.parent_descriptor,
            )
            exchanged = True
            cleanup_staging = False
            os.fsync(self.parent_descriptor)
            _verify_parent_descriptor(
                self.path,
                self.parent_identity,
                required_uid=self.required_uid,
                label="active manifest",
            )
            displaced = self._snapshot_named(
                temporary_name,
                "displaced active manifest",
            )
            observed = self.snapshot_current()
            if (
                not _snapshot_after_exchange_matches(
                    displaced,
                    expected_current,
                )
                or not _snapshot_after_exchange_matches(observed, staged)
            ):
                self._rollback_exchange_if_owned(
                    temporary_name,
                    staged=staged,
                    displaced=displaced,
                )
                raise ActiveSlotOwnershipLost(
                    "active manifest exchange displaced an unexpected version",
                    displaced_sha256=displaced.sha256,
                    exchange_rolled_back=True,
                )
            displaced_now = self._snapshot_named(
                temporary_name,
                "displaced active manifest",
            )
            observed = self.snapshot_current()
            if (
                displaced_now != displaced
                or not _snapshot_after_exchange_matches(observed, staged)
            ):
                raise ActiveSlotOwnershipLost(
                    "active manifest ownership changed before exchange commit",
                    displaced_sha256=displaced_now.sha256,
                )
            os.unlink(temporary_name, dir_fd=self.parent_descriptor)
            os.fsync(self.parent_descriptor)
            observed = self.snapshot_current()
            if not _snapshot_after_exchange_matches(observed, staged):
                raise ActiveSlotOwnershipLost(
                    "active manifest ownership changed after exchange commit",
                    displaced_sha256=displaced.sha256,
                )
            return observed
        except ActiveSlotOwnershipLost:
            raise
        except BaseException as error:
            if exchanged and staged is not None:
                if displaced is None:
                    try:
                        displaced = self._snapshot_named(
                            temporary_name,
                            "displaced active manifest",
                        )
                    except BaseException as ownership_error:
                        raise ActiveSlotOwnershipLost(
                            "active manifest ownership was lost after "
                            "an incomplete exchange",
                        ) from ownership_error
                try:
                    self._rollback_exchange_if_owned(
                        temporary_name,
                        staged=staged,
                        displaced=displaced,
                    )
                except ActiveSlotOwnershipLost as ownership_error:
                    raise ownership_error from error
                raise ActiveSlotOwnershipLost(
                    "active manifest exchange failed and was rolled back",
                    displaced_sha256=displaced.sha256,
                    exchange_rolled_back=True,
                ) from error
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if cleanup_staging:
                try:
                    os.unlink(temporary_name, dir_fd=self.parent_descriptor)
                except FileNotFoundError:
                    pass

    def close(self) -> None:
        lock_descriptor, self.lock_descriptor = self.lock_descriptor, -1
        parent_descriptor, self.parent_descriptor = self.parent_descriptor, -1
        if lock_descriptor >= 0:
            os.close(lock_descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def _exclusive_publish(
    path: Path,
    raw: bytes,
    *,
    mode: int,
    required_uid: int,
) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise TransactionError("authorized AQ4 backup path is not fresh")
    parent, parent_identity = _open_parent_descriptor(
        path,
        "authorized AQ4 backup",
        required_uid=required_uid,
    )
    temporary_name = f".{path.name}.{secrets.token_hex(16)}.tmp"
    descriptor = -1
    published = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        os.fchmod(descriptor, mode)
        os.fchown(descriptor, required_uid, os.fstat(descriptor).st_gid)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise TransactionError("authorized AQ4 backup write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=parent,
            dst_dir_fd=parent,
            follow_symlinks=False,
        )
        os.unlink(temporary_name, dir_fd=parent)
        published = True
        os.fsync(parent)
        _verify_parent_descriptor(
            path,
            parent_identity,
            required_uid=required_uid,
            label="authorized AQ4 backup",
        )
    except FileExistsError as error:
        raise TransactionError(
            "authorized AQ4 backup publication is not fresh"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not published:
            try:
                os.unlink(temporary_name, dir_fd=parent)
            except FileNotFoundError:
                pass
        os.close(parent)
    backup = _read_input(path, "authorized AQ4 backup", MAX_MANIFEST_BYTES)
    if (
        backup.raw != raw
        or backup.identity.links != 1
        or stat.S_IMODE(backup.identity.mode) != mode
        or backup.identity.uid != required_uid
    ):
        raise TransactionError("authorized AQ4 backup differs after publication")


def _observe_candidate(
    preflight_result: TransactionPreflight,
    *,
    stage: str,
) -> dict[str, Any]:
    candidate_now = _read_input(
        preflight_result.candidate.path,
        "frozen candidate served-model manifest",
        MAX_MANIFEST_BYTES,
    )
    active_now = _read_input(
        preflight_result.active.path,
        "actual active served-model manifest",
        MAX_MANIFEST_BYTES,
    )
    equal = (
        candidate_now.raw == preflight_result.candidate.raw
        and active_now.raw == preflight_result.candidate.raw
    )
    if not equal:
        raise TransactionError("actual active manifest bytes differ from candidate")
    return {
        "stage": stage,
        "active_manifest_sha256": active_now.sha256,
        "bytes_equal": True,
    }


def _revalidate_pre_switch(
    request: TransactionRequest,
    preflight_result: TransactionPreflight,
) -> None:
    active = _read_input(
        preflight_result.active.path,
        "actual active served-model manifest",
        MAX_MANIFEST_BYTES,
    )
    candidate = _read_input(
        preflight_result.candidate.path,
        "frozen candidate served-model manifest",
        MAX_MANIFEST_BYTES,
    )
    unit = _read_input(request.systemd_unit, "systemd unit", MAX_INPUT_BYTES)
    environment = _read_input(
        request.environment_file, "systemd environment file", MAX_INPUT_BYTES
    )
    candidate_document = _strict_object(
        candidate.raw, "candidate served-model manifest"
    )
    receipt_path, receipt_sha256 = _promotion_identity(
        candidate_document,
        manifest_parent=candidate.path.parent,
        source_commit=preflight_result.source_commit,
        label="candidate SQ8",
    )
    receipt = _read_input(
        receipt_path,
        "candidate promotion receipt",
        MAX_INPUT_BYTES,
    )
    if (
        active.raw != preflight_result.active.raw
        or candidate.raw != preflight_result.candidate.raw
        or unit.sha256 != preflight_result.systemd_unit_sha256
        or environment.sha256 != preflight_result.environment_sha256
        or receipt_sha256 != preflight_result.candidate_promotion_receipt_sha256
        or receipt.sha256 != preflight_result.candidate_promotion_receipt_sha256
    ):
        raise TransactionError("transaction input changed before candidate switch")


def _repin_transaction_inputs(
    request: TransactionRequest,
    claim: authorization.ClaimRecord,
    preflight_result: TransactionPreflight,
    *,
    policy: authorization.RegistryPolicy,
    runner: CommandRunner,
    now: datetime,
) -> None:
    source_commit, source_tree = _source_identity(request.source_root, runner=runner)
    reloaded = authorization.load_claim(
        request.authorization_path,
        now=now,
        policy=policy,
    )
    candidate = _read_input(
        preflight_result.candidate.path,
        "frozen candidate served-model manifest",
        MAX_MANIFEST_BYTES,
    )
    unit = _read_input(request.systemd_unit, "systemd unit", MAX_INPUT_BYTES)
    environment = _read_input(
        request.environment_file, "systemd environment file", MAX_INPUT_BYTES
    )
    candidate_document = _strict_object(
        candidate.raw, "candidate served-model manifest"
    )
    receipt_path, receipt_sha256 = _promotion_identity(
        candidate_document,
        manifest_parent=candidate.path.parent,
        source_commit=preflight_result.source_commit,
        label="candidate SQ8",
    )
    receipt = _read_input(receipt_path, "candidate promotion receipt", MAX_INPUT_BYTES)
    api_key_sha256: str | None = None
    session_token_sha256: str | None = None
    if request.api_key_file is not None:
        api_key_sha256 = _validate_private_secret(
            request.api_key_file,
            "gateway API key",
            required_uid=policy.required_uid,
        )
    if request.openwebui_session_token_file is not None:
        session_token_sha256 = _validate_private_secret(
            request.openwebui_session_token_file,
            "OpenWebUI session token",
            required_uid=policy.required_uid,
        )
    if (
        source_commit != preflight_result.source_commit
        or source_tree != preflight_result.source_tree
        or reloaded.snapshot.sha256 != claim.snapshot.sha256
        or reloaded.authorization.snapshot.sha256
        != claim.authorization.snapshot.sha256
        or candidate.raw != preflight_result.candidate.raw
        or unit.sha256 != preflight_result.systemd_unit_sha256
        or environment.sha256 != preflight_result.environment_sha256
        or receipt_sha256 != preflight_result.candidate_promotion_receipt_sha256
        or receipt.sha256 != preflight_result.candidate_promotion_receipt_sha256
        or api_key_sha256 != preflight_result.api_key_sha256
        or session_token_sha256
        != preflight_result.openwebui_session_token_sha256
    ):
        raise TransactionError("transaction input identity changed during window")


def default_restoration_probe(
    request: TransactionRequest,
    claim: authorization.ClaimRecord,
    preflight_result: TransactionPreflight,
) -> dict[str, Any]:
    if request.api_key_file is None or request.openwebui_session_token_file is None:
        raise TransactionError(
            "live AQ4 restoration proof requires private API and OpenWebUI credentials"
        )
    try:
        return restoration_proof.collect_live_proof(
            authorization_sha256=claim.authorization.snapshot.sha256,
            claim_sha256=claim.snapshot.sha256,
            active_manifest_path=preflight_result.active.path,
            expected_manifest_sha256=preflight_result.active.sha256,
            expected_worker_sha256=claim.authorization.document["before"][
                "worker_binary_sha256"
            ],
            service_unit=request.service_unit,
            api_key_file=request.api_key_file,
            openwebui_session_token_file=request.openwebui_session_token_file,
            manifest_reader=lambda path: _read_input(
                path,
                "live restored active served-model manifest",
                MAX_MANIFEST_BYTES,
            ).raw,
        )
    except restoration_proof.RestorationProofError as error:
        raise TransactionError("live AQ4 restoration proof failed") from error


def _stage_environment(
    request: TransactionRequest,
    claim: authorization.ClaimRecord,
    preflight_result: TransactionPreflight,
    stage: str,
) -> dict[str, str]:
    return {
        **os.environ,
        "ULLM_CAMPAIGN_TRANSACTION_STAGE": stage,
        "ULLM_CAMPAIGN_AUTHORIZATION": os.fspath(
            claim.authorization.snapshot.path
        ),
        "ULLM_CAMPAIGN_CLAIM": os.fspath(claim.snapshot.path),
        "ULLM_CAMPAIGN_AUTHORIZATION_SHA256": (
            claim.authorization.snapshot.sha256
        ),
        "ULLM_CAMPAIGN_CLAIM_SHA256": claim.snapshot.sha256,
        "ULLM_ACTIVE_MANIFEST": os.fspath(preflight_result.active.path),
        "ULLM_CANDIDATE_MANIFEST": os.fspath(preflight_result.candidate.path),
        "ULLM_CANDIDATE_MANIFEST_SHA256": preflight_result.candidate.sha256,
        "ULLM_AQ4_BACKUP_MANIFEST": claim.authorization.document["rollback"][
            "backup_path"
        ],
    }


def _run_commands(
    commands: Sequence[Sequence[str]],
    *,
    request: TransactionRequest,
    claim: authorization.ClaimRecord,
    preflight_result: TransactionPreflight,
    stage: str,
    runner: CommandRunner,
    deadline: datetime | None = None,
    clock: Clock = utc_now,
) -> None:
    environment = _stage_environment(request, claim, preflight_result, stage)
    for command in commands:
        timeout_seconds = request.command_timeout_seconds
        if deadline is not None:
            remaining = (deadline - clock()).total_seconds()
            if not math.isfinite(remaining) or remaining <= 0:
                raise CandidateWindowExpired(
                    "candidate-active authorization deadline expired"
                )
            timeout_seconds = min(timeout_seconds, remaining)
        if runner is subprocess.run:
            _run_owned_process_group(
                command,
                request=request,
                environment=environment,
                stage=stage,
                timeout_seconds=timeout_seconds,
            )
        else:
            try:
                completed = runner(
                    list(command),
                    cwd=request.source_root,
                    env=environment,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=timeout_seconds,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise TransactionError(f"{stage} command failed") from error
            if completed.returncode != 0:
                raise TransactionError(f"{stage} command failed")
        if deadline is not None and clock() >= deadline:
            raise CandidateWindowExpired(
                "candidate-active authorization deadline expired"
            )


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError as error:
        raise TransactionError(
            "transaction command process group ownership differs"
        ) from error
    return True


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + COMMAND_TERMINATION_GRACE_SECONDS
    while _process_group_exists(process.pid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    if _process_group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        try:
            process.wait(timeout=COMMAND_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired as error:
            raise TransactionError(
                "transaction command process group could not be reaped"
            ) from error


PR_SET_CHILD_SUBREAPER = 36
SUPERVISOR_COMMAND_FAILED = 70
SUPERVISOR_COMMAND_TIMED_OUT = 71
SUPERVISOR_DESCENDANTS_ESCAPED = 72
SUPERVISOR_INTERRUPTED = 73
SUPERVISOR_INTERNAL_ERROR = 74


def _enable_child_subreaper() -> None:
    try:
        function = ctypes.CDLL(None, use_errno=True).prctl
    except (AttributeError, OSError) as error:
        raise TransactionError("child-subreaper containment is unavailable") from error
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    function.restype = ctypes.c_int
    ctypes.set_errno(0)
    if function(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise TransactionError(
            "child-subreaper containment could not be enabled"
        ) from OSError(error_number, os.strerror(error_number))


def _proc_parent_pid(pid: int) -> int | None:
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_bytes()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    closing = raw.rfind(b")")
    if closing < 0:
        return None
    fields = raw[closing + 2 :].split()
    if len(fields) < 2:
        return None
    try:
        return int(fields[1])
    except ValueError:
        return None


def _descendant_processes(root_pid: int) -> set[int]:
    parent_by_pid: dict[int, int] = {}
    try:
        entries = os.scandir("/proc")
    except OSError as error:
        raise TransactionError("process containment cannot inspect /proc") from error
    with entries:
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            pid = int(entry.name)
            parent = _proc_parent_pid(pid)
            if parent is not None:
                parent_by_pid[pid] = parent
    descendants: set[int] = set()
    frontier = {root_pid}
    while frontier:
        next_frontier = {
            pid
            for pid, parent in parent_by_pid.items()
            if parent in frontier and pid not in descendants
        }
        descendants.update(next_frontier)
        frontier = next_frontier
    return descendants


def _signal_processes(processes: set[int], signum: int) -> None:
    for pid in sorted(processes, reverse=True):
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            continue
        except PermissionError as error:
            raise TransactionError(
                "transaction descendant ownership differs"
            ) from error


def _reap_supervisor_children(
    root_command_pid: int,
    root_status: int | None,
) -> int | None:
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return root_status
        if pid == 0:
            return root_status
        if pid == root_command_pid:
            root_status = os.waitstatus_to_exitcode(status)


def _cleanup_supervised_descendants(
    supervisor_pid: int,
    root_command_pid: int,
    root_status: int | None,
) -> tuple[bool, int | None]:
    descendants = _descendant_processes(supervisor_pid)
    if descendants:
        _signal_processes(descendants, signal.SIGTERM)
    graceful_deadline = time.monotonic() + COMMAND_TERMINATION_GRACE_SECONDS
    while time.monotonic() < graceful_deadline:
        root_status = _reap_supervisor_children(
            root_command_pid,
            root_status,
        )
        descendants = _descendant_processes(supervisor_pid)
        if not descendants:
            return True, root_status
        time.sleep(
            min(
                0.02,
                max(0.0, graceful_deadline - time.monotonic()),
            )
        )

    previous: set[int] | None = None
    for _attempt in range(64):
        descendants = _descendant_processes(supervisor_pid)
        if not descendants:
            return True, root_status
        _signal_processes(descendants, signal.SIGSTOP)
        time.sleep(0.005)
        current = _descendant_processes(supervisor_pid)
        if current == previous or current.issubset(descendants):
            descendants |= current
            break
        previous = current
    descendants = _descendant_processes(supervisor_pid)
    if descendants:
        _signal_processes(descendants, signal.SIGKILL)
    kill_deadline = time.monotonic() + COMMAND_TERMINATION_GRACE_SECONDS
    while time.monotonic() < kill_deadline:
        root_status = _reap_supervisor_children(
            root_command_pid,
            root_status,
        )
        descendants = _descendant_processes(supervisor_pid)
        if not descendants:
            return True, root_status
        _signal_processes(descendants, signal.SIGKILL)
        time.sleep(
            min(
                0.02,
                max(0.0, kill_deadline - time.monotonic()),
            )
        )
    root_status = _reap_supervisor_children(root_command_pid, root_status)
    return not _descendant_processes(supervisor_pid), root_status


def _exec_supervised_command(
    command: Sequence[str],
    *,
    source_root: Path,
    environment: dict[str, str],
) -> None:
    try:
        os.setsid()
        os.chdir(source_root)
        null_descriptor = os.open(os.devnull, os.O_RDWR | os.O_CLOEXEC)
        for target in (0, 1, 2):
            os.dup2(null_descriptor, target)
        if null_descriptor > 2:
            os.close(null_descriptor)
        maximum = 1_048_576
        try:
            configured = os.sysconf("SC_OPEN_MAX")
            if isinstance(configured, int) and configured > 3:
                maximum = configured
        except (OSError, ValueError):
            pass
        os.closerange(3, maximum)
        os.execvpe(command[0], list(command), environment)
    except BaseException:
        os._exit(127)


def _subreaper_supervisor(
    command: Sequence[str],
    *,
    source_root: Path,
    environment: dict[str, str],
    timeout_seconds: float,
) -> int:
    _enable_child_subreaper()
    supervisor_pid = os.getpid()
    interrupted = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, request_stop)
    root_command_pid = os.fork()
    if root_command_pid == 0:
        _exec_supervised_command(
            command,
            source_root=source_root,
            environment=environment,
        )
        os._exit(127)

    deadline = time.monotonic() + timeout_seconds
    root_status: int | None = None
    reason: int | None = None
    while reason is None:
        root_status = _reap_supervisor_children(
            root_command_pid,
            root_status,
        )
        if interrupted:
            reason = SUPERVISOR_INTERRUPTED
        elif time.monotonic() >= deadline:
            reason = SUPERVISOR_COMMAND_TIMED_OUT
        elif root_status is not None:
            descendants = _descendant_processes(supervisor_pid)
            if descendants:
                reason = SUPERVISOR_DESCENDANTS_ESCAPED
            elif root_status != 0:
                reason = SUPERVISOR_COMMAND_FAILED
            else:
                return 0
        else:
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))

    cleaned, _root_status = _cleanup_supervised_descendants(
        supervisor_pid,
        root_command_pid,
        root_status,
    )
    return reason if cleaned else SUPERVISOR_INTERNAL_ERROR


def _wait_supervisor(
    supervisor_pid: int,
    *,
    timeout_seconds: float,
) -> int:
    deadline = (
        time.monotonic()
        + timeout_seconds
        + 2 * COMMAND_TERMINATION_GRACE_SECONDS
        + 1.0
    )
    termination_sent = False
    while True:
        try:
            pid, status = os.waitpid(supervisor_pid, os.WNOHANG)
        except ChildProcessError as error:
            raise TransactionError(
                "transaction command supervisor disappeared"
            ) from error
        if pid == supervisor_pid:
            return os.waitstatus_to_exitcode(status)
        if time.monotonic() >= deadline:
            if termination_sent:
                raise CommandContainmentLost(
                    "transaction command supervisor did not contain descendants"
                )
            try:
                os.kill(supervisor_pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            termination_sent = True
            deadline = (
                time.monotonic()
                + 2 * COMMAND_TERMINATION_GRACE_SECONDS
                + 1.0
            )
        time.sleep(0.02)


def _stop_supervisor(supervisor_pid: int) -> None:
    try:
        os.kill(supervisor_pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = (
        time.monotonic()
        + 2 * COMMAND_TERMINATION_GRACE_SECONDS
        + 1.0
    )
    while time.monotonic() < deadline:
        try:
            pid, status = os.waitpid(supervisor_pid, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == supervisor_pid:
            if (
                os.waitstatus_to_exitcode(status)
                == SUPERVISOR_INTERNAL_ERROR
            ):
                raise CommandContainmentLost(
                    "transaction command supervisor lost descendants"
                )
            return
        time.sleep(0.02)
    raise CommandContainmentLost(
        "transaction command supervisor could not contain descendants"
    )


def _run_owned_process_group(
    command: Sequence[str],
    *,
    request: TransactionRequest,
    environment: dict[str, str],
    stage: str,
    timeout_seconds: float | None = None,
) -> None:
    selected_timeout = (
        request.command_timeout_seconds
        if timeout_seconds is None
        else timeout_seconds
    )
    if (
        not math.isfinite(selected_timeout)
        or selected_timeout <= 0
        or selected_timeout > MAX_COMMAND_TIMEOUT_SECONDS
    ):
        raise TransactionError(f"{stage} command timeout is invalid")
    if not hasattr(os, "fork"):
        raise TransactionError(
            f"{stage} command requires child-subreaper containment"
        )
    supervisor_pid = -1
    try:
        supervisor_pid = os.fork()
        if supervisor_pid == 0:
            try:
                result = _subreaper_supervisor(
                    command,
                    source_root=request.source_root,
                    environment=environment,
                    timeout_seconds=selected_timeout,
                )
            except BaseException:
                result = SUPERVISOR_INTERNAL_ERROR
            os._exit(result)
        returncode = _wait_supervisor(
            supervisor_pid,
            timeout_seconds=selected_timeout,
        )
    except BaseException:
        if supervisor_pid > 0:
            _stop_supervisor(supervisor_pid)
        raise
    if returncode == SUPERVISOR_INTERNAL_ERROR:
        raise CommandContainmentLost(
            f"{stage} command descendant containment failed"
        )
    if returncode != 0:
        raise TransactionError(f"{stage} command failed")


def _inventory_file(path: Path, label: str) -> tuple[int, str]:
    snapshot = _read_input(path, label, MAX_OUTPUT_FILE_BYTES)
    return len(snapshot.raw), snapshot.sha256


def _entry_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _scan_output_tree(path: Path) -> dict[str, tuple[int, ...]]:
    try:
        root = path.lstat()
    except OSError as error:
        raise TransactionError("campaign output is unavailable") from error
    if not stat.S_ISDIR(root.st_mode) or stat.S_ISLNK(root.st_mode):
        raise TransactionError("campaign output must be a directory")
    result: dict[str, tuple[int, ...]] = {".": _entry_identity(root)}
    pending = [path]
    seen_directories = {(root.st_dev, root.st_ino)}
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = []
                for entry in iterator:
                    if len(result) + len(entries) >= MAX_OUTPUT_FILES + 1:
                        raise TransactionError(
                            "campaign output has too many entries"
                        )
                    entries.append(entry)
        except OSError as error:
            raise TransactionError(
                "campaign output cannot be enumerated"
            ) from error
        for entry in entries:
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise TransactionError(
                    "campaign output entry cannot be inspected"
                ) from error
            member = Path(entry.path)
            try:
                relative = member.relative_to(path).as_posix()
            except ValueError as error:
                raise TransactionError(
                    "campaign output entry escaped its root"
                ) from error
            if (
                not relative
                or relative.startswith("/")
                or ".." in Path(relative).parts
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_dev != root.st_dev
            ):
                raise TransactionError("campaign output entry is unsafe")
            if stat.S_ISDIR(metadata.st_mode):
                directory_identity = (metadata.st_dev, metadata.st_ino)
                if directory_identity in seen_directories:
                    raise TransactionError(
                        "campaign output directory identity is repeated"
                    )
                seen_directories.add(directory_identity)
                pending.append(member)
            elif not stat.S_ISREG(metadata.st_mode):
                raise TransactionError(
                    "campaign output contains a non-regular member"
                )
            result[relative] = _entry_identity(metadata)
    return result


def _output_inventory(
    path: Path,
    *,
    run_id: str,
    campaign_name: str,
    required_uid: int,
    candidate_raw: bytes,
) -> dict[str, Any]:
    if not path.is_absolute():
        raise TransactionError("campaign output path is not absolute")
    layout = CAMPAIGN_OUTPUT_LAYOUTS.get(campaign_name)
    if layout is None:
        raise TransactionError("campaign output layout is unknown")
    first = _scan_output_tree(path)
    files = {
        relative
        for relative, identity in first.items()
        if relative != "." and stat.S_ISREG(identity[2])
    }
    directories = {
        relative
        for relative, identity in first.items()
        if relative != "." and stat.S_ISDIR(identity[2])
    }
    if (
        files != layout["files"]
        or directories != layout["directories"]
    ):
        raise TransactionError(f"{campaign_name} output layout differs")
    root_identity = first["."]
    if (
        root_identity[3] != required_uid
        or stat.S_IMODE(root_identity[2]) != layout["directory_mode"]
    ):
        raise TransactionError(f"{campaign_name} output root metadata differs")
    selected: dict[str, str] = {}
    inventory = []
    total_bytes = 0
    for relative in sorted(files, key=lambda item: item.encode("utf-8")):
        identity = first[relative]
        if (
            identity[3] != required_uid
            or identity[5] != 1
            or stat.S_IMODE(identity[2]) != layout["file_mode"]
        ):
            raise TransactionError(
                f"{campaign_name} output file metadata differs"
            )
        size, digest = _inventory_file(
            path / relative,
            f"{campaign_name} output member",
        )
        total_bytes += size
        if total_bytes > MAX_OUTPUT_TOTAL_BYTES:
            raise TransactionError("campaign output byte total is invalid")
        inventory.append({"path": relative, "bytes": size, "sha256": digest})
        if (
            relative in SELECTED_ARTIFACTS
            or Path(relative).name in SELECTED_ARTIFACTS
        ):
            selected[relative] = digest
    for relative in directories:
        identity = first[relative]
        if (
            identity[3] != required_uid
            or stat.S_IMODE(identity[2]) != layout["directory_mode"]
        ):
            raise TransactionError(
                f"{campaign_name} output directory metadata differs"
            )
    after_hashing = _scan_output_tree(path)
    if after_hashing != first:
        raise TransactionError("campaign output tree changed during inventory")
    candidate_artifact = _read_input(
        path / "candidate-served-model.json",
        f"{campaign_name} candidate manifest artifact",
        MAX_MANIFEST_BYTES,
    )
    if candidate_artifact.raw != candidate_raw:
        raise TransactionError(
            f"{campaign_name} candidate manifest artifact differs"
        )
    if campaign_name in {"reasoning_release", "reasoning_browser"}:
        evidence_name = (
            "summary.json"
            if campaign_name == "reasoning_release"
            else "browser-evidence.json"
        )
        expected_schema = (
            "ullm.generic_reasoning_release_campaign.v2"
            if campaign_name == "reasoning_release"
            else "ullm.openwebui.reasoning_browser_smoke.v4"
        )
        evidence_snapshot = _read_input(
            path / evidence_name,
            f"{campaign_name} primary evidence",
            MAX_OUTPUT_FILE_BYTES,
        )
        binding_snapshot = _read_input(
            path / "active-manifest-binding.json",
            f"{campaign_name} active binding",
            MAX_OUTPUT_FILE_BYTES,
        )
        try:
            evidence = authorization.strict_json_bytes(
                evidence_snapshot.raw,
                f"{campaign_name} primary evidence",
            )
            binding = authorization.strict_json_bytes(
                binding_snapshot.raw,
                f"{campaign_name} active binding",
            )
        except authorization.AuthorizationError as error:
            raise TransactionError(
                f"{campaign_name} semantic evidence is invalid"
            ) from error
        campaign = binding.get("campaign")
        candidate = binding.get("candidate")
        if (
            evidence.get("schema_version") != expected_schema
            or binding.get("schema_version")
            != "ullm.served_model.active_binding.v1"
            or binding.get("status") != "complete"
            or not isinstance(campaign, dict)
            or campaign.get("name") != campaign_name
            or campaign.get("run_id") != run_id
            or campaign.get("final_path") != os.fspath(path)
            or not isinstance(candidate, dict)
            or candidate.get("sha256") != _sha256(candidate_raw)
        ):
            raise TransactionError(
                f"{campaign_name} semantic lineage differs"
            )
    after_semantic_reads = _scan_output_tree(path)
    if after_semantic_reads != first:
        raise TransactionError("campaign output tree changed during inventory")
    if total_bytes < 1 or total_bytes > MAX_OUTPUT_TOTAL_BYTES:
        raise TransactionError("campaign output byte total is invalid")
    return {
        "run_id": run_id,
        "path": os.fspath(path),
        "kind": "directory",
        "sha256": _sha256(_canonical_json({"files": inventory})),
        "artifact_count": len(inventory),
        "total_bytes": total_bytes,
        "selected_artifacts": selected,
    }


class _TerminationController:
    def __init__(self) -> None:
        self.installed: dict[int, Any] = {}
        self.defer_depth = 0
        self.pending: int | None = None

    def _interrupt(self, signum: int, _frame: Any) -> None:
        if self.defer_depth:
            if self.pending is None:
                self.pending = signum
            return
        raise TransactionInterrupted(f"termination signal {signum}")

    def __enter__(self) -> "_TerminationController":
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                self.installed[signum] = signal.getsignal(signum)
                signal.signal(signum, self._interrupt)
            except (ValueError, OSError):
                continue
        return self

    def __exit__(self, *_exc: Any) -> None:
        for signum, handler in self.installed.items():
            signal.signal(signum, handler)

    def take_pending(self) -> int | None:
        signum, self.pending = self.pending, None
        return signum

    def raise_if_pending(self) -> None:
        signum = self.take_pending()
        if signum is not None:
            raise TransactionInterrupted(f"termination signal {signum}")

    @contextmanager
    def deferred(self) -> Iterator[None]:
        self.defer_depth += 1
        try:
            yield
        finally:
            self.defer_depth -= 1


def _termination_guard() -> _TerminationController:
    return _TerminationController()


def _complete_stage_states(stages: dict[str, str], failed: str | None) -> None:
    for name, value in tuple(stages.items()):
        if value == "pending":
            stages[name] = "skipped"
    if failed is not None:
        stages[failed] = "failed"


def execute_transaction(
    request: TransactionRequest,
    *,
    policy: authorization.RegistryPolicy = authorization.RegistryPolicy(),
    validator: ManifestValidator = default_manifest_validator,
    runner: CommandRunner = subprocess.run,
    inactive_checker: InactiveChecker = default_inactive_checker,
    clock: Clock = utc_now,
    restoration_probe: RestorationProbe = default_restoration_probe,
) -> TransactionResult:
    """Consume one authorization and run the complete locked temporary window."""

    _validate_commands(request.commands)
    with _termination_guard() as termination, ExitStack() as resources:
        claimed_at = clock()
        try:
            with termination.deferred():
                claim = authorization.claim_authorization(
                    request.authorization_path,
                    now=claimed_at,
                    policy=policy,
                )
        except authorization.AuthorizationError as error:
            raise TransactionError("campaign authorization claim failed") from error

        stages = {
            name: "pending" for name in authorization.OUTCOME_STAGE_FIELDS
        }
        stages["claim"] = "passed"
        campaign_results: dict[str, dict[str, Any] | None] = {
            name: None for name in authorization.CAMPAIGN_FIELDS
        }
        observations: list[dict[str, Any]] = []
        active_slot: ActiveSlot | None = None
        preflight_result: TransactionPreflight | None = None
        failure_stage: str | None = None
        primary_error: BaseException | None = None
        switched = False
        ownership_lost = False
        containment_lost = False
        restoration: dict[str, Any] = {
            "expected_manifest_sha256": claim.authorization.document["before"][
                "manifest_sha256"
            ],
            "displaced_manifest_sha256": None,
            "observed_manifest_sha256": None,
            "bytes_equal": False,
            "reverse_reconciliation_passed": False,
            "final_checks_passed": False,
            "model_id": None,
            "format_id": None,
            "worker_binary_sha256": None,
            "proof": None,
        }
        candidate_deadline = claim.authorization.expires_at

        def fail_stage(
            stage: str,
            error: BaseException,
            *,
            prioritize: bool = False,
        ) -> None:
            nonlocal failure_stage, primary_error
            nonlocal ownership_lost, containment_lost
            if isinstance(error, ActiveSlotOwnershipLost):
                ownership_lost = True
                if error.displaced_sha256 is not None:
                    restoration["displaced_manifest_sha256"] = (
                        error.displaced_sha256
                    )
            if isinstance(error, CommandContainmentLost):
                containment_lost = True
            stages[stage] = "failed"
            if failure_stage is None or prioritize:
                failure_stage = stage
            if primary_error is None:
                primary_error = error

        def repin() -> None:
            assert preflight_result is not None
            _repin_transaction_inputs(
                request,
                claim,
                preflight_result,
                policy=policy,
                runner=runner,
                now=clock(),
            )

        def ensure_candidate_window() -> None:
            if clock() >= candidate_deadline:
                raise CandidateWindowExpired(
                    "candidate-active authorization deadline expired"
                )

        def execute_commands(
            stage: str,
            commands: Sequence[Sequence[str]],
            *,
            candidate_active: bool = False,
        ) -> None:
            assert preflight_result is not None
            if candidate_active:
                ensure_candidate_window()
            repin()
            if candidate_active:
                ensure_candidate_window()
            _run_commands(
                commands,
                request=request,
                claim=claim,
                preflight_result=preflight_result,
                stage=stage,
                runner=runner,
                deadline=candidate_deadline if candidate_active else None,
                clock=clock,
            )
            if candidate_active:
                ensure_candidate_window()
            repin()
            if candidate_active:
                ensure_candidate_window()

        outcome_snapshot: authorization.FileSnapshot | None = None
        status = "failed_restore"
        try:
            termination.raise_if_pending()
            try:
                active_slot = ActiveSlot.acquire(
                    request.active_manifest,
                    required_uid=policy.required_uid,
                )
                resources.callback(active_slot.close)
                stages["lock"] = "passed"
            except BaseException as error:
                fail_stage("lock", error)
                raise

            try:
                preflight_result = preflight(
                    request,
                    now=clock(),
                    policy=policy,
                    validator=validator,
                    runner=runner,
                    require_fresh_outputs=True,
                    claimed=claim,
                )
                if (
                    preflight_result.authorization.snapshot.sha256
                    != claim.authorization.snapshot.sha256
                ):
                    raise TransactionError(
                        "authorization changed between claim and transaction"
                    )
                if active_slot.path != preflight_result.active.path:
                    raise TransactionError("locked active manifest path differs")
                authorization.require_window_binding(
                    claim,
                    source_commit=preflight_result.source_commit,
                    source_tree=preflight_result.source_tree,
                    before_manifest_sha256=preflight_result.active.sha256,
                    candidate_manifest_sha256=preflight_result.candidate.sha256,
                    candidate_worker_binary_sha256=preflight_result.candidate_summary[
                        "worker"
                    ]["binary_sha256"],
                    candidate_promotion_receipt_sha256=(
                        preflight_result.candidate_promotion_receipt_sha256
                    ),
                    rollback_backup_path=Path(
                        claim.authorization.document["rollback"]["backup_path"]
                    ),
                )
                inactive_checker(request.inactive_services)
                stages["preflight"] = "passed"
            except BaseException as error:
                fail_stage("preflight", error)
                raise

            try:
                _exclusive_publish(
                    Path(claim.authorization.document["rollback"]["backup_path"]),
                    preflight_result.active.raw,
                    mode=0o444,
                    required_uid=policy.required_uid,
                )
                stages["backup"] = "passed"
            except BaseException as error:
                fail_stage("backup", error)
                raise

            try:
                # Mark the transaction as restoration-required before replace.
                ensure_candidate_window()
                _revalidate_pre_switch(request, preflight_result)
                inactive_checker(request.inactive_services)
                repin()
                ensure_candidate_window()
                switched = True
                active_slot.replace(
                    preflight_result.candidate.raw,
                    preflight_result.active.identity,
                    expected_current=preflight_result.active,
                )
                ensure_candidate_window()
                repin()
                ensure_candidate_window()
                observations.append(
                    _observe_candidate(
                        preflight_result,
                        stage="candidate_activation",
                    )
                )
                stages["candidate_activation"] = "passed"
            except BaseException as error:
                fail_stage("candidate_activation", error)
                raise

            try:
                execute_commands(
                    "candidate_reconciliation",
                    request.commands.candidate_reconciliation,
                    candidate_active=True,
                )
                observations.append(
                    _observe_candidate(
                        preflight_result,
                        stage="candidate_reconciliation",
                    )
                )
                stages["candidate_reconciliation"] = "passed"
            except BaseException as error:
                fail_stage("candidate_reconciliation", error)
                raise

            try:
                execute_commands(
                    "candidate_checks",
                    request.commands.candidate_checks,
                    candidate_active=True,
                )
                observations.append(
                    _observe_candidate(
                        preflight_result,
                        stage="candidate_checks",
                    )
                )
                stages["candidate_checks"] = "passed"
            except BaseException as error:
                fail_stage("candidate_checks", error)
                raise

            campaign_commands = {
                "sq8_full": request.commands.sq8_full,
                "reasoning_release": request.commands.reasoning_release,
                "reasoning_browser": request.commands.reasoning_browser,
            }
            for name in ("sq8_full", "reasoning_release", "reasoning_browser"):
                try:
                    ensure_candidate_window()
                    campaign = claim.authorization.document["campaigns"][name]
                    final_path = Path(campaign["final_path"])
                    authorization.require_campaign_binding(
                        claim,
                        campaign_name=name,
                        run_id=campaign["run_id"],
                        final_path=final_path,
                    )
                    observations.append(
                        _observe_candidate(
                            preflight_result,
                            stage=f"{name}:before",
                        )
                    )
                    if final_path.exists() or final_path.is_symlink():
                        raise TransactionError(
                            f"{name} authorized output is not fresh"
                        )
                    legacy_browser_sidecar = final_path.with_name(
                        f"{final_path.name}.active-binding"
                    )
                    if (
                        name == "reasoning_browser"
                        and (
                            legacy_browser_sidecar.exists()
                            or legacy_browser_sidecar.is_symlink()
                        )
                    ):
                        raise TransactionError(
                            "legacy browser binding sidecar is not authorized"
                        )
                    execute_commands(
                        name,
                        (campaign_commands[name],),
                        candidate_active=True,
                    )
                    ensure_candidate_window()
                    observations.append(
                        _observe_candidate(
                            preflight_result,
                            stage=f"{name}:after",
                        )
                    )
                    inventory = _output_inventory(
                        final_path,
                        run_id=campaign["run_id"],
                        campaign_name=name,
                        required_uid=policy.required_uid,
                        candidate_raw=preflight_result.candidate.raw,
                    )
                    ensure_candidate_window()
                    if (
                        name == "reasoning_browser"
                        and (
                            legacy_browser_sidecar.exists()
                            or legacy_browser_sidecar.is_symlink()
                        )
                    ):
                        raise TransactionError(
                            "browser campaign published an unauthorized sidecar"
                        )
                    campaign_results[name] = inventory
                    stages[name] = "passed"
                except BaseException as error:
                    fail_stage(name, error)
                    raise
        except BaseException as error:
            if primary_error is None:
                primary_error = error
                if failure_stage is None:
                    failure_stage = "preflight"
                    stages["preflight"] = "failed"
        finally:
            deferred_interrupt: TransactionInterrupted | None = None
            if active_slot is not None and preflight_result is not None:
                try:
                    last_restore_error: BaseException | None = None
                    with termination.deferred():
                        if ownership_lost or containment_lost:
                            if containment_lost:
                                raise CommandContainmentLost(
                                    "command descendant containment was lost; "
                                    "AQ4 restoration cannot begin safely"
                                )
                            raise ActiveSlotOwnershipLost(
                                "active manifest ownership was lost; "
                                "AQ4 restoration will not overwrite it"
                            )
                        restore_expected: StableFileSnapshot | None = None
                        if switched:
                            restore_expected = active_slot.snapshot_current()
                            if (
                                restoration["displaced_manifest_sha256"]
                                is None
                                or restore_expected.raw
                                != preflight_result.active.raw
                            ):
                                restoration[
                                    "displaced_manifest_sha256"
                                ] = restore_expected.sha256
                        for _attempt in range(2):
                            try:
                                if switched:
                                    assert restore_expected is not None
                                    restore_current = active_slot.snapshot_current()
                                    if restore_current != restore_expected:
                                        raise ActiveSlotOwnershipLost(
                                            "active manifest changed before "
                                            "intentional AQ4 restoration",
                                            displaced_sha256=(
                                                restore_current.sha256
                                            ),
                                        )
                                    restore_expected = active_slot.replace(
                                        preflight_result.active.raw,
                                        preflight_result.active.identity,
                                        expected_current=restore_current,
                                    )
                                restored = _read_input(
                                    preflight_result.active.path,
                                    "restored active served-model manifest",
                                    MAX_MANIFEST_BYTES,
                                )
                                if (
                                    restored.raw != preflight_result.active.raw
                                    or restored.identity.uid
                                    != preflight_result.active.identity.uid
                                    or restored.identity.gid
                                    != preflight_result.active.identity.gid
                                    or stat.S_IMODE(restored.identity.mode)
                                    != stat.S_IMODE(
                                        preflight_result.active.identity.mode
                                    )
                                    or restored.identity.links != 1
                                    or (
                                        restore_expected is not None
                                        and restored != restore_expected
                                    )
                                ):
                                    raise TransactionError(
                                        "restored active manifest identity differs"
                                    )
                                last_restore_error = None
                                break
                            except ActiveSlotOwnershipLost as error:
                                ownership_lost = True
                                if error.displaced_sha256 is not None:
                                    restoration[
                                        "displaced_manifest_sha256"
                                    ] = error.displaced_sha256
                                last_restore_error = error
                                break
                            except BaseException as error:
                                last_restore_error = error
                        if last_restore_error is not None:
                            raise last_restore_error
                    stages["aq4_restore"] = "passed"
                except BaseException as error:
                    fail_stage("aq4_restore", error, prioritize=True)

                pending_signum = termination.take_pending()
                if pending_signum is not None:
                    deferred_interrupt = TransactionInterrupted(
                        f"termination signal {pending_signum}"
                    )

                if stages["aq4_restore"] == "passed":
                    try:
                        execute_commands(
                            "reverse_reconciliation",
                            request.commands.reverse_reconciliation,
                        )
                        stages["reverse_reconciliation"] = "passed"
                    except BaseException as error:
                        fail_stage(
                            "reverse_reconciliation",
                            error,
                            prioritize=True,
                        )
                    final_commands_passed = False
                    try:
                        execute_commands(
                            "final_checks",
                            request.commands.final_checks,
                        )
                        final_commands_passed = True
                    except BaseException as error:
                        fail_stage("final_checks", error, prioritize=True)
                    if deferred_interrupt is not None:
                        fail_stage(
                            "aq4_restore",
                            deferred_interrupt,
                            prioritize=True,
                        )
                    try:
                        repin()
                        proof = restoration_probe(
                            request,
                            claim,
                            preflight_result,
                        )
                        restoration_proof.validate_proof(
                            proof,
                            authorization_sha256=(
                                claim.authorization.snapshot.sha256
                            ),
                            claim_sha256=claim.snapshot.sha256,
                            active_manifest_path=preflight_result.active.path,
                            expected_manifest_sha256=preflight_result.active.sha256,
                            expected_worker_sha256=claim.authorization.document[
                                "before"
                            ]["worker_binary_sha256"],
                            service_unit=request.service_unit,
                        )
                        restoration.update(
                            observed_manifest_sha256=preflight_result.active.sha256,
                            bytes_equal=True,
                            model_id="ullm-qwen3.5-9b-aq4",
                            format_id="AQ4_0",
                            worker_binary_sha256=proof["worker"][
                                "executable_sha256"
                            ],
                            proof=proof,
                        )
                        if final_commands_passed:
                            stages["final_checks"] = "passed"
                    except BaseException as error:
                        fail_stage("final_checks", error, prioritize=True)
                else:
                    stages["reverse_reconciliation"] = "skipped"
                    stages["final_checks"] = "skipped"
                restoration["reverse_reconciliation_passed"] = (
                    stages["reverse_reconciliation"] == "passed"
                )
                restoration["final_checks_passed"] = (
                    stages["final_checks"] == "passed"
                )
            try:
                with termination.deferred():
                    _complete_stage_states(stages, failure_stage)
                    restoration_proved = (
                        restoration["bytes_equal"]
                        and restoration["reverse_reconciliation_passed"]
                        and restoration["final_checks_passed"]
                        and restoration["model_id"] == "ullm-qwen3.5-9b-aq4"
                        and restoration["format_id"] == "AQ4_0"
                        and restoration["worker_binary_sha256"]
                        == claim.authorization.document["before"][
                            "worker_binary_sha256"
                        ]
                        and restoration["proof"] is not None
                    )
                    if primary_error is None and all(
                        value == "passed" for value in stages.values()
                    ):
                        status = "succeeded_restored"
                        failure_stage = None
                    elif restoration_proved:
                        status = "failed_restored"
                        if failure_stage is None:
                            failure_stage = "candidate_checks"
                            stages[failure_stage] = "failed"
                    else:
                        status = "failed_restore"
                        if failure_stage is None:
                            failure_stage = "aq4_restore"
                            stages[failure_stage] = "failed"

                    completed_at = clock()
                    outcome = {
                        "schema_version": authorization.OUTCOME_SCHEMA,
                        "authorization_id": claim.authorization.document[
                            "authorization_id"
                        ],
                        "authorization_path": os.fspath(
                            claim.authorization.snapshot.path
                        ),
                        "authorization_sha256": (
                            claim.authorization.snapshot.sha256
                        ),
                        "claim_path": os.fspath(claim.snapshot.path),
                        "claim_sha256": claim.snapshot.sha256,
                        "started_at": authorization.utc_timestamp(claimed_at),
                        "completed_at": authorization.utc_timestamp(completed_at),
                        "status": status,
                        "failure_stage": failure_stage,
                        "stages": stages,
                        "candidate_observations": observations,
                        "campaigns": campaign_results,
                        "restoration": restoration,
                    }
                    try:
                        outcome_snapshot = authorization.publish_outcome(
                            claim,
                            outcome,
                            policy=policy,
                        )
                    except authorization.AuthorizationError as error:
                        raise TransactionError(
                            "campaign outcome publication failed after claim consumption"
                        ) from error
            finally:
                if active_slot is not None:
                    active_slot.close()
                    active_slot = None

        assert outcome_snapshot is not None
        result = TransactionResult(
            outcome_snapshot.path,
            outcome_snapshot.sha256,
            status,
        )
        pending_signum = termination.take_pending()
        if pending_signum is not None:
            raise TransactionInterrupted(
                "termination signal received after durable outcome "
                f"{result.outcome_path} ({result.outcome_sha256})"
            )
        if status != "succeeded_restored":
            raise TransactionFailed(
                f"cross-model campaign transaction ended as {status}",
                result=result,
                backup_path=Path(
                    claim.authorization.document["rollback"]["backup_path"]
                ),
                restoration=restoration,
            ) from primary_error
        return result
