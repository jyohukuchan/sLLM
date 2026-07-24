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
import served_model_campaign_runtime_seal as campaign_runtime_seal  # noqa: E402
import served_model_campaign_source_seal as campaign_source_seal  # noqa: E402
import sq8_serving_promotion as sq8_promotion  # noqa: E402
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
SQ8_FULL_MAX_TIMEOUT_SECONDS = 6 * 60 * 60.0
COMMAND_TERMINATION_GRACE_SECONDS = 2.0
CANDIDATE_STABILIZATION_SECONDS = 15 * 60.0
CANDIDATE_STABILIZATION_POLL_SECONDS = 30.0
MAX_CANDIDATE_STABILIZATION_POLLS = 4_096
GIT_RE = re.compile(r"[0-9a-f]{40}\Z")
GIT_BINARY = campaign_source_seal.GIT_BINARY
GIT_COMMAND_PREFIX = campaign_source_seal.GIT_COMMAND_PREFIX
PYTHON_BINARY = "/usr/bin/python3.12"
DOCKER_BINARY = "/usr/bin/docker"
DOCKER_LEASE_WRAPPER_RELATIVE_PATH = Path("tools/ullm-campaign-docker")
DOCKER_LEASE_WRAPPER_ENVIRONMENT = "ULLM_CAMPAIGN_DOCKER"
DOCKER_LEASE_LABEL_ENVIRONMENT = "ULLM_CAMPAIGN_DOCKER_LEASE_LABEL"
DOCKER_LEASE_LABEL_KEY = "com.ultimatellm.served-model-campaign.claim"
DOCKER_LEASE_CONTROL_TIMEOUT_SECONDS = 30.0
DOCKER_LEASE_QUIESCENCE_SECONDS = COMMAND_TERMINATION_GRACE_SECONDS
DOCKER_LEASE_QUIESCENCE_POLL_SECONDS = 0.25
DOCKER_LEASE_MINIMUM_EMPTY_POLLS = 3
DOCKER_LEASE_CLEANUP_TOTAL_TIMEOUT_SECONDS = 95.0
MAX_DOCKER_LEASE_QUIESCENCE_POLLS = 512
MAX_DOCKER_LEASE_CONTAINERS = 256
OPENWEBUI_IMAGE_VERIFIER = Path("tools/verify-openwebui-container-image.py")
NESTED_EXECUTABLE_FLAGS = frozenset(
    {"--docker", "--rocm-smi", "--systemctl"}
)
DOCKER_PRODUCER_TOOL_NAMES = frozenset(
    {
        "run-sq8-full-openwebui-campaign.py",
        "run-generic-reasoning-release-campaign.py",
        "run-openwebui-reasoning-browser-smoke.py",
        "verify-openwebui-container-image.py",
    }
)
FIXED_GATEWAY_API_KEY_PATH = Path("/etc/ullm/openai-api-key")
FIXED_GATEWAY_API_KEY_UID = 0
FIXED_GATEWAY_API_KEY_GID = 1000
FIXED_GATEWAY_API_KEY_MODE = 0o640
FIXED_OPENWEBUI_SESSION_TOKEN_PATH = Path(
    "/run/ullm-campaign-secrets/openwebui-session.jwt"
)
FIXED_OPENWEBUI_SESSION_TOKEN_PARENT = Path("/run/ullm-campaign-secrets")
FIXED_OPENWEBUI_SESSION_TOKEN_PARENT_UID = 0
FIXED_OPENWEBUI_SESSION_TOKEN_PARENT_GID = 1000
FIXED_OPENWEBUI_SESSION_TOKEN_PARENT_MODE = 0o750
CAMPAIGN_EXECUTOR_UID = 1000
CAMPAIGN_EXECUTOR_GID = 1000
CAMPAIGN_EXECUTOR_SUPPLEMENTARY_GROUPS = (27, 44, 984, 992, 1000)
CAMPAIGN_STAGING_OUTPUT_ENVIRONMENT = "ULLM_CAMPAIGN_STAGING_OUTPUT"
CAMPAIGN_SOURCE_ROOT_ENVIRONMENT = "ULLM_CAMPAIGN_SOURCE_ROOT"
FIXED_OPENWEBUI_SESSION_TOKEN_UID = 0
FIXED_OPENWEBUI_SESSION_TOKEN_GID = 1000
FIXED_OPENWEBUI_SESSION_TOKEN_MODE = 0o640
STAGE_BASE_ENVIRONMENT = {
    "HOME": "/nonexistent",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONSAFEPATH": "1",
    "TZ": "UTC",
}
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
AQ4_REASONING_RELEASE_FILES = frozenset(
    {
        "cases.json",
        "lifecycle.json",
        "resource-samples.jsonl",
        "summary.json",
    }
)
CAMPAIGN_OUTPUT_LAYOUTS = {
    "aq4_reasoning_release": {
        "files": AQ4_REASONING_RELEASE_FILES,
        "directories": frozenset(),
        "directory_mode": 0o555,
        "file_mode": 0o444,
    },
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
    aq4_reasoning_release: tuple[tuple[str, ...], ...]
    aq4_reasoning_browser: tuple[tuple[str, ...], ...]
    aq4_bundle: tuple[tuple[str, ...], ...]
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
    source_seal: campaign_source_seal.SourceSeal
    active: StableFileSnapshot
    candidate: StableFileSnapshot
    active_summary: dict[str, Any]
    candidate_summary: dict[str, Any]
    systemd_unit_sha256: str
    environment_sha256: str
    candidate_promotion_receipt_sha256: str
    aq4_source_root: Path
    aq4_source_commit: str
    aq4_source_tree: str
    aq4_source_seal: campaign_source_seal.SourceSeal | None
    aq4_worker_binary: StableFileSnapshot | None
    aq4_promotion_receipt: StableFileSnapshot | None
    aq4_promotion_evidence: StableFileSnapshot | None
    api_key_sha256: str | None
    openwebui_session_token_sha256: str | None
    runtime_artifact_seals: tuple[
        campaign_runtime_seal.RuntimeArtifactSeal, ...
    ] = ()
    runtime_tree_seals: tuple[campaign_runtime_seal.RuntimeTreeSeal, ...] = ()
    candidate_runtime_artifact_seals: tuple[
        campaign_runtime_seal.RuntimeArtifactSeal, ...
    ] = ()
    candidate_runtime_tree_seals: tuple[
        campaign_runtime_seal.RuntimeTreeSeal, ...
    ] = ()
    aq4_runtime_artifact_seals: tuple[
        campaign_runtime_seal.RuntimeArtifactSeal, ...
    ] = ()
    aq4_runtime_tree_seals: tuple[
        campaign_runtime_seal.RuntimeTreeSeal, ...
    ] = ()
    shared_runtime_artifact_seals: tuple[
        campaign_runtime_seal.RuntimeArtifactSeal, ...
    ] = ()
    shared_runtime_tree_seals: tuple[
        campaign_runtime_seal.RuntimeTreeSeal, ...
    ] = ()
    aq4_release_runtime_artifact_seals: tuple[
        campaign_runtime_seal.RuntimeArtifactSeal, ...
    ] = ()
    aq4_release_runtime_tree_seals: tuple[
        campaign_runtime_seal.RuntimeTreeSeal, ...
    ] = ()


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
CandidateStabilizationProbe = Callable[
    [
        TransactionRequest,
        authorization.ClaimRecord,
        TransactionPreflight,
    ],
    dict[str, Any],
]
Sleeper = Callable[[float], None]
MonotonicClock = Callable[[], float]


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
        commands.aq4_reasoning_release,
        commands.aq4_reasoning_browser,
        commands.aq4_bundle,
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


def _docker_lease_wrapper(source_root: Path) -> Path:
    return source_root / DOCKER_LEASE_WRAPPER_RELATIVE_PATH


def _require_docker_lease_wrapper_commands(
    commands: TransactionCommands,
    source_root: Path,
    *,
    recovery_only: bool = False,
) -> None:
    wrapper = os.fspath(_docker_lease_wrapper(source_root))
    candidate_groups: tuple[Sequence[Sequence[str]], ...] = (
        commands.candidate_reconciliation,
        commands.candidate_checks,
        (commands.sq8_full,),
        (commands.reasoning_release,),
        (commands.reasoning_browser,),
    )
    aq4_groups: tuple[Sequence[Sequence[str]], ...] = (
        commands.reverse_reconciliation,
        commands.aq4_reasoning_release,
        commands.aq4_reasoning_browser,
        commands.aq4_bundle,
        commands.final_checks,
    )

    def inspect(groups: Sequence[Sequence[Sequence[str]]]) -> bool:
        found = False
        for group in groups:
            for command in group:
                docker_flags = tuple(
                    index
                    for index, argument in enumerate(command)
                    if argument == "--docker"
                )
                if (
                    len(docker_flags) > 1
                    or any(
                        argument.startswith("--docker=")
                        for argument in command
                    )
                    or any(
                        Path(argument).name == "docker"
                        for argument in command
                        if not argument.startswith("-")
                    )
                ):
                    raise TransactionError(
                        "transaction command bypasses the Docker lease wrapper"
                    )
                for index in docker_flags:
                    if index + 1 >= len(command) or command[index + 1] != wrapper:
                        raise TransactionError(
                            "transaction Docker executable binding differs"
                        )
                    found = True
                wrapper_indexes = tuple(
                    index
                    for index, argument in enumerate(command)
                    if argument == wrapper
                )
                allowed_wrapper_indexes = {
                    index + 1 for index in docker_flags
                }
                direct_wrapper = (
                    len(command) >= 5
                    and command[4] == wrapper
                    and tuple(command[:4])
                    == (PYTHON_BINARY, "-I", "-S", "-B")
                )
                if direct_wrapper:
                    allowed_wrapper_indexes.add(4)
                    found = True
                if any(
                    index not in allowed_wrapper_indexes
                    for index in wrapper_indexes
                ):
                    raise TransactionError(
                        "transaction Docker lease wrapper placement differs"
                    )
                producer_indexes = tuple(
                    index
                    for index, argument in enumerate(command)
                    if Path(argument).name in DOCKER_PRODUCER_TOOL_NAMES
                )
                if producer_indexes:
                    if (
                        len(producer_indexes) != 1
                        or producer_indexes[0] != 4
                        or tuple(command[:4])
                        != (PYTHON_BINARY, "-I", "-S", "-B")
                        or len(docker_flags) != 1
                    ):
                        raise TransactionError(
                            "Docker producer wrapper binding differs"
                        )
                if command[0] == wrapper:
                    raise TransactionError(
                        "direct Docker lease wrapper prefix differs"
                    )
        return found

    candidate_bound = inspect(candidate_groups) if not recovery_only else True
    aq4_bound = inspect(aq4_groups)
    if not candidate_bound or not aq4_bound:
        raise TransactionError(
            "transaction Docker lease wrapper is not bound in every route"
        )


def _docker_lease_label(claim: authorization.ClaimRecord) -> str:
    claim_sha256 = claim.snapshot.sha256
    if authorization.HASH_RE.fullmatch(claim_sha256) is None:
        raise CommandContainmentLost("campaign Docker lease claim hash is invalid")
    return f"{DOCKER_LEASE_LABEL_KEY}={claim_sha256}"


def _docker_lease_control_environment(
    request: TransactionRequest,
    claim: authorization.ClaimRecord,
    stage: str,
) -> dict[str, str]:
    return {
        **STAGE_BASE_ENVIRONMENT,
        "ULLM_CAMPAIGN_TRANSACTION_STAGE": stage,
        DOCKER_LEASE_WRAPPER_ENVIRONMENT: os.fspath(
            _docker_lease_wrapper(request.source_root)
        ),
        DOCKER_LEASE_LABEL_ENVIRONMENT: _docker_lease_label(claim),
    }


def _run_docker_lease_control(
    request: TransactionRequest,
    claim: authorization.ClaimRecord,
    arguments: Sequence[str],
    *,
    runner: CommandRunner,
    stage: str,
    timeout_seconds: float = DOCKER_LEASE_CONTROL_TIMEOUT_SECONDS,
) -> str:
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or timeout_seconds > DOCKER_LEASE_CONTROL_TIMEOUT_SECONDS
    ):
        raise CommandContainmentLost(
            "campaign Docker lease control timeout is invalid"
        )
    command = [DOCKER_BINARY, *arguments]
    try:
        completed = runner(
            command,
            cwd=request.source_root,
            env=_docker_lease_control_environment(request, claim, stage),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=float(timeout_seconds),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CommandContainmentLost(
            "campaign Docker lease control failed"
        ) from error
    if completed.returncode != 0 or not isinstance(completed.stdout, str):
        raise CommandContainmentLost("campaign Docker lease control failed")
    if len(completed.stdout.encode("utf-8")) > (
        MAX_DOCKER_LEASE_CONTAINERS * 65
    ):
        raise CommandContainmentLost("campaign Docker lease inventory is too large")
    return completed.stdout


def _docker_lease_container_ids(
    request: TransactionRequest,
    claim: authorization.ClaimRecord,
    *,
    runner: CommandRunner,
    stage: str,
    timeout_seconds: float = DOCKER_LEASE_CONTROL_TIMEOUT_SECONDS,
) -> tuple[str, ...]:
    raw = _run_docker_lease_control(
        request,
        claim,
        (
            "container",
            "ls",
            "--all",
            "--quiet",
            "--no-trunc",
            "--filter",
            f"label={_docker_lease_label(claim)}",
        ),
        runner=runner,
        stage=stage,
        timeout_seconds=timeout_seconds,
    )
    identifiers = tuple(line for line in raw.splitlines() if line)
    if (
        len(identifiers) > MAX_DOCKER_LEASE_CONTAINERS
        or len(set(identifiers)) != len(identifiers)
        or any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in identifiers)
    ):
        raise CommandContainmentLost(
            "campaign Docker lease inventory is malformed"
        )
    return identifiers


def _require_empty_docker_lease(
    request: TransactionRequest,
    claim: authorization.ClaimRecord,
    *,
    runner: CommandRunner,
    stage: str,
) -> None:
    _settle_docker_lease(
        request,
        claim,
        runner=runner,
        stage=stage,
        remove=False,
    )


def _cleanup_docker_lease(
    request: TransactionRequest,
    claim: authorization.ClaimRecord,
    *,
    runner: CommandRunner,
    stage: str,
) -> None:
    _settle_docker_lease(
        request,
        claim,
        runner=runner,
        stage=stage,
        remove=True,
    )


def _settle_docker_lease(
    request: TransactionRequest,
    claim: authorization.ClaimRecord,
    *,
    runner: CommandRunner,
    stage: str,
    remove: bool,
) -> None:
    monotonic = getattr(runner, "docker_lease_monotonic", time.monotonic)
    sleeper = getattr(runner, "docker_lease_sleep", time.sleep)
    if not callable(monotonic) or not callable(sleeper):
        raise CommandContainmentLost(
            "campaign Docker lease quiescence clock is invalid"
        )

    def now() -> float:
        try:
            value = monotonic()
        except BaseException as error:
            raise CommandContainmentLost(
                "campaign Docker lease quiescence clock failed"
            ) from error
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise CommandContainmentLost(
                "campaign Docker lease quiescence clock is invalid"
            )
        return float(value)

    started = now()
    deadline = started + DOCKER_LEASE_CLEANUP_TOTAL_TIMEOUT_SECONDS
    quiet_since: float | None = None
    empty_polls = 0
    polls = 0
    prior = started
    while True:
        current = now()
        if current < prior or current >= deadline:
            raise CommandContainmentLost(
                "campaign Docker lease quiescence deadline expired"
            )
        remaining = deadline - current
        identifiers = _docker_lease_container_ids(
            request,
            claim,
            runner=runner,
            stage=stage,
            timeout_seconds=min(
                DOCKER_LEASE_CONTROL_TIMEOUT_SECONDS,
                remaining,
            ),
        )
        observed = now()
        if observed < current or observed > deadline:
            raise CommandContainmentLost(
                "campaign Docker lease quiescence deadline expired"
            )
        prior = observed
        polls += 1
        if polls > MAX_DOCKER_LEASE_QUIESCENCE_POLLS:
            raise CommandContainmentLost(
                "campaign Docker lease quiescence poll bound exceeded"
            )
        if identifiers:
            if not remove:
                raise CommandContainmentLost(
                    "campaign Docker lease was not empty before execution"
                )
            remaining = deadline - observed
            if remaining <= 0:
                raise CommandContainmentLost(
                    "campaign Docker lease quiescence deadline expired"
                )
            _run_docker_lease_control(
                request,
                claim,
                ("container", "rm", "--force", *identifiers),
                runner=runner,
                stage=stage,
                timeout_seconds=min(
                    DOCKER_LEASE_CONTROL_TIMEOUT_SECONDS,
                    remaining,
                ),
            )
            after_remove = now()
            if after_remove < observed or after_remove > deadline:
                raise CommandContainmentLost(
                    "campaign Docker lease quiescence deadline expired"
                )
            prior = after_remove
            quiet_since = None
            empty_polls = 0
            continue

        if quiet_since is None:
            quiet_since = observed
        empty_polls += 1
        if (
            empty_polls >= DOCKER_LEASE_MINIMUM_EMPTY_POLLS
            and observed - quiet_since >= DOCKER_LEASE_QUIESCENCE_SECONDS
        ):
            return
        remaining = deadline - observed
        if remaining <= 0:
            raise CommandContainmentLost(
                "campaign Docker lease quiescence deadline expired"
            )
        sleep_seconds = min(
            DOCKER_LEASE_QUIESCENCE_POLL_SECONDS,
            remaining,
        )
        try:
            sleeper(sleep_seconds)
        except BaseException as error:
            raise CommandContainmentLost(
                "campaign Docker lease quiescence sleep failed"
            ) from error


def _run_git(
    source_root: Path,
    arguments: Sequence[str],
    *,
    runner: CommandRunner,
) -> str:
    try:
        completed = runner(
            campaign_source_seal.git_argv(list(arguments)),
            cwd=source_root,
            env=campaign_source_seal.git_environment(),
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
    require_detached: bool = False,
) -> tuple[str, str]:
    commit, tree, _seal = _sealed_source_identity(
        source_root,
        runner=runner,
        required_uid=os.geteuid(),
        require_detached=require_detached,
    )
    return commit, tree


def _sealed_source_identity(
    source_root: Path,
    *,
    runner: CommandRunner,
    required_uid: int,
    require_detached: bool = False,
    expected_seal: campaign_source_seal.SourceSeal | None = None,
) -> tuple[str, str, campaign_source_seal.SourceSeal]:
    try:
        if expected_seal is None:
            sealed = campaign_source_seal.capture_source_seal(
                source_root,
                required_uid=required_uid,
            )
        else:
            if expected_seal.root != source_root:
                raise campaign_source_seal.SourceSealError(
                    "campaign source root differs from its seal"
                )
            sealed = campaign_source_seal.require_source_seal(
                expected_seal,
                required_uid=required_uid,
            )
    except campaign_source_seal.SourceSealError as error:
        raise TransactionError("campaign source is not sealed") from error
    root = sealed.root
    top = Path(
        _run_git(root, ("rev-parse", "--show-toplevel"), runner=runner)
    )
    if top != root:
        raise TransactionError("source root differs from Git top-level")
    commit = _run_git(root, ("rev-parse", "HEAD"), runner=runner)
    tree = _run_git(root, ("rev-parse", "HEAD^{tree}"), runner=runner)
    if GIT_RE.fullmatch(commit) is None or GIT_RE.fullmatch(tree) is None:
        raise TransactionError("source identity is not a full Git object ID")
    status = _run_git(
        root,
        (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=all",
            "--no-renames",
        ),
        runner=runner,
    )
    if status:
        raise TransactionError("campaign source worktree is not clean")
    if require_detached:
        branch = _run_git(
            root,
            ("rev-parse", "--abbrev-ref", "HEAD"),
            runner=runner,
        )
        if branch != "HEAD":
            raise TransactionError("AQ4 campaign source is not detached")
    try:
        campaign_source_seal.require_source_seal(
            sealed,
            required_uid=required_uid,
        )
    except campaign_source_seal.SourceSealError as error:
        raise TransactionError(
            "campaign source changed during Git identity read"
        ) from error
    return commit, tree, sealed


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


def _validate_candidate_promotion_receipt(
    candidate_document: dict[str, Any],
    *,
    receipt_path: Path,
    source_root: Path,
) -> None:
    """Require the production SQ8 receipt and its complete evidence binding."""

    try:
        receipt, evidence = sq8_promotion.validate_receipt(
            receipt_path,
            source_root=source_root,
            verify_live_source=True,
        )
        if receipt.get("schema_version") != sq8_promotion.RECEIPT_SCHEMA:
            raise sq8_promotion.PromotionError(
                "candidate SQ8 promotion receipt schema differs"
            )
        evidence_reference = receipt.get("evidence")
        profile_reference = evidence.get("profile")
        worker = candidate_document.get("worker")
        promotion = candidate_document.get("promotion")
        if (
            not isinstance(evidence_reference, dict)
            or set(evidence_reference) != {"path", "sha256"}
            or not isinstance(evidence_reference.get("path"), str)
            or not isinstance(profile_reference, dict)
            or not isinstance(profile_reference.get("path"), str)
            or not isinstance(worker, dict)
            or not isinstance(worker.get("binary"), str)
            or not isinstance(worker.get("binary_sha256"), str)
            or not isinstance(promotion, dict)
            or not isinstance(promotion.get("source_commit"), str)
        ):
            raise sq8_promotion.PromotionError(
                "candidate SQ8 promotion evidence binding differs"
            )
        sq8_promotion.validate_generator_binding(
            evidence_path=receipt_path.parent / evidence_reference["path"],
            receipt=receipt,
            receipt_path=receipt_path,
            profile_path=Path(profile_reference["path"]),
            source_commit=promotion["source_commit"],
            source_root=source_root,
            worker_binary=Path(worker["binary"]),
            worker_sha256=worker["binary_sha256"],
            manifest=candidate_document,
        )
    except (OSError, sq8_promotion.PromotionError) as error:
        raise TransactionError(
            "candidate SQ8 production promotion receipt is invalid"
        ) from error


def _aq4_promotion_identity(
    active_document: dict[str, Any],
    *,
    manifest_parent: Path,
    authorization_document: dict[str, Any],
) -> tuple[Path, str, Path, str]:
    before = authorization_document["before"]
    aq4_release = authorization_document["aq4_release"]
    receipt_path, receipt_sha256 = _promotion_identity(
        active_document,
        manifest_parent=manifest_parent,
        source_commit=before["promotion_source_commit"],
        label="active AQ4",
    )
    expected_receipt = Path(before["promotion_receipt_path"])
    evidence_reference = aq4_release["promotion_evidence"]
    expected_evidence = Path(evidence_reference["source_path"])
    if (
        receipt_path != expected_receipt
        or receipt_sha256 != before["promotion_receipt_sha256"]
        or aq4_release["promotion_receipt"]["source_path"]
        != os.fspath(expected_receipt)
        or aq4_release["promotion_receipt"]["sha256"] != receipt_sha256
    ):
        raise TransactionError("active AQ4 promotion receipt identity differs")
    receipt = _read_input(
        receipt_path,
        "active AQ4 promotion receipt",
        MAX_INPUT_BYTES,
    )
    if receipt.sha256 != receipt_sha256:
        raise TransactionError("active AQ4 promotion receipt bytes differ")
    receipt_document = _strict_object(
        receipt.raw,
        "active AQ4 promotion receipt",
    )
    reference = receipt_document.get("evidence")
    if (
        set(receipt_document)
        != {"schema_version", "source_commit", "evidence"}
        or receipt_document.get("schema_version")
        != "ullm.aq4_resident_promotion.v1"
        or receipt_document.get("source_commit")
        != before["promotion_source_commit"]
        or not isinstance(reference, dict)
        or set(reference) != {"path", "sha256"}
        or reference.get("sha256") != evidence_reference["sha256"]
    ):
        raise TransactionError("active AQ4 promotion receipt lineage differs")
    raw_evidence_path = reference["path"]
    if not isinstance(raw_evidence_path, str) or not raw_evidence_path:
        raise TransactionError("active AQ4 promotion evidence path is invalid")
    referenced_path = Path(raw_evidence_path)
    if (
        referenced_path.is_absolute()
        or any(part in {"", ".", ".."} for part in referenced_path.parts)
    ):
        raise TransactionError(
            "active AQ4 promotion evidence reference is unsafe"
        )
    destination_receipt = Path(
        aq4_release["promotion_receipt"]["path"]
    )
    destination_evidence = Path(
        aq4_release["promotion_evidence"]["path"]
    )
    if destination_receipt.parent / referenced_path != destination_evidence:
        raise TransactionError(
            "AQ4 promotion copy paths do not preserve receipt lineage"
        )
    referenced_path = receipt_path.parent / referenced_path
    try:
        referenced_path = referenced_path.resolve(strict=True)
        expected_evidence_resolved = expected_evidence.resolve(strict=True)
    except OSError as error:
        raise TransactionError(
            "active AQ4 promotion evidence is unavailable"
        ) from error
    if referenced_path != expected_evidence_resolved:
        raise TransactionError("active AQ4 promotion evidence path differs")
    evidence = _read_input(
        expected_evidence,
        "active AQ4 promotion evidence",
        MAX_INPUT_BYTES,
    )
    if evidence.sha256 != evidence_reference["sha256"]:
        raise TransactionError("active AQ4 promotion evidence bytes differ")
    evidence_document = _strict_object(
        evidence.raw,
        "active AQ4 promotion evidence",
    )
    if (
        evidence_document.get("schema_version")
        != "ullm.aq4_resident_promotion_evidence.v1"
        or evidence_document.get("source_commit")
        != before["promotion_source_commit"]
        or evidence_document.get("worker_binary")
        != before["worker_binary_path"]
        or evidence_document.get("worker_binary_sha256")
        != before["worker_binary_sha256"]
        or evidence_document.get("verified") is not True
        or evidence_document.get("production_receipt_written") is not False
    ):
        raise TransactionError("active AQ4 promotion evidence identity differs")
    return (
        receipt_path,
        receipt.sha256,
        expected_evidence,
        evidence.sha256,
    )


def _read_input(path: Path, label: str, maximum: int) -> StableFileSnapshot:
    try:
        return stable_read_regular(path, label, maximum=maximum)
    except Exception as error:
        raise TransactionError(f"{label} is unavailable or changed") from error


@dataclass(frozen=True, slots=True)
class _ManifestRuntimeSeals:
    worker: campaign_runtime_seal.RuntimeArtifactSeal
    promotion_receipt: campaign_runtime_seal.RuntimeArtifactSeal
    artifacts: tuple[campaign_runtime_seal.RuntimeArtifactSeal, ...]
    trees: tuple[campaign_runtime_seal.RuntimeTreeSeal, ...]


@dataclass(frozen=True, slots=True)
class _CommandRuntimeSeals:
    candidate: tuple[campaign_runtime_seal.RuntimeArtifactSeal, ...]
    aq4: tuple[campaign_runtime_seal.RuntimeArtifactSeal, ...]
    shared: tuple[campaign_runtime_seal.RuntimeArtifactSeal, ...]


def _runtime_path(
    raw: Any,
    *,
    base: Path,
    label: str,
    relative_only: bool = False,
) -> Path:
    if (
        not isinstance(raw, str)
        or not raw
        or "\x00" in raw
        or len(raw.encode("utf-8")) > 4_096
    ):
        raise TransactionError(f"{label} path is invalid")
    selected = Path(raw)
    if os.fspath(selected) != raw or any(
        part in {"", ".", ".."} for part in selected.parts
    ):
        raise TransactionError(f"{label} path is not lexical")
    if relative_only:
        if selected.is_absolute():
            raise TransactionError(f"{label} path is not relative")
        selected = base / selected
    elif not selected.is_absolute():
        selected = base / selected
    try:
        return campaign_runtime_seal._lexical_absolute(selected)
    except campaign_runtime_seal.RuntimeArtifactSealError as error:
        raise TransactionError(f"{label} path is not lexical absolute") from error


def _capture_runtime_artifact(
    path: Path,
    *,
    label: str,
    maximum: int,
    required_uid: int,
) -> campaign_runtime_seal.RuntimeArtifactSeal:
    try:
        return campaign_runtime_seal.capture_runtime_artifact_seal(
            path,
            label=label,
            maximum=maximum,
            required_uid=required_uid,
        )
    except campaign_runtime_seal.RuntimeArtifactSealError as error:
        raise TransactionError(f"{label} runtime artifact is not sealed") from error


def _capture_runtime_tree(
    root: Path,
    *,
    label: str,
    required_uid: int,
) -> campaign_runtime_seal.RuntimeTreeSeal:
    try:
        return campaign_runtime_seal.capture_runtime_tree_seal(
            root,
            label=label,
            required_uid=required_uid,
        )
    except campaign_runtime_seal.RuntimeArtifactSealError as error:
        raise TransactionError(f"{label} runtime tree is not sealed") from error


def _command_executable_paths(
    groups: Sequence[Sequence[Sequence[str]]],
    *,
    label: str,
) -> set[Path]:
    paths: set[Path] = set()
    for group in groups:
        for command in group:
            if not Path(command[0]).is_absolute():
                raise TransactionError(
                    f"{label} command executable is not absolute"
                )
            executable = _runtime_path(
                command[0],
                base=Path("/"),
                label=f"{label} command executable",
            )
            paths.add(executable)
            for index, argument in enumerate(command[:-1]):
                if argument not in NESTED_EXECUTABLE_FLAGS:
                    continue
                if not Path(command[index + 1]).is_absolute():
                    raise TransactionError(
                        f"{label} nested executable for {argument} "
                        "is not absolute"
                    )
                paths.add(
                    _runtime_path(
                        command[index + 1],
                        base=Path("/"),
                        label=(
                            f"{label} nested executable for {argument}"
                        ),
                    )
                )
    return paths


def _capture_command_executable(
    path: Path,
    *,
    label: str,
    required_uid: int,
) -> campaign_runtime_seal.RuntimeArtifactSeal:
    try:
        owner = path.lstat().st_uid
    except OSError as error:
        raise TransactionError(f"{label} is unavailable") from error
    if owner not in {0, required_uid}:
        raise TransactionError(f"{label} owner is untrusted")
    sealed = _capture_runtime_artifact(
        path,
        label=label,
        maximum=MAX_OUTPUT_FILE_BYTES,
        required_uid=owner,
    )
    if not stat.S_IMODE(sealed.snapshot.identity.mode) & 0o111:
        raise TransactionError(f"{label} is not directly executable")
    return sealed


def _capture_command_runtime_seals(
    commands: TransactionCommands,
    *,
    required_uid: int,
    recovery_only: bool = False,
) -> _CommandRuntimeSeals:
    candidate_groups: tuple[Sequence[Sequence[str]], ...]
    if recovery_only:
        candidate_groups = ()
        aq4_groups: tuple[Sequence[Sequence[str]], ...] = (
            commands.reverse_reconciliation,
            commands.final_checks,
        )
    else:
        candidate_groups = (
            commands.candidate_reconciliation,
            commands.candidate_checks,
            (commands.sq8_full,),
            (commands.reasoning_release,),
            (commands.reasoning_browser,),
        )
        aq4_groups = (
            commands.reverse_reconciliation,
            commands.aq4_reasoning_release,
            commands.aq4_reasoning_browser,
            commands.aq4_bundle,
            commands.final_checks,
        )
    candidate_paths = _command_executable_paths(
        candidate_groups,
        label="candidate",
    )
    aq4_paths = _command_executable_paths(aq4_groups, label="AQ4")
    # The wrapper itself is carried by --docker in every campaign command.
    # Its backend is not visible in those vectors, so pin the real Docker CLI
    # explicitly in both normal execution and recovery.
    aq4_paths.update({Path(PYTHON_BINARY), Path(DOCKER_BINARY)})
    if not recovery_only:
        # The transaction adds these two source-bound verifier invocations
        # around both browser campaigns even though they are not stored in the
        # request command vectors.
        candidate_paths.update(
            {Path(PYTHON_BINARY), Path(DOCKER_BINARY)}
        )
    shared_paths = (
        candidate_paths & aq4_paths
    ) | {Path(GIT_COMMAND_PREFIX[0])}
    candidate_paths -= shared_paths
    aq4_paths -= shared_paths

    def capture(
        selected: set[Path],
        scope: str,
    ) -> tuple[campaign_runtime_seal.RuntimeArtifactSeal, ...]:
        return tuple(
            _capture_command_executable(
                path,
                label=f"{scope} command executable {path}",
                required_uid=required_uid,
            )
            for path in sorted(selected, key=lambda value: os.fsencode(value))
        )

    return _CommandRuntimeSeals(
        candidate=capture(candidate_paths, "candidate"),
        aq4=capture(aq4_paths, "AQ4"),
        shared=capture(shared_paths, "shared"),
    )


def _runtime_sha256(raw: Any, label: str) -> str:
    if (
        not isinstance(raw, str)
        or authorization.HASH_RE.fullmatch(raw) is None
    ):
        raise TransactionError(f"{label} is not a lowercase SHA-256")
    return raw


def _manifest_runtime_seals(
    document: dict[str, Any],
    *,
    manifest_path: Path,
    expected_receipt_path: Path,
    label: str,
    required_uid: int,
) -> _ManifestRuntimeSeals:
    worker = document.get("worker")
    tokenizer = document.get("tokenizer")
    product = document.get("product")
    promotion = document.get("promotion")
    if not all(
        isinstance(value, dict)
        for value in (worker, tokenizer, product, promotion)
    ):
        raise TransactionError(f"{label} runtime contract is incomplete")
    assert isinstance(worker, dict)
    assert isinstance(tokenizer, dict)
    assert isinstance(product, dict)
    assert isinstance(promotion, dict)
    manifest_parent = manifest_path.parent
    worker_path = _runtime_path(
        worker.get("binary"),
        base=manifest_parent,
        label=f"{label} worker",
    )
    receipt_path = _runtime_path(
        promotion.get("receipt"),
        base=manifest_parent,
        label=f"{label} promotion receipt",
    )
    if receipt_path != expected_receipt_path:
        raise TransactionError(f"{label} promotion receipt path differs")

    tokenizer_root = _runtime_path(
        tokenizer.get("root"),
        base=manifest_parent,
        label=f"{label} tokenizer root",
    )
    tokenizer_files = tokenizer.get("files")
    if (
        not isinstance(tokenizer_files, dict)
        or not tokenizer_files
        or len(tokenizer_files) > 128
    ):
        raise TransactionError(f"{label} tokenizer file contract differs")

    product_root = _runtime_path(
        product.get("root"),
        base=manifest_parent,
        label=f"{label} product root",
    )
    package = product.get("package")
    artifact = product.get("artifact")
    if not isinstance(package, dict) or (
        artifact is not None and not isinstance(artifact, dict)
    ):
        raise TransactionError(f"{label} product contract differs")

    file_specs: list[tuple[Path, str, int, str]] = [
        (
            worker_path,
            f"{label} worker binary",
            MAX_OUTPUT_FILE_BYTES,
            _runtime_sha256(
                worker.get("binary_sha256"),
                f"{label} worker binary SHA-256",
            ),
        ),
        (
            receipt_path,
            f"{label} promotion receipt",
            MAX_INPUT_BYTES,
            _runtime_sha256(
                promotion.get("receipt_sha256"),
                f"{label} promotion receipt SHA-256",
            ),
        ),
        (
            _runtime_path(
                package.get("manifest_path"),
                base=product_root,
                label=f"{label} package manifest",
                relative_only=True,
            ),
            f"{label} package manifest",
            MAX_INPUT_BYTES,
            _runtime_sha256(
                package.get("manifest_sha256"),
                f"{label} package manifest SHA-256",
            ),
        ),
    ]
    if isinstance(artifact, dict):
        file_specs.append(
            (
                _runtime_path(
                    artifact.get("manifest_path"),
                    base=product_root,
                    label=f"{label} artifact manifest",
                    relative_only=True,
                ),
                f"{label} artifact manifest",
                MAX_INPUT_BYTES,
                _runtime_sha256(
                    artifact.get("manifest_sha256"),
                    f"{label} artifact manifest SHA-256",
                ),
            )
        )
    if any(not isinstance(relative, str) for relative in tokenizer_files):
        raise TransactionError(f"{label} tokenizer path is invalid")
    for relative, expected_sha256 in sorted(
        tokenizer_files.items(),
        key=lambda item: os.fsencode(item[0]),
    ):
        if not isinstance(relative, str):
            raise TransactionError(f"{label} tokenizer path is invalid")
        file_specs.append(
            (
                _runtime_path(
                    relative,
                    base=tokenizer_root,
                    label=f"{label} tokenizer file",
                    relative_only=True,
                ),
                f"{label} tokenizer file {relative}",
                MAX_OUTPUT_FILE_BYTES,
                _runtime_sha256(
                    expected_sha256,
                    f"{label} tokenizer file {relative} SHA-256",
                ),
            )
        )
    if len(
        {
            path
            for path, _name, _maximum, _expected_sha256 in file_specs
        }
    ) != len(file_specs):
        raise TransactionError(f"{label} runtime file paths are not distinct")

    captured: list[campaign_runtime_seal.RuntimeArtifactSeal] = []
    for path, file_label, maximum, expected_sha256 in file_specs:
        sealed = _capture_runtime_artifact(
            path,
            label=file_label,
            maximum=maximum,
            required_uid=required_uid,
        )
        if sealed.snapshot.sha256 != expected_sha256:
            raise TransactionError(f"{file_label} runtime bytes differ")
        captured.append(sealed)
    artifacts = tuple(captured)
    trees = (
        _capture_runtime_tree(
            tokenizer_root,
            label=f"{label} tokenizer",
            required_uid=required_uid,
        ),
        _capture_runtime_tree(
            product_root,
            label=f"{label} product",
            required_uid=required_uid,
        ),
    )
    return _ManifestRuntimeSeals(
        worker=artifacts[0],
        promotion_receipt=artifacts[1],
        artifacts=artifacts,
        trees=trees,
    )


def _require_runtime_seal_collections(
    artifacts: Sequence[campaign_runtime_seal.RuntimeArtifactSeal],
    trees: Sequence[campaign_runtime_seal.RuntimeTreeSeal],
    *,
    required_uid: int,
) -> None:
    if not artifacts or not trees:
        raise TransactionError("transaction runtime seals are unavailable")
    try:
        for sealed in artifacts:
            sealed_uid = sealed.required_uid
            path = sealed.snapshot.path
            mixed_owner_allowed = (
                (
                    "command executable " in sealed.label
                    and sealed_uid in {0, required_uid}
                )
                or (
                    path == FIXED_GATEWAY_API_KEY_PATH
                    and sealed_uid == FIXED_GATEWAY_API_KEY_UID
                )
                or (
                    path == FIXED_OPENWEBUI_SESSION_TOKEN_PATH
                    and sealed_uid == FIXED_OPENWEBUI_SESSION_TOKEN_UID
                )
            )
            if sealed_uid != required_uid and not mixed_owner_allowed:
                raise TransactionError(
                    "transaction runtime artifact seal owner scope differs"
                )
            campaign_runtime_seal.require_runtime_artifact_seal(
                sealed,
                required_uid=sealed_uid,
            )
        for sealed in trees:
            if sealed.required_uid != required_uid:
                raise TransactionError(
                    "transaction runtime tree seal owner scope differs"
                )
            campaign_runtime_seal.require_runtime_tree_seal(
                sealed,
                required_uid=required_uid,
            )
    except campaign_runtime_seal.RuntimeArtifactSealError as error:
        raise TransactionError(
            "transaction runtime artifact seal changed"
        ) from error


def _require_runtime_seals(
    preflight_result: TransactionPreflight,
    *,
    required_uid: int,
    scope: str = "all",
) -> None:
    expected_artifacts = (
        *preflight_result.candidate_runtime_artifact_seals,
        *preflight_result.aq4_runtime_artifact_seals,
        *preflight_result.aq4_release_runtime_artifact_seals,
        *preflight_result.shared_runtime_artifact_seals,
    )
    expected_trees = (
        *preflight_result.candidate_runtime_tree_seals,
        *preflight_result.aq4_runtime_tree_seals,
        *preflight_result.aq4_release_runtime_tree_seals,
        *preflight_result.shared_runtime_tree_seals,
    )
    if (
        preflight_result.runtime_artifact_seals != expected_artifacts
        or preflight_result.runtime_tree_seals != expected_trees
    ):
        raise TransactionError(
            "transaction runtime seal classification differs"
        )
    if scope == "all":
        if (
            not preflight_result.candidate_runtime_artifact_seals
            or not preflight_result.candidate_runtime_tree_seals
        ):
            raise TransactionError(
                "candidate transaction runtime seals are unavailable"
            )
        artifacts = expected_artifacts
        trees = expected_trees
    elif scope == "aq4":
        artifacts = (
            *preflight_result.aq4_runtime_artifact_seals,
            *preflight_result.shared_runtime_artifact_seals,
        )
        trees = (
            *preflight_result.aq4_runtime_tree_seals,
            *preflight_result.shared_runtime_tree_seals,
        )
    elif scope == "aq4_release":
        if not preflight_result.aq4_release_runtime_artifact_seals:
            raise TransactionError(
                "AQ4 release transaction runtime seals are unavailable"
            )
        artifacts = (
            *preflight_result.aq4_runtime_artifact_seals,
            *preflight_result.aq4_release_runtime_artifact_seals,
            *preflight_result.shared_runtime_artifact_seals,
        )
        trees = (
            *preflight_result.aq4_runtime_tree_seals,
            *preflight_result.aq4_release_runtime_tree_seals,
            *preflight_result.shared_runtime_tree_seals,
        )
    else:
        raise TransactionError("transaction runtime seal scope is invalid")
    if (
        not preflight_result.aq4_runtime_artifact_seals
        or not preflight_result.aq4_runtime_tree_seals
        or not preflight_result.shared_runtime_artifact_seals
    ):
        raise TransactionError("AQ4 transaction runtime seals are unavailable")
    _require_runtime_seal_collections(
        artifacts,
        trees,
        required_uid=required_uid,
    )


def _open_command_executable(
    preflight_result: TransactionPreflight,
    command: Sequence[str],
    *,
    scope: str,
) -> int:
    path = Path(command[0])
    if scope == "all":
        artifacts = (
            *preflight_result.candidate_runtime_artifact_seals,
            *preflight_result.aq4_runtime_artifact_seals,
            *preflight_result.aq4_release_runtime_artifact_seals,
            *preflight_result.shared_runtime_artifact_seals,
        )
    elif scope in {"aq4", "aq4_release"}:
        artifacts = (
            *preflight_result.aq4_runtime_artifact_seals,
            *(
                preflight_result.aq4_release_runtime_artifact_seals
                if scope == "aq4_release"
                else ()
            ),
            *preflight_result.shared_runtime_artifact_seals,
        )
    else:
        raise TransactionError("transaction runtime seal scope is invalid")
    matches = tuple(
        sealed
        for sealed in artifacts
        if "command executable " in sealed.label
        and sealed.snapshot.path == path
    )
    if len(matches) != 1:
        raise TransactionError(
            "transaction command executable seal is unavailable"
        )
    sealed = matches[0]
    try:
        return campaign_runtime_seal.open_runtime_artifact_seal(
            sealed,
            required_uid=sealed.required_uid,
        )
    except campaign_runtime_seal.RuntimeArtifactSealError as error:
        raise TransactionError(
            "transaction command executable changed before descriptor pin"
        ) from error


def _validate_private_secret(
    path: Path,
    label: str,
    *,
    required_uid: int,
) -> str:
    if path == FIXED_OPENWEBUI_SESSION_TOKEN_PATH:
        _validate_fixed_session_token_parent(path)
    snapshot = _read_input(path, label, 65_536)
    mode = stat.S_IMODE(snapshot.identity.mode)
    if path == FIXED_GATEWAY_API_KEY_PATH:
        metadata_safe = (
            snapshot.identity.uid == FIXED_GATEWAY_API_KEY_UID
            and snapshot.identity.gid == FIXED_GATEWAY_API_KEY_GID
            and mode == FIXED_GATEWAY_API_KEY_MODE
        )
    elif path == FIXED_OPENWEBUI_SESSION_TOKEN_PATH:
        metadata_safe = (
            snapshot.identity.uid == FIXED_OPENWEBUI_SESSION_TOKEN_UID
            and snapshot.identity.gid == FIXED_OPENWEBUI_SESSION_TOKEN_GID
            and mode == FIXED_OPENWEBUI_SESSION_TOKEN_MODE
        )
    else:
        metadata_safe = (
            snapshot.identity.uid == required_uid and mode == 0o600
        )
    if (
        not metadata_safe
        or snapshot.identity.links != 1
        or not snapshot.raw.rstrip(b"\r\n")
    ):
        raise TransactionError(f"{label} metadata is unsafe")
    return snapshot.sha256


def _validate_fixed_session_token_parent(path: Path) -> None:
    if path != FIXED_OPENWEBUI_SESSION_TOKEN_PATH:
        raise TransactionError("OpenWebUI session token path is not fixed")
    parent_descriptor = -1
    try:
        parent_descriptor, _identity = _open_parent_descriptor(
            path,
            "OpenWebUI session token",
            required_uid=FIXED_OPENWEBUI_SESSION_TOKEN_PARENT_UID,
        )
        metadata = os.fstat(parent_descriptor)
        if (
            path.parent != FIXED_OPENWEBUI_SESSION_TOKEN_PARENT
            or metadata.st_uid
            != FIXED_OPENWEBUI_SESSION_TOKEN_PARENT_UID
            or metadata.st_gid
            != FIXED_OPENWEBUI_SESSION_TOKEN_PARENT_GID
            or stat.S_IMODE(metadata.st_mode)
            != FIXED_OPENWEBUI_SESSION_TOKEN_PARENT_MODE
            or metadata.st_mode & (stat.S_ISUID | stat.S_ISGID)
        ):
            raise TransactionError(
                "OpenWebUI session token parent metadata is unsafe"
            )
        _require_service_entry_xattrs(parent_descriptor)
    except OSError as error:
        raise TransactionError(
            "OpenWebUI session token parent is unavailable"
        ) from error
    finally:
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def _private_secret_seal_uid(path: Path, *, required_uid: int) -> int:
    if path == FIXED_GATEWAY_API_KEY_PATH:
        return FIXED_GATEWAY_API_KEY_UID
    if path == FIXED_OPENWEBUI_SESSION_TOKEN_PATH:
        return FIXED_OPENWEBUI_SESSION_TOKEN_UID
    return required_uid


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
            reloaded_claim = authorization.load_live_claim(
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
    _require_docker_lease_wrapper_commands(
        request.commands,
        request.source_root,
    )
    command_runtime = _capture_command_runtime_seals(
        request.commands,
        required_uid=policy.required_uid,
    )
    wrapper_path = _docker_lease_wrapper(request.source_root)
    wrapper_seals = tuple(
        sealed
        for sealed in command_runtime.shared
        if sealed.snapshot.path == wrapper_path
    )
    if len(wrapper_seals) != 1:
        raise TransactionError(
            "shared Docker lease wrapper runtime seal is unavailable"
        )
    source_commit, source_tree, source_seal = _sealed_source_identity(
        request.source_root,
        runner=runner,
        required_uid=policy.required_uid,
    )
    aq4_source_root = Path(auth.document["aq4_release"]["source"]["root"])
    aq4_source_commit, aq4_source_tree, aq4_source_seal = _sealed_source_identity(
        aq4_source_root,
        runner=runner,
        required_uid=policy.required_uid,
        require_detached=True,
    )
    active = _read_input(
        request.active_manifest,
        "actual active served-model manifest",
        MAX_MANIFEST_BYTES,
    )
    candidate_manifest_seal = _capture_runtime_artifact(
        request.candidate_manifest,
        label="frozen candidate served-model manifest",
        maximum=MAX_MANIFEST_BYTES,
        required_uid=policy.required_uid,
    )
    candidate = candidate_manifest_seal.snapshot
    if (
        active.identity.uid != policy.required_uid
        or active.identity.links != 1
        or stat.S_IMODE(active.identity.mode) != 0o644
    ):
        raise TransactionError("active manifest metadata is unsafe")
    api_key_sha256: str | None = None
    session_token_sha256: str | None = None
    secret_seals: tuple[
        campaign_runtime_seal.RuntimeArtifactSeal, ...
    ] = ()
    secret_paths = (
        request.api_key_file,
        request.openwebui_session_token_file,
    )
    if any(value is not None for value in secret_paths):
        if any(value is None for value in secret_paths):
            raise TransactionError("transaction private credential binding is incomplete")
        assert request.api_key_file is not None
        assert request.openwebui_session_token_file is not None
        if policy.required_uid == 0 and (
            request.api_key_file != FIXED_GATEWAY_API_KEY_PATH
            or request.openwebui_session_token_file
            != FIXED_OPENWEBUI_SESSION_TOKEN_PATH
        ):
            raise TransactionError(
                "production transaction credential paths differ"
            )
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
        api_key_seal = _capture_runtime_artifact(
            request.api_key_file,
            label="gateway API key",
            maximum=65_536,
            required_uid=_private_secret_seal_uid(
                request.api_key_file,
                required_uid=policy.required_uid,
            ),
        )
        session_token_seal = _capture_runtime_artifact(
            request.openwebui_session_token_file,
            label="OpenWebUI session token",
            maximum=65_536,
            required_uid=_private_secret_seal_uid(
                request.openwebui_session_token_file,
                required_uid=policy.required_uid,
            ),
        )
        if (
            api_key_seal.snapshot.sha256 != api_key_sha256
            or session_token_seal.snapshot.sha256
            != session_token_sha256
        ):
            raise TransactionError(
                "transaction private credential changed while sealing"
            )
        secret_seals = (api_key_seal, session_token_seal)
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
    before = auth.document["before"]
    (
        aq4_receipt_path,
        aq4_receipt_sha256,
        aq4_evidence_path,
        aq4_evidence_sha256,
    ) = _aq4_promotion_identity(
        active_document,
        manifest_parent=active.path.parent,
        authorization_document=auth.document,
    )
    receipt_path, receipt_sha256 = _promotion_identity(
        candidate_document,
        manifest_parent=candidate.path.parent,
        source_commit=source_commit,
        label="candidate SQ8",
    )
    candidate_runtime = _manifest_runtime_seals(
        candidate_document,
        manifest_path=candidate.path,
        expected_receipt_path=receipt_path,
        label="candidate SQ8",
        required_uid=policy.required_uid,
    )
    aq4_runtime = _manifest_runtime_seals(
        active_document,
        manifest_path=active.path,
        expected_receipt_path=aq4_receipt_path,
        label="active AQ4",
        required_uid=policy.required_uid,
    )
    aq4_evidence_seal = _capture_runtime_artifact(
        aq4_evidence_path,
        label="active AQ4 promotion evidence",
        maximum=MAX_INPUT_BYTES,
        required_uid=policy.required_uid,
    )
    aq4_worker_binary = aq4_runtime.worker.snapshot
    aq4_promotion_receipt = aq4_runtime.promotion_receipt.snapshot
    aq4_promotion_evidence = aq4_evidence_seal.snapshot
    receipt = candidate_runtime.promotion_receipt.snapshot
    if (
        aq4_worker_binary.path != Path(before["worker_binary_path"])
        or aq4_worker_binary.sha256 != before["worker_binary_sha256"]
        or aq4_promotion_receipt.sha256 != aq4_receipt_sha256
        or aq4_promotion_evidence.sha256 != aq4_evidence_sha256
        or receipt.sha256 != receipt_sha256
    ):
        raise TransactionError("served-model runtime bytes differ")
    _validate_candidate_promotion_receipt(
        candidate_document,
        receipt_path=receipt_path,
        source_root=request.source_root,
    )
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
    active_worker_document = active_summary.get("worker")
    if (
        not isinstance(active_worker_document, dict)
        or active_worker_document.get("protocol") != before["worker_protocol"]
        or active_worker_document.get("binary") != before["worker_binary_path"]
    ):
        raise TransactionError("active AQ4 worker path/protocol differs")
    candidate_worker_document = candidate_summary.get("worker")
    if (
        not isinstance(candidate_worker_document, dict)
        or candidate_worker_document.get("binary")
        != os.fspath(candidate_runtime.worker.snapshot.path)
        or candidate_runtime.worker.snapshot.sha256 != candidate_worker
        or aq4_runtime.worker.snapshot != aq4_worker_binary
        or aq4_runtime.promotion_receipt.snapshot
        != aq4_promotion_receipt
        or candidate_runtime.promotion_receipt.snapshot != receipt
    ):
        raise TransactionError("served-model runtime identity differs")
    unit_seal = _capture_runtime_artifact(
        request.systemd_unit,
        label="systemd unit",
        maximum=MAX_INPUT_BYTES,
        required_uid=policy.required_uid,
    )
    environment_seal = _capture_runtime_artifact(
        request.environment_file,
        label="systemd environment file",
        maximum=MAX_INPUT_BYTES,
        required_uid=policy.required_uid,
    )
    unit = unit_seal.snapshot
    environment = environment_seal.snapshot
    rollback = auth.document["rollback"]
    if (
        unit.sha256 != rollback["systemd_unit_sha256"]
        or environment.sha256 != rollback["environment_sha256"]
    ):
        raise TransactionError("rollback unit/environment identity differs")
    candidate_runtime_artifact_seals = (
        candidate_manifest_seal,
        *candidate_runtime.artifacts,
        *command_runtime.candidate,
    )
    aq4_runtime_artifact_seals = (
        *aq4_runtime.artifacts,
        *command_runtime.aq4,
    )
    # Required only by the fresh AQ4 evidence/bundle phase. It is deliberately
    # excluded from both candidate serving and the minimal AQ4 restore/proof
    # scope so either serving route can be recovered without the other.
    aq4_release_runtime_artifact_seals = (aq4_evidence_seal,)
    shared_runtime_artifact_seals = (
        unit_seal,
        environment_seal,
        *secret_seals,
        *command_runtime.shared,
    )
    candidate_runtime_tree_seals = candidate_runtime.trees
    aq4_runtime_tree_seals = aq4_runtime.trees
    shared_runtime_tree_seals: tuple[
        campaign_runtime_seal.RuntimeTreeSeal, ...
    ] = ()
    runtime_artifact_seals = (
        *candidate_runtime_artifact_seals,
        *aq4_runtime_artifact_seals,
        *aq4_release_runtime_artifact_seals,
        *shared_runtime_artifact_seals,
    )
    runtime_tree_seals = (
        *candidate_runtime_tree_seals,
        *aq4_runtime_tree_seals,
        *shared_runtime_tree_seals,
    )
    _require_runtime_seal_collections(
        runtime_artifact_seals,
        runtime_tree_seals,
        required_uid=policy.required_uid,
    )
    try:
        authorization.require_authorization_window_binding(
            auth,
            source_commit=source_commit,
            source_tree=source_tree,
            aq4_source_root=aq4_source_root,
            aq4_source_commit=aq4_source_commit,
            aq4_source_tree=aq4_source_tree,
            before_manifest_sha256=active.sha256,
            before_worker_protocol=active_worker_document["protocol"],
            before_worker_binary_path=Path(active_worker_document["binary"]),
            before_promotion_receipt_path=aq4_receipt_path,
            before_promotion_receipt_sha256=aq4_receipt_sha256,
            aq4_promotion_evidence_path=aq4_evidence_path,
            aq4_promotion_evidence_sha256=aq4_evidence_sha256,
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
        source_seal,
        active,
        candidate,
        active_summary,
        candidate_summary,
        unit.sha256,
        environment.sha256,
        receipt.sha256,
        aq4_source_root,
        aq4_source_commit,
        aq4_source_tree,
        aq4_source_seal,
        aq4_worker_binary,
        aq4_promotion_receipt,
        aq4_promotion_evidence,
        api_key_sha256,
        session_token_sha256,
        runtime_artifact_seals,
        runtime_tree_seals,
        candidate_runtime_artifact_seals,
        candidate_runtime_tree_seals,
        aq4_runtime_artifact_seals,
        aq4_runtime_tree_seals,
        shared_runtime_artifact_seals,
        shared_runtime_tree_seals,
        aq4_release_runtime_artifact_seals,
        (),
    )


def default_inactive_checker(services: Sequence[str]) -> None:
    for service in services:
        try:
            completed = subprocess.run(
                [
                    "/usr/bin/systemctl",
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
    label: str = "authorized AQ4 backup",
    maximum: int = MAX_MANIFEST_BYTES,
) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise TransactionError(f"{label} path is not fresh")
    parent, parent_identity = _open_parent_descriptor(
        path,
        label,
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
                raise TransactionError(f"{label} write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _rename_noreplace(
            temporary_name,
            path.name,
            source_parent_descriptor=parent,
            destination_parent_descriptor=parent,
        )
        published = True
        os.fsync(parent)
        _verify_parent_descriptor(
            path,
            parent_identity,
            required_uid=required_uid,
            label=label,
        )
    except FileExistsError as error:
        raise TransactionError(
            f"{label} publication is not fresh"
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
    backup = _read_input(path, label, maximum)
    if (
        backup.raw != raw
        or backup.identity.links != 1
        or stat.S_IMODE(backup.identity.mode) != mode
        or backup.identity.uid != required_uid
    ):
        raise TransactionError(f"{label} differs after publication")


RENAME_NOREPLACE = 1
AQ4_STAGING_PREFIX = ".ullm-aq4-campaign-stage-"
SERVICE_PRODUCER_STAGING_PREFIX = ".ullm-service-producer-stage-"
SERVICE_PRODUCER_OUTPUT_NAME = "output"


def _rewrite_authorized_output_argument(
    command: Sequence[str],
    *,
    flag: str,
    authorized_path: Path,
    staging_path: Path,
) -> tuple[str, ...]:
    """Replace only one exact source-derived output value in a fixed command."""

    authorized = os.fspath(_lexical_absolute(authorized_path, "authorized output"))
    staging = os.fspath(_lexical_absolute(staging_path, "AQ4 staging output"))
    positions = [
        index
        for index, argument in enumerate(command)
        if argument == flag
    ]
    if (
        len(positions) != 1
        or positions[0] + 1 >= len(command)
        or command[positions[0] + 1] != authorized
        or authorized == staging
    ):
        raise TransactionError(
            "AQ4 producer fixed output argument differs from authorization"
        )
    rewritten = list(command)
    rewritten[positions[0] + 1] = staging
    return tuple(rewritten)


def _rename_noreplace(
    source_name: str,
    destination_name: str,
    *,
    source_parent_descriptor: int,
    destination_parent_descriptor: int,
) -> None:
    """Move one entry with a kernel-enforced destination-absence condition."""

    if (
        not source_name
        or not destination_name
        or "/" in source_name
        or "/" in destination_name
        or "\x00" in source_name
        or "\x00" in destination_name
    ):
        raise TransactionError("AQ4 staged publication names are invalid")
    try:
        function = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as error:
        raise TransactionError(
            "renameat2(RENAME_NOREPLACE) is unavailable"
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
        source_parent_descriptor,
        os.fsencode(source_name),
        destination_parent_descriptor,
        os.fsencode(destination_name),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        message = "authorized AQ4 output publication is not fresh"
    elif error_number in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
        message = "renameat2(RENAME_NOREPLACE) is unsupported"
    else:
        message = "authorized AQ4 output publication failed"
    raise TransactionError(message) from OSError(
        error_number,
        os.strerror(error_number),
    )


def _same_directory_object(
    left: os.stat_result,
    right: os.stat_result,
) -> bool:
    return (
        stat.S_ISDIR(left.st_mode)
        and stat.S_ISDIR(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_uid == right.st_uid
        and left.st_gid == right.st_gid
    )


def _require_stable_parent_metadata(
    descriptor: int,
    initial: os.stat_result,
    *,
    expected_link_delta: int,
    label: str,
) -> os.stat_result:
    try:
        current = os.fstat(descriptor)
    except OSError as error:
        raise TransactionError(f"{label} parent is unavailable") from error
    if (
        not stat.S_ISDIR(current.st_mode)
        or current.st_dev != initial.st_dev
        or current.st_ino != initial.st_ino
        or current.st_uid != initial.st_uid
        or current.st_gid != initial.st_gid
        or current.st_mode != initial.st_mode
        or current.st_nlink != initial.st_nlink + expected_link_delta
    ):
        raise TransactionError(f"{label} parent metadata changed")
    return current


def _clear_owned_directory(
    descriptor: int,
    *,
    root_device: int,
    required_uid: int,
    remaining: list[int],
) -> None:
    try:
        names = os.listdir(descriptor)
    except OSError as error:
        raise TransactionError(
            "AQ4 private staging directory cannot be enumerated"
        ) from error
    for name in names:
        remaining[0] -= 1
        if remaining[0] < 0 or not name or "/" in name or "\x00" in name:
            raise TransactionError(
                "AQ4 private staging cleanup exceeded its safety bound"
            )
        try:
            metadata = os.stat(
                name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise TransactionError(
                "AQ4 private staging member changed during cleanup"
            ) from error
        if metadata.st_uid != required_uid:
            raise TransactionError(
                "AQ4 private staging contains a foreign-owned member"
            )
        if stat.S_ISDIR(metadata.st_mode):
            if metadata.st_dev != root_device:
                raise TransactionError(
                    "AQ4 private staging contains a foreign mount"
                )
            child = -1
            try:
                child = os.open(name, _directory_flags(), dir_fd=descriptor)
                opened = os.fstat(child)
                if not _same_directory_object(metadata, opened):
                    raise TransactionError(
                        "AQ4 private staging directory changed during cleanup"
                    )
                os.fchmod(child, 0o700)
                _clear_owned_directory(
                    child,
                    root_device=root_device,
                    required_uid=required_uid,
                    remaining=remaining,
                )
            finally:
                if child >= 0:
                    os.close(child)
            try:
                os.rmdir(name, dir_fd=descriptor)
            except OSError as error:
                raise TransactionError(
                    "AQ4 private staging directory cannot be removed"
                ) from error
        else:
            try:
                os.unlink(name, dir_fd=descriptor)
            except OSError as error:
                raise TransactionError(
                    "AQ4 private staging member cannot be removed"
                ) from error
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise TransactionError(
            "AQ4 private staging cleanup cannot be synchronized"
        ) from error


@dataclass(slots=True)
class _PrivateStagingRoot:
    path: Path
    name: str
    authorized: Path
    parent_descriptor: int
    parent_identity: tuple[int, int]
    initial_parent_metadata: os.stat_result
    descriptor: int
    initial_metadata: os.stat_result
    required_uid: int
    label: str
    closed: bool = False

    @classmethod
    def create(
        cls,
        authorized_path: Path,
        *,
        required_uid: int,
        label: str,
    ) -> "_PrivateStagingRoot":
        authorized = _lexical_absolute(authorized_path, label)
        parent, parent_identity = _open_parent_descriptor(
            authorized,
            label,
            required_uid=required_uid,
        )
        name = f"{AQ4_STAGING_PREFIX}{secrets.token_hex(16)}"
        initial_parent_metadata = os.fstat(parent)
        descriptor = -1
        try:
            try:
                os.stat(
                    authorized.name,
                    dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            except OSError as error:
                raise TransactionError(
                    "authorized AQ4 output cannot be inspected"
                ) from error
            else:
                raise TransactionError(
                    "authorized AQ4 output is not fresh"
                )
            os.mkdir(name, 0o700, dir_fd=parent)
            descriptor = os.open(name, _directory_flags(), dir_fd=parent)
            os.fchmod(descriptor, 0o700)
            metadata = os.fstat(descriptor)
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if (
                not _same_directory_object(metadata, named)
                or metadata.st_uid != required_uid
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or metadata.st_nlink != 2
            ):
                raise TransactionError(
                    "AQ4 private staging root metadata is unsafe"
                )
            os.fsync(descriptor)
            os.fsync(parent)
            return cls(
                path=authorized.parent / name,
                name=name,
                authorized=authorized,
                parent_descriptor=parent,
                parent_identity=parent_identity,
                initial_parent_metadata=initial_parent_metadata,
                descriptor=descriptor,
                initial_metadata=metadata,
                required_uid=required_uid,
                label=label,
            )
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.rmdir(name, dir_fd=parent)
            except OSError:
                pass
            os.close(parent)
            raise

    def _authorized_directory_delta(self) -> int:
        try:
            metadata = os.stat(
                self.authorized.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return 0
        except OSError as error:
            raise TransactionError(
                "authorized AQ4 output cannot be inspected"
            ) from error
        return int(stat.S_ISDIR(metadata.st_mode))

    def verify(self) -> os.stat_result:
        if self.closed:
            raise TransactionError("AQ4 private staging root is closed")
        try:
            opened = os.fstat(self.descriptor)
            named = os.stat(
                self.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise TransactionError(
                "AQ4 private staging root is unavailable"
            ) from error
        if (
            not _same_directory_object(opened, named)
            or opened.st_dev != self.initial_metadata.st_dev
            or opened.st_ino != self.initial_metadata.st_ino
            or opened.st_uid != self.required_uid
            or stat.S_IMODE(opened.st_mode) != 0o700
        ):
            raise TransactionError("AQ4 private staging root identity differs")
        _require_stable_parent_metadata(
            self.parent_descriptor,
            self.initial_parent_metadata,
            expected_link_delta=1 + self._authorized_directory_delta(),
            label=self.label,
        )
        _verify_parent_descriptor(
            self.path,
            self.parent_identity,
            required_uid=self.required_uid,
            label=self.label,
        )
        return opened

    def output(self, authorized_path: Path) -> Path:
        authorized = _lexical_absolute(authorized_path, self.label)
        if authorized.parent != self.path.parent:
            raise TransactionError(
                "AQ4 private staging/output parent binding differs"
            )
        self.verify()
        output = self.path / authorized.name
        try:
            os.stat(
                authorized.name,
                dir_fd=self.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return output
        except OSError as error:
            raise TransactionError(
                "AQ4 private staging output cannot be inspected"
            ) from error
        raise TransactionError("AQ4 private staging output is not fresh")

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.verify()
            root = os.fstat(self.descriptor)
            _clear_owned_directory(
                self.descriptor,
                root_device=root.st_dev,
                required_uid=self.required_uid,
                remaining=[MAX_OUTPUT_FILES + 128],
            )
            self.verify()
            if os.fstat(self.descriptor).st_nlink != 2:
                raise TransactionError(
                    "AQ4 private staging root is not empty"
                )
            os.close(self.descriptor)
            self.descriptor = -1
            os.rmdir(self.name, dir_fd=self.parent_descriptor)
            os.fsync(self.parent_descriptor)
            _require_stable_parent_metadata(
                self.parent_descriptor,
                self.initial_parent_metadata,
                expected_link_delta=self._authorized_directory_delta(),
                label=self.label,
            )
        finally:
            if self.descriptor >= 0:
                os.close(self.descriptor)
                self.descriptor = -1
            os.close(self.parent_descriptor)
            self.parent_descriptor = -1
            self.closed = True


@dataclass(slots=True)
class _AdoptionBudget:
    entries: int = 0
    total_bytes: int = 0


def _same_opened_entry(
    named: os.stat_result,
    opened: os.stat_result,
) -> bool:
    return (
        named.st_dev == opened.st_dev
        and named.st_ino == opened.st_ino
        and named.st_mode == opened.st_mode
        and named.st_uid == opened.st_uid
        and named.st_gid == opened.st_gid
        and named.st_nlink == opened.st_nlink
        and named.st_size == opened.st_size
        and named.st_mtime_ns == opened.st_mtime_ns
        and named.st_ctime_ns == opened.st_ctime_ns
    )


def _require_service_entry_xattrs(descriptor: int) -> None:
    try:
        if campaign_runtime_seal._has_posix_acl(descriptor):
            raise TransactionError(
                "campaign producer staging entry has a POSIX ACL"
            )
        if campaign_runtime_seal._has_forbidden_security_xattr(descriptor):
            raise TransactionError(
                "campaign producer staging entry has a file capability"
            )
    except campaign_runtime_seal.RuntimeArtifactSealError as error:
        raise TransactionError(
            "campaign producer staging security metadata is unavailable"
        ) from error


def _service_entry_metadata_is_safe(
    metadata: os.stat_result,
    *,
    root_device: int,
    control_uid: int,
    control_gid: int,
    service_uid: int,
    service_gid: int,
) -> bool:
    return (
        metadata.st_dev == root_device
        and metadata.st_uid in {control_uid, service_uid}
        and metadata.st_gid in {control_gid, service_gid}
        and not metadata.st_mode & (stat.S_ISUID | stat.S_ISGID)
        and not stat.S_IMODE(metadata.st_mode) & 0o022
        and not stat.S_ISLNK(metadata.st_mode)
    )


def _adopt_service_entry(
    parent_descriptor: int,
    name: str,
    *,
    root_device: int,
    control_uid: int,
    control_gid: int,
    service_uid: int,
    service_gid: int,
    budget: _AdoptionBudget,
) -> os.stat_result:
    """Validate and adopt one producer-created entry without following names."""

    if not name or "/" in name or "\x00" in name or name in {".", ".."}:
        raise TransactionError("campaign producer staging name is invalid")
    budget.entries += 1
    if budget.entries > MAX_OUTPUT_FILES:
        raise TransactionError("campaign producer staging has too many entries")
    try:
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except OSError as error:
        raise TransactionError(
            "campaign producer staging entry is unavailable"
        ) from error
    if not _service_entry_metadata_is_safe(
        named,
        root_device=root_device,
        control_uid=control_uid,
        control_gid=control_gid,
        service_uid=service_uid,
        service_gid=service_gid,
    ):
        raise TransactionError(
            "campaign producer staging entry metadata is unsafe"
        )
    descriptor = -1
    try:
        if stat.S_ISDIR(named.st_mode):
            descriptor = os.open(
                name,
                _directory_flags(),
                dir_fd=parent_descriptor,
            )
            opened = os.fstat(descriptor)
            if not _same_opened_entry(named, opened):
                raise TransactionError(
                    "campaign producer staging directory changed while opening"
                )
            _require_service_entry_xattrs(descriptor)
            try:
                children = sorted(os.listdir(descriptor), key=os.fsencode)
            except OSError as error:
                raise TransactionError(
                    "campaign producer staging directory cannot be enumerated"
                ) from error
            if len(children) > MAX_OUTPUT_FILES:
                raise TransactionError(
                    "campaign producer staging has too many entries"
                )
            for child in children:
                _adopt_service_entry(
                    descriptor,
                    child,
                    root_device=root_device,
                    control_uid=control_uid,
                    control_gid=control_gid,
                    service_uid=service_uid,
                    service_gid=service_gid,
                    budget=budget,
                )
        elif stat.S_ISREG(named.st_mode):
            if (
                named.st_nlink != 1
                or named.st_size < 0
                or named.st_size > MAX_OUTPUT_FILE_BYTES
            ):
                raise TransactionError(
                    "campaign producer staging file metadata is unsafe"
                )
            budget.total_bytes += named.st_size
            if budget.total_bytes > MAX_OUTPUT_TOTAL_BYTES:
                raise TransactionError(
                    "campaign producer staging byte total is invalid"
                )
            descriptor = os.open(
                name,
                os.O_RDONLY
                | os.O_CLOEXEC
                | os.O_NOFOLLOW
                | os.O_NONBLOCK,
                dir_fd=parent_descriptor,
            )
            opened = os.fstat(descriptor)
            if not _same_opened_entry(named, opened):
                raise TransactionError(
                    "campaign producer staging file changed while opening"
                )
            _require_service_entry_xattrs(descriptor)
        else:
            raise TransactionError(
                "campaign producer staging contains a special file"
            )
        os.fchown(descriptor, control_uid, control_gid)
        os.fsync(descriptor)
        adopted = os.fstat(descriptor)
        named_after = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            adopted.st_dev != named.st_dev
            or adopted.st_ino != named.st_ino
            or adopted.st_uid != control_uid
            or adopted.st_gid != control_gid
            or not _same_opened_entry(adopted, named_after)
        ):
            raise TransactionError(
                "campaign producer staging adoption identity differs"
            )
        return adopted
    except OSError as error:
        raise TransactionError(
            "campaign producer staging adoption failed"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _clear_service_staging_directory(
    descriptor: int,
    *,
    root_device: int,
    remaining: list[int],
) -> None:
    """Remove a reclaimed staging tree without following hostile entries."""

    try:
        names = os.listdir(descriptor)
    except OSError as error:
        raise TransactionError(
            "campaign producer staging cleanup cannot enumerate entries"
        ) from error
    for name in names:
        remaining[0] -= 1
        if (
            remaining[0] < 0
            or not name
            or "/" in name
            or "\x00" in name
            or name in {".", ".."}
        ):
            raise TransactionError(
                "campaign producer staging cleanup exceeded its bound"
            )
        try:
            metadata = os.stat(
                name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise TransactionError(
                "campaign producer staging changed during cleanup"
            ) from error
        if stat.S_ISDIR(metadata.st_mode):
            if metadata.st_dev != root_device:
                raise TransactionError(
                    "campaign producer staging contains a foreign mount"
                )
            child = -1
            try:
                child = os.open(name, _directory_flags(), dir_fd=descriptor)
                opened = os.fstat(child)
                if (
                    opened.st_dev != metadata.st_dev
                    or opened.st_ino != metadata.st_ino
                ):
                    raise TransactionError(
                        "campaign producer staging changed during cleanup"
                    )
                os.fchmod(child, 0o700)
                _clear_service_staging_directory(
                    child,
                    root_device=root_device,
                    remaining=remaining,
                )
            finally:
                if child >= 0:
                    os.close(child)
            os.rmdir(name, dir_fd=descriptor)
        else:
            os.unlink(name, dir_fd=descriptor)
    os.fsync(descriptor)


def _mode_allows_campaign_executor_traversal(
    metadata: os.stat_result,
) -> bool:
    if metadata.st_uid == CAMPAIGN_EXECUTOR_UID:
        return bool(metadata.st_mode & stat.S_IXUSR)
    if metadata.st_gid in set(CAMPAIGN_EXECUTOR_SUPPLEMENTARY_GROUPS):
        return bool(metadata.st_mode & stat.S_IXGRP)
    return bool(metadata.st_mode & stat.S_IXOTH)


def _require_campaign_executor_parent_traversal(path: Path) -> None:
    absolute = _lexical_absolute(path, "campaign producer output")
    descriptor = -1
    try:
        descriptor = os.open(absolute.anchor, _directory_flags())
        root = os.fstat(descriptor)
        if not _mode_allows_campaign_executor_traversal(root):
            raise TransactionError(
                "campaign producer cannot traverse output ancestry"
            )
        for component in absolute.parent.parts[1:]:
            next_descriptor = os.open(
                component,
                _directory_flags(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
            metadata = os.fstat(descriptor)
            if not _mode_allows_campaign_executor_traversal(metadata):
                raise TransactionError(
                    "campaign producer cannot traverse output ancestry"
                )
    except OSError as error:
        raise TransactionError(
            "campaign producer output ancestry is unavailable"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@dataclass(slots=True)
class _ServiceProducerStaging:
    path: Path
    name: str
    actual: Path
    authorized: Path
    expected_kind: str
    parent_descriptor: int
    parent_identity: tuple[int, int]
    descriptor: int
    device: int
    inode: int
    control_uid: int
    control_gid: int
    published_identity: tuple[int, int] | None = None
    published: bool = False
    committed: bool = False
    closed: bool = False
    tree_seal: campaign_runtime_seal.RuntimeTreeSeal | None = None
    artifact_seal: campaign_runtime_seal.RuntimeArtifactSeal | None = None

    @classmethod
    def create(
        cls,
        authorized_path: Path,
        *,
        required_uid: int,
        expected_kind: str,
        label: str,
    ) -> "_ServiceProducerStaging":
        if expected_kind not in {"directory", "file"}:
            raise TransactionError("campaign producer output kind is invalid")
        if required_uid not in {0, CAMPAIGN_EXECUTOR_UID}:
            raise TransactionError(
                "campaign producer control identity is unsupported"
            )
        authorized = _lexical_absolute(authorized_path, label)
        _require_campaign_executor_parent_traversal(authorized)
        parent, parent_identity = _open_parent_descriptor(
            authorized,
            label,
            required_uid=required_uid,
        )
        control_gid = 0 if required_uid == 0 else os.getegid()
        name = (
            f"{SERVICE_PRODUCER_STAGING_PREFIX}{secrets.token_hex(16)}"
        )
        descriptor = -1
        try:
            for entry_name in (authorized.name, name):
                try:
                    os.stat(
                        entry_name,
                        dir_fd=parent,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    pass
                except OSError as error:
                    raise TransactionError(
                        "campaign producer output freshness cannot be inspected"
                    ) from error
                else:
                    raise TransactionError(
                        "campaign producer authorized output is not fresh"
                    )
            os.mkdir(name, 0o700, dir_fd=parent)
            descriptor = os.open(name, _directory_flags(), dir_fd=parent)
            os.fchown(
                descriptor,
                CAMPAIGN_EXECUTOR_UID,
                CAMPAIGN_EXECUTOR_GID,
            )
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
            os.fsync(parent)
            metadata = os.fstat(descriptor)
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if (
                not _same_directory_object(metadata, named)
                or metadata.st_uid != CAMPAIGN_EXECUTOR_UID
                or metadata.st_gid != CAMPAIGN_EXECUTOR_GID
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or metadata.st_nlink != 2
            ):
                raise TransactionError(
                    "campaign producer private staging root is unsafe"
                )
            return cls(
                path=authorized.parent / name,
                name=name,
                actual=authorized.parent
                / name
                / SERVICE_PRODUCER_OUTPUT_NAME,
                authorized=authorized,
                expected_kind=expected_kind,
                parent_descriptor=parent,
                parent_identity=parent_identity,
                descriptor=descriptor,
                device=metadata.st_dev,
                inode=metadata.st_ino,
                control_uid=required_uid,
                control_gid=control_gid,
            )
        except BaseException:
            if descriptor >= 0:
                try:
                    os.fchown(descriptor, required_uid, control_gid)
                    os.fchmod(descriptor, 0o700)
                except OSError:
                    pass
                os.close(descriptor)
            try:
                os.rmdir(name, dir_fd=parent)
            except OSError:
                pass
            os.close(parent)
            raise

    def _verify_named_root(self) -> os.stat_result:
        if self.closed:
            raise TransactionError("campaign producer staging root is closed")
        try:
            opened = os.fstat(self.descriptor)
            named = os.stat(
                self.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise TransactionError(
                "campaign producer staging root is unavailable"
            ) from error
        if (
            opened.st_dev != self.device
            or opened.st_ino != self.inode
            or named.st_dev != self.device
            or named.st_ino != self.inode
            or not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(named.st_mode)
        ):
            raise TransactionError(
                "campaign producer staging root identity differs"
            )
        _verify_parent_descriptor(
            self.authorized,
            self.parent_identity,
            required_uid=self.control_uid,
            label="campaign producer authorized output",
        )
        return opened

    def reclaim_and_adopt(self) -> None:
        self._verify_named_root()
        try:
            os.fchown(self.descriptor, self.control_uid, self.control_gid)
            os.fchmod(self.descriptor, 0o700)
            os.fsync(self.descriptor)
        except OSError as error:
            raise TransactionError(
                "campaign producer staging root adoption failed"
            ) from error
        root = self._verify_named_root()
        if (
            root.st_uid != self.control_uid
            or root.st_gid != self.control_gid
            or stat.S_IMODE(root.st_mode) != 0o700
        ):
            raise TransactionError(
                "campaign producer staging root adoption differs"
            )
        try:
            names = os.listdir(self.descriptor)
        except OSError as error:
            raise TransactionError(
                "campaign producer staging root cannot be enumerated"
            ) from error
        if names != [SERVICE_PRODUCER_OUTPUT_NAME]:
            raise TransactionError(
                "campaign producer staging root layout differs"
            )
        adopted = _adopt_service_entry(
            self.descriptor,
            SERVICE_PRODUCER_OUTPUT_NAME,
            root_device=self.device,
            control_uid=self.control_uid,
            control_gid=self.control_gid,
            service_uid=CAMPAIGN_EXECUTOR_UID,
            service_gid=CAMPAIGN_EXECUTOR_GID,
            budget=_AdoptionBudget(),
        )
        if (
            self.expected_kind == "directory"
            and not stat.S_ISDIR(adopted.st_mode)
        ) or (
            self.expected_kind == "file"
            and not stat.S_ISREG(adopted.st_mode)
        ):
            raise TransactionError(
                "campaign producer staging output kind differs"
            )
        self.refresh_seal()

    def refresh_seal(self) -> None:
        self._verify_named_root()
        try:
            if self.expected_kind == "directory":
                self.tree_seal = (
                    campaign_runtime_seal.capture_runtime_tree_seal(
                        self.actual,
                        label="adopted campaign producer output",
                        required_uid=self.control_uid,
                    )
                )
                self.artifact_seal = None
            else:
                self.artifact_seal = (
                    campaign_runtime_seal.capture_runtime_artifact_seal(
                        self.actual,
                        label="adopted campaign producer output",
                        maximum=MAX_OUTPUT_FILE_BYTES,
                        required_uid=self.control_uid,
                    )
                )
                self.tree_seal = None
        except campaign_runtime_seal.RuntimeArtifactSealError as error:
            raise TransactionError(
                "adopted campaign producer output seal failed"
            ) from error

    def require_seal(self) -> None:
        try:
            if self.expected_kind == "directory":
                if self.tree_seal is None:
                    raise TransactionError(
                        "campaign producer directory seal is unavailable"
                    )
                campaign_runtime_seal.require_runtime_tree_seal(
                    self.tree_seal,
                    required_uid=self.control_uid,
                )
            else:
                if self.artifact_seal is None:
                    raise TransactionError(
                        "campaign producer file seal is unavailable"
                    )
                campaign_runtime_seal.require_runtime_artifact_seal(
                    self.artifact_seal,
                    required_uid=self.control_uid,
                )
        except campaign_runtime_seal.RuntimeArtifactSealError as error:
            raise TransactionError(
                "adopted campaign producer output changed"
            ) from error

    def publish_directory(self) -> None:
        if self.expected_kind != "directory":
            raise TransactionError(
                "campaign producer publication kind differs"
            )
        self.require_seal()
        output_descriptor = -1
        try:
            before = os.stat(
                SERVICE_PRODUCER_OUTPUT_NAME,
                dir_fd=self.descriptor,
                follow_symlinks=False,
            )
            output_descriptor = os.open(
                SERVICE_PRODUCER_OUTPUT_NAME,
                _directory_flags(),
                dir_fd=self.descriptor,
            )
            opened = os.fstat(output_descriptor)
            if not _same_directory_object(before, opened):
                raise TransactionError(
                    "campaign producer directory changed before publication"
                )
            original_mode = stat.S_IMODE(opened.st_mode)
            # Linux may require owner-write permission on a directory whose
            # ``..`` entry changes.  The reclaimed root is private, so grant
            # it only on the pinned descriptor and restore it before expose.
            os.fchmod(
                output_descriptor,
                original_mode | stat.S_IWUSR | stat.S_IXUSR,
            )
            os.fsync(output_descriptor)
            _rename_noreplace(
                SERVICE_PRODUCER_OUTPUT_NAME,
                self.authorized.name,
                source_parent_descriptor=self.descriptor,
                destination_parent_descriptor=self.parent_descriptor,
            )
            self.published_identity = (before.st_dev, before.st_ino)
            self.published = True
            os.fchmod(output_descriptor, original_mode)
            os.fsync(output_descriptor)
            after = os.stat(
                self.authorized.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
            opened_after = os.fstat(output_descriptor)
            if (
                before.st_dev != after.st_dev
                or before.st_ino != after.st_ino
                or not _same_directory_object(after, opened_after)
                or not stat.S_ISDIR(after.st_mode)
                or after.st_uid != self.control_uid
                or stat.S_IMODE(after.st_mode) != original_mode
            ):
                raise TransactionError(
                    "published campaign producer directory identity differs"
                )
            os.fsync(self.descriptor)
            os.fsync(self.parent_descriptor)
        except OSError as error:
            raise TransactionError(
                "campaign producer directory publication failed"
            ) from error
        finally:
            if output_descriptor >= 0:
                os.close(output_descriptor)

    def publish_file(self, *, label: str) -> None:
        if self.expected_kind != "file":
            raise TransactionError(
                "campaign producer publication kind differs"
            )
        self.require_seal()
        published = _publish_staged_file(
            self.actual,
            self.authorized,
            required_uid=self.control_uid,
            label=label,
        )
        self.published_identity = (
            published.identity.device,
            published.identity.inode,
        )
        self.published = True

    def commit(self) -> None:
        if not self.published:
            raise TransactionError(
                "campaign producer output was not published"
            )
        self.committed = True

    def _remove_uncommitted_publication(self) -> None:
        if (
            not self.published
            or self.committed
            or self.published_identity is None
        ):
            return
        try:
            metadata = os.stat(
                self.authorized.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        if (metadata.st_dev, metadata.st_ino) != self.published_identity:
            raise TransactionError(
                "uncommitted campaign publication identity changed"
            )
        if stat.S_ISDIR(metadata.st_mode):
            descriptor = os.open(
                self.authorized.name,
                _directory_flags(),
                dir_fd=self.parent_descriptor,
            )
            try:
                os.fchmod(descriptor, 0o700)
                _clear_service_staging_directory(
                    descriptor,
                    root_device=metadata.st_dev,
                    remaining=[MAX_OUTPUT_FILES + 128],
                )
            finally:
                os.close(descriptor)
            os.rmdir(
                self.authorized.name,
                dir_fd=self.parent_descriptor,
            )
        elif stat.S_ISREG(metadata.st_mode):
            os.unlink(
                self.authorized.name,
                dir_fd=self.parent_descriptor,
            )
        else:
            raise TransactionError(
                "uncommitted campaign publication kind differs"
            )
        os.fsync(self.parent_descriptor)
        self.published = False

    def close(self) -> None:
        if self.closed:
            return
        pending_error: BaseException | None = None
        try:
            try:
                self._remove_uncommitted_publication()
            except BaseException as error:
                pending_error = error
            try:
                self._verify_named_root()
                try:
                    os.fchown(
                        self.descriptor,
                        self.control_uid,
                        self.control_gid,
                    )
                    os.fchmod(self.descriptor, 0o700)
                except OSError as error:
                    raise TransactionError(
                        "campaign producer staging cleanup cannot reclaim root"
                    ) from error
                _clear_service_staging_directory(
                    self.descriptor,
                    root_device=self.device,
                    remaining=[MAX_OUTPUT_FILES + 128],
                )
                os.close(self.descriptor)
                self.descriptor = -1
                os.rmdir(self.name, dir_fd=self.parent_descriptor)
                os.fsync(self.parent_descriptor)
                _verify_parent_descriptor(
                    self.authorized,
                    self.parent_identity,
                    required_uid=self.control_uid,
                    label="campaign producer authorized output",
                )
            except BaseException as error:
                if pending_error is None:
                    pending_error = error
        finally:
            if self.descriptor >= 0:
                os.close(self.descriptor)
                self.descriptor = -1
            os.close(self.parent_descriptor)
            self.parent_descriptor = -1
            self.closed = True
        if pending_error is not None:
            raise pending_error


@contextmanager
def _service_producer_staging(
    authorized_path: Path,
    *,
    required_uid: int,
    expected_kind: str,
    label: str,
) -> Iterator[_ServiceProducerStaging]:
    staging = _ServiceProducerStaging.create(
        authorized_path,
        required_uid=required_uid,
        expected_kind=expected_kind,
        label=label,
    )
    try:
        yield staging
    finally:
        staging.close()


@dataclass(slots=True)
class _PrivateSiblingDirectory:
    path: Path
    name: str
    authorized: Path
    parent_descriptor: int
    parent_identity: tuple[int, int]
    initial_parent_metadata: os.stat_result
    required_uid: int
    label: str
    published: bool = False
    closed: bool = False

    @classmethod
    def create(
        cls,
        authorized_path: Path,
        *,
        required_uid: int,
        label: str,
    ) -> "_PrivateSiblingDirectory":
        authorized = _lexical_absolute(authorized_path, label)
        parent, parent_identity = _open_parent_descriptor(
            authorized,
            label,
            required_uid=required_uid,
        )
        initial = os.fstat(parent)
        name = (
            f"{AQ4_STAGING_PREFIX}{secrets.token_hex(16)}-"
            f"{authorized.name}"
        )
        try:
            for entry_name, entry_label in (
                (name, "staging name"),
                (authorized.name, "authorized output"),
            ):
                try:
                    os.stat(
                        entry_name,
                        dir_fd=parent,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    pass
                except OSError as error:
                    raise TransactionError(
                        f"AQ4 sibling directory {entry_label} cannot be "
                        "inspected"
                    ) from error
                else:
                    raise TransactionError(
                        f"AQ4 sibling directory {entry_label} is not fresh"
                    )
            _require_stable_parent_metadata(
                parent,
                initial,
                expected_link_delta=0,
                label=label,
            )
            _verify_parent_descriptor(
                authorized,
                parent_identity,
                required_uid=required_uid,
                label=label,
            )
            return cls(
                path=authorized.parent / name,
                name=name,
                authorized=authorized,
                parent_descriptor=parent,
                parent_identity=parent_identity,
                initial_parent_metadata=initial,
                required_uid=required_uid,
                label=label,
            )
        except BaseException:
            os.close(parent)
            raise

    def _verify_parent(self, *, expected_link_delta: int) -> None:
        if self.closed:
            raise TransactionError("AQ4 sibling directory staging is closed")
        _require_stable_parent_metadata(
            self.parent_descriptor,
            self.initial_parent_metadata,
            expected_link_delta=expected_link_delta,
            label=self.label,
        )
        _verify_parent_descriptor(
            self.authorized,
            self.parent_identity,
            required_uid=self.required_uid,
            label=self.label,
        )

    def _authorized_directory_delta(self) -> int:
        try:
            metadata = os.stat(
                self.authorized.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return 0
        except OSError as error:
            raise TransactionError(
                "authorized AQ4 output cannot be inspected"
            ) from error
        return int(stat.S_ISDIR(metadata.st_mode))

    def _open_staged(
        self,
        *,
        expected_mode: int,
    ) -> tuple[int, os.stat_result]:
        self._verify_parent(
            expected_link_delta=1 + self._authorized_directory_delta()
        )
        descriptor = -1
        try:
            named = os.stat(
                self.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
            descriptor = os.open(
                self.name,
                _directory_flags(),
                dir_fd=self.parent_descriptor,
            )
            opened = os.fstat(descriptor)
            if (
                not _same_directory_object(named, opened)
                or opened.st_dev != self.initial_parent_metadata.st_dev
                or opened.st_uid != self.required_uid
                or stat.S_IMODE(opened.st_mode) != expected_mode
            ):
                raise TransactionError(
                    "AQ4 sibling directory staging metadata differs"
                )
            return descriptor, opened
        except FileNotFoundError as error:
            raise TransactionError(
                "AQ4 sibling directory staging output is unavailable"
            ) from error
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            raise

    def require_private(self) -> None:
        descriptor, _ = self._open_staged(expected_mode=0o700)
        os.close(descriptor)

    def publish(self) -> None:
        descriptor, before = self._open_staged(expected_mode=0o555)
        try:
            _rename_noreplace(
                self.name,
                self.authorized.name,
                source_parent_descriptor=self.parent_descriptor,
                destination_parent_descriptor=self.parent_descriptor,
            )
            try:
                after = os.stat(
                    self.authorized.name,
                    dir_fd=self.parent_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise TransactionError(
                    "published AQ4 directory is unavailable"
                ) from error
            if (
                not _same_directory_object(before, after)
                or before.st_mode != after.st_mode
                or stat.S_IMODE(after.st_mode) != 0o555
            ):
                raise TransactionError(
                    "published AQ4 directory identity differs"
                )
            try:
                os.stat(
                    self.name,
                    dir_fd=self.parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise TransactionError(
                    "AQ4 sibling directory staging name remains published"
                )
            os.fsync(descriptor)
            os.fsync(self.parent_descriptor)
            self._verify_parent(expected_link_delta=1)
            self.published = True
        finally:
            os.close(descriptor)

    def close(self) -> None:
        if self.closed:
            return
        descriptor = -1
        try:
            try:
                named = os.stat(
                    self.name,
                    dir_fd=self.parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                self._verify_parent(
                    expected_link_delta=(
                        1
                        if self.published
                        else self._authorized_directory_delta()
                    ),
                )
                return
            except OSError as error:
                raise TransactionError(
                    "AQ4 sibling directory staging cannot be inspected "
                    "for cleanup"
                ) from error
            descriptor = os.open(
                self.name,
                _directory_flags(),
                dir_fd=self.parent_descriptor,
            )
            opened = os.fstat(descriptor)
            if (
                not _same_directory_object(named, opened)
                or opened.st_dev != self.initial_parent_metadata.st_dev
                or opened.st_uid != self.required_uid
            ):
                raise TransactionError(
                    "AQ4 sibling directory staging identity differs"
                )
            os.fchmod(descriptor, 0o700)
            _clear_owned_directory(
                descriptor,
                root_device=opened.st_dev,
                required_uid=self.required_uid,
                remaining=[MAX_OUTPUT_FILES + 128],
            )
            named_after = os.stat(
                self.name,
                dir_fd=self.parent_descriptor,
                follow_symlinks=False,
            )
            opened_after = os.fstat(descriptor)
            if not _same_directory_object(named_after, opened_after):
                raise TransactionError(
                    "AQ4 sibling directory staging changed during cleanup"
                )
            os.close(descriptor)
            descriptor = -1
            os.rmdir(self.name, dir_fd=self.parent_descriptor)
            os.fsync(self.parent_descriptor)
            self._verify_parent(
                expected_link_delta=self._authorized_directory_delta()
            )
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(self.parent_descriptor)
            self.parent_descriptor = -1
            self.closed = True


@contextmanager
def _private_staging_root(
    authorized_path: Path,
    *,
    required_uid: int,
    label: str,
) -> Iterator[_PrivateStagingRoot]:
    staging = _PrivateStagingRoot.create(
        authorized_path,
        required_uid=required_uid,
        label=label,
    )
    try:
        yield staging
    finally:
        staging.close()


@contextmanager
def _private_sibling_staging_directory(
    authorized_path: Path,
    *,
    required_uid: int,
    label: str,
) -> Iterator[_PrivateSiblingDirectory]:
    staging = _PrivateSiblingDirectory.create(
        authorized_path,
        required_uid=required_uid,
        label=label,
    )
    try:
        yield staging
    finally:
        staging.close()


@contextmanager
def _private_sibling_staging_path(
    authorized_path: Path,
    *,
    required_uid: int,
    label: str,
) -> Iterator[Path]:
    """Reserve an unguessable fresh name in the bundle's authorized parent."""

    authorized = _lexical_absolute(authorized_path, label)
    parent, parent_identity = _open_parent_descriptor(
        authorized,
        label,
        required_uid=required_uid,
    )
    initial_parent_metadata = os.fstat(parent)
    name = f"{AQ4_STAGING_PREFIX}{secrets.token_hex(16)}-{authorized.name}"
    staging = authorized.parent / name
    try:
        try:
            for entry_name, entry_label in (
                (name, "staging name"),
                (authorized.name, "authorized output"),
            ):
                try:
                    os.stat(
                        entry_name,
                        dir_fd=parent,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    pass
                except OSError as error:
                    raise TransactionError(
                        f"AQ4 sibling {entry_label} cannot be inspected"
                    ) from error
                else:
                    raise TransactionError(
                        f"AQ4 sibling {entry_label} is not fresh"
                    )
            _verify_parent_descriptor(
                staging,
                parent_identity,
                required_uid=required_uid,
                label=label,
            )
            _require_stable_parent_metadata(
                parent,
                initial_parent_metadata,
                expected_link_delta=0,
                label=label,
            )
            yield staging
        finally:
            try:
                metadata = os.stat(
                    name,
                    dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            except OSError as error:
                raise TransactionError(
                    "AQ4 sibling staging output cannot be inspected for cleanup"
                ) from error
            else:
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != required_uid
                    or metadata.st_nlink != 1
                ):
                    raise TransactionError(
                        "AQ4 sibling staging output identity differs"
                    )
                os.unlink(name, dir_fd=parent)
                os.fsync(parent)
            _verify_parent_descriptor(
                staging,
                parent_identity,
                required_uid=required_uid,
                label=label,
            )
            _require_stable_parent_metadata(
                parent,
                initial_parent_metadata,
                expected_link_delta=0,
                label=label,
            )
    finally:
        os.close(parent)


def _require_private_staged_file(
    path: Path,
    *,
    required_uid: int,
    label: str,
) -> StableFileSnapshot:
    snapshot = _read_input(path, label, MAX_OUTPUT_FILE_BYTES)
    if (
        snapshot.identity.uid != required_uid
        or snapshot.identity.links != 1
        or stat.S_IMODE(snapshot.identity.mode) != 0o600
    ):
        raise TransactionError(f"{label} private staging metadata differs")
    return snapshot


def _publish_staged_file(
    staged_path: Path,
    authorized_path: Path,
    *,
    required_uid: int,
    label: str,
) -> StableFileSnapshot:
    staged = _read_input(staged_path, label, MAX_OUTPUT_FILE_BYTES)
    if (
        staged.identity.uid != required_uid
        or staged.identity.links != 1
        or stat.S_IMODE(staged.identity.mode) != 0o444
    ):
        raise TransactionError(f"{label} immutable staging identity differs")
    _exclusive_publish(
        authorized_path,
        staged.raw,
        mode=0o444,
        required_uid=required_uid,
        label=label,
        maximum=MAX_OUTPUT_FILE_BYTES,
    )
    return _read_input(authorized_path, label, MAX_OUTPUT_FILE_BYTES)


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


def _observe_aq4(
    preflight_result: TransactionPreflight,
    *,
    stage: str,
) -> dict[str, Any]:
    active_now = _read_input(
        preflight_result.active.path,
        "actual active served-model manifest",
        MAX_MANIFEST_BYTES,
    )
    if active_now.raw != preflight_result.active.raw:
        raise TransactionError("actual active manifest bytes differ from AQ4")
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
    _validate_candidate_promotion_receipt(
        candidate_document,
        receipt_path=receipt_path,
        source_root=request.source_root,
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
    repin_aq4_release: bool = True,
    runtime_scope: str = "all",
) -> None:
    _require_runtime_seals(
        preflight_result,
        required_uid=policy.required_uid,
        scope=runtime_scope,
    )
    source_commit, source_tree, _source_seal = _sealed_source_identity(
        request.source_root,
        runner=runner,
        required_uid=policy.required_uid,
        expected_seal=preflight_result.source_seal,
    )
    if repin_aq4_release:
        if preflight_result.aq4_source_seal is None:
            raise TransactionError("AQ4 campaign source seal is unavailable")
        aq4_source_commit, aq4_source_tree, _aq4_source_seal = (
            _sealed_source_identity(
                preflight_result.aq4_source_root,
                runner=runner,
                required_uid=policy.required_uid,
                require_detached=True,
                expected_seal=preflight_result.aq4_source_seal,
            )
        )
    else:
        aq4_source_commit = preflight_result.aq4_source_commit
        aq4_source_tree = preflight_result.aq4_source_tree
    # Repinning is also the restoration/recovery primitive.  It must remain
    # available after the campaign window closes; admission to new campaign
    # work is enforced by the live preflight and candidate-window guards.
    reloaded = authorization.load_claim(
        request.authorization_path,
        now=now,
        policy=policy,
    )
    unit = _read_input(request.systemd_unit, "systemd unit", MAX_INPUT_BYTES)
    environment = _read_input(
        request.environment_file, "systemd environment file", MAX_INPUT_BYTES
    )
    candidate: StableFileSnapshot | None = None
    receipt_sha256: str | None = None
    receipt: StableFileSnapshot | None = None
    if runtime_scope == "all":
        candidate = _read_input(
            preflight_result.candidate.path,
            "frozen candidate served-model manifest",
            MAX_MANIFEST_BYTES,
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
        _validate_candidate_promotion_receipt(
            candidate_document,
            receipt_path=receipt_path,
            source_root=request.source_root,
        )
    aq4_worker = preflight_result.aq4_worker_binary
    aq4_receipt = preflight_result.aq4_promotion_receipt
    aq4_evidence = preflight_result.aq4_promotion_evidence
    if repin_aq4_release:
        if (
            aq4_worker is None
            or aq4_receipt is None
            or aq4_evidence is None
        ):
            raise TransactionError("AQ4 release inputs are not pinned")
        aq4_worker = _read_input(
            aq4_worker.path,
            "active AQ4 worker binary",
            MAX_OUTPUT_FILE_BYTES,
        )
        aq4_receipt = _read_input(
            aq4_receipt.path,
            "active AQ4 promotion receipt",
            MAX_INPUT_BYTES,
        )
        aq4_evidence = _read_input(
            aq4_evidence.path,
            "active AQ4 promotion evidence",
            MAX_INPUT_BYTES,
        )
        aq4_release = claim.authorization.document["aq4_release"]
        for name, source_snapshot in (
            ("promotion_evidence", aq4_evidence),
            ("promotion_receipt", aq4_receipt),
        ):
            destination = Path(aq4_release[name]["path"])
            try:
                destination.lstat()
            except FileNotFoundError:
                continue
            except OSError as error:
                raise TransactionError(
                    f"AQ4 {name} copy cannot be inspected"
                ) from error
            copied = _read_input(
                destination,
                f"AQ4 {name} immutable copy",
                MAX_INPUT_BYTES,
            )
            if (
                copied.raw != source_snapshot.raw
                or copied.identity.uid != policy.required_uid
                or copied.identity.links != 1
                or stat.S_IMODE(copied.identity.mode) != 0o444
            ):
                raise TransactionError(
                    f"AQ4 {name} immutable copy differs"
                )
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
        or aq4_source_commit != preflight_result.aq4_source_commit
        or aq4_source_tree != preflight_result.aq4_source_tree
        or reloaded.snapshot.sha256 != claim.snapshot.sha256
        or reloaded.authorization.snapshot.sha256
        != claim.authorization.snapshot.sha256
        or unit.sha256 != preflight_result.systemd_unit_sha256
        or environment.sha256 != preflight_result.environment_sha256
        or (
            runtime_scope == "all"
            and (
                candidate is None
                or candidate.raw != preflight_result.candidate.raw
                or receipt_sha256
                != preflight_result.candidate_promotion_receipt_sha256
                or receipt is None
                or receipt.sha256
                != preflight_result.candidate_promotion_receipt_sha256
            )
        )
        or (
            repin_aq4_release
            and (
                aq4_worker != preflight_result.aq4_worker_binary
                or aq4_receipt != preflight_result.aq4_promotion_receipt
                or aq4_evidence != preflight_result.aq4_promotion_evidence
            )
        )
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


def default_candidate_stabilization_probe(
    request: TransactionRequest,
    _claim: authorization.ClaimRecord,
    _preflight_result: TransactionPreflight,
) -> dict[str, Any]:
    """Capture the exact live candidate service/process epoch."""

    try:
        service, gateway, worker = restoration_proof._service_identity(
            request.service_unit
        )
    except restoration_proof.RestorationProofError as error:
        raise TransactionError(
            "candidate stabilization service identity is unavailable"
        ) from error
    return {
        "service": service,
        "gateway": gateway,
        "worker": worker,
    }


def _candidate_epoch_identity(
    value: dict[str, Any],
    *,
    request: TransactionRequest,
    preflight_result: TransactionPreflight,
) -> tuple[Any, ...]:
    if not isinstance(value, dict) or set(value) != {
        "service",
        "gateway",
        "worker",
    }:
        raise TransactionError("candidate stabilization epoch is malformed")
    service = value["service"]
    gateway = value["gateway"]
    worker = value["worker"]
    if (
        not isinstance(service, dict)
        or not isinstance(gateway, dict)
        or not isinstance(worker, dict)
    ):
        raise TransactionError("candidate stabilization epoch is malformed")
    expected_worker_sha256 = preflight_result.candidate_summary["worker"][
        "binary_sha256"
    ]
    required_service = {
        "unit",
        "active_state",
        "sub_state",
        "boot_id",
        "n_restarts",
    }
    required_process = {
        "pid",
        "ppid",
        "starttime_ticks",
        "executable_sha256",
    }
    if (
        set(service) != required_service
        or set(gateway) != required_process
        or set(worker) != required_process
        or service["unit"] != request.service_unit
        or service["active_state"] != "active"
        or service["sub_state"] != "running"
        or not isinstance(service["boot_id"], str)
        or not service["boot_id"]
        or type(service["n_restarts"]) is not int
        or service["n_restarts"] < 0
        or any(
            type(process[field]) is not int or process[field] <= 0
            for process in (gateway, worker)
            for field in ("pid", "ppid", "starttime_ticks")
        )
        or gateway["ppid"] != 1
        or worker["ppid"] != gateway["pid"]
        or any(
            not isinstance(process["executable_sha256"], str)
            or re.fullmatch(
                r"[0-9a-f]{64}",
                process["executable_sha256"],
            )
            is None
            for process in (gateway, worker)
        )
        or worker["executable_sha256"] != expected_worker_sha256
    ):
        raise TransactionError("candidate stabilization epoch is invalid")
    return (
        service["unit"],
        service["active_state"],
        service["sub_state"],
        service["boot_id"],
        service["n_restarts"],
        gateway["pid"],
        gateway["ppid"],
        gateway["starttime_ticks"],
        gateway["executable_sha256"],
        worker["pid"],
        worker["ppid"],
        worker["starttime_ticks"],
        worker["executable_sha256"],
    )


def _monitor_candidate_stabilization(
    request: TransactionRequest,
    claim: authorization.ClaimRecord,
    preflight_result: TransactionPreflight,
    *,
    deadline: datetime,
    repin: Callable[[], None],
    clock: Clock,
    probe: CandidateStabilizationProbe,
    sleeper: Sleeper,
    monotonic: MonotonicClock,
) -> dict[str, Any]:
    """Continuously re-pin the candidate and one unchanged live epoch."""

    wall_remaining = (deadline - clock()).total_seconds()
    if (
        not math.isfinite(wall_remaining)
        or wall_remaining < CANDIDATE_STABILIZATION_SECONDS
    ):
        raise CandidateWindowExpired(
            "authorization has insufficient time for candidate stabilization"
        )
    started = monotonic()
    if not math.isfinite(started):
        raise TransactionError("candidate stabilization clock is invalid")
    baseline: tuple[Any, ...] | None = None
    polls = 0
    while True:
        current = monotonic()
        if not math.isfinite(current) or current < started:
            raise TransactionError("candidate stabilization clock is invalid")
        elapsed = current - started
        remaining_stabilization = max(
            0.0,
            CANDIDATE_STABILIZATION_SECONDS - elapsed,
        )
        wall_remaining = (deadline - clock()).total_seconds()
        if (
            not math.isfinite(wall_remaining)
            or wall_remaining <= 0
            or wall_remaining < remaining_stabilization
        ):
            raise CandidateWindowExpired(
                "authorization expired during candidate stabilization"
            )
        repin()
        _observe_candidate(
            preflight_result,
            stage="candidate_stabilization",
        )
        epoch = _candidate_epoch_identity(
            probe(request, claim, preflight_result),
            request=request,
            preflight_result=preflight_result,
        )
        if baseline is None:
            baseline = epoch
        elif epoch != baseline:
            raise TransactionError(
                "candidate service/gateway/worker epoch changed while stabilizing"
            )
        polls += 1
        if polls > MAX_CANDIDATE_STABILIZATION_POLLS:
            raise TransactionError(
                "candidate stabilization exceeded its poll bound"
            )
        if elapsed >= CANDIDATE_STABILIZATION_SECONDS:
            assert baseline is not None
            return {
                "stage": "candidate_stabilization",
                "active_manifest_sha256": preflight_result.candidate.sha256,
                "bytes_equal": True,
                "duration_seconds": CANDIDATE_STABILIZATION_SECONDS,
                "polls": polls,
                "service_boot_id": baseline[3],
                "service_n_restarts": baseline[4],
                "gateway_pid": baseline[5],
                "gateway_starttime_ticks": baseline[7],
                "worker_pid": baseline[9],
                "worker_starttime_ticks": baseline[11],
                "worker_executable_sha256": baseline[12],
            }
        sleep_seconds = min(
            CANDIDATE_STABILIZATION_POLL_SECONDS,
            remaining_stabilization,
        )
        if (
            not math.isfinite(sleep_seconds)
            or sleep_seconds <= 0
            or sleep_seconds > wall_remaining
        ):
            raise CandidateWindowExpired(
                "authorization cannot cover the next stabilization probe"
            )
        sleeper(sleep_seconds)


def _stage_environment(
    request: TransactionRequest,
    claim: authorization.ClaimRecord,
    preflight_result: TransactionPreflight,
    stage: str,
    *,
    producer_staging_output: Path | None = None,
    producer_source_root: Path | None = None,
) -> dict[str, str]:
    environment = {
        **STAGE_BASE_ENVIRONMENT,
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
        DOCKER_LEASE_WRAPPER_ENVIRONMENT: os.fspath(
            _docker_lease_wrapper(request.source_root)
        ),
        DOCKER_LEASE_LABEL_ENVIRONMENT: _docker_lease_label(claim),
    }
    if producer_staging_output is not None:
        environment[CAMPAIGN_STAGING_OUTPUT_ENVIRONMENT] = os.fspath(
            _lexical_absolute(
                producer_staging_output,
                "campaign producer staging output",
            )
        )
        environment[CAMPAIGN_SOURCE_ROOT_ENVIRONMENT] = os.fspath(
            preflight_result.source_seal.root
            if producer_source_root is None
            else _lexical_absolute(
                producer_source_root,
                "campaign producer source root",
            )
        )
    return environment


def _openwebui_image_verifier_command(
    source_root: Path,
) -> tuple[str, ...]:
    return (
        PYTHON_BINARY,
        "-I",
        "-S",
        "-B",
        os.fspath(source_root / OPENWEBUI_IMAGE_VERIFIER),
        "--docker",
        os.fspath(_docker_lease_wrapper(source_root)),
    )


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
    runtime_scope: str = "all",
    producer_staging_output: Path | None = None,
    producer_source_root: Path | None = None,
    fixed_timeout_seconds: float | None = None,
    maximum_timeout_seconds: float = MAX_COMMAND_TIMEOUT_SECONDS,
) -> None:
    environment = _stage_environment(
        request,
        claim,
        preflight_result,
        stage,
        producer_staging_output=producer_staging_output,
        producer_source_root=producer_source_root,
    )
    for command in commands:
        timeout_seconds = (
            request.command_timeout_seconds
            if fixed_timeout_seconds is None
            else fixed_timeout_seconds
        )
        if (
            not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or timeout_seconds > maximum_timeout_seconds
        ):
            raise TransactionError(f"{stage} command timeout is invalid")
        if deadline is not None:
            remaining = (deadline - clock()).total_seconds()
            if not math.isfinite(remaining) or remaining <= 0:
                raise CandidateWindowExpired(
                    "candidate-active authorization deadline expired"
                )
            timeout_seconds = min(timeout_seconds, remaining)
        _require_runtime_seals(
            preflight_result,
            required_uid=preflight_result.source_seal.required_uid,
            scope=runtime_scope,
        )
        if deadline is not None and clock() >= deadline:
            raise CandidateWindowExpired(
                "candidate-active authorization deadline expired"
            )
        executable_descriptor = _open_command_executable(
            preflight_result,
            command,
            scope=runtime_scope,
        )
        try:
            if runner is subprocess.run:
                _run_owned_process_group(
                    command,
                    request=request,
                    environment=environment,
                    stage=stage,
                    timeout_seconds=timeout_seconds,
                    executable_descriptor=executable_descriptor,
                    run_as_campaign_executor=(
                        producer_staging_output is not None
                    ),
                    maximum_timeout_seconds=maximum_timeout_seconds,
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
                    raise TransactionError(
                        f"{stage} command failed"
                    ) from error
                if completed.returncode != 0:
                    raise TransactionError(f"{stage} command failed")
        finally:
            try:
                os.close(executable_descriptor)
            except OSError:
                pass
            _cleanup_docker_lease(
                request,
                claim,
                runner=runner,
                stage=f"{stage}:docker_lease_cleanup",
            )
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


def _drop_campaign_executor_privileges() -> None:
    """Apply the fixed service identity in setgroups→setgid→setuid order."""

    effective_uid = os.geteuid()
    effective_gid = os.getegid()
    if effective_uid == CAMPAIGN_EXECUTOR_UID:
        if effective_gid != CAMPAIGN_EXECUTOR_GID:
            raise TransactionError(
                "campaign executor test identity has an unexpected group"
            )
        return
    if effective_uid != 0:
        raise TransactionError(
            "campaign producer privilege drop requires root supervision"
        )
    try:
        os.setgroups(list(CAMPAIGN_EXECUTOR_SUPPLEMENTARY_GROUPS))
        os.setgid(CAMPAIGN_EXECUTOR_GID)
        os.setuid(CAMPAIGN_EXECUTOR_UID)
    except OSError as error:
        raise TransactionError(
            "campaign producer privilege drop failed"
        ) from error
    if (
        os.geteuid() != CAMPAIGN_EXECUTOR_UID
        or os.getegid() != CAMPAIGN_EXECUTOR_GID
        or tuple(sorted(os.getgroups()))
        != tuple(sorted(CAMPAIGN_EXECUTOR_SUPPLEMENTARY_GROUPS))
    ):
        raise TransactionError(
            "campaign producer privilege identity differs"
        )


def _exec_supervised_command(
    command: Sequence[str],
    *,
    source_root: Path,
    environment: dict[str, str],
    executable_descriptor: int | None,
    run_as_campaign_executor: bool,
) -> None:
    try:
        os.setsid()
        os.chdir(source_root)
        null_descriptor = os.open(os.devnull, os.O_RDWR | os.O_CLOEXEC)
        for target in (0, 1, 2):
            os.dup2(null_descriptor, target)
        if null_descriptor > 2:
            os.close(null_descriptor)
        if run_as_campaign_executor:
            os.umask(0o077)
            _drop_campaign_executor_privileges()
        maximum = 1_048_576
        try:
            configured = os.sysconf("SC_OPEN_MAX")
            if isinstance(configured, int) and configured > 3:
                maximum = configured
        except (OSError, ValueError):
            pass
        if executable_descriptor is None:
            os.closerange(3, maximum)
            os.execvpe(command[0], list(command), environment)
        if executable_descriptor < 3:
            raise TransactionError(
                "sealed command executable descriptor is invalid"
            )
        os.closerange(3, executable_descriptor)
        os.closerange(executable_descriptor + 1, maximum)
        os.execve(executable_descriptor, list(command), environment)
    except BaseException:
        os._exit(127)


def _subreaper_supervisor(
    command: Sequence[str],
    *,
    source_root: Path,
    environment: dict[str, str],
    timeout_seconds: float,
    executable_descriptor: int | None,
    run_as_campaign_executor: bool,
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
            executable_descriptor=executable_descriptor,
            run_as_campaign_executor=run_as_campaign_executor,
        )
        os._exit(127)
    if executable_descriptor is not None:
        try:
            os.close(executable_descriptor)
        except OSError:
            return SUPERVISOR_INTERNAL_ERROR

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
    executable_descriptor: int | None = None,
    run_as_campaign_executor: bool = False,
    maximum_timeout_seconds: float = MAX_COMMAND_TIMEOUT_SECONDS,
) -> None:
    selected_timeout = (
        request.command_timeout_seconds
        if timeout_seconds is None
        else timeout_seconds
    )
    if (
        not math.isfinite(selected_timeout)
        or selected_timeout <= 0
        or not math.isfinite(maximum_timeout_seconds)
        or maximum_timeout_seconds <= 0
        or selected_timeout > maximum_timeout_seconds
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
                    executable_descriptor=executable_descriptor,
                    run_as_campaign_executor=run_as_campaign_executor,
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


def _freeze_regular_output(
    path: Path,
    label: str,
    *,
    required_uid: int,
) -> StableFileSnapshot:
    absolute = _lexical_absolute(path, label)
    parent, parent_identity = _open_parent_descriptor(
        absolute,
        label,
        required_uid=required_uid,
    )
    descriptor = -1
    try:
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent,
        )
        before = os.fstat(descriptor)
        named = os.stat(
            absolute.name,
            dir_fd=parent,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or _entry_identity(before) != _entry_identity(named)
            or before.st_uid != required_uid
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > MAX_OUTPUT_FILE_BYTES
        ):
            raise TransactionError(f"{label} metadata is unsafe")
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        after = os.fstat(descriptor)
        named_after = os.stat(
            absolute.name,
            dir_fd=parent,
            follow_symlinks=False,
        )
        if (
            stat.S_IMODE(after.st_mode) != 0o444
            or after.st_uid != required_uid
            or after.st_nlink != 1
            or _entry_identity(after) != _entry_identity(named_after)
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
        ):
            raise TransactionError(f"{label} changed while being frozen")
        os.fsync(parent)
        _verify_parent_descriptor(
            absolute,
            parent_identity,
            required_uid=required_uid,
            label=label,
        )
    except OSError as error:
        raise TransactionError(f"{label} could not be frozen") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent)
    snapshot = _read_input(absolute, label, MAX_OUTPUT_FILE_BYTES)
    if (
        snapshot.identity.uid != required_uid
        or snapshot.identity.links != 1
        or stat.S_IMODE(snapshot.identity.mode) != 0o444
    ):
        raise TransactionError(f"{label} frozen identity differs")
    return snapshot


def _freeze_aq4_release_output(
    path: Path,
    *,
    required_uid: int,
) -> None:
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
    if files != AQ4_REASONING_RELEASE_FILES or directories:
        raise TransactionError("AQ4 release raw output layout differs")
    root_before = path.lstat()
    if (
        not stat.S_ISDIR(root_before.st_mode)
        or root_before.st_uid != required_uid
    ):
        raise TransactionError("AQ4 release raw output root metadata differs")
    for relative in sorted(files, key=lambda value: value.encode("utf-8")):
        _freeze_regular_output(
            path / relative,
            "AQ4 release raw output member",
            required_uid=required_uid,
        )
    try:
        os.chmod(path, 0o555, follow_symlinks=False)
        directory = os.open(path, _directory_flags())
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as error:
        raise TransactionError(
            "AQ4 release raw output root could not be frozen"
        ) from error
    after = _scan_output_tree(path)
    def same_object(
        before_identity: tuple[int, ...],
        after_identity: tuple[int, ...],
    ) -> bool:
        return (
            before_identity[0] == after_identity[0]
            and before_identity[1] == after_identity[1]
            and before_identity[3] == after_identity[3]
            and before_identity[4] == after_identity[4]
            and before_identity[5] == after_identity[5]
            and before_identity[6] == after_identity[6]
            and before_identity[7] == after_identity[7]
            and after_identity[8] >= before_identity[8]
        )
    if (
        set(after) != set(first)
        or any(
            not same_object(first[relative], after[relative])
            for relative in first
        )
        or stat.S_IMODE(after["."][2]) != 0o555
        or after["."][3] != required_uid
        or any(
            stat.S_IMODE(after[relative][2]) != 0o444
            or after[relative][3] != required_uid
            or after[relative][5] != 1
            for relative in files
        )
    ):
        raise TransactionError("AQ4 release raw output freeze differs")


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


def _require_no_staging_path_references(
    path: Path,
    *,
    forbidden_paths: Sequence[Path],
) -> None:
    needles = tuple(
        os.fsencode(
            _lexical_absolute(value, "forbidden campaign staging path")
        )
        for value in forbidden_paths
    )
    if not needles:
        raise TransactionError("campaign staging reference set is empty")
    metadata = path.lstat()
    if stat.S_ISDIR(metadata.st_mode):
        scanned = _scan_output_tree(path)
        files = (
            path / relative
            for relative, identity in scanned.items()
            if relative != "." and stat.S_ISREG(identity[2])
        )
    elif stat.S_ISREG(metadata.st_mode):
        files = (path,)
    else:
        raise TransactionError("published campaign output kind is unsafe")
    for member in files:
        snapshot = _read_input(
            member,
            "published campaign output member",
            MAX_OUTPUT_FILE_BYTES,
        )
        if any(needle in snapshot.raw for needle in needles):
            raise TransactionError(
                "published campaign output contains a staging pathname"
            )


def _output_inventory(
    path: Path,
    *,
    run_id: str,
    campaign_name: str,
    required_uid: int,
    candidate_raw: bytes,
    semantic_path: Path | None = None,
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
    if campaign_name == "aq4_reasoning_release":
        summary_snapshot = _read_input(
            path / "summary.json",
            "AQ4 reasoning release summary",
            MAX_OUTPUT_FILE_BYTES,
        )
        summary = _strict_object(
            summary_snapshot.raw,
            "AQ4 reasoning release summary",
        )
        if (
            summary.get("schema_version")
            != "ullm.generic_reasoning_release_campaign.v1"
            or summary.get("status") != "incomplete"
            or summary.get("manifest_sha256") != _sha256(candidate_raw)
            or summary.get("model_id") != "ullm-qwen3.5-9b-aq4"
        ):
            raise TransactionError(
                "AQ4 reasoning release semantic identity differs"
            )
    else:
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
        expected_semantic_path = (
            path if semantic_path is None else semantic_path
        )
        evidence_name = (
            "summary.json"
            if campaign_name == "reasoning_release"
            else "browser-evidence.json"
        )
        expected_schema = (
            "ullm.generic_reasoning_release_campaign.v2"
            if campaign_name == "reasoning_release"
            else "ullm.openwebui.reasoning_browser_smoke.v5"
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
            or campaign.get("final_path")
            != os.fspath(expected_semantic_path)
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


def _output_file_inventory(
    path: Path,
    *,
    run_id: str,
    campaign_name: str,
    required_uid: int,
) -> dict[str, Any]:
    snapshot = _read_input(path, f"{campaign_name} output", MAX_OUTPUT_FILE_BYTES)
    if (
        snapshot.identity.uid != required_uid
        or snapshot.identity.links != 1
        or stat.S_IMODE(snapshot.identity.mode) != 0o444
    ):
        raise TransactionError(f"{campaign_name} output metadata differs")
    document = _strict_object(snapshot.raw, f"{campaign_name} output")
    if campaign_name == "aq4_reasoning_browser":
        if (
            document.get("schema_version")
            != "ullm.openwebui.reasoning_browser_smoke.v2"
        ):
            raise TransactionError("AQ4 browser evidence schema differs")
    elif campaign_name == "aq4_bundle":
        if (
            document.get("schema_version")
            != "ullm.generic_reasoning_release_bundle.v1"
            or document.get("status") != "complete"
            or document.get("production_activation_performed") is not False
        ):
            raise TransactionError("AQ4 release bundle identity differs")
    else:
        raise TransactionError("campaign file output layout is unknown")
    return {
        "run_id": run_id,
        "path": os.fspath(path),
        "kind": "file",
        "sha256": snapshot.sha256,
        "artifact_count": 1,
        "total_bytes": len(snapshot.raw),
        "selected_artifacts": {path.name: snapshot.sha256},
    }


def _validate_aq4_release_evidence_identity(
    preflight_result: TransactionPreflight,
    claim: authorization.ClaimRecord,
    evidence_snapshot: StableFileSnapshot,
    *,
    required_uid: int,
) -> None:
    aq4_release = claim.authorization.document["aq4_release"]
    before = claim.authorization.document["before"]
    if (
        evidence_snapshot.identity.uid != required_uid
        or evidence_snapshot.identity.links != 1
        or stat.S_IMODE(evidence_snapshot.identity.mode) != 0o444
    ):
        raise TransactionError("AQ4 release evidence metadata differs")
    evidence = _strict_object(evidence_snapshot.raw, "AQ4 release evidence")
    identity = evidence.get("identity")
    if (
        evidence.get("schema_version")
        != "ullm.generic_reasoning_release_evidence.v1"
        or evidence.get("status") != "complete"
        or evidence.get("source_commit") != preflight_result.aq4_source_commit
        or evidence.get("active_promotion_source_commit")
        != before["promotion_source_commit"]
        or evidence.get("source_commit_aligned") is not True
        or evidence.get("git_worktree_clean") is not True
        or not isinstance(identity, dict)
        or identity.get("manifest_sha256") != preflight_result.active.sha256
        or identity.get("worker_binary_sha256")
        != before["worker_binary_sha256"]
        or identity.get("openwebui_image") != aq4_release["openwebui_image"]
    ):
        raise TransactionError("AQ4 release evidence lineage differs")


def _validate_aq4_release_components(
    preflight_result: TransactionPreflight,
    claim: authorization.ClaimRecord,
    *,
    required_uid: int,
) -> None:
    aq4_release = claim.authorization.document["aq4_release"]
    evidence_snapshot = _read_input(
        Path(aq4_release["release_evidence_path"]),
        "AQ4 release evidence",
        MAX_OUTPUT_FILE_BYTES,
    )
    _validate_aq4_release_evidence_identity(
        preflight_result,
        claim,
        evidence_snapshot,
        required_uid=required_uid,
    )
    validator_snapshot = _read_input(
        Path(aq4_release["release_validator_path"]),
        "AQ4 release validator report",
        MAX_OUTPUT_FILE_BYTES,
    )
    if (
        validator_snapshot.identity.uid != required_uid
        or validator_snapshot.identity.links != 1
        or stat.S_IMODE(validator_snapshot.identity.mode) != 0o444
    ):
        raise TransactionError("AQ4 release validator report metadata differs")
    report = _strict_object(
        validator_snapshot.raw,
        "AQ4 release validator report",
    )
    if (
        report.get("schema_version")
        != "ullm.generic_reasoning_release_validator.v1"
        or report.get("input_schema_version")
        != "ullm.generic_reasoning_release_evidence.v1"
        or report.get("structurally_valid") is not True
        or report.get("gate_eligible") is not True
    ):
        raise TransactionError("AQ4 release validator lineage differs")


def _validate_aq4_browser_components(
    claim: authorization.ClaimRecord,
    *,
    required_uid: int,
) -> None:
    aq4_release = claim.authorization.document["aq4_release"]
    report_snapshot = _read_input(
        Path(aq4_release["browser_validator_path"]),
        "AQ4 browser validator report",
        MAX_OUTPUT_FILE_BYTES,
    )
    if (
        report_snapshot.identity.uid != required_uid
        or report_snapshot.identity.links != 1
        or stat.S_IMODE(report_snapshot.identity.mode) != 0o444
    ):
        raise TransactionError("AQ4 browser validator report metadata differs")
    report = _strict_object(
        report_snapshot.raw,
        "AQ4 browser validator report",
    )
    if (
        report.get("schema_version")
        != "ullm.openwebui.reasoning_browser_smoke_validator.v1"
        or report.get("input_schema_version")
        != "ullm.openwebui.reasoning_browser_smoke.v2"
        or report.get("structurally_valid") is not True
        or report.get("gate_eligible") is not True
    ):
        raise TransactionError("AQ4 browser validator report differs")


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
    candidate_stabilization_probe: CandidateStabilizationProbe = (
        default_candidate_stabilization_probe
    ),
    stabilization_sleeper: Sleeper = time.sleep,
    stabilization_monotonic: MonotonicClock = time.monotonic,
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
        aq4_observations: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        active_slot: ActiveSlot | None = None
        preflight_result: TransactionPreflight | None = None
        failure_stage: str | None = None
        primary_error: BaseException | None = None
        switched = False
        ownership_lost = False
        containment_lost = False
        docker_lease_cleanup_proved = False
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

        def repin(
            *,
            include_aq4_release: bool = True,
            runtime_scope: str = "all",
        ) -> None:
            assert preflight_result is not None
            _repin_transaction_inputs(
                request,
                claim,
                preflight_result,
                policy=policy,
                runner=runner,
                now=clock(),
                repin_aq4_release=include_aq4_release,
                runtime_scope=runtime_scope,
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
            repin_aq4_release: bool = True,
            runtime_scope: str = "all",
            producer_staging_output: Path | None = None,
            producer_source_root: Path | None = None,
            fixed_timeout_seconds: float | None = None,
            maximum_timeout_seconds: float = MAX_COMMAND_TIMEOUT_SECONDS,
        ) -> None:
            assert preflight_result is not None
            if candidate_active:
                ensure_candidate_window()
            repin(
                include_aq4_release=repin_aq4_release,
                runtime_scope=runtime_scope,
            )
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
                runtime_scope=runtime_scope,
                producer_staging_output=producer_staging_output,
                producer_source_root=producer_source_root,
                fixed_timeout_seconds=fixed_timeout_seconds,
                maximum_timeout_seconds=maximum_timeout_seconds,
            )
            if candidate_active:
                ensure_candidate_window()
            repin(
                include_aq4_release=repin_aq4_release,
                runtime_scope=runtime_scope,
            )
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
                    aq4_source_root=preflight_result.aq4_source_root,
                    aq4_source_commit=preflight_result.aq4_source_commit,
                    aq4_source_tree=preflight_result.aq4_source_tree,
                    before_manifest_sha256=preflight_result.active.sha256,
                    before_worker_protocol=(
                        preflight_result.active_summary["worker"]["protocol"]
                    ),
                    before_worker_binary_path=Path(
                        preflight_result.active_summary["worker"]["binary"]
                    ),
                    before_promotion_receipt_path=(
                        preflight_result.aq4_promotion_receipt.path
                    ),
                    before_promotion_receipt_sha256=(
                        preflight_result.aq4_promotion_receipt.sha256
                    ),
                    aq4_promotion_evidence_path=(
                        preflight_result.aq4_promotion_evidence.path
                    ),
                    aq4_promotion_evidence_sha256=(
                        preflight_result.aq4_promotion_evidence.sha256
                    ),
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
                _require_empty_docker_lease(
                    request,
                    claim,
                    runner=runner,
                    stage="docker_lease_preflight",
                )
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
                _monitor_candidate_stabilization(
                    request,
                    claim,
                    preflight_result,
                    deadline=candidate_deadline,
                    repin=repin,
                    clock=clock,
                    probe=candidate_stabilization_probe,
                    sleeper=stabilization_sleeper,
                    monotonic=stabilization_monotonic,
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
                    if name == "reasoning_browser":
                        execute_commands(
                            "reasoning_browser_openwebui_image_before",
                            (
                                _openwebui_image_verifier_command(
                                    request.source_root
                                ),
                            ),
                            candidate_active=True,
                        )
                    with _service_producer_staging(
                        final_path,
                        required_uid=policy.required_uid,
                        expected_kind="directory",
                        label=f"{name} authorized output",
                    ) as producer_staging:
                        execute_commands(
                            name,
                            (campaign_commands[name],),
                            candidate_active=True,
                            producer_staging_output=producer_staging.actual,
                            fixed_timeout_seconds=(
                                SQ8_FULL_MAX_TIMEOUT_SECONDS
                                if name == "sq8_full"
                                else None
                            ),
                            maximum_timeout_seconds=(
                                SQ8_FULL_MAX_TIMEOUT_SECONDS
                                if name == "sq8_full"
                                else MAX_COMMAND_TIMEOUT_SECONDS
                            ),
                        )
                        producer_staging.reclaim_and_adopt()
                        _output_inventory(
                            producer_staging.actual,
                            run_id=campaign["run_id"],
                            campaign_name=name,
                            required_uid=policy.required_uid,
                            candidate_raw=preflight_result.candidate.raw,
                            semantic_path=final_path,
                        )
                        _require_no_staging_path_references(
                            producer_staging.actual,
                            forbidden_paths=(
                                producer_staging.path,
                                producer_staging.actual,
                            ),
                        )
                        producer_staging.require_seal()
                        producer_staging.publish_directory()
                        if name == "reasoning_browser":
                            execute_commands(
                                "reasoning_browser_openwebui_image_after",
                                (
                                    _openwebui_image_verifier_command(
                                        request.source_root
                                    ),
                                ),
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
                        _require_no_staging_path_references(
                            final_path,
                            forbidden_paths=(
                                producer_staging.path,
                                producer_staging.actual,
                            ),
                        )
                        ensure_candidate_window()
                        producer_staging.commit()
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
            try:
                with termination.deferred():
                    _cleanup_docker_lease(
                        request,
                        claim,
                        runner=runner,
                        stage="docker_lease_before_aq4_restore",
                    )
                    docker_lease_cleanup_proved = True
            except BaseException as error:
                docker_lease_cleanup_proved = False
                fail_stage(
                    "aq4_restore" if switched else "preflight",
                    error,
                    prioritize=switched,
                )
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
                            repin_aq4_release=(primary_error is None),
                            runtime_scope="aq4",
                        )
                        stages["reverse_reconciliation"] = "passed"
                    except BaseException as error:
                        fail_stage(
                            "reverse_reconciliation",
                            error,
                            prioritize=True,
                        )
                    sq8_completed = (
                        primary_error is None
                        and deferred_interrupt is None
                        and all(
                            stages[name] == "passed"
                            for name in (
                                "sq8_full",
                                "reasoning_release",
                                "reasoning_browser",
                            )
                        )
                        and stages["reverse_reconciliation"] == "passed"
                    )
                    if sq8_completed:
                        aq4_campaigns = claim.authorization.document[
                            "campaigns"
                        ]
                        try:
                            name = "aq4_reasoning_release"
                            campaign = aq4_campaigns[name]
                            final_path = Path(campaign["final_path"])
                            authorization.require_campaign_binding(
                                claim,
                                campaign_name=name,
                                run_id=campaign["run_id"],
                                final_path=final_path,
                            )
                            ensure_candidate_window()
                            aq4_observations.append(
                                _observe_aq4(
                                    preflight_result,
                                    stage=f"{name}:before",
                                )
                            )
                            if (
                                final_path.exists()
                                or final_path.is_symlink()
                                or Path(
                                    claim.authorization.document[
                                        "aq4_release"
                                    ]["release_evidence_path"]
                                ).exists()
                                or Path(
                                    claim.authorization.document[
                                        "aq4_release"
                                    ]["release_validator_path"]
                                ).exists()
                            ):
                                raise TransactionError(
                                    "AQ4 release authorized output is not fresh"
                                )
                            release_commands = (
                                request.commands.aq4_reasoning_release
                            )
                            if len(release_commands) != 3:
                                raise TransactionError(
                                    "AQ4 release fixed command plan differs"
                                )
                            with _service_producer_staging(
                                final_path,
                                required_uid=policy.required_uid,
                                expected_kind="directory",
                                label="AQ4 release raw output",
                            ) as raw_staging:
                                staged_raw = raw_staging.actual
                                execute_commands(
                                    name,
                                    (release_commands[0],),
                                    candidate_active=True,
                                    runtime_scope="aq4_release",
                                    producer_staging_output=staged_raw,
                                    producer_source_root=(
                                        preflight_result.aq4_source_root
                                    ),
                                )
                                raw_staging.reclaim_and_adopt()
                                _freeze_aq4_release_output(
                                    staged_raw,
                                    required_uid=policy.required_uid,
                                )
                                raw_staging.refresh_seal()
                                _output_inventory(
                                    staged_raw,
                                    run_id=campaign["run_id"],
                                    campaign_name=name,
                                    required_uid=policy.required_uid,
                                    candidate_raw=preflight_result.active.raw,
                                    semantic_path=final_path,
                                )
                                _require_no_staging_path_references(
                                    staged_raw,
                                    forbidden_paths=(
                                        raw_staging.path,
                                        raw_staging.actual,
                                    ),
                                )
                                raw_staging.publish_directory()
                                _output_inventory(
                                    final_path,
                                    run_id=campaign["run_id"],
                                    campaign_name=name,
                                    required_uid=policy.required_uid,
                                    candidate_raw=preflight_result.active.raw,
                                )
                                _require_no_staging_path_references(
                                    final_path,
                                    forbidden_paths=(
                                        raw_staging.path,
                                        raw_staging.actual,
                                    ),
                                )
                                raw_staging.commit()
                            release_evidence_path = Path(
                                claim.authorization.document[
                                    "aq4_release"
                                ]["release_evidence_path"]
                            )
                            with _private_staging_root(
                                release_evidence_path,
                                required_uid=policy.required_uid,
                                label="AQ4 release evidence",
                            ) as evidence_staging:
                                staged_evidence = evidence_staging.output(
                                    release_evidence_path
                                )
                                evidence_command = (
                                    _rewrite_authorized_output_argument(
                                        release_commands[1],
                                        flag="--output",
                                        authorized_path=release_evidence_path,
                                        staging_path=staged_evidence,
                                    )
                                )
                                execute_commands(
                                    name,
                                    (evidence_command,),
                                    candidate_active=True,
                                    runtime_scope="aq4_release",
                                )
                                evidence_staging.verify()
                                _require_private_staged_file(
                                    staged_evidence,
                                    required_uid=policy.required_uid,
                                    label="AQ4 release evidence",
                                )
                                staged_evidence_snapshot = (
                                    _freeze_regular_output(
                                        staged_evidence,
                                        "AQ4 release evidence",
                                        required_uid=policy.required_uid,
                                    )
                                )
                                _validate_aq4_release_evidence_identity(
                                    preflight_result,
                                    claim,
                                    staged_evidence_snapshot,
                                    required_uid=policy.required_uid,
                                )
                                _publish_staged_file(
                                    staged_evidence,
                                    release_evidence_path,
                                    required_uid=policy.required_uid,
                                    label="AQ4 release evidence",
                                )
                            execute_commands(
                                name,
                                (release_commands[2],),
                                candidate_active=True,
                                runtime_scope="aq4_release",
                            )
                            _validate_aq4_release_components(
                                preflight_result,
                                claim,
                                required_uid=policy.required_uid,
                            )
                            aq4_observations.append(
                                _observe_aq4(
                                    preflight_result,
                                    stage=f"{name}:after",
                                )
                            )
                            campaign_results[name] = _output_inventory(
                                final_path,
                                run_id=campaign["run_id"],
                                campaign_name=name,
                                required_uid=policy.required_uid,
                                candidate_raw=preflight_result.active.raw,
                            )
                            stages[name] = "passed"
                        except BaseException as error:
                            fail_stage("aq4_reasoning_release", error)

                    if sq8_completed and primary_error is None:
                        try:
                            name = "aq4_reasoning_browser"
                            campaign = aq4_campaigns[name]
                            final_path = Path(campaign["final_path"])
                            authorization.require_campaign_binding(
                                claim,
                                campaign_name=name,
                                run_id=campaign["run_id"],
                                final_path=final_path,
                            )
                            ensure_candidate_window()
                            aq4_observations.append(
                                _observe_aq4(
                                    preflight_result,
                                    stage=f"{name}:before",
                                )
                            )
                            browser_validator_path = Path(
                                claim.authorization.document["aq4_release"][
                                    "browser_validator_path"
                                ]
                            )
                            if (
                                final_path.exists()
                                or final_path.is_symlink()
                                or browser_validator_path.exists()
                                or browser_validator_path.is_symlink()
                            ):
                                raise TransactionError(
                                    "AQ4 browser authorized output is not fresh"
                                )
                            browser_commands = (
                                request.commands.aq4_reasoning_browser
                            )
                            if len(browser_commands) != 2:
                                raise TransactionError(
                                    "AQ4 browser fixed command plan differs"
                                )
                            with _service_producer_staging(
                                final_path,
                                required_uid=policy.required_uid,
                                expected_kind="file",
                                label="AQ4 browser evidence",
                            ) as browser_staging:
                                staged_browser = browser_staging.actual
                                execute_commands(
                                    "aq4_reasoning_browser_openwebui_image_before",
                                    (
                                        _openwebui_image_verifier_command(
                                            request.source_root
                                        ),
                                    ),
                                    candidate_active=True,
                                    runtime_scope="aq4_release",
                                )
                                execute_commands(
                                    name,
                                    (browser_commands[0],),
                                    candidate_active=True,
                                    runtime_scope="aq4_release",
                                    producer_staging_output=staged_browser,
                                    producer_source_root=(
                                        preflight_result.aq4_source_root
                                    ),
                                )
                                browser_staging.reclaim_and_adopt()
                                _freeze_regular_output(
                                    staged_browser,
                                    "AQ4 browser evidence",
                                    required_uid=policy.required_uid,
                                )
                                browser_staging.refresh_seal()
                                _output_file_inventory(
                                    staged_browser,
                                    run_id=campaign["run_id"],
                                    campaign_name=name,
                                    required_uid=policy.required_uid,
                                )
                                _require_no_staging_path_references(
                                    staged_browser,
                                    forbidden_paths=(
                                        browser_staging.path,
                                        browser_staging.actual,
                                    ),
                                )
                                browser_staging.publish_file(
                                    label="AQ4 browser evidence"
                                )
                                execute_commands(
                                    "aq4_reasoning_browser_openwebui_image_after",
                                    (
                                        _openwebui_image_verifier_command(
                                            request.source_root
                                        ),
                                    ),
                                    candidate_active=True,
                                    runtime_scope="aq4_release",
                                )
                                _output_file_inventory(
                                    final_path,
                                    run_id=campaign["run_id"],
                                    campaign_name=name,
                                    required_uid=policy.required_uid,
                                )
                                _require_no_staging_path_references(
                                    final_path,
                                    forbidden_paths=(
                                        browser_staging.path,
                                        browser_staging.actual,
                                    ),
                                )
                                browser_staging.commit()
                            execute_commands(
                                name,
                                (browser_commands[1],),
                                candidate_active=True,
                                runtime_scope="aq4_release",
                            )
                            _validate_aq4_browser_components(
                                claim,
                                required_uid=policy.required_uid,
                            )
                            aq4_observations.append(
                                _observe_aq4(
                                    preflight_result,
                                    stage=f"{name}:after",
                                )
                            )
                            campaign_results[name] = (
                                _output_file_inventory(
                                    final_path,
                                    run_id=campaign["run_id"],
                                    campaign_name=name,
                                    required_uid=policy.required_uid,
                                )
                            )
                            stages[name] = "passed"
                        except BaseException as error:
                            fail_stage("aq4_reasoning_browser", error)

                    if sq8_completed and primary_error is None:
                        try:
                            name = "aq4_bundle"
                            campaign = aq4_campaigns[name]
                            final_path = Path(campaign["final_path"])
                            authorization.require_campaign_binding(
                                claim,
                                campaign_name=name,
                                run_id=campaign["run_id"],
                                final_path=final_path,
                            )
                            ensure_candidate_window()
                            aq4_observations.append(
                                _observe_aq4(
                                    preflight_result,
                                    stage=f"{name}:before",
                                )
                            )
                            if final_path.exists() or final_path.is_symlink():
                                raise TransactionError(
                                    "AQ4 bundle authorized output is not fresh"
                                )
                            aq4_release = claim.authorization.document[
                                "aq4_release"
                            ]
                            if (
                                preflight_result.aq4_promotion_evidence is None
                                or preflight_result.aq4_promotion_receipt is None
                            ):
                                raise TransactionError(
                                    "AQ4 promotion source pair is not pinned"
                                )
                            repin(runtime_scope="aq4_release")
                            for component_name, source_snapshot in (
                                (
                                    "promotion_evidence",
                                    preflight_result.aq4_promotion_evidence,
                                ),
                                (
                                    "promotion_receipt",
                                    preflight_result.aq4_promotion_receipt,
                                ),
                            ):
                                ensure_candidate_window()
                                _exclusive_publish(
                                    Path(
                                        aq4_release[component_name]["path"]
                                    ),
                                    source_snapshot.raw,
                                    mode=0o444,
                                    required_uid=policy.required_uid,
                                    label=(
                                        f"AQ4 {component_name} immutable copy"
                                    ),
                                    maximum=MAX_INPUT_BYTES,
                                )
                            ensure_candidate_window()
                            repin(runtime_scope="aq4_release")
                            bundle_commands = request.commands.aq4_bundle
                            if len(bundle_commands) != 2:
                                raise TransactionError(
                                    "AQ4 bundle fixed command plan differs"
                                )
                            with _private_sibling_staging_path(
                                final_path,
                                required_uid=policy.required_uid,
                                label="AQ4 release bundle v1",
                            ) as staged_bundle:
                                bundle_command = (
                                    _rewrite_authorized_output_argument(
                                        bundle_commands[0],
                                        flag="--output",
                                        authorized_path=final_path,
                                        staging_path=staged_bundle,
                                    )
                                )
                                execute_commands(
                                    name,
                                    (bundle_command,),
                                    candidate_active=True,
                                    runtime_scope="aq4_release",
                                )
                                _require_private_staged_file(
                                    staged_bundle,
                                    required_uid=policy.required_uid,
                                    label="AQ4 release bundle v1",
                                )
                                _freeze_regular_output(
                                    staged_bundle,
                                    "AQ4 release bundle v1",
                                    required_uid=policy.required_uid,
                                )
                                _output_file_inventory(
                                    staged_bundle,
                                    run_id=campaign["run_id"],
                                    campaign_name=name,
                                    required_uid=policy.required_uid,
                                )
                                _publish_staged_file(
                                    staged_bundle,
                                    final_path,
                                    required_uid=policy.required_uid,
                                    label="AQ4 release bundle v1",
                                )
                            execute_commands(
                                name,
                                (bundle_commands[1],),
                                candidate_active=True,
                                runtime_scope="aq4_release",
                            )
                            aq4_observations.append(
                                _observe_aq4(
                                    preflight_result,
                                    stage=f"{name}:after",
                                )
                            )
                            campaign_results[name] = (
                                _output_file_inventory(
                                    final_path,
                                    run_id=campaign["run_id"],
                                    campaign_name=name,
                                    required_uid=policy.required_uid,
                                )
                            )
                            stages[name] = "passed"
                        except BaseException as error:
                            fail_stage("aq4_bundle", error)
                    if sq8_completed:
                        try:
                            aq4_current = active_slot.snapshot_current()
                            repaired_after_aq4 = (
                                aq4_current.raw != preflight_result.active.raw
                            )
                            if repaired_after_aq4:
                                restoration[
                                    "displaced_manifest_sha256"
                                ] = aq4_current.sha256
                                active_slot.replace(
                                    preflight_result.active.raw,
                                    preflight_result.active.identity,
                                    expected_current=aq4_current,
                                )
                                execute_commands(
                                    "reverse_reconciliation",
                                    request.commands.reverse_reconciliation,
                                    repin_aq4_release=(
                                        primary_error is None
                                    ),
                                    runtime_scope="aq4",
                                )
                            restored_after_aq4 = active_slot.snapshot_current()
                            if (
                                restored_after_aq4.raw
                                != preflight_result.active.raw
                            ):
                                raise TransactionError(
                                    "AQ4 campaign window did not retain "
                                    "the exact restored manifest"
                                )
                        except BaseException as error:
                            fail_stage(
                                "aq4_restore",
                                error,
                                prioritize=True,
                            )
                    final_commands_passed = False
                    try:
                        execute_commands(
                            "final_checks",
                            request.commands.final_checks,
                            repin_aq4_release=(primary_error is None),
                            runtime_scope="aq4",
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
                        final_active = active_slot.snapshot_current()
                        if final_active.raw != preflight_result.active.raw:
                            restoration[
                                "displaced_manifest_sha256"
                            ] = final_active.sha256
                            active_slot.replace(
                                preflight_result.active.raw,
                                preflight_result.active.identity,
                                expected_current=final_active,
                            )
                            stages["reverse_reconciliation"] = "failed"
                            final_commands_passed = False
                            if failure_stage is None:
                                fail_stage(
                                    "reverse_reconciliation",
                                    TransactionError(
                                        "final AQ4 bytes required a second "
                                        "post-command restoration"
                                    ),
                                )
                        exact_final_active = active_slot.snapshot_current()
                        if (
                            exact_final_active.raw
                            != preflight_result.active.raw
                        ):
                            raise TransactionError(
                                "final active manifest is not exact AQ4"
                            )
                    except BaseException as error:
                        fail_stage("aq4_restore", error, prioritize=True)
                    try:
                        docker_lease_cleanup_proved = False
                        _cleanup_docker_lease(
                            request,
                            claim,
                            runner=runner,
                            stage="docker_lease_final_zero",
                        )
                        docker_lease_cleanup_proved = True
                        repin(
                            include_aq4_release=(primary_error is None),
                            runtime_scope="aq4",
                        )
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
                        repin(
                            include_aq4_release=(primary_error is None),
                            runtime_scope="aq4",
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
                        docker_lease_cleanup_proved
                        and restoration["bytes_equal"]
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
                    if (
                        primary_error is None
                        and docker_lease_cleanup_proved
                        and all(value == "passed" for value in stages.values())
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
                        "aq4_observations": aq4_observations,
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
