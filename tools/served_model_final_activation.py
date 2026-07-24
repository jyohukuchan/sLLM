#!/usr/bin/env python3
"""Immutable final SQ8 activation plans and exact-byte AQ4 rollback.

This module deliberately does not contain production defaults for service
commands.  An operator-reviewed operations document supplies direct executable
paths, exact argument vectors, and executable SHA-256 values.  The immutable
final plan binds that document to the successful, restored-AQ4 campaign
transaction and the complete SQ8 release bundle.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import importlib.util
import ipaddress
import json
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn


TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
if os.fspath(TOOLS) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS))

import served_model_campaign_authorization as authorization  # noqa: E402
import served_model_campaign_plan as campaign_plan  # noqa: E402
import served_model_campaign_runtime_seal as runtime_seal  # noqa: E402
import served_model_campaign_source_seal as source_seal  # noqa: E402
from served_model_active_binding import (  # noqa: E402
    MAX_MANIFEST_BYTES,
    StableFileSnapshot,
    stable_read_regular,
)


PLAN_SCHEMA = "ullm.served_model.final_activation_plan.v2"
OPERATIONS_SCHEMA = "ullm.served_model.final_activation_operations.v1"
ACTIVATION_OUTCOME_SCHEMA = "ullm.served_model.final_activation_outcome.v1"
ROLLBACK_OUTCOME_SCHEMA = "ullm.served_model.final_rollback_outcome.v1"
PREFLIGHT_SCHEMA = "ullm.served_model.final_activation_preflight.v2"
AQ4_BUNDLE_SCHEMA = "ullm.generic_reasoning_release_bundle.v1"
AQ4_BUNDLE_VALIDATOR_SCHEMA = "ullm.generic_reasoning_release_bundle_validator.v1"
BUNDLE_SCHEMA = "ullm.generic_reasoning_release_bundle.v2"
BUNDLE_VALIDATOR_SCHEMA = "ullm.generic_reasoning_release_bundle_validator.v2"
SERVED_MODEL_SCHEMA = "ullm.served_model.v2"
AQ4_MODEL_ID = "ullm-qwen3.5-9b-aq4"
AQ4_FORMAT_ID = "AQ4_0"
SQ8_MODEL_ID = "ullm-qwen3-14b-sq8"
SQ8_FORMAT_ID = "SQ8_0"
WORKER_PROTOCOL = "ullm.worker.v2"
ROUTE = "restored_aq4_to_independent_sq8_bundle_v2"
ACTIVATION_CONFIRMATION = "ACTIVATE_SQ8_0_FROM_RESTORED_AQ4"
ROLLBACK_CONFIRMATION = "ROLLBACK_SQ8_0_TO_EXACT_AQ4"
LIVE_PROOF_SCHEMA = "ullm.served_model.final_activation_live_proof.v1"

VALIDATOR_PATH = TOOLS / "validate-served-model.py"
BUNDLE_VALIDATOR_PATH = TOOLS / "validate-generic-reasoning-release-bundle.py"
VALIDATOR_MODULE = "_ullm_final_activation_served_model_validator"
BUNDLE_VALIDATOR_MODULE = "_ullm_final_activation_bundle_validator"

MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_FILE_BYTES = 256 * 1024 * 1024
MAX_OUTPUT_FILES = 16_384
MAX_OUTPUT_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
MAX_COMMANDS_PER_STAGE = 64
MAX_ARGUMENTS = 128
MAX_ARGUMENT_BYTES = 65_536
COMMAND_TERMINATION_GRACE_SECONDS = 2.0
MAX_ENDPOINT_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_LIVE_PROOF_CLOCK_SKEW_SECONDS = 2
MAX_ACTIVE_WINDOW_SECONDS = 7_200
SYSTEMCTL_PATH = Path("/usr/bin/systemctl")
PRODUCTION_PYTHON_PATH = Path("/usr/bin/python3.12")
PRODUCTION_WRAPPER_NAMES = frozenset(
    {
        "prepare-served-model-final-activation.py",
        "rollback-served-model.py",
        "run-served-model-final-activation.py",
    }
)
SOURCE_GIT_TIMEOUT_SECONDS = 10.0
SOURCE_GIT_MAX_BYTES = 4 * 1024 * 1024
OPENWEBUI_SESSION_TOKEN_PARENT = Path("/run/ullm-campaign-secrets")
OPENWEBUI_SESSION_TOKEN_PARENT_UID = 0
OPENWEBUI_SESSION_TOKEN_PARENT_GID = 1000
OPENWEBUI_SESSION_TOKEN_PARENT_MODE = 0o750
RENAME_EXCHANGE = 2
PR_SET_CHILD_SUBREAPER = 36
PR_GET_CHILD_SUBREAPER = 37
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_RE = re.compile(r"[0-9a-f]{40}\Z")
IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
BOOT_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)

OPERATION_STAGES = {
    "candidate_reconciliation",
    "candidate_live_health",
    "reverse_reconciliation",
    "rollback_live_health",
}
ACTIVATION_STAGES = {
    "lock",
    "preflight",
    "candidate_activation",
    "candidate_reconciliation",
    "candidate_live_health",
    "aq4_restore",
    "reverse_reconciliation",
    "rollback_live_health",
    "outcome_publication",
}
ROLLBACK_STAGES = {
    "lock",
    "preflight",
    "aq4_restore",
    "reverse_reconciliation",
    "rollback_live_health",
    "outcome_publication",
}
STAGE_STATES = {"pending", "passed", "failed", "skipped"}
SELECTED_ARTIFACTS = {
    "SHA256SUMS",
    "active-manifest-binding.json",
    "active-manifest-observations.jsonl",
    "browser-evidence.json",
    "browser-validator.json",
    "candidate-served-model.json",
    "lifecycle.json",
    "model-identity.json",
    "release-validation.json",
    "resource-samples.jsonl",
    "summary.json",
    "validation.json",
}
DISALLOWED_COMMAND_EXECUTABLES = {
    "bash",
    "dash",
    "env",
    "node",
    "perl",
    "pkexec",
    "python",
    "python3",
    "ruby",
    "sh",
    "sudo",
    "zsh",
}
LIVE_PROOF_FIELDS = {
    "schema_version",
    "plan_sha256",
    "stage",
    "activation_epoch",
    "captured_at",
    "active_manifest",
    "service",
    "gateway",
    "worker",
    "endpoints",
    "epoch_stable",
    "passed",
}
LIVE_PROOF_ACTIVE_FIELDS = {
    "path",
    "manifest_sha256",
    "model_id",
    "format_id",
    "worker_protocol",
    "worker_binary_sha256",
}
LIVE_PROOF_SERVICE_FIELDS = {
    "unit",
    "active_state",
    "sub_state",
    "boot_id",
    "n_restarts",
    "main_pid",
    "control_group",
    "fragment_path",
    "environment_file_path",
}
LIVE_PROOF_PROCESS_FIELDS = {
    "pid",
    "ppid",
    "starttime_ticks",
    "executable_sha256",
}
LIVE_PROOF_ENDPOINT_FIELDS = {
    "gateway_healthz",
    "gateway_readyz",
    "gateway_models",
    "openwebui_health",
    "openwebui_models",
}
LIVE_PROOF_SPEC_FIELDS = {
    "path",
    "service_unit",
    "gateway_executable_sha256",
    "endpoint_urls",
}
LIVE_PROOF_ENVELOPE_FIELDS = {"reference", "document"}
ENDPOINT_NAMES = frozenset(LIVE_PROOF_ENDPOINT_FIELDS)
FINAL_CAMPAIGN_FIELDS = {
    "aq4_reasoning_release",
    "aq4_reasoning_browser",
    "aq4_bundle",
    "sq8_full",
    "reasoning_release",
    "reasoning_browser",
}


class FinalActivationError(RuntimeError):
    """The final activation or rollback failed closed."""


class FinalActivationInterrupted(BaseException):
    """A termination signal requested exact-byte restoration."""


@dataclass(frozen=True, slots=True)
class PlanRecord:
    snapshot: StableFileSnapshot
    document: dict[str, Any]
    operations: dict[str, Any]
    candidate: StableFileSnapshot
    rollback: StableFileSnapshot
    active: StableFileSnapshot
    aq4_bundle: StableFileSnapshot | None
    bundle: StableFileSnapshot | None
    unit: StableFileSnapshot
    environment: StableFileSnapshot
    campaign_outcome: authorization.FileSnapshot | None
    campaign_outcome_document: dict[str, Any] | None
    candidate_runtime: "ManifestRuntimeSeals | None"
    rollback_runtime: "ManifestRuntimeSeals"
    shared_runtime_artifacts: tuple[runtime_seal.RuntimeArtifactSeal, ...]
    candidate_operation_artifacts: tuple[runtime_seal.RuntimeArtifactSeal, ...]
    rollback_operation_artifacts: tuple[runtime_seal.RuntimeArtifactSeal, ...]
    execution_source: source_seal.SourceSeal


@dataclass(frozen=True, slots=True)
class ManifestRuntimeSeals:
    """Every named file/tree that one served-model worker may consume."""

    manifest: runtime_seal.RuntimeArtifactSeal
    worker: runtime_seal.RuntimeArtifactSeal
    promotion_receipt: runtime_seal.RuntimeArtifactSeal
    promotion_evidence: runtime_seal.RuntimeArtifactSeal
    artifacts: tuple[runtime_seal.RuntimeArtifactSeal, ...]
    trees: tuple[runtime_seal.RuntimeTreeSeal, ...]


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    outcome_path: Path
    outcome_sha256: str
    status: str


ManifestValidator = Callable[[Path], dict[str, Any]]
BundleValidator = Callable[[Path], dict[str, Any]]
CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]
Clock = Callable[[], datetime]
LiveProofLoader = Callable[["PlanRecord", str, str], dict[str, Any]]
LiveStateVerifier = Callable[
    ["PlanRecord", str, dict[str, Any], datetime, datetime, float],
    None,
]


def fail(message: str) -> NoReturn:
    raise FinalActivationError(message)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_timestamp(value: datetime) -> str:
    normalized = value.astimezone(timezone.utc).replace(microsecond=0)
    return normalized.strftime("%Y-%m-%dT%H:%M:%SZ")


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
        raise FinalActivationError("document is not canonicalizable JSON") from error


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    fail("JSON contains a non-finite number")


def _strict_json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FinalActivationError(f"{label} is not strict JSON") from error


def _strict_object(raw: bytes, label: str) -> dict[str, Any]:
    value = _strict_json(raw, label)
    if not isinstance(value, dict):
        fail(f"{label} root is not an object")
    return value


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        fail(f"{label} fields differ")
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        fail(f"{label} is not a lowercase SHA-256")
    return value


def _git(value: Any, label: str) -> str:
    if not isinstance(value, str) or GIT_RE.fullmatch(value) is None:
        fail(f"{label} is not a full lowercase Git object ID")
    return value


def _module_execution_root() -> Path:
    """Return only the source root derived from this loaded module."""

    module_path = Path(__file__)
    if (
        not module_path.is_absolute()
        or Path(os.path.abspath(module_path)) != module_path
        or module_path.resolve(strict=True) != module_path
        or module_path.parent != TOOLS
        or TOOLS.parent != ROOT
    ):
        fail("final activation module source path is not canonical")
    return module_path.parent.parent


def _source_git(
    root: Path,
    arguments: tuple[str, ...],
    label: str,
) -> bytes:
    argv = source_seal.git_argv(
        ["-C", os.fspath(root), *arguments]
    )
    try:
        completed = subprocess.run(
            argv,
            cwd="/",
            env=source_seal.git_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=SOURCE_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise FinalActivationError(
            f"{label} could not be read with hardened Git"
        ) from error
    if (
        completed.returncode != 0
        or completed.stderr
        or len(completed.stdout) > SOURCE_GIT_MAX_BYTES
    ):
        fail(f"{label} differs")
    return completed.stdout


def _require_source_git_identity(
    root: Path,
    *,
    expected_commit: str,
    expected_tree: str,
) -> None:
    _git(expected_commit, "execution source commit")
    _git(expected_tree, "execution source tree")
    expected_reads = (
        (
            ("rev-parse", "--show-toplevel"),
            os.fsencode(root) + b"\n",
            "execution source Git top-level",
        ),
        (
            ("rev-parse", "--verify", "HEAD^{commit}"),
            expected_commit.encode("ascii") + b"\n",
            "execution source Git commit",
        ),
        (
            ("rev-parse", "--verify", "HEAD^{tree}"),
            expected_tree.encode("ascii") + b"\n",
            "execution source Git tree",
        ),
        (
            ("rev-parse", "--abbrev-ref", "HEAD"),
            b"HEAD\n",
            "execution source detached HEAD",
        ),
    )
    for arguments, expected, label in expected_reads:
        if _source_git(root, arguments, label) != expected:
            fail(f"{label} differs")
    status = _source_git(
        root,
        (
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=all",
            "--no-renames",
        ),
        "execution source Git status",
    )
    if status:
        fail("execution source worktree is not clean")


def _capture_source_root(
    root: Path,
    *,
    expected_commit: str,
    expected_tree: str,
    required_uid: int,
) -> source_seal.SourceSeal:
    """Seal one already-derived source root and bind its exact Git identity."""

    try:
        sealed = source_seal.capture_source_seal(
            root,
            required_uid=required_uid,
        )
    except source_seal.SourceSealError as error:
        raise FinalActivationError(
            "final activation execution source is not a protected "
            "standalone clone"
        ) from error
    _require_source_git_identity(
        root,
        expected_commit=expected_commit,
        expected_tree=expected_tree,
    )
    try:
        source_seal.require_source_seal(
            sealed,
            required_uid=required_uid,
        )
    except source_seal.SourceSealError as error:
        raise FinalActivationError(
            "final activation execution source changed during capture"
        ) from error
    _require_source_git_identity(
        root,
        expected_commit=expected_commit,
        expected_tree=expected_tree,
    )
    try:
        source_seal.require_source_seal(
            sealed,
            required_uid=required_uid,
        )
    except source_seal.SourceSealError as error:
        raise FinalActivationError(
            "final activation execution source changed across Git capture"
        ) from error
    return sealed


def _capture_execution_source(
    *,
    expected_commit: str,
    expected_tree: str,
    required_uid: int,
) -> source_seal.SourceSeal:
    """Capture the protected source containing this module, never a caller root."""

    return _capture_source_root(
        _module_execution_root(),
        expected_commit=expected_commit,
        expected_tree=expected_tree,
        required_uid=required_uid,
    )


def _require_execution_source(
    expected: source_seal.SourceSeal,
    *,
    expected_commit: str,
    expected_tree: str,
    required_uid: int,
) -> None:
    root = _module_execution_root()
    if expected.root != root or expected.required_uid != required_uid:
        fail("final activation execution source binding differs")
    try:
        source_seal.require_source_seal(
            expected,
            required_uid=required_uid,
        )
    except source_seal.SourceSealError as error:
        raise FinalActivationError(
            "final activation execution source seal changed"
        ) from error
    _require_source_git_identity(
        root,
        expected_commit=expected_commit,
        expected_tree=expected_tree,
    )
    try:
        source_seal.require_source_seal(
            expected,
            required_uid=required_uid,
        )
    except source_seal.SourceSealError as error:
        raise FinalActivationError(
            "final activation execution source changed across Git repin"
        ) from error


def require_production_entrypoint(wrapper_path: Path) -> None:
    """Admit only the documented root/canonical-Python wrapper invocation."""

    root = _module_execution_root()
    wrapper = Path(wrapper_path)
    expected_wrapper = root / "tools" / wrapper.name
    original_argv = getattr(sys, "orig_argv", None)
    expected_prefix = [
        os.fspath(PRODUCTION_PYTHON_PATH),
        "-I",
        "-S",
        "-B",
        os.fspath(expected_wrapper),
    ]
    if (
        os.geteuid() != 0
        or not wrapper.is_absolute()
        or Path(os.path.abspath(wrapper)) != wrapper
        or wrapper.resolve(strict=True) != wrapper
        or wrapper.name not in PRODUCTION_WRAPPER_NAMES
        or wrapper != expected_wrapper
        or not isinstance(original_argv, list)
        or original_argv[:5] != expected_prefix
        or not sys.flags.isolated
        or not sys.flags.no_site
        or not sys.flags.dont_write_bytecode
        or not sys.flags.safe_path
    ):
        fail(
            "final activation wrapper requires root and exact "
            "/usr/bin/python3.12 -I -S -B absolute invocation"
        )
    try:
        initial = source_seal.capture_source_seal(root, required_uid=0)
    except source_seal.SourceSealError as error:
        raise FinalActivationError(
            "final activation wrapper source is not protected"
        ) from error
    commit_raw = _source_git(
        root,
        ("rev-parse", "--verify", "HEAD^{commit}"),
        "execution source Git commit",
    )
    tree_raw = _source_git(
        root,
        ("rev-parse", "--verify", "HEAD^{tree}"),
        "execution source Git tree",
    )
    try:
        commit = commit_raw.decode("ascii").strip()
        tree = tree_raw.decode("ascii").strip()
    except UnicodeError as error:
        raise FinalActivationError(
            "final activation wrapper source identity is invalid"
        ) from error
    _git(commit, "execution source commit")
    _git(tree, "execution source tree")
    _require_execution_source(
        initial,
        expected_commit=commit,
        expected_tree=tree,
        required_uid=0,
    )


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or IDENTIFIER_RE.fullmatch(value) is None:
        fail(f"{label} is invalid")
    return value


def _text(value: Any, label: str, maximum: int = 4_096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum
        or any(ord(character) < 0x20 for character in value)
    ):
        fail(f"{label} is invalid")
    return value


def _timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or TIMESTAMP_RE.fullmatch(value) is None:
        fail(f"{label} is not a UTC whole-second timestamp")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise FinalActivationError(f"{label} is invalid") from error
    return value


def _timestamp_value(value: Any, label: str) -> datetime:
    return datetime.strptime(_timestamp(value, label), "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )


def _absolute(path_value: Any, label: str, *, must_exist: bool) -> Path:
    if not isinstance(path_value, str):
        fail(f"{label} path is invalid")
    path = Path(path_value)
    if (
        os.fspath(path) != path_value
        or not path.is_absolute()
        or path.anchor != "/"
        or path_value.startswith("//")
        or Path(os.path.abspath(path)) != path
        or path.name in {"", ".", ".."}
        or ".." in path.parts
    ):
        fail(f"{label} path is not lexically canonical")
    if must_exist:
        try:
            path.resolve(strict=True)
        except OSError as error:
            raise FinalActivationError(f"{label} path is unavailable") from error
    return path


def _endpoint_url(value: Any, label: str) -> str:
    text = _text(value, label, 2_048)
    try:
        parsed = urllib.parse.urlsplit(text)
        port = parsed.port
    except ValueError as error:
        raise FinalActivationError(f"{label} is invalid") from error
    if (
        parsed.scheme != "http"
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.hostname
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
        or port is None
    ):
        fail(f"{label} must be a credential-free, explicit-port HTTP URL")
    hostname = parsed.hostname
    if hostname != "localhost":
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError as error:
            raise FinalActivationError(
                f"{label} host must be localhost or an IP literal"
            ) from error
        if not (address.is_loopback or address.is_private or address.is_link_local):
            fail(f"{label} host is outside the local/private boundary")
    return text


def _stable(
    path: Path,
    label: str,
    *,
    maximum: int = MAX_INPUT_BYTES,
    read_only: bool = False,
    single_link: bool = False,
) -> StableFileSnapshot:
    try:
        return stable_read_regular(
            path,
            label,
            maximum=maximum,
            require_read_only=read_only,
            require_single_link=single_link,
        )
    except Exception as error:
        raise FinalActivationError(f"{label} is unavailable or changed") from error


def _runtime_path(
    value: Any,
    *,
    base: Path,
    label: str,
    relative_only: bool = False,
) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or len(value.encode("utf-8")) > 4_096
    ):
        fail(f"{label} path is invalid")
    selected = Path(value)
    if os.fspath(selected) != value or any(
        component in {"", ".", ".."} for component in selected.parts
    ):
        fail(f"{label} path is not lexical")
    if relative_only:
        if selected.is_absolute():
            fail(f"{label} path is not relative")
        selected = base / selected
    elif not selected.is_absolute():
        selected = base / selected
    try:
        return runtime_seal._lexical_absolute(selected)
    except runtime_seal.RuntimeArtifactSealError as error:
        raise FinalActivationError(
            f"{label} path is not lexical absolute"
        ) from error


def _capture_runtime_artifact(
    path: Path,
    *,
    label: str,
    maximum: int,
    required_uid: int,
) -> runtime_seal.RuntimeArtifactSeal:
    try:
        return runtime_seal.capture_runtime_artifact_seal(
            path,
            label=label,
            maximum=maximum,
            required_uid=required_uid,
        )
    except runtime_seal.RuntimeArtifactSealError as error:
        raise FinalActivationError(
            f"{label} runtime artifact is not sealed"
        ) from error


def _capture_runtime_tree(
    path: Path,
    *,
    label: str,
    required_uid: int,
) -> runtime_seal.RuntimeTreeSeal:
    try:
        return runtime_seal.capture_runtime_tree_seal(
            path,
            label=label,
            required_uid=required_uid,
        )
    except runtime_seal.RuntimeArtifactSealError as error:
        raise FinalActivationError(
            f"{label} runtime tree is not sealed"
        ) from error


def _capture_manifest_runtime_seals(
    manifest: StableFileSnapshot,
    document: dict[str, Any],
    *,
    label: str,
    required_uid: int,
) -> ManifestRuntimeSeals:
    """Capture a non-empty exact-file and recursive runtime closure."""

    worker = document.get("worker")
    tokenizer = document.get("tokenizer")
    product = document.get("product")
    promotion = document.get("promotion")
    if not all(
        isinstance(value, dict)
        for value in (worker, tokenizer, product, promotion)
    ):
        fail(f"{label} runtime contract is incomplete")
    assert isinstance(worker, dict)
    assert isinstance(tokenizer, dict)
    assert isinstance(product, dict)
    assert isinstance(promotion, dict)

    manifest_seal = _capture_runtime_artifact(
        manifest.path,
        label=f"{label} manifest",
        maximum=MAX_MANIFEST_BYTES,
        required_uid=required_uid,
    )
    if (
        manifest_seal.snapshot.raw != manifest.raw
        or manifest_seal.snapshot.identity != manifest.identity
    ):
        fail(f"{label} manifest runtime seal differs")

    worker_path = _runtime_path(
        worker.get("binary"),
        base=manifest.path.parent,
        label=f"{label} worker",
    )
    worker_seal = _capture_runtime_artifact(
        worker_path,
        label=f"{label} worker binary",
        maximum=MAX_OUTPUT_FILE_BYTES,
        required_uid=required_uid,
    )
    if worker_seal.snapshot.sha256 != _hash(
        worker.get("binary_sha256"),
        f"{label} worker binary SHA-256",
    ):
        fail(f"{label} worker runtime bytes differ")

    product_root = _runtime_path(
        product.get("root"),
        base=manifest.path.parent,
        label=f"{label} product root",
    )
    receipt_path = _runtime_path(
        promotion.get("receipt"),
        base=manifest.path.parent,
        label=f"{label} promotion receipt",
    )
    try:
        receipt_path.relative_to(product_root)
    except ValueError:
        fail(f"{label} promotion receipt is outside the product root")
    receipt_seal = _capture_runtime_artifact(
        receipt_path,
        label=f"{label} promotion receipt",
        maximum=MAX_INPUT_BYTES,
        required_uid=required_uid,
    )
    if receipt_seal.snapshot.sha256 != _hash(
        promotion.get("receipt_sha256"),
        f"{label} promotion receipt SHA-256",
    ):
        fail(f"{label} promotion receipt bytes differ")
    receipt_document = _strict_object(
        receipt_seal.snapshot.raw,
        f"{label} promotion receipt",
    )
    evidence_reference = _exact(
        receipt_document.get("evidence"),
        {"path", "sha256"},
        f"{label} promotion evidence reference",
    )
    evidence_path = _runtime_path(
        evidence_reference["path"],
        base=receipt_path.parent,
        label=f"{label} promotion evidence",
        relative_only=True,
    )
    try:
        evidence_path.relative_to(product_root)
    except ValueError:
        fail(f"{label} promotion evidence is outside the product root")
    evidence_seal = _capture_runtime_artifact(
        evidence_path,
        label=f"{label} promotion evidence",
        maximum=MAX_INPUT_BYTES,
        required_uid=required_uid,
    )
    if evidence_seal.snapshot.sha256 != _hash(
        evidence_reference["sha256"],
        f"{label} promotion evidence SHA-256",
    ):
        fail(f"{label} promotion evidence bytes differ")

    package = product.get("package")
    artifact = product.get("artifact")
    if not isinstance(package, dict) or (
        artifact is not None and not isinstance(artifact, dict)
    ):
        fail(f"{label} product contract differs")
    declared_files: list[tuple[Path, str, int, str]] = [
        (
            _runtime_path(
                package.get("manifest_path"),
                base=product_root,
                label=f"{label} package manifest",
                relative_only=True,
            ),
            f"{label} package manifest",
            MAX_INPUT_BYTES,
            _hash(
                package.get("manifest_sha256"),
                f"{label} package manifest SHA-256",
            ),
        ),
    ]
    if isinstance(artifact, dict):
        declared_files.append(
            (
                _runtime_path(
                    artifact.get("manifest_path"),
                    base=product_root,
                    label=f"{label} artifact manifest",
                    relative_only=True,
                ),
                f"{label} artifact manifest",
                MAX_INPUT_BYTES,
                _hash(
                    artifact.get("manifest_sha256"),
                    f"{label} artifact manifest SHA-256",
                ),
            )
        )

    tokenizer_root = _runtime_path(
        tokenizer.get("root"),
        base=manifest.path.parent,
        label=f"{label} tokenizer root",
    )
    tokenizer_files = tokenizer.get("files")
    if (
        not isinstance(tokenizer_files, dict)
        or not tokenizer_files
        or len(tokenizer_files) > 128
    ):
        fail(f"{label} tokenizer file contract differs")
    for relative, expected_hash in sorted(
        tokenizer_files.items(),
        key=lambda item: os.fsencode(str(item[0])),
    ):
        if not isinstance(relative, str):
            fail(f"{label} tokenizer path is invalid")
        declared_files.append(
            (
                _runtime_path(
                    relative,
                    base=tokenizer_root,
                    label=f"{label} tokenizer file",
                    relative_only=True,
                ),
                f"{label} tokenizer file {relative}",
                MAX_OUTPUT_FILE_BYTES,
                _hash(
                    expected_hash,
                    f"{label} tokenizer file {relative} SHA-256",
                ),
            )
        )
    all_paths = [
        manifest.path,
        worker_path,
        receipt_path,
        evidence_path,
        *(path for path, _name, _maximum, _hash_value in declared_files),
    ]
    if len(set(all_paths)) != len(all_paths):
        fail(f"{label} runtime file paths are not distinct")

    declared_seals: list[runtime_seal.RuntimeArtifactSeal] = []
    for path, file_label, maximum, expected_hash in declared_files:
        sealed = _capture_runtime_artifact(
            path,
            label=file_label,
            maximum=maximum,
            required_uid=required_uid,
        )
        if sealed.snapshot.sha256 != expected_hash:
            fail(f"{file_label} runtime bytes differ")
        declared_seals.append(sealed)
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
    result = ManifestRuntimeSeals(
        manifest=manifest_seal,
        worker=worker_seal,
        promotion_receipt=receipt_seal,
        promotion_evidence=evidence_seal,
        artifacts=(
            manifest_seal,
            worker_seal,
            receipt_seal,
            evidence_seal,
            *declared_seals,
        ),
        trees=trees,
    )
    _require_manifest_runtime_seals(result, required_uid=required_uid)
    return result


def _require_artifact_seals(
    artifacts: tuple[runtime_seal.RuntimeArtifactSeal, ...],
    *,
    required_uid: int,
) -> None:
    if not artifacts:
        fail("runtime artifact seal collection is empty")
    try:
        for sealed in artifacts:
            runtime_seal.require_runtime_artifact_seal(
                sealed,
                required_uid=required_uid,
            )
    except runtime_seal.RuntimeArtifactSealError as error:
        raise FinalActivationError(
            "final activation runtime artifact seal changed"
        ) from error


def _require_manifest_runtime_seals(
    sealed: ManifestRuntimeSeals,
    *,
    required_uid: int,
) -> None:
    _require_artifact_seals(sealed.artifacts, required_uid=required_uid)
    if not sealed.trees:
        fail("runtime tree seal collection is empty")
    try:
        for tree in sealed.trees:
            runtime_seal.require_runtime_tree_seal(
                tree,
                required_uid=required_uid,
            )
    except runtime_seal.RuntimeArtifactSealError as error:
        raise FinalActivationError(
            "final activation runtime tree seal changed"
        ) from error


def _require_record_runtime_seals(
    record: PlanRecord,
    *,
    required_uid: int,
    scope: str,
) -> None:
    if scope not in {"all", "rollback", "rollback_core"}:
        fail("runtime seal scope is invalid")
    if scope != "rollback_core":
        _require_artifact_seals(
            record.shared_runtime_artifacts,
            required_uid=required_uid,
        )
        _require_artifact_seals(
            record.rollback_operation_artifacts,
            required_uid=required_uid,
        )
    _require_manifest_runtime_seals(
        record.rollback_runtime,
        required_uid=required_uid,
    )
    if scope == "all":
        if record.candidate_runtime is None:
            fail("candidate runtime seal collection is unavailable")
        _require_manifest_runtime_seals(
            record.candidate_runtime,
            required_uid=required_uid,
        )
        _require_artifact_seals(
            record.candidate_operation_artifacts,
            required_uid=required_uid,
        )


def _require_record_execution_source(
    record: PlanRecord,
    *,
    required_uid: int,
) -> None:
    source = record.document["source"]
    _require_execution_source(
        record.execution_source,
        expected_commit=source["commit"],
        expected_tree=source["tree"],
        required_uid=required_uid,
    )


def _load_module(name: str, path: Path) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        fail(f"validator is unavailable: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def default_manifest_validator(path: Path) -> dict[str, Any]:
    try:
        return _load_module(VALIDATOR_MODULE, VALIDATOR_PATH).validation_summary(path)
    except Exception as error:
        raise FinalActivationError("served-model validation failed") from error


def default_bundle_validator(path: Path) -> dict[str, Any]:
    try:
        return _load_module(BUNDLE_VALIDATOR_MODULE, BUNDLE_VALIDATOR_PATH).validate(
            path
        )
    except Exception as error:
        raise FinalActivationError("release-bundle validation failed") from error


def _summary_identity(
    summary: dict[str, Any],
    *,
    snapshot: StableFileSnapshot,
    model_id: str,
    format_id: str,
    label: str,
) -> str:
    worker = summary.get("worker")
    worker_sha256 = worker.get("binary_sha256") if isinstance(worker, dict) else None
    if (
        summary.get("validated") is not True
        or summary.get("manifest_sha256") != snapshot.sha256
        or summary.get("model_id") != model_id
        or summary.get("format_id") != format_id
        or not isinstance(worker, dict)
        or worker.get("protocol") != WORKER_PROTOCOL
        or not isinstance(worker_sha256, str)
        or HASH_RE.fullmatch(worker_sha256) is None
    ):
        fail(f"{label} served-model identity differs")
    return worker_sha256


def _policy_deployment_paths(
    policy: authorization.RegistryPolicy,
) -> tuple[Path, Path]:
    unit = getattr(policy, "systemd_unit_path", None)
    environment = getattr(policy, "environment_file_path", None)
    if not isinstance(unit, Path) or not isinstance(environment, Path):
        fail("registry policy lacks fixed systemd unit/environment paths")
    return (
        _absolute(os.fspath(unit), "policy systemd unit", must_exist=True),
        _absolute(
            os.fspath(environment),
            "policy systemd environment",
            must_exist=True,
        ),
    )


def _disallowed_executable_hashes() -> set[str]:
    digests: set[str] = set()
    for directory in (Path("/usr/bin"), Path("/bin"), Path("/usr/local/bin")):
        for name in DISALLOWED_COMMAND_EXECUTABLES:
            path = directory / name
            try:
                resolved = path.resolve(strict=True)
                snapshot = _stable(
                    resolved,
                    f"disallowed executable {name}",
                    maximum=MAX_INPUT_BYTES,
                )
            except (OSError, FinalActivationError):
                continue
            digests.add(snapshot.sha256)
    return digests


def _operation_document(
    snapshot: StableFileSnapshot,
    *,
    required_uid: int,
    verify_executables: bool,
    executable_stages: set[str] | None = None,
) -> dict[str, Any]:
    if (
        stat.S_IMODE(snapshot.identity.mode) != 0o444
        or snapshot.identity.links != 1
        or snapshot.identity.uid != required_uid
    ):
        fail("operations document must be immutable and owned by the required UID")
    document = _strict_object(snapshot.raw, "operations document")
    if _canonical_json(document) != snapshot.raw:
        fail("operations document is not canonical JSON")
    _exact(
        document,
        {
            "schema_version",
            "review_id",
            "reviewed_at",
            "reviewed_by",
            "timeout_seconds",
            "active_window_timeout_seconds",
            "live_proofs",
            "stages",
        },
        "operations document",
    )
    if document["schema_version"] != OPERATIONS_SCHEMA:
        fail("operations document schema differs")
    _identifier(document["review_id"], "operations.review_id")
    _timestamp(document["reviewed_at"], "operations.reviewed_at")
    _text(document["reviewed_by"], "operations.reviewed_by", 512)
    timeout = document["timeout_seconds"]
    if type(timeout) not in {int, float} or not 0 < timeout <= 3_600:
        fail("operations timeout is invalid")
    active_window_timeout = document["active_window_timeout_seconds"]
    if (
        type(active_window_timeout) not in {int, float}
        or not 0 < active_window_timeout <= MAX_ACTIVE_WINDOW_SECONDS
        or active_window_timeout < timeout
    ):
        fail("operations active-window timeout is invalid")
    live_proofs = _exact(
        document["live_proofs"],
        {"candidate_live_health", "rollback_live_health"},
        "operations.live_proofs",
    )
    proof_paths: set[Path] = set()
    for stage, value in live_proofs.items():
        proof = _exact(
            value,
            LIVE_PROOF_SPEC_FIELDS,
            f"operations.live_proofs.{stage}",
        )
        proof_path = _absolute(
            proof["path"],
            f"operations.live_proofs.{stage}",
            must_exist=False,
        )
        if proof_path in proof_paths:
            fail("candidate and rollback live-proof paths collide")
        proof_paths.add(proof_path)
        _text(
            proof["service_unit"],
            f"operations.live_proofs.{stage}.service_unit",
            512,
        )
        _hash(
            proof["gateway_executable_sha256"],
            f"operations.live_proofs.{stage}.gateway_executable_sha256",
        )
        endpoint_urls = _exact(
            proof["endpoint_urls"],
            ENDPOINT_NAMES,
            f"operations.live_proofs.{stage}.endpoint_urls",
        )
        for endpoint, url in endpoint_urls.items():
            _endpoint_url(
                url,
                f"operations.live_proofs.{stage}.endpoint_urls.{endpoint}",
            )
    stages = _exact(document["stages"], OPERATION_STAGES, "operations.stages")
    disallowed_hashes = _disallowed_executable_hashes()
    for stage, commands in stages.items():
        verify_stage = verify_executables and (
            executable_stages is None or stage in executable_stages
        )
        if (
            not isinstance(commands, list)
            or not commands
            or len(commands) > MAX_COMMANDS_PER_STAGE
        ):
            fail(f"operations stage {stage} must contain bounded commands")
        for index, command in enumerate(commands):
            entry = _exact(
                command,
                {"argv", "executable_sha256"},
                f"operations.{stage}[{index}]",
            )
            argv = entry["argv"]
            if (
                not isinstance(argv, list)
                or not argv
                or len(argv) > MAX_ARGUMENTS
                or any(
                    not isinstance(argument, str)
                    or not argument
                    or "\x00" in argument
                    or len(argument.encode("utf-8")) > MAX_ARGUMENT_BYTES
                    for argument in argv
                )
            ):
                fail(f"operations.{stage}[{index}].argv is invalid")
            executable = _absolute(
                argv[0],
                f"operations.{stage}[{index}].argv[0]",
                must_exist=verify_stage,
            )
            if executable.name in DISALLOWED_COMMAND_EXECUTABLES:
                fail(f"operations.{stage}[{index}] uses a shell or interpreter wrapper")
            expected = _hash(
                entry["executable_sha256"],
                f"operations.{stage}[{index}].executable_sha256",
            )
            if expected in disallowed_hashes:
                fail(f"operations.{stage}[{index}] pins a renamed command wrapper")
            if verify_stage:
                executable_snapshot = _stable(
                    executable,
                    f"operations executable {stage}[{index}]",
                    maximum=MAX_INPUT_BYTES,
                )
                if executable_snapshot.raw.startswith(b"#!"):
                    fail(
                        f"operations executable {stage}[{index}] is an interpreter script"
                    )
                if (
                    executable_snapshot.sha256 != expected
                    or executable_snapshot.identity.uid not in {0, required_uid}
                    or executable_snapshot.identity.links != 1
                    or stat.S_IMODE(executable_snapshot.identity.mode) & 0o022
                    or not executable_snapshot.identity.mode
                    & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                ):
                    fail(f"operations executable {stage}[{index}] differs")
    return document


def _capture_operation_executable_seals(
    operations: dict[str, Any],
    *,
    required_uid: int,
    stages: set[str],
) -> tuple[runtime_seal.RuntimeArtifactSeal, ...]:
    expected_by_path: dict[Path, str] = {}
    labels: dict[Path, str] = {}
    for stage in sorted(operations["stages"]):
        if stage not in stages:
            continue
        for index, command in enumerate(operations["stages"][stage]):
            path = Path(command["argv"][0])
            expected = command["executable_sha256"]
            previous = expected_by_path.setdefault(path, expected)
            if previous != expected:
                fail("reviewed executable path has conflicting SHA-256 values")
            labels.setdefault(path, f"operations executable {stage}[{index}]")
    if not expected_by_path:
        fail("reviewed executable runtime seal collection is empty")
    result: list[runtime_seal.RuntimeArtifactSeal] = []
    for path in sorted(expected_by_path, key=os.fsencode):
        sealed = _capture_runtime_artifact(
            path,
            label=labels[path],
            maximum=MAX_INPUT_BYTES,
            required_uid=required_uid,
        )
        if sealed.snapshot.sha256 != expected_by_path[path]:
            fail(f"{labels[path]} runtime bytes differ")
        result.append(sealed)
    return tuple(result)


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum or value > (1 << 63) - 1:
        fail(f"{label} is invalid")
    return value


def _live_proof_expectation(
    record: PlanRecord,
    stage: str,
) -> tuple[dict[str, Any], str, str, str]:
    if stage == "candidate_live_health":
        identity = record.document["candidate"]
    elif stage == "rollback_live_health":
        identity = record.document["rollback"]
    else:
        fail("live-proof stage is invalid")
    specification = record.document["live_proofs"][stage]
    return (
        specification,
        identity["model_id"],
        identity["format_id"],
        identity["worker_binary_sha256"],
    )


def default_live_proof_loader(
    record: PlanRecord,
    stage: str,
    _activation_epoch: str,
) -> dict[str, Any]:
    specification, _model_id, _format_id, _worker_sha256 = _live_proof_expectation(
        record, stage
    )
    snapshot = _stable(
        Path(specification["path"]),
        f"{stage} live proof",
        maximum=MAX_DOCUMENT_BYTES,
        read_only=True,
        single_link=True,
    )
    if snapshot.identity.uid != record.snapshot.identity.uid:
        fail(f"{stage} live-proof owner differs")
    document = _strict_object(snapshot.raw, f"{stage} live proof")
    if _canonical_json(document) != snapshot.raw:
        fail(f"{stage} live proof is not canonical JSON")
    return document


def _validate_live_proof(
    record: PlanRecord,
    stage: str,
    activation_epoch: str,
    document: dict[str, Any],
    *,
    stage_started: datetime | None = None,
    verified_at: datetime | None = None,
) -> dict[str, Any]:
    specification, model_id, format_id, worker_sha256 = _live_proof_expectation(
        record,
        stage,
    )
    _exact(document, LIVE_PROOF_FIELDS, f"{stage} live proof")
    if (
        document["schema_version"] != LIVE_PROOF_SCHEMA
        or document["plan_sha256"] != record.snapshot.sha256
        or document["stage"] != stage
        or document["activation_epoch"] != activation_epoch
        or document["epoch_stable"] is not True
        or document["passed"] is not True
    ):
        fail(f"{stage} live-proof root identity differs")
    captured_at = _timestamp_value(
        document["captured_at"],
        f"{stage} live proof captured_at",
    )
    if (stage_started is None) != (verified_at is None):
        fail(f"{stage} live-proof freshness boundary is incomplete")
    if stage_started is not None and verified_at is not None:
        lower = stage_started.astimezone(timezone.utc).replace(microsecond=0)
        upper = (
            verified_at.astimezone(timezone.utc).replace(microsecond=0)
            + timedelta(seconds=MAX_LIVE_PROOF_CLOCK_SKEW_SECONDS)
        )
        if captured_at < lower or captured_at > upper:
            fail(f"{stage} live-proof capture is outside its live stage")

    active = _exact(
        document["active_manifest"],
        LIVE_PROOF_ACTIVE_FIELDS,
        f"{stage} live proof active_manifest",
    )
    expected_manifest = (
        record.candidate.sha256
        if stage == "candidate_live_health"
        else record.rollback.sha256
    )
    if active != {
        "path": record.document["active_manifest"]["path"],
        "manifest_sha256": expected_manifest,
        "model_id": model_id,
        "format_id": format_id,
        "worker_protocol": WORKER_PROTOCOL,
        "worker_binary_sha256": worker_sha256,
    }:
        fail(f"{stage} live-proof active model identity differs")

    service = _exact(
        document["service"],
        LIVE_PROOF_SERVICE_FIELDS,
        f"{stage} live proof service",
    )
    if (
        service["unit"] != specification["service_unit"]
        or service["active_state"] != "active"
        or service["sub_state"] != "running"
        or not isinstance(service["boot_id"], str)
        or BOOT_ID_RE.fullmatch(service["boot_id"]) is None
        or service["fragment_path"] != os.fspath(record.unit.path)
        or service["environment_file_path"] != os.fspath(record.environment.path)
        or not isinstance(service["control_group"], str)
        or not service["control_group"].startswith("/")
    ):
        fail(f"{stage} live-proof service identity differs")
    _positive_int(
        service["n_restarts"],
        f"{stage} live proof service restart count",
        allow_zero=True,
    )
    _positive_int(service["main_pid"], f"{stage} live proof service main PID")

    gateway = _exact(
        document["gateway"],
        LIVE_PROOF_PROCESS_FIELDS,
        f"{stage} live proof gateway",
    )
    worker = _exact(
        document["worker"],
        LIVE_PROOF_PROCESS_FIELDS,
        f"{stage} live proof worker",
    )
    for label, process in (("gateway", gateway), ("worker", worker)):
        _positive_int(process["pid"], f"{stage} live proof {label} PID")
        _positive_int(
            process["ppid"],
            f"{stage} live proof {label} PPID",
            allow_zero=label == "gateway",
        )
        _positive_int(
            process["starttime_ticks"],
            f"{stage} live proof {label} starttime",
        )
        _hash(
            process["executable_sha256"],
            f"{stage} live proof {label} executable SHA-256",
        )
    if (
        gateway["pid"] == worker["pid"]
        or worker["ppid"] != gateway["pid"]
        or service["main_pid"] != gateway["pid"]
        or gateway["executable_sha256"] != specification["gateway_executable_sha256"]
        or worker["executable_sha256"] != worker_sha256
    ):
        fail(f"{stage} live-proof process identity differs")

    endpoints = _exact(
        document["endpoints"],
        LIVE_PROOF_ENDPOINT_FIELDS,
        f"{stage} live proof endpoints",
    )
    for name in ("gateway_healthz", "gateway_readyz", "openwebui_health"):
        endpoint = _exact(
            endpoints[name],
            {"status"},
            f"{stage} live proof endpoint {name}",
        )
        if endpoint["status"] != 200:
            fail(f"{stage} live-proof endpoint {name} failed")
    for name in ("gateway_models", "openwebui_models"):
        endpoint = _exact(
            endpoints[name],
            {"status", "model_ids"},
            f"{stage} live proof endpoint {name}",
        )
        if endpoint["status"] != 200 or endpoint["model_ids"] != [model_id]:
            fail(f"{stage} live-proof endpoint {name} differs")

    return {
        "path": specification["path"],
        "sha256": _sha256(_canonical_json(document)),
        "schema_version": LIVE_PROOF_SCHEMA,
        "activation_epoch": activation_epoch,
    }


def _read_boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except (OSError, UnicodeError) as error:
        raise FinalActivationError("kernel boot ID is unavailable") from error
    if BOOT_ID_RE.fullmatch(value) is None:
        fail("kernel boot ID is invalid")
    return value


def _proc_stat_identity(pid: int) -> tuple[int, int]:
    try:
        raw = Path(f"/proc/{pid}/stat").read_bytes()
    except OSError as error:
        raise FinalActivationError(f"live process {pid} is unavailable") from error
    if len(raw) > 65_536:
        fail(f"live process {pid} stat exceeds its byte bound")
    close = raw.rfind(b")")
    if close < 1:
        fail(f"live process {pid} stat is malformed")
    fields = raw[close + 2 :].split()
    if len(fields) < 20:
        fail(f"live process {pid} stat is truncated")
    try:
        ppid = int(fields[1], 10)
        starttime = int(fields[19], 10)
    except ValueError as error:
        raise FinalActivationError(f"live process {pid} stat is invalid") from error
    if ppid < 0 or starttime < 1:
        fail(f"live process {pid} stat identity is invalid")
    return ppid, starttime


def _hash_proc_executable(pid: int) -> str:
    descriptor = -1
    try:
        descriptor = os.open(
            f"/proc/{pid}/exe",
            os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size < 1
            or metadata.st_size > MAX_INPUT_BYTES
        ):
            fail(f"live process {pid} executable metadata is unsafe")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                fail(f"live process {pid} executable changed while read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_dev != metadata.st_dev
            or after.st_ino != metadata.st_ino
            or after.st_size != metadata.st_size
        ):
            fail(f"live process {pid} executable changed while read")
        return _sha256(b"".join(chunks))
    except OSError as error:
        raise FinalActivationError(
            f"live process {pid} executable is unavailable"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _process_live_identity(pid: int) -> dict[str, int | str]:
    ppid, starttime = _proc_stat_identity(pid)
    executable_sha256 = _hash_proc_executable(pid)
    observed_ppid, observed_starttime = _proc_stat_identity(pid)
    if (observed_ppid, observed_starttime) != (ppid, starttime):
        fail(f"live process {pid} epoch changed while inspected")
    return {
        "pid": pid,
        "ppid": ppid,
        "starttime_ticks": starttime,
        "executable_sha256": executable_sha256,
    }


def _parse_systemctl_show(raw: bytes) -> dict[str, str]:
    if len(raw) > MAX_DOCUMENT_BYTES:
        fail("systemctl show output exceeds its byte bound")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as error:
        raise FinalActivationError("systemctl show output is not UTF-8") from error
    result: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in result:
            fail("systemctl show output is malformed")
        result[key] = value
    expected = {
        "ActiveState",
        "SubState",
        "MainPID",
        "NRestarts",
        "ControlGroup",
        "FragmentPath",
        "EnvironmentFiles",
    }
    if set(result) != expected:
        fail("systemctl show properties differ")
    return result


def _systemd_live_state(
    record: PlanRecord,
    stage: str,
    *,
    timeout: float,
) -> dict[str, Any]:
    specification, _model_id, _format_id, _worker_sha256 = _live_proof_expectation(
        record,
        stage,
    )
    systemctl = _stable(
        SYSTEMCTL_PATH,
        "systemctl executable",
        maximum=MAX_INPUT_BYTES,
    )
    if (
        systemctl.identity.uid != 0
        or systemctl.identity.links != 1
        or stat.S_IMODE(systemctl.identity.mode) & 0o022
        or not systemctl.identity.mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        or systemctl.raw.startswith(b"#!")
    ):
        fail("systemctl executable metadata is unsafe")
    command = [
        os.fspath(SYSTEMCTL_PATH),
        "show",
        specification["service_unit"],
        "--no-pager",
        "--property=ActiveState",
        "--property=SubState",
        "--property=MainPID",
        "--property=NRestarts",
        "--property=ControlGroup",
        "--property=FragmentPath",
        "--property=EnvironmentFiles",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd="/",
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={"LANG": "C", "LC_ALL": "C"},
            timeout=max(0.1, timeout),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise FinalActivationError("independent systemd live query failed") from error
    if completed.returncode != 0:
        fail("independent systemd live query failed")
    properties = _parse_systemctl_show(completed.stdout)
    try:
        main_pid = int(properties["MainPID"], 10)
        n_restarts = int(properties["NRestarts"], 10)
    except ValueError as error:
        raise FinalActivationError("systemd numeric properties are invalid") from error
    fragment = Path(properties["FragmentPath"])
    if not fragment.is_absolute() or fragment != record.unit.path:
        fail("live systemd fragment path differs from the plan")
    try:
        environment_tokens = shlex.split(properties["EnvironmentFiles"])
    except ValueError as error:
        raise FinalActivationError("systemd EnvironmentFiles is invalid") from error
    environment_paths = {
        token.removeprefix("-")
        for token in environment_tokens
        if token.startswith("/") or token.startswith("-/")
    }
    if os.fspath(record.environment.path) not in environment_paths:
        fail("live systemd environment path differs from the plan")
    return {
        "unit": specification["service_unit"],
        "active_state": properties["ActiveState"],
        "sub_state": properties["SubState"],
        "n_restarts": n_restarts,
        "main_pid": main_pid,
        "control_group": properties["ControlGroup"],
        "fragment_path": properties["FragmentPath"],
        "environment_file_path": os.fspath(record.environment.path),
    }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


def _runtime_secret_seal(
    path: Path,
    label: str,
    *,
    required_uid: int,
) -> runtime_seal.RuntimeArtifactSeal:
    if path == campaign_plan.API_KEY_FILE:
        expected_mode = 0o640
        expected_uid = 0
        expected_gid = 1000
    elif path == campaign_plan.OPENWEBUI_SESSION_TOKEN_FILE:
        expected_mode = 0o640
        expected_uid = 0
        expected_gid = 1000
    else:
        fail(f"{label} path differs from the fixed credential policy")
    sealed = _capture_runtime_artifact(
        path,
        label=label,
        maximum=65_536,
        required_uid=expected_uid,
    )
    snapshot = sealed.snapshot
    if (
        stat.S_IMODE(snapshot.identity.mode) != expected_mode
        or snapshot.identity.uid != expected_uid
        or snapshot.identity.links != 1
        or (expected_gid is not None and snapshot.identity.gid != expected_gid)
    ):
        fail(f"{label} private metadata differs")
    if path == campaign_plan.OPENWEBUI_SESSION_TOKEN_FILE:
        if not sealed.ancestry:
            fail(f"{label} private parent metadata differs")
        parent = sealed.ancestry[-1]
        if (
            path.parent != OPENWEBUI_SESSION_TOKEN_PARENT
            or parent.path != OPENWEBUI_SESSION_TOKEN_PARENT
            or parent.uid != OPENWEBUI_SESSION_TOKEN_PARENT_UID
            or parent.gid != OPENWEBUI_SESSION_TOKEN_PARENT_GID
            or stat.S_IMODE(parent.mode) != OPENWEBUI_SESSION_TOKEN_PARENT_MODE
        ):
            fail(f"{label} private parent metadata differs")
    return sealed


def _read_runtime_secret(
    path: Path,
    label: str,
    *,
    required_uid: int,
) -> bytearray:
    sealed = _runtime_secret_seal(
        path,
        label,
        required_uid=required_uid,
    )
    value = bytearray(sealed.snapshot.raw.strip())
    if (
        not value
        or len(value) > 16_384
        or any(byte < 0x21 or byte > 0x7E for byte in value)
    ):
        for index in range(len(value)):
            value[index] = 0
        fail(f"{label} is invalid")
    return value


def _capture_live_credential_seals(
    *,
    required_uid: int,
) -> tuple[runtime_seal.RuntimeArtifactSeal, ...]:
    return (
        _runtime_secret_seal(
            campaign_plan.API_KEY_FILE,
            "gateway API key",
            required_uid=required_uid,
        ),
        _runtime_secret_seal(
            campaign_plan.OPENWEBUI_SESSION_TOKEN_FILE,
            "OpenWebUI session token",
            required_uid=required_uid,
        ),
    )


def _require_live_credential_seals(
    sealed: tuple[runtime_seal.RuntimeArtifactSeal, ...],
) -> None:
    if len(sealed) != 2:
        fail("live credential runtime seal collection differs")
    try:
        for artifact in sealed:
            runtime_seal.require_runtime_artifact_seal(
                artifact,
                required_uid=artifact.required_uid,
            )
    except runtime_seal.RuntimeArtifactSealError as error:
        raise FinalActivationError(
            "live credential runtime seal changed"
        ) from error


def _endpoint_live_state(
    name: str,
    url: str,
    *,
    timeout: float,
    required_uid: int,
) -> tuple[int, bytes]:
    secret: bytearray | None = None
    headers = {"Accept": "application/json", "Connection": "close"}
    if name == "gateway_models":
        secret = _read_runtime_secret(
            campaign_plan.API_KEY_FILE,
            "gateway API key",
            required_uid=required_uid,
        )
    elif name == "openwebui_models":
        secret = _read_runtime_secret(
            campaign_plan.OPENWEBUI_SESSION_TOKEN_FILE,
            "OpenWebUI session token",
            required_uid=required_uid,
        )
    if secret is not None:
        try:
            headers["Authorization"] = f"Bearer {secret.decode('ascii')}"
        except UnicodeError as error:
            for index in range(len(secret)):
                secret[index] = 0
            raise FinalActivationError("runtime endpoint credential is invalid") from error
    request = urllib.request.Request(
        url,
        headers=headers,
        method="GET",
    )
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        try:
            with opener.open(request, timeout=max(0.1, timeout)) as response:
                status = int(response.status)
                raw = response.read(MAX_ENDPOINT_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            status = int(error.code)
            raw = error.read(MAX_ENDPOINT_RESPONSE_BYTES + 1)
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            raise FinalActivationError(
                "independent endpoint live query failed"
            ) from error
    finally:
        if secret is not None:
            for index in range(len(secret)):
                secret[index] = 0
        headers.pop("Authorization", None)
    if len(raw) > MAX_ENDPOINT_RESPONSE_BYTES:
        fail("independent endpoint response exceeds its byte bound")
    return status, raw


def _model_ids_from_response(raw: bytes) -> list[str]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FinalActivationError("model endpoint response is not strict JSON") from error
    if isinstance(value, dict):
        candidates = value.get("data", value.get("models"))
    else:
        candidates = value
    if not isinstance(candidates, list):
        fail("model endpoint response lacks a model list")
    model_ids: list[str] = []
    for item in candidates:
        model_id = item.get("id") if isinstance(item, dict) else item
        if not isinstance(model_id, str) or not model_id:
            fail("model endpoint response contains an invalid model ID")
        model_ids.append(model_id)
    return model_ids


def default_live_state_verifier(
    record: PlanRecord,
    stage: str,
    document: dict[str, Any],
    _stage_started: datetime,
    _verified_at: datetime,
    timeout: float,
) -> None:
    """Independently re-observe proof claims from systemd, procfs, and HTTP."""

    specification, model_id, _format_id, _worker_sha256 = _live_proof_expectation(
        record,
        stage,
    )
    service = document["service"]
    if service["unit"] != authorization.FIXED_SERVICE_UNIT:
        fail(f"{stage} live proof targets a non-production service")
    before_gateway = _process_live_identity(document["gateway"]["pid"])
    before_worker = _process_live_identity(document["worker"]["pid"])
    systemd = _systemd_live_state(record, stage, timeout=min(timeout, 10.0))
    if (
        document["gateway"] != before_gateway
        or document["worker"] != before_worker
        or service["boot_id"] != _read_boot_id()
        or any(service[field] != systemd[field] for field in systemd)
    ):
        fail(f"{stage} live proof differs from independent process/systemd state")
    endpoint_urls = specification["endpoint_urls"]
    for name in sorted(ENDPOINT_NAMES):
        status, raw = _endpoint_live_state(
            name,
            endpoint_urls[name],
            timeout=min(timeout, 10.0),
            required_uid=record.snapshot.identity.uid,
        )
        expected = document["endpoints"][name]
        if status != expected["status"]:
            fail(f"{stage} endpoint {name} differs from independent query")
        if name in {"gateway_models", "openwebui_models"}:
            if _model_ids_from_response(raw) != [model_id]:
                fail(f"{stage} endpoint {name} exposes a different model")
    if (
        _process_live_identity(document["gateway"]["pid"]) != before_gateway
        or _process_live_identity(document["worker"]["pid"]) != before_worker
        or service["boot_id"] != _read_boot_id()
    ):
        fail(f"{stage} live process epoch changed across endpoint probes")


def _validate_live_proof_reference(
    record: PlanRecord,
    stage: str,
    reference: dict[str, Any],
    *,
    read_named: bool = True,
) -> None:
    value = _exact(
        reference,
        {"path", "sha256", "schema_version", "activation_epoch"},
        f"{stage} live-proof reference",
    )
    specification, _model_id, _format_id, _worker_sha256 = _live_proof_expectation(
        record, stage
    )
    if (
        value["path"] != specification["path"]
        or value["schema_version"] != LIVE_PROOF_SCHEMA
    ):
        fail(f"{stage} live-proof reference identity differs")
    _hash(value["sha256"], f"{stage} live-proof reference SHA-256")
    epoch = _hash(
        value["activation_epoch"],
        f"{stage} live-proof reference activation epoch",
    )
    if read_named:
        proof = default_live_proof_loader(record, stage, epoch)
        observed = _validate_live_proof(record, stage, epoch, proof)
        if observed != value:
            fail(f"{stage} live-proof reference changed")


def _validate_live_proof_envelope(
    record: PlanRecord,
    stage: str,
    envelope: dict[str, Any],
    *,
    read_named: bool,
) -> None:
    value = _exact(
        envelope,
        LIVE_PROOF_ENVELOPE_FIELDS,
        f"{stage} live-proof envelope",
    )
    reference = value["reference"]
    document = value["document"]
    if not isinstance(reference, dict) or not isinstance(document, dict):
        fail(f"{stage} live-proof envelope values differ")
    _validate_live_proof_reference(
        record,
        stage,
        reference,
        read_named=read_named,
    )
    observed = _validate_live_proof(
        record,
        stage,
        reference["activation_epoch"],
        document,
    )
    if observed != reference:
        fail(f"{stage} embedded live proof differs from its reference")


def _inventory_file(path: Path, label: str) -> tuple[int, str]:
    snapshot = _stable(path, label, maximum=MAX_OUTPUT_FILE_BYTES)
    return len(snapshot.raw), snapshot.sha256


def _output_member_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _enumerate_output_tree(
    path: Path,
) -> tuple[tuple[int, ...], tuple[tuple[str, str, tuple[int, ...]], ...]]:
    try:
        root = path.lstat()
        members = sorted(path.rglob("*"), key=lambda item: item.as_posix())
    except OSError as error:
        raise FinalActivationError("campaign output cannot be enumerated") from error
    if not stat.S_ISDIR(root.st_mode) or stat.S_ISLNK(root.st_mode):
        fail("campaign output directory changed type")
    result: list[tuple[str, str, tuple[int, ...]]] = []
    for member in members:
        try:
            metadata = member.lstat()
        except OSError as error:
            raise FinalActivationError(
                "campaign output member changed during enumeration"
            ) from error
        relative = member.relative_to(path).as_posix()
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            kind = "directory"
        elif stat.S_ISREG(metadata.st_mode):
            kind = "file"
        else:
            fail("campaign output contains a non-regular member")
        result.append((relative, kind, _output_member_identity(metadata)))
    return _output_member_identity(root), tuple(result)


def _output_inventory(path: Path, *, run_id: str) -> dict[str, Any]:
    if not path.is_absolute():
        fail("campaign output path is not absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise FinalActivationError("campaign output is unavailable") from error
    selected: dict[str, str] = {}
    if stat.S_ISREG(metadata.st_mode):
        size, digest = _inventory_file(path, "campaign output")
        selected[path.name] = digest
        inventory = [{"path": path.name, "bytes": size, "sha256": digest}]
        kind = "file"
        inventory_sha256 = digest
    elif stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
        inventory = []
        before_root, before_members = _enumerate_output_tree(path)
        for relative, kind_value, _identity in before_members:
            member = path / relative
            if kind_value == "directory":
                continue
            if len(inventory) >= MAX_OUTPUT_FILES:
                fail("campaign output has too many files")
            size, digest = _inventory_file(member, "campaign output member")
            inventory.append({"path": relative, "bytes": size, "sha256": digest})
            if (
                relative in SELECTED_ARTIFACTS
                or Path(relative).name in SELECTED_ARTIFACTS
            ):
                selected[relative] = digest
        after_root, after_members = _enumerate_output_tree(path)
        if before_root != after_root or before_members != after_members:
            fail("campaign output changed while it was inventoried")
        if not inventory:
            fail("campaign output directory is empty")
        kind = "directory"
        inventory_sha256 = _sha256(_canonical_json({"files": inventory}))
    else:
        fail("campaign output has an unsafe type")
    total_bytes = sum(int(item["bytes"]) for item in inventory)
    if total_bytes < 1 or total_bytes > MAX_OUTPUT_TOTAL_BYTES:
        fail("campaign output byte total is invalid")
    return {
        "run_id": run_id,
        "path": os.fspath(path),
        "kind": kind,
        "sha256": inventory_sha256,
        "artifact_count": len(inventory),
        "total_bytes": total_bytes,
        "selected_artifacts": selected,
    }


def _campaign_outputs_unchanged(
    outcome: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if (
        authorization.CAMPAIGN_FIELDS != FINAL_CAMPAIGN_FIELDS
        or not isinstance(outcome.get("campaigns"), dict)
        or set(outcome["campaigns"]) != FINAL_CAMPAIGN_FIELDS
    ):
        fail("final activation requires all six fresh campaign outputs")
    observed: dict[str, dict[str, Any]] = {}
    for name in sorted(authorization.CAMPAIGN_FIELDS):
        expected = outcome["campaigns"][name]
        if not isinstance(expected, dict):
            fail("successful transaction outcome lacks a campaign output")
        current = _output_inventory(
            Path(expected["path"]),
            run_id=expected["run_id"],
        )
        if current != expected:
            fail(
                f"fresh campaign output changed after the restored-AQ4 transaction: {name}"
            )
        observed[name] = current
    return observed


def _resolve_bundle_component(
    bundle_path: Path,
    component: Any,
    label: str,
) -> tuple[Path, str]:
    entry = _exact(component, {"path", "sha256"}, f"bundle.artifacts.{label}")
    relative = entry["path"]
    if (
        not isinstance(relative, str)
        or not relative
        or len(relative.encode("utf-8")) > 1_024
    ):
        fail(f"bundle.artifacts.{label}.path is invalid")
    child = Path(relative)
    if (
        child.is_absolute()
        or not child.parts
        or any(part in {"", ".", ".."} for part in child.parts)
    ):
        fail(f"bundle.artifacts.{label}.path is unsafe")
    path = bundle_path.parent / child
    snapshot = _stable(path, f"bundle artifact {label}", maximum=MAX_INPUT_BYTES)
    expected = _hash(entry["sha256"], f"bundle.artifacts.{label}.sha256")
    if snapshot.sha256 != expected:
        fail(f"bundle artifact {label} SHA-256 differs")
    return path, expected


def _relative_to_output(
    artifact: Path,
    output: dict[str, Any],
    *,
    label: str,
) -> str:
    root = Path(output["path"])
    if output["kind"] == "file":
        if artifact != root:
            fail(f"{label} is outside its fresh campaign output")
        return root.name
    try:
        relative = artifact.relative_to(root).as_posix()
    except ValueError as error:
        raise FinalActivationError(
            f"{label} is outside its fresh campaign output"
        ) from error
    if not relative or relative.startswith("../"):
        fail(f"{label} is outside its fresh campaign output")
    return relative


def _require_selected_campaign_artifact(
    *,
    bundle_path: Path,
    artifacts: dict[str, Any],
    artifact_name: str,
    output: dict[str, Any],
) -> None:
    path, digest = _resolve_bundle_component(
        bundle_path,
        artifacts[artifact_name],
        artifact_name,
    )
    relative = _relative_to_output(
        path,
        output,
        label=f"bundle artifact {artifact_name}",
    )
    if output["selected_artifacts"].get(relative) != digest:
        fail(f"bundle artifact {artifact_name} is not bound by campaign outcome")


def _campaign_claim_reference(outcome: dict[str, Any]) -> dict[str, Any]:
    claim = _stable(
        Path(outcome["claim_path"]),
        "campaign authorization claim",
        maximum=MAX_DOCUMENT_BYTES,
        read_only=True,
        single_link=True,
    )
    if claim.sha256 != outcome["claim_sha256"]:
        fail("campaign outcome claim bytes changed")
    return {
        "path": os.fspath(claim.path),
        "sha256": claim.sha256,
        "bytes": len(claim.raw),
        "authorization_path": outcome["authorization_path"],
        "authorization_sha256": outcome["authorization_sha256"],
    }


def _bundle_json_artifact(
    bundle_path: Path,
    artifacts: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    path, _digest = _resolve_bundle_component(
        bundle_path,
        artifacts[name],
        name,
    )
    snapshot = _stable(path, f"bundle artifact {name}", maximum=MAX_INPUT_BYTES)
    document = _strict_object(snapshot.raw, f"bundle artifact {name}")
    if _canonical_json(document) != snapshot.raw:
        fail(f"bundle artifact {name} is not canonical JSON")
    return document


def _require_generic_campaign_lineage(
    document: dict[str, Any],
    *,
    name: str,
    output: dict[str, Any],
    claim: dict[str, Any],
) -> None:
    lineage = document.get("campaign_lineage")
    if not isinstance(lineage, dict):
        fail(f"bundle {name} campaign lineage is missing")
    campaign = lineage.get("campaign")
    if (
        lineage.get("schema_version") != "ullm.served_model.campaign_lineage.v2"
        or lineage.get("claim") != claim
        or not isinstance(campaign, dict)
        or campaign.get("name") != name
        or campaign.get("run_id") != output["run_id"]
        or campaign.get("final_path") != output["path"]
        or campaign.get("final_kind") != output["kind"]
    ):
        fail(f"bundle {name} lineage differs from the successful campaign outcome")


def _bind_bundle_to_campaign_outputs(
    bundle_path: Path,
    bundle_document: dict[str, Any],
    bundle_report: dict[str, Any],
    outputs: dict[str, dict[str, Any]],
    outcome: dict[str, Any],
) -> None:
    artifacts = bundle_document["artifacts"]
    for name in (
        "model_campaign_manifest",
        "model_campaign_evidence",
        "model_campaign_validator",
    ):
        _require_selected_campaign_artifact(
            bundle_path=bundle_path,
            artifacts=artifacts,
            artifact_name=name,
            output=outputs["sq8_full"],
        )
    # The browser campaign output is the evidence file itself.  Validator
    # reports and generic release evidence are independently recomputed after
    # the transaction, so their authority is the complete bundle validator.
    _require_selected_campaign_artifact(
        bundle_path=bundle_path,
        artifacts=artifacts,
        artifact_name="browser_evidence",
        output=outputs["reasoning_browser"],
    )
    claim = _campaign_claim_reference(outcome)
    release = _bundle_json_artifact(bundle_path, artifacts, "release_evidence")
    browser = _bundle_json_artifact(bundle_path, artifacts, "browser_evidence")
    campaign_identity = _bundle_json_artifact(
        bundle_path,
        artifacts,
        "model_campaign_evidence",
    )
    _require_generic_campaign_lineage(
        release,
        name="reasoning_release",
        output=outputs["reasoning_release"],
        claim=claim,
    )
    _require_generic_campaign_lineage(
        browser,
        name="reasoning_browser",
        output=outputs["reasoning_browser"],
        claim=claim,
    )
    release_campaign = _exact(
        bundle_report.get("reasoning_release_campaign"),
        {
            "campaign_name",
            "run_id",
            "final_path",
            "kind",
            "sha256",
            "artifact_inventory_sha256",
            "artifact_count",
            "total_bytes",
            "selected_artifacts",
            "claim_path",
            "claim_sha256",
            "authorization_path",
            "authorization_sha256",
        },
        "bundle validator reasoning_release_campaign",
    )
    release_output = outputs["reasoning_release"]
    expected_release_campaign = {
        "campaign_name": "reasoning_release",
        "run_id": release_output["run_id"],
        "final_path": release_output["path"],
        "kind": release_output["kind"],
        "sha256": release_output["sha256"],
        "artifact_inventory_sha256": release["campaign_lineage"].get(
            "artifact_inventory_sha256"
        ),
        "artifact_count": release_output["artifact_count"],
        "total_bytes": release_output["total_bytes"],
        "selected_artifacts": release_output["selected_artifacts"],
        "claim_path": claim["path"],
        "claim_sha256": claim["sha256"],
        "authorization_path": claim["authorization_path"],
        "authorization_sha256": claim["authorization_sha256"],
    }
    _hash(
        release_campaign["artifact_inventory_sha256"],
        "bundle validator reasoning_release_campaign.artifact_inventory_sha256",
    )
    if release_campaign != expected_release_campaign:
        fail(
            "bundle validator reasoning release campaign differs from "
            "the successful campaign outcome"
        )
    if campaign_identity.get("campaign_authorization_claim") != claim:
        fail("SQ8 full campaign claim differs from the successful campaign outcome")


def _validate_bundle(
    bundle: StableFileSnapshot,
    *,
    bundle_validator: BundleValidator,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    document = _strict_object(bundle.raw, "release bundle")
    if _canonical_json(document) != bundle.raw:
        fail("release bundle is not canonical JSON")
    if (
        document.get("schema_version") != BUNDLE_SCHEMA
        or document.get("status") != "complete"
        or document.get("production_activation_performed") is not False
    ):
        fail("release bundle is not a complete unactivated v2 bundle")
    report = bundle_validator(bundle.path)
    if (
        report.get("schema_version") != BUNDLE_VALIDATOR_SCHEMA
        or report.get("input_schema_version") != BUNDLE_SCHEMA
        or report.get("structurally_valid") is not True
        or report.get("gate_eligible") is not True
    ):
        fail("release bundle is not production-gate eligible")
    return document, report, _sha256(_canonical_json(report))


def _aq4_bundle_snapshot(
    outputs: dict[str, dict[str, Any]],
    *,
    required_uid: int,
) -> StableFileSnapshot:
    output = outputs["aq4_bundle"]
    if output["kind"] != "file":
        fail("fresh AQ4 bundle campaign output is not a file")
    path = Path(output["path"])
    snapshot = _stable(
        path,
        "complete AQ4 release bundle",
        read_only=True,
        single_link=True,
    )
    if (
        snapshot.identity.uid != required_uid
        or stat.S_IMODE(snapshot.identity.mode) != 0o444
        or output["selected_artifacts"].get(path.name) != snapshot.sha256
    ):
        fail("fresh AQ4 bundle bytes differ from the campaign outcome")
    return snapshot


def _validate_aq4_bundle(
    bundle: StableFileSnapshot,
    *,
    bundle_validator: BundleValidator,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    document = _strict_object(bundle.raw, "AQ4 release bundle")
    if (
        document.get("schema_version") != AQ4_BUNDLE_SCHEMA
        or document.get("status") != "complete"
        or document.get("production_activation_performed") is not False
    ):
        fail("AQ4 release bundle is not a complete unactivated v1 bundle")
    report = bundle_validator(bundle.path)
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != AQ4_BUNDLE_VALIDATOR_SCHEMA
        or report.get("input_schema_version") != AQ4_BUNDLE_SCHEMA
        or report.get("structurally_valid") is not True
        or report.get("gate_eligible") is not True
    ):
        fail("AQ4 release bundle is not production-gate eligible")
    return document, report, _sha256(_canonical_json(report))


def _bind_aq4_bundle_to_campaign_outputs(
    bundle_path: Path,
    bundle_document: dict[str, Any],
    bundle_report: dict[str, Any],
    outputs: dict[str, dict[str, Any]],
    authorization_document: dict[str, Any],
    *,
    rollback: StableFileSnapshot,
    rollback_worker: str,
) -> None:
    before = authorization_document["before"]
    aq4_release = authorization_document["aq4_release"]
    promotion_source = before["promotion_source_commit"]
    identity = bundle_document.get("identity")
    if (
        bundle_document.get("source_commit") != promotion_source
        or bundle_document.get("active_promotion_source_commit") != promotion_source
        or aq4_release["source"]["commit"] != promotion_source
        or bundle_report.get("source_commit") != promotion_source
        or not isinstance(identity, dict)
        or identity.get("manifest_sha256") != rollback.sha256
        or identity.get("worker_binary_sha256") != rollback_worker
        or identity.get("worker_binary_sha256")
        != before["worker_binary_sha256"]
        or identity.get("openwebui_image") != aq4_release["openwebui_image"]
    ):
        fail("AQ4 release bundle identity differs from the authorized rollback")

    artifacts = _exact(
        bundle_document.get("artifacts"),
        {
            "release_evidence",
            "release_validator",
            "browser_evidence",
            "browser_validator",
            "promotion_evidence",
            "promotion_receipt",
        },
        "AQ4 release bundle artifacts",
    )
    authorized_components: dict[str, tuple[Path, str | None]] = {
        "release_evidence": (
            Path(aq4_release["release_evidence_path"]),
            None,
        ),
        "release_validator": (
            Path(aq4_release["release_validator_path"]),
            None,
        ),
        "browser_validator": (
            Path(aq4_release["browser_validator_path"]),
            None,
        ),
        "promotion_evidence": (
            Path(aq4_release["promotion_evidence"]["path"]),
            aq4_release["promotion_evidence"]["sha256"],
        ),
        "promotion_receipt": (
            Path(aq4_release["promotion_receipt"]["path"]),
            aq4_release["promotion_receipt"]["sha256"],
        ),
    }
    for name, (expected_path, expected_sha256) in authorized_components.items():
        observed_path, observed_sha256 = _resolve_bundle_component(
            bundle_path,
            artifacts[name],
            name,
        )
        if observed_path != expected_path or (
            expected_sha256 is not None
            and observed_sha256 != expected_sha256
        ):
            fail(f"AQ4 bundle {name} differs from its authorization")
        component_snapshot = _stable(
            observed_path,
            f"AQ4 bundle {name}",
            read_only=True,
            single_link=True,
        )
        if (
            component_snapshot.identity.uid != rollback.identity.uid
            or stat.S_IMODE(component_snapshot.identity.mode) != 0o444
            or component_snapshot.sha256 != observed_sha256
        ):
            fail(f"AQ4 bundle {name} is not immutable")
    if outputs["aq4_reasoning_browser"]["kind"] != "file":
        fail("fresh AQ4 reasoning browser output is not the evidence file")
    browser_path = Path(outputs["aq4_reasoning_browser"]["path"])
    browser_snapshot = _stable(
        browser_path,
        "fresh AQ4 reasoning browser evidence",
        read_only=True,
        single_link=True,
    )
    if (
        browser_snapshot.identity.uid != rollback.identity.uid
        or stat.S_IMODE(browser_snapshot.identity.mode) != 0o444
        or outputs["aq4_reasoning_browser"]["selected_artifacts"].get(
            browser_path.name
        )
        != browser_snapshot.sha256
    ):
        fail("fresh AQ4 reasoning browser evidence bytes differ")
    _require_selected_campaign_artifact(
        bundle_path=bundle_path,
        artifacts=artifacts,
        artifact_name="browser_evidence",
        output=outputs["aq4_reasoning_browser"],
    )

    release_output = outputs["aq4_reasoning_release"]
    if release_output["kind"] != "directory":
        fail("fresh AQ4 reasoning release output is not a directory")
    release_root = Path(release_output["path"])
    try:
        release_root_metadata = release_root.lstat()
    except OSError as error:
        raise FinalActivationError(
            "fresh AQ4 reasoning release root is unavailable"
        ) from error
    if (
        not stat.S_ISDIR(release_root_metadata.st_mode)
        or release_root_metadata.st_uid != rollback.identity.uid
        or stat.S_IMODE(release_root_metadata.st_mode) != 0o555
    ):
        fail("fresh AQ4 reasoning release root is not immutable")
    cases_snapshot = _stable(
        release_root / "cases.json",
        "fresh AQ4 reasoning cases",
        maximum=MAX_OUTPUT_FILE_BYTES,
        read_only=True,
        single_link=True,
    )
    lifecycle_snapshot = _stable(
        release_root / "lifecycle.json",
        "fresh AQ4 reasoning lifecycle",
        maximum=MAX_OUTPUT_FILE_BYTES,
        read_only=True,
        single_link=True,
    )
    if any(
        snapshot.identity.uid != rollback.identity.uid
        or stat.S_IMODE(snapshot.identity.mode) != 0o444
        for snapshot in (cases_snapshot, lifecycle_snapshot)
    ):
        fail("fresh AQ4 reasoning release files are not immutable")
    cases = _strict_json(cases_snapshot.raw, "fresh AQ4 reasoning cases")
    lifecycle = _strict_object(
        lifecycle_snapshot.raw,
        "fresh AQ4 reasoning lifecycle",
    )
    if not isinstance(cases, list) or not cases:
        fail("fresh AQ4 reasoning cases are not a nonempty array")

    release = _bundle_json_artifact(
        bundle_path,
        artifacts,
        "release_evidence",
    )
    release_identity = release.get("identity")
    if (
        release.get("schema_version")
        != "ullm.generic_reasoning_release_evidence.v1"
        or release.get("source_commit") != promotion_source
        or release.get("active_promotion_source_commit") != promotion_source
        or not isinstance(release_identity, dict)
        or release_identity != identity
        or release_identity.get("manifest_sha256") != rollback.sha256
        or release_identity.get("worker_binary_sha256") != rollback_worker
        or _canonical_json(release.get("cases")) != _canonical_json(cases)
        or _canonical_json(release.get("lifecycle")) != _canonical_json(lifecycle)
    ):
        fail("AQ4 release evidence differs from fresh authorized campaign output")


def _require_authorized_rollback_identity(
    rollback: StableFileSnapshot,
    rollback_worker: str,
    authorization_document: dict[str, Any],
) -> None:
    before = authorization_document["before"]
    rollback_document = _strict_object(
        rollback.raw,
        "AQ4 rollback manifest",
    )
    promotion = rollback_document.get("promotion")
    if (
        before["manifest_sha256"] != rollback.sha256
        or before["worker_protocol"] != WORKER_PROTOCOL
        or before["worker_binary_sha256"] != rollback_worker
        or not isinstance(promotion, dict)
        or promotion.get("source_commit") != before["promotion_source_commit"]
        or promotion.get("receipt") != before["promotion_receipt_path"]
        or promotion.get("receipt_sha256")
        != before["promotion_receipt_sha256"]
    ):
        fail("AQ4 rollback identity differs from campaign authorization")


def _load_successful_campaign_outcome(
    authorization_path: Path,
    *,
    now: datetime,
    policy: authorization.RegistryPolicy,
) -> tuple[
    authorization.AuthorizationRecord,
    authorization.FileSnapshot,
    dict[str, Any],
]:
    try:
        outcome_snapshot, outcome = authorization.load_outcome(
            authorization_path,
            now=now,
            policy=policy,
        )
        record = authorization.load_authorization(
            authorization_path,
            now=now,
            policy=policy,
            require_fresh_outputs=False,
            enforce_current_window=False,
        )
    except authorization.AuthorizationError as error:
        raise FinalActivationError(
            "campaign authorization/outcome validation failed"
        ) from error
    if (
        authorization.AUTHORIZATION_SCHEMA
        != "ullm.served_model.v2_cross_model_campaign_authorization.v2"
        or authorization.OUTCOME_SCHEMA
        != "ullm.served_model.v2_cross_model_campaign_outcome.v2"
        or record.document.get("schema_version")
        != authorization.AUTHORIZATION_SCHEMA
        or outcome.get("schema_version") != authorization.OUTCOME_SCHEMA
        or outcome["status"] != "succeeded_restored"
        or outcome["failure_stage"] is not None
        or any(value != "passed" for value in outcome["stages"].values())
        or outcome["restoration"]["bytes_equal"] is not True
        or outcome["restoration"]["reverse_reconciliation_passed"] is not True
        or outcome["restoration"]["final_checks_passed"] is not True
    ):
        fail("campaign outcome does not prove a successful restored-AQ4 transaction")
    return record, outcome_snapshot, outcome


def _require_restoration_path(outcome: dict[str, Any], active: Path) -> None:
    proof = outcome["restoration"].get("proof")
    proof_active = proof.get("active_manifest") if isinstance(proof, dict) else None
    if (
        not isinstance(proof_active, dict)
        or proof_active.get("path") != os.fspath(active)
        or proof_active.get("expected_sha256")
        != outcome["restoration"]["expected_manifest_sha256"]
        or proof_active.get("observed_sha256")
        != outcome["restoration"]["observed_manifest_sha256"]
        or proof_active.get("bytes_equal") is not True
    ):
        fail("campaign outcome restoration proof targets a different active path")


def _ensure_fresh_destination(path: Path, label: str) -> None:
    _absolute(os.fspath(path), label, must_exist=False)
    if path.exists() or path.is_symlink():
        fail(f"{label} already exists or is a symlink")
    try:
        parent = path.parent.resolve(strict=True)
        metadata = parent.stat()
    except OSError as error:
        raise FinalActivationError(f"{label} parent is unavailable") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        fail(f"{label} parent directory is unsafe")


def _directory_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        fail("O_DIRECTORY and O_NOFOLLOW are required")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _walk_parent(path: Path) -> int:
    descriptor = -1
    try:
        descriptor = os.open(path.anchor, _directory_flags())
        for component in path.parent.parts[1:]:
            next_descriptor = os.open(component, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _directory_identity(descriptor: int) -> tuple[int, ...]:
    metadata = os.fstat(descriptor)
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_parent(path: Path, label: str) -> int:
    _absolute(os.fspath(path), label, must_exist=False)
    descriptor = -1
    verification = -1
    try:
        descriptor = _walk_parent(path)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            fail(f"{label} parent directory is unsafe")
        verification = _walk_parent(path)
        if _directory_identity(descriptor) != _directory_identity(verification):
            fail(f"{label} parent directory changed while it was opened")
        return descriptor
    except FinalActivationError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise FinalActivationError(
            f"{label} parent is unavailable or traverses a symlink"
        ) from error
    finally:
        if verification >= 0:
            os.close(verification)


def _publish_immutable(
    path: Path,
    document: dict[str, Any],
    *,
    required_uid: int,
) -> StableFileSnapshot:
    if os.geteuid() != required_uid:
        fail("immutable publisher has the wrong effective UID")
    raw = _canonical_json(document)
    if not raw or len(raw) > MAX_DOCUMENT_BYTES:
        fail("immutable output exceeds its byte bound")
    parent_descriptor = _open_parent(path, "immutable output")
    temporary_name = f".{path.name}.{os.getpid()}.{os.urandom(8).hex()}"
    descriptor = -1
    linked = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        os.fchmod(descriptor, 0o444)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                fail("immutable output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise FinalActivationError("immutable output already exists") from error
        linked = True
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not linked:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except OSError:
                pass
        os.close(parent_descriptor)
    snapshot = _stable(
        path,
        "immutable output",
        maximum=MAX_DOCUMENT_BYTES,
        read_only=True,
        single_link=True,
    )
    if snapshot.raw != raw or snapshot.identity.uid != required_uid:
        fail("immutable output differs after publication")
    return snapshot


def prepare_plan(
    *,
    plan_id: str,
    authorization_path: Path,
    candidate_manifest: Path,
    active_manifest: Path,
    rollback_manifest: Path,
    release_bundle: Path,
    systemd_unit: Path,
    environment_file: Path,
    operations_document: Path,
    activation_outcome: Path,
    rollback_outcome: Path,
    output: Path,
    now: datetime,
    policy: authorization.RegistryPolicy = authorization.RegistryPolicy(),
    manifest_validator: ManifestValidator = default_manifest_validator,
    bundle_validator: BundleValidator = default_bundle_validator,
) -> dict[str, Any]:
    """Validate all final-route evidence and publish one immutable plan."""

    _identifier(plan_id, "plan_id")
    for destination, label in (
        (output, "plan output"),
        (activation_outcome, "activation outcome"),
        (rollback_outcome, "rollback outcome"),
    ):
        _ensure_fresh_destination(destination, label)
    if len({output, activation_outcome, rollback_outcome}) != 3:
        fail("plan and outcome destinations must be distinct")

    record, outcome_snapshot, outcome = _load_successful_campaign_outcome(
        authorization_path,
        now=now,
        policy=policy,
    )
    auth = record.document
    execution_source = _capture_execution_source(
        expected_commit=auth["source"]["commit"],
        expected_tree=auth["source"]["tree"],
        required_uid=policy.required_uid,
    )

    candidate = _stable(
        candidate_manifest,
        "SQ8 candidate manifest",
        maximum=MAX_MANIFEST_BYTES,
    )
    rollback = _stable(
        rollback_manifest,
        "AQ4 rollback manifest",
        maximum=MAX_MANIFEST_BYTES,
        read_only=True,
        single_link=True,
    )
    active = _stable(
        active_manifest,
        "actual active manifest",
        maximum=MAX_MANIFEST_BYTES,
    )
    if (
        active.raw != rollback.raw
        or active.identity.uid != policy.required_uid
        or active.identity.links != 1
        or stat.S_IMODE(active.identity.mode) != 0o644
    ):
        fail("actual active bytes are not the exact AQ4 rollback bytes")

    candidate_summary = manifest_validator(candidate.path)
    rollback_summary = manifest_validator(rollback.path)
    candidate_worker = _summary_identity(
        candidate_summary,
        snapshot=candidate,
        model_id=SQ8_MODEL_ID,
        format_id=SQ8_FORMAT_ID,
        label="SQ8 candidate",
    )
    rollback_worker = _summary_identity(
        rollback_summary,
        snapshot=rollback,
        model_id=AQ4_MODEL_ID,
        format_id=AQ4_FORMAT_ID,
        label="AQ4 rollback",
    )
    candidate_document = _strict_object(candidate.raw, "SQ8 candidate manifest")
    rollback_document = _strict_object(rollback.raw, "AQ4 rollback manifest")
    if (
        candidate_document.get("schema_version") != SERVED_MODEL_SCHEMA
        or rollback_document.get("schema_version") != SERVED_MODEL_SCHEMA
    ):
        fail("final route requires v2 candidate and rollback manifests")
    candidate_runtime = _capture_manifest_runtime_seals(
        candidate,
        candidate_document,
        label="SQ8 candidate",
        required_uid=policy.required_uid,
    )
    rollback_runtime = _capture_manifest_runtime_seals(
        rollback,
        rollback_document,
        label="AQ4 rollback",
        required_uid=policy.required_uid,
    )
    if (
        candidate_runtime.worker.snapshot.sha256 != candidate_worker
        or rollback_runtime.worker.snapshot.sha256 != rollback_worker
    ):
        fail("served-model worker bytes differ from validated identity")

    policy_unit, policy_environment = _policy_deployment_paths(policy)
    if (
        active.path != policy.active_manifest_path
        or Path(systemd_unit) != policy_unit
        or Path(environment_file) != policy_environment
    ):
        fail("final route paths differ from the fixed production policy")

    _require_restoration_path(outcome, active.path)
    if (
        auth["candidate"]["manifest_sha256"] != candidate.sha256
        or auth["candidate"]["worker_binary_sha256"] != candidate_worker
        or outcome["restoration"]["observed_manifest_sha256"] != rollback.sha256
    ):
        fail("candidate/rollback identity differs from campaign authorization")
    _require_authorized_rollback_identity(rollback, rollback_worker, auth)

    unit = _stable(systemd_unit, "systemd unit")
    environment = _stable(environment_file, "systemd environment")
    if (
        unit.sha256 != auth["rollback"]["systemd_unit_sha256"]
        or environment.sha256 != auth["rollback"]["environment_sha256"]
    ):
        fail("unit/environment identity differs from campaign authorization")

    promotion = candidate_document.get("promotion")
    if (
        not isinstance(promotion, dict)
        or promotion.get("source_commit") != auth["source"]["commit"]
        or promotion.get("receipt_sha256")
        != auth["candidate"]["promotion_receipt_sha256"]
    ):
        fail("candidate promotion identity differs from authorization")

    outputs = _campaign_outputs_unchanged(outcome)
    aq4_bundle = _aq4_bundle_snapshot(
        outputs,
        required_uid=policy.required_uid,
    )
    (
        aq4_bundle_document,
        aq4_bundle_report,
        aq4_bundle_report_sha256,
    ) = _validate_aq4_bundle(
        aq4_bundle,
        bundle_validator=bundle_validator,
    )
    _bind_aq4_bundle_to_campaign_outputs(
        aq4_bundle.path,
        aq4_bundle_document,
        aq4_bundle_report,
        outputs,
        auth,
        rollback=rollback,
        rollback_worker=rollback_worker,
    )

    bundle = _stable(release_bundle, "complete release bundle")
    bundle_document, bundle_report, bundle_report_sha256 = _validate_bundle(
        bundle,
        bundle_validator=bundle_validator,
    )
    identity = bundle_document.get("identity")
    rollback_target = bundle_document.get("rollback_target")
    if (
        bundle_document.get("source_commit") != auth["source"]["commit"]
        or not isinstance(identity, dict)
        or identity.get("manifest_sha256") != candidate.sha256
        or identity.get("worker_binary_sha256") != candidate_worker
        or not isinstance(rollback_target, dict)
        or rollback_target.get("manifest_sha256") != rollback.sha256
        or rollback_target.get("systemd_unit_sha256") != unit.sha256
        or rollback_target.get("environment_sha256") != environment.sha256
    ):
        fail("release bundle final-route identity differs")
    _bind_bundle_to_campaign_outputs(
        bundle.path,
        bundle_document,
        bundle_report,
        outputs,
        outcome,
    )

    operations = _stable(
        operations_document,
        "reviewed operations document",
        maximum=MAX_DOCUMENT_BYTES,
        read_only=True,
        single_link=True,
    )
    operation_document = _operation_document(
        operations,
        required_uid=policy.required_uid,
        verify_executables=True,
    )
    deployment_runtime_artifacts = tuple(
        _capture_runtime_artifact(
            snapshot.path,
            label=label,
            maximum=maximum,
            required_uid=policy.required_uid,
        )
        for snapshot, label, maximum in (
            (unit, "systemd unit", MAX_INPUT_BYTES),
            (environment, "systemd environment", MAX_INPUT_BYTES),
            (operations, "reviewed operations document", MAX_DOCUMENT_BYTES),
        )
    )
    for sealed, expected in zip(
        deployment_runtime_artifacts,
        (unit, environment, operations),
        strict=True,
    ):
        if (
            sealed.snapshot.raw != expected.raw
            or sealed.snapshot.identity != expected.identity
        ):
            fail("deployment runtime seal differs from the reviewed input")
    shared_runtime_artifacts = deployment_runtime_artifacts
    candidate_operation_artifacts = _capture_operation_executable_seals(
        operation_document,
        required_uid=policy.required_uid,
        stages={"candidate_reconciliation", "candidate_live_health"},
    )
    rollback_operation_artifacts = _capture_operation_executable_seals(
        operation_document,
        required_uid=policy.required_uid,
        stages={"reverse_reconciliation", "rollback_live_health"},
    )
    if any(
        specification["service_unit"] != policy.service_unit
        for specification in operation_document["live_proofs"].values()
    ):
        fail("reviewed live proofs target a different service unit")
    proof_paths = {
        stage: Path(value["path"])
        for stage, value in operation_document["live_proofs"].items()
    }
    for stage, proof_path in proof_paths.items():
        _ensure_fresh_destination(proof_path, f"{stage} live proof")
    if (
        len(
            {
                output,
                activation_outcome,
                rollback_outcome,
                *proof_paths.values(),
            }
        )
        != 5
    ):
        fail("plan, outcome, and live-proof destinations must be distinct")

    document = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": plan_id,
        "prepared_at": utc_timestamp(now),
        "route": ROUTE,
        "production_activation_performed": False,
        "source": {
            "commit": auth["source"]["commit"],
            "tree": auth["source"]["tree"],
        },
        "active_manifest": {
            "path": os.fspath(active_manifest),
            "expected_current_sha256": rollback.sha256,
        },
        "candidate": {
            "path": os.fspath(candidate.path),
            "manifest_sha256": candidate.sha256,
            "model_id": SQ8_MODEL_ID,
            "format_id": SQ8_FORMAT_ID,
            "worker_protocol": WORKER_PROTOCOL,
            "worker_binary_sha256": candidate_worker,
            "promotion_receipt_sha256": auth["candidate"]["promotion_receipt_sha256"],
        },
        "rollback": {
            "path": os.fspath(rollback.path),
            "manifest_sha256": rollback.sha256,
            "model_id": AQ4_MODEL_ID,
            "format_id": AQ4_FORMAT_ID,
            "worker_protocol": WORKER_PROTOCOL,
            "worker_binary_sha256": rollback_worker,
        },
        "aq4_release_bundle": {
            "path": os.fspath(aq4_bundle.path),
            "sha256": aq4_bundle.sha256,
            "schema_version": AQ4_BUNDLE_SCHEMA,
            "validator_schema_version": AQ4_BUNDLE_VALIDATOR_SCHEMA,
            "validator_report_sha256": aq4_bundle_report_sha256,
        },
        "release_bundle": {
            "path": os.fspath(bundle.path),
            "sha256": bundle.sha256,
            "schema_version": BUNDLE_SCHEMA,
            "validator_schema_version": BUNDLE_VALIDATOR_SCHEMA,
            "validator_report_sha256": bundle_report_sha256,
        },
        "campaign": {
            "authorization_path": os.fspath(record.snapshot.path),
            "authorization_sha256": record.snapshot.sha256,
            "authorization_id": auth["authorization_id"],
            "outcome_path": os.fspath(outcome_snapshot.path),
            "outcome_sha256": outcome_snapshot.sha256,
            "outcome_status": "succeeded_restored",
            "completed_at": outcome["completed_at"],
            "outputs": outputs,
        },
        "deployment": {
            "systemd_unit_path": os.fspath(unit.path),
            "systemd_unit_sha256": unit.sha256,
            "environment_path": os.fspath(environment.path),
            "environment_sha256": environment.sha256,
        },
        "operations": {
            "path": os.fspath(operations.path),
            "sha256": operations.sha256,
            "schema_version": OPERATIONS_SCHEMA,
            "review_id": operation_document["review_id"],
            "reviewed_at": operation_document["reviewed_at"],
            "reviewed_by": operation_document["reviewed_by"],
        },
        "live_proofs": operation_document["live_proofs"],
        "outcomes": {
            "activation_path": os.fspath(activation_outcome),
            "rollback_path": os.fspath(rollback_outcome),
        },
    }
    _require_artifact_seals(
        shared_runtime_artifacts,
        required_uid=policy.required_uid,
    )
    _require_artifact_seals(
        candidate_operation_artifacts,
        required_uid=policy.required_uid,
    )
    _require_artifact_seals(
        rollback_operation_artifacts,
        required_uid=policy.required_uid,
    )
    _require_manifest_runtime_seals(
        candidate_runtime,
        required_uid=policy.required_uid,
    )
    _require_manifest_runtime_seals(
        rollback_runtime,
        required_uid=policy.required_uid,
    )
    _require_execution_source(
        execution_source,
        expected_commit=auth["source"]["commit"],
        expected_tree=auth["source"]["tree"],
        required_uid=policy.required_uid,
    )
    _publish_immutable(output, document, required_uid=policy.required_uid)
    _require_execution_source(
        execution_source,
        expected_commit=auth["source"]["commit"],
        expected_tree=auth["source"]["tree"],
        required_uid=policy.required_uid,
    )
    _require_artifact_seals(
        shared_runtime_artifacts,
        required_uid=policy.required_uid,
    )
    _require_artifact_seals(
        candidate_operation_artifacts,
        required_uid=policy.required_uid,
    )
    _require_artifact_seals(
        rollback_operation_artifacts,
        required_uid=policy.required_uid,
    )
    _require_manifest_runtime_seals(
        candidate_runtime,
        required_uid=policy.required_uid,
    )
    _require_manifest_runtime_seals(
        rollback_runtime,
        required_uid=policy.required_uid,
    )
    _require_execution_source(
        execution_source,
        expected_commit=auth["source"]["commit"],
        expected_tree=auth["source"]["tree"],
        required_uid=policy.required_uid,
    )
    return document


def _validate_plan_shape(document: dict[str, Any], *, action: str) -> None:
    _exact(
        document,
        {
            "schema_version",
            "plan_id",
            "prepared_at",
            "route",
            "production_activation_performed",
            "source",
            "active_manifest",
            "candidate",
            "rollback",
            "aq4_release_bundle",
            "release_bundle",
            "campaign",
            "deployment",
            "operations",
            "live_proofs",
            "outcomes",
        },
        "activation plan",
    )
    if (
        document["schema_version"] != PLAN_SCHEMA
        or document["route"] != ROUTE
        or document["production_activation_performed"] is not False
    ):
        fail("activation plan route/schema differs")
    _identifier(document["plan_id"], "plan_id")
    _timestamp(document["prepared_at"], "prepared_at")
    source = _exact(document["source"], {"commit", "tree"}, "source")
    _git(source["commit"], "source.commit")
    _git(source["tree"], "source.tree")
    active = _exact(
        document["active_manifest"],
        {"path", "expected_current_sha256"},
        "active_manifest",
    )
    _absolute(active["path"], "active_manifest", must_exist=True)
    _hash(active["expected_current_sha256"], "active_manifest.expected_current_sha256")
    candidate = _exact(
        document["candidate"],
        {
            "path",
            "manifest_sha256",
            "model_id",
            "format_id",
            "worker_protocol",
            "worker_binary_sha256",
            "promotion_receipt_sha256",
        },
        "candidate",
    )
    rollback = _exact(
        document["rollback"],
        {
            "path",
            "manifest_sha256",
            "model_id",
            "format_id",
            "worker_protocol",
            "worker_binary_sha256",
        },
        "rollback",
    )
    for label, value, model_id, format_id in (
        ("candidate", candidate, SQ8_MODEL_ID, SQ8_FORMAT_ID),
        ("rollback", rollback, AQ4_MODEL_ID, AQ4_FORMAT_ID),
    ):
        _absolute(
            value["path"],
            label,
            must_exist=action == "activate" or label == "rollback",
        )
        _hash(value["manifest_sha256"], f"{label}.manifest_sha256")
        _hash(value["worker_binary_sha256"], f"{label}.worker_binary_sha256")
        if (
            value["model_id"] != model_id
            or value["format_id"] != format_id
            or value["worker_protocol"] != WORKER_PROTOCOL
        ):
            fail(f"{label} model identity differs")
    _hash(
        candidate["promotion_receipt_sha256"],
        "candidate.promotion_receipt_sha256",
    )
    aq4_bundle = _exact(
        document["aq4_release_bundle"],
        {
            "path",
            "sha256",
            "schema_version",
            "validator_schema_version",
            "validator_report_sha256",
        },
        "aq4_release_bundle",
    )
    _absolute(
        aq4_bundle["path"],
        "aq4_release_bundle",
        must_exist=action == "activate",
    )
    for field in ("sha256", "validator_report_sha256"):
        _hash(aq4_bundle[field], f"aq4_release_bundle.{field}")
    if (
        aq4_bundle["schema_version"] != AQ4_BUNDLE_SCHEMA
        or aq4_bundle["validator_schema_version"] != AQ4_BUNDLE_VALIDATOR_SCHEMA
    ):
        fail("AQ4 release bundle schema binding differs")
    bundle = _exact(
        document["release_bundle"],
        {
            "path",
            "sha256",
            "schema_version",
            "validator_schema_version",
            "validator_report_sha256",
        },
        "release_bundle",
    )
    _absolute(
        bundle["path"],
        "release_bundle",
        must_exist=action == "activate",
    )
    for field in ("sha256", "validator_report_sha256"):
        _hash(bundle[field], f"release_bundle.{field}")
    if (
        bundle["schema_version"] != BUNDLE_SCHEMA
        or bundle["validator_schema_version"] != BUNDLE_VALIDATOR_SCHEMA
    ):
        fail("release bundle schema binding differs")
    campaign = _exact(
        document["campaign"],
        {
            "authorization_path",
            "authorization_sha256",
            "authorization_id",
            "outcome_path",
            "outcome_sha256",
            "outcome_status",
            "completed_at",
            "outputs",
        },
        "campaign",
    )
    _absolute(
        campaign["authorization_path"],
        "campaign authorization",
        must_exist=action == "activate",
    )
    _absolute(
        campaign["outcome_path"],
        "campaign outcome",
        must_exist=action == "activate",
    )
    _hash(campaign["authorization_sha256"], "campaign.authorization_sha256")
    _hash(campaign["outcome_sha256"], "campaign.outcome_sha256")
    _identifier(campaign["authorization_id"], "campaign.authorization_id")
    _timestamp(campaign["completed_at"], "campaign.completed_at")
    if campaign["outcome_status"] != "succeeded_restored":
        fail("campaign outcome status binding differs")
    if (
        not isinstance(campaign["outputs"], dict)
        or set(campaign["outputs"]) != FINAL_CAMPAIGN_FIELDS
    ):
        fail("campaign output bindings differ")
    deployment = _exact(
        document["deployment"],
        {
            "systemd_unit_path",
            "systemd_unit_sha256",
            "environment_path",
            "environment_sha256",
        },
        "deployment",
    )
    for name in ("systemd_unit", "environment"):
        _absolute(deployment[f"{name}_path"], f"deployment.{name}", must_exist=True)
        _hash(deployment[f"{name}_sha256"], f"deployment.{name}_sha256")
    operations = _exact(
        document["operations"],
        {
            "path",
            "sha256",
            "schema_version",
            "review_id",
            "reviewed_at",
            "reviewed_by",
        },
        "operations",
    )
    _absolute(operations["path"], "operations", must_exist=True)
    _hash(operations["sha256"], "operations.sha256")
    _identifier(operations["review_id"], "operations.review_id")
    _timestamp(operations["reviewed_at"], "operations.reviewed_at")
    _text(operations["reviewed_by"], "operations.reviewed_by", 512)
    if operations["schema_version"] != OPERATIONS_SCHEMA:
        fail("operations schema binding differs")
    live_proofs = _exact(
        document["live_proofs"],
        {"candidate_live_health", "rollback_live_health"},
        "live_proofs",
    )
    proof_paths: set[Path] = set()
    for stage, value in live_proofs.items():
        specification = _exact(
            value,
            LIVE_PROOF_SPEC_FIELDS,
            f"live_proofs.{stage}",
        )
        proof_path = _absolute(
            specification["path"],
            f"live_proofs.{stage}",
            must_exist=False,
        )
        if proof_path in proof_paths:
            fail("candidate and rollback live-proof paths collide")
        proof_paths.add(proof_path)
        _text(specification["service_unit"], f"live_proofs.{stage}.service_unit", 512)
        _hash(
            specification["gateway_executable_sha256"],
            f"live_proofs.{stage}.gateway_executable_sha256",
        )
        endpoint_urls = _exact(
            specification["endpoint_urls"],
            ENDPOINT_NAMES,
            f"live_proofs.{stage}.endpoint_urls",
        )
        for endpoint, url in endpoint_urls.items():
            _endpoint_url(url, f"live_proofs.{stage}.endpoint_urls.{endpoint}")
    outcomes = _exact(
        document["outcomes"],
        {"activation_path", "rollback_path"},
        "outcomes",
    )
    for field in outcomes:
        _absolute(outcomes[field], f"outcomes.{field}", must_exist=False)
    if outcomes["activation_path"] == outcomes["rollback_path"]:
        fail("activation and rollback outcome paths collide")
    if proof_paths & {
        Path(outcomes["activation_path"]),
        Path(outcomes["rollback_path"]),
    }:
        fail("live-proof and outcome paths collide")


def _load_activation_outcome(
    path: Path,
    *,
    plan: StableFileSnapshot,
    required_uid: int,
    record: PlanRecord | None = None,
) -> tuple[StableFileSnapshot, dict[str, Any]]:
    snapshot = _stable(
        path,
        "activation outcome",
        maximum=MAX_DOCUMENT_BYTES,
        read_only=True,
        single_link=True,
    )
    if snapshot.identity.uid != required_uid:
        fail("activation outcome owner differs")
    document = _strict_object(snapshot.raw, "activation outcome")
    plan_document = _strict_object(plan.raw, "final activation plan")
    if _canonical_json(document) != snapshot.raw:
        fail("activation outcome is not canonical JSON")
    _exact(
        document,
        {
            "schema_version",
            "plan_id",
            "plan_path",
            "plan_sha256",
            "started_at",
            "completed_at",
            "status",
            "failure_stage",
            "stages",
            "before_manifest_sha256",
            "candidate_manifest_sha256",
            "observed_manifest_sha256",
            "live_proofs",
            "restoration",
        },
        "activation outcome",
    )
    if (
        document["schema_version"] != ACTIVATION_OUTCOME_SCHEMA
        or document["plan_id"] != plan_document["plan_id"]
        or document["plan_path"] != os.fspath(plan.path)
        or document["plan_sha256"] != plan.sha256
        or document["status"] != "activated"
        or document["failure_stage"] is not None
    ):
        fail("activation outcome does not prove successful activation")
    started_at = _timestamp(document["started_at"], "activation outcome.started_at")
    completed_at = _timestamp(
        document["completed_at"],
        "activation outcome.completed_at",
    )
    if completed_at < started_at:
        fail("activation outcome timestamps are reversed")
    if (
        document["before_manifest_sha256"]
        != plan_document["rollback"]["manifest_sha256"]
        or document["candidate_manifest_sha256"]
        != plan_document["candidate"]["manifest_sha256"]
        or document["observed_manifest_sha256"]
        != plan_document["candidate"]["manifest_sha256"]
    ):
        fail("activation outcome manifest identity differs")
    stages = _exact(document["stages"], ACTIVATION_STAGES, "activation outcome stages")
    if (
        stages["lock"] != "passed"
        or stages["preflight"] != "passed"
        or stages["candidate_activation"] != "passed"
        or stages["candidate_reconciliation"] != "passed"
        or stages["candidate_live_health"] != "passed"
        or stages["outcome_publication"] != "passed"
        or any(
            stages[name] != "skipped"
            for name in {
                "aq4_restore",
                "reverse_reconciliation",
                "rollback_live_health",
            }
        )
    ):
        fail("activation outcome stages do not prove successful activation")
    restoration = _exact(
        document["restoration"],
        {
            "attempted",
            "manifest_sha256",
            "bytes_equal",
            "reverse_reconciliation_passed",
            "live_health_passed",
        },
        "activation outcome restoration",
    )
    if restoration != {
        "attempted": False,
        "manifest_sha256": plan_document["rollback"]["manifest_sha256"],
        "bytes_equal": False,
        "reverse_reconciliation_passed": False,
        "live_health_passed": False,
    }:
        fail("activation outcome unexpectedly reports rollback")
    live_proofs = _exact(
        document["live_proofs"],
        {"candidate_live_health", "rollback_live_health"},
        "activation outcome live_proofs",
    )
    if (
        not isinstance(live_proofs["candidate_live_health"], dict)
        or live_proofs["rollback_live_health"] is not None
    ):
        fail("activation outcome live-proof presence differs")
    if record is not None:
        _validate_live_proof_envelope(
            record,
            "candidate_live_health",
            live_proofs["candidate_live_health"],
            read_named=False,
        )
    return snapshot, document


def load_plan(
    plan_path: Path,
    *,
    action: str,
    now: datetime,
    policy: authorization.RegistryPolicy = authorization.RegistryPolicy(),
    manifest_validator: ManifestValidator = default_manifest_validator,
    bundle_validator: BundleValidator = default_bundle_validator,
) -> PlanRecord:
    """Revalidate every plan input for activation or manual rollback."""

    if action not in {"activate", "rollback"}:
        fail("plan action is invalid")
    plan = _stable(
        plan_path,
        "final activation plan",
        maximum=MAX_DOCUMENT_BYTES,
        read_only=True,
        single_link=True,
    )
    if plan.identity.uid != policy.required_uid:
        fail("final activation plan owner differs")
    document = _strict_object(plan.raw, "final activation plan")
    if _canonical_json(document) != plan.raw:
        fail("final activation plan is not canonical JSON")
    _validate_plan_shape(document, action=action)
    execution_source = _capture_execution_source(
        expected_commit=document["source"]["commit"],
        expected_tree=document["source"]["tree"],
        required_uid=policy.required_uid,
    )
    policy_unit, policy_environment = _policy_deployment_paths(policy)
    if (
        Path(document["active_manifest"]["path"]) != policy.active_manifest_path
        or Path(document["deployment"]["systemd_unit_path"]) != policy_unit
        or Path(document["deployment"]["environment_path"]) != policy_environment
    ):
        fail("final activation plan paths differ from production policy")

    rollback = _stable(
        Path(document["rollback"]["path"]),
        "AQ4 rollback manifest",
        maximum=MAX_MANIFEST_BYTES,
        read_only=True,
        single_link=True,
    )
    active = _stable(
        Path(document["active_manifest"]["path"]),
        "actual active manifest",
        maximum=MAX_MANIFEST_BYTES,
    )
    if action == "activate":
        candidate = _stable(
            Path(document["candidate"]["path"]),
            "SQ8 candidate manifest",
            maximum=MAX_MANIFEST_BYTES,
        )
    else:
        # A manual rollback is specifically the recovery path for a broken or
        # missing SQ8 release closure.  The successfully activated exact bytes
        # at active.json, plus the immutable activation outcome, are its
        # candidate precondition; the original candidate pathname is not.
        candidate = StableFileSnapshot(
            path=Path(document["candidate"]["path"]),
            raw=active.raw,
            sha256=active.sha256,
            identity=active.identity,
        )
    if (
        active.identity.uid != policy.required_uid
        or active.identity.links != 1
        or stat.S_IMODE(active.identity.mode) != 0o644
    ):
        fail("actual active manifest metadata differs")
    aq4_bundle: StableFileSnapshot | None = None
    bundle: StableFileSnapshot | None = None
    if action == "activate":
        aq4_bundle = _stable(
            Path(document["aq4_release_bundle"]["path"]),
            "complete AQ4 release bundle",
            read_only=True,
            single_link=True,
        )
        bundle = _stable(
            Path(document["release_bundle"]["path"]),
            "complete release bundle",
        )
    unit = _stable(Path(document["deployment"]["systemd_unit_path"]), "systemd unit")
    environment = _stable(
        Path(document["deployment"]["environment_path"]),
        "systemd environment",
    )
    operations_snapshot = _stable(
        Path(document["operations"]["path"]),
        "reviewed operations document",
        maximum=MAX_DOCUMENT_BYTES,
        read_only=True,
        single_link=True,
    )
    operations = _operation_document(
        operations_snapshot,
        required_uid=policy.required_uid,
        verify_executables=True,
        executable_stages=(
            None
            if action == "activate"
            else {"reverse_reconciliation", "rollback_live_health"}
        ),
    )
    if any(
        specification["service_unit"] != policy.service_unit
        for specification in operations["live_proofs"].values()
    ):
        fail("final activation live proof targets a different service unit")

    if (
        candidate.sha256 != document["candidate"]["manifest_sha256"]
        or rollback.sha256 != document["rollback"]["manifest_sha256"]
        or (
            aq4_bundle is not None
            and (
                aq4_bundle.identity.uid != policy.required_uid
                or stat.S_IMODE(aq4_bundle.identity.mode) != 0o444
                or aq4_bundle.sha256 != document["aq4_release_bundle"]["sha256"]
            )
        )
        or (
            bundle is not None
            and bundle.sha256 != document["release_bundle"]["sha256"]
        )
        or unit.sha256 != document["deployment"]["systemd_unit_sha256"]
        or environment.sha256 != document["deployment"]["environment_sha256"]
        or operations_snapshot.sha256 != document["operations"]["sha256"]
        or operations["review_id"] != document["operations"]["review_id"]
        or operations["reviewed_at"] != document["operations"]["reviewed_at"]
        or operations["reviewed_by"] != document["operations"]["reviewed_by"]
        or operations["live_proofs"] != document["live_proofs"]
    ):
        fail("final activation plan input hash differs")

    rollback_document = _strict_object(rollback.raw, "AQ4 rollback manifest")
    candidate_runtime: ManifestRuntimeSeals | None = None
    if action == "activate":
        candidate_document = _strict_object(
            candidate.raw,
            "SQ8 candidate manifest",
        )
        candidate_runtime = _capture_manifest_runtime_seals(
            candidate,
            candidate_document,
            label="SQ8 candidate",
            required_uid=policy.required_uid,
        )
    rollback_runtime = _capture_manifest_runtime_seals(
        rollback,
        rollback_document,
        label="AQ4 rollback",
        required_uid=policy.required_uid,
    )
    deployment_runtime_artifacts = tuple(
        _capture_runtime_artifact(
            snapshot.path,
            label=label,
            maximum=maximum,
            required_uid=policy.required_uid,
        )
        for snapshot, label, maximum in (
            (unit, "systemd unit", MAX_INPUT_BYTES),
            (environment, "systemd environment", MAX_INPUT_BYTES),
            (
                operations_snapshot,
                "reviewed operations document",
                MAX_DOCUMENT_BYTES,
            ),
        )
    )
    for sealed, expected in zip(
        deployment_runtime_artifacts,
        (unit, environment, operations_snapshot),
        strict=True,
    ):
        if (
            sealed.snapshot.raw != expected.raw
            or sealed.snapshot.identity != expected.identity
        ):
            fail("deployment runtime seal differs from the plan input")
    shared_runtime_artifacts = deployment_runtime_artifacts
    candidate_operation_artifacts = (
        _capture_operation_executable_seals(
            operations,
            required_uid=policy.required_uid,
            stages={"candidate_reconciliation", "candidate_live_health"},
        )
        if action == "activate"
        else ()
    )
    rollback_operation_artifacts = _capture_operation_executable_seals(
        operations,
        required_uid=policy.required_uid,
        stages={"reverse_reconciliation", "rollback_live_health"},
    )

    candidate_worker = document["candidate"]["worker_binary_sha256"]
    if action == "activate":
        candidate_worker = _summary_identity(
            manifest_validator(candidate.path),
            snapshot=candidate,
            model_id=SQ8_MODEL_ID,
            format_id=SQ8_FORMAT_ID,
            label="SQ8 candidate",
        )
    rollback_worker = _summary_identity(
        manifest_validator(rollback.path),
        snapshot=rollback,
        model_id=AQ4_MODEL_ID,
        format_id=AQ4_FORMAT_ID,
        label="AQ4 rollback",
    )
    if (
        candidate_worker != document["candidate"]["worker_binary_sha256"]
        or rollback_worker != document["rollback"]["worker_binary_sha256"]
        or (
            candidate_runtime is not None
            and candidate_runtime.worker.snapshot.sha256 != candidate_worker
        )
        or rollback_runtime.worker.snapshot.sha256 != rollback_worker
    ):
        fail("final activation worker identity differs")

    campaign_outcome_snapshot: authorization.FileSnapshot | None = None
    campaign_outcome_document: dict[str, Any] | None = None
    if action == "activate":
        (
            campaign_record,
            campaign_outcome_snapshot,
            campaign_outcome_document,
        ) = _load_successful_campaign_outcome(
            Path(document["campaign"]["authorization_path"]),
            now=now,
            policy=policy,
        )
        _require_restoration_path(campaign_outcome_document, active.path)
        if (
            campaign_record.snapshot.sha256
            != document["campaign"]["authorization_sha256"]
            or campaign_outcome_snapshot.path
            != Path(document["campaign"]["outcome_path"])
            or campaign_outcome_snapshot.sha256
            != document["campaign"]["outcome_sha256"]
            or campaign_record.document["authorization_id"]
            != document["campaign"]["authorization_id"]
            or campaign_outcome_document["completed_at"]
            != document["campaign"]["completed_at"]
            or campaign_record.document["source"] != document["source"]
        ):
            fail("campaign authorization/outcome plan binding differs")
        _require_authorized_rollback_identity(
            rollback,
            rollback_worker,
            campaign_record.document,
        )
        assert aq4_bundle is not None
        assert bundle is not None
        outputs = _campaign_outputs_unchanged(campaign_outcome_document)
        if outputs != document["campaign"]["outputs"]:
            fail("campaign output plan binding differs")
        if (
            aq4_bundle.path != Path(outputs["aq4_bundle"]["path"])
            or aq4_bundle.sha256
            != outputs["aq4_bundle"]["selected_artifacts"].get(aq4_bundle.path.name)
        ):
            fail("AQ4 release bundle plan path differs from fresh campaign output")
        (
            aq4_bundle_document,
            aq4_bundle_report,
            aq4_report_sha256,
        ) = _validate_aq4_bundle(
            aq4_bundle,
            bundle_validator=bundle_validator,
        )
        if (
            aq4_report_sha256
            != document["aq4_release_bundle"]["validator_report_sha256"]
        ):
            fail("AQ4 release bundle validator report changed")
        _bind_aq4_bundle_to_campaign_outputs(
            aq4_bundle.path,
            aq4_bundle_document,
            aq4_bundle_report,
            outputs,
            campaign_record.document,
            rollback=rollback,
            rollback_worker=rollback_worker,
        )

        bundle_document, bundle_report, report_sha256 = _validate_bundle(
            bundle,
            bundle_validator=bundle_validator,
        )
        if report_sha256 != document["release_bundle"]["validator_report_sha256"]:
            fail("release bundle validator report changed")
        _bind_bundle_to_campaign_outputs(
            bundle.path,
            bundle_document,
            bundle_report,
            outputs,
            campaign_outcome_document,
        )

    expected_active = rollback.raw if action == "activate" else candidate.raw
    if active.raw != expected_active:
        fail(f"actual active bytes differ from the exact {action} precondition")
    if document["active_manifest"]["expected_current_sha256"] != rollback.sha256:
        fail("plan expected-current AQ4 identity differs")

    activation_path = Path(document["outcomes"]["activation_path"])
    rollback_path = Path(document["outcomes"]["rollback_path"])
    result = PlanRecord(
        plan,
        document,
        operations,
        candidate,
        rollback,
        active,
        aq4_bundle,
        bundle,
        unit,
        environment,
        campaign_outcome_snapshot,
        campaign_outcome_document,
        candidate_runtime,
        rollback_runtime,
        shared_runtime_artifacts,
        candidate_operation_artifacts,
        rollback_operation_artifacts,
        execution_source,
    )
    if action == "activate":
        _ensure_fresh_destination(activation_path, "activation outcome")
        _ensure_fresh_destination(rollback_path, "rollback outcome")
        for stage, specification in document["live_proofs"].items():
            _ensure_fresh_destination(
                Path(specification["path"]),
                f"{stage} live proof",
            )
    else:
        _load_activation_outcome(
            activation_path,
            plan=plan,
            required_uid=policy.required_uid,
            record=result,
        )
        _ensure_fresh_destination(rollback_path, "rollback outcome")
        _ensure_fresh_destination(
            Path(document["live_proofs"]["rollback_live_health"]["path"]),
            "rollback_live_health live proof",
        )

    _require_record_runtime_seals(
        result,
        required_uid=policy.required_uid,
        scope="all" if action == "activate" else "rollback",
    )
    _require_execution_source(
        result.execution_source,
        expected_commit=document["source"]["commit"],
        expected_tree=document["source"]["tree"],
        required_uid=policy.required_uid,
    )
    return result


def _repin_plan_inputs(
    record: PlanRecord,
    *,
    now: datetime,
    policy: authorization.RegistryPolicy,
    manifest_validator: ManifestValidator,
    bundle_validator: BundleValidator,
) -> None:
    """Re-pin every immutable plan authority without outcome freshness checks."""

    _require_record_execution_source(
        record,
        required_uid=policy.required_uid,
    )
    _require_record_runtime_seals(
        record,
        required_uid=policy.required_uid,
        scope="all",
    )
    if (
        record.aq4_bundle is None
        or record.bundle is None
        or record.campaign_outcome is None
        or record.campaign_outcome_document is None
    ):
        fail("activation release/campaign authorities are unavailable")
    plan = _stable(
        record.snapshot.path,
        "final activation plan",
        maximum=MAX_DOCUMENT_BYTES,
        read_only=True,
        single_link=True,
    )
    candidate = _stable(
        record.candidate.path,
        "SQ8 candidate manifest",
        maximum=MAX_MANIFEST_BYTES,
    )
    rollback = _stable(
        record.rollback.path,
        "AQ4 rollback manifest",
        maximum=MAX_MANIFEST_BYTES,
        read_only=True,
        single_link=True,
    )
    aq4_bundle = _stable(
        record.aq4_bundle.path,
        "complete AQ4 release bundle",
        read_only=True,
        single_link=True,
    )
    bundle = _stable(record.bundle.path, "complete release bundle")
    unit = _stable(record.unit.path, "systemd unit")
    environment = _stable(record.environment.path, "systemd environment")
    operations_snapshot = _stable(
        Path(record.document["operations"]["path"]),
        "reviewed operations document",
        maximum=MAX_DOCUMENT_BYTES,
        read_only=True,
        single_link=True,
    )
    current_snapshots = (
        (plan, record.snapshot, "plan"),
        (candidate, record.candidate, "candidate"),
        (rollback, record.rollback, "rollback"),
        (aq4_bundle, record.aq4_bundle, "AQ4 bundle"),
        (bundle, record.bundle, "bundle"),
        (unit, record.unit, "unit"),
        (environment, record.environment, "environment"),
    )
    for current, expected, label in current_snapshots:
        if (
            current.path != expected.path
            or current.raw != expected.raw
            or current.identity != expected.identity
        ):
            fail(f"{label} changed during final activation")
    if (
        operations_snapshot.sha256 != record.document["operations"]["sha256"]
        or operations_snapshot.identity.uid != policy.required_uid
    ):
        fail("reviewed operations changed during final activation")
    operations = _operation_document(
        operations_snapshot,
        required_uid=policy.required_uid,
        verify_executables=True,
    )
    if operations != record.operations:
        fail("reviewed operations identity changed during final activation")

    candidate_worker = _summary_identity(
        manifest_validator(candidate.path),
        snapshot=candidate,
        model_id=SQ8_MODEL_ID,
        format_id=SQ8_FORMAT_ID,
        label="SQ8 candidate",
    )
    rollback_worker = _summary_identity(
        manifest_validator(rollback.path),
        snapshot=rollback,
        model_id=AQ4_MODEL_ID,
        format_id=AQ4_FORMAT_ID,
        label="AQ4 rollback",
    )
    if (
        candidate_worker != record.document["candidate"]["worker_binary_sha256"]
        or rollback_worker != record.document["rollback"]["worker_binary_sha256"]
    ):
        fail("worker identity changed during final activation")
    campaign_record, outcome_snapshot, outcome = _load_successful_campaign_outcome(
        Path(record.document["campaign"]["authorization_path"]),
        now=now,
        policy=policy,
    )
    if (
        campaign_record.snapshot.sha256
        != record.document["campaign"]["authorization_sha256"]
        or outcome_snapshot.path != record.campaign_outcome.path
        or outcome_snapshot.sha256 != record.campaign_outcome.sha256
        or outcome != record.campaign_outcome_document
    ):
        fail("campaign authority changed during final activation")
    _require_authorized_rollback_identity(
        rollback,
        rollback_worker,
        campaign_record.document,
    )
    _require_restoration_path(outcome, record.active.path)
    outputs = _campaign_outputs_unchanged(outcome)
    if outputs != record.document["campaign"]["outputs"]:
        fail("campaign outputs changed during final activation")
    if aq4_bundle.path != Path(outputs["aq4_bundle"]["path"]):
        fail("AQ4 release bundle path changed during final activation")
    (
        aq4_bundle_document,
        aq4_bundle_report,
        aq4_report_sha256,
    ) = _validate_aq4_bundle(
        aq4_bundle,
        bundle_validator=bundle_validator,
    )
    if (
        aq4_report_sha256
        != record.document["aq4_release_bundle"]["validator_report_sha256"]
    ):
        fail("AQ4 release bundle validation changed during final activation")
    _bind_aq4_bundle_to_campaign_outputs(
        aq4_bundle.path,
        aq4_bundle_document,
        aq4_bundle_report,
        outputs,
        campaign_record.document,
        rollback=rollback,
        rollback_worker=rollback_worker,
    )

    bundle_document, bundle_report, report_sha256 = _validate_bundle(
        bundle,
        bundle_validator=bundle_validator,
    )
    if report_sha256 != record.document["release_bundle"]["validator_report_sha256"]:
        fail("release bundle validation changed during final activation")
    _bind_bundle_to_campaign_outputs(
        bundle.path,
        bundle_document,
        bundle_report,
        outputs,
        outcome,
    )

    # Catch a mutation between a path-based validator read and this boundary.
    for expected, label, maximum, read_only in (
        (record.candidate, "SQ8 candidate manifest", MAX_MANIFEST_BYTES, False),
        (record.rollback, "AQ4 rollback manifest", MAX_MANIFEST_BYTES, True),
        (record.aq4_bundle, "complete AQ4 release bundle", MAX_INPUT_BYTES, True),
        (record.bundle, "complete release bundle", MAX_INPUT_BYTES, False),
    ):
        observed = _stable(
            expected.path,
            label,
            maximum=maximum,
            read_only=read_only,
            single_link=read_only,
        )
        if observed.raw != expected.raw or observed.identity != expected.identity:
            fail(f"{label} changed across validation")
    _require_record_runtime_seals(
        record,
        required_uid=policy.required_uid,
        scope="all",
    )
    _require_record_execution_source(
        record,
        required_uid=policy.required_uid,
    )


def _repin_rollback_inputs(
    record: PlanRecord,
    *,
    policy: authorization.RegistryPolicy,
    manifest_validator: ManifestValidator,
    include_shared: bool = True,
) -> None:
    """Re-pin only authorities needed to restore and run exact AQ4 safely."""

    _require_record_execution_source(
        record,
        required_uid=policy.required_uid,
    )
    _require_record_runtime_seals(
        record,
        required_uid=policy.required_uid,
        scope="rollback" if include_shared else "rollback_core",
    )
    plan = _stable(
        record.snapshot.path,
        "final activation plan",
        maximum=MAX_DOCUMENT_BYTES,
        read_only=True,
        single_link=True,
    )
    rollback = _stable(
        record.rollback.path,
        "AQ4 rollback manifest",
        maximum=MAX_MANIFEST_BYTES,
        read_only=True,
        single_link=True,
    )
    for current, expected, label in (
        (plan, record.snapshot, "plan"),
        (rollback, record.rollback, "rollback"),
    ):
        if (
            current.path != expected.path
            or current.raw != expected.raw
            or current.identity != expected.identity
        ):
            fail(f"{label} changed during exact AQ4 restoration")
    if include_shared:
        unit = _stable(record.unit.path, "systemd unit")
        environment = _stable(record.environment.path, "systemd environment")
        operations_snapshot = _stable(
            Path(record.document["operations"]["path"]),
            "reviewed operations document",
            maximum=MAX_DOCUMENT_BYTES,
            read_only=True,
            single_link=True,
        )
        for current, expected, label in (
            (unit, record.unit, "unit"),
            (environment, record.environment, "environment"),
        ):
            if (
                current.path != expected.path
                or current.raw != expected.raw
                or current.identity != expected.identity
            ):
                fail(f"{label} changed during exact AQ4 restoration")
        if (
            operations_snapshot.sha256 != record.document["operations"]["sha256"]
            or operations_snapshot.identity.uid != policy.required_uid
        ):
            fail("reviewed operations changed during exact AQ4 restoration")
        operations = _operation_document(
            operations_snapshot,
            required_uid=policy.required_uid,
            verify_executables=True,
            executable_stages={
                "reverse_reconciliation",
                "rollback_live_health",
            },
        )
        if operations != record.operations:
            fail("reviewed operations identity changed during exact AQ4 restoration")

    rollback_worker = _summary_identity(
        manifest_validator(rollback.path),
        snapshot=rollback,
        model_id=AQ4_MODEL_ID,
        format_id=AQ4_FORMAT_ID,
        label="AQ4 rollback",
    )
    if (
        rollback_worker != record.document["rollback"]["worker_binary_sha256"]
        or rollback_worker != record.rollback_runtime.worker.snapshot.sha256
    ):
        fail("AQ4 worker identity changed during exact restoration")
    _require_record_runtime_seals(
        record,
        required_uid=policy.required_uid,
        scope="rollback" if include_shared else "rollback_core",
    )
    _require_record_execution_source(
        record,
        required_uid=policy.required_uid,
    )


def _open_activation_lock(active: Path, *, required_uid: int) -> tuple[int, int]:
    parent_descriptor = _open_parent(active, "active manifest")
    name = f".{active.name}.activation.lock"
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
        descriptor = os.open(name, flags, 0o600, dir_fd=parent_descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_uid != required_uid
        ):
            fail("activation lock metadata is unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FinalActivationError("another activation is in progress") from error
        return descriptor, parent_descriptor
    except BaseException:
        os.close(parent_descriptor)
        raise


def _read_entry_snapshot(
    parent_descriptor: int,
    name: str,
    label: str,
) -> tuple[bytes, tuple[int, ...]]:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        raise FinalActivationError(f"{label} cannot be opened") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 1
            or before.st_size > MAX_MANIFEST_BYTES
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & stat.S_IWOTH
        ):
            fail(f"{label} metadata is unsafe")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65_536, MAX_MANIFEST_BYTES - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_MANIFEST_BYTES:
                fail(f"{label} exceeds its byte bound")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)

        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_nlink,
                value.st_uid,
                value.st_gid,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )

        raw = b"".join(chunks)
        if (
            identity(before) != identity(after)
            or identity(after) != identity(named)
            or len(raw) != before.st_size
        ):
            fail(f"{label} changed while being read")
        return raw, _file_anchor(after)
    finally:
        os.close(descriptor)


def _read_entry(parent_descriptor: int, name: str, label: str) -> bytes:
    return _read_entry_snapshot(parent_descriptor, name, label)[0]


def _rename_exchange(parent_descriptor: int, left: str, right: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        fail("renameat2(RENAME_EXCHANGE) is unavailable")
    result = renameat2(
        ctypes.c_int(parent_descriptor),
        ctypes.c_char_p(os.fsencode(left)),
        ctypes.c_int(parent_descriptor),
        ctypes.c_char_p(os.fsencode(right)),
        ctypes.c_uint(RENAME_EXCHANGE),
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise FinalActivationError("active manifest exchange failed") from OSError(
            error_number,
            os.strerror(error_number),
        )


def _atomic_replace_exact(
    *,
    active: Path,
    parent_descriptor: int,
    expected_raw: bytes,
    replacement_raw: bytes,
) -> None:
    before_raw, before_anchor = _read_entry_snapshot(
        parent_descriptor,
        active.name,
        "actual active manifest",
    )
    if before_raw != expected_raw:
        fail("actual active bytes differ at the atomic replace boundary")
    temporary_name = f".{active.name}.final.{os.getpid()}.{os.urandom(8).hex()}"
    descriptor = -1
    temporary_exists = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_descriptor,
        )
        temporary_exists = True
        os.fchmod(descriptor, 0o644)
        view = memoryview(replacement_raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                fail("active manifest staging write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        staged_anchor = _file_anchor(os.fstat(descriptor))
        os.close(descriptor)
        descriptor = -1
        _rename_exchange(
            parent_descriptor,
            temporary_name,
            active.name,
        )
        os.fsync(parent_descriptor)
        displaced_raw, displaced_anchor = _read_entry_snapshot(
            parent_descriptor,
            temporary_name,
            "displaced active manifest",
        )
        active_raw, active_anchor = _read_entry_snapshot(
            parent_descriptor,
            active.name,
            "actual active manifest",
        )
        displaced_matches = (
            displaced_raw == expected_raw and displaced_anchor == before_anchor
        )
        active_owned = active_raw == replacement_raw and active_anchor == staged_anchor
        if not displaced_matches or not active_owned:
            if active_owned:
                _rename_exchange(
                    parent_descriptor,
                    temporary_name,
                    active.name,
                )
                os.fsync(parent_descriptor)
                restored_raw, restored_anchor = _read_entry_snapshot(
                    parent_descriptor,
                    active.name,
                    "reverted active manifest",
                )
                staged_raw, restored_staged_anchor = _read_entry_snapshot(
                    parent_descriptor,
                    temporary_name,
                    "reverted staging manifest",
                )
                if (
                    restored_raw != displaced_raw
                    or restored_anchor != displaced_anchor
                    or staged_raw != replacement_raw
                    or restored_staged_anchor != staged_anchor
                ):
                    fail("active manifest exchange could not be reverted safely")
            os.unlink(temporary_name, dir_fd=parent_descriptor)
            temporary_exists = False
            os.fsync(parent_descriptor)
            fail("actual active entry changed at the atomic exchange boundary")
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        temporary_exists = False
        os.fsync(parent_descriptor)
        final_raw, final_anchor = _read_entry_snapshot(
            parent_descriptor,
            active.name,
            "actual active manifest",
        )
        if final_raw != replacement_raw or final_anchor != staged_anchor:
            fail("active manifest changed after its verified atomic exchange")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_exists:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except OSError:
                pass


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _file_anchor(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
    )


def _open_verified_executable(
    path: Path,
    *,
    expected_sha256: str,
    required_uid: int,
    label: str,
) -> int:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
        before = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            _file_anchor(before) != _file_anchor(named)
            or not stat.S_ISREG(before.st_mode)
            or before.st_size < 1
            or before.st_size > MAX_INPUT_BYTES
            or before.st_uid not in {0, required_uid}
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & 0o022
            or before.st_mode & (stat.S_ISUID | stat.S_ISGID)
            or not before.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        ):
            fail(f"{label} metadata is unsafe")
        if runtime_seal._has_posix_acl(descriptor):
            fail(f"{label} has a POSIX ACL")
        if runtime_seal._has_forbidden_security_xattr(descriptor):
            fail(f"{label} has a file capability")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                fail(f"{label} changed while read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        raw = b"".join(chunks)
        if (
            _file_anchor(after) != _file_anchor(before)
            or _sha256(raw) != expected_sha256
            or raw.startswith(b"#!")
        ):
            fail(f"{label} differs from its reviewed executable")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except OSError as error:
        raise FinalActivationError(f"{label} cannot be opened safely") from error
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _prctl(option: int, argument: int = 0) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.prctl(
        ctypes.c_int(option),
        ctypes.c_ulong(argument),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise FinalActivationError("Linux child-subreaper control failed") from OSError(
            error_number,
            os.strerror(error_number),
        )
    return result


def _get_child_subreaper() -> bool:
    value = ctypes.c_int(0)
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.prctl(
        ctypes.c_int(PR_GET_CHILD_SUBREAPER),
        ctypes.byref(value),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise FinalActivationError("Linux child-subreaper query failed") from OSError(
            error_number,
            os.strerror(error_number),
        )
    return bool(value.value)


def _direct_children() -> dict[int, int]:
    children: dict[int, int] = {}
    try:
        entries = tuple(os.scandir("/proc"))
    except OSError as error:
        raise FinalActivationError("procfs child enumeration failed") from error
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name, 10)
        try:
            ppid, starttime = _proc_stat_identity(pid)
        except FinalActivationError:
            continue
        if ppid == os.getpid():
            children[pid] = starttime
    return children


def _matching_child(pid: int, starttime: int) -> bool:
    try:
        ppid, observed_starttime = _proc_stat_identity(pid)
    except FinalActivationError:
        return False
    return ppid == os.getpid() and observed_starttime == starttime


def _reap_child(pid: int) -> None:
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, ProcessLookupError):
        pass


def _terminate_new_descendants(baseline: dict[int, int]) -> bool:
    found = False
    deadline = time.monotonic() + COMMAND_TERMINATION_GRACE_SECONDS
    signum = signal.SIGTERM
    while True:
        current = {
            pid: starttime
            for pid, starttime in _direct_children().items()
            if baseline.get(pid) != starttime
        }
        if current:
            found = True
        for pid, starttime in current.items():
            if not _matching_child(pid, starttime):
                continue
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass
        for pid in current:
            _reap_child(pid)
        remaining = {
            pid: starttime
            for pid, starttime in _direct_children().items()
            if baseline.get(pid) != starttime
        }
        if not remaining:
            return found
        if time.monotonic() >= deadline:
            if signum == signal.SIGKILL:
                fail("reviewed operation escaped child-subreaper cleanup")
            signum = signal.SIGKILL
            deadline = time.monotonic() + COMMAND_TERMINATION_GRACE_SECONDS
        time.sleep(0.02)


def _terminate_owned_process_group(process: subprocess.Popen[Any]) -> None:
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + COMMAND_TERMINATION_GRACE_SECONDS
    while _process_group_exists(process_group) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.02)
    if _process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=COMMAND_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise FinalActivationError(
            "reviewed operation leader could not be reaped"
        ) from error
    if _process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_owned_process_group(
    argv: list[str],
    *,
    executable_fd: int,
    environment: dict[str, str],
    timeout: float,
    stage: str,
) -> None:
    process: subprocess.Popen[Any] | None = None
    previous_subreaper = _get_child_subreaper()
    if not previous_subreaper:
        _prctl(PR_SET_CHILD_SUBREAPER, 1)
    command_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    returncode: int | None = None
    leaked = False
    try:
        baseline = _direct_children()
        try:
            process = subprocess.Popen(
                argv,
                executable=f"/proc/self/fd/{executable_fd}",
                cwd="/",
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=environment,
                start_new_session=True,
                close_fds=True,
                pass_fds=(executable_fd,),
            )
            returncode = process.wait(timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as error:
            command_error = error
        except BaseException as error:
            command_error = error
        if process is not None and (
            command_error is not None
            or returncode != 0
            or _process_group_exists(process.pid)
        ):
            try:
                _terminate_owned_process_group(process)
            except BaseException as error:
                cleanup_error = error
        try:
            leaked = _terminate_new_descendants(baseline)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
    finally:
        if not previous_subreaper:
            _prctl(PR_SET_CHILD_SUBREAPER, 0)
    if cleanup_error is not None:
        raise FinalActivationError(
            f"{stage} command descendant cleanup failed"
        ) from cleanup_error
    if command_error is not None:
        if isinstance(command_error, (KeyboardInterrupt, SystemExit, FinalActivationInterrupted)):
            raise command_error
        raise FinalActivationError(f"{stage} command failed") from command_error
    if returncode != 0:
        fail(f"{stage} command failed")
    if leaked:
        fail(f"{stage} command left descendant processes")


def _remaining_window(deadline: float, label: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        fail(f"{label} exceeded the plan-bound active window")
    return remaining


def _run_stage(
    record: PlanRecord,
    stage: str,
    *,
    runner: CommandRunner,
    repin: Callable[[], None],
    require_active: Callable[[], None],
    activation_epoch: str,
    live_proof_loader: LiveProofLoader,
    live_state_verifier: LiveStateVerifier,
    clock: Clock,
    deadline: float,
) -> dict[str, Any] | None:
    commands = record.operations["stages"][stage]
    timeout = float(record.operations["timeout_seconds"])
    stage_started = clock()
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "ULLM_FINAL_ACTIVATION_PLAN": os.fspath(record.snapshot.path),
        "ULLM_FINAL_ACTIVATION_PLAN_SHA256": record.snapshot.sha256,
        "ULLM_FINAL_ACTIVATION_STAGE": stage,
        "ULLM_FINAL_ACTIVATION_EPOCH": activation_epoch,
        "ULLM_ACTIVE_MANIFEST": record.document["active_manifest"]["path"],
        "ULLM_CANDIDATE_MANIFEST_SHA256": record.candidate.sha256,
        "ULLM_ROLLBACK_MANIFEST_SHA256": record.rollback.sha256,
    }
    if stage in {"candidate_live_health", "rollback_live_health"}:
        proof_path = Path(record.document["live_proofs"][stage]["path"])
        _ensure_fresh_destination(proof_path, f"{stage} live proof")
        environment["ULLM_FINAL_ACTIVATION_LIVE_PROOF"] = os.fspath(proof_path)
    for index, command in enumerate(commands):
        _remaining_window(deadline, stage)
        repin()
        require_active()
        executable = Path(command["argv"][0])
        executable_fd = _open_verified_executable(
            executable,
            expected_sha256=command["executable_sha256"],
            required_uid=record.snapshot.identity.uid,
            label=f"operations executable {stage}[{index}]",
        )
        try:
            if runner is subprocess.run:
                _run_owned_process_group(
                    command["argv"],
                    executable_fd=executable_fd,
                    environment=environment,
                    timeout=min(timeout, _remaining_window(deadline, stage)),
                    stage=stage,
                )
            else:
                try:
                    completed = runner(
                        command["argv"],
                        cwd="/",
                        check=False,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=environment,
                        timeout=min(timeout, _remaining_window(deadline, stage)),
                    )
                except (OSError, subprocess.TimeoutExpired) as error:
                    raise FinalActivationError(f"{stage} command failed") from error
                if completed.returncode != 0:
                    fail(f"{stage} command failed")
        finally:
            os.close(executable_fd)
        repin()
        require_active()
        _remaining_window(deadline, stage)
    if stage not in {"candidate_live_health", "rollback_live_health"}:
        return None
    _remaining_window(deadline, stage)
    proof = live_proof_loader(record, stage, activation_epoch)
    verified_at = clock()
    reference = _validate_live_proof(
        record,
        stage,
        activation_epoch,
        proof,
        stage_started=stage_started,
        verified_at=verified_at,
    )
    # The production loader requires an immutable file.  Re-open it after
    # validation to bind the named entry to the reference stored in outcomes.
    if live_proof_loader is default_live_proof_loader:
        _validate_live_proof_reference(record, stage, reference)
    live_state_verifier(
        record,
        stage,
        proof,
        stage_started,
        verified_at,
        _remaining_window(deadline, stage),
    )
    repin()
    require_active()
    _remaining_window(deadline, stage)
    return {"reference": reference, "document": proof}


class _TerminationGuard:
    """Raise promptly except while publishing the immutable commit record."""

    def __init__(self) -> None:
        self.installed: dict[int, Any] = {}
        self.depth = 0
        self.pending: int | None = None
        self.committed = False

    def __enter__(self) -> "_TerminationGuard":
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                self.installed[signum] = signal.getsignal(signum)
                signal.signal(signum, self._interrupt)
            except (ValueError, OSError):
                continue
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        for signum, handler in self.installed.items():
            signal.signal(signum, handler)

    def _interrupt(self, signum: int, _frame: Any) -> None:
        if self.committed:
            return
        if self.depth:
            self.pending = signum
            return
        raise FinalActivationInterrupted(f"termination signal {signum}")

    @contextmanager
    def deferred(self) -> Iterator[None]:
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1
            if not self.depth and self.pending is not None and not self.committed:
                signum = self.pending
                self.pending = None
                raise FinalActivationInterrupted(f"termination signal {signum}")

    @contextmanager
    def recovery_deferred(self) -> Iterator[None]:
        """Keep a requested termination pending until recovery is committed."""

        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1

    def commit(self) -> None:
        self.committed = True
        self.pending = None


def _termination_guard() -> _TerminationGuard:
    return _TerminationGuard()


def _complete_stages(
    stages: dict[str, str],
    *,
    failure_stage: str | None,
) -> None:
    for name, state in tuple(stages.items()):
        if state == "pending":
            stages[name] = "skipped"
    if failure_stage is not None:
        stages[failure_stage] = "failed"


def _activation_outcome(
    record: PlanRecord,
    *,
    started_at: datetime,
    completed_at: datetime,
    status: str,
    failure_stage: str | None,
    stages: dict[str, str],
    observed_manifest_sha256: str | None,
    restoration_attempted: bool,
    live_proofs: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    bytes_equal = observed_manifest_sha256 == record.rollback.sha256
    return {
        "schema_version": ACTIVATION_OUTCOME_SCHEMA,
        "plan_id": record.document["plan_id"],
        "plan_path": os.fspath(record.snapshot.path),
        "plan_sha256": record.snapshot.sha256,
        "started_at": utc_timestamp(started_at),
        "completed_at": utc_timestamp(completed_at),
        "status": status,
        "failure_stage": failure_stage,
        "stages": stages,
        "before_manifest_sha256": record.rollback.sha256,
        "candidate_manifest_sha256": record.candidate.sha256,
        "observed_manifest_sha256": observed_manifest_sha256,
        "live_proofs": live_proofs,
        "restoration": {
            "attempted": restoration_attempted,
            "manifest_sha256": record.rollback.sha256,
            "bytes_equal": bytes_equal,
            "reverse_reconciliation_passed": (
                stages["reverse_reconciliation"] == "passed"
            ),
            "live_health_passed": stages["rollback_live_health"] == "passed",
        },
    }


def _observed_hash(parent_descriptor: int, active: Path) -> str | None:
    try:
        return _sha256(
            _read_entry(parent_descriptor, active.name, "actual active manifest")
        )
    except FinalActivationError:
        return None


def _require_execution_authority(
    *,
    expected_plan_sha256: str,
    confirmation: str,
    expected_confirmation: str,
) -> None:
    _hash(expected_plan_sha256, "confirmed plan SHA-256")
    if confirmation != expected_confirmation:
        fail("explicit final-route confirmation differs")


def _require_same_plan(
    initial: PlanRecord,
    locked: PlanRecord,
    *,
    expected_plan_sha256: str,
) -> None:
    if (
        initial.snapshot.sha256 != expected_plan_sha256
        or locked.snapshot.sha256 != expected_plan_sha256
        or initial.snapshot.path != locked.snapshot.path
        or initial.snapshot.raw != locked.snapshot.raw
        or initial.snapshot.identity != locked.snapshot.identity
        or initial.execution_source != locked.execution_source
        or initial.document["active_manifest"]["path"]
        != locked.document["active_manifest"]["path"]
    ):
        fail("confirmed final activation plan changed before the locked boundary")


def execute_activation(
    plan_path: Path,
    *,
    expected_plan_sha256: str,
    confirmation: str,
    policy: authorization.RegistryPolicy = authorization.RegistryPolicy(),
    manifest_validator: ManifestValidator = default_manifest_validator,
    bundle_validator: BundleValidator = default_bundle_validator,
    runner: CommandRunner = subprocess.run,
    live_proof_loader: LiveProofLoader = default_live_proof_loader,
    live_state_verifier: LiveStateVerifier = default_live_state_verifier,
    clock: Clock = utc_now,
) -> ExecutionResult:
    """Execute one fully bound activation; restore AQ4 on every later failure."""

    _require_execution_authority(
        expected_plan_sha256=expected_plan_sha256,
        confirmation=confirmation,
        expected_confirmation=ACTIVATION_CONFIRMATION,
    )
    initial = load_plan(
        plan_path,
        action="activate",
        now=clock(),
        policy=policy,
        manifest_validator=manifest_validator,
        bundle_validator=bundle_validator,
    )
    if initial.snapshot.sha256 != expected_plan_sha256:
        fail("confirmed plan SHA-256 differs before activation")
    active = Path(initial.document["active_manifest"]["path"])
    lock_descriptor: int | None = None
    parent_descriptor: int | None = None
    record: PlanRecord | None = None
    stages = {name: "pending" for name in ACTIVATION_STAGES}
    failure_stage: str | None = None
    switched = False
    started_at = clock()
    primary_error: BaseException | None = None
    outcome_snapshot: StableFileSnapshot | None = None
    live_proofs: dict[str, dict[str, Any] | None] = {
        "candidate_live_health": None,
        "rollback_live_health": None,
    }
    active_deadline: float | None = None
    credential_seals: tuple[runtime_seal.RuntimeArtifactSeal, ...] | None = None
    with _termination_guard() as termination:
        try:
            try:
                lock_descriptor, parent_descriptor = _open_activation_lock(
                    active,
                    required_uid=policy.required_uid,
                )
                stages["lock"] = "passed"
            except BaseException as error:
                failure_stage = "lock"
                primary_error = error
                raise
            try:
                record = load_plan(
                    plan_path,
                    action="activate",
                    now=clock(),
                    policy=policy,
                    manifest_validator=manifest_validator,
                    bundle_validator=bundle_validator,
                )
                _require_same_plan(
                    initial,
                    record,
                    expected_plan_sha256=expected_plan_sha256,
                )
                stages["preflight"] = "passed"
            except BaseException as error:
                failure_stage = "preflight"
                primary_error = error
                raise
            assert parent_descriptor is not None

            def repin() -> None:
                assert record is not None
                _repin_plan_inputs(
                    record,
                    now=clock(),
                    policy=policy,
                    manifest_validator=manifest_validator,
                    bundle_validator=bundle_validator,
                )

            def repin_rollback() -> None:
                assert record is not None
                _repin_rollback_inputs(
                    record,
                    policy=policy,
                    manifest_validator=manifest_validator,
                )

            def repin_rollback_core() -> None:
                assert record is not None
                _repin_rollback_inputs(
                    record,
                    policy=policy,
                    manifest_validator=manifest_validator,
                    include_shared=False,
                )

            def repin_credentials() -> None:
                nonlocal credential_seals
                if live_state_verifier is not default_live_state_verifier:
                    return
                if credential_seals is None:
                    credential_seals = _capture_live_credential_seals(
                        required_uid=policy.required_uid,
                    )
                else:
                    _require_live_credential_seals(credential_seals)

            def repin_candidate_stage() -> None:
                repin()
                repin_credentials()

            def repin_rollback_stage() -> None:
                repin_rollback()
                repin_credentials()

            def require_candidate() -> None:
                assert record is not None
                if (
                    _read_entry(
                        parent_descriptor,
                        active.name,
                        "actual active manifest",
                    )
                    != record.candidate.raw
                ):
                    fail("candidate active bytes changed during final activation")

            def require_rollback() -> None:
                assert record is not None
                if (
                    _read_entry(
                        parent_descriptor,
                        active.name,
                        "actual active manifest",
                    )
                    != record.rollback.raw
                ):
                    fail("AQ4 active bytes changed during final activation rollback")

            try:
                repin()
                require_rollback()
                active_deadline = time.monotonic() + float(
                    record.operations["active_window_timeout_seconds"]
                )
                _remaining_window(active_deadline, "final activation")
                switched = True
                _atomic_replace_exact(
                    active=active,
                    parent_descriptor=parent_descriptor,
                    expected_raw=record.rollback.raw,
                    replacement_raw=record.candidate.raw,
                )
                repin()
                require_candidate()
                _remaining_window(active_deadline, "final activation")
                stages["candidate_activation"] = "passed"
            except BaseException as error:
                failure_stage = "candidate_activation"
                primary_error = error
                raise
            for stage in ("candidate_reconciliation", "candidate_live_health"):
                try:
                    proof_reference = _run_stage(
                        record,
                        stage,
                        runner=runner,
                        repin=repin_candidate_stage,
                        require_active=require_candidate,
                        activation_epoch=os.urandom(32).hex(),
                        live_proof_loader=live_proof_loader,
                        live_state_verifier=live_state_verifier,
                        clock=clock,
                        deadline=active_deadline,
                    )
                    if proof_reference is not None:
                        live_proofs[stage] = proof_reference
                    stages[stage] = "passed"
                except BaseException as error:
                    failure_stage = stage
                    primary_error = error
                    raise
            repin_candidate_stage()
            require_candidate()
            _remaining_window(active_deadline, "final activation")
            if live_proofs["candidate_live_health"] is None:
                fail("candidate live health lacks a structured proof")
            stages["outcome_publication"] = "passed"
            _complete_stages(stages, failure_stage=None)
            outcome = _activation_outcome(
                record,
                started_at=started_at,
                completed_at=clock(),
                status="activated",
                failure_stage=None,
                stages=stages,
                observed_manifest_sha256=record.candidate.sha256,
                restoration_attempted=False,
                live_proofs=live_proofs,
            )
            with termination.deferred():
                _require_record_execution_source(
                    record,
                    required_uid=policy.required_uid,
                )
                outcome_snapshot = _publish_immutable(
                    Path(record.document["outcomes"]["activation_path"]),
                    outcome,
                    required_uid=policy.required_uid,
                )
                _require_record_execution_source(
                    record,
                    required_uid=policy.required_uid,
                )
                termination.commit()
        except BaseException as error:
            if primary_error is None:
                primary_error = error
            if failure_stage is None:
                failure_stage = "outcome_publication"
                stages["outcome_publication"] = "failed"
            if switched and record is not None and parent_descriptor is not None:
                recovery_deadline = time.monotonic() + float(
                    record.operations["active_window_timeout_seconds"]
                )
                try:
                    repin_rollback_core()
                    with termination.recovery_deferred():
                        current = _read_entry(
                            parent_descriptor,
                            active.name,
                            "actual active manifest",
                        )
                        if current == record.candidate.raw:
                            _atomic_replace_exact(
                                active=active,
                                parent_descriptor=parent_descriptor,
                                expected_raw=record.candidate.raw,
                                replacement_raw=record.rollback.raw,
                            )
                        elif current != record.rollback.raw:
                            _atomic_replace_exact(
                                active=active,
                                parent_descriptor=parent_descriptor,
                                expected_raw=current,
                                replacement_raw=record.rollback.raw,
                            )
                    repin_rollback_core()
                    require_rollback()
                    stages["aq4_restore"] = "passed"
                except BaseException:
                    stages["aq4_restore"] = "failed"
                if stages["aq4_restore"] == "passed":
                    for rollback_stage in (
                        "reverse_reconciliation",
                        "rollback_live_health",
                    ):
                        try:
                            proof_reference = _run_stage(
                                record,
                                rollback_stage,
                                runner=runner,
                                repin=repin_rollback_stage,
                                require_active=require_rollback,
                                activation_epoch=os.urandom(32).hex(),
                                live_proof_loader=live_proof_loader,
                                live_state_verifier=live_state_verifier,
                                clock=clock,
                                deadline=recovery_deadline,
                            )
                            if proof_reference is not None:
                                live_proofs[rollback_stage] = proof_reference
                            stages[rollback_stage] = "passed"
                        except BaseException:
                            stages[rollback_stage] = "failed"
                            break
                restored = (
                    stages["aq4_restore"] == "passed"
                    and stages["reverse_reconciliation"] == "passed"
                    and stages["rollback_live_health"] == "passed"
                    and _read_entry(
                        parent_descriptor,
                        active.name,
                        "actual active manifest",
                    )
                    == record.rollback.raw
                )
                _complete_stages(stages, failure_stage=failure_stage)
                status = "failed_restored" if restored else "failed_restore"
                outcome = _activation_outcome(
                    record,
                    started_at=started_at,
                    completed_at=clock(),
                    status=status,
                    failure_stage=failure_stage,
                    stages=stages,
                    observed_manifest_sha256=_observed_hash(
                        parent_descriptor,
                        active,
                    ),
                    restoration_attempted=True,
                    live_proofs=live_proofs,
                )
                try:
                    with termination.deferred():
                        _require_record_execution_source(
                            record,
                            required_uid=policy.required_uid,
                        )
                        outcome_snapshot = _publish_immutable(
                            Path(record.document["outcomes"]["activation_path"]),
                            outcome,
                            required_uid=policy.required_uid,
                        )
                        _require_record_execution_source(
                            record,
                            required_uid=policy.required_uid,
                        )
                        termination.commit()
                except BaseException:
                    pass
            if isinstance(primary_error, (KeyboardInterrupt, SystemExit)):
                raise primary_error
        finally:
            if lock_descriptor is not None:
                os.close(lock_descriptor)
            if parent_descriptor is not None:
                os.close(parent_descriptor)
    if outcome_snapshot is None or record is None:
        if isinstance(primary_error, FinalActivationError):
            raise primary_error
        raise FinalActivationError(
            "final activation failed before an outcome"
        ) from primary_error
    outcome_document = _strict_object(outcome_snapshot.raw, "activation outcome")
    if outcome_document["status"] != "activated":
        raise FinalActivationError(
            "final activation failed and AQ4 restoration was attempted"
        )
    return ExecutionResult(
        outcome_snapshot.path,
        outcome_snapshot.sha256,
        "activated",
    )


def _rollback_outcome(
    record: PlanRecord,
    *,
    activation_outcome_sha256: str,
    started_at: datetime,
    completed_at: datetime,
    status: str,
    failure_stage: str | None,
    stages: dict[str, str],
    observed_manifest_sha256: str | None,
    live_proof: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": ROLLBACK_OUTCOME_SCHEMA,
        "plan_id": record.document["plan_id"],
        "plan_path": os.fspath(record.snapshot.path),
        "plan_sha256": record.snapshot.sha256,
        "activation_outcome_sha256": activation_outcome_sha256,
        "started_at": utc_timestamp(started_at),
        "completed_at": utc_timestamp(completed_at),
        "status": status,
        "failure_stage": failure_stage,
        "stages": stages,
        "expected_current_manifest_sha256": record.candidate.sha256,
        "rollback_manifest_sha256": record.rollback.sha256,
        "observed_manifest_sha256": observed_manifest_sha256,
        "bytes_equal": observed_manifest_sha256 == record.rollback.sha256,
        "live_proof": live_proof,
    }


def execute_rollback(
    plan_path: Path,
    *,
    expected_plan_sha256: str,
    confirmation: str,
    policy: authorization.RegistryPolicy = authorization.RegistryPolicy(),
    manifest_validator: ManifestValidator = default_manifest_validator,
    bundle_validator: BundleValidator = default_bundle_validator,
    runner: CommandRunner = subprocess.run,
    live_proof_loader: LiveProofLoader = default_live_proof_loader,
    live_state_verifier: LiveStateVerifier = default_live_state_verifier,
    clock: Clock = utc_now,
) -> ExecutionResult:
    """Roll a successfully activated plan back to its exact AQ4 bytes."""

    _require_execution_authority(
        expected_plan_sha256=expected_plan_sha256,
        confirmation=confirmation,
        expected_confirmation=ROLLBACK_CONFIRMATION,
    )
    initial = load_plan(
        plan_path,
        action="rollback",
        now=clock(),
        policy=policy,
        manifest_validator=manifest_validator,
        bundle_validator=bundle_validator,
    )
    if initial.snapshot.sha256 != expected_plan_sha256:
        fail("confirmed plan SHA-256 differs before rollback")
    active = Path(initial.document["active_manifest"]["path"])
    activation_snapshot, _activation_document = _load_activation_outcome(
        Path(initial.document["outcomes"]["activation_path"]),
        plan=initial.snapshot,
        required_uid=policy.required_uid,
        record=initial,
    )
    lock_descriptor: int | None = None
    parent_descriptor: int | None = None
    stages = {name: "pending" for name in ROLLBACK_STAGES}
    failure_stage: str | None = None
    record: PlanRecord | None = None
    started_at = clock()
    outcome_snapshot: StableFileSnapshot | None = None
    primary_error: BaseException | None = None
    rollback_live_proof: dict[str, Any] | None = None
    rollback_deadline: float | None = None
    credential_seals: tuple[runtime_seal.RuntimeArtifactSeal, ...] | None = None
    with _termination_guard() as termination:
        try:
            try:
                lock_descriptor, parent_descriptor = _open_activation_lock(
                    active,
                    required_uid=policy.required_uid,
                )
                stages["lock"] = "passed"
            except BaseException as error:
                failure_stage = "lock"
                primary_error = error
                raise
            try:
                record = load_plan(
                    plan_path,
                    action="rollback",
                    now=clock(),
                    policy=policy,
                    manifest_validator=manifest_validator,
                    bundle_validator=bundle_validator,
                )
                _require_same_plan(
                    initial,
                    record,
                    expected_plan_sha256=expected_plan_sha256,
                )
                locked_activation_snapshot, _locked_activation_document = (
                    _load_activation_outcome(
                        Path(record.document["outcomes"]["activation_path"]),
                        plan=record.snapshot,
                        required_uid=policy.required_uid,
                        record=record,
                    )
                )
                if (
                    locked_activation_snapshot.path != activation_snapshot.path
                    or locked_activation_snapshot.raw != activation_snapshot.raw
                    or locked_activation_snapshot.identity
                    != activation_snapshot.identity
                ):
                    fail(
                        "successful activation outcome changed before "
                        "the locked rollback boundary"
                    )
                activation_snapshot = locked_activation_snapshot
                stages["preflight"] = "passed"
            except BaseException as error:
                failure_stage = "preflight"
                primary_error = error
                raise
            assert parent_descriptor is not None

            def repin() -> None:
                assert record is not None
                _repin_plan_inputs(
                    record,
                    now=clock(),
                    policy=policy,
                    manifest_validator=manifest_validator,
                    bundle_validator=bundle_validator,
                )

            def repin_rollback() -> None:
                assert record is not None
                _repin_rollback_inputs(
                    record,
                    policy=policy,
                    manifest_validator=manifest_validator,
                )

            def repin_rollback_core() -> None:
                assert record is not None
                _repin_rollback_inputs(
                    record,
                    policy=policy,
                    manifest_validator=manifest_validator,
                    include_shared=False,
                )

            def repin_credentials() -> None:
                nonlocal credential_seals
                if live_state_verifier is not default_live_state_verifier:
                    return
                if credential_seals is None:
                    credential_seals = _capture_live_credential_seals(
                        required_uid=policy.required_uid,
                    )
                else:
                    _require_live_credential_seals(credential_seals)

            def repin_rollback_stage() -> None:
                repin_rollback()
                repin_credentials()

            def require_candidate() -> None:
                assert record is not None
                if (
                    _read_entry(
                        parent_descriptor,
                        active.name,
                        "actual active manifest",
                    )
                    != record.candidate.raw
                ):
                    fail("candidate active bytes changed during manual rollback")

            def require_rollback() -> None:
                assert record is not None
                if (
                    _read_entry(
                        parent_descriptor,
                        active.name,
                        "actual active manifest",
                    )
                    != record.rollback.raw
                ):
                    fail("AQ4 active bytes changed during manual rollback")

            try:
                repin_rollback()
                require_candidate()
                rollback_deadline = time.monotonic() + float(
                    record.operations["active_window_timeout_seconds"]
                )
                _remaining_window(rollback_deadline, "manual rollback")
                with termination.recovery_deferred():
                    _atomic_replace_exact(
                        active=active,
                        parent_descriptor=parent_descriptor,
                        expected_raw=record.candidate.raw,
                        replacement_raw=record.rollback.raw,
                    )
                repin_rollback()
                require_rollback()
                _remaining_window(rollback_deadline, "manual rollback")
                stages["aq4_restore"] = "passed"
            except BaseException as error:
                failure_stage = "aq4_restore"
                primary_error = error
                raise
            for stage in ("reverse_reconciliation", "rollback_live_health"):
                try:
                    proof_reference = _run_stage(
                        record,
                        stage,
                        runner=runner,
                        repin=repin_rollback_stage,
                        require_active=require_rollback,
                        activation_epoch=os.urandom(32).hex(),
                        live_proof_loader=live_proof_loader,
                        live_state_verifier=live_state_verifier,
                        clock=clock,
                        deadline=rollback_deadline,
                    )
                    if proof_reference is not None:
                        rollback_live_proof = proof_reference
                    stages[stage] = "passed"
                except BaseException as error:
                    failure_stage = stage
                    primary_error = error
                    raise
            repin_rollback_stage()
            require_rollback()
            _remaining_window(rollback_deadline, "manual rollback")
            if rollback_live_proof is None:
                failure_stage = "rollback_live_health"
                fail("AQ4 rollback live health lacks a structured proof")
            stages["outcome_publication"] = "passed"
            _complete_stages(stages, failure_stage=None)
            outcome = _rollback_outcome(
                record,
                activation_outcome_sha256=activation_snapshot.sha256,
                started_at=started_at,
                completed_at=clock(),
                status="rolled_back",
                failure_stage=None,
                stages=stages,
                observed_manifest_sha256=record.rollback.sha256,
                live_proof=rollback_live_proof,
            )
            with termination.deferred():
                _require_record_execution_source(
                    record,
                    required_uid=policy.required_uid,
                )
                outcome_snapshot = _publish_immutable(
                    Path(record.document["outcomes"]["rollback_path"]),
                    outcome,
                    required_uid=policy.required_uid,
                )
                _require_record_execution_source(
                    record,
                    required_uid=policy.required_uid,
                )
                termination.commit()
        except BaseException as error:
            if primary_error is None:
                primary_error = error
            if failure_stage is None:
                failure_stage = "outcome_publication"
                stages["outcome_publication"] = "failed"
            if record is not None and parent_descriptor is not None:
                try:
                    repin_rollback_core()
                    with termination.recovery_deferred():
                        current = _read_entry(
                            parent_descriptor,
                            active.name,
                            "actual active manifest",
                        )
                        if current != record.rollback.raw:
                            _atomic_replace_exact(
                                active=active,
                                parent_descriptor=parent_descriptor,
                                expected_raw=current,
                                replacement_raw=record.rollback.raw,
                            )
                    repin_rollback_core()
                    require_rollback()
                    stages["aq4_restore"] = "passed"
                except BaseException:
                    stages["aq4_restore"] = "failed"
                _complete_stages(stages, failure_stage=failure_stage)
                observed = _observed_hash(parent_descriptor, active)
                outcome = _rollback_outcome(
                    record,
                    activation_outcome_sha256=activation_snapshot.sha256,
                    started_at=started_at,
                    completed_at=clock(),
                    status="rollback_incomplete",
                    failure_stage=failure_stage,
                    stages=stages,
                    observed_manifest_sha256=observed,
                    live_proof=rollback_live_proof,
                )
                try:
                    with termination.deferred():
                        _require_record_execution_source(
                            record,
                            required_uid=policy.required_uid,
                        )
                        outcome_snapshot = _publish_immutable(
                            Path(record.document["outcomes"]["rollback_path"]),
                            outcome,
                            required_uid=policy.required_uid,
                        )
                        _require_record_execution_source(
                            record,
                            required_uid=policy.required_uid,
                        )
                        termination.commit()
                except BaseException:
                    pass
            if isinstance(primary_error, (KeyboardInterrupt, SystemExit)):
                raise primary_error
        finally:
            if lock_descriptor is not None:
                os.close(lock_descriptor)
            if parent_descriptor is not None:
                os.close(parent_descriptor)
    if outcome_snapshot is None or record is None:
        if isinstance(primary_error, FinalActivationError):
            raise primary_error
        raise FinalActivationError(
            "rollback failed before an outcome"
        ) from primary_error
    outcome_document = _strict_object(outcome_snapshot.raw, "rollback outcome")
    if outcome_document["status"] != "rolled_back":
        raise FinalActivationError("rollback reconciliation or live health failed")
    return ExecutionResult(
        outcome_snapshot.path,
        outcome_snapshot.sha256,
        "rolled_back",
    )


def preflight_report(record: PlanRecord, *, action: str) -> dict[str, Any]:
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "ready": True,
        "action": action,
        "plan_id": record.document["plan_id"],
        "plan_sha256": record.snapshot.sha256,
        "active_manifest_sha256": record.active.sha256,
        "candidate_manifest_sha256": record.candidate.sha256,
        "rollback_manifest_sha256": record.rollback.sha256,
        "aq4_release_bundle_sha256": (
            record.aq4_bundle.sha256
            if record.aq4_bundle is not None
            else record.document["aq4_release_bundle"]["sha256"]
        ),
        "release_bundle_sha256": (
            record.bundle.sha256
            if record.bundle is not None
            else record.document["release_bundle"]["sha256"]
        ),
        "campaign_outcome_sha256": (
            record.campaign_outcome.sha256
            if record.campaign_outcome is not None
            else record.document["campaign"]["outcome_sha256"]
        ),
        "execution_source_root": os.fspath(record.execution_source.root),
        "execution_source_fingerprint_sha256": (
            record.execution_source.fingerprint_sha256
        ),
        "active_manifest_changed": False,
        "commands_executed": False,
    }
