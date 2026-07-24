#!/usr/bin/env python3
"""Fail-closed production preflight primitives for one full SQ8 campaign."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import secrets
import selectors
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import Any, NoReturn, Protocol, Sequence, cast


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import served_model_campaign_source_seal as campaign_source_seal  # noqa: E402


PRODUCTION_REPO_ROOT = Path("/home/homelab1/coding-local/ultimateLLM/uLLM-project")
PRODUCTION_PRODUCT_ROOT = Path(
    "/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1"
)
PRODUCTION_PYTHON_EXECUTABLE = Path("/usr/bin/python3.12")
PRODUCTION_PYTHON_PREFIX = (
    os.fspath(PRODUCTION_PYTHON_EXECUTABLE),
    "-I",
    "-S",
    "-B",
)
SEALED_TOOL_LAUNCH_SOURCE = (
    "import runpy,sys;"
    "d,p,*a=sys.argv[1:];"
    "sys.path.insert(0,d);"
    "sys.argv=[p,*a];"
    "runpy.run_path(p,run_name='__main__')"
)
PRODUCTION_LOCK_NAME = "ullm-sq8-full-openwebui-campaign.lock"

PROMOTION_SCHEMA = "ullm.sq8_product_promotion.v1"
HEAD_PROMOTION_TOOL_PATHS = (
    "tools/validate-sq8-product-promotion.py",
    "tools/sq8_canonical_artifact.py",
)
HEAD_HTTP_CLIENT_PATH = "tools/sq8-openwebui-http-client.py"

GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
GIT_TIMEOUT_SECONDS = 10.0
GIT_HEAD_MAX_BYTES = 128
GIT_STATUS_MAX_BYTES = 4 << 20
HEAD_TOOL_MAX_BYTES = 32 << 20
HEAD_HTTP_CLIENT_MAX_BYTES = 1 << 20
PROMOTION_TIMEOUT_SECONDS = 6 * 60 * 60.0
PROMOTION_STDOUT_MAX_BYTES = 2 << 20
PROMOTION_STDERR_MAX_BYTES = 64 << 10
COMMAND_READ_CHUNK_BYTES = 64 << 10
MAX_CANDIDATE_MANIFEST_BYTES = 1 << 20
MAX_RUNTIME_FILE_BYTES = 256 << 20
MAX_RUNTIME_BINDINGS = 256
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
ROCM_ROOT = Path("/opt/rocm-7.2.1")
ROCM_AMD_SMI_ALIAS = Path("/opt/rocm/bin/amd-smi")
ROCM_AMD_SMI_SCRIPT = (
    ROCM_ROOT / "libexec/amdsmi_cli/amdsmi_cli.py"
)
ROCM_PYTHON_TCB_TREES = (
    ROCM_ROOT / "libexec/amdsmi_cli",
    ROCM_ROOT / "share/amd_smi/amdsmi",
)
MAX_ROCM_TCB_ENTRIES = 4096


class ProductionPreflightError(RuntimeError):
    """A production preflight binding or read-only validation failed."""


def fail(message: str) -> NoReturn:
    raise ProductionPreflightError(message)


def _require_canonical_absolute(path: Path, label: str) -> None:
    if not isinstance(path, Path) or not path.is_absolute():
        fail(f"{label} must be an absolute Path")
    if Path(os.path.abspath(path)) != path:
        fail(f"{label} must be lexically canonical")


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeFileBinding:
    path: Path
    sha256: str
    maximum_bytes: int

    def __post_init__(self) -> None:
        _require_canonical_absolute(self.path, "transaction runtime file")
        if (
            SHA256_RE.fullmatch(self.sha256) is None
            or type(self.maximum_bytes) is not int
            or self.maximum_bytes < 1
            or self.maximum_bytes > MAX_RUNTIME_FILE_BYTES
        ):
            fail("transaction runtime file binding differs")


@dataclasses.dataclass(frozen=True, slots=True)
class TransactionRuntimeClosure:
    """Manifest-only runtime paths plus the authorization-sealed source."""

    candidate_manifest: RuntimeFileBinding
    source_root: Path
    source_commit: str
    source_tree: str
    source_seal_sha256: str
    product_root: Path
    tokenizer_root: Path
    worker_binary: Path
    promotion_receipt: Path
    runtime_files: tuple[RuntimeFileBinding, ...]

    def __post_init__(self) -> None:
        for path, label in (
            (self.source_root, "transaction source root"),
            (self.product_root, "candidate product root"),
            (self.tokenizer_root, "candidate tokenizer root"),
            (self.worker_binary, "candidate worker binary"),
            (self.promotion_receipt, "candidate promotion receipt"),
        ):
            _require_canonical_absolute(path, label)
        if (
            GIT_COMMIT_RE.fullmatch(self.source_commit) is None
            or GIT_COMMIT_RE.fullmatch(self.source_tree) is None
            or SHA256_RE.fullmatch(self.source_seal_sha256) is None
            or not self.runtime_files
            or len(self.runtime_files) > MAX_RUNTIME_BINDINGS
            or len({item.path for item in self.runtime_files})
            != len(self.runtime_files)
            or self.candidate_manifest.path
            in {item.path for item in self.runtime_files}
        ):
            fail("transaction runtime closure differs")


@dataclasses.dataclass(frozen=True, slots=True)
class ProductionPreflightSettings:
    """Immutable filesystem and interpreter bindings for production preflight."""

    repo_root: Path
    product_root: Path
    python_executable: Path
    private_runtime_parent: Path
    transaction_runtime: TransactionRuntimeClosure | None = None

    def __post_init__(self) -> None:
        _require_canonical_absolute(self.repo_root, "production repository root")
        _require_canonical_absolute(self.product_root, "production product root")
        _require_canonical_absolute(
            self.python_executable, "production Python executable"
        )
        _require_canonical_absolute(
            self.private_runtime_parent, "production private runtime parent"
        )
        if self.transaction_runtime is not None:
            closure = self.transaction_runtime
            if (
                not isinstance(closure, TransactionRuntimeClosure)
                or self.repo_root != closure.source_root
                or self.product_root != closure.product_root
                or self.python_executable != PRODUCTION_PYTHON_EXECUTABLE
            ):
                fail("transaction production settings differ from their closure")


def production_preflight_settings() -> ProductionPreflightSettings:
    """Return the one fixed production path set for the effective execution user."""

    return ProductionPreflightSettings(
        repo_root=PRODUCTION_REPO_ROOT,
        product_root=PRODUCTION_PRODUCT_ROOT,
        python_executable=PRODUCTION_PYTHON_EXECUTABLE,
        private_runtime_parent=Path("/run/user") / str(os.geteuid()),
    )


def _strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                fail(f"{label} contains duplicate keys")
            result[key] = value
        return result

    if (
        type(raw) is not bytes
        or not raw
        or len(raw) > MAX_CANDIDATE_MANIFEST_BYTES
    ):
        fail(f"{label} exceeds its byte bound")
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda _value: fail(
                f"{label} contains a non-finite number"
            ),
        )
    except ProductionPreflightError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        fail(f"{label} is not strict JSON")
    if type(document) is not dict:
        fail(f"{label} root differs")
    return document


def _strict_manifest_path(
    value: Any,
    *,
    base: Path,
    label: str,
    relative_only: bool = False,
) -> Path:
    if type(value) is not str or not value or "\x00" in value or "//" in value:
        fail(f"{label} path is invalid")
    raw = Path(value)
    if (
        os.path.normpath(value) != value
        or "." in raw.parts
        or ".." in raw.parts
    ):
        fail(f"{label} path must be lexically canonical")
    if raw.is_absolute():
        if relative_only or raw.anchor != "/":
            fail(f"{label} path must be relative")
        selected = raw
    else:
        if raw.anchor or raw.name in {"", ".", ".."}:
            fail(f"{label} relative path is invalid")
        selected = base / raw
    _require_canonical_absolute(selected, label)
    return selected


def _require_runtime_directory(path: Path, label: str) -> None:
    _require_canonical_absolute(path, label)
    try:
        metadata = path.lstat()
    except OSError:
        fail(f"{label} is unavailable")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        fail(f"{label} is not a non-symlink directory")


def _read_bound_runtime_file(
    binding: RuntimeFileBinding,
    label: str,
    *,
    retain: bool = False,
) -> bytes:
    try:
        before = binding.path.lstat()
    except OSError:
        fail(f"{label} is unavailable")
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size < 1
        or before.st_size > binding.maximum_bytes
    ):
        fail(f"{label} identity or size differs")
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    total = 0
    try:
        with binding.path.open("rb") as source:
            opened = os.fstat(source.fileno())
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_nlink,
                opened.st_uid,
                opened.st_gid,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            ) != (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_uid,
                before.st_gid,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ):
                fail(f"{label} changed while opening")
            while True:
                chunk = source.read(COMMAND_READ_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > binding.maximum_bytes:
                    fail(f"{label} exceeded its byte bound")
                digest.update(chunk)
                if retain:
                    chunks.append(chunk)
    except ProductionPreflightError:
        raise
    except OSError:
        fail(f"{label} cannot be read")
    try:
        after = binding.path.lstat()
    except OSError:
        fail(f"{label} changed while being read")
    if (
        after != before
        or total != before.st_size
        or digest.hexdigest() != binding.sha256
    ):
        fail(f"{label} bytes or identity differ")
    return b"".join(chunks)


def _runtime_hash(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} is not a lowercase SHA-256")
    return value


def _source_git(root: Path, arguments: Sequence[str], label: str) -> bytes:
    argv = campaign_source_seal.git_argv(
        ["-C", os.fspath(root), *arguments]
    )
    try:
        result = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=root,
            env=campaign_source_seal.git_environment(),
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        fail(f"{label} failed")
    if (
        result.returncode != 0
        or result.stderr
        or len(result.stdout) > GIT_STATUS_MAX_BYTES
    ):
        fail(f"{label} failed")
    return result.stdout


def _capture_transaction_source(
    source_root: Path,
    *,
    expected_commit: str,
    expected_tree: str,
) -> str:
    _require_canonical_absolute(source_root, "transaction source root")
    if (
        GIT_COMMIT_RE.fullmatch(expected_commit) is None
        or GIT_COMMIT_RE.fullmatch(expected_tree) is None
    ):
        fail("transaction source commit/tree differs")
    try:
        seal = campaign_source_seal.capture_source_seal(
            source_root,
            required_uid=0,
        )
    except campaign_source_seal.SourceSealError:
        fail("transaction execution source is not a root-owned sealed clone")
    expected_lines = (
        (
            ("rev-parse", "--show-toplevel"),
            os.fspath(source_root).encode("utf-8") + b"\n",
            "transaction Git top-level",
        ),
        (
            ("rev-parse", "--verify", "HEAD^{commit}"),
            expected_commit.encode("ascii") + b"\n",
            "transaction Git commit",
        ),
        (
            ("rev-parse", "--verify", "HEAD^{tree}"),
            expected_tree.encode("ascii") + b"\n",
            "transaction Git tree",
        ),
        (
            ("rev-parse", "--abbrev-ref", "HEAD"),
            b"HEAD\n",
            "transaction detached HEAD",
        ),
    )
    for arguments, expected, label in expected_lines:
        if _source_git(source_root, arguments, label) != expected:
            fail(f"{label} differs")
    status = _source_git(
        source_root,
        (
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=all",
            "--no-renames",
        ),
        "transaction Git status",
    )
    if status:
        fail("transaction execution source worktree is not clean")
    try:
        campaign_source_seal.require_source_seal(seal, required_uid=0)
    except campaign_source_seal.SourceSealError:
        fail("transaction execution source changed during validation")
    return seal.fingerprint_sha256


def validate_rocm_python_tcb() -> None:
    """Require the fixed interpreter and versioned AMD SMI import tree."""

    try:
        if (
            ROCM_AMD_SMI_ALIAS.resolve(strict=True) != ROCM_AMD_SMI_SCRIPT
            or PRODUCTION_PYTHON_EXECUTABLE.resolve(strict=True)
            != PRODUCTION_PYTHON_EXECUTABLE
        ):
            fail("ROCm/Python TCB resolution differs")
    except OSError:
        fail("ROCm/Python TCB is unavailable")
    entries = 0
    roots = (
        Path("/opt"),
        ROCM_ROOT,
        ROCM_ROOT / "libexec",
        ROCM_ROOT / "share",
        ROCM_ROOT / "share/amd_smi",
        *ROCM_PYTHON_TCB_TREES,
    )
    for root in roots:
        try:
            metadata = root.lstat()
        except OSError:
            fail("ROCm Python TCB directory is unavailable")
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            fail("ROCm Python TCB directory ownership differs")
    paths = [PRODUCTION_PYTHON_EXECUTABLE]
    for tree in ROCM_PYTHON_TCB_TREES:
        try:
            paths.extend(tree.rglob("*"))
        except OSError:
            fail("ROCm Python TCB cannot be enumerated")
    for path in paths:
        entries += 1
        if entries > MAX_ROCM_TCB_ENTRIES:
            fail("ROCm Python TCB exceeds its entry bound")
        try:
            metadata = path.lstat()
        except OSError:
            fail("ROCm Python TCB entry is unavailable")
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not (
                stat.S_ISDIR(metadata.st_mode)
                or stat.S_ISREG(metadata.st_mode)
            )
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            fail("ROCm Python TCB entry ownership differs")
    if ROCM_AMD_SMI_SCRIPT not in paths:
        fail("resolved AMD SMI script is outside its root-owned TCB")


def transaction_preflight_settings(
    *,
    source_root: Path,
    source_commit: str,
    source_tree: str,
    candidate_manifest_path: Path,
    candidate_manifest_raw: bytes,
    candidate_manifest_sha256: str,
    expected_worker_binary_sha256: str,
) -> ProductionPreflightSettings:
    """Derive every mutable SQ8 runtime root from the claimed manifest bytes."""

    _require_canonical_absolute(candidate_manifest_path, "candidate manifest")
    manifest_binding = RuntimeFileBinding(
        candidate_manifest_path,
        _runtime_hash(candidate_manifest_sha256, "candidate manifest SHA-256"),
        MAX_CANDIDATE_MANIFEST_BYTES,
    )
    if hashlib.sha256(candidate_manifest_raw).hexdigest() != manifest_binding.sha256:
        fail("candidate manifest bytes differ from their transaction SHA-256")
    if _read_bound_runtime_file(
        manifest_binding,
        "candidate served-model manifest",
        retain=True,
    ) != candidate_manifest_raw:
        fail("candidate manifest changed after transaction binding")
    document = _strict_json_object(
        candidate_manifest_raw, "candidate served-model manifest"
    )
    public = document.get("public")
    format_value = document.get("format")
    worker = document.get("worker")
    product = document.get("product")
    tokenizer = document.get("tokenizer")
    promotion = document.get("promotion")
    if (
        document.get("schema_version") != "ullm.served_model.v2"
        or type(public) is not dict
        or public.get("id") != "ullm-qwen3-14b-sq8"
        or type(format_value) is not dict
        or format_value.get("format_id") != "SQ8_0"
        or type(worker) is not dict
        or worker.get("protocol") != "ullm.worker.v2"
        or type(product) is not dict
        or type(tokenizer) is not dict
        or type(promotion) is not dict
        or promotion.get("source_commit") != source_commit
    ):
        fail("candidate SQ8_0 v2 runtime contract differs")
    assert isinstance(worker, dict)
    assert isinstance(product, dict)
    assert isinstance(tokenizer, dict)
    assert isinstance(promotion, dict)
    base = candidate_manifest_path.parent
    product_root = _strict_manifest_path(
        product.get("root"), base=base, label="candidate product root"
    )
    tokenizer_root = _strict_manifest_path(
        tokenizer.get("root"), base=base, label="candidate tokenizer root"
    )
    worker_binary = _strict_manifest_path(
        worker.get("binary"), base=base, label="candidate worker binary"
    )
    promotion_receipt = _strict_manifest_path(
        promotion.get("receipt"),
        base=base,
        label="candidate promotion receipt",
    )
    _require_runtime_directory(product_root, "candidate product root")
    _require_runtime_directory(tokenizer_root, "candidate tokenizer root")
    worker_sha256 = _runtime_hash(
        worker.get("binary_sha256"), "candidate worker binary SHA-256"
    )
    if worker_sha256 != expected_worker_binary_sha256:
        fail("CLI worker SHA-256 differs from the candidate manifest")
    bindings = [
        RuntimeFileBinding(worker_binary, worker_sha256, MAX_RUNTIME_FILE_BYTES),
        RuntimeFileBinding(
            promotion_receipt,
            _runtime_hash(
                promotion.get("receipt_sha256"),
                "candidate promotion receipt SHA-256",
            ),
            16 << 20,
        ),
    ]
    package = product.get("package")
    artifact = product.get("artifact")
    if type(package) is not dict or (
        artifact is not None and type(artifact) is not dict
    ):
        fail("candidate product manifest closure differs")
    assert isinstance(package, dict)
    bindings.append(
        RuntimeFileBinding(
            _strict_manifest_path(
                package.get("manifest_path"),
                base=product_root,
                label="candidate package manifest",
                relative_only=True,
            ),
            _runtime_hash(
                package.get("manifest_sha256"),
                "candidate package manifest SHA-256",
            ),
            16 << 20,
        )
    )
    if isinstance(artifact, dict):
        bindings.append(
            RuntimeFileBinding(
                _strict_manifest_path(
                    artifact.get("manifest_path"),
                    base=product_root,
                    label="candidate artifact manifest",
                    relative_only=True,
                ),
                _runtime_hash(
                    artifact.get("manifest_sha256"),
                    "candidate artifact manifest SHA-256",
                ),
                16 << 20,
            )
        )
    tokenizer_files = tokenizer.get("files")
    if (
        type(tokenizer_files) is not dict
        or not tokenizer_files
        or len(tokenizer_files) > 128
        or any(type(relative) is not str for relative in tokenizer_files)
    ):
        fail("candidate tokenizer file closure differs")
    for relative, expected_sha256 in sorted(
        tokenizer_files.items(), key=lambda item: os.fsencode(item[0])
    ):
        bindings.append(
            RuntimeFileBinding(
                _strict_manifest_path(
                    relative,
                    base=tokenizer_root,
                    label="candidate tokenizer file",
                    relative_only=True,
                ),
                _runtime_hash(
                    expected_sha256, "candidate tokenizer file SHA-256"
                ),
                MAX_RUNTIME_FILE_BYTES,
            )
        )
    if len(bindings) > MAX_RUNTIME_BINDINGS or len(
        {binding.path for binding in bindings}
    ) != len(bindings):
        fail("candidate runtime file paths are not distinct")
    for index, binding in enumerate(bindings):
        _read_bound_runtime_file(binding, f"candidate runtime file {index}")
    source_fingerprint = _capture_transaction_source(
        source_root,
        expected_commit=source_commit,
        expected_tree=source_tree,
    )
    validate_rocm_python_tcb()
    closure = TransactionRuntimeClosure(
        manifest_binding,
        source_root,
        source_commit,
        source_tree,
        source_fingerprint,
        product_root,
        tokenizer_root,
        worker_binary,
        promotion_receipt,
        tuple(bindings),
    )
    return ProductionPreflightSettings(
        repo_root=source_root,
        product_root=product_root,
        python_executable=PRODUCTION_PYTHON_EXECUTABLE,
        private_runtime_parent=Path("/run/user") / str(os.geteuid()),
        transaction_runtime=closure,
    )


def revalidate_transaction_settings(settings: ProductionPreflightSettings) -> None:
    """Re-pin the source and every manifest-declared runtime file."""

    if (
        not isinstance(settings, ProductionPreflightSettings)
        or settings.transaction_runtime is None
    ):
        fail("transaction production settings are unavailable")
    closure = settings.transaction_runtime
    fingerprint = _capture_transaction_source(
        closure.source_root,
        expected_commit=closure.source_commit,
        expected_tree=closure.source_tree,
    )
    if fingerprint != closure.source_seal_sha256:
        fail("transaction execution source seal changed")
    validate_rocm_python_tcb()
    _read_bound_runtime_file(
        closure.candidate_manifest, "candidate served-model manifest"
    )
    for index, binding in enumerate(closure.runtime_files):
        _read_bound_runtime_file(binding, f"candidate runtime file {index}")


def canonical_campaign_lock_path() -> Path:
    """Return the non-overridable host-wide full-campaign lock path."""

    return production_preflight_settings().private_runtime_parent / PRODUCTION_LOCK_NAME


@dataclasses.dataclass(frozen=True, slots=True)
class BoundedCommandResult:
    stdout: bytes
    stderr: bytes
    returncode: int


class CommandRunner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        stdout_limit: int,
        stderr_limit: int,
    ) -> BoundedCommandResult: ...


def _kill_and_wait(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        process.wait(timeout=1.0)
    except (OSError, subprocess.SubprocessError):
        pass


class BoundedCommandRunner:
    """Execute a fixed argv while bounding each output stream during capture."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        stdout_limit: int,
        stderr_limit: int,
    ) -> BoundedCommandResult:
        if (
            not argv
            or any(type(item) is not str or not item or "\x00" in item for item in argv)
            or not isinstance(cwd, Path)
            or not cwd.is_absolute()
            or type(timeout_seconds) not in {int, float}
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or type(stdout_limit) is not int
            or stdout_limit < 1
            or type(stderr_limit) is not int
            or stderr_limit < 1
        ):
            fail("bounded command binding differs")

        process: subprocess.Popen[bytes] | None = None
        selector = selectors.DefaultSelector()
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=os.fspath(cwd),
                env={
                    **campaign_source_seal.git_environment(),
                    "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
                    "LC_ALL": "C",
                    "LANG": "C",
                    "GIT_OPTIONAL_LOCKS": "0",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONSAFEPATH": "1",
                },
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
                start_new_session=True,
            )
            assert process.stdout is not None
            assert process.stderr is not None
            stdout_fd = process.stdout.fileno()
            stderr_fd = process.stderr.fileno()
            selector.register(stdout_fd, selectors.EVENT_READ, "stdout")
            selector.register(stderr_fd, selectors.EVENT_READ, "stderr")
            chunks: dict[str, list[bytes]] = {"stdout": [], "stderr": []}
            totals = {"stdout": 0, "stderr": 0}
            limits = {"stdout": stdout_limit, "stderr": stderr_limit}
            deadline = time.monotonic() + timeout_seconds

            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _kill_and_wait(process)
                    fail("bounded command timed out")
                events = selector.select(min(remaining, 0.25))
                if not events:
                    continue
                for key, _mask in events:
                    stream = cast(str, key.data)
                    try:
                        chunk = os.read(key.fd, COMMAND_READ_CHUNK_BYTES)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fd)
                        continue
                    totals[stream] += len(chunk)
                    if totals[stream] > limits[stream]:
                        _kill_and_wait(process)
                        fail(f"bounded command {stream} exceeded its byte limit")
                    chunks[stream].append(chunk)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_and_wait(process)
                fail("bounded command timed out")
            try:
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                _kill_and_wait(process)
                fail("bounded command timed out")
            return BoundedCommandResult(
                stdout=b"".join(chunks["stdout"]),
                stderr=b"".join(chunks["stderr"]),
                returncode=returncode,
            )
        except ProductionPreflightError:
            raise
        except (OSError, subprocess.SubprocessError):
            if process is not None:
                _kill_and_wait(process)
            fail("failed to execute a bounded command")
        except BaseException:
            if process is not None:
                _kill_and_wait(process)
            raise
        finally:
            selector.close()
            if process is not None and process.poll() is None:
                _kill_and_wait(process)
            if process is not None:
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()


SYSTEM_COMMAND_RUNNER = BoundedCommandRunner()


@dataclasses.dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    mode: int
    links: int
    uid: int
    gid: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _FileIdentity:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            mode=value.st_mode,
            links=value.st_nlink,
            uid=value.st_uid,
            gid=value.st_gid,
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
            ctime_ns=value.st_ctime_ns,
        )


def _same_object(left: _FileIdentity, right: _FileIdentity) -> bool:
    return (
        left.device,
        left.inode,
        left.mode,
        left.uid,
        left.gid,
    ) == (
        right.device,
        right.inode,
        right.mode,
        right.uid,
        right.gid,
    )


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for production preflight")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _open_stable_directory(
    path: Path,
    label: str,
    *,
    private: bool = False,
) -> tuple[int, _FileIdentity]:
    _require_canonical_absolute(path, label)
    descriptor = -1
    try:
        if path.resolve(strict=True) != path:
            fail(f"{label} contains a symbolic link")
        descriptor = os.open(path, _directory_flags())
        identity = _FileIdentity.from_stat(os.fstat(descriptor))
        entry = _FileIdentity.from_stat(path.lstat())
        if (
            identity != entry
            or not stat.S_ISDIR(identity.mode)
            or stat.S_ISLNK(identity.mode)
            or identity.links < 1
        ):
            fail(f"{label} directory identity differs")
        if private and (
            stat.S_IMODE(identity.mode) != 0o700
            or identity.uid != os.geteuid()
            or identity.gid != os.getegid()
        ):
            fail(f"{label} owner or mode differs")
        return descriptor, identity
    except ProductionPreflightError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError:
        if descriptor >= 0:
            os.close(descriptor)
        fail(f"{label} is unavailable without following links")


def _verify_open_directory(
    descriptor: int,
    path: Path,
    expected: _FileIdentity,
    label: str,
) -> None:
    try:
        current = _FileIdentity.from_stat(os.fstat(descriptor))
        entry = _FileIdentity.from_stat(path.lstat())
    except OSError:
        fail(f"{label} became unavailable")
    if current != expected or entry != expected:
        fail(f"{label} changed")


def _require_clean_command(
    result: BoundedCommandResult,
    label: str,
) -> bytes:
    if result.stderr:
        fail(f"{label} wrote to stderr")
    if result.returncode != 0:
        fail(f"{label} failed")
    return result.stdout


def _capture_git_state(
    settings: ProductionPreflightSettings,
    expected_commit: str,
    runner: CommandRunner,
) -> tuple[str, bytes, _FileIdentity]:
    if GIT_COMMIT_RE.fullmatch(expected_commit) is None:
        fail("expected Git commit must be exactly 40 lowercase hexadecimal digits")
    repo_fd, repo_identity = _open_stable_directory(
        settings.repo_root, "production repository root"
    )
    git_prefix = (
        *campaign_source_seal.GIT_COMMAND_PREFIX,
        "-C",
        os.fspath(settings.repo_root),
    )
    try:
        head_before = _require_clean_command(
            runner.run(
                (*git_prefix, "rev-parse", "--verify", "HEAD^{commit}"),
                cwd=settings.repo_root,
                timeout_seconds=GIT_TIMEOUT_SECONDS,
                stdout_limit=GIT_HEAD_MAX_BYTES,
                stderr_limit=GIT_HEAD_MAX_BYTES,
            ),
            "Git HEAD capture",
        )
        status = _require_clean_command(
            runner.run(
                (
                    *git_prefix,
                    "status",
                    "--porcelain=v1",
                    "-z",
                    "--untracked-files=all",
                    "--ignore-submodules=none",
                ),
                cwd=settings.repo_root,
                timeout_seconds=GIT_TIMEOUT_SECONDS,
                stdout_limit=GIT_STATUS_MAX_BYTES,
                stderr_limit=GIT_HEAD_MAX_BYTES,
            ),
            "Git status capture",
        )
        head_after = _require_clean_command(
            runner.run(
                (*git_prefix, "rev-parse", "--verify", "HEAD^{commit}"),
                cwd=settings.repo_root,
                timeout_seconds=GIT_TIMEOUT_SECONDS,
                stdout_limit=GIT_HEAD_MAX_BYTES,
                stderr_limit=GIT_HEAD_MAX_BYTES,
            ),
            "Git HEAD recapture",
        )
        _verify_open_directory(
            repo_fd,
            settings.repo_root,
            repo_identity,
            "production repository root",
        )
    finally:
        os.close(repo_fd)
    expected_raw = expected_commit.encode("ascii") + b"\n"
    if head_before != expected_raw or head_after != expected_raw:
        fail("Git HEAD differs from the explicit expected commit")
    return expected_commit, status, repo_identity


@dataclasses.dataclass(frozen=True, slots=True)
class GitAnchor:
    """An exact HEAD and porcelain-v1 status byte anchor for one repository."""

    settings: ProductionPreflightSettings
    commit: str
    status_raw: bytes
    _repo_identity: _FileIdentity = dataclasses.field(repr=False)

    @classmethod
    def capture(
        cls,
        settings: ProductionPreflightSettings,
        *,
        expected_commit: str,
        runner: CommandRunner = SYSTEM_COMMAND_RUNNER,
    ) -> GitAnchor:
        if not isinstance(settings, ProductionPreflightSettings):
            fail("production preflight settings type differs")
        commit, status, repo_identity = _capture_git_state(
            settings, expected_commit, runner
        )
        return cls(settings, commit, status, repo_identity)

    def revalidate(
        self,
        *,
        runner: CommandRunner = SYSTEM_COMMAND_RUNNER,
    ) -> None:
        commit, status, repo_identity = _capture_git_state(
            self.settings, self.commit, runner
        )
        if (
            commit != self.commit
            or status != self.status_raw
            or repo_identity != self._repo_identity
        ):
            fail("Git anchor drifted from its exact capture")


@dataclasses.dataclass(frozen=True, slots=True)
class _SnapshotFile:
    relative_path: str
    identity: _FileIdentity
    sha256: str


def _write_all(descriptor: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        written = os.write(descriptor, raw[offset:])
        if written <= 0:
            fail("HEAD tool snapshot write made no progress")
        offset += written


def _write_snapshot_file(
    directory_fd: int,
    name: str,
    raw: bytes,
) -> _SnapshotFile:
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, raw)
        os.fsync(descriptor)
        identity = _FileIdentity.from_stat(os.fstat(descriptor))
        entry = _FileIdentity.from_stat(
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        )
        if (
            identity != entry
            or not stat.S_ISREG(identity.mode)
            or stat.S_IMODE(identity.mode) != 0o600
            or identity.links != 1
            or identity.uid != os.geteuid()
            or identity.gid != os.getegid()
            or identity.size != len(raw)
        ):
            fail("HEAD tool snapshot file identity differs")
        return _SnapshotFile(
            relative_path=f"tools/{name}",
            identity=identity,
            sha256=hashlib.sha256(raw).hexdigest(),
        )
    except ProductionPreflightError:
        raise
    except OSError:
        fail("failed to materialize a private HEAD tool snapshot")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_head_blob(
    settings: ProductionPreflightSettings,
    anchor: GitAnchor,
    relative_path: str,
    runner: CommandRunner,
    *,
    stdout_limit: int = HEAD_TOOL_MAX_BYTES,
) -> bytes:
    result = runner.run(
        (
            *campaign_source_seal.GIT_COMMAND_PREFIX,
            "-C",
            os.fspath(settings.repo_root),
            "cat-file",
            "blob",
            f"{anchor.commit}:{relative_path}",
        ),
        cwd=settings.repo_root,
        timeout_seconds=GIT_TIMEOUT_SECONDS,
        stdout_limit=stdout_limit,
        stderr_limit=GIT_HEAD_MAX_BYTES,
    )
    raw = _require_clean_command(result, f"HEAD blob capture for {relative_path}")
    if not raw:
        fail(f"HEAD blob for {relative_path} is empty")
    return raw


def read_pinned_http_client_source(
    settings: ProductionPreflightSettings,
    anchor: GitAnchor,
    *,
    expected_sha256: str,
    runner: CommandRunner = SYSTEM_COMMAND_RUNNER,
) -> bytes:
    """Read the fixed HTTP client blob from anchored HEAD with an exact digest."""

    if (
        not isinstance(settings, ProductionPreflightSettings)
        or not isinstance(anchor, GitAnchor)
        or anchor.settings != settings
        or type(expected_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        fail("pinned HTTP client source binding differs")

    anchor.revalidate(runner=runner)
    raw = _read_head_blob(
        settings,
        anchor,
        HEAD_HTTP_CLIENT_PATH,
        runner,
        stdout_limit=HEAD_HTTP_CLIENT_MAX_BYTES,
    )
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        fail("anchored HEAD HTTP client source SHA-256 differs")
    anchor.revalidate(runner=runner)
    return raw


def _random_snapshot_name() -> str:
    return "ullm-sq8-promotion-head-" + secrets.token_hex(16)


class HeadPromotionToolSnapshotOwner:
    """Own the exact promotion verifier and canonical helper from anchored HEAD."""

    def __init__(
        self,
        settings: ProductionPreflightSettings,
        anchor: GitAnchor,
        root: Path,
        parent_fd: int,
        root_fd: int,
        tools_fd: int,
        parent_identity: _FileIdentity,
        root_identity: _FileIdentity,
        tools_identity: _FileIdentity,
        files: tuple[_SnapshotFile, ...],
    ) -> None:
        self._settings = settings
        self._anchor = anchor
        self._root = root
        self._parent_fd = parent_fd
        self._root_fd = root_fd
        self._tools_fd = tools_fd
        self._parent_identity = parent_identity
        self._root_identity = root_identity
        self._tools_identity = tools_identity
        self._files = files
        self.closed = False

    @property
    def settings(self) -> ProductionPreflightSettings:
        return self._settings

    @property
    def anchor(self) -> GitAnchor:
        return self._anchor

    @property
    def root(self) -> Path:
        return self._root

    @property
    def validator_path(self) -> Path:
        return self._root / HEAD_PROMOTION_TOOL_PATHS[0]

    @property
    def canonical_path(self) -> Path:
        return self._root / HEAD_PROMOTION_TOOL_PATHS[1]

    @classmethod
    def create(
        cls,
        settings: ProductionPreflightSettings,
        anchor: GitAnchor,
        *,
        runner: CommandRunner = SYSTEM_COMMAND_RUNNER,
    ) -> HeadPromotionToolSnapshotOwner:
        if (
            not isinstance(settings, ProductionPreflightSettings)
            or not isinstance(anchor, GitAnchor)
            or anchor.settings != settings
        ):
            fail("HEAD tool snapshot binding differs")
        anchor.revalidate(runner=runner)
        parent_fd, parent_identity = _open_stable_directory(
            settings.private_runtime_parent,
            "production private runtime parent",
            private=True,
        )
        name = _random_snapshot_name()
        root_fd = -1
        tools_fd = -1
        root_identity: _FileIdentity | None = None
        tools_identity: _FileIdentity | None = None
        created_root = False
        created_tools = False
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            created_root = True
            root_fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
            root_identity = _FileIdentity.from_stat(os.fstat(root_fd))
            root_entry = _FileIdentity.from_stat(
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            )
            if (
                root_identity != root_entry
                or stat.S_IMODE(root_identity.mode) != 0o700
                or root_identity.uid != os.geteuid()
                or root_identity.gid != os.getegid()
            ):
                fail("HEAD tool snapshot root identity differs")
            os.mkdir("tools", 0o700, dir_fd=root_fd)
            created_tools = True
            tools_fd = os.open("tools", _directory_flags(), dir_fd=root_fd)
            tools_identity = _FileIdentity.from_stat(os.fstat(tools_fd))
            tools_entry = _FileIdentity.from_stat(
                os.stat("tools", dir_fd=root_fd, follow_symlinks=False)
            )
            if (
                tools_identity != tools_entry
                or stat.S_IMODE(tools_identity.mode) != 0o700
                or tools_identity.uid != os.geteuid()
                or tools_identity.gid != os.getegid()
            ):
                fail("HEAD tool snapshot tools directory identity differs")

            files: list[_SnapshotFile] = []
            for relative_path in HEAD_PROMOTION_TOOL_PATHS:
                raw = _read_head_blob(settings, anchor, relative_path, runner)
                files.append(
                    _write_snapshot_file(tools_fd, Path(relative_path).name, raw)
                )
            os.fsync(tools_fd)
            os.fsync(root_fd)
            os.fsync(parent_fd)
            parent_identity = _FileIdentity.from_stat(os.fstat(parent_fd))
            root_identity = _FileIdentity.from_stat(os.fstat(root_fd))
            tools_identity = _FileIdentity.from_stat(os.fstat(tools_fd))
            anchor.revalidate(runner=runner)
            assert root_identity is not None
            assert tools_identity is not None
            owner = cls(
                settings,
                anchor,
                settings.private_runtime_parent / name,
                parent_fd,
                root_fd,
                tools_fd,
                parent_identity,
                root_identity,
                tools_identity,
                tuple(files),
            )
            owner.revalidate(runner=runner)
            return owner
        except BaseException as error:
            cleanup_failed = False
            if tools_fd >= 0:
                for relative_path in HEAD_PROMOTION_TOOL_PATHS:
                    try:
                        os.unlink(Path(relative_path).name, dir_fd=tools_fd)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        cleanup_failed = True
                try:
                    os.close(tools_fd)
                except OSError:
                    cleanup_failed = True
            if created_tools and root_fd >= 0:
                try:
                    os.rmdir("tools", dir_fd=root_fd)
                except OSError:
                    cleanup_failed = True
            if root_fd >= 0:
                try:
                    os.close(root_fd)
                except OSError:
                    cleanup_failed = True
            if created_root:
                try:
                    os.rmdir(name, dir_fd=parent_fd)
                except OSError:
                    cleanup_failed = True
            try:
                os.close(parent_fd)
            except OSError:
                cleanup_failed = True
            if cleanup_failed:
                error.add_note("private HEAD tool snapshot cleanup also failed")
            raise

    def __enter__(self) -> HeadPromotionToolSnapshotOwner:
        if self.closed:
            fail("HEAD tool snapshot owner is already closed")
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if error is None:
            self.close()
        else:
            try:
                self.close()
            except BaseException:
                error.add_note("private HEAD tool snapshot cleanup also failed")

    def _verify_file(self, expected: _SnapshotFile) -> None:
        name = Path(expected.relative_path).name
        descriptor = -1
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=self._tools_fd,
            )
            before = _FileIdentity.from_stat(os.fstat(descriptor))
            entry = _FileIdentity.from_stat(
                os.stat(name, dir_fd=self._tools_fd, follow_symlinks=False)
            )
            digest = hashlib.sha256()
            total = 0
            while chunk := os.read(descriptor, COMMAND_READ_CHUNK_BYTES):
                total += len(chunk)
                if total > HEAD_TOOL_MAX_BYTES:
                    fail("private HEAD tool snapshot exceeds its byte bound")
                digest.update(chunk)
            after = _FileIdentity.from_stat(os.fstat(descriptor))
            entry_after = _FileIdentity.from_stat(
                os.stat(name, dir_fd=self._tools_fd, follow_symlinks=False)
            )
            if (
                before != expected.identity
                or entry != expected.identity
                or after != expected.identity
                or entry_after != expected.identity
                or total != expected.identity.size
                or digest.hexdigest() != expected.sha256
            ):
                fail("private HEAD tool snapshot changed")
        except ProductionPreflightError:
            raise
        except OSError:
            fail("private HEAD tool snapshot is unavailable")
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def revalidate(
        self,
        *,
        runner: CommandRunner = SYSTEM_COMMAND_RUNNER,
    ) -> None:
        if self.closed:
            fail("HEAD tool snapshot owner is already closed")
        self.anchor.revalidate(runner=runner)
        try:
            parent_current = _FileIdentity.from_stat(os.fstat(self._parent_fd))
            parent_entry = _FileIdentity.from_stat(
                self.settings.private_runtime_parent.lstat()
            )
            root_entries = set(os.listdir(self._root_fd))
            tool_entries = set(os.listdir(self._tools_fd))
            root_current = _FileIdentity.from_stat(os.fstat(self._root_fd))
            root_entry = _FileIdentity.from_stat(
                os.stat(
                    self.root.name,
                    dir_fd=self._parent_fd,
                    follow_symlinks=False,
                )
            )
            tools_current = _FileIdentity.from_stat(os.fstat(self._tools_fd))
            tools_entry = _FileIdentity.from_stat(
                os.stat("tools", dir_fd=self._root_fd, follow_symlinks=False)
            )
        except OSError:
            fail("private HEAD tool snapshot directory is unavailable")
        if (
            not _same_object(parent_current, self._parent_identity)
            or not _same_object(parent_entry, self._parent_identity)
            or root_entries != {"tools"}
            or tool_entries != {Path(value).name for value in HEAD_PROMOTION_TOOL_PATHS}
            or root_current != self._root_identity
            or root_entry != self._root_identity
            or tools_current != self._tools_identity
            or tools_entry != self._tools_identity
        ):
            fail("private HEAD tool snapshot changed")
        for expected in self._files:
            self._verify_file(expected)

    def close(self) -> None:
        if self.closed:
            return
        tampered = False
        try:
            self.revalidate()
        except BaseException:
            tampered = True
        cleanup_failed = False
        for expected in self._files:
            try:
                os.unlink(Path(expected.relative_path).name, dir_fd=self._tools_fd)
            except OSError:
                cleanup_failed = True
        try:
            os.fsync(self._tools_fd)
        except OSError:
            cleanup_failed = True
        try:
            os.close(self._tools_fd)
        except OSError:
            cleanup_failed = True
        finally:
            self._tools_fd = -1
        try:
            tools_entry = _FileIdentity.from_stat(
                os.stat("tools", dir_fd=self._root_fd, follow_symlinks=False)
            )
            if _same_object(tools_entry, self._tools_identity):
                os.rmdir("tools", dir_fd=self._root_fd)
            else:
                tampered = True
        except OSError:
            cleanup_failed = True
        try:
            os.fsync(self._root_fd)
        except OSError:
            cleanup_failed = True
        try:
            os.close(self._root_fd)
        except OSError:
            cleanup_failed = True
        finally:
            self._root_fd = -1
        try:
            root_entry = _FileIdentity.from_stat(
                os.stat(
                    self.root.name,
                    dir_fd=self._parent_fd,
                    follow_symlinks=False,
                )
            )
            if _same_object(root_entry, self._root_identity):
                os.rmdir(self.root.name, dir_fd=self._parent_fd)
            else:
                tampered = True
        except OSError:
            cleanup_failed = True
        try:
            os.fsync(self._parent_fd)
        except OSError:
            cleanup_failed = True
        try:
            os.close(self._parent_fd)
        except OSError:
            cleanup_failed = True
        finally:
            self._parent_fd = -1
            self.closed = True
        if tampered:
            fail("private HEAD tool snapshot changed before cleanup")
        if cleanup_failed:
            fail("failed to remove the private HEAD tool snapshot")


class SealedSourcePromotionTools:
    """Use promotion tools in the root-owned transaction source directly."""

    def __init__(
        self,
        settings: ProductionPreflightSettings,
        anchor: GitAnchor,
    ) -> None:
        self.settings = settings
        self.anchor = anchor
        self.root = settings.repo_root
        self.validator_path = self.root / HEAD_PROMOTION_TOOL_PATHS[0]
        self.canonical_path = self.root / HEAD_PROMOTION_TOOL_PATHS[1]
        self.closed = False

    @classmethod
    def create(
        cls,
        settings: ProductionPreflightSettings,
        anchor: GitAnchor,
    ) -> SealedSourcePromotionTools:
        if (
            not isinstance(settings, ProductionPreflightSettings)
            or settings.transaction_runtime is None
            or not isinstance(anchor, GitAnchor)
            or anchor.settings != settings
        ):
            fail("sealed source promotion tool binding differs")
        owner = cls(settings, anchor)
        owner.revalidate()
        return owner

    def revalidate(
        self,
        *,
        runner: CommandRunner = SYSTEM_COMMAND_RUNNER,
    ) -> None:
        if self.closed:
            fail("sealed source promotion tools are closed")
        self.anchor.revalidate(runner=runner)
        revalidate_transaction_settings(self.settings)
        expected = {
            self.settings.repo_root / relative
            for relative in HEAD_PROMOTION_TOOL_PATHS
        }
        if {self.validator_path, self.canonical_path} != expected:
            fail("sealed source promotion tool paths differ")
        for path in expected:
            try:
                metadata = path.lstat()
            except OSError:
                fail("sealed source promotion tool is unavailable")
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                fail("sealed source promotion tool identity differs")

    def close(self) -> None:
        if self.closed:
            return
        self.revalidate()
        self.closed = True


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("promotion receipt contains a duplicate JSON key")
        result[key] = value
    return result


def _reject_nonfinite_constant(_value: str) -> None:
    fail("promotion receipt contains a non-finite JSON number")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        fail("promotion receipt contains a non-finite JSON number")
    return parsed


def _parse_promotion_receipt(raw: bytes, product_root: Path) -> dict[str, Any]:
    if not raw or len(raw) > PROMOTION_STDOUT_MAX_BYTES:
        fail("promotion receipt size differs")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_parse_finite_float,
        )
    except ProductionPreflightError:
        raise
    except (UnicodeError, ValueError, RecursionError):
        fail("promotion receipt is not strict JSON")
    expected_keys = {
        "schema_version",
        "product_root",
        "created_at",
        "model_revision",
        "artifact",
        "package",
        "read_only",
        "full_payloads",
        "verified",
    }
    if type(value) is not dict or set(value) != expected_keys:
        fail("promotion receipt fields differ")
    receipt = cast(dict[str, Any], value)
    if receipt["schema_version"] != PROMOTION_SCHEMA:
        fail("promotion receipt schema differs")
    if receipt["product_root"] != os.fspath(product_root):
        fail("promotion receipt product root differs")
    for flag in ("full_payloads", "read_only", "verified"):
        if receipt[flag] is not True:
            fail(f"promotion receipt {flag} flag is not true")
    for section in ("artifact", "package"):
        item = receipt[section]
        if type(item) is not dict or item.get("payloads_hashed") is not True:
            fail(f"promotion receipt {section} payload hashing flag is not true")
    return receipt


def run_pinned_full_promotion_validation(
    settings: ProductionPreflightSettings,
    anchor: GitAnchor,
    tools: HeadPromotionToolSnapshotOwner | SealedSourcePromotionTools,
    *,
    runner: CommandRunner = SYSTEM_COMMAND_RUNNER,
) -> dict[str, Any]:
    """Run full validation from legacy snapshots or the sealed v2 source."""

    if (
        not isinstance(settings, ProductionPreflightSettings)
        or not isinstance(anchor, GitAnchor)
        or not isinstance(
            tools,
            (HeadPromotionToolSnapshotOwner, SealedSourcePromotionTools),
        )
        or anchor.settings != settings
        or tools.settings != settings
        or tools.anchor != anchor
        or tools.closed
        or (
            isinstance(tools, SealedSourcePromotionTools)
            and settings.transaction_runtime is None
        )
        or (
            isinstance(tools, HeadPromotionToolSnapshotOwner)
            and settings.transaction_runtime is not None
        )
    ):
        fail("pinned promotion validation binding differs")
    tools.revalidate(runner=runner)
    product_fd, product_identity = _open_stable_directory(
        settings.product_root, "production product root"
    )

    result: BoundedCommandResult | None = None
    primary_error: BaseException | None = None
    try:
        result = runner.run(
            (
                *PRODUCTION_PYTHON_PREFIX,
                "-c",
                SEALED_TOOL_LAUNCH_SOURCE,
                os.fspath(tools.validator_path.parent),
                os.fspath(tools.validator_path),
                os.fspath(settings.product_root),
            ),
            cwd=tools.root,
            timeout_seconds=PROMOTION_TIMEOUT_SECONDS,
            stdout_limit=PROMOTION_STDOUT_MAX_BYTES,
            stderr_limit=PROMOTION_STDERR_MAX_BYTES,
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            _verify_open_directory(
                product_fd,
                settings.product_root,
                product_identity,
                "production product root",
            )
            tools.revalidate(runner=runner)
        except BaseException:
            if primary_error is None:
                raise
            primary_error.add_note(
                "post-validation source or product revalidation also failed"
            )
        finally:
            os.close(product_fd)

    assert result is not None
    stdout = _require_clean_command(result, "full product promotion validation")
    return _parse_promotion_receipt(stdout, settings.product_root)


__all__ = [
    "BoundedCommandResult",
    "BoundedCommandRunner",
    "CommandRunner",
    "GitAnchor",
    "HEAD_HTTP_CLIENT_PATH",
    "HEAD_PROMOTION_TOOL_PATHS",
    "HeadPromotionToolSnapshotOwner",
    "RuntimeFileBinding",
    "SealedSourcePromotionTools",
    "ProductionPreflightError",
    "ProductionPreflightSettings",
    "TransactionRuntimeClosure",
    "canonical_campaign_lock_path",
    "production_preflight_settings",
    "read_pinned_http_client_source",
    "revalidate_transaction_settings",
    "run_pinned_full_promotion_validation",
    "transaction_preflight_settings",
    "validate_rocm_python_tcb",
]
