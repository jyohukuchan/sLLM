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
import time
from pathlib import Path
from types import TracebackType
from typing import Any, NoReturn, Protocol, Sequence, cast


PRODUCTION_REPO_ROOT = Path("/home/homelab1/coding-local/ultimateLLM/uLLM-project")
PRODUCTION_PRODUCT_ROOT = Path(
    "/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1"
)
PRODUCTION_PYTHON_EXECUTABLE = Path("/usr/bin/python3")
PRODUCTION_LOCK_NAME = "ullm-sq8-full-openwebui-campaign.lock"

PROMOTION_SCHEMA = "ullm.sq8_product_promotion.v1"
HEAD_PROMOTION_TOOL_PATHS = (
    "tools/validate-sq8-product-promotion.py",
    "tools/sq8_canonical_artifact.py",
)

GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
GIT_TIMEOUT_SECONDS = 10.0
GIT_HEAD_MAX_BYTES = 128
GIT_STATUS_MAX_BYTES = 4 << 20
HEAD_TOOL_MAX_BYTES = 32 << 20
PROMOTION_TIMEOUT_SECONDS = 6 * 60 * 60.0
PROMOTION_STDOUT_MAX_BYTES = 2 << 20
PROMOTION_STDERR_MAX_BYTES = 64 << 10
COMMAND_READ_CHUNK_BYTES = 64 << 10


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
class ProductionPreflightSettings:
    """Immutable filesystem and interpreter bindings for production preflight."""

    repo_root: Path
    product_root: Path
    python_executable: Path
    private_runtime_parent: Path

    def __post_init__(self) -> None:
        _require_canonical_absolute(self.repo_root, "production repository root")
        _require_canonical_absolute(self.product_root, "production product root")
        _require_canonical_absolute(
            self.python_executable, "production Python executable"
        )
        _require_canonical_absolute(
            self.private_runtime_parent, "production private runtime parent"
        )


def production_preflight_settings() -> ProductionPreflightSettings:
    """Return the one fixed production path set for the effective execution user."""

    return ProductionPreflightSettings(
        repo_root=PRODUCTION_REPO_ROOT,
        product_root=PRODUCTION_PRODUCT_ROOT,
        python_executable=PRODUCTION_PYTHON_EXECUTABLE,
        private_runtime_parent=Path("/run/user") / str(os.geteuid()),
    )


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
                    "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
                    "LC_ALL": "C",
                    "LANG": "C",
                    "GIT_OPTIONAL_LOCKS": "0",
                    "PYTHONDONTWRITEBYTECODE": "1",
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
    git_prefix = ("/usr/bin/git", "-C", os.fspath(settings.repo_root))
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
) -> bytes:
    result = runner.run(
        (
            "/usr/bin/git",
            "-C",
            os.fspath(settings.repo_root),
            "cat-file",
            "blob",
            f"{anchor.commit}:{relative_path}",
        ),
        cwd=settings.repo_root,
        timeout_seconds=GIT_TIMEOUT_SECONDS,
        stdout_limit=HEAD_TOOL_MAX_BYTES,
        stderr_limit=GIT_HEAD_MAX_BYTES,
    )
    raw = _require_clean_command(result, f"HEAD blob capture for {relative_path}")
    if not raw:
        fail(f"HEAD blob for {relative_path} is empty")
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
            parent_current != self._parent_identity
            or parent_entry != self._parent_identity
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
    tools: HeadPromotionToolSnapshotOwner,
    *,
    runner: CommandRunner = SYSTEM_COMMAND_RUNNER,
) -> dict[str, Any]:
    """Run full product validation using only the two private anchored HEAD tools."""

    if (
        not isinstance(settings, ProductionPreflightSettings)
        or not isinstance(anchor, GitAnchor)
        or not isinstance(tools, HeadPromotionToolSnapshotOwner)
        or anchor.settings != settings
        or tools.settings != settings
        or tools.anchor != anchor
        or tools.closed
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
                os.fspath(settings.python_executable),
                "-B",
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
    "HEAD_PROMOTION_TOOL_PATHS",
    "HeadPromotionToolSnapshotOwner",
    "ProductionPreflightError",
    "ProductionPreflightSettings",
    "canonical_campaign_lock_path",
    "production_preflight_settings",
    "run_pinned_full_promotion_validation",
]
