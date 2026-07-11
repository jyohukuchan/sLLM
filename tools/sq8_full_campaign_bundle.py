#!/usr/bin/env python3
"""Own the private staging and one-time publication of a full SQ8 campaign."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import secrets
import shutil
import stat
from pathlib import Path, PurePosixPath
from typing import Callable, NoReturn


COPY_CHUNK_BYTES = 64 << 10
MAX_INLINE_BYTES = 16 << 20
COMPONENT_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

PREVALIDATION_ROOT_FILES = frozenset(
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
    }
)
BROWSER_FILES = frozenset({"openwebui-stop-before.png", "post-header-failure.png"})
VALIDATION_FILE = "release-validation.json"


class CampaignBundleError(RuntimeError):
    """A fail-closed staging or publication error."""


def fail(message: str) -> NoReturn:
    raise CampaignBundleError(message)


@dataclasses.dataclass(frozen=True)
class FileEvidence:
    bytes: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class _Identity:
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
    def from_stat(cls, value: os.stat_result) -> _Identity:
        return cls(
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

    def directory_anchor(self) -> tuple[int, int, int, int, int]:
        return (self.device, self.inode, self.mode, self.uid, self.gid)


@dataclasses.dataclass(frozen=True)
class _ImmutableFileSeal:
    identity: _Identity
    sha256: str


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for full campaign publication")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _file_read_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for full campaign publication")
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


def _entry_identity(parent_fd: int, name: str) -> _Identity:
    try:
        return _Identity.from_stat(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        )
    except OSError:
        fail("campaign bundle entry is unavailable")


def _safe_close(descriptor: int) -> None:
    if descriptor < 0:
        return
    try:
        os.close(descriptor)
    except OSError:
        fail("failed to close a campaign bundle descriptor")


def _write_all(descriptor: int, raw: bytes) -> None:
    offset = 0
    try:
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                fail("campaign artifact write made no progress")
            offset += written
    except CampaignBundleError:
        raise
    except OSError:
        fail("failed to write a campaign artifact")


def _relative_target(relative: str) -> tuple[bool, str]:
    if type(relative) is not str or not relative:
        fail("campaign artifact path is empty")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or relative.startswith("./")
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        fail("campaign artifact path is unsafe")
    if len(pure.parts) == 1 and relative in PREVALIDATION_ROOT_FILES:
        return False, relative
    if (
        len(pure.parts) == 2
        and pure.parts[0] == "browser"
        and pure.parts[1] in BROWSER_FILES
    ):
        return True, pure.parts[1]
    fail("campaign artifact path is outside the frozen bundle layout")


class AtomicCampaignDirectory:
    """Keep work outside evidence and publish only a validator-complete bundle."""

    def __init__(self, final_path: Path, *, uid: int, gid: int):
        if type(uid) is not int or uid < 0 or type(gid) is not int or gid < 0:
            fail("campaign owner binding is invalid")
        absolute_final = Path(os.path.abspath(final_path))
        if absolute_final.name in {"", ".", ".."}:
            fail("campaign destination name is invalid")
        absolute_parent = absolute_final.parent
        try:
            resolved_parent = absolute_parent.resolve(strict=True)
        except OSError:
            fail("campaign destination parent is unavailable")
        if resolved_parent != absolute_parent:
            fail("campaign destination parent contains a symbolic link")
        self.final_path = absolute_final
        self.parent_path = absolute_parent
        self.uid = uid
        self.gid = gid
        self.parent_fd = -1
        self.stage_fd = -1
        self.browser_fd = -1
        self.work_fd = -1
        self.parent_identity: _Identity | None = None
        self.stage_identity: _Identity | None = None
        self.browser_identity: _Identity | None = None
        self.work_identity: _Identity | None = None
        self.published = False
        self.closed = False
        self._components: set[str] = set()
        self._prevalidation_seals: tuple[tuple[str, _ImmutableFileSeal], ...] | None = (
            None
        )
        nonce = secrets.token_hex(12)
        self.stage_name = f".{absolute_final.name}.incomplete-{nonce}"
        self.work_name = f".{absolute_final.name}.work-{nonce}"
        self.stage_path = absolute_parent / self.stage_name
        self.work_path = absolute_parent / self.work_name
        self._open()

    def _open(self) -> None:
        try:
            self.parent_fd = os.open(self.parent_path, _directory_flags())
            parent = _Identity.from_stat(os.fstat(self.parent_fd))
            if not stat.S_ISDIR(parent.mode):
                fail("campaign destination parent is not a directory")
            self.parent_identity = parent
            try:
                os.stat(
                    self.final_path.name,
                    dir_fd=self.parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                fail("campaign destination already exists")
            os.mkdir(self.stage_name, 0o700, dir_fd=self.parent_fd)
            os.mkdir(self.work_name, 0o700, dir_fd=self.parent_fd)
            self.stage_fd = os.open(
                self.stage_name, _directory_flags(), dir_fd=self.parent_fd
            )
            self.work_fd = os.open(
                self.work_name, _directory_flags(), dir_fd=self.parent_fd
            )
            os.mkdir("browser", 0o700, dir_fd=self.stage_fd)
            self.browser_fd = os.open(
                "browser", _directory_flags(), dir_fd=self.stage_fd
            )
            self.stage_identity = _Identity.from_stat(os.fstat(self.stage_fd))
            self.work_identity = _Identity.from_stat(os.fstat(self.work_fd))
            self.browser_identity = _Identity.from_stat(os.fstat(self.browser_fd))
            self._require_owned_directory(self.stage_identity, "campaign stage")
            self._require_owned_directory(self.work_identity, "campaign work root")
            self._require_owned_directory(
                self.browser_identity, "campaign browser directory"
            )
        except CampaignBundleError:
            self.abort()
            raise
        except OSError:
            self.abort()
            fail("failed to create the private campaign directories")

    def __enter__(self) -> AtomicCampaignDirectory:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        error: BaseException | None,
        _tb: object,
    ) -> None:
        if exc_type is not None or not self.published:
            try:
                self.abort()
            except BaseException:
                if error is None:
                    raise
                error.add_note(
                    "campaign bundle abort also failed while preserving the active error"
                )
        else:
            self.close()

    def _require_owned_directory(self, identity: _Identity, label: str) -> None:
        if (
            not stat.S_ISDIR(identity.mode)
            or stat.S_IMODE(identity.mode) != 0o700
            or identity.uid != self.uid
            or identity.gid != self.gid
        ):
            fail(f"{label} mode or owner differs")

    def _require_open(self) -> None:
        if self.closed or self.published:
            fail("campaign directory is no longer writable")

    def artifact_path(self, relative: str) -> Path:
        browser, name = _relative_target(relative)
        return self.stage_path / "browser" / name if browser else self.stage_path / name

    def component_directory(self, label: str) -> Path:
        self._require_open()
        if type(label) is not str or COMPONENT_RE.fullmatch(label) is None:
            fail("campaign component label is invalid")
        if label in self._components:
            fail("campaign component directory is duplicated")
        try:
            os.mkdir(label, 0o700, dir_fd=self.work_fd)
            identity = _entry_identity(self.work_fd, label)
            self._require_owned_directory(identity, "campaign component directory")
        except CampaignBundleError:
            raise
        except OSError:
            fail("failed to create a campaign component directory")
        self._components.add(label)
        return self.work_path / label

    def write_bytes(
        self,
        relative: str,
        raw: bytes,
        *,
        scan: Callable[[bytes, str], None],
    ) -> FileEvidence:
        self._require_open()
        if type(raw) is not bytes or len(raw) > MAX_INLINE_BYTES or not callable(scan):
            fail("campaign artifact bytes or scanner type differs")
        browser, name = _relative_target(relative)
        parent_fd = self.browser_fd if browser else self.stage_fd
        label = f"campaign artifact {relative}"
        scan(raw, label)
        temporary = f".{name}.incomplete-{secrets.token_hex(8)}"
        descriptor = -1
        published = False
        try:
            try:
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                fail("campaign artifact destination already exists")
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
            _write_all(descriptor, raw)
            os.fsync(descriptor)
            identity = _Identity.from_stat(os.fstat(descriptor))
            if (
                not stat.S_ISREG(identity.mode)
                or stat.S_IMODE(identity.mode) != 0o600
                or identity.links != 1
                or identity.uid != self.uid
                or identity.gid != self.gid
                or identity.size != len(raw)
            ):
                fail("campaign artifact identity differs before publication")
            _safe_close(descriptor)
            descriptor = -1
            os.rename(
                temporary,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            published = True
            os.fsync(parent_fd)
            return FileEvidence(len(raw), hashlib.sha256(raw).hexdigest())
        except CampaignBundleError:
            raise
        except OSError:
            fail("failed to publish a campaign artifact")
        finally:
            if descriptor >= 0:
                _safe_close(descriptor)
            if not published:
                try:
                    os.unlink(temporary, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    fail("failed to remove an incomplete campaign artifact")

    def copy_file(
        self,
        source: Path,
        relative: str,
        *,
        expected_bytes: int,
        expected_sha256: str,
        maximum_bytes: int,
        scan: Callable[[bytes, str], None],
    ) -> FileEvidence:
        self._require_open()
        if (
            type(expected_bytes) is not int
            or expected_bytes < 0
            or type(maximum_bytes) is not int
            or maximum_bytes < expected_bytes
            or SHA256_RE.fullmatch(expected_sha256) is None
            or not callable(scan)
        ):
            fail("campaign copy binding is invalid")
        source_path = Path(os.path.abspath(source))
        source_parent_fd = -1
        source_fd = -1
        destination_fd = -1
        published = False
        browser, name = _relative_target(relative)
        destination_parent_fd = self.browser_fd if browser else self.stage_fd
        temporary = f".{name}.incomplete-{secrets.token_hex(8)}"
        label = f"campaign copied artifact {relative}"
        digest = hashlib.sha256()
        total = 0
        try:
            source_parent_fd = os.open(source_path.parent, _directory_flags())
            entry = _entry_identity(source_parent_fd, source_path.name)
            if (
                not stat.S_ISREG(entry.mode)
                or entry.links != 1
                or entry.size != expected_bytes
                or entry.size > maximum_bytes
            ):
                fail("campaign copy source identity or size differs")
            source_fd = os.open(
                source_path.name,
                _file_read_flags(),
                dir_fd=source_parent_fd,
            )
            if _Identity.from_stat(os.fstat(source_fd)) != entry:
                fail("campaign copy source changed while opening")
            try:
                os.stat(name, dir_fd=destination_parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                fail("campaign copied artifact destination already exists")
            destination_fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=destination_parent_fd,
            )
            while True:
                chunk = os.read(source_fd, COPY_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum_bytes:
                    fail("campaign copy source exceeded its byte bound")
                scan(chunk, label)
                digest.update(chunk)
                _write_all(destination_fd, chunk)
            if (
                total != expected_bytes
                or digest.hexdigest() != expected_sha256
                or _Identity.from_stat(os.fstat(source_fd)) != entry
                or _entry_identity(source_parent_fd, source_path.name) != entry
            ):
                fail("campaign copy source content or identity differs")
            os.fsync(destination_fd)
            destination = _Identity.from_stat(os.fstat(destination_fd))
            if (
                not stat.S_ISREG(destination.mode)
                or stat.S_IMODE(destination.mode) != 0o600
                or destination.links != 1
                or destination.uid != self.uid
                or destination.gid != self.gid
                or destination.size != total
            ):
                fail("campaign copied artifact identity differs")
            _safe_close(destination_fd)
            destination_fd = -1
            os.rename(
                temporary,
                name,
                src_dir_fd=destination_parent_fd,
                dst_dir_fd=destination_parent_fd,
            )
            published = True
            os.fsync(destination_parent_fd)
            return FileEvidence(total, digest.hexdigest())
        except CampaignBundleError:
            raise
        except OSError:
            fail("failed to copy a campaign artifact")
        finally:
            for descriptor in (destination_fd, source_fd, source_parent_fd):
                if descriptor >= 0:
                    _safe_close(descriptor)
            if not published:
                try:
                    os.unlink(temporary, dir_fd=destination_parent_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    fail("failed to remove an incomplete copied artifact")

    def _validate_anchor(self, *, stage_entry_name: str | None = None) -> None:
        assert self.parent_identity is not None
        assert self.stage_identity is not None
        assert self.browser_identity is not None
        entry_name = self.stage_name if stage_entry_name is None else stage_entry_name
        if (
            _Identity.from_stat(os.fstat(self.parent_fd)).directory_anchor()
            != self.parent_identity.directory_anchor()
            or _Identity.from_stat(os.fstat(self.stage_fd)).directory_anchor()
            != self.stage_identity.directory_anchor()
            or _entry_identity(self.parent_fd, entry_name).directory_anchor()
            != self.stage_identity.directory_anchor()
            or _Identity.from_stat(os.fstat(self.browser_fd)).directory_anchor()
            != self.browser_identity.directory_anchor()
            or _entry_identity(self.stage_fd, "browser").directory_anchor()
            != self.browser_identity.directory_anchor()
        ):
            fail("campaign directory identity changed")

    def _validate_files(
        self,
        *,
        include_validation: bool,
        stage_entry_name: str | None = None,
    ) -> None:
        self._validate_anchor(stage_entry_name=stage_entry_name)
        expected_root = set(PREVALIDATION_ROOT_FILES) | {"browser"}
        if include_validation:
            expected_root.add(VALIDATION_FILE)
        try:
            if set(os.listdir(self.stage_fd)) != expected_root:
                fail("campaign root layout differs")
            if set(os.listdir(self.browser_fd)) != set(BROWSER_FILES):
                fail("campaign browser layout differs")
            for parent_fd, names in (
                (
                    self.stage_fd,
                    set(PREVALIDATION_ROOT_FILES)
                    | ({VALIDATION_FILE} if include_validation else set()),
                ),
                (self.browser_fd, set(BROWSER_FILES)),
            ):
                for name in names:
                    identity = _entry_identity(parent_fd, name)
                    if (
                        not stat.S_ISREG(identity.mode)
                        or stat.S_IMODE(identity.mode) != 0o600
                        or identity.links != 1
                        or identity.uid != self.uid
                        or identity.gid != self.gid
                        or identity.size < 1
                    ):
                        fail("campaign artifact layout, mode, owner, or size differs")
                    descriptor = os.open(name, _file_read_flags(), dir_fd=parent_fd)
                    try:
                        if _Identity.from_stat(os.fstat(descriptor)) != identity:
                            fail("campaign artifact changed while validating layout")
                    finally:
                        _safe_close(descriptor)
        except CampaignBundleError:
            raise
        except OSError:
            fail("failed to validate the campaign bundle layout")

    def _snapshot_file(
        self,
        parent_fd: int,
        name: str,
        label: str,
    ) -> _ImmutableFileSeal:
        descriptor = -1
        try:
            entry = _entry_identity(parent_fd, name)
            if (
                not stat.S_ISREG(entry.mode)
                or stat.S_IMODE(entry.mode) != 0o600
                or entry.links != 1
                or entry.uid != self.uid
                or entry.gid != self.gid
                or entry.size < 1
            ):
                fail(f"{label} identity differs")
            descriptor = os.open(name, _file_read_flags(), dir_fd=parent_fd)
            if _Identity.from_stat(os.fstat(descriptor)) != entry:
                fail(f"{label} changed while opening")
            digest = hashlib.sha256()
            total = 0
            while True:
                chunk = os.read(descriptor, COPY_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                digest.update(chunk)
            os.fsync(descriptor)
            if (
                total != entry.size
                or _Identity.from_stat(os.fstat(descriptor)) != entry
                or _entry_identity(parent_fd, name) != entry
            ):
                fail(f"{label} changed while snapshotting")
            return _ImmutableFileSeal(entry, digest.hexdigest())
        except CampaignBundleError:
            raise
        except OSError:
            fail(f"failed to snapshot {label}")
        finally:
            if descriptor >= 0:
                _safe_close(descriptor)

    def _snapshot_prevalidation_files(
        self,
        *,
        include_validation: bool,
        stage_entry_name: str | None = None,
    ) -> tuple[tuple[str, _ImmutableFileSeal], ...]:
        self._validate_files(
            include_validation=include_validation,
            stage_entry_name=stage_entry_name,
        )
        snapshots: list[tuple[str, _ImmutableFileSeal]] = []
        for relative in expected_prevalidation_paths():
            browser, name = _relative_target(relative)
            parent_fd = self.browser_fd if browser else self.stage_fd
            snapshots.append(
                (
                    relative,
                    self._snapshot_file(
                        parent_fd,
                        name,
                        f"campaign prevalidation artifact {relative}",
                    ),
                )
            )
        self._validate_anchor(stage_entry_name=stage_entry_name)
        return tuple(snapshots)

    def validate_before_independent_validator(self) -> None:
        self._require_open()
        if self._prevalidation_seals is not None:
            fail("campaign prevalidation inputs were already sealed")
        self._prevalidation_seals = self._snapshot_prevalidation_files(
            include_validation=False
        )

    def _rollback_renamed_stage(self) -> None:
        assert self.stage_identity is not None
        try:
            current = _entry_identity(self.parent_fd, self.final_path.name)
            if current.directory_anchor() != self.stage_identity.directory_anchor():
                fail("refusing to roll back a replaced campaign destination")
            os.rename(
                self.final_path.name,
                self.stage_name,
                src_dir_fd=self.parent_fd,
                dst_dir_fd=self.parent_fd,
            )
            os.fsync(self.parent_fd)
        except CampaignBundleError:
            raise
        except OSError:
            fail("failed to roll back an incomplete campaign publication")

    def publish(self, validation_evidence: FileEvidence) -> Path:
        self._require_open()
        if self._prevalidation_seals is None:
            fail("campaign prevalidation inputs were not sealed")
        if (
            not isinstance(validation_evidence, FileEvidence)
            or type(validation_evidence.bytes) is not int
            or validation_evidence.bytes < 1
            or SHA256_RE.fullmatch(validation_evidence.sha256) is None
        ):
            fail("campaign validation evidence binding differs")
        renamed = False
        try:
            if os.listdir(self.work_fd):
                fail("campaign work root is not empty before publication")
            os.fsync(self.browser_fd)
            os.fsync(self.stage_fd)
            os.fsync(self.parent_fd)
            current_seals = self._snapshot_prevalidation_files(include_validation=True)
            if current_seals != self._prevalidation_seals:
                fail(
                    "campaign prevalidation input changed after independent validation"
                )
            validation_seal = self._snapshot_file(
                self.stage_fd,
                VALIDATION_FILE,
                "campaign independent validation artifact",
            )
            if (
                FileEvidence(validation_seal.identity.size, validation_seal.sha256)
                != validation_evidence
            ):
                fail(
                    "campaign independent validation artifact differs from its evidence"
                )
            _safe_close(self.work_fd)
            self.work_fd = -1
            os.rmdir(self.work_name, dir_fd=self.parent_fd)
            os.rename(
                self.stage_name,
                self.final_path.name,
                src_dir_fd=self.parent_fd,
                dst_dir_fd=self.parent_fd,
            )
            renamed = True
            published_identity = _entry_identity(self.parent_fd, self.final_path.name)
            assert self.stage_identity is not None
            if (
                published_identity.directory_anchor()
                != self.stage_identity.directory_anchor()
            ):
                fail("published campaign directory identity differs")
            if (
                self._snapshot_prevalidation_files(
                    include_validation=True,
                    stage_entry_name=self.final_path.name,
                )
                != self._prevalidation_seals
            ):
                fail("campaign prevalidation input changed during publication")
            published_validation = self._snapshot_file(
                self.stage_fd,
                VALIDATION_FILE,
                "published campaign independent validation artifact",
            )
            if (
                FileEvidence(
                    published_validation.identity.size,
                    published_validation.sha256,
                )
                != validation_evidence
            ):
                fail(
                    "campaign independent validation artifact changed during publication"
                )
            os.fsync(self.parent_fd)
            self.published = True
            _safe_close(self.browser_fd)
            self.browser_fd = -1
            _safe_close(self.stage_fd)
            self.stage_fd = -1
            self.close()
            return self.final_path
        except BaseException as error:
            if renamed and not self.published:
                try:
                    self._rollback_renamed_stage()
                except CampaignBundleError as rollback_error:
                    raise rollback_error from error
            if isinstance(error, CampaignBundleError):
                raise
            if isinstance(error, OSError):
                fail("failed to publish the validated campaign bundle")
            raise

    def clear_component_work(self) -> None:
        self._require_open()
        assert self.work_identity is not None
        if (
            _Identity.from_stat(os.fstat(self.work_fd)).directory_anchor()
            != self.work_identity.directory_anchor()
            or _entry_identity(self.parent_fd, self.work_name).directory_anchor()
            != self.work_identity.directory_anchor()
        ):
            fail("campaign work root identity changed")
        for name in tuple(os.listdir(self.work_fd)):
            if name not in self._components:
                fail("campaign work root contains an unknown entry")
            path = self.work_path / name
            try:
                identity = path.lstat()
            except OSError:
                fail("campaign component work entry is unavailable")
            if not stat.S_ISDIR(identity.st_mode) or stat.S_ISLNK(identity.st_mode):
                fail("campaign component work entry is not a real directory")
            try:
                shutil.rmtree(path)
            except OSError:
                fail("failed to remove campaign component work")
            self._components.remove(name)
        os.fsync(self.work_fd)

    def abort(self) -> None:
        if self.published or self.closed:
            return
        pending: CampaignBundleError | None = None
        for attribute in ("browser_fd", "stage_fd", "work_fd"):
            descriptor = getattr(self, attribute)
            if descriptor >= 0:
                try:
                    _safe_close(descriptor)
                except CampaignBundleError as error:
                    pending = error
                setattr(self, attribute, -1)
        for path, identity in (
            (self.stage_path, self.stage_identity),
            (self.work_path, self.work_identity),
        ):
            try:
                current = _Identity.from_stat(path.lstat())
            except FileNotFoundError:
                continue
            except OSError:
                pending = CampaignBundleError(
                    "failed to inspect a private campaign directory during abort"
                )
                continue
            if (
                identity is not None
                and current.directory_anchor() != identity.directory_anchor()
            ):
                pending = CampaignBundleError(
                    "refusing to remove a replaced private campaign directory"
                )
                continue
            try:
                shutil.rmtree(path)
            except OSError:
                pending = CampaignBundleError(
                    "failed to remove a private campaign directory"
                )
        if self.parent_fd >= 0:
            try:
                _safe_close(self.parent_fd)
            except CampaignBundleError as error:
                pending = error
            self.parent_fd = -1
        self.closed = True
        if pending is not None:
            raise pending

    def close(self) -> None:
        if self.closed:
            return
        pending: CampaignBundleError | None = None
        for attribute in ("browser_fd", "stage_fd", "work_fd", "parent_fd"):
            descriptor = getattr(self, attribute)
            if descriptor >= 0:
                try:
                    _safe_close(descriptor)
                except CampaignBundleError as error:
                    pending = error
                setattr(self, attribute, -1)
        self.closed = True
        if pending is not None:
            raise pending


def expected_prevalidation_paths() -> tuple[str, ...]:
    return tuple(
        sorted(
            set(PREVALIDATION_ROOT_FILES)
            | {f"browser/{name}" for name in BROWSER_FILES},
            key=lambda item: item.encode("utf-8"),
        )
    )


__all__ = [
    "AtomicCampaignDirectory",
    "BROWSER_FILES",
    "CampaignBundleError",
    "FileEvidence",
    "PREVALIDATION_ROOT_FILES",
    "VALIDATION_FILE",
    "expected_prevalidation_paths",
]
