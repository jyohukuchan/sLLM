#!/usr/bin/env python3
"""Revalidate and convert one formal OpenWebUI Stop gate bundle."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import stat
import struct
import sys
import types
import urllib.parse
import zlib
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple, NoReturn, Protocol, cast


VIEW_SCHEMA = "ullm.sq8.openwebui_stop_gate_ingest.view.v1"
CAMPAIGN_PHASE = "cancellation"
CAMPAIGN_CASE = "openwebui_stop_after_visible_content"
RECOVERY_CASE = f"{CAMPAIGN_CASE}-recovery"
CAMPAIGN_SERVICE_UNIT = "ullm-openai.service"
ROOT_FILES = frozenset(
    {"observer.raw.jsonl", "service-journal.raw.jsonl", "summary.json", "browser"}
)
BROWSER_FILES = frozenset(
    {
        "browser-stdout.jsonl",
        "openwebui-stop-summary.json",
        "openwebui-stop-before.png",
    }
)
FILE_LIMITS = {
    "observer": 16 << 20,
    "journal": 64 << 20,
    "summary": (1 << 20) + 1,
    "browser_stdout": 4 << 20,
    "browser_summary": (1 << 20) + 1,
    "screenshot": 64 << 20,
}
FILE_LAYOUT = {
    "observer": (False, "observer.raw.jsonl"),
    "journal": (False, "service-journal.raw.jsonl"),
    "summary": (False, "summary.json"),
    "browser_stdout": (True, "browser-stdout.jsonl"),
    "browser_summary": (True, "openwebui-stop-summary.json"),
    "screenshot": (True, "openwebui-stop-before.png"),
}
REQUIRED_JOURNAL_FIELDS = (
    "__CURSOR",
    "__MONOTONIC_TIMESTAMP",
    "_BOOT_ID",
    "_PID",
    "_SYSTEMD_UNIT",
    "PRIORITY",
    "MESSAGE",
)
MAX_JSON_LINE_BYTES = 1 << 20
COPY_CHUNK_BYTES = 64 << 10
SOURCE_LIMITS = {"gate": 8 << 20, "browser": 2 << 20}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
BOOT_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
CONTENT_IMAGE_RE = re.compile(
    r"(?:(?:[A-Za-z0-9][A-Za-z0-9._/:+-]*)@)?sha256:([0-9a-f]{64})\Z"
)
SERVICE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@:-]{0,127}\.service\Z")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class StopGateIngestError(RuntimeError):
    """A fail-closed conversion error without evidence-derived diagnostics."""


def fail(message: str) -> NoReturn:
    raise StopGateIngestError(message)


class BundleLifecycleClaimProtocol(Protocol):
    raw: bytes
    phase: str
    case_id: str


@dataclasses.dataclass(frozen=True)
class StopGateInputBindings:
    gate_source: Path
    gate_source_sha256: str
    browser_script: Path
    browser_script_sha256: str
    browser_image_reference: str
    browser_image_content_id: str
    openwebui_url: str
    service_unit: str
    service_user: str
    boot_id: str
    control_group: str
    gateway_pid: int
    gateway_starttime_ticks: int
    worker_pid: int
    worker_starttime_ticks: int
    restart_count: int
    uid: int
    gid: int
    forbidden_values: tuple[bytes, ...]


class ScreenshotEvidence(NamedTuple):
    path: Path
    bytes: int
    sha256: str


class StopGateIngestResult(NamedTuple):
    browser_action_records: tuple[dict[str, Any], ...]
    lifecycle_claims: tuple[BundleLifecycleClaimProtocol, ...]
    screenshot_evidence: ScreenshotEvidence
    derived_view: dict[str, Any]


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


@dataclasses.dataclass
class _OpenedFile:
    key: str
    name: str
    parent_fd: int
    fd: int
    identity: _Identity
    maximum: int
    consumed: bool = False
    streamed_bytes: int = 0
    sha256: str | None = None


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for Stop gate ingestion")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _file_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for Stop gate ingestion")
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


def _safe_close(descriptor: int) -> None:
    if descriptor < 0:
        return
    try:
        os.close(descriptor)
    except OSError:
        fail("failed to close a sealed Stop gate descriptor")


def _entry_identity(parent_fd: int, name: str) -> _Identity:
    try:
        return _Identity.from_stat(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        )
    except OSError:
        fail("sealed Stop gate entry is unavailable")


def _require_directory(
    identity: _Identity,
    *,
    mode: int,
    uid: int,
    gid: int,
    links: int,
) -> None:
    if (
        not stat.S_ISDIR(identity.mode)
        or stat.S_IMODE(identity.mode) != mode
        or identity.uid != uid
        or identity.gid != gid
        or identity.links != links
    ):
        fail("Stop gate directory mode, owner, or link count differs")


def _require_file(
    identity: _Identity,
    *,
    maximum: int,
    uid: int,
    gid: int,
) -> None:
    if (
        not stat.S_ISREG(identity.mode)
        or stat.S_IMODE(identity.mode) != 0o600
        or identity.uid != uid
        or identity.gid != gid
        or identity.links != 1
        or identity.size < 1
        or identity.size > maximum
    ):
        fail("Stop gate file layout, owner, mode, link count, or size differs")


class _SecretScanner:
    def __init__(self, values: tuple[bytes, ...]):
        for value in values:
            if type(value) is not bytes or len(value) < 4:
                fail("forbidden Stop evidence values are invalid")
        self.values = values
        self.overlap = max((len(value) for value in values), default=1) - 1
        self.tail = b""

    def consume(self, chunk: bytes) -> None:
        combined = self.tail + chunk
        if any(value in combined for value in self.values):
            fail("Stop gate evidence contains forbidden cleartext")
        self.tail = combined[-self.overlap :] if self.overlap else b""


class _PngValidator:
    """Validate PNG framing and CRCs without retaining image payload chunks."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.signature = False
        self.chunk_index = 0
        self.chunk_type: bytes | None = None
        self.remaining = 0
        self.crc = 0
        self.ihdr = bytearray()
        self.saw_idat = False
        self.idat_closed = False
        self.decompressor: Any | None = None
        self.expected_decoded_bytes = 0
        self.decoded_bytes = 0
        self.row_stride = 0
        self.ended = False

    def consume(self, raw: bytes) -> None:
        if self.ended and raw:
            fail("PNG carries bytes after IEND")
        self.buffer.extend(raw)
        while True:
            if not self.signature:
                if len(self.buffer) < len(PNG_SIGNATURE):
                    return
                if bytes(self.buffer[:8]) != PNG_SIGNATURE:
                    fail("screenshot does not have the PNG signature")
                del self.buffer[:8]
                self.signature = True
            if self.chunk_type is None:
                if len(self.buffer) < 8:
                    return
                length = int.from_bytes(self.buffer[:4], "big")
                chunk_type = bytes(self.buffer[4:8])
                del self.buffer[:8]
                if length > 64 << 20 or not re.fullmatch(rb"[A-Za-z]{4}", chunk_type):
                    fail("PNG chunk framing differs")
                if self.chunk_index == 0 and (chunk_type != b"IHDR" or length != 13):
                    fail("PNG does not begin with one exact IHDR")
                if chunk_type == b"IHDR" and self.chunk_index != 0:
                    fail("PNG IHDR is duplicated")
                if chunk_type == b"IEND" and (length != 0 or not self.saw_idat):
                    fail("PNG IEND ordering differs")
                if self.saw_idat and chunk_type != b"IDAT":
                    self.idat_closed = True
                if chunk_type == b"IDAT" and self.idat_closed:
                    fail("PNG IDAT chunks are not consecutive")
                self.chunk_type = chunk_type
                self.remaining = length
                self.crc = zlib.crc32(chunk_type)
                self.ihdr.clear()
            if self.remaining:
                if not self.buffer:
                    return
                take = min(self.remaining, len(self.buffer))
                part = bytes(self.buffer[:take])
                del self.buffer[:take]
                self.remaining -= take
                self.crc = zlib.crc32(part, self.crc)
                if self.chunk_type == b"IHDR":
                    self.ihdr.extend(part)
                elif self.chunk_type == b"IDAT":
                    self._decode_idat(part)
                if self.remaining:
                    return
            if len(self.buffer) < 4:
                return
            stored_crc = int.from_bytes(self.buffer[:4], "big")
            del self.buffer[:4]
            if stored_crc != self.crc & 0xFFFF_FFFF:
                fail("PNG chunk CRC differs")
            assert self.chunk_type is not None
            if self.chunk_type == b"IHDR":
                self._validate_ihdr(bytes(self.ihdr))
            elif self.chunk_type == b"IDAT":
                self.saw_idat = True
            elif self.chunk_type == b"IEND":
                self._finish_image_data()
                self.ended = True
                self.chunk_type = None
                self.chunk_index += 1
                if self.buffer:
                    fail("PNG carries bytes after IEND")
                return
            self.chunk_type = None
            self.chunk_index += 1

    def _validate_ihdr(self, raw: bytes) -> None:
        if len(raw) != 13:
            fail("PNG IHDR length differs")
        width, height, depth, color, compression, filtering, interlace = struct.unpack(
            ">IIBBBBB", raw
        )
        allowed_depths = {
            0: {1, 2, 4, 8, 16},
            2: {8, 16},
            3: {1, 2, 4, 8},
            4: {8, 16},
            6: {8, 16},
        }
        if (
            not 1 <= width <= 16_384
            or not 1 <= height <= 16_384
            or depth not in allowed_depths.get(color, set())
            or compression != 0
            or filtering != 0
            or interlace != 0
        ):
            fail("PNG IHDR values differ")
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
        row_bytes = (width * channels * depth + 7) // 8
        expected = height * (row_bytes + 1)
        if expected > 128 << 20:
            fail("PNG decoded size exceeds its bound")
        self.row_stride = row_bytes + 1
        self.expected_decoded_bytes = expected
        self.decompressor = zlib.decompressobj()

    def _accept_decoded(self, raw: bytes) -> None:
        if self.decoded_bytes + len(raw) > self.expected_decoded_bytes:
            fail("PNG decoded bytes exceed IHDR dimensions")
        for offset, value in enumerate(raw):
            if (self.decoded_bytes + offset) % self.row_stride == 0 and value > 4:
                fail("PNG scanline filter type differs")
        self.decoded_bytes += len(raw)

    def _decode_idat(self, raw: bytes) -> None:
        if self.decompressor is None or (self.decompressor.eof and raw):
            fail("PNG IDAT zlib stream ordering differs")
        remaining = self.expected_decoded_bytes - self.decoded_bytes
        try:
            decoded = self.decompressor.decompress(raw, remaining + 1)
        except zlib.error as error:
            raise StopGateIngestError("PNG IDAT zlib stream is invalid") from error
        self._accept_decoded(decoded)
        if self.decompressor.unconsumed_tail or self.decompressor.unused_data:
            fail("PNG IDAT contains excess compressed data")

    def _finish_image_data(self) -> None:
        if self.decompressor is None:
            fail("PNG lacks a decodable IDAT stream")
        remaining = self.expected_decoded_bytes - self.decoded_bytes
        try:
            decoded = self.decompressor.flush(remaining + 1)
        except zlib.error as error:
            raise StopGateIngestError("PNG IDAT zlib finalization failed") from error
        self._accept_decoded(decoded)
        if (
            not self.decompressor.eof
            or self.decompressor.unconsumed_tail
            or self.decompressor.unused_data
            or self.decoded_bytes != self.expected_decoded_bytes
        ):
            fail("PNG IDAT decoded size or terminator differs")

    def finish(self) -> None:
        if (
            not self.signature
            or not self.ended
            or self.chunk_type is not None
            or self.buffer
        ):
            fail("PNG stream is incomplete")


class _BundleSnapshot:
    """Retain the exact root/browser tree through final FD re-hashing."""

    def __init__(
        self,
        root: Path,
        *,
        uid: int,
        gid: int,
        forbidden_values: tuple[bytes, ...],
    ) -> None:
        self.root_path = Path(os.path.abspath(root))
        self.uid = uid
        self.gid = gid
        self.forbidden_values = forbidden_values
        self.parent_fd = -1
        self.root_fd = -1
        self.browser_fd = -1
        self.root_identity: _Identity | None = None
        self.browser_identity: _Identity | None = None
        self.files: dict[str, _OpenedFile] = {}
        self.sealed = False
        self.closed = False
        self._open()

    def _open(self) -> None:
        _SecretScanner(self.forbidden_values)
        try:
            self.parent_fd = os.open(self.root_path.parent, _directory_flags())
            self.root_fd = os.open(
                self.root_path.name, _directory_flags(), dir_fd=self.parent_fd
            )
            root = _Identity.from_stat(os.fstat(self.root_fd))
            if _entry_identity(self.parent_fd, self.root_path.name) != root:
                fail("Stop gate root identity changed while opening")
            _require_directory(root, mode=0o700, uid=self.uid, gid=self.gid, links=3)
            if frozenset(os.listdir(self.root_fd)) != ROOT_FILES:
                fail("Stop gate root layout differs")
            browser_entry = _entry_identity(self.root_fd, "browser")
            _require_directory(
                browser_entry, mode=0o700, uid=self.uid, gid=self.gid, links=2
            )
            self.browser_fd = os.open(
                "browser", _directory_flags(), dir_fd=self.root_fd
            )
            browser = _Identity.from_stat(os.fstat(self.browser_fd))
            if browser != browser_entry:
                fail("Stop gate browser identity changed while opening")
            if frozenset(os.listdir(self.browser_fd)) != BROWSER_FILES:
                fail("Stop gate browser layout differs")
            self.root_identity = root
            self.browser_identity = browser
            for key, (in_browser, name) in FILE_LAYOUT.items():
                parent_fd = self.browser_fd if in_browser else self.root_fd
                entry = _entry_identity(parent_fd, name)
                _require_file(
                    entry,
                    maximum=FILE_LIMITS[key],
                    uid=self.uid,
                    gid=self.gid,
                )
                descriptor = os.open(name, _file_flags(), dir_fd=parent_fd)
                opened = _Identity.from_stat(os.fstat(descriptor))
                if opened != entry:
                    os.close(descriptor)
                    fail("Stop gate artifact changed while opening")
                self.files[key] = _OpenedFile(
                    key,
                    name,
                    parent_fd,
                    descriptor,
                    opened,
                    FILE_LIMITS[key],
                )
        except StopGateIngestError:
            self.close()
            raise
        except OSError:
            self.close()
            fail("failed to open the Stop gate bundle without following links")

    def _chunks(self, key: str) -> Iterator[bytes]:
        if self.closed or self.sealed:
            fail("Stop gate bundle snapshot is no longer readable")
        item = self.files.get(key)
        if item is None or item.consumed:
            fail("Stop gate artifact was read outside its fixed schedule")
        try:
            if _Identity.from_stat(os.fstat(item.fd)) != item.identity:
                fail("Stop gate artifact changed before streaming")
            os.lseek(item.fd, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            scanner = _SecretScanner(self.forbidden_values)
            total = 0
            while True:
                chunk = os.read(item.fd, COPY_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > item.maximum:
                    fail("Stop gate artifact exceeded its byte bound")
                scanner.consume(chunk)
                digest.update(chunk)
                yield chunk
            if (
                total != item.identity.size
                or _Identity.from_stat(os.fstat(item.fd)) != item.identity
            ):
                fail("Stop gate artifact changed while streaming")
            item.streamed_bytes = total
            item.sha256 = digest.hexdigest()
            item.consumed = True
        except StopGateIngestError:
            raise
        except OSError:
            fail("failed to stream a Stop gate artifact")

    def read_small(self, key: str) -> bytes:
        return b"".join(self._chunks(key))

    def iter_lines(self, key: str) -> Iterator[bytes]:
        pending = bytearray()
        for chunk in self._chunks(key):
            pending.extend(chunk)
            while True:
                index = pending.find(b"\n")
                if index < 0:
                    if len(pending) > MAX_JSON_LINE_BYTES:
                        fail("Stop gate JSONL line exceeded its bound")
                    break
                raw = bytes(pending[:index])
                del pending[: index + 1]
                if not raw or raw.endswith(b"\r") or len(raw) > MAX_JSON_LINE_BYTES:
                    fail("Stop gate JSONL framing differs")
                yield raw
        if pending:
            fail("Stop gate JSONL artifact lacks its final LF")

    def consume_png(self) -> ScreenshotEvidence:
        validator = _PngValidator()
        for chunk in self._chunks("screenshot"):
            validator.consume(chunk)
        validator.finish()
        byte_count, digest = self.evidence("screenshot")
        return ScreenshotEvidence(self.path("screenshot"), byte_count, digest)

    def evidence(self, key: str) -> tuple[int, str]:
        item = self.files.get(key)
        if item is None or not item.consumed or item.sha256 is None:
            fail("Stop gate artifact was not fully consumed")
        return item.streamed_bytes, item.sha256

    def path(self, key: str) -> Path:
        in_browser, name = FILE_LAYOUT[key]
        return (
            self.root_path / "browser" / name if in_browser else self.root_path / name
        )

    def seal(self) -> None:
        if self.closed or self.sealed:
            fail("Stop gate snapshot cannot be sealed in its current state")
        if any(not item.consumed for item in self.files.values()):
            fail("not every Stop gate artifact was consumed")
        assert self.root_identity is not None
        assert self.browser_identity is not None
        try:
            if (
                frozenset(os.listdir(self.root_fd)) != ROOT_FILES
                or frozenset(os.listdir(self.browser_fd)) != BROWSER_FILES
                or _Identity.from_stat(os.fstat(self.root_fd)) != self.root_identity
                or _entry_identity(self.parent_fd, self.root_path.name)
                != self.root_identity
                or _Identity.from_stat(os.fstat(self.browser_fd))
                != self.browser_identity
                or _entry_identity(self.root_fd, "browser") != self.browser_identity
            ):
                fail("Stop gate directory layout or identity changed")
            for item in self.files.values():
                if (
                    _Identity.from_stat(os.fstat(item.fd)) != item.identity
                    or _entry_identity(item.parent_fd, item.name) != item.identity
                ):
                    fail("Stop gate artifact identity changed before seal")
                os.lseek(item.fd, 0, os.SEEK_SET)
                digest = hashlib.sha256()
                scanner = _SecretScanner(self.forbidden_values)
                total = 0
                while True:
                    chunk = os.read(item.fd, COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    digest.update(chunk)
                    scanner.consume(chunk)
                if (
                    total != item.streamed_bytes
                    or digest.hexdigest() != item.sha256
                    or _Identity.from_stat(os.fstat(item.fd)) != item.identity
                    or _entry_identity(item.parent_fd, item.name) != item.identity
                ):
                    fail("Stop gate artifact hash or identity changed at seal")
            if (
                _Identity.from_stat(os.fstat(self.root_fd)) != self.root_identity
                or _Identity.from_stat(os.fstat(self.browser_fd))
                != self.browser_identity
            ):
                fail("Stop gate directory changed during final hashing")
            self.sealed = True
        except StopGateIngestError:
            raise
        except OSError:
            fail("failed to seal the Stop gate bundle")

    def close(self) -> None:
        if self.closed:
            return
        pending: StopGateIngestError | None = None
        for item in self.files.values():
            try:
                _safe_close(item.fd)
            except StopGateIngestError as error:
                pending = error
        self.files.clear()
        for descriptor in (self.browser_fd, self.root_fd, self.parent_fd):
            try:
                _safe_close(descriptor)
            except StopGateIngestError as error:
                pending = error
        self.browser_fd = self.root_fd = self.parent_fd = -1
        self.closed = True
        if pending is not None:
            raise pending


class _StableSource:
    def __init__(
        self,
        path: Path,
        label: str,
        maximum: int,
        expected_sha256: str,
        forbidden_values: tuple[bytes, ...],
    ) -> None:
        self.path = Path(os.path.abspath(path))
        self.label = label
        self.maximum = maximum
        self.forbidden_values = forbidden_values
        self.parent_fd = -1
        self.fd = -1
        self.identity: _Identity | None = None
        self.raw = b""
        self.sha256 = ""
        if SHA256_RE.fullmatch(expected_sha256) is None:
            fail("Stop source hash binding syntax differs")
        try:
            self.parent_fd = os.open(self.path.parent, _directory_flags())
            entry = _entry_identity(self.parent_fd, self.path.name)
            if (
                not stat.S_ISREG(entry.mode)
                or entry.links != 1
                or entry.size < 1
                or entry.size > maximum
            ):
                fail("bound Stop source is not one bounded regular file")
            self.fd = os.open(self.path.name, _file_flags(), dir_fd=self.parent_fd)
            opened = _Identity.from_stat(os.fstat(self.fd))
            if opened != entry:
                fail("bound Stop source identity changed while opening")
            self.identity = opened
            self.raw, self.sha256 = self._snapshot()
            if self.sha256 != expected_sha256:
                fail("bound Stop source hash differs")
        except StopGateIngestError:
            self.close()
            raise
        except OSError:
            self.close()
            fail("failed to open a bound Stop source without following links")

    def _snapshot(self) -> tuple[bytes, str]:
        assert self.identity is not None
        try:
            if _Identity.from_stat(os.fstat(self.fd)) != self.identity:
                fail("bound Stop source changed before reading")
            os.lseek(self.fd, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            scanner = _SecretScanner(self.forbidden_values)
            total = 0
            while True:
                chunk = os.read(self.fd, COPY_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > self.maximum:
                    fail("bound Stop source exceeded its byte bound")
                scanner.consume(chunk)
                digest.update(chunk)
                chunks.append(chunk)
            if (
                total != self.identity.size
                or _Identity.from_stat(os.fstat(self.fd)) != self.identity
            ):
                fail("bound Stop source changed while reading")
            return b"".join(chunks), digest.hexdigest()
        except StopGateIngestError:
            raise
        except OSError:
            fail("failed to stream a bound Stop source")

    def seal(self) -> None:
        assert self.identity is not None
        if (
            _Identity.from_stat(os.fstat(self.fd)) != self.identity
            or _entry_identity(self.parent_fd, self.path.name) != self.identity
        ):
            fail("bound Stop source identity changed before seal")
        raw, digest = self._snapshot()
        if raw != self.raw or digest != self.sha256:
            fail("bound Stop source bytes changed before seal")
        if (
            _Identity.from_stat(os.fstat(self.fd)) != self.identity
            or _entry_identity(self.parent_fd, self.path.name) != self.identity
        ):
            fail("bound Stop source identity changed during seal")

    def close(self) -> None:
        pending: StopGateIngestError | None = None
        for descriptor in (self.fd, self.parent_fd):
            try:
                _safe_close(descriptor)
            except StopGateIngestError as error:
                pending = error
        self.fd = self.parent_fd = -1
        if pending is not None:
            raise pending


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} SHA-256 syntax differs")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail(f"{label} is invalid")
    return value


def _decimal(value: Any, label: str) -> int:
    if (
        type(value) is not str
        or not value.isascii()
        or not value.isdecimal()
        or len(value) > 32
    ):
        fail(f"{label} is not a canonical decimal timestamp")
    parsed = int(value, 10)
    if value != str(parsed):
        fail(f"{label} representation differs")
    return parsed


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        fail(f"{label} fields differ")
    return cast(dict[str, Any], value)


def _strict_object(gate: Any, raw: bytes, label: str) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], gate.strict_json_object(raw, label))
    except Exception as error:
        raise StopGateIngestError(f"{label} is invalid") from error


def _document(gate: Any, raw: bytes, label: str) -> dict[str, Any]:
    if not raw.endswith(b"\n") or raw.endswith(b"\r\n") or raw.count(b"\n") != 1:
        fail(f"{label} framing differs")
    value = _strict_object(gate, raw[:-1], label)
    try:
        canonical = gate.compact_json(value) + b"\n"
    except Exception as error:
        raise StopGateIngestError(f"{label} cannot be encoded") from error
    if canonical != raw:
        fail(f"{label} is not canonical producer JSON")
    return value


def _normalize_url(raw: str) -> str:
    if type(raw) is not str:
        fail("OpenWebUI URL binding is not text")
    try:
        value = urllib.parse.urlsplit(raw)
    except ValueError:
        fail("OpenWebUI URL binding is invalid")
    if (
        value.scheme not in {"http", "https"}
        or not value.netloc
        or value.username is not None
        or value.password is not None
        or value.path not in {"", "/"}
        or value.query
        or value.fragment
    ):
        fail("OpenWebUI URL binding is not a credential-free origin")
    return urllib.parse.urlunsplit((value.scheme, value.netloc, "", "", ""))


def _source_specs(bindings: StopGateInputBindings) -> dict[str, tuple[Path, str]]:
    return {
        "gate": (bindings.gate_source, bindings.gate_source_sha256),
        "browser": (bindings.browser_script, bindings.browser_script_sha256),
    }


def _validate_bindings(bindings: StopGateInputBindings) -> None:
    if not isinstance(bindings, StopGateInputBindings):
        fail("Stop gate input bindings have the wrong type")
    specs = _source_specs(bindings)
    if any(
        not isinstance(path, os.PathLike)
        or type(digest) is not str
        or SHA256_RE.fullmatch(digest) is None
        for path, digest in specs.values()
    ):
        fail("Stop gate source path or hash binding differs")
    gate_path = Path(os.path.abspath(bindings.gate_source))
    browser_path = Path(os.path.abspath(bindings.browser_script))
    if (
        gate_path.name != "run-openwebui-stop-gate.py"
        or browser_path
        != gate_path.parent.parent / "deploy/openwebui/browser-stop-smoke.cjs"
    ):
        fail("Stop gate source layout differs")
    match = (
        CONTENT_IMAGE_RE.fullmatch(bindings.browser_image_reference)
        if type(bindings.browser_image_reference) is str
        else None
    )
    if (
        match is None
        or type(bindings.browser_image_content_id) is not str
        or bindings.browser_image_content_id != f"sha256:{match.group(1)}"
    ):
        fail("Stop browser image content binding differs")
    _normalize_url(bindings.openwebui_url)
    if (
        type(bindings.service_unit) is not str
        or SERVICE_RE.fullmatch(bindings.service_unit) is None
        or bindings.service_unit != CAMPAIGN_SERVICE_UNIT
    ):
        fail("Stop service unit binding differs")
    if (
        type(bindings.service_user) is not str
        or not bindings.service_user
        or len(bindings.service_user) > 128
        or "\0" in bindings.service_user
        or type(bindings.control_group) is not str
        or not bindings.control_group.startswith("/")
        or len(bindings.control_group) > 4096
        or "\0" in bindings.control_group
        or type(bindings.boot_id) is not str
        or BOOT_ID_RE.fullmatch(bindings.boot_id) is None
    ):
        fail("Stop service text identity binding differs")
    for value, label, minimum in (
        (bindings.gateway_pid, "gateway PID", 1),
        (bindings.gateway_starttime_ticks, "gateway starttime", 1),
        (bindings.worker_pid, "worker PID", 1),
        (bindings.worker_starttime_ticks, "worker starttime", 1),
        (bindings.restart_count, "restart count", 0),
        (bindings.uid, "uid", 0),
        (bindings.gid, "gid", 0),
    ):
        if type(value) is not int or value < minimum:
            fail(f"Stop {label} binding differs")
    if bindings.gateway_pid == bindings.worker_pid:
        fail("Stop gateway and worker PID bindings overlap")
    if type(bindings.forbidden_values) is not tuple or not bindings.forbidden_values:
        fail("Stop API secret binding is absent or mutable")
    for secret in bindings.forbidden_values:
        if (
            type(secret) is not bytes
            or not 16 <= len(secret) <= 4096
            or b"\0" in secret
        ):
            fail("Stop API secret binding syntax differs")


def _load_gate(source: _StableSource) -> tuple[Any, str]:
    module_name = f"_ullm_stop_gate_ingest_{os.getpid()}_{id(source):x}"
    module = types.ModuleType(module_name)
    module.__file__ = os.fspath(source.path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        code = compile(source.raw, os.fspath(source.path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
        if (
            module.GATE_SCHEMA != "ullm.openwebui.stop_gate.v1"
            or module.BROWSER_SCHEMA != "ullm.openwebui.stop_smoke.v1"
            or module.BROWSER_CASE != CAMPAIGN_CASE
            or module.FINAL_ACTIONS
            != (
                "navigate",
                "select_model",
                "submit_chat",
                "wait_visible",
                "click_stop",
                "wait_ready",
                "submit_chat",
                "wait_visible",
                "wait_ready",
            )
        ):
            fail("bound Stop gate public contract differs")
        return module, module_name
    except StopGateIngestError:
        sys.modules.pop(module_name, None)
        raise
    except Exception as error:
        sys.modules.pop(module_name, None)
        raise StopGateIngestError("failed to load the bound Stop gate") from error


def _campaign_claim_factory() -> Callable[
    [bytes, str, str], BundleLifecycleClaimProtocol
]:
    module_name = "sq8_openwebui_campaign"
    path = Path(__file__).with_name("sq8_openwebui_campaign.py")
    module = sys.modules.get(module_name)
    if module is not None:
        module_file = getattr(module, "__file__", None)
        if module_file is None or Path(module_file).resolve() != path.resolve():
            fail("campaign journal contract module identity differs")
    else:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            fail("campaign journal contract module cannot be loaded")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            sys.modules.pop(module_name, None)
            raise StopGateIngestError(
                "campaign journal contract module failed to load"
            ) from error
    factory = getattr(module, "BundleLifecycleClaim", None)
    if (
        not callable(factory)
        or getattr(module, "SERVICE_UNIT", None) != CAMPAIGN_SERVICE_UNIT
    ):
        fail("campaign journal lifecycle claim contract is unavailable")
    return cast(Callable[[bytes, str, str], BundleLifecycleClaimProtocol], factory)


@dataclasses.dataclass(frozen=True)
class _BrowserEvidence:
    interim: dict[str, Any]
    final: dict[str, Any]
    action_records: tuple[dict[str, Any], ...]
    screenshot: ScreenshotEvidence
    stdout_bytes: int
    stdout_sha256: str
    browser_summary_bytes: int
    browser_summary_sha256: str
    socket_event_count: int
    nonce: str
    click_completed_ns: int
    control_requested_ns: int
    control_observed_ns: int
    control_content_sha256: str
    recovery_submit_ns: int
    recovery_done_ns: int


def _browser_guard(gate: Any, bindings: StopGateInputBindings) -> Any:
    return gate.SecretGuard(
        [
            *bindings.forbidden_values,
            _normalize_url(bindings.openwebui_url).encode("utf-8"),
            gate.STOP_PROMPT.encode("utf-8"),
            gate.RECOVERY_PROMPT.encode("utf-8"),
            gate.RECOVERY_MARKER.encode("utf-8"),
        ]
    )


def _validate_browser(
    gate: Any,
    snapshot: _BundleSnapshot,
    bindings: StopGateInputBindings,
) -> _BrowserEvidence:
    lines: list[tuple[bytes, dict[str, Any]]] = []
    for raw in snapshot.iter_lines("browser_stdout"):
        if len(lines) >= 2:
            fail("Stop browser stdout exceeds its exact two records")
        lines.append((raw, _strict_object(gate, raw, "Stop browser stdout")))
    if len(lines) != 2:
        fail("Stop browser stdout record count differs from two")
    interim_raw, interim = lines[0]
    final_raw, final = lines[1]
    browser_summary_raw = snapshot.read_small("browser_summary")
    if browser_summary_raw != final_raw + b"\n":
        fail("Stop browser summary file differs from final stdout")
    screenshot = snapshot.consume_png()
    guard = _browser_guard(gate, bindings)
    try:
        nonce = cast(str, gate.validate_interim(interim, guard))
        producer = cast(
            dict[str, Any],
            gate.validate_final_browser(
                final,
                interim,
                final_raw,
                snapshot.path("browser_summary"),
                snapshot.path("screenshot"),
                guard,
            ),
        )
    except Exception as error:
        raise StopGateIngestError("Stop browser evidence validation failed") from error
    stdout_bytes, stdout_sha256 = snapshot.evidence("browser_stdout")
    browser_summary_bytes, browser_summary_sha256 = snapshot.evidence("browser_summary")
    if producer != {
        "action_count": 9,
        "socket_event_count": len(final["socket_events"]),
        "screenshot_bytes": screenshot.bytes,
        "screenshot_sha256": screenshot.sha256,
        "browser_summary_sha256": browser_summary_sha256,
    }:
        fail("Stop browser producer reconstruction differs")
    actions = final.get("browser_actions")
    if type(actions) is not list or len(actions) != 9:
        fail("Stop browser action count differs")
    normalized_url = _normalize_url(bindings.openwebui_url)
    navigation_url = (
        f"{normalized_url}/?temporary-chat=true&models={gate.MODEL_ID}"
    ).encode("utf-8")
    if actions[0].get("input_sha256") != _sha256(navigation_url):
        fail("Stop browser navigation digest differs from its URL binding")
    action_records: list[dict[str, Any]] = []
    prior_completed = -1
    for action_index, action in enumerate(actions):
        fields = copy.deepcopy(cast(dict[str, Any], action))
        fields["started_monotonic_ns"] = _decimal(
            fields["started_monotonic_ns"], "Stop browser action start"
        )
        fields["completed_monotonic_ns"] = _decimal(
            fields["completed_monotonic_ns"], "Stop browser action completion"
        )
        if fields["started_monotonic_ns"] < prior_completed:
            fail("Stop browser converted action order regressed")
        prior_completed = cast(int, fields["completed_monotonic_ns"])
        action_records.append(
            {
                "record_type": "browser_action",
                "phase": CAMPAIGN_PHASE,
                "case_id": CAMPAIGN_CASE if action_index < 6 else RECOVERY_CASE,
                "fields": fields,
            }
        )
    correlation = cast(dict[str, Any], final["socket_correlation"])
    recovery = cast(dict[str, Any], correlation["recovery"])
    control = cast(dict[str, Any], final["gateway_release_control"])
    if final["screenshot"] != {
        "screenshot_file": "browser/openwebui-stop-before.png",
        "screenshot_bytes": screenshot.bytes,
        "screenshot_sha256": screenshot.sha256,
    }:
        fail("Stop browser screenshot object differs from its FD evidence")
    return _BrowserEvidence(
        interim,
        final,
        tuple(action_records),
        screenshot,
        stdout_bytes,
        stdout_sha256,
        browser_summary_bytes,
        browser_summary_sha256,
        len(cast(list[Any], final["socket_events"])),
        nonce,
        _decimal(correlation["click_completed_monotonic_ns"], "Stop click completion"),
        _decimal(control["requested_monotonic_ns"], "Stop control request"),
        _decimal(control["observed_monotonic_ns"], "Stop control observation"),
        cast(str, control["content_sha256"]),
        _decimal(recovery["submit_completed_monotonic_ns"], "recovery submit"),
        _decimal(recovery["done_observed_monotonic_ns"], "recovery done"),
    )


@dataclasses.dataclass(frozen=True)
class _LifecycleEvidence:
    machine: Any
    events: tuple[dict[str, Any], ...]
    claims: tuple[BundleLifecycleClaimProtocol, ...]
    observer_bytes: int
    observer_sha256: str
    journal_bytes: int
    journal_sha256: str
    target_cancel_to_release_ns: int


def _validate_trace_shape(trace: Any, *, cancelled: bool) -> None:
    expected = (
        "request_admitted",
        "request_started",
        "request_progress",
        "request_first_token",
        *(("request_cancel_requested",) if cancelled else ()),
        "request_released",
    )
    names = tuple(event["event"] for event in trace.events)
    if names != expected:
        fail("Stop lifecycle trace event shape differs")
    admitted, started, progress = trace.events[:3]
    released = trace.events[-1]
    prompt = admitted["prompt_tokens"]
    if (
        started["prompt_tokens"] != prompt
        or progress["prompt_tokens"] != prompt
        or progress["processed_prompt_tokens"] != prompt
        or released["prompt_tokens"] != prompt
        or released["admit_to_release_ns"]
        != released["admit_to_start_ns"] + released["start_to_release_ns"]
        or released["admit_to_start_ns"] != started["admit_to_start_ns"]
    ):
        fail("Stop lifecycle prompt or duration reconstruction differs")
    if cancelled:
        cancellation = trace.events[-2]
        if (
            released["outcome"] != "cancelled"
            or released["cancel_reason"] != "client_disconnect"
            or cancellation["reason"] != "client_disconnect"
            or cancellation["admit_to_cancel_ns"] > released["admit_to_release_ns"]
            or released["completion_tokens"] < 1
            or released["observed_monotonic_ns"] - cancellation["observed_monotonic_ns"]
            > 5_000_000_000
        ):
            fail("Stop cancelled lifecycle result or deadline differs")
    elif (
        released["outcome"] not in {"stop", "length"}
        or released["cancel_reason"] is not None
        or released["completion_tokens"] < 1
    ):
        fail("Stop recovery lifecycle result differs")


def _canonical_claim(record: dict[str, Any]) -> bytes:
    try:
        return json.dumps(
            {field: record[field] for field in REQUIRED_JOURNAL_FIELDS},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (KeyError, TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise StopGateIngestError(
            "Stop lifecycle required journal fields cannot be reconstructed"
        ) from error


def _validate_lifecycle(
    gate: Any,
    snapshot: _BundleSnapshot,
    bindings: StopGateInputBindings,
    browser: _BrowserEvidence,
    claim_factory: Callable[[bytes, str, str], BundleLifecycleClaimProtocol],
) -> _LifecycleEvidence:
    observer_raws: list[bytes] = []
    events: list[dict[str, Any]] = []
    machine = gate.LifecycleMachine()
    last_observed = -1
    for raw in snapshot.iter_lines("observer"):
        if len(observer_raws) >= 11:
            fail("Stop observer lifecycle count exceeds exact 11")
        try:
            event = cast(dict[str, Any], gate.validate_lifecycle_payload(raw))
            machine.consume(copy.deepcopy(event))
        except Exception as error:
            raise StopGateIngestError("Stop observer lifecycle is invalid") from error
        observed = _integer(
            event["observed_monotonic_ns"], "Stop lifecycle observation"
        )
        if observed < last_observed:
            fail("Stop lifecycle global timestamps regressed")
        last_observed = observed
        observer_raws.append(raw)
        events.append(event)
    if (
        len(observer_raws) != 11
        or len(machine.traces) != 2
        or machine.active is not None
    ):
        fail("Stop observer does not contain two complete traces and 11 events")
    target, recovery = machine.traces
    _validate_trace_shape(target, cancelled=True)
    _validate_trace_shape(recovery, cancelled=False)
    target_first = target.event("request_first_token")
    target_cancel = target.event("request_cancel_requested")
    target_release = target.event("request_released")
    recovery_admitted = recovery.event("request_admitted")
    recovery_first = recovery.event("request_first_token")
    recovery_release = recovery.event("request_released")
    actions = cast(list[dict[str, Any]], browser.final["browser_actions"])
    target_visible = _decimal(
        actions[3]["completed_monotonic_ns"], "Stop visible action completion"
    )
    recovery_ready = _decimal(
        actions[8]["completed_monotonic_ns"], "recovery ready action completion"
    )
    if (
        target_first["observed_monotonic_ns"] > target_visible
        or target_cancel["observed_monotonic_ns"] < browser.click_completed_ns
        or target_release["observed_monotonic_ns"]
        < target_cancel["observed_monotonic_ns"]
        or target_release["observed_monotonic_ns"] > browser.control_requested_ns
        or recovery_admitted["observed_monotonic_ns"]
        < target_release["observed_monotonic_ns"]
        or recovery_admitted["observed_monotonic_ns"] < browser.control_observed_ns
        or recovery_admitted["observed_monotonic_ns"] < browser.recovery_submit_ns
        or recovery_first["observed_monotonic_ns"] > browser.recovery_done_ns
        or recovery_release["observed_monotonic_ns"] > recovery_ready
        or (target.request_id, target.completion_id)
        == (recovery.request_id, recovery.completion_id)
    ):
        fail("Stop browser and lifecycle timing or identity correlation differs")
    try:
        gate.validate_gateway_traces(
            machine,
            click_completed_ns=browser.click_completed_ns,
            control_created_ns=browser.control_observed_ns,
            final=True,
        )
    except Exception as error:
        raise StopGateIngestError("Stop gateway trace validation failed") from error

    claims: list[BundleLifecycleClaimProtocol] = []
    cursors: set[str] = set()
    payloads: set[bytes] = set()
    journal_payloads: list[bytes] = []
    prior_journal = -1
    for position, raw in enumerate(snapshot.iter_lines("journal")):
        if position >= 11:
            fail("Stop service journal count exceeds exact 11")
        value = _strict_object(gate, raw, "Stop service journal record")
        if not set(REQUIRED_JOURNAL_FIELDS).issubset(value):
            fail("Stop service journal lacks a required field")
        try:
            cursor, payload = gate.validate_journal_record(
                raw,
                service=bindings.service_unit,
                main_pid=bindings.gateway_pid,
                boot_id=bindings.boot_id,
                cursors=cursors,
                lifecycle_payloads=payloads,
            )
        except Exception as error:
            raise StopGateIngestError("Stop service journal is invalid") from error
        if payload is None:
            fail("Stop service journal contains a non-lifecycle row")
        cursor = cast(str, cursor)
        payload = cast(bytes, payload)
        monotonic = _decimal(
            value["__MONOTONIC_TIMESTAMP"], "Stop journal monotonic timestamp"
        )
        pid = _decimal(value["_PID"], "Stop journal PID")
        priority = _decimal(value["PRIORITY"], "Stop journal priority")
        if (
            monotonic < prior_journal
            or priority > 7
            or pid != bindings.gateway_pid
            or position >= len(observer_raws)
            or payload != observer_raws[position]
            or monotonic * 1000 < events[position]["observed_monotonic_ns"]
            or ("_UID" in value and value["_UID"] != str(bindings.uid))
            or ("_GID" in value and value["_GID"] != str(bindings.gid))
            or (
                "_SYSTEMD_CGROUP" in value
                and value["_SYSTEMD_CGROUP"] != bindings.control_group
            )
        ):
            fail("Stop observer and authoritative journal correlation differs")
        cursors.add(cursor)
        payloads.add(payload)
        prior_journal = monotonic
        journal_payloads.append(payload)
        claims.append(
            claim_factory(
                _canonical_claim(value),
                CAMPAIGN_PHASE,
                CAMPAIGN_CASE if position < 6 else RECOVERY_CASE,
            )
        )
    if (
        len(journal_payloads) != 11
        or journal_payloads != observer_raws
        or len(cursors) != 11
        or len(claims) != 11
        or machine.max_active != 1
    ):
        fail("Stop lifecycle journal count, mapping, or concurrency differs")
    observer_bytes, observer_sha256 = snapshot.evidence("observer")
    journal_bytes, journal_sha256 = snapshot.evidence("journal")
    return _LifecycleEvidence(
        machine,
        tuple(events),
        tuple(claims),
        observer_bytes,
        observer_sha256,
        journal_bytes,
        journal_sha256,
        target_release["observed_monotonic_ns"]
        - target_cancel["observed_monotonic_ns"],
    )


def _validate_summary(
    gate: Any,
    snapshot: _BundleSnapshot,
    bindings: StopGateInputBindings,
    browser: _BrowserEvidence,
    lifecycle: _LifecycleEvidence,
) -> tuple[dict[str, Any], bytes]:
    raw = snapshot.read_small("summary")
    value = _document(gate, raw, "Stop gate summary")
    _exact(
        value,
        {"schema_version", "passed", "service", "browser", "gateway", "artifacts"},
        "Stop gate summary",
    )
    if value["schema_version"] != gate.GATE_SCHEMA or type(value["passed"]) is not bool:
        fail("Stop gate summary schema or verdict representation differs")
    service = _exact(
        value["service"],
        {"unit_sha256", "main_pid_sha256", "user_uid_sha256", "restart_count"},
        "Stop service summary",
    )
    if service != {
        "unit_sha256": _sha256(bindings.service_unit.encode("utf-8")),
        "main_pid_sha256": _sha256(str(bindings.gateway_pid).encode("ascii")),
        "user_uid_sha256": _sha256(str(bindings.uid).encode("ascii")),
        "restart_count": bindings.restart_count,
    }:
        fail("Stop service summary differs from its bindings")
    browser_summary = _exact(
        value["browser"],
        {
            "image_sha256",
            "image_content_digest",
            "script_sha256",
            "action_count",
            "socket_event_count",
            "screenshot_bytes",
            "screenshot_sha256",
            "browser_summary_sha256",
            "stdout_lines",
            "stdout_sha256",
            "stderr_bytes",
            "stderr_sha256",
        },
        "Stop browser summary",
    )
    if browser_summary != {
        "image_sha256": _sha256(bindings.browser_image_reference.encode("utf-8")),
        "image_content_digest": bindings.browser_image_content_id,
        "script_sha256": bindings.browser_script_sha256,
        "action_count": 9,
        "socket_event_count": browser.socket_event_count,
        "screenshot_bytes": browser.screenshot.bytes,
        "screenshot_sha256": browser.screenshot.sha256,
        "browser_summary_sha256": browser.browser_summary_sha256,
        "stdout_lines": 2,
        "stdout_sha256": browser.stdout_sha256,
        "stderr_bytes": 0,
        "stderr_sha256": EMPTY_SHA256,
    }:
        fail("Stop browser summary differs from raw browser evidence")
    target, recovery = lifecycle.machine.traces
    target_release = target.event("request_released")
    recovery_release = recovery.event("request_released")
    gateway = _exact(
        value["gateway"],
        {
            "request_count",
            "maximum_active_requests",
            "cancel_reason",
            "target_outcome",
            "recovery_outcome",
            "target_request_sha256",
            "target_completion_sha256",
            "recovery_request_sha256",
            "recovery_completion_sha256",
            "control_content_sha256",
        },
        "Stop gateway summary",
    )
    if gateway != {
        "request_count": 2,
        "maximum_active_requests": 1,
        "cancel_reason": "client_disconnect",
        "target_outcome": target_release["outcome"],
        "recovery_outcome": recovery_release["outcome"],
        "target_request_sha256": _sha256(target.request_id.encode("utf-8")),
        "target_completion_sha256": _sha256(target.completion_id.encode("utf-8")),
        "recovery_request_sha256": _sha256(recovery.request_id.encode("utf-8")),
        "recovery_completion_sha256": _sha256(recovery.completion_id.encode("utf-8")),
        "control_content_sha256": browser.control_content_sha256,
    }:
        fail("Stop gateway summary differs from raw lifecycle evidence")
    artifacts = _exact(value["artifacts"], {"observer", "journal"}, "Stop artifacts")
    observer = _exact(
        artifacts["observer"],
        {"file", "bytes", "records", "sha256"},
        "Stop observer artifact",
    )
    journal = _exact(
        artifacts["journal"],
        {
            "file",
            "bytes",
            "records",
            "sha256",
            "unique_cursors",
            "stderr_bytes",
            "stderr_sha256",
        },
        "Stop journal artifact",
    )
    if observer != {
        "file": "observer.raw.jsonl",
        "bytes": lifecycle.observer_bytes,
        "records": 11,
        "sha256": lifecycle.observer_sha256,
    } or journal != {
        "file": "service-journal.raw.jsonl",
        "bytes": lifecycle.journal_bytes,
        "records": 11,
        "sha256": lifecycle.journal_sha256,
        "unique_cursors": 11,
        "stderr_bytes": 0,
        "stderr_sha256": EMPTY_SHA256,
    }:
        fail("Stop artifact summary differs from FD evidence")
    return value, raw


def _derived_view(
    gate: Any,
    bindings: StopGateInputBindings,
    sources: dict[str, _StableSource],
    browser: _BrowserEvidence,
    lifecycle: _LifecycleEvidence,
    summary_raw: bytes,
) -> dict[str, Any]:
    target, recovery = lifecycle.machine.traces
    target_release = target.event("request_released")
    recovery_release = recovery.event("request_released")
    view = {
        "schema_version": VIEW_SCHEMA,
        "browser_case": CAMPAIGN_CASE,
        "browser_action_count": 9,
        "browser_socket_event_count": browser.socket_event_count,
        "lifecycle_event_count": 11,
        "request_count": 2,
        "maximum_active_requests": 1,
        "component_summary_sha256": _sha256(summary_raw),
        "screenshot": {
            "file": "browser/openwebui-stop-before.png",
            "bytes": browser.screenshot.bytes,
            "sha256": browser.screenshot.sha256,
        },
        "source_bindings": {
            "gate_source_sha256": sources["gate"].sha256,
            "browser_script_sha256": sources["browser"].sha256,
            "browser_image_reference_sha256": _sha256(
                bindings.browser_image_reference.encode("utf-8")
            ),
            "browser_image_content_id_sha256": _sha256(
                bindings.browser_image_content_id.encode("ascii")
            ),
            "openwebui_url_sha256": _sha256(
                _normalize_url(bindings.openwebui_url).encode("utf-8")
            ),
            "service_unit_sha256": _sha256(bindings.service_unit.encode("utf-8")),
            "service_user_sha256": _sha256(bindings.service_user.encode("utf-8")),
            "boot_id_sha256": _sha256(bindings.boot_id.encode("ascii")),
            "control_group_sha256": _sha256(bindings.control_group.encode("utf-8")),
            "gateway_pid_sha256": _sha256(str(bindings.gateway_pid).encode("ascii")),
            "gateway_starttime_sha256": _sha256(
                str(bindings.gateway_starttime_ticks).encode("ascii")
            ),
            "worker_pid_sha256": _sha256(str(bindings.worker_pid).encode("ascii")),
            "worker_starttime_sha256": _sha256(
                str(bindings.worker_starttime_ticks).encode("ascii")
            ),
            "uid_sha256": _sha256(str(bindings.uid).encode("ascii")),
            "gid_sha256": _sha256(str(bindings.gid).encode("ascii")),
            "restart_count": bindings.restart_count,
        },
        "browser_evidence": {
            "stdout_bytes": browser.stdout_bytes,
            "stdout_sha256": browser.stdout_sha256,
            "summary_bytes": browser.browser_summary_bytes,
            "summary_sha256": browser.browser_summary_sha256,
        },
        "gateway_evidence": {
            "target_request_sha256": _sha256(target.request_id.encode("utf-8")),
            "target_completion_sha256": _sha256(target.completion_id.encode("utf-8")),
            "target_outcome": target_release["outcome"],
            "cancel_reason": target_release["cancel_reason"],
            "cancel_to_release_ns": lifecycle.target_cancel_to_release_ns,
            "recovery_request_sha256": _sha256(recovery.request_id.encode("utf-8")),
            "recovery_completion_sha256": _sha256(
                recovery.completion_id.encode("utf-8")
            ),
            "recovery_outcome": recovery_release["outcome"],
            "target_reset_complete": target_release["reset_complete"],
            "recovery_reset_complete": recovery_release["reset_complete"],
        },
        "raw_artifacts": {
            "observer_bytes": lifecycle.observer_bytes,
            "observer_sha256": lifecycle.observer_sha256,
            "journal_bytes": lifecycle.journal_bytes,
            "journal_sha256": lifecycle.journal_sha256,
        },
    }
    sensitive: list[bytes] = [
        *bindings.forbidden_values,
        _normalize_url(bindings.openwebui_url).encode("utf-8"),
        gate.STOP_PROMPT.encode("utf-8"),
        gate.RECOVERY_PROMPT.encode("utf-8"),
        gate.RECOVERY_MARKER.encode("utf-8"),
        bindings.browser_image_reference.encode("utf-8"),
        bindings.browser_image_content_id.encode("ascii"),
        bindings.service_unit.encode("utf-8"),
        bindings.service_user.encode("utf-8"),
        bindings.boot_id.encode("ascii"),
        bindings.control_group.encode("utf-8"),
        browser.nonce.encode("ascii"),
        target.request_id.encode("utf-8"),
        target.completion_id.encode("utf-8"),
        recovery.request_id.encode("utf-8"),
        recovery.completion_id.encode("utf-8"),
    ]
    for path, _digest in _source_specs(bindings).values():
        sensitive.append(os.fspath(path).encode("utf-8", errors="strict"))
    try:
        encoded = gate.compact_json(view)
    except Exception as error:
        raise StopGateIngestError("Stop derived view cannot be encoded") from error
    scanner = _SecretScanner(tuple(item for item in sensitive if len(item) >= 4))
    scanner.consume(encoded)
    return view


def ingest_stop_gate_bundle(
    bundle: Path,
    bindings: StopGateInputBindings,
) -> StopGateIngestResult:
    """Revalidate a formal Stop gate and expose campaign-ingest records."""

    if not isinstance(bundle, os.PathLike):
        fail("Stop gate bundle path has the wrong type")
    _validate_bindings(bindings)
    sources: dict[str, _StableSource] = {}
    snapshot: _BundleSnapshot | None = None
    module_name: str | None = None
    try:
        for name, (path, digest) in _source_specs(bindings).items():
            sources[name] = _StableSource(
                path,
                f"bound Stop {name} source",
                SOURCE_LIMITS[name],
                digest,
                bindings.forbidden_values,
            )
        gate, module_name = _load_gate(sources["gate"])
        if gate.normalized_url(bindings.openwebui_url) != _normalize_url(
            bindings.openwebui_url
        ) or gate.normalized_browser_image(bindings.browser_image_reference) != (
            bindings.browser_image_reference,
            bindings.browser_image_content_id,
        ):
            fail("bound Stop URL or image normalization differs")
        claim_factory = _campaign_claim_factory()
        snapshot = _BundleSnapshot(
            bundle,
            uid=bindings.uid,
            gid=bindings.gid,
            forbidden_values=bindings.forbidden_values,
        )
        browser = _validate_browser(gate, snapshot, bindings)
        lifecycle = _validate_lifecycle(
            gate, snapshot, bindings, browser, claim_factory
        )
        _summary, summary_raw = _validate_summary(
            gate, snapshot, bindings, browser, lifecycle
        )
        view = _derived_view(gate, bindings, sources, browser, lifecycle, summary_raw)
        snapshot.seal()
        for source in sources.values():
            source.seal()
        return StopGateIngestResult(
            browser.action_records,
            lifecycle.claims,
            browser.screenshot,
            view,
        )
    except StopGateIngestError:
        raise
    except Exception as error:
        raise StopGateIngestError("Stop gate bundle ingestion failed") from error
    finally:
        if module_name is not None:
            sys.modules.pop(module_name, None)
        pending: StopGateIngestError | None = None
        if snapshot is not None:
            try:
                snapshot.close()
            except StopGateIngestError as error:
                pending = error
        for source in reversed(tuple(sources.values())):
            try:
                source.close()
            except StopGateIngestError as error:
                pending = error
        if pending is not None and sys.exc_info()[0] is None:
            raise pending


__all__ = [
    "ScreenshotEvidence",
    "StopGateIngestError",
    "StopGateIngestResult",
    "StopGateInputBindings",
    "VIEW_SCHEMA",
    "ingest_stop_gate_bundle",
]
