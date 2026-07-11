#!/usr/bin/env python3
"""Revalidate and convert one formal post-header OpenWebUI failure bundle."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import ipaddress
import json
import os
import re
import stat
import struct
import sys
import types
import zlib
from pathlib import Path
from typing import Any, Iterator, NamedTuple, NoReturn, Protocol, cast


VIEW_SCHEMA = "ullm.sq8.openwebui_failure_gate_ingest.view.v1"
CAMPAIGN_PHASE = "post_header_failure"
FAILURE_CASE = "post-header-failure"
RECOVERY_CASE = "post-header-recovery"
RESTART_PROBE = "post-header-restart-ready"
CAMPAIGN_SERVICE_UNIT = "ullm-openai.service"
SCREENSHOT_BUNDLE_PATH = "browser/post-header-failure.png"

ROOT_FILES = frozenset(
    {
        "browser",
        "fault-injection.json",
        "readiness-evidence.json",
        "service-journal.raw.jsonl",
        "summary.json",
    }
)
BROWSER_FILES = frozenset(
    {
        "browser-stdout.jsonl",
        "openwebui-failure-summary.json",
        "post-header-failure.png",
    }
)
ROOT_FILE_SPECS = {
    "fault": ("fault-injection.json", 0o600, 1 << 20),
    "readiness": ("readiness-evidence.json", 0o600, 1 << 20),
    "journal": ("service-journal.raw.jsonl", 0o600, 64 << 20),
    "summary": ("summary.json", 0o600, 1 << 20),
}
BROWSER_FILE_SPECS = {
    "browser_stdout": ("browser-stdout.jsonl", 0o600, 4 << 20),
    "browser_summary": ("openwebui-failure-summary.json", 0o400, 1 << 20),
    "screenshot": ("post-header-failure.png", 0o400, 64 << 20),
}
SOURCE_LIMITS = {"gate": 4 << 20, "hook": 4 << 20, "browser": 4 << 20}
COPY_CHUNK_BYTES = 64 << 10
MAX_JSON_LINE_BYTES = 1 << 20
MAX_JOURNAL_RECORDS = 4096

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
BOOT_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
SERVICE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@:-]{0,127}\.service\Z")
CONTENT_IMAGE_RE = re.compile(
    r"(?:(?:[A-Za-z0-9][A-Za-z0-9._/:+-]*)@)?sha256:[0-9a-f]{64}\Z"
)
NETWORK_ID_RE = re.compile(r"[0-9a-f]{64}\Z")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_PNG_CHUNK_BYTES = 64 << 20
MAX_PNG_DECODED_BYTES = 128 << 20


class FailureGateIngestError(RuntimeError):
    """A fail-closed conversion error without evidence-derived diagnostics."""


def fail(message: str) -> NoReturn:
    raise FailureGateIngestError(message)


class BundleLifecycleClaimProtocol(Protocol):
    raw: bytes
    phase: str
    case_id: str


@dataclasses.dataclass(frozen=True)
class FailureGateInputBindings:
    gate_source: Path
    gate_source_sha256: str
    hook_source: Path
    hook_source_sha256: str
    browser_source: Path
    browser_source_sha256: str
    browser_image_reference: str
    browser_image_content_digest: str
    probe_image_reference: str
    probe_image_content_digest: str
    docker_network_id: str
    docker_network_subnet: str
    docker_network_gateway: str
    service_unit: str
    service_user: str
    boot_id: str
    control_group: str
    normal_gateway_pid: int
    normal_gateway_starttime_ticks: int
    normal_worker_pid: int
    normal_worker_starttime_ticks: int
    normal_restart_count: int
    restart_gateway_pid: int
    restart_gateway_starttime_ticks: int
    restart_worker_pid: int
    restart_worker_starttime_ticks: int
    restart_restart_count: int
    uid: int
    gid: int
    forbidden_values: tuple[bytes, ...] = ()


class FailureScreenshotEvidence(NamedTuple):
    source_path: Path
    bundle_path: str
    bytes: int
    sha256: str


class FailureGateIngestResult(NamedTuple):
    browser_action_records: tuple[dict[str, Any], ...]
    fault_injection_record: dict[str, Any]
    lifecycle_claims: tuple[BundleLifecycleClaimProtocol, ...]
    restart_probe_record: dict[str, Any]
    screenshot_evidence: FailureScreenshotEvidence
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
    prefix: bytes = b""


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for failure bundle ingestion")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _file_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for failure bundle ingestion")
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


def _safe_close(descriptor: int) -> None:
    if descriptor < 0:
        return
    try:
        os.close(descriptor)
    except OSError:
        fail("failed to close a sealed failure evidence descriptor")


def _entry_identity(parent_fd: int, name: str) -> _Identity:
    try:
        return _Identity.from_stat(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        )
    except OSError:
        fail("sealed failure evidence entry is unavailable")


class _SecretScanner:
    def __init__(self, values: tuple[bytes, ...]):
        if type(values) is not tuple:
            fail("forbidden failure evidence values must be an immutable tuple")
        for value in values:
            if type(value) is not bytes or len(value) < 4:
                fail("forbidden failure evidence values must be bytes of length >= 4")
        self._values = values
        self._overlap = max((len(value) for value in values), default=1) - 1
        self._tail = b""

    def consume(self, chunk: bytes) -> None:
        combined = self._tail + chunk
        if any(value in combined for value in self._values):
            fail("failure evidence contains forbidden cleartext")
        self._tail = combined[-self._overlap :] if self._overlap else b""


class _PngValidator:
    """Incrementally verify PNG framing, CRCs, and bounded scanline decoding."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.signature_seen = False
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
            fail("failure PNG carries bytes after IEND")
        self.buffer.extend(raw)
        while True:
            if not self.signature_seen:
                if len(self.buffer) < len(PNG_SIGNATURE):
                    return
                if bytes(self.buffer[: len(PNG_SIGNATURE)]) != PNG_SIGNATURE:
                    fail("failure screenshot lacks the PNG signature")
                del self.buffer[: len(PNG_SIGNATURE)]
                self.signature_seen = True
            if self.chunk_type is None:
                if len(self.buffer) < 8:
                    return
                length = int.from_bytes(self.buffer[:4], "big")
                chunk_type = bytes(self.buffer[4:8])
                del self.buffer[:8]
                if (
                    length > MAX_PNG_CHUNK_BYTES
                    or re.fullmatch(rb"[A-Za-z]{4}", chunk_type) is None
                ):
                    fail("failure PNG chunk framing differs")
                if self.chunk_index == 0 and (chunk_type != b"IHDR" or length != 13):
                    fail("failure PNG does not begin with one exact IHDR")
                if chunk_type == b"IHDR" and self.chunk_index != 0:
                    fail("failure PNG IHDR is duplicated")
                if chunk_type == b"IEND" and (length != 0 or not self.saw_idat):
                    fail("failure PNG IEND ordering differs")
                if self.saw_idat and chunk_type != b"IDAT":
                    self.idat_closed = True
                if chunk_type == b"IDAT" and self.idat_closed:
                    fail("failure PNG IDAT chunks are not consecutive")
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
                fail("failure PNG chunk CRC differs")
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
                    fail("failure PNG carries bytes after IEND")
                return
            self.chunk_type = None
            self.chunk_index += 1

    def _validate_ihdr(self, raw: bytes) -> None:
        if len(raw) != 13:
            fail("failure PNG IHDR length differs")
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
            fail("failure PNG IHDR values differ")
        channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
        row_bytes = (width * channels * depth + 7) // 8
        expected = height * (row_bytes + 1)
        if expected > MAX_PNG_DECODED_BYTES:
            fail("failure PNG decoded size exceeds its bound")
        self.row_stride = row_bytes + 1
        self.expected_decoded_bytes = expected
        self.decompressor = zlib.decompressobj()

    def _accept_decoded(self, raw: bytes) -> None:
        if self.decoded_bytes + len(raw) > self.expected_decoded_bytes:
            fail("failure PNG decoded bytes exceed IHDR dimensions")
        first_filter = (-self.decoded_bytes) % self.row_stride
        if any(
            raw[offset] > 4 for offset in range(first_filter, len(raw), self.row_stride)
        ):
            fail("failure PNG scanline filter type differs")
        self.decoded_bytes += len(raw)

    def _decode_idat(self, raw: bytes) -> None:
        if self.decompressor is None or (self.decompressor.eof and raw):
            fail("failure PNG IDAT zlib stream ordering differs")
        pending = raw
        while pending:
            before = len(pending)
            try:
                decoded = self.decompressor.decompress(
                    pending, min(COPY_CHUNK_BYTES, self.expected_decoded_bytes + 1)
                )
            except zlib.error as error:
                raise FailureGateIngestError(
                    "failure PNG IDAT zlib stream is invalid"
                ) from error
            self._accept_decoded(decoded)
            if self.decompressor.unused_data:
                fail("failure PNG IDAT contains excess compressed data")
            pending = self.decompressor.unconsumed_tail
            if pending and len(pending) >= before and not decoded:
                fail("failure PNG IDAT decompression made no progress")

    def _finish_image_data(self) -> None:
        if self.decompressor is None:
            fail("failure PNG lacks a decodable IDAT stream")
        try:
            decoded = self.decompressor.flush(COPY_CHUNK_BYTES)
        except zlib.error as error:
            raise FailureGateIngestError(
                "failure PNG IDAT zlib finalization failed"
            ) from error
        self._accept_decoded(decoded)
        if (
            not self.decompressor.eof
            or self.decompressor.unconsumed_tail
            or self.decompressor.unused_data
            or self.decoded_bytes != self.expected_decoded_bytes
        ):
            fail("failure PNG IDAT decoded size or terminator differs")

    def finish(self) -> None:
        if (
            not self.signature_seen
            or not self.ended
            or self.chunk_type is not None
            or self.buffer
        ):
            fail("failure PNG stream is incomplete")


def _require_directory(
    identity: _Identity, *, mode: int, links: int, uid: int, gid: int
) -> None:
    if (
        not stat.S_ISDIR(identity.mode)
        or stat.S_IMODE(identity.mode) != mode
        or identity.links != links
        or identity.uid != uid
        or identity.gid != gid
    ):
        fail("failure bundle directory layout, mode, owner, or links differ")


def _require_file(
    identity: _Identity,
    *,
    mode: int,
    maximum: int,
    uid: int,
    gid: int,
) -> None:
    if (
        not stat.S_ISREG(identity.mode)
        or stat.S_IMODE(identity.mode) != mode
        or identity.links != 1
        or identity.uid != uid
        or identity.gid != gid
        or identity.size < 1
        or identity.size > maximum
    ):
        fail("failure bundle artifact layout, mode, owner, links, or size differ")


class _BundleSnapshot:
    """Open every fixed bundle entry through pinned directory descriptors."""

    def __init__(
        self,
        root: Path,
        *,
        uid: int,
        gid: int,
        forbidden_values: tuple[bytes, ...],
    ):
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
                fail("failure bundle root identity changed while opening")
            _require_directory(root, mode=0o700, links=3, uid=self.uid, gid=self.gid)
            if frozenset(os.listdir(self.root_fd)) != ROOT_FILES:
                fail("failure bundle root layout differs")
            self.browser_fd = os.open(
                "browser", _directory_flags(), dir_fd=self.root_fd
            )
            browser = _Identity.from_stat(os.fstat(self.browser_fd))
            if _entry_identity(self.root_fd, "browser") != browser:
                fail("failure browser directory identity changed while opening")
            _require_directory(browser, mode=0o700, links=2, uid=self.uid, gid=self.gid)
            if frozenset(os.listdir(self.browser_fd)) != BROWSER_FILES:
                fail("failure browser directory layout differs")
            self.root_identity = root
            self.browser_identity = browser
            for key, (name, mode, maximum) in ROOT_FILE_SPECS.items():
                self._open_file(key, name, self.root_fd, mode, maximum)
            for key, (name, mode, maximum) in BROWSER_FILE_SPECS.items():
                self._open_file(key, name, self.browser_fd, mode, maximum)
        except FailureGateIngestError:
            self.close()
            raise
        except OSError:
            self.close()
            fail("failed to open failure bundle without following links")

    def _open_file(
        self, key: str, name: str, parent_fd: int, mode: int, maximum: int
    ) -> None:
        entry = _entry_identity(parent_fd, name)
        _require_file(entry, mode=mode, maximum=maximum, uid=self.uid, gid=self.gid)
        descriptor = os.open(name, _file_flags(), dir_fd=parent_fd)
        opened = _Identity.from_stat(os.fstat(descriptor))
        if opened != entry:
            os.close(descriptor)
            fail("failure artifact identity changed while opening")
        self.files[key] = _OpenedFile(key, name, parent_fd, descriptor, opened, maximum)

    def __enter__(self) -> _BundleSnapshot:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _chunks(self, key: str) -> Iterator[bytes]:
        if self.closed or self.sealed:
            fail("failure bundle snapshot is no longer readable")
        item = self.files.get(key)
        if item is None or item.consumed:
            fail("failure artifact was requested outside its fixed schedule")
        try:
            if _Identity.from_stat(os.fstat(item.fd)) != item.identity:
                fail("failure artifact changed before streaming")
            os.lseek(item.fd, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            scanner = _SecretScanner(self.forbidden_values)
            png = _PngValidator() if key == "screenshot" else None
            prefix = bytearray()
            total = 0
            while True:
                chunk = os.read(item.fd, COPY_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > item.maximum:
                    fail("failure artifact exceeded its streaming bound")
                digest.update(chunk)
                scanner.consume(chunk)
                if png is not None:
                    png.consume(chunk)
                if len(prefix) < 16:
                    prefix.extend(chunk[: 16 - len(prefix)])
                yield chunk
            if (
                total != item.identity.size
                or _Identity.from_stat(os.fstat(item.fd)) != item.identity
            ):
                fail("failure artifact changed while streaming")
            if png is not None:
                png.finish()
            item.consumed = True
            item.streamed_bytes = total
            item.sha256 = digest.hexdigest()
            item.prefix = bytes(prefix)
        except FailureGateIngestError:
            raise
        except OSError:
            fail("failed to stream a pinned failure artifact")

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
                        fail("failure JSONL line exceeds its bound")
                    break
                raw = bytes(pending[:index])
                del pending[: index + 1]
                if not raw or raw.endswith(b"\r") or len(raw) > MAX_JSON_LINE_BYTES:
                    fail("failure JSONL framing differs")
                yield raw
        if pending:
            fail("failure JSONL artifact lacks its final LF")

    def stream(self, key: str) -> None:
        for _chunk in self._chunks(key):
            pass

    def evidence(self, key: str) -> tuple[int, str, bytes]:
        item = self.files.get(key)
        if item is None or not item.consumed or item.sha256 is None:
            fail("failure artifact has not been fully streamed")
        return item.streamed_bytes, item.sha256, item.prefix

    def seal(self) -> None:
        if (
            self.closed
            or self.sealed
            or any(not item.consumed for item in self.files.values())
        ):
            fail("failure bundle cannot be sealed before complete consumption")
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
                fail("failure bundle directory identity changed before sealing")
            for item in self.files.values():
                if (
                    _Identity.from_stat(os.fstat(item.fd)) != item.identity
                    or _entry_identity(item.parent_fd, item.name) != item.identity
                ):
                    fail("failure artifact identity changed before sealing")
                os.lseek(item.fd, 0, os.SEEK_SET)
                digest = hashlib.sha256()
                scanner = _SecretScanner(self.forbidden_values)
                png = _PngValidator() if item.key == "screenshot" else None
                total = 0
                while True:
                    chunk = os.read(item.fd, COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    digest.update(chunk)
                    scanner.consume(chunk)
                    if png is not None:
                        png.consume(chunk)
                if png is not None:
                    png.finish()
                if (
                    total != item.streamed_bytes
                    or digest.hexdigest() != item.sha256
                    or _Identity.from_stat(os.fstat(item.fd)) != item.identity
                    or _entry_identity(item.parent_fd, item.name) != item.identity
                ):
                    fail("failure artifact hash or identity changed at seal")
            self.sealed = True
        except FailureGateIngestError:
            raise
        except OSError:
            fail("failed to seal failure bundle")

    def close(self) -> None:
        if self.closed:
            return
        pending: FailureGateIngestError | None = None
        for item in self.files.values():
            try:
                _safe_close(item.fd)
            except FailureGateIngestError as error:
                pending = error
        self.files.clear()
        for descriptor in (self.browser_fd, self.root_fd, self.parent_fd):
            try:
                _safe_close(descriptor)
            except FailureGateIngestError as error:
                pending = error
        self.browser_fd = self.root_fd = self.parent_fd = -1
        self.closed = True
        if pending is not None:
            raise pending


class _StableSource:
    def __init__(
        self,
        path: Path,
        *,
        label: str,
        maximum: int,
        expected_sha256: str,
        forbidden_values: tuple[bytes, ...],
    ):
        self.path = Path(os.path.abspath(path))
        self.label = label
        self.maximum = maximum
        self.expected_sha256 = expected_sha256
        self.forbidden_values = forbidden_values
        self.parent_fd = -1
        self.fd = -1
        self.identity: _Identity | None = None
        self.raw = b""
        self.sha256 = ""
        self._open()

    def _open(self) -> None:
        try:
            self.parent_fd = os.open(self.path.parent, _directory_flags())
            entry = _entry_identity(self.parent_fd, self.path.name)
            if not stat.S_ISREG(entry.mode) or entry.links != 1 or entry.size < 1:
                fail("bound failure source is not one regular linked file")
            if entry.size > self.maximum:
                fail("bound failure source exceeds its size limit")
            self.fd = os.open(self.path.name, _file_flags(), dir_fd=self.parent_fd)
            opened = _Identity.from_stat(os.fstat(self.fd))
            if opened != entry:
                fail("bound failure source identity changed while opening")
            self.identity = opened
            self.raw, self.sha256 = self._snapshot()
            if self.sha256 != self.expected_sha256:
                fail("bound failure source hash differs")
        except FailureGateIngestError:
            self.close()
            raise
        except OSError:
            self.close()
            fail("failed to pin a failure source without following links")

    def _snapshot(self) -> tuple[bytes, str]:
        assert self.identity is not None
        try:
            if _Identity.from_stat(os.fstat(self.fd)) != self.identity:
                fail("bound failure source changed before streaming")
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
                    fail("bound failure source exceeded its streaming bound")
                scanner.consume(chunk)
                digest.update(chunk)
                chunks.append(chunk)
            if (
                total != self.identity.size
                or _Identity.from_stat(os.fstat(self.fd)) != self.identity
            ):
                fail("bound failure source changed while streaming")
            return b"".join(chunks), digest.hexdigest()
        except FailureGateIngestError:
            raise
        except OSError:
            fail("failed to stream a bound failure source")

    def seal(self) -> None:
        assert self.identity is not None
        if (
            _Identity.from_stat(os.fstat(self.fd)) != self.identity
            or _entry_identity(self.parent_fd, self.path.name) != self.identity
        ):
            fail("bound failure source identity changed before sealing")
        raw, digest = self._snapshot()
        if raw != self.raw or digest != self.sha256:
            fail("bound failure source bytes changed before sealing")
        if (
            _Identity.from_stat(os.fstat(self.fd)) != self.identity
            or _entry_identity(self.parent_fd, self.path.name) != self.identity
        ):
            fail("bound failure source identity changed during sealing")

    def close(self) -> None:
        pending: FailureGateIngestError | None = None
        for descriptor in (self.fd, self.parent_fd):
            try:
                _safe_close(descriptor)
            except FailureGateIngestError as error:
                pending = error
        self.fd = self.parent_fd = -1
        if pending is not None:
            raise pending


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _validate_bindings(bindings: FailureGateInputBindings) -> None:
    if not isinstance(bindings, FailureGateInputBindings):
        fail("failure gate input bindings have the wrong type")
    for path in (bindings.gate_source, bindings.hook_source, bindings.browser_source):
        if not isinstance(path, os.PathLike):
            fail("failure gate source path has the wrong type")
    for digest in (
        bindings.gate_source_sha256,
        bindings.hook_source_sha256,
        bindings.browser_source_sha256,
    ):
        if type(digest) is not str or SHA256_RE.fullmatch(digest) is None:
            fail("failure gate source hash syntax differs")
    for reference in (bindings.browser_image_reference, bindings.probe_image_reference):
        if type(reference) is not str or CONTENT_IMAGE_RE.fullmatch(reference) is None:
            fail("failure gate image reference is not immutable")
    for content in (
        bindings.browser_image_content_digest,
        bindings.probe_image_content_digest,
    ):
        if (
            type(content) is not str
            or not content.startswith("sha256:")
            or SHA256_RE.fullmatch(content[7:]) is None
        ):
            fail("failure gate image content digest syntax differs")
    if (
        type(bindings.docker_network_id) is not str
        or NETWORK_ID_RE.fullmatch(bindings.docker_network_id) is None
    ):
        fail("failure gate Docker network ID syntax differs")
    try:
        network = ipaddress.ip_network(bindings.docker_network_subnet, strict=True)
        gateway = ipaddress.ip_address(bindings.docker_network_gateway)
    except (TypeError, ValueError):
        fail("failure gate Docker network address syntax differs")
    if (
        network.version != 4
        or gateway.version != 4
        or gateway not in network
        or str(network) != bindings.docker_network_subnet
        or str(gateway) != bindings.docker_network_gateway
    ):
        fail("failure gate Docker network binding differs")
    if (
        type(bindings.service_unit) is not str
        or SERVICE_RE.fullmatch(bindings.service_unit) is None
        or bindings.service_unit != CAMPAIGN_SERVICE_UNIT
    ):
        fail("failure gate service unit differs from the campaign contract")
    for text_value in (bindings.service_user, bindings.control_group):
        if type(text_value) is not str or not text_value or "\0" in text_value:
            fail("failure gate service text binding differs")
    if not bindings.control_group.startswith("/"):
        fail("failure gate control group is not absolute")
    if (
        type(bindings.boot_id) is not str
        or BOOT_ID_RE.fullmatch(bindings.boot_id) is None
    ):
        fail("failure gate boot ID syntax differs")
    positive_values = (
        bindings.normal_gateway_pid,
        bindings.normal_gateway_starttime_ticks,
        bindings.normal_worker_pid,
        bindings.normal_worker_starttime_ticks,
        bindings.restart_gateway_pid,
        bindings.restart_gateway_starttime_ticks,
        bindings.restart_worker_pid,
        bindings.restart_worker_starttime_ticks,
    )
    if any(type(value) is not int or value < 1 for value in positive_values):
        fail("failure gate process epoch binding differs")
    if any(
        type(value) is not int or value < 0
        for value in (
            bindings.normal_restart_count,
            bindings.restart_restart_count,
            bindings.uid,
            bindings.gid,
        )
    ):
        fail("failure gate service numeric binding differs")
    if (
        bindings.normal_gateway_pid == bindings.normal_worker_pid
        or bindings.restart_gateway_pid == bindings.restart_worker_pid
        or bindings.normal_gateway_pid == bindings.restart_gateway_pid
        or bindings.normal_worker_pid == bindings.restart_worker_pid
        or bindings.normal_gateway_starttime_ticks
        == bindings.restart_gateway_starttime_ticks
        or bindings.normal_worker_starttime_ticks
        == bindings.restart_worker_starttime_ticks
        or bindings.restart_restart_count != bindings.normal_restart_count + 1
    ):
        fail("failure gate planned restart epoch differs")
    _SecretScanner(bindings.forbidden_values)


def _pin_sources(bindings: FailureGateInputBindings) -> dict[str, _StableSource]:
    values = {
        "gate": (bindings.gate_source, bindings.gate_source_sha256),
        "hook": (bindings.hook_source, bindings.hook_source_sha256),
        "browser": (bindings.browser_source, bindings.browser_source_sha256),
    }
    return {
        role: _StableSource(
            path,
            label=f"failure {role} source",
            maximum=SOURCE_LIMITS[role],
            expected_sha256=digest,
            forbidden_values=bindings.forbidden_values,
        )
        for role, (path, digest) in values.items()
    }


def _load_validator(source: _StableSource, role: str) -> tuple[Any, str]:
    module_name = f"_ullm_failure_ingest_{role}_{os.getpid()}_{id(source):x}"
    module = types.ModuleType(module_name)
    module.__file__ = os.fspath(source.path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        code = compile(source.raw, os.fspath(source.path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
        if role == "gate" and getattr(module, "GATE_SOURCE_RAW", None) != source.raw:
            fail("executed failure gate source differs from its pinned bytes")
        expected = {
            "gate": "ullm.openwebui.failure_gate.v1",
            "hook": "ullm.openwebui.failure_gate.v1",
        }[role]
        if getattr(module, "GATE_SCHEMA", None) != expected:
            fail("loaded failure validator contract differs")
        return module, module_name
    except FailureGateIngestError:
        sys.modules.pop(module_name, None)
        raise
    except Exception as error:
        sys.modules.pop(module_name, None)
        raise FailureGateIngestError(
            "failed to load a pinned failure validator"
        ) from error


def _claim_factory() -> Any:
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
            raise FailureGateIngestError(
                "campaign journal contract module failed to load"
            ) from error
    factory = getattr(module, "BundleLifecycleClaim", None)
    if (
        not callable(factory)
        or getattr(module, "SERVICE_UNIT", None) != CAMPAIGN_SERVICE_UNIT
    ):
        fail("campaign journal lifecycle claim contract is unavailable")
    return factory


def _strict_object(module: Any, raw: bytes, label: str) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], module.strict_json_object(raw, label))
    except Exception as error:
        raise FailureGateIngestError(f"{label} is invalid") from error


def _canonical_document(hook: Any, raw: bytes, label: str) -> dict[str, Any]:
    if not raw.endswith(b"\n") or raw == b"\n":
        fail(f"{label} framing differs")
    value = _strict_object(hook, raw[:-1], label)
    try:
        expected = hook.compact_json(value) + b"\n"
    except Exception as error:
        raise FailureGateIngestError(f"{label} cannot be canonicalized") from error
    if raw != expected:
        fail(f"{label} is not canonical JSON")
    return value


def _typed_json_equal(hook: Any, observed: Any, expected: Any, label: str) -> None:
    try:
        observed_raw = hook.compact_json(observed)
        expected_raw = hook.compact_json(expected)
    except Exception as error:
        raise FailureGateIngestError(f"{label} cannot be canonicalized") from error
    if observed_raw != expected_raw:
        fail(f"{label} differs from independently reconstructed evidence")


def _read_browser(
    snapshot: _BundleSnapshot, hook: Any
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    str,
    str,
    bytes,
]:
    browser_summary_raw = snapshot.read_small("browser_summary")
    if not browser_summary_raw.endswith(b"\n") or browser_summary_raw == b"\n":
        fail("failure browser summary framing differs")
    browser = _strict_object(hook, browser_summary_raw[:-1], "failure browser summary")
    try:
        actions = cast(list[dict[str, Any]], hook.validate_final_browser(browser))
    except Exception as error:
        raise FailureGateIngestError(
            "failure browser final validation failed"
        ) from error

    lines = list(snapshot.iter_lines("browser_stdout"))
    if len(lines) != 3:
        fail("failure browser stdout record count differs")
    records = [
        _strict_object(hook, raw, "failure browser stdout record") for raw in lines
    ]
    final_events = browser["socket_events"]
    final_target = browser["socket_correlation"]["target"]
    final_controls = browser["controls"]
    try:
        worker_nonce = hook._validate_interim(
            records[0],
            record_type="openwebui_failure_worker_kill_wait",
            action_count=4,
            final_actions=actions,
            final_events=final_events,
            final_target=final_target,
            final_controls=final_controls,
        )
        recovery_nonce = hook._validate_interim(
            records[1],
            record_type="openwebui_failure_gateway_recovery_wait",
            action_count=5,
            final_actions=actions,
            final_events=final_events,
            final_target=final_target,
            final_controls=final_controls,
        )
        hook._bind_redacted_control(
            final_controls["worker_killed"], "worker_killed", worker_nonce
        )
        hook._bind_redacted_control(
            final_controls["gateway_recovered"],
            "gateway_recovered",
            recovery_nonce,
        )
    except Exception as error:
        raise FailureGateIngestError(
            "failure browser interim validation failed"
        ) from error
    if lines[2] + b"\n" != browser_summary_raw:
        fail("failure browser stdout final record differs from its artifact")
    stdout_raw = b"".join(raw + b"\n" for raw in lines)
    return (
        browser,
        actions,
        cast(str, worker_nonce),
        cast(str, recovery_nonce),
        stdout_raw,
    )


def _browser_final(gate: Any, hook: Any, browser: dict[str, Any]) -> Any:
    correlation = browser["socket_correlation"]
    events = browser["socket_events"]

    def first_content(target: str) -> int:
        values = [
            cast(
                int,
                hook.decimal_timestamp(
                    event["observed_monotonic_ns"],
                    "failure socket content timestamp",
                ),
            )
            for event in events
            if event["correlation_target"] == target
            and type(event["content_utf8_bytes"]) is int
            and event["content_utf8_bytes"] > 0
        ]
        if not values:
            fail("failure browser content evidence is absent")
        return min(values)

    recovery = correlation["recovery"]
    controls = browser["controls"]
    return gate.BrowserFinal(
        target=copy.deepcopy(correlation["target"]),
        recovery={
            key: copy.deepcopy(recovery[key])
            for key in (
                "chat_id_utf8_bytes",
                "chat_id_sha256",
                "message_id_utf8_bytes",
                "message_id_sha256",
            )
        },
        visible_completed_ns=hook.decimal_timestamp(
            browser["browser_actions"][3]["completed_monotonic_ns"],
            "failure visible completion",
        ),
        error_observed_ns=hook.decimal_timestamp(
            correlation["error_first_observed_monotonic_ns"], "failure error"
        ),
        cancel_observed_ns=hook.decimal_timestamp(
            correlation["cancel_first_observed_monotonic_ns"], "failure cancel"
        ),
        recovery_submit_started_ns=hook.decimal_timestamp(
            browser["browser_actions"][6]["started_monotonic_ns"],
            "failure recovery submit",
        ),
        recovery_done_ns=hook.decimal_timestamp(
            recovery["done_observed_monotonic_ns"], "failure recovery done"
        ),
        first_target_content_ns=first_content("failure_target"),
        first_recovery_content_ns=first_content("recovery_target"),
        screenshot_sha256=browser["screenshot"]["screenshot_sha256"],
        action_count=len(browser["browser_actions"]),
        socket_event_count=len(events),
        recovery_control_observed_ns=hook.decimal_timestamp(
            controls["gateway_recovered"]["observed_monotonic_ns"],
            "failure recovery control",
        ),
    )


def _read_journal(
    snapshot: _BundleSnapshot,
    gate: Any,
    bindings: FailureGateInputBindings,
    browser_final: Any,
    fault: dict[str, Any],
    claim_factory: Any,
) -> tuple[Any, tuple[BundleLifecycleClaimProtocol, ...], int, int]:
    cursors: set[str] = set()
    lifecycle_payloads: set[bytes] = set()
    lifecycle_records: list[Any] = []
    lifecycle_raw_records: list[bytes] = []
    prior_journal_usec = -1
    record_count = 0
    for raw in snapshot.iter_lines("journal"):
        record_count += 1
        if record_count > MAX_JOURNAL_RECORDS:
            fail("failure service journal record count exceeds its bound")
        record = _strict_object(gate, raw, "failure service journal record")
        for field in (
            "__CURSOR",
            "__MONOTONIC_TIMESTAMP",
            "_BOOT_ID",
            "_PID",
            "_SYSTEMD_UNIT",
            "PRIORITY",
            "MESSAGE",
        ):
            if field not in record:
                fail("failure service journal lacks a required field")
        monotonic_text = record["__MONOTONIC_TIMESTAMP"]
        if type(monotonic_text) is not str or not monotonic_text.isdecimal():
            fail("failure service journal monotonic timestamp differs")
        monotonic = int(monotonic_text, 10)
        if monotonic < prior_journal_usec:
            fail("failure service journal timestamps regress globally")
        prior_journal_usec = monotonic
        pid_text = record["_PID"]
        if type(pid_text) is not str or not pid_text.isdecimal():
            fail("failure service journal PID field differs")
        if record["_SYSTEMD_UNIT"] == bindings.service_unit:
            if (
                pid_text
                not in {
                    str(bindings.normal_gateway_pid),
                    str(bindings.restart_gateway_pid),
                }
                or record.get("_UID") != str(bindings.uid)
                or record.get("_GID") != str(bindings.gid)
            ):
                fail("failure service journal process epoch or owner differs")
        elif (
            pid_text != "1"
            or record["_SYSTEMD_UNIT"] != "init.scope"
            or record.get("UNIT") != bindings.service_unit
            or record.get("SYSLOG_IDENTIFIER") != "systemd"
            or record.get("_UID") != "0"
            or record.get("_GID") != "0"
        ):
            fail("failure service manager journal identity differs")
        try:
            cursor, lifecycle = gate.validate_journal_record(
                raw,
                service=bindings.service_unit,
                boot_id=bindings.boot_id,
                cursors=cursors,
                lifecycle_payloads=lifecycle_payloads,
            )
        except Exception as error:
            raise FailureGateIngestError(
                "failure authoritative journal validation failed"
            ) from error
        cursors.add(cursor)
        if lifecycle is not None:
            lifecycle_payloads.add(lifecycle.raw)
            lifecycle_records.append(lifecycle)
            lifecycle_raw_records.append(raw)
    if record_count < 1 or len(cursors) != record_count:
        fail("failure service journal count or cursor uniqueness differs")
    try:
        evidence = gate.validate_failure_lifecycle(
            lifecycle_records,
            initial_gateway_pid=bindings.normal_gateway_pid,
            recovered_gateway_pid=bindings.restart_gateway_pid,
            fault_started_ns=fault["started_monotonic_ns"],
            fault_completed_ns=fault["completed_monotonic_ns"],
            browser=browser_final,
        )
    except Exception as error:
        raise FailureGateIngestError(
            "failure lifecycle reconstruction failed"
        ) from error
    if evidence.lifecycle_count != 10 or len(lifecycle_raw_records) != 10:
        fail("failure lifecycle count differs from the campaign contract")
    claims: list[BundleLifecycleClaimProtocol] = []
    for record, raw in zip(lifecycle_records, lifecycle_raw_records, strict=True):
        case_id = (
            FAILURE_CASE
            if record.journal_pid == bindings.normal_gateway_pid
            else RECOVERY_CASE
        )
        claims.append(claim_factory(raw, CAMPAIGN_PHASE, case_id))
    if [claim.case_id for claim in claims] != [FAILURE_CASE] * 5 + [RECOVERY_CASE] * 5:
        fail("failure lifecycle claims do not split at the restart epoch")
    return evidence, tuple(claims), record_count, len(cursors)


def _expected_summary(
    bindings: FailureGateInputBindings,
    *,
    browser: dict[str, Any],
    browser_stdout: bytes,
    screenshot_bytes: int,
    screenshot_sha256: str,
    fault_raw: bytes,
    readiness_raw: bytes,
    journal_bytes: int,
    journal_sha256: str,
    journal_records: int,
    lifecycle: Any,
    worker_nonce: str,
    recovery_nonce: str,
) -> dict[str, Any]:
    del journal_bytes, screenshot_bytes

    def control(stage: str, nonce: str) -> bytes:
        return f"ullm.openwebui.failure_control.v1:{stage}:{nonce}\n".encode("ascii")

    return {
        "schema_version": "ullm.openwebui.failure_gate.v1",
        "service": {
            "unit_sha256": _sha256(bindings.service_unit.encode("utf-8")),
            "initial_gateway_pid": bindings.normal_gateway_pid,
            "recovered_gateway_pid": bindings.restart_gateway_pid,
            "initial_worker_pid": bindings.normal_worker_pid,
            "recovered_worker_pid": bindings.restart_worker_pid,
            "initial_worker_starttime_ticks": bindings.normal_worker_starttime_ticks,
            "recovered_worker_starttime_ticks": bindings.restart_worker_starttime_ticks,
            "initial_restart_count": bindings.normal_restart_count,
            "recovered_restart_count": bindings.restart_restart_count,
            "restart_delta": 1,
            "boot_id_sha256": _sha256(bindings.boot_id.encode("ascii")),
        },
        "browser": {
            "image_reference_sha256": _sha256(
                bindings.browser_image_reference.encode("utf-8")
            ),
            "image_content_digest": bindings.browser_image_content_digest,
            "script_sha256": bindings.browser_source_sha256,
            "action_count": len(browser["browser_actions"]),
            "socket_event_count": len(browser["socket_events"]),
            "screenshot_sha256": screenshot_sha256,
            "stdout_lines": 3,
            "stdout_bytes": len(browser_stdout),
            "stdout_sha256": _sha256(browser_stdout),
            "stderr_bytes": 0,
            "stderr_sha256": _sha256(b""),
        },
        "fault": {
            "target_request_sha256": _sha256(
                lifecycle.target_request_id.encode("utf-8")
            ),
            "target_completion_sha256": _sha256(
                lifecycle.target_completion_id.encode("utf-8")
            ),
            "worker_fatal_monotonic_ns": lifecycle.worker_fatal_ns,
            "signal_to_fatal_ns": lifecycle.worker_fatal_ns
            - _canonical_fault_start(fault_raw),
            "fault_artifact_sha256": _sha256(fault_raw),
            "kill_control_sha256": _sha256(control("worker_killed", worker_nonce)),
        },
        "recovery": {
            "request_sha256": _sha256(lifecycle.recovery_request_id.encode("utf-8")),
            "completion_sha256": _sha256(
                lifecycle.recovery_completion_id.encode("utf-8")
            ),
            "admitted_monotonic_ns": lifecycle.recovery_admitted_ns,
            "released_monotonic_ns": lifecycle.recovery_released_ns,
            "outcome": "stop",
            "reset_complete": True,
            "readiness_artifact_sha256": _sha256(readiness_raw),
            "recovery_control_sha256": _sha256(
                control("gateway_recovered", recovery_nonce)
            ),
        },
        "gateway_journal": {
            "lifecycle_count": lifecycle.lifecycle_count,
            "record_count": journal_records,
            "cursor_count": journal_records,
            "raw_sha256": journal_sha256,
            "stderr_bytes": 0,
            "stderr_sha256": _sha256(b""),
        },
        "probe": {
            "image_reference_sha256": _sha256(
                bindings.probe_image_reference.encode("utf-8")
            ),
            "image_content_digest": bindings.probe_image_content_digest,
            "network_id_sha256": _sha256(bindings.docker_network_id.encode("utf-8")),
        },
        "gate_source_sha256": bindings.gate_source_sha256,
    }


def _canonical_fault_start(raw: bytes) -> int:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, ValueError):
        fail("failure fault artifact cannot be reconstructed")
    started = value.get("started_monotonic_ns") if type(value) is dict else None
    if type(started) is not int or started < 1:
        fail("failure fault start cannot be reconstructed")
    return started


def _hook_bindings(hook: Any, bindings: FailureGateInputBindings) -> Any:
    return hook.BundleBindings(
        gate_source_sha256=bindings.gate_source_sha256,
        browser_script_sha256=bindings.browser_source_sha256,
        browser_image_reference_sha256=_sha256(
            bindings.browser_image_reference.encode("utf-8")
        ),
        probe_image_reference_sha256=_sha256(
            bindings.probe_image_reference.encode("utf-8")
        ),
        service_unit_sha256=_sha256(bindings.service_unit.encode("utf-8")),
    )


def _derived_view(
    bindings: FailureGateInputBindings,
    *,
    summary_raw: bytes,
    browser: dict[str, Any],
    fault: dict[str, Any],
    lifecycle: Any,
    journal_records: int,
    journal_sha256: str,
    screenshot: FailureScreenshotEvidence,
    readiness: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": VIEW_SCHEMA,
        "phase": CAMPAIGN_PHASE,
        "cases": {"failure": FAILURE_CASE, "recovery": RECOVERY_CASE},
        "source_sha256": {
            "gate": bindings.gate_source_sha256,
            "hook": bindings.hook_source_sha256,
            "browser": bindings.browser_source_sha256,
        },
        "summary_sha256": _sha256(summary_raw),
        "service": {
            "unit": bindings.service_unit,
            "user": bindings.service_user,
            "control_group": bindings.control_group,
            "normal_gateway_pid": bindings.normal_gateway_pid,
            "normal_worker_pid": bindings.normal_worker_pid,
            "normal_restart_count": bindings.normal_restart_count,
            "restart_gateway_pid": bindings.restart_gateway_pid,
            "restart_worker_pid": bindings.restart_worker_pid,
            "restart_restart_count": bindings.restart_restart_count,
        },
        "browser": {
            "action_count": len(browser["browser_actions"]),
            "socket_event_count": len(browser["socket_events"]),
            "screenshot_file": screenshot.bundle_path,
            "screenshot_bytes": screenshot.bytes,
            "screenshot_sha256": screenshot.sha256,
        },
        "fault": {
            "target_pid": fault["target_pid"],
            "started_monotonic_ns": fault["started_monotonic_ns"],
            "completed_monotonic_ns": fault["completed_monotonic_ns"],
            "worker_fatal_monotonic_ns": lifecycle.worker_fatal_ns,
        },
        "recovery": {
            "ready_completed_monotonic_ns": readiness["recovered"][
                "completed_monotonic_ns"
            ],
            "admitted_monotonic_ns": lifecycle.recovery_admitted_ns,
            "released_monotonic_ns": lifecycle.recovery_released_ns,
        },
        "journal": {
            "record_count": journal_records,
            "lifecycle_count": lifecycle.lifecycle_count,
            "sha256": journal_sha256,
        },
    }


def ingest_failure_gate_bundle(
    bundle: Path, bindings: FailureGateInputBindings
) -> FailureGateIngestResult:
    """Fail closed unless the complete bundle independently proves one restart."""

    _validate_bindings(bindings)
    sources: dict[str, _StableSource] = {}
    loaded_names: list[str] = []
    snapshot: _BundleSnapshot | None = None
    try:
        sources = _pin_sources(bindings)
        gate, gate_name = _load_validator(sources["gate"], "gate")
        loaded_names.append(gate_name)
        hook, hook_name = _load_validator(sources["hook"], "hook")
        loaded_names.append(hook_name)
        if (
            getattr(hook, "PHASE", None) != CAMPAIGN_PHASE
            or getattr(hook, "EXPECTED_ACTIONS", None) is None
            or getattr(gate, "LIFECYCLE_SCHEMA", None) != "ullm.gateway.lifecycle.v1"
        ):
            fail("failure gate or hook campaign contract differs")
        claim_factory = _claim_factory()
        snapshot = _BundleSnapshot(
            bundle,
            uid=bindings.uid,
            gid=bindings.gid,
            forbidden_values=bindings.forbidden_values,
        )

        summary_raw = snapshot.read_small("summary")
        summary = _canonical_document(hook, summary_raw, "failure gate summary")
        if type(summary.get("passed")) is not bool:
            fail("failure producer pass flag is not boolean")
        summary_for_hook = copy.deepcopy(summary)
        summary_for_hook["passed"] = True
        try:
            hook.validate_summary(summary_for_hook, _hook_bindings(hook, bindings))
        except Exception as error:
            raise FailureGateIngestError(
                "failure gate summary validation failed"
            ) from error

        fault_raw = snapshot.read_small("fault")
        fault = _canonical_document(hook, fault_raw, "failure fault artifact")
        readiness_raw = snapshot.read_small("readiness")
        readiness = _canonical_document(
            hook, readiness_raw, "failure readiness artifact"
        )
        try:
            hook.validate_fault(fault, summary_for_hook)
            hook.validate_readiness(readiness, summary_for_hook)
        except Exception as error:
            raise FailureGateIngestError(
                "failure fault or readiness validation failed"
            ) from error
        if (
            readiness["network_id"] != bindings.docker_network_id
            or readiness["subnet"] != bindings.docker_network_subnet
            or readiness["gateway"] != bindings.docker_network_gateway
        ):
            fail("failure readiness Docker network binding differs")

        snapshot.stream("screenshot")
        screenshot_bytes, screenshot_sha256, screenshot_prefix = snapshot.evidence(
            "screenshot"
        )
        if not screenshot_prefix.startswith(b"\x89PNG\r\n\x1a\n"):
            fail("failure screenshot is not a PNG")
        screenshot = FailureScreenshotEvidence(
            source_path=Path(os.path.abspath(bundle))
            / "browser"
            / "post-header-failure.png",
            bundle_path=SCREENSHOT_BUNDLE_PATH,
            bytes=screenshot_bytes,
            sha256=screenshot_sha256,
        )

        browser, actions, worker_nonce, recovery_nonce, stdout_raw = _read_browser(
            snapshot, hook
        )
        if (
            browser["screenshot"]["screenshot_file"] != screenshot.bundle_path
            or browser["screenshot"]["screenshot_bytes"] != screenshot.bytes
            or browser["screenshot"]["screenshot_sha256"] != screenshot.sha256
        ):
            fail("failure screenshot evidence differs from the browser summary")
        browser_final = _browser_final(gate, hook, browser)

        lifecycle, claims, journal_records, cursor_count = _read_journal(
            snapshot,
            gate,
            bindings,
            browser_final,
            fault,
            claim_factory,
        )
        journal_bytes, journal_sha256, _journal_prefix = snapshot.evidence("journal")
        if (
            readiness["initial"]["completed_monotonic_ns"]
            > fault["started_monotonic_ns"]
            or fault["completed_monotonic_ns"]
            > readiness["recovered"]["started_monotonic_ns"]
            or readiness["recovered"]["completed_monotonic_ns"]
            > browser_final.recovery_control_observed_ns
            or browser_final.recovery_control_observed_ns
            > lifecycle.recovery_admitted_ns
        ):
            fail("failure readiness, fault, control, or recovery timeline differs")

        expected = _expected_summary(
            bindings,
            browser=browser,
            browser_stdout=stdout_raw,
            screenshot_bytes=screenshot.bytes,
            screenshot_sha256=screenshot.sha256,
            fault_raw=fault_raw,
            readiness_raw=readiness_raw,
            journal_bytes=journal_bytes,
            journal_sha256=journal_sha256,
            journal_records=journal_records,
            lifecycle=lifecycle,
            worker_nonce=worker_nonce,
            recovery_nonce=recovery_nonce,
        )
        observed_without_passed = {
            key: value for key, value in summary.items() if key != "passed"
        }
        _typed_json_equal(
            hook,
            observed_without_passed,
            expected,
            "failure gate summary",
        )
        if cursor_count != journal_records:
            fail("failure journal cursor count differs")

        action_records = tuple(
            {
                "record_type": "browser_action",
                "phase": CAMPAIGN_PHASE,
                "case_id": FAILURE_CASE,
                "fields": hook.action_hook_fields(action),
            }
            for action in actions
        )
        fault_record = {
            "record_type": "fault_injection",
            "phase": CAMPAIGN_PHASE,
            "case_id": FAILURE_CASE,
            "fields": hook.fault_hook_fields(fault),
        }
        restart_probe = {
            "record_type": "lifecycle_probe",
            "phase": CAMPAIGN_PHASE,
            "case_id": RESTART_PROBE,
            "fields": {
                "probe": RESTART_PROBE,
                "observed_monotonic_ns": readiness["recovered"][
                    "completed_monotonic_ns"
                ],
                "service_active": True,
                "ready_http_status": readiness["recovered"]["status"],
                "control_group": bindings.control_group,
                "gateway_pid": bindings.restart_gateway_pid,
                "gateway_starttime_ticks": bindings.restart_gateway_starttime_ticks,
                "worker_pid": bindings.restart_worker_pid,
                "worker_starttime_ticks": bindings.restart_worker_starttime_ticks,
                "n_restarts": bindings.restart_restart_count,
            },
        }
        view = _derived_view(
            bindings,
            summary_raw=summary_raw,
            browser=browser,
            fault=fault,
            lifecycle=lifecycle,
            journal_records=journal_records,
            journal_sha256=journal_sha256,
            screenshot=screenshot,
            readiness=readiness,
        )
        encoded_view = hook.compact_json(view)
        scanner = _SecretScanner(bindings.forbidden_values)
        scanner.consume(encoded_view)

        snapshot.seal()
        for source in sources.values():
            source.seal()
        return FailureGateIngestResult(
            action_records,
            fault_record,
            claims,
            restart_probe,
            screenshot,
            view,
        )
    except FailureGateIngestError:
        raise
    except Exception as error:
        raise FailureGateIngestError("failure gate bundle ingestion failed") from error
    finally:
        for name in loaded_names:
            sys.modules.pop(name, None)
        pending: FailureGateIngestError | None = None
        if snapshot is not None:
            try:
                snapshot.close()
            except FailureGateIngestError as error:
                pending = error
        for source in sources.values():
            try:
                source.close()
            except FailureGateIngestError as error:
                pending = error
        if pending is not None and sys.exc_info()[0] is None:
            raise pending
