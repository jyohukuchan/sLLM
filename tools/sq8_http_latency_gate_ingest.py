#!/usr/bin/env python3
"""Revalidate and convert one frozen SQ8 HTTP latency gate bundle."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import types
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterator,
    NamedTuple,
    NoReturn,
    Protocol,
    Sequence,
    cast,
)


VIEW_SCHEMA = "ullm.sq8.http_latency_gate_ingest.view.v1"
GATE_SCHEMA = "ullm.sq8.http_latency_gate.v1"
CAMPAIGN_PHASE = "latency"
CAMPAIGN_SERVICE_UNIT = "ullm-openai.service"
EXPECTED_FILES = frozenset(
    {
        "http-client.raw.jsonl",
        "observer.raw.jsonl",
        "service-journal.raw.jsonl",
        "observer-journal-correlation.raw.jsonl",
        "samples.raw.jsonl",
        "input-manifest.json",
        "summary.json",
        "SHA256SUMS",
    }
)
CHECKSUM_INPUTS = tuple(
    sorted(EXPECTED_FILES - {"SHA256SUMS"}, key=lambda value: value.encode("ascii"))
)
RAW_FILES = (
    "http-client.raw.jsonl",
    "observer.raw.jsonl",
    "service-journal.raw.jsonl",
    "observer-journal-correlation.raw.jsonl",
    "samples.raw.jsonl",
)
FILE_LIMITS = {
    **{name: 64 << 20 for name in RAW_FILES},
    "input-manifest.json": 8 << 20,
    "summary.json": 8 << 20,
    "SHA256SUMS": 1 << 20,
}
SOURCE_LIMITS = {
    "gate": 8 << 20,
    "direct": 8 << 20,
    "collector": 8 << 20,
    "http_client": 8 << 20,
    "restart_epoch": 8 << 20,
    "fixture": 8 << 20,
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
MAX_JSON_LINE_BYTES = 8 << 20
MAX_RAW_LINES = 4096
COPY_CHUNK_BYTES = 64 << 10
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
BOOT_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
CONTENT_IMAGE_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
NETWORK_ID_RE = re.compile(r"[0-9a-f]{64}\Z")


class LatencyGateIngestError(RuntimeError):
    """A fail-closed adapter error without evidence-derived diagnostics."""


def fail(message: str) -> NoReturn:
    raise LatencyGateIngestError(message)


class BundleLifecycleClaimProtocol(Protocol):
    raw: bytes
    phase: str
    case_id: str


@dataclasses.dataclass(frozen=True)
class LatencyGateInputBindings:
    gate_source: Path
    gate_source_sha256: str
    direct_source: Path
    direct_source_sha256: str
    collector_source: Path
    collector_source_sha256: str
    http_client_source: Path
    http_client_source_sha256: str
    restart_epoch_file: Path
    restart_epoch_sha256: str
    http_image_id: str
    docker_network_id: str
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
    forbidden_values: tuple[bytes, ...] = ()


class LatencyGateIngestResult(NamedTuple):
    http_records: tuple[dict[str, Any], ...]
    lifecycle_claims: tuple[BundleLifecycleClaimProtocol, ...]
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
    name: str
    fd: int
    identity: _Identity
    maximum: int
    consumed: bool = False
    streamed_bytes: int = 0
    sha256: str | None = None


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for latency bundle ingestion")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _file_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for latency bundle ingestion")
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


def _safe_close(descriptor: int) -> None:
    if descriptor < 0:
        return
    try:
        os.close(descriptor)
    except OSError:
        fail("failed to close a sealed latency evidence descriptor")


def _entry_identity(parent_fd: int, name: str) -> _Identity:
    try:
        return _Identity.from_stat(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        )
    except OSError:
        fail("sealed latency evidence entry is unavailable")


class _SecretScanner:
    def __init__(self, values: tuple[bytes, ...]):
        self.values = values
        self.overlap = max((len(value) for value in values), default=1) - 1
        self.tail = b""

    def consume(self, chunk: bytes) -> None:
        combined = self.tail + chunk
        if any(value in combined for value in self.values):
            fail("latency evidence contains forbidden cleartext")
        self.tail = combined[-self.overlap :] if self.overlap else b""


class _BundleSnapshot:
    """Open and retain the exact flat gate layout through final sealing."""

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
        self.root_identity: _Identity | None = None
        self.files: dict[str, _OpenedFile] = {}
        self.sealed = False
        self.closed = False
        self._open()

    def _open(self) -> None:
        try:
            self.parent_fd = os.open(self.root_path.parent, _directory_flags())
            self.root_fd = os.open(
                self.root_path.name, _directory_flags(), dir_fd=self.parent_fd
            )
            root_identity = _Identity.from_stat(os.fstat(self.root_fd))
            if _entry_identity(self.parent_fd, self.root_path.name) != root_identity:
                fail("latency bundle root identity changed while opening")
            if (
                not stat.S_ISDIR(root_identity.mode)
                or stat.S_IMODE(root_identity.mode) != 0o700
                or root_identity.links != 2
                or root_identity.uid != self.uid
                or root_identity.gid != self.gid
            ):
                fail("latency bundle root owner, mode, or link count differs")
            if frozenset(os.listdir(self.root_fd)) != EXPECTED_FILES:
                fail("latency bundle layout differs")
            self.root_identity = root_identity
            for name in sorted(EXPECTED_FILES, key=lambda value: value.encode("ascii")):
                entry = _entry_identity(self.root_fd, name)
                if (
                    not stat.S_ISREG(entry.mode)
                    or stat.S_IMODE(entry.mode) != 0o600
                    or entry.links != 1
                    or entry.uid != self.uid
                    or entry.gid != self.gid
                    or entry.size < 1
                    or entry.size > FILE_LIMITS[name]
                ):
                    fail("latency artifact owner, mode, link count, or size differs")
                descriptor = os.open(name, _file_flags(), dir_fd=self.root_fd)
                opened = _Identity.from_stat(os.fstat(descriptor))
                if opened != entry:
                    os.close(descriptor)
                    fail("latency artifact identity changed while opening")
                self.files[name] = _OpenedFile(
                    name=name,
                    fd=descriptor,
                    identity=opened,
                    maximum=FILE_LIMITS[name],
                )
        except LatencyGateIngestError:
            self.close()
            raise
        except OSError:
            self.close()
            fail("failed to open latency bundle without following links")

    def _chunks(self, name: str) -> Iterator[bytes]:
        if self.closed or self.sealed:
            fail("latency bundle snapshot is no longer readable")
        item = self.files.get(name)
        if item is None or item.consumed:
            fail("latency artifact was requested outside its fixed read schedule")
        try:
            if _Identity.from_stat(os.fstat(item.fd)) != item.identity:
                fail("latency artifact changed before streaming")
            os.lseek(item.fd, 0, os.SEEK_SET)
            total = 0
            digest = hashlib.sha256()
            scanner = _SecretScanner(self.forbidden_values)
            while True:
                chunk = os.read(item.fd, COPY_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > item.maximum:
                    fail("latency artifact exceeded its streaming bound")
                digest.update(chunk)
                scanner.consume(chunk)
                yield chunk
            if (
                total != item.identity.size
                or _Identity.from_stat(os.fstat(item.fd)) != item.identity
            ):
                fail("latency artifact changed while streaming")
            item.streamed_bytes = total
            item.sha256 = digest.hexdigest()
            item.consumed = True
        except LatencyGateIngestError:
            raise
        except OSError:
            fail("failed to stream latency artifact")

    def iter_lines(self, name: str) -> Iterator[bytes]:
        pending = bytearray()
        count = 0
        for chunk in self._chunks(name):
            pending.extend(chunk)
            while True:
                index = pending.find(b"\n")
                if index < 0:
                    if len(pending) > MAX_JSON_LINE_BYTES:
                        fail("latency JSONL line exceeds its bound")
                    break
                raw = bytes(pending[:index])
                del pending[: index + 1]
                count += 1
                if (
                    not raw
                    or raw.endswith(b"\r")
                    or len(raw) > MAX_JSON_LINE_BYTES
                    or count > MAX_RAW_LINES
                ):
                    fail("latency JSONL framing or line count differs")
                yield raw
        if pending:
            fail("latency JSONL artifact lacks its final LF")

    def read_small(self, name: str) -> bytes:
        return b"".join(self._chunks(name))

    def evidence(self, name: str) -> tuple[int, str]:
        item = self.files.get(name)
        if item is None or not item.consumed or item.sha256 is None:
            fail("latency artifact was not fully consumed")
        return item.streamed_bytes, item.sha256

    def seal(self) -> None:
        if self.closed or self.sealed or self.root_identity is None:
            fail("latency bundle cannot be sealed in its current state")
        if any(not item.consumed for item in self.files.values()):
            fail("not every latency artifact was consumed")
        try:
            if (
                frozenset(os.listdir(self.root_fd)) != EXPECTED_FILES
                or _Identity.from_stat(os.fstat(self.root_fd)) != self.root_identity
                or _entry_identity(self.parent_fd, self.root_path.name)
                != self.root_identity
            ):
                fail("latency bundle root layout or identity changed")
            for item in self.files.values():
                if (
                    _Identity.from_stat(os.fstat(item.fd)) != item.identity
                    or _entry_identity(self.root_fd, item.name) != item.identity
                ):
                    fail("latency artifact identity changed before seal")
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
                    or _entry_identity(self.root_fd, item.name) != item.identity
                ):
                    fail("latency artifact bytes or identity changed at seal")
            if (
                frozenset(os.listdir(self.root_fd)) != EXPECTED_FILES
                or _Identity.from_stat(os.fstat(self.root_fd)) != self.root_identity
                or _entry_identity(self.parent_fd, self.root_path.name)
                != self.root_identity
            ):
                fail("latency bundle root changed during final sealing")
            self.sealed = True
        except LatencyGateIngestError:
            raise
        except OSError:
            fail("failed to seal latency bundle")

    def close(self) -> None:
        if self.closed:
            return
        pending: LatencyGateIngestError | None = None
        for item in self.files.values():
            try:
                _safe_close(item.fd)
            except LatencyGateIngestError as error:
                pending = error
        for descriptor in (self.root_fd, self.parent_fd):
            try:
                _safe_close(descriptor)
            except LatencyGateIngestError as error:
                pending = error
        self.root_fd = self.parent_fd = -1
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
        self.expected_sha256 = expected_sha256
        self.forbidden_values = forbidden_values
        self.parent_fd = -1
        self.fd = -1
        self.identity: _Identity | None = None
        self.raw = b""
        self.sha256 = ""
        if SHA256_RE.fullmatch(expected_sha256) is None:
            fail("bound latency source SHA-256 syntax differs")
        try:
            self.parent_fd = os.open(self.path.parent, _directory_flags())
            entry = _entry_identity(self.parent_fd, self.path.name)
            if (
                not stat.S_ISREG(entry.mode)
                or entry.links != 1
                or entry.size < 1
                or entry.size > maximum
            ):
                fail("bound latency source is not one bounded regular file")
            self.fd = os.open(self.path.name, _file_flags(), dir_fd=self.parent_fd)
            opened = _Identity.from_stat(os.fstat(self.fd))
            if opened != entry:
                fail("bound latency source identity changed while opening")
            self.identity = opened
            self.raw, self.sha256 = self._snapshot()
            if self.sha256 != expected_sha256:
                fail("bound latency source SHA-256 differs")
        except LatencyGateIngestError:
            self.close()
            raise
        except OSError:
            self.close()
            fail("failed to open a bound latency source")

    def _snapshot(self) -> tuple[bytes, str]:
        assert self.identity is not None
        try:
            if _Identity.from_stat(os.fstat(self.fd)) != self.identity:
                fail("bound latency source identity changed before reading")
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
                    fail("bound latency source exceeded its streaming bound")
                digest.update(chunk)
                scanner.consume(chunk)
                chunks.append(chunk)
            if (
                total != self.identity.size
                or _Identity.from_stat(os.fstat(self.fd)) != self.identity
            ):
                fail("bound latency source changed while reading")
            return b"".join(chunks), digest.hexdigest()
        except LatencyGateIngestError:
            raise
        except OSError:
            fail("failed to stream a bound latency source")

    def seal(self) -> None:
        assert self.identity is not None
        if (
            _Identity.from_stat(os.fstat(self.fd)) != self.identity
            or _entry_identity(self.parent_fd, self.path.name) != self.identity
        ):
            fail("bound latency source entry changed before sealing")
        raw, digest = self._snapshot()
        if raw != self.raw or digest != self.sha256:
            fail("bound latency source bytes changed before sealing")
        if (
            _Identity.from_stat(os.fstat(self.fd)) != self.identity
            or _entry_identity(self.parent_fd, self.path.name) != self.identity
        ):
            fail("bound latency source identity changed during sealing")

    def close(self) -> None:
        pending: LatencyGateIngestError | None = None
        for descriptor in (self.fd, self.parent_fd):
            try:
                _safe_close(descriptor)
            except LatencyGateIngestError as error:
                pending = error
        self.fd = self.parent_fd = -1
        if pending is not None:
            raise pending


def _source_specs(
    bindings: LatencyGateInputBindings,
) -> dict[str, tuple[Path, str]]:
    return {
        "gate": (bindings.gate_source, bindings.gate_source_sha256),
        "direct": (bindings.direct_source, bindings.direct_source_sha256),
        "collector": (bindings.collector_source, bindings.collector_source_sha256),
        "http_client": (
            bindings.http_client_source,
            bindings.http_client_source_sha256,
        ),
        "restart_epoch": (
            bindings.restart_epoch_file,
            bindings.restart_epoch_sha256,
        ),
    }


def _validate_bindings(bindings: LatencyGateInputBindings) -> None:
    if not isinstance(bindings, LatencyGateInputBindings):
        fail("latency input bindings have the wrong type")
    specs = _source_specs(bindings)
    if any(
        not isinstance(path, os.PathLike)
        or type(digest) is not str
        or SHA256_RE.fullmatch(digest) is None
        for path, digest in specs.values()
    ):
        fail("latency source path or SHA-256 binding differs")
    paths = {key: Path(os.path.abspath(value[0])) for key, value in specs.items()}
    if (
        len(set(paths.values())) != len(paths)
        or paths["gate"].name != "run-sq8-http-latency-gate.py"
        or paths["direct"] != paths["gate"].with_name("run-sq8-direct-cancel-gate.py")
        or paths["collector"]
        != paths["gate"].with_name("collect-sq8-openwebui-release.py")
        or paths["http_client"]
        != paths["gate"].with_name("sq8-openwebui-http-client.py")
    ):
        fail("latency gate and support source layout differs")
    if CONTENT_IMAGE_RE.fullmatch(bindings.http_image_id) is None:
        fail("latency HTTP image content identity differs")
    if NETWORK_ID_RE.fullmatch(bindings.docker_network_id) is None:
        fail("latency Docker network identity differs")
    if bindings.service_unit != CAMPAIGN_SERVICE_UNIT:
        fail("latency service unit differs from the campaign contract")
    if (
        type(bindings.service_user) is not str
        or not bindings.service_user
        or len(bindings.service_user) > 128
        or "/" in bindings.service_user
        or "\0" in bindings.service_user
        or type(bindings.boot_id) is not str
        or BOOT_ID_RE.fullmatch(bindings.boot_id) is None
        or bindings.control_group != f"/system.slice/{bindings.service_unit}"
    ):
        fail("latency service text identity differs")
    for numeric_value, minimum in (
        (bindings.gateway_pid, 1),
        (bindings.gateway_starttime_ticks, 1),
        (bindings.worker_pid, 1),
        (bindings.worker_starttime_ticks, 1),
        (bindings.restart_count, 1),
        (bindings.uid, 0),
        (bindings.gid, 0),
    ):
        if type(numeric_value) is not int or numeric_value < minimum:
            fail("latency numeric service identity differs")
    if bindings.gateway_pid == bindings.worker_pid:
        fail("latency gateway and worker PID identities collide")
    if type(bindings.forbidden_values) is not tuple or not bindings.forbidden_values:
        fail("latency secret bindings are absent or mutable")
    if len(set(bindings.forbidden_values)) != len(bindings.forbidden_values):
        fail("latency secret bindings are duplicated")
    for secret in bindings.forbidden_values:
        if (
            type(secret) is not bytes
            or not 16 <= len(secret) <= 4096
            or b"\0" in secret
            or b"\r" in secret
            or b"\n" in secret
        ):
            fail("latency secret binding syntax differs")


def _restore_modules(prior: dict[str, types.ModuleType | None]) -> None:
    for name, value in prior.items():
        if value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value


def _load_gate(
    sources: dict[str, _StableSource],
) -> tuple[Any, str]:
    gate_source = sources["gate"]
    module_name = f"_ullm_latency_ingest_{os.getpid()}_{id(gate_source):x}"
    support_names = (
        "_ullm_sq8_http_latency_direct_support",
        "_ullm_sq8_cancel_collector_support",
    )
    prior = {name: sys.modules.get(name) for name in support_names}
    module = types.ModuleType(module_name)
    module.__file__ = os.fspath(gate_source.path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        code = compile(
            gate_source.raw,
            os.fspath(gate_source.path),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)
        if (
            module.GATE_SCHEMA != GATE_SCHEMA
            or module.MODULE_IMPORT_RAW != gate_source.raw
            or module.DIRECT_SUPPORT_RAW != sources["direct"].raw
            or module.DIRECT.COLLECTOR_SUPPORT_RAW != sources["collector"].raw
            or module.HTTP_CLIENT_SHA256 != sources["http_client"].sha256
            or module.DIRECT.SERVICE_UNIT != CAMPAIGN_SERVICE_UNIT
            or module.HTTP_NETWORK_NAME != "open-webui-network"
            or module.HTTP_TARGET != "/v1/chat/completions"
            or len(module.SCHEDULE) != 72
        ):
            fail("executed latency gate source binding differs")
        return module, module_name
    except LatencyGateIngestError:
        sys.modules.pop(module_name, None)
        raise
    except Exception as error:
        sys.modules.pop(module_name, None)
        raise LatencyGateIngestError("failed to load bound latency gate") from error
    finally:
        _restore_modules(prior)


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
            raise LatencyGateIngestError(
                "campaign journal contract module failed to load"
            ) from error
    factory = getattr(module, "BundleLifecycleClaim", None)
    if (
        not callable(factory)
        or getattr(module, "SERVICE_UNIT", None) != CAMPAIGN_SERVICE_UNIT
    ):
        fail("campaign journal lifecycle claim contract is unavailable")
    return cast(Callable[[bytes, str, str], BundleLifecycleClaimProtocol], factory)


def _strict_object(gate: Any, raw: bytes, label: str) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], gate.COL.strict_json_object(raw, label))
    except Exception as error:
        raise LatencyGateIngestError(f"{label} is invalid") from error


def _exact(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        fail(f"{label} fields differ")
    return cast(dict[str, Any], value)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail(f"{label} is invalid")
    return value


def _decimal(value: Any, label: str, *, minimum: int = 0) -> int:
    if (
        type(value) is not str
        or not value.isascii()
        or not value.isdecimal()
        or value != str(int(value, 10))
    ):
        fail(f"{label} representation differs")
    parsed = int(value, 10)
    if parsed < minimum:
        fail(f"{label} is below its minimum")
    return parsed


def _text(value: Any, label: str, *, maximum: int = 65_536) -> str:
    if type(value) is not str or not value or len(value) > maximum or "\0" in value:
        fail(f"{label} is invalid")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        fail(f"{label} is not strict UTF-8")
    return value


def _same_json(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        if set(actual) != set(expected):
            return False
        return all(_same_json(actual[key], expected[key]) for key in expected)
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _same_json(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _canonical_document(gate: Any, raw: bytes, label: str) -> dict[str, Any]:
    if not raw.endswith(b"\n") or raw.endswith(b"\r\n") or raw.count(b"\n") != 1:
        fail(f"{label} framing differs")
    value = _strict_object(gate, raw[:-1], label)
    try:
        if gate.compact_json(value) + b"\n" != raw:
            fail(f"{label} is not canonical producer JSON")
    except LatencyGateIngestError:
        raise
    except Exception as error:
        raise LatencyGateIngestError(f"{label} cannot be encoded") from error
    return value


def _canonical_line(gate: Any, raw: bytes, label: str) -> dict[str, Any]:
    value = _strict_object(gate, raw, label)
    try:
        if gate.compact_json(value) != raw:
            fail(f"{label} is not canonical producer JSON")
    except LatencyGateIngestError:
        raise
    except Exception as error:
        raise LatencyGateIngestError(f"{label} cannot be encoded") from error
    return value


def _service_identity(bindings: LatencyGateInputBindings) -> dict[str, Any]:
    return {
        "unit": bindings.service_unit,
        "user": bindings.service_user,
        "uid": bindings.uid,
        "gid": bindings.gid,
        "control_group": bindings.control_group,
        "gateway_pid": bindings.gateway_pid,
        "gateway_starttime_ticks": bindings.gateway_starttime_ticks,
        "worker_pid": bindings.worker_pid,
        "worker_starttime_ticks": bindings.worker_starttime_ticks,
        "n_restarts": bindings.restart_count,
        "boot_id": bindings.boot_id,
    }


def _open_fixtures(
    gate: Any,
    gate_source: _StableSource,
    forbidden_values: tuple[bytes, ...],
) -> tuple[dict[str, _StableSource], dict[str, Any]]:
    root = (
        gate_source.path.parent.parent
        / "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures"
    )
    sources: dict[str, _StableSource] = {}
    fixtures: dict[str, Any] = {}
    try:
        if tuple(gate.FIXTURE_ORDER) != tuple(gate.FIXTURE_IDENTITIES):
            fail("latency fixture order differs from its fixed identities")
        for fixture_id in gate.FIXTURE_ORDER:
            expected_prompt, expected_sha = gate.FIXTURE_IDENTITIES[fixture_id]
            source = _StableSource(
                root / f"{fixture_id}.json",
                f"bound {fixture_id} fixture",
                SOURCE_LIMITS["fixture"],
                expected_sha,
                forbidden_values,
            )
            sources[fixture_id] = source
            try:
                fixture = gate.load_fixture(source.path, fixture_id)
            except Exception as error:
                raise LatencyGateIngestError(
                    "bound latency fixture validation failed"
                ) from error
            if (
                fixture.raw != source.raw
                or fixture.sha256 != source.sha256
                or fixture.prompt_tokens != expected_prompt
            ):
                fail("bound latency fixture snapshot differs")
            fixtures[fixture_id] = fixture
        return sources, fixtures
    except BaseException:
        for source in reversed(tuple(sources.values())):
            source.close()
        raise


class _Guard:
    def __init__(self, values: tuple[bytes, ...]) -> None:
        self.values = values

    def reject(self, raw: bytes, _label: str) -> None:
        scanner = _SecretScanner(self.values)
        scanner.consume(raw)


@dataclasses.dataclass(frozen=True)
class _HttpCase:
    spec: Any
    observation: Any


def _http_records(
    gate: Any,
    snapshot: _BundleSnapshot,
    fixtures: dict[str, Any],
    forbidden_values: tuple[bytes, ...],
) -> tuple[list[dict[str, Any]], list[_HttpCase], int]:
    iterator = iter(snapshot.iter_lines("http-client.raw.jsonl"))
    line_count = 0

    def next_event(label: str) -> dict[str, Any]:
        nonlocal line_count
        try:
            raw = next(iterator)
        except StopIteration:
            fail(f"{label} is missing")
        line_count += 1
        return _strict_object(gate, raw, label)

    ready = next_event("latency HTTP ready event")
    _exact(
        ready,
        {"schema_version", "event", "observed_monotonic_ns"},
        "latency HTTP ready event",
    )
    if ready["schema_version"] != gate.HTTP_EVENT_SCHEMA or ready["event"] != "ready":
        fail("latency HTTP ready event differs")
    ready_ns = _integer(ready["observed_monotonic_ns"], "latency HTTP ready timestamp")
    try:
        gate.validate_schedule(gate.SCHEDULE)
    except Exception as error:
        raise LatencyGateIngestError(
            "latency request schedule validation failed"
        ) from error
    offline = gate.LatencyHttpClient((), _Guard(forbidden_values), None)
    records: list[dict[str, Any]] = []
    cases: list[_HttpCase] = []
    for spec in gate.SCHEDULE:
        body = gate.request_body(fixtures[spec.fixture_id], spec.max_tokens)
        plan = gate.HttpPlan(spec, body)
        events: list[dict[str, Any]] = []
        while True:
            event = next_event(f"latency HTTP case {spec.sequence} event")
            if event.get("event") in {"ready", "shutdown_complete"}:
                fail("latency HTTP control event appears within a request")
            events.append(event)
            if len(events) > 128:
                fail("one latency HTTP request exceeds its event bound")
            if event.get("event") == "http_response_end":
                break
        event_index = 0

        def read_event(_deadline_ns: int) -> dict[str, Any]:
            nonlocal event_index
            if event_index >= len(events):
                fail("latency HTTP request ended before response_end")
            value = events[event_index]
            event_index += 1
            return value

        offline.active = plan
        offline._read_event = read_event
        try:
            observation = offline.finish(0)
        except LatencyGateIngestError:
            raise
        except Exception as error:
            raise LatencyGateIngestError(
                "raw latency HTTP validation failed"
            ) from error
        if event_index != len(events):
            fail("latency HTTP request retains events after response_end")
        request = events[0]
        if request.get("event") != "http_request" or (
            spec.sequence == 1 and request["connect_completed_monotonic_ns"] < ready_ns
        ):
            fail("latency HTTP request does not follow ready in exact order")
        for event in events:
            record_type = event.get("event")
            if record_type not in {
                "http_request",
                "http_response_start",
                "http_body_chunk",
                "http_response_end",
            }:
                fail("latency HTTP request contains an unsupported session event")
            fields = {
                key: copy.deepcopy(value)
                for key, value in event.items()
                if key not in {"schema_version", "event"}
            }
            if record_type == "http_request":
                fields = {"request_index": spec.sequence, **fields}
            records.append(
                {
                    "record_type": record_type,
                    "phase": CAMPAIGN_PHASE,
                    "case_id": spec.case_id,
                    "fields": fields,
                }
            )
        cases.append(_HttpCase(spec, observation))
    shutdown = next_event("latency HTTP shutdown event")
    _exact(
        shutdown,
        {"schema_version", "event", "observed_monotonic_ns"},
        "latency HTTP shutdown event",
    )
    shutdown_ns = _integer(
        shutdown["observed_monotonic_ns"], "latency HTTP shutdown timestamp"
    )
    if (
        shutdown["schema_version"] != gate.HTTP_EVENT_SCHEMA
        or shutdown["event"] != "shutdown_complete"
        or shutdown_ns < offline.last_response_end_ns
    ):
        fail("latency HTTP shutdown event differs or regresses")
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        fail("latency HTTP evidence continues after shutdown_complete")
    if len(cases) != 72:
        fail("latency HTTP request count differs from 72")
    return records, cases, line_count


@dataclasses.dataclass(frozen=True)
class _LifecycleRecord:
    journal: dict[str, Any]
    event: dict[str, Any]
    payload: bytes


def _journal_and_claims(
    gate: Any,
    snapshot: _BundleSnapshot,
    bindings: LatencyGateInputBindings,
    cases: Sequence[_HttpCase],
    claim_factory: Callable[[bytes, str, str], BundleLifecycleClaimProtocol],
) -> tuple[
    list[BundleLifecycleClaimProtocol],
    list[_LifecycleRecord],
    list[dict[str, Any]],
    dict[str, Any],
    int,
]:
    validator = gate.LatencyRunValidator()
    claims: list[BundleLifecycleClaimProtocol] = []
    lifecycle_records: list[_LifecycleRecord] = []
    computed_samples: list[dict[str, Any]] = []
    cursors: set[str] = set()
    last_monotonic = -1
    current_case_id: str | None = None
    journal_count = 0
    for raw in snapshot.iter_lines("service-journal.raw.jsonl"):
        journal_count += 1
        record = _strict_object(gate, raw, "latency service journal record")
        if set(record) != set(REQUIRED_JOURNAL_FIELDS):
            fail("latency service journal required fields differ")
        cursor = _text(record["__CURSOR"], "latency service journal cursor")
        if cursor in cursors:
            fail("latency service journal cursor is duplicated")
        cursors.add(cursor)
        monotonic = _decimal(
            record["__MONOTONIC_TIMESTAMP"],
            "latency service journal monotonic timestamp",
        )
        pid = _decimal(record["_PID"], "latency service journal PID", minimum=1)
        priority = _decimal(record["PRIORITY"], "latency service journal priority")
        if (
            monotonic < last_monotonic
            or priority > 7
            or record["_BOOT_ID"] != bindings.boot_id
            or record["_SYSTEMD_UNIT"] != bindings.service_unit
            or type(record["MESSAGE"]) is not str
        ):
            fail("latency service journal ordering or bound identity differs")
        last_monotonic = monotonic
        try:
            event = gate.COL.decode_lifecycle_message(record["MESSAGE"])
        except Exception as error:
            raise LatencyGateIngestError(
                "latency service journal MESSAGE is invalid"
            ) from error
        if event is None:
            continue
        if pid != bindings.gateway_pid:
            fail("latency lifecycle journal PID differs from the restart epoch")
        try:
            payload = gate.COL.lifecycle_payload_from_message(record["MESSAGE"])
        except Exception as error:
            raise LatencyGateIngestError(
                "latency journal lifecycle framing is invalid"
            ) from error
        event = cast(dict[str, Any], event)
        observed = _integer(
            event["observed_monotonic_ns"], "latency lifecycle observed timestamp"
        )
        if monotonic * 1000 < observed:
            fail("latency journal timestamp precedes lifecycle observation")
        if current_case_id is None:
            if event["event"] != "request_admitted" or validator.index >= len(cases):
                fail("latency lifecycle trace does not begin with admission")
            case = cases[validator.index]
            validator.begin(case.spec)
            current_case_id = cast(str, case.spec.case_id)
        try:
            validator.consume(event)
        except Exception as error:
            raise LatencyGateIngestError(
                "latency lifecycle state validation failed"
            ) from error
        required = {field: record[field] for field in REQUIRED_JOURNAL_FIELDS}
        try:
            claim_raw = json.dumps(
                required,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8", errors="strict")
        except (TypeError, ValueError, UnicodeError, RecursionError) as error:
            raise LatencyGateIngestError(
                "latency lifecycle claim cannot be reconstructed"
            ) from error
        claims.append(claim_factory(claim_raw, CAMPAIGN_PHASE, current_case_id))
        lifecycle_records.append(_LifecycleRecord(record, event, payload))
        if event["event"] == "request_released":
            try:
                sample = cast(
                    dict[str, Any],
                    validator.complete(cases[len(computed_samples)].observation),
                )
            except Exception as error:
                raise LatencyGateIngestError(
                    "latency HTTP and lifecycle correlation failed"
                ) from error
            computed_samples.append(sample)
            current_case_id = None
    if not cursors or current_case_id is not None:
        fail("latency service journal lacks complete lifecycle evidence")
    try:
        finalized_samples, metrics = validator.finalize()
    except Exception as error:
        raise LatencyGateIngestError("latency schedule finalization failed") from error
    if not _same_json(finalized_samples, computed_samples):
        fail("latency lifecycle samples changed during finalization")
    return claims, lifecycle_records, computed_samples, metrics, journal_count


def _observer_records(
    gate: Any,
    snapshot: _BundleSnapshot,
    lifecycle_records: Sequence[_LifecycleRecord],
) -> int:
    count = 0
    iterator = iter(snapshot.iter_lines("observer.raw.jsonl"))
    for expected in lifecycle_records:
        try:
            raw = next(iterator)
        except StopIteration:
            fail("latency observer lifecycle payload is missing")
        count += 1
        if raw != expected.payload:
            fail("latency observer and journal lifecycle bytes differ")
        try:
            event = gate.COL.decode_lifecycle_payload(raw, "latency observer payload")
        except Exception as error:
            raise LatencyGateIngestError(
                "latency observer lifecycle payload is invalid"
            ) from error
        if not _same_json(event, expected.event):
            fail("latency observer and journal lifecycle values differ")
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        fail("latency observer lifecycle evidence has trailing records")
    if count != len(lifecycle_records):
        fail("latency observer lifecycle count differs")
    return count


def _correlations(
    gate: Any,
    snapshot: _BundleSnapshot,
    bindings: LatencyGateInputBindings,
    lifecycle_records: Sequence[_LifecycleRecord],
) -> int:
    count = 0
    previous_received = -1
    iterator = iter(snapshot.iter_lines("observer-journal-correlation.raw.jsonl"))
    for sequence, evidence in enumerate(lifecycle_records):
        try:
            raw = next(iterator)
        except StopIteration:
            fail("latency observer-journal correlation is missing")
        value = _canonical_line(gate, raw, "latency observer-journal correlation")
        _exact(
            value,
            {
                "schema_version",
                "sequence",
                "cursor",
                "journal_monotonic_usec",
                "journal_pid",
                "observer_received_monotonic_ns",
                "observer_sender_pid",
                "observer_sender_uid",
                "observer_sender_gid",
                "payload_sha256",
                "payload_bytes",
            },
            "latency observer-journal correlation",
        )
        received = _integer(
            value["observer_received_monotonic_ns"],
            "latency observer receipt timestamp",
        )
        event_observed = _integer(
            evidence.event["observed_monotonic_ns"],
            "latency correlated lifecycle timestamp",
        )
        expected = {
            "schema_version": gate.GATE_SCHEMA,
            "sequence": sequence,
            "cursor": evidence.journal["__CURSOR"],
            "journal_monotonic_usec": evidence.journal["__MONOTONIC_TIMESTAMP"],
            "journal_pid": evidence.journal["_PID"],
            "observer_received_monotonic_ns": received,
            "observer_sender_pid": bindings.gateway_pid,
            "observer_sender_uid": bindings.uid,
            "observer_sender_gid": bindings.gid,
            "payload_sha256": hashlib.sha256(evidence.payload).hexdigest(),
            "payload_bytes": len(evidence.payload),
        }
        if (
            received < event_observed
            or received < previous_received
            or not _same_json(value, expected)
        ):
            fail("latency observer-journal correlation differs")
        previous_received = received
        count += 1
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        fail("latency observer-journal correlation has trailing records")
    return count


def _samples(
    gate: Any,
    snapshot: _BundleSnapshot,
    computed_samples: Sequence[dict[str, Any]],
) -> int:
    count = 0
    iterator = iter(snapshot.iter_lines("samples.raw.jsonl"))
    for expected in computed_samples:
        try:
            raw = next(iterator)
        except StopIteration:
            fail("latency sample evidence is missing")
        value = _canonical_line(gate, raw, "latency sample evidence")
        if not _same_json(value, expected):
            fail("latency sample differs from raw HTTP and lifecycle evidence")
        count += 1
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        fail("latency sample evidence has trailing records")
    if count != 72:
        fail("latency sample count differs from 72")
    return count


def _manifest(
    gate: Any,
    snapshot: _BundleSnapshot,
    sources: dict[str, _StableSource],
    fixtures: dict[str, Any],
) -> dict[str, Any]:
    value = _canonical_document(
        gate, snapshot.read_small("input-manifest.json"), "latency input manifest"
    )
    expected = {
        "schema_version": gate.GATE_SCHEMA,
        "record_type": "input_manifest",
        "inputs": [
            {
                "path": "tools/run-sq8-http-latency-gate.py",
                "bytes": len(sources["gate"].raw),
                "sha256": sources["gate"].sha256,
            },
            {
                "path": "tools/run-sq8-direct-cancel-gate.py",
                "bytes": len(sources["direct"].raw),
                "sha256": sources["direct"].sha256,
            },
            {
                "path": "tools/collect-sq8-openwebui-release.py",
                "bytes": len(sources["collector"].raw),
                "sha256": sources["collector"].sha256,
            },
            {
                "path": "tools/sq8-openwebui-http-client.py",
                "bytes": len(sources["http_client"].raw),
                "sha256": sources["http_client"].sha256,
            },
            {
                "path": "resource-restart-epoch.json",
                "bytes": len(sources["restart_epoch"].raw),
                "sha256": sources["restart_epoch"].sha256,
            },
            *[
                {
                    "path": (
                        "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures/"
                        f"{fixture_id}.json"
                    ),
                    "bytes": len(sources[fixture_id].raw),
                    "sha256": sources[fixture_id].sha256,
                }
                for fixture_id in gate.FIXTURE_ORDER
            ],
        ],
        "schedule": [dataclasses.asdict(item) for item in gate.SCHEDULE],
        "request_bodies": [
            {
                "fixture_id": fixture_id,
                "max_tokens": max_tokens,
                "bytes": len(gate.request_body(fixtures[fixture_id], max_tokens)),
                "sha256": hashlib.sha256(
                    gate.request_body(fixtures[fixture_id], max_tokens)
                ).hexdigest(),
            }
            for fixture_id, max_tokens in (
                *((fixture_id, 512) for fixture_id in gate.FIXTURE_ORDER),
                ("exact-p0032", 64),
            )
        ],
    }
    if not _same_json(value, expected):
        fail("latency input manifest differs from bound sources and request bodies")
    return value


def _summary(
    gate: Any,
    snapshot: _BundleSnapshot,
    bindings: LatencyGateInputBindings,
    metrics: dict[str, Any],
    raw_counts: dict[str, int],
) -> dict[str, Any]:
    value = _canonical_document(
        gate, snapshot.read_small("summary.json"), "latency producer summary"
    )
    expected = {
        "schema_version": gate.GATE_SCHEMA,
        "record_type": "summary",
        "passed": True,
        "request_count": 72,
        "max_active": 1,
        "service_identity": _service_identity(bindings),
        "resource_restart_epoch_sha256": bindings.restart_epoch_sha256,
        "http_image_id": bindings.http_image_id,
        "docker_network_name": gate.HTTP_NETWORK_NAME,
        "docker_network_id": bindings.docker_network_id,
        "observer_socket": os.fspath(gate.OBSERVER_SOCKET),
        "observer_event_count": raw_counts["observer.raw.jsonl"],
        "journal_correlation_count": raw_counts[
            "observer-journal-correlation.raw.jsonl"
        ],
        "metrics": metrics,
        "artifacts": {
            name: {
                "bytes": snapshot.evidence(name)[0],
                "lines": raw_counts[name],
                "sha256": snapshot.evidence(name)[1],
            }
            for name in RAW_FILES
        },
    }
    if not _same_json(value, expected):
        fail("latency summary differs from independently reconstructed evidence")
    return value


def _checksums(snapshot: _BundleSnapshot) -> None:
    raw = snapshot.read_small("SHA256SUMS")
    expected = b"".join(
        f"{snapshot.evidence(name)[1]}  {name}\n".encode("ascii")
        for name in CHECKSUM_INPUTS
    )
    if raw != expected:
        fail("latency SHA256SUMS differs from independently hashed artifacts")


def _derived_view(
    gate: Any,
    bindings: LatencyGateInputBindings,
    sources: dict[str, _StableSource],
    samples: Sequence[dict[str, Any]],
    metrics: dict[str, Any],
    http_record_count: int,
    lifecycle_record_count: int,
    journal_record_count: int,
) -> dict[str, Any]:
    prefill_samples = [
        {
            "sequence": sample["sequence"],
            "case_id": sample["case_id"],
            "sample_kind": sample["sample_kind"],
            "sample_index": sample["sample_index"],
            "fixture_id": sample["fixture_id"],
            "prompt_tokens": sample["prompt_tokens"],
            "ttft_ns": sample["ttft_ns"],
            "content_object_count": sample["content_object_count"],
            "release_outcome": sample["release_outcome"],
            "release_completion_tokens": sample["release_completion_tokens"],
        }
        for sample in samples
        if sample["workload"] == "ttft"
    ]
    decode_samples = [
        {
            "sequence": sample["sequence"],
            "case_id": sample["case_id"],
            "sample_kind": sample["sample_kind"],
            "sample_index": sample["sample_index"],
            "fixture_id": sample["fixture_id"],
            "prompt_tokens": sample["prompt_tokens"],
            "decode_elapsed_ns": sample["decode_elapsed_ns"],
            "decode_intervals_ns": copy.deepcopy(sample["decode_intervals_ns"]),
            "decode_tokens_per_second": copy.deepcopy(
                sample["decode_tokens_per_second"]
            ),
            "release_outcome": sample["release_outcome"],
            "release_completion_tokens": sample["release_completion_tokens"],
        }
        for sample in samples
        if sample["workload"] == "decode64"
    ]
    view = {
        "schema_version": VIEW_SCHEMA,
        "request_count": 72,
        "http_record_count": http_record_count,
        "lifecycle_record_count": lifecycle_record_count,
        "journal_record_count": journal_record_count,
        "prefill_ttft": {
            "request_count": len(prefill_samples),
            "metrics": copy.deepcopy(metrics["ttft"]),
            "samples": prefill_samples,
        },
        "decode64": {
            "request_count": len(decode_samples),
            "metrics": copy.deepcopy(metrics["decode64"]),
            "samples": decode_samples,
        },
        "source_bindings": {
            **{f"{key}_sha256": value.sha256 for key, value in sources.items()},
            "http_image_id": bindings.http_image_id,
            "docker_network_id": bindings.docker_network_id,
            "service_unit": bindings.service_unit,
            "service_user": bindings.service_user,
            "boot_id": bindings.boot_id,
            "control_group": bindings.control_group,
            "gateway_pid": bindings.gateway_pid,
            "gateway_starttime_ticks": bindings.gateway_starttime_ticks,
            "worker_pid": bindings.worker_pid,
            "worker_starttime_ticks": bindings.worker_starttime_ticks,
            "restart_count": bindings.restart_count,
            "uid": bindings.uid,
            "gid": bindings.gid,
        },
    }
    try:
        raw = json.dumps(
            view,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise LatencyGateIngestError(
            "derived latency view cannot be encoded"
        ) from error
    _SecretScanner(bindings.forbidden_values).consume(raw)
    for source in sources.values():
        try:
            source_path = os.fspath(source.path).encode("utf-8", errors="strict")
        except UnicodeError:
            fail("bound latency source path is not strict UTF-8")
        if source_path in raw:
            fail("derived latency view contains a host source path")
    return view


def ingest_latency_gate_bundle(
    bundle: Path,
    bindings: LatencyGateInputBindings,
) -> LatencyGateIngestResult:
    _validate_bindings(bindings)
    sources: dict[str, _StableSource] = {}
    snapshot: _BundleSnapshot | None = None
    gate: Any | None = None
    module_name: str | None = None
    try:
        for key, (path, digest) in _source_specs(bindings).items():
            sources[key] = _StableSource(
                path,
                f"bound {key} latency input",
                SOURCE_LIMITS[key],
                digest,
                bindings.forbidden_values,
            )
        gate, module_name = _load_gate(sources)
        if (
            bindings.uid != gate.COL.HTTP_CLIENT_UID
            or bindings.gid != gate.COL.HTTP_CLIENT_GID
        ):
            fail("latency service owner differs from the fixed HTTP client identity")
        fixture_sources, fixtures = _open_fixtures(
            gate, sources["gate"], bindings.forbidden_values
        )
        sources.update(fixture_sources)
        try:
            epoch = gate.load_epoch(sources["restart_epoch"].path)
        except Exception as error:
            raise LatencyGateIngestError("restart epoch validation failed") from error
        if (
            epoch.raw != sources["restart_epoch"].raw
            or epoch.sha256 != sources["restart_epoch"].sha256
            or not _same_json(
                dataclasses.asdict(epoch.service_identity), _service_identity(bindings)
            )
        ):
            fail("restart epoch differs from latency service bindings")
        snapshot = _BundleSnapshot(
            bundle,
            uid=bindings.uid,
            gid=bindings.gid,
            forbidden_values=bindings.forbidden_values,
        )
        http_records, cases, http_lines = _http_records(
            gate, snapshot, fixtures, bindings.forbidden_values
        )
        claims, lifecycle, computed_samples, metrics, journal_lines = (
            _journal_and_claims(
                gate,
                snapshot,
                bindings,
                cases,
                _campaign_claim_factory(),
            )
        )
        observer_lines = _observer_records(gate, snapshot, lifecycle)
        correlation_lines = _correlations(gate, snapshot, bindings, lifecycle)
        sample_lines = _samples(gate, snapshot, computed_samples)
        _manifest(gate, snapshot, sources, fixtures)
        raw_counts = {
            "http-client.raw.jsonl": http_lines,
            "observer.raw.jsonl": observer_lines,
            "service-journal.raw.jsonl": journal_lines,
            "observer-journal-correlation.raw.jsonl": correlation_lines,
            "samples.raw.jsonl": sample_lines,
        }
        _summary(gate, snapshot, bindings, metrics, raw_counts)
        _checksums(snapshot)
        view = _derived_view(
            gate,
            bindings,
            sources,
            computed_samples,
            metrics,
            len(http_records),
            len(lifecycle),
            journal_lines,
        )
        snapshot.seal()
        for source in sources.values():
            source.seal()
        return LatencyGateIngestResult(tuple(http_records), tuple(claims), view)
    except LatencyGateIngestError:
        raise
    except Exception as error:
        raise LatencyGateIngestError("latency bundle ingestion failed") from error
    finally:
        if module_name is not None:
            sys.modules.pop(module_name, None)
        pending: LatencyGateIngestError | None = None
        if snapshot is not None:
            try:
                snapshot.close()
            except LatencyGateIngestError as error:
                pending = error
        for source in reversed(tuple(sources.values())):
            try:
                source.close()
            except LatencyGateIngestError as error:
                pending = error
        if pending is not None and sys.exc_info()[0] is None:
            raise pending


__all__ = [
    "LatencyGateIngestError",
    "LatencyGateIngestResult",
    "LatencyGateInputBindings",
    "VIEW_SCHEMA",
    "ingest_latency_gate_bundle",
]
