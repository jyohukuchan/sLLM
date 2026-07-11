#!/usr/bin/env python3
"""Revalidate and convert one non-GPU SQ8 API contract gate bundle."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import stat
import sys
import types
from pathlib import Path
from typing import Any, Iterator, NamedTuple, NoReturn, Sequence, cast


VIEW_SCHEMA = "ullm.sq8.api_contract_gate_ingest.view.v1"
CAMPAIGN_PHASE = "api_contract"
GATE_SCHEMA = "ullm.sq8.api_contract_gate.v1"
LIFECYCLE_SCHEMA = "ullm.gateway.lifecycle.v1"
EXPECTED_FILES = frozenset(
    {
        "http-client.raw.jsonl",
        "lifecycle-quiet.raw.jsonl",
        "service-journal.raw.jsonl",
        "input-manifest.json",
        "summary.json",
        "SHA256SUMS",
    }
)
CHECKSUM_INPUTS = tuple(
    sorted(EXPECTED_FILES - {"SHA256SUMS"}, key=lambda item: item.encode("utf-8"))
)
SOURCE_MANIFEST_PATHS = {
    "gate": "tools/run-sq8-api-contract-gate.py",
    "direct": "tools/run-sq8-direct-cancel-gate.py",
    "collector": "tools/collect-sq8-openwebui-release.py",
    "http_client": "tools/sq8-openwebui-http-client.py",
    "gateway_app": "services/openai-gateway/src/ullm_openai_gateway/app.py",
    "gateway_errors": "services/openai-gateway/src/ullm_openai_gateway/errors.py",
    "gateway_schemas": "services/openai-gateway/src/ullm_openai_gateway/schemas.py",
}
FILE_LIMITS = {
    "http-client.raw.jsonl": 32 << 20,
    "lifecycle-quiet.raw.jsonl": 4 << 20,
    "service-journal.raw.jsonl": 64 << 20,
    "input-manifest.json": 4 << 20,
    "summary.json": 8 << 20,
    "SHA256SUMS": 1 << 20,
}
SOURCE_LIMITS = {
    "gate": 4 << 20,
    "direct": 4 << 20,
    "collector": 8 << 20,
    "http_client": 4 << 20,
    "gateway_app": 2 << 20,
    "gateway_errors": 1 << 20,
    "gateway_schemas": 2 << 20,
}
REQUIRED_JOURNAL_FIELDS = {
    "__CURSOR",
    "__MONOTONIC_TIMESTAMP",
    "_BOOT_ID",
    "_PID",
    "_SYSTEMD_UNIT",
    "PRIORITY",
    "MESSAGE",
}
MAX_JSON_LINE_BYTES = 1 << 20
COPY_CHUNK_BYTES = 64 << 10
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
BOOT_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
CONTENT_IMAGE_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
NETWORK_ID_RE = re.compile(r"[0-9a-f]{64}\Z")
SERVICE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@:-]{0,127}\.service\Z")


class ApiContractIngestError(RuntimeError):
    """A fail-closed adapter error without evidence-derived diagnostics."""


def fail(message: str) -> NoReturn:
    raise ApiContractIngestError(message)


@dataclasses.dataclass(frozen=True)
class ApiContractInputBindings:
    gate_source: Path
    gate_source_sha256: str
    direct_source: Path
    direct_source_sha256: str
    collector_source: Path
    collector_source_sha256: str
    http_client_source: Path
    http_client_source_sha256: str
    gateway_app_source: Path
    gateway_app_source_sha256: str
    gateway_errors_source: Path
    gateway_errors_source_sha256: str
    gateway_schemas_source: Path
    gateway_schemas_source_sha256: str
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


class ApiContractIngestResult(NamedTuple):
    http_records: tuple[dict[str, Any], ...]
    journal_records: tuple[dict[str, Any], ...]
    quiet_check_records: tuple[dict[str, Any], ...]
    derived_view: dict[str, Any]
    final_journal_cursor: str


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
        fail("O_NOFOLLOW is required for API contract ingestion")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _file_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for API contract ingestion")
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


def _safe_close(fd: int) -> None:
    if fd < 0:
        return
    try:
        os.close(fd)
    except OSError:
        fail("failed to close a sealed API contract descriptor")


def _entry_identity(parent_fd: int, name: str) -> _Identity:
    try:
        return _Identity.from_stat(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        )
    except OSError:
        fail("sealed API contract directory entry is unavailable")


class _SecretScanner:
    def __init__(self, values: tuple[bytes, ...]):
        for value in values:
            if type(value) is not bytes or len(value) < 4:
                fail("forbidden values must be byte strings of length >= 4")
        self._values = values
        self._overlap = max((len(value) for value in values), default=1) - 1
        self._tail = b""

    def consume(self, chunk: bytes) -> None:
        combined = self._tail + chunk
        if any(value in combined for value in self._values):
            fail("API contract evidence contains forbidden cleartext")
        self._tail = combined[-self._overlap :] if self._overlap else b""


class BundleSnapshot:
    """Directory-FD snapshot of the exact six-file final gate layout."""

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
        self.root_identity: _Identity | None = None
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
            root_identity = _Identity.from_stat(os.fstat(self.root_fd))
            if _entry_identity(self.parent_fd, self.root_path.name) != root_identity:
                fail("API contract root identity changed while opening")
            if (
                not stat.S_ISDIR(root_identity.mode)
                or stat.S_IMODE(root_identity.mode) != 0o500
                or root_identity.links != 2
                or root_identity.uid != self.uid
                or root_identity.gid != self.gid
                or frozenset(os.listdir(self.root_fd)) != EXPECTED_FILES
            ):
                fail("API contract bundle root layout, mode, or owner differs")
            self.root_identity = root_identity
            for name in EXPECTED_FILES:
                entry = _entry_identity(self.root_fd, name)
                if (
                    not stat.S_ISREG(entry.mode)
                    or stat.S_IMODE(entry.mode) != 0o600
                    or entry.links != 1
                    or entry.uid != self.uid
                    or entry.gid != self.gid
                    or entry.size > FILE_LIMITS[name]
                    or (name != "service-journal.raw.jsonl" and entry.size < 1)
                ):
                    fail("API contract artifact layout, mode, owner, or size differs")
                fd = os.open(name, _file_flags(), dir_fd=self.root_fd)
                opened = _Identity.from_stat(os.fstat(fd))
                if opened != entry:
                    os.close(fd)
                    fail("API contract artifact identity changed while opening")
                self.files[name] = _OpenedFile(name, fd, opened, FILE_LIMITS[name])
        except ApiContractIngestError:
            self.close()
            raise
        except OSError:
            self.close()
            fail("failed to open the API contract bundle without following links")

    def __enter__(self) -> BundleSnapshot:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _chunks(self, name: str) -> Iterator[bytes]:
        if self.closed or self.sealed:
            fail("API contract bundle snapshot is no longer readable")
        item = self.files.get(name)
        if item is None or item.consumed:
            fail("API contract artifact was requested outside its fixed schedule")
        try:
            if _Identity.from_stat(os.fstat(item.fd)) != item.identity:
                fail("API contract artifact changed before streaming")
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
                    fail("API contract artifact exceeded its streaming bound")
                digest.update(chunk)
                scanner.consume(chunk)
                yield chunk
            if (
                _Identity.from_stat(os.fstat(item.fd)) != item.identity
                or total != item.identity.size
            ):
                fail("API contract artifact changed while streaming")
            item.streamed_bytes = total
            item.sha256 = digest.hexdigest()
            item.consumed = True
        except ApiContractIngestError:
            raise
        except OSError:
            fail("failed to stream an API contract artifact")

    def read_small(self, name: str) -> bytes:
        chunks: list[bytes] = []
        total = 0
        for chunk in self._chunks(name):
            total += len(chunk)
            if total > FILE_LIMITS[name]:
                fail("API contract document exceeded its bound")
            chunks.append(chunk)
        return b"".join(chunks)

    def iter_lines(self, name: str) -> Iterator[bytes]:
        pending = bytearray()
        for chunk in self._chunks(name):
            pending.extend(chunk)
            while True:
                index = pending.find(b"\n")
                if index < 0:
                    if len(pending) > MAX_JSON_LINE_BYTES:
                        fail("API contract JSONL line exceeded its bound")
                    break
                raw = bytes(pending[:index])
                del pending[: index + 1]
                if not raw or raw.endswith(b"\r") or len(raw) > MAX_JSON_LINE_BYTES:
                    fail("API contract JSONL framing differs")
                yield raw
        if pending:
            fail("API contract JSONL lacks a final LF")

    def evidence(self, name: str) -> tuple[int, str]:
        item = self.files.get(name)
        if item is None or not item.consumed or item.sha256 is None:
            fail("API contract artifact has not been completely consumed")
        return item.streamed_bytes, item.sha256

    def seal(self) -> None:
        if self.closed or self.sealed:
            fail("API contract bundle cannot be sealed in its current state")
        if any(not item.consumed for item in self.files.values()):
            fail("not every API contract artifact was consumed before sealing")
        assert self.root_identity is not None
        try:
            if (
                frozenset(os.listdir(self.root_fd)) != EXPECTED_FILES
                or _Identity.from_stat(os.fstat(self.root_fd)) != self.root_identity
                or _entry_identity(self.parent_fd, self.root_path.name)
                != self.root_identity
            ):
                fail("API contract bundle layout or root identity changed")
            for item in self.files.values():
                if (
                    _Identity.from_stat(os.fstat(item.fd)) != item.identity
                    or _entry_identity(self.root_fd, item.name) != item.identity
                ):
                    fail("API contract artifact entry identity changed before sealing")
                os.lseek(item.fd, 0, os.SEEK_SET)
                digest = hashlib.sha256()
                scanner = _SecretScanner(self.forbidden_values)
                total = 0
                while chunk := os.read(item.fd, COPY_CHUNK_BYTES):
                    total += len(chunk)
                    scanner.consume(chunk)
                    digest.update(chunk)
                if (
                    total != item.streamed_bytes
                    or digest.hexdigest() != item.sha256
                    or _Identity.from_stat(os.fstat(item.fd)) != item.identity
                    or _entry_identity(self.root_fd, item.name) != item.identity
                ):
                    fail("API contract artifact hash or identity changed at seal")
            if _Identity.from_stat(os.fstat(self.root_fd)) != self.root_identity:
                fail("API contract directory changed during final hashing")
            self.sealed = True
        except ApiContractIngestError:
            raise
        except OSError:
            fail("failed to seal the API contract bundle")

    def close(self) -> None:
        if self.closed:
            return
        pending: ApiContractIngestError | None = None
        for item in self.files.values():
            try:
                _safe_close(item.fd)
            except ApiContractIngestError as error:
                pending = error
        self.files.clear()
        for fd in (self.root_fd, self.parent_fd):
            try:
                _safe_close(fd)
            except ApiContractIngestError as error:
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
        if SHA256_RE.fullmatch(expected_sha256) is None:
            fail("bound source SHA-256 syntax differs")
        try:
            self.parent_fd = os.open(self.path.parent, _directory_flags())
            entry = _entry_identity(self.parent_fd, self.path.name)
            if (
                not stat.S_ISREG(entry.mode)
                or entry.links != 1
                or entry.size < 1
                or entry.size > maximum
            ):
                fail("bound source is not one bounded regular file")
            self.fd = os.open(self.path.name, _file_flags(), dir_fd=self.parent_fd)
            opened = _Identity.from_stat(os.fstat(self.fd))
            if opened != entry:
                fail("bound source identity changed while opening")
            self.identity = opened
            self.raw, self.sha256 = self._snapshot()
            if self.sha256 != expected_sha256:
                fail("bound source SHA-256 differs")
        except ApiContractIngestError:
            self.close()
            raise
        except OSError:
            self.close()
            fail("failed to open a bound API source")

    def _snapshot(self) -> tuple[bytes, str]:
        assert self.identity is not None
        try:
            if _Identity.from_stat(os.fstat(self.fd)) != self.identity:
                fail("bound source identity changed before reading")
            os.lseek(self.fd, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            scanner = _SecretScanner(self.forbidden_values)
            total = 0
            while chunk := os.read(self.fd, COPY_CHUNK_BYTES):
                total += len(chunk)
                if total > self.maximum:
                    fail("bound source exceeded its streaming bound")
                scanner.consume(chunk)
                digest.update(chunk)
                chunks.append(chunk)
            if (
                total != self.identity.size
                or _Identity.from_stat(os.fstat(self.fd)) != self.identity
            ):
                fail("bound source changed while reading")
            return b"".join(chunks), digest.hexdigest()
        except ApiContractIngestError:
            raise
        except OSError:
            fail("failed to stream a bound API source")

    def seal(self) -> None:
        assert self.identity is not None
        if (
            _Identity.from_stat(os.fstat(self.fd)) != self.identity
            or _entry_identity(self.parent_fd, self.path.name) != self.identity
        ):
            fail("bound source entry changed before sealing")
        raw, digest = self._snapshot()
        if raw != self.raw or digest != self.sha256:
            fail("bound source bytes changed before sealing")
        if (
            _Identity.from_stat(os.fstat(self.fd)) != self.identity
            or _entry_identity(self.parent_fd, self.path.name) != self.identity
        ):
            fail("bound source identity changed during sealing")

    def close(self) -> None:
        pending: ApiContractIngestError | None = None
        for fd in (self.fd, self.parent_fd):
            try:
                _safe_close(fd)
            except ApiContractIngestError as error:
                pending = error
        self.fd = self.parent_fd = -1
        if pending is not None:
            raise pending


def _source_specs(
    bindings: ApiContractInputBindings,
) -> dict[str, tuple[Path, str]]:
    return {
        "gate": (bindings.gate_source, bindings.gate_source_sha256),
        "direct": (bindings.direct_source, bindings.direct_source_sha256),
        "collector": (bindings.collector_source, bindings.collector_source_sha256),
        "http_client": (
            bindings.http_client_source,
            bindings.http_client_source_sha256,
        ),
        "gateway_app": (
            bindings.gateway_app_source,
            bindings.gateway_app_source_sha256,
        ),
        "gateway_errors": (
            bindings.gateway_errors_source,
            bindings.gateway_errors_source_sha256,
        ),
        "gateway_schemas": (
            bindings.gateway_schemas_source,
            bindings.gateway_schemas_source_sha256,
        ),
    }


def _validate_bindings(bindings: ApiContractInputBindings) -> None:
    if not isinstance(bindings, ApiContractInputBindings):
        fail("API contract input bindings have the wrong type")
    specs = _source_specs(bindings)
    if len({Path(os.path.abspath(path)) for path, _sha in specs.values()}) != 7:
        fail("bound API source paths are not unique")
    for path, digest in specs.values():
        if not isinstance(path, os.PathLike) or SHA256_RE.fullmatch(digest) is None:
            fail("bound API source path or SHA-256 differs")
    if (
        bindings.gate_source.name != "run-sq8-api-contract-gate.py"
        or bindings.direct_source
        != bindings.gate_source.with_name("run-sq8-direct-cancel-gate.py")
        or bindings.collector_source
        != bindings.gate_source.with_name("collect-sq8-openwebui-release.py")
        or bindings.http_client_source
        != bindings.gate_source.with_name("sq8-openwebui-http-client.py")
    ):
        fail("bound gate and support source layout differs")
    if CONTENT_IMAGE_RE.fullmatch(bindings.http_image_id) is None:
        fail("bound HTTP image content identity differs")
    if NETWORK_ID_RE.fullmatch(bindings.docker_network_id) is None:
        fail("bound Docker network identity differs")
    if SERVICE_RE.fullmatch(bindings.service_unit) is None:
        fail("bound service unit syntax differs")
    if BOOT_ID_RE.fullmatch(bindings.boot_id) is None:
        fail("bound boot ID syntax differs")
    if (
        type(bindings.service_user) is not str
        or not bindings.service_user
        or "/" in bindings.service_user
        or bindings.control_group != f"/system.slice/{bindings.service_unit}"
    ):
        fail("bound service user or control group differs")
    for value, minimum in (
        (bindings.gateway_pid, 1),
        (bindings.gateway_starttime_ticks, 1),
        (bindings.worker_pid, 1),
        (bindings.worker_starttime_ticks, 1),
        (bindings.restart_count, 0),
        (bindings.uid, 0),
        (bindings.gid, 0),
    ):
        if type(value) is not int or value < minimum:
            fail("bound service numeric identity differs")
    if bindings.gateway_pid == bindings.worker_pid:
        fail("bound gateway and worker PID identities collide")
    if type(bindings.forbidden_values) is not tuple:
        fail("forbidden evidence values must be an immutable tuple")
    _SecretScanner(bindings.forbidden_values)


def _load_gate(
    sources: dict[str, _StableSource],
) -> tuple[Any, str, dict[str, types.ModuleType | None]]:
    gate_source = sources["gate"]
    module_name = f"_ullm_api_contract_ingest_{os.getpid()}_{id(gate_source):x}"
    support_names = (
        "_ullm_sq8_api_contract_direct_support",
        "_ullm_sq8_cancel_collector_support",
    )
    prior = {name: sys.modules.get(name) for name in support_names}
    module = types.ModuleType(module_name)
    module.__file__ = os.fspath(gate_source.path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        code = compile(
            gate_source.raw, os.fspath(gate_source.path), "exec", dont_inherit=True
        )
        exec(code, module.__dict__)
        if (
            module.DIRECT_SUPPORT_RAW != sources["direct"].raw
            or module.DIRECT.COLLECTOR_SUPPORT_RAW != sources["collector"].raw
            or module.HTTP_CLIENT_SHA256 != sources["http_client"].sha256
            or module.GATE_SCHEMA != GATE_SCHEMA
        ):
            fail("executed API gate validator differs from bound sources")
        return module, module_name, prior
    except ApiContractIngestError:
        sys.modules.pop(module_name, None)
        _restore_modules(prior)
        raise
    except Exception as error:
        sys.modules.pop(module_name, None)
        _restore_modules(prior)
        raise ApiContractIngestError(
            "failed to load the bound API gate validator"
        ) from error


def _restore_modules(prior: dict[str, types.ModuleType | None]) -> None:
    for name, module in prior.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _strict_object(gate: Any, raw: bytes, label: str) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], gate.strict_json_object(raw, label))
    except Exception as error:
        raise ApiContractIngestError(f"{label} is invalid") from error


def _exact(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        fail(f"{label} fields differ")
    return cast(dict[str, Any], value)


def _same_json_type_and_value(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        actual_object = cast(dict[str, Any], actual)
        expected_object = cast(dict[str, Any], expected)
        return set(actual_object) == set(expected_object) and all(
            _same_json_type_and_value(actual_object[key], expected_object[key])
            for key in expected_object
        )
    if type(expected) is list:
        actual_array = cast(list[Any], actual)
        expected_array = expected
        return len(actual_array) == len(expected_array) and all(
            _same_json_type_and_value(actual_item, expected_item)
            for actual_item, expected_item in zip(
                actual_array, expected_array, strict=True
            )
        )
    return bool(actual == expected)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail(f"{label} is not an integer >= {minimum}")
    return value


def _decimal(value: Any, label: str) -> int:
    if (
        type(value) is not str
        or not value.isascii()
        or not value.isdecimal()
        or len(value) > 32
    ):
        fail(f"{label} is not a bounded decimal string")
    return int(value, 10)


def _text(value: Any, label: str, *, maximum: int = 4096) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        fail(f"{label} is not bounded non-empty text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        fail(f"{label} is not strict UTF-8")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            fail("JSON document contains a duplicate key")
        value[key] = item
    return value


def _reject_constant(_value: str) -> None:
    fail("JSON document contains a non-finite number")


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        fail("JSON document contains a non-finite number")
    return parsed


def _document(raw: bytes, label: str) -> dict[str, Any]:
    if not raw.endswith(b"\n") or raw.endswith(b"\r\n"):
        fail(f"{label} is not one LF-terminated JSON document")
    try:
        value = json.loads(
            raw[:-1].decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=_parse_float,
            parse_constant=_reject_constant,
        )
    except ApiContractIngestError:
        raise
    except (UnicodeError, ValueError, RecursionError):
        fail(f"{label} is not strict JSON")
    if type(value) is not dict:
        fail(f"{label} root is not an object")
    return cast(dict[str, Any], value)


def _http_records(
    gate: Any, snapshot: BundleSnapshot
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    iterator = iter(snapshot.iter_lines("http-client.raw.jsonl"))
    line_count = 0

    def next_event(label: str) -> tuple[bytes, dict[str, Any]]:
        nonlocal line_count
        try:
            raw = next(iterator)
        except StopIteration:
            fail(f"{label} is missing")
        line_count += 1
        return raw, _strict_object(gate, raw, label)

    _ready_raw, ready = next_event("HTTP ready event")
    _exact(
        ready,
        {"schema_version", "event", "observed_monotonic_ns"},
        "HTTP ready event",
    )
    if ready["schema_version"] != gate.HTTP_EVENT_SCHEMA or ready["event"] != "ready":
        fail("HTTP ready event differs")
    ready_ns = _integer(ready["observed_monotonic_ns"], "HTTP ready timestamp")
    records: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    previous_end = -1
    for case_index, case in enumerate(gate.FROZEN_SCHEDULE, start=1):
        expected_key = f"api-contract-{case_index:02d}-{case.case_id}"
        raw_events: list[tuple[bytes, dict[str, Any]]] = []
        while True:
            raw, event = next_event(f"HTTP case {case_index} event")
            raw_events.append((raw, event))
            if len(raw_events) > gate.MAX_HTTP_EVENTS:
                fail("HTTP case event count exceeds its bound")
            if event.get("event") == "http_response_end":
                break
        try:
            observation = gate.parse_http_events(
                case,
                expected_key,
                [event for _raw, event in raw_events],
                previous_response_end_ns=previous_end,
            )
            case_summary = gate.validate_case_observation(case, observation, case_index)
        except Exception as error:
            raise ApiContractIngestError("raw HTTP case validation failed") from error
        if case_index == 1 and observation.connect_completed_monotonic_ns < ready_ns:
            fail("first HTTP request precedes the client ready event")
        previous_end = observation.response_end_monotonic_ns
        case_summaries.append(cast(dict[str, Any], case_summary))
        for _raw, event in raw_events:
            record_type = event["event"]
            if record_type not in {
                "http_request",
                "http_response_start",
                "http_body_chunk",
                "http_response_end",
            }:
                fail("HTTP case contains a non-campaign event")
            fields = {
                key: copy.deepcopy(value)
                for key, value in event.items()
                if key not in {"schema_version", "event"}
            }
            if record_type == "http_request":
                fields = {"request_index": case_index, **fields}
            records.append(
                {
                    "record_type": record_type,
                    "phase": CAMPAIGN_PHASE,
                    "case_id": case.case_id,
                    "fields": fields,
                }
            )

    _shutdown_raw, shutdown = next_event("HTTP shutdown event")
    _exact(
        shutdown,
        {"schema_version", "event", "observed_monotonic_ns"},
        "HTTP shutdown event",
    )
    shutdown_ns = _integer(shutdown["observed_monotonic_ns"], "HTTP shutdown timestamp")
    if (
        shutdown["schema_version"] != gate.HTTP_EVENT_SCHEMA
        or shutdown["event"] != "shutdown_complete"
        or shutdown_ns < previous_end
    ):
        fail("HTTP shutdown event differs or precedes the final response")
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        fail("HTTP evidence contains events after shutdown_complete")
    return records, case_summaries, line_count, shutdown_ns


def _journal_records(
    gate: Any,
    snapshot: BundleSnapshot,
    bindings: ApiContractInputBindings,
) -> tuple[list[str], list[int], list[dict[str, Any]], int]:
    cursors: list[str] = []
    monotonic_values: list[int] = []
    campaign_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    last_monotonic = -1
    count = 0
    for raw in snapshot.iter_lines("service-journal.raw.jsonl"):
        count += 1
        record = _strict_object(gate, raw, "service journal record")
        if set(record) != REQUIRED_JOURNAL_FIELDS:
            fail("service journal required field set differs")
        cursor = _text(record["__CURSOR"], "service journal cursor", maximum=65_536)
        if cursor in seen:
            fail("service journal cursor is duplicated")
        seen.add(cursor)
        monotonic = _decimal(
            record["__MONOTONIC_TIMESTAMP"], "service journal monotonic timestamp"
        )
        pid = _decimal(record["_PID"], "service journal PID")
        priority = _decimal(record["PRIORITY"], "service journal priority")
        message = record["MESSAGE"]
        if (
            monotonic < last_monotonic
            or pid != bindings.gateway_pid
            or priority > 7
            or record["_BOOT_ID"] != bindings.boot_id
            or record["_SYSTEMD_UNIT"] != bindings.service_unit
            or type(message) is not str
        ):
            fail("service journal ordering or bound identity differs")
        try:
            lifecycle = gate.DIRECT.COL.decode_lifecycle_message(message)
        except Exception as error:
            raise ApiContractIngestError(
                "service journal MESSAGE is invalid"
            ) from error
        if lifecycle is not None:
            fail("non-GPU API contract journal contains a lifecycle event")
        cursors.append(cursor)
        monotonic_values.append(monotonic)
        message_raw = message.encode("utf-8", errors="strict")
        campaign_records.append(
            {
                "record_type": "api_journal_observation",
                "phase": CAMPAIGN_PHASE,
                "case_id": f"api-journal-{count:02d}",
                "fields": {
                    "observation_index": count - 1,
                    "journal_cursor": cursor,
                    "journal_monotonic_usec": monotonic,
                    "journal_pid": pid,
                    "message_utf8_bytes": len(message_raw),
                    "message_sha256": hashlib.sha256(message_raw).hexdigest(),
                },
            }
        )
        last_monotonic = monotonic
    if not cursors:
        fail("service journal contains no authoritative API access records")
    return cursors, monotonic_values, campaign_records, count


def _quiet_checks(
    gate: Any,
    snapshot: BundleSnapshot,
    cursors: Sequence[str],
    journal_count: int,
    case_end_times: Sequence[int],
) -> tuple[list[dict[str, Any]], int]:
    labels = [case.case_id for case in gate.FROZEN_SCHEDULE] + [
        "http-client-shutdown",
        "post-observer-close",
        "final-readiness-and-identity",
    ]
    checks: list[dict[str, Any]] = []
    prior_count = 0
    prior_checked_ns = -1
    line_count = 0
    for raw in snapshot.iter_lines("lifecycle-quiet.raw.jsonl"):
        line_count += 1
        value = _strict_object(gate, raw, "lifecycle quiet check")
        _exact(
            value,
            {
                "schema_version",
                "record_type",
                "sequence",
                "label",
                "checked_monotonic_ns",
                "observer_open",
                "observer_event_count",
                "new_journal_record_count",
                "journal_record_count",
                "journal_cursor",
            },
            "lifecycle quiet check",
        )
        sequence = _integer(value["sequence"], "quiet sequence")
        if sequence >= len(labels):
            fail("lifecycle quiet check count exceeds the fixed schedule")
        checked_ns = _integer(value["checked_monotonic_ns"], "quiet timestamp")
        current_count = _integer(
            value["journal_record_count"], "quiet journal record count"
        )
        new_count = _integer(
            value["new_journal_record_count"], "quiet new journal count"
        )
        observer_event_count = _integer(
            value["observer_event_count"], "quiet observer event count"
        )
        expected_open = sequence <= 10
        if (
            value["schema_version"] != GATE_SCHEMA
            or value["record_type"] != "lifecycle_quiet_check"
            or sequence != line_count - 1
            or value["label"] != labels[sequence]
            or type(value["observer_open"]) is not bool
            or value["observer_open"] is not expected_open
            or observer_event_count != 0
            or checked_ns < prior_checked_ns
            or current_count <= 0
            or current_count > journal_count
            or current_count < prior_count
            or new_count != current_count - prior_count
            or value["journal_cursor"] != cursors[current_count - 1]
        ):
            fail("lifecycle quiet check order, identity, or count differs")
        if sequence < 10 and checked_ns < case_end_times[sequence]:
            fail("lifecycle quiet check precedes its HTTP response end")
        prior_count = current_count
        prior_checked_ns = checked_ns
        checks.append(value)
    if line_count != 13 or prior_count != journal_count:
        fail("lifecycle quiet checks do not cover the complete journal")
    return checks, line_count


def _campaign_quiet_check_records(
    checks: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for value in checks:
        label = cast(str, value["label"])
        records.append(
            {
                "record_type": "lifecycle_quiet_check",
                "phase": CAMPAIGN_PHASE,
                "case_id": label,
                "fields": {
                    "quiet_sequence": value["sequence"],
                    "label": label,
                    "checked_monotonic_ns": value["checked_monotonic_ns"],
                    "observer_open": value["observer_open"],
                    "observer_event_count": value["observer_event_count"],
                    "new_journal_record_count": value["new_journal_record_count"],
                    "journal_record_count": value["journal_record_count"],
                    "journal_cursor": value["journal_cursor"],
                },
            }
        )
    return tuple(records)


def _manifest(
    gate: Any,
    snapshot: BundleSnapshot,
    sources: dict[str, _StableSource],
) -> dict[str, Any]:
    value = _document(snapshot.read_small("input-manifest.json"), "input manifest")
    gateway_sources = {
        SOURCE_MANIFEST_PATHS[key]: sources[key].raw
        for key in ("gateway_app", "gateway_errors", "gateway_schemas")
    }
    try:
        expected = gate.build_input_manifest(
            sources["gate"].raw,
            sources["http_client"].raw,
            gateway_sources,
        )
    except Exception as error:
        raise ApiContractIngestError(
            "bound gate manifest reconstruction failed"
        ) from error
    if not _same_json_type_and_value(value, expected):
        fail("input manifest differs from bound source and request body identities")
    return value


def _service_identity(bindings: ApiContractInputBindings) -> dict[str, Any]:
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


def _summary(
    gate: Any,
    snapshot: BundleSnapshot,
    bindings: ApiContractInputBindings,
    case_summaries: list[dict[str, Any]],
    raw_counts: dict[str, int],
) -> dict[str, Any]:
    value = _document(snapshot.read_small("summary.json"), "producer summary")
    artifacts = {
        name: {
            "bytes": snapshot.evidence(name)[0],
            "lines": raw_counts[name],
            "sha256": snapshot.evidence(name)[1],
        }
        for name in (
            "http-client.raw.jsonl",
            "service-journal.raw.jsonl",
            "lifecycle-quiet.raw.jsonl",
        )
    }
    expected = {
        "schema_version": GATE_SCHEMA,
        "record_type": "summary",
        "model_id": gate.MODEL_ID,
        "request_count": 10,
        "max_active": 1,
        "service_identity": _service_identity(bindings),
        "http_image_id": bindings.http_image_id,
        "docker_network_name": gate.DIRECT.HTTP_NETWORK_NAME,
        "docker_network_id": bindings.docker_network_id,
        "observer_socket": os.fspath(gate.OBSERVER_SOCKET),
        "observer_event_count": 0,
        "lifecycle_event_count": 0,
        "quiet_check_count": 13,
        "cases": case_summaries,
        "artifacts": artifacts,
    }
    if not _same_json_type_and_value(value, expected):
        fail("producer summary differs from independently reconstructed evidence")
    return value


def _checksums(snapshot: BundleSnapshot) -> None:
    raw = snapshot.read_small("SHA256SUMS")
    try:
        text_value = raw.decode("ascii", errors="strict")
    except UnicodeError:
        fail("SHA256SUMS is not ASCII")
    expected_lines = [
        f"{snapshot.evidence(name)[1]}  {name}\n" for name in CHECKSUM_INPUTS
    ]
    if text_value != "".join(expected_lines):
        fail("SHA256SUMS differs from independently hashed artifacts")


def _derived_view(
    gate: Any,
    bindings: ApiContractInputBindings,
    sources: dict[str, _StableSource],
    case_summaries: Sequence[dict[str, Any]],
    http_record_count: int,
    journal_count: int,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for value in case_summaries:
        error = value["error"]
        cases.append(
            {
                "case_index": value["case_index"],
                "case_id": value["case_id"],
                "status": value["status"],
                "request_body_bytes": value["request_body_bytes"],
                "request_body_sha256": value["request_body_sha256"],
                "response_body_bytes": value["response_body_bytes"],
                "response_body_sha256": value["response_body_sha256"],
                "error": None
                if error is None
                else {
                    "type": error["type"],
                    "code": error["code"],
                    "param": error["param"],
                    "message_utf8_bytes": error["message_utf8_bytes"],
                    "message_sha256": error["message_sha256"],
                },
            }
        )
    view = {
        "schema_version": VIEW_SCHEMA,
        "case_count": 10,
        "http_record_count": http_record_count,
        "journal_record_count": journal_count,
        "lifecycle_event_count": 0,
        "quiet_check_count": 13,
        "cases": cases,
        "source_bindings": {
            **{f"{key}_sha256": source.sha256 for key, source in sources.items()},
            "http_image_id": bindings.http_image_id,
            "docker_network_id": bindings.docker_network_id,
            "service_unit": bindings.service_unit,
            "boot_id": bindings.boot_id,
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
        raise ApiContractIngestError("derived API view cannot be encoded") from error
    scanner = _SecretScanner(bindings.forbidden_values)
    scanner.consume(raw)
    for message in (
        gate.INVALID_KEY_MESSAGE,
        gate.QUERY_MESSAGE,
        gate.INVALID_JSON_MESSAGE,
        gate.UNSUPPORTED_MESSAGE,
        gate.MODEL_NOT_FOUND_MESSAGE,
        "API contract preflight",
    ):
        if message.encode("ascii") in raw:
            fail("derived API view contains prompt or public message cleartext")
    for path, _digest in _source_specs(bindings).values():
        try:
            path_raw = os.fspath(path).encode("utf-8", errors="strict")
        except UnicodeError:
            fail("bound source path is not strict UTF-8")
        if path_raw and path_raw in raw:
            fail("derived API view contains a host source path")
    return view


def ingest_api_contract_bundle(
    bundle: Path,
    bindings: ApiContractInputBindings,
) -> ApiContractIngestResult:
    _validate_bindings(bindings)
    sources: dict[str, _StableSource] = {}
    snapshot: BundleSnapshot | None = None
    gate: Any | None = None
    module_name: str | None = None
    prior_modules: dict[str, types.ModuleType | None] = {}
    try:
        for key, (path, digest) in _source_specs(bindings).items():
            sources[key] = _StableSource(
                path,
                f"bound {key} source",
                SOURCE_LIMITS[key],
                digest,
                bindings.forbidden_values,
            )
        gate, module_name, prior_modules = _load_gate(sources)
        snapshot = BundleSnapshot(
            bundle,
            uid=bindings.uid,
            gid=bindings.gid,
            forbidden_values=bindings.forbidden_values,
        )
        records, case_summaries, http_lines, _shutdown_ns = _http_records(
            gate, snapshot
        )
        case_end_times = [
            value["response_end_monotonic_ns"] for value in case_summaries
        ]
        cursors, _journal_monotonic, journal_records, journal_lines = _journal_records(
            gate, snapshot, bindings
        )
        checks, quiet_lines = _quiet_checks(
            gate,
            snapshot,
            cursors,
            journal_lines,
            case_end_times,
        )
        _manifest(gate, snapshot, sources)
        raw_counts = {
            "http-client.raw.jsonl": http_lines,
            "service-journal.raw.jsonl": journal_lines,
            "lifecycle-quiet.raw.jsonl": quiet_lines,
        }
        _summary(gate, snapshot, bindings, case_summaries, raw_counts)
        _checksums(snapshot)
        view = _derived_view(
            gate,
            bindings,
            sources,
            case_summaries,
            len(records),
            journal_lines,
        )
        snapshot.seal()
        for source in sources.values():
            source.seal()
        return ApiContractIngestResult(
            tuple(records),
            tuple(journal_records),
            _campaign_quiet_check_records(checks),
            view,
            cursors[-1],
        )
    except ApiContractIngestError:
        raise
    except Exception as error:
        raise ApiContractIngestError("API contract bundle ingestion failed") from error
    finally:
        if module_name is not None:
            sys.modules.pop(module_name, None)
            _restore_modules(prior_modules)
        if snapshot is not None:
            try:
                snapshot.close()
            except ApiContractIngestError:
                pass
        for source in reversed(tuple(sources.values())):
            try:
                source.close()
            except ApiContractIngestError:
                pass


__all__ = [
    "ApiContractIngestError",
    "ApiContractIngestResult",
    "ApiContractInputBindings",
    "BundleSnapshot",
    "ingest_api_contract_bundle",
]
