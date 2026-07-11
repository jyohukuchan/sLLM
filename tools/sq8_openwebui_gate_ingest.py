#!/usr/bin/env python3
"""Revalidate and convert a combined OpenWebUI smoke/soak gate bundle."""

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
from typing import Any, Callable, Iterator, NamedTuple, NoReturn, Protocol, cast


INGEST_VIEW_SCHEMA = "ullm.sq8.openwebui_gate_ingest.combined_view.v1"
CAMPAIGN_PHASE = "openwebui"
CAMPAIGN_SERVICE_UNIT = "ullm-openai.service"
ROOT_FILES = frozenset(
    {"observer.raw.jsonl", "service-journal.raw.jsonl", "summary.json", "browser"}
)
BROWSER_FILES = frozenset({"browser-stdout.jsonl", "openwebui-soak-summary.json"})
REQUIRED_JOURNAL_FIELDS = (
    "__CURSOR",
    "__MONOTONIC_TIMESTAMP",
    "_BOOT_ID",
    "_PID",
    "_SYSTEMD_UNIT",
    "PRIORITY",
    "MESSAGE",
)
MAX_SUMMARY_BYTES = (1 << 20) + 1
MAX_OBSERVER_BYTES = 16 << 20
MAX_JOURNAL_BYTES = 64 << 20
MAX_BROWSER_STDOUT_BYTES = 32 << 20
MAX_JSON_LINE_BYTES = 1 << 20
MAX_LIFECYCLE_PAYLOAD_BYTES = 64 << 10
MAX_LIFECYCLE_RECORDS = 256
MAX_GATE_SOURCE_BYTES = 2 << 20
MAX_SUPPORT_SOURCE_BYTES = 4 << 20
MAX_BROWSER_SCRIPT_BYTES = 2 << 20
COPY_CHUNK_BYTES = 64 << 10
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
BOOT_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
SERVICE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@:-]{0,127}\.service\Z")


class GateIngestError(RuntimeError):
    """A fail-closed conversion error which never embeds evidence values."""


def fail(message: str) -> NoReturn:
    raise GateIngestError(message)


class BundleLifecycleClaimProtocol(Protocol):
    raw: bytes
    phase: str
    case_id: str


@dataclasses.dataclass(frozen=True)
class GateInputBindings:
    gate_source: Path
    gate_source_sha256: str
    support_source: Path
    support_source_sha256: str
    browser_script: Path
    browser_script_sha256: str
    browser_image_reference: str
    browser_image_content_id: str
    openwebui_base_url: str
    service_unit: str
    boot_id: str
    gateway_pid: int
    uid: int
    gid: int
    restart_count: int
    forbidden_values: tuple[bytes, ...] = ()


class CombinedSoakIngestResult(NamedTuple):
    browser_action_records: tuple[dict[str, Any], ...]
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
        fail("O_NOFOLLOW is required for gate bundle ingestion")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)


def _file_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for gate bundle ingestion")
    return os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)


def _safe_close(fd: int) -> None:
    if fd < 0:
        return
    try:
        os.close(fd)
    except OSError:
        fail("failed to close a sealed evidence descriptor")


def _entry_identity(parent_fd: int, name: str) -> _Identity:
    try:
        return _Identity.from_stat(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        )
    except OSError:
        fail("sealed evidence directory entry is unavailable")


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
        fail("gate bundle directory mode, owner, or link count differs")


def _require_file(
    identity: _Identity,
    *,
    maximum: int,
    mode: int,
    uid: int,
    gid: int,
) -> None:
    if (
        not stat.S_ISREG(identity.mode)
        or stat.S_IMODE(identity.mode) != mode
        or identity.uid != uid
        or identity.gid != gid
        or identity.links != 1
        or identity.size < 1
        or identity.size > maximum
    ):
        fail("gate bundle file layout, owner, mode, link count, or size differs")


class _SecretScanner:
    def __init__(self, values: tuple[bytes, ...]):
        for value in values:
            if type(value) is not bytes or len(value) < 4:
                fail("forbidden evidence values must be byte strings of length >= 4")
        self._values = values
        self._overlap = max((len(value) for value in values), default=1) - 1
        self._tail = b""

    def consume(self, chunk: bytes) -> None:
        combined = self._tail + chunk
        if any(value in combined for value in self._values):
            fail("gate bundle contains forbidden cleartext")
        self._tail = combined[-self._overlap :] if self._overlap else b""


class BundleSnapshot:
    """An openat/O_NOFOLLOW snapshot of the exact combined gate layout."""

    def __init__(
        self,
        root: Path,
        *,
        uid: int,
        gid: int,
        forbidden_values: tuple[bytes, ...] = (),
    ) -> None:
        self._root_path = Path(os.path.abspath(root))
        self._uid = uid
        self._gid = gid
        self._forbidden = forbidden_values
        self._parent_fd = -1
        self._root_fd = -1
        self._browser_fd = -1
        self._root_identity: _Identity | None = None
        self._browser_identity: _Identity | None = None
        self._files: dict[str, _OpenedFile] = {}
        self._sealed = False
        self._closed = False
        self._open()

    def _open(self) -> None:
        if (
            type(self._uid) is not int
            or self._uid < 0
            or type(self._gid) is not int
            or self._gid < 0
        ):
            fail("gate bundle owner binding is invalid")
        _SecretScanner(self._forbidden)
        try:
            self._parent_fd = os.open(self._root_path.parent, _directory_flags())
            self._root_fd = os.open(
                self._root_path.name,
                _directory_flags(),
                dir_fd=self._parent_fd,
            )
            root_identity = _Identity.from_stat(os.fstat(self._root_fd))
            if _entry_identity(self._parent_fd, self._root_path.name) != root_identity:
                fail("gate bundle root identity changed while it was opened")
            _require_directory(
                root_identity,
                mode=0o700,
                uid=self._uid,
                gid=self._gid,
                links=3,
            )
            if frozenset(os.listdir(self._root_fd)) != ROOT_FILES:
                fail("gate bundle root layout differs")
            self._browser_fd = os.open(
                "browser", _directory_flags(), dir_fd=self._root_fd
            )
            browser_identity = _Identity.from_stat(os.fstat(self._browser_fd))
            if _entry_identity(self._root_fd, "browser") != browser_identity:
                fail("gate bundle browser directory identity changed while opened")
            _require_directory(
                browser_identity,
                mode=0o700,
                uid=self._uid,
                gid=self._gid,
                links=2,
            )
            if frozenset(os.listdir(self._browser_fd)) != BROWSER_FILES:
                fail("gate bundle browser layout differs")
            self._root_identity = root_identity
            self._browser_identity = browser_identity
            self._open_file(
                "observer",
                "observer.raw.jsonl",
                self._root_fd,
                MAX_OBSERVER_BYTES,
                0o600,
            )
            self._open_file(
                "journal",
                "service-journal.raw.jsonl",
                self._root_fd,
                MAX_JOURNAL_BYTES,
                0o600,
            )
            self._open_file(
                "summary", "summary.json", self._root_fd, MAX_SUMMARY_BYTES, 0o600
            )
            self._open_file(
                "browser_stdout",
                "browser-stdout.jsonl",
                self._browser_fd,
                MAX_BROWSER_STDOUT_BYTES,
                0o600,
            )
            self._open_file(
                "browser_summary",
                "openwebui-soak-summary.json",
                self._browser_fd,
                MAX_SUMMARY_BYTES,
                0o400,
            )
        except GateIngestError:
            self.close()
            raise
        except OSError:
            self.close()
            fail("failed to open the gate bundle without following links")

    def _open_file(
        self, key: str, name: str, parent_fd: int, maximum: int, mode: int
    ) -> None:
        entry = _entry_identity(parent_fd, name)
        _require_file(entry, maximum=maximum, mode=mode, uid=self._uid, gid=self._gid)
        fd = os.open(name, _file_flags(), dir_fd=parent_fd)
        opened = _Identity.from_stat(os.fstat(fd))
        if opened != entry:
            os.close(fd)
            fail("gate bundle file identity changed while it was opened")
        self._files[key] = _OpenedFile(
            key=key,
            name=name,
            parent_fd=parent_fd,
            fd=fd,
            identity=opened,
            maximum=maximum,
        )

    def __enter__(self) -> BundleSnapshot:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _chunks(self, key: str) -> Iterator[bytes]:
        if self._closed or self._sealed:
            fail("gate bundle snapshot is no longer readable")
        item = self._files.get(key)
        if item is None or item.consumed:
            fail("gate bundle file was requested outside its fixed schedule")
        try:
            before = _Identity.from_stat(os.fstat(item.fd))
            if before != item.identity:
                fail("gate bundle file changed before it was streamed")
            os.lseek(item.fd, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            scanner = _SecretScanner(self._forbidden)
            total = 0
            while True:
                chunk = os.read(item.fd, COPY_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > item.maximum:
                    fail("gate bundle file exceeded its streaming bound")
                digest.update(chunk)
                scanner.consume(chunk)
                yield chunk
            after = _Identity.from_stat(os.fstat(item.fd))
            if after != item.identity or total != item.identity.size:
                fail("gate bundle file changed while it was streamed")
            item.streamed_bytes = total
            item.sha256 = digest.hexdigest()
            item.consumed = True
        except GateIngestError:
            raise
        except OSError:
            fail("failed to stream a gate bundle file")

    def read_small(self, key: str, maximum: int) -> bytes:
        chunks: list[bytes] = []
        total = 0
        for chunk in self._chunks(key):
            total += len(chunk)
            if total > maximum:
                fail("gate bundle document exceeds its bound")
            chunks.append(chunk)
        return b"".join(chunks)

    def iter_lines(self, key: str) -> Iterator[bytes]:
        pending = b""
        for chunk in self._chunks(key):
            pending += chunk
            while True:
                index = pending.find(b"\n")
                if index < 0:
                    if len(pending) > MAX_JSON_LINE_BYTES:
                        fail("gate bundle JSONL line exceeds its bound")
                    break
                raw = pending[:index]
                pending = pending[index + 1 :]
                if not raw or len(raw) > MAX_JSON_LINE_BYTES or raw.endswith(b"\r"):
                    fail("gate bundle JSONL framing differs")
                yield raw
        if pending:
            fail("gate bundle JSONL file lacks its final LF")

    def evidence(self, key: str) -> tuple[int, str]:
        item = self._files.get(key)
        if item is None or not item.consumed or item.sha256 is None:
            fail("gate bundle file has not been fully streamed")
        return item.streamed_bytes, item.sha256

    def seal(self) -> None:
        if self._closed or self._sealed:
            fail("gate bundle snapshot cannot be sealed in its current state")
        if any(not item.consumed for item in self._files.values()):
            fail("not every gate bundle file was consumed before sealing")
        assert self._root_identity is not None
        assert self._browser_identity is not None
        try:
            if (
                frozenset(os.listdir(self._root_fd)) != ROOT_FILES
                or frozenset(os.listdir(self._browser_fd)) != BROWSER_FILES
            ):
                fail("gate bundle layout changed before sealing")
            if (
                _Identity.from_stat(os.fstat(self._root_fd)) != self._root_identity
                or _entry_identity(self._parent_fd, self._root_path.name)
                != self._root_identity
                or _Identity.from_stat(os.fstat(self._browser_fd))
                != self._browser_identity
                or _entry_identity(self._root_fd, "browser") != self._browser_identity
            ):
                fail("gate bundle directory identity changed before sealing")
            for item in self._files.values():
                if (
                    _Identity.from_stat(os.fstat(item.fd)) != item.identity
                    or _entry_identity(item.parent_fd, item.name) != item.identity
                ):
                    fail("gate bundle file identity changed before sealing")
                os.lseek(item.fd, 0, os.SEEK_SET)
                digest = hashlib.sha256()
                scanner = _SecretScanner(self._forbidden)
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
                    fail("gate bundle file hash or identity changed at seal")
            if (
                _Identity.from_stat(os.fstat(self._root_fd)) != self._root_identity
                or _Identity.from_stat(os.fstat(self._browser_fd))
                != self._browser_identity
            ):
                fail("gate bundle directory changed during final hashing")
            self._sealed = True
        except GateIngestError:
            raise
        except OSError:
            fail("failed to seal the gate bundle")

    def close(self) -> None:
        if self._closed:
            return
        pending_error: GateIngestError | None = None
        for item in self._files.values():
            try:
                _safe_close(item.fd)
            except GateIngestError as error:
                pending_error = error
        self._files.clear()
        for fd in (self._browser_fd, self._root_fd, self._parent_fd):
            try:
                _safe_close(fd)
            except GateIngestError as error:
                pending_error = error
        self._browser_fd = self._root_fd = self._parent_fd = -1
        self._closed = True
        if pending_error is not None:
            raise pending_error


class _StableSource:
    def __init__(self, path: Path, label: str, maximum: int, expected_sha256: str):
        self.path = Path(os.path.abspath(path))
        self.label = label
        self.maximum = maximum
        self.parent_fd = -1
        self.fd = -1
        self.identity: _Identity | None = None
        self.raw = b""
        self.sha256 = ""
        if SHA256_RE.fullmatch(expected_sha256) is None:
            fail("source hash binding syntax differs")
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
                fail("bound source identity changed while opened")
            self.identity = opened
            self.raw, self.sha256 = self._snapshot()
            if self.sha256 != expected_sha256:
                fail("bound source hash differs")
        except GateIngestError:
            self.close()
            raise
        except OSError:
            self.close()
            fail("failed to open a bound source without following links")

    def _snapshot(self) -> tuple[bytes, str]:
        assert self.identity is not None
        try:
            if _Identity.from_stat(os.fstat(self.fd)) != self.identity:
                fail("bound source identity changed before reading")
            os.lseek(self.fd, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            total = 0
            while True:
                chunk = os.read(self.fd, COPY_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > self.maximum:
                    fail("bound source exceeds its streaming limit")
                digest.update(chunk)
                chunks.append(chunk)
            if (
                total != self.identity.size
                or _Identity.from_stat(os.fstat(self.fd)) != self.identity
            ):
                fail("bound source changed while reading")
            return b"".join(chunks), digest.hexdigest()
        except GateIngestError:
            raise
        except OSError:
            fail("failed to stream a bound source")

    def seal(self) -> None:
        assert self.identity is not None
        if (
            _Identity.from_stat(os.fstat(self.fd)) != self.identity
            or _entry_identity(self.parent_fd, self.path.name) != self.identity
        ):
            fail("bound source identity changed before sealing")
        raw, digest = self._snapshot()
        if raw != self.raw or digest != self.sha256:
            fail("bound source bytes changed before sealing")
        if (
            _Identity.from_stat(os.fstat(self.fd)) != self.identity
            or _entry_identity(self.parent_fd, self.path.name) != self.identity
        ):
            fail("bound source identity changed during sealing")

    def close(self) -> None:
        pending: GateIngestError | None = None
        for fd in (self.fd, self.parent_fd):
            try:
                _safe_close(fd)
            except GateIngestError as error:
                pending = error
        self.fd = self.parent_fd = -1
        if pending is not None:
            raise pending


def _validate_bindings(bindings: GateInputBindings) -> None:
    if not isinstance(bindings, GateInputBindings):
        fail("gate input bindings have the wrong type")
    if any(
        not isinstance(path, os.PathLike)
        for path in (
            bindings.gate_source,
            bindings.support_source,
            bindings.browser_script,
        )
    ):
        fail("gate input source paths have the wrong type")
    for value in (
        bindings.gate_source_sha256,
        bindings.support_source_sha256,
        bindings.browser_script_sha256,
    ):
        if type(value) is not str or SHA256_RE.fullmatch(value) is None:
            fail("gate input source hash syntax differs")
    if (
        type(bindings.boot_id) is not str
        or BOOT_ID_RE.fullmatch(bindings.boot_id) is None
    ):
        fail("gate input boot ID syntax differs")
    for numeric_value, label, minimum in (
        (bindings.gateway_pid, "gateway PID", 1),
        (bindings.uid, "service uid", 0),
        (bindings.gid, "service gid", 0),
        (bindings.restart_count, "service restart count", 0),
    ):
        if type(numeric_value) is not int or numeric_value < minimum:
            fail(f"gate input {label} differs")
    if (
        type(bindings.service_unit) is not str
        or SERVICE_RE.fullmatch(bindings.service_unit) is None
    ):
        fail("gate input service unit syntax differs")
    if bindings.service_unit != CAMPAIGN_SERVICE_UNIT:
        fail("gate input service unit differs from the campaign journal contract")
    if any(
        type(value) is not str
        for value in (
            bindings.browser_image_reference,
            bindings.browser_image_content_id,
            bindings.openwebui_base_url,
        )
    ):
        fail("gate browser input binding types differ")
    if type(bindings.forbidden_values) is not tuple:
        fail("forbidden evidence values must be an immutable tuple")
    _SecretScanner(bindings.forbidden_values)


def _load_gate(gate: _StableSource, support: _StableSource) -> tuple[Any, str]:
    expected_support = gate.path.with_name("run-openwebui-stop-gate.py")
    if expected_support != support.path:
        fail("gate and support source path binding differs")
    module_name = f"_ullm_combined_gate_ingest_{os.getpid()}_{id(gate):x}"
    module = types.ModuleType(module_name)
    module.__file__ = os.fspath(gate.path)
    module.__package__ = ""
    prior_support = sys.modules.get("_ullm_openwebui_stop_gate_support")
    sys.modules[module_name] = module
    try:
        code = compile(gate.raw, os.fspath(gate.path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
        if (
            module.GATE_SOURCE_RAW != gate.raw
            or module.SUPPORT_SOURCE_RAW != support.raw
        ):
            fail("executed gate source snapshot differs from its input binding")
        return module, module_name
    except GateIngestError:
        sys.modules.pop(module_name, None)
        raise
    except Exception as error:
        sys.modules.pop(module_name, None)
        raise GateIngestError("failed to load the bound gate validator") from error
    finally:
        if prior_support is None:
            sys.modules.pop("_ullm_openwebui_stop_gate_support", None)
        else:
            sys.modules["_ullm_openwebui_stop_gate_support"] = prior_support


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
            raise GateIngestError(
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
        return cast(dict[str, Any], gate.strict_json_object(raw, label))
    except Exception as error:
        raise GateIngestError(f"{label} is invalid") from error


def _exact(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        fail(f"{label} fields differ")
    return cast(dict[str, Any], value)


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
        fail(f"{label} is not a decimal timestamp")
    return int(value, 10)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} SHA-256 syntax differs")
    return value


def _validate_unmaterialized_stream(byte_count: Any, sha256: Any, label: str) -> None:
    count = _integer(byte_count, f"{label} bytes")
    digest = _require_sha(sha256, label)
    if count != 0 or digest != _sha256(b""):
        fail(f"{label} must be empty because its bytes are not materialized")


def _validate_browser_summary(
    gate: Any,
    value: dict[str, Any],
    raw: bytes,
    summary_file_raw: bytes,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "record_type",
        "browser_case",
        "observed_monotonic_ns",
        "chat_count",
        "action_count",
        "socket_event_count",
        "browser_process_count",
        "browser_context_count",
        "browser_context_closed_count",
        "page_count_created",
        "page_count_closed",
        "maximum_open_pages",
        "page_error_count",
        "cancellation_event_count",
        "provider_error_count",
        "case_record_sha256",
        "mode",
        "schedule",
    }
    _exact(value, expected_fields, "combined browser summary")
    expected_count = 21
    expected_case_hashes = [item["record_sha256"] for item in cases]
    expected_socket_events = sum(item["socket_event_count"] for item in cases)
    counts = {
        "chat_count": expected_count,
        "action_count": 105,
        "socket_event_count": expected_socket_events,
        "browser_process_count": 1,
        "browser_context_count": 1,
        "browser_context_closed_count": 1,
        "page_count_created": expected_count,
        "page_count_closed": expected_count,
        "maximum_open_pages": 1,
        "page_error_count": 0,
        "cancellation_event_count": 0,
        "provider_error_count": 0,
    }
    if len(cases) != expected_count:
        fail("combined browser case count differs")
    for field, expected in counts.items():
        if _integer(value[field], f"combined browser summary {field}") != expected:
            fail("combined browser summary counts differ")
    if (
        value["schema_version"] != gate.COMBINED_BROWSER_SCHEMA
        or value["record_type"] != gate.COMBINED_SUMMARY_RECORD_TYPE
        or value["browser_case"] != gate.COMBINED_RUN_CASE
        or value["mode"] != gate.COMBINED_MODE
        or value["schedule"] != gate.schedule_evidence(include_smoke=True)
        or value["case_record_sha256"] != expected_case_hashes
    ):
        fail("combined browser summary identity, mode, schedule, or hashes differ")
    if (
        _decimal(value["observed_monotonic_ns"], "combined browser summary timestamp")
        < cases[-1]["last_action_ns"]
    ):
        fail("combined browser summary precedes the last action")
    if summary_file_raw != raw + b"\n":
        fail("combined browser stdout and summary file bytes differ")
    return {
        "chat_count": expected_count,
        "action_count": 105,
        "socket_event_count": expected_socket_events,
        "browser_summary_bytes": len(summary_file_raw),
        "browser_summary_sha256": _sha256(summary_file_raw),
        "mode": gate.COMBINED_MODE,
        "schedule": gate.schedule_evidence(include_smoke=True),
    }


def _validate_browser(
    gate: Any,
    snapshot: BundleSnapshot,
    base_url: str,
    guard: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    lines: list[tuple[bytes, dict[str, Any]]] = []
    for raw in snapshot.iter_lines("browser_stdout"):
        if len(lines) >= 22:
            fail("combined browser stdout exceeds its exact record count")
        lines.append((raw, _strict_object(gate, raw, "combined browser stdout")))
    if len(lines) != 22:
        fail("combined browser stdout record count differs")
    summary_file_raw = snapshot.read_small("browser_summary", MAX_SUMMARY_BYTES)
    browser_cases: list[dict[str, Any]] = []
    action_records: list[dict[str, Any]] = []
    seen_chat: set[str] = set()
    seen_message: set[str] = set()
    prior_completed = -1
    indices = tuple(gate.case_indices(include_smoke=True))
    if indices != tuple(range(21)):
        fail("bound gate combined schedule differs")
    for case_index, (raw, value) in zip(indices, lines[:-1], strict=True):
        try:
            evidence = cast(
                dict[str, Any],
                gate.validate_browser_case(
                    value,
                    raw,
                    guard,
                    case_index=case_index,
                    base_url=base_url,
                    include_smoke=True,
                ),
            )
        except Exception as error:
            raise GateIngestError("combined browser case validation failed") from error
        chat_hash = cast(str, evidence["chat_id_sha256"])
        message_hash = cast(str, evidence["message_id_sha256"])
        if chat_hash in seen_chat or message_hash in seen_message:
            fail("combined browser chat or message identity is duplicated")
        if cast(int, evidence["first_action_ns"]) < prior_completed:
            fail("combined browser cases overlap or regress")
        seen_chat.add(chat_hash)
        seen_message.add(message_hash)
        prior_completed = cast(int, evidence["last_action_ns"])
        browser_cases.append(evidence)
        browser_case = cast(str, evidence["browser_case"])
        actions = value.get("browser_actions")
        if type(actions) is not list or len(actions) != 5:
            fail("combined browser action count differs")
        for action_value in actions:
            fields = copy.deepcopy(cast(dict[str, Any], action_value))
            fields["started_monotonic_ns"] = _decimal(
                fields["started_monotonic_ns"], "browser action start"
            )
            fields["completed_monotonic_ns"] = _decimal(
                fields["completed_monotonic_ns"], "browser action completion"
            )
            action_records.append(
                {
                    "record_type": "browser_action",
                    "phase": CAMPAIGN_PHASE,
                    "case_id": browser_case,
                    "fields": fields,
                }
            )
    browser_summary = _validate_browser_summary(
        gate,
        lines[-1][1],
        lines[-1][0],
        summary_file_raw,
        browser_cases,
    )
    if len(action_records) != 105:
        fail("combined browser action conversion count differs")
    prior_completed = -1
    expected_cases = tuple(
        gate.browser_case(index, include_smoke=True) for index in indices
    )
    for position, record in enumerate(action_records):
        fields = cast(dict[str, Any], record["fields"])
        if (
            record["case_id"] != expected_cases[position // 5]
            or fields["action_index"] != position % 5
            or type(fields["started_monotonic_ns"]) is not int
            or type(fields["completed_monotonic_ns"]) is not int
            or fields["started_monotonic_ns"] < prior_completed
            or fields["completed_monotonic_ns"] < fields["started_monotonic_ns"]
        ):
            fail("converted browser action order or monotonic timestamps differ")
        prior_completed = fields["completed_monotonic_ns"]
    return browser_cases, action_records, browser_summary


def _validate_journal_and_observer(
    gate: Any,
    snapshot: BundleSnapshot,
    bindings: GateInputBindings,
    browser_cases: list[dict[str, Any]],
    claim_factory: Callable[[bytes, str, str], BundleLifecycleClaimProtocol],
) -> tuple[
    Any,
    list[dict[str, Any]],
    list[BundleLifecycleClaimProtocol],
    int,
    int,
]:
    machine = gate.SoakLifecycleMachine(expected_count=21)
    cursors: set[str] = set()
    lifecycle_payloads: set[bytes] = set()
    captured_payloads: list[bytes] = []
    claims: list[BundleLifecycleClaimProtocol] = []
    journal_records = 0
    prior_monotonic = -1
    active_case: str | None = None
    indices = tuple(gate.case_indices(include_smoke=True))
    for raw in snapshot.iter_lines("journal"):
        journal_records += 1
        if journal_records > 4096:
            fail("combined service journal record count exceeds its bound")
        value = _strict_object(gate, raw, "combined service journal record")
        if (
            type(value.get("__CURSOR")) is not str
            or len(value["__CURSOR"]) > 65_536
            or type(value.get("_PID")) is not str
            or not value["_PID"].isascii()
            or not value["_PID"].isdecimal()
        ):
            fail("combined service journal cursor or PID representation differs")
        try:
            cursor, lifecycle_raw = gate.validate_journal_record(
                raw,
                service=bindings.service_unit,
                main_pid=bindings.gateway_pid,
                boot_id=bindings.boot_id,
                cursors=cursors,
                lifecycle_payloads=lifecycle_payloads,
            )
        except Exception as error:
            raise GateIngestError(
                "combined service journal validation failed"
            ) from error
        if value["PRIORITY"] != str(int(value["PRIORITY"], 10)):
            fail("combined service journal priority representation is not canonical")
        cursors.add(cast(str, cursor))
        monotonic = _decimal(
            value.get("__MONOTONIC_TIMESTAMP"), "service journal monotonic timestamp"
        )
        if value["__MONOTONIC_TIMESTAMP"] != str(monotonic):
            fail("combined service journal monotonic representation is not canonical")
        if monotonic < prior_monotonic:
            fail("combined service journal monotonic timestamps regressed")
        prior_monotonic = monotonic
        if lifecycle_raw is None:
            continue
        lifecycle_raw = cast(bytes, lifecycle_raw)
        if value["_PID"] != str(bindings.gateway_pid):
            fail("combined lifecycle PID representation differs from its binding")
        if (
            len(lifecycle_raw) > MAX_LIFECYCLE_PAYLOAD_BYTES
            or len(captured_payloads) >= MAX_LIFECYCLE_RECORDS
        ):
            fail("combined lifecycle evidence exceeds its retained-memory bound")
        lifecycle_payloads.add(lifecycle_raw)
        try:
            event = cast(dict[str, Any], gate.validate_lifecycle_payload(lifecycle_raw))
            if monotonic * 1000 < cast(int, event["observed_monotonic_ns"]):
                fail("service journal timestamp precedes lifecycle observation")
            machine.consume(event)
        except GateIngestError:
            raise
        except Exception as error:
            raise GateIngestError(
                "combined lifecycle trace validation failed"
            ) from error
        if event["event"] == "request_admitted":
            trace_position = len(machine.traces) - 1
            if not 0 <= trace_position < len(indices):
                fail("combined lifecycle trace count exceeds its schedule")
            active_case = cast(str, browser_cases[trace_position]["browser_case"])
        if active_case is None:
            fail("combined lifecycle trace does not begin with admission")
        captured_payloads.append(lifecycle_raw)
        try:
            claim_raw = json.dumps(
                {field: value[field] for field in REQUIRED_JOURNAL_FIELDS},
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (KeyError, TypeError, ValueError, UnicodeError, RecursionError) as error:
            raise GateIngestError(
                "combined lifecycle claim required fields cannot be reconstructed"
            ) from error
        claims.append(claim_factory(claim_raw, CAMPAIGN_PHASE, active_case))
        if event["event"] == "request_released":
            active_case = None
    if journal_records < len(captured_payloads) or len(cursors) != journal_records:
        fail("combined service journal record or cursor count differs")
    observed_payloads: list[bytes] = []
    for raw in snapshot.iter_lines("observer"):
        if len(observed_payloads) >= 4096:
            fail("combined observer record count exceeds its bound")
        try:
            gate.validate_lifecycle_payload(raw)
        except Exception as error:
            raise GateIngestError(
                "combined observer lifecycle validation failed"
            ) from error
        observed_payloads.append(raw)
    if observed_payloads != captured_payloads:
        fail("combined observer and journal lifecycle bytes differ")
    try:
        correlations = cast(
            list[dict[str, Any]],
            gate.validate_gateway_traces(machine, browser_cases, include_smoke=True),
        )
    except Exception as error:
        raise GateIngestError("combined gateway/browser correlation failed") from error
    return machine, correlations, claims, journal_records, len(captured_payloads)


def _expected_artifact(
    value: Any,
    *,
    expected_fields: set[str],
    file: str,
    actual_bytes: int,
    actual_sha256: str,
    actual_records: int | None = None,
    label: str,
) -> dict[str, Any]:
    result = _exact(value, expected_fields, label)
    if (
        result["file"] != file
        or _integer(result["bytes"], f"{label} bytes") != actual_bytes
        or _require_sha(result["sha256"], label) != actual_sha256
    ):
        fail(f"{label} file, size, or hash differs")
    if (
        actual_records is not None
        and _integer(result["records"], f"{label} records") != actual_records
    ):
        fail(f"{label} record count differs")
    return result


def _validate_gate_summary(
    gate: Any,
    summary: dict[str, Any],
    snapshot: BundleSnapshot,
    bindings: GateInputBindings,
    browser_evidence: dict[str, Any],
    machine: Any,
    correlations: list[dict[str, Any]],
    journal_records: int,
    lifecycle_records: int,
) -> None:
    _exact(
        summary,
        {
            "schema_version",
            "passed",
            "service",
            "browser",
            "gateway",
            "artifacts",
            "mode",
            "schedule",
        },
        "combined gate summary",
    )
    if (
        summary["schema_version"] != gate.COMBINED_GATE_SCHEMA
        or summary["passed"] is not True
        or summary["mode"] != gate.COMBINED_MODE
        or summary["schedule"] != gate.schedule_evidence(include_smoke=True)
    ):
        fail("combined gate summary schema, pass state, mode, or schedule differs")
    service = _exact(
        summary["service"],
        {
            "unit_sha256",
            "main_pid_sha256",
            "user_uid_sha256",
            "user_gid_sha256",
            "boot_id_sha256",
            "restart_count",
            "identity_invariant",
        },
        "combined gate service summary",
    )
    expected_service = {
        "unit_sha256": _sha256(bindings.service_unit.encode("utf-8")),
        "main_pid_sha256": _sha256(str(bindings.gateway_pid).encode("ascii")),
        "user_uid_sha256": _sha256(str(bindings.uid).encode("ascii")),
        "user_gid_sha256": _sha256(str(bindings.gid).encode("ascii")),
        "boot_id_sha256": _sha256(bindings.boot_id.encode("ascii")),
        "restart_count": bindings.restart_count,
        "identity_invariant": True,
    }
    if type(service["restart_count"]) is not int or service != expected_service:
        fail("combined gate service input binding differs")
    browser = _exact(
        summary["browser"],
        {
            "image_reference_sha256",
            "image_content_digest",
            "script_sha256",
            "gate_source_sha256",
            "support_source_sha256",
            "chat_count",
            "action_count",
            "socket_event_count",
            "browser_summary_bytes",
            "browser_summary_sha256",
            "mode",
            "schedule",
            "stdout_lines",
            "stdout_bytes",
            "stdout_sha256",
            "stderr_bytes",
            "stderr_sha256",
        },
        "combined gate browser summary",
    )
    expected_browser_binding = {
        "image_reference_sha256": _sha256(
            bindings.browser_image_reference.encode("utf-8")
        ),
        "image_content_digest": bindings.browser_image_content_id,
        "script_sha256": bindings.browser_script_sha256,
        "gate_source_sha256": bindings.gate_source_sha256,
        "support_source_sha256": bindings.support_source_sha256,
    }
    for field, expected in expected_browser_binding.items():
        if browser[field] != expected:
            fail("combined gate browser source or image binding differs")
    for field, expected in browser_evidence.items():
        if browser[field] != expected:
            fail("combined gate browser evidence summary differs")
    stdout_bytes, stdout_sha = snapshot.evidence("browser_stdout")
    _validate_unmaterialized_stream(
        browser["stderr_bytes"], browser["stderr_sha256"], "browser stderr"
    )
    if (
        _integer(browser["stdout_lines"], "browser stdout lines") != 22
        or _integer(browser["stdout_bytes"], "browser stdout bytes") != stdout_bytes
        or _require_sha(browser["stdout_sha256"], "browser stdout") != stdout_sha
    ):
        fail("combined gate browser process evidence differs")
    gateway = _exact(
        summary["gateway"],
        {
            "request_count",
            "maximum_active_requests",
            "stop_release_count",
            "reset_complete_count",
            "every_admission_after_previous_release",
            "correlations",
        },
        "combined gate gateway summary",
    )
    if (
        _integer(gateway["request_count"], "gateway request count") != 21
        or _integer(gateway["maximum_active_requests"], "gateway maximum active") != 1
        or _integer(gateway["stop_release_count"], "gateway stop release count") != 21
        or _integer(gateway["reset_complete_count"], "gateway reset count") != 21
        or gateway["every_admission_after_previous_release"] is not True
        or gateway["correlations"] != correlations
        or len(machine.traces) != 21
        or machine.active is not None
    ):
        fail("combined gate gateway counts or correlations differ")
    artifacts = _exact(
        summary["artifacts"],
        {"observer", "journal", "browser_stdout", "browser_summary"},
        "combined gate artifact summary",
    )
    observer_bytes, observer_sha = snapshot.evidence("observer")
    _expected_artifact(
        artifacts["observer"],
        expected_fields={"file", "bytes", "records", "sha256"},
        file="observer.raw.jsonl",
        actual_bytes=observer_bytes,
        actual_sha256=observer_sha,
        actual_records=lifecycle_records,
        label="combined observer artifact",
    )
    journal_bytes, journal_sha = snapshot.evidence("journal")
    journal = _expected_artifact(
        artifacts["journal"],
        expected_fields={
            "file",
            "bytes",
            "records",
            "sha256",
            "unique_cursors",
            "lifecycle_records",
            "stderr_bytes",
            "stderr_sha256",
        },
        file="service-journal.raw.jsonl",
        actual_bytes=journal_bytes,
        actual_sha256=journal_sha,
        actual_records=journal_records,
        label="combined journal artifact",
    )
    _validate_unmaterialized_stream(
        journal["stderr_bytes"], journal["stderr_sha256"], "journal stderr"
    )
    if (
        _integer(journal["unique_cursors"], "journal unique cursors") != journal_records
        or _integer(journal["lifecycle_records"], "journal lifecycle records")
        != lifecycle_records
    ):
        fail("combined journal count or stderr evidence differs")
    _expected_artifact(
        artifacts["browser_stdout"],
        expected_fields={"file", "bytes", "records", "sha256"},
        file="browser/browser-stdout.jsonl",
        actual_bytes=stdout_bytes,
        actual_sha256=stdout_sha,
        actual_records=22,
        label="combined browser stdout artifact",
    )
    browser_summary_bytes, browser_summary_sha = snapshot.evidence("browser_summary")
    _expected_artifact(
        artifacts["browser_summary"],
        expected_fields={"file", "bytes", "sha256"},
        file="browser/openwebui-soak-summary.json",
        actual_bytes=browser_summary_bytes,
        actual_sha256=browser_summary_sha,
        label="combined browser summary artifact",
    )


def _derived_view(
    gate: Any,
    summary_raw: bytes,
    correlations: list[dict[str, Any]],
    browser_cases: list[dict[str, Any]],
    lifecycle_records: int,
    bindings: GateInputBindings,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for schedule, correlation, browser in zip(
        gate.schedule_evidence(include_smoke=True),
        correlations,
        browser_cases,
        strict=True,
    ):
        cases.append(
            {
                **copy.deepcopy(schedule),
                "browser_case_sha256": correlation["browser_case_sha256"],
                "action_count": 5,
                "socket_event_count": browser["socket_event_count"],
                "chat_id_sha256": correlation["chat_id_sha256"],
                "message_id_sha256": correlation["message_id_sha256"],
                "request_id_sha256": correlation["request_id_sha256"],
                "completion_id_sha256": correlation["completion_id_sha256"],
                "admitted_monotonic_ns": _decimal(
                    correlation["admitted_monotonic_ns"],
                    "derived admitted timestamp",
                ),
                "released_monotonic_ns": _decimal(
                    correlation["released_monotonic_ns"],
                    "derived released timestamp",
                ),
                "outcome": correlation["outcome"],
                "reset_complete": correlation["reset_complete"],
            }
        )
    return {
        "schema_version": INGEST_VIEW_SCHEMA,
        "mode": gate.COMBINED_MODE,
        "schedule": gate.schedule_evidence(include_smoke=True),
        "chat_count": 21,
        "action_count": 105,
        "lifecycle_record_count": lifecycle_records,
        "maximum_active_requests": 1,
        "stop_release_count": 21,
        "reset_complete_count": 21,
        "component_summary_sha256": _sha256(summary_raw),
        "source_bindings": {
            "gate_source_sha256": bindings.gate_source_sha256,
            "support_source_sha256": bindings.support_source_sha256,
            "browser_script_sha256": bindings.browser_script_sha256,
            "browser_image_reference_sha256": _sha256(
                bindings.browser_image_reference.encode("utf-8")
            ),
            "browser_image_content_id": bindings.browser_image_content_id,
            "service_unit_sha256": _sha256(bindings.service_unit.encode("utf-8")),
            "boot_id_sha256": _sha256(bindings.boot_id.encode("ascii")),
            "gateway_pid_sha256": _sha256(str(bindings.gateway_pid).encode("ascii")),
            "uid_sha256": _sha256(str(bindings.uid).encode("ascii")),
            "gid_sha256": _sha256(str(bindings.gid).encode("ascii")),
            "restart_count": bindings.restart_count,
        },
        "cases": cases,
    }


def _reject_derived_cleartext(
    gate: Any,
    value: dict[str, Any],
    bindings: GateInputBindings,
    machine: Any,
) -> None:
    try:
        raw = gate.compact_json(value)
    except Exception as error:
        raise GateIngestError("derived view cannot be encoded") from error
    sensitive: list[bytes] = [
        *bindings.forbidden_values,
        bindings.openwebui_base_url.encode("utf-8"),
        gate.MODEL_ID.encode("utf-8"),
        gate.MODEL_LABEL.encode("utf-8"),
    ]
    if "@" in bindings.browser_image_reference:
        sensitive.append(bindings.browser_image_reference.encode("utf-8"))
    for index in gate.case_indices(include_smoke=True):
        sensitive.append(gate.case_marker(index, include_smoke=True).encode("utf-8"))
        sensitive.append(gate.case_prompt(index, include_smoke=True).encode("utf-8"))
    for trace in machine.traces:
        sensitive.extend(
            (trace.request_id.encode("utf-8"), trace.completion_id.encode("utf-8"))
        )
    scanner = _SecretScanner(tuple(value for value in sensitive if len(value) >= 4))
    scanner.consume(raw)


def ingest_combined_soak_bundle(
    bundle: Path, bindings: GateInputBindings
) -> CombinedSoakIngestResult:
    """Revalidate a formal smoke1+soak20 bundle and convert its evidence."""

    if not isinstance(bundle, os.PathLike):
        fail("gate bundle path has the wrong type")
    _validate_bindings(bindings)
    gate_source: _StableSource | None = None
    support_source: _StableSource | None = None
    browser_source: _StableSource | None = None
    gate_module_name: str | None = None
    try:
        gate_source = _StableSource(
            bindings.gate_source,
            "combined gate source",
            MAX_GATE_SOURCE_BYTES,
            bindings.gate_source_sha256,
        )
        support_source = _StableSource(
            bindings.support_source,
            "combined gate support source",
            MAX_SUPPORT_SOURCE_BYTES,
            bindings.support_source_sha256,
        )
        browser_source = _StableSource(
            bindings.browser_script,
            "combined browser script",
            MAX_BROWSER_SCRIPT_BYTES,
            bindings.browser_script_sha256,
        )
        gate, gate_module_name = _load_gate(gate_source, support_source)
        try:
            base_url = cast(str, gate.normalized_url(bindings.openwebui_base_url))
            image_reference, content_id = gate.normalized_browser_image(
                bindings.browser_image_reference
            )
        except Exception as error:
            raise GateIngestError(
                "gate URL or browser image input binding differs"
            ) from error
        if (
            image_reference != bindings.browser_image_reference
            or content_id != bindings.browser_image_content_id
        ):
            fail("browser image reference and content ID binding differ")
        static_sensitive: list[bytes] = [
            *bindings.forbidden_values,
            base_url.encode("utf-8"),
            gate.MODEL_ID.encode("utf-8"),
            gate.MODEL_LABEL.encode("utf-8"),
        ]
        if "@" in image_reference:
            static_sensitive.append(image_reference.encode("utf-8"))
        for index in gate.case_indices(include_smoke=True):
            static_sensitive.append(
                gate.case_marker(index, include_smoke=True).encode("utf-8")
            )
            static_sensitive.append(
                gate.case_prompt(index, include_smoke=True).encode("utf-8")
            )
        forbidden = tuple(value for value in static_sensitive if len(value) >= 4)
        guard = gate.SecretGuard(list(forbidden))
        claim_factory = _campaign_claim_factory()
        with BundleSnapshot(
            bundle,
            uid=bindings.uid,
            gid=bindings.gid,
            forbidden_values=forbidden,
        ) as snapshot:
            summary_raw = snapshot.read_small("summary", MAX_SUMMARY_BYTES)
            if not summary_raw.endswith(b"\n") or summary_raw.count(b"\n") != 1:
                fail("combined gate summary framing differs")
            summary = _strict_object(gate, summary_raw[:-1], "combined gate summary")
            try:
                if gate.compact_json(summary) + b"\n" != summary_raw:
                    fail("combined gate summary is not canonical JSON")
            except GateIngestError:
                raise
            except Exception as error:
                raise GateIngestError(
                    "combined gate summary encoding failed"
                ) from error
            browser_cases, actions, browser_evidence = _validate_browser(
                gate, snapshot, base_url, guard
            )
            (
                machine,
                correlations,
                claims,
                journal_records,
                lifecycle_records,
            ) = _validate_journal_and_observer(
                gate,
                snapshot,
                bindings,
                browser_cases,
                claim_factory,
            )
            _validate_gate_summary(
                gate,
                summary,
                snapshot,
                bindings,
                browser_evidence,
                machine,
                correlations,
                journal_records,
                lifecycle_records,
            )
            view = _derived_view(
                gate,
                summary_raw,
                correlations,
                browser_cases,
                lifecycle_records,
                bindings,
            )
            _reject_derived_cleartext(gate, view, bindings, machine)
            snapshot.seal()
        gate_source.seal()
        support_source.seal()
        browser_source.seal()
        return CombinedSoakIngestResult(tuple(actions), tuple(claims), view)
    except GateIngestError:
        raise
    except Exception as error:
        raise GateIngestError("combined gate bundle ingestion failed") from error
    finally:
        if gate_module_name is not None:
            sys.modules.pop(gate_module_name, None)
        pending: GateIngestError | None = None
        for source in (browser_source, support_source, gate_source):
            if source is not None:
                try:
                    source.close()
                except GateIngestError as error:
                    pending = error
        if pending is not None and sys.exc_info()[0] is None:
            raise pending


__all__ = [
    "BundleSnapshot",
    "CombinedSoakIngestResult",
    "GateIngestError",
    "GateInputBindings",
    "INGEST_VIEW_SCHEMA",
    "ingest_combined_soak_bundle",
]
