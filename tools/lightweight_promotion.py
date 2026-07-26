#!/usr/bin/env python3
"""Generic, evidence-producing lightweight served-model promotion.

This module has no campaign, authorization, plan-hash, or candidate-specific
confirmation mechanism. It accepts one manifest, captures actual baseline and
candidate generations, preserves exact active bytes, and restores them when a
candidate cannot stay live.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import html
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate-served-model.py"
DEFAULT_PROMPT_SUITE = ROOT / "docs" / "plans" / "lightweight-promotion-prompt-suite-v0.1.json"
DEFAULT_ACTIVE_MANIFEST = Path("/etc/ullm/served-models/active.json")
DEFAULT_SERVICE = "ullm-openai.service"
DEFAULT_BASE_URL = "http://172.20.0.1:8000"
DEFAULT_GATEWAY_CONTAINER = "open-webui"
DEFAULT_TOKEN_FILE = Path("/etc/ullm/openai-api-key")
DEFAULT_STATE_DIR = Path("/var/lib/ullm/lightweight-promotions")

PROMOTION_TRANSACTION_SCHEMA = "ullm.lightweight_promotion.transaction.v1"
PROMOTION_OUTCOME_SCHEMA = "ullm.lightweight_promotion.outcome.v1"
ROLLBACK_OUTCOME_SCHEMA = "ullm.lightweight_promotion.rollback_outcome.v1"
PROMPT_SUITE_SCHEMA = "ullm.lightweight_promotion_prompt_suite.v1"
MAX_MANIFEST_BYTES = 1_048_576
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_TOKEN_BYTES = 16_384
MAX_SUITE_BYTES = 4 * 1024 * 1024
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
CASE_ID_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
SYSTEMCTL = Path("/usr/bin/systemctl")
DOCKER = Path("/usr/bin/docker")
RENAME_EXCHANGE = 2
CONTAINER_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class PromotionError(RuntimeError):
    """A required promotion or rollback invariant did not hold."""


@dataclass(frozen=True, slots=True)
class Snapshot:
    path: Path
    raw: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class SuiteCase:
    case_id: str
    category: str
    messages: tuple[dict[str, str], ...]
    max_completion_tokens: int
    expected_language: str
    expected_kind: str


def fail(message: str) -> NoReturn:
    raise PromotionError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("JSON has a duplicate key")
        result[key] = value
    return result


def _reject_nonfinite(_: str) -> None:
    fail("JSON contains a non-finite number")


def strict_json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_without_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PromotionError(f"{label} is not strict JSON") from error


def strict_object(raw: bytes, label: str) -> dict[str, Any]:
    value = strict_json(raw, label)
    if not isinstance(value, dict):
        fail(f"{label} must have an object root")
    return value


def canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise PromotionError("value is not canonical JSON") from error


def _reject_symlink_components(path: Path, label: str, *, allow_missing_leaf: bool = False) -> None:
    if not path.is_absolute():
        fail(f"{label} must be an absolute path")
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if allow_missing_leaf and index == len(parts) - 1:
                return
            fail(f"{label} has an absent path component")
        if stat.S_ISLNK(metadata.st_mode):
            fail(f"{label} traverses a symlink")


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_mode,
    )


def read_snapshot(path: Path, label: str, *, maximum: int = MAX_MANIFEST_BYTES) -> Snapshot:
    path = path.absolute()
    _reject_symlink_components(path, label)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PromotionError(f"{label} is unavailable") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > maximum:
            fail(f"{label} is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after) or len(raw) != before.st_size:
            fail(f"{label} changed while being read")
        named = path.lstat()
        if _identity(after) != _identity(named):
            fail(f"{label} changed while being read")
        if not raw or len(raw) > maximum:
            fail(f"{label} has an invalid byte length")
        return Snapshot(path=path, raw=raw, sha256=sha256(raw))
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        written = os.write(descriptor, raw[offset:])
        if written <= 0:
            fail("short file write")
        offset += written


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_directory(path: Path, label: str, *, mode: int = 0o750) -> None:
    path = path.absolute()
    _reject_symlink_components(path, label, allow_missing_leaf=True)
    if path.exists() or path.is_symlink():
        fail(f"{label} already exists")
    try:
        os.mkdir(path, mode)
    except OSError as error:
        raise PromotionError(f"{label} cannot be created") from error
    _fsync_directory(path.parent)


def ensure_state_directory(path: Path) -> Path:
    path = path.absolute()
    if path.exists():
        _reject_symlink_components(path, "state directory")
        metadata = path.stat()
        if not stat.S_ISDIR(metadata.st_mode):
            fail("state directory is not a directory")
        return path
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, mode=0o750)
    _reject_symlink_components(parent, "state directory parent")
    create_directory(path, "state directory", mode=0o750)
    return path


def write_new(path: Path, raw: bytes, label: str, *, mode: int = 0o440) -> Snapshot:
    path = path.absolute()
    _reject_symlink_components(path, label, allow_missing_leaf=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as error:
        raise PromotionError(f"{label} cannot be created") from error
    try:
        _write_all(descriptor, raw)
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return read_snapshot(path, label, maximum=max(MAX_RESPONSE_BYTES, len(raw) + 1))


def write_json_new(path: Path, value: Any, label: str, *, mode: int = 0o440) -> Snapshot:
    return write_new(path, canonical_json(value), label, mode=mode)


def _require_root_for_mutation() -> None:
    if os.geteuid() != 0:
        fail("promotion and rollback execution require root")


def _require_active_parent(active: Path) -> None:
    _reject_symlink_components(active, "active manifest")
    metadata = active.parent.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        fail("active manifest parent is unsafe")


@contextmanager
def active_lock(active: Path) -> Iterator[None]:
    _require_active_parent(active)
    lock_path = active.parent / ".active.json.activation.lock"
    flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise PromotionError("activation lock is unavailable") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != 0:
            fail("activation lock is unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _rename_exchange(parent_fd: int, left: str, right: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        fail("renameat2(RENAME_EXCHANGE) is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd,
        left.encode("utf-8"),
        parent_fd,
        right.encode("utf-8"),
        RENAME_EXCHANGE,
    )
    if result != 0:
        code = ctypes.get_errno()
        raise PromotionError(f"atomic manifest exchange failed: {os.strerror(code)}")


def _entry_raw(parent_fd: int, name: str, label: str) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise PromotionError(f"{label} is unavailable") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0 or metadata.st_size > MAX_MANIFEST_BYTES:
            fail(f"{label} is unsafe")
        raw = bytearray()
        while len(raw) <= MAX_MANIFEST_BYTES:
            chunk = os.read(descriptor, min(65_536, MAX_MANIFEST_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        if len(raw) != metadata.st_size or len(raw) > MAX_MANIFEST_BYTES:
            fail(f"{label} changed while being read")
        return bytes(raw)
    finally:
        os.close(descriptor)


def atomic_switch(active: Path, expected_raw: bytes, replacement_raw: bytes) -> bool:
    """Atomically replace active bytes iff the exchanged-away bytes are expected."""

    if expected_raw == replacement_raw:
        return False
    _require_active_parent(active)
    active = active.absolute()
    before = active.lstat()
    if not stat.S_ISREG(before.st_mode):
        fail("active manifest is not a regular file")
    parent_fd = os.open(active.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    stage = f".{active.name}.lightweight-stage-{secrets.token_hex(16)}"
    exchanged = False
    try:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC
        descriptor = os.open(stage, flags, stat.S_IMODE(before.st_mode), dir_fd=parent_fd)
        try:
            _write_all(descriptor, replacement_raw)
            os.fsync(descriptor)
            os.fchmod(descriptor, stat.S_IMODE(before.st_mode))
        finally:
            os.close(descriptor)
        os.fsync(parent_fd)
        _rename_exchange(parent_fd, stage, active.name)
        exchanged = True
        active_raw = _entry_raw(parent_fd, active.name, "switched active manifest")
        old_raw = _entry_raw(parent_fd, stage, "exchanged rollback manifest")
        if active_raw != replacement_raw or old_raw != expected_raw:
            try:
                _rename_exchange(parent_fd, stage, active.name)
            except PromotionError:
                pass
            fail("active manifest changed concurrently with atomic switch")
        os.unlink(stage, dir_fd=parent_fd)
        os.fsync(parent_fd)
        return True
    except BaseException:
        if exchanged:
            try:
                active_raw = _entry_raw(parent_fd, active.name, "active manifest after failure")
                old_raw = _entry_raw(parent_fd, stage, "staging manifest after failure")
                if active_raw == replacement_raw and old_raw == expected_raw:
                    _rename_exchange(parent_fd, stage, active.name)
                    os.fsync(parent_fd)
            except Exception:
                pass
        try:
            os.unlink(stage, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except OSError as error:
            if error.errno != errno.ENOENT:
                pass
        raise
    finally:
        os.close(parent_fd)


def _python_for_validator() -> str:
    production = Path("/usr/bin/python3")
    return os.fspath(production if production.is_file() else Path(sys.executable))


def validate_manifest(path: Path) -> dict[str, Any]:
    if not VALIDATOR.is_file():
        fail("served-model validator is unavailable")
    completed = subprocess.run(
        [_python_for_validator(), os.fspath(VALIDATOR), "--manifest", os.fspath(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=90,
    )
    if completed.returncode != 0:
        fail("served-model validation failed")
    summary = strict_object(completed.stdout.encode("utf-8"), "served-model validation")
    required = {"validated", "manifest_sha256", "model_id", "format_id", "worker"}
    if not required.issubset(summary) or summary.get("validated") is not True:
        fail("served-model validation returned an incomplete summary")
    if not isinstance(summary["model_id"], str) or not isinstance(summary["worker"], dict):
        fail("served-model validation returned invalid identity")
    return summary


def load_suite(path: Path) -> tuple[SuiteCase, ...]:
    snapshot = read_snapshot(path, "prompt suite", maximum=MAX_SUITE_BYTES)
    document = strict_object(snapshot.raw, "prompt suite")
    if set(document) != {"schema_version", "title", "cases"}:
        fail("prompt suite fields differ")
    if document["schema_version"] != PROMPT_SUITE_SCHEMA:
        fail("prompt suite schema is unsupported")
    cases = document["cases"]
    if not isinstance(cases, list) or not 8 <= len(cases) <= 16:
        fail("prompt suite must contain 8 through 16 cases")
    result: list[SuiteCase] = []
    seen: set[str] = set()
    for item in cases:
        if not isinstance(item, dict) or set(item) != {
            "id",
            "category",
            "max_completion_tokens",
            "expect",
            "messages",
        }:
            fail("prompt suite case fields differ")
        case_id = item["id"]
        category = item["category"]
        if not isinstance(case_id, str) or CASE_ID_RE.fullmatch(case_id) is None or case_id in seen:
            fail("prompt suite case ID is invalid")
        if not isinstance(category, str) or not category:
            fail("prompt suite category is invalid")
        maximum = item["max_completion_tokens"]
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 512:
            fail("prompt suite maximum is invalid")
        expect = item["expect"]
        if not isinstance(expect, dict) or set(expect) != {"language", "kind"}:
            fail("prompt suite expectation is invalid")
        language = expect["language"]
        kind = expect["kind"]
        if language not in {"ja", "en", "any"} or kind not in {"prose", "code", "summary"}:
            fail("prompt suite expectation is unsupported")
        messages = item["messages"]
        if not isinstance(messages, list) or not messages:
            fail("prompt suite messages are invalid")
        normalized: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                fail("prompt suite message fields differ")
            role = message["role"]
            content = message["content"]
            if role not in {"system", "user", "assistant"} or not isinstance(content, str) or not content:
                fail("prompt suite message is invalid")
            normalized.append({"role": role, "content": content})
        seen.add(case_id)
        result.append(
            SuiteCase(
                case_id=case_id,
                category=category,
                messages=tuple(normalized),
                max_completion_tokens=maximum,
                expected_language=language,
                expected_kind=kind,
            )
        )
    return tuple(result)


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
        raise urllib.error.HTTPError(req.full_url, code, "redirect rejected", headers, fp)


def _validate_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        fail("base URL must be a credential-free HTTP origin")
    if parsed.hostname not in {"127.0.0.1", "localhost", "172.20.0.1"}:
        fail("base URL host is outside the local gateway allowlist")
    if parsed.port is None:
        fail("base URL must include an explicit port")
    return base_url.rstrip("/")


def normalize_gateway_container(value: str) -> str | None:
    """Return the local probe container, or ``None`` for direct HTTP."""

    if value == "direct":
        return None
    if CONTAINER_NAME_RE.fullmatch(value) is None:
        fail("gateway container must be a Docker container name or 'direct'")
    return value


def read_token(path: Path) -> str:
    snapshot = read_snapshot(path, "gateway token", maximum=MAX_TOKEN_BYTES)
    try:
        token = snapshot.raw.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise PromotionError("gateway token is not UTF-8") from error
    if not token or "\n" in token or "\r" in token:
        fail("gateway token is invalid")
    return token


def _decode_gateway_response(
    status: int, body: bytes
) -> tuple[int, dict[str, Any] | None, str | None]:
    if len(body) > MAX_RESPONSE_BYTES:
        return status, None, "response_too_large"
    try:
        value = strict_json(body, "gateway response")
    except PromotionError:
        return status, None, "invalid_json"
    if not isinstance(value, dict):
        return status, None, "invalid_json_root"
    return status, value, None


def _container_http_json(
    url: str,
    *,
    token: str | None,
    payload: dict[str, Any] | None,
    timeout_seconds: float,
    gateway_container: str,
) -> tuple[int, dict[str, Any] | None, str | None]:
    """Probe the bridge-bound gateway from a configured local container.

    The deployment firewall intentionally prevents a host-originated request to
    the Docker bridge listener.  Curl receives URL, body, and bearer token via
    stdin config, so the token is not exposed in a process argument or saved
    to an on-disk temporary file.
    """

    if not DOCKER.is_file():
        return 0, None, "docker_unavailable"
    config_lines = [
        f"url = {json.dumps(url, ensure_ascii=False)}",
        "silent",
        "show-error",
        f"max-time = {max(1, int(timeout_seconds + 0.999))}",
    ]
    if token is not None:
        config_lines.append(
            f"header = {json.dumps(f'Authorization: Bearer {token}', ensure_ascii=False)}"
        )
    if payload is not None:
        config_lines.extend(
            [
                'request = "POST"',
                'header = "Content-Type: application/json"',
                f"data-binary = {json.dumps(canonical_json(payload).decode('utf-8'), ensure_ascii=False)}",
            ]
        )
    config_lines.append('write-out = "\\n%{http_code}"')
    config = ("\n".join(config_lines) + "\n").encode("utf-8")
    try:
        completed = subprocess.run(
            [
                os.fspath(DOCKER),
                "exec",
                "-i",
                gateway_container,
                "/usr/bin/curl",
                "--config",
                "-",
            ],
            input=config,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=max(0.1, timeout_seconds) + 3.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0, None, "container_transport"
    if completed.returncode != 0:
        return 0, None, "container_transport"
    body, separator, status_raw = completed.stdout.rpartition(b"\n")
    if not separator:
        return 0, None, "container_protocol"
    try:
        status = int(status_raw)
    except ValueError:
        return 0, None, "container_protocol"
    return _decode_gateway_response(status, body)


def _http_json(
    url: str,
    *,
    token: str | None,
    payload: dict[str, Any] | None,
    timeout_seconds: float,
    gateway_container: str | None,
) -> tuple[int, dict[str, Any] | None, str | None]:
    if gateway_container is not None:
        return _container_http_json(
            url,
            token=token,
            payload=payload,
            timeout_seconds=timeout_seconds,
            gateway_container=gateway_container,
        )
    encoded = None if payload is None else canonical_json(payload)
    request = urllib.request.Request(
        url,
        data=encoded,
        method="GET" if encoded is None else "POST",
        headers={
            **({"Authorization": f"Bearer {token}"} if token is not None else {}),
            **({"Content-Type": "application/json"} if encoded is not None else {}),
        },
    )
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=max(0.1, timeout_seconds)) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
            status = response.status
    except urllib.error.HTTPError as error:
        return error.code, None, f"http_{error.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return 0, None, type(error).__name__.lower()
    return _decode_gateway_response(status, body)


def _model_listing_has_model(value: dict[str, Any], model_id: str) -> bool:
    rows = value.get("data")
    if not isinstance(rows, list):
        return False
    return any(isinstance(row, dict) and row.get("id") == model_id for row in rows)


def wait_for_live_gateway(
    *,
    base_url: str,
    token: str,
    model_id: str,
    timeout_seconds: float,
    gateway_container: str | None,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    delay = 0.20
    attempts: list[dict[str, Any]] = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            fail("gateway did not become ready before the bounded deadline")
        probe_timeout = min(5.0, remaining)
        health_status, health, health_error = _http_json(
            f"{base_url}/healthz",
            token=None,
            payload=None,
            timeout_seconds=probe_timeout,
            gateway_container=gateway_container,
        )
        ready_status, ready, ready_error = _http_json(
            f"{base_url}/readyz",
            token=None,
            payload=None,
            timeout_seconds=probe_timeout,
            gateway_container=gateway_container,
        )
        model_status, models, model_error = _http_json(
            f"{base_url}/v1/models",
            token=token,
            payload=None,
            timeout_seconds=probe_timeout,
            gateway_container=gateway_container,
        )
        passed = (
            health_status == 200
            and health == {"status": "ok"}
            and ready_status == 200
            and ready == {"status": "ready"}
            and model_status == 200
            and models is not None
            and _model_listing_has_model(models, model_id)
        )
        attempts.append(
            {
                "at_monotonic_seconds": round(time.monotonic(), 6),
                "health_status": health_status,
                "ready_status": ready_status,
                "models_status": model_status,
                "health_error": health_error,
                "ready_error": ready_error,
                "models_error": model_error,
                "passed": passed,
            }
        )
        if passed:
            return attempts
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            fail("gateway did not become ready before the bounded deadline")
        time.sleep(min(delay, remaining))
        delay = min(delay * 1.8, 3.0)


def service_state(service: str) -> dict[str, str]:
    completed = subprocess.run(
        [
            os.fspath(SYSTEMCTL),
            "show",
            service,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "NRestarts",
            "-p",
            "MainPID",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        fail("cannot inspect gateway service")
    result: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def restart_service(service: str) -> dict[str, Any]:
    before = service_state(service)
    completed = subprocess.run(
        [os.fspath(SYSTEMCTL), "restart", service],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=45,
    )
    if completed.returncode != 0:
        fail("gateway service restart failed")
    return {
        "at": utc_now(),
        "service": service,
        "before": before,
        "restart_command_succeeded": True,
    }


def _extract_completion(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        fail("completion response has no choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        fail("completion response has no message")
    content = message.get("content")
    if not isinstance(content, str):
        fail("completion response content is not text")
    return content


def _repetition_flags(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    if len(normalized) < 24:
        return []
    flags: list[str] = []
    sentences = [part.strip() for part in re.split(r"[。！？.!?]+", normalized) if len(part.strip()) >= 8]
    if sentences and max(Counter(sentences).values()) >= 3:
        flags.append("repeated_sentence_loop")
    grams = [normalized[index : index + 12] for index in range(0, max(0, len(normalized) - 11), 3)]
    if grams and max(Counter(grams).values()) >= 5:
        flags.append("repeated_phrase_loop")
    return flags


def analyze_text(text: str, case: SuiteCase) -> dict[str, list[str]]:
    blocking: list[str] = []
    attention: list[str] = []
    if not text.strip():
        blocking.append("empty_completion")
    if "\ufffd" in text:
        blocking.append("replacement_character")
    if any(ord(character) < 32 and character not in "\n\r\t" for character in text):
        blocking.append("unexpected_control_character")
    blocking.extend(_repetition_flags(text))
    japanese = sum(
        1
        for character in text
        if "\u3040" <= character <= "\u30ff" or "\u3400" <= character <= "\u9fff"
    )
    latin = sum(1 for character in text if ("a" <= character.lower() <= "z"))
    # These intentionally require a strong signal. Natural code, names, and
    # quotations can mix scripts, whereas a complete loss of the requested
    # language is a practical automatic signal of response abandonment.
    if case.expected_language == "ja" and japanese == 0 and latin >= 30:
        blocking.append("expected_japanese_not_observed")
    if case.expected_language == "en" and japanese >= 25 and latin < 5:
        blocking.append("expected_english_not_observed")
    if case.expected_kind == "code":
        code_marker = re.compile(
            rf"{re.escape(chr(96) * 3)}|\b(def|class|function|return|const|let|var|import|for|while|if)\b|[{{}};]"
        )
        if code_marker.search(text) is None:
            blocking.append("code_structure_not_observed")
    return {"blocking": sorted(set(blocking)), "attention": sorted(set(attention))}


def run_suite(
    *,
    suite: tuple[SuiteCase, ...],
    model_id: str,
    manifest_document: dict[str, Any],
    base_url: str,
    token: str,
    request_timeout_seconds: float,
    output_dir: Path,
    gateway_container: str | None,
) -> list[dict[str, Any]]:
    create_directory(output_dir, "suite output directory", mode=0o750)
    records: list[dict[str, Any]] = []
    reasoning_enabled = isinstance(manifest_document.get("reasoning"), dict)
    for case in suite:
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": list(case.messages),
            "max_completion_tokens": case.max_completion_tokens,
            "seed": 0,
        }
        if reasoning_enabled:
            payload["reasoning_effort"] = "none"
        started = time.monotonic()
        status, response, error = _http_json(
            f"{base_url}/v1/chat/completions",
            token=token,
            payload=payload,
            timeout_seconds=request_timeout_seconds,
            gateway_container=gateway_container,
        )
        record: dict[str, Any] = {
            "case_id": case.case_id,
            "category": case.category,
            "request": payload,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "http_status": status,
            "error": error,
        }
        if status == 200 and response is not None and error is None:
            try:
                text = _extract_completion(response)
                record["content"] = text
                record["character_count"] = len(text)
                record["analysis"] = analyze_text(text, case)
                usage = response.get("usage")
                if isinstance(usage, dict):
                    record["usage"] = usage
            except PromotionError as failure:
                record["error"] = str(failure)
                record["analysis"] = {"blocking": ["invalid_completion"], "attention": []}
        else:
            record["analysis"] = {"blocking": ["request_failure"], "attention": []}
        write_json_new(
            output_dir / f"{case.case_id}.json",
            record,
            f"suite output {case.case_id}",
        )
        records.append(record)
    return records


def compare_suites(
    suite: tuple[SuiteCase, ...],
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_by_id = {str(item["case_id"]): item for item in baseline}
    candidate_by_id = {str(item["case_id"]): item for item in candidate}
    cases: list[dict[str, Any]] = []
    exact_matches = 0
    blocking: list[str] = []
    for definition in suite:
        before = baseline_by_id[definition.case_id]
        after = candidate_by_id[definition.case_id]
        before_text = before.get("content") if isinstance(before.get("content"), str) else None
        after_text = after.get("content") if isinstance(after.get("content"), str) else None
        current_blocking = list(after.get("analysis", {}).get("blocking", []))
        current_attention = list(after.get("analysis", {}).get("attention", []))
        if before_text is not None and after_text is not None:
            if before_text == after_text:
                exact_matches += 1
            if len(before_text) >= 40 and len(after_text) * 10 < len(before_text):
                current_blocking.append("extreme_shortening_vs_active")
            if len(after_text) > max(2_000, len(before_text) * 10):
                current_blocking.append("extreme_lengthening_vs_active")
        else:
            current_blocking.append("missing_baseline_or_candidate_text")
        current_blocking = sorted(set(current_blocking))
        current_attention = sorted(set(current_attention))
        if current_blocking:
            blocking.extend(f"{definition.case_id}:{name}" for name in current_blocking)
        cases.append(
            {
                "case_id": definition.case_id,
                "baseline_characters": len(before_text) if before_text is not None else None,
                "candidate_characters": len(after_text) if after_text is not None else None,
                "output_exact_match": before_text == after_text if before_text is not None and after_text is not None else False,
                "blocking": current_blocking,
                "attention": current_attention,
            }
        )
    return {
        "schema_version": "ullm.lightweight_promotion.comparison.v1",
        "case_count": len(suite),
        "output_exact_match_rate": exact_matches / len(suite),
        "top1_match_rate": {
            "status": "not_available_from_openai_gateway",
            "blocking": False,
        },
        "cases": cases,
        "blocking_findings": blocking,
        "passed": not blocking,
    }


def write_comparison_markdown(
    path: Path,
    suite: tuple[SuiteCase, ...],
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> None:
    before_by_id = {str(item["case_id"]): item for item in baseline}
    after_by_id = {str(item["case_id"]): item for item in candidate}
    lines = [
        "# Lightweight promotion output comparison",
        "",
        "This is an evidence record, not a human approval gate.",
        "",
        f"- Automated blocking findings: {html.escape(json.dumps(comparison['blocking_findings'], ensure_ascii=False))}",
        f"- Exact output-match rate (diagnostic only): {comparison['output_exact_match_rate']:.3f}",
        "",
    ]
    for case in suite:
        before = before_by_id[case.case_id]
        after = after_by_id[case.case_id]
        rows = next(row for row in comparison["cases"] if row["case_id"] == case.case_id)
        lines.extend([f"## {case.case_id}", "", "### Prompt", ""])
        for message in case.messages:
            lines.append(f"- {message['role']}: {html.escape(message['content'])}")
        lines.extend(["", "### Active output", "", "<pre>"])
        lines.append(html.escape(str(before.get("content", before.get("error", "")))))
        lines.extend(["</pre>", "", "### Candidate output", "", "<pre>"])
        lines.append(html.escape(str(after.get("content", after.get("error", "")))))
        lines.extend(["</pre>", "", "### Automated observations", "", "<pre>"])
        lines.append(html.escape(json.dumps(rows, ensure_ascii=False, indent=2)))
        lines.extend(["</pre>", ""])
    write_new(path, ("\n".join(lines) + "\n").encode("utf-8"), "comparison markdown", mode=0o440)


def append_ledger(state_dir: Path, record: dict[str, Any]) -> None:
    path = state_dir / "ledger.jsonl"
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_CLOEXEC, 0o640)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _write_all(descriptor, canonical_json(record))
        os.fsync(descriptor)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    _fsync_directory(state_dir)


def _new_run_directory(state_dir: Path) -> Path:
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{secrets.token_hex(8)}"
    run_dir = state_dir / run_id
    create_directory(run_dir, "promotion state directory", mode=0o700)
    return run_dir


def _error_text(error: BaseException) -> str:
    text = str(error).replace("\n", " ").strip()
    return text[:512] or type(error).__name__


def _response_probe(
    *,
    case: SuiteCase,
    model_id: str,
    manifest_document: dict[str, Any],
    base_url: str,
    token: str,
    request_timeout_seconds: float,
    gateway_container: str | None,
) -> dict[str, Any]:
    reasoning_enabled = isinstance(manifest_document.get("reasoning"), dict)
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": list(case.messages),
        "max_completion_tokens": min(case.max_completion_tokens, 64),
        "seed": 0,
    }
    if reasoning_enabled:
        payload["reasoning_effort"] = "none"
    status, response, error = _http_json(
        f"{base_url}/v1/chat/completions",
        token=token,
        payload=payload,
        timeout_seconds=request_timeout_seconds,
        gateway_container=gateway_container,
    )
    if status != 200 or response is None or error is not None:
        fail("post-rollback response probe failed")
    content = _extract_completion(response)
    analysis = analyze_text(content, case)
    if analysis["blocking"]:
        fail("post-rollback response probe has a blocking text finding")
    return {
        "case_id": case.case_id,
        "http_status": status,
        "content": content,
        "analysis": analysis,
    }


def _prepare_evidence_directory(path: Path) -> Path:
    path = path.absolute()
    create_directory(path, "evidence directory", mode=0o750)
    return path


def preflight_report(
    *,
    active: Snapshot,
    candidate: Snapshot,
    active_validation: dict[str, Any],
    candidate_validation: dict[str, Any],
    suite: tuple[SuiteCase, ...],
) -> dict[str, Any]:
    return {
        "schema_version": "ullm.lightweight_promotion.preflight.v1",
        "ready": True,
        "active_manifest": {
            "path": os.fspath(active.path),
            "sha256": active.sha256,
            "model_id": active_validation["model_id"],
        },
        "candidate_manifest": {
            "path": os.fspath(candidate.path),
            "sha256": candidate.sha256,
            "model_id": candidate_validation["model_id"],
            "worker_sha256": candidate_validation["worker"].get("binary_sha256"),
        },
        "prompt_suite_case_count": len(suite),
        "mutation_requires_yes": True,
    }


def _validate_promotable_inputs(
    *,
    active: Snapshot,
    candidate: Snapshot,
    suite: tuple[SuiteCase, ...],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    active_document = strict_object(active.raw, "active manifest")
    candidate_document = strict_object(candidate.raw, "candidate manifest")
    active_validation = validate_manifest(active.path)
    candidate_validation = validate_manifest(candidate.path)
    if candidate_validation["manifest_sha256"] != candidate.sha256:
        fail("candidate validator hash differs from stable candidate bytes")
    if active_validation["manifest_sha256"] != active.sha256:
        fail("active validator hash differs from stable active bytes")
    return active_document, candidate_document, active_validation, candidate_validation


def _write_input_evidence(
    *,
    evidence_dir: Path,
    active: Snapshot,
    candidate: Snapshot,
    active_validation: dict[str, Any],
    candidate_validation: dict[str, Any],
    suite: tuple[SuiteCase, ...],
    prompt_suite_path: Path,
) -> None:
    # Keep the exact input bytes beside the readable output comparison. The
    # root-owned state copy remains the rollback authority; these evidence
    # copies let later readers independently audit what was compared.
    write_new(
        evidence_dir / "active-manifest-before.json",
        active.raw,
        "active manifest evidence copy",
    )
    write_new(
        evidence_dir / "candidate-manifest.json",
        candidate.raw,
        "candidate manifest evidence copy",
    )
    write_json_new(evidence_dir / "active-validation.json", active_validation, "active validation")
    write_json_new(evidence_dir / "candidate-validation.json", candidate_validation, "candidate validation")
    write_new(
        evidence_dir / "prompt-suite.json",
        read_snapshot(prompt_suite_path, "prompt suite", maximum=MAX_SUITE_BYTES).raw,
        "copied prompt suite",
    )
    write_json_new(
        evidence_dir / "preflight.json",
        preflight_report(
            active=active,
            candidate=candidate,
            active_validation=active_validation,
            candidate_validation=candidate_validation,
            suite=suite,
        ),
        "promotion preflight",
    )


def _prepare_transaction(
    *,
    state_dir: Path,
    active: Snapshot,
    candidate: Snapshot,
    service: str,
    base_url: str,
    gateway_container: str | None,
    prompt_suite: Path,
    evidence_dir: Path,
) -> tuple[Path, Snapshot, Snapshot]:
    run_dir = _new_run_directory(state_dir)
    rollback_snapshot = write_new(
        run_dir / "rollback-active.json",
        active.raw,
        "exact rollback manifest",
        mode=0o440,
    )
    if rollback_snapshot.raw != active.raw:
        fail("saved rollback bytes differ from active snapshot")
    transaction = {
        "schema_version": PROMOTION_TRANSACTION_SCHEMA,
        "run_id": run_dir.name,
        "created_at": utc_now(),
        "active_manifest_path": os.fspath(active.path),
        "candidate_manifest_path": os.fspath(candidate.path),
        "candidate_manifest_sha256": candidate.sha256,
        "rollback_manifest_path": os.fspath(rollback_snapshot.path),
        "rollback_manifest_sha256": rollback_snapshot.sha256,
        "service": service,
        "base_url": base_url,
        "gateway_container": gateway_container,
        "prompt_suite": os.fspath(prompt_suite),
        "evidence_directory": os.fspath(evidence_dir),
    }
    transaction_snapshot = write_json_new(
        run_dir / "transaction.json", transaction, "promotion transaction"
    )
    append_ledger(
        state_dir,
        {
            "schema_version": "ullm.lightweight_promotion.ledger.v1",
            "event": "prepared",
            "at": utc_now(),
            "run_id": run_dir.name,
            "candidate_manifest_sha256": candidate.sha256,
            "rollback_manifest_sha256": rollback_snapshot.sha256,
        },
    )
    return run_dir, rollback_snapshot, transaction_snapshot


def _write_failed_promotion(
    *,
    run_dir: Path,
    state_dir: Path,
    evidence_dir: Path,
    transaction_snapshot: Snapshot,
    candidate: Snapshot,
    rollback_snapshot: Snapshot,
    service_events: list[dict[str, Any]],
    failure: BaseException,
    rollback_detail: dict[str, Any],
) -> None:
    status = (
        "rolled_back_after_candidate_failure"
        if rollback_detail["service_response_verified"]
        else "candidate_failure_rollback_incomplete"
    )
    outcome = {
        "schema_version": PROMOTION_OUTCOME_SCHEMA,
        "status": status,
        "completed_at": utc_now(),
        "transaction_path": os.fspath(transaction_snapshot.path),
        "transaction_sha256": transaction_snapshot.sha256,
        "candidate_manifest_sha256": candidate.sha256,
        "rollback_manifest_sha256": rollback_snapshot.sha256,
        "failure": _error_text(failure),
        "rollback": rollback_detail,
        "service_restart_commands": len(service_events),
    }
    outcome_snapshot = write_json_new(
        run_dir / "activation-outcome.json", outcome, "failed promotion outcome"
    )
    write_json_new(evidence_dir / "outcome.json", outcome, "evidence failed promotion outcome")
    write_json_new(evidence_dir / "service-events.json", service_events, "service events")
    append_ledger(
        state_dir,
        {
            "schema_version": "ullm.lightweight_promotion.ledger.v1",
            "event": status,
            "at": utc_now(),
            "run_id": run_dir.name,
            "outcome_sha256": outcome_snapshot.sha256,
            "candidate_manifest_sha256": candidate.sha256,
            "rollback_manifest_sha256": rollback_snapshot.sha256,
        },
    )


def promote(args: argparse.Namespace) -> dict[str, Any]:
    active_path = Path(args.active_manifest).absolute()
    candidate_path = Path(args.candidate_manifest).absolute()
    prompt_suite_path = Path(args.prompt_suite).absolute()
    base_url = _validate_base_url(args.base_url)
    gateway_container = normalize_gateway_container(args.gateway_container)
    suite = load_suite(prompt_suite_path)
    active = read_snapshot(active_path, "active manifest")
    if not args.yes:
        candidate = read_snapshot(candidate_path, "candidate manifest")
        _, _, active_validation, candidate_validation = _validate_promotable_inputs(
            active=active, candidate=candidate, suite=suite
        )
        return preflight_report(
            active=active,
            candidate=candidate,
            active_validation=active_validation,
            candidate_validation=candidate_validation,
            suite=suite,
        )

    _require_root_for_mutation()
    evidence_dir = _prepare_evidence_directory(Path(args.evidence_dir))
    if args.semantic_self_test:
        source = read_snapshot(candidate_path, "semantic self-test source")
        if source.raw != active.raw:
            fail("semantic self-test source must equal current active bytes")
        semantic = canonical_json(strict_object(source.raw, "semantic self-test source"))
        if semantic == source.raw:
            semantic += b"\n"
        candidate = write_new(
            evidence_dir / "semantic-self-test-candidate.json",
            semantic,
            "semantic self-test candidate",
            mode=0o444,
        )
    else:
        candidate = read_snapshot(candidate_path, "candidate manifest")
    active_document, candidate_document, active_validation, candidate_validation = _validate_promotable_inputs(
        active=active, candidate=candidate, suite=suite
    )
    _write_input_evidence(
        evidence_dir=evidence_dir,
        active=active,
        candidate=candidate,
        active_validation=active_validation,
        candidate_validation=candidate_validation,
        suite=suite,
        prompt_suite_path=prompt_suite_path,
    )
    token = read_token(Path(args.token_file).absolute())
    baseline_attempts = wait_for_live_gateway(
        base_url=base_url,
        token=token,
        model_id=str(active_validation["model_id"]),
        timeout_seconds=args.startup_timeout_seconds,
        gateway_container=gateway_container,
    )
    write_json_new(evidence_dir / "baseline-readiness.json", baseline_attempts, "baseline readiness")
    baseline = run_suite(
        suite=suite,
        model_id=str(active_validation["model_id"]),
        manifest_document=active_document,
        base_url=base_url,
        token=token,
        request_timeout_seconds=args.request_timeout_seconds,
        output_dir=evidence_dir / "active-output",
        gateway_container=gateway_container,
    )
    if any(item.get("analysis", {}).get("blocking") for item in baseline):
        write_json_new(
            evidence_dir / "outcome.json",
            {
                "schema_version": PROMOTION_OUTCOME_SCHEMA,
                "status": "baseline_failed_before_mutation",
                "at": utc_now(),
            },
            "baseline failure outcome",
        )
        fail("active baseline generation failed; active bytes were not changed")

    state_dir = ensure_state_directory(Path(args.state_dir))
    run_dir, rollback_snapshot, transaction_snapshot = _prepare_transaction(
        state_dir=state_dir,
        active=active,
        candidate=candidate,
        service=args.service,
        base_url=base_url,
        gateway_container=gateway_container,
        prompt_suite=prompt_suite_path,
        evidence_dir=evidence_dir,
    )
    return _execute_promotion(
        args=args,
        active=active,
        candidate=candidate,
        active_document=active_document,
        candidate_document=candidate_document,
        active_validation=active_validation,
        candidate_validation=candidate_validation,
        suite=suite,
        token=token,
        base_url=base_url,
        gateway_container=gateway_container,
        baseline=baseline,
        evidence_dir=evidence_dir,
        state_dir=state_dir,
        run_dir=run_dir,
        rollback_snapshot=rollback_snapshot,
        transaction_snapshot=transaction_snapshot,
    )


def _automatic_rollback(
    *,
    args: argparse.Namespace,
    active: Snapshot,
    candidate: Snapshot,
    active_document: dict[str, Any],
    active_validation: dict[str, Any],
    suite: tuple[SuiteCase, ...],
    token: str,
    base_url: str,
    gateway_container: str | None,
    service_events: list[dict[str, Any]],
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "attempted": True,
        "bytes_restored": False,
        "service_response_verified": False,
    }
    try:
        with active_lock(active.path):
            current = read_snapshot(active.path, "candidate active manifest before automatic rollback")
            if current.raw != candidate.raw:
                fail("candidate active bytes drifted before automatic rollback")
            atomic_switch(active.path, candidate.raw, active.raw)
            detail["bytes_restored"] = read_snapshot(
                active.path, "active manifest after automatic rollback"
            ).raw == active.raw
            service_events.append(restart_service(args.service))
            attempts = wait_for_live_gateway(
                base_url=base_url,
                token=token,
                model_id=str(active_validation["model_id"]),
                timeout_seconds=args.startup_timeout_seconds,
                gateway_container=gateway_container,
            )
            detail["readiness_attempts"] = attempts
            detail["response_probe"] = _response_probe(
                case=suite[0],
                model_id=str(active_validation["model_id"]),
                manifest_document=active_document,
                base_url=base_url,
                token=token,
                request_timeout_seconds=args.request_timeout_seconds,
                gateway_container=gateway_container,
            )
            detail["service_response_verified"] = True
    except BaseException as error:
        detail["rollback_error"] = _error_text(error)
    return detail


def _execute_promotion(
    *,
    args: argparse.Namespace,
    active: Snapshot,
    candidate: Snapshot,
    active_document: dict[str, Any],
    candidate_document: dict[str, Any],
    active_validation: dict[str, Any],
    candidate_validation: dict[str, Any],
    suite: tuple[SuiteCase, ...],
    token: str,
    base_url: str,
    gateway_container: str | None,
    baseline: list[dict[str, Any]],
    evidence_dir: Path,
    state_dir: Path,
    run_dir: Path,
    rollback_snapshot: Snapshot,
    transaction_snapshot: Snapshot,
) -> dict[str, Any]:
    service_events: list[dict[str, Any]] = []
    switched = False
    try:
        with active_lock(active.path):
            current = read_snapshot(active.path, "active manifest before switch")
            if current.raw != active.raw:
                fail("active manifest drifted after baseline capture")
            latest_candidate = read_snapshot(candidate.path, "candidate manifest before switch")
            if latest_candidate.raw != candidate.raw:
                fail("candidate manifest drifted after validation")
            switched = atomic_switch(active.path, active.raw, candidate.raw)
            if not switched:
                fail("candidate bytes equal active bytes; use --semantic-self-test for a transactional self-test")
            service_events.append(restart_service(args.service))
            candidate_attempts = wait_for_live_gateway(
                base_url=base_url,
                token=token,
                model_id=str(candidate_validation["model_id"]),
                timeout_seconds=args.startup_timeout_seconds,
                gateway_container=gateway_container,
            )
            write_json_new(
                evidence_dir / "candidate-readiness.json",
                candidate_attempts,
                "candidate readiness",
            )
            candidate_records = run_suite(
                suite=suite,
                model_id=str(candidate_validation["model_id"]),
                manifest_document=candidate_document,
                base_url=base_url,
                token=token,
                request_timeout_seconds=args.request_timeout_seconds,
                output_dir=evidence_dir / "candidate-output",
                gateway_container=gateway_container,
            )
            comparison = compare_suites(suite, baseline, candidate_records)
            write_json_new(evidence_dir / "comparison.json", comparison, "generation comparison")
            write_comparison_markdown(
                evidence_dir / "comparison.md",
                suite,
                baseline,
                candidate_records,
                comparison,
            )
            if comparison["passed"] is not True:
                fail("candidate generation has automated blocking findings")
    except BaseException as error:
        rollback_detail = (
            _automatic_rollback(
                args=args,
                active=active,
                candidate=candidate,
                active_document=active_document,
                active_validation=active_validation,
                suite=suite,
                token=token,
                base_url=base_url,
                gateway_container=gateway_container,
                service_events=service_events,
            )
            if switched
            else {
                "attempted": False,
                "bytes_restored": False,
                "service_response_verified": False,
            }
        )
        _write_failed_promotion(
            run_dir=run_dir,
            state_dir=state_dir,
            evidence_dir=evidence_dir,
            transaction_snapshot=transaction_snapshot,
            candidate=candidate,
            rollback_snapshot=rollback_snapshot,
            service_events=service_events,
            failure=error,
            rollback_detail=rollback_detail,
        )
        if isinstance(error, KeyboardInterrupt):
            raise
        raise PromotionError(
            "candidate failure was rolled back"
            if rollback_detail["service_response_verified"]
            else "candidate failure rollback is incomplete"
        ) from error

    outcome = {
        "schema_version": PROMOTION_OUTCOME_SCHEMA,
        "status": "activated",
        "completed_at": utc_now(),
        "transaction_path": os.fspath(transaction_snapshot.path),
        "transaction_sha256": transaction_snapshot.sha256,
        "candidate_manifest_sha256": candidate.sha256,
        "rollback_manifest_sha256": rollback_snapshot.sha256,
        "evidence_directory": os.fspath(evidence_dir),
        "service_restart_commands": len(service_events),
    }
    outcome_snapshot = write_json_new(
        run_dir / "activation-outcome.json", outcome, "promotion outcome"
    )
    write_json_new(evidence_dir / "outcome.json", outcome, "evidence promotion outcome")
    write_json_new(evidence_dir / "service-events.json", service_events, "service events")
    append_ledger(
        state_dir,
        {
            "schema_version": "ullm.lightweight_promotion.ledger.v1",
            "event": "activated",
            "at": utc_now(),
            "run_id": run_dir.name,
            "outcome_sha256": outcome_snapshot.sha256,
            "candidate_manifest_sha256": candidate.sha256,
            "rollback_manifest_sha256": rollback_snapshot.sha256,
        },
    )
    return {
        "schema_version": PROMOTION_OUTCOME_SCHEMA,
        "status": "activated",
        "activation_outcome": os.fspath(outcome_snapshot.path),
        "activation_outcome_sha256": outcome_snapshot.sha256,
        "evidence_directory": os.fspath(evidence_dir),
        "service_restart_commands": len(service_events),
    }


def _load_rollback_inputs(
    activation_outcome_path: Path,
) -> tuple[Snapshot, dict[str, Any], Snapshot, dict[str, Any], Snapshot, Snapshot, Path]:
    outcome_snapshot = read_snapshot(activation_outcome_path.absolute(), "activation outcome")
    outcome = strict_object(outcome_snapshot.raw, "activation outcome")
    if outcome.get("schema_version") != PROMOTION_OUTCOME_SCHEMA or outcome.get("status") != "activated":
        fail("activation outcome is not a successful lightweight promotion")
    transaction_path = outcome.get("transaction_path")
    transaction_hash = outcome.get("transaction_sha256")
    if (
        not isinstance(transaction_path, str)
        or not isinstance(transaction_hash, str)
        or HASH_RE.fullmatch(transaction_hash) is None
    ):
        fail("activation outcome lacks transaction path")
    transaction_snapshot = read_snapshot(Path(transaction_path), "promotion transaction")
    if transaction_snapshot.sha256 != transaction_hash:
        fail("promotion transaction hash differs from activation outcome")
    transaction = strict_object(transaction_snapshot.raw, "promotion transaction")
    if transaction.get("schema_version") != PROMOTION_TRANSACTION_SCHEMA:
        fail("promotion transaction schema is unsupported")
    active_value = transaction.get("active_manifest_path")
    candidate_hash = transaction.get("candidate_manifest_sha256")
    rollback_value = transaction.get("rollback_manifest_path")
    rollback_hash = transaction.get("rollback_manifest_sha256")
    if (
        not isinstance(active_value, str)
        or not isinstance(rollback_value, str)
        or not isinstance(candidate_hash, str)
        or HASH_RE.fullmatch(candidate_hash) is None
        or not isinstance(rollback_hash, str)
        or HASH_RE.fullmatch(rollback_hash) is None
    ):
        fail("promotion transaction fields are invalid")
    active_path = Path(active_value).absolute()
    rollback_snapshot = read_snapshot(Path(rollback_value), "saved rollback manifest")
    if rollback_snapshot.sha256 != rollback_hash:
        fail("saved rollback manifest hash differs from transaction")
    if outcome.get("candidate_manifest_sha256") != candidate_hash:
        fail("candidate manifest hash differs between outcome and transaction")
    if outcome.get("rollback_manifest_sha256") != rollback_hash:
        fail("rollback manifest hash differs between outcome and transaction")
    current = read_snapshot(active_path, "active manifest")
    return (
        outcome_snapshot,
        outcome,
        transaction_snapshot,
        transaction,
        rollback_snapshot,
        current,
        active_path,
    )


def rollback(args: argparse.Namespace) -> dict[str, Any]:
    (
        outcome_snapshot,
        _outcome,
        _transaction_snapshot,
        transaction,
        rollback_snapshot,
        current,
        active_path,
    ) = _load_rollback_inputs(Path(args.activation_outcome))
    candidate_hash = str(transaction["candidate_manifest_sha256"])
    gateway_container_value = transaction.get("gateway_container", DEFAULT_GATEWAY_CONTAINER)
    if gateway_container_value is not None and not isinstance(gateway_container_value, str):
        fail("promotion transaction gateway container is invalid")
    gateway_container = (
        None
        if gateway_container_value is None
        else normalize_gateway_container(gateway_container_value)
    )
    preflight = {
        "schema_version": "ullm.lightweight_promotion.rollback_preflight.v1",
        "ready": current.sha256 == candidate_hash and current.raw != rollback_snapshot.raw,
        "active_manifest_sha256": current.sha256,
        "expected_candidate_manifest_sha256": candidate_hash,
        "rollback_manifest_sha256": rollback_snapshot.sha256,
        "strict_byte_difference": current.raw != rollback_snapshot.raw,
    }
    if not args.yes:
        return preflight
    _require_root_for_mutation()
    if preflight["ready"] is not True:
        fail("rollback preflight rejects current active bytes")
    evidence_dir = _prepare_evidence_directory(Path(args.evidence_dir))
    write_new(
        evidence_dir / "active-manifest-before-rollback.json",
        current.raw,
        "active manifest before rollback evidence copy",
    )
    write_new(
        evidence_dir / "saved-rollback-manifest.json",
        rollback_snapshot.raw,
        "saved rollback manifest evidence copy",
    )
    base_url = _validate_base_url(str(transaction["base_url"]))
    token = read_token(Path(args.token_file).absolute())
    suite = load_suite(Path(args.prompt_suite).absolute())
    rollback_document = strict_object(rollback_snapshot.raw, "saved rollback manifest")
    rollback_validation = validate_manifest(rollback_snapshot.path)
    events: list[dict[str, Any]] = []
    try:
        with active_lock(active_path):
            locked_current = read_snapshot(active_path, "active manifest before rollback")
            if locked_current.raw != current.raw or locked_current.sha256 != candidate_hash:
                fail("active manifest drifted before rollback")
            if locked_current.raw == rollback_snapshot.raw:
                fail("rollback requires active_snapshot.raw != rollback_snapshot.raw")
            atomic_switch(active_path, locked_current.raw, rollback_snapshot.raw)
            events.append(restart_service(str(transaction["service"])))
            readiness = wait_for_live_gateway(
                base_url=base_url,
                token=token,
                model_id=str(rollback_validation["model_id"]),
                timeout_seconds=args.startup_timeout_seconds,
                gateway_container=gateway_container,
            )
            probe = _response_probe(
                case=suite[0],
                model_id=str(rollback_validation["model_id"]),
                manifest_document=rollback_document,
                base_url=base_url,
                token=token,
                request_timeout_seconds=args.request_timeout_seconds,
                gateway_container=gateway_container,
            )
    except BaseException as error:
        failed = {
            "schema_version": ROLLBACK_OUTCOME_SCHEMA,
            "status": "rollback_incomplete",
            "completed_at": utc_now(),
            "activation_outcome_path": os.fspath(outcome_snapshot.path),
            "failure": _error_text(error),
            "bytes_equal_rollback": read_snapshot(
                active_path, "active manifest after failed rollback"
            ).raw == rollback_snapshot.raw,
            "service_restart_commands": len(events),
        }
        write_json_new(evidence_dir / "rollback-outcome.json", failed, "failed rollback outcome")
        write_json_new(evidence_dir / "service-events.json", events, "rollback service events")
        state_dir = rollback_snapshot.path.parent.parent
        append_ledger(
            state_dir,
            {
                "schema_version": "ullm.lightweight_promotion.ledger.v1",
                "event": "rollback_incomplete",
                "at": utc_now(),
                "run_id": transaction["run_id"],
                "activation_outcome_sha256": outcome_snapshot.sha256,
            },
        )
        raise PromotionError("rollback_incomplete") from error
    rollback_result = {
        "schema_version": ROLLBACK_OUTCOME_SCHEMA,
        "status": "rolled_back",
        "completed_at": utc_now(),
        "activation_outcome_path": os.fspath(outcome_snapshot.path),
        "activation_outcome_sha256": outcome_snapshot.sha256,
        "active_manifest_sha256": rollback_snapshot.sha256,
        "readiness_attempts": readiness,
        "response_probe": probe,
        "service_restart_commands": len(events),
    }
    write_json_new(evidence_dir / "rollback-outcome.json", rollback_result, "rollback outcome")
    write_json_new(evidence_dir / "service-events.json", events, "rollback service events")
    state_dir = rollback_snapshot.path.parent.parent
    rollback_outcome = write_json_new(
        rollback_snapshot.path.parent / "rollback-outcome.json",
        rollback_result,
        "state rollback outcome",
    )
    append_ledger(
        state_dir,
        {
            "schema_version": "ullm.lightweight_promotion.ledger.v1",
            "event": "rolled_back",
            "at": utc_now(),
            "run_id": transaction["run_id"],
            "rollback_outcome_sha256": rollback_outcome.sha256,
            "active_manifest_sha256": rollback_snapshot.sha256,
        },
    )
    return {
        "schema_version": ROLLBACK_OUTCOME_SCHEMA,
        "status": "rolled_back",
        "rollback_outcome": os.fspath(rollback_outcome.path),
        "rollback_outcome_sha256": rollback_outcome.sha256,
        "evidence_directory": os.fspath(evidence_dir),
        "service_restart_commands": len(events),
    }


def add_promotion_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--active-manifest", type=Path, default=DEFAULT_ACTIVE_MANIFEST)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--gateway-container",
        default=DEFAULT_GATEWAY_CONTAINER,
        help="local Docker container used to reach the bridge-bound gateway; use 'direct' for host HTTP",
    )
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--prompt-suite", type=Path, default=DEFAULT_PROMPT_SUITE)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--startup-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--semantic-self-test", action="store_true")
    parser.add_argument("--yes", action="store_true")


def parse_promotion_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote one manifest through the lightweight route.")
    add_promotion_arguments(parser)
    args = parser.parse_args(argv)
    if args.startup_timeout_seconds <= 0 or args.request_timeout_seconds <= 0:
        parser.error("timeouts must be positive")
    try:
        normalize_gateway_container(args.gateway_container)
    except PromotionError as error:
        parser.error(str(error))
    if args.semantic_self_test and not args.yes:
        parser.error("--semantic-self-test requires --yes")
    return args


def parse_rollback_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rollback one successful lightweight promotion.")
    parser.add_argument("--activation-outcome", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--prompt-suite", type=Path, default=DEFAULT_PROMPT_SUITE)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--startup-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    if args.startup_timeout_seconds <= 0 or args.request_timeout_seconds <= 0:
        parser.error("timeouts must be positive")
    return args


def _print_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")))


def promote_main(argv: Sequence[str] | None = None) -> int:
    try:
        _print_report(promote(parse_promotion_args(argv)))
    except Exception as error:
        print(f"lightweight promotion failed: {_error_text(error)}", file=sys.stderr)
        return 1
    return 0


def rollback_main(argv: Sequence[str] | None = None) -> int:
    try:
        _print_report(rollback(parse_rollback_args(argv)))
    except Exception as error:
        print(f"lightweight rollback failed: {_error_text(error)}", file=sys.stderr)
        return 1
    return 0
