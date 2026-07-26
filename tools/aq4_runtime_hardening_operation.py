#!/usr/bin/env python3
"""Sealed operation payload for the AQ4 runtime-hardening activation route.

The root-only ELF launcher seals this file's exact bytes before invoking it
with ``/usr/bin/python3 -I -S -B``.  The payload has three narrowly scoped
actions:

* ``reconcile`` restarts ``ullm-openai.service`` and waits for a bounded,
  coherent readiness proof;
* ``observe`` obtains the same bounded readiness proof without a restart; and
* ``isolated-preflight`` starts only the candidate worker as the gateway user,
  waits for its protocol-ready record, then terminates that isolated process.

It never writes the active manifest.  Endpoint and failure records deliberately
contain only fixed reason codes, HTTP status numbers, and hashes; credentials,
response bodies, command environments, and exception text are never emitted.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import pwd
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SERVICE = "ullm-openai.service"
DOCKER = "/usr/bin/docker"
SYSTEMCTL = "/usr/bin/systemctl"
GATEWAY_CONTAINER = "open-webui"
GATEWAY_URL = "http://172.20.0.1:8000"
OPENWEBUI_HOST = "127.0.0.1"
OPENWEBUI_PORT = 3000
LIVE_ACTIVE_MANIFEST = Path("/etc/ullm/served-models/active.json")
GATEWAY_API_KEY = Path("/etc/ullm/openai-api-key")
OPENWEBUI_SESSION = Path("/run/ullm-campaign-secrets/openwebui-session.jwt")
MODEL_ID = "ullm-qwen3.5-9b-aq4"
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
HIP_GUARD_RE = re.compile(r"ULLM_REQUIRE_HIP_[A-Z0-9_]+\Z")
MAX_RESPONSE_BYTES = 1_048_576
MAX_PROC_ENVIRONMENT_BYTES = 1_048_576
MAX_WORKER_CAPTURE_BYTES = 4 * 1024 * 1024

# The direct candidate controls reached worker-ready in about 4.8 seconds and
# the live control in about 6.1 seconds; the actual gateway completed startup
# in about three seconds.  120 seconds tolerates cold load/host scheduling by
# roughly twenty times that observed interval while remaining a bounded failure.
READINESS_TIMEOUT_SECONDS = 120.0
READINESS_MAX_ATTEMPTS = 15
READINESS_INITIAL_BACKOFF_SECONDS = 0.5
READINESS_MAX_BACKOFF_SECONDS = 8.0
READINESS_STABLE_PID_OBSERVATIONS = 2
PROBE_TIMEOUT_SECONDS = 4.0

ISOLATED_WORKER_TIMEOUT_SECONDS = 120.0
ISOLATED_WORKER_TERMINATE_GRACE_SECONDS = 10.0

LIVE_OBSERVATION_SCHEMA = "ullm.aq4_runtime_hardening_live_observation.v3"
READINESS_FAILURE_SCHEMA = "ullm.aq4_runtime_hardening_readiness_failure.v1"
ISOLATED_WORKER_OBSERVATION_SCHEMA = (
    "ullm.aq4_runtime_hardening_isolated_worker_observation.v1"
)

LIVE_STAGES = frozenset({"candidate_live_proof", "rollback_live_proof"})
RECONCILE_STAGES = frozenset({"candidate_reconcile", "rollback_reconcile"})
ISOLATED_PREFLIGHT_STAGE = "candidate_isolated_preflight"
ENDPOINTS = (
    "gateway_health",
    "gateway_ready",
    "gateway_models",
    "openwebui_health",
    "openwebui_models",
)


class OperationError(RuntimeError):
    """The sealed operation could not establish the required live facts."""


class ReadinessError(OperationError):
    """A bounded readiness wait failed with a secret-free diagnostic."""

    def __init__(self, diagnostic: dict[str, Any]) -> None:
        super().__init__("AQ4 readiness contract was not met")
        self.diagnostic = diagnostic


class IsolatedPreflightError(OperationError):
    """The isolated worker did not produce a valid ready record."""

    def __init__(self, diagnostic: dict[str, Any]) -> None:
        super().__init__("AQ4 isolated candidate worker did not become ready")
        self.diagnostic = diagnostic


def _canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _require_environment() -> dict[str, str]:
    keys = (
        "ULLM_AQ4_RUNTIME_HARDENING_STAGE",
        "ULLM_AQ4_RUNTIME_HARDENING_PLAN_SHA256",
        "ULLM_AQ4_RUNTIME_HARDENING_EPOCH",
        "ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_MANIFEST",
        "ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_SHA256",
    )
    values = {key: os.environ.get(key, "") for key in keys}
    if (
        any(not value for value in values.values())
        or HASH_RE.fullmatch(values["ULLM_AQ4_RUNTIME_HARDENING_PLAN_SHA256"]) is None
        or HASH_RE.fullmatch(values["ULLM_AQ4_RUNTIME_HARDENING_EPOCH"]) is None
        or HASH_RE.fullmatch(values["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_SHA256"]) is None
    ):
        raise OperationError("activation binding environment is invalid")
    active = Path(values["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_MANIFEST"])
    if not active.is_absolute() or active.is_symlink():
        raise OperationError("activation manifest path is unsafe")
    return values


def _stable_read(path: Path, *, maximum: int, expected_mode: int | None = None) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise OperationError("required file is unavailable") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 1
            or before.st_size > maximum
            or before.st_uid != 0
            or before.st_nlink != 1
            or (expected_mode is not None and stat.S_IMODE(before.st_mode) != expected_mode)
            or (expected_mode is None and stat.S_IMODE(before.st_mode) & 0o022)
        ):
            raise OperationError("required file seal is unsafe")
        raw = bytearray()
        while len(raw) <= maximum:
            part = os.read(descriptor, min(65_536, maximum + 1 - len(raw)))
            if not part:
                break
            raw.extend(part)
        after = os.fstat(descriptor)
        named = path.lstat()
        if (
            len(raw) != before.st_size
            or len(raw) > maximum
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            != (named.st_dev, named.st_ino, named.st_size, named.st_mtime_ns, named.st_ctime_ns)
        ):
            raise OperationError("required file changed while being read")
        return bytes(raw)
    finally:
        os.close(descriptor)


def _sha256_path(path: Path) -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise OperationError("executable is unavailable") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 1
            or before.st_size > 512 * 1024 * 1024
            or stat.S_IMODE(before.st_mode) & 0o022
        ):
            raise OperationError("executable seal is unsafe")
        digest = hashlib.sha256()
        total = 0
        while True:
            part = os.read(descriptor, min(1 << 20, 512 * 1024 * 1024 - total + 1))
            if not part:
                break
            total += len(part)
            if total > 512 * 1024 * 1024:
                raise OperationError("executable exceeds its bound")
            digest.update(part)
        after = os.fstat(descriptor)
        if (
            total != before.st_size
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        ):
            raise OperationError("executable changed while being hashed")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _read_secret(path: Path) -> bytearray:
    raw = bytearray(_stable_read(path, maximum=8192, expected_mode=0o640))
    while raw and raw[-1] in b"\r\n":
        raw.pop()
    if not raw or any(value < 0x21 or value > 0x7E for value in raw):
        raise OperationError("credential is invalid")
    return raw


def _run(
    argv: list[str], *, input_value: bytes | None = None, timeout: float
) -> subprocess.CompletedProcess[bytes]:
    kwargs: dict[str, Any] = {
        "check": False,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "timeout": timeout,
    }
    if input_value is None:
        kwargs["stdin"] = subprocess.DEVNULL
    else:
        kwargs["input"] = input_value
    try:
        result = subprocess.run(argv, **kwargs)
    except (OSError, subprocess.SubprocessError) as error:
        raise OperationError("required operation could not run") from error
    if result.returncode != 0:
        raise OperationError("required operation failed")
    return result


def _systemctl_value(*arguments: str) -> str:
    output = _run([SYSTEMCTL, *arguments], timeout=30).stdout.decode("ascii", errors="strict").strip()
    if not output:
        raise OperationError("systemd identity is absent")
    return output


def _service_identity() -> tuple[str, str, int]:
    active = _systemctl_value("is-active", SERVICE)
    sub = _systemctl_value("show", SERVICE, "-p", "SubState", "--value")
    pid_value = _systemctl_value("show", SERVICE, "-p", "MainPID", "--value")
    try:
        pid = int(pid_value)
    except ValueError as error:
        raise OperationError("gateway PID is invalid") from error
    if active != "active" or sub != "running" or pid < 1:
        raise OperationError("gateway is not active/running")
    return active, sub, pid


def _process_identity(pid: int) -> dict[str, Any]:
    try:
        stat_fields = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()
        ppid = int(stat_fields[1])
        starttime = int(stat_fields[19])
        executable = Path(os.readlink(f"/proc/{pid}/exe"))
    except (IndexError, OSError, UnicodeError, ValueError) as error:
        raise OperationError("gateway process identity is unavailable") from error
    if ppid < 1 or starttime < 1:
        raise OperationError("gateway process identity is invalid")
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise OperationError("boot identity is unavailable") from error
    if not boot_id:
        raise OperationError("boot identity is invalid")
    return {
        "boot_id": boot_id,
        "pid": pid,
        "ppid": ppid,
        "starttime": starttime,
        "executable_sha256": _sha256_path(executable),
    }


def _proc_environment(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError as error:
        raise OperationError("gateway environment is unavailable") from error
    if len(raw) > MAX_PROC_ENVIRONMENT_BYTES:
        raise OperationError("gateway environment exceeds its bound")
    values: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item:
            continue
        key, separator, value = item.partition(b"=")
        if not separator:
            continue
        try:
            decoded_key = key.decode("ascii", errors="strict")
            decoded_value = value.decode("ascii", errors="strict")
        except UnicodeError:
            continue
        values[decoded_key] = decoded_value
    return values


def _children(pid: int) -> tuple[int, ...]:
    try:
        raw = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="ascii")
    except OSError as error:
        raise OperationError("gateway child process list is unavailable") from error
    try:
        return tuple(int(item) for item in raw.split() if int(item) > 0)
    except ValueError as error:
        raise OperationError("gateway child process list is invalid") from error


def _command_line(pid: int) -> tuple[str, ...]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError as error:
        raise OperationError("worker command line is unavailable") from error
    if not raw or len(raw) > MAX_PROC_ENVIRONMENT_BYTES:
        raise OperationError("worker command line is invalid")
    try:
        values = tuple(item.decode("utf-8", errors="strict") for item in raw.split(b"\0") if item)
    except UnicodeError as error:
        raise OperationError("worker command line is invalid") from error
    if not values:
        raise OperationError("worker command line is absent")
    return values


def _find_worker_pid(pid: int, *, worker: str, manifest: str) -> int | None:
    """Find the worker child that is actually bound to one manifest path."""

    pending = [pid]
    visited: set[int] = set()
    while pending and len(visited) < 32:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        try:
            children = _children(current)
        except OperationError:
            continue
        for child in children:
            pending.append(child)
            try:
                command = _command_line(child)
            except OperationError:
                continue
            if command[0] != worker:
                continue
            for index, item in enumerate(command[:-1]):
                if item == "--served-model-manifest" and command[index + 1] == manifest:
                    return child
    return None


def _model_ids(raw: bytes) -> list[str]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OperationError("model endpoint response is invalid") from error
    if isinstance(value, dict):
        candidates = value.get("data") if isinstance(value.get("data"), list) else value.get("models")
    else:
        candidates = value
    if not isinstance(candidates, list):
        raise OperationError("model endpoint response shape differs")
    identifiers: set[str] = set()
    for item in candidates:
        if isinstance(item, dict):
            candidate = item.get("id") or item.get("model") or item.get("name")
            if isinstance(candidate, str) and candidate.startswith("ullm-"):
                identifiers.add(candidate)
    return sorted(identifiers)


def _docker_gateway_get(
    path: str, authorization: bytearray | None, *, timeout: float
) -> tuple[int, bytes]:
    config = bytearray(f'url = "{GATEWAY_URL}{path}"\n'.encode("ascii"))
    if authorization is not None:
        config.extend(b'header = "Authorization: Bearer ')
        config.extend(authorization)
        config.extend(b'"\n')
    try:
        result = _run(
            [
                DOCKER,
                "exec",
                "-i",
                GATEWAY_CONTAINER,
                "/usr/bin/curl",
                "--silent",
                "--show-error",
                "--max-time",
                str(max(1, int(timeout + 0.999))),
                "--config",
                "-",
                "--write-out",
                "\n%{http_code}",
            ],
            input_value=bytes(config),
            timeout=timeout + 2,
        )
    finally:
        config[:] = b"\x00" * len(config)
    body, separator, status_value = result.stdout.rpartition(b"\n")
    if not separator or len(body) > MAX_RESPONSE_BYTES:
        raise OperationError("gateway response is invalid")
    try:
        status = int(status_value)
    except ValueError as error:
        raise OperationError("gateway status is invalid") from error
    return status, body


def _openwebui_get(
    path: str, authorization: bytearray | None, *, timeout: float
) -> tuple[int, bytes]:
    headers = {} if authorization is None else {"Authorization": "Bearer " + authorization.decode("ascii")}
    connection = http.client.HTTPConnection(OPENWEBUI_HOST, OPENWEBUI_PORT, timeout=timeout)
    try:
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise OperationError("OpenWebUI response exceeds its bound")
        return response.status, raw
    except OSError as error:
        raise OperationError("OpenWebUI endpoint probe failed") from error
    finally:
        connection.close()


def _read_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _stable_read(path, maximum=4 * 1024 * 1024)
    try:
        manifest = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OperationError("served-model manifest JSON is invalid") from error
    if not isinstance(manifest, dict):
        raise OperationError("served-model manifest root is invalid")
    return manifest, raw


def _read_active(values: dict[str, str]) -> tuple[dict[str, Any], bytes]:
    raw_path = values["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_MANIFEST"]
    manifest, raw = _read_manifest(Path(raw_path))
    if hashlib.sha256(raw).hexdigest() != values["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_SHA256"]:
        raise OperationError("active manifest hash differs")
    return manifest, raw


def _manifest_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    public = manifest.get("public")
    worker = manifest.get("worker")
    product = manifest.get("product")
    if not isinstance(public, dict) or not isinstance(worker, dict) or not isinstance(product, dict):
        raise OperationError("active manifest identity is invalid")
    model_id = public.get("id")
    worker_path = worker.get("binary")
    worker_hash = worker.get("binary_sha256")
    arguments = worker.get("arguments")
    required_environment = worker.get("required_environment")
    package = product.get("package")
    if (
        model_id != MODEL_ID
        or not isinstance(worker_path, str)
        or not Path(worker_path).is_absolute()
        or not isinstance(worker_hash, str)
        or HASH_RE.fullmatch(worker_hash) is None
        or not isinstance(arguments, list)
        or not arguments
        or not all(isinstance(item, str) and item for item in arguments)
        or not isinstance(required_environment, list)
        or not required_environment
        or not all(isinstance(item, str) and HIP_GUARD_RE.fullmatch(item) for item in required_environment)
        or len(required_environment) != len(set(required_environment))
        or not isinstance(package, dict)
        or not isinstance(package.get("manifest_sha256"), str)
        or HASH_RE.fullmatch(package["manifest_sha256"]) is None
        or _sha256_path(Path(worker_path)) != worker_hash
    ):
        raise OperationError("active manifest worker identity differs")
    return {
        "model_id": model_id,
        "worker_path": worker_path,
        "worker_hash": worker_hash,
        "worker_arguments": tuple(arguments),
        "required_environment": tuple(required_environment),
        "package_manifest_sha256": package["manifest_sha256"],
    }


def _endpoint_state(*, ok: bool, status: int | None, cause: str | None) -> dict[str, Any]:
    return {"ok": ok, "status": status, "cause": cause}


def _unprobed_endpoints(cause: str) -> dict[str, dict[str, Any]]:
    return {name: _endpoint_state(ok=False, status=None, cause=cause) for name in ENDPOINTS}


def _probe(
    callback: Callable[[], tuple[int, bytes]], *, require_model: bool = False
) -> dict[str, Any]:
    try:
        status, body = callback()
    except OperationError:
        return _endpoint_state(ok=False, status=None, cause="transport")
    if status != 200:
        return _endpoint_state(ok=False, status=status, cause="http_status")
    if not require_model:
        return _endpoint_state(ok=True, status=status, cause=None)
    try:
        exact = _model_ids(body) == [MODEL_ID]
    except OperationError:
        return _endpoint_state(ok=False, status=status, cause="invalid_response")
    return _endpoint_state(
        ok=exact,
        status=status,
        cause=None if exact else "model_id_mismatch",
    )


def _probe_timeout(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise OperationError("readiness deadline elapsed")
    return min(PROBE_TIMEOUT_SECONDS, max(0.1, remaining))


def _probe_endpoints(deadline: float) -> dict[str, dict[str, Any]]:
    gateway_key: bytearray | None = None
    session: bytearray | None = None
    try:
        try:
            gateway_key = _read_secret(GATEWAY_API_KEY)
        except OperationError:
            gateway_key = None
        try:
            session = _read_secret(OPENWEBUI_SESSION)
        except OperationError:
            session = None
        endpoints = {
            "gateway_health": _probe(
                lambda: _docker_gateway_get("/healthz", None, timeout=_probe_timeout(deadline))
            ),
            "gateway_ready": _probe(
                lambda: _docker_gateway_get("/readyz", None, timeout=_probe_timeout(deadline))
            ),
            "gateway_models": (
                _endpoint_state(ok=False, status=None, cause="credential_unavailable")
                if gateway_key is None
                else _probe(
                    lambda: _docker_gateway_get(
                        "/v1/models", gateway_key, timeout=_probe_timeout(deadline)
                    ),
                    require_model=True,
                )
            ),
            "openwebui_health": _probe(
                lambda: _openwebui_get("/health", None, timeout=_probe_timeout(deadline))
            ),
            "openwebui_models": (
                _endpoint_state(ok=False, status=None, cause="credential_unavailable")
                if session is None
                else _probe(
                    lambda: _openwebui_get(
                        "/api/models", session, timeout=_probe_timeout(deadline)
                    ),
                    require_model=True,
                )
            ),
        }
    except OperationError:
        endpoints = _unprobed_endpoints("deadline_elapsed")
    finally:
        if gateway_key is not None:
            gateway_key[:] = b"\x00" * len(gateway_key)
        if session is not None:
            session[:] = b"\x00" * len(session)
    return endpoints


def _service_document(identity: tuple[str, str, int] | None) -> dict[str, Any]:
    if identity is None:
        return {"unit": SERVICE, "active_state": None, "sub_state": None}
    return {"unit": SERVICE, "active_state": identity[0], "sub_state": identity[1]}


def _readiness_attempt(values: dict[str, str], *, deadline: float) -> dict[str, Any]:
    manifest: dict[str, Any] | None = None
    raw: bytes | None = None
    contract: dict[str, Any] | None = None
    try:
        manifest, raw = _read_active(values)
        contract = _manifest_contract(manifest)
    except OperationError:
        pass

    before_identity: tuple[str, str, int] | None = None
    before_process: dict[str, Any] | None = None
    worker_environment_match = False
    worker_command_match = False
    try:
        before_identity = _service_identity()
        if contract is not None:
            worker_pid = _find_worker_pid(
                before_identity[2],
                worker=contract["worker_path"],
                manifest=values["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_MANIFEST"],
            )
            worker_command_match = worker_pid is not None
            if worker_pid is not None:
                before_process = _process_identity(worker_pid)
                worker_command_match = (
                    before_process["executable_sha256"] == contract["worker_hash"]
                )
                environment = _proc_environment(worker_pid)
                worker_environment_match = (
                    environment.get("ULLM_SERVED_MODEL_MANIFEST")
                    == values["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_MANIFEST"]
                )
    except OperationError:
        before_identity = None
        before_process = None

    endpoints = _probe_endpoints(deadline)

    after_identity: tuple[str, str, int] | None = None
    after_process: dict[str, Any] | None = None
    final_manifest: dict[str, Any] | None = None
    final_raw: bytes | None = None
    try:
        after_identity = _service_identity()
        if contract is not None:
            after_worker_pid = _find_worker_pid(
                after_identity[2],
                worker=contract["worker_path"],
                manifest=values["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_MANIFEST"],
            )
            if after_worker_pid is not None:
                after_process = _process_identity(after_worker_pid)
    except OperationError:
        pass
    try:
        final_manifest, final_raw = _read_active(values)
    except OperationError:
        pass

    file_match = (
        manifest is not None
        and raw is not None
        and final_manifest == manifest
        and final_raw == raw
    )
    process_match = (
        before_identity is not None
        and before_identity == after_identity
        and before_process is not None
        and before_process == after_process
    )
    endpoints_match = all(item["ok"] is True for item in endpoints.values())
    coherent = bool(
        contract is not None
        and file_match
        and process_match
        and worker_environment_match
        and worker_command_match
        and endpoints_match
    )
    if not file_match or contract is None:
        cause = "manifest_mismatch"
    elif before_identity is None or before_process is None:
        cause = "service_not_ready"
    elif not process_match:
        cause = "process_unstable"
    elif not worker_environment_match or not worker_command_match:
        cause = "process_manifest_mismatch"
    elif not endpoints_match:
        cause = "endpoints_incoherent"
    else:
        cause = "ready"
    return {
        "systemd": _service_document(before_identity),
        "process": before_process,
        "manifest": {
            "active_path": values["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_MANIFEST"],
            "active_manifest_sha256": values["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_SHA256"],
            "file_match": file_match,
            "worker_environment_match": worker_environment_match,
            "worker_command_match": worker_command_match,
        },
        "model_id": None if contract is None else contract["model_id"],
        "worker_binary_path": None if contract is None else contract["worker_path"],
        "worker_binary_sha256": None if contract is None else contract["worker_hash"],
        "endpoints": endpoints,
        "coherent": coherent,
        "cause": cause,
    }


def _readiness_document(
    values: dict[str, str],
    *,
    attempt: dict[str, Any],
    attempts: int,
    elapsed_milliseconds: int,
) -> dict[str, Any]:
    return {
        "schema_version": LIVE_OBSERVATION_SCHEMA,
        "plan_sha256": values["ULLM_AQ4_RUNTIME_HARDENING_PLAN_SHA256"],
        "operation_epoch": values["ULLM_AQ4_RUNTIME_HARDENING_EPOCH"],
        "active_manifest_sha256": values["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_SHA256"],
        "model_id": attempt["model_id"],
        "worker_binary_path": attempt["worker_binary_path"],
        "worker_binary_sha256": attempt["worker_binary_sha256"],
        "systemd": attempt["systemd"],
        "process": attempt["process"],
        "manifest": attempt["manifest"],
        "endpoints": attempt["endpoints"],
        "readiness": {
            "timeout_seconds": int(READINESS_TIMEOUT_SECONDS),
            "max_attempts": READINESS_MAX_ATTEMPTS,
            "attempts": attempts,
            "stable_pid_observations": READINESS_STABLE_PID_OBSERVATIONS,
            "elapsed_milliseconds": elapsed_milliseconds,
        },
    }


def _readiness_failure_document(
    values: dict[str, str],
    *,
    attempt: dict[str, Any],
    attempts: int,
    elapsed_milliseconds: int,
) -> dict[str, Any]:
    return {
        "schema_version": READINESS_FAILURE_SCHEMA,
        "plan_sha256": values["ULLM_AQ4_RUNTIME_HARDENING_PLAN_SHA256"],
        "operation_epoch": values["ULLM_AQ4_RUNTIME_HARDENING_EPOCH"],
        "active_manifest_sha256": values["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_SHA256"],
        "stage": values["ULLM_AQ4_RUNTIME_HARDENING_STAGE"],
        "failed_at": _utc_timestamp(),
        "cause": attempt["cause"],
        "systemd": attempt["systemd"],
        "process": attempt["process"],
        "manifest": attempt["manifest"],
        "endpoints": attempt["endpoints"],
        "readiness": {
            "timeout_seconds": int(READINESS_TIMEOUT_SECONDS),
            "max_attempts": READINESS_MAX_ATTEMPTS,
            "attempts": attempts,
            "stable_pid_observations": READINESS_STABLE_PID_OBSERVATIONS,
            "elapsed_milliseconds": elapsed_milliseconds,
        },
    }


def _retry_delay(retry: int) -> float:
    return min(
        READINESS_INITIAL_BACKOFF_SECONDS * (2 ** (retry - 1)),
        READINESS_MAX_BACKOFF_SECONDS,
    )


def wait_for_readiness(
    values: dict[str, str],
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Require two coherent observations of one unchanged gateway PID.

    The maximum of fifteen attempts uses 0.5, 1, 2, 4, then capped eight-second
    waits.  The complete idle schedule is 87.5 seconds, leaving time inside the
    fixed 120-second deadline for bounded endpoint probes.
    """

    started = clock()
    deadline = started + READINESS_TIMEOUT_SECONDS
    stable_process: dict[str, Any] | None = None
    stable_count = 0
    last: dict[str, Any] | None = None
    for number in range(1, READINESS_MAX_ATTEMPTS + 1):
        last = _readiness_attempt(values, deadline=deadline)
        if last["coherent"] is True and isinstance(last["process"], dict):
            if last["process"] == stable_process:
                stable_count += 1
            else:
                stable_process = last["process"]
                stable_count = 1
            if stable_count >= READINESS_STABLE_PID_OBSERVATIONS:
                return _readiness_document(
                    values,
                    attempt=last,
                    attempts=number,
                    elapsed_milliseconds=int((clock() - started) * 1000),
                )
            last = dict(last)
            last["cause"] = "pid_not_stable"
        else:
            stable_process = None
            stable_count = 0
        remaining = deadline - clock()
        if number == READINESS_MAX_ATTEMPTS or remaining <= 0:
            break
        sleep(min(_retry_delay(number), remaining))
    assert last is not None
    raise ReadinessError(
        _readiness_failure_document(
            values,
            attempt=last,
            attempts=number,
            elapsed_milliseconds=max(0, int((clock() - started) * 1000)),
        )
    )


def _service_user_and_working_directory() -> tuple[pwd.struct_passwd, Path]:
    user = _systemctl_value("show", SERVICE, "-p", "User", "--value")
    working_directory = Path(
        _systemctl_value("show", SERVICE, "-p", "WorkingDirectory", "--value")
    )
    if not working_directory.is_absolute() or working_directory.is_symlink() or not working_directory.is_dir():
        raise OperationError("gateway working directory is unsafe")
    try:
        account = pwd.getpwnam(user)
    except KeyError as error:
        raise OperationError("gateway service user is unavailable") from error
    return account, working_directory


def _live_worker_environment() -> dict[str, str]:
    """Read only the effective environment of the current manifest-bound worker.

    systemd's MainPID is the gateway process, not the worker process; the
    worker is the authority for HIP guards, device binding, and the served
    manifest.  Never inherit the gateway's complete environment because it may
    contain credentials unrelated to the worker launch.
    """

    _active, _sub, service_pid = _service_identity()
    live_manifest, _raw = _read_manifest(LIVE_ACTIVE_MANIFEST)
    live_contract = _manifest_contract(live_manifest)
    worker_pid = _find_worker_pid(
        service_pid,
        worker=live_contract["worker_path"],
        manifest=os.fspath(LIVE_ACTIVE_MANIFEST),
    )
    if worker_pid is None:
        raise OperationError("live manifest-bound worker is unavailable")
    environment = _proc_environment(worker_pid)
    if environment.get("ULLM_SERVED_MODEL_MANIFEST") != os.fspath(LIVE_ACTIVE_MANIFEST):
        raise OperationError("live worker manifest environment differs")
    return environment


def _isolated_environment(contract: dict[str, Any], candidate_manifest: str) -> dict[str, str]:
    live = _live_worker_environment()
    required = (
        "HOME",
        "XDG_CACHE_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "HIP_VISIBLE_DEVICES",
        "ULLM_HIP_VISIBLE_DEVICES",
        "ULLM_GPU_LOCK_FILE",
    )
    values: dict[str, str] = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin"}
    for name in required:
        value = live.get(name)
        if not value or any(character in value for character in "\r\n\0"):
            raise OperationError("gateway worker environment is incomplete")
        values[name] = value
    for name in contract["required_environment"]:
        if live.get(name) != "1":
            raise OperationError("gateway HIP guard environment differs")
        values[name] = "1"
    values["ULLM_SERVED_MODEL_MANIFEST"] = candidate_manifest
    values["PYTHONUNBUFFERED"] = "1"
    return values


def _worker_argv(contract: dict[str, Any], candidate_manifest: str) -> list[str]:
    arguments = [candidate_manifest if item == "{manifest}" else item for item in contract["worker_arguments"]]
    if arguments.count(candidate_manifest) != 1:
        raise OperationError("candidate worker arguments do not bind one manifest")
    return [contract["worker_path"], *arguments]


def _drop_to_service_user(account: pwd.struct_passwd) -> Callable[[], None]:
    def drop() -> None:
        os.initgroups(account.pw_name, account.pw_gid)
        os.setgid(account.pw_gid)
        os.setuid(account.pw_uid)

    return drop


def _append_capture(target: bytearray, value: bytes) -> None:
    if len(target) + len(value) > MAX_WORKER_CAPTURE_BYTES:
        raise OperationError("isolated worker output exceeds its bound")
    target.extend(value)


def _ready_record(stdout: bytes, contract: dict[str, Any]) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except (UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or value.get("type") != "ready":
            continue
        if (
            value.get("schema_version") == "ullm.worker.v2"
            and value.get("model") == contract["model_id"]
            and value.get("package_manifest_sha256") == contract["package_manifest_sha256"]
            and isinstance(value.get("device"), str)
            and isinstance(value.get("execution_profile"), str)
        ):
            return value
    return None


def _drain_selector(
    selector: selectors.BaseSelector,
    captures: dict[str, bytearray],
    *,
    timeout: float,
) -> None:
    for key, _mask in selector.select(timeout):
        stream = key.fileobj
        try:
            part = os.read(stream.fileno(), 65_536)
        except OSError:
            part = b""
        if not part:
            try:
                selector.unregister(stream)
            except Exception:
                pass
            continue
        _append_capture(captures[str(key.data)], part)


def _stop_isolated_worker(
    process: subprocess.Popen[bytes],
    selector: selectors.BaseSelector,
    captures: dict[str, bytearray],
) -> int:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + ISOLATED_WORKER_TERMINATE_GRACE_SECONDS
    while process.poll() is None and time.monotonic() < deadline:
        _drain_selector(selector, captures, timeout=0.1)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    while process.poll() is None:
        _drain_selector(selector, captures, timeout=0.1)
    while selector.get_map():
        _drain_selector(selector, captures, timeout=0)
    return process.returncode if process.returncode is not None else -signal.SIGKILL


def isolated_candidate_preflight(values: dict[str, str]) -> dict[str, Any]:
    if values["ULLM_AQ4_RUNTIME_HARDENING_STAGE"] != ISOLATED_PREFLIGHT_STAGE:
        raise OperationError("isolated preflight stage differs")
    manifest, _raw = _read_active(values)
    contract = _manifest_contract(manifest)
    account, working_directory = _service_user_and_working_directory()
    candidate_manifest = values["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_MANIFEST"]
    argv: list[str] = _worker_argv(contract, candidate_manifest)
    environment = _isolated_environment(contract, candidate_manifest)
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    captures = {"stdout": bytearray(), "stderr": bytearray()}
    ready: dict[str, Any] | None = None
    returncode: int | None = None
    try:
        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_directory,
                env=environment,
                close_fds=True,
                start_new_session=True,
                preexec_fn=_drop_to_service_user(account),
            )
        except OSError as error:
            raise OperationError("isolated candidate worker could not start") from error
        if process.stdout is None or process.stderr is None:
            raise OperationError("isolated candidate worker streams are unavailable")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        deadline = started + ISOLATED_WORKER_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            _drain_selector(selector, captures, timeout=min(0.25, deadline - time.monotonic()))
            ready = _ready_record(bytes(captures["stdout"]), contract)
            if ready is not None:
                break
            if process.poll() is not None and not selector.get_map():
                break
        if ready is None:
            raise OperationError("isolated candidate worker readiness timed out")
    except OperationError:
        elapsed = max(0, int((time.monotonic() - started) * 1000))
        diagnostic = {
            "schema_version": ISOLATED_WORKER_OBSERVATION_SCHEMA,
            "plan_sha256": values["ULLM_AQ4_RUNTIME_HARDENING_PLAN_SHA256"],
            "operation_epoch": values["ULLM_AQ4_RUNTIME_HARDENING_EPOCH"],
            "candidate_manifest_sha256": values["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_SHA256"],
            "stage": ISOLATED_PREFLIGHT_STAGE,
            "checked_at": _utc_timestamp(),
            "status": "failed",
            "cause": "worker_not_ready",
            "worker": None,
            "operation": {
                "argv_sha256": hashlib.sha256("\0".join(argv).encode("utf-8")).hexdigest(),
                "stdout_sha256": hashlib.sha256(captures["stdout"]).hexdigest(),
                "stderr_sha256": hashlib.sha256(captures["stderr"]).hexdigest(),
                "stdout_bytes": len(captures["stdout"]),
                "stderr_bytes": len(captures["stderr"]),
                "returncode": returncode,
            },
            "timing": {
                "timeout_seconds": int(ISOLATED_WORKER_TIMEOUT_SECONDS),
                "ready_after_milliseconds": None,
                "elapsed_milliseconds": elapsed,
            },
            "cleanup": {"terminated": False, "returncode": returncode},
            "production_activation_performed": False,
        }
        raise IsolatedPreflightError(diagnostic) from None
    finally:
        if process is not None and selector is not None:
            returncode = _stop_isolated_worker(process, selector, captures)
            selector.close()
    assert ready is not None
    elapsed = max(0, int((time.monotonic() - started) * 1000))
    return {
        "schema_version": ISOLATED_WORKER_OBSERVATION_SCHEMA,
        "plan_sha256": values["ULLM_AQ4_RUNTIME_HARDENING_PLAN_SHA256"],
        "operation_epoch": values["ULLM_AQ4_RUNTIME_HARDENING_EPOCH"],
        "candidate_manifest_sha256": values["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_SHA256"],
        "stage": ISOLATED_PREFLIGHT_STAGE,
        "checked_at": _utc_timestamp(),
        "status": "passed",
        "cause": None,
        "worker": {
            "model_id": ready["model"],
            "package_manifest_sha256": ready["package_manifest_sha256"],
            "device": ready["device"],
            "execution_profile": ready["execution_profile"],
        },
        "operation": {
            "argv_sha256": hashlib.sha256("\0".join(argv).encode("utf-8")).hexdigest(),
            "stdout_sha256": hashlib.sha256(captures["stdout"]).hexdigest(),
            "stderr_sha256": hashlib.sha256(captures["stderr"]).hexdigest(),
            "stdout_bytes": len(captures["stdout"]),
            "stderr_bytes": len(captures["stderr"]),
            "returncode": returncode,
        },
        "timing": {
            "timeout_seconds": int(ISOLATED_WORKER_TIMEOUT_SECONDS),
            "ready_after_milliseconds": elapsed,
            "elapsed_milliseconds": elapsed,
        },
        "cleanup": {"terminated": True, "returncode": returncode},
        "production_activation_performed": False,
    }


def reconcile(values: dict[str, str]) -> dict[str, Any]:
    stage = values["ULLM_AQ4_RUNTIME_HARDENING_STAGE"]
    if stage not in RECONCILE_STAGES:
        raise OperationError("reconciliation stage differs")
    _run([SYSTEMCTL, "restart", SERVICE], timeout=90)
    return wait_for_readiness(values)


def observe(values: dict[str, str]) -> dict[str, Any]:
    stage = values["ULLM_AQ4_RUNTIME_HARDENING_STAGE"]
    if stage not in LIVE_STAGES:
        raise OperationError("observation stage differs")
    return wait_for_readiness(values)


def main() -> int:
    try:
        if os.geteuid() != 0 or len(sys.argv) != 2:
            raise OperationError("operation invocation is invalid")
        values = _require_environment()
        action = sys.argv[1]
        if action == "observe":
            document = observe(values)
        elif action == "reconcile":
            document = reconcile(values)
        elif action == "isolated-preflight":
            document = isolated_candidate_preflight(values)
        else:
            raise OperationError("operation action differs")
        print(_canonical(document))
    except (ReadinessError, IsolatedPreflightError) as error:
        print(_canonical(error.diagnostic))
        print("AQ4 hardening operation failed", file=sys.stderr)
        return 1
    except Exception:
        print("AQ4 hardening operation failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
