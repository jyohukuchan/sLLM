#!/usr/bin/env python3
"""Sealed operation payload for the AQ4 runtime-hardening activation route.

The ELF launcher seals this source's exact bytes before invoking it with
``/usr/bin/python3 -I -S -B``.  It exposes only two actions:

* ``reconcile``: restart ``ullm-openai.service`` and wait for active/running;
* ``observe``: emit the strict, secret-free live-observation document consumed
  by ``aq4_runtime_hardening_activation.py``.

It is limited to the designated AQ4 gateway service.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SERVICE = "ullm-openai.service"
DOCKER = "/usr/bin/docker"
SYSTEMCTL = "/usr/bin/systemctl"
GATEWAY_CONTAINER = "open-webui"
GATEWAY_URL = "http://172.20.0.1:8000"
OPENWEBUI_HOST = "127.0.0.1"
OPENWEBUI_PORT = 3000
GATEWAY_API_KEY = Path("/etc/ullm/openai-api-key")
OPENWEBUI_SESSION = Path("/run/ullm-campaign-secrets/openwebui-session.jwt")
MODEL_ID = "ullm-qwen3.5-9b-aq4"
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
MAX_RESPONSE_BYTES = 1_048_576


class OperationError(RuntimeError):
    """The sealed operation could not establish the required live facts."""


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


def _run(argv: list[str], *, input_value: bytes | None = None, timeout: float) -> subprocess.CompletedProcess[bytes]:
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


def _docker_gateway_get(path: str, authorization: bytearray | None) -> tuple[int, bytes]:
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
                "15",
                "--config",
                "-",
                "--write-out",
                "\n%{http_code}",
            ],
            input_value=bytes(config),
            timeout=30,
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


def _openwebui_get(path: str, authorization: bytearray | None) -> tuple[int, bytes]:
    headers = {} if authorization is None else {"Authorization": "Bearer " + authorization.decode("ascii")}
    connection = http.client.HTTPConnection(OPENWEBUI_HOST, OPENWEBUI_PORT, timeout=15)
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


def _read_active(values: dict[str, str]) -> tuple[dict[str, Any], bytes]:
    raw = _stable_read(Path(values["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_MANIFEST"]), maximum=4 * 1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != values["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_SHA256"]:
        raise OperationError("active manifest hash differs")
    try:
        manifest = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OperationError("active manifest JSON is invalid") from error
    if not isinstance(manifest, dict):
        raise OperationError("active manifest root is invalid")
    return manifest, raw


def observe(values: dict[str, str]) -> None:
    stage = values["ULLM_AQ4_RUNTIME_HARDENING_STAGE"]
    if stage not in {"candidate_live_proof", "rollback_live_proof"}:
        raise OperationError("observation stage differs")
    manifest, raw = _read_active(values)
    public = manifest.get("public")
    worker = manifest.get("worker")
    if not isinstance(public, dict) or not isinstance(worker, dict):
        raise OperationError("active manifest identity is invalid")
    model_id = public.get("id")
    worker_path = worker.get("binary")
    worker_hash = worker.get("binary_sha256")
    if (
        model_id != MODEL_ID
        or not isinstance(worker_path, str)
        or not Path(worker_path).is_absolute()
        or not isinstance(worker_hash, str)
        or HASH_RE.fullmatch(worker_hash) is None
        or _sha256_path(Path(worker_path)) != worker_hash
    ):
        raise OperationError("active manifest worker identity differs")
    active_state, sub_state, pid = _service_identity()
    before_process = _process_identity(pid)
    gateway_key = _read_secret(GATEWAY_API_KEY)
    session = _read_secret(OPENWEBUI_SESSION)
    try:
        gateway_health, _ = _docker_gateway_get("/healthz", None)
        gateway_ready, _ = _docker_gateway_get("/readyz", None)
        gateway_models, gateway_models_body = _docker_gateway_get("/v1/models", gateway_key)
        openwebui_health, _ = _openwebui_get("/health", None)
        openwebui_models, openwebui_models_body = _openwebui_get("/api/models", session)
    finally:
        gateway_key[:] = b"\x00" * len(gateway_key)
        session[:] = b"\x00" * len(session)
    after_active, after_sub, after_pid = _service_identity()
    after_process = _process_identity(after_pid)
    final_manifest, final_raw = _read_active(values)
    endpoints = {
        "gateway_health": gateway_health == 200,
        "gateway_ready": gateway_ready == 200,
        "gateway_models": gateway_models == 200 and _model_ids(gateway_models_body) == [MODEL_ID],
        "openwebui_health": openwebui_health == 200,
        "openwebui_models": openwebui_models == 200 and _model_ids(openwebui_models_body) == [MODEL_ID],
    }
    if (
        before_process != after_process
        or pid != after_pid
        or active_state != after_active
        or sub_state != after_sub
        or raw != final_raw
        or manifest != final_manifest
        or not all(endpoints.values())
    ):
        raise OperationError("live observation does not meet the AQ4 proof contract")
    document = {
        "schema_version": "ullm.aq4_runtime_hardening_live_observation.v1",
        "plan_sha256": values["ULLM_AQ4_RUNTIME_HARDENING_PLAN_SHA256"],
        "operation_epoch": values["ULLM_AQ4_RUNTIME_HARDENING_EPOCH"],
        "active_manifest_sha256": values["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_SHA256"],
        "model_id": model_id,
        "worker_binary_path": worker_path,
        "worker_binary_sha256": worker_hash,
        "systemd": {"unit": SERVICE, "active_state": active_state, "sub_state": sub_state},
        "process": before_process,
        "endpoints": endpoints,
    }
    print(json.dumps(document, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True))


def reconcile(values: dict[str, str]) -> None:
    stage = values["ULLM_AQ4_RUNTIME_HARDENING_STAGE"]
    if stage not in {"candidate_reconcile", "rollback_reconcile"}:
        raise OperationError("reconciliation stage differs")
    _run([SYSTEMCTL, "restart", SERVICE], timeout=600)
    for _ in range(120):
        try:
            active, sub, _pid = _service_identity()
        except OperationError:
            active = sub = ""
        if active == "active" and sub == "running":
            return
        time.sleep(5)
    raise OperationError("gateway did not reconcile")


def main() -> int:
    try:
        if os.geteuid() != 0 or len(sys.argv) != 2:
            raise OperationError("operation invocation is invalid")
        values = _require_environment()
        if sys.argv[1] == "observe":
            observe(values)
        elif sys.argv[1] == "reconcile":
            reconcile(values)
        else:
            raise OperationError("operation action differs")
    except Exception:
        print("AQ4 hardening operation failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
