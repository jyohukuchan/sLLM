#!/usr/bin/env python3
"""Secret-free structured proof that the live route was restored to AQ4_0."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOOLS = Path(__file__).resolve().parent
if os.fspath(TOOLS) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS))

SCHEMA_VERSION = "ullm.served_model.v2_cross_model_restoration_proof.v1"
AQ4_MODEL_ID = "ullm-qwen3.5-9b-aq4"
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
BOOT_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
TIMESTAMP_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z"
)
MAX_RESPONSE_BYTES = 1_048_576
PROOF_FIELDS = {
    "schema_version",
    "authorization_sha256",
    "claim_sha256",
    "captured_at",
    "active_manifest",
    "service",
    "gateway",
    "worker",
    "endpoints",
    "epoch_stable",
    "passed",
}
ACTIVE_FIELDS = {"path", "expected_sha256", "observed_sha256", "bytes_equal"}
SERVICE_FIELDS = {
    "unit",
    "active_state",
    "sub_state",
    "boot_id",
    "n_restarts",
}
PROCESS_FIELDS = {
    "pid",
    "ppid",
    "starttime_ticks",
    "executable_sha256",
}
ENDPOINT_FIELDS = {
    "gateway_healthz",
    "gateway_readyz",
    "gateway_models",
    "openwebui_health",
    "openwebui_models",
}
STATUS_FIELDS = {"status"}
MODELS_FIELDS = {"status", "model_ids"}


class RestorationProofError(RuntimeError):
    """The live AQ4 restoration proof is absent, unsafe, or inconsistent."""


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise RestorationProofError(f"{label} fields differ")
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise RestorationProofError(f"{label} is not a SHA-256")
    return value


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum or value > (1 << 63) - 1:
        raise RestorationProofError(f"{label} is invalid")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise RestorationProofError(
            "restoration proof is not canonicalizable"
        ) from error


def validate_proof(
    document: dict[str, Any],
    *,
    authorization_sha256: str,
    claim_sha256: str,
    active_manifest_path: Path,
    expected_manifest_sha256: str,
    expected_worker_sha256: str,
    service_unit: str,
) -> None:
    """Strictly bind a sanitized live proof to one consumed AQ4 restoration."""

    _exact(document, PROOF_FIELDS, "restoration proof")
    if (
        document["schema_version"] != SCHEMA_VERSION
        or document["authorization_sha256"] != authorization_sha256
        or document["claim_sha256"] != claim_sha256
        or not isinstance(document["captured_at"], str)
        or TIMESTAMP_RE.fullmatch(document["captured_at"]) is None
        or document["epoch_stable"] is not True
        or document["passed"] is not True
    ):
        raise RestorationProofError("restoration proof root identity differs")

    active = _exact(
        document["active_manifest"], ACTIVE_FIELDS, "restoration proof active manifest"
    )
    if (
        active["path"] != os.fspath(active_manifest_path)
        or active["expected_sha256"] != expected_manifest_sha256
        or active["observed_sha256"] != expected_manifest_sha256
        or active["bytes_equal"] is not True
    ):
        raise RestorationProofError("restoration proof active manifest differs")

    service = _exact(document["service"], SERVICE_FIELDS, "restoration proof service")
    if (
        service["unit"] != service_unit
        or service["active_state"] != "active"
        or service["sub_state"] != "running"
        or not isinstance(service["boot_id"], str)
        or BOOT_ID_RE.fullmatch(service["boot_id"]) is None
    ):
        raise RestorationProofError("restoration proof service state differs")
    _positive_int(service["n_restarts"], "restoration proof service restart count", allow_zero=True)

    gateway = _exact(document["gateway"], PROCESS_FIELDS, "restoration proof gateway")
    worker = _exact(document["worker"], PROCESS_FIELDS, "restoration proof worker")
    for label, process in (("gateway", gateway), ("worker", worker)):
        _positive_int(process["pid"], f"restoration proof {label} PID")
        _positive_int(process["starttime_ticks"], f"restoration proof {label} starttime")
        _positive_int(
            process["ppid"],
            f"restoration proof {label} PPID",
            allow_zero=label == "gateway",
        )
        _hash(
            process["executable_sha256"],
            f"restoration proof {label} executable SHA-256",
        )
    if (
        gateway["pid"] == worker["pid"]
        or worker["ppid"] != gateway["pid"]
        or worker["executable_sha256"] != expected_worker_sha256
    ):
        raise RestorationProofError("restoration proof live worker identity differs")

    endpoints = _exact(
        document["endpoints"], ENDPOINT_FIELDS, "restoration proof endpoints"
    )
    for name in ("gateway_healthz", "gateway_readyz", "openwebui_health"):
        endpoint = _exact(
            endpoints[name], STATUS_FIELDS, f"restoration proof endpoint {name}"
        )
        if endpoint["status"] != 200:
            raise RestorationProofError(f"restoration proof endpoint {name} failed")
    for name in ("gateway_models", "openwebui_models"):
        endpoint = _exact(
            endpoints[name], MODELS_FIELDS, f"restoration proof endpoint {name}"
        )
        if endpoint["status"] != 200 or endpoint["model_ids"] != [AQ4_MODEL_ID]:
            raise RestorationProofError(f"restoration proof endpoint {name} differs")


def _sha256_file(
    path: Path,
    label: str,
    maximum: int = 512 * 1024 * 1024,
    *,
    follow_proc_exe: bool = False,
) -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW") and not follow_proc_exe:
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RestorationProofError(f"{label} is unavailable") from error
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise RestorationProofError(f"{label} metadata is unsafe")
        total = 0
        while True:
            chunk = os.read(descriptor, min(1 << 20, maximum - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise RestorationProofError(f"{label} exceeds its size bound")
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RestorationProofError(f"{label} changed while being hashed")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _proc_identity(pid: int) -> dict[str, Any]:
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text(
            encoding="ascii"
        ).split()
        if len(fields) < 22:
            raise ValueError
        ppid = int(fields[3])
        starttime = int(fields[21])
        executable = (Path("/proc") / str(pid) / "exe")
        executable_sha256 = _sha256_file(
            executable,
            "live process executable",
            follow_proc_exe=True,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise RestorationProofError("live process identity is unavailable") from error
    return {
        "pid": pid,
        "ppid": ppid,
        "starttime_ticks": starttime,
        "executable_sha256": executable_sha256,
    }


def _service_identity(
    unit: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        completed = runner(
            [
                "/usr/bin/systemctl",
                "show",
                unit,
                "--property=ActiveState",
                "--property=SubState",
                "--property=MainPID",
                "--property=NRestarts",
                "--value",
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15.0,
        )
        values = completed.stdout.splitlines()
        if completed.returncode != 0 or len(values) != 4:
            raise ValueError
        active_state, sub_state, raw_pid, raw_restarts = values
        gateway_pid = int(raw_pid)
        n_restarts = int(raw_restarts)
        children_raw = (
            Path("/proc")
            / str(gateway_pid)
            / "task"
            / str(gateway_pid)
            / "children"
        ).read_text(encoding="ascii")
        children = tuple(int(value) for value in children_raw.split())
        if len(children) != 1:
            raise ValueError
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except (OSError, subprocess.TimeoutExpired, UnicodeError, ValueError) as error:
        raise RestorationProofError("live service identity is unavailable") from error
    service = {
        "unit": unit,
        "active_state": active_state,
        "sub_state": sub_state,
        "boot_id": boot_id,
        "n_restarts": n_restarts,
    }
    return service, _proc_identity(gateway_pid), _proc_identity(children[0])


def _http_json(
    *,
    port: int,
    target: str,
    authorization: bytes | None = None,
) -> tuple[int, Any]:
    headers: dict[str, str] = {}
    if authorization is not None:
        try:
            headers["Authorization"] = authorization.decode("ascii")
        except UnicodeError as error:
            raise RestorationProofError("endpoint credential encoding differs") from error
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=15.0)
    try:
        connection.request("GET", target, headers=headers)
        response = connection.getresponse()
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise RestorationProofError("endpoint response exceeds its size bound")
        try:
            value = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError):
            value = None
        return response.status, value
    except OSError as error:
        raise RestorationProofError("endpoint probe failed") from error
    finally:
        connection.close()


def _model_ids(value: Any) -> list[str]:
    candidates: Any
    if isinstance(value, dict) and isinstance(value.get("data"), list):
        candidates = value["data"]
    elif isinstance(value, dict) and isinstance(value.get("models"), list):
        candidates = value["models"]
    elif isinstance(value, list):
        candidates = value
    else:
        raise RestorationProofError("model endpoint response shape differs")
    ids: set[str] = set()
    for item in candidates:
        if isinstance(item, dict):
            model_id = item.get("id") or item.get("model") or item.get("name")
            if isinstance(model_id, str) and model_id.startswith("ullm-"):
                ids.add(model_id)
    return sorted(ids)


def _read_secret(path: Path, label: str) -> bytearray:
    # Lazy import avoids the authorization -> proof -> active-binding cycle:
    # active binding itself imports campaign authorization.
    from served_model_active_binding import stable_read_regular

    try:
        snapshot = stable_read_regular(
            path,
            label,
            maximum=65_536,
            require_single_link=True,
        )
        raw = bytearray(snapshot.raw)
    except Exception as error:
        raise RestorationProofError(f"{label} is unavailable") from error
    if (
        snapshot.identity.mode & 0o077
        or not raw
    ):
        raise RestorationProofError(f"{label} is unsafe")
    while raw and raw[-1] in b"\r\n":
        raw.pop()
    if not raw:
        raise RestorationProofError(f"{label} is empty")
    return raw


def collect_live_proof(
    *,
    authorization_sha256: str,
    claim_sha256: str,
    active_manifest_path: Path,
    expected_manifest_sha256: str,
    expected_worker_sha256: str,
    service_unit: str,
    api_key_file: Path,
    openwebui_session_token_file: Path,
    manifest_reader: Callable[[Path], bytes],
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    service_reader: Callable[
        [str], tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
    ] = _service_identity,
) -> dict[str, Any]:
    """Capture live facts twice around secret-safe endpoint probes."""

    raw = manifest_reader(active_manifest_path)
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    before_service, before_gateway, before_worker = service_reader(service_unit)
    api_key = _read_secret(api_key_file, "gateway API key")
    session = _read_secret(openwebui_session_token_file, "OpenWebUI session token")
    try:
        health_status, _ = _http_json(port=8000, target="/healthz")
        ready_status, _ = _http_json(port=8000, target="/readyz")
        models_status, models = _http_json(
            port=8000,
            target="/v1/models",
            authorization=b"Bearer " + bytes(api_key),
        )
        openwebui_health_status, _ = _http_json(port=3000, target="/health")
        openwebui_models_status, openwebui_models = _http_json(
            port=3000,
            target="/api/models",
            authorization=b"Bearer " + bytes(session),
        )
    finally:
        api_key[:] = b"\x00" * len(api_key)
        session[:] = b"\x00" * len(session)
    after_service, after_gateway, after_worker = service_reader(service_unit)
    final_raw = manifest_reader(active_manifest_path)
    epoch_stable = (
        before_service == after_service
        and before_gateway == after_gateway
        and before_worker == after_worker
        and raw == final_raw
    )
    captured = now().astimezone(timezone.utc).replace(microsecond=0)
    proof = {
        "schema_version": SCHEMA_VERSION,
        "authorization_sha256": authorization_sha256,
        "claim_sha256": claim_sha256,
        "captured_at": captured.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "active_manifest": {
            "path": os.fspath(active_manifest_path),
            "expected_sha256": expected_manifest_sha256,
            "observed_sha256": observed_sha256,
            "bytes_equal": observed_sha256 == expected_manifest_sha256,
        },
        "service": before_service,
        "gateway": before_gateway,
        "worker": before_worker,
        "endpoints": {
            "gateway_healthz": {"status": health_status},
            "gateway_readyz": {"status": ready_status},
            "gateway_models": {
                "status": models_status,
                "model_ids": _model_ids(models),
            },
            "openwebui_health": {"status": openwebui_health_status},
            "openwebui_models": {
                "status": openwebui_models_status,
                "model_ids": _model_ids(openwebui_models),
            },
        },
        "epoch_stable": epoch_stable,
        "passed": True,
    }
    validate_proof(
        proof,
        authorization_sha256=authorization_sha256,
        claim_sha256=claim_sha256,
        active_manifest_path=active_manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_worker_sha256=expected_worker_sha256,
        service_unit=service_unit,
    )
    return proof
