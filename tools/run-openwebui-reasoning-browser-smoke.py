#!/usr/bin/env python3
"""Run the hash-only v2 OpenWebUI reasoning browser smoke safely.

The browser script writes only its bounded JSON evidence to stdout.  This
runner executes it in an immutable Playwright container, binds the expected
model identities explicitly, validates the result, and publishes it
atomically.  Prompt, response, token, and credential contents are never
written by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import select
import stat
import subprocess
import sys
import tempfile
import time
import uuid
import socket
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCRIPT = ROOT / "deploy/openwebui/browser-reasoning-smoke.cjs"
VALIDATOR_PATH = ROOT / "tools/validate-openwebui-reasoning-browser-smoke.py"
SERVED_MODEL_VALIDATOR_PATH = ROOT / "tools/validate-served-model.py"
MAX_TOKEN_FILE_BYTES = 65_536
MAX_SCRIPT_BYTES = 1 * 1024 * 1024
MAX_EVIDENCE_BYTES = 1 * 1024 * 1024
IMAGE_RE = re.compile(
    r"(?:[A-Za-z0-9][A-Za-z0-9._/:+-]*@)?sha256:[0-9a-f]{64}\Z"
)
CONTAINER_USER_RE = re.compile(r"[0-9]{1,5}:[0-9]{1,5}\Z")
_VALIDATOR_MODULE_NAME = "_ullm_openwebui_reasoning_browser_validator"
_SERVED_MODEL_MODULE_NAME = "_ullm_reasoning_browser_served_model_validator"
TARGET_GPU_INDEX = "1"


class SmokeError(RuntimeError):
    """Raised when browser smoke evidence cannot be safely published."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_regular(path: Path, label: str, maximum: int) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SmokeError(f"{label} is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SmokeError(f"{label} is not a regular non-symlink file")
    if metadata.st_size <= 0 or metadata.st_size > maximum:
        raise SmokeError(f"{label} exceeds its size bound")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SmokeError(f"{label} cannot be read") from error
    if len(raw) != metadata.st_size or len(raw) > maximum:
        raise SmokeError(f"{label} changed while being read")
    return raw


def _validate_image(value: str) -> str:
    if IMAGE_RE.fullmatch(value) is None:
        raise SmokeError("browser image must be an immutable Docker SHA-256 identity")
    return value


def _validate_container_user(value: str) -> tuple[int, int]:
    if CONTAINER_USER_RE.fullmatch(value) is None:
        raise SmokeError("browser container user must be UID:GID")
    uid_text, gid_text = value.split(":")
    uid, gid = int(uid_text), int(gid_text)
    if uid > 65535 or gid > 65535:
        raise SmokeError("browser container UID/GID is out of range")
    return uid, gid


def _validate_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError as error:
        raise SmokeError("OpenWebUI URL is invalid") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise SmokeError("OpenWebUI URL must be a credential-free HTTP origin")
    return urlunsplit((parsed.scheme, parsed.netloc, "/", "", ""))


def _validate_public_text(value: str, label: str) -> str:
    if not value or len(value.encode("utf-8")) > 65_536:
        raise SmokeError(f"{label} is empty or exceeds its size bound")
    if any(character in value for character in "\r\n\0"):
        raise SmokeError(f"{label} contains a forbidden control character")
    return value


def _load_validator() -> ModuleType:
    existing = sys.modules.get(_VALIDATOR_MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        _VALIDATOR_MODULE_NAME, VALIDATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise SmokeError("browser evidence validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_VALIDATOR_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_VALIDATOR_MODULE_NAME, None)
        raise
    return module


def _load_served_model_validator() -> ModuleType:
    existing = sys.modules.get(_SERVED_MODEL_MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        _SERVED_MODEL_MODULE_NAME, SERVED_MODEL_VALIDATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise SmokeError("served-model validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SERVED_MODEL_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_SERVED_MODEL_MODULE_NAME, None)
        raise
    return module


def _strict_json(raw: bytes) -> dict[str, Any]:
    def without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SmokeError("browser evidence contains duplicate fields")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=without_duplicates)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SmokeError("browser evidence is not strict JSON") from error
    if not isinstance(value, dict):
        raise SmokeError("browser evidence root is not an object")
    return value


def _provider_hashes(document: dict[str, Any]) -> list[str]:
    requests = document.get("provider_requests")
    if not isinstance(requests, list) or len(requests) != 4:
        raise SmokeError("v2 browser evidence does not contain four provider requests")
    hashes: list[str] = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict) or not isinstance(
            request.get("model_id_sha256"), str
        ):
            raise SmokeError(f"provider request {index} has no model identity")
        hashes.append(request["model_id_sha256"])
    return hashes


def _validate_manifest_identity(manifest: Path, model_id: str) -> dict[str, Any]:
    raw = _read_regular(manifest, "served-model manifest", 1_048_576)
    document = _strict_json(raw)
    if document.get("schema_version") != "ullm.served_model.v2":
        raise SmokeError("served-model manifest is not v2")
    public = document.get("public")
    worker = document.get("worker")
    if not isinstance(public, dict) or public.get("id") != model_id:
        raise SmokeError("served-model manifest model ID differs")
    if not isinstance(worker, dict) or worker.get("protocol") != "ullm.worker.v2":
        raise SmokeError("served-model manifest worker protocol is not v2")
    if "reasoning" not in document:
        raise SmokeError("served-model manifest has no reasoning dialect")
    try:
        summary = _load_served_model_validator().validation_summary(manifest)
    except Exception as error:
        raise SmokeError("served-model manifest failed validation") from error
    if summary.get("model_id") != model_id or summary.get("worker", {}).get(
        "protocol"
    ) != "ullm.worker.v2":
        raise SmokeError("served-model validator identity differs")
    return summary


def _bind_model_identity(
    document: dict[str, Any],
    *,
    model_id: str,
    switch_model_id: str,
) -> None:
    expected = _sha256(model_id.encode("utf-8"))
    switch_expected = _sha256(switch_model_id.encode("utf-8"))
    if document.get("schema_version") != "ullm.openwebui.reasoning_browser_smoke.v2":
        raise SmokeError("browser evidence is not the v2 schema")
    if document.get("model_id_sha256") != expected:
        raise SmokeError("browser evidence primary model identity differs")
    if document.get("provider_switch_model_id_sha256") != switch_expected:
        raise SmokeError("browser evidence switch model identity differs")
    if document.get("provider_return_model_id_sha256") != expected:
        raise SmokeError("browser evidence return model identity differs")
    hashes = _provider_hashes(document)
    if hashes[:2] != [expected, expected] or hashes[2:] != [switch_expected, expected]:
        raise SmokeError("browser provider request model sequence differs")


def _atomic_publish(output: Path, raw: bytes) -> None:
    if output.exists() or output.is_symlink():
        raise SmokeError("browser evidence output already exists or is a symlink")
    try:
        parent = output.parent.resolve(strict=True)
        metadata = parent.lstat()
    except OSError as error:
        raise SmokeError("browser evidence output directory is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SmokeError("browser evidence output parent is not a real directory")
    incomplete = parent / f".{output.name}.incomplete-{uuid.uuid4().hex}"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            incomplete,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        with os.fdopen(descriptor, "wb", buffering=0) as destination:
            descriptor = None
            destination.write(raw)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(incomplete, output)
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as error:
        raise SmokeError("browser evidence could not be published atomically") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        incomplete.unlink(missing_ok=True)


def _stop_container(docker: str, name: str) -> None:
    try:
        subprocess.run(
            [docker, "rm", "--force", name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def _service_state(service: str, systemctl: str) -> tuple[str, str]:
    try:
        result = subprocess.run(
            [systemctl, "show", service, "--property=LoadState", "--property=ActiveState", "--value"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SmokeError("service state lookup failed") from error
    states = result.stdout.splitlines()
    if result.returncode != 0 or len(states) != 2:
        raise SmokeError("service state lookup failed")
    return states[0], states[1]


def _service_command(systemctl: str, action: str, service: str) -> None:
    try:
        result = subprocess.run(
            [systemctl, action, service],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SmokeError("service transition failed") from error
    if result.returncode != 0:
        raise SmokeError("service transition failed")


def _target_gpu_processes(rocm_smi: str) -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [rocm_smi, "--showpids", "--json"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SmokeError("GPU ownership preflight failed") from error
    if result.returncode != 0:
        raise SmokeError("GPU ownership preflight failed")
    if not result.stdout.strip():
        return []
    value = _strict_json(result.stdout.encode("utf-8"))
    system = value.get("system")
    if not isinstance(system, dict):
        raise SmokeError("GPU ownership data is missing")
    processes: list[dict[str, Any]] = []
    for pid, description in system.items():
        if not isinstance(pid, str) or not pid.startswith("PID") or not isinstance(description, str):
            raise SmokeError("GPU process identity is malformed")
        fields = [field.strip() for field in description.split(",")]
        if len(fields) < 3:
            raise SmokeError("GPU process description is incomplete")
        try:
            vram = int(fields[2])
        except ValueError as error:
            raise SmokeError("GPU process VRAM is invalid") from error
        visible = {item.strip() for item in fields[1].split(",") if item.strip()}
        if TARGET_GPU_INDEX in visible and vram > 0:
            processes.append({"pid": pid[3:], "process": fields[0], "vram_bytes": vram})
    return processes


def _wait_for_tcp_port(
    docker: str,
    host: str,
    port: int,
    timeout_seconds: float = 120.0,
    readiness_path: str = "/readyz",
) -> None:
    deadline = time.monotonic() + timeout_seconds
    consecutive_successes = 0
    while True:
        try:
            result = subprocess.run(
                [
                    docker,
                    "exec",
                    "open-webui",
                    "python",
                    "-c",
                    (
                        "import urllib.error, urllib.request; "
                        f"url='http://{host}:{port}{readiness_path}'; "
                        "\ntry: "
                        " response=urllib.request.urlopen(url, timeout=2.0); "
                        " raise SystemExit(0 if response.status == 200 else 1)"
                        "\nexcept (urllib.error.HTTPError, urllib.error.URLError): "
                        " raise SystemExit(1)"
                    ),
                ],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
            )
            if result.returncode == 0:
                consecutive_successes += 1
                if consecutive_successes >= 3:
                    return
                time.sleep(0.5)
                continue
            consecutive_successes = 0
            if time.monotonic() >= deadline:
                raise SmokeError("service HTTP port did not become ready")
            time.sleep(0.25)
        except (OSError, subprocess.TimeoutExpired):
            consecutive_successes = 0
            if time.monotonic() >= deadline:
                raise SmokeError("service HTTP port did not become ready")
            time.sleep(0.25)


def _wait_for_gpu_owner(rocm_smi: str, expected: set[str], timeout_seconds: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        processes = _target_gpu_processes(rocm_smi)
        if {process["process"] for process in processes} == expected:
            return
        if time.monotonic() >= deadline:
            raise SmokeError("target R9700 ownership did not reach the expected state")
        time.sleep(0.25)


class _AlternatingServiceCoordinator:
    def __init__(
        self,
        systemctl: str,
        rocm_smi: str,
        ullm_service: str,
        llama_service: str,
        docker: str = "docker",
        ullm_port: int = 8000,
        llama_port: int = 8001,
    ) -> None:
        self.systemctl = systemctl
        self.rocm_smi = rocm_smi
        self.ullm_service = ullm_service
        self.llama_service = llama_service
        self.docker = docker
        self.ullm_port = ullm_port
        self.llama_port = llama_port
        self.owner = "ullm"

    def preflight(self, worker_binary_sha256: str) -> None:
        if _service_state(self.ullm_service, self.systemctl)[1] != "active":
            raise SmokeError("uLLM service must be active before alternating browser smoke")
        if _service_state(self.llama_service, self.systemctl)[1] not in {"inactive", "failed"}:
            raise SmokeError("llama.cpp service must be inactive before alternating browser smoke")
        processes = _target_gpu_processes(self.rocm_smi)
        if {process["process"] for process in processes} != {"ullm-aq4-worker"}:
            raise SmokeError("target R9700 is not exclusively owned by uLLM")
        for process in processes:
            try:
                target = Path(os.readlink(f"/proc/{process['pid']}/exe"))
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
            except OSError as error:
                raise SmokeError("uLLM worker executable cannot be inspected") from error
            if digest != worker_binary_sha256:
                raise SmokeError("uLLM worker executable differs from the v2 manifest")

    def transition(self, phase: str) -> None:
        if phase == "before-switch":
            self.owner = "transition"
            _service_command(self.systemctl, "stop", self.ullm_service)
            _wait_for_gpu_owner(self.rocm_smi, set())
            _service_command(self.systemctl, "start", self.llama_service)
            _wait_for_gpu_owner(self.rocm_smi, {"llama-server"})
            _wait_for_tcp_port(
                self.docker,
                "172.20.0.1",
                self.llama_port,
                readiness_path="/health",
            )
            self.owner = "llama"
            return
        if phase == "before-return":
            self.owner = "transition"
            _service_command(self.systemctl, "stop", self.llama_service)
            _wait_for_gpu_owner(self.rocm_smi, set())
            _service_command(self.systemctl, "start", self.ullm_service)
            _wait_for_gpu_owner(self.rocm_smi, {"ullm-aq4-worker"})
            _wait_for_tcp_port(
                self.docker,
                "172.20.0.1",
                self.ullm_port,
                readiness_path="/readyz",
            )
            self.owner = "ullm"
            return
        raise SmokeError("unknown browser service transition phase")

    def recover(self) -> None:
        if self.owner in {"llama", "transition"}:
            try:
                _service_command(self.systemctl, "stop", self.llama_service)
                _wait_for_gpu_owner(self.rocm_smi, set(), timeout_seconds=30.0)
                _service_command(self.systemctl, "start", self.ullm_service)
            except SmokeError:
                pass
            self.owner = "ullm"


def _wait_for_browser_with_transitions(
    process: subprocess.Popen[bytes],
    server: socket.socket,
    coordinator: _AlternatingServiceCoordinator,
    deadline: float,
) -> int:
    while process.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, 0)
        readable, _, _ = select.select([server], [], [], min(remaining, 0.5))
        if not readable:
            continue
        connection, _ = server.accept()
        try:
            phase = connection.recv(128).decode("ascii").strip()
            try:
                coordinator.transition(phase)
            except SmokeError:
                connection.sendall(b"abort\n")
                raise
            connection.sendall(b"continue\n")
        finally:
            connection.close()
    return process.wait(timeout=max(0.1, deadline - time.monotonic()))


def execute(
    *,
    output: Path,
    manifest: Path,
    token_file: Path,
    browser_image: str,
    openwebui_url: str,
    model_id: str,
    model_name: str,
    switch_model_id: str,
    switch_model_name: str,
    browser_script: Path = DEFAULT_SCRIPT,
    docker: str = "docker",
    timeout_seconds: float = 900.0,
    alternate_r9700_services: bool = False,
    ullm_service: str = "ullm-openai.service",
    llama_service: str = "llama-qwen35-udq4.service",
    systemctl: str = "systemctl",
    rocm_smi: str = "rocm-smi",
    browser_user: str | None = None,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise SmokeError("browser evidence output already exists or is a symlink")
    manifest_summary = _validate_manifest_identity(manifest, model_id)
    token = _read_regular(token_file, "OpenWebUI token file", MAX_TOKEN_FILE_BYTES)
    script = _read_regular(browser_script, "browser smoke script", MAX_SCRIPT_BYTES)
    if not token:
        raise SmokeError("OpenWebUI token file is empty")
    url = _validate_url(openwebui_url)
    image = _validate_image(browser_image)
    model_id = _validate_public_text(model_id, "model ID")
    model_name = _validate_public_text(model_name, "model name")
    switch_model_id = _validate_public_text(switch_model_id, "switch model ID")
    switch_model_name = _validate_public_text(switch_model_name, "switch model name")
    if model_id == switch_model_id:
        raise SmokeError("switch model ID must differ from the candidate model ID")
    container_user = browser_user or f"{os.geteuid()}:{os.getegid()}"
    container_uid, container_gid = _validate_container_user(container_user)
    del token, script

    coordinator = (
        _AlternatingServiceCoordinator(
            systemctl, rocm_smi, ullm_service, llama_service, docker=docker
        )
        if alternate_r9700_services
        else None
    )
    control_directory: tempfile.TemporaryDirectory[str] | None = None
    control_server: socket.socket | None = None
    if coordinator is not None:
        coordinator.preflight(manifest_summary["worker"]["binary_sha256"])
        control_directory = tempfile.TemporaryDirectory(prefix="ullm-browser-transition-")
        control_path = Path(control_directory.name) / "transition.sock"
        control_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        control_server.bind(os.fspath(control_path))
        os.chmod(control_path, 0o600)
        if os.geteuid() == 0:
            os.chown(control_directory.name, container_uid, container_gid)
            os.chown(control_path, container_uid, container_gid)
        control_server.listen(1)

    container_name = f"ullm-reasoning-browser-{uuid.uuid4().hex[:16]}"
    script_path = browser_script.resolve(strict=True)
    token_path = token_file.resolve(strict=True)
    command = [
        docker,
        "run",
        "--rm",
        "--network=host",
        f"--name={container_name}",
        f"--user={container_user}",
        "--pids-limit=256",
        "--security-opt=no-new-privileges",
        "--mount",
        f"type=bind,src={script_path},dst=/run/ullm-browser-reasoning-smoke.cjs,readonly",
        "--mount",
        f"type=bind,src={token_path},dst=/run/secrets/openwebui-token,readonly",
        "--env",
        f"OPENWEBUI_URL={url}",
        "--env",
        "OPENWEBUI_TOKEN_FILE=/run/secrets/openwebui-token",
        "--env",
        "NODE_PATH=/usr/src/app/node_modules",
        "--env",
        f"ULLM_MODEL_ID={model_id}",
        "--env",
        f"ULLM_MODEL_NAME={model_name}",
        "--env",
        f"OPENWEBUI_SWITCH_MODEL_ID={switch_model_id}",
        "--env",
        f"OPENWEBUI_SWITCH_MODEL_NAME={switch_model_name}",
    ]
    if control_server is not None:
        command.extend(
            [
                "--mount",
                f"type=bind,src={control_directory.name},dst=/run/ullm-transition,readonly",
                "--env",
                "OPENWEBUI_TRANSITION_SOCKET=/run/ullm-transition/transition.sock",
            ]
        )
    command.extend(
        [
        "--entrypoint",
        "node",
        image,
        "/run/ullm-browser-reasoning-smoke.cjs",
        ]
    )
    deadline = time.monotonic() + timeout_seconds
    try:
        with tempfile.TemporaryFile() as stdout:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                if control_server is None:
                    return_code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
                else:
                    return_code = _wait_for_browser_with_transitions(
                        process,
                        control_server,
                        coordinator,
                        deadline,
                    )
            except subprocess.TimeoutExpired as error:
                process.kill()
                process.wait(timeout=15.0)
                _stop_container(docker, container_name)
                raise SmokeError("browser smoke timed out") from error
            except BaseException:
                process.kill()
                process.wait(timeout=15.0)
                _stop_container(docker, container_name)
                raise
            if return_code != 0:
                error_output = process.stderr.read(4097) if process.stderr is not None else b""
                detail = error_output.decode("utf-8", errors="replace").strip()
                if len(detail) > 4096:
                    detail = detail[:4096]
                raise SmokeError(
                    "browser smoke container failed"
                    + (f": {detail}" if detail else "")
                )
            stdout.seek(0)
            raw = stdout.read(MAX_EVIDENCE_BYTES + 1)
    except Exception:
        if coordinator is not None:
            coordinator.recover()
        raise
    finally:
        if control_server is not None:
            control_server.close()
        if control_directory is not None:
            control_directory.cleanup()
    if not raw or len(raw) > MAX_EVIDENCE_BYTES:
        raise SmokeError("browser smoke output exceeds its size bound")
    document = _strict_json(raw.strip())
    _bind_model_identity(
        document,
        model_id=model_id,
        switch_model_id=switch_model_id,
    )
    temporary = output.parent / f".{output.name}.validate-{uuid.uuid4().hex}"
    try:
        _atomic_publish(temporary, raw.strip() + b"\n")
        report = _load_validator().validate(temporary)
        if report.get("gate_eligible") is not True:
            raise SmokeError("browser evidence is not gate eligible")
        temporary.unlink(missing_ok=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    _atomic_publish(output, raw.strip() + b"\n")
    return {
        "schema_version": document["schema_version"],
        "output": os.fspath(output.resolve()),
        "model_id_sha256": document["model_id_sha256"],
        "provider_request_count": document["provider_request_count"],
        "manifest_sha256": manifest_summary["manifest_sha256"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--browser-image", required=True)
    parser.add_argument("--openwebui-url", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--switch-model-id", required=True)
    parser.add_argument("--switch-model-name", required=True)
    parser.add_argument("--browser-script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--alternate-r9700-services", action="store_true")
    parser.add_argument("--ullm-service", default="ullm-openai.service")
    parser.add_argument("--llama-service", default="llama-qwen35-udq4.service")
    parser.add_argument("--systemctl", default="systemctl")
    parser.add_argument("--rocm-smi", default="rocm-smi")
    parser.add_argument("--browser-user")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = execute(
            output=args.output,
            manifest=args.manifest,
            token_file=args.token_file,
            browser_image=args.browser_image,
            openwebui_url=args.openwebui_url,
            model_id=args.model_id,
            model_name=args.model_name,
            switch_model_id=args.switch_model_id,
            switch_model_name=args.switch_model_name,
            browser_script=args.browser_script,
            docker=args.docker,
            timeout_seconds=args.timeout_seconds,
            alternate_r9700_services=args.alternate_r9700_services,
            ullm_service=args.ullm_service,
            llama_service=args.llama_service,
            systemctl=args.systemctl,
            rocm_smi=args.rocm_smi,
            browser_user=args.browser_user,
        )
    except Exception as error:
        print(f"OpenWebUI reasoning browser smoke failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
