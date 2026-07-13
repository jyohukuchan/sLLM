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
import stat
import subprocess
import sys
import tempfile
import time
import uuid
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
_VALIDATOR_MODULE_NAME = "_ullm_openwebui_reasoning_browser_validator"
_SERVED_MODEL_MODULE_NAME = "_ullm_reasoning_browser_served_model_validator"


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
    del token, script

    container_name = f"ullm-reasoning-browser-{uuid.uuid4().hex[:16]}"
    script_path = browser_script.resolve(strict=True)
    token_path = token_file.resolve(strict=True)
    command = [
        docker,
        "run",
        "--rm",
        "--network=host",
        f"--name={container_name}",
        f"--user={os.geteuid()}:{os.getegid()}",
        "--pids-limit=256",
        "--security-opt=no-new-privileges",
        "--mount",
        f"type=bind,src={script_path},dst=/run/ullm/browser-reasoning-smoke.cjs,readonly",
        "--mount",
        f"type=bind,src={token_path},dst=/run/secrets/openwebui-token,readonly",
        "--env",
        f"OPENWEBUI_URL={url}",
        "--env",
        "OPENWEBUI_TOKEN_FILE=/run/secrets/openwebui-token",
        "--env",
        f"ULLM_MODEL_ID={model_id}",
        "--env",
        f"ULLM_MODEL_NAME={model_name}",
        "--env",
        f"OPENWEBUI_SWITCH_MODEL_ID={switch_model_id}",
        "--env",
        f"OPENWEBUI_SWITCH_MODEL_NAME={switch_model_name}",
        "--entrypoint",
        "node",
        image,
        "/run/ullm/browser-reasoning-smoke.cjs",
    ]
    deadline = time.monotonic() + timeout_seconds
    with tempfile.TemporaryFile() as stdout:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            remaining = max(0.1, deadline - time.monotonic())
            return_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait(timeout=15.0)
            _stop_container(docker, container_name)
            raise SmokeError("browser smoke timed out") from error
        if return_code != 0:
            raise SmokeError("browser smoke container failed")
        stdout.seek(0)
        raw = stdout.read(MAX_EVIDENCE_BYTES + 1)
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
        )
    except Exception as error:
        print(f"OpenWebUI reasoning browser smoke failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
