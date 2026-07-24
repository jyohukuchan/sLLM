#!/usr/bin/env python3
"""Fail closed unless the running OpenWebUI container has the fixed image."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOOLS = Path(__file__).resolve().parent
if os.fspath(TOOLS) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS))

import served_model_campaign_authorization as authorization


DOCKER = Path("/usr/bin/docker")
MAX_INSPECT_BYTES = 4_096
INSPECT_TIMEOUT_SECONDS = 30.0
INSPECT_FORMAT = (
    "{{json .Id}}\n"
    "{{json .Image}}\n"
    "{{json .Config.Image}}\n"
    "{{json .Name}}\n"
    "{{json .State.Running}}\n"
    "{{json .State.Pid}}\n"
    "{{json .State.StartedAt}}"
)
EXPECTED_IMAGE_ID = authorization.FIXED_OPENWEBUI_IMAGE.rsplit("@", 1)[1]
EXPECTED_CONFIG_IMAGE = authorization.FIXED_OPENWEBUI_CONFIG_IMAGE
EXPECTED_CONTAINER_NAME = f"/{authorization.FIXED_OPENWEBUI_CONTAINER_NAME}"


class VerificationError(RuntimeError):
    """The live container image identity could not be proven."""


@dataclass(frozen=True, slots=True)
class ContainerIdentity:
    container_id: str
    image_id: str
    config_image: str
    name: str
    running: bool
    pid: int
    started_at: str


CommandRunner = Callable[..., subprocess.CompletedProcess[bytes]]


def _canonical_scalar(line: bytes, label: str) -> Any:
    try:
        text = line.decode("ascii")
        value = json.loads(text)
        canonical = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise VerificationError(f"{label} is not canonical JSON") from error
    if canonical != text:
        raise VerificationError(f"{label} is not canonical JSON")
    return value


def _parse_inspect_stdout(raw: bytes) -> ContainerIdentity:
    if (
        not isinstance(raw, bytes)
        or not raw
        or len(raw) > MAX_INSPECT_BYTES
        or not raw.endswith(b"\n")
        or b"\r" in raw
    ):
        raise VerificationError("Docker inspect output framing is invalid")
    lines = raw[:-1].split(b"\n")
    if len(lines) != 7 or any(not line for line in lines):
        raise VerificationError("Docker inspect output shape differs")
    container_id = _canonical_scalar(lines[0], "container ID")
    image_id = _canonical_scalar(lines[1], "container image ID")
    config_image = _canonical_scalar(lines[2], "container config image")
    name = _canonical_scalar(lines[3], "container name")
    running = _canonical_scalar(lines[4], "container running state")
    pid = _canonical_scalar(lines[5], "container PID")
    started_at = _canonical_scalar(lines[6], "container start time")
    if (
        not isinstance(container_id, str)
        or len(container_id) != 64
        or any(character not in "0123456789abcdef" for character in container_id)
        or
        not isinstance(image_id, str)
        or not isinstance(config_image, str)
        or not isinstance(name, str)
        or type(running) is not bool
        or type(pid) is not int
        or pid <= 0
        or not isinstance(started_at, str)
        or not started_at
        or len(started_at.encode("ascii", errors="ignore")) != len(started_at)
        or len(started_at) > 128
    ):
        raise VerificationError("Docker inspect output types differ")
    return ContainerIdentity(
        container_id,
        image_id,
        config_image,
        name,
        running,
        pid,
        started_at,
    )


def verify_container_image(
    *,
    docker: Path = DOCKER,
    runner: CommandRunner = subprocess.run,
) -> ContainerIdentity:
    if (
        not isinstance(docker, Path)
        or not docker.is_absolute()
        or Path(os.path.abspath(docker)) != docker
        or "\x00" in os.fspath(docker)
    ):
        raise VerificationError("Docker executable path is invalid")
    command = [
        os.fspath(docker),
        "inspect",
        "--type",
        "container",
        "--format",
        INSPECT_FORMAT,
        authorization.FIXED_OPENWEBUI_CONTAINER_NAME,
    ]
    try:
        completed = runner(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=INSPECT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise VerificationError("Docker inspect failed") from error
    if type(completed.returncode) is not int or completed.returncode != 0:
        raise VerificationError("Docker inspect failed")
    identity = _parse_inspect_stdout(completed.stdout)
    if identity.image_id != EXPECTED_IMAGE_ID:
        raise VerificationError("running OpenWebUI image ID differs")
    if identity.config_image != EXPECTED_CONFIG_IMAGE:
        raise VerificationError("OpenWebUI container config image differs")
    if identity.name != EXPECTED_CONTAINER_NAME:
        raise VerificationError("OpenWebUI container name differs")
    if identity.running is not True:
        raise VerificationError("OpenWebUI container is not running")
    return identity


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", type=Path, default=DOCKER)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        identity = verify_container_image(docker=args.docker)
    except VerificationError:
        print("OpenWebUI container image verification failed closed", file=sys.stderr)
        return 1
    report = {
        "schema_version": "ullm.openwebui_container_image_verification.v1",
        "status": "passed",
        "container": identity.name,
        "container_id": identity.container_id,
        "image_id": identity.image_id,
        "config_image": identity.config_image,
        "running": identity.running,
        "pid": identity.pid,
        "started_at": identity.started_at,
    }
    print(
        json.dumps(
            report,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
