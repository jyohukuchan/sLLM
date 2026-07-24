from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
MODULE_PATH = TOOLS / "verify-openwebui-container-image.py"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "test_verify_openwebui_container_image_module",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFIER
SPEC.loader.exec_module(VERIFIER)


def inspect_stdout(
    *,
    container_id: str = "1" * 64,
    image_id: str = VERIFIER.EXPECTED_IMAGE_ID,
    config_image: str = VERIFIER.EXPECTED_CONFIG_IMAGE,
    name: str = VERIFIER.EXPECTED_CONTAINER_NAME,
    running: bool = True,
    pid: int = 1234,
    started_at: str = "2026-07-24T00:00:00.000000000Z",
) -> bytes:
    return (
        "\n".join(
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            for value in (
                container_id,
                image_id,
                config_image,
                name,
                running,
                pid,
                started_at,
            )
        )
        + "\n"
    ).encode("ascii")


def test_verifier_uses_read_only_fixed_docker_inspect_vector() -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(command: list[str], **kwargs: object) -> object:
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=inspect_stdout())

    docker = Path("/fixture/docker")
    identity = VERIFIER.verify_container_image(docker=docker, runner=runner)

    assert identity.container_id == "1" * 64
    assert identity.image_id == VERIFIER.EXPECTED_IMAGE_ID
    assert identity.config_image == VERIFIER.EXPECTED_CONFIG_IMAGE
    assert identity.name == VERIFIER.EXPECTED_CONTAINER_NAME
    assert identity.running is True
    assert identity.pid == 1234
    assert identity.started_at == "2026-07-24T00:00:00.000000000Z"
    assert calls == [
        (
            [
                str(docker),
                "inspect",
                "--type",
                "container",
                "--format",
                VERIFIER.INSPECT_FORMAT,
                VERIFIER.authorization.FIXED_OPENWEBUI_CONTAINER_NAME,
            ],
            {
                "check": False,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.DEVNULL,
                "timeout": VERIFIER.INSPECT_TIMEOUT_SECONDS,
            },
        )
    ]


@pytest.mark.parametrize(
    ("raw", "match"),
    (
        (
            inspect_stdout(image_id="sha256:" + "0" * 64),
            "image ID differs",
        ),
        (
            inspect_stdout(config_image="ullm/open-webui:other"),
            "config image differs",
        ),
        (
            inspect_stdout(name="/other"),
            "container name differs",
        ),
        (
            inspect_stdout(running=False),
            "not running",
        ),
    ),
)
def test_verifier_rejects_live_identity_mismatch(raw: bytes, match: str) -> None:
    def runner(_command: list[str], **_kwargs: object) -> object:
        return SimpleNamespace(returncode=0, stdout=raw)

    with pytest.raises(VERIFIER.VerificationError, match=match):
        VERIFIER.verify_container_image(
            docker=Path("/fixture/docker"),
            runner=runner,
        )


@pytest.mark.parametrize(
    "raw",
    (
        b"",
        b" " + inspect_stdout(),
        inspect_stdout().replace(b"true", b"true "),
        inspect_stdout() + b"null\n",
        inspect_stdout(container_id="0" * 63),
        inspect_stdout(container_id="G" * 64),
        inspect_stdout(pid=0),
        inspect_stdout(pid=True),
        inspect_stdout(started_at=""),
        inspect_stdout(started_at="\N{SNOWMAN}"),
        b"x" * (VERIFIER.MAX_INSPECT_BYTES + 1),
    ),
)
def test_verifier_rejects_malformed_or_oversized_inspect_output(
    raw: bytes,
) -> None:
    def runner(_command: list[str], **_kwargs: object) -> object:
        return SimpleNamespace(returncode=0, stdout=raw)

    with pytest.raises(VERIFIER.VerificationError):
        VERIFIER.verify_container_image(
            docker=Path("/fixture/docker"),
            runner=runner,
        )


def test_verifier_fails_closed_on_docker_error() -> None:
    def runner(_command: list[str], **_kwargs: object) -> object:
        return SimpleNamespace(returncode=1, stdout=b"")

    with pytest.raises(VERIFIER.VerificationError, match="Docker inspect failed"):
        VERIFIER.verify_container_image(
            docker=Path("/fixture/docker"),
            runner=runner,
        )
