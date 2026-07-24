from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = ROOT / "tools/ullm-campaign-docker"
LOADER = importlib.machinery.SourceFileLoader(
    "test_served_model_campaign_docker_wrapper_module",
    os.fspath(WRAPPER_PATH),
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
WRAPPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WRAPPER
LOADER.exec_module(WRAPPER)

LEASE = (
    "com.ultimatellm.served-model-campaign.claim=" + "a" * 64
)
ENVIRONMENT = {WRAPPER.LEASE_ENVIRONMENT: LEASE}


@pytest.mark.parametrize(
    ("arguments", "expected"),
    (
        (
            ("run", "--rm", "image"),
            (
                WRAPPER.REAL_DOCKER,
                "run",
                f"--label={LEASE}",
                "--rm",
                "image",
            ),
        ),
        (
            ("create", "image"),
            (
                WRAPPER.REAL_DOCKER,
                "create",
                f"--label={LEASE}",
                "image",
            ),
        ),
        (
            ("container", "run", "--rm", "image"),
            (
                WRAPPER.REAL_DOCKER,
                "container",
                "run",
                f"--label={LEASE}",
                "--rm",
                "image",
            ),
        ),
        (
            ("container", "create", "image"),
            (
                WRAPPER.REAL_DOCKER,
                "container",
                "create",
                f"--label={LEASE}",
                "image",
            ),
        ),
    ),
)
def test_create_vectors_receive_exact_claim_lease_label(
    arguments: tuple[str, ...],
    expected: tuple[str, ...],
) -> None:
    assert WRAPPER.rewrite_arguments(arguments, ENVIRONMENT) == expected


@pytest.mark.parametrize(
    "arguments",
    (
        ("container", "inspect", "open-webui"),
        ("image", "inspect", "sha256:" + "1" * 64),
        ("network", "inspect", "open-webui-network"),
        ("compose", "-f", "/source/compose.yaml", "up", "-d"),
    ),
)
def test_non_create_vectors_are_forwarded_without_label_mutation(
    arguments: tuple[str, ...],
) -> None:
    assert WRAPPER.rewrite_arguments(arguments, ENVIRONMENT) == (
        WRAPPER.REAL_DOCKER,
        *arguments,
    )


@pytest.mark.parametrize(
    ("arguments", "environment"),
    (
        (("run", "image"), {}),
        (("run", "image"), {WRAPPER.LEASE_ENVIRONMENT: "bad"}),
        (("--context", "host", "run", "image"), ENVIRONMENT),
        (("run", "--label=attacker=value", "image"), ENVIRONMENT),
        (("run", "--label", "attacker=value", "image"), ENVIRONMENT),
        (("run", "--label-file=/tmp/labels", "image"), ENVIRONMENT),
        (("run", "-l=attacker=value", "image"), ENVIRONMENT),
        (("run", "-lattacker=value", "image"), ENVIRONMENT),
        (("run", "-itlattacker=value", "image"), ENVIRONMENT),
        (("create", "-Pdil=attacker=value", "image"), ENVIRONMENT),
    ),
)
def test_wrapper_rejects_lease_bypass_vectors(
    arguments: tuple[str, ...],
    environment: dict[str, str],
) -> None:
    with pytest.raises(WRAPPER.DockerLeaseWrapperError):
        WRAPPER.rewrite_arguments(arguments, environment)


def test_short_option_values_containing_l_are_not_misparsed_as_labels() -> None:
    assert WRAPPER.rewrite_arguments(
        ("run", "-plocalhost:8000:8000", "image"),
        ENVIRONMENT,
    ) == (
        WRAPPER.REAL_DOCKER,
        "run",
        f"--label={LEASE}",
        "-plocalhost:8000:8000",
        "image",
    )


def test_main_execs_only_the_fixed_real_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_execve(
        executable: str,
        arguments: list[str],
        environment: dict[str, str],
    ) -> None:
        observed.update(
            executable=executable,
            arguments=arguments,
            environment=environment,
        )
        raise RuntimeError("execve sentinel")

    monkeypatch.setenv(WRAPPER.LEASE_ENVIRONMENT, LEASE)
    monkeypatch.setattr(WRAPPER.os, "execve", fake_execve)
    with pytest.raises(RuntimeError, match="sentinel"):
        WRAPPER.main(("run", "--rm", "image"))
    assert observed["executable"] == WRAPPER.REAL_DOCKER
    assert observed["arguments"] == [
        WRAPPER.REAL_DOCKER,
        "run",
        f"--label={LEASE}",
        "--rm",
        "image",
    ]
    assert observed["environment"][WRAPPER.LEASE_ENVIRONMENT] == LEASE


def test_wrapper_has_single_argument_isolated_python_shebang() -> None:
    assert WRAPPER_PATH.read_bytes().splitlines()[0] == (
        b"#!/usr/bin/python3.12 -I"
    )
    assert WRAPPER_PATH.stat().st_mode & 0o111
    completed = subprocess.run(
        [os.fspath(WRAPPER_PATH), "version"],
        env={},
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
    )
    assert completed.returncode == 125
