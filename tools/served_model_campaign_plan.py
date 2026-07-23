#!/usr/bin/env python3
"""Source-bound production command plan for the AQ4_0/SQ8_0 campaign window."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import served_model_campaign_authorization as authorization


PLAN_ID = "ullm.served_model.v2_cross_model_campaign_plan.v1"
ACTIVE_MANIFEST = authorization.FIXED_ACTIVE_MANIFEST
SYSTEMD_UNIT = authorization.FIXED_SYSTEMD_UNIT_PATH
ENVIRONMENT_FILE = authorization.FIXED_ENVIRONMENT_FILE_PATH
API_KEY_FILE = Path("/etc/ullm/openai-api-key")
OPENWEBUI_SESSION_TOKEN_FILE = Path(
    "/run/ullm/sq8-v2-cross-model-openwebui-session.jwt"
)
SERVICE_UNIT = authorization.FIXED_SERVICE_UNIT
INACTIVE_SERVICES = (SERVICE_UNIT,)
PYTHON = "/usr/bin/python3"
SYSTEMCTL = "/usr/bin/systemctl"
DOCKER = "/usr/bin/docker"
ROCM_SMI = "/opt/rocm/bin/rocm-smi"
HTTP_IMAGE = (
    "sha256:5dce198cca467ce79994ed65e01d03882238f9efdd16a8c6f4bc55151c8a4a54"
)
BROWSER_IMAGE = (
    "sha256:0bd709ea36ffa7204cd60da0fe9707be38eb73c97c7a9d45911ff0e8b7c1e3ea"
)
OPENWEBUI_IMAGE = (
    "ullm/open-webui@sha256:"
    "ef5ae4fbc06abb662eeefe87e584ea7c69e55838f5f08f637057b9108048b409"
)
GATEWAY_CHECK_IMAGE = (
    "ghcr.io/open-webui/open-webui@sha256:"
    "a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff"
)
OPENWEBUI_URL = "http://127.0.0.1:3000"
SQ8_MODEL_ID = "ullm-qwen3-14b-sq8"
SQ8_MODEL_NAME = "uLLM Qwen3 14B SQ8"


class PlanError(ValueError):
    """The fixed production execution plan could not be derived."""


def _tool(source_root: Path, name: str) -> str:
    path = source_root / "tools" / name
    if not path.is_absolute():
        raise PlanError("source-bound tool path is not absolute")
    return os.fspath(path)


def _configure_command(source_root: Path) -> tuple[str, ...]:
    return (
        DOCKER,
        "run",
        "--rm",
        "-v",
        "open-webui:/data",
        "-v",
        f"{API_KEY_FILE}:/run/secrets/ullm-api-key:ro",
        "-v",
        f"{ACTIVE_MANIFEST.parent}:{ACTIVE_MANIFEST.parent}:ro",
        "-v",
        f"{source_root / 'deploy/openwebui/configure.py'}:/configure.py:ro",
        "--entrypoint",
        "python",
        OPENWEBUI_IMAGE,
        "/configure.py",
        "--served-model-manifest",
        os.fspath(ACTIVE_MANIFEST),
        "--base-url",
        "http://172.20.0.1:8000/v1",
    )


def _ready_command() -> tuple[str, ...]:
    return (
        DOCKER,
        "run",
        "--rm",
        "--network",
        "open-webui-network",
        "--entrypoint",
        "curl",
        GATEWAY_CHECK_IMAGE,
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "120",
        "--retry-delay",
        "2",
        "--retry-max-time",
        "600",
        "--retry-all-errors",
        "http://172.20.0.1:8000/readyz",
    )


def derive_commands(
    *,
    source_root: Path,
    authorization_path: Path,
    candidate_manifest: Path,
    authorization_document: dict[str, Any],
) -> Any:
    """Return TransactionCommands without accepting caller-provided vectors."""

    from served_model_campaign_transaction import TransactionCommands

    source = authorization_document["source"]
    candidate = authorization_document["candidate"]
    campaigns = authorization_document["campaigns"]
    common_binding = (
        "--active-binding-mode",
        "v2",
        "--candidate-served-model-manifest",
        os.fspath(candidate_manifest),
        "--active-served-model-manifest",
        os.fspath(ACTIVE_MANIFEST),
        "--expected-served-model-manifest-sha256",
        candidate["manifest_sha256"],
        "--campaign-authorization",
        os.fspath(authorization_path),
    )
    sq8 = campaigns["sq8_full"]
    reasoning = campaigns["reasoning_release"]
    browser = campaigns["reasoning_browser"]
    compose = (
        DOCKER,
        "compose",
        "-f",
        os.fspath(source_root / "deploy/openwebui/compose.yaml"),
        "up",
        "-d",
        "--no-build",
    )
    reconcile = (
        (SYSTEMCTL, "restart", SERVICE_UNIT),
        _configure_command(source_root),
        compose,
    )
    return TransactionCommands(
        candidate_reconciliation=reconcile,
        candidate_checks=(
            (SYSTEMCTL, "is-active", "--quiet", SERVICE_UNIT),
            _ready_command(),
        ),
        sq8_full=(
            PYTHON,
            _tool(source_root, "run-sq8-full-openwebui-campaign.py"),
            "--execute",
            "--expected-commit",
            source["commit"],
            "--expected-worker-binary-sha256",
            candidate["worker_binary_sha256"],
            "--run-id",
            sq8["run_id"],
            "--final-path",
            sq8["final_path"],
            "--api-key-file",
            os.fspath(API_KEY_FILE),
            "--openwebui-session-token-file",
            os.fspath(OPENWEBUI_SESSION_TOKEN_FILE),
            *common_binding,
        ),
        reasoning_release=(
            PYTHON,
            _tool(source_root, "run-generic-reasoning-release-campaign.py"),
            "--output-dir",
            reasoning["final_path"],
            *common_binding,
            "--run-id",
            reasoning["run_id"],
            "--token-file",
            os.fspath(API_KEY_FILE),
            "--http-image",
            HTTP_IMAGE,
            "--service",
            SERVICE_UNIT,
            "--docker",
            DOCKER,
            "--rocm-smi",
            ROCM_SMI,
            "--systemctl",
            SYSTEMCTL,
        ),
        reasoning_browser=(
            PYTHON,
            _tool(source_root, "run-openwebui-reasoning-browser-smoke.py"),
            "--output",
            browser["final_path"],
            *common_binding,
            "--run-id",
            browser["run_id"],
            "--openwebui-session-token-file",
            os.fspath(OPENWEBUI_SESSION_TOKEN_FILE),
            "--browser-image",
            BROWSER_IMAGE,
            "--openwebui-url",
            OPENWEBUI_URL,
            "--model-id",
            SQ8_MODEL_ID,
            "--model-name",
            SQ8_MODEL_NAME,
            "--ullm-service",
            SERVICE_UNIT,
            "--docker",
            DOCKER,
            "--systemctl",
            SYSTEMCTL,
            "--rocm-smi",
            ROCM_SMI,
        ),
        reverse_reconciliation=reconcile,
        final_checks=(
            (SYSTEMCTL, "is-active", "--quiet", SERVICE_UNIT),
            _ready_command(),
        ),
    )
