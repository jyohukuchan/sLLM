#!/usr/bin/env python3
"""Source-bound production command plan for the AQ4_0/SQ8_0 campaign window."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import served_model_campaign_authorization as authorization


PLAN_ID = "ullm.served_model.v2_cross_model_campaign_plan.v2"
ACTIVE_MANIFEST = authorization.FIXED_ACTIVE_MANIFEST
SYSTEMD_UNIT = authorization.FIXED_SYSTEMD_UNIT_PATH
ENVIRONMENT_FILE = authorization.FIXED_ENVIRONMENT_FILE_PATH
API_KEY_FILE = Path("/etc/ullm/openai-api-key")
OPENWEBUI_SESSION_TOKEN_FILE = Path(
    "/run/ullm-campaign-secrets/openwebui-session.jwt"
)
SERVICE_UNIT = authorization.FIXED_SERVICE_UNIT
INACTIVE_SERVICES = (SERVICE_UNIT,)
PYTHON = "/usr/bin/python3.12"
PYTHON_PREFIX = (PYTHON, "-I", "-S", "-B")
SYSTEMCTL = "/usr/bin/systemctl"
DOCKER = "/usr/bin/docker"
ROCM_SMI = "/opt/rocm-7.2.1/libexec/rocm_smi/rocm_smi.py"
HTTP_IMAGE = (
    "sha256:5dce198cca467ce79994ed65e01d03882238f9efdd16a8c6f4bc55151c8a4a54"
)
BROWSER_IMAGE = authorization.FIXED_BROWSER_IMAGE
OPENWEBUI_IMAGE = authorization.FIXED_OPENWEBUI_IMAGE
OPENWEBUI_CONTAINER_NAME = authorization.FIXED_OPENWEBUI_CONTAINER_NAME
OPENWEBUI_CONFIG_IMAGE = authorization.FIXED_OPENWEBUI_CONFIG_IMAGE
GATEWAY_CHECK_IMAGE = (
    "ghcr.io/open-webui/open-webui@sha256:"
    "a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff"
)
OPENWEBUI_URL = "http://127.0.0.1:3000"
SQ8_MODEL_ID = "ullm-qwen3-14b-sq8"
SQ8_MODEL_NAME = "uLLM Qwen3 14B SQ8"
AQ4_MODEL_ID = "ullm-qwen3.5-9b-aq4"
AQ4_MODEL_NAME = "uLLM Qwen3.5 9B AQ4 reasoning"


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


def openwebui_verifier_command(source_root: Path) -> tuple[str, ...]:
    return (
        *PYTHON_PREFIX,
        _tool(source_root, "verify-openwebui-container-image.py"),
        "--docker",
        DOCKER,
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
    aq4_release_campaign = campaigns["aq4_reasoning_release"]
    aq4_browser_campaign = campaigns["aq4_reasoning_browser"]
    aq4_bundle_campaign = campaigns["aq4_bundle"]
    aq4_release = authorization_document["aq4_release"]
    if aq4_release.get("openwebui_image") != OPENWEBUI_IMAGE:
        raise PlanError("authorization OpenWebUI image differs from fixed plan")
    before = authorization_document["before"]
    rollback = authorization_document["rollback"]
    aq4_source_root = Path(aq4_release["source"]["root"])
    aq4_raw_output = Path(aq4_release_campaign["final_path"])
    aq4_browser_output = Path(aq4_browser_campaign["final_path"])
    aq4_bundle_output = Path(aq4_bundle_campaign["final_path"])
    aq4_release_evidence = Path(aq4_release["release_evidence_path"])
    aq4_release_validator = Path(aq4_release["release_validator_path"])
    aq4_browser_validator = Path(aq4_release["browser_validator_path"])
    aq4_manifest = Path(rollback["backup_path"])
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
        openwebui_verifier_command(source_root),
    )
    return TransactionCommands(
        candidate_reconciliation=reconcile,
        candidate_checks=(
            (SYSTEMCTL, "is-active", "--quiet", SERVICE_UNIT),
            _ready_command(),
            openwebui_verifier_command(source_root),
        ),
        sq8_full=(
            *PYTHON_PREFIX,
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
            *PYTHON_PREFIX,
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
            *PYTHON_PREFIX,
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
            "--openwebui-image",
            OPENWEBUI_IMAGE,
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
        aq4_reasoning_release=(
            (
                *PYTHON_PREFIX,
                _tool(
                    aq4_source_root,
                    "run-generic-reasoning-release-campaign.py",
                ),
                "--output-dir",
                os.fspath(aq4_raw_output),
                "--manifest",
                os.fspath(aq4_manifest),
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
            (
                *PYTHON_PREFIX,
                _tool(
                    aq4_source_root,
                    "prepare-generic-reasoning-release-evidence.py",
                ),
                "--cases",
                os.fspath(aq4_raw_output / "cases.json"),
                "--lifecycle",
                os.fspath(aq4_raw_output / "lifecycle.json"),
                "--manifest",
                os.fspath(aq4_manifest),
                "--worker-binary",
                before["worker_binary_path"],
                "--openwebui-image",
                OPENWEBUI_IMAGE,
                "--active-promotion-source-commit",
                before["promotion_source_commit"],
                "--output",
                os.fspath(aq4_release_evidence),
                "--status",
                "complete",
            ),
            (
                *PYTHON_PREFIX,
                _tool(
                    source_root,
                    "publish-generic-reasoning-validator-report.py",
                ),
                "--kind",
                "release",
                "--evidence",
                os.fspath(aq4_release_evidence),
                "--output",
                os.fspath(aq4_release_validator),
                "--require-complete",
            ),
        ),
        aq4_reasoning_browser=(
            (
                *PYTHON_PREFIX,
                _tool(
                    aq4_source_root,
                    "run-openwebui-reasoning-browser-smoke.py",
                ),
                "--output",
                os.fspath(aq4_browser_output),
                "--manifest",
                os.fspath(aq4_manifest),
                "--token-file",
                os.fspath(OPENWEBUI_SESSION_TOKEN_FILE),
                "--browser-image",
                BROWSER_IMAGE,
                "--openwebui-url",
                OPENWEBUI_URL,
                "--model-id",
                AQ4_MODEL_ID,
                "--model-name",
                AQ4_MODEL_NAME,
                "--ullm-service",
                SERVICE_UNIT,
                "--docker",
                DOCKER,
                "--systemctl",
                SYSTEMCTL,
                "--rocm-smi",
                ROCM_SMI,
            ),
            (
                *PYTHON_PREFIX,
                _tool(
                    source_root,
                    "publish-generic-reasoning-validator-report.py",
                ),
                "--kind",
                "browser",
                "--evidence",
                os.fspath(aq4_browser_output),
                "--output",
                os.fspath(aq4_browser_validator),
                "--require-complete",
            ),
        ),
        aq4_bundle=(
            (
                *PYTHON_PREFIX,
                _tool(
                    source_root,
                    "prepare-generic-reasoning-release-bundle.py",
                ),
                "--bundle-version",
                "v1",
                "--release-evidence",
                os.fspath(aq4_release_evidence),
                "--release-validator",
                os.fspath(aq4_release_validator),
                "--browser-evidence",
                os.fspath(aq4_browser_output),
                "--browser-validator",
                os.fspath(aq4_browser_validator),
                "--promotion-evidence",
                aq4_release["promotion_evidence"]["path"],
                "--promotion-receipt",
                aq4_release["promotion_receipt"]["path"],
                "--rollback-manifest",
                os.fspath(aq4_manifest),
                "--systemd-unit",
                os.fspath(SYSTEMD_UNIT),
                "--environment-file",
                os.fspath(ENVIRONMENT_FILE),
                "--output",
                os.fspath(aq4_bundle_output),
                "--status",
                "complete",
            ),
            (
                *PYTHON_PREFIX,
                _tool(
                    source_root,
                    "validate-generic-reasoning-release-bundle.py",
                ),
                os.fspath(aq4_bundle_output),
                "--require-complete",
            ),
        ),
        final_checks=(
            (SYSTEMCTL, "is-active", "--quiet", SERVICE_UNIT),
            _ready_command(),
            openwebui_verifier_command(source_root),
        ),
    )
