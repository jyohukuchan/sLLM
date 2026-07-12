from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_EXAMPLE = (
    ROOT / "deploy/systemd/ullm-openai-manifest.env.example"
)
DROP_IN = ROOT / "deploy/systemd/ullm-openai.service.d/10-served-model.conf"
DEPLOYMENT_README = ROOT / "deploy/README.md"

LEGACY_MODEL_ENVIRONMENT = {
    "ULLM_WORKER_BINARY",
    "ULLM_PRODUCT_ROOT",
    "ULLM_ARTIFACT_DIR",
    "ULLM_PACKAGE_DIR",
    "ULLM_TOKENIZER_DIR",
    "ULLM_MODEL_ID",
    "ULLM_MODEL_NAME",
    "ULLM_MODEL_DESCRIPTION",
    "ULLM_MODEL_REVISION",
    "ULLM_ARTIFACT_CONTENT_SHA256",
    "ULLM_PACKAGE_MANIFEST_SHA256",
    "ULLM_DEVICE",
    "ULLM_EXECUTION_PROFILE",
    "ULLM_MODEL_CONTEXT_LENGTH",
    "ULLM_MAX_NEW_TOKENS",
    "ULLM_VOCAB_SIZE",
    "ULLM_EOS_TOKEN_IDS",
    "ULLM_TOP_K",
    "ULLM_HIP_GUARDS",
    "ULLM_WORKER_EXTRA_ARGS",
    "ULLM_TOKENIZER_PROFILE",
}


def environment_names(path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            names.add(line.split("=", 1)[0])
    return names


def test_manifest_environment_is_operations_only() -> None:
    names = environment_names(ENVIRONMENT_EXAMPLE)

    assert "ULLM_SERVED_MODEL_MANIFEST" in names
    assert not names.intersection(LEGACY_MODEL_ENVIRONMENT)


def test_manifest_drop_in_replaces_legacy_environment_and_runs_validator() -> None:
    text = DROP_IN.read_text(encoding="utf-8")

    assert "EnvironmentFile=\n" in text
    assert "EnvironmentFile=/etc/ullm/openai-gateway-manifest.env" in text
    assert "ExecStartPre=\n" in text
    assert "tools/validate-served-model.py --manifest ${ULLM_SERVED_MODEL_MANIFEST}" in text
    assert "/bin/sh" not in text
    assert "/bin/bash" not in text


def test_activation_hook_examples_are_valid_shell_free_json_commands() -> None:
    text = DEPLOYMENT_README.read_text(encoding="utf-8")
    documents = re.findall(r"--(?:check|reconcile|final-check|rollback)-command-json '([^']+)'", text)

    assert len(documents) == 11
    for document in documents:
        command = json.loads(document)
        assert isinstance(command, list) and command
        assert all(isinstance(argument, str) and argument for argument in command)
        assert command[0].startswith("/")
        assert command[0] not in {"/bin/sh", "/bin/bash", "/usr/bin/sh", "/usr/bin/bash"}
