#!/usr/bin/env python3
"""Validate CI JSON schemas/manifests and the checked-in workflow structure."""

from __future__ import annotations

import re
import sys
import hashlib
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, ROOT, load_manifests, read_json  # noqa: E402
from validate_matrix import main as validate_matrix_main  # noqa: E402
from validate_rmsnorm_g1_contracts import (  # noqa: E402
    SEMANTIC_G1_UPLOAD_PATHS,
    SEMANTIC_G1_UPLOAD_PATH_TEXT,
)

HOST_WORKFLOW_JOBS = {"h0", "h1", "h2", "host-required"}
H3_WORKFLOW_JOBS = {"h3-gfx1030", "h3-gfx1201", "h3-aggregate"}
H3_PUBLIC_RUNTIME_WORKFLOW_JOBS = {"h3-public-runtime"}
H3_RMSNORM_WORKFLOW_JOBS = {"h3-rmsnorm"}
H3_RMSNORM_ACTIONS = {
    "checkout": "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",
    "setup_python": "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1",
    "upload": "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
}
H3_RMSNORM_IMAGE_REFERENCE = "docker.io/rocm/dev-ubuntu-24.04@sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7"
H3_RMSNORM_IMAGE_MANIFEST_DIGEST = "sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7"
H3_RMSNORM_IMAGE_CONFIG_DIGEST = "sha256:4c91c0d850e38a40fd669dd043ab42e9bad9a2b8a38e3f873c5a4eaced9f28cf"
H3_RMSNORM_ENV = {
    "RMSNORM_H3_IMAGE_REFERENCE": H3_RMSNORM_IMAGE_REFERENCE,
    "RMSNORM_H3_IMAGE_MANIFEST_DIGEST": H3_RMSNORM_IMAGE_MANIFEST_DIGEST,
    "RMSNORM_H3_IMAGE_CONFIG_DIGEST": H3_RMSNORM_IMAGE_CONFIG_DIGEST,
    "SLLM_H3_NETWORK_DISABLED": "1",
}
H3_RMSNORM_SCHEMA_FILES = {
    "ci/schema/rmsnorm-h3-compile-v1.schema.json",
    "ci/schema/rmsnorm-h3-artifact-v1.schema.json",
    "ci/schema/rmsnorm-h3-report-v1.schema.json",
    "ci/schema/rmsnorm-h3-aggregate-v1.schema.json",
}
H3_RMSNORM_MATRIX = "ci/matrix/rmsnorm-h3-compile-v1.json"
H3_RMSNORM_WORKFLOW_PATH = ".github/workflows/rmsnorm-h3-compile.yml"
H3_RMSNORM_WORKFLOW_NAME = "h3-rmsnorm-compile-only (non-required)"
H3_RMSNORM_WORKFLOW_TRIGGER = {
    "pull_request": None,
    "push": {"branches": ["main"]},
    "workflow_dispatch": None,
}
SEMANTIC_G1_WORKFLOW_PATH = ".github/workflows/semantic-rmsnorm-g1.yml"
SEMANTIC_G1_WORKFLOW_NAME = "semantic-rmsnorm-g1"
SEMANTIC_G1_WORKFLOW_SHA256 = "a1c0cc85334445c14c15b5be43e979f587a4f2bd8cb8b53690603b65939770fc"
SEMANTIC_G1_WORKFLOW_TRIGGER = {"push": {"branches": ["main"]}, "workflow_dispatch": None}
SEMANTIC_G1_ACTIONS = {
    "checkout": "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",
    "upload": "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
}
SEMANTIC_G1_SCHEMA_FILES = {
    "ci/schema/rmsnorm-semantic-g1-matrix-v1.schema.json",
    "ci/schema/rmsnorm-semantic-g1-artifact-v1.schema.json",
    "ci/schema/rmsnorm-semantic-g1-report-v1.schema.json",
    "ci/schema/rmsnorm-semantic-g1-aggregate-v1.schema.json",
}
G2_SCHEMA_FILES = {
    "ci/schema/rmsnorm-g2-matrix-v1.schema.json",
    "ci/schema/rmsnorm-g2-model-slice-v1.schema.json",
    "ci/schema/rmsnorm-g2-tolerance-v1.schema.json",
    "ci/schema/rmsnorm-g2-runtime-result-v1.schema.json",
    "ci/schema/rmsnorm-g2-artifact-v1.schema.json",
    "ci/schema/rmsnorm-g2-report-v1.schema.json",
    "ci/schema/rmsnorm-g2-aggregate-v1.schema.json",
}
G2_MATRIX = "ci/matrix/rmsnorm-g2-v1.json"
G2_TOLERANCE = "ci/matrix/rmsnorm-g2-tolerance-v1.json"
P0_SCHEMA_FILES = {
    "ci/schema/rmsnorm-p0-matrix-v1.schema.json",
    "ci/schema/rmsnorm-p0-review-policy-v1.schema.json",
    "ci/schema/rmsnorm-p0-artifact-v1.schema.json",
    "ci/schema/rmsnorm-p0-runtime-result-v1.schema.json",
    "ci/schema/rmsnorm-p0-report-v1.schema.json",
    "ci/schema/rmsnorm-p0-review-disposition-v1.schema.json",
    "ci/schema/rmsnorm-p0-aggregate-v1.schema.json",
}
P0_MATRIX = "ci/matrix/rmsnorm-p0-v1.json"
P0_REVIEW_POLICY = "ci/matrix/rmsnorm-p0-review-policy-v1.json"
P0_PUBLIC_PATH_INPUTS = "ci/matrix/rmsnorm-p0-public-path-inputs-v1.json"
PHASE3_STAGE_A_EVIDENCE_PLAN_SCHEMA = "ci/schema/phase3-stage-a-evidence-plan-v1.schema.json"
H3_PUBLIC_RUNTIME_STEP_NAMES = [
    "Checkout immutable candidate",
    "Prepare private public-H3 directories",
    "Verify immutable identity and pinned image",
    "Compile, link, extract, and inspect gfx1030",
    "Compile, link, extract, and inspect gfx1201",
    "Prepare exact public-runtime needs input",
    "Aggregate exactly two public-runtime PASS rows locally",
    "Upload JSON aggregate only",
    "Cleanup generated public-H3 rows and needs",
]
SHA40 = re.compile(r"@[0-9a-f]{40}$")
H3_IMAGE_REFERENCE = "docker.io/rocm/dev-ubuntu-24.04@sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7"
H3_IMAGE_CONFIG_DIGEST = "sha256:4c91c0d850e38a40fd669dd043ab42e9bad9a2b8a38e3f873c5a4eaced9f28cf"
H3_GIT_HELPER_MOUNT = "type=bind,src=/usr/bin/git,dst=/usr/local/bin/git,readonly"
H3_PUBLIC_RUNTIME_ACTIONS = {
    "checkout": "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",
    "upload": "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
}
H3_PUBLIC_RUNTIME_WORKFLOW_TRIGGER = {
    "pull_request": None,
    "push": {"branches": ["main"]},
    "workflow_dispatch": None,
}
H3_PUBLIC_RUNTIME_ENV = {
    "H3_PUBLIC_RUNTIME_IMAGE_REFERENCE": H3_IMAGE_REFERENCE,
    "H3_PUBLIC_RUNTIME_IMAGE_MANIFEST_DIGEST": "sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7",
    "H3_PUBLIC_RUNTIME_IMAGE_CONFIG_DIGEST": H3_IMAGE_CONFIG_DIGEST,
}
H3_PUBLIC_RUNTIME_AGGREGATE_COMMAND = (
    "python3 ci/tools/aggregate_h3_public_runtime_results.py "
    "--needs-json .local-artifacts/h3-public-runtime-needs.json "
    "--artifact-dir .local-artifacts/h3-public-runtime "
    "--output-dir .local-artifacts/h3-public-runtime-aggregate "
    "--strict-ci "
    "--run-id \"${{ github.run_id }}\" "
    "--run-attempt \"${{ github.run_attempt }}\" "
    "--expected-reviewed-sha \"${{ github.sha }}\" "
    "--expected-tested-sha \"${{ github.sha }}\" "
    "--expected-workflow-sha \"${{ github.sha }}\" "
    "--tree-oid \"$(git rev-parse HEAD^{tree})\""
)
H3_AGGREGATE_SCHEMA = "ci/schema/h3-aggregate-v1.schema.json"
H3_PUBLIC_RUNTIME_SCHEMA_FILES = {
    "ci/schema/hip-runtime-compile-v1.schema.json",
    "ci/schema/hip-runtime-artifact-v1.schema.json",
    "ci/schema/hip-runtime-public-report-v1.schema.json",
    "ci/schema/hip-runtime-aggregate-v1.schema.json",
}
H3_PUBLIC_RUNTIME_MATRIX = "ci/matrix/hip-runtime-compile-v1.json"
G1_SCHEMA_FILES = {
    "ci/schema/g1-aggregate-v1.schema.json",
    "ci/schema/g1-report-v1.schema.json",
    "ci/schema/g1-runtime-artifact-v1.schema.json",
}
MODEL_LOCK_SCHEMA = "ci/schema/model-lock-v1.schema.json"
MODEL_LOCK_PATH = "docs/models/locks/qwen3.5-4b-bf16.json"
RUST_DEPENDENCY_SCHEMA = "ci/schema/rust-dependency-policy-v1.schema.json"
RUST_DEPENDENCY_MANIFEST = "ci/dependencies/rust-workspace-v1.json"


def h3_workspace_expectations() -> dict[str, object]:
    """Return the shared read-only checkout mount/workdir contract."""

    return {
        "mount_destination": "/workspace",
        "mount_read_only": True,
        "workdir": "/workspace",
    }


def h3_rmsnorm_row_expectation(target: str, run_id: str, run_attempt: str) -> dict[str, str]:
    """Derive one RMSNorm H3 row's container output path without shell text."""

    if target not in {"gfx1030", "gfx1201"}:
        raise ContractError(f"unsupported RMSNorm H3 target: {target}")
    if not run_id or not run_attempt:
        raise ContractError("RMSNorm H3 row path requires run identity")
    row_id = f"h3-rmsnorm-{target}"
    container_root = f"/tmp/sllm-rmsnorm-h3-{target}-{run_id}-{run_attempt}"
    return {
        "target": target,
        "row_id": row_id,
        "container_output_root": container_root,
        "container_output_dir": f"{container_root}/{row_id}",
    }


def _expected_rmsnorm_verify_run() -> str:
    return """set -eu
test "$(command -v git)" = /usr/bin/git
git --version
TREE_OID="$(git rev-parse HEAD^{tree})"
export TREE_OID
test "$(git rev-parse HEAD)" = "$REVIEWED_SHA"
test "$(git rev-parse HEAD)" = "$TESTED_SHA"
test "$(git rev-parse HEAD)" = "$WORKFLOW_SHA"
test "$(git rev-parse HEAD^{tree})" = "$TREE_OID"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
docker pull "$RMSNORM_H3_IMAGE_REFERENCE"
test "$(docker image inspect --format '{{.Id}}' "$RMSNORM_H3_IMAGE_REFERENCE")" = "$RMSNORM_H3_IMAGE_CONFIG_DIGEST"
docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$RMSNORM_H3_IMAGE_REFERENCE" | grep -F -- "$RMSNORM_H3_IMAGE_MANIFEST_DIGEST" >/dev/null
"""


def _expected_rmsnorm_row_run(target: str) -> str:
    expectation = h3_rmsnorm_row_expectation(target, "${GITHUB_RUN_ID}", "${GITHUB_RUN_ATTEMPT}")
    workspace = h3_workspace_expectations()
    output_root = expectation["container_output_root"]
    row_id = expectation["row_id"]
    workspace_mount = f'type=bind,src=$GITHUB_WORKSPACE,dst={workspace["mount_destination"]},readonly'
    return f'''set -eu
test -n "$RUN_ROOT"
TREE_OID="$(git rev-parse HEAD^{{tree}})"
export TREE_OID
docker run --rm --network none --user "$(id -u):$(id -g)" \\
  --mount "type=bind,src=$RUN_ROOT,dst=/tmp" \\
  --mount "{workspace_mount}" \\
  --mount "type=bind,src=/usr/bin/git,dst=/usr/local/bin/git,readonly" \\
  --env HOME=/tmp/sllm-rmsnorm-h3-home-${{GITHUB_RUN_ID}}-${{GITHUB_RUN_ATTEMPT}} \\
  --env REVIEWED_SHA --env TESTED_SHA --env WORKFLOW_SHA --env TREE_OID \\
  --env GITHUB_RUN_ID --env GITHUB_RUN_ATTEMPT --env SLLM_H3_NETWORK_DISABLED \\
  --env RMSNORM_H3_IMAGE_REFERENCE --env RMSNORM_H3_IMAGE_CONFIG_DIGEST \\
  --workdir {workspace["workdir"]} \\
  "$RMSNORM_H3_IMAGE_REFERENCE" /bin/bash -eu -o pipefail -c '
    mkdir -p "$HOME"
    git config --global --add safe.directory /workspace
    python3 ci/tools/run_rmsnorm_h3_compile.py \\
      --repo {workspace["workdir"]} --row {row_id} \\
      --output-dir "{output_root}" \\
      --strict-ci --pinned-container \\
      --observed-image-reference "$RMSNORM_H3_IMAGE_REFERENCE" \\
      --observed-image-config-digest "$RMSNORM_H3_IMAGE_CONFIG_DIGEST" \\
      --reviewed-sha "$REVIEWED_SHA" --tested-sha "$TESTED_SHA" \\
      --workflow-sha "$WORKFLOW_SHA" --tree-oid "$TREE_OID" \\
      --run-id "${{GITHUB_RUN_ID}}" --run-attempt "${{GITHUB_RUN_ATTEMPT}}"
  '
'''


def _expected_rmsnorm_aggregate_run() -> str:
    collection_root = "/tmp/sllm-rmsnorm-h3-collection-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
    container_aggregate_root = "/tmp/sllm-rmsnorm-h3-aggregate-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
    host_aggregate_root = "$RUN_ROOT/sllm-rmsnorm-h3-aggregate-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
    return f'''set -eu
test -n "$RUN_ROOT"
test -d "$RUN_ROOT/sllm-rmsnorm-h3-gfx1030-${{GITHUB_RUN_ID}}-${{GITHUB_RUN_ATTEMPT}}/h3-rmsnorm-gfx1030"
test -d "$RUN_ROOT/sllm-rmsnorm-h3-gfx1201-${{GITHUB_RUN_ID}}-${{GITHUB_RUN_ATTEMPT}}/h3-rmsnorm-gfx1201"
mkdir -m 700 "$RUN_ROOT/sllm-rmsnorm-h3-collection-${{GITHUB_RUN_ID}}-${{GITHUB_RUN_ATTEMPT}}"
cp -a "$RUN_ROOT/sllm-rmsnorm-h3-gfx1030-${{GITHUB_RUN_ID}}-${{GITHUB_RUN_ATTEMPT}}/h3-rmsnorm-gfx1030" "$RUN_ROOT/sllm-rmsnorm-h3-collection-${{GITHUB_RUN_ID}}-${{GITHUB_RUN_ATTEMPT}}/"
cp -a "$RUN_ROOT/sllm-rmsnorm-h3-gfx1201-${{GITHUB_RUN_ID}}-${{GITHUB_RUN_ATTEMPT}}/h3-rmsnorm-gfx1201" "$RUN_ROOT/sllm-rmsnorm-h3-collection-${{GITHUB_RUN_ID}}-${{GITHUB_RUN_ATTEMPT}}/"
mkdir -p -m 700 "$GITHUB_WORKSPACE/.local-artifacts/rmsnorm-h3-aggregate"
TREE_OID="$(git rev-parse HEAD^{{tree}})"
export TREE_OID
docker run --rm --network none --user "$(id -u):$(id -g)" \\
  --mount "type=bind,src=$RUN_ROOT,dst=/tmp" \\
  --mount "type=bind,src=$GITHUB_WORKSPACE,dst=/workspace,readonly" \\
  --mount "type=bind,src=/usr/bin/git,dst=/usr/local/bin/git,readonly" \\
  --env HOME=/tmp/sllm-rmsnorm-h3-home-${{GITHUB_RUN_ID}}-${{GITHUB_RUN_ATTEMPT}} \\
  --env REVIEWED_SHA --env TESTED_SHA --env WORKFLOW_SHA --env TREE_OID \\
  --env GITHUB_RUN_ID --env GITHUB_RUN_ATTEMPT --env SLLM_H3_NETWORK_DISABLED \\
  --env RMSNORM_H3_IMAGE_REFERENCE --env RMSNORM_H3_IMAGE_CONFIG_DIGEST \\
  --env PYTHONPATH=/tmp/python-packages \\
  --workdir /workspace \\
  "$RMSNORM_H3_IMAGE_REFERENCE" /bin/bash -eu -o pipefail -c '
    mkdir -p "$HOME"
    git config --global --add safe.directory /workspace
    python3 ci/tools/aggregate_rmsnorm_h3_results.py \\
      --repo /workspace --artifact-root "{collection_root}" \\
      --output-dir "{container_aggregate_root}" \\
      --strict-ci --expected-reviewed-sha "$REVIEWED_SHA" \\
      --expected-tested-sha "$TESTED_SHA" --expected-workflow-sha "$WORKFLOW_SHA" \\
      --tree-oid "$(git rev-parse HEAD^{{tree}})" --run-id "${{GITHUB_RUN_ID}}" \\
      --run-attempt "${{GITHUB_RUN_ATTEMPT}}"
  '
cp "{host_aggregate_root}/rmsnorm-h3-aggregate.json" "$GITHUB_WORKSPACE/.local-artifacts/rmsnorm-h3-aggregate/"
cp "{host_aggregate_root}/rmsnorm-h3-aggregate.json.sha256" "$GITHUB_WORKSPACE/.local-artifacts/rmsnorm-h3-aggregate/"
'''


def _expected_rmsnorm_checkout_step() -> dict[str, object]:
    return {
        "name": "Checkout immutable candidate",
        "uses": H3_RMSNORM_ACTIONS["checkout"],
        "with": {"ref": "${{ github.sha }}", "fetch-depth": 0, "persist-credentials": False},
    }


def _expected_rmsnorm_steps() -> list[dict[str, object]]:
    return [
        _expected_rmsnorm_checkout_step(),
        {
            "name": "Prepare private per-run directory",
            "run": (
                "set -eu\n"
                "RUN_ROOT=\"$RUNNER_TEMP/sllm-rmsnorm-h3-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}\"\n"
                "test ! -e \"$RUN_ROOT\"\n"
                "mkdir -m 700 \"$RUN_ROOT\"\n"
                "printf 'RUN_ROOT=%s\\n' \"$RUN_ROOT\" >> \"$GITHUB_ENV\"\n"
            ),
        },
        {
            "name": "Set up Python 3.12",
            "uses": H3_RMSNORM_ACTIONS["setup_python"],
            "with": {"python-version": "3.12.10"},
        },
        {
            "name": "Install aggregate schema requirements",
            "run": (
                "python3 -m pip install --disable-pip-version-check --no-input "
                "--require-hashes --only-binary=:all: --no-deps "
                '--target "$RUN_ROOT/python-packages" '
                "-r ci/requirements-host.txt"
            ),
        },
        {
            "name": "Verify immutable identity and pinned image",
            "env": {
                "REVIEWED_SHA": "${{ github.sha }}",
                "TESTED_SHA": "${{ github.sha }}",
                "WORKFLOW_SHA": "${{ github.sha }}",
            },
            "run": _expected_rmsnorm_verify_run(),
        },
        {
            "name": "Compile, link, extract, and inspect gfx1030",
            "env": {
                "REVIEWED_SHA": "${{ github.sha }}",
                "TESTED_SHA": "${{ github.sha }}",
                "WORKFLOW_SHA": "${{ github.sha }}",
            },
            "run": _expected_rmsnorm_row_run("gfx1030"),
        },
        {
            "name": "Compile, link, extract, and inspect gfx1201",
            "env": {
                "REVIEWED_SHA": "${{ github.sha }}",
                "TESTED_SHA": "${{ github.sha }}",
                "WORKFLOW_SHA": "${{ github.sha }}",
            },
            "run": _expected_rmsnorm_row_run("gfx1201"),
        },
        {
            "name": "Aggregate exact RMSNorm rows without binary upload",
            "env": {
                "REVIEWED_SHA": "${{ github.sha }}",
                "TESTED_SHA": "${{ github.sha }}",
                "WORKFLOW_SHA": "${{ github.sha }}",
            },
            "run": _expected_rmsnorm_aggregate_run(),
        },
        {
            "name": "Upload JSON aggregate only",
            "if": "${{ success() }}",
            "uses": H3_RMSNORM_ACTIONS["upload"],
            "with": {
                "name": "rmsnorm-h3-aggregate",
                "path": ".local-artifacts/rmsnorm-h3-aggregate/rmsnorm-h3-aggregate.json\n"
                ".local-artifacts/rmsnorm-h3-aggregate/rmsnorm-h3-aggregate.json.sha256\n",
                "if-no-files-found": "error",
                "retention-days": 7,
            },
        },
        {
            "name": "Cleanup private per-run directory",
            "if": "${{ always() }}",
            "run": (
                "set -eu\n"
                "if [ -n \"${RUN_ROOT:-}\" ] && [ -d \"$RUN_ROOT\" ]; then\n"
                "  rm -rf -- \"$RUN_ROOT\"\n"
                "fi\n"
            ),
        },
    ]


def _workflow_trigger(document: dict[str, object]) -> object:
    if "on" in document:
        return document["on"]
    return document.get(True)


def _expected_public_runtime_row_run(row_id: str) -> str:
    """Return the exact serial container boundary for one public row."""

    workspace = h3_workspace_expectations()
    workspace_mount = f'type=bind,src=$GITHUB_WORKSPACE,dst={workspace["mount_destination"]},readonly'
    template = """set -eu
TREE_OID="$(git rev-parse HEAD^{tree})"
export TREE_OID
test "$(git rev-parse HEAD)" = "$REVIEWED_SHA"
test "$(git rev-parse HEAD)" = "$TESTED_SHA"
test "$(git rev-parse HEAD)" = "$WORKFLOW_SHA"
test "$(git rev-parse HEAD^{tree})" = "$TREE_OID"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
docker run --rm --network none --user "$(id -u):$(id -g)" \\
  --mount "{workspace_mount}" \\
  --mount "type=bind,src=$GITHUB_WORKSPACE/.local-artifacts/h3-public-runtime/ROW_ID,dst=/output" \\
  --mount "type=bind,src=/usr/bin/git,dst=/usr/local/bin/git,readonly" \\
  --env HOME=/tmp/h3-public-runtime-home \\
  --env REVIEWED_SHA --env TESTED_SHA --env WORKFLOW_SHA --env TREE_OID \\
  --env RUN_ID --env RUN_ATTEMPT --env SLLM_H3_NETWORK_DISABLED \\
  --env H3_PUBLIC_RUNTIME_IMAGE_REFERENCE --env H3_PUBLIC_RUNTIME_IMAGE_CONFIG_DIGEST \\
  --workdir {workdir} \\
  "$H3_PUBLIC_RUNTIME_IMAGE_REFERENCE" /bin/bash -eu -o pipefail -c '
    mkdir -p "$HOME"
    git config --global --add safe.directory /workspace
    exec python3 ci/tools/run_h3_public_runtime_compile.py \\
      --row ROW_ID --repo {workdir} --output-dir /output \\
      --strict-ci --pinned-container \\
      --observed-image-reference "$H3_PUBLIC_RUNTIME_IMAGE_REFERENCE" \\
      --observed-image-config-digest "$H3_PUBLIC_RUNTIME_IMAGE_CONFIG_DIGEST" \\
      --reviewed-sha "$REVIEWED_SHA" --tested-sha "$TESTED_SHA" \\
      --workflow-sha "$WORKFLOW_SHA" --tree-oid "$TREE_OID" \\
      --run-id "$RUN_ID" --run-attempt "$RUN_ATTEMPT"
  '
if find "$GITHUB_WORKSPACE/.local-artifacts/h3-public-runtime/ROW_ID" -xdev -type l -print -quit | grep -q .; then
  echo "H3 public-runtime output contains a symlink" >&2
  exit 1
fi
if find "$GITHUB_WORKSPACE/.local-artifacts/h3-public-runtime/ROW_ID" -xdev \\( ! -uid "$(id -u)" -o ! -gid "$(id -g)" \\) -print -quit | grep -q .; then
  echo "H3 public-runtime container output ownership does not match the GitHub runner user" >&2
  exit 1
fi
"""
    return template.replace("{workspace_mount}", workspace_mount).replace("{workdir}", workspace["workdir"]).replace("ROW_ID", row_id)


def _expected_public_runtime_checkout_step() -> dict[str, object]:
    return {
        "name": "Checkout immutable candidate",
        "uses": H3_PUBLIC_RUNTIME_ACTIONS["checkout"],
        "with": {
            "ref": "${{ github.sha }}",
            "fetch-depth": 0,
            "persist-credentials": False,
        },
    }


def _expected_public_runtime_prepare_step() -> dict[str, object]:
    return {
        "name": "Prepare private public-H3 directories",
        "run": (
            "set -eu\n"
            'ARTIFACT_ROOT="$GITHUB_WORKSPACE/.local-artifacts"\n'
            'ROW_ROOT="$ARTIFACT_ROOT/h3-public-runtime"\n'
            'ROW1030="$ROW_ROOT/h3-public-gfx1030"\n'
            'ROW1201="$ROW_ROOT/h3-public-gfx1201"\n'
            'AGGREGATE_ROOT="$ARTIFACT_ROOT/h3-public-runtime-aggregate"\n'
            'NEEDS_PATH="$ARTIFACT_ROOT/h3-public-runtime-needs.json"\n'
            'if [ -e "$ARTIFACT_ROOT" ] || [ -L "$ARTIFACT_ROOT" ]; then\n'
            '  test -d "$ARTIFACT_ROOT"\n'
            '  test ! -L "$ARTIFACT_ROOT"\n'
            'else\n'
            '  mkdir -m 700 "$ARTIFACT_ROOT"\n'
            'fi\n'
            'test ! -e "$ROW_ROOT"\n'
            'test ! -L "$ROW_ROOT"\n'
            'test ! -e "$ROW1030"\n'
            'test ! -L "$ROW1030"\n'
            'test ! -e "$ROW1201"\n'
            'test ! -L "$ROW1201"\n'
            'test ! -e "$AGGREGATE_ROOT"\n'
            'test ! -L "$AGGREGATE_ROOT"\n'
            'test ! -e "$NEEDS_PATH"\n'
            'test ! -L "$NEEDS_PATH"\n'
            'mkdir -m 700 "$ROW_ROOT"\n'
            'mkdir -m 700 "$ROW1030"\n'
            'mkdir -m 700 "$ROW1201"\n'
            'mkdir -m 700 "$AGGREGATE_ROOT"\n'
            'for path in "$ROW_ROOT" "$ROW1030" "$ROW1201" "$AGGREGATE_ROOT"; do\n'
            '  test -d "$path"\n'
            '  test ! -L "$path"\n'
            '  test "$(stat -c \'%u:%g\' "$path")" = "$(id -u):$(id -g)"\n'
            '  test "$(stat -c \'%a\' "$path")" = 700\n'
            'done\n'
        ),
    }


def _expected_public_runtime_verify_step() -> dict[str, object]:
    return {
        "name": "Verify immutable identity and pinned image",
        "env": {
            "REVIEWED_SHA": "${{ github.sha }}",
            "TESTED_SHA": "${{ github.sha }}",
            "WORKFLOW_SHA": "${{ github.sha }}",
        },
        "run": """set -eu
test "$(command -v git)" = /usr/bin/git
git --version
TREE_OID="$(git rev-parse HEAD^{tree})"
export TREE_OID
test "$(git rev-parse HEAD)" = "$REVIEWED_SHA"
test "$(git rev-parse HEAD)" = "$TESTED_SHA"
test "$(git rev-parse HEAD)" = "$WORKFLOW_SHA"
test "$(git rev-parse HEAD^{tree})" = "$TREE_OID"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
docker pull "$H3_PUBLIC_RUNTIME_IMAGE_REFERENCE"
test "$(docker image inspect --format '{{.Id}}' "$H3_PUBLIC_RUNTIME_IMAGE_REFERENCE")" = "$H3_PUBLIC_RUNTIME_IMAGE_CONFIG_DIGEST"
docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$H3_PUBLIC_RUNTIME_IMAGE_REFERENCE" | grep -F -- "$H3_PUBLIC_RUNTIME_IMAGE_MANIFEST_DIGEST" >/dev/null
""",
    }


def _expected_public_runtime_row_step(row_id: str) -> dict[str, object]:
    target = row_id.removeprefix("h3-public-")
    return {
        "name": f"Compile, link, extract, and inspect {target}",
        "env": {
            "REVIEWED_SHA": "${{ github.sha }}",
            "TESTED_SHA": "${{ github.sha }}",
            "WORKFLOW_SHA": "${{ github.sha }}",
            "RUN_ID": "${{ github.run_id }}",
            "RUN_ATTEMPT": "${{ github.run_attempt }}",
            "SLLM_H3_NETWORK_DISABLED": "1",
        },
        "run": _expected_public_runtime_row_run(row_id),
    }


def _expected_public_runtime_needs_step() -> dict[str, object]:
    return {
        "name": "Prepare exact public-runtime needs input",
        "if": "${{ always() }}",
        "run": (
            "set -eu\n"
            'NEEDS_PATH="$GITHUB_WORKSPACE/.local-artifacts/h3-public-runtime-needs.json"\n'
            'test ! -e "$NEEDS_PATH"\n'
            'test ! -L "$NEEDS_PATH"\n'
            "umask 077\n"
            'printf \'%s\\n\' \'{"state":"PASS","rows":["h3-public-gfx1030","h3-public-gfx1201"]}\' > "$NEEDS_PATH"\n'
            'test -f "$NEEDS_PATH"\n'
            'test ! -L "$NEEDS_PATH"\n'
            'test "$(stat -c \'%u:%g\' "$NEEDS_PATH")" = "$(id -u):$(id -g)"\n'
            'test "$(stat -c \'%a\' "$NEEDS_PATH")" = 600\n'
        ),
    }


def _expected_public_runtime_cleanup_step() -> dict[str, object]:
    return {
        "name": "Cleanup generated public-H3 rows and needs",
        "if": "${{ always() }}",
        "run": """set -eu
ARTIFACT_ROOT="$GITHUB_WORKSPACE/.local-artifacts"
ROW1030="$ARTIFACT_ROOT/h3-public-runtime/h3-public-gfx1030"
ROW1201="$ARTIFACT_ROOT/h3-public-runtime/h3-public-gfx1201"
NEEDS_PATH="$ARTIFACT_ROOT/h3-public-runtime-needs.json"
AGGREGATE_ROOT="$ARTIFACT_ROOT/h3-public-runtime-aggregate"
for path in "$ROW1030" "$ROW1201" "$NEEDS_PATH"; do
  test ! -L "$path"
done
if [ -e "$ROW1030" ]; then
  test -d "$ROW1030"
  rm -rf -- "$ROW1030"
fi
if [ -e "$ROW1201" ]; then
  test -d "$ROW1201"
  rm -rf -- "$ROW1201"
fi
if [ -e "$NEEDS_PATH" ]; then
  test -f "$NEEDS_PATH"
  rm -f -- "$NEEDS_PATH"
fi
test ! -e "$ROW1030"
test ! -L "$ROW1030"
test ! -e "$ROW1201"
test ! -L "$ROW1201"
test ! -e "$NEEDS_PATH"
test ! -L "$NEEDS_PATH"
test -f "$AGGREGATE_ROOT/aggregate.json"
test ! -L "$AGGREGATE_ROOT/aggregate.json"
test -f "$AGGREGATE_ROOT/aggregate.json.sha256"
test ! -L "$AGGREGATE_ROOT/aggregate.json.sha256"
""",
    }


def _expected_public_runtime_aggregate_steps() -> list[dict[str, object]]:
    return [
        _expected_public_runtime_checkout_step(),
        _expected_public_runtime_prepare_step(),
        _expected_public_runtime_verify_step(),
        _expected_public_runtime_row_step("h3-public-gfx1030"),
        _expected_public_runtime_row_step("h3-public-gfx1201"),
        _expected_public_runtime_needs_step(),
        {
            "name": "Aggregate exactly two public-runtime PASS rows locally",
            "if": "${{ always() }}",
            "run": H3_PUBLIC_RUNTIME_AGGREGATE_COMMAND,
        },
        {
            "name": "Upload JSON aggregate only",
            "if": "${{ success() }}",
            "uses": H3_PUBLIC_RUNTIME_ACTIONS["upload"],
            "with": {
                "name": "h3-public-runtime-aggregate",
                "path": ".local-artifacts/h3-public-runtime-aggregate/aggregate.json\n"
                ".local-artifacts/h3-public-runtime-aggregate/aggregate.json.sha256\n",
                "if-no-files-found": "error",
                "retention-days": 7,
            },
        },
        _expected_public_runtime_cleanup_step(),
    ]


def workflow_documents() -> list[tuple[Path, dict[str, object]]]:
    try:
        import yaml
    except ImportError as exc:
        raise ContractError(f"workflow YAML dependency missing: {exc}") from exc
    directory = ROOT / ".github/workflows"
    if not directory.exists():
        raise ContractError(".github/workflows is missing")
    paths = sorted(list(directory.glob("*.yml")) + list(directory.glob("*.yaml")))
    if not paths:
        raise ContractError("no workflow YAML files found")
    documents: list[tuple[Path, dict[str, object]]] = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as stream:
                document = yaml.safe_load(stream)
        except Exception as exc:  # YAML parser and I/O errors are H0 failures.
            raise ContractError(f"{path.relative_to(ROOT)}: YAML parse error: {exc}") from exc
        if not isinstance(document, dict) or not isinstance(document.get("jobs"), dict):
            raise ContractError(f"{path.relative_to(ROOT)}: workflow has no jobs object")
        documents.append((path, document))
    return documents


def _validate_action_pins(path: Path, jobs: dict[str, object]) -> None:
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            raise ContractError(f"{path.relative_to(ROOT)}: job {job_id} is not an object")
        if "continue-on-error" in job:
            raise ContractError(f"{path.relative_to(ROOT)}: job {job_id} uses prohibited continue-on-error")
        for step in job.get("steps", []):
            if not isinstance(step, dict):
                raise ContractError(f"{path.relative_to(ROOT)}: job {job_id} has non-object step")
            if "continue-on-error" in step:
                raise ContractError(f"{path.relative_to(ROOT)}: job {job_id} uses prohibited continue-on-error")
            uses = step.get("uses")
            if isinstance(uses, str) and "/" in uses and not SHA40.search(uses):
                raise ContractError(f"{path.relative_to(ROOT)}: action is not pinned to a full SHA: {uses}")


def _validate_exact_public_runtime_actions(path: Path, jobs: dict[str, object]) -> None:
    expected = {
        "h3-public-runtime": [H3_PUBLIC_RUNTIME_ACTIONS["checkout"], H3_PUBLIC_RUNTIME_ACTIONS["upload"]],
    }
    for job_id, expected_uses in expected.items():
        job = jobs[job_id]
        assert isinstance(job, dict)
        steps = job.get("steps")
        if not isinstance(steps, list):
            raise ContractError(f"{path.relative_to(ROOT)}: {job_id} has no steps")
        actual_uses = [step.get("uses") for step in steps if isinstance(step, dict) and "uses" in step]
        if actual_uses != expected_uses:
            raise ContractError(f"{path.relative_to(ROOT)}: {job_id} action identities/order are not exact")


def _require_trigger(path: Path, document: dict[str, object]) -> None:
    if "on" not in document and True not in document:
        raise ContractError(f"{path.relative_to(ROOT)}: workflow trigger `on` is missing")


def validate_host_workflow(path: Path, document: dict[str, object]) -> list[str]:
    jobs = document["jobs"]
    assert isinstance(jobs, dict)
    if set(jobs) != HOST_WORKFLOW_JOBS:
        raise ContractError(f"{path.relative_to(ROOT)}: host jobs must be exactly {sorted(HOST_WORKFLOW_JOBS)}")
    if "name" not in document or not isinstance(document["name"], str):
        raise ContractError(f"{path.relative_to(ROOT)}: workflow name is missing")
    _require_trigger(path, document)
    warnings: list[str] = []
    for row_id in ("h0", "h1", "h2"):
        job = jobs[row_id]
        if not isinstance(job, dict) or not isinstance(job.get("steps"), list) or not job.get("runs-on"):
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} has invalid runner/steps structure")
        if not isinstance(job.get("timeout-minutes"), int) or job["timeout-minutes"] <= 0:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} has invalid timeout")
        runs = [step.get("run", "") for step in job["steps"] if isinstance(step, dict)]
        joined = "\n".join(value for value in runs if isinstance(value, str))
        if "ci/tools/run_host_suite.py --row " not in joined:
            warnings.append(f"{row_id}: workflow run command is not the canonical ci/tools/run_host_suite.py --row form")
        if "--row-id" in joined:
            warnings.append(f"{row_id}: workflow uses legacy --row-id alias")
        if "ci/run_host_suite.py" in joined:
            warnings.append(f"{row_id}: workflow path ci/run_host_suite.py does not match current ci/tools/run_host_suite.py")
        if "sidecar.json" in "\n".join(str(step.get("with", {}).get("path", "")) for step in job["steps"] if isinstance(step, dict) and isinstance(step.get("with"), dict)):
            warnings.append(f"{row_id}: workflow uploads sidecar.json, but runner emits report.json.sha256")
    aggregate = jobs["host-required"]
    if not isinstance(aggregate, dict) or aggregate.get("needs") != ["h0", "h1", "h2"] or aggregate.get("if") != "${{ always() }}":
        raise ContractError(f"{path.relative_to(ROOT)}: host-required needs/always structure is invalid")
    aggregate_runs = [step.get("run", "") for step in aggregate.get("steps", []) if isinstance(step, dict)]
    aggregate_text = "\n".join(value for value in aggregate_runs if isinstance(value, str))
    if "ci/tools/aggregate_host_results.py" not in aggregate_text:
        warnings.append("host-required: workflow path is not current ci/tools/aggregate_host_results.py")
    if "--needs-json" not in aggregate_text or "--artifact-dir" not in aggregate_text or "--output-dir" not in aggregate_text:
        warnings.append("host-required: aggregate invocation is not the canonical needs/artifact/output CLI")
    _validate_action_pins(path, jobs)
    return warnings


def validate_h3_workflow(path: Path, document: dict[str, object]) -> list[str]:
    """Validate the independent non-required H3 workflow profile."""

    jobs = document["jobs"]
    assert isinstance(jobs, dict)
    if set(jobs) != H3_WORKFLOW_JOBS:
        raise ContractError(f"{path.relative_to(ROOT)}: H3 jobs must be exactly {sorted(H3_WORKFLOW_JOBS)}")
    if document.get("name") != "h3-compile-only (non-required)":
        raise ContractError(f"{path.relative_to(ROOT)}: H3 workflow must be explicitly non-required")
    _require_trigger(path, document)
    if document.get("permissions") != {"contents": "read"}:
        raise ContractError(f"{path.relative_to(ROOT)}: H3 workflow permissions must be contents:read")
    for target in ("gfx1030", "gfx1201"):
        row_id = f"h3-{target}"
        job = jobs[row_id]
        if not isinstance(job, dict) or job.get("runs-on") != "ubuntu-24.04" or job.get("timeout-minutes") != 15:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} must use the GitHub-hosted 15-minute runner")
        if job.get("permissions") != {"contents": "read"}:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} permissions are not contents:read")
        steps = job.get("steps")
        if not isinstance(steps, list):
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} has no steps")
        run_text = "\n".join(step.get("run", "") for step in steps if isinstance(step, dict) and isinstance(step.get("run"), str))
        if "docker pull \"$H3_IMAGE_REFERENCE\"" not in run_text or "docker image inspect" not in run_text or "docker run --rm --network none" not in run_text:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} does not pull, inspect, and run the pinned image with network disabled")
        mount_specs = re.findall(r'--mount\s+"([^"]+)"', run_text)
        workspace_mount = "type=bind,src=$GITHUB_WORKSPACE,dst=/workspace,readonly"
        output_mount = f"type=bind,src=$GITHUB_WORKSPACE/.local-artifacts/h3/{row_id},dst=/output"
        allowed_mounts = {workspace_mount, output_mount, H3_GIT_HELPER_MOUNT}
        old_output_mount = f"{output_mount},rw"
        readonly_output_mount = f"{output_mount},readonly"
        if len(re.findall(r'(?<!\S)--mount(?:\s|=)', run_text)) != len(allowed_mounts) or set(mount_specs) != allowed_mounts:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} has an unexpected Docker mount")
        docker_volume_lines = (
            line for line in run_text.splitlines()
            if "docker run" in line or line.lstrip().startswith(("-v", "--volume"))
        )
        if any(re.search(r'(?<!\S)(?:--volume|-v)(?:\s|=)', line) for line in docker_volume_lines):
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} uses an unapproved Docker volume mount")
        if old_output_mount in mount_specs or any(",rw" in spec for spec in mount_specs):
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} uses the invalid legacy `,rw` output mount syntax")
        if readonly_output_mount in mount_specs:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} output mount must be writable and must not use readonly")
        if mount_specs.count(workspace_mount) != 1:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} source workspace mount must be exactly type=bind with readonly")
        if mount_specs.count(output_mount) != 1:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} output mount must be exactly type=bind without readonly or rw")
        if mount_specs.count(H3_GIT_HELPER_MOUNT) != 1:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} Git helper mount must be exactly /usr/bin/git to /usr/local/bin/git, readonly")
        docker_pull_index = run_text.find('docker pull "$H3_IMAGE_REFERENCE"')
        if any(run_text.find(fragment) < 0 or run_text.find(fragment) > docker_pull_index for fragment in ("test \"$(command -v git)\" = /usr/bin/git", "git --version")):
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} must verify host Git before Docker")
        required_fragments = (
            H3_IMAGE_REFERENCE, "test \"$(command -v git)\" = /usr/bin/git", "git --version",
            "--user \"$(id -u):$(id -g)\"", workspace_mount,
            output_mount, "ci/tools/run_h3_compile.py", f"--row {row_id}",
            "--repo /workspace", "--output-dir /output", "--strict-ci", "--pinned-container",
            "--observed-image-reference", "--observed-image-config-digest", H3_IMAGE_CONFIG_DIGEST,
            "REVIEWED_SHA", "TESTED_SHA", "WORKFLOW_SHA", "SLLM_H3_NETWORK_DISABLED",
            "git config --global --add safe.directory /workspace", "output ownership",
        )
        for fragment in required_fragments:
            if fragment not in run_text and fragment not in str(document.get("env", {})):
                raise ContractError(f"{path.relative_to(ROOT)}: {row_id} is missing H3 boundary {fragment}")
        lowered = run_text.lower()
        for prohibited in ("continue-on-error", "rocminstall", "docker exec", "rocminfo", "hipcc --run", "./device-code-object", "--device", "--gpus", "/var/run/docker.sock", "/dev/kfd", "/dev/dri"):
            if prohibited in lowered:
                raise ContractError(f"{path.relative_to(ROOT)}: {row_id} contains prohibited execution/install form {prohibited}")
        upload_paths = "\n".join(str(step.get("with", {}).get("path", "")) for step in steps if isinstance(step, dict) and isinstance(step.get("with"), dict))
        if f".local-artifacts/h3/{row_id}/*" not in upload_paths:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} does not upload its private row output")

    aggregate = jobs["h3-aggregate"]
    if not isinstance(aggregate, dict) or aggregate.get("needs") != ["h3-gfx1030", "h3-gfx1201"] or aggregate.get("if") != "${{ always() }}":
        raise ContractError(f"{path.relative_to(ROOT)}: H3 aggregate needs/always structure is invalid")
    if aggregate.get("runs-on") != "ubuntu-24.04" or aggregate.get("timeout-minutes") != 2 or aggregate.get("permissions") != {"contents": "read"}:
        raise ContractError(f"{path.relative_to(ROOT)}: H3 aggregate runner/permissions are invalid")
    aggregate_runs = "\n".join(step.get("run", "") for step in aggregate.get("steps", []) if isinstance(step, dict) and isinstance(step.get("run"), str))
    for fragment in ("ci/tools/aggregate_h3_results.py", "--needs-json", "--artifact-dir", "--output-dir", "--strict-ci", "--run-id", "--run-attempt", "--expected-reviewed-sha", "--expected-tested-sha", "--expected-workflow-sha"):
        if fragment not in aggregate_runs:
            raise ContractError(f"{path.relative_to(ROOT)}: H3 aggregate is missing {fragment}")
    aggregate_paths = "\n".join(str(step.get("with", {}).get("path", "")) for step in aggregate.get("steps", []) if isinstance(step, dict) and isinstance(step.get("with"), dict))
    if ".local-artifacts/h3-aggregate/aggregate.json" not in aggregate_paths or ".local-artifacts/h3-aggregate/aggregate.json.sha256" not in aggregate_paths:
        raise ContractError(f"{path.relative_to(ROOT)}: H3 aggregate report/sidecar is not uploaded")
    _validate_action_pins(path, jobs)
    return []


def validate_rmsnorm_h3_workflow(path: Path, document: dict[str, object]) -> list[str]:
    """Validate the dedicated, non-required RMSNorm compile-only workflow."""

    normalized_keys = set(document)
    normalized_keys.discard(True)
    normalized_keys.discard("on")
    if normalized_keys != {"name", "permissions", "env", "jobs"}:
        raise ContractError(f"{path.relative_to(ROOT)}: RMSNorm workflow has unknown or missing top-level keys")
    if document.get("name") != H3_RMSNORM_WORKFLOW_NAME:
        raise ContractError(f"{path.relative_to(ROOT)}: RMSNorm workflow must be explicitly non-required")
    if _workflow_trigger(document) != H3_RMSNORM_WORKFLOW_TRIGGER:
        raise ContractError(f"{path.relative_to(ROOT)}: RMSNorm workflow trigger is not exact")
    if document.get("permissions") != {"contents": "read"}:
        raise ContractError(f"{path.relative_to(ROOT)}: RMSNorm workflow permissions must be contents:read")
    if document.get("env") != H3_RMSNORM_ENV:
        raise ContractError(f"{path.relative_to(ROOT)}: RMSNorm image environment is not exact")

    jobs = document.get("jobs")
    if not isinstance(jobs, dict) or set(jobs) != H3_RMSNORM_WORKFLOW_JOBS:
        raise ContractError(f"{path.relative_to(ROOT)}: RMSNorm jobs must be exactly {sorted(H3_RMSNORM_WORKFLOW_JOBS)}")
    job = jobs["h3-rmsnorm"]
    if not isinstance(job, dict):
        raise ContractError(f"{path.relative_to(ROOT)}: h3-rmsnorm job is not an object")
    if set(job) != {"runs-on", "timeout-minutes", "permissions", "steps"}:
        raise ContractError(f"{path.relative_to(ROOT)}: h3-rmsnorm job has unknown or missing keys")
    if job.get("runs-on") != "ubuntu-24.04" or job.get("timeout-minutes") != 15:
        raise ContractError(f"{path.relative_to(ROOT)}: h3-rmsnorm must use the GitHub-hosted 15-minute runner")
    if job.get("permissions") != {"contents": "read"}:
        raise ContractError(f"{path.relative_to(ROOT)}: h3-rmsnorm permissions are not contents:read")
    if job.get("steps") != _expected_rmsnorm_steps():
        raise ContractError(f"{path.relative_to(ROOT)}: RMSNorm workflow step objects/inputs are not exact")

    steps = job["steps"]
    assert isinstance(steps, list)
    run_text = "\n".join(
        step.get("run", "") for step in steps if isinstance(step, dict) and isinstance(step.get("run"), str)
    )
    if run_text.count("docker run --rm --network none") != 3:
        raise ContractError(f"{path.relative_to(ROOT)}: RMSNorm workflow must isolate all three containers")
    if run_text.count('--user "$(id -u):$(id -g)"') != 3:
        raise ContractError(f"{path.relative_to(ROOT)}: RMSNorm containers must run as the non-root runner user")
    if run_text.count("--strict-ci --pinned-container") != 2:
        raise ContractError(f"{path.relative_to(ROOT)}: both exact RMSNorm rows must use strict pinned-container mode")
    if run_text.count("--row h3-rmsnorm-gfx1030") != 1 or run_text.count("--row h3-rmsnorm-gfx1201") != 1:
        raise ContractError(f"{path.relative_to(ROOT)}: RMSNorm workflow must invoke exactly both target rows")
    if "test -d \"$RUN_ROOT/sllm-rmsnorm-h3-gfx1030-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}/h3-rmsnorm-gfx1030\"" not in run_text:
        raise ContractError(f"{path.relative_to(ROOT)}: aggregate does not require the gfx1030 row")
    if "test -d \"$RUN_ROOT/sllm-rmsnorm-h3-gfx1201-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}/h3-rmsnorm-gfx1201\"" not in run_text:
        raise ContractError(f"{path.relative_to(ROOT)}: aggregate does not require the gfx1201 row")

    expected_mounts = {
        'type=bind,src=$RUN_ROOT,dst=/tmp',
        'type=bind,src=$GITHUB_WORKSPACE,dst=/workspace,readonly',
        H3_GIT_HELPER_MOUNT,
    }
    for target in ("gfx1030", "gfx1201"):
        row_run = _expected_rmsnorm_row_run(target)
        if row_run not in run_text:
            raise ContractError(f"{path.relative_to(ROOT)}: {target} row invocation is not exact")
    mount_specs = re.findall(r'--mount\s+"([^"]+)"', run_text)
    if set(mount_specs) != expected_mounts or any(mount_specs.count(mount) != 3 for mount in expected_mounts):
        raise ContractError(f"{path.relative_to(ROOT)}: RMSNorm workflow mounts are not three private/read-only sets")
    if any(re.search(r"src=/tmp(?:[\",\s])", mount) for mount in mount_specs):
        raise ContractError(f"{path.relative_to(ROOT)}: RMSNorm workflow exposes a broad host /tmp source")
    if run_text.count('type=bind,src=$GITHUB_WORKSPACE,dst=/workspace,readonly') != 3:
        raise ContractError(f"{path.relative_to(ROOT)}: RMSNorm source checkout is not read-only")
    required_fragments = (
        'RUN_ROOT="$RUNNER_TEMP/sllm-rmsnorm-h3-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"',
        'test "$(git rev-parse HEAD)" = "$REVIEWED_SHA"',
        'test "$(git rev-parse HEAD)" = "$TESTED_SHA"',
        'test "$(git rev-parse HEAD)" = "$WORKFLOW_SHA"',
        'test "$(git rev-parse HEAD^{tree})" = "$TREE_OID"',
        'git status --porcelain=v1 --untracked-files=all',
        'docker image inspect --format',
        'RMSNORM_H3_IMAGE_CONFIG_DIGEST',
        'RMSNORM_H3_IMAGE_MANIFEST_DIGEST',
        'SLLM_H3_NETWORK_DISABLED',
        'ci/tools/run_rmsnorm_h3_compile.py',
        'ci/tools/aggregate_rmsnorm_h3_results.py',
        'rmsnorm-h3-aggregate.json',
        'rmsnorm-h3-aggregate.json.sha256',
        'rm -rf -- "$RUN_ROOT"',
    )
    for fragment in required_fragments:
        if fragment not in run_text:
            raise ContractError(f"{path.relative_to(ROOT)}: RMSNorm workflow is missing required boundary {fragment}")
    lowered = run_text.lower()
    for prohibited in (
        "--network bridge", "--network host", "--gpus", "--device", "/dev/kfd", "/dev/dri",
        "/var/run/docker.sock", "docker exec", "rocminfo", "hipcc --run", "rocminstall",
        "fake_hip", "emulation", "fallback", "chmod +x", ".elf", ".bin", "./device-code-object",
    ):
        if prohibited in lowered:
            raise ContractError(f"{path.relative_to(ROOT)}: RMSNorm workflow contains prohibited form {prohibited}")
    upload_steps = [step for step in steps if isinstance(step, dict) and step.get("uses")]
    if [step.get("uses") for step in upload_steps] != [H3_RMSNORM_ACTIONS["checkout"], H3_RMSNORM_ACTIONS["setup_python"], H3_RMSNORM_ACTIONS["upload"]]:
        raise ContractError(f"{path.relative_to(ROOT)}: RMSNorm action identities/order are not exact")
    upload = steps[8]
    assert isinstance(upload, dict)
    upload_path = upload.get("with", {}).get("path", "") if isinstance(upload.get("with"), dict) else ""
    if upload_path != ".local-artifacts/rmsnorm-h3-aggregate/rmsnorm-h3-aggregate.json\n.local-artifacts/rmsnorm-h3-aggregate/rmsnorm-h3-aggregate.json.sha256\n":
        raise ContractError(f"{path.relative_to(ROOT)}: only the aggregate JSON and sidecar may be uploaded")
    _validate_action_pins(path, jobs)
    return []


def _validate_h3_public_runtime_workflow_legacy(path: Path, document: dict[str, object]) -> list[str]:
    """Validate the independent public-runtime compile-only workflow profile."""

    jobs = document["jobs"]
    assert isinstance(jobs, dict)
    if set(jobs) != H3_PUBLIC_RUNTIME_WORKFLOW_JOBS:
        raise ContractError(
            f"{path.relative_to(ROOT)}: H3 public-runtime jobs must be exactly {sorted(H3_PUBLIC_RUNTIME_WORKFLOW_JOBS)}"
        )
    if document.get("name") != "h3-public-runtime-compile-only (non-required)":
        raise ContractError(f"{path.relative_to(ROOT)}: public-runtime workflow must be explicitly non-required")
    _require_trigger(path, document)
    if document.get("permissions") != {"contents": "read"}:
        raise ContractError(f"{path.relative_to(ROOT)}: public-runtime workflow permissions must be contents:read")
    if document.get("env") != H3_PUBLIC_RUNTIME_ENV:
        raise ContractError(f"{path.relative_to(ROOT)}: public-runtime image environment is not exact")

    for target in ("gfx1030", "gfx1201"):
        row_id = f"h3-public-{target}"
        job = jobs[row_id]
        if not isinstance(job, dict) or job.get("runs-on") != "ubuntu-24.04" or job.get("timeout-minutes") != 15:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} must use the GitHub-hosted 15-minute runner")
        if job.get("permissions") != {"contents": "read"}:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} permissions are not contents:read")
        steps = job.get("steps")
        if not isinstance(steps, list):
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} has no steps")
        if steps != _expected_public_runtime_row_steps(row_id):
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} step objects/inputs are not exact")
        expected_step_names = [name.format(target=target) for name in H3_PUBLIC_RUNTIME_ROW_STEP_NAMES]
        actual_step_names = [step.get("name") for step in steps if isinstance(step, dict)]
        if actual_step_names != expected_step_names:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} row step boundary is not exact")
        run_step = steps[2]
        if not isinstance(run_step, dict) or run_step.get("env") != {
            "REVIEWED_SHA": "${{ github.sha }}",
            "TESTED_SHA": "${{ github.sha }}",
            "WORKFLOW_SHA": "${{ github.sha }}",
            "RUN_ID": "${{ github.run_id }}",
            "RUN_ATTEMPT": "${{ github.run_attempt }}",
            "SLLM_H3_NETWORK_DISABLED": "1",
        }:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} runtime environment is not exact")
        run_text = run_step.get("run")
        if not isinstance(run_text, str) or run_text != _expected_public_runtime_row_run(row_id):
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} full row shell boundary is not exact")
        if '  "$H3_PUBLIC_RUNTIME_IMAGE_REFERENCE" /bin/bash ' not in run_text:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} Docker image operand is not the pinned variable")
        if (
            'docker pull "$H3_PUBLIC_RUNTIME_IMAGE_REFERENCE"' not in run_text
            or "docker image inspect" not in run_text
            or "docker run --rm --network none" not in run_text
        ):
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} does not pull, inspect, and run the pinned image with network disabled")
        mount_specs = re.findall(r'--mount\s+"([^"]+)"', run_text)
        workspace_mount = "type=bind,src=$GITHUB_WORKSPACE,dst=/workspace,readonly"
        output_mount = f"type=bind,src=$GITHUB_WORKSPACE/.local-artifacts/h3-public-runtime/{row_id},dst=/output"
        git_mount = H3_GIT_HELPER_MOUNT
        allowed_mounts = {workspace_mount, output_mount, git_mount}
        if len(re.findall(r'(?<!\S)--mount(?:\s|=)', run_text)) != 3 or set(mount_specs) != allowed_mounts:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} has an unexpected Docker mount")
        docker_volume_lines = (
            line for line in run_text.splitlines()
            if "docker run" in line or line.lstrip().startswith(("-v", "--volume"))
        )
        if any(re.search(r'(?<!\S)(?:--volume|-v)(?:\s|=)', line) for line in docker_volume_lines):
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} uses an unapproved Docker volume mount")
        docker_pull_index = run_text.find('docker pull "$H3_PUBLIC_RUNTIME_IMAGE_REFERENCE"')
        for fragment in ('test "$(command -v git)" = /usr/bin/git', "git --version"):
            if run_text.find(fragment) < 0 or run_text.find(fragment) > docker_pull_index:
                raise ContractError(f"{path.relative_to(ROOT)}: {row_id} must verify host Git before Docker")
        required_fragments = (
            "H3_PUBLIC_RUNTIME_IMAGE_REFERENCE",
            "H3_PUBLIC_RUNTIME_IMAGE_CONFIG_DIGEST",
            "H3_PUBLIC_RUNTIME_IMAGE_MANIFEST_DIGEST",
            "test \"$(git rev-parse HEAD)\" = \"$REVIEWED_SHA\"",
            "git status --porcelain=v1 --untracked-files=all",
            "git rev-parse HEAD^{tree}",
            "--user \"$(id -u):$(id -g)\"",
            workspace_mount,
            output_mount,
            "ci/tools/run_h3_public_runtime_compile.py",
            f"--row {row_id}",
            "--repo /workspace",
            "--output-dir /output",
            "--strict-ci",
            "--pinned-container",
            "--observed-image-reference",
            "--observed-image-config-digest",
            "TREE_OID",
            "--tree-oid \"$TREE_OID\"",
            "REVIEWED_SHA",
            "TESTED_SHA",
            "WORKFLOW_SHA",
            "SLLM_H3_NETWORK_DISABLED",
            "git config --global --add safe.directory /workspace",
            "output ownership",
        )
        for fragment in required_fragments:
            if fragment not in run_text and fragment not in str(document.get("env", {})):
                raise ContractError(f"{path.relative_to(ROOT)}: {row_id} is missing public-runtime boundary {fragment}")
        lowered = run_text.lower()
        for prohibited in (
            "continue-on-error", "rocminstall", "docker exec", "rocminfo", "hipcc --run",
            "run_h3_compile.py", "aggregate_h3_results.py", "./device-code-object", "--device",
            "--gpus", "/var/run/docker.sock", "/dev/kfd", "/dev/dri", ".elf", "chmod +x", "model", "fallback",
        ):
            if prohibited in lowered:
                raise ContractError(f"{path.relative_to(ROOT)}: {row_id} contains prohibited form {prohibited}")
        upload_paths = "\n".join(
            str(step.get("with", {}).get("path", ""))
            for step in steps
            if isinstance(step, dict) and isinstance(step.get("with"), dict)
        )
        if f".local-artifacts/h3-public-runtime/{row_id}/**" not in upload_paths:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} does not upload its private row output")

    aggregate = jobs["h3-public-aggregate"]
    if not isinstance(aggregate, dict) or aggregate.get("needs") != ["h3-public-gfx1030", "h3-public-gfx1201"] or aggregate.get("if") != "${{ always() }}":
        raise ContractError(f"{path.relative_to(ROOT)}: H3 public-runtime aggregate needs/always structure is invalid")
    if aggregate.get("runs-on") != "ubuntu-24.04" or aggregate.get("timeout-minutes") != 2 or aggregate.get("permissions") != {"contents": "read"}:
        raise ContractError(f"{path.relative_to(ROOT)}: H3 public-runtime aggregate runner/permissions are invalid")
    aggregate_steps = aggregate.get("steps")
    if not isinstance(aggregate_steps, list) or aggregate_steps != _expected_public_runtime_aggregate_steps():
        raise ContractError(f"{path.relative_to(ROOT)}: H3 public-runtime aggregate step objects/inputs are not exact")
    aggregate_runs = "\n".join(
        step.get("run", "") for step in aggregate.get("steps", []) if isinstance(step, dict) and isinstance(step.get("run"), str)
    )
    needs_steps = [step for step in aggregate.get("steps", []) if isinstance(step, dict) and step.get("name") == "Prepare exact public-runtime needs input"]
    if len(needs_steps) != 1 or needs_steps[0].get("env") != {
        "GFX1030_RESULT": "${{ needs.h3-public-gfx1030.result }}",
        "GFX1201_RESULT": "${{ needs.h3-public-gfx1201.result }}",
    }:
        raise ContractError(f"{path.relative_to(ROOT)}: H3 public-runtime needs environment is not exact")
    needs_run = needs_steps[0].get("run", "")
    if not re.search(r'(?m)^set -eu\s+test "\$GFX1030_RESULT" = success\s+test "\$GFX1201_RESULT" = success\s+', needs_run):
        raise ContractError(f"{path.relative_to(ROOT)}: H3 public-runtime aggregate does not assert successful row needs")
    aggregate_steps = [step for step in aggregate.get("steps", []) if isinstance(step, dict) and step.get("name") == "Aggregate exactly two public-runtime PASS rows"]
    if len(aggregate_steps) != 1 or aggregate_steps[0].get("run", "").strip() != H3_PUBLIC_RUNTIME_AGGREGATE_COMMAND:
        raise ContractError(f"{path.relative_to(ROOT)}: H3 public-runtime aggregate invocation is not exact")
    lowered_aggregate = aggregate_runs.lower()
    for prohibited in ("aggregate_h3_results.py", "run_h3_public_runtime_compile.py", "./", "--gpus", "/dev/kfd", "/dev/dri", "model", "fallback"):
        if prohibited in lowered_aggregate:
            raise ContractError(f"{path.relative_to(ROOT)}: H3 public-runtime aggregate contains prohibited form {prohibited}")
    aggregate_paths = "\n".join(
        str(step.get("with", {}).get("path", ""))
        for step in aggregate.get("steps", [])
        if isinstance(step, dict) and isinstance(step.get("with"), dict)
    )
    if ".local-artifacts/h3-public-runtime-aggregate/aggregate.json" not in aggregate_paths or ".local-artifacts/h3-public-runtime-aggregate/aggregate.json.sha256" not in aggregate_paths:
        raise ContractError(f"{path.relative_to(ROOT)}: H3 public-runtime aggregate report/sidecar is not uploaded")
    _validate_action_pins(path, jobs)
    _validate_exact_public_runtime_actions(path, jobs)
    return []


def validate_h3_public_runtime_workflow(path: Path, document: dict[str, object]) -> list[str]:
    """Validate the exact one-job, serial, JSON-only public H3 profile."""

    normalized_keys = set(document)
    normalized_keys.discard(True)
    normalized_keys.discard("on")
    if normalized_keys != {"name", "permissions", "env", "jobs"}:
        raise ContractError(f"{path.relative_to(ROOT)}: public-runtime workflow has unknown or missing top-level keys")
    if document.get("name") != "h3-public-runtime-compile-only (non-required)":
        raise ContractError(f"{path.relative_to(ROOT)}: public-runtime workflow must be explicitly non-required")
    if _workflow_trigger(document) != H3_PUBLIC_RUNTIME_WORKFLOW_TRIGGER:
        raise ContractError(f"{path.relative_to(ROOT)}: public-runtime workflow trigger is not exact")
    if document.get("permissions") != {"contents": "read"}:
        raise ContractError(f"{path.relative_to(ROOT)}: public-runtime workflow permissions must be contents:read")
    if document.get("env") != H3_PUBLIC_RUNTIME_ENV:
        raise ContractError(f"{path.relative_to(ROOT)}: public-runtime image environment is not exact")

    jobs = document.get("jobs")
    if not isinstance(jobs, dict) or set(jobs) != H3_PUBLIC_RUNTIME_WORKFLOW_JOBS:
        raise ContractError(
            f"{path.relative_to(ROOT)}: public-runtime jobs must be exactly {sorted(H3_PUBLIC_RUNTIME_WORKFLOW_JOBS)}"
        )
    job = jobs["h3-public-runtime"]
    if not isinstance(job, dict):
        raise ContractError(f"{path.relative_to(ROOT)}: h3-public-runtime job is not an object")
    if set(job) != {"name", "runs-on", "timeout-minutes", "permissions", "steps"}:
        raise ContractError(f"{path.relative_to(ROOT)}: h3-public-runtime job has unknown or missing keys")
    if job.get("name") != "h3 public-runtime compile-only (non-required)":
        raise ContractError(f"{path.relative_to(ROOT)}: h3-public-runtime job name is not exact")
    if job.get("runs-on") != "ubuntu-24.04" or job.get("timeout-minutes") != 15:
        raise ContractError(f"{path.relative_to(ROOT)}: h3-public-runtime must use the GitHub-hosted 15-minute runner")
    if job.get("permissions") != {"contents": "read"}:
        raise ContractError(f"{path.relative_to(ROOT)}: h3-public-runtime permissions are not contents:read")
    steps = job.get("steps")
    if not isinstance(steps, list) or steps != _expected_public_runtime_aggregate_steps():
        raise ContractError(f"{path.relative_to(ROOT)}: public-runtime step objects/order are not exact")
    if [step.get("name") for step in steps if isinstance(step, dict)] != H3_PUBLIC_RUNTIME_STEP_NAMES:
        raise ContractError(f"{path.relative_to(ROOT)}: public-runtime step names/order are not exact")

    verify = steps[2]
    if not isinstance(verify, dict) or verify.get("run", "").count("docker pull ") != 1:
        raise ContractError(f"{path.relative_to(ROOT)}: public-runtime must pull the pinned image exactly once")
    verify_run = verify.get("run", "")
    if not isinstance(verify_run, str) or "docker image inspect" not in verify_run:
        raise ContractError(f"{path.relative_to(ROOT)}: public-runtime pinned image inspection is missing")

    expected_mounts = {
        "type=bind,src=$GITHUB_WORKSPACE,dst=/workspace,readonly",
        H3_GIT_HELPER_MOUNT,
    }
    for row_step, row_id in zip(steps[3:5], ("h3-public-gfx1030", "h3-public-gfx1201")):
        assert isinstance(row_step, dict)
        run_text = row_step["run"]
        assert isinstance(run_text, str)
        output_mount = f"type=bind,src=$GITHUB_WORKSPACE/.local-artifacts/h3-public-runtime/{row_id},dst=/output"
        mount_specs = re.findall(r'--mount\s+"([^"]+)"', run_text)
        if set(mount_specs) != expected_mounts | {output_mount}:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} Docker mounts are not exact and isolated")
        if len(mount_specs) != 3 or any(mount_specs.count(mount) != 1 for mount in expected_mounts | {output_mount}):
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} Docker mount multiplicity is not exact")
        if any(re.search(r"(?<!\\S)(?:--volume|-v)(?:\\s|=)", line) for line in run_text.splitlines()):
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} uses an unapproved Docker volume mount")
        if run_text.count("docker run --rm --network none") != 1:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} network isolation is not exact")
        if run_text.count('--user "$(id -u):$(id -g)"') != 1:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} must run as the non-root runner user")
        if "--strict-ci --pinned-container" not in run_text:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} strict CI/pinned-container boundary is missing")
        if "--row " + row_id not in run_text or "--repo /workspace" not in run_text or "--output-dir /output" not in run_text:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} exact runner argv is missing")
        if "--tree-oid \"$TREE_OID\"" not in run_text or "--run-id \"$RUN_ID\"" not in run_text:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} immutable run identity is missing")
        if "output ownership" not in run_text or "-type l" not in run_text:
            raise ContractError(f"{path.relative_to(ROOT)}: {row_id} output ownership/symlink proof is missing")
        lowered = run_text.lower()
        for prohibited in (
            "--network bridge", "--network host", "--gpus", "--device", "/dev/kfd", "/dev/dri",
            "/var/run/docker.sock", "docker exec", "rocminfo", "hipcc --run", "rocminstall",
            "chmod +x", ".elf", ".bin", "./device-code-object", "run_h3_compile.py", "model", "fallback",
        ):
            if prohibited in lowered:
                raise ContractError(f"{path.relative_to(ROOT)}: {row_id} contains prohibited form {prohibited}")

    needs = steps[5]
    aggregate = steps[6]
    upload = steps[7]
    cleanup = steps[8]
    if not isinstance(needs, dict) or needs.get("if") != "${{ always() }}":
        raise ContractError(f"{path.relative_to(ROOT)}: needs preparation must run with always()")
    if not isinstance(aggregate, dict) or aggregate.get("if") != "${{ always() }}" or aggregate.get("run", "").strip() != H3_PUBLIC_RUNTIME_AGGREGATE_COMMAND:
        raise ContractError(f"{path.relative_to(ROOT)}: local aggregate invocation/always boundary is not exact")
    aggregate_run = aggregate.get("run", "")
    if not isinstance(aggregate_run, str) or "docker" in aggregate_run.lower() or "download-artifact" in aggregate_run:
        raise ContractError(f"{path.relative_to(ROOT)}: aggregate must be local without row transport")
    if not isinstance(upload, dict) or upload.get("if") != "${{ success() }}" or upload.get("uses") != H3_PUBLIC_RUNTIME_ACTIONS["upload"]:
        raise ContractError(f"{path.relative_to(ROOT)}: aggregate upload is not success-only and pinned")
    upload_with = upload.get("with")
    if upload_with != {
        "name": "h3-public-runtime-aggregate",
        "path": ".local-artifacts/h3-public-runtime-aggregate/aggregate.json\n"
        ".local-artifacts/h3-public-runtime-aggregate/aggregate.json.sha256\n",
        "if-no-files-found": "error",
        "retention-days": 7,
    }:
        raise ContractError(f"{path.relative_to(ROOT)}: only aggregate JSON and SHA-256 sidecar may be uploaded")
    if not isinstance(cleanup, dict) or cleanup.get("if") != "${{ always() }}" or cleanup != _expected_public_runtime_cleanup_step():
        raise ContractError(f"{path.relative_to(ROOT)}: public-H3 cleanup is not exact or is too broad")
    cleanup_run = cleanup.get("run", "")
    if not isinstance(cleanup_run, str) or "aggregate.json" not in cleanup_run or "aggregate.json.sha256" not in cleanup_run:
        raise ContractError(f"{path.relative_to(ROOT)}: cleanup does not prove aggregate retention")
    for prohibited in (".local-artifacts/*", '"$ARTIFACT_ROOT"', "h3-public-runtime-aggregate"):
        if prohibited in cleanup_run and prohibited != "h3-public-runtime-aggregate":
            raise ContractError(f"{path.relative_to(ROOT)}: cleanup contains a broad or aggregate deletion target")
    _validate_action_pins(path, jobs)
    _validate_exact_public_runtime_actions(path, jobs)
    return []


def _expected_semantic_g1_workflow() -> dict[object, object]:
    """The entire reviewed G1 topology, not a collection of fragments."""

    prepare = """set -eu
RUN_ROOT=\"$RUNNER_TEMP/sllm-semantic-rmsnorm-g1-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}\"
test ! -e \"$RUN_ROOT\"
mkdir -m 700 \"$RUN_ROOT\"
printf 'RUN_ROOT=%s\\n' \"$RUN_ROOT\" >> \"$GITHUB_ENV\"
"""
    verify = """set -eu
test \"$(command -v git)\" = /usr/bin/git
/usr/bin/git --version
for key in remote.origin.url remote.origin.fetch branch.main.remote branch.main.merge; do
  env PATH=/usr/bin:/bin LC_ALL=C GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_SYSTEM=/dev/null GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_COUNT=0 GIT_NO_REPLACE_OBJECTS=1 \\
    /usr/bin/git --no-replace-objects config --local --unset-all \"$key\" || true
done
test \"$(/usr/bin/git --no-replace-objects rev-parse HEAD)\" = \"$REVIEWED_SHA\"
test \"$(/usr/bin/git --no-replace-objects rev-parse HEAD)\" = \"$TESTED_SHA\"
test \"$(/usr/bin/git --no-replace-objects rev-parse HEAD)\" = \"$WORKFLOW_SHA\"
test -n \"$(/usr/bin/git --no-replace-objects rev-parse --show-object-format=storage)\"
test -z \"$(/usr/bin/git --no-replace-objects for-each-ref --format='%(refname)' refs/replace)\"
test -z \"$(/usr/bin/git --no-replace-objects status --porcelain=v1 --untracked-files=all)\"
"""
    execute = """set -eu
test -n \"${RUN_ROOT:-}\"
test -d \"$RUN_ROOT\"
exec /usr/bin/env -i \\
  PATH=/usr/bin:/bin \\
  LC_CTYPE=C.UTF-8 \\
  HOME=\"$HOME\" \\
  CI=true \\
  GITHUB_ACTIONS=true \\
  GITHUB_SHA=\"$GITHUB_SHA\" \\
  GITHUB_WORKSPACE=\"$GITHUB_WORKSPACE\" \\
  RUNNER_TEMP=\"$RUNNER_TEMP\" \\
  RUN_ROOT=\"$RUN_ROOT\" \\
  REVIEWED_SHA=\"$REVIEWED_SHA\" \\
  TESTED_SHA=\"$TESTED_SHA\" \\
  WORKFLOW_SHA=\"$WORKFLOW_SHA\" \\
  GITHUB_RUN_ID=\"$GITHUB_RUN_ID\" \\
  GITHUB_RUN_ATTEMPT=\"$GITHUB_RUN_ATTEMPT\" \\
  GITHUB_WORKFLOW=semantic-rmsnorm-g1 \\
  /usr/bin/python3 -I -S -c '
import fcntl
import os
import subprocess
import sys
workspace = os.environ[\"GITHUB_WORKSPACE\"]
reviewed = os.environ[\"REVIEWED_SHA\"]
result = subprocess.run(
    [\"/usr/bin/git\", \"--no-replace-objects\", \"-C\", workspace, \"show\", f\"{reviewed}:ci/tools/orchestrate_rmsnorm_g1_evidence.py\"],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={\"PATH\": \"/usr/bin:/bin\", \"LC_ALL\": \"C\", \"GIT_CONFIG_NOSYSTEM\": \"1\", \"GIT_CONFIG_SYSTEM\": \"/dev/null\", \"GIT_CONFIG_GLOBAL\": \"/dev/null\", \"GIT_CONFIG_COUNT\": \"0\", \"GIT_NO_REPLACE_OBJECTS\": \"1\"},
    check=False,
)
if result.returncode != 0 or not result.stdout or len(result.stdout) > 4 * 1024 * 1024:
    raise SystemExit(\"cannot create the reviewed sealed semantic G1 controller source\")
fd = os.memfd_create(\"sllm-semantic-g1-controller\", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
offset = 0
while offset < len(result.stdout):
    offset += os.write(fd, result.stdout[offset:])
fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL)
os.set_inheritable(fd, True)
os.execve(
    \"/usr/bin/python3\",
    [\"/usr/bin/python3\", \"-I\", \"-S\", f\"/proc/self/fd/{fd}\", *sys.argv[2:]],
    {**os.environ, \"SLLM_G1_CONTROLLER_FD\": str(fd)},
)
' -- \\
  --artifact-root \"$RUN_ROOT/artifacts\" \\
  --output-dir \"$RUN_ROOT/rmsnorm-semantic-g1-aggregate-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}\" \\
  --run-id \"$GITHUB_RUN_ID\" \\
  --run-attempt \"$GITHUB_RUN_ATTEMPT\" \\
  --reviewed-sha \"$REVIEWED_SHA\" \\
  --tested-sha \"$TESTED_SHA\" \\
  --workflow-sha \"$WORKFLOW_SHA\"
"""
    cleanup = """set -eu
if [ -n \"${RUN_ROOT:-}\" ] && [ -d \"$RUN_ROOT\" ]; then
  rm -rf -- \"$RUN_ROOT\"
fi
"""
    evidence_path = SEMANTIC_G1_UPLOAD_PATH_TEXT
    candidate_env = {"REVIEWED_SHA": "${{ github.sha }}", "TESTED_SHA": "${{ github.sha }}", "WORKFLOW_SHA": "${{ github.sha }}"}
    return {
        "name": SEMANTIC_G1_WORKFLOW_NAME,
        True: SEMANTIC_G1_WORKFLOW_TRIGGER,
        "permissions": {"contents": "read"},
        "concurrency": {"group": "semantic-rmsnorm-g1-${{ github.ref }}", "cancel-in-progress": False},
        "jobs": {
            "semantic-rmsnorm-g1": {
                "runs-on": ["self-hosted", "sllm-semantic-g1", "rocm-7.14"],
                "timeout-minutes": 30,
                "permissions": {"contents": "read"},
                "steps": [
                    {"name": "Checkout immutable candidate", "uses": SEMANTIC_G1_ACTIONS["checkout"], "with": {"ref": "${{ github.sha }}", "fetch-depth": 0, "persist-credentials": False}},
                    {"name": "Prepare private controller directories", "run": prepare},
                    {"name": "Verify exact immutable candidate", "env": candidate_env, "run": verify},
                    {"name": "Run controller-owned serial gfx1030 and gfx1201 evidence", "env": candidate_env, "run": execute},
                    {"name": "Upload controller-validated evidence", "if": "${{ success() }}", "uses": SEMANTIC_G1_ACTIONS["upload"], "with": {"name": "semantic-rmsnorm-g1-evidence", "path": evidence_path, "if-no-files-found": "error", "retention-days": 7}},
                    {"name": "Remove private controller directories", "if": "${{ always() }}", "run": cleanup},
                ],
            }
        },
    }


def validate_semantic_g1_workflow(path: Path, document: dict[str, object]) -> list[str]:
    """Require byte and semantic equality for every protected G1 step."""

    if hashlib.sha256(path.read_bytes()).hexdigest() != SEMANTIC_G1_WORKFLOW_SHA256:
        raise ContractError(f"{path.name}: semantic G1 workflow bytes differ from the complete reviewed canonical form")
    jobs = document.get("jobs")
    try:
        steps = jobs["semantic-rmsnorm-g1"]["steps"]  # type: ignore[index]
        upload = steps[4]
        actual_paths = str(upload["with"]["path"]).splitlines()  # type: ignore[index]
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ContractError(f"{path.relative_to(ROOT)}: semantic G1 upload allowlist is missing") from exc
    if actual_paths != list(SEMANTIC_G1_UPLOAD_PATHS):
        raise ContractError(f"{path.relative_to(ROOT)}: semantic G1 upload must equal the explicit safe allowlist")
    if any("*" in item or "?" in item or any(marker in item.lower() for marker in ("trace", "slice", "weight", "device-code-object", "sllm-rmsnorm-g1-evidence", ".elf", ".so")) for item in actual_paths):
        raise ContractError(f"{path.relative_to(ROOT)}: semantic G1 upload allowlist contains an unsafe path")
    if document != _expected_semantic_g1_workflow():
        raise ContractError(f"{path.relative_to(ROOT)}: semantic G1 workflow topology/order/actions/env/argv/upload/cleanup are not exact")
    if not isinstance(jobs, dict):
        raise ContractError(f"{path.relative_to(ROOT)}: semantic G1 workflow jobs are malformed")
    _validate_action_pins(path, jobs)
    return []


def validate_workflow(path: Path, document: dict[str, object]) -> list[str]:
    """Dispatch to the host-required or independent H3 workflow profile."""

    if path.name == "h3-compile.yml" or document.get("name") == "h3-compile-only (non-required)":
        return validate_h3_workflow(path, document)
    if path.name == "h3-public-runtime-compile.yml" or document.get("name") == "h3-public-runtime-compile-only (non-required)":
        return validate_h3_public_runtime_workflow(path, document)
    if path.name == Path(H3_RMSNORM_WORKFLOW_PATH).name or document.get("name") == H3_RMSNORM_WORKFLOW_NAME:
        return validate_rmsnorm_h3_workflow(path, document)
    if path.name == Path(SEMANTIC_G1_WORKFLOW_PATH).name or document.get("name") == SEMANTIC_G1_WORKFLOW_NAME:
        return validate_semantic_g1_workflow(path, document)
    return validate_host_workflow(path, document)


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        schema_dir = ROOT / "ci/schema"
        schema_paths = sorted(schema_dir.glob("*.schema.json"))
        if not schema_paths:
            raise ContractError("no CI schemas found")
        if H3_AGGREGATE_SCHEMA not in {path.relative_to(ROOT).as_posix() for path in schema_paths}:
            raise ContractError("H3 aggregate schema is not registered for manifest validation")
        schema_names = {path.relative_to(ROOT).as_posix() for path in schema_paths}
        if not G1_SCHEMA_FILES.issubset(schema_names):
            raise ContractError("G1 report, aggregate, and dedicated artifact schemas are not registered")
        if not H3_PUBLIC_RUNTIME_SCHEMA_FILES.issubset(schema_names):
            raise ContractError("H3 public-runtime compile, artifact, and aggregate schemas are not registered")
        if not H3_RMSNORM_SCHEMA_FILES.issubset(schema_names):
            raise ContractError("RMSNorm H3 compile, artifact, report, and aggregate schemas are not registered")
        if not SEMANTIC_G1_SCHEMA_FILES.issubset(schema_names):
            raise ContractError("semantic RMSNorm G1 schemas are not registered")
        if not G2_SCHEMA_FILES.issubset(schema_names):
            raise ContractError("real-weight G2 schemas are not registered")
        if not P0_SCHEMA_FILES.issubset(schema_names):
            raise ContractError("RMSNorm P0 host contract schemas are not registered")
        if PHASE3_STAGE_A_EVIDENCE_PLAN_SCHEMA not in schema_names:
            raise ContractError("Phase 3 Stage A evidence-plan schema is not registered for manifest validation")
        if not (ROOT / H3_RMSNORM_MATRIX).is_file():
            raise ContractError("RMSNorm H3 matrix is not registered for manifest validation")
        if not (ROOT / SEMANTIC_G1_WORKFLOW_PATH).is_file():
            raise ContractError("semantic RMSNorm G1 workflow is not registered for manifest validation")
        if not (ROOT / G2_MATRIX).is_file() or not (ROOT / G2_TOLERANCE).is_file():
            raise ContractError("real-weight G2 matrix/tolerance is not registered for manifest validation")
        if (
            not (ROOT / P0_MATRIX).is_file()
            or not (ROOT / P0_REVIEW_POLICY).is_file()
            or not (ROOT / P0_PUBLIC_PATH_INPUTS).is_file()
        ):
            raise ContractError(
                "RMSNorm P0 matrix/review policy/public-path inputs are not registered for manifest validation"
            )
        if MODEL_LOCK_SCHEMA not in schema_names:
            raise ContractError("model-lock-v1 schema is not registered for manifest validation")
        if RUST_DEPENDENCY_SCHEMA not in schema_names:
            raise ContractError("Rust dependency policy schema is not registered for manifest validation")
        from jsonschema import Draft202012Validator, FormatChecker
        for path in schema_paths:
            schema = read_json(path)
            Draft202012Validator.check_schema(schema)
            if path.name == "hygiene-allowlist-v1.schema.json":
                allowlist = read_json(ROOT / "ci/policy/hygiene-allowlist-v1.json")
                errors.extend(f"{path.relative_to(ROOT)}: {error.message}" for error in Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(allowlist))
            if path.relative_to(ROOT).as_posix() == RUST_DEPENDENCY_SCHEMA:
                dependency_manifest = read_json(ROOT / RUST_DEPENDENCY_MANIFEST)
                errors.extend(
                    f"{path.relative_to(ROOT)}: {error.message}"
                    for error in Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(dependency_manifest)
                )
        load_manifests(ROOT)
        from validate_rmsnorm_h3_contracts import validate_static as validate_rmsnorm_h3_static

        validate_rmsnorm_h3_static(ROOT)
        from validate_h3_public_runtime_contracts import validate_static as validate_h3_public_runtime_static

        validate_h3_public_runtime_static(ROOT)
        from validate_model_lock import validate_lock_file
        from validate_rmsnorm_g2_contracts import validate_contracts as validate_g2_contracts
        from validate_rmsnorm_p0_contracts import validate_contracts as validate_p0_contracts

        validate_lock_file(ROOT / MODEL_LOCK_PATH, schema_path=ROOT / MODEL_LOCK_SCHEMA)
        validate_g2_contracts(ROOT)
        validate_p0_contracts(ROOT)
        if validate_matrix_main() != 0:
            errors.append("matrix/path/test/marker registry validation failed")
    except (ContractError, OSError, ValueError, ImportError, RuntimeError) as exc:
        errors.append(str(exc))
    try:
        for path, document in workflow_documents():
            warnings.extend(f"{path.relative_to(ROOT)}: {warning}" for warning in validate_workflow(path, document))
    except (ContractError, OSError, ValueError) as exc:
        errors.append(str(exc))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    for warning in warnings:
        print(f"workflow integration warning: {warning}")
    print("json/schema/manifests/workflow structure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
