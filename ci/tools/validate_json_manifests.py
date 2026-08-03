#!/usr/bin/env python3
"""Validate CI JSON schemas/manifests and the checked-in workflow structure."""

from __future__ import annotations

import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, ROOT, load_manifests, read_json  # noqa: E402
from validate_matrix import main as validate_matrix_main  # noqa: E402

HOST_WORKFLOW_JOBS = {"h0", "h1", "h2", "host-required"}
H3_WORKFLOW_JOBS = {"h3-gfx1030", "h3-gfx1201", "h3-aggregate"}
SHA40 = re.compile(r"@[0-9a-f]{40}$")
H3_IMAGE_REFERENCE = "docker.io/rocm/dev-ubuntu-24.04@sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7"
H3_IMAGE_CONFIG_DIGEST = "sha256:4c91c0d850e38a40fd669dd043ab42e9bad9a2b8a38e3f873c5a4eaced9f28cf"
H3_GIT_HELPER_MOUNT = "type=bind,src=/usr/bin/git,dst=/usr/local/bin/git,readonly"
H3_AGGREGATE_SCHEMA = "ci/schema/h3-aggregate-v1.schema.json"


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
            "REVIEWED_SHA", "TESTED_SHA", "WORKFLOW_SHA", "ULLM_H3_NETWORK_DISABLED",
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


def validate_workflow(path: Path, document: dict[str, object]) -> list[str]:
    """Dispatch to the host-required or independent H3 workflow profile."""

    if path.name == "h3-compile.yml" or document.get("name") == "h3-compile-only (non-required)":
        return validate_h3_workflow(path, document)
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
        from jsonschema import Draft202012Validator, FormatChecker
        for path in schema_paths:
            schema = read_json(path)
            Draft202012Validator.check_schema(schema)
            if path.name == "hygiene-allowlist-v1.schema.json":
                allowlist = read_json(ROOT / "ci/policy/hygiene-allowlist-v1.json")
                errors.extend(f"{path.relative_to(ROOT)}: {error.message}" for error in Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(allowlist))
        load_manifests(ROOT)
        if validate_matrix_main() != 0:
            errors.append("matrix/path/test/marker registry validation failed")
    except (ContractError, OSError, ValueError, ImportError) as exc:
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
