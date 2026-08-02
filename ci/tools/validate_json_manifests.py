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

EXPECTED_JOBS = {"h0", "h1", "h2", "host-required"}
SHA40 = re.compile(r"@[0-9a-f]{40}$")


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


def validate_workflow(path: Path, document: dict[str, object]) -> list[str]:
    jobs = document["jobs"]
    assert isinstance(jobs, dict)
    if set(jobs) != EXPECTED_JOBS:
        raise ContractError(f"{path.relative_to(ROOT)}: jobs must be exactly {sorted(EXPECTED_JOBS)}")
    if "name" not in document or not isinstance(document["name"], str):
        raise ContractError(f"{path.relative_to(ROOT)}: workflow name is missing")
    # PyYAML 5/6 parses the YAML 1.1 key `on` as True; both spellings are accepted.
    if "on" not in document and True not in document:
        raise ContractError(f"{path.relative_to(ROOT)}: workflow trigger `on` is missing")
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
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            raise ContractError(f"{path.relative_to(ROOT)}: job {job_id} is not an object")
        for step in job.get("steps", []):
            if not isinstance(step, dict):
                raise ContractError(f"{path.relative_to(ROOT)}: job {job_id} has non-object step")
            uses = step.get("uses")
            if isinstance(uses, str) and "/" in uses and not SHA40.search(uses):
                raise ContractError(f"{path.relative_to(ROOT)}: action is not pinned to a full SHA: {uses}")
    return warnings


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        schema_dir = ROOT / "ci/schema"
        schema_paths = sorted(schema_dir.glob("*.schema.json"))
        if not schema_paths:
            raise ContractError("no CI schemas found")
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
