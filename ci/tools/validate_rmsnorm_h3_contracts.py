#!/usr/bin/env python3
"""Validate the dedicated RMSNorm H3 matrix, source identity, and row bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError as exc:  # pragma: no cover - CI must fail closed
    Draft202012Validator = None  # type: ignore[assignment]
    FormatChecker = None  # type: ignore[assignment]
    _JSONSCHEMA_IMPORT_ERROR = exc

try:
    from run_rmsnorm_h3_compile import (
        DEVICE_SYMBOL,
        E_FLAGS,
        LOGICAL_KERNEL,
        PUBLIC_ABI_SYMBOLS,
        PINNED_CONFIG,
        PINNED_IMAGE,
        ROOT,
        ROWS,
        SOURCE_SYMBOL_MAP,
        TARGETS,
        ContractError,
        canonical_bytes,
        read_json,
        sha256_file,
        sha256_json,
        validate_matrix,
    )
except ImportError:  # pragma: no cover - package import path
    from ci.tools.run_rmsnorm_h3_compile import (  # type: ignore[no-redef]
        DEVICE_SYMBOL,
        E_FLAGS,
        LOGICAL_KERNEL,
        PUBLIC_ABI_SYMBOLS,
        PINNED_CONFIG,
        PINNED_IMAGE,
        ROOT,
        ROWS,
        SOURCE_SYMBOL_MAP,
        TARGETS,
        ContractError,
        canonical_bytes,
        read_json,
        sha256_file,
        sha256_json,
        validate_matrix,
    )

SCHEMAS = {
    "compile": "ci/schema/rmsnorm-h3-compile-v1.schema.json",
    "artifact": "ci/schema/rmsnorm-h3-artifact-v1.schema.json",
    "report": "ci/schema/rmsnorm-h3-report-v1.schema.json",
    "aggregate": "ci/schema/rmsnorm-h3-aggregate-v1.schema.json",
}
EXPECTED_FILES = {
    "host": "host-bundle-{target}.elf",
    "host_sidecar": "host-bundle-{target}.elf.sha256",
    "device": "device-code-object-{target}.elf",
    "device_sidecar": "device-code-object-{target}.elf.sha256",
    "metadata": "rmsnorm-h3-artifact.json",
    "metadata_sidecar": "rmsnorm-h3-artifact.json.sha256",
    "report": "rmsnorm-h3-report.json",
    "report_sidecar": "rmsnorm-h3-report.json.sha256",
}


def _schema(repo: Path, name: str) -> dict[str, Any]:
    if Draft202012Validator is None:
        raise ContractError(f"jsonschema is required for dedicated H3 validation: {_JSONSCHEMA_IMPORT_ERROR}")
    path = repo / SCHEMAS[name]
    document = read_json(path)
    try:
        Draft202012Validator.check_schema(document)
    except Exception as exc:  # jsonschema raises several concrete classes
        raise ContractError(f"invalid dedicated RMSNorm schema {SCHEMAS[name]}: {exc}") from exc
    return document


def _validate_schema_document(repo: Path, document: Any, name: str, label: str) -> None:
    schema = _schema(repo, name)
    errors = sorted(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document), key=lambda error: list(error.path))
    if errors:
        raise ContractError(f"{label} fails {SCHEMAS[name]}: {errors[0].message}")


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _reject_symlinks(path: Path, label: str) -> None:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ContractError(f"{label} contains symlink component: {current}")


def _require_regular(path: Path, label: str) -> None:
    _reject_symlinks(path, label)
    if not path.exists() or not path.is_file() or path.is_symlink():
        raise ContractError(f"{label} is missing, symlinked, or not a regular file")


def _sidecar(path: Path, target: Path, label: str) -> str:
    _require_regular(target, f"{label} target")
    _require_regular(path, label)
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"{label} cannot be read") from exc
    if lines != [f"{sha256_file(target)}  {target.name}"]:
        raise ContractError(f"{label} is stale, malformed, or names a different file")
    return sha256_file(path)


def _strict_collection_root(path: Path) -> Path:
    root = _absolute(path)
    _reject_symlinks(root, "artifact root")
    if not root.is_dir() or root.is_symlink():
        raise ContractError("artifact root must be a regular non-symlink directory")
    return root


def _same_identity(value: dict[str, Any], expected_sha: str | None, expected_tree: str | None, label: str) -> None:
    for key in ("reviewed_sha", "tested_sha", "workflow_sha", "git_tree_oid"):
        if not isinstance(value.get(key), str) or len(value[key]) != 40 or any(char not in "0123456789abcdef" for char in value[key]):
            raise ContractError(f"{label} has an invalid {key}")
    if expected_sha is not None and any(value[key] != expected_sha for key in ("reviewed_sha", "tested_sha", "workflow_sha")):
        raise ContractError(f"{label} candidate SHA identity mismatch")
    if expected_tree is not None and value["git_tree_oid"] != expected_tree:
        raise ContractError(f"{label} tree identity mismatch")


def _validate_row(root: Path, row_id: str, repo: Path, expected_sha: str | None, expected_tree: str | None, strict: bool, matrix: dict[str, Any]) -> dict[str, Any]:
    target = row_id.rsplit("-", 1)[-1]
    row_root = root / f"h3-rmsnorm-{target}"
    if row_root.is_symlink() or not row_root.is_dir():
        raise ContractError(f"missing or symlinked row directory: {row_id}")
    names = {entry.name for entry in row_root.iterdir()}
    expected = {value.format(target=target) for value in EXPECTED_FILES.values()}
    if names != expected:
        raise ContractError(f"{row_id} has missing, duplicate, stale, or unknown output entries")
    host = row_root / EXPECTED_FILES["host"].format(target=target)
    device = row_root / EXPECTED_FILES["device"].format(target=target)
    metadata_path = row_root / EXPECTED_FILES["metadata"]
    report_path = row_root / EXPECTED_FILES["report"]
    host_sidecar_sha = _sidecar(row_root / EXPECTED_FILES["host_sidecar"].format(target=target), host, f"{row_id} host sidecar")
    device_sidecar_sha = _sidecar(row_root / EXPECTED_FILES["device_sidecar"].format(target=target), device, f"{row_id} device sidecar")
    metadata_sidecar_sha = _sidecar(row_root / EXPECTED_FILES["metadata_sidecar"], metadata_path, f"{row_id} metadata sidecar")
    report_sidecar_sha = _sidecar(row_root / EXPECTED_FILES["report_sidecar"], report_path, f"{row_id} report sidecar")
    metadata = read_json(metadata_path)
    report = read_json(report_path)
    _validate_schema_document(repo, metadata, "artifact", f"{row_id} metadata")
    _validate_schema_document(repo, report, "report", f"{row_id} report")
    _same_identity(metadata, expected_sha, expected_tree, f"{row_id} metadata")
    _same_identity(report, expected_sha, expected_tree, f"{row_id} report")
    target_name = target
    if metadata.get("row_id") != row_id or metadata.get("target") != target_name or report.get("row_id") != row_id or report.get("target") != target_name:
        raise ContractError(f"{row_id} has a row/target identity mismatch")
    if metadata.get("logical_kernel") != LOGICAL_KERNEL or metadata.get("device_symbol") != DEVICE_SYMBOL or report.get("logical_kernel") != LOGICAL_KERNEL or report.get("device_symbol") != DEVICE_SYMBOL:
        raise ContractError(f"{row_id} has a logical-kernel/device-symbol mismatch")
    if metadata.get("codegen", {}).get("e_flags") != E_FLAGS[target_name] or report.get("codegen", {}).get("e_flags") != E_FLAGS[target_name]:
        raise ContractError(f"{row_id} has an e_flags mismatch")
    if metadata.get("source_symbol_map") != SOURCE_SYMBOL_MAP or report.get("source_symbol_map") != SOURCE_SYMBOL_MAP:
        raise ContractError(f"{row_id} source-symbol evidence is not exact")
    if metadata.get("source_sets") != matrix.get("source_sets") or report.get("source_sets") != matrix.get("source_sets"):
        raise ContractError(f"{row_id} source-set evidence does not match the checked-in matrix")
    if metadata.get("source_sets") != report.get("source_sets"):
        raise ContractError(f"{row_id} source-set evidence differs between metadata and report")
    matrix_digest = sha256_json(matrix)
    workflow_digest = sha256_file(repo / matrix["workflow"]["path"])
    schema_digests = {name: sha256_file(repo / path) for name, path in SCHEMAS.items()}
    for document, label in ((metadata, "metadata"), (report, "report")):
        if document.get("matrix_manifest_sha256") != matrix_digest or document.get("workflow_file_sha256") != workflow_digest or document.get("schema_digests") != schema_digests:
            raise ContractError(f"{row_id} {label} matrix/workflow/schema digest is stale")
        if strict and (document.get("worktree_clean") is not True or document.get("evidence_mode", "required-ci") != "required-ci"):
            raise ContractError(f"{row_id} strict evidence does not prove a clean required-CI identity")
    expected_environment = {"image_reference": PINNED_IMAGE, "image_config_digest": PINNED_CONFIG, "platform": {"os": "linux", "architecture": "amd64"}, "pinned": True, "network_isolated": True}
    if strict:
        if metadata.get("container") != expected_environment or report.get("container") != expected_environment:
            raise ContractError(f"{row_id} strict container/network identity is not exact")
    elif metadata.get("container", {}).get("pinned") is not False or report.get("container", {}).get("pinned") is not False:
        raise ContractError(f"{row_id} local evidence claims an unobserved pinned container")
    for value in (metadata.get("scope"), report.get("scope")):
        if not isinstance(value, dict) or any(value.get(key) not in (False, True) for key in ("compile_only",)):
            raise ContractError(f"{row_id} scope is malformed")
        if value.get("compile_only") is not True or any(value.get(key) is not False for key in ("execution_attempted", "gpu_execution", "model_used", "network_used", "fallback_allowed", "fallback_used", "cpu_fallback_used", "fake_hip", "emulation")):
            raise ContractError(f"{row_id} contains forbidden execution/fallback/model scope")
    host_record = metadata["host_elf"]["file"]
    device_record = metadata["device_code_object"]["file"]
    if host_record["path"] != host.name or device_record["path"] != device.name:
        raise ContractError(f"{row_id} metadata names the wrong artifacts")
    if host_record["sha256"] != sha256_file(host) or device_record["sha256"] != sha256_file(device) or host_record["size_bytes"] != host.stat().st_size or device_record["size_bytes"] != device.stat().st_size:
        raise ContractError(f"{row_id} artifact content hash mismatch")
    if host_record["sidecar_sha256"] != host_sidecar_sha or device_record["sidecar_sha256"] != device_sidecar_sha:
        raise ContractError(f"{row_id} artifact sidecar digest mismatch")
    report_artifact = report["artifact"]
    if report_artifact["metadata_sha256"] != sha256_file(metadata_path) or report_artifact["metadata_sidecar_sha256"] != metadata_sidecar_sha or report_artifact["host_elf_sha256"] != host_record["sha256"] or report_artifact["device_code_object_sha256"] != device_record["sha256"] or report_artifact["host_elf_sidecar_sha256"] != host_sidecar_sha or report_artifact["device_code_object_sidecar_sha256"] != device_sidecar_sha:
        raise ContractError(f"{row_id} report artifact digests are stale or mismatched")
    if report_artifact["metadata"] != metadata_path.name or sha256_file(report_path) == "0" * 64 or report.get("process") != metadata.get("process") or report.get("scope") != metadata.get("scope"):
        raise ContractError(f"{row_id} report artifact record is malformed")
    expected_host_symbols = [{"name": name, "defined": True} for name in PUBLIC_ABI_SYMBOLS]
    if metadata["host_elf"]["public_symbols"] != expected_host_symbols or metadata["host_elf"]["stub_symbols"] != []:
        raise ContractError(f"{row_id} host ABI symbol evidence is not the exact public map")
    if metadata["host_elf"]["bundles"] != [f"hipv4-amdgcn-amd-amdhsa--{target_name}", "host-x86_64-unknown-linux-gnu-"] or metadata["host_elf"]["machine"] != "X86_64":
        raise ContractError(f"{row_id} host ELF bundle or machine evidence is not exact")
    device_evidence = metadata["device_code_object"]
    if device_evidence["target"] != target_name or device_evidence["e_flags"] != E_FLAGS[target_name] or device_evidence["code_object_version"] != "V6" or device_evidence["wavefront_size"] != 32 or device_evidence["features"] != {"xnack": "unsupported", "sramecc": "unsupported", "generic_processor_version": 0} or device_evidence["symbols"] != [{"name": DEVICE_SYMBOL, "defined": True}, {"name": DEVICE_SYMBOL + ".kd", "defined": True}]:
        raise ContractError(f"{row_id} device ELF identity/symbol evidence is not exact")
    return report


def validate_static(repo: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    for name, path in SCHEMAS.items():
        _require_regular(repo / path, f"dedicated schema {path}")
        _schema(repo, name)
    toolchain, matrix, rows = validate_matrix(repo)
    _validate_schema_document(repo, matrix, "compile", "RMSNorm matrix")
    return toolchain, matrix, rows


def validate_artifacts(repo: Path, artifact_root: Path, *, expected_sha: str | None, expected_tree: str | None, strict: bool) -> list[dict[str, Any]]:
    root = _strict_collection_root(artifact_root)
    if strict and (expected_sha is None or expected_tree is None):
        raise ContractError("strict artifact validation requires expected SHA and tree")
    expected_dirs = {f"h3-rmsnorm-{target}" for target in TARGETS}
    entries = list(root.iterdir())
    if {entry.name for entry in entries} != expected_dirs or any(entry.is_symlink() or not entry.is_dir() for entry in entries):
        raise ContractError("artifact collection has missing, duplicate, unknown, or symlinked row directories")
    _, matrix, _ = validate_static(repo)
    reports = [_validate_row(root, row_id, repo, expected_sha, expected_tree, strict, matrix) for row_id in ROWS]
    return reports


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", type=Path, default=ROOT)
    result.add_argument("--artifact-root", type=Path)
    result.add_argument("--expected-reviewed-sha")
    result.add_argument("--expected-tested-sha")
    result.add_argument("--expected-workflow-sha")
    result.add_argument("--tree-oid")
    result.add_argument("--strict-ci", action="store_true")
    result.add_argument("--non-strict-local", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        if bool(args.strict_ci) == bool(args.non_strict_local):
            raise ContractError("choose exactly one of --strict-ci or --non-strict-local")
        validate_static(_absolute(args.repo))
        if args.artifact_root is not None:
            expected_sha = args.expected_reviewed_sha or args.expected_tested_sha or args.expected_workflow_sha
            expected_tree = args.tree_oid
            if args.strict_ci:
                if not expected_sha or args.expected_tested_sha != expected_sha or args.expected_workflow_sha != expected_sha:
                    raise ContractError("strict artifact validation requires all three equal expected SHAs")
                commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True, capture_output=True, check=False).stdout.strip()
                tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=args.repo, text=True, capture_output=True, check=False).stdout.strip()
                status = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=args.repo, text=True, capture_output=True, check=False).stdout
                if commit != expected_sha or tree != expected_tree or status:
                    raise ContractError("strict artifact validation rejects dirty or stale checkout identity")
            validate_artifacts(_absolute(args.repo), _absolute(args.artifact_root), expected_sha=expected_sha, expected_tree=expected_tree, strict=args.strict_ci)
        print("RMSNorm H3 dedicated contracts: PASS")
        return 0
    except (ContractError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"RMSNorm H3 dedicated contracts: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
