#!/usr/bin/env python3
"""Assemble hash-only generic reasoning release evidence from measured cases."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "tools/validate-generic-reasoning-release.py"
SERVED_MODEL_VALIDATOR_PATH = ROOT / "tools/validate-served-model.py"
MAX_CASES_BYTES = 16 * 1024 * 1024
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
IMAGE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:+-]*@sha256:[0-9a-f]{64}\Z")
FORBIDDEN_KEYS = {
    "prompt",
    "response",
    "request_body",
    "response_body",
    "authorization",
    "api_key",
    "token",
    "conversation",
}


class EvidenceError(RuntimeError):
    """Raised when measured cases cannot be safely assembled."""


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError("input cases contain duplicate fields")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise EvidenceError("input cases contain a non-finite number")


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError("input cases must be a regular non-symlink file")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise EvidenceError("failed to read input cases") from error
    if len(raw) > MAX_CASES_BYTES:
        raise EvidenceError("input cases exceed their size bound")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError("input cases are not strict JSON") from error


def _scan_forbidden(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                raise EvidenceError(f"input cases contain forbidden field: {key}")
            _scan_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            _scan_forbidden(child)


def _hash_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"file is not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise EvidenceError(f"failed to hash file: {path}") from error
    return digest.hexdigest()


def _validate_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise EvidenceError(f"{label} is not a lowercase SHA-256")
    return value


def _validate_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise EvidenceError(f"{label} is not a lowercase Git commit")
    return value


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise EvidenceError("failed to resolve Git HEAD") from error
    if result.returncode != 0:
        raise EvidenceError("failed to resolve Git HEAD")
    return _validate_commit(result.stdout.strip(), "source_commit")


def _git_status() -> bytes:
    command = [
        "git",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude).rocprofv3",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise EvidenceError("failed to inspect Git worktree") from error
    if result.returncode != 0:
        raise EvidenceError("failed to inspect Git worktree")
    return bytes(result.stdout)


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise EvidenceError("served-model manifest is not an object")
    tokenizer = value.get("tokenizer")
    if not isinstance(tokenizer, dict) or not isinstance(tokenizer.get("root"), str):
        raise EvidenceError("served-model manifest has no tokenizer root")
    files = tokenizer.get("files")
    if not isinstance(files, dict) or not files:
        raise EvidenceError("served-model manifest has no tokenizer file map")
    root = Path(tokenizer["root"])
    if not root.is_absolute():
        root = path.parent / root
    return value, os.fspath(root)


def _validate_served_model_manifest(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "_ullm_generic_reasoning_served_model_validator",
        SERVED_MODEL_VALIDATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise EvidenceError("served-model validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        module.validation_summary(path)
    except Exception as error:
        raise EvidenceError("served-model manifest failed validation") from error


def _tokenizer_identity(manifest: dict[str, Any], root: Path) -> str:
    files = manifest["tokenizer"]["files"]
    digest = hashlib.sha256()
    root = root.resolve()
    for name in sorted(files):
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise EvidenceError("tokenizer file name is unsafe")
        path = (root / name).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise EvidenceError("tokenizer file escapes its root") from error
        observed = _hash_file(path)
        _validate_hash(files[name], f"manifest tokenizer file {name}")
        if observed != files[name]:
            raise EvidenceError(f"tokenizer file hash differs: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(observed))
    return digest.hexdigest()


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_ullm_generic_reasoning_release_preparer_validator", VALIDATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise EvidenceError("generic release validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_write(path: Path, document: dict[str, Any]) -> None:
    if path.is_symlink() or path.exists():
        raise EvidenceError("output evidence already exists or is a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(raw_path)
        with os.fdopen(descriptor, "w", encoding="ascii") as destination:
            json.dump(document, destination, ensure_ascii=True, allow_nan=False, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(
    cases_path: Path,
    manifest_path: Path,
    worker_path: Path,
    openwebui_image: str,
    active_promotion_source_commit: str,
    output_path: Path,
    *,
    status: str = "incomplete",
) -> dict[str, Any]:
    if status not in {"incomplete", "complete"}:
        raise EvidenceError("evidence status is invalid")
    _validate_commit(active_promotion_source_commit, "active_promotion_source_commit")
    if IMAGE_RE.fullmatch(openwebui_image) is None:
        raise EvidenceError("OpenWebUI image is not content-addressed")
    cases = _read_json(cases_path)
    _scan_forbidden(cases)
    if not isinstance(cases, list) or not cases:
        raise EvidenceError("measured cases must be a nonempty array")
    if len(cases) > 4096:
        raise EvidenceError("measured cases exceed their bound")
    _validate_served_model_manifest(manifest_path)
    manifest, tokenizer_root = _load_manifest(manifest_path)
    source_commit = _git_commit()
    status_raw = _git_status()
    worktree_clean = status_raw == b""
    if status == "complete" and not worktree_clean:
        raise EvidenceError("complete evidence requires a clean Git worktree")
    identity = {
        "manifest_sha256": _hash_file(manifest_path),
        "worker_binary_sha256": _hash_file(worker_path),
        "tokenizer_sha256": _tokenizer_identity(manifest, Path(tokenizer_root)),
        "openwebui_image": openwebui_image,
    }
    document = {
        "schema_version": "ullm.generic_reasoning_release_evidence.v1",
        "status": status,
        "production_activation_performed": False,
        "source_commit": source_commit,
        "active_promotion_source_commit": active_promotion_source_commit,
        "source_commit_aligned": source_commit == active_promotion_source_commit,
        "git_worktree_clean": worktree_clean,
        "git_worktree_status_sha256": hashlib.sha256(status_raw).hexdigest(),
        "identity": identity,
        "cases": cases,
    }
    validator = _load_validator()
    temporary = output_path.parent / f".{output_path.name}.validate"
    try:
        _atomic_write(temporary, document)
        report = validator.validate(temporary)
        if status == "complete" and report["gate_eligible"] is not True:
            raise EvidenceError("complete evidence is not production-gate eligible")
        if output_path.exists() or output_path.is_symlink():
            raise EvidenceError("output evidence already exists or is a symlink")
        os.replace(temporary, output_path)
        return document
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--worker-binary", required=True, type=Path)
    parser.add_argument("--openwebui-image", required=True)
    parser.add_argument("--active-promotion-source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--status", choices=("incomplete", "complete"), default="incomplete")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        document = prepare(
            args.cases,
            args.manifest,
            args.worker_binary,
            args.openwebui_image,
            args.active_promotion_source_commit,
            args.output,
            status=args.status,
        )
    except Exception as error:
        print(f"Generic reasoning release evidence preparation failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": document["schema_version"],
                "output": os.fspath(args.output.resolve()),
                "case_count": len(document["cases"]),
                "git_worktree_clean": document["git_worktree_clean"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
