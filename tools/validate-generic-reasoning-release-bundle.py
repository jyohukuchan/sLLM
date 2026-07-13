#!/usr/bin/env python3
"""Validate the hash-only bundle that joins generic reasoning release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "ullm.generic_reasoning_release_bundle.v1"
VALIDATOR_SCHEMA_VERSION = "ullm.generic_reasoning_release_bundle_validator.v1"
GENERIC_VALIDATOR_PATH = ROOT / "tools/validate-generic-reasoning-release.py"
BROWSER_VALIDATOR_PATH = ROOT / "tools/validate-openwebui-reasoning-browser-smoke.py"
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
MAX_BUNDLE_BYTES = 1 * 1024 * 1024
MAX_COMPONENT_BYTES = 16 * 1024 * 1024
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


class ValidationError(ValueError):
    """Raised when a release bundle violates its contract."""


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValidationError("bundle JSON contains duplicate fields")
        value[key] = child
    return value


def _reject_constant(_value: str) -> None:
    raise ValidationError("bundle JSON contains a non-finite number")


def _json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ValidationError(f"{label} root is not an object")
    return value


def _read_json(path: Path, label: str, maximum: int) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"{label} must be a regular non-symlink file")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValidationError(f"failed to read {label}") from error
    if not raw or len(raw) > maximum:
        raise ValidationError(f"{label} exceeds its size bound")
    return _json_bytes(raw, label), raw


def _hash(value: Any, label: str) -> None:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise ValidationError(f"{label} is not a lowercase SHA-256")


def _commit(value: Any, label: str) -> None:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise ValidationError(f"{label} is not a lowercase Git commit")


def _text(value: Any, label: str, maximum: int = 512) -> None:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
        raise ValidationError(f"{label} is invalid")


def _scan_forbidden(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                raise ValidationError(f"bundle contains forbidden field: {key}")
            _scan_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            _scan_forbidden(child)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValidationError(f"validator is unavailable: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException as error:
        sys.modules.pop(name, None)
        raise ValidationError(f"validator could not be loaded: {path.name}") from error
    return module


def _resolve_component(bundle: Path, value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValidationError(f"{label} fields differ")
    relative = value["path"]
    _text(relative, f"{label}.path", 1024)
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValidationError(f"{label}.path is unsafe")
    candidate = bundle.parent / path
    if any(part.is_symlink() for part in (bundle.parent / part for part in path.parents if str(part) != ".")):
        raise ValidationError(f"{label}.path contains a symlink component")
    if candidate.is_symlink():
        raise ValidationError(f"{label}.path is a symlink")
    base = bundle.parent.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise ValidationError(f"{label}.path escapes the bundle directory") from error
    _hash(value["sha256"], f"{label}.sha256")
    if resolved.is_symlink() or not resolved.is_file():
        raise ValidationError(f"{label} file is unavailable")
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as source:
            remaining = MAX_COMPONENT_BYTES + 1
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
    except OSError as error:
        raise ValidationError(f"failed to hash {label}") from error
    if remaining == 0 or digest.hexdigest() != value["sha256"]:
        raise ValidationError(f"{label} SHA-256 differs")
    return resolved, digest.hexdigest()


def _validate_identity(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "manifest_sha256",
        "worker_binary_sha256",
        "tokenizer_sha256",
        "openwebui_image",
    }:
        raise ValidationError(f"{label} fields differ")
    for field in ("manifest_sha256", "worker_binary_sha256", "tokenizer_sha256"):
        _hash(value[field], f"{label}.{field}")
    _text(value["openwebui_image"], f"{label}.openwebui_image", 1024)
    if "@sha256:" not in value["openwebui_image"] or not HASH_RE.fullmatch(
        value["openwebui_image"].rsplit("@sha256:", 1)[1]
    ):
        raise ValidationError(f"{label}.openwebui_image is not content-addressed")
    return value


def _validate_promotion(
    evidence: dict[str, Any],
    receipt: dict[str, Any],
    manifest_hash: str,
    worker_hash: str,
    source_commit: str,
    evidence_path: Path,
    receipt_path: Path,
) -> None:
    if evidence.get("schema_version") != "ullm.aq4_resident_promotion_evidence.v1":
        raise ValidationError("promotion evidence schema differs")
    if evidence.get("verified") is not True or evidence.get("production_receipt_written") is not False:
        raise ValidationError("promotion evidence is not pre-receipt verified")
    if evidence.get("source_commit") != source_commit:
        raise ValidationError("promotion evidence source commit differs")
    if evidence.get("worker_binary_sha256") != worker_hash:
        raise ValidationError("promotion worker hash differs")
    bundle = evidence.get("ephemeral_bundle")
    if not isinstance(bundle, dict) or bundle.get("manifest_sha256") != manifest_hash:
        raise ValidationError("promotion manifest hash differs")
    if not isinstance(receipt, dict) or set(receipt) != {"schema_version", "source_commit", "evidence"}:
        raise ValidationError("promotion receipt fields differ")
    if receipt.get("schema_version") != "ullm.aq4_resident_promotion.v1" or receipt.get("source_commit") != source_commit:
        raise ValidationError("promotion receipt identity differs")
    reference = receipt["evidence"]
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValidationError("promotion receipt evidence reference differs")
    referenced, referenced_hash = _resolve_component(
        receipt_path,
        reference,
        "promotion receipt evidence",
    )
    if referenced != evidence_path.resolve() or referenced_hash != hashlib.sha256(evidence_path.read_bytes()).hexdigest():
        raise ValidationError("promotion receipt does not bind promotion evidence")


def validate(path: Path) -> dict[str, Any]:
    document, _raw = _read_json(path, "release bundle", MAX_BUNDLE_BYTES)
    _scan_forbidden(document)
    expected = {
        "schema_version",
        "status",
        "production_activation_performed",
        "source_commit",
        "active_promotion_source_commit",
        "identity",
        "artifacts",
        "rollback_target",
    }
    if set(document) != expected or document["schema_version"] != SCHEMA_VERSION:
        raise ValidationError("release bundle root fields differ")
    if document["status"] not in {"incomplete", "complete"}:
        raise ValidationError("release bundle status is invalid")
    if document["production_activation_performed"] is not False:
        raise ValidationError("release bundle claims activation")
    _commit(document["source_commit"], "source_commit")
    _commit(document["active_promotion_source_commit"], "active_promotion_source_commit")
    identity = _validate_identity(document["identity"], "identity")
    rollback = document["rollback_target"]
    if not isinstance(rollback, dict) or set(rollback) != {
        "manifest_sha256",
        "systemd_unit_sha256",
        "environment_sha256",
    }:
        raise ValidationError("rollback_target fields differ")
    for field in rollback:
        _hash(rollback[field], f"rollback_target.{field}")

    artifacts = document["artifacts"]
    names = {
        "release_evidence",
        "release_validator",
        "browser_evidence",
        "browser_validator",
        "promotion_evidence",
        "promotion_receipt",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != names:
        raise ValidationError("release bundle artifacts differ")
    files = {
        name: _resolve_component(path, artifacts[name], name)[0] for name in sorted(names)
    }
    release, _ = _read_json(files["release_evidence"], "release evidence", MAX_COMPONENT_BYTES)
    release_report, _ = _read_json(files["release_validator"], "release validator report", MAX_COMPONENT_BYTES)
    browser, _ = _read_json(files["browser_evidence"], "browser evidence", MAX_COMPONENT_BYTES)
    browser_report, _ = _read_json(files["browser_validator"], "browser validator report", MAX_COMPONENT_BYTES)
    promotion, _ = _read_json(files["promotion_evidence"], "promotion evidence", MAX_COMPONENT_BYTES)
    receipt, _ = _read_json(files["promotion_receipt"], "promotion receipt", MAX_COMPONENT_BYTES)

    if release.get("schema_version") != "ullm.generic_reasoning_release_evidence.v1":
        raise ValidationError("release evidence schema differs")
    if release.get("source_commit") != document["source_commit"] or release.get("active_promotion_source_commit") != document["active_promotion_source_commit"]:
        raise ValidationError("release evidence source identity differs")
    if release.get("identity") != identity:
        raise ValidationError("release evidence identity differs")
    generic_validator = _load_module(
        "_ullm_generic_reasoning_release_bundle_validator",
        GENERIC_VALIDATOR_PATH,
    )
    recomputed_release_report = generic_validator.validate(files["release_evidence"])
    if release_report != recomputed_release_report:
        raise ValidationError("release validator report differs from recomputation")
    if release_report.get("schema_version") != "ullm.generic_reasoning_release_validator.v1":
        raise ValidationError("release validator schema differs")
    release_gate_eligible = release_report.get("gate_eligible") is True
    browser_validator = _load_module(
        "_ullm_openwebui_reasoning_bundle_validator",
        BROWSER_VALIDATOR_PATH,
    )
    recomputed_browser_report = browser_validator.validate(files["browser_evidence"])
    if browser_report != recomputed_browser_report:
        raise ValidationError("browser validator report differs from recomputation")
    if browser_report.get("schema_version") != "ullm.openwebui.reasoning_browser_smoke_validator.v1":
        raise ValidationError("browser validator schema differs")
    browser_gate_eligible = browser_report.get("gate_eligible") is True
    if browser.get("schema_version") != "ullm.openwebui.reasoning_browser_smoke.v1":
        raise ValidationError("browser evidence schema differs")
    _validate_promotion(
        promotion,
        receipt,
        identity["manifest_sha256"],
        identity["worker_binary_sha256"],
        document["source_commit"],
        files["promotion_evidence"],
        files["promotion_receipt"],
    )
    reasons: list[str] = []
    if not release_gate_eligible:
        reasons.append("release validator gate is not eligible")
    if not browser_gate_eligible:
        reasons.append("browser validator gate is not eligible")
    if document["source_commit"] != document["active_promotion_source_commit"]:
        reasons.append("source commit is not aligned with active promotion source")
    if document["status"] != "complete":
        reasons.append("release bundle status is incomplete")
    return {
        "schema_version": VALIDATOR_SCHEMA_VERSION,
        "input_schema_version": SCHEMA_VERSION,
        "structurally_valid": True,
        "gate_eligible": not reasons,
        "source_commit": document["source_commit"],
        "artifact_count": len(files),
        "reasons": reasons,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = validate(args.bundle)
    except Exception as error:
        print(f"Generic reasoning release bundle validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0 if report["gate_eligible"] or not args.require_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
