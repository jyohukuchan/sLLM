#!/usr/bin/env python3
"""Validate the hash-only bundle that joins generic reasoning release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION_V1 = "ullm.generic_reasoning_release_bundle.v1"
SCHEMA_VERSION_V2 = "ullm.generic_reasoning_release_bundle.v2"
SCHEMA_VERSION = SCHEMA_VERSION_V1
VALIDATOR_SCHEMA_VERSION_V1 = "ullm.generic_reasoning_release_bundle_validator.v1"
VALIDATOR_SCHEMA_VERSION_V2 = "ullm.generic_reasoning_release_bundle_validator.v2"
VALIDATOR_SCHEMA_VERSION = VALIDATOR_SCHEMA_VERSION_V1
GENERIC_VALIDATOR_PATH = ROOT / "tools/validate-generic-reasoning-release.py"
BROWSER_VALIDATOR_PATH = ROOT / "tools/validate-openwebui-reasoning-browser-smoke.py"
SERVED_MODEL_VALIDATOR_PATH = ROOT / "tools/validate-served-model.py"
SQ8_PROMOTION_VALIDATOR_PATH = ROOT / "tools/sq8_serving_promotion.py"
SQ8_CAMPAIGN_VALIDATOR_PATH = ROOT / "tools/validate-sq8-openwebui-release.py"
CAMPAIGN_AUTHORIZATION_PATH = ROOT / "tools/served_model_campaign_authorization.py"
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
MAX_BUNDLE_BYTES = 1 * 1024 * 1024
MAX_COMPONENT_BYTES = 16 * 1024 * 1024
MAX_CAMPAIGN_FILE_BYTES = 64 * 1024 * 1024
MAX_CAMPAIGN_TOTAL_BYTES = 512 * 1024 * 1024
SQ8_MODEL_ID = "ullm-qwen3-14b-sq8"
SQ8_FORMAT_ID = "SQ8_0"
SQ8_WORKER_PROTOCOL = "ullm.worker.v2"
V1_ARTIFACT_NAMES = {
    "release_evidence",
    "release_validator",
    "browser_evidence",
    "browser_validator",
    "promotion_evidence",
    "promotion_receipt",
}
V2_ARTIFACT_NAMES = V1_ARTIFACT_NAMES | {
    "model_campaign_manifest",
    "model_campaign_evidence",
    "model_campaign_validator",
}
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
AUTHORIZATION_SCHEMA = (
    "ullm.served_model.v2_cross_model_campaign_authorization.v2"
)
CLAIM_SCHEMA = "ullm.served_model.v2_cross_model_campaign_claim.v2"
CLAIM_FIELDS = {
    "schema_version",
    "authorization_id",
    "authorization_path",
    "authorization_sha256",
    "claimed_at",
    "attempt",
    "max_attempts",
}
AUTHORIZED_CAMPAIGN_NAMES = {
    "aq4_reasoning_release",
    "aq4_reasoning_browser",
    "aq4_bundle",
    "sq8_full",
    "reasoning_release",
    "reasoning_browser",
}
SQ8_CAMPAIGN_NAMES = {"sq8_full", "reasoning_release", "reasoning_browser"}
OUTCOME_SELECTED_ARTIFACTS = {
    "SHA256SUMS",
    "active-manifest-binding.json",
    "browser-validator.json",
    "candidate-served-model.json",
    "model-identity.json",
    "release-validation.json",
    "summary.json",
    "validation.json",
    "active-manifest-observations.jsonl",
    "browser-evidence.json",
    "lifecycle.json",
    "resource-samples.jsonl",
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


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_read_regular(
    path: Path,
    label: str,
    maximum: int,
    *,
    require_immutable: bool = False,
) -> bytes:
    """Read one named regular file without accepting links or byte races."""

    if not path.is_absolute():
        path = path.absolute()
    flags = os.O_RDONLY | os.O_CLOEXEC
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValidationError("O_NOFOLLOW is required for bundle v2 validation")
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValidationError(f"failed to open {label}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise ValidationError(f"{label} exceeds its size bound")
        if require_immutable and (
            stat.S_IMODE(before.st_mode) != 0o444 or before.st_nlink != 1
        ):
            raise ValidationError(
                f"{label} must be an immutable mode-0444 single-link file"
            )
        raw = bytearray()
        while len(raw) <= maximum:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, maximum + 1 - len(raw)),
            )
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        try:
            named = path.lstat()
        except OSError as error:
            raise ValidationError(f"{label} disappeared while being read") from error
        if (
            len(raw) != before.st_size
            or len(raw) > maximum
            or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(named)
        ):
            raise ValidationError(f"{label} changed while being read")
        return bytes(raw)
    finally:
        os.close(descriptor)


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


def _resolve_component_v2(
    bundle: Path,
    value: Any,
    label: str,
) -> tuple[Path, str, bytes]:
    """Resolve and stable-read a v2 component below a canonical bundle root."""

    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValidationError(f"{label} fields differ")
    relative = value["path"]
    _text(relative, f"{label}.path", 1024)
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValidationError(f"{label}.path is unsafe")
    base_path = bundle.parent.absolute()
    try:
        base = base_path.resolve(strict=True)
    except OSError as error:
        raise ValidationError("release bundle directory is unavailable") from error
    if base != base_path:
        raise ValidationError("release bundle directory is not canonical")
    candidate = base / path
    current = base
    for part in path.parts:
        current /= part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise ValidationError(f"{label} file is unavailable") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValidationError(f"{label}.path contains a symlink component")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(base)
    except (OSError, ValueError) as error:
        raise ValidationError(f"{label}.path escapes the bundle directory") from error
    _hash(value["sha256"], f"{label}.sha256")
    raw = _stable_read_regular(resolved, label, MAX_COMPONENT_BYTES)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != value["sha256"]:
        raise ValidationError(f"{label} SHA-256 differs")
    return resolved, digest, raw


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
    gpu_preflight = evidence.get("gpu_exclusive_preflight")
    if not isinstance(gpu_preflight, dict) or set(gpu_preflight) != {
        "tool",
        "gpu_index",
        "positive_vram_processes",
    }:
        raise ValidationError("promotion GPU exclusivity preflight is missing")
    if (
        gpu_preflight.get("tool") != "rocm-smi --showpids --json"
        or gpu_preflight.get("gpu_index") != "1"
        or gpu_preflight.get("positive_vram_processes") != []
    ):
        raise ValidationError("promotion GPU exclusivity preflight failed")
    # The promotion runner must measure before the production receipt exists.
    # Its ephemeral manifest therefore contains a temporary receipt path and
    # cannot have the same byte hash as the final manifest. The final manifest
    # identity is independently bound by release/browser evidence and the
    # activation preflight; here we validate the promotion's source, worker,
    # and receipt binding instead of comparing two inherently different files.
    bundle = evidence.get("ephemeral_bundle")
    if not isinstance(bundle, dict) or not isinstance(bundle.get("manifest_sha256"), str):
        raise ValidationError("promotion ephemeral manifest identity is missing")
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


def _validate_v1(path: Path) -> dict[str, Any]:
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
    names = V1_ARTIFACT_NAMES
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
    if browser.get("schema_version") not in {
        "ullm.openwebui.reasoning_browser_smoke.v1",
        "ullm.openwebui.reasoning_browser_smoke.v2",
    }:
        raise ValidationError("browser evidence schema differs")
    _validate_promotion(
        promotion,
        receipt,
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


def _campaign_paths(
    files: dict[str, Path],
) -> tuple[Path, Path, Path, Path]:
    manifest = files["model_campaign_manifest"]
    evidence = files["model_campaign_evidence"]
    report = files["model_campaign_validator"]
    root = manifest.parent
    if (
        manifest.name != "SHA256SUMS"
        or evidence.name != "model-identity.json"
        or report.name != "release-validation.json"
        or evidence.parent != root
        or report.parent != root
    ):
        raise ValidationError("SQ8 model-campaign artifact locations differ")
    return root, manifest, evidence, report


def _copy_campaign_for_recomputation(
    campaign_root: Path,
    destination: Path,
    expected_files: set[str],
) -> None:
    """Create a private stable byte copy without the published validator report."""

    if not expected_files or "release-validation.json" in expected_files:
        raise ValidationError("SQ8 campaign validator file contract differs")
    expected_root_entries = {
        Path(relative).parts[0] for relative in expected_files
    } | {"release-validation.json"}
    try:
        observed_root_entries = {entry.name for entry in os.scandir(campaign_root)}
    except OSError as error:
        raise ValidationError("failed to enumerate SQ8 campaign") from error
    if observed_root_entries != expected_root_entries:
        raise ValidationError("SQ8 campaign root file set differs")
    browser_expected = {
        Path(relative).name
        for relative in expected_files
        if len(Path(relative).parts) == 2
        and Path(relative).parts[0] == "browser"
    }
    browser_path = campaign_root / "browser"
    try:
        browser_metadata = browser_path.lstat()
        browser_observed = {entry.name for entry in os.scandir(browser_path)}
    except OSError as error:
        raise ValidationError("failed to enumerate SQ8 campaign browser files") from error
    if (
        stat.S_ISLNK(browser_metadata.st_mode)
        or not stat.S_ISDIR(browser_metadata.st_mode)
        or browser_observed != browser_expected
    ):
        raise ValidationError("SQ8 campaign browser file set differs")

    destination.mkdir(mode=0o700)
    (destination / "browser").mkdir(mode=0o700)
    total = 0
    for relative in sorted(expected_files, key=lambda value: value.encode("utf-8")):
        source = campaign_root / relative
        raw = _stable_read_regular(
            source,
            f"SQ8 campaign file {relative}",
            MAX_CAMPAIGN_FILE_BYTES,
        )
        total += len(raw)
        if total > MAX_CAMPAIGN_TOTAL_BYTES:
            raise ValidationError("SQ8 campaign exceeds its aggregate size bound")
        target = destination / relative
        try:
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o600,
            )
            with os.fdopen(descriptor, "wb", buffering=0) as output:
                view = memoryview(raw)
                while view:
                    written = output.write(view)
                    if written is None or written <= 0:
                        raise ValidationError(
                            "failed to isolate SQ8 campaign evidence"
                        )
                    view = view[written:]
                output.flush()
                os.fsync(output.fileno())
        except OSError as error:
            raise ValidationError("failed to isolate SQ8 campaign evidence") from error


def _recompute_sq8_campaign_report(
    *,
    campaign_root: Path,
    campaign_identity: dict[str, Any],
    source_commit: str,
    worker_sha256: str,
    published_report_raw: bytes,
) -> dict[str, Any]:
    validator = _load_module(
        "_ullm_generic_reasoning_bundle_sq8_campaign_validator",
        SQ8_CAMPAIGN_VALIDATOR_PATH,
    )
    expected_files_value = getattr(validator, "BUNDLE_FILES_V2", None)
    if not isinstance(expected_files_value, set) or not all(
        isinstance(value, str) for value in expected_files_value
    ):
        raise ValidationError("SQ8 campaign validator contract is unavailable")
    expected_files = set(expected_files_value)
    served = campaign_identity["served_model_manifest"]
    claim = campaign_identity["campaign_authorization_claim"]
    try:
        with tempfile.TemporaryDirectory(
            prefix="ullm-sq8-bundle-v2-validation-"
        ) as temporary:
            isolated = Path(temporary) / "campaign"
            _copy_campaign_for_recomputation(
                campaign_root,
                isolated,
                expected_files,
            )
            recomputed_raw = validator.validate_full_release_no_publish(
                isolated,
                expected_commit=source_commit,
                expected_worker_binary_sha256=worker_sha256,
                repo_root=ROOT,
                expected_served_model_manifest_sha256=served["sha256"],
                expected_authorization_claim_sha256=claim["sha256"],
                expected_authorization_sha256=claim["authorization_sha256"],
            )
    except ValidationError:
        raise
    except Exception as error:
        raise ValidationError("SQ8 campaign independent recomputation failed") from error
    if (
        not isinstance(recomputed_raw, bytes)
        or recomputed_raw != published_report_raw
    ):
        raise ValidationError(
            "SQ8 campaign validator report differs from recomputation"
        )
    report = _json_bytes(recomputed_raw, "SQ8 campaign validator report")
    if (
        report.get("schema_version")
        != "ullm.sq8.openwebui_release.validation.v2"
        or report.get("release_status") != "complete"
    ):
        raise ValidationError("SQ8 campaign validator report is not complete v2")
    return report


def _validate_sq8_promotion_and_candidate(
    *,
    evidence_path: Path,
    receipt_path: Path,
    receipt_raw: bytes,
    candidate_path: Path,
    candidate_raw: bytes,
    source_commit: str,
    identity: dict[str, Any],
    campaign_identity: dict[str, Any],
) -> None:
    promotion_validator = _load_module(
        "_ullm_generic_reasoning_bundle_sq8_promotion_validator",
        SQ8_PROMOTION_VALIDATOR_PATH,
    )
    try:
        receipt, evidence = promotion_validator.validate_receipt(
            receipt_path,
            expected_evidence_path=evidence_path,
            verify_live_source=True,
        )
    except Exception as error:
        raise ValidationError("SQ8 promotion receipt validation failed") from error
    if (
        receipt.get("schema_version") != "ullm.sq8_serving_promotion.v1"
        or receipt.get("source_commit") != source_commit
        or evidence.get("schema_version")
        != "ullm.sq8_serving_promotion_evidence.v1"
        or not isinstance(evidence.get("worker"), dict)
        or evidence["worker"].get("sha256")
        != identity["worker_binary_sha256"]
    ):
        raise ValidationError("SQ8 promotion source or worker identity differs")

    candidate = _json_bytes(candidate_raw, "SQ8 candidate served-model manifest")
    public = candidate.get("public")
    format_value = candidate.get("format")
    worker = candidate.get("worker")
    promotion = candidate.get("promotion")
    if (
        candidate.get("schema_version") != "ullm.served_model.v2"
        or not isinstance(public, dict)
        or public.get("id") != SQ8_MODEL_ID
        or not isinstance(format_value, dict)
        or format_value.get("format_id") != SQ8_FORMAT_ID
        or not isinstance(worker, dict)
        or worker.get("protocol") != SQ8_WORKER_PROTOCOL
        or worker.get("binary_sha256") != identity["worker_binary_sha256"]
        or not isinstance(promotion, dict)
        or set(promotion) != {"source_commit", "receipt", "receipt_sha256"}
        or promotion.get("source_commit") != source_commit
        or promotion.get("receipt_sha256")
        != hashlib.sha256(receipt_raw).hexdigest()
    ):
        raise ValidationError("SQ8 candidate promotion identity differs")
    if hashlib.sha256(candidate_raw).hexdigest() != identity["manifest_sha256"]:
        raise ValidationError("SQ8 candidate manifest differs from bundle identity")
    campaign_served = campaign_identity["served_model_manifest"]
    if (
        campaign_served.get("sha256") != identity["manifest_sha256"]
        or campaign_served.get("worker_binary_sha256")
        != identity["worker_binary_sha256"]
        or campaign_served.get("promotion_source_commit") != source_commit
        or campaign_served.get("promotion_receipt_sha256")
        != hashlib.sha256(receipt_raw).hexdigest()
    ):
        raise ValidationError("SQ8 campaign candidate identity differs")

    receipt_value = promotion.get("receipt")
    if not isinstance(receipt_value, str):
        raise ValidationError("SQ8 candidate live promotion receipt path is invalid")
    live_receipt_path = Path(receipt_value)
    if not live_receipt_path.is_absolute():
        raise ValidationError("SQ8 candidate live promotion receipt path is not absolute")
    live_receipt_raw = _stable_read_regular(
        live_receipt_path,
        "SQ8 candidate live promotion receipt",
        MAX_COMPONENT_BYTES,
    )
    if live_receipt_raw != receipt_raw:
        raise ValidationError(
            "SQ8 candidate live promotion receipt differs from bundle component"
        )

    served_model_validator = _load_module(
        "_ullm_generic_reasoning_bundle_served_model_validator",
        SERVED_MODEL_VALIDATOR_PATH,
    )
    try:
        summary = served_model_validator.validation_summary(candidate_path)
    except Exception as error:
        raise ValidationError("SQ8 candidate served-model validation failed") from error
    if (
        summary.get("manifest_sha256") != identity["manifest_sha256"]
        or summary.get("model_id") != SQ8_MODEL_ID
        or summary.get("format_id") != SQ8_FORMAT_ID
        or not isinstance(summary.get("worker"), dict)
        or summary["worker"].get("protocol") != SQ8_WORKER_PROTOCOL
        or summary["worker"].get("binary_sha256")
        != identity["worker_binary_sha256"]
    ):
        raise ValidationError("SQ8 candidate served-model summary differs")


def _validate_generic_campaign_lineages(
    *,
    bundle_path: Path,
    campaign_root: Path,
    campaign_run_id: str,
    release: dict[str, Any],
    release_report: dict[str, Any],
    browser: dict[str, Any],
    browser_report: dict[str, Any],
    browser_path: Path,
    campaign_identity: dict[str, Any],
    identity: dict[str, Any],
    source_commit: str,
    rollback: dict[str, Any],
) -> dict[str, Any]:
    """Cross-bind both auxiliary campaigns to the one claimed SQ8 authorization."""

    release_lineage = release.get("campaign_lineage")
    browser_lineage = browser.get("campaign_lineage")
    if not isinstance(release_lineage, dict) or not isinstance(browser_lineage, dict):
        raise ValidationError("generic campaign lineage is missing")
    if (
        release_lineage.get("schema_version")
        != "ullm.served_model.campaign_lineage.v2"
        or browser_lineage.get("schema_version")
        != "ullm.served_model.campaign_lineage.v2"
    ):
        raise ValidationError("generic campaign lineage schema differs")
    claim = campaign_identity["campaign_authorization_claim"]
    if (
        release_lineage.get("claim") != claim
        or browser_lineage.get("claim") != claim
    ):
        raise ValidationError("generic campaign authorization claim differs")
    release_campaign = release_lineage.get("campaign")
    browser_campaign = browser_lineage.get("campaign")
    if (
        not isinstance(release_campaign, dict)
        or not isinstance(browser_campaign, dict)
        or release_campaign.get("name") != "reasoning_release"
        or browser_campaign.get("name") != "reasoning_browser"
        or release_campaign.get("run_id") == browser_campaign.get("run_id")
        or release_campaign.get("final_path") == browser_campaign.get("final_path")
    ):
        raise ValidationError("generic campaign run/output identity differs")
    release_artifacts = release_lineage.get("artifacts")
    browser_artifacts = browser_lineage.get("artifacts")
    if (
        not isinstance(release_artifacts, dict)
        or not isinstance(browser_artifacts, dict)
        or not isinstance(release_artifacts.get("candidate-served-model.json"), dict)
        or not isinstance(browser_artifacts.get("candidate-served-model.json"), dict)
        or release_artifacts["candidate-served-model.json"].get("sha256")
        != identity["manifest_sha256"]
        or browser_artifacts["candidate-served-model.json"].get("sha256")
        != identity["manifest_sha256"]
    ):
        raise ValidationError("generic campaign candidate identity differs")
    release_observations = release_lineage.get("observations")
    browser_observations = browser_lineage.get("observations")
    if (
        not isinstance(release_observations, dict)
        or not isinstance(browser_observations, dict)
        or release_observations.get("count") != 12
        or browser_observations.get("count") != 5
    ):
        raise ValidationError("generic campaign observation coverage differs")

    recomputed_release = release_report.get("campaign_lineage")
    recomputed_browser = browser_report.get("campaign_lineage")
    if (
        not isinstance(recomputed_release, dict)
        or not isinstance(recomputed_browser, dict)
        or recomputed_release.get("claim_sha256") != claim["sha256"]
        or recomputed_release.get("authorization_sha256")
        != claim["authorization_sha256"]
        or recomputed_browser.get("claim_sha256") != claim["sha256"]
        or recomputed_browser.get("authorization_sha256")
        != claim["authorization_sha256"]
        or recomputed_release.get("run_id") != release_campaign["run_id"]
        or recomputed_browser.get("run_id") != browser_campaign["run_id"]
    ):
        raise ValidationError("generic campaign recomputed lineage differs")

    try:
        bundle_root = bundle_path.parent.resolve(strict=True)
        release_root = Path(release_campaign["final_path"]).resolve(strict=True)
        browser_root = Path(browser_campaign["final_path"]).resolve(strict=True)
        release_root.relative_to(bundle_root)
        browser_root.relative_to(bundle_root)
    except (OSError, ValueError, TypeError) as error:
        raise ValidationError(
            "generic campaign final output is outside the bundle root"
        ) from error
    if browser_path.resolve(strict=True) != browser_root / "browser-evidence.json":
        raise ValidationError("browser bundle component is not its claimed final output")

    claim_path = Path(claim["path"])
    authorization_path = Path(claim["authorization_path"])
    try:
        if (
            claim_path.resolve(strict=True) != claim_path
            or authorization_path.resolve(strict=True) != authorization_path
        ):
            raise ValidationError("loaded campaign claim path is not canonical")
    except OSError as error:
        raise ValidationError("loaded campaign claim path is unavailable") from error
    claim_raw = _stable_read_regular(
        claim_path,
        "campaign authorization claim",
        MAX_COMPONENT_BYTES,
        require_immutable=True,
    )
    authorization_raw = _stable_read_regular(
        authorization_path,
        "campaign authorization",
        MAX_COMPONENT_BYTES,
        require_immutable=True,
    )
    if (
        len(claim_raw) != claim["bytes"]
        or hashlib.sha256(claim_raw).hexdigest() != claim["sha256"]
        or hashlib.sha256(authorization_raw).hexdigest()
        != claim["authorization_sha256"]
    ):
        raise ValidationError("loaded campaign claim bytes differ")
    claim_document = _json_bytes(claim_raw, "campaign authorization claim")
    authorization_document = _json_bytes(
        authorization_raw,
        "campaign authorization",
    )
    if (
        set(claim_document) != CLAIM_FIELDS
        or claim_document.get("schema_version") != CLAIM_SCHEMA
        or claim_document.get("authorization_path") != claim["authorization_path"]
        or claim_document.get("authorization_sha256")
        != claim["authorization_sha256"]
        or claim_document.get("attempt") != 1
        or claim_document.get("max_attempts") != 1
        or authorization_document.get("schema_version") != AUTHORIZATION_SCHEMA
        or claim_document.get("authorization_id")
        != authorization_document.get("authorization_id")
    ):
        raise ValidationError("loaded campaign claim identity differs")
    authorization_validator = _load_module(
        "_ullm_generic_reasoning_bundle_campaign_authorization",
        CAMPAIGN_AUTHORIZATION_PATH,
    )
    try:
        authorization_validator.validate_authorization_document(
            authorization_document,
            now=datetime.now(timezone.utc),
            required_uid=os.getuid(),
            validate_prior_outcome=False,
            require_fresh_outputs=False,
            require_bound_inputs=False,
            enforce_current_window=False,
        )
        if (
            authorization_validator.canonical_json_bytes(authorization_document)
            != authorization_raw
            or authorization_validator.canonical_json_bytes(claim_document)
            != claim_raw
        ):
            raise ValidationError(
                "campaign authorization or claim is not canonical JSON"
            )
    except Exception as error:
        if isinstance(error, ValidationError):
            raise
        raise ValidationError("campaign authorization validation failed") from error
    try:
        claimed_at = datetime.fromisoformat(
            claim_document["claimed_at"].replace("Z", "+00:00")
        )
        issued_at = datetime.fromisoformat(
            authorization_document["issued_at"].replace("Z", "+00:00")
        )
        expires_at = datetime.fromisoformat(
            authorization_document["expires_at"].replace("Z", "+00:00")
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValidationError("campaign authorization claim time is invalid") from error
    if (
        claimed_at.tzinfo is None
        or issued_at.tzinfo is None
        or expires_at.tzinfo is None
        or not issued_at <= claimed_at < expires_at
    ):
        raise ValidationError("campaign authorization claim time is out of range")
    authorized_campaigns = authorization_document.get("campaigns")
    if (
        not isinstance(authorized_campaigns, dict)
        or set(authorized_campaigns) != AUTHORIZED_CAMPAIGN_NAMES
        or authorized_campaigns.get("sq8_full")
        != {
            "run_id": campaign_run_id,
            "final_path": os.fspath(campaign_root),
        }
        or authorized_campaigns.get("reasoning_release")
        != {
            "run_id": release_campaign["run_id"],
            "final_path": release_campaign["final_path"],
        }
        or authorized_campaigns.get("reasoning_browser")
        != {
            "run_id": browser_campaign["run_id"],
            "final_path": browser_campaign["final_path"],
        }
    ):
        raise ValidationError(
            "generic campaign lineage differs from its authorization"
        )
    if any(
        not isinstance(authorized_campaigns.get(name), dict)
        for name in SQ8_CAMPAIGN_NAMES
    ):
        raise ValidationError("SQ8 campaign authorization identity is missing")
    authorized_candidate = authorization_document.get("candidate")
    authorized_source = authorization_document.get("source")
    authorized_before = authorization_document.get("before")
    authorized_rollback = authorization_document.get("rollback")
    served_identity = campaign_identity["served_model_manifest"]
    if (
        not isinstance(authorized_source, dict)
        or authorized_source.get("commit") != source_commit
        or not isinstance(authorized_before, dict)
        or authorized_before.get("manifest_sha256")
        != rollback["manifest_sha256"]
        or not isinstance(authorized_rollback, dict)
        or authorized_rollback.get("systemd_unit_sha256")
        != rollback["systemd_unit_sha256"]
        or authorized_rollback.get("environment_sha256")
        != rollback["environment_sha256"]
        or not isinstance(authorized_candidate, dict)
        or authorized_candidate.get("model_id") != SQ8_MODEL_ID
        or authorized_candidate.get("format_id") != SQ8_FORMAT_ID
        or authorized_candidate.get("worker_protocol") != SQ8_WORKER_PROTOCOL
        or authorized_candidate.get("manifest_sha256")
        != identity["manifest_sha256"]
        or authorized_candidate.get("worker_binary_sha256")
        != identity["worker_binary_sha256"]
        or authorized_candidate.get("promotion_source_commit") != source_commit
        or authorized_candidate.get("promotion_receipt_sha256")
        != served_identity["promotion_receipt_sha256"]
    ):
        raise ValidationError("campaign authorization candidate identity differs")

    inventory = [
        {
            "path": name,
            "bytes": release_artifacts[name]["bytes"],
            "sha256": release_artifacts[name]["sha256"],
        }
        for name in sorted(release_artifacts, key=lambda item: item.encode("utf-8"))
    ]
    outcome_inventory_raw = (
        json.dumps(
            {"files": inventory},
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    return {
        "campaign_name": "reasoning_release",
        "run_id": release_campaign["run_id"],
        "final_path": release_campaign["final_path"],
        "kind": "directory",
        "sha256": hashlib.sha256(outcome_inventory_raw).hexdigest(),
        "artifact_inventory_sha256": release_lineage[
            "artifact_inventory_sha256"
        ],
        "artifact_count": len(inventory),
        "total_bytes": sum(item["bytes"] for item in inventory),
        "selected_artifacts": {
            item["path"]: item["sha256"]
            for item in inventory
            if item["path"] in OUTCOME_SELECTED_ARTIFACTS
            or Path(item["path"]).name in OUTCOME_SELECTED_ARTIFACTS
        },
        "claim_path": claim["path"],
        "claim_sha256": claim["sha256"],
        "authorization_path": claim["authorization_path"],
        "authorization_sha256": claim["authorization_sha256"],
    }


def _validate_v2(path: Path, document: dict[str, Any]) -> dict[str, Any]:
    expected_root = {
        "schema_version",
        "status",
        "production_activation_performed",
        "source_commit",
        "active_promotion_source_commit",
        "identity",
        "artifacts",
        "rollback_target",
    }
    if set(document) != expected_root or document["schema_version"] != SCHEMA_VERSION_V2:
        raise ValidationError("release bundle v2 root fields differ")
    _scan_forbidden(document)
    if document["status"] not in {"incomplete", "complete"}:
        raise ValidationError("release bundle v2 status is invalid")
    if document["production_activation_performed"] is not False:
        raise ValidationError("release bundle v2 claims activation")
    _commit(document["source_commit"], "source_commit")
    _commit(
        document["active_promotion_source_commit"],
        "active_promotion_source_commit",
    )
    identity = _validate_identity(document["identity"], "identity")
    rollback = document["rollback_target"]
    if not isinstance(rollback, dict) or set(rollback) != {
        "manifest_sha256",
        "systemd_unit_sha256",
        "environment_sha256",
    }:
        raise ValidationError("release bundle v2 rollback_target fields differ")
    for field in rollback:
        _hash(rollback[field], f"rollback_target.{field}")

    artifacts = document["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != V2_ARTIFACT_NAMES:
        raise ValidationError("release bundle v2 artifacts differ")
    resolved = {
        name: _resolve_component_v2(path, artifacts[name], name)
        for name in sorted(V2_ARTIFACT_NAMES)
    }
    files = {name: value[0] for name, value in resolved.items()}
    raws = {name: value[2] for name, value in resolved.items()}

    release = _json_bytes(raws["release_evidence"], "release evidence")
    release_report = _json_bytes(
        raws["release_validator"],
        "release validator report",
    )
    browser = _json_bytes(raws["browser_evidence"], "browser evidence")
    browser_report = _json_bytes(
        raws["browser_validator"],
        "browser validator report",
    )
    campaign_identity = _json_bytes(
        raws["model_campaign_evidence"],
        "SQ8 campaign model identity",
    )
    if (
        release.get("schema_version")
        != "ullm.generic_reasoning_release_evidence.v2"
        or release.get("source_commit") != document["source_commit"]
        or release.get("active_promotion_source_commit")
        != document["active_promotion_source_commit"]
        or release.get("identity") != identity
    ):
        raise ValidationError("release evidence identity differs")
    generic_validator = _load_module(
        "_ullm_generic_reasoning_release_bundle_v2_validator",
        GENERIC_VALIDATOR_PATH,
    )
    try:
        recomputed_release_report = generic_validator.validate(
            files["release_evidence"]
        )
    except Exception as error:
        raise ValidationError("release evidence recomputation failed") from error
    if (
        release_report != recomputed_release_report
        or release_report.get("schema_version")
        != "ullm.generic_reasoning_release_validator.v2"
    ):
        raise ValidationError("release validator report differs from recomputation")

    if (
        browser.get("schema_version")
        != "ullm.openwebui.reasoning_browser_smoke.v4"
        or browser.get("source_commit") != document["source_commit"]
        or browser.get("identity") != identity
    ):
        raise ValidationError("browser v4 identity differs")
    browser_validator = _load_module(
        "_ullm_openwebui_reasoning_bundle_v2_validator",
        BROWSER_VALIDATOR_PATH,
    )
    try:
        recomputed_browser_report = browser_validator.validate(
            files["browser_evidence"]
        )
    except Exception as error:
        raise ValidationError("browser evidence recomputation failed") from error
    if (
        browser_report != recomputed_browser_report
        or browser_report.get("schema_version")
        != "ullm.openwebui.reasoning_browser_smoke_validator.v2"
    ):
        raise ValidationError("browser validator report differs from recomputation")

    if (
        campaign_identity.get("schema_version")
        != "ullm.sq8.full_campaign.model_identity.v2"
        or set(campaign_identity)
        != {
            "schema_version",
            "record_type",
            "model",
            "promotion_validation",
            "product",
            "tokenizer",
            "oracle",
            "worker",
            "served_model_manifest",
            "campaign_authorization_claim",
        }
    ):
        raise ValidationError("SQ8 campaign model identity schema differs")
    campaign_root, _campaign_manifest, _campaign_evidence, campaign_report = (
        _campaign_paths(files)
    )
    candidate_path = campaign_root / "candidate-served-model.json"
    candidate_raw = _stable_read_regular(
        candidate_path,
        "SQ8 candidate served-model manifest",
        MAX_COMPONENT_BYTES,
    )
    _validate_sq8_promotion_and_candidate(
        evidence_path=files["promotion_evidence"],
        receipt_path=files["promotion_receipt"],
        receipt_raw=raws["promotion_receipt"],
        candidate_path=candidate_path,
        candidate_raw=candidate_raw,
        source_commit=document["source_commit"],
        identity=identity,
        campaign_identity=campaign_identity,
    )
    campaign_report_document = _recompute_sq8_campaign_report(
        campaign_root=campaign_root,
        campaign_identity=campaign_identity,
        source_commit=document["source_commit"],
        worker_sha256=identity["worker_binary_sha256"],
        published_report_raw=raws["model_campaign_validator"],
    )
    if campaign_report != files["model_campaign_validator"]:
        raise ValidationError("SQ8 campaign validator location changed")
    campaign_run_id = campaign_report_document.get("run_id")
    if not isinstance(campaign_run_id, str) or not campaign_run_id:
        raise ValidationError("SQ8 campaign validator run identity differs")
    release_campaign = _validate_generic_campaign_lineages(
        bundle_path=path,
        campaign_root=campaign_root,
        campaign_run_id=campaign_run_id,
        release=release,
        release_report=release_report,
        browser=browser,
        browser_report=browser_report,
        browser_path=files["browser_evidence"],
        campaign_identity=campaign_identity,
        identity=identity,
        source_commit=document["source_commit"],
        rollback=rollback,
    )

    reasons: list[str] = []
    if release_report.get("gate_eligible") is not True:
        reasons.append("release validator gate is not eligible")
    if browser_report.get("gate_eligible") is not True:
        reasons.append("browser validator gate is not eligible")
    if document["source_commit"] != document["active_promotion_source_commit"]:
        reasons.append("source commit is not aligned with active promotion source")
    if document["status"] != "complete":
        reasons.append("release bundle status is incomplete")
    return {
        "schema_version": VALIDATOR_SCHEMA_VERSION_V2,
        "input_schema_version": SCHEMA_VERSION_V2,
        "structurally_valid": True,
        "gate_eligible": not reasons,
        "source_commit": document["source_commit"],
        "artifact_count": len(files),
        "model_campaign_schema_version": campaign_identity["schema_version"],
        "reasoning_release_campaign": release_campaign,
        "reasons": reasons,
    }


def validate(path: Path) -> dict[str, Any]:
    document, _raw = _read_json(path, "release bundle", MAX_BUNDLE_BYTES)
    schema = document.get("schema_version")
    if schema == SCHEMA_VERSION_V1:
        return _validate_v1(path)
    if schema == SCHEMA_VERSION_V2:
        absolute = path.absolute()
        try:
            if absolute.resolve(strict=True) != absolute:
                raise ValidationError("release bundle v2 path is not canonical")
        except OSError as error:
            raise ValidationError("release bundle v2 path is unavailable") from error
        raw = _stable_read_regular(
            absolute,
            "release bundle v2",
            MAX_BUNDLE_BYTES,
            require_immutable=True,
        )
        stable_document = _json_bytes(raw, "release bundle v2")
        if stable_document.get("schema_version") != SCHEMA_VERSION_V2:
            raise ValidationError("release bundle v2 changed before validation")
        report = _validate_v2(absolute, stable_document)
        if (
            _stable_read_regular(
                absolute,
                "release bundle v2",
                MAX_BUNDLE_BYTES,
                require_immutable=True,
            )
            != raw
        ):
            raise ValidationError("release bundle v2 changed during validation")
        report["bundle_sha256"] = hashlib.sha256(raw).hexdigest()
        return report
    raise ValidationError("release bundle schema is unsupported")


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
