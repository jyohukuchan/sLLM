#!/usr/bin/env python3
"""Materialize a served-model manifest from a deployment profile and live files."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence
import re

try:
    import qwen35_aq4_sq8_authorization_lineage as lineage_tool
except ModuleNotFoundError:
    from tools import qwen35_aq4_sq8_authorization_lineage as lineage_tool


ROOT = Path(__file__).resolve().parents[1]
LOADER_PATH = ROOT / "services/openai-gateway/src/ullm_openai_gateway/served_model.py"
PROFILE_SCHEMA = "ullm.served_model.profile.v1"
AQ4_EVIDENCE_SCHEMA = "ullm.aq4_resident_promotion_evidence.v1"
SQ8_OVERLAY_IMPLEMENTATION_ID = "qwen35_aq4_sq8_linear_qkv_z_overlay_v1"
SQ8_OVERLAY_RECEIPT_SCHEMA = "ullm.qwen35_aq4_sq8_overlay_promotion.v1"
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
SQ8_REQUEST_ID_RE = re.compile(r"^sq8-promotion-[0-9a-f]{64}$")


class GenerationError(RuntimeError):
    """Raised when a profile cannot be bound to immutable local files."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GenerationError(f"failed to read {label}") from error
    if not isinstance(value, dict):
        raise GenerationError(f"{label} must be a JSON object")
    return value


def _receipt_value(receipt: dict[str, Any], path: Any, label: str) -> str:
    if (
        not isinstance(path, list)
        or not path
        or not all(isinstance(item, str) and item for item in path)
    ):
        raise GenerationError(f"{label} must be a nonempty string path")
    value: Any = receipt
    for component in path:
        if not isinstance(value, dict) or component not in value:
            raise GenerationError(f"{label} is absent from the promotion receipt")
        value = value[component]
    if not isinstance(value, str) or not value:
        raise GenerationError(f"{label} in the promotion receipt is invalid")
    return value


def _resolve_receipt_file(receipt_path: Path, raw_path: str, label: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute() or not relative.parts or any(
        component in ("", ".", "..") for component in relative.parts
    ):
        raise GenerationError(f"{label} must be a safe relative path")
    unresolved = receipt_path.parent / relative
    if unresolved.is_symlink():
        raise GenerationError(f"{label} must identify a regular non-symlink file")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(receipt_path.parent.resolve())
    except ValueError as error:
        raise GenerationError(f"{label} escapes the promotion directory") from error
    if resolved.is_symlink() or not resolved.is_file():
        raise GenerationError(f"{label} must identify a regular non-symlink file")
    return resolved


def _load_overlay_receipt_tool() -> ModuleType:
    """Load the standalone SQ8 receipt/inventory validator without a package import."""

    path = ROOT / "tools/write-qwen35-aq4-sq8-overlay-promotion-receipt.py"
    spec = importlib.util.spec_from_file_location("_ullm_sq8_overlay_receipt", path)
    if spec is None or spec.loader is None:
        raise GenerationError("SQ8 overlay receipt validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _require_hex(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise GenerationError(f"{label} must be a lowercase hexadecimal digest")
    return value


def _receipt_object(receipt: dict[str, Any], path: Any, label: str) -> Any:
    """Resolve a profile receipt mapping while preserving an explicit null."""

    if (
        not isinstance(path, list)
        or not path
        or not all(isinstance(item, str) and item for item in path)
    ):
        raise GenerationError(f"{label} must be a nonempty string path")
    value: Any = receipt
    for component in path:
        if not isinstance(value, dict) or component not in value:
            raise GenerationError(f"{label} is absent from the promotion receipt")
        value = value[component]
    return value


def _resolve_authorization_audit(value: Any, label: str) -> dict[str, str] | None:
    """Validate and resolve the optional independent authorization audit ref."""

    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise GenerationError(f"{label} must be null or an exact path/SHA object")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise GenerationError(f"{label} path must be absolute")
    audit_path = Path(raw_path)
    if audit_path == Path("/") or audit_path.resolve() != audit_path:
        raise GenerationError(f"{label} path must be canonical")
    if audit_path.is_symlink() or not audit_path.is_file():
        raise GenerationError(f"{label} path must be a regular non-symlink file")
    digest = _require_hex(value.get("sha256"), HEX64_RE, f"{label} SHA-256")
    if _sha256_file(audit_path) != digest:
        raise GenerationError(f"{label} SHA-256 differs")
    return {"path": os.fspath(audit_path), "sha256": digest}


def _resolve_readiness(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"schema", "container", "network", "endpoint"}:
        raise GenerationError(f"{label} shape differs")
    if value.get("schema") != "ullm.bridge_container_readiness.v1":
        raise GenerationError(f"{label} schema differs")
    container = value.get("container")
    network = value.get("network")
    endpoint = value.get("endpoint")
    if not isinstance(container, dict) or set(container) != {"name", "id", "image_id", "config_image"}:
        raise GenerationError(f"{label} container identity differs")
    if (
        container.get("name") != "open-webui"
        or not isinstance(container.get("id"), str)
        or HEX64_RE.fullmatch(container["id"]) is None
        or not isinstance(container.get("image_id"), str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", container["image_id"]) is None
        or not isinstance(container.get("config_image"), str)
        or not container["config_image"]
    ):
        raise GenerationError(f"{label} container identity differs")
    if not isinstance(network, dict) or set(network) != {"name", "id", "driver", "bridge_interface"}:
        raise GenerationError(f"{label} network identity differs")
    network_id = network.get("id")
    if (
        not isinstance(network.get("name"), str)
        or not network["name"]
        or not isinstance(network_id, str)
        or HEX64_RE.fullmatch(network_id) is None
        or network.get("driver") != "bridge"
        or network.get("bridge_interface") != f"br-{network_id[:12]}"
    ):
        raise GenerationError(f"{label} network identity differs")
    expected_body = '{"status":"ready"}'
    expected_endpoint = {
        "url": "http://172.20.0.1:8000/readyz",
        "path": "/readyz",
        "expected_status": 200,
        "expected_body": expected_body,
        "expected_body_sha256": hashlib.sha256(expected_body.encode("ascii")).hexdigest(),
        "timeout_seconds": 5,
    }
    if not isinstance(endpoint, dict) or endpoint != expected_endpoint:
        raise GenerationError(f"{label} endpoint identity differs")
    return json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))


def _validate_sq8_overlay_receipt(
    *,
    profile: dict[str, Any],
    promotion_profile: dict[str, Any],
    receipt: dict[str, Any],
    receipt_path: Path,
    profile_path: Path,
    worker_binary: Path,
    worker_sha256: str,
    product_root: Path,
    artifact_manifest_path: str,
    package_manifest_path: str,
    package_manifest_sha256: str,
    expected_manifest_path: Path | None = None,
    allow_prepared: bool = False,
    prepared_only: bool = False,
) -> dict[str, Any]:
    """Validate the immutable SQ8 overlay publication contract.

    The receipt is intentionally self-contained: stale source identity,
    missing inventory, an unbound authorization audit, or any profile field
    that weakens the receipt indirection is rejected before a served manifest
    can be written.
    """

    legacy_promotion = {
        "receipt",
        "source_commit_from_receipt",
        "required_schema_version",
        "overlay_from_receipt",
        "release_from_receipt",
        "package_from_receipt",
        "actual_evidence_from_receipt",
        "request_id_from_receipt",
        "authorization_audit_from_receipt",
        "readiness_from_receipt",
        "readiness",
        "release_source_commit",
    }
    lineage_promotion = legacy_promotion | {
        "authorization_lineage_from_receipt",
        "authorization_lineage",
    }
    has_lineage_contract = set(promotion_profile) == lineage_promotion
    if set(promotion_profile) != legacy_promotion and not has_lineage_contract:
        raise GenerationError("SQ8 overlay promotion profile contract is incomplete")
    if promotion_profile.get("required_schema_version") != SQ8_OVERLAY_RECEIPT_SCHEMA:
        raise GenerationError("SQ8 overlay promotion receipt schema differs")
    if promotion_profile.get("source_commit_from_receipt") != ["source_commit"]:
        raise GenerationError("SQ8 overlay source commit receipt binding differs")
    if promotion_profile.get("overlay_from_receipt") != ["overlay"]:
        raise GenerationError("SQ8 overlay receipt overlay binding differs")
    if promotion_profile.get("release_from_receipt") != ["release"]:
        raise GenerationError("SQ8 overlay receipt release binding differs")
    if promotion_profile.get("package_from_receipt") != ["package"]:
        raise GenerationError("SQ8 overlay receipt package binding differs")
    if promotion_profile.get("actual_evidence_from_receipt") != ["actual"]:
        raise GenerationError("SQ8 overlay actual evidence receipt binding differs")
    if promotion_profile.get("request_id_from_receipt") != ["request_id"]:
        raise GenerationError("SQ8 overlay request ID receipt binding differs")
    if promotion_profile.get("authorization_audit_from_receipt") != ["authorization_audit"]:
        raise GenerationError("SQ8 overlay authorization audit receipt binding differs")
    if has_lineage_contract and promotion_profile.get("authorization_lineage_from_receipt") != ["authorization_lineage"]:
        raise GenerationError("SQ8 overlay authorization lineage receipt binding differs")
    if promotion_profile.get("readiness_from_receipt") != ["readiness"]:
        raise GenerationError("SQ8 overlay readiness receipt binding differs")
    profile_readiness = _resolve_readiness(
        promotion_profile.get("readiness"), "SQ8 overlay profile readiness"
    )
    release_commit = _require_hex(
        promotion_profile.get("release_source_commit"), HEX40_RE,
        "SQ8 overlay profile release source commit",
    )

    if receipt.get("schema_version") != SQ8_OVERLAY_RECEIPT_SCHEMA:
        raise GenerationError("SQ8 overlay promotion receipt schema differs")
    expected_receipt_keys = {
        "schema_version",
        "status",
        "request_id",
        "source_commit",
        "source_provenance",
        "release",
        "overlay",
        "package",
        "authorization_audit",
        "readiness",
        "actual",
    }
    if has_lineage_contract:
        expected_receipt_keys.add("authorization_lineage")
    if set(receipt) != expected_receipt_keys:
        raise GenerationError("SQ8 overlay promotion receipt shape differs")
    source_commit = _require_hex(
        receipt.get("source_commit"), HEX40_RE, "SQ8 overlay receipt source commit"
    )
    if source_commit != release_commit:
        raise GenerationError("SQ8 overlay receipt source commit is stale or mismatched")
    request_id = receipt.get("request_id")
    if not isinstance(request_id, str) or SQ8_REQUEST_ID_RE.fullmatch(request_id) is None:
        raise GenerationError("SQ8 overlay receipt request ID differs")
    status = receipt.get("status")
    if status not in {"prepared_not_executed", "actual_verified"}:
        raise GenerationError("SQ8 overlay promotion receipt status differs")
    if status == "prepared_not_executed" and not allow_prepared:
        raise GenerationError("SQ8 overlay prepared receipt is not executable")
    if status == "actual_verified" and prepared_only:
        raise GenerationError("SQ8 overlay prepared candidate requires a pending receipt")
    source = receipt.get("source_provenance")
    if not isinstance(source, dict) or set(source) != {"tree_sha256", "archive_sha256"}:
        raise GenerationError("SQ8 overlay source provenance is incomplete")
    _require_hex(source.get("tree_sha256"), HEX40_RE, "SQ8 overlay source tree")
    _require_hex(source.get("archive_sha256"), HEX64_RE, "SQ8 overlay source archive")

    authorization_audit = _resolve_authorization_audit(
        _receipt_object(
            receipt,
            promotion_profile.get("authorization_audit_from_receipt"),
            "SQ8 overlay authorization audit",
        ),
        "SQ8 overlay authorization audit",
    )
    authorization_lineage = None
    if has_lineage_contract:
        authorization_lineage = _receipt_object(
            receipt,
            promotion_profile.get("authorization_lineage_from_receipt"),
            "SQ8 overlay authorization lineage",
        )
        if authorization_lineage != promotion_profile.get("authorization_lineage"):
            raise GenerationError("SQ8 overlay profile/receipt authorization lineage differs")
        if authorization_lineage is not None:
            try:
                lineage_tool.validate_reference(authorization_lineage)
            except lineage_tool.LineageError as error:
                raise GenerationError(
                    f"SQ8 overlay authorization lineage differs: {error}"
                ) from error
    readiness = _resolve_readiness(
        _receipt_object(
            receipt,
            promotion_profile.get("readiness_from_receipt"),
            "SQ8 overlay readiness",
        ),
        "SQ8 overlay readiness",
    )
    if readiness != profile_readiness:
        raise GenerationError("SQ8 overlay profile/receipt readiness differs")

    release = receipt.get("release")
    if not isinstance(release, dict) or set(release) != {"worker", "profile", "served_model"}:
        raise GenerationError("SQ8 overlay release binding is incomplete")
    release_worker = release.get("worker")
    if not isinstance(release_worker, dict) or set(release_worker) != {
        "path", "sha256", "bytes", "mode", "nlink"
    }:
        raise GenerationError("SQ8 overlay worker release binding is incomplete")
    if Path(str(release_worker.get("path"))).resolve() != worker_binary:
        raise GenerationError("SQ8 overlay worker release path differs")
    if release_worker.get("sha256") != worker_sha256:
        raise GenerationError("SQ8 overlay worker release SHA-256 differs")
    if release_worker.get("bytes") != worker_binary.stat().st_size:
        raise GenerationError("SQ8 overlay worker release size differs")
    if release_worker.get("mode") != "0555" or release_worker.get("nlink") != 1:
        raise GenerationError("SQ8 overlay worker release topology differs")
    release_profile = release.get("profile")
    if not isinstance(release_profile, dict) or set(release_profile) != {"path", "sha256"}:
        raise GenerationError("SQ8 overlay profile release binding is incomplete")
    if Path(str(release_profile.get("path"))).resolve() != profile_path.resolve():
        raise GenerationError("SQ8 overlay profile release path differs")
    if release_profile.get("sha256") != _sha256_file(profile_path):
        raise GenerationError("SQ8 overlay profile release SHA-256 differs")
    release_manifest = release.get("served_model")
    if not isinstance(release_manifest, dict) or set(release_manifest) != {
        "path", "semantic_sha256"
    }:
        raise GenerationError("SQ8 overlay served-model release binding is incomplete")
    manifest_path = Path(str(release_manifest.get("path"))).resolve()
    if not manifest_path.is_absolute() or manifest_path == Path("/"):
        raise GenerationError("SQ8 overlay served-model release path is invalid")
    if expected_manifest_path is not None and manifest_path != expected_manifest_path.resolve():
        raise GenerationError("SQ8 overlay served-model release path differs")
    _require_hex(
        release_manifest.get("semantic_sha256"), HEX64_RE,
        "SQ8 overlay served-model semantic SHA-256",
    )

    overlay = receipt.get("overlay")
    expected_overlay_keys = {
        "binding_manifest_path",
        "binding_manifest_sha256",
        "content_sha256",
        "tensor_set_sha256",
        "tensor_count",
        "artifact_inventory",
    }
    if not isinstance(overlay, dict) or set(overlay) != expected_overlay_keys:
        raise GenerationError("SQ8 overlay artifact binding is incomplete")
    binding_path = (product_root / artifact_manifest_path).resolve()
    if Path(str(overlay.get("binding_manifest_path"))).resolve() != binding_path:
        raise GenerationError("SQ8 overlay binding manifest path differs")
    binding_sha256 = _require_hex(
        overlay.get("binding_manifest_sha256"), HEX64_RE,
        "SQ8 overlay binding manifest SHA-256",
    )
    if binding_sha256 != _sha256_file(binding_path):
        raise GenerationError("SQ8 overlay binding manifest SHA-256 differs")
    content_sha256 = _require_hex(
        overlay.get("content_sha256"), HEX64_RE, "SQ8 overlay content SHA-256"
    )
    tensor_set_sha256 = _require_hex(
        overlay.get("tensor_set_sha256"), HEX64_RE, "SQ8 overlay tensor-set SHA-256"
    )
    if overlay.get("tensor_count") != 48:
        raise GenerationError("SQ8 overlay tensor count differs")
    binding = _load_json(binding_path, "SQ8 overlay binding manifest")
    if (
        binding.get("schema_version") != "ullm.qwen35_aq4_sq8_qkv_z_overlay.v2"
        or binding.get("format_id") != "AQ4_0"
        or binding.get("overlay_format_id") != "SQ8_0"
        or binding.get("implementation_id") != SQ8_OVERLAY_IMPLEMENTATION_ID
    ):
        raise GenerationError("SQ8 overlay binding identity differs")
    names = binding.get("tensor_names")
    if not isinstance(names, list) or len(names) != 48 or len(set(names)) != 48:
        raise GenerationError("SQ8 overlay binding tensor set differs")
    if binding.get("content_sha256") != content_sha256 or binding.get("tensor_set_sha256") != tensor_set_sha256:
        raise GenerationError("SQ8 overlay binding SHA identities differ")
    package_ref = binding.get("package")
    if not isinstance(package_ref, dict) or package_ref.get("manifest_sha256") != package_manifest_sha256:
        raise GenerationError("SQ8 overlay binding package identity differs")

    receipt_package = receipt.get("package")
    if not isinstance(receipt_package, dict) or set(receipt_package) != {
        "manifest_path", "manifest_sha256"
    }:
        raise GenerationError("SQ8 overlay package binding is incomplete")
    package_path = (product_root / package_manifest_path).resolve()
    if Path(str(receipt_package.get("manifest_path"))).resolve() != package_path:
        raise GenerationError("SQ8 overlay package manifest path differs")
    if receipt_package.get("manifest_sha256") != package_manifest_sha256:
        raise GenerationError("SQ8 overlay package manifest SHA-256 differs")

    receipt_tool = _load_overlay_receipt_tool()
    try:
        inventory = receipt_tool.artifact_inventory(product_root / Path(artifact_manifest_path).parent)
    except Exception as error:
        raise GenerationError(f"SQ8 overlay artifact inventory unavailable: {error}") from error
    if overlay.get("artifact_inventory") != inventory:
        raise GenerationError("SQ8 overlay artifact inventory differs")
    actual = receipt.get("actual")
    if status == "prepared_not_executed":
        if actual != {"status": "pending", "required": True}:
            raise GenerationError("SQ8 overlay prepared receipt actual evidence differs")
    else:
        if not isinstance(actual, dict) or actual.get("status") != "actual_verified":
            raise GenerationError("SQ8 overlay actual evidence is incomplete")
        prepared_ref = actual.get("prepared_receipt")
        if not isinstance(prepared_ref, dict) or set(prepared_ref) != {"path", "sha256"}:
            raise GenerationError("SQ8 overlay prepared receipt reference is incomplete")
        _require_hex(prepared_ref.get("sha256"), HEX64_RE, "SQ8 overlay prepared receipt SHA-256")
        prepared_raw_path = prepared_ref.get("path")
        if not isinstance(prepared_raw_path, str) or not Path(prepared_raw_path).is_absolute():
            raise GenerationError("SQ8 overlay prepared receipt path must be absolute")
        prepared_path = Path(prepared_raw_path)
        if prepared_path.is_symlink() or not prepared_path.is_file():
            raise GenerationError("SQ8 overlay prepared receipt path must be a regular non-symlink file")
        configured_prepared_path = Path(str(promotion_profile.get("receipt", ""))).resolve()
        if prepared_path != configured_prepared_path:
            raise GenerationError("SQ8 overlay prepared receipt path differs from profile")
        if _sha256_file(prepared_path) != prepared_ref["sha256"]:
            raise GenerationError("SQ8 overlay prepared receipt SHA-256 differs")
        try:
            prepared_value = _load_json(prepared_path, "SQ8 overlay prepared receipt")
        except Exception as error:
            raise GenerationError(f"SQ8 overlay prepared receipt is unreadable: {error}") from error
        if (
            prepared_value.get("schema_version") != SQ8_OVERLAY_RECEIPT_SCHEMA
            or prepared_value.get("status") != "prepared_not_executed"
            or prepared_value.get("request_id") != request_id
            or prepared_value.get("actual") != {"status": "pending", "required": True}
        ):
            raise GenerationError("SQ8 overlay prepared receipt state differs")
        for section in (
            "source_commit", "source_provenance", "overlay", "package",
            "authorization_audit", "authorization_lineage", "readiness",
        ):
            if prepared_value.get(section) != receipt.get(section):
                raise GenerationError(f"SQ8 overlay prepared receipt {section} differs")
        prepared_release = prepared_value.get("release")
        if (
            not isinstance(prepared_release, dict)
            or not isinstance(release, dict)
            or prepared_release.get("worker") != release.get("worker")
            or prepared_release.get("profile") != release.get("profile")
        ):
            raise GenerationError("SQ8 overlay prepared receipt release identity differs")
        maintenance_ref = actual.get("maintenance_evidence")
        executor_ref = actual.get("executor_record")
        if not isinstance(maintenance_ref, dict) or set(maintenance_ref) != {"path", "sha256"}:
            raise GenerationError("SQ8 overlay maintenance evidence reference is incomplete")
        if not isinstance(executor_ref, dict) or set(executor_ref) != {"path", "sha256"}:
            raise GenerationError("SQ8 overlay executor evidence reference is incomplete")
        _require_hex(maintenance_ref.get("sha256"), HEX64_RE, "SQ8 overlay maintenance evidence SHA-256")
        _require_hex(executor_ref.get("sha256"), HEX64_RE, "SQ8 overlay executor record SHA-256")
        maintenance_path = _resolve_receipt_file(
            receipt_path, maintenance_ref["path"], "SQ8 overlay maintenance evidence path"
        )
        executor_path = _resolve_receipt_file(
            receipt_path, executor_ref["path"], "SQ8 overlay executor evidence path"
        )
        if _sha256_file(maintenance_path) != maintenance_ref["sha256"]:
            raise GenerationError("SQ8 overlay maintenance evidence SHA-256 differs")
        if _sha256_file(executor_path) != executor_ref["sha256"]:
            raise GenerationError("SQ8 overlay executor evidence SHA-256 differs")
        try:
            expected_actual = receipt_tool.validate_actual_evidence(
                maintenance_path=maintenance_path,
                executor_path=executor_path,
                output_path=receipt_path,
                profile=profile,
                overlay=overlay,
                package_sha256=package_manifest_sha256,
                request_id=request_id,
                prepared_receipt_path=prepared_path,
            )
        except Exception as error:
            raise GenerationError(f"SQ8 overlay actual evidence validation failed: {error}") from error
        if actual != expected_actual:
            raise GenerationError("SQ8 overlay actual evidence projection differs")
    return {
        "source_commit": source_commit,
        "served_model_semantic_sha256": release_manifest["semantic_sha256"],
        "authorization_audit": authorization_audit,
        "authorization_lineage": authorization_lineage,
        "readiness": readiness,
    }


def _validate_aq4_evidence(
    *,
    profile: dict[str, Any],
    promotion_profile: dict[str, Any],
    receipt: dict[str, Any],
    receipt_path: Path,
    source_commit: str,
    worker_binary: Path,
    worker_sha256: str,
    product_root: Path,
    package_manifest_path: str,
    package_manifest_sha256: str,
) -> None:
    required_schema = promotion_profile.get("required_schema_version")
    if required_schema is None:
        return
    if required_schema != "ullm.aq4_resident_promotion.v1":
        raise GenerationError("profile promotion receipt schema is unsupported")
    if receipt.get("schema_version") != required_schema:
        raise GenerationError("promotion receipt schema differs")

    evidence_path_value = _receipt_value(
        receipt,
        promotion_profile.get("evidence_from_receipt"),
        "AQ4 promotion evidence path",
    )
    evidence_sha256 = _receipt_value(
        receipt,
        promotion_profile.get("evidence_sha256_from_receipt"),
        "AQ4 promotion evidence SHA-256",
    )
    if len(evidence_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in evidence_sha256
    ):
        raise GenerationError("AQ4 promotion evidence SHA-256 is invalid")
    evidence_path = _resolve_receipt_file(
        receipt_path, evidence_path_value, "AQ4 promotion evidence path"
    )
    if _sha256_file(evidence_path) != evidence_sha256:
        raise GenerationError("AQ4 promotion evidence SHA-256 differs")
    evidence = _load_json(evidence_path, "AQ4 promotion evidence")
    if evidence.get("schema_version") != AQ4_EVIDENCE_SCHEMA:
        raise GenerationError("AQ4 promotion evidence schema differs")
    if evidence.get("verified") is not True:
        raise GenerationError("AQ4 promotion evidence is not verified")
    if evidence.get("production_receipt_written") is not False:
        raise GenerationError("AQ4 promotion evidence was not captured before receipt publication")
    gpu_preflight = evidence.get("gpu_exclusive_preflight")
    if not isinstance(gpu_preflight, dict) or set(gpu_preflight) != {
        "tool",
        "gpu_index",
        "positive_vram_processes",
    }:
        raise GenerationError("AQ4 promotion evidence GPU exclusivity preflight is missing")
    if (
        gpu_preflight.get("tool") != "rocm-smi --showpids --json"
        or gpu_preflight.get("gpu_index") != "1"
        or gpu_preflight.get("positive_vram_processes") != []
    ):
        raise GenerationError("AQ4 promotion evidence GPU exclusivity preflight failed")
    if evidence.get("source_commit") != source_commit:
        raise GenerationError("AQ4 promotion evidence source commit differs")
    if evidence.get("worker_binary") != os.fspath(worker_binary):
        raise GenerationError("AQ4 promotion evidence worker path differs")
    if evidence.get("worker_binary_sha256") != worker_sha256:
        raise GenerationError("AQ4 promotion evidence worker SHA-256 differs")

    _validate_aq4_token_comparisons(evidence)
    for mode in ("resident", "legacy"):
        result = evidence.get(mode)
        if not isinstance(result, dict) or result.get("clean_shutdown") is not True:
            raise GenerationError(f"AQ4 promotion evidence {mode} shutdown is not clean")
    child_checks = evidence["resident"].get("child_process_checks")
    if not isinstance(child_checks, list) or not child_checks or any(
        not isinstance(check, dict) or check.get("sibling_engine_count") != 0
        for check in child_checks
    ):
        raise GenerationError("AQ4 resident child-process evidence is incomplete")

    bundle = evidence.get("ephemeral_bundle")
    manifest = bundle.get("manifest") if isinstance(bundle, dict) else None
    if not isinstance(manifest, dict):
        raise GenerationError("AQ4 promotion evidence has no bound manifest")
    expected_worker = _required_object(profile, "worker")
    observed_worker = manifest.get("worker")
    if not isinstance(observed_worker, dict):
        raise GenerationError("AQ4 promotion evidence worker identity is absent")
    for name in ("protocol", "arguments", "required_environment", "identity"):
        if observed_worker.get(name) != expected_worker.get(name):
            raise GenerationError(f"AQ4 promotion evidence worker {name} differs")
    if observed_worker.get("binary") != os.fspath(worker_binary) or observed_worker.get(
        "binary_sha256"
    ) != worker_sha256:
        raise GenerationError("AQ4 promotion evidence worker binding differs")
    for name in ("public", "generation", "format"):
        if manifest.get(name) != _required_object(profile, name):
            raise GenerationError(f"AQ4 promotion evidence profile {name} differs")
    observed_product = manifest.get("product")
    observed_package = (
        observed_product.get("package") if isinstance(observed_product, dict) else None
    )
    if not isinstance(observed_package, dict) or observed_product.get("root") != os.fspath(
        product_root
    ):
        raise GenerationError("AQ4 promotion evidence product identity differs")
    if observed_package != {
        "manifest_path": package_manifest_path,
        "manifest_sha256": package_manifest_sha256,
    }:
        raise GenerationError("AQ4 promotion evidence package identity differs")
    if profile.get("worker", {}).get("protocol") == "ullm.worker.v2":
        _validate_v2_reasoning_evidence(evidence, manifest)


def _validate_aq4_token_comparisons(evidence: dict[str, Any]) -> None:
    comparisons = evidence.get("comparisons")
    resident = evidence.get("resident")
    legacy = evidence.get("legacy")
    resident_cases = resident.get("cases") if isinstance(resident, dict) else None
    legacy_cases = legacy.get("cases") if isinstance(legacy, dict) else None
    if (
        not isinstance(comparisons, list)
        or not comparisons
        or not isinstance(resident_cases, list)
        or not isinstance(legacy_cases, list)
    ):
        raise GenerationError("AQ4 promotion evidence comparisons are incomplete")

    def comparable_cases(cases: list[Any]) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        for case in cases:
            if not isinstance(case, dict) or case.get("id") == "reasoning-budget-zero":
                continue
            case_id = case.get("id")
            tokens = case.get("tokens")
            if (
                not isinstance(case_id, str)
                or not case_id
                or case_id in result
                or not isinstance(tokens, list)
                or not all(isinstance(token, int) and token >= 0 for token in tokens)
            ):
                raise GenerationError("AQ4 promotion evidence token cases are invalid")
            result[case_id] = tokens
        return result

    resident_by_id = comparable_cases(resident_cases)
    legacy_by_id = comparable_cases(legacy_cases)
    if resident_by_id.keys() != legacy_by_id.keys():
        raise GenerationError("AQ4 promotion evidence comparable case IDs differ")
    comparison_ids: set[str] = set()
    for item in comparisons:
        if not isinstance(item, dict):
            raise GenerationError("AQ4 promotion evidence comparisons are incomplete")
        case_id = item.get("id")
        if (
            not isinstance(case_id, str)
            or case_id in comparison_ids
            or item.get("tokens_exact_match") is not True
            or case_id not in resident_by_id
            or resident_by_id[case_id] != legacy_by_id[case_id]
        ):
            raise GenerationError("AQ4 promotion evidence token comparisons differ")
        comparison_ids.add(case_id)
    if comparison_ids != resident_by_id.keys():
        raise GenerationError("AQ4 promotion evidence comparisons are incomplete")


def _validate_v2_reasoning_evidence(
    evidence: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """Recompute the deterministic v2 promotion case from raw token records."""

    reasoning = manifest.get("reasoning")
    worker = manifest.get("worker")
    if not isinstance(reasoning, dict) or not isinstance(worker, dict):
        raise GenerationError("AQ4 v2 promotion evidence lacks reasoning binding")
    resident = evidence.get("resident")
    legacy = evidence.get("legacy")
    if not isinstance(resident, dict) or not isinstance(legacy, dict):
        raise GenerationError("AQ4 v2 promotion evidence lacks worker results")
    resident_ready = resident.get("ready")
    legacy_ready = legacy.get("ready")
    if (
        not isinstance(resident_ready, dict)
        or not isinstance(legacy_ready, dict)
        or resident_ready.get("schema_version") != "ullm.worker.v2"
    ):
        raise GenerationError("AQ4 v2 resident ready schema differs")
    if legacy_ready.get("schema_version") != "ullm.worker.v1":
        raise GenerationError("AQ4 v2 legacy ready schema differs")

    resident_cases = resident.get("cases")
    legacy_cases = legacy.get("cases")
    if not isinstance(resident_cases, list) or not isinstance(legacy_cases, list):
        raise GenerationError("AQ4 v2 promotion evidence cases are incomplete")
    reasoning_cases = [
        case
        for case in resident_cases
        if isinstance(case, dict) and case.get("id") == "reasoning-budget-zero"
    ]
    if len(reasoning_cases) != 1:
        raise GenerationError("AQ4 v2 promotion evidence reasoning case is missing")
    raw_cases = [
        case
        for case in resident_cases
        if isinstance(case, dict) and case.get("id") != "reasoning-budget-zero"
    ]
    if len(raw_cases) != len(legacy_cases):
        raise GenerationError("AQ4 v2 promotion evidence raw case counts differ")
    reasoning_case = reasoning_cases[0]
    request = reasoning_case.get("reasoning")
    if not isinstance(request, dict):
        raise GenerationError("AQ4 v2 promotion reasoning request is absent")
    if (
        request.get("enabled") is not True
        or request.get("budget_tokens") != 0
        or request.get("dialect_id") != reasoning.get("dialect_id")
        or request.get("end_token_ids") != reasoning.get("end_token_ids")
        or request.get("forced_end_token_ids") != reasoning.get("forced_end_token_ids")
        or request.get("reserved_answer_tokens") != reasoning.get("reserved_answer_tokens")
    ):
        raise GenerationError("AQ4 v2 promotion reasoning request differs")
    usage = reasoning_case.get("reasoning_usage")
    forced_end = reasoning.get("forced_end_token_ids")
    reserved_answer = reasoning.get("reserved_answer_tokens")
    tokens = reasoning_case.get("tokens")
    if (
        not isinstance(usage, dict)
        or type(usage.get("reasoning_tokens")) is not int
        or usage.get("reasoning_tokens") != 0
        or not isinstance(forced_end, list)
        or type(reserved_answer) is not int
        or reserved_answer < 1
        or type(usage.get("forced_end_tokens")) is not int
        or usage.get("forced_end_tokens") != len(forced_end)
        or not isinstance(tokens, list)
        or not all(type(token) is int and token >= 0 for token in tokens)
        or len(tokens) < len(forced_end) + reserved_answer
        or tokens[: len(forced_end)] != forced_end
    ):
        raise GenerationError("AQ4 v2 promotion reasoning accounting is incomplete")


def _load_validator() -> ModuleType:
    package_root = LOADER_PATH.parents[1]
    if os.fspath(package_root) not in sys.path:
        sys.path.insert(0, os.fspath(package_root))
    package_name = "ullm_openai_gateway"
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [os.fspath(package_root / package_name)]  # type: ignore[attr-defined]
        package.__package__ = package_name
        sys.modules[package_name] = package
    module_name = "ullm_openai_gateway.served_model"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, LOADER_PATH)
    if spec is None or spec.loader is None:
        raise GenerationError("served-model validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _required_object(parent: dict[str, Any], name: str) -> dict[str, Any]:
    value = parent.get(name)
    if not isinstance(value, dict):
        raise GenerationError(f"profile.{name} must be an object")
    return value


def _served_model_semantic_sha256(document: dict[str, Any]) -> str:
    """Hash a served manifest without the self-referential receipt file hash."""

    value = json.loads(json.dumps(document, ensure_ascii=True, allow_nan=False))
    promotion = value.get("promotion")
    if isinstance(promotion, dict):
        promotion.pop("receipt_sha256", None)
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _publish_create_new(temporary: Path, output_path: Path) -> None:
    """Publish a validated manifest once without replacing an existing path."""

    try:
        os.link(temporary, output_path, follow_symlinks=False)
    except FileExistsError as error:
        raise GenerationError("served-model output already exists or is a symlink") from error
    try:
        temporary.unlink()
    except OSError:
        output_path.unlink(missing_ok=True)
        raise
    directory = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _materialize_profile_document(
    profile_path: Path,
    *,
    expected_manifest_path: Path | None = None,
    receipt_override: dict[str, Any] | None = None,
    receipt_sha256_override: str | None = None,
    validate_receipt: bool = True,
    receipt_path_override: Path | None = None,
    allow_prepared: bool = False,
    prepared_only: bool = False,
) -> dict[str, Any]:
    profile = _load_json(profile_path, "served-model profile")
    if profile.get("schema_version") != PROFILE_SCHEMA:
        raise GenerationError("served-model profile schema is unsupported")

    tokenizer_profile = _required_object(profile, "tokenizer")
    worker_profile = _required_object(profile, "worker")
    product_profile = _required_object(profile, "product")
    promotion_profile = _required_object(profile, "promotion")
    reasoning_profile = profile.get("reasoning")
    if reasoning_profile is not None:
        if not isinstance(reasoning_profile, dict):
            raise GenerationError("profile.reasoning must be an object")
        if worker_profile.get("protocol") != "ullm.worker.v2":
            raise GenerationError("profile.reasoning requires ullm.worker.v2")
    elif worker_profile.get("protocol") == "ullm.worker.v2":
        raise GenerationError("ullm.worker.v2 profile requires reasoning")

    tokenizer_root = Path(str(tokenizer_profile.get("root", ""))).resolve()
    tokenizer_config = _load_json(
        tokenizer_root / "tokenizer_config.json", "tokenizer config"
    )
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise GenerationError("tokenizer config has no string chat template")
    raw_tokenizer_files = tokenizer_profile.get("files")
    if not isinstance(raw_tokenizer_files, list) or not raw_tokenizer_files:
        raise GenerationError("profile.tokenizer.files must be a nonempty array")
    tokenizer_files: dict[str, str] = {}
    for item in raw_tokenizer_files:
        if not isinstance(item, str) or not item or item in tokenizer_files:
            raise GenerationError("profile.tokenizer.files is invalid")
        tokenizer_files[item] = _sha256_file(tokenizer_root / item)

    worker_binary = Path(str(worker_profile.get("binary", ""))).resolve()
    worker_sha256 = _sha256_file(worker_binary)
    product_root = Path(str(product_profile.get("root", ""))).resolve()
    package = _required_object(product_profile, "package")
    package_manifest_path = str(package.get("manifest_path", ""))
    package_manifest_sha256 = _sha256_file(product_root / package_manifest_path)

    artifact_profile = product_profile.get("artifact")
    if artifact_profile is not None and not isinstance(artifact_profile, dict):
        raise GenerationError("profile.product.artifact must be an object or null")
    artifact_manifest_path = (
        str(artifact_profile.get("manifest_path", ""))
        if isinstance(artifact_profile, dict)
        else None
    )

    configured_receipt_path = Path(str(promotion_profile.get("receipt", ""))).resolve()
    receipt_path = (
        receipt_path_override.resolve()
        if receipt_path_override is not None
        else configured_receipt_path
    )
    if receipt_path_override is not None:
        if receipt_path == configured_receipt_path:
            raise GenerationError("receipt path override must identify a separate actual receipt")
    receipt = (
        _load_json(receipt_path, "promotion receipt")
        if receipt_override is None
        else receipt_override
    )
    source_commit = _receipt_value(
        receipt,
        promotion_profile.get("source_commit_from_receipt"),
        "promotion source commit",
    )
    overlay_receipt = None
    authorization_audit: dict[str, str] | None = None
    authorization_lineage: dict[str, Any] | None = None
    readiness: dict[str, Any] | None = None
    if profile.get("format", {}).get("implementation_id") == SQ8_OVERLAY_IMPLEMENTATION_ID:
        authorization_audit = _resolve_authorization_audit(
            _receipt_object(
                receipt,
                promotion_profile.get("authorization_audit_from_receipt"),
                "SQ8 overlay authorization audit",
            ),
            "SQ8 overlay authorization audit",
        )
        readiness = _resolve_readiness(
            _receipt_object(
                receipt,
                promotion_profile.get("readiness_from_receipt"),
                "SQ8 overlay readiness",
            ),
            "SQ8 overlay readiness",
        )
        if readiness != _resolve_readiness(
            promotion_profile.get("readiness"), "SQ8 overlay profile readiness"
        ):
            raise GenerationError("SQ8 overlay profile/receipt readiness differs")
        if "authorization_lineage_from_receipt" in promotion_profile:
            value = receipt.get("authorization_lineage")
            if value is not None:
                if not isinstance(value, dict):
                    raise GenerationError("SQ8 overlay authorization lineage differs")
                authorization_lineage = value
    if validate_receipt:
        if profile.get("format", {}).get("implementation_id") == SQ8_OVERLAY_IMPLEMENTATION_ID:
            if artifact_manifest_path is None:
                raise GenerationError("SQ8 overlay profile requires an artifact manifest")
            overlay_receipt = _validate_sq8_overlay_receipt(
                profile=profile,
                promotion_profile=promotion_profile,
                receipt=receipt,
                receipt_path=receipt_path,
                profile_path=profile_path,
                worker_binary=worker_binary,
                worker_sha256=worker_sha256,
                product_root=product_root,
                artifact_manifest_path=artifact_manifest_path,
                package_manifest_path=package_manifest_path,
                package_manifest_sha256=package_manifest_sha256,
                expected_manifest_path=expected_manifest_path,
                allow_prepared=allow_prepared,
                prepared_only=prepared_only,
            )
        else:
            _validate_aq4_evidence(
                profile=profile,
                promotion_profile=promotion_profile,
                receipt=receipt,
                receipt_path=receipt_path,
                source_commit=source_commit,
                worker_binary=worker_binary,
                worker_sha256=worker_sha256,
                product_root=product_root,
                package_manifest_path=package_manifest_path,
                package_manifest_sha256=package_manifest_sha256,
            )

    artifact: dict[str, str] | None
    if artifact_profile is None:
        artifact = None
    elif isinstance(artifact_profile, dict):
        assert artifact_manifest_path is not None
        artifact = {
            "manifest_path": artifact_manifest_path,
            "manifest_sha256": _sha256_file(product_root / artifact_manifest_path),
            "content_sha256": _receipt_value(
                receipt,
                artifact_profile.get("content_sha256_from_receipt"),
                "artifact content SHA-256",
            ),
        }
    else:
        raise GenerationError("profile.product.artifact must be an object or null")

    document = {
        "schema_version": (
            "ullm.served_model.v2"
            if reasoning_profile is not None
            else "ullm.served_model.v1"
        ),
        "public": _required_object(profile, "public"),
        "generation": _required_object(profile, "generation"),
        "format": _required_object(profile, "format"),
        "tokenizer": {
            "root": os.fspath(tokenizer_root),
            "transformers_version": tokenizer_profile.get("transformers_version"),
            "class": tokenizer_profile.get("class"),
            "chat_template_sha256": hashlib.sha256(
                chat_template.encode("utf-8")
            ).hexdigest(),
            "files": tokenizer_files,
            "template_options": tokenizer_profile.get("template_options"),
        },
        "worker": {
            "protocol": worker_profile.get("protocol"),
            "binary": os.fspath(worker_binary),
            "binary_sha256": worker_sha256,
            "arguments": worker_profile.get("arguments"),
            "required_environment": worker_profile.get("required_environment"),
            "identity": worker_profile.get("identity"),
        },
        "product": {
            "root": os.fspath(product_root),
            "artifact": artifact,
            "package": {
                "manifest_path": package_manifest_path,
                "manifest_sha256": package_manifest_sha256,
            },
        },
        "promotion": {
            "source_commit": source_commit,
            "receipt": os.fspath(receipt_path),
            "receipt_sha256": (
                receipt_sha256_override
                if receipt_sha256_override is not None
                else _sha256_file(receipt_path)
            ),
        },
    }
    if profile.get("format", {}).get("implementation_id") == SQ8_OVERLAY_IMPLEMENTATION_ID:
        document["promotion"]["authorization_audit"] = authorization_audit
        if "authorization_lineage_from_receipt" in promotion_profile:
            document["promotion"]["authorization_lineage"] = authorization_lineage
        document["promotion"]["readiness"] = readiness
    if reasoning_profile is not None:
        document["reasoning"] = reasoning_profile
    if overlay_receipt is not None:
        observed_semantic_sha256 = _served_model_semantic_sha256(document)
        if observed_semantic_sha256 != overlay_receipt["served_model_semantic_sha256"]:
            raise GenerationError("SQ8 overlay served-model semantic SHA-256 differs")
    return document


def materialize(
    profile_path: Path,
    *,
    expected_manifest_path: Path | None = None,
    receipt_path_override: Path | None = None,
) -> dict[str, Any]:
    return _materialize_profile_document(
        profile_path,
        expected_manifest_path=expected_manifest_path,
        receipt_path_override=receipt_path_override,
    )


def generate(
    profile_path: Path,
    output_path: Path,
    *,
    receipt_path_override: Path | None = None,
) -> str:
    # Check the caller-supplied path before normalization; resolving first would
    # turn a symlink into its target and silently allow replacement through it.
    if output_path.is_symlink():
        raise GenerationError("output path must not be a symlink")
    output_path = output_path.resolve()
    document = materialize(
        profile_path,
        expected_manifest_path=output_path,
        receipt_path_override=receipt_path_override,
    )
    if output_path.is_symlink():
        raise GenerationError("output path must not be a symlink")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(document, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{output_path.name}.incomplete-", dir=output_path.parent
        )
        temporary = Path(raw_path)
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        model = _load_validator().load_served_model(temporary)
        _publish_create_new(temporary, output_path)
        temporary = None
        return model.manifest_sha256
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def generate_prepared_candidate(profile_path: Path, output_path: Path) -> str:
    """Materialize the immutable pre-GPU candidate manifest only."""

    if output_path.is_symlink():
        raise GenerationError("output path must not be a symlink")
    output_path = output_path.resolve()
    document = _materialize_profile_document(
        profile_path,
        expected_manifest_path=output_path,
        allow_prepared=True,
        prepared_only=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(document, ensure_ascii=True, allow_nan=False, indent=2) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(prefix=f".{output_path.name}.incomplete-", dir=output_path.parent)
        temporary = Path(raw_path)
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        model = _load_validator().load_served_model(temporary)
        _publish_create_new(temporary, output_path)
        temporary = None
        return model.manifest_sha256
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt-path-override", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        digest = generate(
            args.profile,
            args.output,
            receipt_path_override=args.receipt_path_override,
        )
    except Exception as error:
        print(f"served-model generation failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": "ullm.served_model.generation.v1",
                "manifest_sha256": digest,
                "output": os.fspath(args.output.resolve()),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
