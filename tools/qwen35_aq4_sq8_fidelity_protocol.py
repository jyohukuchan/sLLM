#!/usr/bin/env python3
"""Offline SQ8 calibration/freeze and one-shot holdout protocol.

The protocol is deliberately filesystem-only.  It validates a served SQ8
promotion receipt before calibration is allowed, recomputes all 24 calibration
metrics itself, and consumes the holdout attempt at an irreversible boundary.
It never starts a model, GPU, service, or subprocess.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import sys
from pathlib import Path
from typing import Any


MAX_ROWS = 24
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_CHUNK_ELEMENTS = 65_536
SAFE_INT_MAX = (1 << 63) - 1
SQ8_RECEIPT_SCHEMA = "ullm.qwen35_aq4_sq8_overlay_promotion.v1"
PLAN_SCHEMA = "ullm.qwen35_aq4_sq8_fidelity_plan.v1"
METRICS_SCHEMA = "ullm.qwen35_aq4_sq8_fidelity_metrics.v1"
FREEZE_SCHEMA = "ullm.qwen35_aq4_sq8_fidelity_freeze_receipt.v1"
PREFLIGHT_SCHEMA = "ullm.qwen35_aq4_sq8_fidelity_holdout_preflight.v1"
ATTEMPT_SCHEMA = "ullm.qwen35_aq4_sq8_fidelity_holdout_attempt.v1"
HOLDOUT_SCHEMA = "ullm.qwen35_aq4_sq8_fidelity_holdout_receipt.v1"
LEDGER_SCHEMA = "ullm.qwen35_aq4_sq8_fidelity_attempt_ledger.v1"
REQUEST_RE = re.compile(r"^sq8-promotion-[0-9a-f]{64}$")
HEX_RE = re.compile(r"^[0-9a-f]{64}$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")

METRIC_POLICY: dict[str, dict[str, Any]] = {
    "token_agreement_rate": {"role": "promotion", "direction": "higher", "aggregation": "wilson_lower_one_sided", "margin": None, "relative_margin": None, "absolute_floor": None, "absolute_ceiling": 1.0},
    "topk_overlap_rate_k10": {"role": "promotion", "direction": "higher", "aggregation": "mean", "margin": 0.01, "relative_margin": 0.01, "absolute_floor": 0.1, "absolute_ceiling": 1.0},
    "logits_cosine": {"role": "promotion", "direction": "higher", "aggregation": "mean", "margin": 0.01, "relative_margin": 0.01, "absolute_floor": 0.0, "absolute_ceiling": 1.0},
    "logits_relative_l2": {"role": "promotion", "direction": "lower", "aggregation": "mean", "margin": 0.05, "relative_margin": 0.05, "absolute_floor": 0.0, "absolute_ceiling": 1.0},
    "hidden_cosine": {"role": "promotion", "direction": "higher", "aggregation": "mean", "margin": 0.01, "relative_margin": 0.01, "absolute_floor": 0.0, "absolute_ceiling": 1.0},
    "hidden_relative_l2": {"role": "promotion", "direction": "lower", "aggregation": "mean", "margin": 0.05, "relative_margin": 0.05, "absolute_floor": 0.0, "absolute_ceiling": 1.0},
    "hidden_max_abs": {"role": "diagnostic_only", "direction": "diagnostic", "aggregation": "max", "margin": None, "relative_margin": None, "absolute_floor": None, "absolute_ceiling": None},
    "bf16_top1_retained_in_aq4_top10_rate": {"role": "promotion", "direction": "higher", "aggregation": "wilson_lower_one_sided", "margin": None, "relative_margin": None, "absolute_floor": None, "absolute_ceiling": 1.0},
}
BINARY_METRICS = {"token_agreement_rate", "bf16_top1_retained_in_aq4_top10_rate"}
RELATIVE_L2_METRICS = {"logits_relative_l2", "hidden_relative_l2"}
WILSON_Z = 1.6448536269514722


class ProtocolError(ValueError):
    """A malformed, stale, or unsafe protocol artifact."""


def _int(value: Any, label: str, *, minimum: int = 0, maximum: int = SAFE_INT_MAX) -> int:
    """Validate a protocol integer without Python bool/float aliases."""

    if type(value) is not int or value < minimum or value > maximum:
        raise ProtocolError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _float(value: Any, label: str, *, minimum: float = 0.0, maximum: float = float("inf")) -> float:
    """Validate a protocol float without accepting integer aliases."""

    if type(value) is not float or not math.isfinite(value) or value < minimum or value > maximum:
        raise ProtocolError(f"{label} must be a finite float in [{minimum}, {maximum}]")
    return value


def _exact_int(value: Any, expected: int, label: str) -> int:
    result = _int(value, label, minimum=expected, maximum=expected)
    return result


def _strict_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python bool/int/float coercion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_strict_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_strict_equal(a, b) for a, b in zip(left, right))
    return left == right


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise ProtocolError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _constant(value: str) -> Any:
    raise ProtocolError(f"non-finite JSON constant: {value}")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path, label: str) -> str:
    _regular(path, label)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ProtocolError(f"{label} exceeds bounded size")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, label: str) -> None:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ProtocolError(f"{label} is unavailable") from error
    if path.is_symlink() or not path.is_file() or metadata.st_nlink != 1:
        raise ProtocolError(f"{label} must be a regular non-symlink file")
    current = path.absolute().anchor and Path(path.absolute().anchor) or Path("/")
    for component in path.absolute().parts[1:]:
        current /= component
        if current.is_symlink():
            raise ProtocolError(f"{label} has a symlink component")


def load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _regular(path, label)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ProtocolError(f"{label} exceeds bounded size")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return value, raw


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _regular(path, label)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ProtocolError(f"{label} exceeds bounded size")
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if number > MAX_ROWS:
                raise ProtocolError(f"{label} has more than {MAX_ROWS} rows")
            if len(line) > MAX_JSON_BYTES:
                raise ProtocolError(f"{label} line {number} exceeds bounded size")
            try:
                value = json.loads(line.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ProtocolError(f"invalid {label} line {number}") from error
            if not isinstance(value, dict):
                raise ProtocolError(f"{label} line {number} is not an object")
            rows.append(value)
    return rows


def atomic_json(path: Path, value: Any) -> None:
    if os.path.lexists(path):
        raise ProtocolError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".incomplete", dir=path.parent)
        temporary = Path(raw_path)
        raw = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        with os.fdopen(descriptor, "wb", buffering=0) as stream:
            os.fchmod(stream.fileno(), 0o444)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # link(2) is the no-replace publication boundary.  Two writers can
        # prepare staging files, but exactly one can publish the final path.
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _hex(value: Any, label: str, *, forty: bool = False) -> str:
    pattern = HEX40_RE if forty else HEX_RE
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ProtocolError(f"{label} is not a lowercase hexadecimal digest")
    return value


def _request(value: Any) -> str:
    if not isinstance(value, str) or REQUEST_RE.fullmatch(value) is None:
        raise ProtocolError("SQ8 request_id is invalid")
    return value


def _ref(path: Path, label: str, *, base: Path | None = None) -> dict[str, str]:
    _regular(path, label)
    resolved = path.resolve()
    if base is not None:
        try:
            relative = resolved.relative_to(base.resolve())
        except ValueError as error:
            raise ProtocolError(f"{label} must be below its receipt directory") from error
        if not relative.parts or any(part in (".", "..", "") for part in relative.parts):
            raise ProtocolError(f"{label} path is unsafe")
        raw_path = str(relative)
    else:
        raw_path = str(resolved)
    return {"path": raw_path, "sha256": sha_file(resolved, label)}


def _inside(path: Path, base: Path, label: str) -> Path:
    """Resolve an evidence path and keep it below the receipt directory."""

    resolved = (base / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as error:
        raise ProtocolError(f"{label} must be inside its receipt directory") from error
    _regular(resolved, label)
    return resolved


def _absolute_regular(path_value: Any, label: str) -> Path:
    """Resolve an evidence path without permitting path rewriting or links."""

    if not isinstance(path_value, (str, Path)):
        raise ProtocolError(f"{label} path is invalid")
    path = Path(path_value)
    if not path.is_absolute() or path != path.resolve():
        raise ProtocolError(f"{label} path must be canonical absolute")
    _regular(path, label)
    return path


def _reference(value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ProtocolError(f"{label} reference is incomplete")
    digest = _hex(value.get("sha256"), f"{label} SHA")
    target = _absolute_regular(value.get("path"), label)
    if sha_file(target, label) != digest:
        raise ProtocolError(f"{label} SHA differs")
    return target, digest


def _validate_lineage_reference(value: Any, request_id: str) -> None:
    """Validate the immutable lineage reference and its current audit binding."""

    try:
        from tools import qwen35_aq4_sq8_authorization_lineage as lineage_tool
    except ImportError:
        lineage_tool = None

    if not isinstance(value, dict):
        raise ProtocolError("SQ8 authorization lineage is incomplete")
    schema = value.get("schema_version")
    expected = {
        "schema_version", "input_path", "runtime_path", "sha256", "entries_sha256",
    }
    if schema == "ullm.sq8_authorization_lineage_ref.v2":
        expected |= {"entry_count", "current_implementation_audit"}
    if schema not in {
        "ullm.sq8_authorization_lineage_ref.v1",
        "ullm.sq8_authorization_lineage_ref.v2",
    } or set(value) != expected:
        raise ProtocolError("SQ8 authorization lineage shape differs")
    for key in ("sha256", "entries_sha256"):
        _hex(value.get(key), f"authorization lineage {key}")
    for key in ("input_path", "runtime_path"):
        _absolute_regular(value.get(key), f"authorization lineage {key}")
    if schema == "ullm.sq8_authorization_lineage_ref.v2":
        _int(value.get("entry_count"), "authorization lineage entry count", minimum=1)
        current = value.get("current_implementation_audit")
        current_path, current_sha = _reference(current, "authorization lineage current audit")
        if current_path == Path(value["input_path"]).resolve() or current_path == Path(value["runtime_path"]).resolve():
            raise ProtocolError("authorization lineage current audit path aliases manifest")
    if lineage_tool is not None:
        try:
            lineage_tool.validate_reference(value)
        except Exception as error:
            raise ProtocolError("SQ8 authorization lineage manifest is not independently valid") from error
    # The request is encoded in the current runtime audit source.  The exact
    # full manifest is independently validated by the receipt writer; this
    # phase rechecks the immutable paths and digest lineage without executing
    # any external tool.
    runtime_value, _ = load_json(Path(value["runtime_path"]), "authorization lineage runtime manifest")
    fixed_request = runtime_value.get("source", {}).get("fixed_request_id")
    if fixed_request is not None and fixed_request != request_id:
        raise ProtocolError("authorization lineage request ID differs")


def _served_semantic_sha256(path: Path) -> str:
    document, _ = load_json(path, "SQ8 served model")
    value = json.loads(json.dumps(document, ensure_ascii=True, allow_nan=False))
    promotion = value.get("promotion")
    if isinstance(promotion, dict):
        promotion.pop("receipt_sha256", None)
    return sha_bytes(canonical(value))


def _validate_readiness(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"schema", "container", "network", "endpoint"}:
        raise ProtocolError("SQ8 readiness identity shape differs")
    if value.get("schema") != "ullm.bridge_container_readiness.v1":
        raise ProtocolError("SQ8 readiness schema differs")
    container = value.get("container")
    if not isinstance(container, dict) or set(container) != {"name", "id", "image_id", "config_image"} or container.get("name") != "open-webui" or not isinstance(container.get("id"), str) or re.fullmatch(r"[0-9a-f]{64}", container["id"]) is None or not isinstance(container.get("image_id"), str) or re.fullmatch(r"sha256:[0-9a-f]{64}", container["image_id"]) is None or not isinstance(container.get("config_image"), str) or not container["config_image"]:
        raise ProtocolError("SQ8 readiness container identity differs")
    network = value.get("network")
    network_id = network.get("id") if isinstance(network, dict) else None
    if not isinstance(network, dict) or set(network) != {"name", "id", "driver", "bridge_interface"} or not isinstance(network.get("name"), str) or not network["name"] or not isinstance(network_id, str) or re.fullmatch(r"[0-9a-f]{64}", network_id) is None or network.get("driver") != "bridge" or network.get("bridge_interface") != f"br-{network_id[:12]}":
        raise ProtocolError("SQ8 readiness network identity differs")
    body = '{"status":"ready"}'
    endpoint = value.get("endpoint")
    expected_endpoint = {"url": "http://172.20.0.1:8000/readyz", "path": "/readyz", "expected_status": 200, "expected_body": body, "expected_body_sha256": hashlib.sha256(body.encode("ascii")).hexdigest(), "timeout_seconds": 5}
    if not _strict_equal(endpoint, expected_endpoint):
        raise ProtocolError("SQ8 readiness endpoint identity differs")


def _validate_artifact_inventory(value: Any) -> None:
    fields = {
        "root", "uid", "gid", "directory_count", "directory_mode", "regular_file_count",
        "regular_file_bytes", "regular_file_mode", "regular_file_nlink", "symlink_count",
        "special_count", "entries",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ProtocolError("SQ8 artifact inventory shape differs")
    _int(value.get("uid"), "SQ8 artifact inventory uid")
    _int(value.get("gid"), "SQ8 artifact inventory gid")
    for key in ("directory_count", "regular_file_count", "regular_file_bytes", "symlink_count", "special_count"):
        _int(value.get(key), f"SQ8 artifact inventory {key}")
    _exact_int(value.get("regular_file_nlink"), 1, "SQ8 artifact inventory regular nlink")
    if not isinstance(value.get("root"), str) or not value["root"] or value.get("directory_mode") != "0555" or value.get("regular_file_mode") != "0444":
        raise ProtocolError("SQ8 artifact inventory identity differs")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries or len(entries) > MAX_ROWS * 1024:
        raise ProtocolError("SQ8 artifact inventory entries differ")
    entry_fields = {"path", "kind", "mode", "uid", "gid", "nlink", "bytes"}
    kind_counts = {"directory": 0, "regular": 0, "symlink": 0, "special": 0}
    regular_bytes = 0
    seen_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != entry_fields or not isinstance(entry.get("path"), str) or not entry["path"] or entry.get("kind") not in kind_counts or not isinstance(entry.get("mode"), str):
            raise ProtocolError("SQ8 artifact inventory entry differs")
        entry_path = Path(entry["path"])
        if entry["path"] != "." and (entry_path.is_absolute() or any(part in ("", ".", "..") for part in entry_path.parts)):
            raise ProtocolError("SQ8 artifact inventory entry path is unsafe")
        if entry["path"] in seen_paths:
            raise ProtocolError("SQ8 artifact inventory entry is duplicated")
        seen_paths.add(entry["path"])
        kind_counts[entry["kind"]] += 1
        _int(entry.get("uid"), "SQ8 artifact inventory entry uid")
        _int(entry.get("gid"), "SQ8 artifact inventory entry gid")
        _int(entry.get("nlink"), "SQ8 artifact inventory entry nlink", minimum=1)
        size = _int(entry.get("bytes"), "SQ8 artifact inventory entry bytes")
        if entry.get("uid") != value["uid"] or entry.get("gid") != value["gid"]:
            raise ProtocolError("SQ8 artifact inventory ownership differs")
        expected_mode = "0555" if entry["kind"] == "directory" else "0444"
        if entry["mode"] != expected_mode:
            raise ProtocolError("SQ8 artifact inventory mode differs")
        if entry["kind"] == "regular":
            _exact_int(entry.get("nlink"), 1, "SQ8 artifact inventory regular nlink")
            regular_bytes += size
    if kind_counts["directory"] != value["directory_count"] or kind_counts["regular"] != value["regular_file_count"] or kind_counts["symlink"] != value["symlink_count"] or kind_counts["special"] != value["special_count"] or regular_bytes != value["regular_file_bytes"]:
        raise ProtocolError("SQ8 artifact inventory aggregate differs")
    if value["regular_file_count"] < 1 or value["directory_count"] < 1 or value["symlink_count"] != 0 or value["special_count"] != 0:
        raise ProtocolError("SQ8 artifact inventory safety differs")


def _validate_release_and_profile(
    receipt: dict[str, Any], path: Path, prepared_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Cross-bind release files, package/artifact manifests, and profile."""

    release = receipt.get("release")
    if not isinstance(release, dict) or set(release) != {"worker", "profile", "served_model"}:
        raise ProtocolError("SQ8 release identity is incomplete")
    worker = release.get("worker")
    profile_ref = release.get("profile")
    served = release.get("served_model")
    if not isinstance(worker, dict) or set(worker) != {"path", "sha256", "bytes", "mode", "nlink"}:
        raise ProtocolError("SQ8 worker identity shape differs")
    if not isinstance(profile_ref, dict) or set(profile_ref) != {"path", "sha256"}:
        raise ProtocolError("SQ8 profile identity shape differs")
    if not isinstance(served, dict) or set(served) != {"path", "semantic_sha256"}:
        raise ProtocolError("SQ8 served-model identity shape differs")
    worker_path, worker_sha = _reference({"path": worker.get("path"), "sha256": worker.get("sha256")}, "SQ8 worker")
    metadata = worker_path.stat()
    _exact_int(worker.get("bytes"), metadata.st_size, "SQ8 worker bytes")
    _exact_int(worker.get("nlink"), 1, "SQ8 worker nlink")
    if worker.get("mode") != "0555":
        raise ProtocolError("SQ8 worker topology differs")
    profile_path, _ = _reference(profile_ref, "SQ8 profile")
    served_path = _absolute_regular(served.get("path"), "SQ8 served model")
    served_sha = _hex(served.get("semantic_sha256"), "served-model semantic SHA")
    if _served_semantic_sha256(served_path) != served_sha:
        raise ProtocolError("SQ8 served-model semantic SHA differs")
    profile, _ = load_json(profile_path, "SQ8 overlay profile")
    if profile.get("schema_version") != "ullm.served_model.profile.v1" or profile.get("format", {}).get("implementation_id") != "qwen35_aq4_sq8_linear_qkv_z_overlay_v1":
        raise ProtocolError("SQ8 overlay profile implementation differs")
    profile_worker = profile.get("worker")
    if not isinstance(profile_worker, dict) or _absolute_regular(profile_worker.get("binary"), "SQ8 profile worker") != worker_path:
        raise ProtocolError("SQ8 profile worker path differs")
    worker_identity = profile_worker.get("identity")
    if not isinstance(worker_identity, dict) or worker_identity.get("execution_profile") != "rdna4_aq4_resident_sq8_linear_qkv_z_overlay":
        raise ProtocolError("SQ8 profile execution profile differs")
    promotion = profile.get("promotion")
    if not isinstance(promotion, dict):
        raise ProtocolError("SQ8 profile promotion binding is missing")
    required_profile = {
        "receipt", "source_commit_from_receipt", "required_schema_version",
        "overlay_from_receipt", "release_from_receipt", "package_from_receipt",
        "actual_evidence_from_receipt", "request_id_from_receipt",
        "authorization_audit_from_receipt", "readiness_from_receipt", "readiness",
        "release_source_commit",
    }
    lineage_profile = {"authorization_lineage_from_receipt", "authorization_lineage"}
    if set(promotion) not in (required_profile, required_profile | lineage_profile):
        raise ProtocolError("SQ8 profile promotion binding shape differs")
    _validate_readiness(receipt.get("readiness"))
    if Path(str(promotion.get("receipt"))).resolve() != prepared_path.resolve() or promotion.get("required_schema_version") != SQ8_RECEIPT_SCHEMA or promotion.get("source_commit_from_receipt") != ["source_commit"] or promotion.get("overlay_from_receipt") != ["overlay"] or promotion.get("release_from_receipt") != ["release"] or promotion.get("package_from_receipt") != ["package"] or promotion.get("actual_evidence_from_receipt") != ["actual"] or promotion.get("request_id_from_receipt") != ["request_id"] or promotion.get("authorization_audit_from_receipt") != ["authorization_audit"] or promotion.get("readiness_from_receipt") != ["readiness"] or not _strict_equal(promotion.get("readiness"), receipt.get("readiness")) or promotion.get("release_source_commit") != receipt.get("source_commit"):
        raise ProtocolError("SQ8 profile/receipt identity differs")
    if "authorization_lineage_from_receipt" in promotion:
        lineage = receipt.get("authorization_lineage")
        if lineage is None or promotion.get("authorization_lineage_from_receipt") != ["authorization_lineage"] or promotion.get("authorization_lineage") != lineage:
            raise ProtocolError("SQ8 profile/receipt authorization lineage differs")
    elif "authorization_lineage" in receipt:
        raise ProtocolError("SQ8 receipt has unbound authorization lineage")
    product = profile.get("product")
    if not isinstance(product, dict) or not isinstance(product.get("root"), str):
        raise ProtocolError("SQ8 profile product binding is missing")
    product_root = Path(product["root"])
    if not product_root.is_absolute() or product_root != product_root.resolve() or product_root.is_symlink() or not product_root.is_dir():
        raise ProtocolError("SQ8 product root path differs")
    artifact = product.get("artifact")
    package = product.get("package")
    if not isinstance(artifact, dict) or not isinstance(package, dict) or artifact.get("content_sha256_from_receipt") != ["overlay", "content_sha256"] or package.get("manifest_path") is None:
        raise ProtocolError("SQ8 profile artifact/package binding differs")
    artifact_path = _absolute_regular(product_root / str(artifact.get("manifest_path")), "SQ8 overlay binding manifest")
    package_path = _absolute_regular(product_root / str(package.get("manifest_path")), "SQ8 package manifest")
    overlay = receipt.get("overlay")
    package_receipt = receipt.get("package")
    if not isinstance(overlay, dict) or overlay.get("binding_manifest_path") != str(artifact_path) or sha_file(artifact_path, "SQ8 overlay binding manifest") != overlay.get("binding_manifest_sha256") or not isinstance(package_receipt, dict) or package_receipt.get("manifest_path") != str(package_path) or sha_file(package_path, "SQ8 package manifest") != package_receipt.get("manifest_sha256"):
        raise ProtocolError("SQ8 package/artifact receipt binding differs")
    binding, _ = load_json(artifact_path, "SQ8 overlay binding manifest")
    tensor_names = binding.get("tensor_names")
    if binding.get("schema_version") != "ullm.qwen35_aq4_sq8_qkv_z_overlay.v2" or binding.get("implementation_id") != "qwen35_aq4_sq8_linear_qkv_z_overlay_v1" or binding.get("format_id") != "AQ4_0" or binding.get("overlay_format_id") != "SQ8_0" or not isinstance(tensor_names, list) or len(tensor_names) != 48 or len(set(tensor_names)) != 48 or binding.get("content_sha256") != overlay.get("content_sha256") or binding.get("tensor_set_sha256") != overlay.get("tensor_set_sha256") or not isinstance(binding.get("package"), dict) or binding["package"].get("manifest_sha256") != package_receipt.get("manifest_sha256"):
        raise ProtocolError("SQ8 overlay binding manifest identity differs")
    return worker, served, package_receipt


def _actual_receipt(path: Path) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    receipt, raw = load_json(path, "SQ8 promotion receipt")
    receipt_keys = {"schema_version", "status", "request_id", "source_commit", "source_provenance", "release", "overlay", "package", "authorization_audit", "readiness", "actual"}
    if set(receipt) not in (receipt_keys, receipt_keys | {"authorization_lineage"}):
        raise ProtocolError("SQ8 promotion receipt has unknown or missing fields")
    if receipt.get("schema_version") != SQ8_RECEIPT_SCHEMA or receipt.get("status") != "actual_verified":
        raise ProtocolError("actual_verified SQ8 receipt is required")
    _request(receipt.get("request_id"))
    _hex(receipt.get("source_commit"), "source commit", forty=True)
    source = receipt.get("source_provenance")
    if not isinstance(source, dict) or set(source) != {"tree_sha256", "archive_sha256"}:
        raise ProtocolError("SQ8 source provenance is incomplete")
    _hex(source.get("tree_sha256"), "source tree", forty=True)
    _hex(source.get("archive_sha256"), "source archive")
    overlay = receipt.get("overlay")
    overlay_keys = {"binding_manifest_path", "binding_manifest_sha256", "content_sha256", "tensor_set_sha256", "tensor_count", "artifact_inventory"}
    if not isinstance(overlay, dict) or set(overlay) != overlay_keys:
        raise ProtocolError("SQ8 overlay tensor count must be 48")
    _exact_int(overlay.get("tensor_count"), 48, "SQ8 overlay tensor count")
    content = _hex(overlay.get("content_sha256"), "overlay content")
    tensor_set = _hex(overlay.get("tensor_set_sha256"), "overlay tensor set")
    _validate_artifact_inventory(overlay.get("artifact_inventory"))
    actual = receipt.get("actual")
    required = {"status", "required", "prepared_receipt", "maintenance_evidence", "executor_record", "gpu_exclusive_preflight", "telemetry", "telemetry_binding", "manifest_identity", "output_identity"}
    if not isinstance(actual, dict) or set(actual) != required or actual.get("status") != "actual_verified" or actual.get("required") is not True:
        raise ProtocolError("SQ8 actual evidence is incomplete")
    request_id = receipt["request_id"]
    prepared_ref = actual["prepared_receipt"]
    if not isinstance(prepared_ref, dict) or set(prepared_ref) != {"path", "sha256"}:
        raise ProtocolError("prepared receipt reference is incomplete")
    _hex(prepared_ref.get("sha256"), "prepared receipt SHA")
    prepared_path = _absolute_regular(prepared_ref.get("path"), "prepared receipt")
    if prepared_path == path.resolve():
        raise ProtocolError("prepared and actual receipt cannot be the same file")
    if sha_file(prepared_path, "prepared receipt") != prepared_ref["sha256"]:
        raise ProtocolError("prepared receipt SHA differs")
    prepared, _ = load_json(prepared_path, "prepared receipt")
    prepared_keys = {"schema_version", "status", "request_id", "source_commit", "source_provenance", "release", "overlay", "package", "authorization_audit", "readiness", "actual"}
    if set(prepared) not in (prepared_keys, prepared_keys | {"authorization_lineage"}):
        raise ProtocolError("prepared receipt has unknown or missing fields")
    if prepared.get("schema_version") != SQ8_RECEIPT_SCHEMA or prepared.get("status") != "prepared_not_executed" or prepared.get("request_id") != request_id:
        raise ProtocolError("prepared receipt state differs")
    for key in ("source_commit", "source_provenance", "overlay", "package"):
        if not _strict_equal(prepared.get(key), receipt.get(key)):
            raise ProtocolError(f"prepared receipt {key} differs")
    if not _strict_equal(prepared.get("readiness"), receipt.get("readiness")) or not _strict_equal(prepared.get("authorization_audit"), receipt.get("authorization_audit")):
        raise ProtocolError("prepared receipt readiness/authorization differs")
    if "authorization_lineage" in receipt and not _strict_equal(prepared.get("authorization_lineage"), receipt.get("authorization_lineage")):
        raise ProtocolError("prepared receipt authorization lineage differs")
    prepared_release = prepared.get("release")
    release = receipt.get("release")
    if not isinstance(prepared_release, dict) or not isinstance(release, dict):
        raise ProtocolError("prepared/release identity is incomplete")
    for component in ("worker", "profile"):
        if not _strict_equal(prepared_release.get(component), release.get(component)):
            raise ProtocolError(f"prepared receipt release {component} differs")
    if prepared_release.get("served_model", {}).get("path") != release.get("served_model", {}).get("path"):
        raise ProtocolError("prepared receipt served-model path differs")
    telemetry = actual["telemetry"]
    telemetry_keys = {"schema_version", "projection", "diagnostic_host_staging"}
    if not isinstance(telemetry, dict) or set(telemetry) != telemetry_keys or telemetry.get("schema_version") != "ullm.qwen35_aq4.sq8_promotion_telemetry.v1":
        raise ProtocolError("SQ8 telemetry schema differs")
    projection = telemetry.get("projection")
    projection_keys = {"single_matvec_count", "batch_matvec_count", "pair_matvec_count", "triple_matvec_count", "fallback_count"}
    if not isinstance(projection, dict) or set(projection) != projection_keys:
        raise ProtocolError("SQ8 telemetry projection differs")
    for key in projection_keys:
        _int(projection[key], f"SQ8 telemetry projection {key}")
    if projection["batch_matvec_count"] <= 0 or projection["pair_matvec_count"] <= 0 or any(projection[key] != 0 for key in ("single_matvec_count", "triple_matvec_count", "fallback_count")):
        raise ProtocolError("SQ8 telemetry projection differs")
    staging = telemetry.get("diagnostic_host_staging")
    if not isinstance(staging, dict) or set(staging) != {"read_count", "write_count", "read_bytes", "write_bytes"}:
        raise ProtocolError("SQ8 telemetry host staging differs")
    for key in staging:
        _exact_int(staging[key], 0, f"SQ8 telemetry host staging {key}")
    binding = actual["telemetry_binding"]
    if not isinstance(binding, dict) or set(binding) != {"schema_version", "request_id", "hash_encoding", "telemetry_sha256"}:
        raise ProtocolError("SQ8 telemetry binding shape differs")
    if binding.get("request_id") != request_id or binding.get("hash_encoding") != "canonical_json_ascii_sort_keys_compact_v1" or binding.get("telemetry_sha256") != sha_bytes(canonical(telemetry)):
        raise ProtocolError("SQ8 telemetry binding differs")
    identity = actual["manifest_identity"]
    package = receipt.get("package")
    if not isinstance(package, dict):
        raise ProtocolError("SQ8 package identity is incomplete")
    expected_identity_keys = {"implementation_id", "execution_profile", "artifact_content_sha256", "artifact_manifest_sha256", "package_manifest_sha256"}
    if not isinstance(identity, dict) or set(identity) != expected_identity_keys or identity.get("implementation_id") != "qwen35_aq4_sq8_linear_qkv_z_overlay_v1" or identity.get("execution_profile") != "rdna4_aq4_resident_sq8_linear_qkv_z_overlay" or identity.get("artifact_content_sha256") != content or identity.get("artifact_manifest_sha256") != overlay.get("binding_manifest_sha256") or identity.get("package_manifest_sha256") != package.get("manifest_sha256"):
        raise ProtocolError("SQ8 manifest identity differs")
    output = actual["output_identity"]
    if not isinstance(output, dict) or set(output) != {"token_count", "token_ids_sha256", "token_ids_recorded"} or output.get("token_ids_recorded") is not False:
        raise ProtocolError("SQ8 token output identity differs")
    _exact_int(output.get("token_count"), 2, "SQ8 token count")
    _hex(output.get("token_ids_sha256"), "SQ8 token IDs SHA")
    worker, served, package = _validate_release_and_profile(receipt, path, prepared_path)
    inventory_root = overlay["artifact_inventory"]["root"]
    expected_inventory_root = str(Path(overlay["binding_manifest_path"]).resolve().parent)
    if inventory_root != expected_inventory_root:
        raise ProtocolError("SQ8 artifact inventory root differs from binding root")
    _hex(package.get("manifest_sha256"), "package manifest SHA")
    authorization = receipt.get("authorization_audit")
    if not isinstance(authorization, dict):
        raise ProtocolError("SQ8 actual authorization audit is required")
    authorization_path, _ = _reference(authorization, "SQ8 authorization audit")
    authorization_value, _ = load_json(authorization_path, "SQ8 authorization audit")
    if authorization_value.get("verdict") != "implementation_ready" or authorization_value.get("actual") != "not_executed" or authorization_value.get("fixed_request_id") != request_id:
        raise ProtocolError("SQ8 authorization audit identity differs")
    if "authorization_lineage" in receipt:
        _validate_lineage_reference(receipt["authorization_lineage"], request_id)
    gpu = actual["gpu_exclusive_preflight"]
    if not isinstance(gpu, dict) or set(gpu) != {"mode", "stable_observation_count", "worker_pids", "amd_smi_owners", "kfd_owners", "lock", "vram_headroom_bytes"} or gpu.get("mode") != "maintenance_stable2" or gpu.get("worker_pids") != [] or gpu.get("amd_smi_owners") != [] or gpu.get("kfd_owners") != [] or gpu.get("lock") != {"path": "/run/ullm/device-1.lock", "free": True}:
        raise ProtocolError("SQ8 GPU exclusive preflight evidence differs")
    _exact_int(gpu.get("stable_observation_count"), 2, "SQ8 GPU stable observation count")
    _int(gpu.get("vram_headroom_bytes"), "SQ8 GPU VRAM headroom", minimum=1)
    for key in ("maintenance_evidence", "executor_record"):
        ref = actual[key]
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            raise ProtocolError(f"{key} reference is incomplete")
        _hex(ref.get("sha256"), f"{key} SHA")
        target = _inside(Path(str(ref["path"])), path.parent, key)
        if sha_file(target, key) != ref["sha256"]:
            raise ProtocolError(f"{key} SHA differs")
    maintenance_value, _ = load_json(path.parent / str(actual["maintenance_evidence"]["path"]), "SQ8 maintenance evidence")
    if maintenance_value.get("promotion_request_id") != request_id or maintenance_value.get("schema_version") != "ullm.qwen35_aq4.sq8_overlay_gpu_promotion_maintenance.v1" or maintenance_value.get("status") != "passed" or maintenance_value.get("failure") is not None:
        raise ProtocolError("SQ8 maintenance request/status differs")
    _exact_int(maintenance_value.get("actual_run_count"), 1, "SQ8 maintenance actual run count")
    if not _strict_equal(maintenance_value.get("candidate_pre"), maintenance_value.get("candidate_post")):
        raise ProtocolError("SQ8 maintenance candidate identity changed")
    restore = maintenance_value.get("restore")
    if not isinstance(restore, dict) or restore.get("attempted") is not True or restore.get("passed") is not True:
        raise ProtocolError("SQ8 maintenance restore evidence differs")
    _exact_int(maintenance_value.get("vram_headroom_bytes"), gpu["vram_headroom_bytes"], "SQ8 maintenance VRAM headroom")
    if maintenance_value.get("vram_headroom_bytes") != gpu["vram_headroom_bytes"]:
        raise ProtocolError("SQ8 maintenance VRAM evidence differs")
    observations = maintenance_value.get("stopped_observations")
    if not isinstance(observations, list) or len(observations) < 2:
        raise ProtocolError("SQ8 maintenance stable observations differ")
    for observation in observations[-2:]:
        if not isinstance(observation, dict) or not isinstance(observation.get("service"), dict) or not isinstance(observation.get("owners"), dict):
            raise ProtocolError("SQ8 maintenance observation shape differs")
        service = observation["service"]
        owners = observation["owners"]
        if service.get("active") is not False or service.get("running") is not False or service.get("lock_owned") is not False or any(owners.get(key) != [] for key in ("worker_pids", "amd_pids", "kfd_pids")):
            raise ProtocolError("SQ8 maintenance GPU isolation differs")
        _exact_int(service.get("main_pid"), 0, "SQ8 maintenance main PID")
        _exact_int(service.get("worker_pid"), 0, "SQ8 maintenance worker PID")
    executor_value, _ = load_json(path.parent / str(actual["executor_record"]["path"]), "SQ8 executor record")
    if executor_value.get("schema_version") != "ullm.production_executor_record.v1" or executor_value.get("status") != "ok":
        raise ProtocolError("SQ8 executor record schema/status differs")
    executor_evidence = executor_value.get("sq8_promotion_evidence")
    if not isinstance(executor_evidence, dict) or set(executor_evidence) != {"schema_version", "request_id", "manifest_identity", "telemetry", "telemetry_binding", "output_identity"} or executor_evidence.get("schema_version") != "ullm.qwen35_aq4.sq8_promotion_executor.v1" or executor_evidence.get("request_id") != request_id or not _strict_equal(executor_evidence.get("manifest_identity"), identity) or not _strict_equal(executor_evidence.get("telemetry"), telemetry) or not _strict_equal(executor_evidence.get("telemetry_binding"), binding) or not _strict_equal(executor_evidence.get("output_identity"), output):
        raise ProtocolError("SQ8 executor evidence does not bind actual receipt")
    return receipt, raw, {"request_id": request_id, "content_sha256": content, "tensor_set_sha256": tensor_set, "source_commit": receipt["source_commit"], "source_tree_sha256": source["tree_sha256"], "source_archive_sha256": source["archive_sha256"], "token_ids_sha256": output["token_ids_sha256"], "telemetry_binding": binding, "maintenance_evidence": actual["maintenance_evidence"], "executor_record": actual["executor_record"], "prepared_receipt": prepared_ref, "served_model": served, "worker": worker, "package": package, "vram_headroom_bytes": gpu.get("vram_headroom_bytes", 1)}


def policy() -> dict[str, Any]:
    metrics = {}
    for name, spec in METRIC_POLICY.items():
        item = dict(spec)
        item["sample_minimum"] = MAX_ROWS
        item["observed_domain"] = "[0,1]" if name not in {"hidden_max_abs"} else "[0,+inf)"
        if name in RELATIVE_L2_METRICS:
            item["pathological_rejection_ceiling"] = 1.0
        metrics[name] = item
    return {"schema_version": "ullm.qwen35_aq4_sq8_fidelity_policy.v1", "status": "formula_frozen_unbound", "promotion_eligible": False, "n": MAX_ROWS, "metrics": metrics, "relative_l2_rejection": {"ceiling": 1.0, "action": "reject any observed relative-L2 > 1 before aggregation"}, "holdout_evaluation_allowed_once": True, "retry_permitted": False, "attempt_boundary": {"remaining_before": 1, "remaining_after": 0, "failure_consumes_attempt": True}}


def _split(split_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest, manifest_raw = load_json(split_root / "split-manifest.json", "SQ8 split manifest")
    policy_value, policy_raw = load_json(split_root / "policy.json", "SQ8 policy")
    manifest_fields = {
        "schema_version", "status", "selected_case_count", "calibration_case_count",
        "holdout_case_count", "calibration_sha256", "holdout_sha256", "policy_sha256",
    }
    if set(manifest) != manifest_fields or manifest.get("schema_version") != "ullm.qwen35_aq4_sq8_fidelity_split.v1" or manifest.get("status") != "ready_for_calibration":
        raise ProtocolError("SQ8 split manifest schema/status differs")
    _exact_int(manifest.get("selected_case_count"), 48, "SQ8 selected case count")
    _exact_int(manifest.get("calibration_case_count"), 24, "SQ8 calibration case count")
    _exact_int(manifest.get("holdout_case_count"), 24, "SQ8 holdout case count")
    if not _strict_equal(policy_value, policy()):
        raise ProtocolError("SQ8 policy formula is not frozen")
    calibration = read_jsonl(split_root / "calibration-cases.jsonl", "SQ8 calibration cases")
    holdout = read_jsonl(split_root / "holdout-cases.jsonl", "SQ8 holdout cases")
    if len(calibration) != MAX_ROWS or len(holdout) != MAX_ROWS:
        raise ProtocolError("SQ8 split must contain exactly 24 rows per subset")
    seen: set[str] = set()
    for subset, rows in (("calibration", calibration), ("holdout", holdout)):
        for row in rows:
            if row.get("subset") != subset:
                raise ProtocolError(f"SQ8 {subset} row state differs")
            for key in ("step", "row_count", "cached_prefix_tokens", "generated_tokens", "prompt_tokens", "context_tokens", "prefill_requested_m", "resolved_m"):
                _int(row.get(key), f"SQ8 {subset} {key}", maximum=SAFE_INT_MAX)
            _exact_int(row.get("step"), 0, f"SQ8 {subset} step")
            _exact_int(row.get("row_count"), 1, f"SQ8 {subset} row count")
            _exact_int(row.get("cached_prefix_tokens"), 0, f"SQ8 {subset} cached prefix tokens")
            _exact_int(row.get("generated_tokens"), 0, f"SQ8 {subset} generated tokens")
            if row.get("prompt_tokens") != row.get("context_tokens"):
                raise ProtocolError(f"SQ8 {subset} row state differs")
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or not case_id or case_id in seen:
                raise ProtocolError("SQ8 case identity is not disjoint")
            seen.add(case_id)
            for key in ("case_sha256", "fixture_sha256", "prompt_token_ids_sha256", "context_token_ids_sha256"):
                _hex(row.get(key), f"SQ8 {subset} {key}")
    if manifest.get("calibration_sha256") != sha_file(split_root / "calibration-cases.jsonl", "SQ8 calibration cases") or manifest.get("holdout_sha256") != sha_file(split_root / "holdout-cases.jsonl", "SQ8 holdout cases") or manifest.get("policy_sha256") != sha_file(split_root / "policy.json", "SQ8 policy"):
        raise ProtocolError("SQ8 split manifest file bindings differ")
    return ({"sha256": sha_bytes(manifest_raw), "path": str((split_root / "split-manifest.json").resolve()), "rows": calibration}, {"sha256": sha_file(split_root / "calibration-cases.jsonl", "SQ8 calibration cases"), "path": str((split_root / "calibration-cases.jsonl").resolve()), "rows": calibration}, {"sha256": sha_file(split_root / "holdout-cases.jsonl", "SQ8 holdout cases"), "path": str((split_root / "holdout-cases.jsonl").resolve()), "rows": holdout}, {"sha256": sha_bytes(policy_raw), "path": str((split_root / "policy.json").resolve()), "value": policy_value})


def _identity(actual: dict[str, Any], *, receipt_path: Path, receipt_sha: str, split: dict[str, Any], calibration: dict[str, Any], holdout: dict[str, Any], policy_ref: dict[str, Any], source_v32: Path, receipt_state: str) -> dict[str, Any]:
    return {"sq8_receipt_path": str(receipt_path.resolve()), "sq8_receipt_sha256": receipt_sha, "receipt_state": receipt_state, "request_id": actual["request_id"], "source_commit": actual["source_commit"], "source_tree_sha256": actual["source_tree_sha256"], "source_archive_sha256": actual["source_archive_sha256"], "source_v32_path": str(source_v32.resolve()), "source_v32_sha256": sha_file(source_v32, "source-v32"), "served_model": actual["served_model"], "worker": actual["worker"], "package": actual["package"], "overlay_content_sha256": actual["content_sha256"], "overlay_tensor_set_sha256": actual["tensor_set_sha256"], "token_ids_sha256": actual["token_ids_sha256"], "telemetry_binding": actual["telemetry_binding"], "maintenance_evidence": actual["maintenance_evidence"], "executor_record": actual["executor_record"], "prepared_receipt": actual["prepared_receipt"], "split_manifest_sha256": split["sha256"], "calibration_cases_path": calibration["path"], "calibration_cases_sha256": calibration["sha256"], "holdout_cases_path": holdout["path"], "holdout_cases_sha256": holdout["sha256"], "policy_path": policy_ref["path"], "policy_sha256": policy_ref["sha256"]}


def create_plan(split_root: Path, actual_receipt: Path, source_v32: Path, output: Path) -> None:
    try:
        receipt, receipt_raw, actual = _actual_receipt(actual_receipt)
        actual_verified = True
    except ProtocolError:
        # A prepared receipt can produce a read-only preflight plan.  It must
        # never be accepted by ``freeze`` or either holdout command.
        receipt, receipt_raw = load_json(actual_receipt, "SQ8 prepared receipt")
        if receipt.get("schema_version") != SQ8_RECEIPT_SCHEMA or receipt.get("status") != "prepared_not_executed" or receipt.get("actual") != {"status": "pending", "required": True}:
            raise
        _request(receipt.get("request_id"))
        source = receipt.get("source_provenance")
        overlay = receipt.get("overlay")
        release = receipt.get("release")
        package = receipt.get("package")
        if not isinstance(source, dict) or not isinstance(overlay, dict) or not isinstance(release, dict) or not isinstance(package, dict):
            raise ProtocolError("prepared SQ8 receipt identity is incomplete")
        _exact_int(overlay.get("tensor_count"), 48, "SQ8 overlay tensor count")
        _validate_artifact_inventory(overlay.get("artifact_inventory"))
        _hex(receipt.get("source_commit"), "source commit", forty=True)
        _hex(source.get("tree_sha256"), "source tree", forty=True)
        _hex(source.get("archive_sha256"), "source archive")
        _hex(overlay.get("content_sha256"), "overlay content")
        _hex(overlay.get("tensor_set_sha256"), "overlay tensor set")
        _validate_release_and_profile(receipt, actual_receipt, actual_receipt)
        if overlay["artifact_inventory"]["root"] != str(Path(overlay["binding_manifest_path"]).resolve().parent):
            raise ProtocolError("SQ8 artifact inventory root differs from binding root")
        if "authorization_lineage" in receipt:
            _validate_lineage_reference(receipt["authorization_lineage"], receipt["request_id"])
        actual = {"request_id": receipt["request_id"], "source_commit": receipt["source_commit"], "source_tree_sha256": source["tree_sha256"], "source_archive_sha256": source["archive_sha256"], "content_sha256": overlay["content_sha256"], "tensor_set_sha256": overlay["tensor_set_sha256"], "served_model": release.get("served_model"), "worker": release.get("worker"), "package": package, "token_ids_sha256": None, "telemetry_binding": None, "maintenance_evidence": None, "executor_record": None, "prepared_receipt": {"path": str(actual_receipt.resolve()), "sha256": sha_bytes(receipt_raw)}}
        actual_verified = False
    split, calibration, holdout, policy_ref = _split(split_root)
    split["rows"] = calibration["rows"] + holdout["rows"]
    identity = _identity(actual, receipt_path=actual_receipt, receipt_sha=sha_bytes(receipt_raw), split=split, calibration=calibration, holdout=holdout, policy_ref=policy_ref, source_v32=source_v32, receipt_state="actual_verified" if actual_verified else "prepared_not_executed")
    status = "ready_for_calibration" if actual_verified else "preflight_only"
    observed_headroom = int(actual.get("vram_headroom_bytes", 1) or 0) if actual_verified else 1
    if actual_verified and observed_headroom < 1:
        raise ProtocolError("SQ8 actual evidence lacks positive VRAM headroom")
    plan = {"schema_version": PLAN_SCHEMA, "status": status, "preflight_only": not actual_verified, "actual_verified_required": True, "identity": identity, "policy": policy(), "calibration": {"path": calibration["path"], "sha256": calibration["sha256"], "row_count": MAX_ROWS}, "holdout": {"path": holdout["path"], "sha256": holdout["sha256"], "row_count": MAX_ROWS}, "resource_contract": {"jobs": 1, "case_concurrency": 1, "one_model_load": True, "chunk_elements": MAX_CHUNK_ELEMENTS, "bounded_vectors": True, "bounded_disk": True, "max_rows": MAX_ROWS, "max_case_file_bytes": MAX_JSON_BYTES, "vram_headroom_required": True, "vram_headroom_bytes_min": 1, "vram_observed_headroom_bytes": observed_headroom}, "holdout_state": {"status": "not_started", "evaluations_remaining": 1, "retry_permitted": False}}
    atomic_json(output, plan)


def _check_plan(path: Path) -> tuple[dict[str, Any], bytes]:
    plan, raw = load_json(path, "SQ8 fidelity plan")
    if set(plan) != {"schema_version", "status", "preflight_only", "actual_verified_required", "identity", "policy", "calibration", "holdout", "resource_contract", "holdout_state"}:
        raise ProtocolError("SQ8 plan has unknown or missing fields")
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("actual_verified_required") is not True:
        raise ProtocolError("SQ8 plan schema/state differs")
    holdout_state = plan.get("holdout_state")
    if not isinstance(holdout_state, dict) or set(holdout_state) != {"status", "evaluations_remaining", "retry_permitted"} or holdout_state.get("status") != "not_started" or holdout_state.get("retry_permitted") is not False:
        raise ProtocolError("SQ8 plan schema/state differs")
    _exact_int(holdout_state.get("evaluations_remaining"), 1, "SQ8 plan evaluations remaining")
    status = plan.get("status")
    preflight_only = plan.get("preflight_only")
    identity = plan.get("identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("receipt_state"), str):
        raise ProtocolError("SQ8 plan receipt state is missing")
    if identity.get("receipt_state") == "actual_verified":
        if status != "ready_for_calibration" or preflight_only is not False:
            raise ProtocolError("SQ8 actual plan state is not bound")
    elif identity.get("receipt_state") == "prepared_not_executed":
        if status != "preflight_only" or preflight_only is not True:
            raise ProtocolError("SQ8 prepared-only plan cannot be promoted")
    else:
        raise ProtocolError("SQ8 plan receipt state differs")
    resource = plan.get("resource_contract", {})
    if set(resource) != {"jobs", "case_concurrency", "one_model_load", "chunk_elements", "bounded_vectors", "bounded_disk", "max_rows", "max_case_file_bytes", "vram_headroom_required", "vram_headroom_bytes_min", "vram_observed_headroom_bytes"}:
        raise ProtocolError("SQ8 resource contract has unknown or missing fields")
    _exact_int(resource.get("jobs"), 1, "SQ8 resource jobs")
    _exact_int(resource.get("case_concurrency"), 1, "SQ8 resource case concurrency")
    _exact_int(resource.get("chunk_elements"), MAX_CHUNK_ELEMENTS, "SQ8 resource chunk elements")
    _exact_int(resource.get("max_rows"), MAX_ROWS, "SQ8 resource max rows")
    _exact_int(resource.get("max_case_file_bytes"), MAX_JSON_BYTES, "SQ8 resource max case file bytes")
    _int(resource.get("vram_headroom_bytes_min"), "SQ8 resource minimum VRAM headroom", minimum=1)
    observed_headroom = _int(resource.get("vram_observed_headroom_bytes"), "SQ8 observed VRAM headroom", minimum=1)
    if resource.get("one_model_load") is not True or resource.get("bounded_vectors") is not True or resource.get("bounded_disk") is not True or resource.get("vram_headroom_required") is not True or observed_headroom < resource["vram_headroom_bytes_min"]:
        raise ProtocolError("SQ8 resource contract is unsafe")
    source_v32 = Path(str(plan.get("identity", {}).get("source_v32_path", "")))
    if not source_v32.is_absolute() or source_v32 != source_v32.resolve() or str(source_v32) != identity.get("source_v32_path") or sha_file(source_v32, "source-v32") != plan["identity"].get("source_v32_sha256"):
        raise ProtocolError("SQ8 source-v32 identity differs")
    receipt_path = Path(str(plan.get("identity", {}).get("sq8_receipt_path", "")))
    if not receipt_path.is_absolute() or receipt_path != receipt_path.resolve() or str(receipt_path) != identity.get("sq8_receipt_path") or sha_file(receipt_path, "SQ8 promotion receipt") != plan["identity"].get("sq8_receipt_sha256"):
        raise ProtocolError("SQ8 promotion receipt identity differs")
    try:
        _actual_receipt(receipt_path)
    except ProtocolError:
        prepared_value, _ = load_json(receipt_path, "SQ8 prepared receipt")
        if prepared_value.get("schema_version") != SQ8_RECEIPT_SCHEMA or prepared_value.get("status") != "prepared_not_executed":
            raise
    if not _strict_equal(plan.get("policy"), policy()):
        raise ProtocolError("SQ8 plan policy differs")
    # Re-read every mutable input on every phase.  A plan is a binding, not a
    # capability: changing the receipt, source-v32, policy, or either case set
    # after plan creation invalidates it.
    if source_v32 != source_v32.resolve() or str(source_v32) != identity.get("source_v32_path") or sha_file(source_v32, "source-v32") != identity.get("source_v32_sha256"):
        raise ProtocolError("SQ8 source-v32 identity differs")
    current_receipt_sha = sha_file(receipt_path, "SQ8 promotion receipt")
    if current_receipt_sha != identity.get("sq8_receipt_sha256"):
        raise ProtocolError("SQ8 promotion receipt changed after plan creation")
    if identity.get("receipt_state") == "actual_verified":
        _, _, current = _actual_receipt(receipt_path)
        if current.get("request_id") != identity.get("request_id") or not _strict_equal(current.get("served_model"), identity.get("served_model")) or not _strict_equal(current.get("worker"), identity.get("worker")) or not _strict_equal(current.get("package"), identity.get("package")):
            raise ProtocolError("SQ8 actual receipt identity changed")
    else:
        current_receipt, _ = load_json(receipt_path, "SQ8 prepared receipt")
        if current_receipt.get("schema_version") != SQ8_RECEIPT_SCHEMA or current_receipt.get("status") != "prepared_not_executed" or current_receipt.get("actual") != {"status": "pending", "required": True}:
            raise ProtocolError("SQ8 prepared-only receipt was promoted")
    split_root = Path(str(identity.get("calibration_cases_path", ""))).parent
    split, calibration, holdout, policy_ref = _split(split_root)
    if split.get("sha256") != identity.get("split_manifest_sha256") or calibration.get("sha256") != identity.get("calibration_cases_sha256") or holdout.get("sha256") != identity.get("holdout_cases_sha256") or policy_ref.get("sha256") != identity.get("policy_sha256"):
        raise ProtocolError("SQ8 case or policy binding changed after plan creation")
    if not _strict_equal(plan.get("calibration"), {"path": calibration["path"], "sha256": calibration["sha256"], "row_count": MAX_ROWS}) or not _strict_equal(plan.get("holdout"), {"path": holdout["path"], "sha256": holdout["sha256"], "row_count": MAX_ROWS}):
        raise ProtocolError("SQ8 plan case bindings differ")
    return plan, raw


def _metric_value(value: Any, label: str, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ProtocolError(f"{label} is not finite")
    number = float(value)
    if number < 0 or (name != "hidden_max_abs" and number > 1):
        raise ProtocolError(f"{label} is outside the frozen domain")
    if name in BINARY_METRICS and number not in (0.0, 1.0):
        raise ProtocolError(f"{label} must be binary")
    if name in RELATIVE_L2_METRICS and number > 1:
        raise ProtocolError(f"{label} exceeds pathological relative-L2 ceiling")
    return number


def _rows(metrics: dict[str, Any], expected_rows: list[dict[str, Any]], identity: dict[str, Any]) -> list[dict[str, Any]]:
    if set(metrics) != {"schema_version", "identity", "subset", "rows"}:
        raise ProtocolError("SQ8 metrics has unknown or missing fields")
    if metrics.get("schema_version") != METRICS_SCHEMA or not _strict_equal(metrics.get("identity"), identity) or metrics.get("subset") != "calibration":
        raise ProtocolError("SQ8 metrics identity/subset differs")
    rows = metrics.get("rows")
    if not isinstance(rows, list) or len(rows) != MAX_ROWS:
        raise ProtocolError("SQ8 calibration metrics must contain exactly 24 rows")
    expected = {row.get("case_id"): row for row in expected_rows}
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("case_id") not in expected or row["case_id"] in seen:
            raise ProtocolError("SQ8 metric case identity differs")
        allowed_row_keys = {"case_id", "case_sha256", "fixture_sha256", "prompt_token_ids_sha256", "context_token_ids_sha256", "prompt_tokens", "cached_prefix_tokens", "context_tokens", "generated_tokens", "baseline_mode", "prefill_requested_m", "resolved_m", "step", "row_count", "subset", "metrics"}
        if set(row) != allowed_row_keys:
            raise ProtocolError("SQ8 metric row has unknown or missing fields")
        seen.add(row["case_id"])
        expected_row = expected[row["case_id"]]
        integer_keys = ("prompt_tokens", "cached_prefix_tokens", "context_tokens", "generated_tokens", "prefill_requested_m", "resolved_m", "step", "row_count")
        for key in integer_keys:
            _int(row.get(key), f"SQ8 metric row {row['case_id']} {key}")
        for key in ("case_sha256", "fixture_sha256", "prompt_token_ids_sha256", "context_token_ids_sha256", "baseline_mode", "subset"):
            if row.get(key) != expected_row.get(key):
                raise ProtocolError(f"SQ8 metric row identity differs: {row['case_id']} {key}")
        for key in integer_keys:
            if row.get(key) != expected_row.get(key):
                raise ProtocolError(f"SQ8 metric row identity differs: {row['case_id']} {key}")
        values = row.get("metrics")
        if not isinstance(values, dict) or set(values) != set(METRIC_POLICY):
            raise ProtocolError(f"SQ8 metric set differs: {row['case_id']}")
        for name in METRIC_POLICY:
            _metric_value(values.get(name), f"{row['case_id']}.{name}", name)
    if seen != set(expected):
        raise ProtocolError("SQ8 calibration metric row set differs")
    return rows


def wilson_lower(successes: int, samples: int = MAX_ROWS) -> float:
    if samples != MAX_ROWS or not 0 <= successes <= samples:
        raise ProtocolError("Wilson inputs differ")
    p = successes / samples
    z2 = WILSON_Z * WILSON_Z
    denominator = 1 + z2 / samples
    center = p + z2 / (2 * samples)
    radius = WILSON_Z * math.sqrt((p * (1 - p) + z2 / (4 * samples)) / samples)
    return max(0.0, (center - radius) / denominator)


def recompute(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {name: [float(row["metrics"][name]) for row in rows] for name in METRIC_POLICY}
    derived: dict[str, Any] = {}
    for name, spec in METRIC_POLICY.items():
        values = aggregate[name]
        if name in BINARY_METRICS:
            successes = sum(value == 1.0 for value in values)
            derived[name] = {"calibration_mean": sum(values) / MAX_ROWS, "successes": successes, "confidence_level": 0.95, "wilson_z": WILSON_Z, "bound": wilson_lower(successes), "direction": spec["direction"], "sample_count": MAX_ROWS}
        elif name == "hidden_max_abs":
            derived[name] = {"diagnostic_max": max(values), "bound": None, "direction": "diagnostic", "sample_count": MAX_ROWS}
        else:
            mean = sum(values) / MAX_ROWS
            margin = max(float(spec["margin"]), float(spec["relative_margin"]) * abs(mean))
            bound = mean - margin if spec["direction"] == "higher" else mean + margin
            if spec["absolute_floor"] is not None:
                bound = max(float(spec["absolute_floor"]), bound)
            if spec["absolute_ceiling"] is not None:
                bound = min(float(spec["absolute_ceiling"]), bound)
            derived[name] = {"calibration_mean": mean, "absolute_margin": spec["margin"], "relative_margin": spec["relative_margin"], "effective_margin": margin, "bound": bound, "direction": spec["direction"], "sample_count": MAX_ROWS}
    return derived


FROZEN_ATTEMPT_BOUNDARY = {
    "remaining_before": 1,
    "remaining_after": 0,
    "failure_consumes_attempt": True,
}


def _validate_attempt_boundary(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != set(FROZEN_ATTEMPT_BOUNDARY):
        raise ProtocolError(f"{label} shape differs")
    _exact_int(value.get("remaining_before"), 1, f"{label}.remaining_before")
    _exact_int(value.get("remaining_after"), 0, f"{label}.remaining_after")
    if value.get("failure_consumes_attempt") is not True:
        raise ProtocolError(f"{label}.failure_consumes_attempt differs")


def _validate_freeze_contract(
    receipt: dict[str, Any],
    *,
    plan_path: Path,
    plan: dict[str, Any],
    metrics_path: Path | None = None,
    metrics_sha256: str | None = None,
) -> None:
    expected_keys = {
        "schema_version", "status", "identity", "plan_path", "plan_sha256", "metrics_path",
        "metrics_sha256", "calibration_case_count", "derived_bounds", "holdout_status",
        "holdout_evaluations_remaining", "retry_permitted", "relative_l2_rejection_ceiling",
        "attempt_boundary",
    }
    if set(receipt) != expected_keys:
        raise ProtocolError("SQ8 freeze receipt has unknown or missing fields")
    if receipt.get("schema_version") != FREEZE_SCHEMA or receipt.get("status") != "frozen_calibration_envelope" or not _strict_equal(receipt.get("identity"), plan["identity"]) or receipt.get("plan_path") != str(plan_path.resolve()) or receipt.get("plan_sha256") != sha_file(plan_path, "SQ8 plan"):
        raise ProtocolError("SQ8 freeze receipt binding differs")
    _exact_int(receipt.get("calibration_case_count"), MAX_ROWS, "SQ8 freeze calibration case count")
    _float(receipt.get("relative_l2_rejection_ceiling"), "SQ8 freeze relative-L2 ceiling", minimum=1.0, maximum=1.0)
    _exact_int(receipt.get("holdout_evaluations_remaining"), 1, "SQ8 freeze evaluations remaining")
    if receipt.get("holdout_status") != "not_started" or receipt.get("retry_permitted") is not False:
        raise ProtocolError("SQ8 freeze receipt state differs")
    _validate_attempt_boundary(receipt.get("attempt_boundary"), "SQ8 freeze attempt boundary")
    if metrics_path is not None and metrics_sha256 is not None:
        if receipt.get("metrics_path") != str(metrics_path.resolve()) or receipt.get("metrics_sha256") != metrics_sha256:
            raise ProtocolError("SQ8 freeze metrics binding differs")


def _validate_preflight_contract(value: dict[str, Any], *, plan: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version", "status", "freeze_receipt_sha256", "freeze_receipt_path", "plan_path",
        "plan_sha256", "identity", "holdout_cases_sha256", "holdout_case_count",
        "evaluations_remaining", "retry_permitted", "attempt_boundary",
    }
    if set(value) != expected_keys:
        raise ProtocolError("SQ8 holdout preflight has unknown or missing fields")
    if value.get("schema_version") != PREFLIGHT_SCHEMA or value.get("status") != "ready_for_one_shot_holdout" or not _strict_equal(value.get("identity"), plan["identity"]):
        raise ProtocolError("SQ8 holdout preflight identity differs")
    _exact_int(value.get("holdout_case_count"), MAX_ROWS, "SQ8 preflight holdout case count")
    _exact_int(value.get("evaluations_remaining"), 1, "SQ8 preflight evaluations remaining")
    if value.get("retry_permitted") is not False:
        raise ProtocolError("SQ8 preflight retry policy differs")
    _validate_attempt_boundary(value.get("attempt_boundary"), "SQ8 preflight attempt boundary")


def freeze(plan_path: Path, metrics_path: Path, output: Path) -> None:
    plan, _ = _check_plan(plan_path)
    if plan.get("status") != "ready_for_calibration" or plan.get("preflight_only") is True:
        raise ProtocolError("SQ8 actual_verified receipt is required before calibration freeze")
    metrics, metrics_raw = load_json(metrics_path, "SQ8 calibration metrics")
    rows = _rows(metrics, read_jsonl(Path(plan["calibration"]["path"]), "SQ8 calibration cases"), plan["identity"])
    receipt = {"schema_version": FREEZE_SCHEMA, "status": "frozen_calibration_envelope", "identity": plan["identity"], "plan_path": str(plan_path.resolve()), "plan_sha256": sha_file(plan_path, "SQ8 plan"), "metrics_path": str(metrics_path.resolve()), "metrics_sha256": sha_bytes(metrics_raw), "calibration_case_count": MAX_ROWS, "derived_bounds": recompute(rows), "holdout_status": "not_started", "holdout_evaluations_remaining": 1, "retry_permitted": False, "relative_l2_rejection_ceiling": 1.0, "attempt_boundary": {"remaining_before": 1, "remaining_after": 0, "failure_consumes_attempt": True}}
    atomic_json(output, receipt)


def validate_freeze(plan_path: Path, metrics_path: Path, freeze_path: Path) -> dict[str, Any]:
    """Independently recompute all 24 rows and compare a freeze receipt."""

    plan, _ = _check_plan(plan_path)
    if plan.get("status") != "ready_for_calibration" or plan.get("preflight_only") is True:
        raise ProtocolError("SQ8 actual_verified receipt is required before freeze validation")
    metrics, metrics_raw = load_json(metrics_path, "SQ8 calibration metrics")
    rows = _rows(metrics, read_jsonl(Path(plan["calibration"]["path"]), "SQ8 calibration cases"), plan["identity"])
    receipt, receipt_raw = load_json(freeze_path, "SQ8 freeze receipt")
    _validate_freeze_contract(
        receipt,
        plan_path=plan_path,
        plan=plan,
        metrics_path=metrics_path,
        metrics_sha256=sha_bytes(metrics_raw),
    )
    expected = recompute(rows)
    if not _strict_equal(receipt.get("derived_bounds"), expected):
        raise ProtocolError("SQ8 freeze receipt derived bounds differ from independent recomputation")
    return {"status": "ok", "receipt_sha256": sha_bytes(receipt_raw), "metrics_sha256": sha_bytes(metrics_raw), "row_count": MAX_ROWS}


def _validated_freeze(plan_path: Path, plan: dict[str, Any], freeze_path: Path) -> tuple[dict[str, Any], bytes]:
    freeze, freeze_raw = load_json(freeze_path, "SQ8 freeze receipt")
    _validate_freeze_contract(freeze, plan_path=plan_path, plan=plan)
    metrics_path = Path(str(freeze.get("metrics_path", "")))
    if not metrics_path.is_absolute() or metrics_path != metrics_path.resolve() or str(metrics_path) != freeze.get("metrics_path") or sha_file(metrics_path, "SQ8 calibration metrics") != freeze.get("metrics_sha256"):
        raise ProtocolError("SQ8 freeze metrics binding differs")
    metrics, metrics_raw = load_json(metrics_path, "SQ8 calibration metrics")
    rows = _rows(metrics, read_jsonl(Path(plan["calibration"]["path"]), "SQ8 calibration cases"), plan["identity"])
    if sha_bytes(metrics_raw) != freeze.get("metrics_sha256") or not _strict_equal(freeze.get("derived_bounds"), recompute(rows)):
        raise ProtocolError("SQ8 freeze derived bounds are stale")
    return freeze, freeze_raw


def preflight(freeze_path: Path, plan_path: Path, output: Path) -> None:
    plan, _ = _check_plan(plan_path)
    if plan.get("status") != "ready_for_calibration" or plan.get("preflight_only") is True:
        raise ProtocolError("SQ8 actual_verified receipt is required before holdout preflight")
    freeze_value, freeze_raw = _validated_freeze(plan_path, plan, freeze_path)
    holdout = read_jsonl(Path(plan["holdout"]["path"]), "SQ8 holdout cases")
    if len(holdout) != MAX_ROWS:
        raise ProtocolError("SQ8 holdout case count differs")
    value = {"schema_version": PREFLIGHT_SCHEMA, "status": "ready_for_one_shot_holdout", "freeze_receipt_sha256": sha_bytes(freeze_raw), "freeze_receipt_path": str(freeze_path.resolve()), "plan_path": str(plan_path.resolve()), "plan_sha256": sha_file(plan_path, "SQ8 plan"), "identity": plan["identity"], "holdout_cases_sha256": plan["identity"]["holdout_cases_sha256"], "holdout_case_count": MAX_ROWS, "evaluations_remaining": 1, "retry_permitted": False, "attempt_boundary": {"remaining_before": 1, "remaining_after": 0, "failure_consumes_attempt": True}}
    atomic_json(output, value)


def execute(preflight_path: Path, metrics_path: Path, ledger_path: Path, output: Path, *, crash_after_sentinel: bool = False) -> None:
    preflight_value, preflight_raw = load_json(preflight_path, "SQ8 holdout preflight")
    plan_path = Path(str(preflight_value.get("plan_path", "")))
    if not plan_path.is_absolute() or plan_path != plan_path.resolve() or sha_file(plan_path, "SQ8 fidelity plan") != preflight_value.get("plan_sha256"):
        raise ProtocolError("SQ8 holdout preflight plan binding differs")
    plan, _ = _check_plan(plan_path)
    _validate_preflight_contract(preflight_value, plan=plan)
    if preflight_value.get("plan_path") != str(plan_path.resolve()):
        raise ProtocolError("SQ8 holdout preflight plan path differs")
    if not _strict_equal(plan.get("identity"), preflight_value.get("identity")) or preflight_value.get("holdout_cases_sha256") != plan["identity"].get("holdout_cases_sha256"):
        raise ProtocolError("SQ8 holdout preflight identity differs")
    freeze_path = Path(str(preflight_value.get("freeze_receipt_path", "")))
    if not freeze_path.is_absolute() or freeze_path != freeze_path.resolve():
        raise ProtocolError("SQ8 holdout preflight freeze path differs")
    freeze_value, freeze_raw = _validated_freeze(plan_path, plan, freeze_path)
    if sha_bytes(freeze_raw) != preflight_value.get("freeze_receipt_sha256"):
        raise ProtocolError("SQ8 holdout preflight freeze binding differs")
    if os.path.lexists(ledger_path):
        raise ProtocolError("SQ8 holdout attempt was already consumed; retry is forbidden")
    if os.path.lexists(output):
        raise ProtocolError("SQ8 holdout result must be create-new")
    attempt_id = sha_bytes(b"ullm.qwen35-aq4-sq8-holdout-attempt-v1\0" + sha_bytes(preflight_raw).encode())
    ledger = {"schema_version": LEDGER_SCHEMA, "status": "consumed", "attempt_id": attempt_id, "preflight_sha256": sha_bytes(preflight_raw), "identity": preflight_value["identity"], "remaining_before": 1, "remaining_after": 0, "retry_permitted": False}
    atomic_json(ledger_path, ledger)
    if crash_after_sentinel:
        raise ProtocolError("simulated crash after irreversible attempt boundary")
    try:
        metrics, metrics_raw = load_json(metrics_path, "SQ8 holdout metrics")
        if metrics.get("schema_version") != METRICS_SCHEMA or not _strict_equal(metrics.get("identity"), preflight_value["identity"]) or metrics.get("subset") != "holdout":
            raise ProtocolError("SQ8 holdout metrics identity/subset differs")
        # Holdout rows are checked against the frozen 24-row case set; identity is
        # retained in the preflight and the same strict row validator is reused.
        expected_rows = read_jsonl(Path(preflight_value["identity"].get("holdout_cases_path", "")), "SQ8 holdout cases") if preflight_value["identity"].get("holdout_cases_path") else []
        if not expected_rows:
            raise ProtocolError("SQ8 holdout case path is absent from identity")
        rows = _rows({**metrics, "subset": "calibration"}, expected_rows, preflight_value["identity"])
        bounds_path = Path(str(preflight_value["freeze_receipt_path"]))
        freeze_value, _ = _validated_freeze(plan_path, plan, bounds_path)
        observed = recompute(rows)
        checks: dict[str, bool] = {}
        for name, spec in METRIC_POLICY.items():
            if spec["role"] == "diagnostic_only":
                checks[name] = True
            elif spec["direction"] == "higher":
                checks[name] = observed[name]["bound"] >= freeze_value["derived_bounds"][name]["bound"]
            else:
                checks[name] = observed[name]["bound"] <= freeze_value["derived_bounds"][name]["bound"]
        passed = all(checks.values())
        result = {"schema_version": HOLDOUT_SCHEMA, "attempt_schema": ATTEMPT_SCHEMA, "status": "passed" if passed else "failed", "attempt_id": attempt_id, "preflight_sha256": sha_bytes(preflight_raw), "ledger_sha256": sha_file(ledger_path, "SQ8 attempt ledger"), "metrics_sha256": sha_bytes(metrics_raw), "identity": preflight_value["identity"], "derived_metrics": observed, "gate_checks": checks, "retry_permitted": False, "evaluations_remaining": 0}
    except (ProtocolError, OSError, ValueError) as error:
        result = {"schema_version": HOLDOUT_SCHEMA, "attempt_schema": ATTEMPT_SCHEMA, "status": "failed", "attempt_id": attempt_id, "preflight_sha256": sha_bytes(preflight_raw), "ledger_sha256": sha_file(ledger_path, "SQ8 attempt ledger"), "identity": preflight_value["identity"], "failure": str(error), "retry_permitted": False, "evaluations_remaining": 0}
        atomic_json(output, result)
        raise
    atomic_json(output, result)
    if result["status"] != "passed":
        raise ProtocolError("SQ8 holdout gate failed; the consumed attempt cannot be retried")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan"); p.add_argument("--split-root", type=Path, required=True); p.add_argument("--actual-receipt", type=Path, required=True); p.add_argument("--source-v32", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=lambda a: create_plan(a.split_root, a.actual_receipt, a.source_v32, a.output))
    p = sub.add_parser("freeze"); p.add_argument("--plan", type=Path, required=True); p.add_argument("--metrics", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=lambda a: freeze(a.plan, a.metrics, a.output))
    p = sub.add_parser("validate-freeze"); p.add_argument("--plan", type=Path, required=True); p.add_argument("--metrics", type=Path, required=True); p.add_argument("--freeze", type=Path, required=True); p.set_defaults(func=lambda a: print(json.dumps(validate_freeze(a.plan, a.metrics, a.freeze), sort_keys=True)))
    p = sub.add_parser("preflight-holdout"); p.add_argument("--plan", type=Path, required=True); p.add_argument("--freeze", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=lambda a: preflight(a.freeze, a.plan, a.output))
    p = sub.add_parser("execute-holdout"); p.add_argument("--preflight", type=Path, required=True); p.add_argument("--metrics", type=Path, required=True); p.add_argument("--ledger", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--crash-after-sentinel", action="store_true"); p.set_defaults(func=lambda a: execute(a.preflight, a.metrics, a.ledger, a.output, crash_after_sentinel=a.crash_after_sentinel))
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except (ProtocolError, OSError, ValueError) as error:
        print(f"SQ8 fidelity protocol failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
