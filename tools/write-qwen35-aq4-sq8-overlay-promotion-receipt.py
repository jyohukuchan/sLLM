#!/usr/bin/env python3
"""Publish an immutable Qwen3.5 AQ4/SQ8 overlay promotion receipt.

The writer performs only bounded filesystem and JSON validation.  It never
starts GPU tools or services.  The receipt is published with a same-directory
temporary file followed by an exclusive hard-link, so an existing destination
cannot be replaced and readers never observe a partial JSON document.
"""

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
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

try:
    import qwen35_aq4_sq8_authorization_lineage as lineage_tool
except ModuleNotFoundError:
    from tools import qwen35_aq4_sq8_authorization_lineage as lineage_tool


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_SCHEMA = "ullm.qwen35_aq4_sq8_overlay_promotion.v1"
IMPLEMENTATION_ID = "qwen35_aq4_sq8_linear_qkv_z_overlay_v1"
REQUEST_ID_RE = re.compile(r"^sq8-promotion-[0-9a-f]{64}$")
HEX40 = set("0123456789abcdef")
HEX64 = set("0123456789abcdef")
MAX_JSON_BYTES = 32 * 1024 * 1024


class ReceiptError(RuntimeError):
    """Raised when an overlay receipt cannot be safely published."""


def _request_id(value: Any) -> str:
    if not isinstance(value, str) or REQUEST_ID_RE.fullmatch(value) is None:
        raise ReceiptError("SQ8 promotion request_id must be sq8-promotion-<64 lowercase hex>")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReceiptError(f"{label} must be a regular non-symlink file")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ReceiptError(f"{label} exceeds the JSON size bound")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReceiptError(f"failed to read {label}") from error
    if not isinstance(value, dict):
        raise ReceiptError(f"{label} must be a JSON object")
    return value


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ReceiptError(f"cannot hash non-regular file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise ReceiptError(f"cannot hash {path}") from error
    return digest.hexdigest()


def _hex(value: Any, length: int, label: str) -> str:
    alphabet = HEX40 if length == 40 else HEX64
    if not isinstance(value, str) or len(value) != length or any(c not in alphabet for c in value):
        raise ReceiptError(f"{label} must be lowercase hexadecimal")
    return value


def _mode(metadata: os.stat_result) -> str:
    return f"{stat.S_IMODE(metadata.st_mode):04o}"


def _entry(relative_path: str, metadata: os.stat_result) -> dict[str, Any]:
    if stat.S_ISDIR(metadata.st_mode):
        kind = "directory"
        size = 0
    elif stat.S_ISREG(metadata.st_mode):
        kind = "regular"
        size = metadata.st_size
    elif stat.S_ISLNK(metadata.st_mode):
        kind = "symlink"
        size = metadata.st_size
    else:
        kind = "special"
        size = metadata.st_size
    return {
        "path": relative_path,
        "kind": kind,
        "mode": _mode(metadata),
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "nlink": metadata.st_nlink,
        "bytes": size,
    }


def artifact_inventory(root: Path) -> dict[str, Any]:
    """Return the complete immutable metadata inventory for an overlay root."""

    root = root.resolve()
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise ReceiptError(f"overlay artifact root is unavailable: {root}") from error
    if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
        raise ReceiptError("overlay artifact root must be a regular directory")

    entries: list[dict[str, Any]] = []
    pending: list[tuple[Path, str]] = [(root, ".")]
    while pending:
        directory, prefix = pending.pop()
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise ReceiptError(f"overlay artifact enumeration failed: {directory}") from error
        for child in children:
            relative = child.name if prefix == "." else f"{prefix}/{child.name}"
            try:
                metadata = child.stat(follow_symlinks=False)
            except OSError as error:
                raise ReceiptError(f"overlay artifact metadata unavailable: {relative}") from error
            entries.append(_entry(relative, metadata))
            if stat.S_ISDIR(metadata.st_mode):
                pending.append((Path(child.path), relative))
    entries.insert(0, _entry(".", root_metadata))
    entries.sort(key=lambda item: item["path"])

    directories = [item for item in entries if item["kind"] == "directory"]
    regular_files = [item for item in entries if item["kind"] == "regular"]
    symlinks = [item for item in entries if item["kind"] == "symlink"]
    specials = [item for item in entries if item["kind"] == "special"]
    if not directories or not regular_files:
        raise ReceiptError("overlay artifact inventory is empty")
    if symlinks or specials:
        raise ReceiptError("overlay artifact inventory contains symlinks or special files")
    directory_modes = {item["mode"] for item in directories}
    file_modes = {item["mode"] for item in regular_files}
    identities = {(item["uid"], item["gid"]) for item in entries}
    if directory_modes != {"0555"} or file_modes != {"0444"}:
        raise ReceiptError("overlay artifact modes are not immutable 0555/0444")
    if len(identities) != 1:
        raise ReceiptError("overlay artifact uid/gid identity is not uniform")
    if any(item["nlink"] != 1 for item in regular_files):
        raise ReceiptError("overlay regular file nlink is not one")
    uid, gid = next(iter(identities))
    return {
        "root": os.fspath(root),
        "uid": uid,
        "gid": gid,
        "directory_count": len(directories),
        "directory_mode": "0555",
        "regular_file_count": len(regular_files),
        "regular_file_bytes": sum(item["bytes"] for item in regular_files),
        "regular_file_mode": "0444",
        "regular_file_nlink": 1,
        "symlink_count": 0,
        "special_count": 0,
        "entries": entries,
    }


def served_model_semantic_sha256(document: dict[str, Any]) -> str:
    """Hash a served manifest while omitting its self-referential receipt hash."""

    value = json.loads(json.dumps(document, ensure_ascii=True, allow_nan=False))
    promotion = value.get("promotion")
    if isinstance(promotion, dict):
        promotion.pop("receipt_sha256", None)
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load_generator() -> ModuleType:
    path = ROOT / "tools/generate-served-model.py"
    spec = importlib.util.spec_from_file_location("_ullm_sq8_receipt_generator", path)
    if spec is None or spec.loader is None:
        raise ReceiptError("served-model generator is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _profile_contract(profile: dict[str, Any], output_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if profile.get("format", {}).get("implementation_id") != IMPLEMENTATION_ID:
        raise ReceiptError("profile is not the SQ8 overlay implementation")
    promotion = profile.get("promotion")
    if not isinstance(promotion, dict):
        raise ReceiptError("overlay profile promotion is missing")
    expected_keys = {
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
    lineage_keys = {
        "authorization_lineage_from_receipt",
        "authorization_lineage",
    }
    has_lineage_contract = set(promotion) == expected_keys | lineage_keys
    if set(promotion) != expected_keys and not has_lineage_contract:
        raise ReceiptError("overlay profile receipt contract is incomplete")
    if Path(str(promotion["receipt"])).resolve() != output_path.resolve():
        raise ReceiptError("overlay profile receipt path differs from output")
    if promotion["required_schema_version"] != RECEIPT_SCHEMA:
        raise ReceiptError("overlay profile receipt schema differs")
    if promotion["source_commit_from_receipt"] != ["source_commit"]:
        raise ReceiptError("overlay profile source commit binding differs")
    if promotion["overlay_from_receipt"] != ["overlay"]:
        raise ReceiptError("overlay profile overlay binding differs")
    if promotion["release_from_receipt"] != ["release"]:
        raise ReceiptError("overlay profile release binding differs")
    if promotion["package_from_receipt"] != ["package"]:
        raise ReceiptError("overlay profile package binding differs")
    if promotion["actual_evidence_from_receipt"] != ["actual"]:
        raise ReceiptError("overlay profile actual evidence binding differs")
    if promotion["request_id_from_receipt"] != ["request_id"]:
        raise ReceiptError("overlay profile request ID binding differs")
    if promotion["authorization_audit_from_receipt"] != ["authorization_audit"]:
        raise ReceiptError("overlay profile authorization audit binding differs")
    if has_lineage_contract:
        if promotion.get("authorization_lineage_from_receipt") != [
            "authorization_lineage"
        ]:
            raise ReceiptError("overlay profile authorization lineage binding differs")
        if promotion.get("authorization_lineage") is not None:
            try:
                lineage_tool.validate_reference(promotion["authorization_lineage"])
            except lineage_tool.LineageError as error:
                raise ReceiptError(
                    f"overlay profile authorization lineage differs: {error}"
                ) from error
    if promotion["readiness_from_receipt"] != ["readiness"]:
        raise ReceiptError("overlay profile readiness binding differs")
    _readiness_identity(promotion["readiness"])
    _hex(promotion["release_source_commit"], 40, "overlay profile release source commit")
    worker = profile.get("worker")
    product = profile.get("product")
    if not isinstance(worker, dict) or not isinstance(product, dict):
        raise ReceiptError("overlay profile worker/product is incomplete")
    artifact = product.get("artifact")
    package = product.get("package")
    if not isinstance(artifact, dict) or not isinstance(package, dict):
        raise ReceiptError("overlay profile artifact/package is incomplete")
    return worker, {"product": product, "artifact": artifact, "package": package}


def _gpu_evidence_from_maintenance(value: dict[str, Any]) -> dict[str, Any]:
    """Derive the stable GPU-free observation from wrapper maintenance evidence."""

    if value.get("schema_version") != "ullm.qwen35_aq4.sq8_overlay_gpu_promotion_maintenance.v1":
        raise ReceiptError("maintenance evidence schema differs")
    if value.get("status") != "passed" or value.get("actual_run_count") != 1:
        raise ReceiptError("maintenance evidence is not a completed single run")
    if value.get("failure") is not None:
        raise ReceiptError("maintenance evidence contains a failure")
    restore = value.get("restore")
    if not isinstance(restore, dict) or restore.get("attempted") is not True or restore.get("passed") is not True:
        raise ReceiptError("maintenance restore evidence is incomplete")
    pre = value.get("candidate_pre")
    post = value.get("candidate_post")
    if not isinstance(pre, dict) or not isinstance(post, dict) or pre != post:
        raise ReceiptError("candidate identity changed during the actual run")
    observations = value.get("stopped_observations")
    if not isinstance(observations, list) or len(observations) < 2:
        raise ReceiptError("maintenance stable observation count is insufficient")
    stable = observations[-2:]
    for observation in stable:
        if not isinstance(observation, dict):
            raise ReceiptError("maintenance stopped observation shape differs")
        service = observation.get("service")
        owners = observation.get("owners")
        if not isinstance(service, dict) or not isinstance(owners, dict):
            raise ReceiptError("maintenance stopped observation is incomplete")
        if service.get("active") is not False or service.get("running") is not False:
            raise ReceiptError("maintenance stopped service was not inactive")
        if service.get("main_pid") != 0 or service.get("worker_pid") != 0 or service.get("lock_owned") is not False:
            raise ReceiptError("maintenance stopped service still owns a worker lock")
        for key in ("worker_pids", "amd_pids", "kfd_pids"):
            pids = owners.get(key)
            if not isinstance(pids, list) or pids:
                raise ReceiptError("maintenance stopped GPU owner set is nonempty")
    lock = value.get("lock")
    if not isinstance(lock, dict):
        raise ReceiptError("maintenance candidate lock evidence is missing")
    if lock.get("path") != "/run/ullm/device-1.lock" or lock.get("held") is not True or lock.get("released") is not True:
        raise ReceiptError("maintenance candidate lock evidence differs")
    return {
        "mode": "maintenance_stable2",
        "stable_observation_count": 2,
        "worker_pids": [],
        "amd_smi_owners": [],
        "kfd_owners": [],
        "lock": {"path": "/run/ullm/device-1.lock", "free": True},
    }


def _relative_evidence_ref(path: Path, output_path: Path, label: str) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ReceiptError(f"{label} must be a regular non-symlink file")
    path = path.resolve()
    try:
        relative = path.relative_to(output_path.parent.resolve())
    except ValueError as error:
        raise ReceiptError(f"{label} must be inside the receipt directory") from error
    if any(component in ("", ".", "..") for component in relative.parts):
        raise ReceiptError(f"{label} path is unsafe")
    return {"path": os.fspath(relative), "sha256": sha256_file(path)}


def _absolute_evidence_ref(path: Path, label: str) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ReceiptError(f"{label} must be a regular non-symlink file")
    return {"path": os.fspath(path.resolve()), "sha256": sha256_file(path)}


def _authorization_audit_ref(path: Path | None) -> dict[str, str] | None:
    """Bind the optional independent authorization audit by absolute path/SHA."""

    if path is None:
        return None
    if path.is_symlink() or not path.is_file():
        raise ReceiptError("authorization audit receipt must be a regular non-symlink file")
    resolved = path.resolve()
    if not resolved.is_absolute() or resolved == Path("/"):
        raise ReceiptError("authorization audit receipt path is invalid")
    return {"path": os.fspath(resolved), "sha256": sha256_file(resolved)}


def _readiness_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"schema", "container", "network", "endpoint"}:
        raise ReceiptError("readiness identity shape differs")
    if value.get("schema") != "ullm.bridge_container_readiness.v1":
        raise ReceiptError("readiness identity schema differs")
    container = value.get("container")
    network = value.get("network")
    endpoint = value.get("endpoint")
    if not isinstance(container, dict) or set(container) != {"name", "id", "image_id", "config_image"}:
        raise ReceiptError("readiness container identity differs")
    if (
        container.get("name") != "open-webui"
        or not isinstance(container.get("id"), str)
        or re.fullmatch(r"[0-9a-f]{64}", container["id"]) is None
        or not isinstance(container.get("image_id"), str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", container["image_id"]) is None
        or not isinstance(container.get("config_image"), str)
        or not container["config_image"]
    ):
        raise ReceiptError("readiness container identity differs")
    if not isinstance(network, dict) or set(network) != {"name", "id", "driver", "bridge_interface"}:
        raise ReceiptError("readiness network identity differs")
    network_id = network.get("id")
    if (
        not isinstance(network.get("name"), str)
        or not network["name"]
        or not isinstance(network_id, str)
        or re.fullmatch(r"[0-9a-f]{64}", network_id) is None
        or network.get("driver") != "bridge"
        or network.get("bridge_interface") != f"br-{network_id[:12]}"
    ):
        raise ReceiptError("readiness network identity differs")
    expected_body = '{"status":"ready"}'
    if not isinstance(endpoint, dict) or set(endpoint) != {
        "url", "path", "expected_status", "expected_body",
        "expected_body_sha256", "timeout_seconds",
    }:
        raise ReceiptError("readiness endpoint identity differs")
    if endpoint != {
        "url": "http://172.20.0.1:8000/readyz",
        "path": "/readyz",
        "expected_status": 200,
        "expected_body": expected_body,
        "expected_body_sha256": hashlib.sha256(expected_body.encode("ascii")).hexdigest(),
        "timeout_seconds": 5,
    }:
        raise ReceiptError("readiness endpoint identity differs")
    return json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))


def _validate_sq8_telemetry(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "projection", "diagnostic_host_staging"
    }:
        raise ReceiptError("SQ8 promotion telemetry shape differs")
    if value["schema_version"] != "ullm.qwen35_aq4.sq8_promotion_telemetry.v1":
        raise ReceiptError("SQ8 promotion telemetry schema differs")
    projection = value["projection"]
    expected_projection = {
        "single_matvec_count", "batch_matvec_count", "pair_matvec_count",
        "triple_matvec_count", "fallback_count",
    }
    if not isinstance(projection, dict) or set(projection) != expected_projection:
        raise ReceiptError("SQ8 promotion projection telemetry shape differs")
    if any(type(projection[key]) is not int or projection[key] < 0 for key in expected_projection):
        raise ReceiptError("SQ8 promotion projection telemetry count is invalid")
    if projection["batch_matvec_count"] <= 0 or projection["pair_matvec_count"] <= 0:
        raise ReceiptError("SQ8 promotion requires batch and pair telemetry")
    if any(projection[key] != 0 for key in ("single_matvec_count", "triple_matvec_count", "fallback_count")):
        raise ReceiptError("SQ8 promotion has an unexpected projection path")
    staging = value["diagnostic_host_staging"]
    staging_keys = {"read_count", "write_count", "read_bytes", "write_bytes"}
    if not isinstance(staging, dict) or set(staging) != staging_keys or any(staging[key] != 0 for key in staging_keys):
        raise ReceiptError("SQ8 promotion diagnostic host staging is nonzero")
    return value


def _actual_evidence(
    *,
    maintenance_path: Path | None,
    executor_path: Path | None,
    output_path: Path,
    profile: dict[str, Any],
    overlay: dict[str, Any],
    package_sha256: str,
    request_id: str,
    prepared_receipt_path: Path | None = None,
) -> dict[str, Any]:
    request_id = _request_id(request_id)
    paths = (maintenance_path, executor_path)
    if all(path is None for path in paths):
        return {"status": "pending", "required": True}
    if any(path is None for path in paths):
        raise ReceiptError("actual maintenance and executor evidence must be supplied together")
    assert maintenance_path is not None and executor_path is not None
    maintenance = _read_object(maintenance_path, "maintenance evidence")
    if maintenance.get("promotion_request_id") != request_id:
        raise ReceiptError("maintenance promotion request ID differs")
    gpu = _gpu_evidence_from_maintenance(maintenance)
    executor = _read_object(executor_path, "executor record")
    if executor.get("schema_version") != "ullm.production_executor_record.v1" or executor.get("status") != "ok":
        raise ReceiptError("executor record status/schema differs")
    promotion = executor.get("sq8_promotion_evidence")
    if not isinstance(promotion, dict) or set(promotion) != {
        "schema_version", "request_id", "manifest_identity", "telemetry", "output_identity"
    }:
        raise ReceiptError("executor SQ8 promotion evidence is missing")
    if promotion["schema_version"] != "ullm.qwen35_aq4.sq8_promotion_executor.v1":
        raise ReceiptError("executor SQ8 promotion evidence schema differs")
    if promotion.get("request_id") != request_id:
        raise ReceiptError("executor SQ8 promotion request ID differs")
    manifest_identity = promotion["manifest_identity"]
    expected_identity = {
        "implementation_id": IMPLEMENTATION_ID,
        "execution_profile": profile["worker"]["identity"]["execution_profile"],
        "artifact_content_sha256": overlay["content_sha256"],
        "artifact_manifest_sha256": overlay["binding_manifest_sha256"],
        "package_manifest_sha256": package_sha256,
    }
    if manifest_identity != expected_identity:
        raise ReceiptError("executor manifest identity differs")
    telemetry = _validate_sq8_telemetry(promotion["telemetry"])
    output_identity = promotion["output_identity"]
    if (
        not isinstance(output_identity, dict)
        or set(output_identity) != {"token_count", "token_ids_sha256", "token_ids_recorded"}
        or type(output_identity["token_count"]) is not int
        or output_identity["token_count"] < 1
        or _hex(output_identity.get("token_ids_sha256"), 64, "executor token IDs SHA-256")
        != output_identity.get("token_ids_sha256")
        or output_identity["token_ids_recorded"] is not False
    ):
        raise ReceiptError("executor output identity differs")
    actual = {
        "status": "actual_verified",
        "required": True,
        "maintenance_evidence": _relative_evidence_ref(maintenance_path, output_path, "maintenance evidence"),
        "executor_record": _relative_evidence_ref(executor_path, output_path, "executor record"),
        "gpu_exclusive_preflight": gpu,
        "telemetry": telemetry,
        "manifest_identity": manifest_identity,
        "output_identity": output_identity,
    }
    if prepared_receipt_path is not None:
        actual["prepared_receipt"] = _absolute_evidence_ref(
            prepared_receipt_path, "prepared receipt"
        )
        # Keep the receipt reference first in the serialized object order only
        # for readability; canonical validation is key-set based.
        actual = {
            "status": actual["status"],
            "required": actual["required"],
            "prepared_receipt": actual["prepared_receipt"],
            **{key: actual[key] for key in (
                "maintenance_evidence", "executor_record", "gpu_exclusive_preflight",
                "telemetry", "manifest_identity", "output_identity",
            )},
        }
    return actual


def validate_actual_evidence(
    *,
    maintenance_path: Path,
    executor_path: Path,
    output_path: Path,
    profile: dict[str, Any],
    overlay: dict[str, Any],
    package_sha256: str,
    request_id: str,
    prepared_receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Validate post-run evidence and return its canonical receipt projection."""

    return _actual_evidence(
        maintenance_path=maintenance_path,
        executor_path=executor_path,
        output_path=output_path,
        profile=profile,
        overlay=overlay,
        package_sha256=package_sha256,
        request_id=request_id,
        prepared_receipt_path=prepared_receipt_path,
    )


def _exclusive_write(path: Path, raw: bytes) -> None:
    """Publish bytes atomically without replacing an existing destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(raw_path)
        os.fchmod(descriptor, 0o444)
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(raw)
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, path, follow_symlinks=False)
        os.unlink(temporary)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        metadata = path.lstat()
        if stat.S_IMODE(metadata.st_mode) != 0o444 or metadata.st_nlink != 1:
            raise ReceiptError("published receipt topology differs")
    except FileExistsError as error:
        raise ReceiptError("output receipt already exists or is a symlink") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_receipt(
    *,
    profile_path: Path,
    output_path: Path,
    source_tree_sha256: str,
    source_archive_sha256: str,
    served_model_path: Path,
    request_id: str,
    authorization_audit_path: Path | None = None,
    authorization_lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a strict receipt and publish it once, with no overwrite."""

    output_path = output_path.resolve()
    if output_path.exists() or output_path.is_symlink():
        raise ReceiptError("output receipt already exists or is a symlink")
    profile_path = profile_path.resolve()
    profile = _read_object(profile_path, "SQ8 overlay profile")
    worker_profile, product = _profile_contract(profile, output_path)
    profile_promotion = profile["promotion"]
    has_lineage_contract = "authorization_lineage_from_receipt" in profile_promotion
    if has_lineage_contract:
        if profile_promotion.get("authorization_lineage") != authorization_lineage:
            raise ReceiptError("profile/receipt authorization lineage differs")
    elif authorization_lineage is not None:
        raise ReceiptError("legacy profile cannot bind authorization lineage")
    source_commit = _hex(profile["promotion"]["release_source_commit"], 40, "source commit")
    request_id = _request_id(request_id)
    source_tree_sha256 = _hex(source_tree_sha256, 40, "source tree SHA-256")
    source_archive_sha256 = _hex(source_archive_sha256, 64, "source archive SHA-256")
    authorization_audit = _authorization_audit_ref(authorization_audit_path)
    readiness = _readiness_identity(profile["promotion"]["readiness"])

    worker_path = Path(str(worker_profile.get("binary", ""))).resolve()
    if worker_path.is_symlink() or not worker_path.is_file():
        raise ReceiptError("overlay worker binary is unavailable")
    worker_metadata = worker_path.lstat()
    if stat.S_IMODE(worker_metadata.st_mode) != 0o555 or worker_metadata.st_nlink != 1:
        raise ReceiptError("overlay worker binary topology differs")
    worker_sha256 = sha256_file(worker_path)

    product_root = Path(str(product["product"].get("root", ""))).resolve()
    artifact_manifest_path = (product_root / str(product["artifact"].get("manifest_path", ""))).resolve()
    package_manifest_path = (product_root / str(product["package"].get("manifest_path", ""))).resolve()
    binding = _read_object(artifact_manifest_path, "SQ8 overlay binding manifest")
    if (
        binding.get("schema_version") != "ullm.qwen35_aq4_sq8_qkv_z_overlay.v2"
        or binding.get("format_id") != "AQ4_0"
        or binding.get("overlay_format_id") != "SQ8_0"
        or binding.get("implementation_id") != IMPLEMENTATION_ID
        or not isinstance(binding.get("tensor_names"), list)
        or len(binding["tensor_names"]) != 48
        or len(set(binding["tensor_names"])) != 48
    ):
        raise ReceiptError("SQ8 overlay binding identity differs")
    content_sha256 = _hex(binding.get("content_sha256"), 64, "overlay content SHA-256")
    tensor_set_sha256 = _hex(binding.get("tensor_set_sha256"), 64, "overlay tensor-set SHA-256")
    package_manifest_sha256 = sha256_file(package_manifest_path)
    package_ref = binding.get("package")
    if not isinstance(package_ref, dict) or package_ref.get("manifest_sha256") != package_manifest_sha256:
        raise ReceiptError("overlay binding package SHA-256 differs")
    inventory = artifact_inventory(artifact_manifest_path.parent)
    actual = {"status": "pending", "required": True}

    generator = _load_generator()
    synthetic_receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "prepared_not_executed",
        "request_id": request_id,
        "source_commit": source_commit,
        "source_provenance": {
            "tree_sha256": source_tree_sha256,
            "archive_sha256": source_archive_sha256,
        },
        "release": {
            "worker": {
                "path": os.fspath(worker_path),
                "sha256": worker_sha256,
                "bytes": worker_metadata.st_size,
                "mode": "0555",
                "nlink": 1,
            },
            "profile": {
                "path": os.fspath(profile_path),
                "sha256": sha256_file(profile_path),
            },
            "served_model": {
                "path": os.fspath(served_model_path.resolve()),
                "semantic_sha256": "0" * 64,
            },
        },
        "overlay": {
            "binding_manifest_path": os.fspath(artifact_manifest_path),
            "binding_manifest_sha256": sha256_file(artifact_manifest_path),
            "content_sha256": content_sha256,
            "tensor_set_sha256": tensor_set_sha256,
            "tensor_count": 48,
            "artifact_inventory": inventory,
        },
        "package": {
            "manifest_path": os.fspath(package_manifest_path),
            "manifest_sha256": package_manifest_sha256,
        },
        "authorization_audit": authorization_audit,
        "readiness": readiness,
        "actual": actual,
    }
    if has_lineage_contract:
        synthetic_receipt["authorization_lineage"] = authorization_lineage
    try:
        document = generator._materialize_profile_document(
            profile_path,
            receipt_override=synthetic_receipt,
            receipt_sha256_override=hashlib.sha256(
                (json.dumps(synthetic_receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
            ).hexdigest(),
            validate_receipt=False,
        )
    except Exception as error:
        raise ReceiptError(f"overlay served-model binding could not be reconstructed: {error}") from error
    semantic_sha256 = generator._served_model_semantic_sha256(document)
    synthetic_receipt["release"]["served_model"]["semantic_sha256"] = semantic_sha256
    raw = (json.dumps(synthetic_receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _exclusive_write(output_path, raw)
    return synthetic_receipt


def _load_prepared_receipt(path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise ReceiptError("prepared receipt must be a regular non-symlink file")
    prepared = _read_object(path, "prepared promotion receipt")
    expected_prepared = {
        "schema_version", "status", "request_id", "source_commit", "source_provenance",
        "release", "overlay", "package", "authorization_audit", "readiness", "actual",
    }
    prepared_keys = set(prepared)
    if prepared_keys != expected_prepared and prepared_keys != expected_prepared | {
        "authorization_lineage"
    }:
        raise ReceiptError("prepared receipt shape differs")
    if (
        prepared.get("schema_version") != RECEIPT_SCHEMA
        or prepared.get("status") != "prepared_not_executed"
        or prepared.get("actual") != {"status": "pending", "required": True}
    ):
        raise ReceiptError("prepared receipt is not pending")
    request_id = _request_id(prepared.get("request_id"))
    _hex(prepared.get("source_commit"), 40, "prepared source commit")
    source = prepared.get("source_provenance")
    if not isinstance(source, dict):
        raise ReceiptError("prepared source provenance is missing")
    _hex(source.get("tree_sha256"), 40, "prepared source tree")
    _hex(source.get("archive_sha256"), 64, "prepared source archive SHA-256")
    release = prepared.get("release")
    profile = release.get("profile") if isinstance(release, dict) else None
    if not isinstance(profile, dict) or not isinstance(profile.get("path"), str):
        raise ReceiptError("prepared receipt profile binding is missing")
    worker = release.get("worker") if isinstance(release, dict) else None
    served = release.get("served_model") if isinstance(release, dict) else None
    overlay = prepared.get("overlay")
    package = prepared.get("package")
    authorization_audit = prepared.get("authorization_audit")
    if prepared.get("authorization_lineage") is not None:
        try:
            lineage_tool.validate_reference(prepared["authorization_lineage"])
        except lineage_tool.LineageError as error:
            raise ReceiptError(
                f"prepared authorization lineage differs: {error}"
            ) from error
    if authorization_audit is not None:
        if not isinstance(authorization_audit, dict) or set(authorization_audit) != {"path", "sha256"}:
            raise ReceiptError("prepared authorization audit binding is incomplete")
        raw_audit_path = authorization_audit.get("path")
        if not isinstance(raw_audit_path, str) or not Path(raw_audit_path).is_absolute():
            raise ReceiptError("prepared authorization audit path must be absolute")
        audit_path = Path(raw_audit_path)
        if audit_path.is_symlink() or not audit_path.is_file():
            raise ReceiptError("prepared authorization audit path must be a regular non-symlink file")
        _hex(authorization_audit.get("sha256"), 64, "prepared authorization audit SHA-256")
        if audit_path.resolve() != audit_path:
            raise ReceiptError("prepared authorization audit path must be canonical")
        if sha256_file(audit_path) != authorization_audit["sha256"]:
            raise ReceiptError("prepared authorization audit SHA-256 differs")
    _readiness_identity(prepared.get("readiness"))
    for value, label in (
        (worker.get("sha256") if isinstance(worker, dict) else None, "prepared worker SHA-256"),
        (profile.get("sha256"), "prepared profile SHA-256"),
        (served.get("semantic_sha256") if isinstance(served, dict) else None, "prepared served-model semantic SHA-256"),
        (overlay.get("binding_manifest_sha256") if isinstance(overlay, dict) else None, "prepared binding SHA-256"),
        (overlay.get("content_sha256") if isinstance(overlay, dict) else None, "prepared content SHA-256"),
        (overlay.get("tensor_set_sha256") if isinstance(overlay, dict) else None, "prepared tensor-set SHA-256"),
        (package.get("manifest_sha256") if isinstance(package, dict) else None, "prepared package SHA-256"),
    ):
        _hex(value, 64, label)
    profile_path = Path(profile["path"]).resolve()
    if profile.get("sha256") != sha256_file(profile_path):
        raise ReceiptError("prepared receipt profile SHA-256 differs")
    profile_value = _read_object(profile_path, "overlay profile")
    return prepared, profile_value, request_id


def _actual_output_path(path: Path, basename: str) -> Path:
    # Check the caller path before resolving so dangling symlinks cannot be
    # followed into a new publication target.
    if path.is_symlink() or path.name != basename:
        raise ReceiptError(f"actual receipt output must be the create-new {basename} file")
    resolved = path.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise ReceiptError("actual receipt output already exists or is a symlink")
    return resolved


def _final_publication_path(path: Path) -> Path:
    """Resolve the final path when called by the wrapper's hidden staging dir."""

    parent = path.parent
    marker = parent.name
    if marker.startswith(".") and marker.endswith(".incomplete"):
        final_name = marker[1 : -len(".incomplete")]
        if final_name:
            return parent.parent / final_name / path.name
    return path


def write_actual_receipt(
    prepared_receipt_path: Path,
    maintenance_evidence_path: Path,
    executor_record_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Publish a separate actual-verified receipt after one successful run."""

    if prepared_receipt_path.is_symlink():
        raise ReceiptError("prepared receipt must not be a symlink")
    prepared_path = prepared_receipt_path.resolve()
    output_path = _actual_output_path(output_path, "promotion-actual-receipt.json")
    if output_path == prepared_path:
        raise ReceiptError("actual receipt cannot replace the prepared receipt")
    prepared, profile, request_id = _load_prepared_receipt(prepared_path)
    worker_profile, product = _profile_contract(profile, prepared_path)
    generator = _load_generator()
    try:
        generator._materialize_profile_document(
            Path(str(prepared["release"]["profile"]["path"])),
            expected_manifest_path=Path(str(prepared["release"]["served_model"]["path"])),
            receipt_override=prepared,
            validate_receipt=True,
            allow_prepared=True,
        )
    except Exception as error:
        raise ReceiptError(f"prepared receipt binding could not be revalidated: {error}") from error
    product_root = Path(str(product["product"].get("root", ""))).resolve()
    artifact_manifest_path = (product_root / str(product["artifact"].get("manifest_path", ""))).resolve()
    package_manifest_path = (product_root / str(product["package"].get("manifest_path", ""))).resolve()
    binding = _read_object(artifact_manifest_path, "SQ8 overlay binding manifest")
    if binding.get("implementation_id") != IMPLEMENTATION_ID:
        raise ReceiptError("SQ8 overlay binding identity differs")
    content_sha256 = _hex(binding.get("content_sha256"), 64, "overlay content SHA-256")
    package_manifest_sha256 = sha256_file(package_manifest_path)
    if not isinstance(binding.get("package"), dict) or binding["package"].get("manifest_sha256") != package_manifest_sha256:
        raise ReceiptError("overlay binding package SHA-256 differs")
    prepared_overlay = prepared.get("overlay")
    prepared_package = prepared.get("package")
    expected_overlay = {
        "binding_manifest_path": os.fspath(artifact_manifest_path),
        "binding_manifest_sha256": sha256_file(artifact_manifest_path),
        "content_sha256": content_sha256,
        "tensor_set_sha256": _hex(binding.get("tensor_set_sha256"), 64, "overlay tensor-set SHA-256"),
        "tensor_count": 48,
        "artifact_inventory": artifact_inventory(artifact_manifest_path.parent),
    }
    if prepared_overlay != expected_overlay:
        raise ReceiptError("prepared receipt overlay identity differs from live artifact")
    if prepared_package != {
        "manifest_path": os.fspath(package_manifest_path),
        "manifest_sha256": package_manifest_sha256,
    }:
        raise ReceiptError("prepared receipt package identity differs from live package")
    release = prepared.get("release")
    worker = release.get("worker") if isinstance(release, dict) else None
    profile_release = release.get("profile") if isinstance(release, dict) else None
    worker_path = Path(str(worker.get("path", ""))).resolve() if isinstance(worker, dict) else Path("/")
    if (
        not isinstance(worker, dict)
        or worker_path.is_symlink()
        or not worker_path.is_file()
        or worker.get("sha256") != sha256_file(worker_path)
        or worker.get("bytes") != worker_path.stat().st_size
        or worker.get("mode") != "0555"
        or worker.get("nlink") != 1
    ):
        raise ReceiptError("prepared receipt worker identity differs from live worker")
    profile_path = Path(str(profile_release.get("path", ""))).resolve() if isinstance(profile_release, dict) else Path("/")
    if (
        not isinstance(profile_release, dict)
        or profile_path.is_symlink()
        or not profile_path.is_file()
        or profile_release.get("sha256") != sha256_file(profile_path)
    ):
        raise ReceiptError("prepared receipt profile identity differs from live profile")
    actual = _actual_evidence(
        maintenance_path=maintenance_evidence_path,
        executor_path=executor_record_path,
        output_path=output_path,
        profile=profile,
        overlay={
            "binding_manifest_sha256": sha256_file(artifact_manifest_path),
            "content_sha256": content_sha256,
        },
        package_sha256=package_manifest_sha256,
        request_id=request_id,
        prepared_receipt_path=prepared_path,
    )
    if actual.get("status") != "actual_verified":
        raise ReceiptError("actual receipt evidence is not verified")
    receipt = json.loads(json.dumps(prepared, ensure_ascii=True, allow_nan=False))
    receipt["status"] = "actual_verified"
    receipt["actual"] = actual
    served_path = Path(str(receipt["release"]["served_model"]["path"])).resolve()
    final_output_path = _final_publication_path(output_path)
    try:
        document = generator._materialize_profile_document(
            Path(str(receipt["release"]["profile"]["path"])),
            expected_manifest_path=served_path,
            receipt_override=receipt,
            receipt_path_override=final_output_path,
            receipt_sha256_override=hashlib.sha256(
                (json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
            ).hexdigest(),
            validate_receipt=False,
        )
        receipt["release"]["served_model"]["semantic_sha256"] = generator._served_model_semantic_sha256(document)
    except Exception as error:
        raise ReceiptError(f"actual served-model binding could not be reconstructed: {error}") from error
    raw = (json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
    _exclusive_write(output_path, raw)
    return receipt


def write_failure_receipt(
    prepared_receipt_path: Path,
    maintenance_evidence_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Publish a separate immutable failure receipt without promoting the candidate."""

    if prepared_receipt_path.is_symlink():
        raise ReceiptError("prepared receipt must not be a symlink")
    prepared_path = prepared_receipt_path.resolve()
    output_path = _actual_output_path(output_path, "promotion-failure-receipt.json")
    if output_path == prepared_path:
        raise ReceiptError("failure receipt cannot replace the prepared receipt")
    prepared, _profile, request_id = _load_prepared_receipt(prepared_path)
    maintenance = _read_object(maintenance_evidence_path, "maintenance evidence")
    if maintenance.get("schema_version") != "ullm.qwen35_aq4.sq8_overlay_gpu_promotion_maintenance.v1":
        raise ReceiptError("maintenance evidence schema differs")
    if maintenance.get("promotion_request_id") != request_id:
        raise ReceiptError("maintenance promotion request ID differs")
    if maintenance.get("status") != "failed":
        raise ReceiptError("maintenance evidence is not a failed run")
    receipt = json.loads(json.dumps(prepared, ensure_ascii=True, allow_nan=False))
    receipt["status"] = "actual_failed"
    receipt["actual"] = {
        "status": "failed",
        "required": True,
        "prepared_receipt": _absolute_evidence_ref(prepared_path, "prepared receipt"),
        "maintenance_evidence": _relative_evidence_ref(
            maintenance_evidence_path, output_path, "maintenance evidence"
        ),
        "request_id": request_id,
    }
    raw = (json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
    _exclusive_write(output_path, raw)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--served-model", required=True, type=Path)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--authorization-audit-receipt", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = write_receipt(
            profile_path=args.profile,
            output_path=args.output,
            source_tree_sha256=args.source_tree_sha256,
            source_archive_sha256=args.source_archive_sha256,
            served_model_path=args.served_model,
            request_id=args.request_id,
            authorization_audit_path=args.authorization_audit_receipt,
        )
    except Exception as error:
        print(f"SQ8 overlay promotion receipt publication failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
