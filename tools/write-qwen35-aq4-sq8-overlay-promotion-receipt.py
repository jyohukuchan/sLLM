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
import stat
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_SCHEMA = "ullm.qwen35_aq4_sq8_overlay_promotion.v1"
IMPLEMENTATION_ID = "qwen35_aq4_sq8_linear_qkv_z_overlay_v1"
HEX40 = set("0123456789abcdef")
HEX64 = set("0123456789abcdef")
MAX_JSON_BYTES = 32 * 1024 * 1024


class ReceiptError(RuntimeError):
    """Raised when an overlay receipt cannot be safely published."""


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
        "release_source_commit",
    }
    if set(promotion) != expected_keys:
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
) -> dict[str, Any]:
    paths = (maintenance_path, executor_path)
    if all(path is None for path in paths):
        return {"status": "pending", "required": True}
    if any(path is None for path in paths):
        raise ReceiptError("actual maintenance and executor evidence must be supplied together")
    assert maintenance_path is not None and executor_path is not None
    maintenance = _read_object(maintenance_path, "maintenance evidence")
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
        or not isinstance(output_identity["token_ids_sha256"], str)
        or len(output_identity["token_ids_sha256"]) != 64
        or output_identity["token_ids_recorded"] is not False
    ):
        raise ReceiptError("executor output identity differs")
    return {
        "status": "actual_verified",
        "required": True,
        "maintenance_evidence": _relative_evidence_ref(maintenance_path, output_path, "maintenance evidence"),
        "executor_record": _relative_evidence_ref(executor_path, output_path, "executor record"),
        "gpu_exclusive_preflight": gpu,
        "telemetry": telemetry,
        "manifest_identity": manifest_identity,
        "output_identity": output_identity,
    }


def validate_actual_evidence(
    *,
    maintenance_path: Path,
    executor_path: Path,
    output_path: Path,
    profile: dict[str, Any],
    overlay: dict[str, Any],
    package_sha256: str,
) -> dict[str, Any]:
    """Validate post-run evidence and return its canonical receipt projection."""

    return _actual_evidence(
        maintenance_path=maintenance_path,
        executor_path=executor_path,
        output_path=output_path,
        profile=profile,
        overlay=overlay,
        package_sha256=package_sha256,
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
    maintenance_evidence_path: Path | None = None,
    executor_record_path: Path | None = None,
) -> dict[str, Any]:
    """Create a strict receipt and publish it once, with no overwrite."""

    output_path = output_path.resolve()
    if output_path.exists() or output_path.is_symlink():
        raise ReceiptError("output receipt already exists or is a symlink")
    profile_path = profile_path.resolve()
    profile = _read_object(profile_path, "SQ8 overlay profile")
    worker_profile, product = _profile_contract(profile, output_path)
    source_commit = _hex(profile["promotion"]["release_source_commit"], 40, "source commit")
    source_tree_sha256 = _hex(source_tree_sha256, 40, "source tree SHA-256")
    source_archive_sha256 = _hex(source_archive_sha256, 64, "source archive SHA-256")

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
    )

    generator = _load_generator()
    synthetic_receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": (
            "prepared_not_executed"
            if actual["status"] == "pending"
            else "actual_verified"
        ),
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
        "actual": actual,
    }
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--served-model", required=True, type=Path)
    parser.add_argument("--maintenance-evidence", type=Path)
    parser.add_argument("--executor-record", type=Path)
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
            maintenance_evidence_path=args.maintenance_evidence,
            executor_record_path=args.executor_record,
        )
    except Exception as error:
        print(f"SQ8 overlay promotion receipt publication failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
