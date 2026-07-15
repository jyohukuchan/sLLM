#!/usr/bin/env python3
"""Run the frozen AQ4 P2 holdout exactly once.

``preflight`` is CPU-only and emits an immutable, hash-bound command plan.  ``execute``
consumes that plan, creates a one-shot attempt marker before starting the existing Rust
capture binary, and publishes either an immutable failure receipt or an immutable holdout
result.  The runner never derives or changes the calibration envelope.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAX_ROWS = 24
TOP_K = 10
RUN_SCHEMA = "ullm.aq4_p2_fidelity_holdout_run.v1"
PREFLIGHT_SCHEMA = "ullm.aq4_p2_fidelity_holdout_preflight.v1"
ATTEMPT_SCHEMA = "ullm.aq4_p2_fidelity_holdout_attempt.v1"
FAILURE_SCHEMA = "ullm.aq4_p2_fidelity_holdout_failure.v1"
RESULT_SCHEMA = "ullm.aq4_p2_fidelity_holdout_result.v1"
RECEIPT_SCHEMA = "ullm.aq4_p2_fidelity_freeze_receipt.v1"
METRICS_SCHEMA = "ullm.aq4_p2_fidelity_calibration_metrics.v1"
HEX = set("0123456789abcdef")


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROTOCOL = _load("aq4_fidelity_holdout_protocol", "generate-aq4-p2-fidelity-holdout.py")
SPLIT = _load(
    "aq4_fidelity_holdout_split_validator", "validate-aq4-p2-fidelity-holdout.py"
)
CAPTURE = _load("aq4_fidelity_holdout_capture", "capture-qwen35-aq4-fidelity.py")
FULL_COMPARE = _load(
    "aq4_fidelity_holdout_compare", "compare-qwen35-aq4-p2-calibration.py"
)
SERVED = _load("aq4_fidelity_served_model", "generate-served-model.py")

BUILD_RECEIPT_SCHEMA = "ullm.aq4_fidelity_capture_build_receipt.v1"
GUARD_RECEIPT_SCHEMA = "ullm.aq4_p2_resident_guard_receipt.v1"
SOURCE_HOLDOUT_RECEIPT_SCHEMA = "ullm.aq4_p2_fidelity_holdout_source_cases.v1"
ACTUAL_RECEIPT_SCHEMA = "ullm.qwen35_aq4_sq8_overlay_promotion.v1"
DEVICE_ENV = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


class HoldoutError(ValueError):
    pass


def _stable_sha_info(
    path: Path, label: str, limit: int | None = None
) -> tuple[str, os.stat_result]:
    _regular(path, label, limit=limit)
    digest = hashlib.sha256()
    descriptor = os.open(
        path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (limit is not None and before.st_size > limit)
        ):
            raise HoldoutError(f"{label} stable descriptor topology differs")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = os.lstat(path)
    before_fingerprint = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mode,
        before.st_nlink,
        before.st_mtime_ns,
    )
    after_fingerprint = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mode,
        after.st_nlink,
        after.st_mtime_ns,
    )
    final_fingerprint = (
        final.st_dev,
        final.st_ino,
        final.st_size,
        final.st_mode,
        final.st_nlink,
        final.st_mtime_ns,
    )
    if (
        before_fingerprint != after_fingerprint
        or after_fingerprint != final_fingerprint
    ):
        raise HoldoutError(f"{label} changed while hashing")
    return digest.hexdigest(), after


def _sha(path: Path, label: str, limit: int | None = None) -> str:
    return _stable_sha_info(path, label, limit)[0]


def _no_symlink_components(
    path: Path, label: str, *, missing_leaf: bool = False
) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for index, component in enumerate(absolute.parts[1:], 1):
        current /= component
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if missing_leaf and index == len(absolute.parts) - 1:
                return
            raise HoldoutError(f"{label} path component is unavailable: {current}")
        if stat.S_ISLNK(info.st_mode):
            raise HoldoutError(f"{label} path component is a symlink: {current}")


def _regular(
    path: Path, label: str, *, limit: int | None = None, missing: bool = False
) -> None:
    _no_symlink_components(path, label, missing_leaf=missing)
    try:
        info = os.lstat(path)
    except OSError as error:
        raise HoldoutError(f"{label} metadata unavailable: {error}") from error
    if not stat.S_ISREG(info.st_mode):
        raise HoldoutError(f"{label} must be a regular file")
    if info.st_nlink != 1:
        raise HoldoutError(f"{label} must have exactly one hard link")
    if limit is not None and info.st_size > limit:
        raise HoldoutError(f"{label} exceeds bounded size")


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    limit = 64 * 1024 * 1024
    _regular(path, label, limit=limit)
    descriptor = os.open(
        path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        before = os.fstat(descriptor)
        raw_buffer = bytearray()
        while chunk := os.read(
            descriptor, min(1024 * 1024, limit + 1 - len(raw_buffer))
        ):
            raw_buffer.extend(chunk)
            if len(raw_buffer) > limit:
                raise HoldoutError(f"{label} exceeds bounded size")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = os.lstat(path)
    before_fingerprint = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mode,
        before.st_nlink,
        before.st_mtime_ns,
    )
    after_fingerprint = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mode,
        after.st_nlink,
        after.st_mtime_ns,
    )
    final_fingerprint = (
        final.st_dev,
        final.st_ino,
        final.st_size,
        final.st_mode,
        final.st_nlink,
        final.st_mtime_ns,
    )
    if (
        before_fingerprint != after_fingerprint
        or after_fingerprint != final_fingerprint
    ):
        raise HoldoutError(f"{label} changed while reading")
    raw = bytes(raw_buffer)
    try:
        value = json.loads(
            raw, object_pairs_hook=PROTOCOL.pairs, parse_constant=PROTOCOL.no_constants
        )
    except (UnicodeError, json.JSONDecodeError, PROTOCOL.ProtocolError) as error:
        raise HoldoutError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise HoldoutError(f"{label} root must be an object")
    return value, raw


def _atomic_json(path: Path, value: Any, label: str) -> str:
    if os.path.lexists(path):
        raise HoldoutError(f"refusing to overwrite {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _no_symlink_components(path.parent, f"{label} parent")
    encoded = (
        json.dumps(
            value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False
        ).encode()
        + b"\n"
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    if os.path.lexists(temporary):
        raise HoldoutError(f"incomplete {label} already exists")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    try:
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        os.unlink(temporary)
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(encoded).hexdigest()


def _sha_value(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in HEX for char in value)
    ):
        raise HoldoutError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _identity_file(
    path: Path, label: str, expected_sha: str | None = None
) -> dict[str, Any]:
    digest, info = _stable_sha_info(path, label)
    if expected_sha is not None and digest != _sha_value(
        expected_sha, f"expected {label} SHA"
    ):
        raise HoldoutError(f"{label} SHA differs")
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "bytes": info.st_size,
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "nlink": info.st_nlink,
    }


def _exact_object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise HoldoutError(f"{label} shape differs")
    return value


def _revalidate_identity(identity: Any, label: str) -> dict[str, Any]:
    item = _exact_object(
        identity, {"path", "sha256", "bytes", "mode", "nlink"}, f"{label} identity"
    )
    if (
        item.get("nlink") != 1
        or not isinstance(item.get("bytes"), int)
        or item["bytes"] < 0
    ):
        raise HoldoutError(f"{label} identity topology differs")
    current = _identity_file(Path(item["path"]), label, item["sha256"])
    if current != item:
        raise HoldoutError(f"{label} path/content/size/mode/topology changed")
    return current


def _command_sha(command: Any) -> str:
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item for item in command)
    ):
        raise HoldoutError("execution command is invalid")
    return hashlib.sha256(
        json.dumps(
            command, ensure_ascii=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _git(worktree: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(worktree), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise HoldoutError(f"build worktree git validation failed: {error}") from error
    if result.returncode != 0:
        raise HoldoutError(
            f"build worktree git validation failed: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout.decode("utf-8", errors="strict").strip()


def _package_tree(root: Path) -> dict[str, Any]:
    _no_symlink_components(root, "package root")
    root_info = os.lstat(root)
    if not stat.S_ISDIR(root_info.st_mode):
        raise HoldoutError("package root must be a real directory")
    paths: list[Path] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        base = Path(current)
        for name in directories:
            info = os.lstat(base / name)
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise HoldoutError(f"package directory topology differs: {base / name}")
        for name in files:
            path = base / name
            _regular(path, f"package file {path.relative_to(root)}")
            paths.append(path)
    if not paths:
        raise HoldoutError("package tree is empty")
    aggregate = hashlib.sha256()
    files: dict[str, Any] = {}
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        identity = _identity_file(path, f"package file {relative}")
        files[relative] = identity
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(identity["sha256"]))
        aggregate.update(b"\n")
    return {
        "root": str(root.resolve()),
        "content_sha256": aggregate.hexdigest(),
        "files": files,
    }


def _load_rows(path: Path, subset: str) -> list[dict[str, Any]]:
    _regular(path, f"{subset} cases", limit=16 * 1024 * 1024)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or len(line) > 64 * 1024:
            raise HoldoutError(f"{subset} row {number} is empty or oversized")
        try:
            value = json.loads(
                line,
                object_pairs_hook=PROTOCOL.pairs,
                parse_constant=PROTOCOL.no_constants,
            )
        except (UnicodeError, json.JSONDecodeError, PROTOCOL.ProtocolError) as error:
            raise HoldoutError(f"invalid {subset} row {number}: {error}") from error
        if not isinstance(value, dict) or value.get("case_id") in seen:
            raise HoldoutError(f"{subset} rows contain duplicate case_id")
        if (
            value.get("subset") != subset
            or value.get("step") != 0
            or value.get("row_count") != 1
        ):
            raise HoldoutError(f"{subset} row contract differs: {value.get('case_id')}")
        seen.add(value.get("case_id"))
        rows.append(value)
    if len(rows) != MAX_ROWS:
        raise HoldoutError(f"{subset} rows must contain exactly {MAX_ROWS} entries")
    return rows


def _freeze(
    split_root: Path, freeze_receipt: Path
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]
]:
    for name in (
        "split-manifest.json",
        "policy.json",
        "calibration-cases.jsonl",
        "holdout-cases.jsonl",
        "SHA256SUMS",
    ):
        _regular(split_root / name, name)
    _regular(freeze_receipt, "freeze receipt")
    try:
        result = SPLIT.validate(split_root, freeze_receipt)
    except Exception as error:
        raise HoldoutError(f"split/freeze validation failed: {error}") from error
    manifest, manifest_raw = PROTOCOL.load(
        split_root / "split-manifest.json", "split manifest"
    )
    policy, policy_raw = PROTOCOL.load(split_root / "policy.json", "policy")
    receipt, receipt_raw = PROTOCOL.load(freeze_receipt, "freeze receipt")
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("status") != "frozen_calibration_envelope"
    ):
        raise HoldoutError("freeze receipt is not a frozen calibration envelope")
    if (
        receipt.get("holdout_status") != "not_started"
        or receipt.get("holdout_evaluations_remaining") != 1
    ):
        raise HoldoutError("holdout has already been evaluated or is not frozen")
    holdout_path = split_root / "holdout-cases.jsonl"
    calibration_path = split_root / "calibration-cases.jsonl"
    _load_rows(calibration_path, "calibration")
    _load_rows(holdout_path, "holdout")
    shas = {
        "split_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "policy_sha256": hashlib.sha256(policy_raw).hexdigest(),
        "calibration_cases_sha256": _sha(calibration_path, "calibration cases"),
        "holdout_cases_sha256": _sha(holdout_path, "holdout cases"),
        "freeze_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
    }
    if (
        receipt.get("split_manifest_sha256") != shas["split_manifest_sha256"]
        or receipt.get("policy_sha256") != shas["policy_sha256"]
    ):
        raise HoldoutError("freeze receipt split/policy binding differs")
    if (
        manifest.get("calibration_sha256") != shas["calibration_cases_sha256"]
        or manifest.get("holdout_sha256") != shas["holdout_cases_sha256"]
    ):
        raise HoldoutError("split manifest cases binding differs")
    return manifest, policy, receipt, result, shas


def _actual_verified(path: Path, served_manifest: Path | None = None) -> dict[str, Any]:
    """Validate the full shared SQ8 promotion receipt, including actual lineage."""

    identity = _identity_file(path, "actual-verified receipt")
    value, _ = _read_json(path, "actual-verified receipt")
    _exact_object(
        value,
        {
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
        },
        "actual-verified receipt",
    )
    if (
        value.get("schema_version") != ACTUAL_RECEIPT_SCHEMA
        or value.get("status") != "actual_verified"
    ):
        raise HoldoutError("actual-verified receipt schema/status differs")
    actual = _exact_object(
        value.get("actual"),
        {
            "status",
            "required",
            "prepared_receipt",
            "maintenance_evidence",
            "executor_record",
            "gpu_exclusive_preflight",
            "telemetry",
            "manifest_identity",
            "output_identity",
        },
        "actual evidence",
    )
    if actual.get("status") != "actual_verified" or actual.get("required") is not True:
        raise HoldoutError("actual evidence status differs")
    release = _exact_object(
        value.get("release"), {"worker", "profile", "served_model"}, "actual release"
    )
    profile_ref = _exact_object(
        release.get("profile"), {"path", "sha256"}, "actual profile reference"
    )
    profile_path = Path(str(profile_ref["path"])).resolve()
    profile, _ = _read_json(profile_path, "served-model profile")
    product = _exact_object(
        profile.get("product"),
        {"root", "artifact", "package"},
        "served-model product profile",
    )
    promotion = profile.get("promotion")
    profile_worker = profile.get("worker")
    required_guard_names = (
        profile_worker.get("required_environment")
        if isinstance(profile_worker, dict)
        else None
    )
    if (
        not isinstance(required_guard_names, list)
        or not required_guard_names
        or not all(isinstance(name, str) and name for name in required_guard_names)
        or len(set(required_guard_names)) != len(required_guard_names)
    ):
        raise HoldoutError("served-model profile required guard set differs")
    worker_ref = _exact_object(
        release.get("worker"),
        {"path", "sha256", "bytes", "mode", "nlink"},
        "actual worker reference",
    )
    worker = _revalidate_identity(worker_ref, "actual worker")
    package_ref = _exact_object(
        value.get("package"),
        {"manifest_path", "manifest_sha256"},
        "actual package reference",
    )
    package = _identity_file(
        Path(str(package_ref["manifest_path"])),
        "actual package manifest",
        package_ref["manifest_sha256"],
    )
    try:
        SERVED._validate_sq8_overlay_receipt(
            profile=profile,
            promotion_profile=promotion,
            receipt=value,
            receipt_path=path.resolve(),
            profile_path=profile_path,
            worker_binary=Path(worker["path"]),
            worker_sha256=worker["sha256"],
            product_root=Path(str(product["root"])).resolve(),
            artifact_manifest_path=str(product["artifact"]["manifest_path"]),
            package_manifest_path=str(product["package"]["manifest_path"]),
            package_manifest_sha256=package["sha256"],
            expected_manifest_path=served_manifest.resolve()
            if served_manifest is not None
            else None,
            allow_prepared=False,
            prepared_only=False,
        )
    except Exception as error:
        raise HoldoutError(
            f"shared actual-verified contract failed: {error}"
        ) from error
    lineage: dict[str, Any] = {}
    for name in ("prepared_receipt", "maintenance_evidence", "executor_record"):
        reference = _exact_object(
            actual[name], {"path", "sha256"}, f"actual {name} reference"
        )
        reference_path = Path(str(reference["path"]))
        if not reference_path.is_absolute():
            reference_path = (path.parent / reference_path).resolve()
        lineage[name] = _identity_file(
            reference_path, f"actual {name}", reference["sha256"]
        )
    return {
        "receipt": identity,
        "request_id": value["request_id"],
        "source_commit": value["source_commit"],
        "source_provenance": value["source_provenance"],
        "profile": _identity_file(
            profile_path, "served-model profile", profile_ref["sha256"]
        ),
        "worker": worker,
        "package_manifest": package,
        "required_guard_names": sorted(required_guard_names),
        "lineage": lineage,
    }


def _guard_receipt(path: Path, expected_sha: str) -> dict[str, Any]:
    identity = _identity_file(path, "guard receipt")
    value, _ = _read_json(path, "guard receipt")
    _exact_object(
        value,
        {"schema_version", "status", "guard_sha256", "required_environment"},
        "guard receipt",
    )
    # The Rust capture and resident driver use this canonical digest contract.
    required = value.get("required_environment")
    if (
        value.get("schema_version") != GUARD_RECEIPT_SCHEMA
        or value.get("status") != "ready"
        or not isinstance(required, dict)
        or not required
        or set(required.values()) != {"1"}
    ):
        raise HoldoutError("guard receipt contract differs")
    digest = hashlib.sha256(b"ullm-aq4-p2-resident-guards-v1\0")
    for name in sorted(required):
        digest.update(f"{name}=1\n".encode())
    guard_sha = digest.hexdigest()
    if value.get("guard_sha256") != guard_sha or guard_sha != _sha_value(
        expected_sha, "guard SHA"
    ):
        raise HoldoutError("guard receipt SHA contract differs")
    return {
        "receipt": identity,
        "guard_sha256": guard_sha,
        "required_environment": required,
    }


def _build_receipt(path: Path, capture_binary: Path) -> dict[str, Any]:
    identity = _identity_file(path, "capture build receipt")
    value, _ = _read_json(path, "capture build receipt")
    _exact_object(
        value,
        {"schema_version", "status", "source", "build", "binary"},
        "capture build receipt",
    )
    if (
        value.get("schema_version") != BUILD_RECEIPT_SCHEMA
        or value.get("status") != "ready"
    ):
        raise HoldoutError("capture build receipt schema/status differs")
    source = _exact_object(
        value["source"],
        {"commit", "tree_sha256", "tree_clean", "cargo_lock_sha256"},
        "capture build source",
    )
    build = _exact_object(
        value["build"],
        {"worktree", "command", "exit_status", "log"},
        "capture build invocation",
    )
    binary = _exact_object(
        value["binary"],
        {"path", "sha256", "bytes", "nlink", "mode"},
        "capture build binary",
    )
    if (
        source.get("tree_clean") is not True
        or build.get("exit_status") != 0
        or not isinstance(build.get("command"), str)
        or "--bin ullm-aq4-fidelity-capture" not in build["command"]
    ):
        raise HoldoutError(
            "capture build invocation is not a successful clean binary build"
        )
    for label in ("commit", "tree_sha256"):
        digest = source.get(label)
        if (
            not isinstance(digest, str)
            or len(digest) != 40
            or any(char not in HEX for char in digest)
        ):
            raise HoldoutError(f"capture build source {label} is invalid")
    worktree = Path(str(build["worktree"])).resolve()
    if (
        _git(worktree, "rev-parse", "HEAD") != source["commit"]
        or _git(worktree, "rev-parse", "HEAD^{tree}") != source["tree_sha256"]
        or _git(worktree, "status", "--porcelain")
    ):
        raise HoldoutError("capture build source worktree identity/cleanliness differs")
    cargo_lock = _identity_file(
        worktree / "Cargo.lock", "capture build Cargo.lock", source["cargo_lock_sha256"]
    )
    binary_path = Path(str(binary["path"]))
    if not binary_path.is_absolute():
        binary_path = (path.parent / binary_path).resolve()
    observed_binary = _identity_file(
        binary_path, "capture build binary", binary["sha256"]
    )
    if (
        observed_binary
        != {
            "path": str(binary_path),
            "sha256": binary["sha256"],
            "bytes": binary["bytes"],
            "mode": binary["mode"],
            "nlink": binary["nlink"],
        }
        or binary_path != capture_binary.resolve()
        or observed_binary["mode"] not in {"0555", "0755"}
    ):
        raise HoldoutError("capture build receipt/binary binding differs")
    log_path = Path(str(build["log"]))
    if not log_path.is_absolute():
        log_path = (path.parent / log_path).resolve()
    return {
        "receipt": identity,
        "source": source,
        "cargo_lock": cargo_lock,
        "binary": observed_binary,
        "build_log": _identity_file(log_path, "capture build log"),
    }


def _source_holdout_receipt(
    path: Path, cases_path: Path, shas: dict[str, str]
) -> dict[str, Any]:
    identity = _identity_file(path, "source holdout receipt")
    value, _ = _read_json(path, "source holdout receipt")
    _exact_object(
        value,
        {
            "schema_version",
            "status",
            "subset",
            "observation",
            "row_count",
            "cases",
            "split",
        },
        "source holdout receipt",
    )
    cases = _exact_object(
        value.get("cases"),
        {"path", "sha256", "bytes", "mode", "nlink"},
        "source holdout cases",
    )
    split = _exact_object(value.get("split"), set(shas), "source holdout split binding")
    if (
        value.get("schema_version") != SOURCE_HOLDOUT_RECEIPT_SCHEMA
        or value.get("status") != "ready"
        or value.get("subset") != "holdout"
        or value.get("observation") != "fidelity_holdout_full_context_step0"
        or value.get("row_count") != MAX_ROWS
        or split != shas
    ):
        raise HoldoutError("source holdout receipt contract differs")
    if Path(cases["path"]).resolve() != cases_path.resolve():
        raise HoldoutError("source holdout receipt cases path differs")
    _revalidate_identity(cases, "source holdout cases")
    return {"receipt": identity, "cases": cases, "split": split}


def _artifact_identity(
    root: Path, kind: str, rows: list[dict[str, Any]], subset: str
) -> dict[str, Any]:
    try:
        artifact = CAPTURE._artifact(root, kind)
    except Exception as error:
        raise HoldoutError(f"{kind} artifact validation failed: {error}") from error
    manifest = artifact["manifest"]
    if manifest.get("subset") not in (None, subset):
        raise HoldoutError(f"{kind} artifact subset differs")
    expected = {(row["case_id"], 0): row for row in rows}
    if set(artifact["rows"]) != set(expected):
        raise HoldoutError(f"{kind} artifact must contain exactly the holdout 24 rows")
    for key, row in artifact["rows"].items():
        expected_row = expected[key]
        if (
            row.get("step") != 0
            or row.get("case_id") != key[0]
            or row.get("input_token_ids_sha256")
            != expected_row.get("context_token_ids_sha256")
        ):
            raise HoldoutError(f"{kind} artifact row identity differs: {key[0]}")
    if kind == "aq4_target":
        for directory in (root, root / "vectors"):
            info = os.lstat(directory)
            if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o555:
                raise HoldoutError(
                    f"active artifact directory is not sealed 0555: {directory}"
                )
        for name, path in artifact["tracked"].items():
            info = os.lstat(path)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o444
            ):
                raise HoldoutError(
                    f"active artifact member is not sealed 0444/nlink1: {name}"
                )
    return artifact


def _runtime_identity(
    active: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    runtime = active["manifest"].get("runtime", {})
    nested = runtime.get("runtime", {}) if isinstance(runtime, dict) else {}
    required = {
        "served_model_manifest_sha256": expected["served_model_manifest_sha256"],
        "package_manifest_sha256": expected["package_manifest_sha256"],
        "package_content_sha256": expected["package_content_sha256"],
        "worker_binary_sha256": expected["worker_binary_sha256"],
        "capture_binary_sha256": expected["capture_binary_sha256"],
        "guard_sha256": expected["guard_sha256"],
        "selected_cases_sha256": expected["holdout_cases_sha256"],
        "split_manifest_sha256": expected["split_manifest_sha256"],
        "policy_sha256": expected["policy_sha256"],
        "holdout_cases_sha256": expected["holdout_cases_sha256"],
        "quantized_artifact_revision": expected["quantized_artifact_revision"],
    }
    for field, value in required.items():
        if nested.get(field) != value:
            raise HoldoutError(f"active runtime identity differs: {field}")
    if (
        nested.get("one_process") is not True
        or nested.get("one_model_load") is not True
        or nested.get("gpu_parallelism") != 1
        or runtime.get("model_loads") != 1
    ):
        raise HoldoutError(
            "active runtime one-process/model-load/GPU-parallelism contract differs"
        )
    device = nested.get("device")
    if not isinstance(device, dict) or device != {
        "requested_index": expected["device_index"],
        "device_id": expected["device_id"],
        "backend": expected["device_backend"],
        "name": expected["device_name"],
        "architecture": expected["device_architecture"],
    }:
        raise HoldoutError("active full device identity differs")
    if nested.get("selected_subset") != "holdout":
        raise HoldoutError("active selected subset is not holdout")
    if nested.get("build_sha256") != expected["build_sha256"]:
        raise HoldoutError("active capture build SHA differs")
    state = nested.get("state_evidence")
    expected_state = {
        "contract": "full_context_step_zero_reset_v1",
        "rows_started": MAX_ROWS,
        "rows_completed": MAX_ROWS,
        "clean_before_each_row": True,
        "generation_states_observed": MAX_ROWS,
        "reset_calls": MAX_ROWS,
        "clean_after_each_reset": True,
        "scheduler_mode": "not_used_direct_capture",
        "scheduler_pending_before_each_row": 0,
        "scheduler_pending_after_each_row": 0,
    }
    if state != expected_state:
        raise HoldoutError("active request/scheduler/reset state evidence differs")
    source_identity = expected.get("source_identity", {})
    if nested.get("upstream_model_revision") != source_identity.get(
        "model_revision"
    ) or nested.get("tokenizer_aggregate_sha256") != source_identity.get(
        "tokenizer", {}
    ).get("aggregate_sha256"):
        raise HoldoutError("active upstream source identity differs")
    source_checkpoint_sha = source_identity.get("source_checkpoint", {}).get(
        "aggregate_sha256"
    )
    if (
        source_checkpoint_sha is not None
        and nested.get("source_checkpoint_aggregate_sha256") != source_checkpoint_sha
    ):
        raise HoldoutError("active source checkpoint identity differs")
    return nested


def _source_active_identity(
    source: dict[str, Any], active: dict[str, Any]
) -> dict[str, Any]:
    left = source["manifest"].get("identity", {})
    right = active["manifest"].get("identity", {})
    if (
        left.get("model_id") != right.get("model_id")
        or left.get("model_revision") != right.get("model_revision")
        or left.get("tokenizer", {}).get("aggregate_sha256")
        != right.get("tokenizer", {}).get("aggregate_sha256")
    ):
        raise HoldoutError("source/active source identity differs")
    return {
        "model_id": left.get("model_id"),
        "model_revision": left.get("model_revision"),
        "tokenizer_aggregate_sha256": left.get("tokenizer", {}).get("aggregate_sha256"),
    }


def _compare(
    source: dict[str, Any], active: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    metrics_rows: list[dict[str, Any]] = []
    aggregate = {name: [] for name in PROTOCOL.METRICS}
    with (
        FULL_COMPARE._VALIDATOR.stable_fd(source["hidden"], "source hidden") as (
            source_hidden,
            _,
        ),
        FULL_COMPARE._VALIDATOR.stable_fd(source["logits"], "source logits") as (
            source_logits,
            _,
        ),
        FULL_COMPARE._VALIDATOR.stable_fd(active["hidden"], "active hidden") as (
            active_hidden,
            _,
        ),
        FULL_COMPARE._VALIDATOR.stable_fd(active["logits"], "active logits") as (
            active_logits,
            _,
        ),
    ):
        for split_row in sorted(rows, key=lambda item: item["case_id"]):
            key = (split_row["case_id"], 0)
            left = source["rows"][key]
            right = active["rows"][key]
            if left.get("input_token_ids_sha256") != split_row.get(
                "context_token_ids_sha256"
            ) or right.get("input_token_ids_sha256") != split_row.get(
                "context_token_ids_sha256"
            ):
                raise HoldoutError(f"input identity differs: {split_row['case_id']}")
            source_top = [item["token_id"] for item in left["topk"]]
            active_top = [item["token_id"] for item in right["topk"]]
            hidden = CAPTURE._stream_stats(
                CAPTURE._chunks(
                    source_hidden,
                    left["hidden"]["offset_bytes"],
                    CAPTURE.HIDDEN_SIZE,
                    source["chunk_elements"],
                ),
                CAPTURE._chunks(
                    active_hidden,
                    right["hidden"]["offset_bytes"],
                    CAPTURE.HIDDEN_SIZE,
                    active["chunk_elements"],
                ),
                CAPTURE.HIDDEN_SIZE,
            )
            logits = CAPTURE._stream_stats(
                CAPTURE._chunks(
                    source_logits,
                    left["logits"]["offset_bytes"],
                    CAPTURE.VOCAB_SIZE,
                    source["chunk_elements"],
                ),
                CAPTURE._chunks(
                    active_logits,
                    right["logits"]["offset_bytes"],
                    CAPTURE.VOCAB_SIZE,
                    active["chunk_elements"],
                ),
                CAPTURE.VOCAB_SIZE,
            )
            values = {
                "token_agreement_rate": float(
                    left["greedy_token_id"] == right["greedy_token_id"]
                ),
                "topk_overlap_rate_k10": len(set(source_top) & set(active_top)) / TOP_K,
                "logits_cosine": logits["cosine"],
                "logits_relative_l2": logits["relative_l2"],
                "hidden_cosine": hidden["cosine"],
                "hidden_relative_l2": hidden["relative_l2"],
                "hidden_max_abs": hidden["max_abs"],
                "bf16_top1_retained_in_aq4_top10_rate": float(
                    left["greedy_token_id"] in active_top
                ),
            }
            for name, value in values.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise HoldoutError(
                        f"non-finite holdout metric: {split_row['case_id']}.{name}"
                    )
                numeric = float(value)
                if (
                    name
                    in {
                        "token_agreement_rate",
                        "topk_overlap_rate_k10",
                        "bf16_top1_retained_in_aq4_top10_rate",
                    }
                    and not 0.0 <= numeric <= 1.0
                ):
                    raise HoldoutError(
                        f"holdout metric outside [0,1]: {split_row['case_id']}.{name}"
                    )
                if (
                    name in {"logits_cosine", "hidden_cosine"}
                    and not -1.0 <= numeric <= 1.0
                ):
                    raise HoldoutError(
                        f"holdout cosine outside [-1,1]: {split_row['case_id']}.{name}"
                    )
                if (
                    name
                    not in {
                        "token_agreement_rate",
                        "topk_overlap_rate_k10",
                        "bf16_top1_retained_in_aq4_top10_rate",
                        "logits_cosine",
                        "hidden_cosine",
                    }
                    and numeric < 0.0
                ):
                    raise HoldoutError(
                        f"holdout metric is negative: {split_row['case_id']}.{name}"
                    )
                if (
                    name in {"logits_relative_l2", "hidden_relative_l2"}
                    and float(value) > 1.0
                ):
                    raise HoldoutError(
                        f"pathological relative-L2 > 1: {split_row['case_id']}.{name}"
                    )
                aggregate[name].append(float(value))
            metrics_rows.append(
                {
                    "case_id": split_row["case_id"],
                    "case_sha256": split_row["case_sha256"],
                    "fixture_sha256": split_row["fixture_sha256"],
                    "prompt_token_ids_sha256": split_row["prompt_token_ids_sha256"],
                    "context_token_ids_sha256": split_row["context_token_ids_sha256"],
                    "prompt_tokens": split_row["prompt_tokens"],
                    "context_tokens": split_row["context_tokens"],
                    "baseline_mode": split_row["baseline_mode"],
                    "prefill_requested_m": split_row["prefill_requested_m"],
                    "resolved_m": split_row["resolved_m"],
                    "step": 0,
                    "row_count": 1,
                    "greedy": {
                        "source": left["greedy_token_id"],
                        "active": right["greedy_token_id"],
                        "exact": left["greedy_token_id"] == right["greedy_token_id"],
                    },
                    "ordered_top10": {
                        "source": source_top,
                        "active": active_top,
                        "exact": source_top == active_top,
                        "overlap": values["topk_overlap_rate_k10"],
                    },
                    "metrics": values,
                }
            )
    means = {
        name: (
            max(values)
            if PROTOCOL.METRICS[name]["role"] == "diagnostic_only"
            else sum(values) / len(values)
        )
        for name, values in aggregate.items()
    }
    return metrics_rows, means


def _decision(
    means: dict[str, float], receipt: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    bounds = receipt.get("derived_bounds")
    if not isinstance(bounds, dict) or set(bounds) != set(PROTOCOL.METRICS):
        raise HoldoutError("freeze receipt bounds are incomplete")
    checks: dict[str, Any] = {}
    go = True
    for name, spec in PROTOCOL.METRICS.items():
        item = bounds[name]
        observed = means[name]
        if spec["role"] == "diagnostic_only":
            checks[name] = {"observed": observed, "bound": None, "pass": True}
            continue
        bound = float(item["bound"])
        passed = (
            observed >= bound if spec["direction"] == "higher" else observed <= bound
        )
        checks[name] = {"observed": observed, "bound": bound, "pass": passed}
        go = go and passed
    return ("go" if go else "no_go"), checks


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    manifest, policy, receipt, _validated, shas = _freeze(
        args.split_root, args.freeze_receipt
    )
    actual = _actual_verified(args.actual_verified_receipt, args.served_model_manifest)
    source_manifest = args.source_artifact / "manifest.json"
    source_identity = _identity_file(source_manifest, "source artifact manifest")
    split_rows = _load_rows(args.split_root / "holdout-cases.jsonl", "holdout")
    source = _artifact_identity(
        args.source_artifact, "independent_source_full", split_rows, "holdout"
    )
    source_cases_path = Path(source["manifest"]["cases"]["path"])
    if not source_cases_path.is_absolute():
        source_cases_path = (args.source_artifact / source_cases_path).resolve()
    source_cases_sha = _sha(source_cases_path, "source cases")
    source_case_value, _ = _read_json(source_cases_path, "source cases")
    if (
        source_case_value.get("schema_version")
        != "ullm.qwen35_aq4_source_calibration_cases.v1"
        or len(source_case_value.get("cases", [])) != MAX_ROWS
    ):
        raise HoldoutError("source cases schema/count differs")
    if {item.get("case_id") for item in source_case_value["cases"]} != {
        item["case_id"] for item in split_rows
    }:
        raise HoldoutError("source cases are not exactly the holdout cases")
    if any(
        item.get("observation") != "fidelity_holdout_full_context_step0"
        for item in source_case_value["cases"]
    ):
        raise HoldoutError("source cases lack the formal holdout observation marker")
    if source["manifest"]["cases"].get("sha256") != source_cases_sha:
        raise HoldoutError("source artifact cases SHA differs")
    source_split_shas = {
        name: shas[name]
        for name in (
            "split_manifest_sha256",
            "policy_sha256",
            "calibration_cases_sha256",
            "holdout_cases_sha256",
        )
    }
    source_holdout = _source_holdout_receipt(
        args.source_holdout_receipt, source_cases_path, source_split_shas
    )
    build = _build_receipt(args.build_receipt, args.capture_binary)
    if build["receipt"]["sha256"] != _sha_value(
        args.expected_build_receipt_sha256, "capture build receipt SHA"
    ):
        raise HoldoutError("capture build receipt SHA differs")
    capture_identity = _identity_file(
        args.capture_binary, "capture binary", args.expected_capture_binary_sha256
    )
    if build["binary"] != capture_identity:
        raise HoldoutError("capture binary differs from the build receipt binary")
    served_identity = _identity_file(
        args.served_model_manifest,
        "served model manifest",
        args.expected_served_model_manifest_sha256,
    )
    if capture_identity["mode"] not in {"0555", "0755"}:
        raise HoldoutError("capture binary mode is not executable")
    if actual["package_manifest"]["sha256"] != _sha_value(
        args.expected_package_manifest_sha256, "package manifest SHA"
    ) or actual["worker"]["sha256"] != _sha_value(
        args.expected_worker_binary_sha256, "worker binary SHA"
    ):
        raise HoldoutError(
            "actual-verified package/worker identity differs from the pinned contract"
        )
    package_tree = _package_tree(Path(actual["package_manifest"]["path"]).parent)
    if package_tree["content_sha256"] != _sha_value(
        args.expected_package_content_sha256, "package content SHA"
    ):
        raise HoldoutError("live package content differs from the pinned contract")
    guard = _guard_receipt(args.guard_receipt, args.expected_guard_sha256)
    if guard["required_environment"] != {
        name: "1" for name in actual["required_guard_names"]
    }:
        raise HoldoutError(
            "guard receipt differs from the actual-verified served profile"
        )
    for label, value in (
        ("device backend", args.expected_device_backend),
        ("device name", args.expected_device_name),
        ("device architecture", args.expected_device_architecture),
        ("device ID", args.expected_device_id),
        ("quantized artifact revision", args.expected_quantized_artifact_revision),
    ):
        if not isinstance(value, str) or not value:
            raise HoldoutError(f"{label} must be nonempty")
    if (
        args.device_index < 0
        or args.chunk_elements < 1
        or args.timeout_seconds <= 0
        or not math.isfinite(args.timeout_seconds)
    ):
        raise HoldoutError("device/chunk/timeout execution bounds are invalid")
    expected = {
        **shas,
        "served_model_manifest_sha256": served_identity["sha256"],
        "package_manifest_sha256": actual["package_manifest"]["sha256"],
        "package_content_sha256": package_tree["content_sha256"],
        "worker_binary_sha256": actual["worker"]["sha256"],
        "capture_binary_sha256": capture_identity["sha256"],
        "build_sha256": capture_identity["sha256"],
        "guard_sha256": guard["guard_sha256"],
        "device_index": args.device_index,
        "device_backend": args.expected_device_backend,
        "device_name": args.expected_device_name,
        "device_architecture": args.expected_device_architecture,
        "device_id": args.expected_device_id,
        "quantized_artifact_revision": args.expected_quantized_artifact_revision,
    }
    command = [
        str(args.capture_binary.resolve()),
        "--served-model-manifest",
        str(args.served_model_manifest.resolve()),
        "--split-root",
        str(args.split_root.resolve()),
        "--source",
        str(args.source_artifact.resolve()),
        "--cases-file",
        str(source_cases_path.resolve()),
        "--output",
        str(args.active_output.resolve()),
        "--subset",
        "holdout",
        "--device-index",
        str(args.device_index),
        "--chunk-elements",
        str(args.chunk_elements),
        "--expected-split-manifest-sha256",
        shas["split_manifest_sha256"],
        "--expected-policy-sha256",
        shas["policy_sha256"],
        "--expected-calibration-cases-sha256",
        shas["calibration_cases_sha256"],
        "--expected-holdout-cases-sha256",
        shas["holdout_cases_sha256"],
        "--expected-served-model-manifest-sha256",
        expected["served_model_manifest_sha256"],
        "--expected-package-manifest-sha256",
        expected["package_manifest_sha256"],
        "--expected-worker-binary-sha256",
        expected["worker_binary_sha256"],
        "--expected-guard-sha256",
        expected["guard_sha256"],
        "--expected-device-architecture",
        args.expected_device_architecture,
        "--expected-quantized-artifact-revision",
        args.expected_quantized_artifact_revision,
    ]
    for path, label in (
        (args.active_output, "active output"),
        (args.result_receipt_output, "result receipt"),
        (args.output.parent / "attempt.json", "attempt marker"),
    ):
        if os.path.lexists(path):
            raise HoldoutError(f"{label} already exists; overwrite is forbidden")
    frozen_inputs = {
        "split_manifest": _identity_file(
            args.split_root / "split-manifest.json", "split manifest"
        ),
        "policy": _identity_file(args.split_root / "policy.json", "policy"),
        "calibration_cases": _identity_file(
            args.split_root / "calibration-cases.jsonl", "calibration cases"
        ),
        "holdout_cases": _identity_file(
            args.split_root / "holdout-cases.jsonl", "holdout cases"
        ),
        "split_sums": _identity_file(
            args.split_root / "SHA256SUMS", "split SHA256SUMS"
        ),
        "freeze_receipt": _identity_file(args.freeze_receipt, "freeze receipt"),
        "source_manifest": source_identity,
        "source_cases": _identity_file(source_cases_path, "source cases"),
        "source_holdout_receipt": source_holdout["receipt"],
        "actual_receipt": actual["receipt"],
        "served_manifest": served_identity,
        "profile": actual["profile"],
        "package_manifest": actual["package_manifest"],
        "worker_binary": actual["worker"],
        "guard_receipt": guard["receipt"],
        "capture_binary": capture_identity,
        "build_receipt": build["receipt"],
        "build_cargo_lock": build["cargo_lock"],
        "build_log": build["build_log"],
        **{
            f"source_member_{name.replace('/', '_')}": _identity_file(
                path, f"source artifact {name}"
            )
            for name, path in source["tracked"].items()
        },
        **{f"actual_{name}": item for name, item in actual["lineage"].items()},
    }
    plan = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "ready_for_execute",
        "promotion_eligible": False,
        "subset": "holdout",
        "row_count": MAX_ROWS,
        "strata": {"count": 8, "rows_per_stratum": 3},
        "split_manifest_sha256": shas["split_manifest_sha256"],
        "policy_sha256": shas["policy_sha256"],
        "calibration_cases_sha256": shas["calibration_cases_sha256"],
        "holdout_cases_sha256": shas["holdout_cases_sha256"],
        "freeze_receipt_sha256": shas["freeze_receipt_sha256"],
        "freeze_receipt_path": str(args.freeze_receipt.resolve()),
        "actual_verified_receipt": actual,
        "source_artifact": {
            "path": str(args.source_artifact.resolve()),
            "manifest_sha256": source_identity["sha256"],
            "cases_sha256": source_cases_sha,
            "holdout_receipt": source_holdout,
            "identity": source["manifest"].get("identity"),
        },
        "build_receipt": build,
        "guard_receipt": guard,
        "package_tree": package_tree,
        "frozen_inputs": frozen_inputs,
        "identity": {
            **expected,
            "served_model_manifest_path": served_identity["path"],
            "source_identity": source["manifest"].get("identity"),
        },
        "execution_contract": {
            "one_process": True,
            "one_model_load": True,
            "gpu_parallelism": 1,
            "timeout_seconds": args.timeout_seconds,
            "chunk_elements": args.chunk_elements,
            "capture_binary": capture_identity,
            "device_environment": {name: str(args.device_index) for name in DEVICE_ENV},
            "guard_environment": guard["required_environment"],
            "command": command,
            "command_sha256": _command_sha(command),
        },
        "paths": {
            "split_root": str(args.split_root.resolve()),
            "source_artifact": str(args.source_artifact.resolve()),
            "active_output": str(args.active_output.resolve()),
            "attempt_marker": str((args.output.parent / "attempt.json").resolve()),
            "result_receipt": str(args.result_receipt_output.resolve()),
        },
        "frozen_bounds": receipt["derived_bounds"],
    }
    _atomic_json(args.output, plan, "preflight")
    return {
        "status": "ok",
        "preflight": str(args.output),
        "preflight_sha256": _sha(args.output, "preflight"),
    }


def _failure(
    path: Path,
    plan: dict[str, Any],
    preflight_sha: str,
    kind: str,
    detail: str,
    exit_code: int | None = None,
    *,
    stage: str = "validation",
    errno: int | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": FAILURE_SCHEMA,
        "status": "holdout_failed",
        "holdout_status": "failed",
        "holdout_evaluations_remaining": 1,
        "attempt_consumed": True,
        "retry_permitted": False,
        "partial_artifact_adopted": False,
        "failure_kind": kind,
        "stage": stage,
        "detail": detail,
        "preflight_sha256": preflight_sha,
        "split_manifest_sha256": plan["split_manifest_sha256"],
        "policy_sha256": plan["policy_sha256"],
        "calibration_cases_sha256": plan["calibration_cases_sha256"],
        "holdout_cases_sha256": plan["holdout_cases_sha256"],
        "freeze_receipt_sha256": plan["freeze_receipt_sha256"],
        "actual_verified_receipt": plan["actual_verified_receipt"],
        "identity": plan["identity"],
        "immutable": True,
    }
    if exit_code is not None:
        value["exit_code"] = exit_code
    if errno is not None:
        value["errno"] = errno
    if evidence:
        value["execution_evidence"] = evidence
    _atomic_json(path, value, "failure receipt")
    return value


def _process_census(capture_binary: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    target = str(capture_binary.resolve()).encode()
    for proc in sorted(Path("/proc").glob("[0-9]*"), key=lambda item: int(item.name)):
        try:
            raw = (proc / "cmdline").read_bytes()[: 1024 * 1024]
            arguments = [item for item in raw.split(b"\0") if item]
            exe = os.readlink(proc / "exe")
            if not arguments or arguments[0] != target and os.fsencode(exe) != target:
                continue
            pid = int(proc.name)
            result.append(
                {
                    "pid": pid,
                    "ppid": int((proc / "stat").read_text().split()[3]),
                    "pgid": os.getpgid(pid),
                    "sid": os.getsid(pid),
                    "exe": exe,
                    "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            OSError,
            ValueError,
            IndexError,
        ):
            continue
    return result


def _gpu_process_census() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for proc in sorted(Path("/proc").glob("[0-9]*"), key=lambda item: int(item.name)):
        nodes: set[str] = set()
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target == "/dev/kfd" or target.startswith("/dev/dri/renderD"):
                    nodes.add(target)
            if nodes:
                pid = int(proc.name)
                raw = (proc / "cmdline").read_bytes()[: 1024 * 1024]
                result.append(
                    {
                        "pid": pid,
                        "pgid": os.getpgid(pid),
                        "device_nodes": sorted(nodes),
                        "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            OSError,
            ValueError,
        ):
            continue
    return result


def _process_group_census(process_group: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for proc in sorted(Path("/proc").glob("[0-9]*"), key=lambda item: int(item.name)):
        try:
            pid = int(proc.name)
            if os.getpgid(pid) != process_group:
                continue
            raw = (proc / "cmdline").read_bytes()[: 1024 * 1024]
            result.append(
                {
                    "pid": pid,
                    "ppid": int((proc / "stat").read_text().split()[3]),
                    "pgid": process_group,
                    "exe": os.readlink(proc / "exe"),
                    "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            OSError,
            ValueError,
            IndexError,
        ):
            continue
    return result


def _model_file_census(pid: int, package_root: Path) -> list[str]:
    root = package_root.resolve()
    observed: set[str] = set()
    try:
        for fd in (Path("/proc") / str(pid) / "fd").iterdir():
            try:
                target = Path(os.readlink(fd))
                observed.add(target.resolve().relative_to(root).as_posix())
            except (OSError, ValueError):
                continue
        maps = (Path("/proc") / str(pid) / "maps").read_text(
            encoding="utf-8", errors="replace"
        )[: 16 * 1024 * 1024]
        for line in maps.splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            try:
                observed.add(Path(fields[5]).resolve().relative_to(root).as_posix())
            except (OSError, ValueError):
                continue
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        pass
    return sorted(observed)


def _capture_log_evidence(
    stdout_path: Path,
    stderr_path: Path,
    out_fd: int | None,
    err_fd: int | None,
    evidence: dict[str, Any],
) -> None:
    errors: list[dict[str, Any]] = []
    for label, path, descriptor in (
        ("stdout", stdout_path, out_fd),
        ("stderr", stderr_path, err_fd),
    ):
        if descriptor is None:
            continue
        try:
            os.fsync(descriptor)
            evidence[label] = _identity_file(path, f"capture {label}")
        except (OSError, HoldoutError) as error:
            errors.append(
                {
                    "stream": label,
                    "detail": str(error),
                    "errno": error.errno if isinstance(error, OSError) else None,
                }
            )
    if errors:
        evidence["log_evidence_errors"] = errors


def _validate_plan(plan: dict[str, Any]) -> None:
    if (
        plan.get("schema_version") != PREFLIGHT_SCHEMA
        or plan.get("status") != "ready_for_execute"
        or plan.get("subset") != "holdout"
        or plan.get("row_count") != MAX_ROWS
    ):
        raise HoldoutError("preflight schema/status differs")
    contract = plan.get("execution_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("one_process") is not True
        or contract.get("one_model_load") is not True
        or contract.get("gpu_parallelism") != 1
    ):
        raise HoldoutError("preflight execution contract differs")
    command = contract.get("command")
    if contract.get("command_sha256") != _command_sha(command):
        raise HoldoutError("preflight command SHA differs")
    identity = plan.get("identity", {})
    expected_env = {name: str(identity.get("device_index")) for name in DEVICE_ENV}
    if contract.get("device_environment") != expected_env:
        raise HoldoutError("preflight device environment differs")
    expected_command = [
        contract.get("capture_binary", {}).get("path"),
        "--served-model-manifest",
        identity.get("served_model_manifest_path"),
        "--split-root",
        plan.get("paths", {}).get("split_root"),
        "--source",
        plan.get("paths", {}).get("source_artifact"),
        "--cases-file",
        plan.get("source_artifact", {})
        .get("holdout_receipt", {})
        .get("cases", {})
        .get("path"),
        "--output",
        plan.get("paths", {}).get("active_output"),
        "--subset",
        "holdout",
        "--device-index",
        str(identity.get("device_index")),
        "--chunk-elements",
        str(contract.get("chunk_elements")),
        "--expected-split-manifest-sha256",
        plan.get("split_manifest_sha256"),
        "--expected-policy-sha256",
        plan.get("policy_sha256"),
        "--expected-calibration-cases-sha256",
        plan.get("calibration_cases_sha256"),
        "--expected-holdout-cases-sha256",
        plan.get("holdout_cases_sha256"),
        "--expected-served-model-manifest-sha256",
        identity.get("served_model_manifest_sha256"),
        "--expected-package-manifest-sha256",
        identity.get("package_manifest_sha256"),
        "--expected-worker-binary-sha256",
        identity.get("worker_binary_sha256"),
        "--expected-guard-sha256",
        identity.get("guard_sha256"),
        "--expected-device-architecture",
        identity.get("device_architecture"),
        "--expected-quantized-artifact-revision",
        identity.get("quantized_artifact_revision"),
    ]
    if command != expected_command:
        raise HoldoutError(
            "preflight command differs from the frozen identity projection"
        )
    if (
        command[0] != contract.get("capture_binary", {}).get("path")
        or command.count("--device-index") != 1
        or command[command.index("--device-index") + 1]
        != str(plan.get("identity", {}).get("device_index"))
    ):
        raise HoldoutError("preflight command/capture/device binding differs")


def _revalidate_frozen_plan(plan: dict[str, Any]) -> None:
    frozen_inputs = plan.get("frozen_inputs")
    required = {
        "split_manifest",
        "policy",
        "calibration_cases",
        "holdout_cases",
        "split_sums",
        "freeze_receipt",
        "source_manifest",
        "source_cases",
        "source_holdout_receipt",
        "actual_receipt",
        "served_manifest",
        "profile",
        "package_manifest",
        "worker_binary",
        "guard_receipt",
        "capture_binary",
        "build_receipt",
        "build_cargo_lock",
        "build_log",
        "actual_prepared_receipt",
        "actual_maintenance_evidence",
        "actual_executor_record",
        "source_member_manifest.json",
        "source_member_SHA256SUMS",
        "source_member_rows.jsonl",
        "source_member_vectors_hidden.f32le",
        "source_member_vectors_logits.f32le",
    }
    if not isinstance(frozen_inputs, dict) or not required.issubset(frozen_inputs):
        raise HoldoutError("preflight frozen input inventory is incomplete")
    for name, identity in frozen_inputs.items():
        _revalidate_identity(identity, f"frozen {name}")
    observed_package = _package_tree(Path(plan["package_tree"]["root"]))
    if observed_package != plan["package_tree"]:
        raise HoldoutError("full package tree path/content/topology changed")
    if (
        _actual_verified(
            Path(plan["actual_verified_receipt"]["receipt"]["path"]),
            Path(plan["identity"]["served_model_manifest_path"]),
        )
        != plan["actual_verified_receipt"]
    ):
        raise HoldoutError("formal actual-verified projection changed")
    if (
        _build_receipt(
            Path(plan["build_receipt"]["receipt"]["path"]),
            Path(plan["execution_contract"]["capture_binary"]["path"]),
        )
        != plan["build_receipt"]
    ):
        raise HoldoutError("formal capture build projection changed")
    if (
        _guard_receipt(
            Path(plan["guard_receipt"]["receipt"]["path"]),
            plan["identity"]["guard_sha256"],
        )
        != plan["guard_receipt"]
    ):
        raise HoldoutError("formal guard projection changed")
    if (
        _source_holdout_receipt(
            Path(plan["source_artifact"]["holdout_receipt"]["receipt"]["path"]),
            Path(plan["source_artifact"]["holdout_receipt"]["cases"]["path"]),
            {
                name: plan[name]
                for name in (
                    "split_manifest_sha256",
                    "policy_sha256",
                    "calibration_cases_sha256",
                    "holdout_cases_sha256",
                )
            },
        )
        != plan["source_artifact"]["holdout_receipt"]
    ):
        raise HoldoutError("formal source holdout projection changed")
    if os.path.lexists(plan["paths"]["active_output"]):
        raise HoldoutError("active output appeared before capture")


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    plan, plan_raw = _read_json(args.preflight, "preflight")
    preflight_sha = hashlib.sha256(plan_raw).hexdigest()
    try:
        attempt_path = Path(plan["paths"]["attempt_marker"])
    except (KeyError, TypeError) as error:
        raise HoldoutError("preflight attempt marker path is missing") from error
    if os.path.lexists(attempt_path):
        raise HoldoutError("attempt marker already exists; retry is forbidden")
    if (
        args.receipt_output.resolve()
        != Path(plan.get("paths", {}).get("result_receipt", "")).resolve()
    ):
        raise HoldoutError(
            "execute receipt output differs from the frozen preflight path"
        )
    _validate_plan(plan)
    marker = {
        "schema_version": ATTEMPT_SCHEMA,
        "status": "started",
        "preflight_sha256": preflight_sha,
        "started_unix": time.time(),
        "command_sha256": plan["execution_contract"]["command_sha256"],
    }
    _atomic_json(attempt_path, marker, "attempt marker")
    stdout_path = args.receipt_output.parent / "capture.stdout.log"
    stderr_path = args.receipt_output.parent / "capture.stderr.log"
    evidence: dict[str, Any] = {}
    process: subprocess.Popen[bytes] | None = None
    out_fd: int | None = None
    err_fd: int | None = None
    try:
        _revalidate_frozen_plan(plan)
        evidence["pre_spawn_process_census"] = _process_census(
            Path(plan["execution_contract"]["capture_binary"]["path"])
        )
        evidence["pre_spawn_gpu_process_census"] = _gpu_process_census()
        if evidence["pre_spawn_process_census"]:
            raise HoldoutError(
                "capture process already exists before the one-shot spawn"
            )
        args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
        out_fd = os.open(
            stdout_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444
        )
        err_fd = os.open(
            stderr_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444
        )
        env = os.environ.copy()
        env.update(plan["execution_contract"]["guard_environment"])
        env.update(plan["execution_contract"]["device_environment"])
        try:
            process = subprocess.Popen(
                plan["execution_contract"]["command"],
                stdin=subprocess.DEVNULL,
                stdout=out_fd,
                stderr=err_fd,
                env=env,
                start_new_session=True,
            )
        except OSError as error:
            _capture_log_evidence(stdout_path, stderr_path, out_fd, err_fd, evidence)
            return _failure(
                args.receipt_output,
                plan,
                preflight_sha,
                "spawn",
                f"capture could not start: {error}",
                stage="spawn",
                errno=error.errno,
                evidence=evidence,
            )
        evidence["child_pid"] = process.pid
        evidence["child_process_group"] = os.getpgid(process.pid)
        if evidence["child_process_group"] != process.pid:
            raise HoldoutError("capture child is not process-group leader")
        deadline = time.monotonic() + 0.5
        child_census: list[dict[str, Any]] = []
        while time.monotonic() < deadline and process.poll() is None:
            child_census = _process_census(
                Path(plan["execution_contract"]["capture_binary"]["path"])
            )
            if any(item["pid"] == process.pid for item in child_census):
                break
            time.sleep(0.01)
        evidence["spawn_process_census"] = child_census
        evidence["spawn_process_group_census"] = _process_group_census(process.pid)
        evidence["spawn_gpu_process_census"] = _gpu_process_census()
        observed_model_files = set(
            _model_file_census(process.pid, Path(plan["package_tree"]["root"]))
        )
        observed_gpu_pids = {
            item["pid"] for item in evidence["spawn_gpu_process_census"]
        }
        observed_group_pids = {
            item["pid"] for item in evidence["spawn_process_group_census"]
        }
        timeout_deadline = time.monotonic() + float(
            plan["execution_contract"]["timeout_seconds"]
        )
        while process.poll() is None and time.monotonic() < timeout_deadline:
            remaining = max(0.001, timeout_deadline - time.monotonic())
            try:
                process.wait(timeout=min(1.0, remaining))
            except subprocess.TimeoutExpired:
                observed_gpu_pids.update(item["pid"] for item in _gpu_process_census())
                observed_group_pids.update(
                    item["pid"] for item in _process_group_census(process.pid)
                )
                observed_model_files.update(
                    _model_file_census(process.pid, Path(plan["package_tree"]["root"]))
                )
        evidence["gpu_process_pids_observed_during_capture"] = sorted(observed_gpu_pids)
        evidence["process_group_pids_observed_during_capture"] = sorted(
            observed_group_pids
        )
        evidence["model_package_files_observed_from_proc"] = sorted(
            observed_model_files
        )
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            _capture_log_evidence(stdout_path, stderr_path, out_fd, err_fd, evidence)
            evidence["post_exit_process_census"] = _process_census(
                Path(plan["execution_contract"]["capture_binary"]["path"])
            )
            evidence["post_exit_gpu_process_census"] = _gpu_process_census()
            return _failure(
                args.receipt_output,
                plan,
                preflight_sha,
                "timeout",
                "capture exceeded the frozen timeout",
                process.returncode,
                stage="wait",
                evidence=evidence,
            )
    except Exception as error:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        error_number = error.errno if isinstance(error, OSError) else None
        _capture_log_evidence(stdout_path, stderr_path, out_fd, err_fd, evidence)
        return _failure(
            args.receipt_output,
            plan,
            preflight_sha,
            "partial",
            str(error),
            process.returncode if process is not None else None,
            stage="pre_spawn" if process is None else "census",
            errno=error_number,
            evidence=evidence,
        )
    finally:
        for descriptor in (out_fd, err_fd):
            if descriptor is not None:
                try:
                    os.fsync(descriptor)
                except OSError:
                    pass
                os.close(descriptor)
    assert process is not None
    try:
        evidence["post_exit_process_census"] = _process_census(
            Path(plan["execution_contract"]["capture_binary"]["path"])
        )
        evidence["post_exit_process_group_census"] = _process_group_census(process.pid)
        evidence["post_exit_gpu_process_census"] = _gpu_process_census()
        evidence["stdout"] = _identity_file(stdout_path, "capture stdout")
        evidence["stderr"] = _identity_file(stderr_path, "capture stderr")
    except Exception as error:
        return _failure(
            args.receipt_output,
            plan,
            preflight_sha,
            "partial",
            f"post-exit log/process evidence failed: {error}",
            process.returncode,
            stage="post_exit_evidence",
            errno=error.errno if isinstance(error, OSError) else None,
            evidence=evidence,
        )
    if (
        evidence["post_exit_process_census"]
        or evidence["post_exit_process_group_census"]
    ):
        return _failure(
            args.receipt_output,
            plan,
            preflight_sha,
            "partial",
            "capture process remains after child exit",
            process.returncode,
            stage="post_exit_census",
            evidence=evidence,
        )
    if process.returncode == 0 and (
        not any(
            item["pid"] == process.pid and item["pgid"] == process.pid
            for item in evidence["spawn_process_census"]
        )
        or process.pid not in evidence["gpu_process_pids_observed_during_capture"]
        or evidence["process_group_pids_observed_during_capture"] != [process.pid]
        or not evidence["model_package_files_observed_from_proc"]
    ):
        return _failure(
            args.receipt_output,
            plan,
            preflight_sha,
            "partial",
            "external child/process-group/model-file/GPU-process census is incomplete",
            process.returncode,
            stage="process_census",
            evidence=evidence,
        )
    if process.returncode != 0:
        kind = "oom" if process.returncode in (-signal.SIGKILL, 137) else "nonzero"
        return _failure(
            args.receipt_output,
            plan,
            preflight_sha,
            kind,
            "capture returned nonzero",
            process.returncode,
            stage="capture",
            evidence=evidence,
        )
    try:
        actual_identity = _actual_verified(
            Path(plan["actual_verified_receipt"]["receipt"]["path"]),
            Path(plan["identity"]["served_model_manifest_path"]),
        )
    except Exception as error:
        return _failure(
            args.receipt_output,
            plan,
            preflight_sha,
            "partial",
            f"actual-verified receipt unavailable: {error}",
        )
    if actual_identity != plan["actual_verified_receipt"]:
        return _failure(
            args.receipt_output,
            plan,
            preflight_sha,
            "partial",
            "actual-verified receipt changed after preflight",
        )
    active_root = Path(plan["paths"]["active_output"])
    if not active_root.is_dir() or active_root.is_symlink():
        return _failure(
            args.receipt_output,
            plan,
            preflight_sha,
            "partial",
            "capture output directory is missing",
        )
    try:
        holdout_rows = _load_rows(
            Path(plan["paths"]["split_root"]) / "holdout-cases.jsonl", "holdout"
        )
        if (
            _sha(
                Path(plan["paths"]["split_root"]) / "holdout-cases.jsonl",
                "holdout cases",
            )
            != plan["holdout_cases_sha256"]
            or _sha(
                Path(plan["paths"]["split_root"]) / "calibration-cases.jsonl",
                "calibration cases",
            )
            != plan["calibration_cases_sha256"]
            or _sha(Path(plan["paths"]["split_root"]) / "policy.json", "policy")
            != plan["policy_sha256"]
            or _sha(
                Path(plan["paths"]["split_root"]) / "split-manifest.json",
                "split manifest",
            )
            != plan["split_manifest_sha256"]
        ):
            raise HoldoutError("split identity changed after preflight")
        source_manifest_sha = _sha(
            Path(plan["paths"]["source_artifact"]) / "manifest.json",
            "source artifact manifest",
        )
        if source_manifest_sha != plan["source_artifact"]["manifest_sha256"]:
            raise HoldoutError("source artifact changed after preflight")
        source = _artifact_identity(
            Path(plan["paths"]["source_artifact"]),
            "independent_source_full",
            holdout_rows,
            "holdout",
        )
        source_cases_path = Path(source["manifest"]["cases"]["path"])
        if (
            _sha(source_cases_path, "source cases")
            != plan["source_artifact"]["cases_sha256"]
        ):
            raise HoldoutError("source cases changed after preflight")
        active = _artifact_identity(active_root, "aq4_target", holdout_rows, "holdout")
        _runtime_identity(active, plan["identity"])
        source_identity = _source_active_identity(source, active)
        rows, means = _compare(source, active, holdout_rows)
        freeze, freeze_raw = _read_json(
            Path(plan["freeze_receipt_path"]), "freeze receipt"
        )
        if (
            hashlib.sha256(freeze_raw).hexdigest() != plan["freeze_receipt_sha256"]
            or freeze.get("status") != "frozen_calibration_envelope"
            or freeze.get("holdout_status") != "not_started"
            or freeze.get("holdout_evaluations_remaining") != 1
        ):
            raise HoldoutError("freeze receipt changed or is no longer executable")
        decision, checks = _decision(means, freeze)
    except Exception as error:
        return _failure(args.receipt_output, plan, preflight_sha, "partial", str(error))
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "holdout_result",
        "decision": decision,
        "holdout_status": "complete",
        "holdout_evaluations_remaining": 0,
        "holdout_evaluation_count": 1,
        "promotion_eligible": False,
        "attempt_consumed": True,
        "retry_permitted": False,
        "preflight_sha256": preflight_sha,
        "split_manifest_sha256": plan["split_manifest_sha256"],
        "policy_sha256": plan["policy_sha256"],
        "calibration_cases_sha256": plan["calibration_cases_sha256"],
        "holdout_cases_sha256": plan["holdout_cases_sha256"],
        "freeze_receipt_sha256": plan["freeze_receipt_sha256"],
        "actual_verified_receipt": plan["actual_verified_receipt"],
        "source_artifact_manifest_sha256": source["manifest_sha256"],
        "active_artifact_manifest_sha256": active["manifest_sha256"],
        "identity": {**plan["identity"], "source_identity": source_identity},
        "execution_contract": {
            "one_process": True,
            "one_model_load": True,
            "gpu_parallelism": 1,
            "active_model_loads": active["manifest"]
            .get("runtime", {})
            .get("model_loads"),
            "external_process_evidence": evidence,
            "model_load_proof": "single externally-censused child plus active manifest state evidence",
        },
        "metrics": {
            "row_count": len(rows),
            "means": means,
            "checks": checks,
            "rows": rows,
        },
        "immutable": True,
    }
    try:
        _atomic_json(args.receipt_output, result, "holdout result")
    except (OSError, HoldoutError, ValueError, TypeError) as error:
        return _failure(
            args.receipt_output.with_name(
                f"{args.receipt_output.name}.publication-failure.json"
            ),
            plan,
            preflight_sha,
            "publication",
            f"holdout result publication failed: {error}",
            stage="result_publication",
            errno=error.errno if isinstance(error, OSError) else None,
            evidence=evidence,
        )
    return {
        "status": "ok",
        "decision": decision,
        "receipt": str(args.receipt_output),
        "receipt_sha256": _sha(args.receipt_output, "holdout result"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preflight")
    pre.add_argument("--split-root", type=Path, required=True)
    pre.add_argument("--freeze-receipt", type=Path, required=True)
    pre.add_argument("--actual-verified-receipt", type=Path, required=True)
    pre.add_argument("--source-artifact", type=Path, required=True)
    pre.add_argument("--source-holdout-receipt", type=Path, required=True)
    pre.add_argument("--capture-binary", type=Path, required=True)
    pre.add_argument("--build-receipt", type=Path, required=True)
    pre.add_argument("--guard-receipt", type=Path, required=True)
    pre.add_argument("--served-model-manifest", type=Path, required=True)
    pre.add_argument("--active-output", type=Path, required=True)
    pre.add_argument("--result-receipt-output", type=Path, required=True)
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--expected-served-model-manifest-sha256", required=True)
    pre.add_argument("--expected-package-manifest-sha256", required=True)
    pre.add_argument("--expected-worker-binary-sha256", required=True)
    pre.add_argument("--expected-capture-binary-sha256", required=True)
    pre.add_argument("--expected-build-receipt-sha256", required=True)
    pre.add_argument("--expected-guard-sha256", required=True)
    pre.add_argument("--expected-package-content-sha256", required=True)
    pre.add_argument("--expected-device-backend", required=True)
    pre.add_argument("--expected-device-name", required=True)
    pre.add_argument("--expected-device-architecture", required=True)
    pre.add_argument("--expected-device-id", required=True)
    pre.add_argument("--expected-quantized-artifact-revision", required=True)
    pre.add_argument("--device-index", type=int, default=0)
    pre.add_argument("--chunk-elements", type=int, default=65536)
    pre.add_argument("--timeout-seconds", type=float, default=3600.0)
    exe = commands.add_parser("execute")
    exe.add_argument("--preflight", type=Path, required=True)
    exe.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = _preflight(args) if args.command == "preflight" else _execute(args)
        print(json.dumps(value, ensure_ascii=True, sort_keys=True))
        return 0
    except (HoldoutError, OSError, ValueError) as error:
        print(f"AQ4 P2 fidelity holdout runner failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
