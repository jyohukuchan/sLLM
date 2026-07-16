#!/usr/bin/env python3
"""Stage a capture binary and verify the production service restoration boundary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


MAX_BINARY_BYTES = 256 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024


class RuntimePreparationError(RuntimeError):
    pass


def _promotion_module() -> Any:
    path = Path(__file__).with_name("run-qwen35-aq4-sq8-overlay-gpu-promotion.py")
    spec = importlib.util.spec_from_file_location("qwen35_sq8_promotion_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimePreparationError("cannot load the official SQ8 promotion runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _no_symlink_components(path: Path, label: str, *, include_leaf: bool = True) -> None:
    if not path.is_absolute() or path != path.absolute():
        raise RuntimePreparationError(f"{label} path must be absolute and normalized")
    current = Path(path.anchor)
    parts = path.parts[1:] if include_leaf else path.parent.parts[1:]
    for component in parts:
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if not include_leaf and current == path.parent:
                raise RuntimePreparationError(f"{label} parent is missing")
            if include_leaf and current == path:
                raise
            raise RuntimePreparationError(f"{label} path component is missing: {current}")
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimePreparationError(f"{label} has a symlink component: {current}")


def _identity_from_fd(fd: int, label: str, *, require_single_link: bool) -> dict[str, Any]:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > MAX_BINARY_BYTES:
        raise RuntimePreparationError(f"{label} must be a bounded non-empty regular file")
    if require_single_link and before.st_nlink != 1:
        raise RuntimePreparationError(f"{label} must have exactly one hard link")
    digest = hashlib.sha256()
    total = 0
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, CHUNK_BYTES)
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
    after = os.fstat(fd)
    stable = (before.st_dev, before.st_ino, before.st_size, before.st_nlink, before.st_mtime_ns)
    observed = (after.st_dev, after.st_ino, after.st_size, after.st_nlink, after.st_mtime_ns)
    if stable != observed or total != before.st_size:
        raise RuntimePreparationError(f"{label} changed during hashing")
    return {
        "sha256": digest.hexdigest(), "bytes": total, "device": before.st_dev,
        "inode": before.st_ino, "nlink": before.st_nlink,
        "mode": f"{stat.S_IMODE(before.st_mode):04o}",
    }


def binary_identity(path: Path, label: str, *, require_single_link: bool = True) -> dict[str, Any]:
    _no_symlink_components(path, label)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        return _identity_from_fd(fd, label, require_single_link=require_single_link)
    finally:
        os.close(fd)


def _create_json(path: Path, value: Any) -> None:
    _no_symlink_components(path, "output", include_leaf=False)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o444)
    try:
        raw = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2).encode("ascii") + b"\n"
        written = 0
        while written < len(raw):
            written += os.write(fd, raw[written:])
        os.fsync(fd)
    finally:
        os.close(fd)


def stage_binary(source: Path, output: Path, receipt: Path) -> dict[str, Any]:
    if os.path.lexists(output) or os.path.lexists(receipt):
        raise RuntimePreparationError("staged binary and receipt must be create-new")
    _no_symlink_components(source, "source binary")
    _no_symlink_components(output, "staged binary", include_leaf=False)
    source_flags = os.O_RDONLY | os.O_CLOEXEC
    output_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
        output_flags |= os.O_NOFOLLOW
    source_fd = os.open(source, source_flags)
    output_fd = -1
    try:
        source_identity = _identity_from_fd(source_fd, "source binary", require_single_link=False)
        output_fd = os.open(output, output_flags, 0o555)
        os.lseek(source_fd, 0, os.SEEK_SET)
        copied = 0
        while True:
            chunk = os.read(source_fd, CHUNK_BYTES)
            if not chunk:
                break
            offset = 0
            while offset < len(chunk):
                offset += os.write(output_fd, chunk[offset:])
            copied += len(chunk)
        os.fchmod(output_fd, 0o555)
        os.fsync(output_fd)
        if copied != source_identity["bytes"]:
            raise RuntimePreparationError("staged binary size differs during copy")
    finally:
        if output_fd >= 0:
            os.close(output_fd)
        os.close(source_fd)
    directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    staged = binary_identity(output, "staged binary")
    if staged["sha256"] != source_identity["sha256"] or staged["bytes"] != source_identity["bytes"] or staged["mode"] != "0555":
        raise RuntimePreparationError("staged binary identity differs")
    value = {
        "schema_version": "ullm.aq4_fidelity_capture_staged_binary.v1", "status": "ready",
        "source": {"path": str(source), **source_identity},
        "staged": {"path": str(output), **staged},
        "execution_contract": {"create_new": True, "o_nofollow": True, "fstat": True, "nlink": 1, "child_self_validation_required": True},
    }
    _create_json(receipt, value)
    return value


def validate_binary(path: Path, expected_sha256: str, expected_bytes: int) -> dict[str, Any]:
    identity = binary_identity(path, "staged binary")
    if identity["sha256"] != expected_sha256 or identity["bytes"] != expected_bytes or identity["mode"] != "0555":
        raise RuntimePreparationError("staged binary differs from the execution contract")
    return identity


def _validate_ready(service: dict[str, Any], owners: dict[str, Any], *, expected_nrestarts: int) -> None:
    main_pid = service.get("main_pid")
    worker_pid = service.get("worker_pid")
    if service.get("active") is not True or service.get("running") is not True or service.get("healthy") is not True:
        raise RuntimePreparationError("service is not active/running/healthy")
    if type(main_pid) is not int or main_pid <= 0 or type(worker_pid) is not int or worker_pid <= 0:
        raise RuntimePreparationError("service main/worker PID differs")
    if service.get("nrestarts") != expected_nrestarts or service.get("lock_owned") is not True or service.get("lock_holders") != [main_pid]:
        raise RuntimePreparationError("service restart/lock identity differs")
    for name in ("worker_pids", "amd_pids", "kfd_pids"):
        if owners.get(name) != [worker_pid]:
            raise RuntimePreparationError(f"{name} does not exactly match the service worker")


def snapshot(served_model: Path, output: Path) -> dict[str, Any]:
    promotion = _promotion_module()
    served = promotion.read_object(served_model, "served model")
    readiness = promotion.validate_readiness_contract(served.get("promotion", {}).get("readiness"))
    service = promotion.default_service_snapshot(readiness)
    owners = promotion.default_owner_snapshot()
    _validate_ready(service, owners, expected_nrestarts=service["nrestarts"])
    value = {"schema_version": "ullm.aq4_fidelity_service_snapshot.v1", "status": "ready", "service": service, "owners": owners}
    _create_json(output, value)
    return value


def wait_restored(before_path: Path, served_model: Path, output: Path) -> dict[str, Any]:
    promotion = _promotion_module()
    before = json.loads(before_path.read_text(encoding="utf-8"))
    if not isinstance(before, dict) or before.get("schema_version") != "ullm.aq4_fidelity_service_snapshot.v1":
        raise RuntimePreparationError("pre-service snapshot schema differs")
    served = promotion.read_object(served_model, "served model")
    readiness = promotion.validate_readiness_contract(served.get("promotion", {}).get("readiness"))
    restored = promotion.poll_restored(promotion.default_dependencies(), before["service"], readiness)
    service = promotion.default_service_snapshot(readiness)
    owners = promotion.default_owner_snapshot()
    _validate_ready(service, owners, expected_nrestarts=before["service"]["nrestarts"])
    if service["main_pid"] == before["service"]["main_pid"] or service["worker_pid"] == before["service"]["worker_pid"]:
        raise RuntimePreparationError("restored service did not advance main/worker PID epoch")
    value = {"schema_version": "ullm.aq4_fidelity_service_restore.v1", "status": "passed", "poll": restored, "service": service, "owners": owners}
    _create_json(output, value)
    return value


def _artifact_hashes(root: Path) -> dict[str, str]:
    expected = {"SHA256SUMS", "manifest.json", "rows.jsonl", "vectors/hidden.f32le", "vectors/logits.f32le"}
    observed = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if observed != expected:
        raise RuntimePreparationError("published target file set differs")
    return {name: binary_identity(root / name, f"target {name}")["sha256"] for name in sorted(expected)}


def validate_target_cli(artifact: Path) -> dict[str, Any]:
    before = _artifact_hashes(artifact)
    validator = Path(__file__).with_name("validate-qwen35-aq4-p2-full-calibration.py")
    completed = subprocess.run(
        [sys.executable, str(validator), "--artifact", str(artifact)],
        check=False, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=60,
    )
    if completed.returncode != 0 or completed.stderr:
        raise RuntimePreparationError(f"strict target validator failed: {completed.stderr.strip()}")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimePreparationError("strict target validator output differs") from error
    if report.get("status") != "valid" or report.get("row_count") != 24 or report.get("nonfinite_rows") != 0:
        raise RuntimePreparationError("strict target validator report differs")
    after = _artifact_hashes(artifact)
    if before != after:
        raise RuntimePreparationError("strict target validator modified the published target")
    return {"report": report, "artifact_hashes": before, "validator_modified_artifact": False, "command": [sys.executable, str(validator), "--artifact", str(artifact)]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("stage-binary")
    stage.add_argument("--source", type=Path, required=True); stage.add_argument("--output", type=Path, required=True); stage.add_argument("--receipt", type=Path, required=True)
    validate = sub.add_parser("validate-binary")
    validate.add_argument("--path", type=Path, required=True); validate.add_argument("--expected-sha256", required=True); validate.add_argument("--expected-bytes", type=int, required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--served-model", type=Path, required=True); snap.add_argument("--output", type=Path, required=True)
    restore = sub.add_parser("wait-restored")
    restore.add_argument("--before", type=Path, required=True); restore.add_argument("--served-model", type=Path, required=True); restore.add_argument("--output", type=Path, required=True)
    target = sub.add_parser("validate-target")
    target.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "stage-binary": result = stage_binary(args.source, args.output, args.receipt)
        elif args.command == "validate-binary": result = validate_binary(args.path, args.expected_sha256, args.expected_bytes)
        elif args.command == "snapshot": result = snapshot(args.served_model, args.output)
        elif args.command == "wait-restored": result = wait_restored(args.before, args.served_model, args.output)
        else: result = validate_target_cli(args.artifact)
        print(json.dumps({"status": "ok", "command": args.command, "result": result}, sort_keys=True))
        return 0
    except (RuntimePreparationError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"AQ4 calibration runtime preparation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
