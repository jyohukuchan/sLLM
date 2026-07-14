#!/usr/bin/env python3
"""Run the representative AQ4 P2 full-model target profile through one resident driver.

The driver protocol is deliberately tiny and hash-only: one child process announces one model
load, then receives case/run commands.  A real GPU driver can implement this protocol later; the
planner and fake-driver tests are CPU-only and never touch a service or device.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import select
import socket
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

MAX_JSON_BYTES = 64 * 1024 * 1024
CASE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SCHEMA = "ullm.aq4_p2_resident_batch.v1"
DRIVER_SCHEMA = "ullm.aq4_p2_resident_driver.v2"
ONE_CASE_BUNDLE_SCHEMA = "ullm.aq4_p2_resident_smoke_binding_bundle.v3"
ONE_CASE_ROOT_CONTRACT = "ullm.aq4_p2_resident_smoke_bundle_root.v4"
TRUSTED_ONE_CASE_ID = "p2-representative-full_model-cold_prefill-cold_batched-n128-m128-r9700-rdna4-aq4_0_target"
TRUSTED_ONE_CASE_SHA256 = "d83a420476bde889c7c8014d7982fd52e0f61ab09b888f66415d0ac9fb443ae7"
TRUSTED_OFFICIAL_CASE_SHA256 = "bf18481edf53a70efe840243f735c12cb949d127db024b38b89b5974ad77eb5a"
TRUSTED_SOURCE_MANIFEST_SHA256 = "1fa264c6a7a485e36b1119ca13732ad88e052a8bd502c2addacdff14ff41cbea"
ONE_CASE_MEMBER_CONTRACT = {
    "SUPERSEDED-0fd7993.json": (0o444, "historical_non_executable_bundle_record"),
    "case-binding.json": (0o444, "runtime_bound_case"),
    "dry-run.json": (0o444, "resident_batch_dry_run"),
    "fake-ready.json": (0o444, "synthetic_ready_event"),
    "fixture-index.json": (0o444, "fixture_index"),
    "fixture.json": (0o444, "fixture"),
    "identity.json": (0o444, "resident_identity"),
    "launch-command.json": (0o444, "exact_resident_launch_command"),
    "official-case.json": (0o444, "trusted_official_expansion_case"),
    "package-manifest.json": (0o444, "package_manifest_snapshot"),
    "policy.json": (0o444, "threshold_policy"),
    "preflight.json": (0o444, "synthetic_preflight"),
    "resident-driver": (0o555, "detached_resident_driver"),
    "runner-dry-run-evidence.json": (0o444, "trusted_runner_subprocess_evidence"),
    "served-model.json": (0o444, "served_model_snapshot"),
    "trust-roots.json": (0o444, "independent_trust_roots"),
    "trusted-runner.py": (0o444, "trusted_one_case_smoke_runner"),
}
ONE_CASE_BUNDLE_FILE_MEMBERS = set(ONE_CASE_MEMBER_CONTRACT) - {"dry-run.json", "runner-dry-run-evidence.json"}
ONE_CASE_ROOT_MEMBERS = set(ONE_CASE_MEMBER_CONTRACT) | {"bundle.json", "SHA256SUMS"}
WARMUP_RUNS = 2
MEASURED_RUNS = 10
READY_IDENTITY_KEYS = {
    "binary_sha256",
    "build_git_commit",
    "protocol",
    "worker_binary_sha256",
    "package_manifest_sha256",
    "package_content_sha256",
    "served_model_manifest_sha256",
    "model_id",
    "model_revision",
    "format_id",
    "implementation_id",
    "runtime_device",
    "guard_set_sha256",
}
RUNTIME_DEVICE_KEYS = {
    "runtime_device_index",
    "device_id",
    "backend",
    "name",
    "architecture",
}
DEFAULT_LOCK_PATH = Path("/run/ullm/r9700.lock")


class BatchError(ValueError):
    pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in items:
        if key in value:
            raise BatchError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mode,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )


def _require_absolute_nonsymlink_path(path: Path, label: str) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise BatchError(f"{label} path must be absolute without parent traversal")
    for parent in reversed(path.parents):
        try:
            if stat.S_ISLNK(os.lstat(parent).st_mode):
                raise BatchError(f"{label} path has a symlink parent")
        except FileNotFoundError:
            continue


def read_regular(path: Path, label: str, maximum: int | None = None, *, absolute: bool = False, collect: bool = True) -> tuple[bytes, str, os.stat_result]:
    if absolute:
        _require_absolute_nonsymlink_path(path, label)
    try:
        before = os.lstat(path)
    except OSError as error:
        raise BatchError(f"{label} metadata failed: {error}") from error
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise BatchError(f"{label} must be a single-link regular file")
    if maximum is not None and before.st_size > maximum:
        raise BatchError(f"{label} exceeds the byte bound")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BatchError(f"{label} open failed: {error}") from error
    try:
        opened = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(opened):
            raise BatchError(f"{label} changed while opening")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if maximum is not None and total > maximum:
                raise BatchError(f"{label} exceeds the byte bound")
            digest.update(chunk)
            if collect:
                chunks.append(chunk)
        after = os.lstat(path)
        if _file_identity(before) != _file_identity(after) or _file_identity(before) != _file_identity(os.fstat(descriptor)):
            raise BatchError(f"{label} changed while reading")
        return b"".join(chunks), digest.hexdigest(), before
    finally:
        os.close(descriptor)


def load(path: Path, label: str) -> dict[str, Any]:
    data, _, _ = read_regular(path, label, MAX_JSON_BYTES)
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(BatchError(f"non-finite JSON: {item}")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BatchError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise BatchError(f"{label} root must be an object")
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path, label: str, *, absolute: bool = False) -> str:
    return read_regular(path, label, absolute=absolute, collect=False)[1]


def normalize_fixture_paths(fixture_index: dict[str, Any]) -> None:
    entries = fixture_index.get("cases")
    if not isinstance(entries, list):
        raise BatchError("fixture index cases are missing")
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("fixture_path"), str):
            raise BatchError("fixture index contains an invalid fixture path")
        raw = Path(entry["fixture_path"])
        _require_absolute_nonsymlink_path(raw, "fixture")
        sha_file(raw, "fixture", absolute=True)
        try:
            resolved = raw.resolve(strict=True)
        except OSError as error:
            raise BatchError(f"fixture path resolution failed: {error}") from error
        if resolved != raw:
            raise BatchError("fixture path must already be resolved")
        entry["fixture_path"] = str(resolved)


def validate_driver_argv_schema(command: list[str], identity: dict[str, Any], *, expected_argv: list[str] | None = None) -> None:
    if not command or not all(isinstance(item, str) and item for item in command):
        raise BatchError("driver command is empty or invalid")
    bound = identity.get("resident_driver_identity")
    runtime_device = bound.get("runtime_device") if isinstance(bound, dict) else None
    expected_device_index = runtime_device.get("runtime_device_index") if isinstance(runtime_device, dict) else None
    expected_build_commit = bound.get("build_git_commit") if isinstance(bound, dict) else None
    if (
        len(command) != 7
        or command[1] != "--served-model-manifest"
        or not Path(command[2]).is_absolute()
        or ".." in Path(command[2]).parts
        or command[3] != "--device-index"
        or command[4] != str(expected_device_index)
        or command[5] != "--build-git-commit"
        or command[6] != expected_build_commit
    ):
        raise BatchError("resident driver argv does not match the exact production schema")
    if expected_argv is not None and command != expected_argv:
        raise BatchError("resident driver argv differs from the one-case launch binding")


def validate_driver_command(command: list[str], identity: dict[str, Any], *, expected_argv: list[str] | None = None) -> dict[str, Any]:
    validate_driver_argv_schema(command, identity, expected_argv=expected_argv)
    path = Path(command[0])
    _, digest, metadata = read_regular(path, "resident driver executable", absolute=True, collect=False)
    if metadata.st_mode & 0o111 == 0:
        raise BatchError("resident driver executable is not executable")
    bound = identity.get("resident_driver_identity")
    if not isinstance(bound, dict) or bound.get("binary_sha256") != digest:
        raise BatchError("resident driver executable SHA differs from bound identity")
    return {
        "path": str(path),
        "sha256": digest,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "nlink": metadata.st_nlink,
    }


@contextmanager
def acquire_device_lock(path: Path, run_id: str, driver: dict[str, Any]) -> Iterator[dict[str, Any]]:
    _require_absolute_nonsymlink_path(path, "device lock")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise BatchError(f"device lock open failed: {error}") from error
    owner = {
        "schema_version": "ullm.aq4_p2_device_lock_owner.v1",
        "path": str(path),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "run_id": run_id,
        "acquired_unix_ns": time.time_ns(),
        "driver": driver,
    }
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise BatchError("device lock must be a single-link regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BatchError(f"device lock is already owned: {path}") from error
        payload = canonical(owner) + b"\n"
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise BatchError("device lock owner metadata write failed")
            offset += written
        os.fsync(descriptor)
        yield owner
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def case_hash(case: dict[str, Any]) -> str:
    value = json.loads(json.dumps(case))
    value["case_sha256"] = None
    return sha_bytes(canonical(value))


def atomic_write(path: Path, value: Any) -> None:
    if os.path.lexists(path):
        raise BatchError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    try:
        with temporary.open("x", encoding="utf-8") as target:
            json.dump(value, target, ensure_ascii=True, sort_keys=True, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise BatchError(f"refusing to overwrite {path}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def select_target_cases(expanded: dict[str, Any], fixture_index: dict[str, Any], *, one_case_smoke: bool = False) -> list[dict[str, Any]]:
    if expanded.get("schema_version") != "ullm.aq4_production_p2_expanded.v2":
        raise BatchError("expanded manifest schema differs")
    if fixture_index.get("schema_version") != "ullm.aq4_p2_fixture_index.v1":
        raise BatchError("fixture index schema differs")
    cases = expanded.get("cases")
    if not isinstance(cases, list):
        raise BatchError("expanded cases are missing")
    selected = [
        case for case in cases
        if isinstance(case, dict)
        and case.get("stage_id") == "representative"
        and case.get("scope") == "full_model"
        and case.get("phase") == "cold_prefill"
        and case.get("device", {}).get("device_id") == "r9700-rdna4"
        and case.get("control_id") == "aq4_0_target"
    ]
    expected_cases = 1 if one_case_smoke else 84
    label = "one-case smoke" if one_case_smoke else "representative full_model target profile"
    if len(selected) != expected_cases:
        raise BatchError(f"{label} must contain exactly {expected_cases} target cases, got {len(selected)}")
    selected_ids = [case.get("case_id") for case in selected]
    if len(set(selected_ids)) != len(selected_ids):
        raise BatchError("representative target profile contains duplicate case IDs")
    index_cases = fixture_index.get("cases")
    if not isinstance(index_cases, list):
        raise BatchError("fixture index cases are missing")
    by_id: dict[str, dict[str, Any]] = {}
    for entry in index_cases:
        if not isinstance(entry, dict) or not isinstance(entry.get("case_id"), str) or not entry["case_id"] or entry["case_id"] in by_id:
            raise BatchError("fixture index contains invalid or duplicate case IDs")
        by_id[entry["case_id"]] = entry
    if len(by_id) != fixture_index.get("case_count"):
        raise BatchError("fixture index case coverage differs")
    for case in selected:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or CASE_ID_RE.fullmatch(case_id) is None or case.get("case_sha256") != case_hash(case):
            raise BatchError(f"selected case identity differs: {case_id}")
        entry = by_id.get(case_id)
        if not isinstance(entry, dict) or entry.get("case_sha256") != case.get("case_sha256") or entry.get("prompt_tokens") != case.get("prompt_tokens") or entry.get("context_tokens") != case.get("context_tokens") or entry.get("generated_tokens") != case.get("generated_tokens"):
            raise BatchError(f"fixture index does not bind selected case: {case_id}")
        fixture_path = Path(entry.get("fixture_path", ""))
        if sha_file(fixture_path, "fixture") != entry.get("fixture_sha256"):
            raise BatchError(f"fixture hash differs: {case_id}")
    return sorted(selected, key=lambda case: case["case_id"])


def _identity_self_sha256(identity: dict[str, Any]) -> str:
    value = json.loads(json.dumps(identity))
    value.pop("_path", None)
    value.pop("_sha256", None)
    value["identity_sha256"] = None
    return sha_bytes(canonical(value))


def _run_fake_ready_handshake(path: Path, timeout: float) -> dict[str, Any]:
    child = (
        "import os,sys\n"
        "p=sys.argv[1]\n"
        "f=os.open(p,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_CLOEXEC',0))\n"
        "try:\n"
        " d=os.read(f,67108865)\n"
        " if len(d)>67108864 or os.read(f,1): raise SystemExit(91)\n"
        "finally: os.close(f)\n"
        "sys.stdout.buffer.write(d)\n"
        "sys.stdout.buffer.flush()\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", child, str(path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0 or completed.stderr:
        raise BatchError("one-case smoke fake-ready subprocess handshake failed")
    try:
        value = json.loads(
            completed.stdout.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(BatchError(f"non-finite fake-ready number: {item}")),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BatchError(f"invalid one-case smoke fake-ready subprocess response: {error}") from error
    if not isinstance(value, dict):
        raise BatchError("one-case smoke fake-ready subprocess response is not an object")
    return value


def _run_bundle_validator(path: Path, expected_sha256: str, root: Path, timeout: float) -> dict[str, Any]:
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise BatchError("trusted bundle validator expected SHA is invalid")
    _require_absolute_nonsymlink_path(path, "trusted bundle validator")
    source_raw, source_sha, source_before = read_regular(path, "trusted bundle validator", MAX_JSON_BYTES, absolute=True)
    if source_sha != expected_sha256:
        raise BatchError("trusted bundle validator source differs from expected SHA")
    if not source_raw.startswith(b"#!") and path.suffix != ".py":
        raise BatchError("trusted bundle validator is not a Python source file")
    completed = subprocess.run(
        [sys.executable, str(path), "validate", "--bundle", str(root)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0 or completed.stderr:
        raise BatchError("trusted bundle validator subprocess rejected the bundle")
    if _file_identity(source_before) != _file_identity(os.lstat(path)) or sha_file(path, "trusted bundle validator", absolute=True) != source_sha:
        raise BatchError("trusted bundle validator source changed during subprocess validation")
    try:
        report = json.loads(completed.stdout.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BatchError(f"trusted bundle validator report is invalid: {error}") from error
    if (
        not isinstance(report, dict)
        or set(report) != {"status", "promotion", "run_id"}
        or report.get("status") != "prepared_not_executed"
        or report.get("promotion") is not False
        or not isinstance(report.get("run_id"), str)
        or not report["run_id"]
    ):
        raise BatchError("trusted bundle validator report fields differ")
    return {
        "subprocess_count": 1,
        "source": {"path": str(path), "sha256": source_sha},
        "stdout_sha256": sha_bytes(completed.stdout),
        "report_sha256": sha_bytes(canonical(report)),
        "report": report,
    }


def validate_one_case_smoke_bundle(args: argparse.Namespace, expanded: dict[str, Any], fixture_index: dict[str, Any], identity: dict[str, Any], preflight: dict[str, Any], policy: dict[str, Any], cases: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.bundle_root is None:
        raise BatchError("--bundle-root is required with --one-case-smoke")
    _require_absolute_nonsymlink_path(args.bundle_root, "one-case smoke bundle root")
    try:
        root = args.bundle_root.resolve(strict=True)
    except OSError as error:
        raise BatchError(f"one-case smoke bundle root resolution failed: {error}") from error
    root_metadata = os.lstat(root)
    if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
        raise BatchError("one-case smoke bundle root must be a non-symlink directory")
    names = {entry.name for entry in root.iterdir()}
    if names != ONE_CASE_ROOT_MEMBERS:
        raise BatchError("one-case smoke bundle root exact member coverage differs")
    expected_paths = {
        "expanded": root / "case-binding.json",
        "fixture_index": root / "fixture-index.json",
        "identity": root / "identity.json",
        "preflight": root / "preflight.json",
        "policy": root / "policy.json",
    }
    for name, expected in expected_paths.items():
        supplied = getattr(args, name).resolve(strict=True)
        if supplied != expected:
            raise BatchError(f"one-case smoke {name} is not the bundle root v4 member")
    bundle_path = root / "bundle.json"
    fake_ready_path = root / "fake-ready.json"
    initial: dict[str, tuple[int, ...]] = {}
    member_inventory: dict[str, dict[str, Any]] = {}
    for name in sorted(ONE_CASE_ROOT_MEMBERS):
        metadata = os.lstat(root / name)
        expected_mode = 0o444 if name in {"bundle.json", "SHA256SUMS"} else ONE_CASE_MEMBER_CONTRACT[name][0]
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != expected_mode:
            raise BatchError(f"one-case smoke bundle member type/link/mode differs: {name}")
        initial[name] = _file_identity(metadata)
        role = {"bundle.json": "bundle_manifest", "SHA256SUMS": "sha256_manifest"}[name] if name in {"bundle.json", "SHA256SUMS"} else ONE_CASE_MEMBER_CONTRACT[name][1]
        member_inventory[name] = {"path": str(root / name), "sha256": sha_file(root / name, f"one-case smoke {name}"), "role": role, "type": "regular_file", "nlink": metadata.st_nlink, "mode": f"{expected_mode:04o}"}
    bundle = load(bundle_path, "one-case smoke bundle")
    if bundle.get("schema_version") != ONE_CASE_BUNDLE_SCHEMA or bundle.get("status") != "prepared_not_executed" or bundle.get("promotion") is not False:
        raise BatchError("one-case smoke bundle v3 status/promotion differs")
    if bundle.get("canonical_root") != str(root):
        raise BatchError("one-case smoke bundle/root identity differs")
    if len(cases) != 1:
        raise BatchError("one-case smoke internal case count differs")
    case = cases[0]
    if case.get("case_id") != TRUSTED_ONE_CASE_ID or case.get("case_sha256") != TRUSTED_ONE_CASE_SHA256:
        raise BatchError("one-case smoke trusted case ID/hash differs")
    bindings = bundle.get("bindings")
    files = bundle.get("files")
    if not isinstance(files, dict) or set(files) != ONE_CASE_BUNDLE_FILE_MEMBERS:
        raise BatchError("one-case smoke bundle file role/path coverage differs")
    for name in sorted(ONE_CASE_BUNDLE_FILE_MEMBERS):
        mode, role = ONE_CASE_MEMBER_CONTRACT[name]
        record = files.get(name)
        if not isinstance(record, dict) or set(record) != {"mode", "role", "sha256"} or record.get("mode") != f"{mode:04o}" or record.get("role") != role or not isinstance(record.get("sha256"), str) or SHA256_RE.fullmatch(record["sha256"]) is None:
            raise BatchError(f"one-case smoke bundle file contract differs: {name}")
        if sha_file(root / name, f"one-case smoke {name}") != record["sha256"]:
            raise BatchError(f"one-case smoke bundle file SHA differs: {name}")
    sums_raw, _, _ = read_regular(root / "SHA256SUMS", "one-case smoke SHA256SUMS", MAX_JSON_BYTES)
    try:
        sum_lines = sums_raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise BatchError("one-case smoke SHA256SUMS is not ASCII") from error
    expected_sum_names = set(ONE_CASE_MEMBER_CONTRACT) | {"bundle.json"}
    observed_sums: dict[str, str] = {}
    for line in sum_lines:
        if line.count("  ") != 1:
            raise BatchError("one-case smoke SHA256SUMS syntax differs")
        digest, name = line.split("  ", 1)
        if name in observed_sums or name not in expected_sum_names or SHA256_RE.fullmatch(digest) is None:
            raise BatchError("one-case smoke SHA256SUMS coverage differs")
        observed_sums[name] = digest
    if set(observed_sums) != expected_sum_names:
        raise BatchError("one-case smoke SHA256SUMS exact coverage differs")
    for name, digest in observed_sums.items():
        if sha_file(root / name, f"one-case smoke SHA256SUMS {name}") != digest:
            raise BatchError(f"one-case smoke SHA256SUMS differs: {name}")
    case_binding_sha = sha_file(args.expanded, "one-case smoke case binding")
    expected_binding_keys = {"case_binding_sha256", "case_sha256", "fixture_sha256", "guard_set_sha256", "identity_file_sha256", "identity_self_sha256", "official_case_sha256", "package_content_sha256", "package_manifest_sha256", "policy_sha256", "preflight_sha256", "served_model_manifest_sha256", "worker_binary_sha256"}
    if not isinstance(bindings, dict) or set(bindings) != expected_binding_keys or bindings.get("case_binding_sha256") != case_binding_sha or bindings.get("case_sha256") != TRUSTED_ONE_CASE_SHA256 or bindings.get("official_case_sha256") != TRUSTED_OFFICIAL_CASE_SHA256:
        raise BatchError("one-case smoke bundle case binding/hash differs")
    if files["case-binding.json"]["sha256"] != case_binding_sha:
        raise BatchError("one-case smoke bundle file binding differs")
    if expanded.get("status") != "bound_one_case_smoke" or expanded.get("case_count") != 1 or expanded.get("canonical_case_sha256") != sha_bytes(canonical(cases)) or expanded.get("source_manifest_sha256") != TRUSTED_SOURCE_MANIFEST_SHA256 or expanded.get("official_case_sha256") != TRUSTED_OFFICIAL_CASE_SHA256:
        raise BatchError("one-case smoke case-binding root differs")
    official = load(root / "official-case.json", "one-case smoke official case")
    official_case = official.get("case")
    if official.get("schema_version") != "ullm.aq4_p2_official_case.v1" or official.get("manifest_sha256") != TRUSTED_SOURCE_MANIFEST_SHA256 or not isinstance(official_case, dict) or official_case.get("case_id") != TRUSTED_ONE_CASE_ID or official_case.get("case_sha256") != TRUSTED_OFFICIAL_CASE_SHA256 or case_hash(official_case) != TRUSTED_OFFICIAL_CASE_SHA256:
        raise BatchError("one-case smoke trusted official case differs")
    index_cases = fixture_index.get("cases")
    fixture_entry = index_cases[0] if isinstance(index_cases, list) and len(index_cases) == 1 else None
    fixture_sha = sha_file(root / "fixture.json", "one-case smoke fixture")
    if fixture_index.get("subset") != "resident_one_case_smoke" or fixture_index.get("case_count") != 1 or fixture_index.get("expanded_manifest_sha256") != case_binding_sha or fixture_index.get("served_model_manifest_sha256") != identity.get("hash_binding", {}).get("served_model_manifest_sha256") or not isinstance(fixture_entry, dict) or fixture_entry.get("case_id") != TRUSTED_ONE_CASE_ID or fixture_entry.get("case_sha256") != TRUSTED_ONE_CASE_SHA256 or Path(fixture_entry.get("fixture_path", "")) != root / "fixture.json" or fixture_entry.get("fixture_sha256") != fixture_sha or bindings.get("fixture_sha256") != fixture_sha:
        raise BatchError("one-case smoke fixture index binding differs")
    identity_file_sha = sha_file(root / "identity.json", "one-case smoke identity")
    identity_self_sha = _identity_self_sha256(identity)
    if identity.get("schema_version") != "ullm.aq4_production_p2_identity.v2" or identity.get("status") != "bound" or identity.get("expanded_manifest_sha256") != case_binding_sha or identity.get("hash_binding", {}).get("bound_case_manifest_sha256") != case_binding_sha or identity.get("identity_sha256") != identity_self_sha or bindings.get("identity_file_sha256") != identity_file_sha or bindings.get("identity_self_sha256") != identity_self_sha:
        raise BatchError("one-case smoke identity case binding differs")
    if bindings.get("preflight_sha256") != sha_file(root / "preflight.json", "one-case smoke preflight") or not isinstance(preflight.get("gpu_process_snapshot"), list):
        raise BatchError("one-case smoke preflight binding differs")
    if bindings.get("policy_sha256") != sha_file(root / "policy.json", "one-case smoke policy") or policy.get("schema_version") != "ullm.aq4_production_p2_threshold_policy.v1" or policy.get("status") != "bound":
        raise BatchError("one-case smoke policy binding differs")
    resident_identity = identity.get("resident_driver_identity", {})
    hash_binding = identity.get("hash_binding", {})
    for field in ("guard_set_sha256", "package_content_sha256", "package_manifest_sha256", "served_model_manifest_sha256", "worker_binary_sha256"):
        expected = resident_identity.get(field)
        if bindings.get(field) != expected or (field != "guard_set_sha256" and hash_binding.get(field) != expected):
            raise BatchError(f"one-case smoke {field} binding differs")
    if bindings.get("served_model_manifest_sha256") != files["served-model.json"]["sha256"] or bindings.get("package_manifest_sha256") != files["package-manifest.json"]["sha256"]:
        raise BatchError("one-case smoke served/package snapshot binding differs")
    expected_binary_sha256 = identity.get("resident_driver_identity", {}).get("binary_sha256")
    if not isinstance(expected_binary_sha256, str) or SHA256_RE.fullmatch(expected_binary_sha256) is None or files["resident-driver"]["sha256"] != expected_binary_sha256:
        raise BatchError("one-case smoke resident binary binding is invalid")
    launch = load(root / "launch-command.json", "one-case smoke launch command")
    expected_driver_argv = launch.get("resident_driver_argv")
    if (
        launch.get("schema_version") != "ullm.aq4_p2_resident_launch_command.v1"
        or not isinstance(expected_driver_argv, list)
        or expected_driver_argv[0:1] != [str(root / "resident-driver")]
    ):
        raise BatchError("one-case smoke resident driver argv binding is invalid")
    validate_driver_argv_schema(expected_driver_argv, identity)
    fake_ready = _run_fake_ready_handshake(fake_ready_path, args.timeout)
    session_id, driver_identity = validate_ready(fake_ready, identity, cases, expected_binary_sha256)
    prepared_plan = load(root / "dry-run.json", "one-case smoke prepared dry-run")
    prepared_validation = prepared_plan.get("validation")
    prepared_baseline = prepared_plan.get("baseline_identity")
    if (
        prepared_plan.get("case_count") != 1
        or prepared_plan.get("transaction_count") != WARMUP_RUNS + MEASURED_RUNS
        or prepared_plan.get("smoke_only") is not True
        or prepared_plan.get("promotion_eligible") is not False
        or not isinstance(prepared_validation, dict)
        or prepared_validation.get("fake_ready") != {"path": str(fake_ready_path), "sha256": files["fake-ready.json"]["sha256"]}
        or prepared_validation.get("resident_session_id") != session_id
        or prepared_validation.get("driver_identity") != driver_identity
        or not isinstance(prepared_baseline, dict)
        or prepared_baseline.get("identity_file") != {"path": str(root / "identity.json"), "sha256": identity_file_sha}
    ):
        raise BatchError("one-case smoke prepared dry-run identity/handshake binding differs")
    prepared_evidence = load(root / "runner-dry-run-evidence.json", "one-case smoke prepared runner evidence")
    if (
        prepared_evidence.get("schema_version") != "ullm.aq4_p2_resident_runner_subprocess_evidence.v1"
        or prepared_evidence.get("runner_subprocess_count") != 1
        or prepared_evidence.get("runner_source_sha256") != files["trusted-runner.py"]["sha256"]
        or prepared_evidence.get("plan") != {"path": str(root / "dry-run.json"), "sha256": observed_sums["dry-run.json"]}
    ):
        raise BatchError("one-case smoke prepared runner evidence binding differs")
    validator_path = args.trusted_validator
    validator = _run_bundle_validator(validator_path, args.trusted_validator_sha256, root, args.timeout)
    if {entry.name for entry in root.iterdir()} != ONE_CASE_ROOT_MEMBERS or _file_identity(os.lstat(root)) != _file_identity(root_metadata):
        raise BatchError("one-case smoke bundle root changed during validation")
    for name, before in initial.items():
        if _file_identity(os.lstat(root / name)) != before:
            raise BatchError(f"one-case smoke bundle member changed during validation: {name}")
    return bundle, {
        "mode": "validate_only",
        "root_contract": ONE_CASE_ROOT_CONTRACT,
        "bundle_root": {"path": str(root), "device": root_metadata.st_dev, "inode": root_metadata.st_ino},
        "members": member_inventory,
        "bundle": {"path": str(bundle_path), "sha256": sha_file(bundle_path, "one-case smoke bundle")},
        "fake_ready": {"path": str(fake_ready_path), "sha256": sha_file(fake_ready_path, "one-case smoke fake-ready")},
        "fake_driver_subprocess_count": 1,
        "driver_fake_handshake": "passed",
        "resident_session_id": session_id,
        "driver_identity": driver_identity,
        "resident_driver_argv": expected_driver_argv,
        "trusted_bundle_validator": validator,
    }


def _send(process: subprocess.Popen[str], message: dict[str, Any]) -> None:
    if process.stdin is None:
        raise BatchError("resident driver stdin is unavailable")
    process.stdin.write(json.dumps(message, ensure_ascii=True, sort_keys=True) + "\n")
    process.stdin.flush()


def _recv(process: subprocess.Popen[str], timeout: float) -> dict[str, Any]:
    if process.stdout is None:
        raise BatchError("resident driver stdout is unavailable")
    ready, _, _ = select.select([process.stdout], [], [], timeout)
    if not ready:
        raise BatchError("resident driver response timed out")
    line = process.stdout.readline()
    if not line:
        raise BatchError("resident driver exited before response")
    value = json.loads(line, object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(BatchError(f"non-finite driver number: {item}")))
    if not isinstance(value, dict):
        raise BatchError("resident driver response is not an object")
    return value


def _validate_ready_identity(value: Any, identity: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != READY_IDENTITY_KEYS:
        raise BatchError("resident driver identity fields differ")
    for field in ("binary_sha256", "worker_binary_sha256", "package_manifest_sha256", "package_content_sha256", "served_model_manifest_sha256", "guard_set_sha256"):
        if not isinstance(value[field], str) or SHA256_RE.fullmatch(value[field]) is None:
            raise BatchError(f"resident driver identity.{field} is invalid")
    if not isinstance(value["build_git_commit"], str) or GIT_SHA_RE.fullmatch(value["build_git_commit"]) is None:
        raise BatchError("resident driver identity.build_git_commit is invalid")
    if value["protocol"] != DRIVER_SCHEMA:
        raise BatchError("resident driver identity protocol differs")
    for field in ("model_id", "model_revision", "format_id", "implementation_id"):
        if not isinstance(value[field], str) or not value[field]:
            raise BatchError(f"resident driver identity.{field} is invalid")
    runtime = value["runtime_device"]
    if not isinstance(runtime, dict) or set(runtime) != RUNTIME_DEVICE_KEYS:
        raise BatchError("resident driver runtime device fields differ")
    if type(runtime["runtime_device_index"]) is not int or runtime["runtime_device_index"] < 0:
        raise BatchError("resident driver runtime device index is invalid")
    if not isinstance(runtime["device_id"], (str, int)) or isinstance(runtime["device_id"], bool) or (isinstance(runtime["device_id"], str) and not runtime["device_id"]):
        raise BatchError("resident driver runtime device ID is invalid")
    for field in ("backend", "name", "architecture"):
        if not isinstance(runtime[field], str) or not runtime[field]:
            raise BatchError(f"resident driver runtime device {field} is invalid")
    bound = identity.get("resident_driver_identity")
    if not isinstance(bound, dict) or set(bound) != READY_IDENTITY_KEYS:
        raise BatchError("identity file lacks resident driver identity")
    if bound != value:
        raise BatchError("resident driver identity differs from identity file")
    bound_hashes = identity.get("hash_binding", {})
    for field in ("package_manifest_sha256", "package_content_sha256", "served_model_manifest_sha256", "worker_binary_sha256"):
        if isinstance(bound_hashes, dict) and bound_hashes.get(field) != value[field]:
            raise BatchError(f"resident {field} identity differs")
    if identity.get("build_git_commit") not in (None, value["build_git_commit"]):
        raise BatchError("resident build commit identity differs")
    for case in cases:
        device = case.get("device")
        if not isinstance(device, dict):
            raise BatchError(f"case device identity is missing: {case.get('case_id')}")
        for field in RUNTIME_DEVICE_KEYS:
            expected = device.get(field)
            if field == "architecture" and runtime[field] == "gfx1201" and expected == "RDNA4":
                expected = "gfx1201"
            if field in device and expected != runtime[field]:
                raise BatchError(f"resident runtime device differs from case: {case['case_id']}")
    return value


def validate_ready(value: dict[str, Any], identity: dict[str, Any], cases: list[dict[str, Any]], expected_binary_sha256: str) -> tuple[str, dict[str, Any]]:
    if set(value) != {"event", "schema_version", "model_loads", "resident_session_id", "driver_identity"} or value.get("event") != "ready" or value.get("schema_version") != DRIVER_SCHEMA or type(value.get("model_loads")) is not int or value.get("model_loads") != 1 or not isinstance(value.get("resident_session_id"), str) or not value["resident_session_id"]:
        raise BatchError("resident driver did not prove one model load")
    ready_identity = _validate_ready_identity(value["driver_identity"], identity, cases)
    if ready_identity["binary_sha256"] != expected_binary_sha256:
        raise BatchError("resident driver ready self SHA differs from pre-spawn executable")
    return value["resident_session_id"], ready_identity


def validate_run(value: dict[str, Any], case: dict[str, Any], session_id: str) -> dict[str, Any]:
    required = {"event", "schema_version", "resident_session_id", "case_id", "run_index", "run_kind", "status", "elapsed_ms", "requested_m", "resolved_m", "actual_token_batch_width", "actual_request_batch_width", "timing", "audit", "state", "lifecycle", "reset", "resource", "terminal"}
    if set(value) != required or value.get("event") != "run_complete" or value.get("schema_version") != DRIVER_SCHEMA or value.get("resident_session_id") != session_id or value.get("case_id") != case["case_id"] or value.get("status") not in {"ok", "failed", "oom"}:
        raise BatchError(f"resident driver run identity/status differs: {case['case_id']}")
    if type(value.get("elapsed_ms")) not in {int, float} or value["elapsed_ms"] < 0:
        raise BatchError("resident driver elapsed_ms is invalid")
    reset = value.get("reset")
    terminal = value.get("terminal")
    if not isinstance(terminal, dict) or set(terminal) != {"reuse_forbidden", "reason_code", "oom", "hip_fault"} or type(terminal["reuse_forbidden"]) is not bool or type(terminal["oom"]) is not bool or type(terminal["hip_fault"]) is not bool or not isinstance(terminal["reason_code"], str):
        raise BatchError(f"resident driver terminal fields differ: {case['case_id']}")
    valid_resets = ({"attempted": 1, "complete": 1, "failed": 0}, {"attempted": 1, "complete": 0, "failed": 1}) if terminal["reuse_forbidden"] else ({"attempted": 1, "complete": 1, "failed": 0},)
    if not isinstance(reset, dict) or reset not in valid_resets:
        raise BatchError(f"resident driver reset is not complete: {case['case_id']}")
    if value["status"] == "oom" and (not terminal["oom"] or not terminal["reuse_forbidden"]):
        raise BatchError("resident driver OOM is not terminal")
    if terminal["hip_fault"] and not terminal["reuse_forbidden"]:
        raise BatchError("resident driver HIP fault is reusable")
    if value["status"] == "ok":
        audit = value.get("audit")
        resource = value.get("resource")
        if not isinstance(audit, dict) or audit.get("coverage_complete") is not True or not isinstance(audit.get("deterministic_digest_sha256"), str) or not isinstance(resource, dict) or not resource.get("samples") or not isinstance(resource.get("peak"), dict):
            raise BatchError(f"resident driver terminal audit/resource is incomplete: {case['case_id']}")
        if terminal["reuse_forbidden"] or value.get("requested_m") != case.get("prefill_requested_m") or value.get("resolved_m") != case.get("resolved_m") or value.get("actual_token_batch_width") != case.get("resolved_m") or value.get("actual_request_batch_width") != case.get("request_count"):
            raise BatchError(f"resident driver actual width differs: {case['case_id']}")
        for field in ("timing", "state", "lifecycle"):
            if not isinstance(value.get(field), dict):
                raise BatchError(f"resident driver {field} is incomplete: {case['case_id']}")
    return value


def make_case_raw(case: dict[str, Any], fixture_entry: dict[str, Any], identity_link: dict[str, str], policy_link: dict[str, str], run_id: str, baseline_kind: str, session_id: str, driver_identity: dict[str, Any], device_lock: dict[str, Any], runs: list[dict[str, Any]], failure_reason: str | None = None, live_preflight: dict[str, Any] | None = None) -> dict[str, Any]:
    status = "ok" if not failure_reason and all(run["status"] == "ok" for run in runs) else "oom" if any(run["status"] == "oom" for run in runs) else "failed"
    terminal = {
        "audit_digests": [run.get("audit", {}).get("deterministic_digest_sha256") for run in runs if isinstance(run.get("audit"), dict)],
        "reset_count": sum(1 for run in runs if run.get("reset") == {"attempted": 1, "complete": 1, "failed": 0}),
        "all_resets_complete": all(run.get("reset") == {"attempted": 1, "complete": 1, "failed": 0} for run in runs),
    }
    return {
        "schema_version": "ullm.aq4_p2_resident_batch_raw.v1",
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "status": status,
        "immutable_status": status != "ok",
        "baseline_identity": {
            "run_id": run_id,
            "kind": baseline_kind,
            "identity_file": identity_link,
        },
        "resident": {"session_id": session_id, "model_loads": 1, "driver_identity": driver_identity, "case_reset_count": sum(1 for run in runs if run.get("reset") == {"attempted": 1, "complete": 1, "failed": 0})},
        "device_lock": device_lock,
        "workload": {key: case.get(key) for key in ("scope", "phase", "mode", "prompt_tokens", "cached_prefix_tokens", "context_tokens", "prefill_requested_m", "resolved_m", "request_count", "generated_tokens")},
        "schedule": {"warmup_runs": WARMUP_RUNS, "measured_runs": MEASURED_RUNS, "completed_runs": len(runs)},
        "runs": runs,
        "terminal": terminal,
        "failure_reason": failure_reason,
        "links": {"fixture": {"path": fixture_entry["fixture_path"], "sha256": fixture_entry["fixture_sha256"]}, "identity": identity_link, "policy": policy_link, **({"live_preflight": live_preflight} if live_preflight is not None else {})},
    }


def build_plan(cases: list[dict[str, Any]], expanded_path: Path, fixture_index_path: Path, run_id: str, baseline_kind: str, identity: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    token_sum = sum(int(case["prompt_tokens"]) for case in cases) * (WARMUP_RUNS + MEASURED_RUNS)
    return {
        "schema_version": SCHEMA,
        "status": "dry_run",
        "scope": "full_model",
        "case_count": len(cases),
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "transaction_count": len(cases) * (WARMUP_RUNS + MEASURED_RUNS),
        "prompt_tokens_across_transactions": token_sum,
        "resident_model_loads": 1,
        "baseline_identity": {
            "run_id": run_id,
            "kind": baseline_kind,
            "identity_file": {"path": str(identity.get("_path", "")), "sha256": identity.get("_sha256")},
            "served_model_manifest_sha256": identity.get("hash_binding", {}).get("served_model_manifest_sha256"),
            "worker_binary_sha256": identity.get("hash_binding", {}).get("worker_binary_sha256"),
            "build_git_commit": identity.get("build_git_commit"),
        },
        "links": {"expanded": {"path": str(expanded_path), "sha256": sha_file(expanded_path, "expanded")}, "fixture_index": {"path": str(fixture_index_path), "sha256": sha_file(fixture_index_path, "fixture index")}, "policy": {"path": str(policy.get("_path", "")), "sha256": policy.get("_sha256")}},
    }


def validate_live_preflight(path: Path, args: argparse.Namespace, bundle: dict[str, Any]) -> dict[str, Any]:
    _require_absolute_nonsymlink_path(path, "live preflight")
    raw, digest, before = read_regular(path, "live preflight", MAX_JSON_BYTES, absolute=True)
    if stat.S_IMODE(before.st_mode) != 0o444:
        raise BatchError("live preflight mode must be 0444")
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(BatchError(f"non-finite live preflight number: {item}")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BatchError(f"invalid live preflight: {error}") from error
    exact = {"schema_version", "status", "run_id", "captured_unix_ns", "prepared_preflight", "runtime_mapping", "services", "worker_pids", "compute_owners", "lock", "environment", "vram", "commands"}
    if not isinstance(value, dict) or set(value) != exact or value.get("schema_version") != "ullm.aq4_p2_resident_live_preflight.v1" or value.get("status") != "passed" or value.get("run_id") != args.run_id or type(value.get("captured_unix_ns")) is not int or value["captured_unix_ns"] <= 0:
        raise BatchError("live preflight exact schema/status/run differs")
    prepared = value.get("prepared_preflight")
    expected_prepared = {"path": str(args.bundle_root / "preflight.json"), "sha256": sha_file(args.bundle_root / "preflight.json", "prepared synthetic preflight"), "role": "synthetic_bundle_contract_only"}
    if prepared != expected_prepared:
        raise BatchError("live preflight does not explicitly replace the synthetic bundle preflight")
    mapping = value.get("runtime_mapping")
    expected_device = bundle.get("expected_runtime", {}).get("device", {})
    if not isinstance(mapping, dict) or mapping.get("runtime_device_index") != expected_device.get("runtime_device_index") or mapping.get("visible_token") != "1" or mapping.get("amd_smi_index") != 2 or mapping.get("bdf") != "0000:47:00.0" or mapping.get("uuid") != "a8ff7551-0000-1000-80e9-ddefa2d60f55" or mapping.get("kfd_id") != 51545:
        raise BatchError("live preflight runtime mapping differs")
    services = value.get("services")
    if services != [{"unit": "ullm-openai.service", "active_state": "inactive", "sub_state": "dead", "main_pid": 0}, {"unit": "llama-qwen35-udq4.service", "active_state": "inactive", "sub_state": "dead", "main_pid": 0}]:
        raise BatchError("live preflight service state differs")
    owners = value.get("compute_owners")
    if value.get("worker_pids") != [] or owners != {"amd_smi": [], "kfd": []}:
        raise BatchError("live preflight compute owners differ")
    lock = value.get("lock")
    if not isinstance(lock, dict) or lock.get("path") != str(args.lock_path) or lock.get("free") is not True or type(lock.get("device")) is not int or type(lock.get("inode")) is not int:
        raise BatchError("live preflight lock binding differs")
    expected_environment = bundle.get("expected_runtime", {}).get("environment", {}) | bundle.get("expected_runtime", {}).get("required_guards", {}) | {"ULLM_SERVED_MODEL_MANIFEST": "/etc/ullm/served-models/active.json", "ULLM_BUILD_GIT_COMMIT": bundle.get("resident_driver", {}).get("source_commit")}
    if value.get("environment") != expected_environment:
        raise BatchError("live preflight environment differs")
    vram = value.get("vram")
    if not isinstance(vram, dict) or set(vram) != {"total_bytes", "used_bytes", "free_bytes", "headroom_bytes"} or any(type(vram.get(name)) is not int or vram[name] < 0 for name in vram) or vram["total_bytes"] <= 0 or vram["free_bytes"] != vram["total_bytes"] - vram["used_bytes"] or vram["headroom_bytes"] != vram["free_bytes"]:
        raise BatchError("live preflight VRAM/headroom differs")
    commands = value.get("commands")
    labels = {"sudo-n", "service-ullm-openai.service", "service-llama-qwen35-udq4.service", "old-worker", "amd-smi-list", "rocminfo", "amd-smi-process", "amd-smi-static-vram"}
    if not isinstance(commands, list) or any(not isinstance(item, dict) for item in commands) or {item.get("label") for item in commands} != labels or any(set(item) != {"label", "argv", "exit_code", "stdout_sha256", "stderr_sha256", "captured_unix_ns"} or item.get("exit_code") not in {0, 1} for item in commands):
        raise BatchError("live preflight command evidence differs")
    if _file_identity(before) != _file_identity(os.lstat(path)) or sha_file(path, "live preflight", absolute=True) != digest:
        raise BatchError("live preflight changed during validation")
    return {"path": str(path), "sha256": digest, "device": before.st_dev, "inode": before.st_ino, "captured_unix_ns": value["captured_unix_ns"], "runtime_mapping": mapping, "vram": vram}


def verify_live_preflight(path: Path, link: dict[str, Any]) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise BatchError(f"live preflight final metadata failed: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o444:
        raise BatchError("live preflight final file contract differs")
    if metadata.st_dev != link.get("device") or metadata.st_ino != link.get("inode"):
        raise BatchError("live preflight identity changed after validation")
    if sha_file(path, "live preflight final", absolute=True) != link.get("sha256"):
        raise BatchError("live preflight content changed after validation")


def run_batch(args: argparse.Namespace) -> int:
    if not args.one_case_smoke and (args.bundle_root is not None or args.trusted_validator is not None or args.trusted_validator_sha256 is not None or args.live_preflight is not None):
        raise BatchError("--bundle-root/--trusted-validator require --one-case-smoke")
    if args.one_case_smoke and args.bundle_root is None:
        raise BatchError("--bundle-root is required with --one-case-smoke")
    if args.one_case_smoke and (args.trusted_validator is None or args.trusted_validator_sha256 is None):
        raise BatchError("--trusted-validator and --trusted-validator-sha256 are required with --one-case-smoke")
    if args.one_case_smoke and args.dry_run and args.live_preflight is not None:
        raise BatchError("--live-preflight is forbidden for dry-run")
    if args.one_case_smoke and not args.dry_run and args.live_preflight is None:
        raise BatchError("--live-preflight is required for actual one-case smoke")
    expanded = load(args.expanded, "expanded")
    fixture_index = load(args.fixture_index, "fixture index")
    identity = load(args.identity, "identity")
    preflight = load(args.preflight, "preflight")
    policy = load(args.policy, "policy")
    normalize_fixture_paths(fixture_index)
    expanded_link = {"path": str(args.expanded.resolve()), "sha256": sha_file(args.expanded, "expanded")}
    identity_link = {"path": str(args.identity.resolve()), "sha256": sha_file(args.identity, "identity")}
    preflight_link = {"path": str(args.preflight.resolve()), "sha256": sha_file(args.preflight, "preflight")}
    policy_link = {"path": str(args.policy.resolve()), "sha256": sha_file(args.policy, "policy")}
    identity["_path"], identity["_sha256"] = str(args.identity.resolve()), identity_link["sha256"]
    policy["_path"], policy["_sha256"] = str(args.policy.resolve()), policy_link["sha256"]
    if args.baseline_kind not in {"active-production", "p3-current-head"}:
        raise BatchError("baseline kind must identify one immutable build/run")
    cases = select_target_cases(expanded, fixture_index, one_case_smoke=args.one_case_smoke)
    smoke_validation = None
    smoke_bundle = None
    if args.one_case_smoke:
        smoke_bundle, smoke_validation = validate_one_case_smoke_bundle(args, expanded, fixture_index, identity, preflight, policy, cases)
    live_preflight_link = None
    if args.one_case_smoke and not args.dry_run:
        live_preflight_link = validate_live_preflight(args.live_preflight, args, smoke_bundle)
        smoke_validation["live_preflight"] = live_preflight_link
        preflight_link = live_preflight_link
    plan = build_plan(cases, args.expanded, args.fixture_index, args.run_id, args.baseline_kind, identity, policy)
    if args.one_case_smoke:
        plan.update({"execution_mode": "one_case_smoke", "smoke_only": True, "promotion_eligible": False, "validation": smoke_validation})
    if args.dry_run:
        atomic_write(args.output_dir / "resident-batch.plan.json", plan)
        return 0
    if not args.driver_command:
        raise BatchError("--driver-command is required unless --dry-run is set")
    expected_driver_argv = smoke_validation["resident_driver_argv"] if smoke_validation is not None else None
    driver_executable = validate_driver_command(args.driver_command, identity, expected_argv=expected_driver_argv)
    if live_preflight_link is not None:
        verify_live_preflight(args.live_preflight, live_preflight_link)
    completed_cases = 0
    with acquire_device_lock(args.lock_path, args.run_id, driver_executable) as lock_owner:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        atomic_write(args.output_dir / "resident-batch.lock-owner.json", lock_owner)
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(args.driver_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=False, bufsize=1)
            session_id, driver_identity = validate_ready(_recv(process, args.timeout), identity, cases, driver_executable["sha256"])
            by_id = {entry["case_id"]: entry for entry in fixture_index["cases"]}
            for case in cases:
                if live_preflight_link is not None:
                    verify_live_preflight(args.live_preflight, live_preflight_link)
                fixture_entry = by_id[case["case_id"]]
                sampling = case.get("sampling")
                control = case.get("control")
                if not isinstance(sampling, dict) or not isinstance(control, dict):
                    raise BatchError(f"case sampling/control is missing: {case['case_id']}")
                _send(process, {
                    "command": "case_begin", "schema_version": DRIVER_SCHEMA,
                    "case_id": case["case_id"], "case_sha256": case["case_sha256"],
                    "case_binding": expanded_link, "identity": identity_link,
                    "preflight": {"path": preflight_link["path"], "sha256": preflight_link["sha256"]}, "policy": policy_link,
                    "fixture": {"path": fixture_entry["fixture_path"], "sha256": fixture_entry["fixture_sha256"]},
                    "execution": {
                        "scope": case.get("scope"), "phase": case.get("phase"), "mode": case.get("mode"),
                        "prompt_tokens": case.get("prompt_tokens"), "cached_prefix_tokens": case.get("cached_prefix_tokens"),
                        "context_tokens": case.get("context_tokens"), "generated_tokens": case.get("generated_tokens"),
                        "request_count": case.get("request_count"), "requested_m": case.get("prefill_requested_m"),
                        "resolved_m": case.get("resolved_m"), "sampling": sampling, "control": control,
                    },
                })
                begin = _recv(process, args.timeout)
                if set(begin) != {"event", "schema_version", "resident_session_id", "case_id", "requested_m", "resolved_m", "baseline_clean"} or begin.get("event") != "case_ready" or begin.get("schema_version") != DRIVER_SCHEMA or begin.get("resident_session_id") != session_id or begin.get("case_id") != case["case_id"] or begin.get("requested_m") != case["prefill_requested_m"] or begin.get("resolved_m") != case["resolved_m"] or begin.get("baseline_clean") is not True:
                    raise BatchError(f"resident driver case begin failed: {case['case_id']}")
                runs: list[dict[str, Any]] = []
                reuse_forbidden = False
                for run_index in range(WARMUP_RUNS + MEASURED_RUNS):
                    run_kind = "warmup" if run_index < WARMUP_RUNS else "measured"
                    _send(process, {"command": "run", "schema_version": DRIVER_SCHEMA, "case_id": case["case_id"], "run_index": run_index, "run_kind": run_kind})
                    value = validate_run(_recv(process, args.timeout), case, session_id)
                    if value["run_index"] != run_index or value["run_kind"] != run_kind:
                        raise BatchError(f"resident driver run order differs: {case['case_id']}")
                    runs.append(value)
                    if value["terminal"]["reuse_forbidden"] or value["status"] != "ok":
                        reuse_forbidden = value["terminal"]["reuse_forbidden"]
                        break
                if not reuse_forbidden:
                    _send(process, {"command": "case_end", "schema_version": DRIVER_SCHEMA, "case_id": case["case_id"]})
                    end = _recv(process, args.timeout)
                    case_failed = any(item["status"] != "ok" for item in runs)
                    expected_release = {"commit": int(not case_failed), "discard": int(case_failed), "reset": 1, "baseline_restored": True}
                    if set(end) != {"event", "schema_version", "resident_session_id", "case_id", "release"} or end.get("event") != "case_complete" or end.get("schema_version") != DRIVER_SCHEMA or end.get("resident_session_id") != session_id or end.get("case_id") != case["case_id"] or end.get("release") != expected_release:
                        raise BatchError(f"resident driver case end failed: {case['case_id']}")
                failure_reason = "resident_driver_oom" if any(item["status"] == "oom" for item in runs) else next((item["terminal"]["reason_code"] for item in runs if item["status"] != "ok"), None)
                raw = make_case_raw(case, fixture_entry, identity_link, policy_link, args.run_id, args.baseline_kind, session_id, driver_identity, lock_owner, runs, failure_reason, live_preflight=live_preflight_link)
                if args.one_case_smoke:
                    raw.update({"execution_mode": "one_case_smoke", "smoke_only": True, "promotion_eligible": False})
                atomic_write(args.output_dir / f"{case['case_id']}.raw.json", raw)
                completed_cases += 1
                if reuse_forbidden:
                    raise BatchError(f"resident driver became non-reusable at {case['case_id']}; remaining cases were not executed")
        finally:
            if process is not None:
                try:
                    _send(process, {"command": "shutdown", "schema_version": DRIVER_SCHEMA})
                except (BatchError, OSError):
                    pass
                try:
                    process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if live_preflight_link is not None:
            verify_live_preflight(args.live_preflight, live_preflight_link)
        atomic_write(args.output_dir / "resident-batch.summary.json", {**plan, "status": "complete", "completed_cases": completed_cases, "device_lock": lock_owner})
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--fixture-index", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-kind", choices=("active-production", "p3-current-head"), required=True)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--one-case-smoke", action="store_true", help="run the exact bundle-v3 one-case smoke; never promotion eligible")
    parser.add_argument("--bundle-root", type=Path, help="absolute complete 791a20c bundle root; required by --one-case-smoke")
    parser.add_argument("--trusted-validator", type=Path, help="trusted bundle validator Python source required by --one-case-smoke")
    parser.add_argument("--trusted-validator-sha256", help="expected lowercase SHA-256 of --trusted-validator")
    parser.add_argument("--live-preflight", type=Path, help="immutable live gate sidecar required for actual one-case smoke and forbidden for dry-run")
    parser.add_argument("--driver-command", nargs=argparse.REMAINDER, help="exact resident driver argv; this option and its value must be last")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_batch(args)
    except (BatchError, OSError, subprocess.SubprocessError) as error:
        print(f"AQ4 P2 resident batch failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
