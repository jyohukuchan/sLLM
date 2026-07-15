#!/usr/bin/env python3
"""Validate an in-construction AQ4 resident bundle and emit its dry-run plan.

This bootstrap is intentionally preparation-only.  It has no resident-driver,
GPU, device-lock, service, or actual execution path.  The completed immutable
bundle is validated and executed later by run-aq4-p2-resident-batch.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any


MAX_JSON_BYTES = 64 * 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024
BUNDLE_SCHEMA = "ullm.aq4_p2_resident_smoke_binding_bundle.v4"
BATCH_SCHEMA = "ullm.aq4_p2_resident_batch.v1"
DRIVER_SCHEMA = "ullm.aq4_p2_resident_driver.v2"
WARMUP_RUNS = 2
MEASURED_RUNS = 10
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
CASE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
PREPARED_MEMBER_CONTRACT = {
    "SUPERSEDED-0fd7993.json": ("0444", "historical_non_executable_bundle_record"),
    "case-binding.json": ("0444", "runtime_bound_case"),
    "fake-ready.json": ("0444", "synthetic_ready_event"),
    "fixture-index.json": ("0444", "fixture_index"),
    "fixture.json": ("0444", "fixture"),
    "identity.json": ("0444", "resident_identity"),
    "launch-command.json": ("0444", "exact_resident_launch_command"),
    "official-case.json": ("0444", "trusted_official_expansion_case"),
    "package-manifest.json": ("0444", "package_manifest_snapshot"),
    "policy.json": ("0444", "threshold_policy"),
    "preflight.json": ("0444", "synthetic_preflight"),
    "resident-driver": ("0555", "detached_resident_driver"),
    "served-model.json": ("0444", "served_model_snapshot"),
    "trust-roots.json": ("0444", "independent_trust_roots"),
    "trusted-runner.py": ("0444", "trusted_one_case_smoke_runner"),
}
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


class BootstrapError(ValueError):
    pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in items:
        if key in value:
            raise BootstrapError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        raise BootstrapError(f"{label} path must be absolute without parent traversal")
    for parent in (path, *reversed(path.parents)):
        try:
            metadata = os.lstat(parent)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise BootstrapError(f"{label} path traverses a symlink")


def read_regular(
    path: Path,
    label: str,
    maximum: int | None = None,
    *,
    collect: bool = True,
) -> tuple[bytes, str]:
    try:
        before = os.lstat(path)
    except OSError as error:
        raise BootstrapError(f"{label} metadata failed: {error}") from error
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise BootstrapError(f"{label} must be a single-link regular file")
    if maximum is not None and before.st_size > maximum:
        raise BootstrapError(f"{label} exceeds the byte bound")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BootstrapError(f"{label} open failed: {error}") from error
    try:
        if _file_identity(before) != _file_identity(os.fstat(descriptor)):
            raise BootstrapError(f"{label} changed while opening")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, READ_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if maximum is not None and total > maximum:
                raise BootstrapError(f"{label} exceeds the byte bound")
            digest.update(chunk)
            if collect:
                chunks.append(chunk)
        if _file_identity(before) != _file_identity(os.fstat(descriptor)) or _file_identity(before) != _file_identity(os.lstat(path)):
            raise BootstrapError(f"{label} changed while reading")
        return b"".join(chunks), digest.hexdigest()
    finally:
        os.close(descriptor)


def sha_file(path: Path, label: str) -> str:
    return read_regular(path, label, collect=False)[1]


def load(path: Path, label: str) -> dict[str, Any]:
    raw, _ = read_regular(path, label, MAX_JSON_BYTES)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                BootstrapError(f"non-finite {label}: {item}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BootstrapError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise BootstrapError(f"{label} root must be an object")
    return value


def case_hash(case: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(case))
    clone["case_sha256"] = None
    return sha_bytes(canonical(clone))


def validate_bundle_header(bundle: dict[str, Any]) -> None:
    if bundle.get("schema_version") != BUNDLE_SCHEMA:
        raise BootstrapError("prepared bootstrap bundle schema differs")
    if bundle.get("status") != "prepared_not_executed":
        raise BootstrapError("prepared bootstrap bundle status differs")
    if bundle.get("promotion") is not False:
        raise BootstrapError("prepared bootstrap bundle promotion differs")


def validate_member_inventory(root: Path, bundle: dict[str, Any]) -> None:
    files = bundle.get("files")
    if not isinstance(files, dict) or set(files) != set(PREPARED_MEMBER_CONTRACT):
        raise BootstrapError("prepared bootstrap bundle file coverage differs")
    for name, (mode, role) in PREPARED_MEMBER_CONTRACT.items():
        record = files.get(name)
        if (
            not isinstance(record, dict)
            or set(record) != {"mode", "role", "sha256"}
            or record.get("mode") != mode
            or record.get("role") != role
            or not isinstance(record.get("sha256"), str)
            or SHA256_RE.fullmatch(record["sha256"]) is None
        ):
            raise BootstrapError(f"prepared bootstrap member contract differs: {name}")
        if sha_file(root / name, f"prepared bootstrap {name}") != record["sha256"]:
            raise BootstrapError(f"prepared bootstrap member SHA differs: {name}")


def select_case(expanded: dict[str, Any], fixture_index: dict[str, Any], root: Path) -> dict[str, Any]:
    if expanded.get("schema_version") != "ullm.aq4_production_p2_expanded.v2":
        raise BootstrapError("prepared bootstrap case-binding schema differs")
    cases = expanded.get("cases")
    if not isinstance(cases, list) or len(cases) != 1 or not isinstance(cases[0], dict):
        raise BootstrapError("prepared bootstrap requires exactly one case")
    case = cases[0]
    case_id = case.get("case_id")
    if (
        not isinstance(case_id, str)
        or CASE_ID_RE.fullmatch(case_id) is None
        or case.get("case_sha256") != case_hash(case)
        or case.get("stage_id") != "representative"
        or case.get("scope") != "full_model"
        or case.get("phase") != "cold_prefill"
        or case.get("control_id") != "aq4_0_target"
        or case.get("device", {}).get("device_id") != "r9700-rdna4"
    ):
        raise BootstrapError("prepared bootstrap case identity differs")
    entries = fixture_index.get("cases")
    if (
        fixture_index.get("schema_version") != "ullm.aq4_p2_fixture_index.v1"
        or fixture_index.get("subset") != "resident_one_case_smoke"
        or fixture_index.get("case_count") != 1
        or not isinstance(entries, list)
        or len(entries) != 1
        or not isinstance(entries[0], dict)
    ):
        raise BootstrapError("prepared bootstrap fixture index differs")
    entry = entries[0]
    fixture_path = Path(entry.get("fixture_path", ""))
    _require_absolute_nonsymlink_path(fixture_path, "prepared bootstrap fixture")
    if (
        fixture_path != root / "fixture.json"
        or entry.get("case_id") != case_id
        or entry.get("case_sha256") != case.get("case_sha256")
        or entry.get("prompt_tokens") != case.get("prompt_tokens")
        or entry.get("context_tokens") != case.get("context_tokens")
        or entry.get("generated_tokens") != case.get("generated_tokens")
        or entry.get("fixture_sha256") != sha_file(fixture_path, "prepared bootstrap fixture")
    ):
        raise BootstrapError("prepared bootstrap fixture binding differs")
    return case


def validate_ready_identity(value: Any, identity: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != READY_IDENTITY_KEYS:
        raise BootstrapError("prepared bootstrap driver identity fields differ")
    for field in (
        "binary_sha256",
        "worker_binary_sha256",
        "package_manifest_sha256",
        "package_content_sha256",
        "served_model_manifest_sha256",
        "guard_set_sha256",
    ):
        if not isinstance(value[field], str) or SHA256_RE.fullmatch(value[field]) is None:
            raise BootstrapError(f"prepared bootstrap driver identity {field} differs")
    if not isinstance(value["build_git_commit"], str) or GIT_SHA_RE.fullmatch(value["build_git_commit"]) is None:
        raise BootstrapError("prepared bootstrap driver build commit differs")
    if value.get("protocol") != DRIVER_SCHEMA:
        raise BootstrapError("prepared bootstrap driver protocol differs")
    runtime = value.get("runtime_device")
    if not isinstance(runtime, dict) or set(runtime) != RUNTIME_DEVICE_KEYS:
        raise BootstrapError("prepared bootstrap runtime device fields differ")
    bound = identity.get("resident_driver_identity")
    if bound != value:
        raise BootstrapError("prepared bootstrap driver identity binding differs")
    device = case.get("device")
    if not isinstance(device, dict):
        raise BootstrapError("prepared bootstrap case device differs")
    for field in RUNTIME_DEVICE_KEYS:
        expected = device.get(field)
        if field == "architecture" and runtime[field] == "gfx1201" and expected == "RDNA4":
            expected = "gfx1201"
        if field in device and runtime[field] != expected:
            raise BootstrapError("prepared bootstrap runtime/case device differs")
    return value


def validate_prepared(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path, label in (
        (args.expanded, "case-binding"),
        (args.fixture_index, "fixture-index"),
        (args.identity, "identity"),
        (args.preflight, "preflight"),
        (args.policy, "policy"),
    ):
        _require_absolute_nonsymlink_path(path, f"prepared bootstrap {label}")
    root = args.expanded.parent.resolve(strict=True)
    expected_paths = {
        "expanded": root / "case-binding.json",
        "fixture_index": root / "fixture-index.json",
        "identity": root / "identity.json",
        "preflight": root / "preflight.json",
        "policy": root / "policy.json",
    }
    for name, expected in expected_paths.items():
        if getattr(args, name).resolve(strict=True) != expected:
            raise BootstrapError(f"prepared bootstrap {name} is not the canonical member")
    expanded = load(args.expanded, "prepared bootstrap case-binding")
    fixture_index = load(args.fixture_index, "prepared bootstrap fixture-index")
    identity = load(args.identity, "prepared bootstrap identity")
    preflight = load(args.preflight, "prepared bootstrap preflight")
    policy = load(args.policy, "prepared bootstrap policy")
    bundle_path = root / "bundle.json"
    fake_ready_path = root / "fake-ready.json"
    bundle = load(bundle_path, "prepared bootstrap bundle")
    validate_bundle_header(bundle)
    if bundle.get("canonical_root") != str(root):
        raise BootstrapError("prepared bootstrap canonical root differs")
    if {entry.name for entry in root.iterdir()} != set(PREPARED_MEMBER_CONTRACT) | {"bundle.json"}:
        raise BootstrapError("prepared bootstrap root member coverage differs")
    validate_member_inventory(root, bundle)
    case = select_case(expanded, fixture_index, root)
    case_binding_sha = sha_file(args.expanded, "prepared bootstrap case-binding")
    bindings = bundle.get("bindings")
    if (
        not isinstance(bindings, dict)
        or bindings.get("case_binding_sha256") != case_binding_sha
        or bindings.get("case_sha256") != case.get("case_sha256")
        or expanded.get("status") != "bound_one_case_smoke"
        or expanded.get("case_count") != 1
        or expanded.get("canonical_case_sha256") != sha_bytes(canonical([case]))
        or fixture_index.get("expanded_manifest_sha256") != case_binding_sha
    ):
        raise BootstrapError("prepared bootstrap case/hash binding differs")
    if (
        identity.get("schema_version") != "ullm.aq4_production_p2_identity.v2"
        or identity.get("status") != "bound"
        or identity.get("expanded_manifest_sha256") != case_binding_sha
        or identity.get("hash_binding", {}).get("bound_case_manifest_sha256") != case_binding_sha
        or policy.get("schema_version") != "ullm.aq4_production_p2_threshold_policy.v1"
        or policy.get("status") != "bound"
        or not isinstance(preflight.get("gpu_process_snapshot"), list)
    ):
        raise BootstrapError("prepared bootstrap identity/preflight/policy binding differs")
    fake_ready = load(fake_ready_path, "prepared bootstrap fake-ready")
    if (
        set(fake_ready) != {
            "event",
            "schema_version",
            "model_loads",
            "resident_session_id",
            "driver_identity",
        }
        or fake_ready.get("event") != "ready"
        or fake_ready.get("schema_version") != DRIVER_SCHEMA
        or type(fake_ready.get("model_loads")) is not int
        or fake_ready.get("model_loads") != 1
        or not isinstance(fake_ready.get("resident_session_id"), str)
        or not fake_ready["resident_session_id"]
    ):
        raise BootstrapError("prepared bootstrap fake-ready differs")
    driver_identity = validate_ready_identity(fake_ready.get("driver_identity"), identity, case)
    if driver_identity["binary_sha256"] != bundle.get("resident_driver", {}).get("binary_sha256"):
        raise BootstrapError("prepared bootstrap resident binary binding differs")
    validation = {
        "mode": "validate_only",
        "bundle": {"path": str(bundle_path), "sha256": sha_file(bundle_path, "prepared bootstrap bundle")},
        "fake_ready": {"path": str(fake_ready_path), "sha256": sha_file(fake_ready_path, "prepared bootstrap fake-ready")},
        "driver_fake_handshake": "passed",
        "resident_session_id": fake_ready["resident_session_id"],
        "driver_identity": driver_identity,
    }
    return case, identity, validation


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    case, identity, validation = validate_prepared(args)
    identity_sha = sha_file(args.identity, "prepared bootstrap identity")
    policy_sha = sha_file(args.policy, "prepared bootstrap policy")
    return {
        "schema_version": BATCH_SCHEMA,
        "status": "dry_run",
        "scope": "full_model",
        "case_count": 1,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "transaction_count": WARMUP_RUNS + MEASURED_RUNS,
        "prompt_tokens_across_transactions": int(case["prompt_tokens"]) * (WARMUP_RUNS + MEASURED_RUNS),
        "resident_model_loads": 1,
        "baseline_identity": {
            "run_id": args.run_id,
            "kind": args.baseline_kind,
            "identity_file": {"path": str(args.identity), "sha256": identity_sha},
            "served_model_manifest_sha256": identity.get("hash_binding", {}).get("served_model_manifest_sha256"),
            "worker_binary_sha256": identity.get("hash_binding", {}).get("worker_binary_sha256"),
            "build_git_commit": identity.get("build_git_commit"),
        },
        "links": {
            "expanded": {"path": str(args.expanded), "sha256": sha_file(args.expanded, "prepared bootstrap case-binding")},
            "fixture_index": {"path": str(args.fixture_index), "sha256": sha_file(args.fixture_index, "prepared bootstrap fixture-index")},
            "policy": {"path": str(args.policy), "sha256": policy_sha},
        },
        "execution_mode": "one_case_smoke",
        "smoke_only": True,
        "promotion_eligible": False,
        "validation": validation,
    }


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    if os.path.lexists(path):
        raise BootstrapError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    try:
        with temporary.open("x", encoding="ascii") as target:
            json.dump(value, target, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
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
        raise BootstrapError(f"refusing to overwrite {path}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


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
    parser.add_argument("--one-case-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.one_case_smoke or not args.dry_run:
        parser.error("prepared bootstrap requires --one-case-smoke and --dry-run")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        atomic_write(args.output_dir / "resident-batch.plan.json", build_plan(args))
        return 0
    except (BootstrapError, OSError, TypeError, ValueError) as error:
        print(f"AQ4 P2 prepared bootstrap failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
