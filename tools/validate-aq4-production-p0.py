#!/usr/bin/env python3
"""Independently validate an AQ4 production optimization P0 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class ValidationError(ValueError):
    pass


def read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValidationError("snapshot root is not an object")
    return value


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValidationError("duplicate JSON key")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValidationError(f"non-finite number: {value}")


def digest(path: Path) -> str:
    cursor = path
    while cursor != cursor.parent:
        if cursor.is_symlink():
            raise ValidationError(f"identity path contains symlink component: {path}")
        cursor = cursor.parent
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def check_hash(value: Any, label: str) -> None:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise ValidationError(f"{label} is not a SHA-256")


def validate(snapshot_path: Path) -> dict[str, Any]:
    snapshot = read(snapshot_path)
    if snapshot.get("schema_version") != "ullm.aq4_production_optimization_p0.v1":
        raise ValidationError("unexpected P0 schema")
    if set(snapshot) != {
        "schema_version", "captured_at", "host", "git", "identity", "active_manifest",
        "rollback_binding", "service_topology", "hardware", "measurement_policy", "openwebui", "decision",
    }:
        raise ValidationError("P0 root fields differ")
    identity = snapshot["identity"]
    if not isinstance(identity, dict):
        raise ValidationError("identity is not an object")
    for name in ("manifest", "worker", "package_manifest", "product"):
        item = identity.get(name)
        if name == "product":
            continue
        if not isinstance(item, dict) or item.get("status") != "ok":
            raise ValidationError(f"identity.{name} is unavailable")
        check_hash(item.get("sha256"), f"identity.{name}.sha256")
        actual = digest(Path(item["path"]))
        if actual != item["sha256"]:
            raise ValidationError(f"identity.{name} hash differs")
    rollback = snapshot["rollback_binding"]
    if not isinstance(rollback, dict) or rollback.get("schema_version") != "ullm.aq4_production_optimization_rollback.v1":
        raise ValidationError("rollback binding schema differs")
    for name in ("active_manifest_sha256", "current_environment_sha256"):
        check_hash(rollback.get(name), f"rollback.{name}")
    matches = rollback.get("target_matches_current")
    if not isinstance(matches, dict) or set(matches) != {"manifest", "environment", "systemd_unit"}:
        raise ValidationError("rollback match fields differ")
    if not all(matches.values()):
        raise ValidationError("rollback binding is not bound to the current service state")
    if snapshot["decision"].get("active_product_changed") is not False or snapshot["decision"].get("active_service_changed") is not False:
        raise ValidationError("P0 capture claims an active mutation")
    policy = snapshot["measurement_policy"]
    if policy.get("warmup_runs") != 2 or policy.get("measured_runs") != 10 or policy.get("max_trace_bytes") != 4 * 1024 * 1024:
        raise ValidationError("measurement policy differs")
    if snapshot["openwebui"].get("promotion_gate_required") is not True:
        raise ValidationError("OpenWebUI reconciliation gate was not preserved")
    return {
        "schema_version": "ullm.aq4_production_optimization_p0_validator.v1",
        "status": "valid",
        "snapshot_sha256": digest(snapshot_path),
        "active_manifest_sha256": identity["manifest"]["sha256"],
        "rollback_status": rollback["status"],
        "active_product_changed": False,
        "service_active_observed": snapshot["service_topology"]["systemd"]["active"]["returncode"] == 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = validate(args.snapshot)
    except (ValidationError, OSError) as error:
        print(f"P0 validation failed: {error}", file=sys.stderr)
        return 1
    raw = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(raw, encoding="utf-8")
    else:
        print(raw, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
