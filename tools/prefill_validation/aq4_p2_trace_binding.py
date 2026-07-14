#!/usr/bin/env python3
"""Bind one AQ4 P2 case, result, resource observer, and production trace by SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

MAX_JSON_BYTES = 32 * 1024 * 1024
TRACE_FIELDS = {"schema_version", "status", "scope", "case_id", "case_sha256", "identity", "policy", "result", "resource_observation", "production_trace", "terminal", "binding"}
LINK_FIELDS = {"path", "sha256"}


class TraceBindingError(ValueError):
    pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise TraceBindingError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise TraceBindingError(f"{label} must be a bounded regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(TraceBindingError(f"non-finite JSON: {item}")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TraceBindingError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise TraceBindingError(f"{label} root must be an object")
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha_file(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise TraceBindingError(f"{label} must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def contained(root: Path, path: Path, label: str) -> Path:
    root = root.resolve(strict=True)
    result = path.resolve(strict=True)
    if result != root and root not in result.parents:
        raise TraceBindingError(f"{label} escapes run root")
    return result


def link(root: Path, path: Path, label: str) -> dict[str, str]:
    resolved = contained(root, path, label)
    return {"path": str(resolved), "sha256": sha_file(resolved, label)}


def validate_production_trace(value: dict[str, Any], case: dict[str, Any]) -> None:
    if value.get("schema_version") != "ullm.production_execution_trace.v1" or value.get("status") != "ok" or value.get("scope") != case.get("scope"):
        raise TraceBindingError("production trace schema/status/scope differs")
    # A historical P1 trace has no case identity and is deliberately not reusable for P2.
    if value.get("case_id") != case.get("case_id") or value.get("case_sha256") != case.get("case_sha256"):
        raise TraceBindingError("production trace lacks the exact P2 case binding")
    if not isinstance(value.get("trace_id"), str) or not value["trace_id"]:
        raise TraceBindingError("production trace_id is missing")


def build_trace_sidecar(
    run_root: Path,
    case_path: Path,
    identity_path: Path,
    policy_path: Path,
    result_path: Path,
    resource_observation_path: Path | None,
    production_trace_path: Path | None,
    result: dict[str, Any],
) -> dict[str, Any]:
    root = run_root.resolve(strict=True)
    case = load(case_path, "case")
    identity = load(identity_path, "identity")
    policy = load(policy_path, "policy")
    if result.get("case_id") != case.get("case_id") or result.get("case_sha256") != case.get("case_sha256"):
        raise TraceBindingError("result/case identity differs")
    resource_link = None
    if resource_observation_path is not None:
        resource_link = link(root, resource_observation_path, "resource observation")
    production_link = None
    production_status = "blocked"
    production_reason = "production_trace_missing"
    if production_trace_path is not None:
        production_value = load(production_trace_path, "production trace")
        validate_production_trace(production_value, case)
        production_link = link(root, production_trace_path, "production trace")
        production_status = "valid"
        production_reason = None
    terminal = {
        "audit": result.get("audit"),
        "lifecycle": result.get("lifecycle"),
        "reset": result.get("reset"),
        "outcome": result.get("outcome"),
        "fallback": result.get("fallback"),
    }
    status = "valid" if result.get("status") == "ok" and production_status == "valid" and resource_link is not None else "blocked"
    reasons = []
    if result.get("status") != "ok":
        reasons.append("result_not_ok")
    if production_reason:
        reasons.append(production_reason)
    if resource_link is None:
        reasons.append("resource_observation_missing")
    sidecar: dict[str, Any] = {
        "schema_version": "ullm.aq4_p2_execution_trace.v1",
        "status": status,
        "scope": case.get("scope"),
        "case_id": case.get("case_id"),
        "case_sha256": case.get("case_sha256"),
        "identity": link(root, identity_path, "identity"),
        "policy": link(root, policy_path, "policy"),
        "result": link(root, result_path, "result"),
        "resource_observation": resource_link,
        "production_trace": production_link,
        "terminal": terminal,
        "binding": {"run_root": str(root), "reasons": reasons, "production_trace_status": production_status},
    }
    if set(sidecar) != TRACE_FIELDS:
        raise TraceBindingError("trace sidecar root differs")
    return sidecar


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise TraceBindingError("refusing to overwrite trace sidecar")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=True, sort_keys=True, indent=2)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run_root", "case", "identity", "policy", "result", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--resource-observation", type=Path)
    parser.add_argument("--production-trace", type=Path)
    args = parser.parse_args(argv)
    try:
        value = build_trace_sidecar(args.run_root, args.case, args.identity, args.policy, args.result, args.resource_observation, args.production_trace, load(args.result, "result"))
        write_atomic(args.output, value)
        print(json.dumps({"status": value["status"], "case_id": value["case_id"], "reasons": value["binding"]["reasons"]}, sort_keys=True))
        return 0 if value["status"] == "valid" else 1
    except (TraceBindingError, OSError, ValueError) as error:
        print(f"P2 trace binding failed closed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
