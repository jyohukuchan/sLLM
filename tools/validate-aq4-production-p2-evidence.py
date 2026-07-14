#!/usr/bin/env python3
"""Independently validate P2 case/result/oracle/identity links.

The validator is intentionally CPU-safe and does not promote a component or
full-model result.  A production-server result needs an independent trace
attestation before it can ever become eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class EvidenceError(ValueError):
    pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path, label: str) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise EvidenceError(f"{label} must be a bounded regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(EvidenceError(f"non-finite number: {item}")))
    except (OSError, UnicodeError, json.JSONDecodeError, EvidenceError) as error:
        raise EvidenceError(f"cannot parse {label}: {error}") from error
    reject_nonfinite(value, label)
    return value


def reject_nonfinite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise EvidenceError(f"non-finite number: {label}")
    if isinstance(value, dict):
        for key, child in value.items(): reject_nonfinite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value): reject_nonfinite(child, f"{label}[{index}]")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"{label} must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(value: Any, label: str) -> None:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise EvidenceError(f"{label} is not a lowercase SHA-256")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    expanded = load(args.expanded, "expanded cases")
    identity = load(args.identity, "identity")
    policy = load(args.policy, "policy")
    oracle_sha = sha_file(args.source_oracle, "source oracle")
    if expanded.get("schema_version") != "ullm.aq4_production_p2_expanded.v1": failures.append("expanded_schema")
    if identity.get("schema_version") != "ullm.aq4_production_p2_identity.v1" or identity.get("status") != "bound": failures.append("identity_not_bound")
    if policy.get("status") != "bound": failures.append("policy_unbound")
    manifest_sha = expanded.get("manifest_sha256")
    require_hash(manifest_sha, "expanded.manifest_sha256")
    if identity.get("hash_binding", {}).get("bound_case_manifest_sha256") != manifest_sha: failures.append("manifest_identity_mismatch")
    if identity.get("hash_binding", {}).get("source_oracle_sha256") != oracle_sha: failures.append("source_oracle_mismatch")
    identity_hash = sha_file(args.identity, "identity")
    policy_hash = sha_file(args.policy, "policy")
    if identity.get("hash_binding", {}).get("policy_sha256") != policy.get("hash_binding", {}).get("policy_sha256"): failures.append("policy_identity_mismatch")
    cases = expanded.get("cases") if isinstance(expanded.get("cases"), list) else []
    case_by_id = {case.get("case_id"): case for case in cases if isinstance(case, dict) and isinstance(case.get("case_id"), str)}
    results: list[dict[str, Any]] = []
    result_by_case: dict[str, dict[str, Any]] = {}
    for result_path in args.result:
        try: result = load(result_path, "validation result")
        except EvidenceError as error: failures.append(f"result_parse:{result_path}:{error}"); continue
        results.append(result)
        case_id = result.get("case_id")
        if isinstance(case_id, str): result_by_case[case_id] = result
        case = case_by_id.get(case_id)
        if case is None: failures.append(f"unknown_case:{case_id}"); continue
        if result.get("schema_version") != "ullm.aq4_prefill_validation_result.v1": failures.append(f"result_schema:{case_id}")
        if result.get("case_identity", {}).get("device", {}).get("device_id") != case.get("device", {}).get("device_id"): failures.append(f"case_identity_device:{case_id}")
        raw_link = result.get("raw")
        if not isinstance(raw_link, dict) or not isinstance(raw_link.get("path"), str): failures.append(f"raw_link_missing:{case_id}")
        else:
            raw_path = Path(raw_link["path"])
            try:
                actual = sha_file(raw_path, "raw result")
                if actual != raw_link.get("sha256"): failures.append(f"raw_hash:{case_id}")
                raw = load(raw_path, "raw result")
                if raw.get("case_id") != case_id: failures.append(f"raw_case:{case_id}")
            except (EvidenceError, OSError) as error: failures.append(f"raw_unavailable:{case_id}:{error}")
        source_link = result.get("source_oracle")
        if not isinstance(source_link, dict) or source_link.get("sha256") != oracle_sha: failures.append(f"oracle_link:{case_id}")
        if case.get("phase") in {"cold_prefill", "cached_prefix_prefill"} and case.get("mode") != "all_m1":
            linked = case.get("path_oracle_case_id")
            oracle_case = case_by_id.get(linked)
            if not oracle_case or oracle_case.get("mode") != "all_m1": failures.append(f"path_oracle_case:{case_id}")
            expected_path_sha = case.get("path_oracle_result_sha256")
            if expected_path_sha is not None:
                try: require_hash(expected_path_sha, f"path_oracle_result_sha256:{case_id}")
                except EvidenceError: failures.append(f"path_oracle_hash:{case_id}")
                linked_result = result_by_case.get(linked)
                if linked_result and linked_result.get("raw", {}).get("sha256") != expected_path_sha:
                    failures.append(f"path_oracle_result_mismatch:{case_id}")
            if result.get("path_oracle", {}).get("case_id") not in {None, linked}:
                failures.append(f"path_oracle_result_link:{case_id}")
        status = result.get("status")
        if status not in {"ok", "failed", "oom", "unsupported", "skipped"}: failures.append(f"result_status:{case_id}")
        if result.get("promotion_eligible") and (case.get("scope") != "production_server" or result.get("independent_validation", {}).get("status") != "valid"):
            failures.append(f"unsafe_promotion:{case_id}")
        if case.get("scope") == "production_server" and result.get("promotion_eligible") and result.get("trace") is None:
            failures.append(f"production_trace_missing:{case_id}")
        if result.get("identity_sha256") not in {None, identity_hash}: failures.append(f"result_identity:{case_id}")
        if result.get("policy_sha256") not in {None, policy_hash}: failures.append(f"result_policy:{case_id}")
    production_eligible = bool(results) and not failures and all(result.get("promotion_eligible") is True for result in results)
    return {"schema_version": "ullm.aq4_production_p2_evidence_validator.v1", "status": "valid" if not failures else "invalid", "failure_codes": sorted(set(failures)), "expanded_case_count": len(cases), "result_count": len(results), "source_oracle_sha256": oracle_sha, "identity_sha256": identity_hash, "policy_file_sha256": policy_hash, "promotion_eligible": production_eligible, "production_live_execution": False, "review_note": "CPU/synthetic evidence only; production/live requires parent P1 gate and real trace"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--expanded", type=Path, required=True); parser.add_argument("--identity", type=Path, required=True); parser.add_argument("--policy", type=Path, required=True); parser.add_argument("--source-oracle", type=Path, required=True); parser.add_argument("--result", type=Path, action="append", required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = validate(args)
        if args.output.exists() or args.output.is_symlink(): raise EvidenceError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True); temporary = args.output.with_name(f".{args.output.name}.incomplete")
        with temporary.open("xb") as target:
            target.write((json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode()); target.flush(); os.fsync(target.fileno())
        temporary.replace(args.output)
        print(json.dumps({"status": report["status"], "failure_count": len(report["failure_codes"])}, sort_keys=True)); return 0 if report["status"] == "valid" else 1
    except (EvidenceError, OSError, ValueError) as error:
        print(f"P2 evidence validation failed closed: {error}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
