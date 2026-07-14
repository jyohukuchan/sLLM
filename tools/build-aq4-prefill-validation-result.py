#!/usr/bin/env python3
"""Build one privacy-safe, hash-linked P2 prefill validation result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


class ResultError(ValueError):
    pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ResultError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path, label: str) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ResultError(f"{label} must be a bounded regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(ResultError(f"non-finite number: {item}")))
    except (OSError, UnicodeError, json.JSONDecodeError, ResultError) as error:
        raise ResultError(f"cannot parse {label}: {error}") from error
    return value


def sha_file(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ResultError(f"{label} must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_write(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise ResultError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    raw = (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode()
    with temporary.open("xb") as target:
        target.write(raw); target.flush(); os.fsync(target.fileno())
    temporary.replace(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    case = load(args.case, "case")
    expanded = load(args.expanded, "expanded cases")
    raw = load(args.raw, "raw result")
    identity = load(args.identity, "identity") if args.identity else None
    policy = load(args.policy, "policy") if args.policy else None
    oracle_value = load(args.source_oracle, "source oracle")
    if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
        raise ResultError("case must have a case_id")
    cases = expanded.get("cases") if isinstance(expanded, dict) else None
    if not isinstance(cases, list) or not any(isinstance(value, dict) and value.get("case_id") == case["case_id"] for value in cases):
        raise ResultError("case is not present in expanded manifest")
    if not isinstance(raw, dict) or raw.get("case_id") != case["case_id"]:
        raise ResultError("raw result case identity mismatch")
    oracle_sha = sha_file(args.source_oracle, "source oracle")
    if identity and identity.get("hash_binding", {}).get("source_oracle_sha256") != oracle_sha:
        raise ResultError("source oracle hash does not match identity")
    raw_sha = sha_file(args.raw, "raw result")
    trace_link = None
    if args.trace:
        trace_link = {"path": str(args.trace), "sha256": sha_file(args.trace, "trace")}
    status = args.status or raw.get("status")
    if status not in {"ok", "failed", "oom", "unsupported", "skipped"}:
        raise ResultError("invalid result status")
    elapsed = args.elapsed_ms if args.elapsed_ms is not None else raw.get("elapsed_ms")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or not math.isfinite(float(elapsed)) or elapsed < 0:
        raise ResultError("elapsed_ms must be finite and non-negative")
    throughput = None
    prompt_tokens = case.get("prompt_tokens")
    if prompt_tokens and elapsed > 0:
        throughput = float(prompt_tokens) / (float(elapsed) / 1000.0)
    production = case.get("scope") == "production_server"
    eligible = bool(production and status == "ok" and trace_link and identity and identity.get("status") == "bound" and policy and policy.get("status") == "bound" and args.independent_valid)
    path_oracle = None
    if case.get("path_oracle_case_id"):
        path_oracle = {"case_id": case["path_oracle_case_id"], "result_sha256": case.get("path_oracle_result_sha256")}
    result = {
        "schema_version": "ullm.aq4_prefill_validation_result.v1", "case_id": case["case_id"],
        "case_identity": {key: case.get(key) for key in ("stage_id", "scope", "phase", "mode", "prompt_tokens", "cached_prefix_tokens", "context_tokens", "decode_start_tokens", "prefill_requested_m", "decode_request_count", "device", "control_id", "format_id")},
        "status": status, "raw": {"path": str(args.raw), "sha256": raw_sha},
        "source_oracle": {"path": str(args.source_oracle), "sha256": oracle_sha, "identity": sha_file(args.source_oracle, "source oracle")},
        "trace": trace_link, "measurement": {"elapsed_ms": float(elapsed), "prefill_tokens_per_second": throughput, "prompt_tokens": prompt_tokens},
        "path_oracle": path_oracle,
        "preflight": raw.get("preflight"), "correctness": {"finite_outputs": status == "ok", "source_oracle_consumed": True, "independent_validation": "valid" if args.independent_valid else "not_run"},
        "independent_validation": {"status": "valid" if args.independent_valid else "not_run", "validator": args.validator or None},
        "promotion_eligible": eligible, "promotion_reason": "production_server independent trace and bound policy" if eligible else "CPU/component evidence or independent trace gate is incomplete",
        "identity_sha256": sha_file(args.identity, "identity") if args.identity else None, "policy_sha256": sha_file(args.policy, "policy") if args.policy else None,
        "privacy_contract": {"prompt_content_included": False, "raw_stream_content_included": False, "bounded_artifact_links": True},
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True); parser.add_argument("--expanded", type=Path, required=True); parser.add_argument("--raw", type=Path, required=True); parser.add_argument("--source-oracle", type=Path, required=True)
    parser.add_argument("--identity", type=Path); parser.add_argument("--policy", type=Path); parser.add_argument("--trace", type=Path); parser.add_argument("--status", choices=("ok", "failed", "oom", "unsupported", "skipped")); parser.add_argument("--elapsed-ms", type=float); parser.add_argument("--independent-valid", action="store_true"); parser.add_argument("--validator"); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build(args); atomic_write(args.output, result); print(json.dumps({"status": "ok", "promotion_eligible": result["promotion_eligible"]}, sort_keys=True)); return 0
    except (ResultError, OSError, ValueError) as error:
        print(f"P2 validation result failed: {error}", file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
