#!/usr/bin/env python3
"""Independently validate Qwen3.5-9B AQ4 P2 source/path oracle evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen35_aq4_p2_oracle as oracle  # noqa: E402


def _sha(path: Path) -> str:
    return oracle.sha256_file(path)


def validate_oracle(root: Path, kind: str) -> dict[str, Any]:
    manifest = oracle.validate_manifest(root, expected_kind=kind)
    production_eligible = manifest["promotion_eligible"] and manifest["status"] == "available" and manifest["evidence_class"] == "production"
    blockers: list[str] = []
    if manifest["evidence_class"] == "synthetic_fixture":
        blockers.append("synthetic fixture is not an independent production oracle")
    if kind == "source" and manifest["identity"]["model_revision"] is None:
        blockers.append("source checkpoint revision metadata is unavailable or inconsistent")
    if kind == "path" and manifest["identity"]["artifact"]["artifact_manifest_sha256"] is None:
        blockers.append("path oracle is not bound to an artifact manifest")
    return {
        "schema_version": "ullm.qwen35_aq4_p2_oracle_validator.v1",
        "status": "valid",
        "oracle_kind": manifest["oracle_kind"],
        "manifest_sha256": _sha(root / "manifest.json"),
        "payload_sha256": manifest["payload"]["sha256"],
        "record_count": manifest["payload"]["record_count"],
        "production_eligible": production_eligible and not blockers,
        "blockers": blockers,
    }


def validate_link(root: Path, source_root: Path, path_root: Path) -> dict[str, Any]:
    source = oracle.validate_manifest(source_root, expected_kind="source")
    path = oracle.validate_manifest(path_root, expected_kind="path")
    link = oracle.load_json(root / "manifest.json")
    if not isinstance(link, dict):
        raise oracle.OracleError("link manifest must be an object")
    expected = {"agreement", "created_utc", "evidence_class", "identity", "path", "promotion_eligible", "schema_version", "source", "status"}
    if set(link) != expected:
        raise oracle.OracleError("link manifest keys differ")
    if link["schema_version"] != oracle.LINK_SCHEMA or link["status"] not in {"available", "fixture"}:
        raise oracle.OracleError("link schema or status is invalid")
    if link["evidence_class"] not in {"production", "synthetic_fixture"}:
        raise oracle.OracleError("link evidence_class is invalid")
    oracle.validate_utc(link["created_utc"])
    identity = link["identity"]
    if not isinstance(identity, dict) or set(identity) != {"model_id", "model_revision", "tokenizer_aggregate_sha256"}:
        raise oracle.OracleError("link identity keys differ")
    oracle.ensure_sha256(identity["tokenizer_aggregate_sha256"], "link tokenizer aggregate")
    if identity["model_id"] != source["identity"]["model_id"] or identity["model_id"] != path["identity"]["model_id"] or identity["model_revision"] != source["identity"]["model_revision"] or identity["model_revision"] != path["identity"]["model_revision"]:
        raise oracle.OracleError("link model identity differs")
    if identity["tokenizer_aggregate_sha256"] != source["identity"]["tokenizer"]["aggregate_sha256"] or identity["tokenizer_aggregate_sha256"] != path["identity"]["tokenizer"]["aggregate_sha256"]:
        raise oracle.OracleError("link tokenizer identity differs")
    for key, expected_root, manifest in (("source", source_root, source), ("path", path_root, path)):
        entry = link[key]
        if not isinstance(entry, dict) or set(entry) != ({"manifest_sha256", "payload_sha256"} if key == "source" else {"artifact_manifest_sha256", "manifest_sha256", "package_manifest_sha256", "payload_sha256"}):
            raise oracle.OracleError(f"link {key} keys differ")
        if entry["manifest_sha256"] != _sha(expected_root / "manifest.json") or entry["payload_sha256"] != manifest["payload"]["sha256"]:
            raise oracle.OracleError(f"link {key} hash binding differs")
        if key == "path":
            if entry["artifact_manifest_sha256"] != manifest["identity"]["artifact"]["artifact_manifest_sha256"] or entry["package_manifest_sha256"] != manifest["identity"]["artifact"]["package_manifest_sha256"]:
                raise oracle.OracleError("link path artifact binding differs")
            oracle.ensure_sha256(entry["artifact_manifest_sha256"], "link path artifact hash")
            oracle.ensure_sha256(entry["package_manifest_sha256"], "link path package hash")
    agreement = link["agreement"]
    if not isinstance(agreement, dict) or agreement != oracle.compare_payloads(source_root, source, path_root, path):
        raise oracle.OracleError("link agreement differs from bounded payload comparison")
    if link["promotion_eligible"] is not False:
        raise oracle.OracleError("source/path link must remain non-promotable until production policy accepts it")
    blockers = []
    if link["evidence_class"] == "synthetic_fixture":
        blockers.append("source/path link contains synthetic fixture evidence")
    if not agreement["greedy_token_exact"] or not agreement["topk_exact"] or not agreement["hidden_sample_within_atol"] or not agreement["logit_sample_within_atol"]:
        blockers.append("source/path bounded agreement gate failed")
    return {"schema_version": "ullm.qwen35_aq4_p2_oracle_link_validator.v1", "status": "valid", "manifest_sha256": _sha(root / "manifest.json"), "source_manifest_sha256": link["source"]["manifest_sha256"], "path_manifest_sha256": link["path"]["manifest_sha256"], "agreement": agreement, "promotion_eligible": False, "blockers": blockers}


def probe_source(root: Path, payload: Path | None) -> dict[str, Any]:
    try:
        identity = oracle.inspect_source_model(root)
        source_status = "available"
        source_error = None
    except oracle.OracleError as error:
        identity = None
        source_status = "blocked"
        source_error = str(error)
    forward_status = "available" if payload is not None and payload.is_file() else "blocked"
    blocker = None if forward_status == "available" else "independent BF16/F32 forward summaries are absent; checkpoint metadata alone is not an oracle"
    return {"schema_version": "ullm.qwen35_aq4_source_probe.v1", "status": "valid", "source_model": {"status": source_status, "identity": identity, "error": source_error}, "independent_forward_artifact": {"status": forward_status, "payload": str(payload) if payload else None, "blocker": blocker}, "production_oracle_available": source_status == "available" and forward_status == "available"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_oracle = sub.add_parser("oracle")
    p_oracle.add_argument("root", type=Path)
    p_oracle.add_argument("--kind", choices=("source", "path"), required=True)
    p_oracle.add_argument("--output", type=Path)
    p_link = sub.add_parser("link")
    p_link.add_argument("root", type=Path)
    p_link.add_argument("--source-oracle", type=Path, required=True)
    p_link.add_argument("--path-oracle", type=Path, required=True)
    p_link.add_argument("--output", type=Path)
    p_probe = sub.add_parser("probe")
    p_probe.add_argument("--source-root", type=Path, required=True)
    p_probe.add_argument("--payload", type=Path)
    p_probe.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "oracle":
            report = validate_oracle(args.root, args.kind)
        elif args.command == "link":
            report = validate_link(args.root, args.source_oracle, args.path_oracle)
        else:
            report = probe_source(args.source_root, args.payload)
        raw = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        if args.output:
            if args.output.exists() or args.output.is_symlink():
                raise oracle.OracleError(f"refusing to overwrite report: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(raw, encoding="utf-8")
        else:
            print(raw, end="")
        return 0
    except (oracle.OracleError, OSError, ValueError) as error:
        print(f"Qwen3.5 AQ4 P2 oracle validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
