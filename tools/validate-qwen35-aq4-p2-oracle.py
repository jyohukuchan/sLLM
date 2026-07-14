#!/usr/bin/env python3
"""Independently validate Qwen3.5-9B AQ4 P2 source/path oracle evidence."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen35_aq4_p2_oracle as oracle  # noqa: E402


def _sha(path: Path) -> str:
    return oracle.sha256_file(path)


def _rehash_files(root_raw: str, files: list[dict[str, Any]], label: str) -> None:
    root = Path(root_raw)
    if root.is_symlink() or not root.is_dir():
        raise oracle.OracleError(f"{label} root is missing, symlinked, or not a directory")
    for entry in files:
        path = oracle.safe_relative(root, entry["file"], f"{label} {entry['file']}")
        if path.stat().st_size != entry["bytes"] or _sha(path) != entry["sha256"]:
            raise oracle.OracleError(f"{label} file identity differs: {entry['file']}")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _validate_sha256s(root: Path, manifest: dict[str, Any]) -> None:
    sums_path = oracle.safe_relative(root, "SHA256SUMS", "SHA256SUMS")
    expected_names = {"manifest.json", manifest["payload"]["file"], "runtime.json"}
    entries: dict[str, str] = {}
    for line_number, line in enumerate(sums_path.read_text(encoding="ascii").splitlines(), 1):
        parts = line.split("  ")
        if len(parts) != 2 or parts[1] in entries:
            raise oracle.OracleError(f"SHA256SUMS line {line_number} is invalid or duplicate")
        oracle.ensure_sha256(parts[0], f"SHA256SUMS line {line_number}")
        entries[parts[1]] = parts[0]
    if set(entries) != expected_names:
        raise oracle.OracleError("SHA256SUMS coverage differs")
    actual_names = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise oracle.OracleError("oracle root contains a non-regular artifact")
        actual_names.add(path.name)
    if actual_names != expected_names | {"SHA256SUMS"}:
        raise oracle.OracleError("oracle root file coverage differs from SHA256SUMS")
    for name, digest in entries.items():
        path = oracle.safe_relative(root, name, f"SHA256SUMS target {name}")
        if _sha(path) != digest:
            raise oracle.OracleError(f"SHA256SUMS digest differs: {name}")


def _validate_runtime(root: Path, manifest: dict[str, Any]) -> None:
    runtime_path = oracle.safe_relative(root, "runtime.json", "runtime.json")
    runtime = oracle.load_json(runtime_path)
    if runtime != manifest.get("runtime"):
        raise oracle.OracleError("manifest runtime and runtime.json differ")
    expected_keys = {"device", "dtype", "full_vocab_ranking", "inference_mode", "low_cpu_mem_usage", "low_cpu_mem_usage_blocker", "max_resident_logit_rows", "model_loads", "preflight", "python", "run", "runtime", "safetensors", "torch", "torch_num_interop_threads", "torch_num_threads", "transformers"}
    if not isinstance(runtime, dict) or set(runtime) != expected_keys:
        raise oracle.OracleError("runtime keys differ")
    if runtime["runtime"] != "transformers.AutoModelForCausalLM" or runtime["device"] != "cpu" or runtime["dtype"] != "bfloat16":
        raise oracle.OracleError("runtime CPU/BF16 identity differs")
    if runtime["low_cpu_mem_usage"] is not False or runtime["low_cpu_mem_usage_blocker"] != "accelerate package is unavailable in the installed environment":
        raise oracle.OracleError("runtime low-memory loader status differs")
    if runtime["inference_mode"] is not True or runtime["full_vocab_ranking"] is not True or runtime["max_resident_logit_rows"] != 1 or runtime["model_loads"] != 1:
        raise oracle.OracleError("runtime bounded forward contract differs")
    if runtime["torch_num_threads"] != 1 or runtime["torch_num_interop_threads"] != 1:
        raise oracle.OracleError("runtime thread count differs")
    if runtime["python"] != platform.python_version() or runtime["torch"] != _package_version("torch") or runtime["transformers"] != _package_version("transformers") or runtime["safetensors"] != _package_version("safetensors"):
        raise oracle.OracleError("runtime package versions differ from validator environment")
    preflight = runtime["preflight"]
    expected_preflight = {"checkpoint_bytes", "headroom_factor", "mem_available_bytes", "mem_total_bytes", "required_headroom_bytes", "status"}
    if not isinstance(preflight, dict) or set(preflight) != expected_preflight:
        raise oracle.OracleError("runtime preflight keys differ")
    checkpoint_bytes = sum(entry["bytes"] for entry in manifest["identity"]["source_checkpoint"]["files"] if entry["file"].endswith(".safetensors"))
    if preflight["checkpoint_bytes"] != checkpoint_bytes or preflight["headroom_factor"] != 1.5 or preflight["required_headroom_bytes"] != int(checkpoint_bytes * 1.5) or preflight["status"] != "passed":
        raise oracle.OracleError("runtime preflight checkpoint arithmetic differs")
    if not isinstance(preflight["mem_available_bytes"], int) or not isinstance(preflight["mem_total_bytes"], int) or preflight["mem_available_bytes"] < preflight["required_headroom_bytes"] or preflight["mem_total_bytes"] < preflight["mem_available_bytes"]:
        raise oracle.OracleError("runtime preflight memory observation is invalid")
    run = runtime["run"]
    if not isinstance(run, dict) or set(run) != {"elapsed_seconds", "row_count"} or run["row_count"] != manifest["payload"]["record_count"] or isinstance(run["elapsed_seconds"], bool) or not isinstance(run["elapsed_seconds"], (int, float)) or not math.isfinite(run["elapsed_seconds"]) or run["elapsed_seconds"] <= 0:
        raise oracle.OracleError("runtime run summary differs")
    _validate_sha256s(root, manifest)


def validate_oracle(root: Path, kind: str) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise oracle.OracleError("oracle root must be a regular directory, not a symlink")
    manifest = oracle.validate_manifest(root, expected_kind=kind)
    blockers: list[str] = []
    if manifest["evidence_class"] == "synthetic_fixture":
        blockers.append("synthetic fixture is not an independent production oracle")
    if kind == "source" and manifest["identity"]["model_revision"] is None:
        blockers.append("source checkpoint revision metadata is unavailable or inconsistent")
    if kind == "path":
        if manifest["identity"]["artifact"]["package_manifest_sha256"] is None:
            raise oracle.OracleError("path oracle must bind a package manifest")
        if manifest["identity"]["artifact"]["artifact_manifest_sha256"] is None:
            blockers.append("path oracle is package-bound but the active product has no artifact manifest")
    tokenizer = manifest["identity"]["tokenizer"]
    if {entry["file"] for entry in tokenizer["files"]} != set(oracle.TOKENIZER_FILES):
        raise oracle.OracleError("tokenizer file coverage differs")
    _rehash_files(tokenizer["root"], tokenizer["files"], "tokenizer")
    if kind == "source":
        source_checkpoint = manifest["identity"]["source_checkpoint"]
        _rehash_files(source_checkpoint["root"], source_checkpoint["files"], "source checkpoint")
        checkpoint_root = Path(source_checkpoint["root"])
        index = oracle.load_json(oracle.safe_relative(checkpoint_root, "model.safetensors.index.json", "source checkpoint index"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise oracle.OracleError("source checkpoint index weight map is invalid")
        shards = set(weight_map.values())
        if len(shards) != 4 or any(not isinstance(name, str) for name in shards):
            raise oracle.OracleError("source checkpoint must contain exactly four indexed shards")
        expected_checkpoint_files = {"config.json", "model.safetensors.index.json", *shards}
        if {entry["file"] for entry in source_checkpoint["files"]} != expected_checkpoint_files:
            raise oracle.OracleError("source checkpoint file coverage differs")
        if manifest["evidence_class"] == "production":
            _validate_runtime(root, manifest)
    usable_key = "usable_as_source_evidence" if kind == "source" else "usable_as_path_evidence"
    usable = manifest[usable_key] and manifest["status"] == "available" and manifest["evidence_class"] == "production" and not blockers
    return {
        "schema_version": "ullm.qwen35_aq4_p2_oracle_validator.v1",
        "status": "valid",
        "oracle_kind": manifest["oracle_kind"],
        "manifest_sha256": _sha(root / "manifest.json"),
        "payload_sha256": manifest["payload"]["sha256"],
        "record_count": manifest["payload"]["record_count"],
        usable_key: usable,
        "promotion_eligible": False,
        "blockers": blockers,
    }


def validate_link(root: Path, source_root: Path, path_root: Path) -> dict[str, Any]:
    source_report = validate_oracle(source_root, "source")
    path_report = validate_oracle(path_root, "path")
    source = oracle.validate_manifest(source_root, expected_kind="source")
    path = oracle.validate_manifest(path_root, expected_kind="path")
    link = oracle.load_json(root / "manifest.json")
    if not isinstance(link, dict):
        raise oracle.OracleError("link manifest must be an object")
    expected = {"agreement", "created_utc", "evidence_class", "identity", "path", "promotion_eligible", "schema_version", "source", "status", "usable_as_p2_oracle_link"}
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
            if entry["artifact_manifest_sha256"] is not None:
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
    if path["identity"]["artifact"]["artifact_manifest_sha256"] is None:
        blockers.append("path oracle is package-bound but the active product has no artifact manifest")
    if not agreement["greedy_token_exact"] or not agreement["topk_exact"] or not agreement["hidden_sample_within_atol"] or not agreement["logit_sample_within_atol"]:
        blockers.append("source/path bounded agreement gate failed")
    expected_usable = bool(link["evidence_class"] == "production" and source_report["usable_as_source_evidence"] and path_report["usable_as_path_evidence"] and not blockers)
    if link["usable_as_p2_oracle_link"] is not expected_usable:
        raise oracle.OracleError("link usable_as_p2_oracle_link differs from recomputed agreement")
    return {"schema_version": "ullm.qwen35_aq4_p2_oracle_link_validator.v1", "status": "valid", "manifest_sha256": _sha(root / "manifest.json"), "source_manifest_sha256": link["source"]["manifest_sha256"], "path_manifest_sha256": link["path"]["manifest_sha256"], "agreement": agreement, "usable_as_p2_oracle_link": expected_usable, "promotion_eligible": False, "blockers": blockers}


def probe_source(root: Path, payload: Path | None) -> dict[str, Any]:
    try:
        identity = oracle.inspect_source_model(root)
        source_status = "available"
        source_error = None
    except oracle.OracleError as error:
        identity = None
        source_status = "blocked"
        source_error = str(error)
    forward_status = "available" if payload is not None and payload.is_file() and not payload.is_symlink() else "blocked"
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
