#!/usr/bin/env python3
"""Capture or link bounded Qwen3.5-9B AQ4 P2 source/path oracle evidence.

The command accepts an externally produced JSONL stream.  It never runs a
model, so an absent independent BF16/F32 forward artifact fails closed rather
than being replaced by an AQ4 or same-artifact result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen35_aq4_p2_oracle as oracle  # noqa: E402


def _json(path: Path) -> Any:
    return oracle.load_json(path)


def _sha(path: Path) -> str:
    return oracle.sha256_file(path)


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise oracle.OracleError(f"{label} must be a regular non-symlink file")
    return path


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _tokenizer_identity(files: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    normalized = sorted(
        ({"file": item["file"], "bytes": int(item["bytes"]), "sha256": item["sha256"]} for item in files),
        key=lambda item: item["file"],
    )
    return {"files": normalized, "aggregate_sha256": oracle.canonical_sha256(normalized), "root": str(root.resolve(strict=True))}


def _load_cases(path: Path) -> list[dict[str, Any]]:
    value = _json(_regular(path, "cases"))
    if not isinstance(value, dict) or set(value) != {"cases"} or not isinstance(value["cases"], list):
        raise oracle.OracleError("cases JSON must be {\"cases\": [...]} with exact keys")
    if not value["cases"] or len(value["cases"]) > oracle.MAX_CASES:
        raise oracle.OracleError("cases exceed bounded contract")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value["cases"]):
        if not isinstance(raw, dict) or set(raw) != {"case_id", "prompt_token_ids", "step_count"}:
            raise oracle.OracleError(f"cases[{index}] keys differ")
        case_id = raw["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise oracle.OracleError("case IDs must be unique non-empty strings")
        seen.add(case_id)
        token_ids = raw["prompt_token_ids"]
        if not isinstance(token_ids, list) or not token_ids or len(token_ids) > 4096:
            raise oracle.OracleError(f"cases[{index}].prompt_token_ids is invalid")
        for token_id in token_ids:
            oracle.integer(token_id, f"cases[{index}].prompt_token_ids", minimum=0)
        step_count = oracle.integer(raw["step_count"], f"cases[{index}].step_count", minimum=1)
        if step_count > oracle.MAX_STEPS:
            raise oracle.OracleError("step_count exceeds bounded contract")
        result.append({"case_id": case_id, "prompt_token_count": len(token_ids), "prompt_token_ids_sha256": oracle.canonical_token_ids_hash(token_ids), "step_count": step_count})
    return result


def _copy_payload(source: Path, destination: Path, cases: list[dict[str, Any]]) -> tuple[str, int, int]:
    _regular(source, "payload input")
    expected = {(case["case_id"], step) for case in cases for step in range(case["step_count"])}
    seen: set[tuple[str, int]] = set()
    digest = hashlib.sha256()
    size = 0
    records = 0
    with source.open("r", encoding="utf-8") as incoming, destination.open("w", encoding="utf-8") as outgoing:
        for line_number, line in enumerate(incoming, 1):
            if not line.strip():
                raise oracle.OracleError(f"payload input line {line_number} is empty")
            try:
                raw = json.loads(line, object_pairs_hook=oracle.reject_duplicate_keys, parse_constant=oracle.reject_nonfinite)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise oracle.OracleError(f"invalid payload input line {line_number}: {error}") from error
            record = oracle.validate_payload_record(raw, f"payload input[{line_number}]")
            key = (record["case_id"], record["step"])
            if key not in expected or key in seen:
                raise oracle.OracleError("payload input case/step coverage is invalid")
            seen.add(key)
            encoded = _canonical(record)
            size += len(encoded)
            if size > oracle.MAX_PAYLOAD_BYTES:
                raise oracle.OracleError("payload exceeds bounded limit")
            outgoing.write(encoded.decode("utf-8"))
            digest.update(encoded)
            records += 1
            if records > oracle.MAX_CASES * oracle.MAX_STEPS:
                raise oracle.OracleError("payload record count exceeds bounded limit")
    if seen != expected:
        raise oracle.OracleError("payload input is missing a case/step record")
    return digest.hexdigest(), size, records


def _source_identity(source_root: Path) -> tuple[str, dict[str, Any]]:
    inspected = oracle.inspect_source_model(source_root)
    tokenizer = _tokenizer_identity(inspected["tokenizer_files"], source_root)
    return inspected["model_id"], {"artifact": {"package_manifest_sha256": None, "artifact_manifest_sha256": None}, "model_id": inspected["model_id"], "model_revision": inspected["revision"], "source_checkpoint": oracle.source_checkpoint_identity(inspected), "tokenizer": tokenizer}


def _path_identity(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer_root = Path(args.tokenizer_root) if args.tokenizer_root else None
    if tokenizer_root is None or tokenizer_root.is_symlink() or not tokenizer_root.is_dir():
        raise oracle.OracleError("path oracle requires --tokenizer-root")
    tokenizer_files = []
    for name in args.tokenizer_file:
        path = oracle.safe_relative(tokenizer_root, name, "tokenizer file")
        tokenizer_files.append({"file": name, "bytes": path.stat().st_size, "sha256": _sha(path)})
    package = _regular(Path(args.package_manifest), "package manifest")
    artifact_raw = getattr(args, "artifact_manifest", None)
    artifact = _regular(Path(artifact_raw), "artifact manifest") if artifact_raw is not None else None
    if artifact is not None and artifact.resolve() == package.resolve():
        raise oracle.OracleError("artifact and package manifests must be distinct files")
    model_id = args.model_id
    if not model_id:
        raise oracle.OracleError("path oracle requires --model-id")
    return {"artifact": {"package_manifest_sha256": _sha(package), "artifact_manifest_sha256": _sha(artifact) if artifact is not None else None}, "model_id": model_id, "model_revision": args.model_revision, "source_checkpoint": None, "tokenizer": _tokenizer_identity(tokenizer_files, tokenizer_root)}


def _write_manifest(output: Path, manifest: dict[str, Any], payload_src: Path | None = None) -> None:
    if os.path.lexists(output):
        raise oracle.OracleError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete")
    if os.path.lexists(temporary):
        raise oracle.OracleError(f"incomplete output already exists: {temporary}")
    temporary.mkdir()
    try:
        if payload_src is not None:
            shutil.copyfile(payload_src, temporary / "payload.jsonl")
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        oracle.validate_manifest(temporary)
        os.rename(temporary, output)
    except Exception:
        # Keep incomplete evidence visible and never make a partial oracle look complete.
        raise


def capture(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or os.path.lexists(args.output):
        raise oracle.OracleError(f"refusing to overwrite existing output: {args.output}")
    cases = _load_cases(Path(args.cases))
    with __import__("tempfile").TemporaryDirectory(prefix="qwen35-aq4-oracle-") as temporary:
        payload_path = Path(temporary) / "payload.jsonl"
        payload_sha, payload_bytes, record_count = _copy_payload(Path(args.payload), payload_path, cases)
        if args.kind == "source":
            _, identity = _source_identity(Path(args.source_root))
        else:
            identity = _path_identity(args)
        evidence_class = args.evidence_class
        status = "fixture" if evidence_class == "synthetic_fixture" else "available"
        usable = evidence_class == "production" and (
            args.kind == "source" or identity["artifact"]["artifact_manifest_sha256"] is not None
        )
        manifest = {
            "schema_version": oracle.SCHEMAS[args.kind],
            "oracle_kind": "independent_source" if args.kind == "source" else "same_artifact_all_m1",
            "status": status,
            "evidence_class": evidence_class,
            ("usable_as_source_evidence" if args.kind == "source" else "usable_as_path_evidence"): usable,
            "created_utc": oracle.utc_now(),
            "identity": identity,
            "ranking": oracle.RANKING_CONTRACT,
            "limits": {"max_cases": oracle.MAX_CASES, "max_payload_bytes": oracle.MAX_PAYLOAD_BYTES, "max_sample_values": oracle.MAX_SAMPLE_VALUES, "max_steps": oracle.MAX_STEPS, "max_top_k": oracle.MAX_TOP_K},
            "cases": cases,
            "payload": {"file": "payload.jsonl", "bytes": payload_bytes, "record_count": record_count, "sha256": payload_sha},
        }
        _write_manifest(args.output, manifest, payload_path)
    return oracle.validate_manifest(args.output)


def link(args: argparse.Namespace) -> dict[str, Any]:
    source_root, path_root = Path(args.source_oracle), Path(args.path_oracle)
    source = oracle.validate_manifest(source_root, expected_kind="independent_source")
    path = oracle.validate_manifest(path_root, expected_kind="same_artifact_all_m1")
    if source["identity"]["model_id"] != path["identity"]["model_id"] or source["identity"]["model_revision"] != path["identity"]["model_revision"]:
        raise oracle.OracleError("source/path model identity differs")
    if source["identity"]["tokenizer"]["aggregate_sha256"] != path["identity"]["tokenizer"]["aggregate_sha256"] or source["identity"]["tokenizer"]["files"] != path["identity"]["tokenizer"]["files"]:
        raise oracle.OracleError("source/path tokenizer identity differs")
    if path["identity"]["artifact"]["package_manifest_sha256"] is None:
        raise oracle.OracleError("path oracle must bind the package manifest")
    agreement = oracle.compare_payloads(source_root, source, path_root, path)
    manifest = {
        "schema_version": oracle.LINK_SCHEMA,
        "status": "fixture" if source["evidence_class"] == "synthetic_fixture" or path["evidence_class"] == "synthetic_fixture" else "available",
        "evidence_class": "synthetic_fixture" if source["evidence_class"] == "synthetic_fixture" or path["evidence_class"] == "synthetic_fixture" else "production",
        "created_utc": oracle.utc_now(),
        "identity": {"model_id": source["identity"]["model_id"], "model_revision": source["identity"]["model_revision"], "tokenizer_aggregate_sha256": source["identity"]["tokenizer"]["aggregate_sha256"]},
        "source": {"manifest_sha256": _sha(source_root / "manifest.json"), "payload_sha256": source["payload"]["sha256"]},
        "path": {"manifest_sha256": _sha(path_root / "manifest.json"), "payload_sha256": path["payload"]["sha256"], "artifact_manifest_sha256": path["identity"]["artifact"]["artifact_manifest_sha256"], "package_manifest_sha256": path["identity"]["artifact"]["package_manifest_sha256"]},
        "agreement": agreement,
        "usable_as_p2_oracle_link": False,
        "promotion_eligible": False,
    }
    manifest["usable_as_p2_oracle_link"] = bool(manifest["evidence_class"] == "production" and source["identity"]["model_revision"] is not None and source["usable_as_source_evidence"] and path["usable_as_path_evidence"] and agreement["greedy_token_exact"] and agreement["topk_exact"] and agreement["hidden_sample_within_atol"] and agreement["logit_sample_within_atol"])
    if os.path.lexists(args.output):
        raise oracle.OracleError(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.incomplete")
    if os.path.lexists(temporary):
        raise oracle.OracleError(f"incomplete output already exists: {temporary}")
    temporary.mkdir()
    try:
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.rename(temporary, args.output)
    except Exception:
        raise
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture_parser = sub.add_parser("capture")
    capture_parser.add_argument("--kind", choices=("source", "path"), required=True)
    capture_parser.add_argument("--payload", type=Path, required=True)
    capture_parser.add_argument("--cases", type=Path, required=True)
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--evidence-class", choices=("production", "synthetic_fixture"), default="production")
    capture_parser.add_argument("--source-root", type=Path)
    capture_parser.add_argument("--tokenizer-root", type=Path)
    capture_parser.add_argument("--tokenizer-file", action="append", default=list(oracle.TOKENIZER_FILES))
    capture_parser.add_argument("--artifact-manifest", type=Path)
    capture_parser.add_argument("--package-manifest", type=Path)
    capture_parser.add_argument("--allow-package-only", action="store_true", help="explicitly allow a product with no artifact manifest")
    capture_parser.add_argument("--model-id")
    capture_parser.add_argument("--model-revision")
    link_parser = sub.add_parser("link")
    link_parser.add_argument("--source-oracle", type=Path, required=True)
    link_parser.add_argument("--path-oracle", type=Path, required=True)
    link_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            if args.kind == "source" and args.source_root is None:
                raise oracle.OracleError("source capture requires --source-root")
            if args.kind == "path" and args.package_manifest is None:
                raise oracle.OracleError("path capture requires --package-manifest")
            if args.kind == "path" and args.artifact_manifest is None and not args.allow_package_only:
                raise oracle.OracleError("path capture without --artifact-manifest requires --allow-package-only")
            result = capture(args)
        else:
            result = link(args)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0
    except (oracle.OracleError, OSError, ValueError) as error:
        print(f"Qwen3.5 AQ4 P2 oracle capture failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
