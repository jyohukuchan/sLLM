#!/usr/bin/env python3
"""Verify that a layer-0 hybrid fixture uses the served package's embedding rows.

The Phase 1 CPU hybrid diagnostic originally derives f32 residual rows from the
source BF16 safetensors embedding.  Phase 3c compares that AQ4 CPU diagnostic
with the production GPU M=1 path, whose layer-0 input comes from the package's
BF16 passthrough embedding.  This CPU-only checker verifies every referenced
row bit-for-bit after BF16-to-f32 expansion, without reading the whole embedding
payload into memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


HYBRID_SCHEMA = "ullm.aq4_layer0_hybrid_input_jsonl.v1"
RECEIPT_SCHEMA = "ullm.aq4_layer0_package_embedding_fixture_verification.v1"
EMBEDDING_TENSOR = "model.language_model.embed_tokens.weight"
HIDDEN = 4096
MAX_INPUT_BYTES = 1024 * 1024
MAX_CASES = 128
MAX_CONTEXT = 512


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_token_ids_hash(token_ids: list[int]) -> str:
    return sha256_bytes((json.dumps(token_ids, separators=(",", ":")) + "\n").encode("ascii"))


def safe_relative(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("fixture sidecar path must be relative and must not traverse parents")
    candidate = (root / path).resolve()
    if root.resolve() not in candidate.parents:
        raise ValueError("fixture sidecar path escapes the hybrid input root")
    return candidate


def read_hybrid_input(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = path.read_bytes()
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("hybrid input exceeds byte bound")
    lines = [line for line in raw.splitlines() if line]
    if not lines:
        raise ValueError("hybrid input is empty")
    header = json.loads(lines[0])
    expected_header = {
        "kind",
        "schema_version",
        "tensor_name",
        "dtype",
        "shape",
        "residual_encoding",
        "source_model_index_sha256",
    }
    if not isinstance(header, dict) or set(header) != expected_header:
        raise ValueError("hybrid input header fields differ")
    if (
        header["kind"] != "header"
        or header["schema_version"] != HYBRID_SCHEMA
        or header["tensor_name"] != EMBEDDING_TENSOR
        or header["dtype"] != "f32"
        or header["shape"] != [HIDDEN]
        or header["residual_encoding"] != "f32le_row_major"
    ):
        raise ValueError("hybrid input header contract differs")
    expected_case = {
        "kind",
        "case_id",
        "step",
        "context_token_ids",
        "context_token_ids_sha256",
        "context_length",
        "residual_path",
        "residual_sha256",
        "residual_shape",
        "residual_dtype",
    }
    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for line in lines[1:]:
        case = json.loads(line)
        if not isinstance(case, dict) or set(case) != expected_case:
            raise ValueError("hybrid input case fields differ")
        token_ids = case["context_token_ids"]
        key = (case["case_id"], case["step"])
        if (
            case["kind"] != "case"
            or not isinstance(case["case_id"], str)
            or not case["case_id"]
            or not isinstance(case["step"], int)
            or case["step"] < 0
            or not isinstance(token_ids, list)
            or not token_ids
            or len(token_ids) > MAX_CONTEXT
            or any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in token_ids)
            or case["context_length"] != len(token_ids)
            or case["context_token_ids_sha256"] != canonical_token_ids_hash(token_ids)
            or case["residual_shape"] != [len(token_ids), HIDDEN]
            or case["residual_dtype"] != "f32le"
            or key in seen
        ):
            raise ValueError("hybrid input case contract differs")
        seen.add(key)
        cases.append(case)
    if not cases or len(cases) > MAX_CASES:
        raise ValueError("hybrid input case count is outside bounds")
    return header, cases


def package_embedding(package: Path) -> tuple[Path, dict[str, Any], str]:
    manifest_path = package / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    tensors = manifest.get("passthrough_tensors") if isinstance(manifest, dict) else None
    if not isinstance(tensors, list):
        raise ValueError("package manifest has no passthrough tensors")
    matches = [item for item in tensors if isinstance(item, dict) and item.get("name") == EMBEDDING_TENSOR]
    if len(matches) != 1:
        raise ValueError("package manifest does not identify exactly one embedding passthrough")
    entry = matches[0]
    if entry.get("dtype") != "BF16" or entry.get("shape", [None, None])[1:] != [HIDDEN]:
        raise ValueError("package embedding dtype or hidden shape differs")
    vocab = entry["shape"][0]
    if not isinstance(vocab, int) or vocab <= 0:
        raise ValueError("package embedding vocab differs")
    if entry.get("elements") != vocab * HIDDEN:
        raise ValueError("package embedding element count differs")
    payload_name = entry.get("payload_file")
    if not isinstance(payload_name, str):
        raise ValueError("package embedding payload file is missing")
    payload = safe_relative(package, payload_name)
    if not payload.is_file() or payload.stat().st_size != vocab * HIDDEN * 2:
        raise ValueError("package embedding payload size differs")
    payload_sha256 = entry.get("payload_sha256")
    if not isinstance(payload_sha256, str) or len(payload_sha256) != 64:
        raise ValueError("package embedding payload hash is missing")
    return payload, entry, sha256_bytes(manifest_raw)


def bfloat16_rows_as_f32le(raw: bytes) -> bytes:
    if len(raw) % 2:
        raise ValueError("BF16 row bytes are not aligned")
    expanded = bytearray(len(raw) * 2)
    for index, (value,) in enumerate(struct.iter_unpack("<H", raw)):
        struct.pack_into("<I", expanded, index * 4, value << 16)
    return bytes(expanded)


def verify(package: Path, hybrid_input: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing to overwrite receipt: {output}")
    _, cases = read_hybrid_input(hybrid_input)
    payload, entry, manifest_sha256 = package_embedding(package)
    vocab = entry["shape"][0]
    root = hybrid_input.parent
    receipts: list[dict[str, Any]] = []
    with payload.open("rb") as package_rows:
        for case in cases:
            sidecar = safe_relative(root, case["residual_path"])
            raw = sidecar.read_bytes()
            expected_bytes = case["context_length"] * HIDDEN * 4
            if len(raw) != expected_bytes or sha256_bytes(raw) != case["residual_sha256"]:
                raise ValueError(f"hybrid sidecar identity differs: {case['case_id']}:{case['step']}")
            for row_index, token_id in enumerate(case["context_token_ids"]):
                if token_id >= vocab:
                    raise ValueError(f"token id is outside package embedding vocabulary: {token_id}")
                package_rows.seek(token_id * HIDDEN * 2)
                package_raw = package_rows.read(HIDDEN * 2)
                expected_row = bfloat16_rows_as_f32le(package_raw)
                actual_row = raw[row_index * HIDDEN * 4 : (row_index + 1) * HIDDEN * 4]
                if actual_row != expected_row:
                    raise ValueError(
                        f"hybrid input differs from package embedding at {case['case_id']}:{case['step']} token {token_id}"
                    )
            receipts.append(
                {
                    "case_id": case["case_id"],
                    "step": case["step"],
                    "context_length": case["context_length"],
                    "context_token_ids_sha256": case["context_token_ids_sha256"],
                    "residual_sha256": case["residual_sha256"],
                }
            )
    result = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "valid",
        "comparison": "package BF16 rows expanded to f32le are bit-exact with every hybrid residual row",
        "hybrid_input": str(hybrid_input),
        "hybrid_input_sha256": sha256_file(hybrid_input),
        "package_manifest": str(package / "manifest.json"),
        "package_manifest_sha256": manifest_sha256,
        "package_embedding_payload": str(payload),
        "package_embedding_payload_sha256_declared": entry["payload_sha256"],
        "cases": receipts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--hybrid-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.package, args.hybrid_input, args.output)
    except (OSError, TypeError, ValueError, json.JSONDecodeError, struct.error) as error:
        print(f"verify AQ4 layer0 package embedding fixture failed: {error}")
        return 1
    print(json.dumps({"status": result["status"], "cases": len(result["cases"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
