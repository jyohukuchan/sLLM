#!/usr/bin/env python3
"""Build bounded CPU-only raw-residual fixtures for the layer-0 hybrid probe.

Only the embedding rows named by an explicit context manifest are read from
the BF16 source safetensors file.  This avoids loading the full 2 GB embedding
matrix and preserves the complete prefix needed to replay layer-0 Conv1d and
recurrent state from zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


SCHEMA = "ullm.aq4_layer0_hybrid_input_jsonl.v1"
CONTEXT_SCHEMA = "ullm.aq4_layer0_hybrid_contexts.v1"
EMBEDDING_TENSOR = "model.language_model.embed_tokens.weight"
HIDDEN = 4096
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


def load_contexts(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"schema_version", "cases"}:
        raise ValueError("contexts must contain only schema_version and cases")
    if value["schema_version"] != CONTEXT_SCHEMA or not isinstance(value["cases"], list):
        raise ValueError("contexts schema differs")
    if not value["cases"] or len(value["cases"]) > MAX_CASES:
        raise ValueError("context count is outside bounds")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for item in value["cases"]:
        if not isinstance(item, dict) or set(item) != {"case_id", "step", "context_token_ids"}:
            raise ValueError("context case fields differ")
        case_id = item["case_id"]
        step = item["step"]
        token_ids = item["context_token_ids"]
        if not isinstance(case_id, str) or not case_id or not isinstance(step, int) or step < 0:
            raise ValueError("context case identity differs")
        if not isinstance(token_ids, list) or not token_ids or len(token_ids) > MAX_CONTEXT:
            raise ValueError("context token count is outside bounds")
        if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in token_ids):
            raise ValueError("context token ids must be non-negative integers")
        key = (case_id, step)
        if key in seen:
            raise ValueError(f"duplicate context case/step: {case_id}:{step}")
        seen.add(key)
        result.append({"case_id": case_id, "step": step, "context_token_ids": token_ids})
    return result


def source_embedding_rows(source_model: Path, token_ids: list[int]) -> tuple[torch.Tensor, dict[str, str]]:
    index_path = source_model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or EMBEDDING_TENSOR not in weight_map:
        raise ValueError("source model index has no embedding tensor")
    shard = source_model / weight_map[EMBEDDING_TENSOR]
    if not shard.is_file():
        raise ValueError(f"source embedding shard is missing: {shard}")
    rows: list[torch.Tensor] = []
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        metadata = handle.get_slice(EMBEDDING_TENSOR)
        for token_id in token_ids:
            row = metadata[token_id : token_id + 1]
            if list(row.shape) != [1, HIDDEN] or row.dtype != torch.bfloat16:
                raise ValueError("source embedding row geometry/dtype differs")
            rows.append(row.to(dtype=torch.float32).contiguous())
    values = torch.cat(rows, dim=0).contiguous()
    return values, {
        "source_model_index_sha256": sha256_file(index_path),
        "source_embedding_shard": str(shard),
        "source_embedding_shard_sha256": sha256_file(shard),
    }


def safe_slug(case_id: str, step: int) -> str:
    normalized = "".join(character if character.isalnum() or character in "-_" else "_" for character in case_id)
    return f"{normalized}-step-{step}"


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contexts = load_contexts(args.contexts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sidecar_dir = args.output.parent / "input"
    if sidecar_dir.exists():
        raise ValueError(f"refusing to reuse hybrid sidecar directory: {sidecar_dir}")
    sidecar_dir.mkdir()

    records: list[dict[str, Any]] = []
    index_sha: str | None = None
    for context in contexts:
        token_ids = context["context_token_ids"]
        values, source = source_embedding_rows(args.source_model, token_ids)
        if index_sha is None:
            index_sha = source["source_model_index_sha256"]
        elif index_sha != source["source_model_index_sha256"]:
            raise ValueError("source model index changed while building hybrid input")
        payload = values.numpy().tobytes()
        slug = safe_slug(context["case_id"], context["step"])
        sidecar = sidecar_dir / f"{slug}-residual.f32le"
        if sidecar.exists():
            raise ValueError(f"duplicate hybrid sidecar path: {sidecar}")
        sidecar.write_bytes(payload)
        records.append(
            {
                "kind": "case",
                "case_id": context["case_id"],
                "step": context["step"],
                "context_token_ids": token_ids,
                "context_token_ids_sha256": canonical_token_ids_hash(token_ids),
                "context_length": len(token_ids),
                "residual_path": sidecar.relative_to(args.output.parent).as_posix(),
                "residual_sha256": sha256_bytes(payload),
                "residual_shape": [len(token_ids), HIDDEN],
                "residual_dtype": "f32le",
            }
        )
    if index_sha is None:
        raise ValueError("no hybrid input records were produced")
    header = {
        "kind": "header",
        "schema_version": SCHEMA,
        "tensor_name": EMBEDDING_TENSOR,
        "dtype": "f32",
        "shape": [HIDDEN],
        "residual_encoding": "f32le_row_major",
        "source_model_index_sha256": index_sha,
    }
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(header, ensure_ascii=True, sort_keys=True) + "\n")
        for record in records:
            output.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    return {
        "schema_version": SCHEMA,
        "status": "valid",
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "rows": len(records),
        "source_model_index_sha256": index_sha,
        "device": "cpu-only-safe_open",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare(args)
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"prepare layer0 hybrid input failed: {error}")
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
