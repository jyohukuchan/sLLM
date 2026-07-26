#!/usr/bin/env python3
"""Decode the three native AQ4_0 KV-cache generation outputs side by side."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


DEFAULT_TOKENIZER = Path(
    "/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B"
)


def load_generation(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    generation = value.get("generation")
    if not isinstance(generation, dict):
        raise RuntimeError(f"{path}: no generation object")
    ids = generation.get("generated_token_ids")
    if not isinstance(ids, list) or not all(isinstance(item, int) for item in ids):
        raise RuntimeError(f"{path}: invalid generated token ids")
    dtype = value.get("kv_cache_dtype")
    if not isinstance(dtype, dict):
        raise RuntimeError(f"{path}: no KV dtype provenance")
    return {"path": str(path), "dtype": dtype, "ids": ids}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f32", type=Path, required=True)
    parser.add_argument("--f16", type=Path, required=True)
    parser.add_argument("--fp8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=True
    )
    rows = []
    for label, path in (("F32", args.f32), ("F16", args.f16), ("FP8_E4M3FN", args.fp8)):
        row = load_generation(path)
        row["label"] = label
        row["decoded_text"] = tokenizer.decode(row["ids"], skip_special_tokens=True)
        rows.append(row)
    output = {
        "schema": "ullm.kv_cache_dtype_quality_output.v0.1",
        "tokenizer_root": str(args.tokenizer),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = [
        "# Long-context KV-cache dtype generation",
        "",
        "The same 3,968-token input was used for all rows. FP8 is OCP E4M3FN/S1E4M3.",
        "",
    ]
    for row in rows:
        markdown.extend(
            [
                f"## {row['label']}",
                "",
                str(row["decoded_text"]),
                "",
                "Token IDs: `" + ",".join(map(str, row["ids"])) + "`",
                "",
            ]
        )
    args.output.with_suffix(".md").write_text("\n".join(markdown), encoding="utf-8")


if __name__ == "__main__":
    main()
