#!/usr/bin/env python3
"""Create one deterministic long-context Qwen3.5 quality prompt locally.

The resulting JSON token array is consumed by the native AQ4_0 measurement
binary.  Keeping tokenization outside the GPU window makes the quality input
identical for F32, F16, and FP8 E4M3FN runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


DEFAULT_TOKENIZER = Path(
    "/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B"
)


def canonical_ids_sha256(ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def render_context() -> str:
    facts = [
        "The archive card for north is amber and its index is 17.",
        "The archive card for east is cobalt and its index is 23.",
        "The archive card for south is violet and its index is 37.",
        "The archive card for west is silver and its index is 41.",
        "The safety note says to copy values exactly and never invent a missing card.",
    ]
    paragraphs = []
    for block in range(256):
        paragraphs.append(
            f"Notebook block {block:03d}. " + " ".join(facts) + "\n"
        )
    return "".join(paragraphs)


def render_prompt(tokenizer, target_tokens: int) -> tuple[str, list[int]]:
    messages = [
        {
            "role": "system",
            "content": "Answer the user's final question concisely and preserve stated values exactly.",
        },
        {
            "role": "user",
            "content": (
                render_context()
                + "\nFinal question: What are the color and index of the south archive card? "
                "Reply with exactly one short sentence."
            ),
        },
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = list(tokenizer.encode(rendered, add_special_tokens=False))
    if len(ids) < target_tokens:
        raise RuntimeError(f"long context is too short: got {len(ids)} need {target_tokens}")
    # Keep the final user question *and* the tokenizer's assistant-generation
    # suffix rather than simply truncating the tail.  The leading repeated
    # notebook material supplies the long KV context.
    if len(ids) > target_tokens:
        suffix_count = min(256, target_tokens // 4)
        ids = ids[: target_tokens - suffix_count] + ids[-suffix_count:]
        rendered = tokenizer.decode(ids, skip_special_tokens=False)
    if len(ids) != target_tokens:
        raise RuntimeError(f"token count drifted to {len(ids)} rather than {target_tokens}")
    return rendered, ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--target-tokens", type=int, default=3968)
    args = parser.parse_args()
    if args.target_tokens < 128:
        raise SystemExit("--target-tokens must be at least 128")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=True
    )
    prompt, ids = render_prompt(tokenizer, args.target_tokens)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "quality-input-token-ids.json").write_text(
        json.dumps(ids) + "\n", encoding="utf-8"
    )
    (args.output_dir / "quality-prompt.txt").write_text(prompt, encoding="utf-8")
    metadata = {
        "schema": "ullm.kv_cache_dtype_quality_prompt.v0.1",
        "tokenizer_root": str(args.tokenizer),
        "target_tokens": args.target_tokens,
        "actual_tokens": len(ids),
        "token_ids_sha256_le_u64": canonical_ids_sha256(ids),
        "semantic_question": "south archive card color and index",
        "expected_factual_answer": "violet and 37",
    }
    (args.output_dir / "quality-prompt-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
