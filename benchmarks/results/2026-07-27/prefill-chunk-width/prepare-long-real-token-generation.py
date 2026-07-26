#!/usr/bin/env python3
"""Create a real-token N=4000 chat input for wide-M generation checks.

The prefix is assembled from complete, tokenizer-rendered chat records.  It
does not use padding, a masked row, or an invented token ID.  A final real
chat prompt retains its generation header after the long prefix, so the
resulting completion is readable rather than a continuation of a truncated
template.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=4000)
    return parser.parse_args()


def normalize(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("chat template produced multiple sequences")
        value = value[0]
    if not isinstance(value, list) or not value:
        raise ValueError("chat template produced no token IDs")
    if not all(isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in value):
        raise ValueError("chat template produced invalid token IDs")
    return list(value)


def render(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
    value = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    return normalize(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    args = parse_args()
    if args.target_tokens < 128:
        raise ValueError("--target-tokens must be at least 128")
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite existing output: {args.output_dir}")
    from transformers import AutoTokenizer, __version__ as transformers_version

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    repeated_record = [
        {"role": "system", "content": "Keep the following engineering record as factual context."},
        {
            "role": "user",
            "content": "A synchronized prefill step committed only real token rows and retained a rollback record.",
        },
        {
            "role": "assistant",
            "content": "Recorded: use the same evidence and preserve the service recovery path.",
        },
    ]
    final_messages = [
        {"role": "system", "content": "Answer in concise English."},
        {
            "role": "user",
            "content": "Based on the context, explain why processing a long prefill in fewer real-token chunks can reduce repeated work. Mention one verification step.",
        },
    ]
    record_ids = render(tokenizer, repeated_record, add_generation_prompt=False)
    final_ids = render(tokenizer, final_messages, add_generation_prompt=True)
    if len(final_ids) >= args.target_tokens:
        raise ValueError("final prompt unexpectedly exhausts the target token budget")
    prefix_tokens = args.target_tokens - len(final_ids)
    repeats, tail = divmod(prefix_tokens, len(record_ids))
    token_ids = record_ids * repeats + record_ids[:tail] + final_ids
    if len(token_ids) != args.target_tokens:
        raise AssertionError("long prompt construction length mismatch")
    raw = struct.pack(f"<{len(token_ids)}I", *token_ids)
    args.output_dir.mkdir(parents=True)
    token_path = args.output_dir / "long-prefill-p4000.u32le"
    token_path.write_bytes(raw)
    manifest = {
        "schema_version": "ullm.sq8.prefill_chunk_width.long_real_token_input.v1",
        "scope": "real tokenizer-rendered chat records; no padding, mask, or fabricated token rows",
        "model_dir": str(args.model_dir),
        "transformers_version": transformers_version,
        "target_tokens": args.target_tokens,
        "token_file": token_path.name,
        "token_file_sha256": sha256_bytes(raw),
        "record_token_count": len(record_ids),
        "complete_record_repetitions": repeats,
        "real_token_prefix_from_next_record": tail,
        "final_prompt_token_count": len(final_ids),
        "repeated_record_messages": repeated_record,
        "final_messages": final_messages,
        "chat_template_arguments": {"add_generation_prompt": True, "enable_thinking": False},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
