#!/usr/bin/env python3
"""Render the fixed lightweight prompt suite into Qwen token-ID files.

This is a read-only tokenizer preparation step for an isolated local runtime.
It neither contacts a served model nor changes any promotion state.  The
result keeps both the rendered chat prompt and little-endian u32 token IDs so
the exact inputs to a non-HTTP runtime can be audited and replayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

import lightweight_promotion as PROMOTION


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "docs" / "plans" / "lightweight-promotion-prompt-suite-v0.1.json"
SCHEMA = "ullm.qwen_lightweight_token_suite.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_new(path: Path, value: bytes) -> None:
    with path.open("xb") as target:
        target.write(value)


def normalize_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError("chat template did not return a token-ID list")
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("chat template returned multiple token sequences")
        value = value[0]
    token_ids: list[int] = []
    for index, token_id in enumerate(value):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(f"token {index} is not an integer")
        if token_id < 0 or token_id > 0xFFFFFFFF:
            raise ValueError(f"token {index} is outside u32")
        token_ids.append(token_id)
    if not token_ids:
        raise ValueError("chat template returned no tokens")
    return token_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-suite", type=Path, default=DEFAULT_SUITE)
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    prompt_suite = args.prompt_suite.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output_dir}")
    if not model_dir.is_dir():
        raise SystemExit(f"model directory does not exist: {model_dir}")
    if not prompt_suite.is_file():
        raise SystemExit(f"prompt suite does not exist: {prompt_suite}")

    suite = PROMOTION.load_suite(prompt_suite)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    output_dir.mkdir(parents=True)
    token_dir = output_dir / "tokens"
    rendered_dir = output_dir / "rendered-prompts"
    token_dir.mkdir()
    rendered_dir.mkdir()

    cases: list[dict[str, Any]] = []
    for case in suite:
        messages = [dict(message) for message in case.messages]
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            token_value = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception as error:  # pragma: no cover - tokenizer implementation detail
            raise SystemExit(f"{case.case_id}: failed to apply the Qwen chat template: {error}") from error
        if not isinstance(rendered, str) or not rendered:
            raise SystemExit(f"{case.case_id}: chat template returned an empty rendered prompt")
        try:
            token_ids = normalize_token_ids(token_value)
        except ValueError as error:
            raise SystemExit(f"{case.case_id}: {error}") from error
        raw = struct.pack(f"<{len(token_ids)}I", *token_ids)
        token_relative = Path("tokens") / f"{case.case_id}.u32le"
        rendered_relative = Path("rendered-prompts") / f"{case.case_id}.txt"
        write_new(output_dir / token_relative, raw)
        write_new(output_dir / rendered_relative, rendered.encode("utf-8"))
        cases.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "messages": messages,
                "max_completion_tokens": case.max_completion_tokens,
                "expected_language": case.expected_language,
                "expected_kind": case.expected_kind,
                "prompt_token_count": len(token_ids),
                "prompt_u32le_file": str(token_relative),
                "prompt_u32le_sha256": sha256_bytes(raw),
                "rendered_prompt_file": str(rendered_relative),
                "rendered_prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
            }
        )

    tokenizer_files = {
        name: sha256_file(model_dir / name)
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        if (model_dir / name).is_file()
    }
    manifest = {
        "schema_version": SCHEMA,
        "scope": "offline Qwen tokenizer rendering for isolated local runtime input",
        "model_dir": str(model_dir),
        "prompt_suite": str(prompt_suite),
        "prompt_suite_sha256": sha256_file(prompt_suite),
        "chat_template_arguments": {"add_generation_prompt": True, "enable_thinking": False},
        "tokenizer_files_sha256": tokenizer_files,
        "cases": cases,
    }
    write_new(
        output_dir / "suite.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
