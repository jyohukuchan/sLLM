#!/usr/bin/env python3
"""Record actual N=4000 wide-M completions without a numerical gate."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


WIDTHS = (128, 256, 512, 1024, 2048)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def load_result(path: Path, tokenizer: Any) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: result is not an object")
    requests = value.get("requests")
    if not isinstance(requests, list) or len(requests) != 1 or not isinstance(requests[0], dict):
        raise ValueError(f"{path}: expected exactly one request")
    request = requests[0]
    token_ids = request.get("generated_token_ids")
    if not isinstance(token_ids, list) or not all(
        isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in token_ids
    ):
        raise ValueError(f"{path}: generated token IDs are invalid")
    text = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if not isinstance(text, str):
        raise ValueError(f"{path}: decoder returned non-text")
    text.encode("utf-8", errors="strict")
    return {
        "source_result": str(path),
        "prefill_mode": value.get("prefill_mode"),
        "prefill_chunk_tokens": value.get("prefill_chunk_tokens"),
        "runner_binary_sha256": value.get("runner_binary_sha256"),
        "generated_token_ids": token_ids,
        "generated_token_count": len(token_ids),
        "content": text,
        "character_count": len(text),
        "terminal_reason": request.get("terminal_reason"),
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Long real-token wide-M generation evidence",
        "",
        "The N=4000 prefix is composed solely of tokenizer-rendered real chat tokens and retains a real final generation header. It is not padding or a masked/fabricated row. Text is retained for qualitative inspection; no numerical threshold decides acceptance.",
        "",
    ]
    baseline = summary["runs"]["128"]
    for width in WIDTHS:
        record = summary["runs"][str(width)]
        text_exact = record["content"] == baseline["content"]
        ids_exact = record["generated_token_ids"] == baseline["generated_token_ids"]
        lines.extend(
            [
                f"## M={width}",
                "",
                f"- Generated IDs exact versus M=128: `{ids_exact}`",
                f"- Decoded text exact versus M=128: `{text_exact}`",
                f"- Generated token count: {record['generated_token_count']}",
                "",
                "<pre>",
                html.escape(str(record["content"])),
                "</pre>",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    from transformers import AutoTokenizer, __version__ as transformers_version

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    runs = {
        str(width): load_result(
            args.run_root / "generation-long" / f"m{width}" / "result.json", tokenizer
        )
        for width in WIDTHS
    }
    summary = {
        "schema_version": "ullm.sq8.prefill_chunk_width.long_real_token_generation.v1",
        "scope": "direct isolated SQ8_0 serving, not gateway activation or promotion",
        "transformers_version": transformers_version,
        "not_a_numerical_gate": True,
        "runs": runs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as destination:
        json.dump(summary, destination, ensure_ascii=False, indent=2, sort_keys=True)
        destination.write("\n")
    with args.output_markdown.open("x", encoding="utf-8") as destination:
        destination.write(markdown(summary))
        destination.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
