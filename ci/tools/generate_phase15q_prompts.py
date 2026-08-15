#!/usr/bin/env python3
"""Generate the fixed, tuning-independent Phase 15Q token manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


PROMPTS = [
    ("en-short-01", "Hello, explain why the sky appears blue."),
    ("en-short-02", "Write one sentence about open source software."),
    ("en-reason-01", "A train travels 120 km in 90 minutes. Find its average speed."),
    ("en-reason-02", "Compare breadth-first search and depth-first search."),
    ("en-long-01", "Summarize the tradeoffs among latency, throughput, memory use, and accuracy in a local language model inference engine."),
    ("en-long-02", "Describe a careful experiment that distinguishes correlation from causation, including controls and possible confounders."),
    ("ja-short-01", "日本の四季について短く説明してください。"),
    ("ja-short-02", "量子化された言語モデルの利点を一文で説明して。"),
    ("ja-reason-01", "りんごが12個あり、3人に同じ数ずつ配ると一人何個ですか。理由も説明してください。"),
    ("ja-reason-02", "推論速度と数値精度のトレードオフを比較してください。"),
    ("ja-long-01", "家庭用GPUで大規模言語モデルを動かす際に、メモリ容量、帯域、量子化方式をどのように評価すべきか説明してください。"),
    ("ja-long-02", "再現可能な性能測定を行うために固定すべき条件と、測定結果に添えるべき情報を列挙してください。"),
    ("code-rust-01", "Write a Rust function that returns the maximum value in a non-empty slice."),
    ("code-rust-02", "Explain ownership and borrowing in Rust using a small example."),
    ("code-python-01", "Write a Python function that checks whether a string is a palindrome."),
    ("code-python-02", "Find the bug in: for i in range(len(xs)+1): print(xs[i])"),
    ("code-cpp-01", "Implement binary search in C++17 without recursion."),
    ("code-shell-01", "Show a safe shell command to list files larger than one gigabyte."),
    ("math-arith-01", "Compute 17 * 29 and show the intermediate steps."),
    ("math-arith-02", "What is 7/12 + 5/18? Give the reduced fraction."),
    ("math-algebra-01", "Solve 3x + 7 = 34."),
    ("math-algebra-02", "Factor x^2 - 9x + 20."),
    ("math-prob-01", "A fair coin is flipped three times. What is the probability of exactly two heads?"),
    ("math-geometry-01", "Find the area of a circle with radius 5, in terms of pi."),
    ("unicode-01", "Explain these symbols: 🌙🚀✨."),
    ("unicode-02", "「こんにちは」と“hello”を使って短い対話を書いて。"),
    ("facts-01", "Name the largest ocean on Earth and explain the criterion."),
    ("facts-02", "What does CPU stand for?"),
    ("structure-01", "Return a JSON object with keys name, count, and enabled."),
    ("structure-02", "Give three bullet points about deterministic testing."),
    ("instruction-01", "Answer with only the word blue: what color is a clear daytime sky?"),
    ("instruction-02", "Translate 'good morning' into Japanese, then explain the pronunciation."),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    actual = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
    if actual != args.tokenizer_sha256:
        raise SystemExit("tokenizer SHA-256 differs")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    cases = []
    for case_id, text in PROMPTS:
        token_ids = [2, *tokenizer.encode(text, add_special_tokens=False).ids]
        if len(token_ids) < 3 or any(token < 0 or token >= 262_144 for token in token_ids):
            raise SystemExit(f"invalid token sequence: {case_id}")
        cases.append(
            {
                "id": case_id,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "token_ids": token_ids,
                "comparison_positions": [len(token_ids) - 3, len(token_ids) - 2, len(token_ids) - 1],
            }
        )
    document = {
        "schema_version": "phase15q-prompt-manifest-v1",
        "tokenizer_sha256": actual,
        "tuning_set": False,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(f"Phase 15Q prompts: PASS cases={len(cases)} output={args.output}")


if __name__ == "__main__":
    main()
