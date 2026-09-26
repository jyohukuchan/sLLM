#!/usr/bin/env python3
"""Build Phase86-compatible Tier A prefixes from frozen W8A8 MTP sequences.

This preserves the reviewed rendered prompt IDs from the existing BF16
prefix file and independently verifies every W8A8 sequence against
mtp-bench-v1 before writing a separate fixed-reference campaign input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ValueError("W8A8 prefix validation requires Python tokenizers") from error
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path,
                        default=REPO / "ci/matrix/mtp-bench-v1.json")
    parser.add_argument("--fixture", type=Path,
                        default=REPO / "ci/fixtures/mtp-bench-v1")
    parser.add_argument("--bf16-prefixes", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    matrix = json.loads(args.matrix.read_text())
    baseline = json.loads(args.bf16_prefixes.read_text())
    if matrix.get("schema_version") != "mtp-bench-v1" or baseline.get("schema") != "phase85-a16-mtp-prefix-v1":
        raise ValueError("MTP matrix or BF16 prefix schema differs")
    base_entries = {entry["case_id"]: entry for entry in baseline["entries"]}
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tier_a = [condition for condition in matrix["conditions"] if condition["tier"] == "A"]
    if len(tier_a) != 26 or len(base_entries) != 26:
        raise ValueError("expected exactly 26 Tier A prefix cases")
    entries = []
    for condition in tier_a:
        case_id = condition["cond_id"]
        source = base_entries.get(case_id)
        if source is None:
            raise ValueError(f"BF16 prompt IDs missing for {case_id}")
        prompt_ids = source["prompt_tokens"]
        if len(prompt_ids) != condition["prompt_tokens_rendered"]:
            raise ValueError(f"rendered prompt length differs for {case_id}")
        prompt_path = args.fixture / condition["prompt_file"]
        prompt_bytes = prompt_path.read_bytes()
        if hashlib.sha256(prompt_bytes).hexdigest() != condition["prompt_sha256"]:
            raise ValueError(f"prompt text hash differs for {case_id}")
        prompt = prompt_bytes.decode("utf-8").rstrip("\n")
        rendered = (
            "<|im_start|>user\n" + prompt
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        actual_prompt_ids = tokenizer.encode(rendered, add_special_tokens=False).ids
        if actual_prompt_ids != prompt_ids:
            raise ValueError(f"BF16 prefix prompt token IDs differ for {case_id}")
        reference = condition["reference_sequences"]["w8a8"]
        relative = Path(reference["sequence_file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"W8A8 sequence path escapes fixture: {case_id}")
        sequence_path = args.fixture / relative
        if sha256(sequence_path) != reference["sequence_sha256"]:
            raise ValueError(f"W8A8 sequence file hash differs for {case_id}")
        sequence = json.loads(sequence_path.read_text())
        output_ids = sequence["committed_token_ids"]
        if (sequence.get("cond_id") != case_id or sequence.get("reference") != "w8a8"
                or sequence.get("prompt_sha256") != condition["prompt_sha256"]
                or len(output_ids) != reference["committed_token_count"]
                or any(not isinstance(token, int) or isinstance(token, bool)
                       or token < 0 or token >= 248_320 for token in output_ids)):
            raise ValueError(f"W8A8 sequence identity or tokens differ for {case_id}")
        token_hash = hashlib.sha256(
            ",".join(str(token) for token in output_ids).encode("ascii")
        ).hexdigest()
        if token_hash != reference["committed_token_sha256"]:
            raise ValueError(f"W8A8 sequence token hash differs for {case_id}")
        entries.append({
            "case_id": case_id,
            "seed": source["seed"],
            "prompt_tokens": prompt_ids,
            "output_prefix_tokens": output_ids,
        })
    if {item["case_id"] for item in entries} != set(base_entries):
        raise ValueError("W8A8 and BF16 Tier A case sets differ")
    payload = {
        "schema": "phase85-a16-mtp-prefix-v1",
        "source": str(args.matrix.resolve()),
        "generated_series": "w8a8-frozen-sequence-from-mtp-bench-v1",
        "matrix_sha256": sha256(args.matrix),
        "source_bf16_prefix_sha256": sha256(args.bf16_prefixes),
        "tokenizer_sha256": sha256(args.tokenizer),
        "entries": sorted(entries, key=lambda item: item["case_id"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(entries),
                      "sha256": sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
