#!/usr/bin/env python3
"""Attach the frozen reference sequences to the mtp-bench-v1 manifest and freeze it.

Run after ci/tools/freeze_mtp_bench_sequences.py. The manifest only moves to
state "frozen" when every condition carries both reference sequences and every
recorded prompt hash still matches the file on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
FIXTURE = REPO / "ci" / "fixtures" / "mtp-bench-v1"


def sha256_text(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--provenance", required=True)
    args = parser.parse_args()

    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    provenance = json.loads(pathlib.Path(args.provenance).read_text(encoding="utf-8"))

    by_condition: dict[str, dict[str, dict]] = {}
    for record in provenance["records"]:
        by_condition.setdefault(record["cond_id"], {})[record["reference"]] = record

    problems: list[str] = []
    references = sorted(provenance["references"])

    for condition in manifest["conditions"]:
        cond_id = condition["cond_id"]

        prompt_path = FIXTURE / condition["prompt_file"]
        if not prompt_path.exists():
            problems.append(f"{cond_id}: prompt file missing")
            continue
        if sha256_text(prompt_path) != condition["prompt_sha256"]:
            problems.append(f"{cond_id}: prompt_sha256 mismatch on disk")
            continue

        found = by_condition.get(cond_id, {})
        missing = [name for name in references if name not in found]
        if missing:
            problems.append(f"{cond_id}: missing reference sequence(s) {missing}")
            continue

        sequences = {}
        rendered = set()
        for name in references:
            record = found[name]
            sequence_path = FIXTURE / record["sequence_file"]
            if not sequence_path.exists():
                problems.append(f"{cond_id}/{name}: sequence file missing")
                continue
            if sha256_text(sequence_path) != record["sequence_sha256"]:
                problems.append(f"{cond_id}/{name}: sequence_sha256 mismatch on disk")
                continue
            if not record.get("tokenizer_round_trip_exact", False):
                problems.append(f"{cond_id}/{name}: tokenizer round trip not exact")
            # The committed count may differ from the requested output budget:
            # a condition can stop naturally, and tokenize(output_text) can
            # re-segment by a token. Both are fine for a forced sequence; only a
            # degenerate sequence is a problem.
            if record["committed_token_count"] < 32:
                problems.append(
                    f"{cond_id}/{name}: degenerate sequence of "
                    f"{record['committed_token_count']} tokens")
            sequences[name] = {
                "sequence_file": record["sequence_file"],
                "sequence_sha256": record["sequence_sha256"],
                "committed_token_sha256": record["committed_token_sha256"],
                "committed_token_count": record["committed_token_count"],
                "reported_completion_tokens": record.get("reported_completion_tokens"),
                "finish_reason": record.get("finish_reason"),
                "reference_mtp": {
                    "proposed": record.get("mtp_proposed_draft_tokens", 0),
                    "accepted": record.get("mtp_accepted_draft_tokens", 0),
                    "blocks": record.get("mtp_draft_proposal_blocks", 0),
                },
            }
            rendered.add(int(record["prompt_tokens_rendered"]))

        if len(rendered) != 1:
            problems.append(f"{cond_id}: rendered prompt token counts disagree {sorted(rendered)}")
        else:
            condition["prompt_tokens_rendered"] = rendered.pop()
        condition["prompt_tokens_raw"] = condition.pop("prompt_tokens")
        condition["reference_sequences"] = sequences

    manifest["reference_sequence_provenance"] = {
        "generated_on": provenance["generated_on"],
        "binary_sha256": provenance["binary_sha256"],
        "sampling": provenance["sampling"],
        "sequence_count": provenance["sequence_count"],
        "derivation": "tokenize(output_text); the sequence is a forced input, not a reproduction target",
        "input_mode": "chat template single user turn (--message user:...)",
    }
    manifest["freeze_problems"] = problems
    manifest["state"] = "frozen" if not problems else "draft-pending-freeze"
    if not problems:
        manifest.pop("freeze_requirements", None)

    pathlib.Path(args.manifest).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if problems:
        print("NOT FROZEN — %d problem(s):" % len(problems))
        for problem in problems[:20]:
            print("  " + problem)
        return 1
    print("frozen: %d conditions, %d reference sequences (%s)"
          % (len(manifest["conditions"]), provenance["sequence_count"], ", ".join(references)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
