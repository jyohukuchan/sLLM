#!/usr/bin/env python3
"""Make a verified prefix view of an existing Qwen3.8 raw-logit capture.

This copies existing rows; it does not run an inference engine.  The output
manifest records the source capture and file hashes so a prefix control can
be compared with a shorter teacher-forced run by qwen38_kld_compare.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-case", required=True)
    parser.add_argument("--source-inputs", type=Path, required=True)
    parser.add_argument("--prefix-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_manifest_path = args.source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_inputs = json.loads(args.source_inputs.read_text())["cases"]
    prefix_inputs = json.loads(args.prefix_inputs.read_text())["cases"]
    if len(prefix_inputs) != 1:
        raise ValueError("prefix manifest must contain exactly one case")
    full = next(case for case in source_inputs if case["id"] == args.source_case)
    short = prefix_inputs[0]
    count = len(short["token_ids"])
    if not 0 < count <= len(full["token_ids"]):
        raise ValueError("prefix length is invalid")
    if full["token_ids"][:count] != short["token_ids"]:
        raise ValueError("prefix tokens differ from source tokens")
    item = next(case for case in source_manifest["cases"] if case["id"] == args.source_case)
    positions = item["positions"]
    if positions[:count] != list(range(count)):
        raise ValueError("source capture does not contain the contiguous prefix")
    expected_hashes = {
        hashlib.sha256(json.dumps(full["token_ids"], separators=(",", ":")).encode()).hexdigest(),
        hashlib.sha256(struct.pack("<" + "i" * len(full["token_ids"]), *full["token_ids"])).hexdigest(),
    }
    source_hash = item.get("input_token_ids_sha256", item.get("token_ids_sha256", ""))
    if source_hash.removeprefix("sha256:") not in expected_hashes:
        raise ValueError("source capture token hash differs from source inputs")
    shape = item.get("shape", [item.get("rows"), item.get("vocab_size")])
    if len(shape) != 2 or shape[0] < count or shape[1] != 248320:
        raise ValueError("source logits shape is invalid")
    source_file = args.source / item["logits_file"]
    if source_file.stat().st_size != shape[0] * shape[1] * 4:
        raise ValueError("source logits size differs from manifest")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    destination = args.output / f"{short['id']}.f32"
    remaining = count * shape[1] * 4
    with source_file.open("rb") as reader, destination.open("xb") as writer:
        while remaining:
            block = reader.read(min(remaining, 1024 * 1024))
            if not block:
                raise ValueError("source logits ended inside the requested prefix")
            writer.write(block)
            remaining -= len(block)
    prefix_hash = hashlib.sha256(
        json.dumps(short["token_ids"], separators=(",", ":")).encode()
    ).hexdigest()
    projected_case = {
        "id": short["id"],
        "logits_file": destination.name,
        "logits_file_sha256": sha256_file(destination),
        "shape": [count, shape[1]],
        "positions": list(range(count)),
        "input_token_ids_sha256": "sha256:" + prefix_hash,
        "nonfinite_count": item.get("nonfinite_count", 0),
    }
    result = {
        "schema_version": "qwen38-kld-derived-prefix-view-v1",
        "engine": source_manifest.get("engine"),
        "model": source_manifest.get("model"),
        "derived_from": {
            "source_manifest": str(source_manifest_path.resolve()),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "source_case": args.source_case,
            "source_logits_sha256": sha256_file(source_file),
            "source_inputs": str(args.source_inputs.resolve()),
            "source_inputs_sha256": sha256_file(args.source_inputs),
            "prefix_inputs": str(args.prefix_inputs.resolve()),
            "prefix_inputs_sha256": sha256_file(args.prefix_inputs),
        },
        "cases": [projected_case],
    }
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "case": short["id"], "rows": count}))


if __name__ == "__main__":
    main()
