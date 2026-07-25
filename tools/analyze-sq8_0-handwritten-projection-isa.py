#!/usr/bin/env python3
"""Summarize static gfx1201 ISA/resources for the private SQ8_0 WMMA probe."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


KERNEL = "ullm_sq8_handwritten_gfx1201_m1_wmma_kernel"
RESOURCE = re.compile(
    r"^    \.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|"
    r"sgpr_spill_count|vgpr_count|vgpr_spill_count|wavefront_size):\s+(\d+)$",
    re.MULTILINE,
)
NAME = re.compile(r"^    \.name:\s+(\S+)$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", required=True, type=Path)
    parser.add_argument("--disassembly", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def metadata_block(notes: str) -> str:
    blocks = re.split(r"(?=^  - \.args:)", notes, flags=re.MULTILINE)
    for block in blocks:
        name = NAME.search(block)
        if name and name.group(1) == KERNEL:
            return block
    raise ValueError(f"missing {KERNEL} metadata")


def kernel_disassembly(disassembly: str) -> str:
    marker = f"<{KERNEL}>:"
    start = disassembly.find(marker)
    if start < 0:
        raise ValueError(f"missing {KERNEL} disassembly")
    remaining = disassembly[start + len(marker) :]
    next_symbol = re.search(r"\n[0-9a-f]+ <[^>]+>:\n", remaining)
    return remaining if next_symbol is None else remaining[: next_symbol.start()]


def main() -> int:
    args = parse_args()
    resources = {key: int(value) for key, value in RESOURCE.findall(metadata_block(args.notes.read_text()))}
    instructions = kernel_disassembly(args.disassembly.read_text())
    wmma = len(re.findall(r"\bv_wmma_f32_16x16x16_fp8_fp8\b", instructions))
    barriers = len(re.findall(r"\bs_barrier_(?:signal|wait)\b", instructions))
    result = {
        "schema_version": "ullm.sq8_0.handwritten_projection_isa.v1",
        "target": "gfx1201",
        "kernel": KERNEL,
        "resources": {
            "vgpr_per_thread": resources.get("vgpr_count"),
            "sgpr_per_wave": resources.get("sgpr_count"),
            "static_lds_bytes": resources.get("group_segment_fixed_size"),
            "private_bytes_per_thread": resources.get("private_segment_fixed_size"),
            "vgpr_spills": resources.get("vgpr_spill_count"),
            "sgpr_spills": resources.get("sgpr_spill_count"),
            "wavefront_size": resources.get("wavefront_size"),
        },
        "instructions": {
            "v_wmma_f32_16x16x16_fp8_fp8": wmma,
            "barrier_instructions": barriers,
        },
        "interpretation": (
            "Static compiler metadata only. Runtime active-block occupancy is recorded separately "
            "by hipOccupancyMaxActiveBlocksPerMultiprocessor."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
