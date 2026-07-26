#!/usr/bin/env python3
"""Summarize static gfx1201 ISA/resources for a production AQ4_0 projection kernel."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


RESOURCE = re.compile(
    r"^    \.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|"
    r"sgpr_spill_count|vgpr_count|vgpr_spill_count|wavefront_size):\s+(\d+)$",
    re.MULTILINE,
)
NAME = re.compile(r"^    \.name:\s+(\S+)$", re.MULTILINE)
SYMBOL = re.compile(r"^[0-9a-f]+ <([^>]+)>:$", re.MULTILINE)
INSTRUCTION = re.compile(r"^\s+([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)?)\b")
ADDRESS = re.compile(r"// 0{0,8}([0-9A-Fa-f]+):")
TARGET = re.compile(r"<[^>+]+\+0x([0-9A-Fa-f]+)>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--notes", required=True, type=Path)
    parser.add_argument("--disassembly", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def metadata_block(notes: str, kernel: str) -> str:
    blocks = re.split(r"(?=^  - \.args:)", notes, flags=re.MULTILINE)
    for block in blocks:
        name = NAME.search(block)
        if name and name.group(1) == kernel:
            return block
    raise ValueError(f"missing kernel metadata: {kernel}")


def kernel_body(disassembly: str, kernel: str) -> tuple[int, str]:
    match = re.search(rf"^([0-9a-f]+) <{re.escape(kernel)}>:$", disassembly, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing kernel disassembly: {kernel}")
    start = match.end()
    remaining = disassembly[start:]
    next_symbol = SYMBOL.search(remaining)
    return int(match.group(1), 16), remaining if next_symbol is None else remaining[: next_symbol.start()]


def instruction_rows(body: str) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    for line in body.splitlines():
        mnemonic = INSTRUCTION.match(line)
        address = ADDRESS.search(line)
        if mnemonic is not None and address is not None:
            rows.append((int(address.group(1), 16), mnemonic.group(1), line))
    return rows


def memory_width(mnemonic: str) -> int:
    if "dwordx4" in mnemonic or "b128" in mnemonic:
        return 16
    if "dwordx3" in mnemonic or "b96" in mnemonic:
        return 12
    if "dwordx2" in mnemonic or "b64" in mnemonic:
        return 8
    if "dword" in mnemonic or "b32" in mnemonic:
        return 4
    if "ushort" in mnemonic or "short" in mnemonic or "b16" in mnemonic:
        return 2
    if "ubyte" in mnemonic or "byte" in mnemonic or "b8" in mnemonic:
        return 1
    return 0


def instruction_summary(rows: list[tuple[int, str, str]]) -> dict[str, object]:
    mnemonics = Counter(mnemonic for _, mnemonic, _ in rows)
    loads = {
        mnemonic: count
        for mnemonic, count in sorted(mnemonics.items())
        if mnemonic.startswith(("global_load", "flat_load", "buffer_load"))
    }
    stores = {
        mnemonic: count
        for mnemonic, count in sorted(mnemonics.items())
        if mnemonic.startswith(("global_store", "flat_store", "buffer_store"))
    }
    return {
        "total": len(rows),
        "valu": sum(count for mnemonic, count in mnemonics.items() if mnemonic.startswith("v_")),
        "salu": sum(count for mnemonic, count in mnemonics.items() if mnemonic.startswith("s_")),
        "vector_memory": sum(
            count
            for mnemonic, count in mnemonics.items()
            if mnemonic.startswith(("global_", "flat_", "buffer_"))
        ),
        "static_load_width_bytes": sum(count * memory_width(mnemonic) for mnemonic, count in loads.items()),
        "static_store_width_bytes": sum(count * memory_width(mnemonic) for mnemonic, count in stores.items()),
        "loads": loads,
        "stores": stores,
        "reduction_operations": {
            mnemonic: count
            for mnemonic, count in sorted(mnemonics.items())
            if mnemonic.startswith(("ds_", "s_barrier", "v_readlane", "v_writelane", "v_permlane", "v_perm"))
            or mnemonic in {"v_add_f32", "v_add_f32_e32", "v_add_f32_e64"}
        },
        "arithmetic_operations": {
            mnemonic: count
            for mnemonic, count in sorted(mnemonics.items())
            if mnemonic.startswith(("v_fma", "v_fmac", "v_mul_f32", "v_add_f32", "v_cndmask", "v_bfe", "v_and_b32", "v_lsh", "v_lshr"))
        },
    }


def backedges(rows: list[tuple[int, str, str]], function_start: int) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for address, mnemonic, line in rows:
        if not mnemonic.startswith(("s_branch", "s_cbranch")):
            continue
        target = TARGET.search(line)
        if target is None:
            continue
        target_address = function_start + int(target.group(1), 16)
        if target_address >= address:
            continue
        loop_rows = [row for row in rows if target_address <= row[0] <= address]
        result.append(
            {
                "from": f"0x{address:x}",
                "to": f"0x{target_address:x}",
                "branch": mnemonic,
                "inclusive_static_body": instruction_summary(loop_rows),
            }
        )
    return result


def main() -> int:
    args = parse_args()
    notes = args.notes.read_text(encoding="utf-8")
    disassembly = args.disassembly.read_text(encoding="utf-8")
    resources = {
        key: int(value)
        for key, value in RESOURCE.findall(metadata_block(notes, args.kernel))
    }
    start, body = kernel_body(disassembly, args.kernel)
    rows = instruction_rows(body)
    result = {
        "schema_version": "ullm.aq4_projection_isa.v1",
        "kernel": args.kernel,
        "resources": {
            "vgpr_per_thread": resources.get("vgpr_count"),
            "sgpr_per_wave": resources.get("sgpr_count"),
            "static_lds_bytes": resources.get("group_segment_fixed_size"),
            "private_bytes_per_thread": resources.get("private_segment_fixed_size"),
            "vgpr_spills": resources.get("vgpr_spill_count"),
            "sgpr_spills": resources.get("sgpr_spill_count"),
            "wavefront_size": resources.get("wavefront_size"),
        },
        "static_instruction_totals": instruction_summary(rows),
        "backedges": backedges(rows, start),
        "method_note": (
            "These are static ISA counts. Runtime per-element instruction counts depend on the "
            "g8/g16 branch and loop trip counts; those conditions must be derived separately "
            "from the audited source and deployed shapes."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
