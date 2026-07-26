#!/usr/bin/env python3
"""Summarize exact SQ8_0 HIPRTC matvec code-object resources and ISA."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


KERNELS = (
    "ullm_sq_fp8_matvec_f32_kernel",
    "ullm_sq_fp8_matvec_batch_f32_kernel",
    "ullm_sq_fp8_matvec_pair_f32_kernel",
    "ullm_sq_fp8_matvec_triple_f32_kernel",
)

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
    parser.add_argument("--arch", required=True)
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
    raise ValueError(f"missing {kernel} metadata")


def kernel_disassembly(disassembly: str, kernel: str) -> tuple[int, str]:
    match = re.search(rf"^([0-9a-f]+) <{re.escape(kernel)}>:$", disassembly, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing {kernel} disassembly")
    start = match.end()
    remaining = disassembly[start:]
    next_symbol = SYMBOL.search(remaining)
    body = remaining if next_symbol is None else remaining[: next_symbol.start()]
    return int(match.group(1), 16), body


def instruction_rows(body: str) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    for line in body.splitlines():
        mnemonic = INSTRUCTION.match(line)
        address = ADDRESS.search(line)
        if mnemonic is None or address is None:
            continue
        rows.append((int(address.group(1), 16), mnemonic.group(1), line))
    return rows


def instruction_summary(rows: list[tuple[int, str, str]]) -> dict[str, int]:
    mnemonics = Counter(mnemonic for _, mnemonic, _ in rows)
    return {
        "total": len(rows),
        "valu": sum(count for mnemonic, count in mnemonics.items() if mnemonic.startswith("v_")),
        "salu": sum(count for mnemonic, count in mnemonics.items() if mnemonic.startswith("s_")),
        "fp8_to_f32": mnemonics["v_cvt_f32_fp8_e32"] + mnemonics["v_cvt_f32_fp8"],
        "global_load_u8": mnemonics["global_load_u8"],
        "global_load_sbyte": mnemonics["global_load_sbyte"],
        "global_load_b32": mnemonics["global_load_b32"],
        "global_load_b128": mnemonics["global_load_b128"],
        "global_load_dwordx4": mnemonics["global_load_dwordx4"],
        "buffer_load_b128": mnemonics["buffer_load_b128"],
        "v_cvt_f32_ubyte0": mnemonics["v_cvt_f32_ubyte0_e32"]
        + mnemonics["v_cvt_f32_ubyte0"],
        "v_bfe_u32": mnemonics["v_bfe_u32"],
        "v_and_b32": sum(
            count for mnemonic, count in mnemonics.items() if mnemonic.startswith("v_and_b32")
        ),
        "lds_store_b32": mnemonics["ds_store_b32"],
        "lds_load_b32": mnemonics["ds_load_b32"],
        "barrier_signal": mnemonics["s_barrier_signal"],
        "barrier_wait": mnemonics["s_barrier_wait"],
        "v_rcp_iflag_f32": mnemonics["v_rcp_iflag_f32_e32"] + mnemonics["v_rcp_iflag_f32"],
        "v_mul_hi_u32": mnemonics["v_mul_hi_u32"],
        "v_mul_lo_u32": mnemonics["v_mul_lo_u32"],
        "v_mad_co_u64_u32": mnemonics["v_mad_co_u64_u32"],
        "s_mul_hi_u32": mnemonics["s_mul_hi_u32"],
        "s_mul_u64": mnemonics["s_mul_u64"],
        "v_fmac_f32": mnemonics["v_fmac_f32_e32"] + mnemonics["v_fmac_f32"],
        "v_fma_f32": mnemonics["v_fma_f32_e32"] + mnemonics["v_fma_f32"],
        "v_mul_f32": mnemonics["v_mul_f32_e32"] + mnemonics["v_mul_f32"],
        "v_lshlrev_b32": mnemonics["v_lshlrev_b32_e32"] + mnemonics["v_lshlrev_b32"],
        "v_lshrrev_b32": mnemonics["v_lshrrev_b32_e32"] + mnemonics["v_lshrrev_b32"],
    }


def backedges(rows: list[tuple[int, str, str]], function_start: int) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, (address, mnemonic, line) in enumerate(rows):
        if not (mnemonic.startswith("s_branch") or mnemonic.startswith("s_cbranch")):
            continue
        target = TARGET.search(line)
        if target is None:
            continue
        target_address = function_start + int(target.group(1), 16)
        if target_address >= address:
            continue
        body = [row for row in rows if target_address <= row[0] <= address]
        result.append(
            {
                "from": f"0x{address:x}",
                "to": f"0x{target_address:x}",
                "branch": mnemonic,
                "inclusive_static_body": instruction_summary(body),
            }
        )
    return result


def main() -> int:
    args = parse_args()
    notes = args.notes.read_text(encoding="utf-8")
    disassembly = args.disassembly.read_text(encoding="utf-8")
    result: dict[str, object] = {
        "schema_version": "ullm.sq8_0.decode_matvec_isa.v1",
        "arch": args.arch,
        "kernels": {},
        "method_note": (
            "Instruction totals and loop bodies are static code-object counts. Dynamic per-element "
            "counts require selecting a runtime branch and loop backedge; this report preserves all "
            "identified backward branch bodies for that audit."
        ),
    }
    kernels: dict[str, object] = {}
    for kernel in KERNELS:
        resources = {
            key: int(value)
            for key, value in RESOURCE.findall(metadata_block(notes, kernel))
        }
        function_start, body = kernel_disassembly(disassembly, kernel)
        rows = instruction_rows(body)
        kernels[kernel] = {
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
            "backedges": backedges(rows, function_start),
        }
    result["kernels"] = kernels
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
