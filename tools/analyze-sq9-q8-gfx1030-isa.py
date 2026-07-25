#!/usr/bin/env python3
"""Count instructions and compiler resources in SQ9_0/Q8_0 gfx1030 ISA evidence."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


KERNELS = (
    "ullm_q8_0_w8a8_g32_gemv_isa",
    "ullm_sq9_0_w8a8_g32_gemv_isa",
    "ullm_q8_0_w8a16_g32_gemv_isa",
    "ullm_sq9_0_w8a16_gemv_isa",
    "ullm_q8_0_w8a16_pk_f16_g32_gemv_isa",
    "ullm_sq9_0_w8a16_pk_f16_gemv_isa",
)
PROBE_ELEMENTS = 128
LABEL = re.compile(r"^[0-9a-fA-F]+ <([^>]+)>:$")
OPCODE = re.compile(r"^\s+([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)?)\b")
NOTE_NAME = re.compile(r"^    \.name:\s+(\S+)$", re.MULTILINE)
NOTE_VALUE = re.compile(
    r"^    \.(group_segment_fixed_size|private_segment_fixed_size|sgpr_count|"
    r"sgpr_spill_count|vgpr_count|vgpr_spill_count):\s+(\d+)$",
    re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disassembly", required=True, type=Path)
    parser.add_argument("--notes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def parse_disassembly(path: Path) -> dict[str, list[str]]:
    current: str | None = None
    functions: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        label = LABEL.match(line)
        if label:
            current = label.group(1)
            functions[current] = []
            continue
        if current is None:
            continue
        opcode = OPCODE.match(line)
        if opcode:
            instruction = opcode.group(1)
            functions[current].append(instruction)
            # llvm-objdump can decode code-object alignment bytes after this
            # terminator as s_code_end. They are unreachable padding, so do
            # not assign them to the kernel's static instruction count.
            if instruction == "s_endpgm":
                current = None
    return functions


def parse_notes(path: Path) -> dict[str, dict[str, int]]:
    """Parse compact resource fields emitted by llvm-readelf --notes."""

    resources: dict[str, dict[str, int]] = {}
    text = path.read_text(encoding="utf-8")
    for block in re.split(r"(?=^  - \.args:)", text, flags=re.MULTILINE):
        name = NOTE_NAME.search(block)
        if name is None:
            continue
        resources[name.group(1)] = {
            value.group(1): int(value.group(2)) for value in NOTE_VALUE.finditer(block)
        }
    return resources


def summarize_opcodes(opcodes: list[str]) -> dict[str, Any]:
    counts = Counter(opcodes)
    valu = sum(count for opcode, count in counts.items() if opcode.startswith("v_"))
    scalar = sum(count for opcode, count in counts.items() if opcode.startswith("s_"))
    memory = sum(
        count
        for opcode, count in counts.items()
        if opcode.startswith(("global_", "flat_", "buffer_", "ds_"))
    )
    dot4 = sum(count for opcode, count in counts.items() if opcode.startswith("v_dot4"))
    fp32_fma = sum(
        count
        for opcode, count in counts.items()
        if opcode.startswith(("v_fma_f32", "v_fmac_f32", "v_fma_mix_f32"))
    )
    packed_f16_fma = sum(
        count
        for opcode, count in counts.items()
        if "pk_fma_f16" in opcode or "pk_fmac_f16" in opcode
    )
    int_to_f32 = sum(
        count for opcode, count in counts.items() if opcode.startswith("v_cvt_f32_i")
    )
    half_to_f32 = sum(
        count for opcode, count in counts.items() if opcode.startswith("v_cvt_f32_f16")
    )
    bitfield_and_shift = sum(
        count
        for opcode, count in counts.items()
        if opcode.startswith(
            (
                "v_lshlrev_b16",
                "v_lshlrev_b32",
                "v_lshrrev_b32",
                "v_and_or_b32",
                "v_or_b32",
                "v_and_b32",
                "v_bfe_u32",
                "v_perm_b32",
            )
        )
    )
    return {
        "total_instructions": len(opcodes),
        "valu_instructions": valu,
        "scalar_instructions": scalar,
        "memory_instructions": memory,
        "dot4_i8_instructions": dot4,
        "fp32_fma_or_mix_instructions": fp32_fma,
        "packed_f16_fma_instructions": packed_f16_fma,
        "int_to_f32_conversion_instructions": int_to_f32,
        "half_to_f32_conversion_instructions": half_to_f32,
        "bitfield_and_shift_instructions": bitfield_and_shift,
        "per_128_elements": {
            "dot4_i8_instructions": dot4,
            "fp32_fma_or_mix_instructions": fp32_fma,
            "packed_f16_fma_instructions": packed_f16_fma,
            "int_to_f32_conversion_instructions": int_to_f32,
            "half_to_f32_conversion_instructions": half_to_f32,
            "bitfield_and_shift_instructions": bitfield_and_shift,
        },
        "per_element": {
            "dot4_i8_instructions": dot4 / PROBE_ELEMENTS,
            "fp32_fma_or_mix_instructions": fp32_fma / PROBE_ELEMENTS,
            "packed_f16_fma_instructions": packed_f16_fma / PROBE_ELEMENTS,
            "int_to_f32_conversion_instructions": int_to_f32 / PROBE_ELEMENTS,
            "half_to_f32_conversion_instructions": half_to_f32 / PROBE_ELEMENTS,
            "bitfield_and_shift_instructions": bitfield_and_shift / PROBE_ELEMENTS,
        },
        "opcode_counts": dict(sorted(counts.items())),
    }


def main() -> int:
    args = parse_args()
    functions = parse_disassembly(args.disassembly)
    resources = parse_notes(args.notes)
    missing = [kernel for kernel in KERNELS if kernel not in functions or kernel not in resources]
    if missing:
        raise SystemExit(f"missing expected kernels in ISA evidence: {missing}")
    payload = {
        "schema_version": "ullm.sq9-q8-gfx1030-isa-counts.v1",
        "target": "gfx1030",
        "probe_k": PROBE_ELEMENTS,
        "method": (
            "Counts are emitted instructions in the full fixed-K=128 kernel symbol, including "
            "prologue/epilogue in total_instructions. Per-element fields divide only the named "
            "opcode class by 128; they are static ISA counts, not timing measurements. "
            "bitfield_and_shift_instructions includes address arithmetic, so the raw opcode "
            "histogram is authoritative for separating SQ9_0 plane assembly from addressing."
        ),
        "kernels": {
            kernel: {
                "resources": resources[kernel],
                "instructions": summarize_opcodes(functions[kernel]),
            }
            for kernel in KERNELS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
