#!/usr/bin/env python3
"""Extract SQ8_1 dot4 opcode and static kernel resources from ISA evidence."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_KERNEL = "ullm_sq8_1_dot4_i32_i8_probe"
ARCHITECTURES = ("gfx1030", "gfx1100", "gfx1201", "gfx942", "gfx950")
LEGACY_DOT_ARCHITECTURES = {"gfx1030", "gfx942", "gfx950"}
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
    parser.add_argument("--arch", required=True, choices=ARCHITECTURES)
    parser.add_argument("--kernel", default=DEFAULT_KERNEL)
    parser.add_argument("--require-signed-dot", action="store_true")
    parser.add_argument("--disassembly", required=True, type=Path)
    parser.add_argument("--notes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def kernel_opcodes(path: Path, kernel: str) -> list[str]:
    current: str | None = None
    opcodes: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        label = LABEL.match(line)
        if label:
            current = label.group(1)
            continue
        if current != kernel:
            continue
        opcode = OPCODE.match(line)
        if opcode:
            value = opcode.group(1)
            opcodes.append(value)
            if value == "s_endpgm":
                break
    if not opcodes:
        raise SystemExit(f"missing expected kernel in {path}: {kernel}")
    return opcodes


def kernel_resources(path: Path, kernel: str) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    # ROCm metadata may begin a kernel with `.args` or `.agpr_count`; split
    # at every top-level kernel mapping rather than assuming one ordering.
    for block in re.split(r"(?=^  - \.[a-z_]+:)", text, flags=re.MULTILINE):
        name = NOTE_NAME.search(block)
        if name and name.group(1) == kernel:
            return {match.group(1): int(match.group(2)) for match in NOTE_VALUE.finditer(block)}
    raise SystemExit(f"missing expected kernel metadata in {path}: {kernel}")


def main() -> int:
    args = parse_args()
    opcodes = kernel_opcodes(args.disassembly, args.kernel)
    dot_opcodes = [opcode for opcode in opcodes if opcode.startswith("v_dot4")]
    expected: str | None = None
    if args.require_signed_dot:
        if not dot_opcodes:
            raise SystemExit(f"{args.arch} probe emitted no v_dot4 instruction")
        expected = "v_dot4c_i32_i8" if args.arch in LEGACY_DOT_ARCHITECTURES else "v_dot4_i32_iu8"
        if not any(opcode.startswith(expected) for opcode in dot_opcodes):
            raise SystemExit(
                f"{args.arch} probe did not emit expected {expected}; observed {sorted(set(dot_opcodes))}"
            )
    payload = {
        "schema_version": "ullm.sq8_1.isa.v0.1",
        "arch": args.arch,
        "kernel": args.kernel,
        "semantic": "signed_i8_x_signed_i8_to_i32" if args.require_signed_dot else None,
        "expected_opcode": expected,
        "dot4_opcodes": dot_opcodes,
        "opcode_counts": dict(sorted(Counter(opcodes).items())),
        "resources": kernel_resources(args.notes, args.kernel),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
