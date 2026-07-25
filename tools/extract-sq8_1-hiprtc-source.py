#!/usr/bin/env python3
"""Extract the exact SQ8_1 HIPRTC raw literal for offline ISA compilation."""

from __future__ import annotations

import argparse
from pathlib import Path


PREFIX = 'return R"(\n'
SUFFIX = '\n)";'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-existing-identical", action="store_true")
    args = parser.parse_args()
    source = args.runtime_source.read_text(encoding="utf-8")
    try:
        start = source.index(PREFIX) + len(PREFIX)
        end = source.index(SUFFIX, start)
    except ValueError as exc:
        raise SystemExit("could not locate the SQ8_1 HIPRTC raw literal") from exc
    kernel_source = source[start:end]
    if "ullm_sq8_1_matvec_w8a8_explicit_f32_kernel" not in kernel_source:
        raise SystemExit("extracted source does not contain the SQ8_1 W8A8 kernel")
    output = "#include <hip/hip_runtime.h>\n" + kernel_source + "\n"
    if args.output.exists():
        if args.allow_existing_identical and args.output.read_text(encoding="utf-8") == output:
            return 0
        raise SystemExit(f"refusing to overwrite extracted source: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
