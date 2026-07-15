#!/usr/bin/env python3
"""Independently validate a Qwen3.5 AQ4/SQ8 calibration freeze receipt."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("qwen35_aq4_sq8_fidelity_protocol", HERE / "qwen35_aq4_sq8_fidelity_protocol.py")
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("SQ8 protocol is unavailable")
PROTOCOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROTOCOL)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(PROTOCOL.validate_freeze(args.plan, args.metrics, args.freeze), sort_keys=True))
        return 0
    except (PROTOCOL.ProtocolError, OSError, ValueError) as error:
        print(f"SQ8 freeze validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
