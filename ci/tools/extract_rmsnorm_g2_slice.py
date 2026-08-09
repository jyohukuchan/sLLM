#!/usr/bin/env python3
"""Synthetic-fixture-only G2 slice extractor entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import ContractError, ROOT, read_json  # noqa: E402
from validate_rmsnorm_g2_contracts import extract_synthetic_slice  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if not args.fixture.is_file() or args.fixture.is_symlink():
            raise ContractError("--fixture must be a regular synthetic fixture")
        record = extract_synthetic_slice(args.fixture, read_json(args.record), ROOT)
        args.output.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    except (ContractError, OSError, ValueError) as exc:
        print(f"G2 slice extraction: FAIL: {exc}", file=sys.stderr)
        return 1
    print("G2 synthetic slice record: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
