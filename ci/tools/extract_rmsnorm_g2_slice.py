#!/usr/bin/env python3
"""Synthetic-fixture-only G2 slice extractor entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import ContractError, ROOT, read_json  # noqa: E402
from validate_rmsnorm_g2_contracts import TENSOR_NAME, extract_synthetic_slice, extract_verified_slice_payload  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", type=Path)
    source.add_argument("--cache-root", type=Path)
    parser.add_argument("--tensor", default=TENSOR_NAME)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.tensor != TENSOR_NAME:
            raise ContractError("--tensor is not the locked G2 RMSNorm tensor")
        declared = read_json(args.record)
        if args.fixture is not None:
            if not args.fixture.is_file() or args.fixture.is_symlink():
                raise ContractError("--fixture must be a regular synthetic fixture")
            record = extract_synthetic_slice(args.fixture, declared, ROOT)
        else:
            if args.cache_root is None:
                raise ContractError("--cache-root is required for verified cache extraction")
            record, _payload = extract_verified_slice_payload(args.cache_root, declared, ROOT)
        args.output.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    except (ContractError, OSError, ValueError) as exc:
        print(f"G2 slice extraction: FAIL: {exc}", file=sys.stderr)
        return 1
    print("G2 slice record: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
