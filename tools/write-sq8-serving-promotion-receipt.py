#!/usr/bin/env python3
"""Validate SQ8 evidence and immutably publish its serving-promotion receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import sq8_serving_promotion as promotion


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="current sealed source root; mandatory for build receipt v2",
    )
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = promotion.write_receipt(
            profile_path=args.profile,
            evidence_path=args.evidence,
            output_path=args.output,
            source_root=args.source_root,
        )
    except Exception:
        print("SQ8 serving promotion receipt publication failed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            receipt,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
