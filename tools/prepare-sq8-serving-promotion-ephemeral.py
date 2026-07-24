#!/usr/bin/env python3
"""Publish an SQ8 pre-receipt scaffold and strict ephemeral v2 manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import sq8_serving_promotion as promotion


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-receipt", required=True, type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="current sealed source root; mandatory for build receipt v2",
    )
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = promotion.prepare_ephemeral_manifest(
            build_receipt_path=args.build_receipt,
            source_root=args.source_root,
            profile_path=args.profile,
            receipt_output_path=args.receipt_output,
            manifest_output_path=args.manifest_output,
        )
    except Exception:
        print("SQ8 ephemeral serving manifest preparation failed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": "ullm.sq8_serving_promotion_ephemeral.v1",
                "receipt_path": result["receipt_path"],
                "receipt_sha256": result["receipt_sha256"],
                "manifest_path": result["manifest_path"],
                "manifest_sha256": result["manifest_sha256"],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
