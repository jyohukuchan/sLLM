#!/usr/bin/env python3
"""Build and immutably publish pre-receipt SQ8 serving-promotion evidence."""

from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--ephemeral-manifest", required=True, type=Path)
    parser.add_argument("--cpu-cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        document = promotion.build_evidence(
            build_receipt_path=args.build_receipt,
            source_root=args.source_root,
            profile_path=args.profile,
            ephemeral_manifest_path=args.ephemeral_manifest,
            cpu_cases_path=args.cpu_cases,
        )
        digest = promotion.publish_immutable_json(args.output, document)
        promotion.validate_evidence(
            args.output,
            expected_profile_path=args.profile,
            source_root=args.source_root,
            require_receipt_absent=True,
        )
    except Exception:
        print("SQ8 serving promotion evidence failed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": document["schema_version"],
                "output": os.fspath(args.output.resolve()),
                "sha256": digest,
                "verified": True,
                "production_receipt_written": False,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
