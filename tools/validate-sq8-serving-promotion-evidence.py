#!/usr/bin/env python3
"""Validate immutable SQ8 serving-promotion evidence and all live bindings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import sq8_serving_promotion as promotion


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument(
        "--require-pre-receipt-state",
        action="store_true",
        help="also require that the profile's final serving receipt is still absent",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        document = promotion.validate_evidence(
            args.evidence,
            expected_profile_path=args.profile,
            require_receipt_absent=args.require_pre_receipt_state,
        )
    except Exception:
        print("SQ8 serving promotion evidence validation failed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": "ullm.sq8_serving_promotion_evidence_validator.v1",
                "input_schema_version": document["schema_version"],
                "verified": True,
                "source_commit": document["source"]["commit"],
                "worker_binary_sha256": document["worker"]["sha256"],
                "cpu_case_count": document["cpu_cases"]["case_count"],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
