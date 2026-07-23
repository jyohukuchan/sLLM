#!/usr/bin/env python3
"""Atomically publish one reviewed cross-model v2 campaign authorization."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from served_model_campaign_authorization import (
    AuthorizationError,
    issue_authorization,
    strict_json_bytes,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        raw = args.document.read_bytes()
        document = strict_json_bytes(raw, "authorization input")
        record = issue_authorization(
            document,
            args.output,
            now=datetime.now(timezone.utc),
            source_root=args.source_root,
        )
    except (OSError, AuthorizationError):
        print("campaign authorization issuance failed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": record.document["schema_version"],
                "authorization_id": record.document["authorization_id"],
                "authorization_sha256": record.snapshot.sha256,
                "output": str(record.snapshot.path),
            },
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
