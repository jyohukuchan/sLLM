#!/usr/bin/env python3
"""Validation-only tooling for an already-created semantic RMSNorm G1 aggregate.

This module deliberately has no aggregation, report import, descriptor handoff,
writer, or PASS-returning API.  The only process allowed to construct semantic
G1 evidence is the fixed controller implementation that exists exclusively in
``orchestrate_rmsnorm_g1_evidence.py`` when that file is executed as
``__main__``.  A standalone invocation can inspect a document, but cannot
promote caller-provided bytes into evidence or create output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from common import ContractError, ROOT  # noqa: E402
import validate_rmsnorm_g1_contracts as contracts  # noqa: E402


class AggregateError(ContractError):
    """A standalone aggregate input is malformed or outside canonical scope."""


def validate_document_only(document: Any, *, repo: Path = ROOT) -> None:
    """Refuse positive validation without the live controller anchors."""

    del document, repo
    raise AggregateError("standalone semantic G1 validation is permanently fail-closed; only the live controller may validate PASS")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--document", type=Path, help="existing aggregate JSON to validate; no output is ever written")
    result.add_argument("--repo", type=Path, default=ROOT, help=argparse.SUPPRESS)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.document is None:
        print("semantic RMSNorm G1 aggregate: emission is disabled; --document validates only", file=sys.stderr)
        return 2
    try:
        repo = contracts.canonical_repository(args.repo)
        document_path = Path(args.document).resolve(strict=True)
        document = contracts.read_json(document_path)
        validate_document_only(document, repo=repo)
    except (AggregateError, ContractError, OSError, ValueError) as exc:
        print(f"semantic RMSNorm G1 aggregate validation: FAIL: {exc}", file=sys.stderr)
        return 1
    print("semantic RMSNorm G1 aggregate validation: FAIL-CLOSED (no standalone PASS authority)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
