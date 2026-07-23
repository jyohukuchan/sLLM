#!/usr/bin/env python3
"""Prepare an immutable, evidence-bound SQ8 final activation plan."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path


TOOLS = Path(__file__).resolve().parent
if os.fspath(TOOLS) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS))

import served_model_final_activation as final_activation  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--active-manifest", required=True, type=Path)
    parser.add_argument("--rollback-manifest", required=True, type=Path)
    parser.add_argument("--release-bundle", required=True, type=Path)
    parser.add_argument("--systemd-unit", required=True, type=Path)
    parser.add_argument("--environment-file", required=True, type=Path)
    parser.add_argument("--reviewed-operations", required=True, type=Path)
    parser.add_argument("--activation-outcome", required=True, type=Path)
    parser.add_argument("--rollback-outcome", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        document = final_activation.prepare_plan(
            plan_id=args.plan_id,
            authorization_path=args.authorization,
            candidate_manifest=args.candidate_manifest,
            active_manifest=args.active_manifest,
            rollback_manifest=args.rollback_manifest,
            release_bundle=args.release_bundle,
            systemd_unit=args.systemd_unit,
            environment_file=args.environment_file,
            operations_document=args.reviewed_operations,
            activation_outcome=args.activation_outcome,
            rollback_outcome=args.rollback_outcome,
            output=args.output,
            now=final_activation.utc_now(),
        )
        raw = args.output.read_bytes()
    except Exception:
        print("final activation plan preparation failed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": document["schema_version"],
                "plan_id": document["plan_id"],
                "plan_path": os.fspath(args.output.resolve()),
                "plan_sha256": final_activation._sha256(raw),
                "production_activation_performed": False,
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
