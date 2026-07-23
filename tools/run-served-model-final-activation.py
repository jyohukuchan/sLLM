#!/usr/bin/env python3
"""Preflight by default, or explicitly execute one immutable final SQ8 plan."""

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
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform the production transition; omitted means read-only preflight",
    )
    parser.add_argument("--confirm-plan-sha256")
    parser.add_argument("--confirmation")
    return parser.parse_args(argv)


def _require_mode(args: argparse.Namespace, observed_sha256: str) -> None:
    if args.execute:
        if (
            args.confirm_plan_sha256 != observed_sha256
            or args.confirmation != final_activation.ACTIVATION_CONFIRMATION
        ):
            raise final_activation.FinalActivationError(
                "execute requires the exact plan SHA-256 and activation confirmation"
            )
    elif args.confirm_plan_sha256 is not None or args.confirmation is not None:
        raise final_activation.FinalActivationError(
            "confirmation arguments require --execute"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = final_activation.load_plan(
            args.plan,
            action="activate",
            now=final_activation.utc_now(),
        )
        _require_mode(args, plan.snapshot.sha256)
        if args.execute:
            assert args.confirm_plan_sha256 is not None
            assert args.confirmation is not None
            completed = final_activation.execute_activation(
                args.plan,
                expected_plan_sha256=args.confirm_plan_sha256,
                confirmation=args.confirmation,
            )
            report = {
                "schema_version": final_activation.ACTIVATION_OUTCOME_SCHEMA,
                "status": completed.status,
                "outcome_path": os.fspath(completed.outcome_path),
                "outcome_sha256": completed.outcome_sha256,
            }
        else:
            report = final_activation.preflight_report(plan, action="activate")
    except Exception:
        print("final served-model activation failed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            report,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
