#!/usr/bin/python3
"""Read-only preflight by default; execute one confirmed AQ4 hardening plan."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path


def _bootstrap() -> Path:
    wrapper = Path(__file__)
    tools = wrapper.parent
    root = tools.parent
    expected = ["/usr/bin/python3", "-I", "-S", "-B", os.fspath(wrapper)]
    if (
        os.geteuid() != 0
        or not wrapper.is_absolute()
        or Path(os.path.abspath(wrapper)) != wrapper
        or wrapper.resolve(strict=True) != wrapper
        or getattr(sys, "orig_argv", None)[:5] != expected
        or not sys.flags.isolated
        or not sys.flags.no_site
        or not sys.flags.dont_write_bytecode
        or not sys.flags.safe_path
    ):
        raise RuntimeError("production wrapper requires exact root /usr/bin/python3 -I -S -B invocation")
    for path in (root, tools, wrapper, tools / "aq4_runtime_hardening_activation.py"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) & 0o022:
            raise RuntimeError("production activation control source is unsafe")
    return tools


TOOLS = _bootstrap() if __name__ == "__main__" else Path(__file__).resolve().parent
if os.fspath(TOOLS) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS))

import aq4_runtime_hardening_activation as activation  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--isolated-candidate-preflight", action="store_true")
    parser.add_argument("--confirm-plan-sha256")
    parser.add_argument("--confirmation")
    return parser.parse_args(argv)


def _require_mode(args: argparse.Namespace, observed_sha256: str) -> None:
    if args.execute:
        if (
            args.confirm_plan_sha256 != observed_sha256
            or args.confirmation != activation.ACTIVATION_CONFIRMATION
        ):
            raise activation.ActivationError("execute requires exact plan SHA-256 and literal confirmation")
    elif args.confirm_plan_sha256 is not None or args.confirmation is not None:
        raise activation.ActivationError("confirmation arguments require --execute")
    if args.execute and args.isolated_candidate_preflight:
        raise activation.ActivationError("isolated candidate preflight cannot accompany --execute")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        record = activation.load_plan(args.plan)
        _require_mode(args, record.snapshot.sha256)
        if args.execute:
            result = activation.execute_activation(
                args.plan,
                expected_plan_sha256=args.confirm_plan_sha256,
                confirmation=args.confirmation,
            )
            report: dict[str, object] = {
                "schema_version": activation.OUTCOME_SCHEMA,
                "status": result.status,
                "outcome_path": os.fspath(result.path),
                "outcome_sha256": result.sha256,
            }
        elif args.isolated_candidate_preflight:
            receipt = activation.run_isolated_candidate_preflight(record)
            report = activation.preflight_report(record)
            report["isolated_candidate_preflight"] = {
                "path": os.fspath(receipt.path),
                "sha256": receipt.sha256,
            }
        else:
            report = activation.preflight_report(record)
    except Exception:
        print("AQ4 hardening activation failed", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0 if args.execute or report["ready"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
