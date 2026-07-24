#!/usr/bin/python3.12
"""Preflight/execute exact AQ4 rollback or failed-activation recovery."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path


_BOOTSTRAP_LOCAL_MODULES = (
    "served_model_active_binding.py",
    "served_model_aq4_restoration_proof.py",
    "served_model_campaign_authorization.py",
    "served_model_campaign_plan.py",
    "served_model_campaign_runtime_seal.py",
    "served_model_campaign_source_seal.py",
    "served_model_final_activation.py",
)


def _bootstrap_production_tools() -> Path:
    wrapper = Path(__file__)
    tools = wrapper.parent
    root = tools.parent
    expected_argv = [
        "/usr/bin/python3.12",
        "-I",
        "-S",
        "-B",
        os.fspath(wrapper),
    ]
    if (
        os.geteuid() != 0
        or not wrapper.is_absolute()
        or Path(os.path.abspath(wrapper)) != wrapper
        or wrapper.resolve(strict=True) != wrapper
        or getattr(sys, "orig_argv", None)[:5] != expected_argv
        or not sys.flags.isolated
        or not sys.flags.no_site
        or not sys.flags.dont_write_bytecode
        or not sys.flags.safe_path
    ):
        raise RuntimeError(
            "production wrapper requires exact root "
            "/usr/bin/python3.12 -I -S -B absolute invocation"
        )
    ancestry: list[Path] = []
    selected = root
    while True:
        ancestry.append(selected)
        if selected.parent == selected:
            break
        selected = selected.parent
    for path in ancestry:
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise RuntimeError("production wrapper source ancestry is unsafe")
    for directory in (tools, root / ".git"):
        metadata = directory.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise RuntimeError("production wrapper source directory is unsafe")
    for path in (wrapper, *(tools / name for name in _BOOTSTRAP_LOCAL_MODULES)):
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise RuntimeError("production wrapper import source is unsafe")
    return tools


TOOLS = (
    _bootstrap_production_tools()
    if __name__ == "__main__"
    else Path(__file__).resolve().parent
)
if os.fspath(TOOLS) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS))

import served_model_final_activation as final_activation  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="restore AQ4 and run the plan-bound reverse/health operations",
    )
    parser.add_argument(
        "--recover-failed-activation",
        action="store_true",
        help=(
            "use the durable pre-switch intent to recover a crashed or "
            "failed_restore activation"
        ),
    )
    parser.add_argument("--confirm-plan-sha256")
    parser.add_argument("--confirmation")
    return parser.parse_args(argv)


def _require_mode(args: argparse.Namespace, observed_sha256: str) -> None:
    expected_confirmation = (
        final_activation.RECOVERY_CONFIRMATION
        if args.recover_failed_activation
        else final_activation.ROLLBACK_CONFIRMATION
    )
    if args.execute:
        if (
            args.confirm_plan_sha256 != observed_sha256
            or args.confirmation != expected_confirmation
        ):
            raise final_activation.FinalActivationError(
                "execute requires the exact plan SHA-256 and mode confirmation"
            )
    elif args.confirm_plan_sha256 is not None or args.confirmation is not None:
        raise final_activation.FinalActivationError(
            "confirmation arguments require --execute"
        )


def main(argv: Sequence[str] | None = None) -> int:
    exit_status = 0
    try:
        final_activation.require_production_entrypoint(Path(__file__))
        args = parse_args(argv)
        action = (
            "recovery"
            if args.recover_failed_activation
            else "rollback"
        )
        plan = final_activation.load_plan(
            args.plan,
            action=action,
            now=final_activation.utc_now(),
        )
        _require_mode(args, plan.snapshot.sha256)
        if args.execute:
            assert args.confirm_plan_sha256 is not None
            assert args.confirmation is not None
            if args.recover_failed_activation:
                completed = final_activation.execute_activation_recovery(
                    args.plan,
                    expected_plan_sha256=args.confirm_plan_sha256,
                    confirmation=args.confirmation,
                )
                schema_version = final_activation.ACTIVATION_RECOVERY_SCHEMA
            else:
                completed = final_activation.execute_rollback(
                    args.plan,
                    expected_plan_sha256=args.confirm_plan_sha256,
                    confirmation=args.confirmation,
                )
                schema_version = final_activation.ROLLBACK_OUTCOME_SCHEMA
            report = {
                "schema_version": schema_version,
                "status": completed.status,
                "outcome_path": os.fspath(completed.outcome_path),
                "outcome_sha256": completed.outcome_sha256,
            }
        else:
            report = final_activation.preflight_report(plan, action=action)
            if report["ready"] is not True:
                exit_status = 1
    except Exception:
        print("served-model rollback failed", file=sys.stderr)
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
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
