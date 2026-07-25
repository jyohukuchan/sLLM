#!/usr/bin/python3
"""Prepare one immutable AQ4_0 runtime-hardening activation plan."""

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
        if (
            stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
            or (path.is_file() and metadata.st_nlink != 1)
        ):
            raise RuntimeError("production activation control source is unsafe")
    return tools


TOOLS = _bootstrap() if __name__ == "__main__" else Path(__file__).resolve().parent
if os.fspath(TOOLS) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS))

import aq4_runtime_hardening_activation as activation  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--protected-root", required=True, type=Path)
    parser.add_argument("--control-source", required=True, type=Path)
    parser.add_argument("--control-source-commit", required=True)
    parser.add_argument("--control-tool", required=True, action="append", type=Path)
    parser.add_argument("--promotion-source", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--active-manifest", required=True, type=Path)
    parser.add_argument("--rollback-manifest", required=True, type=Path)
    parser.add_argument("--systemd-unit", required=True, type=Path)
    parser.add_argument("--environment-file", required=True, type=Path)
    parser.add_argument("--credential-file", required=True, action="append", type=Path)
    parser.add_argument("--operations", required=True, type=Path)
    parser.add_argument("--lock-path", default=activation.DEFAULT_LOCK_PATH, type=Path)
    parser.add_argument("--activation-intent", required=True, type=Path)
    parser.add_argument("--activation-outcome", required=True, type=Path)
    parser.add_argument("--activation-recovery", required=True, type=Path)
    parser.add_argument("--rollback-outcome", required=True, type=Path)
    parser.add_argument("--candidate-live-proof", required=True, type=Path)
    parser.add_argument("--rollback-live-proof", required=True, type=Path)
    parser.add_argument("--recovery-audit-directory", required=True, type=Path)
    parser.add_argument("--rollback-audit-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-model-id", default=activation.AQ4_MODEL_ID)
    parser.add_argument(
        "--expected-worker-sha256", default=activation.EXPECTED_AQ4_WORKER_SHA256
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        activation.prepare_plan(
            plan_id=args.plan_id,
            protected_root=args.protected_root,
            control_source=args.control_source,
            control_source_commit=args.control_source_commit,
            control_tool_paths=args.control_tool,
            promotion_source=args.promotion_source,
            candidate_manifest=args.candidate_manifest,
            active_manifest=args.active_manifest,
            rollback_manifest=args.rollback_manifest,
            systemd_unit=args.systemd_unit,
            environment_file=args.environment_file,
            credential_files=args.credential_file,
            operations_document=args.operations,
            lock_path=args.lock_path,
            activation_intent=args.activation_intent,
            activation_outcome=args.activation_outcome,
            activation_recovery=args.activation_recovery,
            rollback_outcome=args.rollback_outcome,
            candidate_live_proof=args.candidate_live_proof,
            rollback_live_proof=args.rollback_live_proof,
            recovery_audit_directory=args.recovery_audit_directory,
            rollback_audit_directory=args.rollback_audit_directory,
            output=args.output,
            expected_model_id=args.expected_model_id,
            expected_worker_sha256=args.expected_worker_sha256,
        )
        record = activation.load_plan(args.output)
        report = activation.preflight_report(record)
    except Exception:
        print("AQ4 hardening activation plan preparation failed", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0 if report["ready"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
