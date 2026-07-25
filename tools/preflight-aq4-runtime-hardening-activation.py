#!/usr/bin/env python3
"""Read-only AQ4 hardening pre-plan admission report.

This command is deliberately usable before the sealed control-source clone and
immutable activation plan exist.  It never creates a path or touches
``active.json``; once a plan exists, use run-aq4-runtime-hardening-activation.py
for the stronger plan-bound preflight.
"""

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

import aq4_runtime_hardening_activation as activation  # noqa: E402


DEFAULT_ROOT = Path("/opt/ullm/aq4-runtime-hardening-v0.1")
DEFAULT_ACTIVE_SHA256 = "5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--active-manifest", type=Path, default=Path("/etc/ullm/served-models/active.json"))
    parser.add_argument("--expected-active-sha256", default=DEFAULT_ACTIVE_SHA256)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--rollback-manifest", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--control-source-parent", type=Path)
    parser.add_argument("--operations", type=Path)
    parser.add_argument("--lock-path", type=Path, default=activation.DEFAULT_LOCK_PATH)
    parser.add_argument("--systemd-unit", type=Path, default=Path("/etc/systemd/system/ullm-openai.service"))
    parser.add_argument("--environment-file", type=Path, default=Path("/etc/ullm/openai-gateway-manifest.env"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.protected_root
    report = activation.preplan_preflight_report(
        active_manifest=args.active_manifest,
        expected_active_sha256=args.expected_active_sha256,
        protected_root=root,
        candidate_manifest=args.candidate_manifest or root / "manifests/aq4-hardened-frozen.json",
        rollback_manifest=args.rollback_manifest or root / "activation/rollback-active-5d015a013dcf70ce.json",
        plan_path=args.plan or root / "activation/activation-plan.json",
        control_source_parent=args.control_source_parent or root / "control-source",
        operations_document=args.operations or root / "activation/reviewed-operations.json",
        lock_path=args.lock_path,
        systemd_unit=args.systemd_unit,
        environment_file=args.environment_file,
    )
    raw = (json.dumps(report, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n").encode("ascii")
    if args.output is not None:
        descriptor = -1
        try:
            descriptor = os.open(
                args.output,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o644,
            )
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise RuntimeError("preflight evidence write made no progress")
                view = view[written:]
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        parent = os.open(args.output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    print(raw.decode("ascii"), end="")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
