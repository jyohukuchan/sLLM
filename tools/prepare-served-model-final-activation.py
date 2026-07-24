#!/usr/bin/python3.12
"""Prepare an immutable, evidence-bound SQ8 final activation plan."""

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
    try:
        final_activation.require_production_entrypoint(Path(__file__))
        args = parse_args(argv)
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
