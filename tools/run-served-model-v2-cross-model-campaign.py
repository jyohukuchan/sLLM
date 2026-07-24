#!/usr/bin/python3.12
"""Preflight or execute the fixed, source-bound AQ4-to-SQ8 campaign plan."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_BOOTSTRAP_LOCAL_MODULES = (
    "served_model_active_binding.py",
    "served_model_aq4_restoration_proof.py",
    "served_model_campaign_authorization.py",
    "served_model_campaign_entrypoint.py",
    "served_model_campaign_plan.py",
    "served_model_campaign_runtime_seal.py",
    "served_model_campaign_source_seal.py",
    "served_model_campaign_transaction.py",
    "sq8_serving_promotion.py",
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
REPOSITORY_ROOT = TOOLS.parent
if os.fspath(TOOLS) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS))

import served_model_campaign_authorization as authorization  # noqa: E402
import served_model_campaign_entrypoint as campaign_entrypoint  # noqa: E402
import served_model_campaign_plan as plan  # noqa: E402
from served_model_campaign_transaction import (  # noqa: E402
    TransactionError,
    TransactionFailed,
    TransactionRequest,
    default_inactive_checker,
    execute_transaction,
    preflight,
)

require_production_entrypoint = (
    campaign_entrypoint.require_production_entrypoint
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="read and validate only; never claim, switch, or run campaign commands",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="claim once and execute the reviewed fixed production plan",
    )
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument(
        "--confirm-authorization-sha256",
        help="required with --execute; exact SHA-256 printed by preflight",
    )
    parser.add_argument(
        "--command-timeout-seconds",
        type=float,
        default=1_800.0,
        help="per-command timeout, bounded by the transaction implementation",
    )
    return parser.parse_args(argv)


def _canonical_existing(path: Path, label: str) -> Path:
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise TransactionError(f"{label} is unavailable") from error


def _request(
    args: argparse.Namespace,
    record: authorization.AuthorizationRecord,
) -> TransactionRequest:
    source_root = _canonical_existing(args.source_root, "source root")
    if source_root != REPOSITORY_ROOT:
        raise TransactionError(
            "campaign runner must execute from the sealed source root"
        )
    candidate = _canonical_existing(
        args.candidate_manifest,
        "candidate served-model manifest",
    )
    commands = plan.derive_commands(
        source_root=source_root,
        authorization_path=record.snapshot.path,
        candidate_manifest=candidate,
        authorization_document=record.document,
    )
    return TransactionRequest(
        authorization_path=record.snapshot.path,
        source_root=source_root,
        candidate_manifest=candidate,
        active_manifest=plan.ACTIVE_MANIFEST,
        systemd_unit=plan.SYSTEMD_UNIT,
        environment_file=plan.ENVIRONMENT_FILE,
        inactive_services=plan.INACTIVE_SERVICES,
        commands=commands,
        command_timeout_seconds=args.command_timeout_seconds,
        service_unit=plan.SERVICE_UNIT,
        api_key_file=plan.API_KEY_FILE,
        openwebui_session_token_file=plan.OPENWEBUI_SESSION_TOKEN_FILE,
    )


def _failure_report(error: TransactionFailed) -> dict[str, Any]:
    restoration = error.restoration
    return {
        "schema_version": (
            "ullm.served_model.v2_cross_model_campaign_execution_failure.v2"
        ),
        "status": error.result.status,
        "outcome_path": os.fspath(error.result.outcome_path),
        "outcome_sha256": error.result.outcome_sha256,
        "backup_path": os.fspath(error.backup_path),
        "restoration": {
            "expected_manifest_sha256": restoration.get(
                "expected_manifest_sha256"
            ),
            "observed_manifest_sha256": restoration.get(
                "observed_manifest_sha256"
            ),
            "bytes_equal": restoration.get("bytes_equal") is True,
            "reverse_reconciliation_passed": restoration.get(
                "reverse_reconciliation_passed"
            )
            is True,
            "final_checks_passed": restoration.get("final_checks_passed")
            is True,
            "live_proof_present": isinstance(restoration.get("proof"), dict),
        },
    }


def _emit(value: dict[str, Any], *, stream: Any = sys.stdout) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        file=stream,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        require_production_entrypoint(Path(__file__))
        args = parse_args(argv)
        if args.execute and os.geteuid() != 0:
            raise TransactionError(
                "campaign execution requires the root transaction supervisor"
            )
        now = datetime.now(timezone.utc)
        record = authorization.load_authorization(
            args.authorization,
            now=now,
            require_fresh_outputs=True,
        )
        if args.execute:
            if (
                args.confirm_authorization_sha256 is None
                or args.confirm_authorization_sha256
                != record.snapshot.sha256
            ):
                raise TransactionError(
                    "execution requires the exact preflight authorization SHA-256"
                )
        elif args.confirm_authorization_sha256 is not None:
            raise TransactionError(
                "authorization SHA-256 confirmation is only valid with --execute"
            )
        request = _request(args, record)
        if args.preflight_only:
            result = preflight(request, now=now)
            default_inactive_checker(request.inactive_services)
            report = {
                "schema_version": (
                    "ullm.served_model.v2_cross_model_campaign_preflight.v2"
                ),
                "ready": True,
                "plan_id": plan.PLAN_ID,
                "authorization_sha256": result.authorization.snapshot.sha256,
                "source_commit": result.source_commit,
                "source_tree": result.source_tree,
                "source_seal_sha256": result.source_seal.fingerprint_sha256,
                "aq4_source_seal_sha256": (
                    result.aq4_source_seal.fingerprint_sha256
                    if result.aq4_source_seal is not None
                    else None
                ),
                "before_manifest_sha256": result.active.sha256,
                "candidate_manifest_sha256": result.candidate.sha256,
                "candidate_worker_binary_sha256": result.candidate_summary[
                    "worker"
                ]["binary_sha256"],
                "active_manifest": os.fspath(plan.ACTIVE_MANIFEST),
                "service_unit": plan.SERVICE_UNIT,
                "claim_created": False,
                "active_manifest_changed": False,
            }
        else:
            completed = execute_transaction(request)
            report = {
                "schema_version": (
                    "ullm.served_model.v2_cross_model_campaign_execution.v2"
                ),
                "plan_id": plan.PLAN_ID,
                "status": completed.status,
                "outcome_path": os.fspath(completed.outcome_path),
                "outcome_sha256": completed.outcome_sha256,
            }
    except TransactionFailed as error:
        _emit(_failure_report(error), stream=sys.stderr)
        return 1
    except (
        authorization.AuthorizationError,
        campaign_entrypoint.ProductionEntrypointError,
        plan.PlanError,
        TransactionError,
    ):
        print("cross-model campaign transaction failed closed", file=sys.stderr)
        return 1
    _emit(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
