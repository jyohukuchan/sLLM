#!/usr/bin/env python3
"""Preflight or execute the fixed, source-bound AQ4-to-SQ8 campaign plan."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOOLS = Path(__file__).resolve().parent
if os.fspath(TOOLS) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS))

import served_model_campaign_authorization as authorization  # noqa: E402
import served_model_campaign_plan as plan  # noqa: E402
from served_model_campaign_transaction import (  # noqa: E402
    TransactionError,
    TransactionFailed,
    TransactionRequest,
    default_inactive_checker,
    execute_transaction,
    preflight,
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
            "ullm.served_model.v2_cross_model_campaign_execution_failure.v1"
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
    args = parse_args(argv)
    try:
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
                    "ullm.served_model.v2_cross_model_campaign_preflight.v1"
                ),
                "ready": True,
                "plan_id": plan.PLAN_ID,
                "authorization_sha256": result.authorization.snapshot.sha256,
                "source_commit": result.source_commit,
                "source_tree": result.source_tree,
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
                    "ullm.served_model.v2_cross_model_campaign_execution.v1"
                ),
                "plan_id": plan.PLAN_ID,
                "status": completed.status,
                "outcome_path": os.fspath(completed.outcome_path),
                "outcome_sha256": completed.outcome_sha256,
            }
    except TransactionFailed as error:
        _emit(_failure_report(error), stream=sys.stderr)
        return 1
    except (authorization.AuthorizationError, plan.PlanError, TransactionError):
        print("cross-model campaign transaction failed closed", file=sys.stderr)
        return 1
    _emit(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
