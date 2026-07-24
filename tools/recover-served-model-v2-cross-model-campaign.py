#!/usr/bin/python3.12
"""Preflight or execute the fixed locked AQ4 campaign recovery route."""

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
    "served_model_campaign_recovery.py",
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
import served_model_campaign_recovery as recovery  # noqa: E402
from served_model_campaign_transaction import (  # noqa: E402
    TransactionError,
    TransactionRequest,
)

require_production_entrypoint = (
    campaign_entrypoint.require_production_entrypoint
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute-recovery", action="store_true")
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--confirm-authorization-sha256")
    parser.add_argument(
        "--command-timeout-seconds",
        type=float,
        default=1_800.0,
    )
    return parser.parse_args(argv)


def _existing(path: Path, label: str) -> Path:
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise recovery.RecoveryError(f"{label} is unavailable") from error


def _lexical_absolute(path: Path, label: str) -> Path:
    """Validate an identity-only path without requiring its former leaf."""

    text = os.fspath(path)
    if (
        not path.is_absolute()
        or text.startswith("//")
        or path.name in {"", ".", ".."}
        or ".." in path.parts
    ):
        raise recovery.RecoveryError(
            f"{label} is not a canonical absolute path"
        )
    normalized = Path(os.path.abspath(text))
    if normalized != path or os.fspath(normalized) != text:
        raise recovery.RecoveryError(
            f"{label} is not a canonical absolute path"
        )
    return normalized


def _request(
    args: argparse.Namespace,
    claim: authorization.ClaimRecord,
) -> TransactionRequest:
    source_root = _existing(args.source_root, "source root")
    if source_root != REPOSITORY_ROOT:
        raise recovery.RecoveryError(
            "recovery runner must execute from the sealed source root"
        )
    # Recovery recognizes SQ8 from the authorization and actual active bytes.
    # The original candidate pathname is retained only as a plan identity and
    # deliberately need not survive a failed candidate window.
    candidate = _lexical_absolute(
        args.candidate_manifest,
        "candidate manifest",
    )
    commands = plan.derive_commands(
        source_root=source_root,
        authorization_path=claim.authorization.snapshot.path,
        candidate_manifest=candidate,
        authorization_document=claim.authorization.document,
    )
    return TransactionRequest(
        authorization_path=claim.authorization.snapshot.path,
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
        now = datetime.now(timezone.utc)
        claim = authorization.load_claim(args.authorization, now=now)
        if args.execute_recovery:
            if (
                args.confirm_authorization_sha256 is None
                or args.confirm_authorization_sha256
                != claim.authorization.snapshot.sha256
            ):
                raise recovery.RecoveryError(
                    "recovery requires the exact authorization SHA-256"
                )
        elif args.confirm_authorization_sha256 is not None:
            raise recovery.RecoveryError(
                "authorization confirmation is only valid with --execute-recovery"
            )
        request = _request(args, claim)
        if args.preflight_only:
            pinned = recovery.preflight_recovery(request, now=now)
            report = {
                "schema_version": (
                    "ullm.served_model.v2_cross_model_campaign_"
                    "recovery_preflight.v2"
                ),
                "ready": True,
                "plan_id": plan.PLAN_ID,
                "authorization_sha256": (
                    pinned.claim.authorization.snapshot.sha256
                ),
                "claim_sha256": pinned.claim.snapshot.sha256,
                "source_seal_sha256": (
                    pinned.transaction_preflight.source_seal.fingerprint_sha256
                ),
                "active_state": pinned.active_state,
                "active_manifest_sha256": pinned.active_before.sha256,
                "backup_manifest_sha256": pinned.backup.sha256,
                "backup_requires_atomic_bootstrap": (
                    pinned.backup_requires_publication
                ),
                "claim_created": False,
                "active_manifest_changed": False,
            }
        else:
            result = recovery.recover_transaction(request)
            report = {
                "schema_version": (
                    "ullm.served_model.v2_cross_model_campaign_recovery_result.v2"
                ),
                "plan_id": plan.PLAN_ID,
                "status": result.status,
                "receipt_path": os.fspath(result.receipt_path),
                "receipt_sha256": result.receipt_sha256,
            }
    except recovery.RecoveryFailed as error:
        _emit(
            {
                "schema_version": (
                    "ullm.served_model.v2_cross_model_campaign_"
                    "recovery_failure.v2"
                ),
                "status": error.result.status,
                "failure_stage": error.result.failure_stage,
                "receipt_path": os.fspath(error.result.receipt_path),
                "receipt_sha256": error.result.receipt_sha256,
            },
            stream=sys.stderr,
        )
        return 1
    except (
        authorization.AuthorizationError,
        campaign_entrypoint.ProductionEntrypointError,
        plan.PlanError,
        recovery.RecoveryError,
        TransactionError,
    ):
        print("cross-model campaign recovery failed closed", file=sys.stderr)
        return 1
    _emit(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
