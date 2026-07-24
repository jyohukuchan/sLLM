#!/usr/bin/python3.12
"""Atomically publish one reviewed cross-model v2 campaign authorization."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path


_BOOTSTRAP_LOCAL_MODULES = (
    "served_model_aq4_restoration_proof.py",
    "served_model_campaign_authorization.py",
    "served_model_campaign_entrypoint.py",
    "served_model_campaign_source_seal.py",
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

import served_model_campaign_entrypoint as campaign_entrypoint  # noqa: E402
from served_model_campaign_authorization import (
    AuthorizationError,
    issue_authorization,
    strict_json_bytes,
)  # noqa: E402


require_production_entrypoint = (
    campaign_entrypoint.require_production_entrypoint
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        require_production_entrypoint(Path(__file__))
        args = parse_args(argv)
        raw = args.document.read_bytes()
        document = strict_json_bytes(raw, "authorization input")
        record = issue_authorization(
            document,
            args.output,
            now=datetime.now(timezone.utc),
            source_root=args.source_root,
        )
    except (
        OSError,
        AuthorizationError,
        campaign_entrypoint.ProductionEntrypointError,
    ):
        print("campaign authorization issuance failed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": record.document["schema_version"],
                "authorization_id": record.document["authorization_id"],
                "authorization_sha256": record.snapshot.sha256,
                "output": str(record.snapshot.path),
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
