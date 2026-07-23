#!/usr/bin/env python3
"""Publish one immutable AQ4 bundle-v1 validator report.

The historical AQ4 bundle-v1 generator consumes separately published release
and browser validator reports.  Their validators intentionally remain
read-only CLIs, so this helper recomputes one report from an immutable private
copy and publishes the exact canonical result without replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if os.fspath(TOOLS) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS))

from served_model_active_binding import (  # noqa: E402
    ActiveBindingError,
    stable_read_regular,
)

RELEASE_VALIDATOR = ROOT / "tools/validate-generic-reasoning-release.py"
BROWSER_VALIDATOR = (
    ROOT / "tools/validate-openwebui-reasoning-browser-smoke.py"
)
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_REPORT_BYTES = 16 * 1024 * 1024
EXPECTED_INPUT_SCHEMAS = {
    "release": {"ullm.generic_reasoning_release_evidence.v1"},
    "browser": {
        "ullm.openwebui.reasoning_browser_smoke.v1",
        "ullm.openwebui.reasoning_browser_smoke.v2",
    },
}
EXPECTED_REPORT_SCHEMAS = {
    "release": "ullm.generic_reasoning_release_validator.v1",
    "browser": "ullm.openwebui.reasoning_browser_smoke_validator.v1",
}


class ReportPublicationError(RuntimeError):
    """An AQ4 bundle-v1 validator report could not be safely published."""


Validator = Callable[[Path], dict[str, Any]]


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ReportPublicationError("evidence contains duplicate fields")
        value[key] = child
    return value


def _reject_constant(_value: str) -> None:
    raise ReportPublicationError("evidence contains a non-finite number")


def _stable_read_immutable(path: Path, label: str, maximum: int) -> bytes:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise ReportPublicationError(f"{label} path is not canonical absolute")
    try:
        snapshot = stable_read_regular(
            path,
            label,
            maximum=maximum,
            require_single_link=True,
            require_read_only=True,
        )
    except ActiveBindingError as error:
        raise ReportPublicationError(
            f"{label} identity differs or is unavailable"
        ) from error
    if stat.S_IMODE(snapshot.identity.mode) != 0o444:
        raise ReportPublicationError(f"{label} identity differs")
    return snapshot.raw


def _strict_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReportPublicationError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ReportPublicationError(f"{label} root is not an object")
    return value


def _canonical_json(document: dict[str, Any]) -> bytes:
    try:
        raw = (
            json.dumps(
                document,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError) as error:
        raise ReportPublicationError(
            "validator report is not canonicalizable"
        ) from error
    if not raw or len(raw) > MAX_REPORT_BYTES:
        raise ReportPublicationError("validator report exceeds its bound")
    return raw


def _load_validator(kind: str) -> Validator:
    path = RELEASE_VALIDATOR if kind == "release" else BROWSER_VALIDATOR
    name = f"_ullm_aq4_report_publisher_{kind}_validator"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing.validate
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ReportPublicationError("validator module is unavailable")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    if not isinstance(module, ModuleType) or not callable(
        getattr(module, "validate", None)
    ):
        raise ReportPublicationError("validator entrypoint is unavailable")
    return module.validate


def _publish_no_replace(path: Path, raw: bytes) -> None:
    if not path.is_absolute() or Path(os.path.abspath(path)) != path:
        raise ReportPublicationError("output path is not canonical absolute")
    if path.exists() or path.is_symlink():
        raise ReportPublicationError("output already exists")
    try:
        parent = path.parent.resolve(strict=True)
        parent_metadata = parent.lstat()
    except OSError as error:
        raise ReportPublicationError("output parent is unavailable") from error
    if (
        parent != path.parent
        or stat.S_ISLNK(parent_metadata.st_mode)
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_mode & stat.S_IWOTH
    ):
        raise ReportPublicationError("output parent identity differs")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if not hasattr(os, "O_NOFOLLOW"):
        raise ReportPublicationError("O_NOFOLLOW is required")
    flags |= os.O_NOFOLLOW
    parent_descriptor = os.open(parent, flags)
    temporary_name = f".{path.name}.incomplete-{uuid.uuid4().hex}"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
            dir_fd=parent_descriptor,
        )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ReportPublicationError(
                    "validator report write made no progress"
                )
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise ReportPublicationError("output already exists") from error
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)
    if _stable_read_immutable(path, "published validator report", MAX_REPORT_BYTES) != raw:
        raise ReportPublicationError("published validator report bytes differ")


def publish(
    *,
    kind: str,
    evidence: Path,
    output: Path,
    require_complete: bool,
    validator: Validator | None = None,
) -> dict[str, Any]:
    if kind not in EXPECTED_INPUT_SCHEMAS:
        raise ReportPublicationError("validator kind differs")
    evidence = evidence.absolute()
    output = output.absolute()
    evidence_raw = _stable_read_immutable(
        evidence,
        f"{kind} evidence",
        MAX_EVIDENCE_BYTES,
    )
    document = _strict_object(evidence_raw, f"{kind} evidence")
    if document.get("schema_version") not in EXPECTED_INPUT_SCHEMAS[kind]:
        raise ReportPublicationError(
            f"{kind} evidence is not an AQ4 bundle-v1 schema"
        )

    selected_validator = validator or _load_validator(kind)
    with tempfile.TemporaryDirectory(prefix="ullm-aq4-validator-") as temporary:
        isolated = Path(temporary) / evidence.name
        isolated.write_bytes(evidence_raw)
        isolated.chmod(0o444)
        try:
            report = selected_validator(isolated)
        except Exception as error:
            raise ReportPublicationError(
                f"{kind} evidence validation failed"
            ) from error
    if not isinstance(report, dict):
        raise ReportPublicationError("validator report root differs")
    if report.get("schema_version") != EXPECTED_REPORT_SCHEMAS[kind]:
        raise ReportPublicationError("validator report schema differs")
    if require_complete and report.get("gate_eligible") is not True:
        raise ReportPublicationError("validator report is not gate eligible")
    if (
        _stable_read_immutable(
            evidence,
            f"{kind} evidence",
            MAX_EVIDENCE_BYTES,
        )
        != evidence_raw
    ):
        raise ReportPublicationError("evidence changed during validation")

    raw = _canonical_json(report)
    _publish_no_replace(output, raw)
    if _strict_object(raw, "validator report") != report:
        raise ReportPublicationError("published validator report differs")
    return {
        "kind": kind,
        "evidence": os.fspath(evidence),
        "evidence_sha256": hashlib.sha256(evidence_raw).hexdigest(),
        "output": os.fspath(output),
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "gate_eligible": report.get("gate_eligible") is True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("release", "browser"), required=True)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = publish(
            kind=args.kind,
            evidence=args.evidence,
            output=args.output,
            require_complete=args.require_complete,
        )
    except Exception:
        print("validator report publication failed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            result,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
