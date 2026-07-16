#!/usr/bin/env python3
"""Build a metric-free P3 diagnostic raw artifact from an upstream P2 rejection."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


SELECTOR = load("aq4_p3_selector_for_qualification_only", ROOT / "tools/select-aq4-p3-candidate.py")
QUALIFICATION = SELECTOR.QUALIFICATION


class RawError(ValueError):
    pass


def build(qualification_path: Path, commit: str, tree_oid: str, archive_path: Path) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RawError("commit must be lowercase 40-hex")
    if re.fullmatch(r"[0-9a-f]{40}", tree_oid) is None:
        raise RawError("tree OID must be lowercase 40-hex")
    qualification_snapshot = SELECTOR.capture(qualification_path)
    qualification = SELECTOR.parse_json(qualification_snapshot)
    result = QUALIFICATION.validate(qualification)
    if result["status"] != "valid_rejected_no_go":
        raise RawError("qualification-only raw requires rejected_no_go")
    archive = SELECTOR.capture(archive_path)
    value = {
        "schema_version": SELECTOR.RAW_SCHEMA,
        "status": "qualification_only_diagnostic",
        "measurement_eligible": False,
        "promotion_eligible": False,
        "promotion_ineligibility_reason": result["reason"],
        "evidence_sha256": None,
        "upstream_qualification": {
            "path": str(qualification_snapshot.path),
            "sha256": qualification_snapshot.sha256,
            "qualification_sha256": result["qualification_sha256"],
            "status": "rejected_no_go",
            "promotion_eligible": False,
            "reason": result["reason"],
        },
        "p3_implementation": {
            "candidate_id": "sequence-output-direct-v1",
            "family": "attention_recurrent",
            "commit": commit,
            "tree_oid": tree_oid,
            "source_archive": {"path": str(archive.path), "sha256": archive.sha256},
            "build_status": "not_built_for_promotion",
            "profile_status": "not_measured",
            "runtime_default": "off",
        },
    }
    value["evidence_sha256"] = SELECTOR.semantic_sha256(value)
    SELECTOR.validate_raw(value)
    return value


def publish(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RawError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode("ascii") + b"\n")
            handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
    finally:
        if temporary.exists(): temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tree-oid", required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = build(args.qualification.resolve(), args.commit, args.tree_oid, args.source_archive.resolve())
        publish(args.output, value)
        print(json.dumps({"status": value["status"], "evidence_sha256": value["evidence_sha256"], "promotion_eligible": False}, sort_keys=True))
        return 0
    except (OSError, ValueError, RawError, SELECTOR.SelectionError, QUALIFICATION.QualificationError) as error:
        print(f"AQ4 P3 qualification-only raw failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
