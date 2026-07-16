#!/usr/bin/env python3
"""Validate and immutably finalize a P3 qualification-only no-eligible package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = (
    "p3-source.tar",
    "qualification-only-raw.json",
    "selection.json",
    "upstream-qualification.json",
)


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


SELECTOR = load("aq4_p3_selector_for_finalizer", ROOT / "tools/select-aq4-p3-candidate.py")
QUALIFICATION = SELECTOR.QUALIFICATION


class FinalizeError(ValueError):
    pass


def sha_file(path: Path) -> str:
    snapshot = SELECTOR.capture(path)
    return snapshot.sha256


def finalize(root: Path) -> dict[str, str]:
    if not root.is_absolute() or root != root.resolve() or root.is_symlink() or not root.is_dir():
        raise FinalizeError("package root must be an absolute canonical directory")
    if (root / "SHA256SUMS").exists() or (root / "SHA256SUMS").is_symlink():
        raise FinalizeError("refusing to overwrite SHA256SUMS")
    if {item.name for item in root.iterdir()} != set(INVENTORY):
        raise FinalizeError("package inventory differs")
    for name in INVENTORY:
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise FinalizeError(f"package file identity differs: {name}")
    qualification_snapshot = SELECTOR.capture(root / "upstream-qualification.json")
    qualification = SELECTOR.parse_json(qualification_snapshot)
    result = QUALIFICATION.validate(qualification)
    if result["status"] != "valid_rejected_no_go":
        raise FinalizeError("package qualification is not rejected_no_go")
    raw_snapshot = SELECTOR.capture(root / "qualification-only-raw.json")
    raw = SELECTOR.parse_json(raw_snapshot)
    source = SELECTOR.validate_raw(raw)
    if source.promotion_eligible or source.p3_implementation is None:
        raise FinalizeError("package raw is not qualification-only")
    expected_q = raw["upstream_qualification"]
    if expected_q["path"] != str(qualification_snapshot.path) or expected_q["sha256"] != qualification_snapshot.sha256:
        raise FinalizeError("package raw qualification binding differs")
    archive = raw["p3_implementation"]["source_archive"]
    if archive != {"path": str((root / "p3-source.tar").resolve()), "sha256": sha_file(root / "p3-source.tar")}:
        raise FinalizeError("package source archive binding differs")
    selection_snapshot = SELECTOR.capture(root / "selection.json")
    selection = SELECTOR.parse_json(selection_snapshot)
    expected_selection = SELECTOR.select([(raw_snapshot, raw)])
    if not QUALIFICATION.strict_equal(selection, expected_selection):
        raise FinalizeError("package selection differs from recomputation")
    if selection["status"] != "no_eligible_candidate" or selection["selected_candidate_id"] is not None or selection["input_binding"]["upstream_qualification_status"] != "rejected_no_go":
        raise FinalizeError("package selection terminal state differs")
    hashes = {name: sha_file(root / name) for name in INVENTORY}
    sums_path = root / "SHA256SUMS"
    descriptor = os.open(sums_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o444)
    try:
        payload = "".join(f"{hashes[name]}  {name}\n" for name in INVENTORY).encode("ascii")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    for name in (*INVENTORY, "SHA256SUMS"):
        os.chmod(root / name, 0o444, follow_symlinks=False)
    os.chmod(root, 0o555, follow_symlinks=False)
    return {
        "status": "immutable_no_eligible",
        "qualification_sha256": result["qualification_sha256"],
        "raw_evidence_sha256": source.semantic_sha256,
        "selection_file_sha256": selection_snapshot.sha256,
        "sha256sums_sha256": hashlib.sha256(sums_path.read_bytes()).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(finalize(args.root.resolve()), sort_keys=True))
        return 0
    except (OSError, ValueError, FinalizeError, SELECTOR.SelectionError, QUALIFICATION.QualificationError) as error:
        print(f"AQ4 P3 no-eligible finalization failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
