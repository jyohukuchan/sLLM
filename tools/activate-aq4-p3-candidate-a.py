#!/usr/bin/env python3
"""Build or validate the production activation contract for AQ4 P3 Candidate A."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str, path: Path) -> Any:
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


SELECTOR = load_tool("aq4_p3_selector_for_activation", ROOT / "tools/select-aq4-p3-candidate.py")
QUALIFICATION = load_tool("aq4_p3_qualification_for_activation", ROOT / "tools/aq4_p3_upstream_qualification.py")
SCHEMA = "ullm.aq4_p3_candidate_a.production_activation.v1"
CANDIDATE_ID = "sequence-output-direct-v1"
CANDIDATE_FAMILY = "attention_recurrent"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_BYTES = 32 * 1024 * 1024


class ActivationError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        missing = sorted(fields - set(value)) if isinstance(value, dict) else sorted(fields)
        unknown = sorted(set(value) - fields) if isinstance(value, dict) else []
        raise ActivationError(f"{label} fields differ: missing={missing}, unknown={unknown}")
    return value


def digest(value: Any, label: str) -> str:
    if type(value) is not str or SHA_RE.fullmatch(value) is None:
        raise ActivationError(f"{label} must be lowercase SHA-256")
    return value


def snapshot(path: Path, label: str) -> tuple[dict[str, Any], bytes, str]:
    if not path.is_absolute() or path != path.resolve() or path.is_symlink():
        raise ActivationError(f"{label} path must be absolute, canonical, and symlink-free")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size <= 0 or before.st_size > MAX_BYTES:
            raise ActivationError(f"{label} must be a bounded single-link regular file")
        raw = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            raw.extend(chunk)
            if len(raw) > MAX_BYTES:
                raise ActivationError(f"{label} exceeds the byte bound")
        after = os.fstat(descriptor)
        current = path.lstat()
        identity = lambda item: (item.st_dev, item.st_ino, item.st_mode, item.st_nlink, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        if identity(before) != identity(after) or identity(before) != identity(current):
            raise ActivationError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(bytes(raw), object_pairs_hook=_pairs, parse_constant=lambda token: (_ for _ in ()).throw(ActivationError(f"non-finite JSON: {token}")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ActivationError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ActivationError(f"{label} must be an object")
    return value, bytes(raw), sha_bytes(bytes(raw))


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise ActivationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def ref(path: Path, label: str) -> dict[str, str]:
    _value, _raw, sha = snapshot(path, label)
    return {"path": str(path), "sha256": sha}


def self_hash(value: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))
    clone["activation_sha256"] = None
    return sha_bytes(canonical(clone))


def derive(selection_path: Path, raw_paths: list[Path]) -> dict[str, Any]:
    if not raw_paths:
        raise ActivationError("at least one promotion raw file is required")
    selection, _selection_raw, selection_file_sha = snapshot(selection_path, "selection artifact")
    sources: list[tuple[Any, dict[str, Any]]] = []
    raw_refs: list[dict[str, str]] = []
    for index, path in enumerate(raw_paths):
        value, raw, file_sha = snapshot(path, f"promotion raw {index}")
        source_snapshot = SELECTOR.Snapshot(path, SELECTOR.file_identity(path.lstat()), file_sha, raw)
        parsed = SELECTOR.validate_raw(value)
        if not parsed.promotion_eligible:
            raise ActivationError("diagnostic raw cannot activate production")
        sources.append((source_snapshot, value))
        raw_refs.append({"path": str(path), "sha256": file_sha, "evidence_sha256": parsed.semantic_sha256})
    recomputed = SELECTOR.select(sources)
    if not QUALIFICATION.strict_equal(selection, recomputed):
        raise ActivationError("selection artifact differs from independent recomputation")
    if selection.get("status") != "selected" or selection.get("selected_candidate_id") != CANDIDATE_ID:
        raise ActivationError("selection artifact does not select Candidate A")
    selected = next((item for item in selection.get("candidates", []) if isinstance(item, dict) and item.get("candidate_id") == CANDIDATE_ID), None)
    if not isinstance(selected, dict) or selected.get("eligible") is not True:
        raise ActivationError("Candidate A is not eligible in selection artifact")
    parsed_sources = [SELECTOR.validate_raw(value) for _, value in sources]
    identities = {json.dumps(source.identity, sort_keys=True) for source in parsed_sources}
    qualifications = {json.dumps(source.upstream_qualification, sort_keys=True) for source in parsed_sources}
    if len(identities) != 1 or len(qualifications) != 1:
        raise ActivationError("activation sources differ in build or qualification")
    build = json.loads(next(iter(identities)))
    qualification = json.loads(next(iter(qualifications)))
    if qualification["status"] != "qualified_go" or qualification["promotion_eligible"] is not True:
        raise ActivationError("activation requires qualified_go upstream P2 evidence")
    return {
        "candidate": {"candidate_id": CANDIDATE_ID, "family": CANDIDATE_FAMILY},
        "build": build,
        "profile": {"selection_file_sha256": selection_file_sha, "raw_evidence": sorted(raw_refs, key=lambda item: (item["evidence_sha256"], item["path"]))},
        "selection": {"path": str(selection_path), "sha256": selection_file_sha, "status": "selected", "selected_candidate_id": CANDIDATE_ID},
        "upstream_qualification": qualification,
    }


def build(selection_path: Path, raw_paths: list[Path]) -> dict[str, Any]:
    value = {"schema_version": SCHEMA, "status": "production_activated", "activation_sha256": None, **derive(selection_path, raw_paths)}
    value["activation_sha256"] = self_hash(value)
    return value


def validate(value: dict[str, Any]) -> dict[str, Any]:
    exact(value, {"schema_version", "status", "activation_sha256", "candidate", "build", "profile", "selection", "upstream_qualification"}, "activation")
    if value["schema_version"] != SCHEMA or value["status"] != "production_activated" or digest(value["activation_sha256"], "activation SHA-256") != self_hash(value):
        raise ActivationError("activation schema/status/self-hash differs")
    candidate = exact(value["candidate"], {"candidate_id", "family"}, "activation candidate")
    if candidate != {"candidate_id": CANDIDATE_ID, "family": CANDIDATE_FAMILY}:
        raise ActivationError("activation candidate differs")
    selection = exact(value["selection"], {"path", "sha256", "status", "selected_candidate_id"}, "activation selection")
    profile = exact(value["profile"], {"selection_file_sha256", "raw_evidence"}, "activation profile")
    if selection["status"] != "selected" or selection["selected_candidate_id"] != CANDIDATE_ID or selection["sha256"] != profile["selection_file_sha256"]:
        raise ActivationError("activation selection projection differs")
    raw_refs = profile["raw_evidence"]
    if not isinstance(raw_refs, list) or not raw_refs:
        raise ActivationError("activation raw evidence is absent")
    selection_path = Path(selection["path"])
    _selected, _raw, selected_sha = snapshot(selection_path, "activation selection")
    if selected_sha != digest(selection["sha256"], "activation selection SHA-256"):
        raise ActivationError("activation selection file differs")
    raw_paths: list[Path] = []
    for index, item in enumerate(raw_refs):
        item = exact(item, {"path", "sha256", "evidence_sha256"}, f"activation raw {index}")
        path = Path(item["path"])
        raw_value, _bytes, file_sha = snapshot(path, f"activation raw {index}")
        if file_sha != digest(item["sha256"], f"activation raw {index} file SHA-256") or SELECTOR.semantic_sha256(raw_value) != digest(item["evidence_sha256"], f"activation raw {index} semantic SHA-256"):
            raise ActivationError("activation raw reference differs")
        raw_paths.append(path)
    derived = derive(selection_path, raw_paths)
    for field in ("candidate", "build", "profile", "selection", "upstream_qualification"):
        if not QUALIFICATION.strict_equal(value[field], derived[field]):
            raise ActivationError(f"activation {field} differs from recomputed selection")
    return {"status": "valid_production_activation", "activation_sha256": value["activation_sha256"], "candidate_id": CANDIDATE_ID}


def publish(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ActivationError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
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
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("build")
    create.add_argument("--selection", type=Path, required=True)
    create.add_argument("--raw", type=Path, action="append", required=True)
    create.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("validate")
    check.add_argument("--activation", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            value = build(args.selection.resolve(), [path.resolve() for path in args.raw])
            validate(value); publish(args.output, value)
            result = validate(value)
        else:
            value, _raw, file_sha = snapshot(args.activation.resolve(), "activation")
            result = {**validate(value), "file_sha256": file_sha}
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, ActivationError, SELECTOR.SelectionError, QUALIFICATION.QualificationError) as error:
        print(f"AQ4 P3 Candidate A activation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
