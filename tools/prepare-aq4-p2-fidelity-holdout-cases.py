#!/usr/bin/env python3
"""Materialize only the frozen 24 AQ4 P2 holdout rows as Rust source cases.

This adapter performs no model work.  It never rewrites the calibration cases and publishes a
single create-new source-cases file whose IDs and token arrays are bound to holdout-cases.jsonl.
"""

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
SCHEMA = "ullm.qwen35_aq4_source_calibration_cases.v1"
RECEIPT_SCHEMA = "ullm.aq4_p2_fidelity_holdout_source_cases.v1"
MAX_ROWS = 24


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROTOCOL = _load("aq4_holdout_cases_protocol", "generate-aq4-p2-fidelity-holdout.py")
SPLIT = _load("aq4_holdout_cases_split", "validate-aq4-p2-fidelity-holdout.py")


class CasesError(ValueError):
    pass


def _regular(path: Path, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            info = os.lstat(current)
        except OSError as error:
            raise CasesError(f"{label} unavailable: {error}") from error
        if stat.S_ISLNK(info.st_mode):
            raise CasesError(f"{label} has symlink component: {current}")
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise CasesError(f"{label} must be a single-link regular file")


def _sha(path: Path, label: str) -> str:
    _regular(path, label)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic(path: Path, value: Any) -> str:
    if os.path.lexists(path):
        raise CasesError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    absolute = path.parent.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if stat.S_ISLNK(os.lstat(current).st_mode):
            raise CasesError(f"output parent has symlink component: {current}")
    encoded = (
        json.dumps(
            value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False
        ).encode()
        + b"\n"
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        os.unlink(temporary)
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(encoded).hexdigest()


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    try:
        SPLIT.validate(args.split_root)
    except Exception as error:
        raise CasesError(f"split validation failed: {error}") from error
    manifest_raw = (args.split_root / "split-manifest.json").read_bytes()
    policy_raw = (args.split_root / "policy.json").read_bytes()
    calibration_sha = _sha(
        args.split_root / "calibration-cases.jsonl", "calibration cases"
    )
    holdout_path = args.split_root / "holdout-cases.jsonl"
    holdout_sha = _sha(holdout_path, "holdout cases")
    expected = {
        "split_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "policy_sha256": hashlib.sha256(policy_raw).hexdigest(),
        "calibration_cases_sha256": calibration_sha,
        "holdout_cases_sha256": holdout_sha,
    }
    supplied = {
        "split_manifest_sha256": args.expected_split_manifest_sha256,
        "policy_sha256": args.expected_policy_sha256,
        "calibration_cases_sha256": args.expected_calibration_cases_sha256,
        "holdout_cases_sha256": args.expected_holdout_cases_sha256,
    }
    if supplied != expected:
        raise CasesError(
            "split/policy/calibration/holdout SHA does not match the pinned contract"
        )
    rows = PROTOCOL.read_jsonl(holdout_path, "holdout cases")
    if len(rows) != MAX_ROWS or any(row.get("subset") != "holdout" for row in rows):
        raise CasesError("holdout cases must contain exactly 24 holdout rows")
    cases = []
    seen: set[str] = set()
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in seen:
            raise CasesError("holdout case IDs are not unique")
        seen.add(case_id)
        fixture = Path(row["fixture_path"])
        if not fixture.is_absolute():
            fixture = (args.split_root / fixture).resolve()
        if _sha(fixture, f"fixture {case_id}") != row.get("fixture_sha256"):
            raise CasesError(f"fixture SHA differs: {case_id}")
        value, _ = PROTOCOL.load(fixture, f"fixture {case_id}")
        item = value.get("cases", [{}])[0]
        tokens = item.get("prompt_token_ids")
        if (
            item.get("case_id") != case_id
            or not isinstance(tokens, list)
            or len(tokens) != row.get("prompt_tokens")
            or PROTOCOL.sha_bytes(PROTOCOL.canonical(tokens))
            != row.get("prompt_token_ids_sha256")
            or PROTOCOL.context_hash(tokens) != row.get("context_token_ids_sha256")
        ):
            raise CasesError(f"fixture/token identity differs: {case_id}")
        cases.append(
            {
                "case_id": case_id,
                "prompt_token_ids": tokens,
                "step_count": 1,
                "semantic_input_id": case_id,
                "observation": "fidelity_holdout_full_context_step0",
            }
        )
    payload = {"schema_version": SCHEMA, "cases": cases}
    output_sha = _atomic(args.output, payload)
    info = os.lstat(args.output)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "ready",
        "subset": "holdout",
        "observation": "fidelity_holdout_full_context_step0",
        "row_count": MAX_ROWS,
        "cases": {
            "path": str(args.output.resolve()),
            "sha256": output_sha,
            "bytes": info.st_size,
            "mode": f"{stat.S_IMODE(info.st_mode):04o}",
            "nlink": info.st_nlink,
        },
        "split": expected,
    }
    receipt_sha = _atomic(args.receipt_output, receipt)
    return {
        "status": "ok",
        "row_count": MAX_ROWS,
        "subset": "holdout",
        "output": str(args.output),
        "output_sha256": output_sha,
        "receipt": str(args.receipt_output),
        "receipt_sha256": receipt_sha,
        **expected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--expected-split-manifest-sha256", required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--expected-calibration-cases-sha256", required=True)
    parser.add_argument("--expected-holdout-cases-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(prepare(args), ensure_ascii=True, sort_keys=True))
        return 0
    except (CasesError, OSError, ValueError) as error:
        print(f"AQ4 P2 holdout cases preparation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
