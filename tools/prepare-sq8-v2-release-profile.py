#!/usr/bin/env python3
"""Freeze a worker-v2/reasoning SQ8 release profile from the legacy template."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Sequence

TOOLS_DIRECTORY = Path(__file__).resolve().parent
if os.fspath(TOOLS_DIRECTORY) not in sys.path:
    sys.path.insert(0, os.fspath(TOOLS_DIRECTORY))

import sq8_serving_promotion as promotion  # noqa: E402


RESULT_SCHEMA = "ullm.sq8_served_model_profile_preparation.v1"
MAX_PROFILE_BYTES = 1_048_576


class ProfileError(RuntimeError):
    """Raised when an SQ8 v2 profile cannot be immutably prepared."""


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    raw = promotion.stable_read(path, label, maximum=MAX_PROFILE_BYTES)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=promotion._without_duplicates,
            parse_constant=promotion._reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProfileError(f"{label} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ProfileError(f"{label} root is not an object")
    return value


def prepare(
    *,
    base_profile: Path,
    worker: Path,
    serving_receipt: Path,
    output: Path,
) -> dict[str, Any]:
    base = _strict_json(base_profile, "base SQ8 served-model profile")
    if (
        set(base)
        != {
            "schema_version",
            "public",
            "generation",
            "format",
            "tokenizer",
            "worker",
            "product",
            "promotion",
        }
        or base.get("schema_version") != "ullm.served_model.profile.v1"
        or not isinstance(base.get("format"), dict)
        or base["format"].get("format_id") != promotion.FORMAT_ID
        or not isinstance(base.get("worker"), dict)
        or base["worker"].get("protocol") != "ullm.worker.v1"
    ):
        raise ProfileError("base SQ8 profile contract differs")
    worker = worker.absolute()
    try:
        worker_metadata = worker.lstat()
    except OSError as error:
        raise ProfileError("final SQ8 worker is unavailable") from error
    if (
        worker.name != "ullm-sq8-worker"
        or stat.S_ISLNK(worker_metadata.st_mode)
        or not stat.S_ISREG(worker_metadata.st_mode)
        or stat.S_IMODE(worker_metadata.st_mode) != 0o555
        or worker_metadata.st_nlink != 1
    ):
        raise ProfileError("final SQ8 worker immutable identity differs")
    _worker_bytes, worker_sha256 = promotion.stable_hash(
        worker,
        "final SQ8 worker",
        required_mode=0o555,
        required_nlink=1,
    )
    product = base.get("product")
    if not isinstance(product, dict) or not isinstance(product.get("root"), str):
        raise ProfileError("base SQ8 product identity differs")
    product_root = Path(product["root"]).absolute()
    serving_receipt = serving_receipt.absolute()
    if (
        serving_receipt.parent != product_root
        or serving_receipt.name in {"", ".", "..", "promotion.json"}
    ):
        raise ProfileError(
            "SQ8 serving receipt must be a distinct file in the product root"
        )
    if serving_receipt.exists() or serving_receipt.is_symlink():
        raise ProfileError("SQ8 serving receipt target already exists")
    worker_profile = dict(base["worker"])
    worker_profile["protocol"] = promotion.WORKER_PROTOCOL
    worker_profile["binary"] = os.fspath(worker)
    artifact = product.get("artifact")
    if not isinstance(artifact, dict):
        raise ProfileError("base SQ8 artifact profile differs")
    product_profile = {
        **product,
        "artifact": {
            **artifact,
            "content_sha256_from_receipt": [
                "product",
                "artifact_content_sha256",
            ],
        },
    }
    document = {
        "schema_version": "ullm.served_model.profile.v1",
        "public": base["public"],
        "generation": base["generation"],
        "format": base["format"],
        "tokenizer": base["tokenizer"],
        "worker": worker_profile,
        "reasoning": dict(promotion.REASONING_CONTRACT),
        "product": product_profile,
        "promotion": {
            "receipt": os.fspath(serving_receipt),
            "source_commit_from_receipt": ["source_commit"],
            "required_schema_version": promotion.RECEIPT_SCHEMA,
            "evidence_from_receipt": ["evidence", "path"],
            "evidence_sha256_from_receipt": ["evidence", "sha256"],
        },
    }
    try:
        promotion._profile_contract(document)
    except Exception as error:
        raise ProfileError("prepared SQ8 v2 profile contract differs") from error
    try:
        promotion.publish_immutable_json(output.absolute(), document)
    except Exception as error:
        raise ProfileError("SQ8 v2 profile publication failed") from error
    observed = _strict_json(output.absolute(), "published SQ8 v2 profile")
    if observed != document:
        raise ProfileError("published SQ8 v2 profile bytes differ")
    raw = promotion.stable_read(
        output.absolute(),
        "published SQ8 v2 profile",
        maximum=MAX_PROFILE_BYTES,
        required_mode=0o444,
        required_nlink=1,
    )
    return {
        "schema_version": RESULT_SCHEMA,
        "profile": os.fspath(output.absolute()),
        "profile_sha256": hashlib.sha256(raw).hexdigest(),
        "worker": os.fspath(worker),
        "worker_sha256": worker_sha256,
        "serving_receipt": os.fspath(serving_receipt),
        "worker_protocol": promotion.WORKER_PROTOCOL,
        "reasoning_dialect": promotion.REASONING_DIALECT,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-profile", required=True, type=Path)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--serving-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = prepare(
            base_profile=args.base_profile,
            worker=args.worker,
            serving_receipt=args.serving_receipt,
            output=args.output,
        )
    except Exception:
        print("SQ8 v2 release profile preparation failed", file=sys.stderr)
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
