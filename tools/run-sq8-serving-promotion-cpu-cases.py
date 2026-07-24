#!/usr/bin/env python3
"""Run and immutably publish the GPU-hidden SQ8 v2 CPU admission cases."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

import sq8_serving_promotion as promotion


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    cargo = shutil.which("cargo")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-receipt", required=True, type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="current sealed source root; mandatory for build receipt v2",
    )
    parser.add_argument("--ephemeral-manifest", required=True, type=Path)
    parser.add_argument(
        "--cargo",
        type=Path,
        default=None if cargo is None else Path(cargo).absolute(),
    )
    parser.add_argument(
        "--python",
        required=True,
        type=Path,
        help="Python executable from the reviewed gateway test environment",
    )
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cargo is None:
        print("SQ8 serving CPU cases failed", file=sys.stderr)
        return 1
    try:
        document = promotion.build_cpu_cases_report(
            build_receipt_path=args.build_receipt,
            source_root=args.source_root,
            ephemeral_manifest_path=args.ephemeral_manifest,
            cargo_path=args.cargo,
            python_path=args.python,
            target_dir=args.target_dir,
        )
        digest = promotion.publish_immutable_json(args.output, document)
        build = promotion.validate_build_receipt(
            args.build_receipt,
            source_root=args.source_root,
        )
        promotion.validate_cpu_cases(
            args.output,
            source_root=promotion.resolve_build_source_root(
                build,
                args.source_root,
            ),
            source_commit=document["source_commit"],
            source_tree=document["source_tree"],
            manifest_sha256=document["served_model_manifest_sha256"],
            worker_sha256=document["worker_binary_sha256"],
            reasoning=promotion.REASONING_CONTRACT,
        )
    except Exception:
        print("SQ8 serving CPU cases failed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": document["schema_version"],
                "output": os.fspath(args.output.resolve()),
                "sha256": digest,
                "test_run_count": document["summary"]["test_run_count"],
                "case_count": document["summary"]["pass_count"],
                "verified": True,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
