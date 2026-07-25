#!/usr/bin/env python3
"""Build or verify an isolated SQ8_1 artifact from a verified SQ8_0 source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sq8_1_artifact import ArtifactError, build_sq8_1_artifact, sha256_file, verify_sq8_1_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sq8_0-artifact", type=Path)
    parser.add_argument("--output-artifact", required=True, type=Path)
    parser.add_argument("--tensor-name", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.verify_only:
            summary = verify_sq8_1_artifact(args.output_artifact)
        else:
            if args.source_sq8_0_artifact is None:
                raise ArtifactError("--source-sq8_0-artifact is required unless --verify-only is used")
            build_sq8_1_artifact(
                args.source_sq8_0_artifact,
                args.output_artifact,
                tensor_names=args.tensor_name or None,
                overwrite=args.overwrite,
            )
            summary = verify_sq8_1_artifact(args.output_artifact)
            summary["artifact"] = str(args.output_artifact)
            summary["artifact_manifest_sha256"] = sha256_file(args.output_artifact / "sq8_1_manifest.json")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except ArtifactError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
