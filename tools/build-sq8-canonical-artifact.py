#!/usr/bin/env python3
"""Import source FP8 weights and block scales into an SQ8_0 canonical artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sq8_canonical_artifact import (
    DEFAULT_COPY_CHUNK_BYTES,
    ArtifactError,
    build_canonical_artifact,
    sha256_file,
    verify_canonical_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", type=Path)
    parser.add_argument("--output-artifact", required=True, type=Path)
    parser.add_argument("--tensor-name", action="append", default=[])
    parser.add_argument(
        "--copy-chunk-bytes",
        type=int,
        default=DEFAULT_COPY_CHUNK_BYTES,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.verify_only:
            summary = verify_canonical_artifact(args.output_artifact)
        else:
            if args.source_model_dir is None:
                raise ArtifactError("--source-model-dir is required unless --verify-only is used")
            build_canonical_artifact(
                args.source_model_dir,
                args.output_artifact,
                tensor_names=args.tensor_name or None,
                copy_chunk_bytes=args.copy_chunk_bytes,
                overwrite=args.overwrite,
            )
            summary = verify_canonical_artifact(args.output_artifact)
            summary["artifact_manifest_sha256"] = sha256_file(
                args.output_artifact / "sq_manifest.json"
            )
            summary["artifact"] = str(args.output_artifact)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except ArtifactError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
