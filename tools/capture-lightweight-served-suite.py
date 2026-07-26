#!/usr/bin/env python3
"""Capture the fixed lightweight generation suite from one served manifest.

This is read-only evidence collection.  It does not swap a manifest, restart a
service, create a release, or make a promotion decision.  The implementation
intentionally reuses the request, readiness, text-analysis, and strict-output
rules shared by the generic promotion tool.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_promotion_module() -> ModuleType:
    path = ROOT / "tools" / "lightweight_promotion.py"
    spec = importlib.util.spec_from_file_location("capture_lightweight_promotion", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared promotion helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROMOTION = load_promotion_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROMOTION.DEFAULT_ACTIVE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-suite", type=Path, default=PROMOTION.DEFAULT_PROMPT_SUITE)
    parser.add_argument("--base-url", default=PROMOTION.DEFAULT_BASE_URL)
    parser.add_argument("--gateway-container", default=PROMOTION.DEFAULT_GATEWAY_CONTAINER)
    parser.add_argument("--token-file", type=Path, default=PROMOTION.DEFAULT_TOKEN_FILE)
    parser.add_argument("--request-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--ready-timeout-seconds", type=float, default=60.0)
    return parser.parse_args()


def manifest_identity(document: dict[str, Any]) -> str:
    public = document.get("public")
    if not isinstance(public, dict) or not isinstance(public.get("id"), str) or not public["id"]:
        raise SystemExit("manifest has no valid public.id")
    return public["id"]


def main() -> int:
    args = parse_args()
    args.manifest = args.manifest.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.prompt_suite = args.prompt_suite.expanduser().resolve()
    args.token_file = args.token_file.expanduser().resolve()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to use an existing output directory: {args.output_dir}")
    if args.request_timeout_seconds <= 0 or args.ready_timeout_seconds <= 0:
        raise SystemExit("timeouts must be positive")
    manifest = PROMOTION.read_snapshot(args.manifest, "served manifest")
    document = PROMOTION.strict_object(manifest.raw, "served manifest")
    model_id = manifest_identity(document)
    suite = PROMOTION.load_suite(args.prompt_suite)
    base_url = PROMOTION._validate_base_url(args.base_url)
    gateway_container = PROMOTION.normalize_gateway_container(args.gateway_container)
    token = PROMOTION.read_token(args.token_file)
    args.output_dir.mkdir(parents=True, mode=0o750)
    try:
        ready_attempts = PROMOTION.wait_for_live_gateway(
            base_url=base_url,
            token=token,
            model_id=model_id,
            timeout_seconds=args.ready_timeout_seconds,
            gateway_container=gateway_container,
        )
        records = PROMOTION.run_suite(
            suite=suite,
            model_id=model_id,
            manifest_document=document,
            base_url=base_url,
            token=token,
            request_timeout_seconds=args.request_timeout_seconds,
            output_dir=args.output_dir / "cases",
            gateway_container=gateway_container,
        )
        blocking = [
            f"{record['case_id']}:{flag}"
            for record in records
            for flag in record.get("analysis", {}).get("blocking", [])
        ]
        output = {
            "schema_version": "ullm.lightweight_served_suite_capture.v1",
            "captured_at": PROMOTION.utc_now(),
            "manifest_path": str(args.manifest),
            "manifest_sha256": manifest.sha256,
            "model_id": model_id,
            "prompt_suite": str(args.prompt_suite),
            "prompt_suite_sha256": PROMOTION.sha256(PROMOTION.read_snapshot(args.prompt_suite, "prompt suite").raw),
            "gateway_container": args.gateway_container,
            "ready_attempts": ready_attempts,
            "case_count": len(records),
            "blocking_findings": blocking,
            "passed": not blocking,
        }
        PROMOTION.write_json_new(args.output_dir / "capture.json", output, "served suite capture")
    except BaseException:
        # Keep a partially captured case directory for diagnosis, but do not
        # stamp it as a successful evidence set.
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
