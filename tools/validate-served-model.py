#!/usr/bin/env python3
"""Validate one served-model manifest and print its non-secret identity."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
LOADER_PATH = ROOT / "services/openai-gateway/src/ullm_openai_gateway/served_model.py"
SUMMARY_SCHEMA = "ullm.served_model.validation.v1"


def load_gateway_validator() -> ModuleType:
    """Load the reviewed gateway validator without importing gateway startup code."""

    package_root = LOADER_PATH.parents[1]
    if os.fspath(package_root) not in sys.path:
        sys.path.insert(0, os.fspath(package_root))
    package_name = "ullm_openai_gateway"
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [os.fspath(package_root / package_name)]  # type: ignore[attr-defined]
        package.__package__ = package_name
        sys.modules[package_name] = package
    module_name = "ullm_openai_gateway.served_model"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, LOADER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("served-model validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def validation_summary(manifest: Path) -> dict[str, Any]:
    loader = load_gateway_validator()
    model = loader.load_served_model(manifest)
    artifact = model.product.artifact
    return {
        "schema_version": SUMMARY_SCHEMA,
        "validated": True,
        "manifest_sha256": model.manifest_sha256,
        "model_id": model.public.id,
        "format_id": model.format.format_id,
        "worker": {
            "binary": os.fspath(model.worker.binary),
            "binary_sha256": model.worker.binary_sha256,
            "protocol": model.worker.protocol,
            "device": model.worker.identity.device,
            "execution_profile": model.worker.identity.execution_profile,
        },
        "product": {
            "root": os.fspath(model.product.root),
            "artifact": (
                None
                if artifact is None
                else {
                    "manifest_path": artifact.manifest_path,
                    "manifest_sha256": artifact.manifest_sha256,
                    "content_sha256": artifact.content_sha256,
                }
            ),
            "package": {
                "manifest_path": model.product.package.manifest_path,
                "manifest_sha256": model.product.package.manifest_sha256,
            },
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = validation_summary(args.manifest)
    except Exception:
        # The validator's detailed exception can include deployment paths. Keep
        # systemd preflight output stable and free of manifest content or secrets.
        print("served-model validation failed", file=sys.stderr)
        return 1
    print(
        json.dumps(
            summary,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
