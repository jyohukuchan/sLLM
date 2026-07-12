#!/usr/bin/env python3
"""Materialize a served-model manifest from a deployment profile and live files."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
LOADER_PATH = ROOT / "services/openai-gateway/src/ullm_openai_gateway/served_model.py"
PROFILE_SCHEMA = "ullm.served_model.profile.v1"
_LOADER_MODULE_NAME = "_ullm_served_model_generator_validator"


class GenerationError(RuntimeError):
    """Raised when a profile cannot be bound to immutable local files."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GenerationError(f"failed to read {label}") from error
    if not isinstance(value, dict):
        raise GenerationError(f"{label} must be a JSON object")
    return value


def _receipt_value(receipt: dict[str, Any], path: Any, label: str) -> str:
    if (
        not isinstance(path, list)
        or not path
        or not all(isinstance(item, str) and item for item in path)
    ):
        raise GenerationError(f"{label} must be a nonempty string path")
    value: Any = receipt
    for component in path:
        if not isinstance(value, dict) or component not in value:
            raise GenerationError(f"{label} is absent from the promotion receipt")
        value = value[component]
    if not isinstance(value, str) or not value:
        raise GenerationError(f"{label} in the promotion receipt is invalid")
    return value


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(_LOADER_MODULE_NAME, LOADER_PATH)
    if spec is None or spec.loader is None:
        raise GenerationError("served-model validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_LOADER_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_LOADER_MODULE_NAME, None)
        raise
    return module


def _required_object(parent: dict[str, Any], name: str) -> dict[str, Any]:
    value = parent.get(name)
    if not isinstance(value, dict):
        raise GenerationError(f"profile.{name} must be an object")
    return value


def materialize(profile_path: Path) -> dict[str, Any]:
    profile = _load_json(profile_path, "served-model profile")
    if profile.get("schema_version") != PROFILE_SCHEMA:
        raise GenerationError("served-model profile schema is unsupported")

    tokenizer_profile = _required_object(profile, "tokenizer")
    worker_profile = _required_object(profile, "worker")
    product_profile = _required_object(profile, "product")
    promotion_profile = _required_object(profile, "promotion")

    tokenizer_root = Path(str(tokenizer_profile.get("root", ""))).resolve()
    tokenizer_config = _load_json(
        tokenizer_root / "tokenizer_config.json", "tokenizer config"
    )
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise GenerationError("tokenizer config has no string chat template")
    raw_tokenizer_files = tokenizer_profile.get("files")
    if not isinstance(raw_tokenizer_files, list) or not raw_tokenizer_files:
        raise GenerationError("profile.tokenizer.files must be a nonempty array")
    tokenizer_files: dict[str, str] = {}
    for item in raw_tokenizer_files:
        if not isinstance(item, str) or not item or item in tokenizer_files:
            raise GenerationError("profile.tokenizer.files is invalid")
        tokenizer_files[item] = _sha256_file(tokenizer_root / item)

    worker_binary = Path(str(worker_profile.get("binary", ""))).resolve()
    product_root = Path(str(product_profile.get("root", ""))).resolve()
    package = _required_object(product_profile, "package")
    package_manifest_path = str(package.get("manifest_path", ""))

    receipt_path = Path(str(promotion_profile.get("receipt", ""))).resolve()
    receipt = _load_json(receipt_path, "promotion receipt")
    source_commit = _receipt_value(
        receipt,
        promotion_profile.get("source_commit_from_receipt"),
        "promotion source commit",
    )

    artifact_profile = product_profile.get("artifact")
    artifact: dict[str, str] | None
    if artifact_profile is None:
        artifact = None
    elif isinstance(artifact_profile, dict):
        artifact_manifest_path = str(artifact_profile.get("manifest_path", ""))
        artifact = {
            "manifest_path": artifact_manifest_path,
            "manifest_sha256": _sha256_file(product_root / artifact_manifest_path),
            "content_sha256": _receipt_value(
                receipt,
                artifact_profile.get("content_sha256_from_receipt"),
                "artifact content SHA-256",
            ),
        }
    else:
        raise GenerationError("profile.product.artifact must be an object or null")

    return {
        "schema_version": "ullm.served_model.v1",
        "public": _required_object(profile, "public"),
        "generation": _required_object(profile, "generation"),
        "format": _required_object(profile, "format"),
        "tokenizer": {
            "root": os.fspath(tokenizer_root),
            "transformers_version": tokenizer_profile.get("transformers_version"),
            "class": tokenizer_profile.get("class"),
            "chat_template_sha256": hashlib.sha256(
                chat_template.encode("utf-8")
            ).hexdigest(),
            "files": tokenizer_files,
            "template_options": tokenizer_profile.get("template_options"),
        },
        "worker": {
            "protocol": worker_profile.get("protocol"),
            "binary": os.fspath(worker_binary),
            "binary_sha256": _sha256_file(worker_binary),
            "arguments": worker_profile.get("arguments"),
            "required_environment": worker_profile.get("required_environment"),
            "identity": worker_profile.get("identity"),
        },
        "product": {
            "root": os.fspath(product_root),
            "artifact": artifact,
            "package": {
                "manifest_path": package_manifest_path,
                "manifest_sha256": _sha256_file(product_root / package_manifest_path),
            },
        },
        "promotion": {
            "source_commit": source_commit,
            "receipt": os.fspath(receipt_path),
            "receipt_sha256": _sha256_file(receipt_path),
        },
    }


def generate(profile_path: Path, output_path: Path) -> str:
    document = materialize(profile_path)
    if output_path.is_symlink():
        raise GenerationError("output path must not be a symlink")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(document, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{output_path.name}.", dir=output_path.parent
        )
        temporary = Path(raw_path)
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        model = _load_validator().load_served_model(temporary)
        os.replace(temporary, output_path)
        temporary = None
        directory = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return model.manifest_sha256
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        digest = generate(args.profile, args.output)
    except Exception as error:
        print(f"served-model generation failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": "ullm.served_model.generation.v1",
                "manifest_sha256": digest,
                "output": os.fspath(args.output.resolve()),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
