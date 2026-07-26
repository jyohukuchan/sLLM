#!/usr/bin/env python3
"""Assemble a sealed, text-only Gemma4 E2B BF16 serving product.

The base ``google/gemma-4-E2B`` tokenizer config deliberately has no chat
template.  This tool therefore refuses to overwrite a native template and
requires an explicit, revision-pinned template input.  The resulting tokenizer
overlay records both base-tokenizer and template provenance, while the package
manifest binds the only two model files the resident executor is allowed to
open.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Sequence


PACKAGE_SCHEMA = "ullm.gemma4_e2b_bf16_package.v1"
TOKENIZER_PROVENANCE_SCHEMA = "ullm.gemma4_e2b_tokenizer_overlay.v1"
FORMAT_ID = "BF16_0"
IMPLEMENTATION_ID = "gemma4_e2b_bf16_rdna4_v1"
BASE_UPSTREAM_ID = "google/gemma-4-E2B"
TEMPLATE_UPSTREAM_ID = "google/gemma-4-E2B-it"
MODEL_FILES = ("config.json", "model.safetensors")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
MAX_TEMPLATE_BYTES = 256 * 1024


class AssemblyError(RuntimeError):
    """Raised when an input or output cannot establish the package closure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n").encode(
        "utf-8"
    )


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


def _reject_symlink_components(path: Path, label: str, *, leaf_may_absent: bool) -> None:
    if not path.is_absolute():
        raise AssemblyError(f"{label} must be absolute")
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        if part in {"", ".", ".."}:
            raise AssemblyError(f"{label} is not canonical")
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if leaf_may_absent and index == len(parts) - 1:
                return
            raise AssemblyError(f"{label} has an absent path component") from None
        except OSError as error:
            raise AssemblyError(f"failed to inspect {label}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise AssemblyError(f"{label} traverses a symlink")


def _safe_regular_file(path: Path, label: str) -> Path:
    _reject_symlink_components(path, label, leaf_may_absent=False)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise AssemblyError(f"{label} is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise AssemblyError(f"{label} must be a single-link regular file")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _safe_regular_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AssemblyError(f"failed to decode {label}") from error
    if not isinstance(value, dict):
        raise AssemblyError(f"{label} must be an object")
    return value


def _validate_source_model(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _reject_symlink_components(source, "source model directory", leaf_may_absent=False)
    try:
        source_metadata = source.lstat()
    except OSError as error:
        raise AssemblyError("source model directory is unavailable") from error
    if not stat.S_ISDIR(source_metadata.st_mode):
        raise AssemblyError("source model directory must be a directory")
    for name in (*MODEL_FILES, *TOKENIZER_FILES):
        _safe_regular_file(source / name, f"source {name}")
    config = _read_json(source / "config.json", "source config")
    tokenizer_config = _read_json(source / "tokenizer_config.json", "source tokenizer config")
    text = config.get("text_config")
    if (
        config.get("architectures") != ["Gemma4ForConditionalGeneration"]
        or config.get("model_type") != "gemma4"
        or not isinstance(text, dict)
        or text.get("model_type") != "gemma4_text"
        or text.get("vocab_size") != 262144
        or tokenizer_config.get("tokenizer_class") != "GemmaTokenizer"
    ):
        raise AssemblyError("source model is not the admitted Gemma4 E2B text contract")
    if tokenizer_config.get("chat_template") not in (None, ""):
        raise AssemblyError(
            "source tokenizer already has a chat template; refusing to replace its native contract"
        )
    return config, tokenizer_config


def _template_bytes(path: Path, revision: str) -> tuple[bytes, str]:
    _safe_regular_file(path, "chat template")
    if not _is_lower_hex(revision, 40):
        raise AssemblyError("chat template revision must be a full lowercase Git SHA-1")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise AssemblyError("chat template must be UTF-8") from error
    if not raw or len(raw) > MAX_TEMPLATE_BYTES or text.encode("utf-8") != raw:
        raise AssemblyError("chat template bytes are invalid")
    return raw, text


def _write_bytes(path: Path, data: bytes) -> None:
    with path.open("xb") as destination:
        destination.write(data)
        destination.flush()
        os.fsync(destination.fileno())
    path.chmod(0o444)


def _copy_file(source: Path, destination: Path) -> dict[str, int | str]:
    _safe_regular_file(source, f"source {source.name}")
    with source.open("rb") as input_file, destination.open("xb") as output_file:
        digest = hashlib.sha256()
        size = 0
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
            output_file.write(chunk)
            size += len(chunk)
        output_file.flush()
        os.fsync(output_file.fileno())
    destination.chmod(0o444)
    actual = {"sha256": digest.hexdigest(), "bytes": size}
    if sha256_file(destination) != actual["sha256"] or destination.stat().st_size != size:
        raise AssemblyError(f"copied {source.name} identity differs")
    return actual


def _seal_directory(path: Path) -> None:
    path.chmod(0o555)


def assemble(
    *,
    source_model_dir: Path,
    chat_template: Path,
    chat_template_revision: str,
    destination: Path,
) -> dict[str, Any]:
    source = source_model_dir.absolute()
    _reject_symlink_components(source, "source model directory", leaf_may_absent=False)
    source = source.resolve(strict=True)
    destination = destination.absolute()
    _validate_source_model(source)
    template_path = chat_template.absolute()
    _reject_symlink_components(template_path, "chat template", leaf_may_absent=False)
    template_raw, template_text = _template_bytes(
        template_path.resolve(strict=True), chat_template_revision
    )
    _reject_symlink_components(destination, "destination", leaf_may_absent=True)
    if destination.exists() or destination.is_symlink():
        raise AssemblyError("destination already exists")
    parent = destination.parent
    if not parent.exists() or parent.is_symlink():
        raise AssemblyError("destination parent is unavailable")

    try:
        destination.mkdir(mode=0o755)
        package_dir = destination / "package"
        model_dir = package_dir / "model"
        tokenizer_dir = destination / "tokenizer"
        package_dir.mkdir(mode=0o755)
        model_dir.mkdir(mode=0o755)
        tokenizer_dir.mkdir(mode=0o755)

        model_identities = {
            name: _copy_file(source / name, model_dir / name) for name in MODEL_FILES
        }
        tokenizer_json_identity = _copy_file(
            source / "tokenizer.json", tokenizer_dir / "tokenizer.json"
        )
        source_tokenizer_config = (source / "tokenizer_config.json").read_bytes()
        source_tokenizer_config_identity = {
            "sha256": hashlib.sha256(source_tokenizer_config).hexdigest(),
            "bytes": len(source_tokenizer_config),
        }
        tokenizer_config = json.loads(source_tokenizer_config.decode("utf-8"))
        tokenizer_config["chat_template"] = template_text
        tokenizer_config_bytes = _canonical_json(tokenizer_config)
        _write_bytes(tokenizer_dir / "tokenizer_config.json", tokenizer_config_bytes)
        tokenizer_config_identity = {
            "sha256": hashlib.sha256(tokenizer_config_bytes).hexdigest(),
            "bytes": len(tokenizer_config_bytes),
        }
        _write_bytes(tokenizer_dir / "chat_template.jinja", template_raw)
        template_sha256 = hashlib.sha256(template_raw).hexdigest()
        provenance = {
            "schema_version": TOKENIZER_PROVENANCE_SCHEMA,
            "base": {
                "upstream_id": BASE_UPSTREAM_ID,
                "config_sha256": model_identities["config.json"]["sha256"],
                "tokenizer_json_sha256": tokenizer_json_identity["sha256"],
                "tokenizer_config_sha256": source_tokenizer_config_identity["sha256"],
            },
            "chat_template": {
                "upstream_id": TEMPLATE_UPSTREAM_ID,
                "revision": chat_template_revision,
                "sha256": template_sha256,
            },
        }
        provenance_bytes = _canonical_json(provenance)
        _write_bytes(tokenizer_dir / "provenance.json", provenance_bytes)
        tokenizer_files = {
            "tokenizer.json": tokenizer_json_identity["sha256"],
            "tokenizer_config.json": tokenizer_config_identity["sha256"],
            "chat_template.jinja": template_sha256,
            "provenance.json": hashlib.sha256(provenance_bytes).hexdigest(),
        }
        package_manifest = {
            "schema_version": PACKAGE_SCHEMA,
            "format_id": FORMAT_ID,
            "implementation_id": IMPLEMENTATION_ID,
            "architecture": {
                "architectures": ["Gemma4ForConditionalGeneration"],
                "model_type": "gemma4",
                "text_model_type": "gemma4_text",
                "vocab_size": 262144,
            },
            "model": {"root": "model", "files": model_identities},
        }
        package_manifest_bytes = _canonical_json(package_manifest)
        _write_bytes(package_dir / "manifest.json", package_manifest_bytes)
        for directory in (model_dir, package_dir, tokenizer_dir, destination):
            _seal_directory(directory)
    except BaseException:
        # A partial product is never a valid package and must remain visibly incomplete.  Do not
        # recurse-delete it here: operations can inspect it without a tool silently removing data.
        raise

    return {
        "schema_version": "ullm.gemma4_e2b_serving_package_assembly.v1",
        "product_root": os.fspath(destination),
        "package_manifest": {
            "path": os.fspath(package_dir / "manifest.json"),
            "sha256": hashlib.sha256(package_manifest_bytes).hexdigest(),
        },
        "tokenizer": {
            "root": os.fspath(tokenizer_dir),
            "class": "GemmaTokenizer",
            "chat_template_sha256": template_sha256,
            "files": tokenizer_files,
            "provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", required=True, type=Path)
    parser.add_argument("--chat-template", required=True, type=Path)
    parser.add_argument("--chat-template-revision", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = assemble(
            source_model_dir=args.source_model_dir,
            chat_template=args.chat_template,
            chat_template_revision=args.chat_template_revision,
            destination=args.destination,
        )
    except Exception as error:
        print(f"Gemma4 E2B package assembly failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
